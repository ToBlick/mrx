# %%
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import xarray as xr

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

n, p, nfp = 8, 3, 3
gvec_eq = xr.open_dataset("data/gvec_stellarator.h5", engine="h5netcdf")
# θ_star = gvec_eq["thetastar"].values    # shape (mρ, mθ, mζ), rho x theta
_ρ = gvec_eq["rho"].values              # shape (mρ,)
_θ = gvec_eq["theta"].values            # shape (mθ,)
_ζ = gvec_eq["zeta"].values             # shape (mζ,)
X1 = gvec_eq["X1"].values               # shape (mρ, mθ, mζ)
X2 = gvec_eq["X2"].values               # shape (mρ, mθ, mζ)
# %%
# Get a deRham sequence to approximate the functions x1(ρ,θ,ζ), x2(ρ,θ,ζ) and x3(ρ,θ,ζ)
mapSeq = DeRhamSequence((n, n, n), (p, p, p), p+2,
                        ("clamped", "periodic", "periodic"),
                        lambda x: x, polar=False, dirichlet=False)

# Set up the interpolation problem:
# ∑ c_ki Λ0[i](ρ,θ*(ρ,θ,ζ),ζ)_j ≈ Xk(ρ,θ,ζ)_j ∀j
# evaluation grid, shape (mρ, mθ, mζ)
ρ, θ, ζ = jnp.meshgrid(_ρ, _θ, _ζ, indexing="ij")
# θ_star = jnp.asarray(θ_star)
pts = jnp.stack([ρ.ravel(),
                 θ.ravel() / (2 * jnp.pi),
                 ζ.ravel() / (2 * jnp.pi) * nfp], axis=1)  # x_hat_js, shape (mρ mθ mζ, 3)

M = jax.vmap(lambda i: jax.vmap(lambda x: mapSeq.Λ0[i](x)[0])(pts))(
    mapSeq.Λ0.ns).T  # Λ0[i](x_hat_j)
y = jnp.stack([X1.ravel(), X2.ravel()], axis=1)  # X_α(x'_j)
c, residuals, rank, s = jnp.linalg.lstsq(M, y, rcond=None)
# %%
X1_h = DiscreteFunction(c[:, 0], mapSeq.Λ0, mapSeq.E0)
X2_h = DiscreteFunction(c[:, 1], mapSeq.Λ0, mapSeq.E0)


@jax.jit
def Phi(x):
    r, θ, ζ = x
    return jnp.array([X1_h(x)[0] * jnp.cos(2 * jnp.pi * ζ / nfp),
                      -X1_h(x)[0] * jnp.sin(2 * jnp.pi * ζ / nfp),
                      X2_h(x)[0]])


# %%
# Assemble Sequence with Gvec mapping
Seq = DeRhamSequence((6, 8, 8), (3, 3, 3), 4,
                     ("clamped", "periodic", "periodic"),
                     Phi, polar=True, dirichlet=True)
Seq.evaluate_1d()
Seq.assemble_all()


# %%
# Set up the B-interpolation problem:
# ∑ c_i Phi*Λ2[i](x_j) ≈ B(x'_j) ∀j
def Λ2_phys(i, x):
    """Evaluate the physical 2-form basis function Phi*Λ2[i] at x."""
    # Pullback of basis function
    DPhix = jax.jacfwd(Phi)(x)  # Jacobian of Phi at x
    J = jnp.linalg.det(DPhix)
    return DPhix @ Seq.Λ2[i](x) / J


def eval_basis_block(i):
    # Evaluate Λ2_phys(i, x) for all points (vectorized over x)
    return jax.vmap(lambda x: Λ2_phys(i, x))(pts[valid_pts])  # (n_valid, 3)


def body_fun(_, i):
    return None, eval_basis_block(i)


valid_pts = (pts[:, 0] > 1e-3) & (pts[:, 0] < 1 - 1e-3)
pts_B = pts[valid_pts]  # avoid singularity on axis and eval. on bdy

# TODO: No double vmaps
# evaluate all basis functions at all interp. points
# Stream through basis functions and collect the results into a scanned array
_, M = jax.lax.scan(body_fun, None, Seq.Λ2.ns)
M = jnp.einsum('il,ljk->ijk', Seq.E2, M)  # Λ2[i](x_hat_j)_k
y = gvec_eq.B.values.reshape(-1, 3)[valid_pts]  # B(x'_j)_k
A = M.reshape(M.shape[0], -1).T
b = y.ravel()
# Solve least squares
B_dof, residuals, rank, s = jnp.linalg.lstsq(A, b, rcond=None)
residuals
# %%
B_h = jax.jit(Pushforward(DiscreteFunction(B_dof, Seq.Λ2, Seq.E2), Seq.F, 2))

# %%
_zeta_plt = jnp.linspace(0, 1, 100, endpoint=False)
B_on_axis = jax.vmap(B_h)(jnp.stack([jnp.ones_like(_zeta_plt) * 5e-3,
                                    jnp.zeros_like(_zeta_plt),
                                    _zeta_plt], axis=1))
plt.plot(_zeta_plt, jnp.linalg.norm(B_on_axis, axis=-1))
# %%
# Are we approx. div-free?
div_B_dof = Seq.strong_div @ B_dof
(div_B_dof @ Seq.M3 @ div_B_dof)**0.5 / (B_dof @ Seq.M2 @ B_dof)**0.5
# %%
eigs = [
    jnp.linalg.eigvalsh(Seq.M0 @ Seq.dd0),
    jnp.linalg.eigvalsh(Seq.M1 @ Seq.dd1),
    jnp.linalg.eigvalsh(Seq.M2 @ Seq.dd2),
    jnp.linalg.eigvalsh(Seq.M3 @ Seq.dd3),
]

expected_nulls = [False,  False,  True, True]
for i, (vals, should_be_zero) in enumerate(zip(eigs, expected_nulls)):
    # --- all eigenvalues should be nonnegative ---
    min_eig = jnp.min(vals)
    assert min_eig > -1e-10, (
        f"dd{i} has negative eigenvalue {min_eig}"
    )

    # --- check smallest eigenvalue matches expected nullspace pattern ---
    λ0 = float(vals[0])
    if should_be_zero:
        assert abs(λ0) < 1e-10, (
            f"dd{i} should have zero eigenvalue (got {λ0})"
        )
    else:
        assert abs(λ0) > 1e-6, (
            f"dd{i} should NOT have zero eigenvalue (got {λ0})"
        )

# Check exactness identities
curl_grad = jnp.max(jnp.abs(Seq.strong_curl @ Seq.strong_grad))
div_curl = jnp.max(jnp.abs(Seq.strong_div @ Seq.strong_curl))
npt.assert_allclose(curl_grad, 0.0, atol=1e-12,
                    err_msg="curl∘grad ≠ 0")
npt.assert_allclose(div_curl, 0.0, atol=1e-12,
                    err_msg="div∘curl ≠ 0")


# %%

def f(x):
    r, θ, ζ = x
    return jnp.sin(2 * jnp.pi * θ) * jnp.sin(2 * jnp.pi * ζ) * jnp.sin(jnp.pi * r) * jnp.ones(1)


@partial(jax.jit, static_argnames=["n"])
def get_err(n):
    Seq = DeRhamSequence((n, n, n), (p, p, p), p+2,
                         ("clamped", "periodic", "periodic"),
                         Phi, polar=True, dirichlet=True)
    Seq.evaluate_1d()
    Seq.assemble_M0()
    f_dof = jnp.linalg.solve(Seq.M0, Seq.P0(f))
    f_h = DiscreteFunction(f_dof, Seq.Λ0, Seq.E0)

    # --- error evaluation ---
    def diff_at_x(x):
        return f(x) - f_h(x)

    def body_fun(carry, x):
        return None, diff_at_x(x)
    _, df = jax.lax.scan(body_fun, None, Seq.Q.x)
    L2_dp = jnp.einsum('ik,ik,i,i->', df, df, Seq.J_j, Seq.Q.w)**0.5
    L2_p = jnp.einsum('ik,ik,i,i->',
                      jax.vmap(f)(Seq.Q.x),
                      jax.vmap(f)(Seq.Q.x),
                      Seq.J_j, Seq.Q.w)**0.5

    error = L2_dp / L2_p
    return error, jnp.min(Seq.J_j), jnp.max(Seq.J_j)


# %%
projection_errs = []
ns = range(4, 10, 1)
for n in ns:
    error, J_min, J_max = get_err(n)
    assert J_min > 0, f"Jacobian has non-positive values for n={n}"
    assert J_max / J_min < 1e9, f"Jacobian severely ill-conditioned for n={n}"
    projection_errs.append(error)
    print(f"n={n}: projection relative L2 error = {projection_errs[-1]:.3e}")

# Check that the error decreases with increasing n at expected rate or faster
rates = -jnp.array([jnp.log(projection_errs[i] / projection_errs[i+1]) /
                   jnp.log(ns[i] / ns[i+1]) for i in range(len(ns)-1)])
assert jnp.mean(rates) >= p + 1, f"Convergence rates too low: {rates}"

# %%
plt.figure(figsize=(6, 4))
plt.loglog(ns, projection_errs, marker='o')
plt.loglog(ns, [projection_errs[-1] * 0.9 * (n/ns[-1])**(-(p + 1)) for n in ns],
           linestyle='--', label=f"O(h^{p + 1})")
plt.xlabel("Resolution n")
plt.ylabel("Relative L2 Projection Error")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.show()

# %%


# %%
# --------------------------------------------------------------------
# Parameters for the sampling grid in (ρ, θ)
# --------------------------------------------------------------------
mρ_vis, mθ_vis = 80, 180
ρ_vals = jnp.linspace(0.0, 1.0, mρ_vis)
θ_vals = jnp.linspace(0.0, 2 * jnp.pi, mθ_vis)
ζ_val = 0.5
# Normalize θ for Λ0 evaluation
θ_norm = (θ_vals / (2 * jnp.pi)) % 1.0

# --------------------------------------------------------------------
# Evaluate curves of constant ρ (vary θ)
# --------------------------------------------------------------------


def eval_map(rho, thetas):
    pts = jnp.stack([
        jnp.full_like(thetas, rho),
        (thetas / (2 * jnp.pi)) % 1.0,
        jnp.ones_like(thetas) * ζ_val
    ], axis=1)
    R = jax.vmap(X1_h)(pts)
    Z = jax.vmap(X2_h)(pts)
    return np.array(R), np.array(Z)

# --------------------------------------------------------------------
# Evaluate curves of constant θ (vary ρ)
# --------------------------------------------------------------------


def eval_map_theta(theta_norm):
    pts = jnp.stack([
        jnp.linspace(0, 1, mρ_vis),
        jnp.full(mρ_vis, theta_norm),
        jnp.ones(mρ_vis) * ζ_val
    ], axis=1)
    R = jax.vmap(X1_h)(pts)
    Z = jax.vmap(X2_h)(pts)
    return np.array(R), np.array(Z)


# --------------------------------------------------------------------
# Plot the deformed polar grid
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

# constant-ρ lines (black)
for ρ in np.linspace(0, 1, 16, endpoint=True):
    R, Z = eval_map(ρ, θ_vals)
    ax.plot(R, Z, color="black", lw=0.8)

# constant-θ lines (red)
for θn in np.linspace(0, 1, 16, endpoint=False):
    R, Z = eval_map_theta(θn)
    ax.plot(R, Z, color="red", lw=0.8)

# formatting
R_axis, Z_axis = eval_map(0.0, jnp.array([0.0]))  # magnetic axis candidate
R_axis, Z_axis = R_axis.item(), Z_axis.item()

ax.set(
    xlabel="$R = X^1$",
    ylabel="$Z = X^2$",
    aspect="equal",
    title=f"Map $(\\rho, \\vartheta^*) \\mapsto (X^1, X^2)$\naxis at (R,Z)=({R_axis:.5f},{Z_axis:.5f})"
)
ax.legend(
    handles=[
        plt.Line2D([0], [0], color="black", label="$\\rho=$ const."),
        plt.Line2D([0], [0], color="red", label="$\\vartheta^*=$ const.")
    ],
    loc="upper right"
)
plt.tight_layout()
plt.show()  # %%

# %%
