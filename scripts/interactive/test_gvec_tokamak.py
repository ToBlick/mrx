# %%
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import xarray as xr

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import gvec_stellarator_map
from mrx.utils import is_running_in_github_actions

jax.config.update("jax_enable_x64", True)
script_dir = Path(__file__).parent / 'script_outputs'
script_dir.mkdir(parents=True, exist_ok=True)
# %%
repo_root = Path(__file__).parent.parent.parent
data_file = repo_root / "data" / "gvec_tokamak.h5"
gvec_eq = xr.open_dataset(data_file, engine="h5netcdf")
# %%
θ_star = gvec_eq["thetastar"].values    # shape (mρ, mθ), rho x theta
_r = gvec_eq["rho"].values              # shape (mρ,)
_θ = gvec_eq["theta"].values            # shape (mθ,)
X1 = gvec_eq["X1"].values              # shape (mρ, mθ, 1)
X2 = gvec_eq["X2"].values              # shape (mρ, mθ, 1)
# %%
n, p = 8, 3
if is_running_in_github_actions():
    n, p = 4, 2
# Get a deRham sequence to approximate the functions X1(ρ,θ) and X2(ρ,θ)
mapSeq = DeRhamSequence((n, n, 1), (p, p, 0), p+2,
                        ("clamped", "periodic", "constant"),
                        lambda x: x, polar=False, dirichlet=False)


# Set up the interpolation problem:
# ∑ c_i Λ0[i](ρ,θ*(ρ,θ),0)_j ≈ X1(ρ,θ)_j

# Evaluation grid:
r, θ = jnp.meshgrid(_r, _θ, indexing="ij")    # shape (mρ, mθ)
θ_star = jnp.asarray(θ_star)                  # same shape
# Build evaluation points in 3D expected by Λ0[i]: set ζ=0
pts = jnp.stack([r.ravel(), θ_star.ravel() / (2 * jnp.pi),
                jnp.zeros(r.size)], axis=1)  # (mρ mθ, 3)

# Design Matrix:
M = jax.vmap(lambda i: jax.vmap(lambda x: mapSeq.Lambda_0[i](x)[0])(pts))(
    mapSeq.Lambda_0.ns).T  # (mρ mθ, n)
# Target values:
y = jnp.stack([X1.ravel(), X2.ravel()], axis=1)  # (mρ mθ, 2)
# %%
c, residuals, rank, s = jnp.linalg.lstsq(M, y, rcond=None)
# %%
X1_h = DiscreteFunction(c[:, 0], mapSeq.Lambda_0, mapSeq.E0)
X2_h = DiscreteFunction(c[:, 1], mapSeq.Lambda_0, mapSeq.E0)

F = jax.jit(gvec_stellarator_map(X1_h, X2_h, nfp=1))

# %%
# Assemble Sequence with Gvec mapping


def f(x):
    r, θ, ζ = x
    return jnp.sin(2 * jnp.pi * θ) * jnp.sin(jnp.pi * r) * jnp.ones(1)


projection_errs = []
if is_running_in_github_actions():
    ns = jnp.arange(4, 7, 2)
    p = 2
else:
    ns = jnp.arange(4, 19, 2)
    p = 3
for n in ns:
    Seq = DeRhamSequence((n, n, 1), (p, p, 0), p+2,
                         ("clamped", "periodic", "constant"),
                         F, polar=True, dirichlet=True)
    Seq.evaluate_1d()
    Seq.assemble_M0()
    f_dof = jnp.linalg.solve(Seq.M0, Seq.P0(f))
    f_h = DiscreteFunction(f_dof, Seq.Lambda_0, Seq.E0)

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
    projection_errs.append(error)
    print(f"n={n}: projection relative L2 error = {error:.3e}")


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
plt.savefig(script_dir / "projection_error.png")
if not is_running_in_github_actions():
    plt.show()  

# %%
# --------------------------------------------------------------------
# Parameters for the sampling grid in (ρ, θ)
# --------------------------------------------------------------------
mρ_vis, mθ_vis = 80, 180
ρ_vals = jnp.linspace(0.0, 1.0, mρ_vis)
θ_vals = jnp.linspace(0.0, 2 * jnp.pi, mθ_vis)

# Normalize θ for Λ0 evaluation
θ_norm = (θ_vals / (2 * jnp.pi)) % 1.0

# --------------------------------------------------------------------
# Evaluate curves of constant ρ (vary θ)
# --------------------------------------------------------------------


def eval_map(rho, thetas):
    pts = jnp.stack([
        jnp.full_like(thetas, rho),
        (thetas / (2 * jnp.pi)) % 1.0,
        jnp.zeros_like(thetas)
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
        jnp.zeros(mρ_vis)
    ], axis=1)
    R = jax.vmap(X1_h)(pts)
    Z = jax.vmap(X2_h)(pts)
    return np.array(R), np.array(Z)


# --------------------------------------------------------------------
# Plot the deformed polar grid
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

# constant-ρ lines (black)
for r in np.linspace(0, 1, 9, endpoint=True):
    R, Z = eval_map(r, θ_vals)
    ax.plot(R, Z, color="black", lw=0.8)

# constant-θ lines (red)
for θn in np.linspace(0, 1, 12, endpoint=False):
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
plt.savefig(script_dir / "deformed_polar_grid.png")

if not is_running_in_github_actions():
    plt.show()  # %%

# %%
Seq.assemble_all()
eigs = [
    jnp.linalg.eigvalsh(Seq.M0 @ Seq.dd0),
    jnp.linalg.eigvalsh(Seq.M1 @ Seq.dd1),
    jnp.linalg.eigvalsh(Seq.M2 @ Seq.dd2),
    jnp.linalg.eigvalsh(Seq.M3 @ Seq.dd3),
]

# %%
expected_nulls = [False,  False,  True, True]
for i, (vals, should_be_zero) in enumerate(zip(eigs, expected_nulls)):
    # --- all eigenvalues should be nonnegative ---
    min_eig = jnp.min(vals)
    if not is_running_in_github_actions():
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

if not is_running_in_github_actions():
    npt.assert_allclose(curl_grad, 0.0, atol=1e-11,
                        err_msg="curl∘grad ≠ 0")
    npt.assert_allclose(div_curl, 0.0, atol=1e-11,
                        err_msg="div∘curl ≠ 0")

# %%
