"""Interactive shape optimization on a torus.

Problem:

    min_a   J(a) = 1/2 * (u(a) - u_bar)^T M_ref (u(a) - u_bar)
    s.t.    S(a) u(a) = f

where

- a are the spline coefficients defining the logical-to-physical map F_a,
- S(a) is the k=0 Hodge Laplacian / grad-grad stiffness matrix on the
  geometry defined by a,
- f is the source RHS vector, assembled once on the reference geometry a*
  and then held fixed (so df/da = 0 in coefficient space),
- M_ref is the reference-domain (logical) 0-form mass matrix, used to
  measure || u(a) - u_bar ||_L2 in the logical domain,
- u_bar = u(a*) is the reference solution on the unperturbed torus.

The gradient is computed by the adjoint method:

    dJ/da = - lambda^T (dS/da) u,     with   S(a) lambda = M_ref (u - u_bar).

The scalar lambda^T S(a) u is a pure function of a through
SplineMap -> SequenceGeometry -> SequenceOperators -> apply_stiffness, so
we get its derivative w.r.t. a via jax.grad while keeping u, lambda
detached with jax.lax.stop_gradient.

Run cell-by-cell in VS Code / Jupyter to inspect the optimization.
"""

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence, SequenceGeometry
from mrx.io import project_sampled_field
from mrx.mappings import SplineMap, toroid_map
from mrx.operators import (
    apply_inverse_shifted_stiffness,
    apply_mass_matrix,
    apply_stiffness,
    assemble_derivative_operators,
    assemble_hodge_operators,
    assemble_mass_operators,
)
from mrx.utils import integrate_against

jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")

# Toggle the metric used for || u(a) - u_bar ||.
#  - False: compare in the logical (reference) domain via M_ref.
#    Minimum at a = a* is J = 0 and the gradient is a single adjoint term.
#  - True:  compare in the physical domain via the mapped mass M(a).
#    Adds a metric-derivative term to the gradient; the landscape now
#    rewards shrinking the domain, so results are only meaningful for
#    small perturbations or with extra regularization.
compare_in_physical_domain = False


# %% ------------------------------------------------------------------
#  Analytic source and reference solution (defined in logical coords)
# --------------------------------------------------------------------

def exact_u(x):
    r, t, z = x
    return 1 / 4 * (r ** 2 - r ** 4) * jnp.cos(2 * jnp.pi * z) * jnp.ones(1)


def source_f(x):
    r, t, z = x
    a = 1 / 3
    R = 1 + a * r * jnp.cos(2 * jnp.pi * t)
    return (
        jnp.cos(2 * jnp.pi * z)
        * (
            -1 / a ** 2 * (1 - 4 * r ** 2)
            - 1 / (a * R) * (r / 2 - r ** 3) * jnp.cos(2 * jnp.pi * t)
            + 1 / 4 * (r ** 2 - r ** 4) / R ** 2
        )
        * jnp.ones(1)
    )


# %% ------------------------------------------------------------------
#  Build the sequence and project the reference torus map to splines
# --------------------------------------------------------------------

p = 3
n = 10
ns = (n, n, n)
ps = (p, p, p)
aa = 1 / 3
torus_map = toroid_map(epsilon=aa)

print("Building sequence (reference/identity map) ...")
seq = DeRhamSequence(ns, ps, 2 * p, types, lambda x: x, polar=True,
                     tol=1e-9, maxiter=500)
seq.evaluate_1d()
seq.assemble_reference_mass_matrix()

n_sample = 50
print(f"Projecting torus map to spline coefficients on a {n_sample}^3 grid ...")
r_lin = jnp.linspace(0, 1, n_sample)
chi_lin = jnp.linspace(0, 1, n_sample)
zeta_lin = jnp.linspace(0, 1, n_sample)
ri, chii, zetai = jnp.meshgrid(r_lin, chi_lin, zeta_lin, indexing="ij")
grid_pts = jnp.stack([ri.ravel(), chii.ravel(), zetai.ravel()], axis=1)
map_grid = jax.vmap(torus_map)(grid_pts)

coeffs_ref = jnp.stack(
    [
        project_sampled_field(
            (r_lin, chi_lin, zeta_lin), map_grid[:, i], seq,
            k=0, dirichlet=False, reference_domain=True,
        )
        for i in range(3)
    ],
    axis=0,
)
print("coeffs_ref shape:", coeffs_ref.shape)


# %% ------------------------------------------------------------------
#  Reference solve on the unperturbed torus geometry
# --------------------------------------------------------------------

print("Assembling reference operators and solving for u_bar ...")
seq.set_spline_map(coeffs_ref)
seq.assemble_mass_matrix(0)
seq.assemble_hodge_laplacian(0)
seq.null_0_dbc = []

# Precompute the source at logical quadrature points once (geometry-independent).
f_jk = jax.lax.map(source_f, seq.quad.x, batch_size=20_000)

rhs_ref = seq.p0_dbc(source_f)
u_bar = seq.apply_inverse_hodge_laplacian(rhs_ref, k=0)


# %% ------------------------------------------------------------------
#  Purely functional operator / RHS builders for the adjoint gradient
# --------------------------------------------------------------------
#
#  `shape_step` reconstructs everything that depends on `coeffs`:
#
#    * the operator bundle via `_operators_from_coeffs`,
#    * the RHS via `rhs_from_coeffs` (which contains the mapped Jacobian
#      inherited from the projection operator p0_dbc).
#
#  Only scalar "bilinear" wrappers need to be differentiated, so the CG
#  solves themselves are not inside any jax.grad trace.
#

def _operators_from_coeffs(coeffs):
    geometry = seq.geometry_from_spline_map(coeffs)
    ops = assemble_mass_operators(seq, geometry, ks=(0,))
    ops = assemble_derivative_operators(seq, geometry, operators=ops, ks=(0,))
    ops = assemble_hodge_operators(seq, geometry, operators=ops, ks=(0,))
    return ops, geometry


def _geometry_from_coeffs(coeffs):
    """Full geometry (metric + inv + jacobian). Used only where all are needed."""
    return seq.geometry_from_spline_map(coeffs)


def _jacobian_j_from_coeffs(coeffs):
    """Just the determinant det(DF) at quadrature -- skips metric + inv33."""
    from mrx.spline_geometry import spline_map_jacobian_j_at_quad
    return spline_map_jacobian_j_at_quad(coeffs, seq.e0_T, seq)


def _hodge0_ops_from_coeffs(coeffs):
    """Assemble only the k=0 Hodge block (what `apply_stiffness` at k=0 needs)."""
    geometry = _geometry_from_coeffs(coeffs)
    ops = assemble_hodge_operators(seq, geometry, ks=(0,))
    return ops


def _mass0_ops_from_coeffs(coeffs):
    """Assemble only the k=0 mass block (what `apply_mass_matrix` at k=0 needs)."""
    geometry = _geometry_from_coeffs(coeffs)
    ops = assemble_mass_operators(seq, geometry, ks=(0,))
    return ops


def _rhs_from_jac(jac_j, f_jk):
    """Project f at quadrature against 0-forms given only the Jacobian det array."""
    w_jk = f_jk * (seq.quad.w * jac_j)[:, None]
    comp_info, comp_shapes = seq._form_comp_info(0)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    integrated = integrate_against(w_jk, comp_info, comp_shapes, quad_shape)
    return seq.e0_dbc @ integrated


def rhs_from_coeffs(coeffs, f_jk):
    """Project `source_f` (evaluated at quadrature) against 0-form basis on geom(coeffs)."""
    return _rhs_from_jac(_jacobian_j_from_coeffs(coeffs), f_jk)


def stiffness_bilinear(coeffs, u_vec, lam_vec):
    # Only the k=0 Hodge block is needed; skip mass + derivative assembly.
    ops = _hodge0_ops_from_coeffs(coeffs)
    Su = apply_stiffness(seq, ops, u_vec, 0, dirichlet=True)
    return jnp.dot(lam_vec, Su)


def mass_quadratic(coeffs, r_vec):
    """0.5 * r^T M(a) r as a pure function of `coeffs` (for physical-domain J)."""
    ops = _mass0_ops_from_coeffs(coeffs)
    Mr = apply_mass_matrix(seq, ops, r_vec, 0, dirichlet=True)
    return 0.5 * jnp.dot(r_vec, Mr)


def rhs_bilinear(coeffs, lam_vec, f_jk):
    """lam^T f(a) -- picks up the geometry dependence of the projected RHS.

    Only the Jacobian determinant enters, so we skip metric / metric_inv.
    """
    jac_j = _jacobian_j_from_coeffs(coeffs)
    return jnp.dot(lam_vec, _rhs_from_jac(jac_j, f_jk))


# %% ------------------------------------------------------------------
#  Objective + adjoint gradient   -- pure, JIT-able
# --------------------------------------------------------------------
#
#  `shape_step` below is a pure function of its JAX inputs (coeffs,
#  u_bar, f_jk).  `seq` is captured via closure and treated as a static
#  Python object: it contributes extraction operators, quadrature,
#  reference-mass matrices, and solver tolerances, but no dynamic state
#  that would invalidate tracing.
#

def apply_M_ref(v):
    return seq._apply_reference_mass_matrix(v, dirichlet=True)


def _solve_poisson(ops, rhs):
    """Solve S(ops) u = rhs for k=0 with eps=0 using the pure operator bundle."""
    return apply_inverse_shifted_stiffness(
        seq, ops, rhs, 0, 0.0, dirichlet=True,
        tol=seq.tol, maxiter=seq.maxiter,
    )


_grad_stiffness_wrt_coeffs = jax.grad(stiffness_bilinear, argnums=0)
_grad_mass_quadratic_wrt_coeffs = jax.grad(mass_quadratic, argnums=0)
_grad_rhs_wrt_coeffs = jax.grad(rhs_bilinear, argnums=0)


def shape_step(coeffs, u_bar, f_jk):
    """Return (J, grad_J, u) for the current coefficients.

    Reconstructs the operator bundle, the projected RHS, and the
    forward / adjoint solves from `coeffs` each call. The adjoint
    gradient is

        dJ/da = -d/da (lam^T S(a) u) + d/da (lam^T f(a))
                [ + 1/2 d/da (r^T M(a) r) ]    (physical-domain only)

    with u, lam, r frozen via stop_gradient.
    """
    ops, _ = _operators_from_coeffs(coeffs)
    rhs = rhs_from_coeffs(coeffs, f_jk)
    u = _solve_poisson(ops, rhs)
    r = u - u_bar

    if compare_in_physical_domain:
        Mr = apply_mass_matrix(seq, ops, r, 0, dirichlet=True)
    else:
        Mr = apply_M_ref(r)

    lam = _solve_poisson(ops, Mr)

    u_sg = jax.lax.stop_gradient(u)
    lam_sg = jax.lax.stop_gradient(lam)
    r_sg = jax.lax.stop_gradient(r)

    grad_J = (
        -_grad_stiffness_wrt_coeffs(coeffs, u_sg, lam_sg)
        + _grad_rhs_wrt_coeffs(coeffs, lam_sg, f_jk)
    )
    if compare_in_physical_domain:
        grad_J = grad_J + _grad_mass_quadratic_wrt_coeffs(coeffs, r_sg)

    J = 0.5 * jnp.dot(r, Mr)
    return J, grad_J, u


shape_step_jit = jax.jit(shape_step)

@jax.jit
def _min_jac_from_coeffs(coeffs):
    """Cheap Jacobian-positivity check: min_q J(F_a)(q) over quadrature points.

    Only evaluates det(DF) -- no metric, no operator assembly.
    """
    return jnp.min(_jacobian_j_from_coeffs(coeffs))


# %% ------------------------------------------------------------------
#  Perturb spline coefficients and run gradient descent
# --------------------------------------------------------------------

key = jax.random.PRNGKey(0)
perturbation_scale = 5e-4 * jnp.linalg.norm(coeffs_ref, 2)

n_axis = 3 * seq.basis_0.nz   # number of polar-axis DOFs (3 per z-slice)
mask = jnp.ones(coeffs_ref.shape[1]).at[:n_axis].set(0.0)   # shape (n_dof,)

perturbation = mask * perturbation_scale * jax.random.normal(key, coeffs_ref.shape)
coeffs = coeffs_ref + perturbation
print(f"Initial ||a - a*|| = {float(jnp.linalg.norm(perturbation)):.4e}")
# Check that the Jacobian is positive for the initial perturbed geometry.
print("min J(F):", float(_min_jac_from_coeffs(coeffs)))

# %%
# Backtracking line search with Jacobian-positivity safeguard.
#
#   - reject the step if the proposed geometry has min J(F) <= jac_floor
#     (i.e. mesh has folded or nearly folded),
#   - else require a sufficient-decrease condition
#         J(new) <= J - c1 * step * ||mask * gJ||^2
#   - shrink `step` by `shrink` and retry; grow it by `grow` on success.
n_iters = 1000
step = 1.0            # initial trial step (grows/shrinks adaptively)
step_max = 1e6
step_min = 1e-9
c1 = 1e-4             # Armijo constant
shrink = 0.5
grow = 2.0            # next step = step * grow on successful step, else step * shrink
max_ls = 40           # max backtracks per iteration
jac_floor = 1e-9      # reject geometries with min jacobian below this

history = []
coeffs_iter = coeffs

for it in range(n_iters):
    J, gJ, u = shape_step_jit(coeffs_iter, u_bar, f_jk)
    J = float(J)
    gJm = mask * gJ
    gnorm2 = float(jnp.sum(gJm * gJm))
    gnorm = float(jnp.sqrt(gnorm2))
    err_a = float(jnp.linalg.norm(coeffs_iter - coeffs_ref))
    history.append((J, err_a, gnorm))

    # --- line search ------------------------------------------------
    accepted = False
    for ls in range(max_ls):
        trial = coeffs_iter - step * gJm
        min_jac = float(_min_jac_from_coeffs(trial))
        if min_jac > jac_floor:
            J_trial, _, _ = shape_step_jit(trial, u_bar, f_jk)
            J_trial = float(J_trial)
            if jnp.isfinite(J_trial) and J_trial <= J - c1 * step * gnorm2:
                coeffs_iter = trial
                accepted = True
                break
        step = max(step * shrink, step_min)

    if (it) % 100 == 0:
        print(
            f"it {it:3d}  J={J:.4e}  ||grad J||={gnorm:.4e}  "
            f"||a - a*||={err_a:.4e}  step={step:.2e}  ls={ls+1}"
        )

    if not accepted:
        print("  line search failed -- stopping")
        break
    # try a bigger step next iteration
    step = min(step * grow, step_max)

# final residual
J_final, _, _ = shape_step_jit(coeffs_iter, u_bar, f_jk)
J_final = float(J_final)
err_final = float(jnp.linalg.norm(coeffs_iter - coeffs_ref))
print(f"final  J={J_final:.4e}  ||a - a*||={err_final:.4e}")


# %% ------------------------------------------------------------------
#  Plot convergence
# --------------------------------------------------------------------

hist = np.array(history)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].semilogy(hist[:, 0], marker="o")
axes[0].set_title("objective J(a)")
axes[0].set_xlabel("iteration")
axes[1].semilogy(hist[:, 1], marker="o")
axes[1].set_title("||a - a*||")
axes[1].set_xlabel("iteration")
axes[2].semilogy(hist[:, 2], marker="o")
axes[2].set_title("||grad J||")
axes[2].set_xlabel("iteration")
plt.tight_layout()
plt.show()

# %%
