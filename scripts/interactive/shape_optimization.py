"""Interactive shape optimization on a torus.

Problem:

    min_a   J(a) = 1/2 * (u(a) - u_bar)^T M_ref (u(a) - u_bar)
    s.t.    S(a) u(a) = f(a)

where

- a are the spline coefficients defining the logical-to-physical map F_a,
- S(a) is the k=0 Hodge Laplacian / grad-grad stiffness matrix on the
  geometry defined by a,
- f(a) is the source RHS vector
- M_ref is the reference-domain (logical) 0-form mass matrix, used to
  measure || u(a) - u_bar ||_L2 in the logical domain,
- u_bar = u(a*) is the reference solution on the unperturbed torus.

The gradient is computed by the adjoint method:

    dJ/da = - lambda^T (dS/da) u,     with   S(a) lambda = M_ref (u - u_bar).

The scalar lambda^T S(a) u is a pure function of a through
SplineMap -> SequenceGeometry -> SequenceOperators -> apply_stiffness, so
we get its derivative w.r.t. a via jax.grad while keeping u, lambda
detached with jax.lax.stop_gradient.
"""

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence, SequenceGeometry
from mrx.io import project_sampled_field
from mrx.mappings import SplineMap, toroid_map
from mrx.operators import (apply_inverse_shifted_hodge_laplacian,
                           apply_mass_matrix, apply_stiffness,
                           operators_from_coeffs)
from mrx.solvers import backtracking_line_search
from mrx.spline_geometry import (min_jacobian_from_coeffs,
                                 spline_map_jacobian_j_at_quad)
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
seq = DeRhamSequence(ns, ps, 2 * p, types, polar=True,
                     tol=1e-9, maxiter=500)
seq.evaluate_1d()
seq.set_map(lambda x: x)
seq.assemble_reference_mass_matrix()

n_sample = 50
print(
    f"Projecting torus map to spline coefficients on a {n_sample}^3 grid ...")
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
# default betti_numbers=(1,1,0,0): null_0_dbc is already empty

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
#    * the operator bundle via `operators_from_coeffs(seq, coeffs, ...)`,
#    * the RHS via `rhs_from_coeffs` (which contains the mapped Jacobian
#      inherited from the projection operator p0_dbc).
#
#  Only scalar "bilinear" wrappers need to be differentiated, so the CG
#  solves themselves are not inside any jax.grad trace.
#

def _jacobian_j_from_coeffs(coeffs):
    """Just the determinant det(DF) at quadrature -- skips metric + inv33."""
    return spline_map_jacobian_j_at_quad(coeffs, seq.e0_T, seq)


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
    ops, _ = operators_from_coeffs(seq, coeffs, ks=(0,), kinds=("hodge",))
    Su = apply_stiffness(seq, ops, u_vec, 0, dirichlet=True)
    return jnp.dot(lam_vec, Su)


def mass_quadratic(coeffs, r_vec):
    """0.5 * r^T M(a) r as a pure function of `coeffs` (for physical-domain J)."""
    ops, _ = operators_from_coeffs(seq, coeffs, ks=(0,), kinds=("mass",))
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
    return apply_inverse_shifted_hodge_laplacian(
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
    ops, _ = operators_from_coeffs(seq, coeffs, ks=(0,))
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

_min_jac_from_coeffs = jax.jit(
    lambda coeffs: min_jacobian_from_coeffs(coeffs, seq.e0_T, seq)
)


# %% ------------------------------------------------------------------
#  Perturb spline coefficients and run gradient descent
# --------------------------------------------------------------------

key = jax.random.PRNGKey(0)
perturbation_scale = 1e-4 * jnp.linalg.norm(coeffs_ref, 2)

n_axis = 3 * seq.basis_0.nz   # number of polar-axis DOFs (3 per z-slice)
mask = jnp.ones(coeffs_ref.shape[1]).at[:n_axis].set(0.0)   # shape (n_dof,)

perturbation = mask * perturbation_scale * \
    jax.random.normal(key, coeffs_ref.shape)
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


def _J_only(trial_coeffs):
    return float(shape_step_jit(trial_coeffs, u_bar, f_jk)[0])


def _feasible(trial_coeffs):
    return float(_min_jac_from_coeffs(trial_coeffs)) > jac_floor


for it in range(n_iters):
    J, gJ, u = shape_step_jit(coeffs_iter, u_bar, f_jk)
    J = float(J)
    gJm = mask * gJ
    gnorm2 = float(jnp.sum(gJm * gJm))
    gnorm = float(jnp.sqrt(gnorm2))
    err_a = float(jnp.linalg.norm(coeffs_iter - coeffs_ref))
    history.append((J, err_a, gnorm))

    result = backtracking_line_search(
        coeffs_iter,
        -gJm,
        J,
        _J_only,
        step_init=step,
        step_min=step_min,
        step_max=step_max,
        c1=c1,
        shrink=shrink,
        grow=grow,
        max_backtracks=max_ls,
        directional_derivative=-gnorm2,
        feasible=_feasible,
    )
    coeffs_iter = result["x"]
    step = result["step"]

    if it % 100 == 0:
        print(
            f"it {it:3d}  J={J:.4e}  ||grad J||={gnorm:.4e}  "
            f"||a - a*||={err_a:.4e}  step={step:.2e}  ls={result['n_backtracks']}"
        )

    if not result["accepted"]:
        print("  line search failed -- stopping")
        break

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
axes[0].semilogy(hist[:, 0])
axes[0].set_title("objective J(a)")
axes[0].set_xlabel("iteration")
axes[1].semilogy(hist[:, 1])
axes[1].set_title("||a - a*||")
axes[1].set_xlabel("iteration")
axes[2].semilogy(hist[:, 2])
axes[2].set_title("||grad J||")
axes[2].set_xlabel("iteration")
plt.tight_layout()
plt.show()

# %%
