"""Interactive debug for the torus Poisson convergence stagnation.

Background
----------
``scripts/config_scripts/test_torus_poisson_sparse.py`` shows expected
algebraic L2 convergence for ``p=1`` but a fixed error floor near ``2e-4``
for ``p >= 2`` that does not improve as ``n`` increases. This harness
isolates plausible causes one cell at a time using the smallest problem
that already exhibits the stagnation (``p=2``, modest ``n``).

Note: ``seq.p0_dbc(f)`` does *not* invert a mass matrix — it just
returns the assembled load vector ``b_i = ∫ f φ_i J dξ`` via
``integrate_against``. The production K0 solve therefore solves
``K0 u_hat = b`` directly, where ``K0`` is the discrete Hodge Laplacian
and ``b`` is that load vector. The only ``q``-dependent ingredients are
the assembled mass / stiffness blocks and the load vector itself.

Headline hypothesis (user-confirmed first probe): the production
``q = 2*p`` is too low for the rational toroidal integrands ``R``,
``1/R``, ``1/R²`` and pins a fixed quadrature-error floor at
``~2e-4``. Cell ``[D]`` is the decisive test.

Supporting cells:
- ``[B]`` measures the best the *discrete space* can represent the
  exact solution by computing ``u_proj = M0⁻¹ p0_dbc(u)`` and the L2
  error ``||u − u_proj||``. This is the true discretization floor; the
  K0 solve cannot beat it.
- ``[C]`` tightens the outer CG tolerance from ``1e-9`` to ``1e-13``
  to rule out the outer Krylov residual as the binding constraint.
- ``[D]`` raises ``q`` from ``2p`` to ``2p+4`` and re-runs the full
  production-style solve.
- ``[E]`` checks discrete consistency of the manufactured pair via
  ``||K0 u_proj − b|| / ||b||`` with the same ``q = 2p`` ingredients.

Run interactively (cell-by-cell) in VS Code; this is *not* a CLI script.
"""

# %% Imports and JAX setup -----------------------------------------------------
from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import assemble_tensor_mass_preconditioner
from mrx.utils import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# Match the production script's batching semantics (serial outer map).
mrx.MAP_BATCH_SIZE_OUTER = None

π = jnp.pi


# %% Problem definition --------------------------------------------------------
def u(x: jnp.ndarray) -> jnp.ndarray:
    """Exact solution: u(r,χ,z) = 1/4 (r² - r⁴) cos(2πz)."""
    r, χ, z = x
    return 1 / 4 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def make_f(a: float):
    """Production source term used by ``test_torus_poisson_sparse.py``."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, χ, z = x
        R = 1 + a * r * jnp.cos(2 * π * χ)
        return (
            jnp.cos(2 * π * z)
            * (
                -1 / a**2 * (1 - 4 * r**2)
                - 1 / (a * R) * (r / 2 - r**3) * jnp.cos(2 * π * χ)
                + 1 / 4 * (r**2 - r**4) / R**2
            )
            * jnp.ones(1)
        )

    return f


def exact_u_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    u_r = 0.25 * (seq.quad.x_x**2 - seq.quad.x_x**4)
    u_z = jnp.cos(2 * π * seq.quad.x_z)
    values = jnp.ones((seq.quad.ny, 1, 1)) * u_r[None, :, None] * u_z[None, None, :]
    return values.reshape(-1, 1)


def l2_relative_error(
    seq: DeRhamSequence, u_h_quad: jnp.ndarray, u_exact_quad: jnp.ndarray
) -> float:
    df = u_exact_quad - u_h_quad
    num = jnp.einsum("ik,ik,i,i->", df, df, seq.jacobian_j, seq.quad.w)
    den = jnp.einsum(
        "ik,ik,i,i->", u_exact_quad, u_exact_quad, seq.jacobian_j, seq.quad.w
    )
    return float((num / den) ** 0.5)


# %% Pick the smallest config that already exhibits the floor -----------------
# Baseline: p=2, n=8 already shows the stagnation in the production run.
P = 2
N = 8
EPSILON = 1 / 3
TYPES = ("clamped", "periodic", "periodic")

ns = (N, 2 * N, N)
ps = (P, P, P)
q = 2 * P
cg_tol = 1e-9
cg_maxiter = 100_000

t0 = time.perf_counter()
seq = DeRhamSequence(
    ns, ps, q, TYPES, polar=True, tol=cg_tol, maxiter=cg_maxiter
)
seq.set_map(toroid_map(epsilon=EPSILON))
seq.evaluate_1d()
seq.assemble_mass_matrix(0)
seq.assemble_mass_matrix(1)
seq.set_operators(
    assemble_tensor_mass_preconditioner(seq, seq.get_operators(), ks=(0,), rank=1)
)
seq.assemble_hodge_laplacian(0)
print(f"assemble: {time.perf_counter() - t0:.2f}s, q={q}, ns={ns}, p={P}")


# %% Cell A: reproduce production error at this small (p,n) -------------------
f_callable = make_f(EPSILON)

rhs = seq.p0_dbc(f_callable)
u_hat = seq.apply_inverse_hodge_laplacian(rhs, 0, dirichlet=True)

quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
comp_info_0, comp_shapes_0 = seq._form_comp_info(0)
u_h_quad = evaluate_at_xq(
    seq.e0_dbc_T @ u_hat, comp_info_0, comp_shapes_0, quad_shape, 1
)
u_exact_quad = exact_u_at_quad(seq)

err = l2_relative_error(seq, u_h_quad, u_exact_quad)
print(f"[A] production-style relative L2 error: {err:.6e}")


# %% Cell B: best discrete representation of u_exact (L2 projection) -------
# Solve M0 u_proj_hat = p0_dbc(u_exact). This is the true L2 projection
# (NOT what p0_dbc on its own does — that returns only the load vector).
# The K0 solve cannot beat this discretization floor.
u_load = seq.p0_dbc(u)
u_proj_hat = seq.apply_inverse_mass_matrix(u_load, 0, dirichlet=True)
u_proj_quad = evaluate_at_xq(
    seq.e0_dbc_T @ u_proj_hat, comp_info_0, comp_shapes_0, quad_shape, 1
)
err_proj = l2_relative_error(seq, u_proj_quad, u_exact_quad)
print(f"[B] discretization floor (||u - Π_L2 u||): {err_proj:.6e}")


# %% Cell C: tighter outer CG tolerance on the K0 solve -----------------------
# Same assembled K0 / rhs / preconditioner; only the outer CG tolerance
# changes. If the error doesn't move, CG residual is not the floor.
seq_tight = DeRhamSequence(
    ns, ps, q, TYPES, polar=True, tol=1e-13, maxiter=200_000
)
seq_tight.set_map(toroid_map(epsilon=EPSILON))
seq_tight.evaluate_1d()
seq_tight.assemble_mass_matrix(0)
seq_tight.assemble_mass_matrix(1)
seq_tight.set_operators(
    assemble_tensor_mass_preconditioner(
        seq_tight, seq_tight.get_operators(), ks=(0,), rank=1
    )
)
seq_tight.assemble_hodge_laplacian(0)

rhs_tight = seq_tight.p0_dbc(f_callable)
u_hat_tight = seq_tight.apply_inverse_hodge_laplacian(
    rhs_tight, 0, dirichlet=True
)
u_h_quad_tight = evaluate_at_xq(
    seq_tight.e0_dbc_T @ u_hat_tight,
    *seq_tight._form_comp_info(0),
    (seq_tight.quad.ny, seq_tight.quad.nx, seq_tight.quad.nz),
    1,
)
err_tight = l2_relative_error(seq_tight, u_h_quad_tight, exact_u_at_quad(seq_tight))
print(f"[C] outer CG tol=1e-13 relative L2 error: {err_tight:.6e}")


# %% Cell D: bump quadrature order from q=2p to q=2p+4 -----------------------
# Production script uses q=2*p which only exactly integrates polynomials
# up to degree 4p-1. The toroidal mapping introduces 1/R, 1/R^2 in the
# RHS integrand, so the projection ``p0_dbc(f)`` and the assembled mass /
# stiffness blocks all carry quadrature error. Increase q and see whether
# the error floor moves.
q_hi = 2 * P + 4
seq_hi = DeRhamSequence(
    ns, ps, q_hi, TYPES, polar=True, tol=cg_tol, maxiter=cg_maxiter
)
seq_hi.set_map(toroid_map(epsilon=EPSILON))
seq_hi.evaluate_1d()
seq_hi.assemble_mass_matrix(0)
seq_hi.assemble_mass_matrix(1)
seq_hi.set_operators(
    assemble_tensor_mass_preconditioner(
        seq_hi, seq_hi.get_operators(), ks=(0,), rank=1
    )
)
seq_hi.assemble_hodge_laplacian(0)

rhs_hi = seq_hi.p0_dbc(f_callable)
u_hat_hi = seq_hi.apply_inverse_hodge_laplacian(rhs_hi, 0, dirichlet=True)
u_h_quad_hi = evaluate_at_xq(
    seq_hi.e0_dbc_T @ u_hat_hi,
    *seq_hi._form_comp_info(0),
    (seq_hi.quad.ny, seq_hi.quad.nx, seq_hi.quad.nz),
    1,
)
err_hi = l2_relative_error(seq_hi, u_h_quad_hi, exact_u_at_quad(seq_hi))
print(f"[D] q=2p+4 (={q_hi}) relative L2 error: {err_hi:.6e}")


# %% Cell E: discrete consistency of the (f, u) pair at q=2p -----------------
# rhs = ∫ f φ_i J dξ is just the load vector. If the manufactured pair
# is consistent at this q, then K0 @ Π_L2(u_exact) ≈ rhs in the discrete
# weak sense. A large residual here at q=2p means the *integrals*
# defining either K0 or rhs are poorly resolved by Gauss-2p quadrature.
K0_u_proj = seq.apply_hodge_laplacian(u_proj_hat, 0, dirichlet=True)
resid = K0_u_proj - rhs
rel_resid = float(jnp.linalg.norm(resid) / jnp.linalg.norm(rhs))
print(f"[E] ||K0 Π_L2(u) - rhs|| / ||rhs|| at q=2p:    {rel_resid:.6e}")

# Same check at the higher quadrature from cell D, for direct comparison.
u_load_hi = seq_hi.p0_dbc(u)
u_proj_hat_hi = seq_hi.apply_inverse_mass_matrix(u_load_hi, 0, dirichlet=True)
K0_u_proj_hi = seq_hi.apply_hodge_laplacian(u_proj_hat_hi, 0, dirichlet=True)
rel_resid_hi = float(
    jnp.linalg.norm(K0_u_proj_hi - rhs_hi) / jnp.linalg.norm(rhs_hi)
)
print(f"[E] ||K0 Π_L2(u) - rhs|| / ||rhs|| at q=2p+4:  {rel_resid_hi:.6e}")
