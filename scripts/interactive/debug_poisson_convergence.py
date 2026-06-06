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
import math

import time

import jax
import jax.numpy as jnp

import mrx
from mrx.assembly import assemble_hodge_laplacian as assemble_hodge_laplacian_tp
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import assemble_tensor_mass_preconditioner
from mrx.solvers import solve_singular_cg
from mrx.quadrature import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# For local CPU debugging, unbatched outer loops have been faster and more
# stable than forcing a positive outer batch size.
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
    values = jnp.ones((seq.quad.ny, 1, 1)) * \
        u_r[None, :, None] * u_z[None, None, :]
    return values.reshape(-1, 1)


def exact_f_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    r = seq.quad.x_x
    chi = seq.quad.x_y
    z = seq.quad.x_z
    cos_chi = jnp.cos(2 * π * chi)[:, None, None]
    cos_z = jnp.cos(2 * π * z)[None, None, :]
    r_term = r[None, :, None]
    R = 1.0 + EPSILON * r_term * cos_chi
    values = cos_z * (
        -1.0 / EPSILON**2 * (1.0 - 4.0 * r_term**2)
        - 1.0 / (EPSILON * R) * (r_term / 2.0 - r_term**3) * cos_chi
        + 0.25 * (r_term**2 - r_term**4) / R**2
    )
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


def build_dense(matvec, n: int) -> jnp.ndarray:
    """Build a dense matrix from a matvec by probing with unit vectors."""
    eye = jnp.eye(n)
    return jax.vmap(matvec, in_axes=1, out_axes=1)(eye)


def _build_k0_case(
    p_loc: int,
    n_loc: int,
    q_loc: int,
    *,
    tol_loc: float = 1e-12,
    maxiter_loc: int = 2_000,
):
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    seq_loc = DeRhamSequence(
        ns_loc, ps_loc, q_loc, TYPES, polar=True, tol=tol_loc, maxiter=maxiter_loc
    )
    seq_loc.set_map(toroid_map(epsilon=EPSILON))
    seq_loc.evaluate_1d()
    seq_loc.assemble_mass_matrix(0)
    seq_loc.assemble_mass_matrix(1)
    seq_loc.set_operators(
        assemble_tensor_mass_preconditioner(
            seq_loc, seq_loc.get_operators(), ks=(0,), rank=1
        )
    )
    seq_loc.assemble_hodge_laplacian(0)

    rhs_loc = seq_loc.p0_dbc(f_callable)
    ci_loc, cs_loc = seq_loc._form_comp_info(0)
    quad_shape_loc = (seq_loc.quad.ny, seq_loc.quad.nx, seq_loc.quad.nz)
    u_exact_loc = exact_u_at_quad(seq_loc)
    return seq_loc, rhs_loc, ci_loc, cs_loc, quad_shape_loc, u_exact_loc


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
    assemble_tensor_mass_preconditioner(
        seq, seq.get_operators(), ks=(0,), rank=1)
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
    ns, ps, q, TYPES, polar=True, tol=1e-12, maxiter=2_000
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
err_tight = l2_relative_error(
    seq_tight, u_h_quad_tight, exact_u_at_quad(seq_tight))
print(f"[C] outer CG tol=1e-12 relative L2 error: {err_tight:.6e}")


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


# %% Cell F: q-sweep at fixed (p, n) -----------------------------------------
# Decisive plot for the quadrature hypothesis: hold (p, n) fixed, vary q,
# report (solve_err, proj_err, consistency_residual). If the curve plateaus
# at machine-precision-floor only for q >> 2p, then q=2p is the bug.
def _solve_at_q(qx: int) -> tuple[float, float, float]:
    seq_q = DeRhamSequence(
        ns, ps, qx, TYPES, polar=True, tol=cg_tol, maxiter=cg_maxiter
    )
    seq_q.set_map(toroid_map(epsilon=EPSILON))
    seq_q.evaluate_1d()
    seq_q.assemble_mass_matrix(0)
    seq_q.assemble_mass_matrix(1)
    seq_q.set_operators(
        assemble_tensor_mass_preconditioner(
            seq_q, seq_q.get_operators(), ks=(0,), rank=1
        )
    )
    seq_q.assemble_hodge_laplacian(0)
    quad_shape_q = (seq_q.quad.ny, seq_q.quad.nx, seq_q.quad.nz)
    ci_q, cs_q = seq_q._form_comp_info(0)
    u_exact_q = exact_u_at_quad(seq_q)

    rhs_q = seq_q.p0_dbc(f_callable)
    u_hat_q = seq_q.apply_inverse_hodge_laplacian(rhs_q, 0, dirichlet=True)
    u_quad_q = evaluate_at_xq(
        seq_q.e0_dbc_T @ u_hat_q, ci_q, cs_q, quad_shape_q, 1
    )
    e_solve = l2_relative_error(seq_q, u_quad_q, u_exact_q)

    u_load_q = seq_q.p0_dbc(u)
    u_proj_q = seq_q.apply_inverse_mass_matrix(u_load_q, 0, dirichlet=True)
    u_proj_quad_q = evaluate_at_xq(
        seq_q.e0_dbc_T @ u_proj_q, ci_q, cs_q, quad_shape_q, 1
    )
    e_proj = l2_relative_error(seq_q, u_proj_quad_q, u_exact_q)

    K_u = seq_q.apply_hodge_laplacian(u_proj_q, 0, dirichlet=True)
    rel = float(jnp.linalg.norm(K_u - rhs_q) / jnp.linalg.norm(rhs_q))
    return e_solve, e_proj, rel


q_list = (2 * P, 2 * P + 2, 2 * P + 4, 2 * P + 8, 4 * P)
print(f"[F] q-sweep at p={P}, ns={ns}")
print(f"    {'q':>4}  {'solve_err':>12}  {'proj_floor':>12}  {'cons_resid':>12}")
for qx in q_list:
    e_s, e_p, r = _solve_at_q(qx)
    print(f"    {qx:>4}  {e_s:>12.4e}  {e_p:>12.4e}  {r:>12.4e}")


# %% Cell G: (p, n) sweep of the L2 projection floor only --------------------
# Cell F showed solve_err == proj_floor at every q, so the CG solve is already
# returning the best L2 projection and the "floor" IS the discretization
# error. This cell measures ||u - Π_L2 u|| as a function of (p, n) with no
# solver in the loop. Expected behaviour: O(h^{p+1}) decay, where h ~ 1/n.
# If the floor does NOT decay at the expected rate, the space/BC setup is the
# bug (e.g. clamped+polar can't represent u near the axis/boundary, or the
# manufactured u violates a regularity constraint of the discrete space).
def _require_valid_resolution(p_loc: int, n_loc: int) -> None:
    if n_loc <= p_loc:
        raise ValueError(f"Need n > p; got n={n_loc}, p={p_loc}")


def _proj_floor(p_loc: int, n_loc: int, q_loc: int | None = None) -> float:
    _require_valid_resolution(p_loc, n_loc)
    qx = 2 * p_loc + 4 if q_loc is None else q_loc
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)
    seq_g = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-13, maxiter=2_000
    )
    seq_g.set_map(toroid_map(epsilon=EPSILON))
    seq_g.evaluate_1d()
    seq_g.assemble_mass_matrix(0)
    u_exact_g = exact_u_at_quad(seq_g)
    u_load_g = seq_g.p0_dbc(u)
    u_proj_g = seq_g.apply_inverse_mass_matrix(u_load_g, 0, dirichlet=True)
    ci_g, cs_g = seq_g._form_comp_info(0)
    quad_shape_g = (seq_g.quad.ny, seq_g.quad.nx, seq_g.quad.nz)
    u_proj_quad_g = evaluate_at_xq(
        seq_g.e0_dbc_T @ u_proj_g, ci_g, cs_g, quad_shape_g, 1
    )
    return l2_relative_error(seq_g, u_proj_quad_g, u_exact_g)


p_list = (1, 2, 3, 4)
n_list = (8, 12, 16)

print(f"[G] L2 projection floor ||u - Π_L2 u|| / ||u|| (q = 2p+4)")
header = "    " + "p\\n".ljust(6) + "".join(f"{n:>12d}" for n in n_list)
print(header)
results: dict[int, list[float]] = {}
for p_loc in p_list:
    row = []
    for n_loc in n_list:
        e = _proj_floor(p_loc, n_loc)
        row.append(e)
    results[p_loc] = row
    line = f"    p={p_loc:<4d}" + "".join(f"{e:>12.3e}" for e in row)
    print(line)

print()
print("[G] observed convergence rates (log2 ratio between successive n):")
for p_loc, row in results.items():
    rates = []
    for i in range(1, len(row)):
        ratio = row[i - 1] / row[i] if row[i] > 0 else float("nan")
        h_ratio = n_list[i] / n_list[i - 1]
        rate = math.log(ratio) / \
            math.log(h_ratio) if ratio > 0 else float("nan")
        rates.append(rate)
    rate_str = "".join(f"{r:>12.2f}" for r in rates)
    print(f"    p={p_loc}  rates: {rate_str}   (expected ~{p_loc + 1})")


# %% Cell H: fixed-p mesh sweep, solve error vs projection floor -------------
# Cell G can show the approximation space is healthy even when the production
# Poisson solve appears to stall. This cell checks exactly where that happens:
# hold p fixed (default p=2), run the full production-style K0 solve at q=2p,
# and compare the solve error to the L2 projection floor as n grows.
def _solve_vs_proj_at_n(p_loc: int, n_loc: int, q_loc: int | None = None) -> tuple[float, float, float]:
    _require_valid_resolution(p_loc, n_loc)
    qx = 2 * p_loc if q_loc is None else q_loc
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)
    seq_h = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=2_000
    )
    seq_h.set_map(toroid_map(epsilon=EPSILON))
    seq_h.evaluate_1d()
    seq_h.assemble_mass_matrix(0)
    seq_h.assemble_mass_matrix(1)
    seq_h.set_operators(
        assemble_tensor_mass_preconditioner(
            seq_h, seq_h.get_operators(), ks=(0,), rank=1
        )
    )
    seq_h.assemble_hodge_laplacian(0)

    ci_h, cs_h = seq_h._form_comp_info(0)
    quad_shape_h = (seq_h.quad.ny, seq_h.quad.nx, seq_h.quad.nz)
    u_exact_h = exact_u_at_quad(seq_h)

    rhs_h = seq_h.p0_dbc(f_callable)
    u_hat_h = seq_h.apply_inverse_hodge_laplacian(rhs_h, 0, dirichlet=True)
    u_quad_h = evaluate_at_xq(seq_h.e0_dbc_T @ u_hat_h,
                              ci_h, cs_h, quad_shape_h, 1)
    solve_err = l2_relative_error(seq_h, u_quad_h, u_exact_h)

    u_load_h = seq_h.p0_dbc(u)
    u_proj_h = seq_h.apply_inverse_mass_matrix(u_load_h, 0, dirichlet=True)
    u_proj_quad_h = evaluate_at_xq(
        seq_h.e0_dbc_T @ u_proj_h, ci_h, cs_h, quad_shape_h, 1
    )
    proj_err = l2_relative_error(seq_h, u_proj_quad_h, u_exact_h)

    gap_ratio = solve_err / proj_err if proj_err > 0 else float("nan")
    return solve_err, proj_err, gap_ratio


def _solve_vs_proj_at_n_fast_k0(
    p_loc: int, n_loc: int, q_loc: int | None = None
) -> tuple[float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    seq_h = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=2_000
    )
    seq_h.set_map(toroid_map(epsilon=EPSILON))
    seq_h.evaluate_1d()
    seq_h.assemble_mass_matrix(0)
    assemble_hodge_laplacian_tp(seq_h, 0)

    if (
        seq_h.grad_grad is None
        or seq_h.e0_dbc is None
        or seq_h.e0_dbc_T is None
        or seq_h.dd0_diaginv_dbc is None
    ):
        raise ValueError(
            "Fast k=0 solve path requires assembled scalar stiffness data")

    ci_h, cs_h = seq_h._form_comp_info(0)
    quad_shape_h = (seq_h.quad.ny, seq_h.quad.nx, seq_h.quad.nz)
    u_exact_h = exact_u_at_quad(seq_h)

    def stiffness_matvec(v: jnp.ndarray) -> jnp.ndarray:
        return seq_h.e0_dbc @ (seq_h.grad_grad @ (seq_h.e0_dbc_T @ v))

    rhs_h = seq_h.p0_dbc(f_callable)
    u_hat_h, _info_h = solve_singular_cg(
        stiffness_matvec,
        rhs_h,
        mass_matvec=lambda v: seq_h.apply_mass_matrix(v, 0, dirichlet=True),
        precond_matvec=lambda v, diaginv=seq_h.dd0_diaginv_dbc: diaginv * v,
        tol=seq_h.tol,
        maxiter=seq_h.maxiter,
    )
    u_quad_h = evaluate_at_xq(seq_h.e0_dbc_T @ u_hat_h,
                              ci_h, cs_h, quad_shape_h, 1)
    solve_err = l2_relative_error(seq_h, u_quad_h, u_exact_h)

    u_load_h = seq_h.p0_dbc(u)
    u_proj_h = seq_h.apply_inverse_mass_matrix(u_load_h, 0, dirichlet=True)
    u_proj_quad_h = evaluate_at_xq(
        seq_h.e0_dbc_T @ u_proj_h, ci_h, cs_h, quad_shape_h, 1
    )
    proj_err = l2_relative_error(seq_h, u_proj_quad_h, u_exact_h)

    gap_ratio = solve_err / proj_err if proj_err > 0 else float("nan")
    return solve_err, proj_err, gap_ratio


P_H = 2
N_LIST_H = (8, 12, 16)

print(
    f"[H] fixed-p sweep at p={P_H}: production solve vs projection floor (q=2p)")
print(f"    {'n':>4}  {'solve_err':>12}  {'proj_floor':>12}  {'solve/proj':>12}")
for n_loc in N_LIST_H:
    solve_err, proj_err, gap_ratio = _solve_vs_proj_at_n(P_H, n_loc)
    print(
        f"    {n_loc:>4}  {solve_err:>12.4e}  {proj_err:>12.4e}  {gap_ratio:>12.4f}")


# %% Cell I: compact local convergence sweep at higher quadrature ------------
# Small non-SLURM convergence check for the full production-style solve using
# the higher quadrature suggested by the earlier diagnostics.
P_LIST_I = (3, 4)
N_LIST_I = (6, 8, 10, 12, 14, 16)

print(f"[I] compact local solve convergence sweep (q = 2p+4)")
header = "    " + "p\\n".ljust(6) + "".join(f"{n:>12d}" for n in N_LIST_I)
print(header)

solve_results_i: dict[int, list[float]] = {}
gap_results_i: dict[int, list[float]] = {}
for p_loc in P_LIST_I:
    solve_row = []
    gap_row = []
    for n_loc in N_LIST_I:
        solve_err, proj_err, gap_ratio = _solve_vs_proj_at_n(
            p_loc, n_loc, q_loc=2 * p_loc + 4
        )
        solve_row.append(solve_err)
        gap_row.append(gap_ratio)
    solve_results_i[p_loc] = solve_row
    gap_results_i[p_loc] = gap_row
    line = f"    p={p_loc:<4d}" + "".join(f"{e:>12.3e}" for e in solve_row)
    print(line)

print()
print("[I] solve/projection ratios at the same (p, n):")
for p_loc, row in gap_results_i.items():
    line = f"    p={p_loc:<4d}" + "".join(f"{g:>12.4f}" for g in row)
    print(line)

print()
print("[I] observed solve convergence rates (log2 ratio between successive n):")
for p_loc, row in solve_results_i.items():
    rates = []
    for i in range(1, len(row)):
        ratio = row[i - 1] / row[i] if row[i] > 0 else float("nan")
        h_ratio = N_LIST_I[i] / N_LIST_I[i - 1]
        rate = math.log(ratio) / \
            math.log(h_ratio) if ratio > 0 else float("nan")
        rates.append(rate)
    rate_str = "".join(f"{r:>12.2f}" for r in rates)
    print(f"    p={p_loc}  rates: {rate_str}   (expected ~{p_loc + 1})")


# %% Cell J: tolerance diagnostic (solver info vs residual vs L2 error) ------
# This isolates whether the CG stopping tolerance can explain a visible
# L2-error plateau by reporting three quantities for the same (p, n, q):
# - solver info (negative means converged, |info| = iteration count)
# - algebraic residual ||K u - rhs||_2 / ||rhs||_2
# - relative L2 error against the manufactured exact solution
def _tol_diagnostic(
    p_loc: int,
    n_loc: int,
    tol_loc: float,
    q_loc: int | None = None,
) -> tuple[int, float, float, jnp.ndarray]:
    qx = 2 * p_loc if q_loc is None else q_loc
    seq_j, rhs_j, ci_j, cs_j, quad_shape_j, u_exact_j = _build_k0_case(
        p_loc, n_loc, qx
    )

    u_hat_j, info_j = seq_j.apply_inverse_hodge_laplacian(
        rhs_j, 0, dirichlet=True, tol=tol_loc, return_info=True
    )

    Ku_j = seq_j.apply_hodge_laplacian(u_hat_j, 0, dirichlet=True)
    rel_alg_resid = float(jnp.linalg.norm(
        Ku_j - rhs_j) / jnp.linalg.norm(rhs_j))

    u_quad_j = evaluate_at_xq(seq_j.e0_dbc_T @ u_hat_j,
                              ci_j, cs_j, quad_shape_j, 1)
    rel_l2_err = l2_relative_error(seq_j, u_quad_j, u_exact_j)
    return int(info_j), rel_alg_resid, rel_l2_err, u_hat_j


def _tol_diagnostic_from_case(
    seq_j: DeRhamSequence,
    rhs_j: jnp.ndarray,
    ci_j,
    cs_j,
    quad_shape_j,
    u_exact_j: jnp.ndarray,
    tol_loc: float,
    guess: jnp.ndarray | None = None,
) -> tuple[int, float, float, jnp.ndarray]:
    u_hat_j, info_j = seq_j.apply_inverse_hodge_laplacian(
        rhs_j, 0, dirichlet=True, tol=tol_loc, guess=guess, return_info=True
    )

    Ku_j = seq_j.apply_hodge_laplacian(u_hat_j, 0, dirichlet=True)
    rel_alg_resid = float(jnp.linalg.norm(
        Ku_j - rhs_j) / jnp.linalg.norm(rhs_j))

    u_quad_j = evaluate_at_xq(seq_j.e0_dbc_T @ u_hat_j,
                              ci_j, cs_j, quad_shape_j, 1)
    rel_l2_err = l2_relative_error(seq_j, u_quad_j, u_exact_j)
    return int(info_j), rel_alg_resid, rel_l2_err, u_hat_j


P_J = 3
N_J = 8
Q_J = 2 * P_J
TOLS_J = (1e-9, 1e-10, 1e-11, 1e-12)

print(f"[J] tolerance diagnostic at p={P_J}, n={N_J}, q={Q_J}")
print(f"    {'tol':>10}  {'info':>8}  {'alg_resid':>12}  {'rel_L2_err':>12}")
seq_j, rhs_j, ci_j, cs_j, quad_shape_j, u_exact_j = _build_k0_case(
    P_J, N_J, Q_J)
guess_j = None
for tol_loc in TOLS_J:
    info_j, rel_alg_resid, rel_l2_err, guess_j = _tol_diagnostic_from_case(
        seq_j, rhs_j, ci_j, cs_j, quad_shape_j, u_exact_j, tol_loc, guess=guess_j
    )
    print(
        f"    {tol_loc:>10.1e}  {info_j:>8d}  {rel_alg_resid:>12.4e}  {rel_l2_err:>12.4e}"
    )


# %% Cell K: direct dense solve vs iterative CG on the same K0 ---------------
# Build a dense view of the DBC k=0 operator by probing with unit vectors,
# then compare the direct dense solve against the production iterative path.
def _direct_vs_iterative_k0(
    p_loc: int,
    n_loc: int,
    tol_loc: float,
    q_loc: int | None = None,
) -> tuple[int, int, float, float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    seq_k, rhs_k, ci_k, cs_k, quad_shape_k, u_exact_k = _build_k0_case(
        p_loc, n_loc, qx
    )
    n_dof = seq_k.n0_dbc

    K_dense = build_dense(
        lambda x: seq_k.apply_hodge_laplacian(x, 0, dirichlet=True), n_dof
    )
    u_direct = jnp.linalg.solve(K_dense, rhs_k)
    u_iter, info_iter = seq_k.apply_inverse_hodge_laplacian(
        rhs_k, 0, dirichlet=True, tol=tol_loc, return_info=True
    )

    rel_coeff_diff = float(
        jnp.linalg.norm(u_iter - u_direct) / jnp.linalg.norm(u_direct)
    )
    rel_alg_resid_direct = float(
        jnp.linalg.norm(K_dense @ u_direct - rhs_k) / jnp.linalg.norm(rhs_k)
    )
    rel_alg_resid_iter = float(
        jnp.linalg.norm(seq_k.apply_hodge_laplacian(
            u_iter, 0, dirichlet=True) - rhs_k)
        / jnp.linalg.norm(rhs_k)
    )

    u_direct_quad = evaluate_at_xq(
        seq_k.e0_dbc_T @ u_direct, ci_k, cs_k, quad_shape_k, 1
    )
    u_iter_quad = evaluate_at_xq(
        seq_k.e0_dbc_T @ u_iter, ci_k, cs_k, quad_shape_k, 1
    )
    rel_l2_direct = l2_relative_error(seq_k, u_direct_quad, u_exact_k)
    rel_l2_iter = l2_relative_error(seq_k, u_iter_quad, u_exact_k)
    return (
        n_dof,
        int(info_iter),
        rel_coeff_diff,
        rel_alg_resid_direct,
        rel_alg_resid_iter,
        rel_l2_direct,
        rel_l2_iter,
    )


P_K = 2
N_K = 12
Q_K = 2 * P_K
TOL_K = 1e-9

print(f"[K] direct dense solve vs iterative CG at p={P_K}, n={N_K}, q={Q_K}")
print(
    f"    {'ndof':>6}  {'info':>8}  {'coeff_diff':>12}  {'dense_resid':>12}  {'iter_resid':>12}  {'dense_L2':>12}  {'iter_L2':>12}"
)
(
    n_dof_k,
    info_iter_k,
    rel_coeff_diff_k,
    rel_alg_resid_direct_k,
    rel_alg_resid_iter_k,
    rel_l2_direct_k,
    rel_l2_iter_k,
) = _direct_vs_iterative_k0(P_K, N_K, TOL_K, q_loc=Q_K)
print(
    f"    {n_dof_k:>6d}  {info_iter_k:>8d}  {rel_coeff_diff_k:>12.4e}  {rel_alg_resid_direct_k:>12.4e}  {rel_alg_resid_iter_k:>12.4e}  {rel_l2_direct_k:>12.4e}  {rel_l2_iter_k:>12.4e}"
)


# %% Cell L: dense projected consistency check --------------------------------
# Build dense views of M0 and K0, form the dense L2 projection u_proj = M0^{-1}
# p0_dbc(u), and test whether the manufactured pair is consistent with the
# assembled discrete operator via ||K0 u_proj - rhs|| / ||rhs||.
def _dense_projected_consistency_k0(
    p_loc: int,
    n_loc: int,
    q_loc: int | None = None,
) -> tuple[int, float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    seq_l, rhs_l, ci_l, cs_l, quad_shape_l, u_exact_l = _build_k0_case(
        p_loc, n_loc, qx
    )
    n_dof = seq_l.n0_dbc

    M_dense = build_dense(
        lambda x: seq_l.apply_mass_matrix(x, 0, dirichlet=True), n_dof
    )
    K_dense = build_dense(
        lambda x: seq_l.apply_hodge_laplacian(x, 0, dirichlet=True), n_dof
    )

    u_load_l = seq_l.p0_dbc(u)
    u_proj_dense = jnp.linalg.solve(M_dense, u_load_l)

    rel_consistency = float(
        jnp.linalg.norm(K_dense @ u_proj_dense - rhs_l) /
        jnp.linalg.norm(rhs_l)
    )
    rel_mass_resid = float(
        jnp.linalg.norm(M_dense @ u_proj_dense - u_load_l) /
        jnp.linalg.norm(u_load_l)
    )

    u_proj_quad_l = evaluate_at_xq(
        seq_l.e0_dbc_T @ u_proj_dense, ci_l, cs_l, quad_shape_l, 1
    )
    rel_proj_l2 = l2_relative_error(seq_l, u_proj_quad_l, u_exact_l)
    return n_dof, rel_consistency, rel_mass_resid, rel_proj_l2


P_L = 2
N_L = 8
Q_L = 2 * P_L

print(f"[L] dense projected consistency at p={P_L}, n={N_L}, q={Q_L}")
print(
    f"    {'ndof':>6}  {'consistency':>12}  {'mass_resid':>12}  {'proj_L2':>12}"
)
n_dof_l, rel_consistency_l, rel_mass_resid_l, rel_proj_l2_l = _dense_projected_consistency_k0(
    P_L, N_L, q_loc=Q_L
)
print(
    f"    {n_dof_l:>6d}  {rel_consistency_l:>12.4e}  {rel_mass_resid_l:>12.4e}  {rel_proj_l2_l:>12.4e}"
)


# %% Cell M: localize the dense consistency residual -------------------------
# Inspect the coefficient residual r = K0 u_proj - rhs and also a quadrature
# representation of M0^{-1} r to see whether the inconsistency is concentrated
# near the axis / boundary or spread throughout the domain.
def _dense_consistency_residual_profile_k0(
    p_loc: int,
    n_loc: int,
    q_loc: int | None = None,
    *,
    top_k: int = 10,
) -> tuple[float, list[tuple[int, float]], float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    seq_m, rhs_m, ci_m, cs_m, quad_shape_m, _u_exact_m = _build_k0_case(
        p_loc, n_loc, qx
    )
    n_dof = seq_m.n0_dbc

    M_dense = build_dense(
        lambda x: seq_m.apply_mass_matrix(x, 0, dirichlet=True), n_dof
    )
    K_dense = build_dense(
        lambda x: seq_m.apply_hodge_laplacian(x, 0, dirichlet=True), n_dof
    )

    u_load_m = seq_m.p0_dbc(u)
    u_proj_dense = jnp.linalg.solve(M_dense, u_load_m)
    coeff_resid = K_dense @ u_proj_dense - rhs_m
    rel_consistency = float(
        jnp.linalg.norm(coeff_resid) / jnp.linalg.norm(rhs_m)
    )

    abs_coeff = jnp.abs(coeff_resid)
    top_idx = jnp.argsort(abs_coeff)[-top_k:][::-1]
    top_entries = [
        (int(idx), float(coeff_resid[idx]))
        for idx in top_idx.tolist()
    ]

    resid_repr = jnp.linalg.solve(M_dense, coeff_resid)
    resid_quad = evaluate_at_xq(
        seq_m.e0_dbc_T @ resid_repr, ci_m, cs_m, quad_shape_m, 1
    )[:, 0]
    abs_resid_quad = jnp.abs(resid_quad)

    weights = seq_m.jacobian_j * seq_m.quad.w
    energy = abs_resid_quad**2 * weights
    total_energy = float(jnp.sum(energy))

    r_quad = jnp.broadcast_to(
        seq_m.quad.x_x[None, :, None], quad_shape_m).reshape(-1)
    axis_mask = r_quad < 0.15
    boundary_mask = r_quad > 0.85
    interior_mask = jnp.logical_and(r_quad > 0.35, r_quad < 0.65)

    def energy_fraction(mask):
        if total_energy == 0.0:
            return 0.0
        return float(jnp.sum(jnp.where(mask, energy, 0.0)) / total_energy)

    return (
        rel_consistency,
        top_entries,
        energy_fraction(axis_mask),
        energy_fraction(boundary_mask),
        energy_fraction(interior_mask),
    )


P_M = 2
N_M = 8
Q_M = 2 * P_M
TOP_K_M = 8

print(
    f"[M] dense consistency residual localization at p={P_M}, n={N_M}, q={Q_M}")
(
    rel_consistency_m,
    top_entries_m,
    axis_frac_m,
    boundary_frac_m,
    interior_frac_m,
) = _dense_consistency_residual_profile_k0(P_M, N_M, q_loc=Q_M, top_k=TOP_K_M)
print(f"    rel_consistency: {rel_consistency_m:.4e}")
print(
    f"    residual energy fractions (mass-inverted quad view): axis={axis_frac_m:.3f}, boundary={boundary_frac_m:.3f}, interior={interior_frac_m:.3f}"
)
print(f"    top {TOP_K_M} coefficient residual entries (index, value):")
for idx_m, val_m in top_entries_m:
    print(f"        {idx_m:>6d}  {val_m: .4e}")


# %% Cell N: unrestricted L2 projection of a non-DBC function ---------------
# Probe whether the boundary-localized issue is tied specifically to DBC
# handling by projecting a smooth scalar that does NOT vanish on the boundary
# with the unrestricted k=0 mass matrix.
def v_non_dbc(x: jnp.ndarray) -> jnp.ndarray:
    r, chi, z = x
    return (
        (1.0 + 0.3 * r**2 + 0.2 * r**4 * jnp.cos(2 * π * chi))
        * jnp.cos(2 * π * z)
        * jnp.ones(1)
    )


def exact_v_non_dbc_at_quad(seq_n: DeRhamSequence) -> jnp.ndarray:
    r = seq_n.quad.x_x
    chi = seq_n.quad.x_y
    z = seq_n.quad.x_z
    vr = 1.0 + 0.3 * r**2
    vt = 0.2 * jnp.cos(2 * π * chi)
    vz = jnp.cos(2 * π * z)
    values = (vr[None, :, None] + vt[:, None, None] *
              r[None, :, None]**4) * vz[None, None, :]
    return values.reshape(-1, 1)


def _non_dbc_projection_probe(
    p_loc: int,
    n_loc: int,
    q_loc: int | None = None,
) -> tuple[float, float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    seq_n = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=2_000
    )
    seq_n.set_map(toroid_map(epsilon=EPSILON))
    seq_n.evaluate_1d()
    seq_n.assemble_mass_matrix(0)

    v_load = seq_n.p0(v_non_dbc)
    v_proj = seq_n.apply_inverse_mass_matrix(v_load, 0, dirichlet=False)

    mass_resid = float(
        jnp.linalg.norm(seq_n.apply_mass_matrix(
            v_proj, 0, dirichlet=False) - v_load)
        / jnp.linalg.norm(v_load)
    )

    ci_n, cs_n = seq_n._form_comp_info(0)
    quad_shape_n = (seq_n.quad.ny, seq_n.quad.nx, seq_n.quad.nz)
    v_proj_quad = evaluate_at_xq(
        seq_n.e0_T @ v_proj, ci_n, cs_n, quad_shape_n, 1)[:, 0]
    v_exact_quad = exact_v_non_dbc_at_quad(seq_n)[:, 0]
    rel_l2_err = l2_relative_error(
        seq_n, v_proj_quad[:, None], v_exact_quad[:, None])

    err_quad = v_exact_quad - v_proj_quad
    weights = seq_n.jacobian_j * seq_n.quad.w
    energy = err_quad**2 * weights
    total_energy = float(jnp.sum(energy))
    r_quad = jnp.broadcast_to(
        seq_n.quad.x_x[None, :, None], quad_shape_n).reshape(-1)

    def energy_fraction(mask):
        if total_energy == 0.0:
            return 0.0
        return float(jnp.sum(jnp.where(mask, energy, 0.0)) / total_energy)

    axis_frac = energy_fraction(r_quad < 0.15)
    boundary_frac = energy_fraction(r_quad > 0.85)
    interior_frac = energy_fraction(
        jnp.logical_and(r_quad > 0.35, r_quad < 0.65))
    return rel_l2_err, mass_resid, axis_frac, boundary_frac, interior_frac


P_N = 2
N_N = 8
Q_N = 2 * P_N

print(
    f"[N] unrestricted L2 projection of non-DBC function at p={P_N}, n={N_N}, q={Q_N}")
print(
    f"    {'rel_L2_err':>12}  {'mass_resid':>12}  {'axis_frac':>10}  {'boundary_frac':>14}  {'interior_frac':>14}"
)
rel_l2_err_n, mass_resid_n, axis_frac_n, boundary_frac_n, interior_frac_n = _non_dbc_projection_probe(
    P_N, N_N, q_loc=Q_N
)
print(
    f"    {rel_l2_err_n:>12.4e}  {mass_resid_n:>12.4e}  {axis_frac_n:>10.3f}  {boundary_frac_n:>14.3f}  {interior_frac_n:>14.3f}"
)


# %% Cell O: self-contained non-DBC projection sweep ------------------------


jax.config.update("jax_enable_x64", True)
mrx.MAP_BATCH_SIZE_OUTER = None

π = jnp.pi
EPSILON = 1 / 3
TYPES = ("clamped", "periodic", "periodic")


def _require_valid_resolution(p_loc: int, n_loc: int) -> None:
    if n_loc <= p_loc:
        raise ValueError(f"Need n > p; got n={n_loc}, p={p_loc}")


def v_non_dbc(x: jnp.ndarray) -> jnp.ndarray:
    r, _chi, z = x
    return (0.5 + r**2 + r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def exact_v_non_dbc_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    vr = 0.5 + seq.quad.x_x**2 + seq.quad.x_x**4
    vz = jnp.cos(2 * π * seq.quad.x_z)
    values = jnp.ones((seq.quad.ny, 1, 1)) * \
        vr[None, :, None] * vz[None, None, :]
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


def _non_dbc_projection_probe(
    p_loc: int,
    n_loc: int,
    q_loc: int | None = None,
) -> tuple[float, float, float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    seq_n = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=2_000
    )
    seq_n.set_map(toroid_map(epsilon=EPSILON))
    seq_n.evaluate_1d()
    seq_n.assemble_mass_matrix(0)

    v_load = seq_n.p0(v_non_dbc)
    v_proj = seq_n.apply_inverse_mass_matrix(v_load, 0, dirichlet=False)

    mass_resid = float(
        jnp.linalg.norm(seq_n.apply_mass_matrix(
            v_proj, 0, dirichlet=False) - v_load)
        / jnp.linalg.norm(v_load)
    )

    ci_n, cs_n = seq_n._form_comp_info(0)
    quad_shape_n = (seq_n.quad.ny, seq_n.quad.nx, seq_n.quad.nz)
    v_proj_quad = evaluate_at_xq(
        seq_n.e0_T @ v_proj, ci_n, cs_n, quad_shape_n, 1)[:, 0]
    v_exact_quad = exact_v_non_dbc_at_quad(seq_n)[:, 0]
    rel_l2_err = l2_relative_error(
        seq_n, v_proj_quad[:, None], v_exact_quad[:, None])

    err_quad = v_exact_quad - v_proj_quad
    weights = seq_n.jacobian_j * seq_n.quad.w
    energy = err_quad**2 * weights
    total_energy = float(jnp.sum(energy))
    r_quad = jnp.broadcast_to(
        seq_n.quad.x_x[None, :, None], quad_shape_n).reshape(-1)

    def energy_fraction(mask):
        if total_energy == 0.0:
            return 0.0
        return float(jnp.sum(jnp.where(mask, energy, 0.0)) / total_energy)

    axis_frac = energy_fraction(r_quad < 0.15)
    boundary_frac = energy_fraction(r_quad > 0.85)
    interior_frac = energy_fraction(
        jnp.logical_and(r_quad > 0.35, r_quad < 0.65))
    return rel_l2_err, mass_resid, axis_frac, boundary_frac, interior_frac


P_LIST_O = (2, 3)
N_LIST_O = (8, 10, 12)

print("[O] unrestricted non-DBC L2 projection convergence check (q = 2p)")
header = "    " + "p\\n".ljust(6) + "".join(f"{n:>12d}" for n in N_LIST_O)
print(header)

results_o: dict[int, list[float]] = {}
for p_loc in P_LIST_O:
    row = []
    for n_loc in N_LIST_O:
        rel_l2_err_o, _mass_resid_o, _axis_frac_o, _boundary_frac_o, _interior_frac_o = _non_dbc_projection_probe(
            p_loc, n_loc, q_loc=2 * p_loc
        )
        row.append(rel_l2_err_o)
    results_o[p_loc] = row
    line = f"    p={p_loc:<4d}" + "".join(f"{e:>12.3e}" for e in row)
    print(line)

print()
print("[O] observed convergence rates (log2 ratio between successive n):")
for p_loc, row in results_o.items():
    rates = []
    for i in range(1, len(row)):
        ratio = row[i - 1] / row[i] if row[i] > 0 else float("nan")
        h_ratio = N_LIST_O[i] / N_LIST_O[i - 1]
        rate = math.log(ratio) / \
            math.log(h_ratio) if ratio > 0 else float("nan")
        rates.append(rate)
    rate_str = "".join(f"{r:>12.2f}" for r in rates)
    print(
        f"    p={p_loc}  rates: {rate_str}   (expected smooth trend, ideally improving with p)"
    )


# %% Cell P: self-contained DBC projection sweep for manufactured u ---------


jax.config.update("jax_enable_x64", True)
mrx.MAP_BATCH_SIZE_OUTER = None

π = jnp.pi
EPSILON = 1 / 3
TYPES = ("clamped", "periodic", "periodic")


def _require_valid_resolution(p_loc: int, n_loc: int) -> None:
    if n_loc <= p_loc:
        raise ValueError(f"Need n > p; got n={n_loc}, p={p_loc}")


def u(x: jnp.ndarray) -> jnp.ndarray:
    r, _chi, z = x
    return 0.25 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def exact_u_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    u_r = 0.25 * (seq.quad.x_x**2 - seq.quad.x_x**4)
    u_z = jnp.cos(2 * π * seq.quad.x_z)
    values = jnp.ones((seq.quad.ny, 1, 1)) * \
        u_r[None, :, None] * u_z[None, None, :]
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


def _dbc_projection_probe_u(
    p_loc: int,
    n_loc: int,
    q_loc: int | None = None,
) -> tuple[float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    seq_p = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=2_000
    )
    seq_p.set_map(toroid_map(epsilon=EPSILON))
    seq_p.evaluate_1d()
    seq_p.assemble_mass_matrix(0)

    u_load = seq_p.p0_dbc(u)
    u_proj = seq_p.apply_inverse_mass_matrix(u_load, 0, dirichlet=True)

    mass_resid = float(
        jnp.linalg.norm(seq_p.apply_mass_matrix(
            u_proj, 0, dirichlet=True) - u_load)
        / jnp.linalg.norm(u_load)
    )

    ci_p, cs_p = seq_p._form_comp_info(0)
    quad_shape_p = (seq_p.quad.ny, seq_p.quad.nx, seq_p.quad.nz)
    u_proj_quad = evaluate_at_xq(
        seq_p.e0_dbc_T @ u_proj, ci_p, cs_p, quad_shape_p, 1)[:, 0]
    u_exact_quad = exact_u_at_quad(seq_p)[:, 0]
    rel_l2_err = l2_relative_error(
        seq_p, u_proj_quad[:, None], u_exact_quad[:, None])
    return rel_l2_err, mass_resid


P_LIST_P = (2, 3)
N_LIST_P = (8, 10, 12)

print("[P] DBC L2 projection convergence check for manufactured u (q = 2p)")
header = "    " + "p\\n".ljust(6) + "".join(f"{n:>12d}" for n in N_LIST_P)
print(header)

results_p: dict[int, list[float]] = {}
for p_loc in P_LIST_P:
    row = []
    for n_loc in N_LIST_P:
        rel_l2_err_p, _mass_resid_p = _dbc_projection_probe_u(
            p_loc, n_loc, q_loc=2 * p_loc
        )
        row.append(rel_l2_err_p)
    results_p[p_loc] = row
    line = f"    p={p_loc:<4d}" + "".join(f"{e:>12.3e}" for e in row)
    print(line)

print()
print("[P] observed convergence rates (log2 ratio between successive n):")
for p_loc, row in results_p.items():
    rates = []
    for i in range(1, len(row)):
        ratio = row[i - 1] / row[i] if row[i] > 0 else float("nan")
        h_ratio = N_LIST_P[i] / N_LIST_P[i - 1]
        rate = math.log(ratio) / \
            math.log(h_ratio) if ratio > 0 else float("nan")
        rates.append(rate)
    rate_str = "".join(f"{r:>12.2f}" for r in rates)
    print(f"    p={p_loc}  rates: {rate_str}   (expected ~{p_loc + 1})")


# %% Cell Q: self-contained unrestricted projection sweep for manufactured f -


jax.config.update("jax_enable_x64", True)
mrx.MAP_BATCH_SIZE_OUTER = None

π = jnp.pi
EPSILON = 1 / 3
TYPES = ("clamped", "periodic", "periodic")


def _require_valid_resolution(p_loc: int, n_loc: int) -> None:
    if n_loc <= p_loc:
        raise ValueError(f"Need n > p; got n={n_loc}, p={p_loc}")


def make_f(a: float):
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, chi, z = x
        R = 1 + a * r * jnp.cos(2 * π * chi)
        return (
            jnp.cos(2 * π * z)
            * (
                -1 / a**2 * (1 - 4 * r**2)
                - 1 / (a * R) * (r / 2 - r**3) * jnp.cos(2 * π * chi)
                + 0.25 * (r**2 - r**4) / R**2
            )
            * jnp.ones(1)
        )

    return f


f_callable = make_f(EPSILON)


def exact_f_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    r = seq.quad.x_x
    chi = seq.quad.x_y
    z = seq.quad.x_z
    cos_chi = jnp.cos(2 * π * chi)[:, None, None]
    cos_z = jnp.cos(2 * π * z)[None, None, :]
    r_term = r[None, :, None]
    R = 1.0 + EPSILON * r_term * cos_chi
    values = cos_z * (
        -1.0 / EPSILON**2 * (1.0 - 4.0 * r_term**2)
        - 1.0 / (EPSILON * R) * (r_term / 2.0 - r_term**3) * cos_chi
        + 0.25 * (r_term**2 - r_term**4) / R**2
    )
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


def _non_dbc_projection_probe_f(
    p_loc: int,
    n_loc: int,
    q_loc: int | None = None,
) -> tuple[float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    seq_q = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=2_000
    )
    seq_q.set_map(toroid_map(epsilon=EPSILON))
    seq_q.evaluate_1d()
    seq_q.assemble_mass_matrix(0)

    f_load = seq_q.p0(f_callable)
    f_proj = seq_q.apply_inverse_mass_matrix(f_load, 0, dirichlet=False)

    mass_resid = float(
        jnp.linalg.norm(seq_q.apply_mass_matrix(
            f_proj, 0, dirichlet=False) - f_load)
        / jnp.linalg.norm(f_load)
    )

    ci_q, cs_q = seq_q._form_comp_info(0)
    quad_shape_q = (seq_q.quad.ny, seq_q.quad.nx, seq_q.quad.nz)
    f_proj_quad = evaluate_at_xq(
        seq_q.e0_T @ f_proj, ci_q, cs_q, quad_shape_q, 1)[:, 0]
    f_exact_quad = exact_f_at_quad(seq_q)[:, 0]
    rel_l2_err = l2_relative_error(
        seq_q, f_proj_quad[:, None], f_exact_quad[:, None])
    return rel_l2_err, mass_resid


P_LIST_Q = (2, 3)
N_LIST_Q = (8, 10, 12)

print("[Q] unrestricted L2 projection convergence check for manufactured f (q = 2p)")
header = "    " + "p\\n".ljust(6) + "".join(f"{n:>12d}" for n in N_LIST_Q)
print(header)

results_q: dict[int, list[float]] = {}
for p_loc in P_LIST_Q:
    row = []
    for n_loc in N_LIST_Q:
        rel_l2_err_q, _mass_resid_q = _non_dbc_projection_probe_f(
            p_loc, n_loc, q_loc=2 * p_loc
        )
        row.append(rel_l2_err_q)
    results_q[p_loc] = row
    line = f"    p={p_loc:<4d}" + "".join(f"{e:>12.3e}" for e in row)
    print(line)

print()
print("[Q] observed convergence rates (log2 ratio between successive n):")
for p_loc, row in results_q.items():
    rates = []
    for i in range(1, len(row)):
        ratio = row[i - 1] / row[i] if row[i] > 0 else float("nan")
        h_ratio = N_LIST_Q[i] / N_LIST_Q[i - 1]
        rate = math.log(ratio) / \
            math.log(h_ratio) if ratio > 0 else float("nan")
        rates.append(rate)
    rate_str = "".join(f"{r:>12.2f}" for r in rates)
    print(
        f"    p={p_loc}  rates: {rate_str}   (expected smooth trend if f is resolved)"
    )


# %% Cell R: self-contained assembled-k0 debug solve sweep ------------------


jax.config.update("jax_enable_x64", True)
mrx.MAP_BATCH_SIZE_OUTER = None

π = jnp.pi
EPSILON = 1 / 3
TYPES = ("clamped", "periodic", "periodic")


def _require_valid_resolution(p_loc: int, n_loc: int) -> None:
    if n_loc <= p_loc:
        raise ValueError(f"Need n > p; got n={n_loc}, p={p_loc}")


def u(x: jnp.ndarray) -> jnp.ndarray:
    r, _chi, z = x
    return 0.25 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def make_f(a: float):
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, chi, z = x
        R = 1 + a * r * jnp.cos(2 * π * chi)
        return (
            jnp.cos(2 * π * z)
            * (
                -1 / a**2 * (1 - 4 * r**2)
                - 1 / (a * R) * (r / 2 - r**3) * jnp.cos(2 * π * chi)
                + 0.25 * (r**2 - r**4) / R**2
            )
            * jnp.ones(1)
        )

    return f


f_callable = make_f(EPSILON)


def exact_u_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    u_r = 0.25 * (seq.quad.x_x**2 - seq.quad.x_x**4)
    u_z = jnp.cos(2 * π * seq.quad.x_z)
    values = jnp.ones((seq.quad.ny, 1, 1)) * \
        u_r[None, :, None] * u_z[None, None, :]
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


def _solve_vs_proj_at_n_fast_k0(
    p_loc: int, n_loc: int, q_loc: int | None = None
) -> tuple[float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    seq_h = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=2_000
    )
    seq_h.set_map(toroid_map(epsilon=EPSILON))
    seq_h.evaluate_1d()
    seq_h.assemble_mass_matrix(0)
    assemble_hodge_laplacian_tp(seq_h, 0)

    if (
        seq_h.grad_grad is None
        or seq_h.e0_dbc is None
        or seq_h.e0_dbc_T is None
        or seq_h.dd0_diaginv_dbc is None
    ):
        raise ValueError(
            "Assembled k0 debug solve requires scalar stiffness data")

    ci_h, cs_h = seq_h._form_comp_info(0)
    quad_shape_h = (seq_h.quad.ny, seq_h.quad.nx, seq_h.quad.nz)
    u_exact_h = exact_u_at_quad(seq_h)

    def stiffness_matvec(v: jnp.ndarray) -> jnp.ndarray:
        return seq_h.e0_dbc @ (seq_h.grad_grad @ (seq_h.e0_dbc_T @ v))

    rhs_h = seq_h.p0_dbc(f_callable)
    u_hat_h, _info_h = solve_singular_cg(
        stiffness_matvec,
        rhs_h,
        mass_matvec=lambda v: seq_h.apply_mass_matrix(v, 0, dirichlet=True),
        precond_matvec=lambda v, diaginv=seq_h.dd0_diaginv_dbc: diaginv * v,
        tol=seq_h.tol,
        maxiter=seq_h.maxiter,
    )
    u_quad_h = evaluate_at_xq(seq_h.e0_dbc_T @ u_hat_h,
                              ci_h, cs_h, quad_shape_h, 1)
    solve_err = l2_relative_error(seq_h, u_quad_h, u_exact_h)

    u_load_h = seq_h.p0_dbc(u)
    u_proj_h = seq_h.apply_inverse_mass_matrix(u_load_h, 0, dirichlet=True)
    u_proj_quad_h = evaluate_at_xq(
        seq_h.e0_dbc_T @ u_proj_h, ci_h, cs_h, quad_shape_h, 1
    )
    proj_err = l2_relative_error(seq_h, u_proj_quad_h, u_exact_h)

    gap_ratio = solve_err / proj_err if proj_err > 0 else float("nan")
    return solve_err, proj_err, gap_ratio


P_LIST_R = (2, 3)
N_LIST_R = (8, 10, 12)

print("[R] assembled-k0 debug solve convergence check (q = 2p)")
header = "    " + "p\\n".ljust(6) + "".join(f"{n:>12d}" for n in N_LIST_R)
print(header)

solve_results_r: dict[int, list[float]] = {}
gap_results_r: dict[int, list[float]] = {}
for p_loc in P_LIST_R:
    solve_row = []
    gap_row = []
    for n_loc in N_LIST_R:
        solve_err_r, proj_err_r, gap_ratio_r = _solve_vs_proj_at_n_fast_k0(
            p_loc, n_loc, q_loc=2 * p_loc
        )
        solve_row.append(solve_err_r)
        gap_row.append(gap_ratio_r)
    solve_results_r[p_loc] = solve_row
    gap_results_r[p_loc] = gap_row
    line = f"    p={p_loc:<4d}" + "".join(f"{e:>12.3e}" for e in solve_row)
    print(line)

print()
print("[R] solve/projection ratios at the same (p, n):")
for p_loc, row in gap_results_r.items():
    line = f"    p={p_loc:<4d}" + "".join(f"{g:>12.4f}" for g in row)
    print(line)

print()
print("[R] observed convergence rates (log2 ratio between successive n):")
for p_loc, row in solve_results_r.items():
    rates = []
    for i in range(1, len(row)):
        ratio = row[i - 1] / row[i] if row[i] > 0 else float("nan")
        h_ratio = N_LIST_R[i] / N_LIST_R[i - 1]
        rate = math.log(ratio) / \
            math.log(h_ratio) if ratio > 0 else float("nan")
        rates.append(rate)
    rate_str = "".join(f"{r:>12.2f}" for r in rates)
    print(f"    p={p_loc}  rates: {rate_str}   (expected ~{p_loc + 1})")


# %% Cell S: p=2 production-vs-assembled k0 comparison ----------------------
# Compare the production k=0 operator path (K0 applied as G^T M1 G with the
# tensor-Hodge preconditioner) against the assembled scalar grad-grad debug
# path on the same manufactured problem. Keep n as small as possible while
# still probing whether the old production floor starts to appear locally.


jax.config.update("jax_enable_x64", True)
mrx.MAP_BATCH_SIZE_OUTER = None
mrx.MAP_BATCH_SIZE_INNER = 0

π = jnp.pi
EPSILON = 1 / 3
TYPES = ("clamped", "periodic", "periodic")


def _require_valid_resolution(p_loc: int, n_loc: int) -> None:
    if n_loc <= p_loc:
        raise ValueError(f"Need n > p; got n={n_loc}, p={p_loc}")


def u(x: jnp.ndarray) -> jnp.ndarray:
    r, _chi, z = x
    return 0.25 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def make_f(a: float):
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, chi, z = x
        R = 1 + a * r * jnp.cos(2 * π * chi)
        return (
            jnp.cos(2 * π * z)
            * (
                -1 / a**2 * (1 - 4 * r**2)
                - 1 / (a * R) * (r / 2 - r**3) * jnp.cos(2 * π * chi)
                + 0.25 * (r**2 - r**4) / R**2
            )
            * jnp.ones(1)
        )

    return f


f_callable = make_f(EPSILON)


def exact_u_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    u_r = 0.25 * (seq.quad.x_x**2 - seq.quad.x_x**4)
    u_z = jnp.cos(2 * π * seq.quad.x_z)
    values = jnp.ones((seq.quad.ny, 1, 1)) * \
        u_r[None, :, None] * u_z[None, None, :]
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


def _progress_s(message: str) -> None:
    print(message, flush=True)


def _compare_k0_realizations_at_n(
    p_loc: int,
    n_loc: int,
    q_loc: int | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    qx = 2 * p_loc if q_loc is None else q_loc
    _require_valid_resolution(p_loc, n_loc)
    ns_loc = (n_loc, 2 * n_loc, n_loc)
    ps_loc = (p_loc, p_loc, p_loc)

    _progress_s(f"[S] n={n_loc}: production path setup")

    seq_prod = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=50_000
    )
    seq_prod.set_map(toroid_map(epsilon=EPSILON))
    seq_prod.evaluate_1d()
    seq_prod.assemble_mass_matrix(0)
    seq_prod.assemble_mass_matrix(1)
    seq_prod.set_operators(
        assemble_tensor_mass_preconditioner(
            seq_prod, seq_prod.get_operators(), ks=(0,), rank=1
        )
    )
    seq_prod.assemble_hodge_laplacian(0)

    ci_prod, cs_prod = seq_prod._form_comp_info(0)
    quad_shape_prod = (seq_prod.quad.ny, seq_prod.quad.nx, seq_prod.quad.nz)
    u_exact_prod = exact_u_at_quad(seq_prod)
    rhs_prod = seq_prod.p0_dbc(f_callable)

    _progress_s(f"[S] n={n_loc}: production path solve")
    u_hat_prod = seq_prod.apply_inverse_hodge_laplacian(
        rhs_prod, 0, dirichlet=True)
    u_quad_prod = evaluate_at_xq(
        seq_prod.e0_dbc_T @ u_hat_prod, ci_prod, cs_prod, quad_shape_prod, 1
    )
    solve_err_prod = l2_relative_error(seq_prod, u_quad_prod, u_exact_prod)

    u_load_prod = seq_prod.p0_dbc(u)
    u_proj_prod = seq_prod.apply_inverse_mass_matrix(
        u_load_prod, 0, dirichlet=True)
    u_proj_quad_prod = evaluate_at_xq(
        seq_prod.e0_dbc_T @ u_proj_prod, ci_prod, cs_prod, quad_shape_prod, 1
    )
    proj_err = l2_relative_error(seq_prod, u_proj_quad_prod, u_exact_prod)

    k_prod_u_proj = seq_prod.apply_hodge_laplacian(
        u_proj_prod, 0, dirichlet=True)
    rel_consistency_prod = float(
        jnp.linalg.norm(k_prod_u_proj - rhs_prod) / jnp.linalg.norm(rhs_prod)
    )

    _progress_s(f"[S] n={n_loc}: assembled-k0 path setup")
    seq_asm = DeRhamSequence(
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=50_000
    )
    seq_asm.set_map(toroid_map(epsilon=EPSILON))
    seq_asm.evaluate_1d()
    seq_asm.assemble_mass_matrix(0)
    assemble_hodge_laplacian_tp(seq_asm, 0)

    ci_asm, cs_asm = seq_asm._form_comp_info(0)
    quad_shape_asm = (seq_asm.quad.ny, seq_asm.quad.nx, seq_asm.quad.nz)
    u_exact_asm = exact_u_at_quad(seq_asm)
    rhs_asm = seq_asm.p0_dbc(f_callable)

    def stiffness_matvec(v: jnp.ndarray) -> jnp.ndarray:
        return seq_asm.e0_dbc @ (seq_asm.grad_grad @ (seq_asm.e0_dbc_T @ v))

    _progress_s(f"[S] n={n_loc}: assembled-k0 path solve")
    u_hat_asm, _info_asm = solve_singular_cg(
        stiffness_matvec,
        rhs_asm,
        mass_matvec=lambda v: seq_asm.apply_mass_matrix(v, 0, dirichlet=True),
        precond_matvec=lambda v, diaginv=seq_asm.dd0_diaginv_dbc: diaginv * v,
        tol=seq_asm.tol,
        maxiter=seq_asm.maxiter,
    )
    u_quad_asm = evaluate_at_xq(
        seq_asm.e0_dbc_T @ u_hat_asm, ci_asm, cs_asm, quad_shape_asm, 1
    )
    solve_err_asm = l2_relative_error(seq_asm, u_quad_asm, u_exact_asm)

    k_asm_u_proj = stiffness_matvec(u_proj_prod)
    rel_consistency_asm = float(
        jnp.linalg.norm(k_asm_u_proj - rhs_prod) / jnp.linalg.norm(rhs_prod)
    )
    rel_operator_gap = float(
        jnp.linalg.norm(k_prod_u_proj - k_asm_u_proj) /
        jnp.linalg.norm(k_asm_u_proj)
    )

    prod_gap = solve_err_prod / proj_err if proj_err > 0 else float("nan")
    asm_gap = solve_err_asm / proj_err if proj_err > 0 else float("nan")
    return (
        solve_err_prod,
        solve_err_asm,
        proj_err,
        prod_gap,
        asm_gap,
        rel_consistency_prod,
        rel_consistency_asm,
        rel_operator_gap,
    )


P_S = 3
N_LIST_S = (6, 8, 10)

print(
    f"[S] p={P_S} production-vs-assembled k0 comparison (q = 2p, minimal n sweep)"
)
print(
    f"[S] batch sizes: inner={mrx.MAP_BATCH_SIZE_INNER}, outer={mrx.MAP_BATCH_SIZE_OUTER}")
print(
    f"    {'n':>4}  {'solve_prod':>12}  {'solve_asm':>12}  {'proj_floor':>12}  {'prod/proj':>10}  {'asm/proj':>10}  {'cons_prod':>11}  {'cons_asm':>11}  {'op_gap':>11}"
)
for n_loc in N_LIST_S:
    t0_s = time.perf_counter()
    _progress_s(f"[S] starting n={n_loc}")
    (
        solve_err_prod_s,
        solve_err_asm_s,
        proj_err_s,
        prod_gap_s,
        asm_gap_s,
        rel_consistency_prod_s,
        rel_consistency_asm_s,
        rel_operator_gap_s,
    ) = _compare_k0_realizations_at_n(P_S, n_loc, q_loc=2 * P_S)
    print(
        f"    {n_loc:>4d}  {solve_err_prod_s:>12.3e}  {solve_err_asm_s:>12.3e}  {proj_err_s:>12.3e}  {prod_gap_s:>10.4f}  {asm_gap_s:>10.4f}  {rel_consistency_prod_s:>11.3e}  {rel_consistency_asm_s:>11.3e}  {rel_operator_gap_s:>11.3e}",
        flush=True,
    )
    _progress_s(f"[S] finished n={n_loc} in {time.perf_counter() - t0_s:.1f}s")

# %%
