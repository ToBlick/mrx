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
    ns, ps, q, TYPES, polar=True, tol=1e-12, maxiter=200_000
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
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-13, maxiter=200_000
    )
    seq_g.set_map(toroid_map(epsilon=EPSILON))
    seq_g.evaluate_1d()
    seq_g.assemble_mass_matrix(0)
    ci_g, cs_g = seq_g._form_comp_info(0)
    quad_shape_g = (seq_g.quad.ny, seq_g.quad.nx, seq_g.quad.nz)
    u_exact_g = exact_u_at_quad(seq_g)
    u_load_g = seq_g.p0_dbc(u)
    u_proj_g = seq_g.apply_inverse_mass_matrix(u_load_g, 0, dirichlet=True)
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
        ns_loc, ps_loc, qx, TYPES, polar=True, tol=1e-12, maxiter=200_000
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
P_LIST_I = (1, 2, 3, 4)
N_LIST_I = (8, 12, 16)

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

# %%
