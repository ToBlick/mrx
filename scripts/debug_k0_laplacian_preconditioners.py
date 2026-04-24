"""Debug non-shifted k=0 Laplace solves with several preconditioners.

This script mirrors the geometry/setup used by the nullspace inverse-iteration
debug script, but only assembles what is needed for k=0 solves. It then
computes only the k=0 nullspace (constant mode for the no-DBC case) and
benchmarks non-shifted solves with a few preconditioners:

- Jacobi
- several damped Jacobi smoothing polynomials
- the FD/Kronecker scalar Hodge preconditioner

Both boundary-condition choices are covered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.io import project_sampled_field
from mrx.mappings import toroid_map
from mrx.nullspace import (_bootstrap_nullspace_guesses, _commit,
                           _initial_guesses, _set_null, find_nullspace_vectors,
                           get_nullspace, init_nullspaces)
from mrx.operators import (_hodge_diaginv, apply_hodge_kron_preconditioner,
                           apply_mass_matrix, apply_stiffness,
                           assemble_fd_hodge_preconditioner,
                           assemble_hodge_operators,
                           assemble_incidence_operators,
                           assemble_kron_mass_preconditioner,
                           assemble_mass_operators)
from mrx.solvers import solve_singular_cg

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N = 7
P = 3
SAMPLE_POINTS = 50
AUTO_SAMPLE_POINTS = True
TOL = 1e-10
MAXITER = 1000
BETTI = (1, 1, 0, 0)

EPS_NULL = TOL**0.5 / 10
ABS_TOL_NULL = None

NUM_RHS = 8
N_MODES = 6
RHS_SEED = 7

JACOBI_SWEEPS = (2, 4, 8)
POWER_ITERS = 20
JACOBI_SAFETY = 1.8

TORUS_EPSILON = 1.0 / 3.0
TORUS_R0 = 1.0

CHECK_POSITIVE_JACOBIAN = True
JACOBIAN_POS_TOL = 0.0


@dataclass
class Stats:
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_rel_res: float
    max_rel_res: float
    avg_time_ms: float
    std_time_ms: float


def _build_sequence(args):
    ns = (args.n, args.n, args.n)
    ps = (args.p, args.p, args.p)
    seq = DeRhamSequence(
        ns,
        ps,
        2 * args.p,
        ("clamped", "periodic", "periodic"),
        lambda x: x,
        polar=True,
        tol=args.tol,
        maxiter=args.maxiter,
        betti_numbers=args.betti,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()

    F_ana = toroid_map(epsilon=args.torus_epsilon, R0=args.torus_r0)
    n_sample = args.sample_points_effective
    r = jnp.linspace(0.0, 1.0, n_sample)
    theta = jnp.linspace(0.0, 1.0, n_sample)
    zeta = jnp.linspace(0.0, 1.0, n_sample)
    ri, thetai, zetai = jnp.meshgrid(r, theta, zeta, indexing="ij")
    pts = jnp.stack([ri.ravel(), thetai.ravel(), zetai.ravel()], axis=1)
    samples = jax.vmap(F_ana)(pts)
    coeffs = jnp.stack(
        [
            project_sampled_field(
                (r, theta, zeta),
                samples[:, i],
                seq,
                k=0,
                dirichlet=False,
                reference_domain=True,
            )
            for i in range(3)
        ],
        axis=0,
    )
    seq.set_spline_map(coeffs)

    if args.check_positive_jacobian:
        _check_positive_jacobian(seq, tol=args.jacobian_pos_tol)

    operators = None
    operators = assemble_mass_operators(
        seq, seq.geometry, operators=operators, ks=(0, 1))
    operators = assemble_kron_mass_preconditioner(seq, operators=operators)
    operators = assemble_fd_hodge_preconditioner(seq, operators=operators)
    operators = assemble_incidence_operators(seq, operators=operators, ks=(0,))
    operators = assemble_hodge_operators(
        seq, seq.geometry, operators=operators, ks=(0,))
    operators = _commit(seq, init_nullspaces(seq, operators))
    return seq, operators


def _check_positive_jacobian(seq, tol=0.0):
    jac = seq.jacobian_j
    if jac is None:
        raise ValueError("Jacobian array is missing on sequence geometry")

    jac = jnp.asarray(jac).reshape(-1)
    finite_mask = jnp.isfinite(jac)
    n_nonfinite = int(jac.size - int(jnp.sum(finite_mask)))
    if n_nonfinite > 0:
        bad_idx = int(jnp.argmax(~finite_mask))
        bad_x = seq.quad.x[bad_idx]
        raise ValueError(
            "Geometry sanity check failed: non-finite Jacobian determinant at "
            f"quadrature index {bad_idx}, x={bad_x.tolist()}"
        )

    min_j = float(jnp.min(jac))
    max_j = float(jnp.max(jac))
    n_nonpositive = int(jnp.sum(jac <= tol))
    print(
        "jacobian_check: "
        f"min={min_j:.6e}, max={max_j:.6e}, "
        f"nonpositive_count={n_nonpositive}/{jac.size}, tol={tol:.3e}"
    )
    if n_nonpositive > 0:
        bad_idx = int(jnp.argmin(jac))
        bad_x = seq.quad.x[bad_idx]
        bad_val = float(jac[bad_idx])
        raise ValueError(
            "Geometry sanity check failed: Jacobian determinant is non-positive "
            f"at quadrature index {bad_idx}, x={bad_x.tolist()}, detJ={bad_val:.6e}"
        )


def _compute_k0_nullspace(seq, operators, eps, abs_tol):
    guesses = _initial_guesses(seq, operators, 0, False, 1)
    operators = _bootstrap_nullspace_guesses(seq, operators, 0, False, guesses)
    vs, info = find_nullspace_vectors(
        seq,
        operators,
        0,
        1,
        eps,
        dirichlet=False,
        x0s=guesses,
        abs_tol=abs_tol,
    )
    operators = _commit(seq, _set_null(operators, 0, False, vs))
    return operators, info[0]


def _smooth_scalar_rhs_batch(seq, key, dirichlet, num_rhs, n_modes):
    proj = seq.p0_dbc if dirichlet else seq.p0
    mr = jnp.arange(1, n_modes + 1, dtype=jnp.float64)
    mt = jnp.arange(n_modes, dtype=jnp.float64)
    mz = jnp.arange(n_modes, dtype=jnp.float64)
    shape = (num_rhs, n_modes, n_modes, n_modes, 4)
    all_coeffs = jax.random.normal(key, shape)

    def make_f(c):
        def f(x):
            r, th, ze = x[0], x[1], x[2]
            br = jnp.sin(jnp.pi * mr * r)
            bt = jnp.stack([
                jnp.cos(2 * jnp.pi * mt * th),
                jnp.sin(2 * jnp.pi * mt * th),
            ], axis=0)
            bz = jnp.stack([
                jnp.cos(2 * jnp.pi * mz * ze),
                jnp.sin(2 * jnp.pi * mz * ze),
            ], axis=0)
            cc = jnp.einsum('ijl,i,j,l->', c[..., 0], br, bt[0], bz[0])
            cs = jnp.einsum('ijl,i,j,l->', c[..., 1], br, bt[0], bz[1])
            sc = jnp.einsum('ijl,i,j,l->', c[..., 2], br, bt[1], bz[0])
            ss = jnp.einsum('ijl,i,j,l->', c[..., 3], br, bt[1], bz[1])
            return cc + cs + sc + ss

        return f

    rhs_list = [proj(make_f(all_coeffs[i])) for i in range(num_rhs)]
    return jnp.stack(rhs_list)


def _project_rhs_to_range(seq, operators, b, dirichlet):
    vs = get_nullspace(operators, 0, dirichlet)
    if vs.shape[0] == 0:
        return b

    out = b
    for v in vs:
        out = out - jnp.dot(v, out) * apply_mass_matrix(
            seq, operators, v, 0, dirichlet=dirichlet
        )
    return out


def _project_rhs_batch_to_range(seq, operators, rhs_batch, dirichlet):
    def _project_one(b):
        return _project_rhs_to_range(seq, operators, b, dirichlet)

    return jax.vmap(_project_one)(rhs_batch)


def _relative_residual(seq, operators, u, b, dirichlet):
    b_proj = _project_rhs_to_range(seq, operators, b, dirichlet)
    residual_dual = apply_stiffness(
        seq, operators, u, 0, dirichlet=dirichlet) - b_proj
    residual_primal = seq.apply_inverse_mass_matrix(
        residual_dual, 0, dirichlet=dirichlet, operators=operators
    )
    b_primal = seq.apply_inverse_mass_matrix(
        b_proj, 0, dirichlet=dirichlet, operators=operators
    )
    num = jnp.sqrt(jnp.abs(jnp.dot(residual_dual, residual_primal)))
    den = jnp.sqrt(jnp.abs(jnp.dot(b_proj, b_primal)))
    return float(num / jnp.where(den > 0, den, 1.0))


def _estimate_scaled_lambda_max(seq, operators, dirichlet, n_iter):
    dinv = _hodge_diaginv(operators, 0, dirichlet)
    dsqrtinv = jnp.sqrt(dinv)
    n = seq.n0_dbc if dirichlet else seq.n0

    def H_mv(y):
        return dsqrtinv * apply_stiffness(
            seq, operators, dsqrtinv * y, 0, dirichlet=dirichlet
        )

    v = jnp.sin(jnp.arange(1, n + 1, dtype=jnp.float64))
    vs = get_nullspace(operators, 0, dirichlet)
    if vs.shape[0] > 0:
        z = vs[0]
        y_null = z / jnp.where(dsqrtinv > 0, dsqrtinv, 1.0)
        y_null_norm_sq = y_null @ y_null
        if float(y_null_norm_sq) > 0.0:
            v = v - (y_null @ v) / y_null_norm_sq * y_null
    v = v / jnp.linalg.norm(v)
    for _ in range(n_iter):
        w = H_mv(v)
        wn = jnp.linalg.norm(w)
        safe = jnp.where(wn > 0, wn, 1.0)
        v = w / safe
    lam = float(v @ H_mv(v))
    return max(lam, 1.0)


def _make_polynomial_jacobi_precond(seq, operators, dirichlet, n_sweeps, omega):
    dinv = _hodge_diaginv(operators, 0, dirichlet)
    dsqrtinv = jnp.sqrt(dinv)

    def H_mv(y):
        return dsqrtinv * apply_stiffness(
            seq, operators, dsqrtinv * y, 0, dirichlet=dirichlet
        )

    def precond(b):
        rhs_scaled = dsqrtinv * b

        def body(_, state):
            y, r = state
            y = y + omega * r
            r = rhs_scaled - H_mv(y)
            return y, r

        y0 = jnp.zeros_like(rhs_scaled)
        r0 = rhs_scaled
        y, _ = jax.lax.fori_loop(0, n_sweeps, body, (y0, r0))
        return dsqrtinv * y

    return precond


def _make_solver(seq, operators, dirichlet, precond_mv):
    def A_mv(x):
        return apply_stiffness(seq, operators, x, 0, dirichlet=dirichlet)

    def M_mv(x):
        return apply_mass_matrix(seq, operators, x, 0, dirichlet=dirichlet)

    vs = get_nullspace(operators, 0, dirichlet)

    @jax.jit
    def solve(b):
        b_proj = _project_rhs_to_range(seq, operators, b, dirichlet)
        u, info = solve_singular_cg(
            A_mv,
            b_proj,
            mass_matvec=M_mv,
            precond_matvec=precond_mv,
            vs=vs,
            tol=seq.tol,
            maxiter=seq.maxiter,
        )
        return u, jnp.abs(info)

    return solve


def _time_solve(seq, operators, dirichlet, solve, rhs_batch):
    u0, _ = solve(rhs_batch[0])
    jax.block_until_ready(u0)

    iters = []
    rel_res = []
    times = []
    for b in rhs_batch:
        t0 = time.perf_counter()
        u, it = solve(b)
        jax.block_until_ready(u)
        dt = time.perf_counter() - t0
        iters.append(int(it))
        rel_res.append(_relative_residual(seq, operators, u, b, dirichlet))
        times.append(dt * 1e3)

    iters = jnp.array(iters)
    rel_res = jnp.array(rel_res)
    times = jnp.array(times)
    return Stats(
        avg_iters=float(iters.mean()),
        std_iters=float(iters.std()),
        max_iters=int(iters.max()),
        avg_rel_res=float(rel_res.mean()),
        max_rel_res=float(rel_res.max()),
        avg_time_ms=float(times.mean()),
        std_time_ms=float(times.std()),
    )


def _run_case(args, seq, operators, dirichlet):
    n = seq.n0_dbc if dirichlet else seq.n0
    key = jax.random.PRNGKey(args.rhs_seed + int(dirichlet))
    rhs_batch = _smooth_scalar_rhs_batch(
        seq, key, dirichlet, args.num_rhs, args.n_modes
    )
    rhs_batch = _project_rhs_batch_to_range(
        seq, operators, rhs_batch, dirichlet)

    jacobi_dinv = _hodge_diaginv(operators, 0, dirichlet)

    def jacobi_precond(x):
        return jacobi_dinv * x

    lam_max = _estimate_scaled_lambda_max(
        seq, operators, dirichlet, args.power_iters
    )
    omega = args.jacobi_safety / lam_max

    variants = [("jacobi", jacobi_precond)]
    for n_sweeps in args.jacobi_sweeps:
        variants.append(
            (
                f"jacobi-x{n_sweeps}",
                _make_polynomial_jacobi_precond(
                    seq, operators, dirichlet, n_sweeps, omega
                ),
            )
        )
    variants.append(
        (
            "kron-fd",
            lambda x: apply_hodge_kron_preconditioner(
                seq, operators, x, 0, dirichlet=dirichlet
            ),
        )
    )

    print("-" * 96)
    print(
        f"target: k=0, dirichlet={dirichlet}, n_dof={n}, num_rhs={args.num_rhs}, "
        f"lambda_max_scaled≈{lam_max:.3e}, omega={omega:.3e}"
    )
    print(
        f"{'variant':>12s} {'avg_it':>8s} {'std_it':>8s} {'max_it':>7s} "
        f"{'avg_rel_res':>14s} {'max_rel_res':>14s} {'avg_ms':>9s} {'std_ms':>8s}"
    )

    for name, precond in variants:
        solve = _make_solver(seq, operators, dirichlet, precond)
        stats = _time_solve(seq, operators, dirichlet, solve, rhs_batch)
        print(
            f"{name:>12s} {stats.avg_iters:8.1f} {stats.std_iters:8.2f} "
            f"{stats.max_iters:7d} {stats.avg_rel_res:14.6e} {stats.max_rel_res:14.6e} "
            f"{stats.avg_time_ms:9.2f} {stats.std_time_ms:8.2f}"
        )


def _build_config():
    sample_points_effective = SAMPLE_POINTS
    if AUTO_SAMPLE_POINTS:
        sample_points_effective = max(SAMPLE_POINTS, 6 * N + 2 * P)
    abs_tol_null = TOL if ABS_TOL_NULL is None else ABS_TOL_NULL
    return SimpleNamespace(
        n=N,
        p=P,
        tol=TOL,
        maxiter=MAXITER,
        betti=BETTI,
        sample_points_effective=sample_points_effective,
        torus_epsilon=TORUS_EPSILON,
        torus_r0=TORUS_R0,
        check_positive_jacobian=CHECK_POSITIVE_JACOBIAN,
        jacobian_pos_tol=JACOBIAN_POS_TOL,
        eps_null=EPS_NULL,
        abs_tol_null=abs_tol_null,
        num_rhs=NUM_RHS,
        n_modes=N_MODES,
        rhs_seed=RHS_SEED,
        jacobi_sweeps=JACOBI_SWEEPS,
        power_iters=POWER_ITERS,
        jacobi_safety=JACOBI_SAFETY,
    )


def main():
    args = _build_config()

    t0 = time.perf_counter()
    print("Assembling k=0 sequence and operators...")
    seq, operators = _build_sequence(args)
    print(f"assembly_time_s={time.perf_counter() - t0:.3f}")

    print()
    print("Computing k=0 nullspace only...")
    t0 = time.perf_counter()
    operators, (n_iters, residual) = _compute_k0_nullspace(
        seq, operators, args.eps_null, args.abs_tol_null
    )
    print(
        f"nullspace_k0_no_dbc: iters={n_iters} residual={residual:.6e} "
        f"time_s={time.perf_counter() - t0:.3f}"
    )

    print()
    print("=" * 96)
    print("Non-Shifted k=0 Laplace Preconditioner Comparison")
    print("=" * 96)
    _run_case(args, seq, operators, False)
    print()
    _run_case(args, seq, operators, True)


if __name__ == "__main__":
    main()
