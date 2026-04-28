"""Debug inverse-power nullspace computation with iteration diagnostics.

This script assembles a baseline DeRham sequence and then runs the same
shift-and-invert loop used by nullspace construction, while printing detailed
per-iteration diagnostics:

- inverse solve wall time
- inner solver iteration count (CG/MINRES info)
- Hodge-Laplacian residual ||L_k v||
- residual decrease and stop reason

Run from repository root, for example:

    .venv/bin/python scripts/debug_nullspace_inverse_iteration.py --n 5 --p 2
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.io import project_sampled_field
from mrx.mappings import toroid_map
from mrx.nullspace import (_bootstrap_nullspace_guesses, _commit,
                           _initial_guesses, _overwrite_nullspace_vector,
                           get_nullspace)
from mrx.operators import (_hodge_diaginv, _kron_available, _mass_diaginv,
                           apply_hodge_kron_preconditioner,
                           apply_inverse_shifted_hodge_laplacian,
                           apply_mass_kron_preconditioner, apply_mass_matrix,
                           apply_projection_matrix, assemble_all_operators,
                           dense_mass_matrix, dense_projection_matrix)
from mrx.solvers import solve_saddle_point_minres

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Configuration (edit these constants directly)
# ---------------------------------------------------------------------------

N = 5
P = 3
SAMPLE_POINTS = 50
AUTO_SAMPLE_POINTS = True
TOL = 1e-10
MAXITER = 1000
INNER_MAXITER = MAXITER
BETTI = (1, 1, 0, 0)

ALL_K = True
K = 3
DIRICHLET = True
N_VECTORS = None

EPS = TOL**0.5 / 10
ABS_TOL = None
PRECOND_KIND = "auto"
SEED = 0
PRINT_EVERY = 1

TORUS_EPSILON = 1.0 / 3.0
TORUS_R0 = 1.0

CHECK_POSITIVE_JACOBIAN = True
JACOBIAN_POS_TOL = 0.0

RUN_POST_NULLSPACE_SHIFTED_EXACT_LIFT_COMPARE = True
POST_EXACT_LIFT_COMPARE_SEED = 23
POST_EXACT_LIFT_COMPARE_MAXITER = MAXITER
POST_EXACT_LIFT_HYBRID_ALPHAS = (0.001, 0.01, 0.1, 1.0)

RUN_SHIFTED_K0_COMPARE = True
K0_COMPARE_NUM_RANDOM = 5
K0_COMPARE_SEED = 31
K0_COMPARE_MAXITER = MAXITER


def _n_vectors(betti_numbers, k, dirichlet):
    b0, b1, b2, _b3 = betti_numbers
    if dirichlet:
        return (0, b2, b1, b0)[k]
    return (b0, b1, b2, 0)[k]


def _validate_betti(betti):
    if len(betti) != 4:
        raise ValueError("BETTI must have length 4, e.g. (1, 1, 0, 0)")
    if betti[0] != 1:
        raise ValueError("betti[0] must be 1")
    if betti[3] != 0:
        raise ValueError("betti[3] must be 0")


def _build_sequence(args):
    ns = (args.n, args.n, args.n)
    ps = (args.p, args.p, args.p)
    seq = DeRhamSequence(
        ns,
        ps,
        2 * args.p,
        ("clamped", "periodic", "periodic"),
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
    # seq.set_map(toroid_map(epsilon=args.torus_epsilon, R0=args.torus_r0))

    if args.check_positive_jacobian:
        _check_positive_jacobian(seq, tol=args.jacobian_pos_tol)

    operators = assemble_all_operators(seq, seq.geometry)
    seq.operators = operators
    return seq, operators


def _check_positive_jacobian(seq, tol=0.0):
    jac = seq.jacobian_j
    if jac is None:
        raise ValueError("Jacobian array is missing on sequence geometry")

    jac = jnp.asarray(jac)
    if jac.ndim != 1:
        jac = jac.reshape(-1)

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
            f"at quadrature index {bad_idx}, x={bad_x.tolist()}, "
            f"detJ={bad_val:.6e}"
        )


def _seed_debug_nullspaces(seq, operators):
    """Seed any trivial nullspaces needed by standalone debug checks."""
    z0 = jnp.ones(seq.n0)
    z0 = z0 / seq.l2_norm(z0, 0, dirichlet=False)
    operators = _bootstrap_nullspace_guesses(seq, operators, 0, False, [z0])
    return _commit(seq, operators)


def _mass_orthogonalise(seq, operators, v, basis, k, dirichlet):
    out = v
    for u in basis:
        coeff = u @ seq.apply_mass_matrix(
            out, k, dirichlet=dirichlet, operators=operators
        )
        out = out - coeff * u
    return out


def _residual_norm(seq, operators, v, k, dirichlet):
    Lv = seq.apply_hodge_laplacian(
        v, k, dirichlet=dirichlet, operators=operators)
    return float(seq.l2_norm(Lv, k, dirichlet=dirichlet))


def _shifted_residual_norm(seq, operators, u, rhs, k, dirichlet, eps):
    Au = seq.apply_hodge_laplacian(
        u, k, dirichlet=dirichlet, operators=operators)
    Au = Au + eps * seq.apply_mass_matrix(
        u, k, dirichlet=dirichlet, operators=operators)
    return float(seq.l2_norm(Au - rhs, k, dirichlet=dirichlet))


def _nullspace_projection_orthogonal_rhs(seq, operators, rhs, z):
    mz = apply_mass_matrix(seq, operators, z, 3, dirichlet=True)
    return rhs - (z @ rhs) * mz


def _dense_exact_k3_lift(seq, operators):
    mass3 = dense_mass_matrix(seq, operators, 3, dirichlet=True)
    p30 = dense_projection_matrix(
        seq, operators, 3, 0, dirichlet_in=False, dirichlet_out=True)
    return jnp.linalg.solve(mass3, p30)


def _wrap_harmonic_coarse_debug(seq, operators, base_precond, eps, k, dirichlet):
    z = get_nullspace(operators, k, dirichlet)[0]
    z = z / seq.l2_norm(z, k, dirichlet=dirichlet)
    mz = apply_mass_matrix(seq, operators, z, k, dirichlet=dirichlet)

    def precond(x):
        alpha = z @ x
        x_perp = x - alpha * mz
        y_perp = base_precond(x_perp)
        beta = z @ apply_mass_matrix(
            seq, operators, y_perp, k, dirichlet=dirichlet)
        return y_perp - beta * z + (alpha / eps) * z

    return precond


def _k3_exact_lift_precond(seq, operators, exact_lift_30, eps, alpha=1.0):
    stiffness_diaginv = _hodge_diaginv(operators, 3, dirichlet=True)
    mass_diaginv_3 = _mass_diaginv(operators, 3, dirichlet=True)
    shifted_diaginv = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv_3)
    exact_lift_03 = exact_lift_30.T

    def base(v):
        smooth = shifted_diaginv * v
        w = exact_lift_03 @ v
        w = apply_hodge_kron_preconditioner(
            seq, operators, w, 0, dirichlet=False, eps=eps)
        w = exact_lift_30 @ w
        return smooth + alpha * w

    return _wrap_harmonic_coarse_debug(
        seq, operators, base, eps, 3, True)


def _solve_shifted_k3_with_upper_precond(seq, operators, rhs, eps, precond_upper,
                                         maxiter):
    u, _, info = solve_saddle_point_minres(
        stiffness_matvec=lambda x: eps * apply_mass_matrix(
            seq, operators, x, 3, dirichlet=True),
        derivative_matvec=lambda s: seq.apply_derivative_matrix(
            s, 2, dirichlet_in=True, dirichlet_out=True, operators=operators),
        derivative_T_matvec=lambda u: seq.apply_derivative_matrix(
            u, 2, dirichlet_in=True, dirichlet_out=True,
            transpose=True, operators=operators),
        mass_lower_matvec=lambda s: apply_mass_matrix(
            seq, operators, s, 2, dirichlet=True),
        b_upper=rhs,
        n_upper=seq.n3_dbc,
        n_lower=seq.n2_dbc,
        precond_upper=precond_upper,
        precond_lower=lambda x: apply_mass_kron_preconditioner(
            seq, operators, x, 2, dirichlet=True),
        mass_upper_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, 3, dirichlet=True),
        tol=seq.tol,
        maxiter=maxiter,
    )
    return u, info


def _run_post_nullspace_shifted_exact_lift_compare(args, seq):
    if not args.run_post_nullspace_shifted_exact_lift_compare:
        return

    operators = seq._require_operators()
    z = get_nullspace(operators, 3, True)[0]
    z = z / seq.l2_norm(z, 3, dirichlet=True)
    mz = apply_mass_matrix(seq, operators, z, 3, dirichlet=True)
    exact_lift_30 = _dense_exact_k3_lift(seq, operators)
    key = jax.random.PRNGKey(args.post_exact_lift_compare_seed)
    raw = jax.random.normal(key, (seq.n3_dbc,))
    rhs_perp = _nullspace_projection_orthogonal_rhs(seq, operators, raw, z)
    rhs_harmonic = args.eps * mz
    rhs_mixed = rhs_perp + rhs_harmonic

    cases = [
        ("perp_random", rhs_perp),
        ("harmonic_only", rhs_harmonic),
        ("mixed", rhs_mixed),
    ]

    print()
    print("=" * 88)
    print("Post-Nullspace Shifted Solve Comparison (Exact Lift)")
    print("=" * 88)
    print(
        f"target: k=3, dirichlet=True, eps={args.eps:.3e}, "
        f"solve_maxiter={args.post_exact_lift_compare_maxiter}, lower=kronecker"
    )
    print(
        f"{'case':>14s} {'variant':>24s} {'iters':>8s} {'residual':>14s} {'solve_s':>10s}"
    )

    variants = [
        (
            "jacobi+coarse",
            lambda b: apply_inverse_shifted_hodge_laplacian(
                seq, operators, b, 3, args.eps,
                dirichlet=True,
                preconditioner='jacobi',
                use_harmonic_coarse=True,
                tol=seq.tol,
                maxiter=args.post_exact_lift_compare_maxiter,
                return_info=True,
            ),
        ),
        (
            "tensor+coarse",
            lambda b: apply_inverse_shifted_hodge_laplacian(
                seq, operators, b, 3, args.eps,
                dirichlet=True,
                preconditioner='tensor',
                use_harmonic_coarse=True,
                tol=seq.tol,
                maxiter=args.post_exact_lift_compare_maxiter,
                return_info=True,
            ),
        ),
    ]
    for alpha in args.post_exact_lift_hybrid_alphas:
        precond_hybrid = _k3_exact_lift_precond(
            seq, operators, exact_lift_30, args.eps, alpha=alpha)
        variants.append(
            (
                f"hybrid-exact-{alpha:.2f}",
                lambda b, precond=precond_hybrid: _solve_shifted_k3_with_upper_precond(
                    seq, operators, b, args.eps, precond,
                    args.post_exact_lift_compare_maxiter),
            )
        )

    for case_name, rhs in cases:
        for variant_name, runner in variants:
            t0 = time.perf_counter()
            u, info = runner(rhs)
            dt = time.perf_counter() - t0
            res = _shifted_residual_norm(
                seq, operators, u, rhs, 3, True, args.eps)
            print(
                f"{case_name:>14s} {variant_name:>24s} {int(abs(float(info))):8d} "
                f"{res:14.6e} {dt:10.3f}"
            )


def _run_shifted_k0_compare(args, seq):
    if not args.run_shifted_k0_compare:
        return

    operators = seq._require_operators()
    key = jax.random.PRNGKey(args.k0_compare_seed)
    z = get_nullspace(operators, 0, False)[0]
    z = z / seq.l2_norm(z, 0, dirichlet=False)
    mz = apply_mass_matrix(seq, operators, z, 0, dirichlet=False)
    raw = jax.random.normal(key, (seq.n0,))
    rhs_perp = raw - (z @ raw) * mz
    rhs_harmonic = args.eps * mz
    rhs_mixed = rhs_perp + rhs_harmonic

    cases = [
        ("perp_random", rhs_perp),
        ("harmonic_only", rhs_harmonic),
        ("mixed", rhs_mixed),
    ]

    print()
    print("=" * 88)
    print("Shifted k=0 Preconditioner Comparison")
    print("=" * 88)
    print(
        f"target: k=0, dirichlet=False, eps={args.eps:.3e}, "
        f"solve_maxiter={args.k0_compare_maxiter}"
    )
    print(
        f"{'case':>14s} {'variant':>10s} {'iters':>8s} {'residual':>14s} {'solve_s':>10s}"
    )

    for case_name, rhs in cases:
        for variant_name, precond_kind, use_harmonic_coarse in (
            ("jacobi", "jacobi", False),
            ("jacobi+coarse", "jacobi", True),
            ("tensor+coarse", "tensor", None),
        ):
            t0 = time.perf_counter()
            u, info = apply_inverse_shifted_hodge_laplacian(
                seq,
                operators,
                rhs,
                0,
                args.eps,
                dirichlet=False,
                preconditioner=precond_kind,
                use_harmonic_coarse=use_harmonic_coarse,
                tol=seq.tol,
                maxiter=args.k0_compare_maxiter,
                return_info=True,
            )
            dt = time.perf_counter() - t0
            res = _shifted_residual_norm(
                seq, operators, u, rhs, 0, False, args.eps)
            print(
                f"{case_name:>14s} {variant_name:>10s} {int(abs(float(info))):8d} "
                f"{res:14.6e} {dt:10.3f}"
            )


def _run_case(args, seq, operators, k, dirichlet, n_vectors_override=None):
    n_dof = getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")
    n_vectors = n_vectors_override
    if n_vectors is None:
        n_vectors = _n_vectors(args.betti, k, dirichlet)
    abs_tol = seq.tol if args.abs_tol is None else args.abs_tol

    print("-" * 88)
    print(
        f"target: k={k}, dirichlet={dirichlet}, n_dof={n_dof}, "
        f"n_vectors={n_vectors}, eps={args.eps:.3e}, abs_tol={abs_tol:.3e}"
    )
    if k >= 1:
        lower_kron_active = (
            args.precond_kind != "jacobi"
            and _kron_available(seq, operators, k - 1)
        )
        upper_label = args.precond_kind
        if args.eps > 0:
            if args.precond_kind == "tensor" and k == 3 and dirichlet:
                upper_label = "shifted-tensor+coarse"
            elif args.precond_kind == "tensor" and k == 0:
                upper_label = "shifted-kron"
            else:
                upper_label = "shifted-jacobi"
        print(
            "minres_precond: "
            f"upper={upper_label}, "
            f"lower={'kronecker' if lower_kron_active else 'jacobi'}"
        )

    if n_vectors == 0:
        print("skip: no harmonic vectors requested for this (k, dirichlet) pair")
        return []

    found = []
    summaries = []
    guesses = _initial_guesses(seq, operators, k, dirichlet, n_vectors)
    operators = _bootstrap_nullspace_guesses(
        seq, operators, k, dirichlet, guesses)
    key = jax.random.PRNGKey(args.seed)

    for idx in range(n_vectors):
        print()
        print(f"--- vector {idx + 1}/{n_vectors} ---")
        if guesses[idx] is not None:
            v = guesses[idx]
            init_source = "analytic"
        else:
            key, sub = jax.random.split(key)
            v = jax.random.normal(sub, (n_dof,))
            init_source = "random"

        v = _mass_orthogonalise(seq, operators, v, found, k, dirichlet)
        v = v / seq.l2_norm(v, k, dirichlet=dirichlet)
        res0 = _residual_norm(seq, operators, v, k, dirichlet)
        print(f"init={init_source:8s} iter=0 residual={res0:.6e}")

        if res0 <= abs_tol:
            operators = _commit(seq, _overwrite_nullspace_vector(
                operators, k, dirichlet, idx, v))
            print(f"status=accepted_initial residual={res0:.6e}")
            found.append(v)
            summaries.append(
                {
                    "index": idx,
                    "status": "accepted_initial",
                    "iters": 0,
                    "residual": res0,
                    "time_s": 0.0,
                    "inner_iters": 0,
                }
            )
            continue

        iter_count = 0
        res_prev = res0
        total_t = 0.0
        total_inner_iters = 0
        status = "maxiter"

        for it in range(1, args.maxiter + 1):
            t0 = time.perf_counter()
            Mv = seq.apply_mass_matrix(
                v, k, dirichlet=dirichlet, operators=operators
            )
            w, solve_info = apply_inverse_shifted_hodge_laplacian(
                seq,
                operators,
                Mv,
                k,
                args.eps,
                dirichlet=dirichlet,
                guess=v,
                preconditioner=args.precond_kind,
                tol=seq.tol,
                maxiter=args.inner_maxiter,
                return_info=True,
            )
            dt = time.perf_counter() - t0
            total_t += dt

            w = _mass_orthogonalise(seq, operators, w, found, k, dirichlet)
            v = w / seq.l2_norm(w, k, dirichlet=dirichlet)

            res = _residual_norm(seq, operators, v, k, dirichlet)
            delta = abs(res - res_prev)
            inner_iters = int(abs(float(solve_info)))
            total_inner_iters += inner_iters
            iter_count = it

            if (it % args.print_every == 0) or (res <= abs_tol):
                print(
                    f"iter={it:4d} residual={res:.6e} delta={delta:.3e} "
                    f"inner_info={int(solve_info):6d} inner_iters={inner_iters:4d} "
                    f"solve_ms={dt * 1e3:8.2f}"
                )

            if res <= abs_tol:
                status = "converged"
                break
            if delta <= abs_tol:
                status = "stalled"
                break
            res_prev = res

        print(
            f"status={status} iters={iter_count} residual={res:.6e} "
            f"total_solve_s={total_t:.3f} total_inner_iters={total_inner_iters}"
        )

        operators = _commit(seq, _overwrite_nullspace_vector(
            operators, k, dirichlet, idx, v))
        found.append(v)
        summaries.append(
            {
                "index": idx,
                "status": status,
                "iters": iter_count,
                "residual": res,
                "time_s": total_t,
                "inner_iters": total_inner_iters,
            }
        )

    return summaries


def _run_debug(args, seq, operators):
    print("=" * 88)
    print("Nullspace inverse-power debug")
    print("=" * 88)
    print(
        f"resolution n={args.n}, p={args.p}, "
        f"sample_points={args.sample_points_effective} "
        f"(base={args.sample_points}, auto={args.auto_sample_points}), "
        f"tol={seq.tol:.3e}, maxiter={seq.maxiter}, "
        f"inner_maxiter={args.inner_maxiter}"
    )
    print(
        f"eps={args.eps:.3e}, precond_kind={args.precond_kind}, "
        f"print_every={args.print_every}"
    )

    all_summaries = []
    if args.all_k:
        for k in (0, 1, 2, 3):
            for dirichlet in (False, True):
                summaries = _run_case(args, seq, operators, k, dirichlet)
                all_summaries.extend(
                    {
                        "k": k,
                        "dirichlet": dirichlet,
                        **s,
                    }
                    for s in summaries
                )
    else:
        summaries = _run_case(
            args, seq, operators, args.k, args.dirichlet, args.n_vectors
        )
        all_summaries.extend(
            {
                "k": args.k,
                "dirichlet": args.dirichlet,
                **s,
            }
            for s in summaries
        )

    print()
    print("=" * 88)
    print("Global Summary")
    print("=" * 88)
    print(
        f"{'k':>2s} {'dbc':>5s} {'vec':>4s} {'status':>16s} {'iters':>8s} "
        f"{'residual':>14s} "
        f"{'inner_iters':>12s} {'solve_s':>10s}"
    )
    for s in all_summaries:
        print(
            f"{s['k']:2d} {str(s['dirichlet']):>5s} {s['index']:4d} "
            f"{s['status']:>16s} {s['iters']:8d} "
            f"{s['residual']:14.6e} {s['inner_iters']:12d} {s['time_s']:10.3f}"
        )


def _build_config():
    _validate_betti(BETTI)
    if PRINT_EVERY < 1:
        raise ValueError("PRINT_EVERY must be >= 1")
    if K not in (0, 1, 2, 3):
        raise ValueError("K must be one of 0, 1, 2, 3")
    if PRECOND_KIND not in ("auto", "none", "jacobi", "tensor"):
        raise ValueError("PRECOND_KIND must be one of auto, none, jacobi, tensor")
    if INNER_MAXITER < 1:
        raise ValueError("INNER_MAXITER must be >= 1")
    if POST_EXACT_LIFT_COMPARE_MAXITER < 1:
        raise ValueError("POST_EXACT_LIFT_COMPARE_MAXITER must be >= 1")
    if not POST_EXACT_LIFT_HYBRID_ALPHAS:
        raise ValueError("POST_EXACT_LIFT_HYBRID_ALPHAS must be non-empty")
    if K0_COMPARE_MAXITER < 1:
        raise ValueError("K0_COMPARE_MAXITER must be >= 1")

    # Keep geometry projection accuracy in step with resolution.
    sample_points_effective = SAMPLE_POINTS
    if AUTO_SAMPLE_POINTS:
        sample_points_effective = max(SAMPLE_POINTS, 6 * N + 2 * P)

    return SimpleNamespace(
        n=N,
        p=P,
        sample_points=SAMPLE_POINTS,
        auto_sample_points=AUTO_SAMPLE_POINTS,
        sample_points_effective=sample_points_effective,
        tol=TOL,
        maxiter=MAXITER,
        inner_maxiter=INNER_MAXITER,
        betti=BETTI,
        all_k=ALL_K,
        k=K,
        dirichlet=DIRICHLET,
        n_vectors=N_VECTORS,
        eps=EPS,
        abs_tol=ABS_TOL,
        precond_kind=PRECOND_KIND,
        seed=SEED,
        print_every=PRINT_EVERY,
        torus_epsilon=TORUS_EPSILON,
        torus_r0=TORUS_R0,
        check_positive_jacobian=CHECK_POSITIVE_JACOBIAN,
        jacobian_pos_tol=JACOBIAN_POS_TOL,
        run_post_nullspace_shifted_exact_lift_compare=RUN_POST_NULLSPACE_SHIFTED_EXACT_LIFT_COMPARE,
        post_exact_lift_compare_seed=POST_EXACT_LIFT_COMPARE_SEED,
        post_exact_lift_compare_maxiter=POST_EXACT_LIFT_COMPARE_MAXITER,
        post_exact_lift_hybrid_alphas=POST_EXACT_LIFT_HYBRID_ALPHAS,
        run_shifted_k0_compare=RUN_SHIFTED_K0_COMPARE,
        k0_compare_seed=K0_COMPARE_SEED,
        k0_compare_maxiter=K0_COMPARE_MAXITER,
    )


def main():
    args = _build_config()

    t0 = time.perf_counter()
    print("Assembling baseline sequence and operators...")
    seq, operators = _build_sequence(args)
    operators = _seed_debug_nullspaces(seq, operators)
    print(f"assembly_time_s={time.perf_counter() - t0:.3f}")

    _run_debug(args, seq, operators)
    _run_post_nullspace_shifted_exact_lift_compare(args, seq)
    _run_shifted_k0_compare(args, seq)


if __name__ == "__main__":
    main()
