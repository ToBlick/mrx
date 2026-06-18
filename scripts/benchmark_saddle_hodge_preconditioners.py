"""Benchmark saddle-point Hodge-Laplacian preconditioners for k=1 and k=2.

Compares two block-diagonal MINRES preconditioners for the unshifted Hodge
Laplacian solve in the currently harmonic-free cases:

- ``baseline``: lower mass block uses Jacobi; upper Schur block uses
  ``exact_jacobi``, i.e. the exact diagonal of the true Schur operator probed
  through repeated applies with exact lower mass solves.
- ``tensor_chebyshev``: lower mass block uses the production tensor mass
  preconditioner; upper Schur block uses a Chebyshev polynomial on the
  approximate Schur operator built from ``D M_precond D^T`` with a tensor mass
  smoother on the k-form block.

The currently supported benchmark cases are

- ``k1_dbc``
- ``k2_free``
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.nullspace import get_nullspace
from mrx.operators import (
    _build_exact_jacobi_preconditioner_apply,
    _build_mass_preconditioner_apply,
    _build_operator_preconditioner_apply,
    _build_schur_apply_from_saddle_preconditioner,
    _build_schur_operator_apply,
    _coerce_saddle_preconditioner_spec,
    _saddle_nullspaces,
    _select_schur_runtime_tuning,
    apply_derivative_matrix,
    apply_laplacian,
    apply_inverse_mass_matrix,
    apply_mass_matrix,
    apply_stiffness,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
)
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from mrx.solvers import solve_saddle_point_minres
from mrx.io import parse_int_list, parse_ns
from test.random_fields import build_random_besov_rhs_batch


jax.config.update("jax_enable_x64", True)


TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)
BESOV_RHS_KWARGS = {
    "s": 1.0,
    "upper_limit": 24,
    "num_modes": 64,
    "scale": 1.0,
    "smoothness_margin": 0.0,
    "normalization_samples": 256,
}


@dataclass(frozen=True)
class Case:
    k: int
    dirichlet: bool

    @property
    def label(self) -> str:
        return f"k{self.k}_{'dbc' if self.dirichlet else 'free'}"

    @property
    def bc(self) -> str:
        return "dbc" if self.dirichlet else "free"


@dataclass(frozen=True)
class Row:
    case: str
    strategy: str
    avg_iters: float
    max_iters: int
    avg_ms: float
    failures: int
    avg_residual: float
    rank: int = -1
    cheb_steps: int = -1


def _parse_cases(text: str) -> tuple[Case, ...]:
    cases: list[Case] = []
    for raw in text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "k1_dbc":
            cases.append(Case(k=1, dirichlet=True))
        elif token == "k2_free":
            cases.append(Case(k=2, dirichlet=False))
        else:
            raise ValueError(
                f"Unknown case {raw!r}; expected comma-separated subset of k1_dbc,k2_free"
            )
    if not cases:
        raise ValueError("At least one saddle benchmark case is required")
    return tuple(cases)


def build_sequence(args) -> DeRhamSequence:
    seq = DeRhamSequence(
        args.ns,
        (args.p, args.p, args.p),
        2 * args.p,
        TYPES,
        polar=True,
        tol=args.tol,
        maxiter=args.maxiter,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(
        rotating_ellipse_map(
            eps=args.rotating_eps,
            kappa=args.rotating_kappa,
            R0=args.rotating_r0,
            nfp=args.rotating_nfp,
        )
    )
    return seq


def build_base_operators(seq: DeRhamSequence, cases: tuple[Case, ...]):
    mass_ks = tuple(
        sorted(
            {
                degree
                for case in cases
                for degree in (case.k - 1, case.k, case.k + 1)
                if 0 <= degree <= 3
            }
        )
    )
    incidence_ks = tuple(
        sorted(
            {
                degree
                for case in cases
                for degree in (case.k - 1, case.k)
                if 0 <= degree <= 2
            }
        )
    )
    operators = assemble_mass_operators(seq, seq.geometry, ks=mass_ks)
    operators = assemble_incidence_operators(seq, operators=operators, ks=incidence_ks)
    seq.set_operators(operators)
    return operators


def build_rhs_batch(seq: DeRhamSequence, case: Case, *, n_rhs: int, seed: int):
    return build_random_besov_rhs_batch(
        seq,
        case.k,
        dirichlet=case.dirichlet,
        n_rhs=n_rhs,
        seed=seed,
        **BESOV_RHS_KWARGS,
    )


def baseline_preconditioner() -> SaddlePointPreconditionerSpec:
    return SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind="jacobi"),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind="tensor"),
            outer=MassPreconditionerSpec(kind="exact_jacobi"),
        ),
        coupled=False,
    )


def tensor_chebyshev_preconditioner(args, *, cheb_steps: int) -> SaddlePointPreconditionerSpec:
    return SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind="tensor", surgery_schur=True),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind="tensor"),
            outer=MassPreconditionerSpec(
                kind="chebyshev",
                steps=cheb_steps,
                power_iterations=args.cheb_power_iterations,
                min_eig_fraction=args.cheb_min_eig_fraction,
            ),
        ),
        coupled=False,
    )


def time_solve(solve, rhs_batch) -> tuple[float, int, float, int, float]:
    x, info, residual = solve(rhs_batch[0])
    jax.block_until_ready((x, info, residual))

    iters: list[int] = []
    times_ms: list[float] = []
    failures = 0
    residuals: list[float] = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        x, info, residual = solve(rhs)
        jax.block_until_ready((x, info, residual))
        info_int = int(info)
        failures += int(info_int >= 0)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iters.append(abs(info_int))
        residuals.append(float(residual))
    return (
        float(jnp.mean(jnp.asarray(iters))),
        int(max(iters)),
        float(jnp.mean(jnp.asarray(times_ms))),
        failures,
        float(jnp.mean(jnp.asarray(residuals))),
    )


def benchmark_case(
    seq: DeRhamSequence,
    operators,
    rhs_batch,
    args,
    *,
    case: Case,
    strategy: str,
    cheb_steps: int = -1,
) -> Row:
    if strategy == "baseline":
        preconditioner = baseline_preconditioner()
    elif strategy == "tensor_chebyshev":
        preconditioner = tensor_chebyshev_preconditioner(args, cheb_steps=cheb_steps)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    saddle_preconditioner = _coerce_saddle_preconditioner_spec(
        seq,
        operators,
        k=case.k,
        preconditioner=preconditioner,
    )
    vs_upper, vs_lower = _saddle_nullspaces(seq, operators, case.k, case.dirichlet)
    suffix = "_dbc" if case.dirichlet else ""
    n_upper = getattr(seq, f"n{case.k}{suffix}")
    n_lower = getattr(seq, f"n{case.k - 1}{suffix}")

    precond_lower = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=case.k - 1,
        dirichlet=case.dirichlet,
        preconditioner=saddle_preconditioner.mass,
        allow_none=True,
    )
    if saddle_preconditioner.schur.outer.kind == "exact_jacobi":
        exact_lower = lambda rhs: apply_inverse_mass_matrix(
            seq,
            operators,
            rhs,
            case.k - 1,
            dirichlet=case.dirichlet,
            preconditioner="jacobi",
        )
        schur_probe_apply = _build_schur_operator_apply(
            seq,
            operators,
            k=case.k,
            dirichlet=case.dirichlet,
            eps=0.0,
            inner_preconditioner_apply=exact_lower,
        )
        precond_upper = _build_exact_jacobi_preconditioner_apply(
            schur_probe_apply,
            n_upper,
            warning_context=f"benchmark baseline exact_jacobi for k={case.k}",
        )
    else:
        schur_apply = _build_schur_apply_from_saddle_preconditioner(
            seq,
            operators,
            k=case.k,
            dirichlet=case.dirichlet,
            eps=0.0,
            saddle_preconditioner=saddle_preconditioner,
        )
        precond_upper = _build_operator_preconditioner_apply(
            seq,
            operators,
            k=case.k,
            dirichlet=case.dirichlet,
            operator_apply=schur_apply,
            preconditioner=saddle_preconditioner.schur.outer,
            allow_none=True,
            orthogonal_vectors=vs_upper,
            runtime_tuning=_select_schur_runtime_tuning(
                operators,
                case.k,
                case.dirichlet,
                0.0,
            ),
        )

    @jax.jit
    def solve(rhs):
        x, _, info = solve_saddle_point_minres(
            stiffness_matvec=lambda x: apply_stiffness(
                seq, operators, x, case.k, dirichlet=case.dirichlet,
            ),
            derivative_matvec=lambda s: apply_derivative_matrix(
                seq,
                operators,
                s,
                case.k - 1,
                dirichlet_in=case.dirichlet,
                dirichlet_out=case.dirichlet,
            ),
            derivative_T_matvec=lambda u: apply_derivative_matrix(
                seq,
                operators,
                u,
                case.k - 1,
                dirichlet_in=case.dirichlet,
                dirichlet_out=case.dirichlet,
                transpose=True,
            ),
            mass_lower_matvec=lambda s: apply_mass_matrix(
                seq, operators, s, case.k - 1, dirichlet=case.dirichlet,
            ),
            b_upper=rhs,
            n_upper=n_upper,
            n_lower=n_lower,
            precond_upper=precond_upper,
            precond_lower=precond_lower,
            mass_upper_matvec=lambda x: apply_mass_matrix(
                seq, operators, x, case.k, dirichlet=case.dirichlet,
            ),
            vs_upper=vs_upper,
            vs_lower=vs_lower,
            tol=args.tol,
            maxiter=args.maxiter,
        )
        residual = apply_laplacian(
            seq,
            operators,
            x,
            case.k,
            dirichlet=case.dirichlet,
            tol=args.tol,
            maxiter=args.maxiter,
        ) - rhs
        r_M = seq.l2_norm(residual, case.k, dirichlet=case.dirichlet)
        b_M = seq.l2_norm(rhs, case.k, dirichlet=case.dirichlet)
        return x, info, r_M / jnp.where(b_M > 0.0, b_M, 1.0)

    avg_it, max_it, avg_ms, failures, avg_residual = time_solve(solve, rhs_batch)
    return Row(
        case=case.label,
        strategy=strategy,
        avg_iters=avg_it,
        max_iters=max_it,
        avg_ms=avg_ms,
        failures=failures,
        avg_residual=avg_residual,
        rank=getattr(args, "_active_rank", -1) if strategy == "tensor_chebyshev" else -1,
        cheb_steps=cheb_steps if strategy == "tensor_chebyshev" else -1,
    )


def print_table(rows: list[Row]) -> None:
    header = (
        f"{'case':>9} {'strategy':>18} {'avg_it':>7} {'max_it':>7} {'avg_ms':>9} {'fails':>6} {'avg_resM':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        label = (
            f"tensor+cheb(r={row.rank},s={row.cheb_steps})"
            if row.strategy == "tensor_chebyshev" and row.rank >= 0
            else row.strategy
        )
        print(
            f"{row.case:>9} {label:>18} {row.avg_iters:>7.1f} {row.max_iters:>7d}"
            f" {row.avg_ms:>9.2f} {row.failures:>6d} {row.avg_residual:>10.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=parse_ns, default=(12, 16, 8))
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--cases", type=_parse_cases, default=(Case(1, True), Case(2, False)))
    parser.add_argument(
        "--strategies",
        type=str,
        default="baseline,tensor_chebyshev",
        help="Comma-separated subset of baseline,tensor_chebyshev.",
    )
    parser.add_argument(
        "--cheb-steps",
        type=_parse_int_list,
        default=(4,),
        help="Comma-separated Schur-outer Chebyshev step counts for tensor_chebyshev.",
    )
    parser.add_argument("--cheb-power-iterations", type=int, default=8)
    parser.add_argument("--cheb-min-eig-fraction", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=None,
                        help="Single rank shortcut for --ranks (deprecated; use --ranks).")
    parser.add_argument(
        "--ranks",
        type=parse_int_list,
        default=(1,),
        help="Comma-separated tensor ranks to compare for tensor_chebyshev.",
    )
    parser.add_argument("--n-rhs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--rotating-eps", type=float, default=1.0 / 7.0)
    parser.add_argument("--rotating-kappa", type=float, default=1.5)
    parser.add_argument("--rotating-r0", type=float, default=3.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    parser.add_argument("--no-inner-schur", dest="inner_schur", action="store_false",
                        help="Disable the inner RT/zeta Schur in the tensor mass preconditioner build.")
    parser.set_defaults(inner_schur=False)
    args = parser.parse_args()

    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    for strategy in strategies:
        if strategy not in ("baseline", "tensor_chebyshev"):
            raise ValueError(f"Unknown strategy {strategy!r}")

    if args.rank is not None:
        ranks = (args.rank,)
    else:
        ranks = args.ranks
    invalid_ranks = tuple(r for r in ranks if r < 1)
    if invalid_ranks:
        raise ValueError(f"--ranks must be >= 1; got {invalid_ranks}")
    invalid_cheb_steps = tuple(step for step in args.cheb_steps if step < 1)
    if invalid_cheb_steps:
        raise ValueError(f"--cheb-steps must be >= 1; got {invalid_cheb_steps}")

    print(f"Building sequence ns={args.ns} p={args.p} ...", flush=True)
    seq = build_sequence(args)
    base_operators = build_base_operators(seq, args.cases)

    for case in args.cases:
        harmonic_count = int(get_nullspace(base_operators, case.k, case.dirichlet).shape[0])
        if harmonic_count != 0:
            raise ValueError(
                f"Case {case.label} is not harmonic-free; got {harmonic_count} harmonic vectors"
            )

    rhs_batches = {
        case.label: build_rhs_batch(
            seq,
            case,
            n_rhs=args.n_rhs,
            seed=args.seed + 1000 * case.k + 17 * int(case.dirichlet),
        )
        for case in args.cases
    }

    rows: list[Row] = []
    if "baseline" in strategies:
        for case in args.cases:
            rows.append(
                benchmark_case(
                    seq,
                    base_operators,
                    rhs_batches[case.label],
                    args,
                    case=case,
                    strategy="baseline",
                    cheb_steps=-1,
                )
            )

    if "tensor_chebyshev" in strategies:
        tensor_ks = tuple(sorted({degree for case in args.cases for degree in (case.k - 1, case.k)}))
        for rank in ranks:
            print(f"Assembling tensor mass preconditioner rank={rank} ...", flush=True)
            operators = assemble_tensor_mass_preconditioner(
                seq,
                operators=base_operators,
                ks=tensor_ks,
                rank=rank,
                cp_kwargs={
                    "k1_inner_schur": args.inner_schur,
                    "k2_inner_schur": args.inner_schur,
                },
            )
            args._active_rank = rank
            for cheb_steps in args.cheb_steps:
                for case in args.cases:
                    rows.append(
                        benchmark_case(
                            seq,
                            operators,
                            rhs_batches[case.label],
                            args,
                            case=case,
                            strategy="tensor_chebyshev",
                            cheb_steps=cheb_steps,
                        )
                    )

    def _sort_key(row: Row):
        order = {"baseline": 0, "tensor_chebyshev": 1}.get(row.strategy, 99)
        return (row.case, order, row.rank, row.cheb_steps)

    rows.sort(key=_sort_key)

    print()
    print_table(rows)


if __name__ == "__main__":
    main()