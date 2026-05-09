"""Benchmark production tensor mass preconditioner against Jacobi and Chebyshev.

Builds the de Rham sequence and base mass operators **once**, then for each
``k`` and each preconditioner choice runs ``solve_singular_cg`` against a
small batch of random RHS and reports avg/max iteration counts and average
wall-clock per solve.

Three preconditioners per ``k``:

- ``tensor``    : production rank-1 Kronecker mass preconditioner
                  (``apply_mass_tensor_preconditioner_ops``).
- ``jacobi``    : pointwise inverse of the assembled mass diagonal
                  (``apply_mass_matrix_preconditioner(kind='jacobi')``).
- ``chebyshev`` : matrix-free Chebyshev polynomial of degree ``--cheb-steps``
                  in the mass operator with Jacobi as inner smoother. Spectral
                  bounds are estimated once at build time via
                  ``update_mass_runtime_tuning``.

Run with ``--help`` for knobs. Defaults are sized to match
``benchmark_richardson_vs_modal.py``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    apply_mass_tensor_preconditioner_ops,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
)
from mrx.operators import (  # noqa: E402
    _build_chebyshev_apply_preconditioner,
    _estimate_chebyshev_lanczos_bounds_apply,
)
from mrx.preconditioners import MassPreconditionerSpec
from mrx.solvers import solve_singular_cg
from mrx.utils import build_random_besov_rhs_batch


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
class Row:
    k: int
    strategy: str
    avg_iters: float
    max_iters: int
    avg_ms: float
    rank: int = -1  # -1 = N/A (not a tensor strategy)


def _parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in text.split(",") if s.strip())


def _parse_ns(text: str) -> tuple[int, int, int]:
    parts = _parse_int_list(text)
    if len(parts) != 3:
        raise ValueError(f"Expected ns as 'nr,nt,nz', got {text!r}")
    return parts  # type: ignore[return-value]


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


def time_solve(solve, rhs_batch) -> tuple[float, int, float]:
    x, it = solve(rhs_batch[0])
    jax.block_until_ready(x)

    iters: list[int] = []
    times_ms: list[float] = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        x, it = solve(rhs)
        jax.block_until_ready(x)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iters.append(int(it))
    return (
        float(jnp.mean(jnp.asarray(iters))),
        int(max(iters)),
        float(jnp.mean(jnp.asarray(times_ms))),
    )


def benchmark_cell(
    seq: DeRhamSequence,
    operators,
    rhs_batch,
    args,
    *,
    k: int,
    strategy: str,
) -> Row:
    dirichlet = not args.free

    operator_apply = lambda x: apply_mass_matrix(
        seq, operators, x, k, dirichlet=dirichlet,
    )

    if strategy == "tensor":
        precond_apply = lambda rhs: apply_mass_tensor_preconditioner_ops(
            seq, operators, rhs, k, dirichlet=dirichlet,
        )
    elif strategy == "jacobi":
        precond_apply = lambda rhs: apply_mass_matrix_preconditioner(
            seq, operators, rhs, k, dirichlet=dirichlet, kind="jacobi",
        )
    elif strategy.startswith("cheb"):
        # Build Jacobi smoother explicitly so we can plug it into Chebyshev
        # without going through the public mass preconditioner machinery
        # (which currently restricts the inner smoother to kind='tensor').
        jacobi_apply = lambda rhs: apply_mass_matrix_preconditioner(
            seq, operators, rhs, k, dirichlet=dirichlet, kind="jacobi",
        )
        suffix = "_dbc" if dirichlet else ""
        dof = int(getattr(seq, f"n{k}{suffix}"))
        spec = MassPreconditionerSpec(
            kind="chebyshev",
            steps=args.cheb_steps,
            power_iterations=8,
            min_eig_fraction=1e-3,
            smoother=MassPreconditionerSpec(kind="jacobi"),
        )
        min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
            operator_apply, jacobi_apply, dof, spec=spec, seed=args.seed,
        )
        precond_apply = _build_chebyshev_apply_preconditioner(
            operator_apply, jacobi_apply,
            steps=args.cheb_steps, min_eig=min_eig, max_eig=max_eig,
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    @jax.jit
    def solve(rhs):
        x, info = solve_singular_cg(
            operator_apply,
            rhs,
            mass_matvec=operator_apply,
            precond_matvec=precond_apply,
            tol=args.tol,
            maxiter=args.maxiter,
        )
        return x, jnp.abs(info)

    avg_it, max_it, avg_ms = time_solve(solve, rhs_batch)
    return Row(k=k, strategy=strategy, avg_iters=avg_it, max_iters=max_it,
               avg_ms=avg_ms, rank=getattr(args, "_active_rank", -1) if strategy == "tensor" else -1)


def print_table(rows: list[Row]) -> None:
    header = (
        f"{'k':>2} {'strategy':>14} "
        f"{'avg_it':>7} {'max_it':>7} {'avg_ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        label = (
            f"tensor(r={row.rank})" if row.strategy == "tensor" and row.rank >= 0
            else row.strategy
        )
        print(
            f"{row.k:>2d} {label:>14} "
            f"{row.avg_iters:>7.1f} {row.max_iters:>7d} {row.avg_ms:>9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=_parse_ns, default=(6, 8, 4))
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--ks", type=_parse_int_list, default=(0, 1, 2, 3))
    parser.add_argument("--strategies", type=str,
                        default="tensor,jacobi,chebyshev",
                        help="Comma-separated subset of tensor,jacobi,chebyshev.")
    parser.add_argument("--cheb-steps", type=int, default=4,
                        help="Polynomial degree for the Chebyshev mass smoother.")
    parser.add_argument("--rank", type=int, default=None,
                        help="Single rank shortcut for --ranks (deprecated; use --ranks).")
    parser.add_argument("--ranks", type=_parse_int_list, default=(1,),
                        help="Comma-separated tensor ranks to compare (>=1). "
                             "rank=1 single Kronecker, rank=2 exact two-Kron via "
                             "Lynch FD, rank>=3 Lynch FD on the leading two terms "
                             "with diagonal-truncated contributions from the rest.")
    parser.add_argument("--n-rhs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--free", action="store_true",
                        help="Solve with free (non-dirichlet) boundary data.")
    parser.add_argument("--rotating-eps", type=float, default=1.0 / 7.0)
    parser.add_argument("--rotating-kappa", type=float, default=1.5)
    parser.add_argument("--rotating-r0", type=float, default=3.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    parser.add_argument("--no-inner-schur", dest="inner_schur", action="store_false",
                        help="Disable the inner RT/zeta Schur in the k=1/k=2 bulk apply "
                             "(use block-diagonal smoothing instead; ~3 einsums vs ~14).")
    parser.set_defaults(inner_schur=False)
    args = parser.parse_args()

    invalid = tuple(k for k in args.ks if k not in (0, 1, 2, 3))
    if invalid:
        raise ValueError(f"Mass degrees must be in 0,1,2,3; got {invalid}")
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    for s in strategies:
        if s not in ("tensor", "jacobi", "chebyshev"):
            raise ValueError(f"Unknown strategy {s!r}")
    if args.rank is not None:
        ranks = (args.rank,)
    else:
        ranks = args.ranks
    invalid_ranks = tuple(r for r in ranks if r < 1)
    if invalid_ranks:
        raise ValueError(f"--ranks must be >= 1; got {invalid_ranks}")

    print(f"Building sequence ns={args.ns} p={args.p} ...", flush=True)
    seq = build_sequence(args)

    base_operators = assemble_mass_operators(seq, seq.geometry, ks=tuple(args.ks))

    dirichlet = not args.free

    rhs_batches = {
        k: build_random_besov_rhs_batch(
            seq, k, dirichlet=dirichlet,
            n_rhs=args.n_rhs, seed=args.seed, **BESOV_RHS_KWARGS,
        )
        for k in args.ks
    }

    rows: list[Row] = []
    # Run non-tensor strategies once (independent of --ranks).
    non_tensor_strategies = tuple(s for s in strategies if s != "tensor")
    if non_tensor_strategies:
        for k in args.ks:
            rhs_batch = rhs_batches[k]
            for strategy in non_tensor_strategies:
                rows.append(
                    benchmark_cell(
                        seq, base_operators, rhs_batch, args,
                        k=k, strategy=strategy,
                    )
                )
    # Tensor strategy: rebuild the preconditioner per rank.
    if "tensor" in strategies:
        for rank in ranks:
            print(f"Assembling tensor preconditioner rank={rank} ...", flush=True)
            operators = assemble_tensor_mass_preconditioner(
                seq, operators=base_operators, ks=tuple(args.ks), rank=rank,
                cp_kwargs={
                    "k1_inner_schur": args.inner_schur,
                    "k2_inner_schur": args.inner_schur,
                },
            )
            args._active_rank = rank
            for k in args.ks:
                rhs_batch = rhs_batches[k]
                rows.append(
                    benchmark_cell(
                        seq, operators, rhs_batch, args,
                        k=k, strategy="tensor",
                    )
                )

    # Sort rows: by k then by strategy/rank for readability.
    def _sort_key(row: Row):
        order = {"jacobi": 0, "chebyshev": 1, "tensor": 2}.get(row.strategy, 99)
        return (row.k, order, row.rank)
    rows.sort(key=_sort_key)

    print()
    print_table(rows)


if __name__ == "__main__":
    main()
