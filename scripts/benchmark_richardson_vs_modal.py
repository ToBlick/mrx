"""Compare the rank-1 + correction + Richardson tensor preconditioner against
the production shared-modal multirank inverse on the mass solves.

Builds the de Rham sequence and base mass operators **once**. For each
(rank, richardson_steps) cell we only rebuild the tensor mass preconditioner
via ``assemble_tensor_mass_preconditioner`` (which reuses the underlying
operators and surgery data). For each cell we benchmark the matrix-free
preconditioned CG that the production code uses, on a small RHS batch, and
report iteration counts and wall-clock per solve.

Two preconditioner variants are compared per cell:

- ``modal``   : ``fit_strategy='multiplicative'``, ``richardson_steps=0``.
                This is the current production multirank shared-modal inverse.
- ``split+R`` : ``fit_strategy='split'``, ``richardson_steps=R``,
                ``richardson_omega=0.0`` (auto-tuned via Lanczos at build
                time). Rank-1 backbone smoother + R Richardson sweeps against
                the full rank-`r` tensor model.

Run with ``--help`` to see knobs. Defaults are small enough to fit on a
modest GPU/CPU; bump ``--ns`` and ``--p`` for production-size sweeps.
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
    apply_mass_tensor_preconditioner_ops,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
)
from mrx.preconditioners import select_boundary_data
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
    rank: int
    richardson_steps: int
    richardson_omega: float
    cp_relative_error: float
    avg_iters: float
    max_iters: int
    avg_ms: float


def _parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in text.split(",") if s.strip())


def _parse_ns(text: str) -> tuple[int, int, int]:
    parts = _parse_int_list(text)
    if len(parts) != 3:
        raise ValueError(f"Expected ns as 'nr,nt,nz', got {text!r}")
    return parts  # type: ignore[return-value]


def _cp_summary(tensor_factors, k: int) -> tuple[float, float]:
    if k == 0:
        b = tensor_factors.bulk
        return float(b.cp_relative_error), float(b.richardson_omega)
    if k == 1:
        return (
            float(max(tensor_factors.arr.cp_relative_error,
                      tensor_factors.theta.cp_relative_error,
                      tensor_factors.zeta.cp_relative_error)),
            float(tensor_factors.arr.richardson_omega),
        )
    if k == 2:
        return (
            float(max(tensor_factors.r_bulk.cp_relative_error,
                      tensor_factors.theta.cp_relative_error,
                      tensor_factors.zeta.cp_relative_error)),
            float(tensor_factors.r_bulk.richardson_omega),
        )
    return float(tensor_factors.cp_relative_error), float(tensor_factors.richardson_omega)


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
    # Warmup / compile.
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
    return float(jnp.mean(jnp.asarray(iters))), int(max(iters)), float(jnp.mean(jnp.asarray(times_ms)))


def benchmark_cell(
    seq: DeRhamSequence,
    operators_base,
    rhs_batch,
    args,
    *,
    k: int,
    strategy: str,
    rank: int,
    richardson_steps: int,
) -> Row:
    dirichlet = not args.free
    cp_kwargs = {
        "tol": args.cp_tol,
        "maxiter": args.cp_maxiter,
        "ridge": args.cp_ridge,
        "fit_strategy": "split" if strategy == "split+R" else "multiplicative",
        "richardson_steps": richardson_steps if strategy == "split+R" else 0,
        # Auto-tune omega when Richardson is on; ignored for the modal variant.
        "richardson_omega": 0.0 if strategy == "split+R" else 1.0,
    }

    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators_base,
        ks=(k,),
        rank=rank,
        cp_kwargs=cp_kwargs,
    )

    factors = select_boundary_data(
        getattr(operators.mass_preconds.tensor, f"k{k}"),
        dirichlet,
        f"Tensor mass k={k}",
    )
    cp_err, omega_actual = _cp_summary(factors, k)

    operator_apply = lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)
    precond_apply = lambda rhs: apply_mass_tensor_preconditioner_ops(
        seq, operators, rhs, k, dirichlet=dirichlet,
    )

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

    return Row(
        k=k,
        strategy=strategy,
        rank=rank,
        richardson_steps=richardson_steps if strategy == "split+R" else 0,
        richardson_omega=omega_actual,
        cp_relative_error=cp_err,
        avg_iters=avg_it,
        max_iters=max_it,
        avg_ms=avg_ms,
    )


def print_table(rows: list[Row]) -> None:
    header = (
        f"{'k':>2} {'strategy':>10} {'rank':>4} {'R':>3} "
        f"{'omega':>9} {'cp_err':>11} {'avg_it':>7} {'max_it':>7} {'avg_ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.k:>2d} {row.strategy:>10} {row.rank:>4d} {row.richardson_steps:>3d} "
            f"{row.richardson_omega:>9.4f} {row.cp_relative_error:>11.3e} "
            f"{row.avg_iters:>7.1f} {row.max_iters:>7d} {row.avg_ms:>9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=_parse_ns, default=(6, 8, 4))
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--ks", type=_parse_int_list, default=(0,1,2,3),
                        help="Mass degrees to benchmark (subset of 0,1,2,3).")
    parser.add_argument("--ranks", type=_parse_int_list, default=(1, 2, 3))
    parser.add_argument("--richardson-steps", type=_parse_int_list, default=(0, 1, 2, 4),
                        help="Richardson sweep counts for the split variant.")
    parser.add_argument("--n-rhs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--free", action="store_true",
                        help="Solve with free (non-dirichlet) boundary data.")
    parser.add_argument("--cp-tol", type=float, default=1e-9)
    parser.add_argument("--cp-maxiter", type=int, default=100)
    parser.add_argument("--cp-ridge", type=float, default=1e-12)
    parser.add_argument("--rotating-eps", type=float, default=1.0 / 7.0)
    parser.add_argument("--rotating-kappa", type=float, default=1.5)
    parser.add_argument("--rotating-r0", type=float, default=3.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    args = parser.parse_args()

    invalid = tuple(k for k in args.ks if k not in (0, 1, 2, 3))
    if invalid:
        raise ValueError(f"Mass degrees must be in 0,1,2,3; got {invalid}")

    print(f"Building sequence ns={args.ns} p={args.p} ...", flush=True)
    seq = build_sequence(args)

    # One assembly of the bare mass operators for the requested degrees.
    operators_base = assemble_mass_operators(seq, seq.geometry, ks=tuple(args.ks))

    rhs_batches = {
        k: build_random_besov_rhs_batch(
            seq, k, dirichlet=not args.free,
            n_rhs=args.n_rhs, seed=args.seed, **BESOV_RHS_KWARGS,
        )
        for k in args.ks
    }

    rows: list[Row] = []
    for k in args.ks:
        rhs_batch = rhs_batches[k]
        for rank in args.ranks:
            # Modal baseline: rebuilt once per rank (richardson_steps ignored).
            rows.append(
                benchmark_cell(
                    seq, operators_base, rhs_batch, args,
                    k=k, strategy="modal", rank=rank, richardson_steps=0,
                )
            )
            # Split + Richardson: sweep over richardson_steps.
            for r_steps in args.richardson_steps:
                rows.append(
                    benchmark_cell(
                        seq, operators_base, rhs_batch, args,
                        k=k, strategy="split+R", rank=rank,
                        richardson_steps=r_steps,
                    )
                )

    print()
    print_table(rows)


if __name__ == "__main__":
    main()
