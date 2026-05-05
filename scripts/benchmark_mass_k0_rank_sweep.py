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
class BenchmarkRow:
    rank: int
    cp_relative_error: float
    cp_final_delta: float
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_ms: float
    std_ms: float
    max_ms: float


def _parse_ns(text: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in text.split(","))
    if len(parts) != 3:
        raise ValueError(f"Expected ns as 'nr,nt,nz', got {text!r}")
    return parts


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of integers")
    return values


def _mass_cp_summary(factors, k: int) -> tuple[float, float]:
    if k == 0:
        return float(factors.bulk.cp_relative_error), float(factors.bulk.cp_final_delta)
    if k == 1:
        return (
            float(max(factors.arr.cp_relative_error, factors.theta.cp_relative_error, factors.zeta.cp_relative_error)),
            float(max(factors.arr.cp_final_delta, factors.theta.cp_final_delta, factors.zeta.cp_final_delta)),
        )
    if k == 2:
        return (
            float(max(factors.r_bulk.cp_relative_error, factors.theta.cp_relative_error, factors.zeta.cp_relative_error)),
            float(max(factors.r_bulk.cp_final_delta, factors.theta.cp_final_delta, factors.zeta.cp_final_delta)),
        )
    return float(factors.cp_relative_error), float(factors.cp_final_delta)


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


def build_rhs_batch(seq: DeRhamSequence, *, k: int, dirichlet: bool, n_rhs: int, seed: int):
    return build_random_besov_rhs_batch(
        seq,
        k,
        dirichlet=dirichlet,
        n_rhs=n_rhs,
        seed=seed,
        **BESOV_RHS_KWARGS,
    )


def time_solve(solve, rhs_batch) -> dict[str, float | int]:
    x, it = solve(rhs_batch[0])
    jax.block_until_ready(x)

    iters = []
    times_ms = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        x, it = solve(rhs)
        jax.block_until_ready(x)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iters.append(int(it))

    iters = jnp.asarray(iters)
    times_ms = jnp.asarray(times_ms)
    return {
        "avg_iters": float(jnp.mean(iters)),
        "std_iters": float(jnp.std(iters)),
        "max_iters": int(jnp.max(iters)),
        "avg_ms": float(jnp.mean(times_ms)),
        "std_ms": float(jnp.std(times_ms)),
        "max_ms": float(jnp.max(times_ms)),
    }


def benchmark_rank(seq: DeRhamSequence, rhs_batch, args, rank: int) -> BenchmarkRow:
    dirichlet = not args.free
    cp_kwargs = {
        "tol": args.cp_tol,
        "maxiter": args.cp_maxiter,
        "ridge": args.cp_ridge,
    }

    operators = None
    operators = assemble_mass_operators(seq, seq.geometry, operators=operators, ks=(args.k,))
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(args.k,),
        rank=rank,
        cp_kwargs=cp_kwargs,
    )

    factors = select_boundary_data(getattr(operators.mass_preconds.tensor, f"k{args.k}"), dirichlet, f"Tensor mass k={args.k}")

    operator_apply = lambda x: apply_mass_matrix(seq, operators, x, args.k, dirichlet=dirichlet)
    preconditioner_apply = lambda rhs: apply_mass_tensor_preconditioner_ops(
        seq,
        operators,
        rhs,
        args.k,
        dirichlet=dirichlet,
    )

    @jax.jit
    def solve(rhs):
        x, info = solve_singular_cg(
            operator_apply,
            rhs,
            mass_matvec=operator_apply,
            precond_matvec=preconditioner_apply,
            tol=args.tol,
            maxiter=args.maxiter,
        )
        return x, jnp.abs(info)

    stats = time_solve(solve, rhs_batch)
    cp_relative_error, cp_final_delta = _mass_cp_summary(factors, args.k)
    return BenchmarkRow(
        rank=rank,
        cp_relative_error=cp_relative_error,
        cp_final_delta=cp_final_delta,
        **stats,
    )


def print_table(rows: list[BenchmarkRow]) -> None:
    print()
    header = (
        f"{'rank':>6} {'cp_err':>12} {'cp_delta':>12} {'avg_it':>8} {'std_it':>8} {'max_it':>8} "
        f"{'avg_ms':>10} {'std_ms':>10} {'max_ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.rank:>6d} {row.cp_relative_error:>12.5e} {row.cp_final_delta:>12.5e} "
            f"{row.avg_iters:>8.1f} {row.std_iters:>8.2f} {row.max_iters:>8d} "
            f"{row.avg_ms:>10.2f} {row.std_ms:>10.2f} {row.max_ms:>10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark mass-k solves for several tensor ranks."
    )
    parser.add_argument("--k", type=int, choices=(0, 1, 2, 3), default=0, help="Mass degree to benchmark")
    parser.add_argument("--ns", type=_parse_ns, default=(8, 16, 8), help="Grid sizes as nr,nt,nz")
    parser.add_argument("--p", type=int, default=3, help="Spline degree in each direction")
    parser.add_argument("--ranks", type=_parse_int_list, default=(1, 2, 3), help="Comma-separated tensor ranks")
    parser.add_argument("--n-rhs", type=int, default=8, help="Number of right-hand sides")
    parser.add_argument("--seed", type=int, default=100, help="PRNG seed for RHS generation")
    parser.add_argument("--free", action="store_true", help="Use free extraction space")
    parser.add_argument("--tol", type=float, default=1e-9, help="CG tolerance")
    parser.add_argument("--maxiter", type=int, default=1000, help="Maximum CG iterations")
    parser.add_argument("--cp-maxiter", type=int, default=100, help="Maximum CP ALS iterations")
    parser.add_argument("--cp-tol", type=float, default=1e-9, help="CP ALS tolerance")
    parser.add_argument("--cp-ridge", type=float, default=1e-12, help="CP ALS ridge regularization")
    parser.add_argument("--rotating-eps", type=float, default=0.33)
    parser.add_argument("--rotating-kappa", type=float, default=1.4)
    parser.add_argument("--rotating-r0", type=float, default=1.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    args = parser.parse_args()

    print("Building sequence...")
    t0 = time.perf_counter()
    seq = build_sequence(args)
    print(f"built in {time.perf_counter() - t0:.2f} s")
    print(f"problem=mass-k{args.k}, ns={args.ns}, p={args.p}, ranks={args.ranks}, dirichlet={not args.free}, n_rhs={args.n_rhs}")
    print("RHS: random Besov-like functions projected into the FEM space")

    rhs_batch = build_rhs_batch(seq, k=args.k, dirichlet=not args.free, n_rhs=args.n_rhs, seed=args.seed)
    rows = [benchmark_rank(seq, rhs_batch, args, rank) for rank in args.ranks]
    print_table(rows)


if __name__ == "__main__":
    main()