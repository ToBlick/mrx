"""Unified preconditioner benchmark driver (single-cell + sweep).

Defaults are anchored at resolution ``(8, 16, 8)`` and ``p=3``.

Compared strategies:

- k=0 (Dirichlet by default): tensor vs jacobi vs chebyshev(3)
- k=1 (Dirichlet by default):
    - schur outer jacobi, schur_diag_mode=tensor_probe
    - schur outer jacobi, schur_diag_mode=diag
    - schur outer chebyshev, schur_diag_mode=tensor_probe
    - schur outer chebyshev(3), schur_diag_mode=diag

The k=1 Schur inner preconditioner is always tensor.

Also benchmarks mass solves for all degrees by default (k=0,1,2,3):

- tensor vs jacobi vs chebyshev(3)

Modes:

- Single-cell mode (phase1-style): one fixed geometry and one `(n,p)` cell.
- Sweep mode (phase2-style): loops over `n`, `p`, and `kappa` lists.

In both modes, tensor-rank and Chebyshev-step sweeps are performed inside each
cell solve to avoid unnecessary base operator re-assembly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrx.derham_sequence import DeRhamSequence
from mrx.io import parse_int_list, parse_ns
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_jacobi_preconditioner,
    assemble_schur_jacobi_preconditioner,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from test.random_fields import build_random_besov_function


jax.config.update("jax_enable_x64", True)


@dataclass
class Row:
    case: str
    strategy: str
    method: str
    k: int
    dirichlet: bool
    avg_iters: float
    max_iters: int
    min_iters: int
    std_iters: float
    avg_solve_ms: float
    min_solve_ms: float
    max_solve_ms: float
    std_solve_ms: float
    final_residual: float
    setup_ms: float
    schur_diag_mode: str
    schur_outer_kind: str
    schur_outer_steps: int
    tensor_rank: int
    cheb_steps: int
    n_r: int
    n_t: int
    n_z: int
    p: int
    epsilon: float
    n_rhs: int
    seed: int
    kappa: float = 0.0
    warmup_ms: float = 0.0
    batch_solve_ms: float = 0.0
    batch_total_ms: float = 0.0
    method_total_ms: float = 0.0


TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)

BESOV_RHS_KWARGS = {
    "upper_limit": 24,
    "num_modes": 64,
    "scale": 1.0,
    "smoothness_margin": 0.0,
    "normalization_samples": 256,
}


def _parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(s.strip()) for s in text.split(",") if s.strip())


def _parse_mode_list(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


def _bc_label(dirichlet: bool) -> str:
    return "dbc" if dirichlet else "nbc"


def _laplace_dirichlet_map(args) -> dict[int, bool]:
    if getattr(args, "nullspace_free_laplace_bcs", False):
        # Nullspace-free Laplace choices on this topology:
        # k=0 DBC, k=1 DBC, k=2 NBC, k=3 NBC.
        return {0: True, 1: True, 2: False, 3: False}
    dirichlet = args.dirichlet.lower() == "dbc"
    return {0: dirichlet, 1: dirichlet, 2: dirichlet, 3: dirichlet}


def _progress(msg: str) -> None:
    print(f"[progress] {msg}", flush=True)


def _expected_row_count(laplace_ks: tuple[int, ...], mass_ks: tuple[int, ...],
                        tensor_ranks: tuple[int, ...], cheb_steps: tuple[int, ...],
                        schur_diag_modes: tuple[str, ...], *,
                        no_jacobi: bool = False) -> int:
    n_ranks = len(tensor_ranks)
    n_steps = len(cheb_steps)

    laplace_total = 0
    for k in laplace_ks:
        if k == 0:
            # jacobi (unless skipped) + chebyshev(steps) once, tensor per rank
            laplace_total += (0 if no_jacobi else 1) + n_steps + n_ranks
        else:
            n_modes = len(schur_diag_modes)
            # per-rank: jacobi (unless skipped) over modes + chebyshev(steps) over modes
            laplace_total += n_ranks * ((0 if no_jacobi else n_modes) + n_modes * n_steps)

    # per mass k: jacobi (unless skipped) once + tensor per rank + chebyshev(steps) per rank
    mass_total = len(mass_ks) * ((0 if no_jacobi else 1) + n_ranks + n_ranks * n_steps)
    return laplace_total + mass_total


def _build_sequence(args, ns: tuple[int, int, int]) -> DeRhamSequence:
    seq = DeRhamSequence(
        ns,
        (args.p, args.p, args.p),
        2 * args.p,
        TYPES,
        polar=True,
        tol=args.cg_tol,
        maxiter=args.cg_maxiter,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    _update_sequence_map(seq, args)
    return seq


def _update_sequence_map(seq: DeRhamSequence, args) -> None:
    seq.set_map(
        rotating_ellipse_map(
            eps=args.epsilon,
            kappa=args.kappa,
            R0=args.r0,
            nfp=args.nfp,
        )
    )


def _build_random_besov_rhs_batch_load(
        seq: DeRhamSequence,
        *,
        form_degree: int,
        dirichlet: bool,
        n_rhs: int,
        seed: int,
    s: float,
    load_frame: str):
    if n_rhs < 1:
        raise ValueError(f"n_rhs must be positive, got {n_rhs}")
    keys = jax.random.split(jax.random.PRNGKey(seed), n_rhs)
    rhs_list = []
    for key in keys:
        source = build_random_besov_function(
            form_degree,
            key=key,
            s=s,
            **BESOV_RHS_KWARGS,
        )

        if form_degree in (0, 3):
            # Scalar forms use scalar load data, while the random-field helper
            # returns shape-(1,) for k in {0,3}.
            load_source = lambda x, src=source: src(x)[0]
        else:
            load_source = source

        rhs_list.append(
            seq.load(
                load_source,
                form_degree,
                dirichlet=dirichlet,
                frame=load_frame,
            )
        )
    return jnp.stack(rhs_list, axis=0)


def _assemble_base_operators(seq: DeRhamSequence):
    timings: dict[str, float] = {}

    ops = seq.get_operators()
    timings["mass_ops_ms"] = 0.0

    t0 = time.perf_counter()
    ops = assemble_mass_jacobi_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3))
    timings["mass_jacobi_direct_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1, 2))
    timings["incidence_ops_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0, 1, 2, 3))
    timings["hodge_ops_ms"] = (time.perf_counter() - t0) * 1e3

    return ops, timings


def _assemble_rank_operators(seq: DeRhamSequence, base_ops, rank: int):
    ops = base_ops
    ops = assemble_tensor_mass_preconditioner(
        seq,
        operators=ops,
        ks=(0, 1, 2, 3),
        rank=rank,
    )
    ops = assemble_tensor_laplacian_preconditioner(
        seq,
        operators=ops,
        ks=(0,),
        rank=rank,
    )
    return seq.set_operators(ops)


def _time_solve(solve, rhs_batch):
    t_warm = time.perf_counter()
    u, info, residual = solve(rhs_batch[0])
    jax.block_until_ready((u, info, residual))
    warmup_ms = (time.perf_counter() - t_warm) * 1e3

    iters: list[int] = []
    times_ms: list[float] = []
    residuals: list[float] = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        u, info, residual = solve(rhs)
        jax.block_until_ready((u, info, residual))
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iters.append(abs(int(info)))
        residuals.append(float(residual))

    iters_arr = jnp.asarray(iters)
    times_arr = jnp.asarray(times_ms)
    batch_solve_ms = float(jnp.sum(times_arr))
    return {
        "avg_iters": float(jnp.mean(iters_arr)),
        "min_iters": int(min(iters)),
        "max_iters": int(max(iters)),
        "std_iters": float(jnp.std(iters_arr)),
        "avg_solve_ms": float(jnp.mean(times_arr)),
        "min_solve_ms": float(jnp.min(times_arr)),
        "max_solve_ms": float(jnp.max(times_arr)),
        "std_solve_ms": float(jnp.std(times_arr)),
        "final_residual": float(jnp.max(jnp.asarray(residuals))),
        "n_rhs": int(len(iters)),
        "warmup_ms": float(warmup_ms),
        "batch_solve_ms": float(batch_solve_ms),
        "batch_total_ms": float(warmup_ms + batch_solve_ms),
    }


def _finalize_method_timing(t0: float, stats: dict, timing: dict[str, float]) -> tuple[float, float]:
    method_total_ms = (time.perf_counter() - t0) * 1e3
    setup_overhead_ms = max(method_total_ms - float(stats["batch_total_ms"]), 0.0)
    timing["methods_total_ms"] += method_total_ms
    timing["warmup_total_ms"] += float(stats["warmup_ms"])
    timing["solve_batch_total_ms"] += float(stats["batch_solve_ms"])
    timing["setup_overhead_total_ms"] += setup_overhead_ms
    return setup_overhead_ms, method_total_ms


def _make_solve(seq, ops, *, k: int, dirichlet: bool, preconditioner, args):
    def solve(rhs):
        u, info = seq.apply_inverse_laplacian(
            rhs,
            k,
            dirichlet=dirichlet,
            operators=ops,
            tol=args.cg_tol,
            maxiter=args.cg_maxiter,
            preconditioner=preconditioner,
            return_info=True,
        )
        residual = seq.apply_laplacian(
            u,
            k,
            dirichlet=dirichlet,
            operators=ops,
        ) - rhs
        rhs_norm = jnp.linalg.norm(rhs)
        rel_res = jnp.linalg.norm(residual) / jnp.where(rhs_norm > 0, rhs_norm, 1.0)
        return u, info, rel_res

    return jax.jit(solve) if bool(getattr(args, "solve_jit", True)) else solve


def _make_mass_solve(seq, ops, *, k: int, dirichlet: bool, preconditioner, args):
    def solve(rhs):
        u, info = seq.apply_inverse_mass_matrix(
            rhs,
            k,
            dirichlet=dirichlet,
            operators=ops,
            tol=args.cg_tol,
            maxiter=args.cg_maxiter,
            preconditioner=preconditioner,
            return_info=True,
        )
        residual = seq.apply_mass_matrix(
            u,
            k,
            dirichlet=dirichlet,
            operators=ops,
        ) - rhs
        rhs_norm = jnp.linalg.norm(rhs)
        rel_res = jnp.linalg.norm(residual) / jnp.where(rhs_norm > 0, rhs_norm, 1.0)
        return u, info, rel_res

    return jax.jit(solve) if bool(getattr(args, "solve_jit", True)) else solve


def _build_saddle_spec(*, outer_kind: str, schur_diag_mode: str, outer_steps: int | None = None):
    outer_kwargs = {
        "kind": outer_kind,
        "schur_diag_mode": schur_diag_mode,
    }
    if outer_steps is not None:
        outer_kwargs["steps"] = int(outer_steps)
    return SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind="tensor"),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind="tensor"),
            outer=MassPreconditionerSpec(**outer_kwargs),
        ),
        coupled=False,
    )


def run_phase1(args, seq: DeRhamSequence | None = None) -> tuple[dict, list[Row]]:
    t_run0 = time.perf_counter()
    if getattr(args, "n", None) is not None:
        ns = (int(args.n), 2 * int(args.n), int(args.n))
    else:
        ns = tuple(int(v) for v in args.ns)

    mass_dirichlet = args.dirichlet.lower() == "dbc"
    laplace_dirichlet_by_k = _laplace_dirichlet_map(args)

    timing = {
        "sequence_ms": 0.0,
        "base_assembly_ms": 0.0,
        "rhs_laplace_ms": 0.0,
        "rhs_mass_ms": 0.0,
        "rank_assembly_ms": 0.0,
        "methods_total_ms": 0.0,
        "warmup_total_ms": 0.0,
        "solve_batch_total_ms": 0.0,
        "setup_overhead_total_ms": 0.0,
    }

    if seq is None:
        t0 = time.perf_counter()
        seq = _build_sequence(args, ns)
        timing["sequence_ms"] = (time.perf_counter() - t0) * 1e3
        _progress(f"sequence built in {timing['sequence_ms']:.1f} ms")
    else:
        expected_ps = (int(args.p), int(args.p), int(args.p))
        if tuple(seq.ns) != tuple(ns):
            raise ValueError(f"Reused sequence has ns={seq.ns}, expected {ns}")
        if tuple(seq.ps) != expected_ps:
            raise ValueError(f"Reused sequence has ps={seq.ps}, expected {expected_ps}")
        t0 = time.perf_counter()
        _update_sequence_map(seq, args)
        timing["sequence_ms"] = (time.perf_counter() - t0) * 1e3
        _progress(f"sequence geometry updated in {timing['sequence_ms']:.1f} ms")

    t0 = time.perf_counter()
    base_ops, base_timings = _assemble_base_operators(seq)
    timing["base_assembly_ms"] = (time.perf_counter() - t0) * 1e3
    timing["mass_ops_ms"] = float(base_timings.get("mass_ops_ms", 0.0))
    timing["mass_jacobi_direct_ms"] = float(base_timings.get("mass_jacobi_direct_ms", 0.0))
    timing["incidence_ops_ms"] = float(base_timings.get("incidence_ops_ms", 0.0))
    timing["hodge_ops_ms"] = float(base_timings.get("hodge_ops_ms", 0.0))
    _progress(
        "base operators assembled "
        f"total={timing['base_assembly_ms']:.1f} ms "
        f"(mass={timing['mass_ops_ms']:.1f}, "
        f"mass_jacobi_direct={timing['mass_jacobi_direct_ms']:.1f}, "
        f"incidence={timing['incidence_ops_ms']:.1f}, "
        f"hodge={timing['hodge_ops_ms']:.1f})"
    )

    laplace_ks = tuple(int(k) for k in args.laplace_ks)
    mass_ks = tuple(int(k) for k in args.mass_ks)
    tensor_ranks = tuple(int(r) for r in args.tensor_ranks)
    cheb_steps = tuple(int(s) for s in args.cheb_steps)
    schur_diag_modes = tuple(str(m) for m in args.schur_diag_modes)

    invalid_laplace = tuple(k for k in laplace_ks if k not in (0, 1, 2, 3))
    if invalid_laplace:
        raise ValueError(f"laplace_ks must be in 0,1,2,3; got {invalid_laplace}")
    invalid_mass = tuple(k for k in mass_ks if k not in (0, 1, 2, 3))
    if invalid_mass:
        raise ValueError(f"mass_ks must be in 0,1,2,3; got {invalid_mass}")
    if not tensor_ranks:
        raise ValueError("tensor_ranks cannot be empty")
    valid_schur_modes = ("tensor_probe", "exact_probe", "diag")
    invalid_schur = tuple(m for m in schur_diag_modes if m not in valid_schur_modes)
    if invalid_schur:
        raise ValueError(
            f"schur_diag_modes must be in {valid_schur_modes}; got {invalid_schur}"
        )

    no_jacobi = bool(getattr(args, 'no_jacobi', False))
    expected_rows = _expected_row_count(
        laplace_ks,
        mass_ks,
        tensor_ranks,
        cheb_steps,
        schur_diag_modes,
        no_jacobi=no_jacobi,
    )
    _progress(
        "benchmark start "
        f"ns={ns} p={int(args.p)} kappa={float(args.kappa)} "
        f"mass_dirichlet={mass_dirichlet} "
        f"laplace_bcs={tuple((k, _bc_label(laplace_dirichlet_by_k[k])) for k in laplace_ks)} "
        f"laplace_ks={laplace_ks} mass_ks={mass_ks} "
        f"tensor_ranks={tensor_ranks} cheb_steps={cheb_steps} "
        f"schur_diag_modes={schur_diag_modes} "
        f"n_rhs={int(args.n_rhs)} expected_rows={expected_rows}"
    )

    _progress("building sequence and base operators")

    rhs_laplace = {}
    for k in laplace_ks:
        laplace_dirichlet = laplace_dirichlet_by_k[int(k)]
        _progress(f"building Laplace RHS batch for k={k}")
        t0 = time.perf_counter()
        rhs_laplace[k] = _build_random_besov_rhs_batch_load(
            seq,
            form_degree=k,
            dirichlet=laplace_dirichlet,
            n_rhs=int(args.n_rhs),
            seed=int(args.seed),
            s=float(args.besov_s),
            load_frame=args.load_frame,
        )
        rhs_ms = (time.perf_counter() - t0) * 1e3
        timing["rhs_laplace_ms"] += rhs_ms
        _progress(f"built Laplace RHS batch for k={k} in {rhs_ms:.1f} ms")

    rhs_mass = {}
    for k in mass_ks:
        _progress(f"building mass RHS batch for k={k}")
        t0 = time.perf_counter()
        rhs_mass[k] = _build_random_besov_rhs_batch_load(
            seq,
            form_degree=int(k),
            dirichlet=mass_dirichlet,
            n_rhs=int(args.n_rhs),
            seed=int(args.seed),
            s=float(args.besov_s),
            load_frame=args.load_frame,
        )
        rhs_ms = (time.perf_counter() - t0) * 1e3
        timing["rhs_mass_ms"] += rhs_ms
        _progress(f"built mass RHS batch for k={k} in {rhs_ms:.1f} ms")

    rows: list[Row] = []
    emitted_rows = 0

    first_rank = tensor_ranks[0]
    for rank in tensor_ranks:
        _progress(f"assembling rank-dependent operators for rank={rank}")
        t_rank = time.perf_counter()
        ops = _assemble_rank_operators(seq, base_ops, rank)
        rank_ms = (time.perf_counter() - t_rank) * 1e3
        timing["rank_assembly_ms"] += rank_ms
        _progress(f"assembled rank={rank} operators in {rank_ms:.1f} ms")

        for k in laplace_ks:
            _progress(f"Laplace solves start rank={rank} k={k}")
            rhs_batch = rhs_laplace[k]
            laplace_dirichlet = laplace_dirichlet_by_k[int(k)]
            case_name = f"k{k}_{_bc_label(laplace_dirichlet)}"
            if k == 0:
                if rank == first_rank:
                    if not no_jacobi:
                        t0 = time.perf_counter()
                        solve = _make_solve(
                            seq, ops, k=0, dirichlet=laplace_dirichlet,
                            preconditioner="jacobi", args=args,
                        )
                        stats = _time_solve(solve, rhs_batch)
                        setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
                        rows.append(Row(
                            case=case_name,
                            strategy="k0_jacobi",
                            method="jacobi",
                            k=0,
                            dirichlet=laplace_dirichlet,
                            avg_iters=stats["avg_iters"],
                            max_iters=stats["max_iters"],
                            min_iters=stats["min_iters"],
                            std_iters=stats["std_iters"],
                            avg_solve_ms=stats["avg_solve_ms"],
                            min_solve_ms=stats["min_solve_ms"],
                            max_solve_ms=stats["max_solve_ms"],
                            std_solve_ms=stats["std_solve_ms"],
                            final_residual=stats["final_residual"],
                            setup_ms=max(setup_ms, 0.0),
                            schur_diag_mode="",
                            schur_outer_kind="",
                            schur_outer_steps=-1,
                            tensor_rank=-1,
                            cheb_steps=-1,
                            n_r=ns[0], n_t=ns[1], n_z=ns[2],
                            p=int(args.p), epsilon=float(args.epsilon),
                            n_rhs=int(args.n_rhs), seed=int(args.seed),
                            warmup_ms=float(stats["warmup_ms"]),
                            batch_solve_ms=float(stats["batch_solve_ms"]),
                            batch_total_ms=float(stats["batch_total_ms"]),
                            method_total_ms=float(method_total_ms),
                        ))
                        emitted_rows += 1

                    for steps in cheb_steps:
                        t0 = time.perf_counter()
                        solve = _make_solve(
                            seq,
                            ops,
                            k=0,
                            dirichlet=laplace_dirichlet,
                            preconditioner=MassPreconditionerSpec(
                                kind="chebyshev",
                                steps=int(steps),
                                smoother=MassPreconditionerSpec(kind="jacobi"),
                            ),
                            args=args,
                        )
                        stats = _time_solve(solve, rhs_batch)
                        setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
                        rows.append(Row(
                            case=case_name,
                            strategy=f"k0_chebyshev_s{int(steps)}",
                            method=f"chebyshev({int(steps)})",
                            k=0,
                            dirichlet=laplace_dirichlet,
                            avg_iters=stats["avg_iters"],
                            max_iters=stats["max_iters"],
                            min_iters=stats["min_iters"],
                            std_iters=stats["std_iters"],
                            avg_solve_ms=stats["avg_solve_ms"],
                            min_solve_ms=stats["min_solve_ms"],
                            max_solve_ms=stats["max_solve_ms"],
                            std_solve_ms=stats["std_solve_ms"],
                            final_residual=stats["final_residual"],
                            setup_ms=max(setup_ms, 0.0),
                            schur_diag_mode="",
                            schur_outer_kind="chebyshev",
                            schur_outer_steps=int(steps),
                            tensor_rank=-1,
                            cheb_steps=int(steps),
                            n_r=ns[0], n_t=ns[1], n_z=ns[2],
                            p=int(args.p), epsilon=float(args.epsilon),
                            n_rhs=int(args.n_rhs), seed=int(args.seed),
                            warmup_ms=float(stats["warmup_ms"]),
                            batch_solve_ms=float(stats["batch_solve_ms"]),
                            batch_total_ms=float(stats["batch_total_ms"]),
                            method_total_ms=float(method_total_ms),
                        ))
                        emitted_rows += 1

                t0 = time.perf_counter()
                solve = _make_solve(
                    seq,
                    ops,
                    k=0,
                    dirichlet=laplace_dirichlet,
                    preconditioner="tensor",
                    args=args,
                )
                stats = _time_solve(solve, rhs_batch)
                setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
                rows.append(Row(
                    case=case_name,
                    strategy=f"k0_tensor_r{rank}",
                    method=f"tensor(r={rank})",
                    k=0,
                    dirichlet=laplace_dirichlet,
                    avg_iters=stats["avg_iters"],
                    max_iters=stats["max_iters"],
                    min_iters=stats["min_iters"],
                    std_iters=stats["std_iters"],
                    avg_solve_ms=stats["avg_solve_ms"],
                    min_solve_ms=stats["min_solve_ms"],
                    max_solve_ms=stats["max_solve_ms"],
                    std_solve_ms=stats["std_solve_ms"],
                    final_residual=stats["final_residual"],
                    setup_ms=max(setup_ms, 0.0),
                    schur_diag_mode="",
                    schur_outer_kind="",
                    schur_outer_steps=-1,
                    tensor_rank=rank,
                    cheb_steps=-1,
                    n_r=ns[0], n_t=ns[1], n_z=ns[2],
                    p=int(args.p), epsilon=float(args.epsilon),
                    n_rhs=int(args.n_rhs), seed=int(args.seed),
                    warmup_ms=float(stats["warmup_ms"]),
                    batch_solve_ms=float(stats["batch_solve_ms"]),
                    batch_total_ms=float(stats["batch_total_ms"]),
                    method_total_ms=float(method_total_ms),
                ))
                emitted_rows += 1
            else:
                for schur_mode in schur_diag_modes:
                    if getattr(args, "preassemble_schur_diags", False):
                        _progress(
                            f"preassembling Schur Jacobi diagonal mode={schur_mode} "
                            f"for k={k} dirichlet={laplace_dirichlet}"
                        )
                        ops = assemble_schur_jacobi_preconditioner(
                            seq,
                            operators=ops,
                            ks=(k,),
                            dirichlet_variants=(laplace_dirichlet,),
                            schur_diag_mode=schur_mode,
                        )

                    mode_short = {
                        "diag": "diag",
                        "tensor_probe": "tp",
                        "exact_probe": "ex",
                    }[schur_mode]
                    if not no_jacobi:
                        t0 = time.perf_counter()
                        solve = _make_solve(
                            seq,
                            ops,
                            k=k,
                            dirichlet=laplace_dirichlet,
                            preconditioner=_build_saddle_spec(
                                outer_kind="jacobi",
                                schur_diag_mode=schur_mode,
                                outer_steps=None,
                            ),
                            args=args,
                        )
                        stats = _time_solve(solve, rhs_batch)
                        setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
                        rows.append(Row(
                            case=case_name,
                            strategy=f"k{k}_jacobi_{mode_short}_r{rank}",
                            method=f"jacobi({mode_short},r={rank})",
                            k=k,
                            dirichlet=laplace_dirichlet,
                            avg_iters=stats["avg_iters"],
                            max_iters=stats["max_iters"],
                            min_iters=stats["min_iters"],
                            std_iters=stats["std_iters"],
                            avg_solve_ms=stats["avg_solve_ms"],
                            min_solve_ms=stats["min_solve_ms"],
                            max_solve_ms=stats["max_solve_ms"],
                            std_solve_ms=stats["std_solve_ms"],
                            final_residual=stats["final_residual"],
                            setup_ms=max(setup_ms, 0.0),
                            schur_diag_mode=schur_mode,
                            schur_outer_kind="jacobi",
                            schur_outer_steps=-1,
                            tensor_rank=rank,
                            cheb_steps=-1,
                            n_r=ns[0], n_t=ns[1], n_z=ns[2],
                            p=int(args.p), epsilon=float(args.epsilon),
                            n_rhs=int(args.n_rhs), seed=int(args.seed),
                            warmup_ms=float(stats["warmup_ms"]),
                            batch_solve_ms=float(stats["batch_solve_ms"]),
                            batch_total_ms=float(stats["batch_total_ms"]),
                            method_total_ms=float(method_total_ms),
                        ))
                        emitted_rows += 1

                    for steps in cheb_steps:
                        t0 = time.perf_counter()
                        solve = _make_solve(
                            seq,
                            ops,
                            k=k,
                            dirichlet=laplace_dirichlet,
                            preconditioner=_build_saddle_spec(
                                outer_kind="chebyshev",
                                schur_diag_mode=schur_mode,
                                outer_steps=int(steps),
                            ),
                            args=args,
                        )
                        stats = _time_solve(solve, rhs_batch)
                        setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
                        rows.append(Row(
                            case=case_name,
                            strategy=f"k{k}_chebyshev_s{int(steps)}_{mode_short}_r{rank}",
                            method=f"chebyshev({int(steps)},{mode_short},r={rank})",
                            k=k,
                            dirichlet=laplace_dirichlet,
                            avg_iters=stats["avg_iters"],
                            max_iters=stats["max_iters"],
                            min_iters=stats["min_iters"],
                            std_iters=stats["std_iters"],
                            avg_solve_ms=stats["avg_solve_ms"],
                            min_solve_ms=stats["min_solve_ms"],
                            max_solve_ms=stats["max_solve_ms"],
                            std_solve_ms=stats["std_solve_ms"],
                            final_residual=stats["final_residual"],
                            setup_ms=max(setup_ms, 0.0),
                            schur_diag_mode=schur_mode,
                            schur_outer_kind="chebyshev",
                            schur_outer_steps=int(steps),
                            tensor_rank=rank,
                            cheb_steps=int(steps),
                            n_r=ns[0], n_t=ns[1], n_z=ns[2],
                            p=int(args.p), epsilon=float(args.epsilon),
                            n_rhs=int(args.n_rhs), seed=int(args.seed),
                            warmup_ms=float(stats["warmup_ms"]),
                            batch_solve_ms=float(stats["batch_solve_ms"]),
                            batch_total_ms=float(stats["batch_total_ms"]),
                            method_total_ms=float(method_total_ms),
                        ))
                        emitted_rows += 1

            _progress(
                f"Laplace solves done rank={rank} k={k} "
                f"rows_emitted={emitted_rows}/{expected_rows}"
            )

        for k in mass_ks:
            _progress(f"mass solves start rank={rank} k={k}")
            rhs_batch = rhs_mass[k]
            case_name = f"M{k}_{_bc_label(mass_dirichlet)}"

            if rank == first_rank and not no_jacobi:
                t0 = time.perf_counter()
                solve = _make_mass_solve(
                    seq, ops, k=k, dirichlet=mass_dirichlet,
                    preconditioner="jacobi", args=args,
                )
                stats = _time_solve(solve, rhs_batch)
                setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
                rows.append(Row(
                    case=case_name,
                    strategy=f"M{k}_jacobi",
                    method="mass_jacobi",
                    k=k,
                    dirichlet=mass_dirichlet,
                    avg_iters=stats["avg_iters"],
                    max_iters=stats["max_iters"],
                    min_iters=stats["min_iters"],
                    std_iters=stats["std_iters"],
                    avg_solve_ms=stats["avg_solve_ms"],
                    min_solve_ms=stats["min_solve_ms"],
                    max_solve_ms=stats["max_solve_ms"],
                    std_solve_ms=stats["std_solve_ms"],
                    final_residual=stats["final_residual"],
                    setup_ms=max(setup_ms, 0.0),
                    schur_diag_mode="",
                    schur_outer_kind="",
                    schur_outer_steps=-1,
                    tensor_rank=-1,
                    cheb_steps=-1,
                    n_r=ns[0], n_t=ns[1], n_z=ns[2],
                    p=int(args.p), epsilon=float(args.epsilon),
                    n_rhs=int(args.n_rhs), seed=int(args.seed),
                    warmup_ms=float(stats["warmup_ms"]),
                    batch_solve_ms=float(stats["batch_solve_ms"]),
                    batch_total_ms=float(stats["batch_total_ms"]),
                    method_total_ms=float(method_total_ms),
                ))
                emitted_rows += 1

            t0 = time.perf_counter()
            solve = _make_mass_solve(
                seq, ops, k=k, dirichlet=mass_dirichlet,
                preconditioner="tensor", args=args,
            )
            stats = _time_solve(solve, rhs_batch)
            setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
            rows.append(Row(
                case=case_name,
                strategy=f"M{k}_tensor_r{rank}",
                method=f"mass_tensor(r={rank})",
                k=k,
                dirichlet=mass_dirichlet,
                avg_iters=stats["avg_iters"],
                max_iters=stats["max_iters"],
                min_iters=stats["min_iters"],
                std_iters=stats["std_iters"],
                avg_solve_ms=stats["avg_solve_ms"],
                min_solve_ms=stats["min_solve_ms"],
                max_solve_ms=stats["max_solve_ms"],
                std_solve_ms=stats["std_solve_ms"],
                final_residual=stats["final_residual"],
                setup_ms=max(setup_ms, 0.0),
                schur_diag_mode="",
                schur_outer_kind="",
                schur_outer_steps=-1,
                tensor_rank=rank,
                cheb_steps=-1,
                n_r=ns[0], n_t=ns[1], n_z=ns[2],
                p=int(args.p), epsilon=float(args.epsilon),
                n_rhs=int(args.n_rhs), seed=int(args.seed),
                warmup_ms=float(stats["warmup_ms"]),
                batch_solve_ms=float(stats["batch_solve_ms"]),
                batch_total_ms=float(stats["batch_total_ms"]),
                method_total_ms=float(method_total_ms),
            ))
            emitted_rows += 1

            for steps in cheb_steps:
                t0 = time.perf_counter()
                solve = _make_mass_solve(
                    seq,
                    ops,
                    k=k,
                    dirichlet=mass_dirichlet,
                    preconditioner=MassPreconditionerSpec(kind="chebyshev", steps=int(steps)),
                    args=args,
                )
                stats = _time_solve(solve, rhs_batch)
                setup_ms, method_total_ms = _finalize_method_timing(t0, stats, timing)
                rows.append(Row(
                    case=case_name,
                    strategy=f"M{k}_chebyshev_s{int(steps)}_r{rank}",
                    method=f"mass_chebyshev({int(steps)},r={rank})",
                    k=k,
                    dirichlet=mass_dirichlet,
                    avg_iters=stats["avg_iters"],
                    max_iters=stats["max_iters"],
                    min_iters=stats["min_iters"],
                    std_iters=stats["std_iters"],
                    avg_solve_ms=stats["avg_solve_ms"],
                    min_solve_ms=stats["min_solve_ms"],
                    max_solve_ms=stats["max_solve_ms"],
                    std_solve_ms=stats["std_solve_ms"],
                    final_residual=stats["final_residual"],
                    setup_ms=max(setup_ms, 0.0),
                    schur_diag_mode="",
                    schur_outer_kind="chebyshev",
                    schur_outer_steps=int(steps),
                    tensor_rank=rank,
                    cheb_steps=int(steps),
                    n_r=ns[0], n_t=ns[1], n_z=ns[2],
                    p=int(args.p), epsilon=float(args.epsilon),
                    n_rhs=int(args.n_rhs), seed=int(args.seed),
                    warmup_ms=float(stats["warmup_ms"]),
                    batch_solve_ms=float(stats["batch_solve_ms"]),
                    batch_total_ms=float(stats["batch_total_ms"]),
                    method_total_ms=float(method_total_ms),
                ))
                emitted_rows += 1

            _progress(
                f"mass solves done rank={rank} k={k} "
                f"rows_emitted={emitted_rows}/{expected_rows}"
            )

    rows.sort(key=lambda row: (row.k, row.strategy))
    for row in rows:
        row.kappa = float(args.kappa)

    _progress(
        f"benchmark complete rows={len(rows)} expected_rows={expected_rows} "
        f"elapsed_s={time.perf_counter() - t_run0:.1f}"
    )

    total_ms = (time.perf_counter() - t_run0) * 1e3
    timing["total_ms"] = total_ms
    timing["untimed_other_ms"] = max(
        total_ms
        - (
            timing["sequence_ms"]
            + timing["base_assembly_ms"]
            + timing["rhs_laplace_ms"]
            + timing["rhs_mass_ms"]
            + timing["rank_assembly_ms"]
            + timing["methods_total_ms"]
        ),
        0.0,
    )

    _progress(
        "timing summary ms: "
        f"sequence={timing['sequence_ms']:.1f}, "
        f"base={timing['base_assembly_ms']:.1f}, "
        f"mass={timing['mass_ops_ms']:.1f}, "
        f"mass_jacobi_direct={timing['mass_jacobi_direct_ms']:.1f}, "
        f"incidence={timing['incidence_ops_ms']:.1f}, "
        f"hodge={timing['hodge_ops_ms']:.1f}, "
        f"rhs_laplace={timing['rhs_laplace_ms']:.1f}, "
        f"rhs_mass={timing['rhs_mass_ms']:.1f}, "
        f"rank_assembly={timing['rank_assembly_ms']:.1f}, "
        f"methods={timing['methods_total_ms']:.1f}, "
        f"warmup={timing['warmup_total_ms']:.1f}, "
        f"solve_batch={timing['solve_batch_total_ms']:.1f}, "
        f"setup_overhead={timing['setup_overhead_total_ms']:.1f}, "
        f"other={timing['untimed_other_ms']:.1f}, total={timing['total_ms']:.1f}"
    )

    summary = {
        "ns": ns,
        "p": int(args.p),
        "epsilon": float(args.epsilon),
        "kappa": float(args.kappa),
        "dirichlet": bool(mass_dirichlet),
        "laplace_dirichlet_by_k": {str(k): bool(v) for k, v in laplace_dirichlet_by_k.items()},
        "laplace_ks": laplace_ks,
        "mass_ks": mass_ks,
        "tensor_ranks": tensor_ranks,
        "cheb_steps": cheb_steps,
        "rows": len(rows),
        "timing_ms": timing,
    }
    return summary, rows


def run_phase2_sweep(args) -> tuple[dict, list[Row], list[dict]]:
    n_values = tuple(int(v) for v in args.n_list)
    p_values = tuple(int(v) for v in args.p_list)
    kappas = tuple(float(v) for v in args.kappas)
    sweep_style = str(getattr(args, "sweep_style", "cartesian")).lower()

    if not n_values:
        raise ValueError("n_list cannot be empty in sweep mode")
    if not p_values:
        raise ValueError("p_list cannot be empty in sweep mode")
    if not kappas:
        raise ValueError("kappas cannot be empty in sweep mode")
    if sweep_style not in ("cartesian", "ofat"):
        raise ValueError(f"sweep_style must be 'cartesian' or 'ofat'; got {sweep_style!r}")

    cheb_values = tuple(int(v) for v in args.cheb_steps)
    if not cheb_values and getattr(args, "cheb_ref", None) is not None:
        raise ValueError("cheb_ref was set but cheb_steps is empty")

    n_ref = int(args.n_ref) if getattr(args, "n_ref", None) is not None else int(n_values[0])
    p_ref = int(args.p_ref) if getattr(args, "p_ref", None) is not None else int(p_values[0])
    kappa_ref = (
        float(args.kappa_ref)
        if getattr(args, "kappa_ref", None) is not None
        else float(kappas[0])
    )
    cheb_ref = None
    if cheb_values:
        cheb_ref = (
            int(args.cheb_ref)
            if getattr(args, "cheb_ref", None) is not None
            else int(cheb_values[0])
        )

    if n_ref not in n_values:
        raise ValueError(f"n_ref={n_ref} must be present in n_list={n_values}")
    if p_ref not in p_values:
        raise ValueError(f"p_ref={p_ref} must be present in p_list={p_values}")
    if kappa_ref not in kappas:
        raise ValueError(f"kappa_ref={kappa_ref} must be present in kappas={kappas}")
    if cheb_values and cheb_ref not in cheb_values:
        raise ValueError(f"cheb_ref={cheb_ref} must be present in cheb_steps={cheb_values}")

    cell_specs: list[tuple[int, int, float, tuple[int, ...], str]] = []
    if sweep_style == "cartesian":
        for n in n_values:
            for p in p_values:
                for kappa in kappas:
                    cell_specs.append((int(n), int(p), float(kappa), cheb_values, "cartesian"))
    else:
        # OFAT: one baseline cell (with full cheb sweep), then vary one axis at a time.
        cell_specs.append((n_ref, p_ref, kappa_ref, cheb_values, "ref+cheb"))
        for n in n_values:
            if int(n) != n_ref:
                cheb = (cheb_ref,) if cheb_ref is not None else ()
                cell_specs.append((int(n), p_ref, kappa_ref, cheb, "vary_n"))
        for p in p_values:
            if int(p) != p_ref:
                cheb = (cheb_ref,) if cheb_ref is not None else ()
                cell_specs.append((n_ref, int(p), kappa_ref, cheb, "vary_p"))
        for kappa in kappas:
            if float(kappa) != kappa_ref:
                cheb = (cheb_ref,) if cheb_ref is not None else ()
                cell_specs.append((n_ref, p_ref, float(kappa), cheb, "vary_kappa"))

    all_rows: list[Row] = []
    all_results: list[dict] = []
    seq_cache: dict[tuple[tuple[int, int, int], int], DeRhamSequence] = {}

    for n, p, kappa, cheb_steps, tag in cell_specs:
        ns = (int(n), 2 * int(n), int(n))
        cache_key = (ns, int(p))
        local_args = argparse.Namespace(
            n=int(n),
            p=int(p),
            epsilon=float(args.epsilon),
            kappa=float(kappa),
            r0=float(args.r0),
            nfp=int(args.nfp),
            dirichlet=args.dirichlet,
            n_rhs=int(args.n_rhs),
            seed=int(args.seed),
            besov_s=float(args.besov_s),
            load_frame=args.load_frame,
            cg_tol=float(args.cg_tol),
            cg_maxiter=int(args.cg_maxiter),
            laplace_ks=tuple(int(v) for v in args.laplace_ks),
            tensor_ranks=tuple(int(v) for v in args.tensor_ranks),
            cheb_steps=tuple(int(v) for v in cheb_steps),
            mass_ks=tuple(int(v) for v in args.mass_ks),
            schur_diag_modes=tuple(str(v) for v in args.schur_diag_modes),
            preassemble_schur_diags=bool(getattr(args, "preassemble_schur_diags", False)),
            nullspace_free_laplace_bcs=bool(getattr(args, "nullspace_free_laplace_bcs", False)),
            out="",
            json_out="",
        )
        print(
            f"Running sweep cell [{tag}] n={n} p={p} kappa={kappa} "
            f"cheb_steps={local_args.cheb_steps} tensor_ranks={local_args.tensor_ranks}",
            flush=True,
        )
        seq = seq_cache.get(cache_key)
        if seq is None:
            seq = _build_sequence(local_args, ns)
            seq_cache[cache_key] = seq
            print(
                f"Reusing sequence cache: created ns={ns} p={p}",
                flush=True,
            )
        else:
            print(
                f"Reusing sequence cache: updating geometry for ns={ns} p={p} kappa={kappa}",
                flush=True,
            )
        summary, rows = run_phase1(local_args, seq=seq)
        summary["sweep_tag"] = tag
        all_results.append(summary)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows produced in sweep mode; check n_list, p_list, kappas")

    all_rows.sort(key=lambda row: (row.n_r, row.n_t, row.n_z, row.p, row.kappa, row.k, row.strategy))
    summary = {
        "mode": "sweep",
        "sweep_style": sweep_style,
        "n_list": n_values,
        "p_list": p_values,
        "kappas": kappas,
        "n_ref": n_ref,
        "p_ref": p_ref,
        "kappa_ref": kappa_ref,
        "cheb_ref": cheb_ref,
        "cheb_steps": cheb_values,
        "rows": len(all_rows),
        "cells": len(all_results),
    }
    return summary, all_rows, all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=parse_ns, default=(8, 16, 8))
    parser.add_argument("--n", type=int, default=None,
                        help="Optional shorthand for ns=(n,2n,n).")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--kappa", type=float, default=1.25)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--dirichlet", choices=("dbc", "nbc"), default="dbc")
    parser.add_argument("--n-rhs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--besov-s", type=float, default=1.0)
    parser.add_argument("--load-frame", choices=("ref", "phys"), default="ref")
    parser.add_argument("--cg-tol", type=float, default=1e-9)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument(
        "--solve-jit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JIT-compile per-method solve wrappers (default: enabled).",
    )
    parser.add_argument(
        "--no-jacobi",
        action="store_true",
        default=False,
        help="Skip Jacobi rows (k0 scalar, k>=1 schur outer, mass). Only tensor and Chebyshev rows are emitted.",
    )
    parser.add_argument("--laplace-ks", type=parse_int_list, default=(0, 1, 2, 3))
    parser.add_argument("--tensor-ranks", type=parse_int_list, default=(1, 2, 3))
    parser.add_argument(
        "--cheb-steps",
        type=parse_int_list,
        default=(),
        help="Comma-separated Chebyshev step counts. Empty disables Chebyshev methods.",
    )
    # Backward-compatible aliases: if set explicitly, map to the new sweep args.
    parser.add_argument("--tensor-rank", type=int, default=None)
    parser.add_argument("--k0-cheb-steps", type=int, default=None)
    parser.add_argument("--k1-cheb-tensor-steps", type=int, default=None)
    parser.add_argument("--mass-ks", type=parse_int_list, default=(0, 1, 2, 3))
    parser.add_argument(
        "--schur-diag-modes",
        type=_parse_mode_list,
        default=("diag", "tensor_probe", "exact_probe"),
        help="Comma-separated Schur Jacobi assembly modes for k>0: diag,tensor_probe,exact_probe.",
    )
    parser.add_argument(
        "--preassemble-schur-diags",
        action="store_true",
        help="Precompute Schur Jacobi diagonals during assembly for the selected --schur-diag-modes.",
    )
    parser.add_argument(
        "--nullspace-free-laplace-bcs",
        action="store_true",
        help="Override Laplace BCs by degree to avoid nullspace cases: k0/k1=DBC, k2/k3=NBC.",
    )
    parser.add_argument("--n-list", type=parse_int_list, default=(),
                        help="Sweep-mode n values (phase2-style).")
    parser.add_argument("--p-list", type=parse_int_list, default=(),
                        help="Sweep-mode p values (phase2-style).")
    parser.add_argument("--kappas", type=_parse_float_list, default=(),
                        help="Sweep-mode kappa values (phase2-style).")
    parser.add_argument("--sweep-style", choices=("cartesian", "ofat"), default="cartesian",
                        help="Sweep mode: full cartesian product or one-factor-at-a-time.")
    parser.add_argument("--n-ref", type=int, default=None,
                        help="OFAT reference n (must be in --n-list).")
    parser.add_argument("--p-ref", type=int, default=None,
                        help="OFAT reference p (must be in --p-list).")
    parser.add_argument("--kappa-ref", type=float, default=None,
                        help="OFAT reference kappa (must be in --kappas).")
    parser.add_argument("--cheb-ref", type=int, default=None,
                        help="OFAT reference chebyshev steps (must be in --cheb-steps).")
    parser.add_argument("--out", type=str, default="",
                        help="Optional CSV output path.")
    parser.add_argument("--json-out", type=str, default="",
                        help="Optional JSON output path.")
    args = parser.parse_args()

    if args.tensor_rank is not None:
        args.tensor_ranks = (int(args.tensor_rank),)
    if args.k0_cheb_steps is not None or args.k1_cheb_tensor_steps is not None:
        legacy_steps = []
        if args.k0_cheb_steps is not None:
            legacy_steps.append(int(args.k0_cheb_steps))
        if args.k1_cheb_tensor_steps is not None:
            legacy_steps.append(int(args.k1_cheb_tensor_steps))
        args.cheb_steps = tuple(sorted(set(legacy_steps)))

    sweep_mode = bool(args.n_list or args.p_list or args.kappas)
    if sweep_mode:
        if not args.n_list:
            args.n_list = (int(args.n),) if args.n is not None else (int(args.ns[0]),)
        if not args.p_list:
            args.p_list = (int(args.p),)
        if not args.kappas:
            args.kappas = (float(args.kappa),)
        summary, rows, sweep_results = run_phase2_sweep(args)
        print(
            f"Unified sweep benchmark style={summary['sweep_style']} "
            f"cells={summary['cells']} rows={summary['rows']} "
            f"n_list={summary['n_list']} p_list={summary['p_list']} kappas={summary['kappas']}"
        )
    else:
        summary, rows = run_phase1(args)
        sweep_results = None
        print(
            f"Preconditioner benchmark ns={summary['ns']} p={summary['p']} "
            f"dirichlet={summary['dirichlet']} rows={summary['rows']}"
        )

    header = f"{'case':>8s}  {'method':>24s}  {'avg_it':>7s}  {'max_it':>7s}  {'solve_ms':>9s}  {'res':>10s}"
    print(header)
    for row in rows:
        print(
            f"{row.case:>8s}  {row.method:>24s}  {row.avg_iters:7.1f}  {row.max_iters:7d}  "
            f"{row.avg_solve_ms:9.2f}  {row.final_residual:10.2e}"
        )

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as fh:
            payload = {"summary": summary, "rows": [asdict(row) for row in rows]}
            if sweep_results is not None:
                payload["cell_summaries"] = sweep_results
            json.dump(payload, fh, indent=2)
        print(f"Wrote {json_path}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
