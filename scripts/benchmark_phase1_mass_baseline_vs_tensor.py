"""Phase 1 block benchmark: baseline vs tensor preconditioner.

Builds the de Rham sequence and base operators **once** at the
reference point, then for every requested block runs a full Cartesian
sweep over preconditioner configurations against a common batch of
random Besov RHS vectors.

Mass configurations per ``k``:

- ``jacobi``  : pointwise inverse of the assembled mass diagonal.
- ``cheb_J(s)``: Chebyshev polynomial of degree ``s`` in the mass
  operator with Jacobi as the inner smoother (steps in {2, 3, 5}).
- ``tensor(rank=r, bulk_cheb=t)``: production tensor mass
  preconditioner with CP rank ``r`` and bulk Chebyshev steps ``t``;
  ``t = 0`` disables the inner Chebyshev acceleration on the tensor
  blocks.  ``r in {1, 2, 3, 5}``, ``t in {0, 2, 3}``.
- ``tensor(..., inner_schur=on)``: additional tensor variant for
    ``k in {1,2}`` that enables the inner RT/zeta Schur option.

Optional extra block:

- ``K0_dbc``: scalar stiffness with Dirichlet data, benchmarked with
    Jacobi, Chebyshev(Jacobi), and the tensor scalar-Hodge preconditioner.

Solver: ``solve_singular_cg`` with ``tol=1e-12``, ``maxiter=1000``.
Boundaries: free (``dirichlet=False``).  Geometry: toroid.

Output:
- terminal table (one row per cell, sorted by k then strategy).
- CSV at ``--out`` with the full schema for the aggregator.

Reference defaults match the agreed Phase 1 anchor:
``ns=(16,32,16)``, ``p=3``, ``eps=0.2``, ``s=1`` Besov regularity.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    apply_hodge_laplacian_preconditioner,
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    apply_mass_tensor_preconditioner_ops,
    apply_stiffness,
    assemble_hodge_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    _build_chebyshev_apply_preconditioner,
    _estimate_chebyshev_lanczos_bounds_apply,
)
from mrx.preconditioners import MassPreconditionerSpec
from mrx.solvers import solve_singular_cg
from mrx.io import parse_int_list, parse_ns
from mrx.utils import build_random_besov_rhs_batch


jax.config.update("jax_enable_x64", True)


TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)

# Reference Phase 1 anchor.
DEFAULT_NS = (16, 32, 16)
DEFAULT_P = 3
DEFAULT_EPS = 0.2
DEFAULT_KAPPA = 1.25
DEFAULT_R0 = 1.0
DEFAULT_NFP = 3
DEFAULT_BESOV_S = 1.0
DEFAULT_TOL = 1e-12
DEFAULT_MAXITER = 1000
DEFAULT_N_RHS = 8

CHEB_BASELINE_STEPS = (2, 3, 5)
TENSOR_RANKS = (1, 2, 3, 5)
TENSOR_BULK_CHEB_STEPS = (0, 2, 3)
K0_STIFFNESS_DBC_CASE = "K0_dbc"


BESOV_RHS_KWARGS = {
    "upper_limit": 24,
    "num_modes": 64,
    "scale": 1.0,
    "smoothness_margin": 0.0,
    "normalization_samples": 256,
}


@dataclass
class Row:
    k: int
    strategy: str
    method: str  # short label for CSV
    avg_iters: float
    max_iters: int
    case: str = ""
    min_iters: int = -1
    std_iters: float = 0.0
    avg_solve_ms: float = 0.0
    min_solve_ms: float = 0.0
    max_solve_ms: float = 0.0
    std_solve_ms: float = 0.0
    setup_ms: float = 0.0
    final_residual: float = 0.0
    rank: int = -1
    cheb_steps: int = -1
    bulk_cheb_steps: int = -1
    inner_schur: bool = False
    n_rhs: int = 0
    seed: int = 0
    n_r: int = 0
    n_t: int = 0
    n_z: int = 0
    p: int = 0
    eps: float = 0.0
    kappa: float = 1.0
    nfp: int = 1
    besov_s: float = 0.0
    dirichlet: bool = False
    hyperparams: str = ""


# --------------------------------------------------------------------------- #
# Sequence builder                                                             #
# --------------------------------------------------------------------------- #


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
            eps=args.eps, kappa=args.kappa, R0=args.r0, nfp=args.nfp,
        )
    )
    return seq


# --------------------------------------------------------------------------- #
# Cell runners                                                                 #
# --------------------------------------------------------------------------- #


def time_solve(solve, rhs_batch):
    # warmup
    x, it = solve(rhs_batch[0])
    jax.block_until_ready(x)

    iters: list[int] = []
    times_ms: list[float] = []
    last_residuals: list[float] = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        x, info = solve(rhs)
        jax.block_until_ready(x)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iters.append(int(info[0]))
        last_residuals.append(float(info[1]))
    iters_arr = jnp.asarray(iters)
    times_arr = jnp.asarray(times_ms)
    return {
        "avg_iters": float(jnp.mean(iters_arr)),
        "min_iters": int(min(iters)),
        "max_iters": int(max(iters)),
        "std_iters": float(jnp.std(iters_arr)),
        "avg_solve_ms": float(jnp.mean(times_arr)),
        "min_solve_ms": float(jnp.min(times_arr)),
        "max_solve_ms": float(jnp.max(times_arr)),
        "std_solve_ms": float(jnp.std(times_arr)),
        "residual": float(jnp.max(jnp.asarray(last_residuals))),
    }


def _mass_case_name(k: int) -> str:
    return f"M{k}"


def _case_sort_key(case: str) -> tuple[int, str]:
    if case == K0_STIFFNESS_DBC_CASE:
        return (4, case)
    if case.startswith("M") and case[1:].isdigit():
        return (int(case[1:]), case)
    return (99, case)


def make_solve(operator_apply, precond_apply, args, *, mass_apply=None):
    if mass_apply is None:
        mass_apply = operator_apply

    @jax.jit
    def solve(rhs):
        x, info = solve_singular_cg(
            operator_apply,
            rhs,
            mass_matvec=mass_apply,
            precond_matvec=precond_apply,
            tol=args.tol,
            maxiter=args.maxiter,
        )
        rhs_norm = jnp.linalg.norm(rhs)
        residual = operator_apply(x) - rhs
        rel = jnp.linalg.norm(residual) / jnp.where(rhs_norm > 0, rhs_norm, 1.0)
        return x, jnp.array([jnp.abs(info), rel])
    return solve


def run_jacobi(seq, operators, k, rhs_batch, args, *, dirichlet) -> Row:
    operator_apply = lambda x: apply_mass_matrix(
        seq, operators, x, k, dirichlet=dirichlet,
    )
    precond_apply = lambda rhs: apply_mass_matrix_preconditioner(
        seq, operators, rhs, k, dirichlet=dirichlet, kind="jacobi",
    )
    t0 = time.perf_counter()
    solve = make_solve(operator_apply, precond_apply, args)
    stats = time_solve(solve, rhs_batch)
    setup_ms = (time.perf_counter() - t0) * 1e3 - stats["avg_solve_ms"]
    return Row(
        case=_mass_case_name(k),
        k=k, strategy="jacobi", method="jacobi",
        avg_iters=stats["avg_iters"], max_iters=stats["max_iters"],
        min_iters=stats["min_iters"], std_iters=stats["std_iters"],
        avg_solve_ms=stats["avg_solve_ms"],
        min_solve_ms=stats["min_solve_ms"],
        max_solve_ms=stats["max_solve_ms"],
        std_solve_ms=stats["std_solve_ms"],
        setup_ms=max(setup_ms, 0.0),
        final_residual=stats["residual"],
        hyperparams=json.dumps({}),
    )


def run_chebyshev(seq, operators, k, rhs_batch, args, *, dirichlet, steps) -> Row:
    operator_apply = lambda x: apply_mass_matrix(
        seq, operators, x, k, dirichlet=dirichlet,
    )
    jacobi_apply = lambda rhs: apply_mass_matrix_preconditioner(
        seq, operators, rhs, k, dirichlet=dirichlet, kind="jacobi",
    )
    suffix = "_dbc" if dirichlet else ""
    dof = int(getattr(seq, f"n{k}{suffix}"))
    spec = MassPreconditionerSpec(
        kind="chebyshev",
        steps=steps,
        smoother=MassPreconditionerSpec(kind="jacobi"),
    )
    t_setup = time.perf_counter()
    min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply, jacobi_apply, dof, spec=spec, seed=args.seed,
    )
    precond_apply = _build_chebyshev_apply_preconditioner(
        operator_apply, jacobi_apply,
        steps=steps, min_eig=min_eig, max_eig=max_eig,
    )
    setup_ms = (time.perf_counter() - t_setup) * 1e3
    solve = make_solve(operator_apply, precond_apply, args)
    stats = time_solve(solve, rhs_batch)
    return Row(
        case=_mass_case_name(k),
        k=k, strategy="chebyshev", method=f"cheb_J(s={steps})",
        avg_iters=stats["avg_iters"], max_iters=stats["max_iters"],
        min_iters=stats["min_iters"], std_iters=stats["std_iters"],
        avg_solve_ms=stats["avg_solve_ms"],
        min_solve_ms=stats["min_solve_ms"],
        max_solve_ms=stats["max_solve_ms"],
        std_solve_ms=stats["std_solve_ms"],
        setup_ms=setup_ms,
        final_residual=stats["residual"],
        cheb_steps=steps,
        hyperparams=json.dumps({"steps": steps, "smoother": "jacobi"}),
    )


def run_tensor(seq, operators, k, rhs_batch, args, *, dirichlet,
               rank, bulk_steps, inner_schur) -> Row:
    operator_apply = lambda x: apply_mass_matrix(
        seq, operators, x, k, dirichlet=dirichlet,
    )
    precond_apply = lambda rhs: apply_mass_tensor_preconditioner_ops(
        seq, operators, rhs, k, dirichlet=dirichlet,
    )
    solve = make_solve(operator_apply, precond_apply, args)
    stats = time_solve(solve, rhs_batch)
    strategy = "tensor_inner_schur" if inner_schur else "tensor"
    method = (
        f"tensor(r={rank},bcheb={bulk_steps},ischur=on)"
        if inner_schur else
        f"tensor(r={rank},bcheb={bulk_steps})"
    )
    return Row(
        case=_mass_case_name(k),
        k=k, strategy=strategy, method=method,
        avg_iters=stats["avg_iters"], max_iters=stats["max_iters"],
        min_iters=stats["min_iters"], std_iters=stats["std_iters"],
        avg_solve_ms=stats["avg_solve_ms"],
        min_solve_ms=stats["min_solve_ms"],
        max_solve_ms=stats["max_solve_ms"],
        std_solve_ms=stats["std_solve_ms"],
        setup_ms=0.0,  # measured outside
        final_residual=stats["residual"],
        rank=rank, bulk_cheb_steps=bulk_steps, inner_schur=inner_schur,
        hyperparams=json.dumps({
            "rank": rank,
            "bulk_cheb_steps": bulk_steps,
            "inner_schur": inner_schur,
        }),
    )


def run_k0_stiffness_jacobi(seq, operators, rhs_batch, args) -> Row:
    operator_apply = lambda x: apply_stiffness(seq, operators, x, 0, dirichlet=True)
    mass_apply = lambda x: apply_mass_matrix(seq, operators, x, 0, dirichlet=True)
    precond_apply = lambda rhs: operators.dd0_diaginv_dbc * rhs
    t0 = time.perf_counter()
    solve = make_solve(operator_apply, precond_apply, args, mass_apply=mass_apply)
    stats = time_solve(solve, rhs_batch)
    setup_ms = (time.perf_counter() - t0) * 1e3 - stats["avg_solve_ms"]
    return Row(
        case=K0_STIFFNESS_DBC_CASE,
        k=0,
        strategy="jacobi",
        method="jacobi",
        avg_iters=stats["avg_iters"],
        max_iters=stats["max_iters"],
        min_iters=stats["min_iters"],
        std_iters=stats["std_iters"],
        avg_solve_ms=stats["avg_solve_ms"],
        min_solve_ms=stats["min_solve_ms"],
        max_solve_ms=stats["max_solve_ms"],
        std_solve_ms=stats["std_solve_ms"],
        setup_ms=max(setup_ms, 0.0),
        final_residual=stats["residual"],
        dirichlet=True,
        hyperparams=json.dumps({}),
    )


def run_k0_stiffness_chebyshev(seq, operators, rhs_batch, args, *, steps) -> Row:
    operator_apply = lambda x: apply_stiffness(seq, operators, x, 0, dirichlet=True)
    mass_apply = lambda x: apply_mass_matrix(seq, operators, x, 0, dirichlet=True)
    jacobi_apply = lambda rhs: operators.dd0_diaginv_dbc * rhs
    dof = int(seq.n0_dbc)
    spec = MassPreconditionerSpec(
        kind="chebyshev",
        steps=steps,
        smoother=MassPreconditionerSpec(kind="jacobi"),
    )
    t_setup = time.perf_counter()
    min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply, jacobi_apply, dof, spec=spec, seed=args.seed,
    )
    precond_apply = _build_chebyshev_apply_preconditioner(
        operator_apply, jacobi_apply,
        steps=steps, min_eig=min_eig, max_eig=max_eig,
    )
    setup_ms = (time.perf_counter() - t_setup) * 1e3
    solve = make_solve(operator_apply, precond_apply, args, mass_apply=mass_apply)
    stats = time_solve(solve, rhs_batch)
    return Row(
        case=K0_STIFFNESS_DBC_CASE,
        k=0,
        strategy="chebyshev",
        method=f"cheb_J(s={steps})",
        avg_iters=stats["avg_iters"],
        max_iters=stats["max_iters"],
        min_iters=stats["min_iters"],
        std_iters=stats["std_iters"],
        avg_solve_ms=stats["avg_solve_ms"],
        min_solve_ms=stats["min_solve_ms"],
        max_solve_ms=stats["max_solve_ms"],
        std_solve_ms=stats["std_solve_ms"],
        setup_ms=setup_ms,
        final_residual=stats["residual"],
        cheb_steps=steps,
        dirichlet=True,
        hyperparams=json.dumps({"steps": steps, "smoother": "jacobi"}),
    )


def run_k0_stiffness_tensor(seq, operators, rhs_batch, args, *, rank,
                            bulk_steps: int = 0) -> Row:
    operator_apply = lambda x: apply_stiffness(seq, operators, x, 0, dirichlet=True)
    mass_apply = lambda x: apply_mass_matrix(seq, operators, x, 0, dirichlet=True)
    tensor_apply = lambda rhs: apply_hodge_laplacian_preconditioner(
        seq, operators, rhs, 0, dirichlet=True, kind="tensor",
    )
    if bulk_steps <= 0:
        precond_apply = tensor_apply
        extra_setup_ms = 0.0
    else:
        # Chebyshev polish on top of the tensor PC: a fixed-degree polynomial
        # in (K0 * T^{-1}) where T^{-1} is the tensor Hodge preconditioner.
        dof = int(seq.n0_dbc)
        spec = MassPreconditionerSpec(
            kind="chebyshev",
            steps=int(bulk_steps),
            smoother=MassPreconditionerSpec(kind="jacobi"),
        )
        t_extra = time.perf_counter()
        min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
            operator_apply, tensor_apply, dof, spec=spec, seed=args.seed,
        )
        precond_apply = _build_chebyshev_apply_preconditioner(
            operator_apply, tensor_apply,
            steps=int(bulk_steps), min_eig=min_eig, max_eig=max_eig,
        )
        extra_setup_ms = (time.perf_counter() - t_extra) * 1e3
    solve = make_solve(operator_apply, precond_apply, args, mass_apply=mass_apply)
    stats = time_solve(solve, rhs_batch)
    return Row(
        case=K0_STIFFNESS_DBC_CASE,
        k=0,
        strategy="tensor",
        method=f"tensor(r={rank},bcheb={bulk_steps})",
        avg_iters=stats["avg_iters"],
        max_iters=stats["max_iters"],
        min_iters=stats["min_iters"],
        std_iters=stats["std_iters"],
        avg_solve_ms=stats["avg_solve_ms"],
        min_solve_ms=stats["min_solve_ms"],
        max_solve_ms=stats["max_solve_ms"],
        std_solve_ms=stats["std_solve_ms"],
        setup_ms=extra_setup_ms,
        final_residual=stats["residual"],
        rank=rank,
        bulk_cheb_steps=int(bulk_steps),
        dirichlet=True,
        hyperparams=json.dumps({"rank": rank, "bulk_cheb_steps": int(bulk_steps)}),
    )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def run_benchmark(args) -> list[Row]:
    """Run the full Phase 1 protocol once for a single ``args`` config.

    Builds the de Rham sequence and base mass operators at the reference
    point, then sweeps Jacobi / Chebyshev(jacobi) / tensor preconditioners
    across the requested mass degrees ``args.ks``.
    """
    # Mass sweeps are free-boundary in Phase 1.
    dirichlet = False

    run_mass = not args.only_k0_stiffness_dbc
    run_k0_stiffness_dbc = args.include_k0_stiffness_dbc or args.only_k0_stiffness_dbc

    invalid = tuple(k for k in args.ks if k not in (0, 1, 2, 3))
    if invalid:
        raise ValueError(f"Mass degrees must be in 0,1,2,3; got {invalid}")
    if not run_mass and not run_k0_stiffness_dbc:
        raise ValueError("No benchmark cases selected")

    print(f"Building sequence ns={args.ns} p={args.p} eps={args.eps} "
          f"kappa={args.kappa} nfp={args.nfp} ...", flush=True)
    seq = build_sequence(args)
    operator_ks = set(args.ks if run_mass else ())
    if run_k0_stiffness_dbc:
        operator_ks.update((0, 1))
    base_operators = assemble_mass_operators(
        seq, seq.geometry, ks=tuple(sorted(operator_ks)),
    )
    if run_k0_stiffness_dbc:
        base_operators = assemble_incidence_operators(seq, operators=base_operators, ks=(0,))
        base_operators = assemble_hodge_operators(
            seq, seq.geometry, operators=base_operators, ks=(0,),
        )

    print(f"Building Besov RHS batches s={args.besov_s} n_rhs={args.n_rhs} ...",
          flush=True)
    rhs_batches = {}
    if run_mass:
        rhs_batches.update({
            _mass_case_name(k): build_random_besov_rhs_batch(
                seq, k, dirichlet=dirichlet,
                n_rhs=args.n_rhs, seed=args.seed,
                s=args.besov_s, **BESOV_RHS_KWARGS,
            )
            for k in args.ks
        })
    if run_k0_stiffness_dbc:
        rhs_batches[K0_STIFFNESS_DBC_CASE] = build_random_besov_rhs_batch(
            seq, 0, dirichlet=True,
            n_rhs=args.n_rhs, seed=args.seed,
            s=args.besov_s, **BESOV_RHS_KWARGS,
        )

    rows: list[Row] = []

    # Baselines.
    if run_mass:
        for k in args.ks:
            rhs_batch = rhs_batches[_mass_case_name(k)]
            if not args.no_jacobi:
                print(f"  M{k}  jacobi", flush=True)
                rows.append(run_jacobi(seq, base_operators, k, rhs_batch, args,
                                       dirichlet=dirichlet))
            if not args.no_chebyshev:
                for steps in args.cheb_baseline:
                    print(f"  M{k}  chebyshev(jacobi) steps={steps}", flush=True)
                    rows.append(run_chebyshev(seq, base_operators, k, rhs_batch,
                                              args, dirichlet=dirichlet, steps=steps))
    if run_k0_stiffness_dbc:
        rhs_batch = rhs_batches[K0_STIFFNESS_DBC_CASE]
        if not args.no_jacobi:
            print(f"  {K0_STIFFNESS_DBC_CASE}  jacobi", flush=True)
            rows.append(run_k0_stiffness_jacobi(seq, base_operators, rhs_batch, args))
        if not args.no_chebyshev:
            for steps in args.cheb_baseline:
                print(f"  {K0_STIFFNESS_DBC_CASE}  chebyshev(jacobi) steps={steps}", flush=True)
                rows.append(run_k0_stiffness_chebyshev(
                    seq, base_operators, rhs_batch, args, steps=steps,
                ))

    # Tensor sweep. One assembly per (rank, bulk_steps, inner_schur variant).
    if not args.no_tensor:
        if run_mass:
            inner_schur_ks = tuple(k for k in args.ks if k in (1, 2))
            tensor_variants: list[tuple[bool, tuple[int, ...]]] = []
            if not args.inner_schur:
                tensor_variants.append((False, tuple(args.ks)))
            if inner_schur_ks:
                tensor_variants.append((True, inner_schur_ks))
            for rank in args.ranks:
                for bulk_steps in args.bulk_cheb:
                    for inner_schur, ks_to_run in tensor_variants:
                        if not ks_to_run:
                            continue
                        t_setup = time.perf_counter()
                        suffix = " ischur=on" if inner_schur else ""
                        print(
                            f"  Assembling tensor rank={rank} bulk_cheb={bulk_steps}{suffix} ...",
                            flush=True,
                        )
                        operators = assemble_tensor_mass_preconditioner(
                            seq, operators=base_operators, ks=tuple(args.ks), rank=rank,
                            cp_kwargs={
                                "block_chebyshev_steps": int(bulk_steps),
                                "k1_inner_schur": inner_schur,
                                "k2_inner_schur": inner_schur,
                            },
                        )
                        jax.block_until_ready(jnp.zeros(()))
                        setup_ms = (time.perf_counter() - t_setup) * 1e3
                        for k in ks_to_run:
                            rhs_batch = rhs_batches[_mass_case_name(k)]
                            label = (
                                f"tensor(r={rank},bcheb={bulk_steps},ischur=on)"
                                if inner_schur else
                                f"tensor(r={rank},bcheb={bulk_steps})"
                            )
                            print(f"    M{k}  {label}", flush=True)
                            row = run_tensor(
                                seq, operators, k, rhs_batch, args,
                                dirichlet=dirichlet,
                                rank=rank, bulk_steps=bulk_steps,
                                inner_schur=inner_schur,
                            )
                            row.setup_ms = setup_ms / max(len(ks_to_run), 1)
                            rows.append(row)
        if run_k0_stiffness_dbc:
            rhs_batch = rhs_batches[K0_STIFFNESS_DBC_CASE]
            for rank in args.ranks:
                t_setup = time.perf_counter()
                print(f"  Assembling {K0_STIFFNESS_DBC_CASE} tensor rank={rank} ...", flush=True)
                operators = assemble_tensor_mass_preconditioner(
                    seq,
                    operators=base_operators,
                    ks=(0,),
                    rank=rank,
                    cp_kwargs={"k0_rank": rank},
                )
                operators = assemble_hodge_operators(
                    seq, seq.geometry, operators=operators, ks=(0,),
                )
                jax.block_until_ready(jnp.zeros(()))
                setup_ms = (time.perf_counter() - t_setup) * 1e3
                for bulk_steps in args.bulk_cheb:
                    print(
                        f"    {K0_STIFFNESS_DBC_CASE}  tensor(r={rank},bcheb={bulk_steps})",
                        flush=True,
                    )
                    row = run_k0_stiffness_tensor(
                        seq, operators, rhs_batch, args,
                        rank=rank, bulk_steps=int(bulk_steps),
                    )
                    row.setup_ms = setup_ms + row.setup_ms
                    rows.append(row)

    # Stamp common metadata onto every row.
    for row in rows:
        row.n_rhs = args.n_rhs
        row.seed = args.seed
        row.n_r, row.n_t, row.n_z = args.ns
        row.p = args.p
        row.eps = args.eps
        row.kappa = args.kappa
        row.nfp = args.nfp
        row.besov_s = args.besov_s
        if row.case != K0_STIFFNESS_DBC_CASE:
            row.dirichlet = dirichlet

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=parse_ns, default=DEFAULT_NS)
    parser.add_argument("--p", type=int, default=DEFAULT_P)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--r0", type=float, default=DEFAULT_R0)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA,
                        help="Rotating-ellipse aspect ratio.")
    parser.add_argument("--nfp", type=int, default=DEFAULT_NFP,
                        help="Number of field periods.")
    parser.add_argument("--ks", type=parse_int_list, default=(0, 1, 2, 3))
    parser.add_argument("--n-rhs", type=int, default=DEFAULT_N_RHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--besov-s", type=float, default=DEFAULT_BESOV_S)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--ranks", type=parse_int_list, default=TENSOR_RANKS)
    parser.add_argument("--bulk-cheb", type=parse_int_list,
                        default=TENSOR_BULK_CHEB_STEPS,
                        help="Bulk Chebyshev step counts for the tensor "
                             "preconditioner. 0 disables.")
    parser.add_argument("--cheb-baseline", type=parse_int_list,
                        default=CHEB_BASELINE_STEPS,
                        help="Polynomial degrees for whole-matrix "
                             "Chebyshev(jacobi) baselines.")
    parser.add_argument("--no-jacobi", action="store_true",
                        help="Skip the plain Jacobi baseline.")
    parser.add_argument("--no-chebyshev", action="store_true",
                        help="Skip Chebyshev(jacobi) baselines.")
    parser.add_argument("--no-tensor", action="store_true",
                        help="Skip the tensor preconditioner sweep.")
    parser.add_argument("--include-k0-stiffness-dbc", action="store_true",
                        help="Append the k=0 Dirichlet stiffness block to the benchmark.")
    parser.add_argument("--only-k0-stiffness-dbc", action="store_true",
                        help="Benchmark only the k=0 Dirichlet stiffness block.")
    parser.add_argument("--out", type=str, default="",
                        help="Optional CSV output path.")
    parser.add_argument("--inner-schur", action="store_true",
                        help="Benchmark only the tensor variant with inner RT/zeta Schur on for k=1,2.")
    parser.set_defaults(inner_schur=False)
    args = parser.parse_args()

    rows = run_benchmark(args)

    # Sort and print.
    def _sort_key(row: Row):
        order = {
            "jacobi": 0,
            "chebyshev": 1,
            "tensor": 2,
            "tensor_inner_schur": 3,
        }.get(row.strategy, 99)
        return (
            _case_sort_key(row.case),
            order,
            row.cheb_steps,
            row.rank,
            row.bulk_cheb_steps,
            row.inner_schur,
        )
    rows.sort(key=_sort_key)

    header = (
        f"{'case':>8} {'method':>26} "
        f"{'avg_it':>7} {'max_it':>7} "
        f"{'solve_ms':>9} {'setup_ms':>9} {'res':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.case:>8} {row.method:>26} "
            f"{row.avg_iters:>7.1f} {row.max_iters:>7d} "
            f"{row.avg_solve_ms:>9.2f} {row.setup_ms:>9.2f} "
            f"{row.final_residual:>10.2e}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
