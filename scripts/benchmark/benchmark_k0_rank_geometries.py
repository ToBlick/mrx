"""k=0 Laplacian + mass preconditioner sweep across geometries (one resolution).

For a single resolution this runs all three geometries (periodic cylinder,
axisymmetric toroid, rotating ellipse) and reports:

  1. k=0 Hodge-Laplacian L_0 condensed CG -- jacobi vs the rank-1 (FD) tensor
     Hodge preconditioner, for both BCs (dbc = nonsingular, free = constant
     nullspace deflated). The k=0 Laplacian atom is CONSTRAINED to rank-1 FD in
     production (the rank-2 radial_dense path is not free-BC safe in a deep CG
     solve -- see docs/hiptmair_xu_preconditioner.md).

  2. Mass-matrix M_k inversion CG -- jacobi vs the tensor mass preconditioner at
     rank 1 and rank 2, for each requested k. The mass is SPD (no nullspace), so
     plain CG. rank-2 mass is the exact 2-term Kronecker inverse (no change vs
     rank-1 needed in principle); this measures whether it helps in practice.

Slurm policy: ONE resolution per job; all geometries inside the job. The launcher
slurm/job_k0_rank_geometries.sh submits one job per resolution.

Usage:
    python scripts/benchmark/benchmark_k0_rank_geometries.py --ns 8,16,4
    python scripts/benchmark/benchmark_k0_rank_geometries.py --ns 8,16,4 \
        --geometries cylinder,toroid,rotating_ellipse --mass-ks 0,1,2 --p 3
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

# Make the sibling benchmark module importable (it lives in this directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark_graddiv_k1_preconditioner as bench  # noqa: E402
from mrx.operators import (  # noqa: E402
    _diagonal_from_matvec,
    _invert_diagonal,
    _nullspace_vectors,
    apply_laplacian_preconditioner,
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    apply_stiffness,
    assemble_incidence_operators,
    assemble_laplacian_operators,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.solvers import solve_singular_cg  # noqa: E402


def _make_args(*, ns, p, geometry, cg_tol, cg_maxiter, n_rhs, seed,
               epsilon, kappa, r0, nfp):
    """argparse-like namespace consumed by bench.build_sequence."""
    return SimpleNamespace(
        ns=ns, p=p, geometry=geometry, cg_tol=cg_tol, cg_maxiter=cg_maxiter,
        n_rhs=n_rhs, seed=seed, epsilon=epsilon, kappa=kappa, r0=r0, nfp=nfp,
    )


def _mass_size(seq, k, dirichlet):
    attr = f"n{k}_dbc" if dirichlet else f"n{k}"
    return int(getattr(seq, attr))


def _time_cg(a_matvec, mass_matvec, precond, n, vs, args, report_rel_tol):
    """jit a batched (deflated) CG solve and time it; return time_solve stats."""
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    rhs_batch = jnp.stack(
        [a_matvec(jax.random.normal(k, (n,), dtype=jnp.float64)) for k in keys],
        axis=0)
    jax.block_until_ready(rhs_batch)

    @jax.jit
    def solve(rhs, a=a_matvec, m=mass_matvec, p=precond, vv=vs):
        x, info = solve_singular_cg(
            a, rhs, mass_matvec=m, precond_matvec=p, vs=vv,
            tol=args.cg_tol, maxiter=args.cg_maxiter)
        r = a(x) - rhs
        rel = jnp.linalg.norm(r) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
        return x, info, rel

    return bench.time_solve(solve, rhs_batch, rel_tol=report_rel_tol)


def run_laplacian(seq, ops_lap, args, report_rel_tol):
    """k=0 L_0 condensed CG: jacobi vs rank-1 FD, both BCs. -> {(bc,method): stats}."""
    out = {}
    for dirichlet in (True, False):
        bc = "dbc" if dirichlet else "free"
        n = int(seq.n0_dbc if dirichlet else seq.n0)
        vs = _nullspace_vectors(ops_lap, 0, dirichlet)

        def a_matvec(v, d=dirichlet):
            return apply_stiffness(seq, ops_lap, v, 0, dirichlet=d)

        def mass_matvec(v, d=dirichlet):
            return apply_mass_matrix(seq, ops_lap, v, 0, dirichlet=d)

        # Fair jacobi: precompute diag(L_0)^{-1} once (the library jacobi apply
        # re-probes the diagonal each call -> traced into the loop).
        l0_diaginv = _invert_diagonal(_diagonal_from_matvec(a_matvec, n))

        def jacobi(v, di=l0_diaginv):
            return di * v

        def tensor(v, d=dirichlet):
            return apply_laplacian_preconditioner(
                seq, ops_lap, v, 0, dirichlet=d, kind="tensor")

        for label, p in (("jacobi", jacobi), ("rank=1 (FD)", tensor)):
            out[(bc, label)] = _time_cg(
                a_matvec, mass_matvec, p, n, vs, args, report_rel_tol)
    return out


def run_mass(seq, ops_base, ops_by_rank, args, mass_ks, report_rel_tol):
    """M_k CG (free BC, SPD): jacobi vs tensor rank 1/2. -> {(k,method): stats}."""
    out = {}
    dirichlet = False
    for k in mass_ks:
        n = _mass_size(seq, k, dirichlet)

        def a_matvec(v, k=k, d=dirichlet):
            return apply_mass_matrix(seq, ops_base, v, k, dirichlet=d)

        # Fair jacobi: precompute diag(M_k)^{-1} once.
        diaginv = _invert_diagonal(_diagonal_from_matvec(a_matvec, n))

        def jacobi(v, di=diaginv):
            return di * v

        methods = [("jacobi", jacobi)]
        for rank, ops_r in ops_by_rank.items():
            def tensor(v, k=k, d=dirichlet, o=ops_r):
                return apply_mass_matrix_preconditioner(
                    seq, o, v, k, dirichlet=d, kind="tensor")
            methods.append((f"rank={rank}", tensor))

        for label, p in methods:
            out[(k, label)] = _time_cg(
                a_matvec, a_matvec, p, n, [], args, report_rel_tol)
    return out


def _print_stats(prefix, s):
    print(f"{prefix}{s['avg_iters']:>8.1f}{s['max_iters']:>8d}"
          f"{s['avg_ms']:>10.1f}{s['max_residual']:>12.2e}"
          f"{s['n_fail']:>5d}/{s['n_total']:<2d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=str, default="8,16,4",
                        help="Single resolution n_r,n_theta,n_zeta (one per job).")
    parser.add_argument("--geometries", type=str,
                        default="cylinder,toroid,rotating_ellipse")
    parser.add_argument("--mass-ks", type=str, default="0,1,2,3",
                        help="Comma-separated form degrees for the mass benchmark.")
    parser.add_argument("--mass-bcheb", type=str, default="0,1,2,3",
                        help="Comma-separated block-Chebyshev polish-step values to "
                             "sweep for the tensor mass (0 = no polish = historical prod).")
    parser.add_argument("--mass-ischur", type=str, default="off,on",
                        help="Comma-separated inner-Schur settings to sweep "
                             "(off = diagonal bulk = historical prod).")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--n-rhs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--report-rel-tol", type=float, default=None)
    cli = parser.parse_args()

    report_rel_tol = cli.cg_tol if cli.report_rel_tol is None else cli.report_rel_tol
    ns = tuple(int(x) for x in cli.ns.split(","))
    geometries = [g.strip() for g in cli.geometries.split(",") if g.strip()]
    mass_ks = [int(x) for x in cli.mass_ks.split(",") if x.strip() != ""]
    bcheb_list = [int(x) for x in cli.mass_bcheb.split(",") if x.strip() != ""]
    ischur_list = [t.strip().lower() in ("on", "true", "1")
                   for t in cli.mass_ischur.split(",") if t.strip() != ""]

    lap_rows: dict = {}   # geom -> {(bc,method): stats}
    mass_rows: dict = {}  # geom -> {(k,method): stats}
    for geometry in geometries:
        print(f"\n[build] {geometry} ns={ns} p={cli.p}", flush=True)
        args = _make_args(
            ns=ns, p=cli.p, geometry=geometry, cg_tol=cli.cg_tol,
            cg_maxiter=cli.cg_maxiter, n_rhs=cli.n_rhs, seed=cli.seed,
            epsilon=cli.epsilon, kappa=cli.kappa, r0=cli.r0, nfp=cli.nfp)
        seq = bench.build_sequence(args)

        # Base operators: extraction + reference mass (from build) + incidence +
        # k=0 stiffness/Hodge. The rank-1 FD k=0 atom for the Laplacian.
        ops = seq.get_operators()
        ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))
        ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0,))
        ops_lap = assemble_tensor_laplacian_preconditioner(
            seq, operators=ops, ks=(0,), rank=1)

        # Warm the matrix-free mass-core caches EAGERLY (outside jit). The core
        # build does host-side numpy work that cannot be traced; if first touched
        # inside the jitted CG (e.g. the k=0 mass used by the free-BC deflation)
        # it raises TracerArrayConversionError. The core cache is keyed by (seq,k)
        # only, so one eager apply per k suffices. k=0,1 cover the Laplacian
        # (L_0 = G_0^T M_1 G_0 + the k=0 deflation mass); mass_ks cover the rest.
        for kk in sorted(set([0, 1] + mass_ks)):
            nkk = _mass_size(seq, kk, False)
            jax.block_until_ready(apply_mass_matrix(
                seq, ops, jnp.zeros((nkk,), dtype=jnp.float64), kk, dirichlet=False))
        lap_rows[geometry] = run_laplacian(seq, ops_lap, args, report_rel_tol)

        # Mass: sweep (block-Chebyshev steps) x (inner-Schur) configs, reusing the
        # built geometry (the expensive part for W7-X). The mass-precond reassembly
        # is cheap. mass_rows[geometry][(bcheb, ischur, k, method)] = stats.
        mass_rows[geometry] = {}
        for bcheb in bcheb_list:
            for ischur in ischur_list:
                cpk = {"block_chebyshev_steps": int(bcheb),
                       "k1_inner_schur": bool(ischur),
                       "k2_inner_schur": bool(ischur)}
                ops_by_rank = {}
                for rank in (1, 2):
                    ops_by_rank[rank] = assemble_tensor_mass_preconditioner(
                        seq, operators=ops, ks=tuple(mass_ks), rank=rank, cp_kwargs=cpk)
                res = run_mass(seq, ops, ops_by_rank, args, mass_ks, report_rel_tol)
                for (k, method), s in res.items():
                    mass_rows[geometry][(bcheb, bool(ischur), k, method)] = s

    # ---- Laplacian table ----
    print("\n\n=== k=0 Hodge-Laplacian L_0 condensed CG: jacobi vs rank-1 FD ===")
    print(f"ns={ns}, p={cli.p}, n_rhs={cli.n_rhs}, cg_tol={cli.cg_tol:.0e}")
    hdr = (f"{'geometry':<17}{'bc':<6}{'method':<14}"
           f"{'avg_it':>8}{'max_it':>8}{'avg_ms':>10}{'max_res':>12}{'fails':>8}")
    print(hdr)
    print("-" * len(hdr))
    for geometry in geometries:
        for bc in ("dbc", "free"):
            for method in ("jacobi", "rank=1 (FD)"):
                s = lap_rows[geometry].get((bc, method))
                if s is not None:
                    _print_stats(f"{geometry:<17}{bc:<6}{method:<14}", s)
            print("-" * len(hdr))

    # ---- Mass table ----
    print("\n\n=== Mass M_k inversion CG (free BC, SPD): jacobi vs tensor rank 1/2 ===")
    print(f"ns={ns}, p={cli.p}, n_rhs={cli.n_rhs}, cg_tol={cli.cg_tol:.0e}; "
          f"sweep block_cheb_steps={bcheb_list} x ischur={['on' if i else 'off' for i in ischur_list]}")
    hdr = (f"{'geometry':<13}{'k':<3}{'bcheb':<6}{'ischur':<7}{'method':<8}"
           f"{'avg_it':>8}{'max_it':>7}{'avg_ms':>10}{'max_res':>12}{'fails':>8}")
    print(hdr)
    print("-" * len(hdr))
    for geometry in geometries:
        rows = mass_rows[geometry]
        for k in mass_ks:
            # jacobi is config-independent: print once.
            sj = rows.get((bcheb_list[0], ischur_list[0], k, "jacobi"))
            if sj is not None:
                _print_stats(f"{geometry:<13}{k:<3}{'-':<6}{'-':<7}{'jacobi':<8}", sj)
            for bcheb in bcheb_list:
                for ischur in ischur_list:
                    itag = "on" if ischur else "off"
                    for method in ("rank=1", "rank=2"):
                        s = rows.get((bcheb, ischur, k, method))
                        if s is not None:
                            _print_stats(
                                f"{geometry:<13}{k:<3}{bcheb:<6}{itag:<7}{method:<8}", s)
            print("-" * len(hdr))


if __name__ == "__main__":
    main()
