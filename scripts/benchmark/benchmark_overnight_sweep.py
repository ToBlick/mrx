"""Comprehensive overnight preconditioner sweep -- ONE (geometry, ns, p) cell.

For a single cell, runs the full battery for BOTH boundary conditions and writes
an incremental CSV (one row per case/method/rank/config/bc). Robust by design:
the CSV header is written once, every row is flushed, and EACH operator group /
solve is wrapped so a failure (or OOM) records a status='error' row instead of
crashing the cell -- a blow-up in one block never loses the rest.

Operator groups (both BCs unless noted):
  - mass k=0..3 : jacobi vs tensor rank {1,2,3}  (rank-3 = phase2 bridge)
  - k=0 Laplacian (condensed CG) : jacobi vs tensor rank-1 FD (rank-2 disabled)
  - k=1 Laplacian (saddle HX)    : jacobi vs projected P_A+P_B with a Chebyshev
                                   L_0 atom, sweeping eps (degree auto from kappa,
                                   recorded), rank {1,2}
  - k=2,k=3 Laplacian (saddle)   : Schur-Jacobi only, schur_diag_mode tensor_probe
                                   (rank-dependent) at rank {1,2}, plus a plain
                                   'diag' baseline (rank-independent)

Nullspace/harmonic vectors are computed with a JACOBI shifted preconditioner
(monkeypatch installed at import, before any sequence build).

The launcher slurm/job_overnight_sweep.sh submits one job per (geometry, ns, p).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.nullspace as _ns  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    MassPreconditionerSpec, SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec, default_mass_preconditioner,
)

# --- Nullspace shifted preconditioner -> JACOBI (install BEFORE build_sequence) ---
_ns_orig = _ns._nullspace_shifted_preconditioner


def _jacobi_shifted(k):
    if k == 0:
        return _ns._validate_nullspace_shifted_preconditioner(
            k, MassPreconditionerSpec(kind="jacobi"))
    return _ns._validate_nullspace_shifted_preconditioner(
        k, SaddlePointPreconditionerSpec(
            mass=default_mass_preconditioner(),
            schur=SchurPreconditionerSpec(
                inner=MassPreconditionerSpec(kind="tensor"),   # validator forces tensor
                outer=MassPreconditionerSpec(kind="jacobi"))))


_ns._nullspace_shifted_preconditioner = _jacobi_shifted

import benchmark_graddiv_k1_preconditioner as bench  # noqa: E402
from benchmark_graddiv_k1_preconditioner import (  # noqa: E402
    build_sequence, make_apply_routines, make_saddle_solve,
    time_solve, time_saddle_solve, make_chebyshev_upper,
    _lanczos_extremal_eigs_precond,
)
from mrx.operators import (  # noqa: E402
    _diagonal_from_matvec, _invert_diagonal, _nullspace_vectors,
    _get_schur_diaginv,
    apply_mass_matrix, apply_mass_matrix_preconditioner, apply_stiffness,
    apply_laplacian_preconditioner, apply_laplacian_approx,
    apply_derivative_matrix,
    assemble_mass_jacobi_preconditioner, assemble_incidence_operators,
    assemble_laplacian_operators, assemble_tensor_mass_preconditioner,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_stiffness_preconditioner,
    assemble_schur_jacobi_preconditioner, assemble_projection_operators,
)
from mrx.solvers import solve_singular_cg  # noqa: E402

CP = {"precompute_coupling": True}
# For the k=0 tensor-Hodge atom: pass full CP params so the EXPLICIT rank=1 is
# honored. Without them, _k0_tensor_hodge_config falls back to the assembled mass
# tensor's rank (e.g. 2) and trips the production rank-1 constraint.
CP_K0 = {"precompute_coupling": True, "maxiter": 100, "tol": 1e-9, "ridge": 1e-12}

FIELDS = ["geometry", "n_r", "n_t", "n_z", "p", "case", "k", "bc", "method",
          "rank", "config", "avg_iters", "max_iters", "avg_ms", "setup_ms",
          "max_residual", "n_fail", "n_total", "n_rhs", "status"]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
class Writer:
    def __init__(self, path, base):
        self.base = base
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        if new:
            self.w.writeheader()
            self.f.flush()

    def row(self, **kw):
        r = dict(self.base)
        for fld in FIELDS:
            r.setdefault(fld, "")
        r.update(kw)
        self.w.writerow({k: r.get(k, "") for k in FIELDS})
        self.f.flush()

    def stats(self, *, case, k, bc, method, rank, config, stats, setup_ms):
        self.row(case=case, k=k, bc=bc, method=method, rank=rank, config=config,
                 avg_iters=f"{stats['avg_iters']:.2f}", max_iters=stats["max_iters"],
                 avg_ms=f"{stats['avg_ms']:.2f}", setup_ms=f"{setup_ms:.1f}",
                 max_residual=f"{stats['max_residual']:.3e}",
                 n_fail=stats["n_fail"], n_total=stats["n_total"], status="ok")

    def err(self, *, case, k, bc, method, rank, config, msg, setup_ms=0.0):
        short = msg.strip().splitlines()[-1][:200] if msg.strip() else "error"
        self.row(case=case, k=k, bc=bc, method=method, rank=rank, config=config,
                 setup_ms=f"{setup_ms:.1f}", status="error:" + short)


def _sz(seq, k, dirichlet):
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


def _time_cg(a_matvec, mass_matvec, precond, n, vs, args):
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    rhs_batch = jnp.stack(
        [a_matvec(jax.random.normal(kk, (n,), dtype=jnp.float64)) for kk in keys], axis=0)
    jax.block_until_ready(rhs_batch)

    @jax.jit
    def solve(rhs, a=a_matvec, m=mass_matvec, p=precond, vv=vs):
        x, info = solve_singular_cg(a, rhs, mass_matvec=m, precond_matvec=p, vs=vv,
                                    tol=args.cg_tol, maxiter=args.cg_maxiter)
        r = a(x) - rhs
        rel = jnp.linalg.norm(r) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
        return x, info, rel

    return time_solve(solve, rhs_batch, rel_tol=args.cg_tol)


# --------------------------------------------------------------------------- #
# assembly (piecewise -- avoids assemble_operators which would build the
# rank-2 k=0 tensor Laplacian and CRASH on the production rank-1 constraint)
# --------------------------------------------------------------------------- #
def base_ops(seq):
    ops = seq.get_operators()
    ops = assemble_mass_jacobi_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3))
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1, 2))
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0, 1, 2, 3))
    return ops


def mass_ops(seq, ops_b, rank):
    return assemble_tensor_mass_preconditioner(
        seq, operators=ops_b, ks=(0, 1, 2, 3), rank=rank, cp_kwargs=CP)


def k0_lap_ops(seq, ops_mass_r1):
    # k=0 tensor Hodge atom is rank-1 only; build on the rank-1 mass ops.
    return assemble_tensor_laplacian_preconditioner(
        seq, operators=ops_mass_r1, ks=(0,), rank=1, cp_kwargs=CP_K0)


def k1_ops(seq, ops_mass_r, rank):
    # k=1 HX needs: tensor mass (rank R, in ops_mass_r) + k=0 tensor Hodge atom
    # (rank-1, for the P_B inner L0) + tensor stiffness k=1 (rank R) + schur jacobi.
    ops = assemble_tensor_laplacian_preconditioner(
        seq, operators=ops_mass_r, ks=(0,), rank=1, cp_kwargs=CP_K0)
    ops = assemble_tensor_stiffness_preconditioner(
        seq, operators=ops, ks=(1,), rank=rank, cp_kwargs=CP)
    ops = assemble_schur_jacobi_preconditioner(
        seq, operators=ops, ks=(1,), dirichlet_variants=(True, False),
        schur_diag_mode='tensor_probe')
    return ops


def saddle_ops(seq, ops_mass_r, k, schur_mode):
    # k=2/k=3 JACOBI-only: needs ONLY the schur-jacobi diagonal. 'tensor_probe'
    # builds it from the tensor MASS already in ops_mass_r (rank-R). Do NOT
    # assemble the tensor STIFFNESS P_A -- it is unused for the jacobi baseline
    # and OOMs the GPU at rank>2 (Lynch), which try/except cannot catch.
    return assemble_schur_jacobi_preconditioner(
        seq, operators=ops_mass_r, ks=(k,), dirichlet_variants=(True, False),
        schur_diag_mode=schur_mode)


# --------------------------------------------------------------------------- #
# operator groups
# --------------------------------------------------------------------------- #
def _cheb_mass_prec(a_matvec, smoother, n, deg):
    """Whole-operator Chebyshev preconditioner for the SPD mass (no deflation
    needed -- mass is SPD, lmin>0). smoother is the jacobi-diag or tensor apply."""
    lmin, lmax = _lanczos_extremal_eigs_precond(a_matvec, smoother, n, steps=50, seed=0)
    lmin = max(lmin, lmax * 1e-5)
    return make_chebyshev_upper(a_matvec, smoother, lmin, lmax, int(deg))


def group_mass(seq, ops_b, ops_mass, deg_list, W, args, setup_ms):
    for dirichlet in (True, False):
        bc = "dbc" if dirichlet else "free"
        for k in (0, 1, 2, 3):
            n = _sz(seq, k, dirichlet)

            def a_matvec(v, k=k, d=dirichlet):
                return apply_mass_matrix(seq, ops_b, v, k, dirichlet=d)

            try:
                diaginv = _invert_diagonal(_diagonal_from_matvec(a_matvec, n))
            except Exception:
                W.err(case=f"M{k}", k=k, bc=bc, method="diag", rank="-", config="-",
                      msg=traceback.format_exc(), setup_ms=setup_ms)
                continue
            jac = lambda v, di=diaginv: di * v

            def tens(rank):
                o = ops_mass[rank]
                return lambda v, k=k, d=dirichlet, o=o: apply_mass_matrix_preconditioner(
                    seq, o, v, k, dirichlet=d, kind="tensor")

            # jacobi baseline
            try:
                st = _time_cg(a_matvec, a_matvec, jac, n, [], args)
                W.stats(case=f"M{k}", k=k, bc=bc, method="jacobi", rank="-",
                        config="-", stats=st, setup_ms=setup_ms)
            except Exception:
                W.err(case=f"M{k}", k=k, bc=bc, method="jacobi", rank="-",
                      config="-", msg=traceback.format_exc(), setup_ms=setup_ms)
            # tensor (rank sweep)
            for rank in ops_mass:
                try:
                    st = _time_cg(a_matvec, a_matvec, tens(rank), n, [], args)
                    W.stats(case=f"M{k}", k=k, bc=bc, method="tensor", rank=rank,
                            config="-", stats=st, setup_ms=setup_ms)
                except Exception:
                    W.err(case=f"M{k}", k=k, bc=bc, method="tensor", rank=rank,
                          config="-", msg=traceback.format_exc(), setup_ms=setup_ms)
            # whole-operator Chebyshev competitors: jacobi smoother + tensor smoother
            for deg in deg_list:
                try:
                    pc = _cheb_mass_prec(a_matvec, jac, n, deg)
                    st = _time_cg(a_matvec, a_matvec, pc, n, [], args)
                    W.stats(case=f"M{k}", k=k, bc=bc, method="cheb-jac", rank="-",
                            config=f"deg={deg}", stats=st, setup_ms=setup_ms)
                except Exception:
                    W.err(case=f"M{k}", k=k, bc=bc, method="cheb-jac", rank="-",
                          config=f"deg={deg}", msg=traceback.format_exc(), setup_ms=setup_ms)
                for rank in (1, 2):
                    if rank not in ops_mass:
                        continue
                    try:
                        pc = _cheb_mass_prec(a_matvec, tens(rank), n, deg)
                        st = _time_cg(a_matvec, a_matvec, pc, n, [], args)
                        W.stats(case=f"M{k}", k=k, bc=bc, method="cheb-tensor", rank=rank,
                                config=f"deg={deg}", stats=st, setup_ms=setup_ms)
                    except Exception:
                        W.err(case=f"M{k}", k=k, bc=bc, method="cheb-tensor", rank=rank,
                              config=f"deg={deg}", msg=traceback.format_exc(),
                              setup_ms=setup_ms)


def group_k0_lap(seq, ops_lap, args, W, setup_ms):
    # Pure forms only: jacobi vs rank-1 FD tensor atom (truncated pinv at the
    # default 1e-12 -- the sweep showed higher tols give no free benefit and 1e-2
    # breaks it; W7-X free stalls regardless = the ζ-rank, recorded for science).
    for dirichlet in (True, False):
        bc = "dbc" if dirichlet else "free"
        n = _sz(seq, 0, dirichlet)

        def a_matvec(v, d=dirichlet):
            return apply_stiffness(seq, ops_lap, v, 0, dirichlet=d)

        def mass_matvec(v, d=dirichlet):
            return apply_mass_matrix(seq, ops_lap, v, 0, dirichlet=d)
        try:
            vs = _nullspace_vectors(ops_lap, 0, dirichlet)
        except Exception:
            W.err(case="K0", k=0, bc=bc, method="nullspace", rank="-", config="-",
                  msg=traceback.format_exc(), setup_ms=setup_ms)
            continue
        try:
            diaginv = _invert_diagonal(_diagonal_from_matvec(a_matvec, n))
            st = _time_cg(a_matvec, mass_matvec, lambda v, di=diaginv: di * v, n, vs, args)
            W.stats(case="K0", k=0, bc=bc, method="jacobi", rank="-", config="-",
                    stats=st, setup_ms=setup_ms)
        except Exception:
            W.err(case="K0", k=0, bc=bc, method="jacobi", rank="-", config="-",
                  msg=traceback.format_exc(), setup_ms=setup_ms)
        try:
            def tensor(v, d=dirichlet):
                return apply_laplacian_preconditioner(seq, ops_lap, v, 0, dirichlet=d, kind="tensor")
            st = _time_cg(a_matvec, mass_matvec, tensor, n, vs, args)
            W.stats(case="K0", k=0, bc=bc, method="tensor", rank=1, config="FD",
                    stats=st, setup_ms=setup_ms)
        except Exception:
            W.err(case="K0", k=0, bc=bc, method="tensor", rank=1, config="FD",
                  msg=traceback.format_exc(), setup_ms=setup_ms)


def _cheb_l0_atom(seq, ops, deg, dirichlet, n0):
    """Deflated tensor-Chebyshev L0^-1 atom at a FIXED integer degree.

    The Chebyshev iterates against the DEFLATED operator s_hat = Pd.L0.Pp (and the
    deflated smoother Pp.M.Pd), so the constant null mode (free BC) never enters the
    recurrence. Without operator deflation, roundoff reseeds the constant between
    applies and the residual polynomial -- which is huge OUTSIDE [lmin,lmax], i.e.
    at eigenvalue 0 -- amplifies it to nan. The SAME deflation is used (a) in the
    Lanczos interval estimate, (b) standalone as the k=0 preconditioner, and (c)
    inside k=1's P_B (which injects this atom as l0_inv_custom). dbc: c0 empty ->
    projectors are identity -> raw operator (unchanged)."""
    def l0(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=dirichlet)

    c0 = jnp.asarray(_nullspace_vectors(ops, 0, dirichlet))
    if c0.shape[0] > 0:
        Mc0 = jnp.stack([apply_mass_matrix(seq, ops, c0[i], 0, dirichlet=dirichlet)
                         for i in range(c0.shape[0])], axis=0)
        cn = jnp.sqrt(jnp.einsum("ij,ij->i", c0, Mc0))
        c0, Mc0 = c0 / cn[:, None], Mc0 / cn[:, None]

        def Pp(x):
            return x - jnp.einsum("i,ij->j", Mc0 @ x, c0)

        def Pd(b):
            return b - jnp.einsum("i,ij->j", c0 @ b, Mc0)

        def smoother(b):
            return Pp(apply_laplacian_preconditioner(seq, ops, Pd(b), 0,
                                                     dirichlet=dirichlet, kind="tensor"))
    else:
        def Pp(x):
            return x
        Pd = Pp

        def smoother(b):
            return apply_laplacian_preconditioner(seq, ops, b, 0, dirichlet=dirichlet,
                                                  kind="tensor")

    def s_hat(x):  # DEFLATED operator: Pd . L0 . Pp (kills roundoff null each apply)
        return Pd(l0(Pp(x)))

    lmin, lmax = _lanczos_extremal_eigs_precond(s_hat, smoother, n0, steps=50,
                                                seed=0, project=Pp)
    lmin = max(lmin, lmax * 1e-5)
    cheb = make_chebyshev_upper(s_hat, smoother, lmin, lmax, int(deg))

    def atom(r):
        return Pp(cheb(Pd(r)))
    return atom


def group_k1_hx(seq, ops_k1, args, W, setup_ms):
    """k=1 HX (projected P_A + P_B; FD block_fd P_A; pure -- no Chebyshev), rank-1,
    both BC. Built with the SAME matvec/rhs construction as the jacobi baseline
    (group_saddle_jacobi: apply_laplacian_approx rhs, apply_stiffness/derivative
    matvecs, tensor M_0 lower precond, harmonic deflation) so jacobi and HX go
    through identical steps -- only the upper preconditioner differs. The jacobi
    baseline for k=1 is produced by group_saddle_jacobi (shared with k=2/k=3)."""
    k = 1
    for dirichlet in (True, False):
        bc = "dbc" if dirichlet else "free"
        n_upper = _sz(seq, k, dirichlet)
        n_lower = _sz(seq, k - 1, dirichlet)
        try:
            vs_upper = _nullspace_vectors(ops_k1, k, dirichlet)
            n_harm = int(jnp.asarray(vs_upper).shape[0])

            def mass_upper(v, d=dirichlet):
                return apply_mass_matrix(seq, ops_k1, v, k, dirichlet=d)

            def a_matvec(v, d=dirichlet):
                return apply_laplacian_approx(seq, ops_k1, v, k, dirichlet=d)

            def stiffness_matvec(v, d=dirichlet):
                return apply_stiffness(seq, ops_k1, v, k, dirichlet=d)

            def derivative_matvec(s, d=dirichlet):
                return apply_derivative_matrix(seq, ops_k1, s, k - 1, dirichlet_in=d,
                                               dirichlet_out=d, transpose=False)

            def derivative_t_matvec(u, d=dirichlet):
                return apply_derivative_matrix(seq, ops_k1, u, k - 1, dirichlet_in=d,
                                               dirichlet_out=d, transpose=True)

            def mass_lower_matvec(s, d=dirichlet):
                return apply_mass_matrix(seq, ops_k1, s, k - 1, dirichlet=d)

            def lower_precond(rhs, d=dirichlet):
                return apply_mass_matrix_preconditioner(seq, ops_k1, rhs, k - 1,
                                                        dirichlet=d, kind="tensor")

            ap = make_apply_routines(seq, ops_k1, pa_mode="block_fd",
                                     grad_project=True, dirichlet_flag=dirichlet)
            keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
            rhs_batch = jnp.stack(
                [a_matvec(jax.random.normal(kk, (n_upper,), dtype=jnp.float64))
                 for kk in keys], axis=0)
            jax.block_until_ready(rhs_batch)

            solve = make_saddle_solve(
                stiffness_matvec, derivative_matvec, derivative_t_matvec,
                mass_lower_matvec, ap["projected_p_a_plus_p_b"], lower_precond,
                n_upper=n_upper, n_lower=n_lower, tol=args.cg_tol,
                maxiter=args.cg_maxiter,
                vs_upper=(vs_upper if n_harm > 0 else None),
                mass_upper_matvec=(mass_upper if n_harm > 0 else None))
            st = time_saddle_solve(solve, rhs_batch, rel_tol=args.cg_tol)
            W.stats(case="K1", k=1, bc=bc, method="HX", rank=1, config="pure",
                    stats=st, setup_ms=setup_ms)
        except Exception:
            W.err(case="K1", k=1, bc=bc, method="HX", rank=1, config="pure",
                  msg=traceback.format_exc(), setup_ms=setup_ms)


def group_saddle_jacobi(seq, ops, rank, k, schur_mode, args, W, setup_ms):
    """k=2 / k=3 Schur-Jacobi-only saddle. schur_mode in {tensor_probe, diag}."""
    config = schur_mode
    rtag = rank if schur_mode == "tensor_probe" else "-"
    for dirichlet in (True, False):
        bc = "dbc" if dirichlet else "free"
        n_upper = _sz(seq, k, dirichlet)
        n_lower = _sz(seq, k - 1, dirichlet)
        try:
            diaginv = _get_schur_diaginv(ops, k, dirichlet, schur_mode)
            if diaginv is None:
                raise RuntimeError(f"schur diaginv None (k={k},{bc},{schur_mode})")

            vs_upper = _nullspace_vectors(ops, k, dirichlet)
            n_harm = int(jnp.asarray(vs_upper).shape[0])

            def mass_upper(v, d=dirichlet):
                return apply_mass_matrix(seq, ops, v, k, dirichlet=d)

            def a_matvec(v, d=dirichlet):
                return apply_laplacian_approx(seq, ops, v, k, dirichlet=d)

            def stiffness_matvec(v, d=dirichlet):
                return apply_stiffness(seq, ops, v, k, dirichlet=d)

            def derivative_matvec(s, d=dirichlet):
                return apply_derivative_matrix(seq, ops, s, k - 1, dirichlet_in=d,
                                               dirichlet_out=d, transpose=False)

            def derivative_t_matvec(u, d=dirichlet):
                return apply_derivative_matrix(seq, ops, u, k - 1, dirichlet_in=d,
                                               dirichlet_out=d, transpose=True)

            def mass_lower_matvec(s, d=dirichlet):
                return apply_mass_matrix(seq, ops, s, k - 1, dirichlet=d)

            def lower_precond(rhs, d=dirichlet):
                return apply_mass_matrix_preconditioner(seq, ops, rhs, k - 1,
                                                        dirichlet=d, kind="tensor")

            keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
            rhs_batch = jnp.stack(
                [a_matvec(jax.random.normal(kk, (n_upper,), dtype=jnp.float64))
                 for kk in keys], axis=0)
            jax.block_until_ready(rhs_batch)

            solve = make_saddle_solve(
                stiffness_matvec, derivative_matvec, derivative_t_matvec,
                mass_lower_matvec, lambda r, di=diaginv: di * r, lower_precond,
                n_upper=n_upper, n_lower=n_lower, tol=args.cg_tol, maxiter=args.cg_maxiter,
                vs_upper=(vs_upper if n_harm > 0 else None),
                mass_upper_matvec=(mass_upper if n_harm > 0 else None))
            st = time_saddle_solve(solve, rhs_batch, rel_tol=args.cg_tol)
            W.stats(case=f"K{k}", k=k, bc=bc, method="jacobi", rank=rtag,
                    config=config, stats=st, setup_ms=setup_ms)
        except Exception:
            W.err(case=f"K{k}", k=k, bc=bc, method="jacobi", rank=rtag,
                  config=config, msg=traceback.format_exc(), setup_ms=setup_ms)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--ns", required=True, help="n_r,n_t,n_z")
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--out", required=True, help="CSV output path (appended).")
    ap.add_argument("--n-rhs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=10000)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--cheb-degs", type=str, default="2,4,8",
                    help="Fixed Chebyshev degrees for the mass cheb-jac/cheb-tensor "
                         "competitors (the only Chebyshev users; k0/k1 L0-cheb dropped).")
    ap.add_argument("--mass-ranks", type=str, default="1,2",
                    help="Ranks for tensor MASS + k2/k3 tensor_probe jacobi "
                         "(mass-only assembly; safe at high rank). k=1 HX is "
                         "always rank-1; k=0 Laplacian rank-1.")
    ap.add_argument("--groups", type=str, default="mass,k0,k1,k2,k3")
    cli = ap.parse_args()

    ns = tuple(int(x) for x in cli.ns.split(","))
    deg_list = [int(x) for x in cli.cheb_degs.split(",") if x.strip()]   # mass cheb only
    mass_ranks = [int(x) for x in cli.mass_ranks.split(",") if x.strip()]
    groups = set(g.strip() for g in cli.groups.split(",") if g.strip())
    args = SimpleNamespace(
        ns=ns, p=cli.p, geometry=cli.geometry, cg_tol=cli.cg_tol,
        cg_maxiter=cli.cg_maxiter, n_rhs=cli.n_rhs, seed=cli.seed,
        epsilon=cli.epsilon, kappa=cli.kappa, r0=cli.r0, nfp=cli.nfp)

    base = {"geometry": cli.geometry, "n_r": ns[0], "n_t": ns[1], "n_z": ns[2],
            "p": cli.p, "n_rhs": cli.n_rhs}
    W = Writer(cli.out, base)
    print(f"[cell] geometry={cli.geometry} ns={ns} p={cli.p} n_rhs={cli.n_rhs} "
          f"maxiter={cli.cg_maxiter} groups={sorted(groups)}", flush=True)

    t0 = time.perf_counter()
    seq = build_sequence(args)
    ops_b = base_ops(seq)
    # Warm matrix-free mass-core caches eagerly (host numpy build, cannot trace).
    for kk in (0, 1, 2, 3):
        jax.block_until_ready(apply_mass_matrix(
            seq, ops_b, jnp.zeros((_sz(seq, kk, False),), dtype=jnp.float64), kk,
            dirichlet=False))
    setup_ms = (time.perf_counter() - t0) * 1e3
    print(f"[cell] build+base assembled in {setup_ms:.1f} ms", flush=True)

    # rank-{1,2,3} tensor mass ops (rank-3 = phase2 bridge)
    ops_mass = {}
    for r in mass_ranks:
        try:
            ops_mass[r] = mass_ops(seq, ops_b, r)
        except Exception:
            W.err(case="assembly", k="", bc="-", method=f"mass_ops(rank={r})",
                  rank=r, config="-", msg=traceback.format_exc(), setup_ms=setup_ms)

    if "mass" in groups and ops_mass:
        group_mass(seq, ops_b, ops_mass, deg_list, W, args, setup_ms)
        print("[cell] mass done", flush=True)

    if "k0" in groups and 1 in ops_mass:
        try:
            ops_lap = k0_lap_ops(seq, ops_mass[1])
            group_k0_lap(seq, ops_lap, args, W, setup_ms)
        except Exception:
            W.err(case="K0", k=0, bc="-", method="assembly", rank=1, config="-",
                  msg=traceback.format_exc(), setup_ms=setup_ms)
        print("[cell] k0 done", flush=True)

    # k=1, k=2, k=3 share an IDENTICAL jacobi baseline: tensor_probe Schur-Jacobi
    # via group_saddle_jacobi (rank sweep, both BC, harmonic-deflated). k=1
    # ADDITIONALLY runs the HX competitor (group_k1_hx) on the same matvec/rhs
    # construction; k=2/k=3 are jacobi-only (per spec).
    for kk in (1, 2, 3):
        if f"k{kk}" not in groups:
            continue
        for r in mass_ranks:
            if r not in ops_mass:
                continue
            try:
                osad = saddle_ops(seq, ops_mass[r], kk, "tensor_probe")
                group_saddle_jacobi(seq, osad, r, kk, "tensor_probe", args, W, setup_ms)
            except Exception:
                W.err(case=f"K{kk}", k=kk, bc="-", method="assembly", rank=r,
                      config="tensor_probe", msg=traceback.format_exc(), setup_ms=setup_ms)
        if kk == 1:
            try:
                ok1 = k1_ops(seq, ops_mass[1], 1)
                group_k1_hx(seq, ok1, args, W, setup_ms)
            except Exception:
                W.err(case="K1", k=1, bc="-", method="hx-assembly", rank=1, config="-",
                      msg=traceback.format_exc(), setup_ms=setup_ms)
        print(f"[cell] k{kk} done", flush=True)

    print(f"[cell] DONE -> {cli.out}", flush=True)


if __name__ == "__main__":
    main()
