"""Greville-collocation bulk MASS preconditioner -- all blocks k=0,1,2,3.

Generalises scripts/debug/greville_bulk_precond_k0.py to every de Rham mass
block. Each block splits (via the production surgery-Schur machinery) into a
clean tensor-product bulk; for k=1,2 the bulk has three vector components, each
its own tensor block. For each component we build the Greville sandwich

    P^{-1} = D^{-1/2} (M0_r^{-1} x M0_t^{-1} x M0_z^{-1}) D^{-1/2}

with UNWEIGHTED 1D masses M0_a (degree p on primal axes, p-1 on the
differentiated axis -- using seq.basis_0.dΛ[axis]) and D the metric weight at the
component's Greville abscissae:
    k=0: J          k=3: 1/J
    k=1 comp i: J g^{ii}      k=2 comp i: g_{ii}/J
The differentiated-axis Greville points are the degree-(p-1) points
(seq.basis_0.dΛ[axis].s.greville_points()). D is floored positive (SPD).

Verification mode (default): matrix-free PCG on the true bulk operator, reporting
iters / wall time / final residual. Use --spectrum for dense kappa (small only).
Tensor/Jacobi baselines are NOT re-run here -- read them from existing sweeps.

Run:  python scripts/debug/greville_bulk_precond.py --geometry toroid --k 0 1 2 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.geometry import compute_geometry_terms  # noqa: E402
from mrx.spline_bases import SplineBasis  # noqa: E402
from mrx.local_assembly import build_matrixfree_mass_apply  # noqa: E402
from mrx.operators import assemble_mass_surgery_preconditioner  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    select_boundary_data,
    _apply_extracted_submatrix,
    _assemble_weighted_1d_mass,
    _restrict_radial_mass,
    _bulk_tensor_shape,
    _tensor_block_indices_k1,
    _tensor_block_indices_k2,
    _arr_shape_k1, _theta_bulk_shape_k1, _zeta_bulk_shape_k1,
    _r_bulk_shape_k2, _theta_shape_k2, _zeta_shape_k2,
    _k3_extracted_shape,
)


# --------------------------------------------------------------------------- #
# Per-(k, component) specifications
# --------------------------------------------------------------------------- #
# diff = (r,t,z) booleans: True => differentiated axis (degree p-1 basis dΛ).
# wkind in {'J','invJ','Jginv','ginvJ'}; comp = metric diagonal index (0/1/2).
def component_specs(seq, k, dirichlet):
    if k == 0:
        return [dict(name="k0", diff=(False, False, False), shape=_bulk_tensor_shape(seq, dirichlet),
                     wkind="J", comp=0)]
    if k == 1:
        return [
            dict(name="arr", diff=(True, False, False), shape=_arr_shape_k1(seq, dirichlet), wkind="Jginv", comp=0),
            dict(name="theta", diff=(False, True, False), shape=_theta_bulk_shape_k1(seq, dirichlet), wkind="Jginv", comp=1),
            dict(name="zeta", diff=(False, False, True), shape=_zeta_bulk_shape_k1(seq, dirichlet), wkind="Jginv", comp=2),
        ]
    if k == 2:
        return [
            dict(name="r", diff=(False, True, True), shape=_r_bulk_shape_k2(seq, dirichlet), wkind="ginvJ", comp=0),
            dict(name="theta", diff=(True, False, True), shape=_theta_shape_k2(seq, dirichlet), wkind="ginvJ", comp=1),
            dict(name="zeta", diff=(True, True, False), shape=_zeta_shape_k2(seq, dirichlet), wkind="ginvJ", comp=2),
        ]
    if k == 3:
        return [dict(name="k3", diff=(True, True, True), shape=_k3_extracted_shape(seq), wkind="invJ", comp=0)]
    raise ValueError(k)


def axis_basis_jk(seq, axis, is_diff):
    primal = [seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk][axis]
    deriv = [seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk][axis]
    return deriv if is_diff else primal


def axis_greville(seq, axis, is_diff):
    """Greville abscissae of the collocation basis on this axis, one per DOF.

    Non-differentiated: the degree-p primal basis. Differentiated: the
    degree-(p-1) derivative basis -- but built FRESH as a proper SplineBasis so
    its knot vector matches its degree. (``dΛ[axis].s`` inherits the parent's
    degree-p knot vector while declaring degree p-1, which puts a spurious DOUBLE
    Greville point at a clamped boundary -- the artifact that made a bulk DOF
    appear to sit at r=0 even though the r=0 spline is in surgery.)
    """
    if not is_diff:
        return seq.basis_0.Λ[axis].greville_points()
    d = seq.basis_0.dΛ[axis]
    return SplineBasis(int(d.n), int(d.p), d.type).greville_points()


def weight_at(wkind, comp, metric, minv, jac):
    if wkind == "J":
        return jac
    if wkind == "invJ":
        return 1.0 / jac
    if wkind == "Jginv":      # k=1: J g^{ii}
        return jac * minv[:, comp, comp]
    if wkind == "ginvJ":      # k=2: g_{ii} / J
        return metric[:, comp, comp] / jac
    raise ValueError(wkind)


def build_greville_component(seq, spec):
    nr, ntc, nzc = (int(s) for s in spec["shape"])
    diff = spec["diff"]
    radial_start = 1 if diff[0] else 2

    rbas = axis_basis_jk(seq, 0, diff[0])
    tbas = axis_basis_jk(seq, 1, diff[1])
    zbas = axis_basis_jk(seq, 2, diff[2])
    M0_r = _restrict_radial_mass(_assemble_weighted_1d_mass(rbas, seq.quad.w_x), radial_start, nr)
    M0_t = _assemble_weighted_1d_mass(tbas, seq.quad.w_y)
    M0_z = _assemble_weighted_1d_mass(zbas, seq.quad.w_z)
    inv_r, inv_t, inv_z = jnp.linalg.inv(M0_r), jnp.linalg.inv(M0_t), jnp.linalg.inv(M0_z)

    grev_r = axis_greville(seq, 0, diff[0])[radial_start:radial_start + nr]
    grev_t = axis_greville(seq, 1, diff[1])
    grev_z = axis_greville(seq, 2, diff[2])
    # Nudge CLAMPED-axis endpoint Greville points (r=0 / r=1) a hair inward.
    # A spline map's clamped-endpoint evaluate() has a constant special-case
    # branch, so jacfwd AT the exact endpoint yields a zero DF column -> det=0
    # (spurious J=0, seen on W7-X at the r=1 layer). Periodic axes are already
    # interior. This recovers the correct positive boundary weight; the metric
    # of an analytic map is unchanged to 1e-7.
    types = seq.basis_0.types
    eps = 1e-7
    if types[0] == "clamped":
        grev_r = jnp.clip(grev_r, eps, 1.0 - eps)
    if types[1] == "clamped":
        grev_t = jnp.clip(grev_t, eps, 1.0 - eps)
    if types[2] == "clamped":
        grev_z = jnp.clip(grev_z, eps, 1.0 - eps)
    rr, tt, zz = jnp.meshgrid(grev_r, grev_t, grev_z, indexing="ij")
    pts = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
    metric, minv, jac = compute_geometry_terms(seq.map, pts)
    D = jnp.asarray(weight_at(spec["wkind"], spec["comp"], metric, minv, jac)).reshape(nr, ntc, nzc)
    # D MUST be positive (SPD). Degenerate collocation points -- the clamped
    # axis spline has its Greville abscissa AT r=0 where the metric weight
    # vanishes / the interpolated map may fold -- are replaced by the positive
    # median (NOT a tiny floor, which would spike 1/sqrt(D) into a spurious
    # near-null mode). That near-axis region is surgery-corrected in the full
    # preconditioner anyway, so a smooth ~average value there is the safe choice.
    valid = jnp.isfinite(D) & (D > 0)
    fin = D[valid]
    scale = jnp.median(fin) if fin.size > 0 else jnp.asarray(1.0)
    n_bad = int((~valid).sum())
    D = jnp.where(valid, D, scale)
    inv_sqrt_D = 1.0 / jnp.sqrt(D)

    def apply(v):
        f = jnp.asarray(v).reshape(nr, ntc, nzc) * inv_sqrt_D
        f = jnp.einsum("ij,jkl->ikl", inv_r, f)
        f = jnp.einsum("ij,kjl->kil", inv_t, f)
        f = jnp.einsum("ij,klj->kli", inv_z, f)
        f = f * inv_sqrt_D
        return f.reshape(-1)

    return apply, nr * ntc * nzc, n_bad


def true_applies(seq, surg, k, dirichlet):
    """Return {comp_name: (true_bulk_apply, n)} for block k. ``surg`` holds the
    assembled mass-surgery factors (unused for k=3)."""
    if k == 3:
        e = seq.e3_dbc if dirichlet else seq.e3
        e_T = seq.e3_dbc_T if dirichlet else seq.e3_T
        mass_raw = build_matrixfree_mass_apply(seq, 3)
        app = lambda x: e @ mass_raw(e_T @ x)
        return {"k3": (app, int(e.shape[0]))}
    out = {}
    if k == 0:
        s = select_boundary_data(surg.mass_preconds.surgery.k0, dirichlet, "k0")
        idx = jnp.arange(s.surgery_size, s.apply_data.size, dtype=jnp.int32)
        out["k0"] = (lambda x, s=s, idx=idx: _apply_extracted_submatrix(s.apply_data, idx, idx, x),
                     int(s.apply_data.size - s.surgery_size))
    elif k == 1:
        s = select_boundary_data(surg.mass_preconds.surgery.k1, dirichlet, "k1")
        bi = _tensor_block_indices_k1(seq, dirichlet)
        for nm, key in (("arr", "r"), ("theta", "theta_bulk"), ("zeta", "zeta_bulk")):
            idx = bi[key]
            out[nm] = (lambda x, s=s, idx=idx: _apply_extracted_submatrix(s.apply_data, idx, idx, x), int(idx.shape[0]))
    elif k == 2:
        s = select_boundary_data(surg.mass_preconds.surgery.k2, dirichlet, "k2")
        bi = _tensor_block_indices_k2(seq, dirichlet)
        for nm, key in (("r", "r_bulk"), ("theta", "theta"), ("zeta", "zeta")):
            idx = bi[key]
            out[nm] = (lambda x, s=s, idx=idx: _apply_extracted_submatrix(s.apply_data, idx, idx, x), int(idx.shape[0]))
    return out


# --------------------------------------------------------------------------- #
# Solvers / spectrum
# --------------------------------------------------------------------------- #
def pcg(A_apply, Minv_apply, n, tol=1e-10, maxiter=3000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n); b /= np.linalg.norm(b)
    x = np.zeros(n)
    r = b.copy()
    z = np.asarray(jax.device_get(Minv_apply(jnp.asarray(r))))
    p = z.copy(); rz = r @ z; last = 1.0
    for it in range(1, maxiter + 1):
        Ap = np.asarray(jax.device_get(A_apply(jnp.asarray(p))))
        alpha = rz / (p @ Ap)
        x += alpha * p; r -= alpha * Ap
        last = np.linalg.norm(r)
        if last < tol:
            return it, last
        z = np.asarray(jax.device_get(Minv_apply(jnp.asarray(r))))
        rz_new = r @ z
        p = z + (rz_new / rz) * p; rz = rz_new
    return maxiter, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid",
                    choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--dirichlet", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry,
                          cg_tol=args.tol, cg_maxiter=10000, epsilon=args.epsilon,
                          kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    nst = "x".join(str(v) for v in args.ns)
    print(f"=== geometry={args.geometry} ns={tuple(args.ns)} p={args.p} "
          f"dirichlet={args.dirichlet} ===", flush=True)
    t_build = time.perf_counter()
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    print(f"build_sequence {(time.perf_counter()-t_build):.1f}s", flush=True)

    surg_ks = tuple(k for k in args.k if k in (0, 1, 2))
    surg = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=surg_ks) if surg_ks else ops

    rows = []
    print(f"\n{'k':>2} {'comp':6} {'n':>7} {'cg_it':>6} {'final_res':>11} "
          f"{'setup_ms':>9} {'solve_ms':>9} {'bad_D':>6}")
    for k in args.k:
        truths = true_applies(seq, surg, k, args.dirichlet)
        for spec in component_specs(seq, k, args.dirichlet):
            nm = spec["name"]
            A_apply, n = truths[nm]
            t0 = time.perf_counter()
            grev, ng, n_bad = build_greville_component(seq, spec)
            grev(jnp.zeros(ng)).block_until_ready()
            setup_ms = (time.perf_counter() - t0) * 1e3
            assert ng == n, f"shape mismatch k={k} {nm}: greville {ng} vs true {n}"
            t0 = time.perf_counter()
            it, res = pcg(A_apply, grev, n, tol=args.tol)
            solve_ms = (time.perf_counter() - t0) * 1e3
            print(f"{k:>2} {nm:6} {n:>7d} {it:>6d} {res:>11.2e} "
                  f"{setup_ms:>9.1f} {solve_ms:>9.1f} {n_bad:>6d}", flush=True)
            rows.append((args.geometry, nst, args.p, int(args.dirichlet), k, nm, n, it, res, setup_ms, solve_ms, n_bad))

    if args.csv:
        import csv
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["geometry", "ns", "p", "dirichlet", "k", "comp", "n",
                            "cg_iters", "final_res", "setup_ms", "solve_ms", "bad_D"])
            w.writerows(rows)
        print(f"-> appended {len(rows)} rows to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
