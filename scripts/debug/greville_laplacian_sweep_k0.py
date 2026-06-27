"""Greville-collocation bulk Laplacian preconditioner -- k=0 VERIFICATION SWEEP.

Matrix-free version of greville_laplacian_bulk_k0.py for sweeping p / resolution
/ geometry (incl. W7-X). The k=0 Hodge Laplacian bulk is the extracted stiffness
restricted to bulk indices (axis core stripped); applied matrix-free via
apply_stiffness (embed -> apply -> extract). Greville inverse: exact additive FD
(_fd_apply_3d) on UNWEIGHTED atoms (shared eigenbasis) + pointwise collocation D
(metric geomean at the 0-form Greville abscissae) + directional anisotropy
constants. SPD, no rank, no CP fit.

Ports the two Greville-point fixes: clamped-endpoint coords nudged inward (spline
map jacfwd gives det=0 exactly at r=1) and non-positive D floored to the median.

Reports PCG iters / wall time / final residual (the free-BC stall test) -> CSV.
Greville ONLY; compare to existing jacobi/tensor-laplacian results offline.
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
from mrx.operators import (  # noqa: E402
    assemble_incidence_operators,
    assemble_mass_operators,
    apply_stiffness,
    _dense_incidence_1d,
    _assemble_unweighted_1d_mass,
    _assemble_weighted_1d_stiffness,
    _restrict_radial_window,
    _bulk_tensor_shape,
    _fd_apply_3d,
)
from mrx.preconditioners import _simultaneous_diagonalize_pair  # noqa: E402


def build_greville_laplacian_apply(seq, bulk_shape, d_mode, eps_shift):
    nr_bulk, nt, nz = (int(s) for s in bulk_shape)
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    M0_r = _restrict_radial_window(_assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), 2, nr_bulk)
    M0_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M0_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K0_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x, g_r), 2, nr_bulk)
    K0_t = _assemble_weighted_1d_stiffness(seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y, g_t)
    K0_z = _assemble_weighted_1d_stiffness(seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z, g_z)
    V_r, lam_r = _simultaneous_diagonalize_pair(M0_r, K0_r)
    V_t, lam_t = _simultaneous_diagonalize_pair(M0_t, K0_t)
    V_z, lam_z = _simultaneous_diagonalize_pair(M0_z, K0_z)

    grev_r = seq.basis_0.Λ[0].greville_points()[2:2 + nr_bulk]
    grev_t = seq.basis_0.Λ[1].greville_points()
    grev_z = seq.basis_0.Λ[2].greville_points()
    e = 1e-7
    if types[0] == "clamped":
        grev_r = jnp.clip(grev_r, e, 1.0 - e)
    if types[1] == "clamped":
        grev_t = jnp.clip(grev_t, e, 1.0 - e)
    if types[2] == "clamped":
        grev_z = jnp.clip(grev_z, e, 1.0 - e)
    rr, tt, zz = jnp.meshgrid(grev_r, grev_t, grev_z, indexing="ij")
    pts = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
    metric, minv, jac = compute_geometry_terms(seq.map, pts)
    jac = jnp.asarray(jac).reshape(nr_bulk, nt, nz)
    a_rr = jac * minv[:, 0, 0].reshape(nr_bulk, nt, nz)
    a_tt = jac * minv[:, 1, 1].reshape(nr_bulk, nt, nz)
    a_zz = jac * minv[:, 2, 2].reshape(nr_bulk, nt, nz)
    D = jnp.cbrt(a_rr * a_tt * a_zz) if d_mode == "geomean" else jac
    valid = jnp.isfinite(D) & (D > 0)
    scale = jnp.median(D[valid]) if D[valid].size > 0 else jnp.asarray(1.0)
    n_bad = int((~valid).sum())
    D = jnp.where(valid, D, scale)
    c_r = float(jnp.mean(a_rr / D)); c_t = float(jnp.mean(a_tt / D)); c_z = float(jnp.mean(a_zz / D))
    alpha = (c_r, c_t, c_z)
    inv_sqrt_D = 1.0 / jnp.sqrt(D)

    def apply(v):
        f = jnp.asarray(v).reshape(nr_bulk, nt, nz) * inv_sqrt_D
        f = _fd_apply_3d(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, f, eps=eps_shift)
        f = f * inv_sqrt_D
        return f.reshape(-1)

    return apply, nr_bulk * nt * nz, n_bad, alpha


def pcg(A_apply, Minv_apply, n, tol=1e-10, maxiter=3000, project=None, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    if project is not None:
        b = project(b)
    b /= np.linalg.norm(b)
    x = np.zeros(n); r = b.copy()
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
        rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
    return maxiter, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid", choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--dirichlet", action="store_true")
    ap.add_argument("--d-mode", default="geomean", choices=["geomean", "jac"])
    ap.add_argument("--eps-shift", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=10000, epsilon=args.epsilon, kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    nst = "x".join(str(v) for v in args.ns)
    print(f"=== LAPLACIAN k=0 geometry={args.geometry} ns={tuple(args.ns)} p={args.p} "
          f"dirichlet={args.dirichlet} D={args.d_mode} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))
    ops = assemble_mass_operators(seq, seq.geometry, operators=ops, ks=(1,))

    n_ext = int(seq.n0_dbc if args.dirichlet else seq.n0)
    bulk_shape = _bulk_tensor_shape(seq, args.dirichlet)
    nb = int(np.prod(bulk_shape))
    core = n_ext - nb
    bulk_idx = jnp.arange(core, n_ext, dtype=jnp.int32)

    def A_apply(x):
        full = jnp.zeros((n_ext,), dtype=jnp.float64).at[bulk_idx].set(x)
        y = apply_stiffness(seq, ops, full, 0, dirichlet=args.dirichlet)
        return y[bulk_idx]

    t0 = time.perf_counter()
    grev, ng, n_bad, alpha = build_greville_laplacian_apply(seq, bulk_shape, args.d_mode, args.eps_shift)
    grev(jnp.zeros(ng)).block_until_ready()
    setup_ms = (time.perf_counter() - t0) * 1e3
    assert ng == nb

    # Free BC: deflate the constant bulk mode (estimate via one inverse-power step
    # on the Greville inverse, then orthogonalise the PCG against it).
    project = None
    if not args.dirichlet:
        ones = np.ones(nb) / np.sqrt(nb)
        project = lambda v, u=ones: v - u * (u @ v)

    it, res = 0, np.nan
    t0 = time.perf_counter()
    it, res = pcg(A_apply, grev, nb, tol=args.tol, project=project)
    solve_ms = (time.perf_counter() - t0) * 1e3
    print(f"bulk n={nb} c=({alpha[0]:.3g},{alpha[1]:.3g},{alpha[2]:.3g}) bad_D={n_bad}", flush=True)
    print(f"{'cg_it':>6} {'final_res':>11} {'setup_ms':>9} {'solve_ms':>9}", flush=True)
    print(f"{it:>6d} {res:>11.2e} {setup_ms:>9.1f} {solve_ms:>9.1f}", flush=True)

    if args.csv:
        import csv
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["geometry", "ns", "p", "dirichlet", "d_mode", "n", "cg_iters",
                            "final_res", "setup_ms", "solve_ms", "bad_D"])
            w.writerow([args.geometry, nst, args.p, int(args.dirichlet), args.d_mode, nb, it, res, setup_ms, solve_ms, n_bad])
        print(f"-> appended to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
