"""Greville-collocation bulk Laplacian preconditioner -- k=0 proof of concept.

The k=0 Hodge Laplacian is the stiffness K_0 = G_0^T M_1 G_0. On the bulk it
separates into THREE Kronecker terms with DIFFERENT metric channels:

    alpha_rr (K_r x M_t x M_z) + alpha_tt (M_r x K_t x M_z) + alpha_zz (M_r x M_t x K_z)

where alpha_aa = J g^{aa} (non-separable). The production route CP-fits each
channel and Lynch-diagonalises -- but the three channels do NOT share a per-axis
eigenbasis, so the FD is inexact (rank-1 "atom" ships; rank>1 blows up -- the
"basement" / free-BC stall, see docs/hiptmair_xu_preconditioner.md).

Greville route: move ALL geometry into a single pointwise collocation diagonal D
(at the 0-form Greville abscissae) and use the UNWEIGHTED atoms, which DO share an
exact per-axis eigenbasis (generalised eig of (M0_a, K0_a)). The constant-
coefficient Laplacian is then inverted EXACTLY by the additive fast-diagonalisation
denominator alpha_r lam_r + alpha_t lam_t + alpha_z lam_z (mrx.operators._fd_apply_3d,
currently dead code). The directional anisotropy enters as three scalar constants
alpha_a; spatial variation enters as D.

    P^{-1} = D^{-1/2} [ c_r K0_r x M0_t x M0_z + c_t M0_r x K0_t x M0_z
                        + c_z M0_r x M0_t x K0_z ]^{-1} D^{-1/2}

with c_a = mean(alpha_aa / D) at the Greville points, D a scalar profile.
SPD by construction, exact FD, NO rank, NO CP fit.

Reports kappa of P^{-1}A, PCG iters, and final residual (the free-BC stall test).
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
    dense_stiffness_matrix,
    _dense_incidence_1d,
    _assemble_unweighted_1d_mass,
    _assemble_weighted_1d_stiffness,
    _restrict_radial_window,
    _k0_stiffness_diagonal_metric_tensors,
    _bulk_tensor_shape,
    _fd_apply_3d,
)
from mrx.preconditioners import _simultaneous_diagonalize_pair  # noqa: E402


def build_greville_laplacian_apply(seq, bulk_shape, d_mode, eps):
    """Greville-collocation bulk inverse for the k=0 Hodge Laplacian."""
    nr_bulk, nt, nz = bulk_shape
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    # UNWEIGHTED 1D atoms (radial restricted to the bulk window [2:2+nr_bulk]).
    M0_r = _restrict_radial_window(_assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), 2, nr_bulk)
    M0_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M0_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K0_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x, g_r), 2, nr_bulk)
    K0_t = _assemble_weighted_1d_stiffness(seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y, g_t)
    K0_z = _assemble_weighted_1d_stiffness(seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z, g_z)

    # Per-axis simultaneous diagonalisation (shared exact eigenbasis).
    V_r, lam_r = _simultaneous_diagonalize_pair(M0_r, K0_r)
    V_t, lam_t = _simultaneous_diagonalize_pair(M0_t, K0_t)
    V_z, lam_z = _simultaneous_diagonalize_pair(M0_z, K0_z)

    # Metric channels at the 0-form bulk Greville grid.
    grev_r = seq.basis_0.Λ[0].greville_points()[2:2 + nr_bulk]
    grev_t = seq.basis_0.Λ[1].greville_points()
    grev_z = seq.basis_0.Λ[2].greville_points()
    rr, tt, zz = jnp.meshgrid(grev_r, grev_t, grev_z, indexing="ij")
    pts = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
    _, minv, jac = compute_geometry_terms(seq.map, pts)
    jac = jnp.asarray(jac).reshape(nr_bulk, nt, nz)
    a_rr = (jac * minv[:, 0, 0].reshape(nr_bulk, nt, nz))
    a_tt = (jac * minv[:, 1, 1].reshape(nr_bulk, nt, nz))
    a_zz = (jac * minv[:, 2, 2].reshape(nr_bulk, nt, nz))

    if d_mode == "geomean":
        D = jnp.cbrt(a_rr * a_tt * a_zz)
    elif d_mode == "jac":
        D = jac
    else:
        raise ValueError(d_mode)
    # Directional anisotropy constants c_a = mean(alpha_aa / D).
    c_r = float(jnp.mean(a_rr / D))
    c_t = float(jnp.mean(a_tt / D))
    c_z = float(jnp.mean(a_zz / D))
    alpha = (c_r, c_t, c_z)
    inv_sqrt_D = 1.0 / jnp.sqrt(D)

    def apply(v):
        f = jnp.asarray(v).reshape(nr_bulk, nt, nz) * inv_sqrt_D
        f = _fd_apply_3d(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, f, eps=eps)
        f = f * inv_sqrt_D
        return f.reshape(-1)

    return apply, alpha


def pcg(A, Minv_apply, n, tol=1e-11, maxiter=3000, seed=0, project=None):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    if project is not None:
        b = project(b)
    b /= np.linalg.norm(b)
    x = np.zeros(n)
    r = b - A @ x
    z = np.asarray(jax.device_get(Minv_apply(jnp.asarray(r))))
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(b)
    last = 1.0
    for it in range(1, maxiter + 1):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        last = np.linalg.norm(r) / bnorm
        if last < tol:
            return it, last
        z = np.asarray(jax.device_get(Minv_apply(jnp.asarray(r))))
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter, last


def spread(A, Minv_apply, n):
    Minv = np.zeros((n, n))
    eye = np.eye(n)
    for j in range(n):
        Minv[:, j] = np.asarray(jax.device_get(Minv_apply(jnp.asarray(eye[:, j]))))
    Minv = 0.5 * (Minv + Minv.T)
    ev = np.linalg.eigvals(Minv @ A).real
    ev = ev[ev > 1e-12 * ev.max()]
    return ev.min(), ev.max(), ev.max() / ev.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid",
                    choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[8, 16, 8])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--dirichlet", action="store_true")
    ap.add_argument("--d-mode", default="geomean", choices=["geomean", "jac"])
    ap.add_argument("--eps", type=float, default=0.0)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry,
                          cg_tol=1e-11, cg_maxiter=10000, epsilon=args.epsilon,
                          kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    print(f"=== LAPLACIAN k=0  geometry={args.geometry} ns={tuple(args.ns)} p={args.p} "
          f"dirichlet={args.dirichlet} D={args.d_mode} eps={args.eps} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))
    ops = assemble_mass_operators(seq, seq.geometry, operators=ops, ks=(1,))

    A_full = np.asarray(jax.device_get(dense_stiffness_matrix(seq, ops, 0, dirichlet=args.dirichlet)))
    A_full = 0.5 * (A_full + A_full.T)
    bulk_shape = _bulk_tensor_shape(seq, args.dirichlet)
    nb = int(np.prod(bulk_shape))
    core = A_full.shape[0] - nb
    A = A_full[core:, core:]
    print(f"bulk shape {bulk_shape} -> n={nb} (core={core})", flush=True)

    t0 = time.perf_counter()
    grev, alpha = build_greville_laplacian_apply(seq, bulk_shape, args.d_mode, args.eps)
    grev(jnp.zeros(nb)).block_until_ready()
    setup_ms = (time.perf_counter() - t0) * 1e3
    print(f"directional constants c = ({alpha[0]:.4g}, {alpha[1]:.4g}, {alpha[2]:.4g})  "
          f"setup {setup_ms:.1f} ms", flush=True)

    # Null projection for free BC (constant bulk mode ~ A's smallest eigenvector).
    project = None
    if not args.dirichlet:
        w, V = np.linalg.eigh(A)
        null = V[:, 0]  # smallest-eigenvalue direction
        project = lambda v, null=null: v - null * (null @ v)

    diagA = np.diag(A)
    jac_apply = lambda v: jnp.asarray(np.asarray(v)) / jnp.asarray(diagA)

    print(f"\n{'precond':14} {'cg_iters':>9} {'final_res':>11} {'lam_min':>11} {'lam_max':>11} {'kappa':>10} {'solve_ms':>9}")
    for name, app in (("jacobi", jac_apply), ("greville", grev)):
        t0 = time.perf_counter()
        it, res = pcg(A, app, nb, project=project)
        solve_ms = (time.perf_counter() - t0) * 1e3
        lo, hi, k = spread(A, app, nb)
        print(f"{name:14} {it:>9d} {res:>11.2e} {lo:>11.3e} {hi:>11.3e} {k:>10.2e} {solve_ms:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
