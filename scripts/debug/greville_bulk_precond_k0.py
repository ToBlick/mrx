"""Greville-collocation bulk mass preconditioner -- k=0 proof of concept.

Tests the route in docs/preconditioner_plan.md against the CP/NTF tensor route
on the *same* bulk block. The surgery-Schur split (handled by the production
machinery) isolates a clean tensor-product bulk; we only precondition that.

For k=0 the bulk is the extracted M_0 restricted to ``bulk_indices`` (the polar
axis rows stripped), an ``(nr_bulk, nt, nz)`` tensor with radial DOFs
``[2 : nr-dirichlet]``. Three bulk preconditioners are compared:

  * jacobi    : 1 / diag(M_bulk)                       (cheap reference)
  * tensor    : CP/NTF rank-r separable inverse        (current production)
  * greville  : D^{-1/2} (M0_r^{-1} x M0_t^{-1} x M0_z^{-1}) D^{-1/2}
                M0_* = UNWEIGHTED 1D masses; D = J at the bulk Greville points.

Greville pulls the (non-separable) Jacobian weight out as a pointwise diagonal
collocated at the Greville abscissae, sandwiching the exactly-separable
unweighted tensor mass -- SPD by construction, no rank, no CP fit.

Reports, per geometry: PCG iterations to 1e-10 and the preconditioned
eigenvalue spread kappa = lambda_max / lambda_min of P^{-1} M_bulk.

Run:  python scripts/debug/greville_bulk_precond_k0.py --geometry toroid
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.geometry import compute_geometry_terms  # noqa: E402
from mrx.operators import assemble_mass_surgery_preconditioner  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    select_boundary_data,
    _apply_extracted_submatrix,
    _assemble_weighted_1d_mass,
    _restrict_radial_mass,
    _bulk_tensor_shape,
    _k0_bulk_weight_tensor,
    _build_diagonal_tensor_block_factors,
    _apply_tensor_diagonal_block_preconditioner,
)


# --------------------------------------------------------------------------- #
# Bulk operator + preconditioners
# --------------------------------------------------------------------------- #
def get_bulk_true_apply(surgery):
    bulk_idx = jnp.arange(surgery.surgery_size, surgery.apply_data.size, dtype=jnp.int32)
    return (lambda x: _apply_extracted_submatrix(surgery.apply_data, bulk_idx, bulk_idx, x),
            int(surgery.apply_data.size - surgery.surgery_size))


def build_greville_apply(seq, bulk_shape, dirichlet):
    """Greville-collocation bulk inverse apply for k=0."""
    nr_bulk, nt, nz = bulk_shape

    # Unweighted 1D masses; radial restricted to the bulk DOFs [2 : 2+nr_bulk].
    raw_mass_r = _assemble_weighted_1d_mass(seq.basis_r_jk, seq.quad.w_x)
    M0_r = _restrict_radial_mass(raw_mass_r, 2, nr_bulk)
    M0_t = _assemble_weighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M0_z = _assemble_weighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    inv_r = jnp.linalg.inv(M0_r)
    inv_t = jnp.linalg.inv(M0_t)
    inv_z = jnp.linalg.inv(M0_z)

    # Bulk Greville grid (r, theta, zeta order, matching the bulk tensor layout).
    grev_r = seq.basis_0.Λ[0].greville_points()[2:2 + nr_bulk]
    grev_t = seq.basis_0.Λ[1].greville_points()
    grev_z = seq.basis_0.Λ[2].greville_points()
    rr, tt, zz = jnp.meshgrid(grev_r, grev_t, grev_z, indexing="ij")
    pts = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)

    # D = J(greville).  k=0 weight is the Jacobian determinant.
    _, _, jac = compute_geometry_terms(seq.map, pts)
    D = jnp.asarray(jac).reshape(nr_bulk, nt, nz)
    # D MUST be positive for SPD. An orientation-preserving map has J>0, but an
    # *interpolated* map (W7-X) can fold (J<=0) or return NaN between quad points,
    # exactly where the Greville abscissae land. Floor to a small positive
    # fraction of the positive median so D^{-1/2} stays finite & SPD.
    finite_pos = D[jnp.isfinite(D) & (D > 0)]
    scale = jnp.median(finite_pos) if finite_pos.size > 0 else jnp.asarray(1.0)
    n_bad = int(((~jnp.isfinite(D)) | (D <= 0)).sum())
    if n_bad:
        print(f"  WARNING: D had {n_bad}/{D.size} non-positive/NaN Greville points "
              f"(floored to {float(1e-6 * scale):.3e}); interpolated map likely folds there",
              flush=True)
    D = jnp.where(jnp.isfinite(D) & (D > 0), D, 1e-6 * scale)
    inv_sqrt_D = 1.0 / jnp.sqrt(D)

    def apply(v):
        f = jnp.asarray(v).reshape(nr_bulk, nt, nz) * inv_sqrt_D
        f = jnp.einsum("ij,jkl->ikl", inv_r, f)
        f = jnp.einsum("ij,kjl->kil", inv_t, f)
        f = jnp.einsum("ij,klj->kli", inv_z, f)
        f = f * inv_sqrt_D
        return f.reshape(-1)

    return apply


def build_cp_apply(seq, weight_tensor, bulk_shape, rank, true_apply):
    factors = _build_diagonal_tensor_block_factors(
        seq, weight_tensor, bulk_shape, rank,
        radial_basis=seq.basis_r_jk, theta_basis=seq.basis_t_jk, zeta_basis=seq.basis_z_jk,
        radial_weights=seq.quad.w_x, theta_weights=seq.quad.w_y, zeta_weights=seq.quad.w_z,
        radial_start=2, cp_maxiter=100, cp_tol=1e-9, cp_ridge=1e-12,
        radial_baseline=None, prior_terms=None, chebyshev_steps=0,
        chebyshev_lanczos_iterations=16, chebyshev_lanczos_max_eig_inflation=1.1,
        chebyshev_lanczos_min_eig_deflation=0.85, chebyshev_lanczos_min_eig_floor_fraction=1e-3,
        chebyshev_seed=100, richardson_steps=0, richardson_omega=1.0,
        true_block_apply=true_apply,
    )
    return lambda v: _apply_tensor_diagonal_block_preconditioner(factors, v)


# --------------------------------------------------------------------------- #
# Linear algebra helpers (host)
# --------------------------------------------------------------------------- #
def dense_from_apply(apply, n):
    cols = []
    eye = np.eye(n)
    for j in range(n):
        cols.append(np.asarray(jax.device_get(apply(jnp.asarray(eye[:, j])))))
    return np.stack(cols, axis=1)


def pcg(A, Minv, n, tol=1e-10, maxiter=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    b = b / np.linalg.norm(b)
    x = np.zeros(n)
    r = b - A @ x
    z = Minv @ r
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(b)
    for it in range(1, maxiter + 1):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) / bnorm < tol:
            return it
        z = Minv @ r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter


def spread(A, Minv):
    # eigenvalues of Minv @ A (SPD product, real positive)
    ev = np.linalg.eigvals(Minv @ A).real
    ev = ev[ev > 1e-14 * ev.max()]
    return ev.min(), ev.max(), ev.max() / ev.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid",
                    choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[8, 16, 8])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--dirichlet", action="store_true")
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry,
                          cg_tol=1e-10, cg_maxiter=10000, epsilon=args.epsilon,
                          kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    print(f"=== geometry={args.geometry} ns={tuple(args.ns)} p={args.p} "
          f"rank={args.rank} dirichlet={args.dirichlet} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0,))
    surgery = select_boundary_data(ops.mass_preconds.surgery.k0, args.dirichlet, "Mass surgery k=0")

    bulk_apply, n = get_bulk_true_apply(surgery)
    bulk_shape = _bulk_tensor_shape(seq, args.dirichlet)
    print(f"bulk shape {bulk_shape} -> n={n}", flush=True)

    weight_tensor = _k0_bulk_weight_tensor(seq)
    grev = build_greville_apply(seq, bulk_shape, args.dirichlet)
    cp = build_cp_apply(seq, weight_tensor, bulk_shape, args.rank, bulk_apply)

    print("densifying bulk operator (probe)...", flush=True)
    A = dense_from_apply(bulk_apply, n)
    A = 0.5 * (A + A.T)
    diagA = np.diag(A)

    preconds = {
        "jacobi": np.diag(1.0 / diagA),
        f"tensor(r={args.rank})": dense_from_apply(cp, n),
        "greville": dense_from_apply(grev, n),
    }

    print(f"\n{'precond':18} {'cg_iters':>9} {'lam_min':>11} {'lam_max':>11} {'kappa':>11}")
    for name, Minv in preconds.items():
        Minv = 0.5 * (Minv + Minv.T)
        it = pcg(A, Minv, n)
        lo, hi, k = spread(A, Minv)
        print(f"{name:18} {it:>9d} {lo:>11.3e} {hi:>11.3e} {k:>11.3e}", flush=True)


if __name__ == "__main__":
    main()
