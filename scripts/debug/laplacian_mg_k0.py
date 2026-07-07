"""Geometric multigrid V-cycle preconditioner for the k=0 Laplacian -- PROTOTYPE.

Plan: docs/laplacian_mg_k0_plan.md. MG acts on the BULK block of the
extracted k=0 stiffness (polar core handled by the same Schur envelope as the
production tensor-hodge preconditioner). Levels are REDISCRETIZED coarse
DeRhamSequences reusing the fine geometry map (no W7-X re-fit); transfers are
tensor-product 1D quasi-interpolation prolongations P = C_ff^{-1} C_cf at the
fine Greville points, restriction = P^T; smoothing is symmetric Chebyshev over
one of three smoother atoms (the comparison this prototype exists to measure):

  jacobi  exact diag(K_0) (sum-factorized quadrature einsums, all 9 metric
          blocks) -- locally exact for every g_aa J, no separability assumption
  fd      the production greville FD atom: D = J (g^rr g^tt g^zz)^{1/3} at
          Greville points + global per-term alpha means (the "1/3 version")
  fdax    axis-averaged FD: D = J at Greville points; per axis the generalized
          eigenpair (M_a, K_a[gbar^aa(x_a)]) with gbar^aa = quad-weighted mean
          of g^aa over the OTHER two axes (radial integration from the first
          interior breakpoint xi_1 outward -- the surgery element is core-
          handled and mean(1/r^2) over (0,1) is quadrature-divergent); denom
          lam_r+lam_t+lam_z, no alphas.
          Captures each g-factor's variation along its own axis; the cross-axis
          part (e.g. g^tt ~ 1/r^2) is the structural residual.

Coarsest solve: dense probe + symmetric pseudoinverse (exact; removes the
coarse-solve confound). Envelope: W = B_mg C0 probed once => ONE V-cycle per
preconditioner apply and a free Schur rebuild. Chebyshev needs only lam_max
(power-of-Lanczos, upper window [lam_max/w, lam_max]); NO lam_min estimation.

  python scripts/debug/laplacian_mg_k0.py --geometry cylinder --ns 8 16 8 --two-level-check
  python scripts/debug/laplacian_mg_k0.py --geometry toroid --ns 12 24 12 --smoothers jacobi,fd,fdax --csv out.csv
"""
from __future__ import annotations

import argparse
import math
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

import benchmark_graddiv_k1_preconditioner as bench  # noqa: E402
from benchmark_graddiv_k1_preconditioner import build_sequence, TYPES, BETTI  # noqa: E402
from mrx.geometry import compute_geometry_terms  # noqa: E402
from mrx.operators import (  # noqa: E402
    assemble_incidence_operators,
    assemble_laplacian_operators,
    assemble_tensor_laplacian_preconditioner,
    apply_stiffness,
    apply_mass_matrix,
    apply_laplacian_preconditioner,
    _nullspace_vectors,
    _bulk_tensor_shape,
    _dense_incidence_1d,
    _assemble_unweighted_1d_mass,
    _assemble_weighted_1d_stiffness,
    _restrict_radial_window,
    _assemble_1d_fd_eigendecomp,
    _fd_apply_3d,
    _assemble_dense_from_apply,
    _diagonal_from_matvec,
    _invert_diagonal,
    _reshape_quadrature_scalar_field,
    _reshape_quadrature_matrix_field,
    _symmetrize,
)
from mrx.preconditioners import (  # noqa: E402
    _estimate_chebyshev_lanczos_bounds_apply,
    _symmetric_pseudoinverse,
)
from mrx.solvers import solve_singular_cg  # noqa: E402


# ---------------------------------------------------------------------------
# Level construction
# ---------------------------------------------------------------------------

def coarsen_ns(ns, p, factors, dirichlet_any=True):
    """One coarsening step. Periodic axes: n//f with floor max(4, p+1).
    Clamped radial: halve ELEMENTS (e = n - p), floor n >= 5 (keeps a
    meaningful bulk window after the polar core strips 2 + the dbc drop 1).
    Returns the coarse ns, or None if no axis can coarsen further."""
    nr, nt, nz = ns
    fr, ft, fz = factors
    ang_floor = max(4, p + 1)

    e_f = nr - p
    e_c = max(2, math.ceil(e_f / fr)) if fr > 1 else e_f
    nr_c = max(5, e_c + p) if fr > 1 else nr

    def ang(n, f):
        return max(ang_floor, n // f) if f > 1 else n

    nt_c, nz_c = ang(nt, ft), ang(nz, fz)
    out = (nr_c, nt_c, nz_c)
    return None if out == tuple(ns) else out


def build_coarse_sequence(fine_seq, ns_c, p, tol, maxiter, r_scale):
    """Rediscretized coarse level reusing the FINE geometry map (no re-fit;
    jacfwd re-evaluated at the coarse quad points only). Same radial knot
    grading rule (r_scale) as the fine level."""
    cseq = bench.DeRhamSequence(
        tuple(int(v) for v in ns_c), (p, p, p), 2 * p, TYPES,
        polar=True, tol=tol, maxiter=maxiter, r_scale=r_scale,
        betti_numbers=BETTI)
    cseq.evaluate_1d()
    cseq.assemble_reference_mass_matrix()
    cseq.set_map(fine_seq.map)
    cops = cseq.get_operators()
    cops = assemble_incidence_operators(cseq, operators=cops, ks=(0,))
    cops = assemble_laplacian_operators(cseq, cseq.geometry, operators=cops, ks=(0,))
    # eager mass-core warm-up (TracerArrayConversionError guard under lax.map/jit)
    for d in (False, True):
        n = int(cseq.n0_dbc if d else cseq.n0)
        jax.block_until_ready(apply_stiffness(
            cseq, cops, jnp.zeros((n,), dtype=jnp.float64), 0, dirichlet=d))
    return cseq, cops


def bulk_slices(seq, dirichlet):
    n_ext = int(seq.n0_dbc if dirichlet else seq.n0)
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nb = int(np.prod(bulk_shape))
    core = n_ext - nb
    return n_ext, bulk_shape, nb, core


def make_bulk_A(seq, ops, dirichlet):
    n_ext, bulk_shape, nb, core = bulk_slices(seq, dirichlet)

    def A(x):
        full = jnp.zeros((n_ext,), dtype=jnp.float64).at[core:].set(x)
        return apply_stiffness(seq, ops, full, 0, dirichlet=dirichlet)[core:]

    return A, n_ext, bulk_shape, nb, core


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------

def build_prolongation_1d(fine_basis, coarse_basis):
    """P = C_ff^{-1} C_cf at the fine Greville points: the fine-space
    quasi-interpolant of a coarse function. Exact injection on nested spaces."""
    tau = fine_basis.greville_points()
    C_ff = fine_basis.collocation_matrix(tau)
    C_cf = coarse_basis.collocation_matrix(tau)
    return jnp.linalg.solve(C_ff, C_cf)


def build_bulk_transfers(fine_seq, coarse_seq, dirichlet, p_fix="none"):
    """(Pr, Pt, Pz) between the two levels' BULK grids. Radial window starts
    at 2 always; the dirichlet flag drops the LAST radial function (encoded in
    each level's own nr_bulk).

    p_fix repairs the axis-side constant-reproduction defect caused by
    dropping the two coarse axis columns (the full P reproduces constants by
    partition of unity; the window slice loses their mass in the first ~p
    fine rows). Both variants touch ONLY the axis-side deficiency -- the dbc
    outer-side truncation is left alone (error satisfies the BC there):
      rownorm: scale row i by 1/(1 - c_i), c_i = mass in dropped cols 0,1
      lump:    add the dropped cols' weights onto the first kept coarse col
    """
    Ps = []
    for ax in range(3):
        P = build_prolongation_1d(fine_seq.basis_0.Λ[ax], coarse_seq.basis_0.Λ[ax])
        if ax == 0:
            nrb_f = _bulk_tensor_shape(fine_seq, dirichlet)[0]
            nrb_c = _bulk_tensor_shape(coarse_seq, dirichlet)[0]
            c = P[2:2 + nrb_f, 0] + P[2:2 + nrb_f, 1]
            P = P[2:2 + nrb_f, 2:2 + nrb_c]
            if p_fix == "rownorm":
                P = P / (1.0 - c)[:, None]
            elif p_fix == "lump":
                P = P.at[:, 0].add(c)
            elif p_fix != "none":
                raise ValueError(f"unknown p_fix {p_fix!r}")
        Ps.append(P)
    return tuple(Ps)


def apply_P(Ps, shape_c, x_c):
    f = jnp.asarray(x_c).reshape(shape_c)
    f = jnp.einsum('ij,jkl->ikl', Ps[0], f)
    f = jnp.einsum('ij,kjl->kil', Ps[1], f)
    f = jnp.einsum('ij,klj->kli', Ps[2], f)
    return f.reshape(-1)


def apply_PT(Ps, shape_f, r_f):
    f = jnp.asarray(r_f).reshape(shape_f)
    f = jnp.einsum('ji,jkl->ikl', Ps[0], f)
    f = jnp.einsum('ji,kjl->kil', Ps[1], f)
    f = jnp.einsum('ji,klj->kli', Ps[2], f)
    return f.reshape(-1)


# ---------------------------------------------------------------------------
# Smoother atoms
# ---------------------------------------------------------------------------

def k0_bulk_stiffness_diagonal(seq, bulk_shape):
    """Exact diag(K_0) on the bulk tensor window via sum-factorized quadrature
    einsums over ALL 9 metric blocks g^{ab} J (the polar-extraction bulk rows
    are unit tensor-product rows, so the tensor-space diagonal IS the bulk
    diagonal after the radial window slice)."""
    nr_bulk, nt, nz = (int(s) for s in bulk_shape)
    # quad-field layout is (ny, nx, nz) = (theta, r, zeta); transpose to (r, theta, zeta)
    jac = jnp.transpose(_reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j), (1, 0, 2))
    minv = jnp.transpose(_reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    W3 = (seq.quad.w_x[:, None, None] * seq.quad.w_y[None, :, None]
          * seq.quad.w_z[None, None, :])
    types = seq.basis_0.types
    n0s = (seq.basis_0.nr, seq.basis_0.nt, seq.basis_0.nz)
    Phi = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    dB = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    dPhi = tuple(_dense_incidence_1d(n0s[ax], types[ax]).T @ dB[ax] for ax in range(3))

    def factor(ax, a, b):
        f1 = dPhi[ax] if ax == a else Phi[ax]
        f2 = dPhi[ax] if ax == b else Phi[ax]
        return f1 * f2  # (n_ax, nq_ax) elementwise product of the two rows

    diag = jnp.zeros((n0s[0], n0s[1], n0s[2]), dtype=jnp.float64)
    for a in range(3):
        for b in range(3):
            Wab = jac * minv[..., a, b] * W3
            t = jnp.einsum('iq,qrs->irs', factor(0, a, b), Wab)
            t = jnp.einsum('jr,irs->ijs', factor(1, a, b), t)
            t = jnp.einsum('ks,ijs->ijk', factor(2, a, b), t)
            diag = diag + t
    return diag[2:2 + nr_bulk].reshape(-1)


def fdax_axis_profiles(seq):
    """gbar^aa(x_a): quad-weighted mean of g^aa over the other two axes.

    The radial integration in the cross-axis means (pt, pz) starts at the
    first interior breakpoint xi_1: the first element is the polar-surgery
    region (core DOFs, handled exactly by the Schur envelope), and the plain
    r-mean of g^tt ~ 1/r^2 over (0,1) is quadrature-divergent -- dominated by
    the innermost Gauss points. Over [xi_1, 1] it is ~1/xi_1, finite and
    resolution-stable."""
    # quad-field layout is (ny, nx, nz) = (theta, r, zeta); transpose to (r, theta, zeta)
    minv = jnp.transpose(
        _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    p_r = seq.ps[0]
    xi1 = jnp.asarray(seq.basis_0.Λ[0].T)[p_r + 1]
    wx_cut = seq.quad.w_x * (jnp.asarray(seq.quad.x_x) >= xi1)
    wx, wy, wz = seq.quad.w_x, seq.quad.w_y, seq.quad.w_z
    sy, sz = jnp.sum(wy), jnp.sum(wz)
    sxc = jnp.sum(wx_cut)
    pr = jnp.einsum('qrs,r,s->q', minv[..., 0, 0], wy, wz) / (sy * sz)
    pt = jnp.einsum('qrs,q,s->r', minv[..., 1, 1], wx_cut, wz) / (sxc * sz)
    pz = jnp.einsum('qrs,q,r->s', minv[..., 2, 2], wx_cut, wy) / (sxc * sy)
    floor = 1e-8

    def clip(v):
        med = jnp.median(v)
        return jnp.maximum(v, floor * jnp.abs(med))

    return clip(pr), clip(pt), clip(pz)


def build_fd_atom(seq, bulk_shape, mode):
    """Additive-FD smoother atom apply. mode='fd' = production greville atom
    (unweighted 1D atoms, D = geomean, alpha means); mode='fdax' = axis-averaged
    weighted 1D stiffnesses, D = J, alpha = 1."""
    nr_bulk, nt, nz = (int(s) for s in bulk_shape)
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    if mode == "fdax":
        pr, pt, pz = fdax_axis_profiles(seq)
        kw_x, kw_y, kw_z = seq.quad.w_x * pr, seq.quad.w_y * pt, seq.quad.w_z * pz
    else:
        kw_x, kw_y, kw_z = seq.quad.w_x, seq.quad.w_y, seq.quad.w_z

    M0_r = _restrict_radial_window(_assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), 2, nr_bulk)
    M0_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M0_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K0_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(seq.basis_r_jk, seq.d_basis_r_jk, kw_x, g_r), 2, nr_bulk)
    K0_t = _assemble_weighted_1d_stiffness(seq.basis_t_jk, seq.d_basis_t_jk, kw_y, g_t)
    K0_z = _assemble_weighted_1d_stiffness(seq.basis_z_jk, seq.d_basis_z_jk, kw_z, g_z)
    V_r, lam_r = _assemble_1d_fd_eigendecomp(M0_r, K0_r)
    V_t, lam_t = _assemble_1d_fd_eigendecomp(M0_t, K0_t)
    V_z, lam_z = _assemble_1d_fd_eigendecomp(M0_z, K0_z)

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
    _, minv, jac = compute_geometry_terms(seq.map, pts)
    jac = jnp.asarray(jac).reshape(nr_bulk, nt, nz)
    if mode == "fd":
        a_rr = jac * minv[:, 0, 0].reshape(nr_bulk, nt, nz)
        a_tt = jac * minv[:, 1, 1].reshape(nr_bulk, nt, nz)
        a_zz = jac * minv[:, 2, 2].reshape(nr_bulk, nt, nz)
        D = jnp.cbrt(a_rr * a_tt * a_zz)
    else:
        D = jac
    valid = jnp.isfinite(D) & (D > 0)
    scale = jnp.median(D[valid]) if int(valid.sum()) > 0 else jnp.asarray(1.0)
    D = jnp.where(valid, D, scale)
    if mode == "fd":
        def _reduce(x):
            xv = x[jnp.isfinite(x) & (x > 0)]
            return float(jnp.mean(xv))
        alpha = (_reduce(a_rr / D), _reduce(a_tt / D), _reduce(a_zz / D))
    else:
        alpha = (1.0, 1.0, 1.0)
    inv_sqrt_D = 1.0 / jnp.sqrt(D)

    def S(v):
        f = jnp.asarray(v).reshape(nr_bulk, nt, nz) * inv_sqrt_D
        f = _fd_apply_3d(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, f, eps=0.0)
        return (f * inv_sqrt_D).reshape(-1)

    return S


def build_smoother_atom(seq, ops, dirichlet, bulk_shape, A_bulk, nb, mode):
    if mode == "jacobi":
        di = _invert_diagonal(k0_bulk_stiffness_diagonal(seq, bulk_shape))
        return lambda v: di * v
    if mode in ("fd", "fdax"):
        return build_fd_atom(seq, bulk_shape, mode)
    raise ValueError(f"unknown smoother {mode!r}")


# ---------------------------------------------------------------------------
# Chebyshev smoothing (upper-window; returns final residual for free)
# ---------------------------------------------------------------------------

def make_cheb_smoother(A, S, steps, lam_min, lam_max):
    d = 0.5 * (lam_max + lam_min)
    c = 0.5 * (lam_max - lam_min)

    def smooth(rhs):
        def body(i, st):
            x, res, dirn, alpha = st
            corr = S(res)
            beta = (0.5 * c * alpha) ** 2
            new_alpha = jnp.where(i == 0, alpha, 1.0 / (d - beta))
            new_dir = jnp.where(i == 0, corr, corr + beta * dirn)
            x = x + new_alpha * new_dir
            res = res - new_alpha * A(new_dir)
            return x, res, new_dir, new_alpha

        x, res, _, _ = jax.lax.fori_loop(
            0, steps, body,
            (jnp.zeros_like(rhs), rhs, jnp.zeros_like(rhs), 1.0 / d))
        return x, res

    return smooth


def estimate_lam_max(A, S, nb, free_bc, seed=0, iters=32):
    orth = (jnp.ones((1, nb), dtype=jnp.float64) / np.sqrt(nb)) if free_bc else None
    _, lam_max = _estimate_chebyshev_lanczos_bounds_apply(
        A, S, nb, lanczos_iterations=iters, lanczos_max_eig_inflation=1.1,
        lanczos_min_eig_deflation=0.85, lanczos_min_eig_floor_fraction=1e-3,
        seed=seed, orthogonal_vectors=orth)
    return float(lam_max)


def _psd_pseudoinverse(mat, relative_tol=1e-8):
    """Pseudoinverse restricted to the POSITIVE part of the spectrum.

    The production `_symmetric_pseudoinverse` inverts eigenvalues by
    magnitude WITH SIGN. The rebuilt free-BC Schur `ass - C0^T B C0` is
    singular-PSD analytically (constant null vector), but with a strong B_mg
    the null direction evaluates to a small value of EITHER sign; a slightly
    negative eigenvalue then survives the magnitude cutoff and is inverted to
    a huge negative -> indefinite envelope (observed min_rayleigh ~ -6e3).
    Projecting onto the positive part restores the true PSD semantics."""
    mat = _symmetrize(mat)
    w, v = jnp.linalg.eigh(mat)
    scale = jnp.max(jnp.abs(w))
    cutoff = relative_tol * jnp.where(scale > 0, scale, 1.0)
    inv = jnp.where(w > cutoff, 1.0 / w, 0.0)
    return _symmetrize((v * inv[jnp.newaxis, :]) @ v.T)


# ---------------------------------------------------------------------------
# V-cycle + Schur envelope
# ---------------------------------------------------------------------------

def make_vcycle(A_list, shape_list, smooth_list, P_list, coarse_inv):
    """Symmetric V-cycle on the finest bulk vector. Unrolled Python recursion
    (static level count) -> fully jittable. levels==1 degenerates to the
    Chebyshev smoother alone (still SPD)."""
    last = len(A_list) - 1

    def vc(r, lvl=0):
        if lvl == last and last > 0:
            return coarse_inv @ r
        if last == 0:
            x, _ = smooth_list[0](r)
            return x
        x1, r1 = smooth_list[lvl](r)                       # pre-smooth
        rc = apply_PT(P_list[lvl], shape_list[lvl], r1)     # restrict
        ec = vc(rc, lvl + 1)
        ccorr = apply_P(P_list[lvl], shape_list[lvl + 1], ec)
        r2 = r1 - A_list[lvl](ccorr)                        # 1 A-apply
        dx, _ = smooth_list[lvl](r2)                        # post-smooth
        return x1 + ccorr + dx

    return vc


def build_envelope(seq, ops, dirichlet, vcycle, nb, core, schur_mode):
    """Core/bulk Schur envelope around the V-cycle. W = B_mg C0 probed once =>
    one V-cycle per apply; Schur rebuild is then a dense product."""
    n_ext = core + nb
    factors = ops.k0_tensor_hodge_precond.dbc if dirichlet else ops.k0_tensor_hodge_precond.free
    C0 = factors.core_coupling            # (nb, core), probed by production assembly

    def core_block(rhs_c):
        full = jnp.zeros((n_ext,), dtype=jnp.float64).at[:core].set(rhs_c)
        return apply_stiffness(seq, ops, full, 0, dirichlet=dirichlet)[:core]

    t0 = time.perf_counter()
    ass = _symmetrize(_assemble_dense_from_apply(core_block, core, sequential=True))
    jax.block_until_ready(ass)
    t_ass = time.perf_counter() - t0

    t0 = time.perf_counter()
    vcycle(jnp.zeros((nb,), dtype=jnp.float64))             # eager warm-up
    W = jax.lax.map(vcycle, C0.T).T                          # (nb, core)
    jax.block_until_ready(W)
    t_w = time.perf_counter() - t0

    if schur_mode == "rebuild":
        schur_inv = _psd_pseudoinverse(_symmetrize(ass - C0.T @ W))
    else:  # 'fd': reuse the production FD-consistent Schur (SPD-safe)
        schur_inv = factors.schur_inv

    def precond(rhs):
        rhs_c, rhs_b = rhs[:core], rhs[core:]
        y = vcycle(rhs_b)
        z = schur_inv @ (rhs_c - C0.T @ y)
        x_b = y - W @ z
        return jnp.concatenate([z, x_b])

    return precond, {"ass_s": t_ass, "w_s": t_w}


# ---------------------------------------------------------------------------
# Checks / harness
# ---------------------------------------------------------------------------

def spd_check(precond, n, seed=0, npairs=3):
    key = jax.random.PRNGKey(seed)
    sym_err, min_ray = 0.0, np.inf
    for i in range(npairs):
        key, k1, k2 = jax.random.split(key, 3)
        u = jax.random.normal(k1, (n,), dtype=jnp.float64)
        v = jax.random.normal(k2, (n,), dtype=jnp.float64)
        Bu, Bv = precond(u), precond(v)
        num = float(jnp.abs(jnp.dot(u, Bv) - jnp.dot(Bu, v)))
        den = float(jnp.linalg.norm(u) * jnp.linalg.norm(Bv) + 1e-300)
        sym_err = max(sym_err, num / den)
        min_ray = min(min_ray, float(jnp.dot(u, Bu)) / float(jnp.dot(u, u)))
    return sym_err, min_ray


def timed_pcg(a, m, precond, rhs, vs, tol, maxiter):
    solve = jax.jit(lambda b: solve_singular_cg(
        a, b, mass_matvec=m, precond_matvec=precond, vs=vs, tol=tol, maxiter=maxiter))
    x, info = solve(rhs)
    jax.block_until_ready((x, info))
    t0 = time.perf_counter()
    x, info = solve(rhs)
    jax.block_until_ready((x, info))
    dt = time.perf_counter() - t0
    r = a(x) - rhs
    rel = float(jnp.linalg.norm(r) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
    return abs(int(info)), dt, rel


def zeta_diag(seq, nfp):
    zg = jnp.linspace(0.0, 1.0, 256, endpoint=False)
    pts = jnp.stack([jnp.full_like(zg, 0.5), jnp.full_like(zg, 0.3), zg], axis=-1)
    _, minv, jac = compute_geometry_terms(seq.map, pts)
    a_zz = np.asarray(jac * minv[:, 2, 2])
    spec = np.abs(np.fft.rfft(a_zz - a_zz.mean()))
    top = np.argsort(spec)[::-1][:5]
    print(f"[zeta-diag] a_zz mean={a_zz.mean():.4g} cv={a_zz.std()/a_zz.mean():.3f} "
          f"dominant zeta modes={list(top)} (map covers one field period; nfp={nfp})", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid",
                    choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--coarsen", type=int, nargs=3, default=[2, 2, 2],
                    help="per-axis coarsening factors (r theta zeta); 1 = keep")
    ap.add_argument("--smoothers", default="jacobi,fd,fdax")
    ap.add_argument("--smooth-steps", type=int, default=2)
    ap.add_argument("--cheb-window", type=float, default=4.0)
    ap.add_argument("--p-fix", default="rownorm", choices=["none", "rownorm", "lump"],
                    help="repair of the axis-side radial-transfer constant defect")
    ap.add_argument("--r-scale", type=float, default=0.5,
                    help="radial knot grading exponent (breakpoints = linspace**r_scale); "
                         "0.5 = equal-area cells in the disk, 1.0 = uniform")
    ap.add_argument("--schur", default="rebuild", choices=["rebuild", "fd"])
    ap.add_argument("--bc", default="both", choices=["dbc", "free", "both"])
    ap.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--two-level-check", action="store_true")
    ap.add_argument("--zeta-diag", action="store_true")
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry,
                          cg_tol=args.tol, cg_maxiter=args.maxiter,
                          epsilon=args.epsilon, kappa=args.kappa, r0=args.r0, nfp=args.nfp,
                          r_scale=args.r_scale)
    smoother_modes = [s.strip() for s in args.smoothers.split(",") if s.strip()]
    dirichlet_flags = {"dbc": [True], "free": [False], "both": [True, False]}[args.bc]
    ns_str = "x".join(str(v) for v in args.ns)
    print(f"=== MG k=0 Laplacian  {args.geometry} ns={tuple(args.ns)} p={args.p} "
          f"levels={args.levels} coarsen={tuple(args.coarsen)} m={args.smooth_steps} "
          f"smoothers={smoother_modes} schur={args.schur} p_fix={args.p_fix} "
          f"cheb_window={args.cheb_window} r_scale={args.r_scale} ===", flush=True)

    # ---- fine level -------------------------------------------------------
    t0 = time.perf_counter()
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0,))
    ops = assemble_tensor_laplacian_preconditioner(seq, operators=ops, ks=(0,))
    for d in (False, True):
        n = int(seq.n0_dbc if d else seq.n0)
        jax.block_until_ready(apply_stiffness(seq, ops, jnp.zeros((n,)), 0, dirichlet=d))
        jax.block_until_ready(apply_mass_matrix(seq, ops, jnp.zeros((n,)), 0, dirichlet=d))
    t_fine = time.perf_counter() - t0
    print(f"[setup] fine level + baseline precond: {t_fine:.1f} s", flush=True)
    if args.zeta_diag:
        zeta_diag(seq, args.nfp)

    # ---- coarse levels ----------------------------------------------------
    levels = [(seq, ops)]
    ns_l = [tuple(args.ns)]
    t0 = time.perf_counter()
    while len(levels) < args.levels:
        nxt = coarsen_ns(ns_l[-1], args.p, tuple(args.coarsen))
        if nxt is None:
            print(f"[setup] coarsening stopped at {ns_l[-1]} (floors reached)", flush=True)
            break
        cseq, cops = build_coarse_sequence(seq, nxt, args.p, args.tol, args.maxiter,
                                           args.r_scale)
        levels.append((cseq, cops))
        ns_l.append(nxt)
    t_coarse = time.perf_counter() - t0
    print(f"[setup] levels: {ns_l}  ({t_coarse:.1f} s)", flush=True)

    csv_f = None
    if args.csv:
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        csv_f = open(args.csv, "a")
        if new:
            csv_f.write("geometry,ns,p,bc,method,levels,coarsen,m,n,cg_it,solve_s,"
                        "ms_per_it,rel_res,lam_max0,p_const_err,sym_err,min_ray,"
                        "setup_smoother_s,setup_ass_s,setup_w_s,setup_coarse_s,"
                        "setup_dense_s\n")

    def row(bc, method, n, it, dt, rel, lam0="", pce="", se="", mr="",
            t_sm="", t_ass="", t_w="", t_dense=""):
        line = (f"{args.geometry},{ns_str},{args.p},{bc},{method},{len(levels)},"
                f"{'-'.join(map(str, args.coarsen))},{args.smooth_steps},{n},{it},"
                f"{dt:.4f},{1e3 * dt / max(it, 1):.4f},{rel:.3e},{lam0},{pce},{se},{mr},"
                f"{t_sm},{t_ass},{t_w},{t_coarse:.2f},{t_dense}\n")
        if csv_f:
            csv_f.write(line)
            csv_f.flush()

    for dirichlet in dirichlet_flags:
        bc = "dbc" if dirichlet else "free"
        free_bc = not dirichlet
        vs = _nullspace_vectors(ops, 0, dirichlet)

        def a_full(v, d=dirichlet):
            return apply_stiffness(seq, ops, v, 0, dirichlet=d)

        def m_full(v, d=dirichlet):
            return apply_mass_matrix(seq, ops, v, 0, dirichlet=d)

        n_ext, _, _, _ = bulk_slices(seq, dirichlet)
        key = jax.random.PRNGKey(args.seed + (0 if dirichlet else 7))
        rhs = a_full(jax.random.normal(key, (n_ext,), dtype=jnp.float64))

        # per-level bulk operators / shapes
        A_list, shape_list, nb_list = [], [], []
        for (sq, op) in levels:
            A, _, bsh, nb, _ = make_bulk_A(sq, op, dirichlet)
            A_list.append(A)
            shape_list.append(bsh)
            nb_list.append(nb)
        _, _, nb0, core0 = bulk_slices(seq, dirichlet)

        # transfers between consecutive levels
        P_list = [build_bulk_transfers(levels[i][0], levels[i + 1][0], dirichlet,
                                       p_fix=args.p_fix)
                  for i in range(len(levels) - 1)]
        pce = ""
        if P_list:
            ones_c = jnp.ones((nb_list[1],), dtype=jnp.float64)
            perr = jnp.abs(apply_P(P_list[0], shape_list[1], ones_c) - 1.0).reshape(shape_list[0])
            # Under dbc the constant is legitimately NOT in the space at r=1 (the
            # dropped last radial function) -> exclude the outer p rows there; the
            # meaningful defect is the axis-side window truncation.
            r_stop = shape_list[0][0] - (args.p if dirichlet else 0)
            pce = float(jnp.max(perr[:r_stop]))
            print(f"[{bc}] P_const_err (axis-side window truncation) = {pce:.3e}", flush=True)

        # coarsest dense solve (exact; pseudo-inverse handles the free-BC
        # near-null bulk constant like the FD atom's modal threshold does)
        t0 = time.perf_counter()
        A_last, nb_last = A_list[-1], nb_list[-1]
        dense_inv = None
        if len(levels) > 1:
            A_last(jnp.zeros((nb_last,), dtype=jnp.float64))
            dense = _symmetrize(_assemble_dense_from_apply(A_last, nb_last, sequential=True))
            dense_inv = _psd_pseudoinverse(dense)
            jax.block_until_ready(dense_inv)
        t_dense = time.perf_counter() - t0
        print(f"[{bc}] coarsest dense probe n={nb_last}: {t_dense:.1f} s", flush=True)

        # baseline
        if args.baseline:
            def base_p(v, d=dirichlet):
                return apply_laplacian_preconditioner(seq, ops, v, 0, dirichlet=d, kind="tensor")
            it, dt, rel = timed_pcg(a_full, m_full, base_p, rhs, vs, args.tol, args.maxiter)
            print(f"[{bc}] baseline(FD tensor-hodge)  it={it:<5d} {dt:.3f}s "
                  f"({1e3 * dt / max(it, 1):.2f} ms/it) rel={rel:.1e}", flush=True)
            row(bc, "baseline", n_ext, it, dt, rel)

        # MG per smoother mode
        for mode in smoother_modes:
            t0 = time.perf_counter()
            smooth_list, lam0 = [], None
            for lvl in range(len(levels) if len(levels) == 1 else len(levels) - 1):
                sq, op = levels[lvl]
                S = build_smoother_atom(sq, op, dirichlet, shape_list[lvl],
                                        A_list[lvl], nb_list[lvl], mode)
                lam_max = estimate_lam_max(A_list[lvl], S, nb_list[lvl], free_bc,
                                           seed=args.seed + lvl)
                if lvl == 0:
                    lam0 = lam_max
                smooth_list.append(make_cheb_smoother(
                    A_list[lvl], S, args.smooth_steps,
                    lam_max / args.cheb_window, lam_max))
            t_sm = time.perf_counter() - t0

            vcycle = make_vcycle(A_list, shape_list, smooth_list, P_list, dense_inv)
            precond, env_t = build_envelope(seq, ops, dirichlet, vcycle, nb0, core0, args.schur)

            se = mr = ""
            if args.two_level_check:
                se, mr = spd_check(precond, n_ext, seed=args.seed)
                print(f"[{bc}/{mode}] SPD: sym_err={se:.2e} min_rayleigh={mr:.3e}", flush=True)
                assert se < 1e-10, f"envelope not symmetric ({se:.2e})"
                assert mr > 0, f"envelope not positive definite ({mr:.3e})"
                if mode == "jacobi" and nb0 <= 4000:
                    di_ref = _diagonal_from_matvec(A_list[0], nb0)
                    di_es = k0_bulk_stiffness_diagonal(seq, shape_list[0])
                    derr = float(jnp.max(jnp.abs(di_es - di_ref) / jnp.maximum(jnp.abs(di_ref), 1e-30)))
                    print(f"[{bc}] jacobi diag einsum vs probe: rel err {derr:.2e}", flush=True)
                    assert derr < 1e-8, f"einsum diagonal mismatch ({derr:.2e})"

            it, dt, rel = timed_pcg(a_full, m_full, precond, rhs, vs, args.tol, args.maxiter)
            print(f"[{bc}] MG({mode:6s}) lam_max={lam0:.3g}  it={it:<5d} {dt:.3f}s "
                  f"({1e3 * dt / max(it, 1):.2f} ms/it) rel={rel:.1e}  "
                  f"[smoother {t_sm:.1f}s, ass {env_t['ass_s']:.1f}s, W {env_t['w_s']:.1f}s]",
                  flush=True)
            method = f"mg-{mode}" if args.p_fix == "none" else f"mg-{mode}+{args.p_fix}"
            row(bc, method, n_ext, it, dt, rel, f"{lam0:.4g}", pce, se, mr,
                f"{t_sm:.2f}", f"{env_t['ass_s']:.2f}", f"{env_t['w_s']:.2f}", f"{t_dense:.2f}")

    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
