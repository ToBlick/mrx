"""Dense condition-number study for the k=0 Hodge-Laplacian preconditioner.

Verifies (all at small DBC sizes, exact dense linear algebra) three questions
that came up while probing the k=0 stiffness  K_0 = G_0^T M_1 G_0  (G_0 the
geometry-free incidence; M_1 the 1-form mass carrying ALL geometry):

1. The "B" idea -- precondition K_0 THROUGH the (good) 1-form mass
   preconditioner instead of modelling the metric in a scalar atom:
       B  = L_0^{-1} (G_0^T P_1 G_0) L_0^{-1},     L_0 = G_0^T G_0     (topological)
   and its M_id-weighted variant (reference-domain FEM Laplacian projection):
       B' = L'^{-1} (G_0^T M_id P_1 M_id G_0) L'^{-1},  L' = G_0^T M_id G_0
   where M_id is the geometry-FREE 1-form mass (metric_inv = I, jacobian = 1),
   so L' is separable -> FD-exact. P_1 ~ M_1^{-1} is the tensor mass precond.
   RESULT: B is bulk-global-leaky and NOT rescued by axis surgery; B' is
   2.6-6x better and flatter across geometry but still kappa~100, dominated by
   the production fd+surgery preconditioner. The B family is closed.

2. Does exact axis (core) surgery rescue B?  Block-LDU with an exact dense
   core A^{-1} + a bulk Schur-inverse T (T = S^{-1} gives kappa=1, the sanity
   check).  RESULT: no -- B's error is bulk-global (regular-decomposition
   curl-leakage), not axis-local, so surgery (which rescues fd) barely moves it.

3. fd vs fdax vs fdbund BULK atom, with the exact core surgery held fixed --
   isolates the metric-lumping choice:
     fd     : D = geomean(a_rr a_tt a_zz), unweighted atoms (production atom)
     fdax   : pull out J (D = J), average the BARE g^aa
     fdbund : average the BUNDLED <g^aa J>, D = 1
   RESULT: fdbund > fdax on every geometry (bundled g^tt J ~ 1/r is milder than
   bare g^tt ~ 1/r^2, so its cross-axis average is more faithful). fd ~ fdbund
   (fd better on toroid/helical, fdbund better on shaped cerfon); all within
   ~15% -> the bulk metric choice is second-order to the core surgery.

4. The off-diagonal g^{r theta} ladder (2026-07-24) -- every current bulk atom
   keeps only the diagonal metric blocks; shaping/helicity turn on g^{r theta},
   and the findings doc flags it as the open lever for the shaped/helical
   kappa growth. Dense bulk MODEL matrices (exact inverse, exact core surgery
   held fixed) isolate what the discarded blocks cost:
     exact bulk      T = inv(K0[bulk,bulk])          -- ceiling for ANY bulk atom
     diag pointwise  all three g^aa J, pointwise      -- cost of dropping off-diag
     diag + g^{rt}   pointwise                        -- gain of restoring rt only
     fdbund kronsum  dense kron-sum of the fdbund 1D  -- cross-check vs the atom
     fdbund + cross  kron-sum + zeta-averaged g^{rt}  -- the PRACTICAL candidate
                     (K2D_{rt} (x) M_z: one dense 2D (r,t) solve per zeta mode)
   Sanity: the all-9-blocks pointwise assembly must reproduce K0[bulk,bulk].

Framework cross-check: kappa(fd + surgery) reproduces the production
apply_laplacian_preconditioner(k=0, tensor) column to a few percent.

Run:  NS="6 12 6" python scripts/debug/verify_hodge_massprecond_k0.py
"""
from __future__ import annotations

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
sys.path.insert(0, HERE)

import benchmark_graddiv_k1_preconditioner as bench  # noqa: E402
import laplacian_mg_k0 as mg  # noqa: E402  (build_fd_atom: fd/fdax/fdbund)
from mrx.preconditioners import _bulk_tensor_shape  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_incidence_matrix,
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    apply_stiffness,
    apply_laplacian_preconditioner,
    build_matrixfree_mass_apply,
    _mass_extraction,
    _dense_incidence_1d,
    _assemble_unweighted_1d_mass,
    _assemble_weighted_1d_stiffness,
    _restrict_radial_window,
    _reshape_quadrature_scalar_field,
    _reshape_quadrature_matrix_field,
)

D = True  # DBC: BETTI=(1,1,0,0) -> K_0 SPD, no constant nullspace


def make_cfg(geometry, ns, **kw):
    base = dict(
        ns=ns, p=3, epsilon=1.0 / 3.0, kappa=1.0, alpha=0.0, r0=1.0, nfp=3,
        cg_tol=1e-10, cg_maxiter=2000, r_scale=0.5, polar_ring1=None,
        geometry=geometry,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def build(geometry, ns, **kw):
    cfg = make_cfg(geometry, ns, **kw)
    seq = bench.build_sequence(cfg)
    ops = bench.assemble_operators(seq, klevel=1)  # mutates seq, returns ops bundle
    return seq, ops


def G0(seq, ops, v):
    return apply_incidence_matrix(seq, ops, v, 0, dirichlet_in=D, dirichlet_out=D,
                                  transpose=False)                       # V0 -> V1


def G0T(seq, ops, w):
    return apply_incidence_matrix(seq, ops, w, 0, dirichlet_in=D, dirichlet_out=D,
                                  transpose=True)                        # V1* -> V0*


def dense(apply_fn, n):
    """Materialise a dense (m x n) matrix from a matrix-free apply on R^n."""
    apply_fn(jnp.zeros((n,), jnp.float64))  # warm host-side static state
    cols = []
    for i in range(n):
        e = jnp.zeros((n,), jnp.float64).at[i].set(1.0)
        cols.append(np.asarray(jax.block_until_ready(apply_fn(e))))
    return np.stack(cols, axis=1)


def cond_precond(P, K):
    """kappa of the SPD-preconditioned operator P@K (real positive spectrum)."""
    M = P @ K
    if not np.all(np.isfinite(M)):
        return float("inf"), float("nan")  # degenerate atom at tiny ns
    ev = np.linalg.eigvals(M)
    im = np.max(np.abs(ev.imag)) / (np.max(np.abs(ev.real)) + 1e-300)
    ev = np.sort(ev.real)
    ev = ev[ev > ev[-1] * 1e-12]  # drop numerical zeros
    return ev[-1] / ev[0], im


def assemble_bulk_model(seq, bulk_shape, terms, ring0=0, zeta_avg=()):
    """Dense bulk MODEL stiffness on the radial-window tensor space.

    terms: iterable of metric-block indices (a, b) to include, each with its
    pointwise quadrature weight g^{ab} J. zeta_avg: subset of terms whose
    weight field is replaced by its quad-weighted zeta-average (the weight
    becomes w(r, theta) -> the term separates as K2D_{r theta} (x) M_zeta,
    i.e. one dense 2D (r, theta) block per zeta mode -- production-plausible).

    The polar-extraction bulk rows are unit tensor-product rows, so restricting
    the full tensor assembly to the radial window IS the bulk block (sanity-
    checked in run(): all 9 pointwise blocks reproduce K0[bulk, bulk])."""
    nr_bulk, nt, nz = (int(s) for s in bulk_shape)
    rw = 2 + ring0
    zeta_avg = set(zeta_avg)
    # quad-field layout is (ny, nx, nz) = (theta, r, zeta); transpose to (r, theta, zeta)
    jac = jnp.transpose(_reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j), (1, 0, 2))
    minv = jnp.transpose(_reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    wx, wy, wz = seq.quad.w_x, seq.quad.w_y, seq.quad.w_z
    W3 = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    types = seq.basis_0.types
    n0s = (seq.basis_0.nr, seq.basis_0.nt, seq.basis_0.nz)
    Phi = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    dB = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    dPhi = tuple(_dense_incidence_1d(n0s[ax], types[ax]).T @ dB[ax] for ax in range(3))

    K = jnp.zeros((n0s[0], n0s[1], n0s[2], n0s[0], n0s[1], n0s[2]), dtype=jnp.float64)
    for (a, b) in terms:
        field = jac * minv[..., a, b]
        if (a, b) in zeta_avg:
            fbar = jnp.einsum('qrs,s->qr', field, wz) / jnp.sum(wz)
            field = fbar[:, :, None] * jnp.ones_like(field)
        Wab = field * W3
        F = [(dPhi[ax] if ax == a else Phi[ax], dPhi[ax] if ax == b else Phi[ax])
             for ax in range(3)]
        t = jnp.einsum('iq,lq,qrs->ilrs', F[0][0], F[0][1], Wab)
        t = jnp.einsum('jr,mr,ilrs->ijlms', F[1][0], F[1][1], t)
        t = jnp.einsum('ks,ns,ijlms->ijklmn', F[2][0], F[2][1], t)
        K = K + t
    Kw = K[rw:rw + nr_bulk, :, :, rw:rw + nr_bulk, :, :]
    nb = nr_bulk * nt * nz
    return np.asarray(Kw.reshape(nb, nb))


def fdbund_kron_model(seq, bulk_shape, ring0=0):
    """Dense kron-sum model the fdbund atom inverts: K_r(x)M_t(x)M_z + cyc.,
    with the bundled per-axis profiles <g^aa J> as the 1D stiffness weights
    (D = 1). Its exact inverse should reproduce kappa(fdbund+surg)."""
    nr_bulk, nt, nz = (int(s) for s in bulk_shape)
    rw = 2 + ring0
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])
    pr, pt, pz = mg.fdax_axis_profiles(seq, ring0, bundle_j=True)
    kw_x, kw_y, kw_z = seq.quad.w_x * pr, seq.quad.w_y * pt, seq.quad.w_z * pz
    M_r = np.asarray(_restrict_radial_window(
        _assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), rw, nr_bulk))
    M_t = np.asarray(_assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y))
    M_z = np.asarray(_assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z))
    K_r = np.asarray(_restrict_radial_window(
        _assemble_weighted_1d_stiffness(seq.basis_r_jk, seq.d_basis_r_jk, kw_x, g_r), rw, nr_bulk))
    K_t = np.asarray(_assemble_weighted_1d_stiffness(seq.basis_t_jk, seq.d_basis_t_jk, kw_y, g_t))
    K_z = np.asarray(_assemble_weighted_1d_stiffness(seq.basis_z_jk, seq.d_basis_z_jk, kw_z, g_z))
    return (np.kron(np.kron(K_r, M_t), M_z)
            + np.kron(np.kron(M_r, K_t), M_z)
            + np.kron(np.kron(M_r, M_t), K_z))


def psd_inv(mat):
    """Positive-part pseudoinverse (mixed averaged-diag + indefinite cross
    models can lose SPD where the averaging under-weights; exact inverse for
    SPD input up to the 1e-8 relative cutoff)."""
    return np.asarray(mg._psd_pseudoinverse(jnp.asarray(mat)))


def run(geometry, ns, **kw):
    t0 = time.time()
    seq, ops = build(geometry, ns, **kw)
    n0 = int(seq.n0_dbc)
    n1 = int(seq.n1_dbc)

    # dense operators on the n0-dim DBC 0-form space
    K0 = dense(lambda v: G0T(seq, ops, apply_mass_matrix(seq, ops, G0(seq, ops, v), 1, dirichlet=D)), n0)
    L0 = dense(lambda v: G0T(seq, ops, G0(seq, ops, v)), n0)
    Mid = dense(lambda v: G0T(seq, ops, apply_mass_matrix_preconditioner(
        seq, ops, G0(seq, ops, v), 1, dirichlet=D, kind="tensor")), n0)

    # geometry-FREE 1-form mass M_id (metric_inv = I, jacobian = 1): retains the
    # spline overlap (reference L^2 Gram) but no geometry. Lp = G0^T M_id G0 is
    # the reference-domain FEM Laplacian -- a better projection metric for B than
    # the purely-topological L0 = G0^T G0 (Euclidean coefficient inner product),
    # and (being constant-coefficient) separable -> FD-invertible exactly.
    nq = seq.geometry.jacobian_j.shape[0]
    id_geo = SimpleNamespace(
        metric_inv_jkl=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (nq, 3, 3)),
        jacobian_j=jnp.ones((nq,), jnp.float64))
    Mid_core = build_matrixfree_mass_apply(seq, 1, id_geo)
    e1, e1_T = _mass_extraction(ops, 1, D)
    def Mid_ext(w):
        return e1 @ Mid_core(e1_T @ w)
    Lp = dense(lambda v: G0T(seq, ops, Mid_ext(G0(seq, ops, v))), n0)
    # consistent M_id-weighted pseudo-inverse: middle = G^T M_id P1 M_id G
    # (NOT G^T P1 G). Exact when M_1 = M_id (identity geometry). P1 ~ M_1^{-1}.
    _P1 = lambda x: apply_mass_matrix_preconditioner(seq, ops, x, 1, dirichlet=D, kind="tensor")
    Midp = dense(lambda v: G0T(seq, ops, Mid_ext(_P1(Mid_ext(G0(seq, ops, v))))), n0)
    Kstiff = dense(lambda v: apply_stiffness(seq, ops, v, 0, dirichlet=D), n0)
    P0hodge = dense(lambda v: apply_laplacian_preconditioner(
        seq, ops, v, 0, dirichlet=D, kind="tensor"), n0)

    K0 = 0.5 * (K0 + K0.T)
    L0 = 0.5 * (L0 + L0.T)
    Lp = 0.5 * (Lp + Lp.T)
    L0inv = np.linalg.inv(L0)
    B = L0inv @ Mid @ L0inv
    Lpinv = np.linalg.inv(Lp)
    Bp = Lpinv @ Midp @ Lpinv         # consistent M_id-weighted pseudo-inverse

    # ---- exact axis (core) surgery: block-LDU with exact dense core A^{-1} + a
    # bulk Schur-inverse T. core = first (n0-nb) DOFs (matches _core_size=3*nz
    # and the fd/fdax/fdbund atom bulk). T = S^{-1} => preconditioner = K0^{-1}
    # (kappa=1 sanity).  We plug B[bulk,bulk] ~ (K0^{-1})[bulk,bulk] = S^{-1}.
    nrb, ntb, nzb = _bulk_tensor_shape(seq, D)
    bulk_shape = (nrb, ntb, nzb)
    nb = nrb * ntb * nzb
    c = n0 - nb
    assert c == 3 * int(seq.basis_0.nz), (c, 3 * int(seq.basis_0.nz))
    A = K0[:c, :c]
    Bm = K0[:c, c:]
    Ai = np.linalg.inv(A)
    AiB = Ai @ Bm
    S = K0[c:, c:] - Bm.T @ AiB          # exact bulk Schur complement

    def surgery_P(Tb):
        P = np.empty_like(K0)
        P[:c, :c] = Ai + AiB @ Tb @ AiB.T
        P[:c, c:] = -AiB @ Tb
        P[c:, :c] = P[:c, c:].T
        P[c:, c:] = Tb
        return P

    # ---- fd (production geomean) vs fdax ("pull out J") vs fdbund (avg <g J>),
    #      bulk atom only, EXACT core surgery fixed -> isolates the metric choice.
    atom_k = {}
    for mode in ("fd", "fdax", "fdbund"):
        T_atom = dense(mg.build_fd_atom(seq, bulk_shape, mode, 0), nb)
        atom_k[mode], _ = cond_precond(surgery_P(T_atom), K0)

    # ---- off-diagonal g^{r theta} ladder: bulk MODEL matrices, exact inverse,
    #      exact core surgery fixed. Isolates what the discarded blocks cost.
    Kb = K0[c:, c:]
    diag_terms = [(0, 0), (1, 1), (2, 2)]
    rt_terms = [(0, 1), (1, 0)]
    all_terms = [(a, b) for a in range(3) for b in range(3)]
    Kmod_all = assemble_bulk_model(seq, bulk_shape, all_terms)
    rel_bulk_model = np.linalg.norm(Kmod_all - Kb) / np.linalg.norm(Kb)  # sanity
    Kmod_diag = assemble_bulk_model(seq, bulk_shape, diag_terms)
    Kmod_diagrt = assemble_bulk_model(seq, bulk_shape, diag_terms + rt_terms)
    Kkron = fdbund_kron_model(seq, bulk_shape)
    Kcross_zavg = assemble_bulk_model(seq, bulk_shape, rt_terms, zeta_avg=rt_terms)
    # practical rungs: average ONLY over zeta, keep (r, theta) pointwise ->
    # every term separates as K2D_{r theta} (x) (M_z or K_z): one dense 2D
    # (r, theta) solve per zeta mode. Same production cost class as the cross
    # candidate, but attacks the DIAGONAL lumping error (the dominant one).
    Kmod_diag_zavg = assemble_bulk_model(seq, bulk_shape, diag_terms, zeta_avg=diag_terms)
    Kmod_diagrt_zavg = assemble_bulk_model(seq, bulk_shape, diag_terms + rt_terms,
                                           zeta_avg=diag_terms + rt_terms)
    ladder_k = {}
    ladder_k['bulk_exact'], _ = cond_precond(surgery_P(psd_inv(Kb)), K0)
    ladder_k['diag_point'], _ = cond_precond(surgery_P(psd_inv(Kmod_diag)), K0)
    ladder_k['diagrt_point'], _ = cond_precond(surgery_P(psd_inv(Kmod_diagrt)), K0)
    ladder_k['diag_zavg'], _ = cond_precond(surgery_P(psd_inv(Kmod_diag_zavg)), K0)
    ladder_k['diagrt_zavg'], _ = cond_precond(surgery_P(psd_inv(Kmod_diagrt_zavg)), K0)
    ladder_k['fdbund_kron'], _ = cond_precond(surgery_P(psd_inv(Kkron)), K0)
    ladder_k['fdbund_cross'], _ = cond_precond(surgery_P(psd_inv(Kkron + Kcross_zavg)), K0)

    # diagnostics
    rel_KvsStiff = np.linalg.norm(K0 - Kstiff) / np.linalg.norm(K0)
    k_raw = np.linalg.cond(K0)
    k_L0, _ = cond_precond(L0inv, K0)
    k_B, imB = cond_precond(B, K0)
    k_H, _ = cond_precond(P0hodge, K0)
    k_surg_exact, _ = cond_precond(surgery_P(np.linalg.inv(S)), K0)  # sanity ~1
    k_B_surg, _ = cond_precond(surgery_P(B[c:, c:]), K0)             # B + surgery
    k_Bp, _ = cond_precond(Bp, K0)                                   # B' (M_id proj)
    k_Bp_surg, _ = cond_precond(surgery_P(Bp[c:, c:]), K0)           # B' + surgery

    dt = time.time() - t0
    print(f"\n=== {geometry:16s} ns={ns} kappa={kw.get('kappa',1.0)} "
          f"alpha={kw.get('alpha',0.0)} nfp={kw.get('nfp',3)} ===")
    print(f"  n0(dbc)={n0}  n1(dbc)={n1}   |K0 - apply_stiffness|/|K0| = {rel_KvsStiff:.2e}")
    print(f"  kappa(K0) raw                                : {k_raw:12.2f}")
    print(f"  kappa(B   K0)   L0=G^T G      (topological)  : {k_B:12.2f}")
    print(f"  kappa(B+surg)   L0=G^T G     + exact axis core: {k_B_surg:12.2f}   core={c}")
    print(f"  kappa(B'  K0)   L'=G^T M_id G (spline,no geom): {k_Bp:12.2f}")
    print(f"  kappa(B'+surg)  L'=G^T M_id G + exact axis core: {k_Bp_surg:12.2f}")
    print(f"  kappa(Hodge K0) production fd-bulk + surgery  : {k_H:12.2f}")
    print(f"  --- bulk atom + EXACT core surgery (isolates the metric choice) ---")
    print(f"  kappa(fd+surg)     geomean D, unweighted atoms: {atom_k['fd']:12.2f}   (cross-check vs Hodge)")
    print(f"  kappa(fdax+surg)   pull out J, avg bare g^aa  : {atom_k['fdax']:12.2f}")
    print(f"  kappa(fdbund+surg) avg <g^aa J>, D=1          : {atom_k['fdbund']:12.2f}")
    print(f"  --- off-diagonal g^{{r,theta}} ladder (bulk MODEL + exact inverse + surgery) ---")
    print(f"  kappa(exact bulk block)   ceiling for any atom: {ladder_k['bulk_exact']:12.2f}")
    print(f"  kappa(pointwise diag)     drop ALL off-diag   : {ladder_k['diag_point']:12.2f}")
    print(f"  kappa(pointwise diag+rt)  restore g^{{rt}} only : {ladder_k['diagrt_point']:12.2f}")
    print(f"  kappa(zavg diag)     2D(r,t)-pointwise, per-zeta-mode: {ladder_k['diag_zavg']:12.2f}")
    print(f"  kappa(zavg diag+rt)  2D(r,t)-pointwise, per-zeta-mode: {ladder_k['diagrt_zavg']:12.2f}")
    print(f"  kappa(fdbund kron-sum)    cross-check vs atom : {ladder_k['fdbund_kron']:12.2f}")
    print(f"  kappa(fdbund + zavg g^{{rt}} cross)  PRACTICAL  : {ladder_k['fdbund_cross']:12.2f}")
    print(f"  |model(all 9) - K0[bulk]|/|K0[bulk]| (sanity) : {rel_bulk_model:12.2e}")
    print(f"  kappa(exact surgery, sanity ~1)              : {k_surg_exact:12.2e}   [{dt:.1f}s]")
    return dict(geometry=geometry, n0=n0, k_raw=k_raw, k_B=k_B, k_B_surg=k_B_surg,
                k_Bp=k_Bp, k_Bp_surg=k_Bp_surg, k_H=k_H,
                k_fd=atom_k['fd'], k_fdax=atom_k['fdax'], k_fdbund=atom_k['fdbund'],
                k_bulk_exact=ladder_k['bulk_exact'], k_diag_point=ladder_k['diag_point'],
                k_diagrt_point=ladder_k['diagrt_point'], k_diag_zavg=ladder_k['diag_zavg'],
                k_diagrt_zavg=ladder_k['diagrt_zavg'], k_fdbund_kron=ladder_k['fdbund_kron'],
                k_fdbund_cross=ladder_k['fdbund_cross'], rel_bulk_model=rel_bulk_model,
                rel_KvsStiff=rel_KvsStiff)


if __name__ == "__main__":
    ns = tuple(int(x) for x in os.environ.get("NS", "6 12 6").split())
    rows = []
    rows.append(run("toroid", ns, kappa=1.0))                       # axisymmetric circular (easy)
    rows.append(run("cerfon", ns, kappa=1.7, alpha=0.4))            # shaped, non-diag (r,theta)
    rows.append(run("rotating_ellipse", ns, kappa=1.5, nfp=2))      # helical (hard)

    print("\n\n===== SUMMARY: bulk atom + EXACT core surgery (kappa; lower=better) =====")
    print(f"{'geometry':16s} {'fd(geomean)':>12s} {'fdax(pullJ)':>12s} {'fdbund(gJ)':>12s} | {'Hodge':>7s}")
    for r in rows:
        print(f"{r['geometry']:16s} {r['k_fd']:12.2f} {r['k_fdax']:12.2f} "
              f"{r['k_fdbund']:12.2f} | {r['k_H']:7.2f}")
    print("(fd+surg should ~= Hodge: framework cross-check.)")
    print("fdax = pull out J (D=J, avg bare g^aa);  fdbund = avg <g^aa J>, D=1.")

    print("\n===== SUMMARY: off-diagonal g^{r,theta} ladder (kappa; lower=better) =====")
    print(f"{'geometry':16s} {'exact-bulk':>10s} {'diag-pt':>8s} {'diag+rt':>8s} "
          f"{'zavg-diag':>9s} {'zavg-d+rt':>9s} {'bund-kron':>9s} {'bund+cross':>10s} | {'fdbund-atom':>11s}")
    for r in rows:
        print(f"{r['geometry']:16s} {r['k_bulk_exact']:10.2f} {r['k_diag_point']:8.2f} "
              f"{r['k_diagrt_point']:8.2f} {r['k_diag_zavg']:9.2f} {r['k_diagrt_zavg']:9.2f} "
              f"{r['k_fdbund_kron']:9.2f} "
              f"{r['k_fdbund_cross']:10.2f} | {r['k_fdbund']:11.2f}")
    print("(diag-pt vs exact-bulk = cost of dropping ALL off-diag; diag+rt = g^{rt}"
          " share;\n zavg-* = (r,t)-pointwise zeta-averaged = one dense 2D solve per"
          " zeta mode -- the practical candidates.)")
