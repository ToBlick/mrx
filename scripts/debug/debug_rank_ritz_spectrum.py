"""Bottom Ritz spectrum of the preconditioned operator smoother . L_0 (k=0
tensor-Hodge surgery preconditioner, production FD bulk inverse), const-deflated,
for rank 1/2/3 -- AND a full characterization of the single small eigenvector
that sets kappa.

The dense-spectrum table (see docs/hiptmair_xu_preconditioner.md, "Tensor rank")
shows that even at rank-1 (the production atom) the free-BC k=0 composite has ONE
small eigenvalue sitting below a gap (rot-ellipse: kappa=6.19, lmin=0.271, 1.95x
gap below an otherwise tight [0.529..1.68] cluster). The rank>1 blow-up rotates a
*different* near-null direction into the basement (23-124x gap). This probe pulls
out the lmin EIGENVECTOR at each rank and reads off *what it is*:

  - where its energy lives: polar-axis (core/surgery) DOFs vs the tensor bulk;
  - its DOF-coefficient structure on the bulk (nr,nt,nz) grid -- radial /
    poloidal / toroidal energy profiles, so you can read off "near the axis"
    (energy at low i_r) vs "near constant / low frequency" (energy at low modes);
  - its FD-modal content (the basis the bulk inverse actually diagonalizes in),
    the dominant mode (i,j,k) and that mode's preconditioner denominator D;
  - whether the rank-1 small mode is the SAME direction as the rank>1 basement
    outlier (cross-rank M0-cosine) -- i.e. "same Schur-interaction mode, milder?"

Full dense eig (B is A-self-adjoint, A=L_0, so eigenvalues are real). Prints the
bottom 15 + top 3, the largest relative gap in the bottom quarter, then the
characterization. Saves per-rank eigenvector + profiles to .npz and a PNG.

Usage (activate the venv first: `source .venv/bin/activate`):
    python scripts/debug/debug_rank_ritz_spectrum.py --geometry toroid --ns 6,12,4 --p 3 --ranks 1,2,3
    python scripts/debug/debug_rank_ritz_spectrum.py --geometry rotating_ellipse --nfp 3 --ns 6,12,4 --p 3 --ranks 1,2,3
"""
import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for _p in (ROOT, SCRIPTS, SCRIPTS / "benchmark", SCRIPTS / "debug"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

jax.config.update("jax_enable_x64", True)

from mrx.operators import (  # noqa: E402
    apply_stiffness, apply_mass_matrix, _nullspace_vectors, _core_size,
    _apply_k0_tensor_hodge_bulk_inverse,
    _apply_k0_tensor_hodge_bulk_forward,
    _apply_k0_tensor_hodge_core_block,
    _apply_k0_tensor_hodge_surgery_to_bulk_coupling,
    _apply_k0_tensor_hodge_bulk_to_surgery_coupling,
    _assemble_k0_tensor_hodge_preconditioner,
    _assemble_dense_from_apply, _symmetrize, _symmetric_pseudoinverse,
)
import benchmark_graddiv_k1_preconditioner as dg  # noqa: E402
from benchmark_graddiv_k1_preconditioner import build_sequence, assemble_operators  # noqa: E402

DBC = False
EXACT_BULK = False
EXACT_MODEL_INV = False
ZETA_FIRST = False
RADIAL_DENSE = False
RADIAL_DENSE_PROD = False   # validate the PRODUCTION radial-dense bulk inverse (mrx.operators)
RADIAL_DENSE_PROD_RANK = 2
RT_COUPLED = False          # rank-2 θ: banded-θ (r,θ)-coupled inverse per ζ-mode
RT_BAND = 1                 # θ Fourier band (1 = cosθ tridiagonal = rank-2 θ)


def make_surgery_smoother(seq, ops, core_size, bulk_inv_fn):
    """Full k=0 Hodge preconditioner (Schur block solve) with EXACT core block +
    couplings and the supplied bulk inverse. Mirrors
    _apply_k0_tensor_hodge_preconditioner. (Inlined from the former zfirst probe.)"""
    def core_block(rc):
        return _apply_k0_tensor_hodge_core_block(seq, ops, core_size, rc, dirichlet=DBC)

    def surgery_to_bulk(rc):     # L_bc @ rc   (bulk from core)
        return _apply_k0_tensor_hodge_surgery_to_bulk_coupling(seq, ops, core_size, rc, dirichlet=DBC)

    def bulk_to_surgery(rb):     # L_cb @ rb   (core from bulk)
        return _apply_k0_tensor_hodge_bulk_to_surgery_coupling(seq, ops, core_size, rb, dirichlet=DBC)

    ass = _symmetrize(_assemble_dense_from_apply(core_block, core_size, sequential=True))
    schur = _symmetrize(_assemble_dense_from_apply(
        lambda rc: ass @ rc - bulk_to_surgery(bulk_inv_fn(surgery_to_bulk(rc))),
        core_size, sequential=True))
    schur_inv = _symmetric_pseudoinverse(schur)

    def smoother(rhs):
        rc, rb = rhs[:core_size], rhs[core_size:]
        y = bulk_inv_fn(rb)
        z = schur_inv @ (rc - bulk_to_surgery(y))
        x_b = y - bulk_inv_fn(surgery_to_bulk(z))
        return jnp.concatenate([z, x_b])
    return smoother


def dense_spectrum(a_matvec, precond, n, project, label=""):
    """Materialize B = (project . precond . a_matvec) on the n unit vectors and
    return its (real) eigenvalues AND eigenvectors, sorted ascending. B is
    A-self-adjoint so its eigenvalues are real; we drop the deflated constant's
    zero eigenvalue. Eigenvectors are columns (np.linalg.eig, unit Euclidean)."""
    cols = []
    t0 = time.perf_counter()
    chunk = max(1, n // 20)  # ~20 progress ticks
    for i in range(n):
        e = jnp.zeros((n,), dtype=jnp.float64).at[i].set(1.0)
        col = project(precond(a_matvec(e)))
        col.block_until_ready() if hasattr(col, "block_until_ready") else None
        cols.append(col)
        if (i + 1) % chunk == 0 or i + 1 == n:
            dt = time.perf_counter() - t0
            rate = (i + 1) / max(dt, 1e-9)
            eta = (n - (i + 1)) / max(rate, 1e-9)
            print(f"    [{label}] materializing B: {i+1:4d}/{n} cols "
                  f"({dt:5.1f}s, {rate:4.1f} col/s, eta {eta:5.1f}s)", flush=True)
    print(f"    [{label}] dense eig on {n}x{n} ...", flush=True)
    B = np.asarray(jnp.stack(cols, axis=1))
    ev, evec = np.linalg.eig(B)
    ev = np.real(ev)
    evec = np.real(evec)
    # the const-deflation leaves an exact zero eigenvalue; drop the near-zeros
    # from deflation only (|ev| below 1e-10 * max), keep the genuine small modes.
    keep = np.abs(ev) > 1e-10 * np.max(np.abs(ev))
    ev, evec = ev[keep], evec[:, keep]
    order = np.argsort(ev)
    return ev[order], evec[:, order]


def _build_exact_bulk_inverse(seq, ops, core_size, n):
    """Dense inverse of the TRUE bulk block K_BB of L_0: probe L_0 on each bulk
    unit vector (embedded in full V0 with zero core) and keep the bulk rows. This
    is the genuine -- non-separable, Neumann-at-outer -- bulk operator. Swapping it
    in for the separable FD model is the decisive test: if the composite outlier
    vanishes (kappa -> ~1), the outlier IS the FD separability error, not the
    surgery/structure or any ill-posedness of the free (Neumann) bilinear form."""
    bulk = n - core_size
    cols = []
    for j in range(bulk):
        e = jnp.zeros((n,), dtype=jnp.float64).at[core_size + j].set(1.0)
        col = apply_stiffness(seq, ops, e, 0, dirichlet=DBC)
        cols.append(np.asarray(col)[core_size:])
    KBB = _symmetrize(jnp.asarray(np.stack(cols, axis=1)))
    KBB_inv = _symmetric_pseudoinverse(KBB)
    return lambda rb, _Ki=KBB_inv: _Ki @ rb


def bulk_pencil_diagnostic(seq, ops, core_size, n, v_full, label):
    """Test the user's hypothesis: is the mode a SMALL-eigenvalue mode of
    L_bulk = K_BB (the bulk block of L_0), not the full L_0? Build the true bulk
    blocks K_BB (of L_0) and M_BB (of M_0), solve the generalized symmetric-definite
    pencil K_BB x = lam M_BB x, and report the smallest eigenvalues plus how well
    the mode's bulk part aligns (M_BB-orthonormal expansion) with the lowest bulk
    eigenvectors. This is the RIGHT energy for the question (bulk block on the bulk
    part), vs the full-operator rho_L0 reported elsewhere."""
    from scipy.linalg import eigh
    bulk = n - core_size
    Kc, Mc = [], []
    for j in range(bulk):
        e = jnp.zeros((n,), dtype=jnp.float64).at[core_size + j].set(1.0)
        Kc.append(np.asarray(apply_stiffness(seq, ops, e, 0, dirichlet=DBC))[core_size:])
        Mc.append(np.asarray(apply_mass_matrix(seq, ops, e, 0, dirichlet=DBC))[core_size:])
    KBB = np.stack(Kc, 1); KBB = 0.5 * (KBB + KBB.T)
    MBB = np.stack(Mc, 1); MBB = 0.5 * (MBB + MBB.T)
    w, V = eigh(KBB, MBB)  # ascending gen-eigs; V columns are M_BB-orthonormal
    vb = np.asarray(v_full, dtype=float)[core_size:]
    nrm = np.sqrt(vb @ MBB @ vb)
    vb_n = vb / max(nrm, 1e-300)
    rho_bulk = float(vb @ KBB @ vb) / float(vb @ MBB @ vb)
    coeffs = V.T @ (MBB @ vb_n)  # expansion of the mode in the bulk eigenbasis
    dom = int(np.argmax(np.abs(coeffs)))
    print(f"\n  [bulk pencil L_bulk=K_BB on the bulk part, {label}]", flush=True)
    print(f"     smallest gen-eigs (K_BB,M_BB): " + " ".join(f"{x:.3e}" for x in w[:6]), flush=True)
    print(f"     bulk cond(K_BB,M_BB) = {w[-1]/max(w[0],1e-300):.3e}  (lam_max={w[-1]:.3e})", flush=True)
    print(f"     mode's bulk Rayleigh rho_bulk = vbᵀK_BB vb / vbᵀM_BB vb = {rho_bulk:.4e}  "
          f"(vs bulk lam_min={w[0]:.3e}, median={np.median(w):.3e})", flush=True)
    print(f"     |M_BB-overlap| of mode with lowest bulk eigvecs: "
          + " ".join(f"#{i}:{abs(coeffs[i]):.3f}" for i in range(min(6, len(coeffs)))), flush=True)
    print(f"     -> mode dominated by bulk eigvec #{dom} (gen-eig {w[dom]:.3e}); "
          f"energy in lowest-3 bulk modes = {float(np.sum(coeffs[:3]**2)):.3f} "
          f"({'IS a low bulk mode' if dom <= 2 else 'NOT a low bulk mode'})", flush=True)


def _bulk_inversion_residual(factors):
    """How exact is the FD bulk inverse on its OWN model operator K~_bb? Apply the
    separable model forward (K~_bb) then the FD inverse and measure the residual.
    For a clean Kronecker sum (rank 1) FD is exact (~1e-15); rank>1 Lynch diagonal
    truncation makes it inexact. This separates the INVERSION error from the
    K_bb~=K~_bb APPROXIMATION error."""
    nr, nt, nz = factors.bulk_shape
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(nr * nt * nz))
    fx = _apply_k0_tensor_hodge_bulk_forward(factors, x)
    ifx = _apply_k0_tensor_hodge_bulk_inverse(factors, fx)
    return float(jnp.linalg.norm(ifx - x) / jnp.linalg.norm(x))


def _build_model_exact_inverse(factors):
    """Exact dense inverse of the SEPARABLE MODEL operator K~_bb (probe the model
    forward apply, invert densely). Swapping this in for the FD apply isolates the
    K_bb~=K~_bb APPROXIMATION error (bulk solve = the model, exactly inverted) from
    the FD/Lynch INVERSION error (the only thing that differs from production)."""
    nr, nt, nz = factors.bulk_shape
    bulk = nr * nt * nz
    cols = []
    for j in range(bulk):
        e = jnp.zeros((bulk,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(_apply_k0_tensor_hodge_bulk_forward(factors, e)))
    Ktilde = _symmetrize(jnp.asarray(np.stack(cols, axis=1)))
    Kt_inv = _symmetric_pseudoinverse(Ktilde)
    return lambda rb, _Ki=Kt_inv: _Ki @ rb


def _build_zeta_first_inverse(seq, ops, core_size, n, bulk_shape):
    """ζ-first / (r,θ)-dense bulk inverse (the user's "constrain ζ to rank 1, invert
    (r,θ) better" model). DFT the periodic ζ axis -- which block-diagonalises any
    ζ-circulant operator -- then invert each dense (r,θ) block (nr*nt) EXACTLY,
    dropping only the ζ-off-diagonal (nfp) coupling. It captures the in-plane metric
    non-separability exactly. On the axisymmetric toroid the bulk is exactly
    ζ-block-diagonal, so this is the exact bulk inverse (kappa->1); on rotating_ellipse
    it keeps only the ζ-diagonal, so any residual outlier is the dropped nfp ζ-coupling."""
    nr, nt, nz = bulk_shape
    rt, bulk = nr * nt, nr * nt * nz
    cols = []
    for j in range(bulk):
        e = jnp.zeros((n,), dtype=jnp.float64).at[core_size + j].set(1.0)
        cols.append(np.asarray(apply_stiffness(seq, ops, e, 0, dirichlet=DBC))[core_size:])
    KBB = np.stack(cols, axis=1); KBB = 0.5 * (KBB + KBB.T)
    K6 = KBB.reshape(nr, nt, nz, nr, nt, nz)
    W = np.exp(-2j * np.pi * np.outer(np.arange(nz), np.arange(nz)) / nz) / np.sqrt(nz)
    # K_hat[ir,it,a,jr,jt,b] = sum_{iz,jz} conj(W[iz,a]) K6 W[jz,b]  (ζ -> Fourier)
    Khat = np.einsum('ia,rtiRTj,jb->rtaRTb', np.conj(W), K6, W)
    Binv = [np.linalg.inv(Khat[:, :, a, :, :, a].reshape(rt, rt)) for a in range(nz)]

    def apply(rb, _Binv=Binv, _W=W):
        r = np.asarray(rb).reshape(nr, nt, nz)
        rhat = np.einsum('ia,rti->rta', np.conj(_W), r)
        xhat = np.empty_like(rhat)
        for a in range(nz):
            xhat[:, :, a] = (_Binv[a] @ rhat[:, :, a].reshape(rt)).reshape(nr, nt)
        x = np.einsum('ia,rta->rti', _W, xhat)
        return jnp.asarray(np.real(x).reshape(-1))
    return apply


def _build_radial_dense_inverse(seq, ops, core_size, n, factors):
    """θ,ζ rank-1 (diagonalize via the FD angular eigenbases V_t, V_z), r rank-2
    (exact dense n_r×n_r block per (θ,ζ) mode). Drops the θ/ζ OFF-diagonal coupling
    -- including the cosθ m±1 term -- but inverts the radial coupling exactly. This
    is the MEASUREMENT: if the composite outlier survives, the cosθ angular coupling
    matters (we need the θ band / block-Thomas); if it dies, rank-1-θ suffices. The
    apply is the GPU form (einsum transforms + batched dense radial solve); built
    here host-side from the true bulk block for the diagnostic."""
    nr, nt, nz = factors.bulk_shape
    bulk = nr * nt * nz
    cols = []
    for j in range(bulk):
        e = jnp.zeros((n,), dtype=jnp.float64).at[core_size + j].set(1.0)
        cols.append(np.asarray(apply_stiffness(seq, ops, e, 0, dirichlet=DBC))[core_size:])
    KBB = np.stack(cols, axis=1); KBB = 0.5 * (KBB + KBB.T)
    K6 = KBB.reshape(nr, nt, nz, nr, nt, nz)
    _, Vt, Vz = _modal_axis_bases(factors)          # M-orthonormal FD angular bases
    Vt, Vz = np.asarray(Vt), np.asarray(Vz)
    # diagonal radial blocks B[j,k] = (Vᵀ K_BB V) restricted to angular mode (j,k)
    A = np.einsum('tj,zk,rtzRTZ->rjkRTZ', Vt, Vz, K6)
    B = np.einsum('Tj,Zk,rjkRTZ->jkrR', Vt, Vz, A)   # (nt, nz, nr, nr)
    Binv = np.linalg.inv(B)
    # apply must be jittable (the Schur build traces it via jax.lax.map) -> jnp
    Vt_j, Vz_j, Bi_j = jnp.asarray(Vt), jnp.asarray(Vz), jnp.asarray(Binv)

    def apply(rb, _Vt=Vt_j, _Vz=Vz_j, _Bi=Bi_j):
        r = rb.reshape(nr, nt, nz)
        y = jnp.einsum('tj,zk,rtz->rjk', _Vt, _Vz, r)        # to angular eigenbasis
        z = jnp.einsum('jkrR,Rjk->rjk', _Bi, y)              # batched radial solve
        x = jnp.einsum('tj,zk,rjk->rtz', _Vt, _Vz, z)        # back
        return x.reshape(-1)
    return apply


def _build_rt_coupled_inverse(seq, ops, core_size, n, factors, theta_band):
    """rank-(2·band+1) θ: keep the θ coupling out to ±`theta_band` Fourier modes
    (cosθ → m±1 is exactly tridiagonal in the θ-DFT basis, so band=1 == rank-2 θ),
    ζ diagonal (rank-1), r dense. Per ζ-mode, invert the (n_r·n_t) banded-θ (r,θ)
    operator (host-side dense pinv here — production would block-Thomas the band).
      band=0   -> rank-1 θ (== radial-dense sanity check)
      band=1   -> rank-2 θ (cosθ tridiagonal)
      band>=nt//2 -> full (r,θ) coupling (exact on the toroid where ζ is rank-1)."""
    nr, nt, nz = factors.bulk_shape
    bulk = nr * nt * nz
    cols = []
    for j in range(bulk):
        e = jnp.zeros((n,), dtype=jnp.float64).at[core_size + j].set(1.0)
        cols.append(np.asarray(apply_stiffness(seq, ops, e, 0, dirichlet=DBC))[core_size:])
    KBB = np.stack(cols, axis=1); KBB = 0.5 * (KBB + KBB.T)
    K6 = KBB.reshape(nr, nt, nz, nr, nt, nz)
    # unitary DFT for θ and ζ -> cosθ becomes exactly m±1 (cyclic tridiagonal)
    Wt = np.exp(-2j * np.pi * np.outer(np.arange(nt), np.arange(nt)) / nt) / np.sqrt(nt)
    Wz = np.exp(-2j * np.pi * np.outer(np.arange(nz), np.arange(nz)) / nz) / np.sqrt(nz)
    A1 = np.einsum('ta,zp,rtzRTZ->rapRTZ', np.conj(Wt), np.conj(Wz), K6, optimize=True)
    Khat = np.einsum('Tb,Zq,rapRTZ->rapRbq', Wt, Wz, A1, optimize=True)   # (r,a,p,R,b,q)
    aa = np.arange(nt)
    cyc = np.minimum(np.abs(aa[:, None] - aa[None, :]), nt - np.abs(aa[:, None] - aa[None, :]))
    band = cyc <= theta_band                                              # (nt,nt) cyclic θ band
    Minv = np.zeros((nz, nr * nt, nr * nt), dtype=complex)
    for p in range(nz):                                                  # diagonal ζ
        Mp = Khat[:, :, p, :, :, p] * band[None, :, None, :]             # (nr,nt,nr,nt), banded θ
        Mp2 = Mp.reshape(nr * nt, nr * nt)
        Mp2 = 0.5 * (Mp2 + Mp2.conj().T)                                 # Hermitian
        Minv[p] = np.linalg.pinv(Mp2, rcond=1e-12, hermitian=True)
    Wt_j, Wz_j, Mi_j = jnp.asarray(Wt), jnp.asarray(Wz), jnp.asarray(Minv)

    def apply(rb, _Wt=Wt_j, _Wz=Wz_j, _Mi=Mi_j):
        r = rb.reshape(nr, nt, nz).astype(jnp.complex128)
        y = jnp.einsum('ta,zp,rtz->rap', jnp.conj(_Wt), jnp.conj(_Wz), r)   # to (θ,ζ) Fourier
        yk = jnp.transpose(y, (2, 0, 1)).reshape(nz, nr * nt)               # per ζ: (r,a) order
        zk = jnp.einsum('pab,pb->pa', _Mi, yk)                              # banded (r,θ) solve
        z = jnp.transpose(zk.reshape(nz, nr, nt), (1, 2, 0))               # (nr,nt,nz)
        x = jnp.einsum('ta,zp,rap->rtz', _Wt, _Wz, z)                      # back
        return jnp.real(x).reshape(-1)
    return apply


def deflated_smoother(seq, ops, cs):
    """const-deflated surgery preconditioner (production FD bulk inverse, the exact
    dense bulk inverse when EXACT_BULK, the exact dense inverse of the separable
    MODEL when EXACT_MODEL_INV, the ζ-first/(r,θ)-dense inverse when ZETA_FIRST, or
    the θ,ζ-diagonal/radial-dense inverse when RADIAL_DENSE) + L_0 + the A-PD projector."""
    _factors = ops.k0_tensor_hodge_precond.dbc if DBC else ops.k0_tensor_hodge_precond.free
    n = int(seq.n0_dbc) if DBC else int(seq.n0)
    if RADIAL_DENSE_PROD:
        # validate the PRODUCTION path: rebuild the k=0 precond with radial_dense=True
        # (mrx.operators) and let the normal apply branch to the radial-dense form.
        pair = _assemble_k0_tensor_hodge_preconditioner(
            seq, ops, rank=RADIAL_DENSE_PROD_RANK, cp_maxiter=100, cp_tol=1e-9,
            cp_ridge=1e-12, precompute_coupling=False, radial_dense=True,
            dirichlet_flags=(DBC,))
        _prodf = pair.dbc if DBC else pair.free
        bulk_inv = lambda rb, _f=_prodf: _apply_k0_tensor_hodge_bulk_inverse(_f, rb)
    elif EXACT_BULK:
        bulk_inv = _build_exact_bulk_inverse(seq, ops, cs, n)
    elif EXACT_MODEL_INV:
        bulk_inv = _build_model_exact_inverse(_factors)
    elif ZETA_FIRST:
        bulk_inv = _build_zeta_first_inverse(seq, ops, cs, n, _factors.bulk_shape)
    elif RADIAL_DENSE:
        bulk_inv = _build_radial_dense_inverse(seq, ops, cs, n, _factors)
    elif RT_COUPLED:
        bulk_inv = _build_rt_coupled_inverse(seq, ops, cs, n, _factors, RT_BAND)
    else:
        bulk_inv = lambda rb, _f=_factors: _apply_k0_tensor_hodge_bulk_inverse(_f, rb)
    sm_raw = make_surgery_smoother(seq, ops, cs, bulk_inv)

    def s_hat(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=DBC)

    # k=0 free has the constant nullspace; k=0 dbc (outer clamp) has none. With no
    # nullspace the M0-orthogonal deflation is just the identity.
    c0 = np.asarray(_nullspace_vectors(ops, 0, DBC))
    if c0.size == 0:
        ident = lambda x: x
        return s_hat, sm_raw, ident

    c0 = jnp.asarray(c0)
    Mc0 = jnp.stack([apply_mass_matrix(seq, ops, c0[i], 0, dirichlet=DBC)
                     for i in range(c0.shape[0])], axis=0)
    cn = jnp.sqrt(jnp.einsum("ij,ij->i", c0, Mc0))
    c0, Mc0 = c0 / cn[:, None], Mc0 / cn[:, None]

    def Pp(x):
        return x - jnp.einsum("i,ij->j", Mc0 @ x, c0)

    def Pd(b):
        return b - jnp.einsum("i,ij->j", c0 @ b, Mc0)

    def sm(b):
        return Pp(sm_raw(Pd(b)))

    return s_hat, sm, Pp


# --------------------------------------------------------------------------- #
# Eigenvector characterization                                                #
# --------------------------------------------------------------------------- #

def _modal_axis_bases(factors):
    """The per-axis bases the bulk inverse actually diagonalizes in: the shared
    modal basis for the rank>1 path, else the rank-1 FD eigenvectors V_*."""
    if factors.bulk_modal_basis_r is not None:
        return (np.asarray(factors.bulk_modal_basis_r),
                np.asarray(factors.bulk_modal_basis_t),
                np.asarray(factors.bulk_modal_basis_z))
    return (np.asarray(factors.bulk_V_r),
            np.asarray(factors.bulk_V_t),
            np.asarray(factors.bulk_V_z))


def _modal_denom(factors):
    """The preconditioner's per-mode denominator D[i,j,k] (its eigenvalue model
    for each FD mode). Mirrors _apply_k0_tensor_hodge_bulk_*_inverse exactly."""
    nr, nt, nz = factors.bulk_shape
    if factors.bulk_modal_denom is not None:
        return np.asarray(factors.bulk_modal_denom).reshape(nr, nt, nz)
    if len(factors.bulk_modal_op_r) > 0:
        D = np.zeros((nr, nt, nz))
        for op_r, op_t, op_z in zip(factors.bulk_modal_op_r,
                                    factors.bulk_modal_op_t,
                                    factors.bulk_modal_op_z):
            D += (np.asarray(op_r)[:, None, None]
                  * np.asarray(op_t)[None, :, None]
                  * np.asarray(op_z)[None, None, :])
        return D
    # rank-1 FD path: denom = a0*lam_r (+) a1*lam_t (+) a2*lam_z  (see _fd_apply_3d).
    a = np.asarray(factors.bulk_alpha)
    lr = np.asarray(factors.bulk_lam_r)
    lt = np.asarray(factors.bulk_lam_t)
    lz = np.asarray(factors.bulk_lam_z)
    return (a[0] * lr[:, None, None]
            + a[1] * lt[None, :, None]
            + a[2] * lz[None, None, :])


def _bulk_to_modal(factors, x_bulk):
    """Project a bulk DOF-coefficient grid onto the FD eigenbasis (V^T x on each
    axis), giving modal coefficients m[i,j,k] -- the representation in which the
    preconditioner's inverse is diagonal."""
    Vr, Vt, Vz = _modal_axis_bases(factors)
    nr, nt, nz = factors.bulk_shape
    m = np.asarray(x_bulk).reshape(nr, nt, nz)
    m = np.einsum('ji,jkl->ikl', Vr, m)
    m = np.einsum('ji,kjl->kil', Vt, m)
    m = np.einsum('ji,klj->kli', Vz, m)
    return m


def _ascii_bar(vals, width=46):
    """One-line-per-entry ASCII bar chart of a 1-D nonnegative profile."""
    vals = np.asarray(vals, dtype=float)
    vmax = float(np.max(vals)) if vals.size and np.max(vals) > 0 else 1.0
    lines = []
    for i, v in enumerate(vals):
        nb = int(round(width * v / vmax))
        lines.append(f"   [{i:2d}] {v:10.3e} |" + "#" * nb)
    return "\n".join(lines)


def characterize_eigvec(v, factors, core_size, m0, label, outdir, const_vec=None, l0=None):
    """Print + save where the small eigenvector v lives (core/bulk, radial /
    poloidal / toroidal profiles in DOF and FD-modal space). v is a length-n0
    vector in the native [core, bulk] layout. m0(x) applies M_0, l0(x) applies
    the true stiffness L_0 (for the near-harmonic Rayleigh-quotient test)."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    nr, nt, nz = factors.bulk_shape
    n0 = v.shape[0]
    assert n0 == core_size + nr * nt * nz, (n0, core_size, (nr, nt, nz))

    v_core = v[:core_size]
    v_bulk = v[core_size:].reshape(nr, nt, nz)

    # energy split, Euclidean and in the M_0 metric
    e_core = float(np.sum(v_core ** 2))
    e_bulk = float(np.sum(v_bulk ** 2))
    Mv = np.asarray(m0(jnp.asarray(v)))
    em_core = float(np.dot(v_core, Mv[:core_size]))
    em_bulk = float(np.dot(v[core_size:], Mv[core_size:]))
    em_tot = em_core + em_bulk

    print(f"\n--- eigvec characterization [{label}] ---", flush=True)
    print(f"  bulk grid (nr,nt,nz) = {(nr, nt, nz)}; core(polar-axis) DOFs = {core_size}", flush=True)
    print(f"  energy split  Euclidean:  core={100*e_core:6.2f}%  bulk={100*e_bulk:6.2f}%", flush=True)
    print(f"  energy split  M_0 metric: core={100*em_core/em_tot:6.2f}%  bulk={100*em_bulk/em_tot:6.2f}%", flush=True)
    if const_vec is not None:
        cv = np.asarray(const_vec, dtype=float)
        ov = abs(float(np.dot(cv, Mv))) / (np.sqrt(abs(float(np.dot(cv, np.asarray(m0(jnp.asarray(cv))))))) + 1e-300)
        print(f"  |<v, const>_M0| (deflation sanity, expect ~0): {ov:.2e}", flush=True)

    # NEAR-HARMONIC TEST: true-operator Dirichlet energy density. If this is
    # anomalously small the mode is a genuine near-harmonic of the Laplacian (a
    # r^{+-m} family member living near the cut-out axis), NOT just an FD-model
    # artifact. Compared against the bulk of the spectrum in main().
    if l0 is not None:
        L0v = np.asarray(l0(jnp.asarray(v)))
        rho_l0 = float(np.dot(v, L0v)) / max(em_tot, 1e-300)  # vT L0 v / vT M0 v
        print(f"  Rayleigh quotient  rho_L0 = vT L0 v / vT M0 v = {rho_l0:.4e} "
              f"(true Dirichlet energy density of this direction)", flush=True)

    # --- DOF-coefficient space: radial / poloidal / toroidal energy profiles ---
    er = np.sum(v_bulk ** 2, axis=(1, 2))   # per i_r
    et = np.sum(v_bulk ** 2, axis=(0, 2))   # per i_t
    ez = np.sum(v_bulk ** 2, axis=(0, 1))   # per i_z
    er_n = er / max(er.sum(), 1e-300)
    com_r = float(np.dot(np.arange(nr), er_n))  # radial centre-of-mass index
    axis_frac0 = float(er_n[0])
    axis_frac01 = float(er_n[:2].sum())
    print(f"\n  [DOF-coefficient space] i_r=0 is the polar axis (innermost radial coeff)", flush=True)
    print(f"  radial energy profile (per i_r, summed over t,z):", flush=True)
    print(_ascii_bar(er_n), flush=True)
    print(f"    radial centre-of-mass index = {com_r:.2f} / {nr-1};  "
          f"frac at i_r=0: {axis_frac0:.3f}, at i_r<=1: {axis_frac01:.3f}  "
          f"({'AXIS-LOCALIZED' if axis_frac01 > 0.5 else 'spread/outer'})", flush=True)
    # signed radial coefficient profile at the dominant (i_t,i_z) slice -- the
    # actual radial SHAPE, to eyeball monotone-decay-toward-axis (log r / r^{-m})
    # vs oscillatory. (i_r=0 is the axis; needs >~8 radial DOFs to read a curve.)
    epeak = np.sum(v_bulk ** 2, axis=0)        # (nt,nz) energy per angular DOF
    it0, iz0 = np.unravel_index(int(np.argmax(epeak)), (nt, nz))
    radial_slice = v_bulk[:, it0, iz0]
    print(f"    signed radial coeff profile at peak angular DOF (i_t={it0},i_z={iz0}), i_r=0..{nr-1} (axis->edge):",
          flush=True)
    print("      " + " ".join(f"{c:+.3e}" for c in radial_slice), flush=True)
    # poloidal Fourier content: periodic B-spline coeffs ~ diagonalize under the
    # DFT, so |FFT| along the theta-DOF axis reads off the wavenumber m. Energy
    # per |m|, summed over r,z (robust to sign cancellation). m=0 would be the
    # theta-flat log r; m!=0 is the r^{+-m} family.
    F = np.fft.fft(v_bulk, axis=1)
    m_en = np.sum(np.abs(F) ** 2, axis=(0, 2))
    half = nt // 2
    m_fold = np.zeros(half + 1)
    for mm in range(nt):
        k = mm if mm <= half else nt - mm
        m_fold[k] += m_en[mm]
    m_fold /= max(m_fold.sum(), 1e-300)
    m_dom = int(np.argmax(m_fold))
    print(f"  poloidal Fourier |m| spectrum (m=0 is theta-flat/log r; m!=0 is r^(+-m)):", flush=True)
    print("    " + "  ".join(f"m{mm}:{m_fold[mm]:.3f}" for mm in range(half + 1)), flush=True)
    print(f"    dominant |m| = {m_dom}  ({'theta-flat (m=0)' if m_dom == 0 else 'NOT log r -- has poloidal structure'})",
          flush=True)
    print(f"  poloidal energy profile (per i_t):", flush=True)
    print(_ascii_bar(et / max(et.sum(), 1e-300)), flush=True)
    print(f"  toroidal energy profile (per i_z):", flush=True)
    print(_ascii_bar(ez / max(ez.sum(), 1e-300)), flush=True)

    # top DOFs
    flat = v_bulk.reshape(-1)
    top = np.argsort(np.abs(flat))[::-1][:8]
    print(f"  top 8 bulk DOFs (i_r,i_t,i_z) -> coeff:", flush=True)
    for idx in top:
        ir, it, iz = np.unravel_index(idx, (nr, nt, nz))
        print(f"     ({ir:2d},{it:2d},{iz:2d}) -> {flat[idx]:+.3e}", flush=True)
    if core_size:
        ic = int(np.argmax(np.abs(v_core)))
        print(f"  largest core(axis) DOF: idx {ic} -> {v_core[ic]:+.3e}", flush=True)

    # --- FD-modal space: which modes the small direction excites ---
    D = _modal_denom(factors)
    m = _bulk_to_modal(factors, v[core_size:])
    me = m ** 2
    me_tot = max(me.sum(), 1e-300)
    mr = me.sum(axis=(1, 2))
    mt = me.sum(axis=(0, 2))
    mz = me.sum(axis=(0, 1))
    dom = np.unravel_index(int(np.argmax(me)), (nr, nt, nz))
    # energy-weighted mean denom, and energy that sits in the lowest-denom decile
    order = np.argsort(D.reshape(-1))
    cume = np.cumsum(me.reshape(-1)[order]) / me_tot
    n_lowdec = max(1, (nr * nt * nz) // 10)
    low_denom_frac = float(np.sum(me.reshape(-1)[order[:n_lowdec]]) / me_tot)
    print(f"\n  [FD-modal space] (the basis the bulk inverse is diagonal in)", flush=True)
    print(f"  radial modal energy (per radial mode):", flush=True)
    print(_ascii_bar(mr / me_tot), flush=True)
    print(f"  poloidal modal energy:", flush=True)
    print(_ascii_bar(mt / me_tot), flush=True)
    print(f"  toroidal modal energy:", flush=True)
    print(_ascii_bar(mz / me_tot), flush=True)
    print(f"    dominant FD mode (i,j,k)={dom}  carries {100*me[dom]/me_tot:.1f}%  "
          f"of bulk modal energy; its precond denom D={D[dom]:.3e} "
          f"(min D over grid = {D.min():.3e}, max = {D.max():.3e})", flush=True)
    print(f"    frac of modal energy in the lowest-denom 10% of modes: {low_denom_frac:.3f} "
          f"({'LOW-FREQUENCY / near-constant-ish' if low_denom_frac > 0.5 else 'not low-freq dominated'})", flush=True)

    # --- save arrays + a PNG ---
    outdir.mkdir(parents=True, exist_ok=True)
    npz = outdir / f"eigvec_{label}.npz"
    np.savez(
        npz,
        eigvec=v, v_core=v_core, v_bulk=v_bulk, bulk_shape=np.array([nr, nt, nz]),
        radial_energy=er, poloidal_energy=et, toroidal_energy=ez,
        modal_radial=mr, modal_poloidal=mt, modal_toroidal=mz,
        modal_denom=D, modal_coeffs=m,
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax[0, 0].bar(range(nr), er_n);           ax[0, 0].set_title("DOF radial (i_r; 0=axis)")
        ax[0, 1].bar(range(nt), et / et.sum());  ax[0, 1].set_title("DOF poloidal (i_t)")
        ax[0, 2].bar(range(nz), ez / ez.sum());  ax[0, 2].set_title("DOF toroidal (i_z)")
        ax[1, 0].bar(range(nr), mr / me_tot);    ax[1, 0].set_title("FD-modal radial")
        ax[1, 1].bar(range(nt), mt / me_tot);    ax[1, 1].set_title("FD-modal poloidal")
        ax[1, 2].bar(range(nz), mz / me_tot);    ax[1, 2].set_title("FD-modal toroidal")
        for a in ax.ravel():
            a.set_ylabel("energy frac")
        fig.suptitle(f"small-eigvec DOF/modal structure [{label}]  "
                     f"core={100*em_core/em_tot:.0f}%M0 bulk={100*em_bulk/em_tot:.0f}%M0")
        fig.tight_layout()
        png = outdir / f"eigvec_{label}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"  saved: {npz.name}, {png.name}  (in {outdir})", flush=True)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"  saved: {npz.name}  (PNG skipped: {exc})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", type=str, default="toroid")
    ap.add_argument("--ranks", type=str, default="1,2,3")
    ap.add_argument("--ns", type=str, default="6,12,4")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.2)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=600)
    ap.add_argument("--dbc", action="store_true",
                    help="clamp the OUTER boundary (Dirichlet) instead of free. The inner "
                         "ring is the cut-out polar axis -- always free, no BC there. Tests "
                         "whether the axis-localized outlier survives an outer clamp (+ "
                         "constant-nullspace removal) or is a free-spectrum artifact.")
    ap.add_argument("--exact-bulk", action="store_true",
                    help="replace the separable FD bulk inverse with the exact dense "
                         "bulk-block inverse. If the outlier vanishes (kappa->~1), it is "
                         "the FD separability error, not surgery/structure.")
    ap.add_argument("--model-exact-inverse", action="store_true",
                    help="invert the separable MODEL K~_bb exactly (dense) instead of via "
                         "FD/Lynch. Isolates the K_bb~=K~_bb approximation error from the "
                         "FD inversion error (the latter is the rank>1 Lynch pathology).")
    ap.add_argument("--radial-dense", action="store_true",
                    help="θ,ζ rank-1 (FD angular eigenbasis) + exact dense radial block per "
                         "angular mode. Measures whether dropping the cosθ angular coupling "
                         "(rank-1 θ) leaves the outlier -> if it survives, cosθ matters.")
    ap.add_argument("--radial-dense-prod", action="store_true",
                    help="validate the PRODUCTION radial-dense bulk inverse (mrx.operators, "
                         "built from the rank-r CP metric) -- should reproduce --radial-dense.")
    ap.add_argument("--radial-dense-prod-rank", type=int, default=2,
                    help="CP rank for the production radial-dense build (>=2 to capture the "
                         "radial coupling).")
    ap.add_argument("--rt-coupled", action="store_true",
                    help="rank-2 θ: keep the θ coupling as a band (cosθ tridiagonal in the "
                         "θ-DFT basis), ζ diagonal, r dense. Tests whether the cosθ band "
                         "pushes κ from ~2.15 toward 1.")
    ap.add_argument("--rt-band", type=int, default=1,
                    help="θ Fourier band width: 0=rank-1 θ, 1=rank-2 θ (cosθ tridiagonal), "
                         "large=full (r,θ) coupling.")
    ap.add_argument("--outdir", type=str, default=None,
                    help="where to write eigvec npz/png (default outputs/diag_ritz/eigvec_<geom>_<ns>)")
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]

    global DBC, EXACT_BULK, EXACT_MODEL_INV, RADIAL_DENSE, RADIAL_DENSE_PROD, RADIAL_DENSE_PROD_RANK
    global RT_COUPLED, RT_BAND
    DBC = bool(args.dbc)
    EXACT_BULK = bool(args.exact_bulk)
    EXACT_MODEL_INV = bool(args.model_exact_inverse)
    RADIAL_DENSE = bool(args.radial_dense)
    RADIAL_DENSE_PROD = bool(args.radial_dense_prod)
    RADIAL_DENSE_PROD_RANK = int(args.radial_dense_prod_rank)
    RT_COUPLED = bool(args.rt_coupled)
    RT_BAND = int(args.rt_band)
    bc_tag = (("dbc" if DBC else "free") + ("_exactbulk" if EXACT_BULK else "")
              + ("_modelexactinv" if EXACT_MODEL_INV else "")
              + ("_radialdense" if RADIAL_DENSE else "")
              + (f"_radialdenseprod{RADIAL_DENSE_PROD_RANK}" if RADIAL_DENSE_PROD else "")
              + (f"_rtband{RT_BAND}" if RT_COUPLED else ""))

    ns_tag = "x".join(str(v) for v in args.ns)
    outdir = Path(args.outdir) if args.outdir else (
        ROOT / "outputs" / "diag_ritz" / f"eigvec_{args.geometry}_{ns_tag}_{bc_tag}")

    dg.DIRICHLET = DBC
    print(f"[diag] DENSE eig spectrum + small-eigvec characterization: "
          f"geometry={args.geometry} ns={args.ns} p={args.p} {bc_tag} BC; ranks={ranks}", flush=True)

    t = time.perf_counter()
    seq = build_sequence(args)
    assemble_operators(seq, rank=ranks[0], klevel=0)
    # This is a k=0-only probe: it only needs the k=0 constant nullspace. Computing
    # the full BETTI=(1,1,0,0) also solves for the k=1 harmonic, which is cheap on
    # the axisymmetric toroid but *very* slow on rotating_ellipse (b_1=1, nontrivial
    # cohomology). Restrict to (1,0,0,0) so rotating_ellipse builds as fast as toroid.
    k0_betti = (dg.BETTI[0], 0, 0, 0)
    seq._compute_nullspaces(k0_betti)
    n0 = int(seq.n0_dbc) if DBC else int(seq.n0)
    cs = int(_core_size(seq))
    print(f"[diag] build in {(time.perf_counter()-t)*1e3:.1f} ms; n0={n0} core={cs}", flush=True)

    small_vecs = {}  # rank -> lmin eigenvector (native n0 layout) for cross-rank compare
    m0_ref = None
    const_ref = None
    for ri, rank in enumerate(ranks):
        print(f"\n[diag] >>> rank {rank} ({ri+1}/{len(ranks)}): assembling operators ...", flush=True)
        ta = time.perf_counter()
        ops = assemble_operators(seq, rank=rank, klevel=0)
        s_hat, sm, Pp = deflated_smoother(seq, ops, cs)
        _inv_res = _bulk_inversion_residual(
            ops.k0_tensor_hodge_precond.dbc if DBC else ops.k0_tensor_hodge_precond.free)
        print(f"[diag]     FD bulk inversion residual on its own model K~_bb: {_inv_res:.2e} "
              f"({'EXACT (rank-1 Kronecker)' if _inv_res < 1e-10 else 'INEXACT (Lynch truncation)'})",
              flush=True)
        print(f"[diag]     assembled + built smoother in {(time.perf_counter()-ta)*1e3:.0f} ms; "
              f"materializing dense operator (n0={n0}) ...", flush=True)
        ritz, vecs = dense_spectrum(s_hat, sm, n0, Pp, label=f"r{rank}")
        lo = ritz[:15]
        hi = ritz[-3:]
        # largest relative gap in the bottom quarter (consecutive ratio)
        nb = max(3, len(ritz) // 4)
        bottom = ritz[:nb]
        ratios = bottom[1:] / np.maximum(bottom[:-1], 1e-300)
        gi = int(np.argmax(ratios))
        print(f"\n========== rank {rank}  ({len(ritz)} nonzero eigs) ==========", flush=True)
        print(f"  kappa = lmax/lmin = {ritz[-1]/max(ritz[0],1e-300):.3e}  "
              f"(lmin={ritz[0]:.3e}, lmax={ritz[-1]:.3e})", flush=True)
        print(f"  bottom 15 eigs: " + " ".join(f"{v:.3e}" for v in lo), flush=True)
        print(f"  top 3 eigs:     " + " ".join(f"{v:.3e}" for v in hi), flush=True)
        print(f"  largest gap in bottom {nb}: ritz[{gi}]={bottom[gi]:.3e} -> "
              f"ritz[{gi+1}]={bottom[gi+1]:.3e}  (ratio {ratios[gi]:.2e}); "
              f"=> {gi+1} mode(s) below the gap", flush=True)

        def m0(x, _ops=ops):
            return apply_mass_matrix(seq, _ops, x, 0, dirichlet=DBC)
        if m0_ref is None:
            m0_ref = m0
            _nv = np.asarray(_nullspace_vectors(ops, 0, DBC))
            const_ref = _nv[0] if _nv.size else None

        # near-harmonic table: true Dirichlet energy density rho_L0 of the bottom
        # eigenvectors vs the spectrum-wide range. If the lmin mode's rho_L0 is
        # near the global minimum it is a genuine near-harmonic; if it is typical,
        # the small composite eigenvalue is purely an FD-preconditioner artifact.
        def rho_L0(vj):
            vj = np.asarray(vj, dtype=float)
            num = float(np.dot(vj, np.asarray(s_hat(jnp.asarray(vj)))))
            den = float(np.dot(vj, np.asarray(m0(jnp.asarray(vj)))))
            return num / max(den, 1e-300)
        all_rho = np.array([rho_L0(vecs[:, j]) for j in range(vecs.shape[1])])
        print(f"  rho_L0 (true Dirichlet energy density) spectrum: "
              f"min={all_rho.min():.3e} median={np.median(all_rho):.3e} max={all_rho.max():.3e}", flush=True)
        print(f"  bottom-eigenvalue modes: lambda(B) -> rho_L0 (rank of rho_L0 among all {len(all_rho)} modes):", flush=True)
        rho_order = np.argsort(all_rho)
        rho_rank = {int(idx): r for r, idx in enumerate(rho_order)}
        for j in range(min(6, vecs.shape[1])):
            print(f"     eig#{j}: lambda(B)={ritz[j]:.3e}  rho_L0={all_rho[j]:.3e}  "
                  f"(rho_L0 rank {rho_rank[j]}/{len(all_rho)-1}"
                  f"{' = NEAR-HARMONIC' if rho_rank[j] <= 2 else ''})", flush=True)

        vmin = vecs[:, 0]  # eigenvector of the smallest eigenvalue
        small_vecs[rank] = vmin
        # bulk pencil test (rank-independent operator; do it once, on the first rank)
        if ri == 0:
            bulk_pencil_diagnostic(seq, ops, cs, vecs.shape[0], vmin, f"{args.geometry}_{bc_tag}")
        _ch_factors = ops.k0_tensor_hodge_precond.dbc if DBC else ops.k0_tensor_hodge_precond.free
        characterize_eigvec(vmin, _ch_factors, cs, m0,
                            f"{args.geometry}_r{rank}_{bc_tag}", outdir, const_vec=const_ref,
                            l0=s_hat)

    # cross-rank: is the rank-1 small mode the same direction as the rank>1
    # basement outlier? (M0-cosine of the lmin eigenvectors)
    if len(small_vecs) > 1 and m0_ref is not None:
        print("\n========== cross-rank lmin-eigvec M0-overlap ==========", flush=True)
        base = ranks[0]
        vb = small_vecs[base]
        Mvb = np.asarray(m0_ref(jnp.asarray(vb)))
        nb_ = np.sqrt(abs(float(np.dot(vb, Mvb))))
        for r in ranks[1:]:
            vr = small_vecs[r]
            Mvr = np.asarray(m0_ref(jnp.asarray(vr)))
            cos = abs(float(np.dot(vb, Mvr))) / (nb_ * np.sqrt(abs(float(np.dot(vr, Mvr)))) + 1e-300)
            print(f"  |cos_M0( lmin_r{base} , lmin_r{r} )| = {cos:.4f}  "
                  f"({'SAME direction (milder seed of the outlier)' if cos > 0.9 else 'DIFFERENT direction'})",
                  flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
