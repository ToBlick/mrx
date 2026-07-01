"""Validate the k=1 HX preconditioner with a TENSOR P_A (greville and CP).

The earlier `greville_k1_stiffness_verify.py` test applied the tensor P_A RAW to
the bare curl-curl K_1 -- no gradient-complement projectors, no P_B -- and it
stalled/NaN'd on W7-X. That is the WRONG test: a block-diagonal P_A only
preconditions the curl complement; the gradient (grad-div) part of the full k=1
Hodge Laplacian L_1 is P_B's job, and the exposed P_A apply must be shielded by
the grad-complement projector (I - Pi_g). This is exactly how the k=2 path uses
the k=1 tensor P_A as its K_1^{-1} atom (make_apply_routines_k2,
benchmark_graddiv_k1_preconditioner.py ~2135-2161) -- where it works.

Here we precondition the condensed k=1 Hodge Laplacian
    L_1 = apply_hodge_laplacian_approx(.,1)   (= S_1 + D_0 M_0^{-1} D_0^T)
with the proper HX preconditioner
    M_HX = (I-Pi_g) P_A (I-Pi_g^*) + P_B,   Pi_g = G_0 L_0^{-1} G_0^T M_1,
           P_B = G_0 L_0^{-1} M_0 L_0^{-1} G_0^T,
and L_0^{-1} done by Chebyshev (cheb-L_0, const-deflated k=0 tensor atom). P_A is
the LIBRARY tensor stiffness preconditioner, greville or CP. We compare, per
geometry / BC:
    none | raw P_A (greville, cp) | HX = projected P_A + P_B (greville, cp)

Run (GPU):
  W7X_MAP_BATCH=256 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 \
    .venv/bin/python scripts/debug/greville_k1_hx_verify.py --geometry toroid
  ... --geometry w7x --nfp 5
"""
from __future__ import annotations
import argparse
import os
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

from benchmark_graddiv_k1_preconditioner import (  # noqa: E402
    build_sequence,
    assemble_operators,
    make_chebyshev_upper,
    _lanczos_extremal_eigs_precond,
)
from mrx.operators import (  # noqa: E402
    _nullspace_vectors,
    apply_stiffness,
    apply_mass_matrix,
    apply_incidence_matrix,
    apply_hodge_laplacian_approx,
    apply_laplacian_preconditioner,
    apply_stiffness_tensor_preconditioner,
    assemble_tensor_stiffness_preconditioner,
)
from mrx.solvers import solve_singular_cg  # noqa: E402
from mrx.nullspace import compute_nullspaces_iterative  # noqa: E402


def build_cheb_l0(seq, ops, d, c0):
    """near-exact L_0^{-1} via Chebyshev on L_0 with the const-deflated k=0 tensor
    atom as smoother (copied from benchmark_graddiv_k1_preconditioner.py ~2205).
    c0 = the k=0 constant nullspace vectors (free BC) or empty (dbc)."""
    n0 = int(seq.n0_dbc if d else seq.n0)
    c0 = jnp.asarray(c0)
    # keep only vectors with non-degenerate M_0-norm (a zero placeholder breaks deflation)
    if c0.shape[0] > 0:
        Mc0 = jnp.stack([apply_mass_matrix(seq, ops, c0[i], 0, dirichlet=d)
                         for i in range(c0.shape[0])], axis=0)
        c0n = jnp.sqrt(jnp.einsum("ij,ij->i", c0, Mc0))
        keep = np.asarray(c0n) > 1e-30
        c0, Mc0, c0n = c0[keep], Mc0[keep], c0n[keep]
    if c0.shape[0] > 0:
        c0, Mc0 = c0 / c0n[:, None], Mc0 / c0n[:, None]

        def defl_primal(x):
            return x - jnp.einsum("i,ij->j", Mc0 @ x, c0)

        def defl_dual(b):
            return b - jnp.einsum("i,ij->j", c0 @ b, Mc0)
    else:
        def defl_primal(x):
            return x
        defl_dual = defl_primal

    def l0_smoother(b):
        return defl_primal(apply_laplacian_preconditioner(
            seq, ops, defl_dual(b), 0, dirichlet=d, kind="tensor"))

    def s_hat0(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=d)

    lmin0, lmax0 = _lanczos_extremal_eigs_precond(
        s_hat0, l0_smoother, n0, steps=30, seed=0, project=defl_primal)
    lmin0 = max(lmin0, lmax0 * 1e-5)
    kap0 = lmax0 / max(lmin0, 1e-300)
    deg0 = int(min(max(int(np.ceil(0.5 * np.sqrt(kap0) * np.log(2.0 / 1e-2))), 1), 100))
    print(f"[atom] cheb_L0 ({'dbc' if d else 'free'}): kappa={kap0:.2e} "
          f"deg={deg0}", flush=True)
    cheb0 = make_chebyshev_upper(s_hat0, l0_smoother, lmin0, lmax0, deg0)
    return lambda r: defl_primal(cheb0(defl_dual(r)))


def build_hx(seq, ops, d, l0_inv, pa_apply):
    """M_HX = (I-Pi_g) pa (I-Pi_g^*) + P_B, with Pi_g/P_B using l0_inv."""
    def G0t(v):   # V1* -> V0*
        return apply_incidence_matrix(seq, ops, v, 0,
                                      dirichlet_in=d, dirichlet_out=d, transpose=True)

    def G0(w):    # V0 -> V1
        return apply_incidence_matrix(seq, ops, w, 0,
                                      dirichlet_in=d, dirichlet_out=d, transpose=False)

    def M1(v):
        return apply_mass_matrix(seq, ops, v, 1, dirichlet=d)

    def M0(w):
        return apply_mass_matrix(seq, ops, w, 0, dirichlet=d)

    def grad_primal_complement(u):     # (I - Pi_g) u : V1 -> V1
        return u - G0(l0_inv(G0t(M1(u))))

    def grad_dual_complement(r):       # (I - Pi_g^*) r : V1* -> V1*
        return r - M1(G0(l0_inv(G0t(r))))

    def p_b(r):                        # G_0 L_0^{-1} M_0 L_0^{-1} G_0^T r
        return G0(l0_inv(M0(l0_inv(G0t(r)))))

    def m_hx(r):
        return grad_primal_complement(pa_apply(grad_dual_complement(r))) + p_b(r)

    return m_hx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=4000)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry,
                          cg_tol=args.tol, cg_maxiter=args.maxiter, epsilon=args.epsilon,
                          kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    print(f"=== k=1 HX (proj tensor P_A + P_B, cheb-L0) verify  {args.geometry} "
          f"ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    ops_cp = assemble_operators(seq, klevel=1, both_bc=True)   # tensor stiff = CP
    # Populate the REAL nullspaces (k=0 constant, k=1 harmonic) — assemble_operators
    # leaves zero placeholders, which silently break the free-BC L_0/L_1 deflation.
    ops_cp, _null_info = compute_nullspaces_iterative(
        seq, ops_cp, betti_numbers=(1, 1, 0, 0))
    # greville variant: overwrite the k=1 tensor stiffness factors (inherits nullspaces)
    ops_grev = assemble_tensor_stiffness_preconditioner(
        seq, operators=ops_cp, ks=(1,), cp_kwargs={"greville": True})

    rows = []  # (bc, name, it, conv, rr)
    for d in (True, False):
        bc = "dbc" if d else "free"
        n1 = int(seq.n1_dbc if d else seq.n1)
        c0 = _nullspace_vectors(ops_cp, 0, d)   # k=0 constant (free) — for cheb-L0 deflation
        vs = _nullspace_vectors(ops_cp, 1, d)   # k=1 harmonic (free) — for outer solve

        def A(v, _d=d):
            return apply_hodge_laplacian_approx(seq, ops_cp, v, 1, dirichlet=_d)

        def mass(v, _d=d):
            return apply_mass_matrix(seq, ops_cp, v, 1, dirichlet=_d)

        b = A(jax.random.normal(jax.random.PRNGKey(700 + d), (n1,), dtype=jnp.float64))

        l0_cp = build_cheb_l0(seq, ops_cp, d, c0)
        l0_grev = build_cheb_l0(seq, ops_grev, d, c0)

        def pa_cp(x, _d=d):
            return apply_stiffness_tensor_preconditioner(seq, ops_cp, x, 1, dirichlet=_d)

        def pa_grev(x, _d=d):
            return apply_stiffness_tensor_preconditioner(seq, ops_grev, x, 1, dirichlet=_d)

        hx_cp = build_hx(seq, ops_cp, d, l0_cp, pa_cp)
        hx_grev = build_hx(seq, ops_grev, d, l0_grev, pa_grev)

        identity = lambda x: x
        methods = [
            ("none", identity),
            ("raw cp", pa_cp),
            ("raw greville", pa_grev),
            ("HX cp", hx_cp),
            ("HX greville", hx_grev),
        ]
        for name, M in methods:
            try:
                x, info = solve_singular_cg(A, b, mass_matvec=mass, precond_matvec=M,
                                            vs=vs, tol=args.tol, maxiter=args.maxiter)
                rr = float(jnp.linalg.norm(A(x) - b) / jnp.maximum(jnp.linalg.norm(b), 1e-30))
                it = abs(int(info))
                conv = int(info) <= 0
                rows.append((bc, name, it, conv, rr))
            except Exception as exc:
                rows.append((bc, name, -1, False, repr(exc)[:48]))

    print(f"\n{'bc':5} {'method':14} {'it':>6} {'conv':>5} {'rel_res':>11}", flush=True)
    for d in (True, False):
        bc = "dbc" if d else "free"
        for (rb, name, it, conv, rr) in rows:
            if rb == bc:
                rr_s = f"{rr:.2e}" if isinstance(rr, float) else str(rr)
                print(f"{bc:5} {name:14} {it:>6} {str(conv):>5} {rr_s:>11}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
