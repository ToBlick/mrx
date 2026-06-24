"""Last test before unifying the Laplacian preconditioners: do we need the
P_A projection, with the production matrix-free tensor-Chebyshev P_B atom?

Fully MATRIX-FREE (no dense pinv / no dense operators) -> ships to higher
resolutions. The P_B inner inverse of L_{k-1} is a tensor-Chebyshev atom:
  * smoother = the k-1 tensor preconditioner with CONSTANT-DEFLATED inner l0,
  * harmonic M-deflated,
  * degree AUTO-SET from the matrix-free Lanczos kappa, d ~ 0.5 sqrt(kappa) ln(2/eps).

  --klevel 2: invert whole L_1; smoother = k=1 Hodge precond (cdefl l0); P_A =
              capped div-div tensor block.
  --klevel 1: invert scalar L_0; smoother = k=0 tensor Hodge precond (cdefl
              const); P_A = block_fd.

For each resolution and eps it compares, on the k saddle (free BC):
  jacobi  vs  RAW P_A+P_B  vs  PROJECTED (I-Pi)P_A(I-Pi)+P_B,
answering (1) is the projection needed with a good production atom, (2) the
h-scaling of the auto-degree, iterations AND wall time (k=1 cheap-scalar smoother
vs k=2 expensive-vector smoother).

Usage:
    python scripts/benchmark/benchmark_unified_tensor_cheb_preconditioner.py --klevel 2 --ns 6,12,4 --p 3 --eps 1e-1,1e-2
    python scripts/benchmark/benchmark_unified_tensor_cheb_preconditioner.py --klevel 1 --ns 6,12,4 --p 3 --eps 1e-1,1e-2
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
    _nullspace_vectors, apply_incidence_matrix, apply_mass_matrix, apply_stiffness,
    apply_laplacian_preconditioner, apply_stiffness_tensor_preconditioner,
    apply_mass_matrix_preconditioner,
)
import mrx.nullspace as _ns  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    MassPreconditionerSpec, SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec, default_mass_preconditioner,
)
import benchmark_graddiv_k1_preconditioner as dg  # noqa: E402
from mrx.solvers import solve_singular_cg  # noqa: E402
from benchmark_graddiv_k1_preconditioner import (  # noqa: E402
    build_sequence, assemble_operators, make_apply_routines,
    make_apply_routines_k2, make_apply_routines_k3, make_saddle_solve,
    time_saddle_solve, time_solve, make_chebyshev_upper,
    _lanczos_extremal_eigs_precond, _diagonal_from_matvec, _invert_diagonal,
)

_ns_orig = _ns._nullspace_shifted_preconditioner


def _robust_jacobi_shifted(k):
    if k == 0:
        return _ns_orig(k)
    return _ns._validate_nullspace_shifted_preconditioner(
        k, SaddlePointPreconditionerSpec(
            mass=default_mass_preconditioner(),
            schur=SchurPreconditionerSpec(
                inner=MassPreconditionerSpec(kind="tensor"),
                outer=MassPreconditionerSpec(kind="jacobi"))))


_ns._nullspace_shifted_preconditioner = _robust_jacobi_shifted


def make_cheb_l0_atom(seq, ops, eps, DBC, n0, *, max_degree=100):
    """Matrix-free tensor-Chebyshev L_0^{-1} atom for the k=1 P_B (and Pi_g):
    s_hat = L_0 = apply_stiffness(.,0) (exact, matrix-free; no mass inverse);
    smoother = k=0 tensor Hodge precond, CONSTANT-DEFLATED for free BC; degree
    auto-set from Lanczos kappa. Mirrors the k=2 cheb_tensor atom one degree down.
    """
    def s_hat(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=DBC)

    c0 = jnp.asarray(_nullspace_vectors(ops, 0, DBC))   # free -> 1 const, dbc -> 0
    if c0.shape[0] > 0:
        Mc0 = jnp.stack([apply_mass_matrix(seq, ops, c0[i], 0, dirichlet=DBC)
                         for i in range(c0.shape[0])], axis=0)
        cn = jnp.sqrt(jnp.einsum("ij,ij->i", c0, Mc0))
        c0, Mc0 = c0 / cn[:, None], Mc0 / cn[:, None]

        def Pp(x):
            return x - jnp.einsum("i,ij->j", Mc0 @ x, c0)

        def Pd(b):
            return b - jnp.einsum("i,ij->j", c0 @ b, Mc0)

        def smoother(b):
            return Pp(apply_laplacian_preconditioner(seq, ops, Pd(b), 0,
                                                     dirichlet=DBC, kind="tensor"))
    else:
        def Pp(x):
            return x
        Pd = Pp

        def smoother(b):
            return apply_laplacian_preconditioner(seq, ops, b, 0, dirichlet=DBC,
                                                  kind="tensor")

    lmin, lmax = _lanczos_extremal_eigs_precond(s_hat, smoother, n0, steps=50,
                                                seed=0, project=Pp)
    lmin = max(lmin, lmax * 1e-5)
    kap = lmax / max(lmin, 1e-300)
    deg = int(min(max(int(np.ceil(0.5 * np.sqrt(kap) * np.log(2.0 / eps))), 1), max_degree))
    print(f"[atom] cheb_L0: kappa={kap:.3e} interval=[{lmin:.3e},{lmax:.3e}] "
          f"eps={eps} -> degree={deg}", flush=True)
    cheb = make_chebyshev_upper(s_hat, smoother, lmin, lmax, deg)

    def atom(r):
        return Pp(cheb(Pd(r)))
    return atom


def make_cheb_accel(seq, ops, k, s_hat, smoother, eps, DBC, *,
                    max_degree=60, kappa_cap=1e3):
    """Cheb-tensor: Chebyshev acceleration of the WHOLE Tensor (HX) preconditioner
    `smoother` on the operator `s_hat` (~ S-hat_k / L_k). This is exactly the
    L(k)^{-1} proxy we feed into L(k+1), used here directly as the L(k)
    preconditioner. Harmonic-deflated (M_k-orth); degree from Lanczos
    kappa(smoother . s_hat). Returns None if kappa is hopeless (> kappa_cap) so we
    SKIP rather than run a capped-degree failure -- e.g. the bare tensor stiffness
    smoother gives kappa~1e5 (down-modes uncontrolled) and is skipped; the HX
    smoother gives kappa~tens and runs at low degree."""
    n = int(getattr(seq, f"n{k}_dbc" if DBC else f"n{k}"))
    H = jnp.asarray(_nullspace_vectors(ops, k, DBC))
    if H.shape[0] > 0:
        MH = jnp.stack([apply_mass_matrix(seq, ops, H[i], k, dirichlet=DBC)
                        for i in range(H.shape[0])], axis=0)
        nrm = jnp.sqrt(jnp.einsum("ij,ij->i", H, MH))
        H, MH = H / nrm[:, None], MH / nrm[:, None]

        def Pp(x):
            return x - jnp.einsum("i,ij->j", MH @ x, H)

        def Pd(b):
            return b - jnp.einsum("i,ij->j", H @ b, MH)
    else:
        def Pp(x):
            return x
        Pd = Pp

    def sm(b):
        return Pp(smoother(Pd(b)))

    lmin, lmax = _lanczos_extremal_eigs_precond(s_hat, sm, n, steps=50, seed=0, project=Pp)
    lmin = max(lmin, lmax * 1e-5)
    kap = lmax / max(lmin, 1e-300)
    if kap > kappa_cap:
        print(f"[cheb-tensor k={k}] kappa={kap:.2e} > {kappa_cap:.0e} -> NOT VIABLE (skip)",
              flush=True)
        return None
    deg = int(min(max(int(np.ceil(0.5 * np.sqrt(kap) * np.log(2.0 / eps))), 1), max_degree))
    print(f"[cheb-tensor k={k}] kappa={kap:.3e} eps={eps} -> degree={deg}", flush=True)
    cheb = make_chebyshev_upper(s_hat, sm, lmin, lmax, deg)
    return lambda r: Pp(cheb(Pd(r)))


def _mass_deflation(seq, ops, k, DBC):
    """M_k-orthogonal deflation onto h^perp for the k-form harmonic(s) h.
    Returns (Pp, Pd): Pp deflates a primal vector (V_k -> V_k), Pd its dual
    adjoint (V_k* -> V_k*). Identity if there is no harmonic. Same idiom as the
    k=1 atom deflation and make_cheb_accel -- the deflation we MUST apply when
    using an inexact L_k^{-1} (which would otherwise amplify the ~0 harmonic)."""
    H = jnp.asarray(_nullspace_vectors(ops, k, DBC))
    if H.shape[0] == 0:
        ident = lambda x: x
        return ident, ident
    MH = jnp.stack([apply_mass_matrix(seq, ops, H[i], k, dirichlet=DBC)
                    for i in range(H.shape[0])], axis=0)
    nrm = jnp.sqrt(jnp.einsum("ij,ij->i", H, MH))
    H, MH = H / nrm[:, None], MH / nrm[:, None]

    def Pp(x):                         # V_k -> V_k
        return x - jnp.einsum("i,ij->j", MH @ x, H)

    def Pd(b):                         # V_k* -> V_k*  (adjoint of Pp)
        return b - jnp.einsum("i,ij->j", H @ b, MH)
    return Pp, Pd


def _make_tensor_hx(seq, ops, k, eps, n0):
    """The Tensor (HX) preconditioner for L(k): projected P_A + P_B with the inner
    L(k-1) inverse a nested cheb atom at accuracy `eps`. Returns (apply, label)."""
    if k == 3:
        _k2_raw = make_apply_routines_k2(
            seq, ops, grad_project=True, atom="cheb_tensor",
            atom_cheb_eps=eps)["projected_p_a_capped_plus_p_b"]
        # Inverting L_2 for the k=3 P_B: M_2-deflate the k=2 harmonic h_2 so the
        # inexact k2_inv does not amplify it (1:1 analog of the k=1 harmonic
        # deflation inside the k=2 atom). No-op when b_2 = 0 (e.g. free BC here).
        _P2p, _P2d = _mass_deflation(seq, ops, 2, False)

        def k2_inv(r, _k2=_k2_raw, _Pp=_P2p, _Pd=_P2d):
            return _Pp(_k2(_Pd(r)))

        def hx(r, _k2=k2_inv):
            y1 = apply_incidence_matrix(seq, ops, r, 2, dirichlet_in=False,
                                        dirichlet_out=False, transpose=True)
            y2 = _k2(y1)
            y3 = apply_mass_matrix(seq, ops, y2, 2, dirichlet=False)
            y4 = _k2(y3)
            return apply_incidence_matrix(seq, ops, y4, 2, dirichlet_in=False,
                                          dirichlet_out=False, transpose=False)
        return hx
    if k == 2:
        return make_apply_routines_k2(seq, ops, grad_project=True, atom="cheb_tensor",
                                      atom_cheb_eps=eps)["projected_p_a_capped_plus_p_b"]
    atom = make_cheb_l0_atom(seq, ops, eps, False, n0)
    return make_apply_routines(seq, ops, pa_mode="block_fd", grad_project=True,
                               dirichlet_flag=False, l0_inv_custom=atom)["projected_p_a_plus_p_b"]


def run_cell(k, seq, ops, args, eps_list, keys):
    """k=1,2,3 saddle: Jacobi vs Tensor (HX) vs Cheb-tensor (Chebyshev of the SAME
    HX). Tensor and Cheb-tensor share the inner L(k-1) inverse."""
    km1 = k - 1
    n_km1, n_k, n0 = int(getattr(seq, f"n{km1}")), int(getattr(seq, f"n{k}")), int(seq.n0)
    vs_upper = _nullspace_vectors(ops, k, False)
    nhk = int(jnp.asarray(vs_upper).shape[0])
    mass_upper = (lambda v: apply_mass_matrix(seq, ops, v, k, dirichlet=False)) if nhk > 0 else None
    if nhk == 0:
        vs_upper = None

    if k == 3:
        ap_jac = make_apply_routines_k3(seq, ops)            # L_3 saddle, S_3=0
    elif k == 2:
        ap_jac = make_apply_routines_k2(seq, ops, grad_project=True, atom="block_fd")
    else:
        ap_jac = make_apply_routines(seq, ops, pa_mode="block_fd", grad_project=True,
                                     dirichlet_flag=False)
    s_hat = ap_jac["a_matvec"]

    def run(label, precond_upper):
        if precond_upper is None:
            print(f"{label:<46} (skipped)", flush=True)
            return
        rhs = jnp.stack([s_hat(jax.random.normal(kk, (n_k,), dtype=jnp.float64))
                         for kk in keys], axis=0)
        jax.block_until_ready(rhs)
        solve = make_saddle_solve(
            ap_jac["stiffness_matvec"], ap_jac["derivative_matvec"],
            ap_jac["derivative_t_matvec"], ap_jac["mass_lower_matvec"],
            precond_upper, ap_jac["lower_tensor_precond"],
            n_upper=n_k, n_lower=n_km1, tol=args.cg_tol, maxiter=args.cg_maxiter,
            vs_upper=vs_upper, mass_upper_matvec=mass_upper)
        st = time_saddle_solve(solve, rhs, rel_tol=1e-9)
        print(f"{label:<46} {st['avg_iters']:>8.1f} {st['max_iters']:>7d} "
              f"{st['avg_ms']:>9.1f} {st['max_residual']:>11.2e} "
              f"{st['n_fail']:>7d}/{st['n_total']:<d}", flush=True)

    hdr = (f"{'k=%d (matrix-free)' % k:<46} {'avg_it':>8} {'max_it':>7} "
           f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}")
    print(f"\n========== k={k}  n_{km1}={n_km1} n_{k}={n_k}  saddle-harm-deflated={nhk} ==========")
    print(hdr)
    print("-" * len(hdr), flush=True)
    run("Jacobi", ap_jac["jacobi_diag"])
    for eps in eps_list:
        print(f"  -- eps={eps:.0e} --", flush=True)
        hx = _make_tensor_hx(seq, ops, k, eps, n0)            # Tensor = HX
        run(f"Tensor (HX, eps={eps:.0e})", hx)
        if not args.no_cheb_tensor:
            run(f"Cheb-tensor (eps={eps:.0e})",                # Chebyshev of the SAME HX
                make_cheb_accel(seq, ops, k, s_hat, hx, eps, False))


def run_cell_k0(seq, ops, args, eps_list, keys):
    """k=0 (condensed scalar CG, free BC; constant nullspace deflated): Jacobi vs
    Tensor (scalar tensor-Hodge) vs Cheb-tensor (Chebyshev of the tensor-Hodge)."""
    DBC = False
    n = int(seq.n0)
    vs = _nullspace_vectors(ops, 0, DBC)

    def a_matvec(v):
        return apply_stiffness(seq, ops, v, 0, dirichlet=DBC)      # L_0 = G_0^T M_1 G_0

    def mass_matvec(v):
        return apply_mass_matrix(seq, ops, v, 0, dirichlet=DBC)

    rhs = jnp.stack([a_matvec(jax.random.normal(kk, (n,), dtype=jnp.float64))
                     for kk in keys], axis=0)
    jax.block_until_ready(rhs)
    l0_diaginv = _invert_diagonal(_diagonal_from_matvec(a_matvec, n))

    def jacobi(v):
        return l0_diaginv * v

    def tensor(v):
        return apply_laplacian_preconditioner(seq, ops, v, 0, dirichlet=DBC, kind="tensor")

    hdr = (f"{'k=0 (condensed CG, free)':<46} {'avg_it':>8} {'max_it':>7} "
           f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}")
    print(f"\n========== k=0  n_0={n}  (condensed CG, const nullspace deflated) ==========")
    print(hdr)
    print("-" * len(hdr), flush=True)

    def run(label, precond):
        if precond is None:
            print(f"{label:<46} (skipped)", flush=True)
            return

        @jax.jit
        def solve(r, p=precond):
            x, info = solve_singular_cg(
                a_matvec, r, mass_matvec=mass_matvec, precond_matvec=p, vs=vs,
                tol=args.cg_tol, maxiter=args.cg_maxiter)
            rel = jnp.linalg.norm(a_matvec(x) - r) / jnp.maximum(jnp.linalg.norm(r), 1e-30)
            return x, info, rel

        st = time_solve(solve, rhs, rel_tol=1e-9)
        print(f"{label:<46} {st['avg_iters']:>8.1f} {st['max_iters']:>7d} "
              f"{st['avg_ms']:>9.1f} {st['max_residual']:>11.2e} "
              f"{st['n_fail']:>7d}/{st['n_total']:<d}", flush=True)

    run("Jacobi", jacobi)
    run("Tensor (tensor-Hodge)", tensor)        # eps-independent (the atom)
    if not args.no_cheb_tensor:
        for eps in eps_list:
            run(f"Cheb-tensor (eps={eps:.0e})",
                make_cheb_accel(seq, ops, 0, a_matvec, tensor, eps, DBC))


def run_cell_mass(seq, ops, args, keys, ks):
    """Benchmark the assembled tensor MASS preconditioners M_k^{-1} for k in `ks`
    (free BC): plain CG on M_k x = b, Jacobi (1/diag M_k) vs Tensor mass precond.
    M_k is SPD nonsingular (no nullspace) -> vs=[]. Free since the tensor mass
    preconditioners are already assembled in the shared build."""
    DBC = False
    print(f"\n========== mass tensors  M_k (free BC), k={tuple(ks)} ==========")
    hdr = (f"{'mass M_k solve':<46} {'avg_it':>8} {'max_it':>7} "
           f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}")
    print(hdr)
    print("-" * len(hdr), flush=True)

    for k in ks:
        n = int(getattr(seq, f"n{k}"))

        def m_matvec(v, _k=k):
            return apply_mass_matrix(seq, ops, v, _k, dirichlet=DBC)

        m_diaginv = _invert_diagonal(_diagonal_from_matvec(m_matvec, n))

        def jacobi(v, _d=m_diaginv):
            return _d * v

        def tensor(v, _k=k):
            return apply_mass_matrix_preconditioner(seq, ops, v, _k, dirichlet=DBC,
                                                    kind="tensor")

        rhs = jnp.stack([m_matvec(jax.random.normal(kk, (n,), dtype=jnp.float64))
                         for kk in keys], axis=0)
        jax.block_until_ready(rhs)

        def run(label, precond):
            @jax.jit
            def solve(r, p=precond):
                x, info = solve_singular_cg(
                    m_matvec, r, mass_matvec=m_matvec, precond_matvec=p, vs=[],
                    tol=args.cg_tol, maxiter=args.cg_maxiter)
                rel = jnp.linalg.norm(m_matvec(x) - r) / jnp.maximum(
                    jnp.linalg.norm(r), 1e-30)
                return x, info, rel

            st = time_solve(solve, rhs, rel_tol=1e-9)
            print(f"{label:<46} {st['avg_iters']:>8.1f} {st['max_iters']:>7d} "
                  f"{st['avg_ms']:>9.1f} {st['max_residual']:>11.2e} "
                  f"{st['n_fail']:>7d}/{st['n_total']:<d}", flush=True)

        run(f"M_{k} Jacobi  (n_{k}={n})", jacobi)
        run(f"M_{k} Tensor  (n_{k}={n})", tensor)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--klevel", type=int, default=2, choices=(0, 1, 2, 3))
    ap.add_argument("--all-k", action="store_true",
                    help="ONE build, then run k=0,1,2,3 (they share the build)")
    ap.add_argument("--ks", type=str, default=None,
                    help="explicit comma list of k-levels (e.g. 0,1,2); overrides "
                         "--all-k/--klevel")
    ap.add_argument("--bench-mass", action="store_true",
                    help="also benchmark the assembled tensor mass preconditioners "
                         "M_k^{-1} for the selected k-levels")
    ap.add_argument("--no-cheb-tensor", action="store_true",
                    help="skip the 'outside' Chebyshev-of-HX (cheb-tensor) runs; "
                         "only Jacobi + Tensor (HX) are timed")
    ap.add_argument("--ns", type=str, default="6,12,4")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--eps", type=str, default="1e-1,1e-2")
    ap.add_argument("--geometry", type=str, default="rotating_ellipse")
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.2)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=600)
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]
    eps_list = [float(e) for e in args.eps.split(",")]
    if args.ks is not None:
        ks = tuple(int(v) for v in args.ks.split(","))
    elif args.all_k:
        ks = (0, 1, 2, 3)
    else:
        ks = (args.klevel,)
    # ONE build shared across all k (the ~300s build dominates and is identical;
    # klevel=3 assembles k=0/1/2/3 stiffness + Schur-Jacobi, so it covers every k).
    build_klevel = 3 if (max(ks) == 3) else 2

    dg.DIRICHLET = False   # production Laplacian BC is free
    print(f"[diag] ns={args.ns} p={args.p} free BC; ks={ks}; eps sweep={eps_list}",
          flush=True)

    t = time.perf_counter()
    seq = build_sequence(args)
    ops = assemble_operators(seq, rank=args.rank, klevel=build_klevel)
    info = seq._compute_nullspaces(dg.BETTI)
    ops = seq._require_operators()
    print(f"[diag] ONE build (klevel={build_klevel}) + robust-jacobi nullspaces in "
          f"{(time.perf_counter() - t) * 1e3:.1f} ms; info={info}", flush=True)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    if args.bench_mass:
        run_cell_mass(seq, ops, args, keys, ks)
    for k in ks:
        if k == 0:
            run_cell_k0(seq, ops, args, eps_list, keys)   # condensed CG, not saddle
        else:
            run_cell(k, seq, ops, args, eps_list, keys)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
