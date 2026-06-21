"""Dense exact-atom diagnostic for the k=2 (div-div) Hodge preconditioner.

The k=2 preconditioner ``projected P_A + P_B`` fails while the k=1 analogue
works. The leading hypothesis (see docs/hiptmair_xu_preconditioner.md and the
k2-laplacian-preconditioner memory) is that the *inner atom* is the wall:

* The curl-subspace correction and the curl-complement projector ``Pi_2`` both
  need an inner inverse ``k1_inv`` of the 1-form Laplacian ``L_1``.
* ``Pi_2 = G_1 k1_inv G_1^T M_2`` is idempotent only if ``k1_inv`` is the *exact*
  inverse; the gap is governed by the preconditioned condition number
  ``kappa(P . A)`` of the atom.
* In k=1 the atom is the scalar ``L_0^{-1}`` (tensor precond, kappa~2 -> single
  apply ~30% accurate -> Pi approximately idempotent -> works). Every available
  *vector*-Laplacian atom (block_fd curl-curl, or the k=1 vector preconditioner)
  is kappa~60 -> single apply ~97% wrong -> Pi badly non-idempotent -> fails.

This script tests that hypothesis directly by giving the k=2 construction an
**exact** ``L_1^{-1}`` -- materialised as a *dense Cholesky factor*. A dense
direct solve is a FIXED LINEAR operator (it applies the exact matrix, identically
for every input), so the outer Krylov solve stays linear: no inner Krylov, no
nonlinearity. This is what lets us probe "what if kappa(atom)=1" without
violating the no-Krylov-in-Krylov rule.

It runs three diagnostics at low resolution (dense O(n^3), so keep n small):

  (A) Projector idempotency ||Pi_2^2 - Pi_2|| / ||Pi_2|| for the exact atom vs
      the block_fd / full_l1 atoms -- direct test of the kappa->idempotency story.
  (B) Preconditioned spectrum eig(P_upper . S_true) where S_true is the exact
      k=2 Schur complement -- clustered near 1 => construction sound, atom was the
      wall; spread => the construction has a deeper flaw.
  (C) The k=2 saddle MINRES with the exact atom vs block_fd vs jacobi.

Usage (submitted to GPU via slurm; CPU py_compile locally):
    python scripts/diag_k2_dense_exact_atom.py --ns 4,8,4 --p 2
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

jax.config.update("jax_enable_x64", True)

from mrx.operators import (
    apply_derivative_matrix,
    apply_incidence_matrix,
    apply_mass_matrix,
    apply_stiffness,
)
# Reuse the proven build/assemble/k2 plumbing so the diagnostic exercises the
# IDENTICAL P_B / projector / P_A structure as the benchmark.
import diag_graddiv_subspace_preconditioner as dg  # noqa: E402
from diag_graddiv_subspace_preconditioner import (  # noqa: E402
    build_sequence,
    assemble_operators,
    make_apply_routines_k2,
    make_saddle_solve,
    time_saddle_solve,
)


def build_dense(fn, n_in, *, label=""):
    """Materialise the matrix of a linear map by probing unit vectors.

    Returns A (n_out x n_in) with A[:, j] = fn(e_j), so A @ x == fn(x). The fn is
    jitted once to avoid per-column retracing; columns are pulled to host as we
    go to keep device memory flat.
    """
    fj = jax.jit(fn)
    cols = []
    t0 = time.perf_counter()
    for j in range(n_in):
        e = jnp.zeros((n_in,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(jax.device_get(fj(e))))
    dt = (time.perf_counter() - t0) * 1e3
    if label:
        print(f"[dense] built {label:<28} ({n_in:>4} probes) in {dt:>8.1f} ms")
    return np.stack(cols, axis=1)


def spectrum_report(name, P, S):
    """Eigen-summary of the preconditioned operator P @ S (real SPD-similar)."""
    ev = np.linalg.eigvals(P @ S)
    ev = np.sort(np.real(ev))
    ev_pos = ev[ev > 1e-12 * ev.max()]
    lo, hi = ev_pos.min(), ev_pos.max()
    cond = hi / lo
    in_band = np.mean((ev_pos > 0.9) & (ev_pos < 1.1)) * 100.0
    n_tiny = int(np.sum(ev <= 1e-12 * ev.max()))
    print(f"{name:<34} eig[min,max]=[{lo:.3e},{hi:.3e}] cond={cond:.3e} "
          f"in[0.9,1.1]={in_band:5.1f}%  near-zero={n_tiny}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="4,8,4")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--geometry", type=str, default="rotating_ellipse")
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.2)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=2000)
    ap.add_argument("--skip-spectrum", action="store_true",
                    help="skip the dense eig analysis (build/MINRES only)")
    ap.add_argument("--dirichlet", dest="dirichlet", action="store_true",
                    help="use DBC (k=2 L_2 is SINGULAR with DBC -- 1 null vector)")
    ap.add_argument("--free", dest="dirichlet", action="store_false",
                    help="use free/no-DBC BC (k=2 L_2 is NONSINGULAR; default)")
    ap.set_defaults(dirichlet=False)
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]

    # k=2 Hodge duality: with BETTI=(1,1,0,0) the k=2 Laplacian is SINGULAR under
    # DBC (1 harmonic, = b_1 via the BC flip) and NONSINGULAR under free BC. Run
    # free so the projected-method failure is not polluted by an outer nullspace.
    DBC = bool(args.dirichlet)
    dg.DIRICHLET = DBC  # the k2 routine + assemble read this module global
    print(f"[diag] BC = {'DBC' if DBC else 'free/no-dbc'} "
          f"(k=2 L_2 {'SINGULAR' if DBC else 'nonsingular'} under this BC)")

    t_build = time.perf_counter()
    seq = build_sequence(args)  # evaluates 1d, sets map, assembles ref mass
    ops = assemble_operators(seq, rank=args.rank, klevel=2)
    print(f"[diag] sequence + operators built in "
          f"{(time.perf_counter() - t_build) * 1e3:.1f} ms")

    n0 = int(seq.n0_dbc if DBC else seq.n0)
    n1 = int(seq.n1_dbc if DBC else seq.n1)
    n2 = int(seq.n2_dbc if DBC else seq.n2)
    print(f"[diag] sizes: n0={n0}, n1={n1}, n2={n2}")

    # ------------------------------------------------------------------
    # Two exact dense atoms, both built by probing -> a fixed linear operator,
    # so the outer Krylov stays linear (no inner Krylov):
    #   * K_1^+  : pseudo-inverse of the curl-curl stiffness K_1=apply_stiffness(1)
    #   * L_1^+  : pseudo-inverse of the full Hodge L_1 = K_1 + D_0 M_0^{-1} D_0^T
    # Both are PSEUDO-inverses (pinv): K_1 is always singular (ker = gradients),
    # and the free-BC L_1 has the b_1 harmonic, so neither is Cholesky-able.
    #
    # ANALYTICAL EXPECTATION (per the user, and on reflection correct): on the
    # projector/P_B input b = G_1^T M_2 r, which satisfies G_0^T b = 0, the Hodge
    # decomposition gives L_1^+ b = K_1^+ b -- so the two atoms SHOULD behave
    # identically. We test both precisely to confirm that (any discrepancy would
    # flag an operator inconsistency, e.g. apply_derivative vs apply_incidence).
    # ------------------------------------------------------------------
    K1 = build_dense(lambda e: apply_stiffness(seq, ops, e, 1, dirichlet=DBC),
                     n1, label="K_1 (curl-curl)")
    M0 = build_dense(lambda e: apply_mass_matrix(seq, ops, e, 0, dirichlet=DBC),
                     n0, label="M_0")
    D0 = build_dense(
        lambda e: apply_derivative_matrix(
            seq, ops, e, 0, dirichlet_in=DBC, dirichlet_out=DBC, transpose=False),
        n0, label="D_0 (grad V0->V1)")
    L1 = K1 + D0 @ np.linalg.solve(M0, D0.T)
    L1 = 0.5 * (L1 + L1.T)  # symmetrise away probe round-off
    print(f"[diag] dense L_1 assembled: shape={L1.shape}, cond(L_1)={np.linalg.cond(L1):.3e}")

    L1_plus = np.linalg.pinv(L1, rcond=1e-8)
    L1_plus = 0.5 * (L1_plus + L1_plus.T)
    L1_plus_j = jnp.asarray(L1_plus)

    def l1_inv_exact(r):  # V1* -> V1, exact full Hodge L_1^+
        return L1_plus_j @ r

    cond_curl = np.linalg.cond(K1)  # nullspace-polluted; informational
    K1_plus = np.linalg.pinv(K1, rcond=1e-8)
    K1_plus = 0.5 * (K1_plus + K1_plus.T)
    K1_plus_j = jnp.asarray(K1_plus)
    print(f"[diag] dense K_1^+ built (pinv rcond=1e-8); cond(K_1)~{cond_curl:.3e} "
          "(nullspace-polluted)")

    def k1_pinv_exact(r):  # V1* -> V1, exact curl-curl pseudo-inverse
        return K1_plus_j @ r

    # The MATCHED atom: invert the operator the projector ACTUALLY composes,
    # A = G_1^T M_2 G_1, built from the SAME apply_incidence + apply_mass applies
    # (extraction-consistent). apply_stiffness(.,1) skips the middle V2 extraction
    # round-trip, so it differs from A by the V2 polar surgery (~2e-3 here) -- and
    # that mismatch, not atom conditioning, is what stops K_1^+/L_1^+ from making
    # Pi_2 = G_1 k1_inv G_1^T M_2 idempotent. pinv(A) inverts exactly what Pi_2
    # sandwiches, so Pi_2 should be idempotent to machine precision.
    G1_inc = build_dense(
        lambda e: apply_incidence_matrix(
            seq, ops, e, 1, dirichlet_in=DBC, dirichlet_out=DBC, transpose=False),
        n1, label="G_1 (incidence)")          # (n2, n1)
    M2 = build_dense(lambda e: apply_mass_matrix(seq, ops, e, 2, dirichlet=DBC),
                     n2, label="M_2")          # (n2, n2)
    A = G1_inc.T @ M2 @ G1_inc                 # (n1, n1) = composed curl-curl
    A = 0.5 * (A + A.T)
    rel_AK = np.linalg.norm(A - K1) / max(np.linalg.norm(K1), 1e-30)
    print(f"[diag] ||A=G_1^T M_2 G_1 - apply_stiffness(.,1)|| / ||.|| = {rel_AK:.3e} "
          "(extraction-round-trip gap)")
    A_plus = np.linalg.pinv(A, rcond=1e-8)
    A_plus = 0.5 * (A_plus + A_plus.T)
    A_plus_j = jnp.asarray(A_plus)

    def k1_matched(r):  # V1* -> V1, pinv of the projector's composed curl-curl
        return A_plus_j @ r

    # k=2 applies for each atom (identical structure, different inner inverse).
    ap_matched = make_apply_routines_k2(
        seq, ops, grad_project=True, atom="custom", k1_inv_custom=k1_matched)
    ap_pinv = make_apply_routines_k2(
        seq, ops, grad_project=True, atom="custom", k1_inv_custom=k1_pinv_exact)
    ap_exact = make_apply_routines_k2(
        seq, ops, grad_project=True, atom="custom", k1_inv_custom=l1_inv_exact)
    ap_bfd = make_apply_routines_k2(seq, ops, grad_project=True, atom="block_fd")
    ap_full = make_apply_routines_k2(seq, ops, grad_project=True, atom="full_l1")
    atoms = {"A^+ matched (G1^T M2 G1)": ap_matched,
             "K_1^+ (apply_stiffness)": ap_pinv,
             "L_1^+ (exact pinv)": ap_exact,
             "block_fd (curl-curl)": ap_bfd,
             "full_l1 (k=1 precond)": ap_full}

    # ------------------------------------------------------------------
    # (A) Projector idempotency: Pi_2 = I - curl_primal_complement.
    # ------------------------------------------------------------------
    print("\n[A] curl-complement projector idempotency "
          "||Pi_2^2 - Pi_2||_F / ||Pi_2||_F")
    I2 = np.eye(n2)
    for name, apx in atoms.items():
        C = build_dense(apx["curl_primal_complement"], n2)
        Pi = I2 - C
        rel = (np.linalg.norm(Pi @ Pi - Pi) /
               max(np.linalg.norm(Pi), 1e-30))
        print(f"    {name:<24} ||Pi^2-Pi||/||Pi|| = {rel:.3e}")

    # ------------------------------------------------------------------
    # (D) Operator consistency: is the assembled stiffness the incidence-Galerkin
    # product the projectors assume? Pi_2 = G_1 K_1^+ G_1^T M_2 is idempotent (with
    # an EXACT K_1^+) iff apply_stiffness(.,1) == G_1^T M_2 G_1 with the SAME
    # incidence G_1 and mass M_2. A nonzero residual here is the construction bug:
    # the atom inverts a different operator than the projector is built from.
    # ------------------------------------------------------------------
    print("\n[D] stiffness vs incidence-Galerkin consistency "
          "||S_k - G_k^T M_{k+1} G_k|| / ||S_k||")
    print(f"    k=1: {rel_AK:.3e}  (extraction-round-trip gap; the projector bug)")
    G2 = build_dense(
        lambda e: apply_incidence_matrix(
            seq, ops, e, 2, dirichlet_in=DBC, dirichlet_out=DBC, transpose=False),
        n2, label="G_2 (incidence)")
    M3 = build_dense(lambda e: apply_mass_matrix(seq, ops, e, 3, dirichlet=DBC),
                     G2.shape[0], label="M_3")
    S2k = build_dense(lambda e: apply_stiffness(seq, ops, e, 2, dirichlet=DBC),
                      n2, label="S_2")
    rel2 = np.linalg.norm(S2k - G2.T @ M3 @ G2) / max(np.linalg.norm(S2k), 1e-30)
    print(f"    k=2: {rel2:.3e}  (V3 middle: surgery trivial -> machine zero)")

    # ------------------------------------------------------------------
    # (B) Preconditioned spectrum eig(P_upper . S_true), S_true exact k=2 Schur.
    # ------------------------------------------------------------------
    if not args.skip_spectrum:
        S2 = build_dense(lambda e: apply_stiffness(seq, ops, e, 2, dirichlet=DBC),
                         n2, label="S_2 (div-div)")
        M1 = build_dense(lambda e: apply_mass_matrix(seq, ops, e, 1, dirichlet=DBC),
                         n1, label="M_1")
        D1 = build_dense(
            lambda e: apply_derivative_matrix(
                seq, ops, e, 1, dirichlet_in=DBC, dirichlet_out=DBC,
                transpose=False),
            n1, label="D_1 (curl V1->V2)")
        S_true = S2 + D1 @ np.linalg.solve(M1, D1.T)
        S_true = 0.5 * (S_true + S_true.T)
        print(f"[diag] dense exact k=2 Schur S_true: cond={np.linalg.cond(S_true):.3e}")

        print("\n[B] preconditioned spectrum eig(P_upper . S_true) "
              "(ideal: all near 1)")
        # Jacobi baseline (diag of approximate Schur).
        jac = ap_exact["jacobi_diag"]
        if jac is not None:
            P_jac = build_dense(jac, n2)
            spectrum_report("jacobi (diag)", P_jac, S_true)
        for name, apx in atoms.items():
            P_up = build_dense(apx["projected_p_a_plus_p_b"], n2)
            spectrum_report(f"projected P_A+P_B [{name}]", P_up, S_true)

    # ------------------------------------------------------------------
    # (C) k=2 saddle MINRES with the exact atom vs block_fd vs jacobi.
    # ------------------------------------------------------------------
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    rhs_batch = jnp.stack(
        [ap_exact["a_matvec"](jax.random.normal(k, (n2,), dtype=jnp.float64))
         for k in keys], axis=0)
    jax.block_until_ready(rhs_batch)

    # raw = NO curl-complement projection around P_A. k=2's tensor P_A blows up
    # on curls (1/lambda, lambda->0) so raw diverges and the projection is
    # mandatory. CAPPED P_A = pinv(S_2, rcond) zeros the curl modes (bounded,
    # like k=1's block_fd pinv) -- test whether a bounded P_A lets raw converge
    # and drop the projection. Idealized cap = exact div-div pseudo-inverse.
    jac_s = ap_exact["jacobi_stiff"]      # diag(S_2) only (zero on curls)
    jac_w = ap_exact["jacobi_diag"]       # whole-space Schur diag
    p_b_L1 = ap_exact["p_b"]              # curl-aux with exact L_1 atom (best)
    cpc = ap_exact["curl_primal_complement"]
    cdc = ap_exact["curl_dual_complement"]

    S2dd = build_dense(lambda e: apply_stiffness(seq, ops, e, 2, dirichlet=DBC),
                       n2, label="S_2 (div-div, for capped P_A)")
    cond_S2 = np.linalg.cond(S2dd)
    p_a_cap = {}
    for rc in (1e-8, 1e-6):
        S2p = np.linalg.pinv(S2dd, rcond=rc)
        S2p = jnp.asarray(0.5 * (S2p + S2p.T))
        p_a_cap[rc] = (lambda r, M=S2p: M @ r)
    print(f"[diag] capped P_A = pinv(S_2, rcond); cond(S_2)~{cond_S2:.2e}")

    def raw_(p_a):
        return lambda r: p_a(r) + p_b_L1(r)

    def proj_(p_a):
        return lambda r: cpc(p_a(cdc(r))) + p_b_L1(r)

    methods = {
        "jacobi (diag)": jac_w,
        # --- reference (tensor / exact-pinv P_A, the established results) ---
        "projected P_A(tensor)+P_B (L_1^+)": ap_exact["projected_p_a_plus_p_b"],
        "raw P_A(tensor)+P_B (L_1^+)": ap_exact["raw_p_a_plus_p_b"],
        # --- capped P_A: does a bounded div-div P_A let RAW converge? ---
        "raw P_A(cap 1e-8)+P_B (L_1^+)": raw_(p_a_cap[1e-8]),
        "raw P_A(cap 1e-6)+P_B (L_1^+)": raw_(p_a_cap[1e-6]),
        "projected P_A(cap 1e-8)+P_B (L_1^+)": proj_(p_a_cap[1e-8]),
        "P_A(cap 1e-8) only": p_a_cap[1e-8],
        # --- jacobi-split variants (no projection, zero-on-curls P_A) ---
        "jacobi(S)+P_B (L_1^+)": lambda r: jac_s(r) + p_b_L1(r),
        "jacobi(whole)+P_B (L_1^+)": lambda r: jac_w(r) + p_b_L1(r),
        # --- atom sensitivity ---
        "raw P_A(cap 1e-8)+P_B (K_1^+)": (lambda r, pa=p_a_cap[1e-8]:
                                          pa(r) + ap_pinv["p_b"](r)),
        "P_B only (L_1^+)": p_b_L1,
    }
    print("\n[C] k=2 saddle MINRES (upper varies, lower fixed=tensor mass M_1)")
    header = (f"{'upper precond':<34} {'avg_it':>8} {'max_it':>7} "
              f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}")
    print(header)
    print("-" * len(header))
    for name, precond_upper in methods.items():
        if precond_upper is None:
            continue
        solve = make_saddle_solve(
            ap_exact["stiffness_matvec"],
            ap_exact["derivative_matvec"],
            ap_exact["derivative_t_matvec"],
            ap_exact["mass_lower_matvec"],
            precond_upper,
            ap_exact["lower_tensor_precond"],
            n_upper=n2, n_lower=n1,
            tol=args.cg_tol, maxiter=args.cg_maxiter,
        )
        stats = time_saddle_solve(solve, rhs_batch, rel_tol=1e-9)
        print(f"{name:<34} {stats['avg_iters']:>8.1f} {stats['max_iters']:>7d} "
              f"{stats['avg_ms']:>9.1f} {stats['max_residual']:>11.2e} "
              f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
