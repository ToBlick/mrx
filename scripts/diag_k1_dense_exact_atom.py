"""Dense diagnostics for the k=1 (grad-div) Hodge preconditioner -- the WORKING
baseline to compare the k=2 dense diagnostics against.

Mirrors scripts/diag_k2_dense_exact_atom.py one degree down. The k=1 projector is
Pi_g = G_0 L_0^{-1} G_0^T M_1 (onto the gradient subspace ran(G_0)); the inner
atom is the scalar k=0 Laplacian inverse L_0^{-1}. We measure, for the wrong
("directly built" apply_incidence = E^T sp E) curl/grad and the TRUE polar
derivative (G = Gram^{-1}.(E^T sp E)), and for both BCs:

  (A) operator consistency  ||apply_stiffness(.,0) - G_0^T M_1 G_0|| / ||.||
      (the k=1 analog of the k=2 2.25e-3 extraction gap; should vanish with true G)
  (B) projector idempotency ||Pi_g^2 - Pi_g|| / ||Pi_g|| for the tensor atom
      (production l0_inv) and the CG-matched atom (l0_inv_exact)
  (C) preconditioned spectrum eig(P_upper . S_true), S_true = exact k=1 Schur
  (D) k=1 saddle MINRES: jacobi vs projected P_A + P_B

Needs full operator assembly (mass etc.) -> run on GPU/SLURM.
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
jax.config.update("jax_enable_x64", True)

import diag_graddiv_subspace_preconditioner as dg  # noqa: E402
from diag_graddiv_subspace_preconditioner import (  # noqa: E402
    build_sequence, assemble_operators, make_apply_routines,
    make_saddle_solve, time_saddle_solve,
)
from mrx.operators import (  # noqa: E402
    apply_stiffness, apply_mass_matrix, apply_derivative_matrix,
    apply_incidence_matrix,
)


def build_dense(fn, n_in):
    cols = []
    for j in range(n_in):
        e = jnp.zeros((n_in,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(jax.device_get(fn(e))))
    return np.stack(cols, axis=1)


def spectrum_report(name, P, S):
    ev = np.sort(np.real(np.linalg.eigvals(P @ S)))
    pos = ev[ev > 1e-12 * ev.max()]
    band = np.mean((pos > 0.9) & (pos < 1.1)) * 100.0
    nz = int(np.sum(ev <= 1e-12 * ev.max()))
    print(f"{name:<40} eig[{pos.min():.3e},{pos.max():.3e}] cond={pos.max()/pos.min():.3e} "
          f"in[0.9,1.1]={band:5.1f}%  near0={nz}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="6,12,4")
    ap.add_argument("--p", type=int, default=3)
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
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]

    t0 = time.perf_counter()
    seq = build_sequence(args)
    ops = assemble_operators(seq, rank=args.rank, klevel=1, both_bc=True)
    print(f"[diag] seq+ops built in {time.perf_counter()-t0:.1f}s")

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)

    for DBC in (True, False):
        bc = "dbc" if DBC else "free"
        n1 = int(seq.n1_dbc if DBC else seq.n1)
        n0 = int(seq.n0_dbc if DBC else seq.n0)
        print(f"\n========== k=1 {bc} (n1={n1}, n0={n0}) ==========")

        # exact k=1 Schur S_true = K_1 + D_0 M_0^{-1} D_0^T (exact M_0^{-1})
        K1 = build_dense(lambda e: apply_stiffness(seq, ops, e, 1, dirichlet=DBC), n1)
        D0 = build_dense(lambda e: apply_derivative_matrix(
            seq, ops, e, 0, dirichlet_in=DBC, dirichlet_out=DBC, transpose=False), n0)
        M0 = build_dense(lambda e: apply_mass_matrix(seq, ops, e, 0, dirichlet=DBC), n0)
        S_true = K1 + D0 @ np.linalg.solve(M0, D0.T)
        S_true = 0.5 * (S_true + S_true.T)

        # Exact L_0^{-1} = pinv(apply_stiffness(.,0)); on the true-G core this is
        # the operator the gradient projector / P_B imply (= composed G_0^T M_1 G_0).
        S0 = build_dense(lambda e: apply_stiffness(seq, ops, e, 0, dirichlet=DBC), n0)
        L0_plus = jnp.asarray(0.5 * (np.linalg.pinv(S0, rcond=1e-10)
                                     + np.linalg.pinv(S0, rcond=1e-10).T))

        def l0_exact(x):  # exact dense L_0^{-1}
            return L0_plus @ x

        # tensor-atom (production) and exact-L_0-atom routines (the k=2-style swap).
        ap_T = make_apply_routines(seq, ops, pa_mode="block_fd", grad_project=True,
                                   dirichlet_flag=DBC)
        ap_E = make_apply_routines(seq, ops, pa_mode="block_fd", grad_project=True,
                                   dirichlet_flag=DBC, l0_inv_custom=l0_exact)

        # (A) consistency on the (now-fixed) core: apply_stiffness(.,0) vs composed.
        def composed_L0(v):
            g0v = apply_incidence_matrix(seq, ops, v, 0, dirichlet_in=DBC, dirichlet_out=DBC)
            m1 = apply_mass_matrix(seq, ops, g0v, 1, dirichlet=DBC)
            return apply_incidence_matrix(seq, ops, m1, 0,
                                          dirichlet_in=DBC, dirichlet_out=DBC, transpose=True)
        relA = np.linalg.norm(S0 - build_dense(composed_L0, n0)) / max(np.linalg.norm(S0), 1e-30)

        # (B) idempotency of Pi_g with tensor vs exact-L_0 atom.
        I1 = np.eye(n1)
        Pi_t = I1 - build_dense(ap_T["project_primal_complement"], n1)
        Pi_e = I1 - build_dense(ap_E["project_primal_complement"], n1)
        idem_t = np.linalg.norm(Pi_t @ Pi_t - Pi_t) / max(np.linalg.norm(Pi_t), 1e-30)
        idem_e = np.linalg.norm(Pi_e @ Pi_e - Pi_e) / max(np.linalg.norm(Pi_e), 1e-30)
        print(f"(A) ||apply_stiffness(.,0) - G_0^T M_1 G_0||/||.|| = {relA:.3e}")
        print(f"(B) idempotency ||Pi_g^2-Pi_g||/||Pi_g||: tensor L_0={idem_t:.3e}  "
              f"exact L_0={idem_e:.3e}")

        # (C) spectrum: jacobi, projected (tensor & exact L_0).
        print("(C) preconditioned spectrum eig(P_upper . S_true):")
        spectrum_report("    jacobi (diag)", build_dense(ap_T["jacobi_diag"], n1), S_true)
        spectrum_report("    projected P_A+P_B (tensor L_0)",
                        build_dense(ap_T["projected_p_a_plus_p_b"], n1), S_true)
        spectrum_report("    projected P_A+P_B (exact L_0)",
                        build_dense(ap_E["projected_p_a_plus_p_b"], n1), S_true)

        # (D) MINRES: raw vs projected, tensor vs exact-L_0 atom.
        rhs_batch = jnp.stack([ap_T["a_matvec"](
            jax.random.normal(k, (n1,), dtype=jnp.float64)) for k in keys], axis=0)
        jax.block_until_ready(rhs_batch)
        vs_upper = dg._nullspace_vectors(ops, 1, DBC)

        def mass_upper(v):
            return apply_mass_matrix(seq, ops, v, 1, dirichlet=DBC)
        print("(D) k=1 saddle MINRES (raw = no gradient-complement projection):")
        methods = [
            ("jacobi (diag)", ap_T["jacobi_diag"], None),
            ("projected P_A+P_B (exact L_0)",
             ap_E["projected_p_a_plus_p_b_with_state"], ap_E["p_a_state"]),
            ("raw P_A+P_B (exact L_0)",
             ap_E["p_a_plus_p_b_with_state"], ap_E["p_a_state"]),
            ("projected P_A+P_B (tensor L_0)",
             ap_T["projected_p_a_plus_p_b_with_state"], ap_T["p_a_state"]),
            ("raw P_A+P_B (tensor L_0)",
             ap_T["p_a_plus_p_b_with_state"], ap_T["p_a_state"]),
        ]
        header = (f"    {'upper precond':<34} {'avg_it':>8} {'max_it':>7} "
                  f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for name, pu, st in methods:
            solve = make_saddle_solve(
                ap_T["stiffness_matvec"], ap_T["derivative_matvec"],
                ap_T["derivative_t_matvec"], ap_T["mass_lower_matvec"],
                pu, ap_T["lower_tensor_precond"],
                n_upper=n1, n_lower=n0, tol=args.cg_tol, maxiter=args.cg_maxiter,
                precond_upper_state=st, vs_upper=(vs_upper if not DBC else None),
                mass_upper_matvec=(mass_upper if not DBC else None))
            stats = (time_saddle_solve(solve, rhs_batch, solve_state=st, rel_tol=1e-9)
                     if st is not None else
                     time_saddle_solve(solve, rhs_batch, rel_tol=1e-9))
            print(f"    {name:<34} {stats['avg_iters']:>8.1f} {stats['max_iters']:>7d} "
                  f"{stats['avg_ms']:>9.1f} {stats['max_residual']:>11.2e} "
                  f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
