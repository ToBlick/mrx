"""Disentangle the two error sources in the projected k=2 preconditioner:
the div-div block ``P_A`` and the embedded ``L_1^{-1}`` atom inside ``P_B`` /
the curl-complement projector ``Pi_2``.

Two questions (user, 2026-06-21):
  Q1 "Find a config that is ONLY limited by the quality of the L_1 inverse in
      P_B -- then we know what to target."
  Q2 "I do not fully trust k=2's P_A yet."

Key realisation: the established 91-it config ``projected P_A(tensor)+P_B(L_1^+)``
ALREADY uses the real, scalable, matrix-free P_A
(``apply_stiffness_tensor_preconditioner(.,2)``); only the dense ``L_1^+`` atom is
non-scalable. So we isolate cleanly:

P_A variants (all materialised dense for analysis, but each MODELS a definite
scalable form):
  * ``tensor``        = apply_stiffness_tensor_preconditioner(.,2): the ACTUAL
                        scalable P_A (uncapped separable 1/lambda -> blows up on
                        curls, hence the projection).
  * ``tensor-capped`` = pinv(tensor FORWARD model, rcond): caps the SEPARABLE
                        model's small singular values -- a scalable capped tensor
                        preconditioner (the block_fd analog for div-div), NOT yet
                        built in prod. This is the scalable target if it works.
  * ``ideal``         = pinv(S_2, rcond): caps the TRUE div-div operator.
                        Idealised, NOT scalable; the best a capped P_A could do.

Atom family (controlled quality, P_A held fixed): a convex blend
  ``atom_t = (1-t) * L_1^+  +  t * P_full``,  t in [0,1]
between the exact dense pseudo-inverse ``L_1^+`` (t=0, the 91-it atom) and the
cheap matrix-free k=1 Hodge preconditioner ``P_full`` (t=1, kappa~60, the only
deployable atom today, which fails). Sweeping t traces iterations vs atom quality.

PART 1 (Q2, atom-independent): eig(P_A . S_2). S_2 = div-div kills curls
(S_2 . curl = 0) so those are EXACT-zero eigenvalues regardless of P_A -- the
NONZERO spread is purely P_A's quality on the co-exact part it owns. Plus the
curl-blowup magnitude ||P_A . curl|| (uncapped tensor should explode).

PART 2 (Q1): for each (P_A, t): k=2 saddle MINRES + Pi_2 idempotency +
kappa(atom_t . L_1). With P_A = ideal this is the "only limited by the atom"
curve = the target spec; comparing P_A = tensor / tensor-capped shows whether the
scalable P_A is trustworthy or shifts the cliff.

Free BC (k=2 L_2 nonsingular). Dense O(n^3) -> keep ns small. GPU/SLURM.
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

from mrx.operators import (  # noqa: E402
    apply_derivative_matrix,
    apply_incidence_matrix,
    apply_mass_matrix,
    apply_stiffness,
    apply_stiffness_tensor_preconditioner,
    apply_stiffness_tensor_forward_model,
    tensor_stiffness_model_available,
)
import diag_graddiv_subspace_preconditioner as dg  # noqa: E402
from diag_graddiv_subspace_preconditioner import (  # noqa: E402
    build_sequence,
    assemble_operators,
    make_apply_routines,
    make_apply_routines_k2,
    make_saddle_solve,
    time_saddle_solve,
)


def build_dense(fn, n_in, *, label=""):
    fj = jax.jit(fn)
    cols = []
    t0 = time.perf_counter()
    for j in range(n_in):
        e = jnp.zeros((n_in,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(jax.device_get(fj(e))))
    dt = (time.perf_counter() - t0) * 1e3
    if label:
        print(f"[dense] built {label:<30} ({n_in:>4} probes) in {dt:>8.1f} ms")
    return np.stack(cols, axis=1)


def nonzero_cond(P, S, rel=1e-10):
    """cond of P@S over its nonzero eigenvalues (curls are exact-zero in S=div-div)."""
    ev = np.sort(np.abs(np.real(np.linalg.eigvals(P @ S))))
    pos = ev[ev > rel * ev.max()]
    return pos.min(), pos.max(), pos.max() / pos.min(), int(np.sum(ev <= rel * ev.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="6,12,4")
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
    ap.add_argument("--rcond", type=float, default=1e-8)
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]

    DBC = False
    dg.DIRICHLET = DBC
    print(f"[diag] BC = free (k=2 L_2 nonsingular)")

    t0 = time.perf_counter()
    seq = build_sequence(args)
    ops = assemble_operators(seq, rank=args.rank, klevel=2)
    print(f"[diag] seq+ops built in {time.perf_counter()-t0:.1f}s")
    if not tensor_stiffness_model_available(ops, 2):
        raise SystemExit("k=2 tensor stiffness MODEL not assembled (needed for capped P_A)")

    n0, n1, n2 = int(seq.n0), int(seq.n1), int(seq.n2)
    n3 = int(seq.n3)
    print(f"[diag] sizes: n0={n0} n1={n1} n2={n2} n3={n3}")

    # ---- dense operators ----
    S2 = build_dense(lambda e: apply_stiffness(seq, ops, e, 2, dirichlet=DBC), n2, label="S_2 (div-div)")
    M1 = build_dense(lambda e: apply_mass_matrix(seq, ops, e, 1, dirichlet=DBC), n1, label="M_1")
    D1 = build_dense(lambda e: apply_derivative_matrix(
        seq, ops, e, 1, dirichlet_in=DBC, dirichlet_out=DBC), n1, label="D_1 (curl V1->V2)")
    K1 = build_dense(lambda e: apply_stiffness(seq, ops, e, 1, dirichlet=DBC), n1, label="K_1 (curl-curl)")
    M0 = build_dense(lambda e: apply_mass_matrix(seq, ops, e, 0, dirichlet=DBC), n0, label="M_0")
    D0 = build_dense(lambda e: apply_derivative_matrix(
        seq, ops, e, 0, dirichlet_in=DBC, dirichlet_out=DBC), n0, label="D_0 (grad V0->V1)")
    G1 = build_dense(lambda e: apply_incidence_matrix(
        seq, ops, e, 1, dirichlet_in=DBC, dirichlet_out=DBC), n1, label="G_1 (incidence)")

    L1 = 0.5 * ((K1 + D0 @ np.linalg.solve(M0, D0.T)) + (K1 + D0 @ np.linalg.solve(M0, D0.T)).T)
    S_true = 0.5 * ((S2 + D1 @ np.linalg.solve(M1, D1.T)) + (S2 + D1 @ np.linalg.solve(M1, D1.T)).T)
    print(f"[diag] cond(L_1)={np.linalg.cond(L1):.2e}  cond(S_true)={np.linalg.cond(S_true):.2e}")

    L1p = jnp.asarray(0.5 * (np.linalg.pinv(L1, rcond=args.rcond)
                             + np.linalg.pinv(L1, rcond=args.rcond).T))

    # cheap matrix-free k=1 Hodge preconditioner (the deployable atom), dense.
    ap_k1 = make_apply_routines(seq, ops, pa_mode="block_fd", grad_project=True, dirichlet_flag=DBC)
    Pfull = jnp.asarray(build_dense(ap_k1["projected_p_a_plus_p_b"], n1, label="P_full (k=1 precond atom)"))

    # ---- three P_A variants (dense apply closures), each modelling a scalable form ----
    PA_tensor_M = jnp.asarray(build_dense(
        lambda e: apply_stiffness_tensor_preconditioner(seq, ops, e, 2, dirichlet=DBC),
        n2, label="P_A tensor (scalable, uncapped)"))
    fwd = build_dense(
        lambda e: apply_stiffness_tensor_forward_model(seq, ops, e, 2, dirichlet=DBC),
        n2, label="tensor FORWARD model (for capped P_A)")
    PA_tcap = jnp.asarray(0.5 * (np.linalg.pinv(fwd, rcond=args.rcond)
                                 + np.linalg.pinv(fwd, rcond=args.rcond).T))
    PA_ideal = jnp.asarray(0.5 * (np.linalg.pinv(S2, rcond=args.rcond)
                                  + np.linalg.pinv(S2, rcond=args.rcond).T))
    PA = {
        "tensor (scalable)":        (lambda r: PA_tensor_M @ r),
        "tensor-capped (scalable)": (lambda r: PA_tcap @ r),
        "ideal pinv(S_2)":          (lambda r: PA_ideal @ r),
    }
    PA_dense = {"tensor (scalable)": np.asarray(PA_tensor_M),
                "tensor-capped (scalable)": np.asarray(PA_tcap),
                "ideal pinv(S_2)": np.asarray(PA_ideal)}

    # =================================================================
    # PART 1 (Q2): trust P_A -- conditioning on co-exact + curl blow-up.
    # =================================================================
    print("\n[P1] P_A quality, atom-INDEPENDENT  (S_2 kills curls -> nonzero eig = co-exact)")
    print(f"     {'P_A':<26} {'coexact eig[min,max]':<28} {'cond':>10} {'#zero':>6} {'||P_A.curl||/||curl||':>22}")
    # a unit curl direction: G_1 @ (random) lives in ker(S_2)=ran(G_1)
    rng = np.random.default_rng(0)
    curl = G1 @ rng.standard_normal(n1)
    curl = curl / max(np.linalg.norm(curl), 1e-30)
    print(f"     [check] ||S_2 . curl|| = {np.linalg.norm(S2 @ curl):.2e} (should be ~0)")
    for name, M in PA_dense.items():
        lo, hi, cond, nz = nonzero_cond(M, S2)
        blow = np.linalg.norm(M @ curl)  # curl norm is 1
        print(f"     {name:<26} [{lo:.3e},{hi:.3e}]      {cond:>10.3e} {nz:>6d} {blow:>22.3e}")

    # =================================================================
    # PART 2 (Q1): atom-quality sweep, P_A held fixed.
    # =================================================================
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    # build rhs from an atom-independent forward apply (use ideal-atom routines)
    ap0 = make_apply_routines_k2(seq, ops, grad_project=True, atom="custom",
                                 k1_inv_custom=(lambda r: L1p @ r))
    rhs_batch = jnp.stack([ap0["a_matvec"](jax.random.normal(k, (n2,), dtype=jnp.float64))
                           for k in keys], axis=0)
    jax.block_until_ready(rhs_batch)

    t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    print("\n[P2] atom blend  atom_t = (1-t) L_1^+ + t P_full   (t=0 exact, t=1 cheap kappa~60)")
    I2 = np.eye(n2)
    header = (f"     {'P_A':<26} {'t':>5} {'k(atom.L1)':>11} {'idemp':>9} "
              f"{'cond(P.Strue)':>13} {'avg_it':>8} {'max_res':>10} {'fails':>6}")
    for ti, t in enumerate(t_values):
        atom_mat = jnp.asarray((1.0 - t) * np.asarray(L1p) + t * np.asarray(Pfull))

        def atom(r, M=atom_mat):
            return M @ r
        # atom-only metrics (P_A-independent)
        ka_lo, ka_hi, ka_cond, _ = nonzero_cond(np.asarray(atom_mat), L1)
        apx = make_apply_routines_k2(seq, ops, grad_project=True, atom="custom",
                                     k1_inv_custom=atom)
        cpc = apx["curl_primal_complement"]
        cdc = apx["curl_dual_complement"]
        p_b = apx["p_b"]
        Pi = I2 - build_dense(cpc, n2)
        idemp = np.linalg.norm(Pi @ Pi - Pi) / max(np.linalg.norm(Pi), 1e-30)

        if ti == 0:
            print(header)
            print("     " + "-" * (len(header) - 5))
        for pa_name, pa in PA.items():
            def precond(r, pa=pa):
                return cpc(pa(cdc(r))) + p_b(r)
            # cond(precond . S_true)
            P_up = build_dense(precond, n2)
            _, _, cond_ps, _ = nonzero_cond(P_up, S_true)
            solve = make_saddle_solve(
                apx["stiffness_matvec"], apx["derivative_matvec"],
                apx["derivative_t_matvec"], apx["mass_lower_matvec"],
                precond, apx["lower_tensor_precond"],
                n_upper=n2, n_lower=n1, tol=args.cg_tol, maxiter=args.cg_maxiter)
            st = time_saddle_solve(solve, rhs_batch, rel_tol=1e-9)
            tag_t = f"{t:>5.2f}" if pa_name == list(PA)[0] else "     "
            print(f"     {pa_name:<26} {tag_t} {ka_cond:>11.3e} {idemp:>9.2e} "
                  f"{cond_ps:>13.3e} {st['avg_iters']:>8.1f} {st['max_residual']:>10.2e} "
                  f"{st['n_fail']:>4d}/{st['n_total']}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
