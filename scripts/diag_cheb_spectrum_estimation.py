"""How good is our Chebyshev hyperparameter auto-tuning?

The Chebyshev/Richardson upper preconditioners need the spectral interval
``[lmin, lmax]`` of the (preconditioned) approximate Schur ``P . S-hat``. We
estimate it with Lanczos (``_lanczos_extremal_eigs`` for the jacobi/diagonal
smoother; ``_lanczos_extremal_eigs_precond`` for the non-diagonal tensor
smoother). This script checks the estimate against the TRUE dense spectrum:

  * true [lmin, lmax]  = extremal eigenvalues of the dense ``P @ S-hat``
  * Lanczos estimate vs Lanczos steps (10, 20, 30, 50)
  * the DEFENSIVE FLOOR ``lmin_used = max(lmin_est, lmax_est * 1e-3)`` -- is it
    active, and does it match or clip the true lmin?
  * #eigs OUTSIDE the estimated interval: BELOW lmin (under-resolved -> the slow
    modes Chebyshev leaves behind) and ABOVE lmax (AMPLIFIED -> can diverge).

Run for k=1 (S-hat ~ L_1), both smoothers, both BCs. Dense O(n^3) -> small ns.
GPU/SLURM.
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
    _lanczos_extremal_eigs, _lanczos_extremal_eigs_precond,
)


def build_dense(fn, n_in, *, label=""):
    fj = jax.jit(fn)
    cols = []
    t0 = time.perf_counter()
    for j in range(n_in):
        e = jnp.zeros((n_in,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(jax.device_get(fj(e))))
    if label:
        print(f"[dense] {label:<28} ({n_in:>4} probes) {(time.perf_counter()-t0)*1e3:>8.1f} ms")
    return np.stack(cols, axis=1)


def report(name, true_ev, est_steps, floor_frac=1e-3):
    """true_ev = sorted positive true eigenvalues of P@Shat; est_steps = {steps: (lo,hi)}."""
    tlo, thi = float(true_ev.min()), float(true_ev.max())
    print(f"\n  [{name}] TRUE eig(P.S_hat): [{tlo:.4e}, {thi:.4e}]  cond={thi/tlo:.2f}  "
          f"(n_pos={true_ev.size})")
    print(f"      {'steps':>5} {'est_lmin':>11} {'est_lmax':>11} {'lmin_used':>11} "
          f"{'floor?':>6} {'lmax err':>9} {'lmin err':>9} {'#below':>7} {'#above':>7}")
    for steps, (elo, ehi) in est_steps.items():
        used = max(elo, ehi * floor_frac)
        floored = "YES" if used > elo + 1e-30 else "no"
        lmax_err = ehi / thi - 1.0
        lmin_err = used / tlo - 1.0
        n_below = int(np.sum(true_ev < used * (1 - 1e-9)))   # under-resolved by Cheby
        n_above = int(np.sum(true_ev > ehi * (1 + 1e-9)))    # AMPLIFIED (dangerous)
        print(f"      {steps:>5} {elo:>11.4e} {ehi:>11.4e} {used:>11.4e} "
              f"{floored:>6} {lmax_err:>+9.1e} {lmin_err:>+9.1e} {n_below:>7d} {n_above:>7d}")


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=2000)
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]

    t0 = time.perf_counter()
    seq = build_sequence(args)
    ops = assemble_operators(seq, rank=args.rank, klevel=1, both_bc=True)
    print(f"[diag] seq+ops built in {time.perf_counter()-t0:.1f}s")

    for DBC in (True, False):
        dg.DIRICHLET = DBC
        bc = "dbc" if DBC else "free"
        n1 = int(seq.n1_dbc if DBC else seq.n1)
        print(f"\n========== k=1 {bc} (n1={n1}) ==========")
        applies = make_apply_routines(seq, ops, pa_mode="block_fd",
                                      grad_project=True, dirichlet_flag=DBC)
        s_hat = applies["a_matvec"]                       # ~ L_1 : V1 -> V1*
        tensor_P = applies["projected_p_a_plus_p_b"]      # ~ L_1^{-1} : V1* -> V1
        schur_diaginv = applies["schur_diaginv"]

        # ---- TRUE spectra (dense) ----
        Shat = build_dense(s_hat, n1, label="S_hat (approx Schur)")
        PT = build_dense(tensor_P, n1, label="P_tensor (k=1 precond)")
        ev_tensor = np.sort(np.real(np.linalg.eigvals(PT @ Shat)))
        ev_tensor = ev_tensor[ev_tensor > 1e-12 * ev_tensor.max()]
        # jacobi smoother: P = diag(schur_diaginv); eig(D^{-1} Shat) = eig(D^{-1/2} Shat D^{-1/2})
        di = np.asarray(schur_diaginv)
        ev_jac = np.sort(np.real(np.linalg.eigvals(np.diag(di) @ Shat)))
        ev_jac = ev_jac[ev_jac > 1e-12 * ev_jac.max()]

        # ---- Lanczos estimates vs steps ----
        est_tensor = {s: _lanczos_extremal_eigs_precond(s_hat, tensor_P, n1, steps=s, seed=args.seed)
                      for s in (10, 20, 30, 50)}
        sqrt_di = jnp.sqrt(jnp.abs(jnp.asarray(di)))

        def m_sym(v):
            return sqrt_di * s_hat(sqrt_di * v)
        est_jac = {s: _lanczos_extremal_eigs(m_sym, n1, steps=s, seed=args.seed)
                   for s in (10, 20, 30, 50)}

        report("TENSOR smoother (used by cheb-tensor)", ev_tensor, est_tensor)
        report("JACOBI smoother (diagonal)", ev_jac, est_jac)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
