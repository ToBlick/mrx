"""Verify the polar/extracted de Rham sequence is EXACT and commutes with
extraction -- the gate before any further k=2 preconditioner work.

The k=2 dense diagnostic found ||apply_stiffness(.,1) - G_1^T M_2 G_1|| = 2.25e-3
(but 1.9e-16 for k=2). apply_stiffness uses TWO extraction operators (middle
space stays full: e . spT . core_M . sp . eT); composing G^T M G via separate
applies inserts a middle projection P = e_{k+1}^T e_{k+1} (SIX extractions).
They are equal iff the full derivative of an extracted k-form already lies in the
extracted (k+1)-space -- i.e. iff the discrete de Rham complex COMMUTES with
extraction. That MUST hold for an exact FEEC sequence. This script checks it
directly, with no mass/stiffness involved, so any failure is purely topological
(incidence + extraction):

  (1) extraction is a partial isometry:  ||e_k e_k^T - I|| (so e_k^T e_k is an
      orthogonal projector onto the extracted subspace);
  (2) commuting / exactness of the COMPLEX with extraction:
      ||(I - e_{k+1}^T e_{k+1}) . sp_k . e_k^T||  -- the "leakage" of the full
      derivative of an extracted form out of the extracted next space. ZERO iff
      the sequence commutes with extraction (the 2-vs-6 extraction question).
  (3) nilpotency in extracted spaces: ||G_{k+1} G_k|| (= 0 always, topological).

Usage: python scripts/diag_derham_exactness.py --ns 6,12,4 --p 2 [--free|--dirichlet]
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
    build_sequence,
    assemble_operators,
)
from mrx.operators import (  # noqa: E402
    _incidence_components,
    _mass_extraction,
    apply_incidence_matrix,
)

DERIV = {0: "grad", 1: "curl", 2: "div"}


def op_leakage(apply_fn, n_in, *, n_trials=6, seed=0):
    """Max relative residual ||apply_fn(x)|| / ||x|| over random probes."""
    key = jax.random.PRNGKey(seed)
    worst = 0.0
    for t in range(n_trials):
        key, sk = jax.random.split(key)
        x = jax.random.normal(sk, (n_in,), dtype=jnp.float64)
        num = float(jnp.linalg.norm(apply_fn(x)))
        den = float(jnp.linalg.norm(x))
        worst = max(worst, num / max(den, 1e-300))
    return worst


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
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=2000)
    ap.add_argument("--dirichlet", dest="dirichlet", action="store_true")
    ap.add_argument("--free", dest="dirichlet", action="store_false")
    ap.set_defaults(dirichlet=False)
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]

    DBC = bool(args.dirichlet)
    dg.DIRICHLET = DBC

    t0 = time.perf_counter()
    seq = build_sequence(args)
    ops = assemble_operators(seq, rank=args.rank, klevel=2)
    print(f"[diag] BC={'DBC' if DBC else 'free'}; build {time.perf_counter()-t0:.1f}s")

    ext_size = {0: int(seq.n0_dbc if DBC else seq.n0),
                1: int(seq.n1_dbc if DBC else seq.n1),
                2: int(seq.n2_dbc if DBC else seq.n2),
                3: int(seq.n3_dbc if DBC else seq.n3)}

    # Build the PROLONGATION matrices E_k (full x ext): column j = e_k^T @ e_j,
    # i.e. the extracted basis expressed in full coefficients. E_k^T = e_k.
    print("\n[setup] building dense prolongation operators E_k (full x ext)")
    E = {}
    pinvE = {}
    for k in range(4):
        e, e_T = _mass_extraction(ops, k, DBC)
        nk = ext_size[k]
        cols = []
        for j in range(nk):
            ej = jnp.zeros((nk,), dtype=jnp.float64).at[j].set(1.0)
            cols.append(np.asarray(jax.device_get(e_T @ ej)))
        E[k] = np.stack(cols, axis=1)                 # (n_full_k, nk)
        pinvE[k] = np.linalg.pinv(E[k], rcond=1e-10)  # (nk, n_full_k) = (E^T E)^-1 E^T
        print(f"    E_{k}: full={E[k].shape[0]:>5}, ext={E[k].shape[1]:>5}, "
              f"||E_k^T E_k - I||={np.linalg.norm(E[k].T @ E[k] - np.eye(nk)):.2e} "
              f"(Gram dev; nonzero is fine, extraction not coeff-orthonormal)")

    # The DECISIVE exactness check: does the full derivative of an extracted
    # k-form land in the extracted (k+1)-space? Use the TRUE orthogonal projector
    # P_R = E (E^T E)^-1 E^T = E @ pinv(E). leak = ||(I - P_R) sp E_k v|| / ||.||.
    # 0 <=> the discrete de Rham complex commutes with extraction (the extracted
    # spaces form a subcomplex) <=> exactness preserved by polar extraction.
    print("\n[A] de Rham complex COMMUTES with extraction "
          "||(I - P_R) sp_k E_k v|| / ||sp_k E_k v||   (0 = exact subcomplex)")
    for k in (0, 1, 2):
        sp, sp_T = _incidence_components(ops, k)
        Ek, Ek1, pE1 = E[k], E[k + 1], pinvE[k + 1]
        key = jax.random.PRNGKey(1)
        worst = 0.0
        for t in range(8):
            key, sk = jax.random.split(key)
            v = np.asarray(jax.device_get(
                jax.random.normal(sk, (ext_size[k],), dtype=jnp.float64)))
            d = np.asarray(jax.device_get(sp @ jnp.asarray(Ek @ v)))   # full V_{k+1}
            d_proj = Ek1 @ (pE1 @ d)                                   # P_R d
            num = np.linalg.norm(d - d_proj)
            worst = max(worst, num / max(np.linalg.norm(d), 1e-300))
        print(f"    k={k} ({DERIV[k]:<4} V{k}->V{k+1}): leak = {worst:.3e}")

    # Full-space nilpotency sanity (topological, must be ~0 regardless).
    print("\n[B] full-space incidence nilpotency  ||sp_{k+1} sp_k x|| / ||sp_k x||")
    for k in (0, 1):
        sp0c, _ = _incidence_components(ops, k)
        sp1c, _ = _incidence_components(ops, k + 1)
        key = jax.random.PRNGKey(3)
        worst = 0.0
        for t in range(4):
            key, sk = jax.random.split(key)
            x = jax.random.normal(sk, (E[k].shape[0],), dtype=jnp.float64)
            a = sp0c @ x
            b = sp1c @ a
            worst = max(worst, float(jnp.linalg.norm(b)) /
                        max(float(jnp.linalg.norm(a)), 1e-300))
        print(f"    k={k}: {worst:.3e}")

    # The user's worry, done RIGHT: nilpotency of the TRUE extracted derivative
    # D_ext,k = pinv(E_{k+1}) sp_k E_k = (E^T E)^-1 E^T sp E (Gram-corrected).
    # apply_incidence_matrix is E_{k+1}^T sp E_k (NO (E^T E)^-1), so composing it
    # twice inserts a spurious E E^T Gram factor and looks non-nilpotent even for
    # a perfectly exact complex -- that is the 1.167 artifact. With the correct
    # derivative this MUST be ~0 if the complex is exact.
    print("\n[C] TRUE extracted-derivative nilpotency "
          "||D_{k+1} D_k v|| / ||D_k v||, D_k = pinv(E_{k+1}) sp_k E_k  (must be ~0)")
    for k in (0, 1):
        sp0c, _ = _incidence_components(ops, k)
        sp1c, _ = _incidence_components(ops, k + 1)
        key = jax.random.PRNGKey(4)
        worst = 0.0
        for t in range(6):
            key, sk = jax.random.split(key)
            v = np.asarray(jax.device_get(
                jax.random.normal(sk, (ext_size[k],), dtype=jnp.float64)))
            d0 = pinvE[k + 1] @ np.asarray(jax.device_get(sp0c @ jnp.asarray(E[k] @ v)))
            d1 = pinvE[k + 2] @ np.asarray(jax.device_get(sp1c @ jnp.asarray(E[k + 1] @ d0)))
            worst = max(worst, np.linalg.norm(d1) / max(np.linalg.norm(d0), 1e-300))
        comp = "div.curl" if k == 1 else "curl.grad"
        print(f"    k={k} ({comp}): {worst:.3e}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
