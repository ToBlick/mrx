"""Does the TRUE polar G form an exact sequence? (CPU, no mass assembly)

apply_incidence = E^T sp E is the "directly built G". On the polar sequence the
extraction E is not a 0/1 selection (E^T E != I), so apply_incidence is NOT the
true topological derivative and is NOT nilpotent there (G_1 G_0 != 0).

The true derivative is G = M^{-1} D (D = E^T(phi,phi')E, M = E^T(phi,phi)E).
Equivalently, on range(E), G = Gram_{k+1}^{-1} . apply_incidence, where
Gram = E^T E = e @ e^T is the (cheap) coefficient Gram of the extraction --
block-diagonal: identity in the bulk, small dense block at the polar axis. This
needs NO 3D mass assembly, so it runs on CPU at modest resolution.

This script checks that the true G is nilpotent on the polar sequence:
  ||G_1 G_0 v|| / ||G_0 v||   (curl . grad)
  ||G_2 G_1 v|| / ||G_1 v||   (div . curl)
for both BCs, and contrasts with the naive apply_incidence composition.

Run: python scripts/diag_true_G_exactness.py   (CPU is fine; ns=6,8,4)
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    assemble_incidence_operators,
    _ensure_extraction_operators,
    _mass_extraction,
    apply_incidence_matrix,
)

TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)
NS = (6, 8, 4)
P = 3


def build_dense(fn, n_in):
    cols = []
    for j in range(n_in):
        e = jnp.zeros((n_in,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(jax.device_get(fn(e))))
    return np.stack(cols, axis=1)


def main():
    t0 = time.perf_counter()
    seq = DeRhamSequence(NS, (P, P, P), 2 * P, TYPES, polar=True, betti_numbers=BETTI)
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map(eps=1.0 / 3.0, kappa=1.2, R0=1.0, nfp=3))
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1, 2))
    ops = _ensure_extraction_operators(seq, ops)
    print(f"[diag] polar seq ns={NS} p={P}; incidence+extraction built in "
          f"{time.perf_counter()-t0:.1f}s (NO mass assembly)")

    def ext_size(k, DBC):
        return int(getattr(seq, f"n{k}_dbc" if DBC else f"n{k}"))

    def gram_inv(k_out, DBC):
        """(E^T E)^{-1} for the extraction of space k_out, dense (small)."""
        e, e_T = _mass_extraction(ops, k_out, DBC)
        n = ext_size(k_out, DBC)
        G = build_dense(lambda u: e @ (e_T @ u), n)
        dev = np.linalg.norm(G - np.eye(n))
        return jnp.asarray(np.linalg.inv(G)), dev

    for DBC in (False, True):
        bc = "DBC" if DBC else "free"
        Ginv = {}
        devs = {}
        for k in (1, 2, 3):
            Ginv[k], devs[k] = gram_inv(k, DBC)

        def G_true(v, k):
            inc = apply_incidence_matrix(seq, ops, v, k,
                                         dirichlet_in=DBC, dirichlet_out=DBC)
            return Ginv[k + 1] @ inc

        def G_naive(v, k):
            return apply_incidence_matrix(seq, ops, v, k,
                                          dirichlet_in=DBC, dirichlet_out=DBC)

        key = jax.random.PRNGKey(0)
        worst_true_cg = worst_true_dc = 0.0
        worst_naive_cg = worst_naive_dc = 0.0
        for _ in range(6):
            key, k0, k1 = jax.random.split(key, 3)
            v0 = jax.random.normal(k0, (ext_size(0, DBC),), dtype=jnp.float64)
            v1 = jax.random.normal(k1, (ext_size(1, DBC),), dtype=jnp.float64)
            # true G
            g0 = G_true(v0, 0); g1 = G_true(g0, 1)
            c1 = G_true(v1, 1); c2 = G_true(c1, 2)
            worst_true_cg = max(worst_true_cg, float(jnp.linalg.norm(g1)) /
                                max(float(jnp.linalg.norm(g0)), 1e-300))
            worst_true_dc = max(worst_true_dc, float(jnp.linalg.norm(c2)) /
                                max(float(jnp.linalg.norm(c1)), 1e-300))
            # naive apply_incidence
            ng0 = G_naive(v0, 0); ng1 = G_naive(ng0, 1)
            nc1 = G_naive(v1, 1); nc2 = G_naive(nc1, 2)
            worst_naive_cg = max(worst_naive_cg, float(jnp.linalg.norm(ng1)) /
                                 max(float(jnp.linalg.norm(ng0)), 1e-300))
            worst_naive_dc = max(worst_naive_dc, float(jnp.linalg.norm(nc2)) /
                                 max(float(jnp.linalg.norm(nc1)), 1e-300))

        print(f"\n[{bc}] extraction Gram deviation ||E^T E - I||: "
              f"V1={devs[1]:.2e} V2={devs[2]:.2e} V3={devs[3]:.2e}")
        print(f"[{bc}] TRUE G = Gram^-1 . apply_incidence  (must be ~0):")
        print(f"      ||G_1 G_0|| / ||G_0||  (curl.grad) = {worst_true_cg:.3e}")
        print(f"      ||G_2 G_1|| / ||G_1||  (div.curl)  = {worst_true_dc:.3e}")
        print(f"[{bc}] naive apply_incidence composition  (nonzero on polar):")
        print(f"      curl.grad = {worst_naive_cg:.3e}   div.curl = {worst_naive_dc:.3e}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
