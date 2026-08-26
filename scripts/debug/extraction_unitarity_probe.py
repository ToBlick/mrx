"""Is the polar extraction E left-unitary, i.e. E E^T = I?

That single property decides whether histopolation can be done in the FULL
tensor-product space and then restricted by one E apply -- the route Guclu and
Campos Pinto use for polar domains.  For ``Pi_polar = E . Pi_full`` to be a
PROJECTOR onto the polar space you need, for any polar function with
coefficients a: ``Pi_full`` returns its full coefficients ``E^T a`` (it is a
projector and the function already lies in the full space), and then
``E (E^T a) = a`` -- exactly ``E E^T = I``.

`apply_incidence_matrix`'s docstring asserts polar extractions are NOT unitary.
If that is right the composition is only an approximation near the axis; if it
is wrong, `_require_full_tensor_space` is simply too strict and the whole
histopolation route is a few lines.  Measured here rather than argued.

Also reported: E^T E, which is the projector onto the polar subspace inside the
full space (idempotent iff E E^T = I), and the same for the boundary-only
extraction, which should be a clean selection.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import rotating_ellipse_map  # noqa: E402


def dense(op, n_in):
    """Materialise a matrix-free operator by applying it to the identity."""
    return np.asarray(jax.vmap(lambda c: op @ c)(jnp.eye(n_in))).T


def report(tag, E, n_full, n_red):
    Ed = dense(E, n_full)                      # (n_red, n_full)
    EEt = Ed @ Ed.T
    Id = np.eye(n_red)
    off = np.abs(EEt - Id)
    print(f"\n[{tag}] E is ({n_red} x {n_full})")
    print(f"  ||E E^T - I||_max = {off.max():.3e}   "
          f"||E E^T - I||_F/sqrt(n) = {np.linalg.norm(EEt - Id)/np.sqrt(n_red):.3e}")
    bad = np.argwhere(off > 1e-12)
    print(f"  rows violating unitarity: {len(np.unique(bad[:, 0])) if len(bad) else 0}"
          f" of {n_red}")
    if len(bad):
        r = np.unique(bad[:, 0])[:6]
        print(f"  first offending rows {r.tolist()}  "
              f"diag there {np.diag(EEt)[r].round(6).tolist()}")
    # idempotency of the induced projector on the full space
    P = Ed.T @ Ed
    print(f"  ||P^2 - P||_max (P = E^T E) = {np.abs(P @ P - P).max():.3e}")
    # entries: is it a pure selection (all 0/1, one per row)?
    nz = np.count_nonzero(np.abs(Ed) > 1e-14, axis=1)
    vals = np.unique(np.round(Ed[np.abs(Ed) > 1e-14], 10))
    print(f"  nonzeros per row: min {nz.min()} max {nz.max()}   "
          f"distinct values: {len(vals)}"
          f"{'  (pure selection)' if nz.max() == 1 and set(np.abs(vals)) <= {1.0} else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default="6,8,6")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--ks", default="1,2,3")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    for polar in (True, False):
        print("=" * 72)
        print(f"polar={polar}  ns={ns} p={cli.p}")
        seq = DeRhamSequence(ns, (cli.p,) * 3, 2 * cli.p,
                             ("clamped", "periodic", "periodic"),
                             polar=polar, tol=1e-12, maxiter=100,
                             betti_numbers=(1, 1, 0, 0))
        seq.evaluate_1d()
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2))
        for k in [int(v) for v in cli.ks.split(",")]:
            basis_n = int(getattr(seq, f"basis_{k}").n)
            report(f"k={k} free", getattr(seq, f"e{k}"),
                   basis_n, int(getattr(seq, f"n{k}")))
            report(f"k={k} dbc ", getattr(seq, f"e{k}_dbc"),
                   basis_n, int(getattr(seq, f"n{k}_dbc")))


if __name__ == "__main__":
    main()
