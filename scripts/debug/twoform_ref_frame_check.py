"""Which reference-frame proxy is the correct input for a k=2 `frame='ref'` load?

`w7x_vacuum_bfield_project.py` builds the seam-free proxy as `DF^T B` and feeds
it to `load_grid_field(..., 2, frame='ref')`. But `projectors.load` documents
`frame='ref'` as "the coefficients of the k-form expanded directly in reference
coordinates ... no pullback is applied", and for a 2-form the pullback is
`B_phys = DF omega_ref / J`, i.e. `omega_ref = J DF^{-1} B`.

Those are different fields on a shaped map:

    J DF^{-1} B  =  omega_ref                 the reference coefficients
    DF^T B       =  (g/J) omega_ref           the coefficients times the 2-form
                                              mass weight

Both are periodic (DF and B pick up the same field-period rotation, which is
orthogonal and cancels), so BOTH kill the zeta=0 seam -- which is why a wrong
one would not show up as a seam but as a smooth systematic error.

Decisive test, no grid and no interpolation: `load(f, 2, frame='phys')` is the
trusted path (it applies the DF pullback internally), so assemble the same
physical field three ways and see which `frame='ref'` proxy reproduces it.

    python scripts/debug/twoform_ref_frame_check.py --geometry w7x
"""
from __future__ import annotations

import argparse
import os
import sys

import jax


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, _ = build_sequence(cli.geometry, ns, cli.p, 1000)
    F = seq.map

    def B_phys(x):
        """An arbitrary smooth PHYSICAL vector field, in Cartesian components."""
        y = F(x)
        return jnp.array([0.3 + 0.2 * y[1], -0.4 * y[0], 0.5 + 0.1 * y[2]])

    def proxy_dft(x):                      # DF^T B   -- what the script uses
        return jax.jacfwd(F)(x).T @ B_phys(x)

    def proxy_jdfinv(x):                   # J DF^-1 B -- the reference 2-form
        DF = jax.jacfwd(F)(x)
        return jnp.linalg.det(DF) * jnp.linalg.solve(DF, B_phys(x))

    ref = np.asarray(seq.load(B_phys, 2, frame='phys'))
    a = np.asarray(seq.load(proxy_dft, 2, frame='ref'))
    b = np.asarray(seq.load(proxy_jdfinv, 2, frame='ref'))

    def rel(u):
        return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))

    print(f"\n=== {cli.geometry} ns={ns} p={cli.p} ===", flush=True)
    print(f"  ||load(B, frame='phys')||           = {np.linalg.norm(ref):.6e}",
          flush=True)
    print(f"  DF^T B      via frame='ref':  rel err = {rel(a):.6e}"
          f"   ||.||={np.linalg.norm(a):.6e}", flush=True)
    print(f"  J DF^-1 B   via frame='ref':  rel err = {rel(b):.6e}"
          f"   ||.||={np.linalg.norm(b):.6e}", flush=True)
    winner = "J DF^-1 B" if rel(b) < rel(a) else "DF^T B"
    print(f"\n  -> {winner} reproduces the physical-frame load.", flush=True)
    # If the two differ by exactly the 2-form mass weight g/J, the ratio of
    # norms should be O(1) but the fields pointwise very different; report a
    # pointwise sample so the failure mode is legible rather than inferred.
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.15, 0.9, size=(4, 3))
    print("\n  pointwise (logical x):  DF^T B   vs   J DF^-1 B", flush=True)
    for x in xs:
        xj = jnp.asarray(x)
        pa = np.asarray(proxy_dft(xj))
        pb = np.asarray(proxy_jdfinv(xj))
        print(f"    x={np.round(x,3)}  {np.round(pa,4)}   {np.round(pb,4)}",
              flush=True)


if __name__ == "__main__":
    main()
