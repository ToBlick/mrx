"""Does `mrx.poincare.logical_field` use the right pullback at k=1 and k=2?

The tracer integrates in LOGICAL coordinates, so it needs the logical
contravariant components of the physical vector field. Those differ by degree
and getting one wrong is invisible: the trace still runs, the surfaces still
look nested, and only iota is wrong.

`mrx.differential_forms.Pushforward` is the library's own trusted map to
physical space:

    k=1   F_* omega = (DF^T)^-1 omega
    k=2   F_* omega = DF omega / det DF

so the identity to check is that pushing MY logical vector forward as a plain
VECTOR (`DF v`, the k=-1 rule) reproduces it:

    k=2   DF @ logical_field(x) / det DF  ==  Pushforward(.,.,2)(x)
    k=1   DF @ logical_field(x)           ==  Pushforward(.,.,1)(x)

The k=2 case says the 2-form coefficients ARE the contravariant components, up
to the Jacobian that divides out of a direction. The k=1 case says
`g^-1 A` with `g = DF^T DF` is right, since
`DF (DF^T DF)^-1 A = (DF^T)^-1 A`.

Both are exact identities, so this is a machine-precision test, not a
tolerance-tuning exercise. It uses a RANDOM dof vector on purpose: the identity
is a property of the pullback, not of any particular field, and a random vector
exercises all three components where a harmonic form may be dominated by one.

    python scripts/debug/poincare_pullback_check.py --geometry w7x
"""
from __future__ import annotations

import argparse
import os
import sys

import jax


import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.differential_forms import DiscreteFunction, Pushforward  # noqa: E402
from mrx.poincare import logical_field  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--n-points", type=int, default=256)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, 10000)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    # Interior points: r=1 is the outer knot where det DF = 0 for a spline map,
    # and r=0 is the polar chart singularity.
    x = jax.random.uniform(jax.random.PRNGKey(5), (cli.n_points, 3))
    x = x.at[:, 0].multiply(0.9).at[:, 0].add(0.05)

    worst = 0.0
    for k, dbc in ((2, True), (1, False), (2, False), (1, True)):
        n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
        dof = jax.random.normal(jax.random.PRNGKey(17 * k + dbc), (n,))

        mine = logical_field(seq, dof, k, dbc)
        basis = seq.basis_2 if k == 2 else seq.basis_1
        extraction = getattr(seq, f"e{k}_dbc" if dbc else f"e{k}")
        ref = Pushforward(DiscreteFunction(dof, basis, extraction), seq.map, k)

        def compare(xi, _mine=mine, _ref=ref, _k=k):
            df = jax.jacfwd(seq.map)(xi)
            pushed = df @ _mine(xi)
            if _k == 2:
                pushed = pushed / jnp.linalg.det(df)
            want = _ref(xi)
            return (jnp.linalg.norm(pushed - want)
                    / jnp.linalg.norm(want))

        rel = jax.vmap(compare)(x)
        rmax, rmed = float(jnp.max(rel)), float(jnp.median(rel))
        worst = max(worst, rmax)
        side = "dbc " if dbc else "free"
        print(f"[k={k} {side}] n={n:7d}  max rel err {rmax:.3e}   "
              f"median {rmed:.3e}", flush=True)

    if worst > 1e-10:
        raise RuntimeError(
            f"pullback mismatch: worst relative error {worst:.3e}. These are "
            "exact identities, so anything above round-off is a wrong rule")
    print(f"[ok] worst {worst:.3e} -- both pullbacks correct", flush=True)


if __name__ == "__main__":
    main()
