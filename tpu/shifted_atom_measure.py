"""How close the shifted atom is to an inverse of ``M + eps L``.

Both Leray solves and the velocity smoothing solve invert ``M + eps L`` and
are preconditioned with the mass atom, which is an inverse of the ``eps = 0``
end of that family. The recorded relative misses are 2.43 on the lower k=2
mass block and 1.88 on the upper k=3 Schur block, and 419 MINRES iterations
follow from them.

:func:`mrx.preconditioners.build_shifted_mass_laplace_atom` factorises the
whole family at once, using the shared 1-D masses that both existing atoms
already build. This measures whether it is actually closer, over the range of
``eps`` a step visits, against the two things it has to beat: the production
mass preconditioner and no preconditioner at all.

The miss is ``||P A x - x|| / ||x||`` on random probes, which is
backend-independent, so this runs anywhere and needs no accelerator. The
shifted atom is a BULK atom and does not treat the polar core, so its miss
carries the core rows; the production mass atom's does not, and that
asymmetry is the point of also reporting ``eps = 0``, where the two model the
same operator and any gap is the core alone.

Usage:
    python tpu/shifted_atom_measure.py --ns 8,16,8 --p 3 --k 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    """Command line for the miss measurement."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, default=2, choices=(0, 1, 2, 3))
    ap.add_argument("--eps", default="0,1e-3,1e-2,1e-1,1,10,100",
                    help="shifts to probe; a step's eta*dt lands inside this")
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--precision", default="float64", help="MRX_DTYPE")
    ap.add_argument("--iterations", action="store_true",
                    help="also solve (M + eps L) x = b with each candidate "
                         "and report the iteration count, which is what the "
                         "step time actually pays for")
    ap.add_argument("--tol", type=float, default=None,
                    help="CG tolerance; default is mrx.sqrt_eps()")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    return ap.parse_args()


def main() -> int:
    """Measure the relative miss of each candidate over a range of shifts."""
    cli = parse_args()
    os.environ["MRX_DTYPE"] = cli.precision

    import jax
    import jax.numpy as jnp
    import numpy as np

    import mrx
    from mrx.geometry import build_sequence
    from mrx.operators import (_mass_extraction, apply_mass_matrix,
                               apply_mass_matrix_preconditioner,
                               apply_stiffness)
    from mrx.preconditioners import (apply_shifted_mass_laplace_atom,
                                     build_shifted_mass_laplace_atom)

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    ops = seq.build_preconditioners()
    k = cli.k
    epss = [float(v) for v in cli.eps.split(",")]

    print("=" * 74)
    print(f"shifted atom vs the production mass atom, k={k}, "
          f"ns={ns} p={cli.p} {mrx.DTYPE}")
    print("=" * 74)

    atom = build_shifted_mass_laplace_atom(seq, k)
    e, e_T = _mass_extraction(seq, k, False)
    n_ext = int(seq.n(k, False))
    print(f"extracted DoFs {n_ext}, {len(atom[0])} components")

    def operator(x, eps):
        """``(M_k + eps L_k) x`` in the extracted space, as the solves see it."""
        return (apply_mass_matrix(seq, x, k, dirichlet=False)
                + eps * apply_stiffness(seq, x, k, dirichlet=False))

    def p_shifted(v, eps):
        """The new atom, sandwiched into the extracted space."""
        return e @ apply_shifted_mass_laplace_atom(atom, e_T @ v, eps)

    def p_mass(v, eps):
        """The production mass preconditioner, which ignores ``eps``."""
        del eps
        return apply_mass_matrix_preconditioner(seq, ops, v, k,
                                                dirichlet=False)

    key = jax.random.PRNGKey(0)
    probes = [jax.random.normal(jax.random.fold_in(key, i), (n_ext,),
                                dtype=mrx.DTYPE)
              for i in range(cli.probes)]

    def miss(precond, eps):
        """``||P A x - x|| / ||x||``, averaged over the probes."""
        vals = []
        for x in probes:
            px = precond(operator(x, eps), eps)
            vals.append(float(jnp.linalg.norm(px - x) / jnp.linalg.norm(x)))
        return float(np.mean(vals))

    candidates = (("shifted atom", p_shifted),
                  ("mass atom", p_mass),
                  ("none", lambda v, eps: v))

    result = {"ns": list(ns), "p": cli.p, "k": k, "dtype": str(mrx.DTYPE),
              "n_ext": n_ext, "rows": {}}

    print(f"\n{'eps':>10} {'shifted atom':>15} {'mass atom':>13} "
          f"{'none':>10} {'best':>15}")
    print("-" * 74)
    for eps in epss:
        row = {name: miss(fn, eps) for name, fn in candidates}
        best = min(row, key=row.get)
        print(f"{eps:>10.4g} {row['shifted atom']:>15.4f} "
              f"{row['mass atom']:>13.4f} {row['none']:>10.4f} {best:>15}")
        result["rows"][str(eps)] = row

    if cli.iterations:
        from mrx.solvers import preconditioned_cg  # noqa: PLC0415

        print("\nCG iterations to solve (M + eps L) x = b")
        print(f"{'eps':>10} {'shifted atom':>15} {'mass atom':>13} "
              f"{'none':>10}")
        print("-" * 74)
        b = probes[0]
        result["iterations"] = {}
        for eps in epss:
            row = {}
            for name, fn in candidates:
                _, info = preconditioned_cg(
                    lambda x, e=eps: operator(x, e), b,
                    M=(None if name == "none"
                       else (lambda v, f=fn, e=eps: f(v, e))),
                    tol=cli.tol, maxiter=2000)
                # ``-k`` converged in k, ``+k`` did not.
                its = int(info)
                row[name] = {"iterations": abs(its), "converged": its <= 0}
            fmt = (f"{row['shifted atom']['iterations']:>15d} "
                   f"{row['mass atom']['iterations']:>13d} "
                   f"{row['none']['iterations']:>10d}")
            print(f"{eps:>10.4g} {fmt}")
            result["iterations"][str(eps)] = row

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
