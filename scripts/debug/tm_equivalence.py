"""Equivalence check for ``mode_beta_correction`` (the ``tm`` knob).

``tg`` (``MRX_BJ_TANG_BC``) adds ``beta e e^T`` to the radial stiffness BEFORE
the fast diagonalisation.  ``tm`` (``MRX_BJ_TANG_MODE``) applies the same
penalty AFTER, as a rank-one Woodbury update per angular mode, so that beta can
depend on the mode.  With the mode factor pinned to 1
(``MRX_BJ_TANG_MODE_FLAT=1``) the two are the SAME operator -- the
fast-diagonalisation eigenvectors are M-orthonormal, so a rank-one update in
the mode basis pulls back to ``beta e e^T (x) M_t (x) M_z``, which is exactly
what ``tg`` adds to the Kronecker sum.

So this must print a relative difference at round-off.  It does not, today:
``tm`` diverges at k=1 and is a no-op at k=2 (see the natural-BC handoff §9),
and until this check passes those are construction errors, not results.

Dense, small, one geometry -- it is a wiring test, not a benchmark.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_jacobi_spectrum import (build_sequence, dense_from_apply,  # noqa: E402
                                   make_preconditioner)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid",
                    choices=("toroid", "rot-ellipse", "w7x"))
    ap.add_argument("--ns", default="6,12,6")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--strength", default="10")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} "
          f"strength={cli.strength}%", flush=True)

    for k in (int(v) for v in cli.ks.split(",")):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            os.environ["MRX_BJ_TANG_MODE_FLAT"] = "0"
            base, _ = make_preconditioner(seq, ops, k, dbc, "ibpd_r3")
            p_base = dense_from_apply(base, n)
            tg, _ = make_preconditioner(seq, ops, k, dbc,
                                        f"ibpd_r3_tg{cli.strength}")
            p_tg = dense_from_apply(tg, n)
            os.environ["MRX_BJ_TANG_MODE_FLAT"] = "1"
            tm, _ = make_preconditioner(seq, ops, k, dbc,
                                        f"ibpd_r3_tm{cli.strength}")
            p_tm = dense_from_apply(tm, n)
            os.environ["MRX_BJ_TANG_MODE_FLAT"] = "0"

            scale = np.abs(p_tg).max()
            d_tm = float(np.abs(p_tm - p_tg).max() / scale)
            # How much either knob moved P at all -- a knob that is a no-op
            # would otherwise "agree" with the baseline and look correct.
            m_tg = float(np.abs(p_tg - p_base).max() / scale)
            m_tm = float(np.abs(p_tm - p_base).max() / scale)
            verdict = ("MATCH" if d_tm < 1e-10 else
                       "tm IS A NO-OP" if m_tm < 1e-14 else "MISMATCH")
            print(f"k={k} dbc={dbc!s:>5} n={n:>5}  "
                  f"||tm-tg||/||tg|| {d_tm:9.2e}  "
                  f"moved by tg {m_tg:9.2e}  tm {m_tm:9.2e}   {verdict}",
                  flush=True)


if __name__ == "__main__":
    main()
