"""S1: is the metric_lumping atom safe on the SHIFTED operator S_k + eps M_k?

`mrx/nullspace.py` pins the harmonic-form inverse iteration to jacobi in two
places, and the file says why rather than merely flagging it:

    "the shift is S_k + eps M_k, not L_k, so the atom's fit there wants
     measuring rather than assuming"

Production agrees at k=0 and disagrees with itself at k>=1:

  operators.py:3232   k=0 scalar path RAISES for kind='metric_lumping' when
                      eps != 0 -- "how the atom fits the shifted operator is
                      unmeasured"
  operators.py:~3629  k>=1 saddle path checks only that the atom is ASSEMBLED.
                      No eps guard at all.

So one concern is enforced at k=0 and silently permitted at k>=1. That
asymmetry is not a defensible resting state in either direction, and this
script produces the evidence that decides which way it resolves.

TWO PARTS.

PART 1 (k>=1): jacobi vs metric_lumping as schur.outer, through the real
inverse iteration. GATED ON FORM QUALITY, NOT ITERATION COUNT. The k=1 free
harmonic form is on record degrading with p -- 8.4e-13 at p=2, 3.0e-04 at p=3,
1.7e-01 at p=5 -- traced to this exact jacobi outer. Iteration count would show
a preconditioner working fine while the ANSWER is wrong at p=5.

PART 2 (k=0): measure the atom's fit to S_0 + eps M_0 DIRECTLY, without going
through the guarded path. Measuring around a guard is not the same as removing
one: this gives the guard's own stated objection the evidence it asks for, and
leaves the decision informed either way. Reported as the preconditioned
condition-number proxy -- how well P^-1 (S_0 + eps M_0) is clustered -- against
the same proxy for jacobi.

    python scripts/debug/nullspace_jacobi_ab.py --geometry toroid --ps 3,4,5
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback



import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.nullspace as ns  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    MassPreconditionerSpec, SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec, default_mass_preconditioner,
)
from verify_block_jacobi import build_sequence  # noqa: E402

EPS = 1e-4          # nullspace.py:569


def saddle_spec(outer_kind):
    return SaddlePointPreconditionerSpec(
        mass=default_mass_preconditioner(),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='metric_lumping'),
            outer=MassPreconditionerSpec(kind=outer_kind),
        ),
        coupled=False,
    )


def form_quality(seq, ops, v, k, dbc):
    """Rayleigh quotient against the generic scale, plus |dv|/|v| where defined."""
    r = ns.harmonic_rayleigh(seq, v, k, dirichlet=dbc, operators=ops)
    g = ns.generic_rayleigh(seq, k, dirichlet=dbc, operators=ops)
    try:
        d = ns.exact_derivative_residual(seq, v, k, dirichlet=dbc)
    except ValueError:
        d = float("nan")
    return r, g, (r / g if g else float("nan")), d


def part1(seq, ops, p, geometry):
    """k>=1 inverse iteration, jacobi vs metric_lumping outer."""
    cells = [(1, False), (1, True), (2, False), (2, True)]
    for k, dbc in cells:
        side = "dbc " if dbc else "free"
        out = {}
        for outer in ("jacobi", "metric_lumping"):
            try:
                # find_nullspace_vectors takes no preconditioner argument -- it
                # calls _nullspace_shifted_preconditioner internally, and
                # _validate_nullspace_shifted_preconditioner REJECTS any outer
                # that is not jacobi. So the arm is injected by substituting
                # the resolver, which also bypasses that validator. A probe may
                # measure around a guard; it must not remove one.
                original = ns._nullspace_shifted_preconditioner
                ns._nullspace_shifted_preconditioner = (
                    lambda _k, _spec=saddle_spec(outer): _spec)
                try:
                    vecs, iters = ns.find_nullspace_vectors(
                        seq, ops, k, n_vectors=1, eps=EPS, dirichlet=dbc)
                finally:
                    ns._nullspace_shifted_preconditioner = original
                if vecs.shape[0] == 0:
                    print(f"[p{p} k{k} {side}] {outer:15s} no harmonic vector",
                          flush=True)
                    continue
                r, g, ratio, dres = form_quality(seq, ops, vecs[0], k, dbc)
                out[outer] = ratio
                print(f"[p{p} k{k} {side}] {outer:15s} "
                      f"rayleigh/generic = {ratio:.3e}  |dv|/|v| = {dres:.3e}  "
                      f"iters = {iters}", flush=True)
            except Exception as exc:                          # noqa: BLE001
                print(f"[p{p} k{k} {side}] {outer:15s} *** {type(exc).__name__}"
                      f": {exc}", flush=True)
                traceback.print_exc()
        if len(out) == 2:
            j, m = out["jacobi"], out["metric_lumping"]
            # ABSOLUTE FLOOR BEFORE THE RATIO. Both arms routinely land at
            # 1e-23..1e-26, i.e. harmonic to machine precision, and a 10x ratio
            # between two numbers that are both zero is noise dressed as a
            # finding. The failure on record is 1.7e-01 at p=5 -- an ABSOLUTE
            # value. Read the absolute first, the band second.
            if max(j, m) < 1e-18:
                verdict = ("both harmonic to machine precision "
                           "-- ratio not meaningful")
            else:
                verdict = ("atom BETTER" if m < j / 2 else
                           "atom WORSE" if m > 2 * j else "comparable")
            print(f"[p{p} k{k} {side}] --> jacobi {j:.3e} vs atom {m:.3e}   "
                  f"{verdict}", flush=True)


def part2(seq, ops, p):
    """k=0: how well does the atom precondition S_0 + eps M_0?

    Around the guard, not through it. Estimates the preconditioned spectral
    spread by Rayleigh quotients of P^-1 A on random vectors -- a proxy, but
    the same proxy for both arms, which is what a comparison needs.
    """
    k, dbc = 0, False
    n = int(getattr(seq, "n0"))
    rng = np.random.default_rng(0)
    vs = [jnp.asarray(rng.standard_normal(n)) for _ in range(12)]

    def shifted(x):
        return (op.apply_stiffness(seq, ops, x, 0, dirichlet=dbc)
                + EPS * op.apply_mass_matrix(seq, ops, x, 0, dirichlet=dbc))

    arms = {}
    for kind in ("jacobi", "metric_lumping"):
        try:
            papply = op._build_scalar_hodge_preconditioner_apply(
                seq, ops, k=k, dirichlet=dbc, eps=0.0,
                preconditioner=MassPreconditionerSpec(kind=kind))
            qs = []
            for v in vs:
                av = shifted(v)
                pav = papply(av)
                qs.append(float(jnp.dot(v, pav) / jnp.dot(v, v)))
            lo, hi = min(qs), max(qs)
            arms[kind] = hi / lo if lo > 0 else float("inf")
            print(f"[p{p} k0 free] {kind:15s} P^-1 A quotient in "
                  f"[{lo:.3e}, {hi:.3e}]  spread = {arms[kind]:.2f}", flush=True)
        except Exception as exc:                              # noqa: BLE001
            print(f"[p{p} k0 free] {kind:15s} *** {type(exc).__name__}: {exc}",
                  flush=True)
    if len(arms) == 2:
        j, m = arms["jacobi"], arms["metric_lumping"]
        print(f"[p{p} k0 free] --> jacobi spread {j:.2f} vs atom {m:.2f}   "
              f"{'atom BETTER' if m < j else 'atom WORSE'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--ps", default="3,4,5")
    ap.add_argument("--skip-k0", action="store_true")
    cli = ap.parse_args()
    nsz = tuple(int(v) for v in cli.ns.split(","))

    print(f"[note ] eps = {EPS:g}, the shift nullspace.py:569 uses.", flush=True)
    print("[note ] GATE ON FORM QUALITY, NOT ITERATIONS: the failure on record "
          "is a wrong ANSWER at p=5, which an iteration count cannot see.",
          flush=True)
    for p in [int(v) for v in cli.ps.split(",")]:
        print(f"\n[geom ] {cli.geometry} ns={nsz} p={p}", flush=True)
        seq, ops = build_sequence(cli.geometry, nsz, p, 3000)
        if not cli.skip_k0:
            part2(seq, ops, p)
        part1(seq, ops, p, cli.geometry)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
