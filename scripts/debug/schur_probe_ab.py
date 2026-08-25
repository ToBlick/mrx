"""raw_kron vs the metric-lumped atom as the Schur-Jacobi probe: A/B.

THIS COMPARISON CAN ONLY BE RUN WHILE raw_kron EXISTS. Once it is deleted the
numbers below are unreproducible forever, so the OUTPUT is the deliverable here,
not the decision it informs -- the decision is already made. Commit the raw log.

Why the deletion forces this measurement. `assemble_schur_jacobi_preconditioner`
probes and STORES 1/diag(A_k) with A_k(x) = S_k x + D_{k-1} B_{k-1} D_{k-1}^T x,
where B_{k-1} is the schur.inner inverse. Today that inner is raw_kron and it is
the ONLY surviving probe mode. Delete raw_kron and there is no backing left, so
the Schur-Jacobi diagonal must be probed from the metric-lumped atom instead --
the jacobi baseline changes not by choice but because nothing else survives.
This measures exactly that forced switch.

Three measurements, in the order they can invalidate each other:

1. LIVENESS. The two probed diagonals must DIFFER. A swap that silently does
   not take effect passes every correctness check perfectly and means nothing,
   so this is asserted before anything else is believed.
2. CORRECTNESS. A converged solve is preconditioner-independent, so the two
   arms must reach the SAME solution. |dx|/|x| above solver tolerance means the
   change is wrong regardless of how the iteration counts look.
3. MERIT. Iteration counts, with outer='jacobi' so the probed diagonal is what
   preconditions the solve. Under outer='block' the atom is the upper-block
   inverse directly and the probe is never consulted -- measuring there would
   be measuring nothing.

A SECOND liveness trap, specific to the merit arm: `outer='jacobi'` calls
`_build_schur_outer_jacobi_diaginv` with `allow_stored_tensor_diaginv=True`, so
if a mode-matched diagonal has been preassembled BOTH arms silently reuse that
one stored vector and the iteration counts come out identical for a reason that
has nothing to do with the preconditioners. This script never assembles one, and
asserts none is present, so each arm probes fresh from its own schur.inner.
(`_build_schur_probe_apply` validates its `mode` token and then ignores it --
the actual backing is `saddle_preconditioner.schur.inner`, which is what the
arms vary.)

    python scripts/debug/schur_probe_ab.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    MassPreconditionerSpec, SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from verify_block_jacobi import build_sequence  # noqa: E402

ARMS = ("raw_kron", "block_jacobi")


def spec_for(inner_kind, outer_kind):
    return SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='block_jacobi'),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind=inner_kind),
            outer=MassPreconditionerSpec(kind=outer_kind),
        ),
    )


def run_geometry(geometry, ns, p, tol, maxiter, ks):
    seq, ops = build_sequence(geometry, ns, p, maxiter)
    print(f"\n[geom ] {geometry} ns={ns} p={p} tol={tol:g}", flush=True)

    for k in ks:
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            side = "dbc " if dbc else "free"

            # (1) LIVENESS -- probe each arm's Schur diagonal directly.
            diags = {}
            for kind in ARMS:
                apply_ = op._build_schur_apply_from_saddle_preconditioner(
                    seq, ops, k=k, dirichlet=dbc, eps=0.0,
                    saddle_preconditioner=spec_for(kind, 'none'))
                diags[kind] = np.asarray(op._diagonal_from_matvec(apply_, n))
            a, b = diags["raw_kron"], diags["block_jacobi"]
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
            if rel <= 1e-12:
                print(f"[diag ] k={k} {side} n={n:7d}  rel={rel:.3e}  "
                      "*** IDENTICAL: the swap is a no-op here, so nothing "
                      "below means anything ***", flush=True)
                continue
            print(f"[diag ] k={k} {side} n={n:7d}  "
                  f"|d_rk - d_atom|/|d_rk| = {rel:.3e}  LIVE", flush=True)

            # Guard the second liveness trap: a preassembled diagonal would be
            # shared by both arms and the merit numbers would be meaningless.
            for kind in ARMS:
                mode = op._coerce_schur_diag_mode(
                    MassPreconditionerSpec(kind='jacobi'), context='ab-guard')
                stored = op._get_schur_diaginv(ops, k, dbc, mode)
                assert stored is None, (
                    f"a Schur diagonal is already stored for k={k}, "
                    f"dirichlet={dbc}, mode={mode!r}; both arms would reuse it "
                    "and the merit comparison would be void")

            # (2)+(3) CORRECTNESS and MERIT.
            rhs = jax.random.normal(jax.random.PRNGKey(7 * k + int(dbc)), (n,))
            out = {}
            for kind in ARMS:
                x, info = op.apply_inverse_hodge_laplacian(
                    seq, ops, rhs, k, dirichlet=dbc, tol=tol,
                    maxiter=maxiter, return_info=True,
                    preconditioner=spec_for(kind, 'jacobi'))
                x.block_until_ready()
                iters = int(jnp.abs(jnp.asarray(info)))
                converged = int(jnp.asarray(info)) < 0
                # The TRUE residual, which stays meaningful when a solve does
                # not converge -- unlike |dx|/|x|, which then compares two
                # arbitrary partial iterates and says nothing about either.
                lx = op.apply_hodge_laplacian(seq, ops, x, k, dirichlet=dbc)
                res = float(np.linalg.norm(np.asarray(lx) - np.asarray(rhs))
                            / np.linalg.norm(np.asarray(rhs)))
                out[kind] = (np.asarray(x), iters, converged, res)
            xr, ir, okr, rr = out["raw_kron"]
            xb, ib, okb, rb = out["block_jacobi"]
            print(f"[solve] k={k} {side} raw_kron {ir:6d} it (conv {okr}) "
                  f"res {rr:.2e}   atom {ib:6d} it (conv {okb}) res {rb:.2e}",
                  flush=True)

            # A converged solve is preconditioner-independent, so the arms must
            # agree -- but ONLY converged solves carry that guarantee. If either
            # hit the iteration cap, |dx|/|x| is undefined as a correctness
            # check and reporting it as a disagreement is simply wrong.
            if okr and okb:
                dx = float(np.linalg.norm(xb - xr)
                           / max(np.linalg.norm(xr), 1e-300))
                verdict = "OK" if dx < 1e-6 else "*** ARMS DISAGREE ***"
                print(f"[check] k={k} {side} |dx|/|x| = {dx:.3e}  {verdict}",
                      flush=True)
            else:
                print(f"[check] k={k} {side} NOT CONVERGED (cap {maxiter}) -- "
                      "correctness undefined; rank the arms by residual above, "
                      "not by |dx|/|x|", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometries", default="toroid,w7x",
                    help="comma-separated; a single-geometry number is a much "
                         "weaker permanent record")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    ks = [int(v) for v in cli.ks.split(",")]

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    print(f"[sha  ] both arms measured at {sha}", flush=True)
    print("[note ] raw_kron is deleted after this run; these numbers cannot "
          "be regenerated.", flush=True)

    for geometry in cli.geometries.split(","):
        run_geometry(geometry.strip(), ns, cli.p, cli.tol, cli.maxiter, ks)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
