"""Where the harmonic 1-form loses its accuracy, and what recovers it.

``compute_nullspaces`` builds the k=1 free harmonic form by stripping the exact
part (a k=0 solve) and then the coexact part (a k=2 FREE solve) off a logical
seed.  On W7-X the Rayleigh quotient of the result degrades with resolution
(3.5e-7 at 8^3 -> 1.3e-3 at 16^3) while every other harmonic form stays at
1e-11 -- and the k=2 free solve is exactly the case whose preconditioner is
known to stall there.

This script:

1. re-runs the construction stage by stage with ``return_info``, so the two
   inner solves report their own iteration counts and convergence flags;
2. measures the Rayleigh quotient after each stage;
3. tries three cures -- iterative refinement of the projection (repeat the two
   strips on the result), a tighter inner tolerance, and an inverse-iteration
   polish seeded with the direct vector -- and reports cost as well as accuracy.

Usage:
    python scripts/debug/diag_nullspace_polish.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.nullspace import (  # noqa: E402
    _logical_constant_seed, compute_nullspaces, find_nullspace_vectors,
    get_nullspace)
from verify_block_jacobi import build_sequence  # noqa: E402


def rayleigh(seq, ops, v, k, dbc):
    lv = op.apply_hodge_laplacian(seq, ops, v, k, dirichlet=dbc)
    mv = op.apply_mass_matrix(seq, ops, v, k, dirichlet=dbc)
    return abs(float(v @ lv)) / float(v @ mv)


def rayleigh_approx(seq, ops, v, k, dbc):
    """Same quotient against the operator the SOLVES actually use.

    ``verify_block_jacobi`` iterates on ``apply_hodge_laplacian_approx`` (the
    Schur-inner mass inverse replaced by one raw_kron apply), so its deflation
    is only as good as the harmonic vector is a kernel vector OF THAT operator.
    A vector can be exactly harmonic for the true ``L_k`` and still leave a
    residual here."""
    lv = op.apply_hodge_laplacian_approx(seq, ops, v, k, dirichlet=dbc)
    mv = op.apply_mass_matrix(seq, ops, v, k, dirichlet=dbc)
    return abs(float(v @ lv)) / float(v @ mv)


def strip(seq, ops, v, tol=None, verbose=""):
    """One pass of the two projections that build the harmonic 1-form.

    Returns ``(v_out, info_leray, info_coexact)``; the infos are the raw solver
    codes (negative = converged, |info| = iterations).
    """
    # Stage 1, Leray: remove grad(q) with q from L_0 free (deflated by the
    # constants).  Inlined from DeRhamSequence.apply_leray_projection so the
    # solver info comes back.
    div_v = -seq.apply_derivative_matrix(
        v, 0, dirichlet_in=False, dirichlet_out=False, transpose=True,
        operators=ops)
    q, info0 = seq.apply_inverse_hodge_laplacian(
        div_v, 0, dirichlet=False, operators=ops, tol=tol, return_info=True)
    v = v + seq.apply_strong_grad(q, False, False)

    # Stage 2: remove the coexact part via L_2 free.
    curl_v_dual = seq.apply_derivative_matrix(
        v, 1, dirichlet_in=False, dirichlet_out=False, operators=ops)
    a, info2 = seq.apply_inverse_hodge_laplacian(
        curl_v_dual, 2, dirichlet=False, operators=ops, tol=tol,
        return_info=True)
    v = v - seq.apply_weak_curl(a, False, False)
    return v, info0, info2


def fmt_info(info):
    i = int(np.asarray(info))
    return f"{abs(i):>5}it {'y' if i < 0 else 'N'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--inner-tol", type=float, default=1e-6)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} "
          f"seq.tol={seq.tol:.1e} seq.maxiter={seq.maxiter}", flush=True)

    t0 = time.perf_counter()
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    t_direct = time.perf_counter() - t0
    v_ref = jnp.asarray(np.asarray(get_nullspace(ops, 1, False))[0])
    print(f"\ndirect route {t_direct:.1f}s   k=1 free rayleigh "
          f"{rayleigh(seq, ops, v_ref, 1, False):.3e}   (vs L_approx "
          f"{rayleigh_approx(seq, ops, v_ref, 1, False):.3e})", flush=True)
    for k, dbc in ((2, True), (3, True), (0, False)):
        vs = np.asarray(get_nullspace(ops, k, dbc))
        if vs.shape[0]:
            print(f"  control k={k} dbc={dbc}: "
                  f"{rayleigh(seq, ops, jnp.asarray(vs[0]), k, dbc):.3e}",
                  flush=True)

    # ---- stage-by-stage, with solver info --------------------------------
    print("\nstage-by-stage (info: iterations, y = converged)", flush=True)
    seed = _logical_constant_seed(seq, ops, 1, False, (0.0, 0.0, 1.0))
    print(f" seed                       rayleigh "
          f"{rayleigh(seq, ops, seed / seq.l2_norm(seed, 1, dirichlet=False), 1, False):.3e}",
          flush=True)
    v = seed
    for sweep in range(3):
        t0 = time.perf_counter()
        v, i0, i2 = strip(seq, ops, v)
        v = v / seq.l2_norm(v, 1, dirichlet=False)
        dt = time.perf_counter() - t0
        print(f" strip {sweep + 1}: L_0 {fmt_info(i0)}  L_2 {fmt_info(i2)} "
              f" {dt:6.1f}s  rayleigh {rayleigh(seq, ops, v, 1, False):.3e}"
              f"  (L_approx {rayleigh_approx(seq, ops, v, 1, False):.3e})",
              flush=True)

    # ---- cure 1b: CHEAP strips ------------------------------------------
    # If refinement is what buys the accuracy, the inner solves no longer have
    # to be accurate -- and the k=2 free one is the expensive half (it runs to
    # maxiter at seq.tol).  Same sweep, loose tolerance.
    for tol in (1e-6, 1e-8):
        v = seed
        total = 0.0
        for sweep in range(3):
            t0 = time.perf_counter()
            v, i0, i2 = strip(seq, ops, v, tol=tol)
            v = v / seq.l2_norm(v, 1, dirichlet=False)
            total += time.perf_counter() - t0
            print(f" tol={tol:.0e} strip {sweep + 1}: L_0 {fmt_info(i0)} "
                  f" L_2 {fmt_info(i2)}  cum {total:6.1f}s  rayleigh "
                  f"{rayleigh(seq, ops, v, 1, False):.3e}"
                  f"  (L_approx {rayleigh_approx(seq, ops, v, 1, False):.3e})",
                  flush=True)

    # ---- cure 2: tighter inner tolerance ---------------------------------
    for tol in (1e-14,):
        t0 = time.perf_counter()
        w, i0, i2 = strip(seq, ops, seed, tol=tol)
        w = w / seq.l2_norm(w, 1, dirichlet=False)
        print(f"\n tol={tol:.0e} single strip: L_0 {fmt_info(i0)} "
              f"L_2 {fmt_info(i2)}  {time.perf_counter() - t0:6.1f}s "
              f" rayleigh {rayleigh(seq, ops, w, 1, False):.3e}", flush=True)

    # ---- cure 3: inverse-iteration polish --------------------------------
    for inner_tol in (cli.inner_tol, 1e-8):
        t0 = time.perf_counter()
        vecs, info = find_nullspace_vectors(
            seq, ops, 1, 1, cli.eps, dirichlet=False, x0s=[v_ref],
            abs_tol=1e-30, inner_tol=inner_tol, maxiter=8)
        dt = time.perf_counter() - t0
        n_it, res, rq = info[0]
        print(f" inv-iter polish eps={cli.eps:.0e} inner_tol={inner_tol:.0e}: "
              f"{n_it} sweeps {dt:6.1f}s  ||Lv||={res:.3e} "
              f" rayleigh {rayleigh(seq, ops, vecs[0], 1, False):.3e}"
              f"  (L_approx {rayleigh_approx(seq, ops, vecs[0], 1, False):.3e})",
              flush=True)


if __name__ == "__main__":
    main()
