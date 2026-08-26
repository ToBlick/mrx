"""Build time, apply time and iteration counts of the production preconditioners.

The numbers a change to ``mrx/preconditioners.py`` or
``mrx/metric_lumping_laplacian.py`` has to be judged by:

* ``build``  -- ``seq.build_preconditioners()`` as a whole, then each
  ``MetricLumpingLaplacian`` / ``MetricLumpingMass`` on its own;
* ``apply``  -- microseconds per call after warm-up, chained so the GPU
  pipeline is measured rather than one dispatch;
* ``check``  -- the norm of one apply on a seeded vector, printed to full
  precision, so "numerically unchanged" is a diff and not a claim;
* ``iters``  -- CG iterations of every mass solve and of the k = 0 Poisson
  solve, MINRES iterations of the k = 1 Hodge-Laplacian solve, all through
  the production ``'auto'`` dispatch.

    SCRIPT=scripts/benchmark/precond_build_apply.py ARGS="--ns 8,16,8" \
        JOB_NAME=precond_bench bash slurm/run.sh
"""
from __future__ import annotations

import argparse
import json
import time

import mrx  # noqa: F401  (selects the working precision from MRX_DTYPE)

import jax
import jax.numpy as jnp
import numpy as np

import mrx.operators as op
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.metric_lumping_laplacian import MetricLumpingLaplacian, MetricLumpingMass
from mrx.nullspace import compute_nullspaces


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    if out is not None:
        jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def apply_us(pre, n, reps, seed):
    """Microseconds per chained apply after warm-up."""
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(n), dtype=mrx.DTYPE)
    for _ in range(5):
        v = pre.apply(v)
    jax.block_until_ready(v)
    t0 = time.perf_counter()
    for _ in range(reps):
        v = pre.apply(v)
    jax.block_until_ready(v)
    return 1e6 * (time.perf_counter() - t0) / reps


def checksum(pre, n, seed):
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(n), dtype=mrx.DTYPE)
    y = pre.apply(v)
    return float(jnp.linalg.norm(y)), float(jnp.sum(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--skip-k1", action="store_true",
                    help="skip the k=1 saddle solve (and the nullspaces)")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    ks = tuple(int(v) for v in cli.ks.split(","))
    p = cli.p
    out = {"ns": list(ns), "p": p, "dtype": str(mrx.DTYPE),
           "device": str(jax.devices()[0]), "tol": cli.tol}
    print(f"[setup] ns={ns} p={p} dtype={mrx.DTYPE} device={jax.devices()[0]} "
          f"mrx={mrx.__file__}", flush=True)

    seq = DeRhamSequence(ns, (p,) * 3, 2 * p,
                         ("clamped", "periodic", "periodic"), polar=True,
                         tol=cli.tol, maxiter=cli.maxiter,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))

    # ---- build ------------------------------------------------------------
    ops, t_build = timed(lambda: seq.build_preconditioners(ks=ks))
    out["build_all_s"] = t_build
    print(f"[build] seq.build_preconditioners: {t_build:.2f} s", flush=True)

    out["build_s"], out["mass_build_s"] = {}, {}
    for k in ks:
        for dbc in (False, True):
            tag = f"k{k}_{'dbc' if dbc else 'free'}"
            _, t = timed(lambda: MetricLumpingLaplacian(seq, ops, k, dbc))
            _, tm = timed(lambda: MetricLumpingMass(seq, ops, k, dbc))
            out["build_s"][tag], out["mass_build_s"][tag] = t, tm
            print(f"[build] {tag}: laplacian {t:.3f} s   mass {tm:.3f} s",
                  flush=True)

    # ---- apply ------------------------------------------------------------
    cache = getattr(seq, op.METRIC_LUMPING_CACHE_ATTR)
    mass_cache = seq._mass_metric_lumping_cache["factors"]
    out["apply_us"], out["mass_apply_us"] = {}, {}
    out["check"], out["mass_check"] = {}, {}
    for k in ks:
        for dbc in (False, True):
            tag = f"k{k}_{'dbc' if dbc else 'free'}"
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            pre, mpre = cache[(k, dbc)], mass_cache[(k, dbc)]
            out["apply_us"][tag] = apply_us(pre, n, cli.reps, 1)
            out["mass_apply_us"][tag] = apply_us(mpre, n, cli.reps, 2)
            out["check"][tag] = checksum(pre, n, 3)
            out["mass_check"][tag] = checksum(mpre, n, 4)
            print(f"[apply] {tag} n={n:7d}: laplacian {out['apply_us'][tag]:8.1f} us"
                  f"   mass {out['mass_apply_us'][tag]:8.1f} us   "
                  f"check {out['check'][tag][0]:.17g} / "
                  f"{out['mass_check'][tag][0]:.17g}", flush=True)

    # ---- iteration counts -------------------------------------------------
    out["mass_iters"] = {}
    for k in ks:
        for dbc in (False, True):
            tag = f"k{k}_{'dbc' if dbc else 'free'}"
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rhs = jnp.asarray(np.random.default_rng(10 + k).standard_normal(n),
                              dtype=mrx.DTYPE)
            (_, info), t = timed(lambda: op.apply_inverse_mass_matrix(
                seq, ops, rhs, k, dirichlet=dbc, tol=cli.tol,
                maxiter=cli.maxiter, return_info=True))
            out["mass_iters"][tag] = int(info)
            print(f"[iters] mass {tag}: {int(info):5d}  ({t:.2f} s)", flush=True)

    if not cli.skip_k1:
        ops, t = timed(lambda: compute_nullspaces(seq, ops))
        seq.set_operators(ops)
        print(f"[setup] nullspaces {t:.1f} s", flush=True)
    out["lap_iters"] = {}
    for k in ((0,) if cli.skip_k1 else (0, 1)):
        if k not in ks:
            continue
        for dbc in (False, True):
            tag = f"k{k}_{'dbc' if dbc else 'free'}"
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rhs = jnp.asarray(np.random.default_rng(20 + k).standard_normal(n),
                              dtype=mrx.DTYPE)
            (_, info), t = timed(lambda: seq.apply_inverse_hodge_laplacian(
                rhs, k, dirichlet=dbc, operators=ops, tol=cli.tol,
                maxiter=cli.maxiter, preconditioner='auto', return_info=True))
            out["lap_iters"][tag] = int(info)
            print(f"[iters] laplacian {tag}: {int(info):5d}  ({t:.2f} s)",
                  flush=True)

    print("[json] " + json.dumps(out), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
