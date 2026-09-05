#!/usr/bin/env python3
"""Run one equilibrium per accelerator, to use all of them instead of one.

A single relaxation occupies one device, because the solve is a chain of
dependent matrix-free applies on ~10^4 DoFs with nothing to shard. A parameter
sweep does not have that problem: each equilibrium is independent, so the
devices can each take one. Nothing here is TPU-specific -- it needs only
``len(jax.devices()) >= 2``, so a multi-GPU node works the same way.

This needs no library change. The relaxation scan is already a pure function of
an equinox ``State``, so ``jax.pmap`` over a stacked batch of initial states is
the whole implementation. What has to be *checked* is that it is genuinely one
equilibrium per device and not N copies of one: the states differ, so the
answers must differ, and each must match what it would have been alone.

Measured on a ``v5litepod-4``: 3.99x over four chips, 5.8 s against 23.1 s run
one at a time, each member reproducing its sequential answer to 5.0e-05. That
factor is only visible if compilation is timed apart from execution -- both
forms compile once, but the serial loop amortises it over four calls while the
pmap pays it on its only one, so a single-shot comparison reads 2.77x. Each
form is therefore run twice here and the second is reported.

    python -u scripts/pmap_sweep.py --ns 8,16,8 --p 3 --steps 4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--spread", type=float, default=0.01,
                    help="fractional separation between the batch members; the "
                         "point is that they are different problems, so this "
                         "must be well above solver tolerance")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--out", default="pmap_sweep.json")
    return ap.parse_args()


def main() -> None:
    cli = parse_args()
    os.environ.setdefault("MRX_DTYPE", cli.precision)

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    import mrx
    from mrx.geometry import build_sequence
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import clebsch_potential_form, potential_two_form
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state

    devices = jax.devices()
    n_dev = len(devices)
    print(f"[env] mrx {mrx.DTYPE}, {n_dev} devices: "
          f"{[d.device_kind for d in devices]}")
    if n_dev < 2:
        print("[skip] pmap over one device proves nothing; "
              "this wants a multi-chip node")
        return

    ns = tuple(int(v) for v in cli.ns.split(","))
    scale = 0.064 / ns[0] ** 2
    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    seq.set_operators(compute_nullspaces(seq, ops))
    B0, _, _ = potential_two_form(seq, clebsch_potential_form(
        load_clebsch(cli.geometry)))
    print(f"[setup] ns={ns} p={cli.p}  {time.perf_counter() - t0:.1f}s")

    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                     velocity_smoothing_order=1, velocity_smoothing_scale=scale)

    # A sweep of n_dev nearby initial fields: same geometry, perturbed IC. Any
    # real sweep would vary a physical parameter instead; what matters for the
    # check is only that the members are distinguishable.
    factors = [1.0 + cli.spread * i for i in range(n_dev)]
    B0s = [B0 * f for f in factors]

    def scan_steps(state):
        def body(state, _):
            state = ts.relaxation_step(state)
            return eqx.tree_at(lambda s: s.B_n, state, state.B_nplus1), None
        out, _ = jax.lax.scan(body, state, None, length=cli.steps)
        return out

    # Compilation has to be timed separately or it swamps the answer: the
    # serial loop compiles once and amortises it over n_dev calls, the pmap
    # compiles once and pays it on the only call, so a single-shot comparison
    # measures the XLA compiler more than the chips. Each form is therefore
    # run twice, and the second run is the throughput number.
    def timed(fn, args):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out.B_n)
        t_cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out.B_n)
        return out, t_cold, time.perf_counter() - t0

    # --- one chip at a time, the reference -------------------------------
    seq_results = []
    t_serial_cold = t_serial = 0.0
    run_one = jax.jit(scan_steps)
    for f, b in zip(factors, B0s):
        out, cold, warm = timed(run_one, (initial_state(b, ts, 1.0),))
        t_serial_cold += cold
        t_serial += warm
        seq_results.append(np.asarray(out.B_n, dtype=float))
        print(f"[serial] factor {f:.3f}: ||B|| = "
              f"{np.linalg.norm(seq_results[-1]):.8e}  "
              f"({cold:.1f}s first, {warm:.1f}s warm)")
    print(f"[serial] {n_dev} equilibria, {cli.steps} steps each: "
          f"{t_serial:.1f}s warm, {t_serial_cold:.1f}s including compilation")

    # --- all chips at once ------------------------------------------------
    states = [initial_state(b, ts, 1.0) for b in B0s]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    run_all = jax.pmap(scan_steps)
    out, t_pmap_cold, t_pmap = timed(run_all, (stacked,))
    par_results = [np.asarray(out.B_n[i], dtype=float) for i in range(n_dev)]
    print(f"[pmap  ] {n_dev} equilibria on {n_dev} chips: {t_pmap:.1f}s warm, "
          f"{t_pmap_cold:.1f}s including compilation")

    # --- the two checks that make it a sweep and not four copies ----------
    agree = []
    for i, (a, b) in enumerate(zip(seq_results, par_results)):
        rel = np.linalg.norm(b - a) / np.linalg.norm(a)
        agree.append(rel)
        print(f"[check ] member {i}: pmap vs serial relative difference {rel:.3e}")

    distinct = []
    for i in range(1, n_dev):
        rel = (np.linalg.norm(par_results[i] - par_results[0])
               / np.linalg.norm(par_results[0]))
        distinct.append(rel)
        print(f"[check ] member {i} vs member 0: {rel:.3e} "
              "(must be large; equal would mean four copies of one problem)")

    tol = float(mrx.sqrt_eps())
    ok_agree = max(agree) < 100 * tol
    # Distinctness is judged against the agreement, not against tol: the bar is
    # that the members differ by far more than pmap perturbs any one of them.
    # An absolute bar of 100*tol would fail a deliberately small sweep spread,
    # which is a fault in the bar and not in the sweep.
    ok_distinct = min(distinct) > 100 * max(agree)
    print(f"\n[verdict] reproduces serial: {ok_agree} "
          f"(max {max(agree):.3e} vs bar {100 * tol:.3e})")
    print(f"[verdict] members distinct : {ok_distinct} "
          f"(min {min(distinct):.3e} vs 100x the agreement, "
          f"{100 * max(agree):.3e})")
    print(f"[verdict] throughput       : {t_serial / t_pmap:.2f}x "
          f"on {n_dev} chips ({t_serial_cold / t_pmap_cold:.2f}x if the "
          "one-off compilation is charged to it)")

    if os.path.dirname(cli.out):
        os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    with open(cli.out, "w") as fh:
        json.dump(dict(devices=n_dev, ns=list(ns), p=cli.p, steps=cli.steps,
                       dtype=str(mrx.DTYPE), t_serial=t_serial, t_pmap=t_pmap,
                       t_serial_cold=t_serial_cold, t_pmap_cold=t_pmap_cold,
                       speedup=t_serial / t_pmap,
                       speedup_with_compilation=t_serial_cold / t_pmap_cold,
                       agreement=agree,
                       distinctness=distinct, ok_agree=bool(ok_agree),
                       ok_distinct=bool(ok_distinct)), fh, indent=2)
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
