#!/usr/bin/env python3
"""Time the matrix-free applies three ways, because one way is not enough.

Roughly 99% of a relaxation step is operator applies, so the per-apply cost is
the only quantity that can explain one backend being slower than another. The
earlier benchmark timed those applies eagerly, and the resulting table does not
compose: summing its numbers into one MINRES iteration of the velocity
smoothing solve gives >= 7.84 ms, hence 27.4 s for that single solve, against
13.03 s measured for a whole step of nine solves.

The reason is that an eager call to something like ``apply_derivative_matrix``
dispatches five separate XLA programs from Python with a round-trip between
each, while inside the relaxation the entire step body is one jitted
``lax.scan`` that XLA fuses. So this measures each apply three ways:

    eager   the raw Python composition, what the old table reported
    jit     the same apply wrapped in one ``jax.jit``, one program
    scan    the apply iterated inside a jitted ``lax.scan``, which is the
            form the relaxation actually runs and the only one entitled to be
            multiplied by an iteration count

``scan`` normalises the carry each iteration, so it costs one extra vector norm
per apply -- a reduction over ~10^4 elements against a sum-factorised apply, so
a percent or two, and it is the price of not overflowing float32 over 50
repeated applies of an operator whose spectral radius is not 1.

What is timed is every apply that appears in a saddle MINRES iteration:

    apply_mass_matrix       M_k, and its E / mass_core / E^T decomposition
    apply_stiffness         K_k = G^T M_{k+1} G, never timed until now
    apply_derivative_matrix D_k = M_{k+1} G_k, forward and transpose
    the metric-lumped mass atom and the fast-diagonalisation Laplacian atom

``apply_laplacian`` is deliberately absent: at k>=1 it calls
``apply_inverse_mass_matrix``, so it is 20-27 CG iterations rather than an
apply, and listing it beside these overstates the cost of an apply by ~150x.

    python -u matvec_bench.py --ns 12,24,12 --p 3 --out matvec.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Callable, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--precision", default=None,
                    help="MRX_DTYPE; default leaves the environment alone")
    ap.add_argument("--matmul-precision", default=None,
                    help="override jax_default_matmul_precision AFTER mrx sets "
                         "it to 'highest'. On a TPU MXU 'highest' means float32 "
                         "emulated in 6 bf16 passes, 'high' 3 and 'default' 1, "
                         "so this is the knob that is a TPU-only tax")
    ap.add_argument("--map-batch-size", type=int, default=None,
                    help="mrx.MAP_BATCH_SIZE_INNER; 0 (the mrx default) is a "
                         "full vmap and is rejected by jax < 0.9, so the "
                         "singularity overlay on NYU Torch needs a positive "
                         "value. It batches the quadrature loop and changes "
                         "setup memory only, not any timing below")
    ap.add_argument("--scan-length", type=int, default=50,
                    help="applies per scan; the per-apply cost is the total/this")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dirichlet", action="store_true", default=True)
    ap.add_argument("--out", default="matvec.json")
    return ap.parse_args()


class Bench:
    """Collect timings as ``{name: {form: seconds}}`` and print as it goes."""

    def __init__(self, repeats: int):
        self.repeats = repeats
        self.rows: dict[str, dict] = {}

    def _time(self, fn: Callable, n_applies: int) -> tuple[float, float]:
        """Return (first call, best steady call) per apply, in seconds."""
        import jax
        times = []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            times.append((time.perf_counter() - t0) / n_applies)
        return times[0], min(times[1:]) if len(times) > 1 else float("nan")

    def record(self, name: str, form: str, fn: Callable, n_applies: int = 1,
               note: str = "") -> Optional[float]:
        import jax
        row = self.rows.setdefault(name, {"note": note})
        try:
            first, steady = self._time(fn, n_applies)
        except Exception as exc:                             # noqa: BLE001
            row[form] = None
            row[f"{form}_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {name:<34} {form:<6} FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            return None
        del jax
        row[form] = steady
        row[f"{form}_first"] = first
        print(f"  {name:<34} {form:<6} {steady * 1e3:9.4f} ms   {note}",
              flush=True)
        return steady


def main() -> None:
    cli = parse_args()
    if cli.precision:
        os.environ["MRX_DTYPE"] = cli.precision

    import jax
    import jax.numpy as jnp

    import mrx
    from mrx.geometry import build_sequence
    from mrx.operators import (_mass_extraction, apply_derivative_matrix,
                               apply_laplacian_preconditioner,
                               apply_mass_matrix,
                               apply_mass_matrix_preconditioner,
                               apply_stiffness, mass_core_apply)

    if cli.matmul_precision:
        # mrx.precision pins 'highest' at import for a GPU reason (TF32 made
        # the W7-X map's dR/dtheta 19% wrong). Overriding after the import is
        # the only way to A/B it without editing the library.
        jax.config.update("jax_default_matmul_precision", cli.matmul_precision)

    devices = jax.devices()
    backend = {
        "jax_version": jax.__version__,
        "platform": devices[0].platform,
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
        "matmul_precision": cli.matmul_precision or "highest (mrx default)",
        "host": platform.platform(),
    }
    print(f"[env] mrx {mrx.DTYPE} on {backend['device_kind']} "
          f"x{backend['device_count']}, matmul {backend['matmul_precision']}",
          flush=True)

    if cli.map_batch_size is not None:
        mrx.MAP_BATCH_SIZE_INNER = cli.map_batch_size

    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    ops = seq.build_preconditioners()
    print(f"[setup] ns={ns} p={cli.p}  {time.perf_counter() - t0:.1f}s",
          flush=True)

    dbc = bool(cli.dirichlet)
    key = jax.random.PRNGKey(0)

    def vec(k: int):
        return jax.random.normal(key, (seq.n(k, dbc),), dtype=mrx.DTYPE)

    bench = Bench(cli.repeats)
    N = cli.scan_length

    def scanned(f: Callable):
        """``f`` applied N times inside one jitted scan, carry normalised."""
        @jax.jit
        def run(v):
            def body(carry, _):
                y = f(carry)
                return y / (jnp.linalg.norm(y) + jnp.finfo(mrx.DTYPE).tiny), None
            out, _ = jax.lax.scan(body, v, None, length=N)
            return out
        return run

    def two_ways(name: str, f: Callable, v, note: str = "") -> None:
        """Eager and jit only, for a map whose output shape differs from its
        input: it cannot be a scan carry, so it is scanned as a round trip."""
        bench.record(name, "eager", lambda: f(v), note=note)
        f_jit = jax.jit(f)
        bench.record(name, "jit", lambda: f_jit(v))

    def three_ways(name: str, f: Callable, v, note: str = "") -> None:
        """Time one apply eagerly, under jit, and inside a scan."""
        two_ways(name, f, v, note=note)
        f_scan = scanned(f)
        bench.record(name, "scan", lambda: f_scan(v), n_applies=N)

    print("\n--- mass, M_k --------------------------------------------------",
          flush=True)
    for k in range(4):
        three_ways(f"apply_mass_matrix k={k}",
                   lambda v, _k=k: apply_mass_matrix(seq, v, _k, dirichlet=dbc),
                   vec(k), note=f"n={seq.n(k, dbc)}")

    print("\n--- mass, decomposed into E / core / E^T ------------------------",
          flush=True)
    for k in range(4):
        e, e_T = _mass_extraction(seq, k, dbc)
        core = mass_core_apply(seq, k)
        n_raw = e.shape[1]
        two_ways(f"  E^T k={k}", lambda v, _t=e_T: _t @ v, vec(k),
                 note=f"{seq.n(k, dbc)}->{n_raw}")
        raw = np.asarray(e_T @ vec(k))
        three_ways(f"  mass_core k={k}", lambda v, _c=core: _c(v),
                   jnp.asarray(raw), note=f"raw n={n_raw}")
        two_ways(f"  E k={k}", lambda v, _e=e: _e @ v,
                 jnp.asarray(np.asarray(core(jnp.asarray(raw)))),
                 note=f"{n_raw}->{seq.n(k, dbc)}")
        # The pair is square, so it is the part of M_k a scan can carry -- but
        # E and E^T are both constants, so XLA is free to fold their product
        # out of the loop. Read the jit column for the extraction cost; the
        # scan column of this row is a lower bound, not the apply.
        three_ways(f"  E E^T k={k}", lambda v, _e=e, _t=e_T: _e @ (_t @ v),
                   vec(k), note="constant product, XLA may hoist it")

    print("\n--- stiffness, K_k = G^T M_{k+1} G -----------------------------",
          flush=True)
    for k in range(3):
        three_ways(f"apply_stiffness k={k}",
                   lambda v, _k=k: apply_stiffness(seq, v, _k, dirichlet=dbc),
                   vec(k), note=f"n={seq.n(k, dbc)}")

    print("\n--- weak derivative, D_k = M_{k+1} G_k -------------------------",
          flush=True)
    for k in range(3):
        # D and D^T change the space they act on, so neither can be a scan
        # carry alone; the pair below is what a saddle iteration applies.
        two_ways(f"apply_derivative k={k}",
                 lambda v, _k=k: apply_derivative_matrix(
                     seq, v, _k, dirichlet_in=dbc, dirichlet_out=dbc),
                 vec(k), note=f"{seq.n(k, dbc)}->{seq.n(k + 1, dbc)}")
        two_ways(f"apply_derivative^T k={k}",
                 lambda v, _k=k: apply_derivative_matrix(
                     seq, v, _k, dirichlet_in=dbc, dirichlet_out=dbc,
                     transpose=True),
                 vec(k + 1), note=f"{seq.n(k + 1, dbc)}->{seq.n(k, dbc)}")
        three_ways(
            f"apply_derivative D^T D k={k}",
            lambda v, _k=k: apply_derivative_matrix(
                seq, apply_derivative_matrix(
                    seq, v, _k, dirichlet_in=dbc, dirichlet_out=dbc),
                _k, dirichlet_in=dbc, dirichlet_out=dbc, transpose=True),
            vec(k), note="the pair, as a saddle iteration uses it")

    print("\n--- preconditioner applies -------------------------------------",
          flush=True)
    for k in range(4):
        three_ways(f"mass atom k={k}",
                   lambda v, _k=k: apply_mass_matrix_preconditioner(
                       seq, ops, v, _k, dirichlet=dbc),
                   vec(k), note="metric_lumping")
        three_ways(f"laplacian atom k={k}",
                   lambda v, _k=k: apply_laplacian_preconditioner(
                       seq, ops, v, _k, dirichlet=dbc, kind='metric_lumping'),
                   vec(k), note="fast diagonalisation")

    out = {"backend": backend, "mrx_dtype": str(mrx.DTYPE),
           "args": {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
                    "dirichlet": dbc, "scan_length": N},
           "n_dofs": {str(k): seq.n(k, dbc) for k in range(4)},
           "rows": bench.rows}
    if os.path.dirname(cli.out):
        os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    with open(cli.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {cli.out}", flush=True)

    print("\n=== eager / jit / scan, per apply in ms ========================",
          flush=True)
    print(f"{'operator':<34} {'eager':>10} {'jit':>10} {'scan':>10} "
          f"{'eager/scan':>11}")
    for name, row in bench.rows.items():
        e, j, s = row.get("eager"), row.get("jit"), row.get("scan")
        if e is None:
            continue
        js = f"{j * 1e3:10.4f}" if j else " " * 10
        if s is None:
            # A rectangular map: no scan form, so no eager/scan ratio.
            print(f"{name:<34} {e * 1e3:10.4f} {js} {'-':>10} {'-':>11}")
            continue
        print(f"{name:<34} {e * 1e3:10.4f} {js} {s * 1e3:10.4f} "
              f"{e / s:10.1f}x")


if __name__ == "__main__":
    main()
