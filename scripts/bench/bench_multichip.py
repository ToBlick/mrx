"""Relax several independent start fields on one multi-chip TPU host with JAX's sharding API (no ``jax.pmap``).

A ct5p-hightpu-4t (v5p) or ct6e-standard-4t (v6e) host has four chips and ``scripts/relax.py`` uses one of them.
This benchmark relaxes ``N`` start fields, the initial field of the geometry scaled by ``--factors``, and times
three ways of running them:

``serial``
    One chip, the fields one after another through the compiled chunk of ``relax.py``
    (``mrx.relaxation.loop._run_chunk``). This is the baseline.
``shard_map``
    ``jax.jit(jax.shard_map(f, mesh=mesh, in_specs=..., out_specs=P("eq")))`` over a 1-D mesh of the chips. The
    stacked fields are a global ``jax.Array`` with ``NamedSharding(mesh, P("eq"))`` and the stepper is replicated
    (``P()``). Every chip runs the single-field program on its own field, so its Krylov loops stop at their own
    iteration counts and its ``lax.cond`` take one branch. No collective is issued. This is what ``jax.pmap`` has
    lowered to since JAX 0.10, written with the public API that the pmap migration guide recommends.
``jit_vmap``
    ``jax.jit(jax.vmap(f, spmd_axis_name="eq"))`` with the batch axis of the fields sharded over an ``Auto`` mesh:
    global-view code partitioned by the compiler. Under ``vmap`` a ``while_loop`` runs until every field has
    converged (an all-reduce of the predicate per iteration) and a ``lax.cond`` whose predicate differs between
    fields evaluates both branches, so this is expected to be slower. It is measured, not assumed.

Every mode does the same work: ``--*-steps`` relaxation steps per field in compiled chunks, no diagnostics in
between. The first call of a mode includes its compilation (or the load from the persistent cache when
``JAX_COMPILATION_CACHE_DIR`` holds it), the second call is the steady state. The compile stages and the cache
hits of each first call are read from ``jax.monitoring`` (:class:`CompileMetrics`). The best residual and the final
field of each parallel run are compared with the serial run of the same field.

    MRX_DTYPE=float32 python -u scripts/bench/bench_multichip.py --geometry data/wout_li383_1.4m.nc \\
        --resolution 8 12 12 --out /gcs/outputs/bench/multichip

Output: ``<out>/multichip_results.json``, rewritten after every mode.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
import traceback

MODES = ("serial", "shard_map", "jit_vmap")
METHODS = ("gradient", "newton")
# A failed mode's record holds the last TRACEBACK_CHARS characters of its traceback, so the JSON stays small.
TRACEBACK_CHARS = 4000

# The jax.monitoring events of JAX 0.11 (jax/_src/dispatch.py, compiler.py, compilation_cache.py): every stage of a
# compile is a duration event, and every persistent-cache lookup a counter event.
TRACE = "/jax/core/compile/jaxpr_trace_duration"
LOWER = "/jax/core/compile/jaxpr_to_mlir_module_duration"
BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"
CACHE_HITS = "/jax/compilation_cache/cache_hits"
CACHE_MISSES = "/jax/compilation_cache/cache_misses"
CACHE_REQUESTS = "/jax/compilation_cache/compile_requests_use_cache"
CACHE_SAVED = "/jax/compilation_cache/compile_time_saved_sec"
CACHE_RETRIEVAL = "/jax/compilation_cache/cache_retrieval_time_sec"


class CompileMetrics:
    """Running sums of the :mod:`jax.monitoring` events of this process (counts, and seconds per duration event),
    so a benchmark can say how much of its wall time was compilation and how much a warm cache saved. Nothing here
    changes what JAX does."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counts = collections.Counter()
        self.seconds = collections.defaultdict(float)
        self.calls = collections.Counter()

    def install(self) -> "CompileMetrics":
        """Register the listeners. Call once, before the first compile."""
        import jax.monitoring  # noqa: PLC0415

        jax.monitoring.register_event_listener(self._on_event)
        jax.monitoring.register_event_duration_secs_listener(self._on_duration)
        return self

    def _on_event(self, event, **kwargs):
        with self._lock:
            self.counts[event] += 1

    def _on_duration(self, event, duration, **kwargs):
        with self._lock:
            self.seconds[event] += float(duration)
            self.calls[event] += 1

    def snapshot(self) -> dict:
        """The sums so far, as plain dicts."""
        with self._lock:
            return dict(counts=dict(self.counts), seconds=dict(self.seconds), calls=dict(self.calls))

    @staticmethod
    def delta(after: dict, before: dict) -> dict:
        """``after - before`` of two :meth:`snapshot` results."""
        return {group: {k: v - before[group].get(k, 0) for k, v in after[group].items()
                        if v - before[group].get(k, 0)}
                for group in ("counts", "seconds", "calls")}

    @staticmethod
    def summary(snap: dict) -> dict:
        """The numbers worth reporting from a :meth:`snapshot` or :meth:`delta`."""
        s, c, n = snap["seconds"], snap["counts"], snap["calls"]
        return dict(
            n_compiles=int(n.get(BACKEND_COMPILE, 0)),
            trace_s=round(s.get(TRACE, 0.0), 3),
            lower_s=round(s.get(LOWER, 0.0), 3),
            backend_compile_s=round(s.get(BACKEND_COMPILE, 0.0), 3),   # includes the cache lookup on a hit
            cache_requests=int(c.get(CACHE_REQUESTS, 0)),
            cache_hits=int(c.get(CACHE_HITS, 0)),
            cache_misses=int(c.get(CACHE_MISSES, 0)),
            cache_retrieval_s=round(s.get(CACHE_RETRIEVAL, 0.0), 3),
            compile_time_saved_s=round(s.get(CACHE_SAVED, 0.0), 3),
        )


def cache_settings() -> dict:
    """The persistent-cache configuration this process runs with, from the environment JAX reads."""
    keys = ("JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES")
    return {k: os.environ.get(k) for k in keys}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="data/wout_li383_1.4m.nc",
                    help="geometry and initial field: a VMEC wout (.nc), GVEC state (.dat) or DESC output (.h5)")
    ap.add_argument("--resolution", type=int, nargs=3, default=(8, 12, 12),
                    help="number of splines in (r, theta, zeta)")
    ap.add_argument("--degree", type=int, default=2, help="spline degree")
    ap.add_argument("--devices", type=int, default=0, help="chips to use, 0 = all")
    ap.add_argument("--factors", type=float, nargs="+", default=None,
                    help="scale factors of the start field, one field each (default 1 + 0.02 i, one per chip)")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                    help="descent methods, run one after the other")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES),
                    help="ways of running the fields to time (see above)")
    ap.add_argument("--gradient-steps", type=int, default=400,
                    help="gradient descent steps per field, a multiple of --gradient-chunk")
    ap.add_argument("--gradient-chunk", type=int, default=200, help="gradient descent steps per compiled chunk")
    ap.add_argument("--newton-steps", type=int, default=20, help="Newton steps per field, a multiple of --newton-chunk")
    ap.add_argument("--newton-chunk", type=int, default=10, help="Newton steps per compiled chunk")
    ap.add_argument("--out", default="outputs/bench/multichip", help="output directory, for multichip_results.json")
    args = ap.parse_args()
    # RelaxConfig rejects a non-multiple too, but only after the setup and under relax.py's --budget.* names.
    budgets = dict(gradient=(args.gradient_steps, args.gradient_chunk), newton=(args.newton_steps, args.newton_chunk))
    for method in args.methods:
        steps, chunk = budgets[method]
        if chunk < 1 or steps < chunk or steps % chunk:
            raise SystemExit(f"--{method}-steps ({steps}) must be a positive multiple of --{method}-chunk ({chunk})")
    return args


def main():
    """Run the requested modes for each requested method and write ``<out>/multichip_results.json``."""
    args = parse_args()
    os.environ.setdefault("MRX_DTYPE", "float32")
    t_process = time.perf_counter()

    metrics = CompileMetrics().install()

    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from jax.sharding import AxisType, NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415

    # MRX_DTYPE must be in the environment before mrx is imported: mrx/precision.py reads it once, at import.
    import mrx  # noqa: PLC0415
    from mrx.nullspace import compute_nullspaces  # noqa: PLC0415
    from mrx.precision import inner, lift  # noqa: PLC0415
    from mrx.relaxation.config import Budget, Descent, Geometry, Method, RelaxConfig  # noqa: PLC0415
    from mrx.relaxation.initial_conditions import initial_field  # noqa: PLC0415
    # _chunk_body and _run_chunk are private: _chunk_body is the step that _run_chunk scans, and chunk_of below
    # rebuilds that scan with the stepper's arrays as an argument, replicated by shard_map and jit_vmap.
    from mrx.relaxation.loop import _chunk_body, _run_chunk, initial_state  # noqa: PLC0415

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "multichip_results.json")
    devices = jax.devices()
    n_dev = args.devices or len(devices)
    devices = devices[:n_dev]
    factors = args.factors or [round(1.0 + 0.02 * i, 4) for i in range(n_dev)]
    if len(factors) % n_dev:
        raise SystemExit(f"{len(factors)} fields do not split evenly over {n_dev} chips")

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    # In a GKE job IMAGE_NAME names the image by digest, <name>@sha256:... (gke/run.sh, gke/job-tpu.yaml).
    results = dict(
        jax=jax.__version__, jaxlib=_version("jaxlib"), libtpu=_version("libtpu"), python=sys.version.split()[0],
        image_name=os.environ.get("IMAGE_NAME"),
        devices=n_dev, device_kind=devices[0].device_kind, platform=devices[0].platform,
        dtype=mrx.DTYPE.name, geometry=args.geometry, resolution=list(args.resolution),
        spline_degree=args.degree, factors=factors, cache=cache_settings(), methods={})

    def write():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=1)

    print(f"[env] jax {jax.__version__}  mrx {mrx.DTYPE.name}  {n_dev} x {devices[0].device_kind}  "
          f"cache {cache_dir or 'off'}  image {os.environ.get('IMAGE_NAME') or 'unknown'}",
          flush=True)

    # --- one-time setup, shared by every mode -------------------------------------------------------
    t0 = time.perf_counter()
    geom = Geometry(path=args.geometry, resolution=tuple(args.resolution), spline_degree=args.degree)
    seq, _ = geom.build()
    t_build = time.perf_counter() - t0
    compute_nullspaces(seq)
    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    B0, ic = initial_field(seq)
    jax.block_until_ready(B0)
    t_ic = time.perf_counter() - t1
    results.update(t_build_s=t_build, t_setup_s=t_setup, t_ic_s=t_ic, ic=ic,
                   setup_compile=CompileMetrics.summary(metrics.snapshot()))
    print(f"[setup] build {t_build:.1f}s  build+nullspaces {t_setup:.1f}s  initial field {t_ic:.1f}s", flush=True)
    write()

    def energy_of(B_vec):
        """Energy 1/2 B.M2 B of one field, on the default device (np.asarray first copies the field to the host)."""
        B_lifted = lift(jnp.asarray(np.asarray(B_vec), dtype=mrx.DTYPE))
        return float(0.5 * inner(B_lifted, seq.odd.M[2] @ B_lifted))

    # Explicit axes for shard_map (the default of jax.make_mesh), Auto axes for the compiler-partitioned vmap.
    mesh = jax.make_mesh((n_dev,), ("eq",), axis_types=(AxisType.Explicit,), devices=devices)
    mesh_auto = jax.make_mesh((n_dev,), ("eq",), axis_types=(AxisType.Auto,), devices=devices)

    for method in args.methods:
        steps = args.newton_steps if method == "newton" else args.gradient_steps
        chunk = args.newton_chunk if method == "newton" else args.gradient_chunk
        cfg = RelaxConfig(geometry=geom, descent=Descent(method=Method(method)), budget=Budget(steps=steps, chunk=chunk))
        ts = cfg.stepper(seq)
        states = [initial_state(B0 * f, ts) for f in factors]
        jax.block_until_ready(states)
        E0_list = [energy_of(st.B_n) for st in states]
        rec = dict(n_steps=steps, chunk=chunk, n_equilibria=len(factors), E0=E0_list,
                   initial_resid=[float(st.best.resid) for st in states])
        results["methods"][method] = rec
        print(f"\n================ {method.upper()} ({steps} steps x {len(factors)} equilibria, chunk {chunk}) "
              f"================", flush=True)

        def timed_first(label, fn):
            """Run ``fn`` once (1 chunk, the call that compiles) and record its time and compile counters."""
            before = metrics.snapshot()
            t = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            dt = time.perf_counter() - t
            comp = CompileMetrics.summary(CompileMetrics.delta(metrics.snapshot(), before))
            print(f"[{method} {label}] first chunk (compile + 1 chunk): {dt:.2f}s  compile: trace {comp['trace_s']:.1f}s "
                  f"lower {comp['lower_s']:.1f}s backend {comp['backend_compile_s']:.1f}s  cache hits "
                  f"{comp['cache_hits']} misses {comp['cache_misses']}", flush=True)
            return out, dt, comp

        # --- serial: one chip, one field after another ------------------------------------------------
        serial_final = None
        if "serial" in args.modes:
            def run(state, it0):
                return _run_chunk(ts, state, jnp.asarray(it0), chunk)

            def relax_one(state):
                """All chunks of one field on one chip, with the best residual after each chunk."""
                chunk_resids = []
                for c in range(steps // chunk):
                    state, _ = run(state, c * chunk)
                    chunk_resids.append(state.best.resid)
                return state, chunk_resids

            try:
                _, t_first, comp = timed_first("serial", lambda: run(states[0], 0)[0])
                walls, finals, E_finals, dE_list = [], [], [], []
                per_chunk_ser = {str((c + 1) * chunk): [] for c in range(steps // chunk)}
                for i, st in enumerate(states):
                    t = time.perf_counter()
                    fin, c_resids = relax_one(st)
                    jax.block_until_ready(fin)
                    walls.append(time.perf_counter() - t)
                    finals.append(fin)
                    for c_idx, cr in enumerate(c_resids):
                        per_chunk_ser[str((c_idx + 1) * chunk)].append(float(cr))
                    E_fin = energy_of(fin.B_n)
                    dE_i = E0_list[i] - E_fin
                    E_finals.append(E_fin)
                    dE_list.append(dE_i)
                    print(f"  [serial eq {i}] factor={factors[i]:.3f}  wall={walls[-1]:.2f}s  "
                          f"best_resid={float(fin.best.resid):.4e}  dE={dE_i:.4e}", flush=True)
                serial_final = finals
                total = float(sum(walls))
                rec["serial"] = dict(t_first_s=t_first, first_call_compile=comp, t_warm_total_s=total,
                                     t_warm_per_eq_s=walls, t_per_eq_s=total / len(states),
                                     t_per_step_per_eq_s=total / len(states) / steps,
                                     best_resid=[float(f.best.resid) for f in finals],
                                     best_resid_by_chunk=per_chunk_ser,
                                     E_final=E_finals, energy_drop=dE_list)
                print(f"[{method} serial] warm total ({len(states)} equilibria on 1 chip): {total:.2f}s "
                      f"({total / len(states):.2f}s / eq)", flush=True)
            except Exception as exc:  # noqa: BLE001  (record and go on with the other modes)
                rec["serial"] = _failure(exc)
            write()

        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
        ts_dyn, ts_static = eqx.partition(ts, eqx.is_array)

        def chunk_of(ts_dyn_, state, it0):
            """``chunk`` steps of one field, the body of ``mrx.relaxation.loop._run_chunk``."""
            ts_ = eqx.combine(ts_dyn_, ts_static)
            return jax.lax.scan(lambda st, it: _chunk_body(ts_, st, it), state, it0 + jnp.arange(1, chunk + 1))

        def relax_batch(fn, ts_args, batch):
            """All chunks of the stacked fields with fn, with each chunk's best residuals, per-step dE and newton_it."""
            chunk_best_resids = []
            chunk_dEs = []
            chunk_newton_its = []
            for c in range(steps // chunk):
                batch, tr = fn(ts_args, batch, jnp.asarray(c * chunk, dtype=jnp.int32))
                chunk_best_resids.append(batch.best.resid)
                chunk_dEs.append(tr["dE"])
                chunk_newton_its.append(tr["newton_it"])
            return batch, chunk_best_resids, chunk_dEs, chunk_newton_its

        # --- shard_map: the per-device program, one field per chip --------------------------------------
        if "shard_map" in args.modes:
            try:
                def per_device(ts_dyn_, local, it0):
                    if jax.tree.leaves(local)[0].shape[0] == 1:      # one field per chip: the serial program
                        st, tr = chunk_of(ts_dyn_, jax.tree.map(lambda x: x[0], local), it0)
                        return jax.tree.map(lambda x: x[None], (st, tr))
                    return jax.lax.map(lambda s: chunk_of(ts_dyn_, s, it0), local)

                # check_vma=True rejects mrx.solvers.refine: its iteration count starts at 0 on all chips, then varies.
                sharded = jax.jit(jax.shard_map(per_device, mesh=mesh, in_specs=(P(), P("eq"), P()),
                                                out_specs=P("eq"), check_vma=False))
                ts_rep = jax.device_put(ts_dyn, NamedSharding(mesh, P()))
                batch0 = jax.device_put(stacked, NamedSharding(mesh, P("eq")))
                rec["shard_map"] = _parallel(
                    "shard_map",
                    lambda: sharded(ts_rep, batch0, jnp.asarray(0, dtype=jnp.int32))[0],
                    lambda: relax_batch(sharded, ts_rep, batch0),
                    timed_first, serial_final, factors, steps, chunk, method, E0_list, energy_of, jax, np)
            except Exception as exc:  # noqa: BLE001
                rec["shard_map"] = _failure(exc)
            write()

        # --- jit_vmap: global view, the batch axis sharded and partitioned by the compiler ---------------
        if "jit_vmap" in args.modes:
            rep, bat = NamedSharding(mesh_auto, P()), NamedSharding(mesh_auto, P("eq"))
            ts_rep = jax.device_put(ts_dyn, rep)
            batch0 = jax.device_put(stacked, bat)
            variant = "vmap(spmd_axis_name='eq')"
            try:
                batched = jax.jit(jax.vmap(chunk_of, in_axes=(None, 0, None), spmd_axis_name="eq"),
                                  in_shardings=(rep, bat, rep), out_shardings=bat)

                def go_first():
                    with jax.set_mesh(mesh_auto):
                        return batched(ts_rep, batch0, jnp.asarray(0, dtype=jnp.int32))[0]

                def go():
                    with jax.set_mesh(mesh_auto):
                        return relax_batch(batched, ts_rep, batch0)
                r = _parallel("jit_vmap", go_first, go, timed_first, serial_final, factors, steps, chunk, method,
                              E0_list, energy_of, jax, np)
            except Exception as exc:  # noqa: BLE001  (retry with the shardings of the inputs alone)
                print(f"[{method} jit_vmap] {variant} failed ({type(exc).__name__}: {exc}); retrying with plain "
                      f"vmap and in_shardings", flush=True)
                variant = "vmap + in_shardings"
                try:
                    batched = jax.jit(jax.vmap(chunk_of, in_axes=(None, 0, None)),
                                      in_shardings=(rep, bat, rep), out_shardings=bat)
                    r = _parallel("jit_vmap",
                                  lambda: batched(ts_rep, batch0, jnp.asarray(0, dtype=jnp.int32))[0],
                                  lambda: relax_batch(batched, ts_rep, batch0),
                                  timed_first, serial_final, factors, steps, chunk, method, E0_list, energy_of, jax, np)
                except Exception as exc2:  # noqa: BLE001
                    r = _failure(exc2)
            r["variant"] = variant
            rec["jit_vmap"] = r
            write()

        for mode in ("shard_map", "jit_vmap"):
            if "serial" in rec and mode in rec and "t_warm_total_s" in rec[mode] and "t_warm_total_s" in rec["serial"]:
                rec[mode]["speedup_vs_serial"] = rec["serial"]["t_warm_total_s"] / rec[mode]["t_warm_total_s"]
                print(f"[{method} {mode}] SPEEDUP vs serial (warm) = {rec[mode]['speedup_vs_serial']:.2f}x",
                      flush=True)
        write()

    results.update(t_process_s=time.perf_counter() - t_process, compile_total=CompileMetrics.summary(metrics.snapshot()))
    write()
    print(f"\n[done] {time.perf_counter() - t_process:.1f}s, wrote {out_path}", flush=True)


def _parallel(mode, first_call, call, timed_first, serial_final, factors, steps, chunk, method,
              E0_list, energy_of, jax, np):
    """Time the first (1-chunk compile) and steady-state call of a parallel mode and compare with serial."""
    _, t_first, comp = timed_first(mode, first_call)
    t = time.perf_counter()
    final, chunk_best_resids, chunk_dEs, chunk_newton_its = call()
    jax.block_until_ready(final)
    warm = time.perf_counter() - t
    n = len(factors)
    best = np.asarray(final.best.resid, dtype=np.float64)
    B_par = np.asarray(final.B_n)
    E_finals = [energy_of(B_par[i]) for i in range(n)]
    dE_list = [E0_list[i] - E_finals[i] for i in range(n)]

    best_resid_by_chunk = {}
    energy_drop_by_chunk = {}
    newton_it_by_chunk = {}
    cum_dE = np.zeros(n, dtype=np.float64)
    for c_idx in range(len(chunk_best_resids)):
        step_num = str((c_idx + 1) * chunk)
        c_res = np.asarray(chunk_best_resids[c_idx], dtype=np.float64)
        best_resid_by_chunk[step_num] = c_res.tolist()
        c_de = np.asarray(chunk_dEs[c_idx], dtype=np.float64)
        cum_dE = cum_dE - np.sum(c_de, axis=1)
        energy_drop_by_chunk[step_num] = cum_dE.tolist()
        if method == "newton":
            c_nit = np.asarray(chunk_newton_its[c_idx], dtype=np.int64)
            newton_it_by_chunk[step_num] = c_nit.tolist()
        print(f"  [{mode} chunk @ step {step_num}] best_resid={[float(f'{x:.4e}') for x in c_res]}  "
              f"cum_dE={[float(f'{x:.4e}') for x in cum_dE]}", flush=True)

    rec = dict(t_first_s=t_first, first_call_compile=comp, t_warm_total_s=warm, t_effective_per_eq_s=warm / n,
               t_effective_per_step_per_eq_s=warm / n / steps, best_resid=best.tolist(),
               best_resid_by_chunk=best_resid_by_chunk,
               energy_drop_by_chunk=energy_drop_by_chunk,
               E_final=E_finals, energy_drop=dE_list,
               sharding=str(final.B_n.sharding))
    if method == "newton":
        rec["newton_it_by_chunk"] = newton_it_by_chunk
    if serial_final is not None:
        ref_best = np.array([float(f.best.resid) for f in serial_final])
        B_par_f64 = np.asarray(B_par, dtype=np.float64)
        B_ser = np.stack([np.asarray(f.B_n, dtype=np.float64) for f in serial_final])
        rec["rel_diff_best_resid_vs_serial"] = (np.abs(best - ref_best) / ref_best).tolist()
        rec["rel_diff_field_vs_serial"] = (np.linalg.norm(B_par_f64 - B_ser, axis=1)
                                           / np.linalg.norm(B_ser, axis=1)).tolist()
    for i in range(n):
        line = f"  [{mode} eq {i}] factor={factors[i]:.3f}  best_resid={best[i]:.4e}  dE={dE_list[i]:.4e}"
        if "rel_diff_best_resid_vs_serial" in rec:
            line += (f"  rel_diff_vs_serial: resid {rec['rel_diff_best_resid_vs_serial'][i]:.3e} "
                     f"field {rec['rel_diff_field_vs_serial'][i]:.3e}")
        print(line, flush=True)
    print(f"[{method} {mode}] warm total ({n} equilibria, {len(final.B_n.sharding.device_set)} chips): {warm:.2f}s "
          f"({warm / n:.2f}s effective / eq)", flush=True)
    return rec


def _failure(exc):
    """Print the traceback of ``exc`` (the exception being handled) and return the error as the mode's record."""
    traceback.print_exc()
    return dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-TRACEBACK_CHARS:])


def _version(dist):
    """The installed version of the distribution ``dist``, or None."""
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return version(dist)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
