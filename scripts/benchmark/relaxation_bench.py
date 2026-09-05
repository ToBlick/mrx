#!/usr/bin/env python3
"""Phase and primitive benchmark for MRX, to compare one backend against another.

The question this answers: when MRX is slow on a given backend, is it slow
because XLA spends minutes compiling, or because the compiled program runs
slowly? Every measurement below therefore reports the FIRST call and the SECOND
call separately. The first includes tracing and compilation; the second is
steady state. A phase with a huge first call and a fast second call is a
compilation problem and a persistent cache fixes it. A phase where both are
slow is a real execution problem and needs the kernel changed.

Run it twice on the same machine so the comparison isolates the backend:

    JAX_PLATFORMS=tpu python -u scripts/benchmark/relaxation_bench.py --out tpu.json
    JAX_PLATFORMS=cpu python -u scripts/benchmark/relaxation_bench.py --out cpu.json
    python -u scripts/benchmark/relaxation_bench.py --compare tpu.json cpu.json

The comparison folds in one reference number for the same code on an H100
(0.41 s/step for W7-X at (12,24,12), from
``docs/research/release_review_sweep_2026-08-27.md``) so a result can be read
against the hardware the code was tuned on. Measured TPU, CPU and H200 numbers
are in ``docs/research/tpu_v5e_benchmark.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# Microbenchmark rows are dispatched this many times per timing sample; a
# single apply is tens of microseconds, below one clock reading.
MICRO_INNER = 50

H100_REFERENCE = {
    "relax_step": 0.41,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    ap.add_argument("--matmul-precision", default=None,
                    help="override jax_default_matmul_precision AFTER mrx sets "
                         "it to 'highest'. On a TPU MXU 'highest' means float32 "
                         "emulated in 6 bf16 passes, 'high' 3 and 'default' 1, "
                         "so this is the knob that is a TPU-only tax. 'default' "
                         "folds the li383 map and must not be used")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="read result JSONs and print the comparison table")
    ap.add_argument("--skip-phases", action="store_true",
                    help="primitives only; skips the multi-minute phase timings")
    ap.add_argument("--skip-relax", action="store_true",
                    help="skip the relaxation phase (the slowest item)")
    ap.add_argument("--relax-steps", type=int, default=5,
                    help="inner steps per timed relaxation call. The per-step "
                         "cost is read off the slope between this and twice "
                         "this, so raising it buys accuracy with wall time")
    ap.add_argument("--map-batch-size", type=int, default=None,
                    help="mrx.MAP_BATCH_SIZE_INNER; 0 (the mrx default) is a "
                         "full vmap and is rejected by jax < 0.9, so the "
                         "singularity overlay on NYU Torch needs a positive "
                         "value. It batches the quadrature loop, so it changes "
                         "setup memory and not the primitives below")
    ap.add_argument("--cache-dir", default=None,
                    help="enable the persistent XLA compilation cache here; "
                         "the point is to see whether repeated compilation of "
                         "the same program can be turned into a cache hit")
    return ap.parse_args()


# --------------------------------------------------------------- timing ---

def _sync(x):
    """Block until every array in a pytree is materialised on device."""
    import jax
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(x)
              if hasattr(leaf, "block_until_ready")]
    for leaf in leaves:
        leaf.block_until_ready()
    return x


class Bench:
    """Collects (first call, second call) timings keyed by name."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def measure(self, name, fn, repeats=3, inner=1, note=""):
        """Time ``fn``, reporting the first call and the best steady-state call.

        The split is the whole point: call 1 pays tracing plus compilation,
        later calls do not.

        ``inner`` repetitions are dispatched back to back and only the final
        result is blocked on. A single microbenchmark call is a few tens of
        microseconds, which is below the resolution of one wall-clock reading,
        and blocking after every call would measure Python dispatch latency
        instead of the kernel. Ops on one device execute in issue order, so
        waiting on the last output waits on all of them.
        """
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            try:
                out = None
                for _ in range(inner):
                    out = fn()
                _sync(out)
            except Exception as exc:                      # noqa: BLE001
                self.rows[name] = {"error": f"{type(exc).__name__}: {exc}",
                                   "note": note}
                print(f"  {name:<38} FAILED  {type(exc).__name__}: {exc}",
                      flush=True)
                return None
            times.append((time.perf_counter() - t0) / inner)

        first = times[0]
        steady = min(times[1:]) if len(times) > 1 else float("nan")
        # Everything after the first call is compilation-free, so the gap
        # between them is the compile cost. With inner > 1 the first sample is
        # already amortised over inner calls, so compile_s understates by that
        # factor; the raw first-call compile figure is only meaningful at
        # inner = 1, which is what the phase timings use.
        self.rows[name] = {"first_s": first, "steady_s": steady,
                           "compile_s": first - steady if len(times) > 1 else None,
                           "inner": inner, "note": note}
        print(f"  {name:<38} first {first:9.3f}s   "
              f"steady {steady * 1e3:9.3f}ms   {note}", flush=True)
        return times


# ----------------------------------------------------------- primitives ---

def bench_structured_scatter(bench, ne=(12, 24, 12), nloc=(4, 4, 4),
                             dtype="float32"):
    """Compare the mass kernel's scatter against a shift-and-add equivalent.

    ``_sumfact_kernel`` used to finish with a ``segment_sum`` whose segment ids
    were ``gx * (Sy * Sz) + gy * Sz + gz`` -- a *separable tensor product* of
    the per-axis global DoF ids. For a periodic B-spline axis ``gx[e, l] = e + l``
    (mod n), so the whole scatter is algebraically

        out[i,j,k] = sum over local (lx,ly,lz) of contrib[i-lx, j-ly, k-lz, ...]

    i.e. a sum of ``prod(nloc)`` shifted dense arrays. That formulation has no
    indexed writes at all, which is the property a TPU cares about, and it is
    what ``_structured_accumulate`` now does. This row is the regression guard:
    it keeps the indexed form measurable next to the one that replaced it.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    nex, ney, nez = ne
    nlx, nly, nlz = nloc
    n_out = nex * ney * nez
    n_contrib = n_out * nlx * nly * nlz

    print(f"\n[primitive] mass-kernel scatter: indexed vs structured\n"
          f"            ({n_contrib} contributions -> {n_out} dofs, "
          f"periodic tensor product)", flush=True)

    rng = np.random.default_rng(7)
    contrib = jnp.asarray(
        rng.standard_normal((nex, ney, nez, nlx, nly, nlz)).astype(dtype))

    # The flat index plan the indexed form needs, periodic.
    gx = (np.arange(nex)[:, None] + np.arange(nlx)[None, :]) % nex
    gy = (np.arange(ney)[:, None] + np.arange(nly)[None, :]) % ney
    gz = (np.arange(nez)[:, None] + np.arange(nlz)[None, :]) % nez
    seg = (gx[:, None, None, :, None, None] * (ney * nez)
           + gy[None, :, None, None, :, None] * nez
           + gz[None, None, :, None, None, :]).astype(np.int32).reshape(-1)
    seg_j = jnp.asarray(seg)

    f_indexed = jax.jit(lambda c: jax.ops.segment_sum(
        c.reshape(-1), seg_j, num_segments=n_out))
    bench.measure("mass scatter: segment_sum (indexed)",
                  lambda: f_indexed(contrib), inner=20,
                  note=f"{n_contrib} indexed writes")

    def structured(c):
        # jnp.roll on a static shift is a pure data movement XLA can express
        # without any gather: every element's destination is known at compile
        # time, so this lowers to dense copies and adds.
        out = jnp.zeros((nex, ney, nez), dtype=c.dtype)
        for lx in range(nlx):
            for ly in range(nly):
                for lz in range(nlz):
                    out = out + jnp.roll(
                        c[:, :, :, lx, ly, lz], shift=(lx, ly, lz),
                        axis=(0, 1, 2))
        return out.reshape(-1)

    f_structured = jax.jit(structured)
    bench.measure("mass scatter: shift-and-add (structured)",
                  lambda: f_structured(contrib), inner=20,
                  note=f"{nlx * nly * nlz} dense shifted adds")

    # Correctness: the two must agree, or the timing means nothing.
    a = np.asarray(f_indexed(contrib))
    b = np.asarray(f_structured(contrib))
    err = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30))
    print(f"  agreement between the two: max rel err {err:.2e}"
          f"  {'OK' if err < 1e-5 else 'MISMATCH'}", flush=True)
    bench.rows["mass scatter: agreement"] = {
        "first_s": None, "steady_s": None, "compile_s": None,
        "note": f"max rel err {err:.2e}"}


def bench_structured_gather(bench, ne=(12, 24, 12), nloc=(4, 4, 4),
                            dtype="float32"):
    """The mirror of :func:`bench_structured_scatter` for the read side.

    ``_to_quadrature`` used to start with ``x_flat[gather_idx]``, over the same
    separable index plan. So the gather is algebraically

        x_local[e,l] = x[(e + l) mod n]

    which is a stack of rolled slices -- again with every source known at
    compile time. Taken axis by axis it is ``nlx + nly + nlz`` rolls rather
    than ``prod(nloc)``, the same factorisation the assembly uses.

    A TPU has no fast path for indexed *writes*, and it turned out to have none
    for indexed *reads* of this size either: the structured form measured 33x
    faster, so ``_structured_gather`` now does this. Kept as the regression
    guard for the read side.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    nex, ney, nez = ne
    nlx, nly, nlz = nloc
    n_in = nex * ney * nez
    n_read = n_in * nlx * nly * nlz

    print(f"\n[primitive] mass-kernel gather: indexed vs structured\n"
          f"            ({n_in} dofs -> {n_read} reads, "
          f"periodic tensor product)", flush=True)

    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.standard_normal(n_in).astype(dtype))

    gx = (np.arange(nex)[:, None] + np.arange(nlx)[None, :]) % nex
    gy = (np.arange(ney)[:, None] + np.arange(nly)[None, :]) % ney
    gz = (np.arange(nez)[:, None] + np.arange(nlz)[None, :]) % nez
    idx = (gx[:, None, None, :, None, None] * (ney * nez)
           + gy[None, :, None, None, :, None] * nez
           + gz[None, None, :, None, None, :]).astype(np.int32)
    idx_j = jnp.asarray(idx)

    f_indexed = jax.jit(lambda v: v[idx_j])
    bench.measure("mass gather: x[gather_idx] (indexed)",
                  lambda: f_indexed(x), inner=20,
                  note=f"{n_read} indexed reads")

    def structured(v):
        # Roll by -l so that entry e of the rolled array is source e + l, then
        # take the first ne entries. One axis at a time: the intermediate after
        # the x pass is (ne_x, Sy, Sz, nlx), so the y pass rolls a smaller
        # array than a full prod(nloc) expansion would.
        a = v.reshape(nex, ney, nez)
        a = jnp.stack([jnp.roll(a, -lx, axis=0)[:nex]
                       for lx in range(nlx)], axis=3)
        a = jnp.stack([jnp.roll(a, -ly, axis=1)[:, :ney]
                       for ly in range(nly)], axis=4)
        a = jnp.stack([jnp.roll(a, -lz, axis=2)[:, :, :nez]
                       for lz in range(nlz)], axis=5)
        return a

    f_structured = jax.jit(structured)
    bench.measure("mass gather: rolled slices (structured)",
                  lambda: f_structured(x), inner=20,
                  note=f"{nlx + nly + nlz} rolls, separable")

    a = np.asarray(f_indexed(x))
    b = np.asarray(f_structured(x))
    err = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30))
    print(f"  agreement between the two: max rel err {err:.2e}"
          f"  {'OK' if err < 1e-5 else 'MISMATCH'}", flush=True)
    bench.rows["mass gather: agreement"] = {
        "first_s": None, "steady_s": None, "compile_s": None,
        "note": f"max rel err {err:.2e}"}


def bench_extraction(bench, seq, dtype):
    """Time the real extraction operator, and report whether it can be hinted.

    ``E`` and ``E^T`` run on every matvec, so their cost is multiplied by the
    ~10^5 Krylov iterations of a full relaxation. The structural report is what
    decides the fix: if a segment index array is already sorted, the hint is
    free to add; if every segment receives exactly one contribution, the
    scatter can be dropped for a plain gather.
    """
    import jax.numpy as jnp
    import numpy as np

    print("\n[primitive] extraction operator E and E^T (real sizes)", flush=True)

    for k in (0, 1, 2):
        try:
            E = seq.E(k, False)
            n_out, n_raw = int(E.forward_shape[0]), int(E.forward_shape[1])
        except Exception as exc:                          # noqa: BLE001
            print(f"  k={k}: skipped ({type(exc).__name__}: {exc})", flush=True)
            continue

        rows = np.asarray(E.rows)
        cols = np.asarray(E.cols)
        nnz = rows.size
        # Forward scatters into rows; the transpose scatters into cols.
        fwd_sorted = bool(np.all(np.diff(rows) >= 0))
        rev_sorted = bool(np.all(np.diff(cols) >= 0))
        fwd_unique = bool(np.unique(rows).size == nnz)
        rev_unique = bool(np.unique(cols).size == nnz)
        struct = (f"nnz={nnz} out={n_out} raw={n_raw} "
                  f"fwd[sorted={fwd_sorted},unique={fwd_unique}] "
                  f"rev[sorted={rev_sorted},unique={rev_unique}]")
        print(f"  k={k} structure: {struct}", flush=True)
        bench.rows[f"extraction_structure_k{k}"] = {
            "first_s": None, "steady_s": None, "compile_s": None,
            "note": struct, "nnz": nnz, "n_out": n_out, "n_raw": n_raw,
            "fwd_sorted": fwd_sorted, "rev_sorted": rev_sorted,
            "fwd_unique": fwd_unique, "rev_unique": rev_unique}

        rng = np.random.default_rng(10 + k)
        raw = jnp.asarray(rng.standard_normal(n_raw).astype(dtype))
        out = jnp.asarray(rng.standard_normal(n_out).astype(dtype))
        bench.measure(f"E   apply k={k}", lambda E=E, v=raw: E @ v,
                      inner=MICRO_INNER, note=f"{n_raw}->{n_out}")
        bench.measure(f"E^T apply k={k}", lambda E=E, v=out: E.T @ v,
                      inner=MICRO_INNER, note=f"{n_out}->{n_raw}")


def bench_mass(bench, seq, dtype):
    """Time the sum-factorised mass apply, the most-used kernel in the code.

    The einsums here are the TPU-friendly part of MRX: dense, contraction
    shaped, exactly what an MXU wants. If this row is competitive and the
    extraction rows are not, the problem is the scatter and not the algebra.
    """
    import jax.numpy as jnp
    import numpy as np
    from mrx.operators import mass_core_apply

    print("\n[primitive] mass apply (sum factorisation + segment_sum)",
          flush=True)

    for k in (0, 1, 2):
        try:
            core = mass_core_apply(seq, k)
            E = seq.E(k, False)
            n_raw = int(E.forward_shape[1])
            rng = np.random.default_rng(20 + k)
            raw = jnp.asarray(rng.standard_normal(n_raw).astype(dtype))
            bench.measure(f"mass_core_apply k={k}",
                          lambda c=core, v=raw: c(v),
                          inner=MICRO_INNER, note=f"raw n={n_raw}")
        except Exception as exc:                          # noqa: BLE001
            print(f"  mass_core_apply k={k}: skipped "
                  f"({type(exc).__name__}: {exc})", flush=True)


def bench_operators(bench, seq, ops, dtype):
    """Time the composite applies that every Krylov iteration runs.

    Deliberately not wrapped in an extra ``jax.jit``: these are measured as the
    library actually calls them, and jitting from here would capture ``seq``'s
    arrays as compile-time constants, which is the very failure mode under
    investigation.
    """
    import jax.numpy as jnp
    import numpy as np
    from mrx.operators import apply_derivative_matrix, apply_laplacian

    rng = np.random.default_rng(1)
    print("\n[primitive] composite operator applies", flush=True)

    for k in (0, 1, 2):
        try:
            n_in = int(seq.n(k, False))
        except Exception:                                 # noqa: BLE001
            continue
        vec = jnp.asarray(rng.standard_normal(n_in).astype(dtype))
        bench.measure(
            f"apply_derivative_matrix k={k}",
            lambda k=k, v=vec: apply_derivative_matrix(
                seq, v, k, dirichlet_in=False, dirichlet_out=False),
            note=f"n={n_in}")

    # The inner CG's ITERATION COUNT, not just its wall time. A backend can be
    # slow because its kernels are slow or because its arithmetic makes the
    # solver converge less well, and those have completely different remedies.
    # Wall time alone cannot tell them apart; this can.
    from mrx.operators import apply_inverse_mass_matrix
    for k in (1, 2):
        try:
            n_in = int(seq.n(k, False))
            rhs = jnp.asarray(rng.standard_normal(n_in).astype(dtype))
            t0 = time.perf_counter()
            x, info = apply_inverse_mass_matrix(
                seq, ops, rhs, k, dirichlet=False, return_info=True)
            _sync(x)
            elapsed = time.perf_counter() - t0
            # preconditioned_cg returns a signed iteration count: negative
            # means it converged in that many iterations, positive means it hit
            # maxiter without converging.
            signed = int(info)
            iters = abs(signed)
            converged = signed <= 0
            note = (f"n={n_in} iters={iters} "
                    f"{'converged' if converged else 'HIT MAXITER'}")
            bench.rows[f"mass_cg_iters_k{k}"] = {
                "first_s": elapsed, "steady_s": None, "compile_s": None,
                "iterations": iters, "converged": converged, "note": note}
            print(f"  {'inverse mass CG k=' + str(k):<38} "
                  f"{elapsed:9.3f}s   {note}", flush=True)
            if iters:
                print(f"  {'-> per CG iteration':<38} "
                      f"{elapsed / iters * 1e3:9.3f}ms", flush=True)
        except Exception as exc:                          # noqa: BLE001
            print(f"  inverse mass CG k={k}: skipped "
                  f"({type(exc).__name__}: {exc})", flush=True)

    # k=0 is a plain stiffness apply; k=1 additionally runs a nested CG mass
    # solve as a lax.while_loop, which is where the nullspace time goes. That
    # makes the k>=1 row a SOLVE, not an apply: it is 20-27 CG iterations, so
    # it belongs beside apply_inverse_mass_matrix above and not beside the
    # matvecs, where it overstates the cost of an apply by about 150x. The
    # names say so, because the earlier results table put it in the matvec
    # column and the figure was read as a per-apply cost.
    for k, dbc in ((0, False), (1, False)):
        try:
            n_in = int(seq.n(k, dbc))
        except Exception:                                 # noqa: BLE001
            continue
        vec = jnp.asarray(rng.standard_normal(n_in).astype(dtype))
        label = (f"apply_laplacian k={k} free" if k == 0
                 else f"SOLVE apply_laplacian k={k} free")
        bench.measure(
            label,
            lambda k=k, dbc=dbc, v=vec: apply_laplacian(
                seq, ops, v, k, dirichlet=dbc),
            repeats=2,
            note=f"n={n_in}" + (" NOT an apply: nested CG" if k >= 1 else ""))


# --------------------------------------------------------------- phases ---

def bench_phases(bench, args, dtype):
    """Time build_sequence, compute_nullspaces and the relaxation loop."""
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    ns = tuple(int(v) for v in args.ns.split(","))

    print("\n[phase] setup", flush=True)
    t0 = time.perf_counter()
    seq, ops = build_sequence(args.geometry, ns, args.p)
    build_s = time.perf_counter() - t0
    bench.rows["build_sequence"] = {"first_s": build_s, "steady_s": None,
                                    "compile_s": None, "note": f"ns={ns} p={args.p}"}
    print(f"  {'build_sequence':<38} {build_s:9.3f}s", flush=True)

    # gap_sweeps=0 throughout: the lambda_1 estimate is a diagnostic, it is not
    # part of the construction, and it cost 6.8 of 9 minutes here.
    t0 = time.perf_counter()
    ops0 = compute_nullspaces(seq, ops, gap_sweeps=0, verbose=False)
    gap0_s = time.perf_counter() - t0
    bench.rows["nullspace_gap_sweeps_0"] = {
        "first_s": gap0_s, "steady_s": None, "compile_s": None,
        "note": "construction only"}
    print(f"  {'compute_nullspaces gap_sweeps=0':<38} {gap0_s:9.3f}s"
          f"   construction only", flush=True)

    seq.set_operators(ops0)
    return seq, ops0


def bench_relaxation(bench, seq, args, dtype):
    """Time the relaxation loop: compile cost, then cost per step."""
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import clebsch_potential_form, potential_two_form
    from mrx.relaxation import TimeStepper, chunk_runner, initial_state

    ns = tuple(int(v) for v in args.ns.split(","))
    print("\n[phase] relaxation", flush=True)

    cb = load_clebsch(seq.equilibrium)
    B0, _, _ = potential_two_form(seq, clebsch_potential_form(cb))

    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                     velocity_smoothing_order=1,
                     velocity_smoothing_scale=0.064 / ns[0] ** 2)
    state0 = initial_state(B0, ts, 1.0)
    _sync(state0.B_n)

    inner = args.relax_steps

    def time_calls(n_steps):
        """First and second call of one compiled chunk. ``chunk_runner``
        builds the jit once; the second call is compile-free."""
        run = chunk_runner(ts, n_steps)
        t0 = time.perf_counter()
        state, _ = run(state0, 0)
        _sync(state.B_n)
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        state, _ = run(state0, 0)
        _sync(state.B_n)
        return first, time.perf_counter() - t0

    first_s, steady_s = time_calls(inner)

    # Per step from the slope between two lengths, not from steady / inner.
    # Each call still pays a fixed launch cost, and dividing by the step
    # count would charge all of that to the steps. Differencing cancels it.
    # The doubled length compiles its own scan, so take its second call too.
    _, long_s = time_calls(2 * inner)
    per_step = (long_s - steady_s) / inner
    overhead_s = steady_s - inner * per_step

    bench.rows["relax_compile"] = {
        "first_s": first_s, "steady_s": steady_s,
        "compile_s": first_s - steady_s,
        "note": f"{inner} inner steps"}
    bench.rows["relax_step"] = {
        "first_s": None, "steady_s": per_step, "compile_s": None,
        "note": f"per step, slope {inner}->{2 * inner}"}
    bench.rows["relax_call_overhead"] = {
        "first_s": None, "steady_s": overhead_s, "compile_s": None,
        "note": "per call, outside the scan"}

    print(f"  {'relaxation ' + str(inner) + ' steps':<38} first {first_s:9.3f}s"
          f"   steady {steady_s:9.4f}s", flush=True)
    print(f"  {'relaxation ' + str(2 * inner) + ' steps':<38} "
          f"{'':>15} steady {long_s:9.4f}s", flush=True)
    print(f"  {'-> per step (slope)':<38} {'':>15} {per_step:15.4f}s", flush=True)
    print(f"  {'-> compile, once':<38} {'':>15} "
          f"{first_s - steady_s:15.4f}s", flush=True)
    print(f"  {'-> per-call overhead outside the scan':<38} {'':>15} "
          f"{overhead_s:15.4f}s", flush=True)


# ------------------------------------------------------------- reporting ---

def backend_info():
    import jax
    devs = jax.devices()
    return {
        "jax_version": jax.__version__,
        "platform": devs[0].platform,
        "device_kind": devs[0].device_kind,
        "device_count": len(devs),
        "python": platform.python_version(),
        "host": platform.node(),
    }


def print_comparison(paths):
    """Print one row per measurement with a column per input JSON."""
    runs = []
    for path in paths:
        with open(path) as fh:
            runs.append(json.load(fh))

    labels = [f"{r['backend']['platform']}/{r['backend']['device_kind']}"[:18]
              for r in runs]
    names: list[str] = []
    for r in runs:
        for key in r["rows"]:
            if key not in names:
                names.append(key)

    width = max(len(n) for n in names) + 2
    header = f"{'measurement':<{width}}" + "".join(f"{lab:>20}" for lab in labels)
    if any(n in H100_REFERENCE for n in names):
        header += f"{'H100 ref':>12}"
    print()
    print(header)
    print("-" * len(header))

    for name in names:
        # Structural rows carry a description, not a duration.
        if name.startswith("extraction_structure"):
            for r in runs:
                row = r["rows"].get(name)
                if row:
                    print(f"{name:<{width}}  {row.get('note', '')}")
                    break
            continue

        line = f"{name:<{width}}"
        vals = []
        for r in runs:
            row = r["rows"].get(name)
            if row is None:
                line += f"{'-':>20}"
                vals.append(None)
                continue
            if "error" in row:
                line += f"{'FAILED':>20}"
                vals.append(None)
                continue
            steady = row.get("steady_s")
            shown = steady if steady and steady == steady else row.get("first_s")
            vals.append(shown)
            # 4 significant digits, not 4 decimal places: these span 10^-5 to
            # 10^3 seconds and fixed-point rounds the microbenchmarks to zero.
            line += f"{shown:>20.4g}" if shown is not None else f"{'-':>20}"
        if name in H100_REFERENCE:
            line += f"{H100_REFERENCE[name]:>12.2f}"
        # The ratio between the first two columns is the headline: how much
        # worse is this backend than the one next to it, on this exact op.
        if len(vals) >= 2 and vals[0] and vals[1]:
            line += f"   x{vals[0] / vals[1]:.1f}"
        print(line)
    print()
    print("Columns are steady-state seconds where available, else first-call.")
    print("The trailing xN is column 1 divided by column 2.")


def main():
    args = parse_args()

    if args.compare:
        print_comparison(args.compare)
        return 0

    os.environ.setdefault("MRX_DTYPE", args.precision)
    os.environ.setdefault("MPLBACKEND", "Agg")

    import jax

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)
        # Defaults are set for long-running training jobs and would skip
        # caching the many small programs this workload compiles.
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)

    import mrx

    if args.precision != str(mrx.DTYPE):
        print(f"ERROR: --precision {args.precision} but mrx is in {mrx.DTYPE}; "
              "MRX_DTYPE was already set elsewhere.", file=sys.stderr)
        return 1

    if args.map_batch_size is not None:
        mrx.MAP_BATCH_SIZE_INNER = args.map_batch_size

    if args.matmul_precision:
        jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    info = backend_info()
    print("=" * 72)
    print("MRX phase and primitive benchmark")
    print("=" * 72)
    for key, value in info.items():
        print(f"  {key:<20} {value}")
    from mrx.precision import REFINE, RESIDUAL_DTYPE, SOLVE_TOL
    print(f"  {'mrx_dtype':<20} {mrx.DTYPE}")
    print(f"  {'residual_dtype':<20} {RESIDUAL_DTYPE}")
    print(f"  {'refine':<20} {REFINE}")
    print(f"  {'solve_tol':<20} {SOLVE_TOL}")
    print(f"  {'matmul_precision':<20} {jax.config.jax_default_matmul_precision}")

    dtype = str(mrx.DTYPE)
    bench = Bench()

    # Primitives first: they are seconds each and they alone can settle the
    # scatter question, so they must not be gated behind a phase that may run
    # for half an hour.
    bench_structured_scatter(bench, dtype=dtype)
    bench_structured_gather(bench, dtype=dtype)

    if not args.skip_phases:
        seq, ops = bench_phases(bench, args, dtype)
        bench_extraction(bench, seq, dtype)
        bench_mass(bench, seq, dtype)
        bench_operators(bench, seq, ops, dtype)
        if not args.skip_relax:
            bench_relaxation(bench, seq, args, dtype)

    result = {"backend": info, "mrx_dtype": dtype,
              "residual_dtype": str(RESIDUAL_DTYPE), "refine": REFINE,
              "solve_tol": SOLVE_TOL, "args": vars(args),
              "rows": bench.rows}
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nresults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
