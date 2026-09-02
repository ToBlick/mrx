#!/usr/bin/env python3
"""Phase and primitive benchmark for MRX, to compare a TPU against a CPU.

The question this answers: when MRX is slow on a TPU, is it slow because XLA
spends minutes compiling, or because the compiled program runs slowly? Every
measurement below therefore reports the FIRST call and the SECOND call
separately. The first includes tracing and compilation; the second is steady
state. A phase with a huge first call and a fast second call is a compilation
problem and a persistent cache fixes it. A phase where both are slow is a real
execution problem and needs the kernel changed.

Run it twice on the same machine so the comparison isolates the backend:

    JAX_PLATFORMS=tpu python -u tpu_bench_mrx.py --out bench_tpu.json
    JAX_PLATFORMS=cpu python -u tpu_bench_mrx.py --out bench_cpu.json
    python -u tpu_bench_mrx.py --compare bench_tpu.json bench_cpu.json

Reference numbers for the same code on one H100, from the MRX docs, are folded
into the comparison so a TPU result can be read against the hardware the code
was actually tuned on:

  - relaxation, W7-X (8,16,8) p=3 float64: setup ~330 s, first step ~90 s of
    compilation, then 0.7-0.9 s/step   (docs/source/concepts/relaxation.md)
  - relaxation, W7-X (12,24,12): 0.36-0.41 s/step
    (docs/research/release_review_sweep_2026-08-27.md)
  - compute_nullspaces gap sweeps, W7-X (12,24,12) p=3: ~17 s
    (mrx/nullspace.py)
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
    "nullspace_gap_sweeps_5": 17.0,
    "relax_step": 0.41,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    ap.add_argument("--out", default=None, help="write results as JSON here")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="read result JSONs and print the comparison table")
    ap.add_argument("--skip-phases", action="store_true",
                    help="primitives only; skips the multi-minute phase timings")
    ap.add_argument("--skip-relax", action="store_true",
                    help="skip the relaxation phase (the slowest item)")
    ap.add_argument("--gap-sweeps", type=int, default=5,
                    help="sweeps for the nullspace diagnostic comparison")
    ap.add_argument("--profile", default=None,
                    help="write a jax.profiler trace to this directory")
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

def bench_scatter_ab(bench, n_out, n_nz, dtype):
    """A/B the four ways to apply a fixed sparse operator.

    This is the direct test of the leading hypothesis. ``segment_sum`` with
    unsorted indices lowers to a general scatter-add. GPUs implement that with
    hardware atomics and it is cheap; TPUs have no fast atomic scatter, so XLA
    serialises or sorts. If the TPU column here is far worse than the CPU one
    while the gather and matmul rows are not, the diagnosis is settled and the
    fix is to change how the operator is applied, not to change the algorithm.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(0)
    seg_unsorted = rng.integers(0, n_out, size=n_nz).astype(np.int32)
    seg_sorted = np.sort(seg_unsorted)
    gather_idx = rng.integers(0, n_out, size=n_nz).astype(np.int32)
    vals = rng.standard_normal(n_nz).astype(dtype)
    x = jnp.asarray(rng.standard_normal(n_out).astype(dtype))

    seg_u = jnp.asarray(seg_unsorted)
    seg_s = jnp.asarray(seg_sorted)
    gid = jnp.asarray(gather_idx)
    v = jnp.asarray(vals)

    print(f"\n[primitive] sparse apply A/B  (n_out={n_out}, nnz={n_nz})", flush=True)

    # Floor first. A single fused multiply-add on 8 floats does no meaningful
    # work, so whatever it costs is the price of dispatching one XLA program to
    # this device. Every row below has to be read against this number: if they
    # all land near it, the kernels are irrelevant and the code is paying for
    # having too many separate device calls.
    tiny = jnp.asarray(np.arange(8, dtype=dtype))
    f_noop = jax.jit(lambda t: t * 1.0001 + 1.0)
    bench.measure("dispatch floor (8-element a*x+b)", lambda: f_noop(tiny),
                  inner=MICRO_INNER, note="cost of one device call")

    # Same total scatter work, but FUSED: one dispatch instead of FUSE_N.
    # If the per-scatter cost collapses here, the op is dispatch bound and the
    # fix is to jit larger regions, not to change the kernel.
    FUSE_N = 20

    def chained(x):
        for _ in range(FUSE_N):
            x = jax.ops.segment_sum(v * x[gid], seg_u, num_segments=n_out)
            x = x / (1.0 + jnp.abs(x).max())
        return x

    f_chained = jax.jit(chained)
    times = bench.measure(f"{FUSE_N} scatters fused in one jit",
                          lambda: f_chained(x), inner=5,
                          note="one dispatch, FUSE_N kernels")
    if times:
        per = min(times[1:]) / FUSE_N
        bench.rows["scatter: fused, per scatter"] = {
            "first_s": None, "steady_s": per, "compile_s": None,
            "inner": 1, "note": f"same work, 1/{FUSE_N} of the dispatches"}
        print(f"  {'-> per scatter inside the fusion':<38}"
              f"{'':>16}steady {per * 1e3:9.3f}ms", flush=True)

    f_unsorted = jax.jit(lambda x: jax.ops.segment_sum(
        v * x[gid], seg_u, num_segments=n_out))
    bench.measure("scatter: segment_sum unsorted", lambda: f_unsorted(x),
                  inner=MICRO_INNER, note="what mrx does today")

    f_sorted = jax.jit(lambda x: jax.ops.segment_sum(
        v * x[gid], seg_s, num_segments=n_out, indices_are_sorted=True))
    bench.measure("scatter: segment_sum sorted", lambda: f_sorted(x),
                  inner=MICRO_INNER, note="hint only, indices presorted")

    # The honest cost of the fix where the indices are NOT already sorted:
    # sorting them at construction means permuting the contributions at every
    # apply, so the gather has to be inside the measurement. Without this row
    # the "sorted" number above would flatter a fix that does not exist.
    perm = jnp.asarray(np.argsort(seg_unsorted).astype(np.int32))
    f_permuted = jax.jit(lambda x: jax.ops.segment_sum(
        (v * x[gid])[perm], seg_s, num_segments=n_out,
        indices_are_sorted=True))
    bench.measure("scatter: permute + sorted segment_sum",
                  lambda: f_permuted(x), inner=MICRO_INNER,
                  note="tier-1 fix, full cost")

    f_gather = jax.jit(lambda x: v * x[gid])
    bench.measure("gather only (no scatter)", lambda: f_gather(x),
                  inner=MICRO_INNER, note="lower bound")

    # Dense equivalent. n_out^2 floats: only meaningful while that fits.
    if n_out <= 12000:
        dense = np.zeros((n_out, n_out), dtype=dtype)
        np.add.at(dense, (seg_unsorted, gather_idx), vals)
        dmat = jnp.asarray(dense)
        f_dense = jax.jit(lambda x: dmat @ x)
        bench.measure("dense matmul equivalent", lambda: f_dense(x),
                      inner=MICRO_INNER,
                      note=f"{dense.nbytes / 1e6:.0f} MB resident")


def bench_structured_scatter(bench, ne=(12, 24, 12), nloc=(4, 4, 4),
                             dtype="float32"):
    """Compare the mass kernel's scatter against a shift-and-add equivalent.

    ``_sumfact_kernel`` finishes with a ``segment_sum`` whose segment ids come
    from ``_flat_dof_plan``, which is
    ``gx * (Sy * Sz) + gy * Sz + gz`` -- a *separable tensor product* of the
    per-axis global DoF ids. For a periodic B-spline axis ``gx[e, l] = e + l``
    (mod n), so the whole scatter is algebraically

        out[i,j,k] = sum over local (lx,ly,lz) of contrib[i-lx, j-ly, k-lz, ...]

    i.e. a sum of ``prod(nloc)`` shifted dense arrays. That formulation has no
    indexed writes at all, which is the property a TPU cares about. This row
    prices the swap before anyone commits to writing it.
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

    # The index plan exactly as _flat_dof_plan builds it, periodic.
    gx = (np.arange(nex)[:, None] + np.arange(nlx)[None, :]) % nex
    gy = (np.arange(ney)[:, None] + np.arange(nly)[None, :]) % ney
    gz = (np.arange(nez)[:, None] + np.arange(nlz)[None, :]) % nez
    seg = (gx[:, None, None, :, None, None] * (ney * nez)
           + gy[None, :, None, None, :, None] * nez
           + gz[None, None, :, None, None, :]).astype(np.int32).reshape(-1)
    seg_j = jnp.asarray(seg)

    f_indexed = jax.jit(lambda c: jax.ops.segment_sum(
        c.reshape(-1), seg_j, num_segments=n_out))
    bench.measure("mass scatter: segment_sum (current)",
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
    bench.measure("mass scatter: shift-and-add (proposed)",
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


def bench_recompilation(bench, seq, ops, dtype):
    """Test whether the inner solve is recompiled on every single call.

    ``apply_laplacian`` for k>=1 runs a nested CG as an eager
    ``jax.lax.while_loop``: nothing wraps it in ``jax.jit``, so each call
    traces a fresh closure, which XLA sees as a new program. On a CPU that is
    an annoyance. On a TPU, where compiling is far more expensive, it can
    dominate everything else.

    The signature is unmistakable: repeated identical calls that never get
    faster. This prints each call individually rather than a summary, because
    the *shape* of the sequence is the evidence. Then it tries the two fixes --
    wrapping the call in ``jax.jit``, and the persistent compilation cache --
    and reports which one actually removes the cost.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mrx.operators import apply_laplacian

    print("\n[experiment] is the inner solve recompiled every call?",
          flush=True)
    rng = np.random.default_rng(31)
    k = 1
    n_in = int(seq.n(k, False))
    vec = jnp.asarray(rng.standard_normal(n_in).astype(dtype))

    per_call = []
    for i in range(4):
        t0 = time.perf_counter()
        _sync(apply_laplacian(seq, ops, vec, k, dirichlet=False))
        dt = time.perf_counter() - t0
        per_call.append(dt)
        print(f"  eager call {i + 1}: {dt:8.3f}s", flush=True)

    flat = max(per_call) / max(min(per_call), 1e-12) < 2.0
    print(f"  -> per-call cost is {'flat' if flat else 'falling'} "
          f"(max/min = {max(per_call) / max(min(per_call), 1e-12):.2f}); "
          f"compare against the jitted steady time below to see how much of "
          f"it is compilation", flush=True)
    bench.rows["laplacian_eager_calls"] = {
        "first_s": per_call[0], "steady_s": min(per_call[1:]),
        "compile_s": None,
        "note": "calls=" + ",".join(f"{t:.3f}" for t in per_call)}

    # Fix candidate: hoist the whole thing into one jitted program, so the
    # while_loop is compiled once and the second call is pure execution.
    jitted = jax.jit(lambda v: apply_laplacian(seq, ops, v, k, dirichlet=False))
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        _sync(jitted(vec))
        times.append(time.perf_counter() - t0)
        print(f"  jitted call {i + 1}: {times[-1]:8.3f}s", flush=True)
    steady = min(times[1:])
    bench.rows["laplacian_jitted"] = {
        "first_s": times[0], "steady_s": steady, "compile_s": times[0] - steady,
        "note": f"speedup vs eager {min(per_call) / max(steady, 1e-12):.1f}x"}
    print(f"  -> jitted steady {steady * 1e3:.1f}ms, "
          f"{min(per_call) / max(steady, 1e-12):.1f}x faster than eager",
          flush=True)


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
    # solve as a lax.while_loop, which is where the nullspace time goes.
    for k, dbc in ((0, False), (1, False)):
        try:
            n_in = int(seq.n(k, dbc))
        except Exception:                                 # noqa: BLE001
            continue
        vec = jnp.asarray(rng.standard_normal(n_in).astype(dtype))
        bench.measure(
            f"apply_laplacian k={k} free",
            lambda k=k, dbc=dbc, v=vec: apply_laplacian(
                seq, ops, v, k, dirichlet=dbc),
            repeats=2,
            note=f"n={n_in}" + (" (nested CG)" if k >= 1 else ""))


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

    # gap_sweeps=0 is the construction alone; gap_sweeps=N adds the diagnostic
    # inverse iteration. The difference is what the diagnostic actually costs,
    # and on the host CPU it was 6.8 of 9 minutes.
    t0 = time.perf_counter()
    ops0 = compute_nullspaces(seq, ops, gap_sweeps=0, verbose=False)
    gap0_s = time.perf_counter() - t0
    bench.rows["nullspace_gap_sweeps_0"] = {
        "first_s": gap0_s, "steady_s": None, "compile_s": None,
        "note": "construction only"}
    print(f"  {'compute_nullspaces gap_sweeps=0':<38} {gap0_s:9.3f}s"
          f"   construction only", flush=True)

    if args.gap_sweeps:
        t0 = time.perf_counter()
        compute_nullspaces(seq, ops, gap_sweeps=args.gap_sweeps, verbose=True)
        gapn_s = time.perf_counter() - t0
        bench.rows[f"nullspace_gap_sweeps_{args.gap_sweeps}"] = {
            "first_s": gapn_s, "steady_s": None, "compile_s": None,
            "note": f"diagnostic costs {gapn_s - gap0_s:.1f}s on top"}
        print(f"  {'compute_nullspaces gap_sweeps=' + str(args.gap_sweeps):<38}"
              f" {gapn_s:9.3f}s   diagnostic adds {gapn_s - gap0_s:.1f}s",
              flush=True)

    seq.set_operators(ops0)
    return seq, ops0


def bench_relaxation(bench, seq, args, dtype):
    """Time the relaxation loop: compile cost, then cost per step."""
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import clebsch_potential_form, potential_two_form
    from mrx.relaxation import (DescentMethod, TimeStepChoice, TimeStepper,
                                relaxation_loop)

    ns = tuple(int(v) for v in args.ns.split(","))
    print("\n[phase] relaxation", flush=True)

    cb = load_clebsch(args.geometry)
    B0, _, _ = potential_two_form(seq, clebsch_potential_form(cb))

    ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                     dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                     eta_every=1, resistive=False, history_size=1,
                     velocity_smoothing_order=1,
                     velocity_smoothing_scale=0.064 / ns[0] ** 2)

    inner = 5
    # Two identical calls. The first compiles the scanned step; the second does
    # not, so (first - second) / inner is the compile cost amortised per step
    # and second / inner is the true per-step cost.
    t0 = time.perf_counter()
    state, _ = relaxation_loop(B0, ts, num_iters_outer=1,
                               num_iters_inner=inner, dt0=1.0,
                               force_tolerance=0.0)
    _sync(state.B_n)
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    state2, _ = relaxation_loop(B0, ts, num_iters_outer=1,
                                num_iters_inner=inner, dt0=1.0,
                                force_tolerance=0.0)
    _sync(state2.B_n)
    second_s = time.perf_counter() - t0

    bench.rows["relax_compile"] = {
        "first_s": first_s, "steady_s": second_s,
        "compile_s": first_s - second_s,
        "note": f"{inner} inner steps"}
    bench.rows["relax_step"] = {
        "first_s": first_s / inner, "steady_s": second_s / inner,
        "compile_s": (first_s - second_s) / inner,
        "note": "per step"}
    print(f"  {'relaxation ' + str(inner) + ' steps':<38} first {first_s:9.3f}s"
          f"   steady {second_s:9.4f}s", flush=True)
    print(f"  {'-> per step':<38} first {first_s / inner:9.3f}s"
          f"   steady {second_s / inner:9.4f}s", flush=True)


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

    info = backend_info()
    print("=" * 72)
    print("MRX phase and primitive benchmark")
    print("=" * 72)
    for key, value in info.items():
        print(f"  {key:<20} {value}")
    print(f"  {'mrx_dtype':<20} {mrx.DTYPE}")
    print(f"  {'matmul_precision':<20} {jax.config.jax_default_matmul_precision}")

    dtype = str(mrx.DTYPE)
    bench = Bench()

    profile_ctx = None
    if args.profile:
        os.makedirs(args.profile, exist_ok=True)
        profile_ctx = jax.profiler.trace(args.profile)
        profile_ctx.__enter__()

    try:
        # Primitives first: they are seconds each and they alone can settle the
        # scatter question, so they must not be gated behind a phase that may
        # run for half an hour.
        bench_scatter_ab(bench, n_out=8700, n_nz=40000, dtype=dtype)
        bench_structured_scatter(bench, dtype=dtype)

        if not args.skip_phases:
            seq, ops = bench_phases(bench, args, dtype)
            bench_extraction(bench, seq, dtype)
            bench_mass(bench, seq, dtype)
            bench_operators(bench, seq, ops, dtype)
            bench_recompilation(bench, seq, ops, dtype)
            if not args.skip_relax:
                bench_relaxation(bench, seq, args, dtype)
    finally:
        if profile_ctx is not None:
            profile_ctx.__exit__(None, None, None)
            print(f"\nprofile trace written to {args.profile}")

    result = {"backend": info, "mrx_dtype": dtype, "args": vars(args),
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
