"""Indexed assembly against the shift assembly, on the production mass kernel.

``_structured_gather`` and ``_structured_accumulate`` replace one indexed
read or write with about a dozen dense shifts, because a TPU has no fast
path for indexed memory. A backend that charges per dispatch wants the
opposite trade. This measures both, on a real :class:`mrx.mass.SumfactPlan`
rather than a synthetic one, and it times them inside a jitted ``lax.scan``
so the number is the cost inside a solver rather than the cost of one
Python call.

Three forms, same semantics (``x_local[e, l] = x[(e + l) mod S]`` and its
adjoint):

    shift            the production helpers
    indexed_peraxis  one gather or segment_sum per axis
    indexed_flat     one gather or segment_sum for the whole element

The decision was fixed before looking: an indexed form is worth landing
only if it beats ``shift`` on the full mass apply, on MPS, by more than the
17% run-to-run spread already measured for a fixed configuration.

    python mps/assembly_ab.py                  # cpu then mps
    python mps/assembly_ab.py --backends mps
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Callable, Sequence

os.environ.setdefault("MRX_X64", "0")

#: Applies chained inside one scan. Dependent, so the compiler cannot fuse
#: them into a single batched kernel; long enough that one launch is noise.
SCAN = 20

#: Best of this many scans.
REPEATS = 3

#: Indexed must be faster than shift by more than this, or it is noise.
NOISE = 0.17


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--backends", default="cpu,mps")
    ap.add_argument("--scan", type=int, default=SCAN)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    return ap.parse_args(argv)


def _axis_index(ne: int, nl: int, size: int):
    """``(e + l) mod size`` as an ``(ne, nl)`` int32 array."""
    import numpy as np

    e = np.arange(ne)[:, None]
    local = np.arange(nl)[None, :]
    return ((e + local) % size).astype(np.int32)


def _flat_index(plan: tuple):
    """Flat C-order index of every element-local read, shape of the local array.

    ``plan`` is one component's ``((ne, nl, S),) * 3``. Entry
    ``[ex, ey, ez, lx, ly, lz]`` is the flat index of DoF
    ``((ex+lx) mod Sx, (ey+ly) mod Sy, (ez+lz) mod Sz)``.
    """
    import numpy as np

    (nex, nlx, sx), (ney, nly, sy), (nez, nlz, sz) = plan
    ix = _axis_index(nex, nlx, sx)[:, None, None, :, None, None]
    iy = _axis_index(ney, nly, sy)[None, :, None, None, :, None]
    iz = _axis_index(nez, nlz, sz)[None, None, :, None, None, :]
    return (ix * (sy * sz) + iy * sz + iz).astype(np.int32)


def _variants(plans: Sequence[tuple]) -> dict[str, tuple[Callable, Callable]]:
    """``{name: (gather, accumulate)}`` closed over ``plans``' index tables.

    The index arrays are built with NumPy here, outside the kernel, so a
    trace bakes them in as constants. That is the form a landed kernel would
    take: the shift plan is a static argument, and the indices are a pure
    function of it.
    """
    import jax
    import jax.numpy as jnp

    from mrx.mass import _structured_accumulate, _structured_gather

    flat = {p: jnp.asarray(_flat_index(p)) for p in plans}
    per = {p: tuple(jnp.asarray(_axis_index(ne, nl, s)) for ne, nl, s in p)
           for p in plans}

    def gather_shift(x, plan):
        return _structured_gather(x, plan)

    def accumulate_shift(y, plan):
        return _structured_accumulate(y, plan)

    def gather_flat(x, plan):
        return x[flat[plan]]

    def accumulate_flat(y, plan):
        (__, __, sx), (__, __, sy), (__, __, sz) = plan
        seg = flat[plan]
        out = jax.ops.segment_sum(y.reshape(-1), seg.reshape(-1),
                                  num_segments=sx * sy * sz)
        return out.reshape(sx, sy, sz)

    def gather_peraxis(x, plan):
        (nex, nlx, sx), (ney, nly, sy), (nez, nlz, sz) = plan
        ix, iy, iz = per[plan]
        a = x.reshape(sx, sy, sz)
        a = jnp.moveaxis(jnp.take(a, ix, axis=0), 1, -1)
        a = jnp.moveaxis(jnp.take(a, iy, axis=1), 2, -1)
        a = jnp.moveaxis(jnp.take(a, iz, axis=2), 3, -1)
        return a

    def accumulate_peraxis(y, plan):
        (nex, nlx, sx), (ney, nly, sy), (nez, nlz, sz) = plan
        ix, iy, iz = per[plan]
        # z, then y, then x: each segment_sum consumes one local axis.
        a = jnp.moveaxis(y, (2, 5), (0, 1)).reshape(nez * nlz, -1)
        a = jax.ops.segment_sum(a, iz.reshape(-1), num_segments=sz)
        a = a.reshape(sz, nex, ney, nlx, nly).transpose(1, 2, 0, 3, 4)
        a = jnp.moveaxis(a, (1, 4), (0, 1)).reshape(ney * nly, -1)
        a = jax.ops.segment_sum(a, iy.reshape(-1), num_segments=sy)
        a = a.reshape(sy, nex, sz, nlx).transpose(1, 0, 2, 3)
        a = jnp.moveaxis(a, (0, 3), (0, 1)).reshape(nex * nlx, -1)
        a = jax.ops.segment_sum(a, ix.reshape(-1), num_segments=sx)
        return a.reshape(sx, sy, sz)

    return {
        "shift": (gather_shift, accumulate_shift),
        "indexed_peraxis": (gather_peraxis, accumulate_peraxis),
        "indexed_flat": (gather_flat, accumulate_flat),
    }


def _kernel(gather: Callable, accumulate: Callable, plan, weights) -> Callable:
    """The production sum-factorised apply with the assembly pair substituted.

    The body is :func:`mrx.mass._sumfact_kernel`. Only the read and the
    assembly differ between variants, which is the whole experiment.
    """
    import jax.numpy as jnp

    from mrx.mass import _from_quadrature, _to_quadrature

    b_r, b_c = plan.Bvals_r, plan.Bvals_c
    pairs, cols = plan.pairs, plan.cols
    starts, shifts, gathers = plan.starts_c, plan.shift_plans, plan.gather_plans

    def kernel(x):
        w = {pair: weights[c] for pair, c in zip(pairs, cols)}
        n_c, n_r = len(b_c), len(b_r)
        u = [_to_quadrature(b_c[c], gather(x[starts[c]:starts[c + 1]], gathers[c]))
             for c in range(n_c)]
        parts = []
        for cr in range(n_r):
            v = sum(w[(cr, cc)] * u[cc] for cc in range(n_c) if (cr, cc) in pairs)
            local = _from_quadrature(b_r[cr], v)
            parts.append(accumulate(local, shifts[cr]).reshape(-1))
        return jnp.concatenate(parts)

    return kernel


def _ops(fn: Callable, x) -> dict[str, int]:
    """Opcode counts in the lowered module of ``fn`` at ``x``."""
    import jax

    text = str(jax.jit(fn).lower(x).as_text())
    found = re.findall(r"\b(?:stablehlo|mhlo)\.([a-z_0-9]+)", text)
    counts: dict[str, int] = {}
    for name in found:
        counts[name] = counts.get(name, 0) + 1
    return counts


def _scan_ms(fn: Callable, x, length: int, repeats: int) -> float:
    """Milliseconds per apply of ``fn`` chained inside one jitted scan.

    The carry is ``0.5 * fn(carry)``: dependent, so the applies stay
    separate kernels, and bounded, so thirty of them cannot overflow.
    """
    import jax

    @jax.jit
    def scanned(v):
        def body(c, _):
            return 0.5 * fn(c), None
        out, _ = jax.lax.scan(body, v, None, length=length)
        return out

    jax.block_until_ready(scanned(x))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(scanned(x))
        best = min(best, time.perf_counter() - t0)
    return 1e3 * best / length


def _rel(a, b) -> float:
    """Max absolute difference over the max absolute value of ``b``."""
    import numpy as np

    denom = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / denom)


def run(geometry: str, ns: tuple[int, int, int], p: int,
        length: int, repeats: int) -> dict:
    """Build one sequence and compare the three assembly forms on it.

    Args:
        geometry: VMEC wout the sequence is built on.
        ns: Resolution.
        p: Spline degree.
        length: Applies per timed scan.
        repeats: Scans; the fastest is kept.

    Returns:
        Per degree and per variant, the in-scan milliseconds, the
        disagreement with the production apply, and the lowered op counts.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mrx.geometry import build_sequence
    from mrx.mass import sumfact_apply

    print("setup", flush=True)
    t0 = time.perf_counter()
    seq, _ = build_sequence(geometry, ns, p)
    print(f"  built in {time.perf_counter() - t0:.0f} s", flush=True)

    out: dict[str, dict] = {}
    for k in (1, 2):
        plan = seq.mass_plan[k]
        weights = seq.geometry.mass_weights[k]
        n = int(seq.E(k, False).forward_shape[1])
        rng = np.random.default_rng(100 + k)
        x = jnp.asarray(rng.standard_normal(n).astype("float32"))
        ref = np.asarray(sumfact_apply(plan, weights, x))

        plans = tuple(dict.fromkeys((*plan.gather_plans, *plan.shift_plans)))
        variants = _variants(plans)
        out[str(k)] = {}
        print(f"\n=== mass_core k={k}  n={n} ===", flush=True)
        for name, (gather, accumulate) in variants.items():
            fn = _kernel(gather, accumulate, plan, weights)
            got = np.asarray(jax.jit(fn)(x))
            err = _rel(got, ref)
            ops = _ops(fn, x)
            ms = _scan_ms(fn, x, length, repeats)
            interesting = {op: ops[op] for op in
                           ("gather", "scatter", "slice", "pad", "concatenate",
                            "reshape", "dot_general", "add", "dynamic_update_slice")
                           if op in ops}
            out[str(k)][name] = {"ms": ms, "err": err, "ops": interesting,
                                 "ops_total": sum(ops.values())}
            flag = "OK" if err < 1e-5 else "MISMATCH"
            print(f"  {name:<18} {ms:8.3f} ms   err {err:.2e} {flag}"
                  f"   ops {interesting}", flush=True)
            if err >= 1e-5:
                raise SystemExit(f"{name} disagrees with the production apply")
    return out


def _report(results: dict[str, dict]) -> None:
    """Print the MPS decision against the precommitted 17% rule."""
    print("\n=== ms per apply, inside a scan ===")
    for plat, by_k in results.items():
        print(f"\n{plat}")
        for k, variants in by_k.items():
            base = variants["shift"]["ms"]
            for name, row in variants.items():
                ratio = base / row["ms"]
                print(f"  k={k} {name:<18} {row['ms']:8.3f} ms   "
                      f"{ratio:5.2f}x vs shift   err {row['err']:.1e}")

    mps = results.get("mps")
    if mps is None:
        return
    print("\n=== decision (indexed must beat shift by more than "
          f"{NOISE:.0%} on MPS) ===")
    for k, variants in mps.items():
        base = variants["shift"]["ms"]
        for name, row in variants.items():
            if name == "shift":
                continue
            gain = base / row["ms"] - 1.0
            verdict = "LANDS" if gain > NOISE else "does not clear the noise"
            print(f"  k={k} {name:<18} {gain:+5.0%}  {verdict}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run every backend in its own process and apply the decision rule.

    Returns:
        0, or 1 if a backend produced no result.
    """
    import subprocess
    import sys

    args = parse_args(argv)
    if os.environ.get("MPS_AB_CHILD"):
        ns = tuple(int(v) for v in args.ns.split(","))
        print("RESULT " + json.dumps(
            run(args.geometry, ns, args.p, args.scan, args.repeats)))
        return 0

    results: dict[str, dict] = {}
    for plat in args.backends.split(","):
        child = {**os.environ, "MPS_AB_CHILD": "1", "JAX_PLATFORMS": plat,
                 "MRX_X64": "0"}
        done = subprocess.run(
            [sys.executable, "-u", __file__, *(argv or sys.argv[1:])],
            env=child, capture_output=True, text=True)
        for line in done.stdout.splitlines():
            if line.startswith("RESULT "):
                results[plat] = json.loads(line[7:])
            else:
                print(line, flush=True)
        if plat not in results:
            print(f"{plat} produced no result:\n{done.stderr[-2000:]}", flush=True)
    if not results:
        return 1
    _report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
