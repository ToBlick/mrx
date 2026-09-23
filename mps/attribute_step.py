"""Where one relaxation step's time goes, operator by operator.

A compiled step is one program, so a profiler cannot see ``apply_mass_matrix``
inside it. What it can see is how many times each leaf runs, because a
``jax.debug.callback`` fires on the host from inside a jitted solver, and the
per-apply cost of those leaves was already measured at this mesh
(``outputs/mps_bench/mps.json``, li383 ``(12,24,12)`` p=3, float32, MPS).

The product of the two is a ranking, not the step's wall clock: the costs
below are one leaf inside a scan of 40, so they do not pay a dispatch per
call the way a standalone timing does. It bounds what fixing the mass kernel
can return.

    MRX_X64=0 JAX_PLATFORMS=mps python mps/attribute_step.py
"""

from __future__ import annotations

import os
import time
from collections import Counter

os.environ.setdefault("MRX_X64", "0")

#: Indexed in-scan cost on MPS at this mesh, milliseconds. li383
#: ``(12, 24, 12)`` p=3, float32, one leaf inside a scan of 40, measured
#: 2026-09-22 with the machine idle. The extraction row is the k=1 Dirichlet
#: apply; the other degrees were 0.058-0.063 ms.
COST_MS: dict[str, float] = {
    "mass_core[0]": 0.220,
    "mass_core[1]": 0.560,
    "mass_core[2]": 0.512,
    "mass_core[3]": 0.158,
    "extraction": 0.066,
}

#: Calls per scan when timing a leaf. Long enough that one launch is noise.
SCAN = 40


def _scan_ms(fn, arg) -> float:
    """Milliseconds per call of ``fn`` inside one jitted scan.

    The carry is the input, unchanged, and each call's result is reduced to a
    scalar the scan returns, so the work stays live and a rectangular ``fn``
    (the incidence stencils) is timed the same way as a square one. The
    scan's own launch is spread over :data:`SCAN` calls.
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def scanned(x):
        def body(c, _):
            return c, jnp.sum(fn(c))
        _, out = jax.lax.scan(body, x, None, length=SCAN)
        return out

    jax.block_until_ready(scanned(arg))
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(scanned(arg))
        best = min(best, time.perf_counter() - t0)
    return 1e3 * best / SCAN


def _instrument() -> Counter:
    """Count leaf calls. Returns the counter the callbacks increment."""
    import jax
    import jax.numpy as jnp

    import mrx.extraction_operators as extraction
    import mrx.metric_lumping_laplacian as lump
    import mrx.operators as operators
    import mrx.quadrature as quadrature

    counts: Counter = Counter()

    def bump(code: int) -> None:
        counts[int(code)] += 1

    orig_mass = operators.mass_core_apply

    def mass_core_apply(seq, k: int):
        fn = orig_mass(seq, k)

        def counted(x):
            jax.debug.callback(bump, jnp.int32(k))
            return fn(x)
        return counted

    operators.mass_core_apply = mass_core_apply

    orig_inc = operators._apply_incidence_mf

    def apply_incidence(op, x):
        code = 10 + op.k + (3 if op.transpose else 0)
        jax.debug.callback(bump, jnp.int32(code))
        return orig_inc(op, x)

    operators._apply_incidence_mf = apply_incidence

    orig_ext = extraction.MatrixFreeExtraction._apply

    def apply_ext(self, x):
        jax.debug.callback(bump, jnp.int32(30))
        return orig_ext(self, x)

    extraction.MatrixFreeExtraction._apply = apply_ext

    def _wrap_apply(cls, code: int, method: str = "apply") -> None:
        orig = getattr(cls, method)

        def wrapped(self, *args, **kwargs):
            fn = orig(self, *args, **kwargs)
            token = jnp.int32(code)
            if method == "apply":
                jax.debug.callback(bump, token)
                return fn

            def counted(x):
                jax.debug.callback(bump, token)
                return fn(x)
            return counted
        setattr(cls, method, wrapped)

    _wrap_apply(lump.MetricLumpingMass, 31)
    _wrap_apply(lump.MetricLumpingLaplacian, 32)
    _wrap_apply(lump.MetricLumpingLaplacian, 33, "shifted_stiffness_apply")

    orig_eval = quadrature.evaluate_at_xq

    def evaluate_at_xq(*args, **kwargs):
        jax.debug.callback(bump, jnp.int32(34))
        return orig_eval(*args, **kwargs)

    quadrature.evaluate_at_xq = evaluate_at_xq

    orig_int = quadrature.integrate_against

    def integrate_against(*args, **kwargs):
        jax.debug.callback(bump, jnp.int32(35))
        return orig_int(*args, **kwargs)

    quadrature.integrate_against = integrate_against
    return counts


def _leaf_costs(seq, ops) -> dict[str, float]:
    """In-scan milliseconds per leaf, the cost inside the relaxation.

    :data:`COST_MS` is this measurement for the mass kernel and the
    extraction, kept as the price model. The rest is printed and not priced.
    """
    import jax.numpy as jnp

    import mrx.operators as operators

    out: dict[str, float] = {}
    for k in (1, 2):
        n_ext = int(seq.n(k, True))
        n_raw = int(seq.E(k, False).forward_shape[1])
        core = operators.mass_core_apply(seq, k)
        out[f"mass_core[{k}] in-scan"] = _scan_ms(core, jnp.ones(n_raw))
        g, g_T = operators._incidence_components(seq, k)
        out[f"incidence[{k}] in-scan"] = _scan_ms(
            lambda x, g=g: g @ x, jnp.ones(int(g.shape[1])))
        out[f"incidence_T[{k}] in-scan"] = _scan_ms(
            lambda x, g=g_T: g @ x, jnp.ones(int(g_T.shape[1])))
        out[f"precond_mass[{k}] in-scan"] = _scan_ms(
            ops.mass_lumping[(k, True)].apply, jnp.ones(n_ext))
        out[f"precond_laplacian[{k}] in-scan"] = _scan_ms(
            ops.laplacian_lumping[(k, True)].apply, jnp.ones(n_ext))
    return out


def main() -> None:
    """Count one step and price it from the measured per-apply costs.

    ``--counts-only`` skips the in-scan leaf timings. ``jax.debug.callback``
    needs a CPU device, which ``JAX_PLATFORMS=mps`` hides, so the count is
    taken with the CPU backend available; the call counts are the
    algorithm's, not the backend's.
    """
    import sys

    import jax

    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state

    counts_only = "--counts-only" in sys.argv
    newton = "--newton" in sys.argv
    print(f"setup  devices={jax.devices()}  newton={newton}", flush=True)
    t0 = time.perf_counter()
    seq, ops = build_sequence("data/wout_li383_low_res_reference.nc", (12, 24, 12), 3)
    compute_nullspaces(seq, gap_sweeps=0, verbose=False)
    print(f"  built in {time.perf_counter() - t0:.0f} s", flush=True)

    if not counts_only:
        print("\nleaf costs inside a scan", flush=True)
        for name, ms in _leaf_costs(seq, ops).items():
            print(f"  {name:<32} {ms:8.3f} ms", flush=True)

    counts = _instrument()
    B0, _ = initial_field(seq)
    if newton:
        ts = TimeStepper(seq=seq, cfl=0.5, history_size=0,
                         velocity_smoothing_order=1, velocity_smoothing_scale=None,
                         newton=True, newton_precond="laplacian", newton_dt_cap=1.0)
    else:
        ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                         velocity_smoothing_order=1, velocity_smoothing_scale=None)
    print("\nseeding the state (not counted)", flush=True)
    state = initial_state(B0, ts, 1.0)
    jax.block_until_ready(state.B_n)
    counts.clear()

    print("one relaxation step", flush=True)
    t0 = time.perf_counter()
    state = ts.relaxation_step(state)
    jax.block_until_ready(state.B_n)
    print(f"  wall {time.perf_counter() - t0:.1f} s  (eager, callbacks on; not the fused cost)",
          flush=True)

    names = {
        0: "mass_core[0]", 1: "mass_core[1]", 2: "mass_core[2]", 3: "mass_core[3]",
        10: "incidence[0]", 11: "incidence[1]", 12: "incidence[2]",
        13: "incidence_T[0]", 14: "incidence_T[1]", 15: "incidence_T[2]",
        30: "extraction",
        31: "precond_mass", 32: "precond_laplacian", 33: "precond_shifted",
        34: "quadrature_eval", 35: "quadrature_integrate",
    }
    print("\ncalls in one step")
    total = 0.0
    rows = []
    for code, name in names.items():
        n = counts[code]
        if not n:
            continue
        cost = COST_MS.get(name)
        share = n * cost if cost is not None else None
        rows.append((name, n, cost, share))
        if share is not None:
            total += share
    for name, n, cost, share in rows:
        if share is None:
            print(f"  {name:<24} {n:7d} calls")
        else:
            print(f"  {name:<24} {n:7d} calls  x {cost:6.3f} ms  = {share:8.1f} ms")
    print(f"\n  priced leaves (indexed in-scan costs): {total:.0f} ms")


if __name__ == "__main__":
    main()
