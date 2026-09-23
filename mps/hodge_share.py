"""What share of one L-BFGS step the k>=1 Hodge solves take, on this backend.

One eager step records every right-hand side (and warm start) that the step
hands to :func:`mrx.operators.apply_inverse_laplacian_hodge`. Then the
compiled step and each recorded solve, compiled on its own with those
arguments, are timed best of three. The share is the solves' sum over the
step. It is an estimate: inside the step the solves share a program with the
rest, and here each is its own launch.

    MRX_X64=0 JAX_PLATFORMS=mps python mps/hodge_share.py [R,T,Z [p [bands [passes]]]]

``bands`` rebuilds only the k=1 Dirichlet Laplacian atom (the hat solve's)
with that many radial bands. ``passes`` is ``TimeStepper.force_hodge_passes``
(0 the solve's own cap; omitted, the stepper's default).
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MRX_X64", "0")


def _best(fn, *args, reps: int = 3) -> float:
    import jax

    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    import jax

    import mrx.operators as operators
    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state

    ns = tuple(int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "12,24,12").split(","))
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    bands = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    passes = int(sys.argv[4]) if len(sys.argv) > 4 else None
    seq, _ = build_sequence("data/wout_li383_low_res_reference.nc", ns, p)
    if bands > 1:
        seq.operators = operators.assemble_metric_lumping_laplacian_preconditioner(
            seq, seq.operators, ks=(1,), dirichlets=(True,), bands=bands)
    compute_nullspaces(seq, gap_sweeps=0, verbose=False)
    B0, _ = initial_field(seq)
    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1, velocity_smoothing_order=1,
                     force_hodge_passes=passes)
    print(f"li383 {ns} p={p} k=1 Dirichlet bands={bands} force_hodge_passes="
          f"{ts.force_hodge_passes}  devices={jax.devices()}", flush=True)
    state = initial_state(B0, ts, 1.0)

    real = operators.apply_inverse_laplacian_hodge
    calls = []

    def recording(seq_, ops_, rhs, k, dirichlet=True, guess=None, **kw):
        calls.append((k, dirichlet, rhs, guess, kw))
        return real(seq_, ops_, rhs, k, dirichlet=dirichlet, guess=guess, **kw)

    operators.apply_inverse_laplacian_hodge = recording
    try:
        ts.relaxation_step(state)
    finally:
        operators.apply_inverse_laplacian_hodge = real
    if any(isinstance(c[2], jax.core.Tracer) for c in calls):
        raise RuntimeError("a Hodge solve was traced inside a jit; its rhs is not concrete")
    print(f"  Hodge solves in one step: {len(calls)} "
          f"(k={[c[0] for c in calls]})", flush=True)

    step = jax.jit(lambda s: ts.relaxation_step(s))
    t_step = _best(step, state)
    t_hodge = 0.0
    for k, d, rhs, guess, kw in calls:
        kw = {kk: v for kk, v in kw.items() if kk in ("tol", "maxiter", "dtype", "max_passes")}
        solve = jax.jit(lambda r, g, k=k, d=d, kw=kw: real(seq, seq.operators, r, k, dirichlet=d,
                                                            guess=g, **kw))
        t = _best(solve, rhs, guess)
        t_hodge += t
        print(f"  k={k} dirichlet={d}: {t * 1e3:8.1f} ms", flush=True)
    print(f"\n  step {t_step * 1e3:.0f} ms, Hodge solves {t_hodge * 1e3:.0f} ms, "
          f"share {t_hodge / t_step:.0%}", flush=True)


if __name__ == "__main__":
    main()
