"""One Newton relaxation from a warm-started field, for the assembly comparison.

Newton is ``scripts/relax.py``'s method outside plain float32, and it was
not what the Apple-GPU timings were taken on. A step is a 300-iteration
MINRES solve whose every matvec is two k=1 mass solves, so it spends far
more of its time in the mass kernel than the L-BFGS descent does. This runs
that step
from Tutorial 4's starting point: the field of a descent that has already
been through its fast phase, not the equilibrium initial condition.

The backend and the assembly come from the environment (``JAX_PLATFORMS``,
``MRX_ASSEMBLY``), the same way a real run selects them. ``--floor-tol 0``
is fixed: the default ``1e-8`` is a squared residual no float32 run here
reaches, so it would only look like a stopping criterion.

    MRX_X64=0 JAX_PLATFORMS=mps python mps/newton_run.py \
        --h5 data/tutorials/li383_relaxation/checkpoints/state_000500.h5 \
        --ns 10,16,16 --p 2 --steps 10 --chunk 5 --out outputs/mps_bench/newton_tut_indexed
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Sequence

os.environ.setdefault("MRX_X64", "0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line of one Newton run.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="10,16,16")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--h5", default=None,
                    help="a descent checkpoint; only its B_n is kept, which is "
                         "how Tutorial 4 starts Newton. Without one, Newton "
                         "starts from the equilibrium initial field")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--newton-tol", type=float, default=0.1)
    ap.add_argument("--newton-maxiter", type=int, default=300)
    ap.add_argument("--precond", default="laplacian",
                    choices=("laplacian", "laplacian2", "mass", "harmonic"))
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the sequence, start Newton from the checkpoint's field, record the run."""
    import h5py
    import jax
    import numpy as np

    from mrx.geometry import build_sequence
    from mrx.mass import _assembly_mode
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state, relax

    args = parse_args(argv)
    ns = tuple(int(v) for v in args.ns.split(","))
    print(f"assembly={_assembly_mode()}  device={jax.devices()[0]}", flush=True)

    t0 = time.perf_counter()
    seq, _ = build_sequence(args.geometry, ns, args.p)
    compute_nullspaces(seq, gap_sweeps=0, verbose=False)
    if args.h5:
        with h5py.File(args.h5, "r") as fh:
            B = np.asarray(fh["B_n"])
        print(f"warm start {args.h5}", flush=True)
    else:
        from mrx.initial_conditions import initial_field
        B, _ = initial_field(seq)
        B = np.asarray(B)
        print("start: equilibrium initial field", flush=True)
    print(f"setup {time.perf_counter() - t0:.0f} s, |B|={np.linalg.norm(B):.4e}", flush=True)

    # history_size 0: Newton replaces the L-BFGS direction. Smoothing stays,
    # because a direction that does not descend falls back to the smoothed force.
    ts = TimeStepper(seq=seq, cfl=0.5, history_size=0, velocity_smoothing_order=1,
                     newton=True, newton_tol=args.newton_tol,
                     newton_maxiter=args.newton_maxiter,
                     newton_precond=args.precond, newton_dt_cap=1.0)
    print(f"precond={args.precond}", flush=True)
    state = initial_state(jax.numpy.asarray(B), ts)
    result = relax(state, ts, steps=args.steps, chunk=args.chunk, floor_tol=0.0, verbose=True)

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "assembly": _assembly_mode(),
        "precond": args.precond,
        "device": str(jax.devices()[0]),
        "ns": list(ns), "p": args.p, "steps": result.steps,
        "wall": result.wall, "stop": result.stop,
        "trace": {k: [float(v) for v in vs] for k, vs in result.trace.items()},
    }
    with open(os.path.join(args.out, "newton.json"), "w") as fh:
        json.dump(payload, fh)
    it = np.asarray(result.trace["newton_it"])
    fb = np.asarray(result.trace["newton_fallback"])
    cos = np.asarray(result.trace["cos"])
    print(f"\n{result.steps} steps in {result.wall:.1f} s "
          f"({result.wall / result.steps:.2f} s/step), stop={result.stop}")
    print(f"MINRES |it| mean {np.abs(it).mean():.0f}, "
          f"at the budget on {int((it > 0).sum())}/{result.steps}, "
          f"fallbacks {int(fb.sum())}")
    dt = np.asarray(result.trace["dt_star"])
    dE = np.asarray(result.trace["dE"])
    print(f"descent cosine min {cos.min():+.4f} mean {cos.mean():+.4f}")
    print(f"dt* {dt[0]:+.4f} .. mean {dt.mean():+.4f};  "
          f"energy removed {-dE.sum():.6e}")
    print(f"wrote {args.out}/newton.json")


if __name__ == "__main__":
    main()
