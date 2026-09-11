"""Time a vacuum solve on the current code and measure the vacuum field's force residual.

    python -u scripts/vacuum_timing.py --geometry data/wout_..._highres.nc --ns 32,64,32 --p 3 --out DIR

float64 throughout (the vacuum sweeps' precision, solve tolerance 1e-10). Stages, each timed:
the sequence build (operators and preconditioners), the harmonic forms by
:func:`mrx.nullspace.compute_nullspaces` called TWICE on the same sequence (the first call
carries every compile, the second is the solve alone), and the Leray-projected Lorentz force of
the k=2 Dirichlet harmonic form, also twice. Reports the harmonic form's Rayleigh quotient and
its force residual in the relaxation's units, ``||F||_M^2 / ||grad(B^2/2)||^2`` with ``F`` the
Leray-projected force, the same with the unprojected ``J x B``, and ``||J|| / ||B||``.
Writes ``DIR/vacuum_timing.json``."""
import argparse
import json
import os
import time

os.environ["MRX_DTYPE"] = "float64"
os.environ["MRX_RESIDUAL_DTYPE"] = "float64"

import jax  # noqa: E402

import mrx  # noqa: E402
from mrx.geometry import build_sequence  # noqa: E402
from mrx.nullspace import compute_nullspaces, harmonic_rayleigh  # noqa: E402
from mrx.relaxation import compute_force, force_scale  # noqa: E402


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--ns", required=True, help="n_r,n_theta,n_zeta")
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-batch", type=int, default=0)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    mrx.MAP_BATCH_SIZE_INNER = cli.map_batch
    os.makedirs(cli.out, exist_ok=True)
    res = dict(geometry=cli.geometry, ns=ns, p=cli.p, precision="float64", map_batch=cli.map_batch,
               jax=jax.__version__, mrx=os.path.dirname(mrx.__file__))
    print(f"[env] mrx from {mrx.__file__}  dtype {mrx.DTYPE} / {mrx.precision.RESIDUAL_DTYPE}", flush=True)

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    res["t_build"] = time.perf_counter() - t0
    res.update(n2=int(seq.n(2, True)), tol=float(seq.tol))
    print(f"[build] ns={ns} p={cli.p} n2={res['n2']} tol={seq.tol:.1e}  {res['t_build']:.1f}s", flush=True)

    for i, key in enumerate(("t_nullspace_first", "t_nullspace_second")):
        _, res[key] = timed(lambda: compute_nullspaces(seq, verbose=(i == 0)))
        h = seq.nullspace(2, True)[0]
        print(f"[nullspace] call {i + 1}: {res[key]:.1f}s", flush=True)
    res["rayleigh"] = harmonic_rayleigh(seq, h, 2, True, ops)

    B = h / seq.l2_norm(h, 2, dirichlet=True)
    scale = force_scale(seq)
    for i, key in enumerate(("t_force_first", "t_force_second")):
        (F, p, J, X, JxX), res[key] = timed(lambda: compute_force(B, seq))
        print(f"[force] call {i + 1}: {res[key]:.1f}s", flush=True)
    s = float(scale(B))
    F_norm = float(seq.l2_norm(F, 2, dirichlet=True))
    JxB_norm = float(seq.l2_norm(JxX, 2, dirichlet=True))
    res.update(scale=s, F_norm=F_norm, JxB_norm=JxB_norm,
               resid_leray=(F_norm / s) ** 2, resid_JxB=(JxB_norm / s) ** 2,
               J_over_B=float(seq.l2_norm(J, 1, dirichlet=True) / seq.l2_norm(B, 2, dirichlet=True)),
               t_total=time.perf_counter() - t0)
    print(f"[vacuum] rayleigh {res['rayleigh']:.3e}  ||F||^2/||grad(B^2/2)||^2 {res['resid_leray']:.3e}  "
          f"||JxB||^2/||grad(B^2/2)||^2 {res['resid_JxB']:.3e}  ||J||/||B|| {res['J_over_B']:.3e}  "
          f"total {res['t_total']:.1f}s", flush=True)
    with open(os.path.join(cli.out, "vacuum_timing.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
