"""Relax a magnetic field toward minimum energy at fixed helicity.

The command line of :func:`mrx.relaxation.relax`: builds the geometry
(:func:`mrx.geometry.build_sequence`), the initial field
(:func:`mrx.initial_conditions.initial_field`) and the stepper
(:class:`mrx.relaxation.TimeStepper`), runs the descent in compiled chunks
until the force residual floors, the step budget is spent or the wall-clock
budget runs out, and writes the run. The descent is ideal, ``B_{n+1} = B_n +
dt curl(u x B)`` (or ``u x H`` with the auxiliary field); reconnection, when
asked for, is one resistive solve between chunks. The fixed point is ``J x B
= grad p`` with ``p`` the Leray multiplier, so the relaxed state is a
finite-beta equilibrium, not a force-free field.

Canonical invocation (one GPU; see slurm/README.md)::

    python -u scripts/relax.py --geometry data/wout_li383_1.4m.nc

Flags, defaults in brackets:
    Geometry, initial condition, discretisation:
      --geometry PATH (required)   the geometry AND the initial condition:
                                   a VMEC wout (.nc) or a GVEC state (.dat)
                                   gives the map and the equilibrium's own
                                   field B = dA' from its Clebsch data; an
                                   analytic geometry file (.json: a map of
                                   mrx.mappings with its parameters and the
                                   profiles iota, Phi', lambda of the
                                   logical-grid field; data/torus.json,
                                   cylinder.json, rot_ellipse.json) gives the
                                   map and that field. Always Leray-projected.
      --nfp N [file value]         field periods of a file that declares
                                   them wrong
      --ns R,T,Z [8,16,16]         spline resolution (also the map's)
      --r-refine a:b:m,... [""]    radial refinement: m uniform cells in each
                                   window [a, b] of the logical radius, the
                                   remaining n_r - p cells spread over the
                                   gaps (mrx.geometry.radial_knots)
      --p P [2]                    spline degree; p+1 Gauss points per span
      --solve-maxiter N [2000]     iteration budget of every inner solve
      --solve-tol TOL [1e-8 float32, 1e-10 float64]  residual tolerance of every solve (float64 residual)
      --precision {float32,float64} [float32]  exported as MRX_DTYPE before
                                   mrx is imported
      --seed m,n,rho0,width [""], --seed-eps EPS [0]
                                   equilibrium files only: a resonant term in
                                   A'_zeta that opens an island at the
                                   |iota| = nfp n / m surface
    Descent:
      --auxiliary-B-field {false,true} [false]
                                   true routes both cross products through
                                   the Dirichlet 1-form H = M_1^-1 P B, the
                                   auxiliary variable that makes the midpoint
                                   scheme conserve the discrete helicity
                                   exactly (H_t = 0 on the wall); false
                                   reads the 2-form B itself
      --scheme {explicit,midpoint} [explicit]
                                   forward Euler, or the midpoint-implicit
                                   induction with the explicit velocity
                                   (Picard on the increment, dt halved on a
                                   blow-up; mrx.relaxation.PICARD_*)
      --history M [1]              L-BFGS secant pairs: 0 is steepest
                                   descent, 1 memoryless BFGS (= CG)
      --velocity-smoothing-order G [0], --velocity-smoothing-scale MU [0.0]
                                   descent direction v = (I - MU L)^-G F
      --cfl C [0.5]                cap the line-search step at C / (largest
                                   logical CFL number of the velocity); inf
                                   disables it
    Budgets and output:
      --steps N [3000]             maximum number of steps
      --seconds S [none]           wall-clock budget of the descent loop
      --chunk N [500]              steps per compiled chunk (one lax.scan):
                                   the trace comes back, the quantities of
                                   interest are sampled (helicity, the two
                                   pressures, beta), a checkpoint and the
                                   outputs are written, and the floor,
                                   reconnect and wall-time tests run, once
                                   per chunk; --steps is a multiple of it
      --reconnect-every K [0]      see "Reconnection series"; 0 = off
      --reconnect-helicity X [0.01] the helicity each reconnection spends,
                                   |dH| / |H|
      --floor-tol TOL [1e-3]       stop when the last chunk's mean relative
                                   force residual ||F||_M / ||grad(B^2/2)||
                                   is below this (the residual is not
                                   monotone; the window mean is the quantity)
      --out DIR [outputs/relax/<date>/<time>]
      --restart PATH               continue from a checkpoint of the same
                                   geometry, mesh, degree and precision

Output (``--out``):
    relax.json           ``params`` (every flag, ``geometry_path`` resolved,
                         ``ic`` the kind of initial condition); ``ic``, the
                         initial field's numbers; the per-step ``trace``, the
                         per-chunk ``qoi``, the ``reconnect`` records and the
                         ``summary`` with the stopping reason (the fields of
                         mrx.relaxation.RelaxResult). Rewritten at every chunk.
    checkpoints/state_<step>.h5
                         the descent state at that step, one file per chunk
                         plus the initial field at step 0
                         (mrx.relaxation.write_checkpoint); the plotters
                         read them next to relax.json, ``--restart`` continues
                         from one, a reconnection's ``it`` names the file it
                         started from.

Reconnection series:
    ``--reconnect-every K`` runs the ideal descent and, every ``K`` steps
    (rounded to a whole number of chunks), reconnects the field with one
    backward-Euler solve of ``(M + eps L) delta = -eps L B``, then restarts
    the optimiser on the diffused field and carries on. The ideal descent is
    a power law, ``resid ~ t^-a`` (a = 0.2 at (16,32,32) p = 2 gamma = 1,
    1/3 at n = 8 and 12), never a plateau, so there is no stall to detect
    and the interval is a choice. The dose is set by the helicity it spends:
    ``eps = X |H| / (2 |int J . B|)`` from ``dH = -2 eps int J . B`` with
    ``X = --reconnect-helicity``; the record carries the target and the
    helicity actually spent. The outcome is the series of ideal equilibria,
    one per reconnection plus the final field, to choose from.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", required=True,
                    help="a VMEC wout (.nc), a GVEC state (.dat) or an analytic geometry (.json)")
    ap.add_argument("--nfp", type=int, default=None,
                    help="field periods; overrides the file's nfp attribute")
    ap.add_argument("--ns", default="8,16,16")
    ap.add_argument("--r-refine", default="",
                    help='radial refinement windows "a:b:m,..." (m cells in [a, b]); "" = uniform')
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--solve-maxiter", type=int, default=2000)
    ap.add_argument("--solve-tol", type=float, default=None)
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    ap.add_argument("--seed", default="",
                    help='resonant seed "m,n,rho0,width" added to the potential (equilibrium files only)')
    ap.add_argument("--seed-eps", type=float, default=0.0,
                    help="its amplitude |dB^rho| / |B^zeta| at rho0 (island width ~ sqrt of it)")
    ap.add_argument("--auxiliary-B-field", default="false", choices=("false", "true"),
                    help="route the cross products through the Dirichlet 1-form H = M_1^-1 P B")
    ap.add_argument("--scheme", default="explicit", choices=("explicit", "midpoint"))
    ap.add_argument("--history", type=int, default=1,
                    help="L-BFGS secant pairs; 0 is steepest descent, 1 memoryless BFGS (= CG)")
    ap.add_argument("--velocity-smoothing-order", type=int, default=0,
                    help="descent direction v = (I - scale L)^-order F; 0 is off")
    ap.add_argument("--velocity-smoothing-scale", type=float, default=0.0,
                    help="length scale of the velocity smoothing")
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=500,
                    help="steps per compiled chunk; trace, qoi sample, checkpoint, outputs and the "
                         "floor / reconnect / wall-time tests once per chunk")
    ap.add_argument("--floor-tol", type=float, default=1e-3,
                    help="stop when the last chunk's mean relative force residual is below this")
    ap.add_argument("--reconnect-every", type=int, default=0,
                    help="reconnect the field with one resistive solve every K steps, rounded "
                         "to whole chunks; 0 = off (see the docstring)")
    ap.add_argument("--reconnect-helicity", type=float, default=0.01,
                    help="the helicity each reconnection spends, |dH| / |H|")
    ap.add_argument("--out", default=None)
    ap.add_argument("--restart", default=None,
                    help="continue from a checkpoints/state_<step>.h5 of the same geometry, "
                         "mesh, degree and precision")
    cli = ap.parse_args(argv)
    cli.auxiliary_B_field = cli.auxiliary_B_field == "true"
    if cli.history < 0:
        ap.error("--history must be non-negative (0 is steepest descent)")
    if cli.chunk < 1 or cli.steps % cli.chunk:
        ap.error("--steps must be a positive multiple of --chunk")
    if not os.path.isfile(cli.geometry):
        ap.error(f"--geometry {cli.geometry!r} is not a file (a .nc, .dat or .json)")
    if cli.seed and cli.geometry.endswith(".json"):
        ap.error("--seed needs an equilibrium file (.nc or .dat)")
    return cli


def main(cli):
    import mrx
    from mrx.geometry import build_sequence, geometry_kind, parse_r_refine
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (IntegrationScheme, TimeStepper, initial_state, read_checkpoint,
                                relax, write_checkpoint)

    if cli.precision != str(mrx.DTYPE):
        raise ValueError(f"--precision {cli.precision} but mrx runs in {mrx.DTYPE}")
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)
    ns = tuple(int(v) for v in cli.ns.split(","))
    out = cli.out or os.path.join("outputs", "relax", time.strftime("%Y-%m-%d"),
                                  time.strftime("%H-%M-%S"))
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    params = dict(vars(cli), ns=list(ns), out=out, geometry_path=os.path.abspath(cli.geometry),
                  ic=geometry_kind(cli.geometry))
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.solve_maxiter, tol=cli.solve_tol,
                              nfp=cli.nfp, r_windows=parse_r_refine(cli.r_refine))
    compute_nullspaces(seq)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} tol={seq.tol:.1e}  "
          f"n2_dbc={seq.n(2, True)}  operators+nullspaces "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    # --- initial condition -----------------------------------------------
    t1 = time.perf_counter()
    seed = None
    if cli.seed:
        m, n, rho0, width = (float(v) for v in cli.seed.split(","))
        seed = (int(m), int(n), rho0, width, cli.seed_eps)
    B0, ic = initial_field(seq, seed)
    results["ic"] = ic
    print(f"[ic] {ic['kind']} IC in {time.perf_counter() - t1:.1f}s: "
          + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                      for k, v in ic.items() if k != "kind"), flush=True)

    # --- the descent -------------------------------------------------------
    ts = TimeStepper(
        seq=seq, auxiliary_B_field=cli.auxiliary_B_field,
        scheme={"explicit": IntegrationScheme.EXPLICIT,
                "midpoint": IntegrationScheme.IMPLICIT_MIDPOINT}[cli.scheme],
        cfl=cli.cfl, history_size=cli.history,
        velocity_smoothing_order=cli.velocity_smoothing_order,
        velocity_smoothing_scale=cli.velocity_smoothing_scale)
    if cli.restart:
        state, it0 = read_checkpoint(cli.restart, ts)
        print(f"[restart] {cli.restart}: descent state at step {it0}", flush=True)
    else:
        state, it0 = initial_state(B0, ts), 0
        write_checkpoint(os.path.join(ckpt_dir, "state_000000.h5"), state, 0)
    params["start_step"] = it0
    print(f"\n=== L-BFGS m={cli.history}  auxiliary-B-field={str(cli.auxiliary_B_field).lower()}  "
          f"scheme={cli.scheme}  smoothing={cli.velocity_smoothing_order}@{cli.velocity_smoothing_scale} "
          f"cfl={cli.cfl}  steps<={cli.steps} chunk={cli.chunk} floor-tol={cli.floor_tol:.1e} "
          f"reconnect-every={cli.reconnect_every}"
          + (f" ({cli.reconnect_helicity:.2%} of H each)" if cli.reconnect_every else "") + " ===",
          flush=True)

    def save(res):
        """The run so far: the checkpoint of this step, then relax.json."""
        it = it0 + res.steps
        write_checkpoint(os.path.join(ckpt_dir, f"state_{it:06d}.h5"), res.state, it)
        last = {k: v[-1] for k, v in res.qoi.items() if k not in ("it", "wall")}
        results.update(
            trace=res.trace, qoi=res.qoi, reconnect=res.reconnect,
            summary=dict(steps=res.steps, stop=res.stop, wall=res.wall,
                         reconnect_every=res.reconnect_every,
                         E0=res.E0, E_removed=-float(sum(res.trace["dE"])), F_final=res.trace["F"][-1],
                         resid_final=res.trace["resid"][-1],
                         resid_window_mean=float(sum(res.trace["resid"][-res.chunk:]) / res.chunk),
                         **last))
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)

    relax(state, ts, steps=cli.steps, chunk=cli.chunk, it0=it0, floor_tol=cli.floor_tol,
          seconds=cli.seconds, reconnect_every=cli.reconnect_every,
          reconnect_helicity=cli.reconnect_helicity, on_chunk=save)
    print(f"wrote {out}/relax.json and {ckpt_dir}/", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"] = cli.precision
    main(cli)
