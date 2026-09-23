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
      --ns R,T,Z [16,32,32]        spline resolution (also the map's)
      --knots-r LIST [""], --knots-theta LIST [""], --knots-zeta LIST [""]
                                   the breakpoints of that axis, comma-
                                   separated from 0 to 1, instead of the
                                   uniform grid; the axis takes its n from
                                   them (cells + p clamped, cells periodic;
                                   mrx.geometry.knot_vector)
      --p P [2]                    spline degree; p+1 Gauss points per span
      --solve-maxiter N [2000]     iteration budget of every inner solve
      --solve-tol TOL [1e-8 with a float64 residual, 1e-10 in float64,
                                   sqrt(eps) ~ 3.5e-4 in plain float32]
      --precision {mixed,float32,float64} [mixed]
                                   mixed: float32 fields and solves with a
                                   float64 residual; float32, float64: both
                                   (MRX_DTYPE and MRX_RESIDUAL_DTYPE, exported
                                   before mrx is imported)
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
      --method {newton,lbfgs} [lbfgs in float32, else newton]
                                   the direction: Newton on the second
                                   variation (the Newton flags below) or the
                                   L-BFGS descent. In plain float32 L-BFGS
                                   reaches the energy floor in about a
                                   minute at (12,24,12) p=3 on an M3 Pro;
                                   Newton had removed less energy after 30,
                                   and at (16,32,16) half as much in the
                                   same time (docs/source/mps.md)
      --history M [1]              L-BFGS secant pairs: 0 is steepest
                                   descent, 1 memoryless BFGS (= CG)
      --velocity-smoothing-order G [1], --velocity-smoothing-scale MU [0.02 / n_r^2]
                                   descent direction v = (I - MU L)^-G F
      --cfl C [0.5]                cap the line-search step at C / (largest
                                   logical CFL number of the velocity); inf
                                   disables it
      --potential-velocity {false,true} [true for the L-BFGS descent]
                                   the projected force as curl a + c h from
                                   the k=1 Hodge solve of curl^T load(J x B)
                                   instead of the Leray saddle solve
                                   (divergence-free to roundoff), the
                                   smoothing on the potential, L-BFGS on
                                   the smoothed forces; Newton and the
                                   auxiliary field have their own routes
      --force-hodge-passes N [1 in float32, else 0]  outer passes of that
                                   k=1 solve, warm-started from the last
                                   step's potential; 0 is the solve's own
                                   cap (two in float32)
    Newton (--method newton): the direction u = curl a with
    curl^T H curl a = curl^T M F by MINRES (mrx.hessian); a non-descending
    direction falls back to the smoothed force.
      --newton-tol TOL [0.1]       relative residual of the MINRES solve
      --newton-maxiter N [300]     its iteration budget per step
      --newton-precond {laplacian,laplacian2,mass,harmonic} [laplacian]
                                   the preconditioner: the k=1 Laplacian
                                   atom, its square, the k=1 mass atom, or
                                   the harmonic atom (the Laplacian atom with
                                   the parallel symbol of the harmonic field
                                   in its denominator, mrx.hessian)
      --newton-dt-cap C [1]        cap the line-search step along a Newton
                                   direction (1 = the Newton step, inf
                                   leaves the line search alone)
    Budgets and output:
      --steps N [100 Newton, 3000 L-BFGS]
                                   maximum number of steps
      --chunk N [10 in float32, else 20 Newton, 500 L-BFGS]
                                   steps per compiled chunk (one lax.scan):
                                   the trace comes back and the floor test
                                   runs once per chunk; --steps is a
                                   multiple of it
      --sample-every N [one sample per about 50 steps in float32, else 1]
                                   full sample, checkpoint and outputs
                                   every N chunks, and always at the exit.
                                   The floor test does not wait for it
      --reconnect-every K [0]      see "Reconnection series"; 0 = off
      --reconnect-helicity X [0.01] the helicity each reconnection spends,
                                   |dH| / |H|
      --floor-tol TOL [0 in float32, else 1e-8]
                                   stop when the last chunk's mean squared
                                   normalised force residual
                                   ||F||^2_M / ||grad(B^2/2)||^2 is below
                                   this (the residual is not monotone; the
                                   window mean is the quantity). Plain
                                   float32 never reaches 1e-8, so the
                                   default there is 0 and the stop is
                                   --energy-floor
      --energy-floor auto|on|off [auto]
                                   stop when the energy trace has stopped
                                   descending (two chunks that each raise
                                   the energy on 3 of every 10 steps or
                                   lower it by at most 5e-4 of the energy
                                   removed; mrx.relaxation.energy_floor)
                                   and return the lowest-energy field.
                                   auto is on in plain float32 and off
                                   when a float64 residual is in use
      --out DIR [outputs/relax/<date>/<time>]
      --warm-from R,T,Z [""]       relax that coarser mesh to its stop first,
                                   then start from this mesh's initial field
                                   plus the coarse relaxation, moved by the
                                   commuting histopolation (div B stays at
                                   round-off; mrx.initial_conditions.
                                   transfer_field). relax.json records the
                                   coarse run under "warm"
      --restart PATH               continue from a checkpoint of the same
                                   geometry, mesh, degree and precision
      --map-batch N [0]            cells per batch of the quadrature loops
                                   (mrx.MAP_BATCH_SIZE_INNER); 0 evaluates
                                   all points in one vmap. Bound it at high
                                   resolution: the initial field's Greville
                                   histopolation asks for 17 GiB at
                                   (64,128,128) p=2 unbounded (8192 there)

Output (``--out``):
    relax.json           ``params`` (every flag, ``geometry_path`` resolved,
                         ``ic`` the kind of initial condition); ``ic``, the
                         initial field's numbers; the per-step ``trace``, the
                         ``qoi`` (one row per sample), the ``reconnect`` records and the
                         ``summary`` with the stopping reason (the fields of
                         mrx.relaxation.RelaxResult). Rewritten at every sample.
    checkpoints/state_<step>.h5
                         the descent state at a sampled step, plus the
                         initial field at step 0
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


#: --precision -> (MRX_DTYPE, MRX_RESIDUAL_DTYPE)
PRECISIONS = {"mixed": ("float32", "float64"), "float32": ("float32", "float32"),
              "float64": ("float64", "float64")}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", required=True,
                    help="a VMEC wout (.nc), a GVEC state (.dat) or an analytic geometry (.json)")
    ap.add_argument("--nfp", type=int, default=None,
                    help="field periods; overrides the file's nfp attribute")
    ap.add_argument("--ns", default="16,32,32")
    for axis in ("r", "theta", "zeta"):
        ap.add_argument(f"--knots-{axis}", default="",
                        help=f'breakpoints of the {axis} axis, comma-separated from 0 to 1; "" = uniform')
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--solve-maxiter", type=int, default=2000)
    ap.add_argument("--solve-tol", type=float, default=None)
    ap.add_argument("--precision", default="mixed", choices=tuple(PRECISIONS))
    ap.add_argument("--seed", default="",
                    help='resonant seed "m,n,rho0,width" added to the potential (equilibrium files only)')
    ap.add_argument("--seed-eps", type=float, default=0.0,
                    help="its amplitude |dB^rho| / |B^zeta| at rho0 (island width ~ sqrt of it)")
    ap.add_argument("--auxiliary-B-field", default="false", choices=("false", "true"),
                    help="route the cross products through the Dirichlet 1-form H = M_1^-1 P B")
    ap.add_argument("--scheme", default="explicit", choices=("explicit", "midpoint"))
    ap.add_argument("--history", type=int, default=1,
                    help="L-BFGS secant pairs; 0 is steepest descent, 1 memoryless BFGS (= CG)")
    ap.add_argument("--velocity-smoothing-order", type=int, default=1,
                    help="descent direction v = (I - scale L)^-order F; 0 is off and fragile: the "
                         "unsmoothed descent stops conserving helicity after ~1e4 steps (numerical "
                         "reconnection)")
    ap.add_argument("--velocity-smoothing-scale", type=float, default=None,
                    help="length scale of the velocity smoothing [mrx.relaxation.SMOOTHING_C / n_r^2]")
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--potential-velocity", default=None, choices=("false", "true"),
                    help="the projected force as curl a + c h (k=1 Hodge solve) instead of the Leray solve "
                         "[true for the L-BFGS descent; Newton and the auxiliary field have their own routes]")
    ap.add_argument("--force-hodge-passes", type=int, default=None,
                    help="outer passes of the force's k=1 Hodge solve; 0 is the solve's own cap "
                         "[1 in plain float32, else 0]")
    ap.add_argument("--method", default=None, choices=("newton", "lbfgs"),
                    help="the direction: Newton on the second variation, or the L-BFGS descent "
                         "[lbfgs in plain float32, newton otherwise]")
    ap.add_argument("--newton-tol", type=float, default=0.1,
                    help="relative residual tolerance of the Newton MINRES solve")
    ap.add_argument("--newton-maxiter", type=int, default=300,
                    help="iteration budget of the Newton MINRES solve per step")
    ap.add_argument("--newton-precond", default="laplacian", choices=("laplacian", "laplacian2", "mass", "harmonic"),
                    help="preconditioner of the Newton solve")
    ap.add_argument("--newton-dt-cap", type=float, default=1.0,
                    help="cap on the line-search step along a Newton direction (1 = the Newton step)")
    ap.add_argument("--steps", type=int, default=None, help="maximum steps [100 Newton, 3000 L-BFGS]")
    ap.add_argument("--chunk", type=int, default=None,
                    help="steps per compiled chunk; the trace comes back and the floor test runs "
                         "once per chunk [10 in float32, else 20 Newton, 500 L-BFGS]")
    ap.add_argument("--sample-every", type=int, default=None,
                    help="full sample, checkpoint and outputs every N chunks, and always at the "
                         "exit [one sample per about 50 steps in float32, else every chunk]")
    ap.add_argument("--floor-tol", type=float, default=None,
                    help="stop when the last chunk's mean squared normalised force residual is below "
                         "this [0 in float32, where 1e-8 is unreachable; 1e-8 otherwise]")
    ap.add_argument("--energy-floor", default="auto", choices=("auto", "on", "off"),
                    help="stop when the energy trace has stopped descending, and return the "
                         "lowest-energy field [auto: on in plain float32, off in mixed and float64]")
    ap.add_argument("--reconnect-every", type=int, default=0,
                    help="reconnect the field with one resistive solve every K steps, rounded "
                         "to whole chunks; 0 = off (see the docstring)")
    ap.add_argument("--reconnect-helicity", type=float, default=0.01,
                    help="the helicity each reconnection spends, |dH| / |H|")
    ap.add_argument("--out", default=None)
    ap.add_argument("--warm-from", default="",
                    help='"R,T,Z": relax that coarse mesh first (same geometry, degree and stops), '
                         "then start this mesh from its initial field plus the coarse relaxation")
    ap.add_argument("--restart", default=None,
                    help="continue from a checkpoints/state_<step>.h5 of the same geometry, "
                         "mesh, degree and precision")
    ap.add_argument("--map-batch", type=int, default=0,
                    help="cells per batch of the quadrature loops (mrx.MAP_BATCH_SIZE_INNER); "
                         "0 = all points in one vmap; bound it at high resolution")
    cli = ap.parse_args(argv)
    if cli.map_batch < 0:
        ap.error("--map-batch must be non-negative (0 is one vmap over all points)")
    cli.auxiliary_B_field = cli.auxiliary_B_field == "true"
    # Plain float32 stops on the energy floor, and L-BFGS gets there
    # first: Newton's 36 s steps (li383 (12,24,12) p=3, Metal) had removed
    # less energy after 30 minutes than L-BFGS does in two.
    cli.method_default = cli.method is None
    if cli.method is None:
        cli.method = "lbfgs" if cli.precision == "float32" else "newton"
    cli.newton = cli.method == "newton"
    if cli.steps is None:
        cli.steps = 100 if cli.newton else 3000
    if cli.floor_tol is None:
        # A squared residual of 1e-8 is below anything a plain float32 force
        # reaches at this mesh (it oscillates from about 1e-3 to 1e-2), so
        # the default would never fire. The energy floor is the stop there.
        cli.floor_tol = 0.0 if cli.precision == "float32" else 1e-8
    if cli.chunk is None:
        cli.chunk = 10 if cli.precision == "float32" else (20 if cli.newton else 500)
    if cli.sample_every is None:
        # A sample is a full force evaluation, about as long as the steps
        # it sits between. The floor test reads the chunk trace, so float32
        # samples about once per 50 steps. Mixed and float64 keep a sample
        # on every chunk.
        cli.sample_every = max(1, round(50 / cli.chunk)) if cli.precision == "float32" else 1
    if cli.sample_every < 1:
        ap.error("--sample-every counts chunks and must be positive")
    cli.potential_velocity = None if cli.potential_velocity is None else cli.potential_velocity == "true"
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
    from mrx.geometry import build_sequence, geometry_kind, parse_knots
    from mrx.initial_conditions import initial_field, transfer_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (IntegrationScheme, TimeStepper, compute_divergence_norm,
                                initial_state, read_checkpoint, relax, write_checkpoint)

    if (str(mrx.DTYPE), str(mrx.precision.RESIDUAL_DTYPE)) != PRECISIONS[cli.precision]:
        raise ValueError(f"--precision {cli.precision} but mrx runs in {mrx.DTYPE} "
                         f"with {mrx.precision.RESIDUAL_DTYPE} residuals")
    mrx.MAP_BATCH_SIZE_INNER = cli.map_batch
    print(f"[env] mrx from {mrx.__file__}  precision {cli.precision} ({mrx.DTYPE} solves, "
          f"{mrx.precision.RESIDUAL_DTYPE} residual)  map batch {cli.map_batch or 'all'}", flush=True)
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
    knots = [parse_knots(s) for s in (cli.knots_r, cli.knots_theta, cli.knots_zeta)]
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.solve_maxiter, tol=cli.solve_tol,
                              nfp=cli.nfp, knots=knots)
    ns = seq.ns
    params.update(ns=list(ns), knots=knots)
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
    def stepper(s):
        return TimeStepper(
            seq=s, auxiliary_B_field=cli.auxiliary_B_field,
            scheme={"explicit": IntegrationScheme.EXPLICIT,
                    "midpoint": IntegrationScheme.IMPLICIT_MIDPOINT}[cli.scheme],
            cfl=cli.cfl, history_size=0 if cli.newton else cli.history,
            velocity_smoothing_order=cli.velocity_smoothing_order,
            velocity_smoothing_scale=cli.velocity_smoothing_scale,
            potential_velocity=cli.potential_velocity,
            force_hodge_passes=cli.force_hodge_passes,
            newton=cli.newton, newton_tol=cli.newton_tol, newton_maxiter=cli.newton_maxiter,
            newton_precond=cli.newton_precond, newton_dt_cap=cli.newton_dt_cap)

    ts = stepper(seq)
    if cli.warm_from and not cli.restart:
        # The coarse mesh relaxes to its own stop, then only its increment
        # moves: the fine run starts from the fine mesh's initial field plus
        # the coarse relaxation (mrx.initial_conditions.transfer_field).
        tc = time.perf_counter()
        ns_c = tuple(int(v) for v in cli.warm_from.split(","))
        seq_c, _ = build_sequence(cli.geometry, ns_c, cli.p, cli.solve_maxiter, tol=cli.solve_tol,
                                  nfp=cli.nfp)
        compute_nullspaces(seq_c)
        B0_c, _ = initial_field(seq_c, seed)
        setup_c = time.perf_counter() - tc
        print(f"\n[warm] coarse ns={seq_c.ns} p={cli.p}: setup {setup_c:.1f}s", flush=True)
        ts_c = stepper(seq_c)
        res_c = relax(initial_state(B0_c, ts_c), ts_c, steps=cli.steps, chunk=cli.chunk,
                      floor_tol=cli.floor_tol,
                      energy_floor={"auto": None, "on": True, "off": False}[cli.energy_floor],
                      sample_every=cli.sample_every)
        tt = time.perf_counter()
        E_base = 0.5 * float(seq.l2_norm_sq(B0, 2))
        B0 = transfer_field(res_c.state.B_n, seq_c, seq, base_from=B0_c, base_to=B0)
        E_warm = 0.5 * float(seq.l2_norm_sq(B0, 2))
        transfer_s = time.perf_counter() - tt
        results["warm"] = dict(ns=list(seq_c.ns), setup=setup_c, steps=res_c.steps,
                               best_step=res_c.best_step, stop=res_c.stop, wall=res_c.wall,
                               elapsed=time.perf_counter() - tc, transfer=transfer_s,
                               E_base=E_base, E_warm=E_warm)
        print(f"[warm] {res_c.steps} coarse steps ({res_c.stop}, field at {res_c.best_step}) in "
              f"{res_c.wall:.1f}s; transfer {transfer_s:.1f}s: fine E {E_base:.8e} -> {E_warm:.8e} "
              f"(removed {E_base - E_warm:.4e}), ||div B|| {float(compute_divergence_norm(B0, seq)):.2e}",
              flush=True)
    if cli.restart:
        state, it0 = read_checkpoint(cli.restart, ts)
        print(f"[restart] {cli.restart}: descent state at step {it0}", flush=True)
    else:
        state, it0 = initial_state(B0, ts), 0
        write_checkpoint(os.path.join(ckpt_dir, "state_000000.h5"), state, 0)
    params["start_step"] = it0
    params["velocity_smoothing_scale"] = float(ts.velocity_smoothing_scale)
    params["potential_velocity"] = bool(ts.potential_velocity)
    params["force_hodge_passes"] = int(ts.force_hodge_passes)
    print(f"\n=== {'newton tol=%.1e maxiter=%d precond=%s' % (cli.newton_tol, cli.newton_maxiter, cli.newton_precond) if cli.newton else 'L-BFGS m=%d' % cli.history}"
          f"{' (the ' + cli.precision + ' default method)' if cli.method_default else ''}"
          f"{'  potential-velocity (force Hodge passes %d)' % ts.force_hodge_passes if ts.potential_velocity else ''}  auxiliary-B-field={str(cli.auxiliary_B_field).lower()}  "
          f"scheme={cli.scheme}  smoothing={cli.velocity_smoothing_order}@{ts.velocity_smoothing_scale:.3e} "
          f"cfl={cli.cfl}  steps<={cli.steps} chunk={cli.chunk} sample-every={cli.sample_every} "
          f"floor-tol={cli.floor_tol:.1e} energy-floor={cli.energy_floor} "
          f"reconnect-every={cli.reconnect_every}"
          + (f" ({cli.reconnect_helicity:.2%} of H each)" if cli.reconnect_every else "") + " ===",
          flush=True)

    def save(res):
        """The run so far: the checkpoint of this step, then relax.json.

        On the float32 energy floor the checkpoint is the lowest-energy
        field, at ``best_step``, not the later chunk the patience windows
        walked through.
        """
        it = it0 + res.best_step
        write_checkpoint(os.path.join(ckpt_dir, f"state_{it:06d}.h5"), res.state, it)
        last = {k: v[-1] for k, v in res.qoi.items() if k not in ("it", "wall")}
        results.update(
            trace=res.trace, qoi=res.qoi, reconnect=res.reconnect,
            summary=dict(steps=res.steps, stop=res.stop, wall=res.wall,
                         best_step=res.best_step,
                         reconnect_every=res.reconnect_every,
                         E0=res.E0, E_removed=res.E0 - res.qoi["E"][-1], F_final=res.trace["F"][-1],
                         resid_final=res.trace["resid"][-1],
                         resid_window_mean=float(sum(res.trace["resid"][-res.chunk:]) / res.chunk),
                         **last))
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)

    relax(state, ts, steps=cli.steps, chunk=cli.chunk, it0=it0, floor_tol=cli.floor_tol,
          energy_floor={"auto": None, "on": True, "off": False}[cli.energy_floor],
          sample_every=cli.sample_every,
          reconnect_every=cli.reconnect_every,
          reconnect_helicity=cli.reconnect_helicity, on_chunk=save)
    print(f"wrote {out}/relax.json and {ckpt_dir}/", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"], os.environ["MRX_RESIDUAL_DTYPE"] = PRECISIONS[cli.precision]
    main(cli)
