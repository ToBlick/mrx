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
      --symmetry {stellarator,field-period,none} [stellarator]
                                   what the map satisfies: nfp field periods
                                   and stellarator symmetry (the map is
                                   projected onto it, the quadrature covers
                                   half the period and the fields keep their
                                   parity: mrx.symmetry), field periods only,
                                   or nothing (zeta in [0, 1] is the whole
                                   torus, nfp = 1); mrx.geometry.SYMMETRIES
      --ns R,T,Z [16,32,32]        spline resolution (also the map's)
      --knots-r LIST [""], --knots-theta LIST [""], --knots-zeta LIST [""]
                                   the breakpoints of that axis, comma-
                                   separated from 0 to 1, instead of the
                                   uniform grid; the axis takes its n from
                                   them (cells + p clamped, cells periodic;
                                   mrx.geometry.knot_vector)
      --p P [2]                    spline degree; p+1 Gauss points per span
      --solve-maxiter N [2000]     iteration budget of every inner solve
      --solve-tol TOL [1e-8 float32, 1e-10 float64]  residual tolerance of every solve (float64 residual)
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
      --helicity-correction {false,true} [false]
                                   remove from the induction field E the one
                                   component (a multiple of the Dirichlet
                                   proxy of B) that changes the discrete
                                   helicity: exact conservation with H
                                   natural, either scheme
                                   (TimeStepper.helicity_correction); the
                                   trace records the multiple as hcorr
      --method {newton,gradient} [newton]
                                   the direction: Newton on the second
                                   variation (the Newton flags below) or
                                   gradient descent on the smoothed force
      --velocity-smoothing-order G [1], --velocity-smoothing-scale MU [0.075 h_r^2]
                                   descent direction v = (I - MU L)^-G F
      --cfl C [0.5]                cap the line-search step at C / (largest
                                   logical CFL number of the velocity); inf
                                   disables it
      --potential-velocity {false,true} [true for the gradient descent]
                                   the projected force as curl a + c h from
                                   the k=1 Hodge solve of curl^T load(J x B)
                                   instead of the Leray saddle solve
                                   (divergence-free to roundoff), the
                                   smoothing on the potential; Newton and the
                                   auxiliary field have their own routes
    Newton (--method newton): the direction u = curl a with
    curl^T H curl a = curl^T M F solved by Newton-MR (mrx.hessian): MINRES
    with the harmonic atom of the current field, the parallel-flow penalty
    in the operator, the nonpositive-curvature exit; the line search along
    the direction is capped at the Newton length dt = 1.
      --newton-penalty KAPPA [3]   the parallel-flow penalty, kappa times the
                                   strain along the field: the one number of
                                   the Newton configuration (3 on every case
                                   measured; 1 is 2x worse, 0.03 lets the
                                   Hessian's null space through)
      --newton-tol TOL [0.1]       the forcing term: the residual of the
                                   Newton system (double precision, the
                                   mass-atom norm) below TOL of the
                                   right-hand side ends the solve
      --newton-maxiter N [200]     MINRES iterations per pass
      --newton-passes N [1]        passes at most: the solve is inexact by
                                   design (200 iterations give the same
                                   relaxation as any tighter solve; with
                                   the strain penalty the 0.1 forcing term
                                   is met late, so more passes only cost)
    Budgets and output:
      --steps N [100 Newton, 3000 gradient]
                                   maximum number of steps
      --chunk N [20 Newton, 500 gradient]
                                   steps per compiled chunk (one lax.scan):
                                   the trace comes back, the quantities of
                                   interest are sampled (helicity, the two
                                   pressures, beta), a checkpoint and the
                                   outputs are written, and the floor,
                                   reconnect and wall-time tests run, once
                                   per chunk; --steps is a multiple of it
      --reconnect-every K [0]      see "Reconnection series"; 0 = off
      --reconnect-helicity X [0.01] the helicity each reconnection spends,
                                   |dH| / |H|
      --floor-tol TOL [1e-8]       stop when the last chunk's mean squared
                                   normalised force residual
                                   ||F||^2_M / ||grad(B^2/2)||^2 is below
                                   this (the residual is not monotone; the
                                   window mean is the quantity)
      --out DIR [outputs/relax/<date>/<time>]
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
    helicity actually spent. ``--reconnect-eps C`` instead applies a constant
    dose ``eps = C h_r^2`` per solve (``h_r`` the physical radial cell,
    mrx.relaxation.radial_cell_sq; on li383 at (16,32,32) C = 0.16 spends
    about 1% of the helicity), a constant resistivity: the ideal
    relaxation is fast and the diffusion slow, so the solves go between
    blocks of ideal steps and the field relaxes back before the next one; the
    helicity spent is then an outcome. ``--reconnect-window A:B`` restricts
    the solves to steps A..B (relax ideally, reconnect gradually, relax
    ideally). The outcome is the series of ideal equilibria, one per
    reconnection plus the final field, to choose from.

Resistive steady state:
    ``--resistivity C`` adds to EVERY step, after the ideal one, a
    backward-Euler step of ``dB/dt = -eta curl (J - J*)`` with the dose
    ``eps = C h_r^2`` (mrx.relaxation.TimeStepper.resistivity), and ``J*``
    the current of ``B*``: the run's start field (the ``--restart``
    checkpoint, a converged ideal run; or the initial field) after one heat
    step of ``c h_r^2``, ``c = --reference-smoothing``, which removes the
    rational-surface sheets of the ideal equilibrium (sustained, they would
    make the start a fixed point) and costs ``O(c h_r^2)`` of the bulk
    current. The run goes to the resistive steady state (the islands open
    and saturate; the force residual is of order ``eps``); a restart of it
    without ``--resistivity`` then relaxes it ideally, the islands frozen in.
    A seed in the initial field of the converged run is in ``B*``: an
    applied perturbation that stays.
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
    ap.add_argument("--symmetry", default="stellarator", choices=("stellarator", "field-period", "none"),
                    help="what the map satisfies (mrx.geometry.SYMMETRIES)")
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
    ap.add_argument("--helicity-correction", default="false", choices=("false", "true"),
                    help="zero the step's discrete helicity change by one scalar correction of E")
    ap.add_argument("--velocity-smoothing-order", type=int, default=1,
                    help="descent direction v = (I - scale L)^-order F; 0 is off and fragile: the "
                         "unsmoothed descent stops conserving helicity after ~1e4 steps (numerical "
                         "reconnection)")
    ap.add_argument("--velocity-smoothing-scale", type=float, default=None,
                    help="length scale of the velocity smoothing [mrx.relaxation.SMOOTHING_C h_r^2, h_r the physical radial cell]")
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--potential-velocity", default=None, choices=("false", "true"),
                    help="the projected force as curl a + c h (k=1 Hodge solve) instead of the Leray solve "
                         "[true for the gradient descent; Newton and the auxiliary field have their own routes]")
    ap.add_argument("--method", default="newton", choices=("newton", "gradient"),
                    help="the direction: Newton on the second variation, or gradient descent on the smoothed force")
    ap.add_argument("--newton-penalty", type=float, default=3.0,
                    help="kappa of the parallel-flow penalty, kappa times the strain along the field")
    ap.add_argument("--newton-tol", type=float, default=0.1,
                    help="the forcing term of the Newton solve: the residual below TOL of the right-hand side")
    ap.add_argument("--newton-maxiter", type=int, default=200, help="MINRES iterations per pass of the Newton solve")
    ap.add_argument("--newton-passes", type=int, default=1, help="passes of the Newton solve at most")
    ap.add_argument("--steps", type=int, default=None, help="maximum steps [100 Newton, 3000 gradient]")
    ap.add_argument("--chunk", type=int, default=None,
                    help="steps per compiled chunk; trace, qoi sample, checkpoint, outputs and the "
                         "floor / reconnect / wall-time tests once per chunk")
    ap.add_argument("--floor-tol", type=float, default=1e-8,
                    help="stop when the last chunk's mean squared normalised force residual is below this")
    ap.add_argument("--reconnect-every", type=int, default=0,
                    help="reconnect the field with one resistive solve every K steps, rounded "
                         "to whole chunks; 0 = off (see the docstring)")
    ap.add_argument("--reconnect-helicity", type=float, default=0.01,
                    help="the helicity each reconnection spends, |dH| / |H|")
    ap.add_argument("--reconnect-eps", type=float, default=None,
                    help="a constant dose eps = C h_r^2 per resistive solve (h_r the physical radial cell) "
                         "instead of the helicity target")
    ap.add_argument("--reconnect-window", default=None,
                    help="A:B, the resistive solves only at steps A..B")
    ap.add_argument("--resistivity", type=float, default=0.0,
                    help="a resistive dose C h_r^2 in every step, E = eta (J - J*): see 'Resistive steady state'")
    ap.add_argument("--reference-smoothing", type=float, default=0.1,
                    help="B* = the start field after one heat step of c h_r^2 (with --resistivity)")
    ap.add_argument("--out", default=None)
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
    cli.helicity_correction = cli.helicity_correction == "true"
    cli.newton = cli.method == "newton"
    if cli.steps is None:
        cli.steps = 100 if cli.newton else 3000
    if cli.chunk is None:
        cli.chunk = 20 if cli.newton else 500
    cli.potential_velocity = None if cli.potential_velocity is None else cli.potential_velocity == "true"
    if cli.chunk < 1 or cli.steps % cli.chunk:
        ap.error("--steps must be a positive multiple of --chunk")
    if not os.path.isfile(cli.geometry):
        ap.error(f"--geometry {cli.geometry!r} is not a file (a .nc, .dat or .json)")
    if cli.seed and cli.geometry.endswith(".json"):
        ap.error("--seed needs an equilibrium file (.nc or .dat)")
    return cli


def main(cli):
    import equinox as eqx
    import mrx
    from mrx.geometry import build_sequence, geometry_kind, parse_knots
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (IntegrationScheme, TimeStepper, initial_state, read_checkpoint,
                                radial_cell_sq, relax, resistive_step, write_checkpoint)

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
                              nfp=cli.nfp, knots=knots, symmetry=cli.symmetry)
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
    h_r_sq = radial_cell_sq(seq)
    params["h_r_sq"] = h_r_sq
    ts = TimeStepper(
        seq=seq, auxiliary_B_field=cli.auxiliary_B_field,
        scheme={"explicit": IntegrationScheme.EXPLICIT,
                "midpoint": IntegrationScheme.IMPLICIT_MIDPOINT}[cli.scheme],
        cfl=cli.cfl,
        helicity_correction=cli.helicity_correction,
        velocity_smoothing_order=cli.velocity_smoothing_order,
        velocity_smoothing_scale=cli.velocity_smoothing_scale,
        potential_velocity=cli.potential_velocity,
        newton=cli.newton, newton_penalty=cli.newton_penalty, newton_tol=cli.newton_tol,
        newton_maxiter=cli.newton_maxiter, newton_passes=cli.newton_passes,
        resistivity=cli.resistivity * h_r_sq)
    if cli.restart:
        state, it0 = read_checkpoint(cli.restart, ts)
        print(f"[restart] {cli.restart}: descent state at step {it0}", flush=True)
    else:
        state, it0 = initial_state(B0, ts), 0
        write_checkpoint(os.path.join(ckpt_dir, "state_000000.h5"), state, 0)
    if cli.resistivity:
        # B*: the start field without its rational-surface sheets
        B_star = state.B_n
        if cli.reference_smoothing:
            B_star = resistive_step(B_star, seq, cli.reference_smoothing * h_r_sq)[0]
        ts = eqx.tree_at(lambda t: t.resistive_reference, ts, B_star, is_leaf=lambda x: x is None)
        print(f"[resistivity] eps {cli.resistivity:g} h_r^2 = {ts.resistivity:.3e} per step; B* = the start field "
              f"after a heat step of {cli.reference_smoothing:g} h_r^2, ||B - B*|| / ||B|| = "
              f"{float(seq.l2_norm(state.B_n - B_star, 2) / seq.l2_norm(state.B_n, 2)):.3e}", flush=True)
    params["start_step"] = it0
    params["velocity_smoothing_scale"] = float(ts.velocity_smoothing_scale)
    params["potential_velocity"] = bool(ts.potential_velocity)
    print(f"\n=== {'newton-MR penalty=%g tol=%.1e maxiter=%d passes=%d' % (cli.newton_penalty, cli.newton_tol, cli.newton_maxiter, cli.newton_passes) if cli.newton else 'gradient descent'}{'  potential-velocity' if ts.potential_velocity else ''}  auxiliary-B-field={str(cli.auxiliary_B_field).lower()}  "
          f"scheme={cli.scheme}{'  helicity-correction' if cli.helicity_correction else ''}  "
          f"smoothing={cli.velocity_smoothing_order}@{ts.velocity_smoothing_scale:.3e} "
          f"cfl={cli.cfl}  steps<={cli.steps} chunk={cli.chunk} floor-tol={cli.floor_tol:.1e} "
          f"reconnect-every={cli.reconnect_every}"
          + ((f" (eps {cli.reconnect_eps:g} h_r^2 each" if cli.reconnect_eps is not None
              else f" ({cli.reconnect_helicity:.2%} of H each")
             + (f", steps {cli.reconnect_window})" if cli.reconnect_window else ")") if cli.reconnect_every else "")
          + (f" resistivity={cli.resistivity:g} h_r^2 (B* smoothed {cli.reference_smoothing:g} h_r^2)"
             if cli.resistivity else "")
          + " ===",
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
                         E0=res.E0, E_removed=res.E0 - res.qoi["E"][-1], F_final=res.trace["F"][-1],
                         resid_final=res.trace["resid"][-1],
                         resid_window_mean=float(sum(res.trace["resid"][-res.chunk:]) / res.chunk),
                         best_step=int(res.state.step_best), best_resid=float(res.state.resid_best),
                         **last))
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)

    res = relax(state, ts, steps=cli.steps, chunk=cli.chunk, it0=it0, floor_tol=cli.floor_tol,
          reconnect_every=cli.reconnect_every,
          reconnect_helicity=cli.reconnect_helicity,
          reconnect_eps=None if cli.reconnect_eps is None else cli.reconnect_eps * h_r_sq,
          reconnect_window=None if cli.reconnect_window is None
          else tuple(int(v) for v in cli.reconnect_window.split(":")), on_chunk=save)
    write_checkpoint(os.path.join(ckpt_dir, "best.h5"),
                     initial_state(res.state.B_best, ts, step=int(res.state.step_best)), int(res.state.step_best))
    print(f"wrote {out}/relax.json and {ckpt_dir}/ (best.h5: step {int(res.state.step_best)}, "
          f"residual {float(res.state.resid_best):.3e})", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"], os.environ["MRX_RESIDUAL_DTYPE"] = PRECISIONS[cli.precision]
    main(cli)
