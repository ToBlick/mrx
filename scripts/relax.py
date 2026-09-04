"""Relax a magnetic field toward minimum energy at fixed helicity.

Builds the geometry, projects an initial field, then descends the magnetic
energy with the incompressible, helicity-conserving flow of
:class:`mrx.relaxation.TimeStepper` until the force residual floors, the step
budget is spent, or the wall-clock budget runs out. The descent is ideal,
``B_{n+1} = B_n + dt curl(u x B)`` (or ``u x H`` with the auxiliary field);
reconnection, when asked for, is one resistive solve between chunks. The
fixed point is ``J x B = grad p`` with ``p`` the Leray multiplier, so the
relaxed state is a finite-beta equilibrium, not a force-free field.

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
      --solve-tol TOL [sqrt(eps)]  inner solve tolerance
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
                                   pressures, beta; see "Two pressures"), a
                                   snapshot, the checkpoint and the outputs
                                   are written, and the floor, reconnect
                                   and wall-time tests run, once per chunk;
                                   --steps is a multiple of it
      --reconnect-every K [0]      see "Reconnection series"; 0 = off
      --reconnect-helicity X [0.01] the helicity each reconnection spends,
                                   |dH| / |H|
      --floor-tol TOL [1e-3]       see "Stopping criterion"
      --out DIR [outputs/relax/<date>/<time>]
      --checkpoint PATH [<out>/state.eqx], --restart PATH  see below

Two pressures:
    The strong pressure ``p`` is the Leray multiplier of ``compute_force``:
    a 3-form, the multiplier of the constrained energy principle, with
    ``dp/dn = 0`` on the wall by construction (the Lorentz force is projected
    onto the Dirichlet 2-form space first, which discards its normal
    component). The weak pressure ``p_w`` (``mrx.relaxation.weak_pressure``)
    projects the same ``J x B`` (or ``J x H``) onto the natural 1-form space
    and Helmholtz-decomposes it there with ``p_w`` in the Dirichlet 0-form
    space, ``p_w = 0`` on the wall; it sees the wall force. At every qoi
    sample the script records
    ``gradp_cmp = ||Pi_2 grad p_w - grad_w p||_{M2} / ||Pi_2 grad p_w||_{M2}``
    (gauge-free; ``grad_w p`` is the 3-form's weak gradient in the Dirichlet
    2-form space, ``Pi_2`` projects the exact ``grad p_w`` onto the same space
    so both lose the same normal trace), ``p_cmp``, the L2 distance of the
    two pressures as functions with their means removed, relative to
    ``p_w``'s, ``weak_resid = ||J x B - grad p_w|| / ||J x B||`` in the
    natural 1-form space,
    ``dpdn_wall = max |dp_w/dn| / max |grad p_w|`` and
    ``JxBn_wall = max |(J x B) . n| / max |grad p_w|`` on the wall,
    ``beta_vol = int p_w dV / int B^2/2 dV`` and ``beta_axis``, the same
    ratio on the coordinate axis (logical r = 0: the innermost radial
    quadrature layer, averaged over theta and zeta). Code units:
    the magnetic pressure is ``B^2/2``.

Stopping criterion:
    The relative force residual ``|F|_M / ||grad(B^2/2)||`` is recorded at
    every step. The run stops when its mean over the last chunk (``--chunk``
    steps) drops below ``--floor-tol``. The relaxation guarantees ``dE/dt <= 0``
    only, so the residual is not monotone; the window mean is the quantity,
    never the last value. Calibration: on the W7-X Clebsch run at (8,16,8)
    p=3 in float64 the residual reaches ~1.7e-3 at step 500 and floors around
    1e-3 by step 1000-3000. In float32 the residual floors at the
    solve-tolerance level (~2e-3 at tol 1e-5), so a ``--floor-tol`` below
    that never fires; ``--steps`` or ``--seconds`` end the run instead.

Output (``--out``):
    relax.json   parameters; the per-step trace (E, F, resid, dt, dt_star,
                 cfl, cos, gain, div, picard_it, picard_resid, dE_meas,
                 dE_pred); the sampled quantities of interest ``qoi``
                 (helicity, ||J||/||B||, wall, and the pressure diagnostics
                 gradp_cmp, p_cmp, weak_resid, dpdn_wall, JxBn_wall,
                 beta_vol, beta_axis); the initial-condition summary ``ic`` and the
                 ``summary`` with the stopping reason, both carrying the
                 same pressure diagnostics. Rewritten at every chunk.
    B.h5         B_ic, B_final, the strong pressures p_ic, p_final (3-form
                 DoFs) and the weak pressures pw_ic, pw_final (Dirichlet
                 0-form DoFs), all evaluated at the field stored next to
                 them, with the run parameters as attributes; ``geometry``
                 as given, ``geometry_path`` resolved and ``ic`` the kind of
                 initial condition (vmec, gvec, or the analytic map's name).
                 Written at every save.

Reconnection series:
    ``--reconnect-every K`` runs the ideal descent and, every ``K`` steps
    (rounded to a whole number of chunks), checkpoints the current field and
    reconnects it with one backward-Euler solve of ``(M + eps L) delta =
    -eps L B``, then restarts the optimiser on the diffused field and carries
    on. The ideal descent is a power law, ``resid ~ t^-a`` with a = 0.2 at
    (16,32,32) p = 2 gamma = 1 and 1/3 at n = 8 and 12, never a plateau (the
    ideal li383 arms, 5000-6000 steps, 2026-09-03), so there is no stall to
    detect and the interval is a choice: any "stalled" test on a power law
    is a step count in disguise. The dose is set by the helicity it spends:
    a resistive increment ``delta = -eps curl curl B`` changes the helicity
    by ``dH = -2 eps int J . B`` to first order, so ``eps = X |H| / (2 |int J
    . B|)`` with ``X = --reconnect-helicity`` and ``J``, ``H`` those of the
    field being reconnected (``results["reconnect"]`` records the target,
    ``eps``, the pairing and the helicity actually spent). Reconnection ``k``
    leaves ``<out>/reconnect/<k>/B.h5`` (the field before the solve with its
    pressures, in the layout of ``B.h5``, so ``poincare_relax.py --fields
    reconnect`` reads it) and ``state.eqx`` (a ``--restart`` file). The run
    ends on ``--steps`` or ``--seconds``; its outcome is the series of ideal
    equilibria, one per reconnection plus the final field, to choose from.

``--checkpoint PATH`` (default ``<out>/state.eqx``) serialises the full
descent state -- B, the pressure and warm-start guesses, the L-BFGS pair --
together with the step number at every save and at the end
(``equinox.tree_serialise_leaves``); ``--restart PATH`` continues from such
a file: the step counter, and with it the snapshot steps, carries on from
the saved step, ``--steps`` counts the steps of THIS run, and the trace and
QoI samples are this run's. The IC diagnostics and ``B_ic`` still refer to
the initial condition built from ``--geometry``, which must be the same.

The trace records the linesearch identity ``dE_pred = -dt (F,u)_M / 2``
against the measured decrease: it is an operator identity (curl adjointness,
the cross-product sign, Leray M-orthogonality) and holds to round-off under
the explicit scheme. Under the midpoint scheme the exact change is ``-dt
(F_mid, u)_M`` with the force at the midpoint field, second order in ``dt``
away from the prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


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
                    help="steps per compiled chunk; trace, qoi sample, snapshot, checkpoint, "
                         "outputs and the floor / reconnect / wall-time tests once per chunk")
    ap.add_argument("--floor-tol", type=float, default=1e-3,
                    help="stop when the last chunk's mean relative force residual is below this")
    ap.add_argument("--reconnect-every", type=int, default=0,
                    help="checkpoint the field and reconnect it with one resistive solve "
                         "every K steps, rounded to whole chunks; 0 = off (see the docstring)")
    ap.add_argument("--reconnect-helicity", type=float, default=0.01,
                    help="the helicity each reconnection spends, |dH| / |H|")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="write the descent state (equinox pytree + step) here at every "
                         "save and at the end; default <out>/state.eqx")
    ap.add_argument("--restart", default=None,
                    help="continue from a --checkpoint file of the same geometry, mesh, "
                         "degree and precision")
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


def make_force_normaliser(seq):
    """``||grad(B^2/2)||_L2``: the scale the force residual is measured against.

    grad p is a real scale too (the scheme converges to J x B = grad p) but
    vanishes in the low-beta limit; grad(B^2/2) has the same units and stays
    O(1). Computed through the sequence: project B^2/2 onto 0-forms (one load
    and one M_0 solve), take the discrete gradient, measure it.
    """
    import jax.numpy as jnp
    from mrx.quadrature import evaluate_at_xq, integrate_against

    quad_shape = seq.quad.shape
    ci0, cs0 = seq._form_comp_info(0)
    ci2, cs2 = seq._form_comp_info(2)

    def normaliser(B_dof):
        B_jk = evaluate_at_xq(seq.E(2, True).T @ B_dof, ci2, cs2, quad_shape, 3)
        bsq = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk)
        f_jk = (0.5 * bsq * seq.quad.w / seq.jacobian_j)[:, None]
        q = seq.E(0) @ integrate_against(f_jk, ci0, cs0, quad_shape)
        w0 = seq.apply_inverse_mass_matrix(q, 0, dirichlet=False)
        g1 = seq.apply_strong_grad(w0, dirichlet_in=False, dirichlet_out=False)
        return seq.l2_norm(g1, 1, dirichlet=False)

    return normaliser


def main(cli):
    import equinox as eqx
    import h5py
    import jax

    import mrx
    from mrx.geometry import build_sequence, geometry_kind, parse_r_refine, read_analytic
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import (analytic_profile_form,
                                        clebsch_potential_form, divergence_norm,
                                        potential_two_form, lambda_dirichlet_energy,
                                        resonant_rho, leray_clean, make_lambda,
                                        make_profiles, project_reference_two_form)
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (IntegrationScheme, TimeStepper, chunk_runner,
                                compute_force, compute_helicity, initial_state, resistive_step,
                                pressure_diagnostics, weak_pressure)
    import jax.numpy as jnp

    if cli.precision != str(mrx.DTYPE):
        raise ValueError(f"--precision {cli.precision} but mrx runs in {mrx.DTYPE}")
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)
    ns = tuple(int(v) for v in cli.ns.split(","))
    kind = geometry_kind(cli.geometry)          # vmec, gvec, or the analytic map's name
    from_file = kind in ("vmec", "gvec")
    geometry_path = os.path.abspath(cli.geometry)
    out = cli.out or os.path.join("outputs", "relax", time.strftime("%Y-%m-%d"),
                                  time.strftime("%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    params = {k: v for k, v in vars(cli).items()}
    params.update(ns=list(ns), out=out, geometry_path=geometry_path, ic=kind)
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.solve_maxiter, tol=cli.solve_tol,
                              nfp=cli.nfp, r_windows=parse_r_refine(cli.r_refine))
    ops = seq.set_operators(compute_nullspaces(seq, ops))
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} tol={seq.tol:.1e}  "
          f"n2_dbc={seq.n(2, True)}  operators+nullspaces "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    # --- initial condition -----------------------------------------------
    if from_file:
        cb = load_clebsch(cli.geometry)
        lam_norm, lam_energy = lambda_dirichlet_energy(cb["lam_h"], seq)
        results["lambda"] = dict(norm_sq=lam_norm, dirichlet_energy=lam_energy)
        print(f"[ic] lambda: ||lam||^2 {lam_norm:.4e}  <lam, L0 lam> {lam_energy:.4e}  "
              f"ratio {lam_energy / lam_norm:.4e}", flush=True)
        print(f"[ic] {kind} Clebsch data from {geometry_path}  nfp={cb['nfp']}  iota (full turn) "
              f"{cb['dchi'][1] / cb['dPhi'][1]:+.5f} -> "
              f"{cb['dchi'][-1] / cb['dPhi'][-1]:+.5f}")
    else:
        prof = read_analytic(cli.geometry)["profile"]
        iota, dPhi = make_profiles(prof["iota"][0], prof["iota"][1], prof["iota_exp"], prof["flux_exp"])
        omega_ref = analytic_profile_form(
            iota, dPhi, make_lambda([(int(m), int(n), float(a)) for m, n, a in prof.get("lambda", [])]))
        print(f"[ic] {kind}: iota {prof['iota'][0]:g} -> {prof['iota'][1]:g} (exponent "
              f"{prof['iota_exp']:g}), Phi' ~ rho^{prof['flux_exp']:g}, "
              f"{len(prof.get('lambda', []))} lambda modes")
    t1 = time.perf_counter()
    if from_file:
        # B = d A' in the complex: exactly divergence-free, nothing to clean.
        # (The pointwise 2-form of clebsch_form, L2-projected and Leray-cleaned,
        # carried the interpolant's derivatives into the current: a 20^3 export
        # relaxed to a chaotic core that way and to the 50^3 core this way.)
        seed = None
        if cli.seed:
            m, n, rho0, width = (float(v) for v in cli.seed.split(","))
            seed = (int(m), int(n), rho0, width, cli.seed_eps)
            print(f"[ic] seed (m, n) = ({int(m)}, {int(n)}) at rho0 {rho0:g}, width {width:g}, "
                  f"eps {cli.seed_eps:.2e}; the file's |iota| = nfp n / m chain is at "
                  f"rho = {resonant_rho(cb, int(m), int(n)):.3f}")
        B0, B_norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
        div_raw = div0 = divergence_norm(seq, B0)
        moved = 0.0
        print(f"[ic] B = dA' from the histopolated potential: ||div B|| {div0:.3e}, "
              f"wall-normal part discarded {wall:.3e}")
    else:
        B0, B_norm = project_reference_two_form(seq, omega_ref)
        div_raw = divergence_norm(seq, B0)
        B0, moved = leray_clean(seq, B0)
        div0 = divergence_norm(seq, B0)
        print(f"[ic] Leray-projected: ||div B|| {div_raw:.3e} -> {div0:.3e}  "
              f"(moved the field by {moved:.3e})")
    H0, _ = compute_helicity(B0, seq, jnp.zeros(seq.n(1, True)))
    E0 = 0.5 * float(seq.l2_norm_sq(B0, 2))
    normaliser = jax.jit(make_force_normaliser(seq))
    gradp0 = float(normaliser(B0))

    @jax.jit
    def force_probe(B, p_guess, H_guess, JxH_guess):
        return compute_force(B, seq, cli.auxiliary_B_field,
                             p_guess=p_guess, H_guess=H_guess, JxH_guess=JxH_guess)

    def pressure_probe_eager(B, p, J, X, pw_guess):
        """The weak pressure of ``compute_force``'s ``(p, J, X)`` at ``B``,
        ||J||/||B||, and the strong/weak comparison (see "Two pressures")."""
        p_w, F_w, v = weak_pressure(J, X, seq, cli.auxiliary_B_field, p_guess=pw_guess)
        JoverB = seq.l2_norm(J, 1) / seq.l2_norm(B, 2)
        return p_w, JoverB, pressure_diagnostics(B, p, p_w, F_w, v, seq)

    # The IC call below runs eagerly and builds every lazily-cached core the
    # probe touches (the 1->2 projection is host-side numpy on first use, and
    # nothing before this point applies it); the loop uses the compiled one.
    pressure_probe = jax.jit(pressure_probe_eager)

    def pressure_line(d):
        return (f"beta_vol={d['beta_vol']:.3e}  beta_axis={d['beta_axis']:.3e}  "
                f"|grad pw - grad p|/|grad pw|={d['gradp_cmp']:.3e}  |pw - p|/|pw|={d['p_cmp']:.3e}  "
                f"weak_resid={d['weak_resid']:.3e}  "
                f"wall dpw/dn={d['dpdn_wall']:.3e}  (JxB).n={d['JxBn_wall']:.3e}")

    F0, p0, J0, X0, _ = compute_force(B0, seq, cli.auxiliary_B_field)
    pw0, JoverB0, diag0 = pressure_probe_eager(B0, p0, J0, X0, jnp.zeros(seq.n(0, True)))
    diag0 = {k: float(v) for k, v in diag0.items()}
    F0n = float(seq.l2_norm(F0, 2))
    resid0 = F0n / gradp0
    print(f"[ic] {kind} IC in {time.perf_counter() - t1:.1f}s  ||B||_M raw "
          f"{B_norm:.6e}  E={E0:.6e}  ||F||={F0n:.4e}  resid "
          f"{resid0:.4e}  H={float(H0):+.6e}",
          flush=True)
    print(f"[ic] {pressure_line(diag0)}", flush=True)
    results["ic"] = dict(B_norm_raw=B_norm, div_raw=div_raw, div=div0,
                         leray_moved=moved, E=E0, F=F0n, gradp=gradp0,
                         resid=resid0, H=float(H0),
                         JoverB=float(JoverB0), **diag0)

    # --- the descent -------------------------------------------------------
    ts = TimeStepper(
        seq=seq, auxiliary_B_field=cli.auxiliary_B_field,
        scheme={"explicit": IntegrationScheme.EXPLICIT,
                "midpoint": IntegrationScheme.IMPLICIT_MIDPOINT}[cli.scheme],
        cfl=cli.cfl, history_size=cli.history,
        velocity_smoothing_order=cli.velocity_smoothing_order,
        velocity_smoothing_scale=cli.velocity_smoothing_scale)
    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])

    state = initial_state(B0, ts)
    it0 = 0
    ckpt = cli.checkpoint or os.path.join(out, "state.eqx")
    if cli.restart:
        state, it_saved = eqx.tree_deserialise_leaves(cli.restart, like=(state, jnp.int32(0)))
        it0 = int(it_saved)
        print(f"[restart] {cli.restart}: descent state at step {it0}", flush=True)
    params["start_step"] = it0
    # One compiled chunk: --chunk steps, the per-step scalars stacked.
    run = chunk_runner(ts, cli.chunk, extra=dict(resid=lambda st: st.F_norm / normaliser(st.B_n)))
    reconnect_fn = jax.jit(lambda B, eps: resistive_step(B, seq, eps))
    # int J . B of the field being reconnected: J a Dirichlet 1-form against
    # the dual 1-form P B; it sets the dose (see "Reconnection series").
    pairing = jax.jit(lambda J, B: J @ seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True))
    reconnect_every = max(1, round(cli.reconnect_every / cli.chunk)) * cli.chunk if cli.reconnect_every else 0
    if reconnect_every != cli.reconnect_every:
        print(f"[reconnect] --reconnect-every {cli.reconnect_every} rounded to {reconnect_every} "
              f"({reconnect_every // cli.chunk} chunks)", flush=True)

    p_ic = np.asarray(state.p)
    pw_ic = np.asarray(pw0)
    snaps = [(0, np.asarray(B0), np.asarray(pw0))]
    pw_guess = pw0
    latest = {"p": p_ic, "pw": pw_ic, "diag": diag0}

    tr = {k: [] for k in ("E", "F", "resid", "dt", "dt_star", "cfl", "div", "cos",
                          "gain", "picard_it", "picard_resid", "dE_meas", "dE_pred")}
    qoi = {k: [] for k in ("it", "helicity", "JoverB", "wall", "gradp_cmp", "p_cmp",
                           "weak_resid", "dpdn_wall", "JxBn_wall", "beta_vol", "beta_axis")}
    results["reconnect"] = []
    E_prev = E0
    stop = "steps"
    t_arm = time.perf_counter()
    t_qoi = 0.0     # time in the qoi samples, saves and reconnections; the recorded wall excludes it
    n_done = 0
    print(f"\n=== L-BFGS m={cli.history}  auxiliary-B-field={str(cli.auxiliary_B_field).lower()}  "
          f"scheme={cli.scheme}  "
          f"smoothing={cli.velocity_smoothing_order}@{cli.velocity_smoothing_scale} cfl={cli.cfl}  "
          f"steps<={cli.steps} chunk={cli.chunk} floor-tol={cli.floor_tol:.1e} "
          f"reconnect-every={reconnect_every}"
          + (f" ({cli.reconnect_helicity:.2%} of H each)" if reconnect_every else "") + " ===",
          flush=True)

    def write_field(path, B_now, p_now, pw_now, snapshots=False, **extra):
        """The field and its pressures next to the IC's, as poincare_relax.py reads them."""
        with h5py.File(path, "w") as fh:
            fh.create_dataset("B_ic", data=np.asarray(B0))
            fh.create_dataset("B_final", data=np.asarray(B_now))
            fh.create_dataset("p_ic", data=p_ic)
            fh.create_dataset("p_final", data=p_now)
            fh.create_dataset("pw_ic", data=pw_ic)
            fh.create_dataset("pw_final", data=pw_now)
            if snapshots:
                fh.create_dataset("snapshot_steps", data=np.array([k for k, _, _ in snaps]))
                fh.create_dataset("B_snapshots", data=np.stack([b for _, b, _ in snaps]))
                fh.create_dataset("pw_snapshots", data=np.stack([w for _, _, w in snaps]))
            for k, v in {**params, **extra}.items():
                fh.attrs[k] = "" if v is None else v

    def save():
        results["trace"] = tr
        results["qoi"] = qoi
        results["summary"] = dict(
            steps=n_done, stop=stop, wall=time.perf_counter() - t_arm - t_qoi,
            E_final=tr["E"][-1], F_final=tr["F"][-1], resid_final=tr["resid"][-1],
            resid_window_mean=resid_now, **latest["diag"])
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)
        eqx.tree_serialise_leaves(ckpt, (state, jnp.int32(it0 + n_done)))
        # The field goes out with every save, not only the last one: a run
        # that hits its wall-time limit leaves the state it reached.
        write_field(os.path.join(out, "B.h5"), state.B_n, latest["p"], latest["pw"], snapshots=True)

    def sample_qoi(state, it):
        """Helicity, ||J||/||B||, the weak pressure and its diagnostics at the
        state's field (the force at the CURRENT field; state.p, H, JxH are
        the step's values at the previous one, they warm-start it and are
        refreshed from it). Appends to ``qoi``; returns the refreshed state."""
        nonlocal pw_guess, latest
        _, p, J, X, JxX = force_probe(state.B_n, state.p, state.H, state.JxH)
        pw_guess, JoverB, diag = pressure_probe(state.B_n, p, J, X, pw_guess)
        h, A_new = get_helicity(state.B_n, seq, state.A)
        H = X if cli.auxiliary_B_field else state.H
        state = eqx.tree_at(lambda s: (s.p, s.H, s.JxH, s.A), state, (p, H, JxX, A_new))
        diag = {k: float(v) for k, v in diag.items()}
        latest = {"p": np.asarray(p), "pw": np.asarray(pw_guess), "diag": diag}
        qoi["it"].append(it)
        qoi["wall"].append(time.perf_counter() - t_arm - t_qoi)
        qoi["JoverB"].append(float(JoverB))
        qoi["helicity"].append(float(h))
        for k, v in diag.items():
            qoi[k].append(v)
        return state, float(h), float(JoverB), diag

    tq = time.perf_counter()
    state, h0, _, _ = sample_qoi(state, it0)   # qoi[...][0] is the start of THIS run
    t_qoi += time.perf_counter() - tq
    for _ in range(cli.steps // cli.chunk):
        state, chunk = run(state, it0 + n_done)
        chunk = {k: np.asarray(v) for k, v in chunk.items()}
        n_done += cli.chunk
        it = it0 + n_done
        with np.errstate(invalid="ignore"):   # a backward line-search step has no gain
            cos = chunk["Fu"] / (chunk["F"] * chunk["v"])
            tr["cos"].extend(cos.tolist())
            tr["gain"].extend(((chunk["Fu"] / chunk["dt"]) ** 0.5 / chunk["v"]).tolist())
        tr["dE_meas"].extend(np.diff(chunk["E"], prepend=E_prev).tolist())
        tr["dE_pred"].extend((-0.5 * chunk["dt"] * chunk["Fu"]).tolist())
        for k in ("E", "F", "resid", "dt", "dt_star", "cfl", "div", "picard_it", "picard_resid"):
            tr[k].extend(chunk[k].tolist())
        E_prev = float(chunk["E"][-1])
        resid_now = float(chunk["resid"].mean())
        tq = time.perf_counter()
        state, h, JoverB, diag = sample_qoi(state, it)
        snaps.append((it, np.asarray(state.B_n), latest["pw"]))
        print(f"  it {it:>5d}  E={E_prev:.8e}  |F|={chunk['F'][-1]:.4e}  "
              f"resid={resid_now:.3e} (chunk mean)  H={h:+.6e}  dH={h - h0:+.3e}  "
              f"dt={chunk['dt'].mean():+.3e}  cos min={np.nanmin(cos):+.4f}  "
              f"divB={chunk['div'].max():.2e}  picard max={int(chunk['picard_it'].max())}  "
              f"[{tq - t_arm - t_qoi:.0f}s solve +{t_qoi:.0f}s qoi]\n           {pressure_line(diag)}",
              flush=True)
        if reconnect_every and n_done % reconnect_every == 0 and n_done < cli.steps:
            k = len(results["reconnect"]) + 1
            rdir = os.path.join(out, "reconnect", str(k))
            os.makedirs(rdir, exist_ok=True)
            # The field before the solve: field, pressures (just sampled), restart file.
            write_field(os.path.join(rdir, "B.h5"), state.B_n, latest["p"], latest["pw"],
                        reconnect=k, reconnect_step=it)
            eqx.tree_serialise_leaves(os.path.join(rdir, "state.eqx"), (state, jnp.int32(it)))
            # The dose from the helicity to spend: dH = -2 eps int J . B.
            _, _, J, _, _ = force_probe(state.B_n, state.p, state.H, state.JxH)
            JB = float(pairing(J, state.B_n))
            eps = cli.reconnect_helicity * abs(h) / (2.0 * abs(JB))
            ev = dict(k=k, it=it, resid=resid_now, eps=eps, JB=JB,
                      helicity_target=cli.reconnect_helicity,
                      F_before=float(state.F_norm), helicity_before=h, JoverB_before=JoverB,
                      **{f"{kk}_before": v for kk, v in diag.items()})
            # Reconnect and restart the optimiser on the diffused field; the
            # qoi gets a second sample at this step, the reconnected field's.
            B_new, info, rel = reconnect_fn(state.B_n, eps)
            state = initial_state(B_new, ts, dt=float(state.dt))
            state, h, JoverB, diag = sample_qoi(state, it)
            ev.update(solve_it=int(info), moved=float(rel), F_after=float(state.F_norm),
                      helicity_after=h, helicity_spent=(h - ev["helicity_before"]) / abs(ev["helicity_before"]),
                      JoverB_after=JoverB, **{f"{kk}_after": v for kk, v in diag.items()})
            results["reconnect"].append(ev)
            print(f"  [reconnect {k}] at it={it}: chunk mean {resid_now:.3e}; "
                  f"eps={eps:.3e} for {cli.reconnect_helicity:.2%} of H ({int(info)} it, "
                  f"moved {float(rel):.2e}); |F| {ev['F_before']:.3e} -> {ev['F_after']:.3e}, "
                  f"H {ev['helicity_before']:+.6e} -> {ev['helicity_after']:+.6e} "
                  f"({ev['helicity_spent']:+.2%}), J/B {ev['JoverB_before']:.3f} -> "
                  f"{ev['JoverB_after']:.3f}; wrote {rdir}", flush=True)
        if resid_now < cli.floor_tol:
            stop = "floor"
            print(f"  [floor] chunk mean of the force residual {resid_now:.3e} below "
                  f"{cli.floor_tol:.1e} at it={it}", flush=True)
        elif cli.seconds is not None and time.perf_counter() - t_arm > cli.seconds:
            stop = "seconds"
            print(f"  [budget] {cli.seconds:.0f} s spent at it={it}", flush=True)
        else:
            save()
        t_qoi += time.perf_counter() - tq
        if stop != "steps":
            break

    wall = time.perf_counter() - t_arm - t_qoi
    dEm, dEp = np.array(tr["dE_meas"]), np.array(tr["dE_pred"])
    ident = np.abs(dEm - dEp) / E0
    resid_tr = np.array(tr["resid"])
    print(f"\n--- {n_done} steps in {wall:.1f}s ({wall / max(n_done, 1):.2f} s/step), "
          f"stopped on: {stop}")
    print(f"    E {E0:.8e} -> {tr['E'][-1]:.8e}  "
          f"({(E0 - tr['E'][-1]) / E0:.4%} of the initial energy removed)")
    print(f"    |F| {F0n:.4e} -> {tr['F'][-1]:.4e}   residual "
          f"{resid0:.4e} -> {resid_tr[-1]:.4e}  (mean over the last "
          f"{min(cli.chunk, n_done)} steps {resid_tr[-cli.chunk:].mean():.4e}, "
          f"min {resid_tr.min():.4e})")
    print(f"    linesearch identity |dE_meas - dE_pred| / E0: median "
          f"{np.median(ident):.3e}  max {ident.max():.3e}"
          + ("  (not an identity under the midpoint scheme)" if cli.scheme == "midpoint" else ""))
    print(f"    energy increases on {int((dEm > 0).sum())}/{n_done} steps;  "
          f"||div B|| max {max(tr['div']):.3e};  ||J||/||B|| "
          f"{qoi['JoverB'][0]:.4e} -> {qoi['JoverB'][-1]:.4e}")
    h = np.array(qoi["helicity"])
    print(f"    helicity {h[0]:+.6e} -> {h[-1]:+.6e}  drift {h[-1] - h[0]:+.3e}"
          f"  relative {(h[-1] - h[0]) / abs(h[0]):+.3e}", flush=True)
    print(f"    pressures at the IC:  {pressure_line(diag0)}")
    print(f"    pressures at the end: {pressure_line(latest['diag'])}", flush=True)
    dts, dt_star = np.array(tr["dt"]), np.array(tr["dt_star"])
    print(f"    CFL cap (C={cli.cfl}) bound on {int((dts < dt_star).sum())}/{n_done} steps;  "
          f"dt/dt* min {(dts / dt_star).min():.3f} mean {(dts / dt_star).mean():.3f};  "
          f"CFL number taken max {(dts * np.array(tr['cfl'])).max():.3f}")
    if cli.scheme == "midpoint":
        pit, pres = np.array(tr["picard_it"]), np.array(tr["picard_resid"])
        print(f"    midpoint solve: increment evaluations mean {pit.mean():.2f}  max {pit.max()};  "
              f"defect max {pres.max():.2e};  unconverged on {int((pres > ts.picard_tol).sum())}/{n_done} "
              f"steps (tolerance {ts.picard_tol:.1e})", flush=True)
    save()
    print(f"wrote {out}/relax.json and {out}/B.h5", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"] = cli.precision
    main(cli)
