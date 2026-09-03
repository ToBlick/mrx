"""Relax a magnetic field toward minimum energy at fixed helicity.

Builds the geometry, projects an initial field, then descends the magnetic
energy with the incompressible, helicity-conserving flow of
:class:`mrx.relaxation.TimeStepper` until the force residual floors, the step
budget is spent, or the wall-clock budget runs out. Each step is
operator-split (Lie): ideal transport, ``B_ideal = B_n + dt curl(u x H)``,
then implicit resistive diffusion of ``B_ideal``; first order in ``dt``. The
fixed point is ``J x B = grad p`` with ``p`` the Leray multiplier, so the
relaxed state is a finite-beta equilibrium, not a force-free field.

Canonical invocation (one GPU; see slurm/README.md)::

    python -u scripts/relax.py --geometry data/GVEC_State_final.dat

Flags, defaults in brackets:
    Geometry and discretisation:
      --geometry G (required)      toroid, cylinder, rot-ellipse, or an
                                   equilibrium file: GVEC state (.dat),
                                   VMEC wout (.nc)
      --nfp N [file value]         field periods of a file that declares
                                   them wrong
      --ns R,T,Z [8,16,16]         spline resolution (also the map's)
      --p P [2]                    spline degree; p+1 Gauss points per span
      --maxiter N [2000]           iteration budget of every inner solve
      --tol TOL [sqrt(eps)]        inner solve tolerance
      --precision {float32,float64} [float32]  exported as MRX_DTYPE before
                                   mrx is imported
    Initial condition (always Leray-projected):
      --ic {clebsch,analytic,dzeta} [clebsch]
          clebsch:  Phi', chi' and lambda from the geometry file
                    (needs a file geometry; GVEC or VMEC)
          analytic: prescribed profiles on the logical grid, no data
          dzeta:    the constant 2-form (0,0,1); relaxes to the harmonic field
      analytic IC only (ignored for --ic clebsch):
      --iota I0,I1 [0.4,0.9]       iota on axis and at the edge
      --iota-exp E [2.0]           iota(rho) = I0 + (I1 - I0) rho^E
      --flux-exp Q [1.0]           Phi'(rho) ~ rho^Q
      --lam SPEC [""]              lambda modes "m,n,amp;..."
    Descent:
      --method {gradient,lbfgs} [lbfgs]
      --history M [1]              L-BFGS secant pairs (1 = memoryless = CG)
      --velocity-smoothing-order G [0], --velocity-smoothing-scale MU [0.0]
                                   descent direction v = (I - MU L)^-G F
      --dt-mode {linesearch,fixed} [linesearch]
                                   the exact energy-minimising step, or --dt0
      --dt0 DT [1.0]               the step for --dt-mode fixed
      --cfl C [0.5]                cap the step at C / (largest logical CFL
                                   number of the velocity); inf disables it
      --eta-max ETA [0.0]          peak resistivity; backward Euler in defect
                                   form after the ideal step, any size is
                                   stable; helicity is not conserved
      --eta-every K [1]            resistive solve every K steps, diffusing
                                   over the accumulated time (float32 needs
                                   K of 10-100 at eta ~ 1e-4)
      --eta-schedule {tanh,constant,linear,pulse} [tanh]
                                   tanh drops eta to ~0 over the middle third
                                   of --steps so the run ends ideal; pulse is
                                   eta-max on the window(s) of --eta-pulse
                                   and 0 elsewhere (the resistive clock is
                                   reset while it is off, so --eta-every
                                   equal to the width makes one solve of
                                   eta * window time per pulse)
      --eta-pulse S,W[,P] [2000,100] pulse start step, width in steps and,
                                   optionally, the period of repeated pulses
      --presmooth K [0]            up to K backward-Euler steps of
                                   dB/dt = -curl curl B on the IC, force off,
                                   before the descent (regularises a coarsely
                                   sampled IC); each step is (M + eps L) in
                                   defect form with eps = --presmooth-eps
      --presmooth-eps EPS [1e-3]   eta dt of one pre-smoothing step
      --presmooth-jb X [none]      stop pre-smoothing once ||J||/||B|| <= X
    Budgets and output:
      --steps N [3000]             maximum number of steps
      --seconds S [none]           wall-clock budget of the descent loop
      --floor-tol TOL [1e-3]       see "Stopping criterion"
      --floor-steps W [100]        number of steps over which the force
                                   residual is averaged before it is compared
                                   with --floor-tol
      --qoi-every N [250]          steps between the quantity-of-interest
                                   samples (helicity, the two pressures,
                                   beta; see "Two pressures")
      --out DIR [outputs/relax/<date>/<time>]

Two pressures:
    The strong pressure ``p`` is the Leray multiplier of ``compute_force``:
    a 3-form, the multiplier of the constrained energy principle, with
    ``dp/dn = 0`` on the wall by construction (``J x H`` is projected onto the
    Dirichlet 2-form space first, which discards ``(J x H) . n``). The weak
    pressure ``p_w`` (``mrx.relaxation.weak_pressure``) projects ``J x H``
    onto the natural 1-form space and Helmholtz-decomposes it there with
    ``p_w`` in the Dirichlet 0-form space, ``p_w = 0`` on the wall; it sees
    the wall force. At every qoi sample the script records
    ``gradp_cmp = ||Pi_2 grad p_w - grad_w p||_{M2} / ||Pi_2 grad p_w||_{M2}``
    (gauge-free; ``grad_w p`` is the 3-form's weak gradient in the Dirichlet
    2-form space, ``Pi_2`` projects the exact ``grad p_w`` onto the same space
    so both lose the same normal trace), ``p_cmp``, the L2 distance of the
    two pressures as functions with their means removed, relative to
    ``p_w``'s, ``weak_resid = ||J x H - grad p_w|| / ||J x H||`` in the
    natural 1-form space,
    ``dpdn_wall = max |dp_w/dn| / max |grad p_w|`` and
    ``JxBn_wall = max |(J x H) . n| / max |grad p_w|`` on the wall,
    ``beta_vol = int p_w dV / int B^2/2 dV`` and ``beta_axis``, the same
    ratio on the coordinate axis (logical r = 0: the innermost radial
    quadrature layer, averaged over theta and zeta). Code units:
    the magnetic pressure is ``B^2/2``.

Stopping criterion:
    The relative force residual ``|F|_M / ||grad(B^2/2)||`` is recorded at
    every step. The run stops when its mean over the last ``--floor-steps``
    steps drops below ``--floor-tol``. The relaxation guarantees ``dE/dt <= 0``
    only, so the residual is not monotone; the window mean is the quantity,
    never the last value. Calibration: on the W7-X Clebsch run at (8,16,8)
    p=3 in float64 the residual reaches ~1.7e-3 at step 500 and floors around
    1e-3 by step 1000-3000. In float32 the residual floors at the
    solve-tolerance level (~2e-3 at tol 1e-5), so a ``--floor-tol`` below
    that never fires; ``--steps`` or ``--seconds`` end the run instead.

Output (``--out``):
    relax.json   parameters; the per-step trace (E, F, resid, dt, dt_star,
                 cfl, cos, gain, div, eta, res_it, res_delta, dE_meas,
                 dE_pred); the sampled quantities of interest ``qoi``
                 (helicity, ||J||/||B||, wall, and the pressure diagnostics
                 gradp_cmp, p_cmp, weak_resid, dpdn_wall, JxBn_wall,
                 beta_vol, beta_axis); the initial-condition summary ``ic`` and the
                 ``summary`` with the stopping reason, both carrying the
                 same pressure diagnostics. Rewritten at every qoi sample.
    B.h5         B_ic, B_final, the strong pressures p_ic, p_final (3-form
                 DoFs) and the weak pressures pw_ic, pw_final (Dirichlet
                 0-form DoFs), all evaluated at the field stored next to
                 them, with the run parameters as attributes; ``geometry``
                 as given and ``geometry_path`` resolved. Written when the
                 loop ends.

The trace records the linesearch identity ``dE_pred = -dt (F,u)_M / 2``
against the measured decrease: it is an operator identity (curl adjointness,
the cross-product sign, Leray M-orthogonality) and holds to round-off when
eta = 0 and --dt-mode linesearch. With eta > 0 the implicit resistive solve
removes energy on top of the ideal step, so ``dE_meas <= dE_pred``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def force_floor_reached(resid, steps, tol):
    """Return True when the mean of the last ``steps`` residuals is below ``tol``.

    ``resid`` is the relative force residual after each step so far. Needs at
    least ``steps`` samples. Pure numpy, so it can be replayed on a trace
    without a GPU.
    """
    if len(resid) < steps:
        return False
    return bool(np.mean(resid[-steps:]) < tol)


def eta_schedule(kind, eta_max, it, steps, pulse=(2000, 100, 0)):
    """Resistivity at step ``it`` of ``steps``. ``pulse`` = (start, width,
    period): ``eta_max`` on ``start <= it < start + width`` and, with a period,
    on every later window ``start + k period``; 0 elsewhere."""
    if eta_max == 0.0:
        return 0.0
    if kind == "pulse":
        start, width, period = pulse
        since = it - start
        if since < 0:
            return 0.0
        if period > 0:
            since %= period
        return eta_max if since < width else 0.0
    frac = it / max(steps, 1)
    if kind == "tanh":
        return eta_max * 0.5 * (1.0 - np.tanh(4.0 * np.pi * (frac - 0.5)))
    if kind == "linear":
        return eta_max * (1.0 - frac)
    return eta_max


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", required=True,
                    help="toroid, cylinder, rot-ellipse, or the path of a GVEC export")
    ap.add_argument("--nfp", type=int, default=None,
                    help="field periods; overrides the file's nfp attribute")
    ap.add_argument("--ns", default="8,16,16")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    ap.add_argument("--ic", default="clebsch", choices=("clebsch", "analytic", "dzeta"))
    analytic = ap.add_argument_group(
        "analytic IC only",
        "The synthetic initial condition B = (0, Phi'(iota - d_zeta lambda), "
        "Phi'(1 + d_chi lambda)) on the logical grid, with the prescribed "
        "rotational transform iota(rho) = iota0 + (iota1 - iota0) rho^iota_exp "
        "and flux profile Phi'(rho) ~ rho^flux_exp. Ignored for --ic clebsch.")
    analytic.add_argument("--iota", default="0.4,0.9", help="iota0,iota1: on axis and at the edge")
    analytic.add_argument("--iota-exp", type=float, default=2.0, help="exponent of the iota profile")
    analytic.add_argument("--flux-exp", type=float, default=1.0, help="exponent of the flux profile")
    analytic.add_argument("--lam", default="", help='lambda modes "m,n,amp;..."')
    ap.add_argument("--method", default="lbfgs", choices=("gradient", "lbfgs"))
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--velocity-smoothing-order", type=int, default=0,
                    help="descent direction v = (I - scale L)^-order F; 0 is off")
    ap.add_argument("--velocity-smoothing-scale", type=float, default=0.0,
                    help="length scale of the velocity smoothing")
    ap.add_argument("--seed", default="",
                    help='resonant seed "m,n,rho0,width" added to the potential (clebsch IC only)')
    ap.add_argument("--seed-eps", type=float, default=0.0,
                    help="its amplitude |dB^rho| / |B^zeta| at rho0 (island width ~ sqrt of it)")
    ap.add_argument("--presmooth", type=int, default=0,
                    help="resistive-only steps on the IC before the descent; 0 is off")
    ap.add_argument("--presmooth-eps", type=float, default=1e-3,
                    help="eta dt of one pre-smoothing step")
    ap.add_argument("--presmooth-jb", type=float, default=None,
                    help="stop pre-smoothing once ||J||/||B|| <= this")
    ap.add_argument("--dt-mode", default="linesearch", choices=("linesearch", "fixed"))
    ap.add_argument("--dt0", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--eta-max", type=float, default=0.0)
    ap.add_argument("--eta-schedule", default="tanh", choices=("tanh", "constant", "linear", "pulse"))
    ap.add_argument("--eta-pulse", default="2000,100",
                    help="pulse schedule: start step, width in steps and optionally the period")
    ap.add_argument("--eta-every", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--floor-tol", type=float, default=1e-3,
                    help="stop when the windowed mean of the relative force residual is below this")
    ap.add_argument("--floor-steps", type=int, default=100,
                    help="number of steps over which the force residual is averaged "
                         "before it is compared with --floor-tol")
    ap.add_argument("--save-every", type=int, default=0,
                    help="store B and the weak pressure every K steps in B.h5 "
                         "(B_snapshots, pw_snapshots, snapshot_steps); 0 is off")
    ap.add_argument("--qoi-every", type=int, default=250,
                    help="steps between the qoi samples: helicity (a k=1 Hodge solve), "
                         "the two pressures and beta (a force evaluation and a k=0 solve)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stepper", default="h", choices=("h", "bonly"),
                    help="h: production step (J x H, u x H); bonly: the experimental "
                         "B-only cross products (mrx.experimental.bonly_relaxation)")
    cli = ap.parse_args(argv)
    pulse = tuple(int(v) for v in cli.eta_pulse.split(","))
    if len(pulse) not in (2, 3):
        ap.error("--eta-pulse wants START,WIDTH or START,WIDTH,PERIOD")
    cli.eta_pulse = pulse + (0,) * (3 - len(pulse))
    if cli.ic == "clebsch" and not os.path.isfile(cli.geometry):
        ap.error(f"--ic clebsch reads the Clebsch data from a GVEC export, and "
                 f"--geometry {cli.geometry!r} is not a file; use --ic analytic")
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
    from mrx.geometry import build_sequence
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import (analytic_profile_form,
                                        clebsch_potential_form, divergence_norm,
                                        potential_two_form, lambda_dirichlet_energy,
                                        resonant_rho,
                                        dzeta_form, leray_clean, make_lambda,
                                        make_profiles, parse_lambda,
                                        project_reference_two_form)
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (DescentMethod, TimeStepChoice, TimeStepper,
                                compute_force, compute_helicity, initial_state, resistive_step,
                                pressure_diagnostics, weak_pressure)
    if cli.stepper == "bonly":  # experimental hook (2026-09-03): J x B and u x B, no H
        from mrx.experimental.bonly_relaxation import (BOnlyTimeStepper as TimeStepper,
                                                       compute_force_bonly as compute_force,
                                                       initial_state_bonly as initial_state)
    import jax.numpy as jnp

    if cli.precision != str(mrx.DTYPE):
        raise ValueError(f"--precision {cli.precision} but mrx runs in {mrx.DTYPE}")
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)
    ns = tuple(int(v) for v in cli.ns.split(","))
    is_file = os.path.isfile(cli.geometry)
    geometry_path = os.path.abspath(cli.geometry) if is_file else cli.geometry
    out = cli.out or os.path.join("outputs", "relax", time.strftime("%Y-%m-%d"),
                                  time.strftime("%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    params = {k: v for k, v in vars(cli).items()}
    params.update(ns=list(ns), out=out, geometry_path=geometry_path)
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter, tol=cli.tol, nfp=cli.nfp)
    ops = seq.set_operators(compute_nullspaces(seq, ops))
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} tol={seq.tol:.1e}  "
          f"n2_dbc={seq.n(2, True)}  operators+nullspaces "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    # --- initial condition -----------------------------------------------
    iota0, iota1 = (float(v) for v in cli.iota.split(","))
    if cli.ic == "clebsch":
        cb = load_clebsch(cli.geometry)
        lam_norm, lam_energy = lambda_dirichlet_energy(cb["lam_h"], seq)
        results["lambda"] = dict(norm_sq=lam_norm, dirichlet_energy=lam_energy)
        print(f"[ic] lambda: ||lam||^2 {lam_norm:.4e}  <lam, L0 lam> {lam_energy:.4e}  "
              f"ratio {lam_energy / lam_norm:.4e}", flush=True)
        print(f"[ic] clebsch from {geometry_path}  nfp={cb['nfp']}  iota (full turn) "
              f"{cb['dchi'][1] / cb['dPhi'][1]:+.5f} -> "
              f"{cb['dchi'][-1] / cb['dPhi'][-1]:+.5f}")
    elif cli.ic == "dzeta":
        omega_ref = dzeta_form()
    else:
        iota, dPhi = make_profiles(iota0, iota1, cli.iota_exp, cli.flux_exp)
        omega_ref = analytic_profile_form(iota, dPhi, make_lambda(parse_lambda(cli.lam)))
    t1 = time.perf_counter()
    if cli.ic == "clebsch":
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
    # --- optional resistive pre-smoothing of the IC -------------------------
    # Backward-Euler diffusion with the force switched off: a few steps take
    # the grid-scale current out of a coarsely sampled IC (||J||/||B|| is
    # the gauge) before the descent starts working on it. The IC written to
    # B.h5 and every "[ic]" number below are the smoothed field's.
    results["presmooth"] = []
    if cli.presmooth > 0:
        smooth = jax.jit(lambda B: resistive_step(B, seq, cli.presmooth_eps))
        for k in range(cli.presmooth):
            _, _, J, _, _ = compute_force(B0, seq)
            jb = float(seq.l2_norm(J, 1) / seq.l2_norm(B0, 2))
            if cli.presmooth_jb is not None and jb <= cli.presmooth_jb:
                print(f"[presmooth] ||J||/||B|| {jb:.4e} <= {cli.presmooth_jb:g}: done", flush=True)
                break
            B0, info, rel = smooth(B0)
            rec = dict(step=k + 1, JoverB_before=jb, moved=float(rel), it=int(info),
                       E=0.5 * float(seq.l2_norm_sq(B0, 2)), div=divergence_norm(seq, B0))
            results["presmooth"].append(rec)
            print(f"[presmooth] step {k + 1}: ||J||/||B|| {jb:.4e} before, eps {cli.presmooth_eps:g}, "
                  f"MINRES {rec['it']} it, moved {rec['moved']:.3e}, E={rec['E']:.6e}, "
                  f"||div B||={rec['div']:.2e}", flush=True)
    H0, _ = compute_helicity(B0, seq, jnp.zeros(seq.n(1, True)))
    E0 = 0.5 * float(seq.l2_norm_sq(B0, 2))
    normaliser = jax.jit(make_force_normaliser(seq))
    gradp0 = float(normaliser(B0))

    @jax.jit
    def force_probe(B, p_guess, H_guess, JxH_guess):
        return compute_force(B, seq, p_guess=p_guess, H_guess=H_guess, JxH_guess=JxH_guess)

    def pressure_probe_eager(B, p, J, H, pw_guess):
        """The weak pressure of ``compute_force``'s ``(p, J, H)`` at ``B``,
        ||J||/||B||, and the strong/weak comparison (see "Two pressures")."""
        p_w, F_w, v = weak_pressure(J, H, seq, p_guess=pw_guess)
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

    F0, p0, J0, Hf0, JxH0 = compute_force(B0, seq)
    pw0, JoverB0, diag0 = pressure_probe_eager(B0, p0, J0, Hf0, jnp.zeros(seq.n(0, True)))
    diag0 = {k: float(v) for k, v in diag0.items()}
    F0n = float(seq.l2_norm(F0, 2))
    resid0 = F0n / gradp0
    print(f"[ic] {cli.ic} IC in {time.perf_counter() - t1:.1f}s  ||B||_M raw "
          f"{B_norm:.6e}  E={E0:.6e}  ||F||={F0n:.4e}  resid "
          f"{resid0:.4e}  H={float(H0):+.6e}",
          flush=True)
    print(f"[ic] {pressure_line(diag0)}", flush=True)
    results["ic"] = dict(B_norm_raw=B_norm, div_raw=div_raw, div=div0,
                         leray_moved=moved, E=E0, F=F0n, gradp=gradp0,
                         resid=resid0, H=float(H0),
                         JoverB=float(JoverB0), **diag0)

    # --- the descent -------------------------------------------------------
    method = {"gradient": DescentMethod.GRADIENT,
              "lbfgs": DescentMethod.LBFGS}[cli.method]
    ts = TimeStepper(
        seq=seq, descent_method=method,
        dt_mode=(TimeStepChoice.ANALYTIC_LINESEARCH if cli.dt_mode == "linesearch"
                 else TimeStepChoice.FIXED),
        cfl=cli.cfl, eta_every=cli.eta_every, resistive=cli.eta_max > 0,
        history_size=cli.history,
        velocity_smoothing_order=cli.velocity_smoothing_order,
        velocity_smoothing_scale=cli.velocity_smoothing_scale)
    apply_M2 = jax.jit(lambda v: seq.apply_mass_matrix(v, 2))
    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])

    @jax.jit
    def step(state):
        state = ts.relaxation_step(state)
        return eqx.tree_at(lambda s: s.B_n, state, state.B_nplus1)

    @jax.jit
    def probe(state):
        """Trace quantities from the post-step state. F_prev and v are the F
        and u the step used, so the linesearch identity is reconstructible."""
        Fu = state.F_prev @ apply_M2(state.v)
        return (0.5 * seq.l2_norm_sq(state.B_n, 2),
                seq.l2_norm(seq.apply_incidence_matrix(
                    state.B_n, 2, dirichlet_in=True, dirichlet_out=True), 3),
                Fu, Fu / (state.F_norm * state.v_norm),
                (Fu / state.dt) ** 0.5 / state.v_norm,
                state.F_norm / normaliser(state.B_n))

    state = initial_state(B0, ts, dt=cli.dt0)
    p_ic = np.asarray(state.p)
    pw_ic = np.asarray(pw0)
    snaps = [(0, np.asarray(B0), np.asarray(pw0))]
    pw_guess = pw0
    latest = {"p": p_ic, "pw": pw_ic, "diag": diag0}

    tr = {k: [] for k in ("E", "F", "resid", "dt", "dt_star", "cfl", "div", "cos",
                          "gain", "eta", "res_it", "res_delta", "dE_meas", "dE_pred")}
    qoi = {k: [] for k in ("it", "helicity", "JoverB", "wall", "gradp_cmp", "p_cmp",
                           "weak_resid", "dpdn_wall", "JxBn_wall", "beta_vol", "beta_axis")}
    E_prev = E0
    stop = "steps"
    t_arm = time.perf_counter()
    t_qoi = 0.0     # time inside the qoi samples; the recorded wall excludes it
    n_done = 0
    print(f"\n=== {cli.method}  m={cli.history} "
          f"smoothing={cli.velocity_smoothing_order}@{cli.velocity_smoothing_scale} "
          f"dt-mode={cli.dt_mode} cfl={cli.cfl} eta-max={cli.eta_max} eta-every={cli.eta_every}  "
          f"steps<={cli.steps} floor-tol={cli.floor_tol:.1e} over {cli.floor_steps} steps ===",
          flush=True)

    def save(final=False):
        results["trace"] = tr
        results["qoi"] = qoi
        window = tr["resid"][-cli.floor_steps:]
        results["summary"] = dict(
            steps=n_done, stop=stop, wall=time.perf_counter() - t_arm - t_qoi,
            E_final=tr["E"][-1] if tr["E"] else E0,
            F_final=tr["F"][-1] if tr["F"] else F0n,
            resid_final=tr["resid"][-1] if tr["resid"] else resid0,
            resid_window_mean=float(np.mean(window)) if window else resid0,
            **latest["diag"])
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)
        # The field goes out with every save, not only the last one: a run
        # that hits its wall-time limit leaves the state it reached.
        with h5py.File(os.path.join(out, "B.h5"), "w") as fh:
            fh.create_dataset("B_ic", data=np.asarray(B0))
            fh.create_dataset("B_final", data=np.asarray(state.B_n))
            fh.create_dataset("p_ic", data=p_ic)
            fh.create_dataset("p_final", data=latest["p"])
            fh.create_dataset("pw_ic", data=pw_ic)
            fh.create_dataset("pw_final", data=latest["pw"])
            if cli.save_every:
                fh.create_dataset("snapshot_steps", data=np.array([k for k, _, _ in snaps]))
                fh.create_dataset("B_snapshots", data=np.stack([b for _, b, _ in snaps]))
                fh.create_dataset("pw_snapshots", data=np.stack([w for _, _, w in snaps]))
            for k, v in params.items():
                fh.attrs[k] = "" if v is None else v

    for it in range(1, cli.steps + 1):
        if cli.eta_max > 0.0:
            eta_now = eta_schedule(cli.eta_schedule, cli.eta_max, it, cli.steps, cli.eta_pulse)
            state = eqx.tree_at(lambda t: t.eta, state, eta_now)
            if eta_now == 0.0:
                # The stepper accumulates the resistive clock until a solve is
                # due; with eta off that clock must not carry into the next
                # window (a pulse after 2000 ideal steps would otherwise
                # diffuse over all of them at once).
                state = eqx.tree_at(lambda t: (t.resistive_time, t.resistive_count), state,
                                    (jnp.zeros(()), jnp.int32(0)))
        state = step(state)
        E, div, Fu, cos, gain, resid = (float(v) for v in probe(state))
        tr["E"].append(E)
        tr["F"].append(float(state.F_norm))
        tr["resid"].append(resid)
        tr["dt"].append(float(state.dt))
        tr["dt_star"].append(float(state.dt_star))
        tr["cfl"].append(float(state.cfl_max))
        tr["div"].append(div)
        tr["cos"].append(cos)
        tr["gain"].append(gain)
        tr["eta"].append(float(state.eta))
        tr["res_it"].append(int(state.resistive_info))
        tr["res_delta"].append(float(state.resistive_delta))
        tr["dE_meas"].append(E - E_prev)
        tr["dE_pred"].append(-0.5 * float(state.dt) * Fu)
        E_prev = E
        n_done = it
        if cli.save_every and (it % cli.save_every == 0 or it == cli.steps):
            # A frame for a movie: the field and its weak pressure now.
            _, p_s, J_s, H_s, _ = force_probe(state.B_n, state.p, state.H, state.JxH)
            pw_s, _, _ = pressure_probe(state.B_n, p_s, J_s, H_s, pw_guess)
            snaps.append((it, np.asarray(state.B_n), np.asarray(pw_s)))
        if it % cli.qoi_every == 0 or it == 1:
            tq = time.perf_counter()
            qoi["it"].append(it)
            qoi["wall"].append(tq - t_arm - t_qoi)
            # The force at the CURRENT field (state.p, H, JxH are the step's
            # values at the previous one; they warm-start it and are
            # refreshed from it). J serves ||J||/||B|| and the weak pressure.
            _, p, J, H, JxH = force_probe(state.B_n, state.p, state.H, state.JxH)
            pw_guess, JoverB, diag = pressure_probe(state.B_n, p, J, H, pw_guess)
            state = eqx.tree_at(lambda s: (s.p, s.H, s.JxH), state, (p, H, JxH))
            diag = {k: float(v) for k, v in diag.items()}
            latest = {"p": np.asarray(p), "pw": np.asarray(pw_guess), "diag": diag}
            qoi["JoverB"].append(float(JoverB))
            for k, v in diag.items():
                qoi[k].append(v)
            h, A_new = get_helicity(state.B_n, seq, state.A)
            state = eqx.tree_at(lambda s: s.A, state, A_new)
            qoi["helicity"].append(float(h))
            h0 = qoi["helicity"][0]
            print(f"  it {it:>5d}  E={E:.8e}  |F|={float(state.F_norm):.4e}  "
                  f"resid={resid:.3e}  H={float(h):+.6e}  "
                  f"dH={float(h) - h0:+.3e}  [{tq - t_arm - t_qoi:.0f}s solve "
                  f"+{t_qoi:.0f}s qoi]\n           {pressure_line(diag)}", flush=True)
            save()
            t_qoi += time.perf_counter() - tq
        elif it <= 5 or it % 20 == 0:
            print(f"  it {it:>5d}  E={E:.8e}  |F|={float(state.F_norm):.4e}  "
                  f"resid={resid:.3e}  "
                  f"dt={float(state.dt):+.3e}  dt*={float(state.dt_star):+.3e}  "
                  f"cfl={float(state.cfl_max) * float(state.dt):.2f}  cos={cos:+.4f}  gain={gain:.2e}  "
                  f"divB={div:.2e}  dE_meas={tr['dE_meas'][-1]:+.3e}  "
                  f"dE_pred={tr['dE_pred'][-1]:+.3e}  res_it={tr['res_it'][-1]}  "
                  f"res_delta={tr['res_delta'][-1]:.2e}",
                  flush=True)
        if force_floor_reached(tr["resid"], cli.floor_steps, cli.floor_tol):
            stop = "floor"
            print(f"  [floor] force residual averaged over the last {cli.floor_steps} "
                  f"steps below {cli.floor_tol:.1e} at it={it}", flush=True)
            break
        if cli.seconds is not None and time.perf_counter() - t_arm > cli.seconds:
            stop = "seconds"
            print(f"  [budget] {cli.seconds:.0f} s spent at it={it}", flush=True)
            break

    wall = time.perf_counter() - t_arm - t_qoi
    if qoi["it"][-1] != n_done:
        # The pressures stored next to B_final are evaluated AT B_final.
        _, p, J, H, JxH = force_probe(state.B_n, state.p, state.H, state.JxH)
        p_w, _, diag = pressure_probe(state.B_n, p, J, H, pw_guess)
        latest = {"p": np.asarray(p), "pw": np.asarray(p_w),
                  "diag": {k: float(v) for k, v in diag.items()}}
    dEm, dEp = np.array(tr["dE_meas"]), np.array(tr["dE_pred"])
    ident = np.abs(dEm - dEp) / E0
    resid_tr = np.array(tr["resid"])
    print(f"\n--- {n_done} steps in {wall:.1f}s ({wall / max(n_done, 1):.2f} s/step), "
          f"stopped on: {stop}")
    print(f"    E {E0:.8e} -> {tr['E'][-1]:.8e}  "
          f"({(E0 - tr['E'][-1]) / E0:.4%} of the initial energy removed)")
    print(f"    |F| {F0n:.4e} -> {tr['F'][-1]:.4e}   residual "
          f"{resid0:.4e} -> {resid_tr[-1]:.4e}  (mean over the last "
          f"{min(cli.floor_steps, n_done)} steps {resid_tr[-cli.floor_steps:].mean():.4e}, "
          f"min {resid_tr.min():.4e})")
    print(f"    linesearch identity |dE_meas - dE_pred| / E0: median "
          f"{np.median(ident):.3e}  max {ident.max():.3e}"
          + ("  (not an identity with eta > 0 or a fixed dt)"
             if cli.eta_max > 0 or cli.dt_mode == "fixed" else ""))
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
    res_it = np.array(tr["res_it"])
    if cli.eta_max > 0:
        solved = res_it != 0
        rd = np.array(tr["res_delta"])[solved]
        print(f"    resistive solve on {int(solved.sum())}/{n_done} steps: MINRES iterations "
              f"mean {np.abs(res_it[solved]).mean():.1f}  max {np.abs(res_it).max()}  "
              f"unconverged on {int((res_it > 0).sum())};  ||delta||/||B|| "
              f"mean {rd.mean():.2e}  max {rd.max():.2e}", flush=True)
    save(final=True)
    print(f"wrote {out}/relax.json and {out}/B.h5", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"] = cli.precision
    main(cli)
