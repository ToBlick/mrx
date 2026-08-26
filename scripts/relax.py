"""Relax a magnetic field toward minimum energy at fixed helicity.

Builds a named geometry, projects an initial field, then descends the magnetic
energy with the incompressible, helicity-conserving flow of
:class:`mrx.relaxation.TimeStepper` until the energy floors, the step budget
is spent, or the wall-clock budget runs out. The fixed point is
``J x B = grad p`` with ``p`` the Leray multiplier, so the relaxed state is a
finite-beta equilibrium, not a force-free field.

Stopping criterion
    The energy decrease over the last ``--floor-window`` steps, relative to
    the energy and per step, ``(E[i-W] - E[i]) / (W |E[i]|)``, falls below
    ``--floor-tol``. The relaxation guarantees ``dE/dt <= 0`` only, so the
    force residual need not be monotone and is not used to stop. Defaults:
    ``W = 100`` and ``10 * eps`` of the working precision (2.2e-15 in
    float64, 1.2e-6 in float32). Calibration on the archived eta=1e-2 arm S10 (energy constant
    to 16 digits from step ~1700): fires at step 1945; the archived arms
    that never floored (S07 at 13018 steps, C1 at 3000) never trigger it.
    ``test/test_relax_floor.py`` replays those traces.

Flags (defaults in brackets)
    Sequence:
      --geometry NAME [quasr44970]   toroid, cylinder, rot-ellipse, w7x or a
                                     name in mrx.gvec.GVEC_GEOMETRIES
      --ns R,T,Z [8,16,8]            spline resolution (also the map's)
      --p P [3]                      spline degree; p+1 Gauss points per span
      --maxiter N [10000]            iteration budget of every inner solve
      --tol TOL [sqrt(eps)]          inner solve tolerance; the 2026-08
                                     campaign used 1e-12
      --precision {float64,float32} [float64]  exported as MRX_DTYPE before
                                     mrx is imported
    Initial condition:
      --ic {logical,clebsch,dzeta} [logical]
          logical: prescribed power-law profiles (no external data)
          clebsch: GVEC's own dPhi_dr, dchi_dr, lambda from the geometry file
          dzeta:   the constant 2-form (0,0,1); relaxes to the harmonic field
      --iota I0,I1 [0.4,0.9]         logical: iota on axis and at the edge
      --iota-exp E [2.0]             logical: iota = I0 + (I1-I0) rho^E
      --flux-exp Q [1.0]             logical: dPhi/drho = rho^Q
      --lam SPEC [""]                logical: lambda modes "m,n,amp;..."
      --no-lambda                    clebsch: zero lambda (fluxes, iota and
                                     helicity must not move; the force must)
      --no-leray-ic                  skip the Leray clean-up of the IC
    Descent:
      --method {gradient,cg,lbfgs} [cg]
      --history M [1]                CG / L-BFGS history length
      --gamma G [0], --mu MU [0.0]   hyperregularisation v = (I - mu L)^-G F
      --dt-mode {linesearch,fixed} [linesearch]
          linesearch takes the exact energy-minimising step, the largest step
          that still lowers E; explicit Euler keeps the frozen-in flux only to
          O(dt^2), so a large step can destroy field-line topology with
          div B and energy monotonicity intact. fixed with a small --dt0 is
          the control for that.
      --dt0 DT [1.0]                 the step for --dt-mode fixed
      --eta-max ETA [0.0]            peak resistivity; eta > 0 lets the field
                                     reconnect, helicity is then not conserved.
                                     The resistive part is backward Euler,
                                     (M2 + dt eta L2) B = M2 B_ideal after the
                                     ideal step, so no eta is too large for
                                     the linesearch dt; the cost is one k=2
                                     MINRES solve per step, its iteration
                                     count is traced as res_it
      --eta-schedule {tanh,constant,linear} [tanh]
                                     tanh drops eta to ~0 over the middle
                                     third of --steps so the run ends ideal
    Budgets and output:
      --steps N [3000]               maximum number of steps
      --seconds S [none]             wall-clock budget of the descent loop
      --floor-tol TOL [10*eps]       see "Stopping criterion"
      --floor-window W [100]
      --diag-every N [250]           steps between helicity / residual samples
                                     (each is a k=1 Hodge solve)
      --out DIR [outputs/relax/<date>/<time>]

Canonical invocations (one GPU, 64 GB; see slurm/README.md)::

    SCRIPT=scripts/relax.py JOB_NAME=relax_w1 TIMEOUT_MIN=90 \\
      ARGS="--geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p 3 --steps 3000" \\
      bash slurm/run.sh

    SCRIPT=scripts/relax.py JOB_NAME=relax_smoke TIMEOUT_MIN=30 \\
      ARGS="--geometry toroid --ns 6,12,6 --p 2 --steps 50 --diag-every 25" \\
      bash slurm/run.sh

Runtime at the reference resolution (w7x-fmm002, ns 8,16,8, p 3, one H100):
geometry, operators and nullspaces ~330 s, first step ~90 s of compilation,
then ~0.9 s per step (~1.4x with --gamma 1); ~8.6 GB host memory. The archived
3000-step arms took 45-60 min.

Output (``--out``)::

    relax.json   parameters, the per-step trace (E, F, dt, cos, gain, div,
                 eta, res_it, dE_meas, dE_pred), the sampled diagnostics (helicity,
                 residual ||F||/||grad(B^2/2)||, ||J||/||B||, wall), the
                 initial-condition summary and the stopping reason;
                 rewritten at every diagnostic sample
    B.h5         B_ic and B_final DoFs with the run parameters as attributes;
                 written when the loop ends

The trace records the linesearch identity ``dE_pred = -dt (F,u)_M / 2``
against the measured decrease: it is an operator identity (curl adjointness,
the cross-product sign, Leray M-orthogonality) and must hold to round-off
when eta = 0 and --dt-mode linesearch. With eta > 0 the implicit resistive
solve removes energy on top of the ideal step, so ``dE_meas <= dE_pred``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def energy_floor_reached(E, window, tol):
    """Return True when the windowed relative energy decrease per step is below ``tol``.

    ``E`` is the energy after each step so far. Needs at least ``window + 1``
    samples. Pure numpy, so it can be replayed on an archived trace without
    a GPU.
    """
    if len(E) <= window:
        return False
    rate = (E[-1 - window] - E[-1]) / (window * abs(E[-1]))
    return bool(rate < tol)


def eta_schedule(kind, eta_max, it, steps):
    """Resistivity at step ``it`` of ``steps``."""
    if eta_max == 0.0:
        return 0.0
    frac = it / max(steps, 1)
    if kind == "tanh":
        return eta_max * 0.5 * (1.0 - np.tanh(4.0 * np.pi * (frac - 0.5)))
    if kind == "linear":
        return eta_max * (1.0 - frac)
    return eta_max


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="quasr44970")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    ap.add_argument("--ic", default="logical", choices=("logical", "clebsch", "dzeta"))
    ap.add_argument("--iota", default="0.4,0.9")
    ap.add_argument("--iota-exp", type=float, default=2.0)
    ap.add_argument("--flux-exp", type=float, default=1.0)
    ap.add_argument("--lam", default="")
    ap.add_argument("--no-lambda", action="store_true")
    ap.add_argument("--no-leray-ic", action="store_true")
    ap.add_argument("--method", default="cg", choices=("gradient", "cg", "lbfgs"))
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--gamma", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--dt-mode", default="linesearch", choices=("linesearch", "fixed"))
    ap.add_argument("--dt0", type=float, default=1.0)
    ap.add_argument("--eta-max", type=float, default=0.0)
    ap.add_argument("--eta-schedule", default="tanh", choices=("tanh", "constant", "linear"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--floor-tol", type=float, default=None)
    ap.add_argument("--floor-window", type=int, default=100)
    ap.add_argument("--diag-every", type=int, default=250)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def make_force_normaliser(seq):
    """``||grad(B^2/2)||_L2``: the scale the force residual is measured against.

    grad p is a real scale too (the scheme converges to J x B = grad p) but
    vanishes in the low-beta limit; grad(B^2/2) has the same units and stays
    O(1). Computed through the sequence: project B^2/2 onto 0-forms (one load
    and one M_0 solve), take the discrete gradient, measure it.
    """
    import jax.numpy as jnp
    from mrx.quadrature import evaluate_at_xq, integrate_against

    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    ci0, cs0 = seq._form_comp_info(0)
    ci2, cs2 = seq._form_comp_info(2)

    def normaliser(B_dof):
        B_jk = evaluate_at_xq(seq.e2_dbc_T @ B_dof, ci2, cs2, quad_shape, 3)
        bsq = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk)
        f_jk = (0.5 * bsq * seq.quad.w / seq.jacobian_j)[:, None]
        q = seq.e0 @ integrate_against(f_jk, ci0, cs0, quad_shape)
        w0 = seq.apply_inverse_mass_matrix(q, 0, dirichlet=False)
        g1 = seq.apply_strong_grad(w0, dirichlet_in=False, dirichlet_out=False)
        return seq.l2_norm(g1, 1, dirichlet=False)

    return normaliser


def main(cli):
    import equinox as eqx
    import h5py
    import jax

    import mrx
    import mrx.operators as op
    from mrx.geometries import build_sequence
    from mrx.gvec import gvec_path, load_clebsch
    from mrx.initial_conditions import (analytic_helicity, clebsch_form,
                                        divergence_norm, dzeta_form,
                                        leray_clean, logical_profile_form,
                                        make_lambda, make_profiles,
                                        parse_lambda,
                                        project_reference_two_form)
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (DescentMethod, IntegrationScheme,
                                TimeStepChoice, TimeStepper, compute_force,
                                compute_helicity, initial_state)
    import jax.numpy as jnp

    if cli.precision != str(mrx.DTYPE):
        raise ValueError(f"--precision {cli.precision} but mrx runs in {mrx.DTYPE}")
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)
    ns = tuple(int(v) for v in cli.ns.split(","))
    floor_tol = 10 * mrx.EPS if cli.floor_tol is None else cli.floor_tol
    out = cli.out or os.path.join("outputs", "relax", time.strftime("%Y-%m-%d"),
                                  time.strftime("%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    params = {k: v for k, v in vars(cli).items()}
    params.update(ns=list(ns), floor_tol=floor_tol, out=out)
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter, tol=cli.tol)
    ops = seq.assemble_all_sparse()
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    ops = op.assemble_metric_lumping_laplacian_preconditioner(
        seq, ops, ks=(0, 1, 2, 3), dirichlets=(False, True))
    seq.set_operators(ops)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} tol={seq.tol:.1e}  "
          f"n2_dbc={seq.n2_dbc}  operators+nullspaces "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    # --- initial condition -----------------------------------------------
    iota0, iota1 = (float(v) for v in cli.iota.split(","))
    H_analytic = float("nan")
    if cli.ic == "clebsch":
        cb = load_clebsch(gvec_path(cli.geometry), seq.basis_0.types)
        omega_ref = clebsch_form(cb, use_lambda=not cli.no_lambda)
        print(f"[ic] clebsch from {gvec_path(cli.geometry)}  nfp={cb['nfp']}  "
              f"lambda={'OFF' if cli.no_lambda else 'on'}  closed axes "
              f"{cb['closed_axes']}  iota (full turn) "
              f"{cb['dchi'][1] / cb['dPhi'][1]:+.5f} -> "
              f"{cb['dchi'][-1] / cb['dPhi'][-1]:+.5f}  flux-function departure "
              f"{cb['iota_spread']:.2e}")
    elif cli.ic == "dzeta":
        omega_ref = dzeta_form()
    else:
        iota, dPhi = make_profiles(iota0, iota1, cli.iota_exp, cli.flux_exp)
        omega_ref = logical_profile_form(iota, dPhi, make_lambda(parse_lambda(cli.lam)))
    t1 = time.perf_counter()
    B0, B_norm = project_reference_two_form(seq, omega_ref)
    if cli.ic == "logical":
        H_analytic = analytic_helicity(iota0, iota1, cli.iota_exp, cli.flux_exp) / B_norm ** 2
    div0 = divergence_norm(seq, B0)
    moved = 0.0
    if not cli.no_leray_ic:
        B0, moved = leray_clean(seq, B0)
        div_after = divergence_norm(seq, B0)
        print(f"[ic] Leray-projected: ||div B|| {div0:.3e} -> {div_after:.3e}  "
              f"(moved the field by {moved:.3e})")
        div0 = div_after
    H0, _ = compute_helicity(B0, seq, jnp.zeros(seq.n1_dbc))
    E0 = 0.5 * float(seq.l2_norm_sq(B0, 2))
    normaliser = jax.jit(make_force_normaliser(seq))
    gradp0 = float(normaliser(B0))
    F0, p0, _, _, _ = compute_force(B0, seq)
    F0n = float(seq.l2_norm(F0, 2))
    print(f"[ic] {cli.ic} IC in {time.perf_counter() - t1:.1f}s  ||B||_M raw "
          f"{B_norm:.6e}  E={E0:.6e}  ||F||={F0n:.4e}  residual "
          f"{F0n / gradp0:.4e}  H={float(H0):+.6e}"
          + (f"  H eq.(1) {H_analytic:+.6e}" if cli.ic == "logical" else ""),
          flush=True)
    results["ic"] = dict(B_norm_raw=B_norm, div=div0, leray_moved=moved,
                         E=E0, F=F0n, gradp=gradp0, H=float(H0),
                         H_analytic=H_analytic)

    # --- the descent -------------------------------------------------------
    method = {"gradient": DescentMethod.GRADIENT,
              "cg": DescentMethod.CONJUGATE_GRADIENT,
              "lbfgs": DescentMethod.LBFGS}[cli.method]
    ts = TimeStepper(
        seq=seq, descent_method=method,
        dt_mode=(TimeStepChoice.ANALYTIC_LINESEARCH if cli.dt_mode == "linesearch"
                 else TimeStepChoice.FIXED),
        timestep_mode=IntegrationScheme.EXPLICIT,
        history_size=cli.history, gamma=cli.gamma, mu=cli.mu)
    apply_M2 = jax.jit(lambda v: seq.apply_mass_matrix(v, 2))
    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])

    @jax.jit
    def step(state):
        state = ts.relaxation_step(state, state.key)
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
                (Fu / state.dt) ** 0.5 / state.v_norm)

    state = initial_state(B0, ts, dt=cli.dt0)

    tr = {k: [] for k in ("E", "F", "dt", "div", "cos", "gain", "eta",
                          "res_it", "dE_meas", "dE_pred")}
    diag = {k: [] for k in ("it", "helicity", "resid", "gradp", "JoverB", "wall")}
    E_prev = E0
    stop = "steps"
    t_arm = time.perf_counter()
    t_diag = 0.0     # time inside the diagnostics; the recorded wall excludes it
    n_done = 0
    print(f"\n=== {cli.method}  m={cli.history} gamma={cli.gamma} mu={cli.mu} "
          f"dt-mode={cli.dt_mode} eta-max={cli.eta_max}  steps<={cli.steps} "
          f"floor-tol={floor_tol:.1e} window={cli.floor_window} ===", flush=True)

    def save(final=False):
        results["trace"] = tr
        results["diagnostics"] = diag
        results["summary"] = dict(
            steps=n_done, stop=stop, wall=time.perf_counter() - t_arm - t_diag,
            E_final=tr["E"][-1] if tr["E"] else E0,
            F_final=tr["F"][-1] if tr["F"] else F0n)
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)
        if final:
            with h5py.File(os.path.join(out, "B.h5"), "w") as fh:
                fh.create_dataset("B_ic", data=np.asarray(B0))
                fh.create_dataset("B_final", data=np.asarray(state.B_n))
                for k, v in params.items():
                    fh.attrs[k] = "" if v is None else v

    for it in range(1, cli.steps + 1):
        if cli.eta_max > 0.0:
            state = eqx.tree_at(lambda t: t.eta, state,
                                eta_schedule(cli.eta_schedule, cli.eta_max, it, cli.steps))
        state = step(state)
        E, div, Fu, cos, gain = (float(v) for v in probe(state))
        tr["E"].append(E)
        tr["F"].append(float(state.F_norm))
        tr["dt"].append(float(state.dt))
        tr["div"].append(div)
        tr["cos"].append(cos)
        tr["gain"].append(gain)
        tr["eta"].append(float(state.eta))
        tr["res_it"].append(int(state.resistive_info))
        tr["dE_meas"].append(E - E_prev)
        tr["dE_pred"].append(-0.5 * float(state.dt) * Fu)
        E_prev = E
        n_done = it
        if it % cli.diag_every == 0 or it == 1:
            td = time.perf_counter()
            diag["it"].append(it)
            diag["wall"].append(td - t_arm - t_diag)
            gp = float(normaliser(state.B_n))
            diag["gradp"].append(gp)
            diag["resid"].append(float(state.F_norm) / gp)
            diag["JoverB"].append(float(seq.l2_norm(seq.apply_weak_curl(state.B_n), 1)
                                        / seq.l2_norm(state.B_n, 2)))
            h, A_new = get_helicity(state.B_n, seq, state.A)
            state = eqx.tree_at(lambda s: s.A, state, A_new)
            diag["helicity"].append(float(h))
            h0 = diag["helicity"][0]
            print(f"  it {it:>5d}  E={E:.8e}  |F|={float(state.F_norm):.4e}  "
                  f"resid={diag['resid'][-1]:.3e}  H={float(h):+.6e}  "
                  f"dH={float(h) - h0:+.3e}  [{td - t_arm - t_diag:.0f}s solve "
                  f"+{t_diag:.0f}s diag]", flush=True)
            save()
            t_diag += time.perf_counter() - td
        elif it <= 5 or it % 20 == 0:
            print(f"  it {it:>5d}  E={E:.8e}  |F|={float(state.F_norm):.4e}  "
                  f"dt={float(state.dt):+.3e}  cos={cos:+.4f}  gain={gain:.2e}  "
                  f"divB={div:.2e}  dE_meas={tr['dE_meas'][-1]:+.3e}  "
                  f"dE_pred={tr['dE_pred'][-1]:+.3e}  res_it={tr['res_it'][-1]}",
                  flush=True)
        if energy_floor_reached(tr["E"], cli.floor_window, floor_tol):
            stop = "floor"
            print(f"  [floor] energy decrease per step below {floor_tol:.1e} "
                  f"over the last {cli.floor_window} steps at it={it}", flush=True)
            break
        if cli.seconds is not None and time.perf_counter() - t_arm > cli.seconds:
            stop = "seconds"
            print(f"  [budget] {cli.seconds:.0f} s spent at it={it}", flush=True)
            break

    wall = time.perf_counter() - t_arm - t_diag
    dEm, dEp = np.array(tr["dE_meas"]), np.array(tr["dE_pred"])
    ident = np.abs(dEm - dEp) / E0
    print(f"\n--- {n_done} steps in {wall:.1f}s ({wall / max(n_done, 1):.2f} s/step), "
          f"stopped on: {stop}")
    print(f"    E {E0:.8e} -> {tr['E'][-1]:.8e}  "
          f"({(E0 - tr['E'][-1]) / E0:.4%} of the initial energy removed)")
    print(f"    |F| {F0n:.4e} -> {tr['F'][-1]:.4e}   residual "
          f"{diag['resid'][0]:.4e} -> {diag['resid'][-1]:.4e}")
    print(f"    linesearch identity |dE_meas - dE_pred| / E0: median "
          f"{np.median(ident):.3e}  max {ident.max():.3e}"
          + ("  (not an identity with eta > 0 or a fixed dt)"
             if cli.eta_max > 0 or cli.dt_mode == "fixed" else ""))
    print(f"    energy increases on {int((dEm > 0).sum())}/{n_done} steps;  "
          f"||div B|| max {max(tr['div']):.3e};  ||J||/||B|| "
          f"{diag['JoverB'][0]:.4e} -> {diag['JoverB'][-1]:.4e}")
    h = np.array(diag["helicity"])
    print(f"    helicity {h[0]:+.6e} -> {h[-1]:+.6e}  drift {h[-1] - h[0]:+.3e}"
          f"  relative {(h[-1] - h[0]) / abs(h[0]):+.3e}", flush=True)
    res_it = np.array(tr["res_it"])
    if cli.eta_max > 0:
        print(f"    resistive solve: MINRES iterations mean {np.abs(res_it).mean():.1f}  "
              f"max {np.abs(res_it).max()}  unconverged on {int((res_it > 0).sum())} steps",
              flush=True)
    save(final=True)
    print(f"wrote {out}/relax.json and {out}/B.h5", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"] = cli.precision
    main(cli)
