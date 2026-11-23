"""Relax a magnetic field toward minimum energy at fixed helicity. This is the command line of :func:`mrx.relaxation.loop.relax`.

The script builds the de Rham sequence of ``--geometry.path`` and the initial field on it, or reads the field of
an earlier run with ``--output.restart``. It can open island chains at the transform the field has (``--seed``, see
:func:`mrx.relaxation.seeding.energy_seed`). The descent runs in compiled chunks of steps until the force residual
stops falling or the step budget is spent. The fixed point satisfies ``J x B = grad p``, where ``p`` is the
pressure that keeps the flow divergence-free, so the result is a finite-beta equilibrium.

With ``--drive.resistivity C`` every step is followed by a resistive step of size ``C h_r^2`` (``h_r`` the radial
cell size) that pulls the current towards that of the ``--drive.reference`` field. The reference is first smoothed
by one heat step of ``--drive.reference-smoothing`` h_r^2 and can be given an island chain with ``--drive.chain``
and ``--drive.eps``.

With ``--pressure.target`` the volumes of the flux shells are unknowns of every Newton step, so the relaxation ends
at the target pressure, a profile (the file's or a VMEC input namelist's, scaled by ``--pressure.scale`` or to
``--pressure.beta``, :mod:`mrx.relaxation.shells`). Measured pressure is fitted by ``scripts/pressure_fit.py``, which
runs this script with its fit as the target. The shells and the target are rebuilt on the flux label of the field
before every chunk. Such a run takes all of ``--budget.steps``:
the force residual does not see the shells, so ``--budget.floor-tol`` does not end it.

With ``--current.target`` the poloidal flux slips between the flux surfaces are unknowns of every Newton step, so the
relaxation ends at the target net toroidal current inside every surface, the geometry file's field's or a VMEC input
namelist's with NCURR = 1, scaled by ``--current.scale`` (:class:`mrx.relaxation.twist.Slips`). The slips are rebuilt
on the flux label before every chunk, the one label of both targets. Such a run takes all of ``--budget.steps`` too.
``--current.start`` adds the target current to the geometry file's field along the field, ``J = sigma(s) B``
(:func:`mrx.relaxation.twist.current_start`), so the run starts at the target current with that field's surfaces.
The sum is the start field, also of ``--mesh.align``'s first move: the pole goes onto its axis (record in ``ic``).
The targets go in over ``--budget.stages`` stages from the start field's own values, one per chunk boundary (1: at
once), recorded under ``continuation`` in ``relax.json``.

With ``--iota.target`` the relaxation ends at the rotational transform of a wout or of a VMEC input namelist with
NCURR = 0. The transform is no control of the energy: it sets the target of the prescribed current in an outer loop.
Before every chunk the transform's weak residual on the flux label is measured and the current target takes a Newton
step, with its matrix from the Gauss-Newton step's response to the target (:func:`mrx.relaxation.twist.iota_newton`).
The loop starts from the current the field carries. With ``--current.start`` it starts from ``--current.target``,
the current the start adds, and owns it from then on: with ``--iota.target`` a current target is only the start's.
The target is kept in every checkpoint (attribute ``iota_target``), so a restart goes on from it. With
``--iota.target`` the seed waits for the transform: it is applied at the first chunk boundary at which every local
mean of the transform's miss, the weak residual, is below :data:`mrx.relaxation.twist.AT_TARGET` of the target's largest
value.

With ``--mesh.align`` the pole of the mesh stays on the magnetic axis (:mod:`mrx.relaxation.mesh_motion`). A run from
the vacuum field of a namelist starts on a mesh moved onto the vacuum's axis, with the vacuum solved on it. Then the
pole moves only at a chunk boundary where the field is at an equilibrium (its squared force residual below
``--mesh.bar``) and its axis is more than ``--mesh.move-tol`` first radial cells off the pole. The field is carried
onto the moved map by the relaxation's induction (:func:`mrx.relaxation.mesh_motion.transfer`). A move happens before
the shells and the slips are rebuilt and does not change the targets. With ``--iota.target`` the loop takes its step
at such an equilibrium before the move and holds its current target from a move until the field is below the bar
again. A fresh start onto a moved map is held the same way: its vacuum is the equilibrium of no target, so the loop
keeps the start's current until the field is first below the bar. The checkpoints keep that hold. The moves keep the
preconditioners, which are built again on the installed map when the chunk's mean MINRES count of the Newton solve
exceeds ``--mesh.rebuild-factor`` times that of the first chunk after their last build, or a Newton solve stopped at
its iteration cap while the field is above the bar.

    python -u scripts/relax.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution 16 32 32

``--help`` lists every flag with its default. The flags are the fields of the dataclasses in
:mod:`mrx.relaxation.config`, one group each, spelled ``--<group>.<field>`` (for example ``--newton.tol``).

Output (in ``--output.out``):
    relax.json                   The configuration (flat, plus facts of the run) under ``params``, then ``ic``,
                                 ``iota`` (the target and before every chunk the transform's weak residual, its miss, the step and the current target), ``seed``, ``drive``, ``pressure`` (the target and before every chunk
                                 and at the end the pressure's error against it and the target's beta), ``current``
                                 (the target's moments and before every chunk and at the end the current's), ``mesh``
                                 (the moves of the start, every move of the axis and every check without a move, the
                                 chunks' MINRES counts and the builds of the preconditioners), the per-step
                                 ``trace``, the per-chunk ``qoi`` and the ``summary``. The file is rewritten after every chunk.
    checkpoints/state_<step>.h5  The descent state at the start and after every chunk
                                 (:func:`mrx.relaxation.loop.write_checkpoint`). ``best.h5`` holds the field with
                                 the lowest residual, since the last move of the mesh. With ``--mesh.align`` every
                                 checkpoint also holds its map (:func:`mrx.relaxation.mesh_motion.write_map`), which a
                                 restart and ``scripts/poincare_trace.py`` install, and ``moved_<step>.h5`` holds the
                                 field right after a move of the axis, on the moved map.
    checkpoints/vacuum.h5        With ``--current.start``: the geometry file's field, without the current.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py


def main(cfg, fit=None):
    """Run the relaxation of ``cfg`` and return its output directory. ``fit(seq, B, T)`` is an outer loop's pressure
    target in place of the profile of ``--pressure.target``: called before every chunk with the field ``B`` and its
    label temperature ``T``, it returns the profile ``p*(s)`` (mu0 p in T^2) and a dict for the record
    (``scripts/pressure_fit.py``)."""
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    import mrx
    from mrx.relaxation.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.paths import data_file, runs_root
    from mrx.relaxation.loop import (BestState, initial_state, radial_cell_sq, read_checkpoint, read_field, relax,
                                     write_checkpoint)
    from mrx.relaxation import mesh_motion
    from mrx.relaxation.mesh_motion import install_checkpoint_map, write_map
    from mrx.relaxation.physics import current, resistive_step, weak_pressure
    from mrx.relaxation.shells import (ERROR_BINS, build_shells, label, load_target, pressure_error, profile_cells,
                                      target_beta)
    from mrx.relaxation.seeding import energy_seed
    from mrx.relaxation.twist import (AT_TARGET, build_slips, current_error, current_on_label, current_start,
                                      iota_newton, iota_residual, load_current_target, load_iota_target,
                                      project_iota_target, spline_centres)
    from mrx.flux_label import ANISOTROPY, LABEL_MARGIN, label_condition, radius_flux

    g, d, n, b, dr = cfg.geometry, cfg.descent, cfg.newton, cfg.budget, cfg.drive
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE.name}  batch {g.max_batch or 'all'}", flush=True)
    out = cfg.output.out or os.path.join(runs_root(), "relax", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # The record holds the flat configuration plus the facts of the run.
    params = dict(cfg.params, out=out, geometry_path=os.path.abspath(data_file(g.path)))
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq = g.build()
    params.update(ic=seq.equilibrium["kind"], resolution=list(seq.ns), knots=g.knots, nfp=seq.nfp)
    compute_nullspaces(seq)
    print(f"[setup] {g.path} resolution {seq.ns} degree {g.spline_degree} tol={seq.tol:.1e}  "
          f"n2_dbc={seq.odd.n(2)}  operators+nullspaces {time.perf_counter() - t0:.1f}s", flush=True)
    # the checkpoints hold the map once it can have moved
    carry_map = bool(cfg.mesh) or g.zeta != "phi"
    if g.zeta == "harmonic":
        params["harmonic"] = seq.equilibrium["harmonic"]
    if g.zeta == "axis-normal":
        params["axis_normal"] = seq.equilibrium["axis_normal"]
    if cfg.output.restart:
        if install_checkpoint_map(seq, cfg.output.restart):
            carry_map = True
            print(f"[restart] the map of {cfg.output.restart} installed", flush=True)
        elif g.zeta != "phi":
            raise ValueError(f"{cfg.output.restart} holds no map, so it was written on the file's sections: restart "
                             "it with --geometry.zeta phi")

    pr, cu = cfg.pressure, cfg.current
    # the profile space of the targets and the loop: --pressure.cells, or half the mesh's radial cells
    cells = profile_cells(seq) if pr.cells is None else pr.cells
    params["profile_cells"] = cells

    def label_check(sq, B, ic, key="label_condition"):
        """``ic`` with the numbers of the flux label's condition on the field ``B`` under ``key``
        (:func:`mrx.flux_label.label_condition`) when the run needs the label, which stops a run below it."""
        if not (pr or cu or cfg.iota or cfg.seed):
            return ic
        iota_m, a_m, R_m = label_condition(sq, sq.odd.evaluate_at_quadrature(B, 2))
        margin = (iota_m * a_m / R_m) ** 2 / pr.anisotropy
        ic[key] = dict(iota=iota_m, a=a_m, R=R_m, margin=margin)
        what = f"(iota a / R)^2 = ({iota_m:.3e} x {a_m:.3g} / {R_m:.3g})^2 = {margin:.3g} kperp (kperp {pr.anisotropy:g})"
        print(f"[label] {key}: {what}", flush=True)
        if margin < LABEL_MARGIN:
            msg = (f"the start field's transform is too small for the flux label: {what}, below {LABEL_MARGIN:g} "
                   "kperp. Start from a VMEC/VMEX wout, or add the target current (--current.start)")
            if pr.label_check:
                raise ValueError(msg)
            print(f"[label] WARNING {msg}", flush=True)
        return ic

    def start_field(sq):
        """``(B, ic, B_file)``: the start field on the installed map of ``sq``, the geometry file's field ``B_file``
        with the target current along it with --current.start (the record of that in ``ic["start"]``)."""
        B_file, ic = initial_field(sq)
        if not cu.start:
            return B_file, label_check(sq, B_file, ic), B_file
        t1 = time.perf_counter()
        mu = load_current_target(sq, cu.target, B_file, cu.scale, anisotropy=pr.anisotropy, label_tol=pr.label_tol,
                                 cells=cells)
        if cu.start == "radius":
            s_q, info = radius_flux(sq, sq.odd.evaluate_at_quadrature(B_file, 2)), 0
        else:
            # the label of the file's field: it needs the condition too
            label_check(sq, B_file, ic, "label_condition_file")
            s_q, _, info = label(sq, B_file, pr.anisotropy, tol=pr.label_tol)
        B_J, hodge_it = current_start(sq, B_file, s_q, mu)
        rel = float(sq.odd.l2_norm(B_J, 2) / sq.odd.l2_norm(B_file, 2))
        ic["start"] = dict(label=str(cu.start), label_it=int(info), hodge_it=int(hodge_it), B_J=rel,
                           seconds=time.perf_counter() - t1)
        print(f"[current] start: J = sigma(s) B on the {cu.start} label (CG {int(info)}), Hodge solve "
              f"{int(hodge_it)}, ||B_J|| / ||B|| {rel:.4e}, {time.perf_counter() - t1:.1f}s", flush=True)
        return B_file + B_J, label_check(sq, B_file + B_J, ic), B_file

    # --- the start field: the initial condition or a restart, then the seed ---
    h_r_sq = radial_cell_sq(seq)
    params["h_r_sq"] = float(h_r_sq)
    ts = cfg.stepper(seq)
    if cfg.mesh:
        results["mesh"] = dict(moves=[], checks=[], chunks=[], builds=[])
    # with --mesh.align: whether the iota loop holds its target since the last move or the start, and the field's
    # residual at this chunk boundary
    schedule = dict(hold=False, resid=None)
    iota_target = None              # the iota loop's current target, set at its first chunk
    if cfg.output.restart:
        state, it0 = read_checkpoint(cfg.output.restart, ts)
        print(f"[restart] {cfg.output.restart}: descent state at step {it0}", flush=True)
        with h5py.File(cfg.output.restart, "r") as fh:
            if cfg.mesh and "mesh_hold" in fh.attrs:
                schedule.update(hold=bool(fh.attrs["mesh_hold"]))
                print(f"[restart] mesh schedule: {schedule}", flush=True)
            if cfg.iota and "iota_target" in fh.attrs:
                iota_target = np.asarray(fh.attrs["iota_target"], dtype=np.float64)
                print(f"[restart] the iota loop's current target, near the wall {iota_target[-1]:+.4e} T m", flush=True)
    else:
        t1 = time.perf_counter()
        if cfg.mesh and seq.equilibrium["kind"] == "indata":
            # the first move needs no transport: the pole goes onto the vacuum's axis, and the vacuum is solved there
            recut = g.zeta == "axis-normal"
            start = mesh_motion.vacuum_start(seq, lambda sq: start_field(sq)[0], recut)
            results["mesh"]["start"] = start
            # the vacuum on the moved map is no equilibrium of the targets: the pressure's rise moves its axis by the
            # Shafranov shift, as a move would. The iota loop holds the start's current until the field is first
            # below the bar
            schedule["hold"] = True
            ts = cfg.stepper(seq)
            print(f"[mesh] start: the pole onto the start field's axis in {len(start['rounds'])} rounds, axis "
                  + " -> ".join(f"{r['axis_cells']:.3g}" for r in start["rounds"]) + " first cells (moves "
                  + ", ".join(f"{r['move_m']:.2e} m x{r['lam']:g}" for r in start["rounds"])
                  + f"), {start['axis_cells']:.2e} cells on the final map, {start['seconds']:.0f} s", flush=True)
        B0, ic, B_file = start_field(seq)
        results["ic"] = ic
        print(f"[ic] {ic['kind']} IC in {time.perf_counter() - t1:.1f}s: "
              + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                          for k, v in ic.items() if not isinstance(v, dict)), flush=True)
        state, it0 = initial_state(B0, ts), 0
    T_label = None                     # the flux label's temperature, solved once for the seed and the targets
    if cfg.iota or cfg.seed:
        t1 = time.perf_counter()
        _, T_label, info = label(seq, state.B_n, ANISOTROPY)
        print(f"[label] flux label of the start field: CG {int(info)}, {time.perf_counter() - t1:.1f}s", flush=True)
    if cfg.seed and not cfg.iota:
        B_seeded, rows = energy_seed(seq, state.B_n, iotas=cfg.seed.iotas, amplitudes=cfg.seed.amplitudes,
                                     scale=cfg.seed.scale, T=T_label)
        results["seed"] = rows
        state = initial_state(B_seeded, ts, step=it0)

    def checkpoint(name, state, step):
        """Write the checkpoint ``name`` of ``state`` at ``step``, with the map when it can have moved."""
        path = os.path.join(ckpt_dir, name)
        write_checkpoint(path, state, step, seq)
        if carry_map:
            write_map(path, seq.map)
        if cfg.mesh or iota_target is not None:
            with h5py.File(path, "a") as fh:
                if cfg.mesh:
                    fh.attrs.update(mesh_hold=int(schedule["hold"]))
                if iota_target is not None:
                    fh.attrs["iota_target"] = iota_target
    checkpoint(f"state_{it0:06d}.h5", state, it0)
    if cu.start and not cfg.output.restart:
        checkpoint("vacuum.h5", initial_state(B_file, ts, step=it0), it0)

    # --- the drive: resistive steps towards the reference current ---------
    if dr:
        B_star, ref_step = read_field(dr.reference, seq)
        print(f"[drive] B* from {dr.reference} (step {ref_step})", flush=True)
        if dr.reference_smoothing:
            # The heat step removes the current sheets of B* on rational surfaces. If kept, they would make the start a fixed point.
            B_star = resistive_step(B_star, seq, dr.reference_smoothing * h_r_sq)[0]
        if dr.chain is not None:
            B_star, results["drive"] = energy_seed(seq, B_star, iotas=(dr.chain,), amplitudes=(dr.eps,))
        ts = ts.with_reference(B_star)
        print(f"[drive] eps {dr.resistivity:g} h_r^2 = {ts.resistivity:.3e} per step, B* smoothed by "
              f"{dr.reference_smoothing:g} h_r^2, ||B - B*|| / ||B|| = "
              f"{float(seq.odd.l2_norm(state.B_n - B_star, 2) / seq.odd.l2_norm(state.B_n, 2)):.3e}", flush=True)
    # --- the prescribed pressure: the shell volumes as Newton unknowns, rebuilt every chunk ---
    targets = bool(pr or cu or cfg.iota)

    def label_of(state):
        """``(s_q, info)``: the flux label of the field of ``state`` at the quadrature points (one label for both
        targets, its temperature kept in ``T_label`` as the next solve's warm start)."""
        nonlocal T_label
        s_q, T_label, info = label(seq, state.B_n, pr.anisotropy, T_label, pr.label_tol)
        return s_q, int(info)
    if pr:
        if fit is None and pr.target.endswith(".json"):
            raise ValueError(f"--pressure.target {pr.target}: measured pressure is fitted by scripts/pressure_fit.py")
        target = load_target(seq, pr.target, pr.scale) if fit is None else None
        results["pressure"] = dict(target=pr.target, rebuilds=[])
        print(f"[pressure] target {pr.target}" + (f", scaled to beta {pr.beta:g}" if pr.beta is not None else ""), flush=True)

        def measure(state, it, s_q, info):
            """``p_q``: the target on the label ``s_q`` of the field of ``state``, at the quadrature points. The
            pressure's error against the target and the target's beta go into the record."""
            profile, record = (target, {}) if fit is None else fit(seq, state.B_n, T_label)
            p_q = profile(s_q)
            beta, scale = target_beta(seq, state.B_n, p_q), {}
            if pr.beta is not None:
                # scaled on this field: the shells change their volumes along the run, and with them the volume
                # mean of a function of the label
                scale = dict(scale=pr.beta / beta)
                p_q, beta = scale["scale"] * p_q, pr.beta
            err, const = pressure_error(seq, weak_pressure(current(state.B_n, seq), state.B_n, seq), s_q, p_q,
                                        ERROR_BINS)
            results["pressure"]["rebuilds"].append(dict(step=it, label_it=info, error=err, constant=const,
                                                        beta=beta, **scale, **record))
            print(f"[pressure] step {it}: |p - p*| / |p*| {err:.3e} in the gauge <p> = <p*> (p = p_w + C, "
                  f"C {const:+.2e} max p*), beta {beta:.5e}, label CG {info}", flush=True)
            return p_q
    # --- the prescribed current: the poloidal flux slips as Newton unknowns, rebuilt every chunk. With --iota.target
    # the current target is the start's and the iota loop's first target, the loop's slips replace these
    cu_slips = bool(cu) and not cfg.iota
    if cu:
        # "file" is the current of the geometry file's field, also on a restart
        if cfg.output.restart:
            B_file = initial_field(seq)[0]
        mu_star = load_current_target(seq, cu.target, B_file, cu.scale, anisotropy=pr.anisotropy,
                                      label_tol=pr.label_tol, cells=cells)
        results["current"] = dict(target=cu.target, x=spline_centres(mu_star.size).tolist(),
                                  target_moments=np.asarray(mu_star).tolist(), rebuilds=[])
        print(f"[current] target {cu.target}" + (f" x {cu.scale:g}" if cu.scale != 1.0 else "")
              + f": mu0 I near the axis {float(mu_star[0]):+.4e}, near the wall {float(mu_star[-1]):+.4e} T m "
              f"(along +zeta, the means over {mu_star.size} splines of the profile space in s)", flush=True)

        def record_current(it, mu, info):
            """The current's moments ``mu`` against the target's into the record."""
            err = current_error(mu, mu_star)
            results["current"]["rebuilds"].append(dict(step=it, label_it=info, error=err,
                                                       moments=np.asarray(mu).tolist()))
            what = "|I - I*| / |I*|" if np.any(np.asarray(mu_star) != 0.0) else "|I| (T m, the target is zero)"
            print(f"[current] step {it}: {what} {err:.3e} over the spline means of mu0 I, near the wall "
                  f"{float(mu[-1]):+.4e} T m (target {float(mu_star[-1]):+.4e}), label CG {info}", flush=True)
    if cfg.iota:
        # the target transform: the current target takes a Newton step on the transform's weak residual before every
        # chunk, from the current the field carries at the first, or from the start's current with --current.start
        if cu and iota_target is None:
            iota_target = np.asarray(mu_star, dtype=np.float64)
        # the target transform projected once onto the profile space, the space of the transform the loop measures
        iota_star = project_iota_target(load_iota_target(cfg.iota.target), cells + 2)
        results["iota"] = dict(target=cfg.iota.target, x=spline_centres(cells + 2).tolist(), rebuilds=[])

        def record_iota(it, rec, info):
            results["iota"]["rebuilds"].append(dict(step=it, label_it=info, **rec))
            print(f"[iota] step {it}: weak residual max {rec['r_max']:.3e} rms {rec['r_rms']:.3e}, of the equilibrium "
                  f"{np.abs(rec['r_eq']).max():.3e}, target near the wall "
                  f"{rec['lam'][-1]:+.4e} T m (mu {rec['mu'][-1]:+.4e}), MINRES {rec['minres'][0]} / "
                  f"{min(rec['minres'][1:])} .. {max(rec['minres'][1:])}, label CG {info}", flush=True)
        iota_max = float(np.abs(np.asarray(iota_star(np.linspace(0.05, 0.95, 19)))).max())

        def seed_converged(state, it, s_q):
            """With a target transform the seed waits for it: it resonates with the field's own transform,
            which the first chunks still move. The field is seeded at the first chunk boundary at which every local
            mean of the miss, the weak residual on the label, is below AT_TARGET of the target's largest value.
            Returns the state and whether it seeded."""
            r = iota_residual(seq, state.B_n, s_q, iota_star, cells + 2)(state.B_n)[0]
            miss = float(np.abs(np.asarray(r)).max())
            if miss >= AT_TARGET * iota_max:
                return state, False
            B_seeded, rows = energy_seed(seq, state.B_n, iotas=cfg.seed.iotas, amplitudes=cfg.seed.amplitudes,
                                         scale=cfg.seed.scale, T=T_label)
            results["seed"] = rows
            results["iota"]["seed_step"] = it
            print(f"[seed] step {it}: the weak residual of the transform {miss:.2e} < {AT_TARGET:g} of its largest value, "
                  f"seeded", flush=True)
            return initial_state(B_seeded, ts, step=it), True
    # --- the pole on the magnetic axis at an equilibrium, the preconditioners built again on need ---
    mm = cfg.mesh
    if mm:
        built = dict(ref=None, last=None, cap=False)

        def align_axis(state, it, before):
            """The state at step ``it`` after the move of the pole onto the field's magnetic axis, if one is due: at an
            equilibrium (residual below --mesh.bar) with the axis more than --mesh.move-tol first cells off the pole.
            ``before`` is the field's squared force residual. The field is carried onto the moved map, its residual
            right before and after the move goes into the record."""
            nonlocal ts
            schedule["resid"] = before
            d, h1 = mesh_motion.axis_offset(seq, mesh_motion.magnetic_axis(seq, state.B_n)[0])
            off = float(np.max(np.linalg.norm(d, axis=1))) / h1
            # not an equilibrium: its axis is not that of the field it relaxes to. The vacuum field of a fresh start
            # is below the bar but the equilibrium of no target: the start has put the pole where it is wanted
            why = ("the start's vacuum" if it == 0 and "start" in results["mesh"] else
                   "above the bar" if not before < mm.bar else
                   f"within {mm.move_tol:g} first cells" if not off > mm.move_tol else None)
            if why is not None:
                results["mesh"]["checks"].append(dict(step=it, axis_cells=off, resid=before, why=why))
                print(f"[mesh] step {it}: residual {before:.3e}, axis {off:.3e} first cells off the pole, no move "
                      f"({why})", flush=True)
                return state
            B, rec = mesh_motion.axis_move(seq, state.B_n)
            state = eqx.tree_at(lambda s_: s_.B_n, state, B)
            ts = cfg.stepper(seq)
            rec.update(step=it, resid_before=before, resid_after=float(initial_state(state.B_n, ts).best.resid))
            # the transform as a function of the toroidal flux and the equilibrium of the targets stay, the field is
            # kicked off it by the transfer's error. The iota loop holds its target until the field is back at an
            # equilibrium: read before, its residual and its Newton direction are the kick's
            schedule.update(resid=rec["resid_after"], hold=True)
            results["mesh"]["moves"].append(rec)
            print(f"[mesh] step {it}: axis {rec['axis_cells']:.3e} first cells off the pole, move {rec['move_m']:.3e} m "
                  f"(x{rec['lam']:g}), det DPhi ratio [{rec['det_ratio_min']:.4f}, {rec['det_ratio_max']:.4f}], "
                  f"transfer {rec['substeps']} substeps, |dB| / |B| {rec['dB_rel']:.2e}, energy {rec['dE_rel']:+.2e}, "
                  f"residual {before:.3e} -> {rec['resid_after']:.3e}, axis Newton {rec['axis']['newton']}, "
                  f"{rec['wall']:.1f}s", flush=True)
            # the warm starts stay, the best field is the best since this move: an older one lives on an older map
            best = BestState(B=state.B_n, resid=jnp.full_like(state.best.resid, jnp.inf), step=jnp.int32(it))
            state = eqx.tree_at(lambda s_: s_.best, state, best)
            checkpoint(f"moved_{it:06d}.h5", state, it)
            return state

        def precondition(it):
            """Build the preconditioners on the installed map if the last chunk's mean MINRES count exceeds
            --mesh.rebuild-factor times that of the first chunk after their last build, or a solve hit its cap while
            the field is above the bar: near the floor a solve at its cap is no sign of an old map."""
            nonlocal ts
            if built["last"] is None:
                return
            if built["ref"] is None:
                built["ref"] = built["last"]
            reason = ("a Newton solve at its cap" if built["cap"] and not schedule["resid"] < mm.bar else
                      f"MINRES {built['last']:.0f} > {mm.rebuild_factor:g} x {built['ref']:.0f}"
                      if built["last"] > mm.rebuild_factor * built["ref"] else None)
            results["mesh"]["chunks"].append(dict(step=it, minres=built["last"], ref=built["ref"], cap=built["cap"]))
            if reason is None:
                return
            t = time.perf_counter()
            seq.build_preconditioners()
            compute_nullspaces(seq, verbose=False)
            ts = cfg.stepper(seq)
            built.update(ref=None, last=None)
            results["mesh"]["builds"].append(dict(step=it, reason=reason, wall=time.perf_counter() - t))
            print(f"[mesh] step {it}: preconditioners built on the installed map ({reason}), "
                  f"{time.perf_counter() - t:.0f}s", flush=True)
    params["start_step"] = it0
    kwargs = cfg.relax_kwargs()
    if targets:
        # the force residual does not see the shells or the slips, so its floor does not end a run that still
        # approaches its target
        kwargs["floor_tol"] = 0.0
    newton = 'Gauss-Newton penalty=%g tol=%.1e maxiter=%d mass-solve-tol=%.0e' % (
        n.penalty, n.tol, n.maxiter, n.mass_solve_tol)
    print(f"\n=== {newton if d.newton else 'gradient descent'}"
          f"  smoothing@{ts.velocity_smoothing_scale:.3e}  steps<={b.steps} chunk={b.chunk} "
          f"floor-tol={kwargs['floor_tol']:.1e}"
          + (f"  drive: resistivity={dr.resistivity:g} h_r^2" if dr else "") + " ===", flush=True)

    # --- the run, in one call: before every chunk the mesh moves when a move is due and the label, the shells and
    # the slips are rebuilt on the field (mrx.relaxation.loop.relax's rebuild)
    # the continuation of the targets (--budget.stages): t, the stage k of the N stages, the start field's own
    # pressure at the quadrature points, its transform's weak residual and its current's moments
    cont = dict(t=0.0 if targets else 1.0, k=0, N=None, p0=None, r0=None, mu0=None)
    results["continuation"] = []

    def shift(tt):
        """The iota loop's residual shift at the continuation's ``tt``: (1 - tt) times the start's residual."""
        return None if tt >= 1.0 or cont["r0"] is None else (1.0 - tt) * cont["r0"]

    def rebuild(state, it):
        """The state and the stepper of the chunk from step ``it``."""
        nonlocal ts, seed_pending, iota_target
        updated = False
        resid = float(initial_state(state.B_n, ts).best.resid) if mm else None
        at_start = it == 0 and "start" in results.get("mesh", {})    # the vacuum of a fresh start, held
        if mm and cfg.iota and not at_start and resid < mm.bar:
            # at an equilibrium the loop reads the field before the mesh moves: the move would kick it off
            s_q, info = label_of(state)
            if iota_target is None:     # the loop starts from the current the field carries
                iota_target = np.asarray(current_on_label(seq, state.B_n, T_label, cells), dtype=np.float64)
            _, iota_target, rec = iota_newton(seq, state.B_n, T_label, s_q, iota_star, ts, iota_target,
                                              shift(cont["t"]))
            record_iota(it, rec, info)
            schedule["hold"], updated = False, True
        if mm:
            state = align_axis(state, it, resid)
            precondition(it)
        if not targets:
            return state, ts
        t_rebuild = time.perf_counter()
        s_q, info = label_of(state)
        if cfg.iota and seed_pending:
            state, seeded = seed_converged(state, it, s_q)
            if seeded:
                seed_pending = False
                s_q, info = label_of(state)
        first = cont["N"] is None
        if first:
            # the continuation starts from the start field's own values
            if pr:
                p_w = weak_pressure(current(state.B_n, seq), state.B_n, seq)
                cont["p0"] = np.asarray(seq.even.evaluate_at_quadrature(p_w, 0))[:, 0]
            if cfg.iota:
                cont["r0"] = np.asarray(iota_residual(seq, state.B_n, s_q, iota_star, cells + 2)(state.B_n)[0],
                                        dtype=np.float64)
            if cu_slips:
                # a start with the target's current has the current part at t = 1 from the start
                cont["mu0"] = (np.asarray(mu_star, dtype=np.float64) if cu.start else
                               np.asarray(current_on_label(seq, state.B_n, T_label, cells), dtype=np.float64))
        p_full = measure(state, it, s_q, info) if pr else None
        if first:
            cont["N"] = b.stages
            results["stages"] = cont["N"]
            print(f"[continuation] {cont['N']} stages", flush=True)
        if cfg.iota and iota_target is None:     # the loop starts from the current the field carries
            iota_target = np.asarray(current_on_label(seq, state.B_n, T_label, cells), dtype=np.float64)

        def build(tt):
            """The stepper with the targets at the continuation's ``tt`` and what to record once it is taken."""
            ts_, todo = ts, dict(iota_target=iota_target)
            if pr:
                p_q = p_full if tt >= 1.0 else (1.0 - tt) * jnp.asarray(cont["p0"]) + tt * p_full
                shells, shell_it = build_shells(seq, state.B_n, s_q, p_q, n.penalty, n.mass_solve_tol, cells)
                todo["shell_it"] = np.asarray(shell_it).tolist()
                ts_ = ts_.with_shells(shells)
            if cu_slips:
                mu_t = mu_star if tt >= 1.0 else (1.0 - tt) * jnp.asarray(cont["mu0"]) + tt * jnp.asarray(mu_star)
                slips, mu = build_slips(seq, state.B_n, T_label, jnp.asarray(mu_t, dtype=T_label.dtype))
                todo["mu"] = mu
                ts_ = ts_.with_slips(slips)
            if cfg.iota:
                if updated or schedule["hold"]:
                    # updated before the move, or held since the last move or the start until the field is below the bar
                    slips = build_slips(seq, state.B_n, T_label, jnp.asarray(iota_target, dtype=T_label.dtype))[0]
                else:
                    slips, todo["iota_target"], todo["iota_rec"] = iota_newton(seq, state.B_n, T_label, s_q, iota_star,
                                                                              ts_, iota_target, shift(tt))
                ts_ = ts_.with_slips(slips)
            return ts_, todo

        if cont["k"] < cont["N"]:
            # the next stage of the continuation
            cont["k"] += 1
            cont["t"] = cont["k"] / cont["N"]
            ts_try, todo = build(cont["t"])
            results["continuation"].append(dict(step=it, t=cont["t"]))
            print(f"[continuation] step {it}: stage {cont['k']} of {cont['N']}, t {cont['t']:.4g}", flush=True)
        else:
            ts_try, todo = build(1.0)
        ts = ts_try
        if pr:
            results["pressure"]["rebuilds"][-1]["shell_it"] = todo["shell_it"]
            print(f"[pressure] step {it}: shell gradients, Poisson CG {min(todo['shell_it'])} .. "
                  f"{max(todo['shell_it'])}", flush=True)
        if cu_slips:
            record_current(it, todo["mu"], info)
        if cfg.iota:
            iota_target = todo["iota_target"]
            if "iota_rec" in todo:
                record_iota(it, todo["iota_rec"], info)
            elif not updated:
                results["iota"].setdefault("held", []).append(it)
                print(f"[iota] step {it}: target held since the last move or the start, residual "
                      f"{schedule['resid']:.3e}", flush=True)
        # the cost of the rebuild of the label, the shells and the slips, outside the steps
        results.setdefault("rebuild_wall", []).append(time.perf_counter() - t_rebuild)
        return state, ts

    def save(res):
        """Write the checkpoint of this step, then rewrite relax.json with the run so far."""
        it = it0 + res.steps
        checkpoint(f"state_{it:06d}.h5", res.state, it)
        trace, qoi = res.trace, res.qoi
        if mm and d.newton:
            nit = np.abs(np.asarray(trace["newton_it"][-res.chunk:]))
            built.update(last=float(nit.mean()), cap=bool(nit.max() >= n.maxiter))
        last = {k: v[-1] for k, v in qoi.items() if k not in ("it", "wall")}
        results.update(
            trace=trace, qoi=qoi,
            summary=dict(steps=res.steps, stop=res.stop, wall=res.wall,
                         E0=qoi["E"][0], E_removed=qoi["E"][0] - qoi["E"][-1], F_final=trace["F"][-1],
                         resid_final=trace["resid"][-1],
                         resid_window_mean=float(sum(trace["resid"][-res.chunk:]) / res.chunk),
                         best_step=int(res.state.best.step), best_resid=float(res.state.best.resid),
                         **last))
        write()

    def write():
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)

    seed_pending = bool(cfg.iota and cfg.seed)
    res = relax(state, ts, it0=it0, on_chunk=save, rebuild=rebuild if targets or mm else None, **kwargs)
    state, it = res.state, it0 + res.steps
    if targets:
        # the end state against the targets, and what the steps removed of the energy less the targets' work
        s_q, info = label_of(state)
        results["summary"].update(W_removed=-sum(res.trace["dW"]))
        if pr:
            measure(state, it, s_q, info)
            end = results["pressure"]["rebuilds"][-1]
            results["summary"].update(beta=end["beta"], pressure_error=end["error"],
                                      pressure_constant=end["constant"])
        if cu:
            record_current(it, current_on_label(seq, state.B_n, T_label, cells), info)
            results["summary"].update(current_error=results["current"]["rebuilds"][-1]["error"])
        if cfg.iota:
            r = np.asarray(iota_residual(seq, state.B_n, s_q, iota_star, cells + 2)(state.B_n)[0], dtype=np.float64)
            results["iota"]["end"] = dict(step=it, label_it=info, r=r.tolist(), r_max=float(np.abs(r).max()))
            print(f"[iota] step {it}: weak residual max {np.abs(r).max():.3e} at the end", flush=True)
            results["summary"].update(iota_r_max=float(np.abs(r).max()))
        write()
    checkpoint("best.h5", initial_state(res.state.best.B, ts, step=int(res.state.best.step)),
               int(res.state.best.step))
    print(f"wrote {out}/relax.json and {ckpt_dir}/ (best.h5: step {int(res.state.best.step)}, "
          f"residual {float(res.state.best.resid):.3e})", flush=True)
    return out


if __name__ == "__main__":
    # The precision must be in the environment before mrx is imported. The full parse follows.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--geometry.precision", dest="precision", default="float64", choices=("float64", "float32"))
    os.environ["MRX_DTYPE"] = pre.parse_known_args()[0].precision
    import tyro
    from mrx.relaxation.config import RelaxConfig
    main(tyro.cli(RelaxConfig, description=__doc__))
