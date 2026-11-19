"""Relax a magnetic field toward minimum energy at fixed helicity. This is the command line of :func:`mrx.relaxation.loop.relax`.

The script builds the de Rham sequence of ``--geometry.path`` and the initial field on it, or reads the field of
an earlier run with ``--output.restart``. It can then open island chains (``--seed``, see
:func:`mrx.relaxation.seeding.energy_seed`). The descent runs in compiled chunks of steps until the force residual
stops falling or the step budget is spent. The fixed point satisfies ``J x B = grad p``, where ``p`` is the
pressure that keeps the flow divergence-free, so the result is a finite-beta equilibrium.

With ``--drive.resistivity C`` every step is followed by a resistive step of size ``C h_r^2`` (``h_r`` the radial
cell size) that pulls the current towards that of the ``--drive.reference`` field. The reference is first smoothed
by one heat step of ``--drive.reference-smoothing`` h_r^2 and can be given an island chain with ``--drive.chain``
and ``--drive.eps``.

With ``--pressure.target`` the volumes of the flux shells are unknowns of every Newton step, so the relaxation ends
at the target pressure, a profile (the file's or a VMEC input namelist's, scaled by ``--pressure.scale`` or to
``--pressure.beta``) or measured data on flux surfaces or at points in space (:mod:`mrx.relaxation.shells`). The
shells and the target are rebuilt on the flux label of the field before every chunk. With ``--seed`` the target is
made constant across the seeded chains (``--pressure.plateau-margin``). Such a run takes all of ``--budget.steps``:
the force residual does not see the shells, so ``--budget.floor-tol`` does not end it.

With ``--mesh.align`` the mesh moves towards the flux surfaces of the field between chunks, from step ``--mesh.start``
on and every ``--mesh.every`` steps, until the flux label is a function of the logical radius to ``--mesh.tol`` and
the magnetic axis sits on the mesh axis (:mod:`mrx.relaxation.mesh_motion`). The field is carried so that the physical
field stays. A move happens before the shells are rebuilt and does not change the target.

The moves keep the preconditioners of the start map. Once the mesh is aligned or out of moves those of the final map
are built.

    python -u scripts/relax.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution 16 32 32

``--help`` lists every flag with its default. The flags are the fields of the dataclasses in
:mod:`mrx.relaxation.config`, one group each, spelled ``--<group>.<field>`` (for example ``--newton.tol``).

Output (in ``--output.out``):
    relax.json                   The configuration (flat, plus facts of the run) under ``params``, then ``ic``,
                                 ``seed``, ``drive``, ``pressure`` (the target, its plateaus and before every chunk
                                 and at the end the pressure's error against it and the target's beta), ``mesh``
                                 (every move of the mesh, the step it was aligned at and the build of the final
                                 map's preconditioners), the per-step
                                 ``trace``, the per-chunk ``qoi`` and the ``summary``. The file is rewritten after every chunk.
    checkpoints/state_<step>.h5  The descent state at the start and after every chunk
                                 (:func:`mrx.relaxation.loop.write_checkpoint`). ``best.h5`` holds the field with
                                 the lowest residual, since the last move of the mesh. With ``--mesh.align`` every
                                 checkpoint also holds its map (:func:`mrx.relaxation.mesh_motion.write_map`), which a
                                 restart and ``scripts/poincare_trace.py`` install, and the state of the
                                 mesh's schedule (attributes mesh_moves, mesh_last, mesh_aligned, -1 for none).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py

def main(cfg):
    import numpy as np

    import mrx
    from mrx.relaxation.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.paths import data_file, runs_root
    from mrx.relaxation.loop import initial_state, radial_cell_sq, read_checkpoint, read_field, relax, write_checkpoint
    from mrx.nullspace import zeta_harmonic_distance
    from mrx.relaxation.mesh_motion import install_checkpoint_map, write_map
    from mrx.relaxation.physics import current, resistive_step, weak_pressure
    from mrx.relaxation.shells import (build_shells, flattened, label, load_target, plateau_intervals, pressure_error,
                                       target_beta)
    from mrx.relaxation.seeding import energy_seed

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
    carry_map = bool(cfg.mesh) or g.zeta == "harmonic"
    if g.zeta == "harmonic":
        params["harmonic"] = seq.equilibrium["harmonic"]
    if cfg.output.restart and install_checkpoint_map(seq, cfg.output.restart):
        carry_map = True
        print(f"[restart] the moved map of {cfg.output.restart} installed", flush=True)

    # --- the start field: the initial condition or a restart, then the seed ---
    h_r_sq = radial_cell_sq(seq)
    params["h_r_sq"] = float(h_r_sq)
    ts = cfg.stepper(seq)
    # the mesh's schedule: the moves made, the step of the last one and the step it was aligned at, kept in every
    # checkpoint so that a restart moves on the steps the run would have moved on
    motion = dict(moves=0, last=None, aligned=None)
    if cfg.output.restart:
        state, it0 = read_checkpoint(cfg.output.restart, ts)
        print(f"[restart] {cfg.output.restart}: descent state at step {it0}", flush=True)
        with h5py.File(cfg.output.restart, "r") as fh:
            if cfg.mesh and "mesh_moves" in fh.attrs:
                motion.update({k: None if v < 0 else int(v) for k, v in
                               (("moves", fh.attrs["mesh_moves"]), ("last", fh.attrs["mesh_last"]),
                                ("aligned", fh.attrs["mesh_aligned"]))})
                print(f"[restart] mesh schedule: {motion}", flush=True)
    else:
        t1 = time.perf_counter()
        B0, ic = initial_field(seq)
        results["ic"] = ic
        print(f"[ic] {ic['kind']} IC in {time.perf_counter() - t1:.1f}s: "
              + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                          for k, v in ic.items() if k != "kind"), flush=True)
        state, it0 = initial_state(B0, ts), 0
    if cfg.seed:
        B_seeded, rows = energy_seed(seq, state.B_n, iotas=cfg.seed.iotas, amplitudes=cfg.seed.amplitudes,
                                     scale=cfg.seed.scale)
        results["seed"] = rows
        state = initial_state(B_seeded, ts, step=it0)

    def checkpoint(name, state, step):
        """Write the checkpoint ``name`` of ``state`` at ``step``, with the map when it can have moved."""
        path = os.path.join(ckpt_dir, name)
        write_checkpoint(path, state, step, seq)
        if carry_map:
            write_map(path, seq.map)
        if cfg.mesh:
            with h5py.File(path, "a") as fh:
                fh.attrs.update({f"mesh_{k}": -1 if v is None else v for k, v in motion.items()})
    checkpoint(f"state_{it0:06d}.h5", state, it0)

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
    pr = cfg.pressure
    T_label = None                        # the label's temperature, the guess of the next label solve
    if pr:
        target = load_target(seq, pr.target, pr.scale, pr.smoothing)
        plateaus = (plateau_intervals(results["seed"], pr.plateau_margin)
                    if cfg.seed and pr.plateau_margin and target.profile is not None else [])
        if plateaus:
            target = target._replace(profile=flattened(target.profile, plateaus))
        results["pressure"] = dict(target=pr.target, plateaus=plateaus, rebuilds=[])
        print(f"[pressure] target {pr.target}" + (f", constant on s in {plateaus}" if plateaus else "")
              + (f", scaled to beta {pr.beta:g}" if pr.beta is not None else ""), flush=True)

        def measure(state, it):
            """``(s_q, p_q)``: the label of the field of ``state`` and the target on it, at the quadrature points. The
            pressure's error against the target and the target's beta go into the record."""
            nonlocal T_label
            s_q, T_label, info = label(seq, state.B_n, pr.anisotropy, T_label, pr.label_tol)
            p_q = target.at(seq, state.B_n, T_label)(s_q)
            beta, scale = target_beta(seq, state.B_n, p_q), {}
            if pr.beta is not None:
                # scaled on this field: the shells change their volumes along the run, and with them the volume
                # mean of a function of the label
                scale = dict(scale=pr.beta / beta)
                p_q, beta = scale["scale"] * p_q, pr.beta
            err, const = pressure_error(seq, weak_pressure(current(state.B_n, seq), state.B_n, seq), s_q, p_q,
                                        int(seq.ns[0]))
            results["pressure"]["rebuilds"].append(dict(step=it, label_it=int(info), error=err, constant=const,
                                                        beta=beta, **scale))
            print(f"[pressure] step {it}: |p - p*| / |p*| {err:.3e} in the gauge <p> = <p*> (p = p_w + C, "
                  f"C {const:+.2e} max p*), beta {beta:.5e}, label CG {int(info)}", flush=True)
            return s_q, p_q
    # --- the motion of the mesh towards the flux surfaces, between chunks ---
    mm = cfg.mesh
    if mm:
        from mrx.relaxation import mesh_motion
        results["mesh"] = dict(moves=[], aligned=None)

        def final_map(it):
            """Build the preconditioners and the harmonic forms of the final map, in place of those of the start
            map that the moves kept."""
            nonlocal ts
            t = time.perf_counter()
            seq.build_preconditioners()
            compute_nullspaces(seq, verbose=False)
            ts = cfg.stepper(seq)
            results["mesh"]["preconditioners"] = dict(step=it, wall=time.perf_counter() - t)
            print(f"[mesh] step {it}: preconditioners of the final map, {time.perf_counter() - t:.0f}s", flush=True)

        def align_mesh(state, it):
            """The state at step ``it`` after a move of the mesh, or ``state`` when no move is due. The label is
            solved on the current mesh, and again on the new one after a move."""
            nonlocal T_label, ts, target
            if (motion["aligned"] is not None or motion["moves"] >= mm.moves or it < mm.start
                    or (motion["last"] is not None and it - motion["last"] < mm.every)):
                return state
            s_q, T_label, info = label(seq, state.B_n, pr.anisotropy, T_label, pr.label_tol)
            _, e_rms, _ = mesh_motion.alignment(seq, s_q)
            d, h1 = mesh_motion.axis_offset(seq, T_label)
            axis_cells = float(np.max(np.linalg.norm(d, axis=1)) / h1)
            if mesh_motion.aligned(e_rms, d, h1, np.inf if mm.axis_only else mm.tol):
                results["mesh"]["aligned"] = dict(step=it, e_rms=e_rms, axis_cells=axis_cells, label_cg=int(info))
                motion["aligned"] = it
                print(f"[mesh] step {it}: aligned, |s - S(r)| rms {e_rms:.3e}, axis {axis_cells:.3f} cells off, "
                      f"no more moves", flush=True)
                if motion["last"] is not None and motion["last"] >= it0:
                    # moved in this run: a restart installs its checkpoint's map with that map's preconditioners
                    final_map(it)
                return state
            B, rec = mesh_motion.move(seq, state.B_n, T_label, velocity=not mm.axis_only)
            # the temperature of the old mesh's logical points warm-starts the label of the new mesh
            s_q, T_label, info_after = label(seq, B, pr.anisotropy, T_label, pr.label_tol)
            _, e_after, e_max_after = mesh_motion.alignment(seq, s_q)
            d, h1 = mesh_motion.axis_offset(seq, T_label)
            rec.update(step=it, label_cg=int(info), label_cg_after=int(info_after), e_rms_after=e_after,
                       zeta_drift=zeta_harmonic_distance(seq)[0],
                       e_max_after=e_max_after, axis_cells_after=float(np.max(np.linalg.norm(d, axis=1)) / h1))
            results["mesh"]["moves"].append(rec)
            motion.update(moves=motion["moves"] + 1, last=it)
            print(f"[mesh] step {it}: move {motion['moves']}, |s - S(r)| rms {rec['e_rms']:.3e} -> {e_after:.3e}, axis "
                  f"{rec['axis_cells']:.2f} -> {rec['axis_cells_after']:.2f} cells, largest move {rec['move_m']:.3e} m "
                  f"(axis step x{rec['lam_axis']:g}, velocity x{rec['lam_v']:g}), det DPhi ratio "
                  f"[{rec['det_ratio_min']:.3f}, {rec['det_ratio_max']:.3f}], physical B changed by "
                  f"{rec['dB_phys']:.2e}, energy by {rec['dE_rel']:+.2e}, |grad zeta - h| / |h| {rec['zeta_drift']:.2e}, "
                  f"label CG {info} / {info_after}, "
                  f"{rec['wall']:.0f}s", flush=True)
            if motion["moves"] < mm.moves:
                ts = cfg.stepper(seq)
            else:
                final_map(it)
            if pr and target.data is not None:
                # measured points are held as logical points of the map they were located in
                target = load_target(seq, pr.target, pr.scale, pr.smoothing)
            return initial_state(B, ts, step=it)
    params["start_step"] = it0
    kwargs = cfg.relax_kwargs()
    if pr:
        # the force residual does not see the shells, so its floor does not end a run that still approaches its target
        kwargs["floor_tol"] = 0.0
    newton = 'Gauss-Newton penalty=%g tol=%.1e maxiter=%d mass-solve-tol=%.0e' % (
        n.penalty, n.tol, n.maxiter, n.mass_solve_tol)
    print(f"\n=== {newton if d.newton else 'gradient descent'}"
          f"  smoothing@{ts.velocity_smoothing_scale:.3e}  steps<={b.steps} chunk={b.chunk} "
          f"floor-tol={kwargs['floor_tol']:.1e}"
          + (f"  drive: resistivity={dr.resistivity:g} h_r^2" if dr else "") + " ===", flush=True)

    # --- the run, in one call, or with the prescribed pressure one call per chunk with the shells rebuilt before it
    every = kwargs["chunk"] if pr or mm else kwargs["steps"]
    kwargs["steps"] = every
    done = dict(trace={}, qoi={})                  # the earlier calls

    def save(res, start):
        """Write the checkpoint of this step, then rewrite relax.json with the run so far."""
        it = start + res.steps
        checkpoint(f"state_{it:06d}.h5", res.state, it)
        trace = {k: done["trace"].get(k, []) + list(v) for k, v in res.trace.items()}
        qoi = {k: done["qoi"].get(k, []) + list(v) for k, v in res.qoi.items()}
        last = {k: v[-1] for k, v in res.qoi.items() if k not in ("it", "wall")}
        results.update(
            trace=trace, qoi=qoi,
            summary=dict(steps=it - it0, stop=res.stop, wall=res.wall,
                         E0=qoi["E"][0], E_removed=qoi["E"][0] - qoi["E"][-1], F_final=trace["F"][-1],
                         resid_final=trace["resid"][-1],
                         resid_window_mean=float(sum(trace["resid"][-res.chunk:]) / res.chunk),
                         best_step=int(res.state.best.step), best_resid=float(res.state.best.resid),
                         **last))
        write()

    def write():
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)

    it = it0
    for _ in range(b.steps // every):
        start = it
        if mm:
            state = align_mesh(state, it)
        if pr:
            s_q, p_q = measure(state, it)
            ts = ts.with_shells(build_shells(seq, state.B_n, s_q, p_q, n.penalty, n.mass_solve_tol))
        res = relax(state, ts, it0=start, on_chunk=lambda r, start=start: save(r, start), **kwargs)
        for key in ("trace", "qoi"):
            for name, v in getattr(res, key).items():
                done[key][name] = done[key].get(name, []) + list(v)
        it, state = start + res.steps, res.state
        if res.stop != "steps":
            break
    if pr:
        # the end state against the target, and what the steps removed of the energy less the target pressure's work
        measure(state, it)
        end = results["pressure"]["rebuilds"][-1]
        results["summary"].update(W_removed=-sum(done["trace"]["dW"]), beta=end["beta"],
                                  pressure_error=end["error"], pressure_constant=end["constant"])
        write()
    checkpoint("best.h5", initial_state(res.state.best.B, ts, step=int(res.state.best.step)),
               int(res.state.best.step))
    print(f"wrote {out}/relax.json and {ckpt_dir}/ (best.h5: step {int(res.state.best.step)}, "
          f"residual {float(res.state.best.resid):.3e})", flush=True)


if __name__ == "__main__":
    # The precision must be in the environment before mrx is imported. The full parse follows.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--geometry.precision", dest="precision", default="float64", choices=("float64", "float32"))
    os.environ["MRX_DTYPE"] = pre.parse_known_args()[0].precision
    import tyro
    from mrx.relaxation.config import RelaxConfig
    main(tyro.cli(RelaxConfig, description=__doc__))
