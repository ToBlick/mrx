"""Relax a magnetic field toward minimum energy at fixed helicity.

The command line of :func:`mrx.relaxation.relax`: builds the geometry
(:func:`mrx.geometry.build_sequence`), the initial field
(:func:`mrx.initial_conditions.initial_field`) and the stepper
(:class:`mrx.relaxation.TimeStepper`), runs the descent in compiled chunks
until the force residual floors or the step budget is spent, and writes the
run. The descent is ideal, ``B_{n+1} = B_n + dt curl(u x B)`` (or ``u x H``
with the auxiliary field); reconnection, when asked for, is one resistive
solve between chunks (``--reconnect-every``) or a resistive dose after every
step (``--resistivity``). The fixed point is ``J x B = grad p`` with ``p``
the Leray multiplier, so the relaxed state is a finite-beta equilibrium, not
a force-free field.

Canonical invocation (one GPU; see slurm/README.md)::

    python -u scripts/relax.py --geometry data/wout_li383_1.4m.nc


Every flag, grouped, with its default: ``--help`` (the dataclasses of mrx.relax_config, which a
tutorial builds in Python); the run's record ``relax.json`` ``params`` is the same configuration flat.

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
    checkpoint, a converged ideal run; or the initial field), or the field
    of the ``--reference`` checkpoint (one common reference for differently
    seeded arms), after one heat
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

#: --precision -> (MRX_DTYPE, MRX_RESIDUAL_DTYPE); read before mrx is imported (mrx.precision fixes the
#: dtypes at import), hence here and not on the configuration
PRECISIONS = {"mixed": ("float32", "float64"), "float32": ("float32", "float32"),
              "float64": ("float64", "float64")}


def main(cfg):
    import equinox as eqx
    import mrx
    from mrx.geometry import geometry_kind
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (initial_state, radial_cell_sq, read_checkpoint, relax, resistive_step,
                                write_checkpoint)

    g, d, n, b, rc, rs = cfg.geometry, cfg.descent, cfg.newton, cfg.budget, cfg.reconnect, cfg.resistive
    if (str(mrx.DTYPE), str(mrx.precision.RESIDUAL_DTYPE)) != PRECISIONS[g.precision]:
        raise ValueError(f"--precision {g.precision} but mrx runs in {mrx.DTYPE} "
                         f"with {mrx.precision.RESIDUAL_DTYPE} residuals")
    mrx.MAP_BATCH_SIZE_INNER = g.map_batch
    print(f"[env] mrx from {mrx.__file__}  precision {g.precision} ({mrx.DTYPE} solves, "
          f"{mrx.precision.RESIDUAL_DTYPE} residual)  map batch {g.map_batch or 'all'}", flush=True)
    out = cfg.output.out or os.path.join("outputs", "relax", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # the record: the configuration, flat, plus the facts of the run
    params = dict(cfg.params, out=out, geometry_path=os.path.abspath(g.path), ic=geometry_kind(g.path))
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq, ops = g.build()
    params.update(ns=list(seq.ns), knots=g.knots)
    compute_nullspaces(seq)
    print(f"[setup] {g.path} ns={seq.ns} p={g.p} tol={seq.tol:.1e}  n2_dbc={seq.odd.n(2, True)}  "
          f"operators+nullspaces {time.perf_counter() - t0:.1f}s", flush=True)

    # --- initial condition -----------------------------------------------
    t1 = time.perf_counter()
    B0, ic = initial_field(seq, cfg.seed.parsed() if cfg.seed else None)
    results["ic"] = ic
    print(f"[ic] {ic['kind']} IC in {time.perf_counter() - t1:.1f}s: "
          + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                      for k, v in ic.items() if k != "kind"), flush=True)

    # --- the descent -------------------------------------------------------
    h_r_sq = radial_cell_sq(seq)
    params["h_r_sq"] = h_r_sq
    ts = cfg.stepper(seq, h_r_sq)
    if cfg.output.restart:
        state, it0 = read_checkpoint(cfg.output.restart, ts)
        print(f"[restart] {cfg.output.restart}: descent state at step {it0}", flush=True)
    else:
        state, it0 = initial_state(B0, ts), 0
        write_checkpoint(os.path.join(ckpt_dir, "state_000000.h5"), state, 0)
    if rs.resistivity:
        # B*: the start field, its rational-surface sheets removed by the heat step (c = 0 keeps them: only the
        # drive then moves the steady state)
        B_star = state.B_n
        if rs.reference:
            import h5py  # noqa: PLC0415
            import jax.numpy as jnp  # noqa: PLC0415
            with h5py.File(rs.reference, "r") as fh:
                B_star = jnp.asarray(fh["B_n"][()], dtype=state.B_n.dtype)
                ref_step = int(fh.attrs["step"])
            print(f"[reference] B* from {rs.reference} (step {ref_step})", flush=True)
        if rs.reference_smoothing:
            B_star = resistive_step(B_star, seq, rs.reference_smoothing * h_r_sq)[0]
        if cfg.drive:
            # the drive dA of the seed, as the difference of the two histopolated fields at the unseeded field's
            # normalisation: d is linear, so the histopolation error of the unperturbed field cancels exactly
            drive = cfg.drive.parsed()
            B_d, ic_d = initial_field(seq, drive)
            dB_drive = B_d * (ic_d["B_norm_raw"] / ic["B_norm_raw"]) - B0
            B_star = B_star + dB_drive
            print(f"[drive] ({drive[0]},{drive[1]}) at rho {ic_d['seed_rho']:.3f}, eps {cfg.drive.eps:g}: "
                  f"||dB_drive|| / ||B|| = {float(seq.odd.l2_norm(dB_drive, 2) / seq.odd.l2_norm(B0, 2)):.3e}",
                  flush=True)
        ts = eqx.tree_at(lambda t: t.resistive_reference, ts, B_star, is_leaf=lambda x: x is None)
        print(f"[resistivity] eps {rs.resistivity:g} h_r^2 = {ts.resistivity:.3e} per step; B* = the "
              f"{'reference' if rs.reference else 'start'} field "
              f"after a heat step of {rs.reference_smoothing:g} h_r^2, ||B - B*|| / ||B|| = "
              f"{float(seq.odd.l2_norm(state.B_n - B_star, 2) / seq.odd.l2_norm(state.B_n, 2)):.3e}", flush=True)
    params["start_step"] = it0
    params["velocity_smoothing_scale"] = float(ts.velocity_smoothing_scale)    # the effective scale
    print(f"\n=== {'newton-MR penalty=%g tol=%.1e maxiter=%d passes=%d' % (n.penalty, n.tol, n.maxiter, n.passes) if d.newton else 'gradient descent'}"
          f"{'  potential-velocity' if d.potential_velocity else ''}  auxiliary-B-field={str(d.auxiliary_B_field).lower()}  "
          f"{'midpoint  ' if d.midpoint else ''}{'helicity-correction  ' if d.helicity_correction else ''}"
          f"smoothing={d.velocity_smoothing_order}@{ts.velocity_smoothing_scale:.3e} "
          f"cfl={d.cfl}  steps<={b.steps} chunk={b.chunk} floor-tol={b.floor_tol:.1e} "
          f"reconnect-every={rc.every}"
          + ((f" (eps {rc.eps:g} h_r^2 each" if rc.eps is not None else f" ({rc.helicity:.2%} of H each")
             + (f", steps {rc.window[0]}:{rc.window[1]})" if rc.window else ")") if rc.every else "")
          + (f" resistivity={rs.resistivity:g} h_r^2 (B* smoothed {rs.reference_smoothing:g} h_r^2)"
             if rs.resistivity else "")
          + " ===", flush=True)

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
                         best_step=int(res.state.best.step), best_resid=float(res.state.best.resid),
                         **last))
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)

    res = relax(state, ts, it0=it0, on_chunk=save, **cfg.relax_kwargs(h_r_sq))
    write_checkpoint(os.path.join(ckpt_dir, "best.h5"),
                     initial_state(res.state.best.B, ts, step=int(res.state.best.step)), int(res.state.best.step))
    print(f"wrote {out}/relax.json and {ckpt_dir}/ (best.h5: step {int(res.state.best.step)}, "
          f"residual {float(res.state.best.resid):.3e})", flush=True)


if __name__ == "__main__":
    # the precision must be in the environment before mrx is imported; the full parse then follows
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--precision", default="mixed", choices=tuple(PRECISIONS))
    os.environ["MRX_DTYPE"], os.environ["MRX_RESIDUAL_DTYPE"] = PRECISIONS[pre.parse_known_args()[0].precision]
    from mrx.cli import parse
    from mrx.relax_config import RelaxConfig
    main(parse(RelaxConfig, description=__doc__))
