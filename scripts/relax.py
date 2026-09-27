"""Relax a magnetic field toward minimum energy at fixed helicity.

The command line of :func:`mrx.relaxation.relax`: builds the geometry
(:func:`mrx.geometry.build_sequence`), the initial field
(:func:`mrx.initial_conditions.initial_field`) and the stepper
(:class:`mrx.relaxation.TimeStepper`), runs the descent in compiled chunks
until the force residual floors or the step budget is spent, and writes the
run. The descent is ideal, ``B_{n+1} = B_n + dt curl(u x B)``; the drive,
when asked for, adds a resistive dose after every step
(``--drive-resistivity``). The fixed point is ``J x B = grad p`` with ``p``
the Leray multiplier, so the relaxed state is a finite-beta equilibrium, not
a force-free field.

Canonical invocation (one GPU; see slurm/README.md)::

    python -u scripts/relax.py --geometry data/wout_li383_1.4m.nc


Every flag, grouped, with its default: ``--help`` (the dataclasses of mrx.relax_config, which a
tutorial builds in Python); the run's record ``relax.json`` ``params`` is the same configuration flat.

Output (``--out``):
    relax.json           ``params`` (every flag, ``geometry_path`` resolved,
                         ``ic`` the kind of initial condition); ``ic``, the
                         initial field's numbers; ``seed`` and ``drive``, the
                         seeded chains (mrx.seeding); the per-step ``trace``,
                         the per-chunk ``qoi`` and the ``summary`` with the
                         stopping reason (the fields of
                         mrx.relaxation.RelaxResult). Rewritten at every chunk.
    checkpoints/state_<step>.h5
                         the descent state at that step, one file per chunk
                         plus the start field at its step
                         (mrx.relaxation.write_checkpoint, with the
                         discretisation as attributes); the tracer reads
                         them, ``--restart`` continues from one.

Seed (``--seed``): the energy criterion of mrx.seeding on the start field,
    the initial condition or the ``--restart`` checkpoint: every resonance
    in the field's iota range (or ``--seed-iotas``) gets SIESTA's parallel
    seed at the amplitudes of least energy, or at ``--seed-amplitudes``.

Drive (``--drive-resistivity C``): adds to EVERY step, after the ideal one, a
    backward-Euler step of ``dB/dt = -eta curl (J - J*)`` with the dose
    ``eps = C h_r^2`` (mrx.relaxation.TimeStepper.resistivity), ``J*`` the
    current of ``B*``, the field of the ``--drive-reference`` checkpoint (a
    converged nested equilibrium, one common reference for differently
    seeded arms) after one heat step of ``c h_r^2``, ``c =
    --drive-reference-smoothing``, which removes the rational-surface sheets
    of the ideal equilibrium (sustained, they would make the start a fixed
    point) and costs ``O(c h_r^2)`` of the bulk current; ``--drive-chain``
    and ``--drive-eps`` add a resonant seed to ``B*``, a drive that stays.
    The run goes to the resistive steady state (the islands open and
    saturate; the force residual is of order ``eps``); a restart of it
    without the drive then relaxes it ideally, the islands frozen in.
"""
from __future__ import annotations

import argparse
import json
import os
import time

#: --precision -> (MRX_DTYPE, MRX_RESIDUAL_DTYPE); read before mrx is imported (mrx.precision fixes the
#: dtypes at import), hence here and not on the configuration (mrx.relax_config.PRECISIONS is the same table)
PRECISIONS = {"mixed": ("float32", "float64"), "float32": ("float32", "float32"),
              "float64": ("float64", "float64")}


def main(cfg):
    import equinox as eqx
    import h5py
    import jax.numpy as jnp
    import mrx
    from mrx.geometry import geometry_kind
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (initial_state, radial_cell_sq, read_checkpoint, relax, resistive_step,
                                write_checkpoint)
    from mrx.seeding import energy_seed

    g, d, n, b, dr = cfg.geometry, cfg.descent, cfg.newton, cfg.budget, cfg.drive
    if (str(mrx.DTYPE), str(mrx.precision.RESIDUAL_DTYPE)) != PRECISIONS[g.precision]:
        raise ValueError(f"--precision {g.precision} but mrx runs in {mrx.DTYPE} "
                         f"with {mrx.precision.RESIDUAL_DTYPE} residuals")
    mrx.MAP_BATCH_SIZE_INNER = g.max_batch
    print(f"[env] mrx from {mrx.__file__}  precision {g.precision} ({mrx.DTYPE} solves, "
          f"{mrx.precision.RESIDUAL_DTYPE} residual)  batch {g.max_batch or 'all'}", flush=True)
    out = cfg.output.out or os.path.join("outputs", "relax", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # the record: the configuration, flat, plus the facts of the run
    params = dict(cfg.params, out=out, geometry_path=os.path.abspath(g.path), ic=geometry_kind(g.path))
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq, ops = g.build()
    params.update(resolution=list(seq.ns), knots=g.knots, nfp=seq.nfp)
    compute_nullspaces(seq)
    print(f"[setup] {g.path} resolution {seq.ns} degree {g.spline_degree} tol={seq.tol:.1e}  "
          f"n2_dbc={seq.odd.n(2, True)}  operators+nullspaces {time.perf_counter() - t0:.1f}s", flush=True)

    # --- the start field: the initial condition or a restart, then the seed --
    h_r_sq = radial_cell_sq(seq)
    params["h_r_sq"] = h_r_sq
    ts = cfg.stepper(seq, h_r_sq)
    if cfg.output.restart:
        state, it0 = read_checkpoint(cfg.output.restart, ts)
        print(f"[restart] {cfg.output.restart}: descent state at step {it0}", flush=True)
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
    write_checkpoint(os.path.join(ckpt_dir, f"state_{it0:06d}.h5"), state, it0, seq)

    # --- the drive: the resistive dose towards the reference current -----
    if dr:
        with h5py.File(dr.reference, "r") as fh:
            B_star = jnp.asarray(fh["B_n"][()], dtype=state.B_n.dtype)
            ref_step = int(fh.attrs["step"])
        print(f"[drive] B* from {dr.reference} (step {ref_step})", flush=True)
        if dr.reference_smoothing:
            # the heat step removes the rational-surface sheets of the ideal equilibrium (sustained, they would
            # make the start a fixed point) and costs O(c h_r^2) of the bulk current
            B_star = resistive_step(B_star, seq, dr.reference_smoothing * h_r_sq)[0]
        if dr.chain is not None:
            B_star, results["drive"] = energy_seed(seq, B_star, iotas=(dr.chain,), amplitudes=(dr.eps,))
        ts = eqx.tree_at(lambda t: t.resistive_reference, ts, B_star, is_leaf=lambda x: x is None)
        print(f"[drive] eps {dr.resistivity:g} h_r^2 = {ts.resistivity:.3e} per step; B* smoothed by "
              f"{dr.reference_smoothing:g} h_r^2, ||B - B*|| / ||B|| = "
              f"{float(seq.odd.l2_norm(state.B_n - B_star, 2) / seq.odd.l2_norm(state.B_n, 2)):.3e}", flush=True)
    params["start_step"] = it0
    print(f"\n=== {'newton-MR penalty=%g tol=%.1e maxiter=%d' % (n.penalty, n.tol, n.maxiter) if d.newton else 'gradient descent'}"
          f"  scheme={d.scheme}  smoothing@{ts.velocity_smoothing_scale:.3e}  steps<={b.steps} chunk={b.chunk} "
          f"floor-tol={b.floor_tol:.1e}"
          + (f"  drive: resistivity={dr.resistivity:g} h_r^2" if dr else "") + " ===", flush=True)

    def save(res):
        """The run so far: the checkpoint of this step, then relax.json."""
        it = it0 + res.steps
        write_checkpoint(os.path.join(ckpt_dir, f"state_{it:06d}.h5"), res.state, it, seq)
        last = {k: v[-1] for k, v in res.qoi.items() if k not in ("it", "wall")}
        results.update(
            trace=res.trace, qoi=res.qoi,
            summary=dict(steps=res.steps, stop=res.stop, wall=res.wall,
                         E0=res.E0, E_removed=res.E0 - res.qoi["E"][-1], F_final=res.trace["F"][-1],
                         resid_final=res.trace["resid"][-1],
                         resid_window_mean=float(sum(res.trace["resid"][-res.chunk:]) / res.chunk),
                         best_step=int(res.state.best.step), best_resid=float(res.state.best.resid),
                         **last))
        with open(os.path.join(out, "relax.json"), "w") as fh:
            json.dump(results, fh, indent=1)

    res = relax(state, ts, it0=it0, on_chunk=save, **cfg.relax_kwargs())
    write_checkpoint(os.path.join(ckpt_dir, "best.h5"),
                     initial_state(res.state.best.B, ts, step=int(res.state.best.step)), int(res.state.best.step), seq)
    print(f"wrote {out}/relax.json and {ckpt_dir}/ (best.h5: step {int(res.state.best.step)}, "
          f"residual {float(res.state.best.resid):.3e})", flush=True)


if __name__ == "__main__":
    # the precision must be in the environment before mrx is imported; the full parse then follows
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--precision", default="float32", choices=tuple(PRECISIONS))
    os.environ["MRX_DTYPE"], os.environ["MRX_RESIDUAL_DTYPE"] = PRECISIONS[pre.parse_known_args()[0].precision]
    from mrx.cli import parse
    from mrx.relax_config import RelaxConfig
    main(parse(RelaxConfig, description=__doc__))
