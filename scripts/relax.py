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

    python -u scripts/relax.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution 16 32 32

``--help`` lists every flag with its default. The flags are the fields of the dataclasses in
:mod:`mrx.relaxation.config`, one group each, spelled ``--<group>.<field>`` (for example ``--newton.tol``).

Output (in ``--output.out``):
    relax.json                   The configuration (flat, plus facts of the run) under ``params``, then ``ic``,
                                 ``seed``, ``drive``, the per-step ``trace``, the per-chunk ``qoi`` and the
                                 ``summary``. The file is rewritten after every chunk.
    checkpoints/state_<step>.h5  The descent state at the start and after every chunk
                                 (:func:`mrx.relaxation.loop.write_checkpoint`). ``best.h5`` holds the field with
                                 the lowest residual.
"""
from __future__ import annotations

import argparse
import json
import os
import time

#: Maps --geometry.precision to (MRX_DTYPE, MRX_RESIDUAL_DTYPE). They must be set before mrx is imported, because mrx.precision fixes the dtypes at import.
PRECISIONS = {"mixed": ("float32", "float64"), "float32": ("float32", "float32"),
              "float64": ("float64", "float64")}


def main(cfg):
    import equinox as eqx
    import h5py
    import jax.numpy as jnp
    import mrx
    from mrx.relaxation.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation.loop import (check_checkpoint, initial_state, radial_cell_sq, read_checkpoint, relax,
                                     write_checkpoint)
    from mrx.relaxation.physics import resistive_step
    from mrx.relaxation.seeding import energy_seed

    g, d, n, b, dr = cfg.geometry, cfg.descent, cfg.newton, cfg.budget, cfg.drive
    mrx.MAP_BATCH_SIZE_INNER = g.max_batch
    print(f"[env] mrx from {mrx.__file__}  precision {g.precision} ({mrx.DTYPE} solves, "
          f"{mrx.precision.RESIDUAL_DTYPE} residual)  batch {g.max_batch or 'all'}", flush=True)
    out = cfg.output.out or os.path.join("outputs", "relax", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # The record holds the flat configuration plus the facts of the run.
    params = dict(cfg.params, out=out, geometry_path=os.path.abspath(g.path))
    results = {"params": params}

    # --- geometry and operators ------------------------------------------
    t0 = time.perf_counter()
    seq, ops = g.build()
    params.update(ic=seq.equilibrium["kind"], resolution=list(seq.ns), knots=g.knots, nfp=seq.nfp)
    compute_nullspaces(seq)
    print(f"[setup] {g.path} resolution {seq.ns} degree {g.spline_degree} tol={seq.tol:.1e}  "
          f"n2_dbc={seq.odd.n(2)}  operators+nullspaces {time.perf_counter() - t0:.1f}s", flush=True)

    # --- the start field: the initial condition or a restart, then the seed ---
    h_r_sq = radial_cell_sq(seq)
    params["h_r_sq"] = float(h_r_sq)
    ts = cfg.stepper(seq)
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

    # --- the drive: resistive steps towards the reference current ---------
    if dr:
        check_checkpoint(dr.reference, seq)
        with h5py.File(dr.reference, "r") as fh:
            B_star = jnp.asarray(fh["B_n"][()], dtype=state.B_n.dtype)
            ref_step = int(fh.attrs["step"])
        print(f"[drive] B* from {dr.reference} (step {ref_step})", flush=True)
        if dr.reference_smoothing:
            # The heat step removes the current sheets of B* on rational surfaces. If kept, they would make the start a fixed point.
            B_star = resistive_step(B_star, seq, dr.reference_smoothing * h_r_sq)[0]
        if dr.chain is not None:
            B_star, results["drive"] = energy_seed(seq, B_star, iotas=(dr.chain,), amplitudes=(dr.eps,))
        ts = eqx.tree_at(lambda t: t.resistive_reference, ts, B_star, is_leaf=lambda x: x is None)
        print(f"[drive] eps {dr.resistivity:g} h_r^2 = {ts.resistivity:.3e} per step, B* smoothed by "
              f"{dr.reference_smoothing:g} h_r^2, ||B - B*|| / ||B|| = "
              f"{float(seq.odd.l2_norm(state.B_n - B_star, 2) / seq.odd.l2_norm(state.B_n, 2)):.3e}", flush=True)
    params["start_step"] = it0
    newton = 'newton-MR penalty=%g tol=%.1e maxiter=%d mass=%s' % (
        n.penalty, n.tol, n.maxiter, 'solve' if ts.mass_inverse is None else 'chebyshev-%d' % ts.mass_inverse.steps)
    print(f"\n=== {newton if d.newton else 'gradient descent'}"
          f"  smoothing@{ts.velocity_smoothing_scale:.3e}  steps<={b.steps} chunk={b.chunk} "
          f"floor-tol={b.floor_tol:.1e}"
          + (f"  drive: resistivity={dr.resistivity:g} h_r^2" if dr else "") + " ===", flush=True)

    def save(res):
        """Write the checkpoint of this step, then rewrite relax.json with the run so far."""
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
    # The precision must be in the environment before mrx is imported. The full parse follows.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--geometry.precision", dest="precision", default="mixed", choices=tuple(PRECISIONS))
    os.environ["MRX_DTYPE"], os.environ["MRX_RESIDUAL_DTYPE"] = PRECISIONS[pre.parse_known_args()[0].precision]
    import tyro
    from mrx.relaxation.config import RelaxConfig
    main(tyro.cli(RelaxConfig, description=__doc__))
