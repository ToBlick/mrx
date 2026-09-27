r"""Trace field lines of relaxation checkpoints and archive their Poincare sections -- the expensive half.

Every checkpoint named on the command line (``checkpoints/state_<step>.h5``, ``best.h5``, a seeded
checkpoint, ...) is traced with ``mrx.poincare.poincare`` on the sequence its attributes describe
(:func:`mrx.relaxation.checkpoint_attrs`: resolution, degree, knots, symmetry) over the geometry file
``--geometry`` (the run's wout / GVEC state / analytic .json), and the weak pressure of the field
(``mrx.relaxation.weak_pressure``, zero on the wall) is evaluated at every crossing. The archive
``trace.npz`` is written next to the first checkpoint's run directory; ``scripts/poincare_plot.py``
renders it on the login node, so a change to the figure never repeats the trace. Every checkpoint of
one call is traced in ONE archive so the plotter can put them on one iota and one pressure scale.

    python -u scripts/poincare_trace.py --geometry data/wout_li383_1.4m.nc outputs/run/checkpoints/state_*.h5

Archive (numpy .npz): ``fields`` (the checkpoints' names, in order), ``planes``, ``resolution``, ``p``,
``nfp``, ``symmetry``, ``steps`` (per period), ``source``, ``trace_precision``; per field ``<f>_label``,
``<f>_step``, ``<f>_bsq`` (the volume mean of |B|^2, the plotter's pressure normalisation),
``<f>_iota``, ``<f>_iota_err``, ``<f>_iota_scatter``, ``<f>_seed_r``, ``<f>_keep``, ``<f>_chaotic``,
``<f>_shown``, ``<f>_drift``; per field and plane ``<f>_zeta<plane>_{R,Z,axisR,axisZ,logr,logth}`` and
``<f>_zeta<plane>_pressure``, the weak pressure at every crossing. Trace RESULTS only -- every rendering
choice is the plotter's.

Runtime: sequence build 1-3 min, then ~40 s per traced field at (16,32,32) on one H100 (the weak
pressure adds two solves per field). The integrator is compiled once per mesh and schedule.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class Trace:
    """Trace the field lines of relaxation checkpoints and archive their Poincare sections."""
    checkpoints: tuple[str, ...] = field(metadata=dict(
        positional=True, nargs="+", help="relaxation checkpoints (.h5), traced in this order into one archive"))
    geometry: str = field(metadata=dict(help="the geometry file the checkpoints were relaxed on"))
    lines: int = field(default=100, metadata=dict(help="field lines per field, from the axis to the edge"))
    periods: int = field(default=400, metadata=dict(help="toroidal periods per line"))
    planes: str = field(default="5", metadata=dict(
        help="a count of zeta planes over what the symmetry leaves distinct, or the planes as fractions of a period"))
    seed: int = field(default=0, metadata=dict(help="the random seed of the poloidal angles"))
    precision: Literal["float32", "float64"] = field(default="float32", metadata=dict(help="tracing precision"))
    out: Optional[str] = field(default=None, metadata=dict(
        help="archive path [trace.npz in the parent of the first checkpoint's directory]"))


def main(cli):
    import time

    import h5py
    import jax
    import jax.numpy as jnp

    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import build_sequence
    from mrx.poincare import planes_for, poincare, steps_for
    from mrx.relaxation import checkpoint_attrs, compute_force, weak_pressure

    planes = int(cli.planes) if cli.planes.isdigit() else [float(v) for v in cli.planes.split(",")]
    attrs = checkpoint_attrs(cli.checkpoints[0])
    fields, steps_of, dofs = [], {}, {}
    for path in cli.checkpoints:
        name = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(path, "r") as fh:
            dofs[name] = np.asarray(fh["B_n"], dtype=np.float64)
            steps_of[name] = int(fh.attrs["step"])
        fields.append(name)
    archive = cli.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cli.checkpoints[0]))),
                                      "trace.npz")
    source = f"{os.path.basename(cli.geometry)} {attrs['ns']} p={attrs['p']}, relaxed in {attrs['precision']}"
    print(f"[trace] {source}: fields {fields}, tracing in {cli.precision}", flush=True)

    seq, _ = build_sequence(cli.geometry, attrs["ns"], attrs["p"], knots=attrs["knots"], symmetry=attrs["symmetry"])
    planes = planes_for(seq, planes)
    print(f"[trace] nfp={seq.nfp}, symmetry {seq.symmetry}, planes {[f'{v:g}' for v in planes]} "
          f"({steps_for(planes)} steps per period)", flush=True)
    volume = float(jnp.sum(seq.quad.w * seq.jacobian_j))
    # The weak pressure is a diagnostic of the field, not state: two solves per field.
    pw, bsq = {}, {}
    for name in fields:
        B = jnp.asarray(dofs[name])
        _, _, J, X, _ = compute_force(B, seq)
        pw[name] = np.asarray(weak_pressure(J, X, seq)[0], dtype=np.float64)
        bsq[name] = float(seq.odd.l2_norm_sq(B, 2)) / volume

    @jax.jit
    def weak_at(pd, x):
        return jax.vmap(DiscreteFunction(pd, seq.basis_0, seq.even.E(0, True)))(x)[:, 0]

    def pressure_at(name, lr, lth, zeta):
        """The weak pressure at logical ``(lr, lth, zeta)``."""
        x = jnp.stack([jnp.asarray(lr).ravel(), jnp.asarray(lth).ravel(),
                       jnp.broadcast_to(jnp.asarray(zeta), lr.shape).ravel()], axis=1)
        return np.asarray(weak_at(jnp.asarray(pw[name]), x)).reshape(lr.shape)

    sections = {"fields": np.array(fields), "planes": np.array(planes), "resolution": np.array(seq.ns),
                "p": attrs["p"], "nfp": seq.nfp, "symmetry": np.array(seq.symmetry),
                "steps": steps_for(planes), "source": np.array(source),
                "trace_precision": np.array(cli.precision)}
    for name in fields:
        t_field = time.perf_counter()
        B = dofs[name]
        assert B.shape == (seq.odd.n(2, True),), (B.shape, seq.odd.n(2, True))
        res = poincare(seq, B, lines=cli.lines, periods=cli.periods, planes=planes, seed=cli.seed, name=name)
        sections[f"{name}_label"] = np.array(f"{name} (step {steps_of[name]})")
        sections[f"{name}_step"] = steps_of[name]
        sections[f"{name}_bsq"] = bsq[name]
        for key in ("iota", "iota_err", "iota_scatter", "seed_r", "keep", "chaotic", "shown", "drift"):
            sections[f"{name}_{key}"] = np.asarray(res[key])
        for plane, sec in res["sections"].items():
            tag = f"{name}_zeta{plane:g}"
            for key, arr in sec.items():
                sections[f"{tag}_{key}"] = arr
            sections[f"{tag}_pressure"] = pressure_at(name, sec["logr"], sec["logth"], plane)
        shown = res["shown"]
        span = (f"iota {float(res['iota'][shown].min()):.4f}..{float(res['iota'][shown].max()):.4f}"
                if shown.any() else "no line converged")
        print(f"[{name}] B^zeta/|B| in [{res['bz_over_b'][0]:+.3e}, {res['bz_over_b'][1]:+.3e}]; "
              f"trace {res['walltime']:.1f}s, field {time.perf_counter() - t_field:.1f}s, "
              f"{int((~res['keep']).sum())}/{res['keep'].size} lost, "
              f"{int((res['keep'] & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
    np.savez_compressed(archive, **sections)
    print(f"  -> {archive}", flush=True)


if __name__ == "__main__":
    # the precision must be in the environment before mrx is imported; the full parse then follows
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--precision", default="float32", choices=("float32", "float64"))
    os.environ["MRX_DTYPE"] = pre.parse_known_args()[0].precision
    from mrx.cli import parse
    sys.exit(main(parse(Trace, description=__doc__)))
