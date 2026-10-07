r"""Trace the field lines of relaxation checkpoints and save their Poincare sections. This is the expensive half.

Every checkpoint named on the command line (``checkpoints/state_<step>.h5``, ``best.h5``, a seeded checkpoint and
so on) is traced with ``mrx.diagnostics.poincare.poincare``. The sequence is rebuilt from the resolution, degree,
knots and symmetry stored in the checkpoint (:func:`mrx.relaxation.loop.checkpoint_attrs`) on the geometry file
``--geometry``, which is the run's VMEC wout, GVEC state or DESC output. The weak pressure of each field
(``mrx.relaxation.physics.weak_pressure``, zero on the wall) is evaluated at every crossing.

The archive ``trace.npz`` goes into the run directory of the first checkpoint, the parent of its ``checkpoints/``
folder. ``scripts/poincare_plot.py`` draws it and is cheap enough for the login node, so changing the figure never
repeats the trace. All checkpoints of one call go into ONE archive, so the plotter can put them on one iota scale
and one pressure scale.

    python -u scripts/poincare_trace.py --geometry data/wout_li383_1.4m.nc outputs/run/checkpoints/state_*.h5

The archive (numpy .npz) holds ``fields`` (the checkpoint names, in order), ``planes``, ``resolution``, ``p``,
``nfp``, ``symmetry``, ``steps`` (per period), ``source``, ``trace_precision`` and ``section_labels`` (the names of
the section coordinates stored as ``R``, ``Z``: ``R``, ``Z``, or ``X1``, ``X2`` of a G-frame). Per field ``<f>`` it holds
``<f>_label``, ``<f>_step``, ``<f>_bsq`` (the volume mean of |B|^2, which the plotter uses to normalise the
pressure), ``<f>_iota``, ``<f>_seed_r``, ``<f>_keep``, ``<f>_chaotic``, ``<f>_shown`` and ``<f>_drift``. Per field
and plane it holds ``<f>_zeta<plane>_{R,Z,axisR,axisZ,logr,logth}`` and ``<f>_zeta<plane>_pressure``, the weak
pressure at every crossing. The archive holds trace results only. Every choice about the drawing is left to the
plotter.

The integrator is compiled once per mesh and step schedule. The weak pressure costs two extra solves per field.
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tyro


@dataclass(frozen=True)
class Trace:
    """Trace the field lines of relaxation checkpoints and archive their Poincare sections."""
    checkpoints: tyro.conf.Positional[tuple[str, ...]]
    """The relaxation checkpoints (.h5), traced in this order into one archive."""
    geometry: str
    """The geometry file the checkpoints were relaxed on."""
    lines: int = 160
    """The number of field lines per field, started from the axis to the edge."""
    periods: int = 400
    """The number of toroidal periods each line is followed."""
    planes: int | tuple[float, ...] = 5
    """The number of zeta planes spread over the part of the period the symmetry leaves distinct, or the planes themselves as fractions of a period (for example 0 0.25 0.5)."""
    seed: int = 0
    """The random seed for the poloidal start angles."""
    precision: Literal["float32", "float64"] = "float32"
    """The floating-point precision of the trace."""
    out: Optional[str] = None
    """The archive path. Unset, it is trace.npz in the parent of the first checkpoint's directory."""

    def __post_init__(self):
        if not self.checkpoints:
            raise ValueError("name at least one checkpoint to trace")


def main(cli):
    import h5py

    from mrx.geometry import build_sequence
    from mrx.diagnostics.poincare import planes_for, steps_for, trace_archive
    from mrx.relaxation.loop import check_checkpoint, checkpoint_attrs

    attrs = checkpoint_attrs(cli.checkpoints[0])
    fields = {}
    for path in cli.checkpoints:
        with h5py.File(path, "r") as fh:
            fields[os.path.splitext(os.path.basename(path))[0]] = (np.asarray(fh["B_n"], dtype=np.float64),
                                                                    int(fh.attrs["step"]))
    archive = cli.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cli.checkpoints[0]))),
                                      "trace.npz")
    source = f"{os.path.basename(cli.geometry)} {attrs['ns']} p={attrs['p']}, relaxed in {attrs['precision']}"
    print(f"[trace] {source}: fields {list(fields)}, tracing in {cli.precision}", flush=True)

    seq, _ = build_sequence(cli.geometry, attrs["ns"], attrs["p"], knots=attrs["knots"], symmetry=attrs["symmetry"])
    planes = planes_for(seq, cli.planes)
    print(f"[trace] nfp={seq.nfp}, symmetry {seq.symmetry}, planes {[f'{v:g}' for v in planes]} "
          f"({steps_for(planes)} steps per period)", flush=True)
    for path in cli.checkpoints:
        check_checkpoint(path, seq)
    sections, _ = trace_archive(seq, fields, lines=cli.lines, periods=cli.periods, planes=planes, seed=cli.seed,
                                source=source)
    os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
    np.savez_compressed(archive, **sections)
    print(f"  -> {archive}", flush=True)


if __name__ == "__main__":
    # The precision must be in the environment before mrx is imported. The full parse follows.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--precision", default="float32", choices=("float32", "float64"))
    os.environ["MRX_DTYPE"] = pre.parse_known_args()[0].precision
    sys.exit(main(tyro.cli(Trace, description=__doc__)))
