r"""Trace the field lines of relaxation checkpoints and save their Poincare sections. This is the expensive half.

Every checkpoint named on the command line (``checkpoints/state_<step>.h5``, ``best.h5``, a seeded checkpoint and
so on) is traced by ``mrx.diagnostics.poincare.trace_archive``, all of them in one batch. The sequence is rebuilt
from the resolution, degree, knots, symmetry, grading and theta cells of the map stored in the checkpoint
(:func:`mrx.relaxation.loop.checkpoint_attrs`) on the geometry file ``--geometry``, which is the run's VMEC wout,
GVEC state or DESC output. The weak pressure of each field
(``mrx.relaxation.physics.weak_pressure``, zero on the wall) is evaluated at every crossing.

The archive ``trace.npz`` goes into the run directory of the first checkpoint, the parent of its ``checkpoints/``
folder. ``scripts/poincare_plot.py`` draws it and is cheap enough for the login node, so changing the figure never
repeats the trace. All checkpoints of one call go into ONE archive, so the plotter can put them on one iota scale
and one pressure scale.

    python -u scripts/poincare_trace.py --geometry data/wout_li383_1.4m.nc <run>/checkpoints/state_*.h5

The archive (numpy .npz) holds ``fields`` (the checkpoint names, in order), ``planes``, ``resolution``, ``p``,
``nfp``, ``symmetry``, ``steps`` (per period), ``source``, ``sections`` (``zeta`` or ``phi``, see ``--sections``)
and ``section_labels`` (the names of the section coordinates stored as ``R``, ``Z``: ``R``, ``Z``, or ``X1``, ``X2``
of a G-frame). Per field ``<f>`` it holds
``<f>_label``, ``<f>_step``, ``<f>_bsq`` (the volume mean of |B|^2, which the plotter uses to normalise the
pressure), ``<f>_iota``, ``<f>_seed_r``, ``<f>_keep``, ``<f>_chaotic``, ``<f>_shown``, ``<f>_round`` (the seeding
round, see ``--reseed``), ``<f>_escape_period`` (``inf`` for a line that stayed inside) and ``<f>_drift``, and with
``--padding-lines`` the same entries for the padding lines under ``<f>_pad_...``. Per field and plane it holds
``<f>_zeta<plane>_{R,Z,axisR,axisZ,logr,logth,logz}`` (with ``--sections phi`` ``zeta<plane>`` names the plane
``phi``, and the crossings of a line after it left the domain are NaN) and ``<f>_zeta<plane>_pressure``, the weak
pressure at every crossing. The archive holds trace results only. Every choice about the drawing is left to the
plotter.

The integrator is compiled once per mesh and step schedule. The weak pressure costs two extra solves per field.
"""
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
    """The number of planes spread over the part of the period the symmetry leaves distinct, or the planes themselves as fractions of a period (for example 0 0.25 0.5)."""
    sections: Literal["zeta", "phi"] = "zeta"
    """The sections: the map's ``zeta = const`` (free, the coordinates of the map's sections), or the planes ``phi = const`` of the physical toroidal angle in cylindrical ``(R, Z)``, which differ from the map's on a G-frame state and cost a map evaluation per step there."""
    seed: int = 0
    """The random seed for the poloidal start angles."""
    min_steps: int = 300
    """The fewest integration steps per field period. The planes can raise it, and the archive's drift says whether the steps resolve the field."""
    padding_lines: int = 0
    """Further lines seeded uniformly over ``--padding-r-min < r < 0.97`` (the edge) and traced both ways for ``--padding-periods`` periods, archived under ``<f>_pad_...``: dense coverage of a region whose lines leave within a few periods."""
    padding_periods: int = 60
    """Periods traced for the padding lines."""
    padding_r_min: float = 0.7
    """Inner logical radius of the padding lines."""
    reseed: int = 0
    """Further rounds of ``lines`` lines seeded on the radii of the lines the previous round lost through the boundary, so that the region of open field lines is populated up to the boundary. A lost line is cut at the boundary and its crossings before that are archived."""
    out: Optional[str] = None
    """The archive path. Unset, it is trace.npz in the parent of the first checkpoint's directory."""

    def __post_init__(self):
        if not self.checkpoints:
            raise ValueError("name at least one checkpoint to trace")
        names = field_names(self.checkpoints)
        if len(set(names)) != len(names):
            raise ValueError(f"the checkpoints must have distinct names, got {names}")


def field_names(paths):
    """The archive names of the fields in the checkpoints ``paths``: the file names without the extension, those
    that occur more than once (``best.h5`` of two runs) prefixed by their run directory, the parent of
    ``checkpoints/``."""
    stems = [os.path.splitext(os.path.basename(path))[0] for path in paths]

    def run(path):
        d = os.path.dirname(os.path.abspath(path))
        return os.path.basename(os.path.dirname(d) if os.path.basename(d) == "checkpoints" else d)
    return [f"{run(path)}_{stem}" if stems.count(stem) > 1 else stem for path, stem in zip(paths, stems)]


def main(cli):
    from mrx.geometry import build_sequence
    from mrx.diagnostics.poincare import planes_for, steps_for, trace_archive
    from mrx.relaxation.loop import checkpoint_attrs, read_field

    attrs = checkpoint_attrs(cli.checkpoints[0])
    archive = cli.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cli.checkpoints[0]))),
                                      "trace.npz")
    source = f"{os.path.basename(cli.geometry)} {attrs['ns']} p={attrs['p']}"
    seq = build_sequence(cli.geometry, attrs["ns"], attrs["p"], knots=attrs["knots"],
                         symmetry=attrs["symmetry"], graded=attrs["graded"], map_theta=attrs["map_theta"])
    fields = {name: read_field(path, seq) for name, path in zip(field_names(cli.checkpoints), cli.checkpoints)}
    planes = planes_for(seq, cli.planes)
    print(f"[trace] {source}: fields {list(fields)}, nfp={seq.nfp}, symmetry {seq.symmetry}, planes "
          f"{[f'{v:g}' for v in planes]} ({steps_for(planes, cli.min_steps)} steps per period), sections "
          f"{cli.sections}", flush=True)
    sections, _ = trace_archive(seq, fields, lines=cli.lines, periods=cli.periods, planes=planes, seed=cli.seed,
                                source=source, reseed=cli.reseed,
                                padding=((cli.padding_lines, cli.padding_periods, cli.padding_r_min)
                                         if cli.padding_lines else None),
                                sections=cli.sections, min_steps=cli.min_steps)
    os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
    np.savez_compressed(archive, **sections)
    print(f"  -> {archive}", flush=True)


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Trace, description=__doc__)))
