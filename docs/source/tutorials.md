# Tutorials

The scripts in `scripts/tutorials/` introduce MRX one concept at a time. Each
script explains itself in its docstring and its comments.

| script | what it shows | geometry |
|---|---|---|
| `1_qa_geometry.py` | reading a file, building the sequence | QA |
| `2_qa_vacuum_field.py` | the vacuum field as a harmonic 2-form, Poincare sections | QA |
| `3_li383_newton.py` | Newton relaxation from the file's own field | li383 |
| `4_li383_island_seed.py` | island seeds by the energy criterion | li383 |
| `5_li383_drive.py` | the resistive drive towards a reference current | li383 |
| `6_qa_shape_optimization.py` | shape optimization for quasi-axisymmetry by automatic differentiation | QA |

QA is `data/wout_LandremanPaul2021_QA_lowres.nc`, the quasi-axisymmetric
vacuum equilibrium of Landreman & Paul (2021). li383 is
`data/wout_li383_low_res_reference.nc`, the NCSX configuration. All
tutorials run on the mesh (12, 16, 16) at spline degree 2 in the default
mixed precision, except Tutorial 6 (degree 3, float64). They take, in
minutes and including about a minute of setup:

| tutorial | 1 | 2 | 3 | 4 | 5 | 6 | all |
|---|---|---|---|---|---|---|---|
| one H100 | 1.5 | 4 | 6 | 7.5 | 7.5 | 3 | 30 |
| 8 CPU cores (Xeon 8470QL) | 2 | 2 | 11.5 | 6 | 7 | 13 | 42 |

## Running them

Each tutorial runs as a script or cell by cell in a notebook, where it
takes its defaults:

```bash
python -u scripts/tutorials/1_qa_geometry.py
python -u scripts/tutorials/1_qa_geometry.py --geometry.path data/desc_LandremanPaul2021_QA.h5 --geometry.resolution 12 24 24
SCRIPT=scripts/tutorials/3_li383_newton.py JOB_NAME=li383_newton bash slurm/run.sh
```

The command line is built with `tyro`, and its `--geometry` and `--budget`
groups are those of `scripts/relax.py`. `--help` lists every option.

Every tutorial writes to `outputs/tutorials/<script name>/`. Tutorials 4 and
5 start from the run of the previous tutorial there when it has the same
mesh, and build what is missing themselves otherwise. The Poincare sections
of Tutorials 2, 4 and 5 are written as `trace.npz` and drawn into
`poincare/` as `scripts/poincare_plot.py` draws them, so
`python scripts/poincare_plot.py outputs/tutorials/<script name>` redraws
them with other options.
