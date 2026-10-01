# Getting started

## Install

MRX needs Python 3.11 or newer. Clone the repository and install it in
editable mode:

```bash
git clone https://github.com/ToBlick/mrx.git
cd mrx
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

On a GPU machine, install the CUDA build of JAX as well:

```bash
pip install "jax[cuda12]"
```

Without it, JAX runs on the CPU.

MRX runs in mixed precision by default: float32 fields and solves,
refined against a float64 residual. `MRX_DTYPE=float64` or plain float32
(`MRX_RESIDUAL_DTYPE=float32`) is chosen before `mrx` is imported (see
[Precision](concepts/precision.md)).

## Run the tests

```bash
pytest
```

The suite reads only files tracked in the repository. On a cluster every
run is a GPU job through `slurm/run.sh` (see
[Running on a cluster](cluster.md) and [Testing strategy](concepts/testing_strategy.md)).

## Data files

Every geometry is passed by path: `--geometry.path /path/to/file` to
`scripts/relax.py`, `--geometry /path/to/file` to the other scripts, `build_sequence("/path/to/file", ns, p)` in code. A GVEC state
(`GVEC_State_*.dat`), a VMEC `wout_*.nc` or a DESC output (`*.h5`) gives the map and the initial
field ([Equilibrium input](concepts/equilibrium_input.md)). An analytic
shape is used by writing a mock equilibrium file from its formulas, for
example a VMEC wout or a GVEC state, as `test/synthetic_gvec.py` and
`test/synthetic_desc.py` do. The repository ships
`data/wout_li383_low_res_reference.nc`, `data/wout_li383_1.4m.nc`,
`data/wout_LandremanPaul2021_QA_lowres.nc`, the same QA device as a DESC output
`data/desc_LandremanPaul2021_QA.h5`, and the li383 DESC file
`data/desc_li383_low_res_reference.h5`.

## Next steps

- [Tutorials](tutorials.md) go from a file to a relaxed, reconnected field in five scripts, and a sixth optimizes the shape.
- [Solve a relaxation problem](relaxation.md) runs `scripts/relax.py`.
- [Architecture](concepts/architecture.md) orients you in the code.
