# Getting started

## Install

MRX needs Python 3.11 or newer. Clone the repository and install it in
editable mode with the cluster extras:

```bash
git clone https://github.com/ToBlick/mrx.git
cd mrx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[SLURM]"
```

The `SLURM` extra adds Hydra, its submitit launcher, and HDF5 support. On a
GPU machine, install the CUDA build of JAX as well:

```bash
pip install "jax[cuda12]"
```

Without it, JAX runs on the CPU.

## Precision

MRX runs in one floating-point precision per process. Set `MRX_DTYPE` to
`float64` (the default) or `float32` before importing `mrx`:

```bash
MRX_DTYPE=float32 python my_script.py
```

`mrx.DTYPE` is the working dtype, `mrx.EPS` its machine epsilon. Every
tolerance that depends on roundoff is written as `mrx.eps(c)` or
`mrx.sqrt_eps(c)`. See [Precision](concepts/precision.md).

## Run the tests

Locally, on the CPU (a few minutes on four cores, in either precision):

```bash
pytest
```

Tests that read files outside the repository are marked `needs_data` and
skip when the file is absent. On a cluster, every run is a GPU job. Submit
the suite through `slurm/run.sh`:

```bash
SCRIPT="-m pytest -q test" JOB_NAME=tests TIMEOUT_MIN=45 bash slurm/run.sh
```

The first line of every log is `mrx from: <path>`. It names the checkout
that was imported. Read it before you read the result: an editable install
pinned to another checkout shadows the one you meant to test. See
[Running on a cluster](cluster.md).

## Data files

Geometry files are passed by path: `--geometry /path/to/w7x_fmm002_clebsch_mrx.h5`
to the scripts, `build_sequence("/path/to/file.h5", ns, p)` in code. Both take
the flat-schema GVEC export (`.h5`) or GVEC's own state file
(`GVEC_State_*.dat`), read in closed form -- see the
[GVEC interface](concepts/gvec_mrx_interface.md). Only the gpu-tier
data test reads a directory name, `MRX_DATA` (default `data/`).

## Next steps

- [Solve a Poisson problem](poisson.md) builds a sequence, installs a map,
  and solves the Hodge Laplacians.
- [Solve a relaxation problem](relaxation.md) runs `scripts/relax.py`.
- [Concepts](concepts.md) orients you in the code.
