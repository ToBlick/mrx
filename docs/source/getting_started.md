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

## Precision

The working precision is `float32` by default, with every solve refined
against a float64 residual; `MRX_DTYPE=float64` before importing `mrx`
switches the working precision. See [Precision](concepts/precision.md).

## Run the tests

Locally, on the CPU (a few minutes on four cores):

```bash
pytest
```

Every file the suite reads is in the repository. On a cluster, every run
is a GPU job through `slurm/run.sh`; see
[Running on a cluster](cluster.md).

## Data files

Equilibrium files are passed by path: `--geometry /path/to/GVEC_State_final.dat`
to the scripts, `build_sequence("/path/to/file.dat", ns, p)` in code. Both take
GVEC's own state file (`GVEC_State_*.dat`) or a VMEC `wout_*.nc`, read in
closed form -- see the [GVEC interface](concepts/gvec_mrx_interface.md).
The suite runs on the tracked `data/wout_li383_low_res_reference.nc`.

## Next steps

- [Tutorials](tutorials.md) go from a file to a relaxed field in five scripts.
- [Solve a relaxation problem](relaxation.md) runs `scripts/relax.py`.
- [Architecture](concepts/architecture.md) orients you in the code.
