# Running on an Apple GPU

MRX runs on Apple Silicon GPUs through
[jax-mps](https://github.com/tillahoffmann/jax-mps), a PJRT plugin that
compiles the StableHLO a JAX program lowers to and executes it with
[MLX](https://github.com/ml-explore/mlx). Nothing is provisioned and nothing
bills: unlike `tpu/`, the hardware is the laptop.

The whole suite passes there, 59/59, in the plain float32 configuration. The
useful summary of the performance is that it is not a laptop-sized version of
the GPU story: the heavy sum-factorised applies run about 1.7x faster than
the M3 Pro's CPU at `(16,32,16)` p=3, the small indexed applies and
preconditioner atoms run 4-6x *slower* at every size measured, and at the
test resolution `(8,12,12)` p=2 the GPU loses across the board. Whether a run
is faster is a question about its mesh.

## The environment

`jax-mps` pins the jaxlib minor version it was built against, because it
deserializes StableHLO bytecode: a mismatch fails at the first JIT compile.
Version 0.10.11 is built for jaxlib 0.10.x, which is what MRX runs on, and
ships a `py3-none-macosx_14_0_arm64` wheel, so no LLVM build is needed
despite what the upstream README describes.

```bash
conda create -n mrx_mps --clone mrx
conda activate mrx_mps
pip install "jax-mps==0.10.11"
python -c "import jax; print(jax.devices())"     # [MpsDevice(id=0)]
```

Keep it in its own environment. The plugin registers itself at priority 500
against the CPU backend's 0, so merely installing it makes `mps` the default
backend for every JAX program in that environment, MRX or not. For the same
reason every CPU comparison below passes `JAX_PLATFORMS=cpu` explicitly.

Requirements are macOS 14 or later on Apple Silicon and Python 3.12 or later.
Measurements here are an M3 Pro on macOS 27.

## The configuration: `MRX_X64=0`

```bash
MRX_X64=0 JAX_PLATFORMS=mps python -m pytest test
```

Metal has no float64, and this is the way it differs from a TPU. A TPU also
has no float64, but XLA:TPU silently downcasts it, so the TPU configuration
only has to drop the residual precision (`MRX_RESIDUAL_DTYPE=float32`) and
can leave 64-bit mode on. MLX instead *refuses* the buffer:

```
MLX does not support float64 (F64). Use
jax.config.update('jax_enable_x64', False) or ensure your arrays are float32.
```

Under 64-bit mode a dtype-less constructor like `jnp.zeros(n)` or
`jnp.linspace(0, 1, n)` makes float64, and `mrx/` has dozens of them; on a
TPU they cost nothing and on Metal any one of them aborts the run. So
`MRX_X64=0` turns 64-bit mode off at the root rather than chasing the
constructors, which makes every one of them float32 and leaves nothing that
the backend can refuse.

It implies the rest of the float32-only configuration and refuses to be
combined with anything else: `MRX_DTYPE=float64` and
`MRX_RESIDUAL_DTYPE=float64` both raise at import, rather than failing at the
first buffer transfer several minutes into a run. What is left is the plain
float32 configuration that the TPU work already validated -- `REFINE` off,
`SOLVE_TOL = sqrt(eps) = 3.5e-4` -- so the tolerances are the measured ones
and not new. See {doc}`concepts/precision`, and read the accuracy caveats
there before trusting a number: plain float32 is the configuration for runs
that stop near a residual of 5e-4.

`MRX_X64` is read from the environment rather than detected from the backend
because `jax.devices()` would initialise the backend before
`jax_enable_x64` could be set.

## Checking the backend before blaming MRX

```bash
python mps/probe_ops.py
```

One small program per JAX construct MRX depends on, run on both backends with
the same inputs and compared: the sum-factorised `einsum` of `mrx.mass`, the
gather + `segment_sum` of `mrx.extraction_operators` (including out-of-bounds
segment indices, which must be dropped), the data-dependent `while_loop` of
the Krylov solvers, the stacking `lax.scan` of the relaxation, the spline
`dynamic_slice`, and the `cholesky`/`solve_triangular`/`eigh` chain of the
preconditioner atoms. All 14 match the CPU to float32 rounding on
jax-mps 0.10.11.

This exists because the alternative diagnosis path is bad: a missing
StableHLO handler surfaces as a compile error thrown from inside a
hundred-line fused kernel, and a wrong one surfaces as a plausible number.
Run it first after any upgrade of `jax-mps`, `jax` or macOS.

## What was measured

Suite, `MRX_X64=0`, one M3 Pro, wall clock for `pytest test`:

| backend | configuration | result |
|---|---|---|
| mps | plain float32 | 59 passed, 189 s |
| cpu | plain float32 | 59 passed, 209 s |
| cpu | mixed float32/float64 (the default, x64 on) | 59 passed, 273 s |

The suite is compile-bound, so that near-parity is mostly a statement about
XLA-versus-MLX compile time, not about execution. The parts of it that are
execution-bound disagree with each other: `test_poisson.py` alone is 24.8 s on
the GPU against 45.3 s on the CPU, while `test_relaxation.py`'s longest case
is 54.0 s against 37.4 s. The Poisson tests are Krylov solves over
sum-factorised applies, which the GPU wins; the relaxation is a jitted
`lax.scan`, whose per-iteration cost on this plugin grows with the trip count
(jax-mps [#215](https://github.com/tillahoffmann/jax-mps/issues/215)).

Per-apply cost, `scripts/benchmark/matvec_bench.py`, the `scan` column
(the form the relaxation actually runs), ms per apply:

| operator | `(8,12,12)` p=2 cpu / mps | `(16,32,16)` p=3 cpu / mps |
|---|---|---|
| `apply_mass_matrix` k=0 | 0.18 / 0.44 | 1.01 / 0.58 |
| `apply_mass_matrix` k=1 | 0.43 / 1.09 | 2.51 / 1.60 |
| `apply_stiffness` k=0 | 0.43 / 1.16 | 2.77 / 1.58 |
| `apply_stiffness` k=1 | 0.32 / 1.09 | 2.54 / 1.50 |
| `apply_derivative D^T D` k=0 | 0.85 / 2.05 | 5.51 / 3.13 |
| `apply_derivative D^T D` k=2 | 0.12 / 0.73 | 1.90 / 0.86 |
| `E E^T` k=0 | 0.006 / 0.083 | 0.024 / 0.094 |
| mass atom k=0 | 0.004 / 0.135 | 0.022 / 0.134 |

Two things are visible. The heavy applies cross over between the two meshes
and settle at about 1.7x in the GPU's favour at the larger one. The light
ones never cross over, because MPS has a per-kernel floor of roughly 0.08 ms
that the CPU does not: `E E^T` and the mass atom cost the same on the GPU at
both resolutions, which is the signature of a cost that is dispatch and not
arithmetic. Those light applies are the preconditioner, so a Krylov iteration
pays the floor once per iteration no matter how large the mesh is.

The implication for a real run is that the mesh has to be large enough that
the sum-factorised applies dominate the preconditioner, and `(8,12,12)` p=2
is far below that. Do not port a laptop-sized test case to the GPU and expect
a speed-up.

## `JAX_MPS_ASYNC_DISPATCH=1`

The plugin suggests this on startup and it is worth knowing what it does and
does not do. It roughly halves the cost of *eager* applies -- `apply_stiffness`
k=0 goes from 6.46 ms to 2.84 ms, `D^T D` k=1 from 12.55 ms to 5.54 ms at
`(16,32,16)` p=3 -- and leaves the `jit` and `scan` columns unchanged within
noise. Production runs the relaxation step as one jitted `lax.scan`, so the
flag buys nothing there; interactive and script use, which dispatches applies
from Python one at a time, roughly doubles in speed. It halved the wall clock
of the benchmark itself.

It is documented as experimental. Nothing in the suite was run with it.

## Known limits

- Plain float32 only. There is no iterative refinement on this backend, so
  the accuracy discussion in {doc}`concepts/precision` about what float32
  alone does applies in full.
- `lax.scan` per-iteration cost grows with the trip count
  ([#215](https://github.com/tillahoffmann/jax-mps/issues/215)), which is the
  relaxation's hot loop and the one place the GPU loses to the CPU on a test
  that is not compile-bound.
- Single device. The plugin exposes one `MpsDevice` and no collectives, so
  `scripts/pmap_sweep.py` and the sharded paths have nothing to spread over.
- The plugin is experimental and prints so on import. It pins a jaxlib minor
  version; upgrading `jax` in `mrx_mps` without upgrading `jax-mps` to match
  breaks bytecode deserialization at the first compile.
- `eigh`, `qr` and `svd` are executed on the CPU inside MLX rather than on
  the GPU. MRX only reaches them at preconditioner build time, so this costs
  a transfer once per geometry.
