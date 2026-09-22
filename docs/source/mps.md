# Running on an Apple GPU

MRX runs on Apple Silicon GPUs through
[jax-mps](https://github.com/tillahoffmann/jax-mps), a PJRT plugin that
compiles the StableHLO a JAX program lowers to and executes it with
[MLX](https://github.com/ml-explore/mlx). Nothing is provisioned and nothing
bills: unlike `tpu/`, the hardware is the laptop.

The whole suite passes there, 61/61, in the plain float32 configuration, and
a real li383 relaxation runs to completion. This backend charges a flat
~0.22 ms per dispatch, so an operation is faster or slower than the CPU
according to whether it is bigger or smaller than that floor. The mass
kernel used to be on the wrong side of it, because it had been written as a
dozen dense shifts in place of one indexed read: that is the form a TPU
wants and the form Metal does not. On Metal the kernel is one gather and
one ``segment_sum`` instead (``MRX_ASSEMBLY``, below). With that, 100 L-BFGS
steps at `(12,24,12)` p=3 take **283.8 s on the GPU against 352.5 s on the
CPU**. The numbers are under "What was measured" below, and
`docs/research/mps_benchmark_2026-09-21.md` is the full record.

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
source mps/env.sh          # MRX_X64=0, JAX_PLATFORMS=mps, and the knob record
python -m pytest test
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
| mps | plain float32, indexed assembly | 61 passed, 150 s |
| cpu | mixed float32/float64, shift assembly (the default) | 61 passed, 300 s |
| cpu | plain float32, before the assembly tests existed | 59 passed, 209 s |

The suite is compile-bound, so that near-parity is mostly a statement about
XLA-versus-MLX compile time, not about execution. The parts of it that are
execution-bound disagree with each other: `test_poisson.py` alone is 24.8 s on
the GPU against 45.3 s on the CPU, while `test_relaxation.py`'s longest case
is 54.0 s against 37.4 s. The section below is why.

### A real run, li383 `(12,24,12)` p=3 float32

These are `--method lbfgs`. The default method is Newton, and it behaves
differently; that run is the next section. `scripts/relax.py --method lbfgs
--steps 100 --chunk 25`, the same mesh as the v5e / H200 / H100 table in
`docs/research/tpu_v5e_benchmark.md`:

| | CPU | MPS, indexed assembly | |
|---|---|---|---|
| 100 L-BFGS steps, end to end | 352.5 s | **283.8 s** | MPS 1.24x |
| per step | 3.52 s | **2.84 s** | MPS 1.24x |
| the same run before the assembly change | 353.7 s | 593.9 s | CPU 1.68x |
| one nested-CG Laplacian solve, k=1 | 284.3 ms | **74.9 ms** | MPS 3.8x |
| inverse mass CG k=1, 20 iterations | 587 ms | **158 ms** | MPS 3.7x |

The mass apply inside a scan, the change itself, at this mesh: 1.226 ms the
shift form against **0.503 ms** the indexed form at k=1 (2.44x), and 1.200
against **0.475 ms** at k=2 (2.52x). On the CPU the same comparison is 1.07x
and 0.94x, inside the noise, which is why the CPU keeps the shifts.

Run-to-run spread on this machine is about 17%, so nothing is claimed on a
smaller margin than that.

`--floor-tol` defaults to `1e-8`, a squared normalised residual. A float32
run sits at a few times `1e-3`, so the criterion cannot fire and the run
stops on the step count. Pass `--floor-tol 0`.

### Newton, the default method

A Newton step is a 300-iteration MINRES solve whose matvec is three k=1 mass
solves, so it is more mass-bound than the descent, and `newton_tol=0.1` is
not reached: every step spends the whole budget. That tolerance is the
truncation, not a target, and `newton_it` reads `+300`.

On the benchmark mesh, two steps from the equilibrium field, the indexed
matvec and the shift matvec agree. The force at step 1 agrees to 8e-7 and
the descent cosine to 3e-7, and both fall back to the smoothed force on that
one step and not the next:

| | CPU, shift | MPS, shift | MPS, indexed |
|---|---|---|---|
| 2 Newton steps | 101.3 s | 120.9 s | **86.7 s** |
| per step | 50.6 s | 60.5 s | **43.4 s** |
| fallbacks | 1/2 | 1/2 | 1/2 |

That indexed column is not what a Newton run does. On Tutorial 4's mesh,
`(10,16,16)` p=2, ten steps from the shipped descent state, the same indexed
matvec changed the direction: the descent cosine collapsed to 0.007 at step
3 where the shift assembly held 0.13, and 3 of 10 steps fell back to the
smoothed force against none. Three hundred iterations of a matvec accurate
to `sqrt(eps)` is enough room for a 1e-7 reordering of the sum to matter, and
it mattered there. So `newton_direction` pins its own matvec to the shift
assembly, whatever the backend. The force evaluation around it stays indexed.
With that pin the tutorial run has no fallbacks, and it costs the shift
step, 39 s, not the indexed 31 s.

| Tutorial 4, 10 steps | per step | fallbacks | cosine, minimum |
|---|---|---|---|
| CPU, shift | 21.8 s | 0/10 | 0.20 |
| MPS, shift | 38.9 s | 0/10 | 0.13 |
| MPS, indexed, before the pin | 31.3 s | 3/10 | 0.007 |

At `(10,16,16)` p=2 the GPU loses to the CPU either way. That is the small
mesh, where it has lost before.

### The floor, measured directly

Steady ms per apply at `(12,24,12)` p=3:

| primitive | CPU | MPS |
|---|---|---|
| `E` apply k=0 / k=1 / k=2 | 0.040 / 0.032 / 0.037 | 0.246 / 0.223 / 0.216 |
| `E^T` apply k=0 / k=1 / k=2 | 0.054 / 0.053 / 0.052 | 0.248 / 0.240 / 0.231 |
| `mass_core_apply` k=1 | 1.269 | 1.868 |
| `apply_derivative_matrix` k=1 | 1.779 | 9.172 |
| `apply_laplacian` k=1 free (nested CG) | 284.257 | **74.916** |

Read the MPS column across the first two rows: six different operations over
three form degrees and two array sizes, all between 0.216 and 0.248 ms. A cost
that ignores what it is computing is dispatch, not arithmetic. The CPU does
the same work in 0.03-0.05 ms, so those applies lose 4-7x; the nested CG is
large enough to bury the floor and wins 3.8x.

A compiled relaxation chunk is ~21,000 StableHLO ops of which only 1,271 are
`dot_general` -- the rest is slice, gather, reshape and concatenate. 94% data
movement against a fixed per-dispatch charge is the entire result.

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

## `MRX_ASSEMBLY`

```bash
MRX_ASSEMBLY=indexed   # one gather and one segment_sum per element
MRX_ASSEMBLY=shift     # the dense shifted copies, the TPU and CPU form
```

Unset, Metal takes `indexed` and every other backend takes `shift`, from
`jax.default_backend()`. The two compute the same operator: the gather is
the same integer map and agrees exactly, and the assembly is the same sum in
a different order and agrees to float32 roundoff. `shift` traces the
kernel every non-Metal run has always traced, so nothing about a CPU or GPU
result moves.

It is a static branch of the one mass kernel, not a second code path through
the solvers. One L-BFGS step applies that kernel 2,757 times at k=1 and
1,228 times at k=2, which is why a 2.5x on the kernel is a 2.1x on the step
(593.9 s down to 283.8 s) and is what puts the GPU ahead of the CPU on that
method. The Newton matvec does not take the branch: see above.

## Pass `--chunk 25`

Compile is still uncacheable, so this is still worth about 2.3x, on top of
the assembly change above.

The persistent compilation cache does not work on this plugin: it writes zero
entries and a second run recompiles in full, where the same script on the CPU
backend writes 1118 entries and recompiles 2.7x faster. So every MPS run pays
its compile. Meanwhile compile cost is *linear* in the chunk (about 7.9 s per
step of chunk at `(12,24,12)` p=3) while the steady per-step cost is *flat* in
it -- 5506.8 against 5504.8 ms/step at chunks of 5 and 10. A long chunk
therefore buys nothing and costs a great deal.

`scripts/relax.py` defaults to chunk 500 under L-BFGS. For a 500-step run:

| | compile | stepping | total |
|---|---|---|---|
| `--chunk 500` | ~65 min | ~46 min | ~111 min |
| `--chunk 25` | ~3 min | ~46 min | ~49 min |

Every other knob was swept and none of them helped; `mps/env.sh` records which
and by how much, so the sweep does not need repeating. `mps/env_sweep.py` runs
it again if a version bump makes that worth doing.

## Trusting a GPU run

Plain float32 relaxation is sensitive to roundoff, and the two backends
separate. Over 100 L-BFGS steps on li383 `(12,24,12)` p=3, the force residual
agrees to 2e-6 at step 1 -- roundoff carried through solves whose own
tolerance is `sqrt(eps) = 3.5e-4` -- then grows past 1% by step 12, and by
step 100 the runs are on different trajectories. The indexed and shifted
forms on the GPU itself agree to 4e-6 at step 1 and then separate the same
way, which is the check that the faster kernel is the same operator.

That is the descent amplifying a roundoff-sized perturbation, not a backend
defect, and the conserved quantities say both runs are physical: helicity
drifts by 3.3e-5 relative on the GPU and 7.3e-5 on the CPU, `||div B||`
stays at 1.5e-06 and 1.2e-06, and the energy decreases on 91 and 81 of 100
steps. Do not read the helicity figures as the GPU being more accurate; they
are two samples of a chaotic trajectory. Compare backends on invariants and
on converged states, never step by step.

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
- The persistent compilation cache is inert here: every run recompiles. Keep
  the chunk small, as above.
- A per-dispatch floor of about 0.22 ms. It is what the indexed mass
  assembly is avoiding, and it is still what the small extraction applies
  pay. It is not the `lax.scan` trip-count growth of
  jax-mps [#215](https://github.com/tillahoffmann/jax-mps/issues/215); that
  was tested with `mps/scan_scaling.py` and does not reach MRX, on either
  backend, at either mesh.
- `MLX_METAL_FAST_SYNCH` aborts with `[metal::Device] Unable to load kernel
  input_coherent` on 0.10.11. `MLX_ENABLE_TF32` must never be set: it undoes
  the `jax_default_matmul_precision='highest'` that
  {doc}`concepts/precision` relies on.
- Single device. The plugin exposes one `MpsDevice` and no collectives, so
  `scripts/pmap_sweep.py` and the sharded paths have nothing to spread over.
- The plugin is experimental and prints so on import. It pins a jaxlib minor
  version; upgrading `jax` in `mrx_mps` without upgrading `jax-mps` to match
  breaks bytecode deserialization at the first compile.
- `eigh`, `qr` and `svd` are executed on the CPU inside MLX rather than on
  the GPU. MRX only reaches them at preconditioner build time, so this costs
  a transfer once per geometry.
