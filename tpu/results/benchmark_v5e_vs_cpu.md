# MRX on a v5e: what was slow, and why

Measured on `mrx-scaling-on-tpus`, 1-2 September 2026, one `v5litepod-4` node
in `us-west4-a`, JAX 0.11.1, li383 at `--ns 12,24,12 --p 3`, float32,
`jax_default_matmul_precision=highest`.

The control column is **the host CPU of the same VM**, run through the same
driver with `RUN_PLATFORM=cpu`, so the software stack, geometry and precision
are identical and only the backend differs. Both columns run the same fixed
code, so the comparison is not confounded by the fixes below. Reproduce with
`tpu_bench_mrx.py`.

## The headline

| Measurement | v5e before | v5e after | Same VM's CPU | H100 (docs) |
|---|---|---|---|---|
| `build_sequence` (warm cache) | 203 s | **35.5 s** | 40.6 s | - |
| `compute_nullspaces`, `gap_sweeps=0` | 222 s | **17.5 s** | 55.1 s | - |
| `mass_core_apply` k=1 | 6.60 ms | **0.505 ms** | 3.79 ms | - |
| `mass_core_apply` k=2 | 5.00 ms | **0.459 ms** | 4.86 ms | - |
| `E` apply k=1 | 1.999 ms | **0.181 ms** | 0.102 ms | - |
| `apply_derivative_matrix` k=1 | 7.15 ms | **2.74 ms** | 6.28 ms | - |
| `apply_laplacian` k=1 (nested CG) | 10 020 ms | **76.4 ms** | 108 ms | - |
| relaxation, per step | 100.3 s | **13.03 s** | 45.7 s | 0.41 s |

The v5e went from 1.7x slower than the VM's own CPU to **3.5x faster**, and
setup from about 7 minutes to under a minute. It is now about 32x behind one
H100 on a single equilibrium, down from 105x.

Four things did that, in order of size: the persistent compilation cache, the
gather, the assembly, and compiling the extraction operator's two ops
together. The first is configuration; the rest are in `mrx/mass.py` and
`mrx/extraction_operators.py`.

## Fix 1: XLA was recompiling, not the device computing

MRX's inner solves run as eager `jax.lax.while_loop`s. Nothing wraps them in
`jax.jit`, so each call traces a fresh closure and XLA compiles a program it
has never seen. On a v5e, one `apply_laplacian` k=1 call cost about 10 s of
compilation to perform about 20 ms of arithmetic.

The signature was repeated identical calls that never got faster:

```
apply_laplacian k=1, no compilation cache:   first 9.963 s   second 9.854 s
apply_laplacian k=1, compilation cache on:   first 10.173 s  second 0.105 s
```

Wall-clock phase timings cannot distinguish compiling from computing, which is
why an earlier version of this analysis concluded the hardware was simply
unsuited to the workload.

**Fix:** the persistent compilation cache, now set by `run_on_tpu.sh`.

```bash
JAX_COMPILATION_CACHE_DIR=/mnt/data/jax_cache
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1
```

The thresholds matter as much as the directory: their defaults are tuned for a
few large training programs and skip almost every kernel this workload
compiles.

## Fix 2 and 3: the mass kernel holds no indices at all

`_sumfact_kernel` began with a gather, `x[gather_idx]`, and ended with a
`jax.ops.segment_sum`. Indexed access is the one thing a TPU has no fast path
for. But both index plans come from `_flat_dof_plan`, a separable tensor
product of per-axis DoF ids, and on a tensor-product B-spline axis
`g[e, l] = (e + l) mod n`. So both are algebraically pure data movement with
every source and destination known at compile time: rolled slices one way,
shifted dense adds the other.

| at (12,24,12) p=3 | v5e indexed | v5e structured | CPU indexed | CPU structured |
|---|---|---|---|---|
| gather, 3456 dofs read 221k times | 1.624 ms | **0.049 ms** | 0.070 ms | 0.113 ms |
| scatter, 221k contributions | 2.011 ms | **0.060 ms** | 0.398 ms | 0.303 ms |

That table is the whole story of this hardware. On the indexed forms the v5e
is 23x and 5x *slower* than the VM's CPU; on the structured forms it is 2.8x
and 5x *faster*. The work is identical and the answers agree exactly for the
gather and to 3.4e-07 in float32 for the scatter. Only the access pattern
changed.

Attribution said which half to do first. Of `mass_core_apply` k=1 at 3.196 ms,
the gather was 0.854 ms per component against 0.100 ms for the forward
einsums, 0.076 ms for the reverse and 0.050 ms for the assembly: roughly 80%
of the kernel spent reading 3456 numbers.

`mrx/mass.py` checks the shift structure per axis and falls back to the index
tensors if any axis fails, so a differently built basis still assembles
correctly. When the plan holds the index tensors are removed from the kernel
signature rather than left unused, which is also what silenced the
`Constant folding an instruction is taking > 1s` warnings: with no index
tensor there is nothing to fold.

## Fix 4: the extraction operator was two eager ops

`MatrixFreeExtraction._apply` dispatched a gather and a `segment_sum`
separately. `E` and `E^T` measured a flat 1.93 ms whether there were 4584 or
11484 non-zeros, against a 0.037 ms floor for dispatching one device call, so
the cost was per indexed op rather than per element. Compiling the pair as one
program took it to 0.16-0.20 ms, a 10x, and it is a five-line change.

## The compilation cache on a fresh node

A node without a warm data disk pays full compilation. GCS can carry the cache
between nodes, and it works, but not as well as a local disk:

| setup = `build_sequence` + `compute_nullspaces` | v5e |
|---|---|
| no cache | 143 s |
| `gs://` cache, cold (writing 177 entries) | 210 s |
| `gs://` cache, warm | 98 s |
| local data disk, warm | **53 s** |

So a warm bucket is worth 1.5x over nothing and a warm disk is worth 2.7x. Use
the data disk when the zone has one; the bucket is the fallback for a node
that does not.

Two traps, both of which cost time here:

- **JAX does not fail on a `gs://` cache path it cannot use.** Without
  `etils[epath,epath-gcs]` installed it writes nothing, reads nothing and says
  nothing; the only symptom is that the run is slow. The first measurement
  above was taken this way and looked like a working cache with no benefit.
  `startup.sh` now installs it, and `gcs_cache_smoke.py` proves a path in
  about ten seconds before a real run depends on it.
- **Put the bucket in the node's region.** Every miss is a round trip.

## Precision

float32 on the MXU does not degrade the solves. Inverse-mass CG at the same
tolerance took **the same iteration count** on both backends:

| | v5e float32 | CPU float32 |
|---|---|---|
| inverse mass CG k=1 | 20 iterations | 20 iterations |
| inverse mass CG k=2 | 24 iterations | 24 iterations |

## Fixes that sounded right and were not

Measured, not argued about:

| Idea | Result on v5e | Verdict |
|---|---|---|
| `indices_are_sorted=True` on the scatter | 0.615 ms vs 0.533 ms unsorted | **slower**; and sorting properly, i.e. permuting the contributions each apply, is 0.873 ms |
| Dense matmul instead of the extraction operator | 0.408 ms vs 0.533 ms, 303 MB resident | 1.3x is not worth the memory |
| Per-call dispatch overhead is the problem | one device call is 0.037 ms; 20 scatters fused into one `jit` cost 0.531 ms each vs 0.533 ms unfused | it is real device work, not launch overhead |

The sorted-indices hint is genuinely useful on GPUs, where a scatter lowers to
hardware atomics. It does nothing on this hardware.

## What is left, and what would actually help

`relaxation_loop` is already `@jax.jit` around a `lax.scan`, so the 13 s per
step contains no compilation. What remains is that one li383 step is a long
chain of small dependent kernels: the largest single apply now takes half a
millisecond on arrays of about 10^4 floats, which is a rounding error of a
v5e's width.

Two things follow, and both are design changes rather than flags. Widening the
per-step work is the only route to using the chip properly, either at much
higher resolution or by solving many equilibria at once with `vmap`. And a
`v5litepod-4` is four chips of which MRX uses one, so there is a further 4x
sitting idle behind a sharded or ensembled solve.

The remaining eager `lax.while_loop`s are worth revisiting too: wrapping
`apply_laplacian` in `jax.jit` measured 4.5 ms against 75 ms eager, a 17x that
this work did not take because it changes the library's call structure.

## Reproducing this

```bash
PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --gap-sweeps 0 --skip-relax --out outputs/bench/tpu.json

PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=cpu \
  ./run_on_tpu.sh --gap-sweeps 0 --skip-relax --out outputs/bench/cpu.json

python tpu_bench_mrx.py --compare script_outputs/bench/{tpu,cpu}.json
```

Drop `--skip-relax` to include the relaxation timing, which adds about 4
minutes on a TPU and 9 on the CPU. `--gap-sweeps 5` adds the `lambda_1`
diagnostic, which is not part of the construction and cost 6.8 of 9 setup
minutes when it was on by default.

The figures in this directory are from the 100-step li383 run of the previous
session (41.3 s/step, before fixes 2-4), with `||F||` falling 5.540e-02 ->
1.004e-02 and helicity conserved to 5.6e-07 relative. The physics is unchanged
by this work: every fix is a data-movement rewrite verified to agree exactly
or to float32 round-off.
