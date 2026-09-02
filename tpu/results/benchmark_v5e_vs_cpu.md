# MRX on a v5e: what was slow, and why

Measured on `mrx-scaling-on-tpus`, 2 September 2026, one `v5litepod-4` node in
`us-west4-a`, JAX 0.11.1, li383 at `--ns 12,24,12 --p 3`, float32,
`jax_default_matmul_precision=highest`.

The control column is **the host CPU of the same VM**, run through the same
driver with `RUN_PLATFORM=cpu`, so the software stack, geometry and precision are
identical and only the backend differs. Reproduce with `tpu_bench_mrx.py`.

## The headline

| Measurement | v5e before | v5e after | Same VM's CPU | H100 (docs) |
|---|---|---|---|---|
| `build_sequence` | 203 s | **41 s** | 39 s | - |
| `compute_nullspaces`, `gap_sweeps=0` | 222 s | **51 s** | 61 s | - |
| `apply_laplacian` k=1 (nested CG) | 10 020 ms | **98 ms** | 275 ms | - |
| `mass_core_apply` k=1 | 6.60 ms | **3.20 ms** | 3.88 ms | - |
| `mass_core_apply` k=2 | 5.00 ms | **2.52 ms** | 5.88 ms | - |
| relaxation, per step | 100.3 s | **43.1 s** | 59.7 s | 0.41 s |

The v5e went from 1.7x slower than the VM's own CPU to 1.4x faster, and setup
fell from about 7 minutes to 1.5. It remains roughly 105x behind one H100 on a
single equilibrium.

Confirmed end to end on a real 100-step li383 run on the fixed v5e: 1.9 min
setup, 41.3 s/step, 72 minutes total, with `||F||` falling 5.540e-02 ->
1.004e-02 and helicity conserved to 5.6e-07 relative. The figures in this
directory are from that run.

## Why it was slow: XLA was recompiling, not the device computing

MRX's inner solves run as eager `jax.lax.while_loop`s. Nothing wraps them in
`jax.jit`, so each call traces a fresh closure and XLA compiles a program it has
never seen. On a v5e, one `apply_laplacian` k=1 call cost about 10 s of
compilation to perform about 20 ms of arithmetic.

The signature was repeated identical calls that never got faster:

```
apply_laplacian k=1, no compilation cache:   first 9.963 s   second 9.854 s
apply_laplacian k=1, compilation cache on:   first 10.173 s  second 0.105 s
```

That is the whole diagnosis. Wall-clock phase timings cannot distinguish
compiling from computing, which is why an earlier version of this analysis
concluded the hardware was simply unsuited to the workload.

**Fix:** the persistent compilation cache, now set by `run_on_tpu.sh`.

```bash
JAX_COMPILATION_CACHE_DIR=/mnt/data/jax_cache
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1
```

The thresholds matter as much as the directory: their defaults are tuned for a
few large training programs and skip almost every kernel this workload compiles.

## Second fix: the mass kernel's scatter, done as shifted dense adds

`_sumfact_kernel` ended in a `jax.ops.segment_sum`, and indexed writes are the
one operation a TPU has no fast path for. But the segment ids come from
`_flat_dof_plan`, a separable tensor product of per-axis DoF ids, and on a
periodic B-spline axis `g[e, l] = (e + l) mod n`. The scatter is therefore
algebraically a sum of shifted dense arrays with every destination known at
compile time.

| 221k contributions into 3456 DoFs | v5e |
|---|---|
| `jax.ops.segment_sum` | 2.011 ms |
| shifted dense adds | **0.061 ms** |
| agreement, float32 | 3.4e-07 |

`mrx/mass.py` now checks the shift structure per axis and falls back to
`segment_sum` if any axis fails, so a differently built basis still assembles
correctly. Verified against the full test suite (249 passed, 4 skipped) and
against the scatter path directly at float64, where the two agree to 2e-16.

This also silenced the `Constant folding an instruction is taking > 1s`
warnings on `scatter-add`: with no index tensor there is nothing left to fold.

## Three fixes that sounded right and were not

Measured, not argued about:

| Idea | Result on v5e | Verdict |
|---|---|---|
| `indices_are_sorted=True` on the scatter | 0.615 ms vs 0.533 ms unsorted | **slower**; and sorting properly, i.e. permuting the contributions each apply, is 0.873 ms |
| Dense matmul instead of the extraction operator | 0.408 ms vs 0.533 ms, 303 MB resident | 1.3x is not worth the memory |
| Per-call dispatch overhead is the problem | one device call is 0.037 ms; 20 scatters fused into one `jit` cost 0.531 ms each vs 0.533 ms unfused | it is real device work, not launch overhead |

The sorted-indices hint is genuinely useful on GPUs, where a scatter lowers to
hardware atomics. It does nothing on this hardware.

## What is left, and what would actually help

`relaxation_loop` is already `@jax.jit` around a `lax.scan`, so the 43 s per step
contains no compilation. The remaining gap to an H100 is that one li383 step is a
long chain of small dependent kernels: a mass apply does ~3 ms of work on arrays
of ~10^4 floats, which uses a rounding error of a v5e's width.

Two things follow. Widening the per-step work is the only route to using this
hardware properly, either by going to much higher resolution or by solving many
equilibria at once with `vmap`, and the latter is a design change rather than a
flag. And on the raw indexed-access primitives a v5e is 6-7x slower than this
VM's CPU (scatter 0.533 vs 0.090 ms, gather 0.263 vs 0.037 ms) while being 12x
*faster* on a dense matmul (0.408 vs 4.91 ms) -- which is a compact statement of
what the hardware is and is not for.

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

Drop `--skip-relax` to include the relaxation timing, which adds about 10
minutes on a TPU and 6 on the CPU. `--gap-sweeps 5` adds the `lambda_1`
diagnostic, which is not part of the construction and cost 6.8 of 9 setup
minutes when it was on by default.
