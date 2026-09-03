# MRX on a v5e: what was slow, and why

Measured on `mrx-scaling-on-tpus`, 1-2 September 2026 in `us-west4-a` and
2-3 September in `us-south1-a`, one `v5litepod-4` node, JAX 0.11.1, li383 at
`--ns 12,24,12 --p 3`, float32, `jax_default_matmul_precision=highest` unless
a row says otherwise.

Two control columns, both running the same tree as the v5e:

- **The host CPU of the same VM**, through the same driver with
  `RUN_PLATFORM=cpu`, so the software stack, geometry and precision are
  identical and only the backend differs.
- **One NVIDIA H200** on NYU Torch (`gpu_baseline.slurm`), the same scripts at
  the same resolution in both float32 and float64. It is an H200 rather than
  an H100 because `h100` is not open to this account; for this comparison that
  is conservative, since an H200 is the same GH100 compute with 4.8 TB/s of
  HBM3e against 3.35, and these matvecs are bandwidth-bound.

Reproduce with `tpu_bench_mrx.py` and `matvec_bench.py`.

## The headline

| Measurement | v5e before | v5e after | Same VM's CPU | One H200 |
|---|---|---|---|---|
| `build_sequence` (warm cache) | 203 s | **35.5 s** | 40.6 s | - |
| `compute_nullspaces`, `gap_sweeps=0` | 222 s | **17.5 s** | 55.1 s | - |
| `mass_core_apply` k=1 | 6.60 ms | **0.505 ms** | 3.79 ms | - |
| `mass_core_apply` k=2 | 5.00 ms | **0.459 ms** | 4.86 ms | - |
| `E` apply k=1 | 1.999 ms | **0.181 ms** | 0.102 ms | - |
| `apply_derivative_matrix` k=1 | 7.15 ms | **2.74 ms** | 6.28 ms | - |
| `apply_laplacian` k=1 (nested CG) | 10 020 ms | **76.4 ms** | 108 ms | - |
| relaxation, per step (mass precond) | 100.3 s | **11.72 s** | 45.7 s | 12.61 s |
| relaxation, per step (laplacian precond) | -- | **8.05 s** | -- | -- |

The last row of the "after" column predates the folded factorization below;
the mass-preconditioner row includes it. Folding took the step from 13.04 s to
**11.72 s** on the v5e and 17.92 s to **12.61 s** on the H200 (24.16 to 15.23
in float64), with the trajectory unchanged on both.

The v5e went from 1.7x slower than the VM's own CPU to **3.5x faster**, and
setup from about 7 minutes to under a minute. Four things did that, in order
of size: the persistent compilation cache, the gather, the assembly, and
compiling the extraction operator's two ops together. The first is
configuration; the rest are in `mrx/mass.py` and `mrx/extraction_operators.py`,
and they are the next four sections.

**The v5e is also not behind a datacentre GPU on this workload; it is ahead of
one.** Earlier versions of this file claimed the v5e was "32x behind one
H100", from a `0.41 s/step` figure that was never the same measurement --
[a warm-start A/B on W7-X](../../docs/research/release_review_sweep_2026-08-27.md),
not li383. The H200 column above is the like-for-like measurement, and the
v5e is **1.08x ahead** of it (it was 1.37x before the folded factorization,
which helped the GPU more). The comparison, not the hardware, was the
problem. "The matvec baseline", "The step, composed from measured parts" and
"Why the GPU loses" measure where that goes, and the short version is that the
v5e runs the relaxation at its own matvec rate while the GPU pays 8x over
its.

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

Those attribution rows are **upper bounds on the pre-fix kernel, not shares of
the post-fix one**, and reading them the other way is what made it look as
though a 0.854 ms gather sat inside a 0.506 ms `mass_core_apply`. Isolating one
component forces XLA to materialise an intermediate the fused kernel never
writes -- at k=1 the gather expands 3 456 values to 124 416 -- so an isolated
row can exceed its own share and the rows do not sum to the whole.
`tpu_bench_mrx.py` now labels them `(upper bound)`.

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

## The matvec baseline

About 99% of the compute is matrix-vector applies, so this is the table that
any backend claim has to rest on. Everything above it in this file was
measured *eagerly*, one operator at a time, and that turns out to overstate
what the relaxation pays by a large and backend-dependent factor. Each
operator here is therefore also timed **inside a jitted `lax.scan`** of length
50, which is the form the relaxation actually runs. `matvec_bench.py`, li383
`(12,24,12)` p=3, all five columns the same script and the same tree.

Per-apply cost in **ms**, scan form:

| operator | v5e before fold | v5e after fold | H200 f32 | H200 f64 | VM CPU f32 |
|---|---|---|---|---|---|
| `apply_mass_matrix` k=1 | 0.836 | **0.716** | 0.090 | 0.102 | 1.417 |
| `apply_mass_matrix` k=2 | 0.749 | **0.659** | 0.082 | 0.100 | 1.369 |
| &nbsp;&nbsp;`mass_core` k=1 | 0.504 | **0.394** | 0.105 | 0.120 | 1.406 |
| &nbsp;&nbsp;`mass_core` k=2 | 0.458 | **0.365** | 0.090 | 0.108 | 1.356 |
| &nbsp;&nbsp;`E` k=1 | 0.299 | 0.295 | 0.406 | 0.059 | 0.043 |
| &nbsp;&nbsp;`E^T` k=1 | 0.311 | 0.297 | 0.322 | 0.393 | 0.027 |
| `apply_stiffness` k=1 | 0.790 | **0.697** | 0.108 | - | 1.472 |
| `apply_stiffness` k=2 | 0.433 | **0.390** | 0.047 | - | 0.561 |
| `apply_derivative` `D^T D` k=1 | 1.537 | **1.354** | 0.170 | - | 2.832 |
| mass atom k=2 (precond) | 0.065 | 0.065 | 0.036 | - | 0.030 |
| laplacian atom k=2 (precond) | 0.072 | 0.072 | 0.086 | - | 0.044 |

The H200 columns are noisy at this scale -- its eager numbers move by 2x
between adjacent rows for the same operator -- so read them as a magnitude,
not to three digits. The v5e columns are stable to the last digit shown.

Three things this settles.

**`apply_stiffness` was never timed before** and it is not cheap: `K_k = G_k^T
M_{k+1} G_k` contains a mass apply, and at k=2 it costs 58% of a mass apply on
its own. Likewise `apply_derivative_matrix` is not an incidence operation --
it is `D_k = M_{k+1} G_k`, so most of its cost is the mass kernel plus two
extractions.

**`apply_laplacian` is not an apply and has been relabelled.** At k>=1 it
calls `apply_inverse_mass_matrix`, so the 76.4 ms in the headline table is
20-27 CG iterations, not a matvec. `tpu_bench_mrx.py` now prints it as
`SOLVE apply_laplacian`.

**Eager microbenchmarks overstate the scan form, and by wildly different
amounts per backend**: 1.0-1.6x on CPU, 1.3-6.8x on the v5e, and 5.9-66x on
the H200. Any table that mixes the two, or that compares an eager number on
one machine with a fused number on another, is measuring dispatch. That is
what the old primitive rows were doing.

Where the extraction operator is concerned the ordering inverts: `E` and `E^T`
are 7-11x *cheaper* on the CPU than on either accelerator, because they are
pure data movement with no arithmetic to hide behind. They are 700 of a
step's ~28 700 applies, so this does not matter, but it is why the "structured
extraction operator" idea below stays rejected.

## The step, composed from measured parts

The old claim that a step is "37 478 applies at 0.35 ms each" was circular:
the 0.35 ms was obtained by dividing 13.03 s by 37 478, not measured. And
summing the *eager* primitives gave >= 7.84 ms per smoothing iteration, i.e.
27.4 s for one solve against 13.03 s measured for all nine -- a 2.8x
contradiction with itself.

Counting calls through `mrx.operators` (backend-independent), one MINRES
iteration of the velocity smoothing solve is:

| term | per iteration |
|---|---|
| `apply_stiffness` k=2 | 1 |
| `apply_mass_matrix` k=2 | 1 |
| `apply_mass_matrix` k=1 | 1 |
| `apply_derivative_matrix` k=1, forward and transpose | 2 |
| mass-atom preconditioner, k=2 and k=1 | 2 |

Priced with the scan-form costs above, at the solve's 3 497 iterations. Every
number in this section and the next is from **before** the folded
factorization: they are a reconciliation of a step against its own parts, and
both sides moved together, so re-deriving them post-fold would change the
levels and not the conclusion.

| | v5e `highest` | v5e `high` | H200 f32 | H200 f64 | VM CPU f32 |
|---|---|---|---|---|---|
| per iteration | 3.69 ms | 2.97 ms | 0.629 ms | 0.783 ms | 6.24 ms |
| smoothing solve | **12.90 s** | 10.37 s | 2.20 s | 2.74 s | 21.83 s |
| whole step, measured | 13.04 s | - | 17.92 s | 24.16 s | 45.7 s |

On the v5e the two accounts now agree. The smoothing solve is 74.6% of the
step's applies, so 13.04 s attributes about 9.73 s to it, against 12.90 s
composed: **1.33x**, down from 2.8x, and in the direction a residual should
go. The scan form still slightly overstates, because the real MINRES body
fuses all seven applies and the vector updates into one XLA program while the
benchmark's scan fuses one operator at a time.

The H200 column does not agree, by 6.1x in the other direction, and that
disagreement is the subject of the next section.

Reproduce the table and the composition with
`python tpu/summarize_matvec.py LABEL=PATH ...`.

## Why the GPU loses, and what it means

The same arithmetic says the H200 should be **5.9x faster per matvec** than
the v5e. Measured end to end it is **1.37x slower**. The whole discrepancy is
how much each machine pays over its own matvec budget.

Comparing like with like on the velocity smoothing solve, whose composed cost
is in the table above and whose share of a step's applies (74.6%) is the same
on every backend:

| | composed from matvecs | attributed from the step | measured / composed |
|---|---|---|---|
| v5e float32 | 12.90 s | 13.04 x 0.746 = 9.73 s | **0.75x** |
| H200 float32 | 2.20 s | 17.92 x 0.746 = 13.37 s | **6.1x** |

The v5e runs the relaxation slightly *faster* than its own isolated matvecs
predict, which is what fusing the whole MINRES body into one program should
do. The H200 runs it 6.1x slower than its matvecs predict. Those two together
are the 8.1x that turns a 5.9x per-matvec advantage into a 1.37x end-to-end
deficit.

The likely cause is the shape of the workload rather than anything about MRX:
a step is roughly 200 000 *sequentially dependent* kernels -- ~28 700 applies,
several kernels each, in Krylov recurrences where every iteration needs the
last one's answer. A TPU executes the whole `lax.scan` body as one on-device
program; a GPU issues each kernel, and the H200's ~15 s per step of
non-arithmetic time over 250 000 launches is about 60 us apiece, the right
order for launch and synchronisation on a chain this long. This has not been
confirmed with a GPU profile, so treat the mechanism as the best available
explanation and the timings as the measurement.

It holds at the other resolution too, and more strongly, which is what the
launch-bound explanation predicts: a smaller problem has the same number of
sequential kernels doing less work each.

| li383 p=3, per step | v5e f32 | H200 f32 | H200 f64 |
|---|---|---|---|
| `(12,24,12)` | **13.04 s** | 17.92 s | 24.16 s |
| `(8,16,8)` | **2.29 s** | 10.52 s | 11.43 s |

At `(8,16,8)` the v5e is **4.6x faster** than the H200 in the same dtype.

One figure this does *not* contradict: `docs/source/concepts/relaxation.md`
reports 0.7-0.9 s/step on an H100 at `(8,16,8)` p=3 float64, and that is
**W7-X FMM002, not li383**. Different geometry, different conditioning,
different iteration counts, so it is not comparable to the 11.43 s above and
neither number impugns the other. The figure that was genuinely misused is the
`0.41 s/step`, which was quoted as an H100 li383 step and is a warm-start A/B
on W7-X.

The consequence is the useful part: **this workload is a good fit for a TPU
and a poor one for a GPU**, and the reason is the sequential Krylov chain, not
precision or bandwidth. It also means the leverage on both machines is the
same one -- fewer iterations -- which is an apply-count question, addressed
below and backend-independent.

## `jax_default_matmul_precision` is a real TPU tax, and `high` is the floor

[`mrx/precision.py`](../../mrx/precision.py) pins `jax_default_matmul_precision`
to `"highest"`. The reason recorded there is entirely about GPUs: Ampere and
later run float32 matmuls in TF32, which made the W7-X map's `dR/dtheta` 19%
wrong. float64 is unaffected, so the setting is **free on the GPU and not free
on the TPU**, where the MXU multiplies bf16 natively and float32 is emulated in
six passes at `highest`, three at `high`, one at `default`.

It is worth 1.2-1.55x on the kernels that dominate, from the table above:

| | `highest` -> `high` |
|---|---|
| `mass_core` k=1 | 0.504 -> 0.324 ms, **1.55x** |
| `mass_core` k=2 | 0.458 -> 0.295 ms, 1.55x |
| `apply_mass_matrix` k=2 | 0.749 -> 0.586 ms, 1.28x |
| `apply_stiffness` k=2 | 0.433 -> 0.383 ms, 1.13x |
| composed smoothing iteration | 3.69 -> 2.97 ms, **1.24x** |

The preconditioner atoms and the extraction operator do not move at all, which
is the expected signature: they are data movement, not matmul.

The question the setting was introduced to answer is whether the geometry
survives. Measured directly on the v5e with `map_precision.py`, li383
`(12,24,12)` p=3 float32, against the same process at `highest`:

| setting | det DF range | folds | max rel err on DF | on det DF |
|---|---|---|---|---|
| `highest` | [1.504e-02, 2.616e+00] | no | - | - |
| `high` | [1.484e-02, 2.620e+00] | no | 1.9e-04 | 2.7e-03 |
| `default` | [**-1.319e-01**, 3.014e+00] | **yes** | 3.2e-02 | 4.8e-01 |

So the requirement is real and it is the geometry that imposes it: at
`default` the map folds outright -- `det DF` goes negative and `set_geometry`
refuses the map -- with a 48% error on the determinant. `high` is the floor,
and at 1.9e-04 on `DF` it is comfortably inside it.

End to end the prediction holds and the trajectory does not. Same node, same
session, li383 `(12,24,12)` p=3 float32, five steps:

| | per step | residual after 5 steps | `dt` | `dH/H` |
|---|---|---|---|---|
| `highest` | 13.04 s | 3.09e-02 | 3.10 | -9.4e-06 |
| `high` | **10.71 s**, 1.22x | 3.74e-02 | 1.70 | -2.9e-06 |

The 1.22x measured matches the 1.24x composed from the matvec table, which is
a good check on the step model. But `high` **took a different trajectory**:
the adaptive controller chose `dt = 1.70` against `3.10` and the run is
*behind* after the same five steps, 3.74e-02 against 3.09e-02. That is not
round-off in the reported digits, it is a different sequence of steps. So the
1.22x is a per-step figure that is at least partly, and possibly wholly, spent
on making less progress per step; a wall-clock-to-target comparison would need
a much longer run.

**No library change was made**, for three reasons that compound. `high` on a
GPU *is* TF32, the exact setting that produced the documented 19% error, so
the default cannot simply move and any change would have to be
platform-conditional. The measured cost is a 2.7e-03 error on `det DF`, and
`default` -- one step further -- folds the map outright, so there is one
notch of headroom and no margin beyond it. And the trajectory result above
means the speedup is not established as a real one. It is available per-run
instead: both `matvec_bench.py` and `tpu_bench_mrx.py` take
`--matmul-precision high`, with these numbers as the price.

## Fixes that sounded right and were not

Measured, not argued about:

| Idea | Result on v5e | Verdict |
|---|---|---|
| `indices_are_sorted=True` on the scatter | 0.615 ms vs 0.533 ms unsorted | **slower**; and sorting properly, i.e. permuting the contributions each apply, is 0.873 ms |
| Dense matmul instead of the extraction operator | 0.408 ms vs 0.533 ms, 303 MB resident | 1.3x is not worth the memory |
| Per-call dispatch overhead is the problem | one device call is 0.037 ms; 20 scatters fused into one `jit` cost 0.531 ms each vs 0.533 ms unfused | it is real device work, not launch overhead |

The sorted-indices hint is genuinely useful on GPUs, where a scatter lowers to
hardware atomics. It does nothing on this hardware.

The third row is specifically about the **v5e**, and it is worth keeping
straight against the GPU section above: dispatch is not what makes a v5e
kernel slow, and dispatch appears to be exactly what makes the H200's step
slow. The two are consistent -- a TPU runs the fused scan body as one program
and has almost no per-kernel cost to pay, which is the same fact from both
ends.

## What is left: the step is tens of thousands of operator applies

`relaxation_loop` is already `@jax.jit` around a `lax.scan`, so the 13 s per
step contains no compilation. It is worth asking what the arithmetic *should*
cost. From the primitives above, a step ought to be roughly 350 ms: about 350
mass-core applies at 0.5 ms and 700 extraction applies at 0.18 ms. That is 37x
off the measured 13.04 s, and for a while the assumed explanation was that a
li383 step is a chain of kernels too small to fill the chip.

That was wrong. The per-apply arithmetic was right; the apply *count* was
wrong, by two orders of magnitude. Because the step is a jitted scan, every
solver's `info` is discarded before anything can read it, so the iteration
counts had never been looked at. Running one step eagerly on a host, where
`info` is concrete, gives this (li383, `(12,24,12)` p=3, float32,
`tol = sqrt_eps = 3.45e-04`, `maxiter = 10000`):

| # | solver | n | iterations | converged | applies/iter | applies |
|---|---|---|---|---|---|---|
| 0 | CG, inverse mass | 8700 | 0 | yes | 2 | 0 |
| 1 | CG, inverse mass k=1 | 8124 | 25 | yes | 2 | 50 |
| 2 | CG, inverse mass | 8376 | 0 | yes | 2 | 0 |
| 3 | MINRES, Leray k=3 | 2880 | 221 | yes | 7 | 1 547 |
| 4 | CG, inverse mass k=2 | 8376 | 27 | yes | 2 | 54 |
| 5 | **MINRES, velocity smoothing k=2** | 8376 | **3 497** | yes | 7 | **24 479** |
| 6 | MINRES, Leray k=3 | 2880 | 361 | yes | 7 | 2 527 |
| 7 | CG, inverse mass k=2 | 8376 | 22 | yes | 2 | 44 |
| 8 | CG, inverse mass k=1 | 8124 | 21 | yes | 2 | 42 |

**About 28 700 operator applies in one step.** The chip was never the problem
and the kernels are not too small: the step really does ask for tens of
thousands of them, and section "The step, composed from measured parts" above
prices that count against measured per-apply costs and gets the step time back
to within 1.33x.

The `applies/iter` column is measured, by wrapping the five entry points in
`mrx.operators` and counting calls through one iteration. Earlier versions of
this table did not have it and used **8 for the smoothing solve and 16 for the
Leray solves**, which put the total at 37 478 and, worse, was inconsistent
between two solves of the same shape. The 16 was one solve's whole trace,
setup included, divided by its iterations; the steady-state cost of both is
the same 7. Every share below is recomputed on the measured column.

Three MINRES solves are 99.3% of the work, and one of them, the velocity
smoothing solve `(M_2 + eps L_2) x = M_2 u`, is 85% on its own. The six CG
solves everybody assumes are the cost are 190 applies, 0.7%.

The iteration count scales with the problem. At `(8,16,8)`, `n(2) = 2192`, the
same solve took 951 iterations; at `(12,24,12)`, `n(2) = 8376`, it takes 3 497.
DoFs up 3.8x, iterations up 3.7x -- linear, which is the signature of a
preconditioner that is not controlling the condition number. It is not
stagnation and not a float32 artefact: every solve converges, and `eps` shrinks
as `0.064 / n_r^2`, so the system becomes *more* mass-dominated with resolution
while the iteration count rises anyway.

None of this is a TPU property. **Apply count is backend-independent**: the
same counts appear on the CPU and on the H200, so a preconditioner that halves
them halves every column and cannot open or close a gap between two of them.
Per-apply *cost* is the backend-dependent half, and that is what the matvec
table measures. Conflating the two is what produced the earlier claim that the
GPU was 30x ahead. It is recorded here because a TPU made it visible: when a
single apply costs half a millisecond, 28 700 of them is the whole run.

**This was the highest-value remaining item, and it was not a TPU change.**

### It was a preconditioner chosen for the wrong term

`_coerce_diffusion_preconditioner_spec` justified the mass atom with a comment
claiming `eps * lambda_max(M^-1 L) ~ 0.26` at `(8,16,8)`, i.e. that
`M + eps L` is mass-dominated there. Power iteration on `M^-1 L_hodge` -- the
operator the saddle system actually reduces to, so including the `d d^T` half
that `apply_stiffness` alone omits -- says otherwise:

| ns | eps | lambda_max(M^-1 L) | eps*lambda_max | claimed |
|---|---|---|---|---|
| (8,16,8) | 1.000e-03 | 9.153e+04 | **91.5** | ~0.26 |
| (12,24,12) | 4.444e-04 | 8.107e+05 | **360.3** | ~0.26 |

Wrong by 350x, and not resolution-flat either. The flatness argument was that
`eps ~ n_r^-2` cancels `lambda_max ~ n_r^2`; measured, `lambda_max` grows 8.9x
across those two while `eps` falls only 2.25x, so refining moves *further* from
mass dominance. The residual history shows steady convergence with no plateau
throughout, so this was an ill-conditioned system being solved honestly.

The fix is the metric-lumped **Laplacian** atom as `(1/eps) P_L`, a
`'laplacian'` kind that `TimeStepper.smooth_velocity` now selects. MINRES
iterations to `sqrt_eps` on li383 k=2, float64, lower is better:

| eps | mass | laplacian |
|---|---|---|
| 1e-06 | **498** | 3682 |
| 1e-04 | **750** | 2919 |
| 1e-03 (the smoothing eps at n_r=8) | 2130 | **1655** |
| 1e-02 | 5774 | **950** |

At `(12,24,12)`, eps=4.44e-04: **8493 vs 3636**. The scaling improves as well
as the level -- mass goes 2130 to 8493 (4.0x), the new kind 1655 to 3636 (2.2x)
-- so the crossover moves the right way with refinement. `'auto'` stays mass,
because below eps ~ 1e-4 mass is much better, and that is where the resistive
step's `dt * eta` lives.

**On this v5e, same node and session, `(12,24,12)` p=3 float32:**

| velocity-smoothing preconditioner | 5 steps, steady | per step |
|---|---|---|
| `auto`, the mass atom | 65.11 s | 13.02 s |
| `laplacian` | 40.25 s | **8.05 s** |

**1.62x on the whole relaxation.** The `auto` figure reproduces the 13.03 s of
the earlier session to three significant figures, and a third session on the
patched tree measured 13.04 s, which is what makes this an apples-to-apples
comparison rather than two measurements of different machines.
The trajectory is unchanged: six li383 steps agree to 4.3e-09 relative against
a solver tolerance of 1.49e-08.

Three alternatives were measured and rejected. `'jacobi'`, the shifted diagonal
that *does* know about `eps L`, is far worse (14372 iterations at `(8,16,8)`,
no convergence at `(12,24,12)`). Unpreconditioned does not converge.
Preconditioned CG on the reduced SPD operator cuts outer iterations 18x -- 462
against 8493 -- and pays every bit of it back in the inner mass solve needed to
apply `d d^T`, 54264 inner CG iterations, landing at wall-clock parity. Worth
recording that an inexact inner solve costs only 3% more outer iterations, so
flexible CG is not the obstacle; the inner solve simply is not cheap.

The condition number is still resolution-dependent, so this is a large constant
rather than a cure. The principled object is a separable `(M + eps L)` atom,
right at every eps, which `docs/research/OPEN.md` section 3.9 already proposes.

### The Leray projections, now about 28% of a step

The Laplacian preconditioner cuts the smoothing solve's iterations by about
2.3x, from 24 479 applies to roughly 10 500, while the two Leray projections
are untouched at 1 547 and 2 527. So they go from **14% of a step to about
28%**, which is what made them worth measuring for the first time. This is an
**apply-count** item: it moves every backend by the same factor and closes no
TPU-versus-GPU gap. `leray_measure.py`, li383 `(8,16,8)` p=3 float64, 419
MINRES iterations.

Per iteration the k=3 saddle solve costs **7 applies**: one `apply_stiffness`,
two `apply_mass_matrix`, two `apply_derivative_matrix`, and the two block
preconditioners. This is the measurement that corrects the 16 in the table
above: tracing the whole solve gives 32 primitive calls, but most of those are
setup, and dividing them by the iteration count inflates the steady state by
2.3x.

The reason it takes 419 iterations is that neither block preconditioner is
close to an exact inverse. Relative miss, `||P A x - x|| / ||x||`:

| block | relative miss |
|---|---|
| lower, k=2 mass | 2.43 |
| upper, k=3 Schur | 1.88 |

A miss of 1.88 on the upper block is the answer to "the metric-lumped
Laplacian atom at k=3 *is* a fast-diagonalisation solve, so why 361
iterations": it is an exact inverse of the *lumped* operator, and the lumped
operator is not close to `S_3`. `lambda_max(P M_2)` is 2.66, so the lower
block is off by a similar factor.

### A separable `(M + eps L)` atom: built, measured, reverted

The obvious response to those misses is a preconditioner that models the
operator the solves actually invert. `docs/research/OPEN.md` section 3.9
proposed one and gave two reasons it looked hard. **Both are wrong**: the atom
was built and measured, and then removed again for the reason below.

*"The atoms share no generalised eigenbasis."* They already do. The mass atom
and the Laplacian atom build their 1-D axis masses through different code for
different purposes, and both come out **bit-identical** -- verified for k=0..3
and every component. Both leave
the axis mass unweighted and carry the metric outside as a diagonal. Only
*which* diagonal differs, and that is a choice.

*"`eps = eta * (accumulated dt)` changes every step, so the factorisation
would be rebuilt."* It would not. Where `V.T M V = I` and `V.T L V =
diag(mu)`, `M + eps L` is `1 + eps * mu` -- diagonal for **every** `eps`, same
`V`. One build serves the whole line search, and `eps` stays a tracer.

The construction works. Relative miss at k=2, li383 `(8,16,8)` p=3 float64:

| eps | shifted atom | mass atom (today) |
|---|---|---|
| 0 | 3.26 | **2.92** |
| 4.4e-4 (operative) | **2.20** | 6.02 |
| 1e-2 | **2.94** | 116.0 |
| 1e-1 | **3.27** | 1158.2 |

Bounded over four decades where the mass atom grows linearly. **And it still
loses**, because CG counts iterations and not norms: 185 against 149 at the
operative shift.

The reason is visible at `eps = 0`, where both model the same operator with
the same 1-D factors and the only difference is that the new atom has no
polar-core block: **82 iterations against 57**. That 1.44x core deficit is
larger than anything the shift handling wins back. Correcting for it projects
~1.16x at the shift the *velocity smoothing* solve uses, `eps = 0.064 / n_r^2
= 4.4e-4`, where `M + eps L` is still mass-dominated and the mass atom is
already close to the right preconditioner.

**One case is not settled by this, and it is the case the open item is about.**
The *resistive* solve is the other user of `M + eps L`, and it runs much
further out: `docs/research/OPEN.md` section 3.9 measures `eta = 1e-1` giving
`eps ~ 0.17`, where the mass atom needs 612-1938 iterations and at `eta = 1`
fails to converge at all. That is squarely inside the range measured above,
not outside it, and it is where the norm advantage is largest -- 3.27 against
1158 at `eps = 0.1`. The iteration count there was *also* worse (it hit the
2000 cap against the mass atom's 1595), but at `eps = 0.1` the operator is
Laplacian-dominated and the untreated core rows matter most, so that number
confounds the core deficit with the atom itself and cannot separate them. The
honest statement is that **the atom is refuted for the smoothing solve and
untested for the resistive one**, and separating them needs the core block.

**It cannot help the Leray upper block at all**, contrary to the plan that
asked for it. That block inverts the Schur complement `S_3`, not `M_3 + eps
L_3`, and `L_3` is *identically zero* -- there is no 4-form for `G_3` to map
into, measured as `||L_3 x|| = 0` exactly. A shifted atom at k=3 **is** the
mass atom. The 1.88 miss needs a Schur model, which is a different object and
is not this.

The code is **not** kept. An atom that is wired into no solve and loses on the
one comparison that decides adoption is a maintenance cost with no caller, so
`build_shifted_mass_laplace_atom`, its test and its measurement script were
reverted once the numbers above were recorded. They are the deliverable. The
construction is a page of `_simultaneous_diagonalize_pair` on the existing 1-D
factors and is quick to rebuild; anyone revisiting it should start with the
polar-core block and expect ~1.16x, not the four decades of norm advantage.

Every alternative available today is worse, so nothing was changed:

| variant | iterations | seconds |
|---|---|---|
| `auto`, metric-lumped outer, uncoupled | **419** | **1.78** |
| `schur.outer='jacobi'` | 647 | 2.42 |
| `coupled=True` | 616 | 2.67 |
| `schur.outer='none'` | 5 878 | 17.25 |

Warm-starting the second projection does help and is already on: **419 cold
against 386 warm**, a 1.08x, which also disposes of the worry that the warm
second Leray costs more than the first. Getting past this needs a
preconditioner that is actually near `S_3`, not a different assembly of the
ones that exist -- the same conclusion the smoothing solve reached, and the
same object would serve both.

### Which roof the mass kernel is under: neither

An earlier version of this section claimed one indexed read survived in
`_to_quadrature` and was worth 4.2x. **That was wrong.** `_shift_plan`
succeeds on all three axes of all three k=2 components, so `_structured_gather`
runs and the indexed read is dead code on li383. A clamped radial axis still
satisfies `g[e, l] == (e + l) % S`, because for `ne = n - p` elements and
`nloc = p + 1` local DoFs the largest index `(ne - 1) + p = n - 1` never
wraps. `gather_cost.py` was timing the fallback path, and has been deleted
rather than fixed: its premise was a v5e-versus-H200 deficit that no longer
exists, and the path it measured is unreachable on this geometry.
`roofline.py` prints the plan per component so this cannot be assumed again.

What the kernel actually asks for, counted rather than estimated
(`roofline.py`, `mass_core_apply` k=2 at `(12,24,12)` p=3):

- 2 592 elements x 64 quadrature points = 165 888 points, 9 792 DoFs
- **19.08 MFLOP** against **4.06 MB** of essential HBM traffic, so 4.70
  FLOP/byte. 98% of those bytes are the six memoised metric weight blocks;
  the vectors themselves are 0.078 MB.

Placed against each machine's roofs. These are the numbers **before** the
folded factorization two sections down, because they are the diagnosis that
led to it; the v5e row is 0.365 ms and 52.3 GFLOP/s after:

| | per apply | achieved | of compute peak | of bandwidth peak |
|---|---|---|---|---|
| v5e `highest` | 0.458 ms | 41.6 GFLOP/s, 8.9 GB/s | **0.13%** | **1.1%** |
| v5e `high` | 0.295 ms | 64.6 GFLOP/s, 13.8 GB/s | 0.10% | 1.7% |
| H200 f32 | 0.127 ms | 150 GFLOP/s, 31.9 GB/s | **0.22%** | **0.7%** |
| VM CPU f32 | 1.356 ms | 14.1 GFLOP/s, 3.0 GB/s | - | - |

**Neither machine is near either roof.** Both are two to three orders of
magnitude below compute peak and two below bandwidth peak, so the honest
statement is that this kernel is limited by neither, and the 3.6x between them
is not a bandwidth or a FLOPs story.

The matmul-precision A/B says where the v5e's time does go. Halving the bf16
passes bought 1.55x on `mass_core`; if `f` is the matmul fraction then
`1 / ((1 - f) + f/2) = 1.55` gives **f = 0.71**. So about 71% of the kernel is
MXU work, achieving 0.13% of the MXU's peak.

The reason is the contraction dimension. `roofline.py` reports every
contraction in the kernel, and they are all the same size:

```282:284:/Users/aak572/mrx/mrx/mass.py
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x_local)
    t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', By, t1)
    return jnp.einsum('zsf,xyzqrf->xyzqrs', Bz, t2)
```

`b`, `d`, `f` are `nloc`, which is 3 or 4; the row half contracts `q = p + 1
= 4`. The batch is large (165 888) but **K is 4 against a 128-wide systolic
array**. The obvious hypothesis is that sum factorization is a pessimization
here: it splits one wide contraction into three narrow ones to save FLOPs,
which is the right trade on a CPU and neutral on a GPU, but here FLOPs are
free at 0.13% of peak.

### The width is not the problem, and the sweep that says so

If the MXU were padding K = 4 up to 128, a matmul at fixed output size would
cost the same at every K up to 128. A sweep measured exactly that, at
the kernel's own batch of 165 888 and its own output width of 4, inside a
jitted scan:

| K | per apply | useful GFLOP/s |
|---|---|---|
| 4 | 0.0152 ms | 349 |
| 8 | 0.0156 ms | 679 |
| 16 | 0.0188 ms | 1 132 |
| 64 | 0.0480 ms | 1 771 |
| 128 | 0.0913 ms | 1 860 |

**Padding is not the limit.** Cost is flat from 4 to 8 and then tracks K: 128
costs 6.0x what 4 does for 32x the arithmetic. So a K = 4 contraction is not
being charged for 128, and the hypothesis above is wrong as stated.

What the sweep does establish is the size of the prize, and it is not where
the hypothesis said. A plain K = 4 matmul at this batch reaches **349
GFLOP/s**; `mass_core_apply`, whose contractions are all K = 3-4, reaches
**41.6**. The kernel is **8.4x off what its own contraction width achieves on
this chip**, and efficiency still rises 5.3x from K = 4 to K = 128. Both point
the same way: not *padded* narrow contractions, just *too many* of them, each
too small to amortise what surrounds it.

### Folding two stages into one: 1.5x the FLOPs, 1.2-1.7x faster

Five ways of doing the same element transform were priced against each other,
all timed inside a jitted scan at the real shapes, on one k=2 component:

| formulation | v5e | H200 f32 | VM CPU |
|---|---|---|---|
| `chain3`, three stages, K = nloc (before) | 1.00x | 1.00x | 1.00x |
| `chain3_bt`, contracted axis moved to the front | 0.98x | 0.91x | 0.91x |
| **`fold2`, y and z contracted jointly, K = 9-16** | **1.48-1.70x** | **1.23-1.49x** | **1.62x** |
| `fold3`, all three, K = 36 | 0.87-0.94x | 1.01x | 0.69x |
| `gemm`, K = 36 flat | 1.28-1.38x | 1.31-1.37x | 0.75x |

All five agree with the current chain to float32 round-off, 1.2e-7 or better.

`chain3_bt` gaining nothing rules out layout: XLA was already handling the
transposes, so the six-axis einsums were not the problem either. `fold3`
losing despite reaching 153 GFLOP/s against `chain3`'s 34 rules out "wider is
simply better" -- it does 4.8x the arithmetic and needs a 24 MB per-element
basis tensor, and neither buys its way back. **`fold2` is the sweet spot**:
two stages instead of three, 1.5x the FLOPs, and a fused table of 166 KB that
carries no radial extent at all (it is the three-axis table divided by
`ne_x * qx * nlx`, which is 144 here).

This is now `_fuse_yz` in `mrx/mass.py`, used by both halves of the kernel:
the row half contracts `(r,s)` against `(c,e)`, which needs exactly the tensor
the column half contracts the other way. It is an einsum reassociation, so
there is no basis it cannot handle and it needs no fallback, unlike the shift
plan above. End to end:

| | before fold | after fold |
|---|---|---|
| v5e, s/step | 13.04 | **11.72** |
| H200 f32, s/step | 17.92 | **12.61** |
| H200 f64, s/step | 24.16 | **15.23** |

The trajectory is unchanged on both backends -- v5e `|F| = 3.09e-02`, `dH/H =
-6.98e-06`, `dt = 3.11` at iteration 5, exactly as before. The GPU gains more
than the TPU, which narrows the v5e's lead from 1.37x to 1.08x: the fold is a
portable improvement, not a TPU workaround.

Note what the composite operators did *not* gain. `mass_core` k=2 improved
1.25x but `apply_mass_matrix` k=2 only 1.14x, because `E` and `E^T` are
untouched and now make up **45% of a mass apply on the v5e** (0.365 ms of core
against 0.297 ms of extraction). They are 7-11x cheaper on the CPU, being pure
data movement with nothing to hide behind. That moves a structured extraction
operator from "under 1% of a step" to the largest remaining TPU-specific item,
and it is the one item in this file whose priority the fold changed.

### The rest, in order of what they would buy

- **Four chips instead of one, measured at 3.99x.** A `v5litepod-4` is four
  chips and MRX uses one. For a parameter sweep this needs no library change:
  the scan is already a pure function of an equinox `State`, so `jax.pmap` over
  a stacked batch of initial states runs one equilibrium per chip. On real
  chips, four equilibria 1% apart at `(8,16,8)` p=3 float32, 4 steps each:
  **5.8 s pmapped against 23.1 s one at a time**, i.e. four chips do four
  problems in the time one chip does one. Each reproduced its own sequential
  answer to 5.0e-05 in float32 against a 3.5e-02 bar, and the members differ
  from one another by the 1-3% they were built to, which is what distinguishes
  four problems from four copies of one. `pmap_sweep.py`.

  The scaling is only visible if compilation is timed separately. Both forms
  compile once, but the serial loop amortises that over four calls while the
  `pmap` pays it on its only call, so comparing single shots reads 2.77x, and
  an earlier attempt that also had a cold XLA cache read 1.56x. Both figures
  measure the compiler. The script now runs each form twice and reports the
  second, which is what a sweep of any real length would see.
- **`apply_laplacian` under `jax.jit`**: 4.5 ms against 75 ms eager, a 17x. Not
  in the relaxation's hot path -- it appears only in resistive steps and in the
  per-outer-iteration helicity -- and it changes the library's call structure,
  so it was not taken.
- **A structured extraction operator.** `E` can be reformulated as a
  contiguous copy plus a small dense block: only 36/60/24 of the extracted rows
  actually mix for k=0/1/2, k=3 is pure selection, and over 99% of rows are
  single value-1.0 pass-throughs. It is the natural mirror of what fixes 2 and
  3 did to the mass kernel. **This entry is now out of date in the caller's
  favour and is the top TPU-side candidate.** It said "under 1% of a step",
  which counted only the ~700 standalone extraction applies. But `E` and `E^T`
  also sit inside every `apply_mass_matrix`, and after the folded
  factorization they are **45% of one on the v5e** -- 0.297 ms of extraction
  against 0.365 ms of kernel, where before the fold the kernel was the larger
  half. They are 7-11x cheaper on the CPU, which is the signature of indexed
  data movement with no arithmetic to hide behind, exactly what fixes 2 and 3
  addressed. Not done here, and it is the next thing to do.

## Reproducing this

```bash
PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --skip-relax --out outputs/bench/tpu.json

PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=cpu \
  ./run_on_tpu.sh --skip-relax --out outputs/bench/cpu.json

python tpu_bench_mrx.py --compare script_outputs/bench/{tpu,cpu}.json
```

Drop `--skip-relax` to include the relaxation timing, which adds about 4
minutes on a TPU and 9 on the CPU. The `lambda_1` diagnostic is no longer
run: it is not part of the construction and cost 6.8 of the 9 setup minutes
when it was on by default.

The matvec table, the GPU column and the step composition:

```bash
# v5e, and again with --matmul-precision high
PUSH_FILES=matvec_bench.py SCRIPT=matvec_bench.py OUTDIR=outputs/matvec \
  VM_NAME=my-tpu-vm ZONE=<zone> SYNC_LOCAL_MRX=1 LOCAL_MRX=/path/to/mrx \
  ./run_on_tpu.sh --ns 12,24,12 --p 3 --out outputs/matvec/v5e.json

# one H200 on NYU Torch, both dtypes, matvecs and relaxation
sbatch tpu/gpu_baseline.slurm

# join them and compose the smoothing solve
python tpu/summarize_matvec.py v5e=...json H200-f32=...json cpu=...json
```

`SYNC_LOCAL_MRX=1` matters more than it looks. A fresh node clones the branch
from the remote, so a session started before a fix is pushed will silently
benchmark the unfixed tree: that is what produced a `mass_core_apply k=1` of
6.87 ms against 0.505 ms for the same operator, and it took an A/B across four
dispatch protocols to establish that the discrepancy was not a measurement
artefact before anyone thought to check `git log` on the node.

The roofline count and the map-precision probe are separate scripts, and both
run anywhere, with or without an accelerator:

```bash
python tpu/roofline.py --ns 12,24,12 --p 3 --k 2 \
  --measured-ms 0.3653 --peak-tflops 33 --peak-gbs 819
python tpu/map_precision.py --matmul-precision {highest,high,default}
```

The one-off scripts behind the MXU sweep, the five-way factorization A/B, the
separable-atom norms and the dispatch A/B are not in the tree. Each answered
one question, the answer is in the section above with its numbers, and none of
them is a thing to re-run: keeping them would have implied the question was
still open.

`roofline.py` needs no device: it counts FLOPs, essential bytes and the
contraction dimension of every einsum from the sequence alone, and takes the
measured time and the machine's peaks as arguments. It also prints whether the
shift plan holds per component, which is what decides between
`_structured_gather` and the indexed fallback -- the question `gather_cost.py`
was deleted for getting wrong.

The 100-step li383 run of the previous session (41.3 s/step, before fixes 2-4)
had `||F||` falling 5.540e-02 -> 1.004e-02 with helicity conserved to 5.6e-07
relative; its figures are not kept here, only described in section 9 of the
guide. The physics is unchanged by this work: every fix is a data-movement
rewrite verified to agree exactly or to float32 round-off.
