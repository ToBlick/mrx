# MRX on a Cloud TPU v5e: what was slow, and why (2026-09-03)

Measured on `mrx-scaling-on-tpus`, 1-3 September 2026, one `v5litepod-4` node,
JAX 0.11.1, li383, float32, `jax_default_matmul_precision=highest` unless a row
says otherwise. Two control columns run the same tree: the **host CPU of the
same VM** (through the same driver with `RUN_PLATFORM=cpu`, so only the backend
differs) and **one NVIDIA H200** on NYU Torch under Singularity. It is an H200
rather than an H100 because `h100` is not open to this account; for this
comparison that is conservative, since an H200 is the same GH100 compute with
4.8 TB/s of HBM3e against 3.35, and these matvecs are bandwidth-bound.

Reproduce with `scripts/benchmark/relaxation_bench.py` and
`scripts/benchmark/matvec_bench.py`. Operational instructions are in
`docs/source/tpu.md`; this note is the numbers.

## The headline

At `--ns 12,24,12 --p 3`:

| Measurement | v5e before | v5e after | Same VM's CPU | One H200 |
|---|---|---|---|---|
| `build_sequence` (warm cache) | 203 s | **38.8 s** | 42.9 s | 61.6 s |
| `compute_nullspaces`, `gap_sweeps=0` | 222 s | **7.53 s** | 11.5 s | 19.2 s |
| `mass_core_apply` k=1 | 6.60 ms | **0.420 ms** | 3.79 ms | - |
| `mass_core_apply` k=2 | 5.00 ms | **0.402 ms** | 3.38 ms | - |
| `E` apply k=1 | 1.999 ms | **0.182 ms** | 0.100 ms | - |
| `apply_derivative_matrix` k=1 | 7.15 ms | **3.02 ms** | 4.77 ms | - |
| `apply_laplacian` k=1 (nested CG) | 10 020 ms | **85.7 ms** | 102 ms | - |
| relaxation, per step | 100.3 s | **3.40 s** | 12.74 s | 12.21 s |

The v5e and CPU columns are one session on one node, so they differ only in
`JAX_PLATFORMS`. The H200 is the same commit and script, float32; in float64 it
is 13.07 s/step. The v5e runs a step **3.7x faster than the CPU it is bolted
to** and **3.6x faster than the H200**, and setup went from about 7 minutes to
46 seconds.

The v5e went from 1.7x slower than the VM's own CPU to 3.7x faster. Four things
did that, in order of size: the persistent compilation cache, the gather, the
assembly, and compiling the extraction operator's two ops together. The first
is configuration; the rest are in `mrx/mass.py` and `mrx/extraction_operators.py`.

The per-step figure is 3.40 s rather than the 11.72 s it was, because the
development branch replaced the smoothing solve's saddle MINRES with the split
identity and the shifted-stiffness atom, cutting its iteration count about 15x.
That change is not this branch's, and it moved the v5e and the CPU together:
the *ratio* between the columns is where this branch's work shows up.

## Fix 1: XLA was recompiling, not the device computing

MRX's inner solves are eager `lax.while_loop`s, so without a persistent cache
XLA recompiles them on every call. One `apply_laplacian` at k=1 cost about 10 s
of compiling to do about 20 ms of arithmetic, and repeated identical calls never
got faster; with the cache the second call is 105 ms. The thresholds matter as
much as the directory, since their defaults skip nearly every kernel this
workload compiles. Setup on a fresh node costs 143 s with no cache, 98 s from a
warm GCS bucket and 53 s from a warm local disk.

## Fixes 2 and 3: the mass kernel holds no indices at all

Both ends of `_sumfact_kernel` used an index tensor: a gather `x[gather_idx]`
to read the element-local input and a `segment_sum` to write the contributions
back. Both plans are separable tensor products of per-axis element-to-DoF maps,
and on a tensor-product B-spline basis every axis map is the pure shift
`g[e, l] = (e + l) mod S`. So both ends are pure data movement with every source
and destination known at compile time: rolled slices in, shifted dense adds out.

| at (12,24,12) p=3 | v5e indexed | v5e structured | CPU indexed | CPU structured |
|---|---|---|---|---|
| gather | 1.624 ms | 0.049 ms | 0.070 ms | 0.113 ms |
| scatter | 2.011 ms | 0.060 ms | 0.398 ms | 0.303 ms |

On the indexed forms a v5e is 23x and 5x *slower* than the VM's own CPU; on the
structured forms it is 2.8x and 5x faster. Identical work, identical answers.
This is the transferable lesson: **if a kernel is slow on this hardware, look
for an index tensor first.**

The shift is a precondition rather than an optimisation. `_shift_plan` raises if
an axis map is not a shift, because there is no other assembly to fall back to
and a basis numbered differently would read and write silently wrong. That is
safe because the library builds only one topology family: `polar=False` is
rejected outright, and every component of every k-form of every buildable
sequence satisfies the shift.

Separately, folding the sum factorization's y and z stages into one contraction
of width `nly * nlz` trades 1.5x the arithmetic for one fewer stage. It is worth
1.48-1.70x on a v5e, 1.23-1.49x on an H200 and 1.62x on a CPU -- narrow
contractions cost per contraction, not per FLOP, on every backend tested.
Folding all three axes loses everywhere: 4.8x the FLOPs is too much to buy back,
and the basis tensor grows two orders of magnitude.

## The matvec baseline

About 99% of the compute is matrix-vector applies, so this is the table any
backend claim rests on. Timing operators *eagerly*, one at a time, overstates
what the relaxation pays by a large and backend-dependent factor, so each
operator here is timed **inside a jitted `lax.scan`** of length 50, which is the
form the relaxation runs. Per-apply cost in **ms**, scan form:

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

The H200 columns are noisy at this scale -- its eager numbers move by 2x between
adjacent rows for the same operator -- so read them as a magnitude. The v5e
columns are stable to the last digit shown.

Three things this settles. **`apply_stiffness` is not cheap**: `K_k = G^T M G`
contains a mass apply and at k=2 costs 58% of one on its own, and
`apply_derivative_matrix` is `D_k = M_{k+1} G_k`, so most of its cost is the
mass kernel plus two extractions. **`apply_laplacian` is not an apply**: at
k>=1 it calls `apply_inverse_mass_matrix`, so it is 20-27 CG iterations.
**Eager microbenchmarks overstate the scan form by wildly different amounts per
backend** -- 1.0-1.6x on CPU, 1.3-6.8x on the v5e, 5.9-66x on the H200 -- so any
table that mixes the two is measuring dispatch.

Where the extraction operator is concerned the ordering inverts: `E` and `E^T`
are 7-11x *cheaper* on the CPU than on either accelerator, because they are pure
data movement with no arithmetic to hide behind.

## Why the GPU loses

Per matvec the arithmetic says an H200 should be **5.9x faster** than the v5e.
Measured end to end it was **1.37x slower** (at the pre-split-solve step cost of
13.04 s against 17.92 s). The whole discrepancy is how much each machine pays
over its own matvec budget: composing the velocity-smoothing solve from the
scan-form costs and its per-iteration call counts gives 12.90 s on the v5e
against 9.73 s attributed from the measured step -- a residual of 0.75x, i.e.
the real fused body is slightly *cheaper* than the sum of its parts. The same
composition on the H200 gives 2.20 s against 13.37 s attributed: **6.1x** over
budget.

A relaxation step is a chain of order 250 000 sequentially dependent kernels.
The v5e executes the whole `lax.scan` body as one on-device program and has
almost no per-kernel cost; the GPU pays dispatch on each. This is consistent
with the directly measured result that dispatch is *not* what makes a v5e kernel
slow: one device call is 0.037 ms, and 20 scatters fused into one `jit` cost
0.531 ms each against 0.533 ms unfused. Same fact from both ends.

## `jax_default_matmul_precision` is a real TPU tax, and `high` is the floor

The MXU multiplies bfloat16 natively, so float32 at `highest` is six passes,
`high` three and `default` one; in float64 the setting does nothing. It is worth
up to 1.55x on the mass kernel and 1.22x on a relaxation step. It is kept at
`highest` anyway: at `high` the geometry map carries a 1.9e-04 relative error on
`DF`, and at `default` the map **folds** -- `det DF` reaches -1.3e-01 and
`set_geometry` refuses it. `high` also moved the adaptive stepper onto a
different trajectory, which spends much of what it saves.

float32 on the MXU is not otherwise a numerical concern: inverse-mass CG at the
same tolerance takes the same iteration count on both backends, 20 at k=1 and 24
at k=2.

## Fixes that sounded right and were not

Measured, not argued about:

| Idea | Result on v5e | Verdict |
|---|---|---|
| `indices_are_sorted=True` on the scatter | 0.615 ms vs 0.533 ms unsorted | **slower**; sorting properly is 0.873 ms |
| Dense matmul instead of the extraction operator | 0.408 ms vs 0.533 ms, 303 MB resident | 1.3x is not worth the memory |
| Per-call dispatch overhead is the problem | one device call 0.037 ms; 20 scatters fused 0.531 ms each vs 0.533 unfused | real device work, not launch overhead |
| Narrow contractions underuse the 128-wide MXU | cost is flat from K=4 to K=8, then tracks K | refuted: K=4 is not charged as K=128 |
| Structured extraction operator | `E`/`E^T` are 700 of a step's ~28 700 applies | not where the time is |

The sorted-indices hint is genuinely useful on GPUs, where a scatter lowers to
hardware atomics. It does nothing here.

## The apply count, not the apply cost

For a while the assumed explanation for a 13 s step was that a li383 step is a
chain of kernels too small to fill the chip: from the primitives a step ought to
be roughly 350 ms, which is 37x off.

That was wrong. The per-apply arithmetic was right; the apply *count* was wrong,
by two orders of magnitude. Because the step is a jitted scan, every solver's
`info` is discarded before anything can read it, so the iteration counts had
never been looked at. Running one step eagerly on a host, where `info` is
concrete, showed the velocity-smoothing MINRES dominating everything else at
thousands of iterations. That is a **backend-independent** property, so the
preconditioner work it prompted (upstream's split solve) helps CPU and GPU by
the same factor. A TPU only made it visible.

## Using all four chips

A `v5litepod-4` is four chips where one relaxation uses one, because the solve is
a chain of dependent applies on ~10^4 DoFs with nothing to shard. A parameter
sweep has no such problem: `jax.pmap` over a stacked batch of initial states
measured **3.99x** on four chips, 5.8 s against 23.1 s run one at a time, each
member reproducing its sequential answer to 5.0e-05. No library change was
needed -- the relaxation scan is already a pure function of an equinox `State`.

That factor is only visible if compilation is timed apart from execution. Both
forms compile once, but the serial loop amortises it over four calls while the
pmap pays it on its only one, so a single-shot comparison reads 2.77x and a cold
XLA cache reads 1.56x. `scripts/pmap_sweep.py` runs each form twice and reports
the second; it needs only two devices, so it works on a multi-GPU node too.
