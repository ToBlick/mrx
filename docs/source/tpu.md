# Running on a Cloud TPU

`tpu/` is the TPU counterpart to `slurm/`. It differs in one structural
way: on a cluster the machine already exists and you queue for it, while a
TPU does not exist until you create it. So these scripts provision the
hardware from your laptop as well as run on it, and nothing deletes it for
you afterwards.

Read `tpu/TPU_GUIDE.md` before a first session. This page is the summary.

## Before anything else

```bash
gcloud auth login
gcloud config set project <project>
cd tpu && ./check_quota.sh          # read-only preflight, ~40 s
```

`check_quota.sh` reports quota, subnets and accelerator availability per
zone without creating anything.

Quota is not permission. This project holds 512 chips of v5e quota and
Compute Engine still refuses the machine type with
`403 ... not allowed to use the machine type [ct5lp-hightpu-4t]`. v5e is
reachable only through the Cloud TPU API (`gcloud compute tpus tpu-vm`),
which the scripts use; v6e goes through Compute Engine. `zones.sh` records
which candidate takes which API, so the ladder is walked correctly.

## Getting a node and running on it

```bash
VM_NAME=mrx-tpu ./acquire_tpu.sh --acquire-only

SCRIPT=scripts/tutorials/li383_relaxation.py \
  OUTDIR=outputs/tutorials/li383_relaxation \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_TIMEOUT=7200 \
  ./run_on_tpu.sh --ns 12,24,12 --p 3

# Nothing else will. v5e is a TPU API node, v5p/v6e are GCE instances.
gcloud compute tpus tpu-vm delete mrx-tpu --zone=<zone>
gcloud compute instances delete mrx-tpu --zone=<zone>
```

| variable | meaning | default |
|---|---|---|
| `SCRIPT` | path relative to the mrx checkout on the VM | the Poisson report |
| `OUTDIR` | directory the script writes, pulled back afterwards | |
| `VM_NAME`, `ZONE` | the node to use | |
| `RUN_PLATFORM` | `tpu` or `cpu`, to run a stage on the host | `tpu` |
| `RUN_DTYPE` | `float32` or `float64` | `float32` |
| `RUN_TIMEOUT` | seconds | 7200 |
| `PUSH_FILES` | local files to copy into the checkout on the VM | |
| `SYNC_LOCAL_MRX` | rsync a local working tree over the VM's checkout | |

Everything after `run_on_tpu.sh` is passed to the script. Jobs are launched
detached under `setsid` into a log that is streamed back, because a dropped
`gcloud ssh` silently re-runs its `--command` and a second process on the
same chip fails with `The TPU is already in use`.

Capacity is scarce and a request can fail for reasons that are not capacity;
`acquire_tpu.sh` retries the ladder, re-parks expired requests and stops on
the classifications that will never succeed.

## Precision

A TPU has no float64. Set `MRX_DTYPE=float32`, which `run_on_tpu.sh` does;
`mrx.precision` also sets `jax_default_matmul_precision=highest` so the MXU
does not silently drop to bfloat16. On the toroidal Poisson problem the
float32 TPU result matches the float32 CPU reference exactly.

That setting is the one precision knob that costs a TPU and not a GPU: the MXU
multiplies bfloat16 natively, so float32 at `highest` is six passes, `high` is
three and `default` is one, while in float64 the setting does nothing at all.
It is worth up to 1.55x on the mass kernel, and it is kept anyway. Measured on
li383 `(12,24,12)` p=3 in float32, `high` costs a 1.9e-04 relative error on
`DF` and `default` **folds the map** -- `det DF` reaches -1.3e-01 and
`set_geometry` refuses it. `high` also changed the relaxation's trajectory,
which spends much of the 1.22x it saves per step. Both `matvec_bench.py` and
`tpu_bench_mrx.py` accept `--matmul-precision` if you want to measure it
yourself.

Where double precision is required -- field-line tracing for a Poincare
section is the case in practice -- run that stage on the host CPU of the
same node with `RUN_PLATFORM=cpu RUN_DTYPE=float64`.

## Run with the compilation cache on

This is the single largest effect measured on this hardware, and
`run_on_tpu.sh` sets it:

```bash
JAX_COMPILATION_CACHE_DIR=/mnt/data/jax_cache
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1
```

The thresholds matter as much as the directory. Their defaults are tuned
for a few large training programs and skip nearly every kernel this
workload compiles.

A node in a zone with no data disk has nowhere local to keep the cache.
`JAX_CACHE_DIR` also takes a `gs://` path, which works but is second best:
setup costs 143 s with no cache, 98 s from a warm bucket and 53 s from a
warm local disk. Put the bucket in the node's region, and prove the path
with `tpu/gcs_cache_smoke.py` first -- without `etils[epath,epath-gcs]`
installed JAX writes nothing, reads nothing and reports nothing, so a
misconfigured bucket looks like nothing more than a slow run.

MRX's inner solves are eager `lax.while_loop`s, not wrapped in `jax.jit`,
so without a cache XLA recompiles them on every call. One `apply_laplacian`
at k=1 cost about 10 s of compiling to do about 20 ms of arithmetic, and
repeated identical calls never got faster. With the cache the second call
is 105 ms.

## What the hardware is and is not for

Measured on one `v5litepod-4`, li383 at `--ns 12,24,12 --p 3` in float32,
against the host CPU of the same VM so that only the backend differs. Per
matvec one H200 is about 4x faster than the v5e; end to end the v5e is slightly
ahead, because a relaxation step is a chain of roughly 250 000 sequentially
dependent kernels and the v5e executes the whole `lax.scan` body as a single
on-device program. This workload suits a TPU.

Getting there took three changes, and two of the three are library changes
rather than configuration: the compilation cache above; removing every index
tensor from the mass kernel, since the gather and the scatter at its two ends
are both separable shift maps and so pure data movement with sources and
destinations known at compile time; and folding the sum factorization's y and
z stages into one contraction, which trades 1.5x the arithmetic for one fewer
stage and is worth 1.2-1.7x on every backend, GPU and CPU included.

The index tensors are the transferable part:

| at (12,24,12) p=3 | v5e indexed | v5e structured | CPU indexed | CPU structured |
|---|---|---|---|---|
| gather | 1.624 ms | 0.049 ms | 0.070 ms | 0.113 ms |
| scatter | 2.011 ms | 0.060 ms | 0.398 ms | 0.303 ms |

On the indexed forms a v5e is 23x and 5x slower than the VM's own CPU; on the
structured forms it is 2.8x and 5x faster. Identical work, identical answers.
**If a kernel is slow on this hardware, look for an index tensor first.**

Two things that surprised us and are worth knowing before you profile. The cost
was never compilation -- `relaxation_loop` is already `@jax.jit` around a
`lax.scan`, so the time was pure device execution, and the per-apply arithmetic
was right while the apply *count* was not. And apply count is a
backend-independent property, so the preconditioner work it led to helps CPU and
GPU by the same factor; a TPU only made it visible. A `v5litepod-4` is also four
chips where MRX uses one, so a parameter sweep takes all four through `jax.pmap`
with no library change, measured at 3.99x.

float32 on the MXU is not a numerical concern: inverse-mass CG at the same
tolerance takes the same iteration count on both backends (20 at k=1, 24 at
k=2).

Benchmark a solver before committing a long run, and read the two calls
separately, because the first is XLA compiling and the second is the device
computing:

```bash
PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --skip-relax --out outputs/bench/tpu.json
# again with RUN_PLATFORM=cpu, then
python tpu_bench_mrx.py --compare script_outputs/bench/{tpu,cpu}.json
```

`tpu/results/benchmark_v5e_vs_cpu.md` is where the per-backend tables, the
step composition and the per-operator costs are maintained, along with several
hypotheses that sounded right and that measurement refuted. Read the staleness
note at its headline before quoting a per-step number: they predate the
development branch's reformulation of the smoothing solve.

## Cost

A Cloud TPU API node has no `--max-run-duration`, so `tpu/idle_reaper.sh` runs
on every node and deletes it after 20 minutes with nothing running, no login
session and no accelerator held. That is a safety net rather than a policy: a
failed script still costs up to twenty minutes of a v5e. Audit after every
session:

```bash
gcloud compute tpus tpu-vm list --zone=-
gcloud compute instances list
gcloud compute disks list
```

`zones.sh` caps persistent data disks at `MAX_DATA_DISKS` (2) because
sweeping zones otherwise leaves one behind in each; `./acquire_tpu.sh --gc`
removes the surplus.
