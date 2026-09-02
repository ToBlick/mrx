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

Measured on one `v5litepod-4` in `us-west4-a`, li383 at `--ns 12,24,12
--p 3` in float32, against the host CPU of the same VM so that only the
backend differs:

| | v5e before | v5e after | same VM's CPU | H100 |
|---|---|---|---|---|
| `build_sequence` | 203 s | 35.5 s | 40.6 s | - |
| `compute_nullspaces` | 222 s | 17.5 s | 55.1 s | - |
| `mass_core_apply` k=1 | 6.60 ms | 0.505 ms | 3.79 ms | - |
| `apply_laplacian` k=1 | 10 020 ms | 76.4 ms | 108 ms | - |
| relaxation, per step | 100.3 s | 13.03 s | 45.7 s | 0.41 s |

Two things account for that. The compilation cache above, and removing every
index tensor from the mass kernel: the gather and the scatter at its two
ends are both separable shift maps, so both are pure data movement with
sources and destinations known at compile time.

| at (12,24,12) p=3 | v5e indexed | v5e structured | CPU indexed | CPU structured |
|---|---|---|---|---|
| gather | 1.624 ms | 0.049 ms | 0.070 ms | 0.113 ms |
| scatter | 2.011 ms | 0.060 ms | 0.398 ms | 0.303 ms |

That is the most transferable fact here. On the indexed forms a v5e is 23x
and 5x slower than the VM's own CPU; on the structured forms it is 2.8x and
5x faster. Identical work, identical answers. If a kernel is slow on this
hardware, look for an index tensor first.

What is left is not compilation either. `relaxation_loop` is already
`@jax.jit` around a `lax.scan`, so the 13 s per step is pure device execution.
From the primitives above it ought to be 350 ms. The per-apply arithmetic is
right and the apply *count* is not: **one step is 37 478 operator applies**,
which at 0.35 ms each is exactly the 13 s. Three MINRES solves are 99.5% of
them, and the velocity smoothing solve alone takes 3 497 iterations and is
75% of the step; the inverse-mass CG solves usually blamed for the cost take
20-27 iterations and are 0.5%. The count grows linearly with the DoFs while
the system becomes more mass-dominated, which points at the preconditioner.

That is not a TPU property -- the same counts hold on CPU and GPU, where each
apply is ~30x cheaper -- but a TPU makes it visible, and it is the largest
speedup available to MRX on any backend. A `v5litepod-4` is also four chips
and MRX uses one; a parameter sweep can take all four with `jax.pmap` over a
stacked batch of initial states, with no library change.

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

`tpu/results/benchmark_v5e_vs_cpu.md` has the full table, including three
fixes that sounded right and that measurement refuted.

## Cost

A Cloud TPU API node bills until deleted, and a failed script does not stop
it. Audit after every session:

```bash
gcloud compute tpus tpu-vm list --zone=-
gcloud compute instances list
gcloud compute disks list
```

`zones.sh` caps persistent data disks at `MAX_DATA_DISKS` (2) because
sweeping zones otherwise leaves one behind in each; `./acquire_tpu.sh --gc`
removes the surplus.
