# Running on a Cloud TPU

`tpu/` is the TPU counterpart to `slurm/`. It differs in one structural
way: on a cluster the machine already exists and you queue for it, while a
TPU does not exist until you create it. So these scripts provision the
hardware from your laptop as well as run on it, and a node bills until
something deletes it -- which is why `tpu/idle_reaper.sh` runs on every node.

| script | role |
|---|---|
| `zones.sh` | shared config: the candidate ladder, failure classification |
| `check_quota.sh` | read-only quota preflight |
| `acquire_tpu.sh` | acquire a node and run on it; `--once`, `--acquire-only` |
| `run_on_tpu.sh` | drive one session on a node you already hold |
| `startup.sh` | builds the environment on the node (runs there, not here) |
| `idle_reaper.sh` | deletes the node when it goes idle (runs there, not here) |
| `gcs_cache_smoke.py` | check a GCS compilation cache before relying on it |

Benchmarks live in `scripts/benchmark/` and are not TPU-specific.

## Before anything else

```bash
gcloud auth login
gcloud config set project <project>
cd tpu && ./check_quota.sh          # read-only quota preflight, ~10 s
```

Quota is the one thing worth checking before a sweep, because it is the only
failure retrying cannot fix: a region with a hard 0 never produces a node.
Everything else the ladder discovers in seconds per candidate by trying, and
`zones.sh` names the reason. Quota is also only a ceiling, never an allocation,
so an `OK` means a create is permitted, not that hardware is free.

Quota is also not permission. This project holds 512 chips of v5e quota and
Compute Engine still refuses the machine type with
`403 ... not allowed to use the machine type [ct5lp-hightpu-4t]`. v5e is
reachable only through the Cloud TPU API (`gcloud compute tpus tpu-vm`); v5p
goes through Compute Engine. `zones.sh` records which candidate takes which
API, so the ladder is walked correctly.

The first thing that will confuse you is a create that appears to hang. It is
not hung: `FLEX_START` is Dynamic Workload Scheduler, and if no capacity is free
the instance is created `PENDING` while the scheduler waits up to
`--request-valid-for-duration` for hardware. `gcloud` blocks for that whole
window, so `2h` blocks for two hours. Ctrl-C does not cancel it -- the request
lives server-side -- and cancelling a `PENDING` flex-start is slow, one took 39
minutes to delete. `acquire_tpu.sh` therefore never queues: every attempt fails
fast, and a success is always a real VM. Google exposes no queue position and no
global capacity view, so a sweep-and-sleep loop is the only strategy available.

## Getting a node and running on it

```bash
VM_NAME=mrx-tpu ./acquire_tpu.sh --acquire-only

SCRIPT=scripts/tutorials/li383_relaxation.py \
  OUTDIR=outputs/tutorials/li383_relaxation \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_TIMEOUT=7200 \
  ./run_on_tpu.sh --ns 12,24,24 --p 3

# The reaper will do this after 20 idle minutes; this is how to do it now.
# v5e is a TPU API node, v5p is a GCE instance.
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
| `MRX_BRANCH` | branch to measure; checked out before the run | `static-dynamic-refactor` |

Everything after `run_on_tpu.sh` is passed to the script. Jobs are launched
detached under `setsid` into a log that is streamed back, because a dropped
`gcloud ssh` silently re-runs its `--command` and a second process on the
same chip fails with `The TPU is already in use`.

Capacity is scarce and a request can fail for reasons that are not capacity;
`acquire_tpu.sh` walks the ladder, sleeps and walks it again. `zones.sh` names
the failures:
`STOCKOUT` (no hardware, nothing you can fix), `NOT_ALLOWLISTED` (wrong API,
quota is irrelevant), `QUOTA`, `NO_SUBNET` and `POLICY` (the location org
policy, usually), `DISK_INCOMPATIBLE` (v5p rejects hyperdisk-balanced; the
sweep retries once without the data disk, since losing persistence beats
losing the zone) and `TRANSIENT` (a Google-side blip, retried once).

`startup.sh` builds the environment on the node and is idempotent: it mounts
the data disk at `/mnt/data` when one is attached and otherwise falls back to
the boot disk, installs Miniforge and `jax[tpu]`, clones `MRX_BRANCH`, and
writes `/mnt/data/.mrx_env_ready` once a JAX and MRX smoke test passes. A cold
build is 4-12 minutes; a warm data disk skips to the sentinel and computes
within a minute. Both `startup.sh` and `run_on_tpu.sh` move the checkout to
`MRX_BRANCH`, the latter because the branch is otherwise fixed at create time
in instance metadata and a node acquired with the default would quietly measure
the wrong tree. It also `chmod 1777 /tmp/tpu_logs`, which is not cosmetic:
root creates that directory, so an ordinary user cannot write to it and libtpu
emits `Could not open the log file ... Permission denied` several times a
second, burying real tracebacks.

## Precision

A TPU has no float64. Set `MRX_DTYPE=float32`, which `run_on_tpu.sh` does;
`mrx.precision` also sets `jax_default_matmul_precision=highest` so the MXU
does not silently drop to bfloat16. On the toroidal Poisson problem the
float32 TPU result matches the float32 CPU reference exactly.

That setting is the one precision knob that costs a TPU and not a GPU: the MXU
multiplies bfloat16 natively, so float32 at `highest` is six passes, `high` is
three and `default` is one, while in float64 the setting does nothing at all.
It is worth up to 1.55x on the mass kernel, and it is kept anyway. Measured on
li383 in float32, `high` costs a 1.9e-04 relative error on `DF` and `default`
**folds the map** -- `det DF` reaches -1.3e-01 and `set_geometry` refuses it.
`high` also changed the relaxation's trajectory, which spends much of the 1.22x
it saves per step. Both benchmarks in `scripts/benchmark/` accept
`--matmul-precision` if you want to measure it yourself.

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
warm local disk. Put the bucket in the node's region, and run
`tpu/gcs_cache_smoke.py` against it before relying on it -- without
`etils[epath,epath-gcs]` installed JAX writes nothing, reads nothing and
reports nothing, so a misconfigured bucket looks like nothing more than a
slow run.

MRX's inner solves are eager `lax.while_loop`s, not wrapped in `jax.jit`,
so without a cache XLA recompiles them on every call. One `apply_laplacian`
at k=1 cost about 10 s of compiling to do about 20 ms of arithmetic, and
repeated identical calls never got faster. With the cache the second call
is 105 ms.

## What the hardware is and is not for

Per matvec one H200 is faster than a v5e; end to end the v5e is ahead, because
a relaxation step is a chain of roughly 250 000 sequentially dependent kernels
and the v5e executes the whole `lax.scan` body as a single on-device program.
This workload suits a TPU. The numbers are in
`docs/research/tpu_v5e_benchmark.md`.

Getting there took three changes, and two of the three are library changes
rather than configuration: the compilation cache above; removing every index
tensor from the mass kernel, since the gather and the scatter at its two ends
are both separable shift maps and so pure data movement with sources and
destinations known at compile time; and folding the sum factorization's y and
z stages into one contraction, which trades 1.5x the arithmetic for one fewer
stage and is worth 1.2-1.7x on every backend, GPU and CPU included.

The index tensors are the transferable part. At `(12,24,12)` p=3 the gather
costs 1.624 ms indexed and 0.049 ms structured, the scatter 2.011 ms and
0.060 ms; on the indexed forms a v5e is 23x and 5x *slower* than the VM's own
CPU, and on the structured forms 2.8x and 5x faster. Identical work, identical
answers. **If a kernel is slow on this hardware, look for an index tensor
first.**

Two things that surprised us and are worth knowing before you profile. The cost
was never compilation -- `relaxation_loop` is already `@jax.jit` around a
`lax.scan`, so the time was pure device execution, and the per-apply arithmetic
was right while the apply *count* was not. And apply count is a
backend-independent property, so the preconditioner work it led to helps CPU and
GPU by the same factor; a TPU only made it visible. A `v5litepod-4` is also four
chips where MRX uses one, so a parameter sweep takes all four through `jax.pmap`
with no library change, measured at 3.99x (`scripts/pmap_sweep.py`, which needs
only two devices and so works on a multi-GPU node too).

float32 on the MXU is not a numerical concern: inverse-mass CG at the same
tolerance takes the same iteration count on both backends (20 at k=1, 24 at
k=2).

Benchmark a solver before committing a long run, and read the two calls
separately, because the first is XLA compiling and the second is the device
computing:

```bash
SCRIPT=scripts/benchmark/relaxation_bench.py OUTDIR=outputs/bench \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --skip-relax --out outputs/bench/tpu.json
# again with RUN_PLATFORM=cpu, then
python scripts/benchmark/relaxation_bench.py \
  --compare script_outputs/bench/{tpu,cpu}.json
```

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

`zones.sh` caps persistent data disks at `MAX_DATA_DISKS` (2) because sweeping
zones otherwise leaves one behind in each. Delete the surplus with
`gcloud compute disks delete my-data-disk --zone=<zone>`.
