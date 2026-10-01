# Running on a Cloud TPU

`experimental/tpu/` is the TPU counterpart to `slurm/`. A TPU node does not exist until
you create it and bills until something deletes it, so these scripts
provision the hardware from your laptop as well as run on it, and
`experimental/experimental/tpu/idle_reaper.sh` runs on every node.

| script | role |
|---|---|
| `zones.sh` | shared config: the candidate ladder, failure classification |
| `check_quota.sh` | read-only quota preflight |
| `acquire_tpu.sh` | acquire a node and run on it (`--once`, `--acquire-only`, `--queue`) |
| `run_on_tpu.sh` | drive one session on a node you already hold |
| `startup.sh` | builds the environment on the node (runs there) |
| `idle_reaper.sh` | deletes the node when it goes idle (runs there) |
| `gcs_cache_smoke.py` | checks a GCS compilation cache before you rely on it |

## Before anything else

```bash
gcloud auth login
gcloud config set project <project>
cd tpu && ./check_quota.sh          # read-only, ~10 s
```

Quota is the one failure retrying cannot fix, and it is a ceiling, not an
allocation: `OK` means a create is permitted, not that hardware is free.
v5e is reachable only through the Cloud TPU API (`gcloud compute tpus
tpu-vm`), v5p through Compute Engine. `zones.sh` records which candidate
takes which API.

A flex-start create blocks while Dynamic Workload Scheduler waits for
capacity, and Ctrl-C does not cancel the server-side request.
`acquire_tpu.sh` therefore never queues by default: every attempt fails
fast, and a success is a real VM. `--queue` adds standing requests through
the Queued Resources API. The first request granted wins and every other
one is cancelled then, and all of them are cancelled when the daemon exits.
After a `kill -9`, check `gcloud compute tpus queued-resources list`
yourself.

## Getting a node and running on it

```bash
VM_NAME=mrx-tpu ./acquire_tpu.sh --acquire-only

SCRIPT=scripts/tutorials/3_li383_newton.py \
  OUTDIR=outputs/tutorials/li383_newton \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_TIMEOUT=7200 \
  ./run_on_tpu.sh --ns 12,24,24 --p 3

# the reaper does this after 20 idle minutes. v5e is a TPU API node, v5p a GCE instance
gcloud compute tpus tpu-vm delete mrx-tpu --zone=<zone>
gcloud compute instances delete mrx-tpu --zone=<zone>
```

| variable | meaning | default |
|---|---|---|
| `SCRIPT` | path relative to the mrx checkout on the VM | |
| `OUTDIR` | directory the script writes, pulled back afterwards | |
| `VM_NAME`, `ZONE` | the node to use | |
| `RUN_PLATFORM` | `tpu` or `cpu`, to run a stage on the host | `tpu` |
| `RUN_DTYPE` | `float32` or `float64` | `float32` |
| `RUN_TIMEOUT` | seconds | 7200 |
| `MRX_BRANCH` | branch checked out before the run | see `zones.sh` |

Everything after `run_on_tpu.sh` is passed to the script. The job runs
detached under `setsid` into a log that is streamed back, because a dropped
`gcloud ssh` re-runs its command and a second process on the chip fails
with `The TPU is already in use`.

`zones.sh` names the failures: `STOCKOUT`, `NOT_ALLOWLISTED` (wrong API),
`QUOTA`, `NO_SUBNET`, `POLICY`, `DISK_INCOMPATIBLE` (retried once without the
data disk) and `TRANSIENT` (retried once).

`startup.sh` is idempotent: it mounts the data disk at `/mnt/data` (the boot
disk without one), installs Miniforge and `jax[tpu]`, clones `MRX_BRANCH`
and writes `/mnt/data/.mrx_env_ready` once a smoke test passes. A cold
build takes minutes, and a warm data disk skips to the sentinel.

## Precision

A TPU has no float64: `run_on_tpu.sh` and `startup.sh` set
`MRX_DTYPE=float32` and `MRX_RESIDUAL_DTYPE=float32`, the package default
(`docs/source/concepts/precision.md`). `mrx.precision` sets
`jax_default_matmul_precision=highest`, so the MXU does not drop to
bfloat16 (lower settings fold the li383 map). A stage that needs float64,
such as a Poincare trace, runs on the host CPU of the same node with
`RUN_PLATFORM=cpu RUN_DTYPE=float64`.

## Compilation cache

`run_on_tpu.sh` sets

```bash
JAX_COMPILATION_CACHE_DIR=/mnt/data/jax_cache
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1
```

The thresholds matter as much as the directory: the defaults skip nearly
every kernel MRX compiles, so without them each new process compiles
everything again. `JAX_CACHE_DIR` also takes a `gs://` path (the bucket in
the node's region). Run `experimental/tpu/gcs_cache_smoke.py` against it first, since
without `etils[epath,epath-gcs]` JAX silently reads and writes nothing.

## Cost

A Cloud TPU API node has no `--max-run-duration`, so `experimental/experimental/tpu/idle_reaper.sh`
deletes it after 20 minutes with nothing running, no login session and no
accelerator held. Audit after every session:

```bash
gcloud compute tpus tpu-vm list --zone=-
gcloud compute instances list
gcloud compute disks list
```

`zones.sh` caps persistent data disks at `MAX_DATA_DISKS` (2). Delete the
surplus with `gcloud compute disks delete <disk> --zone=<zone>`.
