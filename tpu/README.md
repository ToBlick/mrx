# MRX on Cloud TPUs - quickstart

Scripts for getting a Google Cloud TPU and running MRX on it.
Read [TPU_GUIDE.md](TPU_GUIDE.md) before your first run; it is short and it
covers the traps that are not obvious from the Google documentation.

This directory lives in the mrx repository as `tpu/`, the TPU counterpart to
`slurm/`, and is also distributed standalone as `tpu_access_kit.zip` (build it
with `./make_kit.sh`). Every script resolves its own directory, so both copies
run unchanged. The TPU is provisioned from your laptop, not from a login node:
these scripts create the machine as well as run on it.

## Prerequisites

- `gcloud` installed and authenticated: `gcloud auth login`
- A default project set: `gcloud config set project YOUR_PROJECT`
- The Compute Engine and Cloud TPU APIs enabled on that project
- `rg` (ripgrep) on your machine - the scripts use it for log classification
- bash 3.2 is enough (the scripts avoid bash 4 features so they run on stock macOS)

## Quickstart

```bash
# 1. Read-only preflight: quota, subnets, machine/accelerator availability (~40 s)
./check_quota.sh

# 2. Keep trying until hardware lands. Capacity is scarce; this can take a while.
VM_NAME=mrx-tpu ./acquire_tpu.sh --acquire-only

# 3. Run the Poisson example (builds the environment on first use)
VM_NAME=mrx-tpu ZONE=<zone from step 2> ./run_on_tpu.sh --n 6 8 --p 2

# 4. DELETE IT. Whichever path won, it bills until you do. A v5e is a Cloud
#    TPU API node; v5p and v6e are GCE instances. Run both, one will say
#    "not found", and that is the cheap outcome.
gcloud compute tpus tpu-vm delete mrx-tpu --zone=<zone>
gcloud compute instances delete mrx-tpu --zone=<zone>
```

Results land in `script_outputs/`.

## Running your own MRX script

```bash
SCRIPT=scripts/tutorials/li383_relaxation.py \
  OUTDIR=outputs/tutorials/li383_relaxation \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_TIMEOUT=7200 \
  ./run_on_tpu.sh --ns 12,24,12 --p 3
```

`SCRIPT` is relative to the mrx repo root on the VM, `OUTDIR` is the directory
that script writes into, and everything after `run_on_tpu.sh` is passed through
to the script. Add `RUN_PLATFORM=cpu RUN_DTYPE=float64` to run a stage on the
host CPU in double precision.

## Benchmarking a solver before you commit a long run

```bash
PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --skip-relax --out outputs/bench/tpu.json
# same again with RUN_PLATFORM=cpu, then:
python tpu_bench_mrx.py --compare script_outputs/bench/{tpu,cpu}.json
```

Every row is timed twice and the first and second calls are reported
separately, because that is what separates XLA compiling from the device
computing. Getting this backwards is what made the relaxation solver look
unusable on a v5e for a while; see section 6.1 of the guide.

## Five things that will save you a day

1. **Quota is not the same as permission.** v5e has 512 chips of quota on this
   project and still returns `403 ... not allowed to use the machine type`
   through Compute Engine. It is reachable only through the Cloud TPU API.
   See section 2 of the guide.
2. **A hanging `gcloud compute instances create` is not hung.** `FLEX_START`
   queues the request for up to `--request-valid-for-duration`. Use `0` to fail
   fast; these scripts do.
3. **Nothing deletes a Cloud TPU API node for you.** Audit after every
   session, on both surfaces, because a v5e is not a GCE instance:

   ```bash
   gcloud compute instances list
   gcloud compute tpus tpu-vm describe mrx-tpu --zone=<zone>
   ```

4. **Run with the XLA compilation cache on.** `run_on_tpu.sh` sets it for you.
   MRX's inner solves are eager `lax.while_loop`s, so without a cache XLA
   recompiles them on every call: one `apply_laplacian` cost 10 s of compiling
   to do 20 ms of work. The cache takes that to 105 ms, a 93x difference, and
   it is the single largest effect measured on this hardware.
5. **Indexed access is what a TPU is bad at, not arithmetic.** Replacing the
   mass kernel's gather and scatter with dense shifted reads and adds took them
   from 1.624 ms and 2.011 ms to 0.049 ms and 0.060 ms, which is the difference
   between 5-23x *slower* than the VM's own CPU and 3-5x *faster*. If something
   is slow here, look for an index tensor before you blame the hardware.

## What is in `results/`

The measured output from the session the guide describes, so you can compare
against a known-good run before trusting your own:

| File | What it is |
|---|---|
| `benchmark_v5e_vs_cpu.md` | v5e against the same node's host CPU, phase by phase, and the fixes that were tried and refuted |
| `poisson_summary.md`, `poisson_results.json` | Toroidal Poisson on v5e vs the CPU float32 reference |

The figures from that session ship in the standalone kit but are not committed
to the mrx repository, so `results/` here may or may not contain them. Either
way you generate your own: the 100-step li383 run writes `trace.png` (force and
energy) and `torus_pw.png` (weak pressure on the torus), and
`poincare_relax.py` writes `poincare_ic_zeta0.png` and
`poincare_final_zeta{0,0.25,0.5}.png`. Section 9 of the guide describes what
each should look like, which is what you actually need to check yours against.

## Files

| File | Purpose |
|---|---|
| `TPU_GUIDE.md` | The full guide. Start here. |
| `zones.sh` | Candidate ladder and failure classification (shared config) |
| `check_quota.sh` | Read-only preflight |
| `probe_capacity.sh` | Spot-create probes to test actual capacity |
| `launch_tpu.sh` | Fail-fast launcher walking the ladder |
| `acquire_tpu.sh` | Daemon that retries until hardware lands |
| `run_on_tpu.sh` | Drives one session: wait, run, pull results |
| `startup.sh` | VM startup script that builds the conda environment |
| `watch_request.sh` | Non-blocking status of a queued request |
| `mrx_tpu_report.py` | Poisson driver that checks TPU vs CPU reference |
| `tpu_bench_mrx.py` | Phase/primitive benchmark; splits compile from execute |
| `profile_top_ops.py` | Reduces a `jax.profiler` trace to a top-N op table |
| `gcs_cache_smoke.py` | Proves a `gs://` compilation cache path before you rely on it |
| `make_kit.sh` | Builds `tpu_access_kit.zip`, the standalone copy of this directory |
