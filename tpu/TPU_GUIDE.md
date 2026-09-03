# Running MRX on Google Cloud TPUs

A practical guide for the group. It is organised around the things that went
wrong, because the happy path is three commands and the traps cost days.

Everything here was measured on project `mrx-scaling-on-tpus` in
August-September 2026. Where a number or an error string appears, it is quoted
from an actual run rather than from documentation.

---

## TL;DR

```bash
./check_quota.sh                 # read-only preflight, ~40 s
./acquire_tpu.sh --acquire-only  # keeps trying until hardware lands
ZONE=<zone> ./run_on_tpu.sh      # builds the env, runs MRX, pulls results
# ALWAYS. v5e is a TPU API node, v5p/v6e are GCE instances; run both.
gcloud compute tpus tpu-vm delete mrx-tpu --zone=<zone>
gcloud compute instances delete mrx-tpu --zone=<zone>
```

Two things to take away before anything else:

1. **A Cloud TPU API node has no `--max-run-duration` and bills until you delete
   it.** Nothing will stop it for you. See [Cost discipline](#7-cost-discipline).
2. **Always run with the compilation cache on, which `run_on_tpu.sh` now does.**
   Without it the relaxation solver could not finish its setup phase in 40
   minutes on a v5e; with it, setup takes about 46 s and a step costs 3.4 s
   against the same VM's CPU at 12.7 s. The cause was XLA recompiling the same
   inner solve on every call, not the device being slow. Read
   [section 6.1](#61-what-actually-runs-well-and-what-does-not) before you plan
   a campaign: it also shows why replacing indexed gathers and scatters with
   dense shifted reads and adds turned this hardware from 5-23x slower than the
   VM's own CPU into 3-5x faster on the same operation. A v5e is **not**
   behind a datacentre GPU on this workload; one H200 running the same script
   is slightly slower per step.
3. **The 12 s per step was ~28 700 operator applies, not slow hardware.**
   Three MINRES solves were 99.3% of them and one was 85% on its own. That
   turned out to be a preconditioner chosen for the wrong term -- the operator
   is Laplacian-dominated where a comment claimed it was mass-dominated -- and
   fixing it was worth 1.62x on the whole relaxation. It is identical on a CPU
   and a GPU, so this is not a TPU fix.
   `results/benchmark_v5e_vs_cpu.md` has the numbers.
4. **A `v5litepod-4` is four chips and one equilibrium uses one of them.** A
   parameter sweep can have all four for the cost of a `jax.pmap`: measured
   **3.99x on real chips**.

---

## 1. The thing that will confuse you first

The setup doc tells you to run something like this:

```bash
gcloud compute instances create my-tpu-vm \
  --zone=us-east5-b --machine-type=ct6e-standard-4t \
  --provisioning-model=FLEX_START --request-valid-for-duration=2h ...
```

and it appears to hang. It is not hung. `FLEX_START` is Dynamic Workload
Scheduler: if no capacity is free, the instance is created in `PENDING` and the
scheduler waits up to `--request-valid-for-duration` for hardware. `gcloud`
blocks for the whole window. With `2h` you have told it to block for two hours.

Three consequences worth internalising:

- Your terminal is not broken, and Ctrl-C does not cancel the request. The
  request lives server-side. Check it with `./watch_request.sh`, or
  `gcloud compute instances list` (it shows as `PENDING`).
- Cancelling a `PENDING` flex-start instance is slow. One took **39 minutes**
  to delete. Letting a short window expire is usually faster than cancelling.
- Google exposes **no queue position and no global capacity view**. There is no
  way to ask "how long will I wait". `gcloud alpha compute advice capacity`
  requires Compute Alpha allowlisting, and `gcloud beta compute advice
  calendar-mode` rejects the TPU machine families outright.

Use `--request-valid-for-duration=0` to make the request fail immediately
instead of queuing. That is what `launch_tpu.sh` does on every attempt, so a
sweep across zones takes seconds per zone rather than hours.

---

## 2. There are two different APIs, and the choice is not yours

This is the single biggest time sink, so it gets its own section.

TPUs can be created two ways, and they are genuinely different services:

**Compute Engine** treats the TPU as a machine type:

```bash
gcloud compute instances create NAME --machine-type=ct5p-hightpu-4t ...
```

**Cloud TPU API** treats it as its own resource:

```bash
gcloud compute tpus tpu-vm create NAME --accelerator-type=v5litepod-4 \
    --version=tpu-ubuntu2204-base
```

On this project each generation is reachable through exactly one of them:

- **v5e** only through the **Cloud TPU API**. Through Compute Engine it returns

  ```
  HTTPError 403: This user agent is not allowed to use the machine type
  [ct5lp-hightpu-4t].
  ```

  while the project holds **512 chips** of `TPU-LITE-PODSLICE-V5` quota in that
  very region. The allowlist for the TPU-as-GCE-instance path is a **separate
  gate from quota**, it is not visible in any read-only API, and no amount of
  quota will open it.

- **v5p** works through **Compute Engine** (`ct5p-hightpu-4t`), with 768 chips
  of quota. It has been in continuous stockout, but it is permitted.

- **v6e** is effectively unavailable: the `CT6E` limit is a hard **0.0** in
  us-east5 and us-east4, and every other reachable zone has been stocked out.

The practical rule: **if you have quota and creates still fail, check which API
you are using before you ask for more quota.** The scripts here encode the
mapping in `zones.sh` as a `gen:type:zone:model:api` ladder, so you do not have
to remember it.

The two APIs also differ in ways that bite later:

- SSH is `gcloud compute tpus tpu-vm ssh`, not `gcloud compute ssh`.
- Listing needs `--zone`; `gcloud compute tpus tpu-vm list` **errors out**
  without one, and with `--zone=-` it prints short names with no zone attached.
- The healthy state is `READY`, not `RUNNING`.
- There is no `--max-run-duration`. See section 7.

---

## 3. Reading quota correctly

Do not read `TPUS-PER-TPU-FAMILY`. It is a poor guide: it reads "unset" both for
regions that permit creates and for regions that hard-fail at zero. Read the
per-generation bucket instead:

- v5e: `TPU-LITE-PODSLICE-V5-per-project-region`
- v5p: `TPU-V5P-per-project-region`
- v6e: `PREEMPTIBLE-TPU-V6E-per-project-region`

```bash
gcloud beta quotas info describe TPU-LITE-PODSLICE-V5-per-project-region \
    --service=compute.googleapis.com --project=YOUR_PROJECT --format=json
```

`--project` is **mandatory**; without it the command fails with
`Exactly one of (--folder | --organization | --project) must be specified`
even when a default project is configured.

`check_quota.sh` does all of this and adds the two other preconditions that
silently kill creates: whether the region has a subnet in the default VPC, and
whether the type is actually offered in that zone. It caches to
`.quota_cache.json` and takes about 40 seconds; a blind zone sweep to learn the
same thing took 25 minutes.

**Quota is not capacity.** Having 512 chips of quota tells you nothing about
whether a chip is free right now. Conversely a `QUOTA_EXCEEDED` rejection is
mildly good news: the request got past the capacity check first.

### Org policy

`constraints/gcp.resourceLocations` restricts this project to US locations.
Several non-US regions do carry TPU quota, but they are unreachable: the
auto-mode default VPC has no subnet there, and the create dies at network
validation with `No default subnetwork was found` long before capacity is
consulted. That error looks like "unsupported machine type" if you are not
looking carefully, which is why `zones.sh` classifies `NO_SUBNET` and `POLICY`
as distinct failure kinds.

---

## 4. Precision: the part that actually affects your physics

TPUs have no usable float64. MRX defaults to float64, so you must switch it:

```bash
export MRX_DTYPE=float32
```

MRX reads `MRX_DTYPE` at import time in `mrx/precision.py`, so it has to be set
**before** `import mrx`. The scripts here set it in `/etc/profile.d/mrx.sh` on
the VM.

Critically, `mrx/precision.py` also sets

```python
jax.config.update("jax_default_matmul_precision", "highest")
```

Without this the TPU's MXU silently truncates float32 matmul inputs to bfloat16.
That is not a rounding detail: on W7-X geometry the map derivative `dR/dtheta`
comes out **19% wrong**, and `det DF` can go negative, which corrupts the
geometry rather than merely degrading it. `mrx_tpu_report.py` asserts the
setting at startup so a regression cannot pass silently.

With float32 and `highest`, the toroidal Poisson tutorial on v5e reproduced the
CPU float32 reference to 0.00% (section 9), so float32 on TPU is trustworthy for
this class of problem.

The full setting has three values and only two are usable. `highest` is the
default and what everything here was measured with. `high` is the floor: it buys
roughly 1.2x on a relaxation step, at the cost of the adaptive stepper taking a
different trajectory, so it is opt-in for someone who knows what they are
trading. `default` is not an option at all -- it folds the geometry map, driving
`det DF` negative, and the run fails rather than quietly degrading. The
measurements behind all three are in `results/benchmark_v5e_vs_cpu.md`.

Where float64 still
matters, run that stage on the host CPU instead:

```bash
JAX_PLATFORMS=cpu MRX_DTYPE=float64 python -u scripts/poincare_relax.py ...
```

The TPU VM host has plenty of cores, and field-line tracing is cheap relative to
the relaxation, so this costs little and keeps the accuracy-sensitive stage in
double precision.

### One process at a time

A TPU can be held by exactly one process. A second one gets:

```
ABORTED: The TPU is already in use by process with pid 23472.
```

This bites whenever a driver script imports JAX and then shells out to another
script that also uses JAX. `mrx_tpu_report.py` runs each JAX phase in its own
short-lived subprocess for exactly this reason.

---

## 5. Getting the environment onto the machine

`startup.sh` runs as the VM's startup script and is idempotent. It:

1. mounts `my-data-disk` at `/mnt/data` when one is attached, and otherwise
   falls back to a boot-disk directory at the same path (losing persistence but
   not the run);
2. installs Miniforge, creates a `python=3.12` conda env, installs `jax[tpu]`;
3. clones or fast-forwards `mrx` on the `static-dynamic-refactor` branch;
4. `chmod 1777 /tmp/tpu_logs`;
5. writes `/mnt/data/.mrx_env_ready` as a sentinel and smoke-tests JAX + MRX.

A cold build takes about 4-12 minutes. A warm data disk skips to the sentinel
and starts computing within a minute. It also `chmod -R a+rwX`es the repo, for
the reason in section 6.2.

Two details worth knowing:

- **Use the `static-dynamic-refactor` branch.** `main` is stale and its
  Laplacian assembly raises a reshape `TypeError` inside the k=0 tensor Hodge
  preconditioner.
- **`chmod 1777 /tmp/tpu_logs` is not cosmetic.** The directory is created by
  root, so an ordinary user cannot write there and libtpu emits
  `Could not open the log file ... Permission denied` several times per second,
  which buries real tracebacks. This hid a genuine error for an entire debugging
  cycle. Belt and braces: also set `TPU_STDERR_LOG_LEVEL=3`.

---

## 6. Running things

```bash
# the Poisson regression driver
ZONE=us-south1-a ./run_on_tpu.sh --n 6 8 --p 2

# any script in the mrx repo, on the TPU
SCRIPT=scripts/tutorials/li383_relaxation.py \
  OUTDIR=outputs/tutorials/li383_relaxation \
  ZONE=us-south1-a RUN_TIMEOUT=7200 ./run_on_tpu.sh --ns 12,24,12 --p 3

# the same thing on the VM's host CPU, for a like-for-like comparison
SCRIPT=scripts/tutorials/li383_relaxation.py \
  OUTDIR=outputs/tutorials/li383_cpu RUN_PLATFORM=cpu RUN_DTYPE=float32 \
  ZONE=us-south1-a RUN_TIMEOUT=10800 ./run_on_tpu.sh --ns 12,24,12 --p 3

# a float64 stage on the host CPU
SCRIPT=scripts/poincare_relax.py OUTDIR=outputs/tutorials/li383_relaxation \
  RUN_PLATFORM=cpu RUN_DTYPE=float64 ZONE=us-south1-a ./run_on_tpu.sh \
  outputs/tutorials/li383_relaxation/B.h5 --planes 0,0.25,0.5
```

`RUN_PLATFORM=cpu` is not a fallback for when things break; it is a measurement
tool. Running both on the same VM is the only way to know whether the TPU is
actually helping your workload, and it costs nothing extra. Section 6.1 is what
that comparison produced here.

Mind the timeouts. They interact, and they are ordered deliberately:

- `RUN_TIMEOUT` (default 7200 s) bounds the remote command.
- `SETUP_TIMEOUT` (default 2400 s) bounds the environment build.
- `MAX_RUN_DURATION` (default `4h`, `zones.sh`) is the **GCE** self-termination
  window, paired with `--instance-termination-action=DELETE`. It does not merely
  stop the job at expiry, it **deletes the VM**. Keep it above
  `SETUP_TIMEOUT + RUN_TIMEOUT` or you will lose a long run and its output.
  The Cloud TPU API path ignores this entirely.

### 6.1 What actually runs well, and what does not

This is the finding that should shape how the group uses the allocation, and it
is the one the Google documentation cannot tell you.

**The toroidal Poisson tutorial is a good fit.** Dense, regular, matmul-shaped
work. It ran in 107 s at `--n 6 8 --p 2` and reproduced the CPU float32
reference to 0.00%.

**The relaxation solver was initially unusable on v5e, and four changes fixed
it.** An earlier version of this guide said the relaxation was simply the wrong
shape for the hardware. That was wrong, and the way it was wrong is worth
knowing, because the same mistake is easy to repeat: the run was not slow, it
was *compiling*, over and over, and no amount of staring at phase wall-clock
times distinguishes those two.

The measurements are in `results/benchmark_v5e_vs_cpu.md` and are not repeated
here; that document is where they are kept current, and this one tells you what
to do. The short version: five changes took the v5e from **1.7x slower than the
VM's own CPU to 3.7x faster**, with setup down from seven minutes to about 46
seconds. Roughly 2.4x of that was configuration -- the compilation cache below
-- and the rest was changes inside `mrx/` itself.

An H200 is about 4x faster per matvec and slightly *slower* per step, because a
step is a chain of ~250 000 sequentially dependent kernels and a TPU executes
the whole scan body as one on-device program. (An earlier version of this guide
claimed the v5e was "32x behind a single H100". That came from a warm-start A/B
on different geometry and was never a like-for-like comparison.)

The rest of this section is the part you have to act on: the cache settings, and
the one lesson that generalises to any kernel you write.

**Fix 1, and it is almost all of the win: turn on the persistent compilation
cache.** MRX's inner solves run as eager `jax.lax.while_loop`s. Nothing wraps
them in `jax.jit`, so every call traces a fresh closure, XLA sees a program it
has never seen, and compiles it again. On a CPU that is a nuisance. On a v5e,
where compilation is far more expensive, one `apply_laplacian` k=1 call cost
about 10 s of compiling to do 20 ms of arithmetic. The tell was unmistakable
once looked for: repeated identical calls that never got faster.

`run_on_tpu.sh` now sets this for you:

```bash
JAX_COMPILATION_CACHE_DIR=/mnt/data/jax_cache
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1
```

The two thresholds matter as much as the directory. Their defaults are tuned
for a few large training programs and would skip almost every kernel this
workload compiles. Point the directory at the data disk and the cache survives
the node, so the *next* session starts warm.

A node in a zone with no data disk has nowhere local to keep it. `JAX_CACHE_DIR`
also accepts a `gs://` path, which works but is second best:

| setup = `build_sequence` + `compute_nullspaces` | v5e |
|---|---|
| no cache | 143 s |
| `gs://` cache, cold (writing 177 entries) | 210 s |
| `gs://` cache, warm | 98 s |
| local data disk, warm | **53 s** |

Two traps if you use a bucket. Put it in the node's own region, or every miss
is a cross-region round trip. And note that **JAX does not fail on a `gs://`
path it cannot use**: without `etils[epath,epath-gcs]` installed it writes
nothing, reads nothing and reports nothing, so the run is simply slow. That
cost an hour here. `startup.sh` now installs it.

There is no longer a smoke test for this, so check it by hand the first time a
bucket is used: run anything twice and watch whether the second run is faster,
and confirm the bucket is non-empty with `gcloud storage ls gs://<bucket>/mrx`.
An empty bucket after a run means the cache is not being written, and the only
symptom you will otherwise get is a run that stays slow.

**Fixes 2 and 3: the mass kernel holds no index tensors at all.** The
sum-factorised kernel began with a gather, `x[gather_idx]`, and ended with a
`jax.ops.segment_sum`. Indexed access is the one thing a TPU has no fast path
for. Both index plans come from `_flat_dof_plan`, a separable tensor product of
the per-axis DoF ids, and on a tensor-product B-spline axis
`g[e, l] = (e + l) mod n`. So both are algebraically pure data movement with
every source and destination known at compile time: rolled slices one way,
shifted dense adds the other.

| at (12,24,12) p=3 | v5e indexed | v5e structured | CPU indexed | CPU structured |
|---|---|---|---|---|
| gather, 3456 dofs read 221k times | 1.624 ms | **0.049 ms** | 0.070 ms | 0.113 ms |
| scatter, 221k contributions | 2.011 ms | **0.060 ms** | 0.398 ms | 0.303 ms |

**That table is the most useful thing in this guide.** On the indexed forms the
v5e is 23x and 5x *slower* than the VM's own CPU; on the structured forms it is
2.8x and 5x *faster*. Identical work, identical answers -- exactly for the
gather, to 3.4e-07 in float32 for the scatter. Only the access pattern changed.
If your kernel is slow on a TPU, look here first.

`mass.py` checks the shift structure per axis and falls back to the index
tensors if any axis fails, so a basis built differently still assembles
correctly. When the plan holds the index tensors are dropped from the kernel
signature rather than left unused, which is also what removed the `Constant
folding an instruction is taking > 1s` warnings.

**Fixes 4 and 5** removed the two remaining index-shaped costs. The extraction
operator's gather and `segment_sum` are now compiled as one program instead of
dispatched as two eager ops, and the sum factorization's y and z stages are
folded into a single wider contraction, trading 1.5x the arithmetic for one
fewer stage and returning 1.2-1.7x on every backend, GPU and CPU included.

**Read the write-up before optimising anything here**, for two reasons. It
records four hypotheses that sounded right and were refuted by measurement,
including the one that motivated the fold -- which was wrong about *why* the
fold would work, even though the fold itself won. And it records where the time
actually goes, which is not where anyone guessed: a step is about 28 700
operator applies, three MINRES solves are 99.3% of them, and the velocity
smoothing solve alone was 85% of a step until it was repreconditioned. Apply
count is backend-independent, so none of that opens or closes a gap between
machines, but it is where the remaining wins are.

**Four chips instead of one, measured at 3.99x.** A `v5litepod-4` is four chips
and MRX uses one. For a parameter sweep that needs no library change: the scan
is a pure function of an equinox `State`, so `jax.pmap` over a stacked batch of
initial states runs one equilibrium per chip. Measured that way on a real
`v5litepod-4`, four equilibria 1% apart at `(8,16,8)` p=3 float32 took **5.8 s
on four chips against 23.1 s one at a time**. Each
reproduced its own sequential answer to 5.0e-05 in float32, well inside the
3.5e-02 bar, and the members differ from each other by the 1-3% they were built
to differ by, which is the check that this is four problems and not four copies
of one.

Time the compilation separately or you will measure the compiler. Both forms
compile once; the serial loop then amortises it over four calls and the `pmap`
pays it on its only call, so a single-shot comparison of the two reads 2.77x,
and a first attempt that also ran with a cold XLA cache read 1.56x. Neither is
a property of the hardware. Time the second call of each form, which is what a
real sweep of any length gets.

One flag-sized item also remains: wrapping `apply_laplacian` in `jax.jit`
measured 4.5 ms against 75 ms eager, a 17x. It is not in the relaxation's hot
path, and it changes how the library is called rather than what it computes, so
it was not taken.

**Precision is not a concern.** Inverse-mass CG at the same tolerance took the
same iteration count on both backends (20 at k=1, 24 at k=2), so float32 on the
MXU is not degrading convergence.

**Practical guidance.** Never conclude "this hardware is wrong for this code"
from wall-clock phase times alone. Time everything twice: first call versus
second call separates compiling from computing, and those have completely
different remedies. `tpu_bench_mrx.py` does exactly this, and
`RUN_PLATFORM=cpu` gives you a same-VM control for free, since you are already
paying for the node.

```bash
# the whole diagnosis, ~12 min, on a node you already hold
PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --skip-relax --out outputs/bench/tpu.json

PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=cpu \
  ./run_on_tpu.sh --skip-relax --out outputs/bench/cpu.json

python tpu_bench_mrx.py --compare script_outputs/bench/{tpu,cpu}.json
```

When the benchmark says a phase is slow but not which operation inside it is,
`--profile` writes a `jax.profiler` trace. Reduced to XLA ops ranked by self
time it is the only table worth reading here, and it names the bottleneck
instead of inferring it from wall-clock phase times.

```bash
PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --skip-relax --profile outputs/bench/trace
```

Read the trace with TensorBoard, or with `tensorboard-plugin-profile`'s own
converter. That plugin is not part of JAX and its module path moves between
releases, which is worth knowing before you conclude the trace is unreadable.

### Finite-beta stellarator equilibria

Two steps. Relax on the TPU, then trace on the host CPU.

The split is not arbitrary. The descent is robust in float32 and that is the
production choice, so it can use the TPU, which since the section 6.1 fixes is
faster than this VM's CPU. Field-line tracing integrates for hundreds of
toroidal periods and accumulates error, so it wants float64, which a TPU does
not have; it is also cheap, 7 s per field, so there is nothing to gain.

Neither script runs the `lambda_1` diagnostic any more. It reports a number,
it is not part of the construction, and it cost 6.8 of the 9 setup minutes
when it was on by default.

```bash
OUT=outputs/tutorials/li383_relaxation

# 1. relax in float32 on the TPU -> B.h5, trace.png, torus_pw.png
#    (~13 s/step, so ~22 min for 100 steps, plus <1 min setup on a warm cache)
SCRIPT=scripts/tutorials/li383_relaxation.py OUTDIR=$OUT \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_PLATFORM=tpu RUN_DTYPE=float32 RUN_TIMEOUT=7200 \
  ./run_on_tpu.sh --ns 12,24,12 --p 3 --precision float32 \
                  --outer 10 --inner 10 --out $OUT

# 2. trace in float64 on the host CPU -> poincare_*.png, sections.npz  (~2 min)
SCRIPT=scripts/poincare_relax.py OUTDIR=$OUT \
  VM_NAME=mrx-tpu ZONE=<zone> RUN_PLATFORM=cpu RUN_DTYPE=float64 \
  ./run_on_tpu.sh $OUT/B.h5 --planes 0,0.25,0.5 --precision float64 --out $OUT
```

Raise `--outer` for a longer descent; `--floor-tol 1e-3` stops it early if the
force gets there. There is no wall-clock flag on this tutorial, so the step
count is your only control, which is why the calibration habit matters.

`--outer` is the number of host synchronisation points and `--inner` the
compiled steps between them, so `--outer 10 --inner 10` is 100 steps with 10
progress lines. Keep `--inner` fixed across runs: changing it changes the
`lax.scan` length, which is a new program to compile.

li383 (NCSX, nfp=3) ships with the repo as
`data/wout_li383_low_res_reference.nc`, whitelisted in an otherwise-ignored
`data/`. **W7-X is not in the repo.** Its GVEC state
(`data/GVEC_State_final.dat`) is a symlink on the cluster. Drop that file into
`data/` and the identical pipeline runs W7-X:

```bash
python -u scripts/relax.py --geometry data/GVEC_State_final.dat --ns 12,24,12 --p 3
```

`poincare_relax.py` reads the geometry path out of `B.h5`'s attributes as an
absolute path (`/mnt/data/mrx/data/...`), so it must run somewhere that path
resolves. That is the main reason to run the tracing stage on the VM rather than
pulling `B.h5` to your laptop first.

Run the relaxation in float32 and the tracing in float64. The descent is robust
in single precision and it is the production choice, whereas field-line tracing
integrates for hundreds of toroidal periods and accumulates error; the measured
half-step drift was 9.5e-07 in float64. Tracing is cheap, so the extra precision
costs almost nothing.

---

### 6.2 Two traps in driving long remote jobs

Both of these cost real time here, and neither is obvious.

**`gcloud ... ssh --command` silently retries, and can start your job twice.**
A dropped or slow connection makes `gcloud` reconnect and re-run the command.
The second copy dies immediately with `The TPU is already in use by process with
pid N` — while the first copy keeps running, detached, with its stdout going
nowhere. What you see is a confusing "already in use" error; what is actually
happening is that your job is running fine and you have lost its output. It also
means a dropped connection during a multi-hour run loses everything.

`run_on_tpu.sh` therefore launches remote work under `setsid`, redirects it to a
timestamped log on the VM, records the exit status in a marker file, and polls.
The job survives a dropped connection instead of being orphaned by one. If you
drive a job by hand, do the same:

```bash
setsid bash -c 'python -u script.py > /mnt/data/runs/job.log 2>&1' &
```

**The repo is cloned by root, so your outputs directory is not writable.**
`startup.sh` runs as root, so `/mnt/data/mrx` lands root-owned and the first
thing a tutorial does — `os.makedirs("outputs/...")` — fails with
`PermissionError: [Errno 13] Permission denied: 'outputs'`, several minutes into
the run. `startup.sh` now does `chmod -R a+rwX` on the repo after the clone.

## 7. Cost discipline

**Cloud TPU API nodes have no `--max-run-duration`.** A `v5litepod-4` runs
roughly USD 5-6/hour, and none of the things you would expect to stop it do: not
a job finishing, not your SSH session closing, not your laptop sleeping.

What stops it is `idle_reaper.sh`, which every node now installs from
`startup.sh`. It deletes the node after `IDLE_TIMEOUT_MIN` minutes (default 20,
set in `zones.sh`) with no environment python running, nobody logged in, and no
accelerator device held. Any one of those resets the clock, so it will not
interrupt a long run, a long think, or an SSH session left open. `0` installs it
without arming it.

Two things to know about it. It does not arm until the environment sentinel
exists, so a 10-12 minute cold build is not mistaken for idleness -- but that
gate expires after 45 minutes, because a node whose setup failed is precisely
the node that would otherwise bill forever. And it needs `tpu.nodes.delete` on
the node's service account; `startup.sh` checks that at install time and prints
a loud warning to the serial log if it is missing, so the failure is visible
rather than silent. Verify on a new node shape with:

```bash
gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone="${ZONE}" \
  --command='/usr/local/bin/mrx_idle_reaper.sh --check'
```

It prints which API it would call, with which node name and zone, whether the
sentinel is present and whether it currently reads the node as busy. It deletes
nothing.

The reaper is a safety net, not a substitute for deleting a node you are done
with. Twenty minutes of a v5e is still about two dollars.

Two real incidents from building these scripts, both worth learning from:

1. The daemon acquired a v5e node, checked for state `READY` exactly once while
   it was still `CREATING`, concluded nothing was available, logged "nothing
   available", and **walked away from a live billing TPU**. It was caught by
   reading the log. It now polls for up to four minutes.
2. The same daemon queried `tpus tpu-vm list --zone=-`, which prints short
   names, so its zone-extraction regex silently returned nothing and it could
   never recognise its own node.

Habits that prevent this:

```bash
# after every session, and any time you are unsure
gcloud compute instances list
for z in us-south1-a us-east5-a us-east5-b us-central1-a; do
  gcloud compute tpus tpu-vm describe mrx-tpu --zone=$z --format="value(state)" 2>/dev/null
done
```

Note also that a sweep can leave debris, and disks are the sneaky case because
they bill quietly forever while attached to nothing. An early version created a
100 GB data disk in *every* zone it probed; six of them. Even after fixing that
so a disk is created only in a zone that actually wins, four accumulated over
successive sessions, because capacity decides the zone and capacity moves.

Two guards now: `ensure_data_disk` refuses to exceed `MAX_DATA_DISKS` (default
2) across all zones, and `acquire_tpu.sh --gc` deletes unattached strays. Keep
the cap low. A cold environment build is only about four minutes, so a warm disk
in a zone you did not land in is worth far less than the rent.

```bash
gcloud compute disks list          # check this as routinely as instances list
./acquire_tpu.sh --gc              # delete unattached my-data-disk volumes
```

---

## 8. Failure messages, decoded

`zones.sh` classifies create failures. What each means:

- `STOCKOUT` - no hardware. `ZONE_RESOURCE_POOL_EXHAUSTED` or
  `Insufficient capacity`. Retry later or elsewhere; nothing you can fix.
- `NOT_ALLOWLISTED` - `user agent is not allowed to use the machine type`.
  Wrong API (see section 2), or you need an allowlist request. Quota is
  irrelevant here.
- `QUOTA` - quota exhausted. Capacity was probably fine.
- `NO_SUBNET` - the default VPC has no subnet in that region, usually a
  downstream effect of the location org policy.
- `POLICY` - blocked by `constraints/gcp.resourceLocations`.
- `DISK_INCOMPATIBLE` - `hyperdisk-balanced disk type cannot be used by
  ct5p-hightpu-4t machine type`. v5p rejects hyperdisk-balanced. The launcher
  retries once without the data disk, since losing persistence beats losing the
  zone.
- `TRANSIENT` - a Google-side internal error. Retried once automatically; it
  says nothing about capacity.

---

## 9. What a real run produced

Physics, not timings. Per-step and per-operator performance for every backend
is in `results/benchmark_v5e_vs_cpu.md`, which is the only place those numbers
are maintained; this section is what came out of the machine.

All measured on one `v5litepod-4` node (v5e, 4 chips) in us-south1-a, JAX 0.11.1,
Python 3.12. The host is 112 vCPU / 188 GB.

**Machine and environment**

- Device `TPU v5 lite`, `device_count 4`, `matmul_precision highest`
- Aggregate throughput **71.7 TFLOP/s** float32 (4096^2 pmapped matmul)
- Cold environment build (miniforge + `jax[tpu]` + mrx): about 4 minutes
- Acquisition: 6.5 minutes of sweeping 25 candidates before us-south1-a yielded

**Toroidal Poisson on the TPU** (`--n 6 8 --p 2`, float32), 107 s wall:

| p | n | TPU | CPU float32 reference | deviation | CG iters |
|---|---|---|---|---|---|
| 2 | 6 | 1.0754e-02 | 1.0754e-02 | 0.00% | 6 |
| 2 | 8 | 3.5617e-03 | 3.5617e-03 | 0.00% | 7 |

**li383 finite-beta relaxation** (`--ns 12,24,12 --p 3 --precision float32`,
100 steps as `--outer 10 --inner 10`), **on the v5e** with the
section 6.1 fixes:

- Geometry: nfp=3, `det DF` in [2.976e-01, 2.124e+00]
- Initial condition: `||div B|| = 1.8e-06`, wall-normal part 0.0
- Force residual `||F||`: 5.540e-02 -> 2.51e-02 -> 1.35e-02 -> **1.004e-02** (5.5x)
- Energy released `E_0 - E = 5.743e-05`
- Helicity `+5.002e-03 -> +5.002e-03`, `dH = +2.8e-09` (5.6e-07 relative)
- `||div B||` at the end: 1.9e-06
- Weak pressure on axis: 3.7304e-02 (in `||B||_M = 1` units)
- Cost of that run: 1.9 min setup, then 41.3 s/step; 72 minutes end to end

That run predates fixes 2-4, and the per-step timings for every backend live in
`results/benchmark_v5e_vs_cpu.md` rather than being repeated here. Read them
there with the staleness note at its headline: they were measured before the
development branch reformulated the smoothing solve, and are high.

**Poincare sections** (`poincare_relax.py`, `--planes 0,0.25,0.5`, float64 on the
host CPU, 160 field lines x 400 periods), 2 minutes end to end:

| Field | Lines lost | Chaotic | h/2 drift | iota range |
|---|---|---|---|---|
| Initial condition | 0 / 160 | 0 | 9.50e-07 | 0.4041 .. 0.6584 |
| Relaxed (100 steps, TPU) | 0 / 160 | 4 | 2.84e-05 | 0.4082 .. 0.6584 |

Both fields keep nested surfaces across all three planes, and the iota range
matches the 0.40-0.66 that li383 is documented to have. The four chaotic lines in
the relaxed field are consistent with a partial descent. Note the tracing stage
is *cheap*: 7 s per field. Essentially all of the 2 minutes is rebuilding the
sequence, which is the same setup cost as the relaxation.

Two honest caveats on the relaxation run. The residual is **not monotone**: it
rose to 7.76e-02 at step 10, above its starting value, before falling to
1.00e-02. That is expected for this descent, and the tutorial docstring is
explicit that "the force residual need not fall monotonically, what matters is
the floor it settles at". And 100 steps is a **partial descent**: the tutorial
documents a clean nested floor at roughly 1000 steps with velocity smoothing, a
few hours at the step times in the benchmark write-up. What is shown here is a
well-behaved 5.5x reduction in force with helicity conserved to 5.6e-07, not a
converged equilibrium.

The MRX solve is single-device matrix-free CG, so it occupies one chip; the
71.7 TFLOP/s figure is the headroom available to a sharded implementation, not
what the solve achieves.

---

## 10. Files

- `zones.sh` - shared config: the candidate ladder and failure classification
- `check_quota.sh` - read-only preflight; start here
- `probe_capacity.sh` - spot-create probes to test real capacity
- `launch_tpu.sh` - fail-fast launcher that walks the ladder
- `acquire_tpu.sh` - daemon that retries until hardware lands
- `run_on_tpu.sh` - drives one session: wait, run, pull results
- `startup.sh` - VM startup script that builds the environment
- `idle_reaper.sh` - deletes the node after `IDLE_TIMEOUT_MIN` idle minutes;
  the only self-termination a Cloud TPU API node has
- `watch_request.sh` - non-blocking status of a queued request
- `mrx_tpu_report.py` - Poisson driver with the CPU-reference check
- `tpu_bench_mrx.py` - phase and primitive benchmark; separates compile time
  from execute time, and has a `--compare` mode for the TPU/CPU table
- `matvec_bench.py` - times each operator apply eagerly, jitted, and inside a
  `lax.scan`, which is the form the relaxation actually runs
- `make_kit.sh` - builds `tpu_access_kit.zip`, the standalone copy of this
  directory

### Environment overrides

Which script reads what. Anything not listed is internal.

| Variable | Read by | Default | What it does |
|---|---|---|---|
| `VM_NAME` | all | `my-tpu-vm` | the node's name; the examples here use `mrx-tpu` |
| `ZONE` | `run_on_tpu.sh` | auto-detected | skips the GCE-then-TPU-API lookup |
| `GENERATIONS`, `ZONES`, `MACHINE_TYPE`, `MODELS`, `APIS` | `zones.sh` | unset | restrict the candidate ladder |
| `MAX_RUN_DURATION` | `zones.sh` | `4h` | GCE self-termination; must exceed `RUN_TIMEOUT` |
| `DATA_DISK`, `DATA_SNAPSHOT` | `zones.sh` | `my-data-disk`, `my-data-snapshot` | the persistent environment |
| `MAX_DATA_DISKS` | `zones.sh` | 2 | refuses to create more, after a sweep once left five |
| `IMAGE_PROJECT`, `IMAGE_FAMILY`, `TPU_RUNTIME` | `zones.sh` | see file | boot image per API path |
| `RUN_TIMEOUT` | `run_on_tpu.sh` | 7200 | bounds the remote command |
| `SETUP_TIMEOUT` | `run_on_tpu.sh` | 2400 | bounds the first-boot environment build |
| `POLL_SECONDS`, `RUN_POLL` | `run_on_tpu.sh` | 20 | how often setup and the run are checked |
| `RUN_PLATFORM` | `run_on_tpu.sh` | unset | `cpu` forces the host CPU, for float64 stages |
| `RUN_DTYPE` | `run_on_tpu.sh` | `float32` | sets `MRX_DTYPE` remotely |
| `TPU_API` | `run_on_tpu.sh` | `auto` | 1 for a Cloud TPU API node, 0 for a GCE instance |
| `JAX_CACHE_DIR` | `run_on_tpu.sh` | `/mnt/data/jax_cache` | persistent XLA cache; a `gs://` path works |
| `SCRIPT`, `OUTDIR`, `PUSH_FILES` | `run_on_tpu.sh` | unset | run something other than the Poisson report |
| `SYNC_LOCAL_MRX`, `LOCAL_MRX` | `run_on_tpu.sh` | 0 | overlay an uncommitted working tree |
| `SWEEP_INTERVAL`, `MAX_HOURS`, `MAX_SESSIONS` | `acquire_tpu.sh` | 180, 12, 3 | daemon pacing and stopping conditions |
| `LOCK_FILE`, `ACQUIRE_LOG` | `acquire_tpu.sh` | `.acquire.lock`, `acquire.log` | single-instance guard and log |
| `MAX_PARALLEL` | `probe_capacity.sh` | 6 | concurrent probes |
| `KEEP_LOGS` | `probe_capacity.sh` | 0 | keeps the per-probe logs the trap would delete |
| `CACHE_FILE`, `CACHE_TTL` | `check_quota.sh` | see file | caches the quota read |
| `PROJECT` | `zones.sh` | `gcloud config` | the project every call is scoped to |

### Iterating on mrx itself

`SYNC_LOCAL_MRX=1` overlays your local working tree onto the VM's checkout, so
a fix can be measured on real hardware before it is committed anywhere. Only
git-tracked files are sent: the full directory is 568 MB of untracked HDF5
output and notebooks, while the 170 tracked files are 1.7 MB. The run log
records the local SHA and how many tracked files are modified, so every
measurement is attributable to a known tree.

```bash
SYNC_LOCAL_MRX=1 LOCAL_MRX=~/mrx \
  SCRIPT=scripts/tutorials/li383_relaxation.py OUTDIR=outputs/tutorials/li383_relaxation \
  VM_NAME=my-tpu-vm ZONE=<zone> ./run_on_tpu.sh --outer 10 --inner 10
```

`PUSH_FILES` copies extra local files into the repo directory, which is how
`tpu_bench_mrx.py` runs as a `SCRIPT` while still resolving the repo's relative
`data/` paths.
