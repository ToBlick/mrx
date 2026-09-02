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
   minutes on a v5e; with it, setup takes under a minute and a step costs 13 s
   against the same VM's CPU at 46 s. The cause was XLA recompiling the same
   inner solve on every call, not the device being slow. Read
   [section 6.1](#61-what-actually-runs-well-and-what-does-not) before you plan
   a campaign: it also shows why replacing indexed gathers and scatters with
   dense shifted reads and adds turned this hardware from 5-23x slower than the
   VM's own CPU into 3-5x faster on the same operation. A v5e is still about
   32x behind one H100 on a single li383 equilibrium.
3. **The 13 s per step was 37 478 operator applies, not slow hardware.**
   Three MINRES solves were 99.5% of them and one was 75% on its own. That
   turned out to be a preconditioner chosen for the wrong term -- the operator
   is Laplacian-dominated where a comment claimed it was mass-dominated -- and
   fixing it takes the step to **8.05 s, a 1.62x**, measured here against the
   old preconditioner on the same node. It is identical on a CPU and a GPU, so
   this is not a TPU fix. Section 6.1 has the numbers; mrx PR #19 has the
   change.
4. **A `v5litepod-4` is four chips and one equilibrium uses one of them.** A
   parameter sweep can have all four for the cost of a `jax.pmap`: measured
   **3.99x on real chips**, `pmap_sweep.py`.

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

**Measured result:** with float32 and `highest`, the toroidal Poisson tutorial
on v5e reproduced the CPU float32 reference *exactly*:

| p | n | TPU | CPU reference | deviation |
|---|---|---|---|---|
| 2 | 6 | 1.0754e-02 | 1.0754e-02 | 0.00% |
| 2 | 8 | 3.5617e-03 | 3.5617e-03 | 0.00% |

So float32 on TPU is trustworthy for this class of problem. Where float64 still
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

Everything below was measured with `tpu_bench_mrx.py` at `--ns 12,24,12 --p 3`,
float32, on one v5e node, with the host CPU of the *same VM* as the control, so
only the backend differs. Both columns run the same fixed code.

| Measurement | v5e before | v5e after | Same VM's CPU | H100 (docs) |
|---|---|---|---|---|
| `build_sequence` (warm cache) | 203 s | **35.5 s** | 40.6 s | - |
| `compute_nullspaces`, `gap_sweeps=0` | 222 s | **17.5 s** | 55.1 s | - |
| `apply_laplacian` k=1 (nested CG) | 10 020 ms | **76.4 ms** | 108 ms | - |
| `mass_core_apply` k=1 | 6.60 ms | **0.505 ms** | 3.79 ms | - |
| `E` apply k=1 | 1.999 ms | **0.181 ms** | 0.102 ms | - |
| relaxation, per step (mass precond) | 100.3 s | **13.03 s** | 45.7 s | 0.41 s |
| relaxation, per step (laplacian precond) | -- | **8.05 s** | -- | -- |

So the TPU went from 1.7x *slower* than the VM's own CPU to **3.5x faster**, a
7.7x end-to-end gain, and setup dropped from 7 minutes to under one. It is now
about 32x behind a single H100, down from 105x.

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
cost an hour here. `startup.sh` now installs it, and

```bash
python gcs_cache_smoke.py --cache gs://<bucket>/mrx --platform tpu
```

proves the path in about ten seconds before a real run depends on it.

**Fixes 2 and 3: the mass kernel now holds no index tensors at all.**
`_sumfact_kernel` began with a gather, `x[gather_idx]`, and ended with a
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

That table is the most useful thing in this guide. On the indexed forms the
v5e is 23x and 5x *slower* than the VM's own CPU; on the structured forms it
is 2.8x and 5x *faster*. Identical work, identical answers -- exactly for the
gather, to 3.4e-07 in float32 for the scatter. Only the access pattern
changed. If your kernel is slow on a TPU, look here first.

The gather was the bigger half and was found by attribution rather than
guessed: of `mass_core_apply` k=1 at 3.196 ms, the gather was 0.854 ms per
component against 0.100 ms for the forward einsums, 0.076 ms for the reverse
and 0.050 ms for the assembly. About 80% of the kernel was spent reading 3456
numbers.

`mass.py` checks the shift structure per axis and falls back to the index
tensors if any axis fails, so a basis built differently still assembles
correctly. When the plan holds the index tensors are dropped from the kernel
signature rather than left unused, which is also what removed the
`Constant folding an instruction is taking > 1s` warnings: with no index
tensor there is nothing left for XLA to fold.

**Fix 4: the extraction operator was two eager ops.**
`MatrixFreeExtraction._apply` dispatched a gather and a `segment_sum`
separately. `E` and `E^T` measured a flat 1.93 ms whether there were 4584 or
11484 non-zeros, against a 0.037 ms floor for dispatching one device call, so
the cost was per indexed op and not per element. Compiling the pair as one
program took it to 0.16-0.20 ms.

**Three things that sounded right and were not.** All three were measured
rather than argued about, which is the only reason we know:

- *Passing `indices_are_sorted=True` to the scatter.* On v5e the sorted call was
  **slower** (0.615 ms vs 0.533 ms), and sorting properly, which means permuting
  the contributions at every apply, was worse still (0.873 ms). The hint helps
  on GPUs. It does nothing here.
- *Replacing the extraction operator with a dense matmul.* Only 1.3x (0.408 ms
  vs 0.533 ms) and it costs 303 MB resident. Not worth it.
- *Blaming per-call dispatch overhead.* One device call costs 0.037 ms, and
  fusing 20 scatters into a single `jit` gave 0.531 ms per scatter against
  0.533 ms unfused. The time is real device work, not launch overhead.

**What is still 32x off an H100, and why.** `relaxation_loop` is already
`@jax.jit` around a `lax.scan`, so the 13 s per step is genuine compiled device
execution with no compilation left in it. The tempting explanation is that a
li383 step is a chain of kernels too small to fill the chip. That is wrong, and
it is worth knowing why, because it changes what you would fix.

From the numbers above a step ought to cost about 350 ms: ~350 mass-core
applies at 0.5 ms plus ~700 extraction applies at 0.18 ms. It costs 13.03 s.
The per-apply arithmetic is right; the apply *count* is wrong by two orders of
magnitude. **One step is 37 478 operator applies**, which at 0.35 ms each is
exactly the 13 s. Nobody had counted, because the step is a jitted scan and
every solver's `info` is discarded inside it.

Run one step eagerly, where `info` is concrete, and three MINRES solves turn
out to be 99.5% of the work. The velocity smoothing solve alone,
`(M_2 + eps L_2) x = M_2 u` at k=2, takes **3 497 iterations** and is 75% of
the step. The six inverse-mass CG solves everyone assumes are the cost take
20-27 iterations and are 0.5%. The count scales linearly with the DoFs (951
iterations at `(8,16,8)`, 3 497 at `(12,24,12)`) even though `eps ~ 1/n_r^2`
makes the system more mass-dominated as you refine, which points at the
preconditioner rather than at float32 or at stagnation -- every solve converges.

None of that is a TPU property. The same counts hold on a CPU and have always
held on the GPU, where each apply is ~30x cheaper and so nobody noticed. See
`results/benchmark_v5e_vs_cpu.md` for the full per-solve table.

**That solve has since been fixed, and it was a preconditioner.** The mass
preconditioner was chosen on the strength of a comment asserting
`eps * lambda_max(M^-1 L) ~ 0.26` at `(8,16,8)`, i.e. that `M + eps L` is
mass-dominated there. Power iteration says **91.5**, and 360.3 at
`(12,24,12)`: `lambda_max` grows 8.9x between those two resolutions while
`eps` falls only 2.25x, so refining moves *further* from mass dominance. The
operator is Laplacian-dominated by two orders of magnitude and was being
preconditioned for the wrong term.

Using the metric-lumped Laplacian atom instead, as `(1/eps) P_L`, is a
selectable kind (`'laplacian'`) and is what `TimeStepper.smooth_velocity` now
asks for. Measured on this v5e, same node, same session, `(12,24,12)` p=3
float32, 5 steps steady-state:

| velocity-smoothing preconditioner | per step |
|---|---|
| `auto`, the mass atom | 13.02 s |
| `laplacian` | **8.05 s** |

**1.62x on the whole relaxation**, from one preconditioner argument. The
`auto` figure reproduces the 13.03 s measured in the earlier session to three
significant figures, which is what makes the comparison trustworthy. The
trajectory is unchanged: six steps agree to 4.3e-09 relative against a solver
tolerance of 1.49e-08. See mrx PR #19; the underlying condition number is
still resolution-dependent, so this is a large constant rather than a cure.

**Four chips instead of one, measured at 3.99x.** A `v5litepod-4` is four chips
and MRX uses one. For a parameter sweep that needs no library change: the scan
is a pure function of an equinox `State`, so `jax.pmap` over a stacked batch of
initial states runs one equilibrium per chip. `pmap_sweep.py` does exactly
that, and on a real `v5litepod-4` four equilibria 1% apart at `(8,16,8)` p=3
float32 took **5.8 s on four chips against 23.1 s one at a time**. Each
reproduced its own sequential answer to 5.0e-05 in float32, well inside the
3.5e-02 bar, and the members differ from each other by the 1-3% they were built
to differ by, which is the check that this is four problems and not four copies
of one.

Time the compilation separately or you will measure the compiler. Both forms
compile once; the serial loop then amortises it over four calls and the `pmap`
pays it on its only call, so a single-shot comparison of the two reads 2.77x,
and a first attempt that also ran with a cold XLA cache read 1.56x. Neither is
a property of the hardware. The script runs each form twice and reports the
second, which is what a real sweep of any length gets.

```bash
PUSH_FILES=pmap_sweep.py SCRIPT=pmap_sweep.py OUTDIR=outputs/pmap \
  VM_NAME=mrx-tpu ./run_on_tpu.sh --ns 8,16,8 --p 3 --steps 4 \
  --out outputs/pmap/pmap_sweep.json
```

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
`--profile` writes a `jax.profiler` trace and `profile_top_ops.py` reduces that
trace to the only table worth reading, XLA ops ranked by self time. That names
the bottleneck instead of inferring it from wall-clock phase times.

```bash
PUSH_FILES=tpu_bench_mrx.py SCRIPT=tpu_bench_mrx.py OUTDIR=outputs/bench \
  VM_NAME=my-tpu-vm ZONE=<zone> RUN_PLATFORM=tpu \
  ./run_on_tpu.sh --skip-relax --profile outputs/bench/trace

python profile_top_ops.py script_outputs/bench/trace --top 20
```

The trace converter is in `tensorboard-plugin-profile`, not in JAX, and its
module path moves between releases; the script tries several and exits 2 with a
`pip install` line rather than failing obscurely. A missing plugin costs you
this table, not the benchmark.

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

**Cloud TPU API nodes have no `--max-run-duration`. They bill until deleted.**
A `v5litepod-4` runs roughly USD 5-6/hour. Nothing stops it: not a job finishing,
not your SSH session closing, not your laptop sleeping.

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

## 9. Reference numbers

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

That run predates fixes 2-4. Re-timed afterwards on the same geometry the step
is **13.03 s**, so the same 100 steps is about 22 minutes. For scale, this VM's
host CPU running the identical fixed code is 45.7 s/step, and one H100 is
documented at 0.36-0.41 s/step.

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
documents a clean nested floor at roughly 1000 steps with velocity smoothing,
which at 13.03 s/step is about 3.6 hours. What is shown here is a well-behaved
5.5x reduction in force with helicity conserved to 5.6e-07, not a converged
equilibrium.

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
- `watch_request.sh` - non-blocking status of a queued request
- `mrx_tpu_report.py` - Poisson driver with the CPU-reference check
- `tpu_bench_mrx.py` - phase and primitive benchmark; separates compile time
  from execute time, and has a `--compare` mode for the TPU/CPU table
- `profile_top_ops.py` - reduces a `jax.profiler` trace to a top-N op table
- `pmap_sweep.py` - runs one equilibrium per chip and checks that they are
  four problems rather than four copies of one
- `gcs_cache_smoke.py` - proves a `gs://` compilation cache path in ~10 s,
  because JAX does not fail on one it cannot reach
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
