# TPU support: what the branch does, and what the numbers turned out to be (2026-09-05)

`tpu-support-v2` against `static-dynamic-refactor`, PR #20. This note is the
handoff: what changed, which decisions were taken in the two merges, what the
measurements say after two confounds were removed, and what is closed rather
than open.

The one-line answer to the question the branch set out to ask: **a Cloud TPU
v5e does not suit this workload, and the evidence for that got stronger, not
weaker, every time a measurement error was corrected.**

## 1. What the branch changes

Three groups, none of which depends on the others.

**The mass kernel loses its index tensors.** On a tensor-product B-spline
basis the element-to-DoF map of every axis is the pure shift `(e + l) mod S`,
so the element-local read and the assembly are stacks of rolled slices rather
than gathers and scatters against materialised index arrays.
`_structured_gather` and `_structured_accumulate` in
[mrx/mass.py](../../mrx/mass.py) do that, `_shift_plan` is the precondition
that decides whether an axis is usable, and it raises rather than falling back
because the library builds only tensor-product bases and one numbered
otherwise would read and write silently wrong. `_to_quadrature` stayed a pure
contraction: the read is a separate step and the kernel does it.

**`tpu/` is the `slurm/` analogue.** It differs structurally, because on a
cluster the machine exists and you queue for it, while a TPU does not exist
until you create it and bills until something deletes it. So the scripts
provision hardware from a laptop as well as run on it, and every node carries
`idle_reaper.sh`. Operationally the interesting parts are the candidate ladder
with failure classification ([tpu/zones.sh](../../tpu/zones.sh)), the
acquire daemon and its queued-request handling
([tpu/acquire_tpu.sh](../../tpu/acquire_tpu.sh)), and the persistent
compilation cache, which is the single largest effect measured on this
hardware.

**Two benchmarks.** `scripts/benchmark/relaxation_bench.py` (phases and per
step) and `scripts/benchmark/matvec_bench.py` (per kernel). Neither is
TPU-specific; both ran on all three backends.

## 2. The two merges, and what was taken from whom

**`b2c0df9`**, five commits, of which `2705719` gave the solves one stopping
criterion: the true residual in the mass-atom norm. Two conflicts:

- [mrx/mass.py](../../mrx/mass.py): took upstream's plan-on-the-sequence API
  (`SumfactPlan`, `mass_plan`, `projection_plan`, `attach_weights`,
  `sumfact_apply`) **and** this branch's structured shifts. `SumfactPlan`
  carries `shift_plans` and `gather_plans` where it used to carry index
  tensors; the kernel is unchanged otherwise.
- [test/test_poisson.py](../../test/test_poisson.py): took upstream's
  mass-atom residual and re-measured iteration counts.

That merge also dropped this branch's dtype pin in `preconditioned_cg`, which
`2705719` superseded, and the `xfail` on
`test_reconnection_spends_the_helicity_asked_for`, which an upstream rewrite
fixed.

**`ae7293b`**, three commits, and the interesting one is directly about this
PR. Taking them in order:

- `ad28ca4`: `compute_nullspaces(seq)` installs the bundle itself. Nine call
  sites had been writing `seq.set_operators(compute_nullspaces(seq, ops))`, a
  second install of what was already installed. Both benchmarks were among
  them and are fixed.
- `8297c83`: the velocity smoothing scale is one rule,
  `SMOOTHING_C / n_r^2 = 0.02 / n_r^2`, and `TimeStepper` takes it when asked
  for `None`. The benchmarks had the old hand-computed `0.064 / n_r^2` at both
  call sites and would otherwise have measured 3.2x the smoothing production
  runs at. They now pass `None` and record what they got.
- `ae7293b`: `test/conftest.py` no longer forces `MRX_DTYPE=float64`, so a
  bare `pytest` tests the production configuration, and `slurm/suite.sh`
  submits three. Written in response to this PR's float32 report; see §4.

Four conflicts, all of them this branch's float32 floors against upstream's,
and upstream's were taken in every case. The reason is in §4.

## 3. The measurements, and the two confounds removed to get them

`(12,24,24)` p=3 li383 throughout. One `v5litepod-1` (JAX 0.11.1) against one
NVIDIA H200 on NYU Torch under Singularity.

Two published figures were withdrawn on the way here, for two different
reasons, and both are worth stating because the second is the subtler mistake.

**Withdrawn once: the per-step numbers were compile time.** Found by Tobi in
review. `relaxation_loop` built its jitted scan inside its own body, so every
call retraced and recompiled; the benchmark timed two calls and divided the
second by the step count on the assumption that it was compile-free. It was a
second compile. Separately, dividing a whole call by its step count charges
the steps for work that does not scale with them. `chunk_runner` builds the
scan once and the benchmark reports compile, per step and per-call overhead
separately. The H200 went from a published 17.03 s/step to 0.352 s, a factor
of 48, and Tobi's independent H100 figure of ~0.5 s/step is the corroboration.

**Withdrawn twice: the comparison was not at matched tolerance.** The
1.800 s against 0.352 s, "5.1x", published on 2026-09-05, compared an H200 at
the float32 default of the day (`sqrt(eps) = 3.5e-4`) against a v5e at
`SOLVE_TOL = 1e-6`. A tolerance is a work knob, so that charged the v5e for
iterations the H200 was never asked to do.

Re-measured with both backends plain float32 at `1e-6` on the same tree
(`6af991f`):

| | v5e | one H200 |
|---|---|---|
| inverse-mass CG iterations, k=1 / k=2 | 95 / 98 | 95 / 98 |
| one compile-free 5-step call | 14.308 s | 1.489 s |
| **per step** | **2.862 s** | **0.298 s** |
| compile, first call only | 81.44 s | 60.95 s |

**9.6x for the H200.** Correcting the confound moved the gap from 5.1x to
9.6x: it made the TPU look worse. The identical iteration counts are what say
the tolerance is no longer the variable.

Each backend at its own production default. These two are **not** a
like-for-like pair -- they differ in tolerance, in refinement, and the TPU one
was taken after `8297c83` made the smoothing scale `0.02 / n_r^2` where the
matched pair used `0.064 / n_r^2` -- so they say what each machine costs as
configured and nothing about a ratio:

| configuration | per step (5-step call) |
|---|---|
| H200, refined float32, `SOLVE_TOL = 1e-8` -- the GPU default | 0.410 s |
| v5e, plain float32, `3.5e-4` -- the TPU default since 2026-09-05 | 1.130 s |

Moving the v5e from `1e-6` to the reachable `3.5e-4` takes it from 2.862 s to
1.130 s a step, **2.5x**, and its inverse-mass CG from 95 and 98 iterations to
22 and 26. An unreachable tolerance was not a small tax on this machine.

**A method note that matters, because it turned into a result.** Per step
above is one compile-free call divided by its steps, not the slope between a
5-step and a 10-step call that the earlier tables used. The slope is the
better estimator when it works, since differencing cancels the fixed per-call
cost -- but it implied a per-call overhead of **-17 s** on the v5e, which is
arithmetically impossible.

What that says is that the steps are not equal: the second half of the
trajectory costs more than the first, and the slope charges the difference to
the steps and the remainder to a fixed cost that must then come out below
zero. Every configuration shows some of this, and one shows a lot:

| | steps 1-5, per step | steps 6-10, per step | ratio |
|---|---|---|---|
| v5e, `1e-6` | <= 2.862 s | 6.270 s | **2.19x** |
| v5e, `3.5e-4` | <= 1.130 s | 1.263 s | 1.12x |
| H200, `1e-6` | <= 0.298 s | 0.318 s | 1.07x |
| H200, `1e-8` refined | <= 0.410 s | 0.425 s | 1.04x |

The stepper is adaptive and its inner solves are iterative, so some rise is
expected. The v5e at `1e-6` is the outlier, and the cause is the tolerance
rather than the machine: float32 cannot reach `1e-6`, so solves burn their
maximum passes, and more of them do the further into the trajectory the run
gets -- the same effect `ae7293b` measures on the Poisson cases, seen from the
other end. At `3.5e-4` the same machine falls to 1.12x, the GPU's shape.

Dividing by the step count charges the steps for the call's fixed work, so
every figure is an upper bound on the steps it averages rather than an error
of unknown sign, and comparing the same chunk length across machines compares
the same work. The matched ratio is 9.6x on the same 5-step call and **14.8x
on the same 10-step call**, widening for the reason above. Every call was
repeated three times and agreed to the millisecond.
`relaxation_bench.py` now withdraws its own slope row when the implied
overhead is negative, reports the ratio that caused it, and prints every
repeat.

**Where the 9.6x lives.** Not in iteration counts, which are equal, and not in
any one kernel: a single inverse-mass CG iteration is 1.7x apart (4.73 ms
against 2.86 ms) and the mass core apply at k=1 is 7x (0.800 ms against
0.114 ms), while the whole step is 9.6x. The gap widens with each layer of
nesting, so it is in the many nested solves a step runs.

## 4. float32, and why upstream's floors were taken

Running the suite in the two float32 configurations found six Poisson residual
assertions failing, and in plain float32 (the TPU configuration) the
relaxation NaN'd at step 10. Both causes are now understood and neither is a
wrong solution.

The NaN was in `refine()`. At `SOLVE_TOL = 1e-6` extra correction passes were
*raising* a float32 residual, and the loop kept the raised one.
[mrx/solvers.py](../../mrx/solvers.py) now discards a pass that does not
strictly decrease a finite residual and keeps the last improving iterate. That
fix is this branch's and survives the merge; `test/test_refine.py` covers it.

The residual assertions were the working-precision floor, and upstream reached
the same conclusion independently and more precisely. `ae7293b` measures that
the true residual float32 arithmetic attains is 2.4e-4 on the k=2 Hodge split
and 4.2e-4 on the k=3 saddle of the test mesh, that five of the eight Poisson
solves burned their passes at `1e-6`, and restores the plain-float32 default
to `sqrt(eps) = 3.5e-4`. It also solves the test's own reference in the
residual precision, so `info < 0` can be required unconditionally instead of
only when the tolerance dominates the floor -- which is strictly better than
this branch's conditional bands, and is why those were dropped.

Verified after the merge, in the `mrx` conda environment: **56 passed in each
of refined float32 (bare `pytest`), plain float64, and plain float32**, being
upstream's 54 plus this branch's `_shift_plan` and `refine` tests.

## 5. Operational: first fulfilment wins

A `--queue` run on 2026-09-05 left **four nodes billing at once**, in
`us-west4-a`, `us-west1-c`, `us-east1-d` and `us-east1-b`. `claim_first_node`
already cancelled the losers; it was reached too late and from one path only.
The main loop called it after a node reported `READY`, and between iterations
slept `SWEEP_INTERVAL` (180 s) with a ladder walk on top. A sweep win never
called it at all.

The sleep is now `sleep_watching_the_queue`, which polls the standing requests
every `QUEUE_POLL_S` (15 s; one `describe` per queued zone in parallel, 2 s
measured over 15 zones) and claims the first one granted. Granted means
anything past `WAITING_FOR_RESOURCES`, deliberately including `ACCEPTED` and
`PROVISIONING`, where no node exists yet -- cancelling then is what stops one
appearing. Claiming that early broke the rule `cancel_queued_requests` relied
on (it deletes without `--force`, and the API refuses to delete a request
holding a node, which protects a winner only once it has one), so the winner's
zone is now skipped by name, and the claimed node is waited for in its own
zone rather than by falling through to another ladder sweep.

**Measured on its first run.** At 14:02 on 2026-09-05 the sweep won
`us-west4-a` and the new call on that path tore down four nodes that had come
up in `us-east1-c`, `us-west1-c`, `us-east1-d` and `us-central1-b`. That is
the same four-node incident, prevented.

## 6. Closed, and why

- **A single chip is the right slice.** Four chips are 0.1 ms apart from one
  (1.7999 s against 1.7998 s) because the solve is single-device, and a
  `v5litepod-4` is atomic, so the alternative was paying for four chips to use
  one. All 14 `v5litepod-4` rungs are out of `DEFAULT_CANDIDATES` (53
  candidates to 39). `scripts/pmap_sweep.py` is kept: it is for independent
  equilibria across devices, which is the one shape that would use them.
- **`jax_default_matmul_precision=highest` stays**, though it is a TPU-only
  tax worth up to 1.55x on the mass kernel. At `high` the geometry map carries
  1.9e-04 relative error on `DF`; at `default` it **folds**, `det DF` reaching
  -1.3e-01, and `set_geometry` refuses it.
- **The dispatch argument keeps its mechanism and loses its conclusion.**
  Eager microbenchmarks do overstate the scan form most on the backend paying
  most per call (1.0-1.6x on CPU, 1.3-6.8x on the v5e, 5.9-66x on the H200).
  It is not large enough to decide the step.
- **Ideas that were tried and did not work** are in the benchmark note's own
  table: sorted scatter indices, a dense extraction matmul, a structured
  extraction operator, and the theory that narrow contractions underuse the
  128-wide MXU (refuted: cost is flat from K=4 to K=8).

## 7. Open

- The 9.6x is attributed to nested-solve depth by three measurements at
  different layers, not by an op-level trace. A `jax.profiler` trace would
  name the ops; it was started on the v5e and the node was idle-reaped before
  it was parsed. Worth doing only if someone intends to act on it.
- `compute_nullspaces` is the one phase where the v5e is worst of the three
  backends, at 3.2x the same VM's CPU. Nothing here was aimed at it, and it
  includes its own compile, so part of the gap is not arithmetic.

## Sources

- [tpu_v5e_benchmark.md](tpu_v5e_benchmark.md) -- the numbers, per kernel and
  per phase, with the withdrawn tables and why they were withdrawn.
- [docs/source/tpu.md](../source/tpu.md) -- how to run any of it.
- [docs/source/concepts/mass.md](../source/concepts/mass.md) -- the kernel.
