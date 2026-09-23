# Running on an Apple GPU

MRX runs on Apple Silicon GPUs through
[jax-mps](https://github.com/tillahoffmann/jax-mps), a PJRT plugin that
compiles the StableHLO a JAX program lowers to and executes it with
[MLX](https://github.com/ml-explore/mlx). Nothing is provisioned and nothing
bills: unlike `tpu/`, the hardware is the laptop.

The whole suite passes there, 61/61, in the plain float32 configuration, and
a real li383 relaxation runs to completion. This backend charges a flat
~0.22 ms per dispatch, so an operation is faster or slower than the CPU
according to whether it is bigger or smaller than that floor. The mass
kernel used to be on the wrong side of it, because it had been written as a
dozen dense shifts in place of one indexed read: that is the form a TPU
wants and the form Metal does not. On Metal the kernel is one gather and
one ``segment_sum`` instead (``MRX_ASSEMBLY``, below). With that, 100 L-BFGS
steps at `(12,24,12)` p=3 take **283.8 s on the GPU against 352.5 s on the
CPU**. The numbers are under "What was measured" below, and
`docs/research/mps_benchmark_2026-09-21.md` is the full record.

## The environment

`jax-mps` pins the jaxlib minor version it was built against, because it
deserializes StableHLO bytecode: a mismatch fails at the first JIT compile.
Version 0.10.11 is built for jaxlib 0.10.x, which is what MRX runs on, and
ships a `py3-none-macosx_14_0_arm64` wheel, so no LLVM build is needed
despite what the upstream README describes.

```bash
conda create -n mrx_mps --clone mrx
conda activate mrx_mps
pip install "jax-mps==0.10.11"
python -c "import jax; print(jax.devices())"     # [MpsDevice(id=0)]
```

Keep it in its own environment. The plugin registers itself at priority 500
against the CPU backend's 0, so merely installing it makes `mps` the default
backend for every JAX program in that environment, MRX or not. For the same
reason every CPU comparison below passes `JAX_PLATFORMS=cpu` explicitly.

Requirements are macOS 14 or later on Apple Silicon and Python 3.12 or later.
Measurements here are an M3 Pro on macOS 27.

## The configuration: `MRX_X64=0`

```bash
source mps/env.sh          # MRX_X64=0, JAX_PLATFORMS=mps, and the knob record
python -m pytest test
```

Metal has no float64, and this is the way it differs from a TPU. A TPU also
has no float64, but XLA:TPU silently downcasts it, so the TPU configuration
only has to drop the residual precision (`MRX_RESIDUAL_DTYPE=float32`) and
can leave 64-bit mode on. MLX instead *refuses* the buffer:

```
MLX does not support float64 (F64). Use
jax.config.update('jax_enable_x64', False) or ensure your arrays are float32.
```

Under 64-bit mode a dtype-less constructor like `jnp.zeros(n)` or
`jnp.linspace(0, 1, n)` makes float64, and `mrx/` has dozens of them; on a
TPU they cost nothing and on Metal any one of them aborts the run. So
`MRX_X64=0` turns 64-bit mode off at the root rather than chasing the
constructors, which makes every one of them float32 and leaves nothing that
the backend can refuse.

It implies the rest of the float32-only configuration and refuses to be
combined with anything else: `MRX_DTYPE=float64` and
`MRX_RESIDUAL_DTYPE=float64` both raise at import, rather than failing at the
first buffer transfer several minutes into a run. What is left is the plain
float32 configuration that the TPU work already validated -- `REFINE` off,
`SOLVE_TOL = sqrt(eps) = 3.5e-4` -- so the tolerances are the measured ones
and not new. See {doc}`concepts/precision`, and read the accuracy caveats
there before trusting a number: plain float32 is the configuration for runs
that stop near a residual of 5e-4.

`MRX_X64` is read from the environment rather than detected from the backend
because `jax.devices()` would initialise the backend before
`jax_enable_x64` could be set.

## Checking the backend before blaming MRX

```bash
python mps/probe_ops.py
```

One small program per JAX construct MRX depends on, run on both backends with
the same inputs and compared: the sum-factorised `einsum` of `mrx.mass`, the
gather + `segment_sum` of `mrx.extraction_operators` (including out-of-bounds
segment indices, which must be dropped), the data-dependent `while_loop` of
the Krylov solvers, the stacking `lax.scan` of the relaxation, the spline
`dynamic_slice`, and the `cholesky`/`solve_triangular`/`eigh` chain of the
preconditioner atoms. All 14 match the CPU to float32 rounding on
jax-mps 0.10.11.

This exists because the alternative diagnosis path is bad: a missing
StableHLO handler surfaces as a compile error thrown from inside a
hundred-line fused kernel, and a wrong one surfaces as a plausible number.
Run it first after any upgrade of `jax-mps`, `jax` or macOS.

## What was measured

Suite, `MRX_X64=0`, one M3 Pro, wall clock for `pytest test`:

| backend | configuration | result |
|---|---|---|
| mps | plain float32, indexed assembly | 61 passed, 150 s |
| cpu | mixed float32/float64, shift assembly (the default) | 61 passed, 300 s |
| cpu | plain float32, before the assembly tests existed | 59 passed, 209 s |

The suite is compile-bound, so that near-parity is mostly a statement about
XLA-versus-MLX compile time, not about execution. The parts of it that are
execution-bound disagree with each other: `test_poisson.py` alone is 24.8 s on
the GPU against 45.3 s on the CPU, while `test_relaxation.py`'s longest case
is 54.0 s against 37.4 s. The section below is why.

### A real run, li383 `(12,24,12)` p=3 float32

These are `--method lbfgs`, the plain float32 default since 2026-09-23
(Newton stays the default in mixed precision and float64; see "Newton" below). `scripts/relax.py --method lbfgs
--steps 100 --chunk 25`, the same mesh as the v5e / H200 / H100 table in
`docs/research/tpu_v5e_benchmark.md`:

| | CPU | MPS, indexed assembly | |
|---|---|---|---|
| 100 L-BFGS steps, end to end | 352.5 s | **283.8 s** | MPS 1.24x |
| per step | 3.52 s | **2.84 s** | MPS 1.24x |
| the same run before the assembly change | 353.7 s | 593.9 s | CPU 1.68x |
| one nested-CG Laplacian solve, k=1 | 284.3 ms | **74.9 ms** | MPS 3.8x |
| inverse mass CG k=1, 20 iterations | 587 ms | **158 ms** | MPS 3.7x |

The mass apply inside a scan, the change itself, at this mesh: 1.226 ms the
shift form against **0.503 ms** the indexed form at k=1 (2.44x), and 1.200
against **0.475 ms** at k=2 (2.52x). On the CPU the same comparison is 1.07x
and 0.94x, inside the noise, which is why the CPU keeps the shifts.

Run-to-run spread on this machine is about 17%, so nothing is claimed on a
smaller margin than that.

The potential-velocity Hodge solve, which an L-BFGS step uses for the force,
was then capped at two outer passes in plain float32. It had been taking six,
and the later passes are the ones `refine` discards when they stop improving
the residual. Laplacian-preconditioner applies in one step went from 1482 to
393. The same 20 steps with the cap lifted remove the same energy to 0.2%
and finish with a force within 1%. That uncapped run, which had the
machine to itself, took 61.2 s. The capped figure quoted against it, 41.9 s,
was taken while other jobs were running. Re-timed with nothing else running,
best of three: **26.5 s on the GPU (1.32 s/step; the three runs were 27.2,
27.6 and 26.5 s) against 40.3 s on the CPU (2.02 s/step; 40.3, 40.7 and
40.3 s)**. The cap is about 2.3x against the idle uncapped run, not the
1.46x the overlapping measurement gave, and the GPU is 1.5x the CPU on this
window. Both remove 4.9e-5 of energy. The earlier 100-step averages
(2.84 s and 3.52 s) are the run before this cap, so the two windows are not
the same comparison; the iteration count is the change.

In plain float32 the run stops on the energy, not on the residual.
`--floor-tol` defaults to 0 there and to `1e-8` in mixed precision and
float64. A squared residual of `1e-8` is below anything the float32 force
reaches: at `(12,24,12)` p=3 it sits between about `1e-3` and `1e-2` and
then blows up, while the energy is already walking uphill. That is the
finding of `docs/research/floor_study_2026-09-05.md`. The per-step `dE`
in the trace is the increment `<B_{n+1} - B_n, M(B_{n+1} + B_n)> / 2`,
so it resolves a change far below one ulp of `E`. `relax` stops when the
last two chunks have each either raised the energy on at least 3 of every
10 steps or lowered it by at most `5e-4 * removed`. It returns the
lowest-energy field seen at a chunk boundary, and prints that going
further is a restart in `--precision mixed`. (The first version stopped
on a band of `max(0.01 * removed, 4 * eps * |E0|)`; that stops the
production mesh 8% short, see "The remaining speedups" below.) Mixed precision and float64 are unchanged, because
a float64 residual keeps converging. The switch is
`--energy-floor auto|on|off`.

The float32 chunk is 10, and a full sample (the force, the weak pressure,
the helicity, the checkpoint) runs every five chunks. The stop test does
not wait for the sample. Sampling every chunk put 33 s of diagnostics
beside 149 s of steps at a step-90 exit; sampling every 50 steps left 9 s
beside 108 s. That saving is inside the 17% run-to-run spread, so it is
not a speedup. It is what makes a ten-step chunk cheap to test. The
per-step residual scale, an `M_0` solve of `B^2/2` inside the scan, was
20 L-BFGS steps at 33.2 s with it and 33.1 s without. It stays in the step.

Time to that floor, L-BFGS, from the equilibrium field, best of three,
with the two-pass force solve and the first floor rule (the current
figures are in "The remaining speedups" below). The old default is 3000
steps. It never meets `--floor-tol 1e-8`. At the
idle 1.32 s/step measured above, that budget is 66 min; today's floor
runs on the same mesh averaged 1.4-1.5 s/step, so the same budget is
about 70-75 min.

| | exit | wall, best of 3 | the three runs | energy removed |
|---|---|---|---|---|
| MPS `(12,24,12)` | steps 70, 90, 100 | **108 s** | 108, 139, 144 s | 5.77e-5, 5.98e-5, 6.19e-5 |
| CPU `(12,24,12)` | step 90 | **122 s** | 123, 122, 122 s | 6.05e-5 |
| MPS `(16,32,16)` | step 130 | **255 s** | 286, 255, 277 s | 6.32e-5, 6.29e-5, 6.22e-5 |
| CPU `(16,32,16)` | step 130 | **432 s** | 432, 432, 451 s | 6.31e-5 |

A 300-step calibration at `(12,24,12)`, with the stop turned off, reached
its best energy of 6.10e-5 at step 94 and was uphill after that: by step
300 the energy had risen by 1.2e-3. The exit on that trace returns the
step-80 field, 6.02e-5, which is 1.4% less. The live exits above are the
same quantity. The 108 s run is the one that removed 5.77e-5; that
trajectory peaked there, and the exit returned its own peak. Float32
runs separate by about that much. The spread of the time to the floor is
the step the trace goes flat, which is wider than the 17% on a fixed
step count. The CPU runs repeated to the last digit of the energy. On
the larger mesh the best of three is 255 s against about 100 min for
3000 steps at the 2.0 s/step those runs averaged, about 24x. On the
smaller mesh 108 s against 66 min is about 37x, and the slow exit of
that three, 144 s with 6.19e-5 removed, is 28x.

Newton does not get that factor, which is why it is no longer the plain
float32 default. On Metal
at `(12,24,12)` one run exited at step 50 in 1795 s, 35.9 s/step, having
removed 5.48e-5. That is short of the L-BFGS floor near 6.1e-5: the last
two chunks had fallen under the band. Another run was still descending at
step 30, 5.70e-5 removed, last window 1.7e-6, after 996 s. The old budget
is 100 steps, about an hour at 36 s/step, so the exit is about 2x when it
fires near step 50, not the 20x a shared floor would have been. L-BFGS
has removed more energy in two minutes than Newton has after half an
hour. On the CPU the same mesh did 20 Newton steps in 435 s, 21.8 s/step,
and had removed 4.57e-5 without stalling. A step-50 exit at that rate
would be about 18 min, against 122 s for L-BFGS there. These are single
runs, not a best of three: the spread is which step the windows go flat,
and a third half-hour run would not change the comparison. The
`(16,32,16)` Newton floor was not run. One Newton step is already
several L-BFGS steps, and the small mesh is where the two methods meet
the same energy.

### The remaining speedups (2026-09-23)

Where a floor run's time goes, from the logs of the runs above (the steps
are the compiled chunks; "other" is samples, checkpoints and the stop):

| run | elapsed | setup | IC | steps | other |
|---|---|---|---|---|---|
| MPS `(12,24,12)` | 150 s | 20.3 s | 1.2 s | 107.7 s | 9 s |
| MPS `(16,32,16)` | 309 s | 24.6 s | 1.2 s | 254.9 s | 18 s |
| CPU `(12,24,12)` | 202 s | 32 s | 3.5 s | 122 s | 30 s |
| CPU `(16,32,16)` | 545 s | 42 s | 2.7 s | 432 s | 53 s |

Setup plus compile is 15% of the Metal run and 17% of the CPU one at
`(12,24,12)`. The setup is the preconditioners (14.2 s of 28.9 s on the
CPU: mass lumping 7.5 s, Laplacian lumping 6.6 s), `DeRhamSequence` 8.8 s,
the map 2.9 s, and the harmonic forms 13 s. A cache of the preconditioners
and harmonic forms would save about 10% end to end, under the 17% spread,
so it was not built.

**The force's Hodge solve takes one pass.** The step's k=1 Hodge solve
(`L_1 a = curl^T load(J x B)`) is most of the step:
`mps/hodge_share.py` records the right-hand sides one step hands it and
times that solve compiled on its own against the compiled step, best of
three:

| | step, two passes | Hodge share | step, one pass | Hodge share | |
|---|---|---|---|---|---|
| MPS `(12,24,12)` p=3 | 1884 ms | 41% | **1185 ms** | 34% | 1.59x |
| MPS `(16,32,16)` p=3 | 3247 ms | 59% | **2007 ms** | 52% | 1.62x |
| MPS `(16,32,32)` p=2 | 3047 ms | 59% | **1709 ms** | 52% | 1.78x |
| CPU `(12,24,12)` p=3 | 1456 ms | 85% | **814 ms** | 79% | 1.79x |
| CPU `(16,32,16)` p=3 | 3126 ms | 86% | **1908 ms** | 76% | 1.64x |

Plain float32 capped the split's outer loop at two passes, and the second
cost as much as the first (hat CG 141 then 138 iterations at
`(12,24,12)`, `mps/solve_attr.py`). The force's solve is warm-started from
the last step's potential, so one pass is enough there. The energy traces
of one and two passes agree step for step (6.03e-5 removed at step 70 in
both) until the two-pass run breaks down at step 90. A cold solve still
needs both: with one pass the k=1 and k=2 manufactured solutions in
`test_poisson.py` do not converge. So the cap is the step's,
`TimeStepper.force_hodge_passes` (`--force-hodge-passes`): 1 in plain
float32, and the solve's own cap in mixed precision and float64, where the
passes are how a float32 Krylov solve reaches a float64 tolerance.

The hat solve is still 52% of the Metal step at the larger meshes (and
160-180 iterations a pass), over the 40% at which a stronger k=1
preconditioner was to be tried. The one the atom has is radial bands
(additive Schwarz over three overlapping bands, only the k=1 Dirichlet
atom). It cuts the hat iterations from 279 to 217 at `(12,24,12)`, but
each apply is three band solves:

| k=1 Dirichlet bands | MPS `(12,24,12)`, 2 passes / 1 pass | MPS `(16,32,32)`, 1 pass | CPU `(12,24,12)`, 2 passes / 1 pass |
|---|---|---|---|
| 1 | 1884 / 1185 ms | 1709 ms | 1456 / 814 ms |
| 3 | 2349 / 1437 ms | 1627 ms | 1154 / 688 ms |

Slower on Metal at the benchmark mesh, 5% at the production one, 15% on
the CPU after the one pass: not taken. The same finding as the float64
band study of 2026-09-06 (`docs/research/newton_second_variation_2026-09-06.md`).

**The floor rule, re-set on the production tail.** A 2000-step
calibration at `(16,32,32)` p=2, the stop off, removed 7.27e-5 at best;
by step 540 it had 99.5% of that. It descends monotonically for 800 steps,
at 1e-7 to 3e-7 per ten steps between steps 150 and 400. The first rule
fired on that trace at step 120 with 92.4% (the band `0.01 * removed` was
6.7e-7), and its ulp term, 2.4e-7, would have fired at step 200. The
per-step changes separate the two regimes cleanly: in a descent every step
lowers the energy; at the floor 3 to 7 of every 10 steps raise it
(`(12,24,12)` from step 80, `(16,32,16)` from step 130, `(16,32,32)` from
about step 800). The rule is now two chunks that each raise the energy on
3 of their 10 steps or lower it by at most `5e-4 * removed`. Replayed on
every recorded trace, it fires where the old rule did at `(12,24,12)` and
`(16,32,16)` (steps 90 and 140 on the long calibrations, 99.5% and 100% of
their best) and at step 570 on the production trace, 99.7%. `1e-3` would
fire at 460 with 98.8%, too near the 2% budget.

**Time to the floor with both changes**, best of three:

| | exit | wall, best of 3 | the three runs | energy removed | before |
|---|---|---|---|---|---|
| MPS `(12,24,12)` | steps 90-110 | **62.9 s** | 82.6, 62.9, 67.2 s | 6.39e-5, 6.14e-5, 6.26e-5 | 108 s, 5.77-6.19e-5 |
| CPU `(12,24,12)` | step 120 | **89.6 s** | 89.6, 90.5, 91.5 s | 6.45e-5 | 122 s, 6.05e-5 |
| MPS `(16,32,32)` p=2 | step 620 | **599 s** (one run) | 599 s | 7.24e-5 | see below |

At `(12,24,12)` that is 1.7x on Metal and 1.4x on the CPU, and more energy
on both: the two-pass run broke down at step 90, the one-pass one keeps
descending to step 110-120. The production run is 0.97 s/step against the
calibration's 1.83, and its 7.24e-5 is 99.6% of the calibration's 2000-step
best (3653 s); the calibration first had that energy near step 540, about
990 s. It is one run: at 10 minutes it is the one production figure here,
and the per-step times of the Metal runs sit inside the 17%.

**L-BFGS is the plain float32 default method.** Newton, the previous
default, against the L-BFGS exits above (the L-BFGS figures are the
two-pass ones, so the gap is now larger):

| | Newton | L-BFGS at the same wall |
|---|---|---|
| MPS `(12,24,12)` | 50 steps, 1795 s, 5.48e-5 | 6.19e-5 by 144 s |
| CPU `(12,24,12)` | 20 steps, 435 s, 4.57e-5 | 6.05e-5 by 122 s |
| MPS `(16,32,16)` | 5 steps, 204 s, 2.93e-5 | 6.22-6.32e-5 by 255-286 s |
| CPU `(16,32,16)` | 10 steps, 478 s, 3.28e-5 (3.27e-5 at 430 s) | 6.31e-5 by 432 s |

`--method` defaults to `lbfgs` in plain float32 and to `newton` in mixed
precision and float64, where the relaxation goes past the float32 floor.

**The optimiser's parameters are not a speedup.** One run each at
`(12,24,12)` on Metal, the stop on, with the one-pass step: history 3
57.0 s (5.95e-5), 5 77.5 s (6.38e-5), 10 73.4 s (6.53e-5); cfl 1.0 69.3 s
(6.41e-5); smoothing scale 0.5x 56.0 s (6.12e-5), 2x 63.4 s (6.43e-5),
against the default's 62.9-82.6 s. The best two, best of three: smoothing
0.5x 56.0 s (56.0, 61.7, 63.4 s; 6.00-6.12e-5) and history 3 56.3 s (56.3,
57.0, 89.8 s; 5.95-6.18e-5), 11% and 10% under the default's best, with
less energy removed. Inside the spread; the defaults stay.

**Coarse to fine is not faster to the floor.** `--warm-from R,T,Z`
relaxes a coarse mesh first and moves the relaxation increment onto the
fine mesh's own initial field by histopolation, which commutes with `d`
(`mrx.initial_conditions.transfer_field`; the harmonic part of `B`, 97%
of it, is not carried by a potential, so the field itself is moved). From
`(9,16,16)` into `(16,32,32)` p=2 (nested: 7 into 14 radial elements):
the coarse floor took 230 steps, 97 s, 131 s with its setup and the
transfer, and the moved field starts the fine run with 6.54e-5 removed,
`||div B||` 2.2e-5. The fine run then stops at step 800 after 825 s with
7.87e-6 more, 7.32e-5 in all: 1.1% more than from the initial field, in
956 s against 599 s. At equal energy the warm start is ahead by less than
the spread (95% of the from-IC floor at 153 s against 187 s, 98% at 347 s
against 366 s, the from-IC floor itself at about 528 s against 599 s). It
stays an option, not the default.

### Newton, the mixed and float64 default

A Newton step is a 300-iteration MINRES solve whose matvec is two k=1 mass
solves. It used to be three: `dJ` and `W` are both `M_1^-1` of a dual
1-form and both enter only through `load(B x .)`, so one solve for
`X = dJ + W / 2` replaces them, and the two remaining loads are integrated
once. `newton_tol=0.1` is not reached: every step spends the whole budget.
That tolerance is the truncation, not a target, and `newton_it` reads
`+300`. One step from Tutorial 4's field is 27.1 s after the merge against
40.3 s before it. A smaller MINRES budget does not help: over five steps
the force never moves and the energy change is a few ulps either way, so
the default stays 300.

On the benchmark mesh, two steps from the equilibrium field, the indexed
matvec and the shift matvec agree. The force at step 1 agrees to 8e-7 and
the descent cosine to 3e-7, and both fall back to the smoothed force on that
one step and not the next:

| | CPU, shift | MPS, shift | MPS, indexed |
|---|---|---|---|
| 2 Newton steps | 101.3 s | 120.9 s | **86.7 s** |
| per step | 50.6 s | 60.5 s | **43.4 s** |
| fallbacks | 1/2 | 1/2 | 1/2 |

That indexed column is not what a Newton run does. On Tutorial 4's mesh,
`(10,16,16)` p=2, ten steps from the shipped descent state, the same indexed
matvec changed the direction: the descent cosine collapsed to 0.007 at step
3 where the shift assembly held 0.13, and 3 of 10 steps fell back to the
smoothed force against none. Three hundred iterations of a matvec accurate
to `sqrt(eps)` is enough room for a 1e-7 reordering of the sum to matter, and
it mattered there. So `newton_direction` pins its own matvec to the shift
assembly, whatever the backend. The force evaluation around it stays indexed.
With that pin the tutorial run has no fallbacks, and it costs the shift
step, 39 s, not the indexed 31 s.

| Tutorial 4, 10 steps | per step | fallbacks | cosine, minimum |
|---|---|---|---|
| CPU, shift | 21.8 s | 0/10 | 0.20 |
| MPS, shift | 38.9 s | 0/10 | 0.13 |
| MPS, indexed, before the pin | 31.3 s | 3/10 | 0.007 |

At `(10,16,16)` p=2 the GPU loses to the CPU either way. That is the small
mesh, where it has lost before.

On energy removed per wall second, Newton loses to L-BFGS from both states
this was measured, and the default is unchanged. From the equilibrium field,
20 L-BFGS steps remove 4.9e-5 in 26.5 s (1.8e-6 per second) and the force
goes from 5.5e-2 to 7.9e-2; two Newton steps remove 3.1e-5 in 102.5 s
(3.0e-7 per second, one of the two a fallback) and the force goes to 0.19.
From Tutorial 4's step-500 field both are at the float32 floor of the
energy: 50 L-BFGS steps remove 3.0e-8 in 21.3 s and the force goes from
1.19e-3 to 1.14e-3, and 3 Newton steps remove 7.4e-8 in 119 s with no
fallbacks while the force stays at 9.8e-4. Newton does not earn its cost
later in this relaxation. The time that was left to take back was in the
L-BFGS step, whose largest solves are the k=1 Hodge hat solves; see "The
remaining speedups" below.

### The floor, measured directly

Steady ms per apply at `(12,24,12)` p=3:

| primitive | CPU | MPS |
|---|---|---|
| `E` apply k=0 / k=1 / k=2 | 0.040 / 0.032 / 0.037 | 0.246 / 0.223 / 0.216 |
| `E^T` apply k=0 / k=1 / k=2 | 0.054 / 0.053 / 0.052 | 0.248 / 0.240 / 0.231 |
| `mass_core_apply` k=1 | 1.269 | 1.868 |
| `apply_derivative_matrix` k=1 | 1.779 | 9.172 |
| `apply_laplacian` k=1 free (nested CG) | 284.257 | **74.916** |

Read the MPS column across the first two rows: six different operations over
three form degrees and two array sizes, all between 0.216 and 0.248 ms. A cost
that ignores what it is computing is dispatch, not arithmetic. The CPU does
the same work in 0.03-0.05 ms, so those applies lose 4-7x; the nested CG is
large enough to bury the floor and wins 3.8x.

A compiled relaxation chunk is ~21,000 StableHLO ops of which only 1,271 are
`dot_general` -- the rest is slice, gather, reshape and concatenate. 94% data
movement against a fixed per-dispatch charge is the entire result.

Per-apply cost, `scripts/benchmark/matvec_bench.py`, the `scan` column
(the form the relaxation actually runs), ms per apply:

| operator | `(8,12,12)` p=2 cpu / mps | `(16,32,16)` p=3 cpu / mps |
|---|---|---|
| `apply_mass_matrix` k=0 | 0.18 / 0.44 | 1.01 / 0.58 |
| `apply_mass_matrix` k=1 | 0.43 / 1.09 | 2.51 / 1.60 |
| `apply_stiffness` k=0 | 0.43 / 1.16 | 2.77 / 1.58 |
| `apply_stiffness` k=1 | 0.32 / 1.09 | 2.54 / 1.50 |
| `apply_derivative D^T D` k=0 | 0.85 / 2.05 | 5.51 / 3.13 |
| `apply_derivative D^T D` k=2 | 0.12 / 0.73 | 1.90 / 0.86 |
| `E E^T` k=0 | 0.006 / 0.083 | 0.024 / 0.094 |
| mass atom k=0 | 0.004 / 0.135 | 0.022 / 0.134 |

Two things are visible. The heavy applies cross over between the two meshes
and settle at about 1.7x in the GPU's favour at the larger one. That table
predates the indexed mass kernel. Five L-BFGS steps at `(16,32,16)` p=3
after it, and after the Hodge cap, best of three with the machine idle:
**15.6 s on the GPU (3.12 s/step; 15.7, 16.5 and 15.6 s) against 37.8 s on
the CPU (7.56 s/step; 39.3, 37.9 and 37.8 s)**, 2.4x. An earlier single pair,
14.7 s against 39.8 s, gave 2.7x; the GPU number was a fast draw. The GPU is
already ahead at `(12,24,12)`. The light
ones never cross over, because MPS has a per-kernel floor of roughly 0.08 ms
that the CPU does not: `E E^T` and the mass atom cost the same on the GPU at
both resolutions, which is the signature of a cost that is dispatch and not
arithmetic. Those light applies are the preconditioner, so a Krylov iteration
pays the floor once per iteration no matter how large the mesh is.

The implication for a real run is that the mesh has to be large enough that
the sum-factorised applies dominate the preconditioner, and `(8,12,12)` p=2
is far below that. Do not port a laptop-sized test case to the GPU and expect
a speed-up.

## `MRX_ASSEMBLY`

```bash
MRX_ASSEMBLY=indexed   # one gather and one segment_sum per element
MRX_ASSEMBLY=shift     # the dense shifted copies, the TPU and CPU form
```

Unset, Metal takes `indexed` and every other backend takes `shift`, from
`jax.default_backend()`. The two compute the same operator: the gather is
the same integer map and agrees exactly, and the assembly is the same sum in
a different order and agrees to float32 roundoff. `shift` traces the
kernel every non-Metal run has always traced, so nothing about a CPU or GPU
result moves.

It is a static branch of the one mass kernel, not a second code path through
the solvers. One L-BFGS step applies that kernel 2,757 times at k=1 and
1,228 times at k=2 before the Hodge cap, which is why a 2.5x on the kernel is
a 2.1x on the step (593.9 s down to 283.8 s) and is what puts the GPU ahead
of the CPU on that method. After the cap the same step applies it 769 and 334
times. The Newton matvec does not take the branch: see above.

## Chunk size no longer matters

Compile is still uncacheable: the persistent cache writes zero entries on
this plugin and a second run recompiles in full, where the same script on the
CPU writes 1118 entries and recompiles 2.7x faster. It used to also be
*linear* in the chunk, about 7.9 s per step of chunk, which made `--chunk 25`
worth about 2.3x against the L-BFGS default of 500.

That was the program from before the indexed kernel and the Hodge cap. Two
chunks back to back, same mesh, L-BFGS, now:

| chunk | compile (first chunk minus second) | steady s/step |
|---|---|---|
| 5 | 1.9 s | 0.92 |
| 10 | 0.5 s | 1.13 |
| 25 | 3.1 s | 0.99 |

The differences sit inside the 17% run-to-run spread. Compile no longer
grows with the chunk. In plain float32 the default chunk is 10, because
that is the window the energy-floor stop reads.

Every other knob was swept and none of them helped; `mps/env.sh` records which
and by how much, so the sweep does not need repeating. `mps/env_sweep.py` runs
it again if a version bump makes that worth doing.

## Trusting a GPU run

Plain float32 relaxation is sensitive to roundoff, and the two backends
separate. Over 100 L-BFGS steps on li383 `(12,24,12)` p=3, the force residual
agrees to 2e-6 at step 1 -- roundoff carried through solves whose own
tolerance is `sqrt(eps) = 3.5e-4` -- then grows past 1% by step 12, and by
step 100 the runs are on different trajectories. The indexed and shifted
forms on the GPU itself agree to 4e-6 at step 1 and then separate the same
way, which is the check that the faster kernel is the same operator.

That is the descent amplifying a roundoff-sized perturbation, not a backend
defect, and the conserved quantities say both runs are physical: helicity
drifts by 3.3e-5 relative on the GPU and 7.3e-5 on the CPU, `||div B||`
stays at 1.5e-06 and 1.2e-06, and the energy decreases on 91 and 81 of 100
steps. Do not read the helicity figures as the GPU being more accurate; they
are two samples of a chaotic trajectory. Compare backends on invariants and
on converged states, never step by step.

## `JAX_MPS_ASYNC_DISPATCH=1`

The plugin suggests this on startup and it is worth knowing what it does and
does not do. It roughly halves the cost of *eager* applies -- `apply_stiffness`
k=0 goes from 6.46 ms to 2.84 ms, `D^T D` k=1 from 12.55 ms to 5.54 ms at
`(16,32,16)` p=3 -- and leaves the `jit` and `scan` columns unchanged within
noise. Production runs the relaxation step as one jitted `lax.scan`, so the
flag buys nothing there; interactive and script use, which dispatches applies
from Python one at a time, roughly doubles in speed. It halved the wall clock
of the benchmark itself.

It is documented as experimental. Nothing in the suite was run with it.

## Known limits

- Plain float32 only. There is no iterative refinement on this backend, so
  the accuracy discussion in {doc}`concepts/precision` about what float32
  alone does applies in full.
- The persistent compilation cache is inert here: every run recompiles. Keep
  the chunk small, as above.
- A per-dispatch floor of about 0.22 ms. It is what the indexed mass
  assembly is avoiding, and it is still what the small extraction applies
  pay. It is not the `lax.scan` trip-count growth of
  jax-mps [#215](https://github.com/tillahoffmann/jax-mps/issues/215); that
  was tested with `mps/scan_scaling.py` and does not reach MRX, on either
  backend, at either mesh.
- `MLX_METAL_FAST_SYNCH` aborts with `[metal::Device] Unable to load kernel
  input_coherent` on 0.10.11. `MLX_ENABLE_TF32` must never be set: it undoes
  the `jax_default_matmul_precision='highest'` that
  {doc}`concepts/precision` relies on.
- Single device. The plugin exposes one `MpsDevice` and no collectives, so
  `scripts/pmap_sweep.py` and the sharded paths have nothing to spread over.
- The plugin is experimental and prints so on import. It pins a jaxlib minor
  version; upgrading `jax` in `mrx_mps` without upgrading `jax-mps` to match
  breaks bytecode deserialization at the first compile.
- `eigh`, `qr` and `svd` are executed on the CPU inside MLX rather than on
  the GPU. MRX only reaches them at preconditioner build time, so this costs
  a transfer once per geometry.
