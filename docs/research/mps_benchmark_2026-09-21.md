# MRX on an Apple GPU: where the time goes

An M3 Pro, macOS 27, `jax-mps` 0.10.11 on jax 0.10.x, the plain float32
configuration (`MRX_X64=0`) throughout. The shift-assembly numbers are MRX
at `a1d0b26`; the indexed-assembly numbers are the change recorded below,
on the same machine. The reference mesh is li383 `(12,24,12)` p=3,
deliberately the one behind the v5e / H200 / H100 table in
`tpu_v5e_benchmark.md`.

## The headline

The number to quote for a plain float32 relaxation is the time to the
energy floor, not a fixed step count. L-BFGS reaches it in about two
minutes at `(12,24,12)` p=3 and about four minutes at `(16,32,16)` p=3,
against the old default of 3000 steps, which is about an hour and cannot
stop: `--floor-tol 1e-8` is a squared residual float32 never reaches.
The table is in "Time to the float32 floor", below. The rows here are the
earlier measurement, from before the Hodge cap and before that stop.

The GPU now wins the L-BFGS relaxation, which is not the default method.
The mass kernel was a dozen dense shifts where Metal wants one indexed read;
with that swapped, and only on Metal, 100 L-BFGS steps at this mesh take
283.8 s on the GPU against 352.5 s on the CPU. Newton is the default, and
its matvec is pinned back to the shift assembly: on Tutorial 4's mesh the
indexed form changed the direction. Both are below.

| measurement | CPU | MPS | |
|---|---|---|---|
| `relax.py`, 100 steps, indexed assembly | 352.5 s | **283.8 s** | MPS 1.24x |
| the same run with the shift assembly | **353.7 s** | 593.9 s | CPU 1.68x |
| per step, indexed | 3.52 s | **2.84 s** | MPS 1.24x |
| per step, shift assembly | 3.54 s | 5.94 s | CPU 1.68x |
| per step, `relaxation_bench.py`, shift assembly | 3.68 s | 4.72 s | CPU 1.28x |
| per step, `scan_scaling.py` chunk 5, shift assembly | 3.59 s | 5.51 s | CPU 1.53x |
| one nested-CG Laplacian solve, k=1 | 284.3 ms | **74.9 ms** | MPS 3.8x |
| inverse mass CG k=1, 20 iterations | 587 ms | **158 ms** | MPS 3.7x |
| inverse mass CG k=2, 24 iterations | 674 ms | **158 ms** | MPS 4.3x |

The per-step numbers disagree with each other by more than a quiet machine
would allow: the run-to-run spread of a fixed configuration here is about
17%, so nothing is claimed on a smaller margin than that. The two
end-to-end rows are the ones to quote. The CPU figure reproduced: the shift
assembly re-run on the same revision came back at 352.5 s against 353.7 s
the first time, and the printed trajectory matched to the last digit, which
is the check that `shift` is the kernel it was.

## The assembly change

`_structured_gather` and `_structured_accumulate` exist because a TPU has no
fast path for indexed memory. Measured on a v5e they were 33x faster than a
gather and a `segment_sum`, and that is the form the kernel has shipped
since. It is the wrong trade against a per-dispatch charge. One compiled
mass apply lowers to 115-126 `slice`s and 61-66 `concatenate`s on top of its
12 `dot_general`s; the indexed form is 6 gathers, 6 scatters and 1
concatenate, with the same 12 `dot_general`s. The index is built with NumPy
from the static shift plan at trace time, so it is a constant of the
executable and `SumfactPlan` does not change.

Inside a jitted scan, milliseconds per apply, best of three:

| | CPU shift | CPU indexed | MPS shift | MPS indexed |
|---|---|---|---|---|
| k=1 | 1.312 | 1.231 (1.07x) | 1.226 | **0.503 (2.44x)** |
| k=2 | 1.134 | 1.205 (0.94x) | 1.200 | **0.475 (2.52x)** |

The decision rule was fixed first: land it only if the indexed form beat
the shift form on MPS by more than the 17% noise. It beat it by 2.5x. On
the CPU it is inside the noise in both directions, so the CPU, and a TPU,
keep the shifts. A per-axis indexed form (one gather per axis rather than
one for the element) also won on MPS, by 1.9x, and lost to the flat form,
so it was not landed. `mps/assembly_ab.py` is the comparison.

What a step actually calls, counted with a host callback on one step so the
Krylov iterations are real ones: before the Hodge cap, the mass kernel 2,757
times at k=1 and 1,228 times at k=2. Priced at the in-scan costs above, that
is 4.9 s of a 5.9 s step, and the indexed form brings it down by 2.9 s. The
measured step went from 5.94 s to 2.84 s. The model and the run agree. After
the cap the same count is 769 at k=1 and 334 at k=2, and the extraction
applies fall from 10,545 to 2,883 (`mps/attribute_step.py`). A Newton step
with the three mass solves applied the k=1 mass kernel 22,645 times; the
merged matvec does two of those three inverses.

The other leaves do not deserve the same treatment. Inside a scan the
incidence stencils cost 0.06-0.14 ms and the mass preconditioner 0.27 ms,
against 1.2 ms for the mass kernel; the 5x the derivative lost as a
standalone call was the mass kernel inside it plus a launch, not the
stencil. 10,545 extraction applies remain, each on the dispatch floor, and
they are the next place the floor shows.

`MRX_ASSEMBLY=indexed` is the default on Metal and `shift` everywhere else,
from `jax.default_backend()`. It is a static argument of `_sumfact_kernel`,
so the two forms do not share a compilation cache entry and the shift trace
is the trace it has always been. The gather agrees exactly and the assembly
agrees to float32 roundoff (`test_indexed_assembly_matches_the_shift_form`). Over
the 100 steps the indexed and shifted GPU runs agree to 4e-6 in the force
at step 1 and then separate, the same float32 chaos as GPU against CPU.

## Newton, measured after the assembly change

`relax.py` defaults to `--method newton`. A step is 300 MINRES iterations
with no criterion of their own (`tol=0.0` inside `newton_direction`), and
each matvec was three k=1 mass solves and is now two (see below), so the mass kernel is a larger share
of a Newton step than of an L-BFGS step. `newton_tol=0.1` is the truncation,
not a target the Laplacian atom reaches, so `newton_it` is `+300`.
`--floor-tol 1e-8` cannot fire in float32; these runs pass `--floor-tol 0`.
The float32 default is now that, plus the energy-floor stop below.

The suite, re-run after the kernel change: 61 passed in 150 s on MPS with
the indexed assembly, and 61 passed in 300 s on the CPU in the default mixed
configuration, which still traces `shift`. `test_hessian.py` on MPS: the
second variation matches the energy derivatives, and the Newton direction
has `||div u|| = 1.3e-9` and a descent cosine of +0.59.

Two Newton steps from the equilibrium field at `(12,24,12)` p=3, before the
matvec was pinned, so this is the indexed kernel against the shift kernel on
the same step:

| | CPU shift | MPS shift | MPS indexed |
|---|---|---|---|
| wall, 2 steps | 101.3 s | 120.9 s | 86.7 s |
| per step | 50.6 s | 60.5 s | 43.4 s |
| fallbacks | 1/2 | 1/2 | 1/2 |

Step 1 agrees between indexed and shift to 8e-7 in `|F|` and 3e-7 in the
descent cosine (0.994 on all three). The indexed kernel is 1.39x the shift
kernel and 1.17x the CPU. This is the mesh where the change is honest.

Tutorial 4 is not. `(10,16,16)` p=2, 10 steps, warm-started from
`data/tutorials/li383_relaxation/checkpoints/state_000500.h5`:

| | per step | fallbacks | minimum descent cosine |
|---|---|---|---|
| CPU, shift | 21.8 s | 0/10 | 0.20 |
| MPS, shift | 38.9 s | 0/10 | 0.13 |
| MPS, indexed | 31.3 s | 3/10 | 0.007 |

The indexed and shift forces agree at step 1 (9.811e-4 against 9.810e-4).
The directions do not: cosine 0.186 against 0.242 at step 1, 0.007 against
0.127 at step 3, and then three fallbacks and a force spike to 2.9e-3 that
the shift run never has. A matvec good to `sqrt(eps) = 3.5e-4`, repeated 300
times, makes a 1e-7 reordering visible. `newton_direction` therefore pins
`_assembly_override` to `shift` for its own trace. The force around it stays
on the backend's assembly. Re-run with the pin, five steps on the tutorial
mesh: no fallbacks, 39.0 s/step, the shift cost. The GPU still loses to the
CPU at this resolution, indexed or not.

The index constants are not the cost. Distinct component plans total 3.56 MB
at `(12,24,12)` p=3 and 9.13 MB at `(16,32,16)` p=3, under a millisecond to
build in NumPy and about 5 ms to copy to the device, once per trace.

## Three levers after the indexed assembly

The rule for landing any of them was the same 17% bar, at both k=1 and k=2,
inside a dependent scan of 20 applies. None of the kernel changes cleared it.
One solve change did.

**Fusing the three component gathers does not.** `indexed_fused` in
`mps/assembly_ab.py` concatenates the three element indices into one gather
and one `segment_sum`. The components have different shapes, so they are not
padded. The lowered module goes from 6 gathers and 6 scatters to 2 and 2,
and the 12 `dot_general`s stay. On MPS the time goes from 0.499 ms to 0.493 ms
at k=1 (1%) and from 0.604 ms to 0.519 ms at k=2 (16%). The op count fell and
the time did not, which is MLX already fusing the three gathers. On the CPU
the fused form is slower (0.89x and effectively tied). Not landed.

**Folding `E` into the element index does not survive the timing.** A step
issues 10,545 extraction applies. Standalone, `E` and `E^T` cost about 0.09 ms
each on MPS inside a scan, and 0.09 ms times 10,545 is about 1.0 s, more than
the 0.84 s of a step that is not the mass kernel. Inside `apply_mass_matrix`
the pair is already fused: the full apply is 0.580 ms against a core of
0.513 ms at k=1, and 0.566 against 0.526 at k=2. A composed kernel that
gathers the extracted DoFs directly agrees with the sandwich to 1e-7 and
measures 1.19x and 1.03x on MPS, 0.83x and 0.74x on the CPU. Short of 17% at
k=2, and a regression on the CPU. Not landed. It is `_composed_mass` in
`mps/assembly_ab.py`.

**The Hodge split owns the preconditioner traffic, and capping its outer loop
does.** `mps/solve_attr.py` tags every Laplacian-atom apply by the solve that
built it. All 1,482 in one L-BFGS step are the Hodge split
(`apply_inverse_laplacian_hodge`), which is the potential-velocity force.
The shifted-stiffness atom, 61 applies, is the velocity smoothing and nothing
else. The 1,482 are six passes of the outer `refine` loop. In plain float32
that loop's later passes are the ones it discards when a pass stops
improving the residual, and each discarded pass has already paid a CG whose
largest solve was measured at 121 to 442 iterations. `apply_inverse_laplacian_hodge`
now asks `_pair_loop` for two passes when there is no residual-precision view,
and leaves the full budget of six in mixed precision, where each pass gains
the inner tolerance. The count drops to 393. `test_poisson.py` and
`test_relaxation.py` pass in float32 (14 passed, 193 s) and in the default
mixed configuration (14 passed, 328 s). The same 20 L-BFGS steps with the
cap lifted (six passes) take 61.2 s and remove 4.865e-5 of energy against
4.874e-5 with the cap; the final force agrees to 1% and the first five
steps' residuals to 0.2%. The trajectories then wander (one step's residual
differs by 18%) because the line search sees a slightly different force, and
they finish in the same place. The later passes were not buying accuracy.

The same waste is not in the other solves. With the cap in place, one L-BFGS
step's refinement passes were 2/2 improving for the Hodge split, 2/2 for the
saddle, 1/1 for the smoothing and 5/5 for the mass inverse
(`mps/solve_attr.py`). The mass inverse uses almost the whole budget and
every pass lowers the residual, so `MAX_PASSES` stays 6. The cap belongs on
the Hodge split, which is where the discarded passes were.

## The Newton matvec, measured and not landed

`second_variation` was tried with its k=1 masses inverted by one apply of the
mass atom instead of a nested solve. That option was tried and removed. The atom
is a fixed linear SPD map, which the nested solve in plain float32 is not,
and a step gets much cheaper: two steps from the equilibrium initial
condition at `(12,24,12)` p=3 are about 100-130 s with the solve and
23.8 s with the atom (12 s/step). The solve was re-timed idle, best of
three: 102.5 s, and the three runs span 102.5, 128.9 and 126.1 s, wider
than the 17% bar, so the earlier 108.4 s was inside that noise and a Newton
step from this state is about a minute. It is not the same step. On Tutorial
4's field (`(10,16,16)` p=2, the shipped step-500 state) the solve does not
fall back, and one idle step is 40.3 s best of three (40.3, 41.0, 41.4 s),
which confirms the earlier 39.7 s. The step itself moves between those
runs: `dt*` 1.20, 0.93 and 0.99, energy removed 3.2e-8, 2.1e-8 and 1.8e-8.
The single earlier sample (`dt*` 1.67, cosine 0.242, 6.78e-8) is the same
kind of step and is not a stable target. The atom, measured once, points
similarly (cosine 0.229) and then parts: `dt*` 0.050 and 2.83e-10 of energy,
in 2.9 s. From the initial condition the trajectories split at step 1 (the
solve falls back to the smoothed force, the atom does not, and over two
steps the solve removes about 3.1e-5 against the atom's 1.73e-5). The option was tried and removed. The Newton matvec therefore
stays the nested solve, and it stays pinned to the shift assembly: that pin
is what kept Tutorial 4 from falling back, and a fixed linear matvec was the
condition for removing it.

Two of those three inverses are the same one. ``dJ`` and ``W`` are both
``M_1^-1`` of a dual 1-form and both enter only through ``load(B x .)``, so
one solve for ``X = dJ + W / 2`` replaces them. On a random 2-form the merged
apply agrees with the three-solve apply to 1e-5 at `(12,24,12)` p=3 and to
2e-6 at `(10,16,16)` p=2, inside the solve tolerance of 3.5e-4, and the
energy identity still holds. A separate Newton step does not repeat to 1e-7:
the three-solve's own tutorial steps, back to back, already span `dt*` 0.93
to 1.20 and a cosine 0.13 to 0.19. The merged step sits in that same range
(`dt*` 1.20, 1.49, 1.25; cosine 0.17 to 0.20; no fallbacks; `|F|` agrees to
1e-7), and from the initial condition it falls back on the same first step
with the same `dt*` 0.014 and removes the same 3.1e-5 over two steps.

The wall clock does move. One tutorial step, best of three: 27.1 s against
40.3 s, 1.5x, and the two ranges do not overlap (27.1-28.2 s against
40.3-41.4 s). Two steps from the initial condition: 72.9 s best of three
(72.9, 100.4, 98.5 s) against 102.5 s (102.5, 128.9, 126.1 s), 1.4x.
Summing the two remaining loads pointwise and integrating once, which is the
same dual vector, does not: one tutorial step took 27.8 s.

A fixed Chebyshev inverse of the same mass, preconditioned by the atom, does
not become the default either. Lanczos of the atom-preconditioned `M_1` on
the tutorial mesh gives the spectrum `[6.9e-2, 2.6]`, and the degree that
puts the error bound at `sqrt(eps)` is 27. One apply of that inverse matches
a mass solve to the tolerance on the test fixture. On the tutorial field it
does not take the solve's step. Three runs, no fallbacks: `dt*` 1.45 to 1.46,
which is inside the solve's own spread, but the energy removed is 4.6e-7
against the solve's 3e-8, about 15 times more, and the gate was 10%. A step
that removes more energy is still a different step, so the Chebyshev option
was tried and removed and the shift pin stays: the condition
for removing it was this gate. The wall was 24.4 s best of three, inside the
noise of the merged solve's 27 s, so it was not faster either.

The MINRES budget stays 300. Five steps from the tutorial field, the merged
solve, no fallbacks at any of these:

| `newton_maxiter` | wall | energy change | force |
|---|---|---|---|
| 100 | 44.9 s | removed 8.8e-8 | 9.81e-4, unchanged |
| 200 | 87.8 s | rose 7.5e-8 | 9.81e-4, unchanged |
| 300 | 129.5 s | rose 2.0e-7 | 9.81e-4, unchanged |

The force does not move and every energy change is a few ulps of an energy
near 0.5, so 100 iterations "winning" on energy per second is the sign of
roundoff, not a faster relaxation. The wall scales with the budget. The
default is unchanged.

`jax-mps` 0.11.0 with jax 0.11.2, in a clone of the environment
(`mrx_mps011`; `mrx_mps` is still jax 0.10.2 and jax-mps 0.10.11), runs the
same tutorial Newton step in 27.6 s. That is inside the 27.1-28.2 s of
0.10.11, so the newer plugin does not move the dispatch-bound part of this
step. The base environment was not changed.

The harmonic preconditioner does not become the default either. On the same
tutorial field, with the shipped matvec, one step is 40 s with the
Laplacian atom (no fallback; `dt*` and the energy move between runs, see
above) and 40.2 s
with the harmonic atom, which falls back to the smoothed force and removes
1.6e-9. With the mass-atom matvec the harmonic step does remove more of the
line search's energy per second (1.1e-8 in 3.0 s against 2.8e-10 in 2.9 s
for the Laplacian atom), and that matvec is not what a run uses. From the
equilibrium initial condition at `(12,24,12)` p=3 the harmonic atom falls
back as well, so the energy it removes is the smoothed force's. The default
stays `laplacian`.

Twenty L-BFGS steps at `(12,24,12)` p=3, `--floor-tol 0`, after the cap:

| | CPU, shift | MPS, indexed |
|---|---|---|
| 20 steps, best of 3, idle | **40.3 s** | **26.5 s** |
| per step | 2.02 s | **1.32 s** |
| the three runs | 40.3, 40.7, 40.3 s | 27.2, 27.6, 26.5 s |

An earlier pair, 54.8 s and 41.9 s, was taken while other jobs were running.
The 100-step averages above (3.52 s and 2.84 s) are from before the cap, and
a 20-step window is not the same average as a 100-step one. The GPU is 1.5x
the CPU on this window, and both remove 4.9e-5 of energy. The one idle run
with the cap lifted took 61.2 s, so the cap is about 2.3x, not the 1.46x
that overlapping pair implied. There is no fused-assembly column: that form
was not landed, so the Metal default is still `indexed`.

## Which method removes energy

Energy removed per wall second, Metal, indexed, the machine idle. Newton is
still the default in `scripts/relax.py`; this does not change it.

| state | method | steps | wall | energy removed | per second | force |
|---|---|---|---|---|---|---|
| equilibrium IC, `(12,24,12)` p=3 | L-BFGS | 20 | 26.5 s | 4.89e-5 | 1.8e-6 | 5.5e-2 to 7.9e-2 |
| same | Newton | 2 | 102.5 s | 3.09e-5 | 3.0e-7 | 5.5e-2 to 0.19 |
| Tutorial 4 step 500, `(10,16,16)` p=2 | L-BFGS | 50 | 21.3 s | 3.0e-8 | 1.4e-9 | 1.19e-3 to 1.14e-3 |
| same | Newton | 3 | 119 s | 7.4e-8 | 6.2e-10 | 9.8e-4, unchanged |

From the initial condition L-BFGS removes about 6x more energy per second.
The Newton run falls back on its first step, and its force grows further.
From the step-500 field both are at the float32 floor of an energy near
0.5, the force does not come down, and L-BFGS is still ahead per second.
Newton does not earn its minute later in this relaxation. The time left to
take back is in the L-BFGS step. One such step's largest solves are the
Hodge `L^_1` hat solves, 138-141 CG iterations (`mps/solve_attr.py`), and
the lever on those is the k=1 Laplacian atom.

## Time to the float32 floor

The residual is the wrong stop in plain float32. At `(12,24,12)` p=3 it
oscillates between about `1e-3` and `1e-2` and later blows up, while the
energy has already stopped descending and then walks uphill. That split
is `docs/research/floor_study_2026-09-05.md`. The per-step `dE` is the
increment `<B_{n+1} - B_n, M(B_{n+1} + B_n)> / 2`, an increment against an
`O(1)` field, so the trace resolves a change below one ulp of `E ≈ 0.5`.
`relax` stopped when the last two chunks of 10 had each changed the energy
by at most `max(0.01 * removed, 4 * eps * |E0|)`. On these meshes the
`0.01 * removed` term binds, at about `6e-7`. (That rule stops the
production mesh 8% short. It was replaced on 2026-09-23 by two chunks that
each raise the energy on 3 of 10 steps or lower it by at most
`5e-4 * removed`; see the last section.) The run returns the
lowest-energy field seen at a chunk boundary. Going further is a restart
in `--precision mixed`, where a float64 residual keeps converging and
this stop stays off. It is also off in plain float64. It is on for any
plain float32 backend, not only Metal.

A 300-step L-BFGS calibration at `(12,24,12)`, stop off, chunk 10, reached
6.10e-5 at step 94. The same trace would have exited at step 90 and
returned the step-80 field, 6.02e-5, 1.4% less. By step 300 the energy
had risen by 1.2e-3. At `(16,32,16)` the 150-step calibration peaked at
6.24e-5 at step 136; the exit on that trace is step 140, returning step
130 at 6.22e-5. Ten Newton steps on the smaller mesh, 362 s, removed
4.32e-5 and were still descending. L-BFGS had removed 4.89e-5 by step 20,
in about 40 s. The default method was not changed.

With the stop on, from the equilibrium field, chunk 10, one sample per
50 steps. Wall is the compiled steps. Best of three:

| | exit | wall, best of 3 | the three runs | energy removed |
|---|---|---|---|---|
| MPS `(12,24,12)` | steps 70, 90, 100 | 108 s | 108, 139, 144 s | 5.77e-5, 5.98e-5, 6.19e-5 |
| CPU, shift, `(12,24,12)` | step 90 | 122 s | 123, 122, 122 s | 6.05e-5 |
| MPS `(16,32,16)` | step 130 | 255 s | 286, 255, 277 s | 6.32e-5, 6.29e-5, 6.22e-5 |
| CPU, shift, `(16,32,16)` | step 130 | 432 s | 432, 432, 451 s | 6.31e-5 |

The 108 s run is the trajectory that removed 5.77e-5. It peaked there,
0.2% above the field the exit returned, and a longer run of that same
trajectory does not get the 6.10e-5 the calibration found. The run that
removed 6.19e-5 took 144 s. The old 3000-step budget at the idle
1.32 s/step is 66 min, about 37x the fast exit and 28x the slow one.
At `(16,32,16)` the three exits averaged 2.0 s/step, so 3000 steps is
about 100 min against 255 s, about 24x. The CPU energy repeated to the
last digit; Metal's did not, which is the same float32 separation as the
rest of this note. The spread of the time to the floor is the step the
windows go flat. On a fixed step count the clock spread is still about
17%, and the per-step times inside one mesh sit inside it.

Sampling every chunk cost 33 s beside 149 s of steps at the step-90
exit. Every 50 steps, the other time was 9 s beside 108 s. A sample is a
full force evaluation, and the stop does not use it, so the float32
default is one sample per about 50 steps. The saving is inside the 17%
spread. The per-step residual probe, `||grad(B^2/2)||` by an `M_0` solve
inside the scan, was 33.2 s and 33.1 s for the same 20 L-BFGS steps. It
stays in the step.

Newton, then still the default, does not share the floor. On Metal at
`(12,24,12)` the exit was step 50, 1795 s, 35.9 s/step, 5.48e-5 removed.
Another run of 30 steps, 996 s, was still descending (5.70e-5, last
window 1.7e-6). The old budget of 100 steps is about an hour at 36 s/step,
so this exit is about 2x, not 20x, and the energy is short of the L-BFGS
floor. On the CPU, 20 Newton steps took 435 s (21.8 s/step) and removed
4.57e-5 without stalling. A step-50 exit at that rate is about 18 min,
against 122 s for L-BFGS. Neither Newton figure is a best of three.
`(16,32,16)` was not run to a Newton floor. The default method was left
as it is then; it is L-BFGS in plain float32 since 2026-09-23 (last section).

## The per-dispatch floor, measured

`relaxation_bench.py` primitives at `(12,24,12)` p=3, steady ms per apply:

| primitive | CPU | MPS | MPS/CPU |
|---|---|---|---|
| `E` apply k=0 | 0.040 | 0.246 | 6.15x |
| `E` apply k=1 | 0.032 | 0.223 | 6.97x |
| `E` apply k=2 | 0.037 | 0.216 | 5.84x |
| `E^T` apply k=0 | 0.054 | 0.248 | 4.59x |
| `E^T` apply k=1 | 0.053 | 0.240 | 4.53x |
| `E^T` apply k=2 | 0.052 | 0.231 | 4.44x |
| mass gather `x[gather_idx]` | 0.043 | 0.167 | 3.88x |
| mass scatter `segment_sum` | 0.121 | 0.247 | 2.04x |
| `mass_core_apply` k=0 | 0.535 | 0.697 | 1.30x |
| `mass_core_apply` k=1 | 1.269 | 1.868 | 1.47x |
| `mass_core_apply` k=2 | 1.280 | 1.969 | 1.54x |
| `apply_derivative_matrix` k=0 | 1.544 | 5.001 | 3.24x |
| `apply_derivative_matrix` k=1 | 1.779 | 9.172 | 5.16x |
| `apply_derivative_matrix` k=2 | 0.817 | 4.247 | 5.20x |
| `apply_laplacian` k=0 free | 2.324 | 5.666 | 2.44x |
| `apply_laplacian` k=1 free (nested CG) | 284.257 | 74.916 | **0.26x** |

Read the MPS column down the first six rows: 0.246, 0.223, 0.216, 0.248,
0.240, 0.231. Three different form degrees, two different directions, two
different array sizes, and the cost does not move. **That flat ~0.22 ms is the
per-dispatch floor**, and it is not arithmetic -- the CPU does the same work
in 0.03-0.05 ms. Everything else in the table follows from it: an operation
smaller than the floor loses by whatever margin it is smaller, and the one
operation large enough to bury it wins 3.8x.

This is why the suite disagrees with itself. `test_poisson.py` is 24.8 s on
MPS against 45.3 s on CPU, because a Poisson solve is almost entirely the
bottom row. A relaxation step is a few hundred of the top rows.

## The op census

`JAX_MPS_DUMP_OPTIMIZED_IR=<dir>` writes the post-optimisation StableHLO. One
compiled relaxation chunk is ~21,000 ops:

| op | count | | op | count |
|---|---|---|---|---|
| `slice` | 3681 | | `select` | 1317 |
| `gather` | 3060 | | `dot_general` | **1271** |
| `reshape` | 2658 | | `transpose` | 882 |
| `concatenate` | 1969 | | `multiply` | 859 |
| `broadcast_in_dim` | 1759 | | `pad` | 464 |
| `add` | 1428 | | `scatter` | 352 |

1,271 of ~21,000 ops are the arithmetic. The other 94% is data movement that
XLA fuses into its kernels and that MLX, on this evidence, does not fuse as
aggressively. That ratio multiplied by a fixed per-dispatch cost is the whole
result, and it says the lever is op count in `mrx/`, not configuration.

## What was measured and rejected

**`lax.scan` trip count (jax-mps [#215](https://github.com/tillahoffmann/jax-mps/issues/215)) -- refuted.**
The issue reports per-iteration cost growing with the trip count, which would
have made the relaxation quadratic in the chunk. `mps/scan_scaling.py` at
`(8,12,12)` p=2, marginal ms per step between successive chunk lengths:

| chunk | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| CPU | -- | 175.6 | 177.1 | 147.3 | 170.2 |
| MPS | -- | 641.1 | 632.6 | 562.5 | 532.8 |

Both flat; if anything MPS improves. At `(12,24,12)` p=3 MPS gave 5506.8 and
5504.8 ms/step at chunks 5 and 10, which is flat to 0.04%. The effect does not
reach MRX. This also withdraws the library change that was planned on the
strength of it -- consolidating `chunk_runner`'s 13 separately stacked trace
scalars into one array -- which would have been unjustified work.

**The MLX command-buffer knobs -- no effect.** Against a fixed chunk, ms/step:
baseline 584.9, `ops=200+mb=1024` 548.7, `async_dispatch` 603.5,
`ops_per_buffer=50` 685.2, `=200` 703.8, `=1000` 729.3. The baseline
re-measured on its own gave 614.1 and 525.9, so the entire spread of the sweep
is the noise of the baseline. `MLX_METAL_FAST_SYNCH` is not merely useless but
broken on this build: it aborts with `[metal::Device] Unable to load kernel
input_coherent`.

**Controls confirming the stack is already working.** `JAX_MPS_NO_OPTIMIZE=1`
costs 1.28x (752.2 ms/step) and `MLX_DISABLE_COMPILE=1` costs 2.0x (1148.3
ms/step). The plugin's IR passes and MLX's kernel fusion are both on and both
earning their keep, which is a large part of why there is no configuration win
left to find.

**`jax-mps` 0.10.12.dev847 -- no movement.** 673.5 and 609.0 ms/step against
0.10.11's 614.1 and 525.9. Inside noise, trending slightly worse. Per the
gate set in advance, 0.11.x was not tried.

## The compilation cache does not work on MPS

Same script, same JAX cache settings, two consecutive runs:

| backend | cache entries written | compile, run 1 | compile, run 2 |
|---|---|---|---|
| CPU | 1118 | 9.02 s | 3.37 s |
| MPS | **0** | 7.50 s | 8.58 s |

The plugin never populates the persistent cache, so every MPS run recompiles
from scratch. Before the indexed assembly and the Hodge cap, that compile was
linear in the chunk, about 7.9 s per step of chunk, and `--chunk 25` instead
of the L-BFGS default of 500 was worth about 2.3x on a 500-step run.

It is not, now. Two chunks back to back at `(12,24,12)` p=3, L-BFGS, the
first chunk minus the second is the compile, and the second is the step:

| chunk | first chunk | second chunk | compile | s/step |
|---|---|---|---|---|
| 5 | 6.5 s | 4.6 s | 1.9 s | 0.92 |
| 10 | 11.8 s | 11.3 s | 0.5 s | 1.13 |
| 25 | 27.9 s | 24.9 s | 3.1 s | 0.99 |

The compile is 1-3 s and does not grow with the chunk, and the step is flat
at about 1 s. A 500-step run is about 8-10 minutes whichever of these is
chosen. The program those 7.9 s were measured on applied the Laplacian atom
1,482 times a step; this one applies it 393.

## Where the GPU pulls ahead

Five L-BFGS steps at `(16,32,16)` p=3, the mesh where the heavy applies were
winning 1.7x before the indexed kernel, best of three with the machine
idle: **15.6 s on the GPU (3.12 s/step) against 37.8 s on the CPU
(7.56 s/step)**, 2.4x. One earlier pair, 14.7 s against 39.8 s, read 2.7x
because the GPU run was a fast draw. The GPU was already ahead at
`(12,24,12)` (1.32 s/step against 2.02 s on the idle 20-step window after
the cap), so the crossover is below that mesh. A Newton step's
calls, from the equilibrium initial condition, are a different shape: the
direction was three nested k=1 mass solves (22,645 mass-kernel applies
and 27,148 mass-atom applies; the merged matvec is two of the three) and
then this particular step falls back to the smoothed force. That is why a
Newton step stays near a minute while an L-BFGS step is near a second. The
merge brings one tutorial step from 40.3 s to 27.1 s.

## The physics agrees; the trajectories do not

Both backends ran li383 `(12,24,12)` p=3, L-BFGS, 100 steps, chunk 25. The
force residual by step, and its relative difference:

| step | 1 | 2 | 5 | 10 | 12 | 30 | 100 |
|---|---|---|---|---|---|---|---|
| rel. diff in \|F\| | 6.0e-06 | 2.5e-04 | 1.3e-04 | 4.7e-04 | >1e-02 | 6.9e-01 | 1.6e-01 |

Step 1 agrees to 6e-6. That is float32 roundoff carried through solves whose
own tolerance is `sqrt(eps) = 3.5e-4`, so the two backends are computing the
same step. From there the difference grows exponentially and passes 1% at step
12: a gradient descent run at a loose tolerance amplifies a roundoff-sized
perturbation, and after 100 steps the two runs are on different trajectories
(`||J||/||B||` ends at 0.82 on MPS against 1.57 on CPU).

This is a property of plain float32 relaxation, not of the backend, and the
conserved quantities confirm it -- both runs stay physical:

| | MPS | CPU |
|---|---|---|
| energy removed | 6.03e-05 (0.0121%) | 3.55e-05 (0.0071%) |
| helicity drift, relative | -2.70e-06 | +7.33e-05 |
| `\|\|div B\|\|` max | 1.49e-06 | 1.24e-06 |
| energy increases | 16/100 steps | 19/100 steps |

Do not read the helicity column as the GPU being more accurate: these are two
samples of a chaotic trajectory, not a convergence study. What it does support
is that neither run is diverging or losing solenoidality, i.e. the GPU result
is a valid relaxation and not a broken one.

## What would actually make this faster

Not configuration -- that surface is exhausted above, and the two controls
show the plugin and MLX are already extracting what there is. The measurement
points at one thing: **94% of the ops in a relaxation chunk are data
movement**, and this backend charges ~0.22 ms per dispatch for them. Cutting
the slice / gather / reshape / concatenate count in `mrx/` is the lever, which
is the same kind of work as the data-movement rewrites in
[PR #17](https://github.com/ToBlick/mrx/pull/17) and is out of scope here.

Worth noting what that would be worth: it cannot make the GPU slower than it
already is at the small applies, and the nested-CG row shows a 3.8x waiting on
the other side of the floor. The hardware is not the limit.

## Reproducing

```bash
source mps/env.sh
python mps/probe_ops.py                        # 14 constructs vs CPU
python mps/scan_scaling.py                     # the #215 test
python mps/env_sweep.py                        # the knob sweep
python scripts/benchmark/relaxation_bench.py --ns 12,24,12 --p 3 \
    --precision float32 --relax-steps 5
python scripts/relax.py --ns 12,24,12 --p 3 --precision float32 \
    --method lbfgs --steps 100 --chunk 25
# plain float32 now stops on the energy floor (chunk 10, floor-tol 0),
# with L-BFGS and the one-pass force solve by default:
python scripts/relax.py --ns 12,24,12 --p 3 --precision float32 --steps 200
# the Hodge solve's share of one step, and its cost with k=1 bands / two passes:
python mps/hodge_share.py 12,24,12 3 [bands [passes]]
```

Every CPU comparison needs `JAX_PLATFORMS=cpu` explicitly, because installing
the plugin makes `mps` the default backend for the whole environment.

## The remaining speedups (2026-09-23)

The tables are in `docs/source/mps.md`, "The remaining speedups". In short:

- **Landed: one outer pass for the force's k=1 Hodge solve in plain
  float32** (`TimeStepper.force_hodge_passes`, `--force-hodge-passes`).
  The solve was 41-59% of the Metal step and 85% of the CPU one
  (`mps/hodge_share.py`), and its second float32 pass cost as much as the
  first. The solve is warm-started from the last step's potential, so one
  pass takes the same energy steps. The compiled step is 1.6-1.8x faster at
  `(12,24,12)`, `(16,32,16)` and `(16,32,32)` on both backends. Time to the
  floor at `(12,24,12)`, best of three: Metal 108 s to 62.9 s, CPU 122 s
  to 89.6 s, with more energy removed. A cold solve still needs two passes
  (`test_poisson.py`), so the global cap is unchanged.
- **Landed: the floor rule.** The production calibration, `(16,32,32)`
  p=2, 2000 steps, descends monotonically for 800 steps on a tail the old
  band called a floor at step 120 (92.4% of the best). The new rule counts
  the steps that raise the energy. It fires at step 570 on that trace
  (99.7%) and where the old one did on the benchmark meshes. The production
  run with both changes stops at step 620, 599 s, 7.24e-5 removed (99.6%
  of the 2000-step best, which took 3653 s).
- **Landed: L-BFGS is the plain float32 default method.** Newton removes
  half the energy in the same wall time at `(16,32,16)` on both backends
  (Metal 5 steps, 204 s, 2.93e-5; CPU 10 steps, 478 s, 3.28e-5), and less
  than L-BFGS after 30 minutes at `(12,24,12)`.
- **Not taken:** a preconditioner and harmonic-form cache (setup plus
  compile is 15-17% of a benchmark floor run; about 10% saved); radial bands
  on the k=1 Dirichlet atom (Metal slower at `(12,24,12)`, 5% at
  `(16,32,32)`, 15% on the CPU after the one pass); history 3, 5, 10, cfl
  1.0, smoothing 0.5x and 2x (the best two 10-11% faster with less energy,
  inside the spread); `--warm-from` coarse to fine (1.1% more energy, 956 s
  to its floor against 599 s; at equal energy less than the spread ahead).
  The warm start stays an option.
