# Handoff 2026-09-03: the chunked relaxation loop and the reconnect controller

Decided with Tobias on 2026-09-03 (li383 session, branch li383-followups). Status
at the end of this document.

## What is being changed and why

`scripts/relax.py` (the driver of every li383 arm) dispatches ONE jitted
relaxation step per Python iteration and converts fourteen scalars to Python
floats after each, a device-to-host sync per step. It does that for a
per-step trace (||F||, the line-search identity, dt against dt*, cos, gain) and
for the floor and stall detectors that read it. `mrx.relaxation.relaxation_loop`
(the tutorials, the lean test) already does the right thing, a `lax.scan` of
`num_iters_inner` steps per dispatch, but records nothing per step. Two loops
around one step, the driver's being the slow one on small meshes.

Measured on `li383_pub/h16_p2_g1` ((16,32,32) p = 2 gamma = 1, the rung of the
paper), the per-step trace is not needed for any decision:

| statistic of the residual at the floor | value |
|---|---|
| scatter within a 100-step chunk (coefficient of variation) | 2% |
| a single sample against its chunk mean, median / p90 | 1.6% / 8% |
| change of consecutive 100-step chunk means | 0.4% (p90 0.8%) |
| drop per 500-step chunk along the run | 43, 11, 7, 6, 3.5, 3.9, 2.4, 1.8, 1.7% |
| drop per 1000-step chunk | 38, 11, 6.7, 3.8% |
| one QoI sample (helicity solve, force, weak pressure, beta) | 3.4 s, against 58 s per 100 steps |

So: a chunk MEAN is quiet (a single sample is not, 8% tails), the descent is
a slow power law rather than a plateau, and "stalled" is a rate test that two
consecutive 500-step chunk means resolve cleanly (2% per chunk against a few
tenths of a percent of noise). QoI, snapshot, checkpoint and save every chunk
cost 6% at n = 16. There is no reason for separate cadences.

## The decisions

1. **One inner loop in the library**: `chunk_runner(ts, n_chunk, eta_of_step,
   extra)` in `mrx/relaxation.py` returns a jitted `run(state, it0) ->
   (state, trace)`: `lax.scan` over `n_chunk` steps whose body returns the
   per-step scalars as the scan's stacked output (`trace`, a dict of
   `(n_chunk,)` arrays) while the state (B, the L-BFGS pair, the guesses) is
   the carry and comes out once. Scalars: E, F, v, dt, dt_star, cfl, div, Fu
   (`F_prev . M v`, the line-search identity), eta, res_it, res_delta, plus
   whatever `extra[name](state)` adds (the driver adds `resid = F /
   ||grad(B^2/2)||`). `eta_of_step(it)` is a traced function of the step
   count (the schedules become `jnp`), applied in the body; the resistive
   clock is reset while eta is 0, as before. Compile time is that of the body
   and independent of `n_chunk` (a `While` trip count); no `unroll`.
2. `relaxation_loop` is rewritten on the runner with its signature and its
   per-chunk traces unchanged (tutorials and `test/test_relaxation.py` keep
   working); it just no longer owns a scan of its own.
3. **The driver keeps its own thin outer loop** (60 lines: QoI, save, stall,
   reconnect, floor, budget) on the runner. Making it the `callback` of
   `relaxation_loop` was considered and rejected: the callback would have to
   replace the state, stop the run and own the printing, and the library
   loop's own record/print would duplicate it.
4. **One cadence, `--chunk N` (default 500)**: trace transfer, QoI sample,
   snapshot, checkpoint, save, stall test, floor test and wall-time test all
   at the chunk boundary. `--qoi-every`, `--save-every`, `--floor-steps`,
   `--stall-steps` are deleted (not deprecated). `--steps` must be a multiple
   of `--chunk`. Snapshots (for movies) are always stored, one per chunk.
   The first QoI sample is taken at `it0` before the first chunk, so
   `qoi["helicity"][0]` is H_0 (previously the sample at step 1).
5. **Stall = two chunks**: stalled when the last chunk mean of the residual is
   above `(1 - --stall-tol)` times the previous chunk's, both chunks after
   the last reconnection; `--stall-tol` default 0.02 per chunk (the old 5% per
   1000 steps). Floor: last chunk mean below `--floor-tol`.
6. `gamma = 1` velocity smoothing is the default descent; chunk statistics
   are calibrated on gamma = 1 arms only (gamma = 0 scatters more).
7. **Upper limit on N** is wall time per chunk, nothing in JAX: work lost on a
   kill (up to one chunk, the checkpoint sits at the boundary), controller
   latency (a stall is acted on at the next boundary), visibility. N = 500 is
   5 min at n = 16 (0.58 s/step) and 38 min at n = 32 (4.6 s/step); keep N
   fixed across meshes since the stall threshold is per chunk. Reverse-mode
   differentiation through the loop would store N carries and need
   `jax.checkpoint` on the body; nobody does that today.

## The reconnect controller (already in, b449e97; adapted to chunks here)

`--reconnect`: run until stalled, write the stalled equilibrium to
`<out>/stalls/<k>/B.h5` (layout of `B.h5`, `poincare_relax.py` reads it) and
`state.eqx` (a `--restart` file), one backward-Euler solve `(M + eps L) delta
= -eps L B` with `eps = --reconnect-eps h^2` (c = 0.01: a tenth of a cell, the
scale the h-independent ideal floor is made of; helicity price exact, dH =
-2 eps int J.B, 2.3% of H_0 per stall at n = 8, about 0.6% at n = 16),
`initial_state` on the diffused field, carry on. No accept/reject, no dose
adaptation. `results["stalls"]` records step, floor, eps, |F|, helicity, J/B
and the pressures before and after. Smoke at (8,16,16): floors 5.8e-4, 3.3e-4,
2.9e-4 at three stalls (section 5h of `li383_sweep_results_2026-09-02.md`).
Full arm `reconnect_h16_p2_g1` running on the pre-chunk driver (job 17419921).

## Files

- `mrx/relaxation.py`: `chunk_runner`; `relaxation_loop` on top of it.
- `scripts/relax.py`: chunk loop; `eta_schedule` returns a traced function;
  `--chunk`; flags deleted as above; docstring usage block and the
  `--reconnect` / `--checkpoint` paragraphs updated; end-of-run summary uses
  the last chunk.
- `scripts/li383_pub.sh`: drop `--save-every` / `--qoi-every` /
  `--stall-steps` from COMMON and the arms; smoke `--chunk 100`.
- `docs/source` mentions of the deleted flags, if any.
- The figure script reads `trace` and `qoi` by key and is unaffected
  (`qoi["it"]` now starts at `it0`).

## Verification (all on GPU through `slurm/run.sh`)

1. ruff, import, `--help`.
2. Checkpoint chain: 200 steps against 120 + 80 with a pulse, float32
   (8,16,16), `--chunk 10` (the job-scratch `ckpt_smoke.py` with the new
   flags): identical schedule and step accounting, trajectories within the
   run-to-run round-off.
3. `reconnect smoke` at (8,16,16), `--chunk 100 --steps 3000`: a series of
   stalls like the pre-chunk smoke's; `stalls/<k>/` populated; relax.json has
   `stalls`; the figure script renders the stall table.
4. Lean suite (`test/`, about 4 min): `test_relaxation` unchanged.
5. A 1000-step (16,32,32) gamma = 1 arm with `--chunk 500` against
   `h16_p2_g1`'s first 1000 steps: chunk means of ||F|| within the
   round-off scatter (identical arithmetic per step; only the cadence moved).

## Status (2026-09-03, evening)

Landed in b54a0ff on li383-followups (branch pushed). Verification, all on
GPU:

1. ruff, import, `--help`, the traced schedules (pulse windows, tanh, linear
   checked on CPU), the multiple-of-chunk check: pass.
2. Checkpoint chain, float32 (8,16,16) `--chunk 20`, 200 against 120 + 80
   with a pulse on steps 150 .. 169: restart at 120, identical schedule and
   accounting, "SMOKE OK".
3. `reconnect_smoke` ((8,16,16) p = 2, `--chunk 100`, 3000 steps): four
   stalls at 900, 1500, 2100, 2700 with their `stalls/<k>/` files, the qoi
   from step 0, a snapshot per chunk, the stall table rendered by the figure
   script. Stalls 1 and 4 were declared on chunks where the residual ROSE
   (5.1e-4 -> 1.0e-3, 2.84e-4 -> 2.98e-4): the rate test counts a rise as
   "not descending", which is right, and at N = 100 on a gamma = 0 coarse
   mesh such excursions are visible. The calibration holds for N = 500,
   gamma = 1.
4. Lean suite: 51 passed in 325 s on the worktree.
5. `chunk_h16` (1000 steps, `--chunk 500`) against `h16_p2_g1`'s first 1000
   steps: chunk means of ||F|| within 0.2% and 0.9%, energy within 1e-9,
   line-search identity 2e-9 -- identical arithmetic, only the cadence moved.
6. Same-node A/B (old driver / new / old, 1000 steps each, second 500
   timed): 0.557 / 0.563 / 0.558 s per step. No cost; the 0.66 s/step of a
   first run was node variation.

The full reconnect arm `reconnect_h16_p2_g1` (8000 steps, still the per-step
driver, job 17419921) stalled at 3948 / 5691 / 7423 with floors 5.55e-4 /
4.61e-4 / 4.24e-4 at 0.63 / 0.62 / 0.62% of H_0 per solve; section 5h of
`li383_sweep_results_2026-09-02.md` has the table and the sections.

Open: nothing in the loop itself. The reconnect arms of the paper should be
rerun on the chunked driver if their per-step trace matters for a figure
(the stall test differs: two 500-step chunks instead of 200-step blocks over
1000 steps), otherwise the existing arm stands.
