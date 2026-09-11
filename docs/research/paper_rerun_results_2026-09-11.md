# The paper's cheap block rerun on the released method, 2026-09-11

Tobias: "Launch the cheap jobs. store results in a new separate folder.
20 GPUh approved." Twenty-nine arms on e680ab4 (the refinement fix; the
head is ae9630b, additive), li383, (16,32,32) p=2 unless stated, mixed
precision, smoothing before the L-BFGS combination (the code's only form),
potential velocity route except the Leray comparison rows, CFL 0.5,
`--floor-tol 0`, chunk 500. Records: `outputs/paper_rerun_2026-09-11/<arm>/`
of the `runtimes` worktree (relax.json + checkpoints), job logs under
`outputs/paper_rerun_jobs/2026-09-11/`. Block cost 19.5 GPU h (sacct).
The midpoint scheme is omitted from the paper (Tobias, 17:00); its four
arms ran before the decision and are not tabulated. Tables by
`pr_table.py` (tmp of this session); the paper's numbers in parentheses.

## Findings

1. **The two velocity routes agree to three digits where the paper said
   they do** (m=0 and gamma=0 rows: 11.37 / 11.37, 38.8 / 39.5, 98 / 97),
   and Leray with smoothing reaches 1.71e-8 (paper's smooth-first Leray
   row 2.08e-8). The potential route with smoothing stands at 5.84e-8 at
   5000 steps against the paper's 1.18e-8: the two runs coincide step for
   step through 1500 steps (392 / 80 / 98 vs 394 / 80 / 95 per 500), then
   diverge from round-off after the spike every m=1 arm has at steps
   1000-1500 (Leray too: 1210 vs 1241 x1e-8 chunk max). At 10 000 the
   anchor is at 2.0e-8, at 18 000 at 0.35e-8. **A 5000-step residual
   scatters by x3-5 between realisations of one method**; the velocity
   and smoothing tables cannot resolve below that at 5000 steps. Report
   10 000-step residuals, or the minimum over the run, or say so.
2. **Helicity correction: exact.** Explicit step, float64, 1000 steps:
   dH/H = -3.5e-15 with the correction, -1.5e-6 with the auxiliary field,
   -1.0e-5 with the plain field; mixed: -1.9e-7 / -1.0e-6 / -1.0e-5. On
   the 1.4m file the plain drift is 7x smaller than the paper's on the
   low-res reference file (7-8e-5), so the table's numbers change with the
   geometry file alone.
3. **The tolerance-1e-6 spike did not recur.** 32 000 steps on the
   potential route: helicity flat at -9.4e-6, residual monotone to
   0.13e-8, the lowest of every arm; tol 1e-10 reaches 0.36e-8 at 14 500,
   the anchor (1e-8) 0.35e-8 at 18 000. The spike paragraph, the "600x"
   comparison and the figure's story go; the tolerance figure becomes
   "1e-6 to 1e-10 indistinguishable in the residual and the helicity",
   with 1e-6 at half the cost (0.206 vs 0.297 vs 0.426 s/step).
4. **m = 5 reconnects**: helicity +1.25e-4 over the run, the jump at steps
   8000-8500 with a residual spike to 8e-6, residual 14e-8 at the end
   against 0.8e-8 before the event. m = 0 is fine (2.5e-8 at 18 000,
   -1.1e-5). The m-sweep paragraph changes: m = 5 is no longer "similar".
5. **Seeded (5,1) arm reconnects**: an event at steps 1000-1500 (chunk max
   2.2e-4), helicity drifting up to +4.4e-6 (paper -9.0e-6), residual
   27.5e-8 at 10 000 (paper 8.4e-8). The (6,1) arm is fine: 1.28e-8
   (paper 2.93e-8), helicity -8.9e-6, one event at 8000 of +1e-6. The
   seeded table's (5,1) row and its width claim need the trace (running).
6. **Smoothing-constant sweep under smooth-first**: c = 0.0064 reconnects
   in its last chunk (residual 2.4e-6, helicity +4e-7), c = 0.64 is the
   lowest at 5000 steps (3.6e-8 against the paper's 11e-8), c = 0.02-0.2
   sit at 5.8-10e-8. With the x3-5 scatter, the sweep at 5000 steps
   orders only the two ends; the "flat optimum 0.02-0.064" is not
   supported, and the s/step column now rises 0.21 -> 0.63 with c.
7. **n = 12**: 1.08e-8 at 20 000 steps, helicity -2.0e-5, 0.168 s/step.
8. **Cost**: potential route with smoothing 0.297 s/step, Leray 0.417,
   order 0 0.21-0.26; the whole block 19.5 GPU h.

## tab:velocity_choices (5000 steps; residual = last-1000 mean x1e8)

| row | route | s/step today (paper) | F2 today (paper) | dE/E today (paper) | dH/H today (paper) |
|---|---|---|---|---|---|
| m=1, g=1 | potential | 0.297 (0.451) | 5.84 (1.18) | -3.86 (-3.94) | -10.31 (-11.1) |
| m=1, g=1 | Leray (smooth first) | 0.417 (0.643) | 1.71 (2.08) | -3.86 (-3.94) | -10.41 (-11.4) |
| m=0, g=1 | potential | 0.301 (0.45) | 11.37 (11.4) | -3.42 (-3.3) | -6.88 (-6.97) |
| m=0, g=1 | Leray | 0.415 (0.645) | 11.37 (11.4) | -3.33 (-3.26) | -7.25 (-7.06) |
| m=1, g=0 | potential | 0.214 (0.318) | 38.83 (39.8) | -4.01 (-4.02) | -17.01 (-17.0) |
| m=1, g=0 | Leray | 0.249 (0.398) | 39.54 (39.6) | -4.06 (-4.0) | -16.91 (-17.3) |
| m=0, g=0 | potential | 0.259 (0.383) | 98.28 (94.4) | -3.85 (-3.74) | -1.67 (-1.12) |
| m=0, g=0 | Leray | 0.225 (0.355) | 97.12 (101) | -3.85 (-3.66) | -1.67 (-1.67) |

## tab:seeded (10 000 steps)

| seed | s/step today (paper) | F2 today (paper) | dE/E today (paper) | dH/H today (paper) | F2 min |
|---|---|---|---|---|---|
| none | 0.297 (0.654) | 2.01 (2.27) | -3.92 (-3.76) | -9.85 (-9.94) | 1.55 |
| (6,1) | 0.296 (0.655) | 1.28 (2.93) | -4.22 (-3.95) | -8.92 (-9.29) | 0.82 |
| (5,1) | 0.297 (0.629) | 27.54 (8.36) | -6.73 (-5.08) | 4.37 (-9.01) | 17.24 |

## tab:epsilon_sweep (5000 steps; paper F2 / dE e6 / dH e5 / s/step)

| c | s/step today (paper) | F2 today (paper) | dE/E today (paper) | dH/H x1e5 today (paper) |
|---|---|---|---|---|
| 0 | 0.214 (0.4) | 38.8 (45) | -4.01 (-4.0) | -1.70 (-1.7) |
| 0.0064 | 0.290 (0.61) | 243.5 (7.8) | -4.58 (-3.76) | 0.04 (-0.9) |
| 0.02 | 0.297 (0.69) | 5.8 (5.3) | -3.86 (-3.66) | -1.03 (-0.9) |
| 0.064 | 0.307 (0.78) | 9.1 (7.3) | -3.50 (-3.46) | -0.85 (-0.8) |
| 0.2 | 0.378 (0.91) | 10.0 (10) | -3.42 (-3.4) | -0.74 (-0.7) |
| 0.64 | 0.629 (1.11) | 3.6 (11) | -3.76 (-3.32) | -0.98 (-0.7) |

## sweeps: m, tol, h (full length; residual = last-1000 mean, min over the run; dH over the run)

| arm | steps | s/step today (paper) | F2 last 1000 | F2 min | dH/H x1e6 (paper bound) |
|---|---|---|---|---|---|
| m0 | 18000 | 0.299 (0.645) | 2.49 | 2.41 | -10.6 (<= 1.9e-5 in all sweeps) |
| anchor_pot_m1g1 | 18000 | 0.297 (0.672) | 0.35 | 0.30 | -9.8 () |
| m5 | 17000 | 0.320 (0.691) | 14.26 | 0.78 | +125.1 () |
| tol1e-6 | 32000 | 0.206 (0.42) | 0.13 | 0.12 | -9.4 (+2.6e-5 with the spike) |
| tol1e-10 | 14500 | 0.426 (0.808) | 0.36 | 0.31 | -10.6 () |
| h12 | 20000 | 0.168 (0.339) | 1.08 | 0.92 | -20.4 (-1.9e-5) |

## tab:helicity_preservation, explicit rows (1000 steps, no smoothing, 1.4m file; paper: low-res reference file)

| precision | field | dH/H today | paper |
|---|---|---|---|
| float64 | B | -1.03e-05 | 7.19e-05 |
| float64 | aux H | -1.46e-06 | 1.99e-05 |
| float64 | B + helicity correction | -3.46e-15 | -- |
| mixed | B | -1.02e-05 | 8.46e-05 |
| mixed | aux H | -1.02e-06 | 3.04e-05 |
| mixed | B + helicity correction | -1.86e-07 | -- |

## Seeded arms: sections and widths (traces 18390819/22, `scripts/poincare_trace.py`, five planes; the paper's measure)

| arm | width ic | width final | change (paper) | chaotic lines ic -> final (paper) | the other chain at the end |
|---|---|---|---|---|---|
| (6,1), iota = 1/2 | 0.1637 rho = 2.62 h_r | 0.1630 = 2.61 h_r | -0.4% (+2.0%) | 2 -> 9 (1 -> 4) | 3/5: 0.023 rho = 0.37 h_r |
| (5,1), iota = 3/5 | 0.1455 = 2.33 h_r | 0.1394 = 2.23 h_r | -4.2% (+4.7%) | 1 -> 10 (0 -> 1) | 1/2: 0.034 rho = 0.54 h_r |

Both seeded chains survive within the paper's "+-5%" (2.6 and 2.2 h_r,
sections `<arm>/poincare/poincare_final_zeta0.375.pdf`). New against the
paper: the end states carry 9-10 chaotic lines instead of 1-4, and the
unseeded resonance of each arm opens a small chain (0.4-0.5 h_r), the 1/2
chain of the (5,1) arm being the reconnection its helicity drift (+2e-5)
recorded. The unseeded anchor's sections at step 10 000 are traced
separately (job 18391109).

Unseeded anchor (potential route, `anchor_pot_m1g1`, traced at steps 10 000
and 18 000, job 18391109; frames in `anchor_pot_m1g1/poincare/`): at
10 000 both resonances carry a grid-scale chain, 1/2 at 0.36 h_r and 3/5 at
0.35 h_r, five chaotic lines (the paper's anchor: 3/5 at 0.47 h_r, no 1/2
chain); at 18 000: 0.25 and 0.36 h_r, four chaotic lines. So the seeded
chains at 2.2-2.6 h_r stand six to seven times above the unseeded field's,
as in the paper.
