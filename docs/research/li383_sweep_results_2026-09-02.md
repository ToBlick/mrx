# li383 (NCSX) relaxation: sweep results and seeded islands -- 2026-09-02

Status: reruns and seeded arms running (section 4, 5 filled as they land).
Read it for: every li383 relaxation number that exists, which arms back which figure, the seeded-island result.
Do not read it for: the solver internals (`shifted_split_2026-09-02.md`) or the wout reader (`docs/source/concepts/`).

## 1. Case and settings

- Reference `data/wout_li383_low_res_reference.nc`: NCSX li383, nfp = 3, ns = 16, iota 0.405 (axis) .. 0.656 (edge). Resonances on the way out: iota = 1/2 at rho = 0.55, 6/11 at 0.68, 9/16 at 0.72, 3/5 at 0.80.
- Second reference `data/wout_li383_1.4m.nc`: same case, ns = 49, iota 0.394 .. 0.655 (1/2 at rho 0.54, 3/5 at 0.79).
- IC `--ic clebsch`: `B = dA'` from the wout series in closed form, div-free and wall-tangent exactly. Every wout run of 2026-08-28/29 used the reader BEFORE the per-mode axis conditions (3406d38 ff.); `li383_axisfix` and everything from 2026-09-01 on use the current one.
- Descent: L-BFGS history 1 (identical to the earlier `--method cg`), analytic line search, CFL 0.5, eta = 0, stop when the 100-step mean of `||F||_M` drops below 1e-3 or at 6000 steps, `B` and `p_w` saved every 100 steps.
- gamma = 1: `v = (I - mu L)^{-1} F` with mu = 0.064 / n_r^2 (4.4e-4 on 12 radial cells, 2.5e-4 on 16, 1.1e-4 on 24).
- Seed (`--seed m,n,rho0,width --seed-eps eps`): resonant term `eps |Phi'(rho0)| / m  g(rho) cos 2 pi (m theta - n zeta)` in `A'_zeta`, `g` Gaussian of the given width tapered to zero at the wall; eps = `|dB^rho| / |B^zeta|` at rho0; the chain sits at |iota| = nfp n / m; island full width about `1.6 sqrt(eps nfp / (m |iota'|))` in rho. Under ideal descent the topology is frozen, so a seeded island that grows to an eps-independent width marks a tearing-unstable surface, one that shrinks back a stable one.
- Poincaré: 160 lines, 400 crossings per line, traced in float64, three planes zeta = 0, 0.25, 0.5; iota from the fit, chaotic lines flagged by the h/2 drift.
- Runs: `scripts/li383_sweep.sh` (2026-08 sweep, branch li383-sweep, merged into static-dynamic-refactor 2e93424; its `--method cg` no longer parses) and `scripts/li383_pub.sh` (this note's reruns and seeds, run against e815a86). Ledgers `outputs/li383_sweep/jobs.tsv`, `outputs/li383_pub/jobs.tsv`.

## 2. The 2026-08 sweep (pre axis-fix reader)

`outputs/li383_sweep/<arm>/relax.json`, sections in `<arm>/poincare/`, movies (31 frames, zeta = 0.5) for `r12_p3_g0` and `r12_p3_g1`. s/step excludes setup; dH/H_0 is the relative helicity change over the run; chaotic = lines flagged by the h/2 drift, out of 160.

| arm | ns | p | g | prec | steps | stop | s/step | `||F||` 0 -> end | dH/H_0 | beta_vol | chaotic (final) | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r12_p1_g0 | 12,24,12 | 1 | 0 | f32 | 4025 | floor | 0.09 | 1.26e-1 -> 2.13e-3 | -1.2e-3 | 0.0212 | -- | 0.11 |
| r12_p2_g0 | 12,24,12 | 2 | 0 | f32 | 6000 | steps | 0.19 | 5.24e-2 -> 1.99e-3 | +3.1e-5 | 0.0413 | 11 | 0.33 |
| r12_p3_g0 | 12,24,12 | 3 | 0 | f32 | 6000 | steps | 0.33 | 5.50e-2 -> 1.11e-3 | +5.3e-5 | 0.0422 | 3 | 0.56 |
| r12_p3_g0_f64 | 12,24,12 | 3 | 0 | f64 | 6000 | steps | 1.10 | 5.50e-2 -> 1.30e-3 | -1.8e-5 | 0.0422 | 3 | 1.85 |
| r12_p3_g1 | 12,24,12 | 3 | 1 | f32 | 6000 | steps | 1.14 | 5.50e-2 -> 1.87e-3 | -3.4e-5 | 0.0420 | 3 | 1.91 |
| r12_p3_g1_f64 | 12,24,12 | 3 | 1 | f64 | 3000 | steps | 2.11 | 5.50e-2 -> 2.28e-3 | +7.1e-6 | 0.0417 | 5 | 1.77 |
| r12_p4_g0 | 12,24,12 | 4 | 0 | f32 | 6000 | steps | 0.54 | 5.61e-2 -> 2.97e-3 | +3.4e-5 | 0.0427 | 6 | 0.92 |
| r12_p4_g1 | 12,24,12 | 4 | 1 | f32 | 6000 | steps | 1.53 | 5.62e-2 -> 2.13e-3 | -6.1e-5 | 0.0416 | 8 | 2.56 |
| r16_p2_g0 | 16,32,16 | 2 | 0 | f32 | 6000 | steps | 0.31 | 5.53e-2 -> 1.86e-3 | -5.0e-5 | 0.0417 | 2 | 0.53 |
| r16_p3_g0 | 16,32,16 | 3 | 0 | f32 | 6000 | steps | 0.61 | 5.60e-2 -> 2.83e-3 | +3.9e-5 | 0.0423 | 4 | 1.03 |
| r16_p3_g1 | 16,32,16 | 3 | 1 | f32 | 6000 | steps | 1.85 | 5.60e-2 -> 2.63e-3 | -6.3e-5 | 0.0418 | 3 | 3.11 |
| r16_p4_g0 | 16,32,16 | 4 | 0 | f32 | 6000 | steps | 1.10 | 5.63e-2 -> 7.30e-3 | -3.1e-5 | 0.0425 | 12 | 1.86 |
| r24_p3_g0 | 24,48,24 | 3 | 0 | f32 | 6000 | steps | 1.84 | 5.63e-2 -> 1.63e-2 | -3.0e-5 | 0.0407 | 10 | 3.11 |
| r24_p3_g1 | 24,48,24 | 3 | 1 | f32 | 5630 | seconds | 5.27 | 5.63e-2 -> 3.63e-3 | -1.5e-4 | 0.0412 | 12 | 8.28 |

- Energy: every arm releases (1.0 .. 1.5)e-4 of E_0 (`r12_p3_g0`: 0.49999994 -> 0.49992573). The VMEC state is within 1e-4 of the relaxed energy; the descent works on the residual and the topology, not on the energy.
- Helicity (eta = 0): |dH/H_0| <= 6e-5 in all arms bar `r12_p1_g0` (1.2e-3) and `r24_p3_g1` (1.5e-4). The drift is not monotone in h at fixed steps (see `gamma1-velocity-smoothing-status`: never compare dH/H_0 across meshes).
- Floor: at (12,24,12) every degree reaches (1 .. 3)e-3 in 4 .. 6 k steps; at (16,32,16) 2 .. 3e-3; at (24,48,24) the unsmoothed descent sits at 1.6e-2 after 6000 steps and only gamma = 1 gets to 3.6e-3. The floor is set by the reference (section 3), not by the mesh.
- Precision: f64 reproduces the f32 floor (1.30e-3 vs 1.11e-3) and helicity behaviour at 3.3x the cost per step.
- Topology (`r12_p3_g0`, zeta = 0.5 final): iota locks at 3/5 (5-lobe chain at the edge), 1/2 (mid radius), 6/11 and 9/16 (small plateaus), a ring of small islands next to the axis at 9/22; p_w stays a flux function; 0 of 160 lines lost, 3 chaotic. Final iota range 0.409 .. 0.659 (IC 0.403 .. 0.658).

Post-fix reruns of two arms (`outputs/li383_axisfix/`, L-BFGS): `r16_p3_g0` 6000 steps, 0.63 s/step, 5.58e-2 -> 3.31e-3, dH/H_0 -2.0e-5, 4 chaotic; `r24_p3_g0` 6000 steps, 2.05 s/step, 5.59e-2 -> 1.54e-2, dH/H_0 -6.1e-6. Same floors as before the fix.

## 3. The reference sets the floor

`outputs/li383_hires/r16_p2_g1` (`wout_li383_1.4m.nc`, ns = 49; (16,32,16) p = 2, gamma = 1, mu 2.5e-4, current reader): `||F||` 1.29e-2 -> 8.65e-4, floored at step 648 (1.36 s/step, 0.26 GPU-h), dH/H_0 -3.8e-6, beta_vol 0.0437, final iota 0.392 .. 0.660, 2 chaotic lines. The ns = 16 reference starts at 5.5e-2 and needs 4 .. 6 k steps to reach 1e-3; the ns = 49 reference starts 4x lower and floors below 1e-3 in 648.

The same-mesh comparison (`hi_r12_p3_g0` vs `r12_p3_g0`, both (12,24,12) p = 3, gamma = 0, current reader) is in section 4.

## 4. Reruns on the current reader (2026-09-02, `outputs/li383_pub/`)

Arms behind the figures of section 6, rerun against e815a86 with the settings of section 1. Budget 10 GPU-h.

| arm | ns | p | g | prec | reference | steps | stop | s/step | `||F||` 0 -> end | dH/H_0 | beta_vol | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r12_p1_g0 | 12,24,12 | 1 | 0 | f32 | ns 16 | | | | | | | | |
| r12_p2_g0 | 12,24,12 | 2 | 0 | f32 | ns 16 | | | | | | | | |
| r12_p3_g0 | 12,24,12 | 3 | 0 | f32 | ns 16 | | | | | | | | |
| r12_p4_g0 | 12,24,12 | 4 | 0 | f32 | ns 16 | | | | | | | | |
| r12_p3_g0_f64 | 12,24,12 | 3 | 0 | f64 | ns 16 | | | | | | | | |
| r12_p3_g1 | 12,24,12 | 3 | 1 | f32 | ns 16 | | | | | | | | |
| r16_p3_g1 | 16,32,16 | 3 | 1 | f32 | ns 16 | | | | | | | | |
| hi_r12_p3_g0 | 12,24,12 | 3 | 0 | f32 | ns 49 | | | | | | | | |

(`r16_p3_g0` and `r24_p3_g0` on the current reader: `outputs/li383_axisfix/`, section 2.)

## 5. Seeded islands (2026-09-02, `outputs/li383_pub/`)

All (12,24,12) p = 3 unless marked, ns = 16 reference, seed width 0.1. (6, 1) is the iota = 1/2 surface at rho 0.551, (5, 1) the 3/5 surface at rho 0.798. Predicted seed island widths (rho): (6, 1) 0.06 / 0.10 / 0.19 at eps 1e-3 / 3e-3 / 1e-2 (iota' 0.35); (5, 1) 0.13 at 3e-3 (iota' 0.28). Budget 10 GPU-h.

| arm | seed (m, n) | eps | g | ns | steps | stop | s/step | `||F||` 0 -> end | dH/H_0 | island width final (rho) | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s61_e1e-3_g0 | (6, 1) | 1e-3 | 0 | 12,24,12 | | | | | | | | |
| s61_e3e-3_g0 | (6, 1) | 3e-3 | 0 | 12,24,12 | | | | | | | | |
| s61_e1e-2_g0 | (6, 1) | 1e-2 | 0 | 12,24,12 | | | | | | | | |
| s61_e3e-3_g1 | (6, 1) | 3e-3 | 1 | 12,24,12 | | | | | | | | |
| s51_e3e-3_g0 | (5, 1) | 3e-3 | 0 | 12,24,12 | | | | | | | | |
| s51_e3e-3_g1 | (5, 1) | 3e-3 | 1 | 12,24,12 | | | | | | | | |
| r16_s61_e3e-3_g1 | (6, 1) | 3e-3 | 1 | 16,32,16 | | | | | | | | |

## 6. Figures for the paper

1. Case and IC: three Poincaré planes of the wout IC with the iota / p_w panel (`hi_r12_p3_g0/poincare/poincare_ic_*`).
2. Force residual vs step: degree scan at (12,24,12) gamma = 0; resolution scan at p = 3 gamma = 0 (r12, r16 from `li383_axisfix`, r24 from `li383_axisfix`); gamma = 0 vs 1 at r12 and r16. Generator: `li383_traces.py` (job scratch; to be moved to `scripts/`).
3. Reference floor: `r12_p3_g0` vs `hi_r12_p3_g0`, residual vs step on one axis (table row in section 4).
4. Energy released and helicity drift vs step, eta = 0 (same generator).
5. Final topology: zeta = 0.5 section of `r12_p3_g0` (logical chart + iota / p_w panel) and a frame strip from its movie (steps 0, 1000, 2000, 4000, 6000).
6. Precision: one table row (`r12_p3_g0` vs `r12_p3_g0_f64`).
7. Seeded islands: final zeta = 0.5 sections of the (6, 1) eps ladder, gamma = 0, with the island width vs eps; gamma = 1 arm alongside.
8. Optional solver row: smoothing solve on li383 p = 3, MINRES 2134 / 8478 / 20362 at (8,16,8) / (12,24,12) / (16,32,16) vs split + shifted-stiffness atom 145 / 249 (`shifted_split_2026-09-02.md`).

Not shown: `r24_p3_g0` (not floored in the step budget), `r16_p4_g0` (worst floor, 7.3e-3).

## 7. Open

- `r24_p3_g1` exists only on the old reader (8.3 GPU-h); rerun if the r24 rung goes in.
- The 2026-08-28 baseline `outputs/vmec_sections/li383_relaxed` is gone; `r12_p3_g0` of section 4 replaces it.
- Island width from the sections: measured by hand from the logical chart so far; a width extractor over `sections.npz` (O-point to X-point separation at fixed theta) would make the eps ladder quantitative.
