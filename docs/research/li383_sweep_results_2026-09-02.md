# li383 (NCSX) relaxation: sweep results and seeded islands -- 2026-09-02

Status: complete 2026-09-03 (reruns, seeded arms, floor-1e-4 arms, gamma = 1 h-sweep at p = 2; 25 GPU-h). Seeded arms follow the (n, 2n, 2n) rule of 2026-09-02; the reruns keep the (n, 2n, n) meshes of the 2026-08 sweep they reproduce. `outputs/li383_pub/README.md` explains that folder.
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
- Mesh: the 2026-08 sweep and its reruns use (n, 2n, n); since 2026-09-02 the default is (n, 2n, 2n) (`resolution-rule-n-2n-2n`), which the seeded arms use.
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
| r12_p1_g0 | 12,24,12 | 1 | 0 | f32 | ns 16 | 6000 | steps | 0.10 | 1.34e-01 -> 2.72e-03 | -3.8e-03 | 0.0209 | -- | 0.16 |
| r12_p2_g0 | 12,24,12 | 2 | 0 | f32 | ns 16 | 3147 | floor | 0.21 | 5.35e-02 -> 7.79e-04 | 5.9e-05 | 0.0422 | 10 | 0.18 |
| r12_p3_g0 | 12,24,12 | 3 | 0 | f32 | ns 16 | 5445 | floor | 0.36 | 5.54e-02 -> 8.08e-04 | 2.3e-05 | 0.0425 | 4 | 0.54 |
| r12_p4_g0 | 12,24,12 | 4 | 0 | f32 | ns 16 | 6000 | steps | 0.57 | 5.59e-02 -> 3.98e-03 | 2.0e-05 | 0.0427 | 4 | 0.94 |
| r12_p3_g0_f64 | 12,24,12 | 3 | 0 | f64 | ns 16 | 6000 | steps | 1.12 | 5.54e-02 -> 1.11e-03 | -3.4e-05 | 0.0422 | 1 | 1.86 |
| r12_p3_g1 | 12,24,12 | 3 | 1 | f32 | ns 16 | 5376 | floor | 0.39 | 5.54e-02 -> 8.56e-04 | 6.7e-05 | 0.0416 | 1 | 0.59 |
| r16_p3_g1 | 16,32,16 | 3 | 1 | f32 | ns 16 | 6000 | steps | 0.74 | 5.58e-02 -> 1.53e-03 | -1.0e-05 | 0.0410 | 3 | 1.23 |
| hi_r12_p3_g0 | 12,24,12 | 3 | 0 | f32 | ns 49 | 1654 | floor | 0.39 | 1.34e-02 -> 8.05e-04 | -6.2e-06 | 0.0427 | 3 | 0.18 |

(`r16_p3_g0` and `r24_p3_g0` on the current reader: `outputs/li383_axisfix/`, section 2.)

- The reader change helps: p = 2, 3 and gamma = 1 reach the 1e-3 floor in 3.1 .. 5.4 k steps where the old-reader runs used the 6000 steps at 1 .. 2e-3. p = 1 does not floor (helicity drift 4e-3, worst arm); p = 4 does not floor at either reader (4.0e-3 vs 3.0e-3 before).
- gamma = 1 costs 0.39 s/step against 0.36 unsmoothed at (12,24,12) (1.14 in August, MINRES smoothing solve) and floors in the same number of steps; at (16,32,16) it ends at 1.5e-3 in 6000 steps (2.6e-3 before).
- Reference floor at the same mesh (`figures/reference_floor.png`): ns = 16 starts at 5.5e-2 and needs 5445 steps to 1e-3; ns = 49 starts at 1.3e-2 and needs 1654.
- float64 ends at 1.11e-3 after 6000 steps against the float32 floor 8.1e-4 at 5445, same helicity behaviour, 3.1x the cost per step.
- Energy released (0.9 .. 1.2)e-4 of E_0; |dH/H_0| <= 7e-5 for p >= 2 (`figures/energy_helicity.png`).
- Topology (`r12_p3_g0`, zeta = 0.5 final, movie 0 .. 6000 in `r12_p3_g0/movie/`): chains at 3/5, 9/16, 6/11 with iota locked, p_w a flux function, 0 of 160 lost, 4 chaotic. The ns = 49 reference relaxes to a cleaner nested state.

## 5. Seeded islands (2026-09-02, `outputs/li383_pub/`)

New arms, so they follow the 2026-09-02 mesh rule (n, 2n, 2n) (`resolution-ratio-study-2026-09.md`: the toroidal count was the only under-resolved direction) and use the ns = 49 reference (IC residual 0.013 instead of the ns = 16 file's own 0.054). (12,24,24) p = 3 unless marked, seed width 0.1, mu as in section 1 (n_r unchanged). (6, 1) is the iota = 1/2 surface at rho 0.544, (5, 1) the 3/5 surface at rho 0.794. Predicted seed island widths (rho): (6, 1) 0.06 / 0.10 / 0.19 at eps 1e-3 / 3e-3 / 1e-2 (iota' 0.36); (5, 1) 0.13 at 3e-3 (iota' 0.28). `hi_r12x24_p3_g0` is the unseeded control. A first launch of these arms on the ns = 16 file at (12,24,12) was cancelled after 15 min (1.7 GPU-h) when the rule arrived. Budget 10 GPU-h.

The island width is measured from `poincare/sections.npz` (`scripts/li383_pub_figures.py`): "plateau" = extent of the seed radii whose fitted iota sits within 2e-3 of nfp n / m, "excursion" = the largest peak-to-peak logical r over one plane's crossings among those lines (a line inside the island near its separatrix spans the full width). Chaotic lines are excluded from both.

| arm | seed (m, n) | eps | g | ns | steps | stop | s/step | `||F||` 0 -> end | dH/H_0 | width plateau / excursion (rho) | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi_r12x24_p3_g0 | -- | 0 | 0 | 12,24,24 | 1426 | floor | 0.55 | 1.53e-02 -> 7.06e-04 | 1.3e-06 | -- | 0 | 0.22 |
| s61_e1e-3_g0 | (6, 1) | 1e-03 | 0 | 12,24,24 | 1444 | floor | 0.55 | 1.53e-02 -> 6.18e-04 | -6.1e-06 | 0.031 / 0.060 | 2 | 0.22 |
| s61_e3e-3_g0 | (6, 1) | 3e-03 | 0 | 12,24,24 | 1699 | floor | 0.54 | 1.53e-02 -> 6.28e-04 | -9.3e-07 | 0.079 / 0.098 | 2 | 0.25 |
| s61_e1e-2_g0 | (6, 1) | 1e-02 | 0 | 12,24,24 | 1593 | floor | 0.54 | 1.53e-02 -> 6.84e-04 | -4.6e-06 | 0.153 / 0.167 | 6 | 0.24 |
| s61_e3e-3_g1 | (6, 1) | 3e-03 | 1 | 12,24,24 | 991 | floor | 0.67 | 1.53e-02 -> 9.16e-04 | -1.2e-05 | 0.083 / 0.096 | 4 | 0.18 |
| s51_e3e-3_g0 | (5, 1) | 3e-03 | 0 | 12,24,24 | 1516 | floor | 0.55 | 1.53e-02 -> 6.93e-04 | -2.5e-06 | 0.080 / 0.088 | 1 | 0.23 |
| s51_e3e-3_g1 | (5, 1) | 3e-03 | 1 | 12,24,24 | 930 | floor | 0.67 | 1.53e-02 -> 9.12e-04 | -2.8e-05 | 0.081 / 0.084 | 1 | 0.17 |
| r16_s61_e3e-3_g1 | (6, 1) | 3e-03 | 1 | 16,32,32 | 681 | floor | 1.49 | 1.51e-02 -> 8.40e-04 | -1.1e-05 | 0.077 / 0.095 | 2 | 0.28 |

IC widths (plateau / excursion, rho): (6, 1) 0.025 / 0.028 at eps 1e-3, 0.074 / 0.093 at 3e-3, 0.147 / 0.164 at 1e-2; (5, 1) 0.074 / 0.084 at 3e-3. The tracer spacing is 0.006.

- Every seeded island ends within one tracer spacing of its seed width, at both surfaces, at gamma = 0 and 1, and at both meshes: the (6, 1) chain at iota = 1/2 and the (5, 1) chain at 3/5 are tearing-stable at eps <= 1e-2 on li383. Width scales as sqrt(eps) (excursion 0.060 / 0.098 / 0.167 vs the seed formula 0.06 / 0.10 / 0.19), so no eps-independent saturated width appears.
- The seed does not change the descent: seeded and unseeded arms floor in 1.4 .. 1.7 k steps at gamma = 0 and 0.7 .. 1.0 k at gamma = 1, all from 1.53e-2 (`figures/seeded.png`, left).
- Sections: `s61_e1e-2_g0/poincare/poincare_final_zeta0.5.png` is the figure (six O-points at logical r 0.55, nested elsewhere, p_w a flux function); the eps ladder is `figures/seeded.png` (right).

### 5b. Past the 1e-3 floor (`--floor-tol 1e-4`, launched 2026-09-02 17:00)

Same arms as above, run on to the step / wall cap.

| arm | seed (m, n) | eps | g | ns | steps | stop | s/step | `||F||` 0 -> end | dH/H_0 | width plateau / excursion (rho) | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi_r12x24_p3_g0_f4 | -- | 0 | 0 | 12,24,24 | 6000 | steps | 0.48 | 1.53e-02 -> 7.70e-04 | 1.7e-06 | -- | 2 | 0.80 |
| s61_e3e-3_g0_f4 | (6, 1) | 3e-03 | 0 | 12,24,24 | 6000 | steps | 0.48 | 1.53e-02 -> 5.41e-04 | 1.1e-06 | 0.080 / 0.097 | 2 | 0.80 |
| s61_e3e-3_g1_f4 | (6, 1) | 3e-03 | 1 | 12,24,24 | 6000 | steps | 0.57 | 1.53e-02 -> 5.67e-04 | -3.4e-06 | 0.082 / 0.098 | 0 | 0.95 |

- The residual bottoms out at about 6e-4 on the ns = 49 reference at (12,24,24) p = 3 (100-step window means 6.0e-4, 6.7e-4, 5.8e-4): 3.5 .. 6x more steps than the 1e-3 stop buy a factor 1.2 .. 1.6, seeded or not, gamma = 0 or 1.
- The islands do not move: (6, 1) at eps 3e-3 is 0.080 / 0.097 wide after 6000 steps against 0.079 / 0.098 at the 1e-3 stop and 0.074 / 0.093 at the IC. The tearing-stable verdict holds past the floor.

### 5c. gamma = 1 under h-refinement at fixed p = 2 (launched 2026-09-02 21:53)

Does the relaxed residual go down with the mesh? Five rungs on the ns = 49 reference, mesh (n, 2n, 2n), p = 2, gamma = 1 with mu = 0.064 / n^2, `--floor-tol 1e-5` (never reached) and 5000 steps; wall caps 0.5 / 1 / 2 / 5 / 10 h. `min` is the smallest `||F||` on the trace, `window` the 100-step mean at the stop.

| arm | ns | mu | steps | stop | s/step | `||F||` 0 -> end | min | window | dH/H_0 | beta_vol | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h8_p2_g1 | 8,16,16 | 1.0e-03 | 5000 | steps | 0.15 | 2.82e-02 -> 5.99e-04 | 5.38e-04 | 5.52e-04 | -2.9e-05 | 0.0425 | 5 | 0.21 |
| h12_p2_g1 | 12,24,24 | 4.4e-04 | 5000 | steps | 0.30 | 1.55e-02 -> 5.42e-04 | 5.22e-04 | 5.77e-04 | -2.7e-05 | 0.0434 | 1 | 0.42 |
| h16_p2_g1 | 16,32,32 | 2.5e-04 | 5000 | steps | 0.58 | 1.52e-02 -> 5.19e-04 | 4.95e-04 | 5.40e-04 | -1.7e-05 | 0.0431 | 1 | 0.80 |
| h24_p2_g1 | 24,48,48 | 1.1e-04 | 5000 | steps | 1.77 | 1.51e-02 -> 5.31e-04 | 4.66e-04 | 5.26e-04 | -1.1e-05 | 0.0428 | 0 | 2.46 |
| h32_p2_g1 | 32,64,64 | 6.3e-05 | 5000 | steps | 4.61 | 1.50e-02 -> 5.60e-04 | 4.93e-04 | 5.94e-04 | -1.0e-05 | 0.0426 | 0 | 6.40 |

- The bottom does not move with h: after 5000 steps the residual sits at (5.0 .. 5.8)e-4 on every rung from (8,16,16) to (32,64,64) (`figures/hsweep_p2.png`, top right, against n^-1 and n^-2 guides). The IC residual is set by the file, 1.5e-2 from n = 12 on (2.8e-2 at n = 8, where the mesh under-resolves the reference), so the descent takes out a factor 30 that is the same at every resolution.
- The traces are not at a fixed point: over the second half of each run `||F||` still falls as step^(-0.1 .. -0.3) at a line-search step of about 2 that is h-independent under gamma = 1. Whatever remains at 5e-4 is the case's residual under ideal descent (rational surfaces, the file's own imbalance), not a discretisation error; more steps buy a little on every mesh, more cells buy nothing.
- Energy released converges with h: E_0 - E = 1.5e-5, 2.1e-6, 8.9e-7, 6.6e-7, 6.9e-7 for n = 8 .. 32, i.e. the ns = 49 state is within 1.4e-6 of the relaxed energy on the fine meshes (the ns = 16 file released 1e-4, section 4). The absolute helicity change shrinks with h too, -1.5e-7, -1.4e-7, -8.7e-8, -5.4e-8, -5.0e-8 (H_0 = 5.0e-3).
- Rotational transform (`figures/hsweep_p2.png`, bottom left): the final profiles of n = 12 .. 32 lie on the reference (axis 0.395 / 0.394 / 0.392 / 0.392 against 0.396 / 0.395 / 0.394 / 0.394 at the IC, edge 0.659 .. 0.661); n = 8 sits 0.004 high at the axis. The gamma = 1 axis dip is at most 0.002 on li383 at n >= 12 (the W7-X arms of 2026-08-27 dipped 0.005 on every mesh).
- Topology (`h24_p2_g1/poincare/poincare_final_zeta0.5.png`, same at n = 32): nested surfaces throughout, 0 chaotic lines at n >= 24 (5 at n = 8), a smooth iota profile with no resolved plateau at 1/2 or 3/5, p_w a flux function. The ns = 49 reference relaxes without forming the chains the ns = 16 file forms (section 4).
- Cost: 0.15 / 0.30 / 0.58 / 1.77 / 4.61 s/step, i.e. about h^-3 from n = 16 up; the five rungs took 10.3 GPU-h of the 20 budgeted (plus about 0.5 for the sections).

## 6. Figures for the paper

1. Case and IC: three Poincaré planes of the wout IC with the iota / p_w panel (`hi_r12_p3_g0/poincare/poincare_ic_*`).
2. Force residual vs step (`figures/force_convergence.png`): degree scan at (12,24,12) gamma = 0; resolution scan at p = 3 gamma = 0 (r12 from section 4, r16 and r24 from `li383_axisfix`); gamma = 0 vs 1 at r12 and r16. Generator: `scripts/li383_pub_figures.py`.
3. Reference floor (`figures/reference_floor.png`): `r12_p3_g0` vs `hi_r12_p3_g0`, residual vs step on one axis.
4. Energy released and helicity drift vs step, eta = 0 (`figures/energy_helicity.png`).
5. Final topology: zeta = 0.5 section of `r12_p3_g0` (logical chart + iota / p_w panel) and a frame strip from its movie (steps 0, 1000, 2000, 4000, 6000).
6. Precision: one table row (`r12_p3_g0` vs `r12_p3_g0_f64`).
7. Seeded islands: final zeta = 0.5 section of `s61_e1e-2_g0` and the island width vs eps (`figures/seeded.png`, right); the gamma = 1 and (16,32,32) widths as a sentence.
8. gamma = 1 h-sweep at p = 2 (`figures/hsweep_p2.png`): residual vs step for five meshes, floor vs n with the IC residual, final iota profiles, energy released.
9. Optional solver row: smoothing solve on li383 p = 3, MINRES 2134 / 8478 / 20362 at (8,16,8) / (12,24,12) / (16,32,16) vs split + shifted-stiffness atom 145 / 249 (`shifted_split_2026-09-02.md`).

Not shown: `r24_p3_g0` (not floored in the step budget), `r16_p4_g0` (worst floor, 7.3e-3).

## 7. Open

- Floor rule since 2026-09-02: `--floor-tol 1e-4` and let the step / wall cap end the run (section 5b: the 1e-3 stop sat 1.2 .. 1.6x above the bottom). The section 4 arms ran at 1e-3.
- `r24_p3_g1` exists only on the old reader (8.3 GPU-h); rerun if the r24 rung goes in. The reruns keep (n, 2n, n); a paper that adopts (n, 2n, 2n) throughout needs the section 4 arms redone at that mesh (about 2x the cost).
- The 2026-08-28 baseline `outputs/vmec_sections/li383_relaxed` is gone; `r12_p3_g0` of section 4 replaces it.
- Seeded islands: only eps <= 1e-2 at two surfaces; a larger seed (3e-2, width 0.3 in rho) or a surface closer to the axis (9/22) would test whether any li383 surface tears.
