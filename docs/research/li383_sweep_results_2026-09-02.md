# li383 (NCSX) relaxation: sweep results and seeded islands -- 2026-09-02

Status: 2026-09-03: reruns, seeded arms, floor-1e-4 arms, gamma = 1 h-sweep at p = 2, resistivity sweep, B-only helicity pairs, p-sweep at (16,32,32) complete (52 GPU-h); resistive pulses in section 5g; the reconnect series (stall -> checkpoint -> reconnect, the controller as decided) in section 5h, full arm done (three stalls, widths in the table). Seeded arms follow the (n, 2n, 2n) rule of 2026-09-02; the reruns keep the (n, 2n, n) meshes of the 2026-08 sweep they reproduce. `outputs/li383_pub/README.md` explains that folder.
Read it for: every li383 relaxation number that exists, which arms back which figure, the seeded-island result.
Do not read it for: the solver internals (`shifted_split_2026-09-02.md`) or the wout reader (`docs/source/concepts/`).

## 1. Case and settings

- Reference `data/wout_li383_low_res_reference.nc`: NCSX li383, nfp = 3, ns = 16, iota 0.405 (axis) .. 0.656 (edge). Resonances on the way out: iota = 1/2 at rho = 0.55, 6/11 at 0.68, 9/16 at 0.72, 3/5 at 0.80.
- Second reference `data/wout_li383_1.4m.nc`: same case, ns = 49, iota 0.394 .. 0.655 (1/2 at rho 0.54, 3/5 at 0.79).
- IC `--ic clebsch`: `B = dA'` from the wout series in closed form, div-free and wall-tangent exactly. Every wout run of 2026-08-28/29 used the reader BEFORE the per-mode axis conditions (3406d38 ff.); `li383_axisfix` and everything from 2026-09-01 on use the current one.
- Descent: L-BFGS history 1 (identical to the earlier `--method cg`), analytic line search, CFL 0.5, eta = 0, stop when the 100-step mean of `||F||_M` drops below 1e-3 or at 6000 steps, `B` and `p_w` saved every 100 steps.
- gamma = 1: `v = (I - mu L)^{-1} F` with mu = 0.064 / n_r^2 (4.4e-4 on 12 radial cells, 2.5e-4 on 16, 1.1e-4 on 24).
- Seed (`--seed m,n,rho0,width --seed-eps eps`): resonant term `eps |Phi'(rho0)| / m  g(rho) cos 2 pi (m theta - n zeta)` in `A'_zeta`, `g` Gaussian of the given width tapered to zero at the wall; eps = `|dB^rho| / |B^zeta|` at rho0; the chain sits at |iota| = nfp n / m; island full width about `1.6 sqrt(eps nfp / (m |iota'|))` in rho. Under ideal descent the topology is frozen, so a seeded island that grows to an eps-independent width marks a tearing-unstable surface, one that shrinks back a stable one.
- Poincaré: 160 lines, 400 crossings per line, traced in float64, three planes zeta = 0, 0.25, 0.5; iota from the fit, chaotic lines flagged by the h/2 drift. Island width (since 2026-09-03): the largest max(r) - min(r) in logical rho over all crossings on all planes of any non-chaotic line whose fitted iota sits on the rational to within 2e-3 (the earlier tables list this as the second, "excursion" number next to the iota-plateau extent, which is dropped).
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

### 5d. gamma = 1 under p-refinement at fixed (16,32,32) (launched 2026-09-03 09:58)

The p-sweep complementary to 5c, at its (16,32,32) rung (also the resistivity sweep's mesh): p = 1 .. 4, gamma = 1 with mu 2.5e-4, floor 1e-5 (never reached), 5000 steps; `h16_p2_g1` is the 5c arm. Same columns as 5c with the degree in place of the mesh.

| arm | p | mu | steps | stop | s/step | `||F||` 0 -> end | min | window | dH/H_0 | beta_vol | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h16_p1_g1 | 1 | 2.5e-04 | 5000 | steps | 0.17 | 1.30e-01 -> 5.30e-03 | 4.98e-03 | 2.28e-03 | 1.1e-04 | 0.0330 | -- | 0.24 |
| h16_p2_g1 | 2 | 2.5e-04 | 5000 | steps | 0.58 | 1.52e-02 -> 5.19e-04 | 4.95e-04 | 5.40e-04 | -1.7e-05 | 0.0431 | 1 | 0.80 |
| h16_p3_g1 | 3 | 2.5e-04 | 5000 | steps | 1.22 | 1.51e-02 -> 6.49e-04 | 5.95e-04 | 6.49e-04 | -2.8e-06 | 0.0425 | 1 | 1.69 |
| h16_p4_g1 | 4 | 2.5e-04 | 5000 | steps | 2.34 | 1.51e-02 -> 1.93e-03 | 8.36e-04 | 1.04e-03 | -3.1e-06 | 0.0423 | 1 | 3.25 |

- p = 1 does not resolve the file: the IC residual is 1.3e-1 instead of 1.5e-2, ||J||/||B|| 3.7 instead of 0.66, beta 0.033 instead of 0.043, and the descent releases 7.8e-4 of E_0 (a thousand times the p >= 2 value) to end at 5e-3. It is a different problem, not a rung of the same ladder. Its sections cannot be traced: B^zeta reaches zero near the axis (logical r = 0.03) in both the IC and the final field, so the toroidal-angle tracer refuses (an arclength tracer would be needed).
- From p = 2 up the floor RISES with the degree at fixed n: window means 5.4e-4, 6.5e-4, 1.0e-3 at p = 2, 3, 4, with the late slope flattening from step^-0.18 to -0.09. Energy released and helicity change converge (8.9e-7 / 6.9e-7 / 6.6e-7 and -8.7e-8 / -1.4e-8 / -1.5e-8), so the field is the same; what degrades is the descent. The final iota profiles of p = 2, 3, 4 coincide except at the axis, where p = 4 ends 0.006 above the IC (0.400 vs 0.394; p = 2 and 3 within 0.001), 1 chaotic line each (`figures/psweep_p16.png`).
- The cause at p = 4 is the one-pair L-BFGS direction: on 340 of the 5000 steps it is not a descent direction (cos(F, u) < 0) and the exact line search steps backwards (dt < 0; the energy still falls, since the step is the minimiser along the line either way). p = 2 and 3 have one such step each. The mean step at p = 4 is 4.5 against 2 at p = 2, 3: the secant model over-estimates the curvature scale. Per step p = 4 costs 4x p = 2 (2.34 vs 0.58 s) for a worse floor.
- Verdict: at fixed resolution p = 2 is the working degree for the relaxation, p = 3 costs 2x for the same field, p = 4 is not worth running with history 1. Together with 5c: neither h- nor p-refinement lowers the ideal floor, which is the constraint floor that section 5e opens with resistivity.
- Cost: 0.24 + 1.69 + 3.25 = 5.2 GPU-h for the three new rungs.

### 5e. Resistivity sweep (2026-09-03, `outputs/li383_eta/`)

Same mesh, degree and smoothing as the h-sweep's (16,32,32) rung: p = 2, gamma = 1 with mu 2.5e-4, 5000 steps, floor 0 (a first launch at floor 1e-5 stopped the eta >= 1e-5 arms inside the resistive phase; those runs are kept as `*_floor1e-5`). Resistivity is backward Euler in defect form after the ideal step, `--eta-schedule tanh`: eta_max for the first third of the run, dropped to ~0 over the middle third, ideal for the last third. `--eta-every K` batches the solve so each carries eta K dt = 2e-5 (dt about 2 under gamma = 1), the smallest correction float32 resolves; K = 1000 at 1e-8 is why the ladder stops there. Seven rungs, each unseeded and with the (6, 1) seed at eps 3e-3, plus the eta = 0 twins (`li383_pub/h16_p2_g1`, `s61_eta0`). `last 500` is the mean residual over the ideal tail; `dH` the absolute helicity change (H_0 = 5.0e-3); `(6,1) width` the final island width of the seeded arms.

| arm | eta_max | K | seed | steps | stop | s/step | `||F||` 0 -> end | min | last 500 | J/B 0 -> end | beta_vol 0 -> end | dH (abs) | (6,1) width | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eta1e-8 | 1e-08 | 1000 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 4.23e-04 | 3.98e-04 | 4.27e-04 | 0.659 -> 0.615 | 0.0442 -> 0.0402 | -3.7e-05 | -- | 2 | 0.82 |
| eta3e-8 | 3e-08 | 300 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 3.50e-04 | 3.38e-04 | 3.58e-04 | 0.659 -> 0.563 | 0.0442 -> 0.0356 | -1.3e-04 | -- | 5 | 0.82 |
| eta1e-7 | 1e-07 | 100 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 2.71e-04 | 2.67e-04 | 3.00e-04 | 0.659 -> 0.451 | 0.0442 -> 0.0261 | -4.3e-04 | -- | 5 | 0.81 |
| eta3e-7 | 3e-07 | 30 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 1.64e-04 | 1.55e-04 | 1.78e-04 | 0.659 -> 0.311 | 0.0442 -> 0.0150 | -1.1e-03 | -- | 1 | 0.81 |
| eta1e-6 | 1e-06 | 10 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 6.98e-05 | 5.64e-05 | 6.51e-05 | 0.659 -> 0.167 | 0.0442 -> 0.0051 | -2.4e-03 | -- | 0 | 0.82 |
| eta1e-5 | 1e-05 | 1 | -- | 5000 | steps | 0.63 | 1.52e-02 -> 1.35e-05 | 5.11e-06 | 1.27e-05 | 0.659 -> 0.005 | 0.0442 -> 0.0001 | -4.9e-03 | -- | 0 | 0.88 |
| eta1e-4 | 1e-04 | 1 | -- | 5000 | steps | 0.70 | 1.52e-02 -> 3.32e-06 | 3.10e-06 | 3.59e-06 | 0.656 -> 0.000 | 0.0440 -> 0.0000 | -5.0e-03 | -- | 1 | 0.98 |
| s61_eta1e-8 | 1e-08 | 1000 | (6, 1) 3e-03 | 5000 | steps | 0.58 | 1.52e-02 -> 4.29e-04 | 4.07e-04 | 4.24e-04 | 0.659 -> 0.615 | 0.0442 -> 0.0402 | -3.7e-05 | 0.091 | 2 | 0.81 |
| s61_eta3e-8 | 3e-08 | 300 | (6, 1) 3e-03 | 5000 | steps | 0.58 | 1.52e-02 -> 3.58e-04 | 3.43e-04 | 3.58e-04 | 0.659 -> 0.563 | 0.0442 -> 0.0356 | -1.3e-04 | 0.078 | 3 | 0.81 |
| s61_eta1e-7 | 1e-07 | 100 | (6, 1) 3e-03 | 5000 | steps | 0.59 | 1.52e-02 -> 3.06e-04 | 2.63e-04 | 3.02e-04 | 0.659 -> 0.451 | 0.0442 -> 0.0261 | -4.3e-04 | 0.075 | 6 | 0.81 |
| s61_eta3e-7 | 3e-07 | 30 | (6, 1) 3e-03 | 5000 | steps | 0.59 | 1.52e-02 -> 1.66e-04 | 1.53e-04 | 1.82e-04 | 0.659 -> 0.312 | 0.0442 -> 0.0151 | -1.1e-03 | 0.076 | 0 | 0.81 |
| s61_eta1e-6 | 1e-06 | 10 | (6, 1) 3e-03 | 5000 | steps | 0.59 | 1.52e-02 -> 5.98e-05 | 5.65e-05 | 6.18e-05 | 0.659 -> 0.167 | 0.0442 -> 0.0051 | -2.4e-03 | 0.000 | 0 | 0.82 |
| s61_eta1e-5 | 1e-05 | 1 | (6, 1) 3e-03 | 5000 | steps | 0.64 | 1.52e-02 -> 4.27e-06 | 3.35e-06 | 3.83e-06 | 0.659 -> 0.004 | 0.0442 -> 0.0000 | -4.9e-03 | 0.000 | 1 | 0.89 |
| s61_eta1e-4 | 1e-04 | 1 | (6, 1) 3e-03 | 5000 | steps | 0.70 | 1.52e-02 -> 3.92e-06 | 3.07e-06 | 3.59e-06 | 0.656 -> 0.000 | 0.0440 -> 0.0000 | -5.0e-03 | 0.000 | 1 | 0.97 |

| eta_max | dose int eta dt (trace) | sqrt(dose) in cells (h = 1/16) | final iota range | (6,1) width unseeded / seeded | (5,1) width |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.393 .. 0.660 | 0 / 0.093 | 0.027 |
| 1e-8 | 5.5e-5 | 0.12 | 0.397 .. 0.655 | 0 / 0.091 | 0.041 |
| 3e-8 | 1.7e-4 | 0.21 | 0.403 .. 0.646 | 0.008 / 0.078 | 0.081 |
| 1e-7 | 6.0e-4 | 0.39 | 0.421 .. 0.620 | 0.029 / 0.075 | 0.180 |
| 3e-7 | 1.8e-3 | 0.68 | 0.459 .. 0.581 | 0.072 / 0.076 | gone (3/5 outside the profile) |
| 1e-6 | 5.4e-3 | 1.2 | 0.529 .. 0.540 | gone | gone |
| 1e-5 | 4.2e-2 | 3.3 | 0.440 .. 0.489 | vacuum | vacuum |

The dose is what matters (section 5g): the tanh schedule's effective duration is int eta dt / eta_max = 5500 .. 6000 time units (dt about 2 per step over the first 3000 steps), not the 3000 of the first estimate.

- The floor drops with eta from the first rung: 5.2e-4 (ideal) -> 4.3e-4 (1e-8) -> 3.6e-4 -> 3.0e-4 -> 1.8e-4 -> 6.5e-5 -> 1.3e-5 -> 3.6e-6, while the current fraction J/B goes 0.645 -> 0.615 -> 0.563 -> 0.451 -> 0.311 -> 0.167 -> 0.005 -> 0. Nothing plateaus while eta is on: J/B, beta and helicity decay exponentially in time at every rung until the schedule cuts eta (`figures/eta_traces.png`). There is no eta at which a resistive equilibrium with the current intact is reached; the lower floors are the residuals of lower-current fields.
- What the small rungs do is open the rational surfaces the ideal descent held closed (`figures/eta_islands.png`). The (5, 1) chain at 3/5 grows from nothing to 0.04 / 0.08 / 0.18 in rho at 1e-8 / 3e-8 / 1e-7 (five O-points at logical r 0.8, iota locked on 0.6, p_w flat across them: `eta1e-7/poincare/poincare_final_zeta0.5.png`) and is gone at 3e-7 because the flattened profile no longer reaches 3/5. The (6, 1) chain at 1/2 opens later, 0.008 / 0.029 / 0.072 at 3e-8 / 1e-7 / 3e-7. The current sheets of the ideal descent reconnect at a diffusion length of an eighth of a cell (1e-8): the h-independent floor of section 5c is an ideal-constraint floor.
- Seeded and unseeded arms agree to 2% in residual, J/B, beta and helicity at every rung. The seeded (6, 1) island shrinks from 0.093 to 0.075 by 1e-7 while the unseeded one grows to 0.072 at 3e-7: both settle on the same resistive width, about 0.075 in rho, independent of the seed. That is the width the surface wants once it may reconnect; the ideal runs of section 5 froze whichever width they were given.
- From 1e-6 up the equilibrium is gone: iota flattens to 0.53 .. 0.54 at 1e-6 and to the wall's vacuum profile 0.44 .. 0.49 at 1e-5, beta and helicity to zero. The bulk-mode estimate of the useful range (10 eta T ~ 1 at 1e-5) was off by two decades: the current lives at wavenumbers around 20, not at the bulk scale.
- Cost: 14 arms x 0.8 GPU-h plus 1.7 for the floored first attempts and 0.1 per sections job; 14 GPU-h in all.

### 5f. Helicity without the auxiliary H: the B-only step (2026-09-03, `outputs/li383_bonly/`)

The production step projects B onto the 1-forms, H = M_1^-1 P B, and forms J x H and u x H; helicity is conserved because its discrete rate is the integral of (u x H) . H, zero pointwise. `mrx/experimental/bonly_relaxation.py` (behind `scripts/relax.py --stepper bonly`, a hook to be removed) forms J x B and u x B from the 2-form directly. The energy identity survives ((u x B) . J = -u . (J x B) pointwise with the same J); the helicity rate becomes the integral of (u x B) . (H_h - B_h), a product of two projection errors. Twins of the production arms, helicity every 50 steps:

| arm pair | mesh, p, precision | steps | s/step | energy identity max | H-form dH | B-only dH | ratio |
|---|---|---|---|---|---|---|---|
| h16_p2_g1 / bonly_h16_p2_g1 | (16,32,32) p2 f32 | 5000 | 0.58 / 0.58 | 1.3e-7 / 1.8e-7 | -8.7e-8 | -7.6e-8 | 0.9 |
| h12_p2_f64_h / _bonly | (12,24,24) p2 f64 | 5000 | 0.95 / 0.93 | 3.5e-14 / 3.7e-14 | -3.1e-8 | -3.4e-8 | 1.1 |
| h8_p1_h / _bonly | (8,16,16) p1 f32 | 5000 | 0.08 / 0.08 | 3.9e-4 / 3.5e-4 | +2.6e-5 | +6.2e-5 | 2.4 |

- Dropping H does break helicity conservation, but at p >= 2 the effect is under the time-stepping error the production scheme already has: in float64, with the energy identity at 3e-14, the H-form still drifts 3e-8 over 5000 steps. E perp H makes the RATE vanish (the semi-discrete statement); the explicit step B + dt curl E changes the helicity by dt^2 (E, curl E) per step, and the line-search step of about 2 makes that the dominant term.
- The B-only term is a tenth of it at p = 2 and the larger term only at p = 1 on the coarse mesh (2.4x). H_0 = 5.0e-3 in every arm.
- A clean demonstration of "H conserves, B does not" needs either p = 1 or a helicity-conserving time integrator (midpoint in A) as the baseline. Not done. The dt^2 attribution is argued, not measured; a float64 H-form arm with the CFL cap forcing dt down by 4 would confirm it (about 1.3 GPU-h).
- Cost: 0.8 + 1.3 + 1.3 + 0.1 + 0.1 GPU-h.
- Follow-up 2026-09-04 (branch implicit-midpoint, `docs/research/implicit_midpoint_2026-09-04.md` section 4.1, `outputs/midpoint_sweep/{ex,mp}_small_f64_bonly`): with the midpoint-implicit induction step on (8,16,16) p = 2 float64 the first-step time error disappears in B-only as well (-2.6e-9 against -2.5e-7 explicit), and what remains, shared by both B-only arms, is the projection error of the pairing int (u x B_mid) . (H_dir - B_mid): a state-dependent excursion to -1.3e-6 that returns to about zero by step 1000, not a monotone leak; Dirichlet-H midpoint holds 5e-12 throughout. The reading above (B-only at or below explicit at p >= 2) was the time error masking the projection error.

### 5g. Resistive pulses after an ideal phase, and the pulse controller (2026-09-03, `outputs/li383_pulse/`)

Question: does a resistive dose do more when it is applied after the ideal descent has built its current sheets than when it is spread over the descent (5e)? `--eta-schedule pulse --eta-pulse S,W[,P]` (driver, 164b9c4) holds eta_max on a window of W steps from step S (repeating every P) and zero elsewhere, with the stepper's resistive clock reset while it is off, so `--eta-every W` makes the pulse ONE backward-Euler solve of eps = eta x window time. Arms at the (16,32,32) p = 2 gamma = 1 rung: 2000 ideal steps, a 100-step pulse, ideal to 5000; eta = 1.5e-7 / 5e-7 / 1.5e-6 (doses 3.4e-5 / 1.1e-4 / 3.1e-4, one solve each), and 5e-7 repeated every 1000 steps (three solves, 3.2e-4 in all).

| arm | eta_max | K | seed | steps | stop | s/step | `||F||` 0 -> end | min | last 500 | J/B 0 -> end | beta_vol 0 -> end | dH (abs) | (6,1) width | chaotic | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pulse1.5e-6 | 2e-06 | 100 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 3.67e-04 | 3.45e-04 | 3.67e-04 | 0.659 -> 0.538 | 0.0442 -> 0.0339 | -2.1e-04 | -- | 9 | 0.82 |
| pulse1.5e-7 | 1e-07 | 100 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 4.50e-04 | 4.16e-04 | 4.48e-04 | 0.659 -> 0.622 | 0.0442 -> 0.0410 | -2.7e-05 | -- | 2 | 0.81 |
| pulse5e-7 | 5e-07 | 100 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 4.16e-04 | 3.85e-04 | 4.14e-04 | 0.659 -> 0.591 | 0.0442 -> 0.0383 | -8.3e-05 | -- | 9 | 0.82 |
| pulse5e-7_cyc | 5e-07 | 100 | -- | 5000 | steps | 0.59 | 1.52e-02 -> 3.67e-04 | 3.57e-04 | 4.22e-04 | 0.659 -> 0.519 | 0.0442 -> 0.0320 | -2.4e-04 | -- | 8 | 0.81 |

The comparison at equal dose (the trace's int eta dt; the tanh rungs' is 5.5e-5 / 1.7e-4 / 6.0e-4 at eta_max 1e-8 / 3e-8 / 1e-7):

| dose | schedule | last-500 residual | J/B end | dH | (5,1) width | (6,1) width |
|---|---|---|---|---|---|---|
| 3.4e-5 | pulse 1.5e-7 | 4.5e-4 | 0.622 | -2.7e-5 | 0.041 | 0 |
| 5.5e-5 | tanh 1e-8 | 4.3e-4 | 0.615 | -3.7e-5 | 0.042 | 0 |
| 1.1e-4 | pulse 5e-7 | 4.1e-4 | 0.591 | -8.3e-5 | 0.066 | 0 |
| 1.7e-4 | tanh 3e-8 | 3.6e-4 | 0.563 | -1.3e-4 | 0.081 | 0.008 |
| 3.1e-4 | pulse 1.5e-6 | 3.7e-4 | 0.538 | -2.1e-4 | 0.102 | 0.032 |
| 3.2e-4 | 3 x pulse 5e-7 | 4.2e-4 (3.7e-4 min) | 0.519 | -2.4e-4 | 0.117 | 0 |
| 6.0e-4 | tanh 1e-7 | 3.0e-4 | 0.451 | -4.3e-4 | 0.180 | 0.029 |

- The dose is the controlling variable and the timing is nearly irrelevant: on the dose axis (`li383_pulse/figures/pulse_islands.png`) the pulses sit on the tanh curve for residual, J/B, helicity (0.8 of helicity lost per unit dose either way) and the 3/5 island width at the two smaller doses, and slightly below it at the largest. Three pulses of 1.1e-4 do what one of 3.1e-4 does. There is no selectivity to gain from timing; a single large solve also leaves more chaotic lines (9 vs 5).
- Mechanism (`pulse_traces.png`): a pulse drops J/B, beta and helicity in one step and kicks the residual up by 3x, since the diffused field is out of balance around the former sheets; the descent needs about a thousand steps to work that off and rebuilds sheets elsewhere, whereas a small continuous eta reconnects the sheets as they form. Only the operator would buy selectivity (hyper-resistivity, k^4 against k^2).
- Machinery that came out of it and stays: the pulse schedule; `--checkpoint / --restart` (the full descent state as an equinox pytree with the step number, `equinox.tree_serialise_leaves`; verified by a 120 + 80 step chain against a 200-step run: identical schedule and step accounting, trajectories differing only by the run-to-run round-off of the GPU descent, which already separates two identical 120-step runs by 18%). An accept/reject pulse controller (fire at a stall, judge at the next, revert when it did not help) was smoke-tested and dropped the same day: the dose finding says the decision it automates is a budget, not a timing (0.7% of H_0 buys 17% of the floor, the next 2% another 16%, the next 6% another 17%). Section 5h has what replaced it.
- Cost: 4 x 0.8 GPU-h plus 0.3 for the smokes.

### 5h. The reconnect series: checkpoint -> reconnect every K steps (2026-09-03, `outputs/li383_pulse/reconnect_*`)

Late 2026-09-03 the stall test was retired for a plain interval, `--reconnect-every K` (see the end of this section: the ideal descent is a power law, nothing stalls). The arms below ran with the stall test; on disk they have since been renamed to the interval layout (`reconnect/<k>/` instead of `stalls/<k>/`, `reconnect<k>` tags in the sections, `results["reconnect"]` with `k` and `resid`), the wording of the runs is kept as it was.

Decided 2026-09-03 (Tobias): no accept/reject. The descent runs until it stalls, the stalled equilibrium is checkpointed, one resistive solve reconnects it, the descent continues; the outcome is a series of stalled ideal equilibria at decreasing helicity, and the user picks. `--reconnect` in the driver (b449e97): stalled = the block mean of the residual (blocks of a fifth of `--stall-steps`, default 1000) dropped by less than `--stall-tol` (5%) over `--stall-steps`, on the history since the last reconnection; stall k writes `<out>/stalls/<k>/B.h5` (field and pressures in the layout of `B.h5`, so `poincare_relax.py` reads it) and `state.eqx` (a `--restart` file, so any stall can be continued with another dose); the dose is `eps = c h^2`, `c = --reconnect-eps` (0.01: a diffusion length of a tenth of a cell, the scale the h-independent floor of 5c is made of), one backward-Euler solve, then `initial_state` on the diffused field (fresh L-BFGS pair). `results["stalls"]` has step, floor, eps, |F|, helicity, ||J||/||B|| and the pressures before and after each solve. The helicity price of a solve is exact, dH = -2 eps int J.B, so it could also be set from a helicity fraction per stall; the grid-scale rule was kept because it scales with h.

Inner/outer loop: the ideal descent is the inner loop (helicity conserved, terminates on the stall test), the reconnection is the outer loop (changes the topology, restarts the inner loop from a checkpoint). Host-side Python for the outer loop is the right split: the stall test is a scalar over a 1000-step history, the checkpoint is I/O, the restart changes the optimiser state; none of it belongs inside the jit. What the driver does badly is dispatching one jitted step per Python iteration with a host sync per step (needed for its per-step trace); `run_relaxation` in the library already scans N steps per dispatch with a callback, and the driver should become that callback once the arms are done.

Smoke `reconnect_smoke_prechunk` (the per-step driver, stall window 300 steps), (8,16,16) p = 2 gamma = 0, 3000 steps, eps = 1.6e-4 (c = 0.01 at h = 1/8), 0.14 s/step:

| stall | step | floor | |F| before -> after | H before -> after (dH / H_0) | J/B before -> after | beta_vol before -> after |
|---|---|---|---|---|---|---|
| 1 | 1040 | 5.8e-4 | 8.0e-4 -> 2.0e-3 | 5.007e-3 -> 4.888e-3 (-2.4%) | 0.643 -> 0.554 | 0.0415 -> 0.0353 |
| 2 | 1924 | 3.3e-4 | 3.4e-4 -> 1.7e-3 | 4.888e-3 -> 4.776e-3 (-2.3%) | 0.580 -> 0.511 | 0.0368 -> 0.0317 |
| 3 | 2739 | 2.9e-4 | 2.9e-4 -> 1.5e-3 | 4.776e-3 -> 4.671e-3 (-2.2%) | 0.530 -> 0.475 | 0.0329 -> 0.0287 |

- Each reconnection lowers the next floor (5.8e-4 -> 3.3e-4 -> 2.9e-4) with diminishing returns (x1.7, then x1.2) at a flat helicity price (2.2 .. 2.4% of H_0 per solve at this coarse mesh, where eps = c h^2 is 4x the n = 16 value); between stalls the descent recovers ||J||/||B|| and beta partly (0.554 -> 0.580, 0.0353 -> 0.0368) and then stalls lower. The reconnected equilibria carry less current and less pressure, which is the physics of the dose (5e), now as a discrete series. A solve takes 44 MINRES iterations and moves the field by 0.2%.
Full arm `reconnect_h16_p2_g1`, (16,32,32) p = 2 gamma = 1, 8000 steps (4544 s, 0.57 s/step), the per-step driver's detector (5% over 1000 steps), eps = 3.9e-5 (c = 0.01 at h = 1/16: the pulse1.5e-7 dose, per stall). Widths from the sections of the stalled fields (`poincare/poincare_stall<k>_*`: `poincare_relax.py --fields ic,final,stalls` traces them in one call with the IC and the final field, so all share one iota and one p colour scale; `li383_pub.sh sections NAME`):

| stall | step | floor | |F| before -> after | H before -> after (dH / H_0) | J/B before -> after | beta_vol before -> after | (5,1) width | (6,1) width | chaotic |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3948 | 5.55e-4 | 5.4e-4 -> 8.1e-4 | 5.0111e-3 -> 4.9794e-3 (-0.63%) | 0.646 -> 0.609 | 0.0432 -> 0.0402 | 0.021 | 0 | 4 |
| 2 | 5691 | 4.61e-4 | 4.4e-4 -> 7.5e-4 | 4.9793e-3 -> 4.9482e-3 (-0.62%) | 0.618 -> 0.590 | 0.0407 -> 0.0384 | 0.044 | 0 | 1 |
| 3 | 7423 | 4.24e-4 | 4.1e-4 -> 6.9e-4 | 4.9482e-3 -> 4.9177e-3 (-0.62%) | 0.599 -> 0.574 | 0.0389 -> 0.0369 | 0.059 | 0.008 | 5 |

- The first stall IS the ideal floor of the rung (5.55e-4 against the 5.4e-4 window mean of `h16_p2_g1`), reached at step 3948; the series then reads 5.55e-4 -> 4.61e-4 -> 4.24e-4 (-17%, -8%) at a flat price of 0.62% of H_0 per solve. The exchange rate is the dose sweep's: 0.6% of H_0 for a 17% floor drop was the tanh 1e-8 rung's (0.7% for 17%). The residual kick is 1.5x here (2.5x at n = 8: the dose per cell is 4x smaller), the field moves by 0.08% per solve, MINRES takes 64 iterations.
- Island widths follow the cumulative dose: 3/5 at 0.021 (the ideal state; `h16_p2_g1` has 0.027 at 5000), 0.044 after one solve (dose 3.9e-5; pulse1.5e-7 at 3.4e-5 gave 0.041), 0.059 after two (7.8e-5; between the pulse arms at 3.4e-5 and 1.1e-4), 0.067 at step 8000 after three (1.2e-4; pulse5e-7 at 1.1e-4 gave 0.066), the 1/2 chain still closed. Each stalled equilibrium is an ideal fixed point at its own helicity, which the sections show as nested surfaces with one growing chain: the paper's "pick your equilibrium" series.
- Between stalls the descent recovers a third of the current the solve removed (0.609 -> 0.618, 0.590 -> 0.599) and about a third of beta, then stalls lower. Cost 1.3 GPU-h.
- The driver has since moved to the chunked loop (`docs/research/handoff_2026-09-03_chunked_relaxation_loop.md`, b54a0ff): `--chunk 500` is the single cadence, verified against this arm's driver on the same node (0.56 s/step both, chunk means within 1%); the arm itself ran on the per-step driver.

**Nothing stalls: the ideal descent is a power law (2026-09-03, late).** Tobias observed that the log-log slope of the residual looks constant; checked on every ideal gamma = 1 li383 arm (`li383_pub`, `li383_sweep`, the ideal part of this arm; 500-step block means, local exponent between consecutive blocks, fit over steps >= 1000; `outputs/li383_pulse/figures/ideal_slopes_g1.png`, job-scratch `slopes.py`):

| arm | mesh, p | a (fit) | first half | second half |
|---|---|---|---|---|
| h8_p2_g1 | (8,16,16) 2 | 0.34 | 0.37 | 0.29 |
| h12_p2_g1 | (12,24,24) 2 | 0.30 | 0.33 | 0.25 |
| h16_p2_g1 | (16,32,32) 2 | 0.20 | 0.22 | 0.15 |
| reconnect_h16_p2_g1, steps < 3948 | (16,32,32) 2 | 0.20 | | 0.21 |
| h24_p2_g1 | (24,48,48) 2 | 0.22 | 0.23 | 0.23 |
| h32_p2_g1 | (32,64,64) 2 | 0.23 | 0.31 | 0.10 |
| h16_p1_g1 | (16,32,32) 1 | 0.66 | 0.78 | 0.46 |
| h16_p3_g1 | (16,32,32) 3 | 0.15 | 0.20 | 0.07 |
| r12 / r16 / r24_p3_g1 | (n,2n,n) 3 | 0.50 / 0.38 / 0.27 | | constant |

- `resid ~ t^-a` with a constant local exponent (block-to-block scatter about 0.05) to the end of every run; no plateau. The four blocks before this arm's first "stall" read 0.21, 0.19, 0.21, 0.20: the stall test fired because the drop per 1000 steps of a t^-0.2 law crosses 5% at step 4000 (t = a N / tol), not because anything flattened. A few (n,2n,2n) arms drift to 0.1 in their last 1000 steps, within two block noises. gamma = 0 arms are not power laws at this block length (local exponent between -1 and 2 from block to block).
- Consequence: a stall test at any tolerance is a step count in disguise, scaling the tolerance with the chunk only fixes that count, and an exponent test would never fire. The interval is now chosen outright: `--reconnect-every K`, rounded to whole chunks, no solve on the last chunk; the launcher's full arm is `--reconnect-every 2000` on 8000 steps (to be rerun for the paper), the smoke `--reconnect-every 600` at `--chunk 100` (job 17458707: reconnections at 600 / 1200 / 1800 / 2400, none on the last chunk, 2.4 / 2.3 / 2.2 / 2.1% of H_0 per solve, the same prices as the per-step smoke at its detector's steps). The series should be described as ideal equilibria sampled along a power-law descent at a chosen interval, not as stalled equilibria.
- The reconnections still buy time: the first interval's t^-0.2 law would need about 16000 steps to reach the residual this arm had at step 7000 after two solves, at 1.2% of H_0.

**The ladder, `reconnect_ladder_h16_p2_g1` (2026-09-04, job 17461050, `li383_pub.sh reconnect ladder`).** The same rung, `--reconnect-every 2000 --reconnect-eps 0.02 --steps 18000`: eight solves of eps = 7.8e-5 (1.2% of H_0 each) from step 0 at a uniform interval, nine ideal equilibria (eight checkpoints under `reconnect/<k>/` plus the final field), 18000 steps in 10063 s (0.56 s/step). Sections of all ten fields in one call (`poincare/poincare_{ic,reconnect<k>,final}_zeta*.png`, one colour scale); composite `li383_pulse/figures/ladder_zeta0.png`; the run's own trace figure in its directory, `reconnect_ladder_h16_p2_g1/figures/ladder_traces.png` with `figures/pgf/ladder_traces.pgf` (`li383_pub_figures.py ladder_figure`, house style, black / purple / teal: one panel, the residual on the left axis, beta and the relative helicity change in per cent on the right axis; block means with sd, reconnections dotted); `li383_pulse/figures/reconnect_traces.png` has it next to the three-rung arm and the ideal run. Since 2026-09-04 every `.pgf` and its `-img*.png` companions live in a `pgf/` subfolder of the figure or section directory.

| k | step | resid (chunk mean before) | H before -> after (dH / H_0) | J/B before -> after | beta_vol before -> after | (5,1) width | (6,1) width | chaotic |
|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | 6.55e-4 | 5.0112e-3 -> 4.9492e-3 (-1.24%) | 0.648 -> 0.586 | 0.0433 -> 0.0383 | 0.020 | 0 | 2 |
| 2 | 4000 | 4.53e-4 | 4.9493e-3 -> 4.8896e-3 (-1.21%) | 0.601 -> 0.556 | 0.0392 -> 0.0355 | 0.057 | 0.007 | 5 |
| 3 | 6000 | 4.04e-4 | 4.8895e-3 -> 4.8320e-3 (-1.18%) | 0.570 -> 0.531 | 0.0363 -> 0.0333 | 0.077 | 0.014 | 10 |
| 4 | 8000 | 3.83e-4 | 4.8319e-3 -> 4.7762e-3 (-1.15%) | 0.543 -> 0.510 | 0.0340 -> 0.0314 | 0.099 | 0.021 | 9 |
| 5 | 10000 | 3.50e-4 | 4.7761e-3 -> 4.7221e-3 (-1.13%) | 0.519 -> 0.490 | 0.0320 -> 0.0297 | 0.117 | 0.028 | 4 |
| 6 | 12000 | 3.51e-4 | 4.7220e-3 -> 4.6695e-3 (-1.11%) | 0.499 -> 0.473 | 0.0302 -> 0.0282 | 0.134 | 0.014 | 4 |
| 7 | 14000 | 3.35e-4 | 4.6694e-3 -> 4.6183e-3 (-1.09%) | 0.481 -> 0.457 | 0.0287 -> 0.0268 | 0.148 | 0.012 | 4 |
| 8 | 16000 | 3.21e-4 | 4.6182e-3 -> 4.5684e-3 (-1.08%) | 0.464 -> 0.442 | 0.0272 -> 0.0256 | 0.164 | 0 | 8 |
| final | 18000 | 3.07e-4 | 4.5684e-3 (-8.84% in all) | 0.449 | 0.0260 | 0.180 | 0.023 | 5 |

- The 3/5 chain grows monotonically with the cumulative dose, 0.020 -> 0.164 over the eight rungs and 0.180 at the end (dose 6.25e-4, the tanh 1e-7 arm's); the 1/2 chain opens to 0.028 at rung 5 and the width measure loses it afterwards (the axis iota falls from 0.60 to 0.54 as current leaves, moving the resonance outward into the 3/5 chain's neighbourhood; the sections show small 1/2 ovals at r = 0.5 from rung 4 on). Rung 2 sits at the dose of the three-rung arm's rung 3 (7.8e-5) and gives the same width, 0.057 against 0.059.
- The residual level before each solve falls with the dose and flattens: 6.55, 4.53, 4.04, 3.83, 3.50, 3.51, 3.35, 3.21 (x1e-4), final 3.07e-4; the kick per solve is 2.1x at this c (1.5x at c = 0.01) and is gone within the first 500-step chunk. The helicity price per rung is exact and shrinks slightly with the current, 1.24% -> 1.08%; J/B goes 0.648 -> 0.449, beta_vol 0.043 -> 0.026, and between solves the descent recovers about a quarter of the current the solve removed.
- Cost 2.8 GPU-h for the run, 12 min for the ten sections.

## 6. Figures for the paper

1. Case and IC: three Poincaré planes of the wout IC with the iota / p_w panel (`hi_r12_p3_g0/poincare/poincare_ic_*`).
2. Force residual vs step (`figures/force_convergence.png`): degree scan at (12,24,12) gamma = 0; resolution scan at p = 3 gamma = 0 (r12 from section 4, r16 and r24 from `li383_axisfix`); gamma = 0 vs 1 at r12 and r16. Generator: `scripts/li383_pub_figures.py`.
3. Reference floor (`figures/reference_floor.png`): `r12_p3_g0` vs `hi_r12_p3_g0`, residual vs step on one axis.
4. Energy released and helicity drift vs step, eta = 0 (`figures/energy_helicity.png`).
5. Final topology: zeta = 0.5 section of `r12_p3_g0` (logical chart + iota / p_w panel) and a frame strip from its movie (steps 0, 1000, 2000, 4000, 6000).
6. Precision: one table row (`r12_p3_g0` vs `r12_p3_g0_f64`).
7. Seeded islands: final zeta = 0.5 section of `s61_e1e-2_g0` and the island width vs eps (`figures/seeded.png`, right); the gamma = 1 and (16,32,32) widths as a sentence.
8. gamma = 1 h-sweep at p = 2 (`figures/hsweep_p2.png`): residual vs step for five meshes, floor vs n with the IC residual, final iota profiles, energy released.
9. p-sweep at (16,32,32) (`figures/psweep_p16.png`): the h-sweep's twin with the degree on the axis; p = 1 without an iota panel.
10. Resistivity (`li383_eta/figures/eta_islands.png` and `eta_traces.png`): island width at 1/2 and 3/5 vs eta_max, seeded and unseeded; traces of residual, J/B, beta, helicity, energy vs step per rung; the zeta = 0.5 section of `eta1e-7`.
11. Helicity without H: one table row per pair (section 5f).
11b. Pulses (`li383_pulse/figures/pulse_islands.png`): island width vs dose, pulse vs tanh; one sentence that the dose is what matters.
11c. Reconnect series (`li383_pulse/figures/reconnect_traces.png` and the stall table): the residual with the stalls marked, one section per stalled equilibrium; the paper's 'pick your equilibrium' figure.
12. Optional solver row: smoothing solve on li383 p = 3, MINRES 2134 / 8478 / 20362 at (8,16,8) / (12,24,12) / (16,32,16) vs split + shifted-stiffness atom 145 / 249 (`shifted_split_2026-09-02.md`).

Not shown: `r24_p3_g0` (not floored in the step budget), `r16_p4_g0` (worst floor, 7.3e-3).

## 7. Open

- Floor rule since 2026-09-02: `--floor-tol 1e-4` and let the step / wall cap end the run (section 5b: the 1e-3 stop sat 1.2 .. 1.6x above the bottom). The section 4 arms ran at 1e-3.
- `r24_p3_g1` exists only on the old reader (8.3 GPU-h); rerun if the r24 rung goes in. The reruns keep (n, 2n, n); a paper that adopts (n, 2n, 2n) throughout needs the section 4 arms redone at that mesh (about 2x the cost).
- The 2026-08-28 baseline `outputs/vmec_sections/li383_relaxed` is gone; `r12_p3_g0` of section 4 replaces it.
- Seeded islands: only eps <= 1e-2 at two surfaces; a larger seed (3e-2, width 0.3 in rho) or a surface closer to the axis (9/22) would test whether any li383 surface tears.
