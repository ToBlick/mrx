# li383 (NCSX) relaxation sweep -- 2026-08-28

Case: `data/wout_li383_low_res_reference.nc` (NCSX li383, nfp = 3, VMEC iota
0.40 .. 0.66), Clebsch IC in closed form from the wout file, cg descent with
the line-search step, CFL cap 0.5, eta = 0, `--floor-tol 1e-3`,
`--save-every 100` (B and p_w every 100 steps into B.h5). Launcher:
`scripts/li383_sweep.sh`; job ids in `outputs/li383_sweep/jobs.tsv`.

Baseline (`outputs/vmec_sections/li383_relaxed`, main-next): (12,24,12) p=3
float32, 3000 steps in 1042 s (0.35 s/step), |F| 5.5e-2 -> 2.0e-3, NOT
floored (window mean 2.5e-3). Its zeta = 0.5 final section: nested core,
flat iota at 3/5 (edge, 5-lobe chain), 1/2 (mid radius chain), 6/11 and 9/16
small plateaus, a ring of tiny islands next to the axis at 9/22.

Budget: 50 GPU-h in total, sections included.

## Planned matrix (wave 1, launched 13:08)

Cost scaling used for the estimates: baseline 0.35 s/step at (12,24,12) p=3;
h-refinement ~2.8x per ladder step (W7-X 2026-08-25 table: 0.87 -> 2.71 ->
7.66 s/step for 8^3 -> 12^3 -> 16^3); p: 0.26x (p=1), 0.53x (p=2), 1.7x
(p=4); gamma=1 ~2.3x per step (W7-X (12,24,12): 1.7 vs 0.74 s/step); float64
~2x. mu = 0.064 h^2 with h = 1/n_r, the rule of the 2026-08-26 W7-X study
(4.4e-4 on (12,24,12), 2.5e-4 on (16,32,16)). gamma=1 reached its floor
within 2000 steps on W7-X (12,24,12) (dt is 8-16x larger than gamma=0), so it
gets the same 6000-step cap, not more.

| arm | ns | p | gamma | mu | precision | steps cap | est s/step | est GPU-h | TIMEOUT |
|---|---|---|---|---|---|---|---|---|---|
| r12_p3_g0 | 12,24,12 | 3 | 0 | - | f32 | 6000 | 0.35 | 0.6 | 3 h |
| r16_p3_g0 | 16,32,16 | 3 | 0 | - | f32 | 6000 (14000 s) | 0.9 | 1.5 | 6 h |
| r24_p3_g0 | 24,48,24 | 3 | 0 | - | f32 | 6000 (30000 s) | 3.5 | 5.8 | 20 h |
| r12_p1_g0 | 12,24,12 | 1 | 0 | - | f32 | 6000 | 0.1 | 0.2 | 2 h |
| r12_p2_g0 | 12,24,12 | 2 | 0 | - | f32 | 6000 | 0.2 | 0.35 | 2.5 h |
| r12_p4_g0 | 12,24,12 | 4 | 0 | - | f32 | 6000 (12000 s) | 0.65 | 1.1 | 5 h |
| r12_p3_g1 | 12,24,12 | 3 | 1 | 4.4e-4 | f32 | 6000 (14000 s) | 0.85 | 1.4 | 6 h |
| r16_p3_g1 | 16,32,16 | 3 | 1 | 2.5e-4 | f32 | 6000 (25000 s) | 2.2 | 3.7 | 12 h |
| r12_p3_g0_f64 | 12,24,12 | 3 | 0 | - | f64 | 6000 (12000 s) | 0.7 | 1.2 | 5 h |

Wave-1 relaxations: ~16 GPU-h estimated. Sections (ic, final at zeta =
0, 0.25, 0.5 for every arm, ~0.1-0.3 GPU-h each) plus three snapshot movies
(~0.5 GPU-h each): ~3 GPU-h. Wave 2 (r16 p=4, r24 p=2, longer gamma=1 or
r24 gamma=1) is decided from the wave-1 costs within the remaining ~30 GPU-h.
