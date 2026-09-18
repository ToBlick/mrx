# The paper's descent block rerun at m = 0 (2026-09-17)

**Status:** done 2026-09-17; the paper's descent block is gradient descent, the L-BFGS memory is gone from the code.
**Read for:** the m = 0 numbers behind the paper's descent figures and tables, and what the red TeX blocks refer to.
**Do not read for:** the L-BFGS results (historical: `lbfgs_batch` notes of 2026-09-13).

Tobias: the L-BFGS arms with memory are too finicky for the paper (Powell restart and all);
the descent block is steepest descent on the smoothed force, `--history 0`. Every arm of
the September batch (`lbfgs_2026-09-13/batch.sh` and the helicity block of
`newton_sweeps_2026-09-13/batch.sh`) rerun with m = 0, nothing else changed: li383 (16,32,32)
p = 2 mixed, `--method lbfgs --floor-tol 0`, potential route, smooth first. Records in
`outputs/lbfgs_m0_2026-09-17/` of the `newton-atom-smoothing` worktree (code b8b218e on
64ffbd9; the L-BFGS path is the released one). Twelve jobs 18649473-84 and 18649507, about
5.5 GPU-hours. The gamma = 0 arm runs 7000 steps instead of 20000 (Tobias: comparable wall
time to gamma = 1's 5000). The 40000-step Leray ladder behind figure 1 was NOT rerun (Tobias:
figure 1 can use whatever).

| arm | steps | s/step | floor (min F2) | to 1e-6 / 1e-7 / 1e-8 | last 1000 | dH/H x1e6 |
|---|---|---|---|---|---|---|
| anchor, c 0.02 | 5000 | 0.30 | 9.99e-8 | 449 / 4995 / -- | 1.14e-7 | -6.9 |
| c 0.004 | 5000 | 0.33 | 8.47e-8 | 703 / 4371 / -- | 5.77e-7 | -8.0 |
| c 0.1 | 5000 | 0.35 | 1.36e-7 | 545 / -- / -- | 1.51e-7 | -7.7 |
| c 0.5 | 5000 | 0.56 | 1.75e-7 | 667 / -- / -- | 1.93e-7 | -8.2 |
| gamma 0 | 7000 | 0.26 | 1.78e-7 | 1582 / -- / -- | 6.52e-7 (rising) | -- |
| Leray | 5000 | 0.42 | 9.99e-8 | 449 / 4995 / -- | 1.14e-7 | -6.8 |
| September anchor (m = 1) | 5000 | 0.32 | 1.88e-9 | 320 / 1190 / 1481 | 3.8e-8 | -10.6 |

Helicity (1000 steps, gamma 0): mixed plain 5.11e-6, corrected 5.58e-7; float64 plain
5.67e-6, corrected 3.46e-16 (September m = 1: 1.00e-5 / 1.86e-7, 1.03e-5 / 1.73e-15).

What changes in the paper's story:
- Steepest descent reaches 1e-7 at step 4995 and never 1e-8 in 5000 steps; the floor is 1e-7
  instead of 2e-9. The Newton comparison (Fig. newton_vs_lbfgs, Tab. newton_convergence)
  is now against that.
- The c sweep has no restart column; 0.02 is still the choice (1e-7 at 4995), 0.004 floors
  lowest (8.5e-8) but is slower to 1e-6; 0.1 and 0.5 never reach 1e-7. The "too little
  smoothing breaks L-BFGS" sentence no longer applies.
- gamma = 0 at 7000 steps does not break down; it is a wide oscillating band with a
  rising tail (min 1.8e-7, last-1000 mean 6.5e-7). The 20000-step blow-up of September is
  not in the new figure.
- The helicity correction gains one order in mixed precision (not two), round-off in double.
- The memory sweep (m = 0, 1, 5) is dropped: paragraph, figure and appendix row.

Applied to `outputs/paper` (main checkout, 2026-09-17 late): figures `lbfgs_smoothing.pdf`,
`newton_vs_lbfgs.pdf`; tables `smoothing_table.tex` (restart column dropped), `velocity_table.tex`
(floor unit 1e-8), `helicity_table.tex`, the descent row of both `newton_convergence_table*.tex`,
rows of `appendix_run_parameters*.tex` and `hyperparameters_table.tex` edited and coloured red;
20 blocks of `mrx_jcp.tex` wrapped in `\begingroup\color{red}` ... `\endgroup` with a
`% TODO(m = 0):` comment carrying the new numbers; the paper compiles (49 pages). Originals in
`outputs/paper/old_m1/`. The figure script copy with the m = 0 manifest:
`outputs/paper_m0/paper_figures.py` (this worktree), its output in `outputs/paper_m0/out/`.

**Later the same evening.** The descent's label in the paper is "gradient descent" (Tobias: no
mention of m; the memory is dropped from the paper and the code). The gamma = 0 arm is an exact
2-cycle (lag-1 correlation of the residual -1.000: residual 2.7e-6 / 4.6e-7 with steps 3e-5 /
1.7e-4 on every pair of steps, steepest descent zig-zagging on the stiffest grid-scale mode), so
`lbfgs_smoothing.pdf` draws its odd and even steps as two dashed lines (`Arm(..., branches=2)` in
`outputs/paper_m0/paper_figures.py`); both branches descend parallel to the smoothed run, the
lower ending at 1.7e-7 against 1.1e-7. The L-BFGS memory was then removed from the code on this
branch (`history_size`, the two-loop recursion, Powell's restart, `--history`, `--method lbfgs`
-> `--method gradient`).

**The 2-cycle explained (2026-09-18).** Steepest descent with an exact line search converges to
a period-2 orbit in the plane of the stiffest and softest active modes (Akaike 1959), and in that
limit `1/dt_1 + 1/dt_2 = lambda_max + lambda_min`. gamma = 0: steps 3.00e-5 / 1.74e-4, sum
3.905e4, the top eigenvalue of the Hessian that the spectrum session's Lanczos run found
(3.9e4); residual branches a factor 5.80 apart, contraction per pair 0.99942, `dt = dt*` on every
step, cos(u, F) = 1. gamma = 1: the same cycle (lag-1 correlation -1.000) with branches 5% apart,
steps 9.4e-4 / 8.9e-4, sum 2.19e3: the smoother lowers the effective top eigenvalue 18-fold; rate
per pair 0.9995, cos(u, F) 0.85. Written into `outputs/paper/mrx_jcp.tex` (caption of
Fig. gamma_sweep and the two paragraphs after it, red, marked NEW TEXT) with the reference
`akaike_successive_1959` added to the bibliography; the paper compiles (50 pages).
