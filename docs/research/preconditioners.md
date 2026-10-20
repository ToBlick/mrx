# Preconditioners: measurements

Iteration counts and timings of the preconditioners in use and of those that lost.

Setting unless stated: li383 p = 2, float64, one H200, odd view, Dirichlet, CG to 1e-8. M = (48,64,24) and L =
(64,80,32), graded from 2026-10-17 on. Geometries of the 2026-08 records: toroid = analytic with eps 1/3, rot-ell =
rotating ellipse with nfp 3, W7-X = fitted map with nfp 5, QA = Landreman-Paul QA wout, quasr9983, quasr44970 and
hegna = GVEC exports. Their mesh is (n,2n,n) with p = 3.

## 1. Laplacian atom on graded spaces: strips and band-wise profiles (2026-10-17)

Level blocks -> strips with band-wise profiles, its / ms:

| solve | M | L |
|---|---|---|
| S_0 | 135 / 32.1 -> 64 / 19.6 | 173 / 76.1 -> 68 / 35.8 |
| S_1 co-exact | 170 / 46.5 -> 90 / 32.2 | 215 / 112 -> 95 / 62.2 |
| S_2 co-exact | 101 / 21.6 -> 80 / 22.9 | 121 / 38.7 -> 83 / 37.7 |
| L_1 | 1263 / 987 -> 763 / 668 | 1691 / 2562 -> 886 / 1441 |
| L_2 | 1647 / 2532 -> 922 / 1766 | 2239 / 6523 -> 1096 / 3840 |
| L_3 saddle | 1418 / 800 -> 555 / 363 | 1887 / 1778 -> 601 / 661 |
| M_1 | 60 / 13.8 -> 56 / 14.6 | 60 / 26.3 -> 57 / 27.7 |
| M_2 + 0.01 L_2, no strips | 6254 / 2376 -> 4768 / 1939 | 10956 / 7794 -> 6914 / 5228 |

- Each part alone fails on M. Strips alone give S_0 78, S_1 115, L_1 967, L_3 658. Band-wise profiles alone give S_0
  125, S_1 160, L_1 1219, L_3 1398.
- The uniform space with the atom on L takes S_0 182, S_1 263, S_2 254, L_3 1972 for 235.6k Dirichlet 1-form DoFs,
  against 188.6k graded.
- Cost: a Laplacian atom apply at k = 1, 2 goes 0.055 -> 0.105 ms on M and 0.098 -> 0.19 ms on L. It is
  launch-bound. The mass atom goes 0.06 -> 0.10 ms on L, so M_1, M_2 and S_2 are 3-7 % slower in time. The k = 1
  atom stores 4.6M -> 6.6M numbers on L. Build +43 s on L.

## 2. Angular profile of the stiffness diagonal, STIFFNESS_PROFILE_EXPONENT = 0.5 (2026-10-18)

- Graded (18,32,12), dense, its / kappa, without -> with the profile: S_1 71 / 68 -> 42 / 24, S_0 48 / 33 -> 39 /
  25, S_2 60 / 64 -> 25 / 8. Free spaces: S_1 85 -> 57, S_2 61 -> 28, S_0 65 -> 52.
- The exponent: S_1 on (18,32,12) takes 50 / 42 / 42 / 49 its at 0.25 / 0.5 / 0.6 / 1.
- Matching the diagonal of the Hodge Laplacian instead gives S_1 77 its, worse than without a profile.
- table_bench with the S_2 multigrid, its / ms, without -> with:

| solve | M | L |
|---|---|---|
| S_1 | 89 / 32.2 -> 57 / 20.6 | 95 / 63.3 -> 62 / 41.1 |
| S_0 | 63 / 18.9 -> 53 / 16.0 | 68 / 37.1 -> 59 / 32.6 |
| L_1 | 279 / 93.9 -> 231 / 77.5 | 300 / 183.3 -> 258 / 157.0 |
| L_2 | 363 / 132.8 -> 263 / 96.6 | 386 / 253.2 -> 289 / 189.3 |
| L_3 split | 156 / 61.6 -> 109 / 45.1 | 165 / 113.8 -> 118 / 82.2 |

- Masses and the saddle solve are unchanged. S_2 by the atom alone goes 78 / 22.5 -> 25 / 7.4 on M and takes 26 /
  12.3 on L. The V-cycle takes 11 / 5.7 and 11 / 8.3.
- Shifted solves on (18,32,12): eps 1e-2 k = 1 508 -> 334 and k = 2 868 -> 540. eps 1e-4 136 -> 139 and 199 -> 214.
- Tried and dropped: own profiles for radial bands of the outer level (S_1 71 -> 70). An exact solve of the Hodge
  Laplacian on the rings r > 0.7 around the atom gave S_1 71 -> 54 at three applies per step.
- Study: `outputs/multipatch/graded/edge/NOTES.md`.

## 3. Geometric multigrid for the co-closed S_2 (2026-10-17)

- Atom -> V-cycle, its / ms. Graded: (18,32,12) 75 -> 11, 9.4 -> 5.5. M 104 -> 11, 25.4 -> 9.5. L 125 -> 11,
  44.9 -> 12.5. Uniform: M 208 -> 10, 38.5 -> 7.1. L 257 -> 10, 66.5 -> 9.6.
- Extra build 26-33 s and 30-75 stored numbers per DoF on M and L.
- As the preconditioner of the hat operator L^_2 it failed: (12,20,12) 226 -> 1613 its graded, 215 -> 3302
  uniform.
- Study: `outputs/multipatch/graded/multigrid/NOTES.md`.

## 4. Slab mass model for M_1, M_2 (2026-10-16 .. 18)

All slab mass-solve timings in this section were measured with the solves stopping in the mass-atom norm. Since
2026-10-18 the M_1 and M_2 solves stop in the norm of the slab model itself.

- kappa of P M_k by 40 Lanczos steps, atom -> slab: (12,20,12) M_1 37.8 -> 3.30 (49 -> 15 its), M_2 38.0 -> 3.05.
  M (uniform) M_1 45.5 -> 3.48 (55 -> 15 its), M_2 45.4 -> 3.34. Even view and field period alike.
- MassChebyshev degree at tol 1e-4: 33 -> 9 on (12,20,12), 36 -> 9 on M.
- Dense factor on M: 151M stored numbers, apply 0.83 ms against 0.05 ms for the atom and 0.17 ms for the mass
  apply. The M_1 solve took 15 ms against 12 ms with the atom.
- FACTOR_DTYPE = float32 (default since 2026-10-17): kappa unchanged at 3.48, apply 0.52 ms, M_1 solve 12.4 ms with
  the dense factor. M_1 solves are 4-16 % faster than with float64 factors at half the bytes.
- THETA_HALF_WIDTH = 4: on M kappa 3.49, apply 0.41 ms, 21M stored, M_1 solve 11.2 against 11.9 ms for the atom. On
  L with float32 factors kappa 3.43, apply 0.70 ms, 47M stored, solve 19.1 against 22.1 ms. A half-width of 2 gives kappa
  4.02 and a solve of 17.9 ms on L.
- One product per sweep step (2026-10-17): M_1 solve on graded M 11.0 -> 9.0 ms, on graded L 23.1 -> 17.0 ms, at
  2.7x the stored numbers. With float32 factors 8.3 ms on M and 14.1 ms on L. Slower variants that were tried: the
  backward sweep as a scatter (2.5 ms), separate batched passes, two blocks per step (0.61 ms).
- Theta-zeta ring slabs instead of zeta slabs leave kappa at 49.0, the atom's value.
- Default for the M_1, M_2 solves since 2026-10-18, graded, atom -> slab: M 55 its / 10.4 ms -> 15 / 8.4 ms, L 56 /
  17.2 -> 15 / 14.0 ms. Laplacian solves are unchanged. Build +32 s on M.
- Inside the Newton mass polynomial the slab cut the degree 37 -> 9 and a Newton step 2.6 -> 2.2 s on graded M (second
  chunk). It stays out for symmetry, not speed: its float32 factors leave the Hessian symmetric only to 2.5e-5.
- As the norm of the Laplacian solves and inside the hat solves the slab made them slower: co-exact S_1 121 against
  58 ms, L_2 4.8 against 2.7 s (2026-10-17).
- Build one radial block at a time (2026-10-20, li383 graded p = 2, model k = 1 Dirichlet, at 36 new the k = 1 free
  model because the first one also compiles). Peak memory of the build above its start, old -> new, CPU 32 cores:
  (36,60,36) 6.4 -> 0.67 GB, (48,80,48) 21.4 -> 2.1 GB, (60,100,60)
  50.4 -> 3.5 GB, (72,120,72) 105 -> 5.8 GB. Per DoF 33 / 47 / 55 / 67 kB old, 3.4 / 4.5 / 3.8 / 3.7 kB new, the
  stored tables included. H200: 53 -> 4.0 GB at 60, the old build ran out of memory at 72, the new one peaks 23.2 GB
  with the rest of the build. Build per model on the H200 30-46 s at 60 for both, 63-83 s new at 72. CPU 72 2321 s
  old, 1376 s new (different nodes). On one node the new build takes 15-30 % longer at (16,32,12), 12-17 % at
  (24,40,24) and 0-12 % at (36,60,36).

## 5. Slab Laplacian model for S_0 (2026-10-17, removed 2026-10-18)

Removed because its factors are dense in theta, n^4 storage in all.

- Dirichlet S_0, atom -> slab, its / ms: M 149 -> 10, 35.3 -> 9.4, kappa 147 -> 2.6, 57M stored. L 182 -> 10,
  76.8 -> 17.5, 157M stored. W7-X (32,64,36) 98 -> 10, 21.1 -> 7.7.
- Inside L_1, which holds two S_0 solves: M 1518 -> 789 its, 1120 -> 515 ms. W7-X 563 -> 256 ms. On L the count did
  not move (1875 -> 1877) and the time went 2621 -> 2364 ms.
- The same model on the Hodge surrogate of the Dirichlet 1-forms gave co-exact S_1 221 -> 17 its but only 57 -> 51
  ms, and L_1 1124 -> 3245 ms with 510M stored numbers. Not kept.
- Theta band on graded M, its / ms: dense 9 / 8.8, half-width 8 14 / 13.9, 4 19 / 21.9, 2 33 / 31. The banded apply
  is launch-bound, so the band saves storage only.

## 6. Boundary-penalty scale, PRODUCTION_BC_SCALE = 3.0 (2026-08-24)

Setting: saddle MINRES with the atom as outer Schur block, (12,24,12) p = 3, tol 1e-10, maxiter 10000, free rows
only. The penalty is alpha_k = <m_k sqrt(g^rr)> / <m_k/J> / h_last times s.

| geometry k | s=0 | 1 | 2 | 2.83 | 4 | 8 | 32 | optimum |
|---|---|---|---|---|---|---|---|---|
| toroid 1 | 599 | 375 | 337 | 319 | 300 | 275 | 271 | 16 |
| toroid 3 | 411 | 228 | 180 | 162 | 140 | 130 | 143 | 5.7 |
| rot-ell 1 | 1367 | 1033 | 989 | 973 | 972 | 1011 | 1444 | 4 |
| rot-ell 2 | 2625 | 1841 | 1754 | 1729 | 1714 | 1754 | 2469 | 4 |
| W7-X 1 | 2832 | 2096 | 2125 | 2202 | 2292 | 2686 | 4424 | 1.4 |
| W7-X 2 | 6887 | 4652 | 4635 | 4737 | 4940 | 5610 | 9366 | 1.4 |
| W7-X 3 | 1445 | 733 | 617 | 576 | 555 | 558 | 678 | 5.7 |
| quasr9983 3 | 720 | 476 | 388 | 354 | 316 | 249 | 210 | 16 |
| quasr44970 2 | 6245 | 4489 | 4231 | 4133 | 4113 | 4150 | 5185 | 5.7 |
| hegna 1 | 4036 | 3325 | 3721 | 3952 | 4282 | 5303 | 9523 | 0.5 |
| hegna 2 | >=10000 | 9068 | 9886 | stall | stall | stall | stall | 1 |

- Optima span 0.5 to 16. s = 0 is the worst value in 17 of 18 rows.
- Worst-geometry geomean excess over each row's optimum: s = 2 1.36x, 3 1.27x, 4 1.21x, 8 1.38x. s = 3 is
  insensitive over about [2, 8].
- Per-geometry calibration was retired: a perfect prediction of the optimum buys at most 27 %.
- A derived coefficient that needs a weak inverse ties on toroid and W7-X and wins 8-54 % on quasr9983. Its build
  costs 11.6-11.9 s against 1.1 s, and the total time of the penalty is 0.751 of it.
- Inside the Hodge solves (QA, n = 8..32, 2026-09-02) the optimum is s = 1-3 at every n, worth 25 %.

## 7. Atoms against point Jacobi and raw Kronecker masses (2026-08-22, old harness)

These ran on the surrogate S_k + D B D^T with a bespoke CG. Ratios hold. Absolute counts do not describe the code.

- Laplacian, iterations atom / Jacobi, median 0.31 over 120 cells and 0.25 for n >= 24:

  | geometry k bc | n=8 | 16 | 24 | 32 |
  |---|---|---|---|---|
  | toroid 1 free | 0.29 | 0.19 | 0.14 | 0.12 |
  | cylinder 1 free | 0.28 | 0.25 | 0.23 | 0.21 |
  | rot-ell 1 free | 0.57 | 0.50 | 0.41 | 0.36 |
  | W7-X 3 free | 0.54 | 0.31 | 0.23 | 0.19 |
  | W7-X 1 free | 0.56 | 0.67 | 0.60 | 0.52 |

- Over p = 2..5 at n = 12 point Jacobi grows 7.6-12.2x, the atom 2.3-2.8x.
- Total time, rot-ell n = 20 k = 1 free: atom 26.3 s, Jacobi 53.7 s.
- Mass, metric lumping against raw Kronecker factors, CG to 1e-8: 0.83x iterations at the median and 0.70-0.77x at
  k = 1, 2, equal build. Point Jacobi took 150-750 its at p = 3 and up to the 5000 cap at p = 5.

## 8. Shifted-stiffness atom for M_k + eps S_k (2026-09-02)

CG on (M_k + eps S_k) x = M_k u, eps = 0.064/n_r^2, tol 1.5e-8, p = 3, Dirichlet. Iterations of the mass atom /
the shifted atom / J in the 1-D masses / M in the 1-D masses:

| case | k=2 | k=1 |
|---|---|---|
| li383 (8,16,8) | 153 / 69 / 100 / 99 | 181 / 74 / 111 / 108 |
| li383 (12,24,12) | 371 / 117 / 203 / 199 | 422 / 128 / 222 / 222 |
| toroid (12,24,12) | 82 / 44 / 87 / 79 | 99 / 49 / 103 / 97 |
| QA (12,24,12) | 618 / 205 / 348 / 335 | 734 / 220 / 424 / 408 |
| W7-X (12,24,12) | 233 / 89 / 151 / 146 | 270 / 84 / 171 / 167 |

- The shifted atom is 1.9-3.3x better than the mass atom. Its count grows as n_r^1.3, the mass atom's as n_r^2.2.
- It loses as eps -> 0. li383 (8,16,8), shifted against mass atom: eps 1e-6 195 vs 100, 1e-4 150 vs 145, 1e-3 145
  vs 329, 1e-2 309 vs 810. The crossover is at eps n_r^2 ~ 0.006.

## 9. Newton system preconditioner (li383 (16,32,32) p = 2 float64, 2026-09-07 .. 18)

- Condition an exact model inverse would leave: 12.7 for the harmonic Gauss-Newton model, 6.8e4 for the Laplacian
  atom, 2.7e5 unpreconditioned. The Hessian is Gauss-Newton to about 1 % on li383.
- Fraction of the exact Newton decrement at 300 / 1000 / 4000 MINRES its: Laplacian atom 8 / 24 / 83 %, harmonic
  sandwich 58 / 93 / 99 %, factored preconditioner 6 / 16 / 59 %.
- The harmonic atom at 100 MINRES took 5.0 s/step against 12.5 s/step for the Laplacian atom (H100, 2026-09-11).
- The old floor kappa = 3 of the harmonic atom (12.1-15.4) sat 25-1000x above the lumped strain (0.01-0.5). It was a
  regulariser and was replaced by the penalty in units of the strain (relaxation.md 3).
- The harmonic atom mis-models the field-aligned near-null modes u = f B by about 1e3.

## 10. Decided against / removed

Boundary-term variants (2026-08-21/22, old harness, against the scalar term):
- Hard u.n = 0 by a 1e4 penalty: 250 its against 76 at k = 1 free, worse than no term.
- Cross term at the computed coupling: indefinite on every geometry, 5-15x worse or capped at 40000. Full-rank
  cross term: SPD but 1.35-1.7x worse.
- Exact 2-D face weight by Woodbury: 1.35-1.9x worse (rot-ell n = 20 1283 vs 661).
- Trace pin: no change (76 / 76). Evicting the trace ring to a Jacobi diagonal: 1.5x worse.
- Tangential penalty: its optimum differs per geometry (0 / 10 / 30). A Dirichlet-side term: 62 -> 382 its at 5 %.
  A Nitsche term: indefinite, 76 -> 293-2723 its.
- Outer dense rings and a truncated-Fourier coarse space: fewer iterations but no gain in time. k = 1 free n = 20,
  build + CG: rot-ell coarse space 22.3 s, scalar term 26.3 s, Jacobi 53.7 s, one dense ring 94.1 s. W7-X 84.5 /
  112-121 / 188.5 / 164.2 s. Storage linear in the DoFs, 102 MB at 43300 DoFs against 0.1 MB for the atom.
- 2-D separable ring atom: fails on the outer rings (toroid k = 3 free 65 vs 24, W7-X k = 2 free 616 vs 200).
- Radial bands, li383 (12,24,24) (2026-09-07): 10-18 % fewer its on Dirichlet k = 1, 2, nothing on free.

k = 0 atoms (2026-06 .. 08-18, all replaced by metric lumping):
- Fast diagonalisation with rank > 1 corrections: an outlier 23-124x below the spectrum gave kappa 1e5, a Chebyshev
  degree explosion and OOM.
- Greville-combined atom: not h-independent, W7-X free stalled at rank 1 (5664 its).
- Geometric multigrid for k = 0 (2026-07/08): the best two-level fat-core cycle won only on free walls (1.05-1.4x)
  and lost on Dirichlet everywhere. W7-X (12,24,24): 18-19 its against 64, but about 10 ms per iteration against 1
  ms, so the separable atom was 1.8-2.8x faster in wall time. A point-Jacobi smoother needed 47-180 its against 5-15
  for the separable atom.

Mass preconditioners (2026-06 .. 08-17):
- Greedy CP past rank 1: an indefinite surrogate and NaN on W7-X (12,24,12).
- Inner Schur: half the iterations at 3x the wall time. Block-Chebyshev polish: 10x fewer iterations at 10x the
  wall time. Lumped block-SGS for the k = 1, 2 coupling: 291-350 its against 75-80.

Auxiliary space and Chebyshev (2026-06, shelved):
- HX for k = 1 grad-div on toroid (6,12,4): 96 its / 117 ms against Jacobi 386 / 290 ms. k = 2 was h-flat at 33-34
  its but 3.3x slower in wall time. At k = 3 Jacobi won, 0.47 s against 10.4 s.
- Chebyshev-8 on the approximate Schur complement: 67 against 96 its but 3x slower in wall time. Whole-operator
  Chebyshev fails on the singular free k = 0, 1.
- k = 1 saddle with P_A + P_B on W7-X (2026-08-13): kappa_eff 7.6e6 Dirichlet and 7.9e7 free, a continuum of small
  modes.

Newton preconditioner variants (li383 (16,32,32), 2026-09-07):
- Coarse space of the lowest Lanczos pairs: 269 / 266 / 266 / 264 its to 0.1 for 0 / 10 / 30 / 100 pairs.
- Symbol variants with bundled weights or powers 1/2, 3/4: spread 1.1e3-9.6e3, none beats the sandwich.
- Factored preconditioner: the best model spread (243 its) but the worst direction on the true system, energy
  1.2e-8 against 6.9-8.6e-8.
- Levenberg-Marquardt shift lambda M_2 at 1e-3 of the top: 6 its, but the residual stays at 4e-4, above the descent.
