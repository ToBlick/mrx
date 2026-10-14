# Solvers and precision: measurements

The solve routes (Hodge split, shifted split, saddle MINRES at k = 3, refinement) are described in
the Sphinx docs. This file keeps the numbers behind them. Default precision since 2026-09-25 is
plain float32 with tol 1e-5. "mixed" = float64 residual + refinement, tol 1e-8 (the paper's runs).
Rows marked "tol 3.5e-4" are from the 2026-09-05..24 plain-float32 default sqrt(eps).

## 1. Hodge split for L_k^-1, k = 1, 2 (QA p=3, tol 1.5e-8, 2026-09-02)

x_perp iterations [cumulative incl. the g-solve and the correction] and wall:

| case | saddle MINRES | split |
|---|---|---|
| n=16 k1 dbc | 1022 / 2.7 s | 307 [450] / 3.1 s |
| n=16 k1 free | 2584 / 7.7 s | 1458 [1707] / 5.5 s |
| n=16 k2 dbc | 2555 / 8.8 s | 367 [987] / 4.7 s |
| n=16 k2 free | 7122 / 9.6 s | 1636 [4415] / 9.5 s |
| n=24 k1 dbc | 1553 / 7.1 s | 418 [602] / 4.1 s |
| n=24 k1 free | 3889 / 26.2 s | 2174 [2481] / 14.1 s |
| n=24 k2 dbc | 4556 / 35.5 s | 500 [1337] / 8.6 s |
| n=24 k2 free | cap 10000 / 35.2 s | 2390 [6485] / 29.9 s |

- Cumulative gain 1.5-3.4x iterations, 1-4x wall.
- PCG on the singular S_k fails: the rhs b - M D g is consistent only to the g-solve tolerance
  (4.7e-6 relative, QA k1 free). Residual 1.8e-5 at it 80, 1.1 at 160, 2e9 at 320. In production
  k1 dbc floored at 1e-6, k1 free / k2 / k3 blew up to 1e14-1e20. Hence PCG runs on the SPD L^_k.
- Split vs the singular-S_k route gap (2.5x dbc, 12x free at k=1): the preconditioner re-injects
  exact-form components (kernel fraction of the search direction 0.4-0.9).
- k=3 split (two L^_2 solves) vs MINRES, QA n=16: 368 + 403 vs 694 it, error 2.6e-8 vs 5.0e-8, but
  exact residual 3.8e-6 vs 8.7e-8 (M_2^-1 amplification). The Leray projection needs the exact
  residual, so k = 3 stays on saddle MINRES.
- The split's exact residual in the coefficient 2-norm is 10-100x looser than MINRES's at equal error.
  In the M^-1 norm they agree (n=16 MINRES vs split: k1 dbc 2.4e-8 vs 7.2e-8, k1 free 8.7e-8 vs 5.9e-8,
  k2 dbc 2.1e-8 vs 2.5e-7, k2 free 7.5e-8 vs 5.1e-8).
- Free-BC errors 1e-5..1e-4 at tol 1.5e-8 for both solvers: err/res ~5000 free (near-harmonic modes,
  eigenvalue ~1/R^2), ~1 dbc.

## 2. Shifted split for (M_k + eps L_k)^-1 (li383 p=3 float64, k=2 dbc, tol 1.5e-8, 2026-09-02)

- Premises: weak-vs-strong D and S consistency 2.5e-16..3.2e-16. ||S_2 D_1 v|| relative
  2.0e-16..2.5e-16.
- Iterations (split = k2 CG + k1 CG, both with the mass atom, PR19 = MINRES with 1/eps P_L):

  | mesh | eps | MINRES, mass atom | MINRES, PR19 | split | ||dx||_M |
  |---|---|---|---|---|---|
  | (8,16,8) | 1e-6 | 497 | 3623 | 100 | 3.6e-8 |
  | (8,16,8) | 1e-3 (= 0.064/n_r^2) | 2134 | 1675 | 330 | 5.0e-8 |
  | (8,16,8) | 1e-2 | 5798 | 943 | 808 | 8.2e-8 |
  | (12,24,12) | 4.44e-4 (0.064/n_r^2) | 8478 | 3650 | 772 | 7.7e-8 |
  | (16,32,16) | 2.5e-4 (0.064/n_r^2) | 20362 | 6397 | 1326 | 8.8e-8 |

- At 0.064/n_r^2 the split takes 6.5x / 11x / 15x fewer iterations than MINRES with the same atom,
  5.1x / 4.7x / 4.8x fewer than PR19. Wins at every eps (no crossover). True residual through the
  nested Laplacian 6e-8..7e-7 (split) vs 3e-8..1e-7 (MINRES).
- With the shifted-stiffness atom (preconditioners.md section 7) the whole smoothing solve is 145 /
  249 it at (8,16,8) / (12,24,12), from 2134 / 8478. Growth ~n_r^1.3. Before the split the smoothing
  solve was 75% of a relaxation step (issue #18). These benchmarks used 0.064/n_r^2. Production
  smoothing is 0.02/n_r^2 since 2026-09-05 (3.2x smaller shift, fewer iterations).
- The resistive MINRES it replaced (W7-X fmm002 (8,16,8) p=3, tol 1e-12, 2026-08-26/27): eta 1e-3
  383 it/step, 1e-2 612 mean / 1938 max, eta 1e-1 and 1 hit maxiter 10000 on 8-10 of the first steps
  (energy +3e-7, div B 6e-6).

- Shifted solve on li383 (48,64,24) p=2 float64, one H200, against PR #33 (2026-10-14). Two changes:
  - **Gradient correction.** The shifted atom models M + eps L and is too small on gradients by 1 / (1 + eps lambda),
    where M + eps S is S_{k-1}. The preconditioner of M_j + eps S_j adds G P_{j-1} G^T.
    - Single CG on M_1 + eps S_1, odd view: eps 1e-2 4228 -> 220, eps 1 > 5000 -> 274. An exact 0-form solve in
      its place gains little: 211, 247.
  - **Three solves as the Hodge split** (z, x_perp, a), in place of x = x_k - eps G z.
    - With two solves the error of z reached the residual as eps M G M_{k-1}^-1 r_z. At eps = 1 every pass gained
      only a factor 8 and the solve stopped at the pass limit. Tightening the (k-1)-form solve 1000 times made it
      converge in 2 passes, the k-form solve did not matter.
    - With three, the error of z cancels and each solve stops on its own residual at tol / 3. Every eps below
      converges in one pass.
    - Tried and dropped: judging the two solves on their contribution to the posed residual (an M_k apply per
      iteration, +50 %, then tested every 16 iterations), and the same in refine (two passes per solve, 0.49
      s/step).

  seq.shifted(k, eps).solve on M_k z, iterations and time, PR #33 -> both changes:

  | eps | k = 1 (even view) | k = 2 (odd view) |
  |---|---|---|
  | 7.4e-6 | 901, 0.36 s -> 548, 0.22 s | 1254, 0.48 s -> 1015, 0.44 s |
  | 1e-4 | 1190, 0.47 s -> 491, 0.18 s | 1629, 0.60 s -> 631, 0.28 s |
  | 1e-3 | 3032, 1.23 s -> 568, 0.21 s | 4321, 1.53 s -> 695, 0.30 s |
  | 1e-2 | 15533, 6.3 s -> 622, 0.23 s | 23315, 8.0 s -> 821, 0.35 s |
  | 1e-1 | unconverged, 24.8 s -> 703, 0.27 s | unconverged, 35.5 s -> 1205, 0.50 s |
  | 1 | unconverged, 24.9 s -> 984, 0.40 s | unconverged, 43.5 s -> 2538, 0.99 s |

  - The Laplacian and the shifted solve share the split: delta M + L with delta = None (static, the Laplacian) or
    1 / eps. L_1, L_2 solve times unchanged against main (1.10 -> 1.11 s, 2.58 -> 2.59 s, even view). The
    reported count now sums the three solves.
  - The velocity smoothing only needs curl of the result, and (M + mu L)^-1 and (M + mu S)^-1 differ by an exact
    form: it is one solve of M_1 + mu S_1 (`part="coexact"`). Gradient relaxation, 200 steps, paired twice on one
    node: 0.260 / 0.260 -> 0.228 / 0.226 s/step, the same energy and residual. With the full three-solve
    smoothing it was 0.307.

## 3. Saddle MINRES: the diagonal lower block (2026-08-24)

- The saddle lower block was silently a per-DoF diagonal (the default mass preconditioner was gated
  on a deleted tensor kind). A standalone MINRES with the real mass preconditioner took 84 it where
  the library took 9612.
- After the fix, (12,24,12) p=3, tol 1e-10, maxiter 10000, 3 geometries x k = 1..3 x free/dbc: Jacobi
  outer converged 2/18 -> 18/18. Atom outer 13/18 -> 18/18. Atom-outer iterations: toroid k2 free
  9612 -> 314, k2 dbc 9683 -> 301, quasr k1 free 6317 -> 857, W7-X k2 dbc 10000+ -> 1522, W7-X k2 free
  10000+ -> 4659. The atom as outer Schur block is still worth 2.51x on top (36776 vs 14664 total).
  W7-X p=5 k=2 free: 123 it with the mass preconditioner on.
- CG on L_2 with nested mass CG converged in 993 it but cost 1007 s vs 30 s (Krylov-in-Krylov): the
  formulation was fine, the preconditioner was the fault.
- CGS vs MGS in the MINRES Lanczos step: identical counts (75/75, 84/84, 81/81), max
  |v.r1|/(|v||r1|) 1e-17..1e-20. MINRES phibar tracks the true dual residual within ~7x.

## 4. Harmonic forms

- Seeds from histopolated constants (exactly closed, one solve each): Rayleigh quotient / lambda_1
  1e-11 for every QA form, vs up to 4.5e-4 with the old M^-1 load((0,0,1)) seeds (stalled at p = 4,
  n >= 16) (2026-09-02). Betti numbers (1,1,0,0): harmonic forms at (k=0, free), (1, free), (2, dbc),
  (3, dbc).
- Gate: Rayleigh quotient 7e-27..5e-26 on every geometry and resolution (2026-08-24). QA vacuum
  2e-13..5e-12, ||F||(h) 6e-14..4e-13 (2026-08-28). Across four solver arms whose fields agreed to 5
  digits, ||L v|| spanned 1e-8..4.6e-3 while the Rayleigh quotient stayed ~1e-13: judge harmonic
  forms by the Rayleigh quotient, not ||L v||.
- W7-X p=5 k=1 free harmonic Rayleigh 6.6e-9 -> 4.7e-24 once the Laplacian solve used the atom
  (2026-08-24).
- k=1 free harmonic form before the saddle mass fix (relL2 = sqrt(v.Lv / v.Mv), 2026-08-24): toroid p3
  6.5e-12. Rot-ell p2/p3/p5 3.9e-12 / 1.3e-11 / 3.0e-2. W7-X 8.4e-13 / 3.0e-4 / 1.7e-1. quasr9983 p3
  4.5e-7. quasr44970 p3 7.7e-4. Hegna p3 4.7e-3. Not the inner tolerance: W7-X p=2 tracks tol 1:1
  (4.9e-8 at 1e-8 .. 6.2e-13 at 1e-14), p=3 flat 3.0e-4 and p=5 flat 1.71e-1 across 1e-8..1e-14.
  W7-X p=3 7.9e-3 at maxiter 10000 vs 3.0e-4 at 20000: the budget was hiding a stall.
- k=1 free harmonic form at fixed tol degrades with n and p (QA, 2026-08-30, before the k=1 fix
  0c3aa4d): p=3 Rayleigh 6e-14 (n_el 5), 1e-10 (17), 7e-8 (21), 2e-5 (29), 3e-3 (45). p=4 n_el 29
  6.4e-4. k=2 stays ~1e-12.
- A tolerance is not an accuracy: W7-X k=1 free harmonic Rayleigh 1.2e-13 / 9.3e-8 / 1.6e-6 at 8^3 /
  12^3 / 16^3 because an inner L_2 solve asked for 1e-13 and exited at maxiter silently (2026-08-20).

Decided against:
- Inverse-iteration construction (W7-X 32^3 geometry, p=3, eps 1e-4, 2026-08-17): the k=2 solve fails
  above n ~ 18 (Rayleigh 2.3e-10 at n=18, 2.8e-6 at 20, 1.7e-3 at 24), kappa(S + eps M) ~
  lambda_max/eps with lambda_1 = 21.84. Direct Hodge and inverse iteration agree to 5 digits at
  (8,16,16) but direct cannot bootstrap b2 > 0. Rank-1 harmonic coarse correction: +1-2 sweeps and 5
  orders worse residual.
- Inverse-iteration polish of a computed form (6 steps, eps 1e-4) walks away: W7-X p3 3.0e-4 ->
  1.4e-2, quasr9983 4.5e-7 -> 1.8e-5 (2026-08-24).

### Co-closed solve: S_k alone for a right-hand side G_k^T w (2026-10-09/10)

`seq.L[k].solve(G_k^T w, part="coexact", gauge=False)` (named `solve_coclosed(w)` when these notes were
written, and in the tables below), branch `laplacian-solvers`. Scripts and logs: `outputs/hodge_gamma/`,
`outputs/laplacian_solvers/` (main checkout), bench notes `boxes_and_exact_forms_2026-10-09.md`.

- The atom preconditions `S_k` well and the exact-form term `M G W G^T M` of `L^_k` badly (dense spectra of the
  multigrid session, 12x20x20: condition 1903 / 1130 with the term, 93 / 103 for `S_1` alone, li383 / W7-X).
- Weighting the term by gamma is worse the smaller gamma gets (24x42x42, `b = S_1 x`, float64 to 1e-8: 467 / 384
  iterations at 1, 698 / 633 at 0.1, none at 0.001). gamma = 0, the singular `S_k`: 147 / 148 (k = 1), 145 / 142
  (k = 2). Scaling the atom's own model of the term up is worse too.
- Condition: the right-hand side must be consistent. PCG on a singular `S_k` converges for any SPD preconditioner
  on a consistent right-hand side, with a bounded kernel part in the iterate (Kaasschieter 1988). An inconsistent
  share delta stays in the residual, and the iterate blows up once the residual reaches about delta.
- Two requirements, each measured on the 2026-10-07 case (li383 (12,16,16), start field, `G_1^T JxB`):
  1. `G^T w` formed in the residual precision. Formed in float32 the second pass runs to maxiter, `|a|_M` 3e5
     against 1e-3.
  2. Restart on the 2-norm of the residual that the pass reduces. `solve_singular_cg` with `A_res` (`refine`,
     dual-norm test) diverges even in float64: `|a|_M` 2e14, curl error 18. This was the 2026-10-07 failure.
     No longer the case since conjugate gradients stop on `dual_norm` themselves (2026-10-12, section below):
     the solve is one call of `solve_singular_cg` and the 2-norm restart is gone.
- Inside the Hodge split it does not work: the middle right-hand side `b - M G g` is consistent only to the
  tolerance of the `g` solve. `S_k` alone there: count -60000, error 1e59. With the stop tied to a measured
  inconsistency `L_1` converges, `L_2` does not (inconsistent at 0.1 and 2 in the atom's norm).
- Ladder meshes M, start field, real `J x B`, second identical call, solve time in s (signed count):

  | | li383 (48,64,24) mixed | float64 | W7-X (32,64,36) mixed | float64 |
  |---|---|---|---|---|
  | force potential, `L[1].solve` | 0.709 (+1105) | 1.399 (+1692) | 0.356 (+567) | 0.688 (+890) |
  | `solve_coclosed` | 0.072 (+289) | 0.100 (+336) | 0.040 (+162) | 0.062 (+221) |
  | `solve_coclosed`, gauge | 0.122 | 0.200 | 0.072 | 0.124 |
  | Leray k=2, saddle MINRES | 1.067 (+2255) | 2.157 (+3700) | 0.334 (+729) | 1.067 (+1953) |
  | `L[2].solve_coclosed(M_3 G_2 v, gauge=True)` | 0.105 (+229) | 0.146 (+290) | 0.067 (+145) | 0.088 (+169) |

  The Hodge-split count is that of its middle solve only. `curl a` agrees with the Hodge split to 4e-7 (mixed)
  and 5e-12 (float64), with the gauge solve `a` itself to 2e-7 / 3e-7 and 2e-12 / 4e-12. The gauge part without
  it is 0.41 / 0.46 of the co-closed potential in the M norm.
- Leray by the co-closed route (`sigma` with `S_2 sigma = G_2^T M_3 G_2 v`, closed part removed by a co-closed
  solve on `S_1`, `F = v - sigma`, no pressure): `F` agrees with the saddle's to 2e-7 / 7e-7 (mixed) and
  4e-10 / 7e-9 (float64). Its divergence is larger: `|div F| / |F|` 1.1e-5 / 9.0e-6 against 9.7e-9 / 3.2e-5 in
  mixed, 2.8e-8 / 2.2e-7 against 1e-12 in float64. The stop is on the 2-norm residual of `S_2`, relative to `|v|`,
  and `|F|` is 0.05 `|v|`. NOT wired into `leray`.
- In a relaxation (main 732f824 with the solve patched in, (24,32,32), a mesh now retired): li383 gradient
  descent 400 steps, same energy change (-1.35e-6) and residual trace, 47.8 -> 19.7 s per 200 steps. Newton 20
  steps li383 and W7-X: same energies, no gain (the force potential is a small share of a step). The gauge share
  `|G_0^T M_1 a| / |M_1 a|` stays between 0.17 and 0.41 over 400 steps.
- Run to 6000 steps on the branch, li383 (48,48,28) gradient descent, mixed: residual 2.4e-4 -> 3.2e-7, still
  falling as a power law (no floor reached), 0.21 s/step, helicity drift 2.9e-6 relative, `|dE - dE_ls| / E0`
  median 3.7e-10. Energy increases on 1098 steps (all below 6e-10, float32 round-off of `E`) and `||div B||`
  grows to 1.7e-4. Both also occur with the Hodge split (33 of 400 steps and 4e-6 after 400 steps in both arms
  of the pair above), no production run of this length on this mesh was made for comparison.
- Not covered: plain float32, seeded or driven runs, p = 3, the Newton solve (same kernel, different operator),
  the pass tolerance below the inner tolerance, more than 400 steps of the gauge share.
- A closed-only shortcut (`b = M_k G c`, one solve `S_{k-1} a = M_{k-1} c`) was not built. `M_{k-1} c` is
  consistent only for a co-closed `c`, and no caller holds one.

### Stopping on the M^-1 norm of the residual of the equation (2026-10-12/13)

li383 1.4 m, p = 2, float64, tol 1e-8, one H200, second call. Before: `main` at `11f93d81` (conjugate gradients
stopped on the norm of their own preconditioner, MINRES on `phibar`, the splits on their block residuals).

- Before, a pass that met its own test was judged in `dual_norm`, missed, and ran again: every Laplacian and
  saddle solve took two passes and ended at 1e-13 to 1e-16.
- Krylov iterations stopping on `dual_norm`, Hodge sub-solves at tol / 3, iterations and s, before -> after:

  | | (48,64,24) | (64,80,32) |
  |---|---|---|
  | `L_0` random | 324 -> 167 | 399 -> 204 |
  | `L_1` force potential | 1346, 0.96 -> 768, 0.57 | 1644, 2.18 -> 963, 1.34 |
  | `L_3` random | 3086, 1.64 -> 1491, 0.87 | 4038, 3.55 -> 1925, 1.82 |
  | Leray of `J x B` | 2912, 1.57 -> 1398, 0.82 | 3724, 3.28 -> 1759, 1.67 |

  One pass each. MINRES costs 0.54 -> 0.59 ms and 0.88 -> 0.95 ms per iteration for carrying the residual.
- With only that change the residual of `L_k x = b` computed outside the solve ended above the tolerance where
  `b` has an exact part: `L_1` random 1.2e-8, `L_2` random 3.8e-8, `L_2` odd (`M_2 B`) 1.0e-6 on (48,64,24).
  The block test did not control it: the lower residual counts about 1 / h times.
- Judged on `upper - M_k G M_{k-1}^-1 lower` these end at 2.2e-14, 1.1e-13 and 3.4e-12 in two passes (`L_1`
  random 679 -> 1432 iterations, 0.52 -> 1.06 s). A right-hand side `G_k^T w` stays at one pass.
- With the saddle solve judged on `r_u + D_2 M_2^-1 r_sigma` and on `r_sigma`, the number the loop stops on
  against the outside residual: `L_0` random 9.0e-9 / 9.0e-9, force potential 2.2e-9 / 2.2e-9, `L_3` random
  9.7e-9 / 1.1e-8, `L_2` odd 2.3e-12 / 3.3e-12 on (48,64,24). On (64,80,32): 9.2e-9 / 9.2e-9, 2.2e-9 / 2.2e-9,
  9.7e-9 / 1.1e-8, 5.0e-12 / 5.7e-12. Leray of `J x B` then takes 1433 iterations, 0.92 s and 1800, 1.86 s.
- Mixed on (48,64,24): 2 to 3 passes per solve as before, `L_2` random and odd 4 (3 before), ending at 3.9e-11
  and 7.2e-9.
- Newton solve at the li383 start field (review of 2026-10-10): on (12,16,16) MINRES met its own test after 26
  iterations and `newton_mr` returned -26. It returns +48 now. On (32,44,20) and (48,64,24) the solve ends at
  maxiter before and after.

### Part solves against the full solve (2026-10-13)

Same setting. The full solve of the same right-hand side -> the part, in s. The parts agree with the full solve
to 6e-7 or better.

| | (48,64,24) | (64,80,32) |
|---|---|---|
| `L_1` co-exact, `gauge=False` | 0.78 -> 0.094 | 1.75 -> 0.197 |
| `L_1` co-exact | 0.78 -> 0.150 | 1.75 -> 0.324 |
| `L_2` co-exact, `gauge=False` | 2.08 -> 0.054 | 4.67 -> 0.087 |
| `L_2` co-exact | 2.08 -> 0.148 | 4.74 -> 0.304 |
| `L_1` exact | 1.77 -> 0.105 | 4.08 -> 0.228 |
| `L_2` exact | 4.28 -> 0.801 | 9.55 -> 1.789 |
| `L_3` exact | 1.07 -> 0.722 | 2.30 -> 1.545 |

- Mixed, co-exact and exact with the gauge: `L_1` 0.62 -> 0.119 and 1.02 -> 0.078, `L_2` 1.54 -> 0.136 and
  3.51 -> 0.864, `L_3` exact 1.34 -> 0.802 on (48,64,24).
- The exact part as two stiffness solves (`S_{k-1} a = M_{k-1} g` second) ran to maxiter and drifted for k = 2, 3
  once it was stopped on its residual (about 10000 iterations, wrong by 1e9): `M_{k-1} g` is consistent only to
  the tolerance. The second solve is one of `L^_{k-1}`.
- The gauge solve of an already co-exact guess has only round-off on its right-hand side: a restart from its own
  result ran to maxiter. Only the co-exact solve without its gauge takes a guess.
- What that guess is worth, relax.py, gradient, 100 steps, force-potential solve with / without the potential
  of the step before: 199 / 235 iterations per solve on (48,64,24), 237 / 286 on (64,80,32), 0.39 / 0.40 and
  0.76 / 0.78 s/step.
- Harmonic 2-form by the co-exact solve at `HARMONIC_TOL` = 0.1 of the tolerance (2026-10-11). Rayleigh quotient
  at tol / at 0.1 tol / with the `L_1` solve of `main`: float64 (48,64,24) 1.5e-14 / 1.6e-16 / 8.1e-25, mixed
  3.5e-13 / 2.4e-13 / 2.4e-13, plain float32 (48,64,24) 1.1e-3 / 4.7e-5 / 1.2 and (32,44,20) 5.5e-4 / 3.7e-6 /
  3.5e-3. In plain float32 the solve no longer converges at 0.03 tol on (48,64,24).
- relax.py, `main` -> branch, s/step and `E_0 - E`:

  | | s/step | `E_0 - E` |
  |---|---|---|
  | li383 (48,64,24), gradient, 200 steps | 1.09 -> 0.33 | 1.1500e-6 both |
  | li383 (64,80,32), gradient, 200 steps | 2.38 -> 0.68 | 1.1250e-6 both |
  | li383 (48,64,24), p = 3, gradient, 200 steps | 2.28 -> 0.97 | 1.0797e-6 both |
  | W7-X (32,64,36), gradient, 200 steps | 0.56 -> 0.19 | 2.6208e-7 both |
  | li383 (48,64,24), Newton, 40 steps | 5.90 -> 5.35 | 1.6694e-6 -> 1.6692e-6 |
  | li383 (64,80,32), Newton, 40 steps | 9.41 -> 8.22 | 1.6374e-6 -> 1.6365e-6 |
  | li383 (48,64,24), p = 3, Newton, 40 steps | 8.70 -> 7.76 | 1.5652e-6 -> 1.5659e-6 |
  | W7-X (32,64,36), Newton, 40 steps | 3.89 -> 3.44 | 3.2689e-7 both |

  With the stopping change alone the gradient step is 0.69 and 1.44 s on the two li383 meshes. The gradient step
  on (48,64,24) read between 0.33 and 0.42 s on identical code.
- Not run: Newton in mixed precision on the final code, W7-X above (32,64,36), (64,80,32) at p = 3.
- Plain float32 on the suite's mesh li383 (12,20,12), tol 1e-5 (2026-10-13): 7 of 46 tests fail, `main` passes its
  35 on the same mesh. Leray of a random field: the first MINRES pass stops after 270 iterations on the residual
  it carries, the residual formed from the solution reads 1.17e-5, and the second pass, asked for another 1e-5
  at the float32 floor, runs to maxiter 10000 and returns NaN. `L_3` random: 6 passes, 50278 iterations, ends at
  1.1e-5. Harmonic 1-form: Rayleigh quotient 1.3e-8 against 1.2e-10 on `main`. On (8,12,12) plain float32 passed.
  Not fixed.

### Where the harmonic deflation lives (2026-10-10, reasoned from the code, not measured)

- `solve_singular_cg` wraps the operator as `project_dual . A . project_primal` and the preconditioner as
  `project_primal . P . project_dual`: 4 projections per CG iteration. The saddle MINRES does the same on its
  upper block. With m = 0 harmonic forms the projections are the identity.
- Until this change `project_primal` applied the mass matrix (`vs @ M x`). With m >= 1 that was 2 mass applies
  per iteration on top of the matvec: the odd Dirichlet 2-forms, the free 1-forms, the even free 0-forms and
  Dirichlet 3-forms, and so 2 `M_3` applies per iteration of the Leray saddle solve. An `L^_2` iteration had 5
  mass-type applies where the operator needs 3. The force potential (even, Dirichlet k = 1) has m = 0.
- Now `project_primal(x) = x - ((M vs) @ x) @ vs` with the stored `M vs` (M symmetric). Same projection in exact
  arithmetic. The suite was NOT rerun with it on this cluster (branch `deflation-mass-apply`): run it first.
- Moving the deflation out of the inner solves into the split (steps 1 and 5 of the Hodge solve) would then save
  only 4 m dot products per iteration. It stays per iteration.
- If it were moved: `L^_k` is singular on the harmonic forms, so the inner solve converges only on a deflated
  right-hand side, like the co-closed solve. Steps 1 and 5 alone are not enough. The degree-(k-1) solution `g`
  must be deflated between steps 2 and 4, because `M_{k-1} g` carries its harmonic part into step 4's right-hand
  side and makes it inconsistent. These are per-pass projections in the residual precision.
- An undeflated working-precision pass sees a harmonic part of about 1e-7 of its right-hand side (the float64
  residual cast to float32). It must not stop below about 1e-6.

## 5. Operator identities at round-off

- quasr44970, tol 1e-12 (2026-08-25): curl adjointness 4.0e-13, div(P_Leray v) 4.7e-12, Leray
  M-orthogonality 1.6e-14. div.curl with the mass-projected curl (M2^-1 D1) 1.3e-10 vs raw incidence
  8.6e-16. The two curls agree to 1e-12, so dB = curl E uses the incidence: div B conserved to
  6.7e-14 over 3000 steps.
- Hessian identities, li383 (16,32,32) float64 (2026-09-06): gradient vs force pairing 5e-9 / 7e-8,
  quadratic form vs ||Q||^2 + (B,R) 8e-12, symmetry 6e-9, (B,HB)/|B|^2 1e-20. |HB|/|Hu| 2.2e-4 at the
  IC, 4.7e-6 at step 5000. (u,Hu)/||Q_u||^2 = 1.00-1.015: on li383 the Hessian is Gauss-Newton to ~1%.
- Line-search identity <B,dB>_M = -(F,u)_M: 1.3e-11 relative / 5.6e-17 absolute (2026-08-25).
- Discriminator: a converged solver whose relative error is FLAT in n means a wrong source/rhs, not
  the solver (nbc_k1 37.256, dbc_k2 1.7818 flat to 4 figures: a missing metric factor). A solve that
  reports convergence but is not converged degrades with n instead (2026-08-25).

## 6. Precision

### Attainable residuals and refinement

- float32 attainable true residual on the test mesh (2026-09-05): 2.4e-4 on the k=2 Hodge split,
  4.2e-4 on the k=3 saddle. 5 of 8 Poisson solves could not reach 1e-6 in plain float32.
- refine() at 1e-6 in float32 kept iterates whose extra passes RAISED the residual -> relaxation NaN
  at step 10. Fixed: keep the last strictly improving finite iterate (2026-09-05).
- Float32 underflow in refine (fixed 5887b72, 2026-09-05): at tol 1e-10 on a 1e-12 rhs the float64
  residual cast to float32 gives denormal squared norms (p^T A p, r^T z 1e-44 -> 0, alpha = inf), NaN
  at step 9 (tol 1e-10) and 2394 (p = 4). Fix: the inner solve gets r/||r||.
- refine early stop (stop on the first pass not lowering the true residual) broke mixed precision on
  hard systems (2026-09-11): n = 48 residual 10-50x the paper's from step 2 with |F| rising. Newton
  (32,64,64) 13 consecutive steps dt = 0, cos(u,F) = 0 (the "Newton stall mode"). Removed in e680ab4:
  0 stalled steps, F2 within 1-4% of the paper's arm.
- float32 floor of the QA vacuum distance D: 3.4e-4 (float64 floors 4.5e-5..8e-5) (2026-08-28).
- Trap: under Hydra+submitit, jax_enable_x64 set at module top was ignored and the solve ran float32
  (CG met its M-norm criterion in 11 it while the Euclidean residual was 4.9e-7 vs 8.2e-14 in
  float64): the old "Poisson floor ~3e-4 at p = 3". Set JAX_ENABLE_X64 in the environment (2026-08).
- TPU matmul precision (v5e bf16 MXU: highest = 6 passes, high = 3, default = 1): high/default are
  worth up to 1.55x on the mass kernel, 1.22x per step, but at `high` DPhi carries 1.9e-4 relative error
  and at `default` the map folds (det DPhi down to -1.3e-1). float32 at highest is fine: inverse-mass CG
  20 (k1) / 24 (k2) it on both backends at the same tol (2026-09-03).

### Pollution law (relaxation)

- The force Leray's remnant makes the direction error tol/resid of |F|. Its energy term is
  0.1 tol / resid^2 of the descent (measured 1.4e-4 at resid 1.5e-3, 9.4e-3 at 4.8e-4, tol 1.5e-8,
  li383 float64, 2026-09-04). A tolerance buys a residual ~sqrt(tol/10): 3e-4 at 1e-6, 3e-5 at 1e-8,
  3e-6 at 1e-10. Plain float32 at tol 3.5e-4 is pollution-limited near force residual ~6e-3
  (issue #21).
- Confirmed at tol 1e-6, li383 (16,32,32) p=2 mixed (2026-09-05): block energy sums turn positive
  (+2.5e-7 / 1000 steps) before step 8410, then cos -> 0, dt 2.4 -> 0.03, resid 5e-4 -> 1.2e-1, one
  step releases 6.4e-6 (2x the first 5000 steps' total). The 1/2 chain opens (0.030, closed at tol
  1e-8 / 1e-10). tol 1e-10 reproduces tol 1e-8 to 3 digits at 1.2x cost.
- The coefficient does NOT transfer to plain float32: order 1 follows the float64 residual power law
  to 1e-4 while the stored field's energy falls behind and RISES by 5e-7 at 8000-9000 steps
  (tol/resid = 2 there). Order 0 turns around at resid 1e-3. In plain float32 the stored-field energy
  is the witness. The residual can lie.

### Floor study (li383 (16,32,32) p=2, 20000 steps, L-BFGS m=1 era, 2026-09-05)

| arm (order, tol) | resid at 20000 | E_0 - E | dH | s/step |
|---|---|---|---|---|
| float64 o1 (1e-10) | 8.9e-5 | 1.92e-6 | -5.3e-8 | 1.21 |
| mixed o1 (1e-8) | 8.8e-5 | 1.91e-6 | -5.3e-8 | 0.65 |
| plain f32 o1 (3.5e-4) | 1.02e-4 | 1.35e-6 | -5.3e-8 | 0.23 |
| float64 o0 | 4.1e-3 | 4.85e-6 | +4.3e-6 | 0.71 |
| mixed o0 | 4.0e-3 | 4.77e-6 | +4.3e-6 | 0.39 |
| plain f32 o0 | 3.2e-2 | -5.2e-5 | - | 0.09 |

- Float64 and mixed on one curve to 2%. Power law a = 0.5-0.6, no floor. (s/step are pre-e680ab4.)
- float32 storage: E ~ 0.5 has ulp ~6e-8. Per-step dE is noise (positive on ~40% of steps even in
  mixed). Only block sums are readable. Float64 per-step dE matches the line-search prediction to
  1e-4..1e-6 with 0 increases.
- Plain-float32 trajectories diverge from round-off within tens of steps. Events are
  per-realisation (a second realisation differs already in the first chunk).
- Tolerance sweep on the released code (li383 (16,32,32) p=2, L-BFGS m=1, 2026-09-11): tol 1e-6 /
  1e-8 / 1e-10 reach 0.13e-8 (32000 steps) / 0.35e-8 (18000) / 0.36e-8 (14500), helicity -9.4e-6 /
  -9.8e-6 / -10.6e-6: indistinguishable, at 0.206 / 0.297 / 0.426 s/step.

### Newton path

- li383 (16,32,32) (2026-09-13): plain float32 floors at F2 1.4e-7 (1.6 s/step). Mixed 4.7e-9 (4.05)
  and float64 4.2e-9 (6.2) agree. Inner tol 1e-6 / 1e-8 / 1e-10: identical trajectories and floor
  (4.6e-9) at 3.1 / 4.05 / 5.0 s/step. Float64 at tol 1e-10 reproduced the mixed floor to the digit
  (2.03e-9 vs 2.34e-9): the floor is the discretisation (current sheets), not the tolerance
  (2026-09-07).
- MINRES inner solve (2026-09-18, li383 + W7-X (16,32,32)): with the parallel penalty the true
  residual falls 0.20 / 0.14 / 0.081 / 0.034 / 0.018 at 50 / 100 / 200 / 400 / 800 it, direction energy
  saturating (200: 78%, 400: 94%). WITHOUT the penalty the residual RISES 0.22 -> 0.32 and the
  direction energy grows without bound (4x the converged value at 800).
- In mixed precision the P-norm estimate converges ~3x slower than in float64. Constant tolerances
  and the Nocedal-Wright sqrt forcing sequence all land on F2 1.53-1.55e-10. None is cheaper than a
  fixed 200 it (the problem never enters the quadratic regime). Warm-start trap: at step 20 MINRES tol
  0.3 / 0.2 stops after ONE iteration at true residual 0.7.
- CG-Steihaug fails on W7-X (2026-09-18): negative-curvature exit at j = 0 on every step from ~40,
  197/200 fallbacks, dt 0.02, stuck at 1.7e-5. LOBPCG at that state: penalised Hessian lambda_min
  -1.8e-3 (bare: four negative, -3.3e-3..-5.5e-4). At the converged MINRES state +7.4e-4. The second
  variation is genuinely indefinite away from equilibrium on W7-X. CG self-locks, MINRES passes.
  Trust-region Newton-CG works at 2x cost and 2x worse residual. rho settles at exactly 2.00 (model
  decrease <F,u>/2 while the induction step gains ~<F,u>).
- Newton-MR nonpositive-curvature exit never fired on li383 or W7-X (results identical to 3 digits).
  Pass loop 3 x 100 it at tol 0.1 = fixed 200 it to 3 digits (2026-09-18).

### Decided against / removed

- Velocity Leray (second projection), li383 (16,32,32) p=3 (2026-09-04): plain float32 without it is
  3.5x faster but not a descent (energy ends +8.9e-7 above start, resid 40-57% higher). float64 and
  mixed identical with and without (float64 2.47 vs 0.95 s/step). Removed 2026-09-05 once every
  solve is refined.
- Warm starts of the Krylov solves: ~1% (quasr44970 8^3 p=3, 1.16 vs 1.17 s/step, trajectory
  identical, 2026-08-25). 14% on W7-X (12,24,12) (0.36 vs 0.41 s/step, 2026-08-28). Kept.
- Midpoint Picard in mixed precision stalls at the float32 rounding of B_mid (defect ~1e-5 of the
  increment), so dt is halved to 1/16 every step. In float64 it converges at dt = 1 in 5 sweeps
  (2026-09-07/11). Fully implicit midpoint diverges at the line-search dt (dt* is 35x the Picard
  limit). Anderson (depth 2-10) and Laplacian preconditioning of the defect (0.87 -> 0.10 in 20 sweeps)
  do not rescue it (2026-09-04).
- Dominated Hodge-solve alternatives (QA): (I - Pi) P_1 (I - Pi^T) + G L_0^-1 M_0 L_0^-1 G^T inside
  MINRES. Chebyshev inner 4-5x the atom's wall at n=24 dbc. PCG inner n=24 dbc 319-374 it at 41-94 s
  vs 1550 it at 7 s. Inner tol 1e-2 stalls. CG on the true L_1 with inner mass PCG: dbc n=16 353-428
  outer, 2500-5800 inner, 146-184 s. Free n=16 fails (6000 outer, residual 3.0, 96k inner, 46 min)
  (2026-09-02).
