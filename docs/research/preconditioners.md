# Preconditioners: measurements

What the production preconditioner is and how it is built is in the Sphinx docs. This file keeps
the numbers that decided it and the numbers that killed the alternatives.

Geometry key: cylinder = zero angular metric variation (control). Toroid = analytic, eps = 1/3.
Rot-ellipse = eps 0.33, kappa 1.5, nfp 3. W7-X = nfp 5 fitted map. QA = Landreman-Paul QA wout
(nfp 2). quasr9983 / quasr44970 / hegna = GVEC exports. "n" = n_r. Mesh (n,2n,n) unless stated.
Default p = 3, CG/MINRES tol 1e-10.

**Old harness.** Everything dated <= 2026-08-23 was measured on the surrogate `S_k + D B D^T`
(B = one mass-preconditioner apply) solved by a bespoke CG, and the production saddle lower block
was then silently a per-DoF diagonal (see solvers_precision.md). Relative A/B conclusions and
mechanisms from that era survive. Absolute iteration counts do not describe production.

**Old normalisation.** The boundary-term scale values 0.03-1.0 below (2026-08-21/22) are in the old
derived-alpha (`ibpd`) normalisation. Production is the penalty convention
`alpha_k = <m_k sqrt(g^rr)> / <m_k/J> / h_last` times `PRODUCTION_BC_SCALE = 3.0` (section 3).

## 1. Laplacian atom (metric lumping): construction facts

- Weight the stiffnesses, not the masses: component weight = [mass weight of c] * g^aa, curl-curl
  weight indexed by the THIRD axis. Fixing the axis index: toroid k=1 1180 -> 57 it. Applying the
  component factor twice: invisible at k=0, 5x at k=1 (2026-08-20).
- Honest derivative-spline 1-D stiffness vs the "roundtrip" `M^d G A^-1 G^T M^d` factor (then the
  default, never measured): roundtrip lost all 28 A/B rows, median 4.4x, max 9.8x more iterations
  (W7-X k=1 free 1103 vs 6540). Identical at k=0 as required (2026-08-22, old harness).
- p = 1: the DG-0 jump stand-in must be written on cell values (D-splines are unit-integral,
  conjugate by diag(1/h)). Before the fix the atom was 2.5-6x worse than Jacobi at every k >= 1.
  After: toroid 12^3 k=1 free 395 Jacobi / 98 fixed / 706 before. W7-X 1260 / 588 / 1852.
  1-D generalised eigenvalue ratio jump/F (r / theta / zeta, toroid 8^3): 0.15 / 0.015 / 0.0017
  before, 7.4 / 3.7 / 0.11 after, vs 10.9 / 6.5 / 0.21 for p >= 2 (2026-08-20).
- Using the round-trip F itself as the derivative-axis factor: 5-8x worse (600 vs 79). As a
  Kronecker-sum summand 425 vs 50 (toroid k=1 dbc), 2691 vs 145 (W7-X) (2026-08-20).
- Polar core: `extra_rings = 3` is the knee for the Laplacian (1-2 do nothing). The mass wants 0
  (2026-08-20).
- Harmonic (not arithmetic) averaging of the radial profile makes the polar cut unnecessary: bulk
  free/dbc without the cut arithmetic 33/27, harmonic 24/21. Full grid 73 vs 82 (2026-08).
- Coarse-grid floor of the atom: n >= p + 2, measured across k and both BCs (2026-08-25).
- k=0 atom: 311 -> 74 (dbc), 584 -> 114 (free) iterations vs Jacobi, 0.25 s assembly (2026-08-25).
  In compute_nullspaces (W7-X dbc): 277 -> 45 it, 8.58 -> 1.36 s, 6.15 s assembly, break-even
  after 0.9 solves (2026-08-24).
- Dirichlet rows are bit-identical under any change to the boundary term (hard invariant, caught a
  guard bug that added entries on the periodic axes) (2026-08).

## 2. Natural-BC boundary term: coefficient and mechanism (2026-08-21/22)

### Where the term lives

- Only at k = 1, 2, 3 free. Vanishes identically under Dirichlet (every dbc row identical across all
  variants to +-1 it, e.g. 216/217/217) and at k = 0 (the atom's `w d_r u = 0` with w = J g^rr is
  already the operator's exact natural condition). k=0 free is the best-conditioned case: 43 it vs
  Jacobi 398 (9.2x), n = 12.
- Trace component per degree: k=1 u_r (partner V_0), k=2 u_t, u_z with partner = the OTHER
  tangential component (3-c, from the wedge on the face), k=3 the single component (partner V_2
  c=r). Wrong k=2 partner: toroid 12^3 k=2 free 158 vs 62 it.
- D-spline trace vector e = dLam_r(1) is one-hot at the clamped end for p = 1..4 (second entry O(eps),
  5e-8..1.2e-7 relative, from evaluating at 1 - 1e-8). Lam'(1) of the value basis has two nonzeros.

### Derived coefficient

- alpha = mu_0 <J sqrt(g^rr)> <sqrt(g^rr)> at r = 1, the same J g^rr expression for every k, added as
  a rank-1 `alpha e e^T` to K_r (FD eigenbasis untouched).
- mu_0 = (M_r^logical)^-1[last,last] is metric-free, bit-identical across geometries: 66.73954 (p=2),
  93.88387 (p=3, n=12), 52.26074 (p=3, n=8), 134.4128 (p=5). alpha identical down all four (k,c) rows
  at p = 2, 3, 5 (degree-independent as derived).
- <S><P>/<SP> = 1.0000 toroid, 0.945 rot-ellipse, 0.912 W7-X.
- The derived alpha stiffens the boundary row 8.2x: toroid (8,16,8) alpha = 2.049e3,
  alpha e_last^2 = 4.611e5 vs K_r[-1,-1] = 5.626e4, identical for (k,c) = (1,0),(2,1),(2,2),(3,0).
- The previously shipped `exact` coefficient was 8-14x too small (13.9x toroid, 12.1x W7-X): two
  implicit metric factors (partner weight inside an inverted M_r, w_comp counted twice under diag
  lumping, a factor g^rr = 9.0 / 15.0 / 6.7 on toroid / rot-ell / W7-X), each cancelled by a fitted
  compensation, so the wrong version passed its own sweep. It was worse than no term on rot-ell k=1
  free (1056 vs 636) and k=2 (1063 vs 689).

### Why a scale on alpha at all (old normalisation)

- Against the right control (same atom, alpha = 0), extra_rings 3, scale 1: the term is a 2.6-5.5x
  win where the face weight is effectively scalar (toroid all k: term/no-term 0.35 -> 0.26,
  0.28 -> 0.21, 0.29 -> 0.18 at n = 8 -> 12, k=3 on every geometry 0.27-0.38) and does nothing or
  harms at k=1/2 on shaped geometries (rot-ell 0.86-0.98, W7-X 1.05-1.36, rising with n).
- Dense spec(PL), (6,12,6) p=3 free. cond(PL) / high outliers / low outliers / lambda_min:

  | geometry, k | s = 0 | s = 0.10 | s = 0.30 | s = 1.00 |
  |---|---|---|---|---|
  | cylinder 1 | 486/23/0/0.334 | 41/5/0/0.334 | 23/0/0/0.333 | 19/0/0/0.327 |
  | rot-ell 1 | 3118/56/8/0.066 | 596/42/9/0.064 | 605/34/13/0.056 | 1224/28/27/0.026 |
  | W7-X 1 | 8464/67/18/0.029 | 2698/59/20/0.024 | 3433/52/27/0.018 | 7303/48/42/0.0078 |
  | W7-X 3 | 450/17/0/0.152 | 60/1/0/0.147 | 43/0/0/0.136 | 66/2/0/0.090 |

- cond(PL) has an interior minimum in s = 0.06-0.22 on every shaped case: HIGH outliers (face row
  too soft) fall 56 -> 28 while LOW outliers (row over-stiffened) rise 8 -> 27 as s goes 0 -> 1.
  min eig(P) / min eig(P)|_{s=0} = 1/(1 + r s) to 3% over s in [0.06, 1], r = alpha e^2 / K_r[-1,-1]
  (8.196 toroid, 7.27 fitted rot-ell). The optimum is a kappa-balance point.
- The boundary DOF is decoupled in the Kronecker sum, so a large penalty abandons that row's
  preconditioning (P's inverse on it -> 0) rather than imposing the condition. Hard u.n = 0 by a
  1e4 penalty: 250 it k=1 free (vs 76), 334 k=2, worse than no term.
- Cylinder (zero-coupling control): zero low outliers at every s including s = 1, lambda_min moves
  2% (0.3341 -> 0.3274, vs rot-ell 0.0662 -> 0.0260, 60%), cond monotone to s = 1, iterations flat
  to noise over s = 0.10-2.00 from n = 16. Cost of s = 1 vs each cell's best: cylinder 1.00-1.04,
  toroid k=1/2 1.00, rot-ell 1.43-2.01, W7-X 1.62-2.40 (k=1, n = 8..24), k=3 1.05-1.31. Hence the
  scale compensates dropped within-ring angular / cross-component coupling, not a derivation error.
- Ring-block predictor (depth-d outer rings R): ||A(s) - L[R,R]|| is minimised at s ~ 1 (alpha is the
  best NORM fit), cond(L[R,R], A(s)) at s = 0.55 (cylinder, toroid), 0.22 / 0.15 (rot-ell (6,12,6) /
  (8,16,8)), 0.06 (W7-X, both meshes). Matches the measured iteration optimum within one sweep point
  on all four geometries, no solve needed.
- REFUTED, the DtN/Schur account: the Schur correction removes only 17.6-34.4% of tr(L[R,R]) at
  depth 1 and 0.4-1.1% at depth 4, and picks the same or a larger scale (toroid depth 1: raw 0.55,
  Schur 2.0).
- High outliers at k=1/2 free live on the components with NO boundary term (k=1 tangential,
  k=2 w_r), 4 of 4 cases incl. W7-X (6,12,6) (k=1 cond 4648, 20 high / 53 low, k=2 cond 4751, 16 / 75),
  i.e. exactly the DOFs Dirichlet deletes. Outlier count grows with n_t n_z (rot-ell 42 -> 98 as the
  ring goes 72 -> 128 DOFs, cond 837 -> 2247 from (6,12,6) to (8,16,8)). The demanded angular cutoff
  m95 grows ~n_t/3 (4.11 / 5.23 / 6.81 at n_t = 12 / 16 / 20).
- The best scale drifts weakly with everything: falls with n (W7-X k=1 0.10 at n=8, 0.06 at n=12,
  < 0.03 censored at n >= 16), with p (rot-ell k=1 0.15 / 0.11 / 0.11 / 0.05 at p = 2..5), with
  geometry (n=12: toroid 0.55, rot-ell 0.11, W7-X 0.06). No degree law (1/(2p+1) fails: k=1 and k=3
  argmin disagree ~2x at the same p).
- Minimax over 82 cells (4 geometries, n = 8..32, p = 2..5), worst ratio to each cell's best:
  s = 0.03 1.76, 0.06 1.45, 0.10 1.19, 0.15 1.23, 0.22 1.36, 1.00 2.40. Reconfirmed over 168 cells
  after the mass swap. Too small is bounded (no term: 1.40-3.74x). Too large is unbounded
  (x3: W7-X k=1 free 2998 it, toroid k=1 free 76 at x1, 86 at x3, 250 at x1e4).
- Penalty vs each cell's optimum at the chosen scale: median 1.01 over 96 free cells, 1.00-1.07 for
  n >= 16, worst 1.55 (toroid k=3 n=12 p=2). Every > 1.2 case at p = 2 or n = 8.
- The scalar term captures 66-74% of the Jacobi-to-best iteration gain at zero storage.

## 3. Boundary-penalty scale s on the real solve: PRODUCTION_BC_SCALE = 3.0 (2026-08-24)

Setting: penalty convention above, saddle MINRES with the atom as outer Schur block,
(12,24,12) p=3, tol 1e-10, maxiter 10000, free rows only (s does not enter Dirichlet rows). Three
runs merged on a ratio-sqrt(2) grid 0..512. Overlap cells agree to <= 0.5% (toroid k1 375/376,
W7-X k1 2096/2101, quasr44970 k2 4489/4502).

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

- Every row brackets. Optima span 0.5 (hegna k1) to 16 (quasr9983). All turn around by s = 32.
  s >= 64 pushes converged cells past maxiter. s = 0 is the worst value in 17 of 18 rows.
- Worst-geometry geomean excess over own optimum: s = 2 1.36x, 2.83 1.28x, 3 1.27x, 4 1.21x
  (minimax), 5.66 1.25x, 8 1.38x. A factor-32 spread in argmin costs at most 27%.
- On the 13 rows converging everywhere: geomean 2.16x at s = 0, 1.14x at s = 3, 1.05x at s = 5.7-8.
  Robustness (all 18, stalls charged 10000) flat 35.3k-35.7k over s in [1, 4]. s in [0.25, 2] the only
  zero-stall band (the one stall is hegna k2 free, 9068-10000 over the whole grid).
- Verdict: keep s = 3. Insensitive over ~[2, 8]. Per-geometry calibration retired
  (alpha_exact / alpha_penalty predicted argmin with 1.85x spread vs 9.55x null, but a perfect
  prediction buys <= 27%).
- Derived `product` coefficient (needs a weak inverse) vs the shipped penalty on the fixed real
  solve: tie on toroid and W7-X. quasr9983 penalty worse k1 free 857 vs 926 (+8%), k2 free 778 vs
  873 (+12%), k3 free 225 vs 346 (+54%). Total time penalty/product = 0.751, all build cost (product
  11.6-11.9 s at k=1 free vs penalty 1.1 s). Product at 0.10 and penalty at 2.83 equivalent overall
  (+0.8% total iterations).
- On the metric-lumped atom inside the Hodge solves (QA, n = 8..32, 2026-09-02): optimum s = 1-3 at
  every n, worth only 25%.

## 4. Atom vs point Jacobi (2026-08-22, old harness, 168 cells, n = 8..32, p = 2..5)

- Iteration ratio atom/Jacobi median 0.31 over 120 cells, 0.25 for n >= 24, improving with n:

  | geometry k bc | n=8 | 16 | 24 | 32 |
  |---|---|---|---|---|
  | toroid 1 free | 0.29 | 0.19 | 0.14 | 0.12 |
  | cylinder 1 free | 0.28 | 0.25 | 0.23 | 0.21 |
  | rot-ell 1 free | 0.57 | 0.50 | 0.41 | 0.36 |
  | W7-X 3 free | 0.54 | 0.31 | 0.23 | 0.19 |
  | W7-X 1 free | 0.56 | 0.67 | 0.60 | 0.52 |

- W7-X k=1 free is the hardest cell throughout (n = 32: Jacobi 6021, atom 3102 it).
- In p (n = 12): point Jacobi grows 7.6-12.2x over p = 2..5, the atom 2.3-2.8x (W7-X k=1 free Jacobi
  1332 -> 9827, atom 891 -> 1474).
- Earlier toroid free, Jacobi -> atom, 6^3..16^3: k=0 214 -> 31 ... 460 -> 52, k=1 249 -> 49 ...
  597 -> 94, k=2 174 -> 43 ... 499 -> 75, k=3 84 -> 25 ... 299 -> 42. Atom grows ~n^0.6 (2026-08-20).
- Dirichlet speedup Jacobi/atom, n = 12, extra_rings 3 (k = 0/1/2/3): toroid 7.31 / 6.06 / 6.10 /
  7.44, rot-ell 4.58 / 4.52 / 3.85 / 5.14, W7-X 5.49 / 4.88 / 4.73 / 5.93. k=1 dbc spectrum cond
  12.7-45.7, zero boundary outliers. k=1 free 44 outliers of 894, cond 768, extreme mode on the wall.
- The Jacobi baseline flatters: the modelled weak-term diagonal vs the exact probed diagonal agree
  to 0.8% at k=0 but cost up to 21% extra iterations at k >= 1 on shaped geometries (rot-ell
  0.82-0.94, W7-X 0.79-0.96). After the mass swap that closed form has 22% median / 114% max error
  (2026-08-25).
- Total time, rot-ell n=20 k=1 free: atom 26.3 s vs Jacobi 53.7 s.

## 5. Atom h-scaling inside the Hodge solves (2026-09-02, metric-lumped atom, p = 3 = p = 4)

- Saddle MINRES iterations n = 8 -> 32: toroid k0 dbc 16 -> 30, free 20 -> 44, k1 dbc 91 -> 229, free
  130 -> 298, k2 dbc 114 -> 293, free 128 -> 304 (x1.9-2.6 for h/4, kappa ~ 1/h). QA k0 dbc 39 -> 108,
  free 76 -> 184, k1 dbc 452 -> 1936, free 1393 -> 7702, k2 dbc 870 -> 6277, free 3122 -> maxiter at
  n = 24 (from n = 16 at p = 4). Free costs 3.5x dbc on QA (1.4x toroid) and drifts up with n.
- Three mechanisms (Lanczos on (L_k, P_k)):
  1. k=0, every geometry: lambda_max ~ n/log n (cylinder 4.7 / 6.7 / 8.6 at n = 16/24/32, mode 68-99%
     in the first radial bin: <g^tt J> averaged over r replaces 1/r by log(1/h)). lambda_min ~ n^-0.85
     (m = 0 axis mode). kappa ~ n^1.7.
  2. k >= 1 Dirichlet: exact forms in the bulk, 85-100% of the energy in the weak half, under-rated
     18x (toroid) to 64x (QA), growing n^0.65.
  3. k >= 1 free QA: smooth gradients at the wall, rho = 100-115, lambda_max 500-900 growing n^1.3.
- Mechanisms 2 and 3 are removed by the Hodge split (solvers_precision.md). Mechanism 1 stays.

## 6. Mass preconditioner: metric lumping

- Metric lumping vs raw_kron (CG to 1e-8, p = 3, 2026-08-22): 0.83x iterations at the median,
  0.70-0.77x at k = 1, 2, holding with h, flat in p, equal build (~1.6-2.7 s at p = 3, ~6-9 s at p = 5).
  Only regression ~5% at k = 0 (7-17 it). Point Jacobi 150-750 it (p = 3), up to the 5000 cap at p = 5.
  n = 20: W7-X k1 free 88 -> 58, k2 free 87 -> 59, rot-ell k1 free 47 -> 33, toroid k1 13 -> 11.
  W7-X p=5 n=12 k1 free 115 -> 93. Inside L_k (the mass preconditioner sits in the weak term): 0.91x
  median, better in 12/16 cells, up to 0.79x on Dirichlet rows.
- raw_kron vs metric lumping as the Schur-Jacobi probe backing ((12,24,12) p=3, tol 1e-10, outer
  Jacobi, 2026-08-25, the last possible A/B before raw_kron was deleted, the raw logs were never
  committed): probed diagonals differ 16-45%.
  Converged cells toroid k1 dbc 1136 -> 1059 (-6.8%), k2 free 1302 -> 1195 (-8.2%), k3 free
  461 -> 430 (-6.7%). W7-X k1 dbc 2334 -> 1952 (-16.4%), k2 free 5865 -> 5890 (+0.4%), k3 free
  1076 -> 1051 (-2.3%) (repeat pass 2340 -> 1952, 5862 -> 5896, 1076 -> 1050). Arms agree to
  2e-10..1e-9. Repeat-pass spread 0-0.26% for both arms. k1 free,
  k2 dbc, k3 dbc hit 20000 on both arms: a property of the Jacobi outer block, not of the mass.
- E E^T = diag(C C^T, I) exactly (cross block 0.0), C C^T blocks <= 3x3. Coupled rows 3n_z / 5n_z /
  2n_z / 0 for k = 0/1/2/3. The exact mass diagonal is computable probe-free (agrees with probing to
  4-6e-16) (2026-08-17).
- Mass matvec is O(N p^4) (q = 2p): (8,16,8) matvec vs atom apply p=2 317 vs 44 us, p=3 558 vs 42,
  p=4 1328 vs 48 (p=3 -> 4 jump 2.38x, predicted 2.37x). (12,24,12) k=1 matvec 11961 us vs applies
  10-295 us, us/iteration flat to +-4% across variants: the iteration count is the only currency
  (2026-08-17).

## 7. Shifted-stiffness atom for (M_k + eps S_k) (2026-09-02)

CG on (M_k + eps S_k) x = M_k u, eps = 0.064/n_r^2, tol 1.5e-8, p = 3 dbc. Iterations mass atom /
shifted S atom / joint-J / joint-M 1-D masses:

| case | k=2 | k=1 |
|---|---|---|
| li383 (8,16,8) | 153 / 69 / 100 / 99 | 181 / 74 / 111 / 108 |
| li383 (12,24,12) | 371 / 117 / 203 / 199 | 422 / 128 / 222 / 222 |
| toroid (12,24,12) | 82 / 44 / 87 / 79 | 99 / 49 / 103 / 97 |
| QA (12,24,12) | 618 / 205 / 348 / 335 | 734 / 220 / 424 / 408 |
| W7-X (12,24,12) | 233 / 89 / 151 / 146 | 270 / 84 / 171 / 167 |

- 1.9-3.3x over the mass atom everywhere. Putting J into the 1-D masses is not the limit. Growth
  n_r^1.3 (shifted) vs n_r^2.2 (mass atom).
- The shifted atom loses as eps -> 0 (its implied mass is worse than the mass atom's exact
  diagonal). li383 (8,16,8), both split solves, shifted vs mass atom: eps 1e-6 195 vs 100. 1e-4 150 vs
  145. 1e-3 145 vs 329. 1e-2 309 vs 810. Crossover eps n_r^2 ~ 0.006: velocity smoothing sits 10x
  above, a resistive eta dt (1e-6..1e-4) below (pays up to 2x on ~100 it).
- A standalone separable (M + eps L) atom lost on iterations (no polar-core block).
- Before the split, (M + eps L) with diag(M)^-1 -> metric lumping (W7-X fmm002 8^3 p=3 gamma=1, 300
  steps, 2026-08-25): mu = 1e-3 (eps lambda_max ~0.26) 2.74 -> 2.10 s/step (-23%). Mu = 1e-2 null.
  Controlled A/B over 3000 steps: energy removed equal to 4 digits, |dH|/H per dE 30.09 vs 28.98,
  2.74 vs 1.78 s/step (1.54x).

## 8. Newton system (second variation) preconditioner

State: li383 (16,32,32) p=2 float64, step-5000 descent state, unless dated 2026-09-17/18.

- Condition an exact model inverse would leave (2026-09-07): 12.7 for the harmonic Gauss-Newton model
  ||curl(u x c h)||^2 (Hessian to 3% above the Ritz value ~30) vs 6.8e4 for the Laplacian model vs
  2.7e5 unpreconditioned. The Laplacian model's Rayleigh ratio spans 5 decades (1.7e-6..0.11): the
  |B|^2 field-line anisotropy.
- Harmonic "sandwich" atom P_h = W P_L W^T (Fourier-diagonal parallel symbol around the k=1 Laplacian
  atom): at 300 MINRES it its direction removes 6-8x the energy of the Laplacian atom's
  (DeltaE* 6.9-8.6e-8 vs 1.06e-8).
- Fraction of the exact Newton decrement (1.13e-7) recovered at 300 / 1000 / 4000 MINRES it:
  Laplacian atom 8 / 24 / 83%. Sandwich 58 / 93 / 99%. Factored preconditioner
  (C^+ X^+ S_1^+ ... + eps P_L) 6 / 16 / 59%. True residual in the mass-atom norm decays ~N^-1/2 for
  all (0.34 at 300, 0.075-0.09 at 4000) and says nothing about energy (residual = stiff modes,
  energy = flat modes). Iteration counts to a tolerance are not comparable across preconditioners
  (each measures its own norm). Direction quality is.
- Why mode-diagonal constructions fail on li383 (pitch probe): lumped pitch of h = iota_h/nfp =
  0.16-0.20 (iota_h 0.49-0.59). Smearing 1 - w_tz^2/(w_tt w_zz) = 0.10-0.25 (VMEC theta not straight
  field line). |h^r|/|h| = 0.05-0.08. A resonant direction is a packet of (m, n + k nfp) modes each
  carrying 10-25% of a non-resonant curvature.
- Floors (2026-09-17): the old kappa = 3 floor 3(2pi)^2(h_t^2 + h_z^2) = 12.1-15.4 vs the physical
  lumped strain (diag of S^T S per logical direction) 0.01-0.5, i.e. 25-1000x below: kappa = 3 was a
  regulariser, not a strain model. Theta strain dominates and grows outward (det DPhi in [0.29, 2.3]).
  h's innermost layer has an axis artefact in the rho strain (201), B's does not (0.023).
- The atom mis-models the field-aligned null modes u = f B by ~1e3: <u,Hu>/<u,H_h u> = 0.001-0.05 on
  them (2026-09-17). Remedy (parallel-flow penalty) is in relaxation.md.
- Harmonic profiles: W7-X h^zeta 0.053, floor scale (2pi)^2(h_t^2 + h_z^2) 0.115. li383 h^theta
  0.043-0.058, h^zeta 0.32-0.35, scale ~4 (35x) (2026-09-17).

## 9. Measurement hygiene

- Noise floor of iteration counts (same operator, two arms, 29 cells): mean +0.13%, max 2.4%,
  concentrated at k=1 free (harmonic deflation not bit-reproducible). k=3 <= 0.6%. Two builds of one
  configuration differ by ~1e-14 on ~1.7% of rows (dense polar core) (2026-08-22).
- Preconditioner apply is 0.09-0.20 ms of a 36-60 ms CG iteration (< 0.2%): build cost decides the
  total-time ranking (2026-08-22).
- A substituted preconditioner does not fail, it gets slower: the relaxation loop's Leray (k=3 dbc)
  and helicity (k=1 dbc) solves ran on a substituted Jacobi diagonal (~2.5x iterations), then on
  none, because nothing on that path assembled the atom (2026-08-25).

## 10. Decided against / removed

Boundary term variants (2026-08-21/22, each measured against the scalar term):
- Cross term at the computed rho (0.63 toroid, 0.40 W7-X): P indefinite on every geometry and k (min
  eig -1.36e-1 ... -6.96e-4), 5-15x worse or capped at 40000. Full-rank cross term: SPD but cond 3048
  vs 115 (toroid k=1), 1.35-1.7x worse, 2.6x at n = 20.
- Exact 2-D face weight by quadrature (Woodbury): 1.35-1.9x worse (rot-ell n=20 1283 vs 661).
  Diagonal capacitance diverges.
- Trace pin (evict the trace ring to the dense probe): no-op (76/76, 620 -> 607, 1834 -> 1874), the
  failing modes carry zero u_r. Evicting to a Jacobi diagonal costs 1.5x (76 -> 195). Pinning the
  non-trace components: rot-ell cond 837 -> 291 but low outliers 33 -> 62. Toroid net loss 75 -> 464.
- Tangential penalty: interior optimum per geometry (0 / 10 / 30), damages pure-u_z modes (toroid
  lambda_min 3.8e-2 -> 5.4e-3). Dirichlet-side term: catastrophic at 5% (toroid k=1 dbc 62 -> 382).
  Nitsche consistency term: indefinite (76 -> 293 / 1422 / 2723). Mode-dependent beta: equivalence
  check off by 3.39e3.
- Outer dense rings and the truncated-Fourier coarse space `fm`: win on iterations, lose or barely
  win on total time. k=1 free n=20, build + CG seconds: rot-ell fm3 22.3, scalar term 26.3, Jacobi
  53.7, o1 94.1, o2 141.2. W7-X fm3 84.5, scalar 112-121, o1 164.2, Jacobi 188.5, o2 193.4, scale 1.0
  250.6. fm storage linear in n_dof (20.7 / 51.0 / 102.2 MB at 8700 / 21584 / 43300 dof, vs atom
  0.1 MB). Its additive form cannot cure high outliers (cond 837 -> 617), only the hybrid does
  (cond 104). Outer rings hurt at k = 0 (rot-ell 65 -> 182) and under dbc (W7-X k3 75 -> 183).
- 2-D separable ring atom: matches the dense probe on inner rings (~1.8x cheaper build), fails on
  outer ones (toroid k3 free 65 vs 24, W7-X k2 free 616 vs 200): the wall needs nonlocal radial
  coupling.
- Radial bands (Kronecker sum per band), li383 (12,24,24) p=3, 1 vs 3 bands (k0F/k0D/k1F/k1D/k2F/
  k2D/k3F/k3D): 127/180/2968/830/3594/1012/1188/884 vs 150/88/2840/742/3631/926/1194/941. 10-18% on
  Dirichlet k = 1, 2, nothing on free (2026-09-07).

k=0 atom line (2026-06 .. 2026-08-18, all replaced by metric lumping):
- Separable-FD free-BC outlier is missing RADIAL coupling: exact dense bulk inverse kappa 1.000.
  Separable model inverted exactly rank-1 kappa 2.91 (FD 3.06), rank-2 1.79 vs FD 51 (inversion error
  at rank > 1). Dense radial block per angular mode: toroid 3.06 -> 2.15, rot-ell 6.19 -> 2.91, but
  stalls at ~1e-6 on free BC (constant null space not deflated per mode). Exact on the cylinder.
- Modal-radial atom with core Schur kept: toroid 8^3 22 -> 13, 12^3 32 -> 14. W7-X bulk dbc/free
  61/45 -> 47/34. Never run on the full grid.
- Greville-combined k=0 atom (avg it Jacobi / rank-1 FD / combined): toroid dbc 202/11/42, free
  336/18/59, W7-X dbc 227/45/81, free 414/5664 (stall)/128. Not h-independent.
- rank > 1 CP/FD Hodge atoms: an isolated outlier of smoother*L_0 23-124x below the active spectrum
  -> kappa 1e5, Chebyshev degree explosion, OOM.
- k=0 channel weights are power laws on all geometries (alpha_rr ~ r, alpha_thth ~ 1/r, alpha_zz ~ r).
  Angular spread at the edge cylinder 0%, toroid ~24%, W7-X ~60%. W7-X metric channels zeta-rank ~8
  (toroid 1). Dominant W7-X coupling is rho-theta (~0.49 normalised), not theta-zeta.
- fdbund single-level atom vs baseline (2026-08-13), dbc | free: W7-X (16,32,32) 80 -> 62 |
  117 -> 85. Toroid (24,48,24) 41 -> 29 | 59 -> 36. Rot-ell (16,32,16) 67 -> 68 | 84 -> 73.
- Geometric multigrid (2026-07/08): best 2-level fat-core MG won only free-BC wall (1.05-1.4x), lost
  dbc everywhere, ~2k lines. Point-Jacobi smoother: 47-180 CG it vs 5-15 for the separable atom
  (bulk polar anisotropy g^thth ~ 1/r^2). W7-X (12,24,24): baseline 64/99 it, MG 18-19/23 but
  ~10 ms/it vs ~1, baseline 1.8-2.8x faster wall. lambda_max(S A) ~ 1.7/xi_1 (sqrt(n_el) with
  equal-area grading). Polar C^2 = fat-core ring 1 in lambda_max to all digits (toroid 1.61, cerfon
  2.28, rot-ell 3.55) with 560 vs 664 DoFs.
- Any rebuilt singular Schur needs a positive-part pseudo-inverse: inverting by magnitude with sign
  gives -O(1e3) Rayleigh quotients. A Schur core rebuilt through the fdbund bulk floors W7-X CG at
  ~1e-2 (indefinite core). k=0 deflation dead: zero modes below 0.03 lambda_max.

Mass preconditioners (2026-06 .. 2026-08-17):
- Greedy CP past rank 1 has sign-changing factors -> indefinite surrogate -> NaN on W7-X 12x24x12 dbc
  (k1 p4/p5, k2 p5). NTF rank 2 k2 p5 fails at ranks 2-4 and costs quality on the toroid (~12 it vs
  ~5). Fast diagonalisation is exact for at most two Kronecker terms.
- W7-X weight-tensor ranks: 1/r-type weights rank 1. r-growing weights r-rank 3, theta-rank 8-10.
  theta-zeta cross-section rank 3-7. r is always the low-rank axis. 28% rank-2 residual on the hard
  weights, p-independent.
- pow2 extraction sandwich, toroid p=3: k0 15/13 it (tensor 11/10, Jacobi 538/785), k1 19/17 (tensor
  13/12, Jacobi 538/608). E^T instead of E^+ drifts (37/39).
- Greville-collocated mass (max CG it k0..k3, p = 1..5, free): cylinder 8/9/9/8, toroid 12/14/14/10,
  W7-X 27/136/96/18. ~40-50x fewer than Jacobi.
- inner Schur: halves iterations, ~3x wall. Block-Chebyshev polish: ~10x fewer iterations, ~10x
  slower wall (each step ~6 mass matvecs). Lumped block-SGS for the k=1/2 coupling: 291-350 it vs
  75-80 baseline.

Auxiliary space (HX) and Chebyshev (2026-06, shelved):
- k=1 grad-div toroid (6,12,4): Jacobi ~386 it / ~290 ms vs projected P_A + P_B ~96 it / ~117 ms. k=2
  h-flat 33-34 it but ~3.3x slower wall. k=3 Jacobi wins (186 it / 0.47 s vs 280 it / 10.4 s).
- Chebyshev-8 on the approximate Schur beats the tensor atom on iterations (67 vs 96), loses ~3x
  wall. Whole-operator Chebyshev is dead on singular free k=0/k=1.
- k=1 saddle P_A + P_B on W7-X (2026-08-13): dense P S spectrum dbc kappa_eff 7.6e6, free 7.9e7 with
  3397/4176 modes < 1e-4 max, a continuum (not deflatable). The "stall" at ~1e-5 was an honest crawl.

Newton preconditioner variants (li383 (16,32,32), 2026-09-07):
- Coarse space of the k lowest Lanczos pairs: MINRES to 0.1 = 269/266/266/264 it for k = 0/10/30/100.
  The spread is at the STIFF end (sandwich mis-scales stiff modes up to 3 decades).
- Symbol variants (bundled weights, powers 1/2, 3/4): spread 1.1e3-9.6e3, none beats the sandwich.
- Factored preconditioner: best model spread (1e2-3.7e2, 243 it) but worst direction on the true
  system (1.2e-8 vs 6.9-8.6e-8). Transposes exact (asymmetry 5e-16), so it is a property.
- Levenberg-Marquardt shift lambda M_2 (shift 39 = 1e-3 of the top): 6 it, leaves resid 4e-4, above
  the descent. The shift keeps flat modes gradient-like.
