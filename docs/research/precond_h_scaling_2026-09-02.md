> **Status:** open, in progress (2026-09-02)
> **Read this for:** how the iteration count of every production Hodge solve grows with resolution, which of the atom's approximations is responsible, and what `bc_scale` can and cannot buy back
> **Do not read for:** the construction of the atom (`preconditioner_technical_note_source.md`) or why `s = 3.0` (`s_scale_2026-08-25.md`); the harmonic-form bug that started this is recorded in section 1 and closed

# The metric-lumped atom loses h-independence

Branch `vmec-axis-guard` (worktree of `static-dynamic-refactor`). All runs
via `slurm/run.sh`, QA = `data/wout_LandremanPaul2021_QA_lowres.nc`
(nfp=2), toroid = the analytic `toroid` map, tol `sqrt(eps)` = 1.5e-8
(float64), `maxiter = 10000`, production bundle
(`seq.build_preconditioners()`: metric-lumped mass and Laplacian atoms for
every `(k, BC)`, `bc_scale = 3.0`). "cap" = the solve returned `info =
+10000`, not converged.

Probe scripts live in the job scratch of this session
(`~/.claude/jobs/81d8eef9/tmp/{bc_scaling_diag,hscale_probe,precond_probe}.py`);
they are 100-line drivers over the public `seq` API and are reproduced in
section 6 in enough detail to rewrite.

## 1. How it surfaced: the k=1 free harmonic form (closed, commit 0c3aa4d)

`scripts/analytic_vacuum.py` Route A (H = grad psi + alpha h_1, QA, coil
field) stalled at p=4: relerr 4.3e-5 -> 8.5e-5 -> 2.4e-4 for n_el = 20, 24,
28 while Route C (B = curl A, no harmonic term) kept O(h^4). The
`harm_ratio` column -- `harmonic_rayleigh(h_1) / lambda_1`, the squared
relative error of the stored harmonic form -- rose in lockstep: 5.9e-8,
5.0e-6, 6.3e-5, 4.5e-4 (n = 24 .. 36).

Instrumenting `compute_nullspaces` step by step
(`outputs/analytic_vacuum_pscan/2026-09-02/07-47-49/h1_chain_diag.log`):

| step (QA p=4) | n=24: `|C v|/|v|`, rq/lambda_1 | n=32 |
| --- | --- | --- |
| seed `M_1^-1 load((0,0,1), frame='ref')` | 2.04, 1.21 | 2.04, 1.20 |
| after Leray (L_0 NBC) | 2.05, 1.21 | 2.05, 1.21 |
| L_2 FREE solve | **cap** | **cap** |
| after coexact removal | 4.1e-3, 4.9e-6 | 4.0e-2, 4.5e-4 |

The seed was not `d zeta`: `load(frame='ref')` takes the *integrand*
(`g^-1 u` at k=1, `g u / J` at k=2), not the primal coefficients -- its
docstring said otherwise -- so `M_1^-1 load` returned the form with
CONTRAVARIANT (0,0,1), `|curl|/|v| ~ 2` at every resolution. All of that
curl had to be removed by the k=2 FREE saddle solve, which is the most
expensive solve in the code (section 3) and exhausts its budget from n=16 at
p=4. Its leftover curl was the Route-A error.

Fix: seed with `seq.interpolate((0,0,1), 1, frame='ref')` -- histopolation
takes primal components and is a direct solve, so the seed is closed to
round-off (checked, raises above `tol`) -- and the Leray projection alone is
the harmonic form. No L_2 solve. Rayleigh ratio 1e-11 at every rung; Route A
p=4 relerr 3.6e-5 / 1.7e-5 / 9.5e-6 at n_el 20/24/28, order 4.27 (was
3.21). The k=2 Dirichlet form keeps its L_1 dbc solve: it is the Hodge star
of `d zeta`, metric-weighted, never in `V^2`. Tests 36/36.

What it left behind is the question of this note: the k=2 free solve should
not have been anywhere near 10000 iterations.

## 2. BC and k as multipliers (QA, p=4)

`bc_scaling_diag.py`: same rhs recipe (`S_k w` for a projected smooth `w`,
harmonic forms deflated), MINRES iterations.
`outputs/analytic_vacuum_pscan/2026-09-02/08-49-41/bc_scaling_diag.log`.

| n | k=1 dbc | k=1 free | k=2 dbc | k=2 free |
| --- | --- | --- | --- | --- |
| 12 | 972 | 3099 | 2173 | 7255 |
| 16 | 1390 | 4708 | 3485 | cap (res 1.8e-7) |
| 20 | 1774 | 6274 | 4849 | cap (6.5e-6) |
| 24 | 2113 | 7853 | 6227 | cap (6.8e-5) |

Free costs 3.3-3.7x dbc at every n; k=2 costs ~2.5x k=1 at the same BC.
Both are multipliers on a count that itself grows ~linearly in n. Once
capped, the residual the solve stops at degrades fast (1.8e-7 -> 6.8e-5
over n = 16 -> 24), which is the staircase of section 1.

## 3. Iterations vs n, all solves, both geometries (`hscale_probe.py`)

`outputs/hscale/2026-09-02/10-11-{16,22,28,34}/hscale_{tor,qa}_p{3,4}.log`.
Counts at n = 8, 12, 16, 24, 32 (`(n, 2n, n)`); "growth" = count(32) /
count(8), i.e. over h/4.

**Toroid**

| p | k=0 dbc | k=0 free | k=1 dbc | k=1 free | k=2 dbc | k=2 free |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 16, 20, 22, 26, 30 | 20, 26, 30, 38, 44 | 91, 119, 146, 189, 229 | 130, 172, 206, 259, 298 | 114, 154, 187, 246, 293 | 128, 159, 190, 247, 304 |
| 4 | 18, 21, 25, 30, 35 | 21, 28, 34, 42, 49 | 112, 149, 184, 249, 300 | 158, 215, 260, 336, 397 | 146, 198, 239, 311, 375 | 161, 209, 252, 327, 404 |
| growth | x1.9 | x2.3 | x2.6 | x2.4 | x2.6 | x2.5 |

**QA**

| p | k=0 dbc | k=0 free | k=1 dbc | k=1 free | k=2 dbc | k=2 free |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 39, 55, 70, 91, 108 | 76, 100, 122, 157, 184 | 452, 735, 1019, 1547, 1936 | 1393, 2505, 3583, 5705, 7702 | 870, 1624, 2554, 4563, 6277 | 3122, 6088, 8801, cap, cap |
| 4 | 42, 61, 77, 103, 123 | 79, 108, 135, 175, 207 | 574, 972, 1384, 2116, 2667 | 1642, 3107, 4706, 7861, cap | 1120, 2174, 3489, 6228, 8466 | 3581, 7265, cap, cap, cap |
| growth | x2.8 | x2.5 | x4.4 | >=x5.5 | x7.4 | -- |

Reading, three separate facts:

1. **Nothing is h-independent, not even k=0 on the toroid.** Every toroid
   column grows x1.9-2.6 for h/4: iterations ~ n^0.5-0.7, kappa ~ 1/h. That
   is the signature of a two-block non-overlapping split with no coarse
   space -- the uncoupled bulk / polar-core Jacobi of the atom -- common to
   every `(k, BC)` and both geometries. At 30-50 iterations for k=0 it was
   never visible.
2. **k>=1 on the toroid is sound**: same slope as k=0, base ~7x. The
   per-component Kronecker sum with the derivative-axis stiffness for the
   weak half is a Hodge-Laplacian equivalent when the metric is diagonal
   and separable.
3. **QA breaks k>=1 only.** k=0 on QA is 3x the toroid at the same slope --
   the axis-averaged weights cost a constant, as they should. k=1 is 5-25x
   the toroid AND grows ~n (kappa ~ h^-2); k=2 grows ~n^1.45 (kappa ~
   h^-3). p is irrelevant (p=3 = p=4 throughout). Geometry-dependent and
   resolution-dependent together points at what the atom drops that is zero
   on the toroid and large on QA: the off-diagonal metric couplings (the
   theta-zeta block) and with them the cross-component curl couplings. Why
   dropping a bounded coupling costs a *growing* constant rather than a
   fixed one is the open mechanism question.

The free/dbc ratio is flat ~1.4 on the toroid but drifts on QA (k=1 p=4:
2.9, 3.2, 3.4, 3.7, >=3.8) -- a scalar face weight where the true one varies
around the wall, and possibly an h-scaling of the rank-one term that is
slightly off on QA. `s = 3.0` was settled at (12,24,12) p=3 only
(`s_scale_2026-08-25.md`); its n-dependence was never measured.

## 4. In flight (2026-09-02, jobs 17332xxx, logs under `outputs/hscale/2026-09-02/10-54-*/`)

- `precond_probe.py --mode bcscale`: k=1,2 free iterations vs
  `bc_scale in {0, 0.3, 1, 3, 10, 30}` at n = 12, 16, 24 (QA) and 12, 24
  (toroid), p=3, nullspaces transplanted between bundles. The term is
  `s * (1/h_last) * <g^rr J>_face`: by design the `1/h` carries all the
  h-dependence and the face average all the geometry (TB, 2026-09-02). So
  the sharp reading is: optimal `s` independent of n -> the factorisation
  holds and only the amplitude is in question; optimal `s` drifting with n
  -> `1/h` is not the true h-scaling of the boundary block on QA, where a
  scalar face average can be right in norm and still wrong as a
  preconditioner (the alpha-vs-P distinction of the natural-BC note).

Multigrid is explicitly NOT on the table unless forced (TB): the data does
not point there -- the boundary term cannot be the QA k>=1 growth (dbc has
no boundary term and grows the same way), and the toroid baseline (kappa ~
1/h, 30 iterations at k=0) is a coarse-space effect of multigrid-free
size. Candidates are local changes inside the atom (section 5).
- `precond_probe.py --mode lanczos`: Lanczos on the pencil `(L_k, P)` with
  `L_k` applied exactly (nested mass solve; a once-only diagnostic, never
  inside a solve), 60 steps, full reorthogonalisation in the L-inner
  product, harmonic forms deflated. k = 0, 1, 2, both BCs, toroid and QA,
  n = 16 and 24. Reports the extreme Ritz values of `P^-1 L_k` (which end
  runs away with n) and, for the two extreme Ritz vectors, the radial
  energy profile in 10 bins axis -> wall and the dominant `(m, n)` angular
  modes at rho = 0.5 (which coupling, where).

## 4b. Results of the section-4 probes (2026-09-02, all p=3)

**`bc_scale` (`outputs/hscale/2026-09-02/10-54-4*/bcscale_*.log`).** QA
k=1 free vs `s in {0, 0.3, 1, 3, 10, 30}`: n=12: 3365/2831/2607/**2515**/
2696/3604; n=16: 4736/3914/3650/**3578**/4018/5564; n=24: 7109/5792/
**5605**/5709/6816/9760 (dbc: 733/1020/1546). Optimum at s = 1-3 at every
n -> the factorisation `s / h <g^rr J>` holds (no drift) and the amplitude
is right, but the whole s-dependence is worth ~25% and the best s still
leaves free at 3.4-3.7x dbc. On the toroid s ~ 10 brings free to dbc
(k=1: 149 vs 118; k=2: 142 vs 154). Closed: `s` is not the lever.

**Lanczos on `(L_k, P)`** (`lanczos_{tor,qa,cyl}*.log`), 60-80 steps,
n=16 -> 24 (cylinder to 32):

| | toroid k=0 | cylinder k=0 | QA k=0 | QA k=1 dbc | QA k=1 free | QA k=2 free |
| --- | --- | --- | --- | --- | --- | --- |
| lambda_max | 5.07 -> 7.26 | 4.68 -> 6.69 -> 8.55 | 9.94 -> 14.6 | 64 -> 83 | 503 -> 863 | 522 -> 894 |
| lambda_min | 0.28 -> 0.25 | 0.28 -> 0.19 -> 0.15 | 0.12 -> 0.09 | 0.03 -> 0.04 | 0.11 -> 0.16 | 0.18 -> 0.24 |
| MAX mode | axis (97%) | axis (99%) | axis (80-97%) | bulk, single comp | **wall** | **wall** |
| MIN mode | mid-radius | **axis** (m=0) | mid-radius | global (0,±3) | global (0,±2) | wall-ish |

Three mechanisms, each with a location:

1. **k=0: the axis, both ends.** `lambda_max ~ n / log n` (predicted from
   the r-averaged `g^tt J ~ 1/r`: 5.8 / 7.5 / 9.2 for n = 16 / 24 / 32,
   measured on the cylinder 4.7 / 6.7 / 8.6, mode 68-99% in the first
   radial bin). On the cylinder -- where 1/r is the ONLY non-constant
   weight -- lambda_min ALSO falls, ~ n^-0.85, with an m=0 mode 53-95% in
   the first two bins. So kappa(k=0) ~ n^1.7 there, all of it the axis
   region: high-m modes under-stiffened (1/r averaged away), smooth m=0
   modes over-stiffened (the r-profiles of the other terms + the uncoupled
   core). Confirmed: the core problem IS the r-average.
2. **k>=1 Dirichlet: gradients (k=1) / curls (k=2) in the bulk.** The MAX
   Ritz vectors are single-component with 85-100% of their L-energy in the
   WEAK half `M_k D M_{k-1}^-1 D^T M_k`; k=1 `u_r` (a gradient), k=2
   `u_theta` (a curl). The atom under-rates them 18x on the toroid, 64x on
   QA, growing ~n^0.65 on QA. Single-pattern families
   (`coefpat_*.log`) reach rho = 5 (toroid) / 13 (QA: `u_theta`, smooth r,
   m=4 -- an angular gradient); the eigenvector is a combination. NOT the
   theta-zeta coupling on high angular frequencies (the modes are angularly
   smooth) and NOT a wrong weight power (the a=c weight is `(g^cc)^2 J` as
   intended).
3. **k>=1 free, QA only: smooth gradients at the wall.** rho = 100-115 on
   the SMOOTHEST patterns (`u_r` or `u_theta`, radially smooth, angularly
   constant, weak share 1.0), lambda_max 500-900 growing ~n^1.3. The
   toroid has the same in miniature (rho 10.6, s ~ 10 would fix it); on QA
   a scalar face weight cannot represent it.

Every lambda_max mechanism at k=1 is an exact form. Hence:

## 4c. The gradient sandwich (TB's proposal, 2026-09-02)

`L_1 G = G L_0`, so on `x = G f` only the weak half acts and equals
`M_1 G M_0^-1 L_0 f`. Therefore

    Q = G L_0^-1 M_0 L_0^-1 G^T      satisfies    Q L_1 (G f) = G f,

exact on gradients with two scalar solves (note the `M_0` in the middle).
It must be a SANDWICH, not a sum: the atom's defect on these modes is
lambda_max (it over-applies), so it has to be kept off the gradient
subspace:

    P_1 = (I - Pi) P_atom (I - Pi^T) + G L_0^-1 M_0 L_0^-1 G^T,
    Pi  = G L_0^-1 G^T M_1   (the M_1-orthogonal projector onto gradients).

This is the k=1 "inner grad-complement sandwich" of
`k2-laplacian-preconditioner` and plausibly why the additive `P_A + P_B`
arms of 2026-08-13 stalled on W7-X (additive, and a nonlinear inner solve
inside MINRES -- next point).

**Krylov-in-Krylov.** Flexible MINRES does not exist in a usable form (the
short recurrence needs one fixed SPD preconditioner; variable ones
stagnate). FGMRES is the fallback (long recurrence). The clean route keeps
MINRES: replace `L_0^-1` by a FIXED-degree Chebyshev polynomial in
`P_0 L_0` (the k=0 atom as inner preconditioner) -- linear and symmetric,
so `Q = G C M_0 C G^T` is SPSD by construction and there is no Krylov
inside. NOT "m steps of PCG" (rhs-dependent coefficients = nonlinear =
MINRES stall). Chebyshev needs the spectral bounds of `P_0 L_0`, a few
Lanczos steps at build time (QA k=0: [0.12, 9.9] at n=16; lambda_max ~
n/log n so per resolution). sqrt(kappa) ~ 10-13 -> ~x0.83 per degree ->
degree ~15 for preconditioner quality. Cost per `P_1` apply: four scalar
solves (two in the sandwich, two in `Q`) ~ 60 k=0 atom applies + 60 `S_0`
matvecs; per-iteration cost x10-20, so k=1 must drop from 2000-8000 to
below ~200-400 on QA to win. What the sandwich leaves: the k=0 axis
behaviour, inherited by the inner solve (Chebyshev degree ~ n^0.85 for
fixed quality -- a cost drift, removed independently by the per-mode
dense-radial k=0 fix), and the non-gradient (`S_1`-equivalent) part of the
atom, unmeasured so far. k=2's mirror needs `L_1^-1` -- one level at a
time.

Prototype as a probe (no production change): `solve_saddle_point_minres`
takes `precond_upper` as a callable; `G`, `G^T`, `M_0`, `S_0`, the k=0
atom and the Lanczos bounds all exist. Measure QA k=1 dbc/free at n=16, 24
against 1019/3583 and 1547/5705.

## 5. What would follow

Decided by section 4, not before:

- lambda_max growing, modes high-frequency in theta-zeta at mid-radius ->
  the dropped theta-zeta metric block; candidate: keep a cross term
  (rank-2 in the coupled plane, cf. `metric_weight_separability_rule`).
- lambda_min shrinking, modes localised at the axis -> the bulk/core split;
  candidate: couple the core (a Schur complement or overlap), or a coarse
  correction.
- lambda_min shrinking, modes smooth and global at k>=1 only -> the atom
  mishandles the near-kernel of curl-curl on a non-orthogonal metric
  (auxiliary-space territory, `hiptmair_xu_preconditioner.md`).
- extreme modes at the wall, free only -> the face term; `bc_scale`
  sweep says whether a scalar can fix it.

## 6. Reproduction

Every probe: `build_sequence(geometry, ns, p)` ->
`compute_nullspaces(seq, ops, gap_sweeps=0)` -> for `(k, dbc)`: `w =
M_k^-1 load(f_smooth)`, `rhs = S_k w`, `apply_inverse_laplacian(rhs, k,
dirichlet=dbc, return_info=True)`; `info` is the signed iteration count.
`bc_scale` variants: `ops = seq.build_preconditioners(bc_scale=s)`, then
`init_nullspaces` + `_set_null` from the reference bundle (the harmonic
forms do not depend on `s`: it vanishes at k=0 and under Dirichlet).
Lanczos: `T = P L` is self-adjoint in `<x, y>_L = x^T L y`; run Lanczos in
that inner product with `P = apply_laplacian_preconditioner(.,
kind='metric_lumping')` and `L = apply_laplacian` (exact); Ritz vectors
evaluated with `DiscreteFunction` on a (40, 32, 32) logical grid.
