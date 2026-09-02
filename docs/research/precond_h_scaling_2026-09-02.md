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
