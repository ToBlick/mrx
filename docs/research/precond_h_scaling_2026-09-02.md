> **Status:** in progress (2026-09-02); the Hodge-split solve is in production for k = 1, 2, the saddle MINRES stays for k = 3 and for the shifted Laplacian
> **Read this for:** the definitions, the solve strategies line by line, and the measured iteration counts / h-scaling of every Hodge solve
> **Do not read for:** the construction of the metric-lumped atom (`preconditioner_technical_note_source.md`) or the `bc_scale` derivation (`s_scale_2026-08-25.md`)

# Hodge-Laplacian solves: operators, strategies, numbers

Branch `vmec-axis-guard`. QA = `data/wout_LandremanPaul2021_QA_lowres.nc`
(nfp 2), toroid = analytic `toroid`, p = 3 unless stated, tol
`sqrt(eps)` = 1.5e-8, maxiter 10000, resolution `(n, 2n, n)`. Probe scripts:
`~/.claude/jobs/81d8eef9/tmp/*.py` (drivers over the public `seq` API).

## 1. Operators

All on extracted DoF vectors, Dirichlet (`dbc`) or natural (`free`) spaces.

| symbol | definition | code |
| --- | --- | --- |
| `M_k` | mass matrix of k-forms | `apply_mass_matrix(v, k)` |
| `D_k` | incidence (exterior derivative) k -> k+1; `D_k D_{k-1} = 0` | `apply_incidence_matrix(v, k)`; `D_0 = G`, `D_1 = C`, `D_2 = D` |
| `S_k` | strong stiffness `D_k^T M_{k+1} D_k`; kernel = exact forms `range(D_{k-1})` + harmonic forms | `apply_stiffness(v, k)` |
| `L_k` | Hodge Laplacian `S_k + M_k D_{k-1} M_{k-1}^{-1} D_{k-1}^T M_k`; `D_{k-1}^T S_k = 0`, `L_k D_{k-1} = D_{k-1} L_{k-1}` | `apply_laplacian(v, k)` (exact, nested mass solve; diagnostics only) |
| `P_k` | metric-lumped Laplacian atom, `~ L_k^{-1}` | `apply_laplacian_preconditioner(v, k, kind='metric_lumping')` |
| `W_k` | metric-lumped mass atom, `~ M_k^{-1}` | `apply_mass_matrix_preconditioner(v, k, kind='metric_lumping')` |
| `L^_k` | `S_k + M_k D_{k-1} W_{k-1} D_{k-1}^T M_k`, SPD (up to harmonic forms) | `_hat_solve` |
| `h_k` | harmonic forms, stored on the bundle; deflated from every solve | `nullspace(k, dbc)` |

Betti `(1, 1, 0, 0)`: harmonic forms exist at `(0, free)`, `(1, free)`,
`(2, dbc)`, `(3, dbc)` only.

`any-W identity`: for `L^_k x^ = b` and ANY SPD `W`, the exact-orthogonal
part of `x^` equals that of `L_k^{-1} b`; only the exact part depends on
`W`. (Project onto exact forms: `D_{k-1}^T L^_k x^ = S_{k-1} W S_{k-1} f^ =
D_{k-1}^T b`, so `W S_{k-1} f^ = S_{k-1}^{-1} D_{k-1}^T b` is W-independent,
and the complement equation follows.)

## 2. Solves

### 2.1 `L_0^{-1}` (unchanged)

PCG on `S_0` with `P_0`, constants deflated (free).

### 2.2 `L_k^{-1}`, k = 1, 2: Hodge split (`apply_inverse_laplacian_hodge`)

`solve(b, j)` = PCG on `L^_j` with `P_j`, `h_j` deflated (`j = 0`: 2.1).

    g      = solve(D_{k-1}^T b, k-1)                  # exact part of L_k^-1 b is D_{k-1} S_{k-1}^-1 M_{k-1} g
    x_perp = solve(b - M_k D_{k-1} g, k)              # rhs consistent: D_{k-1}^T(.) = D_{k-1}^T b - S_{k-1} g ~ 0
    a      = solve(M_{k-1} g - D_{k-1}^T M_k x_perp, k-1)
    x      = x_perp + D_{k-1} a

- `a` combines the exact part `S_{k-1}^{-1} M_{k-1} g` with the removal of
  the exact content `D_{k-1} S_{k-1}^{-1} D_{k-1}^T M_k x_perp` that the
  `L^_k` solve leaves in `x_perp` (zero for the analytic `x_perp`, which is
  coexact: `D_{k-1}^T D_k^T = 0`).
- The exact part of each intermediate is W-dependent and irrelevant:
  `g`'s enters `x_perp` only through `D_{k-1} g` (k=2: `C` kills gradients)
  and `x_perp`'s is removed by `a`.  At k=1 the `j = 0` solves are the
  deflated scalar PCG (exact kernel, 1-dim).
- `guess` warm-starts the `x_perp` solve.  `info` = that solve's count.
- The weak half `D M^{-1} D^T` never appears; no saddle system, no MINRES,
  no Krylov inside Krylov.

Why `L^_k` and not `S_k`: PCG on the singular `S_k` needs a rhs consistent
to round-off; `b - M_k D_{k-1} g` is consistent only to the tolerance of
the `g`-solve (4.7e-6 relative on QA k=1 free at tol 1.5e-8).  The kernel
component of the residual is then invisible to `S_k` and, once the range
residual falls below it (~it 80), the recurrence breaks: residual 1.8e-5 at
it 80, 1.1 at it 160, 2e9 at it 320 (`drift_qa_k1free.log`).  With `S_k`
in production: k=1 dbc a residual floor of 1e-6, k=1 free / k=2 / k=3 blow
up to 1e14-1e20 (`hodge_bench_qa.log`).

### 2.3 `L_3^{-1}` (unchanged): saddle MINRES

`S_3 = 0`; the split would be two `L^_2` solves (`g = solve(D^T b, 2)`,
`z = solve(M_2 g, 2)`, `x = D z`).  Measured QA n=16: 368 + 403 iterations
vs 694 MINRES, error `|x-w|/|w|` 2.6e-8 vs 5.0e-8, but exact residual
`|L_3 x - b|/|b|` 3.8e-6 vs 8.7e-8: the split stops on the `L^_2` residual,
the exact one carries `M_2^{-1}` amplification of it, and the Leray
projection (the consumer, every relaxation step) needs the exact one.  No
gain, so MINRES stays.

### 2.4 Shifted `(L_k + eps M_k)^{-1}` (unchanged): saddle MINRES

Inverse iteration only (`compute_nullspaces_iterative`,
`estimate_spectral_gap`).

## 3. Harmonic forms (`compute_nullspaces`)

Order: `(3, dbc)`, `(2, dbc)`, `(0, free)`, `(1, free)`.  Seeds are
histopolated constants, hence exactly closed (checked, raises above tol);
a closed seed has no coexact part, so each form costs one solve.

    h_3 = M_3^-1 1                                   # closed form
    s_2 = interpolate((0,0,1), 2, dbc, frame='ref')   # dr^dchi, D s_2 = 0
    h_2 = s_2 - C L_1^-1(dbc) C^T M_2 s_2            # one k=1 dbc solve (no harmonic form there)
    h_0 = 1
    s_1 = interpolate((0,0,1), 1, free, frame='ref')  # d zeta, C s_1 = 0
    h_1 = s_1 - G L_0^-1(free) G^T M_1 s_1           # Leray projection, one k=0 solve

`load(frame='ref')` takes the INTEGRAND (`g^-1 u` at k=1, `g u / J` at
k=2), not primal coefficients; `interpolate(frame='ref')` takes primal
coefficients.  The old `M_k^-1 load((0,0,1))` seeds were the forms with
contravariant `(0,0,1)`, `|curl|/|v| ~ 2`, and needed a `L_2` free (k=1)
or `L_3` dbc Leray (k=2) solve to strip the coexact part: the former ran
out of budget at p=4, n>=16 (Rayleigh ratio of `h_1` up to 4.5e-4, the
Route-A stall of `scripts/analytic_vacuum.py`; commit 0c3aa4d), the latter
is a k=3 solve which under 2.2 deflates against `h_2` itself.

`harmonic_rayleigh(h)/lambda_1` is the squared relative error of a form:
1e-11 for every form on QA now (was up to 4.5e-4).

## 4. Numbers (QA p=3, warm wall time; `res` = exact `L_k` residual in the coefficient 2-norm, `err` = `|x-w|_M/|w|_M`)

Split iterations are given as `x_perp` count (what `info` reports) and,
in brackets, the cumulative `g + x_perp + a` count (`hodge_counts.log`);
the `(k-1)`-level sub-solves are cheap at k=1 (k=0 PCG) and two thirds of
the work at k=2 (two k=1-level solves).

| n | k, BC | saddle MINRES | Hodge split (production for k=1,2) | singular `S_k` (rhs `S_k w`; not viable) |
| --- | --- | --- | --- | --- |
| 16 | 1 dbc | 1022 its, 2.7 s, res 4e-8, err 1.1e-7 | **307 [450]**, 3.1 s, 4e-7, 3.7e-8 | 120, 3.0 s |
| 16 | 1 free | 2584, 7.7 s, 1e-8, 5.1e-5 | **1458 [1707]**, 5.5 s, 1e-8, 9.0e-5 | 118, 4.0 s |
| 16 | 2 dbc | 2555, 8.8 s, 2e-7, 8.1e-8 | **367 [987]**, 4.7 s, 5e-6, 6.7e-8 | -- |
| 16 | 2 free | 7122, 9.6 s, 4e-8, 1.9e-5 | **1636 [4415]**, 9.5 s, 1e-7, 1.7e-5 | -- |
| 16 | 3 dbc | 696, 3.0 s, 6e-8, 4.5e-8 | 399, 3.5 s, 4e-6, 1.9e-8 (not used) | -- |
| 24 | 1 dbc | 1553, 7.1 s, 5e-8, 2.3e-7 | **418 [602]**, 4.1 s, 2e-6, 4.5e-8 | 164, 3.5 s |
| 24 | 1 free | 3889, 26.2 s, 2e-8, 7.1e-5 | **2174 [2481]**, 14.1 s, 5e-8, 2.6e-4 | 164, 5.2 s |
| 24 | 2 dbc | 4556, 35.5 s, 3e-7, 8.1e-8 | **500 [1337]**, 8.6 s, 4e-5, 8.4e-8 | -- |
| 24 | 2 free | **cap 10000**, 35.2 s, 2e-7, 1.6e-4 | **2390 [6485]**, 29.9 s, 5e-7, 9.0e-6 | -- |
| 24 | 3 dbc | 997, 6.3 s, 1e-7, 6.1e-8 | 555, 7.1 s, 2e-5, 1.8e-7 (not used) | -- |

Cumulative gain 1.5-3.4x in iterations, 1-4x in wall time.  The k=1 FREE
solve (1500-2200 iterations, called twice inside every k=2 free solve) is
where the leverage is.

- Tolerance 1.5e-8 on each solver's own preconditioned residual
  (`sqrt(r^T P r)`, ~ the energy norm of the error, relative to the same
  norm of `b`).  Errors of 1e-5-1e-4 on the FREE solves, for BOTH solvers,
  are the conditioning of the free Laplacians after deflating the one
  harmonic form (near-harmonic global modes, eigenvalue ~1/R^2):
  err/res ~ 5000 there, ~1 for Dirichlet.
- The split's exact residual in the coefficient 2-norm is 10-100x looser
  than MINRES's at equal error: the DoF scaling of `M_{k-1}^{-1}` acting
  on the `L^` residual.  In the `M^{-1}` norm (the L2 norm of the residual
  function) they agree, n=16, MINRES vs split: k=1 dbc 2.4e-8 vs 7.2e-8,
  k=1 free 8.7e-8 vs 5.9e-8, k=2 dbc 2.1e-8 vs 2.5e-7, k=2 free 7.5e-8 vs
  5.1e-8.  Report and test residuals in that norm (mass atom as the cheap
  proxy); never the 2-norm.
- Test suite (`test/`, k=1,2 split, k=3 MINRES): 246 passed, 7 skipped.

The gap between the split and the singular route (2.5x dbc, 12x free at
k=1) is the preconditioner re-injecting exact-form components (kernel
fraction of the search direction 0.4-0.9) that on `L^_k` have real,
mismatched eigenvalues.  Two candidate closers, neither singular: the
atom's derivative-axis term as the 1-D round trip `M^D G W_1d G^T M^D`
(the structure of `L^`'s weak half); an exact metric-free projector
`I - G (G^T G)^-1 G^T` (Kronecker sum, fast-diagonalisable, polar rows as
a dense core).

Other arms, all dominated: sandwich `(I-Pi) P_1 (I-Pi^T) + G L_0^-1 M_0
L_0^-1 G^T` in MINRES with Chebyshev inner (m=32: 275 / 409 its, 4-5x the
wall time of the atom at n=24 dbc) or PCG inner (fine to 1e-4, stalls at
1e-2); CG on `L_1` with an inner mass PCG (Simoncini-Szyld) --
`gradsw_qa_ss.log`; any-W on the full `b` (306 / 1465 = the split).

## 5. h-scaling of the atom (`hscale_*.log`, `lanczos_*.log`, `coefpat_*.log`)

Iterations of the saddle MINRES, n = 8 -> 32 (h/4), p=3 = p=4 throughout:

| | k=0 dbc | k=0 free | k=1 dbc | k=1 free | k=2 dbc | k=2 free |
| --- | --- | --- | --- | --- | --- | --- |
| toroid | 16 -> 30 | 20 -> 44 | 91 -> 229 | 130 -> 298 | 114 -> 293 | 128 -> 304 |
| QA | 39 -> 108 | 76 -> 184 | 452 -> 1936 | 1393 -> 7702 | 870 -> 6277 | 3122 -> cap (n=24) |

Three mechanisms, by Lanczos on the pencil `(L_k, P_k)` and coefficient
patterns:

1. **k=0, every geometry, both ends of the spectrum at the axis.**
   `lambda_max ~ n / log n` (cylinder 4.7 / 6.7 / 8.6 at n = 16 / 24 / 32,
   mode 68-99% in the first radial bin): the theta-stiffness weight
   `<g^tt J>` is averaged over r, so the `1/r` at the first ring is
   replaced by `log(1/h)`.  `lambda_min ~ n^-0.85` on the cylinder, an m=0
   mode at the axis.  kappa ~ n^1.7.  Fix shape: per angular mode a banded
   radial solve with each term's own radial weight (not a graded knot
   vector: that buys a constant and spends axis resolution).
2. **k>=1 Dirichlet: exact forms in the bulk.**  MAX modes single-component,
   angularly smooth, 85-100% of their energy in the weak half; the atom
   under-rates them 18x (toroid) to 64x (QA), growing n^0.65 on QA.  Made
   irrelevant by the Hodge split (the weak half is gone).
3. **k>=1 free, QA: smooth gradients at the wall.**  rho = 100-115 on the
   smoothest patterns, lambda_max 500-900 growing n^1.3; `bc_scale` cannot
   touch it (optimum s = 1-3 at every n, worth 25%, free stays 3.4-3.7x
   dbc).  Also made irrelevant by the split (exact forms).

What the split leaves: mechanism 1 (the k=0 solves it calls, and the atom
on the exact-orthogonal complement at k>=1 -- toroid k=1 dbc 91 -> 229 is
that), and the QA cross-component coupling on the complement, unmeasured
in isolation.

## 6. Open

- Close the split-vs-singular gap (section 4).
- Mechanism 1: the per-mode radial atom.
- Switch downstream residual checks (the Leray divergence test included)
  to the `M^{-1}` norm; the 2-norm is the h-fickle one (section 4).
- `OPEN.md` 3.13: recheck `nbc_k1` under 0c3aa4d.
