> **SUPERSEDED 2026-08-22** by `natural_bc_coefficient_handoff.md`, which
> continues this and carries the derivation, the refutations and all the
> measurements. Kept for the day-by-day record.

# Handoff 2026-08-19 — the natural BC in the tensor block-Jacobi atom

Continues `handoff_2026-08-18_preconditioner_pivot.md` and the reasoning in
`tensor_preconditioners.md`. Everything here is in `mrx/experimental/`;
production defaults (raw_kron masses, closed-form Jacobi Laplacians) are
untouched.

Commits: `a7e1aba` (natural BC + 2-D ring atom), `fb9619b` (k=2 partner fix).

## The one-line result

The free-BC gap in the tensor block-Jacobi Laplacian atom was the weak block's
natural boundary condition, and it has a closed form. `bc_entry="exact"` is now
the default, costs **nothing** (no probes, one scalar per component), and gives
**5.2-9.3x over Jacobi on all eight toroid cases** — beating the dense
outer-ring variant `o2` in 6 of 8 while building in half the time.

## What the term is

`W_k` is a weak codifferential, `<delta_h u, tau> = <u, d tau>`. At free BC the
test function `tau` is unconstrained, so integration by parts leaves a surface
term: `u.n` at k=1, `w x n` at k=2, the normal trace at k=3, nothing at k=0.
That is a genuine natural condition (`u.n = 0` etc.), enforced not by removing a
DOF but by a mesh-dependent penalty — the boundary distribution is projected
into `V_{k-1}`, and its norm grows like `1/h`.

Discretely the term is `E^T M_{k-1}^{-1} E`. `E` factors as a Kronecker product
(radial `dLam_i(1) Lam_j(1)`, angular `M_t (x) M_z`); when the V_k component's
angular bases match those of the V_{k-1} component it pairs with, the angular
masses cancel against the same factors in `M_{k-1}^{-1}` and ONE scalar
survives. With `Lam(1) = e_last` for a clamped spline:

    alpha = <w_comp * sqrt(g^rr)>_{theta,zeta at r=1}      # face measure
          * (M_r^{(k-1)})^{-1}[last, last]                 # the 1/h
    K_r  += alpha * e e^T ,      e = dLam_r(1)

`w_comp` is the component's own mass weight. Because this reuses the SAME
`M_t`, `M_z` as the first Kronecker term, it merges into `K_r` as a rank-one
update: `(K_r + alpha e e^T, M_r)` is still one generalized eigenproblem, so
fast diagonalisation, cost and storage are all unchanged.

### Which component, and which partner

The trace lives exactly where the component's RADIAL axis is a derivative axis.
The partner in V_{k-1} is NOT always the same index — at k=2 the cross product
in `int (w x n).tau` swaps the tangential components:

| k | component with a trace | pairs with |
| --- | --- | --- |
| 0 | none (`W_0 = 0`) | — |
| 1 | `c = r` (normal) | `V_0`, the only component |
| 2 | `c = theta, zeta` (tangential) | `V_1` at `3 - c` — the OTHER one |
| 3 | the single component | `V_2` at `c = r` |

The bases confirm the swap: `V_2` at `c=theta` has angular bases (theta primal,
zeta derivative), which are `V_1`'s at `c=zeta`. Getting this wrong is not a
perturbation — the two weights differ by `(R/a)^2` on a toroid, and fixing it
took toroid 12^3 k=2 free from 158 to 62 iterations.

### Three corrections, two of opposite sign

The previous `direct` weight was wrong in three ways, and only all three
together work:

1. **Scale.** The bare surface integral misses the `M_{k-1}^{-1}` amplification.
   Measured directly: hand-set multiplier best at 3 / 4.5 / 6.5 for
   `n_r = 6 / 8 / 12` (toroid, k=1 free, angular fixed) — i.e. `~0.55/h`. With
   the inverse-mass entry computed, the residual sweep peaks at x1 and the
   h-dependence is gone.
2. **Measure.** `g^rr` should be `sqrt(g^rr)`: the surface element on `r=const`
   is `sqrt(g_tt g_zz - g_tz^2) = J sqrt(g^rr)`, exact whenever
   `g_rt = g_rz = 0` and needing NO assumption on `g_tz` (the large off-diagonal
   on W7-X). ALONE this is a regression (toroid k=1 free 107 vs 80) because it
   divides by ~3 on that geometry — the two corrections had been partially
   cancelling in the old weight.
3. **Axis.** The fallback branch lacked an `a == 0` guard and was adding entries
   on the PERIODIC theta/zeta axes, which have no boundary. Cost k=3 free 97
   against 87 for no correction at all, and perturbed the Dirichlet cases where
   the term must vanish identically.

**Regression check: every Dirichlet row must be identical across
nobc/direct/exact.** It is, now.

## Numbers (toroid, p=3, unshifted, tol 1e-10)

| | | jacobi | exact | o2 | nobc |
| --- | --- | --- | --- | --- | --- |
| 12x24x12 | k=0 free | 399 | **43** | 90 | 43 |
| | k=0 dbc | 234 | **32** | 33 | 32 |
| | k=1 free | 444 | **76** | 130 | 292 |
| | k=1 dbc | 376 | **62** | 74 | 62 |
| | k=2 free | 346 | **62** | 75 | 292 |
| | k=2 dbc | 470 | **77** | 116 | 77 |
| | k=3 free | 188 | 36 | **33** | 186 |
| | k=3 dbc | 306 | **41** | 79 | 41 |
| 8x16x8 | k=1 free | 315 | **57** | 77 | 172 |
| | k=2 free | 212 | **45** | 49 | 166 |
| | k=3 free | 113 | 29 | **24** | 98 |

W7-X 8x16x8 free: k=2 705 -> 577, k=3 164 -> 94. **k=1 free REGRESSES**
(586 -> 605), the known-stalling case and the one where the orthogonal-metric
assumption behind both the weight formula and the face measure is worst; only
`o2` (223) helps there.

## Refuted

* **Hard `u.n = 0` as a replacement.** Penalty x1e4 gives 250 iterations at k=1
  free and 334 at k=2 — worse than no term at all. The atom wants the finite
  penalty, not an eliminated DOF. So "use the dbc scalar Laplacian on that
  component" is not the fix, even though the underlying BC reading is right.
* **A rank-1 fit of the face weight.** It would break fast diagonalisation:
  `e e^T (x) Mt~ (x) Mz~` with `Mt~ != Mt` is not a summand of the Kronecker
  sum, and `V_t^T Mt~ V_t` is dense in the shared eigenbasis. Moot anyway — the
  partner fix closed the k=2 gap that motivated it.
* **`exact` + `o2` is exactly `o2`.** The dense outer ring already cuts those
  rows from the atom's radial window, so the two are alternatives, not additive.

## 2-D ring atom (`core_mode="atom2d"`)

Restricting the Kronecker sum to one radial index leaves
`K_r[i,i] M_t (x) M_z + M_r[i,i] (K_t (x) M_z + M_t (x) K_z)`, inverted by two
small eigendecompositions, with ring-LOCAL angular profiles (the radial
direction collapsed by that ring's own basis, not a global mean).

* **inner rings: yes.** Matches the dense probe at ~1.8x less build; the
  iteration gap shrinks with resolution (1.3-1.7x at 8^3, 1.1x at 12^3) while
  the saving grows — the probe costs `rings * n_t * n_z` operator applies, the
  atom does not.
* **outer rings: no.** Toroid k=3 free 65 vs 24, W7-X k=2 free 616 vs 200, often
  worse than leaving the rows in the bulk atom. The outer ring's value is
  nonlocal radial coupling (Steklov/DtN); no separable factor carries it.

So `core_mode="atom2d"` means dense outer + separable inner.

## Open

1. **W7-X k=1 free.** Untouched by all of this (1052-1214 at 12^3); `o2` (389)
   remains the only thing that works, at 2x the build. This is the same case
   flagged in `greville-k1-hx-status`. The orthogonal-metric assumption is the
   prime suspect — W7-X's `g_tz` is its largest off-diagonal and the weight
   formula `w(k,c,a) = [mass weight] * g^{aa}` assumes it away.
2. **Woodbury with the corrected weight.** `bc_entry="woodbury"` was measured as
   adding nothing, but that used the old, wrong `alpha`. `U^T A^-1 U` IS
   diagonal in the angular eigenbasis, so it is affordable; worth one re-test
   now that the weight is right.
3. **k=2 with a non-scalar face weight** — only reachable through Woodbury (see
   above), and no longer obviously needed.

## Reproducing

    SCRIPT=scripts/debug/verify_block_jacobi.py \
      ARGS="--geometry toroid --ns 12,24,12 --p 3 \
            --arms jacobi,blockjac_r3exact,blockjac_r3o2" \
      JOB_NAME=v OUTSUB=v bash slurm/job_diag_run.sh

Arm suffixes: `rN` inner rings, `oN` outer rings, `exact`/`face`/`nobc` boundary
mode, `bcsN` penalty multiplier (diagnostic only, `MRX_BJ_BC_SCALE`), `a2d`
2-D ring atoms, `modal` modal-radial.

## Infrastructure

Node `x3101c0s17b0n0` has a broken cuSolver — `gpusolverDnCreate` fails at
handle creation and every job landing there dies in the 1-D mass inverse. It
cost five jobs today. `slurm/job_diag_run.sh` now excludes it by default;
override with `EXCLUDE=`.
