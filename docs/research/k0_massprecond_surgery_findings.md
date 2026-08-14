# k=0 Hodge preconditioner: metric handling, axis surgery, and the atom choice

**Session 2026-07-22.** Dense condition-number study of the k=0 stiffness
`K_0 = G_0^T M_1 G_0` (`operators.py:574`), settling several questions about how
the anisotropic metric enters the preconditioner. All numbers from
`scripts/debug/verify_hodge_massprecond_k0.py` (exact dense `kappa` at small
DBC sizes, toroid / cerfon / rotating-ellipse). Framework cross-check: the
reconstructed `fd + surgery` reproduces the production
`apply_laplacian_preconditioner(k=0, tensor)` column to a few percent, and the
exact-Schur surgery gives `kappa = 1.00` (block-LDU validated).

## Structural facts (verified)

- `K_0 = G_0^T M_1 G_0`: `G_0` (discrete gradient / incidence) is **geometry-free**;
  `M_1` (1-form mass) carries **all** geometry — its weight is the full `g^{-1} J`
  tensor, i.e. the conflicting `g_rr / g_tt / g_zz` **are** the `M_1` weight.
- The production `apply_laplacian_preconditioner(k=0, tensor)` is
  **`fd`-bulk + exact dense polar-core Schur surgery**:
  - bulk = the `fd` atom, *literally* — `D = cbrt(a_rr a_tt a_zz)` (geometric mean),
    unweighted separable 1D atoms, `alpha` mixing, `D^{-1/2} fd_apply D^{-1/2}`
    (`operators.py:_assemble_k0_greville_bulk_factors`; identical to
    `build_fd_atom(mode='fd')` in `laplacian_mg_k0.py`).
  - core = `3 * nz` near-axis DOFs (`preconditioners.py:_core_size`), inverted
    **exactly** as a small dense Schur block + core/bulk coupling.
- `kappa(Hodge)` grows with h AND shaping, helical worst:
  toroid 1.5/2.3/3.4, cerfon 2.6/4.5/7.1, rot-ellipse 4.7/11.1/17.2 at ns=4/6/8.
  → single-level; MG still justified; target regime = shaped + helical + fine-h.

## Resolution of the "metric can't be lumped" question

Splits by **region**:
- **Off-axis (bulk): lumping works.** The `g_rr/g_tt/g_zz` anisotropy is *bounded*
  away from the axis, so the geomean-`D` `fd` atom is adequate.
- **On-axis: lumping is impossible** (`g^tt ~ 1/r^2 -> inf`) — so it is *not*
  lumped; it is **solved exactly** by the `3*nz` core Schur surgery.

The bulk metric model is **second-order**; the core surgery dominates.

## The "B" idea (route through the mass preconditioner) — CLOSED

`B = L_0^{-1} (G_0^T P_1 G_0) L_0^{-1}` with `L_0 = G_0^T G_0`, `P_1 ~ M_1^{-1}`
(the tensor mass preconditioner). Motivation: handle the metric via the *good*
mass preconditioner instead of a scalar atom. Result: `kappa ~ 100-1000`, grows
with h, dominated by production Hodge (`kappa 2-17`).

- **Axis surgery does NOT rescue B.** With the exact core removed, `kappa` barely
  moves (toroid 393 -> 391, cerfon 321 -> 318 at ns=6). B's error is **bulk-global
  curl-leakage** (the regular-decomposition constant — `M_1` mixing gradient and
  non-gradient 1-forms), *not* axis-local. Surgery rescues `fd` (axis-local `1/r^2`
  disease); it cannot rescue B (bulk-global disease).
- **`M_id`-projection variant `B'` helps but is still dominated.** Use the
  geometry-free 1-form mass `M_id` (metric = I, J = 1) — separable, FD-exact — so
  `L' = G_0^T M_id G_0` is the reference-domain FEM Laplacian, a better projection
  metric than the topological `G_0^T G_0`. The *consistent* construction is the
  `M_id`-weighted pseudo-inverse
  `B' = L'^{-1} (G_0^T M_id P_1 M_id G_0) L'^{-1}` (exact when `M_1 = M_id`).
  `B'` is 2.6-6x better than B and flatter across geometry (154/122/164 at ns=6),
  but still `kappa ~ 100`, grows with h. The reference-`L^2` projection is the
  *right* projection, yet cannot lift B out of "dominated by fd+surgery."

## fd vs fdax vs fdbund (bulk atom, exact core surgery fixed)

`kappa` at ns=6 (isolates the metric-lumping choice; core surgery identical):

| geometry        | fd (geomean) | fdax (pull out J) | fdbund (avg <g^aa J>) | Hodge |
|-----------------|-------------:|------------------:|----------------------:|------:|
| toroid          |     2.41     |       3.08        |        2.53           | 2.28  |
| cerfon          |     4.70     |       5.43        |        4.43           | 4.50  |
| rotating_ellipse|    10.87     |      12.44        |       12.01           | 11.06 |

- **`fdbund` > `fdax` on every geometry.** Averaging the *bundled* `<g^aa J>` beats
  pulling `J` out pointwise and averaging the *bare* `g^aa`. Why: bundled
  `g^tt J ~ 1/r` is milder than bare `g^tt ~ 1/r^2`, so its cross-axis average is
  a more faithful representative of the bulk (the 1/r^2 field is dominated by the
  innermost points even with the xi_1 cutoff).
- **`fdbund ~ fd` on `kappa`** at these (mildly anisotropic) test geometries:
  `fd` marginally better on toroid (2.41 vs 2.53) and rot-ellipse (10.87 vs 12.01);
  `fdbund` better on shaped cerfon (4.43 vs 4.70); all within ~15%.
- **But `fdbund` is the better-MOTIVATED atom.** `fd`'s `D = cbrt(a_rr a_tt a_zz)` is
  a *single isotropic scalar* — the same value in all three directions at each
  point, exactly right only when `a_rr = a_tt = a_zz`. For any real anisotropy it
  mis-scales all three directions and leans on 3 *global* `alpha` constants to
  compensate; pointwise it gets *none* of the directional weights right. `fdbund`
  instead carries each axis's own weight `<g^aa J>` — the three distinct directional
  weights are represented (as own-axis profiles). It trades pointwise resolution
  for **directional correctness**, which is the property that matters for an
  anisotropic operator. The `kappa`-parity on mild geometries is exactly what to
  expect: the geomean's mis-scaling only bites hard on **strong** anisotropy
  (W7-X), where `fdbund` should pull ahead. **This remains unconfirmed** — see
  Decisions: no local geometry is anisotropic enough to test it (the ellipse
  folds at `kappa = 2`), and on the one helical case we can build, `fd` wins.

## Decisions / direction

- **Adopt `fdbund` over `fdax`**: strictly better `kappa` on every geometry,
  cheaper (`D = 1`), and better-motivated (bundled `<g^aa J>` vs bare `g^aa`).
- **`fdbund` vs the production `fd` (geomean) — DEFERRED to a real W7-X test; do
  NOT switch the shipped atom yet.** The theory says `fdbund` (each directional
  weight represented) should beat the geomean (one isotropic scalar, gets none
  right pointwise) on *strong* anisotropy — but **no local geometry confirms it**,
  and the nearest evidence leans the other way: `fd` wins on toroid (2.41 vs 2.53)
  AND on the helical rotating-ellipse (10.87 vs 12.01, the closest local W7-X
  proxy); `fdbund` wins only on cerfon (4.43 vs 4.70). The rotating-ellipse
  ellipticity proxy **cannot reach strong anisotropy** — the map folds at
  `kappa = 2` (min Jacobian <= 0), and the valid ceiling `kappa = 1.5` is only
  mildly anisotropic (raw `kappa ~ 969`, ~ toroid). So the `fdbund`-payoff claim is
  unconfirmed locally. Gate the switch on a real W7-X (cluster) run; if `fdbund`
  wins there, change `_assemble_k0_greville_bulk_factors` (`mrx/operators.py`) to
  per-axis bundled-average weighted 1D stiffnesses (`<g^aa J>`, `D = 1`) mirroring
  `build_fd_atom(mode='fdbund')`, behind the production k=0 convergence tests.
- **B family closed** — do not pursue routing K_0^{-1} through the mass
  preconditioner; the curl-leakage is intrinsic and surgery-immune.
- ~~**Open lever for the shaped/helical growth**: the off-diagonal `g^{r theta}`~~
  **REFUTED 2026-07-24** — see the off-diagonal ladder section below. Dropping
  ALL off-diagonal blocks costs almost nothing; the shaped/helical growth is
  the *averaging/separability error of the diagonal weights*.

## The off-diagonal g^{r theta} ladder (2026-07-24) — lever closed, real lever identified

Dense bulk MODEL matrices (exact inverse, exact core surgery held fixed) in
`verify_hodge_massprecond_k0.py`; sanity: the all-9-blocks pointwise assembly
reproduces `K0[bulk,bulk]` to ~3e-16. `kappa` at ns=6 / ns=8:

| rung (rot-ellipse)                          | ns=6  | ns=8  |
|---------------------------------------------|------:|------:|
| exact bulk block (ceiling for ANY atom)     |  1.53 |  1.65 |
| pointwise diag only (drop ALL off-diag)     |  2.60 |  3.45 |
| pointwise diag + g^{rt}                     |  1.63 |  1.81 |
| zeta-avg diag (2D (r,t) solve per z-mode)   |  8.01 | 10.15 |
| zeta-avg diag + rt                          |  8.02 | 10.11 |
| fdbund kron-sum (= atom, cross-check)       | 12.01 | 20.50 |
| fdbund + zeta-avg g^{rt} cross              | 12.05 | 20.58 |

(cerfon ns=8: ceiling 1.46, diag-pt 2.29, diag+rt 1.46, zavg 2.29/1.46,
fdbund 5.61. toroid: all pointwise/zavg rungs = ceiling — orthogonal map,
g^{rt} = 0, zeta-independent.)

Conclusions:

- **The off-diagonal hypothesis is REFUTED as the dominant lever.** Dropping
  all off-diagonals costs only 1.5 -> 2.6 (ns=6 helical); the current atoms sit
  at ~12. Restoring g^{rt} on top of an *averaged* diagonal gains nothing
  (12.05 vs 12.01) — the cross term is second-order to the diagonal averaging
  error. (g^{rt} IS essentially all of the off-diagonal content: diag+rt ~=
  ceiling everywhere.)
- **The dominant error is averaging/separability of the DIAGONAL weights,
  specifically along zeta on helical geometry.** zeta-averaging alone (keeping
  (r,t) pointwise, one dense 2D solve per zeta mode — production-plausible)
  falls from 2.60 to 8.01 on the rot-ellipse: the helical metric genuinely
  varies along zeta in reference coordinates. On axisymmetric geometry
  zeta-averaging is lossless (as it must be).
- **The h-growth of the shipped atoms is averaging error, not off-diagonal
  loss**: ceiling and pointwise rungs are nearly h-stable (1.53->1.65,
  2.60->3.45) while the averaged atoms grow (fdbund 12->20.5, zavg 8->10.2).
- **No practical single-level atom upgrade pays.** The best production-shaped
  candidate (2D-per-zeta-mode solves) buys only ~1.5x on the hard geometry at
  real cost. The remaining spread is 3D-pointwise variation — exactly what a
  V-cycle coarse correction (or an expensive block structure, e.g. per-zeta-slab
  block-Jacobi, which would sit between diag-pt and zavg by construction)
  addresses. Whether ANY of this matters inside MG is the jacobi-vs-fd
  smoother A/B (`scripts/debug/run_mg_k0_jacobi_ab.sh`).

## Point-Jacobi vs fd/fdbund inside the two-level MG (2026-07-24) — jacobi OUT

`run_mg_k0_jacobi_ab.sh`: 8^3, two-level, fat-core R=1, anchored-xi1, auto-m,
CG to 1e-10. fd-family: cheb-lo 0.85 (m=2); jacobi (= exact diag(K_0), all 9
metric blocks pointwise, ~free apply): relative windows kappa=4 (m=3) and
kappa=9 (m=4). CG iterations, dbc/free:

| geometry         | fd (m=2) | fdbund (m=2) | jac w4 (m=3) | jac w9 (m=4) | 1-level baseline |
|------------------|---------:|-------------:|-------------:|-------------:|-----------------:|
| toroid           |   5 / 8  |    5 / 5     |   61 / 168   |   47 / 113   |     19 / 28      |
| cerfon           |  10 / 13 |   10 / 12    |   65 / 180   |   49 / 124   |     28 / 37      |
| rotating_ellipse |  12 / 14 |   15 / 13    |   63 / 160   |   48 / 117   |     40 / 52      |

- **ms/it is ~equal across all MG arms** (18-29 ms; A-applies dominate, the fd
  atom's tensor transforms are noise) => iterations ~ total cost. **The fd
  atoms win by 5-13x. MG-jacobi even loses to the single-level baseline.**
- **Window tuning cannot rescue jacobi**: w4->w9 trades 61->47 its at 7->9
  A-applies/cycle -- total A-applies flat (427 vs 423). Structural, not tuning.
- **Why**: the polar frame itself. g^tt ~ 1/r^2 makes the theta/r coupling
  ratio range ~1..36 across the BULK (not just the axis); theta-oscillatory
  modes at small-but-not-core radii are algebraically smooth yet invisible to
  geometric coarsening -- the classic anisotropic point-smoother failure. The
  fd atoms' averaged 1D theta-stiffness (~1/xi_1 weight) captures exactly this.
  Free-BC is worse still (near-null constant handling).
- fd vs fdbund inside MG: statistical tie (fdbund lower lam_max, sometimes
  more its) -- consistent with deferring the atom switch to W7-X.

**Smoother question CLOSED, from both sides**: jacobi (cheaper) costs 10x
iterations; anything fancier than fd/fdbund (ladder above) buys <= 1.5x kappa
on the hardest geometry at real complexity cost. The fd-family separable atom
+ exact core surgery is the sweet spot. Remaining MG-vs-single-level payoff
question is fine-h/W7-X (cluster-gated): at 8^3 the single-level baseline
still wins wall-clock (~3 ms/it vs ~20), but its iteration count grows with
geometry/h (19->40 dbc) while MG(fd) stays ~5-15.
