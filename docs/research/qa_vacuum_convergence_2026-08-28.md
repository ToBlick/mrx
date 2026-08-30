> **Status:** current (2026-08-28)
> **Read this for:** how well a VMEC vacuum equilibrium agrees with MRX's discrete harmonic field, how each converges with resolution, and where the residual lives
> **Do not read for:** the wout reader (`mrx/vmec.py` docstring) or the Clebsch IC route (`relaxation_ic_2026-08-25.md`, `coarse_gvec_export_2026-08-26.md` section 5)

# QA vacuum field: VMEC wout vs the discrete harmonic form, by resolution -- 2026-08-28

Question (Tobias): take the vacuum solution behind `outputs/vmec_sections/qa_relaxed`
(`data/wout_LandremanPaul2021_QA_lowres.nc`, `presf == 0`), do NOT relax it, and see how
it converges against MRX's vacuum field from the nullspace construction; the harmonic
field's scale is free and has to be fitted; decide how the error is measured.

Script: `scripts/vacuum_convergence.py` (branch `qa-vacuum`). Runs: `outputs/qa_vacuum/`
of that worktree (`rung_*/result.json`, `fields.npz`, `convergence.json`, `convergence.png`,
`residual_zeta0.png`; the json and the two figures are copied to
`docs/research/qa_vacuum_convergence_2026-08-28/`). Every rung is one slurm GPU job,
float64 (`seq.tol` = 1.5e-8; float32 would sit at 3.4e-4, above everything measured here).

## What is compared

Per rung `(ns, p)`, both in the Dirichlet 2-form space `V_2^h`:

- `B_w`: the wout field by the production IC route (`load_clebsch` ->
  `clebsch_potential_form` -> `potential_two_form`), i.e. the commuting projection
  `Pi_2 B_VMEC`: `||div B|| ~ 1e-15`, wall-normal part 0, toroidal flux `phi_edge` to 1e-15,
  Tesla per field period (`||B_w||_M` 0.5308, rms |B| 0.998 T vs the file's `volavgB` 0.999).
- `h`: `seq.nullspace(2, True)[0]` of `compute_nullspaces` (direct Hodge construction),
  M-normalised, sign free.

The map is refit at every rung (`build_gvec_map(map_ns=ns, p=p)`: R, Z collocated at the
Greville points of a polar sequence of the same ns, p), so the geometry error is O(h^{p+1})
and all rungs converge to one object, the vacuum field inside VMEC's LCFS. The clamped radial
knots have `n_r - p` uniform elements, so the spaces are not nested and `h := 1/(n_r - p)`.

**Scale.** `c = <B_w, h>_M / <h, h>_M`; then `D = ||B_w - c h||_M / ||B_w||_M = sin(theta)`, the
M-angle between the fields. The flux-matched scale `c_flux = Phi(B_w)/Phi(h)` equals `c` to
1e-11 at every rung: not a cross-check but a theorem of the discrete Hodge decomposition
(the curl part of `B_w` is M-orthogonal to `h` and carries no section flux). Fit and flux
match are the same thing; there is nothing to choose.

**Errors.** `D` (same space, no transfer); `||F||_M` of each field at `||B||_M = 1`
(`compute_force`); `E_h = ||h(h) - h(h_ref)||`, `E_w` likewise, in Cartesian components on a
fixed logical grid (Gauss x midpoint, 48x96x48; `E` on the same grid reproduces `D` to 4
digits) against the finest rung; the map difference `|F_h - F_ref|/L` alongside; and the
bulk/axis split at `rho = 0.1`. Field-line iota (`mrx.poincare.section_figure`, 160 seeds,
200 periods) against the file's `iotaf(s = rho^2)`.

**Gate.** The harmonic form's `v^T L v / v^T M v` against `lambda_1` (printed by
`compute_nullspaces` since d2bd6a5): 2.3e-13, 8.4e-13, 1.4e-12, 4.7e-12 along the ladder,
4.3e-13 / 1.8e-12 for p=2 / 4 -- all far below the 1e-10 gate. `||F||_M(h)` is 6e-14 to 4e-13
at every rung (the solver floor; a harmonic form has `J = 0` by construction).

## Results

Ladder at p=3 (reference for `E`: (24,48,24)); p-sweep at 9 radial elements and (24,12)
angular cells. Rates are `ln(e_prev/e)/ln(h_prev/h)` against the previous rung.

| rung | h | n2 | ratio | D | rate | D bulk (rho>=0.1) | rate | D axis (rho<0.1) | ||F||_M(B_w) | rate | E_h | rate | map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (8,16,8) p3 | 1/5 | 2 192 | 2.3e-13 | 2.718e-3 | -- | 2.714e-3 | -- | 3.11e-3 | 3.291e-2 | -- | 6.897e-3 | -- | 5.74e-4 |
| (12,24,12) p3 | 1/9 | 8 376 | 8.4e-13 | 5.671e-4 | 2.67 | 5.658e-4 | 2.67 | 6.73e-4 | 1.057e-2 | 1.93 | 1.340e-3 | 2.79 | 8.09e-5 |
| (16,32,16) p3 | 1/13 | 21 024 | 1.4e-12 | 2.810e-4 | 1.91 | 1.874e-4 | 3.00 | 1.98e-3 | 3.085e-3 | 3.35 | 4.723e-4 | 2.84 (*) | 1.84e-5 |
| (24,48,24) p3 | 1/21 | 74 928 | 4.7e-12 | 3.925e-4 | -0.70 | 5.159e-5 | 2.69 | 3.67e-3 | 1.976e-3 | 0.93 | ref | | ref |
| (11,24,12) p2 | 1/9 | 7 512 | 4.3e-13 | 7.820e-4 | | 7.688e-4 | | 1.42e-3 | 1.176e-2 | | 5.636e-3 | | 2.48e-4 |
| (13,24,12) p4 | 1/9 | 9 240 | 1.8e-12 | 5.276e-4 | | 5.010e-4 | | 1.64e-3 | 1.080e-2 | | 9.516e-4 | | 2.77e-5 |

(*) the pair against the reference itself is biased low. Least-squares slopes over the
p=3 ladder: `D` 1.43 (floored), `D bulk` **2.78**, `||F||_M(B_w)` 2.06, `E_h` **2.80**
(2.79 without the last pair), `E_w` 2.78, map 3.58.

Wall per rung (one H100, JIT included): 238 / 276 / 323 / 597 s along the ladder, of which
the nullspace construction is 36 / 54 / 91 / 276 s; the p-sweep 243 / 340 s.

### The rates

- **MRX converges at O(h^p) for the 2-form field, O(h^{p+1}) for the map** (`E_h` slope 2.80
  for p=3, map 3.58 -- both biased low by the finite reference). `h` and `B_w` converge to the
  reference at the same rate (`E_w` 2.78), as the projection of a smooth field should.
- **`D` in the bulk converges at the same O(h^p) with no floor down to 5.2e-5** (2.71e-3 ->
  5.66e-4 -> 1.87e-4 -> 5.16e-5, rates 2.67 / 3.00 / 2.69). That is the physics statement:
  away from the axis, VMEC lowres (mpol = ntor = 8, ns = 75, fsq 1e-13) is the vacuum field
  of its own boundary to better than 5e-5 relative in L2, and the number is still falling,
  so it is an upper bound on VMEC's bulk error, not the error.
- **The global `D` floors at 3-4e-4 and the floor is the axis.** At (24,48,24), 98.3 % of
  `||B_w - c h||^2` sits inside `rho < 0.1` (1.1 % of the volume); pointwise `|B_w - c h|/|B_w|`
  is 4.4e-3 rms and 2.0e-2 max there against 3-5e-5 rms in every shell outside `rho = 0.3`
  (`residual_zeta0.png`). The axis residual GROWS with resolution (6.7e-4 -> 2.0e-3 ->
  3.7e-3), and so does `||J||_M(B_w)` (0.047 -> 0.17 -> 0.37): a real non-harmonic feature of
  the reconstructed VMEC field that finer meshes resolve rather than smooth. `||F||_M(B_w)`
  keeps falling (1.98e-3 at the top) because the Leray projection removes the gradient
  part of that localised `J x B`.
- Where the axis feature comes from, two candidates, not separated here: (i) the wout's
  half-mesh lambda has its first node at `rho = sqrt(0.5/74) = 0.082`, so inside `rho < 0.08`
  our clamped spline of `lmns` is an extrapolation, and `B_w` there is built from it; (ii)
  VMEC's own axis region (the `m > 0` axis rows are an extrapolation the reader pins to 0;
  VMEC's radial resolution is poorest at the axis). Either way it is the wout DATA at the
  axis, not the MRX discretisation: `h` itself converges cleanly there (`E_h` has no axis
  anomaly).
- **The p-sweep at 9 radial elements is angular-limited, not a p-rate.** `D` 7.8e-4 / 5.7e-4 /
  5.3e-4 and `E_h` 5.6e-3 / 1.3e-3 / 9.5e-4 for p = 2 / 3 / 4: p=4 buys only 30 % over p=3
  because the (24,12) angular cells are the limit (ntor = 8 modes per period on 12 zeta
  cells). The map, dominated by low modes, does improve with p (2.5e-4 / 8.1e-5 / 2.8e-5).
  A p-study needs the angular cells raised with p.

### Iota

Field-line iota vs `iotaf(s = rho^2)`, max over regular lines with seed `rho >= 0.1` (the
innermost seed at `rho = 0.02` is an axis probe with an ill-defined winding and dominates
the raw max):

| rung | B_w | h |
| --- | --- | --- |
| (8,16,8) p3 | 2.9e-5 | 1.8e-3 |
| (12,24,12) p3 | 1.3e-5 | 1.6e-4 |
| (16,32,16) p3 | 8.3e-6 | 4.5e-4 |
| (24,48,24) p3 | 9.2e-6 | 6.0e-5 |
| (11,24,12) p2 | 7.2e-5 | 8.3e-3 |
| (13,24,12) p4 | 8.3e-6 | 4.4e-4 |

`B_w`'s iota is VMEC's at every resolution (a commuting projection keeps the flux surfaces;
the 1e-5 is the tracer). `h`'s iota converges to it with h (7e-4 -> 1e-4 -> 4e-5 -> 2e-5 in
the mid-radius difference `iota_h - iota_w`), with the same angular limit in the p-sweep.

### The relaxation run's fields, at (12,24,12) p3

`B_ic` of `qa_relaxed/B.h5` (float32) reproduces the float64 `B_w` to 5.8e-7 in the M-norm
(provenance check). `B_final` (430 CG steps, floored): `D` 9.5e-5 against 5.7e-4 for the IC,
`||F||_M` 1.65e-3 against 1.06e-2. The relaxation moved the field 6x closer to the harmonic
form of its mesh, and its force floor is below the finest projection's (1.98e-3): the descent
removes VMEC's axis current that the projection preserves.

## What it means

1. A VMEC vacuum equilibrium and MRX's nullspace field are the same object away from the
   axis, to at least 5e-5, converging at O(h^3) at p=3. The `test_vmec.py` gate
   (5.67e-4 at (12,24,12)) is a mesh-limited number, not a VMEC number.
2. The route validation is stronger than the earlier GVEC-h5 note's: no stored field
   sample, no rotation trap, no finite-sample floor -- the wout Fourier data are the input
   and the harmonic form uses none of them beyond R, Z.
3. The wout field's only defect is at the magnetic axis (2 % of |B| pointwise inside
   `rho < 0.1` at 21 radial elements). For relaxation ICs this is the region the descent
   fixes first; for a vacuum-field reference it means: compare in the bulk, or fix the
   axis reconstruction (extend lambda's half-mesh spline through the axis with the
   `rho^m` regularity instead of the clamped extrapolation) and re-measure `D_axis`.
4. Convergence orders: 2-form field O(h^p), map O(h^{p+1}), iota of `h` ~O(h^2-3).

## Open

- Which of (i) the lambda half-mesh extrapolation or (ii) VMEC's axis is the axis residual:
  rebuild `lmns`'s spline with the axis pinned (`lambda(0) = 0` for `m > 0`, `rho^m`) and
  rerun (24,48,24); if `D_axis` drops with it, the reader owns the floor.
- A p-study with angular cells scaled with p (e.g. (n_el + p, 4 n_el, 2 n_el)).
- (32,64,32) reference to un-bias the last `E` pair (est. 30-60 min; not needed for the
  conclusions above).

## Follow-up (2026-08-28, later the same day): the axis residual was the reader

The Open item above was tested in three steps on `static-dynamic-refactor`
(commits cb65d2c, 3406d38/c7cc252, c532378/8ab03e8), each rerun through this
script at (12,24,12) and (24,48,24), float64:

| reader / map state | (12,24,12) D | (24,48,24) D | ‖F‖_M(B_w) | ‖J‖(B_w) |
|---|---|---|---|---|
| this note: sampled map, half-mesh lambda extrapolated over the axis | 5.67e-4 | 3.92e-4 | 1.98e-3 | 0.374 |
| lambda pinned at the axis and the edge (`_lambda_nodes`) | 5.67e-4 | 3.98e-4 | 2.00e-3 | 0.379 |
| map L2-projected from the series (branch `analytic-map`) | 3.73e-4 | 3.97e-4 | 1.98e-3 | 0.375 |
| + axis parity of every mode (`c'(0) = 0` even m, `c''(0) = 0` odd m) | 3.73e-4 | 2.34e-4 | 1.14e-3 | 0.163 |
| + full axis behaviour: every derivative of order < m or wrong parity, up to the degree (`_axis_orders`) | 3.82e-4 | **8.4e-5** | **6.7e-4** | **0.056** |

(i) the lambda extrapolation was NOT it; (ii) the map projection helps the
coarse rung only (map error); (iii) the axis residual is the radial refit of
`rmnc`/`zmns`/`lmns` leaving the `rho^m` structure of each mode unenforced --
the cone `docs/research/analytic_map_2026-08-28.md` measured. With the full
axis behaviour enforced, ‖J‖ at (24,48,24) is at the coarse-rung level (no
longer growing) and D continues the O(h^3) trend across the ladder. The
p-sweep and the (16,32,16)/(8,16,8) rungs of the table above were measured
with the OLD reader; `convergence_axis.png` (below) is the ladder on the final
reader.

Ladder on the final reader (`convergence_axis.png`, `convergence_axis.json`;
`residual_zeta0_axis.png` the (24,48,24) residual map): D = 2.24e-3 /
3.82e-4 / 1.30e-4 / 8.36e-5 at h = 0.200 / 0.111 / 0.077 / 0.048, LS slope of
the global D 2.36 (bulk 2.64) with no axis floor; ‖F‖_M(B_w) 9.15e-2 /
9.38e-3 / 2.10e-3 / 6.68e-4, slope 3.49; the harmonic form's self-convergence
2.77 (map 3.65) as before.

## Extension (2026-08-29): the (32,64,32) reference, p = 1..4, and the dual harmonic field

Branch `qa-extend` (from `static-dynamic-refactor` c9e6b8e: L2-projected map, the
wout axis fixes, the `_periodic_symbol` fix). The whole ladder was **rerun on this
reader** -- the numbers below supersede every table above (those mixed the OLD and
axis readers); read this section for the current state. Three additions to
`scripts/vacuum_convergence.py`:

1. **(32,64,32) p=3** as the reference, so `E` is measured against a rung finer than
   the top of the study instead of against (24,48,24) itself.
2. **p = 1, 2, 4** at 9 radial elements ((10,24,12) p1, (11,24,12) p2, (13,24,12)
   p4), alongside the existing (12,24,12) p3, so the abscissa `h = 1/(n_r - p) = 1/9`
   is shared.
3. **`h1 = seq.nullspace(1, False)[0]`**, the k=1 free (no-BC) harmonic 1-form. On
   the solid torus `b1 = 1`, so `h1` and `h2 = nullspace(2, True)[0]` are both
   1-dimensional and Poincare dual: the SAME vacuum toroidal field in two form
   degrees. `h1`'s own gate mirrors `h2`'s with `|curl h1|/|h1|` (the `d h1 = 0`
   half) in place of `|div h2|`. Both are pushed to lab-frame `(3,)`-vectors at the
   quadrature points (covariant rule `(DF^T)^{-1} h1_hat` for the 1-form, Piola
   `DF h2_hat / J` for the 2-form; `DF`, `J` explicit), fit by one physical-L2 scale,
   and compared.

### The completed p=3 ladder (reference (32,64,32))

Grid 48x96x48. `h = 1/(n_r - p)`. Rates against the previous rung.

| rung | h | n2 | D | rate | D bulk | rate | D axis | ‖F‖_M(B_w) | E_h | rate | map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (8,16,8) p3 | 0.200 | 2 192 | 2.239e-3 | -- | 2.240e-3 | -- | 2.18e-3 | 9.15e-2 | 6.44e-3 | -- | 4.4e-4 |
| (12,24,12) p3 | 0.111 | 8 376 | 3.816e-4 | 3.01 | 3.703e-4 | 3.06 | 9.47e-4 | 9.38e-3 | 1.33e-3 | 2.68 | 6.0e-5 |
| (16,32,16) p3 | 0.077 | 21 024 | 1.297e-4 | 2.94 | 8.974e-5 | 3.85 | 8.88e-4 | 2.10e-3 | 4.11e-4 | 3.19 | 1.3e-5 |
| (24,48,24) p3 | 0.048 | 74 928 | 8.361e-5 | 0.91 | 5.877e-5 | 0.88 | 5.65e-4 | 6.68e-4 | 1.37e-4 | 2.29 | 1.2e-6 |
| **(32,64,32) p3** | 0.034 | 182 336 | 8.219e-5 | 0.05 | 6.275e-5 | -0.20 | 5.01e-4 | 6.34e-4 | ref | | ref |

LS slopes over the ladder: `D` 1.91, `D bulk` 2.11, `‖F‖_M(B_w)` **2.94**, `E_h`
**2.72** (2.86 without the finite-reference pair), `E_w` 2.71, map **4.06**. Wall
132 / 164 / 297 / 580 / **1044** s (the (32,64,32) rung is 17 min on one H100, well
inside its budget -- no OOM at MEM_GB 128).

- **The (32,64,32) rung shows the global `D` and the bulk `D` have BOTTOMED, not
  merely slowed.** `D` 8.36e-5 -> 8.22e-5 and `D bulk` 5.88e-5 -> 6.28e-5 across the
  last pair (the bulk even ticks up inside the noise): the reconstructed VMEC vacuum
  field sits at a **physics floor of ~8e-5 global, ~6e-5 in the bulk**, and the
  earlier note's "still falling, an upper bound" is now resolved to a floor. `‖F‖_M`
  floors with it (6.68e-4 -> 6.34e-4). The extra rung was worth it: it converts the
  open "the bulk `D` has no floor down to 5e-5" into a measured floor.
- **`E_h` is a clean O(h^3)** (slope 2.72, or 2.86 dropping the reference-adjacent
  pair) and the **map is O(h^{p+1}) = O(h^4)** (slope 4.06), both now un-biased by a
  finite reference. `E_w` tracks `E_h` (2.71): `h` and `B_w` converge to the finest
  rung at the same rate, as the projection of one smooth field should.

### p-scan at 9 radial elements (h = 1/9), (24,12) angular

| p | rung | D | D bulk | ‖F‖_M(B_w) | E_h | harmonic ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | (10,24,12) | 7.44e-2 | 7.46e-2 | 5.05e-1 | 1.39e-1 | 2.0e-13 |
| 2 | (11,24,12) | 3.949e-4 | 3.79e-4 | 9.57e-3 | 5.57e-3 | 4.5e-13 |
| 3 | (12,24,12) | 3.816e-4 | 3.70e-4 | 9.38e-3 | 1.33e-3 | 8.4e-13 |
| 4 | (13,24,12) | 3.599e-4 | 3.37e-4 | 7.52e-3 | 9.40e-4 | 1.9e-12 |

- **p=1 builds and passes its harmonic gate (ratio 2e-13) but the FIELD is
  worthless**: `D` 7.4e-2, `‖B‖` near the axis 4e-4 T (should be ~1 T), `D axis`
  ~800. A degree-1 2-form cannot represent the axis field of this geometry; the gate
  certifies `h` is harmonic, not that it resolves `B_w`. Reported and kept as the
  low end, not dropped.
- **p=2..4 is angular-limited, not a p-rate**, exactly as before: `D` 3.95e-4 ->
  3.82e-4 -> 3.60e-4 barely moves (the (24,12) cells and ntor=8 modes are the wall),
  while `E_h`, which is not axis-dominated, does improve with p (5.57e-3 -> 1.33e-3
  -> 9.40e-4). A p-study needs the angular cells scaled with p.

### The dual harmonic field: h1 (1-form) vs h2 (2-form)

`convergence_h1.png`; per-rung in `convergence_extend.json`
(`harmonic1`, `rep_independence`, `D_grid_h1`).

| rung | h | ratio h1 | \|curl h1\|/\|h1\| | E_h1 | resid ‖v1−s v2‖/‖v1‖ (bulk) | M-cos(h1,h2) | cos(B_w,h1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (8,16,8) p3 | 0.200 | 6.2e-14 | 4.3e-7 | 2.36e-3 | 5.71e-3 | 0.99998376 | 0.99998126 |
| (12,24,12) p3 | 0.111 | 1.7e-13 | 7.4e-7 | 4.23e-4 | 1.29e-3 | 0.99999917 | 0.99999910 |
| (16,32,16) p3 | 0.077 | 3.1e-13 | 9.7e-7 | 1.52e-4 | 4.06e-4 | 0.99999992 | 0.99999991 |
| (24,48,24) p3 | 0.048 | **6.6e-8** | 4.8e-4 | 1.09e-4 | 9.98e-5 | 0.99999999 | 0.99999999 |
| (32,64,32) p3 | 0.034 | **1.8e-5** | 8.0e-3 | ref | 1.16e-4 | 0.99999999 | 0.99999999 |
| (11,24,12) p2 | 0.111 | 1.2e-13 | 6.0e-7 | 1.83e-3 | 6.56e-3 | 0.99997841 | 0.99997833 |
| (13,24,12) p4 | 0.111 | 2.9e-13 | 9.6e-7 | 3.53e-4 | 9.82e-4 | 0.99999952 | 0.99999945 |

**The headline: the two representations are the same physical vacuum field, and they
converge to each other at O(h^p).** The scale-fitted residual between the lab-frame
vectors falls 5.71e-3 -> 1.29e-3 -> 4.06e-4 -> ~1e-4 along the p=3 ladder (LS slope
**2.82**), the M-cosine reaches **+0.99999999** (nine 9s), and `cos(B_w, h1)` -- the
production VMEC field against the k=1 harmonic form, two entirely separate
constructions -- reaches the same. The discrete vacuum solution does not depend on
the form degree it is carried in. The p-sweep shows the same p-improvement (resid
6.56e-3 / 1.29e-3 / 9.82e-4 for p=2/3/4).

**One caveat, and it is a real one about the k=1 solve, not the physics.** The k=1
free harmonic FORM is less robust than the k=2 Dirichlet one at the fixed `seq.tol`:
its Rayleigh ratio is `O(tol^2)` (6e-14 .. 3e-13) on the coarse three rungs but
**floors and then grows with n** -- 6.6e-8 at (24,48,24), 1.8e-5 at (32,64,32) --
**failing the 1e-10 gate at the two finest rungs**, with `|curl h1|/|h1|` rising in
step (4.8e-4, 8.0e-3). h2's ratio stays ~1e-12 throughout. So the k=1 construction
leaves a small coexact ripple on `h1` that finer meshes make worse, not better, at
this tol. Two consequences: (i) **`h1`'s own self-convergence `E_h1` floors** (slope
**2.21** vs `E_h` 2.72) because the (32,64,32) reference `h1` is itself contaminated;
(ii) despite that, the **field-level agreement with `h2` is unharmed to ~1e-4** --
the ripple is a small-amplitude, high-curl coexact mode that barely moves the L2 of
the field, so `resid_h1_vs_h2` and the cosines stay on the O(h^p) trend. If `h1`
itself is ever wanted to spectral accuracy at high resolution the k=1 free Hodge
solve needs a tighter tol or a better-conditioned construction; for the vacuum field
it represents, it is already right.

### What the extension adds

1. The bulk `D` **floors at ~6e-5** (global ~8e-5) -- the open "(32,64,32) reference"
   item is closed and the floor is now a number, not an upper bound.
2. The 2-form field is **O(h^3)** and the map **O(h^4)** with the finite-reference
   bias removed (`E_h` 2.72/2.86, map 4.06).
3. p=1 is unusable (axis field lost) though it passes the harmonic gate; p=2..4 is
   angular-limited at 9 radial elements.
4. **The k=1 free and k=2 Dirichlet harmonic fields are one physical vacuum field**
   (M-cosine nine 9s, residual O(h^3) to ~1e-4) -- representation independence of the
   discrete solution -- with the one caveat that the k=1 form's harmonic RATIO
   degrades with resolution at fixed tol (gate fails at the two finest rungs) while
   the field it carries does not.

Figures: `convergence_extend.png` (same-space `D`/`F` and vs-finest `E`/map ladders
with the p-sweep overlaid), `convergence_h1.png` (`h1` self-convergence and the
`h1`-vs-`h2` representation residual), `residual_zeta0_extend.png` (the (32,64,32)
residual at zeta=0, max 7.9e-4, still axis-localised). JSON: `convergence_extend.json`.

## Full p x resolution grid (2026-08-29, later the same day)

Follow-up (Tobias): fill in the grid so every p has a full ladder, not one point.
The angular cells are fixed per column to the p=3 ladder's value, so degree is the
only thing that changes at a given `h`:
`n_elements -> (n_theta, n_zeta) = 5->(16,8), 9->(24,12), 13->(32,16), 21->(48,24),
29->(64,32)` and `n_r = n_elements + p`. 12 rungs added ((6,16,8)..(30,64,32) p1,
(7,16,8)..(31,64,32) p2, (9,16,8)..(33,64,32) p4); ~1.8 GPU-h on one H100 each,
float64. All merge into the same `outputs/qa_vacuum/`; figures regenerated.

### D on the full grid

`D = ||B_w - c h||_M / ||B_w||_M` at `h = 1/n_elements = 0.200 / 0.111 / 0.077 /
0.048 / 0.034` (n_elements 5 / 9 / 13 / 21 / 29). `convergence_grid.png`.

| p | n_el 5 | n_el 9 | n_el 13 | n_el 21 | n_el 29 | LS slope (full) | LS slope (pre-floor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.260e-1 | 7.442e-2 | 5.232e-2 | 3.269e-2 | 2.375e-2 | 0.95 | **0.92** |
| 2 | 1.885e-3 | 3.949e-4 | 1.425e-4 | 8.091e-5 | 8.170e-5 | 1.84 | **2.70** |
| 3 | 2.239e-3 | 3.816e-4 | 1.297e-4 | 8.361e-5 | 8.219e-5 | 1.91 | **2.98** |
| 4 | 1.314e-3 | 3.599e-4 | 9.290e-5 | 7.970e-5 | 8.233e-5 | 1.65 | **2.72** |

- **p=1 is not flat -- it converges at O(h^1)**, the degree rate (slope 0.92), and the
  task-1 note's "flat at ~7e-2" was an artefact of reading a single point. The FIELD is
  still unusable: global `D` only reaches 2.4e-2 (2.4 %) at 29 elements, and `|B|` on
  the axis is ~4e-4 T against ~1 T (the degree-1 2-form cannot carry the axis field).
  But the RATE is the correct O(h^p). In the bulk it is even better -- `D_bulk` 1.26e-1
  -> 2.78e-3, slope ~2.2 -- so the p=1 defect is entirely the axis (`D_axis` 27..800):
  bulk p=1 is a real O(h^2) field, the axis is where degree 1 fails.
- **Every p >= 2 bottoms at the same ~8e-5 physics floor** (the reconstructed VMEC
  field's own distance from the harmonic field of its boundary, section above), and the
  **higher p reaches it at a coarser mesh**: p=4 is on the floor by n_el=13, p=2 and p=3
  by n_el=21. That is the O(h^p) signature -- the steeper rate meets the floor sooner.
- **The full-ladder LS slope hides this and even inverts it** (p=4 scores 1.65, the
  LOWEST, because it floors earliest and spends three of five rungs on the plateau). The
  meaningful number is the **pre-floor slope** over the three coarsest rungs, which
  steepens cleanly with degree: **0.92 / 2.70 / 2.98 / 2.72** for p = 1 / 2 / 3 / 4 (p=4
  is slightly compressed because its third rung, n_el=13, already touches the floor, so
  only two of its rungs are truly pre-floor). Read the grid figure by the descent rate
  before the plateau, not the LS fit through it.

### The dual harmonic field across the grid

`h1` (k=1 free) vs `h2` (k=2 dbc), the representation-independence residual
`||v1 - s v2|| / ||v1||` of the lab-frame vectors and the M-cosine, on every rung
(`convergence_h1.png`, right panel):

- **The two representations agree on the whole grid, converging O(h^p) to ~1e-4**
  (resid slope 2.82 along the p=3 ladder; M-cosine reaches +0.99999999 at p=3/p=4 fine,
  and +0.9988 even for the broken p=1 field). Representation independence holds at every
  degree; the number of nines tracks the field quality (three for p=1, eight for p>=3).
- **The k=1 caveat sharpens: the k=1 free harmonic-FORM gate now fails on the finest
  one-to-two rungs of every p >= 2, and worse at higher p.** `h1`'s Rayleigh ratio (PASS
  `<= 1e-10`) degrades to 4.2e-8 (p2, n_el=29), 1.8e-5 (p3, n_el=29) and **6.4e-4** (p4,
  n_el=29), with `|curl h1|/|h1|` rising to 4.7e-2 there. The k=2 form stays ~1e-12
  throughout. So the k=1 free Hodge construction floors at the fixed `seq.tol` and the
  floor grows with BOTH resolution and degree. Consequence unchanged from task 1: it
  contaminates `h1` itself (its self-convergence `E_h1` floors, slope 2.21) but not the
  vacuum field it carries (`resid_h1_vs_h2` at (33,64,32) p4 is still 7.2e-4, M-cosine
  +0.99999974). If `h1` is ever needed to spectral accuracy at high p or resolution the
  k=1 free solve needs a tighter tol or a better-conditioned construction.

### A GPU-memory note (the reader/script, not the physics)

The (33,64,32) p=4 rung OOMed the H100 on the first attempt -- a ~10 GiB coefficient-
window gather in the map/projection evaluation, materialised over the whole quadrature
grid at once (`MAP_BATCH_SIZE_INNER = 0`). Fixed by batching that evaluation: the script
now honours `MRX_MAP_BATCH_SIZE_INNER` (exported per job) and the rung completes in
30 min at batch 262 144. Only that one rung needs it; the default is unchanged.

### What the grid adds

1. p=1 converges at O(h^1) (correcting "flat/broken" -> "right rate, wrong field: the
   axis is lost, the bulk is O(h^2)").
2. The O(h^p) rate steepens with p in the pre-floor region (0.92 / 2.70 / 2.98 / 2.72);
   all p >= 2 share the ~8e-5 physics floor and higher p reaches it at coarser mesh.
   The full LS slope is not the rate -- the plateau drags it toward 2 and inverts the
   p-ordering.
3. Representation independence (h1 == h2) holds across the whole grid; the k=1 form's
   own harmonic gate fails on the finest rungs of every p >= 2 (worse at higher p) while
   the field it carries does not.

Figures (regenerated on the full grid): `convergence_grid.png` (D vs h, one ladder per
p, h^p guides, the ~8e-5 floor line -- the headline), `convergence_extend.png`
(same-space p=3 with the faint off-p D ladders + vs-finest E/map), `convergence_h1.png`
(h1 self-convergence and the representation residual, one ladder per p),
`residual_zeta0_extend.png`. JSON: `convergence_extend.json` (`slopes.D_by_p`,
`slopes.D_by_p_prefloor`, `slopes.D_floor`, per-rung `harmonic1` / `rep_independence`).

## Densified grid + the common-grid resolution fix (2026-08-30)

Follow-up (Tobias): the ladder (n_el 5,9,13,21,29) was too sparse to read the slope
or the elbow onto the ~8e-5 floor. Densified to **n_el = 5, 7, 9, 11, 13, 17, 21, 25,
29, 37, 45** for p = 2, 3, 4 (p=1 skipped -- its O(h) line and broken field are already
clear), angular fixed per column `(n_theta, n_zeta) = (2(n_el+3), n_el+3)`, `n_r = n_el+p`.
18 new rungs, ~9 GPU-h (the (49,96,48) p4 rung alone is 112 min: a 59-min k=1 Hodge solve
plus the batch-slowed post-processing). Batching env `MRX_MAP_BATCH_SIZE_INNER` as before.

### The common evaluation grid had to be raised (it was under-resolving the fine end)

The cross-resolution `E` comparisons sample every rung on the fixed common grid; that
grid must OUT-resolve the finest rung or the finest field is under-sampled and `E` is
corrupted at the fine end. With the finest rungs now ~(49,96,48), the old default
`(48,96,48)` does not out-resolve them. Added a `--regrid` mode to
`vacuum_convergence.py` (rebuilds only the geometry + bases, re-pushes the STORED DoFs
onto a new `--grid`, no Hodge solve -- cheap) and re-evaluated **all 38 rungs on
`--grid 192,288,144`** (radial 192 Gauss = 4x the finest 48 DoFs; angular 288/144 = 3x,
still spectrally exact for the mpol=ntor=8 field). Regrid is exact: `D_grid` reproduces
the same-space `D` to all digits at every rung.

**Grid-independence check** (E_h of three p=3 rungs vs the p=3 finest, at the grid and
finer), the number Tobias asked for:

| test rung | E_h @128,192,96 | @192,288,144 | @256,384,192 |
|---|---|---|---|
| (8,16,8) coarse | 6.434e-3 | 6.433e-3 (0.00%) | -- |
| (20,40,20) mid | 1.885e-4 | 1.884e-4 (0.01%) | -- |
| (40,80,40) finest pair | 2.605e-5 | 2.459e-5 (−5.6%) | 2.423e-5 (−1.4%) |

Coarse and mid pairs are grid-independent to <0.01% at any of these grids. Only the
finest PAIR (n_el 37 vs the n_el 45 reference) -- where the difference field is ~2e-5
relative and hardest to sample -- is grid-sensitive: `(48,96,48)` was ~7% high there,
`192,288,144` is within ~1.5% of the 2x grid. **We therefore report on `192,288,144`**;
the self-convergence slopes below shift by at most ~0.1 (p2 1.79->1.67 on the coarser
grid) and are stable on it. **Caveat on the earlier sections of this note:** they used
`(48,96,48)` with a `(32,64,32)` finest -- only 1.5x radial margin -- so their
finest-PAIR `E` values (the `(24,48,24)->(32,64,32)` pair, already flagged there as
"biased low") carry the same ~5-7% grid caveat; their SLOPES, dominated by the
grid-independent coarser pairs, stand.

### The dense D grid

`D = ||B_w - c h||_M / ||B_w||_M`, grid-independent (same-space). `convergence_grid.png`.

| p | n_el 5 | 7 | 9 | 11 | 13 | 17 | 21 | 25 | 29 | 37 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.89e-3 | 8.67e-4 | 3.95e-4 | 1.87e-4 | 1.42e-4 | 1.04e-4 | 8.09e-5 | 8.03e-5 | 8.17e-5 | 8.35e-5 | 8.40e-5 |
| 3 | 2.24e-3 | 1.02e-3 | 3.82e-4 | 1.97e-4 | 1.30e-4 | 8.28e-5 | 8.36e-5 | 8.03e-5 | 8.22e-5 | 8.37e-5 | 8.41e-5 |
| 4 | 1.31e-3 | 9.11e-4 | 3.60e-4 | 1.25e-4 | 9.29e-5 | 9.69e-5 | 7.97e-5 | 8.35e-5 | 8.23e-5 | 8.39e-5 | 8.42e-5 |

(h = 1/n_el = 0.200 down to 0.0222.) Pre-elbow slope (fit over rungs with D > 2x the
floor 8.0e-5) and the elbow (first on-floor n_el):

| p | pre-elbow slope | fit window (n_el) | elbow at n_el |
|---|---|---|---|
| 1 | 0.95 | 5–29 (never elbows) | — |
| 2 | **2.91** | 5–11 | 13 |
| 3 | **3.13** | 5–11 | 13 |
| 4 | 2.14 | 5–9 | 11 |

- **The elbow onto the ~8e-5 floor is now resolved per p, and it moves to COARSER mesh
  as p rises** (p2/p3 elbow at n_el 13, p4 already at n_el 11). The pre-elbow descent
  steepens with degree (p2 2.91, p3 3.13).
- **p=4 reaches the floor so fast there is no clean pre-elbow window** (only n_el 5,7,9
  sit above 2x the floor, and n_el 9 is already ~4.5x it), so its D-fit slope (2.14) is
  compressed -- not a real O(h^4) failure. The self-convergence below measures p=4
  cleanly instead.

### Per-p self-convergence (each p vs its OWN finest, n_el = 45)

`B_w` on the common grid against that p's own finest rung; free of the single-global-
reference bias. `convergence_selfconv.png`. This is the clean O(h^p) statement:

| p | self-conv slope (E_w) | tracks |
|---|---|---|
| 1 | 0.89 | h^1 |
| 2 | 1.67 | ~h^2 |
| 3 | 2.83 | h^3 |
| 4 | 3.46 | ~h^4 |

The rate steepens cleanly and monotonically with degree, each ladder hugging its `h^p`
guide over the whole window (p2 1.6e-2->5.9e-4, p3 6.8e-3->2.5e-5, p4 4.4e-3->5.0e-6
across n_el 5->37). The slopes sit a little below the integer p only because the finite
n_el=45 reference biases the last pair -- exactly the effect the grid-resolution fix
minimised. **This is the figure to read for the convergence order; the D grid shows the
approach to the physics floor.**

### The dual harmonic field: the k=1 caveat is now the dominant fine-end effect

The denser ladder and the finer reference expose what the coarse sweep could not: the
k=1 free harmonic-FORM solve does not merely fail the gate on the last rung -- it makes
`h1` NON-convergent, and the representation residual TURNS UP. Along the p=3 ladder:

| n_el | 5 | 9 | 13 | 17 | 21 | 25 | 29 | 37 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| ratio h1 | 6e-14 | 2e-13 | 3e-13 | 1e-10 | 7e-8 | 2e-6 | 2e-5 | 4e-4 | 3e-3 |
| resid h1↔h2 | 5.7e-3 | 1.3e-3 | 4.1e-4 | 1.9e-4 | 1.0e-4 | **7.4e-5** | 1.2e-4 | 5.6e-4 | 1.7e-3 |

- The two representations agree BEST at n_el ≈ 25 (residual 7.4e-5, M-cosine
  +0.999999997, nine 9s) and then DIVERGE at finer meshes, because the k=1 form's
  Rayleigh ratio climbs past the 1e-10 gate at n_el ≈ 17 and reaches 3e-3 (|curl h1|/|h1|
  = 0.1) at n_el = 45. `h1` itself no longer converges (`E_h1` slope ~0.25, flat). The
  earlier note's "same field to eight nines, resid O(h^3)" was the LEFT side of this
  minimum (its finest was n_el 29); it is right there but is not the whole story.
- `h2` (k=2 dbc) stays at ratio ~1e-12 throughout, and `B_w` self-converges cleanly
  (above), so this is purely the k=1 free Hodge construction at fixed `seq.tol`, not the
  physics. Anyone using `h1` at n_el ≳ 20 needs a tighter k=1 solve; for the vacuum field
  itself use `h2` or `B_w`.

### What the densification adds

1. The D elbow onto the ~8e-5 floor is resolved and moves to coarser mesh with p; the
   pre-elbow slope steepens (p2 2.91, p3 3.13); p4 floors too fast to fit in D.
2. Clean per-p self-convergence of `B_w`: **0.89 / 1.67 / 2.83 / 3.46** for p = 1..4 --
   textbook O(h^p) steepening, now that the reference is n_el=45 and the common grid
   (192,288,144) out-resolves it.
3. The common eval grid had to be raised from (48,96,48) to (192,288,144); grid-
   independence verified (<0.01% coarse/mid, ~1.5% at the finest pair). Earlier sections'
   finest-pair E values carry a ~5-7% caveat; their slopes stand.
4. The k=1 free harmonic form's representation residual has a MINIMUM at n_el≈25 and
   rises after -- the k=1 solve, not the physics, is the fine-end limit for `h1`.

Figures (on the 192,288,144 grid): `convergence_grid.png` (dense D vs h, per-p pre-elbow
slope fits + floor line -- the elbow figure), `convergence_selfconv.png` (per-p
self-convergence of `B_w`, the clean O(h^p)), `convergence_h1.png` (the h1 turnaround),
`convergence_extend.png`, `residual_zeta0_extend.png`. JSON: `convergence_extend.json`.
