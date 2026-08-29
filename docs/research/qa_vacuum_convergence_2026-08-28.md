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
