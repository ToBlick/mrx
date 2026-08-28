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
