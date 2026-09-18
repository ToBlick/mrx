# The harmonic atom's floor computed, its profiles from B, and the smoothed Newton direction (2026-09-17)

Branch `worktree-newton-atom-smoothing` off `newton-second-variation` (1b9bfd3). Three changes
to the Newton preconditioner and direction, each a switch, and one seven-arm study on li383
(16,32,32) p=2 mixed, Newton from the VMEC field, 100 steps, chunk 20, `--floor-tol 0
--dt-floor 0`, against the two baselines of 2026-09-11 (`kappa_sweep/k3_it100`: h, kappa 3,
no regularised search; `stepreg_sweep/k3_it100_c0.1`: the same with C = 0.1).

## 1. What was changed

**The floor computed instead of tuned** (`--harmonic-floor strain`,
`TimeStepper.newton_atom_floor=None`). With `div u = div h = 0`,
`curl(u x h) = h . grad u - u . grad h`. Per mode the lumped parallel derivative is
`i k_par I` (anti-Hermitian) and the strain `S^i_c = d_c h^i` of a curl-free field is
symmetric, so the normal form of `i k_par I - S` is `k_par^2 I + S^T S` with no cross term:
the dropped term enters the symbol as a plain addition. `mrx.hessian.harmonic_atom_profiles`
takes the logical contravariant components (reference over the Jacobian) on the quadrature
grid, central-differences them along the three logical axes (periodic in the angles, one-sided
at the radial ends) and angle-averages `sum_i (d_c h^i)^2` per radial layer and logical
direction `c`: the diagonal of `S^T S`, the floor of component `c`. No parameter. Traceable in
the field.

**The profiles from B** (`--harmonic-field B`, `newton_atom_field="B"`). The atom reads two
angle-averaged profiles and (with the strain floor) three more; nothing in the apply depends on
where they come from. From `B` the atom is rebuilt inside the jitted step from the current
field: one quadrature evaluation of `B`, three differences, five radial means, two
interpolations, next to 100 MINRES iterations of three k=1 mass solves each. The field's own
rotational transform places the resonances instead of the vacuum field's.

**The smoothed Newton direction** (`--newton-smoothing true`, `newton_smoothing=True`). The
descent's smoother on the Newton POTENTIAL, `(M_1 + mu L_1)^-1 M_1 a` with the velocity
smoothing scale (0.02 / n_r^2) and order, then the curl, so the direction stays
divergence-free exactly (the potential route's construction). The 2026-09-12 prototype
(section 8 of `harmonic_atom_and_helicity_correction_2026-09-11.md`) smoothed the 2-form
velocity with the regularised search ON and the two fought: rejected. This time the arms with
smoothing run with the regularised search OFF, the arm that was never run.

## 2. The profiles (smoke job 18648518, li383 (16,32,32), the VMEC field at step 0)

`kappa=3 floor = 3 (2 pi)^2 (h^theta^2 + h^zeta^2)`; strain floor per logical direction.

| r | field | h^theta | h^zeta | kappa=3 floor | strain rho | strain theta | strain zeta |
|---|---|---|---|---|---|---|---|
| 0.008 | h | 0.0428 | 0.316 | 12.1 | 201 | 10.1 | 2.5 |
| 0.135 | h | 0.0429 | 0.317 | 12.1 | 0.085 | 0.064 | 0.052 |
| 0.421 | h | 0.0443 | 0.321 | 12.5 | 0.010 | 0.088 | 0.056 |
| 0.706 | h | 0.0478 | 0.331 | 13.2 | 0.015 | 0.230 | 0.084 |
| 0.992 | h | 0.0575 | 0.346 | 14.6 | 0.039 | 0.494 | 0.147 |
| 0.008 | B | 0.0398 | 0.312 | 11.7 | 0.023 | 0.002 | 0.035 |
| 0.421 | B | 0.0482 | 0.317 | 12.2 | 0.008 | 0.051 | 0.043 |
| 0.706 | B | 0.0607 | 0.330 | 13.3 | 0.018 | 0.165 | 0.054 |
| 0.992 | B | 0.0769 | 0.353 | 15.4 | 0.027 | 0.495 | 0.129 |

- The lumped strain is 0.01 to 0.5: 25 to 1000 times below the kappa = 3 floor. In kappa
  units the physical floor is 0.001 to 0.04. **kappa = 3 never modelled the strain; it is a
  regulariser** (the 2026-09-11 sweep chose it because kappa = 1 went past the floor sooner).
- The theta strain dominates and grows outward: the Jacobian's theta variation (det DF in
  [0.29, 2.3]) is what `u . grad h` sees, not the radial profile.
- h's innermost layer carries an axis artefact in the rho strain (201): the reference
  rho component over a Jacobian that vanishes at the axis, differenced on the first
  quadrature points. B does not (0.023). One layer, a floor: harmless, noted.
- B's theta profile is 30% above h's at the edge (0.077 vs 0.058, iota_edge 0.65 against the
  vacuum transform): the resonant surfaces of the two atoms differ.

## 3. The arms (`outputs/atom_study/`, jobs 18648539-18648545)

| arm | field | floor | C | smoothing |
|---|---|---|---|---|
| h_strain_c0.1 | h | strain | 0.1 | - |
| h_strain_c0 | h | strain | 0 | - |
| B_strain_c0.1 | B | strain | 0.1 | - |
| B_strain_c0 | B | strain | 0 | - |
| B_k3_c0.1 | B | 3 | 0.1 | - |
| h_k3_c0_smooth | h | 3 | 0 | on |
| B_strain_c0_smooth | B | strain | 0 | on |

Results: section 4 (pending).
