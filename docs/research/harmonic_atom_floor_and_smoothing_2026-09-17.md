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

## 4. Round one: without a penalty in the operator

`F2` the squared normalised residual per step, `last20` the mean of steps 81-100, `E_rem` the
energy removed, `dt` the mean accepted step, `fb` fallbacks to the smoothed force, MINRES 100
iterations on every step unless noted.

| arm | min F2 (step) | F2 at 30 / 60 / 90 | last20 | E_rem | dH/H | dt | fb |
|---|---|---|---|---|---|---|---|
| h, kappa 3, C 0 (baseline) | 4.9e-9 (33) | 5.7e-9 / 8.6e-9 / 7.6e-9 | 7.9e-9 | 2.20e-6 | +7e-7 | 0.90 | 0 |
| h, kappa 3, C 0.1 (baseline) | 3.9e-9 (65) | 5.8e-9 / 3.9e-9 / 4.5e-9 | 4.5e-9 | 1.97e-6 | -1.1e-5 | 0.25 | 0 |
| h, strain, C 0 | 2.1e-8 (15) | 2.8e-8 / 2.5e-8 / 5.7e-8 | 5.6e-8 | 3.02e-6 | +6.9e-5 | 0.95 | 0 |
| B, strain, C 0 | 4.8e-8 (19) | 1.1e-7 / 6.9e-8 / 6.7e-8 | 6.9e-8 | 2.77e-6 | +3.6e-5 | 0.95 | 0 |
| h, strain, C 0.1 | 6.8e-9 (54) | 1.1e-8 / 6.8e-9 / 7.8e-9 | 7.8e-9 | 1.98e-6 | -9.3e-6 | 0.11 | 6 |
| B, strain, C 0.1 | 8.5e-9 (54) | 2.1e-8 / 8.8e-9 / 9.3e-9 | 9.3e-9 | 1.99e-6 | -6.8e-6 | 0.14 | 7 |
| B, kappa 3, C 0.1 | 3.8e-9 (70) | 6.2e-9 / 3.8e-9 / 4.2e-9 | 4.3e-9 | 1.98e-6 | -1.1e-5 | 0.26 | 0 |
| h, kappa 3, C 0, smoothed | 5.6e-6 (3) | 3.8e-5 / 1.7e-5 / 8.0e-6 | 8.0e-6 | 1.84e-6 | -9.0e-6 | 0.26 | 6 (MINRES 27) |
| B, strain, C 0, smoothed | 2.5e-6 (99) | 1.4e-5 / 6.6e-6 / 2.8e-6 | 3.0e-6 | 1.84e-6 | -5.9e-6 | 0.11 | 46 (MINRES 57) |

- **The physical strain floor without a penalty goes past the resolved floor.** Minimum at
  step 15-19, then the residual climbs to 6e-8 while the energy removed grows 25-40% above
  the baselines and the helicity drifts 50-100x more: the signature of the 2026-09-11 kappa
  = 1 and 300-iteration arms, stronger. The peer session's spectrum note of the same day
  (`hessian_spectrum_2026-09-17.md`) explains it: the Hessian's soft end is a field-aligned
  near-null continuum (`u ~ f B`, eigenvalues 1e-4..1e-1) that the atom mis-models by 1e3,
  and the force's round-off components along it, divided by those eigenvalues, are
  parallel-flow directions 10-100x the physical step. **kappa = 3 was Levenberg-Marquardt
  damping of that continuum inside the preconditioner**, by accident of the iteration count.
  The physical strain (25-1000x smaller) removes the damping and Newton descends the
  grid-scale energy.
- The regularised search (C = 0.1) contains it only by cutting the step to a tenth, with
  fallbacks; the floor it holds (7.8e-9) is the Laplacian atom's, not kappa 3's (4.5e-9).
- **Profiles from B**: 5% below the h atom at the same settings. Neutral on li383, where
  B is 96% harmonic; the mechanism (the field's own transform places the resonances) costs
  nothing and stays available.
- **Smoothing the Newton potential is a negative a second time**, now with the regularised
  search off: the residual sits at 1e-5..1e-6, MINRES converges early (27-57 iterations) on
  a direction that is then often not a descent direction (46 fallbacks of 100 on the B arm).
  The rough content of the Newton direction is not separable from its descent: the
  direction resolves the force's rough part, and filtering it leaves the smooth part
  pointing wrong. Together with section 8 of the 2026-09-11 note (the 2-form smoother with
  the search on): closed.

## 5. Round two: the strain floor with the parallel penalty in the operator

The peer session's remedy (64ffbd9, `--newton-parallel-penalty ALPHA`): `H + alpha M_par`,
`<v, M_par u> = int (v.B)(u.B)/|B|^2 J`, in the units of `H` against `M_2`, lifts the
continuum to `alpha` and leaves the energy descent unchanged (`<F, fB> = 0`). The atom's
consistent floor is `strain + alpha` (`harmonic_preconditioner(..., shift=alpha)`, b8b218e).
Arms (jobs 18648567-71): B strain alpha {0.3, 1, 3} C 0; h strain alpha 1 C 0; B strain
alpha 1 C 0.1.

## 6. Round two: the strain floor with the penalty, 100 steps

| arm | job | min F2 (step) | F2 at 30 / 60 / 90 | last20 | E_rem | dH/H | dt | fb |
|---|---|---|---|---|---|---|---|---|
| B, strain, alpha 0.3, C 0 | 18648568 | 2.5e-10 (100) | 1.2e-9 / 5.1e-10 / 2.7e-10 | 2.7e-10 | 2.00e-6 | -1.0e-5 | 1.00 | 0 |
| B, strain, alpha 1, C 0 | 18648567 | 5.6e-10 (100) | 3.3e-9 / 1.3e-9 / 6.7e-10 | 6.7e-10 | 1.96e-6 | -9.8e-6 | 1.00 | 0 |
| B, strain, alpha 3, C 0 | 18648569 | 2.7e-9 (100) | 8.4e-9 / 4.5e-9 / 3.0e-9 | 3.0e-9 | 1.88e-6 | -9.6e-6 | 1.00 | 0 |
| h, strain, alpha 1, C 0 | 18648570 | 5.6e-10 (100) | 3.3e-9 / 1.3e-9 / 6.7e-10 | 6.7e-10 | 1.95e-6 | -9.7e-6 | 1.00 | 0 |
| B, strain, alpha 1, C 0.1 | 18648571 | 5.6e-10 (100) | 3.3e-9 / 1.3e-9 / 6.7e-10 | 6.7e-10 | 1.96e-6 | -9.8e-6 | 1.00 | 0 |
| control: B, strain, alpha 0, C 0 | 18648542 | 4.8e-8 (19) | climbs to 6.9e-8 | 6.9e-8 | 2.77e-6 | +3.6e-5 | 0.95 | 0 |
| best baseline: h, kappa 3, C 0.1 | - | 3.9e-9 (65) | 5.8e-9 / 3.9e-9 / 4.5e-9 | 4.5e-9 | 1.97e-6 | -1.1e-5 | 0.25 | 0 |

- **With the continuum lifted in the operator, the physical floor works**: the residual is
  still falling at step 100 (alpha 0.3: 1.2e-9, 5.1e-10, 2.7e-10 at 30 / 60 / 90), 17x below
  the best baseline, at the Newton step on every step, no fallbacks, with the SAME energy
  removed and the same helicity drift as the baselines: the same equilibrium, more
  accurately. What September called the resolved floor (4e-9) was the method's floor.
- **Neither kappa nor the regularised search is needed.** C 0.1 equals C 0 to every digit
  (the search never binds at dt = 1); h equals B to three digits (li383 is 96% harmonic).
- Smaller alpha is better down to 0.3 (each factor 3 in alpha is a factor ~2 in the
  residual); alpha 0 climbs, so there is an optimum or a cliff below 0.3: round three.
- The atom and the operator now agree on both ends: the stiff end by the parallel symbol,
  the soft end by `alpha`, the strain in between. One physical parameter, `alpha`, in the
  units of the Hessian against the mass, replaces kappa (a tuned floor), the regularised
  search (a tuned step cut) and the dt floor (a tuned stop).
- The line search's own optimum along the penalised direction is far beyond the cap:
  `dt*` mean 9 (steps 1-20) to 117 (steps 81-100) with the step held at 1 by
  `newton_dt_cap`. The damped operator overestimates the curvature along the soft
  perpendicular modes, so the direction is short there. Whether a longer step descends
  faster or reintroduces the parallel flows is the cap test of round three.

## 7. Round three (running): alpha {0.03, 0.1, 0.3} at 200 steps, (32,64,64) at alpha 0.3, the step cap {4, inf}

Jobs 18648612-15, 18648617-18.

**The step cap stays at 1** (`B_strain_a0.3_c0_cap{4,inf}`, 100 steps): with the cap
lifted the line search settles at dt = 2.0 from step 40 on (dt* mean 1.96-2.01, the
September value) and the residual sits at 1.1e-7 (chunk means 1.7e-7, 1.1e-7, 1.1e-7,
1.2e-7 at 40-100) against 2.7e-10 with the cap: the resolved modes' residual is reflected
at dt* = 2, as the 2026-09-08 reasoning for the cap said. The dt* means of 9-117 in the
capped runs are a few soft-mode steps, not a systematically short direction.

**The cliff is between alpha 0.03 and 0.1** (200-step arms at step 100): alpha 0.3 2.7e-10,
0.1 8.3e-10, 0.03 4.0e-9 with the energy removed growing (2.25e-6 against 2.00e-6) and
the helicity drift turning positive, the first signs of the undamped descent of section 4.
The optimum is at 0.3 or above it, not below.

**200 steps** (`B_strain_a{0.03,0.1,0.3}_c0_200`, jobs 18648612-14, 4.05 s/step):

| alpha | min F2 (step) | F2 at 30 / 60 / 90 / 100 / 200 | E_rem at 100 / 200 | dH/H | dt | fb |
|---|---|---|---|---|---|---|
| 0.03 | 2.1e-9 (135) | 1.7e-8 / 6.2e-9 / 4.4e-9 / 4.0e-9 / 2.4e-9 | 2.25e-6 / 2.63e-6 | +3.1e-5 | 1.00 | 0 |
| 0.1 | 4.2e-10 (178) | 3.2e-9 / 1.7e-9 / 8.5e-10 / 8.3e-10 / 4.6e-10 | 2.07e-6 / 2.19e-6 | -5.6e-7 | 1.00 | 0 |
| 0.3 | 1.9e-10 (189) | 1.2e-9 / 5.1e-10 / 2.7e-10 / 2.7e-10 / 1.9e-10 | 2.00e-6 / 2.05e-6 | -9.4e-6 | 1.00 | 0 |

- alpha 0.3 keeps descending through step 200 (chunk means 2.35e-10 -> 1.89e-10 over
  120-200) with the energy nearly flat (+2.5% over the second hundred steps) and the
  helicity flat: the equilibrium of the baselines, resolved further. 20x below the
  September floor at step 200, matching the peer's kappa 3 + alpha 1 arm (2.2e-10 at 200).
- alpha 0.1 plateaus at 4.6e-10 with the energy still moving (+5% over 100-200); alpha
  0.03 floors at 2.4e-9 with the energy removed 30% above the baselines and the helicity
  rising: the undamped descent, mild. The soft continuum must be lifted to about 0.3 in
  the units of H against M_2, i.e. to the middle of the perpendicular spectrum's soft end
  (the peer's LOBPCG: the softest perpendicular modes at 0.07-0.16 with alpha 1).
**(32,64,64)** (`B_strain_a0.3_c0_h32`, job 18648615, 16.6 s/step) against the September
kappa 3 arm (`kappa_sweep/k3_it100_h32`, 16.5 s/step), both 100 steps:

| arm | F2 at 30 / 60 / 90 | min F2 (step) | last20 | E_rem | dH/H | dt | fb |
|---|---|---|---|---|---|---|---|
| alpha 0.3, strain, C 0 | 9.5e-9 / 1.4e-9 / 5.7e-10 | 4.5e-10 (98) | 5.7e-10 | 1.70e-6 | -3.0e-6 | 1.00 | 0 |
| kappa 3, C 0 (September) | 3.6e-8 / 6.9e-9 / 3.5e-9 | 3.1e-9 (82) | 3.7e-9 | 1.67e-6 | -2.5e-6 | 0.65 | 0 |

Chunk means 1.7e-5, 1.2e-8, 2.4e-9, 1.1e-9, 5.7e-10: 1e-8 near step 40 (the September arm:
53, the paper's Laplacian arm: 256), 7x lower at step 100, the same energy and helicity,
still descending. The gain carries to the resolution where W7-X lost its surfaces.

- **Candidate default: `--newton-parallel-penalty 0.3 --harmonic-floor strain`, C 0, dt
  floor off, cap 1.** Field h or B (identical on li383; B costs nothing and follows the
  transform). Pending the (32,64,64) sections and Tobias's word.

## 8. Surfaces: the alpha 0.3 (16,32,32) final against the September best

Poincare traces (`poincare_trace.py --fields ic,final --periods 400`, 160 lines, five
planes; job 18648633; `outputs/atom_study/B_strain_a0.3_c0/{trace.npz,poincare/}`) against
the archives on disk:

| state | F2 | chaotic lines | at r | regular-line drift | iota |
|---|---|---|---|---|---|
| VMEC IC | 2.5e-4 | 0 / 160 | - | 2.2e-4 | 0.3947..0.6603 |
| alpha 0.3, strain, 100 steps (this study) | 2.5e-10 | 3 / 160 | 0.54, 0.67, 0.80 | 5.2e-3 | 0.3935..0.6603 |
| alpha 0.3, strain, 200 steps (this study, job 18648652) | 1.9e-10 | 4 / 160 | 0.11, 0.64, 0.67, 0.68 | 3.2e-3 | 0.3935..0.6603 |
| September h16 best (kappa 3, 200 steps) | 4e-9 | 4 / 160 | 0.55, 0.72, 0.80, 0.86 | 8.8e-4 | 0.3939..0.6599 |
| peer: kappa 3 + alpha 1, 200 steps | 2.2e-10 | 3 / 160 | 0.55, 0.68, 0.74 | 4.8e-3 | 0.3932..0.6602 |
| peer: kappa 3, alpha 0, 200 steps (control) | 1.3e-8 climbing | 51 / 160 | 0.10..0.89 | 1.3e-3 | 0.3952..0.6600 |

The zeta = 0 sections of the first four are the same picture (at 200 steps the 3/5 chain at r ~ 0.78 is a little more visible than at 100, as in the September best): nested surfaces, the same
small chains at 1/2 (r ~ 0.55) and 3/7 (r ~ 0.28), the same wavy edge outside r ~ 0.75,
the September best's islands in the 0.45-0.7 band if anything slightly larger, the iota
profiles identical in range and shape. A residual 17x lower did not cost surface quality at
(16,32,32). The control with the continuum undamped, 200 steps past the floor, is the
destroyed-surfaces case of W7-X 2026-09-12 reproduced on li383: 51 chaotic lines. The one
number that moved is the regular-line drift (5e-3 against 9e-4); the pictures do not show
what it measures. The (32,64,64) arm is the stricter test (W7-X lost its surfaces there).
