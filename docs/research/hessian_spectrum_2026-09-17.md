# The spectrum of the Newton Hessian: where the soft modes are, 2026-09-17

Branch `newton-second-variation`. Scripts, archives and figures in
`outputs/newton_second_variation/spectrum/`. Budget 20 GPU h (Tobias), spent ~2.5.

Question (Tobias): what do we know about the spectrum of the Hessian, and are the
outlier modes at the rational surfaces? The 2026-09-06 note had 150 Lanczos steps at
(16,32,32) on an L-BFGS state and presumed the seven lowest Ritz vectors resonant
without looking at them. This note looks.

## 1. States and operators

Two li383 (16,32,32) p=2 states, float64:

* **reconnected**: `outputs/newton_demos/seeded_reconnect/unseeded`, step 300, the
  paper's unseeded run after four 3 % reconnections, with the 3/5 chain at r = 0.83;
  iota 0.426 (axis) .. 0.616, flat at ~3/7 for r < 0.25.
* **nested**: `outputs/newton_sweeps_2026-09-13/h16`, best.h5 (step 39), the Newton
  floor, nested surfaces; iota 0.394 (axis) .. 0.660, 3/7 at r = 0.27, 1/2 at 0.53,
  3/5 at 0.79.

The Hessian is `second_variation(seq, B, J)` (the full symmetric form, curvature term
kept), on divergence-free velocities u = curl a, a in V^1_0, in the M_2 inner product:
the operator Newton inverts. The resonant surfaces are the rationals n/m with n a
multiple of nfp = 3, located on the trace archive's iota(r).

## 2. Krylov probes only pin the extremes (`probe_spectrum.py`)

200 Lanczos steps on H (M_2 metric, full reorthogonalisation) and 200 Arnoldi steps on
A P for the harmonic and the Laplacian atom (A = curl^T H curl, dual 1-forms, Krylov
space in range(A) so the gradient kernel stays out):

| probe | reconnected | nested |
|---|---|---|
| H, Ritz range | [0.069, 3.90e4] | [0.067, 3.89e4] |
| A P, harmonic atom | [3.9e-3, 510], spread 1.3e5 | [4.0e-3, 508], spread 1.3e5 |
| A P, Laplacian atom | [1.7, 2.27e5], spread 1.3e5 | [1.7, 2.26e5], spread 1.3e5 |

The two states agree to 2 % on every number: the Krylov extremes carry no information
about the islands. The high end converges (the 3.9e4 reproduces the September number)
and is the grid-scale (m, n) ~ (12, 16) content at the axis, where the cells are
smallest. The low end does NOT converge: the lowest Ritz values form a smooth ladder
(0.07, 0.8, 2.2, 4.4, 7.4, ... for H; 0.004, 0.012, 0.024, 0.043, ... for the harmonic
atom, ~ index^2) and the Ritz vectors are grid-scale speckle with no structure
(`reconnected_sections_lanczos.png`, `_harmonic.png`). Those are mixtures, not modes,
and the September "seven lowest Ritz values" were the same. A 200-step Krylov probe
cannot answer the outlier question on an operator with a spread of 6e5.

The 1.3e5 spread of the preconditioned operator is the extremes only; its bulk sits at
10..100 (`reconnected_spectrum.pdf`). The low tail of the harmonic-atom spectrum is
high-m grid-scale content (m = 12..15) near the axis: the atom over-corrects the stiff
end there, the "stiff-end commutation mismatch" of the September note.

## 3. Converged lowest block: LOBPCG (`lobpcg_spectrum.py`)

Block LOBPCG (ortho variant, block 32, 200 iterations, 2 s per iteration at 16^3) for
the lowest eigenpairs of H u = lambda M_2 u on u = curl a, preconditioned with
K = curl P curl^T (P the harmonic atom; a curl, so the iterates stay divergence-free
with no projection). Everything is saved every 10 iterations and the run resumes from
its archive. Per mode (`diagnostics.py`): the radial energy density
e(r) = int |u|^2 J dtheta dzeta, the (m, n) power of u^r on 18 surfaces, sections,
the energy fraction inside the first radial cell, and the field alignment
int (u . B)^2 / |B|^2 J / int |u|^2 J (1 for u = f B). `plot_spectrum.py` draws and
tabulates.

### 3.1 Unmasked: the soft end is a field-aligned near-null continuum at the axis

Both states: the 32 lowest eigenvalues run from 2.6e-4 to 0.16 and are still sliding
down at iteration 200 (a continuum, not isolated outliers). Every mode has 45..65 % of
its energy inside the first radial cell (r < 1/14) and is 70..98 % field-aligned. The
dominant helicity is (7, -1) in the reconnected state, where iota_axis = 3/7, and
(8, -1) / (14, -1) in the nested state, where iota_axis = 0.40 - the local resonance
of the axis region either way. The two states give the same ladder (lambda_0 = 2.8e-4
vs 2.9e-4): the islands are invisible to the softest block.

Reading: these are the discrete remnant of the exact null space u = f B of the
continuous Hessian (curl(f B x B) = 0 for any f; div u = 0 needs B . grad f = 0, so f
a flux function anywhere and a resonant f on a rational surface). The polar cells at
the axis are the smallest and give the discrete space the most room to represent them.
The harmonic atom mis-models them by 10^3 (Rayleigh ratio <u, H u> / <u, H_h u> =
0.001..0.05): on a field-aligned mode the atom's symbol collapses to its floor kappa,
the true Hessian to ~0.

### 3.2 Axis excluded: the soft modes sit on the rational surfaces, island or not

`--mask-cells L` restricts to potentials a = 0 on the first L radial cells (u = curl a
vanishes there too). With the first cell masked, nested state (no islands), the softest
modes are

| # | lambda | (m, n) | r of the (m, n) power peak | resonance |
|---|---|---|---|---|
| 0 | 2.6e-3 | (6, -1) | 0.60 | 1/2 at 0.53 |
| 1-4 | 3.2e-3 .. 5.5e-3 | (5, -1) | 0.65 .. 0.80 | 3/5 at 0.79 |
| 5-15 | 5.8e-3 .. 1.3e-2 | (7, -1) | 0.25 .. 0.30 | 3/7 at 0.27 |

within one or two cells of the surface, 55..77 % field-aligned. The reconnected state
gives the same helicities: (5, -1) at r = 0.75..0.95 (the 3/5 chain), (6, -1) at
0.3..0.65, (7, -1) spread over r < 0.3 (its iota is flat at ~3/7 there). With two cells
masked the reconnected state's softest modes are the (5, -1) ones on the 3/5 island
(lambda 3.7e-4 .. 2e-3), the island's closed lines giving the parallel flows the most
room. (The two-cell block reaching lower eigenvalues than the one-cell block, on a
nested subspace, says the one-cell block was far from converged: the eigenvalues in
these tables are upper bounds, the locations and helicities are what converged.)

So the answer to the question: yes, the soft modes are at the rational surfaces -
because the rational surfaces are where the field-aligned null flows u = f B with a
resonant f are divergence-free, not because of the islands. A nested state has them
at the same surfaces. And at the very bottom of the spectrum sits the axis, which
hosts the same kind of mode with the local helicity.

## 4. What it means for Newton

A parallel velocity component changes B by nothing, so these modes cannot damage the
surfaces by themselves. What they do is amplify: the right-hand side <u_j, F>_M is zero
in the continuum but not on the mesh, and Newton divides it by lambda_j ~ 1e-3..1e-4.
That noise is what the regularised line search (eps = C h_r^2) and the truncated MINRES
keep in check today, the latter by accident: the atom over-estimates their stiffness by
10^3, so 100 MINRES iterations barely move them. Near the floor, where F is round-off,
this is a candidate mechanism for the dt-floor stall.

Section 5 measures the force projections and the remedy.

## 5. Force projection and the parallel-flow penalty

**Force projection.** With ||F||_M = 1 at the state, the coefficient of the Newton
step along a soft mode is <u_j, F>_M / lambda_j:

| block | sum_j <u_j,F>^2 | max |<u_j,F>| | max proj/lambda | rms proj/lambda |
|---|---|---|---|---|
| reconnected, unmasked (axis) | 2.2e-4 | 7e-3 | 2.1 | 0.40 |
| nested, unmasked (axis) | 1.8e-4 | 7e-3 | 0.9 | 0.24 |
| reconnected, 1 cell masked | 6.2e-3 | 3e-2 | 7.8 | 2.0 |
| nested, 1 cell masked | 6.1e-3 | 5e-2 | 5.7 | 1.6 |
| reconnected, 2 cells masked (3/5 island modes) | 1.6e-2 | 5e-2 | 24 | 9.0 |

The force puts almost nothing on these modes (2e-4 .. 2e-2 of its energy on a block of
32), but a bulk mode with lambda ~ 10..100 contributes ~1e-3..1e-2 to the step and the
soft ones 1..24: an exact Newton step would be parallel-flow garbage ten to a hundred
times the physical step. The regularisation eps = C h_r^2 = 5e-4 is the size of these
lambda and halves them at best. What tames them in the released configuration is the
truncated MINRES with the harmonic atom, which believes them 10^3 stiffer and so moves
them by proj / (atom stiffness) ~ 1e-2 in 100 iterations: Levenberg-Marquardt damping
at the atom's floor level, by accident of the iteration count. (The peer session
measured the floor: kappa = 3 puts it at ~12 on every layer while the physical lumped
strain is 0.01..0.5, so the floor has been standing in for exactly this.)

**The penalty.** H_alpha = H + alpha M_par, <v, M_par u> = int (v . B)(u . B) / |B|^2 J:
Levenberg-Marquardt on the parallel component alone (Tobias's phrase). It lifts the
null space to alpha, leaves the energy descent unchanged (<F, f B> = 0) and the
perpendicular step untouched; one quadrature load per Hessian apply. alpha in the units
of H against M_2, a parallel unit mode sees lambda + alpha; the atom's consistent floor
is then strain + alpha. LOBPCG on H_1 (`reconnected_par1_lobpcg.npz`,
`nested_par1_lobpcg.npz`):

| | without penalty | alpha = 1 |
|---|---|---|
| lowest 32 eigenvalues | 2.6e-4 .. 0.16, still sliding at it 200 | 0.069 .. 0.16 (reconnected), 0.066 .. 0.15 (nested), flattening |
| alignment | 0.70 .. 0.98 | 0.01 .. 0.03 |
| <u,Hu> / <u,H_h u> | 0.001 .. 0.05 | 0.06 .. 0.30 |
| max proj / lambda | 2.1 / 0.9 | 0.026 / 0.032 |

The soft end is perpendicular now: an m = 1 axis shift at 0.07 in both states, then
(7, -1) / (6, -1) / (8, -1) modes at r = 0.15..0.25 (the flat-iota region near the
axis, 3..10 % of the energy in the axis cell), nothing at the 1/2 or 3/5 surfaces. The
atom is 3..15x off on them (the floor kappa = 3 is ~12 against their 0.07), which is
the remaining preconditioner question and the peer session's (strain floor).

Code: `second_variation(seq, B, J, parallel_penalty=alpha)`,
`TimeStepper.newton_parallel_penalty`, `relax.py --newton-parallel-penalty A` (default
0), commit 64ffbd9.

## 6. Newton arms with the penalty

li383 (16,32,32) p=2 mixed from the VMEC IC, harmonic atom, 100 MINRES iterations, NO
floor stop and no dt floor (the behaviour past the resolved floor is the point), F2 the
squared normalised residual (chunk means), E_rem the energy removed, dH/H the helicity
drift. The released configuration (kappa = 3, C = 0.1, alpha = 0) is
newton_sweeps_2026-09-13/h16: floor 4.7e-9 at step 39 (floor stop).

**Released atom (kappa = 3) + alpha**, 200 steps, jobs 18648564-66,
`outputs/newton_second_variation/spectrum/arms/`:

| arm | F2@60 | F2@100 | F2@140 | F2@200 | E_rem | dH/H | dt |
|---|---|---|---|---|---|---|---|
| C 0.1, alpha 1 | 1.77e-9 | 6.65e-10 | 3.61e-10 | 2.19e-10 | 1.98e-6 | -1.0e-5 | 1.000 every step |
| C 0, alpha 1 | 1.77e-9 | 6.66e-10 | 3.61e-10 | 2.19e-10 | 1.98e-6 | -1.1e-5 | 1.000 every step |
| C 0, alpha 0 (control) | 7.8e-9 | 7.9e-9 | 7.5e-9 | 1.34e-8, climbing | 2.90e-6 | +6.5e-5 | 0.57 .. 1 |

The penalty arms descend 20x past the old floor and are still descending at step 200;
C is irrelevant to every digit (the regularised search never binds at dt = 1). The
control is the failure mode in the run: it floors at 8e-9, then climbs, removes 50 %
more energy, and the helicity RISES - the parallel-flow garbage going through the mesh.

**Physical (strain) floor + alpha**, the peer session's arms (worktree
`.claude/worktrees/newton-atom-smoothing`, branch `worktree-newton-atom-smoothing` on
64ffbd9, `outputs/atom_study/`; atom floor = lumped strain + alpha, `--harmonic-floor
strain`, field h or B), 100 steps, jobs 18648567-71:

| arm | min F2 (step) | F2@30 | F2@60 | F2@90 | E_rem | dH/H | dt | fallbacks |
|---|---|---|---|---|---|---|---|---|
| B strain, alpha 0.3, C 0 | 2.50e-10 (100) | 1.24e-9 | 5.08e-10 | 2.72e-10 | 2.00e-6 | -1.0e-5 | 1.000 | 0 |
| B strain, alpha 1, C 0 | 5.58e-10 (100) | 3.34e-9 | 1.31e-9 | 6.68e-10 | 1.96e-6 | -9.8e-6 | 1.000 | 0 |
| B strain, alpha 3, C 0 | 2.70e-9 (100) | 8.37e-9 | 4.51e-9 | 3.03e-9 | 1.88e-6 | -9.6e-6 | 1.000 | 0 |
| h strain, alpha 1, C 0 | 5.59e-10 (100) | 3.29e-9 | 1.30e-9 | 6.69e-10 | 1.95e-6 | -9.7e-6 | 1.000 | 0 |
| B strain, alpha 1, C 0.1 | 5.58e-10 (100) | 3.34e-9 | 1.31e-9 | 6.68e-10 | 1.96e-6 | -9.8e-6 | 1.000 | 0 |
| B strain, alpha 0, C 0 (control) | 4.82e-8 (19) | 1.1e-7 | 6.9e-8 | 6.7e-8 | 2.77e-6 | +3.6e-5 | 0.95 | 0 |
| h strain, alpha 0, C 0.1 | 6.76e-9 (54) | 1.1e-8 | 6.8e-9 | 7.8e-9 | 1.98e-6 | -9.3e-6 | 0.11 | 6 |
| h kappa 3, alpha 0, C 0.1 (released) | 3.85e-9 (65) | 5.8e-9 | 3.9e-9 | 4.5e-9 | 1.97e-6 | -1.1e-5 | 0.25 | 0 |

Same energy removed and helicity drift as the released configuration, i.e. the same
equilibrium, resolved further: the 4e-9 "resolved floor" of September was the method's,
not the mesh's. Smaller alpha is better down to 0.3 (0.3 < 1 < 3, ~2x each); alpha 0
climbs. h and B profiles identical at alpha 1 (li383 is 96 % harmonic). With the
strain floor and no penalty the direction is exact enough on the continuum to descend
the grid-scale energy (min at step 15-19, then up, 40 % more energy removed, 100x the
helicity drift): kappa = 3 was the Levenberg-Marquardt damping. Smoothing the Newton
potential (mu = 0.02/n_r^2) is a negative in both configurations (fallbacks, MINRES
converging to nothing).

Peer's round three (jobs 18648612-15): alpha {0.03, 0.1, 0.3} at 200 steps, and
(32,64,64) at alpha 0.3.

**Open before a default change: the surfaces.** On W7-X, Newton past the floor was what
destroyed the surfaces (2026-09-12), and no state 20x past the old floor has been traced
yet. Poincare traces of the alpha 1 and the control final states: jobs 18648630/31
(`arms/c0_a1`, `arms/c0_a0`), section 7.

Suggested default if the sections are clean (peer's proposal, I agree):
`--newton-parallel-penalty 0.3 --harmonic-floor strain --step-regularisation 0`, dt
floor off; kappa deleted, the regularised search kept for the reconnection series at
most.

## 7. Surfaces past the old floor

Poincare sections (160 lines, 400 periods, five planes; `arms/<arm>/poincare/`):

| state | F2 | chaotic lines | iota |
|---|---|---|---|
| h16 ic (VMEC) | - | 0 / 160 | 0.395 .. 0.660 |
| h16 best, released, step 39 | 4.7e-9 | 4 / 160 (r 0.55, 0.72, 0.80, 0.86) | 0.394 .. 0.660 |
| C 0, alpha 1, step 200 | 2.2e-10 | 3 / 160 | 0.393 .. 0.660 |
| peer: B strain, alpha 0.3, step 100 | 2.5e-10 | 3 / 160 (r 0.54, 0.67, 0.80) | 0.394 .. 0.660 |
| C 0, alpha 0 control, step 200 | 1.3e-8 | **51 / 160** | flat at 1/2 over r = 0.6 .. 0.7, scattered outside |

The penalty states 20x past the old floor are as clean as the released floor state:
the same nested structure, the same small chains at 1/2 (r ~ 0.55) and 3/7 (r ~ 0.28),
the same wavy edge outside r 0.75, the same iota profile. The unregularised control is
stochastic outside r ~ 0.4 with iota flattened at 1/2 - the W7-X picture of 2026-09-12
(Newton past the floor destroying the surfaces), reproduced on li383 by taking the
damping away. So the surface destruction was the parallel-flow null space, not the
depth of the descent.

One number moved and is not understood: the regular-line drift is 4.8e-3 (alpha 1) and
5.2e-3 (alpha 0.3) against 8.8e-4 for the September best; the sections do not show what
it measures. Open, one sentence's worth. The (32,64,64) alpha 0.3 arm (peer, job
18648615) is the stricter test, since that is the mesh where W7-X lost its surfaces.

## 8. Summary

* The Hessian's soft end is a continuum of field-aligned flows u = f B, the discrete
  remnant of its exact null space, at the axis and on the rational surfaces with the
  resonant helicity, island or not. Not resonant outliers. Krylov extremes cannot see
  this; LOBPCG with the atom can.
* The force puts round-off on them and Newton divides by 1e-4..1e-3: an exact step
  would be parallel-flow garbage 10-100x the physical step. kappa = 3 in the atom and the
  100-iteration truncation were the Levenberg-Marquardt damping in disguise; eps = C h_r^2
  is the same size as the eigenvalues and does little.
* alpha M_par in the operator (Levenberg-Marquardt on the parallel component alone)
  lifts the null space, leaves the descent unchanged, and lets Newton go 20x past the
  September floor at dt = 1 with the surfaces intact; with the atom's floor at the
  physical strain, alpha 0.3 is the best measured (peer). Candidate default:
  `--newton-parallel-penalty 0.3 --harmonic-floor strain --step-regularisation 0`, dt
  floor off; pending the (32,64,64) sections and Tobias.
