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

(filled in from job 18648058)
