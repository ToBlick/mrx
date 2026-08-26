# Coarse GVEC exports and the Clebsch initial condition -- 2026-08-26

Question: how small can the example GVEC file in the repository be, and what
regularises a coarsely sampled initial condition? Reference: `w7x_fmm002`
(50^3, 27 MB), relaxed on `(8,16,8)` p=3, float64, 300 CG steps, Clebsch IC;
sections at zeta = 0 after 400 periods. All runs in `outputs/trim/` of the
`main-next` worktree (relax.json, B.h5, poincare/).

**2026-08-27, section 5:** the question is moot for equilibria with a GVEC
state file -- MRX reads `GVEC_State_*.dat` in closed form, 330 KB, and the
initial force residual drops 10x against the 50^3 export at (16,32,32).

## 1. Trimming and grids

MRX reads `eval_points, R, Z, pressure, clebsch/{dPhi_dr, dchi_dr, LA}`
(9 doubles per point); the rest of a full export (`B`, `beta`, `Phi`, `chi`,
`dLA_*`, `grad_*`) is 16 of 25 columns. `scripts/trim_gvec_export.py` writes
the read set gzip-compressed and, with `--n-*`, resamples every field through
the data-node spline fit onto a new grid (`--radial edge` refines toward the
wall; `fit_scalar_spline` now places its knots from the sample, so a graded
grid is a valid input).

| file | size | ||J||/||B|| at IC | H_0 | relaxed core |
|---|---|---|---|---|
| 50^3 (9 columns, gzip) | 2.6 MB | 0.291 | -1.84e-4 | reference; bit-identical data, trace agrees to 2.5e-7 in E |
| 33^3 uniform / edge | 0.82 MB | 0.312 | -4.5e-4 | intact; small axis iota dip (0.87); 20/19 chain at the wall |
| 20x33x33 | 0.51 MB | 0.313 | -4.6e-4 | = 33^3 to 3 digits |
| 40x20x24 | 0.45 MB | 0.407 | -4.0e-4 | chaotic core (10/11) |
| 20^3 uniform / edge | 0.20 MB | 0.418 | -7.8e-4 | chaotic core, pressure plateau, beta_vol 16% low |

Findings. (i) The core survives iff ||J||/||B|| at the IC stays near the
50^3 value; that number tracks the ANGULAR sample (20 -> 0.41, 33 -> 0.31),
while H_0 tracks the overall resolution and has not converged even at 33^3.
(ii) Radial grading (uniform vs edge-refined) changes nothing in the
relaxation (H_0, F, sections identical), although it cuts the interpolation
error of d_rho d_theta LA in the outer 20% of the radius by 8x -- that error
is not what makes the current. (iii) Two identical 20^3 runs differ in which
core surfaces survive (GPU run-to-run noise over 300 steps): at 20^3 the
core is marginal, and section-to-section differences below that are not
evidence. (iv) The file is Fourier-truncated by GVEC: nothing above m, n ~ 16
in the angles, so >= 33 angular samples reproduce it exactly.

## 2. What the excess current is

`lambda_dirichlet_energy` (<lam, L_0 lam> of the 0-form interpolant) is
18.2-18.5 on EVERY grid, 50^3 to 20^3: at mesh resolution lambda is the same.
The excess current of the projected route came from differentiating the
data-grid spline interpolant pointwise at the quadrature points (autodiff of
`lam_h`), which sees the wiggles between data nodes; the interpolation error
of d_rho d_theta LA from 20 radial samples is 3% rms / 10% max and no
smoother fit (least squares, smoothing spline) gets closer to the 50-point
truth. The remedy is structural, not a filter:

**`B = dA'`** (`clebsch_potential_form` + `potential_two_form`,
`--ic clebsch`). `A' = (-LA dPhi_dr, 2 pi Phi, -(2 pi/nfp) chi)` needs only
VALUES of lambda; histopolated on the free 1-form space (the wall trace
2 pi Phi_edge is the toroidal flux, the Dirichlet harmonic content) and
curled by the exact incidence operator into the Dirichlet 2-form space.
div B = 0 to 1e-16, no Leray step, and B^rho = 0 on EVERY rho-face exactly
(the circulation of A' around a rho-face vanishes: both tangential
components are functions of rho alone), so the IC is exactly nested --
tracer h/2 drift 2.8e-6 vs 3.4e-5 for the projected IC -- and the 20^3 IC
is indistinguishable from the 50^3 one.

| route, file | ||J||/||B||_0 | F_0 | F_final | dH/H_0 (one mesh, H_0 = -1.9e-4; see 3b) | beta_vol | core |
|---|---|---|---|---|---|---|
| projected, 50^3 | 0.291 | 1.89e-2 | 8.4e-4 | +8.5e-4 | 1.30e-2 | intact |
| potential, 50^3 | 0.261 | 1.55e-2 | 5.8e-4 | -7.1e-4 | 1.29e-2 | intact |
| projected, 33^3 | 0.312 | 1.79e-2 | 1.0e-3 | +5.9e-4 | 1.23e-2 | intact, axis dip |
| potential, 33^3 | 0.284 | 1.52e-2 | 6.5e-4 | +2.8e-4 | 1.22e-2 | intact |
| projected, 20^3 | 0.418 | 2.11e-2 | 1.3e-3 | +1.1e-3 | 1.09e-2 | chaotic |
| potential, 20^3 | 0.394 | 1.87e-2 | 1.0e-3 | +1.3e-3 | 1.07e-2 | **intact, = 50^3 core** |

The potential route is better on every gauge at every resolution and is the
production IC as of this note. Open: every potential run relaxes to a ~4 cm
core with iota = 1 (a real 5/5 structure, not the diagnostic -- a re-seeded
probe still orbits 3 x 12 cm) that the projected runs do not have and the
potential IC does not have either (it starts at GVEC's 0.915); the two
relaxed fields agree to 0.1% of |B| there, which is half the poloidal
field. RESOLVED (OPEN 3.10): at (12,24,12), 2000 steps, the core is GVEC's
flat 0.916 with a small 5/5 edge chain and ||F|| 1.9e-4 -- the iota = 1 core and
the wide islands are (8,16,8) artefacts; the innermost ~4 cm there are
mesh-limited (gamma = 1 turns the same core into an axis dip instead).

## 3. Regularisers tried on the 20^3 file (projected route)

| arm | ||J||/||B||_0 | result |
|---|---|---|
| CFL 0.01 (bound on 27/300 steps) | 0.418 | no change; CFL 0.03 never binds (max CFL number 0.028) |
| velocity smoothing gamma=1, mu=1e-2 | 0.418 | laminar-looking surfaces but F up 30%, beta_vol -35%, dH/H_0 8x, p_w not a surface function: unconverged |
| **`--presmooth`** (force-free backward Euler on B, eps 1e-3 x1 or 3e-4 x2, stop at ||J||/||B|| <= 0.3) | 0.27-0.29 | core = 50^3, beta_vol restored to 1.31e-2, H_0 untouched; small iota rise (0.96) on the innermost lines |
| lambda smoothing (M_0 + h L_0)^-1 M_0 | h=1e-4 x1: 0.391; x2, x3: 0.43, 0.47; h=1e-3..1e-2: 0.59-1.22 | shallow optimum, then it ADDS current (moves lambda away from the equilibrium's, whose Dirichlet energy is HIGHER); at h=1e-2 the relaxed field is stochastic. Removed from the code. |

lambda = 0 is not the vacuum field: it is the field whose lines are straight
in the map's angles, which in a shaped stellarator carries the full
unbalanced current (||J||/||B|| ~ 1.2); the equilibrium's lambda is what
makes J x B ~ grad p. Smoothing lambda moves toward that field.

## 3b. Long runs at (12,24,12): gamma = 1, mu ~ h^2, finite resistivity

All on the potential IC from the 50^3 file, 2000 steps, CFL cap 0.5 (never
binds), floor reached (||F|| flat over the last ~500 steps). The reference
is `outputs/trim/relax50pot_hires2000` (`gamma = 0`): F_final 1.75e-4,
E_0 - E 2.38e-5, dH +8.8e-8, beta_vol 1.149e-2, beta_axis 3.43e-2,
nested surfaces, flat iota_axis 0.916 (GVEC's), small 5/5 edge chain.

**Helicity is quoted in absolute units from here on** (`||B||_M = 1`,
`H = <A, B + B_harm>` with the Dirichlet potential, `compute_helicity`).
The field's own helicity is a near-cancellation: it vanishes at constant
iota, W7-X's shear is small, and the discrete value of the SAME IC is
-1.9e-4, -2.9e-5, -5.4e-6, +3.5e-6, +6.3e-6 on (8,16,8), (12,24,12),
(16,32,16), (12,24,24), (16,32,32) -- converging toward zero with a sign
change. Every `dH/H_0` quoted earlier in this note and in OPEN 3.11 divided
by that vanishing number and is misleading across meshes (2026-08-27,
`outputs/trim/traces_all_HF_time.png`).

| arm | F_final | E_0 - E | dH = H_final - H_0 | beta_vol | section | wall |
|---|---|---|---|---|---|---|
| ideal, gamma = 0 | 1.75e-4 | 2.38e-5 | +8.8e-8 | 1.149e-2 | nested, iota_axis 0.916 | 1472 s |
| eta_max 1e-6, tanh | 1.74e-4 | 2.38e-5 | +1.1e-7 | 1.148e-2 | identical to ideal | 1797 s |
| eta_max 1e-4, tanh | 1.35e-4 | 2.44e-5 | **+2.5e-6** | 1.096e-2 | nested, same core; 5/5 only an iota plateau, edge slivers marginally clearer | 1818 s |
| gamma = 1, mu = 4.4e-4 | 1.34e-4 (floor, t = 40) | 2.26e-5 | -9.9e-8 (jump -8.5e-8 by step 100) | 1.166e-2 | nested, edge 5/5 + 20/19, **axis dip to iota 0.910**, flat-top p_w | 3388 s |
| gamma = 1, mu = 1e-3 | 1.36e-4 (floor, t = 77) | 2.24e-5 | +1.1e-7 (jump +1.2e-7 by step 100) | 1.158e-2 | same as mu = 4.4e-4 | 3254 s |

gamma = 1 on this mesh (`outputs/trim/traces_mu_h2.png`): both mu arms
reach the floor detector at 1.34e-4 by t = 40-77, where the gamma = 0 run,
stopped by its step count at t = 5.7, is still descending at 1.75e-4 -- same
energy, beta_vol +1.5% (was +8% on (8,16,8)), ||J||/||B|| at the floor 0.061
vs 0.054 (was 0.078 vs 0.059), both mu indistinguishable except in the sign
of the helicity jump. The helicity traces differ in kind: gamma = 1 jumps
once in the first ~100 steps and is then conserved to the last digit shown
over t = 40-77, gamma = 0 drifts continuously (+8.8e-8 by t = 5.7 and
still moving); the two errors are the same size (~1e-7) on this mesh. The iota profile dips at the axis on every gamma = 1
run on both meshes (0.910 here, GVEC 0.915), where gamma = 0 is flat; the
smoothing solve (M_2 + mu L_2)^-1 acts on the polar patch (m <= 1) differently
from the bulk, so the dip is a mesh-scale artefact of the smoothing, not an
equilibrium feature.

**Finite resistivity.** eta_max = 1e-6 is below the numerical floor of
this discretisation: force, energy, pressure and section are those of the
ideal run to three digits; the only trace is the helicity, whose extra
drift (+2e-8 over t = 5 on top of the ideal run's +8.8e-8) puts the
resistive dissipation at ~1/4 of the numerical one, and the resistive solve `(M_2 + eps L_2) delta = -eps L_2 B` costs
~270-300 MINRES iterations per step (+22% wall). At eta_max = 1e-4
(`outputs/trim/relax50pot_hires2000_eta1e-4`, 2026-08-27) the force floor
is 25% lower and beta_vol 4.6% lower, the helicity drift is 2.5e-6 (28x
the ideal run's), and the
islands are NOT visibly wider: the section is nested with the same core,
the 5/5 chain at rho ~ 0.8 is an iota plateau on the outboard side and the
25/24 and 20/19 edge slivers are marginally clearer. A 100x resistivity
reconnects nothing because there is nothing resonant to reconnect: island
width is set by the resonant drive (Pfirsch-Schlueter current ~ beta,
boundary shape), not by how much reconnection is allowed. The seeded-island
sweep (section 3d) measures that drive.

**GVEC's own resolution.** GVEC ran the case at m, n <= 16 Fourier modes,
i.e. 32 angular points per period; `outputs/trim/gvecres32_g0` is the
(16,32,32) p=3 gamma = 0 run of the same IC (3000 steps, 2.4 h):

| mesh | steps, t | mean dt | F (last 50) | E_0 - E | dH (abs) | beta_vol | beta_axis | section |
|---|---|---|---|---|---|---|---|---|
| (8,16,8) | floor, 24.8 | 0.022 | 1.58e-4 | 1.02e-4 | +5.2e-7 | 1.183e-2 | -- | iota = 1 core, wide islands |
| (12,24,12) | 2000, 5.7 | 0.0028 | 1.75e-4 | 2.30e-5 | +8.8e-8 | 1.149e-2 | 3.43e-2 | nested, flat 0.916 core, small 5/5 |
| (16,32,32) | 3000, 3.6 | 0.0012 | 1.77e-4 | 1.48e-5 | -2.0e-8 | 1.115e-2 | 3.39e-2 | nested at zeta = 0 and 0.5, flat 0.916 core, 5/5 + 25/24 + 20/19 slivers, outer ~15% weakly stochastic (h/2 drift 8e-3; not at the floor) |

The core is GVEC's on every mesh from (12,24,12) up; what refinement buys
is a smaller energy drop from the IC (1.0e-4 -> 2.3e-5 -> 1.5e-5, the IC is
closer to the discrete equilibrium) and a beta_vol that falls 3% per
refinement level toward the GVEC value. The price is the time step: the
line-search dt scales like h^2 (0.022 -> 0.0028 -> 0.0012), so reaching the
same relaxation time costs h^-5 -- (16,32,32) needs ~10^4 steps (8 h) for the
t = 5.7 of the (12,24,12) run. Relaxation at GVEC's resolution is a
gamma = 1 job, not a gamma = 0 one.

**mu ~ h^2 (`outputs/trim/traces_mu_h2.png`).** The arms were designed at
fixed mu / h^2 = 0.064 in logical units: mu = 1e-3 on (8,16,8), 4.4e-4 on
(12,24,12), 2.5e-4 on (16,32,16). The early helicity jump, the excess current
and the excess beta over gamma = 0 measure what the smoothing does to the
answer:

| mesh | mu | mu / h^2 | gamma = 1 helicity jump (abs) | gamma = 0 drift on the same mesh (abs) | excess ||J||/||B|| | excess beta_vol |
|---|---|---|---|---|---|---|
| (8,16,8) | 3e-4 | 0.019 | -4.1e-7 | +5.2e-7 (floor) | +32% | +8% |
| (8,16,8) | 1e-3 | 0.064 | (300 steps only) | | -- | -- |
| (12,24,12) | 4.4e-4 | 0.064 | -8.5e-8 | +8.8e-8 (t = 5.7) | +13% | +1.5% |
| (12,24,12) | 1e-3 | 0.144 | +1.2e-7 | +8.8e-8 | +13% | +0.8% |
| (16,32,16) | 2.5e-4 | 0.064 | -9.2e-8 | -8e-9 (t = 2.2; max excursion 2.9e-8) | +29% (gamma = 0 twin not at its floor) | -0.8% |

In absolute helicity the gamma = 1 jump shrinks with h at fixed mu / h^2
(4.1e-7 -> 0.9e-7 -> 0.9e-7; the (8,16,8) point is mu = 3e-4, the closest
long run) and stays within a factor 1-3 of the gamma = 0 drift on the same
mesh; the earlier reading of a "non-monotone jump" was the relative number
divided by an H_0 that itself falls 35x over these meshes. It is not a
step-size effect (the (16,32,16) run has the smallest early dt of the
gamma = 1 runs and the same absolute jump as (12,24,12)). What does NOT
shrink is the iota dip at the axis (0.910-0.912 on all three meshes) and
the excess current at the floor. Verdict on mu ~ h^2: consistent with the
helicity data (a slowly converging error of the size of the gamma = 0
drift), not established by it -- three meshes a factor 2 apart with one
point each cannot fix an exponent -- and silent on the axis dip, which is
the actual gamma = 1 artefact to fix. What the three meshes agree on:
gamma = 1 reaches the force floor at the same energy as gamma = 0, with a
one-time helicity error of the size of gamma = 0's cumulative one and then
exact conservation, 8-16x larger stable steps, and the axis dip. OPEN 3.11
holds the dip and the jump mechanism.

## 3c. The relaxation movie

`outputs/movie/relax50pot_12x24x24/relaxation_zeta0.5.{mp4,gif}` (H.264
3100x960 at 8 fps, 43 s; 6 MB gif): (12,24,24) p=3, gamma = 0, potential IC,
`--save-every 2`, floor detector at step 1269 (t = 4.9, ||F|| 1.33e-4,
beta_vol 1.137e-2); 347 frames = steps 0-498 every 2 and 500-1268 every 8,
each a zeta = 0.5 section of 120 lines x 400 crossings with the RZ window,
the iota/pressure axes and the pressure colour range pinned to global
limits (`scripts/poincare_relax.py --fields snapshots`, `render_section
(limits=...)`). What it shows: the IC is nested; the first ~100 steps
scatter the outer half while the pressure peak rises 0.18 -> 0.28 x 1e-2
(the fast energy drop); the surfaces then heal from the axis outward and the
peak settles at 0.26; the core stays flat at 0.916 throughout. The
(8,16,8) proof-of-concept movie (`outputs/movie/relax50pot/`, 31 frames)
is superseded.

## 3d. Seeded islands: the 5/5 chain is tearing-stable

`scripts/relax.py --seed 5,1,0.8,0.1 --seed-eps EPS` adds the resonant
term `eps |Phi'(rho0)|/m g(rho) cos(2 pi (5 theta + zeta))` to `A'_zeta`
(the sign follows the file's iota < 0) before `B = dA'`, so div B and the
wall condition stay exact and `eps` is the resonant normal field
`|dB^rho|/|B^zeta|` at the chain, which the file's profile puts at
rho = 0.831. Pendulum estimate of the seeded full width,
`1.6 sqrt(eps nfp / (m |iota'|))` with `|iota'| = 0.31`: 2.9%, 5.7%, 11.5%
of rho for eps = 1e-4, 4e-4, 1.6e-3. (12,24,24) p=3, gamma = 0, no
resistivity, run to the floor detector (`outputs/trim/seed_eps*`, sections
at zeta = 0 and 0.5 for the IC and the floor, `traces_seed.png`):

| eps | seeded chain at t = 0 (zeta = 0.5) | at the floor | steps, t | F (last 50) | E_0 - E | dH (abs; H_0 = +3.5e-6) | beta_vol |
|---|---|---|---|---|---|---|---|
| 0 (movie run) | none | small 5/5, iota plateau ~1-2 cm | 1269, 4.9 | 1.33e-4 | 1.31e-5 | +6e-9 | 1.137e-2 |
| 1e-4 | barely visible | indistinguishable from unseeded | 1256, 4.9 | 1.33e-4 | 1.31e-5 | +4e-9 | 1.138e-2 |
| 4e-4 | clear chain, ~2-3 cm | same small chain as unseeded | 1261, 4.9 | 1.32e-4 | 1.31e-5 | +4e-9 | 1.137e-2 |
| 1.6e-3 | wide chain, ~5 cm (of ~80 cm outboard) | shrunk to the same small chain | 1260, 4.9 | 1.34e-4 | 1.31e-5 | +3e-9 | 1.137e-2 |
| **1e-2** (seed width 0.2) | five O-points, iota plateau ~15 cm (~19% of the outboard minor radius; pendulum 29% of rho) | **persists at the same width**, p_w flat across it, thin stochastic layer to the wall | 2500 (not at the floor), 9.2 | 2.46e-4 | 1.33e-5 | +7e-8 | 1.131e-2 |

The seed does what the estimate says at t = 0 (widths visibly ~ sqrt(eps),
at the predicted surface). What happens next has two regimes, set by the
mesh: the three seeds up to 1.6e-3 (2.9-11.5% of rho against a radial cell
of 8% at n_rho = 12) are squeezed back to the width the unseeded run has,
and their final states are eps-independent and identical to the unseeded
one in E, ||F||, beta and helicity to three digits; the 1e-2 seed (29% of
rho, ~3.5 cells) keeps its width through 2500 steps, the pressure flattens
across the island and a thin stochastic layer forms between the island and
the wall (`seed_eps1e-2/poincare/poincare_final_zeta0.5.png`) -- an MHD
equilibrium with a 5/5 island, force floor still descending at 2.5e-4
(current sheets at the separatrix). Read: under exact ideal dynamics no
seeded island can disappear (topology is frozen), so the small ones were
removed by the numerical reconnection of the (12,24,24) mesh, whose floor
is one to two radial cells; a resolved island is neither healed nor grown
by the descent -- the 5/5 surface is tearing-STABLE (no free energy for
growth, which is also why resistivity in section 3b reconnected nothing)
but an island of any resolved width is an admissible equilibrium of the
same file. Widths were read off the sections by eye (the outboard iota
plateau at 5/5 in the midplane profile); a crossing-based width measurement
is the missing piece if the sqrt(eps) law is to be quoted as a number.
Practical rule: seed at least ~3 radial cells wide, or refine. To get
islands that GROW the drive has to change: a higher-beta export (the
`w7x_ini_conv` file, beta_axis 10.6%, run 2026-08-27), a resonant (5,5)
boundary deformation, or a vacuum region beyond the LCFS.

## 4. Recommendation

Superseded by section 5 wherever a GVEC state file exists: pass the
`.dat`. For grid-only exports: the potential route (done), the
`--presmooth` option, and >= 33 angular samples; 20x33x33 at 0.5 MB
behaves as 33^3. Gauges to print for any new source: ||J||/||B|| at
t = 0, ||F||_0, H_0, and their convergence in the mesh.

## 5. The closed-form route supersedes the export (2026-08-27)

`mrx/gvec_state.py` reads GVEC's own `GVEC_State_*.dat` (330 KB for
FMM002): R, Z and lambda as degree-5 radial B-splines x Fourier series, the
profiles at the 15 radial interpolation points. Evaluated at the export's
grid it reproduces the pyGVEC h5 to 2e-14 (R, Z, lambda, Phi'), so nothing
was lost in the export -- but the way MRX CONSUMED the export was lossy:
the map collocated R, Z at its Greville points through a linear
`RegularGridInterpolator` bridge over the 50^3 grid, and lambda went
through a data-node spline fit. With the series MRX evaluates both where
it needs them (`StateField`, JAX, exact under autodiff). Same mesh, same
code, 300 steps of gamma = 0 (`outputs/trim/dat_*`, `traces_dat_vs_h5_*.png`):

| | (12,24,24) grid | (12,24,24) closed form | (16,32,32) grid | (16,32,32) closed form |
|---|---|---|---|---|
| ||J||/||B||_0 | 0.111 | **0.080** | 0.167 | **0.055** |
| ||F||_0 | 4.2e-3 | 2.7e-3 | 7.1e-3 | **7.5e-4** |
| beta_vol at t = 0 | 5.4e-3 | 1.06e-2 | 5.9e-3 | 1.09e-2 |
| ||F|| after 300 steps | 2.0e-3 | 1.6e-4 | 2.4e-3 | 1.3e-4 |
| E_0 - E after 300 steps | 1.2e-5 | 8e-7 | 1.4e-5 | 4e-8 |
| det DF range (map) | [0.097, 18.5] | [0.097, 19.0] | | |

Read: the grid route's current ROSE with refinement because the mesh was
resolving the bridges' interpolation noise, not the equilibrium; the
closed-form current falls with h, the IC starts at GVEC's beta (the grid
IC started at half of it and had to relax up), and 300 steps reach the
force floor that took the grid runs 1300-3000 steps -- at (16,32,32) the
IC releases 300x less energy, i.e. it is the discrete equilibrium up to
the mesh's own approximation. Sections (`dat_*/poincare/`, both planes,
IC and final, 160 lines x 400 periods): at (16,32,32) the IC and the
300-step state are nested to the wall -- 0 lines lost, 0 chaotic, no 5/5
chain, no stochastic outer band, p_w a surface function -- where the grid
route's (16,32,32) run had 5/5 + 25/24 + 20/19 slivers and a weakly
stochastic outer 15% after 3000 steps; at (12,24,24) 0 chaotic at t = 0
and 2 marginal lines at the end. The edge structure of every h5-based run
was the bridges' noise. `dat_16x32x32_long` (4000 steps, floor-tol 1e-5)
follows the force below the 300-step floor.
Consequences: the example file for the repository is the `.dat`; the
grid-sampling study of sections 1-3 (Nyquist rule, 20^3 vs 33^3 vs 50^3,
presmoothing) applies only to exports that have no state file behind
them; every W7-X number in this note obtained through the h5 carries the
bridge's floor (||F||_0 ~ 4e-3 to 7e-3) that the closed form does not.
`--presmooth` and `trim_gvec_export.py` stay for grid-only sources.

