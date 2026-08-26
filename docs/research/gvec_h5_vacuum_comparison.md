> **Status:** current
> **Read this for:** comparing an MRX vacuum field against a GVEC h5 export, and why the simsopt files are recoverable
> **Do not read for:** the projection recipe (w7x_vacuum_bfield_handoff.md) or the Clebsch IC route (relaxation_ic_2026-08-25.md)

# Comparing MRX vacuum fields to GVEC (h5-only)

Status: 2026-08-17. Script: `scripts/debug/w7x_vacuum_bfield_project.py`.
Companion to `docs/research/w7x_vacuum_bfield_handoff.md` (which documents the
projection recipe and the geometry/frame traps, and which this note **corrects**
on the simsopt files -- see "The simsopt files are recoverable").

## What this answers

Does MRX's discrete vacuum field -- computed as the harmonic form of the FEEC de
Rham complex, with no reference to any external field -- agree with the vacuum
field GVEC produces?

Short answer: **yes on the true vacuum cases, to the expected spline order,
until the finite GVEC sample floors it.** On W7-X the agreement converges like
`O(h^4)` from `4.4e-3` down to `~2.5e-4`, then flattens to order `~0.75` as the
FEM out-resolves the stored 32^3 samples. On the two `quasr` GVEC equilibria the
error stalls at 1-2.5 % from the very first resolution, and that stall is *not*
discretisation error -- those stored fields are not curl-free, so they are not
harmonic forms at any resolution. See "The quasr floor".

Two things postdate the first draft of this note and change its conclusions:

- **The k=2 solve stops converging above `n ~ 18`** and its reported error above
  that point is meaningless. This is a solver failure, not a floor, and the
  convergence control is partly to blame. See "The k=2 solver failure".
- **Both simsopt files are rotated exactly one field period off their own R,Z**
  and are recoverable by counter-rotating the stored vectors, which drops
  `quasr0044970_simsopt_B` from 88 % error to 1.09 %. See "The simsopt files are
  recoverable".

## No GVEC installation is needed

Every comparison here reads a single self-contained h5. Five files in `data/`
were produced by GVEC and dumped to disk (attribute
`B_flux_source: gvec_state.evaluate`), so GVEC has already run; nothing at
comparison time imports it. The two `scripts/julianne/*.py` drivers, by
contrast, cannot run in this repo at all -- they import a `scripts/wip/` tree
plus `mrx.gvec_jax_map` / `mrx.io_nfs_map` that exist on another branch, and
their default path additionally wants the `gvec` and `simsopt` packages.

### Input contract

One h5, flat arrays in C order with zeta fastest, covering one field period:

| dataset / attr | shape | meaning |
| --- | --- | --- |
| `eval_points` | (N,3) | logical (rho,theta,zeta) in [0,1) |
| `R`, `Z` | (N,) | cylindrical position, physical units |
| `B` | (N,3) | Cartesian reference field [T] |
| `n_rho`/`n_theta`/`n_zeta` | attrs | grid shape (simsopt files spell these `precomputed_nr/ntheta/nzeta`) |
| `nfp` | attr | field periods |

Only `R`, `Z`, `nfp` feed the computation. `B` is used **solely** for the
comparison -- the eigenproblem never sees it, since the harmonic form is fixed
by the geometry and the Betti numbers alone. Its amplitude is not fixed
(a nullvector is scale-free; the physical scale comes from enclosed flux), so
every comparison is best-fit-scaled.

### Files

| file | grid | nfp | usable |
| --- | --- | --- | --- |
| `W7X-vacuum.h5` | 32^3 | 5 | yes, true vacuum (beta ~ 1e-13) |
| `w7x_vacuum_co_contra.h5` | 50^3 | 5 | yes, same equilibrium sampled finer |
| `quasr_0009983.h5` | 50^3 | 2 | GVEC equilibrium, beta = 0 -- but see below |
| `quasr_0044970.h5` | 50^3 | 3 | GVEC equilibrium, beta = 0 -- but see below |
| `quasr00*_simsopt_B.h5` | 8x16x8 | 3 / 2 | Biot-Savart; **both need `--rotate-b-periods 1`** |
| `W7-X.h5` | 50^3 grid-shaped | 5 | geometry only, different layout, needs its own loader |
| `gvec_nfp3_hegna_80cubed_clebsch.h5` | 80^3 | 3 | **not vacuum** (4000 Pa, carries `J`) |

## Pipeline

1. **Geometry.** Fit interpolatory tensor B-splines to `R`, `Z` (one square
   collocation solve per axis, `n_basis = n_data`), then build
   `F = (R cos a, sign*R sin a, Z)` with `a = 2 pi zeta / nfp`. `sign` is
   auto-picked so `det(DF) > 0` -- the only orientation requirement, but a hard
   one: `J < 0` makes M2 indefinite and CG returns NaN. Lengths are normalised
   to major radius 1.
2. **Harmonic form.** On betti `(1,1,0,0)` both the k=2 Dirichlet and k=1
   natural harmonic spaces are one-dimensional. Seeded by the constant
   *reference-frame* form -- `(0,0,1)`, i.e. `dzeta` for k=1 and `drho ^ dtheta`
   for k=2. That seed is purely topological; the geometry enters only through
   the `M_k^{-1}` applied to its dual load, which is what makes it valid on an
   arbitrary stellarator rather than just a toroid. It also carries zero flux
   through rho=1, so it already lies in the Dirichlet space.
3. **Comparison.** Push the form forward to Cartesian at the stored nodes,
   best-fit-scale, report relative error.

## Results

### W7-X convergence (`W7X-vacuum.h5`, 32^3, stride 1, p=3)

Mean relative error in the Cartesian frame, at `ns = (n, 2n, 2n)`, p=3. Columns
are the L2 projection of GVEC's own sampled `B` (which uses the stored field)
and the two harmonic forms (which do not). Each `rate` is the observed order
against its predecessor, `ln(e_prev / e) / ln(n / n_prev)`.

| n | V2 dofs | L2 projection | rate | harmonic k=2 | rate | harmonic k=1 | rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 4 640 | 4.3201e-3 | -- | 4.4507e-3 | -- | 3.8467e-3 | -- |
| 10 | 9 640 | 2.3637e-3 | 2.70 | 2.4609e-3 | 2.66 | 2.1427e-3 | 2.62 |
| 12 | 17 328 | 1.1767e-3 | 3.83 | 1.1898e-3 | 3.99 | 1.0080e-3 | 4.14 |
| 14 | 28 280 | 6.8353e-4 | 3.52 | 6.6793e-4 | 3.75 | 5.1849e-4 | 4.31 |
| 16 | 43 072 | 3.8633e-4 | 4.27 | 3.5125e-4 | 4.81 | 2.5326e-4 | 5.37 |
| 18 | 62 280 | 4.8179e-4 | -1.87 | 4.6892e-4 (*) | -2.45 | 2.4988e-4 | 0.11 |
| 20 | 86 480 | 4.0949e-4 | 1.54 | 4.4577e-4 (*) | 0.48 | 2.2236e-4 | 1.11 |
| 22 | 116 248 | 3.3966e-4 | 1.96 | 1.5808e-3 (*) | -13.28 | 2.0499e-4 | 0.85 |
| 24 | 152 160 | 2.1381e-4 | 5.32 | 3.2600e-3 (*) | -8.32 | 1.8712e-4 | 1.05 |

(*) k=2 solve did **not** converge at these resolutions; the numbers are
reported for the record but are not measurements of discretisation error. See
"The k=2 solver failure". The dof counts at n >= 18 are from the closed form
below, not read off those runs -- their logs were lost with the temp directory.

Overall order across the clean range n=8 -> 16: **3.48** (projection), **3.66**
(k=2), **3.92** (k=1).

Two regimes:

- **n <= 16, clean convergence** at `O(h^{p+1})` for p=3, and rising through the
  range (the first interval is polluted by the coarse-grid error not yet being
  asymptotic). The harmonic form and the projection of GVEC's own sampled field
  converge *together* and agree with each other to a few percent of their own
  value at every resolution. That is the strongest statement available here: two
  independent constructions, one using GVEC's field and one ignoring it
  entirely, land on the same answer.
- **n > 16, the finite-sample floor.** Read this off the k=1 column, which is the
  only one that stays converged throughout. Its rate collapses from 5.37 to
  0.11 / 1.11 / 0.85 / 1.05, i.e. **order 0.75 overall across n=16 -> 24**
  against ~4 below. The projection column flattens in the same place and goes
  non-monotone (rate -1.87, then 1.54, 1.96, 5.32 -- scattered around a floor
  rather than converging), which is what an independent construction hitting the
  same noise level should do. The floor for this file is around **2e-4**, and it
  is a property of the 32^3 stored sample, not of MRX.

#### Dof counts in closed form

For the polar sequence at `ns = (n, 2n, 2n)`, p=3, betti `(1,1,0,0)`, with the
k=2 Dirichlet constraint applied:

```
V2 (free) = 12 n^3 - 24 n^2 + 4 n
V1        = V2 + 6 n = 12 n^3 - 24 n^2 + 10 n
```

Exact on all five measured resolutions (4 640 / 9 640 / 17 328 / 28 280 /
43 072 and 4 688 / 9 700 / 17 400 / 28 364 / 43 168). The leading `12 n^3` is
just the three 2-form components at `n * 2n * 2n` each; the lower-order terms are
the polar-axis and Dirichlet reductions. Extrapolating:

| n | 18 | 20 | 22 | 24 | 28 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| V2 | 62 280 | 86 480 | 116 248 | 152 160 | 244 720 | 368 768 |
| V1 | 62 388 | 86 600 | 116 380 | 152 304 | 244 888 | 368 960 |

The same floor at a much higher level, on the *other* W7-X file at stride 2
(50^3 data subsampled to 25^3, so a coarser geometry fit of the same
equilibrium):

| n | V2 dofs | harmonic k=2 | rate | harmonic k=1 | rate |
| --- | --- | --- | --- | --- | --- |
| 8 | 4 640 | 5.278e-3 | -- | 6.1606e-3 | -- |
| 10 | 9 640 | 3.753e-3 | 1.53 | 4.3516e-3 | 1.56 |
| 12 | 17 328 | 2.833e-3 | 1.54 | 3.1396e-3 | 1.79 |
| 14 | 28 280 | 2.422e-3 | 1.02 | 2.6765e-3 | 1.04 |
| 16 | 43 072 | 2.180e-3 | 0.79 | 2.3619e-3 | 0.94 |

The rate never reaches the p+1=4 of the stride-1 case -- it starts at ~1.5 and
decays monotonically to ~0.8, i.e. this case is already floor-limited at n=8 and
never has a clean asymptotic regime. It decelerates toward ~2e-3, an order of
magnitude above the stride-1 floor. Same equilibrium, same solver; the only
difference is 25^3 versus 32^3 geometry samples. That is a clean demonstration
that the floor tracks sampling density, and it is why the stride-1 runs are the
interesting ones to push. (Both harmonic forms agree here, k=2 and k=1 tracking
each other to ~10 %, because n <= 16 is below the k=2 solver failure.)

### Cross-geometry at ns=(8,16,16)

| file | nfp | harmonic k=2 | harmonic k=1 | projection | non-harmonic amplitude of GVEC B |
| --- | --- | --- | --- | --- | --- |
| `W7X-vacuum` (32^3, stride 1) | 5 | 4.45e-3 | 3.85e-3 | 4.32e-3 | -- |
| `w7x_vacuum_co_contra` (50^3, stride 2) | 5 | 5.28e-3 | 6.16e-3 | 1.78e-2 | 1.23e-2 |
| `quasr_0044970` (50^3, stride 2) | 3 | 1.334e-2 | 1.364e-2 | 1.498e-2 | 1.85e-2 |
| `quasr_0009983` (50^3, stride 2) | 2 | 2.384e-2 | 2.271e-2 | 1.188e-2 | 2.61e-2 |

The two W7-X files agree with each other (4.45e-3 vs 5.28e-3) despite being
independent samplings of the same equilibrium at different densities -- a useful
consistency check. Note also that the *projection* degrades sharply on the
strided file (1.78e-2 vs 4.32e-3) while the harmonic form barely moves: the
projection has to represent the sampled `B` and so is sensitive to sampling
density, whereas the harmonic form never touches it.

### The quasr floor

Refining the FEM while holding the data fixed, mean relative error of the
harmonic k=2 form (stride 2):

| n | `w7x_vacuum_co_contra` (vacuum) | rate | `quasr_0009983` | rate | `quasr_0044970` | rate |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 5.278e-3 | -- | 2.3841e-2 | -- | 1.3341e-2 | -- |
| 12 | 2.833e-3 | 1.53 | 2.3660e-2 | 0.02 | 1.2878e-2 | 0.09 |
| 16 | 2.180e-3 | 0.91 | 2.3540e-2 | 0.02 | 1.2663e-2 | 0.06 |
| change | **-59 %** | | **-1.3 %** | | -5.1 % | |

The quasr rates are 0.02-0.09 -- indistinguishable from zero, across a 2x
refinement. The L2 projection of the same fields *does* keep improving over that
range (`quasr_0009983`: 1.188e-2 -> 9.76e-3 -> 8.77e-3, rates 0.49 and 0.37), so
the FEM is not saturated -- only the harmonic form is. A flat error under
refinement while a projection on the same space keeps falling is the signature
of a modelling mismatch, not a numerical one.

Note that the projection rates on the quasr files are themselves well under 1,
far from the p+1=4 seen on W7-X. That is expected and not a second problem:
those files are stride-2 subsamples, so the projection is fitting a coarsely
sampled field and is sampling-limited in the same way the stride-2 W7-X
projection was (1.78e-2 versus 4.32e-3 for stride 1).

The reading is that the quasr files are GVEC *equilibria* with beta = 0 rather
than vacuum fields: zero pressure still permits a force-free field carrying
current, and a field with current is not curl-free, hence not representable as
a harmonic form at any resolution. The W7-X files are explicitly vacuum and
converge.

### Harmonic content: a good absolute check, a bad trend

Projecting `B` onto the same Dirichlet V2 the harmonic form occupies and taking
the M2-energy fraction along the harmonic direction gives, at ns=(8,16,16):

| file | energy fraction | non-harmonic amplitude |
| --- | --- | --- |
| `w7x_vacuum_co_contra` | 0.999849 | 1.23e-2 |
| `quasr_0044970` | 0.999658 | 1.85e-2 |
| `quasr_0009983` | 0.999321 | 2.61e-2 |
| `quasr0044970_simsopt_B` (un-derotated) | **0.2260** | 8.80e-1 |

For `quasr_0009983` the non-harmonic amplitude (2.61 %) tracks the harmonic-form
field error (2.38 %) almost exactly -- the field genuinely isn't curl-free.

As an **absolute** indicator this is excellent -- three orders of separation
between the broken file and the healthy ones. As a **trend** it does not work,
and the sweep says so: on `w7x_vacuum_co_contra` the fraction *drifts down* with
resolution (0.999849 -> 0.999786 -> 0.999710 -> 0.999635 -> 0.999528 over
n=8..16, i.e. non-harmonic amplitude growing 1.23e-2 -> 2.17e-2) even though
that case's field error is simultaneously falling by 59 %. The measured fraction
carries the projection's own representation error and any boundary mismatch, and
refinement resolves more of both. Use the field-error trend as the
discriminator, not this number's derivative.

## The simsopt files are recoverable

`docs/research/w7x_vacuum_bfield_handoff.md` records that
`quasr0044970_simsopt_B.h5` has its `B` rotated relative to its own `R`, `Z`,
and concludes that "no comparison frame can undo it -- fix upstream". The first
half is right, the second is too pessimistic: no *choice of comparison frame*
helps, but an explicit counter-rotation of the stored vectors does.

`--rotate-scan N` scans rigid z-rotations of the reference `B` and reports the
error against angle. Because the harmonic form never uses `B`, the solve happens
once and the scan is pure numpy afterwards, so it is nearly free. At
ns=(8,16,16):

| file | nfp | best delta | error at delta=0 | error at best delta |
| --- | --- | --- | --- | --- |
| `quasr0044970_simsopt_B` | 3 | 120.00 deg = **+1.0000 field periods** | 87.96 % | **1.09 %** |
| `quasr0009983_simsopt_B` | 2 | 180.00 deg = **+1.0000 field periods** | 21.79 % | **1.08 %** |
| `quasr_0044970` (GVEC) | 3 | 0.00 deg | 1.33 % | 1.33 % |
| `w7x_vacuum_co_contra` (GVEC) | 5 | 0.00 deg | 0.53 % | 0.53 % |

Both simsopt files are off by exactly one field period, at two different `nfp`,
and both land at ~1.08 % once corrected -- in line with the good GVEC files. The
two GVEC files return delta=0, so the scan does not manufacture false positives.
`--rotate-b-periods 1` applies the correction for real runs.

Three consequences:

1. The files are usable, not junk.
2. The handoff note's separate diagnosis of `quasr0009983_simsopt_B` -- that its
   ~22 % was "residual poloidal GVEC-vs-Biot-Savart difference, plausibly real"
   -- is **refuted**. It is the same one-field-period rotation.
3. **The cylindrical-fraction test is nfp-dependent and cannot be trusted
   alone.** At nfp=2 the rotation is 180 deg, which maps the device onto itself
   and leaves `frac toroidal` at a healthy 0.988; at nfp=3 the 120 deg rotation
   wrecks it (0.492). The rotation scan has no such blind spot.

### Convergence after de-rotation

Both files with `--rotate-b-periods 1`, stride 1, `ns = (n, 2n, 2n)`. Solver
health is clean at every point (k=2 Rayleigh ~1e-21, k=1 ~3e-12 / 5e-12, 1-3
sweeps), so all of these are genuine measurements.

`quasr0044970_simsopt_B.h5`, nfp=3:

| n | V2 dofs | L2 projection | rate | harmonic k=2 | rate | harmonic k=1 | rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 400 | 1.8129e-2 | -- | 2.3487e-2 | -- | 1.3929e-2 | -- |
| 6 | 1 752 | 4.4414e-3 | 3.47 | 1.2294e-2 | 1.60 | 1.0864e-2 | 0.61 |
| 8 | 4 640 | 1.9223e-3 | 2.91 | 1.0855e-2 | 0.43 | 1.0548e-2 | 0.10 |
| 10 | 9 640 | 1.1370e-3 | 2.35 | 1.0642e-2 | 0.09 | 1.0492e-2 | 0.02 |
| 12 | 17 328 | 7.8346e-4 | 2.04 | 1.0601e-2 | 0.02 | 1.0510e-2 | -0.01 |
| 16 | 43 072 | 3.8400e-4 | 2.48 | 1.0546e-2 | 0.02 | 1.0516e-2 | -0.00 |
| 20 | 86 480 | 1.7180e-4 | 3.60 | 1.0522e-2 | 0.01 | 1.0516e-2 | 0.00 |
| 24 | 152 160 | 1.0202e-4 | 2.86 | 1.0479e-2 (*) | 0.02 | 1.0518e-2 | -0.00 |

(*) k=2 Rayleigh quotient is 2.1e-5 at n=24 (against 8.3e-14 at n=16 and 6.4e-11
at n=20), so the same k=2 solver failure documented for W7-X reaches this
geometry by n=24. The k=1 column is the trustworthy one there.

`quasr0009983_simsopt_B.h5`, nfp=2:

| n | V2 dofs | L2 projection | rate | harmonic k=2 | rate | harmonic k=1 | rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 400 | 1.9408e-2 | -- | 2.4513e-2 | -- | 1.0622e-2 | -- |
| 6 | 1 752 | 3.1965e-3 | 4.45 | 1.1857e-2 | 1.79 | 1.0631e-2 | -0.00 |
| 8 | 4 640 | 7.7236e-4 | 4.94 | 1.0776e-2 | 0.33 | 1.0732e-2 | -0.03 |
| 10 | 9 640 | 4.5298e-4 | 2.39 | 1.0755e-2 | 0.01 | 1.0719e-2 | 0.01 |
| 12 | 17 328 | 3.7164e-4 | 1.09 | 1.0757e-2 | -0.00 | 1.0706e-2 | 0.01 |
| 16 | 43 072 | 2.4030e-4 | 1.52 | 1.0738e-2 | 0.01 | 1.0721e-2 | -0.00 |
| 20 | 86 480 | 1.0522e-4 | 3.70 | 1.0727e-2 | 0.00 | 1.0721e-2 | 0.00 |
| 24 | 152 160 | 5.4820e-5 | 3.58 | 1.0725e-2 | 0.00 | 1.0723e-2 | -0.00 |

Across a **6x resolution span** the harmonic error is flat to **0.3 %** (s70 k=1)
and **0.1 %** (s83 k=1), pinned at 1.05e-2 and 1.07e-2, while the projection on
the same spaces falls by 95 % and 93 % (overall rates 2.67 and 2.41). Every
harmonic rate from n=8 on is within 0.02 of zero. This is as clean a separation
between a modelling floor and a discretisation error as the data can give.

The two files land on nearly the *same* floor despite different `nfp`, different
coil sets and different equilibria. On `quasr0009983` the k=1 column is flat
from n=4 onward. Harmonic energy fractions are healthy (0.99975 and 0.99987)
and themselves flat to 5 decimal places across the span, but the
non-harmonic amplitude again *grows* with n (1.38e-2 -> 1.58e-2 and
1.06e-2 -> 1.12e-2) -- the same wrong-direction drift documented above for
`w7x_vacuum_co_contra`, and another reason not to read that number's derivative.

**The projection column on these two files is not a usable control.** These are
by far the sparsest files in `data/` -- `N = 1024` points (8x16x8) against 32 768
for `W7X-vacuum` and 125 000 for the 50^3 files. The projection is built from the
sampled grid via `load_grid_field` and then scored *at those same data nodes*, so
what matters is the dof-to-data ratio:

| | data points N | V2 dofs, n=12 | ratio | ratio at n=24 |
| --- | --- | --- | --- | --- |
| `*_simsopt_B.h5` | 1 024 | 17 328 | **16.9x** | 149x |
| `W7X-vacuum.h5` | 32 768 | 17 328 | 0.53x | 4.6x |

On W7-X the projection is scored against more data than it has freedom, so its
convergence is a genuine control and the "two independent constructions agree"
argument holds. On the simsopt files it has 17x more dofs than scoring points at
n=12 and can very nearly interpolate the data, so its falling error establishes
little. Do not read it as "the FEM is not saturated" -- an earlier draft of this
note did, and that was wrong.

The **harmonic** columns are unaffected by this: the harmonic form never touches
`B`, so its error at those 1024 points is a fair measurement. The 1.06 % floor
is real. What it *is* remains unestablished, and refining `n` cannot settle it:

- (a) geometry-representation error, frozen by the coarse fit. The R,Z splines
  use `n_basis = n_data`, so the geometry carries only **8** dof in rho and 8 in
  zeta regardless of `n`. At n=12 the FEM resolves a geometry that is itself
  8-dof in zeta. The harmonic form is *determined* by that geometry, so this
  error propagates straight through.
- (b) a real difference between the Biot-Savart coil field and a harmonic form
  on the GVEC boundary.

The discriminator: `quasr_0044970.h5` is the same device at 50^3. Computing the
harmonic form on the 50^3 geometry and comparing it against the 8x16x8 simsopt
`B` separates (a) from (b). That needs cross-grid evaluation, which the script
does not do today.

### Negative control

Before de-rotation, `quasr0044970_simsopt_B.h5` at ns=(8,16,16) reads:

| check | reading | verdict |
| --- | --- | --- |
| `[RZ]` fit residual | 0.00 % of span | **passes** -- R,Z are rotation-invariant and cannot see this |
| L2 projection of `B` | 2.9e-3 | **passes** -- a projection fits whatever it is handed |
| cylindrical fractions | toroidal 0.492, radial 0.852 | fails (should be ~0.99 / ~0.0) |
| harmonic k=2 field error | 87.96 % | fails |
| harmonic energy fraction | 0.2260 | fails |

Neither the geometry check nor the projection detects the corruption. Only the
harmonic comparison does. That makes the harmonic-content fraction a usable
file-validity test in its own right.

## The k=2 solver failure

Above `n ~ 18` the k=2 harmonic solve stops converging, while k=1 on the same
geometry stays clean all the way to n=24. The Rayleigh quotient catches it
exactly as advertised; `||L v||` does not:

| n | k=2 sweeps | k=2 residual `norm(L2 v)` | k=2 Rayleigh | k=2 field err | k=1 Rayleigh | k=1 field err |
| --- | --- | --- | --- | --- | --- | --- |
| 14 | 4 | 8.45e-3 | 5.83e-14 | 6.68e-4 | 5.27e-12 | 5.18e-4 |
| 16 | 4 | 1.34e-1 | 2.49e-12 | 3.51e-4 | 1.09e-11 | 2.53e-4 |
| 18 | 3 | 8.01e+0 | 2.33e-10 | 4.69e-4 | 4.16e-11 | 2.50e-4 |
| 20 | 3 | 1.71e+3 | 2.76e-06 | 4.46e-4 | 6.25e-11 | 2.22e-4 |
| 22 | 3 | 2.69e+4 | 3.71e-04 | 1.58e-3 | 5.63e-11 | 2.05e-4 |
| 24 | 2 | 7.56e+4 | 1.71e-03 | 3.26e-3 | 1.07e-10 | 1.87e-4 |

Note the sweep count *falls* as the problem gets harder -- 4, 4, 3, 3, 3, 2. The
iteration is quitting earlier precisely where it needs to work harder.

### Mechanism

Two independent contributors, and it is worth separating them because only one
is a bug.

**1. The convergence control is keyed to a metric that drifts with `h`.**
`inverse_iteration` measures `res = ||L_k v||` and uses it for both exits: the
absolute test `res > abs_tol` (default `1e-6`) and the stall guard
`res < stall_ratio * res_prev`. But `mrx/nullspace.py` documents `||L_k v||` as
"a dual vector in the primal mass norm [whose] scale drifts with resolution" --
and the table above shows exactly that drift, seven orders of it, at essentially
constant vector quality up to n=16. The consequence:

- At n=8, `||L2 v|| = 9.0e-8 < abs_tol`, so the loop exits cleanly on tolerance.
- At n >= 14, `||L2 v||` is 8.5e-3 and up. **The absolute test is unreachable at
  any resolution above ~10**, so the stall guard becomes the sole arbiter.
- The stall guard is a *relative* test on that noisy drifting quantity. Once the
  sweep-to-sweep change in `res` is comparable to the inner solve's own noise,
  `res < 0.9 * res_prev` fails immediately and the loop exits after 2-3 sweeps.

So the failure is silent by construction: the guard reports "converged" for a
vector whose Rayleigh quotient is 1.7e-3.

**2. The shifted system gets harder as `h` shrinks, at fixed `eps`.**
`kappa(S_k + eps M_k) ~ lambda_max / eps`, and `lambda_max = O(h^-2)`, so the
inner solve conditioning degrades as `n^2` while `eps` is held at `1e-4`. The
argument in "Solver settings that matter" for a fixed, non-mesh-scaled `eps` is
an *accuracy* argument and remains correct -- but it is silent about cost and
conditioning, which is where this bites. This is compounded by the Schur-Jacobi
preconditioner, whose assembly cost also explodes (see the timings below) and
which is the k>=1 saddle preconditioner that k=2 leans on most heavily.

**3. k=2 nests a vector mass solve that k=1 does not.** From
`L_k = G_k^T M_{k+1} G_k + M_k G_{k-1} M_{k-1}^{-1} G_{k-1}^T M_k`, the exact
apply at k=1 inverts `M_0` -- a *scalar* mass, easy -- while at k=2 it inverts
`M_1`, the *1-form vector* mass, which is the documented W7-X weak point (see
the k=1/k=2 vector-mass coupling work: CG blowup split between the coupling and
lump-fidelity halves, block-SGS tried and regressed). `apply_laplacian`
(`mrx/operators.py:5128`) takes `tol`/`maxiter` precisely because this is a real
nested Krylov solve, run at 1e-7 / 3000 in the driver. **This asymmetry alone
predicts "k=2 fails, k=1 does not" without invoking Schur-Jacobi at all.**

So there are two distinct failure sites, needing different fixes:

- **Site A, measurement.** Every residual/RQ evaluation calls the exact `L`,
  hence a nested `M_1^{-1}` solve per sweep at k=2. If that is noisy, the stall
  guard reacts to mass-solve noise rather than to the iterate.
- **Site B, solve.** The shifted saddle solve and its Schur-Jacobi
  preconditioner.

Site B is certainly real: the *field* error degrades too, and it is computed by
pushforward, independent of any mass solve -- so the vector genuinely is bad,
not merely mis-measured. Site A is likely also present: the sweep count *falls*
(4, 4, 3, 3, 3, 2) as the problem gets harder, which is the signature of a guard
tripping earlier, not of an iteration working harder.

That the sweeps stop *contracting* -- rather than merely being cut short -- says
(2)/(3) are real and not just (1). With `eps=1e-4` and `lambda_1 = O(1)` the
per-sweep contraction should be `eps/(lambda_1+eps) ~ 1e-4`, so two good sweeps
would give a Rayleigh quotient near 1e-8. Getting 1.7e-3 means the sweeps are
not contracting, i.e. the inner solve is returning a poor `w`.

### Measured: which site is it?

Five arms at n=20 on `W7X-vacuum.h5` stride 1, `--no-projection`, k=2 Rayleigh
quotient (k=1 shown for contrast; it was never failing):

| arm | k=2 RQ | vs control | k=1 RQ |
| --- | --- | --- | --- |
| `eps=1e-4` (control) | 2.783e-06 | -- | 6.13e-11 |
| `eps=1e-3` | 5.306e-08 | **52x better** | 5.98e-11 |
| `eps=1e-2` | 2.250e-08 | **124x better** | 5.81e-11 |
| `--stall-ratio 0.99` | 2.704e-06 | 1.03x -- **no effect** | 6.13e-11 |
| `--direct` | 1.535e-09 | 1800x better | 7.04e-05 (**worse**) |

Three conclusions, and the middle one is the one that would have been guessed
wrong:

1. **Raising `eps` fixes it.** Two orders of magnitude on the Rayleigh quotient
   for a shift 100x larger, with k=1 untouched. The conditioning argument
   (`kappa ~ lambda_max / eps`) was the correct diagnosis. With
   `lambda_1 = 21.84` the per-sweep contraction goes 4.6e-6 -> 4.6e-4 across
   the ladder, so even the loosest arm needs only a handful of sweeps.
2. **The stall guard is not the cause.** Loosening it from 0.9 to 0.99 changes
   the result by 3 %. The sweeps genuinely are not contracting, so this is
   **site B (the shifted solve)**, not site A (premature exit on a noisy
   measurement). The Rayleigh-quotient stopping test is still worth doing --
   it makes the failure *visible* -- but it is not the cure.
3. **`--direct` is not a free swap.** It gives the best k=2 by a wide margin,
   but degrades k=1 by six orders (6e-11 -> 7e-5). Whatever it fixes in the
   k=2 path it breaks in the k=1 path, so it cannot simply replace inverse
   iteration for both.

### Fixes, in order of cost

1. **Stop the silent failure: switch both exits to the Rayleigh quotient.**
   `L_k` as implemented (`derham_sequence.py`) is dual-valued, so `w @ Lw` is a
   genuine duality pairing carrying eigenvalue units, while
   `l2_norm(Lw) = sqrt((Lw)^T M (Lw))` measures that dual vector in the *primal*
   M-norm -- the uncompensated mass factor is exactly the `h`-dependence. Only
   `res_new` and `cond_fn` change; `body_fn` already computes `w @ Lw`.

   **Threshold on `sqrt(RQ / lambda_1)`, not on RQ.** The Rayleigh quotient is
   *quadratic* in the eigenvector error: for `v = v_harm + delta * w`,
   `RQ ~ delta^2 * lambda` while the residual is `~ delta * lambda`. Thresholding
   naively on RQ rebuilds the silent-failure mode one level down -- RQ = 1e-10 is
   an eigenvector error of ~1e-5, not 1e-10. `sqrt(RQ / lambda_1)` is a direct,
   mesh-independent estimate of that error, and `--gap-check` already measures
   `lambda_1`. **Measured: `lambda_1 = 21.84`** (W7-X, n=20, `--gap-check`).
   Calibrating against the k=2 error excess over the converged k=1 result:

   | n | RQ | sqrt(RQ/lambda_1) | k2 err - k1 err | ratio |
   | --- | --- | --- | --- | --- |
   | 18 | 2.3e-10 | 3.27e-6 | 2.2e-4 | 67 |
   | 20 | 2.8e-6 | 3.55e-4 | 2.2e-4 | 0.63 |
   | 22 | 3.7e-4 | 4.12e-3 | 1.4e-3 | 0.33 |
   | 24 | 1.7e-3 | 8.85e-3 | 3.1e-3 | 0.35 |

   The estimator **overestimates the eigenvector error by about 3x** wherever
   solver error dominates. That is the right direction for a stopping test --
   an upper bound means a converged verdict is trustworthy -- but it is a bound,
   not an estimate, and the threshold must be set knowing that. At n=18 it
   correctly reports that the solver contributes nothing, and the excess there
   is sampling noise (the projection jumped at n=18 too).

   (An earlier draft assumed `lambda_1 ~ 100`, back-fitted from a single row,
   and claimed agreement within 1.4x. The measured value is 21.84 and the
   honest figure is ~3x conservative.)

   **No k=3 exception is needed** -- and an earlier draft of this note wrongly
   claimed one. `<L_k v, v> = ||d_k v||^2 + ||delta_k v||^2` at every k. At k=3
   the first term drops, leaving `L_3 = M_3 G_2 M_2^{-1} G_2^T M_3` and
   `v^T L_3 v = ||delta_3 v||^2`, which is not identically zero and vanishes
   exactly on the harmonic space (`ker L_3 = ker delta_3 = H_3`, since
   `d_3 = 0`). Degeneracy would need *both* terms to vanish, which happens only
   for a complex of length one. The comment at `mrx/nullspace.py:672-675` is the
   source of the confusion: it says the Rayleigh quotient is zero for k=3, but
   it means the **stiffness-only** form `v^T S v` with `S_3 = 0`, a different
   quadratic form from the full `L_3` the code actually evaluates. That comment
   should be corrected alongside this fix, or the exception will be reintroduced.

   This one is nearly free and should land regardless of what else is done.
2. **Propagate the inner solve's converged flag.** Right now a shifted solve
   that exhausts `CG_MAXITER` (3000 in the driver) returns silently, and the
   stall guard then reads its noise as convergence. A failed inner solve should
   be reported, not absorbed.
3. **Raise `eps` and take more outer sweeps. MEASURED, this is the fix.**
   The classic shift-invert trade, and it is favourable here: `eps=1e-2` cuts
   `kappa` by 100x -- roughly
   10x fewer inner CG iterations -- at the price of a per-sweep contraction of
   `1e-2` instead of `1e-4`, so four sweeps still reach 1e-8. Crucially there is
   **no accuracy penalty**, because the discrete harmonic space lies exactly in
   `ker(L_k)` and a larger shift only slows the linear contraction. The n=20
   ladder above confirms it: 52x at `eps=1e-3`, 124x at `eps=1e-2`, k=1
   unaffected. This is the cheap answer and no new preconditioner is needed.
   **Recommendation: default `eps=1e-2`**, re-running the n>=18 k=2 column
   afterwards.
4. **Attack `assemble_schur_jacobi_preconditioner`.** Now demoted: (3) does it,
   so this is only needed if pushing well past n=24 re-exposes the limit. It is also the cost bottleneck
   (below), so it is the single thing to fix if the goal is to push past n ~ 20.

Until (1) lands, **read the Rayleigh quotient, not the field error**, when
judging any high-resolution harmonic run. A field error computed from a
non-converged nullvector is not a discretisation measurement.

## Where the setup time goes

W7-X 32^3, stride 1, seconds, from `--no-projection` runs with the `[time]`
instrumentation:

| phase | n=8 | n=12 | n=16 | scaling |
| --- | --- | --- | --- | --- |
| DeRhamSequence ctor (quad mesh) | 5.0 | 6.1 | 7.0 | flat |
| evaluate_1d | 5.4 | 5.7 | 5.0 | flat |
| **set_map** (geometry at quad pts) | 67.1 | 242.7 | 606.9 | ~n^3.2 |
| assemble_mass_surgery | 29.5 | 32.8 | 31.5 | flat |
| assemble_tensor_mass | 155.1 | 233.0 | 200.7 | ~flat |
| assemble_incidence | 15.1 | 17.7 | 19.7 | flat |
| **assemble_schur_jacobi** | 28.6 | 174.0 | **873.0** | ~n^5 |
| **total setup** | **306** | **712** | **1744** | |

Preconditioner assembly is 70 % of setup at n=8 and 63 % at n=16, but the
composition shifts: at n=8 the single largest item is the tensor-mass CP
decomposition (155 s), and by n=16 that has been overtaken by
`assemble_schur_jacobi_preconditioner`, which grows with an exponent of roughly
4.5-5.6 -- far steeper than anything else. `set_map` is second and grows as a
clean `n^3.2`, matching the `n^3` quadrature count; it evaluates the spline map
plus a `jacfwd` at every quadrature point.

Extrapolating the `n^5`: at n=32, `set_map` should be ~90 minutes but
Schur-Jacobi assembly would be on the order of **8 hours**. That is consistent
with the observed runs -- n=28 took 8h14m and n=32 hit a 10 h wall.

**The same component is both the cost bottleneck and the accuracy limit**:
Schur-Jacobi is the k>=1 saddle preconditioner, it is the phase whose cost
explodes, and k=2 is exactly the solve that stops converging above n ~ 18 while
k=1 (which leans on it less) stays clean to n=24. That is what to attack to push
past n ~ 20 -- not `set_map`, which is merely cubic and predictable.

## Solver settings that matter

Established by direct measurement on W7-X; see `mrx/nullspace.py` docstrings.

- **`eps = 1e-4`, fixed -- not mesh-scaled.** In FEEC the discrete harmonic
  space lies *exactly* in `ker(L_k)`, so there is no h-dependent near-null floor
  to chase. The only requirement is `eps << lambda_1`, and `lambda_1` is a
  continuum quantity. The old `1e-3/n_r^2` heuristic shrank `eps` with h, which
  bought no outer convergence (already ~1 sweep) and made the shifted solve
  worse conditioned. Measured: `eps = 1e-4` and `eps = 1.5625e-5` give identical
  iteration counts and identical field error. **But** this is an accuracy
  argument only -- see fix (3) above, where raising `eps` is proposed precisely
  because the *conditioning* of the shifted solve does depend on it.
- **`inner_tol = 1e-6`. Do not loosen.** It sets an accuracy *floor*: the
  perturbed iteration's fixed point is displaced by `O(inner_tol)`, so the outer
  residual plateaus there. At `1e-3` the k=2 field error was **30 %**. This
  interacts with the stall guard, which will happily accept the plateau as
  convergence -- the two must be set together.
- **Rank-1 harmonic coarse correction: off.** Sound in principle (it collapses
  the lone `1/eps` outlier the shift creates, and it is *not* circular, since at
  `eps > 0` the shifted solve does no nullspace deflation at all) but measured
  to cost 1-2 extra sweeps and a 5-orders-worse residual for an identical field.
- **Use the Rayleigh quotient, not `||Lv||`.** Across four solver arms producing
  fields identical to five significant figures, `||Lv||` spanned five orders
  (1e-8 to 4.6e-3) while the Rayleigh quotient was uniformly ~1e-13. In the
  broken `inner_tol` run it read 2.3e+02 and correctly flagged the failure.
  `||Lv||` measures a dual vector in the primal mass norm and its scale drifts.
  **This applies to the solver's own stopping test too, where it is not yet
  honoured** -- that is fix (1) above and the direct cause of the silent k=2
  failure.

### Direct vs iterative

Both routes are available. On W7-X at ns=(8,16,16) they agree to five
significant figures (k=2: 0.44507 % both; k=1: 0.38467 % both).

| route | cost | knobs |
| --- | --- | --- |
| inverse iteration (default) | 1-2 shifted saddle solves | eps, inner_tol, stall, maxiter |
| direct Hodge decomposition | 2 unshifted Hodge solves (41 s + 21 s) | none |

The direct route removes every tolerance knob, but it is **not self-sufficient
at `b2 > 0`**: its two stages invert `L_{k+/-1}`, whose kernels have dimension
`b2`, and building those needs the forms under construction -- a genuine
circular dependency, not merely an ordering one. `compute_nullspaces` therefore
rejects `b2 > 0` with an explanation, and inverse iteration (whose shift removes
the singularity) remains the default and the bootstrap.

## Reproducing

Logs go to `outputs/gvec_h5_convergence/<date>/` -- **not** a session temp
directory. An earlier campaign wrote to `$CLAUDE_JOB_DIR/tmp` and its results
were lost when that directory was cleaned.

```bash
J=outputs/gvec_h5_convergence/$(date +%F); mkdir -p $J
sbatch --account=extremedata --partition=gpu-h100l --gres=gpu:1 \
       --mem=128G --time=10:00:00 --job-name=w7x_n12 --output=$PWD/$J/w7x_n12.log --wrap "
  cd /kfs3/scratch/tblickhan/mrx && source .venv/bin/activate
  export XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=256
  python -u scripts/debug/w7x_vacuum_bfield_project.py \
    --h5 data/W7X-vacuum.h5 --stride 1 --ns 12 24 24; echo EXIT=\$?"
```

Use `python -u`: without it stdout is block-buffered off a tty and a multi-hour
job shows an empty log until it exits.

Useful flags: `--rotate-scan N` (scan rigid z-rotations of the reference `B`;
detects frame corruption that R,Z checks and projections both miss),
`--rotate-b-periods F` (apply the correction -- **required, F=1, for both
simsopt files**), `--gap-check` (measure `lambda_1` and the `eps` margin),
`--direct` (Hodge-decomposition route), `--no-projection` (harmonics only, and
much cheaper -- it skips the pushforward diagnostic, the memory hot spot),
`--eps` / `--inner-tol` / `--stall-ratio` / `--coarse` (the knobs above).

### Cluster notes

- `--mem` is **required**; `DefMemPerCPU=1024` means omitting it caps the job at
  1 GB, which cannot even hold the JAX import. The full test suite peaks at
  ~7.9 GB; these runs want 64-128 G.
- JAX caps itself at **59.3 GiB of the 80 GB H100** via
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`; `preallocate=false` defers allocation
  but does not lift that cap. The pushforward diagnostic is the memory hot spot
  (a single 33 GiB tensor at `W7X_MAP_BATCH=2048`) and its size is set by the
  data grid, not by `--ns`. `W7X_MAP_BATCH=256` is safe; the batching is
  `lax.map` chunking, so results are unchanged.
- `gpu-h100s` has `MaxTime=04:00:00`; use `gpu-h100l` for longer runs. At p=3,
  n=24 takes ~4 h wall and n=28 ~8 h, almost all of it setup.

## Open

- **Landed 2026-08-17:** de-rotated simsopt convergence n=4..24 (above), and
  the five k=2 solver probes at n=20 (`eps` ladder, stall guard, direct).
- **Cancelled, no result:** `w7x_vacuum_co_contra` at **stride 1** (n=8,12,16).
  All three sat ~5 h pegged on a single core without emitting a second log line
  and were killed. The blocker is the geometry-fit stage, which builds a
  `DeRhamSequence` at *data* resolution: stride 2 builds it at (25,25,25),
  stride 1 at (50,50,50) -- 8x the elements, host-side and serial. **Anyone
  retrying stride 1 on a 50^3 file must restructure that fit first**; more wall
  clock will not help. The open question it was meant to answer -- whether the
  finite-sample floor drops below the stride-2 ~2e-3 when all 50^3 samples are
  used -- is still open.
- **Not yet done:** fix (1), the Rayleigh-quotient stopping test. Everything
  above n ~ 18 in the k=2 column has to be re-run once it lands.
- **Open question:** whether the 1.06 % simsopt floor is coarse-zeta geometry
  error or a real Biot-Savart-vs-harmonic difference. Needs cross-grid
  evaluation against the 50^3 `quasr_*.h5` files.
