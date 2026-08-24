# Poincaré sections of the harmonic field — 2026-08-24

`mrx/poincare.py` + `scripts/debug/poincare_vacuum.py` + `scripts/debug/poincare_replot.py`.
Run: `outputs/poincare_v2/2026-08-24/17-02-56/`, ns = (12,24,12), p = 3, 48 lines,
200 field periods, 24 prescribed steps/period, sections at ζ = 0 and ζ = 0.5.

## What the tracer does differently

1. **The toroidal angle is the independent variable.** Dividing the field-line
   ODE through by `B^ζ` gives `dr/dζ = B^r/B^ζ`, `dθ/dζ = B^θ/B^ζ`, so crossings
   land at `ζ = ζ₀ + m` exactly. Nothing is detected, nothing is interpolated,
   and there is no per-crossing root-finding error to accumulate over thousands
   of turns. Exact wherever `B^ζ ≠ 0`, which for a toroidal field is everywhere.

2. **The step schedule is prescribed** (`diffrax.StepTo`), so lanes do not
   couple. An adaptive controller runs a vmapped batch on the smallest step any
   lane asks for; chunking bounds how many healthy seeds one pathological seed
   holds up, it does not isolate it. See the benchmark below. The price is that
   fixed steps carry no error estimate, so `step_convergence` earns the step
   count by h-vs-h/2 refinement and the number is printed on every figure.

3. **The state is `(u,v) = (r cos 2πθ, r sin 2πθ)`.** `B^θ ~ 1/r` near the polar
   axis, so `dθ/dζ` diverges there and the innermost seeds — the ones resolving
   the axis and the low-shear core — are the ones an integrator handles worst.
   In the Cartesian chart the `1/r` cancels against the `O(r)` length of the
   same coordinate vector.

Iota is measured about the magnetic axis, tracked *per phase within the period*
from a dedicated innermost probe, by a least-squares slope on the unwrapped
angle rather than an endpoint difference — on an island the endpoints are two
arbitrary points on a bounded oscillation. The fit residual is reported with it.
This replaces the KS-uniformity screen in `mrx/plotting.py`, which was
discarding lines whose angle had been measured about `r = 0` instead of about
the axis: a bad centre, not a bad line.

## Results (k = 2, essential BC)

| geometry | nfp | iota | h/2 step drift | lines lost |
|---|---|---|---|---|
| toroid | 1 | 0.0000 – 0.0000 | 3.6e-15 | 0/48 |
| rot-ellipse | 3 | 0.0000 – 0.0000 | 4.7e-05 | 0/48 |
| quasr9983 | 2 | 0.0971 – 0.0980 | 1.2e-08 | 0/48 |
| quasr44970 | 3 | 0.4823 – 0.5051 | 2.4e-06 | 0/48 |
| hegna | 3 | 0.4457 – 0.7057 | 8.4e-05 | 0/48 |
| w7x | 5 | 0.8523 – 0.9481 | 1.3e-05 | 0/48 |

Two of these are known answers and both come out right.

* **toroid** — an axisymmetric vacuum field has iota = 0, so every line is a
  fixed point of the return map. Radial drift is 0 and iota is 0 to 1e-17.
* **rot-ellipse** — also exactly 0 (5e-12), and this one is worth spelling out
  because it is surprising until you look at the map. See the next section.
* **w7x** — 0.851 at the axis rising to 0.948 at the edge, which is the
  published standard-configuration vacuum range, from a harmonic form the code
  computes for itself. The profile plateaus near 0.909 ≈ 10/11 and the section
  shows an island chain at exactly the two seeds where the angle-fit residual
  spikes.

## `rotating_ellipse_map` does not rotate

Zero vacuum transform on a rotating ellipse would be wrong. Zero on *this* map
is forced by symmetry, because the map does not rotate anything:

```
R - R0 = eps * nu(zeta)            * r * cos(2 pi theta)
Z      = eps * nu(zeta + 0.5/nfp)  * r * sin(2 pi theta)
```

The two semi-axes lie along `R̂` and `Ẑ` at every `zeta` — there is no tilt term
and no cross term. With `kappa = 1.5` the section runs tall (0.5 × 1.5) at
`zeta = 0`, through an exact circle at `zeta = 0.25`, to wide (1.5 × 0.5) at
`zeta = 0.5`. It *pulsates*; the ellipticity phase is locked at 0 or π/2 and
only the magnitude oscillates.

That leaves the domain invariant under `(X, Y, Z) -> (X, Y, -Z)`, i.e.
`theta -> -theta`, at every `zeta`. The harmonic field is unique up to scale and
the reflection preserves toroidal circulation, so it maps `B` to `+B`; a
reflection-invariant field has zero net poloidal winding, hence `iota = -iota`
and `iota = 0`. Measured 5e-12, with a Z excursion of 1e-10.

l = 2 vacuum transform comes from the ellipse axis *rotating* with `zeta`. In
complex form a real rotating ellipse is

```
(R - R0) + i Z  ∝  r ( e^{2 pi i theta} + delta e^{-2 pi i theta} e^{2 pi i nfp zeta_phys} )
```

where the second term's phase advances. Adding that phase — or equivalently a
tilt angle `alpha(zeta) = pi nfp zeta_phys` applied to the section — turns this
into a geometry with transform. Nothing in the library is wrong: `rot-ellipse`
exists as a metric-variation test case for the preconditioner, where iota is
irrelevant. Only the name misleads.

## The two harmonic routes disagree in the core

Both `null_2_dbc` (Neumann harmonic 2-form, `n·B = 0`) and `null_1` (absolute
harmonic 1-form, `n⌟A = 0`) are d- and δ-closed, boundary-tangent, and live on a
one-dimensional harmonic space, so they are the same physical field reached by
two different solve chains. `field_agreement` reports the max angle between them
over 512 random points, and the iota profiles measure the same thing again:

| geometry | max angle [rad] | iota k=2 | iota k=1 |
|---|---|---|---|
| toroid | 0.0 | 0.0000 – 0.0000 | 0.0000 – 0.0000 |
| rot-ellipse | 7.1e-03 | 0.0000 – 0.0000 | 0.0000 – 0.0000 |
| quasr44970 | 3.2e-02 | 0.4823 – 0.5051 | 0.4766 – 0.5050 |
| hegna | 4.8e-02 | 0.4457 – 0.7057 | 0.4467 – 0.6798 |
| quasr9983 | 7.0e-02 | 0.0971 – 0.0980 | 0.0933 – 0.0980 |
| w7x | 1.8e-01 | 0.8523 – 0.9481 | 0.8333 – 0.9459 |

The edges agree to 0.2 % everywhere. The **cores** do not: W7-X differs by 2.2 %
on the axis, quasr9983 by 3.9 %, hegna by 3.7 % at the edge. So the 0.18 rad max
angle on W7-X is not one bad point near the `r → 1` map singularity — it is a
systematic core disagreement between the two nullspace routes at this
resolution. The k=2 value is the one matching the published W7-X axis figure.
Not chased further here; it wants an h- and p-refinement study.

## Batch coupling, measured

49 seeds × 20 periods, including JIT compile in each arm.

| field | prescribed, vmap | adaptive, vmap | adaptive, chunk 8 |
|---|---|---|---|
| w7x k=2 | 9.9 s | 12.8 s | 22.6 s |
| w7x k=1 | 26.0 s | 43.0 s | 66.9 s |
| hegna k=1 | 21.3 s | 36.0 s | 57.3 s |
| quasr9983 k=1 | 22.4 s | 35.4 s | 59.9 s |
| **quasr44970 k=1** | **22.4 s** | **215.0 s** | — |

The last row is the failure mode: one seed's adaptive step collapsed and dragged
the whole vmap to 9.6× the prescribed cost. Note also that **chunking is worse
than not chunking** — eight-seed chunks run sequentially and each pays its own
worst-seed step, where a single vmap pays the global worst once. The old
`min(8, nseeds)` default was the worst of the three.

## Open

* **The k=1 traces are not step-converged.** h/2 drift is 2.3e-02 on quasr44970
  and 4.4e-03 on hegna, against 1e-05 – 1e-08 for k=2. 24 steps/period is not
  enough for the 1-form field; the iota profiles still land within 1.2 % of the
  k=2 ones, but the k=1 numbers above should not be quoted without a rerun at
  more steps. The k=2 traces are converged.
* The core disagreement between the two harmonic routes (above).
* `mrx/plotting.py`'s `integrate_fieldlines` / `get_periodic_intersections` /
  `get_iota_log` are now superseded for this purpose but still used by
  `scripts/config_scripts/poincare_plots.py`, which traces relaxation states
  rather than a harmonic field. Porting it is untouched work.

## How to run it

```bash
# one job per geometry, ~5 min each on an H100 (the nullspace solves dominate)
GEOMS="rot-ellipse w7x quasr9983 quasr44970 hegna toroid" OUTSUB=poincare \
  ARGS="--ns 12,24,12 --p 3 --seeds 48 --periods 200 --steps 24 --saves 8 --planes 0,0.5" \
  bash slurm/job_poincare_vacuum.sh

# change presentation without re-solving (login node, <1 s, no GPU)
python scripts/debug/poincare_replot.py outputs/poincare/*/trace_*.npz

# quantify the batch coupling
... --bench --bench-periods 20
```

`--steps` must be a multiple of `--saves` so every saved value is a step
endpoint rather than a dense-interpolation value. `--saves` must exceed twice
the poloidal turns per period or the angle unwrap aliases; 8 covers everything
here.

## Next steps, in the order I would do them

1. **Rerun the k=1 fields at `--steps 96`** and check whether the core iota
   moves. That is one cheap job and it decides whether the k=2/k=1 core
   disagreement is a discretisation statement about the harmonic forms or an
   artefact of an under-resolved k=1 trace. Until it is done, the k=1 iota
   column above is not quotable.
2. **h- and p-refine W7-X** (ns 12→16→20 at p=3, then p=4) and watch the axis
   iota against the published 0.85. If both routes converge to it, the core
   disagreement is just resolution; if one stalls, that route has a problem.
3. Optionally add a tilt to `rotating_ellipse_map` (or a new
   `helical_ellipse_map`) to get a *cheap analytic* geometry with nonzero
   vacuum transform — useful as a tracer test case with a known answer, which
   right now only the toroid provides.
