# GVEC → MRX interface

What MRX needs out of a GVEC export, why, and which conventions must be stated
rather than inferred. Names in `code font` are GVEC `state.evaluate()`
quantities ([GVEC docs](https://gvec.readthedocs.io/)).

## 0. The premise

**MRX's primal reference 2-form components ARE GVEC's `sqrt(g) B^i`.** Since
`B_phys = DF omega / J`, the two codes describe the same object in the same
frame. So a field does not need to be *resampled as a vector* — it can be
*rebuilt from scalars*, and the structural guarantees then hold independently
of interpolation quality.

That is the whole reason for this interface. Reconstructing from scalars gives
`div B = 0`, `B.n = 0` and nested flux surfaces EXACTLY, and leaves the fluxes,
`iota` and the helicity exact even when `lambda` is interpolated badly. Fitting
the Cartesian `B` instead puts every interpolation error straight onto `div B`,
`B.n` and the fluxes, and then needs a Leray clean-up to hide it.

## 1. The contract

```
sqrt(g) B^rho   = 0
sqrt(g) B^theta = dchi_dr - dPhi_dr * dLA_dz
sqrt(g) B^zeta  = dPhi_dr * (1 + dLA_dt)
```

Verified on `gvec_nfp3_hegna_80cubed_clebsch.h5` against that file's own `B`,
using its own `grad_rho / grad_theta / grad_zeta`:

| identity | measured |
| --- | --- |
| `sqrt(g) B^rho = 0` | 3.8e-16 |
| `B^theta` relation | ratio 1.00000000, std 2.9e-13 |
| `B^zeta` relation | ratio 1.00000000, std 1.7e-16 |

Any new export must reproduce this check before it is trusted — see §5.

## 2. Required

### 2.1 The field

| GVEC quantity | why |
| --- | --- |
| `LA` | **lambda as the SCALAR.** Not its derivatives — see §3. |
| `dPhi_dr` | toroidal flux derivative |
| `dchi_dr` | poloidal flux derivative |
| `iota` | **send it directly.** `dchi_dr/dPhi_dr` is 0/0 at the axis (both vanish); a stored profile removes the division and is the dimensionless O(1) quantity anyway. |
| `p` | pressure — the end-to-end validation target |

### 2.2 The map and grid

`X1`, `X2` (or `pos`) → R, Z; plus the evaluation points and the attributes
`n_rho`, `n_theta`, `n_zeta`, `nfp`.

### 2.3 Geometry — send GVEC's own, do not make MRX re-derive it

| GVEC quantity | why |
| --- | --- |
| `Jac` | the Jacobian directly, rather than MRX forming `1/[grad_rho . (grad_theta x grad_zeta)]` |
| `g_tt`, `g_tz`, `g_zz` | the angular metric block. MRX's energy model needs the surface averages `<g_tt/J>`, `<g_tz/J>`, `<g_zz/J>`; having GVEC's own metric lets us validate MRX's **spline map** against GVEC's geometry, which separates "our map is wrong" from "our field is wrong". |
| `grad_rho`, `grad_theta`, `grad_zeta` | needed for the §1 verification |
| `B` | likewise — without it the §1 check cannot be run at all |

`g_rr`, `g_rt`, `g_rz` and `e_rho/e_theta/e_zeta` are welcome but not required.

### 2.4 Cross-checks — cheap, and each one localises a different failure

| GVEC quantity | what it pins down |
| --- | --- |
| `F` | **GVEC's own MHD force residual.** This sets the FLOOR for our `\|\|F\|\|/\|\|B\|\|`; without it we cannot tell "our operator is wrong" from "the equilibrium is only converged to 1e-6". |
| `J`, `I_tor`, `I_pol` | lets us check `J x B = grad p` against GVEC's own current instead of only against its pressure |
| `B_theta_avg`, `B_zeta_avg` | flux-surface averages — a direct check on our `<B^chi>/<B^zeta>` recovery of `iota` |
| `mod_B` | cheap check on the reconstructed energy |
| `V`, `dp_dr`, `diota_dr` | volume profile and profile gradients; saves differentiating interpolated data |
| `dLA_dt`, `dLA_dz` | **cross-check ONLY** — see §3. Their real value is pinning the angle convention of §4.1. |

## 3. Storage rule: derivative or parent?

> **Store the derivative — unless two derivatives are tied by an exactness
> identity, in which case store the parent.**

* **`LA` (parent).** `div B = 0` holds because
  `d_zeta(lam_theta) = d_theta(lam_zeta)` — the mixed partials cancel. That
  survives interpolation ONLY if both derivatives come from the same
  interpolant. Two independently interpolated derivative fields degrade
  `div B` from round-off to the interpolation error, discarding the one
  guarantee this interface exists to provide.
* **`dPhi_dr`, `dchi_dr` (derivatives).** Nothing differentiates them —
  `div B` applies only `d_theta` and `d_zeta`, and both are rho-only — so no
  identity needs protecting, and integrating is stable where differentiating
  is not.

`Phi` and `chi` themselves are optional (they are the integrals). Do send the
scalar `Phi(edge)` for the physical flux normalisation.

## 4. Conventions that MUST be stated, not inferred

### 4.1 Angle units

On hegna the angular derivatives are with respect to **radians**
(`theta_G = 2 pi theta`, `zeta_G = 2 pi zeta / nfp`) while `eval_points` is
normalised to [0,1]. This is documented nowhere and had to be recovered by
finite differences:

```
FD(d/dtheta_norm) / dLA_dt = 6.274   vs  2 pi     = 6.283
FD(d/dzeta_norm)  / dLA_dz = 2.0905  vs  2 pi/nfp = 2.0944
```

Please write it into the attributes, e.g. `angle_units: "radians"`,
`zeta_convention: "per_field_period"`, `radial_label: "rho = sqrt(s)"`.

**Why it matters:** converting into MRX's normalised coordinates gives

```
Phi'(rho) = 2 pi * dPhi_dr
iota(rho) = (1/nfp) * dchi_dr / dPhi_dr
lambda    = LA / (2 pi)
```

MRX's `zeta` spans ONE FIELD PERIOD, so the transform per MRX toroidal turn is
`1/nfp` of the transform per full turn. **A missed `1/nfp` makes `iota` nfp
times too large — 5x on W7-X — while passing every structural gate.**

### 4.2 Radial orientation — the existing exports contradict each other

R-spread → 0 marks the magnetic axis (the surface collapses to a curve):

| file | nfp | `axis_radial_index` | R-spread @ idx 0 | @ idx -1 |
| --- | --- | --- | --- | --- |
| hegna_80cubed_clebsch | 3 | 0 | 0.0407 | 3.0194 |
| quasr_0009983 | 2 | 0 | 0.1287 | 0.3168 |
| w7x_vacuum_co_contra | 5 | **49** | 0.7458 | 1.5769 |
| w7x_ini_mrx | 5 | **49** | 0.7190 | 1.5730 |

Hegna is self-consistent. Both W7-X files claim the axis is at index 49 — the
end with the LARGER spread, i.e. the edge — and neither end has spread ≈ 0, so
the axis may not be sampled at all. **Make `rho = 0` the axis and say so.**

### 4.3 The radial label — is `r` the flux or its square root?

VMEC and GVEC both appear in the wild with `s = normalised toroidal flux` and
with `rho = sqrt(s)`. That choice changes every profile, so state it.

**What the hegna export actually does** — measured, not read off a doc:

```
Phi / rho^2   = 0.15915  at every radius       (= 1/2pi)
dPhi_dr / rho = 0.31831  at every radius       (= 1/pi)
Phi ~ rho^k     ->  k = 2.0000
dPhi_dr ~ rho^k ->  k = 1.0000
```

so `Phi(rho) = rho^2 / 2pi` exactly: **`rho` is the SQUARE ROOT of normalised
toroidal flux**, and `dPhi_dr` is linear in `rho`.

Crucially, `dPhi_dr` is the derivative with respect to the *same* `rho` that
indexes `eval_points` — the file carries both `Phi` and `dPhi_dr`, so this is
checkable without trusting anything:

```
cumulative int(dPhi_dr d rho)  vs  stored Phi  ->  max rel 3.5e-16
```

(`chi` gives 3.9e-5, looser only because it carries slight angular variation.)
**Please keep shipping both `Phi` and `dPhi_dr`** — that redundancy is what
makes the convention self-verifying.

**The diagnostic**, if the label ever changes:

| `dPhi_dr` profile | radial label |
| --- | --- |
| proportional to `rho` | `rho = sqrt(s)` — current |
| constant | `s` |

**Why it matters:** switching to `s` while `eval_points` stays `rho` introduces
a factor `ds/drho = 2 rho`. That is RADIUS-DEPENDENT, so it distorts the field's
radial profile rather than its overall scale, and no global normalisation
absorbs it. Note that `iota = dchi_dr/dPhi_dr` is IMMUNE, since a common chain
factor cancels — a wrong radial label corrupts `Phi'` and the field profile
while leaving `iota` looking perfectly correct, which is exactly the failure
that survives a spot-check.

### 4.4 Angular sampling

Prefer half-open `[0,1)`. Hegna is closed `[0,1]` (80 points, step 1/79) while
quasr is half-open (50 points, step 1/50). MRX detects which, but the
disagreement has already caused one bug: a periodic spline with
`n_basis = n_data` is singular on the closed sample unless the duplicated
endpoint is dropped.

## 5. What MRX checks on load

Every one of these is a measurement, not an assumption. A new export should be
put through them before it is used for anything.

1. **§1 identity** against the file's own `B` and grad vectors — pins the
   conventions of §4.1 rather than inheriting hegna's.
2. `max|B^rho| / max|B^zeta|` — nested surfaces survived the projection.
3. `||div B||` by both the combinatorial and mass-projected routes.
4. Leray projection is a no-op.
5. `<B^chi>/<B^zeta>` against the stored `iota` — catches a missed `1/nfp` or a
   sign flip. (`build_gvec_map` MIRRORS raw GVEC data to keep `det DF > 0`, so
   the orientation is measured, never assumed.)
6. Reconstructed `<B^2>/2` against the metric-based energy model.
7. Recovered `p(rho)` against the stored `p` — the end-to-end test of map,
   representation and force operator together, floored by `F` (§2.4).
8. Helicity — must be unchanged when lambda is switched off, since lambda is a
   pure gauge transformation.

## 6. Traps, all of them already paid for

* **`load(frame='ref')` does not take the primal components.** `M_k` carries a
  `g/J` weight, so `M_k^{-1} load` returns `omega` with `B_phys = DF omega / J`,
  while `frame='ref'` wants `g omega / J`. Passing `omega` fails SILENTLY —
  every structural gate passes and only `iota` comes out wrong. Push forward and
  use `frame='phys'`.
* **Cartesian `Bx, By` are zeta-quasiperiodic**, rotating by `R_z(-2pi/nfp)`
  per field period, and need de-rotating before interpolation. `LA` is a scalar
  in logical coordinates and has no such seam — another reason to ship scalars.
* **Do not evaluate a large tensor fit through a dense basis.** A
  `80 x 79 x 79` fit has 499280 functions; a dense identity extraction is
  1.81 TiB, and even matrix-free, evaluating all `n` basis functions per point
  is unusable. Contract the tensor product instead — `O(n1+n2+n3)` per point,
  and it stays differentiable, which §3 requires.

## 7. The synthetic export (what the test suite reads)

`test/synthetic_gvec.py` (`write_synthetic_gvec`) writes a file in this schema from
closed formulas, so the whole route -- `build_gvec_map`, `load_clebsch`,
`clebsch_form`, the projection -- can be checked against known answers with
no data file (`test/test_synthetic_gvec.py`). The layout is that of
`w7x_fmm002_clebsch_mrx.h5`, measured 2026-08-26:

| item | real file | synthetic |
| --- | --- | --- |
| datasets | `eval_points (N,3)`, flat `R`, `Z`, `pressure`, `clebsch/{Phi, chi, dPhi_dr, dchi_dr, LA, dLA_*, grad_*}`, `B`, `beta` | `eval_points`, `R`, `Z`, `pressure`, `clebsch/{Phi, chi, dPhi_dr, dchi_dr, LA}` (what is read, plus the two parents of §3) |
| attributes | `n_rho`, `n_theta`, `n_zeta`, `nfp` (int64) and provenance | the same four, the formula parameters, and the §4.1 convention strings |
| order | C order over `(rho, theta, zeta)`, float64 | same |
| rho | `i/(n-1)`, first point `0.1/(n-1)` (off axis) | same |
| grid | any strictly increasing tensor grid: the axes are read from `eval_points`, and the lambda fit places its knots from the sample (`knots_at_data`), so the radial sample may be refined toward the edge; angles must be half-open on `[0, 1)` starting at 0 | uniform |
| how many points | the angles are GVEC-Fourier-truncated, so Nyquist decides: `n_theta >= 2 m_max + 1`, `n_zeta >= 2 n_max + 1` per field period (this file: m, n <= 16, so 33) -- below that the aliased lambda derivatives put grid-scale current into the IC and the relaxed core goes chaotic; the radial direction has no cutoff, 20 points carry the topology and 33 the profiles to ~1e-4 (`docs/research/coarse_gvec_export_2026-08-26.md`) | 17 x 24 x 8 |
| theta, zeta | half-open `i/n`; zeta spans one field period (the file is stellarator-symmetric on its zeta grid) | same |
| theta orientation | counter-clockwise in (R, Z) from the outboard midplane; `det DF > 0` then selects `Y = -R sin(2 pi zeta/nfp)` | same |
| profiles | `Phi = Phi_edge rho^2` (0.332 Wb), `dchi_dr/dPhi_dr = iota` a flux function, -0.915 to -1.07 | `Phi = Phi_edge rho^2`, `iota = iota0 + iota1 rho^2` per turn, `chi = int iota Phi'` closed |
| lambda | `LA` in radians, derivatives w.r.t. radian angles | `lam_amplitude rho sin(theta_G) (1 + 0.3 cos(nfp zeta_G))` |
| pressure | 75 kPa on axis, beta 1.8% | `p0 (1 - rho^2)` at the given on-axis `beta` |

The map is the circular torus `R = R0 + a rho cos(theta_G)`,
`Z = a rho sin(theta_G)`: concentric surfaces with no Shafranov shift, which
is an equilibrium only at large aspect ratio and low beta. Keep `beta`
small; the test suite uses 1e-3. Nothing in MRX consumes the pressure.

```python
from test.synthetic_gvec import write_synthetic_gvec
torus = write_synthetic_gvec("torus_clebsch_mrx.h5", R0=1.0, a=1/3, nfp=5,
                             n_rho=17, n_theta=24, n_zeta=8, iota=(-0.9, -0.15),
                             Phi_edge=3.1416 / 9, lam_amplitude=0.05, beta=1e-3)
```

`torus` holds the formulas (`R`, `Z`, `dPhi_dr`, `dchi_dr`, `iota`, `LA`,
`pressure`), differentiable with `jax.grad`, so a test can build the §1
contract from them and compare. `scripts/relax.py --geometry <file>` runs on
the synthetic file like on any export.

One loader assumption this file exposed: `load_clebsch` used to decide
whether a periodic axis is closed from `LA` (first and last sample equal),
which mistakes a lambda without angular variation -- `LA = 0` in
particular -- for a closed sample. It now decides from the axis coordinates
(last point 1 or `1 - step`), the rule the map reader always applied.

## From a GVEC state file

MRX also takes GVEC's own ``GVEC_State_*.dat``, and takes it in closed
form. `mrx.gvec` parses the radial B-spline x Fourier representation
of `X1 = R`, `X2 = Z` and `LA = lambda` (series `sum f_mn(s) trig(m theta -
n zeta)` with `n` a multiple of `nfp`, degree-5 clamped B-splines on GVEC's
element grid) and the profiles `Phi`, `iota`, `p` at the radial
interpolation points (`chi' = iota Phi'`). `StateField` evaluates a field
at any logical point in JAX, so `build_gvec_map` collocates `R` and `Z` at
the map's Greville points from the series -- no 50^3 grid and no linear
interpolation bridge in between, the map error is the map space's own --
and `load_clebsch` hands the initial condition `lambda` in closed form. Every
`--geometry` and `build_sequence` argument accepts the `.dat`. Against the
pyGVEC export of W7-X FMM002 the series reproduce `R`, `Z`, `lambda` and
`Phi'` to round-off and `chi'`, `p` to `1e-5` (the interpolation floor of
the 15 profile samples). There is no state-to-grid exporter: the grid
route below is for equilibria that come without a state file.
