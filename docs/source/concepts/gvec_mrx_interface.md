# GVEC → MRX interface

What MRX reads from a GVEC equilibrium, why, and which conventions must be
stated rather than inferred. Names in `code font` are GVEC quantities
([GVEC docs](https://gvec.readthedocs.io/)). The input is GVEC's own state
file `GVEC_State_*.dat`, read in closed form by `mrx/gvec.py`; a VMEC
`wout_*.nc` is refit into the same representation by `mrx/vmec.py` and
everything below applies to it verbatim. There is no other input route.

## 0. The premise

**MRX's primal reference 2-form components ARE GVEC's `sqrt(g) B^i`.** Since
`B_phys = DF omega / J`, the two codes describe the same object in the same
frame. So a field does not need to be *resampled as a vector* — it is
*rebuilt from scalars*, and the structural guarantees then hold independently
of the map fit.

That is the whole reason for this interface. Reconstructing from scalars gives
`div B = 0`, `B.n = 0` and nested flux surfaces EXACTLY, and leaves the fluxes,
`iota` and the helicity exact. Fitting the Cartesian `B` instead puts every
interpolation error straight onto `div B`, `B.n` and the fluxes, and then
needs a Leray clean-up to hide it.

## 1. The contract

```
sqrt(g) B^rho   = 0
sqrt(g) B^theta = dchi_dr - dPhi_dr * dLA_dz
sqrt(g) B^zeta  = dPhi_dr * (1 + dLA_dt)
```

Verified (2026-08) on a pyGVEC export against that file's own `B`, using
its own `grad_rho / grad_theta / grad_zeta`:

| identity | measured |
| --- | --- |
| `sqrt(g) B^rho = 0` | 3.8e-16 |
| `B^theta` relation | ratio 1.00000000, std 2.9e-13 |
| `B^zeta` relation | ratio 1.00000000, std 1.7e-16 |

## 2. The state file

`GVEC_State_*.dat` carries, in this order (`mrx.gvec.read_state`):

| block | content |
| --- | --- |
| `grid: nElems`, `grid: sp` | the radial element grid `sp` on `[0, 1]` |
| `global` | `nfp`, `hmap` |
| `X1_base`, `X2_base`, `LA_base` | per field: `nbase`, degree, continuity, number of modes, `sin_cos` (1 sine, 2 cosine) |
| `X1`, `X2`, `LA` | per mode `m, n` (`n` already multiplied by `nfp`) the `nbase` radial B-spline coefficients: `X1 = R` (cosine), `X2 = Z` (sine), `LA = lambda` (sine, radians) |
| `at X1_base IP point positions` | `s, Phi, chi, iota, pressure` at the radial interpolation points (the Greville points of the `X1` basis) |
| `a_minor, r_major, volume` | scalars |

The series is `sum f_mn(s) trig(m theta_G - n zeta_G)` in GVEC's radian
angles; `s` is the radial label (`Phi = Phi_edge s^2`, so `s` is the
square root of the normalised toroidal flux — what MRX calls `rho`).
`mrx.gvec.StateField` evaluates one block at a logical point in JAX (the
radial basis on GVEC's own clamped knots, the angles as
`2 pi (m theta - n zeta / nfp)`), so

- `build_gvec_map` builds the map's polar spline coefficients of `R` and
  `Z` from the series coefficients mode by mode (`series_spline_dofs`: the
  radial splines L2-projected onto the map's radial basis, the angular
  modes in closed form through the periodic B-spline's Fourier symbol) —
  no evaluation grid at all, the map error is the map space's own
  (`docs/research/analytic_map_2026-08-28.md`);
- `load_clebsch` tabulates the profile splines (`Phi'`, `chi' = iota Phi'`,
  `p`) on 401 uniform radii and hands `lambda` over as the closed-form
  `StateField`, which the initial condition histopolates.

Against the pyGVEC export of W7-X FMM002 the series reproduce `R`, `Z`,
`lambda` and `Phi'` to round-off and `chi'`, `p` to `1e-5` (the
interpolation floor of the 15 profile samples).

## 3. Storage rule: derivative or parent?

> **Store the derivative — unless two derivatives are tied by an exactness
> identity, in which case store the parent.**

* **`LA` (parent).** `div B = 0` holds because
  `d_zeta(lam_theta) = d_theta(lam_zeta)` — the mixed partials cancel. That
  survives any fit ONLY if both derivatives come from the same function.
  The state carries `LA` itself, and the production initial condition
  (`clebsch_potential_form`) never differentiates it at all: the discrete
  `d` does.
* **`Phi'`, `chi'` (derivatives).** Nothing differentiates them —
  `div B` applies only `d_theta` and `d_zeta`, and both are rho-only — so no
  identity needs protecting. They come from the profile splines through the
  state's `Phi`, `iota` samples.

## 4. Conventions that MUST be stated, not inferred

### 4.1 Angle units

GVEC's angles are radians, `theta_G = 2 pi theta` and
`zeta_G = 2 pi zeta / nfp` in MRX's normalised coordinates, and `n` in the
mode table is a multiple of `nfp`. Converting gives

```
Phi'(rho) = 2 pi * dPhi_dr
iota(rho) = (1/nfp) * dchi_dr / dPhi_dr
lambda    = LA / (2 pi)
```

MRX's `zeta` spans ONE FIELD PERIOD, so the transform per MRX toroidal turn is
`1/nfp` of the transform per full turn. **A missed `1/nfp` makes `iota` nfp
times too large — 5x on W7-X — while passing every structural gate.**

### 4.2 The radial label

VMEC and GVEC both appear in the wild with `s = normalised toroidal flux` and
with `rho = sqrt(s)`. That choice changes every profile. GVEC's state uses
`rho = sqrt(s)` (`Phi = Phi_edge s^2` in the profile block: `dPhi_dr` is
linear in `s`); VMEC's wout uses the flux itself, and `mrx.vmec` refits every
mode in `rho = sqrt(s)` so the odd-`m` axis behaviour is analytic. The
diagnostic, if a file ever changes: `dPhi_dr` proportional to `rho` means
`rho = sqrt(s)`; constant means `s`. A wrong label introduces the
radius-dependent factor `ds/drho = 2 rho`, which distorts the field's radial
profile while leaving `iota = dchi_dr / dPhi_dr` untouched — the failure that
survives a spot-check.

### 4.3 Handedness and `nfp`

`mrx.gvec._map_with_sign` uses `Y = -R sin(2 pi zeta/nfp)`, which
mirrors raw GVEC data; `build_gvec_map` measures the sign that gives
`det DF > 0` instead of assuming it. `nfp` enters the map as the angle
`2 pi zeta / nfp`, so a wrong value wraps one field period through the
wrong angle with a healthy Jacobian to hide it; every reader takes an
`nfp` override.

## 5. What MRX checks

Every one of these is a measurement, not an assumption; a new equilibrium
should be put through them before it is used for anything.

1. `||div B||` of the initial condition — round-off for `B = dA'`.
2. The wall-normal part discarded by the Dirichlet restriction
   (`potential_two_form` returns it) — zero for a file whose `A'` is a
   function of `rho` alone on the wall.
3. `<B^chi>/<B^zeta>` against the stored `iota` — catches a missed `1/nfp`
   or a sign flip.
4. The force residual `||F||_M` at `||B||_M = 1` — the end-to-end test of
   map, representation and force operator together, floored by how well
   the equilibrium is converged.
5. Helicity — unchanged when lambda is switched off, since lambda is a
   pure gauge transformation.

`scripts/relax.py --geometry <state file>` runs 1, 2 and 4; the
synthetic state below is where the answers are known in closed form.

## 6. Traps, all of them already paid for

* **`load(frame='ref')` does not take the primal components.** `M_k` carries a
  `g/J` weight, so `M_k^{-1} load` returns `omega` with `B_phys = DF omega / J`,
  while `frame='ref'` wants `g omega / J`. Passing `omega` fails SILENTLY —
  every structural gate passes and only `iota` comes out wrong. Push forward and
  use `frame='phys'`.
* **Cartesian `Bx, By` are zeta-quasiperiodic**, rotating by `R_z(-2pi/nfp)`
  per field period. `LA` is a scalar in logical coordinates and has no such
  seam — the reason the interface is built on scalars.
* **Gridded exports do not converge.** MRX used to read a tensor-grid export
  of the same quantities and bridge it to the Greville points by linear
  interpolation; that bridge has an O(h^2) bias that no mesh refinement
  removes, and every W7-X number obtained through it carried a force floor
  the closed form does not (`docs/research/coarse_gvec_export_2026-08-26.md`).
  The route was removed 2026-08-28.

## 7. The synthetic state (what the test suite reads)

`test/synthetic_gvec.py` (`write_synthetic_state`) writes a state file in
this layout from closed formulas, so the whole route -- `read_state`,
`build_gvec_map`, `load_clebsch`, the potential, the projection -- is
checked against known answers with no data file (the parser by
`test/test_readers.py`; `test/synthetic_gvec.py` writes the state):

| item | synthetic state |
| --- | --- |
| map | the circular torus `R = R0 + a rho cos(theta_G)`, `Z = a rho sin(theta_G)`: modes `(0, 0)` and `(1, 0)` of `X1`, `(1, 0)` of `X2`, radial coefficients of `1` and `rho` (the Greville abscissae) |
| lambda | `lam_amplitude rho sin(theta_G) (1 + 0.3 cos(nfp zeta_G))`: modes `(1, 0)`, `(1, +nfp)`, `(1, -nfp)` of `LA` |
| profiles | `Phi = Phi_edge rho^2`, `iota = iota0 + iota1 rho^2` per turn, `chi = int iota Phi'`, `p = p0 (1 - rho^2)` at the given on-axis `beta`, sampled at the Greville points |
| basis | degree 5 on 10 uniform elements (GVEC's W7-X defaults) |

Every radial function is in the spline space, so a correct parser gives
back the formulas to round-off. Concentric circular surfaces are an
equilibrium only at large aspect ratio and low beta; keep `beta` small (the
suite uses 1e-3) and read the force residual as a property of the choice.

```python
from test.synthetic_gvec import write_synthetic_state
torus = write_synthetic_state("GVEC_State_torus.dat", R0=1.0, a=1/3, nfp=5,
                              iota=(-0.9, -0.15), Phi_edge=3.1416 / 9,
                              lam_amplitude=0.05, beta=1e-3)
```

`torus` holds the formulas (`R`, `Z`, `dPhi_dr`, `dchi_dr`, `iota`, `LA`,
`pressure`), differentiable with `jax.grad`, so a test can build the §1
contract from them and compare. `scripts/relax.py --geometry <file>` runs on
the synthetic state like on any other.
