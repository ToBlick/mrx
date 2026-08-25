# Relaxation initial conditions from logical profiles — 2026-08-25

How to build a relaxation IC without interpolating `B` from GVEC. The short
answer is that GVEC's representation and ours are the *same object*, so the
field can be rebuilt from three scalars instead of resampled as a vector — and
when no data exists at all, the same ansatz gives analytic ICs and, on a
cylinder, exact analytic equilibria.

Status: derivations complete and checked against data. Landed: the cylinder
structure gates (§7.1), all four screw-pinch arms against the closed-form
pressure (§7.3), both toroid arms including one that OVERTURNED a claim in this
document (§8.1), and the lambda equation on both a toroid and a real
stellarator (§9.1). Still queued: §7.2.

Scripts: `scripts/debug/logical_profile_ic.py`,
`scripts/debug/gvec_clebsch_ic.py`, `scripts/debug/analytic_ic_verify.py`,
`scripts/debug/lambda_warmstart.py`.
Jobs: `slurm/job_logical_ic.sh`, `slurm/job_analytic_ic.sh`.
Commits: `cc685fc`, `029c653`, `cb395de`, `84618a6`, `fc8a27e`.

---

## 1. The ansatz, and what it gives for free

A k=2 DBC field in the REFERENCE 2-form frame (component order
`dchi^dzeta, dr^dzeta, dr^dchi` = the rho-, chi- and zeta-directed logical
fluxes):

```
B_hat^rho  = 0
B_hat^chi  = Phi'(rho) ( iota(rho) - dlambda/dzeta )
B_hat^zeta = Phi'(rho) (     1     + dlambda/dchi  )
```

On **any** geometry, before any solve:

* `B^rho = 0` ⟹ B is tangent to every `rho = const` surface, so the IC has
  perfectly nested flux surfaces. Pushforward maps tangent fields to tangent
  fields, so this holds in physical space too.
* `div B = 0` for **any** lambda: `d_chi(B^chi) + d_zeta(B^zeta) =
  Phi'(-lam_zetachi + lam_chizeta) = 0`, the mixed partials cancelling.
* `B.n = 0`, which is exactly what the k=2 Dirichlet space enforces.
* Field lines obey `dchi/dzeta = B^chi/B^zeta`, so `iota(rho)` IS the
  rotational transform, with no metric in the way.

**This is the GVEC/VMEC representation.** The reference 2-form components are
exactly `sqrt(g) B^i`, since `B_phys = DF B_hat / J`. Confirmed against the
literature: `educational_VMEC` states `Bsupu = phip*(iota - lambda_v)/SQRT(g)`,
identical to `B^chi` above, with `theta* = theta + lambda` the PEST angle and
`theta_B = theta + lambda + iota nu` the Boozer one.

## 2. What lambda does and does not change

| quantity | changed by lambda? | why |
| --- | --- | --- |
| `div B`, `B.n`, nested surfaces | **no** | mixed partials cancel; `B^rho` is set, not fitted |
| fluxes `Phi`, `X`, and `iota` | **no** | lambda enters only via exact angular derivatives, which average to zero over a surface |
| helicity | **no** | lambda is a pure gauge shift `A -> A + d mu` with `d_rho mu = Phi' lambda`, mu single-valued |
| energy, force, Pfirsch-Schlueter spread | **yes** | these are not surface averages |

So within the whole GVEC family, **H depends only on `(Phi, iota)` and is blind
to lambda**. That splits the labour cleanly: `Phi, iota` are the ideal
invariants the IC fixes and relaxation conserves; **lambda is what the
relaxation has to generate.**

## 3. Helicity — exact, analytic, metric-free

With `A = Phi(rho) dchi - X(rho) dzeta` (toroidal and poloidal flux),
`A ^ dA = (Phi X' - X Phi') drho^dchi^dzeta`, so with chi, zeta on [0,1):

```
H = int_0^1 (Phi X' - X Phi') drho = int_0^1 Phi(rho)^2 W'(rho) drho,   W = X/Phi
```

No metric appears — helicity is topological, so this is as exact on W7-X as on
a cylinder. Consequences:

* **H = 0 identically when iota is CONSTANT.** Helicity comes only from SHEAR
  in W, never from the transform itself.
* The weight is `Phi^2`, so edge shear counts far more than core shear.
* For `Phi' = rho^q`, `iota = iota0 + Diota rho^e` it closes:

```
H = Diota * e / [ (q+1)(q+e+1)(2q+e+2) ]
```

  linear in the shear. **Verified against direct quadrature to ~1e-13 over six
  profile shapes**, including the zero-shear case (returns 1.4e-16).

CAVEAT: on a torus (b1 = 1) helicity is gauge-ambiguous by the harmonic 1-form.
The above is the value in that natural gauge; `mrx.relaxation.compute_helicity`
uses a different one (co-exact A via the Hodge Laplacian, plus the harmonic
remainder), so the two differ by the harmonic contribution.

## 4. Relation to Clebsch

When iota is constant, `X = iota Phi` and `B = dPhi ^ d(chi - iota zeta)` —
exactly Clebsch, `B = grad alpha x grad beta`, with `beta` the straight-field-
line angle. In the Clebsch gauge `A = alpha grad beta`, so `A.B` vanishes
POINTWISE — which is *why* constant iota gives `H = 0`. The two statements are
the same fact.

So the two-function Clebsch is the doubly-degenerate corner: `lambda = 0` AND
zero shear. Restoring shear generalises it one way (H != 0); restoring lambda
generalises it the other (H unchanged) and recovers full GVEC.

## 5. Reading it from GVEC — better than interpolating B

`data/gvec_nfp3_hegna_80cubed_clebsch.h5` carries every ingredient by name:
`clebsch/dPhi_dr`, `clebsch/dchi_dr`, `clebsch/LA`, plus `pressure`.
**Verified against that file's own B**, using its own `grad_rho/theta/zeta`:

```
sqrt(g) B^rho   = 0                            measured 3.8e-16
sqrt(g) B^theta = dchi_dr - dPhi_dr * dLA_dz   ratio 1.00000000, std 2.9e-13
sqrt(g) B^zeta  = dPhi_dr * (1 + dLA_dt)       ratio 1.00000000, std 1.7e-16
```

### 5.1 UNITS — and they are not the identity

`eval_points` is normalised to [0,1] on all three axes, but the DERIVATIVES are
with respect to RADIAN angles `theta_G = 2 pi theta`, `zeta_G = 2 pi zeta/nfp`.
Measured by finite differences of `LA` against the stored derivatives:

```
FD(d/dtheta_norm) / dLA_dt = 6.274   vs  2 pi     = 6.283
FD(d/dzeta_norm)  / dLA_dz = 2.0905  vs  2 pi/nfp = 2.0944
```

(the gap is the O(h^2) FD error at 80^3). Converting into MRX's normalised
coordinates gives three rules:

```
Phi'(rho) = 2 pi * dPhi_dr
iota(rho) = (1/nfp) * dchi_dr / dPhi_dr
lambda    = LA / (2 pi)
```

The `1/nfp` is physically right — MRX's zeta spans ONE FIELD PERIOD — and
**without it the reconstructed iota is nfp times too large.**

### 5.2 Why this beats `interpolate_B`

`interpolate_B` fits the Cartesian vector field and then repairs it: every
interpolation error lands on `div B`, on `B.n`, and on the fluxes, and
`P_Leray` cleans up afterwards. Through the Clebsch scalars instead, **the
guarantees do not depend on the fit**: `div B = 0` and `B.n = 0` exactly, and
the fluxes, iota and helicity are exact *even if lambda is interpolated badly*
(§2). lambda is also a SCALAR in logical coordinates, so it is immune to the
zeta quasi-periodicity seam that forces Cartesian `Bx, By` to be de-rotated
(`w7x_vacuum_bfield_handoff.md`).

### 5.3 Store the scalar, not the derivatives

`div B = 0` holds because `d_zeta(lam_theta) = d_theta(lam_zeta)`, and that
identity survives interpolation **only if both derivatives come from the same
interpolant**. Reading `dLA_dt` and `dLA_dz` as two independently interpolated
fields degrades `div B` from round-off to the interpolation error. So read
`clebsch/LA` and fit it with an interpolatory collocation spline
(`n_basis = n_data`, the fit `mrx.io.load_grid_field` step 1 does).

The **opposite** call for `dPhi_dr`, `dchi_dr`, for the opposite reason:
nothing differentiates them, so no identity needs protecting, and integrating
is stable where differentiating is not.

> **Rule.** Store the derivative — unless two derivatives are tied by an
> exactness identity, in which case store the parent.

Checked lossless: in consistent radian units the stored derivatives satisfy the
mixed-partial identity to 6.6e-3 relative, i.e. they really are derivatives of
one lambda.

### 5.4 How big is lambda? Large

| measure (hegna, nfp=3) | value |
| --- | --- |
| `\|LA\|` max | 0.549 rad = **31.4 deg** poloidal shift |
| `lam_chi` max (modulates `B^zeta` as `1+lam_chi`) | **0.826** -> 83% variation over a surface |
| `lam_zeta` max (enters `B^chi` as `iota - lam_zeta`) | **0.353** |

`lam_chi` grows 0.018 at rho=0.05 to 0.62 at rho=0.95. The consequential number
is `lam_zeta` against iota:

```
rho     max|lam_zeta|   |iota|   ratio
0.101       0.3327      0.1510    2.20
0.253       0.3524      0.1562    2.26
0.506       0.3337      0.1748    1.91
0.759       0.3352      0.2058    1.63
0.949       0.1690      0.2371    0.71
```

**lam_zeta is about twice iota through most of the volume.** Setting lambda = 0
does not perturb `B^chi`; it changes it by more than 100% pointwise and flips
its sign over part of each surface. The surface AVERAGES stay exact — that is
the §2 guarantee — but lambda = 0 is a far worse pointwise approximation than
one would guess.

## 6. Where the pressure comes in

The causality runs backwards from how it is usually asked. In VMEC/GVEC,
`p(s)` and `iota(s)` (or the current profile) plus the boundary are INPUTS;
lambda and the surface shapes R, Z are OUTPUTS. **There is no formula
`p = f(Phi', iota, lambda)`.**

But once the MAP is frozen — ours is, it is GVEC's own R, Z — B is determined
completely by the three, so p follows from force balance:

```
p'(rho) = < (J x B).grad rho > / < |grad rho|^2 >
```

exact in any geometry, but a diagnostic (needs `J x B`) rather than a
predictor. Physically lambda is where the pressure lives: `div J = 0` with
`J_perp = (B x grad p)/B^2` forces a parallel current obeying
`B.grad(J_par/B) = -div J_perp`, linear in `p'`. That is the Pfirsch-Schlueter
current. At low beta it splits lambda into a geometry-and-iota part plus a
piece linear in `p'`.

### 6.1 The predictive part is the energy, and it is geometry-general

The cylindrical screw-pinch formula is NOT the general rule. The geometry
enters only through three surface averages plus the volume element:

```
a = <g_chichi/J>   b = <g_chizeta/J>   c = <g_zetazeta/J>   V' = <J>

u(rho) = 1/2 Phi'^2 [ a iota^2 + 2 b iota + c ] / V'
```

all pure geometry — no solve, no assembly. It reduces to
`(B_theta^2 + B_z^2)/2` on the cylinder map identically. Then:

* `u` is a QUADRATIC in iota with `a > 0`, so it inverts: given a target
  `du/drho` you can solve for the iota that delivers it, per surface, from the
  metric alone. That is the design knob.
* `b` is a purely 3-D term — zero on the cylinder, nonzero wherever the
  theta-zeta metric coupling is (`metric-weight-separability-rule`). It makes
  the energy-minimising transform `iota*(rho) = -b/a`, not zero: **the geometry
  has a preferred iota.**
* In 3-D `(J x B).grad rho / |grad rho|^2` is generically NOT a flux function,
  so no `(iota, Phi')` is an equilibrium. Its spread over a surface is the
  Pfirsch-Schlueter drive — a quantity to measure, not a defect.

### 6.2 The cylinder DOES close, exactly

On `cylinder_map(a, h)` with lambda = 0 and rho-only profiles the configuration
is a straight screw pinch, and the exact balance is

```
dp/drho = -d/drho[(B_theta^2 + B_z^2)/2] - B_theta^2/rho          (1)
B_theta = omega^chi/(a h),   B_z = omega^zeta/(2 pi a^2 rho)
```

with `r = a rho` and the factor `a` cancelling out entirely. For `Phi' = rho^q`
and POLYNOMIAL iota every term is a polynomial, so p follows by exact
antidifferentiation — **no quadrature, no fit**. Checked: (1) reproduces the
z-pinch of `test/test_relaxation.py` to **2.2e-16**, including `p(0) = 5/3`.

So polynomial profiles on a cylinder give a whole FAMILY of manufactured
equilibria (exact B, exact p, exact iota, closed-form H), generalising the
single z-pinch, which is the `B_z = 0` corner.

**Anything else is not an equilibrium.** In particular on a torus, see §8.

## 7. The GPU arms

### 7.1 LANDED: `lic_cyl` — the cylinder passes every structural gate

`outputs/logical_ic/2026-08-25/03-02-51/cyl.log`, iota 0.4 -> 0.9 (exp 2),
`Phi' = rho`, ns=(12,24,12) p=3.

| gate | result |
| --- | --- |
| `max\|B^rho\|/max\|B^zeta\|` | 4.8e-18 axis band, **1.1e-16 bulk** |
| `\|\|div B\|\|` (incidence AND strong_div) | 3.6e-14 |
| `\|\|P_Leray B - B\|\|` | 4.7e-16 |
| iota from `<B^chi>/<B^zeta>` | ratio **1.0000**, POSITIVE |
| eq. (4) vs measured `<B^2>/2` | 3.7e-05 |
| `b = <g_chizeta/J>`, `iota* = -b/a` | **exactly 0** |
| `\|\|F\|\|/\|\|B\|\|` | **3.0e-12** |
| Pfirsch-Schlueter spread `std/\|mean\|` | **2.4e-14** median |

Three things settled:

* **The L2 route costs nothing here.** `B^rho` at 1e-16 means the metric
  coupling did NOT reintroduce a radial component on this geometry, so
  unlocking histopolation (§10) is not urgent — though the cylinder is the
  easy case and W7-X may differ.
* **The sign convention is settled**: `<B^chi>/<B^zeta>` comes back as `+iota`,
  no flip. The `dr^dzeta = -dzeta^dr` worry was unfounded.
* **The ansatz really is an equilibrium on a cylinder** — force at round-off
  and the radial force a flux function to machine precision, exactly as §6.2
  requires.

`lic_cyl_noshear` (constant `iota = 0.7`) adds the helicity check: eq. (2)
returns **`+0.000000e+00`** at zero shear, exactly as the closed form requires,
and `iota` is recovered as `0.700000` at every radius (ratio `+1.0000`, a
constant profile being exactly representable). Force 1.68e-12, PS spread
2.13e-14. `compute_helicity` returns `+7.81` for the same field -- so for a
constant-iota field the ENTIRE value MRX reports is the harmonic gauge term and
none of it is shear helicity, which is the §3 ambiguity made concrete.

The one non-zero number, `[press] slope 1.5094, unexplained 3.05e-02`, is
EXPECTED and is not an error: eq. (4)'s `-du/drho` omits the tension term
`-B_theta^2/rho` of the exact balance (1). Predicting the slope from (1)
directly gives **1.5153 against the measured 1.5094** (0.4% apart) and residual
2.64e-2 against 3.05e-2. The tension term carries 33-49% of `dp/drho` across
the radius. So the measured pressure agrees with the exact screw-pinch balance
to sub-percent; the direct test is `aic_sp_*`, which uses the FULL balance and
should return slope 1.0 with no fit.

### 7.2 Arm ledger

Output dirs:
`outputs/analytic_ic/2026-08-25/03-30-32/`,
`outputs/logical_ic/2026-08-25/03-02-51/`,
`outputs/lambda_ws/2026-08-25/03-49-49/`.

| arm | what it decides |
| --- | --- |
| ~~`aic_sp_{sheared,flat,zero,q2}`~~ | LANDED — see §7.3 |
| ~~`aic_tor_vacuum`~~ | LANDED, and the arm was MIS-SPECIFIED — see §8.1. Replaced by `--flux vacuum` (job 16764594) |
| ~~`aic_tor_sheared`~~ | LANDED — 8.2x force reduction from eq. (2); see §8.1 |
| ~~`lic_cyl`~~ | LANDED -- see §7.1 |
| `lic_gvec` | GVEC reconstruction; the invariance test (H and the iota column must not move with `--no-lambda`; force and pressure must) |
| `lws_toroid` | general lambda solve vs the closed form |
| `lws_hegna` | general lambda solve vs GVEC's own lambda |

### 7.3 LANDED: the screw pinch, against the closed-form pressure

All four cylinder arms, `ns=(12,24,4)` p=3. `p` is compared with NO fitted
parameter -- B is normalised and p is quadratic in B, so the measured p is
scaled back by `B_norm^2`; only the solver-defined additive constant is removed.

| arm | iota(rho), Phi' | `\|\|F\|\|/\|\|B\|\|` | `p` vs exact | `B^rho` rel | `\|\|div B\|\|` |
| --- | --- | --- | --- | --- | --- |
| `sp_sheared` | `0.4+0.5 rho^2`, `rho` | 1.29e-12 | **1.80e-03** | 1.17e-16 | 3.5e-14 |
| `sp_flat` | `0.7`, `rho` | 1.07e-12 | **9.01e-04** | 1.47e-16 | 4.1e-14 |
| `sp_q2` | `0.3+0.6 rho^2`, `rho^2` | 3.80e-12 | 4.84e-03 | 1.10e-16 | 3.1e-14 |
| `sp_zero` | `0`, `rho` | **7.61e-15** | n/a | **0.0** | 1.2e-14 |

Three independent profile shapes agree with the closed-form pressure to **under
0.5%, nothing fitted**, with the force at ~1e-12. This is the first test in this
document that checks `compute_force` against an EXTERNAL truth rather than an
internal consistency relation. `sp_q2` is loosest in the expected direction:
`Phi' = rho^2` makes B_z non-uniform and pushes p to a higher-degree polynomial.

**`sp_zero` is the control that makes the rest meaningful.** With `iota = 0` and
`q = 1` the field is a uniform axial field, current-free, so the force must be
pure round-off -- it comes back at **7.61e-15**, three orders below the others,
and `max|B^rho|` is IDENTICALLY zero. Without it, the ~1e-12 elsewhere could not
be distinguished from a floor in the diagnostic.

Two caveats about the script's own metrics, neither affecting the above:

* The reported `dp/drho` errors (1.14e-02 to 3.19e-02) are dominated by the
  diagnostic, not the solver: applying `np.gradient` to the EXACT p on the same
  41-point grid already gives 1.60e-02. The `p` column is the informative one.
  Fix would be to apply the same finite difference to both sides.
* `sp_zero` has `p = 0` analytically, so its relative-error denominator is zero
  and both pressure metrics print `inf`. Harmless, but uninformative -- the
  force and `B^rho` are what decide that arm.

## 8. The toroid: lambda = 0 is not even the vacuum field

For `toroid_map` at kappa=1, `sqrt(g) = 4 pi^2 eps^2 rho R` with
`R = R0 + eps rho cos(2 pi chi)`, so

```
B_phi = Phi'(rho) (1 + lam_chi) / (2 pi eps^2 rho)
```

and **R cancels**: with lambda = 0 the toroidal field has no 1/R dependence at
all. Requiring `R B_phi` to be a flux function forces

```
1 + lam_chi = <1/R>^-1 / R,    <1/R>_chi = 1/sqrt(R0^2 - eps^2 rho^2)      (2)
```

(the average verified numerically to 2e-16). The toroid is axisymmetric so
`lam_zeta = 0` identically and `div B` stays exactly zero. Expanding for small
`e = eps rho/R0` recovers the familiar large-aspect-ratio result:

```
lambda ~ -e sin(theta) + (e^2/4) sin(2 theta) + O(e^3)
```

Even with (2) it is **still not a full equilibrium**: Grad-Shafranov also
constrains the surface SHAPES (the Shafranov shift), which is a property of the
map and not something lambda can supply.

### 8.1 MEASURED — and (2) alone does NOT give the vacuum field

Both toroid arms landed on 2026-08-25 and one of them overturned what this
section originally claimed.

| arm | lambda = 0 | closed-form lambda | |
| --- | --- | --- | --- |
| `iota = 0` | 3.93e-05 | 1.73e-02 | prediction INVERTED — see below |
| `iota = 0.4 + 0.5 rho^2` | 9.02e-02 | **1.10e-02** | **8.2x better**, as designed |

**The iota != 0 arm vindicates (2)**: an 8.2x force reduction, with the 1.10e-02
residual being exactly the Shafranov term this section already predicted would
survive. Eq. (2) itself was never in doubt -- `lws_toroid` reproduces it from
the general lambda equation to 1.30e-08 (§9).

**The iota = 0 arm was mis-specified**, and both halves of the original claim
were wrong for separate reasons:

* A purely toroidal VACUUM field needs `R B_phi` to be a **global constant**,
  not a flux function. Eq. (2) only forces `1 + lam_chi = c(rho)/R`, leaving
  `R B_phi = c(rho) Phi'(rho) / (2 pi eps^2 rho)` free to vary with rho. With
  `Phi' = rho` it goes as `sqrt(R0^2 - eps^2 rho^2)` -- a measured 1.054x
  spread -- so the field carries poloidal current and a nonzero force is
  CORRECT.
* The lambda = 0 force was small for an unrelated reason. With `Phi' = rho` and
  `iota = 0` the R cancels out of `B_phi` entirely, leaving `B_phi = const`, so
  `J x B = grad(-B_phi^2 ln R)` is a **pure gradient** and `P_Leray` removes all
  of it. A small force there means "the residual was a gradient", NOT "the field
  was an equilibrium". Distinguishing those two is the whole point of the arm.

The fix is on the FLUX side, not lambda:

```
Phi'(rho) = rho <1/R> = rho / sqrt(R0^2 - eps^2 rho^2)        (--flux vacuum)
```

which gives `B_phi = 1/(2 pi eps^2 R)` and `R B_phi` constant to 1e-12. THAT
pair -- eq. (2) together with this Phi' -- is the vacuum field, and it is the
arm whose force must collapse. Job 16764594.

**How the error got in, which is the transferable part.** The original
derivation was CORRECT: verifying (2) against the known vacuum field
`B_phi = B0 R0 / R` produced `Phi' = 2 pi eps^2 B0 R0 * rho * <1/R>` -- WITH the
`<1/R>`. The script then exposed only `--flux-exp`, i.e. `Phi' = rho^q`, which
cannot express `rho <1/R>`. So a correct derivation was silently discarded at
the parameterisation step, and the arm went on to "test" a pair that was never
the vacuum field.

> **Lesson.** An inexpressive knob can quietly drop a correct derivation, and
> nothing downstream complains -- the run completes, the gates pass, and only
> the physics is wrong. The guard is to check the derived object against the
> code's actually reachable set before trusting an arm labelled "decisive".

### 8.2 WHERE THE L2 ROUTE FIRST LEAKS -- and it is not what I first said

`aic_tor_sheared` is the first case in this document where the L2 projection
fails to hold the structure exactly:

| case | `max\|B^rho\|/max\|B^zeta\|` | `\|\|div B\|\|` |
| --- | --- | --- |
| cylinder, every arm | ~1e-16 | ~3e-14 |
| toroid, `iota = 0` | 4.2e-16 | 3.5e-15 |
| **toroid, `iota != 0`** | **4.6e-09** | **9.4e-08** |

Structure holds at machine precision until `B^chi = Phi' iota(rho)` is nonzero
AND the geometry varies over a surface -- then it loses seven orders.

**The mechanism is NOT off-diagonal metric coupling.** That was my first
explanation and it is wrong: `toroid_map` at kappa = 1 has an EXACTLY DIAGONAL
metric, `g_rt = g_rz = g_tz == 0` identically, same as the cylinder. What
distinguishes them is the CHI-DEPENDENCE of the diagonal mass weights `g_ii/J`
over a surface:

```
spread of g_ii/J over chi     rho=0.25   rho=0.50   rho=0.95
  toroid  (all three i)         16.7%      33.8%      66.8%
  cylinder                       0.0%       0.0%       0.0%
```

tracking `eps rho / R0` exactly. So the leak condition is **chi-dependent
metric weights**, not "toroidal geometry" -- the toroid is just the cheapest
example of one.

HYPOTHESIS, not verified: with `g` diagonal the full-space mass matrix is
block-diagonal per component, so a zero comp0 load should stay zero; the leak
would then have to enter through the POLAR EXTRACTION, since
`M2_reduced = e2 M2_full e2^T` is not block-diagonal where `e2` mixes components
near the axis, and a chi-independent weight makes those cross terms cancel by
angular symmetry while a chi-dependent one does not. The polar-mixing step has
NOT been checked.

Two falsifiable predictions follow, both already measurable:

1. the leak should be LOCALISED near the axis rather than spread through the
   bulk. `logical_profile_ic.py` reports an axis band separately and the axis
   looks ~30x CLEANER on both cylinder arms (4.7e-18 vs 1.4e-16) -- but that
   number is CONFOUNDED and should not be read as 30x. Both bands are divided
   by a single GLOBAL `max|B^zeta|` (`logical_profile_ic.py:409`), while
   `B^zeta ~ Phi'(rho) = rho` is itself about 7x smaller inside `rho < 0.15`.
   So roughly a factor 7 is pure normalisation artefact and only the remaining
   ~4x is potentially real. Weak evidence against polar mixing, not strong.

   The honest form of this test is a PER-SURFACE ratio
   `max|B^rho(rho)| / max|B^zeta(rho)|` reported as a profile, which is also
   the only form that discriminates the mechanisms at all: weight-spread
   predicts the leak tracks `eps rho / R0` (quadrupling from rho=0.25 to 0.95),
   polar-mixing predicts concentration in the first rings, and neither predicts
   flat. Two band maxima cannot tell those apart. Added to
   `gvec_clebsch_ic.py`; `logical_profile_ic.py` still has the global
   normaliser and its two landed runs are not re-run for this.
2. the leak should scale with the surface spread of `g_ii/J`, so **hegna should
   leak considerably more than the toroid's 4.6e-09**, since a shaped
   stellarator varies far more than 34% over a surface.

Prediction (2) is what decides §10.1: if hegna returns ~1e-9 the weight-spread
explanation is wrong and should be dropped; if it returns 1e-5 or worse, the
case for exact histopolation stops being "it might help" and becomes a number.

## 9. Solving for lambda instead of approximating it

`booz_xform`'s documentation treats lambda as GIVEN — *"we assume that we know
the quantity lambda"* — because it is an equilibrium output. But at FIXED
geometry it is determined by a small LINEAR problem.

B is linear in lambda, so `W = 1/2 int (B^i g_ij B^j)/J` is quadratic in it.
Setting `dW/dlambda = 0` and using `g_ij B^j/J = B_i`:

```
d_chi(Phi' B_zeta) - d_zeta(Phi' B_chi) = 0   <=>   (curl B).grad rho = 0
```

The energy-minimising lambda is exactly the one making the **current tangent to
the flux surfaces**. Written out:

```
div(A grad lam) = -div(b),   A = adj(G)/J,   b = (1/J)(g_zz + iota g_cz,
                                                      -g_cz - iota g_cc)
```

* `A` is the adjugate of an SPD 2x2, hence SPD: **elliptic**.
* lambda enters only through ANGULAR derivatives, so **flux surfaces decouple
  completely** — one independent 2-D problem per rho.
* the domain is a 2-torus, so a truncated Fourier basis makes each surface a
  small dense SPD solve. Nullspace = constants = lambda's gauge.

No equilibrium iteration, no data file, any geometry.

**Self-consistency:** with `g_cz = 0` and no zeta dependence it collapses to
`d_chi[g_zz(1+lam_chi)/J] = 0`; for `toroid_map`, `g_zz/J = R/(eps^2 rho)`,
giving `(1 + lam_chi) ∝ 1/R` — exactly (2). So (2) is not ad hoc; it is the
axisymmetric solution of the lambda equation.

### 9.1 LANDED: it works, and it is cheap

| run | result |
| --- | --- |
| `lws_toroid` (axisymmetric, mpol=10) | general solve vs the closed form (2): worst relative residual **1.30e-08** over 17 surfaces, 4e-14 near the axis. **207 ms/surface** |
| `lws_hegna` (nfp=3, mpol=8 ntor=6, 220 coeffs) | vs GVEC's OWN lambda: median corr **+0.9984** on `lam_chi`, **+0.9992** on `lam_zeta`; median relative residual **0.061** and **0.045**. **3.06 s/surface** |

`lws_toroid` is mutual validation: the general elliptic solve and the closed
form (2) were derived independently, so agreeing to 1e-8 confirms both, and
confirms (2) is the axisymmetric special case of the lambda equation rather than
a coincidence.

`lws_hegna` is the headline. **A data-free, fixed-geometry energy minimisation
reproduces a real stellarator's lambda to about 5% in both components.** The bar
it clears: the 1/R closed form manages corr +0.83 on `lam_chi` and captures NONE
of `lam_zeta` by construction -- and `lam_zeta` is the ~2x larger effect
(§5.4). The comparison is also UNFAIR to the solve, which was fed a constant
`iota = 0.17` while hegna's actual `iota_MRX` runs 0.150 -> 0.237; it still hits
0.99 correlation, which says lambda here is dominated by geometry rather than by
iota, and that feeding the true profile should only improve it.

Cost is the practical point: setup (sequence + nullspaces) takes 190-240 s in
these runs, so solving lambda on every flux surface is ~1.5% of setup. The warm
start is effectively free.

**The limit is the edge, and it is MODEL ERROR, not truncation.** `lam_zeta`
degrades to residual 0.378 (corr +0.943) at rho = 0.95, `lam_chi` to 0.104.
Job 16764437 re-ran the identical case at `mpol=14, ntor=10, n_ang=48` -- 608
coefficients against 220, a 2.8x increase -- to separate the two candidates:

| rho | `lam_chi` 220 -> 608 | `lam_zeta` 220 -> 608 |
| --- | --- | --- |
| 0.100 | 0.0850 -> 0.0853 | 0.0452 -> 0.0479 |
| 0.525 | 0.0489 -> 0.0490 | 0.0394 -> 0.0406 |
| **0.950** | 0.1040 -> **0.1046** | 0.3778 -> **0.3769** |
| median | 0.0606 -> 0.0607 | 0.0452 -> 0.0479 |

**Nothing moved.** The edge residual changes by 0.2% and several values get
marginally WORSE. The Fourier series was already converged at 220 coefficients,
so the residual is the FIXED-GEOMETRY assumption: VMEC relaxes R, Z and lambda
together and we hold the surfaces frozen, which costs most at rho = 0.95.

Two consequences worth recording, because both stop work that looked justified:

* **Discretising the lambda equation on the 2-D `r = const` surfaces with the
  FEEC machinery would buy NOTHING in accuracy.** A spline space would reproduce
  these numbers; the error is not in the basis. Improving lambda requires
  letting the surfaces move, i.e. solving the VMEC problem, not rediscretising.
* The cost argument for splines also fails: 3.06 -> 3.38 s/surface for 2.8x the
  coefficients, so the DENSE O(N^3) mode solve is not the bottleneck (quadrature
  is) at any resolution worth using. The remaining reasons to want surface FEEC
  are local radial refinement and one-code-path tidiness -- both real, neither
  worth building 2-D surface geometry plumbing that does not exist.

Incidental: the hegna map came back `sign=-1` (mirrored, as `gvec_geometry.py`
warns) and every correlation is still POSITIVE, so the mirror does not flip
lambda's sign relative to GVEC's.

**CAVEAT:** this is the FIXED-GEOMETRY lambda. Full VMEC varies R, Z and lambda
together, so the surfaces relax too; ours cannot. `lws_hegna` measures what
that costs. The bar to beat: the 1/R form (2) manages corr **+0.83** on
`lam_chi` (76% of whose variance is axisymmetric) and captures **none** of
`lam_zeta` — which is 0.000 axisymmetric by construction and the ~2x larger
effect.

## 10. Traps found, both would have been silent

**`load(frame='ref')` does not take the primal reference components.** `M_2`
carries a `g/J` weight (`M2_ij = int Lambda_i^T g Lambda_j / J`), so
`M_2^{-1} load` returns `omega` with `B_phys = DF omega / J` — what
`DiscreteFunction` evaluates — while `frame='ref'` pairs its argument straight
against the basis and therefore wants `g omega / J`. Handing it `omega` builds
a different field, off by a component- and rho-dependent metric factor **even
on a cylinder**. Fix: push `omega` forward and use `frame='phys'`, which
recovers `g omega / J` on its own.

**Histopolation is unavailable for this IC.** It would be the structure-
preserving route (local, tensor-product, commutes with `d`), but
`_require_full_tensor_space` rejects any nontrivial extraction — which rules out
both `dirichlet=True` (`n2_dbc < basis_2.n`) and `polar=True`, and this IC is
both. `interpolate` gained `frame='ref'` for k=1,2 (the counterpart `load`
already had) in anticipation, but it sits behind that guard. The scripts
therefore use `load` + `M_2^{-1}` and MEASURE what the metric coupling costs
(the `B^rho` gate) rather than assuming it away.

### 10.1 Unlocking it is cheaper than it first looked

The literature route is exactly "histopolate on the FULL tensor-product space,
then restrict with one coefficient apply" —
[arXiv:2505.15996](https://arxiv.org/html/2505.15996v1) (Güçlü & Campos Pinto),
which defines

```
Pi_Z = P_Z . Pi_W
```

with `Pi_W` the tensor-product geometric projector and `P_Z` a **local,
explicit, matrix-free** conforming projection acting on coefficients.
Idempotency comes from the coefficient rules being self-consistent, NOT from a
biorthogonality condition: for the C^0 sequence (Thm 3.8) `P_Z` averages the
pole coefficients (`phibar_0j = (1/n_theta) sum_k phi_0k`), zeroes the `i=0`
circumferential coefficients, and sets the `i=1` ones to finite differences
(`vbar^theta_1j = v^s_0(j+1) - v^s_0j`); the near-axis parameters come from
angular DFTs (`gamma_1 = (2/n_theta) sum_j phi_1j cos theta_j`, likewise sin).

**`_histopolate_2form` already implements this shape** — it computes the full
tensor coefficients and returns `e @ concat(c0, c1, c2)`. Only the guard stops
it running. So the cost is:

| case | work |
| --- | --- |
| `E^T E` idempotent (MRX's `e` IS the conforming projection) | delete one guard call + a test — hours |
| not idempotent | port the paper's explicit local `P_Z` — ~a day, formulas given |

`scripts/debug/extraction_unitarity_probe.py` measures exactly the paper's
criterion (`\|\|P^2 - P\|\|` for `P = E^T E`, plus `\|\|E E^T - I\|\|` and the row
structure) across k=1,2,3, free and dbc, polar and non-polar. **Job 16762711,
queued.**

Two things already known: `BoundaryOperator._element` returns 0/1 with exactly
one nonzero per row, so the NON-polar DBC extraction is a pure SELECTION and
`e @ c_full` is trivially correct there — the guard is simply too strict for
that case. And the C^0 formulas above do not transfer verbatim: MRX runs C^1
polar (`xi` shape `(3, 2, n_theta)`), so the coefficient rules differ.

Note the priority: §7.1 measured `B^rho` at **1.1e-16 in the bulk** on the
cylinder through the L2 route, so this may be unnecessary. `lic_gvec` (hegna,
real theta-zeta coupling) is the arm that decides whether it is worth doing at
all.

## 11. Data note

**hegna is the only finite-beta export** (p: 1.0e5 -> 2.1e4 Pa);
`quasr9983`, `quasr44970` and `w7x_vacuum_co_contra` all have `beta_max = 0`.
`data/w7x_ini_mrx.h5` has `beta_mean = 5.8%`, `beta_max = 13%` — a genuinely
finite-beta W7-X — but carries only `B` and `pressure`, **no `clebsch/` group**.
Re-exporting that one with `dPhi_dr / dchi_dr / LA` would make it the best
target in the set.

## 12. Method note: two "decisive" arms that could not decide

Worth recording because it happened TWICE in one session, out of roughly a dozen
designated arms, and the same cheap move caught both.

**`aic_tor_vacuum` (§8.1).** Labelled decisive on the grounds that eq. (2) at
`iota = 0` "should BE the vacuum field". The derivation behind it was correct
and carried a `<1/R>` factor in `Phi'`. The script then exposed only
`--flux-exp`, i.e. `Phi' = rho^q`, which cannot express `rho <1/R>`. So the arm
never contained the vacuum field at all. The run completed, every structural
gate passed, and only the physics was wrong.

**`gvec_clebsch_ic.py:329` (§8.2).** Labelled the real test of §10.1, i.e. of
whether the L2 leak scales with metric weight spread. But it reported ONE global
scalar for `B^rho`, so it could emit no radial information and could not have
discriminated the candidate mechanisms whatever the answer came out to be.

The failure mode is identical: **stating what an arm would prove without
checking what it could actually emit.** In the first case the input
parameterisation could not reach the intended field; in the second the output
was too coarse to carry the intended signal. Neither shows up as an error --
both runs succeed and produce plausible numbers.

What caught both was the same move, and it is cheap: **read the source of the
thing that is supposed to produce the evidence, not the claim about it.**

> **Guard.** Before calling an arm decisive, check two reachable sets against
> the discriminating quantity: what the input knobs can EXPRESS, and what the
> output actually EMITS. A correct derivation upstream does not survive a
> parameterisation that cannot represent it, and a correct mechanism question
> does not survive a diagnostic that integrates the answer away.

Corollary observed here: a scalar summary of a field that varies over the domain
is almost never the right diagnostic for a question about WHERE something
happens. Report the profile.

## 13. Sources

- booz_xform theory (VMEC angles, lambda and nu):
  https://hiddensymmetries.github.io/booz_xform/theory.html
- educational_VMEC `vmec_info.md`:
  https://github.com/jonathanschilling/educational_VMEC/blob/master/vmec_info.md
- Landreman & Sengupta, near-axis expansion at arbitrary order:
  https://arxiv.org/pdf/1911.02659
- Guclu & Campos Pinto, broken-FEEC on polar domains with tensor-product
  splines (the Pi_Z = P_Z . Pi_W construction; use the HTML render, the PDF
  extract mangles the equations): https://arxiv.org/html/2505.15996v1
- Toshniwal, Speleers, Hiemstra & Hughes, multi-degree C^k smooth polar
  splines, CMAME 316 (2017) 1005-1061 -- the extraction operator itself.
- Hirshman, Transformation from VMEC to Boozer Coordinates:
  https://princetonuniversity.github.io/STELLOPT/docs/Transformation%20from%20VMEC%20to%20Boozer%20Coordinates.pdf
