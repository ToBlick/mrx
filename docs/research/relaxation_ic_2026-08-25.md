# Relaxation initial conditions from logical profiles — 2026-08-25

How to build a relaxation IC without interpolating `B` from GVEC. The short
answer is that GVEC's representation and ours are the *same object*, so the
field can be rebuilt from three scalars instead of resampled as a vector — and
when no data exists at all, the same ansatz gives analytic ICs and, on a
cylinder, exact analytic equilibria.

Status: derivations complete and checked against data; the GPU arms were
queued at the end of the session and had not landed (`§7`).

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

## 7. Open — the GPU arms

Queued at session end, none landed. Output dirs:
`outputs/analytic_ic/2026-08-25/03-30-32/`,
`outputs/logical_ic/2026-08-25/03-02-51/`,
`outputs/lambda_ws/2026-08-25/03-49-49/`.

| arm | what it decides |
| --- | --- |
| `aic_sp_{sheared,flat,zero,q2}` | cylinder: `\|\|F\|\|/\|\|B\|\|` must sit at discretisation error, and `dp/drho` must match (1) with NO fitted parameter |
| `aic_tor_vacuum` | **decisive**: at iota=0 the closed-form lambda should BE the vacuum field (J=0), so its force must collapse; lambda=0 leaves O(1) |
| `aic_tor_sheared` | residual force EXPECTED — see §8 |
| `lic_cyl` | the six structure gates on the L2 route |
| `lic_gvec` | GVEC reconstruction; the invariance test (H and the iota column must not move with `--no-lambda`; force and pressure must) |
| `lws_toroid` | general lambda solve vs the closed form |
| `lws_hegna` | general lambda solve vs GVEC's own lambda |

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
(the `B^rho` gate) rather than assuming it away. Lifting the guard for
selection-type extractions is the follow-up if that number is large.

## 11. Data note

**hegna is the only finite-beta export** (p: 1.0e5 -> 2.1e4 Pa);
`quasr9983`, `quasr44970` and `w7x_vacuum_co_contra` all have `beta_max = 0`.
`data/w7x_ini_mrx.h5` has `beta_mean = 5.8%`, `beta_max = 13%` — a genuinely
finite-beta W7-X — but carries only `B` and `pressure`, **no `clebsch/` group**.
Re-exporting that one with `dPhi_dr / dchi_dr / LA` would make it the best
target in the set.

## 12. Sources

- booz_xform theory (VMEC angles, lambda and nu):
  https://hiddensymmetries.github.io/booz_xform/theory.html
- educational_VMEC `vmec_info.md`:
  https://github.com/jonathanschilling/educational_VMEC/blob/master/vmec_info.md
- Landreman & Sengupta, near-axis expansion at arbitrary order:
  https://arxiv.org/pdf/1911.02659
- Hirshman, Transformation from VMEC to Boozer Coordinates:
  https://princetonuniversity.github.io/STELLOPT/docs/Transformation%20from%20VMEC%20to%20Boozer%20Coordinates.pdf
