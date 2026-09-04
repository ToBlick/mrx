# Implicit midpoint for the relaxation: the auxiliary-variable scheme (2026-09-04)

Branch `implicit-midpoint` off `static-dynamic-refactor`, commit 0c67abc and
followers. GPU budget 10 h; spent as listed in section 6.

## 1. The claim, and what is exactly conserved

The discrete helicity of `compute_helicity`, `Q(B) = <A, B + B_harm>` with
`A = L_1^{-1} D_1^T M_2 B` the Coulomb-gauge Dirichlet potential and
`B_harm = B - D_1 A`, is a quadratic form in `B`. Along the ideal descent
`dB/dt = D_1 E`, `E = M_1^{-1} load(u x H)`, `H = M_1^{-1} P B`:

```
dQ/dt = 2 E^T P B - phi^T D_0^T P (B + B_harm)      (dA/dt = E - D_0 phi, Coulomb gauge)
      = 2 E^T M_1 H - 0                             (D_0^T P = -P D_2, D_2 B = D_2 B_harm = 0)
      = 2 int H_h . (u_h x H_h) = 0                  (pointwise at every quadrature node)
```

The pairing of the 2-form `B` with the 1-form `E` never sees `B` itself,
only its proxy `H`, and `E` is the projection of `u x H`: the triple
product vanishes for ANY velocity `u`. The semi-discrete flow conserves
`Q` exactly, so the explicit scheme's drift is time-integration error, and
a scheme that evaluates `E` at the MIDPOINT field keeps `Q` exactly:
`Q(B_{n+1}) - Q(B_n) = 2 dt <B_mid, E_mid> = 0`.

**The one condition the code did not meet.** The identity `E^T P B = E^T M_1
H` needs `E` and `H` in the same 1-form space. In `compute_force` `H` is
NATURAL (tangential trace free at the wall, as the proxy of a wall-tangent
`B` must be) and `E` is DIRICHLET (so that `D_1 E` keeps `B . n = 0`). Then
`M_1 E = R^T load_free(u x H)` drops the tangential wall DoFs of the load,
and `E^T P B = int (u x H_h) . Pi_dir H_h` is the wall layer `-int (u x H_h)
. (H_h - Pi_dir H_h)`: zero in the continuum (`u x H` is normal at the
wall), `O(h^p)` discretely. Both schemes leak helicity through it at the
same rate per unit "distance". With `dirichlet_H=True` (`--dirichlet-H`)
the spaces coincide and the midpoint scheme is exact to the solves; the
price is `H_t = 0` at the wall, a boundary layer in the force.

## 2. Why the velocity stays explicit

The first implementation evaluated EVERYTHING at the midpoint, `B_{n+1} =
B_n + dt dB((B_n + B_{n+1})/2)` with `dB` the whole increment (force,
direction, Leray, `E`, curl), a nonlinear fixed point solved by Picard.
It diverges at the line-search `dt`. Measured on li383 (8,16,16) p=2,
gradient direction, float32 (`outputs/midpoint/2026-09-04/10-38-06`):

| sweep | defect | increment norm |
|---|---|---|
| 0 | 1.53 | 7.5e-3 |
| 1 | 1.08 | 4.2e-3 |
| 2 | 1.05 | 2.3e-2 |
| 3 | 1.05 | 0.38 |
| 4 | 1.00 | 8.3 |
| 5 | 1.00 | 4.0e3 |
| 6 | nan | 1.8e12 |

The linearisation of `B -> dB(B)` is the descent operator `-K(B) M_2` with
`K = C_H Pi C_H^T`, `C_H u = curl(u x H)`: on the part of `B` perpendicular
to `H` it is `|H|^2 curl curl`, largest eigenvalue `|H|^2 / h^2`. The
line-search `dt*` is not a time step, it is a ray minimiser, and it sits
35x above the Picard limit `dt lambda_max / 2 < 1` (halving to `dt*/16`
still did not contract in 20 sweeps). Exact line-search descent tolerates
that; a fixed-point iteration does not.

Two rescues were tried and dropped:

* **Anderson acceleration** (depth 2, 5, 10, least squares in the `M_2`
  norm): still 4 halvings and ~100 evaluations per step, defect stalling
  at `1e-2 .. 1e-4` in float32 (`10-45-17/mp_anderson_probe.log`).
* **Laplacian preconditioning** of the defect, `(M_2 + eps L_2)^{-1} M_2 (g(x)
  - x)` with `eps = dt |H|^2_rms / 2`: converges at `dt*` but sublinearly,
  defect 0.87 -> 0.10 in 20 sweeps in float64 (`11-15-50/mp_sweep64.log`).
  The descent operator is soft exactly where the Laplacian is stiff (the
  force-free perturbations, `curl delta B || H`), so the preconditioned
  spectrum is `[1/(1 + eps lambda_max), 1]`, the unpreconditioned one
  flipped, same condition number ~36.

Newton is a Krylov solve inside a Krylov solve (the Jacobian goes through
five inner solves) and was not attempted. None of this is needed: the
helicity identity holds for ANY `u`, so `u` and `dt` can stay the
predictor's and only the induction is midpoint-implicit,

```
B_{n+1} = B_n + dt curl(u x H_mid),   H_mid = M_1^{-1} P (B_n + B_{n+1}) / 2,
```

LINEAR in the increment with contraction constant `dt |u| / 2h`, small
because `u` is the force (`dt* |u| ~ h^2 |F| / |H|^2` under the line
search). Plain Picard, one k=1 mass solve for `H_mid`, one for `E` and the
topological curl per sweep, warm-started.

Smoke, li383 (8,16,16) p=2 float32, 100 steps (`outputs/midpoint/smoke_*`):

| arm | s/step | evaluations/step | energy increases | E removed | helicity drift (abs) |
|---|---|---|---|---|---|
| explicit, gradient | 0.53 | 1 | 0 | 0.0096% | +1.56e-7 |
| midpoint, gradient | 0.58 | 2.01 | 0 | 0.0096% | +1.28e-7 |
| midpoint, L-BFGS | 0.59 | 2.03 | 0 | -- | -1.97e-7 |

One sweep converges (`picard_tol = 10 seq.tol = 3.4e-3` in float32; the
midpoint correction itself is `~1e-3` of the increment, so in float32 the
scheme is barely distinguishable from the explicit one and the helicity
floor is the solver tolerance in both). The float64 arms below are the
test of the claim.

## 3. Energy

The exact change under the scheme is `E_{n+1} - E_n = -dt <u, F_mid>_M`
with the force at the midpoint field: descent while the predictor's
velocity correlates with the midpoint force, second order in `dt`, not the
line search's guarantee. Measured: 0 energy increases in every arm so far.

## 4. The study (li383, `scripts/midpoint_sweep.sh`, `outputs/midpoint_sweep/`)

All arms L-BFGS (m = 1), line search, cfl 0.5, eta = 0, helicity sampled
every 100 steps; `scripts/midpoint_figures.py` draws the traces.

### 4.1 Scheme x H-space, (8,16,16) p=2, float64, 1000 steps

| arm | s/step | eval/step | E removed | ||F|| final | helicity drift abs | relative |
|---|---|---|---|---|---|---|
| explicit, natural H | 0.37 | 1 | 0.0152% | 1.07e-3 | -5.50e-7 | -1.1e-4 |
| midpoint, natural H | 0.39 | 3.54 | 0.0151% | 1.10e-3 | -6.57e-7 | -1.3e-4 |
| explicit, Dirichlet H | 0.38 | 1 | 0.0257% | 4.93e-3 | +2.22e-7 | +4.5e-5 |
| **midpoint, Dirichlet H** | 0.39 | 3.94 | 0.0257% | 5.17e-3 | **+5.09e-12** | **+1.0e-9** |

Sampled drift `H(it) - H_0` (float64, `||B||_M = 1`, `H_0 = 4.987e-3`):

| it | 1 | 100 | 300 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|
| explicit, natural H | -2.6e-7 | -3.8e-7 | -7.2e-7 | -8.3e-7 | -1.0e-6 | -8.1e-7 |
| midpoint, natural H | -4.7e-8 | -2.8e-7 | -6.1e-7 | -7.2e-7 | -8.9e-7 | -7.0e-7 |
| explicit, Dirichlet H | -2.2e-7 | -1.2e-7 | -5.5e-8 | -3.3e-8 | -2.2e-8 | +1.0e-9 |
| midpoint, Dirichlet H | -2.2e-13 | +3.4e-12 | +5.8e-13 | +3.1e-12 | +5.0e-13 | +4.9e-12 |

* With the natural `H` the two schemes drift together: the wall-layer
  leak of section 1, not the time integrator, and the midpoint scheme
  only removes the first step's `O(dt^2)` error (-4.7e-8 against -2.6e-7).
* With the Dirichlet `H` the midpoint scheme holds the helicity to
  `5e-12` over 1000 steps of mean `dt` 2.3 (max 8.3), four to five
  orders below the explicit scheme on the same setup; the Picard defect
  ends at `3e-17`. The explicit Dirichlet-H drift is dominated by its
  first step and relaxes back as the force shrinks (the per-step error is
  `O(dt^2 |E|^2)`).
* Cost: 3.5-3.9 evaluations per step, 5% wall-clock over the explicit
  step (the two k=1 mass solves per sweep are cheap against the force).
  No halving, no unconverged step, energy monotone in every arm.
* Descent is unchanged: the same energy removed and the same force floor
  as the explicit scheme with the same `H` space. The `H` space itself
  matters more than the scheme: Dirichlet `H` removes 1.7x more energy and
  lands on a 5x higher force floor, its boundary layer (`H_t = 0` at the
  wall) being a different problem.

### 4.2 Production mesh (12,24,24) p=3, float32, natural H, 3000 steps

| arm | s/step | eval/step | E removed | ||F|| final (mean last 100) | helicity drift abs | relative |
|---|---|---|---|---|---|---|
| explicit | 0.50 | 1 | 0.0146% | 3.28e-3 (3.33e-3) | +1.44e-8 | +2.9e-6 |
| midpoint | 0.50 | 2.02 | 0.0145% | 2.91e-3 (3.27e-3) | +4.28e-8 | +8.6e-6 |

Same wall-clock to the digit (the extra k=1 mass solve pair per step is
below the noise of the force evaluation), same descent, and both drifts
at the float32 floor: the qoi helicity itself is a k=1 Hodge solve at
`sqrt(eps_32)`, and `picard_tol = 3.4e-3` makes the midpoint step the
explicit one to within the midpoint correction. The "energy increases"
count of the driver (584 and 604 of 3000) is the float32 resolution of
`E ~ 0.5` (ulp 6e-8) against per-step changes of `1e-8` at the end of the
run, in both arms alike. No halving, no unconverged step.

## 5. Verdict

FILLED IN BELOW.

## 6. GPU time

FILLED IN BELOW.
