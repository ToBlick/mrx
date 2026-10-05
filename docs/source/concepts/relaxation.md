# Relaxation

`mrx/relaxation/` (`loop.py`, `physics.py`) descends the magnetic energy `E = ||B||^2_{M_2} / 2` of a
divergence-free Dirichlet 2-form `B` (`B . n = 0`) along an incompressible,
wall-tangent flow, which conserves the helicity up to grid-scale effects. The fixed point is
`J x B = grad p` with `p` the Lagrange multiplier: a finite-beta equilibrium.
`scripts/relax.py` is the driver ([Solve a relaxation problem](../relaxation.md)).
On a half-period sequence `B`, `J` and `E` live on `seq.odd`, the velocity,
the force and the pressures on `seq.even` ([Architecture](architecture.md)).

## 1. The force

The relaxation forms the force by the potential route:

1. `J = seq.M[1].solve(seq.D[1].T @ B)`, the weak curl: one k = 1 mass solve.
2. `load(J x B)`, the Lorentz force tested against the 2-form basis, not projected.
3. `S_1 a = curl^T load(J x B)`: one curl-curl solve for the force potential `a`
   (`seq.L[1].solve(b, part="coexact", gauge=False)`). The force is `F = curl a + c h`, with `c h` its harmonic part (zero for
   the even force of a half-period sequence). `a` is fixed only up to a gradient, which `curl` removes.

`F` is the divergence-free projection of `J x B`, the Riesz representative of `-grad E` among
divergence-free fields, with no pressure solve. Every solve is warm-started from the previous step's value.
`compute_force(B, seq, p_guess, JxB_guess, J_guess, F_guess)` returns the same projection by the Leray
route, a k = 3 saddle solve that also gives the strong pressure `p`. The sampler calls it once per chunk
for that pressure.

## 2. The step

`TimeStepper(seq, newton, newton_penalty, newton_tol, newton_maxiter, newton_mass_tol, resistivity,
resistive_reference)`
is an `eqx.Module`. `relaxation_step(state)` does one forward-Euler step
`B_{n+1} = B_n + dt curl(u x B)`:

1. **Direction** `u`. With `newton`, the Newton direction (below). Without,
   the smoothed force,
   `u = curl (M_1 + mu L_1)^{-1} M_1 a = (M_2 + mu L_2)^{-1} M_2 F`, with
   `mu = SMOOTHING_C h_r^2` (`h_r^2` = `radial_cell_sq(seq)`, the squared
   physical radial cell). The smoothing damps the radial two-cell mode on
   any mesh and commutes with the divergence. Only the curl enters, and
   `(M_1 + mu L_1)^{-1}` and `(M_1 + mu S_1)^{-1}` differ by an exact form,
   so it is one solve of `M_1 + mu S_1`
   (`seq.shifted(1, mu).solve(..., part="coexact")`).
2. **Electric field** `E = M_1^{-1} load(u x B)`, one k = 1 mass solve.
3. **Increment** `dB = G_1 E`, the topological curl: `div B` is conserved
   exactly and the helicity to the solves.
4. **Step size**. `dt_star = <F, u>_M / ||dB||^2_M` minimises the energy
   along the increment, and `dt = min(dt_star, CFL / cfl_max)`, with `cfl_max`
   the largest logical CFL number of `u`, and `dt <= 1` (the Newton length)
   with Newton.
5. **Resistive step**, with `resistivity > 0`: `resistive_step` below,
   towards `resistive_reference`.

The energy change of every step is recorded exactly (`dE`) next to the line
search's prediction (`dE_ls`). For a divergence-free `u` they agree to
round-off. The force residual is not monotone.

### Newton

Along the flow of a divergence-free `u` the energy expands to second order
with the gradient `-load(J x B)` and the symmetric Hessian

```
(u, H v) = (Q_u, Q_v)_M + [(B, curl(u x Q_v))_M + (B, curl(v x Q_u))_M] / 2,    Q_u = curl(u x B),
```

at an equilibrium minus the ideal-MHD force operator at `p = 0`
(`mrx.relaxation.newton.second_variation`). `newton_direction` solves
`curl^T H curl a = curl^T load(J x B)` for `u = curl a`, divergence-free by
construction. The right-hand side needs no projection, because `curl^T`
annihilates the gradient and the harmonic part of the force exactly
(`G_2 G_1 = 0`). It is solved by MINRES preconditioned by the harmonic atom (the Laplacian
atom scaled by the lumped parallel derivative of `B`), warm-started from
the previous potential, until the residual falls below `newton_tol` of the
right-hand side or after `newton_maxiter` iterations, with an exit on
nonpositive curvature. `H` vanishes on field-aligned flows `u = f B`, and a
penalty of `newton_penalty` times the strain along the field lifts that
null space.

Every apply of `H` contains two k = 1 mass solves, so the Newton solve runs
a Krylov solve inside MINRES, and the mass solves dominate its cost. By
default (`newton_mass_tol = 1e-4`, `None` keeps the solves) they are
replaced by `MassChebyshev`, the Chebyshev polynomial of degree `k` in the
mass atom times `M_1`
(`mrx.solvers.chebyshev`). It costs `k` mass applies and runs no loop. It is
linear and symmetric, so `H` stays symmetric and MINRES applies unchanged.
The spectral interval comes from 60 PCG iterations at construction
(`mrx.solvers.lanczos_bounds`), widened by 5 %, and `k` is the smallest
degree whose error bound `2 rho^k` is below `newton_mass_tol`. The
right-hand side and the line search are unchanged, so only the Newton
direction is approximate. The error of `H` is 3 to 4 times that of the mass
inverse.

### Compressible relaxation with an advected pressure

With `TimeStepper(..., compressible=True)` the pressure is prescribed rather
than a multiplier. A divergence-free flow cannot do work against a pressure
(`int u . grad p = 0`), so a prescribed pressure needs a compressible flow.
The state carries `p_n`, a free 0-form on `seq.even.free`, frozen into the
flow at `gamma = 0`:

1. **Force** `F = M_2^{-1} load(J x B - grad p)`, not projected, one k = 2
   mass solve. `load(grad p)` is metric-free.
2. **Direction** `u`, a Dirichlet 2-form with divergence. Without Newton,
   the smoothed force `(M_2 + mu L_2)^{-1} load(J x B - grad p)`.
3. **Pressure** `dp = -M_0^{-1} load(u . grad p)` on the free 0-forms, one
   k = 0 mass solve, and `p_{n+1} = p_n + dt dp` with the step of `B`.
4. **Parallel smoothing** `(M_0 + eps K_par) delta = -eps K_par p`, with
   `K_par` the form `int (b . grad p)(b . grad w)` and
   `eps = PRESSURE_SMOOTHING h_zeta^2` (10 toroidal cells squared,
   `--pressure.smoothing`). The Galerkin advection creates variation of `p`
   along the field, which no equilibrium can balance. The smoothing removes
   it and keeps `int p dV` (the constant is in the kernel of `K_par`), so it
   leaves `L` unchanged.

The step lowers `L = int |B|^2/2 - p dV`, which is quadratic in `dt` (its
pressure part is linear), so `dt_star` keeps its form and `dE`, `dE_ls` are
the changes of `L`. The constant is in the free 0-forms, so
`int dp dV = -int u . grad p dV` holds exactly. The fixed point is
`J x B = grad p` with `p` conserved per flux tube as a function of the
enclosed toroidal flux (the ideal flow freezes both). `int p dV` and beta
are not conserved, since the flow compresses. A pressure that varies along
field lines (in islands or chaos) has no fixed point.

The Hessian of `L` adds to `H` the pressure term
`(u, H_p v) = [l(u)^T M_0^{-1} d(v) + l(v)^T M_0^{-1} d(u)] / 2`, with `l(u)`
the load of `u . grad p` and `d(u)` that of `div u` on the free 0-forms, the
discrete form of `int (u . grad p) div v` (`second_variation(..., p=p)`).
`compressible_newton_direction` solves `H u = load(J x B - grad p)` for `u`
itself by the same Newton-MR, preconditioned by `CompressibleAtom`: the
harmonic atom on the divergence-free part and `(beta S_2)^+`, the cost of
compressing the field, on the rest.

`initial_pressure(seq, B, beta)` gives the start pressure: the equilibrium
file's pressure profile as a function of `r`, scaled to the volume beta
`int p dV / int |B|^2/2 dV`.

### Prescribed pressure by a volume outer loop

`mrx.relaxation.pressure_loop.prescribe_pressure` keeps the incompressible
relaxation and prescribes `p*(s)` from outside. Incompressible flow keeps
the volume `V'(s)` of every flux shell, and the relaxed pressure is the
multiplier of that constraint, so the loop controls `V'(s)` per bin of the
flux label `s` (`mrx.flux_label`). Each iteration relaxes for a few steps,
bins the weak pressure `p_w`, and moves volume with one compressible ideal
step `u = M_2^{-1} D_2^T L_3^{-1} load(g)` (`div u = g`, `u . n = 0`) with
`g = -gain (p_w - p*) / <|B|^2>`. The sign matters: at fixed flux a
compressed shell carries a stronger field, and since `p + B^2/2` is nearly
constant across the surfaces its pressure measured from the wall drops.

### Resistive step and drive

`resistive_step(B, seq, eps, B_ref)` is one backward-Euler step of
`dB/dt = -eta curl(curl B - J_ref)` over `eps = eta dt`, `J_ref` the current of
`B_ref`, solved for the increment,
`(M_2 + eps L_2) delta = -eps L_2 (B - B_ref)`.
For a divergence-free `B - B_ref` the solution is a curl, `delta = G_1 a` with
`(M_1 + eps S_1) a = -eps G_1^T M_2 (B - B_ref)`, one 1-form solve
(`seq.shifted(1, eps).solve(..., part="coexact")`). So `div B` stays at
round-off, since `G_2 G_1 = 0`. It is unconditionally stable. The drive
of `scripts/relax.py --drive.resistivity C` applies it after every ideal
step with `eps = C h_r^2` towards the reference field's current. The field
then goes to the resistive steady state of that current.

## 3. Diagnostics

- `compute_helicity(B, seq, A_guess)`: `A` from one curl-curl solve
  `S_1 A = D_1^T B` (`seq.L[1].solve(b, part="coexact", gauge=False)`), then `H = <A, P_{21}(B + B_harm)>` with
  `B_harm = B - curl A`.
- `compute_divergence_norm(B, seq)`: `||G_2 B||`, no solve.
- `force_scale(seq, B) = ||grad(|B|^2/2)||`: the scale of the force residual
  `resid = ||F||^2_M / ||grad(|B|^2/2)||^2`, O(1) at any beta.
- `weak_pressure(J, B, seq)` and `beta_vol(B, p_w, seq)`: below.

### Two pressures

**Strong** `p` (from `compute_force`, a 3-form, computed by the sampler once
per chunk): the multiplier of the constrained principle. The force is projected onto the Dirichlet 2-forms
first, which discards its normal component, so `dp/dn = 0` on the wall and
`p` is defined up to a constant. It is the right multiplier for the
descent and blind to the wall force. It is part of the state.

**Weak** `p_w` (from `weak_pressure`, a Dirichlet 0-form): `J x B` is
projected onto the natural 1-forms, which keep its normal component, and
split as `v = F_w + grad p_w` with `p_w = 0` on the wall (one k = 0 Dirichlet
solve). At a fixed point `F_w` vanishes and `dp_w/dn` is the wall force.
Read `p_w` for the pressure profile and beta:
`beta_vol = int p_w dV / int |B|^2/2 dV` in code units. It is computed by the
sampler at every chunk and by `scripts/poincare_trace.py` at the crossings.

## 4. The loop

`State` holds `B_n`, `B_nplus1`, `p_n`, `p_nplus1` (the advected pressure, zero unless compressible), `dt`, `dt_star`, `cfl_max` and three
subtrees: `warm` (the warm starts), `last` (the last step's force,
velocity and iteration counts) and `best` (the field of lowest residual,
its residual and step). `initial_state(B, ts)` evaluates the force once, so
the first step's solves start warm.

`relax(state, ts, steps, chunk, it0, floor_tol, on_chunk)` runs the steps
in compiled chunks (`chunk_runner`: one `lax.scan` of `chunk` steps,
returning the per-step trace), samples the diagnostics once per chunk
(`make_sampler`: energy, helicity, `||J||/||B||`,
`int J . B`, `beta_vol`), calls `on_chunk`, and stops when the chunk mean of
`resid` falls below `floor_tol` or after `steps`. It returns a
`RelaxResult` (the state, the trace, the samples, the stopping reason).
`write_checkpoint` / `read_checkpoint` store and restore a state as one
HDF5 file with the discretisation as attributes (`checkpoint_attrs`).

## 5. Initial conditions

`mrx/relaxation/initial_conditions.py` builds the field in the reference 2-form frame
(`det DPhi B^i`):

```
B_ref = Psi'(r) (0, iota(r) - d lambda / d zeta, 1 + d lambda / d theta)
```

with `Psi` the toroidal flux and `lambda` the angle shift to straight field
lines. This field is divergence-free and tangent to the wall for any
`lambda` and any geometry. `initial_field(seq)` builds it from the
equilibrium file with `potential_two_form(seq)`: the Clebsch potential
`A' = (-lambda Psi', 2 pi Psi, -(2 pi/nfp) chi)`, with `chi` the poloidal
flux, is histopolated on the free 1-forms (its wall trace carries the
toroidal flux), and `B = dA'` is taken by the incidence curl into the
Dirichlet 2-forms. So `div B = 0` to round-off, and no derivative of
`lambda` is ever sampled. It also returns the wall-normal part the
Dirichlet restriction discards, a check of the file.

The field is normalised to `||B||_M = 1`.
