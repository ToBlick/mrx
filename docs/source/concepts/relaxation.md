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

1. `J = seq.M[1].solve(seq.D[1].T @ B)`, the weak curl (`seq.weak_curl(B)`): one k = 1 mass solve.
2. `load(J x B)`, the Lorentz force tested against the 2-form basis, not projected.
3. `L_1 a = curl^T load(J x B)`: one k = 1 Hodge solve for the force potential `a`. The force is
   `F = curl a + c h`, with `c h` its harmonic part (zero for the even force of a half-period sequence).

`F` is the divergence-free projection of `J x B`, the Riesz representative of `-grad E` among
divergence-free fields, with no pressure solve. Every solve is warm-started from the previous step's value.
`compute_force(B, seq, p_guess, JxB_guess, J_guess, F_guess)` returns the same projection by the Leray
route, a k = 3 saddle solve that also gives the strong pressure `p`. The sampler calls it once per chunk
for that pressure.

## 2. The step

`TimeStepper(seq, newton, newton_penalty, newton_tol, newton_maxiter, resistivity, resistive_reference)`
is an `eqx.Module`. `relaxation_step(state)` does one forward-Euler step
`B_{n+1} = B_n + dt curl(u x B)`:

1. **Direction** `u`. With `newton`, the Newton direction (below). Without,
   the smoothed force,
   `u = curl (M_1 + mu L_1)^{-1} M_1 a = (M_2 + mu L_2)^{-1} M_2 F`, with
   `mu = SMOOTHING_C h_r^2` (`h_r^2` = `radial_cell_sq(seq)`, the squared
   physical radial cell). The smoothing damps the radial two-cell mode on
   any mesh and commutes with the divergence.
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

### Resistive step and drive

`resistive_step(B, seq, eps, B_ref)` is one backward-Euler step of
`dB/dt = -eta curl(curl B - J_ref)` over `eps = eta dt`, `J_ref` the current of
`B_ref`, solved for the increment,
`(M_2 + eps L_2) delta = -eps L_2 (B - B_ref)`
(`seq.shifted(2, eps).solve`, two SPD solves). It is
unconditionally stable and keeps `div B` at the solver tolerance. The drive
of `scripts/relax.py --drive.resistivity C` applies it after every ideal
step with `eps = C h_r^2` towards the reference field's current. The field
then goes to the resistive steady state of that current.

## 3. Diagnostics

- `compute_helicity(B, seq, A_guess)`: `A` from one k = 1 Hodge solve of
  `L_1 A = D_1^T B`, then `H = <A, P_{21}(B + B_harm)>` with
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

`State` holds `B_n`, `B_nplus1`, `dt`, `dt_star`, `cfl_max` and three
subtrees: `warm` (the warm starts), `last` (the last step's force,
velocity and iteration counts) and `best` (the field of lowest residual,
its residual and step). `initial_state(B, ts)` evaluates the force once, so
the first step's solves start warm.

`relax(state, ts, steps, chunk, it0, floor_tol, on_chunk)` runs the steps
in compiled chunks (`chunk_runner`: one `lax.scan` of `chunk` steps,
returning the per-step trace), samples the diagnostics once per chunk
(`make_sampler`: energy in the residual precision, helicity, `||J||/||B||`,
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
