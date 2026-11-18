# The design of MRX

How MRX works and why, one entry per design choice: what the code does, the reason in a clause, and the page that
explains it. The explanation is in the docs under `docs/source/` (page names below are relative to it), the
numbers in `docs/research/`. Where each piece lives is in `code-map.md`.

What was tried and not adopted is in the "Decided against / removed" sections of `docs/research/preconditioners.md`
(section 10), `docs/research/relaxation.md` (section 6) and `docs/research/solvers_precision.md` (section 9). A method
listed there is reopened only with new measurements.

## Discretisation and spaces

- **A discrete de Rham complex** of tensor-product B-splines, clamped in `r`, periodic in the angles. The
  derivatives are incidence matrices, so `curl grad = 0` and `div curl = 0` hold to round-off and the metric enters
  only through the masses (`concepts/architecture.md` 1-2).
- **1-forms covariant, 2-forms contravariant densities.** Products are metric-free L2 loads, so induction conserves
  flux exactly (`concepts/architecture.md` 2).
- **C^1 polar splines at the axis**, three barycentric functions per `zeta` plane, so a field is smooth where the
  polar map collapses a ring (`concepts/polar.md` 1-2).
- **Graded polar spaces** give each ring at least `n_theta r^(1/3)` theta cells. The exponent keeps order 2, and the
  stiffness solves keep flat counts under refinement. The map stays on uniform theta (`map_theta`). The graded
  spaces are an exact subcomplex (`concepts/polar.md` 3). The measured meshes of `docs/research/` are graded.
- **Matrix-free masses by sum factorisation** with the metric weights at the quadrature points. No matrix is stored
  (`concepts/mass.md` 1-3).
- **Quadrature `p + 1` points in `r`, `p` in the angles.** `p` points keep the optimal rates, the radial axis keeps
  `p + 1` for the `r` and `1 / r` factors of the polar metric (`concepts/mass.md` 4). Applies are 1.5-2.1x faster than
  with `p + 1` everywhere (`docs/research/performance.md` 1).
- **Stellarator symmetry by half-period quadrature and parity views**, half the DoFs on the same kernels
  (`concepts/architecture.md` 6). A half period is 1.4x faster per step than one field period at n = 32 and 1.7x
  at n = 48, li383 p = 2 (`docs/research/performance.md` 4).

## Geometry and input

- **One map, `FrameMap`**, `Phi = R_z(phi) (a + X1 N + X2 B + X3 T)`. Every file is read in the frame its solver
  supplies, cylindrical for VMEC, DESC and cylindrical GVEC states, GVEC's own frame for a G-frame state, with
  `X3 = 0`. No file is cut into another frame. `X3` lets a moving mesh or a shape optimisation move points out of
  their plane. The coefficients are the L2 projection of the file's series onto the polar 0-forms. The map is a
  pytree, so a new geometry of the same mesh reuses every compiled solve and is differentiable
  (`concepts/equilibrium_input.md` 3).
- **The initial field is rebuilt from scalars**: the Clebsch potential on the free 1-forms, `B = dA'`. So
  `div B = 0`, `B . n = 0` and nested surfaces hold exactly and no derivative of `lambda` is sampled
  (`concepts/equilibrium_input.md` 1, `concepts/relaxation.md` 5).
- **Right-handed angles, positive Jacobian.** A VMEC wout is read with `theta -> -theta` (`sharp_bits.md`).

## Solvers and preconditioners

- **Every inverse is a Krylov solve** on matrix-free operators, PCG for SPD operators and MINRES for the Newton
  system. Nothing larger than the dense polar core is stored (`concepts/preconditioning.md`, introduction).
- **Metric-lumping atoms** for every degree: a Kronecker model with axis-mean metric weights, inverted by fast
  diagonalisation, plus the probed polar core. One apply is a few small dense products, and the storage grows like
  the DoFs (`concepts/preconditioning.md` 2). `PRODUCTION_BC_SCALE = 3.0` for every geometry: the count is
  insensitive over about [2, 8] (`docs/research/preconditioners.md` 6).
- **The slab model preconditions `M_1`, `M_2`** (`mass_atom="slab"`). It keeps the radial-poloidal coupling of
  shaped sections, which the atom loses. Condition number about 3.5 against about 45 for the atom, cut to a theta
  band (`THETA_HALF_WIDTH = 4`) so its memory grows like the DoFs (`concepts/preconditioning.md` 2,
  `docs/research/preconditioners.md` 4).
- **Multigrid for the co-closed `S_2`**, opt-in (`laplacian_atom="multigrid"`). `S_2` sees the metric only through
  `1 / J`, and a V-cycle solves it in about 11 iterations on every mesh measured (`concepts/preconditioning.md` 2,
  `docs/research/preconditioners.md` 3).
- **Hodge split for `L_1`, `L_2`**: three SPD solves on the stiffnesses instead of a saddle-point system. Each inner
  tolerance follows from how its residual counts in the posed residual, so one pass meets the tolerance
  (`concepts/preconditioning.md` 1).
- **`L_3` and the Leray projection by the split at k = 3**: two `S_2` solves and one `S_1` gauge, 116 iterations and
  40 ms on li383 graded (48,64,24). The divergence of a projected force falls to 2.5e-9 of it
  (`concepts/preconditioning.md` 1).
- **Shifted solves `M_k + eps L_k`** as three SPD solves. The error of the first cancels, because `G^T S_k = 0` and
  `G^T M_k G = S_{k-1}` (`concepts/preconditioning.md` 1).
- **Part solves.** A co-exact right-hand side needs only the singular `S_k`, which converges because the right-hand
  side lies in its range (`concepts/preconditioning.md` 1).
- **Harmonic forms** from a closed seed by at most one solve each, judged by their Rayleigh quotient
  (`concepts/architecture.md` 4, `docs/research/solvers_precision.md` 6).
- **One stopping criterion**: the true residual of the posed equation in the mass-atom norm, which is
  mesh-independent. A mass solve stops in the norm of its own preconditioner (`concepts/precision.md`). Every
  solve of the relaxation is warm-started from the step before (`concepts/relaxation.md` 1).

## Precision

- **float64 by default, float32 for a device without float64.** In float32 the Krylov iterations stay float32 and
  the solution and residual are double-single, so the solves reach the same `SOLVE_TOL = 1e-8`
  (`concepts/precision.md`).
- **Preconditioners and Krylov directions are float32 in both precisions** (`LOW_DTYPE`, `mrx/precision.py`).
- **Matmul precision "highest".** Lower settings carry 1.9e-4 relative error in `DPhi` or fold the map on a TPU
  (`docs/research/solvers_precision.md` 7).
- **The tolerance sets the floor**: a tolerance buys a force residual of about `sqrt(tol / 10)` on li383 float64
  (`docs/research/solvers_precision.md` 7, `concepts/precision.md` "Tolerances").

## Relaxation

- **Energy descent along divergence-free, wall-tangent flows**, `dB = G_1 E` with `E = M_1^-1 load(u x B)`. So
  `div B` is conserved exactly and the helicity to the solves. The fixed point is `J x B = grad p` with `p` the
  multiplier (`concepts/relaxation.md`, introduction and 2).
- **The force by the potential route**: one `S_1` solve per step, no pressure solve. It equals the Leray route
  (`compute_force`) to 3 digits and is 26-29 % cheaper per step in gradient descent (`concepts/relaxation.md` 1,
  `docs/research/relaxation.md` 1).
- **The Newton direction** solves the Gauss-Newton system, positive semidefinite, for `u = curl a`, divergence-free by
  construction (`concepts/relaxation.md` "Newton"). Settings and their reasons:
  - MINRES with reorthogonalised Lanczos vectors, 1.6-2.6x fewer iterations than without.
  - Preconditioner `spline_preconditioner`, the Laplacian atom scaled by a spline model of the Hessian, one number per
    1-form coefficient. About 70 MINRES iterations per step over steps 11-30 on li383 (48,64,24) and W7-X
    (32,64,36), graded (`docs/research/preconditioners.md` 9).
  - `NEWTON_TOL = 0.1`, `NEWTON_MAXITER = 300`. 0.1 reaches a force residual of 1e-8 first or level on li383 and
    W7-X (`docs/research/relaxation.md` 7).
  - `NEWTON_MASS_SOLVE_TOL = 1e-6` for the two `M_1` solves inside every Hessian apply. MINRES is not flexible, and at
    1e-4 the relaxation stalls.
  - `NEWTON_PENALTY = 3.0` times the strain along the field lifts the field-aligned null space. 3 is within 6 % of
    the best on li383 and W7-X (`docs/research/relaxation.md` 3).
- **Gradient descent** is the smoothed force, one solve of `M_1 + mu S_1` with `mu = SMOOTHING_C h_r^2`
  (`SMOOTHING_C = 0.075`). It damps the two-cell radial mode on any mesh and commutes with the divergence
  (`concepts/relaxation.md` 2).
- **Step size**: the exact line search `dt_star`, capped at `CFL = 0.5` logical cells and at 1 with Newton. The
  maximal step far from equilibrium destroys surfaces (`docs/research/relaxation.md` 2-3).
- **Resistive step** by backward Euler as one 1-form solve of `M_1 + eps S_1`, unconditionally stable with `div B`
  at round-off (`concepts/relaxation.md` "Resistive step and drive").
- **Island seeds** at every resonance in the iota range, profiles from the energy criterion `a* = -G^-1 g`
  (`mrx/relaxation/seeding.py`). A seeded island needs about 3 radial cells (`docs/research/relaxation.md` 5).
- **Stopping**: the chunk mean of `resid` below `--budget.floor-tol`, or the step budget. The field of lowest
  residual is kept as `best.h5` (`relaxation.md` "Stopping").

## Diagnostics

- **Two pressures**: the strong `p` of the descent (`compute_force`, on request) and the weak `p_w`, zero on the
  wall, for profiles and `beta_vol` (`concepts/relaxation.md` "Two pressures").
- **`resid = ||F||^2_M / ||grad(B^2/2)||^2`**, of order one at any beta (`concepts/relaxation.md` 3).
- **Poincare sections** trace with `zeta` as time, all fields of a run in one batch, and render separately
  (`relaxation.md` "Poincare sections", `sharp_bits.md` "Poincare sections").
