# Testing strategy

MRX has one lean test tier in two precisions. Bare `pytest` runs the suite
in float64, and `MRX_DTYPE=float32 pytest` in float32 with 64-bit mode off
([Precision](precision.md)). `bash slurm/suite.sh` submits both as GPU jobs,
and the GitHub workflow runs both on a CPU. The suite reads only files
tracked in the repository.

## The statements

A test states a mathematical fact about the discretisation or the method, an
equation whose two sides come from different code: a solve and an apply, a
flow and a formula, a closed form and a reader, a difference quotient and a
derivative. A feature is covered by the statements it depends on, not by a
file of its own, and no test runs `scripts/relax.py`.

| file | statement |
|---|---|
| `test_double_single.py` | the error-free transformations of double-single arithmetic are exact under the compiler and its kernels match float64 on the host |
| `test_derham.py` | the spaces form a de Rham complex: `G_{k+1} G_k = 0` on every space and parity view, the mass matrices are SPD with `D_k = M_{k+1} G_k` and adjoint projection pairs, the extracted derivatives are the raw ones, interpolation commutes with the derivative, axis included, the harmonic forms span the cohomology `(1, 1, 0, 0)` of the solid torus, and the distance of `grad zeta` from the harmonic 1-form vanishes on an axisymmetric torus |
| `test_readers.py`, `test_indata.py` | the GVEC, DESC and VMEC readers reproduce synthetic states and li383 with right-handed angles, every profile type of a VMEC input namelist is its closed form, or its density's integral for a current, with the namelist's units and signs |
| `test_fill.py` | the map filled from a namelist's boundary is the axis at `rho = 0` and the boundary at `rho = 1` mode by mode, the guessed axis leaves the fill unfolded and maximises its objective, and the fill cut by the planes normal to its axis on the fixture's spaces meets the boundary within 1 % of the minor radius with a positive Jacobian |
| `test_shape_ad.py` | the vacuum field of the shape derivative is harmonic, its derivative matches central differences, and the hand-derived frame metric is that of the map's Jacobian |
| `test_solvers.py` | every solve inverts its operator to a multiple of `seq.tol`, and conjugate gradients and MINRES stop on the given norm of the true residual |
| `test_poisson.py` | the Hodge Laplacians k = 0..3 recover manufactured solutions within measured bands, with the same error on the two representations of the domain, the preconditioners of a map precondition a nearby one, and the slab model conditions `M_1` |
| `test_flux_label.py` | the label of a field with nested surfaces is constant on them, falls to the wall and encloses `r^2`, and the label at the shells' tolerance, cold or warm-started, is the converged one to twice that tolerance |
| `test_poincare.py` | the field-line tracer follows closed-form lines exactly, gives the file's transform, and its sections `phi = const` are the sections `zeta = const` on a cylindrical map and those of the file's map on the cut one |
| `test_pressure_fit.py` | the posterior of `scripts/pressure_fit.py` is calibrated, the least-squares fit when the noise vanishes and the prior where no data lie |
| `test_newton.py` | the second variation reproduces the energy along the ideal flow, `E'(0) = -(u, J x H)` and `E''(0) = (u, H u)` with polarisation and the Gauss-Newton part, the spline atom is SPD, gradient-descent steps descend and keep `div B` and the helicity, and the resistive step satisfies its energy identity |
| `test_shells.py` | a shell velocity is a gradient with the divergence of its shell function, its row is the derivative of the energy less the target pressure's work |
| `test_twist.py` | a slip changes the field by a divergence-free form without a `zeta` component, its transpose is its transpose, the energy along it is quadratic with a positive definite block, the moments are the weak current's spline means, and one slip update zeroes the weak residual of the target transform |
| `test_harmonic_flow.py` | with stellarator symmetry the force has no harmonic part, and without it the bordered Newton step zeroes the harmonic row |
| `test_mesh_motion.py` | the axis finder puts the file's axis on its pole, the transfer by the induction keeps `div B` and the flux and returns there and back to its time error, one move puts the pole on the axis of the field it carries, and the cut by the planes normal to the axis keeps the surfaces and carries the field |
| `test_solovev.py` | on the exact Solov'ev tokamak (`test/solovev.py`) the start field, its weak current, the label and the transform of its weak moments are the exact ones to measured bands, and the suite's relaxation runs: ten Newton steps reach the round-off of the force without leaving the projection's distance from the exact field, and ten more with the exact pressure and the start's current as targets end at both |

The li383 fixtures of `test/conftest.py` build the equilibrium
(`data/wout_li383_low_res_reference.nc`) at `(16, 32, 12)` per field period,
`p = 2` on the graded polar space of production, in two representations with
the same mesh: `half` (the production sequence, stellarator symmetry, half a
period integrated) and `period` (one field period without the reflection).
`seq` is `half`, `b0` the equilibrium's own field on it. Each is built once
per session. No test builds another sequence of li383: a statement on another
map installs it on the fixture with the fixture's preconditioners
(`map_installed`) and the fixture is put back afterwards. The Solov'ev
fixture is the one other sequence, a cheap axisymmetric mesh with an exact
solution. The convergence orders, which need a second mesh, are measured
outside the suite.

The suite's time goes into the solves, not into compiling. Compiled programs
are cached on disk all the same (`MRX_XLA_CACHE`, default
`$MRX_DATA/cache/xla`, or `outputs/xla_cache` in the checkout without
`MRX_DATA`). On 8 cluster CPU cores from a cold cache the suite takes
17 minutes in float64 and 24 in float32 (2026-11-22).

## What a test asserts

Mathematical statements, phrased so that they fail when the mathematics is
wrong and not when an implementation detail moves:

- exact identities to a multiple of `mrx.eps()`.
- solver-based quantities to a multiple of `seq.tol`.
- measured bands, a fixed factor above a measured error, stated next to
  the value. A wrong metric factor or a broken preconditioner moves these
  by a factor, run-to-run noise by a few percent.
- lengths as a fraction of the device's minor radius (VMEC's
  `sqrt(<A> / pi)`, `test.conftest.minor_radius`), 1 % unless a tighter
  measured bound is stated, so that a bound carries over between devices.

A new test states which mathematical claim it checks at which tolerance class, on the
fixtures above.
Dense references and anything that probes every degree of freedom do not
belong in the suite.
