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
| `test_derham.py` | on shearedE: the spaces form a de Rham complex on every space and parity view, the mass matrices are SPD with `D_k = M_{k+1} G_k`, the extracted derivatives are the raw ones, interpolation commutes with the derivative, the harmonic forms span `(1, 1, 0, 0)`, and `grad zeta` is harmonic on an axisymmetric torus |
| `test_symmetry.py` | on shearedE: every production functional of a symmetric state is the same on the half period and on the whole period, and an odd integrand is not |
| `test_readers.py`, `test_indata.py` | the GVEC, DESC and VMEC readers reproduce synthetic states and li383 with right-handed angles, and every profile of a VMEC input namelist is its closed form |
| `test_fill.py` | on shearedE: the fill from a namelist's boundary is the axis and the boundary mode by mode, the guessed axis maximises its objective, and the fill cut by the planes normal to its axis meets the boundary within 1 % of the minor radius |
| `test_shape_ad.py` | on shearedE: the vacuum field of the shape derivative is harmonic, its derivative matches central differences, and the frame metric is that of the map's Jacobian |
| `test_solvers.py` | every solve inverts its operator to a multiple of `seq.tol`, and conjugate gradients and MINRES stop on the given norm of the true residual |
| `test_poisson.py` | the Hodge Laplacians recover manufactured solutions within measured bands, and the preconditioners of a map precondition a nearby one |
| `test_exact.py` | on the Solov'ev tokamak: the projected field, current and pressure against the exact ones, the observation error of the transform, Newton steps from a perturbed state back to the discrete equilibrium, and the exact pressure and transform as targets are a fixed point next to the exact state |
| `test_flux_label.py` | the label is the exact normalised toroidal flux to the discretisation, and the label at the shells' tolerance is the converged one |
| `test_poincare.py` | the tracer follows closed-form lines exactly, gives the exact transform, and its sections `phi = const` are those of the map's sections |
| `test_seeding.py` | the energy seed at `iota = 3/4` runs: it finds the (4, 3) chain and keeps `div B` at round-off |
| `test_pressure_fit.py` | the posterior of `scripts/pressure_fit.py` is calibrated |
| `test_newton.py` | the second variation reproduces the energy along the ideal flow with `E'(0) = -(u, J x H)`, the spline atom is SPD, gradient-descent steps descend and keep `div B` and the helicity, and the resistive step satisfies its energy identity |
| `test_shells.py` | a shell velocity is a gradient with its divergence, its row is the derivative of the energy less the target pressure's work |
| `test_twist.py` | the slips are divergence-free without a `zeta` part, the energy along them is quadratic with a positive block, the moments are the weak current's, and one slip update zeroes the weak iota residual of the exact transform |
| `test_mesh_motion.py` | the axis finder finds the exact axis on the pole, the transfer by the induction returns there and back to its time error, one move puts the pole on the axis, and on shearedE the cut by the planes normal to the axis keeps the surfaces and carries the field |
| `test_pipeline.py` | the nested-surface solve finds the exact axis, and the whole pipeline from a namelist ends at the exact equilibrium (skipped until the target ramp from the vacuum exists) |

The fixtures of `test/conftest.py`: the Solov'ev tokamak of `test/solovev.py`,
an exact axisymmetric equilibrium written as a wout and a namelist, on one
session-scoped sequence at `(12, 16, 8)`, `p = 2`, graded, and the exact
stellarator shearedE of `test/landreman.py` at `(12, 16, 12)` for the
statements that need a non-axisymmetric domain. No test builds another
sequence, except the whole-period twin of shearedE in `test_symmetry.py`. A
statement on another map installs it on a fixture with the fixture's
preconditioners (`map_installed`). The convergence orders, which need a
second mesh, are measured outside the suite.

The suite's time goes into compiling and solving. Compiled programs
are cached on disk (`MRX_XLA_CACHE`, default `$MRX_DATA/cache/xla`, or
`outputs/xla_cache` in the checkout without `MRX_DATA`). On 8 cluster CPU
cores from a cold cache the suite takes 14 minutes in float64 and 21 in float32 (2026-11-22).

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
