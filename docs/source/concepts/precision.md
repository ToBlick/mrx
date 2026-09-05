# Precision

MRX has a working precision, chosen once per process, and a residual
precision, always float64. Fields, operators and Krylov iterations run in
the working precision; the residual of every solve is evaluated in float64
and the solution accumulated there (iterative refinement), so a float32
run's solves are as accurate as the residual tolerance says, not as the
float32 Krylov iteration alone could make them.

## The switch

Set `MRX_DTYPE` to `float32` (the default) or `float64` before importing
`mrx`. `mrx/precision.py` reads it and sets `jax_default_matmul_precision`
to `"highest"` so that float32 dot products run at full float32 precision
rather than TF32 (the spline derivatives of a stellarator map lose digits
in TF32, and `det DF` went negative at the axis). 64-bit mode is always on
(the residual precision needs it): Python scalars are weakly typed and do
not promote, NumPy-built arrays would, so every built object (a sequence,
its geometry, a preconditioner bundle) goes through `cast_arrays` at the
end of its construction, and the relaxation test asserts that no leaf of
the state leaves the working dtype.

The module exports:

| name | value |
|---|---|
| `DTYPE` | the working dtype (`mrx.DTYPE`) |
| `RESIDUAL_DTYPE` | float64, or float32 with `MRX_RESIDUAL_DTYPE=float32`: plain float32 solves, the configuration of a machine without float64 (a TPU) |
| `REFINE` | `DTYPE != RESIDUAL_DTYPE`: the solves refine |
| `SOLVE_TOL` | the default relative residual of a solve, in the residual precision: 1e-8 at float32 refined, 1e-10 at float64, sqrt(eps) = 3.5e-4 for plain float32 |
| `inner_tol(tol)` | the tolerance of one working-precision pass, the square root of `tol`: two passes per solve, not a second hyperparameter |
| `MAX_PASSES` | 6 |
| `EPS`, `eps(c)`, `sqrt_eps(c)` | the machine epsilon of the working dtype and its multiples |

## One stopping criterion

`mrx.solvers.refine(apply_res, solve, b, x0, tol, norm)` is the outer loop
of every solve: the residual `b - A x` of the true operator, in float64 on
the view (the operator itself in a plain configuration), measured by
`norm` -- the metric-lumped mass atom of the residual's space, `sqrt(r^T P
r)` with `P ~ M^-1`, the L2 norm of the residual's Riesz representative up
to a mesh-independent factor, so the criterion is h-independent -- and
the correction by `solve` from zero, `x` accumulated in float64, until
`norm(b - A x) <= tol norm(b)` or `MAX_PASSES` passes. Under refinement the
inner solve runs to `inner_tol(tol)` per pass, so a cold solve meets 1e-8
in two passes and a warm start with a 1% defect in two as well; in a plain
configuration it runs at `tol` and the loop is the check that its own
preconditioned criterion did not stop short of the true residual. The two
composite solves, the Hodge split and the shifted split, refine on the
pair `(x, w)` their inner solves produce, with the saddle residual of the
pair in the block mass-atom norm (`_pair_loop` in `mrx/operators.py`): one
outer loop, their inner solves stopping on their own criteria at the inner
tolerance.

The residual-precision operator is `DeRhamSequence.residual`: a shallow
copy of the sequence with the geometry, quadrature, extraction and polar
stencils in float64 and the mass applies rebuilt on them (the bases,
incidence and preconditioners are shared), built once per geometry on
first use, about twice the geometry's memory; `None` at a float64 working
dtype. The harmonic forms and the polar cores of the preconditioners are
built on the view as well: their construction was limited by the working
precision, not by the tolerance (`mrx.nullspace._builder`,
[preconditioning.md](preconditioning.md)).

Results come back in the working dtype. A caller that keeps computing with
the accurate solution asks for it, `apply_inverse_mass_matrix(...,
dtype=RESIDUAL_DTYPE)`, and the Leray projection does exactly that for the
force: the gradient part `sigma` it removes is the size of `J x B` while
the force `J x B - sigma` is a thousandth of it at a relaxed state, so
`compute_force` solves for `J x B` in float64, the saddle solve returns
`sigma` in float64, and the force is formed there and rounded once when it
is stored. A float32 force with a tolerance relative to `J x B` carried a
divergence remnant 22 times its own size
(`docs/research/velocity_leray_ab_2026-09-04.md`).

## Solver tolerance

`DeRhamSequence(tol=None)` and every solver in `mrx/solvers.py` default to
`SOLVE_TOL`; an explicit `tol` is used as given, in the residual precision.
`scripts/relax.py --solve-tol` sets it, `--precision` exports `MRX_DTYPE`
before `mrx` is imported.

The tolerance is the one number of a solve; the pass and cut-off constants
are guards. The relaxation ties it to its floor: the force `F = J x B -
grad p` carries the pressure solve's residual relative to `|J x B|` while
`F` is `resid` times that, so the gradient-part remnant's energy term is
`0.1 tol / resid^2` of the descent, a tenth of it at `resid = sqrt(tol)`:
a run to 1e-4 wants 1e-8, a run to 1e-5 wants 1e-10, float64. `relax`
prints that residual at the start and enforces nothing.

## What the working precision buys

At production meshes the step is bound by kernel-launch latency, not
memory bandwidth, so float32 buys memory (half the operators, geometry and
state), not time: on li383 `(16,32,32)` p=3 refined float32 runs at 1.0
s/step, float64 at 1.4, plain float32 at 0.3, all three reaching the same
residual over 2000 steps. Plain float32 is the configuration for runs that
stop near a residual of 5e-4. In float32 storage the per-step energy change
is at the rounding of the stored field: the trace's `dE` sums are right,
its single steps are noise; the sampler's `E` per chunk is exact to the
field's own rounding.

The suite runs in the three configurations (`slurm/suite.sh`,
[testing_strategy.md](testing_strategy.md)).
