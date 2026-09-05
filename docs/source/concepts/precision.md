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
rather than TF32. 64-bit mode is always on (the residual precision needs
it); nothing else in the package touches `jax_enable_x64`. Python scalars
are weakly typed and do not promote; NumPy-built arrays would, so every
built object (a sequence, its geometry, a preconditioner bundle) is passed
through `cast_arrays` at the end of its construction, which pins every
stored floating array to the working dtype, and the hot paths give their
constructors an explicit dtype. The relaxation test asserts that no leaf
of the state leaves the working dtype.

The module exports:

| name | value |
|---|---|
| `DTYPE` | the working dtype (`mrx.DTYPE`) |
| `RESIDUAL_DTYPE` | float64, or float32 with `MRX_RESIDUAL_DTYPE=float32`: the float32-only configuration of a machine without float64 (a TPU), plain float32 solves |
| `REFINE` | `DTYPE != RESIDUAL_DTYPE`: the solves refine |
| `SOLVE_TOL` | default relative residual of a solve, in the residual precision: 1e-8 at float32 refined, 1e-10 at float64, 1e-6 for plain float32 |
| `INNER_TOL` | 1e-4, the relative tolerance of one working-precision pass |
| `MAX_PASSES` | 6 |
| `EPS`, `eps(c)`, `sqrt_eps(c)`, `solve_tol(c)` | the machine epsilon of the working dtype and its multiples |

## Refined solves

`mrx.solvers.refine(apply_res, solve, b, x0, tol)` runs the outer loop:
the residual `b - A x` by `apply_res` in float64, the correction by
`solve` in the working precision from zero to `INNER_TOL`, `x` accumulated
in float64, until the residual is below `tol |b|` or `MAX_PASSES` passes.
Each pass takes the residual down by about `INNER_TOL`, so a cold solve
meets 1e-8 in two passes and a warm start with a 1% defect in two as well.
`solve_singular_cg` and `solve_saddle_point_minres` take the
residual-precision operator (`A_res`, `saddle_res`) and refine when given
one; every solve through the sequence passes it (the mass solves, the
k=0 Laplacian, the Hodge-split hat solves, the saddle MINRES, the shifted
split), so every solve returns a result accurate to `SOLVE_TOL` in float64.

The residual-precision operator is `DeRhamSequence.residual`: a shallow
copy of the sequence with the geometry, quadrature, extraction and polar
stencils cast to float64 and the mass applies rebuilt on them (the 1-D
basis tables re-evaluated in float64; the bases, incidence and
preconditioners are shared). Built once per geometry on first use, about
twice the geometry's memory; `None` at a float64 working dtype, where the
solves are plain.

The harmonic forms and the gap estimate are built on the view as well
(`mrx.nullspace`): a once-per-geometry construction whose float32 version
was limited by the working precision, not by the tolerance (the k=2
Dirichlet form's Rayleigh quotient 1e-6 on li383 and 1e-3 on QA in
float32, refined or not, against 1e-16 in float64; the gap sweeps'
shifted saddle solves did not converge in float32). Built on the view at
the float64 default tolerance and stored in the working dtype, the k=3,
k=0 and k=1 forms read 1e-11 or better in a float32 process; the k=2
Dirichlet form reads 1.2e-7 on li383 and 4e-4 on QA (from 9.2e-7 and
1.3e-3), still short of float64, and the gap sweeps still fail. Its
limit is measured (`outputs/prune_smoke/probe_k2.py`): the k=1
Hodge-split solve it rests on reports convergence on the hat operator's
preconditioned residual while its true residual is 2e-6 to 1e-5, and the
form's weak half is that residual squared: a stopping-criterion problem.
The Rayleigh quotient itself is evaluated in the residual precision
(`laplacian_pair`): `v^T L v` is a cancellation that floored at float32
round-off whatever the form.

Results come back in the working dtype. A caller that keeps computing
with the accurate solution asks for it: `apply_inverse_mass_matrix(...,
dtype=RESIDUAL_DTYPE)`. The Leray projection does exactly that for the
force. The gradient part `sigma` it removes is the size of `J x B` while
the force `J x B - sigma` is a thousandth of it at a relaxed state, so
forming the difference in the working precision would cost three digits
and a solve to the working precision's tolerance relative to `J x B`
would leave an O(1) error in the force. `compute_force` therefore solves
for `J x B` in float64, the saddle solve returns `sigma` in float64, the
force is formed there and rounded once when it is stored.

## Why: what float32 alone does

Measured 2026-09-04 on li383 (16,32,32) p=3 at a relaxed state
(`docs/research/velocity_leray_ab_2026-09-04.md`): every solve of the
step converges in float32 to a relative residual of 1e-7, but at the old
default sqrt(eps) = 3.5e-4 the force carried a divergence remnant 22
times its own size (0.04 at 1e-7); the energy could not show a step's
descent at all (`E` moves by less than a float32 ulp per step, which is
why the trace records the exact per-step change `dE` from the increment
instead of `E`); and a second Leray projection of the velocity, the
identity in exact arithmetic, was the most expensive solve of the step
because it was removing that remnant. Float32 with a float64 residual
removes the cause: the solves reach 1e-8 relative to `J x B`, the force
is accurate to float32 rounding, and the residual floor is set by the
storage of `B`, not by the solver.

## What it costs and what it buys

Measured on li383 (16,32,32) p=3, 2000 relaxation steps
(`docs/research/velocity_leray_ab_2026-09-04.md`): the production step
runs at 1.02 s/step in float32 refined (tol 1e-8), 1.44 s/step in
float64 (tol 1e-10; 0.95 at 1.5e-8) and 0.33 s/step in plain float32
(`MRX_RESIDUAL_DTYPE=float32`, tol 1e-6), all three reaching the same
residual over 2000 steps (4.0e-4 to 4.5e-4), while float32 with the old
tolerance sat at 8.4e-4. Plain float32 is the configuration for runs
that stop near a residual of 5e-4: below that the gradient-part term of
its tolerance shows in the energy sums. At these meshes the step is bound by
kernel-launch latency, not memory bandwidth, so the working precision
buys memory (half the operators, geometry and state), not time. The
accuracy of a solve costs about a hundred MINRES iterations per decade
on the k=3 saddle; the tolerance is the cost knob. The velocity's
gradient part relative to the descent grows as `0.1 tol / resid^2`, so a
run aimed at a residual below 1e-4 wants `--solve-tol 1e-10` or float64.
In float32 storage the per-step energy change is at the rounding of the
stored field: the trace's `dE` sums are right, its single steps are
noise; float64 gives a clean per-step trace.

## Solver tolerance

`DeRhamSequence(tol=None)` and every solver in `mrx/solvers.py` default
to `SOLVE_TOL`. An explicit `tol` is used as given, in the residual
precision. `scripts/poisson_study.py --tol` defaults to `1e-9` because the
archived convergence numbers were measured there. `scripts/relax.py` takes
`--precision` (default float32) and stops when the mean over the last
chunk (`--chunk` steps) of the relative force residual drops below
`--floor-tol`. The scripts set `MRX_DTYPE` from `--precision` before
importing `mrx`.

Every test tolerance is expressed through `eps()` or the solver tolerance;
the suite passes in both precisions.
