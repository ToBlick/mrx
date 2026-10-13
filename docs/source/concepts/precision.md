# Precision

MRX runs in one of two precisions, chosen by the environment variable
`MRX_DTYPE` before `mrx` is imported (`mrx/precision.py`).

| `MRX_DTYPE` | arrays | JAX 64-bit mode | solves |
|---|---|---|---|
| `float64` (the default) | float64 | on | Krylov, true residual checked |
| `float32` | float32 | off | Krylov in float32, refined in double-single |

`float32` is for a device without float64. No float64 array is created on
the device. `scripts/relax.py --geometry.precision {float64,float32}` sets
the variable before its import, and every checkpoint records it.

## Double-single

A float32 Krylov solve stalls above its tolerance: the residual `b - A x`
cancels to the round-off of `A x`, which is `1e-7` of a vector much larger
than the residual. So in float32 every solve is an iterative refinement.
The solution is accumulated as a double-single number, a pair `(hi, lo)` of
float32 with about 48 bits, and the residual is formed in that arithmetic
(`mrx/double_single.py`). It rests on error-free transformations, which
return a float32 sum or product together with its exact rounding error.
The Krylov iterations, the preconditioners and the projections stay
float32.

There is one rule. An operator applied to a float array returns a float
array, and applied to a pair (`mrx.double_single.DS`) it returns a pair.
`seq.M[k] @ x`, `seq.S[k] @ x`, `seq.G[k] @ x` and the others follow it, so
the same operator code serves the iteration and the residual. Three kernels
have a pair version: the sparse matrices (extraction, polar stencils), the
table contractions of the sum-factorised apply, and the differences of the
incidence. The extraction and stencil weights are kept to pair accuracy,
since the identities `G_{k+1} G_k = 0` near the axis hold to their
round-off. The geometry and the basis tables are float32 data.

`mrx.precision.lift(x)` makes a pair of a float array and
`mrx.precision.to_working(x)` rounds one. Both are the identity in float64.

- A solve returns a float array. With `accurate=True` it returns the
  solution as it was accumulated. Ask for that where the solution is
  subtracted from something close to it: `compute_force` does for `J x B`.
- A right-hand side `G_k^T w` for a stiffness solve
  (`L[k].solve(b, part="coexact")`) must lie in the range of `G_k^T` to the
  accuracy of the solve. Form it from a lifted vector, `G[k].T @ lift(w)`.
- The harmonic forms are computed and stored as pairs.
- The fields of a relaxation are float32. `div B` stays at `1e-6` and the
  energy change of one step is at the rounding of the stored field.

## One stopping criterion

`mrx.solvers.refine` is the outer loop of a solve in both precisions: the
true residual `b - A x`, measured in the norm of the metric-lumped mass atom
of the residual's space (a cheap proxy of the `M^{-1}` norm, so the
criterion is mesh-independent), corrected by a Krylov solve until
`norm(b - A x) <= tol norm(b)`. Each pass asks its Krylov solve for the
reduction that is still missing, with a margin, and never for more than
the precision can deliver: `tol` in float64 and `sqrt(tol)` in float32,
where a solve from zero takes two passes. A part of `b` outside the range
of the operator (a harmonic form, the other parity on a half-period
sequence) is removed from `b` once, in the arithmetic of `b`. The composite
solves (the Hodge split and the shifted split) run under one outer loop on
the pair of unknowns their inner solves produce.

## Tolerances

`DeRhamSequence(tol=None)` and every solver default to
`mrx.precision.SOLVE_TOL = 1e-8` in both precisions. An explicit `tol`
(`scripts/relax.py --geometry.solve-tol`) is used as given. The tolerance
bounds the force residual a run can resolve: the pressure solve's error
enters the force relative to `|J x B|`.

Test tolerances are multiples of `eps()` or of the solver tolerance
([Testing strategy](testing_strategy.md)).

## Measured

li383 (8,12,12), p = 2, one H200, 2026-10-13, float32 against float64.
20 Newton steps: residual `5.618e-8` against `5.626e-8`, 4.29 against
3.12 s/step. 400 gradient steps: residual `5.786e-6` against `5.761e-6`,
0.12 against 0.05 s/step. Not run on a device without float64: the check
there is that the error-free transformations survive its compiler
(`test/test_double_single.py`).
