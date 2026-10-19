# Precision

MRX runs in float64 throughout. Importing `mrx` turns on JAX 64-bit mode
(`mrx/precision.py`). The fields, the geometry, the operators, the Krylov
iterations and the residuals are all float64, and there is no switch. MRX
needs a backend with float64, a CPU or a CUDA GPU.

Earlier versions also ran in float32 and in a mixed configuration, float32
solves refined against a float64 residual. Both are gone. Plain float32
solves stall above their tolerance on shaped maps, and float32 fields leave
`div B` at `1e-6` and the energy change of a step at the rounding of the
stored field. On li383 (48,64,24), one H200 (2026-10-11), mixed precision
saved 7 % of a Newton step (5.62 s against 6.02 s) and 29 % of a gradient
step (0.75 s against 1.05 s). A checkpoint written by such a run is read
into float64.

| name | value |
|---|---|
| `SOLVE_TOL` | the default relative residual of a solve, `1e-8` |
| `MAX_PASSES` | the passes a solve may take |
| `EPS`, `eps(c)`, `sqrt_eps(c)` | multiples of the machine epsilon |

## One stopping criterion

`mrx.solvers.refine` is the outer loop of a solve: the true residual
`b - A x`, measured in the norm of the metric-lumped mass atom of the
residual's space (a cheap proxy of the `M^{-1}` norm, so the criterion is
mesh-independent), corrected by the Krylov solve until
`norm(b - A x) <= tol norm(b)`. A Krylov iteration carries its residual by
a recurrence, and the loop checks that this did not stop short. A mass solve
stops in the norm of its own preconditioner instead, which conjugate gradients
carry without a second apply: the slab model's for `k = 1, 2`, the atom's for
`k = 0, 3`.

The residual that is judged is that of the equation posed, not of the
blocks a composite solve works with. The Hodge split runs under one outer
loop on the pair `(x, w)` with block residuals `upper` and `lower`. With
`w` eliminated the residual of `(eps M_k + L_k) x = b` is
`upper - M_k G M_{k-1}^{-1} lower`, and that is what the loop measures. The
tolerances of its three solves are set by how their residuals count in that
residual ([Preconditioning](preconditioning.md), Hodge split), so that one
pass normally meets the tolerance. At k = 3 the pair is `(u, sigma)` and the
block residual is that of the saddle-point system. The mass atom stands for each inverse
mass in these expressions. The number the loop stops on and the residual
computed outside the solve then differ by up to a factor 1.6
(li383 (48,64,24) and (64,80,32), 2026-10-12).

## Tolerances

`DeRhamSequence(tol=None)` and every solver default to `SOLVE_TOL`. An
explicit `tol` (`scripts/relax.py --geometry.solve-tol`) is used as given.
The tolerance bounds the force residual a run can resolve: the pressure
solve's error enters the force relative to `|J x B|`.

Test tolerances are multiples of `eps()` or of the solver tolerance
([Testing strategy](testing_strategy.md)).
