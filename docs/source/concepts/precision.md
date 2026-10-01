# Precision

MRX has a working precision, for the fields, the operators and the Krylov
iterations, and a residual precision, for the residual and the solution of
every solve. Both are fixed once per process from the environment, before
`mrx` is imported (`mrx/precision.py`):

| configuration | `MRX_DTYPE` | `MRX_RESIDUAL_DTYPE` | default tolerance |
|---|---|---|---|
| mixed (the default) | `float32` (default) | `float64` (default) | `1e-8` |
| float32 | `float32` | `float32` | `1e-5` |
| float64 | `float64` | `float64` (default) | `1e-10` |

In float32 and float64 every solve is a plain Krylov solve. In mixed
precision every solve is an iterative refinement: the Krylov iteration runs
in float32, the residual is formed in float64 and the correction
accumulated there. Mixed reaches double-precision tolerances at a cost
between float32 and float64, and it is the default because plain float32
can stall on strongly shaped geometries ([Sharp bits](../sharp_bits.md)).
Plain float32 runs on hardware without float64 (a TPU) and is the fastest.

`scripts/relax.py --geometry.precision {float32,mixed,float64}` and the other
scripts export both variables before importing `mrx`.
`mrx.precision.current_precision()` names the configuration a process runs
in, and every checkpoint records it.

## The switch

`mrx.precision` sets `jax_default_matmul_precision` to `"highest"`, so
float32 products are not rounded to TF32 on a GPU or to bfloat16 on a TPU,
and keeps 64-bit mode on (the residual precision needs it). Python scalars
are weakly typed and do not promote. Every built object (a sequence, its
geometry, a preconditioner bundle) goes through `cast_arrays`, which pins
every stored floating array to the working dtype.

| name | value |
|---|---|
| `DTYPE` | the working dtype (also `mrx.DTYPE`) |
| `RESIDUAL_DTYPE` | the residual dtype |
| `REFINE` | `DTYPE != RESIDUAL_DTYPE`: the solves refine |
| `SOLVE_TOL` | the default relative residual of a solve (table above) |
| `inner_tol(tol)` | the tolerance of one working-precision pass of a refined solve, `sqrt(tol)` |
| `MAX_PASSES` | the passes a refined solve may take |
| `EPS`, `eps(c)`, `sqrt_eps(c)` | multiples of the working dtype's machine epsilon |

## One stopping criterion

`mrx.solvers.refine` is the outer loop of every solve in every
configuration: the true residual `b - A x` of the outer operator, measured
in the norm of the metric-lumped mass atom of the residual's space (a
cheap proxy of the `M^{-1}` norm, so the criterion is mesh-independent),
corrected by the Krylov solve until `norm(b - A x) <= tol norm(b)`. In a
plain configuration the loop checks that the Krylov iteration's own
preconditioned criterion did not stop short. In mixed precision each pass
solves to `inner_tol(tol)`, two passes per solve. The composite solves (the
Hodge split and the shifted split) run under one outer loop on the pair of
unknowns their inner solves produce, with the saddle-point residual.

The residual operator is `seq.residual`: a copy of the sequence with the
geometry, quadrature, extraction and polar stencils in float64 and
everything else shared. The constructor builds it, and every `set_map` hands
it the new geometry. It is `None` when the solves do
not refine. The harmonic forms are built on it and stored in the working
dtype.

Results come back in the working dtype. A caller that keeps computing with
the accurate solution asks for it with `dtype=RESIDUAL_DTYPE`:
`compute_force` solves for `J x B` that way, because the force
`J x B - sigma` is a small difference of large fields and is formed in the
residual precision before it is rounded once.

## Tolerances

`DeRhamSequence(tol=None)` and every solver default to `SOLVE_TOL`. An
explicit `tol` (`scripts/relax.py --geometry.solve-tol`) is used as given, in the
residual precision. The tolerance bounds the force residual a run can
resolve: the pressure solve's error enters the force relative to
`|J x B|`, so a run aimed far below the float32 floor wants mixed or
float64. In float32 storage the per-step energy change is at the rounding
of the stored field. Sums of the trace's `dE` are right, single steps are
noise.

Test tolerances are multiples of `eps()` or of the solver tolerance, so
the same assertion holds in every configuration
([Testing strategy](testing_strategy.md)).
