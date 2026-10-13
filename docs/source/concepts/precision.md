# Precision

MRX has a working precision, for the fields, the operators and the Krylov
iterations, and a residual precision, for the residual and the solution of
every solve. Both are fixed once per process from the environment, before
`mrx` is imported (`mrx/precision.py`):

| configuration | `MRX_DTYPE` | `MRX_RESIDUAL_DTYPE` | default tolerance |
|---|---|---|---|
| float64 (the default) | `float64` (default) | `float64` (default) | `1e-8` |
| mixed | `float32` | `float64` (default) | `1e-8` |
| float32 | `float32` | `float32` | `1e-5` |

In float32 and float64 every solve is a plain Krylov solve. In mixed
precision every solve is an iterative refinement: the Krylov iteration runs
in float32, the residual is formed in float64 and the correction
accumulated there. Float64 is the default. On li383 (48,64,24), one H200 (2026-10-11), a Newton
step costs 6.02 s in float64 against 5.62 s in mixed precision, and a
gradient step 1.05 s against 0.75 s, both at tolerance `1e-8`.
`div B` stays at round-off in float64, where mixed precision leaves `1e-6`. Plain float32 runs on hardware without float64 (a TPU), but its
Laplacian solves do not reach their tolerance on production meshes
([Sharp bits](../sharp_bits.md)).

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
corrected by the Krylov solve until `norm(b - A x) <= tol norm(b)`. The
Krylov iteration inside stops on the same norm of its own residual, so in a
plain configuration a pass that meets its test also meets the outer one and
a single Krylov solve takes one pass. In mixed precision each pass solves to
`inner_tol(tol)`, two passes per solve.

The residual that is judged is that of the equation posed, not of the
blocks a composite solve works with. The Hodge split runs under one outer
loop on the pair `(x, w)` with block residuals `upper` and `lower`. With
`w` eliminated the residual of `(eps M_k + L_k) x = b` is
`upper - M_k G M_{k-1}^{-1} lower`, and that is what the loop measures. Its
three solves stop at a third of the pass tolerance and the two of the
shifted split at a half, because the residual of a split is at most the sum
of those of its solves. The saddle solve of `L_3` is judged on the residual
of the equation for `u`, `r_u + D_2 M_2^{-1} r_sigma`, and on `r_sigma`,
because `sigma` is returned as well. The mass atom stands for each inverse
mass in these expressions. The number the loop stops on and the residual
computed outside the solve then differ by up to a factor 1.6
(li383 (48,64,24) and (64,80,32), 2026-10-12). A Hodge solve whose
right-hand side has an exact part takes two passes in float64: the first
leaves a lower residual at the tolerance, which counts about `1 / h` times
in the residual of the equation.

The residual operator is `seq.residual`: a copy of the sequence with the
geometry, quadrature, extraction and polar stencils in float64 and
everything else shared. The constructor builds it, and every `set_geometry_from_map` hands
it the new geometry. It is `None` when the solves do
not refine. The harmonic forms are built on it and stored in float64:
a form rounded to float32 is `1e-7` from the kernel, and the deflated
solves do not converge below that.

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
