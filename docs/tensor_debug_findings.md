# Tensor Preconditioner Findings

This note records the final validated findings from the tensor-preconditioner
debugging and benchmark work.

## 1. What Was Fixed

Three structural issues were settled.

- `k = 0` mass tensor apply now uses the proper scalar surgery-plus-bulk Schur
  structure instead of a weaker block-diagonal surrogate.
- `k = 1` mass tensor apply now matches the actual surgery-first extracted
  structure and supports the optional coupled bulk treatment.
- the public scalar `kind="tensor"` Laplacian route now aliases the assembled
  scalar tensor-Hodge apply rather than an older helper path.

So the current production tensor applies match the intended assembled tensor
models.

## 2. What The Dense Validation Established

The important validation result is algebraic, not historical.

- `k = 0` mass: the production tensor apply matches the inverse of its
  assembled model to machine precision; remaining error is model error.
- `k = 1` mass: the production tensor apply matches the inverse of its
  assembled model to machine precision; remaining error is model quality, not a
  Schur-apply bug.
- scalar `k = 0` Hodge: the production tensor apply matches the inverse of its
  assembled model to machine precision; remaining error is model quality.

So the active production issue is no longer “is the routed algebra wrong?” The
active issue is only how good the tensor model itself is.

## 3. Mass Benchmark Findings

The recent benchmark comparison now gives a clean practical result.

- `k = 0` mass tensor Schur is a clear runtime and iteration winner over whole
  Jacobi and whole Jacobi-Chebyshev.
- `k = 3` direct tensor mass inversion is also a clear runtime and iteration
  winner.
- `k = 1` and `k = 2` mass tensor routes are strong, but the optional inner
  coupled bulk Schur is not the runtime winner on the tested rotating-ellipse
  family.
- For both `k = 1` and `k = 2`, turning the inner bulk Schur on reduces
  iterations only modestly, while increasing runtime by a large factor.

So the current practical reading is:

- keep the inner coupled bulk Schur as an optional comparison or robustness
  mode,
- prefer `k1_inner_schur = False` and `k2_inner_schur = False` as the default
  practical choice on the tested geometry family.

## 4. Chebyshev Findings

The benchmark also clarified the role of Chebyshev.

- Whole-matrix Chebyshev on Jacobi is a useful baseline and can reduce
  iteration counts substantially.
- Chebyshev wrapped around an already strong tensor route usually lowers
  iterations further, but often loses in wall-clock time.

So the current benchmark conclusion is that better iteration counts alone are
not enough to justify extra polynomial work once the tensor route is already
strong.

## 5. Final Takeaway

The final tensor-preconditioner findings are:

- the active tensor applies are algebraically correct,
- scalar tensor routes are mature and strong,
- vector mass tensor routes are mature,
- the dominant practical question for `k = 1` and `k = 2` is not whether to add
  more Schur logic, but whether the extra coupled bulk work pays for itself,
- on the current benchmark family, it does not.

That is the final state of the debugging story that should guide further use of
the tensor preconditioners.
