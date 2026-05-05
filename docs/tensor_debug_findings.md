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

## 5. Forward-Model Diagnostics

The recent small-case forward-model checks on the rotating-ellipse family make
the model-quality picture much sharper than the solve benchmarks alone.

- `k = 2` `div_div`: the regular-space rank-1 tensor model is decent as a
  forward model, with about `2.1%` Frobenius error on `ns = (4, 8, 4)`,
  `p = 3`.
- But the extracted-space sandwich of that same rank-1 `k = 2` model is much
  worse, with about `12.9%` Frobenius error, and the extracted bulk-only error
  is essentially the same.
- That extracted `k = 2` miss is not a surgery bug. It is mostly a rank issue:
  on the same case, extracted-space Frobenius error drops to about `0.65%` at
  rank `2`, `0.36%` at rank `3`, and `0.058%` at rank `4`.
- Scalar `k = 0` stiffness is the clearest bad rank-1 case: the extracted
  forward-model error is about `33%` in Frobenius norm, and the extracted
  bulk-only error is even worse at about `45%`. So the weakness is in the
  bulk model itself, not in the surgery wrapping.
- Rank-1 mass forward-model quality is degree dependent on the same test case:
  `k = 0` is good (`~1.6%` full Frobenius, `~4.7%` bulk-only), `k = 1` is bad
  (`~24%` full and bulk-only), and `k = 2` / `k = 3` are moderate (`~5%`).
- Higher-rank mass checks changed the practical recommendation substantially:
  rank `2` gave large solve improvements for every mass degree on
  `ns = (8, 16, 8)`, while scalar `k = 0` mass was already essentially exact as
  a forward model at rank `2`. The measured solve counts were roughly
  `11 -> 3` for `k = 0`, `28 -> 14 -> 13` for `k = 1`, `26 -> 14 -> 12.5` for
  `k = 2`, and `11 -> 6 -> 6` for `k = 3` as the rank increased from `1` to
  `2` to `3`.
- Scalar `k = 0` stiffness did not follow that pattern. After fixing the local
  multirank projection bug, rank `2+` no longer blew up, but still did not
  improve the bulk forward model materially. So the remaining stiffness issue
  is not just insufficient rank in the current construction.

So the current forward-model reading is:

- higher rank is genuinely useful for all mass blocks, with rank `2` the main
  practical winner,
- eager production assembly now reflects that by defaulting the mass blocks to
  per-degree rank `2` while keeping scalar stiffness on its rank-`1` fallback,
- scalar `k = 0` stiffness remains a model-construction problem rather than a
  simple rank shortage,
- `k = 2` higher-form tensor modeling is viable, but rank `1` is too
  restrictive after extraction on the tested mapped case.

## 6. Final Takeaway

The final tensor-preconditioner findings are:

- the active tensor applies are algebraically correct,
- scalar tensor routes are mature and strong,
- vector mass tensor routes are mature,
- the dominant practical question for `k = 1` and `k = 2` is not whether to add
  more Schur logic, but whether the extra coupled bulk work pays for itself,
- on the current benchmark family, it does not,
- and the remaining weak spots are now identified as bulk-model quality issues
  rather than routing bugs.

That is the final state of the debugging story that should guide further use of
the tensor preconditioners.
