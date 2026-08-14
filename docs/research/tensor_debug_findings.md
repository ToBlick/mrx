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

The later `eps` sweep sharpened that conclusion. Comparing `eps = 1/3` against
`eps = 1/7` at `ns = (16, 16, 16)` and `p = 3` showed that the lower-`eps`
geometry is uniformly easier, but not by the same amount in every regime.

| k | rank | inner | `cp_err` change | iteration change | time change |
| --- | ---: | :---: | ---: | ---: | ---: |
| 0 | 1 | n/a | `-56.6%` | `-32.8%` | `-27.8%` |
| 0 | 2 | n/a | `-57.7%` | `0.0%` | `-2.2%` |
| 0 | 4 | n/a | `-57.7%` | `0.0%` | `0.0%` |
| 1 | 1 | on | `-1.5%` | `-16.8%` | `-15.9%` |
| 1 | 2 | on | `-55.4%` | `-19.4%` | `-17.4%` |
| 1 | 4 | on | `-69.4%` | `-11.3%` | `-9.6%` |
| 1 | 1 | off | `-1.5%` | `-13.7%` | `-13.0%` |
| 1 | 2 | off | `-55.4%` | `-10.7%` | `-10.0%` |
| 1 | 4 | off | `-69.4%` | `-1.6%` | `-1.4%` |
| 2 | 1 | on | `-4.1%` | `-13.0%` | `-12.2%` |
| 2 | 2 | on | `-57.3%` | `-25.0%` | `-22.3%` |
| 2 | 4 | on | `-66.7%` | `0.0%` | `-0.5%` |
| 2 | 1 | off | `-4.1%` | `-8.9%` | `-8.5%` |
| 2 | 2 | off | `-57.3%` | `-9.4%` | `-8.9%` |
| 2 | 4 | off | `-66.7%` | `-2.4%` | `-2.4%` |
| 3 | 1 | n/a | `-60.0%` | `-27.3%` | `-23.6%` |
| 3 | 2 | n/a | `-84.0%` | `-16.7%` | `-15.4%` |
| 3 | 4 | n/a | `-81.9%` | `0.0%` | `+0.9%` |

Here negative percentages mean that the `eps = 1/7` case is easier than the
`eps = 1/3` case. The important pattern is:

- lower `eps` improves fit quality for every mass block, often by more than
  `50%`,
- `k = 1` and `k = 2` see the most meaningful solve-time benefit in the
  practical rank-`2` regime,
- once a block is already near saturation, lower `eps` still improves the fit,
  but does not buy much extra wall-clock speed,
- and the inner-Schur wall-clock penalty remains dominant at both `eps`
  values.

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
- Replacing that old proxy-field multirank builder with the operator-aware
  scalar stiffness fit changed the forward-model picture substantially. On the
  small rotating-ellipse case, extracted Frobenius error dropped from about
  `18.5%` at rank `1` to `11.7%` at rank `2` and `10.7%` at rank `4`, while
  the corresponding bulk-only errors were about `25.3%`, `16.0%`, and `14.6%`.
  So the redesign made rank help monotonically again and confirmed that the
  main remaining miss is still in the bulk model.
- But the first scalar-Laplace solve benchmark with that new builder showed a
  remaining inverse-quality gap: at `ns = (6, 8, 4)`, `p = 3`, Dirichlet rank
  `1` solved in about `25` iterations / `6.68 ms`, while rank `2` and rank
  `4` took about `30.5` / `7.92 ms` and `29.5` / `7.64 ms`. So the new builder
  is better as an operator model, but the higher-rank inverse is not yet a
  practical solver win.
- The later large-case mixed benchmark at `ns = (16, 32, 8)`, `p = 3` made the
  same point much more strongly. For scalar `k = 0` stiffness, rank `1` took
  about `58.8` iterations / `287.8 ms`, rank `2` degraded to about
  `107.2` / `518.6 ms`, and rank `4` failed at the `1000`-iteration cap, even
  while the CP fit error improved monotonically from about `3.65e-1` to
  `7.59e-2` to `1.08e-2`.
- That same large-case sweep also confirmed that the mass-side tensor story is
  still healthy. `k = 0` mass strongly prefers rank `2`; `k = 3` is already
  saturated by rank `2`; and `k = 1, 2` keep improving in iteration count with
  higher rank while wall-clock still clearly favors `inner_schur = off`.
  On this case, rank `4` with `inner_schur = off` was the fastest of the
  tested tensor variants for both `k = 1` and `k = 2`, while the
  inner-Schur-on variants remained much too expensive in time.

So the current forward-model reading is:

- higher rank is genuinely useful for all mass blocks, with rank `2` the main
  practical winner,
- eager production assembly now reflects that by defaulting the mass blocks to
  per-degree rank `2` while keeping scalar stiffness on its rank-`1` fallback,
- scalar `k = 0` stiffness remains an inverse-construction problem after the
  fit-target fix rather than a simple rank shortage,
- `k = 2` higher-form tensor modeling is viable, but rank `1` is too
  restrictive after extraction on the tested mapped case.

One useful design lesson from the later prior experiments is that the fit and
the inverse should be discussed separately. The current toroidal-prior path
uses analytic geometry only to normalize the coefficient fields before CP-ALS.
If `P` is the prior and `C_fit` the learned residual, then the modeled field is
reconstructed as `P * C_fit`, which is multiplicative at the coefficient level
but becomes an additive sum of separable operator terms after expansion. The
inverse is still the same shared-modal multirank inverse.

That is different from an explicit `eps`-expansion with Richardson correction.
In that alternative route one would keep an analytic low-rank backbone
`A_0 + eps A_1 + ...` separate, apply or solve that structured backbone using
the underlying 1-D factors, and then use Richardson on the full operator
residual. The advantage of the shared modal basis is generic multirank
robustness; the advantage of the expansion/Richardson route is that it can
preserve more of the simple low-rank structure and the associated cheap 1-D
solves. The benchmark story therefore does not reduce to "better fit or better
inverse". It is also about which inverse family best exploits the structure of
the fitted model.

The current preferred next experiment sharpens that further. Instead of either
keeping the fully multiplicative prior or introducing many symmetric additive
branches at once, the most promising hybrid is:

- a rank-`1` leading analytic backbone term,
- followed by one correction channel that carries the remaining rank budget,
- fitted greedily by residual subtraction.

In other words, the next candidate model is not
`P * C_fit` and not yet a fully coupled additive hierarchy. It is a split of
the form `B0 * C0 + B1 * C1_tilde`, where `C0` is rank `1` so the leading term
remains easy to invert, and `C1_tilde` absorbs the remaining error. That is the
most direct way to test whether preserving a true rank-`1` backbone is more
valuable than squeezing a slightly better global coefficient fit out of the
shared-modal route.

The first split-fit checks clarified both the promise and the limitation of
that idea.

- For `mass-k0` at `ns = (4, 8, 4)`, `p = 3`, split rank `2` preserved the
  intended structure very cleanly: the backbone carried about `99.3%` of the
  tensor norm and the correction-to-backbone norm ratio was only about `0.12`.
  So `k = 0` is genuinely in a small-correction regime.
- For `mass-k1` at the same size, split rank `2` was too rigid. The `arr` and
  `theta` blocks still had backbone-only residuals around `36%` to `38%`, and
  the final forward model was much worse than the multiplicative baseline.
- But `mass-k1` split rank `3` recovered strongly: the forward model became
  better than the multiplicative rank-`2` baseline on that case, with about
  `6.23e-3` Frobenius error versus about `8.09e-3` for the multiplicative
  comparison run.
- The diagnostics also showed that `k = 1` is not in a "tiny correction"
  regime for every block. The `arr` and `theta` correction-to-backbone norm
  ratios were about `0.39` to `0.40`, while the `zeta` block was much smaller,
  around `0.11`.

So the current mass-side split-fit reading is:

- the rank-`1` backbone constraint is not itself the problem,
- one residual rank-`1` term is too small for `k = 1` on the tested mapped
  case,
- allowing a modestly larger residual channel fixes most of that loss,
- and the right solver experiment is now a backbone inverse plus a small number
  of residual-correction steps, not a stricter correction fit.

## 6. Final Takeaway

The final tensor-preconditioner findings are:

- the active tensor applies are algebraically correct,
- scalar tensor routes are mature and strong,
- vector mass tensor routes are mature,
- the dominant practical question for `k = 1` and `k = 2` is not whether to add
  more Schur logic, but whether the extra coupled bulk work pays for itself,
- on the current benchmark family, it does not,
- on the scalar stiffness side, improving the fit target was necessary and did
  improve the model, but it was not sufficient to make higher-rank tensor
  Hodge solves better yet,
- and the remaining weak spots are now identified as bulk-model quality issues
  rather than routing bugs.

That is the final state of the debugging story that should guide further use of
the tensor preconditioners.
