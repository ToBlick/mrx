# Mass Preconditioners

This note records the current production picture for the mass preconditioners in
`mrx`. It is intentionally short and only describes the active design.

The strategic status is now simple: the mass preconditioners are essentially
done. The eager default policy is rank `3` for all four mass blocks. The
remaining work is to keep the benchmark picture current and trim stale options
that no longer affect the assembled tensor route.

## 1. Shared Design

The production tensor route does not try to approximate the inverse of the full
extracted mass matrix directly. Instead it:

- keeps the extracted-space surgery rows exact through a small dense Schur
  complement,
- approximates only the bulk tensor blocks,
- fits the diagonal mapped coefficient fields on the quadrature grid,
- and builds tensor-diagonal block inverses from those fitted fields.

The active diagonal coefficient fields are:

- `k = 0`: `J`,
- `k = 1`: `J g^{rr}`, `J g^{theta theta}`, `J g^{zeta zeta}`,
- `k = 2`: `g_rr / J`, `g_theta theta / J`, `g_zeta zeta / J`,
- `k = 3`: `1 / J`.

Higher ranks are supported by the tensor block machinery. Recent solve and
forward-model checks now point to rank `3` as the stronger practical default
across all four mass blocks on the tested rotating-ellipse family.

So the mass question is no longer "which family should we use?" The answer is
the tensor route with eager default rank `3`. The open question is only how
much validation we want to keep in the tree.

## 2. Degree-by-Degree Structure

### `k = 0`

`k = 0` is the scalar surgery case.

- The extracted matrix is split into a small core block and one scalar bulk
  tensor block.
- The core is handled by a dense Schur solve.
- The bulk is handled by a scalar tensor inverse built from a fit of `J`.

So the active route is:

- outer scalar core Schur,
- scalar tensor bulk inverse.

### `k = 1`

`k = 1` uses a surgery-first extracted ordering.

- The extracted `theta` and `zeta` surgery rows form the outer Schur block.
- The bulk is split into `r`, `theta_bulk`, and `zeta_bulk` tensor blocks.
- The tensor route can optionally treat the bulk by an additional coupled inner
  Schur, but that coupling is not required for the outer surgery model.

So the active route is:

- outer surgery Schur,
- tensor bulk blocks for `r`, `theta_bulk`, and `zeta_bulk`,
- optional inner bulk Schur coupling.

The assembly-time toggle is:

- `cp_kwargs["k1_inner_schur"] = True` for the coupled bulk model,
- `cp_kwargs["k1_inner_schur"] = False` for pure diagonal tensor bulk blocks.

### `k = 2`

`k = 2` has the same overall philosophy with a smaller surgery block.

- The extracted `r` surgery rows form the outer Schur block.
- The bulk is split into `r_bulk`, `theta`, and `zeta` tensor blocks.
- The tensor route can optionally treat the bulk by an additional coupled inner
  Schur, but the outer surgery split remains the dominant structure.

So the active route is:

- outer surgery Schur,
- tensor bulk blocks for `r_bulk`, `theta`, and `zeta`,
- optional inner bulk Schur coupling.

The assembly-time toggle is:

- `cp_kwargs["k2_inner_schur"] = True` for the coupled bulk model,
- `cp_kwargs["k2_inner_schur"] = False` for pure diagonal tensor bulk blocks.

### `k = 3`

`k = 3` is the second scalar case.

- There is no surgery split.
- The extracted matrix is treated as one scalar tensor block.
- The inverse apply uses the tensor model built from a fit of `1 / J`.

So the active route is:

- direct scalar tensor inverse,
- no surgery Schur.

## 3. Baselines And Practical Winners

The useful baselines remain:

- whole-matrix Jacobi,
- whole-matrix Chebyshev built on Jacobi.

Those are still useful for comparison, but they are not the preferred
production routes.

The current benchmark picture on the rotating-ellipse family is:

- `k = 0` mass: scalar Schur plus tensor bulk is decisively better than whole
  Jacobi and Jacobi-Chebyshev,
- `k = 3` mass: direct scalar tensor inversion is decisively better than whole
  Jacobi and Jacobi-Chebyshev,
- `k = 1` and `k = 2` mass: the outer surgery Schur plus diagonal tensor bulk
  blocks already delivers most of the gain,
- the optional inner bulk Schur for `k = 1` and `k = 2` reduces iteration
  counts only slightly on the tested family, but increases runtime
  substantially,
- wrapping Chebyshev around an already strong tensor route often lowers
  iteration counts but usually does not improve wall-clock time.

So the current practical recommendation is:

- `k = 0`: use the scalar Schur-plus-tensor route,
- `k = 1`: prefer `k1_inner_schur = False` unless a harder case shows a clear
  robustness benefit from the coupled bulk model,
- `k = 2`: prefer `k2_inner_schur = False` unless a harder case shows a clear
  robustness benefit from the coupled bulk model,
- `k = 3`: use the direct scalar tensor route.

## 4. Forward-Model Diagnostics

The recent small-case forward-model checks help separate model quality from
solve-path effects.

- `k = 0` mass is a good rank-1 tensor model on the tested mapped case:
  about `1.6%` full extracted Frobenius error and about `4.7%` bulk-only.
- `k = 1` mass is a weak rank-1 tensor model on the same case: about `24%`
  Frobenius error both on the full extracted operator and on the bulk-only
  restriction. So this is a bulk-model issue, not a surgery artifact.
- `k = 2` mass is moderate at rank `1`: about `5.3%` Frobenius error, again
  with bulk-only error at essentially the same level.
- `k = 3` mass is also moderate at rank `1`, with about `5.5%` Frobenius
  error.

So the current rank-1 model-quality ordering is:

- good: `k = 0`,
- moderate: `k = 2`, `k = 3`,
- bad: `k = 1`.

Those rank-1 diagnostics turned out to be directionally correct but too
conservative about useful production ranks. The later higher-rank checks gave a
cleaner picture:

- `k = 0` mass is effectively a rank-2 geometry on the tested family. Forward
  error drops to near machine precision at rank `2`, and the solve count drops
  from about `11` iterations to about `3`.
- `k = 1` mass improves strongly from rank `1` to rank `2`, with a smaller
  further gain at rank `3`. On `ns = (8, 16, 8)`, the solve count dropped from
  about `28` to `14` to `13`.
- `k = 2` mass shows the same pattern, with the main gain at rank `2` and a
  smaller extra gain at rank `3`. On the same case, the solve count dropped
  from about `26` to `14` to about `12.5`.
- `k = 3` mass also benefits strongly from rank `2`, but shows no practical
  gain from rank `3`. On the same case, the solve count dropped from about
  `11` to `6`, then stayed there.

So the practical higher-rank conclusion is:

- rank `2` is a good default for all mass blocks,
- rank `3` is only a plausible extra option for `k = 1` or `k = 2`,
- and rank `2` already captures essentially all of the useful gain for `k = 0`
  and `k = 3`.

The larger mixed solve benchmark at `ns = (16, 32, 8)`, `p = 3` keeps that
overall recommendation but adds one useful refinement.

- `k = 0` mass still clearly prefers rank `2`: the solve count dropped from
  about `11.4` to `4`, with no further gain at rank `4`.
- `k = 1` mass still strongly prefers `inner_schur = off` in wall-clock time,
  but on this larger case rank `4` with `inner_schur = off` was the fastest of
  the tested tensor variants, at about `23.5` iterations / `153.2 ms` versus
  about `32.8` / `164.2 ms` at rank `1`.
- `k = 2` mass shows the same qualitative pattern: `inner_schur = off` remains
  the timing winner, and rank `4` with `inner_schur = off` was marginally the
  fastest tested tensor option at about `23` iterations / `147.2 ms`, versus
  about `29.6` / `147.6 ms` at rank `1`.
- `k = 3` mass is already saturated by rank `2`: the solve count dropped from
  about `11` to `6`, with rank `4` essentially tied in time.

So the updated practical reading is:

- keep rank `2` as the production default for all mass blocks,
- keep `inner_schur = off` as the production default for `k = 1` and `k = 2`,
- but treat rank `4` as a legitimate exposed tuning option for larger `k = 1`
  and `k = 2` cases where setup cost is acceptable and the extra iteration
  reduction matters.

One more free-boundary sweep at `ns = (16, 32, 16)`, `p = 3` is worth noting
because it sharpens that picture more convincingly.

- `k = 0` mass did **not** improve from rank `1` to rank `2` on this larger
  free case, but then improved strongly again at rank `3` and `4`: about
  `12.2` iterations / `8.52 ms` at rank `1`, about `13.0` / `9.63 ms` at rank
  `2`, and about `4.0` / `3.72 ms` at rank `3` and `4`.
- `k = 1` mass improved steadily across the ranks, with the main practical win
  now clearly at rank `3`: about `35.8` iterations / `348.5 ms` at rank `1`,
  about `30.0` / `297.6 ms` at rank `2`, and about `26.0` / `260.1 ms` at
  rank `3` and `4`.
- `k = 2` mass showed the same pattern: about `34.8` iterations / `332.2 ms`
  at rank `1`, about `29.0` / `282.4 ms` at rank `2`, and about `25.2` to
  `25.0` / `247.8` to `244.5 ms` at rank `3` and `4`.
- `k = 3` mass also kept improving beyond rank `2`, though more mildly: about
  `15` iterations / `4.20 ms` at rank `1`, about `7` / `2.35 ms` at rank `2`,
  and about `6` / `2.07 ms` at rank `3` and `4`.

So the current default should be read as a policy choice, not as the pointwise
best rank on every tested case:

- rank `3` is now the chosen eager default for all mass blocks,
- but free-boundary `k = 0` and `k = 3` now have concrete larger tests where
  rank `3` materially helps,
- and free-boundary `k = 1` / `k = 2` on the larger tested case now also look
  better at rank `3` than at rank `2`, with rank `4` giving little extra.

That said, the current evidence now favors the rank-3 policy directly: taken
at face value, this larger free-boundary sweep makes rank `3` the stronger
all-around candidate, while rank `4` adds little extra.

The current production default follows that recommendation in the eager
operator-assembly path: the mass blocks are assembled with per-degree tensor
ranks `k0 = k1 = k2 = k3 = 3`, while the scalar stiffness/Hodge fallback rank
remains at `1`.

## 5. Analytic Priors And Inversion Strategy

The recent toroidal-prior experiments are useful because they separate two
different questions:

1. how much analytic geometry should be built into the coefficient model,
2. how that modeled operator should then be inverted.

For the mass-side coefficient fields the natural leading toroidal factors are
the major-radius expressions

- `r R`,
- `R / r`,
- `r / R`,
- `1 / (r R)`,

with `R` shorthand for the toroidal major-radius factor. In the current prior
implementation those factors are used on the fit side, not as a separate exact
inverse formula. Concretely, if `W` is the coefficient field and `P` is the
known prior, the code fits the residual `C ≈ W / P` and then reconstructs the
modeled field as

$$
W_{\mathrm{model}} = P \cdot C_{\mathrm{fit}}.
$$

So the prior and the learned factors combine multiplicatively at the
coefficient-field level. Because both `P` and `C_fit` are represented as sums
of separable terms, the final assembled tensor block is still an additive sum
of separable Kronecker products after expansion.

That is different from an additive analytic expansion in `eps`. In an
`eps`-expansion route one writes the operator itself as

$$
A = A_0 + \varepsilon A_1 + \varepsilon^2 A_2 + \cdots,
$$

keeps a very simple analytic backbone `A_0`, and then uses that backbone as a
cheap inverse or smoother while the higher-order terms are treated as
corrections.

These two inverse strategies have different strengths.

Shared modal basis:

- Pros:
  - generic multirank inverse for arbitrary learned terms,
  - one uniform implementation for all tensor blocks,
  - currently the strongest fully automatic inverse when the coefficient model
    is not already extremely simple.
- Cons:
  - once the modeled field contains more than one separable term, the direct
    rank-1 inverse is lost and the block falls back to a dense shared-basis
    modal solve,
  - this does not exploit the original 1-D banded operator structure,
  - performance can become sensitive to the quality of the learned coefficient
    model.

Analytic expansion plus Richardson:

- Pros:
  - keeps the analytic rank-1 backbone explicit,
  - can exploit axis-by-axis 1-D solves against the underlying banded factors
    instead of building a dense modal basis,
  - Richardson correction uses the true residual of the full operator rather
    than relying on a truncated inverse series.
- Cons:
  - requires a problem-specific analytic expansion,
  - is less generic than the shared modal basis,
  - and if the analytic backbone is too weak, Richardson may need several
    correction steps before it is competitive.

This is why the current implementation uses the prior only to simplify the fit,
not yet to change the inverse. That is the safer step. The open design question
is whether the scalar and low-rank mass routes should eventually switch from
"prior plus shared modal basis" to an explicit analytic backbone plus a few
Richardson correction steps.

The current preferred next experiment is now more specific than a generic
additive `eps` expansion. Rather than giving every analytic branch the same
status, the better hybrid is:

- keep one leading analytic backbone term,
- require its learned coefficient to be rank `1`,
- and spend the remaining rank budget on a correction channel.

So instead of the current multiplicative form

$$
W_{\mathrm{model}} = P \cdot C_{\mathrm{fit}},
$$

or a fully symmetric additive model with many equally weighted learned
coefficients, the preferred ansatz is

$$
W_{\mathrm{model}} \approx B_0 \odot C_0 + B_1 \odot \widetilde C_1,
$$

where `B_0` is the leading analytic basis term, `C_0` is constrained to rank
`1`, `B_1` is the correction channel, and `\widetilde C_1` carries the rest of
the rank budget, for example rank `m - 1` if the total budget is `m`.

The reason this is attractive is solver-side rather than fit-side. Because
both `B_0` and `C_0` are rank `1`, the backbone term `B_0 \odot C_0` is still a
true rank-`1` tensor block and keeps the simple inverse structure. The
correction term can then be treated as the flexible residual, which is exactly
the setting where Richardson correction is most plausible.

For the toroidal families it is useful to write that analytic backbone
explicitly. Let

$$
R(r,\theta) = 1 + \varepsilon r \cos\theta.
$$

Then the four mass-side coefficient families are

- `k = 0`: `r R`,
- `k = 1` / `k = 2` angular or toroidal blocks: `R / r` or `r / R`,
- `k = 3`: `1 / (r R)`.

The `R`-type families are exact finite expansions:

$$
rR = r + \varepsilon r^2 \cos\theta,
$$

$$
\frac{R}{r} = \frac{1}{r} + \varepsilon \cos\theta.
$$

So those are exact two-term additive backbones. By contrast the inverse-type
families require a truncated series:

$$
\frac{1}{R} = 1 - \varepsilon r \cos\theta + \varepsilon^2 r^2 \cos^2\theta - \cdots,
$$

which gives

$$
\frac{r}{R} = r - \varepsilon r^2 \cos\theta + \varepsilon^2 r^3 \cos^2\theta - \cdots,
$$

$$
\frac{1}{rR} = \frac{1}{r} - \varepsilon \cos\theta + \varepsilon^2 r \cos^2\theta - \cdots.
$$

So the power of `r` grows with the order kept in the `eps` expansion, not with
the CP rank. This is one of the reasons the additive analytic route is often
easier to reason about than the multiplicative prior route.

For that hybrid rank-allocation plan, the natural leading backbones are:

- `k = 0`: use `B_0 = r` and put the toroidal correction into the `r^2 cos(theta)`
  channel,
- `k = 1` and `k = 2` for the `R / r` families: use `B_0 = 1 / r` and put the
  toroidal correction into the `cos(theta)` channel,
- `k = 1` and `k = 2` for the `r / R` families: use `B_0 = r` and treat the
  first inverse correction `- r^2 cos(theta)` as the residual channel,
- `k = 3`: use `B_0 = 1 / r` and treat `- cos(theta)` as the first correction
  channel.

That keeps the easiest analytically invertible part in the backbone and uses
the remaining rank budget only where the toroidal correction is actually
needed.

The practical consequence is that the most attractive Richardson backbones are:

- `k = 0`: the exact two-term `r + eps r^2 cos(theta)` backbone,
- `k = 3`: the zeroth- or first-order truncated `1/rR` backbone,
- `k = 1` and `k = 2`: the exact `R/r` backbone first, and then the first-order
  `r/R` truncation only if needed.

Those choices keep the backbone as simple as possible while still encoding the
dominant toroidal structure. They are therefore the best candidates for an
analytic expansion plus Richardson experiment.

For the first implementation, the fit itself should also stay simple. The
recommended procedure is a greedy two-stage residual fit rather than a fully
coupled additive optimization:

$$
R^{(0)} = W,
$$

fit rank-`1` `C_0` from

$$
B_0 \odot C_0 \approx R^{(0)},
$$

then update

$$
R^{(1)} = R^{(0)} - B_0 \odot C_0,
$$

and fit the remaining rank budget in the correction channel,

$$
B_1 \odot \widetilde C_1 \approx R^{(1)}.
$$

No extra normalization or polishing sweep is needed for the first pass. The
important structural point is simply that the leading term remains rank `1`
and easy to invert, while the correction term absorbs the remaining error.

The first forward-model checks refined that plan in an important way.

For `mass-k0` on the small rotating-ellipse case `ns = (4, 8, 4)`, `p = 3`, a
split rank-`2` model already behaved as intended structurally: the backbone
term carried about `99.3%` of the tensor norm, the correction-to-backbone norm
ratio was about `0.12`, and the residual left after the backbone alone was also
about `0.12`. So for `k = 0` the split really is in a "dominant backbone plus
small correction" regime.

But that same check also showed why the split and multiplicative paths cannot
be compared only by the printed rank parameter. The current multiplicative path
expands each learned residual mode through the multi-term toroidal prior, so a
printed rank-`m` fit can assemble more than `m` final separable terms. The
split path does not: split rank `m` literally means one rank-`1` backbone term
plus a free rank-`m - 1` residual.

The more important result came from `mass-k1`.

- At split rank `2`, the model was too rigid. The `arr` and `theta` blocks
  still had about `36%` to `38%` residual after the backbone alone, and the
  full forward error was much worse than the multiplicative baseline.
- At split rank `3`, the same `k = 1` path became competitive and in fact beat
  the multiplicative rank-`2` forward model on that test. The split-rank-`3`
  run gave about `6.23e-3` Frobenius error versus about `8.09e-3` for the
  multiplicative rank-`2` model.

The diagnostic picture explains why. On that same `k = 1`, split-rank-`3`
case:

- the `arr` and `theta` blocks were not in a "tiny correction" regime,
  with correction-to-backbone norm ratios around `0.39` to `0.40`,
- the `zeta` block was much closer to that regime, around `0.11`,
- and the backbone-only residuals were about `0.36` to `0.38` for `arr` and
  `theta`, versus about `0.11` for `zeta`.

So the current reading is:

- `k = 0` really does look like "easy backbone plus small correction",
- `k = 1` does not at split rank `2`,
- but `k = 1` becomes promising once the correction channel is allowed more
  than one rank-`1` term,
- and the backbone is still dominant enough that a backbone solve plus a few
  residual-correction steps is a plausible inverse strategy.

This is therefore the practical next-step policy for the split model:

- keep the rank-`1` backbone requirement,
- let the residual channel use the remaining rank budget freely,
- and evaluate inversion strategies on top of that split rather than forcing
  the correction itself to remain rank `1`.

## 6. Geometry Sensitivity: `eps = 1/3` vs `eps = 1/7`

The latest rotating-ellipse sweep compared the same benchmark at
`ns = (16, 16, 16)`, `p = 3`, ranks `1, 2, 4`, and inner Schur on/off for
`k = 1, 2`, with `eps = 1/3` and `eps = 1/7`.

Lower `eps` makes every mass block easier. The strongest and most consistent
effect is on model quality: the tensor coefficient fits become substantially
more low rank. The solve-time improvement is more modest once a block is
already near its practical floor.

Representative practical rows are:

| k | rank | inner | `cp_err` change | iteration change | time change |
| --- | ---: | :---: | ---: | ---: | ---: |
| 0 | 2 | n/a | `-57.7%` | `0.0%` | `-2.2%` |
| 1 | 2 | off | `-55.4%` | `-10.7%` | `-10.0%` |
| 2 | 2 | off | `-57.3%` | `-9.4%` | `-8.9%` |
| 3 | 2 | n/a | `-84.0%` | `-16.7%` | `-15.4%` |

So the practical reading is:

- `k = 0` and `k = 3` were already close to saturation at rank `2`, so lower
  `eps` mainly improves fit quality rather than runtime.
- `k = 1` and `k = 2` remain the most geometry-sensitive blocks in solve time,
  but the gain is still moderate: about `9%` to `10%` faster in the practical
  rank-`2`, inner-Schur-off regime.
- The inner-Schur conclusion does not change with `eps`: for `k = 1` and
  `k = 2`, turning the coupled inner Schur off is still much faster in
  wall-clock time on the tested family.

## 7. Final Summary

The final mass-preconditioner picture is simple.

- keep the extracted-space special rows exact through a small Schur solve,
- compress the diagonal mapped coefficient fields rather than the inverse,
- use tensor block inverses only on the regular bulk blocks,
- and keep the optional inner coupled bulk Schur for `k = 1` and `k = 2` as a
  benchmarked option rather than as the default practical choice,
- while treating rank `2` as the practical default tensor rank across
  `k = 0, 1, 2, 3` on the tested geometry family,
- with `k = 2` rank `3` left as an exposed tuning option rather than the
  default because the measured extra solve gain has not yet been weighed
  against additional setup cost.
