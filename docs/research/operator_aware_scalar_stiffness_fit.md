# Operator-Aware Scalar Stiffness Fit

This note records the next design step for the scalar `k = 0` tensor-Hodge
bulk model. The current scalar stiffness route is algebraically correct, but
its multirank fit targets the wrong object. This note explains what needs to
change and why that is different from the `k = 1` mass case, even though both
use the same diagonal mapped metric fields

- `alpha_rr = J g^{rr}`,
- `alpha_thetatheta = J g^{theta theta}`,
- `alpha_zetazeta = J g^{zeta zeta}`.

## 1. Current Scalar Stiffness Structure

On the bulk scalar space, the mapped `k = 0` stiffness operator is a sum of
three directional Kronecker terms,

$$
A_0
=
K_r(\alpha_{rr}) \otimes M_t \otimes M_z
+
M_r \otimes K_t(\alpha_{\theta\theta}) \otimes M_z
+
M_r \otimes M_t \otimes K_z(\alpha_{\zeta\zeta}).
$$

Here `K_r`, `K_t`, and `K_z` are weighted 1-D stiffness matrices, while `M_r`,
`M_t`, and `M_z` are the corresponding 1-D mass matrices.

The important point is that all three terms act on the same scalar degrees of
freedom. Every scalar basis function participates in all three directional
pieces through its derivatives. The operator is therefore not "one geometric
field per block". It is one scalar block built from a sum of directional
contributions.

## 2. Why The Current Multirank Builder Misses

The present rank-`r` scalar builder first fits a shared surrogate field and
then recovers directional active factors from that proxy. That approach is too
indirect for stiffness.

Improving the low-rank fit of a shared proxy field does not guarantee a better
approximation of

$$
K_r \otimes M_t \otimes M_z
+
M_r \otimes K_t \otimes M_z
+
M_r \otimes M_t \otimes K_z.
$$

This is exactly what the forward-model diagnostics showed:

- the scalar stiffness apply matches the inverse of its assembled tensor model,
- the rank-1 bulk model itself is poor,
- and increasing the rank inside the current proxy-field construction does not
  materially improve the assembled bulk operator.

So the next fix is not a better inverse apply and not a higher rank inside the
same fit. The fix is to change the fit target.

## 3. Why This Differs From `k = 1` Mass

The `k = 1` mass problem uses the same diagonal mapped fields
`J g^{rr}`, `J g^{theta theta}`, and `J g^{zeta zeta}`, but the operator sees
them in a different way.

For `k = 1` mass, the extracted bulk unknowns split into directional blocks:

- `r`,
- `theta_bulk`,
- `zeta_bulk`.

Each block corresponds to basis functions that point in exactly one coordinate
direction. The `r` basis functions only carry the `r` component, the
`theta_bulk` basis functions only carry the `theta` component, and the
`zeta_bulk` basis functions only carry the `zeta` component.

That means the diagonal metric fields already line up with the operator
decomposition:

- the `r` mass block is built from `alpha_rr`,
- the `theta` mass block is built from `alpha_thetatheta`,
- the `zeta` mass block is built from `alpha_zetazeta`.

So in the `k = 1` mass case, fitting each diagonal field and building a tensor
block from it is already operator-aware enough. The fitted object and the
assembled block structure match.

That is not true for scalar stiffness. There is only one scalar unknown field,
not three directional unknown blocks. The three metric fields appear inside a
sum of directional derivative energies on the same scalar space.

This is the core distinction:

- `k = 1` mass: one directional field per directional block,
- scalar `k = 0` stiffness: three directional fields summed on one scalar
  block.

So even though the geometric tensors are the same, the correct low-rank target
is not the same.

## 4. Operator-Aware Bulk Ansatz

The replacement multirank model should approximate the bulk operator directly,
not a proxy field. A natural rank-`R` ansatz is

$$
A_0^{(R)}
=
\sum_{q=1}^R \Big[
K_r(a_r^{(q)}) \otimes M_t(b_t^{(q)}) \otimes M_z(c_z^{(q)})
+
M_r(\tilde a_r^{(q)}) \otimes K_t(\tilde b_t^{(q)}) \otimes M_z(\tilde c_z^{(q)})
+
M_r(\hat a_r^{(q)}) \otimes M_t(\hat b_t^{(q)}) \otimes K_z(\hat c_z^{(q)})
\Big].
$$

There are two reasonable ways to parameterize this.

### Option A: Independent Directional CP Fits

Fit each coefficient tensor separately:

- `alpha_rr`,
- `alpha_thetatheta`,
- `alpha_zetazeta`.

Then build the three directional operator sums directly from those separate
fits. This is the simplest operator-aware correction.

Pros:

- easy to implement,
- directly targets the directional operator terms,
- no fake shared field.

Cons:

- no shared per-axis basis across the three directions,
- inverse application of the final sum still needs an approximate shared-basis
  treatment.

### Option B: Joint Operator-Aware Shared Basis

Fit the three directional coefficient tensors jointly, but only through a
parameterization that preserves the directional operator structure. For each
rank component, keep separate directional active factors but choose them to be
compatible with one shared modal basis per axis.

This is closer to the current shared-diagonalization inverse, but it must be
driven by the directional operator terms themselves rather than by one
surrogate field.

Pros:

- keeps the final inverse route close to the current shared-basis machinery,
- more likely to preserve efficient modal inversion.

Cons:

- harder fit problem,
- requires a real operator-side objective rather than coefficient-side ALS.

## 5. Recommended First Step

The first implementation step should be Option A, not Option B.

That means:

1. fit `alpha_rr`, `alpha_thetatheta`, and `alpha_zetazeta` separately,
2. assemble the multirank bulk operator directly as a sum of fitted
   directional Kronecker terms,
3. keep the current scalar surgery Schur wrapper unchanged,
4. reuse the existing shared-basis inverse machinery only after the operator
   terms have been assembled from the directional fits.

This is the smallest change that makes the fit target operator-aware.

## 6. Validation Target

Validation should stay on the operator, not on the fitted fields.

The relevant checks are:

1. bulk forward-model error against the exact scalar stiffness bulk apply,
2. extracted forward-model error after the surgery sandwich,
3. solve iterations and wall time against the current scalar tensor-Hodge
   preconditioner,
4. comparison with the current rank-`r` proxy-field builder at the same rank.

The note-worthy success criterion is not just a better CP error. It is a
material reduction in bulk forward-model error and then a matching improvement
in solve behavior.

On the first small extracted-space check that criterion was met. For the
rotating-ellipse case `ns = (4, 8, 4)`, `p = 3`, rank `2`, the operator-aware
builder gave about `11.7%` extracted Frobenius error, with sampled forward
error around `11.3% +- 2.0%`. That is still not mass-quality, but it is a
clear improvement over the earlier scalar stiffness baseline of about `33%`
extracted Frobenius error.

The follow-up bulk-only checks clarified where the remaining miss lives.

- rank `1`: full extracted Frobenius error about `18.5%`, bulk-only about
  `25.3%`,
- rank `2`: full extracted Frobenius error about `11.7%`, bulk-only about
  `16.0%`,
- rank `4`: full extracted Frobenius error about `10.7%`, bulk-only about
  `14.6%`.

So the exact surgery rows and extracted coupling are not the main remaining
defect. They actually soften the final extracted-space error relative to the
bulk block by itself. The dominant remaining issue is still the bulk tensor
model.

These data also show the new and old constructions differ in exactly the way we
wanted:

- rank now helps monotonically at the operator level,
- the big gain is from rank `1` to rank `2`,
- and rank `4` gives only a smaller extra gain.

So the operator-aware redesign fixed the main pathology, but the remaining bulk
error still has diminishing returns with rank. That makes the next likely limit
the bulk ansatz itself rather than the Schur wrapper.

The first solve benchmark adds an important qualifier. On the Dirichlet scalar
Laplace solve at `ns = (6, 8, 4)`, `p = 3`, the new builder did not yet turn
that forward-model improvement into a better preconditioner as the rank
increased:

| rank | avg iters | avg ms |
| ---: | ---: | ---: |
| 1 | `25.0` | `6.68` |
| 2 | `30.5` | `7.92` |
| 4 | `29.5` | `7.64` |

So the current intermediate conclusion is:

- the operator-aware redesign is genuinely better as an operator model,
- but higher rank is still not improving the actual scalar-Laplace solve,
- and the remaining problem is therefore no longer just the fit target.

The larger Dirichlet rotating-ellipse benchmark at `ns = (16, 32, 8)`,
`p = 3` makes that point much more strongly. There the scalar tensor-Hodge
solve behaved as follows:

| rank | `cp_err` | avg iters | avg ms |
| ---: | ---: | ---: | ---: |
| 1 | `3.65e-1` | `58.8` | `287.84` |
| 2 | `7.59e-2` | `107.2` | `518.55` |
| 4 | `1.08e-2` | `1000.0` | `4758.14` |

So higher rank keeps improving the directional coefficient fit while making
the actual scalar inverse much worse, eventually unusable. That large-case
result is strong evidence that the active defect is now the higher-rank
inverse construction itself, specifically the shared-basis/modal-denominator
approximation used to invert the operator-aware directional sum.

This also clarifies why the comparison with `k = 1` mass matters. On the same
large case, the `k = 1` mass block sees essentially the same rank-by-rank CP
fit errors, but its solve behavior improves with rank instead of collapsing.
So the low-rank geometric fit is not the root problem anymore. The remaining
instability is specific to the scalar stiffness inverse.

The likely interpretation is that a better forward approximation of the bulk
operator is not yet translating into a better inverse approximation under the
current whitening/shared-basis inverse ansatz. That keeps rank `1` as the
practical scalar-Laplace default for now, even though the higher-rank
operator-aware builder is clearly the better modeling direction.

## 7. Practical Conclusion

The scalar stiffness preconditioner does not primarily need more rank. It
needs a fit whose algebra matches the actual operator.

The mass-side tensor route succeeded because its block structure already lines
up with the directional coefficient fields. The scalar stiffness route does not
have that property, because one scalar basis participates in all three
directional derivative energies at once.

So the next stiffness-side improvement should be described as:

- not "higher-rank scalar stiffness",
- but "operator-aware multirank scalar stiffness".

At the current intermediate stage, that phrase should be read carefully:

- operator-aware fitting is now the right construction principle,
- but the corresponding inverse model still needs more work before higher rank
  becomes a practical solver improvement.

## 8. Prior Combination And Inversion Choices

The recent toroidal-prior work clarifies two distinct pieces of the scalar
stiffness design.

First, how the analytic prior is combined with the learned low-rank model.
The current implementation uses the prior multiplicatively at the coefficient
level. If `P` denotes the known geometric prior and `C_fit` the learned
residual, then the modeled directional coefficient is built as

$$
W_{\mathrm{model}} = P \cdot C_{\mathrm{fit}}.
$$

This does **not** mean that the assembled operator is a multiplicative product
of two operators. Both `P` and `C_fit` are represented as sums of separable
terms, so after expansion the final directional coefficient is still an
additive sum of separable terms, and the operator is assembled from that sum in
the usual way.

Second, how the resulting operator should be inverted. There are two natural
choices.

Shared modal basis:

- treat the full modeled directional sum as a multirank tensor operator,
- diagonalize approximately in one shared basis per axis,
- and invert by dividing by the resulting modal denominator.

This is generic and currently the active scalar multirank inverse path, but it
also means that once the model contains more than one separable term, the
special cheap rank-1 inverse is gone.

Analytic expansion plus Richardson:

- keep a simple analytic backbone such as the leading toroidal term or the
  `eps = 0` block,
- apply or solve that backbone axis-by-axis,
- and then use a few Richardson steps with the full operator apply to correct
  the residual.

The attractive feature of that second route is that it can exploit the
structured 1-D solves of the analytic rank-1 backbone instead of building a
dense shared modal basis. That is exactly why it is interesting for the scalar
and other low-rank-preference cases.

The tradeoff is that Richardson needs a good backbone and a sensible damping
parameter, while the shared modal basis is more generic and less tied to one
specific geometric expansion. So the real design choice is not just
"learned factors or analytic factors". It is

- generic multirank inverse via shared modal basis,
- versus analytic low-rank backbone plus residual correction.

The current code takes the conservative route: use the analytic prior to make
the fit easier, but keep the generic shared-modal inverse. The open question is
whether the scalar route should move further and use that same analytic
information to build a better low-rank backbone for Richardson correction.

For comparison with the mass-side discussion, note that the additive expansion
viewpoint is also cleaner here than the multiplicative one. The multiplicative
prior says

$$
W_{\mathrm{model}} = P \cdot C_{\mathrm{fit}},
$$

so one learned residual multiplies every analytic prior term. In an additive
expansion one would instead write the modeled coefficient or operator as a sum
of explicit toroidal terms, for example a backbone plus first-order and
second-order `eps` corrections. That keeps the analytic pieces visible and is
the more natural starting point for a Richardson scheme built around a simple
low-rank backbone.

The preferred next scalar experiment is therefore not a fully symmetric
multi-branch additive fit. It is the same hybrid split as on the mass side:

- choose one leading analytic backbone term `B_0`,
- constrain its learned coefficient `C_0` to rank `1`,
- and put the rest of the model into one correction channel
  `B_1 \odot \widetilde C_1` with the remaining rank budget.

So the intended model is

$$
W_{\mathrm{model}} \approx B_0 \odot C_0 + B_1 \odot \widetilde C_1,
$$

with `C_0` rank `1` and `\widetilde C_1` carrying the remaining rank.

That bias is deliberate. The backbone term is the part we would like to keep
cheap to invert axis-by-axis. The correction term is allowed to be more
flexible because it is the part we would expect to handle with Richardson or a
small number of residual corrections rather than a direct analytic inverse.

For the first implementation, the fit should be greedy rather than globally
coupled:

$$
R^{(0)} = W,
$$

fit rank-`1` `C_0` from `B_0 \odot C_0 \approx R^{(0)}`,

$$
R^{(1)} = R^{(0)} - B_0 \odot C_0,
$$

and then fit `\widetilde C_1` with the remaining rank budget from
`B_1 \odot \widetilde C_1 \approx R^{(1)}`.

That is the cleanest way to preserve a true rank-`1` leading block while still
giving the model enough flexibility to capture the toroidal correction.