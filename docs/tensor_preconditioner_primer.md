# Tensor Preconditioner Primer

This note explains what the production `tensor` preconditioner in `mrx` is
actually approximating, why it works well for mass-like operators, and why the
same idea does not extend directly to the Schur operator in the mixed
Hodge-Laplacian solves.

It should be read together with:

- [docs/iterative_solver_primer.md](iterative_solver_primer.md) for the full
  solver routing,
- [docs/mass_preconditioners.md](mass_preconditioners.md) for the
  degree-by-degree extracted block structures.

## 1. Pure Tensor Scalar Case

Start with the simplest possible situation: scalar `k = 0`, no geometry, no
extraction surgery.

Let the scalar basis be a tensor product,

$$
\phi_{abc}(r,\theta,\zeta)
= \phi_a^{(r)}(r)\, \phi_b^{(\theta)}(\theta)\, \phi_c^{(\zeta)}(\zeta).
$$

Without geometry, the scalar mass bilinear form is just the reference-domain
$L^2$ product,

$$
m_0(u,v) = \int u v\,dr\,d\theta\,d\zeta,
$$

so the matrix entries factor exactly,

$$
(M_0)_{abc,a'b'c'}
= \left(\int \phi_a^{(r)}\phi_{a'}^{(r)}\,dr\right)
  \left(\int \phi_b^{(\theta)}\phi_{b'}^{(\theta)}\,d\theta\right)
  \left(\int \phi_c^{(\zeta)}\phi_{c'}^{(\zeta)}\,d\zeta\right).
$$

Equivalently,

$$
M_0 = M_r \otimes M_\theta \otimes M_\zeta.
$$

Then the inverse is equally simple,

$$
M_0^{-1} = M_r^{-1} \otimes M_\theta^{-1} \otimes M_\zeta^{-1},
$$

and applying the inverse costs only three 1-D inverse applications.

The tensor-product assembly code is exactly exploiting this kind of structure.
In quadrature form, each 1-D mass matrix is

$$
M_r = B_r\,\operatorname{diag}(w_r)\,B_r^T,
\qquad
M_\theta = B_\theta\,\operatorname{diag}(w_\theta)\,B_\theta^T,
\qquad
M_\zeta = B_\zeta\,\operatorname{diag}(w_\zeta)\,B_\zeta^T,
$$

with basis-evaluation matrices $B_r, B_\theta, B_\zeta$ and 1-D quadrature
weights $w_r, w_\theta, w_\zeta$.

This is the good situation. The whole tensor program tries to recover as much of
this structure as possible, even after extraction and geometry have complicated
the real operator.

## 2. When Surgery Appears

Even before geometry enters, the extracted polar spaces can destroy the pure
tensor picture near the axis or in the fused boundary rows.

The key point is that the production operator is not built on the raw tensor
space alone. It is first pushed through an extraction operator,

$$
M_k^{\mathrm{ext}} = E_k M_k E_k^T.
$$

If $E_k$ were itself one clean tensor product on the full space, the tensor
structure would survive globally. But in the polar extracted spaces, $E_k$
contains a small number of special rows that fuse or constrain degrees of
freedom near the axis and boundaries. Those rows are not a uniform tensor
product acting on the whole coefficient grid.

That is where the surgery comes from: not from the geometry, but from the
extracted-space identifications.

That is why the tensor route does not try to approximate the whole extracted
matrix by one tensor inverse. Instead it splits the extracted matrix into:

- a small dense surgery/core block,
- one or more bulk blocks that still look tensor-like.

The dense part is handled exactly by a Schur complement. The tensor machinery is
used only on the bulk.

In the scalar `k = 0` case this gives the core-plus-bulk split

$$
M_0^{\mathrm{ext}} =
\begin{pmatrix}
A_{cc} & A_{cb} \\
A_{bc} & A_{bb}
\end{pmatrix},
$$

where the code chooses the scalar core size

$$
n_c = 3 n_\zeta,
$$

and treats the remaining rows as the tensor bulk. In other words, the first
few extracted rows are considered special polar rows, while the rest of the
matrix is still organized like a radial bulk tensor window.

where the preconditioner uses:

- a tensor inverse for the bulk approximation to $A_{bb}$,
- a dense inverse of the Schur complement
  $A_{cc} - A_{cb} A_{bb}^{-1} A_{bc}$.

Operationally, if $B_{bb}^{-1}$ is the tensor bulk inverse model, the solve is

$$
S_c = A_{cc} - A_{cb} B_{bb}^{-1} A_{bc},
$$

$$
x_c = S_c^{-1}\left(r_c - A_{cb} B_{bb}^{-1} r_b\right),
\qquad
x_b = B_{bb}^{-1}\left(r_b - A_{bc} x_c\right).
$$

So the non-tensor part is kept small and exact, while the tensor approximation
is used only where the extracted matrix still has a regular bulk layout.

So the first correction to the pure tensor picture is: keep the non-tensor part
small and dense, and keep the tensor approximation only on the bulk.

## 3. How Geometry Breaks Bulk Tensor Structure

Once geometry is turned on, the mass integrand is no longer constant. For the
scalar mass it is weighted by $J$, and for the vectorial cases by diagonal
entries of $J g^{-1}$ or $g/J$.

For scalar `k = 0`, the mapped bilinear form is

$$
m_0(u,v) = \int_{\widehat\Omega} u(\xi) v(\xi) J(\xi)\,d\xi.
$$

On the tensor quadrature grid this becomes

$$
(M_0)_{abc,a'b'c'}
\approx
\sum_{i,j,k}
w_i^{(\theta)} w_j^{(r)} w_k^{(\zeta)}
J_{ijk}
\phi_b^{(\theta)}(\theta_i) \phi_{b'}^{(\theta)}(\theta_i)
\phi_a^{(r)}(r_j) \phi_{a'}^{(r)}(r_j)
\phi_c^{(\zeta)}(\zeta_k) \phi_{c'}^{(\zeta)}(\zeta_k).
$$

This is the same tensor quadrature that appears in the assembly code:

$$
M_0
= \sum_{i,j,k}
J_{ijk}
\bigl(B_r(:,j) B_r(:,j)^T\bigr)
\otimes
\bigl(B_\theta(:,i) B_\theta(:,i)^T\bigr)
\otimes
\bigl(B_\zeta(:,k) B_\zeta(:,k)^T\bigr),
$$

with the quadrature weights absorbed into the 1-D weighted mass factors.

That means the bulk operator is no longer a single Kronecker product. Instead it
is a sum of many weighted tensor-product contributions.

The exact tensor-product case survives only if the mapped coefficient itself is
separable. For example, if

$$
J_{ijk} = \alpha_i \beta_j \gamma_k,
$$

then

$$
M_0
= M_\theta[\alpha] \otimes M_r[\beta] \otimes M_\zeta[\gamma],
$$

where

$$
M_r[\beta] = B_r\,\operatorname{diag}(w_r \odot \beta)\,B_r^T,
$$

and similarly for the other two axes.

But for a general mapped geometry the sampled tensor $J_{ijk}$ is not of that
form. The Jacobian is evaluated pointwise from the map derivative $DF$, and the
metric entries come from nonlinear expressions such as

$$
J = \det DF,
\qquad
g = DF^T DF,
\qquad
g^{-1} = (DF^T DF)^{-1}.
$$

Even if the map itself has simple structure, these derived fields are usually
not exactly rank-1 on the quadrature grid. That is the precise sense in which
geometry breaks the tensor structure.

So geometry breaks the exact tensor form even in the bulk. The difficulty is not
the small extracted surgery rows anymore. The difficulty is that the mapped
coefficient field varies over the 3-D quadrature grid.

This is the central approximation problem in the tensor preconditioner.

## 4. How CP-ALS Restores A Useful Bulk Model

The recovery step is coefficient compression.

The code samples the relevant diagonal coefficient field on the tensor
quadrature grid and fits it by a short CP decomposition,

$$
W(i,j,k) \approx \sum_{m=1}^R \omega_m a_m(i)b_m(j)c_m(k).
$$

This is the `_cp_als_3tensor(...)` step in the implementation.

For the scalar case, $W = J$. For the vectorial cases, $W$ is one of the
diagonal fields

$$
J g^{rr},\quad J g^{\theta\theta},\quad J g^{\zeta\zeta},
\quad \frac{g_{rr}}{J},\quad \frac{g_{\theta\theta}}{J},
\quad \frac{g_{\zeta\zeta}}{J},\quad \frac{1}{J}.
$$

Each CP term generates one separable weighted tensor block. Writing

$$
W^{(m)}_{ijk} = \omega_m a_m(i)b_m(j)c_m(k),
$$

the corresponding 1-D weighted mass factors are

$$
M_\theta^{(m)} = B_\theta\,\operatorname{diag}(w_\theta \odot a_m)\,B_\theta^T,
$$

$$
M_r^{(m)} = B_r\,\operatorname{diag}(w_r \odot (\omega_m b_m))\,B_r^T,
$$

$$
M_\zeta^{(m)} = B_\zeta\,\operatorname{diag}(w_\zeta \odot c_m)\,B_\zeta^T,
$$

with the appropriate radial restriction applied afterward if the extracted bulk
starts away from the axis.

Each rank-1 term produces one tensor-product block model: one weighted 1-D mass
matrix in `r`, one in `theta`, and one in `zeta`. So the bulk model becomes a
short sum of separable tensor blocks instead of one dense 3-D object.

At the block level this means

$$
A_{bb} \approx \widetilde A_{bb}
= \sum_{m=1}^R M_r^{(m)} \otimes M_\theta^{(m)} \otimes M_\zeta^{(m)}.
$$

If the fit has only one term, then we are back in the ideal tensor-product
situation and the preconditioner stores direct 1-D inverses.

If the fit has several terms, the code still avoids a dense 3-D inverse. It
builds a shared modal basis in each axis, stores the resulting modal diagonals,
and applies the inverse by:

- transforming the right-hand side into modal coordinates,
- dividing by the summed separable denominator,
- transforming back.

More explicitly, after building modal bases $V_r$, $V_\theta$, $V_\zeta$, the
code applies

$$
y = (V_r^T \otimes V_\theta^T \otimes V_\zeta^T) x,
$$

then divides modewise by

$$
d_{abc} = \sum_{m=1}^R
\lambda_{r,a}^{(m)}\lambda_{\theta,b}^{(m)}\lambda_{\zeta,c}^{(m)},
$$

and finally transforms back. So even in the multi-term case, the expensive part
is still reduced to 1-D basis transforms and pointwise division in modal space.

So CP-ALS gets us back to a computationally good situation on the bulk: not an
exact single tensor product, but a short structured representation that can be
applied by 1-D algebra plus elementwise modal division.

## 5. What Is Actually Stored

For a given degree `k`, the code first forms the extracted matrix

$$
M_k^{\mathrm{ext}} = E_k M_k E_k^T,
$$

using the relevant extraction operator `E_k` for free or Dirichlet boundary
conditions.

It then identifies the small extracted rows that do not belong to the regular
bulk tensor window.

For the cases that currently matter in production, these surgery sets are
explicit:

- `k = 0`: a scalar core of size $3 n_\zeta$,
- `k = 1`: the first $2 n_\zeta$ rows of the extracted `theta` component and
  the first $3 d_\zeta$ rows of the extracted `zeta` component,
- `k = 2`: the first $2 d_\zeta$ rows of the extracted `r` component,
- `k = 3`: no surgery block.

These are not arbitrary numerical thresholds. They are structural facts about
how the extracted polar spaces are laid out in coefficient space.

Then it stores:

- the exact small Schur/surgery data coming from the extracted matrix,
- tensor-factor data for the bulk blocks.

The bulk models are built from the diagonal mapped coefficient fields:

- `k = 0`: `J`,
- `k = 1`: `J g^{rr}`, `J g^{theta theta}`, `J g^{zeta zeta}`,
- `k = 2`: `g_rr / J`, `g_theta theta / J`, `g_zeta zeta / J`,
- `k = 3`: `1 / J`.

These fields are sampled on the tensor quadrature grid as 3-tensors and then fit
by CP-ALS. Each rank-1 term is converted into three weighted 1-D mass matrices,
with the appropriate radial restriction for the extracted bulk window.

So the tensor object stores neither the full extracted matrix nor a dense bulk
inverse. It stores exactly the data needed by the structured solve:

- Schur coupling matrices such as $A_{cb}$ and $A_{bc}$,
- a dense inverse of the small Schur complement,
- either direct 1-D inverse factors for a rank-1 block,
- or shared modal bases plus per-term modal diagonals for a multi-term block.

This also explains the cost split.

Setup is the expensive part. It must:

- sample the mapped coefficient field on the full tensor quadrature grid,
- run CP-ALS on that 3-tensor,
- assemble the weighted 1-D mass factors for each CP term,
- build the shared modal bases when the rank is larger than one,
- and form the small dense Schur complements coming from surgery.

Apply is much cheaper than setup. In the rank-1 case it is essentially three
1-D inverse applications plus the small dense Schur solve. In the multi-term
case it is three basis transforms, one elementwise modal division, three back
transforms, and again the small dense Schur solve. So the tensor route is not a
cheap-to-assemble preconditioner, but it is designed to amortize that setup cost
over many Krylov iterations or many right-hand sides.

So the stored tensor object is not the original coefficient tensor itself. It is
the Schur data plus the axis-wise factors needed to apply the inverse bulk model.

## 6. Scalar `k = 0` In Full

The scalar case is the cleanest complete example.

1. Build the extracted matrix and split it into core and bulk.
2. Fit the bulk coefficient field `J` by CP-ALS.
3. Build the bulk inverse model from the CP terms.
4. Form the dense Schur complement with that bulk inverse.
5. Apply the preconditioner by bulk solve, Schur solve, then back-substitution.

Written out more explicitly:

1. Assemble the raw mapped scalar mass matrix from the quadrature samples of
  $J$.
  
2. Extract it with $E_0$ to get $M_0^{\mathrm{ext}} = E_0 M_0 E_0^T$.

3. Permute or view it as

  $$
  M_0^{\mathrm{ext}} =
  \begin{pmatrix}
  A_{cc} & A_{cb} \\
  A_{bc} & A_{bb}
  \end{pmatrix},
  $$

  with a small core and a tensor-shaped bulk.

4. Sample the bulk coefficient tensor

  $$
  W_{ijk} = J_{ijk},
  $$

  and fit it by CP-ALS.

5. Convert each CP term into weighted 1-D mass factors and build a structured
  approximation $B_{bb}^{-1}$ to the bulk inverse.

6. Form

  $$
  S_c = A_{cc} - A_{cb} B_{bb}^{-1} A_{bc},
  $$

  invert $S_c$ densely, and store the result.
7. Apply the preconditioner by Schur back-substitution.

So `k = 0` shows the whole pattern already:

- exact dense treatment where extraction forces it,
- tensorized treatment where the bulk still has separable structure.

## 7. Vectorial Cases

The vectorial cases follow the same principle but with more block structure.

### `k = 1`

This is the most involved case.

- There is an outer surgery Schur for the extracted `theta` and `zeta` surgery
  rows.
- Inside the bulk, the `(r, theta_bulk)` part is still coupled, so it gets its
  own inner Schur completion.
- The three diagonal coefficient fields `J g^{rr}`, `J g^{theta theta}`,
  `J g^{zeta zeta}` are each fit by CP-ALS and turned into tensor bulk models.

The important point is that `k = 1` has two different reasons to lose a trivial
tensor inverse:

- extraction introduces the outer surgery rows,
- the algebraic coupling between the `r` and `theta_bulk` components introduces
  an inner block Schur even after the surgery rows have been removed.

So the production inverse is not

$$
\left(M^{\mathrm{ext}}_1\right)^{-1}
\approx B_r^{-1} \otimes B_\theta^{-1} \otimes B_\zeta^{-1},
$$

but a nested block factorization whose diagonal inverse pieces are tensorized.

So `k = 1` is not a single tensor inverse. It is a nested Schur factorization
whose diagonal bulk pieces are tensorized.

### `k = 2`

This mirrors `k = 1` but is simpler.

- There is one outer surgery Schur on the extracted `r` surgery block.
- The remaining `r_bulk`, `theta`, and `zeta` blocks are treated as separate
  tensor bulk inverses.
- Their coefficients are the diagonal entries of `g/J`.

So `k = 2` still has extraction-driven surgery, but no inner `(r,theta)` bulk
coupling analogous to the `k = 1` case. That is why the implementation is much
closer to “one outer Schur plus three tensor diagonal blocks”.

So `k = 2` keeps the same idea with less coupling than `k = 1`.

### `k = 3`

This is the second scalar case.

- There is no extracted surgery Schur.
- The whole extracted block is one scalar tensor block.
- The coefficient field is `1/J`.

So `k = 3` is the clean end of the program again: geometry still breaks exact
tensor separability, but extraction no longer adds a separate dense surgery
layer.

So `k = 3` is again close to the simple tensor picture, except that geometry has
first been compressed by CP-ALS.

## 8. Why Tensor Does Not Work As A Schur-Outer Object

For mixed degrees `k = 1, 2, 3`, the Schur operator is not a mass block. In the
shifted case it has the form

$$
S = L_k + \varepsilon M_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T.
$$

Even if the inverse mass action `M_{k-1}^{-1}` is approximated by a tensor
preconditioner, the composed operator

$$
D_{k-1} M_{k-1,\mathrm{tensor}}^{-1} D_{k-1}^T
$$

is not itself a tensor mass block on the extracted `k`-form space.

The reason is structural, not just implementation detail:

- `D` and `D^T` mix components and extracted-space blocks,
- the result lives on the Schur space, not on the lower mass space,
- after this sandwiching, the operator no longer has the direct bulk tensor
  form that the mass preconditioner exploits.

So the tensor mass object is valid for the inner inverse mass action, but not as
an outer preconditioner for the Schur operator itself.

This is exactly why the code allows:

- `mass = tensor`,
- `schur.inner = tensor`,

but rejects:

- `schur.outer = tensor`.

The outer Schur preconditioner must instead be something that can act on the
full Schur operator as an operator, such as:

- `jacobi`,
- `richardson`,
- `chebyshev`.

Those polynomial outer layers do not assume tensor structure of the Schur
matrix. They only need a cheaper approximate action, usually Jacobi, plus an
operator apply for the Schur system.

## 9. Programming View

At the code level there are three different objects involved:

- assembled tensor factors stored in `SequenceOperators.mass_preconds`,
- user-facing configuration objects such as `MassPreconditionerSpec` and
  `SaddlePointPreconditionerSpec`,
- callable preconditioner applies built inside `mrx/operators.py` and passed
  into CG or MINRES.

So `tensor` is not one monolithic solver object. It is a configuration choice
that tells the operator layer to use the already assembled tensor factors when
building the preconditioner apply.

## 10. Alternative Geometry Boundary

One plausible future variant is to store `DF` itself in tensorial low-rank form
and only form the derived geometry quantities when an assembly path needs them.

The attraction is that `DF` can have substantially lower tensor rank than the
derived quantities `J`, `g`, `g^{-1}`, or the diagonal coefficient fields built
from them.

The tradeoff is exactly the usual one:

- upside: lower-rank geometric storage,
- downside: more expensive runtime evaluation, because determinants, metric
  products, and pointwise inverses move from setup time into the active code
  path.

So this is a plausible alternative abstraction boundary, but it shifts work from
stored coefficient tensors to repeated nonlinear geometry reconstruction.

## 11. Practical Rule

Use the tensor route when the object you want to invert is still fundamentally a
mass-like block.

Do not expect the tensor route to survive the Schur sandwich unchanged. Once the
operator has become `D M^{-1} D^T` plus stiffness and shift terms, the correct
outer object is an operator preconditioner on the Schur space, not another mass
tensor inverse.