# The true polar discrete derivative `G` (matrix-free)

Status and reference for the discrete exterior derivative (grad/curl/div) on the
**polar** FEEC B-spline complex `V0 -G0-> V1 -G1-> V2 -G2-> V3`. This documents a
correctness bug in the directly-built incidence apply, the fix that shipped, the
analytic `ξ`-stencil that is the intended end state, and where that stands.

## The problem

The discrete derivative on coefficients is the **topological** incidence: if `f`
are the coefficients of a `k`-form and `g` of its derivative `d f` in the
`(k+1)`-form basis, then `g = G f` with `G` a metric-free `±1` matrix. The FEEC
identity is

```
M_{k+1} g = D f,     D = (φ^{(k+1)}, d φ^{(k)})  (weak-derivative mixed mass)
=>  g = M_{k+1}^{-1} D f =: G f      (G is topological; D is never formed)
```

On the **non-polar** sequence the extraction `E` (boundary/periodicity) is a 0/1
selection, so `EᵀE = I` and the directly-built apply

```
apply_incidence_matrix(v, k) = e_out · sp · e_in^T  =  E_{k+1}^T sp E_k
```

*is* the topological `G` and is nilpotent (`d∘d = 0`). On the **polar** sequence
the axis extraction is **not** a 0/1 selection (it fuses several B-splines at
`r=0` with `ξ`-weights), so `EᵀE ≠ I` (deviation ~14 for V0/V1, ~1.3 for V2, 0
for V3), and `apply_incidence` **omits the inverse**:

```
true G = M^{-1} D = (E^T E)^{-1} E^T sp E   ≠   E^T sp E = apply_incidence
```

Consequence: on polar, `apply_incidence` is **not nilpotent** — `curl·grad ≈ 1`,
`div·curl ≈ 0.02` (vs machine zero with the true `G`). The existing
`test_operators.py` nilpotency tests passed only because `_SEQ` there is
`polar=False` + identity map (the file comments this explicitly). The error is
localized to the polar-axis fusion DoFs; far from the axis `G` is plain `±1`.

This bug silently degraded the vector-form preconditioners (it made the k=1
gradient projector non-idempotent and broke the k=2 projector); see
`hiptmair_xu_preconditioner.md`.

## The analytic `ξ`-stencil (the intended matrix-free `G`)

`G` is local: coefficient differences in the bulk, with `ξ`-weighted axis
stencils. `ξ[p,i,j]` are the polar mapping coefficients (as in
`build_extraction`); the two reusable differences are
`val_r = ξ[p,1,j] − ξ[p,0,j]` (radial) and `val_θ = ξ[p,1,j+1] − ξ[p,1,j]`
(angular). Bases: `N` = primal spline, `D` = derivative spline; superscripts `ps`
(radial), `pχ` (angular); `F` = ζ spline. Apex DoFs are `f_{0k}, f_{1k}, f_{2k}`
(the "+3"); first bulk ring is `f_{2jk}` (full radial index 2).

### grad `G_0` : V0 → V1   (1-form comps: s = `Dᵖˢ⊗Nᵖᵡ`, χ = `Nᵖˢ⊗Dᵖᵡ`)

```
[∂_s f ; ∂_χ f] =
   Σ_{ℓ=1}^{2} (f_{ℓk} − f_{0k}) Σ_j [ (ξ^ℓ_{1j} − ξ^ℓ_{0j}) D^ps_0 N^pχ_j ;        # apex
                                       (ξ^ℓ_{1,j+1} − ξ^ℓ_{1j}) N^ps_1 D^pχ_j ]
 + Σ_j ( f_{2jk} − Σ_{ℓ=0}^{2} f_{ℓk} ξ^ℓ_{1j} ) [ D^ps_1 N^pχ_j ; 0 ]             # first ring (radial)
 + Σ_{i>1,j} (f_{i+1,jk} − f_{ijk}) [ D^ps_i N^pχ_j ; 0 ]                          # bulk radial (±1)
 + Σ_{i>1,j} (f_{i,j+1,k} − f_{ijk}) [ 0 ; N^ps_i D^pχ_j ]                         # bulk angular (±1)
```

The two apex 1-form basis functions added at the axis are
```
Λ̃¹_{2,ℓ-1} = Σ_j [ (ξ^ℓ_{1j} − ξ^ℓ_{0j}) D^ps_0 N^pχ_j ;
                   (ξ^ℓ_{1,j+1} − ξ^ℓ_{1j}) N^ps_1 D^pχ_j ; 0 ],   ℓ = 1,2 (ξ^1, ξ^2).
```
(These `val_r`/`val_θ` weights are exactly the V1 extraction surgery rows in
`build_extraction`, lines 619/639.)

### curl `G_1` : V1 → V2   (in-plane third comp; `e_1`=s, `e_2`=χ coeffs)

3rd (ζ, `ê₃`) component on the standard tensor basis `Λ̃²_{3,(ijk)} = Dᵖˢ_i Dᵖᵡ_j Fⁿ_k ê₃` (i>0):
```
∂_s E1_{(2)} − ∂_χ E1_{(1)} =
   Σ_j [ e_{2,(2jk)} − Σ_{ℓ=0}^{1} e_{2,(ℓk)} (ξ^{ℓ+1}_{1,j+1} − ξ^{ℓ+1}_{1j})       # axis (first ring)
         − e_{1,(1,j+1,k)} + e_{1,(1jk)} ] D^ps_1 D^pχ_j
 + Σ_{i>1,j} ( e_{2,(i+1,jk)} − e_{2,(ijk)} − e_{1,(i,j+1,k)} + e_{1,(ijk)} ) D^ps_i D^pχ_j   # bulk (±1)
```
The other two 2-form components come from the grad stencil applied to the 1-form
ζ-component `E1_{h,3} = e₃ᵀΛ⁰`, crossed with `ê₃`: `Λ̃²_{1,ℓ} = Λ̃¹_{2,ℓ} × ê₃`.

### div `G_2` : V2 → V3   (`b_1`, `b_2` = first two 2-form comps)

```
∂_s B2_{(1)} + ∂_χ B2_{(2)} =
   Σ_j [ b_{1,(2jk)} − Σ_{ℓ=0}^{1} b_{1,(ℓk)} (ξ^{ℓ+1}_{1,j+1} − ξ^{ℓ+1}_{1j})       # axis (first ring)
         + b_{2,(1,j+1,k)} − b_{2,(1jk)} ] D^ps_1 D^pχ_j
 + Σ_{i>1,j} ( b_{1,(i+1,jk)} − b_{1,(ijk)} + b_{2,(i,j+1,k)} − b_{2,(ijk)} ) D^ps_i D^pχ_j   # bulk (±1)
```

All three: bulk = plain `±1` incidence; axis = `val_r`/`val_θ` (`ξ`-difference)
apex/first-ring stencils — coefficient differences and `ξ` weights only, **no
mass, no inverse**. (`build_extraction` already uses the same `ξ`-differences:
k=1 lines 619/639, k=2 lines 727/747.)

## Where we stand

**The full polar `G` is now matrix-free and inverse-free — no assembly inverse
at all.** grad `G_0` and curl `G_1` are analytic `ξ`-stencils; div `G_2` needs no
correction (its output V3 extraction is unitary). The old `Gram⁻¹` precompute
(`_build_inc_gram_inv` → `inc_gram_inv_*`) is **no longer built**; it survives
only as the bit-exact oracle for the validation harness.

**Shipped — analytic `ξ`-stencil grad `G_0`** (`build_grad_stencil_g0`, wired into
`apply_incidence_matrix` `k==0`). Built from the incidence pattern + `ξ` with no
inverse, all four `(din,dout)` BC pairs (fwd+transpose), stored `g0_grad_{di}{do}[_T]`.
Stencil: theta-surgery = `±1` apex differences `apex(p,m)−apex(0,m)`; zeta-surgery
= periodic-`z` apex differences; r-slice (comp0) DoF `i` = `f_{i+2}−f_{i+1}` with
the first ring `i=0` carrying the `−ξ^ℓ_{1j}` apex couplings (dbc drops the outer
`+f`); theta/zeta-bulk = `±1`.

**Shipped — analytic `ξ`-stencil curl `G_1`** (`build_curl_stencil_g1`, wired into
`apply_incidence_matrix` `k==1`). Same construction one degree up, stored
`g1_curl_{di}{do}[_T]`. Full-space curl `P=−d_z b+d_t c`, `Q=d_z a−d_r c`,
`R=−d_t a+d_r b` (a=s,b=χ,c=ζ); V1 input fusion inverted by an `expand_v1` helper
(the V1 analog of grad's `expand` — apex/surgery DoFs carry `ξ` weights). The only
fused V2 *output* DoFs are the comp0 surgery rows, whose stencil is the axis form
`surgery(pl,m) = [θ_s(pl,m) − θ_s(pl,(m+1)%dz)] + [ζ_s(pl+1,m) − ζ_s(0,m)]`
(`= −d_z χ + ∂_θ ζ` at the axis). The V2 `Gram⁻¹` cancels analytically.

**Div `G_2` — already matrix-free, no stencil needed.** Output V3's extraction is
a plain `0/1` selection (`‖E₃E₃ᵀ − I‖ = 0`, unitary), so `Gram₃ = I` and
`apply_incidence(.,2)` is already the true div; input V2 fusion is handled by the
raw `E₂ᵀ` expansion.

Validation (CPU, polar `rotating_ellipse`, `ns=6,8,4` p=3 **and** `4,4,3` p=2,
`scripts/diag_grad_analytic_stencil.py`): analytic grad **and** curl == the
independent `Gram⁻¹` oracle **and** the shipped `apply_incidence` to **≤7.8e-16**,
all four `(din,dout)` pairs, forward + transpose; `curl∘grad` and `div∘curl`
nilpotency ~1e-16 (both BCs). Gram devs `V1=1.56, V2=0.35, V3=0.00`.
`pytest test/test_operators.py` — 56 passed (non-polar fallback stays bit-identical).

**No remaining assembly inverse.** Polar is detected Gram-free via
`_extraction_is_polar` (one `E Eᵀ x ≠ x` probe). On non-polar the stencil fields
stay `None` → raw incidence fallback (bit-identical). `_build_inc_gram_inv` /
`_inc_gram_inv` and the `inc_gram_inv_*` fields remain only as the oracle / the
(always-`None`) div lookup; nothing is precomputed.

## Pointers

- Analytic stencils: `mrx/operators.py` — `build_grad_stencil_g0`,
  `build_curl_stencil_g1`, `_grad_stencil`, `_curl_stencil`,
  `_extraction_is_polar`, `SequenceOperators.{g0_grad,g1_curl}_*`; validation
  `scripts/diag_grad_analytic_stencil.py`.
- Oracle (test-only): `mrx/operators.py` — `_build_inc_gram_inv`, `_inc_gram_inv`.
- Index layout + `ξ` conventions: `mrx/extraction_operators.py` —
  `build_extraction`, `_k1_row_slices`, `_lambda_col_index`.
- Tests: `test/test_operators.py` — `test_curl_of_grad_is_zero`,
  `test_div_of_curl_is_zero` (non-polar) and the polar nilpotency test.
- Diagnostics: `scripts/diag_derham_exactness.py`,
  `scripts/diag_true_G_exactness.py`.
