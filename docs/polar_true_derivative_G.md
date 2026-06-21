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

**Shipped (in `mrx/operators.py`, `apply_incidence_matrix`, validated):** the
true `G` applied **in place**, equivalently `G = Gram_{k+1}^{-1} · (Eᵀ sp E)`,
`Gram = EᵀE`. Construction:
- `Gram` is built **sparsely** (`E` from the extraction's own triplets,
  `E·Eᵀ` sparse — no `todense` of the incidence/extraction), and is
  block-diagonal (identity bulk + a small polar-axis fusion block `S×S`);
- only the small `S×S` axis block is inverted (dense, tiny), stored as a sparse
  `inc_gram_inv_{1,2,3}[_dbc]` (`identity + axis correction`);
- runtime is a **single sparse matvec** (`Gram⁻¹ · (Eᵀ sp E)`, and the proper
  adjoint for transpose) — **inverse-free and `todense`-free at runtime**;
- bit-identical to the old apply on non-polar (`Gram = I` there).

Validation (CPU, polar `rotating_ellipse`, `ns=6,8,4`, p=3): new apply == dense
`Gram⁻¹·raw` reference ≤ 6.7e-16; `d∘d = 0` to ≤ 3.4e-16 (both BCs, k=0/1);
`pytest test/test_operators.py` — 56 passed (incl. a new polar nilpotency test).

**Not yet shipped — the pure analytic `ξ`-stencil.** The intended end state
(assemble `G` directly from the formulas above, zero inverse even at assembly).
Status: grad `G_0` was prototyped and validated **bit-exact** against the
`Gram⁻¹` oracle (max err 8.9e-16, both BCs); the radial subtlety is that the
r-slice DoF `i` maps to `D_{i+1} = f_{i+2} − f_{i+1}` (full-r index), the first
ring `i=0` carries the `−ξ^ℓ_{1j}` apex-fusion couplings, and the dbc outer
boundary drops the trailing `+f`. **curl `G_1` and div `G_2` were not completed**
— three separate attempts judged shipping all-three-bit-exact into a
used-everywhere core operator in one pass too risky relative to the gain (the
shipped Gram apply is already inverse-free at runtime; the analytic only removes
the one-time `S×S` assembly inverse).

**Remaining gap vs the ideal:** a small dense `S×S` axis-block inverse at
**assembly** (the "small dense ops near the axis" worst case). To remove it,
finish the analytic curl/div using the formulas above, validated column-by-column
against the current `Gram⁻¹` apply (a bit-exact oracle), then drop
`_build_inc_gram_inv`.

## Pointers

- Apply + construction: `mrx/operators.py` — `apply_incidence_matrix`,
  `_build_inc_gram_inv`, `_inc_gram_inv`, `SequenceOperators.inc_gram_inv_*`.
- Index layout + `ξ` conventions: `mrx/extraction_operators.py` —
  `build_extraction`, `_k1_row_slices`, `_lambda_col_index`.
- Tests: `test/test_operators.py` — `test_curl_of_grad_is_zero`,
  `test_div_of_curl_is_zero` (non-polar) and the polar nilpotency test.
- Diagnostics: `scripts/diag_derham_exactness.py`,
  `scripts/diag_true_G_exactness.py`.
