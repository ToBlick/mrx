# Manufactured solutions for the toroidal Hodge–Laplacian convergence tests

Reference notes so we stop re-deriving these. All verified by finite-difference
Laplace–Beltrami against the real `toroid_map` metric (see commit history /
`scripts/config_scripts/test_torus_poisson_all_k*.py`).

## Geometry: `toroid_map(epsilon=ε, R0=1, kappa=1)`

Reference coordinates `(r, χ, ζ) ∈ [0,1]³`, angles are 2π-scaled **inside the map**.

```
R   = 1 + ε r cos(2πχ)
g   = diag( ε²,  4π²ε²r²,  4π²R² )          (diagonal metric)
J   = √det g = 4π²ε²rR
g⁻¹ = diag( 1/ε², 1/(4π²ε²r²), 1/(4π²R²) )
```

Because the angles carry a factor 2π in the map, `g_ζζ = 4π²R²` **cancels** the
4π² produced by `∂ζ²cos(2πζ)`. Net effect: no stray 4π² in the ζ-Laplacian, e.g.
`-Δ cos(2πζ) = cos(2πζ)/R²`.

## Hodge star ⋆: Ω¹ → Ω² (cyclic vector-proxy convention)

The framework uses the **cyclic** convention (all-positive divergence/curl). The
2-form ref proxy slot order is `(χζ, rζ, rχ) = (Bʳ, Bᵡ, Bᶻ)`. For a 1-form with
covariant components `α = (α_r, α_χ, α_ζ)`:

```
⋆α = ( J/g_rr · α_r ,  J/g_χχ · α_χ ,  J/g_ζζ · α_ζ )
   = ( 4π²rR · α_r ,   R/r · α_χ ,     ε²r/R · α_ζ )      ← ALL POSITIVE
```

Verified: coefficients equal `J/g_ii` exactly, and `‖⋆α‖₂ = ‖α‖₁` using the
2-form error weights `weights2 = (g^χχg^ζζ, g^rrg^ζζ, g^rrg^χχ)` with measure `J`.

## Sign conventions

- Laplace–Beltrami `Δ_LB` is negative semi-definite; Laplacian `L_k`
  is positive semi-definite. For scalars `L₀ = -Δ_LB`, so **`L₀ u₀ = f₀`** with
  `f₀ = -Δ_LB u₀` and `δ₁ du₀ = +f₀`.
- Chain identities: `L_k(dα) = d(L_{k-1}α)` and `L_k(⋆α) = ⋆(L_{n-k}α)`.

---

## Consolidated script — `test_torus_poisson_all_k_sparse.py`

A single script sweeps the eight `(k, boundary-condition)` Laplace problems.
Cases pair under Hodge duality `⋆: k ↔ (3-k)`, `NBC ↔ DBC`, into four generators:

| generator | cases | exact / source |
|---|---|---|
| `cos(2πζ)` | k0 NBC, k3 DBC | `u = cos(2πζ)`, `f₀ = cos(2πζ)/R²` |
| `cos(πr²/2)` | k0 DBC, k3 NBC | `u = cos(πr²/2)`, `f₀ = (2πs + π²r²c)/ε² + πrs cos(2πχ)/(εR)` |
| `cos(2πζ) dζ` | k1 NBC, k2 DBC | `ω₁ = cos(2πζ) dζ`, `ω₂ = ⋆ω₁`; `f₁ = grad σ`, `f₂ = ⋆f₁` |
| `cos(πr²/2)cos(2πζ) dζ` | k1 DBC, k2 NBC | `ω = cos(πr²/2)cos(2πζ) dζ`, `σ = -div ω`, `f₁ = dσ + curl·curl ω`, `f₂ = ⋆f₁` |

with `s = sin(πr²/2)`, `c = cos(πr²/2)`.

The script's `CASES` list enables **all eight (k, BC) pairs**: k0 NBC, k0 DBC,
k1 NBC, k1 DBC, k2 NBC, k2 DBC, k3 NBC, k3 DBC.  All enabled cases share one `DeRhamSequence` and one assembly pass; the
Schur-Jacobi preconditioner is probed for `ks=(1,2,3)` × `{NBC, DBC}`, and one
`compute_nullspaces_iterative` call (`betti=(1,1,0,0)`) supplies the harmonic
projectors for every `(k, dirichlet)` pair (`_n_vectors` yields NBC dims
`1,1,0,0` and DBC dims `0,0,1,1`).  Per-k error dispatch picks the extraction
transpose `e{k}_dbc_T` (DBC) or `e{k}_T` (NBC).

| field | formula |
|---|---|
| `u₀` cos generator (k0 NBC, k3 DBC) | `cos(2πζ)`,  `f₀ = cos(2πζ)/R²` |
| `u₀` radial generator (k0 DBC, k3 NBC) | `cos(πr²/2)`,  `f₀ = (2πs + π²r²c)/ε² + πrs cos(2πχ)/(εR)` |
| `u₃` (proxy scalar) | generator scalar,  `f₃ = f₀·J` |
| `ω₁` (k1 NBC, covariant) | `(0, 0, cos 2πζ)`  ≡ `cos(2πζ) dζ` |
| `σ = δ₁ω₁ = -div ω₁` | `sin(2πζ) / (2π R²)` |
| `f₁ = L₁ω₁ = grad σ` (covariant) | `∂(σ)` via autodiff (curl·curl ω₁ = 0) |
| `ω₂ = ⋆ω₁` (proxy χζ,rζ,rχ) | `(0, 0, ε²r cos2πζ / R)` *(k=2 DBC exact)* |
| `f₂ = ⋆f₁` (proxy χζ,rζ,rχ) | `⋆(grad σ)` *(k=2 DBC source)* |

### Radial cosine generator (`u = cos(πr²/2)`) — k0 DBC, k3 NBC

```
u   = cos(πr²/2)                   (u|_{r=1} = 0, smooth at r=0)
f₀  = -Δu = (2πs + π²r²c)/ε² + πrs cos(2πχ)/(ε R)
   with s = sin(πr²/2), c = cos(πr²/2)
```

Derivation (only `∂_r u` nonzero):
```
∂_r u = -πr sin(πr²/2)

-Δu = -(1/J) ∂_r( J g^{rr} ∂_r u )
  = (2πs + π²r²c)/ε² + πrs cos(2πχ)/(ε R)
```
FD check: analytic and numeric `-Δu` agree to machine precision at sample points.

### Why k=1/k=2 work (toroidal construction)

Take `ω₁ = f(ζ) dζ` (only the ζ-covariant component nonzero):
- **u·n = 0** automatically (ω_r = 0, diagonal metric). ✓ natural k=1 NBC.
- **curl-free**: `dω₁ = f'(ζ) dζ∧dζ = 0` for ANY f. So `curl·curl ω₁ = 0` and
  `f₁ = L₁ω₁ = grad σ`, a pure gradient with `σ = -div ω₁ = f'(ζ)/(4π²R²)`·(−1)
  → for `f=cos2πζ`, `σ = sin(2πζ)/(2πR²)`.
- **harmonic-orthogonal**: harmonic 1-form is `h = dζ`. `⟨ω₁,h⟩ = ε²(∫ r/R)·∫f(ζ)dζ`,
  so orthogonality ⟺ **`f` has zero mean**. `cos(2πζ)` qualifies. ✓

The Hodge dual `ω₂ = ⋆ω₁` then serves **k=2 DBC**:
- **u·n = 0**: the normal (χζ) slot of ⋆ω₁ is zero. ✓ essential k=2 DBC.
- **σ = δ₂ω₂ = 0**: ω₂ is co-closed because ω₁ is closed (⋆ intertwines d↔δ). ✓
- **harmonic-orthogonal**: ⋆ is an isometry, so `⟨ω₂,⋆h⟩₂ = ⟨ω₁,h⟩₁ = 0`. ✓

All three k=2 facts verified by FD: normal slot = 0, σ-formula matches, `f₂=⋆f₁`.

### k=1 DBC (`ω = cos(πr²/2)cos(2πζ) dζ`, generator 4)

The k=1 DBC vector-Laplace problem needs essential BCs `u×n = 0` (tangential trace
zero) and `σ = -div u = 0` at the wall.  Take

```
ω = cos(πr²/2)cos(2πζ) dζ,
```

covariant

```
(0, 0, cos(πr²/2)cos(2πζ)).
```

- **boundary conditions still hold**: now `div ω` is nonzero in the interior, but
  because `ω_ζ` still carries the factor `cos(πr²/2)`, both `u×n = 0` and
  `σ = -div ω = 0` hold at `r=1`.
- **u×n = 0**: the only tangential component is `ω_ζ = cos(πr²/2)cos(2πζ)`, which vanishes at
  `r=1`. ✓
- **no nullspace** for k=1 DBC, so no orthogonality constraint.

Define

```
σ = δ₁ω = -div ω = cos(πr²/2) sin(2πζ) / (2π R²).
```

Then the strong-form source is the full

```
f₁ = L₁ω = dσ + δdω.
```

Writing `Cχ = cos(2πχ)`, `Sχ = sin(2πχ)`, `S = sin(2πζ)`, `Z = cos(2πζ)`, the
covariant source components are

```
f₁_r = - ε Cχ c S / (π R³)
f₁_χ =   2 ε r Sχ c S / R³
f₁_ζ =   Z [ c/R² + (2πs + π²r²c)/ε² - πrs Cχ/(εR) ]
```

with `s = sin(πr²/2)`, `c = cos(πr²/2)`.

So unlike the old radial-only field, the mixed source probes **all three**
covariant 1-form components.

The Hodge dual

```
ω₂ = ⋆ω = (0, 0, ε²r cos(πr²/2)cos(2πζ)/R)
```

(proxy `rχ` slot) serves **k=2 NBC** with `f₂ = ⋆f₁`.  It is no longer closed in
the interior, but its boundary trace still vanishes at the wall because the same
factor `cos(πr²/2)` kills the tangential trace at `r=1`.

---

## Appendix: adding a ζ-dependence to the radial cosine generator

Question: can we take the current radial cosine generator and also make it depend
on `ζ`, e.g.

```
ω(r,χ,ζ) = φ(r) h(ζ) dζ,
φ(r) = cos(πr²/2),
h(ζ) = cos(2πζ)
```

without breaking the k=1/k=2 boundary conditions?

Short answer: **yes on the boundary, no globally**.  The field is no longer
divergence-free everywhere, but it still satisfies the essential k=1 DBC
conditions at `r=1`, and its Hodge dual still satisfies the k=2 NBC boundary
conditions.  The price is that the source is now the full
`f₁ = dδω + δdω`, not just `δdω`.

Below we work this out explicitly for

```
φ(r) = cos(πr²/2),   s(r) = sin(πr²/2),
h(ζ) = cos(2πζ),     S(ζ) = sin(2πζ),
R = 1 + ε r cos(2πχ).
```

We also write

```
Cχ = cos(2πχ),   Sχ = sin(2πχ),   Z = cos(2πζ).
```

### Step 1: boundary conditions

Take the covariant 1-form

```
ω = φ(r) Z dζ = (0, 0, φ(r) Z).
```

Since `φ(1) = cos(π/2) = 0`, the tangential trace vanishes at the wall:

```
u×n = 0  at r = 1.
```

For the second k=1 DBC condition,

```
σ = δ₁ω = -div ω.
```

Because

```
J g^{ζζ} ω_ζ = (ε² r / R) φ(r) Z,
```

we get

```
σ = -(1/J) ∂_ζ(J g^{ζζ} ω_ζ)
  = -φ(r) h'(ζ) / (4π² R²).
```

For `h(ζ) = cos(2πζ)`, `h'(ζ) = -2π sin(2πζ)`, so

```
σ = φ(r) S(ζ) / (2π R²).
```

Again `φ(1) = 0`, so `σ = 0` at the wall.  Thus both essential k=1 DBC
conditions still hold.

### Step 2: the gradient part `dσ`

Since `σ` is a scalar, `dσ` is the covariant gradient:

```
dσ = (∂rσ, ∂χσ, ∂ζσ).
```

For `σ = φ(r) S(ζ) / (2π R²)`, this gives

```
∂rσ = -S(ζ) [ r s(r) / (2 R²) + ε Cχ φ(r) / (π R³) ]
∂χσ =  2 ε r Sχ φ(r) S(ζ) / R³
∂ζσ =  φ(r) Z / R²
```

so

```
dσ = (
  -S(ζ) [ r s/(2R²) + ε Cχ φ/(πR³) ],
   2 ε r Sχ φ S(ζ) / R³,
   φ Z / R²
).
```

The important new feature is the **nonzero χ-component** coming from the `χ`
dependence of `R^{-2}`.

### Step 3: the curl-curl part `δdω`

First,

```
dω = ∂r(φ Z) dr∧dζ = φ'(r) Z dr∧dζ = -π r s(r) Z dr∧dζ.
```

For this special ansatz, `δdω` still has a simple closed form:

```
δdω = (
   r s(r) S(ζ) / (2 R²),
   0,
   Z [ (2π s(r) + π²r²φ(r))/ε² - π r s(r) Cχ / (ε R) ]
).
```

The `ζ`-component is exactly the old radial-cosine k=1 DBC source multiplied by
`Z = cos(2πζ)`.  The `r`-component is new and comes from the `ζ`-derivative in
`δdω` once `h(ζ)` is no longer constant.

### Step 4: the full strong-form 1-form source `f₁ = dσ + δdω`

Adding the two pieces gives substantial cancellation in the `r`-component:

```
f₁ = dσ + δdω
```

with covariant components

```
f₁_r   = - ε Cχ φ(r) S(ζ) / (π R³)
f₁_χ   =   2 ε r Sχ φ(r) S(ζ) / R³
f₁_ζ   =   Z [ φ(r)/R²
               + (2π s(r) + π²r²φ(r))/ε²
               - π r s(r) Cχ / (ε R) ]
```

So for the candidate mixed generator

```
ω = cos(πr²/2) cos(2πζ) dζ,
```

the strong-form k=1 source is

```
f₁ covariant = (
  - ε cos(2πχ) cos(πr²/2) sin(2πζ) / (π R³),
    2 ε r sin(2πχ) cos(πr²/2) sin(2πζ) / R³,
    cos(2πζ) [ cos(πr²/2)/R²
               + (2π sin(πr²/2) + π²r² cos(πr²/2))/ε²
               - π r sin(πr²/2) cos(2πχ)/(ε R) ]
).
```

### Step 5: the k=2 source `f₂ = ⋆f₁`

Using the cyclic proxy convention

```
⋆α = ( 4π² rR α_r,  R/r α_χ,  ε²r/R α_ζ ),
```

the strong-form 2-form source is the proxy

```
f₂ = ⋆f₁ = (f₂_χζ, f₂_rζ, f₂_rχ)
```

with

```
f₂_χζ = -4π ε r Cχ φ(r) S(ζ) / R²
f₂_rζ =  2 ε φ(r) Sχ S(ζ) / R²
f₂_rχ =  ε² r/R · f₁_ζ
```

or, expanding the third slot,

```
f₂_rχ = r Z [ ε² φ(r)/R³
              + (2π s(r) + π²r²φ(r))/R
              - ε π r s(r) Cχ / R² ].
```

### Step 6: what changes conceptually?

For the old radial-only k=1 DBC generator, `σ = 0` everywhere and the source was
just `δdω`.  After adding a `ζ` dependence,

```
σ ≠ 0   in the interior,
```

so the source is genuinely the full

```
f₁ = dδω + δdω.
```

The boundary conditions still work because `φ(1)=0`, but the field is no longer
globally divergence-free.

### Step 7: cheap validation

The formulas above were checked pointwise against the geometric identity

```
f₁ = dσ + δdω
```

using the actual `toroid_map` metric and JAX autodiff at three interior sample
points.  The analytic and direct-evaluation results matched to about `1e-14` in
the maximum componentwise difference.

### Practical note for the script

These are the **strong-form** sources.  The current script's load convention for
vector forms still applies unchanged:

- k=1: pass the **raised** source `G^{-1} f₁_cov` into `seq.load(..., k=1)`
- k=2: pass the **scaled proxy** `(G/J) f₂_proxy` into `seq.load(..., k=2)`

So if we decide to use this mixed generator in the all-k script, only the source
builders change; the load-convention fix stays the same.
