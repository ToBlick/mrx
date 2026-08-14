# Pole regularity of polar splines in mrx: C⁰ / C¹ / C² (`polar_order`)

**Status (2026-07-09):** C² (and C⁰) polar 0-form extraction implemented,
verified, and convergence-validated for the k=0 pipeline;
`DeRhamSequence(polar_order={0,1,2})`. Derivation cross-validated
point-for-point against Toshniwal–Speleers–Hiemstra–Hughes, *Multi-degree
smooth polar splines*, CMAME 316 (2017) 1005–1061
(`docs/1-s2.0-S004578251631533X-main.pdf`; the C¹ special case is Ch. 5 of
the Holderied STRUPHY thesis). One deliberate deviation from the paper —
**collocated C² instead of exact C²** — is documented below with its
rationale and measured cost (none, at the PDE level).

## 1. Setting

Logical coordinates (s, χ) with the radial clamped basis N_i(s) (rings
i = 0, 1, 2, …) and periodic χ-basis N_j(χ), degenerate spline map
x_h(s,χ) = Σ_i P_i(χ) N_i(s) with the entire ring-0 control ring collapsed
onto the pole (P₀ ≡ pole). A scalar spline is f = Σ_i c_i(χ) N_i(s),
c_i(χ) = Σ_j c_{ij} N_j(χ). "C^k at the pole" means f matches, on the
collapsed edge s = 0 and for every χ, the s-jets of q∘x_h for a single
polynomial q of degree ≤ k (jet matching; this is also the paper's
operative definition, their Eqs. 3.7–3.8).

## 2. The ring conditions (jet-matching derivation)

Matching ∂ᵐ_s f(0,χ) = ∂ᵐ_s (q∘x_h)(0,χ) for m ≤ k, with
q(x) = q₀ + q₁·x + xᵀQx and ΔP_i := P_i − pole, gives

    m=0 (C⁰):  c₀(χ) = q₀                                     — ring 0 constant
    m=1 (C¹):  c₁(χ) = q₀ + q₁·ΔP₁(χ)                          — affine in ring-1 offsets
    m=2 (C²):  c₂(χ) = q₀ + q₁·ΔP₂(χ) + ρ · ΔP₁(χ)ᵀ Q ΔP₁(χ),
               ρ = 2 N₁'(0)² / N₂''(0).

Only rings 0..k are constrained; the polar DOF block has dimension
(k+1)(k+2)/2 = 1 / 3 / 6. Two structural facts worth noting:

- **The Hessian enters through ring 1, not ring 2.** The naive
  generalization of the C¹ barycentric recipe — "evaluate q at the ring-2
  control points" — is wrong: the quadratic form is contracted with the
  ring-1 tangent curve, scaled by the knot-dependent constant ρ. (In the
  paper this is Eq. (3.33): our ρ is their (C^η₁,₁)²/C^η₂,₂, the 2 being
  Taylor convention; their α-term — the map's second radial derivative at
  the pole — is our q₁·ΔP₂ term after the N₁''ΔP₁ cancellation.)
- **C¹ is radial-knot-independent, C² is not.** The end-derivative
  constant cancels in the m=1 condition (which is why the classical ξ
  weights carry no knot data) but survives in m=2. `get_xi2` therefore
  computes N₁'(0), N₂''(0) by AD of the actual radial basis — any grading
  (equal-area, anchored-ξ₁, custom knots) is handled automatically.

## 3. The obstruction, and the two ways out

ΔP₁(χ)ᵀQΔP₁(χ) is a **product of splines** — degree 2p in χ — while
c₂(χ) lives in the degree-p space. Hence **exact C² with respect to the
discrete map is impossible in a fixed degree-p tensor space.** This is the
precise content of the thesis' unexplained "arbitrary C^k in principle
possible", and the reason every implementation known to us stops at C¹.

Two resolutions:

1. **Exact (Toshniwal et al. — the "multi-degree" in their title).** Make
   the angular *function* space rich enough to contain the products: for
   C², angular degree ≥ 6 with the *map* confined to the half-degree
   (≤ 3), non-rational space (their Definition 3.1 "k-compatibility",
   Remark 3.11, Section 3.3.4). C² then holds pointwise everywhere.
2. **Collocated (mrx).** Keep degree p everywhere and sample the quadratic
   term at the Greville angles: c_{2j} = q₀ + q₁·ΔP_{2j} + ρ ΔP₁ⱼᵀQΔP₁ⱼ.
   The pole 2-jet is exact at the collocation angles and O(h^{p+1}) in
   between — the same sampled-coefficient class as the C¹ construction's
   own pole jets (which are spline-sampled trigonometric functions, not
   exact ones).

mrx uses (2). Rationale: (1) would force angular degree 6 through the
whole solver stack (DOF count, operator bandwidth, quadrature cost) to
buy pointwise-exact smoothness that the weak formulation never sees. The
measured verdict (§5): the collocated space loses **nothing** in L2
convergence, including on solutions whose axis representation exercises
the ring-2 quadratic DOFs.

## 4. Basis and implementation

The 6 polar jets are the quadratic Bernstein polynomials on the C¹
equilateral control triangle (paper: Definition 3.2 uses degree-k
Bernstein for all k; Appendix A lists the k ≤ 2 pole jets). Consequences,
all verified numerically:

- partition of unity exact on every ring (Σ_α B_α = 1 ⇒ Σq₀ = 1, Σq₁ = 0,
  ΣQ = 0) ⇒ constants exactly representable;
- the affine (Q = 0) subspace reproduces the C¹ rings-0/1 structure ⇒
  **V⁰_C² ⊂ V⁰_C¹ ⊂ V⁰_C⁰** (each a genuine subspace; lstsq residuals
  ~1e-16);
- nonnegativity of the weights is *not* enforced (the paper inflates the
  C² triangle iteratively, τ₂ up to 8ρ₂, for convex-hull/geometric-design
  reasons); analysis needs only PoU, so mrx keeps the C¹ τ.

Code map:

| what | where |
|---|---|
| C² weights (6, 3, nθ) | `get_xi2(nt, basis_r, ring1, ring2)` — `mrx/extraction_operators.py` |
| C¹ weights, now map-adaptable | `get_xi(nt, ring1=None)` (thesis Eq. 5.7–5.9; `None` = circle) |
| generalized extraction | `PolarExtractionOperator` reads `n_polar`, `ring_depth` off the ξ shape; k ∈ {0, −1} fully general, k ∈ {1,2,3} guarded to C¹ ξ |
| sequence knob | `DeRhamSequence(polar_order={0,1,2})`; C⁰ is ξ = ones((1,1,nθ)) |
| verification suite | `scripts/debug/verify_c2_polar.py` |
| convergence study | `scripts/debug/poisson_k0_c2_convergence.py` (`--orders 0,1,2`) |

**Scope**: only the k=0 pipeline is C²/C⁰-consistent. This is sound
because `apply_stiffness`/`apply_mass_matrix`/`load` sandwich the TENSOR
incidence and TENSOR mass between E⁰'s — E¹ is never touched. The k ≥ 1
extractions and the analytic polar grad/curl stencils encode the C¹
surgery (skipped on `polar_order != 1` sequences); extending the de Rham
complex to C² (polar 1-form space containing grad V̄⁰_C², commuting
C̄Ḡ = 0) is the deferred rework, mechanical via the Bernstein identities
but a few hundred lines of `extraction_operators.py` plus tests. The
production k=0 FD preconditioner also hardcodes the C¹ core layout
(3nz core, radial window at 2) — MG-prototype wiring of `polar_order=2`
must generalize that split (core 6nz, window at 3).

## 5. Verification and convergence results

`verify_c2_polar.py` (all PASS): AD end-derivatives vs finite differences;
PoU 4e-16; constants through the assembled E⁰ 4e-16; subspace-of-C¹
3e-16; and the decisive **Taylor-remainder scaling at the pole** —
quadratic-fit residual on shrinking circles: random C² element orders
3.00/3.00/3.00 (a genuine 2-jet), generic C¹ element (random ring 2)
stalls below 2.

Poisson k=0, toroid ε=1/3, dbc, p=3, manufactured solutions with poloidal
mode m = 0, 1, 2 (all smooth in physical coordinates; **m=2 is the
ring-2-critical case**), source terms by AD through the metric,
n ∈ {4, 6, 8, 12}, relative L2 at quadrature points:

| case | order | n=6 | n=8 | n=12 | final rate |
|---|---|---|---|---|---|
| m0 | C¹ / C² | 2.072e-3 / 2.072e-3 | 4.771e-4 / 4.771e-4 | 7.882e-5 / 7.882e-5 | 4.44 / 4.44 |
| m1 | C¹ / C² | 1.657e-3 / 1.657e-3 | 4.447e-4 / 4.447e-4 | 7.701e-5 / 7.701e-5 | 4.32 / 4.32 |
| m2 | C¹ / C² | 2.672e-3 / 2.672e-3 | 6.527e-4 / 6.527e-4 | 1.101e-4 / 1.101e-4 | 4.39 / 4.39 |
| m2 | C⁰ | 2.664e-3 | 6.521e-4 | 1.101e-4 | 4.39 |

**C² errors are identical to C¹ to every printed digit at every
resolution**, at 10–16% fewer DOFs, with consistently fewer
Jacobi-CG iterations (the removed near-axis DOFs were the stiff ones:
6 vs 10–11 at n=4). C⁰ (the larger space) tracks with marginally smaller
errors, as it must. A wrong ring-2 condition would have collapsed the m=2
rate toward ~2.5; it did not — the collocated-C² deviation from the paper
is empirically free for analysis.

## 6. Why this matters for the MG / preconditioner line

The measured fat-core results (`docs/research/laplacian_mg_k0_plan.md`) showed that
exactly solving the innermost bulk ring absorbs the dominant axis share of
the smoother-atom spread κ — but fat-core's Schur core grows as
3nz + nt·nz and its setup W-probe with it. `polar_order=2` moves the same
region *into the function space*: the exactly-solved core is **6nz
forever**, with no W-probe growth and no approximation penalty (§5). Next
measurement: wire `--polar-order {0,1,2}` into
`scripts/debug/laplacian_mg_k0.py` (generalize the bulk window to start at
ring `1+polar_order` and probe the 6nz core) and compare κ against
fat-core R=1 on toroid/cerfon/rotating-ellipse; the C⁰ arm doubles as the
control that shows how much pole regularity itself (not just core size)
matters for the spread.

Map adaptation: `get_xi(nt, ring1=…)` / `get_xi2(…, ring1, ring2)` accept
actual control-ring offsets (`ring1_control_points()` extracts them from
a poloidal map's Greville interpolant). The circle default is exact
whenever ∂F/∂r at the axis is pure m=±1 (includes ellipses); shaped
cross-sections (cerfon triangularity, W7-X) are the map-dependent case.
Measured on cerfon: ξ adaptation is **solver-invariant by construction**
(the bulk block never sees ξ; the core is exactly solved) — its payoff is
near-axis approximation accuracy, to be demonstrated with a manufactured-
solution study, not iteration counts. ζ-dependent axes (stellarators) need
per-ζ-plane ξ — a `PolarExtractionOperator` refactor, deferred.
