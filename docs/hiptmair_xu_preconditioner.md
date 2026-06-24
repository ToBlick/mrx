# Higher-form Hodge-Laplacian preconditioners (HX/AMS-style)

Consolidated reference for preconditioning the Hodge Laplacians `L_k` of the
FEEC B-spline de Rham complex `V0 -G0-> V1 -G1-> V2 -G2-> V3` (grad / curl / div)
on the polar (clamped × periodic × periodic) torus. It began as an HX/AMS
postmortem for `k=1`; a gradient-subspace projection then made an additive
`P_A + P_B` upper block beat Jacobi, and the same building blocks have since
been extended and measured across the whole family. The consolidated status and
building blocks are below; the detailed algebra, derivations, and experiment
history that follow are `k=1` (the first case made to beat Jacobi) but transfer
to the other degrees.

## Building blocks

Two ingredients, reused across degrees:

1. **Scalar tensor Hodge preconditioner** for the `k=0` Laplacian
   `L_0 = G_0^T M_1 G_0` (and its `k=3` dual). A CP-tensor / fast-diagonalization
   inverse with an outer surgery-Schur split for the polar axis. It is
   **near-exact and cheap** and is the workhorse "atom" that the vector-form
   corrections reach down to. Both Dirichlet (`dbc`) and free (`no-dbc`)
   variants are built.

2. **Additive auxiliary-space upper/Schur block** for the vector Laplacians,
   `P = (I-Π) P_A (I-Π)^T + P_B`, where
   - `P_A` is a tensor stiffness inverse (`block_fd`: per-component FD bulk +
     surgery Schur) for the curl-curl / div-div part,
   - `P_B = G X G^T` reaches down to the scalar atom on the gradient/exact
     subspace (for `k=1`, `X = L_0^{-1} M_0 L_0^{-1}`), and
   - `Π` is the subspace projection that confines `P_A` to the complement of
     `P_B`; without it the two interfere — the decisive ingredient for `k=1`.

Vector-form solves are saddle-point MINRES (the exact `M_{k-1}` sits in the
lower block, so `M_{k-1}^{-1}` is never forward-applied); `P` is the upper-block
(Schur) preconditioner. The scalar `k=0` solve is plain (deflated) CG. The
upper preconditioner acts on the **approximate Schur**
`Ŝ = S_k + D_{k-1} M̂_{k-1}^{-1} D_{k-1}^T` where `M̂^{-1}` is the mass
*preconditioner* (one apply), the same operator the Jacobi diagonal is probed
from — the true Schur's exact `M^{-1}` is never formed.

## Consolidated status (2026-06)

| degree | operator | best preconditioner | vs jacobi | status |
| --- | --- | --- | --- | --- |
| k=0 | grad-grad `L_0` | scalar tensor Hodge | ~10× iters, ~7× wall | works, both BCs, nullspace-robust |
| k=1 | grad-div `L_1` | projected `P_A + P_B`, **tensor-Cheb `L_0` atom** | **~2× wall, beats jacobi** | works, both BCs; with the production block_fd `P_A`, **raw == projected** (projection-free). P_B atom = degree-~7 Chebyshev on `L_0` (k=0 tensor smoother, κ≈6 **h-flat**) — matches exact `L_0⁺`; ~11× fewer iters than jacobi (49 vs 553 free), converges where jacobi stalls. See *Unified tensor-Chebyshev* below |
| k=2 | div-div `L_2` | projected `P_A(cap) + P_B`, **tensor-Cheb `L_1` atom (nested cheb-`L_0`)** | iters better but eroding; wall *slower* & widening | projection **mandatory** (raw fails). P_B atom = Chebyshev on `L_1`, smoother = k=1 Hodge precond whose inner `L_0⁻¹` is a near-exact nested cheb-`L_0`. **Degree near-flat** (κ 9.3→14.3→16.7, degree 5→6→7 over ns 6,12,4→8,16,6→9,18,7), but at the rough ε=0.1 atom **iterations grow** (130→208→221) faster than jacobi (462→590) so the iter advantage erodes (3.6×→2.7×); a more accurate atom (ε=0.01, deg 11) keeps iters flatter (109→123). Wall is **not competitive** and the gap widens (~5.6× slower at 9,18,7). See *Unified tensor-Chebyshev* below |
| k=3 | `L_3 = D_2 M_2⁻¹ D_2ᵀ` (no stiffness) | **jacobi** | jacobi wins | pure P_B (S_3=0): unified `P_3 = G_2 L_2⁻¹ M_2 L_2⁻¹ G_2ᵀ` with the k=2 preconditioner as inner `L_2⁻¹` **converges** (recursion closes, construction sound) but **loses to jacobi** decisively — 280 it / 10.4 s vs jacobi 186 it / 0.47 s. `L_3` is well-conditioned and tiny (n₃=192) so jacobi suffices; the nested-twice k=2 atom is far too expensive. Use jacobi. (Earlier sideways-transfer-to-k=0 route was rank-deficient.) |

> **The discrete derivative `G` is now fully matrix-free / inverse-free (2026-06).**
> grad `G_0` and curl `G_1` ship as explicit analytic `±1`/`−ξ` sparse stencils
> built straight from the incidence pattern + polar mapping coefficients `ξ` (no
> mass, no inverse, bit-exact to ≤2.2e-16 vs the `Gram⁻¹∘incidence` oracle, both
> BCs, forward+transpose); div `G_2` was already matrix-free (the V3 extraction is
> a 0/1 selection, `Gram₃=I`). The whole `inc_gram_inv` precompute machinery is
> removed — **no assembly inverse remains** in the derivative. See
> `docs/polar_true_derivative_G.md`.

> **Major correctness fix (2026-06): the polar discrete derivative `G`.** The
> directly-built incidence apply (`apply_incidence_matrix = Eᵀ sp E`) is **not**
> the true topological derivative on the *polar* sequence — the polar-axis
> extraction `E` is not a 0/1 selection (`EᵀE ≠ I`), so it omits the inverse in
> `G = M⁻¹D = (EᵀE)⁻¹ Eᵀ sp E`, and consequently `d∘d ≠ 0` near the axis
> (`curl·grad ≈ 1`, not 0). The existing `test_operators` nilpotency tests
> passed only because they run on a *non-polar* (identity-map) sequence where
> `E` is a 0/1 selection. `apply_incidence_matrix` is now fixed **in place** to
> apply the true `G` (a precomputed sparse correction, `G = Gram_{k+1}⁻¹ · Eᵀ sp E`
> with `Gram = EᵀE` built sparsely, only the small polar-axis block inverted; the
> apply is a single sparse matvec, no `todense`, no inverse at runtime; identical
> to the old apply on non-polar). This restores `d∘d = 0` to ~3e-16 on polar
> (both BCs) and is the change that makes the k=1 projector genuinely idempotent
> and unlocks k=2. New polar nilpotency test added; full `test_operators` passes.

## Unified tensor-Chebyshev preconditioner (2026-06-24)

The family is now unified under one **recursive** template for the `P_B` inner
inverse of `L_{k-1}`: a **fixed-degree Chebyshev iteration with a tensor
smoother**, fully matrix-free (no dense `pinv`, no inner Krylov). The recursion
bottoms out at the near-exact scalar `k=0` tensor Hodge preconditioner (κ≈6):

- **k=1** atom inverts `L_0`: Chebyshev on `L_0 = apply_stiffness(·,0)` (exact,
  matrix-free) with the k=0 tensor smoother.
- **k=2** atom inverts `L_1`: Chebyshev on the approximate Schur `Ŝ_1` with the
  k=1 Hodge preconditioner as smoother — and **that smoother's own inner
  `L_0⁻¹` is itself a nested near-exact cheb-`L_0`** (the key fix, below).

Three pieces make it work and keep it cheap:

1. **Auto-degree from Lanczos.** The Chebyshev degree is set at build time from
   the matrix-free Lanczos estimate of `κ = λmax/λmin` of the (deflated,
   preconditioned) operator, via the standard bound `d ≈ ½√κ · ln(2/ε)`. So the
   degree adapts per geometry/resolution; no hand-tuning. (`make_chebyshev_upper`
   guards a degenerate interval.)
2. **Constant-deflation of the nested free-BC `l_0`.** `apply_laplacian_
   preconditioner(·,0,tensor)` does **not** deflate the free `k=0` constant
   nullspace; left raw inside the k=1 smoother it over-amplifies near-constant
   `V0` modes that grad carries into the near-harmonic `V1` modes. M_0-orthogonally
   projecting the constant out was **the** fix: it dropped `cond(P_hodge·L_1)`
   from **152 → 9** and removed the near-null cluster. (Isolation confirmed the
   limiter was the rough inner `l_0`, *not* block_fd `P_A`: with an exact `l_0`,
   block_fd already gives cond 9 with zero near-null modes; exact `P_A` only
   sharpens 9 → 1.4.)
3. **Fusion — the projector and `P_B` share the `L_{k-1}` inverse.** The projected
   apply `(I-Π)P_A(I-Π) + P_B` is rewritten to **2 inner solves instead of 4**
   (algebraically identical, symmetry preserved): the dual-complement and `P_B`
   share `y = k1_inv(G₁ᵀr)`, and their outer solves combine into
   `pa + G₁·k1_inv(M₁y − G₁ᵀM₂·pa)`. The k=1 smoother is likewise the pre-existing
   fused form (2 `L_0⁻¹` instead of 4). Compounded ≈ 4× fewer nested cheb-`L_0`
   chains per k=2 iteration; wall 6888 → **4209 ms** at identical 130 iters.

**Projection is always on (a free simplification after fusion).** The fused
projected form does the *same* number of inner `L_{k-1}` solves as raw (the
projector shares its solve with `P_B`: 2 either way), so the projection adds no
expensive work; it is identical to raw for k=1 (68==68) and mandatory for k=2.
The unified preconditioner therefore always uses the projected form — no
raw/no-projection branch to choose. (Per-ε wall has a sweet spot: the degree
`d≈½√κ·ln(2/ε)` trades iterations against per-apply cost — e.g. k=2 ns 6,12,4
ε=1e-1→deg 5, 130 it, 4209 ms vs ε=1e-2→deg 9, 109 it, 6205 ms — so larger ε
(lower degree) is often the wall optimum; pick ε at the iteration/cost knee.)

**Results (RE, p=3, free BC):**

| | jacobi | unified projected | atom κ / degree | raw == projected? |
| --- | --- | --- | --- | --- |
| k=1 ns 6,12,4 | 551 it / 511 ms | 49 it / 204 ms | cheb-`L_0` κ≈6 / deg 7 | **yes** (projection-free) |
| k=1 ns 10,20,6 | 600 it *(stalls)* | 71 it / 534 ms | κ≈6.7 / deg 7 (h-flat) | yes |
| k=2 ns 6,12,4 | 462 it / 1265 ms | 130 it / **4209 ms** (fused) | cheb-`L_1` κ 9.3 / deg 5 | **no** (projection mandatory) |
| k=2 ns 8,16,6 | 584 it / 2522 ms | 208 it / 11.7 s | κ 14.3 / deg 6 | no |
| k=2 ns 9,18,7 | 590 it / 3119 ms | 221 it / 17.5 s | κ 16.7 / deg 7 | no |
| k=3 ns 6,12,4 | 186 it / 474 ms | 280 it / 10.4 s (converges) | nested k=2 atom | n/a (pure P_B, no projector) |

**Verdict.**
- **k=1 (and k=0): ship it.** The cheb-`L_0` atom matches the exact inverse,
  the degree is h-flat (~7), and with the production block_fd `P_A` the gradient
  **projection is unnecessary** (raw == projected). It beats jacobi ~2× on wall
  and converges where jacobi stalls — a clear improvement over the prior k=1
  preconditioner. Ship the constant-deflation of the `k=0` apply too (it is
  strictly correct and helps any nested free `L_0⁻¹`).
- **k=2: degree-scalable and matrix-free, but not wall-competitive.** Nesting
  the near-exact cheb-`L_0` made the atom's κ nearly h-flat (degree 5→6 over the
  range, vs the rough-`l_0`'s 21→42), so the construction *scales* in degree and
  iterations. Projection stays **mandatory** (raw fails at every resolution).
  After fusion the apply is as cheap as the construction allows, but the deep
  nesting (outer Chebyshev × k=1 smoother × inner cheb-`L_0`) leaves it ~3.3×
  *slower* than jacobi on wall time. It is an **iteration/robustness** tool, not
  a wall-time win at these sizes. A genuine wall win would need a cheaper
  near-exact `L_1` inverse than "polynomial over the k=1 preconditioner" — e.g. a
  multilevel vector-Laplacian solver.
- **k=3: use jacobi.** The unified pure-P_B form converges but loses badly
  (above); `L_3` is small and well-conditioned, so jacobi is the right tool.

**ε wall sweet-spot.** The degree `d≈½√κ·ln(2/ε)` trades outer iterations
against per-apply cost, so wall is U-shaped in ε. With the (then-fixed) near-exact
inner `l_0`, k=2 ns 6,12,4: ε=0.01→deg 9, 109 it, 6.2 s; **ε=0.1→deg 5, 130 it,
4.2 s (min)**; ε=0.3–0.5→deg 3 but iterations jump to 225 (4.5 s). So larger ε did
*not* win there — but the inner `l_0` was overkill at every ε. The inner-ε is now
**tied to the swept outer ε by default** (`l0_cheb_eps=None` → uses `atom_cheb_eps`),
so rougher outer ε also cheapens the inner atom; whether that shifts the wall
optimum to larger ε is being measured (consolidated `--all-k` run).

**Competing "Chebyshev-outside" preconditioner (`make_cheb_tensor_upper`).** Since
the outer saddle MINRES is itself a Krylov accelerator, a *simpler* upper-block
preconditioner is a Chebyshev directly on `Ŝ_k` whose only smoother is the cheap
tensor *stiffness* preconditioner — no HX projector, no nested P_B (one tensor
apply per Chebyshev step, very cheap). Expected caveat: the tensor stiffness
smoother is ≈0 on the down/gradient modes (models `S_k` only, not `D M⁻¹ Dᵀ`), the
exact gap P_B fills, so it may need huge degree / fail on those modes. Run
head-to-head against the HX form in the consolidated test (`--all-k`); results
pending.

## Tensor rank: why rank=1 is the default (2026-06-24)

**rank=1 is the production default for the stiffness/Hodge atom, and the only
viable rank for it.** Rank>1 makes the Laplacian preconditioners *dramatically
slower* — OOM-kill (rank 3 on rotating-ellipse), degree-31 GPU-compile hang
(rank 2 on axisymmetric). The mass path is the opposite (see below); the
pathology is specific to **stiffness/Hodge**.

**Why rank>1 hurts.** The k=0 Hodge apply
(`_apply_k0_tensor_hodge_bulk_shared_inverse`) uses **Lynch fast-diagonalization,
which is exact only at rank 1.** The per-axis eigenbasis `V_r/V_t/V_z` is built
from the rank-1 *leading* CP term; for rank>1 the apply keeps only the
**diagonal** of each extra term in that fixed basis and discards the off-diagonal
coupling. So a *better* CP field fit (`cp_rel_err` drops monotonically with rank)
yields a *worse* diagonal-truncated inverse. The deeper mechanism: the rank>1 fit
rotates a hyper-ill-conditioned bulk near-null eigenvector, corrupting the
core-Schur interaction — Frobenius accuracy of the separable operator is
irrelevant (a 3.4%-accurate rank-2 operator gives κ≈1e5 while the 5.9%-accurate
rank-1 gives κ≈2.9).

**The blow-up is exactly ONE isolated outlier, not a smear** (dense
`eig(smoother∘L_0)`, `scripts/debug/debug_rank_ritz_spectrum.py`, results in
`outputs/diag_ritz/2026-06-24/`, ns 6,12,4 p=3 free BC):

| geometry | rank | κ | λmin | gap below | active spectrum (λ₂ … λmax) |
| --- | --- | --- | --- | --- | --- |
| rotating-ellipse | 1, 2 | 6.19 | 0.271 | 1.95× | [0.529 … 1.68] |
| rotating-ellipse | **3** | **384** | **4.3e-3** | **124×** | [0.536 … 1.65] |
| toroid | 1 | 3.06 | 0.600 | 1.25× | [0.748 … 1.83] |
| toroid | **2, 3** | **51** | **3.0e-2** | **23.5×** | [0.709 … 1.54] |

The decisive observation is the rightmost column: **everything above the single
outlier is rank-invariant** — the active spectrum (hence the *effective* κ≈2–3)
is unchanged across rank. rank>1 leaves the operator excellent on the whole space
*except one direction*, which it drops into the basement. The threshold is
geometry-dependent (rot-ellipse breaks at rank 3, toroid at rank 2) but it is
always a *single* mode below a large (23–124×) gap. This mode is **not** the
operator's constant nullspace (the cheb-`L_0` atom already deflates that); it is a
preconditioner-induced near-null mode of the *composite* `smoother∘L_0`.

**Mass path is healthy** (and the contrast that localizes the bug): the `M_0`
Jacobian `R = 1 + εr cosθ` is genuinely rank-2, and the *mass* tensor
preconditioner *improves* with rank (`M_0` iters 4.8 → 3.5 → 3.0). Only the
stiffness/Hodge path has the Lynch-exactness problem.

**Refuted fixes (do not retry).** (1) denom positivity loss — false, the
pre-floor denom stays SPD across rank; (2) "CP-ALS is broken" — false, it fits
better with rank; (3) NTF / nonnegative joint CP — removed
the indefinite "monster" terms but still blew up; (4) **pseudo-inverse /
deflation of `M_bulk` *operator* modes** — κ never
recovers for any threshold, *and* zeroing modes breaks the good rank-1 case. The
outlier is a Schur-interaction mode of the composite, **not** an eigenvector of
the bulk operator, so you cannot deflate it out of `M_bulk`.

### Do we need to deflate the outlier, or can the outer solve absorb it?

It depends entirely on **what consumes the atom**, and the answer splits cleanly:

- **Direct consumption by an adaptive outer Krylov (PCG on the condensed k=0
  system, or the k=0 block inside the saddle MINRES): no deflation needed —
  ignore it.** A single isolated outlier behind a 23–124× gap is the textbook
  easy case for CG: it is resolved in the Krylov space in ~1 iteration and the
  rest of convergence follows the active κ≈2–3. The empirical proof is already in
  hand: **condensed k=0 CG goes 18 → 21 iters across the rank sweep** — +3
  iterations to absorb the rogue mode, no deflation. The *only* discipline
  required here is to **stop letting the raw κ drive anything**: the phantom κ's
  sole concrete cost is inflating the auto-degree `d ≈ ½√κ·ln(2/ε)` until it caps
  → OOM. Cap the degree / tune the interval on `[λ₂, λmax]` and the "problem"
  evaporates. The fix in this regime is *"don't trust raw κ,"* not deflation.

- **Consumption by the non-adaptive nested Chebyshev atom (the near-exact
  fixed-linear `L_{k-1}⁻¹` used inside k=1/k=2, where inner Krylov is
  forbidden): you cannot ignore it.** Chebyshev is a fixed polynomial for an
  interval `[lmin, lmax]`; *below* its interval the polynomial **grows**
  (`|p(λ)| → ∞`), so tuning on `[λ₂, λmax]` and ignoring the outlier makes the
  atom **amplify** that direction and inject garbage downstream (the documented
  "cheb on a singular operator amplifies the near-null → projected k=2 diverges"
  failure). There is no outer CG between the polynomial and the mode to catch it —
  the Chebyshev *is* the solver for that operator. So here you must deflate (or
  include the outlier in the interval and eat the OOM degree). And note there is
  **no cheaper-than-deflation middle option**: tuning honestly on `[λ₂, λmax]`
  already requires the Lanczos step that *finds* the outlier eigenpair, at which
  point deflating the vector you are already holding is essentially free.

**Recommendation.** Do **not** build deflated-Chebyshev machinery to enable
rank>1 in the nest. Either (a) keep the nested atoms at **rank-1** (the
production default — rank-1 has no outlier at all, so the question is moot exactly
where it would hurt), or (b) if rank>1 is ever wanted, restrict it to a slot
where an **adaptive outer Krylov consumes the atom directly** (the condensed k=0
CG), and there ignore the outlier for free (cap the auto-degree so the phantom κ
does not size the work). The 1-vector deflated Chebyshev is only worth
implementing if we acquire a concrete need for a rank>1 *near-exact fixed-linear*
`L_0⁻¹` — and we currently have no such need, since rank-1 is near-exact (κ≈6)
and the stiffness path shows no accuracy gain from rank>1 (unlike mass).

Surviving diagnostics for this thread: `scripts/benchmark/benchmark_unified_tensor_cheb_preconditioner.py`
(rank sweep across k) and `scripts/debug/debug_rank_ritz_spectrum.py` (the dense
outlier-spectrum probe behind the single-outlier diagnosis; raw spectra in
`outputs/diag_ritz/2026-06-24/`). The one-off probes that established the rest of
this section (CP-ALS mechanism, NTF, outlier-mode extraction, bulk pinv sweep,
ζ-first decoupling, capped-degree) were removed once their conclusions were
settled here — cheap to re-derive from the two survivors if needed.

## k=2 resolution (the true-G fix + capped `P_A`)

Pre-fix, every k=2 variant lost to jacobi (≈480 it) and the projected forms
*diverged* (res ≈ 0.5). The root cause was **not** a too-rough atom (the original
diagnosis) but the **wrong polar curl** in the projector/`P_B`: `apply_incidence`
was `Eᵀ sp E`, so the projector `Π₂ = G₁·atom·G₁ᵀM₂` sandwiched a *different*
operator than the atom inverted (`apply_stiffness(·,1) ≠ G₁ᵀM₂G₁` by `2.25e-3`),
and `Π₂` was never idempotent. With the true `G` (above), `apply_stiffness(·,1) =
G₁ᵀM₂G₁`, so the natural atom makes `Π₂` idempotent (`‖Π²−Π‖`: 0.088 → 2e-14) and
the construction converges. Dense diagnostics (`scripts/debug/debug_k2_dense_exact_atom.py`,
`rotating_ellipse`, free BC):

| upper precond (k=2) | iters | note |
| --- | --- | --- |
| jacobi (diag) | 226 | baseline |
| **projected `P_A(cap)+P_B(L_1)`** | **33** | best (capped exact div-div `P_A`) |
| projected `P_A(tensor)+P_B(L_1)` | 91 | rough tensor `P_A` |
| jacobi(whole) + `P_B(L_1)` | 146 | no projection, no `P_A` |
| raw `P_A(cap)+P_B(L_1)` | 1289 | converges (cap → bounded) but slow |
| raw `P_A(tensor)+P_B(L_1)` | **diverges** | uncapped div-div blows up on curls |

Three load-bearing facts:

- **`P_B` wants the full `L_1` inverse, not the bare curl-curl `K_1`.** The
  squared `P_B = G_1·atom·M_1·atom·G_1ᵀ` needs the `M_1`-metric inverse; `L_1⁺`
  gives cond≈9 (91 it) vs `K_1⁺` cond≈236 (251 it). (On `P_B`'s co-exact input
  the two are analytically equal, but the middle `M_1` of the squared form
  breaks Euclidean `K_1⁺`.) The exact-component sweep (2026-06-23) confirms this
  with an exact `P_A` and exact projector: whole-`L_1` **31/33** vs
  stiffness-`K_1`-only **162/242** it (dbc/free), 5–7×; `cond(pinv L_1)≈3.6e3`
  vs `cond(pinv K_1)≈1.6e18`. At k=1 whole-`L_0` ≡ stiffness(0) (identical it),
  so the whole-vs-stiffness split only bites at k≥2.
- **`P_A` must be bounded on curls.** The div-div tensor inverse is `1/λ` and
  blows up on the curl null (`S_2≈0` there), so raw diverges and the projection
  is mandatory. **Capping** it (`pinv(S_2, rcond)` — zero the curl modes, as
  k=1's block_fd `pinv` does on gradients) makes raw converge *and* sharpens the
  projected method (33 vs 91 it). Cap threshold is insensitive (1e-8 ≈ 1e-6).
- **Projection is mandatory at k=1 *and* k=2 — no structural asymmetry**
  (exact-component sweep, 2026-06-23). The raw−projected *operator* difference
  `‖P_A−(I-Π)P_A(I-Π)ᵀ‖/‖P_A‖` is large at every degree/BC (k=1 ≈1.2, k=2
  ≈1.9–2.4), with ~70° principal angles between `range(S_k)` (Euclidean
  co-exact) and the `M`-orthogonal complement `range(I-Π)` — the projection
  genuinely changes `P_A` everywhere. With an *exact* `P_A`, raw fails/blows up
  in all four cells (k1-dbc 457, else 600/fail) while projected gives
  9 / 11 / 31 / 33. The old "k=1 raw == projected (44 == 44)" was an artifact of
  the *approximate* block_fd `P_A`, whose imperfection happens to damp the leak
  (approx-`P_A` raw converges at k=1: 66/90, but still *fails* at k=2). The
  k=1↔k=2 difference is purely **quantitative**: the output leak
  `‖Π P_A‖/‖P_A‖` is 0.67 at k=1 vs 0.95–1.01 at k=2, so k=2 raw fails harder
  and tolerates no approximation.

Atom-conditioning hierarchy (the lever throughout): k=0 scalar tensor Hodge
`cond(P_0·L_0) ≈ 2.3 (dbc) / 6.2 (free)` — near-exact; the cheap vector
curl-curl/`L_1` atoms are `cond ≈ 60` — the gap that distinguishes the
near-effortless k=0/k=1 from k=2. Open production item: a *cheap* `L_1` curl-aux
atom near the exact one's cond≈9 (the 33–91 it results use exact dense atoms).

Headline `k=1` benchmark (`ns=(6,12,4)`, `p=3`, toroid, `ε=1/3`; scalar bulk +
fused projector, dense couplings on):

| upper precond | iters | wall |
| --- | --- | --- |
| `jacobi (diag)` | ~386 | ~266 ms |
| projected `P_A + P_B` (`P.T P_S P + P_B`) | **~96** | **~115 ms** |

Diagnostic harness: `scripts/benchmark/benchmark_graddiv_k1_preconditioner.py`. Form
degree is selected by `--klevel {0,1,2,3}` (`0` = the `k=0` nullspace test, `1` =
the default `k=1` path, `2`/`3` run their dedicated benchmarks). `--k1-both-bc`
runs the `k=1` dbc-vs-free comparison. The consolidated new-result sections
(nullspace robustness; Richardson/Chebyshev baselines; `k=2`; `k=3`) follow
immediately; the detailed `k=1` algebra and history are kept below as reference.

## Session 2026-06-21: matrix-free `G`, cheap-atom attempts, and h-scaling

This session (a) finished the matrix-free `G`, (b) chased a *cheap* near-exact
`L_1` atom so the k=2 projected method needs no dense `pinv`, and (c) ran the
h-scaling tests that reframe what "works" means. **Net: the projected k=2
construction is sound and ~h-flat *with* a near-exact atom; the one missing
production piece is a cheap, scalable near-exact `L_1` atom.**

### Matrix-free `G` complete
grad `G_0` and curl `G_1` are now explicit analytic inverse-free sparse stencils
(`build_grad_stencil_g0`, `build_curl_stencil_g1` in `mrx/operators.py`),
bit-exact (≤2.2e-16) vs the `Gram⁻¹∘incidence` oracle on the polar
`rotating_ellipse`, both BCs, forward+transpose; `div∘curl`/`curl∘grad`
nilpotency ~1e-16; `pytest test/test_operators.py` 56 passed. div `G_2` was
already matrix-free (V3 unitary). The `inc_gram_inv` precompute + its small
axis-block inverse are deleted; polar is detected Gram-free via a one-probe
`_extraction_is_polar`. Validated bit-exact against the `Gram⁻¹∘incidence`
oracle; the polar nilpotency regression ships in `test/test_operators.py`.

### The cheap-atom attempt: Chebyshev with the tensor smoother
The right way to turn the cond≈28 k=1 tensor preconditioner into a near-exact
`L_1⁻¹` is a **fixed-degree Chebyshev iteration with that preconditioner as the
inner smoother** (not jacobi). Interval auto-tuned by an A-inner-product Lanczos
(`_lanczos_extremal_eigs_precond`). On the **k=1 saddle (dbc, nonsingular)** this
works cleanly: `cheb-tensor-{1,2,3,5,8,10}` → `{80,66,52,32,23,19}` iters (vs
plain 88), all 0/4, ~3× better per degree than the jacobi smoother. So a cheap
near-exact `L_1⁻¹` *exists* for the well-conditioned case.

**But it fails as the k=2 embedded atom**, for a now-understood reason: the k=2
atom runs on **free-BC `L_1`, which is singular** (the `b₁` cohomology harmonic),
and the k=2 *projector* gives it no outer deflation (unlike the k=1 saddle, which
deflates). Chebyshev's polynomial `p(λ)≈1/λ` *blows up at the `λ≈0` harmonic*, so
acceleration is actively harmful in the projector (worse than the un-accelerated
`full_l1`, which merely *converges slowly*). The independent
hyperparameter-estimation diagnostic pinned the mechanism: Lanczos estimates `λmax` perfectly and `λmin` well, but the
**defensive interval floor `lmin = max(lmin, lmax·1e-3)`** clips 14 genuine small
modes on the free-BC operator (cond 1834 there) — Chebyshev then under-resolves /
amplifies exactly those near-null modes. Explicit harmonic **deflation** of the
atom removed the crash path but the deflated form still NaN'd in the squared
`P_B`; deprioritized in favor of the route below.

### Capping `P_A` like k=1 (route b) — converges but does **not** scale
"Handle k=2 `P_A` like k=1's": the cap primitive already exists
(`_symmetric_pseudoinverse(M, relative_tol)`). Mimicked k=1's block_fd for k=2
(`_build_k2_block_fd_preconditioner`: tensor bulk + re-probed **PSD-pinv-capped**
surgery Schur). This bounds the div-div `P_A` on the curl null, and **raw
`P_A(capped) + P_B`, no projector, converges** (824 it, 0/4) where uncapped raw
diverged. With a *cheap* atom the projection now *hurts* (1229 > 824) — confirming
`Π₂` needs a near-exact atom. **However the h-test refutes raw-capped as a
scalable solution:**

| ns (n2) | jacobi | raw `P_A(capped)+P_B` |
| --- | --- | --- |
| 6,12,4 (584) | 226 | 824 |
| 8,16,6 (1740) | 343 | 1825 |
| 10,20,6 (2892) | 463 | **2000 diverges** |

It grows faster than jacobi and fails at the largest grid. Two structural reasons:
(1) **raw mode is inherently slower than projected even with a perfect atom**
(dense, low res: projected+`L_1⁺` = 33 but raw+`L_1⁺` = 1289) because the
projection removes `P_A`'s leak into the exact subspace — and this is needed at
k=1 too: the exact-component sweep shows exact-`P_A` raw blows up at k=1 as well,
so the apparent k=1 "raw == projected" was an approximate-`P_A` artifact (see the
asymmetry bullet above); (2) the cheap/capped `P_A` is not h-flat.

### h-scaling reframes the goal: atom quality is the lever
| construction | ns 6,12,4 | ns 8,16,6 | ns 10,20,6 | trend |
| --- | --- | --- | --- | --- |
| k=2 jacobi | 226 | 343 | 463 | grows ~2× |
| **k=2 projected `P_A(cap)+P_B(L_1⁺ exact)`** | **33** | **34** | — | **FLAT (h-independent); ~10× jacobi, gap grows** |
| k=2 projected `P_A(tensor)+P_B(L_1⁺ exact)` | 91 | 107 | — | mild growth (uncapped `P_A`) |
| k=2 raw `P_A(cap)+P_B(L_1⁺ exact)` | 1289 | 600\* | diverges | raw is the wrong mode even with exact components (\*capped maxiter) |
| k=2 raw `P_A(capped)+P_B(full_l1 cheap)` | 824 | 1825 | diverges | worse than jacobi |
| k=1 projected `P_A+P_B` (tensor `L_0` atom, cheap) | 88 | 139 | 218 | grows; advantage erodes |

The clean conclusion: **the projected k=2 construction is h-INDEPENDENT when its
two components are good — capped `P_A` *and* a near-exact atom give 33→34 (flat),
10× fewer iters than jacobi with the gap growing.** Capping `P_A` improves
*h-scaling*, not just convergence (uncapped 91→107 vs capped 33→34); raw fails
even with exact components (the blocks don't separate); and the cheap tensor atoms
grow with h. So the construction is *right*, and the **production gap is exactly two
cheap/matrix-free pieces, both known to help scaling: (1) a capped tensor `P_A`**
(the `_symmetric_pseudoinverse` cap primitive exists — apply it to the modal
inverse) **and (2) a cheap near-exact `L_1` atom** (Chebyshev-with-tensor-smoother
is the candidate, once the singular-free-`L_1` deflation is robust in the squared
`P_B`). Both are currently dense `pinv` (analysis-only).

### Scalability audit (apply paths are O(N); one setup-time concern)
All **per-apply** dense work in the k=0/1/2 preconditioners is ≤ the polar-axis
surgery block `S` (`schur_inv @ ·`) plus per-axis fast-diagonalization einsums
(`d`); nothing per-iteration scales with the full 3D DoF count `N`. The one
production dense op larger than `S` is **setup-time only**: the CP-ALS metric-fit
SVDs (`preconditioners.py:712-714`), which factor mode-unfoldings of the full-3D
quadrature metric (PLANE-sized, `O(N·direction)`, one-time, thin). Guardrails:
keep tensor `rank ≤ 2` (rank≥3 hits an `N×N` dense-surrogate fallback) and keep
the `.todense()` dense-reference builders off the hot path (they are diagnostic
only). To make *setup* fully scale, replace the three CP-init SVDs with a
randomized / per-direction-reduced fit.

## Outcome

Status update (2026-06): for the `k=1` grad-div diagnostic, an additive
`P_A + P_B` upper block **does** beat the Schur-outer Jacobi baseline once
`P_A` is restricted to the curl-dominated complement by a gradient-subspace
projection (see "Gradient-subspace projection" below). On the standard `k=1`
benchmark (`ns=(6,12,4)`, `p=3`, toroid, `ε=1/3`):

| upper precond | iters | wall time |
| --- | --- | --- |
| `jacobi (diag)` | ~386 | ~290 ms |
| `P_A + P_B` (raw, no projection) | ~347 | ~787 ms |
| `P_A + P_B` (gradient-projected) | ~96 | ~267 ms |
| `P_A + P_B` (gradient-projected, vector-FD true-basis stabilized) | **~84** | **~230 ms** |

So the projected variant cuts iterations ~4× and is faster in wall time than
Jacobi; the current best experimental variant (true-basis vector-FD with
stabilized per-mode fallback) improves this further to ~84 it / ~230 ms.
Adding any Jacobi on top (`α·jacobi + P_A + P_B`) still hurts (`α=0` is best),
because the projection already removes what Jacobi was patching.

At slightly higher resolutions the picture changes:

- `ns=(7,14,5)`: true-basis vector-FD remains better than projected baseline
  block_fd (about `98.5` it / `388 ms` vs `115.8` it / `456 ms`) but still
  loses wall time to Jacobi (`~304 ms`).
- `ns=(7,14,8)`: true-basis vector-FD remains much better than projected
  baseline block_fd (`~124.8` it / `660.8 ms` vs baseline `~116.5` it /
  `463.3 ms` at `7,14,5`; baseline `7,14,8` run still pending), and also loses
  wall time to Jacobi (`~493.8 ms`).

So the new vector-coupled model is a clear iteration-quality win, but not yet a
scalable wall-time winner at the larger tested grids.

This reverses the earlier postmortem conclusion. The blocker was never
indefiniteness or Schur tuning; it was that the raw `P_A` and `P_B` were both
acting on the gradient subspace and interfering. `k=2` and `k=3` have since been
retried (see the consolidated sections up top): the same auxiliary-space idea
does **not** yet net-win there, because the scalar atom that makes `k=1` work
(`L_0^{-1}`, near-exact) has no equally-cheap counterpart for the `k=2` curl
block (`K_1^{-1}` is only a rough one-shot apply) or the `k=3` transfer (the
`V0↔V3` map is rank-deficient by basis order).

## What Was Consolidated

This file is the single markdown reference for the higher-form Hodge-Laplacian
preconditioners: the scalar `k=0` tensor Hodge inverse, the `k=1` projected
`P_A + P_B` upper block, the `k=2`/`k=3` auxiliary-space attempts, the
Jacobi / Richardson / Chebyshev baselines, and nullspace handling for free BCs.
Related high-level docs treat HX as historical rather than active roadmap.

## Canonical Debug Script

Use exactly one diagnostic script:
`scripts/benchmark/benchmark_graddiv_k1_preconditioner.py`, driven by
`slurm/job_diag_graddiv_pa_compare.sh` (env-var overrides → SLURM). Form degree
via `--klevel`; the `k=3` duality-transfer prototype now lives inside this
script (`--klevel 3`) rather than as a separate entry point.

## Nullspace robustness (k=0 and k=1, both BCs)

The tensor preconditioners remain effective when the operator is **singular**
(free / no-dbc BCs, with the harmonic deflated in the Krylov solve). Confirmed
on `rotating_ellipse` (`nfp=3`, `κ=1.2`, `ns=(8,16,6)` for `k=0`; `toroid`,
`ns=(6,12,4)` for `k=1`):

| degree / BC | nullspace | jacobi | tensor |
| --- | --- | --- | --- |
| k=0 dbc | 0 | 156 it | **16 it** |
| k=0 free | 1 (constant) | 222 it | **21 it** |
| k=1 dbc | 0 | 386 it | **96 it** |
| k=1 free | 1 (harmonic) | 546 it | **168 it** |

Notes:
- The no-dbc paths were never actually broken — they had simply never been
  exercised on correctly-sized input. `k=0` free has `n0 = 594` (`= 18` surgery
  `+ 6·16·6` bulk) vs `n0_dbc = 498`.
- Free is intrinsically harder than dbc (larger space + Neumann-type spectrum),
  but the tensor preconditioner keeps its ~4–10× iteration edge at every BC.
- Saddle nullspace handling: `solve_saddle_point_minres` deflates the harmonic
  via `vs_upper` / `mass_upper_matvec`; the `k=1` free harmonic count is
  `b_1 = 1`. Run with `--k1-both-bc` (or `--klevel 0` for the scalar test).
- **Timing caveat:** `apply_laplacian_preconditioner(kind="jacobi")` re-probes
  `diag(L_k)` (an `O(n)` stiffness sweep) on every call, which gets traced into
  the CG loop and inflates Jacobi wall times by orders of magnitude. For fair
  timing always **precompute the diagonal once** (the saddle benchmarks already
  use the stored `schur_diaginv`).

## Baselines: Richardson and Chebyshev on the approximate Schur

As a "is the tensor method worth it?" check, the Schur (upper) preconditioner
can instead be a fixed-degree Richardson or Chebyshev iteration on the
**approximate Schur** `Ŝ = apply_laplacian_approx(·,k)` (the `M→M̂` operator),
with the Jacobi diagonal as the inner smoother and the `[λmin, λmax]` of the
Jacobi-preconditioned `Ŝ` estimated once by Lanczos. Fixed degree ⇒ a linear,
symmetric operator ⇒ a valid MINRES preconditioner (no Krylov-in-Krylov).

`k=1` toroid `ns=(6,12,4)`, scalar bulk, 4 RHS:

| upper precond | iters | wall |
| --- | --- | --- |
| jacobi | 386 | 266 ms |
| **tensor `P.T P_S P + P_B`** | **96** | **115 ms** |
| richardson-{2,3,5,8} | 218 / 205 / 157 / 114 | 393 / 504 / 594 / 656 ms |
| chebyshev-{2,3,5,8} | 238 / 168 / 106 / **67** | 321 / 338 / 355 / 357 ms |

Conclusions:
- **On iterations**, Chebyshev catches and passes the tensor method
  (`chebyshev-8` = 67 < 96; `chebyshev-5` ≈ tensor). So the tensor method is
  *not* fundamentally better-conditioning than a good polynomial smoother on `Ŝ`.
- **On wall time**, the tensor method wins ~3×: its apply costs ~one `Ŝ`, while
  Chebyshev-`d` costs `d` (expensive) `Ŝ` applies. The tensor preconditioner's
  value is its **cheap per-apply cost**, not a lower iteration count.
- Richardson < Chebyshev at every degree (Chebyshev is the optimal polynomial);
  low-degree Chebyshev shows near-tolerance fails (non-monotone polynomial +
  imperfect `[λmin, λmax]`), degree ≥ 5 converges cleanly.

## k=2 (div-div) — open problem

> **Historical (pre-true-`G`).** This section recorded the diagnosis *before* the
> polar-`G` fix, when every k=2 variant lost to jacobi and the projected forms
> diverged. That was a **wrong-polar-curl** artifact, not a real atom-quality wall:
> with the true `G` the projected method converges and beats jacobi, and is
> ~h-flat with a near-exact atom. See "k=2 resolution (the true-G fix…)" and
> "Session 2026-06-21" above for the current state. The text below is kept for the
> record (the "rough atom" framing here was superseded — the real walls turned out
> to be the polar curl, then atom *h-scaling*).

The `k=2` Hodge Laplacian `L_2 = S_2 + D_1 M_1^{-1} D_1^T` is the degree-shifted
analog of `k=1`: div-div stiffness `S_2 = G_2^T M_3 G_2` (singular on curls
`ran(G_1)`) plus a curl-handling term. The natural transfer of the `k=1`
construction is `P_A` = div-div tensor inverse in a **curl**-complement
sandwich, plus `P_B = G_1 K_1^{-1} M_1 K_1^{-1} G_1^T` (curl-subspace
correction), where the inner atom is now `K_1^{-1}` (the 1-form curl-curl
inverse) instead of the scalar `L_0^{-1}`.

This does **not** beat Jacobi (`rotating_ellipse nfp=3`, `ns=(8,16,6)`,
jacobi = 480 it):

- Projected `P.T P_S P + P_B` and all 2×2 projection-ablation corners **fail**
  (stall / diverge); every projection added makes it *worse* (monotone).
- Additive forms (`jacobi + P_B`, clean-split `jacobi(S) + P_B`) converge but at
  ~2000 it (4× *slower* than jacobi).

Root cause: `k=1` works because its atom `L_0^{-1}` is near-exact and cheap, so
the projectors built from it are clean and the `P_B` double-inverse is
well-scaled. The `k=2` atom `K_1^{-1}` is only a rough `block_fd` one-shot apply
— fatal both as a *projector ingredient* (`Π_2 = G_1 K_1^{-1} G_1^T M_2` is then
not idempotent and injects error) and as an *additively-scaled block* (wrong
magnitude relative to Jacobi). An accurate `K_1^{-1}` would require an inner
Krylov solve, which is disallowed. This is the active thread.

## k=3 — auxiliary-space "sideways" transfer

`L_3 = D_2 M_2^{-1} D_2^T` (no stiffness; saddle with a zero upper block). Its
dual is `k=0`, so a natural preconditioner maps a `V3` residual **sideways** to
the scalar `k=0` space — *not* via the derivative (which has a huge kernel) but
via the metric-free **cross-mass** `V0↔V3` (the stored `p03`/`p30` "projection"
blocks; the complementary-degree `L²` pairing is metric-independent — the Piola
Jacobians cancel), realized as the Galerkin transfer `T = M_3^{-1} C` with
`M_3^{-1}` the tensor mass preconditioner — then inverts with the near-exact
`k=0` Hodge preconditioner and maps back. BC swap: run `k=3` free so its dual is
`k=0` dbc (the working tensor `L_0^{-1}`).

Result (`rotating_ellipse nfp=3`, `ns=(8,16,6)`, jacobi = 240 it):
`P_3 = T L_0^{-1} T^T` is a near-exact inverse **on its range** (62 it, ~4×
fewer than jacobi) but **hard-stalls** because `T: V0(498) → V3(576)` is
**rank-deficient** — `k=0` (degree `p`) and `k=3` (degree `p-1`) are
different-order scalar spaces, so the transfer can never be surjective, leaving
a basis-order complement unpreconditioned. Adding a smoother (`jacobi + transfer`,
and an `α·jacobi + transfer` sweep) converges but never beats jacobi: the
bottleneck shifts to that complement, which only jacobi covers, at its native
rate. So the sideways transfer is correct and powerful on its range but, in the
additive form, does not net-win for `k=3`.

## Retained Algebra (Useful Reference)

Even though HX/AMS is archived as a production direction, the algebra used in
the diagnostics is still useful.

For the higher-form upper block, we used the additive split

$$
P = P_A + P_B,
$$

with:

- `P_A` — the **native-space stiffness inverse** (`block_fd`: a per-component
  fast-diagonalization bulk inverse plus an exact surgery Schur for the polar
  axis), approximating the curl-curl (`k=1`) / div-div (`k=2`) part.
- `P_B` — the **auxiliary-space (exact/gradient-subspace) correction**
  `P_B = G X G^T`, mapping the residual down to the scalar `k=0` space, inverting
  there, and mapping back. `X` is built from the near-exact scalar `L_0^{-1}`.

In production `P_A` is restricted to the complement of `P_B` by the
gradient-subspace projection `Π` (below), i.e. `P = (I-Π) P_A (I-Π)^T + P_B`.

For the `k=1` grad-div diagnostic path,

$$
L_1 \approx K_1 + D_0 M_0^{-1} D_0^T,
$$

and

$$
P_B = G_0 L_0^{-1} M_0 L_0^{-1} G_0^T.
$$

This is the key practical point: applying $P_B$ uses **two scalar Laplacian
inverse applies** (`L_0^{-1}` appears twice), with one scalar mass apply in
between.

In operator pipeline form (input residual $r \in V_1^*$):

$$
\begin{aligned}
y_1 &= G_0^T r, \\
y_2 &= L_0^{-1} y_1, \\
y_3 &= M_0 y_2, \\
y_4 &= L_0^{-1} y_3, \\
u   &= G_0 y_4.
\end{aligned}
$$

So the "two scalar Laplacians" statement is literal for this construction:
the correction cost is dominated by two scalar `k=0` inverse-Laplacian
preconditioner applies.

### Derivation Sketch (Why This Form)

For `k=1`, the difficult component is the exact/gradient subspace generated by
`G_0`. If a 1-form unknown is written as

$$
u = G_0 \phi,
$$

then the grad-div-like contribution acts through scalar operators on `\phi`.
An auxiliary-space correction therefore maps 1-form residuals down to scalar
space with `G_0^T`, applies scalar inverse operators there, and maps back with
`G_0`.

That motivates a generic structure

$$
P_B = G_0 X G_0^T,
$$

where `X` is a scalar-space operator approximating the inverse action needed on
the exact component.

In the diagnostic construction we used

$$
X = L_0^{-1} M_0 L_0^{-1},
$$

which gives

$$
P_B = G_0 L_0^{-1} M_0 L_0^{-1} G_0^T.
$$

So the two `L_0^{-1}` factors appear because the scalar-space core is a
left-right inverse pairing around `M_0`. Operationally: pull residual to
scalar dual, invert once, apply scalar mass, invert again, push back to
1-form space.

## Technical Summary Of Why It Stalled (Historical) — And How It Was Unblocked

The gradient/exact-subspace correction idea (`P_B`) is useful in isolation. The
original postmortem concluded that the missing piece was *"an effective, cheap,
and stable native-space correction (`P_A`)"* and that the gap was wall-time
efficiency.

That conclusion was incomplete. The `block_fd` `P_A` was already a serviceable
curl-subspace preconditioner; the real blocker was that **raw `P_A` and `P_B`
both acted on the gradient subspace and interfered additively**. Restricting
`P_A` to the curl complement with the gradient-subspace projection (above)
unblocked it: projected `P_A + P_B` now beats Jacobi on iterations and wall
time for `k=1`. The lesson is that subspace *complementarity*, not more bulk
accuracy or Schur tuning, was the decisive factor.

## Current `P_A` Path In The Debug Script

The canonical script is `scripts/benchmark/benchmark_graddiv_k1_preconditioner.py`.
Based on the current benchmark sweep, non-performing `P_A` choices were removed
from the script interface. The retained `P_A` path is `--pa-mode block_fd`.

`block_fd` targets the extracted `k=1` curl-curl stiffness `K_1` and wraps its
bulk apply in a surgery Schur complement model. It can run with either:

- diagonal bulk blocks (`--pa-block-inner-schur` off, default), or
- inner bulk Schur coupling (`--pa-block-inner-schur` on).

### Shared Outer Structure: Surgery Schur

Both prototypes use the same surgery/bulk split: extracted `k=1` DOFs are
partitioned into *surgery* rows `S` (polar axis + periodicity identifications)
and a large *bulk* block `B` on a tensor-product index grid. `mode3x3_fd`
builds this split directly with `_build_k1_stiffness_surgery_factors`; `block_fd`
reuses the already assembled split from `ops.k1_tensor_stiff_precond`. Writing

$$
K_1 =
\begin{pmatrix} K_{SS} & K_{SB} \\ K_{BS} & K_{BB} \end{pmatrix},
$$

the preconditioner uses the exact block factorization

$$
K_1^{-1} =
\begin{pmatrix} I & 0 \\ -\tilde K_{BB}^{-1} K_{BS} & I \end{pmatrix}
\begin{pmatrix} \Sigma^{-1} & 0 \\ 0 & \tilde K_{BB}^{-1} \end{pmatrix}
\begin{pmatrix} I & -K_{SB}\tilde K_{BB}^{-1} \\ 0 & I \end{pmatrix},
\qquad
\Sigma = K_{SS} - K_{SB}\,\tilde K_{BB}^{-1}\,K_{BS}.
$$

The surgery Schur `Σ` is small (its size is mesh-independent), so it is formed
densely by probing the chosen bulk surrogate `K̃_BB⁻¹`, then inverted with PSD
clipping: eigenvalues below a relative cutoff (and any non-positive ones) are
deflated to zero. This guarantees the surgery contribution stays SPD even when
the bulk surrogate is only approximate.

### Retained Path — `--pa-mode block_fd`: Production Tensor Bulk Inverse

#### Step 1 — Surgery/bulk split

The extracted `k=1` DOF vector is split into two disjoint index sets before any
solve begins. This split is taken from the already-assembled tensor stiffness
payload `ops.k1_tensor_stiff_precond`, which stores it as
`payload.surgery.surgery_indices` and `payload.surgery.bulk_indices`.

- **Surgery indices** (`S`): the polar-axis rows and periodicity-identification
  rows. These form a small, mesh-resolution-independent set. Their coupling
  structure is irregular (not tensor-product) and must be handled separately.
- **Bulk indices** (`B`): all remaining DOFs. Their structure is a true
  tensor-product grid over `(r, θ, ζ)` mode indices, which makes fast
  diagonalization applicable.

#### Step 2 — Bulk surrogate inverse $\tilde K_{BB}^{-1}$

`block_fd` does **not** build a new bulk inverse. It directly calls the
already-assembled production tensor stiffness factors, via either:

- `_apply_k1_bulk_diagonal_preconditioner` (default, `--pa-block-inner-schur`
  off): the three vector components `r`, `θ_bulk`, `ζ_bulk` are each inverted
  **independently** by their own fast-diagonalization (FD) block.
- `_apply_k1_bulk_preconditioner` (`--pa-block-inner-schur` on): an additional
  inner Schur sweep couples the `r`/`θ_bulk` block against `ζ_bulk`.

**How a single FD block inverse works** (each of the three components):

Each component's bulk block is approximated as a sum of separable Kronecker
terms. Specifically, `assemble_tensor_stiffness_preconditioner` (in
`mrx/operators.py`) CP-decomposes the diagonal metric weight fields (`J g^{rr}`,
`J g^{θθ}`, `J g^{ζζ}`) and builds per-component bulk factors using
`_build_mass_referenced_tensor_block_factors` (in `mrx/preconditioners.py`).
That builder does the following at assembly time:

1. For each of the three axes `(r, θ, ζ)`, simultaneously diagonalize the
   reference mass matrix `M_axis` and the operator contribution `A_axis`
   (stiffness or identity depending on which axis is "active" for this
   component). This yields per-axis eigenvectors `fd_V_axis` and eigenvalues
   `fd_lam_axis` satisfying `V^T M V = I` and `V^T A V = diag(lam)`.
2. Form the modal denominator tensor as the sum over CP rank terms:

   $$
   D_{ijk} = \sum_{\text{terms}} d^r_i \cdot d^t_j \cdot d^z_k,
   \quad d^{axis}_i = \text{diag}(V_{axis}^T A_{axis} V_{axis})
   $$

   and store the precomputed elementwise reciprocal `fd_inv_denom = 1 / D`.

At **apply time** `_apply_tensor_diagonal_block_preconditioner` does:

$$
x = V_r \left( V_\theta \left( V_\zeta \cdot
    \frac{1}{D} \cdot V_\zeta^T \right) V_\theta^T \right) V_r^T \, b,
$$

implemented as six sequential `jnp.einsum` calls (forward transform on each
axis, pointwise divide, inverse transform on each axis). This is the entire
bulk cost per component: three matrix-vector products in 1D plus a pointwise
scale.

#### Step 3 — Schur complement for the surgery rows

With $\tilde K_{BB}^{-1}$ in hand, the surgery Schur complement

$$
\Sigma = K_{SS} - K_{SB}\,\tilde K_{BB}^{-1}\,K_{BS}
$$

is formed **densely** by probing:  for each column $e_i$ of the identity on the
surgery block, evaluate $K_{SS} e_i - K_{SB}(\tilde K_{BB}^{-1}(K_{BS} e_i))$.
The resulting small dense matrix is symmetrised and then pseudoinverted with PSD
clipping: eigenvalues below `pinv_rtol × max_eigenvalue` (and any negative ones)
are deflated to zero. The stored `schur_inv` is the resulting
positive-semidefinite pseudoinverse.

This is the step implemented in `_build_k1_block_fd_preconditioner` in the
script (not in library code).

#### Step 4 — Full preconditioner apply

Given an input vector $v$ partitioned into surgery part $v_S$ and bulk part
$v_B$, the apply computes:

$$
y = \tilde K_{BB}^{-1} v_B
$$
$$
z = \Sigma^{-1} (v_S - K_{SB} y)
$$
$$
x_B = y - \tilde K_{BB}^{-1}(K_{BS} z)
$$

and scatters $(z, x_B)$ back to the full index layout. This costs exactly
**two** calls to the bulk surrogate inverse per preconditioner apply (step 1
and the second use in step 3). The Schur solve $\Sigma^{-1}$ is a small dense
matrix-vector multiply (surgery size is mesh-independent).

#### What the approximation misses

`block_fd` is a stack of approximations to $K_1^{-1}$. From innermost (per-axis)
outward, the deliberate simplifications are:

1. **Separable-metric (CP) bulk surrogate.** The metric weight fields
   $J g^{rr}, J g^{\theta\theta}, J g^{\zeta\zeta}$ are non-separable; they are
   replaced by a finite CP sum of separable products, and only that surrogate is
   inverted. Empirically rank 1 ≈ rank 3, so the CP residual is **not** the
   dominant error for this geometry.
2. **Off-diagonal metric dropped.** Only the three diagonal weights enter; the
   non-orthogonal-map terms $g^{r\theta}, g^{r\zeta}, g^{\theta\zeta}$ are absent.
3. **Cross-component (curl) coupling dropped.** The default bulk inverse treats
   the three vector components as independent scalar problems — the single
   largest structural approximation. The optional inner Schur recovers only
   `r/θ_bulk ↔ ζ_bulk` and is far too slow (~13× apply cost) to compete.
4. **Surgery Schur inherits the bulk error**, and is PSD-clipped: near-null
   surgery modes are deflated to zero (required for MINRES), not solved.
5. **The gradient nullspace is not handled by $P_A$ at all.** $K_1$ is singular
   on gradients, so the FD modal denominators for gradient modes are near zero
   and raw `block_fd` acts with large, uncontrolled gain there. This is *by
   design* delegated to $P_B$, and is exactly why the gradient-subspace
   projection below is needed (see "Why the low modes fail").

In short: `block_fd` is a good preconditioner for the
**curl-dominated, component-decoupled, separable-metric** part of $K_1$, and
defers the gradient subspace (to $P_B$), the cross-component coupling, and the
off-diagonal metric (to nothing).

Properties observed (rank 1 and rank 3):

- `sym(P_A)` is effectively PSD (min eigenvalue at roundoff).
- Raw `P_A` alone does not converge the saddle solve.
- CP rank 1 ≈ rank 3 (the FD approximation saturates at rank 1).
- Inner bulk Schur is not competitive (large runtime increase).
- The `P_A S` spectrum has a large exact zero block (the gradient nullspace), so
  `cond_abs` is meaningless; the curl-restricted `cond_curl` is the meaningful
  quality metric and is flat across rank.

### Gradient-subspace projection (the change that beats Jacobi)

Raw `block_fd` acts with large, uncontrolled gain on the gradient (curl-free)
subspace, where $K_1$ is singular and $P_B$ is responsible. The exact-projector
diagnostic quantifies this: **83% of raw `P_A`'s output energy lives in the
gradient subspace** (`raw_grad_frac_exact mean=0.83`). So raw `P_A` is not a
curl preconditioner that merely grazes the gradient modes — it is *dominated*
by gradient output. Adding `P_A + P_B` therefore swamps `P_B`'s careful
gradient correction with a large, meaningless gradient vector. The fix is to
restrict `P_A` to the $M_1$-orthogonal complement of the gradient subspace with
a symmetric projection sandwich (flag `--pa-grad-project`):

$$
\tilde P_A = (I - \Pi)\,P_A\,(I - \Pi^\ast),
\qquad
\Pi = G_0\,L_0^{-1}\,G_0^T M_1,
\quad
\Pi^\ast = M_1 G_0\,L_0^{-1}\,G_0^T,
$$

where $L_0 = G_0^T M_1 G_0$ is the scalar gradient-space operator and
$L_0^{-1}$ is approximated, matrix-free, by the same rank-1 tensor `k=0`
Laplacian preconditioner used inside $P_B$. $\Pi$ maps $V_1 \to V_1$ (it acts
on `P_A`'s primal output); $\Pi^\ast$ maps $V_1^\ast \to V_1^\ast$ (it acts on
the dual residual input). The two are adjoints, so the sandwich is symmetric
and the composite keeps `P_A`'s signature $V_1^\ast \to V_1$. Both arms reuse
existing applies (two incidence applies, one `L_0^{-1}`, one $M_1$), so the
projection adds only modest cost.

Key point: the projection does **not** need an exact $L_0^{-1}$ to work. With
the cheap tensor $L_0^{-1}$, $\Pi$ is not perfectly idempotent, yet the
sandwich still drives the gradient energy of `P_A`'s output down enough to give
the ~96 iteration / ~267 ms result. The cleanup is partial, not complete: with
the tensor $L_0^{-1}$ the projected output still retains ~0.44 gradient energy
(`proj_grad_frac_exact mean=0.44`), since an inexact $L_0^{-1}$ cannot remove
all of it. A more accurate (but still non-Krylov, matrix-free) gradient-space
inverse would push that residual down and is a candidate lever for further
improvement.

#### How to read the overlap metrics (and what *not* to conclude)

The diagnostic prints two families of numbers. They do **not** agree, and the
gradient-energy fraction is the correct lens:

- **Gradient-energy fraction (exact projector)** — the trustworthy quality
  signal. Raw `P_A` = 0.83 (the pathology), projected = 0.44 (imperfect
  cleanup). This tracks the solver outcome: lower gradient energy ⇒ less
  interference with `P_B` ⇒ fewer iterations.
- **Gradient-energy fraction (tensor projector)** — the same quantity measured
  with the inexact tensor $L_0^{-1}$; it reads ~1.09 (above 1, impossible for a
  true projection) purely because the tensor projector is not idempotent. Use
  it only to see the artifact, not as a real fraction.
- **`|M-cosine(P_A, P_B)|`** — *do not over-interpret this.* Intuition says a
  low cosine means complementary subspaces and a good blend, but the data
  contradicts that: the projection that **improves** the solver 4× **raises**
  the cosine (0.077 raw → 0.57 projected). The cosine measures directional
  alignment of the two correction vectors, which is not the same as harmful
  subspace overlap, and here it moves the "wrong" way. It is retained for the
  record but is not a reliable overlap indicator for this problem.

#### Diagnostic vs. preconditioner: two different $L_0$

There is a subtlety worth recording, and it is why the exact-projector fraction
is trustworthy only after a fix. The overlap diagnostic measures what fraction
of `P_A`'s output lives in the gradient subspace by applying $\Pi$. For that
fraction to be a true energy fraction in $[0,1]$, the $L_0$ inverted inside
$\Pi$ must be **exactly** $G_0^T M_1 G_0$ built from the *same* incidence and
mass applies the sandwich uses. The library `apply_stiffness(·,0)` is
matrix-free but uses the **un-extracted core** $M_1$ with extraction only on
the 0-form side, i.e. a *different* $M_1$ than the `e1`-extracted
$M_1^\text{dbc}$ in the projector. Feeding that inconsistent $L_0$ to the
diagnostic produced gradient fractions above 1 (an artifact, not real energy).
The script's `l0_inv_exact` therefore builds its $L_0$ matvec from the same
DBC-extracted `apply_incidence_matrix` + `apply_mass_matrix(·,1)` chain as the
projector, so $\Pi$ is idempotent by construction and the exact-projector
fraction lands in $[0,1]$ (measured 0.83 raw / 0.44 projected, as above).

### Retired Script Options

The following options were removed from the script interface after repeated
underperformance or instability in this benchmark workflow:

- legacy/debug `P_A` modes (`cross`, `kinv`, `stiffness_tensor*`),
- script-only `mode3x3_fd`,
- multiplicative/symmetric-composition method rows.

The remaining comparison set is additive-only:

- `jacobi (diag)`,
- `jacobi(K)+P_B`,
- `P_A + P_B`,
- `jacobi + P_A + P_B`,
- plus `alpha*jacobi(diag) + P_A + P_B` via `--jacobi-scale-sweep`.

### Interpreting Iteration Counts Correctly

When a method does not reach the target residual, its iteration count is **not
meaningful**. Only converged methods should be compared on iterations or wall
time.

### Current Takeaway

- The retained `block_fd` variant is numerically PSD in this harness.
- **With the gradient-subspace projection (`--pa-grad-project`), `P_A + P_B`
  beats `jacobi (diag)` on both iterations (~96 vs ~386) and wall time (~267 ms
  vs ~290 ms).**
- A more principled experimental bulk model,
  `--pa-block-vector-fd-true-basis`, can improve further to ~84 iterations and
  ~230 ms when stabilized with per-mode scalar fallback on ill-conditioned
  coupled symbols.
- At `ns=(7,14,5)` and `ns=(7,14,8)`, the true-basis vector-FD model remains
  better than projected baseline block_fd on iterations and `cond_curl`, but
  still loses wall-time to Jacobi. Current blocker is `P_A` apply cost scaling,
  not loss of convergence.
- Without the projection, raw `P_A + P_B` still **converges** (residual reaches
  ~1.7e-10, below the 1e-9 "essentially done" line — the strict-tol "fail" flag
  is a preconditioned-vs-true-norm artifact, not a stall); the projection is a
  ~2.3× **accelerator** on iterations, not a make-or-break for convergence. It
  helps by stopping `P_A` and `P_B` from interfering on the gradient subspace.
- Adding Jacobi on top of the projected blend does not help: the alpha sweep
  has its minimum at `α=0`.
- CP rank is neutral (rank 1 = rank 3); the FD bulk approximation saturates at
  rank 1. The remaining lever is not CP rank but **vector coupling quality** in
  the curl bulk model.
- Open items: now also confirmed on `rotating_ellipse` (`nfp=3`, rank sweep
  1/2/3 — see "Rotating-ellipse robustness" below); still to do is more
  resolution coverage, retry `k=2,3`, and decide whether the projected
  `P_A + P_B` should be promoted out of the diagnostic harness into a production
  path.

### Why the low modes fail: the k=0 ↔ k=1 nullspace story

The decisive insight from the spectrum/leakage diagnostics is that the few
catastrophic modes are **not** a coordinate or axis defect — they are the exact
cohomology kernel of curl-curl, surfacing in the radial component.

**The singular symbols are the gradient kernel.** `K_1 = curl^T curl`
annihilates every gradient (`curl∘grad = 0`), so the near-`1e-28` eigenvalues
are a true nullspace (dimension `n0_dbc`), not ill-conditioning. The
min-eigenvector of each flagged 3x3 symbol is consistently `[1,0,0]` because
(a) the block is nearly diagonal there (off-diagonal ~`1e-15`) and (b) the
near-zero eigenvalue sits in the radial slot. Low-`(m,n)` gradients
`grad φ ≈ ∂_r φ · r̂` are radial-dominant (the θ/ζ parts scale with the angular
wavenumber), so the lowest gradient modes live almost entirely in the radial
component. Surgery does **not** — and should not — remove this: the kernel is a
property of curl-curl throughout the domain, not a `1/r` axis artifact.

**Why k=0 pinv "just works" but k=1 raw `P_S` does not.** Both operators are
singular (k=0 has the constant; k=1 has the whole gradient subspace). The
difference is *alignment with the inversion basis*:

| | k=0 (scalar) | k=1 (vector) |
| --- | --- | --- |
| nullspace | the constant: **one** separable tensor mode | gradient subspace: dim `n0_dbc`, **skew** to the blocks |
| inversion basis | fast-diag tensor modes | same-mode 3x3 blocks |
| alignment | kernel **is** a basis vector ⇒ pinv zeroes it exactly | ~48% modal leakage ⇒ kernel is off-block |
| result | exact surgery | block-local cap cannot emulate the projection |

A per-block pinv/regularization is the faithful k=1 analogue of "pinv the
constant" only if the kernel were block-aligned. The 48% leakage says it is not,
which is why `raw P_S + P_B` lets `P_A`/`P_B` interfere on the gradient subspace
and so converges only to the strict-tol edge (~1.7e-10, still below 1e-9 — slow,
not broken) while the projector `C P_S C^*` — which deletes the gradient subspace
*globally* before `P_S` sees it — is the faithful pseudoinverse and converges
~2.3× faster and cleanly under tol.

**The plateau is block error, not the kernel.** Projected `jacobi(S)` (same
projector `C`) converges cleanly to ~`6e-11`, while projected `P_S` stalls near
~`2e-10` despite far fewer iterations. Identical projector, different inner
model ⇒ the residual stagnation is the block approximation error (the leakage),
not the projection or the nullspace.

**The model error is radial mode-coupling ("banded mode").** The same-mode 3x3
symbol assumes `K_1`, in the per-component fast-diag basis, only connects mode
`m` to mode `m`. But curl contains `∂_r`, which is **not** diagonal in the
radial fast-diag basis — it couples neighboring radial indices. The per-axis
diagnostic confirms this quantitatively: on a symmetry-breaking
`rotating_ellipse` (κ=1.2, nfp=3) the off-diagonal modal energy is
`radial=9.69e-01 poloidal=2.89e-02 toroidal=1.14e-03 mixed=9.29e-04` — ~97%
radial even when ζ-symmetry is broken. So the band must extend in **radial**
only; θ/ζ stay (approximately) diagonal.

**The implemented fix (`--pa-block-radial-banded`).** Because
`TYPES = (clamped, periodic, periodic)` keeps the poloidal/toroidal mode
*counts* identical across the three vector components (periodic derivative
spaces have the same dimension; only the radial count differs), the angular
index `(i_t, i_z)` is a common label. The builder therefore assembles, per
angular mode, the **dense block `(V^T K_1 V)` over the joint
`(component, radial mode)` space** — size `(R0+R1+R2)×(R0+R1+R2)` — by probing
the true extracted bulk operator one modal unit at a time, then SPD-clips and
inverts each. This is **exact** in radial + cross-component coupling and
block-diagonal only in the measured ~3% angular leakage; it subsumes the
same-mode 3x3 (which is the `R*=1` radial truncation) and the "tridiagonal
radial" idea (the dense radial block is cheap since `R* ≈ ns_r`). Build cost is
`N_bulk` true-operator probes (same order as the true-basis path); apply is a
batched `(n_ang, Rsum, Rsum)` solve in the modal coordinates. The builder
prints `radial-banded angular leakage` — the residual energy outside the
retained angular block — which should land near the measured ~3% (vs the 0.48
mean leakage of the 3x3) as the cheap success signal before MINRES.

**What this means for k=0.** The same construction exists for the scalar
Laplacian `L_0`, but there it is a *luxury*, not a fix, for two structural
reasons:
- **No cross-component coupling.** `L_0` is scalar (one field), so the entire
  3x3 dimension collapses — the angular block is just `R_radial × R_radial`.
  Only the radial mode-coupling survives; the dominant defect at k=1 (the
  vector coupling next to a huge gradient kernel) has no k=0 analogue.
- **No near-kernel for the coupling to interact with.** `L_0` is coercive (SPD
  with a spectral gap); its only kernel is the **constant**, a single separable
  tensor mode that the diagonal `fd_inv_denom` + pinv already handles exactly.
  The radial mode-coupling is a small perturbation of a *dominant* diagonal, so
  it shifts eigenvalues without creating near-singular blocks and k=0 converges
  in a handful of iterations. At k=1 the identical relative coupling is
  catastrophic precisely because it sits next to the `~0` gradient eigenvalues,
  where the inverse amplifies any aliasing of the near-null radial modes.

So k=0 "just works" not because it lacks radial mode-coupling — it has it — but
because that coupling never meets a near-kernel. The radial-banded block is the
right *k=1* fix and only an optional accuracy knob at k=0 (relevant only if a
badly distorted geometry made the diagonal FD denom a poor preconditioner).

**Per-axis leakage diagnostic.** Whether the *angular* directions also leak is
measured rather than assumed: the FD builder buckets the true operator's
off-diagonal modal energy by axis (radial / poloidal / toroidal / mixed) and
prints it. The axisymmetric `toroid_map` cannot test this (κ only elongates the
cross-section; it stays ζ-independent), so use `--geometry rotating_ellipse`
(symmetry-breaking `nu(ζ)`) to probe whether the band must extend into θ/ζ.
`--leakage-only` stops after assembly for a cheap measurement.

### MINRES verdict: the radial band buys little here (ns=(6,12,4), p=3, toroid)

Both the radial-banded block and the same-mode 3x3 were run on the *identical*
geometry/seed with `P.T P_S P + P_B` (gradient-projected `P_A` + `P_B`), 4
random consistent RHS, MINRES tol `1e-10`:

| upper precond | avg_it | avg_ms | max_res | fails |
| --- | --- | --- | --- | --- |
| jacobi (diag) | 386 | ~330 | 9e-11 | 0/4 |
| 3x3 true-basis | 112.5 | 314 | 2.18e-10 | 3/4 |
| radial-banded | 101.0 | 346 | 4.24e-11 | 0/4 |

**Read the `max_res`/`fails` columns with care — they are NOT comparable across
preconditioners.** MINRES minimizes the *preconditioned* residual, so each row
measures convergence in its own norm; a fixed `1e-10` threshold on `max_res`
therefore lands differently for each preconditioner, and the 3x3's `2.18e-10`
vs the radial-banded's `4.24e-11` is a norm-equivalence constant (factor ~5),
not instability. The 3x3 is converging normally. The honest comparison is on
the norm-independent quantities:
- **Iterations are comparable:** 112.5 (3x3) vs 101.0 (radial-banded) — a ~10%
  edge to the band, well within run-to-run / RHS variation.
- **Spectra are essentially identical:** cond_curl 127 (3x3) vs 122
  (radial-banded), both tightly clustered (3x3 p05-p95 0.97-2.05; radial-banded
  0.79-2.80).
- **Per-iteration cost favors the 3x3** (2.8 vs 3.4 ms/it).

**Conclusion: at this resolution the radial band buys essentially nothing.** It
drops the modal `angular leakage` from 0.48 to ~0.05 and resolves the four
`~1e-27` low modes that the 3x3 only caps, but that extra fidelity does not
translate into materially fewer MINRES iterations or better conditioning — the
same-mode 3x3 already captures the iteration-relevant part of the coupling. The
band only adds apply cost (~20%/it) and an `O(n^6)` probing build that does not
scale. So the **same-mode 3x3 remains the working model**; the radial-banded
path stays as a diagnostic/accuracy reference behind its flag.

The defect could still matter where the radial coupling is genuinely stronger
(higher radial resolution, strongly distorted geometry); that is the only
regime where re-testing the band — ideally via the scalable `O(n^3)`
tensor-assembled build below, not the probing one — would be worthwhile.

**Open question / next direction.** The expensive part is *building* the
per-angular blocks by probing a global operator. The same blocks are assemblable
in `O(n_terms · n^3)` directly from the stored tensor terms (`term_r/term_t/
term_z`): in the FD basis that diagonalizes the angular factors, the block at
angular mode `(i_t,i_z)` is `Σ_terms (V_t^T T_t V_t)_{i_t i_t} (V_z^T T_z
V_z)_{i_z i_z} (V_r^T T_r V_r)` — a weighted sum of small `R_r×R_r` radial
matrices, no global matvecs, jittable. That would keep the same `O(n^4)` apply
while dropping the build from `O(n^6)` to `O(n^3)`, at the price of using the
diagonal-metric tensor model instead of the true operator (the same
approximation the non-true-basis vector-FD path already makes; the ~3% angular
leakage suggests it is accurate enough). Build it only if a stronger-coupling
regime makes the band actually pay off in iterations.

### Three-resolution sweep: the scalar bulk wins, coupling is counterproductive

Running all three bulk models (scalar / same-mode 3x3 / radial-banded) inside
the *same* `P.T P_S P + P_B` slot, on `toroid`, p=3, 4 RHS, MINRES tol `1e-10`,
across three grids (avg MINRES iterations):

| ns | jacobi | **scalar** | radial-banded | 3x3 |
| --- | --- | --- | --- | --- |
| (6,12,4) | 387 | **96** | 101 | 112 |
| (8,16,5) | 387 | **126** | 146 | 175 |
| (10,20,6) | 670 | **200** | 225 | 283 |

The ordering **scalar < radial-banded < 3x3 < jacobi** is stable across
resolution and the gaps do *not* close as `n` grows. This refutes the
hypothesis that the bulk coupling would start to pay off at higher resolution:
**the richer the bulk model, the more MINRES iterations it needs.** Restoring
the discarded spectral information (the off-diagonal curl coupling) is, here,
mildly counterproductive — consistent with the structural fact (below) that the
per-component *diagonal* blocks of `K_1` already carry the iteration-relevant
content.

**Why the scalar diagonal is already a faithful model.** Each per-component
diagonal block of `K_1 = curl^T curl` is a clean Kronecker sum, because `curl`
differentiates each component only in the two directions *transverse* to it
(never `∂_r a_r`, `∂_θ a_θ`, `∂_ζ a_ζ`). So the self-energy of component `a_r`
is `∫ (∂_θ a_r)^2 + (∂_ζ a_r)^2`, giving

```
K_rr = M_r ⊗ K_θ ⊗ M_ζ  +  M_r ⊗ M_θ ⊗ K_ζ   (2 terms)
```

— the k=0 Laplacian `K_r⊗M⊗M + M⊗K_θ⊗M + M⊗M⊗K_ζ` (3 terms) with the
component's *own* derivative direction dropped. Scalar FD diagonalizes each
`(K,M)` axis pair and inverts this block *exactly* (for a separable metric) via
a scalar modal denominator `D_jk = λ^θ_j + λ^ζ_k`. All the "complicated"
`SxDxS · SxSxD` curl cross-structure is purely **off-diagonal** (inter-component
coupling) or **metric non-separability** — neither of which the scalar model
needs, and restoring the former does not help convergence. The one genuinely
non-separable, off-diagonal piece that *does* matter — the gradient nullspace —
is handled by `P_B` + the projector, not by the bulk FD at all.

**Caveat.** This is `toroid` (axisymmetric) at moderate resolution. The coupling
might still matter on a strongly distorted geometry; but on the evidence so far
the bulk-coupling direction is closed as a negative result.

### Rotating-ellipse robustness: projection accelerates (~2.3×), rank is neutral

The earlier results are all on the axisymmetric `toroid`. To check that the two
load-bearing conclusions survive a genuinely 3D, symmetry-breaking geometry, the
raw-vs-projected `P_S` comparison was rerun on `rotating_ellipse` (`nfp=3`,
`κ=1.2`, `ε=1/3`), `ns=(6,12,4)`, `p=3`, scalar bulk, precompute on, 4 random
consistent RHS, MINRES tol `1e-10`, swept over CP rank 1/2/3:

| rank | upper precond | avg_it | avg_ms | fails |
| --- | --- | --- | --- | --- |
| 1 | jacobi (diag) | 406.5 | 304 | 0/4 |
| 1 | raw `P_S + P_B` | 340.2 | 344 | **4/4** |
| 1 | `P.T P_S P + P_B` | **150.2** | **184** | 0/4 |
| 2 | jacobi (diag) | 407.0 | 335 | 0/4 |
| 2 | raw `P_S + P_B` | 339.5 | 366 | **4/4** |
| 2 | `P.T P_S P + P_B` | **148.8** | **194** | 0/4 |
| 3 | jacobi (diag) | 406.2 | 331 | 0/4 |
| 3 | raw `P_S + P_B` | 338.0 | 363 | **4/4** |
| 3 | `P.T P_S P + P_B` | **150.0** | **194** | 0/4 |

Two takeaways, both consistent with the toroid story:

- **The projection is an accelerator, not a make-or-break.** Raw `P_S + P_B`
  (no gradient-subspace sandwich) reaches `max_res ≈ 1.66e-10` in ~341
  iterations — **below the 1e-9 "essentially converged" line**, so the "4/4
  fail" flag is purely the preconditioned-vs-true-norm gap against the strict
  1e-10 tol, *not* a stall. Its iteration count is real and meaningful: raw
  genuinely drives the residual to machine-ish levels. The projected
  `P.T P_S P + P_B` is faster — it converges in ~150 it (beating both raw's 341
  and jacobi's 407 by ~2.3–2.7×) and ~190 ms wall — and it cleanly crosses the
  strict tol. So the projection buys a ~2.3× speedup over raw, but raw already
  converges; the projection is not required for convergence (re-confirmed on
  current HEAD, job 14500415, 2026-06-20: jacobi 407.2/8.7e-11, raw
  341.5/**1.66e-10**, projected 149.8/5.0e-11).
- **CP rank is neutral.** 150.2 / 148.8 / 150.0 iterations across rank 1/2/3
  (≈±1 iter, flat wall time) confirms the rank-1 saturation seen on the toroid
  carries over to a non-axisymmetric metric. Rank 1 remains the right default.

Diagnostic note: the gradient-overlap metric on `rotating_ellipse` reads raw
`grad_exact ≈ 0.565` and projected `≈ 0.85` — i.e. it appears to *raise* the
gradient-energy fraction, opposite to the toroid (raw 0.83 → proj 0.44). Since
the MINRES outcome is unambiguous (projected converges, raw fails), this is
treated as the overlap metric being unreliable on this geometry (consistent with
the `l0_inv_exact` / non-idempotent-projector caveat above), not a real signal.
Not investigated further.

### Apply cost: the projector and P_B share one gradient-solve (4 → 2)

With the bulk model settled as scalar, the remaining apply cost is the
projection + `P_B`, not the bulk. The whole `C P_S C^* + P_B` collapses
analytically because of the single identity

$$
L_0 = G_0^\top M_1 G_0,
$$

i.e. the scalar Dirichlet Laplacian *is* the gradient–mass–gradient form. Define
the shared "gradient-solve" atom `E := G_0 L_0^{-1}` (and `E^* = L_0^{-1}
G_0^T`). Then all three pieces are built from `E`:

$$
\Pi = E\,G_0^\top M_1,\quad \Pi^\* = M_1\,G_0\,E^\*,\quad
P_B = E\,M_0\,E^\*,\quad S := E\,G_0^\top = G_0 L_0^{-1} G_0^\top .
$$

The naive sandwich writes **four** `L_0^{-1}` solves (one in `C^*`, one in `C`,
two in `P_B`). Two pairs coincide: the `C^*` inner solve and the `P_B` inner
solve are the *same* vector `q = L_0^{-1} G_0^T r`; the `C` outer solve and the
`P_B` outer solve are both `G_0 L_0^{-1}(·)`. So with `q = L_0^{-1} G_0^T r` and
`w = P_S (r - M_1 G_0 q)`,

$$
\boxed{\,M r \;=\; w \;+\; G_0 L_0^{-1}\!\big(M_0\,q - G_0^\top M_1 w\big)\,}
$$

uses **2** `L_0^{-1}` solves and is *algebraically identical* to `C P_S C^* +
P_B` (symmetry, hence iteration count, preserved). This is the method key
`P.T P_S P + P_B (fused)`.

**Measured (ns=(6,12,4), scalar bulk):** fused reproduces the non-fused
iteration count exactly (96 = 96) and trims wall-clock ~10% (276 → 250 ms).
**The small gain is itself a finding:** halving the `L_0^{-1}` solves barely
moves the clock, so the scalar `L_0` tensor preconditioner apply is *cheap* —
the `P_A` apply cost is **not** in the auxiliary Laplacian. It is in the surgery
couplings (next section).

### The real apply cost: dense surgery coupling (the win)

Each `P_A` apply contains two surgery↔bulk couplings, and each one runs a
**full matrix-free curl-curl apply** `K_1 = G_1^T M_2 G_1` (the surgery's
`apply_data` is the *extracted stiffness*, `row_extraction @ K_1(col_extraction_t
@ x)`) just to extract a tiny bulk↔axis block. That apply is an `M_2` mass apply
sandwiched between two sparse incidence applies, so it is `O(n^3 p^6)` (the `M_2`
`p^6` dominates). But the surgery space is the *polar-axis* treatment, a 1-D set
of DOFs with `surgery_size = O(n_ζ)` (≈ 20-30 in these runs). So the coupling
block `C = R K_1 E^T` is `(bulk × O(n_ζ))` — tiny and storable.

Precompute `C` once at build time (now the **production default**, set on the
`K1MassSurgeryPreconditionerFactors` payload at construction; `n_s` extra
curl-curl probes, same cost as the Schur build it sits beside). Then the
per-apply couplings become dense matvecs `C @ rhs_s` and `C^T @ rhs_b` (`K_1` is
symmetric, so the bulk→surgery block is exactly `C^T`), cost `O(bulk · n_ζ)` —
the entire `p^6` factor is removed. The Schur complement is unchanged, so the
preconditioner is **bit-identical** (iteration count must not move). The
diagnostic harness can disable it for A/B timing with `--no-precompute-coupling`
(launcher `NO_PRECOMPUTE_COUPLING=1`).

**Measured (scalar bulk, fused, toroid, p=3, 4 RHS, MINRES tol `1e-10`):**

| ns | jacobi it | jacobi ms | P_A it | P_A ms (mat-free coup) | **P_A ms (dense coup)** | speedup vs jacobi |
| --- | --- | --- | --- | --- | --- | --- |
| (6,12,4) | 387 | 290 | 96 | 250 | **158** | **1.83×** |
| (8,16,5) | 387 | 391 | 126 | 664 | **297** | **1.32×** |
| (10,20,6) | 670 | 846 | 202 | 1746 | **611** | **1.38×** |

Iteration counts are identical with and without the dense coupling (96/126/202),
confirming exactness. **This flips the verdict: before the dense coupling,
jacobi *won* wall-clock at the two larger grids (the `P_A` apply was too
expensive); after it, `P_A` beats jacobi on both axes — ~3.3× fewer iterations
*and* faster — at every tested resolution.** The dense coupling kills exactly the
`p^6` cost that grew faster than jacobi's diagonal multiply.

### Same trick for the k=0 auxiliary Laplacian (compounding win)

The fused `l0_inv` (the k=0 Hodge preconditioner `L_0 = G_0^T M_1 G_0`) fires
**twice per MINRES iteration** (the inner `q` and the fused outer solve), and it
has the *identical* structure: a core/bulk Schur split whose core↔bulk couplings
are reconstructed each call by a full matrix-free k=0 stiffness apply (an `M_1`
mass apply between two incidences, `O(n^3 p^6)`). The core (axis) size is small,
so the same precompute applies: the dense core↔bulk block `C0` is stored on the
`K0TensorHodgePreconditionerFactors` payload (production default), and
`l0_inv`'s couplings become `C0 @ /C0^T @` matvecs. Exact (iteration count
unchanged); same `--no-precompute-coupling` opt-out for A/B timing.

**Measured (both precomputes on — the production default):**

| ns | jacobi ms | mat-free coup | +dense K_1 | **+dense K_0** | final speedup |
| --- | --- | --- | --- | --- | --- |
| (6,12,4) | 290 | 250 | 158 | **117** | **2.47×** |
| (8,16,5) | 389 | 664 | 297 | **208** | **1.87×** |
| (10,20,6) | 845 | 1746 | 611 | **418** | **2.02×** |

Iterations still 96/126/202 (exact). Stacking the two precomputes takes `P_A`
from *losing* wall-clock at scale (1746 vs 845 at the largest grid) to **~2×
faster than jacobi at every resolution**, on top of ~3.3× fewer iterations. The
k=0 coupling removed two `M_1`-class applies per iteration, which is why it gave
a larger absolute cut than the single k=1 `K_1` coupling pair.

This also retroactively vindicates the un-jitted phase profile: it was
unreliable in *absolute* terms (29 ms reported vs ~2.6 ms/it actual, ~10× off
from per-kernel dispatch + `block_until_ready` barriers), but its *relative*
read — `bulk1 : schur : bulk2 = 1 : 1.5 : 2.4`, i.e. "the couplings dominate,
not the bulk solve" — was correct: the dense coupling removed exactly the
`schur`/`bulk2` cost it flagged. The trustworthy metric remains the *jitted*
MINRES `avg_ms`.

**Residual floor.** Three full-size costs remain untouched: (1) the fused
outer's `M_1 G_0 q` and `M_1 w` (full 1-form `M_1` applies, outside `P_S`);
(2) the lower k=0 **mass** block (`lower_tensor_precond`), whose own core↔bulk
coupling is still reconstructed matrix-free — the same precompute trick would
apply, but to a *different* operator (`M_0`, not `K_0`); (3) the saddle
`A_matvec`. Together these set the floor. Going below it needs a cheaper mass
apply itself (matrix-free vs stored — see `benchmark_matvec_sparse.py`), or the
same dense-coupling trick extended to the lower mass block, not more
preconditioner algebra on the upper block.

### Does regrouping change the "P_B handles the gradient block" picture? No.

Fully expanding the sandwich with `Π = S M_1`, `Π^* = M_1 S`,

$$
C P_S C^\* = P_S - \Pi P_S - P_S \Pi^\* + \underbrace{\Pi P_S \Pi^\*}_{=\,E\,(G_0^\top M_1 P_S M_1 G_0)\,E^\*},
$$

the quartic term has the **exact** form of `P_B = E M_0 E^*` — same outer
`G_0 L_0^{-1}(·) L_0^{-1} G_0^T`, with `M_0` replaced by `G_0^T M_1 P_S M_1 G_0`.
So one *can* regroup into a single "effective gradient-subspace solve"
`E (M_0 + G_0^T M_1 P_S M_1 G_0) E^*` plus the cross terms. But this regrouping
**does not change the interpretation**, because `C P_S C^*` has range *exactly*
in the curl complement — `Π (C P_S C^*) = 0`, since `Π C = Π(I - Π) = 0`:

$$
\Pi\,(P_S - \Pi P_S - P_S \Pi^\* + \Pi P_S \Pi^\*) = \Pi P_S - \Pi P_S - \Pi P_S \Pi^\* + \Pi P_S \Pi^\* = 0 .
$$

The gradient content of the `P_B`-shaped quartic term is *exactly cancelled* by
the `-Π P_S` and `-P_S Π^*` cross terms. So the "term from `P_A` that looks like
`P_B`" is an artifact of tearing a pure-curl-complement operator into pieces
that individually carry gradient content summing to zero — **`P_A` contributes
nothing net to the gradient block.** The grouping-independent truth is that `M`
is block-diagonal on complementary subspaces: `C^*` annihilates the gradient
part of the input (`C^* M_1 G_0 = 0`) and `C P_S C^*` outputs into the curl
complement; `P_B = E M_0 E^*` only reads the gradient potential (`E^*`) and
outputs into `ran(G_0)`. They share the factor `E` for *computation* (why 4
solves are 2), not because of any conceptual overlap. The honest reading stays:
**`P_S` on the curl complement, `P_B` on the gradient subspace.**

### Recommended configuration

The current best k=1 upper preconditioner is the **scalar bulk model + fused
projector + precomputed dense surgery coupling + precomputed dense k=0 Hodge
coupling**, i.e. `P.T P_S P + P_B (fused)` with the *default* (scalar)
`block_fd` bulk (no `--pa-block-*` coupling flag). Both dense couplings are now
the **production default** (built on the preconditioner payloads at
construction), so no extra flags are needed. Across
(6,12,4)/(8,16,5)/(10,20,6) it is ~3.3× fewer iterations than jacobi *and* ~2×
faster wall-clock (2.47×/1.87×/2.02×). The bulk coupling models (3x3,
radial-banded) are dominated on every axis and are retained only as diagnostic
references behind their flags.

### Stabilization currently in place

The **scalar** bulk model is the working `block_fd` default (no coupling, no
SPD-clipping of low modes — the gradient nullspace it would otherwise need to
clip is handled by `P_B` + the projector). The coupled models keep a guard for
their near-singular low modes: the 3x3 runs coupled solves only where
`min_eig(3x3) > cutoff` (cutoff on the PSD-clipping relative scale), else a
mode-local, symmetry-preserving scalar fallback. The flagged set is a small
low-index family — modes `0..3` at `ns=(6,12,4)`, `0..4` at higher grids,
**not** an `n_z` prefix — so forcing `--pa-block-vector-fd-low-mode-exclude n_z`
over-regularizes. This matters only for the (now-dominated) coupled variants.

### Next steps

1. **Dense couplings are now the production default** (both `K_1` surgery and
  `K_0` Hodge core), set on the preconditioner payloads at construction via
  `_build_k1_stiffness_surgery_factors` / `_assemble_k0_tensor_hodge_preconditioner`
  (toggle: `cp_kwargs["precompute_coupling"]`, default `True`; harness opt-out
  `--no-precompute-coupling`). No further wiring needed for the upper block.
2. **Next remaining lever: the lower k=0 mass block.** `lower_tensor_precond`
  still reconstructs its core↔bulk coupling matrix-free (an `M_0` apply). The
  same dense-coupling trick applies (different operator), and it fires once per
  iteration — the obvious next measurement.
3. **Then the hard floor is the `M_1` apply itself** (the fused outer
  `M_1 G_0 q` / `M_1 w`, plus the saddle `A_matvec`), not preconditioner
  algebra. Going below it needs a cheaper mass apply (matrix-free vs stored —
  see `benchmark_matvec_sparse.py`).
4. **Retain the coupled models as references only.** 3x3 and radial-banded are
  dominated; keep `--pa-block-vector-fd-true-basis` / `--pa-block-radial-banded`
  for diagnostics. Revisit the band only on a strongly distorted geometry where
  the bulk coupling might start to matter.
5. **Robustness checks before promotion** across geometry/resolution; track the
  coupled models' flagged low-mode family if they are ever reconsidered.

**Pick up from here (2026-06-24):**

6. **Tensor rank decision is settled — keep rank=1.** The rank>1 blow-up is a
  single isolated outlier in `smoother∘L_0`, and the outer adaptive CG already
  absorbs it (+3 iters) where it consumes the atom directly; the non-adaptive
  nested Chebyshev cannot, but rank-1 has no outlier there. Do not build the
  1-vector deflated Chebyshev absent a concrete rank>1 need (see *Tensor rank*).
7. **k=2 wall-win lever (the live open problem): a cheaper near-exact `L_1`
  inverse.** The construction is sound, degree-scalable, and matrix-free; the only
  thing keeping it from beating jacobi on wall time is the cost of "polynomial
  over the k=1 preconditioner" as the inner `L_1⁻¹`. A multilevel
  vector-Laplacian solve for `L_1` is the candidate. This is where to resume the
  k=2/k=3 thread.
8. **Resolve the competing "Chebyshev-outside" head-to-head** (`make_cheb_tensor_upper`,
  `--all-k` run): a Chebyshev directly on `Ŝ_k` with only the cheap tensor
  stiffness smoother, no HX projector. Confirm whether it fails on the gradient
  modes as expected, or offers a cheaper-but-weaker alternative for k=2.

### Re-running retained cases

```bash
# RECOMMENDED: scalar bulk + fused projector. The dense couplings (k=1 surgery
# K_1 and k=0 Hodge K_0) are now the production default, so no extra flags are
# needed. ~2x faster than jacobi AND ~3.3x fewer iterations at every resolution.
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-grad-project --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B (fused)"

# CURRENT BEST: gradient-projected P_A + P_B vs Jacobi (k=1)
# (method key "P.T P_S P + P_B" = projected P_A + P_B; P_S is P_A here)
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-grad-project --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B"

# FUSED projected apply: identical operator, 2 L_0^{-1} solves instead of 4.
# Verify avg_it matches the non-fused row exactly (algebraic identity).
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-grad-project --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B,P.T P_S P + P_B (fused)"

# ROOT-CAUSE FIX: radial-banded bulk (full radial+component per angular mode).
# Prints "radial-banded angular leakage" (~3% target vs 0.48 for the 3x3).
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-grad-project --pa-block-radial-banded --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B" \
  --pa-stiffness-spectrum active

# Cheap per-axis leakage measurement on a symmetry-breaking geometry
# (stops after P_A assembly; no property/overlap/spectrum/MINRES work)
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --geometry rotating_ellipse --kappa 1.2 --nfp 3 \
  --pa-block-vector-fd --leakage-only --system saddle

# Experimental: true-basis vector-FD coupled bulk with per-mode fallback
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-grad-project --pa-block-vector-fd-true-basis --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B" \
  --pa-stiffness-spectrum active

# Same with compact diagnostics and P_A phase profiling
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-grad-project --pa-block-vector-fd-true-basis --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B" \
  --pa-stiffness-spectrum active \
  --compact-output --pa-profile

# Probe low-index mode diagnostics on a slightly higher resolution
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --ns 7,14,8 \
  --pa-grad-project --pa-block-vector-fd-true-basis --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B" \
  --pa-stiffness-spectrum active --pa-block-vector-fd-report-k 16

# block_fd at tensor rank 3 (default bulk model, no projection)
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-mode block_fd --rank 3 \
  --methods "jacobi (diag),jacobi(K)+P_B,raw P_S + P_B,jacobi + raw P_S + P_B"

# optional inner bulk Schur in block_fd
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-mode block_fd --pa-block-inner-schur --rank 3 \
  --methods "jacobi (diag),jacobi(K)+P_B,raw P_S + P_B,jacobi + raw P_S + P_B"

# alpha sweep with single-compile runtime parameter path
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-mode block_fd --rank 3 \
  --methods "jacobi (diag),jacobi(K)+P_B,P_A + P_B,jacobi + P_A + P_B" \
  --jacobi-scale-sweep "0,0.5,1,2,4"
```

Note on spectrum diagnostics: `--pa-stiffness-spectrum all` includes an
`active:<pa-mode>` row in addition to `jacobi` and `dense pinv`, and the table
now reports `cond_curl` (curl-subspace-restricted condition number) plus `nnz`
(count of nonzero modes) alongside the nullspace-polluted `cond_abs`. The
curl/nullspace split cutoff is set by `--pa-stiffness-spectrum-curl-cutoff`.

## Current Policy

- Production higher-form solves stay on the existing saddle MINRES path for now;
  the projected `P_A + P_B` win (`k=1`) is demonstrated in the diagnostic harness
  and has not yet been promoted to a production code path.
- Keep Jacobi Schur outer as the reliability baseline. It remains the best
  option for `k=2` and `k=3`, where the auxiliary-space variants do not yet
  net-win (see the consolidated `k=2`/`k=3` sections).
- The gradient-subspace-projected `P_A + P_B` (`--pa-grad-project`) is the
  validated `k=1` direction; it now also works under free BCs with a deflated
  harmonic. Nullspace robustness is confirmed for `k=0` and `k=1` (both BCs).
- Baselines: a degree-5–8 Chebyshev on the approximate Schur is
  iteration-competitive with the tensor method but ~3× slower in wall time; the
  tensor method's edge is its cheap per-apply cost.
- `k=1` (and `k=0`) are ready to promote: the unified tensor-Cheb `L_0` atom
  matches the exact inverse, the degree is h-flat (~7), it beats jacobi ~2× on
  wall and converges where jacobi stalls, and with the production block_fd `P_A`
  the gradient projection is unnecessary (raw == projected). Ship the
  constant-deflation of the `k=0` apply alongside it.
- `k=2` is degree-scalable, matrix-free, and a robustness win (converges where
  jacobi is marginal) but **~3.3× slower than jacobi on wall** — the deep nesting
  (outer Chebyshev × k=1 smoother × inner cheb-`L_0`) is the cost. The one open
  lever for a wall win is a **cheaper near-exact `L_1` inverse** than "polynomial
  over the k=1 preconditioner" — i.e. a multilevel vector-Laplacian solver. `k=3`
  stays on jacobi.
- **Tensor rank stays at 1** for the stiffness/Hodge atom. rank>1 is OOM-slow via
  a single phantom outlier; the deflated-Chebyshev fix is *not* worth building
  unless a concrete need for a rank>1 near-exact fixed-linear `L_0⁻¹` appears (see
  *Tensor rank*). Where an adaptive outer Krylov consumes the atom directly,
  rank>1 can be used without deflation by capping the auto-degree.
- Open theme tying `k=2`/`k=3` together: still the absence of a near-exact,
  *cheap* auxiliary inverse for the vector curl block.

If the projected blend is promoted, define acceptance criteria up front (clear
wall-time win across the target problem family, stable iteration behavior, and
clean integration with nullspace handling).

