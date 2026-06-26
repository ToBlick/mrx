# Higher-form Hodge-Laplacian preconditioners (HX/AMS-style)

Reference for preconditioning the Hodge Laplacians `L_k` of the FEEC B-spline de
Rham complex `V0 -G0-> V1 -G1-> V2 -G2-> V3` (grad / curl / div) on the polar
(clamped × periodic × periodic) torus. Two reusable building blocks, measured
across the whole degree family; the consolidated status, the building blocks, the
k=0 radial-dense fix (the 2026-06 result), and the k=1+ auxiliary-space construction
follow. Older derivation/session history is compressed to summaries.

## Consolidated status (2026-06)

| degree | operator | best preconditioner | vs jacobi | status |
| --- | --- | --- | --- | --- |
| k=0 | grad-grad `L_0` | scalar tensor Hodge atom | ~10× iters, ~7× wall | **ship.** Both BCs, nullspace-robust. Free-BC `κ≈6` was one radial-coupling mode the separable FD misses — diagnosed and fixed: a **dense radial block per angular mode** (`radial_dense`) drops it to `κ≈2` (`3.06→2.15` toroid, `6.19→2.91` rot-ellipse). See *k=0 atom*. |
| k=1 | grad-div `L_1` | projected `P_A + P_B`, tensor-Cheb `L_0` atom | **~2× wall, beats jacobi** | **ship.** Both BCs. With the production `block_fd` `P_A`, raw == projected (projection-free). `P_B` atom = degree-~7 Chebyshev on `L_0` (κ≈6 h-flat) — matches exact `L_0⁺`; ~11× fewer iters than jacobi (49 vs 553 free), converges where jacobi stalls. See *k=1+*. |
| k=2 | div-div `L_2` | projected `P_A(cap) + P_B`, nested cheb atoms | iters better, wall **slower** | degree-scalable, matrix-free, projection mandatory — but ~3.3× slower than jacobi on wall (deep nesting). Robustness tool, not a wall win. Needs a cheaper near-exact `L_1` inverse (multilevel). |
| k=3 | `L_3 = D_2 M_2⁻¹ D_2ᵀ` | **jacobi** | jacobi wins | `L_3` tiny + well-conditioned; the unified pure-`P_B` form converges but loses badly (280 it / 10.4 s vs jacobi 186 it / 0.47 s). Use jacobi. |

> **The discrete derivative `G` is fully matrix-free / inverse-free (2026-06).** grad
> `G_0` and curl `G_1` ship as analytic `±1`/`−ξ` sparse stencils (`build_grad_stencil_g0`,
> `build_curl_stencil_g1` in `mrx/operators.py`), bit-exact (≤2.2e-16) vs the
> `Gram⁻¹∘incidence` oracle, both BCs; div `G_2` was already matrix-free (`Gram₃=I`).
> The polar-`G` correctness fix (the directly-built incidence apply omitted the
> polar-axis inverse, so `d∘d ≠ 0` near the axis) restored `d∘d = 0` to ~3e-16 and
> unlocked the idempotent k=1 projector and k=2. See `docs/polar_true_derivative_G.md`.

## Building blocks

1. **Scalar tensor Hodge atom** for the `k=0` Laplacian `L_0 = G_0^T M_1 G_0`
   (and its `k=3` dual): a CP-tensor / fast-diagonalization (FD) inverse with an
   outer surgery-Schur split for the polar axis. Near-exact and cheap; the workhorse
   the vector-form corrections reach down to. Both `dbc` and `no-dbc` variants.

2. **Additive auxiliary-space upper/Schur block** for the vector Laplacians,
   `P = (I-Π) P_A (I-Π)^T + P_B`:
   - `P_A` — native-space stiffness inverse (`block_fd`: per-component FD bulk +
     surgery Schur) for the curl-curl / div-div part;
   - `P_B = G X G^T` reaches down to the scalar atom on the gradient/exact subspace
     (for k=1, `X = L_0^{-1} M_0 L_0^{-1}`, two `L_0⁻¹` applies);
   - `Π = G_0 L_0^{-1} G_0^T M_1` confines `P_A` to the curl complement.

Vector solves are saddle-point MINRES (the exact `M_{k-1}` sits in the lower block,
so `M_{k-1}^{-1}` is never forward-applied); `P` is the upper-block (Schur)
preconditioner. The scalar `k=0` solve is plain (deflated) CG. The upper
preconditioner acts on the approximate Schur `Ŝ = S_k + D_{k-1} M̂_{k-1}^{-1} D_{k-1}^T`
where `M̂^{-1}` is the mass *preconditioner* (one apply) — the true Schur's exact
`M^{-1}` is never formed.

---

# k=0 atom: the scalar tensor Hodge inverse

`L_0 = G_0^T M_1 G_0`. The bulk operator with a diagonal-metric assumption separates
as `K_r⊗M_t⊗M_z + M_r⊗K_t⊗M_z + M_r⊗M_t⊗K_z` (different metric channels
`α_rr = Jg^{rr}`, `α_θθ`, `α_ζζ` on the three summands). Each channel is CP-fit; the
rank-1 leading terms define per-axis FD bases `V_r/V_t/V_z`; the polar axis is handled
by an outer surgery-Schur split (small, mesh-independent). The bulk apply is six
einsums (forward FD transform, divide by a modal denominator, back transform). The
core↔bulk couplings are precomputed dense blocks (`core_coupling`), so the per-apply
cost is `O(N)` with no `p^6` mass applies.

## The conditioning outlier and the radial-dense fix (2026-06)

The free-BC k=0 atom had `κ≈6` (rot-ellipse) / `κ≈3` (toroid) driven by **one** mode,
and CP rank>1 drove that same mode into an OOM basement (`κ` 51–384). The full
diagnosis (`scripts/debug/debug_rank_ritz_spectrum.py`, dense `eig(smoother∘L_0)`,
new flags `--dbc/--exact-bulk/--model-exact-inverse/--radial-dense` + a bulk-pencil
probe):

- **One mode, every rank.** Cross-rank `M_0`-cosine of the `λmin` eigenvector is
  0.98–1.0; rank-1 carries it mildly, rank>1 deepens it. Robust across CP rank,
  resolution (ns 6,12,4 & 7,14,4), and spline degree (p=1,2,3).
- **It is the separable FD approximation, full stop.** Replacing the FD bulk inverse
  with the *exact dense* bulk-block inverse (`--exact-bulk`) collapses the whole
  composite to **κ=1.000**. Not surgery, not structure, not ill-posedness.
- **Approximation vs inversion error split.** Inverting the separable *model* `K̃_bb`
  exactly (`--model-exact-inverse`) gives rank-1 κ=2.91 (≈ FD 3.06) but rank-2 κ=1.79
  (vs FD 51). So the **rank-1 outlier is the `K_bb≈K̃_bb` approximation** and the
  **rank>1 basement is the Lynch inversion error** (the rank-2 model is *better* but
  Lynch inverts it catastrophically; even rank-1 FD is inexact — the three metric
  channels give three rank-1 factors that don't share one per-axis eigenbasis).
- **Free-BC specific; not a harmonic; not a `K_bb` near-kernel.** Clamping the *outer*
  boundary (`--dbc`) removes the outlier at every rank (toroid κ 3.06→1.38). The mode
  is not a near-harmonic of `L_0` (its Dirichlet energy ranks ~21/202 — middling; the
  genuine near-harmonics are *different*, well-preconditioned modes), nor a small
  eigenvalue of `K_bb` (lowest eig 8.3; the mode aligns with mid-spectrum bulk mode #8).
  Its `M_0`-orthogonality to the constant is forced by deflation, not physical.

**It is the separable FD's missing *radial* coupling** — `∂_r` is not diagonal in the
radial FD basis (the "~97% radial" leakage). The fix: diagonalise θ,ζ with the FD
eigenbases (rank-1) and invert an **exact dense `n_r×n_r` radial block per (θ,ζ) mode**:

| geometry | production FD | **radial-dense** | exact bulk |
| --- | --- | --- | --- |
| toroid | κ=3.06, outlier @0.600 | **κ=2.15, no outlier** (validated prod: 2.162) | 1.00 |
| rotating-ellipse | κ=6.19, outlier @0.271 | **κ=2.91, no outlier** | 1.00 |

radial-dense *drops* both the cosθ angular coupling and the nfp ζ coupling yet still
removes the outlier on both geometries — so neither is needed. The residual κ≈2–3 is
a benign *spread* (no isolated mode → does not inflate the Chebyshev degree).

### Production form (implemented: `cp_kwargs={"radial_dense": True}`, rank≥2)

`B[j,k] = Σ_terms A_r · diag(V_tᵀA_tV_t)_j · diag(V_zᵀA_zV_z)_k`, batched-inverted;
apply = angular `einsum` transforms + a batched radial matvec — device-side, jittable,
`n_r×n_r` blocks only (never the `O((n_r n_θ)³)` dense (r,θ) block). It is the exact
inverse of the "rank-2 in r, rank-1 in (θ,ζ)" model, a strict generalization of FD
(rank-1 metric → `B = K_r + (λ_t+λ_z) M_r` → the FD denom). Wired in `mrx/operators.py`:
`bulk_radial_block_inv` on `K0TensorHodgePreconditionerFactors`,
`_apply_k0_tensor_hodge_bulk_radial_dense`, gated by `radial_dense` through
`_assemble_k0_stiffness_fd_bulk_factors`. Default path is byte-unchanged (flag off →
FD). Cost: build `O(n_θ n_ζ n_r³)`, apply `O(N·n_r)` — `n_r×` the FD `O(N)`; for large
`n_r` exploit the radial **banding** (banded solve → `O(N·p)`), not low rank (the block
is full-rank, well-conditioned; low rank doesn't apply).

```python
# build (setup): one dense n_r x n_r radial inverse per (θ,ζ) mode
radial = jnp.stack([t.r for t in terms])                  # (T, n_r, n_r)
diag_t = jnp.stack([_modal_diagonal_from_basis(Vt, t.t) for t in terms])   # (T, n_t)
diag_z = jnp.stack([_modal_diagonal_from_basis(Vz, t.z) for t in terms])   # (T, n_z)
B    = jnp.einsum('tj,tk,trs->jkrs', diag_t, diag_z, radial)               # (n_t,n_z,n_r,n_r)
Binv = jnp.linalg.inv(B)
# apply: einsum transforms + one batched matvec, no host roundtrip
def bulk_inverse(rb):
    r = rb.reshape(n_r, n_t, n_z)
    y = jnp.einsum('tj,zk,rtz->rjk', Vt, Vz, r)   # to angular eigenbasis (V^T)
    z = jnp.einsum('jkrs,sjk->rjk', Binv, y)      # dense radial solve per (j,k)
    return jnp.einsum('tj,zk,rjk->rtz', Vt, Vz, z).reshape(-1)   # back (V)
```

### Metric fit: drop CP-ALS, use a sequential SVD

With (θ,ζ) pinned to rank-1 and r dense, the fit is asymmetric → a deterministic
sequential/truncated HOSVD (no ALS): (1) rank-1 SVD over ζ → `h(ζ)`, `V_ζ`; (2)
rank-1 SVD over θ → `g(θ)`, `V_θ`; (3) rank-r SVD over r → the radial profiles. This
is *safe* where CP-ALS rank>1 was not: the radial rank is inverted **densely** (no
Lynch truncation), so Frobenius-optimal SVD is correct and more radial rank strictly
helps. Two caveats: do the radial SVD on the angular-*diagonal* weight (the even
θ-harmonics — cos²θ from `1/R²` — land on the `m=0` diagonal with an `r²/R₀` profile;
the odd cosθ part is off-diagonal and safely dropped — the term-based assembly drops
it automatically since its modal diagonal is ~0); and verify against the true-operator
`--radial-dense` (production matched it: κ 2.162 vs 2.153).

### Rank diagnostics (`scripts/debug/debug_radial_dense_ranks.py`)

Toroid ns 6,12,4, CP rank 2: `α_rr` is **(r,θ) rank 2, ζ rank 1**; `α_θθ` rank 1+~1%
correction; `α_ζζ` (the `1/R²` channel) (r,θ) rank **2→3→5→6** (decaying ~20×/order,
so rank-2 ≈ 95%, rank-3 ≈ 99.6%). The radial blocks are full-rank SPD (cond 4.8–69),
with cross-mode variation `‖B[j,k]−mean‖/‖mean‖` = 0.13–0.99 — that variation *is* the
radial coupling radial-dense captures (0 ⇒ radial-dense ≡ FD). So: **ζ rank-1 is exact;
(r,θ) rank-2 captures the dominant channel; bump CP rank to 3 to chase the `α_ζζ` tail
at zero extra inversion cost.**

### Why rank-1 θ is the sweet spot (rank-2 θ tested, not worth it)

You *are* punished for dropping the rank-2 θ (cosθ) coupling — toroid κ goes 1.0
(exact) → 2.15 (radial-dense), `~ε = r/R₀ ~ 1/3`. But it's a benign **cluster**, not an
**outlier**: the composite is `B = I + L_diag⁻¹ L_off`, and the cosθ coupling is a
bounded `m±1` shift among well-conditioned θ modes — θ is periodic, **no near-null
direction** for it to blow up against. The *radial* coupling the separable FD dropped
hit the near-axis near-null mode (the `1/r` metric singularity), so its inverse
amplified it into the outlier; radial-dense removes that by inverting the radial block
densely. So the asymmetry isn't "θ is secretly rank-1" — it's that the polar axis gives
*r* a near-singular direction and θ has none.

Keeping rank-2 θ exactly = a block-tridiagonal-in-θ solve (cosθ → `m±1` in the θ-DFT
basis), solved by block-Thomas (`O(n_θ n_ζ n_r³)` build, ~2× apply, sequential in θ via
`lax.scan`, + Sherman–Morrison for the periodic corner). But the κ(band) test
(`--rt-coupled --rt-band`) shows it **doesn't pay**: band 0 (rank-1 θ) 2.55 → band 1
(rank-2 θ) 2.22 → band ~3 reaches κ=1. The θ-coupling is bandwidth ~3 at ε=1/3 (the
`1/R` harmonics decay like `εᵏ`), so a tridiagonal solve captures one ε-order and
stalls; reaching κ≈1 needs a band-3 solve, already a big fraction of the full (r,θ)
dense cost. **Ship rank-1 θ** (radial-dense, fully parallel, κ≈2 benign); only go to the
full (r,θ) solve if a measurement shows the cluster width costs outer iterations.

## Tensor rank policy

**rank=1 is the production default for the *separable* atom.** rank>1 OOMs via the
single outlier — now understood as the radial coupling + the Lynch inversion error, not
a "phantom" or a "Schur-interaction mode." The fix is **radial-dense** (above), whose
radial rank is a *free* accuracy knob (dense inversion → no Lynch). For the separable
FD path keep rank-1; for radial-dense use CP rank≥2.

**Enforced in code (2026-06): only rank 1 and rank 2 are supported.**
`assemble_tensor_laplacian_preconditioner` now maps the rank directly to the policy —
`rank=1` is the separable FD inverse; `rank>=2` *auto-selects* `radial_dense` (so the
unsupported separable-FD-at-rank>1 OOM path is unreachable), and any rank outside
`{1, 2}` raises. The k=0 config fallback default is rank 1 (was 3). Mass preconditioners
are unaffected (rank 1/2 both fine — a rank-2 metric is an exact 2-term Kronecker sum).

**Caveat — `radial_dense` (rank=2) is NOT yet free-BC safe in a deep CG solve.**
A condensed-CG sweep to `tol=1e-10` across cylinder / toroid / rotating-ellipse at p=3
(`scripts/benchmark/benchmark_k0_rank_geometries.py`, `slurm/job_k0_rank_geometries.sh`)
shows:
- **dbc, all geometries:** rank=2 converges, ~9–10 it, ≈ rank=1 (the κ outlier is a
  single mode CG absorbs in ~1 extra iter, so radial_dense's κ 6→2 buys ~nothing here).
- **free BC, toroid + rotating-ellipse:** rank=2 **stalls at ~1e-6 (4/4 fail)** at every
  resolution, while rank=1 FD converges cleanly to ~1e-11 (11–18 it). The separable
  **cylinder** free BC is the exception — radial_dense is *exact* there (1 it), because
  the cylinder metric is genuinely (θ,ζ)-separable.
- The earlier radial_dense validation was a dense `κ(smoother∘L_0)` spectral check
  (`debug_rank_ritz_spectrum --radial-dense-prod`), **never a CG-to-1e-10 solve**, so the
  free-BC convergence stall slipped through. Likely cause: the dense per-mode block
  inverse (`_batched_floored_spd_inverse`, `rtol=1e-12`) does not deflate the constant
  nullspace consistently with the FD denom-floor + core-surgery path it replaces (its
  docstring *assumes* the bulk is SPD / nullspace-free, which fails free-BC + curved).
- **Net:** until the free-BC stall is root-caused, rank=2 radial_dense is a dbc-only /
  separable-geometry tool; rank=1 FD remains the robust production k=0 atom for both BCs.
  Both ranks beat jacobi by ~10–15× iters / ~10× wall wherever they converge.

## Mass matrices: exact at rank-2, no change needed (all k)

The **mass** has no derivatives, so a rank-2 metric gives exactly **2** Kronecker terms
(vs the stiffness's 6 from three derivative directions). Two terms → two matrices per
axis → a **pair**, always simultaneously diagonalizable → exact inverse with
`denom = 1 + λ_r λ_θ λ_ζ`. This is already `_build_kron_sum_fd_factors` /
`_simultaneous_diagonalize_pair`, and it is why the *mass* preconditioner **improves**
with rank (iters 4.8→3.5→3.0) while the stiffness blows up. A **CP-rank-2** fit is
rank-2 in all three coordinates simultaneously and inverts exactly, symmetric in r/θ/ζ
— **no privileged axis, no dense radial block, no change to ship.** Limits: exactness
is tied to ≤2 *terms* (CP rank); an unaligned Tucker-(2,2,2) → up to 8 terms is *not*
exact this way. Scalar masses (k=0,3) are clean; vector masses (k=1,2) are exact per
component under the diagonal-metric model — keeping the off-diagonal metric `g^{rθ}`
adds cross-component terms past 2.

---

# k=1+ vector Laplacians: projected `P_A + P_B`

For k=1, an additive `P_A + P_B` upper block beats the Schur-outer Jacobi baseline once
`P_A` is restricted to the curl-complement by the gradient-subspace projection. The
blocker was never indefiniteness or Schur tuning — it was that raw `P_A` and `P_B` both
acted on the gradient subspace and interfered (83% of raw `P_A`'s output energy is in
the gradient subspace). Restricting `P_A` to the `M_1`-orthogonal complement of `ran(G_0)`
unblocks it.

`P_B = G_0 L_0^{-1} M_0 L_0^{-1} G_0^T` — two scalar `L_0⁻¹` applies with one `M_0`
between. The fused projected apply (`L_0 = G_0^T M_1 G_0`) collapses the 4 `L_0⁻¹`
solves to 2 (algebraically identical, symmetry preserved):
`M r = w + G_0 L_0^{-1}(M_0 q − G_0^T M_1 w)` with `q = L_0^{-1} G_0^T r`,
`w = P_S(r − M_1 G_0 q)`.

**Headline (toroid ns 6,12,4, p=3, ε=1/3, scalar bulk, dense couplings on):**

| upper precond | iters | wall |
| --- | --- | --- |
| jacobi (diag) | ~386 | ~290 ms |
| projected `P_A + P_B` (fused) | **~96** | **~117 ms** (~2.5× faster) |

The wins, in order of impact: (1) the **gradient-subspace projection** (~4× fewer iters
vs raw); (2) **dense surgery couplings** precomputed once (`C = R K_1 E^T`, tiny
`bulk×O(n_ζ)`) — removes the `p^6` `M_2` apply from each surgery↔bulk coupling, flipping
the verdict from "jacobi wins wall at scale" to "`P_A` beats jacobi on both axes at every
resolution"; (3) the **same dense-coupling trick for the inner k=0 `L_0`** (fires twice
per iter), compounding to ~2× faster than jacobi. With the production `block_fd` `P_A`
the projection is unnecessary (raw == projected — its imperfection happens to damp the
leak); the projection is mandatory only with an exact `P_A` and at k≥2.

**Unified tensor-Chebyshev `P_B` atom.** The `L_{k-1}⁻¹` inside `P_B` is a fixed-degree
Chebyshev iteration with a tensor smoother, matrix-free, bottoming out at the near-exact
k=0 atom (κ≈6): k=1 inverts `L_0` directly; k=2 inverts `Ŝ_1` with the k=1 Hodge
preconditioner as smoother, whose inner `L_0⁻¹` is a nested cheb-`L_0`. Degree auto-set
from a matrix-free Lanczos `κ` estimate (`d ≈ ½√κ·ln(2/ε)`). Constant-deflating the
nested free `l_0` was the fix that dropped `cond(P_hodge·L_1)` 152→9. Results (RE, p=3,
free): k=1 ns 6,12,4 = 49 it / 204 ms (vs jacobi 551/511); ns 10,20,6 = 71 it / 534 ms
(jacobi stalls at 600). h-flat degree ~7.

**k=2 / k=3.** k=2 projected `P_A(cap)+P_B` is h-flat in iters with a near-exact atom
(33→34 across grids, ~10× jacobi) but ~3.3× slower on wall (deep nesting); projection
mandatory; capping `P_A` on the curl null is mandatory (`pinv(S_2)`); `P_B` needs the
full `L_1` inverse, not bare `K_1` (cond 9 vs 236). k=3 stays on jacobi (`L_3` tiny;
the recursion closes but loses decisively). Open lever for a k=2 wall win: a cheaper
near-exact `L_1` inverse than "polynomial over the k=1 preconditioner" — a multilevel
vector-Laplacian solver.

## Nullspace robustness (k=0 and k=1, both BCs)

The tensor preconditioners stay effective when the operator is singular (free BC, the
harmonic deflated in the Krylov solve), confirmed on `rotating_ellipse` / `toroid`:

| degree / BC | nullspace | jacobi | tensor |
| --- | --- | --- | --- |
| k=0 dbc | 0 | 156 it | **16 it** |
| k=0 free | 1 (constant) | 222 it | **21 it** |
| k=1 dbc | 0 | 386 it | **96 it** |
| k=1 free | 1 (harmonic) | 546 it | **168 it** |

Saddle nullspace handling: `solve_saddle_point_minres` deflates the harmonic via
`vs_upper`/`mass_upper_matvec`. **Timing caveat:** `apply_laplacian_preconditioner(kind="jacobi")`
re-probes `diag(L_k)` on every call (an `O(n)` stiffness sweep traced into the CG loop);
precompute the diagonal once for fair timing.

## Baselines (k=1, are tensor methods worth it?)

A fixed-degree Richardson/Chebyshev on the approximate Schur `Ŝ` with the Jacobi
diagonal smoother is a valid MINRES preconditioner. On iterations, `chebyshev-8` (67 it)
*beats* the tensor method (96) — so the tensor atom is not fundamentally
better-conditioning. On **wall** the tensor method wins ~3×: its apply costs ~one `Ŝ`,
Chebyshev-`d` costs `d`. The tensor preconditioner's value is its cheap per-apply cost.

---

# Diagnostics and commands

Canonical k=1 harness: `scripts/benchmark/benchmark_graddiv_k1_preconditioner.py`
(`--klevel {0,1,2,3}`; `--pa-grad-project`; `--k1-both-bc`), driven by
`slurm/job_diag_graddiv_pa_compare.sh`.

k=0 atom spectrum/diagnosis: `scripts/debug/debug_rank_ritz_spectrum.py` — dense
`eig(smoother∘L_0)` with the bulk-inverse variants that established the radial-dense
result: `--dbc` (outer clamp → no outlier), `--exact-bulk` (true dense bulk → κ=1),
`--model-exact-inverse` (separable model exact → splits approximation vs Lynch error),
`--radial-dense` (the fix), `--radial-dense-prod[ --radial-dense-prod-rank N]` (validate
the production build), `--rt-coupled --rt-band B` (rank-(2B+1) θ band test), plus a
bulk-pencil probe and Rayleigh / Fourier-`m` readouts on the `λmin` eigenvector.

k=0 rank report: `scripts/debug/debug_radial_dense_ranks.py` — metric (r,θ)/ζ separable
ranks and radial-block rank/cond/cross-mode variation.

```bash
# k=1 current best (projected P_A + P_B vs jacobi; dense couplings are prod default)
python scripts/benchmark/benchmark_graddiv_k1_preconditioner.py \
  --pa-grad-project --system saddle \
  --methods "jacobi (diag),P.T P_S P + P_B (fused)"

# k=0 radial-dense diagnosis + validation
python scripts/debug/debug_rank_ritz_spectrum.py --geometry toroid --ns 6,12,4 --p 3 --ranks 1,2,3
python scripts/debug/debug_rank_ritz_spectrum.py --geometry toroid --ns 6,12,4 --p 3 --ranks 1 --radial-dense
python scripts/debug/debug_rank_ritz_spectrum.py --geometry toroid --ns 6,12,4 --p 3 --ranks 1 --radial-dense-prod --radial-dense-prod-rank 2
python scripts/debug/debug_radial_dense_ranks.py --geometry toroid --ns 6,12,4 --p 3 --rank 2
```

# Current policy

- **Ship k=0 and k=1.** k=0: the unified tensor-Cheb `L_0` atom matches the exact
  inverse, h-flat degree ~7, ~2× wall vs jacobi, converges where jacobi stalls; ship the
  constant-deflation of the `k=0` apply too. The **radial-dense** bulk inverse is the
  recommended atom when tighter conditioning is wanted (removes the free-BC outlier,
  `κ≈2`); the separable FD default stays at rank-1.
- **Tensor rank stays at 1** for the separable atom; **radial-dense** is the fix for the
  free-BC outlier (radial rank is a free knob; metric fit is a sequential SVD, not CP-ALS).
  Rank-2 θ (block-tridiagonal) was tested and is *not* worth it — rank-1 θ is the sweet
  spot (residual is a benign cluster, not an outlier).
- **Mass preconditioners need no change** — a rank-2 metric is an exact 2-term Kronecker
  sum, inverted by the existing simultaneous diagonalization (all k, scalar / diagonal-metric
  vector).
- **k=2** is degree-scalable, matrix-free, a robustness win, but ~3.3× slower than jacobi
  on wall; the open lever is a cheaper near-exact `L_1` inverse (multilevel). **k=3 stays
  on jacobi.**
- Keep Jacobi Schur outer as the reliability baseline for k=2/k=3.
- If the projected blend is promoted to production, define acceptance criteria up front
  (clear wall-time win across the target family, stable iteration behavior, clean
  nullspace integration).
