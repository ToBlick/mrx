# Greville → production refactor — handoff

**Goal:** make Greville collocation the production preconditioner atom everywhere and
delete the CP/tensor-fit machinery, behind a compact API. See
`docs/preconditioner_plan.md` for the *why*; this file is *where we are*.

## Branches
- **`greville-prod`** — working branch, from `main` (`c0e616a`).
- **`cp-tensor-archive`** (`04ed96b`) — full CP machinery, pre-deletion. Not pushed (also
  recoverable from history).

## Target API (end state)
Compact surface replacing the `cp_kwargs` sprawl:
- `kind ∈ {"jacobi", "greville"}` — atom (drop CP/`tensor`; `none` for baselines).
- `chebyshev: int` — degree, 0 = off.
- `bulk_schur: bool` — default off; k≥1 nested 3×3 Schur across (r,θ,ζ).
- structural: `ks`, `dirichlet`.

Production greville config (plan §D4/§E): **combined** weight, **geometric** D=cbrt(channels),
**arithmetic** α. Mass = D^{−1/2}(M0^{−1}⊗)D^{−1/2}; stiffness = D^{−1/2}·additive-FD·D^{−1/2}.

## DONE on `greville-prod`
| commit | what |
|---|---|
| `46e601f` | greville **mass** k=0–3 production-integrated (alongside CP). |
| `04ed96b` | inner-Schur → **`bulk_schur`** (one flag, default off). ~30–45% W7-X, inert toroid. |
| `14ac736`/`ccb502e` | greville is the **production default** for mass + k=0 Laplacian. Deleted k=0 CP-FD path + dead schur modes (kept only `tensor_probe`). CP core kept as stiffness substrate. |
| `e5fd04f` | greville **P_A stiffness** atom (k=1 curl-curl / k=2 div-div), opt-in `cp_kwargs={'greville':True}`. **VERIFIED — see below.** |

## Step 3 — greville P_A stiffness atom: VERIFIED (raw)
GPU, p3, 12³. `greville_k1_stiffness_verify.py` (raw atom).
- **k=2 div-div: greville wins everywhere** — toroid 15/17, W7-X 16/18 vs cp ~37 (dbc/free).
  Production-ready.
- **k=1 curl-curl raw: ties cp on toroid** (43/39); **on W7-X greville AND cp both fail** (raw
  block-diagonal P_A leaks onto gradients). Not a greville-only defect — but see the atom
  diagnosis: greville is a *cruder* fit than cp on W7-X and that bites inside HX too.

## HX (P_A + P_B) — k=1 result (development; prod default stays Schur-diag-Jacobi)
`M_HX = (I−Π_g)·P_A·(I−Π_g*) + P_B`, `P_B = G0 L0⁻¹ M0 L0⁻¹ G0ᵀ`, `Π_g = G0 L0⁻¹ G0ᵀ M1`,
`L0⁻¹` = **cheb-L0** (constant-deflated free BC). Deflates **gradients** (Π_g) + **harmonic
forms** (`vs`). Built in `greville_k1_hx_verify.py` (library tensor P_A, greville + cp). Verified:
- **Toroid: HX works, ~40 it, greville TIES cp** (dbc 43/40, free 39/40). raw P_A (cp & greville)
  fails 4000 it → confirms raw was the wrong test; **greville P_A is fine in HX on benign geometry.**
- **W7-X: HX does NOT fully converge in 4000 it.** HX-cp res 0.013/0.0015 (dbc/free) — big gain
  over raw 0.6/0.33 but stalls (worse than unprecond. none 2e-7/3e-6). **HX-greville NaN (dbc) /
  0.42 (free).** Two causes (see below): rough inner L0⁻¹ on W7-X, and the greville atom defect.

## greville k=1 atom — diagnosis (task #2)
The cross-channel weighting is **correct** (verified: comp's two stiff axes b,c get `∂_b²←β_cc`,
`∂_c²←β_bb`, β=g_aa/J — the 2-form curl energy). The W7-X metric is **mostly diagonal** (NOT
off-diagonal — earlier claim was wrong); the hard part is that the diagonal entries vary **high-rank
in the θ-ζ plane** like `cos(θ−nfp·ζ)` (~60% angular spread, [[metric-weight-separability-rule]]).
The atom collapses that high-rank weight field to a **rank-1 constant in space** (single geomean
`D=√(β_bb β_cc)` + scalar `α=mean(√(β_cc/β_bb))`). This rank-1 collapse is the real limitation —
**but k=0 does the same collapse and survives** (conditions to ~68, SPD), so it does not by itself
explain why k=1 NaNs. The k=1-specific failure mode is **not yet pinned**; leading suspects: curl-curl
is *singular* (atom is its pseudo-inverse → a mis-scaled rank-1 weight misplaces the near-kernel),
and the *cross-channel* pairing of two different high-rank fields (geomean `D` collapses badly where
β_bb, β_cc are out of phase). Cheap probe to settle it: densify greville P_A on W7-X and inspect its
eigenvalues (indefinite vs near-singular vs inf/nan; bulk vs surgery Schur).

## W7-X angular atom — design (the next step)
**FD invertibility constraint (user):** a single per-axis eigenbasis can carry **either**
different-per-axis scalar weights (`α_rr,α_θθ,α_ζζ`) **or** one spatially-varying weight (a single
pointwise `D` sandwich, common to all axes) — *never both*. The current atom straddles (geomean
`D(x)` + ratio `α`); rank-1 best-effort, fine on torus, inadequate on W7-X. Three tiers to fix:
1. **Band-aid (stay in FD):** drop the straddle — pick spatially-constant per-axis `α` *or* a single
   varying `D` with equal `α`. Clean SPD atom (likely no NaN), but W7-X stays mediocre.
2. **θ-ζ out of FD:** FD in r (benign), solve the θ-ζ cross-section directly per radial eigenmode —
   there you may have both varying & anisotropic weights. Cost: a θ-ζ block solve (`nθ·nζ`) per r-mode.
3. **(recommended) Fourier-banded** — the W7-X variation is `cos(θ−nfp·ζ)`, high-rank in (θ,ζ) but
   ONE helical mode → in Fourier `(m_θ,m_ζ)` it couples only neighbours → the θ-ζ block is
   banded/tridiagonal → cheap. (Or transform to helical `u=θ−nfp·ζ` → 1-D → separable, FD works.)
   **Also lifts k=0** (same rank-1 collapse, only conditions to ~68 on W7-X) — general W7-X atom.
**Reality check DONE** (`scripts/debug/w7x_metric_fft.py`, job 14796148, `logs/w7x_fft_14796148.out`):
2-D FFT of β=g_aa/J over (θ,ζ) at bulk radii. Result: **β_θθ, β_rr vary ~55%** (confirms the wall),
dominated by the **(m_θ=±2, m_ζ=±1)** mode (W7-X m=2 bean shaping, n=1/period — NOT the (±1,∓nfp)
I'd guessed); **~70–78% of AC in that one ± pair, ~4–7 modes reach 90%** → low-rank but not rank-1.
**β_ζζ is nearly flat (~10%)** → ζ-weight can stay constant; only θ/r weights need the angular term.
So: tier-3 Fourier-banded is viable with a **moderate** band (≈|m_θ|≤3,|m_ζ|≤2 for ~90%), not band-1;
tier-2 (dense θ-ζ per radial mode) is the safe option and cheap (nθ·nζ modest, captures all modes).
A definiteness guard on the current atom would meanwhile make greville stall (not NaN) like cp.

## DONE — nullspace solver (mrx/nullspace.py)
- `_nullspace_shifted_preconditioner` k≥1: Richardson-1 outer → **prod default tensor + jacobi
  Schur outer** (reads stored `schur_diaginv`, else probes once). Faster inverse iteration.
- k=1 inverse-iteration guess: physical `1/R e_ζ` → **logical (0,0,1)** via `seq.load(...,
  frame='ref')` (geometry-robust; coincides with 1/R only on a torus). Validated on toroid (k=0
  const + k=1 harmonic correct). W7-X validated indirectly via the HX run.

## DONE — W7-X dense matrices + conditioning + visualization
`scripts/debug/w7x_dense_matrices.py` / `_hi.py` / `w7x_dense_derivatives.py` (GPU): dense M0–M3,
K0–K2 (K3≡0), L1–L3, D0–D2 → `outputs/w7x_matrices/{dbc,free}/*.npy`; κ in `kappas.txt`,
`kappas_hi.txt`. κ (dbc uncond/jacobi/best): M0 1.4e4/6.2e3/**1.85**, M1 5.6e6/4.5e3/**78**,
M2 1.9e6/2.0e3/**112**, M3 3.7e3/410/**1.39** (greville mass); K0 2.2e3/609/**68** (greville
Hodge); K1(L1) & K2(L2) best == jacobi (tensor_probe Schur diag = diag(L); greville lives in the
operator). Plots: `scripts/plotting/w7x_matrix_fill.py` (per-matrix fill + log-mag, violet
surgery/component lines) + `w7x_saddle_fill.py` (saddle `[[S_k,D],[Dᵀ,−M]]`, only the 2×2 line).
Surgery cores: k0/k3 form 3·nz, k1 form 5·nz, k2 form 2·nz; 1-form comps 2940/5532, 2-form 2616/5496.

## Remaining
- **greville k=1 atom**: add the W7-X angular term (non-separable) and/or a definiteness guard;
  re-test HX-greville on W7-X. Optional: tighter inner cheb-L0 (eps 1e-3/1e-4) to test whether the
  HX-**cp** W7-X stall is just inner-L0⁻¹ accuracy vs a deeper P_A/angular-wall limit.
- **Step 4 — final CP-core deletion** once greville stiffness is trusted on the production path:
  flip stiffness default to greville; delete CP core (`_greedy_cp_terms`, `_cp_als_3tensor`,
  `_build_*_tensor_block_factors*`), CP fields, **Richardson**, **Chebyshev sub-knobs** (keep degree).
- **Step 5 — compact API:** `cp_kwargs` → top-level `kind`/`chebyshev`/`bulk_schur`.
- Cleanup: dead CP-mass `else`-branches in `build_mass_tensor_preconditioner`; stale comments.

## Uncommitted (this session — `greville-prod` working tree, HEAD `cf0548a`)
Not yet committed; review before committing.
- **`mrx/nullspace.py`** — PRODUCTION change: nullspace solver → tensor+jacobi outer; k=1 guess →
  logical (0,0,1). Validated on toroid; W7-X only indirectly. (`docs/{handoff,preconditioner_plan}.md` edited.)
- New scripts (untracked): `scripts/debug/{greville_k1_hx_verify,w7x_dense_matrices,
  w7x_dense_matrices_hi,w7x_dense_derivatives,w7x_metric_fft}.py`,
  `scripts/plotting/{w7x_matrix_fill,w7x_saddle_fill}.py`.
- Outputs (gitignored/large): `outputs/w7x_matrices/` (npy, kappas, figures).
- In-flight job: `14796148` (metric FFT reality check) → `logs/w7x_fft_14796148.out`.

## Environment
- venv: `.venv/bin/python` (system python3 too old for `match`).
- **No CPU solves / CPU too slow at 12³** — GPU via slurm: `--partition=gpu-h100
  --account=extremedata --gpus-per-node=1 --cpus-per-task=32 --mem=128G`, env
  `XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=256 OMP_NUM_THREADS=8`.
- Scripts in `scripts/debug/`: `greville_k1_stiffness_verify.py` (raw atom), `greville_k1_hx_verify.py`
  (HX, calls `compute_nullspaces_iterative(betti=(1,1,0,0))` to populate the real k0/k1 nullspaces —
  `assemble_operators` leaves zero placeholders that silently break free-BC deflation),
  `w7x_dense_matrices*.py`, `w7x_dense_derivatives.py`. Plotting in `scripts/plotting/`.
  Sequence via `benchmark_graddiv_k1_preconditioner.build_sequence(cfg)`; solve info signed
  (negative=converged, |info|=iters).
