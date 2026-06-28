# Greville → production refactor — handoff

**Goal:** make the Greville-collocation preconditioner the production atom everywhere
and delete the CP/tensor-fit machinery, behind a compact API. Staged, GPU-verified,
CP preserved on a branch. See `docs/preconditioner_plan.md` for the *why* (the route,
the D0 theorem, why greville = robust production atom). This file is the *where we are*.

## Branches
- **`greville-prod`** — the working branch (all stages below). Branched from `main` (`c0e616a`).
- **`cp-tensor-archive`** (`04ed96b`) — snapshot with the full CP machinery intact, before any
  deletion. `git checkout cp-tensor-archive` to return to CP. **Not pushed** — consider
  `git push -u origin cp-tensor-archive` for durability. CP is also recoverable from history.

## Target API (the end state)
Replace the `cp_kwargs` sprawl with a compact surface on the preconditioner assemblers/apply:
- `kind ∈ {"jacobi", "greville"}` — the atom (drop `tensor`/CP; `none` only for baselines).
- `chebyshev: int` — degree, 0 = off (optional smoother on top; hardcode the Lanczos sub-knobs).
- `bulk_schur: bool` — default **off**. k≥1 only: nested 3×3 Schur across the (r,θ,ζ) vector
  components (on) vs omit off-diagonals (off). Already renamed/wired (stage 3).
- structural: `ks`, `dirichlet`.

Production greville config (locked, see plan §D4/§E): **combined** weight mode, **geometric**
$D$ = cbrt(channels), **arithmetic** α. Mass = $D^{-1/2}(M_0^{-1}\otimes)D^{-1/2}$; Laplacian/
stiffness = $D^{-1/2}\cdot$ additive-FD (`_fd_apply_3d`) $\cdot D^{-1/2}$.

## DONE — committed on `greville-prod`
| commit | stage | what |
|---|---|---|
| `46e601f` | 1 | greville **mass** k=0–3 production-integrated (alongside CP). `_build_greville_mass_block_factors` (preconditioners.py), greville fields on `TensorDiagonalBlockInverseFactors`, sandwich apply. |
| `f3593cf` | 2 | greville-backed **Schur-Jacobi probe** for vector Laplacians — NO code (the `tensor_probe` probe reads `mass_preconds.tensor`, operators.py ~6934, which is now greville). Verify scripts only. |
| `04ed96b` | 3 | rename inner-Schur → **`bulk_schur`** (one flag, default off). Verified live: ~30–45% on W7-X, inert on toroid (orthogonal metric → no coupling). |
| `14ac736` | 4a | delete experimental greville Laplacian variants (radial_dense/sylvester/pencil/pair_d, geom/minimax α). |
| `ccb502e` | 4b | **greville is the production DEFAULT** for mass + k=0 Laplacian. Deleted the k=0 CP-FD path + dead schur modes (kept only `tensor_probe`). Trimmed `K0TensorHodgePreconditionerFactors` to 11 fields. **CP core + stiffness preconditioner KEPT** (P_A substrate). −641 lines. Verified all paths converge on toroid+w7x with the new defaults. |

Verified iteration counts (p3, 12³, new defaults, no `greville` kwarg):
- mass k0–3: cyl 7–10, toroid 7–11, w7x k0/k3 7–10 / k1/k2 55–62.
- k=0 Laplacian: toroid 34/50 (dbc/free), w7x 66/97.
- vector Laplacian (greville mass + probe): toroid 454–838, w7x 1070–3380. All converge.
- greville vs tensor on the Laplacians: **tie** at p3; greville's edge is robustness (k=0 W7-X *free* where CP stalls; high-p W7-X vector mass where CP NaN'd).

## IN PROGRESS — step 3: greville P_A (k=1 curl-curl / k=2 div-div stiffness atoms)
**Code written but UNCOMMITTED→WIP-committed, and UNVERIFIED.** (committed as WIP right after this
doc — see `git log`; marked clearly.) Built by a fork; needs GPU spectral validation before trusting.
- New: `_build_greville_stiffness_block_factors(seq, *, k, shape, diff, comp)` (preconditioners.py
  ~2484, next to the mass builder). Additive-FD on unweighted atoms + $D^{-1/2}$ sandwich.
- Apply: extended the greville branch in `_apply_tensor_diagonal_block_preconditioner`
  (preconditioners.py ~795): `greville_inv_r` set → product sandwich (mass); else → D-sandwiched
  additive FD ($D^{-1/2}V\,\mathrm{diag}(1/\mathrm{denom})V^\top D^{-1/2}$) for stiffness.
- Reuses existing `fd_V_*`/`fd_lam_*`/`fd_inv_denom` fields + `greville_inv_sqrt_D` (NO new fields).
- Wiring: `assemble_tensor_stiffness_preconditioner(cp_kwargs={'greville': True})` → ternary at all
  6 block sites (k1 arr/θ/ζ, k2 r/θ/ζ). CP is still the default here (greville opt-in).
- **k=2 deviation from scope:** the fork used the unified additive-FD form for k=2 too (single
  Kronecker term; singular stiffness deflated via `_modal_regularized_inverse_denom`) instead of
  the product+pinv the scoping suggested. Cleaner (one apply path) but review.

**⚠️ MUST VALIDATE before trusting (fork's flags):**
1. **k=1 cross-channel weighting** (curl structure: K_b←channel c, K_c←channel b). Read directly
   only for the *arr* block; θ/ζ blocks **inferred by curl symmetry** — validate per-block iters.
2. k=1 geomean-$D$ + arithmetic-α is an approximation (2 channel weights, one $D$) — validate on **W7-X**.
3. Unified diff/stiff-axis rule (stiff = primal axes) — confirm k2 θ/ζ and k1 θ/ζ blocks converge.
4. k=2 (D=1/J, α=1, single channel) should be clean.

**How to validate:** assemble greville stiffness (`cp_kwargs={'greville':True}`), solve the k=1,2
stiffness system, compare iters to jacobi AND to the CP stiffness preconditioner (still present) for
parity, on toroid + w7x. Mirror `scripts/debug/greville_stage4b_verify.py` (use
`apply_stiffness_tensor_preconditioner` + a stiffness matvec / `apply_stiffness`).

## REMAINING
- **Finish step 3:** validate greville P_A per above; fix the inferred k=1 θ/ζ weighting if iters are
  bad; commit a verified version.
- **Step 4 — final CP-core deletion** (the "all CP gone" milestone). Once greville P_A is validated:
  flip `assemble_tensor_stiffness_preconditioner` to greville-default; delete the shared CP core
  (`_greedy_cp_terms`, `_cp_als_3tensor`, `_build_tensor_block_factors_from_terms`,
  `_build_diagonal_tensor_block_factors`, `_build_mass_referenced_tensor_block_factors`), the CP
  fields on `TensorDiagonalBlockInverseFactors`, **Richardson** (`richardson_steps/omega` + applies),
  and **Chebyshev sub-knobs** (keep only the degree; hardcode lanczos iters/inflation/deflation/
  floor/seed). Verify stiffness (greville) + the HX benchmark still run.
- **Step 5 — compact API.** Consolidate `cp_kwargs` → top-level `kind`/`chebyshev`/`bulk_schur`;
  prune apply-side `kind` strings (`tensor`/`auto`/`richardson`); update callers.
- **Cleanup:** 8 now-unreachable CP-mass `else`-branches left in `build_mass_tensor_preconditioner`
  (4b hardwired `greville=True` above them); a stale `radial_pencil_d` comment in
  `_apply_k0_tensor_hodge_bulk_shared_inverse`'s vicinity (that function was deleted in 4b — recheck);
  update `docs/preconditioner_plan.md` to record production integration is done.

## OUT OF SCOPE (future, by user's call)
- **HX-on-greville proper** (the P_A+P_B auxiliary-space combination). "We'll look at HX-style after."
  This refactor just makes greville the atom everywhere and removes CP. Current production vector
  Laplacian stays **greville mass + probed Schur-Jacobi** (works, stage 2). HX = 1-to-1 atom swap later.

## Environment / workflow notes
- venv python: `/kfs3/scratch/tblickhan/mrx/.venv/bin/python` (system `python3` is too old for `match`).
- **No CPU solves** — verify on GPU via slurm. Pattern: `sbatch --partition=gpu-h100
  --account=extremedata --gpus-per-node=1 --cpus-per-task=32 --mem=128G`, env
  `XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128 OMP_NUM_THREADS=8 ...`. (debug-gpu caps 2 jobs.)
- Verify scripts: `scripts/debug/greville_mass_prod_verify.py`, `greville_vec_laplacian_verify.py`,
  `greville_bulk_schur_verify.py`, `greville_stage4b_verify.py`, `greville_vs_tensor_vec_lap.py`.
  Build the sequence via `benchmark_graddiv_k1_preconditioner.build_sequence(cfg)`; solve info is a
  signed scalar (negative=converged, |info|=iters).
- Each stage: implement → `py_compile mrx/operators.py mrx/preconditioners.py` → GPU-verify → commit.
