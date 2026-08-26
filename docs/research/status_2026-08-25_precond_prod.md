> **Status:** superseded by audit_2026-08-25_production.md
> **Read this for:** the state of the precond-prod branch on 2026-08-25
> **Do not read for:** anything current; the branch is merged

# Status for redistribution — `precond-prod`, 2026-08-25

Agent working in `.claude/worktrees/poincare`, branch **`precond-prod`**
(renamed from `worktree-poincare`). 26 commits, **local only, never pushed**.
Based on `greville-prod` at `7e89525`; that branch has since moved **+16
commits** (relaxation-IC work, s=3 decision, docs), so a rebase is owed.

Two jobs got mixed into one branch: a Poincaré field-line tracer, and the
production preconditioner consolidation. The agreed plan is to split them:
`greville-prod -> precond-prod -> poincare-tracer`, with `mrx/poincare.py` and
everything `poincare_*` on the tracer branch.

---

## 1. What is verified

| item | evidence |
| --- | --- |
| Explicit-preconditioner API | pytest 199 passed / 2 skipped / 4 xfailed |
| Stage A (`probed_jacobi`, `exact_jacobi` deleted) | pytest 199 passed |
| Stage B (3303 lines of retired tensor/surgery deleted) | pytest 199 passed, **identical counts to before** |
| Exact sequence | `test_curl_of_grad_is_zero`, `test_div_of_curl_is_zero`, `test_polar_complex_is_exact` |
| Nullspaces | `test_stored_nullspace_vectors_are_harmonic` + 4 more; harmonic Rayleigh ratio 1e-26 across 6 geometries |
| k=1/k=2 pullbacks | max rel err **1.8e-15** over 4 (k, BC) pairs on 3 geometries |
| Poisson, analytical, nbc_k0 | order ~4.5 at p=3 (expect 4), ‖curl‖ 1e-16 |
| Poisson, analytical, k1 | order ~4.7 at p=3 |
| k=0 atom pays for itself | 311→74 (dbc), 584→114 (free) iterations, 0.25 s assembly |
| Atom coarse-grid floor | `n >= p + 2`, measured across k and both BCs |

## 2. What is NOT verified

* **The rebase onto current `greville-prod` has not happened.** Every number
  above is on the old base. Their `mrx/derham_sequence.py` hunk is at 393-402
  and mine at ~459, and they touch `mrx/projectors.py` which I do not, so it
  should be clean — but "should" is not "is".
* **The split has not happened**, so neither branch has been tested alone.
  A green suite on the combined tree says nothing about `precond-prod` by
  itself.
* **Stages C and D are not started** (see §5).

## 3. Findings that affect other people's work

### 3.1 The relaxation loop runs its innermost solve unpreconditioned

`apply_leray_projection(u, k=2)` -> `apply_inverse_hodge_laplacian(div_v, 3,
dirichlet=True)`, called once per force evaluation (`relaxation.py:48`) AND once
per Picard iteration (`:305`). `compute_helicity` (`:13`) adds a k=1 dbc solve.
Nothing on the relaxation path assembles the block-Jacobi atom, and
`assemble_block_jacobi_laplacian_preconditioner` is called from exactly two
places in the tree: `mrx/nullspace.py` and one test.

Until 2026-08-25 that silently substituted a jacobi diagonal (~2.5x the
iterations). It now runs with **no** preconditioner, by design — the fix is one
`seq.set_map_and_preconditioners(map)` at relaxation setup. **Not done here**
because it changes an unmeasured production hot path. Audit item 3.1 from the
other side.

### 3.2 `test_torus_poisson_dbc_k2_sparse` does not converge — pre-existing

Relative L2 error **1.7818 / 1.7819 / 1.7819** at n = 6 / 8 / 10. Flat. MINRES
reports `converged=True` at every n and the harmonic form is clean (‖L₂h‖
4.9e-10, curl 7.7e-12, div 1.7e-12).

Not caused by the repoint: the only edit was deleting the tensor mass assembly,
and `apply_inverse_laplacian` at k=2 takes the EXACT saddle path, where
`M_{k-1}` is a block and is never inverted — so the mass preconditioner cannot
change the converged answer. (The one path where it could is
`apply_hodge_laplacian_approx`, which uses the mass preconditioner as the inner
inverse of the weak term and so changes the OPERATOR; this script does not use
it.) Looks like `w2_exact` carrying a harmonic component the deflated solve
necessarily removes.

**Caveat: this is reasoning plus a mechanism check, not a measurement.** I tried
to run the same study on `greville-prod` for a direct before/after, and the
harness refuses commands referencing a second worktree. One run there settles
it.

### 3.3 The convergence study was measuring a retired preconditioner

All nine `scripts/config_scripts/test_torus_poisson_*_sparse.py` hard-coded
`kind='tensor', surgery_schur=True` plus `assemble_tensor_mass_preconditioner`
— retired 2026-08-17, replaced again 2026-08-22. **Any convergence number ever
published from them describes the retired stack, not production.** Repointed at
`'auto'` here. Audit item 3.3.

### 3.4 The Laplacian atom cache was not invalidated by `set_map`

The mass cache keys on geometry identity; the Laplacian atom cache was a plain
`{(k, BC): atom}` dict. `set_map` after assembly left it factoring the previous
metric — silently, as slow convergence. `set_geometry` now drops the atoms.
Nothing in `mrx/` triggered it, but any geometry-updating loop (shape
optimisation, moving boundary) would have.

### 3.5 Two GVEC data files carry wrong metadata

* `axis_pert_dR5e-05_dZ3.75e-05.h5` and `interior_pert_dR5e-05_dZ3.75e-05.h5`
  declare **`nfp = 2`**; their R/Z is `quasr0044970` (nfp=**3**) shifted by
  exactly the amplitudes in their own filenames. Confirmed three ways: the
  offsets, the det DF range, and the resulting iota. `GVEC_NFP_OVERRIDE`
  corrects it in code, but **fix it at source** — nfp=2 wraps one field period
  through 180° instead of 120°, a different domain behind a healthy Jacobian.
* `w7x_ini_mrx.h5` declares `axis_radial_index = 49`; the axis is at rho[0]
  (mean theta-extent 3.4e-3 there against 1.8 at rho=1).
* Four new files carry only `precomputed_*` sizes, not `n_rho`/`n_theta`/
  `n_zeta`. Loader handles both now.

### 3.6 Two geometry files produce folded maps

`quasr0065575` at ns=(12,24,12) has **det DF spanning [−0.236, +1.543]** — a
sign change, i.e. a fold, which no handedness choice fixes. `quasr0065530`
builds at ns=(8,16,8) and (12,24,12) p=3 and **degenerates at (16,32,16) and at
p=4**, so its R/Z carries a near-fold that a finer spline resolves into a real
one. Both raise at setup rather than solving on a negative Jacobian.

---

## 4. Files claimed (release on merge)

`mrx/{operators,preconditioners,nullspace,derham_sequence,poincare}.py`,
`mrx/experimental/{mass_surgery,tensor_stiffness,k0_core_schur}.py` (deleted),
`test/{test_block_jacobi_laplacian,test_preconditioners}.py`,
`test/experimental/` (deleted), all `scripts/config_scripts/test_torus_poisson_*`,
`scripts/debug/{poincare_*,k0_block_default,atom_*,harmonic_audit,gvec_geometry,
verify_block_jacobi,bench_real_solves}.py`, `conf/config_poisson_verify.yaml`,
`docs/research/handoff_2026-08-24_poincare.md`.

**Not claimed, untouched:** `mrx/relaxation.py`, `mrx/projectors.py`,
`mrx/block_jacobi_laplacian.py` (read only), everything under
`scripts/interactive/`, `scripts/benchmark/`, `scripts/deprecated/`.

## 5. Open work, in dependency order

1. **Rebase** onto current `greville-prod`, re-run pytest. (task #5)
2. **Split** into `precond-prod` -> `poincare-tracer`; 5 commits are pure
   production, 17 pure tracer/docs, and 3 entangled (`8abc899`, `93547b6`,
   `d4c759f` — each a `mrx/nullspace.py` change plus its counterpart print in
   the Poincaré driver, so they split at the file boundary). pytest each branch
   alone.
3. **`test_torus_poisson_k0_sparse` needs `assemble_incidence_operators`** —
   one line. The block atom probes the Laplacian (`probe_core_block` ->
   `apply_hodge_laplacian_approx`) and needs `G0`; the tensor path built its
   factors from the metric alone. Currently fails with
   `ValueError: Incidence operator G0 is required to apply K0`.
4. **Stage C** — replace `raw_kron` as `schur.inner` with the mass atom (the
   `block_jacobi` branch already exists in
   `_build_schur_apply_from_saddle_preconditioner`), then delete raw_kron
   (152 refs). **Note: raw_kron is not dead today** — the Schur operator is
   built before the outer branch is taken, so its factors are used even when
   the atom serves the apply. This changes the Schur OPERATOR, not just a
   preconditioner slot, so it needs its own convergence check, not just a green
   suite.
5. **Stage D** — rename `block`/`block_jacobi` -> `tensor`, leaving the three
   modes none/jacobi/tensor (~174 refs, 40 files, incl. class names,
   `BLOCK_JACOBI_CACHE_ATTR`, `assemble_block_jacobi_laplacian_preconditioner`,
   and the module `mrx/block_jacobi_laplacian.py`). The `surgery_schur` FIELD on
   `MassPreconditionerSpec` still exists with no consumers; remove it here,
   since dropping a dataclass field touches every spec construction.

## 6. Poincaré results (complete, no further work needed)

`outputs/poincare_final/2026-08-24/21-26-16/` — 24 of 25 cells, 96 figures,
mirrored to `/kfs3/scratch/tblickhan/mrx/outputs/`. Full account in
`docs/research/handoff_2026-08-24_poincare.md`. Known answers reproduced: toroid
iota 0 to 1e-17, W7-X 0.851–0.948 (published vacuum range), rot-ellipse exactly
0 by a reflection symmetry the map forces. One open physics item: the k=1
natural-BC field is genuinely chaotic on the quasr44970/65530 family — not a
tracer defect (`B^zeta/|B|` stays above 0.77, and the step drift is flat across
ns 8/12/16, which rules out both a singular parameterisation and resolution).
