# Geometric Multigrid Preconditioner for the k=0 Laplacian (prototype)

**Status (2026-07-07):** `scripts/debug/laplacian_mg_k0.py` and
`slurm/job_laplacian_mg_k0.sh` fully written. Phase 0 (cylinder 8×16×8
`--two-level-check`, CPU) PASSED the machinery gates but exposed a real
problem:

- All SPD gates pass (sym_err ~1e-17, min_rayleigh > 0, both BCs); jacobi
  einsum diagonal matches the probed diagonal to 9e-16; runs end-to-end.
- MG(fd): 8/9 it (dbc/free) vs baseline 24/26 — but on a cylinder the metric
  is exactly separable, so the FD atom is a near-exact solve; this validates
  the cycle wiring, not MG strength.
- **MG(jacobi): 62/94 it; MG(fdax): 62/77 it — far WORSE than baseline.**
  These smoothers actually exercise the coarse-grid correction, and
  `P_const_err = 0.63` (constant not reproduced in the ~p-wide axis-side
  layer of the radial window; plan called it "benign" — it is not). Prime
  suspect: transfer defect near the axis kills smooth-error correction.
  Second suspect: upper Chebyshev window `[λmax/4, λmax]` too narrow for a
  factor-2 coarsening with a plain Jacobi atom (lam_max jacobi 4.6–8.3,
  fdax 1.23).
- NEXT: fix the radial transfer (e.g. build P on the FULL radial basis incl.
  the two axis functions, then restrict rows/cols to the window; or
  renormalize P rows to reproduce constants) and/or widen `--cheb-window`;
  rerun Phase 0, expect MG(jacobi) ≲ baseline before moving to toroid.

Laptop note: cylinder/toroid/rotating_ellipse run on CPU out of the box; w7x
needs the fitted map data (gitignored `data/`) and a GPU — cluster only.
Repro: `python scripts/debug/laplacian_mg_k0.py --geometry cylinder --ns 8 16 8 --two-level-check`
(~2.5 min CPU).

## Context

The k=0 scalar Laplacian `K₀ = G₀ᵀ M₁ G₀` is solved by PCG with a single-level greville
fast-diagonalization (FD) preconditioner (`_assemble_k0_greville_bulk_factors`,
`mrx/operators.py:1380`, inside the core/bulk Schur envelope at `operators.py:1536/1603`).
Two problems: (1) single-level → h-dependent CG iteration growth; (2) a **structural
obstruction**: any single-sandwich FD form forces the θ- and ζ-summands to share their
radial weight profile, but the true profiles differ by ~r² (`w_θθ ~ 1/r` vs `w_ζζ ~ r`) —
so the FD atom's metric handling is *necessarily* some averaging of the g-factors.

Plan: a **geometric multigrid V-cycle** on the bulk (h-independence), with a **three-way
smoother comparison** that measures how the metric should enter the smoother:

- **jacobi** — Chebyshev over `diag(K₀)`: locally exact for all three `g_aa·J` weights, no
  separability assumption.
- **fd** — Chebyshev over the production atom as-is: `D = J·(g^rr g^θθ g^ζζ)^{1/3}` at
  Greville points + global per-term scalars α (the "1/3 version").
- **fdax** — NEW axis-averaged FD: `D = J` at Greville points; per axis, generalized
  eigendecomp of the pair `(M_a, K_a[ḡ^aa(x_a)])` with `ḡ^aa(x_a)` = quad-weighted mean of
  `g^aa` over the other two axes; denom `λ_r+λ_θ+λ_ζ`, **no α scalars**. Captures each
  g-factor's variation along its own axis (g_rr in r, g_θθ in θ, g_ζζ in ζ — the W7-X
  shaping/helical variation); the cross-axis part (e.g. g^θθ ~ 1/r²) is the measured
  residual — structurally unavoidable. Exactly diagonalizable: one mass + one weighted
  stiffness per axis.

Decisions made: full 3D coarsening default with per-axis knob; prototype script only
(production wiring deferred); single-level production FD stays as the baseline to beat.

## Files

- **NEW** `scripts/debug/laplacian_mg_k0.py` — the prototype (all logic in-script; no
  `mrx/` changes this pass).
- **NEW** `slurm/job_laplacian_mg_k0.sh` — per-geometry sbatch on `gpu-h100s` /
  `extremedata` (W7-X: `XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128`), CSVs to
  `outputs/laplacian_mg_k0/<stamp>/`.

## Architecture

MG acts only on the **bulk** block `A = A_bb` (embed bulk vector with core=0 →
`apply_stiffness(seq, ops, ·, 0, dirichlet)` → slice `[core:]`; core = `3·nz`,
`nr_bulk = nr − 2 − int(dirichlet)`, radial window **starts at 2 always**; the dirichlet
flag drops the LAST radial function — slice `P_r_full[2:2+nr_bulk_f, 2:2+nr_bulk_c]`).

### Levels (rediscretized, geometry reused)
Coarse level = new `DeRhamSequence(ns_c, (p,p,p), 2p, TYPES, polar=True)` +
`evaluate_1d()` + `assemble_reference_mass_matrix()` + `set_map(fine_seq.map)` (reuses the
fitted W7-X map; jacfwd only at coarse quad points, ~1/8 cost) +
`assemble_incidence_operators(ks=(0,))` + `assemble_laplacian_operators(ks=(0,))` + eager
`apply_stiffness` warm-up per BC (mass-core cache / tracer guard). Recipe:
`scripts/benchmark/benchmark_k0_rank_geometries.py:200-219`.

`coarsen_ns` (knob `--coarsen fr,ft,fz`, default `2,2,2`; `--levels`, default 2):
- periodic θ/ζ: `n_c = n_f // f`, floor `max(4, p+1)`; W7-X ζ floor `2·nfp` (checked
  empirically by a `--zeta-diag` mode: count oscillation periods of `a_zz(r₀,θ₀,ζ)`).
- clamped r: halve elements — `e_c = max(2, ceil((n_f−p)/2))`, `n_c = e_c + p`, floor
  `n_c ≥ 5` (dbc). Non-nesting is fine (transfers are quasi-interpolation).

### Transfers (tensor-product 1D, R = Pᵀ)
`P_axis = solve(C_ff, C_cf)` where `C_ff = fine.collocation_matrix(fine_greville)` (square,
Schoenberg–Whitney) and `C_cf = coarse.collocation_matrix(fine_greville)`
(`spline_bases.py:136`; periodic wrap already handled). Exact injection on nested periodic
axes; O(h^{p+1}) quasi-interpolation on the non-nested radial axis. Radial slicing per the
window rule above. Apply = 3 einsums (mirror `_fd_apply_3d` style). Known benign defect:
window truncation breaks exact constant reproduction in a ~p-wide layer at the axis —
report `‖P·1_c − 1_b‖_∞` per level (`P_const_err`), accept unless Phase-1 shows otherwise.

### Smoothers (`--smoother {jacobi,fd,fdax}`)
All three wrapped in symmetric Chebyshev on `[λ_max/window, λ_max]` (`--cheb-window 4`,
`--smooth-steps m` default 2): λ_max per level from the existing Lanczos helper
(`_estimate_chebyshev_lanczos_bounds_apply`, `preconditioners.py:902`) with
constant-mode deflation via `orthogonal_vectors` for free BC; **discard the Lanczos λ_min**.
Local ~30-line copy of the Chebyshev body (`preconditioners.py:1757`) returning
`(x, residual)` — saves 1 A-apply per smoothing pass.
- jacobi: `diag(K₀)` bulk slice via the `diag_EAET_direct` pattern used by
  `assemble_mass_jacobi_preconditioner` (`operators.py:1707`) on the sparse
  `G₀ᵀ M₁_sp G₀` product (reuse the scalar-hodge 'jacobi' kind if already wired).
- fd: the level's `_assemble_k0_greville_bulk_factors` + `_fd_apply_3d` (eps=0 null
  threshold active).
- fdax: in-script builder using `_assemble_weighted_1d_stiffness` + 1D generalized
  eigendecomp (`_assemble_1d_fd_eigendecomp`, `operators.py:2715`); profiles `ḡ^aa(x_a)`
  from the resident quad-point metric; D = J collocated at the 0-form Greville grid.

### V-cycle (symmetric, SPD)
```
V(r): x1, r1 = smooth(r)          # pre, m A-applies (residual free from cheb loop)
      ec = coarse_solve(Pᵀ r1)
      c  = P ec;  r2 = r1 − A c   # 1 A-apply
      dx, _ = smooth(r2)          # post, m A-applies
      return x1 + c + dx
```
2m+1 fine A-applies per cycle. SPD: `B = 2T − TAT + (I−TA) P B_c Pᵀ (I−AT)` with T a
symmetric Chebyshev polynomial (λ(TA) ⊂ (0,2) by the window), R=Pᵀ, B_c SPD → B PD.
Numeric gates: symmetry error < 1e−12, positive Rayleigh quotients on random vectors.

### Coarsest solve (`--coarse-solve {dense,fd,fdax}`, default dense)
Dense probe of the coarsest bulk operator (sequential `lax.map`, n_c coarse applies at
setup) + Cholesky. Exact — removes the coarse-solve confound from the smoother comparison.
n_c is small by construction (~0.3–3k DOFs; ≤80 MB dense worst case).

### Schur envelope (rebuilt, one V-cycle per apply)
Probe `ass` (3nz×3nz) and `C0` (n_bulk×3nz) exactly as production; then precompute
`W = B_mg C0` (3nz V-cycles, once) → `schur = sym(ass − C0ᵀ W)` →
`schur_inv = _symmetric_pseudoinverse`. Apply: `y = V(rhs_b)`;
`z = schur_inv (rhs_c − C0ᵀ y)`; `x_b = y − W z` — **one V-cycle per preconditioner
apply** (not two), and the Schur rebuild is free given W. Knob `--schur {rebuild,fd}`
(reusing the FD-based schur_inv is SPD-safe — the envelope is SPD for any SPD S⁻¹, B).

### Outer solve & measurement
`solve_singular_cg` with `vs = _nullspace_vectors(ops, 0, dirichlet)` (constant, free BC).
`jax.jit` around the whole solve closure (operators captured, compile over rhs); warm +
timed runs so ms/it is marginal cost (compile-overhead lesson from the mass benchmark).
Baseline = production `assemble_tensor_laplacian_preconditioner` + tensor-hodge apply.
`--levels 1` degenerates to Chebyshev-smoother-only PCG — free single-level comparison of
the three smoothers before MG enters.

## Cost model (expectations)

MG-PCG per iteration ≈ 6.9 fine-A units (m=2) / 4.4 (m=1) vs baseline ≈ 1.4 →
**break-even ≈ 5× (m=2) / 3× (m=1) iteration reduction**. Expect MG to lose wall-clock at
(8,16,8) and win at ≥(16,32,16) if h-flat. Setup (probes, W, Lanczos, coarse geometry)
reported separately in the CSV (`setup_*` columns).

## Validation matrix

`{cylinder, toroid, w7x} × {dbc, free} × ns {(8,16,8),(12,24,12),(16,32,16),(24,48,24)}
× smoother {jacobi, fd, fdax} × m {1,2}` at p=3, vs baseline; W7-X additionally
`--coarsen {2,2,2 (ζ-floored), 2,2,1}`; (24,48,24) additionally `--levels {2,3}`.

Success: (1) MG iters grow ≤ ~1.3× across the h-sweep (vs baseline growth); (2) wall-clock
beats baseline at (16,32,16)+; (3) no free-BC stall on toroid/W7-X (recomputed residual
consistent); (4) SPD checks pass. The smoother comparison answers: does local-exact
Jacobi beat both FD averagings, and how much does axis-averaging (fdax) recover vs the
1/3-version (fd)?

## Phasing

0. Script skeleton + level/transfer builders; `--two-level-check` on cylinder (8,16,8):
   SPD asserts, `P_const_err`, nested-periodic P exactness.
1. Two-level cylinder, dbc+free, all three smoothers, m∈{1,2} — gate: SPD pass, h-flat.
2. Toroid dbc+free — first curved-geometry + free-BC-stall test.
3. W7-X on GPU (`--zeta-diag` first; coarsen-policy sweep).
4. (24,48,24), levels 2 vs 3, `--schur rebuild` vs `fd`; full CSV for the final table.

Production wiring into `operators.py` (equinox factor leaves, `cp_kwargs={"mg_levels":…}`)
is **deferred** until Phases 1–4 validate.

## Verification

- Local quick check (CPU ok): `python scripts/debug/laplacian_mg_k0.py --geometry cylinder
  --ns 8 16 8 --two-level-check` → SPD asserts pass, MG-PCG converges, iters printed.
- Sweep: `bash slurm/job_laplacian_mg_k0.sh` (per-geometry jobs on gpu-h100s); compare
  `outputs/laplacian_mg_k0/<stamp>/*.csv`: MG vs baseline iters across ns (h-flatness),
  ms/it, setup costs, smoother ranking.
