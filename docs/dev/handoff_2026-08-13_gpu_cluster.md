# Handoff — 2026-08-13: k=0 MG preconditioner moves to GPU / cluster

Single entry point for the k=0 Laplacian multigrid line. Supersedes and
folds in the four stacked addenda of the retired
`handoff_2026-07-09_polar_c2_mg.md` (removed; all technical detail lives on in
`docs/laplacian_mg_k0_plan.md` (running log + theory), `docs/dev/k0_massprecond_surgery_findings.md`
(the dense-κ studies), and `docs/polar_pole_regularity.md` (C⁰/C¹/C² theory + convergence)).

## Where the work stands — the local phase is DONE

The k=0 MG prototype (`scripts/debug/laplacian_mg_k0.py`) and the C⁰/C¹/C²
polar surgery are committed on `greville-prod` (NOT merged to main). Every
question answerable on a laptop CPU at 8³ has been answered; what remains is
**scale** — h-refinement, true multilevel, W7-X, and wall-clock — all of
which need the GPU. Concretely:

1. **Smoother question CLOSED (both directions).** Point-Jacobi is out
   (47–180 CG its vs 5–15 for the fd-family at ~equal ms/it — A-applies
   dominate — and it even loses to the single-level baseline; cause is the
   bulk-wide polar anisotropy `g^{θθ}~1/r²`, the classic point-smoother
   failure). And nothing fancier than the separable fd/fdbund atom pays: the
   off-diagonal `g^{rθ}` lever is refuted (dropping ALL off-diagonals costs
   κ 1.5→2.6 helical vs atoms at ~12); the real residual is
   averaging/separability of the *diagonal* weights, and the best
   production-shaped fix (2D-per-ζ-mode solves) buys only ~1.5× at real cost.
   ⇒ **fd-family separable atom + exact core surgery is the sweet spot.**
   (`k0_massprecond_surgery_findings.md`; `run_mg_k0_jacobi_ab.sh`.)

2. **C²-on-axis == fat-core R=1, but cheaper (priority-1, DONE).**
   `--polar-order {0,1,2}` is wired into the prototype. C² reproduces the
   fat-core λmax(S·A) to all printed digits on every geometry/atom (8³ dbc,
   fd): toroid 1.61, cerfon 2.28, rot-ellipse 3.55 — because both start the
   bulk window at ring 3 — but with **fewer DOFs** (n=560 vs 664) and a **~3×
   smaller Schur core** (6·nz=48 vs fat 3·nz+nt·nz=152): C² *eliminates* the
   near-axis freedom rather than solving it exactly. The C⁰ control (window
   ring 1, λmax 3.9–7.3) confirms the spread tracks exact-region *extent*, not
   pole smoothness per se. ⇒ **C² is the production-preferred realization of
   the fat-core gain**; fat-core stays the C¹-compatible fallback (k≥1,
   free-BC). (`run_mg_k0_polar_order_ab.sh`,
   `outputs/laplacian_mg_k0/polar_order_ab_20260805/`.)

3. **h-scaling (C¹) is close but not textbook-flat — and the fix is
   predicted, not measured.** With the auto-m window rule
   (`--cheb-lo 0.85 --auto-m`, m~n^{1/4}), MG(fd) grows ×1.4 dbc / ×1.2 free
   over 8³→16³ (toroid) vs baseline ×1.7, and the MG:baseline lead *widens*
   with size (2.8×→3.75×). Residual non-flatness is NOT the smoother (damping
   matched) — leading suspects are the **rediscretized non-Galerkin coarse
   operator** and the fine/coarse ξ₁-gap annulus. Crucially,
   `λmax(S·A) ≈ 1.7/ξ₁ + h-independent-shaping`, and fat-core/C² removes
   exactly the `1.7/ξ₁` axis part → **C² should restore near h-flatness**.
   This is the single most valuable unmeasured claim (see experiment B).

## Cluster experiment matrix (priority order)

Driver: `slurm/job_laplacian_mg_k0.sh` (one sbatch/geometry on `gpu-h100s` /
`extremedata`). It now takes an `EXTRA_ARGS` env passthrough and defaults
`SMOOTHERS=fd,fdbund` (fdax retired — fdbund is the adopted alternative). All
knobs are env vars; presets below.

**A. W7-X reconfirm + atom decision + λmax prediction.** The motivating
regime. Reconfirm (12,24,24) baseline 64/99 vs MG(fd) 18/23; add (16,32,32).
`fd` vs `fdbund` is the *open atom decision* (locally a tie; theory says
fdbund pulls ahead only on strong anisotropy = W7-X). Falsifiable check: **fd
λmax ≈ 8.7 (dbc) at (16,32,32)** — if far off, the axis/shaping factorization
misses an interaction. `--spectrum-diag` gives the definitive mode geography.
```
GEOMETRIES=w7x NS_LIST="12,24,24 16,32,32" SMOOTHERS=fd,fdbund bash slurm/job_laplacian_mg_k0.sh
```

**B. fat-core/C² h-sweep — does C² restore h-flatness? (highest-value new run.)**
All C²/fat-core data is 8³ only. Sweep 8³→16³ and check whether MG(C²) is
h-flat where MG(C¹) grew ×1.4. If yes, C² fixes what auto-m only papered over.
```
GEOMETRIES="toroid cerfon rotating_ellipse" NS_LIST="8,16,8 12,24,12 16,32,16" \
  EXTRA_ARGS="--polar-order 2 --anchor-xi1" bash slurm/job_laplacian_mg_k0.sh
# baseline/C¹ reference arm: EXTRA_ARGS="--fat-core 1 --anchor-xi1" (same window, C¹ layout)
```

**C. True multilevel scaling — completely untested.** *Every* result to date
is two-level with a dense *exact* pseudoinverse coarse solve (deliberately, to
remove the coarse-solve confound). So current iteration counts are best-case;
a real 3+ level V-cycle with the non-Galerkin rediscretized coarse operator
could degrade. Run (24,48,24) at `LEVELS=3`.
```
GEOMETRIES=toroid NS_LIST="24,48,24" LEVELS=3 COARSEN="2,2,2" bash slurm/job_laplacian_mg_k0.sh
```

**D. m=1 smoothing A/B on the κ≈1.6 C² configs.** C²/fat-core drops
axisymmetric geometries to κ≈1.9, where a single damped atom apply (m=1) may
suffice → 3 A-units/cycle, break-even ~3×. Needs a one-line auto-m floor 2→1
in the script (`m = max(1, ...)`) behind a flag before this can run.

**E. GPU wall-clock crossover.** At every size reachable so far AND at W7-X
(12,24,24) on the cluster, the single-level baseline still wins wall-clock —
the 3–4× iteration lead hasn't beaten the m=4 per-cycle A-apply penalty. The
crossover is expected only at larger W7-X on GPU (baseline iteration growth +
GPU utilization). This is *the* open question; A/B/C feed it.

## Gotchas / invariants (carry these to the cluster)

- **Env for W7-X:** `XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128`
  (the recursive-spline map jacfwd OOMs under full vmap; batched lax.map).
  Needs the fitted map data in gitignored `data/W7-X.h5`.
- **`--polar-order != 1` is dbc-only, no production baseline.** The shipped FD
  preconditioner warm-up (`assemble_laplacian_operators`) hardcodes the C¹
  polar layout, so order≠1 uses incidence-only assembly; free-BC needs the
  nullspace vectors that assembly fills → not yet supported for order≠1.
  Threading invariant: `ring0 = fat_core + (order-1)` drives ALL bulk-window
  indexing (atoms/transfers/profiles/diagonal).
- **PSD pseudoinverse trap.** Any rebuilt Schur (free BC, strong
  preconditioner) can land the analytic null direction slightly negative;
  `_symmetric_pseudoinverse` inverts by magnitude WITH sign → −O(10³)
  Rayleigh. The prototype uses `_psd_pseudoinverse` (positive part only).
  **Production `mrx/preconditioners.py:_symmetric_pseudoinverse` carries the
  same latent trap — fix when wiring production.**
- **fdax is retired** (fdbund strictly better: bundled `⟨g^{aa}J⟩` milder than
  bare `g^{aa}`, cheaper D=1, better-motivated). The atom decision is now
  fd vs fdbund, W7-X-gated.
- **Background/長 runs:** local Bash tasks die at 10 min — the cluster is
  precisely to escape this; use the sbatch driver, not nohup.

## Rerun / repro

- Local sanity (~2.5 min CPU): `python scripts/debug/laplacian_mg_k0.py --geometry cylinder --ns 8 16 8 --two-level-check`
- Local C² smoke: `python scripts/debug/laplacian_mg_k0.py --geometry toroid --ns 6 12 6 --smoothers fd --cheb-lo 0.85 --auto-m --two-level-check --bc dbc --polar-order 2`
- Dense-κ studies (laptop): `scripts/debug/verify_hodge_massprecond_k0.py` (off-diag ladder), `scripts/debug/run_mg_k0_jacobi_ab.sh`, `scripts/debug/run_mg_k0_polar_order_ab.sh`.
- CSV merge on cluster: `awk 'FNR==1&&NR!=1{next}1' outputs/laplacian_mg_k0/<stamp>/mg_*.csv > .../merged.csv`

## Backlog (not cluster-gated)

C² de Rham for k≥1 (Bernstein-identity rework of E¹/E², few hundred lines);
per-ζ-plane ξ for stellarator maps; `get_xi2` map-adapted C² (currently
circle-only); production wiring (PSD pseudoinverse, r_scale=0.5, auto-m rule,
coarsest Cholesky + explicit deflation instead of thresholding); full-space MG
experiment (`--envelope full`, kills the Schur probe + W precompute) — design
sketched in the plan doc.
