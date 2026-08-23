# MRX production solver configuration

**The single authority on what runs in production.** Last revised 2026-08-22
(preconditioner stack replaced; see "2026-08-22 change" below). Research
alternatives live in `docs/research/` with reopen conditions.

## Preconditioners

| solve | preconditioner | where |
| --- | --- | --- |
| Mass matrices, all k (incl. saddle lower blocks) | **`kind='block_jacobi'`** -- separable bulk with the polar core PROBED AND INVERTED DENSELY (no `E+` pseudoinverse) | `mrx/experimental/block_jacobi_laplacian.py:BlockJacobiMass` |
| Laplacians, **k = 0,1,2,3** | **`kind='block'`** -- the tensor block-Jacobi atom: per-component Kronecker sum inverted by fast diagonalisation, dense polar core, plus a rank-one natural-BC term at `bc_scale=0.10` | `mrx/experimental/block_jacobi_laplacian.py:BlockJacobiLaplacian` |

Build the Laplacian one once per `(k, BC)` with
`assemble_block_jacobi_laplacian_preconditioner(seq, ops)`; then
`seq.apply_laplacian_preconditioner(v, k, dirichlet)` (`kind='auto'` picks
`'block'` when assembled, else `'jacobi'`). It takes NO required parameters.

### 2026-08-22 change: the preconditioner stack was replaced

The mass went `tensor/CP -> raw_kron (2026-08-17) -> block_jacobi (2026-08-22)`;
the Laplacian at k>=1 went `Schur-outer Jacobi -> the block atom`. Both are
measured over four geometries, n = 8..32, p = 2..5:

* **Laplacian vs point Jacobi: median 0.31 over 120 cells, 0.25 for n >= 24**,
  and the ratio IMPROVES with refinement (3-8x at production resolution). The
  advantage also grows with degree -- Jacobi degrades 7.6-12.2x over p = 2..5
  where the block arms grow 2.3-2.8x.
* **Mass vs raw_kron: 0.83x iterations median, 0.70-0.77x at k=1,2**, at equal
  build cost.

**This overturns the 2026-08-14 "k>0 policy" below.** That assessment concluded
the k>0 option space was closed and Jacobi should stay, on the grounds that
anything beating Jacobi on hard geometry must contain a faithful `L0` solve.
The block atom does beat it, everywhere, without one. The section is kept for
its reasoning and its reopen conditions, but its VERDICT is superseded --
see `docs/research/preconditioner_technical_note_source.md`.

Explicitly **not** in production: multigrid (any k), Chebyshev acceleration
(anywhere), CP rank>1 stiffness fits, HX / auxiliary-space transfers, dense
outer-ring probes (`outer_rings`), and the truncated-Fourier coarse correction
(`fm`, opt-in only, `mrx/experimental/block_jacobi_coarse.py`).

Research machinery lives in **`mrx/experimental/`**: `tensor_stiffness.py`
(the k>=1 block_fd P_A atoms), `chebyshev.py`, and `block_jacobi_coarse.py`
(`fm`). NOTE that `block_jacobi_laplacian.py` also lives there but IS now
production -- it should move to `mrx/` when convenient.

Preconditioner kinds. Laplacian: `none` / `jacobi` / **`block`** /
`probed_jacobi` / `tensor` (k=0, retired). Mass: `none` / `jacobi` /
**`block_jacobi`** / `raw_kron` / `tensor` (retired).
`kind='probed_jacobi'` is the exact `diag(L_k)` by probing -- the honest
REFERENCE for benchmarks, never a candidate (O(N) applies to build).
`kind='jacobi'` is NOT that at k>=1: its weak half is a Kronecker mass MODEL,
and it costs up to 21% extra iterations against the exact diagonal.
Unmaintained demo scripts: `scripts/deprecated/`.

## Knobs

- `PRODUCTION_BC_SCALE = 0.10` scales the natural-BC term. EMPIRICAL (a
  kappa-balance point, not a derived factor); minimax over 168 cells, median
  penalty 1.01 against each cell's own optimum. `MRX_BJ_BC_SCALE` overrides.
- `MRX_MASS_KIND=raw_kron` reverts the mass swap wholesale.
- **Any new traced entry point that solves must first call
  `operators.warm_mass_preconditioner_cache`.** Mass factors build lazily and
  the build is host-side numpy, so a cold cache inside a `jax.lax.while_loop`
  dies. This bit once, in `nullspace.py`.
- KNOWN REGRESSION: `build_weak_term_diagonal` is still calibrated for
  raw_kron, so `kind='jacobi'` costs 1-10% more than it used to and
  `test_weak_term_diagonal_matches_exact_rows` skips unless the mass is
  raw_kron. Top open item in `docs/research/HANDOFF_open_items.md`.
- Mass CP fits (retired path): non-negative (NTF) default; `MRX_CP_GREEDY`
  reverts.
- Solvers: k=0 = deflated CG (condensed); k>=1 = saddle MINRES with
  harmonic deflation. No RUNTIME Krylov-in-Krylov anywhere (the k=0
  core-Schur rebuild runs 3*n_zeta assembly-time CG solves; the result
  is a fixed dense matrix).
- The pre-2026-08 collocated k=0 atom (`fdlegacy`) and the `MRX_K0_ATOM`
  knob were DELETED 2026-08-14: the bundled "fd" atom is the only k=0
  atom, and the core-Schur rebuild now uses EXACT bulk solves (the
  collocated atom's last role was the one-sided Schur probe).

## Verified numbers (2026-08, CG/MINRES iterations to 1e-10)

- k=0 "fd" vs old atom: W7-X (12,24,24) 64->54 dbc / 100->79 free; toroid
  (16,32,16) 32->24 / 45->30. Vs Jacobi: ~7-10x wall.
- k=0 exact core-Schur rebuild vs the retired collocated probe
  (2026-08-14): toroid (8,16,8) 22/25 vs 22/31; W7-X (12,24,24) 53 dbc /
  80 free vs 56/87, at equal preconditioner assembly cost (~37s GPU).
- k>=1 Jacobi is measured-optimal in the relaxation class (l1-Jacobi
  10-30% worse; mass-as-preconditioner 7-11x worse; mass/point smoothed MG
  refuted; ledger in `docs/research/handoff_2026-08-13_eod.md`).

## The research shelf (what would replace Jacobi at k>=1, and when)

The k=1 coupled-atom + exact-L0 solver is mathematically settled
(80-172 its geometry-independent, 3-9x Jacobi wall in prototype); reopen =
one wiring day (jit + dense-L0 Cholesky) when k=1 solves bottleneck
production. Details + all other shelf items: `docs/research/`.

## k>0 policy: final assessment (2026-08-14) — VERDICT SUPERSEDED 2026-08-22

> Kept for the reasoning and the reopen conditions. Its conclusion ("Jacobi
> stays") is no longer production: the tensor block-Jacobi atom beats Jacobi by
> 2-8x at k=1,2,3 without a faithful `L0` solve, which is the premise this
> assessment rests on.

The k>0 option space is closed — full reasoning in
`docs/research/k_gt0_final_assessment.md`. Summary: any method that beats
Jacobi on hard geometry must contain a faithful L0 solve (measured +
literature-confirmed); everything cheaper is break-even at best. Jacobi
stays; the coupled+dense-L0 shelf is the only sanctioned opt-in.
Production timestep solves are shifted (`M/dt + eta*L`), which collapses
the pure-Laplacian wall to one-off solves; an optional ~1h Lanczos check
at the production shift would confirm this.
