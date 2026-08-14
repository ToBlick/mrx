# MRX production solver configuration

**The single authority on what runs in production** (decided 2026-08-14;
research alternatives live in `docs/research/` with reopen conditions).

## Preconditioners

| solve | preconditioner | where |
| --- | --- | --- |
| Mass matrices, all k (incl. saddle lower blocks) | Kronecker/tensor mass preconditioners (exact at rank 2 for masses; NTF/non-negative CP fits) | `mrx/preconditioners.py` |
| k=0 Laplacian | **"fd" atom** = the bundled per-axis `<g^{aa}J>` tensor-Hodge atom + thin C1 polar core Schur (fd-probed) | `mrx/operators.py:_assemble_k0_greville_bulk_factors` |
| k=1,2,3 Laplacians | **Schur-outer Jacobi**: diagonal tensor-probed on the approximate Schur `S_hat = S_k + D M_hat^{-1} D^T`, with the tensor MASS preconditioner as `M_hat^{-1}` in the weak term (`schur_diag_mode='tensor_probe'`) | `mrx/operators.py` |

Explicitly **not** in production: multigrid (any k), Chebyshev acceleration
(anywhere), CP rank>1 stiffness fits, HX / auxiliary-space transfers.

Research machinery lives in **`mrx/experimental/`** (not imported by mrx
core): `tensor_stiffness.py` (the k>=1 block_fd P_A atoms) and
`chebyshev.py` (chebyshev/richardson/lanczos). Production preconditioner
kinds are `none`/`jacobi`/`tensor` only (`exact_jacobi` additionally as a
debug schur outer). Unmaintained demo scripts: `scripts/deprecated/`.

## Knobs

- Mass CP fits: non-negative (NTF) default; `MRX_CP_GREEDY` reverts.
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

## k>0 policy: final assessment (2026-08-14)

The k>0 option space is closed — full reasoning in
`docs/research/k_gt0_final_assessment.md`. Summary: any method that beats
Jacobi on hard geometry must contain a faithful L0 solve (measured +
literature-confirmed); everything cheaper is break-even at best. Jacobi
stays; the coupled+dense-L0 shelf is the only sanctioned opt-in.
Production timestep solves are shifted (`M/dt + eta*L`), which collapses
the pure-Laplacian wall to one-off solves; an optional ~1h Lanczos check
at the production shift would confirm this.
