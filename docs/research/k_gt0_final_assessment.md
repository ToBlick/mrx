# k>0 Laplacian preconditioning: final assessment

> **STATUS: DECISION RECORD (2026-08-14) — closes the k>0 preconditioner
> campaign.** Production policy lives in `docs/PRODUCTION.md`; this file is
> the reasoning behind it and the conditions under which it could reopen.

## The structural fact everything reduces to

`S_hat = S_k + D M_hat^{-1} D^T` splits M_k-orthogonally along the exact
sequence into (a) the **complement subspace** governed by the stiffness term
S_k (curl energy at k=1), and (b) the **range-of-d subspace** (gradients at
k=1) governed by the weak term, which is spectrally the (k-1)-Laplacian
pushed up. Every option is judged by what it does to each subspace.

## Two measured facts that bound the whole design space

1. **Pencil exactness bound.** Per axis, two SPD matrices are always a
   pencil — simultaneously diagonalizable, invertible exactly at full
   rank-1 metric weights. The two-term curl blocks of k=1 sit exactly at
   this bound (rank1/coupled atoms). Adding the weak term makes it three
   terms and forces shared eigenbases across all of them — i.e. metric
   lumping (the no-mixing constraint). On the S_k side lumping is *fine*:
   the coupled 3x3 atom with faithful L0 gives geometry-independent counts
   (80-95 its on cylinder/toroid/rot-ell, 154/172 on W7-X).
2. **Gradient-subspace steepness.** The range-of-d subspace *refuses*
   lumping on W7-X. One exact-inverse-of-lumped-L0 apply (the fd atom,
   which as a k=0 CG preconditioner is excellent: 54/79 its) still costs
   the k=1 solve 1614/3817 its; ~3 applies (Newton-Schulz) only halves the
   gap; the exact L0 gives 154/172. The fidelity demand is steep, so no
   lumped treatment of the weak term closes it.

**Corollary:** any method that beats Jacobi on hard geometry contains a
faithful L0 = (k-1)-Laplacian solve. That is precisely the auxiliary-space
(Hiptmair-Xu) structure, and the literature agrees: the industry-standard
AMS/ADS (hypre/MFEM) and the matrix-free high-order H(curl) work
(Barker-Kolev) all route through AMG on the nodal spaces. There is no
simple standalone method we missed (GenEO/Schwarz-H(curl): setup
eigenproblems; LOR: re-imports AMG; Vanka/patch: declined; the
Sangalli-Tani fast-diagonalization family is scalar/Stokes only — its
vector extension hits exactly the kappa-continuum we measured).

## Option ledger (final)

| option | gradient-subspace treatment | measured / projected (k=1 W7-X dbc/free) | verdict |
| --- | --- | --- | --- |
| Schur-outer Jacobi, tensor_probe | none (flat) | 950/1509 its, kappa compact 1.7e3/1.3e4; relaxation class exhausted (l1, mass-precond, SGS, Chebyshev all lose) | **production default** |
| rank-1 pencil atoms (profile/rank1/greville-D) | P_B with one-atom L0 | kappa continuum 7.6e6; rank1 2908 its | refuted |
| coupled 3x3 atom + atom-L0 | lumped | 1614/3817 | loses to Jacobi wall |
| coupled + **dense L0** | exact | **154/172 = 3.3x/9.5x Jacobi wall** | only measured winner; dense Cholesky memory O(n0^2) caps it at n0 ~ 20k; h-scaling would need MG-L0 |
| coupled + MG-L0 | exact-ish (V-cycles) | tracks dense within ~10% | works, but re-imports MG — violates the no-MG/simplicity decision |
| P12^T B_div P12 auxiliary | stiffness-like replacement, but B_div is itself a lumped atom | 3/4 geometries OK; W7-X stalls on term-scale imbalance (Lanczos-calibration fix specified) | even calibrated it lands in the lumped-gradient class => break-even at best; collocated transfer adds conditioning risk (cf. the k=3 transfer refutation) |
| additive lumped weak symbol (untested: promote the sigma-regularizer to the calibrated lumped D M~^-1 D^T per-mode tt^T) | lumped, additive (no projector, so no error amplification) | projected: gradient block conditioned like fd-preconditioned L0 => ~400-600 its at 2-3x Jacobi per-it cost => break-even | not worth building |

Why the near-exact NTF Kronecker M0^-1 can't rescue the lumped class:
simultaneous diagonalizability needs shared per-axis bases across *all*
terms; the NTF weights break it. Near-exact masses do not transfer to
near-exact atoms — that is the no-mixing constraint, not an implementation
gap.

**k=2** is the Hodge mirror of k=1 (strong div-div side easy — the greville
atom already wins there; weak curl side is the hard subspace): same verdict.
**k=3** is pure weak term; if it ever matters, the right shape is the k=0
fd recipe verbatim on V3 (per-axis pencils with D_a M~^-1 D_a^T, Lynch) —
an afternoon of work. Today: Jacobi, low value.

## The regime argument that closes the book

Production k>0 solves are predominantly **shifted** (`M/dt + eta*L`).
Mass-dominated shifts collapse the difficulty: Jacobi/Kronecker-mass
converge in tens of iterations regardless of geometry. The entire W7-X
wall lives in the pure-Laplacian limit, which occurs in *one-off* solves
(vacuum/harmonic fields, steady states) that tolerate ~1000 cheap
iterations. The value of beating Jacobi is bounded by one-off frequency.

Cheapest possible overturn-check (~1 GPU-hour, optional): Lanczos kappa of
the tensor-probe-preconditioned shifted operator at production eta*dt on
W7-X. kappa <~ 1e3 confirms the wall is irrelevant to timestepping.

## Decision

1. **Jacobi (`tensor_probe`) stays the production default for all k>0.**
2. The single sanctioned exception: **coupled + dense-L0 opt-in** for hard
   pure-Laplacian one-offs at n0 <= ~20k, wired only if profiling shows
   those solves dominate a real workflow (reopen = ~1 day: jit + Cholesky).
3. Nothing else gets built. Campaign closed.
