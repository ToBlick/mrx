# Handoff 2026-08-24 — the k=1 free harmonic form, and the k>=1 saddle preconditioner

Found while validating the A5 swap on the quasr geometries. **It is not a
quasr problem and not a new one**: it is present in every W7-X p=5 cell of the
bc-alpha sweep, silently.

## The symptom

`compute_nullspaces` reports the k=1 NBC harmonic form's quality as
`q = v.L v / v.M v`, `relL2 = sqrt(q)`. Measured across every run on disk:

| geometry | p=2 | p=3 | p=5 |
| --- | --- | --- | --- |
| toroid | — | 6.5e-12 | — |
| rot-ellipse | 3.9e-12 | 1.3e-11 | **3.0e-02** |
| W7-X | 8.4e-13 | **3.0e-04** | **1.7e-01** |
| quasr9983 | — | 4.5e-07 | — |
| quasr44970 | — | 7.7e-04 | — |
| hegna | — | 4.7e-03 | — |

Degrades with **p**, not with the device. A 17% error at W7-X p=5 is the
deflation vector for every k=1 free solve in those cells.

## What it is NOT

Two hypotheses tested and **refuted**:

1. **Not the inner tolerance.** `scripts/debug/harmonic_k1_free_diag.py` reruns
   step 3 (`a = L_2^{-1,free} D_1 v`) at inner tol 1e-8 / 1e-10 / 1e-12 / 1e-14:

   | | 1e-8 | 1e-10 | 1e-12 | 1e-14 |
   | --- | --- | --- | --- | --- |
   | W7-X p=2 | 4.9e-08 | 5.7e-10 | 5.6e-12 | 6.2e-13 |
   | W7-X p=3 | 3.04e-04 | 3.01e-04 | 3.05e-04 | 3.00e-04 |
   | W7-X p=5 | 1.708e-01 | 1.709e-01 | 1.711e-01 | 1.708e-01 |
   | hegna p=3 | 4.73e-03 | 4.72e-03 | 4.73e-03 | 4.72e-03 |

   At p=2 the form tracks the tolerance 1:1 — it is exactly as good as that
   solve and no better, which is the mechanism. At p>=3 it is **flat across six
   orders of magnitude**: the solve floors and more accuracy is unobtainable.
   (An early read of the p=2 row alone said "the form's error IS the inner
   residual". True at p=2, false at p>=3.)

2. **Not fixable by inverse-iteration polish.** Seeded from the direct vector,
   6 steps at eps=1e-4: W7-X p=3 goes 3.0e-04 -> 1.4e-02, quasr9983
   4.5e-07 -> 1.8e-05, W7-X p=5 barely moves (1.708e-01 -> 1.596e-01). Only
   hegna improves, and only on step 1 before degrading. It walks AWAY. The
   shifted solve it uses goes through the same saddle path, so it inherits the
   same problem.

## What it is

`apply_inverse_hodge_laplacian` at k>=1 routes to `solve_saddle_point_minres`,
whose preconditioner is block-diagonal with

    schur.outer = jacobi          <-- mrx/operators.py:_materialize_default_saddle_preconditioner
    schur.inner = raw_kron
    mass        = default

and whose valid outer kinds were `('none','jacobi','exact_jacobi')` — **the
block-Jacobi Laplacian atom, i.e. the production Laplacian preconditioner, was
not reachable there at all.** Worse, that jacobi is doubly approximate: per
`apply_hodge_laplacian_preconditioner`'s own docstring, for k >= 1 the weak
half of the diagonal is a Kronecker mass MODEL, not the operator's own
`D M^-1 D^T`.

So the entire nullspace construction — and every other k>=1 Laplacian solve in
the library — runs on the per-DoF diagonal.

### MINRES is not the problem; do not replace it with CG

An intermediate A/B ran "CG on L_2 directly" and converged in 993 iterations to
a form 6 orders better than the saddle route, which looked like an indictment
of the formulation. **It is not.** `apply_hodge_laplacian` calls
`apply_inverse_mass_matrix` — a nested mass CG inside every operator apply.
That arm is Krylov-in-Krylov (against the project convention) and cost 1007 s
against the saddle route's 30 s. The saddle formulation exists precisely to
turn that nested solve into a variable, the system is symmetric indefinite, and
MINRES is the right method for it. The fix is the preconditioner, not the
formulation.

## The fix under test

The Schur complement of the saddle system IS `L_k = S_k + D M^-1 D^T`, which is
exactly what the block-Jacobi atom approximates, and MINRES requires only that
its preconditioner be SPD — which the atom is (`test_preconditioner_is_spd`).
So `'block'` belongs as `schur.outer`.

Wired in `mrx/operators.py` (2026-08-24): added to `valid_outer_kinds` in both
the spec and string forms, exempted from `_validate_inner_tensor_only_spec`
(it needs no `schur.inner`), and given a dispatch branch that calls the atom
directly. **The DEFAULT is unchanged** pending the measurement:
`scripts/debug/harmonic_k1_precond_ab.py`, arms `saddle+jacobi` vs
`saddle+BLOCK`, on W7-X p=3/p=5 and hegna p=3
(`outputs/harmonic_ab/2026-08-24/06-16-08/`).

If block converges MINRES where jacobi stalls, flipping
`_materialize_default_saddle_preconditioner` to prefer it when assembled is a
one-line change — with a blast radius of every k>=1 solve in MRX, so it wants
its own regression pass.

## Consequences already banked

- `scripts/debug/verify_block_jacobi.py` reverted to the DeRhamSequence
  defaults (`tol` 1e-12, `maxiter` 10_000), from `inner_tol` 1e-13 /
  `maxiter` 20000. The old comment claimed "the W7-X free L_2 solves are known
  to need all of it"; they were stalling and the budget was hiding it. Note
  W7-X p=3 reads 3.0e-04 at maxiter 20000 and 7.9e-03 at 10000 — the saddle
  route consumes whatever it is given and still does not converge.
- **The bc-alpha sweep's W7-X p=5 k=1 cells used a 17%-wrong deflation
  vector.** All arms in a job share one nullspace, so the A/B is fair, but the
  absolute counts and the fitted `s*` are suspect. Those cells fed Result 2
  (p-portability). Rerun them once this is fixed before trusting that.

---

# RESOLVED 2026-08-24 — the k>=1 saddle solves were running without the mass preconditioner

One line in `mrx/operators.py`:

```python
def _materialize_default_mass_preconditioner(seq, operators, *, k):
    if _tensor_available(seq, operators, k):        # <-- the bug
        return default_mass_preconditioner()
    return MassPreconditionerSpec(kind='jacobi')
```

`default_mass_preconditioner()` meant `kind='tensor'` when this was written, and
tensor DOES need an eager assembly, so the fallback was correct then. It has
meant `'block_jacobi'` since 2026-08-22, and block_jacobi is always buildable.
The gate therefore silently downgraded the saddle preconditioner's LOWER block
to a per-DoF diagonal in the normal case (`assemble_mass_jacobi_preconditioner`
alone does not make `_tensor_available` true).

So every k>=1 Laplacian solve in MRX ran with the production mass
preconditioner switched off. The path that *does* use it,
`apply_mass_matrix_preconditioner(kind='auto')`, resolves through
`_resolve_legacy_mass_preconditioner` -> `default_mass_preconditioner()`
unconditionally -- which is why a standalone MINRES with an otherwise identical
setup converged in 84 iterations where the library needed 9612.

Fix: return `default_mass_preconditioner()` unconditionally.

## Measured (12x24x12, p=3, tol 1e-10, maxiter 10000)

Convergence of `apply_inverse_hodge_laplacian` over 18 cells
(3 geometries x k=1,2,3 x free/dbc):

| schur.outer | before | after |
| --- | --- | --- |
| jacobi (library default) | **2/18** | **18/18** |
| block @ a0 | 13/18 | 18/18 |
| block @ a5 | 13/18 | 18/18 |

Iterations, `outer=block`: toroid k2 free 9612 -> **314**; toroid k2 dbc
9683 -> **301**; quasr k1 free 6317 -> **857**; w7x k2 dbc 10000! -> **1522**;
w7x k2 free 10000! -> **4659**. Five to thirty times fewer, everywhere.

**The block atom as `schur.outer` is still worth 2.51x on top** (36776 vs 14664
total iterations across the 18 cells), so both changes earn their place. Keep
`'block'` wired in; its default is still untouched pending a decision.

## What this retires

- "k=2 is badly conditioned / genuinely hard" -- no, it was unpreconditioned.
  W7-X p=5 k=2 free takes 123 iterations with the mass preconditioner on.
- "relax the tolerance to 1e-6/1e-8" -- moot at 300-900 iterations. The earlier
  measurement showed `phibar` tracks the true dual residual to within ~7x, so
  relaxing was a real accuracy trade, never free. Leave tol at 1e-10.
- **CGS vs MGS in the Lanczos step: REFUTED.** Identical iteration counts
  (75/75, 84/84, 81/81) and `max |v.r1|/(|v||r1|)` of 1e-17..1e-20 -- no
  orthogonality loss at all. `mrx/solvers.py:minres` needs no change; the
  in-tree alpha ordering is fine.
- The MINRES `abs()` removals stand on their own merits (see
  [[no-defensive-code]]) but were not the cause; `neg_ip` was 0 everywhere.

## a0 vs a5 on the fixed real solve

Iterations tie on toroid and W7-X. On quasr9983 a5 is worse: k=1 free 857 vs
926 (+8%), k=2 free 778 vs 873 (+12%), k=3 free 225 vs 346 (**+54%**). The
out-of-sample finding survives the fix.

On TOTAL time a5 still wins, `a5/a0 = 0.751`, entirely on build cost: the
`product` amplification needs a weak inverse and takes 11.6-11.9 s to build at
k=1 free against `penalty`'s 1.1 s. With the solves now 2-11 s instead of
minutes, that one-time 10 s gap is proportionally LARGER than before. So the
choice is genuinely: build once and solve a few times -> a5; solve many times
on quasr-class geometries -> a0. Still the user's call.
