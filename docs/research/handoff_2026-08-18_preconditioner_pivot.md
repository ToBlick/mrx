# Handoff 2026-08-18 — production preconditioner pivot

Suite green at **179 passed, 1 skipped, 4 xfailed** (800 s); experimental suite
**30 passed** (256 s, opt-in). Lint at the 50-error pre-existing baseline.

## Production is now: raw_kron masses + Jacobi Laplacians

| | preconditioner | diagonal |
| --- | --- | --- |
| masses k=0..3 | `raw_kron` (Kronecker on the raw grid, `E+` transfer) | -- |
| Laplacians k=0..3 | Jacobi | k=0 **closed form**; k>=1 still probed |

`default_mass_preconditioner()` -> `kind='raw_kron'`;
`_materialize_default_scalar_hodge_preconditioner` -> `kind='jacobi'` at every k;
`schur_diag_mode='raw_kron_probe'`.

### Retired to `mrx/experimental/` (kept, not deleted)

* `mass_surgery.py` — surgery/Schur + tensor mass path (55 defs). Reachable as
  `kind='tensor'`, re-exported lazily from `mrx.preconditioners` via a module
  `__getattr__`. NOTE: that hook serves attribute access only, NOT bare global
  lookups inside `preconditioners.py` — which is why the whole tensor path had
  to move as a dependency closure.
* `k0_core_schur.py` — the k=0 core-Schur Laplacian preconditioner. Better per
  solve on axisymmetric geometry (1.2-3.6x) but 26-136 s of assembly vs Jacobi's
  ~1 s, so it only repays past 24-372 solves.
* `modal_radial.py` — modal-radial k=0 bulk atom + the per-k pencil reduction.

Tests for all of the above: `test/experimental/` (in `norecursedirs`; run with
`pytest test/experimental`).

## Closed-form diagonals: what is done and verified

| quantity | status | error vs probe |
| --- | --- | --- |
| `diag(M_k)` raw, k=0..3 | done, in production | 4-5e-16 |
| `diag(S_k)` raw, k=0..3 | done (`S_3 = 0`) | 2.5-5.7e-16 |
| `diag(E S_0 E^T)` extracted, both BCs | done, **in production** | 4.3-8.8e-16 |
| polar/core rows without any probe | demonstrated | 3.3e-16 |
| `diag(A S B)` sum-factorized, S diagonal | identity verified | 1.4e-16 |
| **tensorized weak term** | **identity verified, NOT implemented** | 7.9e-16 |

`L_0 = S_0`, so the k=0 Laplacian Jacobi is fully probe-free today. k>=1 still
carries the weak term and is probed at O(N) applies.

## Next: the tensorized weak-term diagonal

`G` is a SUM over axes of Kronecker terms, so the weak term expands into a
double sum in which every term is a pure Kronecker product::

    M G B G^T M = sum_{d,d'}  M . G^(d) . B . G^(d')^T . M

    diag(term_{d,d'})_i = prod_a [ M_a g_a^(d) B_a g_a^(d')^T M_a ]_{i_a i_a}

with `g_a^(d) = g_a` if `a == d` else identity. Nine axis pairs x three small 1D
matrix products, then an outer product: O(N). Measured saving vs the O(N^2)
probe: **642x** at 12x24x12, **13658x** at 64x128x64.

Remaining work is wiring real 1D factors in: weighted 1D masses per component,
their inverses for `B`, `_dense_incidence_1d` for `g_a` — and deciding how the
non-separable `D` scaling enters. A rank-1 `D` folds into the 1D factors
cleanly; a general one does not, and the existing separability notes (`1/r`
weights rank-1, `theta-zeta` coupled) say that choice is geometry-dependent.
**Measure it, do not assume.**

Do NOT approximate `M` by its diagonal to sidestep this — it discards the mass
coupling entirely and was rejected on 2026-08-18.

## Traps found today (do not repeat)

* **Do not batch `_diagonal_from_matvec`.** A `vmap` over 16 canonical vectors
  fuses into a transpose kernel that spills registers and crashes ptxas
  (`ptxas fatal: Internal compiler error`, 94 test errors inside the
  `lax.while_loop` in `find_nullspace_vectors`). `lax.map` keeps kernels small.
  Comment in place at the definition.
* **Harmonic profiles do not generalise off-axis.** Section 7.2 measured them as
  never worse, but on stellarators the pure-FD atom with harmonic own-axis
  profiles degrades badly: W7-X 88 -> 152 iterations at 16x32x32. Keep the
  arithmetic profiles and `wx_cut`.
* **W7-X map: the toroidal sign must be auto-detected** (`det(DF) > 0`); W7-X
  resolves to `sign = -1` and hardcoding `+sin` yields a NaN geometry that a
  `jac.min() <= 0` guard does NOT catch — test `np.isfinite` first.
* **Do not rebuild a `DeRhamSequence` at data resolution** for a geometry fit:
  stride-1 on a 50^3 file stalls for hours, single-core.
* Read a job's FINAL line. `pytest -q` output is block-buffered; a partial dot
  count says nothing about progress.

## Test cost

1065 s -> 800 s. What worked: `p=3 -> 2` in the conftest fixture (matvec is
O(N p^4)), and moving the tensor/CP-fit and core-Schur assembly out of the
session fixture into `test/experimental/`. What did NOT work: cutting the map
sample grid 40^3 -> 16^3, worth **0.09 s**. `--durations=15` is now on; use it.

Remaining hot spots: `torus_seq` setup 175 s, and two geometry tests
(`test_greville_interpolation_R_Z` 63 s, `test_spline_map_approximates_analytic`
32 s) at ~12% of the suite, untouched.
