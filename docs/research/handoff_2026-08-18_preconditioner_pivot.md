# Handoff 2026-08-18 — production preconditioner pivot

Suite green at **179 passed, 1 skipped, 4 xfailed** (800 s); experimental suite
**30 passed** (256 s, opt-in). Lint at the 50-error pre-existing baseline.

## Production is now: raw_kron masses + Jacobi Laplacians

| | preconditioner | diagonal |
| --- | --- | --- |
| masses k=0..3 | `raw_kron` (Kronecker on the raw grid, `E+` transfer) | -- |
| Laplacians k=0..3 | Jacobi | **closed form at every k** |

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
| tensorized weak term | **implemented**, see below | 8e-16 vs its own model |

`L_0 = S_0`, so the k=0 Laplacian Jacobi was already probe-free. k>=1 carried
the weak term and was probed at O(N) applies; that is now closed form too.

## The tensorized weak-term diagonal — DONE

`_laplacian_diaginv` is probe-free at every k. `MRX_LAPLACIAN_DIAG_PROBE=1`
restores the old probe as an exactness oracle for tests and A/Bs.

New code, all in `mrx/preconditioners.py` under
"Closed-form diagonal of the WEAK term":

| symbol | what |
| --- | --- |
| `build_extracted_laplacian_diagonal` | `diag(E L_k E^T)`, k>=1 — the entry point `_laplacian_diaginv` calls |
| `build_weak_term_diagonal` | `diag(E W_k E^T)` alone |
| `build_weak_term_raw_diagonal` | the raw-DOF closed form |
| `_extraction_projector_kron_terms` | exact Kronecker expansion of `Pi` |
| `_rank1_diagonal_split` | closed-form rank-1 split of a positive 3-D diagonal |
| `_kron_mass_model_1d` | factored out of `build_mass_raw_kron_factors`; one model, two callers |

### The algebra, and the one thing the earlier sketch got wrong

The sketch above assumed `M` is a pure Kronecker product. **It is not.** The
raw_kron model — the same object the production mass preconditioner inverts — is

    M~ = Lam (A_r x A_t x A_z) Lam ,   Lam = sqrt(diag(M) / diag(A_r x A_t x A_z))

a Kronecker product SANDWICHED by a non-separable diagonal. Writing out
`M G M^-1 G^T M` there are six such diagonals, two per mass. The two OUTERMOST
multiply the finished diagonal pointwise, so they are **kept exact and cost
nothing**. The remaining four are interior — they land between a 1-D mass and
the incidence — and a diagonal does not push through a Kronecker factorization,
so they must be separated.

Missing this is not a subtle error: dropping the inner `Lam` silently gives a
**95% median** diagonal error. The split is rank-1 and closed form (axis
averages, no iteration, no CP fit, no extra term pairs); **geometric** (log)
averaging beats arithmetic 2-20x at every k and is the default. Cost of the
split, against the same expansion with every `Lam` exact: median
2.4e-2 / 5.3e-3 / 2.5e-3 at k=1/2/3.

Note `K = Sig (x)_a Cinv_a Sig` with `Sig = Lam^-1` is *exactly* the inverse of
the raw_kron model, so "the mass preconditioner" and "the Kronecker model of the
mass, inverted" are the same object here. Nothing is modelled twice.

### Pi is NOT optional and NOT diagonal

`Pi = E^T (E E^T)^{-1} E` for the LOWER extraction absorbs both pseudoinverses
of `B = (E+)^T K E+`. Two cheap surrogates were measured and **both fail**, on a
10x8x6 toroid at k=1, relative to the exact expansion's 3.1e-2 max:

| Pi model | p99 | max |
| --- | --- | --- |
| bulk indicator mask (separable, free) | 8.9e-1 | 9.0e-1 |
| exact leverage diagonal `diag(Pi)` | 9.1e-1 | 9.2e-1 |
| **exact Kronecker expansion** | **3.1e-2** | **3.2e-2** |

The error sits exactly on the near-axis rows, i.e. where a Jacobi diagonal
matters most. The exact expansion is cheap because the polar ring is radially
thin: `Pi` is the identity on bulk DOFs, zero on dropped ones, and on the ring a
z-invariant block whose SVD in the `(i_r, j_r)` vs `(i_t, j_t)` grouping has rank
<= `ring_depth^2 <= 4` per component pair. **Truncating nothing** — verified to
2e-16, and 0.11 s to build at 12x16x12. Every structural assumption (bulk is a
radial slab, one zeta per coupled row, the ring block is z-invariant) is checked
against the actual `E` and raises rather than silently degrading.

### Coupled rows

The `n_polar * n_zeta` coupled extracted rows would need off-diagonal raw
entries of `W`. They are taken EXACTLY instead, by one apply each — a few
hundred applies against the probe's one per extracted row, and it puts the exact
value where the mass model is least accurate. Warm the apply on a concrete
vector before `lax.map`: the matrix-free mass plan is host-built and raises
`TracerArrayConversionError` if constructed inside the trace.

### Accuracy, and why it is enough

Against the exact probe, toroid 8x8x6 p=3: **median 2-3.5%, p90 6-15%, max
12-32%**, every entry positive. Positivity is structural, not luck — the
construction is `diag(X A^-1 X^T)` with `A` SPD, so it is SPSD however bad the
split gets.

The bound that matters: if `d~_i / d_i` lies in `[1/rho, rho]` the two
preconditioned operators are spectrally equivalent within `rho^2`, so CG
iterations grow by at most `rho`. At the measured worst case `rho = 1.32` that
is <=1.32x **worst case**, ~1.03x at the median. Jacobi is a factor-of-2
instrument; a 3% diagonal is not what limits it.

### A/B: `scripts/debug/laplacian_jacobi_ab.py`

Four arms into the same shifted-Jacobi CG: `none`, `probe` (the exact oracle,
`MRX_LAPLACIAN_DIAG_PROBE=1`), `stiff` (weak term DROPPED — the cheap fallback;
undefined at k=3, where `S_3 = 0` and the Laplacian *is* the weak term), and
`closed`. Writes JSON after every row via `--out`; `--build-only` skips CG and
times/compares the builds instead. Launch with `slurm/job_diag_run.sh`.

Both sweeps below: 8x16x8, p=3, gpu-h100, eps=1e-4, CG tol 1e-8, cap 3000.
Iterations, with build seconds in brackets.

**Toroid** (slurm 16343230, `outputs/diag_jacobi_ab/toroid_8x16x8_p3.json`)

| k | bc | n | none | probe | stiff (weak dropped) | closed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | free | 2344 | 3000 (cap) | **389** [4.5] | 1713 [2.0] | **391** [7.0] |
| 1 | dbc | 2088 | 3000 (cap) | **221** [2.7] | 450 [0.7] | **220** [5.1] |
| 2 | free | 2320 | 3000 (cap) | **181** [2.3] | 3000 (cap) [0.3] | **184** [5.6] |
| 2 | dbc | 2192 | 3000 (cap) | **351** [2.1] | 3000 (cap) [0.2] | **347** [5.6] |
| 3 | free | 768 | 418 | **94** [0.6] | n/a | **95** [3.1] |
| 3 | dbc | 768 | 919 | **213** [0.5] | n/a | **213** [3.1] |

**W7-X** (slurm 16343231, `outputs/diag_jacobi_ab/w7x_8x16x8_p3.json`)

| k | bc | n | none | probe | stiff (weak dropped) | closed | closed/probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | free | 2344 | 3000 (cap) | **1082** [4.5] | 3000 (cap) [1.8] | **1245** [7.1] | 1.15 |
| 1 | dbc | 2088 | 3000 (cap) | **403** [2.7] | 1085 [0.7] | **405** [5.2] | 1.00 |
| 2 | free | 2320 | 3000 (cap) | **701** [2.4] | 3000 (cap) [0.3] | **752** [5.7] | 1.07 |
| 2 | dbc | 2192 | 3000 (cap) | **586** [2.2] | 3000 (cap) [0.2] | **645** [5.6] | 1.10 |
| 3 | free | 768 | 637 | **119** [0.5] | n/a | **142** [3.2] | 1.19 |
| 3 | dbc | 768 | 1410 | **284** [0.5] | n/a | **317** [3.1] | 1.12 |

Three conclusions:

1. **On the toroid the closed form matches the exact probe iteration for
   iteration** — within +-3 at every k and both BCs, and *lower* than the probe
   in two of six. The 2-3.5% median diagonal error costs nothing there.
2. **On W7-X it costs 0-19%, median ~11%.** That is the rank-1 split of the
   inner `Lam` showing up exactly where the separability notes said it would —
   the theta-zeta plane. It is a real, measurable penalty and it is the number
   to beat; it is also comfortably inside the `rho`-bound and nowhere near
   changing the viability of the method.
3. **Dropping the weak term is not an option, and W7-X makes that emphatic.**
   `stiff` fails to converge in 3000 iterations at k=2 on both geometries and
   both BCs, and on W7-X it also fails at k=1 free. Where it does converge it is
   2-4.4x worse. The weak term has to be there; this is what the closed form
   buys.

**The build column does not favour the closed form at n~2300, and that is
expected.** There the probe is ~2300 kernel-launch-bound applies at ~2 ms —
cheap *because the problem is tiny*. Almost none of the closed form's 3-7 s is
the O(N) algebra (`Pi` terms 0.11 s, 18 term pairs at k=1, the pair sum ~0); it
is fixed overhead: two `build_mass_diagonal` JIT compiles at levels k and k-1,
plus the compile for the coupled-row applies. A constant, against a probe cost
that grows as `n_ext x (cost per apply)`.

**Measured crossover** (`--build-only`, toroid, p=3, gpu-h100, four resolutions;
build seconds, and the relative error of the *shifted Jacobi diagonal actually
used* -- `1/(diag(L) + eps/diag(M))` -- not of the weak term alone). All four
jobs have now landed.

Build seconds, probe / closed:

| k | bc | 8x16x8 (n~2.3k) | 12x24x12 (n~8.7k) | 16x32x16 (n~21k) | 20x40x20 (n~43k) | speedup at 20 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | free | 15.0 / 7.9 | 27.8 / 8.5 | 88.2 / 8.7 | **271.1 / 9.3** | **29x** |
| 1 | dbc | 4.5 / 5.2 | 15.8 / 5.6 | 72.7 / 5.9 | **250.3 / 6.3** | **40x** |
| 2 | free | 6.5 / 5.7 | 15.3 / 6.0 | 56.8 / 6.1 | **185.6 / 6.5** | **29x** |
| 2 | dbc | 4.7 / 5.7 | 13.3 / 5.9 | 53.6 / 6.2 | **180.4 / 6.3** | **29x** |
| 3 | free | 3.2 / 3.1 | 3.7 / 3.2 | 5.6 / 3.2 | **11.5 / 3.5** | 3.3x |
| 3 | dbc | 2.9 / 3.1 | 3.5 / 3.2 | 5.3 / 3.3 | **11.2 / 3.5** | 3.2x |

Relative error of the preconditioner entries, median (max):

| k | bc | 8x16x8 | 12x24x12 | 16x32x16 | 20x40x20 | median rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | free | 1.6e-3 (2.8e-2) | 6.1e-4 (2.4e-2) | 4.6e-4 (2.2e-2) | 3.6e-4 (2.1e-2) | O(h) |
| 1 | dbc | 3.1e-3 (6.5e-2) | 9.6e-4 (3.4e-2) | 5.5e-4 (2.3e-2) | 4.5e-4 (2.1e-2) | O(h) |
| 2 | free | 1.0e-2 (4.5e-2) | 5.6e-3 (3.3e-2) | 3.1e-3 (3.4e-2) | 1.9e-3 (3.4e-2) | O(h^2) |
| 2 | dbc | 1.7e-2 (8.2e-2) | 6.9e-3 (4.1e-2) | 3.4e-3 (3.4e-2) | 2.0e-3 (3.4e-2) | O(h^2) |
| 3 | free | 1.7e-2 (4.0e-2) | 8.0e-3 (3.6e-2) | 4.0e-3 (3.6e-2) | 2.6e-3 (3.5e-2) | O(h^2) |
| 3 | dbc | 2.5e-2 (7.3e-2) | 9.3e-3 (3.6e-2) | 4.7e-3 (3.6e-2) | 2.9e-3 (3.5e-2) | O(h^2) |

Three things fall out, and the third is new:

1. **The crossover is past by 12x24x12 (n~8.7k) and then it runs away.** The
   closed form's build time is flat under an 18x growth in DOFs -- 7.9 -> 9.3 s
   at k=1 free, 5.2 -> 6.3 dbc, 5.7 -> 6.5 at k=2 -- while the probe grows
   ~3.1-3.4x per resolution step (it is `n_ext` applies, each also getting more
   expensive). By 20x40x20 the closed form is **29-40x faster at k=1/2** and
   3.2x at k=3, where `n` is 3x smaller. Extrapolating the probe one more step
   puts it near 15 minutes per build; that is the wall this removed.
2. **The model gets better under refinement, and the rate is clean.** Median
   error falls at every k and both BCs across all four points: `O(h)` at k=1 and
   `O(h^2)` at k=2/3 (fitted per step; the first 8 -> 12 step is faster than
   asymptotic in every row). p90 falls with it -- k=1 free goes 2.2e-2 ->
   4.9e-3. The support-averaged weights do become more locally separable as
   elements shrink, so the rank-1 split of the inner `Lam` costs *less* at
   production resolution, exactly as hoped. This de-risks the W7-X gap.
3. **The max does NOT refine away -- it plateaus.** Yesterday's two-point read
   said the max falls too; with four points that is wrong. It settles at
   **2.1e-2 (k=1), 3.4e-2 (k=2), 3.5e-2 (k=3)** and is flat or very slightly
   rising over the last two steps. Two tells about what that row is: once the grid
   is fine enough the free and dbc runs report the **same max to 13 digits**
   (k=2 and k=3 from 16x32x16 on, k=1 at 20x40x20; at coarser grids they still
   differ), so in the limit the worst row is a shared INTERIOR row, not a
   boundary artefact; and being resolution-independent it is a fixed geometric
   feature, not a discretisation error --
   i.e. a spot where the metric weight is genuinely not rank-1 in the
   theta-zeta plane, which is what the separability notes predict. It is the
   same term the W7-X gap is made of, and it will not be refined away; it has to
   be modelled (see the `(diag(M)/diag(M^))^2` rescale below).

Practically this is good news anyway: worst-case `rho` at production resolution
is **1.034**, down from the 1.08 quoted at 12x24x12, so <=1.034x iterations from
the diagonal model on this geometry.

Full data: `outputs/diag_jacobi_ab/scaling_toroid_*_p3.json`, one row flushed as
it completes. The 642x / 13658x figures earlier in this document are operation
counts and remain unmeasured; the numbers above are the real ones at these
sizes.

### What is still open

* **Close the W7-X 11% gap.** The rank-1 split of the inner `Lam` is the
  dominant error term and W7-X has now priced it: 0-19% extra iterations,
  median ~11%. One free lever is untried — splitting the inner `Lam` breaks the
  model's exact `diag(M~) = diag(M)`, and rescaling the result by
  `(diag(M)/diag(M^))^2` costs nothing because both diagonals are already closed
  form. Try that before anything with more terms in it. The scaling sweep
  sharpens the case: the *max* error plateaus under refinement (see above), so
  the worst rows are a modelling defect that resolution will not fix.
* Cache `build_mass_diagonal` / the raw_kron factors across the two levels.
  They are essentially the whole of the closed form's build cost, and raw_kron
  already pays for them elsewhere — a caching fix, not an algorithmic one.
* No test asserts the *iteration* behaviour yet; `test/test_preconditioners.py`
  pins the exact parts (`Pi` expansion) and the diagonal error against exact
  rows, and the A/B script is the iteration check.

## Traps found today (do not repeat)

* **`M` is not a Kronecker product.** raw_kron is `Lam (x) A (x) Lam`. Dropping
  the inner `Lam` when expanding `M G M^-1 G^T M` costs a 95% median error and
  looks like a plumbing bug rather than an algebra one. See above.
* **Do not approximate `Pi`.** Both cheap surrogates leave ~90% error on the
  near-axis rows. The exact expansion is 0.11 s.
* **Warm a matrix-free apply before `lax.map`.** Its element plan is host-built;
  tracing it raises `TracerArrayConversionError` from `DerivativeSpline.evaluate`,
  which reads as a JAX bug and is not one.
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

The closed-form weak-term tests add ~47 s (`test_weak_term_diagonal_matches_
exact_rows` 26 s, `test_laplacian_jacobi_diagonal_is_positive` 20 s, the six
`Pi`-expansion cases ~1.5 s total). All 8 pass.

Remaining hot spots: `torus_seq` setup 175 s, and two geometry tests
(`test_greville_interpolation_R_Z` 63 s, `test_spline_map_approximates_analytic`
32 s) at ~12% of the suite, untouched.
