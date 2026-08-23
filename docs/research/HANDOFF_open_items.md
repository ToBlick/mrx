# Handoff — open items on the block-Jacobi preconditioner

2026-08-22, end of the production-landing work. Companion:
`preconditioner_technical_note_source.md` (self-contained, everything measured).

**State: landed and green.** Commits `cc54f37 .. ead6819` on `greville-prod`.
Full suite 199 passed, 2 skipped, 4 xfailed. Nothing is half-applied.

## What is in production now

| | |
| --- | --- |
| Laplacian preconditioner | `BlockJacobiLaplacian`, k=0..3, free and dbc, via `kind='block'` (`kind='auto'` picks it when assembled) |
| its parameters | **none required.** `bc_entry="ibpd"`, `bc_scale=0.10`, `lumped="diag"`, `ktilde_mode="honest"` are defaults |
| mass preconditioner | `kind='block_jacobi'` (was `raw_kron`; `MRX_MASS_KIND=raw_kron` reverts wholesale) |
| reference baseline | `kind='probed_jacobi'` -- exact `diag(L_k)`, O(N) applies, never a candidate |
| tests | `test/test_block_jacobi_laplacian.py`, 13 |

`mrx/experimental/block_jacobi_laplacian.py` went 2099 -> 1154 lines; the class
went from 17 constructor kwargs to 7; nine `MRX_BJ_*` env knobs to one.

---

## Open items, in priority order

### 1. Model the weak term under the new mass — the only outright regression

`build_weak_term_diagonal` (`mrx/preconditioners.py`) models `D M^-1 D^T` under
the KRONECKER MASS MODEL and was calibrated when `M^-1` was raw_kron. Under
`block_jacobi` its error against the exact operator grows from **~2-4% median /
~30% max to 22% / 114%** (k=1 dbc, spline toroid 8,16,8 p=2).

Consequence: `test_preconditioners.py::test_weak_term_diagonal_matches_exact_rows`
now SKIPS unless the mass is raw_kron, with the numbers in the skip reason.
`kind='jacobi'` costs 1-10% more iterations than it used to (cylinder k=1 free
262 -> 287, W7-X k=1 free 1658 -> 1668).

I widened the max bound, hit the median, and stopped -- that is chasing a test
rather than hearing it. **The fix is to model the new mass, not to move the
bound.** `jacobi` is a shared production artefact (it is the `auto` fallback and
feeds `build_extracted_laplacian_diagonal`) and should not stay wrong.

### 2. Make the `BlockJacobiMass` BUILD jit-safe

Today the build is host-side numpy, so a COLD cache inside a traced loop dies.
That is worked around by `operators.warm_mass_preconditioner_cache`, called
before `nullspace.py`'s `while_loop`. **Any new traced entry point that solves
must warm first** -- this is a live footgun, not a settled design.

Tobias's observation is that it should not be necessary: the probe vectors are
one-hot on STATIC row indices, and no structural property of the preconditioner
depends on the metric payload -- only values do. Three mechanical blockers:

* `np.linalg.inv` per axis -> `jnp.linalg.inv`;
* `np.linalg.eigh` on the core -> `jnp.linalg.eigh`;
* the data-dependent mask `keep_w = |w| > tol*max` (dynamic shape) ->
  a static-shape pseudo-inverse, `jnp.where(|w| > tol*max, 1/w, 0.0)`, then
  `(v * inv_w) @ v.T`.

Doing this removes the warm-up requirement entirely.

### 3. Auto-compute `bc_scale`, removing the last fitted constant

`bc_scale = 0.10` is empirical. §5.2 of the companion shows a purely LOCAL
computation predicts it: form `L[R,R]` on the outer rings by probe, form the
atom's `inv(P(s))[R,R]`, and minimise the generalised condition number over `s`.
Ordering is exact and magnitudes land within one sweep point on all four
geometries (cylinder 0.55, toroid 0.55, rot-ellipse 0.15, W7-X 0.06).

Done ONCE at setup it would adapt to k, p, n and geometry for free. It needs a
cheaper probe than the full ring (a coarse ring, or a few modes) to pay at
production sizes. Script: `scripts/debug/bc_schur_effective.py`.

### 4. `fm` coverage

`mrx/experimental/block_jacobi_coarse.py` is a real 1.18-1.32x on total time
but is untested at k=0 and under Dirichlet, and its fixed mode box is
asymptotically under-resolved (`m95 ~ n_t/3`). Opt-in only. If it is ever
promoted, the box has to grow with `n`.

### 5. Smaller, and genuinely optional

* Delete `ktilde_mode="roundtrip"`. It loses every A/B row. Removing it also
  removes the `ratios`/`alpha` machinery threaded through `_fd_apply_3d`, which
  is why it was not bundled with the Phase-2 deletion.
* `bc_scale` at p=2 is the worst case anywhere (1.55 on toroid k=3). If p=2 ever
  matters, it wants ~0.30 rather than 0.10.

---

## Traps that cost real time — read before touching this code

**`ruff` does NOT catch dead calls inside a jitted closure.** After deleting the
capacitance/ring machinery, ruff passed while `_cap_arrays`, `_apply_cap_jax`
and `apply_ring_atom` were still called inside `m_apply`. All three would have
crashed at first apply. **Grep the deleted names as well as linting.**

**Do not script regex edits over the sweep harnesses.** I mangled
`verify_block_jacobi.py` twice on multi-line kwargs, and shipped four F821s in
`block_jacobi_spectrum.py` in commit `cab0d9a` (fixed in `544cc23`) because I
read a ruff run across a `git stash` cycle. Hand-edit the call sites; lint that
file immediately, not in a batch.

**Never assert bit-identity between two builds.** The dense polar core is not
bit-reproducible: two builds of the IDENTICAL configuration differ by ~1e-14 on
~1.7% of rows. Use a relative tolerance, and measure the floor with a
same-configuration control rather than guessing it.

**Every inertness test needs a positive control.** "Changing X does not move the
preconditioner" passes just as happily against a preconditioner that lost the
feature entirely. Assert the same comparison in the configuration where the
feature IS live, and check the two are orders apart.

**Test the thing production does, not the thing that is easy to call.** The mass
readiness test called the built apply with a CONCRETE array and never under
`jax.jit`. It passed against a `BlockJacobiMass` that could not be used in
production at all, and I told Tobias the swap was one line twice on that basis.

**The mass preconditioner is inside the operator.** `apply_hodge_laplacian_approx`
uses it as the inner inverse of the weak term, so changing it changes `L_k` at
k >= 1. Any mass change invalidates recorded Laplacian numbers until re-measured.

---

## Where the data is

`outputs/` on the cluster, all 2026-08-21/22:

| dir | what |
| --- | --- |
| `diag_newstack`, `diag_newstackp` | **the production stack**, n=8..32 and p=2..5 (Tables A, B) |
| `diag_massab`, `diag_massp` | mass A/B, raw_kron vs block_jacobi (Table D) |
| `diag_masslap` | the Laplacian under each mass |
| `diag_ref` | jacobi vs probed_jacobi vs block (Table E) |
| `diag_bcpspec` | the kappa-balance spectra (Table F) |
| `diag_bcschur` | the ring-block predictor (Table G) |
| `diag_bcpn`, `diag_bcplow`, `diag_bcphi`, `diag_bcpp` | the `bc_scale` sweeps, 82 cells |
| `diag_cyl`, `diag_cylp` | cylinder, the zero-coupling control |
| `diag_alphaverify` | `mu_0` metric-free and `alpha` degree-independent at p=2,3,5 |
| `diag_fmcost` | storage and build crossover (Table H) |

Prior narrative, with every dead end and why: `natural_bc_coefficient_handoff.md`
(§12-§19) and `production_simplification_plan.md`.
