> **Status:** resolved; outcome in audit_2026-08-25_production.md
> **Read this for:** how the atom went into production and what was deleted
> **Do not read for:** current names; BlockJacobi* became MetricLumping* on 2026-08-25

# Moving the block-Jacobi Laplacian into production — a simplification plan

2026-08-22. Goal, in Tobias's words: **one production method with minimal free
parameters**, with **probed jacobi** (saddle points: probe `D M^-1 D^T` using
the production mass preconditioner as `M^-1`) as the reference to measure
against.

Evidence base: `docs/research/natural_bc_coefficient_handoff.md` (§15-§19).

## 0. Where production is today

| | today | after |
| --- | --- | --- |
| Laplacian precond, k=0 | `kind='tensor'` (k=0 ONLY) | one path, k=0..3 |
| Laplacian precond, k=1,2,3 | **silently falls back to point jacobi** (`apply_laplacian_preconditioner`, `kind='auto'`) | the same one path |
| free parameters | -- | **one documented constant, no required kwargs** |
| `BlockJacobiLaplacian` | **17 constructor kwargs + 9 env vars** | 0 required kwargs, 1 env override |
| reference baseline | point jacobi | probed jacobi on `D M^-1 D^T` |

The k>=1 fallback to point jacobi is the thing being replaced. That is where
the 2x lives (§15.1: rot-ellipse n=20 k=1 free, total time 26.3s vs 53.7s).

## 1. Phase 0 — resolve two defaults BEFORE landing (blocking, cheap)

1. **`ktilde_mode` -- ANSWERED 2026-08-22, and the class default is WRONG.**
   `outputs/diag_ktilde/`, four geometries x k=0..3 x free and dbc at n=12
   (bcp10, the landing scale). **`roundtrip` loses every single row**, 28 of 28:

   | geom | k=1 free | k=1 dbc | k=2 free | k=3 free |
   | --- | --- | --- | --- | --- |
   | cylinder | 70 / **689** | 68 / 632 | 75 / 455 | 63 / 274 |
   | toroid | 110 / **789** | 92 / 551 | 95 / 634 | 63 / 391 |
   | rot-ellipse | 416 / **1549** | 218 / 878 | 453 / 1362 | 142 / 566 |
   | w7x | 1103 / **6540** | 320 / 2918 | -- | -- |

   (honest / roundtrip). Median ratio **4.37x**, worst 9.84x, and `roundtrip`
   is worse than plain JACOBI in several rows. k=0 is identical (1.00) on every
   geometry, as it must be: `ktilde_mode` only differs on DERIVATIVE axes and
   k=0 has none -- the same reason `bc_scale` is inert at k=0.

   **Action: flip the default to `"honest"`** in `component_factors`,
   `build_bulk_atom` and `BlockJacobiLaplacian`. No production caller relies on
   it; every debug script that sets it explicitly already passes `"honest"`,
   and the only `"roundtrip"` users are the `rt` A/B arms in
   `verify_block_jacobi.py` / `block_jacobi_spectrum.py`, which should keep
   working for the record.

   This is exactly the kind of implicit default that §3's two hidden metric
   factors hid behind, and it would have shipped a 3-10x regression.
2. **`bc_scale = 0.10`.** Settled (§19.4): minimax 1.11 over the shaped
   geometries and over n>=16, across 82 cells / 4 geometries / k=1,2,3 /
   n=8..32 / p=2..5. No further sweeps needed.

## 2. Phase 1 — the production surface

```python
BlockJacobiLaplacian(seq, operators, k, dirichlet)      # no kwargs required
seq.apply_laplacian_preconditioner(v, k, dirichlet, kind='block')
```

Fixed internally, documented, NOT knobs:

| setting | value | why |
| --- | --- | --- |
| `bc_entry` | `"ibpd"` | the derived coefficient; metric-free `mu_0`, degree-independent, verified at p=2,3,5 (§12.4) |
| `bc_scale` | `0.10` | empirical kappa-balance constant (§14.1, §17.5, §19.2) |
| `lumped` | `"diag"` | every result in the handoff assumes it |
| everything else | off | `extra_rings=outer_rings=pin_trace=coarse_rings=0` |

`bc_scale` stays overridable by `MRX_BJ_BC_SCALE` for diagnostics, and gets a
docstring saying it is EMPIRICAL, that the optimum drifts down with n and p,
and that p>=5 or very high resolution may prefer 0.05.

## 3. Phase 2 — delete the refuted machinery (this is the simplification)

The module grew ~810 lines in `9eff2ff` alone. Almost all of it is refuted.

**Delete outright** -- each was measured and lost; §-refs are the evidence:

| what | kwargs / env | verdict |
| --- | --- | --- |
| `wibp`, `wibpd`, `woodbury`, `wdiag` | `bc_entry` | exact 2-D face shape: 1.35-1.9x WORSE at the corrected scale (§14.3) |
| `ibpr`, `ibpf` | `bc_entry` | the cross term: `ibpr` is INDEFINITE everywhere (§12.2) |
| `ibp`, `ibps`, `face`, `direct`, `exact` | `bc_entry` | superseded by `ibpd`; `exact` is worse than NO term at k=1/2 free (§12.11) |
| the pin | `pin_trace`, `pin_mode`, `pin_set` | measured no-op on the high outliers (§11) |
| tangential penalty | `MRX_BJ_TANG_BC` | needs a fitted number per geometry; superseded |
| mode-dependent beta | `MRX_BJ_TANG_MODE`, `_FLAT` | **BROKEN**, measured (§9) |
| Dirichlet-side term | `MRX_BJ_DBC_BC` | catastrophic even at 5% (§9) |
| Nitsche | `MRX_BJ_NITSCHE` | diverges (§9) |
| p=1 jump diagnostics | `MRX_BJ_D0_SCALE`, `MRX_BJ_D0_FORM` | one-off, fixed and landed |
| indefiniteness floor | `MRX_BJ_BC_CLAMP` | only existed for `ibpf` |

That is **11 `bc_entry` variants -> 2** (`"ibpd"`, `False`) and **9 env vars ->
1**.

**Park, do not delete** -- real but not default:

* `coarse_rings/modes/set/mode/trunc` (`fm`): a genuine further 1.18-1.32x on
  total time (§15.1), but 5 parameters, memory linear in `n_dof`, an
  additive-vs-hybrid correctness trap already hit once, `m95 ~ n_t/3`, and
  untested at k=0 and under Dirichlet. Move to its own module, opt-in.
* `outer_rings` / `extra_rings`: retired as production (§15.1 -- 4-6x off on
  total time, `o2` slower than jacobi on W7-X) but KEEP as the diagnostic they
  were: the exact-boundary upper bound on what any boundary method can buy.
  Demote to a clearly-labelled diagnostic argument.
* `radial="modal"`, `core_mode="atom2d"` -- check for users, then park.

## 4. Phase 3 — the reference: probed jacobi

The structure already exists: `SchurPreconditionerSpec(inner=..., outer=...)`
in `mrx/preconditioners.py`, with `inner` defaulting to the production
`raw_kron` mass preconditioner and `outer` to jacobi, and
`schur_diag_mode='raw_kron_probe'` already in `MassPreconditionerSpec`.

What to build: the **probed diagonal of `D M^-1 D^T`**, with `M^-1` the
production mass preconditioner rather than an exact inverse, and wire it in as
the benchmark baseline in place of point jacobi.

Why it is the right reference, not just a better one:

* it is what a saddle-point solve actually has available, so the comparison is
  honest about cost;
* point jacobi is a WEAK baseline that flatters everything -- it degrades
  7.6-12.2x over p=2..5 while every block arm grows only 2.3-2.8x (§12.7), so
  ratios against it improve with p for reasons that have nothing to do with the
  method;
* it shares the mass preconditioner with the rest of production, so it needs no
  new tuning of its own.

## 5. Phase 4 — tests (currently ZERO for this module)

`test/` has no `block_jacobi` coverage. This is the main risk in promoting it.
Minimum set, all cheap:

1. **The Dirichlet invariant**: `bc_entry` must vanish identically under an
   essential condition -- every dbc row bit-identical across `bc_scale` values.
   This is the check that caught the periodic-axis bug (§"Gotcha", memory).
2. **k=0 has no boundary term**: `trace_components(0) == ()`, and `bc_scale` is
   a no-op at k=0 (§17.6).
3. **`bc_scale=0` reproduces `bc_entry=False`** bit-for-bit.
4. **SPD**: `min eig(P) > 0` for k=0..3, free and dbc, on two geometries. This
   is what `ibpr` failed (§12.2) and a rank-one update can in principle break.
5. **Iteration regression**: k=1 Dirichlet (non-singular, so no nullspace
   needed), against a recorded count with a generous band.
6. **The term earns its place**: k=3 FREE (also non-singular, and the term is
   live and large there) asserted against the SAME solve with the term off.
   Tests 1-5 all run where the term is inert or only check its PRESENCE; this
   is the only one that guards its MAGNITUDE.

DONE 2026-08-22: `test/test_block_jacobi_laplacian.py`, 9 tests, all passing,
~38 s of test time on top of the shared session fixture (which the rest of the
suite already pays for). Each inertness check carries a POSITIVE CONTROL -- the
same comparison where the term is live -- so none of them can pass against a
preconditioner that simply lost the boundary term. Thresholds come from the
measured separation (inert ~1e-11, live 2.5e-3 at the weakest k), not from
round numbers.

## 6. Phase 5 — later, optional

* **Remove the last constant.** §17.5 shows a depth-3 ring-block match predicts
  `bc_scale` per geometry with no solve: ordering exact and magnitudes within
  one sweep point on all four geometries (cylinder 0.55, toroid 0.55,
  rot-ellipse 0.15, W7-X 0.06). Needs a cheaper probe (coarse ring / few modes)
  to pay at production sizes. If it works, `bc_scale` stops being a parameter
  at all and adapts to k, p, n and geometry for free.
* **`fm` as opt-in** for the hardest cases, once tested at k=0 and dbc.

## 7. What NOT to do

* Do not re-derive the coefficient. It is right (§12.4, and the cylinder proves
  it: zero coupling => `s=1` optimal => the derivation is exact, §19.2).
* Do not make `bc_scale` k-, p- or geometry-dependent. Buys <=15% and turns one
  constant into a table (§16.6, §17.6).
* Do not chase the W7-X k=1 floor. Bracket-censored at 0.03 in five sweeps, the
  curve is flat there, and 0.10 costs <=1.11 (§16.5, §19.4).
* Do not revive the banded capacitance. It exists to make `outer_rings`
  affordable, and `outer_rings` is 4-6x off the pace before any banding
  (§15.1).

## 8. Order and effort

| phase | effort | blocking? |
| --- | --- | --- |
| 0. `ktilde_mode` A/B | **DONE -- flip the default to `honest`** | was blocking |
| 1. production surface | small | -- |
| 2. delete refuted machinery | **DONE 2026-08-22 -- 2099 -> 1332 lines** | -- |
| 3. probed-jacobi reference | medium (new code) | no -- can follow |
| 4. tests | small | should precede the default flip |
| 5. auto `bc_scale`, `fm` opt-in | later | no |

Suggested landing order: **0 -> 4 -> 1 -> 2 -> 3**. Tests before the default
flip, deletion after the new default is proven, and the reference last so it
measures the finished thing.


---

## 9. PHASE 2 AS EXECUTED (2026-08-22)

`mrx/experimental/block_jacobi_laplacian.py`: **2099 -> 1332 lines (-37%)**.

| surface | before | after |
| --- | --- | --- |
| `bc_entry` variants | 11 | **2** (`"ibpd"`, `False`) |
| `MRX_BJ_*` env knobs | 9 | **1** (`MRX_BJ_BC_SCALE`) |
| constructor kwargs | 17 | 12 |

Deleted outright (each measured and lost; § refs in the handoff): the
`ibpr`/`ibpf` cross-term corrections, `wibp`/`wibpd`/`woodbury`/`wdiag` and the
whole capacitance/Woodbury path, `ibp`/`ibps`/`face`/`direct`/`exact`, the
`pin` (all three kwargs and its `core_rows` plumbing), Nitsche consistency, the
`tg` tangential penalty, `tm`/`mode_beta_correction` (BROKEN, measured), the
`atom2d` 2-D ring atoms, `radial="modal"`, `TransferK3Preconditioner`, and the
two p=1 diagnostic knobs (the fix they compared is landed).

KEPT deliberately:

* `_weak_inverse_amplification`, `_face_metric_scalar`, `face_operator` -- no
  longer reachable from the class, but they are what `bc_alpha_compare.py`,
  `edge_vector_check.py` and `face_weight_probe.py` use to VALIDATE the derived
  coefficient. Those are the scripts to re-run if the derivation is questioned.
* `extra_rings` / `outer_rings` and `probe_core_block` -- the exact-boundary
  diagnostic and the upper bound on what any boundary method can buy.
* `BlockJacobiMass` -- now wired as the mass swap candidate (§4).
* `coarse_*` (5 kwargs, `fm`) -- STILL THERE. Extracting it to its own module
  is a refactor rather than a deletion, and it is a measured 1.18-1.32x option.
  **This is the remaining Phase 2 work.**
* `ktilde_mode` -- kept as a knob (correctly defaulted now). Deleting
  `"roundtrip"` would also remove the `ratios`/`alpha` machinery threaded
  through `build_ring_atom` and `_fd_apply_3d`; worth doing, not worth
  bundling with this.

**Method note.** `ruff` catches F821 at module scope but **did not** catch three
dead calls inside the jitted `m_apply` closure (`_cap_arrays`, `_apply_cap_jax`,
`apply_ring_atom`). Grep for the deleted names as well as linting -- the memory
note about F821-after-deletions understates it.

---

## 10. THE NEW STACK, MEASURED (overnight 2026-08-22, 40 jobs, no failures)

`outputs/diag_newstack/` (h) + `outputs/diag_newstackp/` (p). Block-Jacobi
Laplacian at `bc_scale=0.10` with **block_jacobi as the mass**, four geometries,
n = 8..32, p = 2..5. 168 cells.

### 10.1 `PRODUCTION_BC_SCALE = 0.10` SURVIVES THE MASS SWAP

Penalty of the flat 0.10 against each cell's own optimum, free BC, p=3:

| geom | k | n=8 | 12 | 16 | 20 | 24 | 28 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cylinder | 1 | 1.19 | 1.10 | 1.02 | 1.02 | 1.01 | 1.01 | **1.00** |
| toroid | 1 | 1.29 | 1.19 | 1.11 | 1.03 | 1.02 | 1.01 | **1.01** |
| rot-ell | 1 | 1.04 | 1.00 | 1.00 | 1.00 | 1.02 | 1.01 | **1.00** |
| w7x | 1 | 1.00 | 1.01 | 1.04 | 1.07 | 1.05 | 1.07 | **1.07** |
| cylinder | 3 | 1.15 | 1.05 | 1.00 | 1.00 | 1.01 | 1.00 | **1.00** |
| toroid | 3 | 1.24 | 1.11 | 1.05 | 1.00 | 1.00 | 1.00 | **1.00** |
| rot-ell | 3 | 1.07 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.01** |
| w7x | 3 | 1.14 | 1.02 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |

**Median 1.01 over 96 free cells, and it IMPROVES with resolution** -- every
geometry converges to 1.00-1.07 by n >= 16. The worst case anywhere is 1.55,
and every case above 1.2 is at **p=2** or n=8:

| p-sweep, n=12, 0.10 penalty | p=2 | p=3 | p=4 | p=5 |
| --- | --- | --- | --- | --- |
| toroid k=3 | **1.55** | 1.11 | 1.00 | 1.00 |
| toroid k=1 | 1.39 | 1.19 | 1.07 | 1.05 |
| cylinder k=3 | 1.32 | 1.05 | 1.00 | 1.01 |
| w7x k=3 | 1.27 | 1.02 | 1.00 | 1.00 |
| rot-ell k=1 | 1.00 | 1.00 | 1.00 | 1.00 |

So the constant is worst exactly where the problem is cheapest (p=2, coarse
meshes) and is essentially free at production resolution and degree. **No
re-fit.** The argmin trend is also unchanged by the swap: it drifts DOWN with n
(rot-ellipse 0.30 -> 0.06, W7-X pinned at 0.06), toroid stays highest, cylinder
converges to 0.10 -- the same ordering §16.2 and §19.3 found under raw_kron.

### 10.2 THE HEADLINE: block@0.10 vs point jacobi, new stack

| geom | k, bc | n=8 | 16 | 24 | 32 |
| --- | --- | --- | --- | --- | --- |
| toroid | 1 free | 0.29 | 0.19 | 0.14 | **0.12** |
| toroid | 3 free | 0.44 | 0.22 | 0.16 | **0.13** |
| cylinder | 1 free | 0.28 | 0.25 | 0.23 | **0.21** |
| w7x | 3 free | 0.54 | 0.31 | 0.23 | **0.19** |
| rot-ell | 1 free | 0.57 | 0.50 | 0.41 | **0.36** |
| w7x | 1 free | 0.56 | 0.67 | 0.60 | **0.52** |

**Median 0.31 over 120 cells; 0.25 for n >= 24.** The ratio IMPROVES with
refinement everywhere -- 3-8x at production resolution. The hardest cell
remains W7-X k=1 free (0.52), which is the same case that has been hardest
throughout.

Read against `probed_jacobi` rather than `jacobi` these are ~20% less
flattering at k=1/2 on the shaped geometries (§ reference A/B) -- but probed
jacobi is O(N) applies to build and is a reference, not a candidate.

### 10.3 Status

| phase | |
| --- | --- |
| 0 `ktilde_mode` | done -- default was 3-10x wrong |
| 1 production surface + dispatch | done |
| 2 delete refuted machinery | done, 2099 -> 1154 lines; `fm` moved out |
| 3 probed-jacobi reference | done |
| 4 tests | done, 13 |
| 5 mass -> block_jacobi | **done**, with one recorded regression (§ commit 5f0d69e) |

OPEN, in rough priority order:

1. **Model the weak term under the new mass.** `build_weak_term_diagonal` is
   calibrated for raw_kron; under block_jacobi its error goes from ~2-4% median
   to 22% (k=1 dbc), which is why
   `test_weak_term_diagonal_matches_exact_rows` now skips. Costs `kind='jacobi'`
   1-10% today. It is a shared production artefact and should not stay wrong.
2. **Make the BlockJacobiMass BUILD jit-safe**, removing the warm-up
   requirement. Tobias's observation is that it should be possible: the probe
   vectors are one-hot on STATIC row indices and no structural property depends
   on the metric payload. The three blockers are mechanical -- `np.linalg.inv`
   and `np.linalg.eigh` -> `jnp`, and the data-dependent `keep_w` mask ->
   a static-shape `jnp.where(|w| > tol, 1/w, 0)` pseudo-inverse.
3. Auto-compute `bc_scale` from the §17.5 ring-block match, removing the last
   fitted constant.
4. `fm` (`mrx/experimental/block_jacobi_coarse.py`) is untested at k=0 and dbc.
