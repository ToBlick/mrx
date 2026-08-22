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
| 2. delete refuted machinery | medium, mostly deletion | -- |
| 3. probed-jacobi reference | medium (new code) | no -- can follow |
| 4. tests | small | should precede the default flip |
| 5. auto `bc_scale`, `fm` opt-in | later | no |

Suggested landing order: **0 -> 4 -> 1 -> 2 -> 3**. Tests before the default
flip, deletion after the new default is proven, and the reference last so it
measures the finished thing.
