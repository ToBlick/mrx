# Plan: vacuum-field projection & harmonic 2-form — what belongs in the library

Context: `scripts/debug/w7x_vacuum_bfield_project.py` + `quasr_covcontra_verify.py`
grew a lot of *general* machinery while getting the W7-X / QUASR vacuum fields to
project cleanly. This plan walks the pipeline stage by stage, marks each piece
**LIB (exists)**, **LIB (to add)**, or **SCRIPT-only**, and proposes the library
API so the scripts shrink to I/O + reporting.

See also: `docs/w7x_vacuum_bfield_handoff.md` (the validated recipe & numbers) and
memory `geometry-df-primitive-storage`.

## 0. The two workflows (shared spine)

- **A. Project a sampled B onto V2** — interpolate a gridded Cartesian field and
  L2-project it onto the 2-form space.
- **B. Vacuum field as a harmonic 2-form** — solve for the k=2 (DBC) harmonic
  representative, seeded by the logical dζ covector.

Both share: load geometry → build an **orientation-correct, length-normalized**
map → assemble the sequence/operators. A only adds a factorized grid-load; B only
adds the harmonic solve.

## 1. Pipeline stages

| # | Stage | What it does | Location today | Target |
|---|-------|--------------|----------------|--------|
| 1 | Read gridded h5 | `eval_points,R,Z,B` + `n_*`,`nfp` attrs → arrays | SCRIPT | SCRIPT (+ optional generic loader) |
| 2 | Interpolatory grid fit | gridded R/Z (any field) → tensor-spline coeffs, per-axis collocation | SCRIPT (`fit_coeffs`) over LIB `_solve_tensor_collocation_axis` | **LIB (to add)**: `fit_tensor_spline` |
| 3 | Build map | `F=(R cos a, ±R sin a, Z)`, `a=2πζ/nfp` | LIB `stellarator_map` (hardcodes `−sin`) | **LIB (to add)**: orientation-safe builder |
| 4 | Orientation | pick toroidal sign so `det(DF)>0` (else M2 indefinite → NaN) | SCRIPT (auto-detect) | **LIB (to add)**: fold into (3) |
| 5 | Length normalization | scale R,Z so major radius `R0=1` (spectrum `O(1)`) | SCRIPT | **LIB (to add)**: `normalize_lengths` / map option |
| 6 | DF at data nodes | analytic factorized DF on the tensor grid | SCRIPT (`map_and_DF_on_grid`,`grid_eval`) | **LIB (to add)**: generalize `spline_map_F_DF_at_quad` to arbitrary grid |
| 7 | Pull-back-first | Piola k-form pullback at nodes (periodic → seam-free) | SCRIPT | **LIB (to add)**: `frame='phys'` of the grid load = pull-back-first |
| 8 | Factorized grid load | fit + eval-at-quad + integrate → dual | **LIB (exists)** `io.load_grid_field` | LIB (done) |
| 9 | Geometry storage | DF-primitive (`DF_jkl`+inv+jac); metric = property | **LIB (exists)** `geometry.py` | LIB (done) |
| 10 | Project | `M2⁻¹ · load` | **LIB (exists)** `apply_inverse_mass_matrix` | LIB |
| 11 | Harmonic seed | logical dζ=(0,0,1)→V2 (DBC), `M2⁻¹` supplies metric | SCRIPT (inline) | **LIB (to add)**: `vacuum_2form_seed` |
| 12 | Harmonic solve | inverse iteration on k=2 DBC nullspace | **LIB (exists)** `find_nullspace_vectors` + incidence/schur ops | LIB |
| 13 | Solve tolerances | `eps`,`abs_tol` — must not be absolute across scales | SCRIPT constants | **LIB (to add)**: scale-relative defaults (or rely on (5)) |
| 14 | Pushforward / error report | reconstruct at nodes, relative errors, ζ-seam profile | SCRIPT | SCRIPT |

## 2. Proposed library additions (API sketches)

### 2.0 `Frame` enum (foundational, library-wide)
Replace the stringly-typed `frame='ref'|'phys'` everywhere with an explicit enum
— clearer intent, IDE-completable, no typo-silently-wrong:
```
class Frame(enum.Enum):
    PHYSICAL = "physical"   # components in the physical (Cartesian) frame
    LOGICAL  = "logical"    # components already in the logical/reference frame
                            # (i.e. the reference k-form proxy)
# 'phys' -> Frame.PHYSICAL,  'ref' -> Frame.LOGICAL
```
Lives near the pushforward/pullback definitions (`mrx.differential_forms`, or a
small `mrx.frames`). Touches every `frame=` site: `projectors.load`, `seq.load`,
`io.load_grid_field` (and the new grid load). Decide once: accept **only** the enum
(hard switch, update all callers/scripts) vs also coerce the legacy strings during
a deprecation window. Everything below uses `Frame.PHYSICAL` / `Frame.LOGICAL`.

### 2.1 `fit_tensor_spline(axes, values, *, degree=3, types) -> coeffs`  (stage 2)
Interpolatory tensor B-spline coefficients from data on a logical grid
(`n_basis = n_data` per axis, factorized `_solve_tensor_collocation_axis`).
Already implemented *inside* `load_grid_field`; extract it so the **map fit**
(R,Z→coeffs) reuses it instead of the hand-rolled `fit_coeffs`. Place: `mrx.io`
or `mrx.projectors`.

### 2.2 Orientation-safe stellarator map  (stages 3–4)
`stellarator_map` currently hardcodes `Y=−R sin`. Add orientation handling:
```
stellarator_map(R, Z, nfp, *, orient='auto')   # 'auto' | +1 | -1
# 'auto': evaluate det(DF) on a coarse logical sample, choose the toroidal sign
#         giving det(DF) > 0.  Returns (map, sign).
```
Rationale: `det(DF)>0` is a hard requirement (indefinite M2 → NaN), and different
files need different signs (W7-X `−1`, QUASR `+1`). This is not W7-X-specific.

### 2.3 `normalize_lengths(R, Z, *, r0=None) -> (R', Z', r0)`  (stage 5)
Scale lengths so major radius `=1` (`r0` default `0.5*(Rmin+Rmax)`). Pure,
2 lines, but belongs in the library as documented non-dimensionalization: the
projection is **scale-invariant** (`B_rec=B`), so it only normalizes the
Hodge-Laplacian spectrum to `O(1)` — which is what makes fixed solver tolerances
(stage 13) transfer across geometries. Could instead be a flag on the map builder.

### 2.4 DF on an arbitrary tensor grid  (stage 6)
`geometry.spline_map_F_DF_at_quad` already evaluates F/DF at the sequence's quad
grid by sum-factorization. Generalize the eval-point set (or add
`spline_map_DF_on_grid(coeffs, axes_1d)`) so the covariant proxy (stage 7) can get
DF at the **data nodes** without `jax.jacfwd` over N points. The analytic
stellarator DF (with the `±sin` rotation block, `map_and_DF_on_grid`) is the
closed form; expose it alongside the generic spline path.

### 2.5 Physical grid load is pull-back-first, always  (stage 7) — the key lesson
Projecting a **quasi-periodic Cartesian field** (Bx,By rotate by `2π/nfp` per
period) with a periodic spline gives a ζ=0 Gibbs seam. Fix: pull the field back to
the reference frame **at the data nodes** *then* interpolate — the pulled-back
proxy is periodic; the Cartesian field is not.

**DECIDED — this is just what `frame='phys'` means for a grid load; not a separate
mode.** There is no reason to ever interpolate the physical components *before*
pulling back (that is simply wrong on a periodic domain), so `load_grid_field`
takes only two frames (see 2.0 for the enum):
- `Frame.LOGICAL`  — input is already the reference k-form proxy; interpolate + load.
- `Frame.PHYSICAL` — input is physical components; **pull back at the data nodes**
  (→ periodic proxy), interpolate, then `M_k`-weighted load. Periodic-safe by
  construction.
```
load_grid_field(axes, B_cart, seq, k, frame=Frame.PHYSICAL)
#   physical, grid  ->  proxy = pullback_k(DF_nodes, B)   (per (2.4), at data nodes)
#                   ->  interpolate periodic proxy  ->  M_k-weighted load
```
(Note: the *pointwise* `seq.load(callable, Frame.PHYSICAL)` needs no change — a
callable is evaluated exactly at the quad points, so there is no interpolation and
no seam. The pull-back-first rule is specific to **grid** loads.)

This supersedes the earlier `frame='phys_prepull'` idea. It also means the current
`load_grid_field(frame='phys')` (which interpolates the Cartesian field *then*
pulls back at quad points — the seam-y order, validated only against the pointwise
Cartesian load) must be **reimplemented** as pull-back-first.

Needs (2.4) (DF at the data nodes). Bonus: the Piola proxy `J·DF⁻¹·B = adj(DF)·B`
is finite even at the singular ρ=1 knot (no explicit `/det`), so the wall needs no
special nudging on the load side.

**DECIDED — use the genuine Piola k-form pullback, loaded with the mass-matrix
weight** (not the covariant dual):
- prepulled proxy = the *reference* k-form pullback (matches the `Pullback` class):
  k=2 → `p = J·DF⁻¹·B`, k=1 → `p = DFᵀ·B`. Both are periodic.
- load it with the **standard `M_k` inner-product weight** (k=2 → `g/J`, k=1 →
  `G⁻¹·J`) — i.e. the load is just `M_k` applied to the interpolated proxy against
  the basis. Same weight the mass matrix uses; no bespoke convention.

*Why not the covariant dual `DFᵀB` (weight `w`)?* It gives the **identical dual** —
`∫ Λᵀ(g/J)(J·DF⁻¹·B) = ∫ Λᵀ DFᵀB` (the `J` and metric cancel), so `DFᵀB`+weight-`w`
is the same load with the metric pre-absorbed onto the test side. The covariant
form is one contraction cheaper, but the Piola form is chosen for **conceptual
uniformity**: the reference proxy is then the genuine k-form pullback (its
pushforward is `B` directly — a free sanity check), and the load weight is exactly
the mass weight for every `k`. The extra contractions are irrelevant — this is a
one-time **setup** step, not a hot path.

This single option captures the whole seam-free-projection recipe as library
behavior; the script stops building the proxy by hand.

### 2.6 Vacuum harmonic field  (stages 11–13)
Wrap the seed + solve:
```
vacuum_2form(seq, *, maxiter=..., tol=...) -> (dof, iters, residual)
# seed: M2^-1 load(logical (0,0,1), k=2, dirichlet=True, frame='ref')
# refine: find_nullspace_vectors(..., k=2, dirichlet=True, x0s=[seed])
```
Also: replace the toroid-only `_toroidal_vacuum_field` initial guess in
`nullspace.py:_initial_guesses` (k=2 DBC) with this logical-(0,0,1) seed
(geometry-robust; already flagged in memory `logical-dzeta-vacuum-ic`). Make the
default `tol`/`eps` **scale-relative** (normalize by a characteristic Laplacian
eigenvalue or the M2 norm) so they behave identically across geometries — or
document that (2.3) normalization is the prerequisite.

## 3. What stays SCRIPT-only

- h5 parsing / dataset names (format-specific). A thin generic loader for the
  `eval_points/R/Z/B + n_*/nfp` layout is optional sugar, not core.
- CLI/argparse, resolution & degree choices.
- **Reporting/diagnostics**: pushforward-and-compare at the data nodes, relative
  error, ζ-seam spread. This is inherently `O(N_data)` and analysis-specific
  (it's also why the script needs `--stride` at 50³) — keep it out of the library.

## 4. Knobs reference (every parameter, where it lives)

| Knob | Meaning | Lives | Default |
|------|---------|-------|---------|
| `--h5` / dataset names | input file/layout | SCRIPT | — |
| `--nfp` | field periods (else attr) | SCRIPT→map | attr |
| `--ns`, `--p` | projection resolution / degree | SCRIPT→seq | (8,16,16), 3 |
| `--fit-degree` | interpolatory fit degree | SCRIPT→(2.1) | 3 |
| `--stride` | data-grid subsample (diagnostic cost) | SCRIPT | 1 |
| `--eval-eps` | nudge off singular ρ=1 knot | SCRIPT/(2.4) | 1e-6 |
| orientation sign | det(DF)>0 | (2.2) | auto |
| `--r0` / normalize | major-radius scaling | (2.3) | 0.5(Rmin+Rmax) |
| `frame` | `Frame.LOGICAL` / `Frame.PHYSICAL` (physical = pull-back-first for grids) | (2.0)/(2.5) | `Frame.PHYSICAL` |
| `FIT_Q` | fit_seq quad order (unused → keep tiny) | SCRIPT/(2.1) | 1 |
| `eps`,`abs_tol`,`maxiter` | harmonic inverse-iteration | (2.6) | scale-relative |

## 5. Phased implementation

0. **`Frame` enum** (2.0) — introduce `Frame.PHYSICAL`/`Frame.LOGICAL`, migrate all
   `frame=` sites (`projectors.load`, `seq.load`, `io.load_grid_field`) and scripts.
   Foundational; do it first so later phases use the enum from the start.
1. **Extract `fit_tensor_spline`** (2.1) from `load_grid_field`; repoint the map
   fit at it. Low risk, immediate dedup.
2. **Orientation-safe map** (2.2) + **`normalize_lengths`** (2.3). Small, high
   value (removes the two subtlest script foot-guns; fixes tolerance transfer).
3. **DF-on-grid** (2.4) + **`frame='phys_prepull'`** (2.5). The core reuse —
   turns the covariant-proxy recipe into a one-call library feature.
4. **`vacuum_2form`** wrapper (2.6) + scale-relative tolerances; retire the inline
   §5 and the `_toroidal_vacuum_field` guess.
5. Script collapses to: load h5 → `stellarator_map(orient='auto')` (normalized) →
   `load_grid_field(frame='phys_prepull')` / `vacuum_2form` → report.

## 6. Open design decisions

- Normalization (2.3): a standalone helper vs a flag on the map/sequence? (Leaning
  helper — keep the map dumb.)
- ~~separate `phys_prepull` mode?~~ **DECIDED: no — pull-back-first *is*
  `frame='phys'` for grid loads (§2.5). Two frames only: `ref`, `phys`.**
- ~~covariant dual `DFᵀB` vs Piola `J·DF⁻¹·B` for the prepulled proxy~~ **DECIDED:
  Piola pullback + mass-matrix weight (§2.5). Equivalent dual, a few more
  contractions, but uniform across `k` and it's a setup step, not hot.**
- **Follow-up:** the existing `load_grid_field(frame='phys')` implements the seam-y
  interpolate-then-pullback order and must be reimplemented pull-back-first (needs
  2.4). The `frame='ref'` path (validated bit-identical) is unaffected.
- Tolerances (2.6): make scale-relative *and* rely on (2.3), or pick one? (Do
  both — normalization is hygiene, scale-relative tol is a safety net.)
- Should `SequenceGeometry` expose `DF` at arbitrary points (not just quad)? (2.4)
  — decide once (2.5) needs it.
- `Frame` enum (2.0): hard switch (enum only) vs coerce legacy `'ref'`/`'phys'`
  strings during a deprecation window? Where does it live —
  `mrx.differential_forms` or a new `mrx.frames`?
```
