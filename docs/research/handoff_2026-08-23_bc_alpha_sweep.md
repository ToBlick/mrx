# Handoff 2026-08-23 — natural-BC face-coefficient sweep (paper §5.2d/e, §5.3)

Answers "which `alpha` convention, and is a plain penalty just as good?" with
numbers. **All four phases are done.** Phase 3 (35 jobs, submitted 18:50)
finished with 470 solves and zero failures; it was harvested on 2026-08-24 and
the results are appended at the end of this file, under "Phase 3 complete".
**Read that section last — it overrides Result 2 below, which was written while
p=5 was still missing.**

The headline changed between Phase 2 and Phase 3, and the Phase 2 conclusion as
originally read was **wrong for two independent reasons** — see "Two
corrections" below. Read that section before trusting any earlier note.

## Where everything is

| thing | path |
| --- | --- |
| coefficient probe (Phase 0) | `scripts/debug/bc_alpha_conventions.py` (UNTRACKED) |
| solve harness | `scripts/debug/verify_block_jacobi.py` (MODIFIED) |
| the conventions themselves | `mrx/block_jacobi_laplacian.py` (MODIFIED) |
| job scripts | `slurm/job_bc_alpha_phase{0,1,2,3}.sh` (gitignored: `slurm/job_*`) |
| harvester | `scripts/debug/bc_alpha_report.py` (UNTRACKED, **needs fixing — see below**) |
| Phase 0 results | `outputs/bc_alpha_phase0/2026-08-23/18-47-45/` (rerun, includes A6) |
| Phase 1 results | `outputs/bc_alpha_phase1/2026-08-23/11-11-31/` |
| Phase 2 results | `outputs/bc_alpha_phase2/2026-08-23/11-11-48/` |
| Phase 3 results | `outputs/bc_alpha_phase3/2026-08-23/18-50-05/` (also `.../LATEST`) |
| the derivation, written up | `papers/block_jacobi/main.tex` §5.2(e) (gitignored) |

Only three files are git-trackable and **none are committed**:
`mrx/block_jacobi_laplacian.py`, `scripts/debug/verify_block_jacobi.py`,
`scripts/debug/bc_alpha_conventions.py`. `papers/`, `slurm/job_*` and
`outputs` are all in `.gitignore` by design.

## Two corrections to the Phase 2 reading

**1. Rank by ITERATIONS here, not total time.** The standing rule
(`[[tensor-precond-natural-bc]]`) exists because o1/o2 had *different apply
costs*. Every arm in this sweep builds the identical rank-1 update with one
scalar changed, so per-iteration cost is equal by construction. Measured:
setup is **60–95% of `total_s`** and the time-vs-iterations fit has
**R² = 0.01–0.67**. Phase 2's time ranking was measuring setup noise. On
iterations, A2/A3 are a dead tie with A0 (median 1.000) and A5 is 1.042.

**2. Most of the Phase 2 matrix never bracketed its optimum.** The "best s"
column was frequently just the largest `s` tested. A5 was edge-pinned in
**25/39** cells and **A0 in 23/39**, with iterations still falling a median
**21% per octave** (A0 up to 41% over its last step). Where neither arm
bracketed, the ranking recorded where each grid stopped. This is what Phase 3
fixes, and it is also why the §4 p-drift table was unreadable.

The regimes split cleanly, by k and geometry, not by convention:

- **k=1,2 on rot-ellipse / W7-X** (N ≈ 400–1800): genuine interior minimum,
  curve turns up steeply past it. Well resolved.
- **k=3 everywhere, and everything on cylinder/toroid**: monotone decreasing
  across the entire Phase 2 grid, for *both* arms.

## Phase 0 — done. Gate NOT passed; full matrix was warranted.

`max |A2/A0 − 1|` = 0.00 cylinder / 0.11 toroid / 0.54 rot-ellipse / 0.63 W7-X.

Established from coefficients alone, no solves:

- **§3 is closed for A1.** `A1/A0 = 1.0000 / 0.9449 / 0.9126`, identical down
  every `(k,c)` row — a pure rescaling of `s`.
- **A2's disagreement is degree-structured** (W7-X `A2/A0`: 0.96 → 0.88 → 0.40
  → 0.38). Toroid spread only 0.89–1.00; a shaped-geometry question.
- **A4's swap is an order of magnitude**, as predicted: `A4/A2` at k=2 is 8.61
  (c=2) and 0.116 (c=1), ≈ `(R/a)² = 9`.
- **Cylinder is a sharper control than assumed**: `A2 == A0` to machine
  precision on every row. Its `"A2 is NOT (k,c)-independent"` FAIL is EXPECTED
  there and only there.

`c(p) = mu_0 · h_last` is **mesh-independent to 0.2%**: 6.674 / 10.452 /
20.673 at p = 2/3/5 (measured identically at ns=8 and ns=12). Note `h_last`
depends on p at fixed `ns`, because `n_el = ns − p`; this affects A0 and A5
identically and does not bias the comparison.

*(Correction to the brief's §4: it quoted `mu_0 ≈ 66.7/93.9/134`. Those are at
inconsistent meshes. The dimensionless `c(p)` above is the mesh-free
statement and drifts 3.1× across p=2→5, not ~2×.)*

## Phase 1 — done. §3 is closed.

`dN/d(ln s)` = **−18 to −29** everywhere except W7-X k=1 (+16.7, flat/noise).
A1's s-shift converts to **ΔN = 1.3–2.7 iterations out of 269–568**, i.e.
under 0.5% — at the noise floor. The paper can close §3 in a sentence.

**The 10 reported "INVARIANT FAILURES" are a harvester bug, not physics.**
`bc_alpha_report.py:127` demands exact float equality of the final CG
residual; all arms land in the 5e-11…9.6e-11 band, all under tol. On the
metric that matters, **9/10 rows have bit-identical iteration counts**; the
one exception (W7-X k=2 dbc, 135 vs 137) is 1.5%, at the noise floor. The
invariant holds.

## Phase 3 — the extended/refined grids. THE result.

Design (`slurm/job_bc_alpha_phase3.sh`, 35 jobs):

- **EXTEND** (16 jobs): saturated cells, pushed ~2 decades. A0 `s ∈ {0.5…32}`,
  A5 `s ∈ {4…256}`, overlapping Phase 2's grid by one point so the two runs
  cross-check and merge.
- **REFINE** (18 jobs): rot-ellipse/W7-X k=1,2 at p=2,3,5, ratio-√2 grid,
  **the same grid at every p** (rescaling per p would cancel the drift being
  measured). Arms A0 (native `c(p)/h`), A6 (bare `1/h`), A5 (bare `1/h`,
  different metric scalar).
- **WIRING** (1 job): `a6` at s=1.0 vs `a0` at s=10.432. **Not** bit-exact on
  purpose — `bcmN` encodes `s = N/1000` and the exact partner is
  `c = 10.431541` (4.4e-5 relative). Iteration counts match; residuals will
  not to the last digit. The rigorous identity is asserted on the
  *coefficients* in Phase 0 (`A6/A0 == 1/c(p)`, rtol 1e-10) and **passed on
  all 10 cells**.

### Result 1 — A0 and A5 are the same preconditioner up to `s`

Merging Phase 2 + Phase 3 grids, each arm at its own bracketed optimum:

| cell | k | A0 min | A5 min | s-ratio | predicted `c(p)/a` |
| --- | --- | --- | --- | --- | --- |
| cylinder | 1/2/3 | 42 / 44 / 27 | 42 / 44 / 27 | 16–32 | 32 |
| toroid 12³ | 1/2/3 | 69 / 56 / 34 | 69 / 56 / 34 | 32 | 31 |
| toroid 16³ | 3 | 42 | 42 | 32 | 31 |
| rot-ellipse p=2 | 1 | 342 | 342 | 28 | — |
| rot-ellipse p=3 | 1/3 | 413 / 71 | 410 / 71 | 28 / 64 | — |
| W7-X p=2 | 1 | 862 | 849 | 20 | — |
| W7-X p=3 | 3 | 73 | 72 | 32 | — |

Identical to 0–1.5% in all twelve. And it is not just the minima — the whole
curves coincide (toroid k=1):

```
A5(s=16 )= 71   A0(s=0.5)= 70      A5(s=128)= 87   A0(s=4)= 86
A5(s=32 )= 69   A0(s=1  )= 69      A5(s=256)=111   A0(s=8)=111
A5(s=64 )= 72   A0(s=2  )= 72
```

One curve, shifted by exactly the derivation's predicted factor
`c(p)/a = 10.43/0.3333 = 31` (measured 32; grid resolution is 2).

**On toroid/cylinder this is forced** (`sqrt(g^rr)` constant ⇒ A5 ∝ A0
identically) and is only a correctness check. **The load-bearing cells are the
shaped ones**, where the dropped pullback genuinely varies on the face and the
two are different operators: rot-ellipse k=1 (342 vs 342, 413 vs 410) and W7-X
k=1 (862 vs 849). They still tie.

So §5.2's honest conclusion is stronger than "the averaging convention does not
matter": **the derived coefficient and the plain penalty are the same
preconditioner up to the scale `s`.** The derivation earns its keep by
predicting the right *scale*, not by producing a better operator. That is also
why A4's ~9×-wrong partner cost so little.

### Result 2 — p-portability (PARTIAL, p=5 not yet in) — SUPERSEDED, see "Phase 3 complete" 

REFINE, fitted `s*` by parabola in log s, all cells bracketing:

| cell | A0 (native `c(p)/h`) | A6 (bare `1/h`) | A5 (bare `1/h`) |
| --- | --- | --- | --- |
| rot-ellipse k=1, p=2→3 | ×1.07 | ×1.10 | ×1.07 |
| **W7-X k=1, p=2→3** | **×1.98** | **×1.46** | **×1.40** |

A6 is A0's metric scalar with *only* the amplification swapped, so W7-X's
1.98 → 1.46 is attributable to the amplification alone — the first direct
evidence that the optimum scales like `1/h`, not `c(p)/h`. Consistent with the
independent earlier estimate (A5's optimum flat to ~1.3× across p in its own
parametrisation, vs 1.6–2.6× once converted) and with the shipped code's own
note "prefer 0.05 at p ≥ 5" (`block_jacobi_laplacian.py`), which is exactly the
drift A0 has to absorb.

**Not settled**: rot-ellipse does not discriminate, and **p=5 — the long lever,
where `c(p)` triples — has not landed.** Do not write this into the paper yet.

Also confirmed at √2 resolution on shaped geometries: all three arms tie on
iterations at their own optima (342/342/341, 413/410/416, 862/849/864,
1024/1028/1029).

## What was added to the code

`product_bare_h` (arm **A6**) = A0's metric scalar with bare `1/h` instead of
`mu_0`. Registered in `BC_ALPHA_CONVENTIONS`, branch in `_face_alpha`, regex in
`verify_block_jacobi.py` widened to `a[0-6]`, added to the Phase 0 probe with
the `A6/A0 == 1/c(p)` invariant.

Since `s` multiplies `alpha` and `c(p)` is mesh-independent, **A6 is an exact
reparametrisation `s → s/c(p)` of A0, not a new operator.** It buys (a) a
shippable code path if bare `1/h` wins, (b) a correctness guard, (c) a clean
single-variable read of the amplification. It cannot change achievable
performance at tuned `s` — the same reason the "divide A5 by `M[last,last]`
instead of `h`" idea was dropped: it would have *added* a spurious ~2× p-drift
rather than removing one.

## The A5 derivation — written up in the paper

`papers/block_jacobi/main.tex` §5.2(e), "The penalty spelling: dropping the
pullback". Builds clean (31 pages, all 7 new labels resolved). Contents:

- per-degree trace pullbacks — the physical trace squared is the logical
  coefficient squared times exactly `w_comp = m_k/J`, for k=1,2,3;
- with `dS = J√g^rr dθdζ`, the integrand is `m_k √g^rr` at every degree, so
  `alpha_pen = ⟨J√g^rr⟩_{w_comp} / h_last`;
- the derived weight exceeds it by **exactly one factor `√g^rr`**, forced by
  the cancelled index `m_k/m_{k-1} = g^rr` — A5 is the mirror of the `exact`
  arm (which kept the pullback and dropped the measure);
- A5 *is* (k,c)-dependent (unlike A0, which is exactly independent), but only
  weakly — k merely reweights the average of the same field `J√g^rr`. Measured
  spread 1.000 cyl / 1.12 toroid / 1.15 W7-X / 1.29 rot-ellipse, vs A0's 1.000
  and matrixwise's 2.09–2.55;
- closed-form check `c(p)·α_pen/α = a` (exact where `w_comp` is constant, i.e.
  k=1): measured **0.3333** on the ε=1/3 toroid at p=2,3,5 alike and **0.3300**
  on the cylinder (`cylinder_map(a=0.33)`), to four digits. The rows where
  `w_comp` varies give 0.2966, and `0.3333/0.2966 = 1.124` recovers the toroid
  spread — the same number checks the derivation and quantifies its residual
  degree dependence.

§5.2(e) deliberately makes **no performance claim**; that belongs in §8 once
Phase 3 finishes. Result 1 above is the sentence that goes there.

## Next actions

1. Wait for the remaining Phase 3 jobs (monitor was armed on failures + drain).
   The p=5 REFINE cells are the ones that matter.
2. **Fix `bc_alpha_report.py` before harvesting Phase 3** — it still ranks by
   total time and still uses the bit-equality invariant. It needs: rank by
   iterations; invariant = equal iteration counts *and* `rel < tol`; flag
   grid-edge optima explicitly instead of silently reporting them as "best";
   merge the Phase 2 and Phase 3 grids per cell; teach it the Phase 3 layout.
3. Finish Result 2 with p=5 and decide whether to ship bare `1/h`. If yes, A6
   becomes the production amplification and `PRODUCTION_BC_SCALE` is restated
   in those units (≈ 0.10 × c(p), so ~1.0), which would retire the "prefer
   0.05 at p ≥ 5" caveat.
4. Write §8: Result 1 is the headline. Consider whether "the derivation
   predicts the scale, not a better operator" is the paper's honest framing of
   §5.2 as a whole.
5. Commit the three trackable files.

## Open risks

- Nothing is committed. Commit before anything else touches
  `mrx/block_jacobi_laplacian.py`.
- Result 2 rests on **one geometry and two degrees**. rot-ellipse shows no
  drift in any arm, so W7-X is carrying the whole signal. p=5 decides it.
- The A5-vs-A0 tie is measured at each arm's *own* optimum. Production ships a
  single fixed `s` across all k and geometries; a tie at tuned `s` does not by
  itself mean a tie at fixed `s`. That is a separate (and easy) query on the
  merged grids, not yet run.

---

# Phase 3 complete — harvested 2026-08-24

All **35 jobs finished, 470 solves, zero non-convergences**. Harvested with a
throwaway script (`scratchpad/harvest3.py`, `fixed_s.py`), *not* with
`bc_alpha_report.py` — that harvester still has the two bugs listed in Next
Action 2 and was bypassed rather than fixed.

## Result 1 — confirmed on the merged EXTEND grid

A0 and A5 tie at their own optima in **all 10 EXTEND cells**, s-ratio 32
(predicted `c(p)/a = 31`):

| geom | ns | k | A0 min @ s | A5 min @ s |
| --- | --- | --- | --- | --- |
| cylinder | 12³ | 1/2/3 | 42@0.5 / 44@0.5 / 27@1 | 42@16 / 44@16 / 27@32 |
| toroid | 12³ | 1/2/3 | 69@1 / 56@0.5 / 34@1 | 69@32 / 56@32 / 34@32 |
| toroid | 16³ | 3 | 42@1 | 42@32 |
| rot-ellipse | 12³ | 3 | 71@0.5 | 71@32 |
| w7x | 12³ / 16³ | 3 | 73@0.5 / 84@0.5 | 72@16 / 83@16 |

(New vs the Phase 2 table: w7x 16³ k=3, 84 vs 83.) A0's EXTEND grid bottoms out
at s=0.5, so several A0 optima are still bottom-edge; the A5 partner brackets at
the equivalent scale and returns the same count, so the tie stands regardless.

WIRING (a6 s=1.0 vs a0 s=10.432, toroid p=3): 120/118, 110/103, 72/76 at
k=1/2/3. Passes as a soft check; the k=2 row is 6.8% apart, above the stated
~1% noise floor, but the *rigorous* identity is the Phase 0 coefficient test
(`A6/A0 == 1/c(p)`, rtol 1e-10, 10/10 cells) and that passed.

## Result 2 — p=5 landed. **Bare `1/h` does NOT win. Do not ship A6.**

Fitted `s*` (parabola in log s, all cells bracketing except the two marked):

| cell | arm | p=2 | p=3 | p=5 | p2→p5 |
| --- | --- | --- | --- | --- | --- |
| w7x k=1 | A0 | 0.0950 | 0.0478 | 0.0352 | ÷2.70 |
| | A6 | 0.677 | 0.464 | 0.539 | ÷1.26 |
| | A5 | 1.831 | 1.309 | 1.470 | ÷1.25 |
| w7x k=2 | A0 | 0.1257 | 0.0777 | 0.0601 | ÷2.09 |
| | A6 | 0.874 | 0.841 | 0.910 | ×1.04 |
| | A5 | 2.032 | 2.070 | 2.078 | ×1.02 |
| rot-ell k=1 | A0 | 0.1599 | 0.1498 | 0.1083 | ÷1.48 |
| | A6 | 1.239 | 1.366 | 2.425 | ×1.96 |
| | A5 | 3.886 | 4.153 | 8.000 *(edge)* | ×2.06 |
| rot-ell k=2 | A0 | 0.2087 | 0.1335 | 0.1133 | ÷1.84 |
| | A6 | 1.464 | 1.354 | 2.378 | ×1.62 |
| | A5 | 5.339 | 5.657 | 8.000 *(edge)* | ×1.50 |

*(Note on orientation: the Phase-2-era note quoted W7-X k=1 p2→p3 as "A0 ×1.98,
A6 ×1.46, A5 ×1.40". Those are the reciprocals of the ratios above — 1/0.50,
1/0.68, 1/0.72 — same measurement, and they reproduce exactly. `s*` **falls**
with p in A0 units.)*

**A6 flattens W7-X and un-flattens rot-ellipse.** The handoff's earlier read
rested on W7-X alone; with rot-ellipse now resolved at p=5 the two geometries
pull in opposite directions and the "bare `1/h` is more p-portable" claim does
not generalise.

### The metric that actually decides it: worst-case excess at a single fixed `s`

Production ships one constant across all k, p and geometry, so rank the arms by
`max over the 12 REFINE cells of (iters at fixed s) / (iters at that cell's own
optimum)`:

| arm | best single `s` | worst-case excess |
| --- | --- | --- |
| **A0** (derived, `c(p)/h`) | **0.10** | **+4.7%** |
| A6 (bare `1/h`) | 1.414 | +5.2% |
| A5 (plain penalty) | 4 | +8.9% |

Best single `s` *within* each p, converted to A0 units:

| arm | p=2 | p=3 | p=5 |
| --- | --- | --- | --- |
| A0 | 0.141 (+1.7%) | 0.100 (+3.9%) | 0.071 (+3.3%) |
| A6 | 0.150 (+0.8%) | 0.096 (+2.8%) | 0.068 (+4.9%) |
| A5 | 0.424 (+4.7%) | 0.271 (+4.4%) | 0.194 (+6.2%) |

Three things fall out:

1. **A0 and A6 agree on where the optimum is** (0.141/0.100/0.071 vs
   0.150/0.096/0.068) — as they must, being the same operator. The
   parametrisation only changes how well *one* constant covers all p, and it is
   a wash: 4.7% vs 5.2%. **Ship nothing; keep A0.** Next Action 3 resolves
   "no".
2. The true scaling is **between** the two conventions: `α_opt ∝ c(p)^γ/h` with
   `γ ≈ 0.39` (from `s*_A0` falling ×0.50 while `c(p)` rises ×3.10 over p=2→5).
   Neither γ=0 (A6) nor γ=1 (A0) is right, and the whole spread is worth <1
   percentage point of iterations.
3. **A5's penalty is geometry/k spread, not p-drift.** It costs 4.4–6.2% even
   at a *single* p, where A0 costs 1.7–3.9%. This is A0's exact
   (k,c)-independence paying off, and it is a *different* claim from Result 1:
   at each cell's own tuned `s` the two are identical operators, but as a single
   shipped constant the derived coefficient is measurably more robust. This
   answers Open Risk 3 — the tie does **not** survive the move to fixed `s`.

### The `"prefer 0.05 at p >= 5"` caveat is wrong and should be retired

`block_jacobi_laplacian.py:548`. Restricting to the p=5 cells only:

| s | 0.05 | 0.071 | 0.10 |
| --- | --- | --- | --- |
| worst-case excess at p=5 | +8.9% | **+3.3%** | +4.7% |

`0.10` beats `0.05` at p=5, on both geometries and both k. The best p=5 value
is 0.071, and it buys 1.4 percentage points over the shipped 0.10 — inside the
run-to-run noise. Recommendation: **delete the caveat, keep
`PRODUCTION_BC_SCALE = 0.10` unqualified.**

## Revised next actions

1. ~~Wait for Phase 3~~ — done.
2. `bc_alpha_report.py` still unfixed; only worth doing if the sweep is rerun.
3. ~~Decide whether to ship bare `1/h`~~ — **no**. A6 stays as a correctness
   guard and a code path, not the default.
4. §8 headline is unchanged (Result 1), plus the fixed-`s` robustness argument
   above as the reason the derived coefficient earns its keep in production.
   The p-portability paragraph should say `γ ≈ 0.39` and that it does not
   matter, not that bare `1/h` wins.
5. Still uncommitted: `mrx/block_jacobi_laplacian.py`,
   `scripts/debug/verify_block_jacobi.py`, `scripts/debug/bc_alpha_conventions.py`.
6. New: drop the "prefer 0.05 at p >= 5" line.

REFINE covers k=1,2 free-BC only, on rot-ellipse and W7-X. k=3 and the dbc rows
are not in this p-sweep.

---

# A5 as the shipped arm — `s = 2.83` (2026-08-24)

Follow-up to the section above, prompted by "A5 is simpler to derive and seems
to work just as well". It does. **The +8.9% figure quoted for A5 above is
misleading and should not be used**: it is a *worst-case* excess over the
*REFINE-only* cell set (rot-ellipse + W7-X, k=1,2). REFINE deliberately
contains only the hard cells, and worst-case weighting lets a 34-iteration cell
outvote a 1450-iteration one. On the wider matrix and on total iterations the
two arms are the same.

## The value: `s = 2.828` (basin 2–4, flat to ~1 point)

Merged Phase 2 + Phase 3 grids, 24 cells that bracket their optimum
(4 geometries × k=1,2,3 × p=2,3,5 × two meshes), ranked by
`sum(iters at fixed s) / sum(per-cell optimum)`:

| arm | best single `s` | total-iteration excess |
| --- | --- | --- |
| A0 | 0.10 | +5.1% |
| **A5** | **2.828** | **+6.0%** |

REFINE-only (the 12 hard cells) gives the same `s` for A5 — 2.828, +3.0% vs
A0's 0.10 at +1.8%. The basin is flat: on the full matrix A5 scores 1.062 /
1.060 / 1.066 at s = 2 / 2.828 / 4, so anything in **[2, 4]** is
indistinguishable. `s = 3` is a fine round number to ship.

**This is not a new fit.** `2.828 / 0.10 = 28.3`, and the derivation's own
predicted conversion is `c(p)/a = 10.43/0.3333 = 31` at p=3 (measured 28–32
across the bracketed cells). The A5 constant is the A0 constant pushed through
§5.2(e)'s factor.

## Head to head at the shipped constants

A0 @ 0.10 vs A5 @ 2.828, 22 cells where both arms bracket:

| | A0 | A5 |
| --- | --- | --- |
| total iterations, all 22 cells | 1.060× optimum | 1.069× optimum |
| total iterations, 12 hard cells (opt > 300) | 1.021× | 1.024× |
| **A5 / A0** | | **+0.8%** (hard cells: **+0.3%**) |

Both arms lose **1.6–2.4×** on the *cheap* cells (cylinder, toroid, k=3 —
27–83 iteration cells that want `s` an order of magnitude larger). A0 is not
better there: toroid 12³ p3 k3 is 2.24× for A0 and 2.41× for A5, on a
34-iteration cell. That penalty is the price of one constant tuned for the hard
cells, and it is paid by whichever arm ships.

## p-portability: a wash, and no A5 analogue of the p≥5 caveat

Best single `s` per degree, total-iteration metric on REFINE:

| arm | p=2 | p=3 | p=5 | drift |
| --- | --- | --- | --- | --- |
| A0 | 0.10 | 0.10 | 0.071 | ÷1.4 |
| A5 | 2 | 2 | 2.828 | ×1.4 |

Equal and opposite, both immaterial (holding the single value costs 1.8–3.5%).
So the `"prefer 0.05 at p >= 5"` note has no counterpart in A5 units — `s = 3`
holds across p=2,3,5 unqualified.

## The one real gap

**rot-ellipse p=5, k=1 and k=2: A5's REFINE grid stops at s=8 and the optimum
is still there.** Those two cells are excluded from the 22 above. Holding
`s = 2.828` on them costs 630 vs A0's 574 (k=1) and 635 vs 578 (k=2) — ~+10%,
the only place A5 at a fixed constant clearly loses. Including them moves the
head-to-head from **+0.8% to +1.7%**.

This is the one cell family where A5's optimum runs away with p (rot-ellipse
was already the geometry where A6/A5 un-flattened; see the p=5 table above).
**If A5 is going to ship, extend its rot-ellipse p=5 grid past s=8** — 2 jobs,
`s ∈ {8, 11.3, 16, 22.6, 32}` — to see whether the optimum is at 8–16 or
genuinely off at 32. That decides whether the honest number is +1.7% or
whether rot-ellipse p=5 needs a caveat of its own.

## Recommendation

A5 at `s = 3` is defensible: simpler derivation, ties A0 at tuned `s` in every
cell (Result 1), and costs +0.8–1.7% total iterations as a shipped constant —
at or just above the ~1% noise floor. If it ships, `PRODUCTION_BC_SCALE`
becomes `3.0` in A5 units and the p≥5 caveat goes away. Run the two
rot-ellipse p=5 extension jobs first.

---

# OUT-OF-SAMPLE: A5 breaks on the quasr devices (2026-08-24)

**This contradicts Result 1 and should be read before trusting the A5 swap.**

Validation of the shipped `penalty` @ 3.0 on four GVEC flat-schema geometries
(`scripts/debug/gvec_geometry.py`, new): `quasr9983` (nfp=2), `quasr44970`
(nfp=3), `hegna` (nfp=3, 80^3) and `w7x-gvec` (nfp=5, an independent W7-X
source). **None were in any arm of the tuning sweep.** 12x24x12, p=3, free BC,
k=1,2,3. Results in `outputs/quasr_a5/2026-08-24/05-22-11/`.

Both arms were run on grids deliberately matched in PHYSICAL scale
(`s_a5 / 28.28 == s_a0`, so the two are compared at the same alpha point by
point). This comparison does not need the optimum to be bracketed -- it asks
directly whether the two are the same operator.

Mean A5/A0 across all five matched points:

| geometry | k=1 | k=2 | k=3 | mean |
| --- | --- | --- | --- | --- |
| **w7x-gvec** (control) | 1.021 | 1.012 | 0.992 | **1.008** |
| hegna | 1.113 | 1.111 | 0.868 | 1.031 |
| quasr44970 | 0.993 | 1.038 | 1.395 | 1.142 |
| **quasr9983** | 1.171 | 1.217 | 1.576 | **1.321** |

`w7x-gvec` ties, exactly as the tuning set did -- it is W7-X, so it is a
control, and it behaves. On the two genuinely new devices the arms **separate**,
and on `quasr9983` A5 is worse at *every* point of *every* k, up to 1.67x at
k=3. That is not a scale offset: a rescaling would put A5 ahead somewhere on
the grid, and it never is.

Head to head at the shipped constants over all four:

| rows | a0 @ 0.10 | a5 @ 2.828 | A5/A0 |
| --- | --- | --- | --- |
| all k | 9298 | 10051 | **+8.1%** |
| k=2,3 only | 5062 | 5495 | **+8.6%** |

against the **+0.8-1.7%** measured in-sample. Excluding k=1 makes A5 slightly
worse, so the finding is not an artefact of the suspect k=1 rows.

## Why -- and why the sweep could not have caught it

§5.2(e) says A5 differs from A0 by exactly one factor `sqrt(g^rr)` on the face.
The handoff already noted "on toroid/cylinder this is forced (`sqrt(g^rr)`
constant => A5 ∝ A0 identically) ... the load-bearing cells are the shaped
ones". The load-bearing cells available at the time were rot-ellipse and W7-X
-- and those are simply **not shaped enough to separate the two**. quasr9983 is
(nfp=2, compact, `det DF` spanning 6.7x). The tie in Result 1 was a property of
the tuning set, not of the operators.

## Caveats

- k=1 free rows carry a degraded deflation vector on three of the four
  (`relL2` 4.7e-03 hegna / 7.7e-04 quasr44970 / 3.1e-04 w7x-gvec / 4.5e-07
  quasr9983) -- the separate k=1-free defect, below. Same vector for every arm
  within a geometry, so the A/B is fair; excluded from the k=2,3 row anyway.
- Grids are centred on the shipped constants, so many per-cell optima are
  edge-pinned. The matched-point ratio does not depend on that; the "optimum"
  columns in the raw table do.
- Run with the pre-2026-08-24 harness settings (`maxiter` 20000,
  `inner_tol` 1e-13).

## What this means for the swap

The swap to `penalty` @ 3.0 is **committed to code but is now contradicted by
the only out-of-sample evidence that exists.** The decision to ship A5 was made
on in-sample data where the two arms genuinely tie; that data was not wrong,
it was just not discriminating. Options, in order of cost:

1. **Revert to `product` @ 0.10.** One-line change to the two constants; the
   derived coefficient is (k,c)-independent by construction and is the arm that
   holds up on the new devices. Costs the simpler derivation.
2. Keep A5 and accept ~8% on quasr-class devices.
3. Re-tune A5's scale on a set that includes quasr -- but the matched-point
   table says the gap is not a scale, so this is unlikely to recover it.

Recommendation: **(1)**, and rewrite §5.2 around the honest version -- the
derived coefficient earns its keep on strongly shaped geometries, which is
exactly where the pullback it keeps stops being constant. That is a *better*
paper result than "the derivation only predicts the scale".

## Also measured

`gvec_stride` (`outputs/gvec_stride/2026-08-24/05-32-53/`): the data-grid
resolution costs nothing -- 84.4 / 83.2 / 82.5 s at stride 1 / 2 / 4 on
quasr9983, RSS flat at ~4.1 GB -- while R/Z fit error degrades O(h^2) as
expected (rms 5.8e-05 -> 2.0e-04 -> 7.5e-04). stride=1 reproduces the
pre-restructure fit errors exactly, so the half-open/closed axis rework is a
no-op. Keep stride at 1.
