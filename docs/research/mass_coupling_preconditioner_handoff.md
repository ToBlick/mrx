> **STATUS: RESEARCH / EXPERIMENTAL — superseded for production by `docs/PRODUCTION.md`.**

# Vector-mass (k=1/2) preconditioner: off-diagonal coupling & lump fidelity

**Resume file: `docs/mass_coupling_preconditioner_handoff.md`.** Settled findings and
dead ends (separability rule, bcheb regression, greville route, etc.):
`docs/preconditioner_lessons.md`.

## Problem

The production greville mass preconditioner is a **per-component block-diagonal lump**
(`_build_greville_mass_block_factors`, `mrx/preconditioners.py:2417`; production default,
`greville=True` hard-set at `:2644`). On W7-X it needs **k=1/2 = 72–80 CG iters** vs
**k=0/3 = 9–12** (scalars). Toroid: all k = 8–12. Goal: get k=1/2 near the scalar level.

### How the lump works (and what it drops)
Each V1 basis fn is a scalar spline × one unit covector, so
`M1[I,J] = ∫ φ_I φ_J · (G⁻¹)_{c(I),c(J)} · J` — **one** metric entry per (I,J), picked by the
component pair. Same-component → diagonal `g^{cc}`; cross-component → **off-diagonal** `g^{rθ}`
etc. (k=2 uses `G` not `G⁻¹`, weight `g_{cc'}/J`.)

The lump builds only the 3 diagonal blocks, each `D^{-1/2}(M0_r⁻¹⊗M0_t⁻¹⊗M0_z⁻¹)D^{-1/2}`
with `D` = the diagonal metric weight collocated **pointwise at that component's Greville
abscissae** (k=1 `jac*minv[:,c,c]`, k=2 `metric[:,c,c]/jac`; `preconditioners.py:2471-2474`).
No averaging (that's the legacy CP route `_build_diagonal_tensor_block_factors`). The
**off-diagonal metric `g^{rθ},g^{rζ},g^{θζ}` is never evaluated** → coupling is dropped.
Effective weight on entry (I,J) is `√(D_I D_J)` = **geometric mean of the two Greville-node
weights**, which can sit up to `p` knot-spans apart. Inner 2 radial rings removed by surgery.

## What we found (the gap is TWO comparable halves)

Ceiling test (`scripts/debug/greville_mass_coupling_ceiling.py`): solve, with the SAME greville
preconditioner, the **block-diagonal operator** `M_bd = Σ_c P_c M P_c` (off-diagonal blocks
zeroed) vs the true `M`. `P_c` from the preconditioner's own surgery slices
(`_surgery_slices_k1/k2`).

| geom | k | Mfull_it | Mbd_it | ‖offdiag‖/‖diag‖ |
|------|---|---------|--------|------------------|
| w7x  | 0/3 | 12 / 9 | =Mfull | ~1e-16 (scalar) |
| w7x  | 1 | 76–80 | **31** | 0.20 |
| w7x  | 2 | 72–75 | **30** | 0.03 |
| toroid | 1/2 | 11 | 12 | ~6e-3 / 3e-5 |

- **Coupling half: 31 → 75** (factor ~2.4). Removing off-diagonal blocks more than halves iters.
- **Diagonal-lump-fidelity half: 12 → 31** (factor ~2.6). Even coupling-free, the *vector*
  per-component weight `g^{cc}·J` varies much more than scalar `J` on W7-X → the pointwise
  geometric-mean lump is a worse block solve. This is independent of coupling.
- **k=2 coupling is tiny (0.03) yet Mfull=74** → k=2's cost is almost ALL diagonal lump
  fidelity; k=1 is split between the two.
- Toroid control: coupling≈0 ⇒ `M_bd≈Mfull≈12` — harness validated, projections tile correctly.
- **Reaching k=0 needs BOTH** a coupling method AND a better diagonal lump. Block-SGS alone
  caps around ~31.

### Coupling structure (from `greville_lumping_diagnostics.py`, section A)
W7-X normalized off-diagonal `|g_ij|/√(g_ii g_jj)`: **ρ-θ ≈ 0.49** (p90 0.82, max 0.94),
ρ-ζ & θ-ζ ≈ 0.19 — for both G and G⁻¹. Diagonal-approx spectral error 0.64 mean, >1 tail.
**Correction to prior memory:** the coupling that drives the preconditioner is **ρ-θ**, not θ-ζ.
Toroid: metric analytically diagonal (~1e-16).

## Scripts (all in scratch; same mount as /scratch/tblickhan/mrx)

- `scripts/debug/greville_lumping_diagnostics.py` — geometry diagnostics from g at quad/Greville
  points: (A) off-diagonal coupling, (B) sub-cell weight CoV, (B2) geometric-mean-of-endpoints
  error per axis/lag, (C) production CG iters. `--r-drop 2` excludes the surgery rings from all
  sections (added after the first W7-X run; the ρ-axis tails in the first W7-X output are
  axis-pollution — ignore them).
- `scripts/debug/greville_mass_coupling_ceiling.py` — the ceiling test above, now with
  **timings** (assembly + per-solve wall + ms/iter, warm-up then timed, block_until_ready) and
  `--csv`. `M_bd` does 3 mass-applies per matvec (the P_c projection sum), so its ms/it should
  be ~3× Mfull's — watch whether fewer iters actually beats more-expensive iters in wall-clock.
- `slurm/job_mass_coupling_ceiling.sh` — per-geometry sbatch (debug-gpu, extremedata, 1 GPU, 4h).
  Submits ONE job per geometry in order cylinder→toroid→w7x; on the 1-job debug limit use
  `GEOMETRIES=w7x bash slurm/job_mass_coupling_ceiling.sh` (runs both ns=(12,24,12) and
  (16,32,16) inside one job). CSVs → `outputs/mass_coupling_ceiling/<stamp>/`.

## Results 2026-07-07 morning (post-diagnosis; block-SGS TRIED AND REGRESSED)

The timings job landed AND the recommended block-SGS was built and measured
(`outputs/mass_coupling_ceiling/2026-07-07/05-37-13/`, W7-X 12×24×12, GPU). The SGS code was
then **removed** from `greville_mass_coupling_ceiling.py` in the 06:11 rewrite — the numbers
below are the record (full log: `.../05-37-13/slurm_logs/w7x.log`).

| k | bc | Mfull | Mbd (ceiling) | Schur+Jacobi | Schur+SGS[rt,rz,tz] |
|---|----|-------|---------------|--------------|---------------------|
| 1 | dbc  | 75 it, 1.08s | 31 it, 1.97s | 66 it, 1.06s | **291 it**, 1.34s |
| 1 | free | 80 it, 1.06s | 31 it, 1.93s | 69 it, 1.03s | **350 it**, 1.35s |
| 2 | dbc  | 72 it, 1.15s | 30 it, 2.11s | 59 it, 1.07s | **106 it**, 1.23s |
| 2 | free | 75 it, 1.13s | 30 it, 2.09s | 62 it, 1.07s | **117 it**, 1.23s |

- **Block-SGS with lumped mixed-mass off-diagonal blocks REGRESSED** (k=1: 291–350 vs 75–80
  baseline) despite converging (solerr ~1e-9). Lumped off-diagonal block rel-errors were
  rθ≈0.17, rζ≈0.11, θζ≈0.11 — apparently too inaccurate: an SGS sweep with wrong L/U *adds*
  error instead of removing coupling. k=2 (tiny coupling 0.03) also regressed 72→106.
  Conclusion: **the lumped-L/U form of SGS is dead**; if SGS is retried it needs the TRUE
  off-diagonal blocks (e.g. via full-M applies, 1 extra M-apply per sweep) — untested.
- **Schur+Jacobi variant mildly helps**: 75→66 (k=1), 72→59 (k=2), and wins wall-clock
  slightly. Small but free.
- `M_bd` ms/it ≈ 4.4× `Mfull` (63.6 vs 14.4), worse than the 3× model — even reaching the
  ~31-iter block-diagonal ceiling would only break even on the wall at these sizes.
- 16×32×16: job killed after k=0 (assembly 166s, k0 = 11 it — iteration count h-flat so far);
  k=1/2 scaling question still open.
- Follow-up comparison `scripts/debug/mass_jacobi_vs_greville.py`
  (`outputs/jacobi_vs_greville/2026-07-07/06-03-12/`): plain point-Jacobi-PCG on the mass vs
  the production greville lump, W7-X + toroid, both resolutions. Greville wins iterations
  massively (e.g. W7-X k=0: 11 vs 630) but its ms/it is 14–100× Jacobi's, so **wall-clock is
  roughly a tie** (W7-X k=1 16³: greville 1.13s vs jacobi 2.00s; k=0: jacobi 0.41s vs
  greville 0.41s; toroid k=1: 0.96s vs 1.65s). Jacobi iters are h-growing (630→692), greville
  h-flat — greville's advantage should widen with resolution.

## Next steps / decision tree (revised after the SGS regression)

- ~~Build block-SGS with lumped off-diagonal blocks~~ — **done, regressed, dropped** (above).
- **Attack the diagonal lump first** (was the "other half"): replace the pointwise Greville
  sample with a **support-integrated / true-weighted-mass diagonal** `D_I = ⟨w⟩ over Φ_I`
  (kills the geometric-mean-of-far-apart-nodes error). Now the primary lever: it's the whole
  story for k=2 (coupling 0.03) and half of k=1, and it doesn't add per-iteration cost.
- Coupling half, if revisited: SGS with TRUE off-diagonal blocks (full-M apply per sweep), or
  dominant-pair 2×2 on r-θ only (rθ≈0.49 is the big one). Note the cost model is unfavorable
  (Mbd 4.4×/it); any coupling treatment must beat ~31 it × ~3–4× cost vs 75 × 1.
- Chebyshev/polynomial over block-Jacobi: previously regressed (bcheb=0 reverted default;
  see `docs/preconditioner_lessons.md`).

Stiffness (curl-curl / div-div P_A) is deferred — user has a separate plan there. The k=0
Laplacian multigrid prototype is a separate thread: `docs/laplacian_mg_k0_plan.md`.
