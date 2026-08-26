> **Status:** resolved; executed, results in preconditioner_technical_note_source.md section 9
> **Read this for:** why every measurement before 2026-08-24 was superseded
> **Do not read for:** the results

# Sweep plan — regenerating the paper's numbers

Everything measured before 2026-08-24 is superseded. Two independent reasons,
both found on 2026-08-24:

1. **The harness solved the wrong operator.** `verify_block_jacobi.py` ran a
   bespoke CG on `apply_hodge_laplacian_approx` (`S_k + D B D^T`, `B` = one
   mass-preconditioner apply). The library's own solve is
   `apply_inverse_hodge_laplacian` -> saddle MINRES on the exact
   `S_k + D M^-1 D^T`. Absolute iteration counts from the old harness do not
   describe the production path.
2. **The production path was unpreconditioned.**
   `_materialize_default_mass_preconditioner` gated on `_tensor_available` and
   fell back to a per-DoF diagonal, so the saddle solve's lower block was not
   the production mass preconditioner. Fixed; the library default went from
   2/18 to 18/18 cells converged, with 5-30x fewer iterations.

Plus one scope change: the code now ships **only** the boundary penalty
(`alpha_k = <m_k sqrt(g^rr)> / <m_k/J> / h`, `s = 3`). The `product` /
`halves` / `matrixwise` / `product_bare_h` conventions are gone, so every arm
that named one is unrunnable.

**Nothing from the bc-alpha sweep survives as a paper number.** The
*conclusions* about the conventions do (they were relative comparisons on a
fixed operator, and the penalty-vs-derived separation on quasr reproduced on
the real solve), but every table has to be rebuilt.

Run all of it through `scripts/debug/bench_real_solves.py`, which drives the
real solve and reads iteration counts from MINRES's own `info` (`-k` converged,
`+k` not). Geometries: `toroid`, `rot-ellipse`, `w7x`, `quasr9983`,
`quasr44970`, `hegna` — the last three are out-of-sample for everything tuned
so far and are what caught the convention difference.

---

## S1 — Baseline table: against point Jacobi *(paper table i)*

The headline. `schur.outer = block` vs `schur.outer = jacobi`, both with the
fixed mass preconditioner below.

- 6 geometries x k=1,2,3 x {free, dbc} = 36 cells, `12,24,12`, p=3, tol 1e-10.
- Report iterations, build time, solve time; ratio block/jacobi.
- ~2 jobs per geometry. **Cheap. Do this first** — it is the one table the
  paper cannot do without.

## S2 — The scale, and the basin *(paper table ii)*

`s` over `{1, 1.41, 2, 2.83, 4, 5.66, 8}` on the same 36 cells.

- Confirms the `[2,4]` basin claim on the real solve. The old basin was
  measured on the approx operator with an unpreconditioned lower block, so it
  is not evidence for anything now.
- Report worst-case and total-iteration excess against each cell's own optimum.
- 7 arms x 6 geometries. The single biggest run; ~1 job per geometry.

## S3 — Degree dependence of the optimal `s` *(paper table iii, NEW)*

This is the measurement the rewritten §Results promises and we have never made.
`alpha_k` is degree dependent by construction; the question is whether a
per-degree `s_k` is worth having and whether it is geometry-stable.

- Per-`k` optimum from S2's grid, per geometry. No new solves if S2 stores
  everything.
- **The decisive plot**: `s_k^opt / s_1^opt` against geometry. If it is
  constant across devices, ship a k-table. If it tracks the spread of `J` on
  the face, then a constant table is wrong and the paper says so.
- Also report the coefficient ratio directly from
  `_face_alpha`, no solves: the (k,c) spread was 1.000 on the cylinder (where
  `J` is constant at `r=1` and the penalty is exact) against 1.29 on
  rot-ellipse. Extend to quasr/hegna. **Cheap; run alongside S1.**

## S4 — `h` and `p` scaling *(paper table iv)*

- `ns in {8,16,8 | 12,24,12 | 16,32,16}` at p=3, and p in {2,3,5} at `12,24,12`.
- toroid + w7x + quasr9983 only; k=1,2 free and dbc.
- Iteration count vs `n` at fixed `s=3` is the claim that matters (does the
  preconditioner hold up under refinement).
- Note `s` is expressed against a bare `1/h`, so there is no `c(p)` drift to
  absorb; the old "prefer 0.05 at p>=5" caveat does not apply to the penalty
  and must not reappear.

## S5 — Nullspace quality gate *(not a table; a correctness precondition)*

The k=1 free harmonic form degrades with `p` (W7-X relL2 8.4e-13 / 3.0e-04 /
1.7e-01 at p=2/3/5) and every k=1-free and k=2-dbc solve deflates against it.
Before trusting any singular-row number:

- wire the documented cure — inverse iteration seeded from the direct vector,
  `find_nullspace_vectors(..., x0s=[v], inner_tol=1e-8)`, 2 sweeps, ~3e-24
  independent of `h` (`tensor_preconditioners.md` §8);
- re-measure `relL2` on all six geometries at p=2,3,5 and gate S1/S2/S4 on it.

**This blocks the singular rows of every other sweep.** Do it with S1.

---

## Order

1. **S1 + S3-coefficients + S5** — cheap, and S5 gates the rest.
2. **S2** — the big one; needs S5 clean.
3. **S4** — refinement, after `s` is settled by S2.
4. **S3-solves** — falls out of S2's stored grid.

## Do not re-run

- Anything comparing `product`/`halves`/`matrixwise`/`product_bare_h`. The
  conventions are deleted; the comparison is closed and its conclusion is a
  remark in the paper, not a table.
- The `L_approx`-vs-exact transfer test. The paper now states the solve path
  explicitly, so the surrogate is not used anywhere.
- The MINRES CGS/MGS variants. Refuted: identical counts,
  `max|v.r1| ~ 1e-18`.
