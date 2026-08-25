# Research / experimental — NOT production

Everything here is research record: campaign handoffs, plans, findings, and
refuted or shelved approaches. Nothing here describes what runs in production —
that is `docs/PRODUCTION.md`, the single authority.

Docs carrying a `> **SUPERSEDED/OUTDATED**` banner at the top are kept for the
record only; the banner names what replaced them.

## Start here — the current preconditioner (2026-08-22)

- **`preconditioner_technical_note_source.md`** — the consolidated account:
  construction, the derivation of the natural-BC coefficient, why the 0.10
  scale exists, what was refuted, and every measurement as a table. Written to
  be read without the cluster or the data.
- **`HANDOFF_open_items.md`** — what is unfinished, in priority order, plus the
  traps that cost real time.
- `production_simplification_plan.md` — how it got into production (phases,
  what was deleted, what was kept and why).
- `natural_bc_coefficient_handoff.md` — the full day-by-day record, §1-§19,
  including every dead end and the reason it died. Long; read §0 then §15-§19.

## Earlier campaigns (context, partly superseded)

- `handoff_2026-08-13_eod.md` / `handoff_2026-08-13_gpu_cluster.md` — the
  2026-08 preconditioner campaign (k=0 fdbund swap, MG verdicts, the k=1
  coupled+L0 solution, the relaxation-class ledger) + research shelf.
- `mass_preconditioner_pivot.md` — the tensor→raw_kron mass pivot; its `E+`
  analysis is why block_jacobi's dense core is worth it.
- `k_gt0_final_assessment.md` — closed the k>0 space in 2026-08-14. **Verdict
  overturned**, reasoning still useful.
- `laplacian_mg_k0_plan.md` — k=0 MG theory/log (shelved: research branch).
- `hiptmair_xu_preconditioner.md` — HX/aux-space reference.

## Relaxation initial conditions (2026-08-25)

- **`relaxation_ic_2026-08-25.md`** — how to build a relaxation IC without
  interpolating `B`. Our reference 2-form components ARE GVEC/VMEC's
  `sqrt(g) B^i`, so the field rebuilds from three scalars
  (`dPhi_dr`, `dchi_dr`, `LA`) instead of a resampled vector — verified against
  the hegna export's own B with no free factor. Also: helicity in closed form
  (metric-free, and zero for constant iota); why lambda changes the energy but
  never the fluxes or the helicity; the exact screw-pinch pressure for
  polynomial profiles; and the lambda equation (elliptic, decoupled per flux
  surface) as a data-free warm start. §10 records two silent traps —
  `load(frame='ref')` wants `g omega / J`, not `omega`, and histopolation is
  blocked for DBC/polar spaces.

## Unrelated topics

- `w7x_vacuum_bfield_handoff.md`, `gvec_h5_vacuum_comparison.md`,
  `harmonic_vacuum_library_plan.md` — vacuum-field and harmonic-form work.
- `poisson_convergence_submitit_bug.md` — a self-contained bug log.
