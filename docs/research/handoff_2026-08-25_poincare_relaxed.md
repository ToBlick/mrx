# Handoff — Poincaré sections of a relaxed field, 2026-08-25

Branch `poincare-plotter`, on top of `f9e6d4a`. Worktree
`.claude/worktrees/poincare-plotter`. **NOT PUSHED** — same missing GitHub
credentials as the two previous Poincaré handoffs.

Continues `handoff_2026-08-25_poincare_plotter.md`, which closed with

> Nothing here has been exercised on a real pressure, because no path in this
> repo produces one for the Poincaré tracer [...] this tracer traces harmonic
> nullspace fields.

That is now half-answered: there **is** a path from a relaxation state to a
section, and it is `scripts/debug/poincare_relaxed.py`. The pressure half is
still open — `relax_from_nfs.py` stores no `p_dof`, so these figures are the
three-panel iota form and `render_section(pressure=...)` is still unexercised
on real physics.

---

## 1. What this produces

`scripts/debug/poincare_relaxed.py <state.h5>` reads a relaxation state file —
`B_dof_final`, `B_dof_initial`, `R_dof`, `Z_dof`, and the run's own config in a
root attribute — and traces both states. Nothing solves; the file is
self-contained and the map is rebuilt from the stored 0-form DOFs rather than
refitted, so it is the *same* map the relaxation ran on and not a new fit.

**Both states, on one colour scale, is the point.** `initial` is the input
field after interpolation and the Leray projection; `final` is what relaxation
made of it. Fitted separately, two iota scales make the same hue mean different
transforms in each figure and the comparison is worthless. `render_section` now
takes `iota_lim` and the driver pins it across both states; `poincare_replot.py`
grew `--iota-lim` for the same reason.

## 2. Four gates, all of which fail as a plausible picture

Each one was added because its failure mode renders rather than raises.

1. **Map round-trip** against the point cloud the config names (`nfs_file`).
   A map rebuilt with the wrong `nfp`, degree or ζ flip still plots — as a
   differently-shaped device.
2. **Jacobian keeps one sign.**
3. **`len(B_dof) == seq.n2_dbc`.** A resolution mismatch is a reshaped vector
   with the right norm and no meaning.
4. **`|D2 B| / (|D2| |B|)`** against `DIV_TOL = 1e-6`. This is the only gate
   that tests *which space* the DOFs live in rather than how many there are: a
   different radial grading or pole extraction can match the dimension by
   coincidence and scores O(1) here. The threshold separates two regimes six
   orders apart — the Leray projection's own solve tolerance below, O(1) above
   — and must not be tightened into an audit of the projection.

Measured on `data/w7x_fmm002_relaxed_100.h5`: round-trip max |ΔR| 1.22e-02 m /
|ΔZ| 1.91e-02 m over 120000 points (that is the map fit's own residual at
`ns_map = 8,12,12 p=3`, not an error here), Jacobian 2.7e-01 … 1.7e+01,
`n2_dbc = 720`, weak divergence 1.3e-11 and 1.4e-11.

## 3. The step-drift metric was measuring the wrong thing

The first full run reported `h/2 drift 5.7e-01` — the exact signature
`handoff_2026-08-25_poincare_plotter.md` §1.2 names as a broken ζ
parameterisation ("drift that does not fall under refinement"). It refined to
0.51 at 48 steps/period and 0.71 at 96. It was not a broken trace:

| steps/period | iota range (final) | chaotic / 64 |
|---|---|---|
| 24 | 1.1739 … 1.2235 | 34 |
| 48 | 1.1738 … 1.2236 | 36 |
| 96 | 1.1739 … 1.2235 | 38 |

Four decimals of agreement across a 4× refinement. `step_convergence` was
sampling seeds by stride, and this field's stride hit the **chaotic** ones,
where two nearby trajectories separate exponentially and the h vs h/2
displacement saturates at the size of the stochastic region no matter how small
h is. It measures the Lyapunov exponent, not the integration error.

`trace_and_classify` now draws the drift subsample from the regular lines only,
and returns `drift_lines` so the figure can say how many it used. On the same
run the number becomes **1.6e-04** (final) and **4.5e-04** (initial). `drift`
is NaN when no line is regular, which is the honest answer — on such a trace
the step cannot be checked this way at all.

This is worth carrying: **on a field with a stochastic region the drift number
is only meaningful over the regular lines**, and the two failure modes it
cannot distinguish otherwise look identical.

## 4. What the sections show

`outputs/poincare_relaxed/2026-08-25/12-29-15/`, 96 seeds, 1000 periods, four
planes per field period, png and pdf.

* **The relaxation heals the edge.** `initial` carries a 5/4 island chain at the
  edge (four lobes, plainly visible at ζ = 0.5) and a broad stochastic band
  inside it. In `final` both are replaced by nested surfaces out to the
  boundary, and iota narrows from 1.1518–1.2500 to 1.1739–1.2340.
* **The core is stochastic in both**, out to logical r ≈ 0.45 — about half the
  volume, 51 of 96 lines. It is not integration error (§3) and not a broken
  parameterisation (`B^ζ/|B| ≥ 0.140`, worst point at logical r = 0.021, i.e.
  at the polar axis, which is a coordinate effect and not a property of the
  field). At `fem ns = 6,8,8 p=3` with iota ≈ 1.19–1.21 and very low shear
  across the core, a destroyed core is what one would expect; whether it
  survives resolution is **not tested here** and is the obvious next run.
* Magnetic axis to coordinate axis is 6.6e-02 m at ζ = 0 and 2.4e-02 m at
  ζ = 0.5, against a minor radius of ~0.28 m at ζ = 0. `--seed-from axis` is
  the default in this driver for that reason; `poincare_vacuum.py` still
  defaults to `coord`.

## 5. Shared code, and one behaviour change to `poincare_vacuum.py`

`section_RZ`, `surface_label` (+ `SURFACE_LABELS`) and `trace_and_classify`
moved into `mrx/poincare.py`; `poincare_vacuum.py` now calls them instead of
carrying its own copies.

**`poincare_vacuum.py` figures now grey out chaotic lines.** They did not
before — `poincare_replot.py` classified, the direct render did not, so the two
paths disagreed about the same trace. `trace_and_classify` classifies once, and
archives `chaotic`, `nfp`, `saves_per_period` and `label` so `poincare_replot.py`
prefers the run's own classification (it recomputed about a centre that excluded
the axis probe, which is not the centre the iota used) and no longer needs its
`NFP`/`LABEL` tables for a new-style archive.

**Not re-verified end to end.** `poincare_vacuum.py` is `ruff`-clean and the
moved code is exercised by the new driver, but no nullspace run has been made
since the edit. A `--geometry toroid --periods 40` run is the cheap check.

## 6. `PYTHONPATH` — this bites every worktree job

`python scripts/debug/x.py` puts the **script's** directory on `sys.path`,
never the working directory, so `import mrx` resolves to the venv's editable
install, which points at the main checkout. A worktree job without
`PYTHONPATH=$WT` silently runs the worktree's script against the main
checkout's library and reports the result as a test of the branch.
`slurm/job_poincare_relaxed.sh` sets it. `slurm/job_poincare_render_check.sh`
**does not** — so the 2026-08-25 render-check verification (job 16767729) ran
`mrx.poincare` from the main checkout, and what it verified is not what is on
this branch. Worth re-running.

## 7. Not done

* No pressure. `relax_from_nfs.py` stores no `p_dof`, so `--split-iota-p` and
  the p panel remain unexercised on real physics.
* No resolution scan of the stochastic core (§4).
* `slurm/job_poincare_relaxed.sh` is committed here; `.gitignore` carries
  `slurm/job_*`, so it needed `git add -f` like the four legacy scripts.
* Seeds are spread uniformly from axis to edge, so ~half of them land in the
  stochastic core and add nothing. Fine as a default; `--r-min` biases outward
  when the core is known to be bad.
