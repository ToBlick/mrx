> **Status:** current
> **Read this for:** the Poincare plotter, the relaxed-state tracer and the synthetic-pressure caveat
> **Do not read for:** tracer physics (handoff_2026-08-24_poincare.md)

# Handoff — Poincaré plotter and relaxed-state tracer, 2026-08-25

Supersedes the two separate 2026-08-25 Poincaré handoffs. Continues
`handoff_2026-08-24_poincare.md`, which stands unchanged — nothing here
revisits the tracer's physics.

`fd757eb` (plotter) is merged into `greville-prod`. **`5d30a3c` (relaxed-state
tracer, shared driver glue, drift-metric fix) is NOT** — it is only on
`poincare-plotter`, worktree `.claude/worktrees/poincare-plotter`, and neither
branch is pushed (no GitHub credentials in this environment, same as the two
prior handoffs).

## Status

| | |
|---|---|
| Arbitrary ζ slices — `--planes` / `--n-planes`, mutually exclusive | SETTLED |
| `B^ζ` gate **raises** at `BZETA_MIN_FRACTION = 0.05`, no clamp | SETTLED |
| `SECTION_CMAP = gist_rainbow` | SETTLED |
| Three-panel layout: section, iota profile, p | SETTLED |
| iota-above / p-below split about the magnetic axis | SETTLED as a flag — **not the default**, §3.1 |
| Relaxed-state tracer `poincare_relaxed.py` + its four gates | SETTLED (unmerged) |
| Step drift measured over regular lines only | SETTLED (unmerged) |
| **A real pressure through this tracer** | **OPEN — §3.2, the load-bearing caveat** |
| Does the stochastic core survive refinement? | OPEN — §2.1, never tested |
| `poincare_vacuum.py` re-run after the shared-helper refactor | OPEN |
| Axis-probe blob at the section centre | OPEN |
| Crossing density vs the reference figure | OPEN |
| Reproducing either verification run | OPEN — job scripts are gitignored, §5 |

## 1. What shipped in the plotter (`fd757eb`)

**Arbitrary ζ slices.** `--n-planes N` puts N evenly spaced sections over one
field period (`ζ = k/N`); `--planes` takes explicit positions. Mutually
exclusive — a run states either where the sections are or how many.

**The `B^ζ` gate now FAILS.** `require_zeta_parameterisation` raises
`BzetaParameterisationError` on a sign change in `B^ζ` over the interior, and
on `|B^ζ|/|B|` coming within `BZETA_MIN_FRACTION = 0.05` of zero, naming the
measured range and the worst logical `(r, θ, ζ)`. It used to warn and trace
anyway, which put the burden on whoever read the log — and the failure it
guards against renders as a *plausible chaotic sea*, so nobody would.

The threshold is not tuned: the quasr family, **including** the genuinely
chaotic k=1 cases, measured ≥ 0.774, so a field that trips this is
qualitatively different from anything seen, not marginally worse.

**No clamp**, and the reason is recorded in both directions. Clamped to `+ε` the
RHS becomes `~1/ε · B^r` and the line flies off. Clamped on the negative side of
a genuine zero crossing it flips the sign of the whole RHS and the line traces
**backwards** — a rendered plot with no NaN and no warning.

**`gist_rainbow`.** One colour per nested surface; a hue cycle separates
*adjacent* surfaces, which is what the eye follows here. Turbo's luminance ramp
suits a continuous field and is worse for a stack of discrete curves. `--cmap`
on `poincare_replot.py` overrides.

**Three panels and the split.** With a `pressure` array the third panel becomes
p (the iota profile is unchanged); `split_iota_p=True` colours the section by
iota above the magnetic axis and p below it, with both colorbars and a dotted
line at the axis. "Above" is defined against the **magnetic** axis, not `Z = 0`
— splitting on `Z = 0` cuts a Shafranov-shifted plasma off-centre and the two
halves are not the same surfaces. The dividing line is the *mean* of the tracked
axis crossings; the axis wanders ~1e-3 of the minor radius over a period and one
sample would tilt the split by that much. The split raises without both
`pressure` and `axis_RZ` rather than drawing half a panel. Chaotic lines keep
dark grey in **both** halves and stay out of the colour limits and the fit.

## 2. What shipped in the tracer (`5d30a3c`, unmerged)

`scripts/debug/poincare_relaxed.py` traces a relaxation state file directly —
both `B_dof_initial` and `B_dof_final`, on one shared iota scale (`iota_lim`,
new on `render_section`; `--iota-lim` on replot). Two separately fitted scales
make the same hue mean a different transform in each figure, so the pair is not
comparable by colour at all.

Four gates run before any line is traced, each because its failure *renders*
rather than raises: the map round-trips against the cloud the config names, the
Jacobian keeps one sign, `len(B_dof) == n2_dbc`, and `|D2 B| / (|D2| |B|)` sits
at the Leray solve's tolerance rather than O(1). The last is the only one that
tests *which space* the DOFs live in — a different radial grading or pole
extraction can match the dimension by coincidence.

The gates are what let a state file be traced without knowing what wrote it,
and that is not hypothetical: `data/w7x_fmm002_relaxed_100.h5` has no committed
producer, and its config carries `map.flip_r` / `auto_flip_r` /
`auto_flip_jacobian` keys that nothing in the tree consumes. Not worth chasing
— but it is the reason the gates check the rebuilt space *against the DOFs*
rather than trusting the config that travels with them.

`section_RZ`, `surface_label` and `trace_and_classify` moved into
`mrx/poincare.py`; `poincare_vacuum.py` calls them and, as a consequence, now
greys chaotic lines in its own figures. It did not before, so it and
`poincare_replot.py` disagreed about the same trace.

### 2.1 What the sections measured — `w7x_fmm002_relaxed_100`

The one result here that is **not** recoverable from the code or the commits.
96 seeds, 1000 periods, four planes per field period, both states.
`outputs/poincare_relaxed/2026-08-25/12-29-15/` **in the main checkout** —
`outputs/` is gitignored, so the worktree copy dies with the worktree.

* **The relaxation heals the edge.** The input field carries a 5/4 island chain
  at the boundary (four lobes, plainest at ζ = 0.5) and a broad stochastic band
  inside it. In the relaxed state both are replaced by nested surfaces out to
  the wall, and iota narrows from 1.1518–1.2500 to 1.1739–1.2340.
* **The core is stochastic in both**, out to logical r ≈ 0.45 — about half the
  volume, 51 of 96 lines. Not integration error (§4) and not a bad
  parameterisation: `B^ζ/|B| ≥ 0.140`, worst at logical r = 0.021, i.e. *at the
  polar axis*, which is a coordinate effect and not a property of the field. At
  `fem ns = 6,8,8 p=3` with iota ≈ 1.19–1.21 and near-zero shear across the
  core, a destroyed core is what one would expect — but **whether it survives
  refinement is not tested**, and that is the obvious next run.
* **The magnetic axis is far off the coordinate axis**: 6.6e-02 m at ζ = 0 and
  2.4e-02 m at ζ = 0.5, against a minor radius of ~0.28 m at ζ = 0. That is why
  `--seed-from axis` is the default in this driver and `coord` in
  `poincare_vacuum.py`; seeding from `r = 0` here leaves a hole in the middle.
* The map round-trip residual (max |ΔR| 1.22e-02 m over 120000 points) is the
  **map fit's own** residual at `ns_map = 8,12,12 p=3`. It is not error
  introduced by rebuilding, and the gate should not be tightened against it.

## 3. The two judgements

### 3.1 The up/down split should not be the default

It reads acceptably — the axis line separates the halves cleanly and `plasma`
against `gist_rainbow` do not collide. But it buys the comparison by **throwing
away half the iota information and half the p information**, and a W7-X section
is not up/down symmetric, so the two halves are not the same set of surfaces
sampled twice. Worth having as a flag. Not the default figure.

### 3.2 Nothing here has met a real pressure

**Scope this precisely: it is a statement about what this tracer's inputs were
on 2026-08-25, not about the repo today.**

At the time, no path in this repo produced a pressure for this tracer. p lives
in the relaxation route (`mrx/plotting.py`, "color by pressure if available")
and this tracer traces harmonic nullspace fields. The wiring was proven with a
**synthetic** p, labelled `SYNTHETIC p` on the check figure for that reason; the
physics path was unexercised.

Two consequences that outlive the caveat:

* **The vertical stripes in the check figure are the synthetic p, not a bug.**
  It is driven by a peaked function of distance from the axis, which is *not* a
  flux function, so p varies along each line and every line draws a stripe. A
  real equilibrium p is nearly constant on a flux surface and collapses each
  stripe to a point — which makes **stripe width a free diagnostic of how far
  from a flux function the pressure is.** Do not read the check figure as a
  pressure profile.
* Whoever first traces a relaxed equilibrium should expect to plumb p from the
  relaxation state into `render_section(pressure=...)`.

Since then the relaxation workstream has run the tracer on Clebsch ICs from
finite-beta W7-X equilibria — see `handoff_2026-08-25_relaxation_prelim.md` for
their numbers; not restated here. That closes the *input* half. It does not
close this item: `relax_from_nfs.py` stores no `p_dof`, so §2's own runs are the
three-panel iota form and `render_section(pressure=...)` and `--split-iota-p`
are still unexercised on real physics.

## 4. One trap worth carrying

The first relaxed run reported `h/2 drift 5.7e-01` — character for character the
signature §1 documents as a broken ζ parameterisation, "drift that does not fall
under refinement". It refined to 0.51 at 48 steps/period and 0.71 at 96. It was
not a broken trace: iota agreed to four decimals (1.1739…1.2235) across the same
4× refinement.

`step_convergence` was sampling seeds by stride and this field's stride hit the
**chaotic** ones, where two nearby trajectories separate exponentially and the
h vs h/2 displacement saturates at the size of the stochastic region no matter
how small h is. It measures the Lyapunov exponent, not the integration error.
`trace_and_classify` now draws the subsample from the regular lines only and
reports `drift_lines`; the number becomes 1.6e-04. It is NaN when no line is
regular — on such a trace the step cannot be checked this way at all.

**Never read a large drift as a broken parameterisation without first checking
that iota is stable under refinement.**

## 5. Open, with the cheapest next step

* **Real pressure (§3.2).** Cheapest: have the relaxation route store `p_dof`
  beside `B_dof_*`, then pass it to `render_section(pressure=...)`.
* **`poincare_vacuum.py` unverified since the refactor.** `ruff`-clean and the
  moved code is exercised by the new driver, but no nullspace run since.
  Cheapest: `--geometry toroid --periods 40`.
* **Merge `5d30a3c`.** Cheapest: cherry-pick onto `greville-prod`; the plotter
  code is untouched by the histopolation / Stage C+D / docs work.
* **The axis-probe seed** (entry 0, the small orbit `seed_from_axis` keeps around
  the axis) classifies as chaotic and renders as a dark blob at the section
  centre. Cosmetic, and the expected consequence of its angle being the
  difference of two nearly identical floats — but a publication figure wants it
  excluded or drawn as a marker.
* **Density is not matched to the reference figure** (`data/poincare_plot_pretty_w7x.pdf`,
  gitignored, read from the main checkout): it carries far more crossings per
  line than the archived trace. That is a `--periods` choice at trace time, not
  a rendering one.
* **Neither verification is reproducible from the repo.** `.gitignore` carries
  `slurm/job_*`. `slurm/job_poincare_render_check.sh` — which produced the
  plotter verification (GPU job 16767729) — is **not committed** and must be
  re-created or the script run directly under slurm.
  `slurm/job_poincare_relaxed.sh` **is** committed, via `git add -f`.
* **Worktree jobs need `PYTHONPATH=$WT`.** `python scripts/debug/x.py` puts the
  *script's* directory on `sys.path`, never the cwd, so `import mrx` resolves to
  the venv's editable install — the main checkout. `job_poincare_relaxed.sh`
  sets it; `job_poincare_render_check.sh` does not, so job 16767729 verified
  `mrx.poincare` from the main checkout rather than the branch. Re-run it.
