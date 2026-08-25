# Handoff — Poincaré plotter refinement, 2026-08-25

Branch `poincare-plotter`, one commit `fd757eb` on top of `greville-prod`
@ `44fbda3`. Worktree `.claude/worktrees/poincare-plotter`.

**NOT PUSHED.** `git push` fails with `could not read Username for
'https://github.com'` — no credentials in this environment, the same limitation
`handoff_2026-08-24_poincare.md` recorded. The branch is local; someone with
credentials needs to push it.

Continues `handoff_2026-08-24_poincare.md`, which stands unchanged — nothing
here revisits the tracer's physics, only its interface and its figure.

---

## 1. What changed

### 1.1 Arbitrary zeta slices

`--n-planes N` puts N evenly spaced sections over one field period,
`zeta = k/N` for `k = 0..N-1`. `--planes` still takes explicit positions. They
are **mutually exclusive** and the driver errors if given both — a run states
either where the sections are or how many, never both.

### 1.2 The B^zeta gate now FAILS

`zeta_component_report` printed a warning and traced anyway. That put the
burden on whoever read the log to spot one line among hundreds, and the failure
it guards against renders as a plausible chaotic sea — so nobody would.

`mrx.poincare.require_zeta_parameterisation()` now raises
`BzetaParameterisationError` on

* a **sign change** in `B^zeta` over the interior, and
* `|B^zeta|/|B|` coming within `BZETA_MIN_FRACTION = 0.05` of zero,

naming the measured range and the worst logical `(r, theta, zeta)`.

The threshold is not tuned. The quasr family — *including* the genuinely
chaotic k=1 cases — measured `>= 0.774` (`handoff_2026-08-24_poincare.md` §4.2),
so a field that trips this is qualitatively different from anything seen, not
marginally worse. The near-zero arm matters as much as the sign-change arm: a
stiff `B^r/B^zeta` under a *prescribed* step schedule surfaces as drift that
does not fall under refinement, which is precisely the signature that was read
as chaos once already.

**No clamp**, per `no-defensive-code`, and the docstring records why in both
directions: clamped to `+eps` the RHS becomes `~1/eps * B^r` and the line flies
off; clamped on the negative side of a genuine zero crossing it flips the sign
of the whole RHS and the line traces **backwards** — a rendered plot with no NaN
and no warning.

### 1.3 Colour by iota and by p, and the up/down split

`render_section` takes an optional per-crossing `pressure`:

* with it, the third panel becomes the **p profile** (the three panels: section,
  the iota profile *unchanged*, p);
* `split_iota_p=True` colours the section by **iota above the magnetic axis and
  p below it**, with both colorbars and a dotted line at the axis.

Pressure is **optional** because these fields are harmonic and carry none — a
vacuum run leaves it `None` and gets exactly the previous figure. This is what
Tobias asked for when he said the split is a flag you turn off for vacuum
fields.

The split **raises** without a pressure array and without `axis_RZ`, rather than
drawing half a panel. "Above" and "below" are defined against the *magnetic*
axis: splitting on `Z = 0` would cut a Shafranov-shifted plasma off-centre and
the two halves would not be the same set of surfaces. The dividing line is the
**mean** of the tracked axis crossings — the axis wanders by ~1e-3 of the minor
radius over a period and one sample would tilt the split by that much.

Chaotic lines keep dark grey (0.25) in **both** halves and stay out of the
colour limits and the profile fit, so the split cannot assign them an iota.

### 1.4 Pretty

`SECTION_CMAP = "gist_rainbow"`, matching the reference figure
(`data/poincare_plot_pretty_w7x.pdf` — gitignored, read from the main checkout).
With one colour per nested surface a hue cycle separates *adjacent* surfaces,
which is what the eye follows here; turbo's luminance ramp suits a continuous
field and is worse for a stack of discrete curves. `--cmap` on
`poincare_replot.py` overrides.

What the reference actually is, for whoever matches it next: a landscape
742x270 pt figure, physical `R-z` section (W7-X, R 4.8–6.0 m) beside a logical
`r-theta` panel, a thin `gist_rainbow` colorbar ticked with resonant rationals
(10/11, 15/16, 15/17, 20/21, 20/23), tiny rasterised markers on white. Its
third element is the colorbar, not a profile panel.

---

## 2. Verification

GPU job `16767729`, `scripts/debug/poincare_render_check.py`, rendering from an
archived trace (`outputs/render_check/2026-08-25/06-19-03/`). Presentation only
— it never solves — but it went through slurm like everything else.

```
=== B^zeta gate ===
  [PASS] healthy field passes: |B^z|/|B| min 0.994
  [PASS] sign change raises: B^zeta CHANGES SIGN over the interior
  [PASS] near-zero raises: comes within 7.071e-03 of zero (tol 0.05)
=== renders ===
  wrote w7x_zeta0_iota.png
  wrote w7x_zeta0_split.png
  [PASS] split refuses without a pressure array
  [PASS] split refuses without axis_RZ
ALL CHECKS PASSED
```

The w7x k=2 section re-renders with the 10/11 island lobes at top and bottom and
iota 0.851–0.948 — the published standard-configuration vacuum range — so the
colormap change did not disturb the content. One chaotic line, greyed.

`ruff` clean on all four touched files.

---

## 3. Judgement calls, for Tobias

**The split reads acceptably, with a real cost.** The axis line separates the
halves cleanly and `plasma` against `gist_rainbow` do not collide (purple→yellow
against a full spectrum). But it buys the comparison by throwing away half the
iota information and half the p information, and a W7-X section is not up/down
symmetric, so the two halves are not the same surfaces sampled twice. It is
worth having as a flag; I would not make it the default figure.

**The vertical stripes in the p panel are the synthetic pressure, not a bug.**
The check drives it with a peaked function of distance from the axis, which is
*not* a flux function, so p varies along each line and every line draws a
vertical stripe. A real equilibrium `p` is nearly constant on a flux surface and
would collapse each stripe to a point — which makes stripe width a free
diagnostic of how far from a flux function the pressure is. Do not read the
check figure as a pressure profile; it is labelled `SYNTHETIC p` on the figure
for that reason.

**Nothing here has been exercised on a real pressure**, because no path in this
repo produces one for the Poincaré tracer — pressure lives in the relaxation
route (`mrx/plotting.py:928`, "color by pressure if available"), and this tracer
traces harmonic nullspace fields. The wiring is proven; the physics path is not.
Whoever first traces a relaxed equilibrium should expect to plumb `p` from the
relaxation state into `render_section(pressure=...)` and will be the first to
see a real p panel.

---

## 4. Not done

* `slurm/job_poincare_render_check.sh` exists in the worktree but is **not
  committed** — `.gitignore` carries `slurm/job_*` and only four legacy scripts
  are tracked. Re-create it or run the check script directly under slurm.
* The axis probe seed (entry 0, the small orbit `seed_from_axis` keeps around
  the axis) classifies as chaotic and renders as a dark blob at the centre of
  the section. Cosmetic, and it is the expected consequence of its angle being
  the difference of two nearly identical floats — but on a publication figure it
  wants either excluding from the render or drawing as a marker.
* The reference's density is not matched: it carries far more crossings per line
  than the 1001 in the archived trace, which is a `--periods` choice at trace
  time, not a rendering one.
