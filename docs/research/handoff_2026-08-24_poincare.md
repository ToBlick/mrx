# Handoff — Poincaré tracer and the overnight sweeps, 2026-08-24

Branch `worktree-poincare`, based on `greville-prod` at `7e89525`. **Local only,
never pushed** — push needs credentials I do not have.

Code: `mrx/poincare.py`, `scripts/debug/poincare_{vacuum,replot,converge,
pullback_check}.py`, `scripts/debug/k0_block_default.py`,
`slurm/job_poincare_{vacuum,sweep}.sh`, `slurm/job_k0_block.sh`.
Prod change: `mrx/operators.py`, `mrx/preconditioners.py`, `mrx/nullspace.py`.

---

## 1. The tracer

Four things differ from `mrx.plotting.integrate_fieldlines`.

**Toroidal angle as the independent variable.** Dividing the field-line ODE by
`B^zeta` gives `dr/dzeta = B^r/B^zeta`, `dtheta/dzeta = B^theta/B^zeta`, so
crossings land at `zeta = zeta_0 + m` *exactly*. No crossing detection, no
interpolation, no accumulated root-finding error over thousands of turns.

**Prescribed step schedule** (`diffrax.StepTo`), so lanes do not couple.
Adaptive controllers run a vmapped batch on the smallest step any lane asks for;
chunking bounds how many healthy seeds one bad seed holds up, it does not
isolate it. Benchmark, 49 seeds x 20 periods, compile included:

| field | prescribed, vmap | adaptive, vmap | adaptive, chunk 8 |
| --- | --- | --- | --- |
| w7x k=2 | 9.9 s | 12.8 s | 22.6 s |
| w7x k=1 | 26.0 s | 43.0 s | 66.9 s |
| **quasr44970 k=1** | **22.4 s** | **215.0 s** | — |

The last row is the failure mode: one seed's step collapsed and dragged the
whole vmap to 9.6x. Note chunking is *worse* than not chunking — eight-seed
chunks run sequentially, each paying its own worst-seed step. The old
`min(8, nseeds)` default was the worst of the three. The price of a prescribed
schedule is no error estimate, so `step_convergence` earns the step count by
h-vs-h/2 refinement and the number is printed on every figure.

**Cartesian cross-section chart.** `B^theta ~ 1/r` near the polar axis, so
`dtheta/dzeta` diverges there and the innermost seeds — the ones resolving the
core — are the ones an integrator handles worst. In
`(u,v) = (r cos 2pi theta, r sin 2pi theta)` the `1/r` cancels against the
`O(r)` length of the same coordinate vector.

**Iota about the tracked magnetic axis**, per phase within the period, by a
least-squares slope on the unwrapped angle rather than an endpoint difference
(on an island the endpoints are two arbitrary points on a bounded oscillation).
The fit residual comes back with it and separates surface from island from
chaos. This replaces the KS-uniformity screen in `mrx/plotting.py`, which was
discarding lines whose angle had been measured about `r = 0` instead of about
the axis — a bad centre, not a bad line.

### 1.1 Surface label — settled

The iota profile is drawn against the **outboard midplane distance from the
magnetic axis** (`--profile-x midplane`, the default). Nested curves cross a
fixed ray at strictly increasing distance, so it is monotone *by nesting*.
Reversals along the seed ordering, measured on the archived traces:

| case | `sqrt(A/pi)` | mean distance | **midplane** |
| --- | --- | --- | --- |
| w7x k2 | 1 (−7.9e-03) | 1 (−5.1e-04) | **0** |
| w7x-ini k2 | 2 (−1.5e-02) | 1 (−2.7e-03) | **0** |
| hegna k2 | 0 | 0 | **0** |
| quasr9983 k2 | 0 | 0 | **0** |
| quasr65530 k2 | 0 | 0 | **0** |
| quasr44970 k1 | 1 (−1.2e-03) | 1 (−1.7e-03) | **0** |
| quasr65530 k1 (broken) | 0 | 1 | **3 NaN** |

The mean is non-monotone because it averages over the *crossing points*, whose
angular distribution is set by the field-line dynamics rather than by the
surface; fixing the ray removes that weighting. On the broken k=1 trace the
midplane label returns NaN for the surfaces that are not surfaces, so it doubles
as a diagnostic rather than laundering garbage into a plausible number.

A surface meets the midplane twice. Both argmins minimise `|alpha|`, so they
bracket `alpha = 0` and the inboard crossing at `alpha = +-pi` can never win —
`arctan2`'s branch cut falling inboard is what disambiguates. **Outboard is a
convention, not a robustness argument**: I predicted the inboard side would be
the risky one because a bean section carries its indentation there, and measured
it is not — the relative residual of a linear `r(alpha)` fit either side is
~3e-4 on both, inboard slightly better. Concave curvature is not a ray crossing
twice.

### 1.2 Seeding

`seed_line` walks out at constant *logical* radius, i.e. from the coordinate
axis, which is only the magnetic axis when the two coincide. On **w7x-ini**
(beta 4.2%) they are 4.86e-02 m apart — `r = 0` is the *equilibrium's*
Shafranov-shifted axis, the vacuum field's is 4.9 cm inboard — so the innermost
seed traced a surface at `a_eff = 0.1017` and the section had a hole in the
middle. On vacuum w7x the offset is 6.1e-04 m and there is no hole. That 4.9 cm
is a measurement of the Shafranov shift, not a defect.

`seed_from_axis` (`--seed-from axis`) finds the axis with one short probe trace
and lays seeds along the ray from there to the edge. Entry 0 stays the `r_axis`
probe: it is `axis_track`'s centre and must keep a small *orbit* around the
axis, not sit on it, or its own angle is the difference of two identical floats.

`axis_offset` is printed for every run and is the screen for this.

---

## 2. Results

`outputs/poincare_night/2026-08-24/20-39-41/` — 25 cells, coord seeding.
`outputs/poincare_final/2026-08-24/21-26-16/` — **the run to use**: axis
seeding, logical panel, rational ticks, midplane profile axis, chaos greying.
24 of 25 cells completed.

### 2.0 Three geometry files fold — the map is degenerate, loudly

| cell | failure |
| --- | --- |
| `quasr65575` ns=(12,24,12) p=3 | `no handedness gives det DF > 0`; det DF spans **[−0.236, +1.543]** |
| `quasr65530` ns=(16,32,16) p=3 | `geometry is degenerate` (`jac.min() <= 0`) |
| `quasr65530` ns=(12,24,12) p=4 | same |

det DF *changing sign* is a fold, and no handedness fixes a fold — the sign
search only chooses an orientation. quasr65530 builds fine at ns=(8,16,8) and
(12,24,12) p=3 and folds above that, so the R/Z data carries a near-fold that a
coarse spline smooths over and a finer one resolves into a real one. Both jobs
raised at setup rather than solving on a negative Jacobian, which is the
behaviour to keep. quasr65575 is unusable at this resolution as it stands;
quasr65530 is usable only at or below ns=(12,24,12) p=3, and note its k=1 field
is also the chaotic one.

Iota, k=2 essential BC, ns=(12,24,12) p=3:

| geometry | nfp | iota | k2 drift | lost |
| --- | --- | --- | --- | --- |
| toroid | 1 | 0.0000 – 0.0000 | 3.6e-15 | 0/48 |
| rot-ellipse | 3 | 0.0000 – 0.0000 | 4.7e-05 | 0/48 |
| quasr65530 | 4 | 0.0883 – 0.0955 | 4.5e-05 | 0/48 |
| quasr9983 | 2 | 0.0971 – 0.0980 | 2.2e-10 | 0/48 |
| quasr44970 | 3 | 0.4823 – 0.5051 | 1.8e-07 | 0/48 |
| hegna | 3 | 0.4457 – 0.7057 | 9.9e-06 | 0/48 |
| w7x | 5 | 0.8523 – 0.9481 | 5.3e-07 | 0/48 |
| w7x-ini | 5 | 0.8675 – 1.0000 | 1.6e-06 | 0/48 |

Two known answers come out right: the axisymmetric **toroid** gives iota 0 to
1e-17 with zero radial drift, and **w7x** gives 0.851 rising to 0.948, the
published standard-configuration vacuum range, from a harmonic form the code
computes for itself.

**rot-ellipse is exactly 0 (5e-12) and this is forced**, not incidental.
`rotating_ellipse_map` builds an axis-aligned ellipse, `R-R0 = eps nu(zeta) r
cos`, `Z = eps nu(zeta+0.5/nfp) r sin`, with no tilt term: the section runs tall
at `zeta=0`, through an exact circle at 0.25, to wide at 0.5. It *pulsates*. The
domain is then invariant under `Z -> -Z` at every zeta, the harmonic field maps
to itself, and a reflection-invariant field has zero net poloidal winding. l=2
transform needs the ellipse *axis* to rotate. The name misleads; the map is fine
as the metric-variation preconditioner test case it exists to be.

### 2.1 Resonant rationals

The colorbar carries only the rationals an island can form on: `iota = n/m` with
`n = 0 mod nfp`, since an nfp-periodic field carries only those toroidal
harmonics. Reproduced from geometry alone what the trace had already found:
**w7x has exactly one resonance in range, 10/11 = 0.9091**, and that is where
the measured profile flattens, where the angle-fit residual spikes on two seeds,
and where the island lobes sit in the section. w7x-ini reaches 5/5 = 1 at its
edge. quasr9983 and quasr65530 have none, matching their clean nested surfaces.
hegna has seven, matching its being the worst behaved.

### 2.2 Perturbation invariance — passes

`pert-axis` and `pert-interior` at ns=(12,24,12) both give iota
**0.4461 – 0.4688**, identical to four decimals. Both move only the map interior
with the boundary fixed (`perturb_boundary_max_abs_dR_m = 0`), so the domain and
therefore the harmonic field are unchanged, and a displaced magnetic axis does
not perturb it.

**Their `nfp = 2` attribute is wrong; they are quasr0044970, nfp = 3.** Their
R/Z is `quasr0044970_gvec_nr8_nt16_nz8` shifted by exactly the amplitudes in
their filenames (max|dR| 5.000e-05, max|dZ| 3.750e-05) and that device is nfp=3;
against quasr0009983, which their `dof_npy` and `perturb_source_h5` name, they
differ by 0.15. Their `geometry_source` and `template_h5` agree with the
measurement. Confirmed three ways: the offsets, a det DF range matching
quasr44970's, and iota matching. `GVEC_NFP_OVERRIDE` corrects it — nfp=2 would
wrap one field period through 180 degrees instead of 120, a different domain and
a different iota behind a healthy positive Jacobian. **Worth fixing at source.**

Also: `w7x_ini_mrx.h5`'s `axis_radial_index = 49` is wrong (the axis is at
rho[0]; mean theta-extent 3.4e-3 there against 1.8 at rho=1), and four of the
new files carry only `precomputed_*` sizes, not `n_rho`/`n_theta`/`n_zeta`.

### 2.3 Do not build ns finer than the data grid

The perturbed pair and `quasr44970-c` share an 8x16x8 grid. At *matched*
ns=(8,16,8) they give 0.4838–0.5060 and the independent 50^3 `quasr44970` file
gives 0.4823–0.5051 at ns=(12,24,12) — agreement to 0.4%. At ns=(12,24,12) on
the 8^3 data they give 0.4461–0.4688, an 8% outlier: 12 radial DOFs over 8
radial data points over-resolves a piecewise-linear RGI interpolant and
reproduces its kinks.

---

## 3. Solver verification

**Every nullspace solve is exact.** `harmonic_rayleigh` (`v^T L v / v^T M v`,
quoted against a random vector because the quotient is not dimensionless) reads
**7e-27 to 5e-26** across every geometry and resolution, with `|dv|/|v|` at
1e-11. This is the gate `handoff_2026-08-24_harmonic_k1_free.md` specified and
never implemented; it now lives in `mrx/nullspace.py` and prints on every run.
My earlier "not confident in k=1" was wrong in its diagnosis.

**Both pullbacks are correct**, max relative error **1.8e-15** over all four
(k, BC) pairs on three geometries (`poincare_pullback_check.py`): k=2
coefficients *are* the contravariant components (`DF B/J` is Piola), k=1 needs
`g^-1 A` (since `DF (DF^T DF)^-1 A = DF^-T A`). Tested against
`mrx.differential_forms.Pushforward` with a *random* dof, because a harmonic
form can be dominated by one component and hide an error in the others.

**Preconditioners resolve to the intended stack and now prove it**: `mass =
block_jacobi` (metric-lumped — its `Lam_c = sqrt(diag(M_k)/diag(A_r x A_t x
A_z))` is the support-averaged metric weight), `schur.outer = block` (the atom
at `PRODUCTION_BC_SCALE = 3.0`, i.e. A5), `lumped="diag"` on both. A resolved
outer that is not `block` now *raises*, and `RuntimeWarning` is promoted to an
error before `compute_nullspaces`, so neither of the 2026-08-24 silent
downgrades can recur. `schur.inner` is deliberately not printed: it is still
raw_kron in the spec but with `outer='block'` the atom *is* the upper-block
inverse and the field does no work.

---

## 4. Open

### 4.1 The k=2 / k=1 angle DOES converge — resolved, it was the statistic

Both forms are exactly harmonic in their own complex (V_2 with essential BCs,
V_1 without), so the angle between them is discretisation error. Read on the
MEDIAN over 512 sample points it converges cleanly, roughly 2nd-4th order:

| geometry | ns 8 | ns 12 | ns 16 | ns 12, p=4 |
| --- | --- | --- | --- | --- |
| w7x | 0.0471 | 0.0153 | 0.0113 | 0.0069 |
| quasr9983 | 0.0195 | 0.0067 | 0.0037 | 0.0050 |
| quasr44970 | 0.0204 | 0.0031 | 0.0014 | 0.0012 |
| hegna | 0.0261 | 0.0058 | 0.0017 | 0.0028 |

The MAX over the same points does not, and that is what I alarmed about: w7x
0.3797 / 0.1829 / 0.1627, quasr9983 0.1802 / 0.0697 / 0.1916, quasr44970
0.2424 / 0.0315 / 0.1128 -- two of three worse at 16^3. One sample out of 512,
near `r -> 1` where the spline map is nearly singular, sets it. Quote the
median; the max is a map-singularity probe, not a convergence measure.

### 4.2 k=1 traces break on the quasr44970/65530 family — it is chaos

quasr44970's k=1 step drift is **3.5e-02 → 2.8e-02 → 3.6e-02** across ns
8/12/16. *Flat.* Refinement does not touch it, so it is not a resolution
deficit. quasr65530 k=1 loses 21 of 48 lines with drift 1.5e-01 and a section
that reads as a chaotic sea. w7x and quasr9983 sit at 1e-06–1e-08 for k=1 at
ns >= 12, and the k=2 arm is clean everywhere (1e-10 to 5e-4, zero lost).

**My `B^zeta` hypothesis is REFUTED.** I proposed that the tracer's division by
`B^zeta` was the culprit. Measured on quasr65530 over 4096 interior points:

    ns=(12,24,12)  k2: B^zeta/|B| in [+0.828, +1.000]
                   k1: B^zeta/|B| in [+0.774, +1.000]
    ns=(8,16,8)    k2: B^zeta/|B| in [+0.976, +1.000]

It never approaches zero and never changes sign, so the reparameterisation is
well conditioned for both fields and there is nothing wrong with the ODE.

**The corrected reading is chaos.** A step drift that does not fall when the
step is refined, lines escaping to r >= 1, and a meaningless least-squares angle
slope are all what a stochastic region produces: two integrations at different
step sizes diverge exponentially there no matter how accurate each one is. The
k=1 form sits 0.276 rad from the k=2 one, wide enough for its island chains to
overlap, and once they overlap the chaos is present at any resolution — which is
exactly why quasr44970's k=1 drift is flat at 3e-02 across ns 8/12/16.

So the k=1 sections on this family are showing real chaos of a slightly-wrong
field, not a broken tracer. They should be labelled that way, not published as
flux surfaces.

### 4.2.1 Chaotic lines get no iota — done

A chaotic line HAS no rotational transform, so painting it on the iota colour
scale invents one. Chaos is real physics and gets plotted; it is drawn in **dark
grey** (0.25), the historic convention for "iota could not be inferred", and is
excluded from the colour limits and from the profile fit. Lost/escaped lines
stay a lighter grey (0.55).

The classifier is `iota_convergence`: iota over the first half of the trace
against the second. A quasi-periodic winding number converges like `1/N` so the
halves agree; a chaotic one does not converge at all. Measured:

| case | half-split med / p90 / max | angle residual med / max |
| --- | --- | --- |
| w7x k2 | 9.1e-07 / 8.7e-06 / 7.6e-05 | 1.1e-02 / 2.2e-02 |
| quasr9983 k2 | 1.7e-06 / 3.5e-06 / 3.6e-06 | 7.6e-03 / 1.0e-02 |
| quasr44970 k1 | 9.1e-07 / 8.1e-06 / 4.5e-04 | 7.2e-03 / 1.5e-02 |
| hegna k2 | 3.3e-06 / 4.5e-05 / 4.5e-03 | 2.4e-02 / 1.3e-01 |
| **quasr65530 k1** | **5.6e-04** / 2.5e-03 / 5.8e-03 | 2.0e-02 / 1.1e-01 |

Three orders of magnitude of separation, so `CHAOS_TOL = 1e-4` is not delicate.
**The angle-fit residual does NOT separate** — hegna's clean lines score 2.4e-02
against the chaotic sea's 2.0e-02 — so the claim in section 1 that the residual
distinguishes surface from island from chaos was too optimistic; it flags
islands, not chaos. Classified counts: quasr65530 k1 25 chaotic / 21 lost of 48
(only two lines have an iota at all), hegna k2 4 chaotic, w7x k2 none.

The classification is post-hoc from the archived `(u,v)`, so
`poincare_replot.py` applies it to any run for free. **The `poincare_final`
figures still need one replot pass** — the driver was deliberately not edited
while 25 jobs were queued against it.

**This exposes a flaw in `step_convergence`**: it conflates integration error
with Lyapunov divergence, and reports a huge number for a perfectly integrated
chaotic line. It is a max over seeds, so one chaotic line dominates it. Fix
after the sweep (changing its return type would break jobs in flight): report
the median alongside the max, and measure over a short horizon where algebraic
integration error still dominates the exponential separation. The
angle-fit residual already identifies which lines are chaotic and should be the
thing that gates it.

**Clamping `B^zeta` is not the answer** and was considered: with the denominator
clamped to 1e-9 the right-hand side becomes ~1e9 * B^r and the line flies off,
and if `B^zeta` genuinely crossed zero, clamping to +1e-9 on the negative side
flips the sign of the whole RHS and the line silently traces backwards — a
rendered plot with no NaN and no warning. Here it would never fire anyway.

### 4.3 Activate the k=0 atom in `compute_nullspaces`

The default now consults the atom at k=0 (audit item 3.1/3.7 — the solve path
could not reach what `apply_laplacian_preconditioner(kind='auto')` already
picked). Measured on w7x dbc: **277 → 45 iterations, 8.58 → 1.36 s**, 6.15 s to
assemble, break-even after **0.9 solves**. It repays on the first solve, which
is the opposite of the reasoning I used to leave `compute_nullspaces` at
`ks=(1,2,3)`. The free-BC arm needs re-reading first — my harness omitted
`compute_nullspaces`, so L_0's constant kernel had nothing to deflate against
and both arms ran to 200000 iterations; fixed, rerun queued (job 16743527).

Not touched, both with stated reasons in the audit: **3.1** the production
timestep solve (`_coerce_diffusion_preconditioner_spec`, `valid_kinds` without
`block_jacobi`) — the hot path, unmeasured; **3.2** inverse iteration pinned to
`schur.outer='jacobi'` — the shifted operator is `S_k + eps M_k`, not `L_k`, and
the atom's fit there is unmeasured. `kind='block'` *raises* for `eps != 0` for
the same reason, deliberately without a fallback.

### 4.4 Housekeeping

- `mrx/plotting.py`'s `integrate_fieldlines` / `get_periodic_intersections` /
  `get_iota_log` are superseded here but still used by
  `scripts/config_scripts/poincare_plots.py`, which traces relaxation states.
- Results are copied out of the worktree to `/kfs3/scratch/tblickhan/mrx/outputs/`
  by a background watcher when the last job clears; `outputs/` is gitignored, so
  anything left in the worktree dies with it.

---

## 5. How to run

```bash
# full sweep, one job per (geometry, ns, p)
SPECS="w7x:12,24,12:3 hegna:12,24,12:3" OUTSUB=poincare \
  ARGS="--seeds 48 --periods 1000 --steps 96 --saves 8 --planes 0,0.5 \
        --seed-from axis --profile-x midplane" \
  bash slurm/job_poincare_sweep.sh

# change presentation without re-solving (login node, <1 s, no GPU)
python scripts/debug/poincare_replot.py outputs/.../trace_*.npz

# overlay resolutions / perturbations against the physical label
python scripts/debug/poincare_converge.py --label ns8=... --label ns12=... \
  --title w7x --out conv_w7x.png
```

`--steps` must be a multiple of `--saves` so every saved value is a step
endpoint, not a dense-interpolation value. `--saves` must exceed twice the
poloidal turns per period or the angle unwrap aliases; 8 covers everything here.
