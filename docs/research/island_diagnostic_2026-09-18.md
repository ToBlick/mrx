# The Cary-Hanson island diagnostic (2026-09-18)

`mrx.poincare.fixed_points`, `island_width`, `islands`: the O and X points
of an island chain by Newton on the return map, Greene's residue from the
tangent map, and the pendulum width from the residue -- a width that does
not depend on which seeds happened to lock onto the chain, unlike the
`max(r) - min(r)` of the locked lines the figures quote. Tobias
2026-09-17: "Do that, it is good to have."

## What it does

* `return_map(field, dof, periods)`: `y -> Phi(y)` in the `(u, v)` chart of
  the `zeta = 0` plane after `periods` field periods, the same fixed-step
  Tsit5 integration as `trace`, jitted and reverse-mode differentiable.
  Float64 in and out whatever the field's dtype.
* `fixed_points(seq, dof, periods, guesses)`: Newton on `Phi(y) - y` with
  `S = jacrev(Phi)`, a capped step, 12 iterations from every guess at once
  (vmap). Residue `R = 1/2 - tr S / 4`: `0 < R < 1` an O-point, `R < 0` an
  X-point, `R > 1` hyperbolic with reflection. `det S` should be 1 (a
  divergence-free field preserves the flux measure, and at a fixed point
  the chart's weights cancel) -- its departure is the tangent map's
  integration error. `defect = |Phi(y) - y|`.
* `island_width(R, m, iota', nfp) = 4 nfp arcsin(sqrt R) / (pi m^2 |iota'|)`
  in logical `r`, `iota'` per toroidal turn as `poincare` reports it. The
  derivation (docstring): with `phi = m theta - n zeta` in turns and
  `zeta` in periods, `iota_p = iota / nfp`, the thin-island pendulum has
  small oscillations `omega = 2 pi m sqrt(eps iota_p')` per period and the
  separatrix at `dr = 2 sqrt(eps / iota_p')`; eliminating `eps` the full
  width is `2 omega / (pi m |iota_p'|)`; the `m`-period return map rotates
  by `m omega` about the O-point with `R = sin^2(m omega / 2)`.
* `islands(seq, dof, m, n, res, iota_prime=None)`: the chain radius from
  the section `res` (where the regular lines' iota crosses `nfp n / m`),
  guesses `(r, 0)` and `(r, 1/(2m))` -- one per kind: the `m` O-points and
  `m` X-points alternate every `1/(2m)` in `theta`, and for even `m` both
  symmetry lines `theta = 0, 1/2` are the SAME kind (the first version
  guessed the two symmetry lines and found the O-point twice).

## The check (`island_diagnostic_2026-09-18/check_islands.py`)

li383 (10,16,16) p=2, the (6,1) seed at `rho0 = 0.544`, width 0.1, eps 3e-3
and 1e-2; `poincare(lines=48, periods=200)`; the unseeded field's section
gives the shear `iota' = 0.3002` by a linear fit over `+-0.1` around the
chain (10 lines).

| eps | point | r | theta | residue | det S | defect | width (residue) | seed estimate | section max-min |
|---|---|---|---|---|---|---|---|---|---|
| 3e-3 | O | 0.5423 | 0.0000 | +0.0842 | 1.00005 | 1.5e-15 | 0.104 | 0.113 | 0.052 (2 lines) |
| 3e-3 | X | 0.5405 | 0.0778 | -0.0751 | 1.0033 | 1.7e-15 | | | |
| 1e-2 | O | 0.5438 | 0.0000 | +0.3001 | 1.0008 | 1.0e-15 | 0.205 | 0.207 | 0.144 (5 lines) |
| 1e-2 | X | 0.5373 | 0.0777 | -0.2102 | 1.0034 | 7.4e-16 | | | |

The seed estimate is `1.6 sqrt(eps nfp / (m |iota'|))` (the docs' formula
for `--seed-eps`). The residue width agrees with it to 8% and 1%; the
section's `max(r) - min(r)` is a lower bound, half the width with 48
seeds over the radius (two and five of them inside the island).

Three things the check taught:

1. **The tangent map needs more steps than the trajectory.** At 24 steps
   per period the O-point residue was 0.0877 / 0.3057 with `det S =
   0.989`; at 48, 0.0831 / 0.3005 (det 1.006); at 96, 0.0842 / 0.3001
   (det 1.00005 / 1.0008). `TANGENT_STEPS_PER_PERIOD = 96` is the default
   of `fixed_points`; the trajectory's 24 stay for `poincare`.
2. **The shear must be the unperturbed profile's.** The island flattens
   iota over its width, and a fit to the seeded section follows the seed:
   0.32 at eps 3e-3, 0.22 at 1e-2 (locked lines excluded), against 0.30
   unseeded -- a 30% error in the width at the larger seed. `islands`
   takes `iota_prime=`; the fit is the fallback for a thin island.
3. **The chain radius from the seeded profile is only a guess** (0.566 and
   0.484 for a chain at 0.544 -- the flattened profile crosses the target
   away from the chain), and Newton does not care: it converged from both
   to the same points to 1e-15. `r_chain` is reported for what it is.

The X-point's `det S` converges more slowly (1.0033 at 96 steps): the
hyperbolic tangent map amplifies its own integration error. The residue
does not suffer (-0.0744, -0.0751, -0.0751).

## The paper's island widths, re-measured (2026-09-18)

`island_diagnostic_2026-09-18/paper_widths.py` (the seeded-reconnection
table, the ladder `ladder3`, the seeded table `seed61_2` / `seed51_2`;
runs of the newton branch) and `paper_widths_mesh.py` (fig:mesh_refinement,
`outputs/li383_pulse/reconnect_l5_{h16,h32u,h32r}_p2_g1`, the refined mesh
rebuilt with the `radial_knots` of commit 3740157). Logs under
`outputs/half_period/2026-09-18/{10-03-47,10-05-51,11-20-05}`; about a
minute per evaluation (12 Newton steps with a Jacobian through m periods
at 96 steps), so 54 entries do not fit a 90-minute job. Widths in h_r =
1/16, "residue" with the run's own shear at that field (linear fit over
+-0.12, locked lines out); equilibrium shears 0.472 / 0.311 / 0.215 at
the 3/5, 1/2, 3/7 surfaces (r = 0.797, 0.543, 0.268).

* Residue over section width is 1.35-1.6 on every clean entry (one O and
  one X found, defect 1e-15): the section measure is a lower bound by a
  steady factor, the trends survive.
* Seeded table: 1/2 chain 3.7 -> 4.0 (paper 2.6 -> 2.7), 3/5 chain 3.2 ->
  2.9 (paper 2.3 -> 2.1); "within 10%" holds. The (5,1) run's island keeps
  shrinking after its best state: R 0.275 -> 0.125 by step 200.
* Ladder, 3/5 chain: paper 0.8, 1.5, 2.0, 2.5, 3.0; residue n/m, 2.4, 2.8,
  3.6, 4.5 with R = 0.160, 0.168, 0.192, 0.206 while the local shear falls
  0.42 -> 0.26: the late growth is shear flattening, the resonant field
  nearly saturates after the second event. With the equilibrium shear the
  widths read 2.1-2.4: for an island as wide as the fit window quote R.
* Seeded-reconnection table, 3/5 chain final: section 3.4 / 3.3 / 3.4,
  residues 0.198 / 0.133 / 0.211 (unseeded / (5,1) / (6,1)): the widths do
  NOT converge across the runs the way the section numbers suggest. The
  dip in the (5,1) row is a PHASE FLIP: the seeded O-point sits on theta =
  0, the chain the reconnection grows has it at theta = 0.108; the seeded
  island closes around event 2 and re-opens in the natural phase.
* Seeded 1/2 chain: R 0.39, 0.62, 0.27, 0.16, 0.066, 0.025; final width
  1.0 h_r against the paper's "> 1.0 h_r".
* Mesh figure: 3/5 chain after the solve 0.0605 / 0.0538 / 0.0578 in r
  (final 0.0634 / 0.0596 / 0.0556): agreement to 6%, not the 2% of the
  section widths 0.049 / 0.050 / 0.050.
* Not measured: the 3/7 chain everywhere and the unseeded 1/2 chain
  (|R| < 0.02, the two guesses return fixed points of the same kind --
  presumably a doubled chain, fixed points every 1/(4m); guesses at
  1/(4m) would settle it), and the 3/5 chain as it opens (paper 0.7-0.8).

## CORRECTION (2026-09-18, afternoon): the residue is not a width here

Tobias: the three final sections of the seeded-reconnection runs show no
visible difference in the 3/5 chain. `ray_widths.py` measured the
separatrix directly: the O-point by Newton, 121 seeds on the radial ray
through it (spacing 0.06 h_r), 300 periods, locked = iota within 2e-3.

| field | paper (section) | ray scan, largest excursion | ray extent | residue width quoted above |
|---|---|---|---|---|
| (5,1) seed, initial | 2.3 | 2.38 | 2.29 | 3.2 |
| unseeded, final | 3.4 | 3.41 | 3.23 | 5.4 |
| (5,1) seed, final | 3.3 | 3.27 | 3.06 | 4.2 |
| (6,1) seed, final | 3.4 | 3.39 | 3.18 | 5.6 |

So the paper's section widths are RIGHT (160 lines seed the chain well
enough; the "lower bound" of the (10,16,16) check was 48 lines), the three
finals agree, and the residue-to-width conversion OVERESTIMATES by
1.35-1.6 on these fields. It matched the seed's pendulum estimate in the
check because both are the same constant-shear single-harmonic model;
the seeded perturbation is a Gaussian of width 0.1, comparable to the
island, and the late states have an iota profile that flattens and turns
over outside the chain. Withdrawn from the section above: "the section
measure is a lower bound by a steady factor", "the widths do not converge
across the runs", "the mesh figure agrees to 6% not 2%" (unverified by a
direct measurement), and every residue WIDTH of a wide or edge island.
What stands: the residues themselves, the existence and position of the
fixed points, and the phase flip of the (5,1)-seeded chain (O-point on
theta = 0 initially, at 0.108 from event 3 on). Use the diagnostic for
the O-point's position and rotation rate and to PLACE the seeds of a
width measurement (the ray through the O-point); quote widths from the
traced lines.


## The diagnostic as it stands: find every chain, measure its width (2026-09-18, evening)

Tobias: "The Cary-Hanson stuff should stay as a diagnostic, which should
try and find all islands in the state and then give the width for them."
`mrx.poincare.islands(seq, B, res=None, m_max=12, n_theta=8, ...)`:

1. the iota profile of the section's regular lines (`res`, traced with
   `poincare`'s defaults when left out -- random poloidal seeds, so the
   profile is a guess of where to look, nothing more);
2. `resonances(iota_lo, iota_hi, nfp, m_max)`: the coprime `(m, n)` with
   `nfp n / m` in range; `_chain_radii`: every radius where the profile
   meets the rational (a plateau of locked lines, or a crossing; a
   reversed-shear profile meets it twice);
3. `fixed_points` from `n_theta = 8` guesses across one chain period `1/m`
   -- the two-guess version missed doubled chains and any chain off the
   symmetry line; distinct converged points within `window` of the radius
   are the chain's O and X points;
4. the width by tracing: `ray_seeds = 81` lines on the radial ray through
   the strongest O-point, `+- 0.2`, all chains in ONE batched trace of 300
   periods; the locked lines contiguous with the O-point are the island,
   `width` their largest `max(r) - min(r)` over eight planes per period;
5. a chain with no locked line through its O-point is dropped: on the
   unseeded initial field the first version reported an (11,2) "chain"
   with residue 0.005 and zero width -- an intact rational surface, whose
   residue is zero only up to the tangent map's integration error.

The return map over `m` periods is the one-period map composed `m` times
(the field is periodic in zeta), the tangent map the product of the
one-period Jacobians by `jax.jacfwd` through diffrax's `ForwardMode`
adjoint, `m` a traced loop bound: one compile for every chain order. The
first implementation differentiated the whole `m`-period trace in reverse
mode and compiled once per `m` and per call (a minute an evaluation).
First timing, li383 (16,32,32), unseeded initial field: section 61 s,
island search 116 s including its compile.

`island_width` (the pendulum formula) and `return_map` are gone from the
module; the scripts of this folder that import them (`check_islands.py`,
`paper_widths*.py`, `ray_widths.py`) ran at commit 0ff2369 and are the
record of the tables above. `find_islands.py` is the check of the new
function on fields with known islands (job 18657661; its results are not
in this note yet).
