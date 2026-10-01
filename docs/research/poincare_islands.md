# Poincare sections and islands: measurements

The tracer, the plotter and the island diagnostic are described in the Sphinx docs. This file keeps
the calibration numbers and the traps.

## 1. Tracer calibration

- Iota checks (2026-08-24): toroid 0 to 1e-17. W7-X vacuum 0.851-0.948 (the published range).
  Rotating ellipse exactly 0 by a reflection symmetry the map forces.
- Chaos classifier = iota over the first vs second half of a line, median / p90: W7-X k2 9.1e-7 /
  8.7e-6, quasr9983 k2 1.7e-6, quasr44970 k1 9.1e-7, hegna k2 3.3e-6 / 4.5e-5 vs the chaotic
  quasr65530 k1 5.6e-4 / 2.5e-3: three orders of separation, CHAOS_TOL 1e-4 is not delicate. The
  angle-fit residual does NOT separate chaos (clean hegna 2.4e-2 vs chaotic 2.0e-2). It flags islands
  (2026-08-24).
- h vs h/2 drift on chaotic lines measures the Lyapunov exponent, not integration error: a relaxed
  W7-X state read 5.7e-1 (0.51 at 48 steps/period, 0.71 at 96) while iota agreed to 4 decimals across
  the 4x refinement. Over regular lines only 1.6e-4 (2026-08-25). Rank fields on |dH| and axis offset,
  not on drift over all lines.
- W7-X needs more than 24 steps per period: h/2 drift 0.09-0.17 at 24, 1e-4 at 96 (2026-09-12).
- The k=1 natural-BC harmonic fields of quasr44970/65530 are genuinely chaotic: drift 3.5e-2 /
  2.8e-2 / 3.6e-2 flat over ns 8/12/16. The k=1 form sits 0.276 rad from the k=2 one. B^zeta/|B| >=
  0.774 everywhere (0.828 for k2), so the zeta reparametrisation is not the cause. A B^zeta gate at
  0.05 is far below anything measured (2026-08-24).
- Iota surface label: the outboard-midplane distance from the magnetic axis gives 0 reversals on 7
  cases vs 1-2 for sqrt(A/pi) or the mean distance (which weights by crossing density). NaN on
  non-surfaces (2026-08-24).
- Magnetic vs coordinate axis: W7-X beta 4.2% export 4.86 cm apart (the Shafranov shift), vacuum
  0.6 mm. Relaxed fmm002 (6,8,8) 6.6 cm at zeta = 0, 2.4 cm at 0.5. Seed from the magnetic axis, not
  r = 0 (2026-08-24/25).

## 2. Island width

- Width measure: largest max(r) - min(r) in logical rho of any non-chaotic line whose fitted iota is
  on the rational to 2e-3. Floor of the measure (spread of lines within 2e-3 of the rational, nested
  32^3): 0.028 / 0.023 / 0.019 at 3/5, 1/2, 3/7. 0.054 at 16^3 (2026-09-22). Tracer spacing in the
  li383 sections 0.006.
- Section widths are right with 160 lines. 48 lines gave ~half (a lower bound). Ray scan through the
  O-point (121 seeds, 300 periods) vs the section widths (160 lines), in h_r: (5,1) initial 2.38 vs
  2.3, unseeded final 3.41 vs 3.4, (5,1) final 3.27 vs 3.3, (6,1) final 3.39 vs 3.4 (2026-09-18).
- The seed-selection chain count is the spectral cutoff m <= n_theta/2 (8 / 17 / 28 at n_theta 16 /
  24 / 32) (2026-09-22).
- Seed width law: pendulum 1.6 sqrt(eps nfp / (m |iota'|)) matches at t = 0. Chains narrower than ~3
  radial cells are healed by numerical reconnection (relaxation.md section 7).
- Seed-amplitude and resistive-demo results: relaxation.md section 7.

## 3. Cary-Hanson residue

- Fixed points (li383 (10,16,16) p=2, (6,1) seed at rho 0.544, 2026-09-18): Newton converges to 1e-15
  from both guesses. The tangent map needs 96 steps/period (24: residue 0.0877 vs 0.0842, det S
  0.989, 96: det S 1.00005 at the O-point, 1.0033 at the X-point). The shear must be the UNSEEDED
  profile's (0.30, a fit to the seeded section gives 0.32 / 0.22, a 30% width error at eps 1e-2).
- Residue width w = 4 omega / (m |iota'|), omega = arccos(1 - 2R) / (2pi m): li383 3/5 chain 0.0496 vs
  0.0503 from the lines (residue 0.2). The seed's pendulum estimate to 8% / 1% (same model, not an
  independent check). VMEC IC 9e-3 (an eighth of a cell).
- Residue is not a width: on the li383 (16,32,32) paper fields the pendulum conversion overestimates
  the traced separatrix 1.35-1.6x (residue widths 3.2 / 5.4 / 4.2 / 5.6 h_r vs traced 2.3 / 3.4 / 3.3 /
  3.4). 1.7x over for wide chains (residue 0.9-0.96). Use the residue for existence and phase only.
- Equilibrium shears 0.472 / 0.311 / 0.215 at 3/5, 1/2, 3/7 (r = 0.797 / 0.543 / 0.268). An intact
  rational surface gives residue ~0.005 and zero width: report a chain only if locked lines cross its
  O-point. Ladder 3/5: R 0.160 -> 0.206 while the local shear falls 0.42 -> 0.26 (late growth is shear
  flattening).
- The (5,1)-seeded chain flips phase (O-point at theta 0 when seeded, 0.108 once the reconnection
  grows it).
- On W7-X FMM002 the fixed points found were not consistent (two fixed points 0.087 apart in theta at
  one radius) (2026-09-17).

## 4. Sections of relaxed states

- Penalty Newton states 20x past the old floor, li383 (16,32,32), 160 lines x 400 periods x 5 planes
  (2026-09-17): 3-4/160 chaotic (as the best earlier state, 4/160), iota 0.393..0.660 unchanged. The
  unpenalised control 200 steps past the floor: 51/160 chaotic, iota flattened at 1/2 over r
  0.6-0.7. (32,64,64): 0/160 chaotic. W7-X FMM002 (32,64,64) both configurations 0/160, iota
  0.915..1.056.
- Regular-line drift of the deep Newton states is higher than before and unexplained: 4.8e-3
  (alpha 1) / 5.2e-3 (alpha 0.3) / 3.2e-3 (200 steps) vs 8.8e-4 for the earlier best. (32,64,64)
  2.2e-3 vs 1.6e-3.
- W7-X seeds (2026-09-13): the seed n counts field periods. (5,1) at rho 0.83 is the iota = 1 chain.
  Seeded 3.03 h_r -> 2.74 h_r after Newton (within 10%). The natural unseeded chain 0 -> 0.65 h_r. An
  (11,2) seed is damped to a quarter cell at m = 11 on 32 poloidal cells.
- Early W7-X fmm002 relaxation (8^3 p=3, 2026-08-25) shows clean chains at 5/6, 10/11, 5/5. eta 1e-2
  removes them (vacuum-like, perfectly nested).
