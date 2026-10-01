# Open questions

Measured effects without an explanation, and measurements never taken. Each item names where the
evidence lives.

## Preconditioners and solvers

- The k=1/2 free boundary defect on shaped geometries is a layer (rings 2 and 3 keep paying) whose
  outlier count grows with n_t n_z. No cheap structural fix found. Why W7-X k=1 free stays the
  hardest cell (0.52 of Jacobi at n = 32 vs 0.12-0.36 elsewhere) and why free lags dbc 4-6x
  (preconditioners.md sections 2, 4, 5).
- A depth-3 ring-block generalised eigenproblem reproduced the iteration-optimal scale on 4
  geometries x 2 meshes (k=1, old normalisation) but was never built into a self-computed scale.
- The s = 3 basin is measured only at (12,24,12) p=3. Its h and p dependence on the real solve is
  unmeasured (alpha carries 1/h_last, so s should be h-independent by construction).
- hegna k=2 free does not converge at any s (>= 9068 of 10000 at its optimum), the fourth independent
  flag on that geometry.
- Hodge split: the k=0 axis mechanism (kappa ~ n^1.7) remains. The QA cross-component coupling on the
  exact-orthogonal complement is unmeasured in isolation. The 2.5-12x split-vs-singular gap.
- n_r^1.3 growth of the shifted-stiffness atom: dropped cross-component blocks of S_k (div-div,
  curl-curl, not small) vs missing polar-core/bulk coupling. Measurable with the same probe.
- k=1 free harmonic form degraded with n and p at fixed tolerance (QA, 2026-08-30). Not re-measured
  after the k=1 fix 0c3aa4d.
- TPU: the 9.6x v5e/H200 gap is attributed to nested-solve depth by three layer measurements, no
  op-level profile. compute_nullspaces is 3.2x slower on v5e than on the VM CPU (partly compile).

## Relaxation

- Regular-line drift of the deep penalty-Newton states 3-5e-3 vs 9e-4 before (poincare_islands.md
  section 4): what does it measure?
- High-degree descent: at p = 4, 5 the descent directions go non-descending (cos < 0) and the p = 4/5
  trajectories on the released code deviate 5-15x from the paper's arms with accepted steps 0.84 vs
  5.6-7.7 (not the refinement stop). The axis iota hook grows with p. Smooth-first at high degree or a
  degree-dependent code issue.
- Newton floor exponent in h (4.7x from n = 12 to 16, 1.9x from 16 to 24 under the pre-penalty
  configuration) not re-measured with the penalty.
- The J x Q cross term is ~1% on li383 (far from ideal instability). Near marginal stability the
  Gauss-Newton preconditioner model is expected to fail. W7-X's second variation is indefinite away
  from equilibrium (measured).
- gamma = 1 axis iota dip (0.910 vs 0.915 on W7-X, mesh-independent) from smoothing the polar patch.
  Untested remedy: taper mu to 0 at the axis. Whether the helicity jump takes a few steps or ~100.
- Half period still pays ~2x compile (45-52 vs 26 s) and cold setup (229 vs 107 s at n = 32). Half vs
  field period with the parity DoF reduction at n = 32, 48 not measured.

## Geometry and discretisation

- nbc_k1 Poisson order ~3.3 instead of 4 (under-integration excluded: dbc_k2 reaches 4.4 at the same
  quadrature). Cheapest test is the projection of omega_1 (no solve).
- Toroid B^rho leak 4.6e-9 with an exactly diagonal metric: unexplained. Whether a polar
  histopolation with the explicit local conforming projection removes the L2 B^rho leak on shaped
  geometry.
- L2 projection onto the polar space itself (E M E^T) vs tensor projection then restriction: rings
  0-1 differ by O(h^{p+1}), not measured.
- lambda invariance (lambda changes energy and force, never fluxes, iota or helicity) never tested on
  the real solve path.
- The 1.05-1.07% floor of the de-rotated simsopt W7-X files: coarse 8x16x8 geometry vs a real
  Biot-Savart difference.
- Cary-Hanson on W7-X FMM002: inconsistent fixed points. Which one is the 5/5 O-point.
