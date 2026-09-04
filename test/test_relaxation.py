"""One relaxation run on li383: the production descent lowers the energy.

The initial condition is the state's own field, ``B = dA'`` from the
histopolated Clebsch potential (exactly divergence-free); the stepper is
``scripts/relax.py``'s production configuration with velocity smoothing of
order 1 (``docs/source/tutorials.md``). Over ``OUTER x INNER`` steps the
energy must fall at every recorded point, the force norm must drop by the
measured factor, helicity must be conserved and ``div B`` must stay at
roundoff.
"""
import numpy as np

from mrx.relaxation import DescentMethod, TimeStepChoice, TimeStepper, relaxation_loop
from test.conftest import NS

OUTER, INNER = 2, 25
# ||F||_end / ||F||_0 after OUTER x INNER steps on li383 (8, 12, 12) p=2,
# measured 2026-09-02: 0.113 .. 0.143 in float64, 0.154 in float32 across
# three runs (the line search is not bitwise reproducible); band 1.25x the
# largest.
FORCE_DROP = 0.20
# |H_end - H_0| / (2 E_0): helicity is conserved to the solves' tolerance,
# so the band is a multiple of seq.tol in either precision.
HELICITY_DRIFT_TOL = 25.0


def test_relaxation_lowers_the_energy(seq, b0):
    B0 = b0
    ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                     dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                     eta_every=1, resistive=False, history_size=1,
                     velocity_smoothing_order=1, velocity_smoothing_scale=0.064 / NS[0] ** 2)
    state, traces = relaxation_loop(B0, ts, num_iters_outer=OUTER, num_iters_inner=INNER,
                                    dt0=1.0, force_tolerance=1e-12)
    E = np.asarray(traces["energy"], dtype=float)
    F = np.asarray(traces["force_norm"], dtype=float)
    H = np.asarray(traces["helicity"], dtype=float)
    div = float(traces["divergence_B"][-1])
    print(f"\n  {OUTER * INNER} steps: E {E[0]:.6e} -> {E[-1]:.6e}, ||F|| {F[0]:.3e} -> {F[-1]:.3e} "
          f"({F[-1] / F[0]:.3f}), dH/2E0 {abs(H[-1] - H[0]) / (2 * E[0]):.2e}, ||div B|| {div:.1e}")
    assert np.all(np.diff(E) < 0.0), f"energy not monotone: {E}"
    assert F[-1] < FORCE_DROP * F[0], f"||F|| {F[0]:.3e} -> {F[-1]:.3e}"
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT_TOL * seq.tol * 2 * E[0], \
        f"helicity {H[0]:.6e} -> {H[-1]:.6e}"
    assert div < 1e3 * seq.tol * np.sqrt(2 * E[-1]), f"||div B|| {div:.2e}"


def test_midpoint_conserves_helicity(seq, b0):
    """Midpoint-implicit induction: Picard converges on every recorded
    step, the energy falls, and helicity is conserved to the solves' tolerance."""
    from mrx.relaxation import IntegrationScheme
    ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                     dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                     eta_every=1, resistive=False, history_size=1,
                     scheme=IntegrationScheme.IMPLICIT_MIDPOINT)
    state, traces = relaxation_loop(b0, ts, num_iters_outer=2, num_iters_inner=10,
                                    dt0=1.0, force_tolerance=1e-12)
    E = np.asarray(traces["energy"], dtype=float)
    H = np.asarray(traces["helicity"], dtype=float)
    it = traces["picard_iterations"][1:]
    resid = traces["picard_residual"][1:]
    print(f"\n  20 midpoint steps: E {E[0]:.6e} -> {E[-1]:.6e}, dH/2E0 "
          f"{abs(H[-1] - H[0]) / (2 * E[0]):.2e}, increment evaluations {it}, residuals {resid}")
    assert np.all(np.diff(E) < 0.0), f"energy not monotone: {E}"
    assert max(resid) < ts.picard_tol, f"Picard did not converge: {resid}"
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT_TOL * seq.tol * 2 * E[0], \
        f"helicity {H[0]:.6e} -> {H[-1]:.6e}"
