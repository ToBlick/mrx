"""The relaxation run on li383: the production loop lowers the energy, and
the midpoint scheme with the auxiliary field conserves helicity.

The initial condition is the state's own field, ``B = dA'`` from the
histopolated Clebsch potential (exactly divergence-free); the stepper is
``scripts/relax.py``'s production configuration with velocity smoothing of
order 1. Over ``STEPS`` steps in chunks of ``CHUNK`` the energy must fall at
every step, the force norm must drop by the measured factor, helicity must be
conserved and ``div B`` must stay at roundoff; a reconnection in the middle
must spend the helicity it was asked to, and a checkpoint must round-trip.
"""
import os

import numpy as np

from mrx.relaxation import (IntegrationScheme, TimeStepper, initial_state, read_checkpoint,
                            relax, write_checkpoint)
from test.conftest import NS

STEPS, CHUNK = 50, 25
# ||F||_end / ||F||_0 after 50 steps on li383 (8, 12, 12) p=2, measured
# 2026-09-02: 0.113 .. 0.143 in float64, 0.154 in float32 across three runs
# (the line search is not bitwise reproducible); band 1.25x the largest.
FORCE_DROP = 0.20
# |H_end - H_0| / (2 E_0): helicity is conserved to the solves' tolerance,
# so the band is a multiple of seq.tol in either precision.
HELICITY_DRIFT_TOL = 25.0


def test_relaxation_lowers_the_energy(seq, b0, tmp_path):
    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                     velocity_smoothing_order=1, velocity_smoothing_scale=0.064 / NS[0] ** 2)
    saved = []
    res = relax(initial_state(b0, ts), ts, steps=STEPS, chunk=CHUNK, verbose=False,
                on_chunk=lambda r: saved.append(r.steps))
    E = np.asarray(res.trace["E"], dtype=float)
    F = np.asarray(res.trace["F"], dtype=float)
    H = np.asarray(res.qoi["helicity"], dtype=float)
    div = float(res.trace["div"][-1])
    E0 = E[0] - res.trace["dE_meas"][0]
    print(f"\n  {STEPS} steps: E {E0:.6e} -> {E[-1]:.6e}, ||F|| {F[0]:.3e} -> {F[-1]:.3e} "
          f"({F[-1] / F[0]:.3f}), dH/2E0 {abs(H[-1] - H[0]) / (2 * E0):.2e}, ||div B|| {div:.1e}")
    assert res.stop == "steps" and res.steps == STEPS and saved == [CHUNK, STEPS]
    assert np.all(np.diff(E) < 0.0), f"energy not monotone: {E}"
    assert F[-1] < FORCE_DROP * F[0], f"||F|| {F[0]:.3e} -> {F[-1]:.3e}"
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT_TOL * seq.tol * 2 * E0, \
        f"helicity {H[0]:.6e} -> {H[-1]:.6e}"
    assert div < 1e3 * seq.tol * np.sqrt(2 * E[-1]), f"||div B|| {div:.2e}"

    # A checkpoint round-trips leaf for leaf, and a restart continues the count.
    path = os.path.join(tmp_path, "state.h5")
    write_checkpoint(path, res.state, STEPS)
    state, step = read_checkpoint(path, ts)
    assert step == STEPS
    assert np.array_equal(np.asarray(state.s_history), np.asarray(res.state.s_history))
    assert float(state.dt) == float(res.state.dt)


def test_reconnection_spends_the_helicity_asked_for(seq, b0):
    """One reconnection at the first chunk boundary, 2% of the helicity: the
    dose estimate ``eps = X |H| / (2 |int J . B|)`` is first order, the
    measured price must be within a third of the target and of its sign."""
    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                     velocity_smoothing_order=1, velocity_smoothing_scale=0.064 / NS[0] ** 2)
    res = relax(initial_state(b0, ts), ts, steps=STEPS, chunk=CHUNK, verbose=False,
                reconnect_every=CHUNK, reconnect_helicity=0.02)
    assert len(res.reconnect) == 1 and res.reconnect_every == CHUNK
    ev = res.reconnect[0]
    print(f"\n  reconnection at it {ev['it']}: eps {ev['eps']:.3e}, helicity spent {ev['helicity_spent']:+.3%} "
          f"for a target of -2%, J/B {ev['JoverB_before']:.3f} -> {ev['JoverB_after']:.3f}")
    assert ev["it"] == CHUNK
    assert -0.027 < ev["helicity_spent"] < -0.013, ev["helicity_spent"]
    assert ev["JoverB_after"] < ev["JoverB_before"]
    assert len(res.qoi["it"]) == 4 and res.qoi["it"][1] == res.qoi["it"][2] == CHUNK


def test_midpoint_conserves_helicity(seq, b0):
    """Midpoint-implicit induction with the auxiliary field: Picard converges
    on every step, the energy falls, and helicity is conserved to the solves'
    tolerance."""
    ts = TimeStepper(seq=seq, auxiliary_B_field=True, cfl=0.5, history_size=1,
                     scheme=IntegrationScheme.IMPLICIT_MIDPOINT)
    res = relax(initial_state(b0, ts), ts, steps=20, chunk=10, verbose=False)
    E = np.asarray(res.trace["E"], dtype=float)
    H = np.asarray(res.qoi["helicity"], dtype=float)
    E0 = E[0] - res.trace["dE_meas"][0]
    it, resid = res.trace["picard_it"], res.trace["picard_resid"]
    print(f"\n  20 midpoint steps: E {E0:.6e} -> {E[-1]:.6e}, dH/2E0 "
          f"{abs(H[-1] - H[0]) / (2 * E0):.2e}, increment evaluations max {max(it)}, defect max {max(resid):.2e}")
    assert np.all(np.diff(E) < 0.0), f"energy not monotone: {E}"
    assert max(resid) < ts.picard_tol, f"Picard did not converge: {max(resid)}"
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT_TOL * seq.tol * 2 * E0, \
        f"helicity {H[0]:.6e} -> {H[-1]:.6e}"
