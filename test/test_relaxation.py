"""The relaxation run on li383: the production Newton loop lowers the energy.

The initial condition is the state's own field, ``B = dA'`` from the
histopolated Clebsch potential (exactly divergence-free); the stepper is
``scripts/relax.py``'s default, Newton-MR on the second variation with the
harmonic atom and the parallel-flow penalty, the line search capped at the
Newton length. Over ``STEPS`` steps in chunks of ``CHUNK`` the energy must
fall at every step, the force norm must drop, helicity must be conserved,
``div B`` must stay at roundoff, the best state must be the step of the
lowest residual, no leaf may leave the working dtype, and a checkpoint must
round-trip.

One relaxation run by default: the suite is compile-bound and every stepper
configuration is its own compile of the scan body. The gradient-descent run
(the potential route, the paper's descent block) is ``optional`` -- ``pytest
-m optional``; the helicity correction and the reconnection series are not
run on purpose (2026-09-20).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.precision import DTYPE, eps, sqrt_eps
from mrx.relaxation import TimeStepper, initial_state, read_checkpoint, relax, write_checkpoint

STEPS, CHUNK = 10, 5
# ||F||_end / ||F||_0 after 10 Newton steps on li383 (8, 12, 12) p=2.
# PROVISIONAL (2026-09-20, not yet measured on this mesh): to be replaced by
# 1.25x the measured drop once the suite has run.
NEWTON_FORCE_DROP = 0.5
# |H_end - H_0| / (2 E_0): the helicity drifts by the rounding of the stored
# field, not by a solve (the solves are refined to 1e-8 in float64 whatever
# the working dtype), so the band is a multiple of sqrt(eps) of the working
# dtype: 3.5e-4 in float32, 1.5e-8 in float64.
HELICITY_DRIFT_TOL = 25.0


def _check_run(seq, ts, res, steps, chunk, saved, force_drop, tmp_path):
    dE = np.asarray(res.trace["dE"], dtype=float)
    F = np.asarray(res.trace["F"], dtype=float)
    H = np.asarray(res.qoi["helicity"], dtype=float)
    div = float(res.trace["div"][-1])
    E0, E1 = res.E0, res.E0 + dE.sum()
    print(f"\n  {steps} steps: E {E0:.6e} -> {E1:.6e}, ||F|| {F[0]:.3e} -> {F[-1]:.3e} "
          f"({F[-1] / F[0]:.3f}), dH/2E0 {abs(H[-1] - H[0]) / (2 * E0):.2e}, ||div B|| {div:.1e}")
    assert res.stop == "steps" and res.steps == steps and saved == [chunk, steps]
    leaked = {path for path, leaf in jax.tree_util.tree_flatten_with_path(res.state)[0]
              if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
              and jnp.asarray(leaf).dtype != DTYPE}
    assert not leaked, f"state leaves not in the working dtype: {leaked}"
    # The per-step dE is formed in the stored precision: once the true change
    # per step is below an epsilon of the energy a step can read +5e-9; the
    # descent is monotone to that roundoff, strictly in float64.
    assert np.all(dE < eps() * E0), f"energy not monotone: {dE}"
    assert F[-1] < force_drop * F[0], f"||F|| {F[0]:.3e} -> {F[-1]:.3e}"
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT_TOL * sqrt_eps() * 2 * E0, \
        f"helicity {H[0]:.6e} -> {H[-1]:.6e}"
    assert div < 1e3 * seq.tol * np.sqrt(2 * E1), f"||div B|| {div:.2e}"
    resid = np.asarray(res.trace["resid"], dtype=float)
    assert float(res.state.resid_best) <= resid.min() and int(res.state.step_best) == resid.argmin(), \
        (float(res.state.resid_best), resid.min(), int(res.state.step_best), resid.argmin())

    # A checkpoint round-trips leaf for leaf, and a restart continues the count.
    path = os.path.join(tmp_path, "state.h5")
    write_checkpoint(path, res.state, steps)
    state, step = read_checkpoint(path, ts)
    assert step == steps
    assert np.array_equal(np.asarray(state.F_prev), np.asarray(res.state.F_prev))
    assert float(state.dt) == float(res.state.dt)


def test_newton_relaxation_lowers_the_energy(seq, b0, tmp_path):
    ts = TimeStepper(seq=seq, cfl=0.5, newton=True)
    saved = []
    res = relax(initial_state(b0, ts), ts, steps=STEPS, chunk=CHUNK, verbose=False,
                on_chunk=lambda r: saved.append(r.steps))
    _check_run(seq, ts, res, STEPS, CHUNK, saved, NEWTON_FORCE_DROP, tmp_path)
    it = np.asarray(res.trace["newton_it"])
    print(f"  MINRES iterations per step: {np.abs(it).tolist()}")
    assert np.all(it != 0), "a step without a Newton solve"


# ||F||_end / ||F||_0 after 50 gradient-descent steps on li383 (8, 12, 12)
# p=2 (the smoothed force on the potential route), measured 2026-09-17 0.205
# in mixed precision (the line search is not bitwise reproducible); band 1.25x.
GRADIENT_STEPS, GRADIENT_CHUNK, GRADIENT_FORCE_DROP = 50, 25, 0.26


@pytest.mark.optional
def test_gradient_descent_lowers_the_energy(seq, b0, tmp_path):
    """The paper's descent block: gradient descent on the smoothed force
    through the potential route, ``scripts/relax.py --method gradient``."""
    ts = TimeStepper(seq=seq, cfl=0.5, velocity_smoothing_order=1)
    saved = []
    res = relax(initial_state(b0, ts), ts, steps=GRADIENT_STEPS, chunk=GRADIENT_CHUNK, verbose=False,
                on_chunk=lambda r: saved.append(r.steps))
    _check_run(seq, ts, res, GRADIENT_STEPS, GRADIENT_CHUNK, saved, GRADIENT_FORCE_DROP, tmp_path)
