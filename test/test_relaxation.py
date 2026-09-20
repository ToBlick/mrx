"""The relaxation run on li383: the production gradient-descent loop lowers
the energy.

The initial condition is the state's own field, ``B = dA'`` from the
histopolated Clebsch potential (exactly divergence-free); the stepper is
``scripts/relax.py``'s gradient-descent configuration with velocity smoothing
of order 1 (the potential route). Over ``STEPS`` steps in chunks of ``CHUNK``
the energy must fall at every step, the force norm must drop by the measured
factor, helicity must be conserved and ``div B`` must stay at roundoff, the
best state must be the step of the lowest residual, and a checkpoint must
round-trip. One relaxation run: the suite is compile-bound and every stepper
configuration is its own compile of the scan body, so the helicity
correction, the reconnection series and the Newton loop are not run here on
purpose (2026-09-20).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import DTYPE, eps, sqrt_eps

from mrx.relaxation import TimeStepper, initial_state, read_checkpoint, relax, write_checkpoint

STEPS, CHUNK = 50, 25
# ||F||_end / ||F||_0 after 50 steps on li383 (8, 12, 12) p=2: gradient
# descent on the smoothed force, measured 2026-09-17 0.205 in mixed precision
# (the L-BFGS direction of before, removed that day, gave 0.113 .. 0.154 on
# 2026-09-02; the line search is not bitwise reproducible); band 1.25x.
FORCE_DROP = 0.26
# |H_end - H_0| / (2 E_0): the helicity drifts by the rounding of the stored
# field, not by a solve (the solves are refined to 1e-8 in float64 whatever
# the working dtype), so the band is a multiple of sqrt(eps) of the working
# dtype: 3.5e-4 in float32, 1.5e-8 in float64.
HELICITY_DRIFT_TOL = 25.0


def test_relaxation_lowers_the_energy(seq, b0, tmp_path):
    ts = TimeStepper(seq=seq, cfl=0.5, velocity_smoothing_order=1)
    saved = []
    res = relax(initial_state(b0, ts), ts, steps=STEPS, chunk=CHUNK, verbose=False,
                on_chunk=lambda r: saved.append(r.steps))
    dE = np.asarray(res.trace["dE"], dtype=float)
    F = np.asarray(res.trace["F"], dtype=float)
    H = np.asarray(res.qoi["helicity"], dtype=float)
    div = float(res.trace["div"][-1])
    E0, E1 = res.E0, res.E0 + dE.sum()
    print(f"\n  {STEPS} steps: E {E0:.6e} -> {E1:.6e}, ||F|| {F[0]:.3e} -> {F[-1]:.3e} "
          f"({F[-1] / F[0]:.3f}), dH/2E0 {abs(H[-1] - H[0]) / (2 * E0):.2e}, ||div B|| {div:.1e}")
    assert res.stop == "steps" and res.steps == STEPS and saved == [CHUNK, STEPS]
    leaked = {path for path, leaf in jax.tree_util.tree_flatten_with_path(res.state)[0]
              if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
              and jnp.asarray(leaf).dtype != DTYPE}
    assert not leaked, f"state leaves not in the working dtype: {leaked}"
    # The per-step dE is formed in the stored precision: once the true change
    # per step is below an epsilon of the energy (6e-8 in float32, after ~45
    # of these steps in the plain float32 configuration) a step can read
    # +5e-9; the descent is monotone to that roundoff, strictly in float64.
    assert np.all(dE < eps() * E0), f"energy not monotone: {dE}"
    assert F[-1] < FORCE_DROP * F[0], f"||F|| {F[0]:.3e} -> {F[-1]:.3e}"
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT_TOL * sqrt_eps() * 2 * E0, \
        f"helicity {H[0]:.6e} -> {H[-1]:.6e}"
    assert div < 1e3 * seq.tol * np.sqrt(2 * E1), f"||div B|| {div:.2e}"
    resid = np.asarray(res.trace["resid"], dtype=float)
    assert float(res.state.resid_best) <= resid.min() and int(res.state.step_best) == resid.argmin(), \
        (float(res.state.resid_best), resid.min(), int(res.state.step_best), resid.argmin())

    # A checkpoint round-trips leaf for leaf, and a restart continues the count.
    path = os.path.join(tmp_path, "state.h5")
    write_checkpoint(path, res.state, STEPS)
    state, step = read_checkpoint(path, ts)
    assert step == STEPS
    assert np.array_equal(np.asarray(state.F_prev), np.asarray(res.state.F_prev))
    assert float(state.dt) == float(res.state.dt)
