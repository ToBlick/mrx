"""The force-residual stopping criterion of ``scripts/relax.py``.

Numpy only: ``scripts/relax.py`` is loaded by path and imports mrx only
inside ``main``. The trace is synthetic: a decaying residual with a
non-monotone oscillation on top, which is what a relaxation produces (the
scheme guarantees ``dE/dt <= 0``, not a falling force). The criterion is the
mean over the last ``--floor-steps`` steps, never the last value, so a single
dip below the tolerance must not stop the run.
"""
import importlib.util
import os

import numpy as np

STEPS = 100
TOL = 1e-3


def _relax_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "relax.py")
    spec = importlib.util.spec_from_file_location("relax_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_residual(n=3000, floor=8e-4, amp=6e-4, decay=600.0):
    """``floor + 2e-3 exp(-i/decay)`` with a ripple of amplitude ``amp``.

    Non-monotone by construction: consecutive values rise on about half of
    the steps, and the ripple carries single values below ``TOL`` long before
    the window mean gets there.
    """
    i = np.arange(n)
    return floor + 2e-3 * np.exp(-i / decay) + amp * np.sin(i / 3.0) ** 2 * np.cos(i / 7.0)


def _first_floor_step(resid, steps, tol, floor_reached):
    for i in range(1, len(resid) + 1):
        if floor_reached(resid[:i], steps, tol):
            return i
    return None


def test_floor_needs_a_full_window():
    relax = _relax_module()
    assert not relax.force_floor_reached([0.0] * (STEPS - 1), STEPS, TOL)
    assert relax.force_floor_reached([0.0] * STEPS, STEPS, TOL)


def test_floor_is_the_window_mean_not_the_last_value():
    relax = _relax_module()
    r = _synthetic_residual()
    assert (np.diff(r) > 0).mean() > 0.3          # the trace is not monotone
    first_dip = int(np.argmax(r < TOL)) + 1
    fired = _first_floor_step(r, STEPS, TOL, relax.force_floor_reached)
    assert fired is not None
    assert fired > first_dip + STEPS, (fired, first_dip)
    # the mean over the window at the firing step is what dropped below TOL
    assert np.mean(r[fired - STEPS:fired]) < TOL
    assert np.mean(r[fired - STEPS - 1:fired - 1]) >= TOL
    # and it stays fired on the tail of the trace
    assert relax.force_floor_reached(r, STEPS, TOL)


def test_floor_silent_on_a_residual_that_only_dips():
    relax = _relax_module()
    r = _synthetic_residual(floor=1.2e-3, amp=8e-4)
    assert (r < TOL).any()                        # single values below TOL
    assert _first_floor_step(r, STEPS, TOL, relax.force_floor_reached) is None
