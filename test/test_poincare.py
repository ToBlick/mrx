"""The field-line tracer of :mod:`mrx.diagnostics.poincare`, on the equilibrium's own field.

``b0`` (:func:`mrx.relaxation.initial_conditions.potential_two_form`) is the curl of a vector potential whose
``theta`` and ``zeta`` components depend on ``r`` alone, so ``B^r = 0`` exactly: its field lines stay on their
logical surfaces and wind with the rotational transform of the file. The traced iota of every regular line
matches ``|iota(r)|`` of the file at the line's radius within a measured band, the discretisation error of the
field, and halving the step moves the lines by less than a measured drift.
"""
import numpy as np

from mrx.diagnostics.poincare import poincare

#: 1.25x ``max |iota_traced - |iota_file|| / max |iota_file|`` on li383 (16, 32, 12) p=2, 8 lines over 60 periods,
#: measured 1.06e-3 (2026-10-20). A tracer that missed the factor nfp or a wrong winding would be off by O(1).
IOTA_BAND = 1.33e-3
#: 1.25x the step-halving drift of the same trace in logical radius, measured 3.23e-4.
DRIFT_BAND = 4.1e-4


def test_traced_iota_is_the_file_iota(seq, b0):
    """The traced iota of the regular lines is the file's iota at their radius, and the step is resolved."""
    res = poincare(seq, b0, lines=8, periods=60, planes=1)
    shown = np.asarray(res["shown"])
    r = np.asarray(res["seed_r"])[shown]
    want = np.abs(np.asarray(seq.equilibrium["profiles"]["iota"](r)))
    err = float(np.abs(np.asarray(res["iota"])[shown] - want).max() / want.max())
    print(f"\n  {int(shown.sum())} of {shown.size} lines regular, iota {want.min():.4f}..{want.max():.4f}, "
          f"relative error {err:.2e}, drift {float(res['drift']):.2e}")
    assert shown.all()                  # the field is integrable: no line is chaotic or leaves the domain
    assert err < IOTA_BAND
    assert float(res["drift"]) < DRIFT_BAND
