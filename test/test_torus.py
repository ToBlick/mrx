"""The flux-surface solve of :mod:`mrx.diagnostics.torus`, on the equilibrium's own field.

``b0`` (:func:`mrx.relaxation.initial_conditions.potential_two_form`) has ``B^r = 0`` exactly, so its flux surfaces
are the logical surfaces ``r = const``, their rotational transform is the file's, sign included, and the toroidal
flux inside them grows like ``r^2``. The solve finds the surface through a point without being told any of this:
it starts from a circle of points in the cross-section, not from a surface with its straight-field-line angle.
"""
import numpy as np

from mrx.diagnostics.torus import toroidal_flux, torus

#: The logical radii of the surfaces, away from the low-order rationals of li383 (iota 1/2 at r 0.55, 3/5 at 0.80).
RADII = np.array([0.2, 0.5, 0.9])
#: 1.25x ``max |iota - iota_file| / max |iota_file|`` on li383 (16, 32, 12) p=2 with a (33, 33) grid, measured
#: 1.00e-3 in float64 and float32 (2026-11-16). It is the discretisation error of the field, as in
#: ``test_poincare.py``. A missed factor nfp or a wrong sign would be off by O(1).
IOTA_BAND = 1.26e-3
#: 1.25x the largest distance of the three surfaces from ``r = const`` in logical radius, measured 1.47e-5 in
#: float64 and 1.49e-5 in float32.
RADIUS_BAND = 1.87e-5
#: 1.25x the largest relative error of ``flux(r) / flux(0.5) = (r / 0.5)^2``, measured 2.45e-4.
FLUX_BAND = 3.07e-4


def test_tori_are_the_logical_surfaces(seq, b0):
    """The surfaces through three points are ``r = const`` with the file's iota and a flux like ``r^2``."""
    tori = [torus(seq, b0, (r, 0.0)) for r in RADII]
    want = np.asarray(seq.equilibrium["profiles"]["iota"](RADII))
    err = float(np.abs(np.array([t.iota for t in tori]) - want).max() / np.abs(want).max())
    off = max(float(np.abs(np.hypot(*np.moveaxis(np.asarray(t.y), -1, 0)) - r).max()) for t, r in zip(tori, RADII))
    flux = np.array([toroidal_flux(seq, b0, t) for t in tori])
    ratio = float(np.abs(flux / flux[1] / (RADII / RADII[1]) ** 2 - 1.0).max())
    print(f"\n  Newton steps {[t.iterations for t in tori]}, iota error {err:.2e}, off the surface {off:.2e}, "
          f"flux ratio error {ratio:.2e}, residuals {[f'{t.residual:.1e}' for t in tori]}")
    assert all(t.iterations > 0 for t in tori)
    assert err < IOTA_BAND
    assert off < RADIUS_BAND
    assert ratio < FLUX_BAND
