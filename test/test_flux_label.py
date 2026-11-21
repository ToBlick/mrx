"""The flux label of a field with nested surfaces is a function of the flux.

The li383 fixture field is the VMEC equilibrium, tangent to the logical surfaces ``r = const`` with
``r = sqrt(s)``. So the anisotropic-diffusion temperature (:func:`mrx.flux_label.flux_label`) is constant on every
radial quadrature layer up to the anisotropy, peaks at the axis and falls to the wall, and the flux it encloses
(:func:`mrx.flux_label.enclosed_flux`) is ``r^2``.

The prescribed pressure takes the label as its radial coordinate, solved to a loose tolerance and warm-started from
the label of the field before. Neither may move it by more than a small multiple of that tolerance.
"""
import jax.numpy as jnp
import numpy as np

from mrx.flux_label import ANISOTROPY, enclosed_flux, enclosed_flux_map, flux_label, label_spline
from mrx.relaxation.seeding import energy_seed
from mrx.relaxation.shells import LABEL_TOL, label


def test_flux_label_is_a_function_of_the_flux(seq, b0):
    B_jk = seq.odd.evaluate_at_quadrature(b0, 2)
    T, info = flux_label(seq, B_jk, anisotropy=1e-5)
    assert int(info) > 0
    nx = int(seq.quad.shape[0])
    r = np.asarray(seq.quad.x_x)
    T_q = np.asarray(seq.even.evaluate_at_quadrature(T, 0)[:, 0]).reshape(nx, -1)
    # the angular spread on a layer over max T: 1.14e-3 at kperp = 1e-5, 1.27e-4 at 1e-6 on the fixture.
    # The discrete flux functions are exactly null for the parallel part (Bhat^r = 0), so there is no floor.
    assert (T_q.std(axis=1) / T_q.max()).max() < 1.43e-3
    assert np.all(np.diff(T_q.mean(axis=1)) < 0)
    # |s - r^2| over the volume, 7.0e-4 measured on the fixture (2.45e-3 with the sorted and binned label)
    s_q = np.asarray(enclosed_flux(seq, T, B_jk)).reshape(nx, -1)
    wJ = np.asarray(seq.quad.w * seq.jacobian_j).reshape(nx, -1)
    err = np.sum(wJ * np.abs(s_q - r[:, None] ** 2)) / np.sum(wJ)
    print(f"\n  mean |s - r^2| {err:.3e}")
    assert err < 9e-4
    # the label spline is 1 at the smallest coefficient of T and 0 at the largest, to round-off
    a, b, _ = label_spline(seq, T, B_jk)
    s_of = enclosed_flux_map(seq, T, B_jk)
    assert abs(float(s_of(a)) - 1.0) < 1e-13 and abs(float(s_of(b))) < 1e-13


def test_label_at_the_shells_tolerance(seq, b0):
    """The pressure target reads the label of :func:`mrx.relaxation.shells.label` at its tolerance ``LABEL_TOL`` and
    warm-starts it from the label of the field before. That label is the converged one (tolerance 1e-9) to a small
    multiple of ``LABEL_TOL``, solved cold or from the label of another field: here the fixture's field seeded with
    its island chains (``energy_seed``), whose label differs by 5.5e-3. Measured at the production anisotropy
    ``|s - s_ref|`` up to 9.4e-7 cold and 2.0e-6 warm in float64 (2026-11-21, the label spline), against the bound
    ``2 LABEL_TOL``. 14 s on four CPU cores after the fixture, 48 s in float32."""
    s_ref, _, info = label(seq, b0, ANISOTROPY, None, 1e-9)
    assert int(info) > 0
    s_cold, _, info_cold = label(seq, b0, ANISOTROPY)
    B1, _ = energy_seed(seq, b0, verbose=False)
    _, T1, _ = label(seq, B1, ANISOTROPY)
    s_warm, _, info_warm = label(seq, b0, ANISOTROPY, T1)
    cold, warm = float(jnp.max(jnp.abs(s_cold - s_ref))), float(jnp.max(jnp.abs(s_warm - s_ref)))
    print(f"\n  max |s - s_ref| cold {cold:.2e} (CG {int(info_cold)}), warm {warm:.2e} (CG {int(info_warm)})")
    assert int(info_cold) > 0 and int(info_warm) > 0
    assert cold < 2 * LABEL_TOL and warm < 2 * LABEL_TOL
