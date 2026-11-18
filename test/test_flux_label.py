"""The flux label of a field with nested surfaces is a function of the flux.

The li383 fixture field is the VMEC equilibrium, tangent to the logical surfaces ``r = const`` with
``r = sqrt(s)``. So the anisotropic-diffusion temperature (:func:`mrx.flux_label.flux_label`) is constant on every
radial quadrature layer up to the anisotropy, peaks at the axis and falls to the wall, and the flux it encloses
(:func:`mrx.flux_label.enclosed_flux`) is ``r^2``.
"""
import numpy as np

from mrx.flux_label import enclosed_flux, enclosed_flux_map, flux_label, label_axis


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
    assert np.allclose(np.asarray(label_axis(seq, T))[:, 0], r[0])
    # |s - r^2| over the volume, 2.45e-3 measured on the fixture: the label resolves one radial layer
    s_q = np.asarray(enclosed_flux(seq, T, B_jk)).reshape(nx, -1)
    wJ = np.asarray(seq.quad.w * seq.jacobian_j).reshape(nx, -1)
    assert np.sum(wJ * np.abs(s_q - r[:, None] ** 2)) / np.sum(wJ) < 3.1e-3
    assert float(enclosed_flux_map(seq, T, B_jk)(0.0)) == 1.0
