"""The flux label of a field with nested surfaces is its normalised toroidal flux.

The projected exact field of shearedE is tangent to the logical surfaces ``r = const`` of the wout, Landreman's
surfaces ``rho = const``. So the anisotropic-diffusion temperature (:func:`mrx.flux_label.flux_label`) is constant on
every radial quadrature layer up to the anisotropy and falls from the axis to the wall, and the flux it encloses
(:func:`mrx.flux_label.enclosed_flux`) is the exact normalised toroidal flux ``s`` to the discretisation.

The prescribed pressure takes the label as its radial coordinate, solved to a loose tolerance and warm-started from
the label of the field before. Neither may move it by more than a small multiple of that tolerance.
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.flux_label import ANISOTROPY, enclosed_flux_map, flux_label, label_spline
from mrx.precision import eps
from mrx.relaxation.shells import LABEL_TOL, label

#: The angular spread of ``T`` on a radial layer over its maximum at ``kperp = 1e-5``, and the volume rms of the label
#: against the exact ``s`` at the production anisotropy and 1e-9: PENDING measured.
SPREAD, S_RMS = 2e-3, 1e-3


def test_flux_label_is_the_toroidal_flux(eq, seq, b0):
    """``T`` is constant on the radial layers up to the anisotropy and falls to the wall, the label against the exact
    ``s`` of shearedE at the quadrature points is within :data:`S_RMS` in the volume rms, and the label spline is 1
    and 0 at the extreme coefficients of ``T`` to round-off."""
    B_jk = seq.odd.evaluate_at_quadrature(b0, 2)
    T, info = flux_label(seq, B_jk, anisotropy=1e-5)
    assert int(info) > 0
    nx = int(seq.quad.shape[0])
    T_q = np.asarray(seq.even.evaluate_at_quadrature(T, 0)[:, 0]).reshape(nx, -1)
    spread = float((T_q.std(axis=1) / T_q.max()).max())
    s_q, _, info = label(seq, b0, ANISOTROPY, None, 1e-9)
    X = np.asarray(jax.vmap(seq.map)(seq.quad.x))
    w = np.asarray(seq.quad.w * seq.jacobian_j)
    rms = float(np.sqrt(np.sum(w * (np.asarray(s_q) - eq.s_at(X)) ** 2) / np.sum(w)))
    a, b, _ = label_spline(seq, T, B_jk)
    s_of = enclosed_flux_map(seq, T, B_jk)
    print(f"\n  angular spread of T {spread:.2e}, label against the exact s {rms:.3e} (rms)")
    assert spread < SPREAD
    assert np.all(np.diff(T_q.mean(axis=1)) < 0)
    assert int(info) > 0 and rms < S_RMS
    # the label spline is 1 at the smallest coefficient of T and 0 at the largest, to round-off (2 eps in float32)
    assert abs(float(s_of(a)) - 1.0) < eps(10) and abs(float(s_of(b))) < eps(10)


def test_label_at_the_shells_tolerance(seq, b0):
    """The pressure target reads the label of :func:`mrx.relaxation.shells.label` at its tolerance ``LABEL_TOL`` and
    warm-starts it from the label of the field before. That label is the converged one (tolerance 1e-9) to a small
    multiple of ``LABEL_TOL``, solved cold or from the label of another field: here the field plus the curl of a
    smooth potential, 1e-2 of its norm. Bound ``2 LABEL_TOL``."""
    s_ref, _, info = label(seq, b0, ANISOTROPY, None, 1e-9)
    assert int(info) > 0
    s_cold, _, info_cold = label(seq, b0, ANISOTROPY)
    odd = seq.odd

    def A(x):
        t = 2 * jnp.pi * x[1]
        return jnp.array([0.0, 0.0, x[0] * (1.0 - x[0] ** 2) * jnp.cos(t - 2 * jnp.pi * x[2])])
    curl = odd.G[1] @ odd.interpolate(jax.jit(A), 1)
    B1 = b0 + 1e-2 * curl * odd.l2_norm(b0, 2) / odd.l2_norm(curl, 2)
    _, T1, _ = label(seq, B1, ANISOTROPY)
    s_warm, _, info_warm = label(seq, b0, ANISOTROPY, T1)
    cold, warm = float(jnp.max(jnp.abs(s_cold - s_ref))), float(jnp.max(jnp.abs(s_warm - s_ref)))
    print(f"\n  max |s - s_ref| cold {cold:.2e} (CG {int(info_cold)}), warm {warm:.2e} (CG {int(info_warm)})")
    assert int(info_cold) > 0 and int(info_warm) > 0
    assert cold < 2 * LABEL_TOL and warm < 2 * LABEL_TOL
