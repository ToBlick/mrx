"""Gauss quadrature on the logical unit cube, and evaluation of spline forms at its points.

- :class:`QuadratureRule` is the tensor-product Gauss rule on which all integrals of a de Rham
  sequence are computed. It is built once per sequence from the knots and does not depend on the
  geometry.
- :func:`evaluate_at_xq` evaluates a form from its spline coefficients at all quadrature points,
  and :func:`integrate_against` does the transpose: it integrates given quadrature values against
  every basis function. Most users reach these through
  :class:`~mrx.derham_sequence.DeRhamSequence` rather than directly.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mrx.pytree import register_arrays


@register_arrays
class QuadratureRule:
    """Tensor-product Gauss quadrature on the logical unit cube ``(r, theta, zeta)``.

    Each axis uses ``p`` Gauss points on every knot span of the spline basis. The flattened
    points are ordered r-major (``zeta`` varies fastest, then ``theta``, then ``r``), so a field
    given at the flat points reshapes to the ``(nx, ny, nz)`` grid with ``field.reshape(shape)``.

    With ``half_zeta=True`` the rule covers only ``zeta`` in ``[0, 1/2]``, with doubled weights.
    It is meant for stellarator-symmetric integrands, whose two half periods contribute
    equally (see :mod:`mrx.symmetry`), and needs ``zeta = 1/2`` to be a knot.

    Attributes:
        x_x, x_y, x_z: the 1-D points per axis.
        w_x, w_y, w_z: the 1-D weights per axis.
        x, w: the ``(n, 3)`` points and ``(n,)`` weights of the tensor-product rule.
        nx, ny, nz: the number of points per axis, and ``shape = (nx, ny, nz)``.
        ne_x, ne_y, ne_z: the number of knot spans covered per axis, counted from the start of
            the axis.
    """

    def __init__(self, form, p, half_zeta=False):
        """Build the rule on the knot spans of ``form`` (a
        :class:`~mrx.differential_forms.DifferentialForm`, usually the 0-forms) with ``p`` Gauss
        points per span."""
        spans = [b.T[b.p:-b.p] for b in form.bases[0].bases]
        if half_zeta:
            T_z = np.asarray(spans[2])
            fold = np.flatnonzero(np.abs(T_z - 0.5) < 1e-12)
            if fold.size != 1:
                raise ValueError("a half-period quadrature needs zeta = 1/2 as a knot: an even "
                                 "number of uniform zeta cells, or breakpoints containing 1/2")
            spans[2] = jnp.asarray(T_z[:int(fold[0]) + 1])
        (x_x, w_x), (x_y, w_y), (x_z, w_z) = [composite_quad(T, p) for T in spans]
        if half_zeta:
            w_z = 2.0 * w_z
        self.ne_x, self.ne_y, self.ne_z = (int(T.size) - 1 for T in spans)
        n = w_x.size * w_y.size * w_z.size
        x_q = jnp.stack(jnp.meshgrid(x_x, x_y, x_z, indexing='ij'), axis=-1)
        w_q = w_x[:, None, None] * w_y[None, :, None] * w_z[None, None, :]

        self.x_x, self.x_y, self.x_z = x_x, x_y, x_z
        self.w_x, self.w_y, self.w_z = w_x, w_y, w_z
        self.x = x_q.reshape(n, 3)
        self.w = w_q.reshape(n)
        self.nx, self.ny, self.nz = x_x.size, x_y.size, x_z.size
        self.shape = (self.nx, self.ny, self.nz)


def composite_quad(T, p):
    """Return the points and weights ``(x_q, w_q)`` of ``p``-point Gauss on each interval between
    consecutive breakpoints ``T``. The rule is exact for piecewise polynomials of degree
    ``2p - 1``."""
    xi, wi = np.polynomial.legendre.leggauss(p)
    xi = jnp.asarray(xi)
    wi = jnp.asarray(wi)

    def _rescale(a, b):
        return (xi + 1) / 2 * (b - a) + a, wi * (b - a) / 2

    x_q, w_q = jax.vmap(_rescale)(T[:-1], T[1:])
    return jnp.ravel(x_q), jnp.ravel(w_q)


def evaluate_at_xq(dofs, comp_info, comp_shapes, quad_shape, d):
    """Return the values, shape ``(n_q, d)``, of a form at all quadrature points.

    ``dofs`` are the spline coefficients of all components before the extraction ``E`` of the
    sequence (which handles the polar axis and the boundary conditions), that is ``E.T @ dofs``
    for DoFs of the sequence. ``comp_info`` holds for each component the
    output index it adds to and its 1-D basis values at the quadrature points of each axis.
    ``comp_shapes`` are the coefficient grid shapes of the components, ``quad_shape`` is
    ``seq.quad.shape`` and ``d`` the number of output components.
    """
    f = jnp.zeros((d,) + quad_shape, dtype=dofs.dtype)
    offset = 0
    for c, (out_dim, R, T, Z) in enumerate(comp_info):
        s = comp_shapes[c]
        n_c = s[0] * s[1] * s[2]
        V = dofs[offset:offset + n_c].reshape(s)
        val = jnp.einsum('ijk,ia,jb,kc->abc', V, R, T, Z)
        f = f.at[out_dim].add(val)
        offset += n_c
    return f.transpose(1, 2, 3, 0).reshape(-1, d)


def integrate_against(f_jk, comp_info):
    """Return the integrals of the quadrature values ``f_jk``, shape ``(n_q, d)``, against every
    basis function of ``comp_info``. This is the transpose of :func:`evaluate_at_xq`.

    ``f_jk`` must already include the quadrature weights. The result belongs to the spline
    coefficients before the extraction, so a caller applies ``E`` to it."""
    quad_shape = tuple(int(tab.shape[1]) for tab in comp_info[0][1:])
    d = f_jk.shape[1]
    f = f_jk.reshape(quad_shape + (d,)).transpose(3, 0, 1, 2)
    parts = []
    for in_dim, R, T, Z in comp_info:
        val = jnp.einsum('ia,jb,kc,abc->ijk', R, T, Z, f[in_dim])
        parts.append(val.ravel())
    return jnp.concatenate(parts)
