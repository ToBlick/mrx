"""Gauss quadrature on the logical cube, and the tensor-product evaluate / integrate pair.

Composite Gauss rules per axis (one Gauss point for a constant basis),
their tensor product in :class:`QuadratureRule`, and the sum-factorised
:func:`evaluate_at_xq` / :func:`integrate_against` adjoint pair that every
load, cross product and quadrature-point evaluation in the package goes through.
"""

import numpy as np
import jax
import jax.numpy as jnp


class QuadratureRule:
    """Tensor-product Gauss quadrature on the logical cube of a 0-form basis.

    One 1-D rule per axis, chosen by :func:`select_quadrature` from the axis
    basis: composite ``p``-point Gauss on the knot spans of a clamped or
    periodic spline basis, a single Gauss point for a constant basis. The
    3-D rule is their tensor product, flattened **r-major** -- the flat index
    runs fastest over ``zeta``, then ``theta``, then ``r``, so a flat
    quadrature field is the ``(nx, ny, nz)`` array ``field.reshape(shape)``
    with axes ``(r, theta, zeta)``; that is what :func:`evaluate_at_xq`,
    :func:`integrate_against` and every element-layout reshape rely on.

    Attributes:
        x_x, x_y, x_z: 1-D quadrature points per axis.
        w_x, w_y, w_z: 1-D quadrature weights per axis.
        x: ``(n, 3)`` tensor-product points in the flat order above.
        w: ``(n,)`` tensor-product weights, the product of the axis weights.
        nx, ny, nz: points per axis; ``shape = (nx, ny, nz)``; ``n`` their product.
        ns: ``arange(n)``, the flat point index.
    """

    def __init__(self, form, p):
        """Build the rule for the axis bases of ``form`` with ``p`` Gauss points per span.

        Args:
            form: A :class:`~mrx.differential_forms.DifferentialForm` whose
                first component's axis bases select the 1-D rules.
            p: Number of Gauss points per knot span.
        """
        (x_x, w_x), (x_y, w_y), (x_z, w_z) = [
            select_quadrature(b, p) for b in form.bases[0].bases]
        n = w_x.size * w_y.size * w_z.size
        x_q = jnp.stack(jnp.meshgrid(x_x, x_y, x_z, indexing='ij'), axis=-1)
        w_q = w_x[:, None, None] * w_y[None, :, None] * w_z[None, None, :]

        self.x_x, self.x_y, self.x_z = x_x, x_y, x_z
        self.w_x, self.w_y, self.w_z = w_x, w_y, w_z
        self.x = x_q.reshape(n, 3)
        self.w = w_q.reshape(n)
        self.nx, self.ny, self.nz = x_x.size, x_y.size, x_z.size
        self.shape = (self.nx, self.ny, self.nz)
        self.n = n
        self.ns = jnp.arange(n)


def composite_quad(T, p):
    """Composite p-point Gauss quadrature over the intervals defined by knot vector T.

    Args:
        T: Knot vector (breakpoints), shape ``(n_intervals + 1,)``.
        p: Number of Gauss points per interval; exact for polynomials of degree ``<= 2p-1``.

    Returns:
        Tuple ``(x_q, w_q)`` of concatenated quadrature points and weights on ``[T[0], T[-1]]``.
    """
    xi, wi = np.polynomial.legendre.leggauss(p)
    xi = jnp.asarray(xi)
    wi = jnp.asarray(wi)

    def _rescale(a, b):
        return (xi + 1) / 2 * (b - a) + a, wi * (b - a) / 2

    x_q, w_q = jax.vmap(_rescale)(T[:-1], T[1:])
    return jnp.ravel(x_q), jnp.ravel(w_q)


def spectral_quad(p):
    """Single-interval p-point Gauss quadrature on ``[0, 1]``.

    Args:
        p: Number of Gauss points; exact for polynomials of degree ``<= 2p-1``.

    Returns:
        Tuple ``(x_q, w_q)`` of quadrature points and weights on ``[0, 1]``.
    """
    xi, wi = np.polynomial.legendre.leggauss(p)
    return jnp.asarray((xi + 1) / 2), jnp.asarray(wi / 2)


def select_quadrature(basis, n):
    """Select the appropriate quadrature rule for a given basis.

    Args:
        basis: A ``SplineBasis`` instance.
        n: Number of Gauss points per interval.

    Returns:
        Tuple ``(x_q, w_q)`` of quadrature points and weights.
    """
    if basis.type in ('clamped', 'periodic'):
        return composite_quad(basis.T[basis.p:-basis.p], n)
    elif basis.type == 'constant':
        return spectral_quad(1)


# ---------------------------------------------------------------------------
# Tensor-product evaluation / integration helpers
# ---------------------------------------------------------------------------

def evaluate_at_xq(dofs, comp_info, comp_shapes, quad_shape, d):
    """Evaluate a k-form at quadrature points using tensor-product structure.

    Parameters
    ----------
    dofs : array, shape (n_total,)
        Internal DOF vector (already contracted with extraction matrices).
    comp_info : list of (output_dim, R, T, Z)
        For each component ``c``: output dimension index and 1D basis arrays
        ``R`` (shape ``(s1_c, nq_r)``), ``T`` (shape ``(s2_c, nq_t)``),
        ``Z`` (shape ``(s3_c, nq_z)``).
    comp_shapes : list of tuples ``(s1_c, s2_c, s3_c)``
        DOF grid shape per component.
    quad_shape : tuple ``(nq_r, nq_t, nq_z)``
        ``seq.quad.shape``.
    d : int
        Number of output dimensions.

    Returns
    -------
    f_jk : array, shape ``(n_q, d)``
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


def integrate_against(f_jk, comp_info, comp_shapes, quad_shape):
    """Integrate quadrature-point values against a k-form basis.

    The adjoint of :func:`evaluate_at_xq` (transpose action).

    Parameters
    ----------
    f_jk : array, shape ``(n_q, d)``
        Values at quadrature points (already multiplied by quadrature weights).
    comp_info : list of ``(input_dim, R, T, Z)``
        Per-component input dimension and 1D basis arrays.
    comp_shapes : list of tuples ``(s1_c, s2_c, s3_c)``
    quad_shape : tuple ``(nq_r, nq_t, nq_z)``
        ``seq.quad.shape``.

    Returns
    -------
    result : array, shape ``(n_total,)``
    """
    d = f_jk.shape[1]
    f = f_jk.reshape(quad_shape + (d,)).transpose(3, 0, 1, 2)
    parts = []
    for c, (in_dim, R, T, Z) in enumerate(comp_info):
        val = jnp.einsum('ia,jb,kc,abc->ijk', R, T, Z, f[in_dim])
        parts.append(val.ravel())
    return jnp.concatenate(parts)
