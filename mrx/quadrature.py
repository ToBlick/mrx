"""
Quadrature rules for numerical integration in finite element analysis.

This module provides:
- Composite Gauss quadrature for clamped and periodic bases
- Spectral quadrature for constant bases
"""

import numpy as np
import jax
import jax.numpy as jnp


class QuadratureRule:
    """
    A class for handling quadrature rules in finite element analysis.

    This class implements various quadrature rules for numerical integration
    in three-dimensional space. It supports different types of basis functions
    and provides efficient computation of quadrature points and weights.

    Attributes:
        x_x (array): Quadrature points in x-direction
        x_y (array): Quadrature points in y-direction
        x_z (array): Quadrature points in z-direction
        w_x (array): Quadrature weights in x-direction
        w_y (array): Quadrature weights in y-direction
        w_z (array): Quadrature weights in z-direction
        x (array): Combined quadrature points in 3D space
        w (array): Combined quadrature weights
    """

    def __init__(self, form, p):
        """
        Initialize the quadrature rule.

        Args:
            form: The differential form defining the basis functions
            p (int): Number of quadrature points per direction
        """
        # Select appropriate quadrature rules for each direction
        (x_x, w_x), (x_y, w_y), (x_z, w_z) = [
            select_quadrature(b, p) for b in form.bases[0].bases]

        # Combine quadrature points and weights in 3D
        x_s = [x_x, x_y, x_z]
        w_s = [w_x, w_y, w_z]
        d = 3
        n = w_x.size * w_y.size * w_z.size

        # Create 3D grid of quadrature points and weights
        x_q = jnp.array(jnp.meshgrid(*x_s))  # shape d, n1, n2, n3, ...
        x_q = x_q.transpose(*range(1, d+1), 0).reshape(n, d)
        w_q = jnp.array(
            jnp.meshgrid(*w_s)).transpose(*range(1, d+1), 0).reshape(n, d)
        w_q = jnp.prod(w_q, 1)

        # Store quadrature points and weights
        self.x_x = x_x
        self.x_y = x_y
        self.x_z = x_z
        self.w_x = w_x
        self.w_y = w_y
        self.w_z = w_z
        self.x = x_q
        self.w = w_q
        self.nx = x_x.size
        self.ny = x_y.size
        self.nz = x_z.size
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
    quad_shape : tuple ``(nq_t, nq_r, nq_z)``
    d : int
        Number of output dimensions.

    Returns
    -------
    f_jk : array, shape ``(n_q, d)``
    """
    f = jnp.zeros((d,) + quad_shape)
    offset = 0
    for c, (out_dim, R, T, Z) in enumerate(comp_info):
        s = comp_shapes[c]
        n_c = s[0] * s[1] * s[2]
        V = dofs[offset:offset + n_c].reshape(s)
        # V[i,j,k], R[i,a], T[j,b], Z[k,c] -> f[b,a,c]  (quad_shape = nq_t, nq_r, nq_z)
        val = jnp.einsum('ijk,ia,jb,kc->bac', V, R, T, Z)
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
    quad_shape : tuple ``(nq_t, nq_r, nq_z)``

    Returns
    -------
    result : array, shape ``(n_total,)``
    """
    d = f_jk.shape[1]
    # Reshape to (d, nq_t, nq_r, nq_z)
    f = f_jk.reshape(quad_shape + (d,)).transpose(3, 0, 1, 2)
    parts = []
    for c, (in_dim, R, T, Z) in enumerate(comp_info):
        val = jnp.einsum('ia,jb,kc,bac->ijk', R, T, Z, f[in_dim])
        parts.append(val.ravel())
    return jnp.concatenate(parts)
