"""Spline spaces of differential k-forms on the logical cube, and how forms transform under the map.

- :class:`DifferentialForm` is the tensor-product spline basis of the k-forms, ``k = 0, 1, 2, 3``.
  It fixes the number of components, their coefficient grids and the number of coefficients.
- :class:`DiscreteFunction` turns a coefficient vector on such a basis into a function that can
  be evaluated at logical points.
- :class:`Pushforward` and :class:`Pullback` move a k-form between the logical cube and the
  physical domain under the map ``Phi``.
- :func:`det33`, :func:`adj33` and :func:`inv33` are the determinant, adjugate and inverse of a
  3x3 matrix in closed form.
"""

import math

import jax
import jax.numpy as jnp

from mrx.spline_bases import (
    DerivativeSpline,
    SplineBasis,
    TensorBasis,
    contract_local,
)

#: For each form degree, the axes on which each component uses the derivative splines.
DERIV_AXES = {0: ((),), 1: ((0,), (1,), (2,)), 2: ((1, 2), (0, 2), (0, 1)), 3: ((0, 1, 2),)}


class DifferentialForm:
    """The tensor-product spline basis of the k-forms on the logical cube.

    A 0-form has one component (a scalar), a 1-form and a 2-form have three (vector proxies),
    and a 3-form has one (a density). Each component is a tensor product of 1-D bases, using
    the splines Lambda on some axes and their derivative splines dLambda on the others, as listed
    in :data:`DERIV_AXES`.

    Args:
        k: the form degree, 0 to 3.
        ns, ps, types: the number of splines, the degree and ``'clamped'`` or ``'periodic'``
            for each axis.
        Ts: the knot vectors for each axis, or ``None`` for uniform knots.
    """

    def __init__(self, k, ns, ps, types, Ts=None):
        if k not in DERIV_AXES:
            raise ValueError("Degree k must be 0, 1, 2 or 3")
        self.k = k
        Ts = [None] * 3 if Ts is None else Ts
        self.Lambda = [SplineBasis(n, p, type, T) for n, p, type, T in zip(ns, ps, types, Ts)]
        self.dLambda = [DerivativeSpline(b) for b in self.Lambda]
        self.types = types
        self.nr, self.nt, self.nz = ns
        self.dr, self.dt, self.dz = (n - (t == "clamped") for n, t in zip(ns, types))
        n_axis = ((self.nr, self.nt, self.nz), (self.dr, self.dt, self.dz))
        self._deriv_axes = DERIV_AXES[k]
        self.bases = tuple(TensorBasis([(self.dLambda if a in d else self.Lambda)[a] for a in range(3)])
                           for d in self._deriv_axes)
        self.shape = tuple(tuple(n_axis[a in d][a] for a in range(3)) for d in self._deriv_axes)
        sizes = [math.prod(s) for s in self.shape] + [0, 0]
        self.n1, self.n2, self.n3 = sizes[:3]
        self.n = sum(sizes)

    def derivative_axes(self, c):
        """Return the axes on which component ``c`` uses the derivative splines."""
        return self._deriv_axes[c]

    def raw_blocks(self, raw):
        """Split a vector of spline coefficients (before the extraction of the sequence) into
        one coefficient grid per component, with the shapes in :attr:`shape`."""
        blocks, start = [], 0
        for shape in self.shape:
            size = math.prod(shape)
            blocks.append(raw[start:start + size].reshape(shape))
            start += size
        return tuple(blocks)

    def contract(self, blocks, x):
        """Return the value at the logical point ``x`` of the form with the coefficient grids
        ``blocks`` from :meth:`raw_blocks`. The shape is ``(1,)`` for ``k = 0, 3`` and ``(3,)``
        otherwise."""
        local = {}
        for basis in self.bases:
            for b, xi in zip(basis.bases, x):
                if id(b) not in local:
                    local[id(b)] = b.evaluate_local(xi)
        return jnp.stack([
            contract_local(block, tuple(local[id(b)] for b in basis.bases))
            for block, basis in zip(blocks, self.bases)
        ])


class DiscreteFunction:
    """A discrete form that can be evaluated at logical points.

    ``dof`` are the degrees of freedom on the form basis (a :class:`DifferentialForm`,
    the second argument). If they come from a de Rham
    sequence, pass its extraction ``E`` (for example ``seq.E(k)``), which maps them to the
    spline coefficients. With ``E=None`` the ``dof`` are the spline coefficients themselves."""

    def __init__(self, dof, Lambda, E=None):
        self.Lambda = Lambda
        self.raw = Lambda.raw_blocks(dof if E is None else E.T @ dof)

    def __call__(self, x):
        return self.Lambda.contract(self.raw, x)


class Pushforward:
    """The pushforward of the logical k-form ``f`` under the map ``Phi``.

    Calling it at a logical point ``x`` returns the physical form at ``Phi(x)``. With the
    Jacobian ``DPhi`` and ``J = det DPhi`` the rules are ``f`` for ``k = 0``, ``DPhi^-T f`` for
    ``k = 1``, ``DPhi f / J`` for ``k = 2`` and ``f / J`` for ``k = 3``.
    """

    def __init__(self, f, Phi, k):
        self.k = k
        self.f = f
        self.Phi = Phi

    def __call__(self, x):
        if self.k == 0:
            return self.f(x)
        DPhi = jax.jacfwd(self.Phi)(x)
        if self.k == 1:
            return inv33(DPhi).T @ self.f(x)
        if self.k == 2:
            return DPhi @ self.f(x) / det33(DPhi)
        return self.f(x) / det33(DPhi)


class Pullback:
    """The pullback of the physical k-form ``f`` under the map ``Phi``.

    Calling it at a logical point ``x`` returns the logical form there. With ``y = Phi(x)`` and
    ``J = det DPhi`` the rules are ``f(y)`` for ``k = 0``, ``DPhi^T f(y)`` for ``k = 1``,
    ``J DPhi^-1 f(y)`` for ``k = 2`` and ``J f(y)`` for ``k = 3``. The ``k = 2`` rule stays
    finite on the polar axis, where ``J = 0``.
    """

    def __init__(self, f, Phi, k):
        self.k = k
        self.f = f
        self.Phi = Phi

    def __call__(self, x):
        y = self.Phi(x)
        if self.k == 0:
            return self.f(y)
        DPhi = jax.jacfwd(self.Phi)(x)
        if self.k == 1:
            return DPhi.T @ self.f(y)
        if self.k == 2:
            return adj33(DPhi) @ self.f(y)      # J DPhi^-1 = adj(DPhi), finite where J = 0
        return self.f(y) * det33(DPhi)


def det33(mat: jnp.ndarray) -> jnp.ndarray:
    """Return the determinant of a 3x3 matrix."""
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    return m1 * (m5 * m9 - m6 * m8) - m2 * (m4 * m9 - m6 * m7) + m3 * (m4 * m8 - m5 * m7)


def adj33(mat: jnp.ndarray) -> jnp.ndarray:
    """Return the adjugate ``det(A) A^-1`` of a 3x3 matrix. Unlike the inverse it is finite
    where ``det A = 0``, as on the polar axis."""
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    return jnp.array([
        [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
        [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
        [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
    ])


def inv33(mat: jnp.ndarray) -> jnp.ndarray:
    """Return the inverse of a 3x3 matrix, ``adj33(A) / det33(A)``."""
    return adj33(mat) / det33(mat)
