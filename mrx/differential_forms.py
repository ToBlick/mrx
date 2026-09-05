"""Discrete differential k-forms on tensor-product spline spaces.

Provides :class:`DifferentialForm` (basis), :class:`DiscreteFunction`
(DOF vector + basis), :class:`Pushforward`, and :class:`Pullback`.
"""

import math
from typing import Callable

import jax
import jax.numpy as jnp

from mrx.spline_bases import (
    DerivativeSpline,
    SplineBasis,
    TensorBasis,
    contract_local,
)


class DifferentialForm:
    """Discrete k-form on a 3-D tensor-product spline space.

    ``k=0`` scalar (nodes); ``k=1`` 1-form (edges); ``k=2`` 2-form (faces);
    ``k=3`` volume form (cells). The three components of a 1- or 2-form are
    tensor products of the primal basis ``Λ`` and the derivative basis
    ``dΛ`` in the pattern of the de Rham complex.
    """
    d: int
    k: int
    n: int
    nr: int
    nt: int
    nz: int
    pr: int
    pt: int
    pz: int
    ns: jnp.ndarray

    def __init__(self, k, ns, ps, types, Ts=None):
        """Args:
            k: Form degree (0, 1, 2 or 3).
            ns: Number of DOFs in each direction.
            ps: Polynomial degrees in each direction.
            types: Boundary condition types (``'clamped'``, ``'periodic'``,
                ``'constant'``) for each direction.
            Ts: Knot vectors; ``None`` uses uniform knots.
        """
        self.d = len(ns)
        self.k = k
        if Ts is None:
            Ts = [None] * self.d
        self.Λ = [
            SplineBasis(n, p, type, T) for n, p, type, T in zip(ns, ps, types, Ts)
        ]
        self.dΛ = [DerivativeSpline(b) for b in self.Λ]
        self.types = types

        self.pr, self.pt, self.pz = ps
        self.nr, self.nt, self.nz = ns
        if types[0] == "clamped":
            self.dr = self.nr - 1
        else:
            self.dr = self.nr
        if types[1] == "clamped":
            self.dt = self.nt - 1
        else:
            self.dt = self.nt
        if types[2] == "clamped":
            self.dz = self.nz - 1
        else:
            self.dz = self.nz

        if k == 0:
            self.bases = (TensorBasis(self.Λ),)
            self.shape = ((self.nr, self.nt, self.nz),)
            self.n1 = self.nr * self.nt * self.nz
            self.n2 = 0
            self.n3 = 0
        elif k == 1:
            self.bases = (
                TensorBasis([self.dΛ[0], self.Λ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.dΛ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.Λ[1], self.dΛ[2]]),
            )
            self.shape = (
                (self.dr, self.nt, self.nz),
                (self.nr, self.dt, self.nz),
                (self.nr, self.nt, self.dz),
            )
            self.n1 = self.dr * self.nt * self.nz
            self.n2 = self.nr * self.dt * self.nz
            self.n3 = self.nr * self.nt * self.dz
        elif k == 2:
            self.bases = (
                TensorBasis([self.Λ[0], self.dΛ[1], self.dΛ[2]]),
                TensorBasis([self.dΛ[0], self.Λ[1], self.dΛ[2]]),
                TensorBasis([self.dΛ[0], self.dΛ[1], self.Λ[2]]),
            )
            self.shape = (
                (self.nr, self.dt, self.dz),
                (self.dr, self.nt, self.dz),
                (self.dr, self.dt, self.nz),
            )
            self.n1 = self.nr * self.dt * self.dz
            self.n2 = self.dr * self.nt * self.dz
            self.n3 = self.dr * self.dt * self.nz
        elif k == 3:
            self.bases = (TensorBasis(self.dΛ),)
            self.shape = ((self.dr, self.dt, self.dz),)
            self.n1 = self.dr * self.dt * self.dz
            self.n2 = 0
            self.n3 = 0
        else:
            raise ValueError("Degree k must be 0, 1, 2 or 3")
        self.n = self.n1 + self.n2 + self.n3
        self.ns = jnp.arange(self.n)

    def raw_blocks(self, raw):
        """Split a raw (pre-extraction) coefficient vector into one tensor per
        component, shaped as :attr:`shape`: components in order, each raveled in
        C order."""
        blocks, start = [], 0
        for shape in self.shape:
            size = math.prod(shape)
            blocks.append(raw[start:start + size].reshape(shape))
            start += size
        return tuple(blocks)

    def contract(self, blocks, x):
        """Value at logical point ``x`` of the form with raw coefficient
        ``blocks`` (from :meth:`raw_blocks`): shape ``(1,)`` for ``k = 0, 3``,
        ``(3,)`` otherwise.

        Each 1-D basis (``Λ[d]`` or ``dΛ[d]``) is evaluated once at ``x[d]``
        on its ``p + 1`` nonzero functions, and each component contracts its
        own ``prod(p_d + 1)`` coefficient window -- ``O(p^3)`` per point.
        """
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
    """A discrete function as a linear combination of k-form basis functions.

    The extraction is folded into the coefficients once at construction
    (``raw = E^T dof``); evaluation then touches only the basis functions
    that are nonzero at the point.
    """

    def __init__(self, dof, Λ, E=None):
        """Args:
            dof: Coefficient vector (DOFs).
            Λ: Underlying :class:`DifferentialForm`.
            E: Extraction matrix; ``None`` is the identity.
        """
        self.dof = dof
        self.Λ = Λ
        self.E = E
        self.raw = Λ.raw_blocks(dof if E is None else E.T @ dof)

    def __call__(self, x):
        """Evaluate at logical point ``x``."""
        return self.Λ.contract(self.raw, x)


class Pushforward:
    """Pushforward of a k-form under the logical-to-physical map F.

    Let J = det(DF).  Transformation rules (ω evaluated at x):

        k= 0   F_* ω = ω
        k= 1   F_* ω = (DFᵀ)⁻¹ · ω
        k= 2   F_* ω = DF · ω / J           
        k= 3   F_* ω = ω / J
    """

    def __init__(self, f, F, k):
        """Args:
            f: The form to push forward.
            F: Logical-to-physical map.
            k: Form degree.
        """
        self.k = k
        self.f = f
        self.F = F

    def __call__(self, x):
        """Evaluate the pushed-forward form at logical point ``x``."""
        if self.k == 0:
            return self.f(x)
        DF = jax.jacfwd(self.F)(x)
        if self.k == 1:
            return inv33(DF).T @ self.f(x)
        elif self.k == 2:
            return DF @ self.f(x) / det33(DF)
        elif self.k == 3:
            return self.f(x) / det33(DF)
        raise ValueError("k must be 0, 1, 2 or 3")


class Pullback:
    """Pullback of a k-form under the logical-to-physical map F.

    Let J = det(DF).  Transformation rules (ω evaluated at F(x)):

        k= 0   F* ω = ω∘F
        k= 1   F* ω = DFᵀ · (ω∘F)
        k= 2   F* ω = J · DF⁻¹ · (ω∘F)
        k= 3   F* ω = J · (ω∘F)
    """

    def __init__(self, f, F, k):
        """Args:
            f: The form to pull back.
            F: Logical-to-physical map.
            k: Form degree.
        """
        self.k = k
        self.f = f
        self.F = F

    def __call__(self, x):
        """Evaluate the pulled-back form at logical point ``x``."""
        y = self.F(x)
        if self.k == 0:
            return self.f(y)
        DF = jax.jacfwd(self.F)(x)
        if self.k == 1:
            return DF.T @ self.f(y)
        elif self.k == 2:
            return adj33(DF) @ self.f(y)      # J DF^-1 = adj(DF), finite at det DF = 0
        elif self.k == 3:
            return self.f(y) * det33(DF)
        raise ValueError("k must be 0, 1, 2 or 3")


# ---------------------------------------------------------------------------
# Math utility functions (geometry, calculus, norms)
# ---------------------------------------------------------------------------

def det33(mat: jnp.ndarray) -> jnp.ndarray:
    """Determinant of a 3×3 matrix via the explicit Sarrus rule."""
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    return m1 * (m5 * m9 - m6 * m8) - m2 * (m4 * m9 - m6 * m7) + m3 * (m4 * m8 - m5 * m7)


def adj33(mat: jnp.ndarray) -> jnp.ndarray:
    """Adjugate (transposed cofactor matrix) of a 3×3 matrix.

    ``adj(A) = det(A) A^-1``, but computed directly from the cofactors, so it
    stays FINITE where ``det(A) -> 0``.  That matters on the polar axis, where
    ``det DF = 0`` exactly at the clamped radial Greville point: the primal
    2-form pullback ``omega = J DF^-1 v = adj(DF) v`` is well defined there,
    while ``det33(mat) * inv33(mat)`` would evaluate ``0 * inf = nan``.
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    return jnp.array([
        [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
        [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
        [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
    ])


def inv33(mat: jnp.ndarray) -> jnp.ndarray:
    """Inverse of a 3×3 matrix via the explicit adjugate formula.
    """
    return adj33(mat) / det33(mat)


def jacobian_determinant(f: Callable) -> Callable:
    """Return a function that computes ``det(jacfwd(f))`` at a point."""
    return lambda x: det33(jax.jacfwd(f)(x))
