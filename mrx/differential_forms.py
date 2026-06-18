"""Discrete differential k-forms on tensor-product spline spaces.

Provides :class:`DifferentialForm` (basis), :class:`DiscreteFunction`
(DOF vector + basis), :class:`Pushforward`, and :class:`Pullback`.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp

import mrx
from mrx.spline_bases import DerivativeSpline, SplineBasis, TensorBasis


class DifferentialForm:
    """Discrete k-form on a 3-D tensor-product spline space.

    ``k=0`` — scalar; ``k=1`` — 1-form (edge); ``k=2`` — 2-form (face);
    ``k=3`` — volume form; ``k=-1`` — vector field (3 copies of the
    0-form space).  **Note:** ``k=-1`` is incompatible with polar setups
    because the polar extraction operator reduces the 0-form DOF count
    asymmetrically across the three components.
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
            k: Form degree (0, 1, 2, 3, or -1 for a vector field).
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

        self.vecs = jnp.eye(self.d)

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
        elif k == -1:
            self.bases = (
                TensorBasis([self.Λ[0], self.Λ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.Λ[1], self.Λ[2]]),
                TensorBasis([self.Λ[0], self.Λ[1], self.Λ[2]]),
            )
            self.shape = (
                (self.nr, self.nt, self.nz),
                (self.nr, self.nt, self.nz),
                (self.nr, self.nt, self.nz),
            )
            self.n1 = self.nr * self.nt * self.nz
            self.n2 = self.nr * self.nt * self.nz
            self.n3 = self.nr * self.nt * self.nz
        else:
            raise ValueError(
                "Degree k must be 0, 1, 2, 3, or -1 (vector field)")
        self.n = self.n1 + self.n2 + self.n3
        self.ns = jnp.arange(self.n)

    def _vector_index(self, idx):
        """Return ``(component, local_index)`` for a global DOF index."""
        if self.k == 0 or self.k == 3:
            return jnp.int32(0), idx
        elif self.k == 1 or self.k == 2 or self.k == -1:
            n1, n2 = self.n1, self.n2
            category = jnp.int32(idx >= n1) + jnp.int32(idx >= n1 + n2)
            index = jnp.int32(idx - n1 * (idx >= n1) - n2 * (idx >= n1 + n2))
            return category, index

    def _ravel_index(self, c, i, j, k):
        """Return the global DOF index for component ``c`` and grid indices ``(i,j,k)``."""
        if self.k == 0:
            rav = jnp.ravel_multi_index(
                (i, j, k), (self.nr, self.nt, self.nz), mode="clip"
            )
        elif self.k == 1:
            n1, n2 = self.n1, self.n2
            rav = jnp.where(
                c == 0,
                jnp.ravel_multi_index(
                    (i, j, k), (self.dr, self.nt, self.nz), mode="clip"
                ),
                jnp.where(
                    c == 1,
                    n1
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.nr, self.dt, self.nz), mode="clip"
                    ),
                    n1
                    + n2
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.nr, self.nt, self.dz), mode="clip"
                    ),
                ),
            )
        elif self.k == 2:
            n1, n2 = self.n1, self.n2
            rav = jnp.where(
                c == 0,
                jnp.ravel_multi_index(
                    (i, j, k), (self.nr, self.dt, self.dz), mode="clip"
                ),
                jnp.where(
                    c == 1,
                    n1
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.dr, self.nt, self.dz), mode="clip"
                    ),
                    n1
                    + n2
                    + jnp.ravel_multi_index(
                        (i, j, k), (self.dr, self.dt, self.nz), mode="clip"
                    ),
                ),
            )
        elif self.k == 3:
            rav = jnp.ravel_multi_index(
                (i, j, k), (self.dr, self.dt, self.dz), mode="clip"
            )
        elif self.k == -1:
            n1, n2 = self.n1, self.n2
            _rav = jnp.ravel_multi_index(
                (i, j, k), (self.nr, self.nt, self.nz), mode="clip"
            )
            rav = jnp.where(
                c == 0,
                _rav,
                jnp.where(
                    c == 1,
                    n1 + _rav,
                    n1 + n2 + _rav,
                ),
            )
        return jnp.int32(rav)

    def _unravel_index(self, idx):
        """Return ``(component, i, j, k)`` for a global DOF index."""
        if self.k == 0:
            return jnp.int32(0), *jnp.unravel_index(idx, (self.nr, self.nt, self.nz))
        elif self.k == 1:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.dr, self.nt, self.nz))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(
                        ijk, (self.nr, self.dt, self.nz))),
                    jnp.array(jnp.unravel_index(
                        ijk, (self.nr, self.nt, self.dz))),
                ),
            )
            return c, i, j, k
        elif self.k == 2:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.nr, self.dt, self.dz))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(
                        ijk, (self.dr, self.nt, self.dz))),
                    jnp.array(jnp.unravel_index(
                        ijk, (self.dr, self.dt, self.nz))),
                ),
            )
            return c, i, j, k
        elif self.k == 3:
            return jnp.int32(0), *jnp.unravel_index(idx, (self.dr, self.dt, self.dz))
        elif self.k == -1:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.array(jnp.unravel_index(
                ijk, (self.nr, self.nt, self.nz)))
            return c, i, j, k

    def __call__(self, x, i):
        """Alias for :meth:`evaluate`."""
        return self.evaluate(x, i)

    def __getitem__(self, i):
        """Return ``lambda x: self(x, i)``."""
        return lambda x: self.evaluate(x, i)

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def __len__(self):
        return self.n

    def evaluate(self, x, i):
        """Evaluate basis function ``i`` at logical point ``x``."""
        category, index = self._vector_index(i)
        if self.k == 0 or self.k == 3:
            return jnp.ones(1) * self.bases[0](x, index)
        elif self.k == 1 or self.k == 2 or self.k == -1:
            e = jnp.zeros(3).at[category].set(1)
            val = jnp.where(
                category == 0,
                self.bases[0](x, index),
                jnp.where(
                    category == 1, self.bases[1](
                        x, index), self.bases[2](x, index)
                ),
            )
            return e * val


class DiscreteFunction:
    """A discrete function as a linear combination of k-form basis functions."""

    def __init__(self, dof, Λ, E=None):
        """Args:
            dof: Coefficient vector (DOFs).
            Λ: Underlying :class:`DifferentialForm`.
            E: Extraction matrix; defaults to the identity.
        """
        self.dof = dof
        self.Λ = Λ
        self.n = Λ.n
        self.ns = jnp.arange(self.n)
        self.E = E if E is not None else jnp.eye(self.n)

    def __call__(self, x):
        """Evaluate at logical point ``x``."""
        return self.dof @ (self.E @ jax.vmap(self.Λ, (None, 0))(x, self.ns))


class Pushforward:
    """Pushforward of a k-form under the logical-to-physical map F.

    Let J = det(DF).  Transformation rules (ω evaluated at x):

        k= 0   F_* ω = ω
        k= 1   F_* ω = (DFᵀ)⁻¹ · ω
        k= 2   F_* ω = DF · ω / J           (Piola)
        k= 3   F_* ω = ω / J
        k=−1   F_* v = DF · v               (vector field)
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
        elif self.k == 1:
            return inv33(jax.jacfwd(self.F)(x)).T @ self.f(x)
        elif self.k == 2:
            return (
                jax.jacfwd(self.F)(x)
                @ self.f(x)
                / jnp.linalg.det(jax.jacfwd(self.F)(x))
            )
        elif self.k == 3:
            return self.f(x) / jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == -1:
            return (
                jax.jacfwd(self.F)(x)
                @ self.f(x)
            )


class Pullback:
    """Pullback of a k-form under the logical-to-physical map F.

    Let J = det(DF).  Transformation rules (ω evaluated at F(x)):

        k= 0   F* ω = ω∘F
        k= 1   F* ω = DFᵀ · (ω∘F)
        k= 2   F* ω = J · DF⁻¹ · (ω∘F)     (Piola)
        k= 3   F* ω = J · (ω∘F)
        k=−1   F* v = DF⁻¹ · (v∘F)         (vector field)
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
        elif self.k == 1:
            return jax.jacfwd(self.F)(x).T @ self.f(y)
        elif self.k == 2:
            return (
                inv33(jax.jacfwd(self.F)(x))
                @ self.f(y)
                * jnp.linalg.det(jax.jacfwd(self.F)(x))
            )
        elif self.k == 3:
            return self.f(y) * jnp.linalg.det(jax.jacfwd(self.F)(x))
        elif self.k == -1:
            return (
                inv33(jax.jacfwd(self.F)(x)) @ self.f(y)
            )


# ---------------------------------------------------------------------------
# Math utility functions (geometry, calculus, norms)
# ---------------------------------------------------------------------------

def det33(mat: jnp.ndarray) -> jnp.ndarray:
    """Determinant of a 3×3 matrix via the explicit Sarrus rule."""
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    return m1 * (m5 * m9 - m6 * m8) - m2 * (m4 * m9 - m6 * m7) + m3 * (m4 * m8 - m5 * m7)


def inv33(mat: jnp.ndarray) -> jnp.ndarray:
    """Inverse of a 3×3 matrix via the explicit adjugate formula.
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = (m1 * (m5 * m9 - m6 * m8)
           + m4 * (m8 * m3 - m2 * m9)
           + m7 * (m2 * m6 - m3 * m5))
    return jnp.array([
        [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
        [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
        [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
    ]) / det


def safe_inv33(mat: jnp.ndarray, *, tol: float = 1e-10) -> jnp.ndarray:
    """Return ``inv33(mat)`` when well-conditioned, else the zero matrix.

    This is the singular-safe variant for modal block solves and other places
    where nullspace modes should be deflated instead of inverted.
    """
    det = det33(mat)

    def _singular(_):
        return jnp.zeros((3, 3), dtype=mat.dtype)

    def _nonsingular(_):
        return inv33(mat)

    return jax.lax.cond(jnp.abs(det) < tol, _singular, _nonsingular, operand=None)


def jacobian_determinant(f: Callable) -> Callable:
    """Return a function that computes ``det(jacfwd(f))`` at a point."""
    return lambda x: jnp.linalg.det(jax.jacfwd(f)(x))


def div(F: Callable) -> Callable:
    """Return a function that computes the divergence of vector field ``F``."""
    def div_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.trace(DF) * jnp.ones(1)
    return div_F


def curl(F: Callable) -> Callable:
    """Return a function that computes the curl of vector field ``F`` in 3D."""
    def curl_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.array([DF[2, 1] - DF[1, 2],
                          DF[0, 2] - DF[2, 0],
                          DF[1, 0] - DF[0, 1]])
    return curl_F


def grad(F: Callable) -> Callable:
    """Return a function that computes the gradient of scalar field ``F``."""
    def grad_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.ravel(DF)
    return grad_F


def l2_product(f: Callable,
               g: Callable,
               Q: Any,
               F: Callable = lambda x: x) -> jnp.ndarray:
    """L2 inner product ``<f, g>`` over the domain defined by quadrature ``Q``.

    Args:
        f: First integrand ``ξ -> array``.
        g: Second integrand ``ξ -> array``.
        Q: Quadrature rule with ``Q.x`` (points) and ``Q.w`` (weights).
        F: Optional coordinate map; Jacobian determinant is included.

    Returns:
        Scalar inner product value.
    """
    J_i = jax.lax.map(jacobian_determinant(F), Q.x,
                      batch_size=mrx.MAP_BATCH_SIZE_INNER)
    f_ij = jax.lax.map(f, Q.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    g_ij = jax.lax.map(g, Q.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return jnp.einsum("ij,ij,i,i->", f_ij, g_ij, J_i, Q.w)


def double_map(f, xs, ys):
    """Apply ``f(x, y)`` over all ``(xs[i], ys[j])`` via nested ``lax.map``.

    Returns an array of shape ``(len(xs), len(ys), ...)``.
    """
    def outer(x):
        return jax.lax.map(lambda y: f(x, y), ys,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return jax.lax.map(outer, xs, batch_size=mrx.MAP_BATCH_SIZE_OUTER)
