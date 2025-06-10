"""
Projector classes for finite element differential forms.

This module provides classes for projecting functions onto finite element spaces
in the context of differential forms. It supports projections of k-forms
(k = 0, 1, 2, 3) and includes functionality for handling coordinate transformations
and curl projections.

The module implements two main classes:
- Projector: For standard projections of k-forms
- CurlProjection: For projecting curl operations on differential forms
"""

import jax
import jax.numpy as jnp

from mrx.Utils import div, grad, inv33, jacobian_determinant

__all__ = ['Projector', 'CurlProjection']


class Projector:
    """
    A class for projecting functions onto finite element spaces.

    This class implements projection operators for differential forms of various
    degrees (k = 0, 1, 2, 3). It supports coordinate transformations through
    the mapping F and can handle extraction operators through E.

    Attributes:
        Λ: The domain operator defining the finite element space
        Q: Quadrature rule for numerical integration
        n (int): Total size of the operator
        ns (array): Array of indices for the finite element space
        F (callable): Coordinate transformation function, defaults to identity
        M (array): Extraction operator matrix, defaults to identity
    """

    def __init__(self, Λ, Q, F=None, E=None):
        """
        Initialize the projector.

        Args:
            Λ: Domain operator defining the finite element space
            Q: Quadrature rule for numerical integration
            F (callable, optional): Coordinate transformation function.
                                 Defaults to identity mapping.
            E (array, optional): Extraction operator matrix.
                              Defaults to identity matrix.
        """
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E

    # def __call__(self, f):
    #     return self.M @ self.projection(f)

    # def projection(self, f):
    #     # evaluate all basis functions at all quadrature points
    #     Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(self.Q.x, self.ns) # n x n_q x d
    #     # evalute the given function at all quadrature points
    #     fjk = jax.vmap(f)(self.Q.x) # n_q x d
    #     # evaluate the jacobian of F at all quadrature points
    #     wj = self.Q.w # n_q
    #     return jnp.einsum("ijk,jk,j->i", Λijk, fjk, wj)

    def __call__(self, f):
        """
        Project a function onto the finite element space.

        Args:
            f (callable): Function to project

        Returns:
            array: Projection coefficients
        """
        if self.Λ.k == 0:
            return self.M @ self.zeroform_projection(f)
        elif self.Λ.k == 1:
            return self.M @ self.oneform_projection(f)
        elif self.Λ.k == 2:
            return self.M @ self.twoform_projection(f)
        elif self.Λ.k == 3:
            return self.M @ self.threeform_projection(f)

    def zeroform_projection(self, f):
        """
        Project a scalar function (0-form).

        Args:
            f (callable): Scalar function to project

        Returns:
            array: Projection coefficients for the 0-form
        """
        # Evaluate all basis functions at quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, self.ns)  # n x n_q x 1
        # Evaluate the given function at quadrature points
        fjk = jax.vmap(f)(self.Q.x)  # n_q x 1
        # Evaluate the jacobian of F at quadrature points
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,jk,j,j->i", Λijk, fjk, Jj, wj)

    def oneform_projection(self, A):
        """
        Project a vector field (1-form).

        Args:
            A (callable): Vector field to project

        Returns:
            array: Projection coefficients for the 1-form
        """
        DF = jax.jacfwd(self.F)

        def _A(x):
            return inv33(DF(x)).T @ A(x)

        def _Λ(x, i):
            return inv33(DF(x)).T @ self.Λ(x, i)
        Ajk = jax.vmap(_A)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Ajk, Jj, wj)

    def twoform_projection(self, B):
        """
        Project a 2-form field.

        Args:
            B (callable): 2-form field to project

        Returns:
            array: Projection coefficients for the 2-form
        """
        # DF = jax.jacfwd(self.F)

        def _B(x):
            return B(x)

        def _Λ(x, i):
            return self.Λ(x, i)
        Bjk = jax.vmap(_B)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        # Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j->i", Λijk, Bjk, wj)
        # return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, 1/Jj, wj)

    def threeform_projection(self, f):
        """
        Project a volume form (3-form).

        Args:
            f (callable): Volume form to project

        Returns:
            array: Projection coefficients for the 3-form
        """
        # Evaluate all basis functions at quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x 1
        fjk = jax.vmap(f)(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,jk,j->i", Λijk, fjk, wj)
        # Jj = jax.vmap(jacobian(self.F))(self.Q.x)  # n_q x 1
        # return jnp.einsum("ijk,jk,j,j->i", Λijk, fjk, 1/Jj, wj)


class CurlProjection:
    """
    Given one-form A and two-form B, computes ∫ B·(A × Λ[i]) dx for all i,
    where Λ[i] is the i-th basis function of the one-form space.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        """
        Initialize the curl projector.

        Args:
            Λ: Domain operator defining the finite element space
            Q: Quadrature rule for numerical integration
            F (callable, optional): Coordinate transformation function.
                                 Defaults to identity mapping.
            E (array, optional): Extraction operator matrix.
                              Defaults to identity matrix.
        """
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E

    def __call__(self, A, B):
        """
        Project the curl operation between forms A and B.

        Args:
            A (callable): One-form field
            B (callable): Two-form field

        Returns:
            array: Projection coefficients
        """
        return self.M @ self.projection(A, B)

    def projection(self, A, B):
        """
        Compute the projection of (B, A × Λ[i]).

        Given a one-form A and two-form B, computes the projection of their
        cross product with the basis functions.

        Args:
            A (callable): One-form field
            B (callable): Two-form field

        Returns:
            array: Projection coefficients
        """
        DF = jax.jacfwd(self.F)

        def _B(x):
            return DF(x) @ B(x)

        def _Λ(x, i):
            # Note: cross products of one-forms transform like two-forms
            return DF(x) @ jnp.cross(A(x), self.Λ(x, i))

        # Compute projections
        Bjk = jax.vmap(_B)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, 1/Jj, wj)


class GradientProjection:
    """
    Given zero-form p and two-form u, computes 
    ∫ ( grad(p)·u Λ[i] + Ɣ p div(u) Λ[i] ) dx 
    for all i, where Λ[i] is the i-th basis function of the zero-form space.
    """

    def __init__(self, Λ, Q, F=None, E=None, Ɣ=5/3):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        self.Ɣ = Ɣ
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E

    def __call__(self, p, u):
        return self.M @ self.projection(p, u)

    def projection(self, p, u):
        def q(x):
            return grad(p)(x) @ u(x) + self.Ɣ * p(x) * div(u)(x)

        # Compute projections
        qjk = jax.vmap(q)(self.Q.x)  # n_q x 1
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j->i", Λijk, qjk, wj)


# class EFieldProjector:
#     """
#     Given four one-forms A, B, C and D,
#     computes ∫ ((A × B - D) × C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the one-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C, D):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C, D)

#     def projection(self, A, B, C, D):
#         """
#         Compute the projection of ∫ ((A × B - D) × C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field
#             D (callable): Zero-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - D(x), C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)


# class ForceProjector:
#     """
#     Given three one-forms A, B and C,
#     computes ∫ (A × B - C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the one-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C)

#     def projection(self, A, B, C):
#         """
#         Compute the projection of ∫ (A × B - C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - C(x), C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)


# class ForceProjector2:
#     """
#     Given three one-forms A, B and C,
#     computes ∫ (A × B - C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the two-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C)

#     def projection(self, A, B, C):
#         """
#         Compute the projection of ∫ (A × B - C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - C(x),  C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)


# class EFieldProjector:
#     """
#     Given four one-forms A, B, C and D,
#     computes ∫ ((A × B - D) × C) · Λ[i] dx for all i,
#     where Λ[i] is the i-th basis function of the one-form space.
#     """

#     def __init__(self, Λ, Q, F=None, E=None):
#         """
#         Initialize the curl projector.

#         Args:
#             Λ: Domain operator defining the finite element space
#             Q: Quadrature rule for numerical integration
#             F (callable, optional): Coordinate transformation function.
#                                  Defaults to identity mapping.
#             E (array, optional): Extraction operator matrix.
#                               Defaults to identity matrix.
#         """
#         self.Λ = Λ
#         self.Q = Q
#         self.n = Λ.n
#         self.ns = Λ.ns
#         if F is None:
#             self.F = lambda x: x
#         else:
#             self.F = F
#         if E is None:
#             self.M = jnp.eye(self.n)
#         else:
#             self.M = E

#     def __call__(self, A, B, C, D):
#         """
#         Project the curl operation between forms A and B.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field

#         Returns:
#             array: Projection coefficients
#         """
#         return self.M @ self.projection(A, B, C, D)

#     def projection(self, A, B, C, D):
#         """
#         Compute the projection of ∫ ((A × B - D) × C) · Λ[i] dx.

#         Args:
#             A (callable): One-form field
#             B (callable): One-form field
#             C (callable): One-form field
#             D (callable): Zero-form field

#         Returns:
#             array: Projection coefficients
#         """
#         DF = jax.jacfwd(self.F)

#         def u(x):
#             DFx = DF(x)
#             Jx = jnp.linalg.det(DFx)
#             return jnp.cross(1/Jx * DFx.T @ DFx @ jnp.cross(A(x), B(x)) - D(x), C(x))

#         # Compute projections
#         ujk = jax.vmap(u)(self.Q.x)  # n_q x d
#         Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
#             self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
#         Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
#         wj = self.Q.w
#         return jnp.einsum("ijk,jk,j,j->i", Λijk, ujk, 1/Jj, wj)
