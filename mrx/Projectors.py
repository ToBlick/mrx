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

from mrx.Utils import inv33, jacobian_determinant

__all__ = ['Projector']


class Projector:
    """
    A class for projecting functions onto finite element spaces.

    Functions are represented as functions of the logical coordinate ξ in the 
    physical (x,y,z) frame, for example:
    v(ξ) = v_x(ξ) e_x + v_y(ξ) e_y + v_z(ξ) e_z

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
            self.E = jnp.eye(self.n)
        else:
            self.E = E

    def __call__(self, f):
        """
        Project a function onto the finite element space.

        Args:
            f (callable): Function to project

        Returns:
            array: Projection coefficients
        """
        if self.Λ.k == 0:
            return self.E @ self.zeroform_projection(f)
        elif self.Λ.k == 1:
            return self.E @ self.oneform_projection(f)
        elif self.Λ.k == 2:
            return self.E @ self.twoform_projection(f)
        elif self.Λ.k == 3:
            return self.E @ self.threeform_projection(f)
        elif self.Λ.k == -1:
            return self.E @ self.vectorfield_projection(f)

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

    def oneform_projection(self, v):
        """
        Project a vector-valued function to a 1-form.

        Args:
            A (callable): Vector field to project

        Returns:
            array: Projection coefficients for the 1-form
        """
        DF = jax.jacfwd(self.F)

        def _v(x):
            return inv33(DF(x)) @ v(x)

        Ajk = jax.vmap(_v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Ajk, Jj, wj)

    def twoform_projection(self, v):
        """
        Project to a 2-form.

        Args:
            v (callable): vector field to project - in physical coordinates

        Returns:
            array: Projection coefficients for the 2-form
        """
        DF = jax.jacfwd(self.F)

        def _v(x):
            return DF(x).T @ v(x)

        def _Λ(x, i):
            return self.Λ(x, i)
        Bjk = jax.vmap(_v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j->i", Λijk, Bjk, wj)

    def threeform_projection(self, f):
        """
        Project a volume form (3-form).

        Args:
            f (callable): function

        Returns:
            array: Projection coefficients for the 3-form
        """
        # Evaluate all basis functions at quadrature points
        Λijk = jax.vmap(jax.vmap(self.Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x 1
        fjk = jax.vmap(f)(self.Q.x)  # n_q x 1
        wj = self.Q.w  # n_q
        return jnp.einsum("ijk,jk,j->i", Λijk, fjk, wj)

    def vectorfield_projection(self, v):
        """
        Project to a vector field.

        Args:
            v (callable): vector field to project - in physical coordinates

        Returns:
            array: Projection coefficients for the vector field
        """
        DF = jax.jacfwd(self.F)

        def _v(x):
            return DF(x).T @ v(x)

        def _Λ(x, i):
            return self.Λ(x, i)
        Bjk = jax.vmap(_v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, Jj, wj)
