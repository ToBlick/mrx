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

from mrx.utils import integrate_against, inv33, jacobian_determinant

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

    def __init__(self, Seq, k):
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
        self.k = k
        self.Seq = Seq

    def __call__(self, f):
        """
        Project a function onto the finite element space.

        Args:
            f (callable): Function to project

        Returns:
            array: Projection coefficients
        """
        if self.k == 0:
            return self.Seq.E0 @ self.zeroform_projection(f)
        elif self.k == 1:
            return self.Seq.E1 @ self.oneform_projection(f)
        elif self.k == 2:
            return self.Seq.E2 @ self.twoform_projection(f)
        elif self.k == 3:
            return self.Seq.E3 @ self.threeform_projection(f)

    def zeroform_projection(self, f):
        """
        Project a scalar function (0-form).

        Args:
            f (callable): Scalar function to project

        Returns:
            array: Projection coefficients for the 0-form
        """
        # Evaluate the given function at quadrature points
        f_jk = jax.vmap(f)(self.Seq.Q.x)  # n_q x 1
        w_jk = f_jk * (self.Seq.Q.w * self.Seq.J_j)[:, None]
        return integrate_against(self.Seq.get_Λ0_ijk, w_jk, self.Seq.Λ0.n)

    def oneform_projection(self, v):
        """
        Project a vector-valued function to a 1-form.

        Args:
            A (callable): Vector field to project

        Returns:
            array: Projection coefficients for the 1-form
        """
        DF = jax.jacfwd(self.Seq.F)

        def _v(x):
            return inv33(DF(x)) @ v(x)

        # Evaluate the given function at quadrature points
        A_jk = jax.vmap(_v)(self.Seq.Q.x)  # n_q x d
        w_jk = A_jk * (self.Seq.Q.w * self.Seq.J_j)[:, None]

        return integrate_against(self.Seq.get_Λ1_ijk, w_jk, self.Seq.Λ1.n)

    def twoform_projection(self, v):
        """
        Project to a 2-form.

        Args:
            v (callable): vector field to project - in physical coordinates

        Returns:
            array: Projection coefficients for the 2-form
        """
        DF = jax.jacfwd(self.Seq.F)

        def _v(x):
            return DF(x).T @ v(x)

        # Evaluate the given function at quadrature points
        B_jk = jax.vmap(_v)(self.Seq.Q.x)  # n_q x d

        w_jk = B_jk * (self.Seq.Q.w)[:, None]

        return integrate_against(self.Seq.get_Λ2_ijk, w_jk, self.Seq.Λ2.n)

    def threeform_projection(self, f):
        """
        Project a volume form (3-form).

        Args:
            f (callable): function

        Returns:
            array: Projection coefficients for the 3-form
        """
        # Evaluate the given function at quadrature points
        f_jk = jax.vmap(f)(self.Seq.Q.x)  # n_q x 1
        w_jk = f_jk * (self.Seq.Q.w)[:, None]
        return integrate_against(self.Seq.get_Λ3_ijk, w_jk, self.Seq.Λ3.n)
