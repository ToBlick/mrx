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

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import jax
from jax import Array

import mrx
from mrx.utils import integrate_against_deprecated, inv33

if TYPE_CHECKING:
    from mrx.derham_sequence import DeRhamSequence


# Type aliases for callable functions used in projections
ScalarFunction = Callable[[Array], Array]  # ξ -> scalar (with trailing dim)
VectorFunction = Callable[[Array], Array]  # ξ -> 3D vector


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
        k (int): Degree of the differential form (0, 1, 2, or 3)
        seq : DeRham sequence object
        dirichlet (bool): Whether to use dirichlet boundary conditions
    """

    k: Literal[0, 1, 2, 3]
    seq: DeRhamSequence
    dirichlet: bool = True

    def __init__(self, seq: DeRhamSequence, k: Literal[0, 1, 2, 3], dirichlet: bool = True) -> None:
        """
        Initialize the projector.

        Args:
            seq : DeRham sequence object
            k : Degree of the differential form
            dirichlet : Whether to use dirichlet boundary conditions
        """
        self.k = k
        self.seq = seq
        self.dirichlet = dirichlet

    def __call__(self, f: ScalarFunction | VectorFunction) -> Array:
        """
        Project a function onto the finite element space.

        Args:
            f (callable): Function to project

        Returns:
            array: Projection coefficients
        """

        if self.k == 0:
            e = self.seq.e0_dbc if self.dirichlet else self.seq.e0
            return e @ self.zeroform_projection(f)
        elif self.k == 1:
            e = self.seq.e1_dbc if self.dirichlet else self.seq.e1
            return e @ self.oneform_projection(f)
        elif self.k == 2:
            e = self.seq.e2_dbc if self.dirichlet else self.seq.e2
            return e @ self.twoform_projection(f)
        elif self.k == 3:
            e = self.seq.e3_dbc if self.dirichlet else self.seq.e3
            return e @ self.threeform_projection(f)
        # TODO: Consider raising an error for invalid k values
        raise ValueError(f"Invalid k value: {self.k}. Must be 0, 1, 2, or 3.")

    def zeroform_projection(self, f: ScalarFunction) -> Array:
        """
        Project a scalar function (0-form).

        Args:
            f (callable): Scalar function to project

        Returns:
            array: Projection coefficients for the 0-form
        """
        # Evaluate the given function at quadrature points
        f_jk: Array = jax.lax.map(
            f, self.seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)  # n_q x 1
        w_jk: Array = f_jk * (self.seq.quad.w * self.seq.jacobian_j)[:, None]
        return integrate_against_deprecated(self.seq.eval_basis_0_ijk, w_jk, self.seq.basis_0.n)

    def oneform_projection(self, v: VectorFunction) -> Array:
        """
        Project a vector-valued function to a 1-form.

        Args:
            A (callable): Vector field to project

        Returns:
            array: Projection coefficients for the 1-form
        """
        DF = jax.jacfwd(self.seq.map)

        def _v(x: Array) -> Array:
            return inv33(DF(x)) @ v(x)

        # Evaluate the given function at quadrature points
        A_jk: Array = jax.lax.map(
            _v, self.seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)  # n_q x d
        w_jk: Array = A_jk * (self.seq.quad.w * self.seq.jacobian_j)[:, None]

        return integrate_against_deprecated(self.seq.eval_basis_1_ijk, w_jk, self.seq.basis_1.n)

    def twoform_projection(self, v: VectorFunction) -> Array:
        """
        Project to a 2-form.

        Args:
            v (callable): vector field to project - in physical coordinates

        Returns:
            array: Projection coefficients for the 2-form
        """
        DF = jax.jacfwd(self.seq.map)

        def _v(x: Array) -> Array:
            return DF(x).T @ v(x)

        # Evaluate the given function at quadrature points
        B_jk: Array = jax.lax.map(
            _v, self.seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)  # n_q x d

        w_jk: Array = B_jk * (self.seq.quad.w)[:, None]

        return integrate_against_deprecated(self.seq.eval_basis_2_ijk, w_jk, self.seq.basis_2.n)

    def threeform_projection(self, f: ScalarFunction) -> Array:
        """
        Project a volume form (3-form).

        Args:
            f (callable): function

        Returns:
            array: Projection coefficients for the 3-form
        """
        # Evaluate the given function at quadrature points
        f_jk: Array = jax.lax.map(
            f, self.seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)  # n_q x 1
        w_jk: Array = f_jk * (self.seq.quad.w)[:, None]
        return integrate_against_deprecated(self.seq.eval_basis_3_ijk, w_jk, self.seq.basis_3.n)
