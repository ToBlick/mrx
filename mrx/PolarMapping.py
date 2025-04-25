"""
Polar mapping utilities for finite element analysis.

This module provides classes and functions for handling polar coordinate transformations
and boundary conditions in finite element computations.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix


class LazyExtractionOperator:
    """
    A class for extracting boundary conditions and handling polar mappings.

    This class implements operators for handling boundary conditions and polar
    coordinate transformations in finite element computations. It supports different
    form degrees (k = 0, 1, 2, 3) and handles both inner and outer boundaries.

    Attributes:
        k (int): Degree of the differential form
        Λ: Reference to the domain operator
        ξ: Polar mapping coefficients
        nr (int): Number of points in r-direction
        nχ (int): Number of points in χ-direction
        nζ (int): Number of points in ζ-direction
        dr (int): Number of points in r-direction after boundary conditions
        dχ (int): Number of points in χ-direction after boundary conditions
        dζ (int): Number of points in ζ-direction after boundary conditions
        o (int): Offset for boundary conditions (1 for zero BC, 0 otherwise)
        n1 (int): Size of first component
        n2 (int): Size of second component
        n3 (int): Size of third component
        n (int): Total size of the operator
        M: Assembled operator matrix
    """

    def __init__(self, Λ, ξ, zero_bc):
        """
        Initialize the extraction operator.

        Args:
            Λ: Domain operator
            ξ: Polar mapping coefficients
            zero_bc (bool): Whether to apply zero boundary conditions
        """
        self.k = Λ.k
        self.Λ = Λ
        self.ξ = ξ
        self.nr, self.nχ, self.nζ = Λ.nr, Λ.nχ, Λ.nζ
        self.dr, self.dχ, self.dζ = Λ.dr, Λ.dχ, Λ.dζ
        self.o = 1 if zero_bc else 0  # offset for boundary conditions

        # Set component sizes based on form degree
        if self.k == 0:
            self.n1 = ((self.nr - 2 - self.o) * self.nχ + 3) * self.nζ
            self.n2 = 0
            self.n3 = 0
        if self.k == 1:
            self.n1 = (self.dr - 1) * self.nχ * self.nζ
            self.n2 = ((self.nr - 2 - self.o) * self.dχ + 2) * self.nζ
            self.n3 = ((self.nr - 2 - self.o) * self.nχ + 3) * self.dζ
        if self.k == 2:
            self.n1 = ((self.nr - 2 - self.o) * self.dχ + 2) * self.dζ
            self.n2 = (self.dr - 1) * self.nχ * self.dζ
            self.n3 = (self.dr - 1) * self.dχ * self.nζ
        if self.k == 3:
            self.n1 = (self.dr - 1) * self.dχ * self.dζ
            self.n2 = 0
            self.n3 = 0
        self.n = self.n1 + self.n2 + self.n3
        self.M = self.assemble()

    def __getitem__(self, idx):
        """Get operator element at specified index."""
        return self.M[idx]

    def __array__(self):
        """Convert operator to numpy array."""
        return np.array(self.M)

    def _vector_index(self, idx):
        """
        Convert linear index to vector component and local index.

        Args:
            idx (int): Linear index

        Returns:
            tuple: (category, index) where category indicates the vector component
                  and index is the local index within that component
        """
        n1, n2 = self.n1, self.n2
        if self.k == 0 or self.k == 3:
            return 0, idx
        elif self.k == 1 or self.k == 2:
            category = jnp.int32(idx >= n1) + jnp.int32(idx >= n1 + n2)
            index = idx - n1 * jnp.int32(idx >= n1) - \
                n2 * jnp.int32(idx >= n1 + n2)
            return category, index

    def _element(self, row_idx, col_idx):
        """
        Compute the operator element at specified indices.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index

        Returns:
            jnp.ndarray: The operator element value
        """
        if self.k == 0:
            # Handle 0-forms
            return jnp.where(
                row_idx < 3 * self.nζ,
                self._inner_zeroform(
                    row_idx, col_idx, self.nr, self.nχ, self.nζ),
                self._outer_zeroform(
                    row_idx - 3 * self.nζ, col_idx, self.nr, self.nχ, self.nζ
                ),
            )
        if self.k == 1:
            # Handle 1-forms
            cat_row, row_idx = self._vector_index(row_idx)
            cat_col, col_idx = self.Λ._vector_index(col_idx)
            return jnp.where(
                cat_row == 0,
                # r-component
                self._threeform(row_idx, col_idx, self.dr, self.nχ, self.nζ)
                * jnp.int32(cat_row == cat_col),
                jnp.where(
                    cat_row == 1,
                    # χ-component
                    jnp.where(
                        row_idx < 2 * self.nζ,
                        self.inner_oneform_r(
                            row_idx, col_idx, self.dr, self.nχ, self.nζ
                        )
                        * jnp.int32(cat_col == 0)
                        + self.inner_oneform_χ(
                            row_idx, col_idx, self.nr, self.dχ, self.nζ
                        )
                        * jnp.int32(cat_col == 1),
                        self._outer_zeroform(
                            row_idx - 2 * self.nζ, col_idx, self.nr, self.dχ, self.nζ
                        )
                        * jnp.int32(cat_row == cat_col),
                    ),
                    # ζ-component
                    jnp.where(
                        row_idx < 3 * self.dζ,
                        self._inner_zeroform(
                            row_idx, col_idx, self.nr, self.nχ, self.dζ
                        )
                        * jnp.int32(cat_row == cat_col),
                        self._outer_zeroform(
                            row_idx - 3 * self.dζ, col_idx, self.nr, self.nχ, self.dζ
                        )
                        * jnp.int32(cat_row == cat_col),
                    ),
                ),
            )
        if self.k == 2:
            # Handle 2-forms
            cat_row, row_idx = self._vector_index(row_idx)
            cat_col, col_idx = self.Λ._vector_index(col_idx)
            return jnp.where(
                cat_row == 0,
                # r-component
                jnp.where(
                    row_idx < 2 * self.nζ,
                    self.inner_oneform_χ(
                        row_idx, col_idx, self.nr, self.dχ, self.dζ)
                    * jnp.int32(cat_col == 0)
                    - self.inner_oneform_r(row_idx, col_idx,
                                           self.dr, self.nχ, self.dζ)
                    * jnp.int32(cat_col == 1),
                    self._outer_zeroform(
                        row_idx - 2 * self.nζ, col_idx, self.nr, self.dχ, self.dζ
                    )
                    * jnp.int32(cat_row == cat_col),
                ),
                jnp.where(
                    cat_row == 1,
                    # χ-component
                    self._threeform(row_idx, col_idx,
                                    self.dr, self.nχ, self.dζ)
                    * jnp.int32(cat_row == cat_col),
                    # ζ-component
                    self._threeform(row_idx, col_idx,
                                    self.dr, self.dχ, self.nζ)
                    * jnp.int32(cat_row == cat_col),
                ),
            )
        if self.k == 3:
            # Handle 3-forms
            return self._threeform(row_idx, col_idx, self.nr, self.nχ, self.nζ)

    def _inner_zeroform(self, row_idx, col_idx, nr, nχ, nζ):
        """
        Compute inner zero-form basis function.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nχ (int): Number of points in χ-direction
            nζ (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        p, m = jnp.unravel_index(row_idx, (3, nζ))
        i, j, k = jnp.unravel_index(col_idx, (nr, nχ, nζ))
        return jnp.int32(k == m) * jnp.int32(i < 2) * self.ξ[p, i, j]

    def _outer_zeroform(self, row_idx, col_idx, nr, nχ, nζ):
        """
        Compute outer zero-form basis function.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nχ (int): Number of points in χ-direction
            nζ (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        i, j, k = jnp.unravel_index(row_idx, (nr, nχ, nζ))
        return jnp.int32(
            col_idx == jnp.ravel_multi_index(
                (i + 2, j, k), (nr, nχ, nζ), mode="clip")
        ) * jnp.where(self.o == 1, jnp.int32(i != nr - 1), 1)

    def inner_oneform_r(self, row_idx, col_idx, nr, nχ, nζ):
        """
        Compute inner one-form basis function in r-direction.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nχ (int): Number of points in χ-direction
            nζ (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        p, m = jnp.unravel_index(row_idx, (2, nζ))
        p += 1
        i, j, k = jnp.unravel_index(col_idx, (nr, nχ, nζ))
        return (
            jnp.int32(k == m) * jnp.int32(i == 0) *
            (self.ξ[p, 1, j] - self.ξ[p, 0, j])
        )

    def inner_oneform_χ(self, row_idx, col_idx, nr, nχ, nζ):
        """
        Compute inner one-form basis function in χ-direction.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nχ (int): Number of points in χ-direction
            nζ (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        p, m = jnp.unravel_index(row_idx, (2, nζ))
        p += 1
        i, j, k = jnp.unravel_index(col_idx, (nr, nχ, nζ))
        return (
            jnp.int32(k == m)
            * jnp.int32(i == 1)
            * (self.ξ[p, 1, jnp.mod(j + 1, nχ)] - self.ξ[p, 1, j])
        )

    def _threeform(self, row_idx, col_idx, nr, nχ, nζ):
        """
        Compute three-form basis function.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nχ (int): Number of points in χ-direction
            nζ (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        i, j, k = jnp.unravel_index(row_idx, (nr, nχ, nζ))
        return jnp.int32(
            col_idx == jnp.ravel_multi_index(
                (i + 1, j, k), (nr, nχ, nζ), mode="clip")
        )

    def assemble(self):
        """Assemble the complete operator matrix."""
        return jax.vmap(jax.vmap(self._element, (None, 0)), (0, None))(
            jnp.arange(self.n), jnp.arange(self.Λ.n)
        )


def get_xi(_R, _Y, Λ0, Q):
    """
    Compute polar mapping coefficients.

    This function computes the coefficients for the polar mapping transformation
    based on given R and Y functions.

    Args:
        _R (callable): Function defining the R-coordinate transformation
        _Y (callable): Function defining the Y-coordinate transformation
        Λ0: Reference to the domain operator

    Returns:
        tuple: (ξ, R_hat, Y_hat, Λ0, τ) where:
            - ξ: Polar mapping coefficients
            - R_hat: Projected R-coordinates
            - Y_hat: Projected Y-coordinates
            - Λ0: Reference to the domain operator
            - τ: Scaling parameter
    """
    nr, nχ, nζ = Λ0.nr, Λ0.nχ, Λ0.nζ
    P = Projector(Λ0, Q)
    M = LazyMassMatrix(Λ0, Q).M

    def R(x):
        return _R(x[0], x[1])

    def Y(x):
        return _Y(x[0], x[1])

    R0 = _R(0, 0)
    Y0 = _Y(0, 0)

    R_hat = jnp.linalg.solve(M, P(R))
    Y_hat = jnp.linalg.solve(M, P(Y))

    cR = R_hat.reshape(nr, nχ, nζ)
    cY = Y_hat.reshape(nr, nχ, nζ)
    ΔR = cR[1, :, 0] - R0
    ΔY = cY[1, :, 0] - Y0
    τ = jnp.max(
        jnp.array(
            [
                jnp.max(-2 * ΔR),
                jnp.max(ΔR - jnp.sqrt(3) * ΔY),
                jnp.max(ΔR + jnp.sqrt(3) * ΔY),
            ]
        )
    )
    ξ00 = jnp.ones(nχ) / 3
    ξ01 = 1 / 3 + 2 / (3 * τ) * ΔR
    ξ10 = jnp.ones(nχ) / 3
    ξ11 = 1 / 3 - 1 / (3 * τ) * ΔR + jnp.sqrt(3) / (3 * τ) * ΔY
    ξ20 = jnp.ones(nχ) / 3
    ξ21 = 1 / 3 - 1 / (3 * τ) * ΔR - jnp.sqrt(3) / (3 * τ) * ΔY
    # (3, 2, nχ) -> l, i, j
    ξ = jnp.array([[ξ00, ξ01], [ξ10, ξ11], [ξ20, ξ21]])
    return ξ, R_hat, Y_hat, Λ0, τ
