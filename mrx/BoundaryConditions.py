"""
Boundary condition operators for cube-like domains in differential forms.

This module provides a LazyBoundaryOperator class that handles boundary conditions
for differential forms on cube-like domains. It supports various types of boundary
conditions including Dirichlet and Neumann conditions.

The operator is implemented using JAX for efficient computation and supports
different form degrees (k = 0, 1, 2, 3) in three-dimensional space.
"""

import jax
import jax.numpy as jnp
import numpy as np


# Bpundary extraction operator for cube-like domains


class LazyBoundaryOperator:
    """
    A lazy boundary operator for handling boundary conditions in differential forms.

    This class implements boundary condition operators for differential forms
    on cube-like domains. It supports different types of boundary conditions
    and form degrees.

    Attributes:
        k (int): Degree of the differential form (0, 1, 2, or 3)
        Λ: Reference to the domain operator
        nr (int): Number of points in r-direction after boundary conditions
        nχ (int): Number of points in χ-direction after boundary conditions
        nζ (int): Number of points in ζ-direction after boundary conditions
        dr (int): Number of points in r-direction
        dχ (int): Number of points in χ-direction
        dζ (int): Number of points in ζ-direction
        n1 (int): Size of first component
        n2 (int): Size of second component
        n3 (int): Size of third component
        n (int): Total size of the operator
        M: Assembled operator matrix
    """

    def __init__(self, Λ, types):
        """
        Initialize the boundary operator.

        Args:
            Λ: Domain operator
            types (tuple): Tuple of boundary condition types for each direction.
                          Can be 'dirichlet' (zero at boundaries) or other types
                          (no boundary conditions).
        """
        self.k = Λ.k
        self.Λ = Λ
        self.nr = Λ.nr - 2 if types[0] == 'dirichlet' else Λ.nr
        self.nχ = Λ.nχ - 2 if types[1] == 'dirichlet' else Λ.nχ
        self.nζ = Λ.nζ - 2 if types[2] == 'dirichlet' else Λ.nζ
        self.dr, self.dχ, self.dζ = Λ.dr, Λ.dχ, Λ.dζ

        if self.k == 0:
            self.n1 = self.nr * self.nχ * self.nζ
            self.n2 = 0
            self.n3 = 0
        if self.k == 1:
            self.n1 = self.dr * self.nχ * self.nζ
            self.n2 = self.nr * self.dχ * self.nζ
            self.n3 = self.nr * self.nχ * self.dζ
        elif self.k == 2:
            self.n1 = self.nr * self.dχ * self.dζ
            self.n2 = self.dr * self.nχ * self.dζ
            self.n3 = self.dr * self.dχ * self.nζ
        elif self.k == 3:
            self.n1 = self.dr * self.dχ * self.dζ
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
            tuple: (category, local_index) where category indicates the vector
                  component and local_index is the index within that component
        """
        if self.k == 0 or self.k == 3:
            return 0, idx
        elif self.k == 1 or self.k == 2:
            n1, n2 = self.n1, self.n2
            category = jnp.int32(idx >= n1) + jnp.int32(idx >= n1 + n2)
            local_idx = idx - n1 * jnp.int32(idx >= n1) - n2 * jnp.int32(idx >= n1 + n2)
            return category, local_idx

    def _unravel_index(self, idx):
        """
        Convert linear index to multi-dimensional coordinates.

        Args:
            idx (int): Linear index

        Returns:
            tuple: (category, i, j, k) where category indicates the vector
                  component and (i,j,k) are the spatial coordinates
        """
        if self.k == 0:
            return 0, *jnp.unravel_index(idx, (self.nr, self.nχ, self.nζ))
        elif self.k == 1:
            category, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                category == 0,
                jnp.array(jnp.unravel_index(ijk, (self.dr, self.nχ, self.nζ))),
                jnp.where(
                    category == 1,
                    jnp.array(jnp.unravel_index(ijk, (self.nr, self.dχ, self.nζ))),
                    jnp.array(jnp.unravel_index(ijk, (self.nr, self.nχ, self.dζ)))
                )
            )
            return category, i, j, k
        elif self.k == 2:
            category, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                category == 0,
                jnp.array(jnp.unravel_index(ijk, (self.nr, self.dχ, self.dζ))),
                jnp.where(
                    category == 1,
                    jnp.array(jnp.unravel_index(ijk, (self.dr, self.nχ, self.dζ))),
                    jnp.array(jnp.unravel_index(ijk, (self.dr, self.dχ, self.nζ)))
                )
            )
            return category, i, j, k
        elif self.k == 3:
            return 0, *jnp.unravel_index(idx, (self.dr, self.dχ, self.dζ))

    def _element(self, row_idx, col_idx):
        """
        Compute the operator element at specified indices.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index

        Returns:
            jnp.ndarray: The operator element value
        """
        cat_row, i, j, k = self._unravel_index(row_idx)
        cat_col, r_idx, m, n = self.Λ._unravel_index(col_idx)

        if self.k == 0:
            # Handle 0-forms with boundary conditions
            return (
                (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == r_idx)
                 + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == r_idx-1))
                * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                   + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                   + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1))
            )
        elif self.k == 1:
            # Handle 1-forms with boundary conditions
            return jnp.where(
                cat_row == cat_col,
                jnp.where(
                    cat_row == 0,
                    jnp.int32(i == r_idx)
                    * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                       + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                    * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                       + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1)),
                    jnp.where(
                        cat_row == 1,
                        (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == r_idx)
                         + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == r_idx-1))
                        * jnp.int32(j == m)
                        * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                           + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1)),
                        (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == r_idx)
                         + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == r_idx-1))
                        * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                           + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                        * jnp.int32(k == n)
                    )
                ),
                0
            )
        elif self.k == 2:
            # Handle 2-forms with boundary conditions
            return jnp.where(
                cat_row == cat_col,
                jnp.where(
                    cat_row == 0,
                    (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == r_idx)
                     + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == r_idx-1))
                    * jnp.int32(j == m)
                    * jnp.int32(k == n),
                    jnp.where(
                        cat_row == 1,
                        jnp.int32(i == r_idx)
                        * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                           + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                        * jnp.int32(k == n),
                        jnp.int32(i == r_idx)
                        * jnp.int32(j == m)
                        * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                           + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1))
                    )
                ),
                0
            )
        elif self.k == 3:
            return jnp.int32(row_idx == col_idx)

    def assemble(self):
        """
        Assemble the complete boundary operator matrix.

        Returns:
            jnp.ndarray: The assembled operator matrix
        """
        return jax.vmap(
            jax.vmap(self._element, (None, 0)), (0, None)
        )(jnp.arange(self.n), jnp.arange(self.Λ.n))
