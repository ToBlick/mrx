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

__all__ = ['LazyBoundaryOperator']


# Boundary extraction operator for cube-like domains
class LazyBoundaryOperator:
    """
    A lazy boundary operator for handling boundary conditions in differential forms.

    This class implements boundary condition operators for differential forms
    on cube-like domains. It supports different types of boundary conditions
    and form degrees.

    Attributes:
        k (int): Degree of the differential form (0, 1, 2, or 3)
        Lambda_0 (DifferentialForm)
        types (tuple): Tuple of boundary condition types for each direction.
        nr (int): Number of points in r-direction after boundary conditions
        nt (int): Number of points in θ-direction after boundary conditions
        nz (int): Number of points in ζ-direction after boundary conditions
        dr (int): Number of points in r-direction
        dt (int): Number of points in θ-direction
        dz (int): Number of points in ζ-direction
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
            Λ (DifferentialForm)
            types (tuple): Tuple of boundary condition types for each direction.
                          Can be 'dirichlet' (zero at boundaries), 'half' (zero only at x=1)
                          or other types (no boundary conditions).
        """
        self.k = Λ.k
        self.Lambda = Λ

        def get_dim(original_dim, bc_type):
            if bc_type == "dirichlet":
                return original_dim - 2
            elif bc_type == "right":
                return original_dim - 1
            elif bc_type == "left":
                return original_dim - 1
            else:
                return original_dim

        self.nr, self.nt, self.nz = get_dim(self.Lambda.nr, types[0]), get_dim(
            self.Lambda.nt, types[1]), get_dim(self.Lambda.nz, types[2])
        self.dr, self.dt, self.dz = self.Lambda.dr, self.Lambda.dt, self.Lambda.dz
        self.types = types

        if self.k == 0:
            self.n1 = self.nr * self.nt * self.nz
            self.n2 = 0
            self.n3 = 0
        if self.k == 1:
            self.n1 = self.dr * self.nt * self.nz
            self.n2 = self.nr * self.dt * self.nz
            self.n3 = self.nr * self.nt * self.dz
        elif self.k == 2:
            self.n1 = self.nr * self.dt * self.dz
            self.n2 = self.dr * self.nt * self.dz
            self.n3 = self.dr * self.dt * self.nz
        elif self.k == 3:
            self.n1 = self.dr * self.dt * self.dz
            self.n2 = 0
            self.n3 = 0
        elif self.k == -1:
            self.n1 = self.nr * self.nt * self.nz
            self.n2 = self.nr * self.nt * self.nz
            self.n3 = self.nr * self.nt * self.nz
        self.n = self.n1 + self.n2 + self.n3

    def matrix(self):
        """Wrapper for the assemble method."""
        return self.assemble()

    def __array__(self):
        """Convert operator to numpy array."""
        return np.array(self.matrix())

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
            return jnp.int32(0), idx
        elif self.k == 1 or self.k == 2 or self.k == -1:
            n1, n2 = self.n1, self.n2
            category = jnp.int32(idx >= n1) + jnp.int32(idx >= n1 + n2)
            local_idx = jnp.int32(
                idx - n1 * (idx >= n1) - n2 * (idx >= n1 + n2))
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
            return jnp.int32(0), *jnp.unravel_index(idx, (self.nr, self.nt, self.nz))
        elif self.k == 1:
            category, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                category == 0,
                jnp.array(jnp.unravel_index(ijk, (self.dr, self.nt, self.nz))),
                jnp.where(
                    category == 1,
                    jnp.array(jnp.unravel_index(
                        ijk, (self.nr, self.dt, self.nz))),
                    jnp.array(jnp.unravel_index(
                        ijk, (self.nr, self.nt, self.dz))),
                ),
            )
            return category, i, j, k
        elif self.k == 2:
            category, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                category == 0,
                jnp.array(jnp.unravel_index(ijk, (self.nr, self.dt, self.dz))),
                jnp.where(
                    category == 1,
                    jnp.array(jnp.unravel_index(
                        ijk, (self.dr, self.nt, self.dz))),
                    jnp.array(jnp.unravel_index(
                        ijk, (self.dr, self.dt, self.nz))),
                ),
            )
            return category, i, j, k
        elif self.k == 3:
            return jnp.int32(0), *jnp.unravel_index(idx, (self.dr, self.dt, self.dz))
        elif self.k == -1:
            category, ijk = self._vector_index(idx)
            i, j, k = jnp.array(jnp.unravel_index(
                ijk, (self.nr, self.nt, self.nz)))
            return category, i, j, k

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
        cat_col, p, m, n = self.Lambda._unravel_index(col_idx)

        target_l = jnp.where(jnp.logical_or(
            self.types[0] == "dirichlet", self.types[0] == "left"), p - 1, p)
        target_m = jnp.where(jnp.logical_or(
            self.types[1] == "dirichlet", self.types[1] == "left"), m - 1, m)
        target_n = jnp.where(jnp.logical_or(
            self.types[2] == "dirichlet", self.types[2] == "left"), n - 1, n)

        if self.k == 0:
            return jnp.int32((i == target_l) * (j == target_m) * (k == target_n))
        elif self.k == 1:
            return jnp.int32(
                jnp.where(
                    cat_row == cat_col,
                    jnp.where(
                        cat_row == 0,
                        (i == p) * (j == target_m) * (k == target_n),
                        jnp.where(
                            cat_row == 1,
                            (i == target_l) * (j == m) * (k == target_n),
                            (i == target_l) * (j == target_m) * (k == n),
                        ),
                    ),
                    0,
                )
            )
        elif self.k == 2 or self.k == -1:
            return jnp.int32(
                jnp.where(
                    cat_row == cat_col,
                    jnp.where(
                        cat_row == 0,
                        (i == target_l) * (j == m) * (k == n),
                        jnp.where(
                            cat_row == 1,
                            (i == p) * (j == target_m) * (k == n),
                            (i == p) * (j == m) * (k == target_n),
                        ),
                    ),
                    0,
                )
            )
        elif self.k == 3:
            return jnp.int32(row_idx == col_idx)

    def assemble(self):
        """
        Assemble the complete boundary operator matrix.

        Returns:
            jnp.ndarray: The assembled operator matrix
        """
        return jax.vmap(jax.vmap(self._element, (None, 0)), (0, None))(
            jnp.arange(self.n), jnp.arange(self.Lambda.n)
        )
