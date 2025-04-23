"""Boundary condition operators for differential forms on cube-like domains.

This module provides functionality for handling boundary conditions in the context of
differential forms discretized on cube-like domains. It supports both Dirichlet and
periodic boundary conditions for differential forms of various degrees (0-forms through
3-forms).

The main class, LazyBoundaryOperator, constructs boundary operators that can be applied
to differential form coefficients to enforce the specified boundary conditions. The
operator is represented as a sparse matrix that maps coefficients from the full space
to a reduced space that satisfies the specified boundary conditions.

Example:
    # Create a differential form with Dirichlet boundary conditions
    Λ = DifferentialForm(k=1, ns=(5,5,5), ps=(3,3,3), ('clamped', 'clamped', 'clamped'))
    
    # Create boundary operator with Dirichlet conditions
    B = LazyBoundaryOperator(Λ, ('dirichlet', 'dirichlet', 'dirichlet'))
    
    # Apply boundary conditions to coefficients
    coeffs_bc = B.M @ coeffs
"""

import jax
import jax.numpy as jnp
import numpy as np


# Bpundary extraction operator for cube-like domains


class LazyBoundaryOperator:
    """Boundary operator for differential forms on cube-like domains.

    This class constructs a boundary operator that enforces specified boundary conditions
    on differential forms. The operator is represented as a sparse matrix that maps
    coefficients from the full space to a reduced space satisfying the boundary conditions.

    The operator supports:
    - Dirichlet boundary conditions (zero at boundaries)
    - Periodic boundary conditions
    - Mixed boundary conditions (different conditions in different directions)

    For k-forms (k=0,1,2,3), the operator handles:
    - k=0: Scalar fields (zero at Dirichlet boundaries)
    - k=1: Vector fields (zero normal component at Dirichlet boundaries)
    - k=2: 2-forms (zero tangential components at Dirichlet boundaries)
    - k=3: 3-forms (volume forms)

    Attributes:
        k (int): Degree of the differential form (0,1,2,3)
        Λ (DifferentialForm): The differential form this operator acts on
        nr (int): Number of points in r-direction after boundary conditions
        nχ (int): Number of points in χ-direction after boundary conditions
        nζ (int): Number of points in ζ-direction after boundary conditions
        dr (int): Derived quantity for r-direction
        dχ (int): Derived quantity for χ-direction
        dζ (int): Derived quantity for ζ-direction
        n1 (int): Number of basis functions for first component
        n2 (int): Number of basis functions for second component
        n3 (int): Number of basis functions for third component
        n (int): Total number of basis functions after boundary conditions
        M (jax.Array): The assembled boundary operator matrix
    """

    def __init__(self, Λ, types):
        """Initialize the boundary operator.

        Args:
            Λ (DifferentialForm): The differential form this operator acts on
            types (tuple): Boundary condition types for each direction (r,χ,ζ).
                Each element can be:
                - 'dirichlet': Zero boundary conditions at 0 and 1
                - anything else: No boundary conditions (e.g., periodic)
        """
        # types can be:
        # - 'dirichlet' (zero at 0 and 1)
        # - else (no boundary conditions)
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

    def __getitem__(self, i):
        """Get an element of the boundary operator matrix.

        Args:
            i: Index into the matrix

        Returns:
            The matrix element at index i
        """
        return self.M[i]

    def __array__(self):
        """Convert the boundary operator matrix to a numpy array.

        Returns:
            numpy.ndarray: The boundary operator matrix as a numpy array
        """
        return np.array(self.M)

    def _vector_index(self, i):
        """Convert a flat index to a (component, index) pair.

        For k-forms with k>0, this splits the flat index into the component
        number (0,1,2) and the index within that component's basis functions.

        Args:
            i (int): Flat index into the basis functions

        Returns:
            tuple: (component_number, index_within_component)
        """
        if self.k == 0 or self.k == 3:
            return 0, i
        elif self.k == 1 or self.k == 2:
            n1, n2 = self.n1, self.n2
            category = jnp.int32(i >= n1) + jnp.int32(i >= n1 + n2)
            index = i - n1 * jnp.int32(i >= n1) - n2 * jnp.int32(i >= n1 + n2)
            return category, index

    def _unravel_index(self, idx):
        """Convert a flat index to (component, i, j, k) coordinates.

        This converts a flat index into the basis functions to a tuple of:
        - component number (for k>0)
        - i,j,k coordinates in the r,χ,ζ directions

        Args:
            idx (int): Flat index into the basis functions

        Returns:
            tuple: (component, i, j, k) coordinates
        """
        if self.k == 0:
            return 0, *jnp.unravel_index(idx, (self.nr, self.nχ, self.nζ))
        elif self.k == 1:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.dr, self.nχ, self.nζ))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(ijk, (self.nr, self.dχ, self.nζ))),
                    jnp.array(jnp.unravel_index(ijk, (self.nr, self.nχ, self.dζ)))
                )
            )
            return c, i, j, k
        elif self.k == 2:
            c, ijk = self._vector_index(idx)
            i, j, k = jnp.where(
                c == 0,
                jnp.array(jnp.unravel_index(ijk, (self.nr, self.dχ, self.dζ))),
                jnp.where(
                    c == 1,
                    jnp.array(jnp.unravel_index(ijk, (self.dr, self.nχ, self.dζ))),
                    jnp.array(jnp.unravel_index(ijk, (self.dr, self.dχ, self.nζ)))
                )
            )
            return c, i, j, k
        elif self.k == 3:
            return 0, *jnp.unravel_index(idx, (self.dr, self.dχ, self.dζ))

    def _element(self, row_idx, col_idx):
        """Compute an element of the boundary operator matrix.

        This computes the (row_idx,col_idx) element of the boundary operator matrix, which maps
        coefficients from the full space (col_idx) to the reduced space (row_idx). The computation
        depends on:
        - The degree k of the form
        - The boundary conditions in each direction
        - The component (for k>0)

        Args:
            row_idx (int): Row index (reduced space)
            col_idx (int): Column index (full space)

        Returns:
            int: 1 if basis function col_idx contributes to reduced basis function row_idx, 0 otherwise
        """
        cI, i, j, k = self._unravel_index(row_idx)
        cJ, r_idx, m, n = self.Λ._unravel_index(col_idx)
        if self.k == 0:
            # for example: dirichlet boundary condition in r and ζ:
            # row_idx ∈ [0, (nr-2) nχ (nζ-2)]
            # col_idx ∈ [0,   nr   nχ   nζ  ]
            return (
                (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == r_idx)
                    + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == r_idx-1))
                * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                   + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                   + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1))
            )
        elif self.k == 1:
            # for the x-component, it is dr x nχ x nζ
            return jnp.where(cI == cJ,
                             jnp.where(cI == 0,
                                       jnp.int32(i == r_idx)
                                       * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                                          + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                                       * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                                           + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1)),
                                       # for the y-component, it is nr x dχ x nζ
                                       jnp.where(cI == 1,
                                                 (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == r_idx)
                                                  + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == r_idx-1))
                                                 * jnp.int32(j == m)
                                                 * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                                                     + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1)),
                                                 # for the z-component, it is nr x nχ x dζ
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
            # for the x-component, it is nr x dχ x dζ
            return jnp.where(cI == cJ,
                             jnp.where(cI == 0,
                                       (jnp.int32(self.nr == self.Λ.nr) * jnp.int32(i == r_idx)
                                        + jnp.int32(self.nr != self.Λ.nr) * jnp.int32(i == r_idx-1))
                                       * jnp.int32(j == m)
                                       * jnp.int32(k == n),
                                       # for the y-component, it is dr x nχ x dζ
                                       jnp.where(cI == 1,
                                                 jnp.int32(i == r_idx)
                                                 * (jnp.int32(self.nχ == self.Λ.nχ) * jnp.int32(j == m)
                                                    + jnp.int32(self.nχ != self.Λ.nχ) * jnp.int32(j == m-1))
                                                 * jnp.int32(k == n),
                                                 # for the z-component, it is nr x nχ x dζ
                                                 jnp.int32(i == r_idx)
                                                 * jnp.int32(j == m)
                                                 * (jnp.int32(self.nζ == self.Λ.nζ) * jnp.int32(k == n)
                                                     + jnp.int32(self.nζ != self.Λ.nζ) * jnp.int32(k == n-1))
                                                 )
                                       ),
                             0
                             )
        elif self.k == 3:
            # Handle 3-forms (volume forms)
            # Identity mapping as 3-forms are already in reduced space
            return jnp.int32(row_idx == col_idx)

    def assemble(self):
        """Assemble the boundary operator matrix.

        This constructs the full boundary operator matrix by computing all elements
        using _element(). The result is a sparse matrix that maps coefficients from
        the full space to the reduced space.

        Returns:
            jax.Array: The assembled boundary operator matrix
        """
        return jax.vmap(jax.vmap(self._element, (None, 0)), (0, None))(jnp.arange(self.n), jnp.arange(self.Λ.n))
