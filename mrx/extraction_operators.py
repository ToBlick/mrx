"""
Polar mapping utilities for finite element analysis.

This module provides classes and functions for handling polar coordinate transformations
and boundary conditions in finite element computations.
"""

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

import mrx


class PolarExtractionOperator:
    """
    A class for extracting boundary conditions and handling polar mappings.

    This class implements operators for handling boundary conditions and polar
    coordinate transformations.

    Attributes:
        k (int): Degree of the differential form
        Λ: 
        xi: Polar mapping coefficients
        nr (int): Number of points in r-direction
        nt (int): Number of points in θ-direction
        nz (int): Number of points in ζ-direction
        dr (int): Number of points in r-direction after boundary conditions
        dt (int): Number of points in θ-direction after boundary conditions
        dz (int): Number of points in ζ-direction after boundary conditions
        o (int): Offset for boundary conditions (1 for zero BC, 0 otherwise)
        n1 (int): Size of first component
        n2 (int): Size of second component
        n3 (int): Size of third component
        n (int): Total size of the operator
    """

    def __init__(self, Lambda, xi, zero_bc):
        """
        Initialize the extraction operator.

        Args:
            Λ: Domain operator
            ξ: Polar mapping coefficients
            zero_bc (bool): Whether to apply zero boundary conditions
        """
        self.k = Lambda.k
        self.Lambda = Lambda
        self.ξ = xi
        self.nr, self.nt, self.nz = Lambda.nr, Lambda.nt, Lambda.nz
        self.dr, self.dt, self.dz = Lambda.dr, Lambda.dt, Lambda.dz
        self.o = 1 if zero_bc else 0  # offset for boundary conditions

        # Set component sizes based on form degree
        if self.k == 0:
            self.n1 = ((self.nr - 2 - self.o) * self.nt + 3) * self.nz
            self.n2 = 0
            self.n3 = 0
        if self.k == 1:
            self.n1 = (self.dr - 1) * self.nt * self.nz
            self.n2 = ((self.nr - 2 - self.o) * self.dt + 2) * self.nz
            self.n3 = ((self.nr - 2 - self.o) * self.nt + 3) * self.dz
        if self.k == 2:
            self.n1 = ((self.nr - 2 - self.o) * self.dt + 2) * self.dz
            self.n2 = (self.dr - 1) * self.nt * self.dz
            self.n3 = (self.dr - 1) * self.dt * self.nz
        if self.k == 3:
            self.n1 = (self.dr - 1) * self.dt * self.dz
            self.n2 = 0
            self.n3 = 0
        if self.k == -1:
            self.n1 = ((self.nr - 2 - self.o) * self.nt + 3) * self.nz
            self.n2 = ((self.nr - 2 - self.o) * self.nt + 3) * self.nz
            self.n3 = ((self.nr - 2 - self.o) * self.nt + 3) * self.nz
        self.n = self.n1 + self.n2 + self.n3

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
        elif self.k == 1 or self.k == 2 or self.k == -1:
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
                row_idx < 3 * self.nz,
                self._inner_zeroform(
                    row_idx, col_idx, self.nr, self.nt, self.nz),
                self._outer_zeroform(
                    row_idx - 3 * self.nz, col_idx, self.nr, self.nt, self.nz
                ),
            )
        if self.k == 1:
            # Handle 1-forms
            cat_row, row_idx = self._vector_index(row_idx)
            cat_col, col_idx = self.Lambda._vector_index(col_idx)
            return jnp.where(
                cat_row == 0,
                # r-component
                self._threeform(row_idx, col_idx, self.dr, self.nt, self.nz)
                * jnp.int32(cat_row == cat_col),
                jnp.where(
                    cat_row == 1,
                    # θ-component
                    jnp.where(
                        row_idx < 2 * self.nz,
                        self.inner_oneform_r(
                            row_idx, col_idx, self.dr, self.nt, self.nz
                        )
                        * jnp.int32(cat_col == 0)
                        + self.inner_oneform_θ(
                            row_idx, col_idx, self.nr, self.dt, self.nz
                        )
                        * jnp.int32(cat_col == 1),
                        self._outer_zeroform(
                            row_idx - 2 * self.nz, col_idx, self.nr, self.dt, self.nz
                        )
                        * jnp.int32(cat_row == cat_col),
                    ),
                    # ζ-component
                    jnp.where(
                        row_idx < 3 * self.dz,
                        self._inner_zeroform(
                            row_idx, col_idx, self.nr, self.nt, self.dz
                        )
                        * jnp.int32(cat_row == cat_col),
                        self._outer_zeroform(
                            row_idx - 3 * self.dz, col_idx, self.nr, self.nt, self.dz
                        )
                        * jnp.int32(cat_row == cat_col),
                    ),
                ),
            )
        if self.k == 2:
            # Handle 2-forms
            cat_row, row_idx = self._vector_index(row_idx)
            cat_col, col_idx = self.Lambda._vector_index(col_idx)
            return jnp.where(
                cat_row == 0,
                # r-component
                jnp.where(
                    row_idx < 2 * self.nz,
                    self.inner_oneform_θ(
                        row_idx, col_idx, self.nr, self.dt, self.dz)
                    * jnp.int32(cat_col == 0)
                    - self.inner_oneform_r(row_idx, col_idx,
                                           self.dr, self.nt, self.dz)
                    * jnp.int32(cat_col == 1),
                    self._outer_zeroform(
                        row_idx - 2 * self.nz, col_idx, self.nr, self.dt, self.dz
                    )
                    * jnp.int32(cat_row == cat_col),
                ),
                jnp.where(
                    cat_row == 1,
                    # θ-component
                    self._threeform(row_idx, col_idx,
                                    self.dr, self.nt, self.dz)
                    * jnp.int32(cat_row == cat_col),
                    # ζ-component
                    self._threeform(row_idx, col_idx,
                                    self.dr, self.dt, self.nz)
                    * jnp.int32(cat_row == cat_col),
                ),
            )
        if self.k == 3:
            # Handle 3-forms
            return self._threeform(row_idx, col_idx, self.nr, self.nt, self.nz)
        if self.k == -1:
            # Handle vector fields
            cat_row, row_idx = self._vector_index(row_idx)
            cat_col, col_idx = self.Lambda._vector_index(col_idx)
            return jnp.where(
                row_idx < 3 * self.nz,
                self._inner_zeroform(
                    row_idx, col_idx, self.nr, self.nt, self.nz),
                self._outer_zeroform(
                    row_idx - 3 * self.nz, col_idx, self.nr, self.nt, self.nz
                ),
            ) * jnp.int32(cat_row == cat_col)

    def _inner_zeroform(self, row_idx, col_idx, nr, nt, nz):
        """
        Compute inner zero-form basis function.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nt (int): Number of points in θ-direction
            nz (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        p, m = jnp.unravel_index(row_idx, (3, nz))
        i, j, k = jnp.unravel_index(col_idx, (nr, nt, nz))
        return jnp.int32(k == m) * jnp.int32(i < 2) * self.ξ[p, i, j]

    def _outer_zeroform(self, row_idx, col_idx, nr, nt, nz):
        """
        Compute outer zero-form basis function.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nt (int): Number of points in θ-direction
            nz (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        i, j, k = jnp.unravel_index(row_idx, (nr, nt, nz))
        return jnp.int32(
            col_idx == jnp.ravel_multi_index(
                (i + 2, j, k), (nr, nt, nz), mode="clip")
        ) * jnp.where(self.o == 1, jnp.int32(i != nr - 1), 1)

    def inner_oneform_r(self, row_idx, col_idx, nr, nt, nz):
        """
        Compute inner one-form basis function in r-direction.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nt (int): Number of points in θ-direction
            nz (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        p, m = jnp.unravel_index(row_idx, (2, nz))
        p += 1
        i, j, k = jnp.unravel_index(col_idx, (nr, nt, nz))
        return (
            jnp.int32(k == m) * jnp.int32(i == 0) *
            (self.ξ[p, 1, j] - self.ξ[p, 0, j])
        )

    def inner_oneform_θ(self, row_idx, col_idx, nr, nt, nz):
        """
        Compute inner one-form basis function in θ-direction.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nt (int): Number of points in θ-direction
            nz (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        p, m = jnp.unravel_index(row_idx, (2, nz))
        p += 1
        i, j, k = jnp.unravel_index(col_idx, (nr, nt, nz))
        return (
            jnp.int32(k == m)
            * jnp.int32(i == 1)
            * (self.ξ[p, 1, jnp.mod(j + 1, nt)] - self.ξ[p, 1, j])
        )

    def _threeform(self, row_idx, col_idx, nr, nt, nz):
        """
        Compute three-form basis function.

        Args:
            row_idx (int): Row index
            col_idx (int): Column index
            nr (int): Number of points in r-direction
            nt (int): Number of points in θ-direction
            nz (int): Number of points in ζ-direction

        Returns:
            jnp.ndarray: The basis function value
        """
        i, j, k = jnp.unravel_index(row_idx, (nr, nt, nz))
        return jnp.int32(
            col_idx == jnp.ravel_multi_index(
                (i + 1, j, k), (nr, nt, nz), mode="clip")
        )

    def _append_triplets(self, rows, cols, data, *, row_idx, col_idx, values):
        col_idx = np.asarray(col_idx, dtype=np.int32).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid = values != 0.0
        if not np.any(valid):
            return
        nnz = int(np.count_nonzero(valid))
        rows.append(np.full(nnz, row_idx, dtype=np.int32))
        cols.append(col_idx[valid])
        data.append(values[valid])

    def _lambda_col_index(self, component, i, j, k):
        if self.k == 0:
            return np.ravel_multi_index(
                (i, j, k),
                (self.Lambda.nr, self.Lambda.nt, self.Lambda.nz),
                mode="clip",
            )
        if self.k == 1:
            if component == 0:
                return np.ravel_multi_index(
                    (i, j, k),
                    (self.Lambda.dr, self.Lambda.nt, self.Lambda.nz),
                    mode="clip",
                )
            if component == 1:
                return self.Lambda.n1 + np.ravel_multi_index(
                    (i, j, k),
                    (self.Lambda.nr, self.Lambda.dt, self.Lambda.nz),
                    mode="clip",
                )
            return self.Lambda.n1 + self.Lambda.n2 + np.ravel_multi_index(
                (i, j, k),
                (self.Lambda.nr, self.Lambda.nt, self.Lambda.dz),
                mode="clip",
            )
        if self.k == 2:
            if component == 0:
                return np.ravel_multi_index(
                    (i, j, k),
                    (self.Lambda.nr, self.Lambda.dt, self.Lambda.dz),
                    mode="clip",
                )
            if component == 1:
                return self.Lambda.n1 + np.ravel_multi_index(
                    (i, j, k),
                    (self.Lambda.dr, self.Lambda.nt, self.Lambda.dz),
                    mode="clip",
                )
            return self.Lambda.n1 + self.Lambda.n2 + np.ravel_multi_index(
                (i, j, k),
                (self.Lambda.dr, self.Lambda.dt, self.Lambda.nz),
                mode="clip",
            )
        if self.k == 3:
            return np.ravel_multi_index(
                (i, j, k),
                (self.Lambda.dr, self.Lambda.dt, self.Lambda.dz),
                mode="clip",
            )
        if self.k == -1:
            base = np.ravel_multi_index(
                (i, j, k),
                (self.Lambda.nr, self.Lambda.nt, self.Lambda.nz),
                mode="clip",
            )
            if component == 0:
                return base
            if component == 1:
                return self.Lambda.n1 + base
            return self.Lambda.n1 + self.Lambda.n2 + base
        raise ValueError(f"Unsupported form degree k={self.k}")

    def assemble_sparse_tensor(self):
        """Assemble the operator from its explicit tensor-product sparsity pattern."""
        xi = np.asarray(self.ξ)
        rows = []
        cols = []
        data = []

        if self.k == 0:
            for p in range(3):
                for m in range(self.nz):
                    row_idx = np.ravel_multi_index((p, m), (3, self.nz))
                    js = np.arange(self.nt, dtype=np.int32)
                    for i in range(2):
                        col_idx = np.ravel_multi_index(
                            (np.full(self.nt, i, dtype=np.int32), js, np.full(self.nt, m, dtype=np.int32)),
                            (self.nr, self.nt, self.nz),
                            mode="clip",
                        )
                        self._append_triplets(
                            rows,
                            cols,
                            data,
                            row_idx=row_idx,
                            col_idx=col_idx,
                            values=xi[p, i, :],
                        )

            radial = self.nr - 2 - self.o
            outer_offset = 3 * self.nz
            for i in range(radial):
                for j in range(self.nt):
                    for k in range(self.nz):
                        row_idx = outer_offset + np.ravel_multi_index(
                            (i, j, k),
                            (radial, self.nt, self.nz),
                        )
                        col_idx = self._lambda_col_index(0, i + 2, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

        elif self.k == 1:
            for i in range(self.dr - 1):
                for j in range(self.nt):
                    for k in range(self.nz):
                        row_idx = np.ravel_multi_index(
                            (i, j, k), (self.dr - 1, self.nt, self.nz)
                        )
                        col_idx = self._lambda_col_index(0, i + 1, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

            theta_offset = self.n1
            for p_local in range(2):
                p = p_local + 1
                for m in range(self.nz):
                    row_idx = theta_offset + np.ravel_multi_index(
                        (p_local, m), (2, self.nz)
                    )
                    js_theta = np.arange(self.dt, dtype=np.int32)
                    col_theta = self.Lambda.n1 + np.ravel_multi_index(
                        (
                            np.full(self.dt, 1, dtype=np.int32),
                            js_theta,
                            np.full(self.dt, m, dtype=np.int32),
                        ),
                        (self.Lambda.nr, self.Lambda.dt, self.Lambda.nz),
                        mode="clip",
                    )
                    val_theta = xi[p, 1, np.mod(js_theta + 1, self.dt)] - xi[p, 1, js_theta]
                    self._append_triplets(
                        rows,
                        cols,
                        data,
                        row_idx=row_idx,
                        col_idx=col_theta,
                        values=val_theta,
                    )

                    js_r = np.arange(self.nt, dtype=np.int32)
                    col_r = np.ravel_multi_index(
                        (
                            np.zeros(self.nt, dtype=np.int32),
                            js_r,
                            np.full(self.nt, m, dtype=np.int32),
                        ),
                        (self.Lambda.dr, self.Lambda.nt, self.Lambda.nz),
                        mode="clip",
                    )
                    val_r = xi[p, 1, js_r] - xi[p, 0, js_r]
                    self._append_triplets(
                        rows,
                        cols,
                        data,
                        row_idx=row_idx,
                        col_idx=col_r,
                        values=val_r,
                    )

            radial = self.nr - 2 - self.o
            theta_outer_offset = theta_offset + 2 * self.nz
            for i in range(radial):
                for j in range(self.dt):
                    for k in range(self.nz):
                        row_idx = theta_outer_offset + np.ravel_multi_index(
                            (i, j, k), (radial, self.dt, self.nz)
                        )
                        col_idx = self._lambda_col_index(1, i + 2, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

            zeta_offset = self.n1 + self.n2
            for p in range(3):
                for m in range(self.dz):
                    row_idx = zeta_offset + np.ravel_multi_index(
                        (p, m), (3, self.dz)
                    )
                    js = np.arange(self.nt, dtype=np.int32)
                    for i in range(2):
                        col_idx = self.Lambda.n1 + self.Lambda.n2 + np.ravel_multi_index(
                            (
                                np.full(self.nt, i, dtype=np.int32),
                                js,
                                np.full(self.nt, m, dtype=np.int32),
                            ),
                            (self.Lambda.nr, self.Lambda.nt, self.Lambda.dz),
                            mode="clip",
                        )
                        self._append_triplets(
                            rows,
                            cols,
                            data,
                            row_idx=row_idx,
                            col_idx=col_idx,
                            values=xi[p, i, :],
                        )

            zeta_outer_offset = zeta_offset + 3 * self.dz
            for i in range(radial):
                for j in range(self.nt):
                    for k in range(self.dz):
                        row_idx = zeta_outer_offset + np.ravel_multi_index(
                            (i, j, k), (radial, self.nt, self.dz)
                        )
                        col_idx = self._lambda_col_index(2, i + 2, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

        elif self.k == 2:
            for p_local in range(2):
                p = p_local + 1
                for m in range(self.dz):
                    row_idx = np.ravel_multi_index((p_local, m), (2, self.dz))
                    js_theta = np.arange(self.dt, dtype=np.int32)
                    col_theta = np.ravel_multi_index(
                        (
                            np.full(self.dt, 1, dtype=np.int32),
                            js_theta,
                            np.full(self.dt, m, dtype=np.int32),
                        ),
                        (self.Lambda.nr, self.Lambda.dt, self.Lambda.dz),
                        mode="clip",
                    )
                    val_theta = xi[p, 1, np.mod(js_theta + 1, self.dt)] - xi[p, 1, js_theta]
                    self._append_triplets(
                        rows,
                        cols,
                        data,
                        row_idx=row_idx,
                        col_idx=col_theta,
                        values=val_theta,
                    )

                    js_r = np.arange(self.nt, dtype=np.int32)
                    col_r = self.Lambda.n1 + np.ravel_multi_index(
                        (
                            np.zeros(self.nt, dtype=np.int32),
                            js_r,
                            np.full(self.nt, m, dtype=np.int32),
                        ),
                        (self.Lambda.dr, self.Lambda.nt, self.Lambda.dz),
                        mode="clip",
                    )
                    val_r = -(xi[p, 1, js_r] - xi[p, 0, js_r])
                    self._append_triplets(
                        rows,
                        cols,
                        data,
                        row_idx=row_idx,
                        col_idx=col_r,
                        values=val_r,
                    )

            radial = self.nr - 2 - self.o
            comp0_outer_offset = 2 * self.dz
            for i in range(radial):
                for j in range(self.dt):
                    for k in range(self.dz):
                        row_idx = comp0_outer_offset + np.ravel_multi_index(
                            (i, j, k), (radial, self.dt, self.dz)
                        )
                        col_idx = self._lambda_col_index(0, i + 2, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

            comp1_offset = self.n1
            for i in range(self.dr - 1):
                for j in range(self.nt):
                    for k in range(self.dz):
                        row_idx = comp1_offset + np.ravel_multi_index(
                            (i, j, k), (self.dr - 1, self.nt, self.dz)
                        )
                        col_idx = self._lambda_col_index(1, i + 1, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

            comp2_offset = self.n1 + self.n2
            for i in range(self.dr - 1):
                for j in range(self.dt):
                    for k in range(self.nz):
                        row_idx = comp2_offset + np.ravel_multi_index(
                            (i, j, k), (self.dr - 1, self.dt, self.nz)
                        )
                        col_idx = self._lambda_col_index(2, i + 1, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

        elif self.k == 3:
            for i in range(self.dr - 1):
                for j in range(self.dt):
                    for k in range(self.dz):
                        row_idx = np.ravel_multi_index(
                            (i, j, k), (self.dr - 1, self.dt, self.dz)
                        )
                        col_idx = self._lambda_col_index(0, i + 1, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )
        else:
            raise ValueError(f"Sparse tensor assembly is not implemented for k={self.k}")

        if data:
            row_idx = jnp.asarray(np.concatenate(rows), dtype=jnp.int32)
            col_idx = jnp.asarray(np.concatenate(cols), dtype=jnp.int32)
            values = jnp.asarray(np.concatenate(data), dtype=jnp.float64)
            indices = jnp.stack([row_idx, col_idx], axis=-1)
        else:
            values = jnp.zeros((0,), dtype=jnp.float64)
            indices = jnp.zeros((0, 2), dtype=jnp.int32)
        return jsparse.BCOO((values, indices), shape=(self.n, self.Lambda.n))

    def assemble_sparse(self):
        """Assemble the operator as a sparse BCOO matrix."""
        return self.assemble_sparse_tensor()

    def sparse_matrix(self):
        """Wrapper for assemble_sparse."""
        return self.assemble_sparse()


def get_xi(nt):
    """
    Compute polar mapping coefficients.

    Parameters
    ----------
    nt : int
        Number of points in poloidal θ-direction.

    Returns
    -------
    ξ : jnp.ndarray
        Polar mapping coefficients.
        Shape: (3, 2, nθ)
    """
    theta_js = (jnp.arange(nt) / nt) * 2 * jnp.pi

    M = jnp.array([
        [1/3, 0],
        [-1/6, jnp.sqrt(3)/6],
        [-1/6, -jnp.sqrt(3)/6]
    ])

    cos_js = jnp.cos(theta_js)
    sin_js = jnp.sin(theta_js)

    Es = 1/3 + M @ jnp.array([cos_js, sin_js])  # shape (3, nθ)

    ξ00 = jnp.ones(nt) / 3
    ξ10 = jnp.ones(nt) / 3
    ξ20 = jnp.ones(nt) / 3

    ξ01 = Es[0]
    ξ11 = Es[1]
    ξ21 = Es[2]
    # (3, 2, nθ) -> l, i, j
    ξ = jnp.array([[ξ00, ξ01], [ξ10, ξ11], [ξ20, ξ21]])
    return ξ


# Boundary extraction operator for cube-like domains
class BoundaryOperator:
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
        return mrx.double_map(
            self._element,
            jnp.arange(self.n),
            jnp.arange(self.Lambda.n),
        )

    def assemble_sparse(self):
        """Assemble the operator as a sparse BCOO matrix.

        Maps over rows sequentially, computing one row at a time with
        batched map over columns. Non-zero indices and values are collected
        and assembled into a BCOO sparse matrix.

        Returns:
            jsparse.BCOO: The sparse operator matrix of shape (n, Lambda.n).
        """
        ncols = self.Lambda.n
        nrows = self.n
        col_indices = jnp.arange(ncols)

        # Each row has at most 1 non-zero (selection/permutation matrix)
        max_nnz = 1

        def process_row(row_idx):
            row = jax.lax.map(lambda col_idx: self._element(
                row_idx, col_idx), col_indices, batch_size=mrx.MAP_BATCH_SIZE_INNER)
            nz_mask = row != 0
            order = jnp.argsort(~nz_mask, stable=True)
            vals = row[order][:max_nnz]
            cols = col_indices[order][:max_nnz]
            nz_count = jnp.sum(nz_mask)
            valid = jnp.arange(max_nnz) < nz_count
            vals = jnp.where(valid, vals, 0.0)
            cols = jnp.where(valid, cols, 0)
            return vals, cols

        all_vals, all_cols = jax.lax.map(
            process_row, jnp.arange(nrows), batch_size=mrx.MAP_BATCH_SIZE_OUTER
        )  # (nrows, max_nnz)

        row_indices = jnp.broadcast_to(
            jnp.arange(nrows)[:, None], (nrows, max_nnz)
        )
        indices = jnp.stack([row_indices.ravel(), all_cols.ravel()], axis=-1)
        data = all_vals.ravel()

        return jsparse.BCOO((data, indices), shape=(nrows, ncols))

    def sparse_matrix(self):
        """Wrapper for assemble_sparse."""
        return self.assemble_sparse()


def bc_extraction_op(
    e_bcoo,
    e_dbc_bcoo,
    n_full: int,
):
    """Build the extraction operator for Dirichlet boundary DOFs.

    Returns a BCOO matrix of shape (n_bc, n_full) that selects the
    DOFs present in ``e`` (unrestricted) but absent from ``e_dbc`` (DBC),
    i.e. the DOFs that are set to zero by the homogeneous Dirichlet BC.

    Uses the identity: columns present in e but not e_dbc satisfy
        (e.T @ 1  -  e_dbc.T @ 1)[i] == 1
    """
    indicator = np.array(
        e_bcoo.T @ jnp.ones(e_bcoo.shape[0])
        - e_dbc_bcoo.T @ jnp.ones(e_dbc_bcoo.shape[0])
    )
    bc_cols = np.where(indicator > 0.5)[0]
    n_bc = len(bc_cols)
    if n_bc == 0:
        return jsparse.BCOO(
            (jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)),
            shape=(0, n_full))
    indices = jnp.array(np.stack([np.arange(n_bc), bc_cols], axis=-1))
    return jsparse.BCOO((jnp.ones(n_bc), indices), shape=(n_bc, n_full))
