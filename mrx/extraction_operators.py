"""
Polar mapping utilities for finite element analysis.

This module provides classes and functions for handling polar coordinate transformations
and boundary conditions in finite element computations.
"""

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

import mrx


class MatrixFreeExtraction(eqx.Module):
    """Matrix-free polar/boundary extraction operator.

    Applies ``E`` (forward) and ``E^T`` (transpose) as a cached
    gather/scatter using a static sparsity pattern instead of a stored BCSR
    matmul. The forward operator maps a full pre-extraction DoF vector (size
    ``forward_shape[1]``) to the extracted/constrained vector (size
    ``forward_shape[0]``); the transpose maps back.

    The index pattern (``rows``, ``cols``) and weights (``vals``) are computed
    once from the assembled sparse operator. The same pattern is reused by the
    surgery preconditioner through :meth:`to_bcoo`, so no BCSR needs to be
    materialised or stored for the matvec path.

    ``rows``/``cols``/``vals`` are always stored in the *forward* orientation;
    the :attr:`transposed` flag selects how they are consumed.
    """

    rows: jnp.ndarray
    cols: jnp.ndarray
    vals: jnp.ndarray
    forward_shape: tuple = eqx.field(static=True)
    transposed: bool = eqx.field(static=True)

    @classmethod
    def from_bcoo(cls, bcoo, transposed: bool = False):
        """Build a matrix-free extraction from an assembled BCOO matrix."""
        indices = bcoo.indices
        return cls(
            rows=jnp.asarray(indices[:, 0], dtype=jnp.int32),
            cols=jnp.asarray(indices[:, 1], dtype=jnp.int32),
            vals=jnp.asarray(bcoo.data, dtype=jnp.float64),
            forward_shape=(int(bcoo.shape[0]), int(bcoo.shape[1])),
            transposed=transposed,
        )

    @property
    def shape(self):
        if self.transposed:
            return (self.forward_shape[1], self.forward_shape[0])
        return self.forward_shape

    @property
    def dtype(self):
        return self.vals.dtype

    @property
    def data(self):
        """Nonzero values in the current orientation (BCOO-compatible)."""
        return self.vals

    @property
    def indices(self):
        """``(nnz, 2)`` COO indices in the current orientation."""
        if self.transposed:
            return jnp.stack([self.cols, self.rows], axis=1)
        return jnp.stack([self.rows, self.cols], axis=1)

    @property
    def T(self):
        return MatrixFreeExtraction(
            rows=self.rows,
            cols=self.cols,
            vals=self.vals,
            forward_shape=self.forward_shape,
            transposed=not self.transposed,
        )

    def _apply(self, x):
        x = jnp.asarray(x)
        if self.transposed:
            # E^T: gather from extracted rows, scatter into raw cols.
            gather_idx, segment_idx, num_segments = (
                self.rows, self.cols, self.forward_shape[1])
        else:
            # E: gather from raw cols, scatter into extracted rows.
            gather_idx, segment_idx, num_segments = (
                self.cols, self.rows, self.forward_shape[0])
        weights = self.vals if x.ndim == 1 else self.vals[:, None]
        contributions = weights * x[gather_idx]
        return jax.ops.segment_sum(
            contributions, segment_idx, num_segments=num_segments)

    def __matmul__(self, x):
        return self._apply(x)

    def __call__(self, x):
        return self._apply(x)

    def to_bcoo(self):
        """Materialise the (orientation-aware) sparse pattern as a BCOO."""
        if self.transposed:
            indices = jnp.stack([self.cols, self.rows], axis=1)
            shape = (self.forward_shape[1], self.forward_shape[0])
        else:
            indices = jnp.stack([self.rows, self.cols], axis=1)
            shape = self.forward_shape
        return jsparse.BCOO((self.vals, indices), shape=shape)

    def todense(self):
        return self.to_bcoo().todense()

    def restrict_rows(self, row_indices):
        """Return a copy with the row dimension restricted to ``row_indices``.

        Works in the current orientation (respects ``transposed``). The result
        keeps only nonzeros whose row (in current orientation) falls in
        ``row_indices``, with rows remapped to a contiguous 0-based range.
        Returns a new :class:`MatrixFreeExtraction` — no BCOO materialised.
        """
        row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
        n_new = int(row_indices.shape[0])
        # "Row in current orientation" lives in self.cols if transposed, else self.rows.
        if self.transposed:
            src = self.cols
            n_old = self.forward_shape[1]
        else:
            src = self.rows
            n_old = self.forward_shape[0]
        remap = jnp.full((n_old,), -1, dtype=jnp.int32)
        remap = remap.at[row_indices].set(jnp.arange(n_new, dtype=jnp.int32))
        new_src = remap[src]
        mask = new_src >= 0
        new_vals = self.vals[mask]
        if self.transposed:
            return MatrixFreeExtraction(
                rows=self.rows[mask],
                cols=new_src[mask],
                vals=new_vals,
                forward_shape=(self.forward_shape[0], n_new),
                transposed=True,
            )
        else:
            return MatrixFreeExtraction(
                rows=new_src[mask],
                cols=self.cols[mask],
                vals=new_vals,
                forward_shape=(n_new, self.forward_shape[1]),
                transposed=False,
            )

    def restrict_cols(self, col_indices):
        """Return a copy with the column dimension restricted to ``col_indices``.

        Works in the current orientation (respects ``transposed``). The result
        keeps only nonzeros whose column (in current orientation) falls in
        ``col_indices``, with columns remapped to a contiguous 0-based range.
        Returns a new :class:`MatrixFreeExtraction` — no BCOO materialised.
        """
        col_indices = jnp.asarray(col_indices, dtype=jnp.int32)
        n_new = int(col_indices.shape[0])
        # "Col in current orientation" lives in self.rows if transposed, else self.cols.
        if self.transposed:
            src = self.rows
            n_old = self.forward_shape[0]
        else:
            src = self.cols
            n_old = self.forward_shape[1]
        remap = jnp.full((n_old,), -1, dtype=jnp.int32)
        remap = remap.at[col_indices].set(jnp.arange(n_new, dtype=jnp.int32))
        new_src = remap[src]
        mask = new_src >= 0
        new_vals = self.vals[mask]
        if self.transposed:
            return MatrixFreeExtraction(
                rows=new_src[mask],
                cols=self.cols[mask],
                vals=new_vals,
                forward_shape=(n_new, self.forward_shape[1]),
                transposed=True,
            )
        else:
            return MatrixFreeExtraction(
                rows=self.rows[mask],
                cols=new_src[mask],
                vals=new_vals,
                forward_shape=(self.forward_shape[0], n_new),
                transposed=False,
            )


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

    def _k1_row_slices(self):
        theta_surgery = slice(0, 2 * self.nz)
        zeta_surgery = slice(theta_surgery.stop, theta_surgery.stop + 3 * self.dz)
        r_slice = slice(zeta_surgery.stop, zeta_surgery.stop + (self.dr - 1) * self.nt * self.nz)
        theta_bulk = slice(r_slice.stop, r_slice.stop + (self.nr - 2 - self.o) * self.dt * self.nz)
        zeta_bulk = slice(theta_bulk.stop, theta_bulk.stop + (self.nr - 2 - self.o) * self.nt * self.dz)
        return {
            "theta_surgery": theta_surgery,
            "zeta_surgery": zeta_surgery,
            "r": r_slice,
            "theta_bulk": theta_bulk,
            "zeta_bulk": zeta_bulk,
        }

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
            slices = self._k1_row_slices()
            cat_col, col_idx = self.Lambda._vector_index(col_idx)
            return jnp.where(
                row_idx < slices["theta_surgery"].stop,
                self.inner_oneform_r(
                    row_idx, col_idx, self.dr, self.nt, self.nz
                )
                * jnp.int32(cat_col == 0)
                + self.inner_oneform_θ(
                    row_idx, col_idx, self.nr, self.dt, self.nz
                )
                * jnp.int32(cat_col == 1),
                jnp.where(
                    row_idx < slices["zeta_surgery"].stop,
                    self._inner_zeroform(
                        row_idx - slices["theta_surgery"].stop, col_idx, self.nr, self.nt, self.dz
                    )
                    * jnp.int32(cat_col == 2),
                    jnp.where(
                        row_idx < slices["r"].stop,
                        self._threeform(
                            row_idx - slices["zeta_surgery"].stop, col_idx, self.dr, self.nt, self.nz
                        )
                        * jnp.int32(cat_col == 0),
                        jnp.where(
                            row_idx < slices["theta_bulk"].stop,
                            self._outer_zeroform(
                                row_idx - slices["r"].stop, col_idx, self.nr, self.dt, self.nz
                            )
                            * jnp.int32(cat_col == 1),
                            self._outer_zeroform(
                                row_idx - slices["theta_bulk"].stop, col_idx, self.nr, self.nt, self.dz
                            )
                            * jnp.int32(cat_col == 2),
                        ),
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

    def build_extraction(self):
        """Build the MatrixFreeExtraction from the explicit tensor-product sparsity pattern."""
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
            slices = self._k1_row_slices()
            theta_offset = slices["theta_surgery"].start
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

            zeta_offset = slices["zeta_surgery"].start
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

            r_offset = slices["r"].start
            for i in range(self.dr - 1):
                for j in range(self.nt):
                    for k in range(self.nz):
                        row_idx = r_offset + np.ravel_multi_index(
                            (i, j, k), (self.dr - 1, self.nt, self.nz)
                        )
                        col_idx = self._lambda_col_index(0, i + 1, j, k)
                        self._append_triplets(
                            rows, cols, data, row_idx=row_idx, col_idx=[col_idx], values=[1.0]
                        )

            radial = self.nr - 2 - self.o
            theta_outer_offset = slices["theta_bulk"].start
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

            zeta_outer_offset = slices["zeta_bulk"].start
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
            rows_arr = jnp.asarray(np.concatenate(rows), dtype=jnp.int32)
            cols_arr = jnp.asarray(np.concatenate(cols), dtype=jnp.int32)
            vals_arr = jnp.asarray(np.concatenate(data), dtype=jnp.float64)
        else:
            rows_arr = jnp.zeros((0,), dtype=jnp.int32)
            cols_arr = jnp.zeros((0,), dtype=jnp.int32)
            vals_arr = jnp.zeros((0,), dtype=jnp.float64)
        return MatrixFreeExtraction(
            rows=rows_arr, cols=cols_arr, vals=vals_arr,
            forward_shape=(self.n, self.Lambda.n),
            transposed=False,
        )


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

    def build_extraction(self):
        """Build the MatrixFreeExtraction by probing each row against all columns.

        Maps over rows sequentially, computing one row at a time with
        batched map over columns. Non-zero indices and values are collected
        into a MatrixFreeExtraction (gather/scatter apply, no matrix stored).
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
        return MatrixFreeExtraction(
            rows=row_indices.ravel().astype(jnp.int32),
            cols=all_cols.ravel().astype(jnp.int32),
            vals=all_vals.ravel(),
            forward_shape=(nrows, ncols),
            transposed=False,
        )


def bc_extraction_op(
    e,
    e_dbc,
    n_full: int,
):
    """Build the extraction operator for Dirichlet boundary DOFs.

    Returns a :class:`MatrixFreeExtraction` of shape ``(n_bc, n_full)`` that
    selects the DOFs present in ``e`` (unrestricted) but absent from ``e_dbc``
    (DBC), i.e. the DOFs that are set to zero by the homogeneous Dirichlet BC.

    Uses the identity: columns present in e but not e_dbc satisfy
        (e.T @ 1  -  e_dbc.T @ 1)[i] == 1
    """
    indicator = np.array(
        e.T @ jnp.ones(e.shape[0])
        - e_dbc.T @ jnp.ones(e_dbc.shape[0])
    )
    bc_cols = np.where(indicator > 0.5)[0]
    n_bc = len(bc_cols)
    return MatrixFreeExtraction(
        rows=jnp.asarray(np.arange(n_bc, dtype=np.int32)),
        cols=jnp.asarray(bc_cols.astype(np.int32)),
        vals=jnp.ones(n_bc, dtype=jnp.float64),
        forward_shape=(n_bc, n_full),
        transposed=False,
    )
