"""
Polar mapping utilities for finite element analysis.

This module provides classes and functions for handling polar coordinate transformations
and boundary conditions in finite element computations.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mrx


class MatrixFreeExtraction(eqx.Module):
    """Indexed linear operator applied as gather + ``segment_sum``.

    Holds the COO triplets ``(rows, cols, vals)`` of a sparse matrix and
    applies it (forward) or its transpose: the forward apply maps a vector 
    of size ``forward_shape[1]`` to one of size ``forward_shape[0]``. 
    Used for the polar/boundary extraction ``E``and for the analytic polar 
    grad/curl stencils.

    ``rows``/``cols``/``vals`` are always stored in the *forward* orientation;
    the :attr:`transposed` flag selects how they are consumed. Duplicate
    ``(row, col)`` entries are summed by the apply.
    """

    rows: jnp.ndarray
    cols: jnp.ndarray
    vals: jnp.ndarray
    forward_shape: tuple = eqx.field(static=True)
    transposed: bool = eqx.field(static=True)

    @classmethod
    def from_coo(cls, rows, cols, vals, shape):
        """Build the operator from host COO triplets and a ``(n_row, n_col)`` shape."""
        return cls(
            rows=jnp.asarray(np.asarray(rows, dtype=np.int32)),
            cols=jnp.asarray(np.asarray(cols, dtype=np.int32)),
            vals=jnp.asarray(np.asarray(vals), dtype=mrx.DTYPE),
            forward_shape=(int(shape[0]), int(shape[1])),
            transposed=False,
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
        # xi is (3, 2, nt): C¹ at the pole, rings 0-1 -> 3 polar functions.
        self.n_polar, self.ring_depth = 3, 2
        self.nr, self.nt, self.nz = Lambda.nr, Lambda.nt, Lambda.nz
        self.dr, self.dt, self.dz = Lambda.dr, Lambda.dt, Lambda.dz
        self.o = 1 if zero_bc else 0  # offset for boundary conditions

        # Set component sizes based on form degree
        if self.k == 0:
            self.n1 = ((self.nr - self.ring_depth - self.o) * self.nt
                       + self.n_polar) * self.nz
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
        raise ValueError(f"Unsupported form degree k={self.k}")

    def _append_bulk_selector(self, rows, cols, data, *, row_offset, row_shape,
                              component, i_offset):
        """Append the identity block that copies the bulk DOFs ``(i + i_offset, j, k)``
        of ``component`` onto the rows ``row_offset + ravel(i, j, k)``."""
        i, j, k = (ax.ravel() for ax in np.indices(row_shape))
        n = i.shape[0]
        rows.append((row_offset + np.arange(n)).astype(np.int32))
        cols.append(np.asarray(self._lambda_col_index(
            component, i + i_offset, j, k), dtype=np.int32))
        data.append(np.ones(n, dtype=np.float64))

    def build_extraction(self):
        """Build the MatrixFreeExtraction from the explicit tensor-product sparsity pattern."""
        xi = np.asarray(self.ξ)
        rows = []
        cols = []
        data = []

        if self.k == 0:
            for p in range(self.n_polar):
                for m in range(self.nz):
                    row_idx = np.ravel_multi_index((p, m), (self.n_polar, self.nz))
                    js = np.arange(self.nt, dtype=np.int32)
                    for i in range(self.ring_depth):
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

            radial = self.nr - self.ring_depth - self.o
            self._append_bulk_selector(
                rows, cols, data, row_offset=self.n_polar * self.nz,
                row_shape=(radial, self.nt, self.nz), component=0,
                i_offset=self.ring_depth)

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

            radial = self.nr - 2 - self.o
            self._append_bulk_selector(
                rows, cols, data, row_offset=slices["r"].start,
                row_shape=(self.dr - 1, self.nt, self.nz), component=0,
                i_offset=1)
            self._append_bulk_selector(
                rows, cols, data, row_offset=slices["theta_bulk"].start,
                row_shape=(radial, self.dt, self.nz), component=1,
                i_offset=2)
            self._append_bulk_selector(
                rows, cols, data, row_offset=slices["zeta_bulk"].start,
                row_shape=(radial, self.nt, self.dz), component=2,
                i_offset=2)

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
            self._append_bulk_selector(
                rows, cols, data, row_offset=2 * self.dz,
                row_shape=(radial, self.dt, self.dz), component=0,
                i_offset=2)
            self._append_bulk_selector(
                rows, cols, data, row_offset=self.n1,
                row_shape=(self.dr - 1, self.nt, self.dz), component=1,
                i_offset=1)
            self._append_bulk_selector(
                rows, cols, data, row_offset=self.n1 + self.n2,
                row_shape=(self.dr - 1, self.dt, self.nz), component=2,
                i_offset=1)

        elif self.k == 3:
            self._append_bulk_selector(
                rows, cols, data, row_offset=0,
                row_shape=(self.dr - 1, self.dt, self.dz), component=0,
                i_offset=1)
        else:
            raise ValueError(f"Sparse tensor assembly is not implemented for k={self.k}")

        if data:
            rows_arr = jnp.asarray(np.concatenate(rows), dtype=jnp.int32)
            cols_arr = jnp.asarray(np.concatenate(cols), dtype=jnp.int32)
            vals_arr = jnp.asarray(np.concatenate(data), dtype=mrx.DTYPE)
        else:
            rows_arr = jnp.zeros((0,), dtype=jnp.int32)
            cols_arr = jnp.zeros((0,), dtype=jnp.int32)
            vals_arr = jnp.zeros((0,), dtype=mrx.DTYPE)
        return MatrixFreeExtraction(
            rows=rows_arr, cols=cols_arr, vals=vals_arr,
            forward_shape=(self.n, self.Lambda.n),
            transposed=False,
        )


def get_xi(nt):
    """Polar extraction weights ξ^ℓ_{ij}: barycentric coordinates of the
    first two control rings with respect to the equilateral control
    triangle (Toshniwal et al. CMAME 2017; Holderied thesis Eqs. 5.7–5.9).

    Ring 0 sits at the pole (triangle centroid) → weights 1/3. Ring 1 gets
    the barycentric coordinates of the unit-circle points
    ``(cos θ_j, sin θ_j)``: the logical-disk specialisation, which is exact
    whenever ``∂F/∂r`` at the axis is a pure ``m = ±1`` mode. A constant of
    ``nt``; the sequence builds it once.

    Parameters
    ----------
    nt : int
        Number of points in poloidal θ-direction.

    Returns
    -------
    ξ : jnp.ndarray
        Polar extraction weights, shape ``(3, 2, nθ)`` indexed ``(ℓ, i, j)``.
    """
    theta_js = (jnp.arange(nt) / nt) * 2 * jnp.pi
    dR, dY = jnp.cos(theta_js), jnp.sin(theta_js)

    s3 = jnp.sqrt(3.0)
    tau = jnp.max(jnp.array([jnp.max(-2.0 * dR),
                             jnp.max(dR - s3 * dY),
                             jnp.max(dR + s3 * dY)]))
    ξ1 = jnp.stack([1/3 + 2.0 * dR / (3.0 * tau),
                    1/3 - dR / (3.0 * tau) + s3 * dY / (3.0 * tau),
                    1/3 - dR / (3.0 * tau) - s3 * dY / (3.0 * tau)])  # (3, nθ)
    ξ0 = jnp.full((3, nt), 1.0 / 3.0)
    # (3, 2, nθ) -> l, i, j
    return jnp.stack([ξ0, ξ1], axis=1)


# Boundary extraction operator for cube-like domains
def bc_extraction_op(
    e,
    e_dbc,
    n_full: int,
):
    """Build the extraction operator for Dirichlet boundary DOFs.

    Returns a :class:`MatrixFreeExtraction` of shape ``(n_bc, n_full)`` that
    selects the DOFs present in ``e`` (unrestricted) but absent from ``e_dbc``
    (DBC), i.e. the DOFs that are set to zero by the homogeneous Dirichlet BC.
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
        vals=jnp.ones(n_bc, dtype=mrx.DTYPE),
        forward_shape=(n_bc, n_full),
        transposed=False,
    )
