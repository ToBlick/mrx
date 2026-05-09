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
import jax.numpy as jnp
import numpy as np
from jax import Array

import mrx
from mrx.extraction_operators import get_xi
from mrx.utils import integrate_against, inv33

if TYPE_CHECKING:
    from mrx.derham_sequence import DeRhamSequence


# Type aliases for callable functions used in projections
ScalarFunction = Callable[[Array], Array]  # ξ -> scalar (with trailing dim)
VectorFunction = Callable[[Array], Array]  # ξ -> 3D vector


def _as_single_component(values: Array) -> Array:
    """Normalize a scalar or length-1 array to shape (1,)."""
    return jnp.reshape(jnp.asarray(values), (1,))


def _solve_tensor_collocation_axis(matrix: Array, values: Array, axis: int) -> Array:
    """Solve one tensor-product collocation axis against a square 1D matrix."""
    moved = jnp.moveaxis(values, axis, 0)
    solved = jnp.linalg.solve(matrix, moved.reshape(matrix.shape[0], -1))
    return jnp.moveaxis(solved.reshape(moved.shape), 0, axis)


def _apply_tensor_operator_axis(matrix: Array, values: Array, axis: int) -> Array:
    """Apply one tensor-product operator axis against a dense 1D matrix."""
    moved = jnp.moveaxis(values, axis, 0)
    applied = matrix @ moved.reshape(matrix.shape[1], -1)
    out_shape = (matrix.shape[0],) + moved.shape[1:]
    return jnp.moveaxis(applied.reshape(out_shape), 0, axis)


def _leggauss_rule(order: int) -> tuple[Array, Array]:
    xi, w = np.polynomial.legendre.leggauss(order)
    return jnp.asarray(xi), jnp.asarray(w)


def _interval_rule(span: Array, order: int) -> tuple[Array, Array]:
    xi_ref, w_ref = _leggauss_rule(order)
    a, b = span
    center = 0.5 * (a + b)
    halfwidth = 0.5 * (b - a)
    return center + halfwidth * xi_ref, halfwidth * w_ref


def _quadrature_order_from_basis_1d(basis) -> int:
    return max(2, basis.p + 2)


def _require_full_tensor_space(extraction: Array, full_size: int, label: str) -> None:
    if extraction.shape[0] != full_size:
        raise NotImplementedError(
            f"{label} is currently implemented only for full tensor-product "
            "spaces without nontrivial extraction."
        )


def _require_clamped_histopolation(d_basis, label: str) -> None:
    if d_basis.type != 'clamped':
        raise NotImplementedError(
            f"{label} currently supports only clamped Greville histopolation axes."
        )


def _is_polar_zeroform_shape(seq: "DeRhamSequence", size: int, *, dirichlet: bool) -> bool:
    nr, nt, nz = seq.basis_0.nr, seq.basis_0.nt, seq.basis_0.nz
    expected = ((nr - 3) if dirichlet else (nr - 2)) * nt * nz + 3 * nz
    return size == expected


def _is_polar_oneform_shape(seq: "DeRhamSequence", size: int, *, dirichlet: bool) -> bool:
    nr, nt, nz = seq.basis_1.nr, seq.basis_1.nt, seq.basis_1.nz
    dr, dt, dz = seq.basis_1.dr, seq.basis_1.dt, seq.basis_1.dz
    offset = 1 if dirichlet else 0
    expected = (
        (dr - 1) * nt * nz
        + ((nr - 2 - offset) * dt + 2) * nz
        + ((nr - 2 - offset) * nt + 3) * dz
    )
    return size == expected


def _matching_discrete_dofs(f, basis, extraction) -> Array | None:
    """Return coefficients when ``f`` is already represented in the target space."""
    dof = getattr(f, 'dof', None)
    form = getattr(f, 'Λ', None)
    transform = getattr(f, 'E', None)
    if dof is None or form is not basis:
        return None
    if transform is None:
        return None
    if transform is extraction:
        return jnp.asarray(dof)
    return None


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
    bc: bool = False

    def __init__(self, seq: DeRhamSequence, k: Literal[0, 1, 2, 3], dirichlet: bool = True, bc: bool = False) -> None:
        """
        Initialize the projector.

        Args:
            seq : DeRham sequence object
            k : Degree of the differential form
            dirichlet : Whether to use dirichlet boundary conditions
            bc : If True, project onto the Dirichlet boundary DOFs only
                 (uses e_k_bc instead of e_k or e_k_dbc).
                 Takes precedence over `dirichlet`.
        """
        self.k = k
        self.seq = seq
        self.dirichlet = dirichlet
        self.bc = bc

    def __call__(self, f: ScalarFunction | VectorFunction) -> Array:
        """
        Project a function onto the finite element space.

        Args:
            f (callable): Function to project

        Returns:
            array: Projection coefficients
        """

        if self.k == 0:
            if self.bc:
                e = self.seq.e0_bc
            elif self.dirichlet:
                e = self.seq.e0_dbc
            else:
                e = self.seq.e0
            return e @ self.zeroform_projection(f)
        elif self.k == 1:
            if self.bc:
                e = self.seq.e1_bc
            elif self.dirichlet:
                e = self.seq.e1_dbc
            else:
                e = self.seq.e1
            return e @ self.oneform_projection(f)
        elif self.k == 2:
            if self.bc:
                e = self.seq.e2_bc
            elif self.dirichlet:
                e = self.seq.e2_dbc
            else:
                e = self.seq.e2
            return e @ self.twoform_projection(f)
        elif self.k == 3:
            if self.bc:
                e = self.seq.e3_bc
            elif self.dirichlet:
                e = self.seq.e3_dbc
            else:
                e = self.seq.e3
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
            lambda x: _as_single_component(f(x)),
            self.seq.quad.x,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        )
        w_jk: Array = f_jk * (self.seq.quad.w * self.seq.jacobian_j)[:, None]
        comp_info, comp_shapes = self.seq._form_comp_info(0)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

    def zeroform_interpolation(self, f: ScalarFunction) -> Array:
        """Interpolate a scalar function by Greville collocation.

        This first implementation is intentionally restricted to the underlying
        tensor-product 0-form space before any nontrivial extraction. That is,
        it currently supports the smallest clean case where the chosen 0-form
        extraction operator is the identity on the full tensor-product basis.
        """
        if self.bc:
            raise NotImplementedError(
                "Greville interpolation is not implemented on boundary-only 0-form spaces."
            )

        e = self.seq.e0_dbc if self.dirichlet else self.seq.e0
        exact_dofs = _matching_discrete_dofs(f, self.seq.basis_0, e)
        if exact_dofs is not None:
            return exact_dofs
        if _is_polar_zeroform_shape(self.seq, e.shape[0], dirichlet=self.dirichlet):
            return self._polar_zeroform_interpolation(f)
        if e.shape[0] != self.seq.basis_0.n:
            raise NotImplementedError(
                "Greville interpolation is currently implemented only for full "
                "tensor-product 0-form spaces without nontrivial extraction."
            )

        bases = self.seq.basis_0.Λ
        x_r = bases[0].greville_points()
        x_t = bases[1].greville_points()
        x_z = bases[2].greville_points()

        r, t, z = jnp.meshgrid(x_r, x_t, x_z, indexing='ij')
        x = jnp.stack([r.ravel(), t.ravel(), z.ravel()], axis=-1)
        values = jax.lax.map(
            lambda xi: _as_single_component(f(xi)),
            x,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        ).reshape(len(x_r), len(x_t), len(x_z))

        coll_r = bases[0].collocation_matrix(x_r)
        coll_t = bases[1].collocation_matrix(x_t)
        coll_z = bases[2].collocation_matrix(x_z)

        coeffs = _solve_tensor_collocation_axis(coll_r, values, axis=0)
        coeffs = _solve_tensor_collocation_axis(coll_t, coeffs, axis=1)
        coeffs = _solve_tensor_collocation_axis(coll_z, coeffs, axis=2)
        return e @ coeffs.reshape(-1)

    def _polar_zeroform_interpolation(self, f: ScalarFunction) -> Array:
        """Interpolate a scalar 0-form on the Holderied polar blue grid."""
        points = self._polar_zeroform_points()

        values = jax.lax.map(
            lambda xi: _as_single_component(f(xi)),
            points,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        ).reshape(-1)

        e = self.seq.e0_dbc if self.dirichlet else self.seq.e0
        basis_indices = self.seq.basis_0.ns
        collocation = jax.lax.map(
            lambda xi: e @ jax.vmap(
                lambda basis_idx: self.seq.basis_0(xi, basis_idx)[0]
            )(basis_indices),
            points,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        )
        return jnp.linalg.solve(collocation, values)

    def _polar_zeroform_points(self) -> Array:
        """Return the Holderied blue-grid points in reduced-coefficient order."""
        bases = self.seq.basis_0.Λ
        x_r = bases[0].greville_points()
        x_t = bases[1].greville_points()
        x_z = bases[2].greville_points()

        nr, nt, nz = self.seq.basis_0.nr, self.seq.basis_0.nt, self.seq.basis_0.nz
        radial_start = 2
        radial_stop = nr - (1 if self.dirichlet else 0)
        points = [
            jnp.array([x_r[1], x_t[p], x_z[m]])
            for p in range(3)
            for m in range(nz)
        ]
        for i in range(radial_start, radial_stop):
            for j in range(nt):
                for k in range(nz):
                    points.append(jnp.array([x_r[i], x_t[j], x_z[k]]))
        return jnp.asarray(points)

    def zeroform_quasi_interpolation(self, f: ScalarFunction) -> Array:
        """Approximate Greville interpolation by dividing by local basis diagonals."""
        if self.bc:
            raise NotImplementedError(
                "Quasi interpolation is not implemented on boundary-only 0-form spaces."
            )

        e = self.seq.e0_dbc if self.dirichlet else self.seq.e0
        exact_dofs = _matching_discrete_dofs(f, self.seq.basis_0, e)
        if exact_dofs is not None:
            return exact_dofs
        bases = self.seq.basis_0.Λ

        if _is_polar_zeroform_shape(self.seq, e.shape[0], dirichlet=self.dirichlet):
            points = self._polar_zeroform_points()
            values = jax.lax.map(
                lambda xi: _as_single_component(f(xi)),
                points,
                batch_size=mrx.MAP_BATCH_SIZE_INNER,
            ).reshape(-1)
            basis_indices = self.seq.basis_0.ns
            diagonal = jax.lax.map(
                lambda i: (
                    e @ jax.vmap(
                        lambda basis_idx: self.seq.basis_0(points[i], basis_idx)[0]
                    )(basis_indices)
                )[i],
                jnp.arange(points.shape[0]),
                batch_size=mrx.MAP_BATCH_SIZE_INNER,
            )
            return values / diagonal

        _require_full_tensor_space(e, self.seq.basis_0.n, "0-form quasi interpolation")
        x_r = bases[0].greville_points()
        x_t = bases[1].greville_points()
        x_z = bases[2].greville_points()

        r, t, z = jnp.meshgrid(x_r, x_t, x_z, indexing='ij')
        x = jnp.stack([r.ravel(), t.ravel(), z.ravel()], axis=-1)
        values = jax.lax.map(
            lambda xi: _as_single_component(f(xi)),
            x,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        ).reshape(len(x_r), len(x_t), len(x_z))

        coll_r = bases[0].collocation_matrix(x_r)
        coll_t = bases[1].collocation_matrix(x_t)
        coll_z = bases[2].collocation_matrix(x_z)
        diagonal = jnp.einsum(
            'i,j,k->ijk',
            jnp.diag(coll_r),
            jnp.diag(coll_t),
            jnp.diag(coll_z),
        )
        return (values / diagonal).reshape(-1)

    def _wrap_periodic_point(self, xi: Array) -> Array:
        wrapped = []
        for axis, basis in enumerate(self.seq.basis_0.Λ):
            coord = xi[axis]
            if basis.type == 'periodic':
                coord = jnp.mod(coord, 1.0)
            wrapped.append(coord)
        return jnp.asarray(wrapped)

    def _oneform_pullback(self, v: VectorFunction) -> VectorFunction:
        DF = jax.jacfwd(self.seq.map)

        def pullback(x: Array) -> Array:
            x_eval = self._wrap_periodic_point(x)
            return inv33(DF(x_eval)) @ v(x_eval)

        return pullback

    def _full_oneform_histopolation_dofs(self, v: VectorFunction) -> tuple[Array, Array, Array]:
        lam_r, lam_t, lam_z = self.seq.basis_0.Λ
        d_r, d_t, d_z = self.seq.basis_0.dΛ
        pts_r = lam_r.greville_points()
        pts_t = lam_t.greville_points()
        pts_z = lam_z.greville_points()
        spans_r = d_r.greville_spans()
        spans_t = d_t.greville_spans()
        spans_z = d_z.greville_spans()
        q_r = _quadrature_order_from_basis_1d(d_r.s)
        q_t = _quadrature_order_from_basis_1d(d_t.s)
        q_z = _quadrature_order_from_basis_1d(d_z.s)
        pullback = self._oneform_pullback(v)

        def integrate_component_0(span_r: Array, t_val: Array, z_val: Array) -> Array:
            xs_r, ws_r = _interval_rule(span_r, q_r)
            x = jnp.stack([xs_r, jnp.full(xs_r.shape, t_val), jnp.full(xs_r.shape, z_val)], axis=-1)
            values = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 0]
            return jnp.sum(values * ws_r)

        def integrate_component_1(r_val: Array, span_t: Array, z_val: Array) -> Array:
            xs_t, ws_t = _interval_rule(span_t, q_t)
            x = jnp.stack([jnp.full(xs_t.shape, r_val), xs_t, jnp.full(xs_t.shape, z_val)], axis=-1)
            values = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 1]
            return jnp.sum(values * ws_t)

        def integrate_component_2(r_val: Array, t_val: Array, span_z: Array) -> Array:
            xs_z, ws_z = _interval_rule(span_z, q_z)
            x = jnp.stack([jnp.full(xs_z.shape, r_val), jnp.full(xs_z.shape, t_val), xs_z], axis=-1)
            values = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 2]
            return jnp.sum(values * ws_z)

        comp0 = jnp.asarray([
            [
                [integrate_component_0(span_r, t_val, z_val) for z_val in pts_z]
                for t_val in pts_t
            ]
            for span_r in spans_r
        ])
        comp1 = jnp.asarray([
            [
                [integrate_component_1(r_val, span_t, z_val) for z_val in pts_z]
                for span_t in spans_t
            ]
            for r_val in pts_r
        ])
        comp2 = jnp.asarray([
            [
                [integrate_component_2(r_val, t_val, span_z) for span_z in spans_z]
                for t_val in pts_t
            ]
            for r_val in pts_r
        ])
        return comp0, comp1, comp2

    def _full_oneform_collocation_matrix(self) -> Array:
        lam_r, lam_t, lam_z = self.seq.basis_0.Λ
        d_r, d_t, d_z = self.seq.basis_0.dΛ
        coll_r = lam_r.collocation_matrix()
        coll_t = lam_t.collocation_matrix()
        coll_z = lam_z.collocation_matrix()
        hist_r = d_r.histopolation_matrix()
        hist_t = d_t.histopolation_matrix()
        hist_z = d_z.histopolation_matrix()

        comp0 = jnp.kron(coll_z, jnp.kron(coll_t, hist_r))
        comp1 = jnp.kron(coll_z, jnp.kron(hist_t, coll_r))
        comp2 = jnp.kron(hist_z, jnp.kron(coll_t, coll_r))

        zeros01 = jnp.zeros((comp0.shape[0], comp1.shape[1]), dtype=comp0.dtype)
        zeros02 = jnp.zeros((comp0.shape[0], comp2.shape[1]), dtype=comp0.dtype)
        zeros10 = jnp.zeros((comp1.shape[0], comp0.shape[1]), dtype=comp0.dtype)
        zeros12 = jnp.zeros((comp1.shape[0], comp2.shape[1]), dtype=comp0.dtype)
        zeros20 = jnp.zeros((comp2.shape[0], comp0.shape[1]), dtype=comp0.dtype)
        zeros21 = jnp.zeros((comp2.shape[0], comp1.shape[1]), dtype=comp0.dtype)
        return jnp.block([
            [comp0, zeros01, zeros02],
            [zeros10, comp1, zeros12],
            [zeros20, zeros21, comp2],
        ])

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

        comp_info, comp_shapes = self.seq._form_comp_info(1)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

    def oneform_histopolation(self, v: VectorFunction) -> Array:
        """Histopolate a vector-valued 1-form by edge moments and point values."""
        if self.bc:
            raise NotImplementedError(
                "Greville histopolation is not implemented on boundary-only 1-form spaces."
            )

        e = self.seq.e1_dbc if self.dirichlet else self.seq.e1
        exact_dofs = _matching_discrete_dofs(v, self.seq.basis_1, e)
        if exact_dofs is not None:
            return exact_dofs
        if _is_polar_oneform_shape(self.seq, e.shape[0], dirichlet=self.dirichlet):
            return self._polar_oneform_histopolation(v)
        _require_full_tensor_space(e, self.seq.basis_1.n, "1-form histopolation")

        lam_r, lam_t, lam_z = self.seq.basis_0.Λ
        d_r, d_t, d_z = self.seq.basis_0.dΛ
        coll_r = lam_r.collocation_matrix()
        coll_t = lam_t.collocation_matrix()
        coll_z = lam_z.collocation_matrix()
        hist_r = d_r.histopolation_matrix()
        hist_t = d_t.histopolation_matrix()
        hist_z = d_z.histopolation_matrix()

        comp0, comp1, comp2 = self._full_oneform_histopolation_dofs(v)

        coeff0 = _solve_tensor_collocation_axis(hist_r, comp0, axis=0)
        coeff0 = _solve_tensor_collocation_axis(coll_t, coeff0, axis=1)
        coeff0 = _solve_tensor_collocation_axis(coll_z, coeff0, axis=2)

        coeff1 = _solve_tensor_collocation_axis(coll_r, comp1, axis=0)
        coeff1 = _solve_tensor_collocation_axis(hist_t, coeff1, axis=1)
        coeff1 = _solve_tensor_collocation_axis(coll_z, coeff1, axis=2)

        coeff2 = _solve_tensor_collocation_axis(coll_r, comp2, axis=0)
        coeff2 = _solve_tensor_collocation_axis(coll_t, coeff2, axis=1)
        coeff2 = _solve_tensor_collocation_axis(hist_z, coeff2, axis=2)

        return e @ jnp.concatenate([
            coeff0.reshape(-1),
            coeff1.reshape(-1),
            coeff2.reshape(-1),
        ])

    def _polar_oneform_histopolation(self, v: VectorFunction) -> Array:
        """Histopolate a polar 1-form using Holderied-style reduced DOFs."""
        e = self.seq.e1_dbc if self.dirichlet else self.seq.e1
        e_t = self.seq.e1_dbc_T if self.dirichlet else self.seq.e1_T
        comp0, comp1, comp2 = self._full_oneform_histopolation_dofs(v)
        full_dofs = jnp.concatenate([
            comp0.reshape(-1),
            comp1.reshape(-1),
            comp2.reshape(-1),
        ])
        reduced_dofs = e @ full_dofs

        full_collocation = self._full_oneform_collocation_matrix()
        e_dense = jnp.asarray(e.todense())
        e_t_dense = jnp.asarray(e_t.todense())
        reduced_collocation = e_dense @ full_collocation @ e_t_dense
        return jnp.linalg.solve(reduced_collocation, reduced_dofs)

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

        comp_info, comp_shapes = self.seq._form_comp_info(2)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

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
            lambda x: _as_single_component(f(x)),
            self.seq.quad.x,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        )
        w_jk: Array = f_jk * (self.seq.quad.w)[:, None]
        comp_info, comp_shapes = self.seq._form_comp_info(3)
        quad_shape = (self.seq.quad.ny, self.seq.quad.nx, self.seq.quad.nz)
        return integrate_against(w_jk, comp_info, comp_shapes, quad_shape)

    def twoform_histopolation(self, v: VectorFunction) -> Array:
        """Histopolate a vector-valued 2-form on the full tensor-product space.

        This first implementation targets the reference/nonpolar full tensor
        space before any nontrivial extraction.
        """
        if self.bc:
            raise NotImplementedError(
                "Greville histopolation is not implemented on boundary-only 2-form spaces."
            )

        e = self.seq.e2_dbc if self.dirichlet else self.seq.e2
        exact_dofs = _matching_discrete_dofs(v, self.seq.basis_2, e)
        if exact_dofs is not None:
            return exact_dofs
        _require_full_tensor_space(e, self.seq.basis_2.n, "2-form histopolation")

        d_r, d_t, d_z = self.seq.basis_0.dΛ
        _require_clamped_histopolation(d_r, "2-form histopolation")
        _require_clamped_histopolation(d_t, "2-form histopolation")
        _require_clamped_histopolation(d_z, "2-form histopolation")

        lam_r, lam_t, lam_z = self.seq.basis_0.Λ
        coll_r = lam_r.collocation_matrix()
        coll_t = lam_t.collocation_matrix()
        coll_z = lam_z.collocation_matrix()
        hist_r = d_r.histopolation_matrix()
        hist_t = d_t.histopolation_matrix()
        hist_z = d_z.histopolation_matrix()

        pts_r = lam_r.greville_points()
        pts_t = lam_t.greville_points()
        pts_z = lam_z.greville_points()
        spans_r = d_r.greville_spans()
        spans_t = d_t.greville_spans()
        spans_z = d_z.greville_spans()

        DF = jax.jacfwd(self.seq.map)

        def pullback(x: Array) -> Array:
            return DF(x).T @ v(x)

        def integrate_component_0(r_val: Array, span_t: Array, span_z: Array) -> Array:
            q_t = _quadrature_order_from_basis_1d(d_t.s)
            q_z = _quadrature_order_from_basis_1d(d_z.s)
            xs_t, ws_t = _interval_rule(span_t, q_t)
            xs_z, ws_z = _interval_rule(span_z, q_z)
            tt, zz = jnp.meshgrid(xs_t, xs_z, indexing='ij')
            x = jnp.stack([jnp.full(tt.size, r_val), tt.ravel(), zz.ravel()], axis=-1)
            values = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 0]
            weights = (ws_t[:, None] * ws_z[None, :]).reshape(-1)
            return jnp.sum(values * weights)

        def integrate_component_1(span_r: Array, t_val: Array, span_z: Array) -> Array:
            q_r = _quadrature_order_from_basis_1d(d_r.s)
            q_z = _quadrature_order_from_basis_1d(d_z.s)
            xs_r, ws_r = _interval_rule(span_r, q_r)
            xs_z, ws_z = _interval_rule(span_z, q_z)
            rr, zz = jnp.meshgrid(xs_r, xs_z, indexing='ij')
            x = jnp.stack([rr.ravel(), jnp.full(rr.size, t_val), zz.ravel()], axis=-1)
            values = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 1]
            weights = (ws_r[:, None] * ws_z[None, :]).reshape(-1)
            return jnp.sum(values * weights)

        def integrate_component_2(span_r: Array, span_t: Array, z_val: Array) -> Array:
            q_r = _quadrature_order_from_basis_1d(d_r.s)
            q_t = _quadrature_order_from_basis_1d(d_t.s)
            xs_r, ws_r = _interval_rule(span_r, q_r)
            xs_t, ws_t = _interval_rule(span_t, q_t)
            rr, tt = jnp.meshgrid(xs_r, xs_t, indexing='ij')
            x = jnp.stack([rr.ravel(), tt.ravel(), jnp.full(rr.size, z_val)], axis=-1)
            values = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 2]
            weights = (ws_r[:, None] * ws_t[None, :]).reshape(-1)
            return jnp.sum(values * weights)

        comp0 = jnp.asarray([
            [
                [integrate_component_0(r_val, span_t, span_z) for span_z in spans_z]
                for span_t in spans_t
            ]
            for r_val in pts_r
        ])
        comp1 = jnp.asarray([
            [
                [integrate_component_1(span_r, t_val, span_z) for span_z in spans_z]
                for t_val in pts_t
            ]
            for span_r in spans_r
        ])
        comp2 = jnp.asarray([
            [
                [integrate_component_2(span_r, span_t, z_val) for z_val in pts_z]
                for span_t in spans_t
            ]
            for span_r in spans_r
        ])

        coeff0 = _solve_tensor_collocation_axis(coll_r, comp0, axis=0)
        coeff0 = _solve_tensor_collocation_axis(hist_t, coeff0, axis=1)
        coeff0 = _solve_tensor_collocation_axis(hist_z, coeff0, axis=2)

        coeff1 = _solve_tensor_collocation_axis(hist_r, comp1, axis=0)
        coeff1 = _solve_tensor_collocation_axis(coll_t, coeff1, axis=1)
        coeff1 = _solve_tensor_collocation_axis(hist_z, coeff1, axis=2)

        coeff2 = _solve_tensor_collocation_axis(hist_r, comp2, axis=0)
        coeff2 = _solve_tensor_collocation_axis(hist_t, coeff2, axis=1)
        coeff2 = _solve_tensor_collocation_axis(coll_z, coeff2, axis=2)

        return e @ jnp.concatenate([
            coeff0.reshape(-1),
            coeff1.reshape(-1),
            coeff2.reshape(-1),
        ])

    def threeform_histopolation(self, f: ScalarFunction) -> Array:
        """Histopolate a scalar 3-form on the full tensor-product space."""
        if self.bc:
            raise NotImplementedError(
                "Greville histopolation is not implemented on boundary-only 3-form spaces."
            )

        e = self.seq.e3_dbc if self.dirichlet else self.seq.e3
        exact_dofs = _matching_discrete_dofs(f, self.seq.basis_3, e)
        if exact_dofs is not None:
            return exact_dofs
        _require_full_tensor_space(e, self.seq.basis_3.n, "3-form histopolation")

        d_r, d_t, d_z = self.seq.basis_0.dΛ
        _require_clamped_histopolation(d_r, "3-form histopolation")
        _require_clamped_histopolation(d_t, "3-form histopolation")
        _require_clamped_histopolation(d_z, "3-form histopolation")

        hist_r = d_r.histopolation_matrix()
        hist_t = d_t.histopolation_matrix()
        hist_z = d_z.histopolation_matrix()
        spans_r = d_r.greville_spans()
        spans_t = d_t.greville_spans()
        spans_z = d_z.greville_spans()

        def integrate_volume(span_r: Array, span_t: Array, span_z: Array) -> Array:
            q_r = _quadrature_order_from_basis_1d(d_r.s)
            q_t = _quadrature_order_from_basis_1d(d_t.s)
            q_z = _quadrature_order_from_basis_1d(d_z.s)
            xs_r, ws_r = _interval_rule(span_r, q_r)
            xs_t, ws_t = _interval_rule(span_t, q_t)
            xs_z, ws_z = _interval_rule(span_z, q_z)
            rr, tt, zz = jnp.meshgrid(xs_r, xs_t, xs_z, indexing='ij')
            x = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
            values = jax.lax.map(
                lambda xi: _as_single_component(f(xi)),
                x,
                batch_size=mrx.MAP_BATCH_SIZE_INNER,
            ).reshape(len(xs_r), len(xs_t), len(xs_z))
            weights = ws_r[:, None, None] * ws_t[None, :, None] * ws_z[None, None, :]
            return jnp.sum(values * weights)

        moments = jnp.asarray([
            [
                [integrate_volume(span_r, span_t, span_z) for span_z in spans_z]
                for span_t in spans_t
            ]
            for span_r in spans_r
        ])

        coeffs = _solve_tensor_collocation_axis(hist_r, moments, axis=0)
        coeffs = _solve_tensor_collocation_axis(hist_t, coeffs, axis=1)
        coeffs = _solve_tensor_collocation_axis(hist_z, coeffs, axis=2)
        return e @ coeffs.reshape(-1)

# TODO: requires testing still
def surface_integral(f: ScalarFunction, seq: "DeRhamSequence") -> Array:
    """Integrate a scalar function over the outer boundary r = 1.

    The surface element is  dS = ‖∂_θ F × ∂_ζ F‖ dθ dζ  evaluated at r = 1.
    Quadrature in (θ, ζ) is reused from ``seq.quad``.

    Parameters
    ----------
    f : callable  ξ → array of shape (1,)
        Function of logical coordinates, called at ξ = (1, θ_q, ζ_q).
    seq : DeRhamSequence

    Returns
    -------
    scalar Array
    """
    nt, nz = seq.quad.ny, seq.quad.nz
    X_t, X_z = jnp.meshgrid(seq.quad.x_y, seq.quad.x_z, indexing='ij')
    xi_bdy = jnp.stack(
        [jnp.ones(nt * nz), X_t.ravel(), X_z.ravel()], axis=-1
    )  # (nt*nz, 3)

    DF = jax.jacfwd(seq.map)

    def _integrand(xi: Array) -> Array:
        dF = DF(xi)
        surf_jac = jnp.linalg.norm(jnp.cross(dF[:, 1], dF[:, 2]))
        return jnp.squeeze(f(xi)) * surf_jac

    vals = jax.lax.map(
        _integrand, xi_bdy, batch_size=mrx.MAP_BATCH_SIZE_INNER
    )  # (nt*nz,)
    w_bdy = jnp.outer(seq.quad.w_y, seq.quad.w_z).ravel()
    return jnp.dot(vals, w_bdy)

# TODO: requires testing still
class BoundaryProjector:
    """Project a k-form onto the Dirichlet boundary DOFs via a surface integral.

    Computes the boundary load vector

        b_i = ∫_{r=1} g(ξ) · trace(φ_i)(ξ) dS,

    then selects the BC DOF values via the ``e_k_bc`` extraction operator.

    ``g`` follows the same convention as :class:`Projector`: for k = 0 and 3,
    a scalar function ξ → (1,); for k = 1 and 2, a vector function
    ξ → (3,) in the physical (x, y, z) frame.

    All quadrature-dependent quantities (surface Jacobian, boundary quad
    points, r-spline values at r = 1) are computed once in ``__init__`` and
    reused across calls.
    """

    def __init__(self, seq: "DeRhamSequence", k: Literal[0, 1, 2, 3]) -> None:
        self.seq = seq
        self.k = k

        # r-spline values at r = 1, shapes (n_r,) and (n_dr,)
        lam_r  = seq.basis_0.Λ[0]
        dlam_r = seq.basis_0.dΛ[0]
        self._basis_r_1   = jax.vmap(lam_r,  (None, 0))(1.0, lam_r.ns)
        self._d_basis_r_1 = jax.vmap(dlam_r, (None, 0))(1.0, dlam_r.ns)

        # 2D boundary quadrature grid (θ, ζ) at r = 1
        nt, nz = seq.quad.ny, seq.quad.nz
        X_t, X_z = jnp.meshgrid(seq.quad.x_y, seq.quad.x_z, indexing='ij')
        xi_bdy = jnp.stack(
            [jnp.ones(nt * nz), X_t.ravel(), X_z.ravel()], axis=-1
        )  # (nt*nz, 3)

        # DF at all boundary quad points, shape (nt, nz, 3, 3)
        # DF[t,z,i,j] = ∂F_i/∂ξ_j
        DF = jax.jacfwd(seq.map)
        DF_bdy = jax.lax.map(
            DF, xi_bdy, batch_size=mrx.MAP_BATCH_SIZE_INNER
        ).reshape(nt, nz, 3, 3)  # (nt, nz, 3, 3)

        # Surface Jacobian ‖∂_θ F × ∂_ζ F‖ and the unnormalized surface normal
        surf_normal = jnp.cross(DF_bdy[:, :, :, 1], DF_bdy[:, :, :, 2])  # (nt, nz, 3)
        surf_jac = jnp.linalg.norm(surf_normal, axis=-1)                  # (nt, nz)

        w_bdy = jnp.outer(seq.quad.w_y, seq.quad.w_z)               # (nt, nz)
        J_bdy = jnp.linalg.det(DF_bdy)                                  # (nt, nz)
        self._xi_bdy       = xi_bdy                                  # (nt*nz, 3)
        self._DF_bdy       = DF_bdy                                  # (nt, nz, 3, 3)
        self._DF_inv_bdy   = jax.vmap(inv33)(DF_bdy.reshape(-1, 3, 3)).reshape(nt, nz, 3, 3)
        self._J_bdy        = J_bdy                                   # (nt, nz)
        self._w_surf       = w_bdy * surf_jac                        # (nt, nz)
        self._nt = nt
        self._nz = nz

    def __call__(self, g: ScalarFunction | VectorFunction | Array) -> Array:
        """Compute the boundary load vector for prescribed boundary data g.

        Parameters
        ----------
        g : callable or array
            If callable: ξ → (1,) for k = 0 or 3; ξ → (3,) in physical frame
            for k = 1 or 2.  Evaluated at the boundary quad points.

            If array of shape (ny*nx*nz, d): precomputed values at the full 3D
            quad grid (e.g. from ``oneform_projection``).  The θ,ζ quad points
            are the same as for the boundary; the r-dimension is irrelevant for
            boundary data, so slice ``[:, 0, :, :]`` is used.

        Returns
        -------
        Array of shape (n_k_bc,)
        """
        seq = self.seq
        nt, nz = self._nt, self._nz

        if callable(g):
            g_jk = jax.lax.map(
                g, self._xi_bdy, batch_size=mrx.MAP_BATCH_SIZE_INNER
            ).reshape(nt, nz, -1)  # (nt, nz, d)
        else:
            # Precomputed 3D values; any r-slice gives the same (θ,ζ) grid
            nx = seq.quad.nx
            g_jk = jnp.asarray(g).reshape(nt, nx, nz, -1)[:, 0, :, :]  # (nt, nz, d)

        if self.k == 0:
            return self._project_0form(g_jk)
        elif self.k == 1:
            return self._project_1form(g_jk)
        elif self.k == 2:
            return self._project_2form(g_jk)
        else:
            raise NotImplementedError("BoundaryProjector: k = 3 not implemented")

    def _project_0form(self, g_jk: Array) -> Array:
        seq = self.seq
        wg = g_jk[:, :, 0] * self._w_surf                          # (nt, nz)
        part = jnp.einsum('jk,bj,ck->bc',
                          wg, seq.basis_t_jk, seq.basis_z_jk)      # (n_t, n_z)
        b_full = jnp.einsum('a,bc->abc',
                            self._basis_r_1, part).ravel()
        return seq.e0_bc @ b_full

    def _project_1form(self, g_jk: Array) -> Array:
        """Transform physical → logical covariant (DF^{-1}) then integrate."""
        seq = self.seq
        nt, nz = self._nt, self._nz

        g_log = jnp.einsum('tzij,tzj->tzi', self._DF_inv_bdy, g_jk)  # (nt, nz, 3)

        # r-component: dΛ_r^a(1) ⊗ Λ_t^b ⊗ Λ_z^c
        wg0 = g_log[:, :, 0] * self._w_surf
        part_r = jnp.einsum('jk,bj,ck->bc', wg0,
                            seq.basis_t_jk, seq.basis_z_jk)
        b_r = jnp.einsum('a,bc->abc', self._d_basis_r_1, part_r).ravel()

        # θ-component: Λ_r^a(1) ⊗ dΛ_t^b ⊗ Λ_z^c
        wg1 = g_log[:, :, 1] * self._w_surf
        part_t = jnp.einsum('jk,bj,ck->bc', wg1,
                            seq.d_basis_t_jk, seq.basis_z_jk)
        b_t = jnp.einsum('a,bc->abc', self._basis_r_1, part_t).ravel()

        # ζ-component: Λ_r^a(1) ⊗ Λ_t^b ⊗ dΛ_z^c
        wg2 = g_log[:, :, 2] * self._w_surf
        part_z = jnp.einsum('jk,bj,ck->bc', wg2,
                            seq.basis_t_jk, seq.d_basis_z_jk)
        b_z = jnp.einsum('a,bc->abc', self._basis_r_1, part_z).ravel()

        return seq.e1_bc @ jnp.concatenate([b_r, b_t, b_z])

    def _project_2form(self, g_jk: Array) -> Array:
        """Pull back g to logical covariant 2-form (DF^T g / J) and integrate
        against each reference basis group weighted by surf_jac."""
        seq = self.seq

        # Pullback: (DF^T g / J)[tz, j] = Σ_i DF[tz,i,j] g[tz,i] / J[tz]
        g_log = jnp.einsum('tzij,tzi->tzj', self._DF_bdy, g_jk) / self._J_bdy[:, :, None]  # (nt, nz, 3)

        # r-component: Λ_r^a(1) ⊗ dΛ_θ^b ⊗ dΛ_ζ^c
        wg0 = g_log[:, :, 0] * self._w_surf
        part_r = jnp.einsum('tz,bz,ct->bc', wg0,
                            seq.d_basis_z_jk, seq.d_basis_t_jk)
        b_r = jnp.einsum('a,bc->abc', self._basis_r_1, part_r).ravel()

        # θ-component: dΛ_r^a(1) ⊗ Λ_θ^b ⊗ dΛ_ζ^c
        wg1 = g_log[:, :, 1] * self._w_surf
        part_t = jnp.einsum('tz,bz,ct->bc', wg1,
                            seq.d_basis_z_jk, seq.basis_t_jk)
        b_t = jnp.einsum('a,bc->abc', self._d_basis_r_1, part_t).ravel()

        # ζ-component: dΛ_r^a(1) ⊗ dΛ_θ^b ⊗ Λ_ζ^c
        wg2 = g_log[:, :, 2] * self._w_surf
        part_z = jnp.einsum('tz,bz,ct->bc', wg2,
                            seq.basis_z_jk, seq.d_basis_t_jk)
        b_z = jnp.einsum('a,bc->abc', self._d_basis_r_1, part_z).ravel()

        return seq.e2_bc @ jnp.concatenate([b_r, b_t, b_z])

    def evaluate_trace(self, u: Array) -> Array:
        """Evaluate the trace of a discrete k-form at the boundary quad points.

        Given the full (unreduced) DOF vector ``u`` of shape ``(n_k,)``,
        reconstruct the field values at the ``(nt, nz)`` boundary quad points.

        No coordinate map evaluation is needed:

        * k = 0: scalar ``f(1, θ, ζ)``, shape ``(nt, nz)``.
        * k = 1: logical components ``E_log = (E_r, E_θ, E_ζ)`` at r = 1,
          shape ``(nt, nz, 3)``.  The physical tangential vector is
          ``DF^{-T} E_log`` using the precomputed ``self._DF_inv_bdy``.
        * k = 2: normal flux ``B_log_r = B_phys · (∂_θF × ∂_ζF)`` at r = 1,
          shape ``(nt, nz)``.  The Jacobian J cancels exactly because
          ``B_phys = (1/J) DF B_log``, so no DF evaluation is needed.

        Parameters
        ----------
        u : Array, shape ``(n_k,)``
            Full DOF vector in the unreduced space (i.e. *not* BC-extracted).

        Returns
        -------
        Array of shape ``(nt, nz)`` for k = 0 or 2, ``(nt, nz, 3)`` for k = 1.
        """
        if self.k == 0:
            return self._eval_trace_0form(u)
        elif self.k == 1:
            return self._eval_trace_1form(u)
        elif self.k == 2:
            return self._eval_trace_2form(u)
        else:
            raise NotImplementedError("evaluate_trace: k = 3 not implemented")

    def _eval_trace_0form(self, u: Array) -> Array:
        seq = self.seq
        n_r = self._basis_r_1.shape[0]
        n_t = seq.basis_t_jk.shape[0]
        n_z = seq.basis_z_jk.shape[0]
        u_3d = u.reshape(n_r, n_t, n_z)
        # f(1, θ_q, ζ_q) = Σ_{a,b,c} u[a,b,c] Λ_r^a(1) Λ_t^b(θ_q) Λ_z^c(ζ_q)
        return jnp.einsum('abc,a,bt,cz->tz', u_3d,
                          self._basis_r_1, seq.basis_t_jk, seq.basis_z_jk)

    def _eval_trace_1form(self, u: Array) -> Array:
        """Return logical components E_log at r = 1, shape (nt, nz, 3).

        No DF is applied here; physical E_phys = DF^{-T} E_log via
        ``einsum('tzji,tzj->tzi', self._DF_inv_bdy, E_log)`` if needed.
        """
        seq = self.seq
        n_dr = self._d_basis_r_1.shape[0]
        n_r  = self._basis_r_1.shape[0]
        n_t  = seq.basis_t_jk.shape[0]
        n_dt = seq.d_basis_t_jk.shape[0]
        n_z  = seq.basis_z_jk.shape[0]
        n_dz = seq.d_basis_z_jk.shape[0]
        n1_r = n_dr * n_t * n_z
        n1_t = n_r  * n_dt * n_z
        u_r = u[:n1_r].reshape(n_dr, n_t, n_z)
        u_t = u[n1_r:n1_r + n1_t].reshape(n_r, n_dt, n_z)
        u_z = u[n1_r + n1_t:].reshape(n_r, n_t, n_dz)
        # E_log_r = Σ u_r[a,b,c] dΛ_r^a(1) Λ_t^b Λ_z^c
        E_r = jnp.einsum('abc,a,bt,cz->tz', u_r,
                         self._d_basis_r_1, seq.basis_t_jk, seq.basis_z_jk)
        # E_log_t = Σ u_t[a,b,c] Λ_r^a(1) dΛ_t^b Λ_z^c
        E_t = jnp.einsum('abc,a,bt,cz->tz', u_t,
                         self._basis_r_1, seq.d_basis_t_jk, seq.basis_z_jk)
        # E_log_z = Σ u_z[a,b,c] Λ_r^a(1) Λ_t^b dΛ_z^c
        E_z = jnp.einsum('abc,a,bt,cz->tz', u_z,
                         self._basis_r_1, seq.basis_t_jk, seq.d_basis_z_jk)
        return jnp.stack([E_r, E_t, E_z], axis=-1)  # (nt, nz, 3)

    def _eval_trace_2form(self, u: Array) -> Array:
        """Return B_phys · surf_normal = B_log_r at r = 1, shape (nt, nz).

        J cancels: B_phys · surf_normal = (1/J)(DF B_log) · surf_normal
                                        = (1/J) J B_log_r = B_log_r.

        This is the *unscaled* normal flux (integrated against the surface
        element).  To get the pointwise normal component B_phys · n̂ divide
        by the surface Jacobian ‖∂_θF × ∂_ζF‖, accessible as
        ``bp.surf_jac()``.
        """
        seq = self.seq
        n_r  = self._basis_r_1.shape[0]
        n_dt = seq.d_basis_t_jk.shape[0]
        n_dz = seq.d_basis_z_jk.shape[0]
        n2_r = n_r * n_dt * n_dz
        u_r = u[:n2_r].reshape(n_r, n_dt, n_dz)
        # B_log_r = Σ u_r[a,b,c] Λ_r^a(1) dΛ_t^b dΛ_z^c
        return jnp.einsum('abc,a,bt,cz->tz', u_r,
                          self._basis_r_1, seq.d_basis_t_jk, seq.d_basis_z_jk)
