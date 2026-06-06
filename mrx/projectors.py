"""
Load vector assembly and interpolation for finite element differential forms.

Public API
----------
load(seq, f, k, dirichlet=False, bc=False)
    Assemble the dual load vector  v_i = ∫ Λ^k_i · f dx  for a k-form.
    Completely matrix-free; only the extraction matrices on ``seq`` are needed.

interpolate(seq, f, k, dirichlet=False)
    Compute primal DOFs by Greville interpolation (k=0) or histopolation
    (k=1,2,3).  Collocation/histopolation matrices are built lazily on each
    call.  TODO: cache them on the sequence object if profiling shows this
    is a bottleneck.

Both functions are also available as ``seq.load(...)`` and
``seq.interpolate(...)`` on :class:`~mrx.derham_sequence.DeRhamSequence`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import mrx
from mrx.extraction_operators import get_xi
from mrx.differential_forms import inv33
from mrx.quadrature import integrate_against

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


def _extraction(seq, k: int, dirichlet: bool, bc: bool):
    """Pick the right extraction matrix for degree k."""
    if bc:
        return getattr(seq, f'e{k}_bc')
    elif dirichlet:
        return getattr(seq, f'e{k}_dbc')
    else:
        return getattr(seq, f'e{k}')


def _extraction_T(seq, k: int, dirichlet: bool):
    """Pick the right transpose extraction matrix for degree k."""
    suffix = '_dbc_T' if dirichlet else '_T'
    return getattr(seq, f'e{k}{suffix}')


# ---------------------------------------------------------------------------
# Load vector assembly  (matrix-free, works on any seq after set_map)
# ---------------------------------------------------------------------------

def load(seq: "DeRhamSequence", f, k: int,
         dirichlet: bool = False, bc: bool = False):
    """Assemble the dual k-form load vector  v_i = ∫ Λ^k_i · f(ξ) w(ξ) dξ.

    Parameters
    ----------
    seq : DeRhamSequence
    f : callable  ξ → (1,) for k=0,3;  ξ → (3,) for k=1,2.
        Arguments are logical coordinates; values are in the physical frame.
    k : int  Form degree (0, 1, 2, 3).
    dirichlet : bool  Use Dirichlet-constrained DOFs.
    bc : bool  Use boundary-trace DOFs (takes precedence over dirichlet).

    Returns
    -------
    Array  Dual load vector of length n_k (or n_k_dbc / n_k_bc).
    """
    e = _extraction(seq, k, dirichlet, bc)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    comp_info, comp_shapes = seq._form_comp_info(k)

    if k == 0:
        f_jk = jax.lax.map(
            lambda x: _as_single_component(f(x)),
            seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]

    elif k == 1:
        DF = jax.jacfwd(seq.map)

        def _pullback(x):
            return inv33(DF(x)) @ f(x)

        f_jk = jax.lax.map(_pullback, seq.quad.x,
                            batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]

    elif k == 2:
        DF = jax.jacfwd(seq.map)

        def _pullback(x):
            return DF(x).T @ f(x)

        f_jk = jax.lax.map(_pullback, seq.quad.x,
                            batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * seq.quad.w[:, None]

    elif k == 3:
        f_jk = jax.lax.map(
            lambda x: _as_single_component(f(x)),
            seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * seq.quad.w[:, None]

    else:
        raise ValueError(f"k must be 0, 1, 2 or 3, got {k}")

    return e @ integrate_against(w_jk, comp_info, comp_shapes, quad_shape)


# ---------------------------------------------------------------------------
# Interpolation / histopolation  (collocation matrices built lazily)
# ---------------------------------------------------------------------------

def interpolate(seq: "DeRhamSequence", f, k: int, dirichlet: bool = False):
    """Compute primal DOFs by Greville interpolation (k=0) or histopolation (k=1,2,3).

    Collocation and histopolation matrices are built lazily on each call.
    TODO: cache them on the sequence object if profiling shows this is a
    bottleneck.

    Parameters
    ----------
    seq : DeRhamSequence
    f : callable  ξ → (1,) for k=0,3;  ξ → (3,) for k=1,2.
    k : int  Form degree (0, 1, 2, 3).
    dirichlet : bool  Use Dirichlet-constrained DOFs.

    Returns
    -------
    Array  Primal DOF vector.
    """
    if k == 0:
        return _interpolate_0form(seq, f, dirichlet)
    elif k == 1:
        return _histopolate_1form(seq, f, dirichlet)
    elif k == 2:
        return _histopolate_2form(seq, f, dirichlet)
    elif k == 3:
        return _histopolate_3form(seq, f, dirichlet)
    else:
        raise ValueError(f"k must be 0, 1, 2 or 3, got {k}")


def _wrap_periodic_point(seq, xi):
    wrapped = []
    for axis, basis in enumerate(seq.basis_0.Λ):
        coord = xi[axis]
        if basis.type == 'periodic':
            coord = jnp.mod(coord, 1.0)
        wrapped.append(coord)
    return jnp.asarray(wrapped)


def _oneform_pullback(seq, v):
    DF = jax.jacfwd(seq.map)

    def pullback(x):
        x_eval = _wrap_periodic_point(seq, x)
        return inv33(DF(x_eval)) @ v(x_eval)

    return pullback


def _interpolate_0form(seq, f, dirichlet: bool) -> Array:
    """Greville collocation for a scalar 0-form."""
    e = seq.e0_dbc if dirichlet else seq.e0
    exact = _matching_discrete_dofs(f, seq.basis_0, e)
    if exact is not None:
        return exact
    _require_full_tensor_space(e, seq.basis_0.n, "0-form interpolation")

    bases = seq.basis_0.Λ
    x_r = bases[0].greville_points()
    x_t = bases[1].greville_points()
    x_z = bases[2].greville_points()

    r, t, z = jnp.meshgrid(x_r, x_t, x_z, indexing='ij')
    pts = jnp.stack([r.ravel(), t.ravel(), z.ravel()], axis=-1)
    values = jax.lax.map(
        lambda xi: _as_single_component(f(xi)), pts,
        batch_size=mrx.MAP_BATCH_SIZE_INNER,
    ).reshape(len(x_r), len(x_t), len(x_z))

    # TODO: cache these on seq if collocation matrix build time is significant
    coll_r = bases[0].collocation_matrix(x_r)
    coll_t = bases[1].collocation_matrix(x_t)
    coll_z = bases[2].collocation_matrix(x_z)

    coeffs = _solve_tensor_collocation_axis(coll_r, values, axis=0)
    coeffs = _solve_tensor_collocation_axis(coll_t, coeffs, axis=1)
    coeffs = _solve_tensor_collocation_axis(coll_z, coeffs, axis=2)
    return e @ coeffs.reshape(-1)


def _full_oneform_histopolation_dofs(seq, v):
    lam_r, lam_t, lam_z = seq.basis_0.Λ
    d_r, d_t, d_z = seq.basis_0.dΛ
    pts_r = lam_r.greville_points()
    pts_t = lam_t.greville_points()
    pts_z = lam_z.greville_points()
    spans_r = d_r.greville_spans()
    spans_t = d_t.greville_spans()
    spans_z = d_z.greville_spans()
    q_r = _quadrature_order_from_basis_1d(d_r.s)
    q_t = _quadrature_order_from_basis_1d(d_t.s)
    q_z = _quadrature_order_from_basis_1d(d_z.s)
    pullback = _oneform_pullback(seq, v)

    def integrate_component_0(span_r, t_val, z_val):
        xs_r, ws_r = _interval_rule(span_r, q_r)
        x = jnp.stack([xs_r, jnp.full(xs_r.shape, t_val),
                        jnp.full(xs_r.shape, z_val)], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 0]
        return jnp.sum(vals * ws_r)

    def integrate_component_1(r_val, span_t, z_val):
        xs_t, ws_t = _interval_rule(span_t, q_t)
        x = jnp.stack([jnp.full(xs_t.shape, r_val), xs_t,
                        jnp.full(xs_t.shape, z_val)], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 1]
        return jnp.sum(vals * ws_t)

    def integrate_component_2(r_val, t_val, span_z):
        xs_z, ws_z = _interval_rule(span_z, q_z)
        x = jnp.stack([jnp.full(xs_z.shape, r_val),
                        jnp.full(xs_z.shape, t_val), xs_z], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 2]
        return jnp.sum(vals * ws_z)

    comp0 = jnp.asarray([
        [[integrate_component_0(sr, t, z) for z in pts_z] for t in pts_t]
        for sr in spans_r
    ])
    comp1 = jnp.asarray([
        [[integrate_component_1(r, st, z) for z in pts_z] for st in spans_t]
        for r in pts_r
    ])
    comp2 = jnp.asarray([
        [[integrate_component_2(r, t, sz) for sz in spans_z] for t in pts_t]
        for r in pts_r
    ])
    return comp0, comp1, comp2


def _histopolate_1form(seq, v, dirichlet: bool) -> Array:
    """Greville histopolation for a 1-form."""
    e = seq.e1_dbc if dirichlet else seq.e1
    exact = _matching_discrete_dofs(v, seq.basis_1, e)
    if exact is not None:
        return exact
    _require_full_tensor_space(e, seq.basis_1.n, "1-form histopolation")

    lam_r, lam_t, lam_z = seq.basis_0.Λ
    d_r, d_t, d_z = seq.basis_0.dΛ

    # TODO: cache these on seq if matrix build time is significant
    coll_r = lam_r.collocation_matrix()
    coll_t = lam_t.collocation_matrix()
    coll_z = lam_z.collocation_matrix()
    hist_r = d_r.histopolation_matrix()
    hist_t = d_t.histopolation_matrix()
    hist_z = d_z.histopolation_matrix()

    comp0, comp1, comp2 = _full_oneform_histopolation_dofs(seq, v)

    c0 = _solve_tensor_collocation_axis(hist_r, comp0, axis=0)
    c0 = _solve_tensor_collocation_axis(coll_t, c0, axis=1)
    c0 = _solve_tensor_collocation_axis(coll_z, c0, axis=2)

    c1 = _solve_tensor_collocation_axis(coll_r, comp1, axis=0)
    c1 = _solve_tensor_collocation_axis(hist_t, c1, axis=1)
    c1 = _solve_tensor_collocation_axis(coll_z, c1, axis=2)

    c2 = _solve_tensor_collocation_axis(coll_r, comp2, axis=0)
    c2 = _solve_tensor_collocation_axis(coll_t, c2, axis=1)
    c2 = _solve_tensor_collocation_axis(hist_z, c2, axis=2)

    return e @ jnp.concatenate([c0.reshape(-1), c1.reshape(-1), c2.reshape(-1)])


def _histopolate_2form(seq, v, dirichlet: bool) -> Array:
    """Greville histopolation for a 2-form."""
    e = seq.e2_dbc if dirichlet else seq.e2
    exact = _matching_discrete_dofs(v, seq.basis_2, e)
    if exact is not None:
        return exact
    _require_full_tensor_space(e, seq.basis_2.n, "2-form histopolation")

    d_r, d_t, d_z = seq.basis_0.dΛ
    _require_clamped_histopolation(d_r, "2-form histopolation")
    _require_clamped_histopolation(d_t, "2-form histopolation")
    _require_clamped_histopolation(d_z, "2-form histopolation")

    lam_r, lam_t, lam_z = seq.basis_0.Λ

    # TODO: cache these on seq if matrix build time is significant
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

    DF = jax.jacfwd(seq.map)

    def pullback(x):
        return DF(x).T @ v(x)

    def int0(r_val, span_t, span_z):
        q_t = _quadrature_order_from_basis_1d(d_t.s)
        q_z = _quadrature_order_from_basis_1d(d_z.s)
        xs_t, ws_t = _interval_rule(span_t, q_t)
        xs_z, ws_z = _interval_rule(span_z, q_z)
        tt, zz = jnp.meshgrid(xs_t, xs_z, indexing='ij')
        x = jnp.stack([jnp.full(tt.size, r_val), tt.ravel(), zz.ravel()], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 0]
        return jnp.sum(vals * (ws_t[:, None] * ws_z[None, :]).reshape(-1))

    def int1(span_r, t_val, span_z):
        q_r = _quadrature_order_from_basis_1d(d_r.s)
        q_z = _quadrature_order_from_basis_1d(d_z.s)
        xs_r, ws_r = _interval_rule(span_r, q_r)
        xs_z, ws_z = _interval_rule(span_z, q_z)
        rr, zz = jnp.meshgrid(xs_r, xs_z, indexing='ij')
        x = jnp.stack([rr.ravel(), jnp.full(rr.size, t_val), zz.ravel()], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 1]
        return jnp.sum(vals * (ws_r[:, None] * ws_z[None, :]).reshape(-1))

    def int2(span_r, span_t, z_val):
        q_r = _quadrature_order_from_basis_1d(d_r.s)
        q_t = _quadrature_order_from_basis_1d(d_t.s)
        xs_r, ws_r = _interval_rule(span_r, q_r)
        xs_t, ws_t = _interval_rule(span_t, q_t)
        rr, tt = jnp.meshgrid(xs_r, xs_t, indexing='ij')
        x = jnp.stack([rr.ravel(), tt.ravel(), jnp.full(rr.size, z_val)], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 2]
        return jnp.sum(vals * (ws_r[:, None] * ws_t[None, :]).reshape(-1))

    comp0 = jnp.asarray([
        [[int0(r, st, sz) for sz in spans_z] for st in spans_t]
        for r in pts_r
    ])
    comp1 = jnp.asarray([
        [[int1(sr, t, sz) for sz in spans_z] for t in pts_t]
        for sr in spans_r
    ])
    comp2 = jnp.asarray([
        [[int2(sr, st, z) for z in pts_z] for st in spans_t]
        for sr in spans_r
    ])

    c0 = _solve_tensor_collocation_axis(coll_r, comp0, axis=0)
    c0 = _solve_tensor_collocation_axis(hist_t, c0, axis=1)
    c0 = _solve_tensor_collocation_axis(hist_z, c0, axis=2)

    c1 = _solve_tensor_collocation_axis(hist_r, comp1, axis=0)
    c1 = _solve_tensor_collocation_axis(coll_t, c1, axis=1)
    c1 = _solve_tensor_collocation_axis(hist_z, c1, axis=2)

    c2 = _solve_tensor_collocation_axis(hist_r, comp2, axis=0)
    c2 = _solve_tensor_collocation_axis(hist_t, c2, axis=1)
    c2 = _solve_tensor_collocation_axis(coll_z, c2, axis=2)

    return e @ jnp.concatenate([c0.reshape(-1), c1.reshape(-1), c2.reshape(-1)])


def _histopolate_3form(seq, f, dirichlet: bool) -> Array:
    """Greville histopolation for a scalar 3-form."""
    e = seq.e3_dbc if dirichlet else seq.e3
    exact = _matching_discrete_dofs(f, seq.basis_3, e)
    if exact is not None:
        return exact
    _require_full_tensor_space(e, seq.basis_3.n, "3-form histopolation")

    d_r, d_t, d_z = seq.basis_0.dΛ
    _require_clamped_histopolation(d_r, "3-form histopolation")
    _require_clamped_histopolation(d_t, "3-form histopolation")
    _require_clamped_histopolation(d_z, "3-form histopolation")

    # TODO: cache these on seq if matrix build time is significant
    hist_r = d_r.histopolation_matrix()
    hist_t = d_t.histopolation_matrix()
    hist_z = d_z.histopolation_matrix()
    spans_r = d_r.greville_spans()
    spans_t = d_t.greville_spans()
    spans_z = d_z.greville_spans()

    def integrate_volume(span_r, span_t, span_z):
        q_r = _quadrature_order_from_basis_1d(d_r.s)
        q_t = _quadrature_order_from_basis_1d(d_t.s)
        q_z = _quadrature_order_from_basis_1d(d_z.s)
        xs_r, ws_r = _interval_rule(span_r, q_r)
        xs_t, ws_t = _interval_rule(span_t, q_t)
        xs_z, ws_z = _interval_rule(span_z, q_z)
        rr, tt, zz = jnp.meshgrid(xs_r, xs_t, xs_z, indexing='ij')
        x = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
        values = jax.lax.map(
            lambda xi: _as_single_component(f(xi)), x,
            batch_size=mrx.MAP_BATCH_SIZE_INNER,
        ).reshape(len(xs_r), len(xs_t), len(xs_z))
        weights = ws_r[:, None, None] * ws_t[None, :, None] * ws_z[None, None, :]
        return jnp.sum(values * weights)

    moments = jnp.asarray([
        [[integrate_volume(sr, st, sz) for sz in spans_z] for st in spans_t]
        for sr in spans_r
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
