"""Matrix-free mass matrices of the k-forms and their diagonals.

The mass matrix of the k-forms is ``M_k = int Lambda^k . W_k . Lambda^k`` over the logical domain,
with the geometric weight ``W_0 = J``, ``W_1 = J G^{-1}``, ``W_2 = G / J`` and ``W_3 = 1 / J``
(``G = DPhi^T DPhi`` the metric tensor and ``J = det DPhi`` the Jacobian determinant of the
logical-to-physical map ``Phi``). No matrix is stored.
Every product ``M_k x`` is computed from the basis values at the quadrature points. The projection
masses ``int Lambda^{k_row} . Lambda^{k_col}`` between different forms work the same way with the
weight 1. All of these act on the full tensor-product spline coefficients, before the boundary and
polar constraints are applied. The callers in :mod:`mrx.operators` add those constraints.

The product is a global sum factorisation: the coefficients of each component are taken to the quadrature grid
by one contraction per axis with the 1-D basis tables, multiplied by the weights there, and tested back the same
way. The components are zero-padded to one grid shape and handled as one batch, so a product costs a handful of
large contractions instead of a few small ones per component, which matters on a GPU where every launched kernel
costs a few microseconds whatever its size. On a TPU each contraction uses only the nonzero band of the
tables, as shifted slices of the grid (:mod:`mrx.banded`), elsewhere the dense tables.

The work is split into a static part and a geometry part. :func:`mass_plan` and
:func:`projection_plan` hold the basis tables and are built once with the de Rham sequence.
:func:`attach_weights` computes the weights from a geometry and stores them on it, so a new map
``Phi`` only recomputes the weights and reuses the compiled code. :func:`sumfact_apply` applies a plan with its
weights, and :func:`build_mass_diagonal` returns ``diag(M_k)``.
"""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import mrx.banded as banded
import mrx.double_single as ds
from mrx.banded import Band
from mrx.precision import PAIRS


class SumfactPlan(eqx.Module):
    """The geometry-independent part of a mass or projection apply: the 1-D basis tables of the row and the
    column form at the quadrature points, one ``(n_comp, N_a, Q_a)`` array per axis ``a`` with every component
    zero-padded to the largest coefficient count ``N_a``. The sequence builds one per mass matrix
    (:func:`mass_plan`) and one per projection (:func:`projection_plan`). The geometric weights are stored on
    the geometry instead (:func:`attach_weights`). In float32 the plan also holds the tables in the gather form
    of the double-single apply (:class:`mrx.double_single.Tables`). ``bands_r`` and ``bands_c`` hold the same
    tables without their zeros (:class:`mrx.banded.Band`), or ``None`` where a band does not reproduce them."""

    tables_r: tuple
    tables_c: tuple
    shapes_r: tuple = eqx.field(static=True)
    shapes_c: tuple = eqx.field(static=True)
    gather_r: Optional[tuple] = None
    gather_c: Optional[tuple] = None
    bands_r: Optional[tuple] = None
    bands_c: Optional[tuple] = None


def _bands(tables, spans):
    bands = tuple(Band.from_dense(t, e) for t, e in zip(tables, spans))
    return None if any(b is None for b in bands) else bands


def _plan(tables_r, tables_c, shapes_r, shapes_c, spans):
    gather_r = gather_c = None
    if PAIRS:
        gather_r = tuple(ds.Tables.from_dense(t) for t in tables_r)
        gather_c = gather_r if tables_c is tables_r else tuple(ds.Tables.from_dense(t) for t in tables_c)
    bands_r = _bands(tables_r, spans)
    bands_c = bands_r if tables_c is tables_r else _bands(tables_c, spans)
    return SumfactPlan(tables_r, tables_c, shapes_r, shapes_c, gather_r, gather_c, bands_r, bands_c)


def _spans(seq):
    """The number of knot spans the quadrature covers per axis."""
    return (seq.quad.ne_x, seq.quad.ne_y, seq.quad.ne_z)


def _padded_tables(seq, k):
    """The zero-padded 1-D tables of the k-forms and the coefficient grid shape of each component."""
    info = seq._form_comp_info(k, banded=False)
    shapes = tuple(tuple(int(n) for n in s) for s in getattr(seq, f"basis_{k}").shape)
    tables = []
    for a in range(3):
        n_max = max(s[a] for s in shapes)
        t = np.zeros((len(shapes), n_max, int(info[0][1 + a].shape[1])))
        for c, (_, *axes) in enumerate(info):
            t[c, :shapes[c][a]] = np.asarray(axes[a], dtype=np.float64)
        tables.append(jnp.asarray(t))
    return tuple(tables), shapes


def mass_plan(seq, k):
    """The :class:`SumfactPlan` of ``M_k``."""
    tables, shapes = _padded_tables(seq, k)
    return _plan(tables, tables, shapes, shapes, _spans(seq))


def projection_plan(seq, k_row, k_col):
    """The :class:`SumfactPlan` of the projection mass ``int Lambda^{k_row} . Lambda^{k_col}`` (weight 1).

    It maps ``k_col``-form coefficients to ``k_row``-form moments.
    """
    tables_r, shapes_r = _padded_tables(seq, k_row)
    tables_c, shapes_c = _padded_tables(seq, k_col)
    return _plan(tables_r, tables_c, shapes_r, shapes_c, _spans(seq))


def _mass_weight(k, metric, metric_inv, jac):
    """The pointwise weight of ``M_k``, shape ``(n_q, n, n)``: ``J`` for k=0, ``J G^{-1}`` for k=1, ``G / J`` for
    k=2 and ``1 / J`` for k=3."""
    if k == 0:
        return jac[:, None, None]
    if k == 3:
        return (1.0 / jac)[:, None, None]
    return metric_inv * jac[:, None, None] if k == 1 else metric / jac[:, None, None]


def attach_weights(seq, geometry):
    """Return ``geometry`` with the weights of all mass and projection matrices attached.

    ``mass_weights[k]`` holds the weight of ``M_k`` and ``reference_weights[n]`` that of the projection masses
    between forms of ``n`` components, both times the quadrature weights and of shape ``(Q_r, Q_theta, Q_zeta,
    n, n)``. The sequence calls this when a map is installed. Code that installs a geometry by hand, such as the
    shape-derivative code, must call it too.
    """
    metric, metric_inv, jac = geometry.metric_jkl, geometry.metric_inv_jkl, geometry.jacobian_j
    q = tuple(int(n) for n in seq.quad.shape)
    w = seq.quad.w[:, None, None]

    def grid(W):
        return (W * w).reshape(q + W.shape[1:])

    mass_weights = {k: grid(_mass_weight(k, metric, metric_inv, jac)) for k in range(4)}
    reference = {n: grid(jnp.broadcast_to(jnp.eye(n, dtype=jac.dtype), (jac.shape[0], n, n))) for n in (1, 3)}
    return eqx.tree_at(lambda g: (g.mass_weights, g.reference_weights), geometry,
                       (mass_weights, reference), is_leaf=lambda x: x is None)


def sumfact_apply(plan, weights, x):
    """Apply ``x -> int Lambda_row . W . Lambda_col x`` given a :class:`SumfactPlan` and its weights.

    ``x`` holds the full tensor-product coefficients. Boundary and polar constraints are left to
    the caller. On a half-period sequence the result is twice the integral over the half period.
    A :class:`~mrx.double_single.DS` ``x`` gives a double-single result from the same tables and weights.

    The choice of tables (:func:`mrx.banded.enabled`) is made here, outside the compiled applies: inside one jit
    its first trace would be reused whatever :func:`mrx.banded.dense` says later.
    """
    if isinstance(x, ds.DS):
        return _apply_ds(plan, weights, x)
    if banded.enabled() and plan.bands_r is not None and plan.bands_c is not None:
        return _apply_banded(plan, weights, x)
    return _apply_dense(plan, weights, x)


@eqx.filter_jit
def _apply_ds(plan, weights, x):
    """:func:`sumfact_apply` of a :class:`~mrx.double_single.DS` vector."""
    return ds.sumfact(plan.gather_r, plan.gather_c, plan.shapes_r, plan.shapes_c, weights, x)


def _stack_padded(x, shapes, n_max):
    """The concatenated component grids of ``x``, zero-padded to ``n_max`` and stacked."""
    parts, offset = [], 0
    for s in shapes:
        n = s[0] * s[1] * s[2]
        parts.append(jnp.pad(x[offset:offset + n].reshape(s), [(0, n_max[a] - s[a]) for a in range(3)]))
        offset += n
    return jnp.stack(parts)


def _unstack(y, shapes):
    return jnp.concatenate([y[c, :s[0], :s[1], :s[2]].reshape(-1) for c, s in enumerate(shapes)])


@eqx.filter_jit
def _apply_dense(plan, weights, x):
    """:func:`sumfact_apply` with the dense tables."""
    R, T, Z = plan.tables_c
    x = _stack_padded(x, plan.shapes_c, tuple(t.shape[1] for t in plan.tables_c))
    u = jnp.einsum('cijk,cia,cjb,ckd->abdc', x, R, T, Z)
    v = jnp.einsum('abdrc,abdc->rabd', weights, u)
    R, T, Z = plan.tables_r
    return _unstack(jnp.einsum('ria,rjb,rkd,rabd->rijk', R, T, Z, v), plan.shapes_r)


@eqx.filter_jit
def _apply_banded(plan, weights, x):
    """:func:`sumfact_apply` with the banded tables: one shifted-slice contraction per axis (:mod:`mrx.banded`)."""
    u = _stack_padded(x, plan.shapes_c, tuple(b.n for b in plan.bands_c))
    for a, b in enumerate(plan.bands_c):
        u = banded.forward(u, b, a + 1)                         # (c, Q_r, Q_t, Q_z)
    v = jnp.einsum('abdrc,cabd->rabd', weights, u)
    for a, b in enumerate(plan.bands_r):
        v = banded.backward(v, b, a + 1)
    return _unstack(v, plan.shapes_r)


def build_mass_diagonal(seq, k):
    """Return ``diag(M_k)`` on the installed geometry, for the full tensor-product coefficients.

    The components are concatenated. The diagonal is computed directly, without applying ``M_k``.
    """
    plan, W = seq.mass_plan[k], seq.geometry.mass_weights[k]
    R, T, Z = plan.tables_c
    parts = []
    for c, s in enumerate(plan.shapes_c):
        # squared basis tables: the row and column bases coincide on the diagonal
        d = jnp.einsum('ia,jb,kd,abd->ijk', R[c] ** 2, T[c] ** 2, Z[c] ** 2, W[..., c, c])
        parts.append(d[:s[0], :s[1], :s[2]].reshape(-1))
    d = jnp.concatenate(parts)
    if seq.half_period:
        # d holds twice the half-period integrals. An entry and its mirror image each hold
        # one half of the full integral, so their unsigned mean is the full diagonal.
        from mrx.symmetry import symmetrize  # noqa: PLC0415
        d = symmetrize(d, seq.reflection_plan[k], 1, signed=False)
    return d
