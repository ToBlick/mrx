"""
Load vector assembly and interpolation for finite element differential forms.

Public API
----------
load(seq, f, k, dirichlet=False, bc=False)
    Assemble the dual load vector  v_i = ∫ Λ^k_i · f dx  for a k-form.
    Completely matrix-free; only the extraction matrices on ``seq`` are needed.

interpolate(seq, f, k, dirichlet=False)
    Compute primal DOFs by Greville interpolation (k=0) or histopolation
    (k=1,2,3).  The Greville points, spans, collocation/histopolation
    matrices and span quadrature rules depend on the knot vectors only; they
    are built once per sequence and cached on it (:func:`_greville_data`).

Both functions are also available as ``seq.load(...)`` and
``seq.interpolate(...)`` on :class:`~mrx.derham_sequence.DeRhamSequence`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import sparse
from scipy.sparse import csgraph

import mrx
from mrx.differential_forms import DifferentialForm, adj33
from mrx.geometry import _tp_evaluate, map_jacobian_at
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


def _quadrature_order_from_basis_1d(basis) -> int:
    return max(2, basis.p + 2)


def _span_quadrature(basis, spans: Array) -> tuple[Array, Array]:
    """Gauss rule on every Greville span, SPLIT at the knots the span contains.

    Greville spans straddle an interior knot whenever the degree is EVEN --
    ``g_i = i + (p+1)/2`` in units of the knot spacing, integral for odd p and
    half-integral for even p -- and Gauss-Legendre is exact only for
    POLYNOMIALS.  A spline has a derivative jump at each knot, so a single rule
    spanning one converges algebraically rather than exactly: measured on an
    off-centre knot, 40 points still left 3.6e-07 where splitting is exact at 2.

    ``SplineBasis.histopolation_matrix`` splits identically, so the matrix and
    the moments are built with the SAME rule -- which matters independently of
    exactness, since ``solve(H, m) = c`` needs ``m = H c`` and quadrature is
    linear.  (It clips EVERY knot into the span and carries the zero-width
    pieces; here only the knots strictly inside a span cut it, and the rule is
    padded to the widest span with zero-width pieces at ``b``.  The nonzero
    points and weights are the same, so the two rules are identical as
    functionals -- what differs is that a 3-form cell used to be evaluated at
    ``((n_knots + 1) q)^3`` points, ~1e5 at n = 6, and is now evaluated at
    ``((1 + k_max) q)^3``, with ``k_max`` the largest number of interior knots
    in any span: 0 or 1 on a uniform knot vector.)

    Spans and knots are concrete, so the bookkeeping is host-side numpy; it is
    O(n) scalars per axis and runs once per sequence (see
    :func:`_greville_data`).

    Returns
    -------
    xs, ws : (n_spans, n_pts) quadrature points and weights, row per span.
    """
    xi_ref, w_ref = np.polynomial.legendre.leggauss(
        _quadrature_order_from_basis_1d(basis))
    spans = np.asarray(spans)
    knots = np.unique(np.asarray(basis.T))
    cuts = [np.concatenate([[a], knots[(knots > a) & (knots < b)], [b]])
            for a, b in spans]
    width = max(len(c) for c in cuts)
    cuts = np.stack([np.concatenate([c, np.full(width - len(c), c[-1])])
                     for c in cuts])
    lo, hi = cuts[:, :-1], cuts[:, 1:]
    centers = 0.5 * (lo + hi)
    halfwidths = 0.5 * (hi - lo)
    xs = (centers[:, :, None] + halfwidths[:, :, None] * xi_ref).reshape(len(spans), -1)
    ws = (halfwidths[:, :, None] * w_ref).reshape(len(spans), -1)
    return jnp.asarray(xs), jnp.asarray(ws)


class _GrevilleAxis(NamedTuple):
    """Everything Greville interpolation needs about one logical axis."""
    coll: Array          # (n, n)   parent-basis collocation matrix at the Greville points
    hist: Array          # (nd, nd) derivative-basis histopolation matrix on the Greville spans
    point_rule: tuple    # (pts[:, None], ones): one point per cell, unit weight
    span_rule: tuple     # (xs, ws) of _span_quadrature: one Gauss rule per cell


def greville_axes(seq) -> tuple[_GrevilleAxis, _GrevilleAxis, _GrevilleAxis]:
    """Per-axis Greville data of ``seq``: points, spans, collocation and
    histopolation matrices and the span quadrature. They depend on the 1-D
    knot vectors only, so the sequence builds them once in ``__init__``
    (``seq.greville``).
    """
    axes = []
    for lam, d in zip(seq.basis_0.Λ, seq.basis_0.dΛ):
        pts = lam.greville_points()
        axes.append(_GrevilleAxis(
            coll=lam.collocation_matrix(pts),
            hist=d.histopolation_matrix(),
            point_rule=(pts[:, None], jnp.ones((pts.shape[0], 1), dtype=pts.dtype)),
            span_rule=_span_quadrature(d, d.greville_spans())))
    return tuple(axes)


#: PERIODIC SPANS CROSS THE SEAM AT EVEN p.  Periodic Greville points sit ON
#: knots for odd p and HALFWAY between knots for even p, so at even p the last
#: sorted span is [1 - h/2, 1 + h/2] -- and at odd p a point that is 0 up to
#: rounding can wrap to 1 - eps and cross it too (n=6, p=3 did).  The moments
#: below wrap their quadrature points (``_wrap_periodic_point``) and so
#: integrate the periodic extension; ``SplineBasis.evaluate`` does NOT extend
#: periodically past x = 1 (the image of basis function p' is missing from the
#: extended knot vector), so ``histopolation_matrix`` wraps its points too.
#: Until it did, H and the moments shared the RULE but not the INTEGRAND, and
#: k >= 1 was not a projector at even p (7e-2 .. 1.3e-1) while passing at odd
#: p on the power-of-two fixtures -- the loophole in "same rule => m = H c".
#: See docs/research/handoff_2026-08-25_histopolation.md, section 7.
#:
#: Interpolation and histopolation are done on the FULL tensor-product space
#: and then restricted onto the extracted space.  That composition is the
#: construction of Guclu & Campos Pinto (arXiv:2505.15996),
#: ``Pi_Z = P_Z . Pi_W``: the tensor-product geometric projector followed by a
#: local, explicit, matrix-free conforming projection on the coefficients.
#:
#: RETRACTION.  Commit 1cf9cbd's message, and an earlier version of this
#: comment, said "idempotency comes from the coefficient rules being
#: self-consistent, not from any biorthogonality condition on the extraction".
#: That is TRUE OF THE PAPER'S ``P_Z`` AND FALSE OF MRX'S EXTRACTION -- the
#: claim was imported across an operator boundary it does not cross.  MRX's
#: ``E`` is not ``P_Z``: measured, ``||E E^T - I||_max = 1.556`` at k=1 and
#: ``0.352`` at k=2.  So
#: ``e @ c_full`` alone is NOT a projector, and the k=0 round-trip duly came
#: back at 5.29e-01.  Supplying ``(E E^T)^{-1}`` explicitly, as
#: :func:`_conforming_restriction` does, is therefore the CORRECT CONSTRUCTION
#: for a non-biorthogonal extraction -- not a fudge factor bolted on to force a
#: test green.  With it, k=0 round-trips at 2.5e-16.
#:
#: The same retracted claim was the stated justification for removing the two
#: guards below.  Removing them still looks right -- the construction does work
#: once the extraction is handled properly -- but the REASON recorded at the
#: time was not sound.  History is not rewritten; this is the retraction.
#:
#: Two guards used to sit here and both were stale:
#:   * a full-tensor-space check, which rejected every nontrivial extraction --
#:     i.e. both ``dirichlet=True`` and ``polar=True`` -- although the restrict
#:     step above is exactly what makes those cases work;
#:   * a clamped-only check on the histopolation axes, although
#:     ``SplineBasis.greville_spans`` has handled periodic axes (wrapping the
#:     final span by +1) for as long as it has existed, and raises its own
#:     NotImplementedError for anything it does not support.
#:
#: ``test_projectors.py`` pins the property that matters: interpolating a
#: function that already lives in the target space returns its own DOFs.


def _conforming_restriction(e, c_full):
    """Restrict full tensor-product coefficients onto the extracted space.

    ``a = (E E^T)^{-1} E c_full``.  ``E^T (E E^T)^{-1} E`` is idempotent, so
    interpolating a function that ALREADY lies in the extracted space returns
    its own DOFs exactly.  Plain ``e @ c_full`` does not have that property --
    measured, the k=0 round-trip came back at 5.29e-01 without this.

    ``E`` is a pure SELECTION on every row but the polar ones (the same
    ``counts > 1`` discriminator ``block_jacobi_laplacian.core_rows`` uses), and
    the polar surgery acts only in (rho, theta) while the zeta index is carried
    along untouched.  So ``E E^T`` is the IDENTITY PLUS SMALL DENSE BLOCKS --
    one per zeta slice per affected component, of size ``n_polar``.  Confirmed
    against the component sizes: k=1 contributes ``2*nz + 3*dz`` such rows and
    k=2 contributes ``2*dz``, reproducing the measured 30-of-606 and 12-of-588
    exactly; k=3 contributes none and is already a pure selection.

    The blocks are inverted DENSELY -- the same separable-bulk-plus-dense-core
    idiom as ``BlockJacobiMass``, and for the same reason: an ``E+``
    pseudoinverse is what that analysis rejected.  For a pure selection (k=3)
    this returns immediately.

    TODO: cache on the sequence alongside the collocation matrices if it ever
    shows up in a profile; the blocks depend only on the extraction.
    """
    a = e @ c_full
    rows = np.asarray(e.rows)
    counts = np.bincount(rows, minlength=int(e.forward_shape[0]))
    core = counts > 1
    if not core.any():
        return a                       # pure selection: E E^T = I already

    E = sparse.csr_matrix(
        (np.asarray(e.vals), (rows, np.asarray(e.cols))), shape=e.forward_shape)
    gram = (E @ E.T).tocsr()
    _, labels = csgraph.connected_components(gram, directed=False)

    out = np.asarray(a).copy()
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(labels.max() + 2))
    for lab in np.unique(labels[core]):
        idx = order[bounds[lab]:bounds[lab + 1]]
        out[idx] = np.linalg.solve(gram[np.ix_(idx, idx)].toarray(), out[idx])
    return jnp.asarray(out)


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
        return seq.E_bc(k)
    elif dirichlet:
        return seq.E(k, True)
    else:
        return seq.E(k)


# ---------------------------------------------------------------------------
# Load vector assembly  (matrix-free, works on any seq after set_map)
# ---------------------------------------------------------------------------

def load(seq: "DeRhamSequence", f, k: int,
         dirichlet: bool = False, bc: bool = False,
         frame: str = 'phys'):
    """Assemble the dual k-form load vector  v_i = ∫ Λ^k_i · f(ξ) w(ξ) dξ.

    Parameters
    ----------
    seq : DeRhamSequence
    f : callable  ξ → (1,) for k=0,3;  ξ → (3,) for k=1,2.
        Arguments are logical coordinates.  Interpretation depends on `frame`.
    k : int  Form degree (0, 1, 2, 3).
    dirichlet : bool  Use Dirichlet-constrained DOFs.
    bc : bool  Use boundary-trace DOFs (takes precedence over dirichlet).
    frame : {'phys', 'ref'}
        ``'phys'`` (default): ``f`` returns components in the physical frame;
        a DF-based pullback is applied internally.  This (with
        :func:`load_grid_field`) is the only consumer of the raw map
        Jacobian, which the geometry does not store: ``DF`` is recomputed at
        the quadrature points with :func:`mrx.geometry.map_jacobian_at`, once
        per call.

        ``'ref'``: ``f`` returns the coefficients of the k-form expanded
        directly in reference coordinates dr, dχ, dζ (and their wedge
        products); no pullback is applied. Concretely:

        - k=0: scalar ``u(ξ)``
        - k=1: covariant ref components ``(u_r, u_χ, u_ζ)``
        - k=2: ref 2-form proxy ``(u_χζ, u_rζ, u_rχ)`` (same slot order as
          ``_form_comp_info(2)``)
        - k=3: scalar coefficient ``A(ξ)`` in ``A dr∧dχ∧dζ`` (i.e.
          ``A = f_phys·J``)

    Returns
    -------
    Array  Dual load vector of length n_k (or n_k_dbc / n_k_bc).
    """
    if frame not in ('phys', 'ref'):
        raise ValueError(f"frame must be 'phys' or 'ref', got {frame!r}")

    e = _extraction(seq, k, dirichlet, bc)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    comp_info, comp_shapes = seq._form_comp_info(k)

    if k == 0:
        # Scalar: reference and physical frames are identical.
        f_jk = jax.lax.map(
            lambda x: _as_single_component(f(x)),
            seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]

    elif k == 1:
        if frame == 'phys':
            # DF^{-1} v = G^{-1} DF^T v  (G = DF^T DF).  DF is not stored on
            # the geometry (this pullback and load_grid_field are its only
            # consumers); it is recomputed here, once per load.
            v_q = jax.lax.map(f, seq.quad.x,
                               batch_size=mrx.MAP_BATCH_SIZE_INNER)
            DF_q = map_jacobian_at(seq.map, seq.quad.x)
            DFt_v = jnp.einsum('qji,qj->qi', DF_q, v_q)         # DF^T @ v
            f_jk = jnp.einsum('qij,qj->qi', seq.metric_inv_jkl, DFt_v)
        else:
            f_jk = jax.lax.map(f, seq.quad.x,
                                batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]

    elif k == 2:
        if frame == 'phys':
            # 2-form pullback DF^T v, with DF recomputed on demand (see k=1).
            v_q = jax.lax.map(f, seq.quad.x,
                               batch_size=mrx.MAP_BATCH_SIZE_INNER)
            DF_q = map_jacobian_at(seq.map, seq.quad.x)
            f_jk = jnp.einsum('qji,qj->qi', DF_q, v_q)          # DF^T @ v
        else:
            f_jk = jax.lax.map(f, seq.quad.x,
                                batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * seq.quad.w[:, None]

    elif k == 3:
        f_jk = jax.lax.map(
            lambda x: _as_single_component(f(x)),
            seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        if frame == 'phys':
            w_jk = f_jk * seq.quad.w[:, None]
        else:
            # A = f_phys * J  ⟹  f_phys = A / J,  weight = w (no J factor)
            w_jk = f_jk * (seq.quad.w / seq.jacobian_j)[:, None]

    else:
        raise ValueError(f"k must be 0, 1, 2 or 3, got {k}")

    return e @ integrate_against(w_jk, comp_info, comp_shapes, quad_shape)


# ---------------------------------------------------------------------------
# Interpolation / histopolation  (collocation matrices built lazily)
# ---------------------------------------------------------------------------

def interpolate(seq: "DeRhamSequence", f, k: int, dirichlet: bool = False,
                frame: str = 'phys'):
    """Compute primal DOFs by Greville interpolation (k=0) or histopolation (k=1,2,3).

    The Greville points, spans, collocation/histopolation matrices and span
    quadrature rules are built once per sequence and cached on it
    (:func:`_greville_data`); each call evaluates ``f`` and solves.

    Parameters
    ----------
    seq : DeRhamSequence
    f : callable  ξ → (1,) for k=0,3;  ξ → (3,) for k=1,2.
    k : int  Form degree (0, 1, 2, 3).
    dirichlet : bool  Use Dirichlet-constrained DOFs.
    frame : {'phys', 'ref'}
        'phys' (default): ``f`` returns components in the physical frame, which
        are pulled back before the moments are taken.  'ref': ``f`` already
        returns reference-frame components, in exactly the convention
        :func:`load` takes with ``frame='ref'`` -- that is, the pulled-back
        integrand ``DF^T v``, NOT the primal coefficient vector.

        These are not the same object for k = 2.  ``M_2`` carries a ``g/J``
        weight (``M2_ij = int Lambda_i^T g Lambda_j / J``), so the DoFs from
        ``M_2^{-1} load`` are the PRIMAL components ``omega`` with
        ``B_phys = DF omega / J`` -- what
        :class:`~mrx.differential_forms.DiscreteFunction` evaluates -- whereas
        ``frame='ref'`` wants ``g omega / J``.  To build a field from known
        primal components, push them forward and use ``frame='phys'``; the
        pullback recovers ``g omega / J`` on its own.

        Only k = 1 and k = 2 have a pullback in this path; k = 0 is a scalar,
        for which the two frames coincide.  k = 3 is rejected: its
        histopolation carries no Jacobian factor, so its convention does not
        line up with :func:`load`'s and resolving that is a separate question.

    Returns
    -------
    Array  Primal DOF vector.
    """
    if frame not in ('phys', 'ref'):
        raise ValueError(f"frame must be 'phys' or 'ref', got {frame!r}")
    if k == 0:
        return _interpolate_0form(seq, f, dirichlet)
    elif k == 1:
        return _histopolate_1form(seq, f, dirichlet, frame)
    elif k == 2:
        return _histopolate_2form(seq, f, dirichlet, frame)
    elif k == 3:
        if frame == 'ref':
            raise ValueError(
                "frame='ref' is not defined for k=3 histopolation; see the "
                "`frame` note in interpolate.__doc__")
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


def _oneform_pullback(seq, v, frame: str = 'phys'):
    """Physical 1-form proxy -> PRIMAL reference components.

    ``Pushforward`` (differential_forms.py:301) is the authority:
    ``F_* omega = (DF^T)^-1 omega`` at k=1, so ``omega = DF^T v_phys``.

    This is NOT what ``load`` uses.  ``load`` builds a DUAL vector and pairs
    against the pushed-forward basis, giving ``DF^-1 v``, and ``M_1^{-1}``
    then converts back to primal -- correct for that path.  Histopolation has
    no mass solve to undo the weight, so it needs the primal pullback directly.
    ``DF^-1`` here was the k=-1 VECTOR-FIELD rule, off by ``G^-1``.

    The transpose has the side benefit of being finite on the polar axis, where
    ``det DF -> 0`` and the clamped radial Greville points land exactly.
    """
    if frame == 'ref':
        return lambda x: v(_wrap_periodic_point(seq, x))
    DF = jax.jacfwd(seq.map)

    def pullback(x):
        x_eval = _wrap_periodic_point(seq, x)
        return DF(x_eval).T @ v(x_eval)

    return pullback


def _interpolate_0form(seq, f, dirichlet: bool) -> Array:
    """Greville collocation for a scalar 0-form."""
    e = seq.E(0, True) if dirichlet else seq.E(0)
    exact = _matching_discrete_dofs(f, seq.basis_0, e)
    if exact is not None:
        return exact

    axes = seq.greville
    x_r, x_t, x_z = (ax.point_rule[0][:, 0] for ax in axes)

    r, t, z = jnp.meshgrid(x_r, x_t, x_z, indexing='ij')
    pts = jnp.stack([r.ravel(), t.ravel(), z.ravel()], axis=-1)
    values = jax.lax.map(
        lambda xi: _as_single_component(f(xi)), pts,
        batch_size=mrx.MAP_BATCH_SIZE_INNER,
    ).reshape(len(x_r), len(x_t), len(x_z))

    coeffs = values
    for j, ax in enumerate(axes):
        coeffs = _solve_tensor_collocation_axis(ax.coll, coeffs, axis=j)
    return _conforming_restriction(e, coeffs.reshape(-1))


def _twoform_pullback(seq, v, frame: str = 'phys'):
    """Physical 2-form proxy -> PRIMAL reference components.

    ``Pushforward`` gives ``F_* omega = DF omega / J`` at k=2, so
    ``omega = J DF^-1 v = adj(DF) v``.  ``DF^T`` -- what this used to be -- is
    ``load``'s DUAL pairing, correct there because ``M_2^{-1}`` undoes the
    ``g/J`` weight afterwards, and wrong here where nothing does: it returned
    ``g omega / J``.

    ``adj(DF)`` is built from the cofactors (:func:`~mrx.differential_forms.adj33`)
    rather than as ``det * inv``, so it stays finite where ``det DF -> 0``.
    """
    if frame == 'ref':
        return lambda x: v(_wrap_periodic_point(seq, x))
    DF = jax.jacfwd(seq.map)

    def pullback(x):
        x_eval = _wrap_periodic_point(seq, x)
        return adj33(DF(x_eval)) @ v(x_eval)

    return pullback


def _greville_moments(seq, fn, rules) -> Array:
    """Integrate the scalar ``fn`` over every cell of a tensor-product rule.

    ``rules`` holds, per logical axis, ``(xs, ws)`` of shape ``(n_j, q_j)``: a
    point axis has ``q_j = 1`` with unit weight, a histopolated axis carries the
    split Gauss rule of :func:`_span_quadrature`.  The result has shape
    ``(n_r, n_t, n_z)``.

    One ``lax.map`` over the flattened cell grid, ``batch_size =
    mrx.MAP_BATCH_SIZE_INNER``, with the ``q_r q_t q_z`` points of a cell
    vmapped inside it.  This used to be a triple-nested Python comprehension
    that dispatched one ``lax.map`` per cell -- ``n_r n_t n_z`` separate
    device programs per component.
    """
    sizes = [xs.shape[0] for xs, _ in rules]
    idx = [i.ravel() for i in jnp.meshgrid(
        *[jnp.arange(n) for n in sizes], indexing='ij')]
    cells = tuple((xs[i], ws[i]) for (xs, ws), i in zip(rules, idx))

    def integrate(cell):
        (xr, wr), (xt, wt), (xz, wz) = cell
        rr, tt, zz = jnp.meshgrid(xr, xt, xz, indexing='ij')
        x = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
        w = (wr[:, None, None] * wt[None, :, None] * wz[None, None, :]).ravel()
        return jnp.sum(jax.vmap(fn)(x) * w)

    return jax.lax.map(
        integrate, cells, batch_size=mrx.MAP_BATCH_SIZE_INNER).reshape(sizes)


def _histopolate_vector(seq, pullback, e, histopolated) -> Array:
    """Greville histopolation of a vector-valued form, component by component.

    ``histopolated(c, j)`` says whether component ``c`` is histopolated
    (span rule, histopolation matrix) or collocated (Greville point,
    collocation matrix) along axis ``j``: for a 1-form the component's own
    axis, for a 2-form the two others.
    """
    axes = seq.greville
    coeffs = []
    for c in range(3):
        rules = tuple(ax.span_rule if histopolated(c, j) else ax.point_rule
                      for j, ax in enumerate(axes))
        m = _greville_moments(seq, lambda x, c=c: pullback(x)[c], rules)
        for j, ax in enumerate(axes):
            m = _solve_tensor_collocation_axis(
                ax.hist if histopolated(c, j) else ax.coll, m, axis=j)
        coeffs.append(m.reshape(-1))
    return _conforming_restriction(e, jnp.concatenate(coeffs))


def _histopolate_1form(seq, v, dirichlet: bool, frame: str = 'phys') -> Array:
    """Greville histopolation for a 1-form."""
    e = seq.E(1, True) if dirichlet else seq.E(1)
    exact = _matching_discrete_dofs(v, seq.basis_1, e)
    if exact is not None:
        return exact
    return _histopolate_vector(
        seq, _oneform_pullback(seq, v, frame), e, lambda c, j: j == c)


def _histopolate_2form(seq, v, dirichlet: bool, frame: str = 'phys') -> Array:
    """Greville histopolation for a 2-form.

    Periodic Greville spans run past 1 (greville_spans wraps the last one by
    +1), so evaluation points must be folded back before v or the map sees
    them -- the pullbacks do that via ``_wrap_periodic_point``.
    """
    e = seq.E(2, True) if dirichlet else seq.E(2)
    exact = _matching_discrete_dofs(v, seq.basis_2, e)
    if exact is not None:
        return exact
    return _histopolate_vector(
        seq, _twoform_pullback(seq, v, frame), e, lambda c, j: j != c)


def _histopolate_3form(seq, f, dirichlet: bool) -> Array:
    """Greville histopolation for a scalar 3-form."""
    e = seq.E(3, True) if dirichlet else seq.E(3)
    exact = _matching_discrete_dofs(f, seq.basis_3, e)
    if exact is not None:
        return exact

    axes = seq.greville
    moments = _greville_moments(
        seq, lambda x: _as_single_component(f(_wrap_periodic_point(seq, x)))[0],
        tuple(ax.span_rule for ax in axes))
    coeffs = moments
    for j, ax in enumerate(axes):
        coeffs = _solve_tensor_collocation_axis(ax.hist, coeffs, axis=j)
    return _conforming_restriction(e, coeffs.reshape(-1))


# ---------------------------------------------------------------------------
# Gridded data
# ---------------------------------------------------------------------------
#
# The production route from a GVEC equilibrium is the closed-form state file 
# (mrx.gvec: StateField collocated at the Greville points, lambda histopolated 
# from the series), which makes an interpolatory fit of gridded samples unnecessary 
# and measurably worse. 
# This is the projector for data that only exists on a grid.

def load_grid_field(axes, values, seq, k, *, dirichlet=False, frame='ref',
                    degree=3):
    """Factorized dual load of a field sampled on a regular logical grid.

    Interpolatory-spline analogue of the pointwise ``seq.load(callable)`` for
    grid-sampled data.  Steps, all sum-factorized (no pointwise ``lax.map`` and
    no per-quad-point basis sweep):

    1. fit an interpolatory tensor-product B-spline to ``values`` — one square
       collocation solve per axis (``n_basis = n_data``);
    2. evaluate it at ``seq``'s quadrature grid via :func:`_tp_evaluate` — three
       1D contractions, ``O(N_q (n1+n2+n3))`` instead of ``O(N_q·n1·n2·n3)``;
    3. apply the k-form frame pullback and quadrature weight (mirrors
       :func:`mrx.projectors.load`; ``frame='phys'`` recomputes ``DF`` at the
       quadrature points with :func:`mrx.geometry.map_jacobian_at` -- this
       and ``load`` are the only consumers of ``DF``, which the geometry does
       not store);
    4. integrate against the k-form basis and extract.

    Returns the **dual load vector** (same as ``seq.load``); pass it to
    ``seq.apply_inverse_mass_matrix`` for the projected DOFs.

    Parameters
    ----------
    axes : tuple of 1-D arrays ``(x1, x2, x3)``  logical grid nodes per axis.
    values : array  ``(n1,n2,n3)`` for k=0,3;  ``(n1,n2,n3,3)`` for k=1,2
        (flattened variants accepted).
    seq : DeRhamSequence  target sequence.
    k : {0,1,2,3}  form degree.
    dirichlet : bool  use Dirichlet-constrained DOFs.
    frame : {'ref','phys'}  interpretation of ``values`` (see
        :func:`mrx.projectors.load`).
    degree : int  spline degree of the interpolatory fit.
    """

    if frame not in ('ref', 'phys'):
        raise ValueError(f"frame must be 'ref' or 'phys', got {frame!r}")

    x1, x2, x3 = (jnp.asarray(a) for a in axes)
    n1, n2, n3 = len(x1), len(x2), len(x3)
    ncomp = 1 if k in (0, 3) else 3
    C = jnp.asarray(values).reshape(n1, n2, n3) if ncomp == 1 \
        else jnp.asarray(values).reshape(n1, n2, n3, 3).transpose(3, 0, 1, 2)
    C = C.reshape(ncomp, n1, n2, n3)

    # 1. interpolatory fit basis (n_basis = n_data per axis, seq's BC types)
    fit = DifferentialForm(0, (n1, n2, n3), (degree,) * 3, seq.basis_0.types)
    br, bt, bz = fit.Λ
    solve = (br.collocation_matrix(x1), bt.collocation_matrix(x2),
             bz.collocation_matrix(x3))
    for a in range(3):
        C = _solve_tensor_collocation_axis(solve[a], C, axis=a + 1)  # comp is axis 0

    # 2. factorized evaluation at seq's quadrature grid.  M_axis = fit basis at
    #    seq's per-axis 1D quad points (r<->x_x, t<->x_y, z<->x_z).
    Mr = br.collocation_matrix(seq.quad.x_x).T          # (n1, nqr)
    Mt = bt.collocation_matrix(seq.quad.x_y).T          # (n2, nqt)
    Mz = bz.collocation_matrix(seq.quad.x_z).T          # (n3, nqz)
    f = _tp_evaluate(C, Mr, Mt, Mz)                     # (ncomp, nqr, nqt, nqz)
    # flatten with the same (0,2,1,3) transpose the geometry path uses, so the
    # per-quad-point order matches seq.quad.x / seq.quad.w (meshgrid 'xy': t,r,z).
    f_q = f.transpose(0, 2, 1, 3).reshape(ncomp, -1).T  # (n_q, ncomp)

    # 3. frame pullback + quadrature weight (mirrors mrx.projectors.load)
    w = seq.quad.w
    if k == 0:
        w_jk = f_q * (w * seq.jacobian_j)[:, None]
    elif k == 1:
        if frame == 'phys':                             # DF^-1 f = G^-1 DF^T f
            DF_q = map_jacobian_at(seq.map, seq.quad.x)
            DFt = jnp.einsum('qji,qj->qi', DF_q, f_q)
            f_q = jnp.einsum('qij,qj->qi', seq.metric_inv_jkl, DFt)
        w_jk = f_q * (w * seq.jacobian_j)[:, None]
    elif k == 2:
        if frame == 'phys':                             # DF^T f
            DF_q = map_jacobian_at(seq.map, seq.quad.x)
            f_q = jnp.einsum('qji,qj->qi', DF_q, f_q)
        w_jk = f_q * w[:, None]
    else:  # k == 3
        w_jk = f_q * (w if frame == 'phys' else w / seq.jacobian_j)[:, None]

    # 4. integrate against the k-form basis + extraction
    comp_info, comp_shapes = seq._form_comp_info(k)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    e = seq.E(k, dirichlet)
    return e @ integrate_against(w_jk, comp_info, comp_shapes, quad_shape)
