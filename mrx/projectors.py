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

from typing import TYPE_CHECKING, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import sparse
from scipy.sparse import csgraph

import mrx
from mrx.extraction_operators import get_xi
from mrx.differential_forms import adj33, inv33
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


def _leggauss_rule(order: int) -> tuple[Array, Array]:
    xi, w = np.polynomial.legendre.leggauss(order)
    return jnp.asarray(xi), jnp.asarray(w)


def _interval_rule(span: Array, order: int, knots: Array | None = None
                   ) -> tuple[Array, Array]:
    """Gauss rule on ``span``, SPLIT at any knots the span contains.

    Greville spans straddle an interior knot whenever the degree is EVEN --
    ``g_i = i + (p+1)/2`` in units of the knot spacing, integral for odd p and
    half-integral for even p -- and Gauss-Legendre is exact only for
    POLYNOMIALS.  A spline has a derivative jump at each knot, so a single rule
    spanning one converges algebraically rather than exactly: measured on an
    off-centre knot, 40 points still left 3.6e-07 where splitting is exact at 2.

    ``SplineBasis.histopolation_matrix`` splits identically, so the matrix and
    the moments are built with the SAME rule -- which matters independently of
    exactness, since ``solve(H, m) = c`` needs ``m = H c`` and quadrature is
    linear.
    """
    xi_ref, w_ref = _leggauss_rule(order)
    a, b = span
    if knots is None:
        center = 0.5 * (a + b)
        halfwidth = 0.5 * (b - a)
        return center + halfwidth * xi_ref, halfwidth * w_ref

    cuts = jnp.clip(jnp.unique(knots), a, b)
    cuts = jnp.sort(jnp.concatenate([jnp.array([a]), cuts, jnp.array([b])]))
    lo, hi = cuts[:-1], cuts[1:]
    centers = 0.5 * (lo + hi)
    halfwidths = 0.5 * (hi - lo)
    xs = (centers[:, None] + halfwidths[:, None] * xi_ref[None, :]).reshape(-1)
    ws = (halfwidths[:, None] * w_ref[None, :]).reshape(-1)
    return xs, ws


def _quadrature_order_from_basis_1d(basis) -> int:
    return max(2, basis.p + 2)


#: ***********************************************************************
#: *  KNOWN LIMITATION -- EVEN SPLINE DEGREE.  READ BEFORE RELYING ON     *
#: *  interpolate() AT k >= 1.                                            *
#: *                                                                      *
#: *  At ODD p these operators are EXACT: interpolating a function that   *
#: *  already lies in the target space returns its own DOFs to machine    *
#: *  precision, at every k = 0,1,2,3 and both BCs (measured ~5e-16).     *
#: *                                                                      *
#: *  At EVEN p, k >= 1 is NOT a projector -- the same round-trip comes   *
#: *  back at 7e-2 to 1.3e-1.  k = 0 is exact at both parities.           *
#: *  THE CAUSE IS UNKNOWN.  Two mechanisms were proposed and both were   *
#: *  REFUTED by measurement:                                             *
#: *    - the extraction: k=3 has E E^T = I to 0.000 and still fails;     *
#: *    - quadrature exactness: splitting spans at knots did NOT fix it   *
#: *      and made it slightly WORSE (5.7e-2 -> 7.1e-2).  It could not    *
#: *      have been the cause: solve(H, m) = c needs only m = H c, and    *
#: *      that holds by LINEARITY whenever H and m share a rule, exact    *
#: *      or not.                                                         *
#: *                                                                      *
#: *  conf/config_relax_from_nfs.yaml runs ps = 4, which is EVEN.  Its    *
#: *  ingest goes through collocation (n_basis = n_data) so it is         *
#: *  PROBABLY unaffected -- that has NOT been verified per call site.    *
#: *                                                                      *
#: *  test_projectors.py parametrises the identity fixture over p=2 and   *
#: *  p=3, so the p2 cases fail on purpose and are the standing reminder. *
#: *  See docs/research/handoff_2026-08-25_histopolation.md.              *
#: ***********************************************************************
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
#: ``0.352`` at k=2 (``scripts/debug/extraction_unitarity_probe.py``).  So
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
    pseudoinverse is what that analysis rejected.  For a pure selection (k=3,
    and every non-polar sequence) this returns immediately.

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
        return getattr(seq, f'e{k}_bc')
    elif dirichlet:
        return getattr(seq, f'e{k}_dbc')
    else:
        return getattr(seq, f'e{k}')


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
        'phys' (default): f returns components in the physical frame.
            A DF-based pullback is applied internally.
        'ref': f returns the coefficients of the k-form expanded directly in
            reference coordinates dr, dχ, dζ (and their wedge products).
            No pullback is applied.  Concretely:
              k=0: scalar  u(ξ)
              k=1: covariant ref components  (u_r, u_χ, u_ζ)
              k=2: ref 2-form proxy  (u_χζ, u_rζ, u_rχ)  (same slot order as
                   _form_comp_info(2))
              k=3: scalar coefficient A(ξ) in  A dr∧dχ∧dζ  (i.e. A = f_phys·J)

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
            # DF^{-1} v = G^{-1} DF^T v  (G = DF^T DF); reuse the precomputed
            # DF and inverse metric instead of re-running jax.jacfwd on the map.
            v_q = jax.lax.map(f, seq.quad.x,
                               batch_size=mrx.MAP_BATCH_SIZE_INNER)
            DFt_v = jnp.einsum('qji,qj->qi', seq.DF_jkl, v_q)   # DF^T @ v
            f_jk = jnp.einsum('qij,qj->qi', seq.metric_inv_jkl, DFt_v)
        else:
            f_jk = jax.lax.map(f, seq.quad.x,
                                batch_size=mrx.MAP_BATCH_SIZE_INNER)
        w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]

    elif k == 2:
        if frame == 'phys':
            # 2-form pullback DF^T v; reuse the precomputed DF instead of
            # re-running jax.jacfwd on the map.
            v_q = jax.lax.map(f, seq.quad.x,
                               batch_size=mrx.MAP_BATCH_SIZE_INNER)
            f_jk = jnp.einsum('qji,qj->qi', seq.DF_jkl, v_q)    # DF^T @ v
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

    Collocation and histopolation matrices are built lazily on each call.
    TODO: cache them on the sequence object if profiling shows this is a
    bottleneck.

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
    e = seq.e0_dbc if dirichlet else seq.e0
    exact = _matching_discrete_dofs(f, seq.basis_0, e)
    if exact is not None:
        return exact

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


def _full_oneform_histopolation_dofs(seq, v, frame: str = 'phys'):
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
    pullback = _oneform_pullback(seq, v, frame)

    def integrate_component_0(span_r, t_val, z_val):
        xs_r, ws_r = _interval_rule(span_r, q_r, d_r.T)
        x = jnp.stack([xs_r, jnp.full(xs_r.shape, t_val),
                        jnp.full(xs_r.shape, z_val)], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 0]
        return jnp.sum(vals * ws_r)

    def integrate_component_1(r_val, span_t, z_val):
        xs_t, ws_t = _interval_rule(span_t, q_t, d_t.T)
        x = jnp.stack([jnp.full(xs_t.shape, r_val), xs_t,
                        jnp.full(xs_t.shape, z_val)], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 1]
        return jnp.sum(vals * ws_t)

    def integrate_component_2(r_val, t_val, span_z):
        xs_z, ws_z = _interval_rule(span_z, q_z, d_z.T)
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


def _histopolate_1form(seq, v, dirichlet: bool, frame: str = 'phys') -> Array:
    """Greville histopolation for a 1-form."""
    e = seq.e1_dbc if dirichlet else seq.e1
    exact = _matching_discrete_dofs(v, seq.basis_1, e)
    if exact is not None:
        return exact

    lam_r, lam_t, lam_z = seq.basis_0.Λ
    d_r, d_t, d_z = seq.basis_0.dΛ

    # TODO: cache these on seq if matrix build time is significant
    coll_r = lam_r.collocation_matrix()
    coll_t = lam_t.collocation_matrix()
    coll_z = lam_z.collocation_matrix()
    hist_r = d_r.histopolation_matrix()
    hist_t = d_t.histopolation_matrix()
    hist_z = d_z.histopolation_matrix()

    comp0, comp1, comp2 = _full_oneform_histopolation_dofs(seq, v, frame)

    c0 = _solve_tensor_collocation_axis(hist_r, comp0, axis=0)
    c0 = _solve_tensor_collocation_axis(coll_t, c0, axis=1)
    c0 = _solve_tensor_collocation_axis(coll_z, c0, axis=2)

    c1 = _solve_tensor_collocation_axis(coll_r, comp1, axis=0)
    c1 = _solve_tensor_collocation_axis(hist_t, c1, axis=1)
    c1 = _solve_tensor_collocation_axis(coll_z, c1, axis=2)

    c2 = _solve_tensor_collocation_axis(coll_r, comp2, axis=0)
    c2 = _solve_tensor_collocation_axis(coll_t, c2, axis=1)
    c2 = _solve_tensor_collocation_axis(hist_z, c2, axis=2)

    return _conforming_restriction(
        e, jnp.concatenate([c0.reshape(-1), c1.reshape(-1), c2.reshape(-1)]))


def _histopolate_2form(seq, v, dirichlet: bool, frame: str = 'phys') -> Array:
    """Greville histopolation for a 2-form."""
    e = seq.e2_dbc if dirichlet else seq.e2
    exact = _matching_discrete_dofs(v, seq.basis_2, e)
    if exact is not None:
        return exact

    d_r, d_t, d_z = seq.basis_0.dΛ

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

    # Periodic Greville spans run past 1 (greville_spans wraps the last one by
    # +1), so evaluation points must be folded back before v or the map sees
    # them -- the 1-form path does the same via _oneform_pullback.
    pullback = _twoform_pullback(seq, v, frame)

    def int0(r_val, span_t, span_z):
        q_t = _quadrature_order_from_basis_1d(d_t.s)
        q_z = _quadrature_order_from_basis_1d(d_z.s)
        xs_t, ws_t = _interval_rule(span_t, q_t, d_t.T)
        xs_z, ws_z = _interval_rule(span_z, q_z, d_z.T)
        tt, zz = jnp.meshgrid(xs_t, xs_z, indexing='ij')
        x = jnp.stack([jnp.full(tt.size, r_val), tt.ravel(), zz.ravel()], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 0]
        return jnp.sum(vals * (ws_t[:, None] * ws_z[None, :]).reshape(-1))

    def int1(span_r, t_val, span_z):
        q_r = _quadrature_order_from_basis_1d(d_r.s)
        q_z = _quadrature_order_from_basis_1d(d_z.s)
        xs_r, ws_r = _interval_rule(span_r, q_r, d_r.T)
        xs_z, ws_z = _interval_rule(span_z, q_z, d_z.T)
        rr, zz = jnp.meshgrid(xs_r, xs_z, indexing='ij')
        x = jnp.stack([rr.ravel(), jnp.full(rr.size, t_val), zz.ravel()], axis=-1)
        vals = jax.lax.map(pullback, x, batch_size=mrx.MAP_BATCH_SIZE_INNER)[:, 1]
        return jnp.sum(vals * (ws_r[:, None] * ws_z[None, :]).reshape(-1))

    def int2(span_r, span_t, z_val):
        q_r = _quadrature_order_from_basis_1d(d_r.s)
        q_t = _quadrature_order_from_basis_1d(d_t.s)
        xs_r, ws_r = _interval_rule(span_r, q_r, d_r.T)
        xs_t, ws_t = _interval_rule(span_t, q_t, d_t.T)
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

    return _conforming_restriction(
        e, jnp.concatenate([c0.reshape(-1), c1.reshape(-1), c2.reshape(-1)]))


def _histopolate_3form(seq, f, dirichlet: bool) -> Array:
    """Greville histopolation for a scalar 3-form."""
    e = seq.e3_dbc if dirichlet else seq.e3
    exact = _matching_discrete_dofs(f, seq.basis_3, e)
    if exact is not None:
        return exact

    d_r, d_t, d_z = seq.basis_0.dΛ

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
        xs_r, ws_r = _interval_rule(span_r, q_r, d_r.T)
        xs_t, ws_t = _interval_rule(span_t, q_t, d_t.T)
        xs_z, ws_z = _interval_rule(span_z, q_z, d_z.T)
        rr, tt, zz = jnp.meshgrid(xs_r, xs_t, xs_z, indexing='ij')
        x = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
        values = jax.lax.map(
            lambda xi: _as_single_component(f(_wrap_periodic_point(seq, xi))), x,
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
    return _conforming_restriction(e, coeffs.reshape(-1))


# TODO: requires testing still
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
