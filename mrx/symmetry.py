r"""Stellarator symmetry: the reflection ``(r, theta, zeta) -> (r, -theta, -zeta)`` on the spline spaces.

A stellarator-symmetric equilibrium is invariant under this reflection, and each field is either
even or odd under it:

- odd (parity ``-1``): ``B``, ``A``, ``J``, ``H``, the harmonic forms
- even (parity ``+1``): the velocity, the force ``J x B``, the pressure and its gradient

A half-period sequence (``seq.half_period``) integrates over half of the zeta period only and
uses the parity of each field to recover the full-period result. This module supplies the pieces:
the reflection ``R`` of the raw spline coefficients (a signed index permutation, which needs
uniform periodic angular axes), the symmetrisation ``(x + parity R x) / 2`` that turns doubled
half-period integrals into full-period ones, the DoF spaces of one parity behind ``seq.odd`` and
``seq.even`` (:func:`parity_extraction`), and the parity projector :class:`FreeProjector` for
preconditioners. The sequence builds all of these itself when it is created.
"""
from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.extraction_operators import build_extraction, core_gram_inverse, get_xi, row_products, summed_coo
from mrx.precision import RESIDUAL_DTYPE


def reflection_permutation(n: int, p: int) -> np.ndarray:
    """The index permutation of ``x -> -x`` on the ``n`` uniform periodic B-splines of degree ``p``,
    ``B_j(-x) = B_{(p - 1 - j) mod n}(x)``. For a derivative basis pass its own ``(n, p)``."""
    return (p - 1 - np.arange(n)) % n


def is_uniform_periodic(basis) -> bool:
    """Whether ``basis`` is a uniform periodic B-spline basis on ``[0, 1]``."""
    if basis.type != "periodic":
        return False
    unique = np.asarray(basis.T[basis.p:basis.p + basis.n + 1])
    return bool(np.allclose(unique, np.linspace(0.0, 1.0, basis.n + 1), atol=1e-6, rtol=0.0))


#: The sign of each component (r, theta, zeta) of a 1-form or a 2-form under the reflection.
#: Both transform alike because the reflection has determinant ``+1``.
COMPONENT_SIGNS = (1.0, -1.0, -1.0)


def _component_axis_bases(form, k, c):
    """The three 1-D bases of component ``c`` of the k-form ``form``."""
    if k in (0, 3):
        return [form.Lambda[a] for a in range(3)] if k == 0 else [form.dLambda[a] for a in range(3)]
    bases = [form.Lambda[a] for a in range(3)] if k == 1 else [form.dLambda[a] for a in range(3)]
    bases[c] = form.dLambda[c] if k == 1 else form.Lambda[c]
    return bases


def reflection_plan(seq, k):
    """The reflection of the raw k-form coefficients of ``seq``, as a tuple with one entry
    ``(theta permutation, zeta permutation, sign, shape)`` per component. The tuple is hashable
    and is passed to :func:`reflect` as a static argument. Raises ``ValueError`` when an angular
    basis is not uniform periodic."""
    form = (seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)[k]
    n_comp = 3 if k in (1, 2) else 1
    plan = []
    for c in range(n_comp):
        bases = _component_axis_bases(form, k, c)
        for axis in (1, 2):
            if not is_uniform_periodic(bases[axis]):
                raise ValueError("the reflection is an index permutation only on uniform "
                                 "periodic angular bases")
        sign = COMPONENT_SIGNS[c] if n_comp == 3 else 1.0
        plan.append((tuple(reflection_permutation(bases[1].n, bases[1].p).tolist()),
                     tuple(reflection_permutation(bases[2].n, bases[2].p).tolist()),
                     sign, tuple(int(v) for v in form.shape[c])))
    return tuple(plan)


@functools.partial(jax.jit, static_argnames=("plan", "signed"))
def reflect(x, plan, signed=True):
    """``R x`` for a raw k-form coefficient vector ``x``. With ``signed=False`` the component
    signs are left out and only the angular indices are permuted."""
    out, off = [], 0
    for perm_t, perm_z, sign, shape in plan:
        n_c = int(np.prod(shape))
        X = x[off:off + n_c].reshape(shape)
        RX = X[:, jnp.asarray(perm_t), :][:, :, jnp.asarray(perm_z)]
        out.append(((sign if signed else 1.0) * RX).ravel())
        off += n_c
    return jnp.concatenate(out)


def symmetrize(x, plan, parity, signed=True):
    """``(x + parity R x) / 2``, the part of the raw k-form coefficient vector ``x`` with the
    given parity. ``parity`` is ``+1`` or ``-1``, a Python int or a traced scalar."""
    return 0.5 * (x + parity * reflect(x, plan, signed))


def parity_of(x, plan):
    """The parity ``+1.0`` or ``-1.0`` of a raw coefficient vector ``x`` of definite parity, as a
    traced scalar. It is the sign of ``x . R x``, which equals ``+-|x|^2``."""
    return jnp.where(jnp.vdot(x, reflect(x, plan)) >= 0.0, 1.0, -1.0).astype(x.dtype)


def mirror_zeta_1d(seq, M, derivative):
    """The full-period version of a 1-D matrix ``M`` assembled on the zeta axis of a half-period
    sequence. ``M = 2 int_half B_i B_j w`` with an even weight ``w`` becomes
    ``(M + P M P^T) / 2``, where ``P`` is the reflection of the primal zeta basis or, with
    ``derivative``, of the derivative basis. On a full-period sequence ``M`` is returned unchanged."""
    if not seq.half_period:
        return M
    b = seq.basis_0.dLambda[2] if derivative else seq.basis_0.Lambda[2]
    perm = reflection_permutation(b.n, b.p)
    return 0.5 * (M + M[perm][:, perm])


def mirror_component(seq, X, derivative_axes):
    """The full-period version of a per-DoF quantity ``X`` on the raw ``(n_r, n_t, n_z)`` grid of
    one component, computed on a half-period sequence (for example a squared basis function
    integrated against an even weight). It is the mean of ``X`` and its mirror image under the
    angular index permutation. ``derivative_axes`` lists the axes that carry the derivative basis.
    On a full-period sequence ``X`` is returned unchanged."""
    if not seq.half_period:
        return X
    perms = []
    for axis in (1, 2):
        b = seq.basis_0.dLambda[axis] if axis in derivative_axes else seq.basis_0.Lambda[axis]
        perms.append(reflection_permutation(b.n, b.p))
    return 0.5 * (X + X[:, perms[0], :][:, :, perms[1]])


@functools.partial(jax.jit, static_argnames=("plan_out", "plan_in"))
def symmetrize_like(y, x, plan_out, plan_in):
    """The part of ``y`` with the parity of ``x``, ``symmetrize(y, plan_out, parity_of(x, plan_in))``,
    as one compiled call."""
    return symmetrize(y, plan_out, parity_of(x, plan_in))


def raw_reflection(plan):
    """The reflection of ``plan`` on the flat raw vector as NumPy arrays ``(perm, sign)``, with
    ``(R x)[i] = sign[i] x[perm[i]]``."""
    n_raw = sum(int(np.prod(shape)) for *_, shape in plan)
    perm, sign, off = np.empty(n_raw, dtype=np.int64), np.empty(n_raw), 0
    for perm_t, perm_z, sgn, shape in plan:
        n_c = int(np.prod(shape))
        idx = np.arange(n_c).reshape(shape)
        perm[off:off + n_c] = off + idx[:, list(perm_t), :][:, :, list(perm_z)].ravel()
        sign[off:off + n_c] = sgn
        off += n_c
    return perm, sign


def _free_reflection(seq, k, dirichlet, e):
    """The reflection on the extracted DoFs of ``e``, ``R_free = (E E^T)^-1 E R E^T``, as
    ``(perm, sign, core, ERE_core)``. Away from the polar core rows ``R_free`` is the signed
    permutation ``(perm, sign)``, which is the identity on the core rows. On the core rows the
    dense block ``ERE_core = (E R E^T)[core, core]`` is returned instead."""
    # a bulk row selects one raw DoF, so its image is the bulk row that owns the reflected DoF
    perm, sign = raw_reflection(seq.reflection_plan[k])
    rows, cols, vals = e.entries()
    n_free = e.forward_shape[0]
    core = np.asarray(seq.core[(k, dirichlet)])
    is_core = np.zeros(n_free, dtype=bool)
    is_core[core] = True
    bulk = ~is_core[rows]
    owner = np.full(e.forward_shape[1], -1)
    owner[cols[bulk]] = rows[bulk]
    target = owner[perm[cols[bulk]]]
    if np.bincount(rows[bulk], minlength=n_free)[~is_core].min() != 1 or (target < 0).any() or is_core[target].any():
        raise RuntimeError("the free-space reflection is not a signed permutation of the bulk rows")
    if (owner[perm[cols[~bulk]]] >= 0).any():
        raise RuntimeError("the free-space reflection couples core and bulk rows")
    value = np.zeros(n_free)
    value[rows[bulk]] = vals[bulk]
    perm_free, sign_free = np.arange(n_free), np.ones(n_free)
    perm_free[rows[bulk]] = target
    sign_free[rows[bulk]] = vals[bulk] * sign[cols[bulk]] * value[target]
    core_rows = e.entries(core)
    reflected = (core_rows[0], perm[core_rows[1]], core_rows[2] * sign[core_rows[1]])
    return perm_free, sign_free, core, row_products(reflected, core_rows, (core.size, core.size))


def _orbits(seq, k, dirichlet, parity):
    """The DoFs of parity ``s = parity`` on the extracted k-form space of a half-period sequence,
    as ``(col, weight, n_red, core_red)``. Extracted DoF ``i`` enters reduced DoF ``col[i]``
    (``-1`` for none) with the weight ``weight[i]``. Since ``R_free`` is a signed permutation, the
    reduced basis vectors are ``(e_i + s sign_i e_R(i)) / sqrt 2`` for a pair ``i, R(i)`` and
    ``e_i`` for a fixed point of sign ``s``. The reduced DoFs built from the polar core rows
    (``core_red``) come last."""
    s = int(parity)
    # rebuilt in float64 from the host data, so the working sequence and its float64 twin get the same basis
    basis = (seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)[k]
    e, _ = build_extraction(basis, get_xi(seq.ns[1], basis.Lambda[1].p), dirichlet, dtype=np.float64)
    perm_free, sign_free, core, ERE_core = _free_reflection(seq, k, dirichlet, e)
    if core.size:
        # a signed permutation on the core rows too: the surgery weights sit at the splines' centres (get_xi)
        tol = 1e3 * float(np.finfo(np.float64).eps)
        C = core_gram_inverse(e, core) @ ERE_core
        C = np.where(np.abs(C) < tol, 0.0, C)
        if np.any(np.count_nonzero(C, axis=1) != 1) or np.abs(np.abs(C[C != 0]) - 1.0).max() > tol:
            raise RuntimeError("the free-space reflection is not a permutation on the polar core rows: "
                               "the surgery weights are not at the splines' centres (get_xi)")
        ci, cj = np.nonzero(C)
        perm_free[core[ci]], sign_free[core[ci]] = core[cj], np.sign(C[ci, cj])

    n_free = perm_free.size
    is_core = np.zeros(n_free, dtype=bool)
    is_core[core] = True
    order = np.concatenate([np.flatnonzero(~is_core), core])        # the core rows last
    pos = np.empty(n_free, dtype=np.int64)
    pos[order] = np.arange(n_free)
    fixed = perm_free[order] == order
    # an orbit's DoF is numbered at its first row in that order. A fixed point gets one only at its own parity
    leads = order[np.where(fixed, sign_free[order] == s, pos[order] < pos[perm_free[order]])]
    col, weight = np.full(n_free, -1), np.zeros(n_free)
    col[leads] = np.arange(leads.size)
    pair = leads[perm_free[leads] != leads]
    weight[leads] = np.where(perm_free[leads] == leads, 1.0, np.sqrt(0.5))
    col[perm_free[pair]] = col[pair]
    weight[perm_free[pair]] = s * sign_free[pair] * np.sqrt(0.5)
    return col, weight, leads.size, np.unique(col[core][col[core] >= 0])


def _orbit_maps(X):
    """``(col, weight)`` of a reduction ``X`` (free <- reduced) of :func:`parity_extraction`."""
    rows, cols, vals = X.entries()
    col, weight = np.full(X.forward_shape[0], -1), np.zeros(X.forward_shape[0])
    col[rows], weight[rows] = cols, vals
    return col, weight


def parity_extraction(seq, k, dirichlet, parity):
    """The DoF space of one parity on the extracted k-form space, as ``(E_red, X, core_red)``.

    ``X`` maps reduced DoFs to extracted DoFs and has orthonormal columns, so ``X.T`` reduces an
    extracted vector of this parity. ``E_red = X^T E`` is the extraction of the reduced space and
    satisfies ``E_red R = parity E_red``. ``core_red`` are the reduced DoFs built from the polar
    core rows. Both operators are :class:`~mrx.extraction_operators.MatrixFreeExtraction` in the
    sequence's dtype. :meth:`~mrx.derham_sequence.DeRhamSequence.parity_view` builds its spaces
    with this function."""
    col, weight, n_red, core = _orbits(seq, k, dirichlet, parity)
    e = seq.extraction[(k, dirichlet)]
    rows, cols, vals = e.entries()
    keep = col[rows] >= 0
    E_red = summed_coo(col[rows[keep]], cols[keep], weight[rows[keep]] * vals[keep],
                       (n_red, e.forward_shape[1]), dtype=e.dtype)
    free = np.flatnonzero(col >= 0)
    X = summed_coo(free, col[free], weight[free], (col.size, n_red), dtype=e.dtype)
    return E_red, X, core


def reduced_dirichlet_dofs(X_free, X_dirichlet, index):
    """:func:`~mrx.extraction_operators.dirichlet_dofs` on the parity-reduced spaces. ``X_free`` and
    ``X_dirichlet`` are the reductions of :func:`parity_extraction` and ``index`` the Dirichlet DoFs of
    the full space among its free DoFs."""
    col_f, w_f = _orbit_maps(X_free)
    col_d, w_d = _orbit_maps(X_dirichlet)
    rows = np.flatnonzero(col_d >= 0)
    reduced = np.full(X_dirichlet.forward_shape[1], -1)
    reduced[col_d[rows]] = col_f[index[rows]]
    if (reduced < 0).any() or not np.array_equal(w_d[rows], w_f[index[rows]]) \
            or not np.array_equal(col_f[index[rows]], reduced[col_d[rows]]):
        raise RuntimeError("the reduced Dirichlet space is not the reduced free space without its wall functions")
    return reduced


def reduce_operator(X_out, S, X_in, dtype):
    """``X_out^T S X_in``: an operator ``S`` between extracted spaces (the grad or curl stencil)
    restricted to the parity-reduced spaces of :func:`parity_extraction`. The restriction is exact
    because the exterior derivative commutes with the reflection."""
    col_out, w_out = _orbit_maps(X_out)
    col_in, w_in = _orbit_maps(X_in)
    rows, cols, vals = S.entries()
    keep = (col_out[rows] >= 0) & (col_in[cols] >= 0)
    return summed_coo(col_out[rows[keep]], col_in[cols[keep]], w_out[rows[keep]] * vals[keep] * w_in[cols[keep]],
                      (X_out.forward_shape[1], X_in.forward_shape[1]), dtype=dtype)


class FreeProjector(eqx.Module):
    """The parity projector on the extracted k-form space of a half-period sequence, used to wrap
    a preconditioner.

    With ``R_free`` the reflection on the extracted DoFs and ``s`` a parity,
    ``Pi = (I + s R_free) / 2`` projects a DoF vector onto parity ``s`` and ``Pi^T`` does the same
    for a dual vector (a right-hand side or a residual). Called with a preconditioner apply ``P``
    it gives ``Pi P Pi^T``, with ``s`` read off the input. This is symmetric, positive on the
    vectors of that parity and ignores the round-off of the other parity in a residual, so the
    residual norm of a CG preconditioned with it measures only what the half-period operators
    can reduce. The projections run in the residual precision :data:`mrx.precision.RESIDUAL_DTYPE`
    and return the input's dtype. In the mixed precision configuration that is float64, so the
    float64 residuals and the float64 probes of the preconditioners' axis blocks
    (:mod:`mrx.metric_lumping`) are split into their parities without float32 round-off.

    Obtain it from ``seq.free_projector(k)``. The sequence builds one per space."""

    perm: jnp.ndarray
    inv_perm: jnp.ndarray
    sign: jnp.ndarray
    core: jnp.ndarray
    block: jnp.ndarray
    blockT: jnp.ndarray
    has_core: bool = eqx.field(static=True)

    def __init__(self, seq, k, dirichlet):
        e = seq.extraction[(k, dirichlet)]
        perm_free, sign_free, core, ERE_core = _free_reflection(seq, k, dirichlet, e)
        if core.size:
            core_block = core_gram_inverse(e, core) @ ERE_core
        else:
            core_block = np.zeros((0, 0))
        self.perm = jnp.asarray(perm_free)
        self.inv_perm = jnp.asarray(np.argsort(perm_free))
        self.sign = jnp.asarray(sign_free, dtype=RESIDUAL_DTYPE)
        self.core = jnp.asarray(core)
        self.block = jnp.asarray(core_block, dtype=RESIDUAL_DTYPE)
        self.blockT = jnp.asarray(core_block.T, dtype=RESIDUAL_DTYPE)
        self.has_core = bool(core.size)

    def reflect_free(self, y):
        """``R_free y`` for an extracted DoF vector ``y``."""
        r = self.sign * y[self.perm]
        return r.at[self.core].set(self.block @ y[self.core]) if self.has_core else r

    def reflect_free_T(self, r):
        """``R_free^T r`` for a dual vector ``r``."""
        y = (self.sign * r)[self.inv_perm]
        return y.at[self.core].set(self.blockT @ r[self.core]) if self.has_core else y

    def post(self, y, s):
        """The projection ``(I + s R_free) y / 2`` of the DoF vector ``y``, computed in the
        residual precision and returned in ``y``'s dtype."""
        y_res = jnp.asarray(y, RESIDUAL_DTYPE)
        return (0.5 * (y_res + s * self.reflect_free(y_res))).astype(jnp.asarray(y).dtype)

    def dual(self, r, s):
        """The projection ``(I + s R_free^T) r / 2`` of the dual vector ``r``, computed in the
        residual precision and returned in ``r``'s dtype."""
        r_res = jnp.asarray(r, RESIDUAL_DTYPE)
        return (0.5 * (r_res + s * self.reflect_free_T(r_res))).astype(jnp.asarray(r).dtype)

    def parity(self, r):
        """The parity ``+1.0`` or ``-1.0`` of ``r``: the sign of ``r . R_free r``, which is
        ``+-|r|^2`` for a vector of definite parity. This works for dual vectors too, since the
        rows away from the polar core decide the sign."""
        r_res = jnp.asarray(r, RESIDUAL_DTYPE)
        return jnp.where(jnp.vdot(r_res, self.reflect_free(r_res)) >= 0.0, 1.0, -1.0)

    def projectors(self, b):
        """The functions ``(project_primal, project_dual)`` at the parity of the right-hand side
        ``b``, for a solver to combine with its deflation projectors."""
        return self.with_sign(self.parity(b))

    def with_sign(self, s):
        """The functions ``(project_primal, project_dual)`` for the parity ``s``."""
        return (lambda y: self.post(y, s)), (lambda r: self.dual(r, s))

    def __call__(self, apply, x):
        """``Pi P Pi^T x`` for the preconditioner apply ``P = apply``, at the parity of ``x``."""
        s = self.parity(x)
        return self.post(apply(self.dual(x, s)), s)
