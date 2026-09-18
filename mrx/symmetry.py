r"""Stellarator symmetry on the DoF grids: the reflection and the parity projector.

A stellarator-symmetric map satisfies ``F(r, -theta, -zeta) = S F(r, theta,
zeta)`` with ``S = diag(1, -1, -1)``, the rotation by ``pi`` about the ``X``
axis (:mod:`mrx.mappings`). Every field of the relaxation then has a definite
PARITY under that rotation: the magnetic field, the vector potential, the
current and the auxiliary field are ODD (``B(Sx) = -S B(x)``: the standard
``(B_R, B_phi, B_Z)(R, -phi, -Z) = (-B_R, B_phi, B_Z)``), the velocity, the
force ``J x B``, the pressure and its gradient are EVEN. In logical
components the reflection ``phi: (r, theta, zeta) -> (r, -theta, -zeta)`` has
``det Dphi = +1``, so a covariant (1-form) and a contravariant density
(2-form) transform alike, ``(X_r, X_theta, X_zeta)(phi x) = parity * (X_r,
-X_theta, -X_zeta)(x)``, and a 0-form or 3-form as ``parity * X(phi x)``.

On a uniform periodic B-spline axis the reflection ``x -> -x`` is an index
permutation, ``B_j(-x) = B_{(p - 1 - j) mod n}(x)`` -- for the derivative
basis with ITS OWN ``(n, p)``, since it is the degree-``(p - 1)`` B-spline on
the knots trimmed by one at either end. So on the raw DoF grid of every
form the reflection is a permutation of the two angular axes times the
component signs, and the projector onto the fields of one parity is
``Pi = (I + parity * R) / 2`` with ``R`` that reflection.

That projector is what makes the HALF-PERIOD quadrature exact
(:class:`mrx.quadrature.QuadratureRule` with ``half_zeta``): for a field of
definite parity every quadrature-side integrand ``f v_j`` is even up to the
reflection of the basis function, so ``int_full f v_j = 2 int_half f v_j``
holds only after the two mirror images ``j`` and ``R j`` are combined --
``Pi`` applied to the doubled half-period moments IS the full-period moment
vector, for every ``j`` including those whose support straddles the fold.
The kernels stay parity-agnostic; the projector is applied by the
sequence's reductions (:meth:`mrx.derham_sequence.DeRhamSequence.symmetrize`),
which is why every reduction on a half-period sequence takes the field's
``parity``. Scalar integrals of even integrands (energy, helicity, the
force norm) are correct with the doubled weights as they stand.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


def reflection_permutation(n: int, p: int) -> np.ndarray:
    """Index permutation implementing ``x -> -x`` on a uniform periodic basis:
    ``B_j(-x) = B_{(p - 1 - j) mod n}(x)`` for the ``n`` uniform periodic
    B-splines of degree ``p``."""
    return (p - 1 - np.arange(n)) % n


def is_uniform_periodic(basis) -> bool:
    """Whether ``basis`` is a uniform periodic B-spline basis on ``[0, 1]``."""
    if getattr(basis, "type", None) != "periodic":
        return False
    unique = np.asarray(basis.T[basis.p:basis.p + basis.n + 1])
    return bool(np.allclose(unique, np.linspace(0.0, 1.0, basis.n + 1), atol=1e-6, rtol=0.0))



#: The sign each component of a vectorial form picks up under the
#: reflection ``(theta, zeta) -> (-theta, -zeta)``.
COMPONENT_SIGNS = (1.0, -1.0, -1.0)


def _component_axis_bases(form, k, c):
    """The three 1-D bases of component ``c`` of the k-form ``form``."""
    if k in (0, 3):
        return [form.Λ[a] for a in range(3)] if k == 0 else [form.dΛ[a] for a in range(3)]
    bases = [form.Λ[a] for a in range(3)] if k == 1 else [form.dΛ[a] for a in range(3)]
    bases[c] = form.dΛ[c] if k == 1 else form.Λ[c]
    return bases


def reflection_plan(seq, k):
    """The static reflection of the raw k-form DoF grid: per component the
    ``(theta permutation, zeta permutation, sign)`` and the component
    shape, in the layout of the raw applies (components concatenated).
    Requires uniform periodic angular bases (:func:`mrx.mappings.angular_reflection_allowed`)."""
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
    """``R x``: the raw k-form DoF vector reflected by the plan of
    :func:`reflection_plan`, the component signs included (``signed``) or
    the bare permutation of the two angular axes."""
    out, off = [], 0
    for perm_t, perm_z, sign, shape in plan:
        n_c = int(np.prod(shape))
        X = x[off:off + n_c].reshape(shape)
        RX = X[:, jnp.asarray(perm_t), :][:, :, jnp.asarray(perm_z)]
        out.append(((sign if signed else 1.0) * RX).ravel())
        off += n_c
    return jnp.concatenate(out)


def symmetrize(x, plan, parity, signed=True):
    """``(x + parity * R x) / 2`` on a raw k-form DoF vector: the projector
    onto the fields of ``parity`` ``+1`` (even) or ``-1`` (odd), a Python
    int or a traced scalar."""
    return 0.5 * (x + parity * reflect(x, plan, signed))


def parity_of(x, plan):
    """The parity of a raw DoF vector of definite parity, as a traced
    ``+-1.0``: the sign of ``x . R x``, which is ``+-|x|^2``. What the mass
    and projection applies of a half-period sequence read off their input
    (they commute with ``R``, so the output has the same parity)."""
    return jnp.where(jnp.vdot(x, reflect(x, plan)) >= 0.0, 1.0, -1.0).astype(x.dtype)


def mirror_zeta_1d(seq, M, derivative):
    """A 1-D matrix assembled on the zeta axis of a half-period sequence
    (``2 * int_half B_i B_j w``, ``w`` even), made the full-period one:
    ``(M + P M P^T) / 2`` with ``P`` the reflection of the axis basis, the
    primal one or its ``derivative`` basis. A full-period sequence's ``M``
    is returned as is."""
    if not seq.half_period:
        return M
    b = seq.basis_0.dΛ[2] if derivative else seq.basis_0.Λ[2]
    perm = reflection_permutation(b.n, b.p)
    return 0.5 * (M + M[perm][:, perm])


def mirror_component(seq, X, derivative_axes):
    """A per-DoF reduction over the half period on one component's raw
    ``(n_r, n_t, n_z)`` grid (a squared basis against an even weight), made
    the full-period one by averaging with its mirror image: the bare
    permutation of the two angular axes, each by the basis of that axis
    (``derivative_axes`` names the axes on the derivative basis)."""
    if not seq.half_period:
        return X
    perms = []
    for axis in (1, 2):
        b = seq.basis_0.dΛ[axis] if axis in derivative_axes else seq.basis_0.Λ[axis]
        perms.append(reflection_permutation(b.n, b.p))
    return 0.5 * (X + X[:, perms[0], :][:, :, perms[1]])


def free_reflection(seq, k, dirichlet):
    """The reflection on the EXTRACTED k-form space as a sparse matrix,
    ``R_free = (E E^T)^-1 E R E^T``: ``E^T`` lifts to the raw grid, ``R`` is
    the signed permutation of :func:`reflection_plan`, and the conforming
    restriction ``(E E^T)^-1 E`` brings the (still conforming: the polar
    constraint set is reflection invariant) result back. ``E E^T`` is the
    identity but for the small dense polar blocks, so this is a sparse
    matrix, built once per ``(k, dirichlet)`` (NumPy/SciPy, cached on the
    sequence). What :meth:`DeRhamSequence.project_parity` applies."""
    from scipy import sparse  # noqa: PLC0415
    from scipy.sparse import csgraph  # noqa: PLC0415

    cache = seq.__dict__.setdefault("_free_reflection", {})
    key = (int(k), bool(dirichlet))
    if key in cache:
        return cache[key]
    e = seq.E(k, dirichlet)
    n_free, n_raw = (int(v) for v in e.forward_shape)
    E = sparse.csr_matrix((np.asarray(e.vals, dtype=np.float64),
                           (np.asarray(e.rows), np.asarray(e.cols))), shape=(n_free, n_raw))
    perm, sign, off = np.empty(n_raw, dtype=np.int64), np.empty(n_raw), 0
    for perm_t, perm_z, s, shape in seq.reflection_plan[k]:
        n_c = int(np.prod(shape))
        idx = np.arange(n_c).reshape(shape)
        perm[off:off + n_c] = off + idx[:, list(perm_t), :][:, :, list(perm_z)].ravel()
        sign[off:off + n_c] = s
        off += n_c
    R = sparse.csr_matrix((sign, (np.arange(n_raw), perm)), shape=(n_raw, n_raw))
    gram = (E @ E.T).tocsr()
    _, labels = csgraph.connected_components(gram, directed=False)
    counts = np.bincount(labels)
    diag = gram.diagonal()
    inv = sparse.lil_matrix((n_free, n_free))
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(labels.max() + 2))
    for lab in range(labels.max() + 1):
        idx = order[bounds[lab]:bounds[lab + 1]]
        if counts[lab] == 1:
            inv[idx[0], idx[0]] = 1.0 / diag[idx[0]]
        else:
            inv[np.ix_(idx, idx)] = np.linalg.inv(gram[np.ix_(idx, idx)].toarray())
    cache[key] = (inv.tocsr() @ E @ R @ E.T).tocsr()
    return cache[key]


@functools.partial(jax.jit, static_argnames=("plan_out", "plan_in"))
def symmetrize_like(y, x, plan_out, plan_in):
    """``symmetrize(y, plan_out, parity_of(x, plan_in))`` in one compiled
    call: what a half-period mass or projection apply does to its output
    ``y`` given its input ``x``."""
    return symmetrize(y, plan_out, parity_of(x, plan_in))


def _extraction_gram_core(seq, k, dirichlet):
    """``(core, inverse)``: the rows of the extracted k-form space where
    ``E E^T`` is not the identity (the polar rows, where the extraction fuses
    raw DoFs) and the dense inverse of ``E E^T`` on them; ``E E^T`` is the
    identity plus those blocks, which do not couple to the rest. Host-side,
    cached on the sequence."""
    from scipy import sparse  # noqa: PLC0415

    cache = seq.__dict__.setdefault("_gram_core", {})
    key = (int(k), bool(dirichlet))
    if key in cache:
        return cache[key]
    e = seq.E(k, dirichlet)
    n_free, n_raw = (int(v) for v in e.forward_shape)
    E = sparse.csr_matrix((np.asarray(e.vals, dtype=np.float64),
                           (np.asarray(e.rows), np.asarray(e.cols))), shape=(n_free, n_raw))
    gram = (E @ E.T).tocsr()
    counts = np.bincount(np.asarray(e.rows), minlength=n_free)
    core = np.flatnonzero(counts > 1)
    inverse = np.linalg.inv(gram[np.ix_(core, core)].toarray()) if core.size else np.zeros((0, 0))
    cache[key] = (core, inverse)
    return cache[key]


def free_projector(seq, k, dirichlet):
    """``(y, x) -> Pi y`` on the EXTRACTED k-form space of a half-period
    sequence, ``Pi`` the projector onto the parity of ``x``: jitted and
    device-only, for the preconditioners. ``R_free y = (E E^T)^-1 E R E^T y``
    with the small dense inverse of :func:`_extraction_gram_core`; the
    extraction is used in float64 so a float64 probe of the atoms stays
    exact, the result is in ``y``'s dtype. ``None`` on a full-period
    sequence.

    The metric-lumping atoms are only approximately reflection-equivariant
    (the polar rows), and a preconditioned CG whose preconditioner leaks
    the other parity feeds the half-period applies vectors they are not
    exact on: ``Pi P Pi`` is what they apply, SPD on the pure subspace."""
    if not seq.half_period:
        return None
    e = seq.E(k, dirichlet)
    e64 = jax.tree_util.tree_map(
        lambda a: a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a, e)
    plan = seq.reflection_plan[k]
    core, inverse = _extraction_gram_core(seq, k, dirichlet)
    core_j, inv_j = jnp.asarray(core), jnp.asarray(inverse, dtype=jnp.float64)

    @jax.jit
    def project(y, x):
        y64 = jnp.asarray(y, jnp.float64)
        s = parity_of(e64.T @ jnp.asarray(x, jnp.float64), plan)
        c = e64 @ reflect(e64.T @ y64, plan)
        if core.size:
            c = c.at[core_j].set(inv_j @ c[core_j])
        return (0.5 * (y64 + s * c)).astype(jnp.asarray(y).dtype)
    return project
