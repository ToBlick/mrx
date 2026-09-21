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

import equinox as eqx
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


@functools.partial(jax.jit, static_argnames=("plan_out", "plan_in"))
def symmetrize_like(y, x, plan_out, plan_in):
    """``symmetrize(y, plan_out, parity_of(x, plan_in))`` in one compiled
    call: what a half-period mass or projection apply does to its output
    ``y`` given its input ``x``."""
    return symmetrize(y, plan_out, parity_of(x, plan_in))


def raw_reflection(plan):
    """The reflection of a raw k-form DoF vector as a signed permutation,
    ``(R x)[i] = sign[i] x[perm[i]]``, from the plan of :func:`reflection_plan`."""
    n_raw = sum(int(np.prod(shape)) for *_, shape in plan)
    perm, sign, off = np.empty(n_raw, dtype=np.int64), np.empty(n_raw), 0
    for perm_t, perm_z, sgn, shape in plan:
        n_c = int(np.prod(shape))
        idx = np.arange(n_c).reshape(shape)
        perm[off:off + n_c] = off + idx[:, list(perm_t), :][:, :, list(perm_z)].ravel()
        sign[off:off + n_c] = sgn
        off += n_c
    return perm, sign


def parity_basis(seq, k, dirichlet, parity):
    """The basis ``X`` (``n_free x n_red``, scipy CSR, orthonormal columns) of
    the k-form space of one PARITY on a half-period sequence, and ``E`` (scipy
    CSR) the polar and boundary extraction it is built on. The extraction of the
    reduced space is ``E_red = X^T E`` (``n_red x n_raw``), ``E`` the polar and
    boundary extraction (raw -> free) and ``X`` (``n_free x n_red``, orthonormal
    columns) a basis of the ``parity``-eigenspace of the free-space reflection
    ``R_free = (E E^T)^-1 E R E^T``. A field of parity ``s`` on the free space
    is ``R_free v = s v``, equivalently ``R E^T v = s E^T v`` on the raw grid,
    so every reduced DoF is a raw field of that parity and the half-period
    kernels (which return ``2 x`` the half-period moments) are exact through
    ``E_red`` with no projection: ``E_red (2 m_half) = E_red m_full``.

    The basis: ``R_free`` is a signed permutation on the bulk rows (one
    reduced DoF per pair ``{i, R(i)}``, ``(e_i + s sign_i e_R(i)) / sqrt 2``;
    a fixed point on the fold planes ``zeta = 0, 1/2`` is kept when its sign
    is ``s`` and dropped otherwise) and a small dense block on the polar core
    rows (an orthonormal basis of its ``s``-eigenspace). Built once on the
    host as COO triplets, applied like ``E`` (one gather + ``segment_sum``).
    Asserted at build time: ``E_red R = s E_red`` and ``X^T X = I``. Like
    ``E``, ``E_red`` is not row-orthonormal on the polar core (``E_red E_red^T
    = X^T (E E^T) X``, the identity plus the reduced core block); the two
    parities partition the free space, ``n_+ + n_- = n_free``."""
    from scipy import sparse  # noqa: PLC0415
    from mrx.extraction_operators import PolarExtractionOperator, get_xi  # noqa: PLC0415

    s = int(parity)
    if s not in (1, -1):
        raise ValueError("parity is +1 or -1")
    # The basis is built from the extraction in float64, rebuilt from the exact host
    # data (as the residual view rebuilds its own): the working sequence and its
    # float64 twin then get the SAME X -- the core eigenvectors are a basis choice,
    # and a refinement loop that alternates between the two views must agree on it.
    basis = (seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)[k]
    e = PolarExtractionOperator(basis, get_xi(seq.ns[1]), dirichlet).build_extraction(dtype=np.float64)
    TOL = 1e3 * float(np.finfo(np.float64).eps)
    n_free, n_raw = (int(v) for v in e.forward_shape)
    E = sparse.csr_matrix((np.asarray(e.vals, dtype=np.float64), (np.asarray(e.rows), np.asarray(e.cols))),
                          shape=(n_free, n_raw))
    perm, sign = raw_reflection(seq.reflection_plan[k])
    R = sparse.csr_matrix((sign, (np.arange(n_raw), perm)), shape=(n_raw, n_raw))
    gram = (E @ E.T).tocsr()
    core = seq.core_rows(k, dirichlet)
    is_core = np.zeros(n_free, dtype=bool)
    is_core[core] = True
    bulk = np.flatnonzero(~is_core)
    ERE = (E @ R @ E.T).tocsr()
    block = ERE[bulk].tocoo()
    if block.nnz != bulk.size or np.any(is_core[block.col]):
        raise RuntimeError("the free-space reflection is not a permutation on the bulk rows")
    if core.size and np.abs(ERE[np.ix_(core, bulk)]).max() > 0:
        raise RuntimeError("the free-space reflection couples core and bulk rows")
    perm_free, sign_free = np.arange(n_free), np.ones(n_free)
    perm_free[bulk[block.row]], sign_free[bulk[block.row]] = block.col, block.data

    # the bulk orbits
    rows, cols, vals = [], [], []          # of X (n_free x n_red)
    seen = np.zeros(n_free, dtype=bool)
    j = 0
    for i in bulk:
        if seen[i]:
            continue
        r = perm_free[i]
        if r == i:
            seen[i] = True
            if sign_free[i] == s:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)
                j += 1
        else:
            seen[i] = seen[r] = True
            rows += [i, r]
            cols += [j, j]
            vals += [np.sqrt(0.5), s * sign_free[i] * np.sqrt(0.5)]
            j += 1
    # the polar core: the s-eigenspace of the core block, orthonormal
    if core.size:
        inverse = np.linalg.inv(gram[np.ix_(core, core)].toarray())
        C = inverse @ ERE[np.ix_(core, core)].toarray()
        # C is an involution up to round-off, so the dimension of the s-eigenspace is
        # the number of eigenvalues near s, and its basis the right singular vectors
        # of C - s I with the smallest singular values.
        candidates = []
        for M in (C, C.T):
            n_s = int(np.sum(np.abs(np.linalg.eigvals(M) - s) < 0.5))
            _, _, vt = np.linalg.svd(M - s * np.eye(core.size))
            candidates.append(vt[core.size - n_s:].T if n_s else np.zeros((core.size, 0)))
        for null in candidates:
            n_c = null.shape[1]                                     # one reduced DoF per column
            core_rows = np.tile(core, n_c)                          # column-major: column j fills core rows
            core_cols = np.repeat(np.arange(j, j + n_c), core.size)
            X_try = sparse.coo_matrix((np.concatenate([vals, null.ravel(order="F")]),
                                       (np.concatenate([rows, core_rows]), np.concatenate([cols, core_cols]))),
                                      shape=(n_free, j + n_c)).tocsr()
            E_red = (X_try.T @ E).tocsr()
            if abs(E_red @ R - s * E_red).max() < TOL:
                break
        else:
            raise RuntimeError(f"no parity basis of the polar core rows satisfies E_red R = {s} E_red")
    else:
        X_try = sparse.coo_matrix((vals, (rows, cols)), shape=(n_free, j)).tocsr()
        E_red = (X_try.T @ E).tocsr()
    n_red = E_red.shape[0]
    if abs(E_red @ R - s * E_red).max() > TOL:
        raise RuntimeError("E_red R != s E_red")
    if abs(X_try.T @ X_try - sparse.identity(n_red)).max() > TOL:
        raise RuntimeError("X^T X != I")
    return X_try, E, np.arange(j, n_red)


def parity_extraction(seq, k, dirichlet, parity):
    """``(E_red, X, core)`` of :func:`parity_basis` as :class:`MatrixFreeExtraction`
    operators: ``E_red`` (``n_red x n_raw``, raw -> reduced), the expansion ``X``
    (``n_free x n_red``, reduced -> free; ``X.T`` reduces a free vector of that
    parity), and the reduced DoFs built from the polar core (the last block of
    ``X``'s columns: the dense core of the reduced space, known from the
    construction)."""
    from mrx.extraction_operators import MatrixFreeExtraction  # noqa: PLC0415
    from scipy import sparse  # noqa: PLC0415
    X, _, core = parity_basis(seq, k, dirichlet, parity)
    e = seq.E(k, dirichlet)                               # the sequence's own extraction, in its dtype
    E = sparse.csr_matrix((np.asarray(e.vals, dtype=np.float64), (np.asarray(e.rows), np.asarray(e.cols))),
                          shape=e.forward_shape)
    E_red = (X.T @ E).tocoo()
    X = X.tocoo()
    return (MatrixFreeExtraction.from_coo(E_red.row, E_red.col, E_red.data, E_red.shape, dtype=e.dtype),
            MatrixFreeExtraction.from_coo(X.row, X.col, X.data, X.shape, dtype=e.dtype), core)


def reduce_operator(X_out, S, X_in, dtype):
    """``X_out^T S X_in`` for a free-space operator ``S`` (a
    :class:`MatrixFreeExtraction`, e.g. a polar grad or curl stencil) between the
    parity bases ``X_in`` and ``X_out`` of :func:`parity_basis`: the operator on
    the reduced spaces (``d`` commutes with the reflection, so the product is exact)."""
    from scipy import sparse  # noqa: PLC0415
    from mrx.extraction_operators import MatrixFreeExtraction  # noqa: PLC0415
    Sm = sparse.csr_matrix((np.asarray(S.vals, dtype=np.float64), (np.asarray(S.rows), np.asarray(S.cols))),
                           shape=S.forward_shape)
    red = (X_out.T @ Sm @ X_in).tocoo()
    return MatrixFreeExtraction.from_coo(red.row, red.col, red.data, red.shape, dtype=dtype)


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
    core = seq.core_rows(k, dirichlet)
    inverse = np.linalg.inv(gram[np.ix_(core, core)].toarray()) if core.size else np.zeros((0, 0))
    cache[key] = (core, inverse)
    return cache[key]


class FreeProjector(eqx.Module):
    """``Pi P Pi^T`` for a preconditioner ``P`` on the EXTRACTED k-form space
    of a half-period sequence: ``pre`` projects the input (a dual vector)
    onto the parity it reads off it, ``post`` the output (a primal vector)
    onto the same parity. A pytree of its arrays (a signed permutation of
    the bulk rows, the dense polar core block), so that inside a jitted
    function of the sequence they are traced inputs, not captured
    constants (:mod:`mrx.pytree`); eagerly it is a handful of gathers. The
    primal projector is
    ``(I + s R_free) / 2`` with ``R_free = (E E^T)^-1 E R E^T``, the dual one
    its transpose ``E R E^T (E E^T)^-1``, ``(E E^T)^-1`` the identity but for
    the dense polar block of :func:`_extraction_gram_core`; the extraction is
    used in float64 so a float64 probe of the atoms stays exact.

    The metric-lumping atoms are only approximately reflection-equivariant
    (the polar rows), and a residual assembled by cancellation carries an
    impure round-off part. ``Pi P Pi^T`` is symmetric, positive on the pure
    subspace, returns pure vectors, and is blind to that part -- so a CG
    measuring in its norm stops where it should instead of chasing what the
    half-period applies cannot reduce."""

    def __init__(self, seq, k, dirichlet):
        from scipy import sparse  # noqa: PLC0415

        e = seq.E(k, dirichlet)
        plan = seq.reflection_plan[k]
        rows, cols = np.asarray(e.rows), np.asarray(e.cols)
        vals = np.asarray(e.vals, dtype=np.float64)
        n_free, n_raw = (int(v) for v in e.forward_shape)
        # The raw reflection as a signed permutation: (R x)[i] = sign[i] x[perm[i]].
        perm, sign, off = np.empty(n_raw, dtype=np.int64), np.empty(n_raw), 0
        for perm_t, perm_z, sgn, shape in plan:
            n_c = int(np.prod(shape))
            idx = np.arange(n_c).reshape(shape)
            perm[off:off + n_c] = off + idx[:, list(perm_t), :][:, :, list(perm_z)].ravel()
            sign[off:off + n_c] = sgn
            off += n_c
        core, inverse = _extraction_gram_core(seq, k, dirichlet)
        # R_free = (E E^T)^-1 E R E^T. The extraction is a selection on every
        # row but the polar core rows, and the reflection maps ring DoFs to
        # ring DoFs, so R_free is a signed permutation of the bulk rows plus
        # a dense block on the core rows: one gather and a tiny matvec.
        E = sparse.csr_matrix((vals, (rows, cols)), shape=(n_free, n_raw))
        R = sparse.csr_matrix((sign, (np.arange(n_raw), perm)), shape=(n_raw, n_raw))
        ERE = (E @ R @ E.T).tocsr()
        is_core = np.zeros(n_free, dtype=bool)
        is_core[core] = True
        bulk = np.flatnonzero(~is_core)
        block = ERE[bulk].tocoo()
        if block.nnz != bulk.size or np.any(is_core[block.col]):
            raise RuntimeError("the free-space reflection is not a permutation on the bulk rows")
        perm_free, sign_free = np.arange(n_free), np.ones(n_free)
        perm_free[bulk[block.row]], sign_free[bulk[block.row]] = block.col, block.data
        core_block = inverse @ ERE[np.ix_(core, core)].toarray() if core.size else np.zeros((0, 0))
        if core.size and np.abs(ERE[np.ix_(core, bulk)]).max() > 0:
            raise RuntimeError("the free-space reflection couples core and bulk rows")
        self.perm = jnp.asarray(perm_free)
        self.inv_perm = jnp.asarray(np.argsort(perm_free))
        self.sign = jnp.asarray(sign_free, dtype=jnp.float64)
        self.core = jnp.asarray(core)
        self.block = jnp.asarray(core_block, dtype=jnp.float64)
        self.blockT = jnp.asarray(core_block.T, dtype=jnp.float64)
        self.has_core = bool(core.size)

    perm: jnp.ndarray
    inv_perm: jnp.ndarray
    sign: jnp.ndarray
    core: jnp.ndarray
    block: jnp.ndarray
    blockT: jnp.ndarray
    has_core: bool = eqx.field(static=True)

    def reflect_free(self, y):
        """``R_free y``: the signed permutation of the bulk rows, the dense
        block on the polar core rows."""
        r = self.sign * y[self.perm]
        return r.at[self.core].set(self.block @ y[self.core]) if self.has_core else r

    def reflect_free_T(self, r):
        y = (self.sign * r)[self.inv_perm]
        return y.at[self.core].set(self.blockT @ r[self.core]) if self.has_core else y

    def post(self, y, s):
        """The primal projector ``(I + s R_free) / 2`` on ``y``, in float64,
        returned in ``y``'s dtype."""
        y64 = jnp.asarray(y, jnp.float64)
        return (0.5 * (y64 + s * self.reflect_free(y64))).astype(jnp.asarray(y).dtype)

    def dual(self, r, s):
        """The dual projector ``(I + s R_free^T) / 2`` on ``r``."""
        r64 = jnp.asarray(r, jnp.float64)
        return (0.5 * (r64 + s * self.reflect_free_T(r64))).astype(jnp.asarray(r).dtype)

    def parity(self, r):
        """The sign of ``r . R r``, ``+-|r|^2`` for a vector of definite
        parity (a dual one too: the bulk rows decide)."""
        r64 = jnp.asarray(r, jnp.float64)
        return jnp.where(jnp.vdot(r64, self.reflect_free(r64)) >= 0.0, 1.0, -1.0)

    def pre(self, x):
        """``(dual projection of x onto its own parity, that parity)``."""
        s = self.parity(x)
        return self.dual(x, s), s

    def projectors(self, b):
        """``(project_primal, project_dual)`` of the parity of the dual
        vector ``b`` (a solve's right-hand side): what a solver composes with
        its deflation projectors so that its iterates stay of that parity
        and its residuals lose the round-off of the other one -- the part
        the half-period applies cannot reduce, which otherwise dominates a
        converged residual and runs the inner CG to its cap."""
        return self.with_sign(self.parity(b))

    def with_sign(self, s):
        """``(project_primal, project_dual)`` for the parity ``s``."""
        return (lambda y: self.post(y, s)), (lambda r: self.dual(r, s))

    def __call__(self, apply, x):
        """``Pi P Pi^T x`` for the raw preconditioner apply ``apply``."""
        xp, s = self.pre(x)
        return self.post(apply(xp), s)


def free_projector(seq, k, dirichlet):
    """The :class:`FreeProjector` of ``(k, dirichlet)``, or ``None`` on a
    full-period sequence."""
    return FreeProjector(seq, k, dirichlet) if seq.half_period else None
