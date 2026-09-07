"""Matrix-free operator bundle (:class:`SequenceOperators`), assembly of its fields, and the operator applies and solves."""
from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax.numpy as jnp

from mrx.extraction_operators import MatrixFreeExtraction
from mrx.mass import sumfact_apply
import numpy as np

from mrx.preconditioners import _assemble_weighted_1d_mass, _symmetrize
from mrx.precision import RESIDUAL_DTYPE, inner_tol
from mrx.solvers import deflation_projectors, refine, solve_saddle_point_minres, solve_singular_cg
import mrx
def _nullspace_vectors(operators, k: int, dirichlet: bool):
    """Return the stacked nullspace array for ``(k, dirichlet)``."""
    from mrx.nullspace import get_nullspace
    return get_nullspace(operators, k, dirichlet)


class SequenceOperators(eqx.Module):
    """Everything built FROM a geometry: preconditioners and harmonic forms.

    The sequence itself is static (bases, extraction, incidence); this bundle
    holds every factorisation of the installed metric -- the metric-lumped
    mass and Laplacian atoms, the preconditioners of every solve -- and the
    nullspace vectors. Nothing here is built on first use:
    ``DeRhamSequence.build_preconditioners`` builds it against the geometry
    installed at that moment, and a new geometry means calling it again.
    :func:`new_operators` is the empty bundle (zero nullspaces, no
    preconditioners).
    """

    # Metric-lumped mass and Laplacian atoms, keyed ``(k, dirichlet)``. Plain
    # Python objects (each holds a payload pytree and a jitted apply); the
    # bundle is closed over by the solvers, never passed through a JAX
    # transform, so that is fine. Built by
    # :func:`assemble_mass_metric_lumping_preconditioner` and
    # :func:`assemble_metric_lumping_laplacian_preconditioner`.
    mass_lumping: Optional[dict] = None
    laplacian_lumping: Optional[dict] = None
    # Harmonic forms of the k-form Laplacians, keyed ``(k, dirichlet)``: an
    # array ``(n_vectors, n_k)``, one nullspace vector per row. The shapes are
    # topological (Betti numbers); the values belong to the geometry. Zero
    # until computed, so deflation is a no-op.
    nullspaces: Optional[dict] = None


def new_operators(seq) -> SequenceOperators:
    """The empty bundle for ``seq``: zero nullspaces, no preconditioners."""
    from mrx.nullspace import init_nullspaces  # noqa: PLC0415
    return init_nullspaces(seq, SequenceOperators(mass_lumping={}, laplacian_lumping={},
                                                  nullspaces={}))


def _require_bundle(operators):
    if operators is None:
        raise ValueError(
            "no operator bundle: call seq.build_preconditioners() after set_map")
    return operators


def _assemble_weighted_1d_stiffness(
        primal_basis: jnp.ndarray,
        derivative_basis: jnp.ndarray,
        weights: jnp.ndarray,
        incidence: jnp.ndarray) -> jnp.ndarray:
    mass_d = _assemble_weighted_1d_mass(derivative_basis, weights)
    stiffness = incidence.T @ (mass_d @ incidence)
    return _symmetrize(stiffness)


# ---------------------------------------------------------------------------
# Fast-diagonalisation Hodge-Laplacian preconditioner.
#
# For a 0-form on the reference cube the discrete Hodge Laplacian
# ``L_0 = K_0`` is a Kronecker SUM
#
#     L_0 ≈  K_r ⊗ M_t ⊗ M_z + M_r ⊗ K_t ⊗ M_z + M_r ⊗ M_t ⊗ K_z ,
#
# with 1-D mass ``M_a = ∫ B^p_a (B^p_a)^T`` and 1-D stiffness
# ``K_a = ∫ (∂B^p_a)(∂B^p_a)^T = G_a^T M^d_a G_a`` (incidence relation).
# Reducing the per-axis generalised eigenproblem ``K_a v = λ M_a v`` to a
# standard one via Cholesky gives an ``M``-orthonormal eigenbasis and the
# inverse can be applied as three small dense matmuls per axis combined with
# a divide by ``Σ_i α_i λ_i`` on the 3-tensor.  ``α_i = ⟨J·g^{ii}⟩_quad``
# captures the leading metric anisotropy on the mapped domain.
# ---------------------------------------------------------------------------


def _dense_incidence_1d(n0: int, typ: str) -> jnp.ndarray:
    """Return the dense 1-D incidence matrix ``G_a`` for axis basis type.

    ``clamped``: ``(G c)_j = c_{j+1} - c_j`` on ``n0 - 1`` rows;
    ``periodic``: the same with the index wrapped, ``n0`` rows;
    ``constant``: the zero ``(n0, n0)`` matrix.
    """
    if typ == 'clamped':
        n_out = n0 - 1
        j = jnp.arange(n_out)
        return jnp.zeros((n_out, n0), dtype=mrx.DTYPE).at[j, j].set(-1.0).at[j, j + 1].set(1.0)
    if typ == 'periodic':
        j = jnp.arange(n0)
        return jnp.zeros((n0, n0), dtype=mrx.DTYPE).at[j, j].set(-1.0).at[j, (j + 1) % n0].set(1.0)
    if typ == 'constant':
        return jnp.zeros((n0, n0), dtype=mrx.DTYPE)
    raise ValueError(f"Unknown basis type {typ!r}")

def _fd_forward(V_r, V_t, V_z, x):
    """``y = V^T x`` on all three axes of a 3-tensor."""
    y = jnp.einsum('ji,jkl->ikl', V_r, x)
    y = jnp.einsum('ji,kjl->kil', V_t, y)
    return jnp.einsum('ji,klj->kli', V_z, y)


def _fd_backward(V_r, V_t, V_z, y):
    """``x = V y`` on all three axes of a 3-tensor."""
    y = jnp.einsum('ij,jkl->ikl', V_r, y)
    y = jnp.einsum('ij,kjl->kil', V_t, y)
    return jnp.einsum('ij,klj->kli', V_z, y)


def _fd_denominator(lam_r, lam_t, lam_z, alpha):
    return (alpha[0] * lam_r[:, None, None]
            + alpha[1] * lam_t[None, :, None]
            + alpha[2] * lam_z[None, None, :])


def _fd_apply_3d(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, x, eps: float = 0.0):
    """Apply ``(L + eps M)^{-1}`` via fast diagonalisation on a 3-tensor ``x``.

    ``eps`` is a Python float (the ``eps == 0`` branch is static); a traced
    shift goes through :func:`_fd_apply_3d_shifted`.
    """
    y = _fd_forward(V_r, V_t, V_z, x)
    denom = _fd_denominator(lam_r, lam_t, lam_z, alpha) + eps
    if eps == 0:
        # The pure-constant 0-form is in the null space; threshold relative
        # to the largest entry so we don't amplify it into a huge spurious
        # negative direction.
        denom_max = jnp.max(jnp.abs(denom))
        null_mask = jnp.abs(denom) < mrx.sqrt_eps(6.7e-3) * denom_max
        safe = jnp.where(null_mask, 1.0, denom)
        y = jnp.where(null_mask, 0.0, y / safe)
    else:
        y = y / denom
    return _fd_backward(V_r, V_t, V_z, y)


def _fd_apply_3d_shifted(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, x, shift):
    """Apply ``(sum_a alpha_a K_a-term + shift M)^{-1}`` on a 3-tensor ``x``.

    ``shift > 0`` may be traced: no branch, nothing singular. This is the
    shifted-stiffness atom's kernel (``shift = 1/eps``).
    """
    y = _fd_forward(V_r, V_t, V_z, x)
    y = y / (_fd_denominator(lam_r, lam_t, lam_z, alpha) + shift)
    return _fd_backward(V_r, V_t, V_z, y)


def mass_core_apply(seq, k: int):
    """The raw-DOF-space callable ``x -> M_k @ x`` of the installed geometry.

    Acts in the unextracted tensor-product DOF space, matrix-free (the
    sum-factorised kernel never materialises ``M_k``). Built by
    ``DeRhamSequence.set_geometry``.
    """
    if seq.geometry is None:
        raise ValueError("no geometry installed: call seq.set_map first")
    plan, weights = seq.mass_plan[k], seq.geometry.mass_weights[k]
    return lambda x: sumfact_apply(plan, weights, x)


# ---------------------------------------------------------------------------
# Topological incidence matrices (geometry-independent strong derivatives)
# ---------------------------------------------------------------------------
#
# On a FEEC B-spline de Rham complex the exterior derivative at the DoF level
# is a topological incidence matrix with entries in {-1, 0, +1}. The 1-D
# building block maps 0-form DoFs (nodes) to 1-form DoFs (edges) via
#
#     (G c)_j = c_{j+1} - c_j           (periodic: indices mod n)
#
# so the 3-D operators are Kronecker sums/products of these with identities.
# Because the incidence is geometry-independent, it does not need to be
# re-assembled when the spline map changes.

# ---------------------------------------------------------------------------
# Matrix-free topological incidence (G0/G1/G2 and transposes)
#
# The incidence is a {-1, 0, +1} difference stencil, so it never needs to be
# stored. In non-flattened (tensor) form the apply is just per-axis forward
# differences (grad/curl/div) or their adjoints, which makes the zero structure
# explicit. ``_MatrixFreeIncidence`` carries only static shape metadata and
# applies via reshape + difference.
# ---------------------------------------------------------------------------

def _diff_fwd(V, axis: int, typ: str):
    """Forward 1-D incidence (discrete derivative) along ``axis``.

    ``clamped``: ``(G c)_j = c_{j+1} - c_j`` (size shrinks by one);
    ``periodic``: ``c_{(j+1) mod n} - c_j`` (size preserved);
    ``constant``: derivative of a constant is zero (size preserved).
    """
    if typ == 'clamped':
        return jnp.diff(V, axis=axis)
    if typ == 'periodic':
        return jnp.roll(V, -1, axis=axis) - V
    if typ == 'constant':
        return jnp.zeros_like(V)
    raise ValueError(f"Unknown basis type {typ!r}")


def _diff_adj(Y, axis: int, typ: str):
    """Adjoint of :func:`_diff_fwd` along ``axis`` (transpose incidence)."""
    if typ == 'clamped':
        pad_end = [(0, 0)] * Y.ndim
        pad_end[axis] = (0, 1)
        pad_start = [(0, 0)] * Y.ndim
        pad_start[axis] = (1, 0)
        return jnp.pad(-Y, pad_end) + jnp.pad(Y, pad_start)
    if typ == 'periodic':
        return jnp.roll(Y, 1, axis=axis) - Y
    if typ == 'constant':
        return jnp.zeros_like(Y)
    raise ValueError(f"Unknown basis type {typ!r}")


def _prod3(shape) -> int:
    return int(shape[0] * shape[1] * shape[2])


def _split3(x, shapes):
    """Split a flat vector into three 3-D component arrays of ``shapes``."""
    n0 = _prod3(shapes[0])
    n1 = _prod3(shapes[1])
    a = x[:n0].reshape(shapes[0])
    b = x[n0:n0 + n1].reshape(shapes[1])
    c = x[n0 + n1:].reshape(shapes[2])
    return a, b, c


def _apply_incidence_mf(op, x):
    """Apply a :class:`_MatrixFreeIncidence` operator to flat vector ``x``."""
    types = op.types
    tr, tt, tz = types
    s0, s1, s2, s3 = op.s0, op.s1, op.s2, op.s3
    s1_r, s1_t, s1_z = s1
    s2_r, s2_t, s2_z = s2

    if op.k == 0 and not op.transpose:
        # G0 grad: 0-form -> (d_r, d_t, d_z).
        V = x.reshape(s0)
        return jnp.concatenate([
            _diff_fwd(V, 0, tr).ravel(),
            _diff_fwd(V, 1, tt).ravel(),
            _diff_fwd(V, 2, tz).ravel(),
        ])
    if op.k == 0 and op.transpose:
        a, b, c = _split3(x, s1)
        out = (_diff_adj(a, 0, tr)
               + _diff_adj(b, 1, tt)
               + _diff_adj(c, 2, tz))
        return out.ravel()

    if op.k == 1 and not op.transpose:
        # G1 curl: (a, b, c) -> (P, Q, R).
        a, b, c = _split3(x, s1)
        P = -_diff_fwd(b, 2, tz) + _diff_fwd(c, 1, tt)
        Q = _diff_fwd(a, 2, tz) - _diff_fwd(c, 0, tr)
        R = -_diff_fwd(a, 1, tt) + _diff_fwd(b, 0, tr)
        return jnp.concatenate([P.ravel(), Q.ravel(), R.ravel()])
    if op.k == 1 and op.transpose:
        P, Q, R = _split3(x, s2)
        a = _diff_adj(Q, 2, tz) - _diff_adj(R, 1, tt)
        b = -_diff_adj(P, 2, tz) + _diff_adj(R, 0, tr)
        c = _diff_adj(P, 1, tt) - _diff_adj(Q, 0, tr)
        return jnp.concatenate([a.ravel(), b.ravel(), c.ravel()])

    if op.k == 2 and not op.transpose:
        # G2 div: (a, b, c) -> d_r a + d_t b + d_z c.
        a, b, c = _split3(x, s2)
        out = (_diff_fwd(a, 0, tr)
               + _diff_fwd(b, 1, tt)
               + _diff_fwd(c, 2, tz))
        return out.ravel()
    if op.k == 2 and op.transpose:
        Y = x.reshape(s3)
        return jnp.concatenate([
            _diff_adj(Y, 0, tr).ravel(),
            _diff_adj(Y, 1, tt).ravel(),
            _diff_adj(Y, 2, tz).ravel(),
        ])
    raise ValueError(f"Unsupported incidence apply (k={op.k}, transpose={op.transpose})")


class _MatrixFreeIncidence(eqx.Module):
    """Lazy {-1,0,+1} incidence operator applied as a difference stencil.

    Carries only static shape metadata (no stored matrix). Supports the matvec
    protocol (``@`` / ``__call__``) used throughout the solve path.
    """
    k: int = eqx.field(static=True)
    transpose: bool = eqx.field(static=True)
    types: tuple = eqx.field(static=True)
    s0: tuple = eqx.field(static=True)
    s1: tuple = eqx.field(static=True)
    s2: tuple = eqx.field(static=True)
    s3: tuple = eqx.field(static=True)
    shape: tuple = eqx.field(static=True)

    def __matmul__(self, x):
        return _apply_incidence_mf(self, x)

    def __call__(self, x):
        return _apply_incidence_mf(self, x)

    @property
    def T(self):
        return _MatrixFreeIncidence(
            k=self.k,
            transpose=not self.transpose,
            types=self.types,
            s0=self.s0, s1=self.s1, s2=self.s2, s3=self.s3,
            shape=(self.shape[1], self.shape[0]),
        )


def _incidence_shapes(seq):
    """Return the four DoF shape groups ``(s0, s1, s2, s3)`` for ``seq``."""
    s0 = tuple(int(v) for v in seq.basis_0.shape[0])
    s3 = tuple(int(v) for v in seq.basis_3.shape[0])
    s1 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_1.shape)
    s2 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_2.shape)
    return s0, s1, s2, s3


def build_matrixfree_incidence(seq, k: int):
    """Return ``(Gk, Gk_T)`` as matrix-free incidence operators."""
    types = tuple(seq.basis_0.types)
    s0, s1, s2, s3 = _incidence_shapes(seq)
    if k == 0:
        n_in = _prod3(s0)
        n_out = sum(_prod3(c) for c in s1)
    elif k == 1:
        n_in = sum(_prod3(c) for c in s1)
        n_out = sum(_prod3(c) for c in s2)
    elif k == 2:
        n_in = sum(_prod3(c) for c in s2)
        n_out = _prod3(s3)
    else:
        raise ValueError("k must be 0, 1 or 2")
    common = dict(k=k, types=types, s0=s0, s1=s1, s2=s2, s3=s3)
    g = _MatrixFreeIncidence(transpose=False, shape=(n_out, n_in), **common)
    g_T = _MatrixFreeIncidence(transpose=True, shape=(n_in, n_out), **common)
    return g, g_T


def _stencil_grid(*dims):
    """Flattened C-order index grids of ``np.arange(d)`` for each dim, so the
    flat position of ``(i, j, k)`` is ``ravel_multi_index((i, j, k), dims)``."""
    return [g.reshape(-1) for g in np.meshgrid(*(np.arange(d) for d in dims),
                                               indexing='ij')]


class _StencilTriplets:
    """COO triplet collector; ``emit`` drops zero weights and masked columns."""

    def __init__(self):
        self.rows, self.cols, self.data = [], [], []

    def emit(self, rows, cols, data):
        rows, cols = np.broadcast_arrays(rows, cols)
        data = np.broadcast_to(np.asarray(data, dtype=np.float64), rows.shape)
        keep = data != 0.0
        self.rows.append(rows[keep])
        self.cols.append(cols[keep])
        self.data.append(data[keep])

    def operator(self, shape, dtype=None):
        """Return the collected triplets as a :class:`MatrixFreeExtraction`
        with values of ``dtype`` (the working dtype by default).

        Duplicates are summed on the host first so the device arrays hold one
        entry per nonzero.
        """
        import scipy.sparse as _sps
        coo = _sps.coo_matrix(
            (np.concatenate(self.data),
             (np.concatenate(self.rows).astype(np.int32),
              np.concatenate(self.cols).astype(np.int32))),
            shape=shape).tocsr().tocoo()
        return MatrixFreeExtraction.from_coo(coo.row, coo.col, coo.data, shape, dtype=dtype)


def build_grad_stencil_g0(seq, xi, dirichlet_in: bool, dirichlet_out: bool, dtype=None):
    """Analytic, INVERSE-FREE polar discrete gradient ``G_0`` (V0 -> V1).

    Builds the true strong gradient on extracted DoFs as an indexed operator
    straight from the incidence pattern and the polar mapping coefficients
    ``xi`` (shape ``(3, 2, nt)``) -- coefficient differences and ``xi`` weights
    only, NO mass and NO matrix inverse. This is the closed form of
    ``Gram_1^{-1} (E_1 sp_0 E_0^T)``; the axis-fusion inverse cancels to clean
    ``+/-1`` / ``-xi[l,1,j]`` stencils (verified bit-exact vs that oracle).

    Layout (see ``extraction_operators.build_extraction`` k=0/k=1 branches):
    V0 extracted = apex ``(p,m) -> p*nz+m`` (p in 0..2) then bulk
    ``(i,j,k) -> 3 nz + ravel((i,j,k),(radial0,nt,nz))`` with full radial ``i+2``.
    V1 extracted = theta_surgery ``[0,2 nz)`` | zeta_surgery ``[2 nz, 2 nz+3 dz)``
    | r-slice (comp0) | theta_bulk (comp1) | zeta_bulk (comp2). The full-space
    grad is ``d_r f``, ``d_theta f`` (periodic), ``d_z f`` (periodic), with the
    near-axis full radial rows 0/1 expanded as ``f(0,j,k)=sum_p xi[p,0,j] apex``,
    ``f(1,j,k)=sum_p xi[p,1,j] apex``.

    Every block is emitted as whole index grids (no per-DoF Python loop): the
    row of bulk entry ``(i, j, k)`` is its ravelled position plus the block
    offset, and ``expand`` applies the apex/bulk column rule to index arrays.
    """
    xi = np.asarray(xi)
    nr, nt, nz = (int(v) for v in seq.basis_0.shape[0])
    dr = nr - 1            # clamped r derivative count
    dt, dz = nt, nz        # periodic theta, z -> derivative count == primal
    o0 = 1 if dirichlet_in else 0
    o1 = 1 if dirichlet_out else 0
    radial0 = nr - 2 - o0  # V0 bulk radial rings (full radial >= 2)
    radial1 = nr - 2 - o1  # V1 comp1/comp2 bulk radial rings

    base_bulk0 = 3 * nz
    out = _StencilTriplets()

    def expand(r, a, j, k, s):
        """``s`` times the full V0 DoF ``(a, j, k)`` on rows ``r`` (arrays)."""
        for ring in (0, 1):
            m = a == ring
            for p in range(3):
                out.emit(r[m], p * nz + k[m], s * xi[p, ring, j[m]])
        m = (a >= 2) & (a - 2 < radial0)
        out.emit(r[m], base_bulk0 + ((a[m] - 2) * nt + j[m]) * nz + k[m], s)

    # V1 extracted row offsets (must match _k1_row_slices with o == o1).
    r_theta_s = 0
    r_zeta_s = 2 * nz
    r_r = 2 * nz + 3 * dz
    r_theta_b = r_r + (dr - 1) * nt * nz
    r_zeta_b = r_theta_b + radial1 * dt * nz

    # theta_surgery: apex difference  apex(p_local+1, m) - apex(0, m)
    pl, m = _stencil_grid(2, nz)
    out.emit(r_theta_s + pl * nz + m, (pl + 1) * nz + m, 1.0)
    out.emit(r_theta_s + pl * nz + m, m, -1.0)

    # zeta_surgery: periodic z-difference of the apex DoFs
    p, m = _stencil_grid(3, dz)
    out.emit(r_zeta_s + p * dz + m, p * nz + (m + 1) % nz, 1.0)
    out.emit(r_zeta_s + p * dz + m, p * nz + m, -1.0)

    # r-slice (comp0, radial grad):  full(i+2,j,k) - full(i+1,j,k)
    i, j, k = _stencil_grid(dr - 1, nt, nz)
    r = r_r + np.arange(i.size)
    expand(r, i + 2, j, k, 1.0)
    expand(r, i + 1, j, k, -1.0)

    # theta_bulk (comp1, angular grad, periodic):  full(i+2,j+1) - full(i+2,j)
    i, j, k = _stencil_grid(radial1, dt, nz)
    r = r_theta_b + np.arange(i.size)
    expand(r, i + 2, (j + 1) % nt, k, 1.0)
    expand(r, i + 2, j, k, -1.0)

    # zeta_bulk (comp2, z grad, periodic):  full(i+2,k+1) - full(i+2,k)
    i, j, k = _stencil_grid(radial1, nt, dz)
    r = r_zeta_b + np.arange(i.size)
    expand(r, i + 2, j, (k + 1) % nz, 1.0)
    expand(r, i + 2, j, k, -1.0)

    n0 = int(seq.n(0, True) if dirichlet_in else seq.n(0))
    n1 = int(seq.n(1, True) if dirichlet_out else seq.n(1))
    return out.operator((n1, n0), dtype=dtype)


def build_curl_stencil_g1(seq, xi, dirichlet_in: bool, dirichlet_out: bool, dtype=None):
    """Analytic, INVERSE-FREE polar discrete curl ``G_1`` (V1 -> V2).

    The degree-1 analog of :func:`build_grad_stencil_g0`: the true strong curl on
    extracted DoFs as an indexed operator from the incidence pattern and the
    polar coefficients ``xi`` (shape ``(3, 2, nt)``) -- coefficient differences and
    ``xi`` weights only, NO mass and NO matrix inverse. The closed form of
    ``Gram_2^{-1} (E_2 sp_1 E_1^T)``; the V2 axis-fusion inverse cancels to clean
    ``+/-1`` / ``xi``-difference stencils (verified bit-exact vs that oracle).

    Full-space curl (a=s, b=chi, c=zeta -> V2 comps P,Q,R; see ``_apply_incidence_mf``):
    ``P=-d_z b + d_t c``, ``Q=d_z a - d_r c``, ``R=-d_t a + d_r b``. V1 input fusion
    is inverted by ``expand_v1`` (the V1 analog of grad's ``expand``); the only fused
    V2 *output* DoFs are the comp0 surgery rows, whose stencil is the axis form of
    ``P = -d_z(chi apex) + d_t(zeta apex)``.
    """
    xi = np.asarray(xi)
    nr, nt, nz = (int(v) for v in seq.basis_0.shape[0])
    dr, dt, dz = nr - 1, nt, nz
    o_in = 1 if dirichlet_in else 0
    o_out = 1 if dirichlet_out else 0
    radial_in = nr - 2 - o_in
    radial_out = nr - 2 - o_out

    # --- V1 extracted (input) columns + fusion-inverting expand ---
    base_r1 = 2 * nz + 3 * dz
    base_tb1 = base_r1 + (dr - 1) * nt * nz
    base_zb1 = base_tb1 + radial_in * dt * nz
    out = _StencilTriplets()

    def c_ths(pl, m):                                  # V1 theta_surgery col
        return pl * nz + m

    def c_zes(p, m):                                   # V1 zeta_surgery col
        return 2 * nz + p * dz + m

    def expand_v1(r, comp, a, j, k, s):
        """``s`` times the full V1 DoF ``(comp, a, j, k)`` on rows ``r``."""
        if comp == 0:                                  # s, full radial a in [0,dr)
            m = a == 0
            for pl in range(2):
                out.emit(r[m], c_ths(pl, k[m]),
                         s * (xi[pl + 1, 1, j[m]] - xi[pl + 1, 0, j[m]]))
            m = (a >= 1) & (a - 1 < dr - 1)
            out.emit(r[m], base_r1 + ((a[m] - 1) * nt + j[m]) * nz + k[m], s)
        elif comp == 1:                                # chi, full radial a in [0,nr)
            m = a == 1
            for pl in range(2):
                out.emit(r[m], c_ths(pl, k[m]),
                         s * (xi[pl + 1, 1, (j[m] + 1) % dt] - xi[pl + 1, 1, j[m]]))
            m = (a >= 2) & (a - 2 < radial_in)
            out.emit(r[m], base_tb1 + ((a[m] - 2) * dt + j[m]) * nz + k[m], s)
        else:                                          # zeta, full radial a in [0,nr)
            for ring in (0, 1):
                m = a == ring
                for p in range(3):
                    out.emit(r[m], c_zes(p, k[m]), s * xi[p, ring, j[m]])
            m = (a >= 2) & (a - 2 < radial_in)
            out.emit(r[m], base_zb1 + ((a[m] - 2) * nt + j[m]) * dz + k[m], s)

    # --- V2 extracted (output) row offsets (match build_extraction k==2) ---
    n1_v2 = (radial_out * dt + 2) * dz   # comp0 extracted size (2dz surgery + bulk)
    n2_v2 = (dr - 1) * nt * dz           # comp1 extracted size
    r_c0b = 2 * dz                       # comp0 bulk start
    r_c1 = n1_v2                         # comp1 bulk start
    r_c2 = n1_v2 + n2_v2                 # comp2 bulk start

    # comp0 surgery [0,2dz): P axis = -d_z(chi apex) + (zeta apex difference)
    pl, m = _stencil_grid(2, dz)
    r = pl * dz + m
    out.emit(r, c_ths(pl, m), 1.0)
    out.emit(r, c_ths(pl, (m + 1) % dz), -1.0)
    out.emit(r, c_zes(pl + 1, m), 1.0)
    out.emit(r, c_zes(0, m), -1.0)

    # comp0 bulk: P[i+2,j,k] = -d_z(chi) + d_t(zeta)
    i, j, k = _stencil_grid(radial_out, dt, dz)
    r = r_c0b + np.arange(i.size)
    expand_v1(r, 1, i + 2, j, (k + 1) % nz, -1.0)
    expand_v1(r, 1, i + 2, j, k, 1.0)
    expand_v1(r, 2, i + 2, (j + 1) % nt, k, 1.0)
    expand_v1(r, 2, i + 2, j, k, -1.0)

    # comp1 bulk: Q[i+1,j,k] = d_z(s) - d_r(zeta)
    i, j, k = _stencil_grid(dr - 1, nt, dz)
    r = r_c1 + np.arange(i.size)
    expand_v1(r, 0, i + 1, j, (k + 1) % nz, 1.0)
    expand_v1(r, 0, i + 1, j, k, -1.0)
    expand_v1(r, 2, i + 2, j, k, -1.0)
    expand_v1(r, 2, i + 1, j, k, 1.0)

    # comp2 bulk: R[i+1,j,k] = -d_t(s) + d_r(chi)
    i, j, k = _stencil_grid(dr - 1, dt, nz)
    r = r_c2 + np.arange(i.size)
    expand_v1(r, 0, i + 1, (j + 1) % nt, k, -1.0)
    expand_v1(r, 0, i + 1, j, k, 1.0)
    expand_v1(r, 1, i + 2, j, k, 1.0)
    expand_v1(r, 1, i + 1, j, k, -1.0)

    n1 = int(seq.n(1, True) if dirichlet_in else seq.n(1))
    n2 = int(seq.n(2, True) if dirichlet_out else seq.n(2))
    return out.operator((n2, n1), dtype=dtype)


def _grad_stencil(seq, dirichlet_in: bool, dirichlet_out: bool, transpose: bool):
    """The analytic inverse-free polar grad ``G_0`` of ``seq``."""
    g = seq.g0_grad[(bool(dirichlet_in), bool(dirichlet_out))]
    return g.T if transpose else g


def _curl_stencil(seq, dirichlet_in: bool, dirichlet_out: bool, transpose: bool):
    """The analytic inverse-free polar curl ``G_1`` of ``seq``."""
    g = seq.g1_curl[(bool(dirichlet_in), bool(dirichlet_out))]
    return g.T if transpose else g


def _incidence_components(seq, k: int):
    """``(G_k, G_k^T)`` of ``seq``: the raw {-1, 0, +1} incidence stencils."""
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    return getattr(seq, f"g{k}"), getattr(seq, f"g{k}_T")


def apply_incidence_matrix(seq, v, k: int,
                           dirichlet_in: bool = True,
                           dirichlet_out: bool = True,
                           transpose: bool = False):
    """Apply the strong exterior-derivative ``G_k`` on extracted DoF spaces.

    The raw extracted incidence is ``E_out sp E_in^T`` (``sp`` has entries in
    ``{-1, 0, +1}``). On polar sequences the extraction is non-unitary at the
    axis, so the raw form is NOT the topological derivative and ``d.d != 0``;
    there the analytic polar stencils (grad for k=0, curl for k=1) are applied
    instead. Div (k=2) needs no correction: the V3 extraction is unitary.
    """
    if k == 0:
        return _grad_stencil(seq, dirichlet_in, dirichlet_out, transpose) @ v
    if k == 1:
        return _curl_stencil(seq, dirichlet_in, dirichlet_out, transpose) @ v
    # Div: the V3 extraction is a 0/1 selection, so the raw incidence through
    # the extractions is already the exact strong derivative.
    sp, sp_T = _incidence_components(seq, k)
    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        seq, k, dirichlet_in, dirichlet_out)
    if transpose:
        return e_in @ (sp_T @ (e_out_T @ v))
    return e_out @ (sp @ (e_in_T @ v))


#: The projection masses ``P_{k_in k_out}`` that ``set_geometry`` builds: a
#: ``k_in``-form in, a dual ``k_out``-form out, so the rows live in the
#: ``k_out`` space and the columns in the ``k_in`` space (the raw core is
#: ``seq.projection_plan[(k_out, k_in)]``). Until 2026-09-04 the scalar
#: pairs were tabulated the other way round, and ``(0, 3)`` took a 3-form.
_PROJECTION_PAIRS = ((2, 1), (1, 2), (0, 3), (3, 0))


def projection_core_apply(seq, k_in: int, k_out: int):
    """The raw-DOF apply of the projection mass ``P_{k_in k_out}`` (built by ``set_geometry``)."""
    if (k_in, k_out) not in _PROJECTION_PAIRS:
        raise ValueError(
            "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported")
    if seq.geometry is None:
        raise ValueError("no geometry installed: call seq.set_map first")
    plan, weights = seq.projection_plan[(k_out, k_in)], seq.geometry.reference_weights
    return lambda x: sumfact_apply(plan, weights, x)


def extraction(seq, k: int, dirichlet: bool):
    """The extraction ``E`` of the free (``dirichlet=False``) or Dirichlet ``k``-form space of ``seq``."""
    return seq.E(k, dirichlet)


def _mass_extraction(seq, k: int, dirichlet: bool):
    e = extraction(seq, k, dirichlet)
    return e, e.T


def _derivative_extraction(seq, k: int, dirichlet_in: bool, dirichlet_out: bool):
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    e_in = extraction(seq, k, dirichlet_in)
    e_out = extraction(seq, k + 1, dirichlet_out)
    return e_in, e_in.T, e_out, e_out.T


def _projection_extraction(seq, k_in: int, k_out: int,
                           dirichlet_in: bool, dirichlet_out: bool):
    if (k_in, k_out) not in _PROJECTION_PAIRS:
        raise ValueError(
            "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported")
    e_in = extraction(seq, k_in, dirichlet_in)
    e_out = extraction(seq, k_out, dirichlet_out)
    return e_in, e_in.T, e_out


def apply_mass_matrix(seq, v, k: int, dirichlet: bool = True):
    """Apply a mass matrix from an explicit operator bundle."""
    core = mass_core_apply(seq, k)
    e, e_T = _mass_extraction(seq, k, dirichlet)
    return e @ core(e_T @ v)


def apply_projection_matrix(seq, v,
                            k_in: int, k_out: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True):
    """Apply the projection mass ``P_{k_in k_out}`` (matrix-free, memoised on ``seq``)."""
    core = projection_core_apply(seq, k_in, k_out)
    e_in, e_in_T, e_out = _projection_extraction(seq, k_in, k_out, dirichlet_in, dirichlet_out)
    return e_out @ core(e_in_T @ v)


def apply_derivative_matrix(seq, v, k: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True,
                            transpose: bool = False):
    """Apply a weak derivative matrix from an explicit operator bundle.

    ``D_k = M_{k+1} G_k`` is applied as a composition of matrix-free applies;
    the full ``D_k`` is never materialised.
    """
    g_sp, g_sp_T = _incidence_components(seq, k)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply D{k}")
    m_apply = mass_core_apply(seq, k + 1)

    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(seq, k, dirichlet_in, dirichlet_out)

    if transpose:
        # D^T v = G^T M^T v = G^T (M v) (M is symmetric)
        return e_in @ (g_sp_T @ m_apply(e_out_T @ v))
    return e_out @ m_apply(g_sp @ (e_in_T @ v))


def apply_mass_matrix_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                     dirichlet: bool = True):
    """Apply the metric-lumped mass atom of the bundle for ``(k, dirichlet)``
    to ``v``, in the sequence's precision."""
    return _mass_atom(operators, k, dirichlet).apply_in(seq.dtype)(v)


def apply_inverse_mass_matrix(seq, operators: SequenceOperators, rhs, k: int,
                              dirichlet: bool = True, guess=None,
                              tol: Optional[float] = None,
                              maxiter: Optional[int] = None,
                              return_info: bool = False, dtype=None):
    """Solve with the inverse mass matrix: PCG with the metric-lumped mass
    atom, refined against the residual view (:func:`mrx.solvers.refine`).
    The result is in the working dtype unless ``dtype`` names the residual
    precision, for a caller that keeps computing with it."""
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    rhs, guess = _plain(seq, rhs, guess)
    on, inner = _outer(seq, tol)
    x, info = solve_singular_cg(
        lambda x: apply_mass_matrix(seq, x, k, dirichlet=dirichlet),
        rhs,
        jnp.zeros((0, rhs.shape[0]), dtype=rhs.dtype),
        mass_matvec=lambda x: apply_mass_matrix(
            seq, x, k, dirichlet=dirichlet),
        precond_matvec=_mass_atom(operators, k, dirichlet).apply,
        x0=guess,
        tol=tol,
        maxiter=maxiter,
        A_res=lambda x: apply_mass_matrix(on, x, k, dirichlet=dirichlet),
        norm=_dual_norm(operators, k, dirichlet),
        inner_tol=inner, inner_dtype=seq.dtype,
    )
    x = _out(seq, x, dtype)
    return (x, info) if return_info else x


def _pair_loop(seq, operators, on, k, dirichlet, eps, tol, maxiter, split, b, guess, vs):
    """The outer loop of a composite solve of ``(eps M_k + L_k) x = b``,
    on the pair ``(x, w)`` the split produces (``w = M_{k-1}^-1 D^T M x``:
    the Hodge split's first unknown ``g``, the shifted split's lower-level
    unknown ``z``), with the saddle residual

        upper = b - S x - eps M x - M D w,   lower = D^T M x - M_{k-1} w

    -- two applies, no nested inverse, the same criterion as the k=3
    saddle solve, in the block mass-atom norm of the two residual spaces
    -- and the correction that drives BOTH blocks: eliminating ``dw`` from
    the saddle correction gives ``(eps M + L) dx = upper - M D y`` and
    ``dw = dg + y`` with ``y = M_{k-1}^-1 lower``, a mass solve at the
    tolerance on a residual. ``split(rhs) -> (dx, dg, info)`` is the
    inner solve from zero, its own solves stopping on their own criteria at
    the inner tolerance. ``vs`` is the kernel of the operator, deflated
    from the upper residual: the harmonic forms of level ``k`` for the
    Laplacian, none for the shifted operator. Returns ``(x, info)``, ``x``
    in the residual precision.
    """
    n_k, n_l = seq.n(k, dirichlet), seq.n(k - 1, dirichlet)
    nu, nl = _dual_norm(operators, k, dirichlet), _dual_norm(operators, k - 1, dirichlet)
    b64 = b.astype(RESIDUAL_DTYPE)

    def M(s, v, j):
        return apply_mass_matrix(s, v, j, dirichlet=dirichlet)

    # The kernel is deflated from the upper residual (a singular L_k: the
    # solution is harmonic-orthogonal and b's harmonic component is not
    # reducible); the lower block has none to remove.
    _, project_dual_k = deflation_projectors(jnp.asarray(vs, dtype=RESIDUAL_DTYPE),
                                             lambda v: M(on, v, k))

    def project_dual(r):
        return jnp.concatenate([project_dual_k(r[:n_k]), r[n_k:]])

    def D(s, v):
        return apply_incidence_matrix(s, v, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet)

    def DT(s, v):
        return apply_incidence_matrix(s, v, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
                                      transpose=True)

    def residual(p):
        x, w = p[:n_k], p[n_k:]
        Mx = M(on, x, k)
        upper = b64 - apply_stiffness(on, x, k, dirichlet=dirichlet) - eps * Mx - M(on, D(on, w), k)
        lower = DT(on, Mx) - M(on, w, k - 1)
        return project_dual(jnp.concatenate([upper, lower]))

    def norm(r):
        return jnp.sqrt(nu(r[:n_k]) ** 2 + nl(r[n_k:]) ** 2)

    def solve(r):
        r_u, r_l = r[:n_k], r[n_k:]
        y = apply_inverse_mass_matrix(seq, operators, r_l, k - 1, dirichlet=dirichlet,
                                      tol=tol, maxiter=maxiter)
        # The saddle residual's lower block is -(D^T M x - M w) = -lower, so
        # eliminating dw gives (eps M + L) dx = upper - M D y, dw = dg + y.
        dx, dg, info = split(r_u - M(seq, D(seq, y), k))
        return jnp.concatenate([dx, dg + y]), info

    x0 = jnp.zeros(n_k, dtype=seq.dtype) if guess is None else guess
    p0 = jnp.concatenate([x0, jnp.zeros(n_l, dtype=seq.dtype)])
    b_packed = jnp.concatenate([b64, jnp.zeros(n_l, dtype=RESIDUAL_DTYPE)])
    p, info = refine(None, solve, b_packed, x0=p0, tol=tol, norm=norm, inner_dtype=seq.dtype,
                     residual=residual, project_dual=project_dual)
    return p[:n_k], info


def _outer(seq, tol):
    """``(on, inner_tol)`` of a solve through ``seq``: the sequence whose
    operators the outer loop of :func:`~mrx.solvers.refine` measures the
    true residual with (the float64 view, or ``seq`` itself when it is in
    the residual precision) and the inner solve's tolerance per pass
    (the square root of ``tol`` under refinement,
    :func:`~mrx.precision.inner_tol`; ``tol`` itself when the inner solve
    is the whole solve and the loop only checks it)."""
    res = seq.residual
    return (res, inner_tol(tol)) if res is not None else (seq, tol)


def _dual_norm(operators, k: int, dirichlet: bool):
    """The norm every solve's stopping criterion uses on a dual k-form:
    ``sqrt(r^T P r)`` with ``P`` the metric-lumped mass atom of the space,
    an approximation of ``M_k^-1`` that is spectrally equivalent to it
    independently of ``h``, so the criterion is the L2 norm of the
    residual's Riesz representative up to a mesh-independent factor. A
    mass solve would be exact and cost a solve per check; the atom is the
    measured middle ground."""
    P = _mass_atom(operators, k, dirichlet).apply

    def norm(r):
        return jnp.sqrt(r @ P(r))
    return norm


def _out(seq, x, dtype=None):
    """A solve's result in the sequence's dtype (the working dtype; the
    residual dtype on the float64 view), or in ``dtype``."""
    return x.astype(seq.dtype if dtype is None else dtype)

def _plain(seq, *arrays):
    """The right-hand side and guesses of a solve on a sequence that does not
    refine, in that sequence's dtype: the plain Krylov iteration runs in it.
    A refined solve keeps them as given (:func:`~mrx.solvers.refine` casts
    the right-hand side to the residual precision itself)."""
    if seq.residual is not None:
        return arrays
    return tuple(None if a is None else jnp.asarray(a).astype(seq.dtype) for a in arrays)



def apply_stiffness(seq, v, k: int, dirichlet: bool = True):
    """Apply a stiffness matrix from an explicit operator bundle.

    ``K_k = G_k^T M_{k+1} G_k`` is applied as a composition of matrix-free
    applies; the full ``K_k`` is never materialised.
    """
    if k == 3:
        return jnp.zeros_like(v)
    g_sp, g_sp_T = _incidence_components(seq, k)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply K{k}")
    m_apply = mass_core_apply(seq, k + 1)

    e, e_T = _mass_extraction(seq, k, dirichlet)
    return e @ (g_sp_T @ m_apply(g_sp @ (e_T @ v)))


def _mass_atom(operators, k: int, dirichlet: bool):
    """The metric-lumped mass atom for ``(k, dirichlet)`` from the bundle.

    Never built here: a missing atom is a missing
    :meth:`~mrx.derham_sequence.DeRhamSequence.build_preconditioners`.
    """
    atoms = _require_bundle(operators).mass_lumping or {}
    try:
        return atoms[(int(k), bool(dirichlet))]
    except KeyError:
        raise ValueError(
            f"metric_lumping mass preconditioner for k={k}, dirichlet={dirichlet} is "
            "not built; seq.build_preconditioners() builds it for the installed "
            "geometry") from None


def assemble_mass_metric_lumping_preconditioner(
        seq, operators: SequenceOperators,
        *, ks: Sequence[int] = (0, 1, 2, 3),
        dirichlet_variants: Optional[Sequence[bool]] = None,
        **kwargs) -> SequenceOperators:
    """Build the metric-lumped mass preconditioner for the given degrees.

    A :class:`~mrx.metric_lumping_laplacian.MetricLumpingMass` per
    ``(k, dirichlet)``, stored on ``operators.mass_lumping``. The build probes
    a dense polar core, so it is not free; it is done here, once, against the
    installed geometry, and nowhere else.
    """
    from mrx.metric_lumping_laplacian import MetricLumpingMass  # noqa: PLC0415
    operators = _require_bundle(operators)
    if dirichlet_variants is None:
        dirichlet_variants = (True, False)
    atoms = dict(operators.mass_lumping or {})
    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError(
                "metric_lumping mass preconditioner supports k=0..3")
        for dirichlet in dirichlet_variants:
            atoms[(int(k), bool(dirichlet))] = MetricLumpingMass(
                seq, operators, int(k), bool(dirichlet), **kwargs)
    return eqx.tree_at(lambda ops: ops.mass_lumping, operators, atoms,
                       is_leaf=lambda x: x is None or isinstance(x, dict))


def assemble_metric_lumping_laplacian_preconditioner(
        seq, operators: SequenceOperators, ks=(0, 1, 2, 3),
        dirichlets=(False, True), **kwargs):
    """Build the tensor block-Jacobi Laplacian preconditioner for ``L_k``.

    This is the production Laplacian preconditioner for k = 0..3; see
    ``docs/research/production_simplification_plan.md``.

    Build ONCE per (k, BC) -- the atom is a factorisation, not a per-apply
    computation. It is stored on ``operators.laplacian_lumping``.

    ``kwargs`` go to :class:`MetricLumpingLaplacian`. The defaults are already the
    production configuration -- pass nothing.

    NEEDS ``n >= p + 2``. ``component_factors`` forms ``A^-1 M`` per axis
    (``A`` a 1-D mass weighted by the stiffness profile) and takes its mean
    eigenvalue as a scale; below that the solve goes non-finite and numpy
    raises ``LinAlgError: Array must not contain infs or NaNs`` from inside
    ``eigvals``. ``n - p`` is the number of radial elements, so ``n = 4`` at
    ``p = 3`` is a ONE-element radial mesh. Measured on a toroid: at
    ``p = 3``, ``n = 4`` fails for
    k = 0, 1, 2 in both BCs and ``n = 5, 6, 8, 12`` all build; k = 3 builds even
    at ``n = 4``. The geometry is healthy throughout, so this is the 1-D
    factorisation, not the map.

    Returns the bundle with the atoms installed.
    """
    from mrx.metric_lumping_laplacian import MetricLumpingLaplacian  # noqa: PLC0415
    operators = _require_bundle(operators)
    atoms = dict(operators.laplacian_lumping or {})
    for k in ks:
        for dbc in dirichlets:
            atoms[(int(k), bool(dbc))] = MetricLumpingLaplacian(
                seq, operators, int(k), bool(dbc), **kwargs)
    return eqx.tree_at(lambda ops: ops.laplacian_lumping, operators, atoms,
                       is_leaf=lambda x: x is None or isinstance(x, dict))


def _laplacian_atom(operators, k: int, dirichlet: bool):
    """The metric-lumped Laplacian atom for ``(k, dirichlet)`` from the bundle."""
    atoms = _require_bundle(operators).laplacian_lumping or {}
    try:
        return atoms[(int(k), bool(dirichlet))]
    except KeyError:
        raise ValueError(
            f"metric_lumping Laplacian atom for k={k}, dirichlet={dirichlet} is not "
            "built; seq.build_preconditioners() builds it for the installed geometry") from None


def _shifted_atom_apply(operators, k: int, dirichlet: bool, eps):
    """The preconditioner of ``M_k + eps S_k``: the shifted-stiffness atom.

    The strong-half (primal-axis) Kronecker terms of the metric-lumped
    Laplacian atom, divided by ``1 + eps lambda`` in their eigenbasis, i.e.
    ``(M^ + eps S^)^-1`` for the atom's own separable mass, plus the dense
    ``(M + eps S)^-1`` on the core rows. Measured on li383 p=3 at the
    smoothing eps against the mass atom (CG iterations on ``M_k + eps
    S_k``): k=2 69 vs 153 at (8,16,8), 117 vs 371 at (12,24,12); k=1 74 vs
    181 and 128 vs 422. Two "consistent" factorisations with the Jacobian in
    the 1-D masses were measured and lost to this plain shift (~200 at
    (12,24,12)). Its implied mass is a worse M than the mass atom's, so
    below ``eps n_r^2 ~ 0.006`` (the resistive step's eta dt) the mass atom
    wins by up to 2x on ~100 iterations; the smoothing eps is 10x above the
    crossover. docs/research/shifted_split_2026-09-02.md.
    """
    return _laplacian_atom(operators, k, dirichlet).shifted_stiffness_apply(eps)


def apply_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                   dirichlet: bool = True):
    """Apply the metric-lumped Laplacian atom of the bundle for ``(k, dirichlet)`` to ``v``."""
    del seq
    return _laplacian_atom(operators, k, dirichlet).apply(v)


def _hat_solve(seq, operators, b, k: int, dirichlet: bool, *, tol, maxiter):
    """PCG on ``L^_k = S_k + M_k D_{k-1} W D_{k-1}^T M_k``, ``W`` the mass atom:
    an inner solve of the Hodge split, stopping on its own preconditioned
    criterion at ``tol`` (the split's outer loop is the pair loop of
    :func:`apply_inverse_laplacian_hodge`).

    The strong stiffness ``S_k`` is singular on the exact forms, and PCG on
    it is NOT viable: the right-hand sides of the Hodge split are consistent
    only to the tolerance of the solve that produced them, that kernel
    component is invisible to ``S_k``, and the atom amplifies it on every
    iteration (5-13x at k=1 Dirichlet -> a residual floor of 1e-6; 100-500x
    on the free and k>=2 kernels -> exponential blow-up, measured 2026-09-02).

    ``L^_k`` is SPD for ANY SPD ``W``: the kernel component now has a positive
    eigenvalue and is damped instead of fed back.  The solution's exact part
    depends on ``W`` (it is wrong), its exact-orthogonal part does not (see
    :func:`apply_inverse_laplacian_hodge`, which uses only that part or
    corrects the rest in closed form).  ``W`` = the metric-lumped mass atom of
    level ``k-1`` puts the exact-form eigenvalues at the same ``h^-2`` scale
    as ``S_k`` on its range, which is what the componentwise-Laplacian atom
    expects; those modes are barely excited by a right-hand side of the
    split anyway.  Harmonic forms deflated. ``W`` is part of the OPERATOR
    and is applied in the sequence's precision (``apply_in``): on the
    float64 view its float32 payload would perturb ``L^_k`` at 1e-7.
    """
    b, = _plain(seq, b)

    def D(v):
        return apply_incidence_matrix(seq, v, k - 1, dirichlet_in=dirichlet,
                                      dirichlet_out=dirichlet)

    def DT(v):
        return apply_incidence_matrix(seq, v, k - 1, dirichlet_in=dirichlet,
                                      dirichlet_out=dirichlet, transpose=True)

    def M(v):
        return apply_mass_matrix(seq, v, k, dirichlet=dirichlet)

    W = _mass_atom(operators, k - 1, dirichlet).apply_in(seq.dtype)

    def L_hat(x):
        return apply_stiffness(seq, x, k, dirichlet=dirichlet) + M(D(W(DT(M(x)))))

    return solve_singular_cg(
        L_hat, b,
        mass_matvec=M,
        precond_matvec=_laplacian_atom(operators, k, dirichlet).apply,
        vs=_nullspace_vectors(operators, k, dirichlet),
        tol=tol,
        maxiter=maxiter,
    )


def apply_inverse_laplacian_hodge(seq, operators: SequenceOperators, rhs, k: int,
                                  dirichlet: bool = True, guess=None,
                                  tol: Optional[float] = None,
                                  maxiter: Optional[int] = None,
                                  return_info: bool = False, dtype=None):
    """``L_k^{-1} rhs`` for ``k >= 1`` by Hodge splitting (TB, 2026-09-02).

    ``L_k = S_k + M_k D_{k-1} M_{k-1}^{-1} D_{k-1}^T M_k`` and
    ``D_{k-1}^T S_k = 0``, so projecting ``L_k x = b`` onto the exact forms
    ``x = D_{k-1} a + x_perp`` gives the exact part in closed form:

        S_{k-1} g = D_{k-1}^T b          (g := M_{k-1}^{-1} S_{k-1} a; consistent,
                                          D_{k-2}^T D_{k-1}^T = 0)
        S_k x_perp = b - M_k D_{k-1} g   (consistent: D_{k-1}^T of it is 0)
        S_{k-1} a = M_{k-1} g - D_{k-1}^T M_k x_perp
        x = x_perp + D_{k-1} a

    Every solve is a PCG on a strong stiffness with its own atom -- no saddle
    system, no MINRES, no mass inverse anywhere, and the weak half
    ``D M^{-1} D^T`` never appears: the atom's model of it, the part that was
    measured to cost 4-35x in iterations, only ever acts on exact forms.

    The strong stiffnesses are singular on the exact forms and are NOT solved
    as such (:func:`_hat_solve`): each solve is on the SPD
    ``L^_j = S_j + M_j D W D^T M_j`` instead, whose exact-orthogonal solution
    is the one wanted for any SPD ``W``.  The exact part of each intermediate
    is wrong and irrelevant -- ``g`` and ``a`` are only ever used through
    ``D_{k-1}`` or paired against ``x_perp``, and the exact part of ``x_perp``
    itself is removed by the closing ``S_{k-1}`` solve, which at ``k = 1`` is
    the explicitly deflated scalar solve (:func:`apply_inverse_laplacian`,
    ``k = 0``).

    ``k = 3`` is NOT split (``S_3 = 0``: it would be two ``L^_2`` solves in
    series, measured at 771 iterations against 694 saddle MINRES on QA n=16,
    with the exact residual 40x worse because the split stops on the
    ``L^_2`` residual and the exact one carries ``M_2^{-1}`` amplification --
    which the Leray projection, its consumer, needs); it stays on the saddle
    MINRES.  ``guess`` warm-starts the ``L^_k`` solve; ``return_info`` reports
    its signed count.  Harmonic forms of every level are deflated; the
    solution is harmonic-orthogonal in ``M_k``.
    """
    if k not in (1, 2):
        raise ValueError(f"apply_inverse_laplacian_hodge: k must be 1 or 2, got {k}")
    operators = _require_bundle(operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    rhs, guess = _plain(seq, rhs, guess)
    d = dirichlet

    def D(v, j):
        return apply_incidence_matrix(seq, v, j, dirichlet_in=d, dirichlet_out=d)

    def DT(v, j):
        return apply_incidence_matrix(seq, v, j, dirichlet_in=d, dirichlet_out=d,
                                      transpose=True)

    def M(v, j):
        return apply_mass_matrix(seq, v, j, dirichlet=d)

    # The split's three solves are inner solves: each stops on its own
    # operator's preconditioned criterion at the inner tolerance, and the
    # outer loop is on the pair (x, g) with the saddle residual
    # (:func:`_pair_loop`).
    on, inner = _outer(seq, tol)

    def solve(b, j):
        if j == 0:
            return _k0_solve(seq, operators, b, d, tol=inner, maxiter=maxiter)
        return _hat_solve(seq, operators, b, j, d, tol=inner, maxiter=maxiter)

    def split(b):
        # g IS M_{k-1}^-1 D^T M x of the returned x (the closing solve says
        # S a = M g - D^T M x_perp): the pair the outer loop carries.
        g, _ = solve(DT(b, k - 1), k - 1)
        x_perp, info = solve(b - M(D(g, k - 1), k), k)
        a, _ = solve(M(g, k - 1) - DT(M(x_perp, k), k - 1), k - 1)
        return x_perp + D(a, k - 1), g, info

    x, info = _pair_loop(seq, operators, on, k, d, 0.0, tol, maxiter,
                         lambda r: split(r.astype(seq.dtype)), rhs, guess,
                         _nullspace_vectors(operators, k, d))
    x = _out(seq, x, dtype)
    return (x, info) if return_info else x


def _k0_solve(seq, operators, b, dirichlet, *, tol, maxiter, guess=None, on=None, inner=None):
    """The deflated PCG on ``S_0`` with the k=0 Laplacian atom, under the
    outer loop of :func:`~mrx.solvers.refine` measuring on ``on`` with the
    inner tolerance ``inner``; on its own preconditioned criterion at
    ``tol`` when ``on`` is None -- the closing solve of the k=1 Hodge
    split, whose outer loop is the split's pair loop."""
    return solve_singular_cg(
        lambda x: apply_stiffness(seq, x, 0, dirichlet=dirichlet),
        b,
        mass_matvec=lambda x: apply_mass_matrix(seq, x, 0, dirichlet=dirichlet),
        precond_matvec=_laplacian_atom(operators, 0, dirichlet).apply,
        x0=guess,
        vs=_nullspace_vectors(operators, 0, dirichlet),
        tol=tol,
        maxiter=maxiter,
        A_res=None if on is None else (lambda x: apply_stiffness(on, x, 0, dirichlet=dirichlet)),
        norm=_dual_norm(operators, 0, dirichlet),
        inner_tol=inner, inner_dtype=seq.dtype,
    )


def apply_inverse_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                  dirichlet: bool = True, guess=None,
                                  tol: Optional[float] = None,
                                  maxiter: Optional[int] = None,
                                  return_info: bool = False, dtype=None):
    """Solve with the inverse of the unshifted Hodge Laplacian ``L_k``.

    ``k = 0``: the deflated scalar PCG below.  ``k = 1, 2``: the Hodge-split
    solve :func:`apply_inverse_laplacian_hodge` -- SPD PCGs on the strong
    stiffnesses, no saddle system.  ``k = 3``: the saddle-point MINRES
    (``S_3 = 0``, nothing to split; see the split's docstring), which is
    also the solver of the SHIFTED Laplacian at every k
    (:func:`apply_inverse_shifted_laplacian`).  Every solve is
    preconditioned by the metric-lumped atoms of the bundle. The result is
    in the working dtype unless ``dtype`` names the residual precision (the
    solution the outer loop accumulated, before its rounding to float32).
    """
    operators = _require_bundle(operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    rhs, guess = _plain(seq, rhs, guess)

    if k == 0:
        on, inner = _outer(seq, tol)
        u, info = _k0_solve(seq, operators, rhs, dirichlet, tol=tol, maxiter=maxiter,
                            guess=guess, on=on, inner=inner)
        u = _out(seq, u, dtype)
        return (u, info) if return_info else u

    if k == 3:
        u, _, info = apply_inverse_laplacian_saddle(
            seq, operators, rhs, 3, 0.0, dirichlet=dirichlet, guess=guess,
            tol=tol, maxiter=maxiter)
        u = _out(seq, u, dtype)
        return (u, info) if return_info else u
    return apply_inverse_laplacian_hodge(
        seq, operators, rhs, k, dirichlet=dirichlet, guess=guess,
        tol=tol, maxiter=maxiter, return_info=return_info, dtype=dtype)


def apply_inverse_laplacian_saddle(seq, operators: SequenceOperators, rhs, k: int,
                                   eps: float, dirichlet: bool = True, guess=None,
                                   sigma_guess=None, tol: Optional[float] = None,
                                   maxiter: Optional[int] = None):
    """The saddle-point MINRES on ``L_k + eps M_k`` (``k >= 1``): ``(u, sigma, info)``.

        | S_k + eps M_k   D_{k-1} | | u     |   | rhs |
        | D_{k-1}^T      -M_{k-1} | | sigma | = | 0   |

    ``u`` solves ``(L_k + eps M_k) u = rhs`` and ``sigma = M_{k-1}^-1
    D_{k-1}^T u`` is its weak codifferential -- for the k=3 solve of the
    Leray projection the gradient part it removes, handed back here rather
    than recomputed by a mass solve. Block-diagonal preconditioner: the
    Laplacian atom of level ``k`` on the upper block (the Schur complement
    of the saddle system IS ``L_k``, which is what the atom approximates,
    and MINRES needs its preconditioner SPD, which the atom is) and the
    mass atom of level ``k - 1`` on the lower block. At ``eps = 0`` the
    harmonic forms of level ``k`` are deflated from the upper block and the
    ``eps M_k`` term is not applied; the lower block needs no deflation
    (see :func:`~mrx.solvers.solve_saddle_point_minres`). ``guess`` and
    ``sigma_guess`` warm-start the two blocks together. Refined against the
    residual view; ``u`` and ``sigma`` then come back in the residual
    precision (the Leray projection forms its force from them before it
    rounds).
    """
    operators = _require_bundle(operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    rhs, guess, sigma_guess = _plain(seq, rhs, guess, sigma_guess)
    res, inner = _outer(seq, tol)

    def stiffness_on(s, x):
        if eps == 0:
            return apply_stiffness(s, x, k, dirichlet=dirichlet)
        return (apply_stiffness(s, x, k, dirichlet=dirichlet)
                + eps * apply_mass_matrix(s, x, k, dirichlet=dirichlet))

    vs_upper = (_nullspace_vectors(operators, k, dirichlet) if eps == 0
                else jnp.zeros((0, rhs.shape[0]), dtype=rhs.dtype))

    def saddle_res(u, s):
        return (stiffness_on(res, u) + apply_derivative_matrix(
                    res, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
                apply_derivative_matrix(res, u, k - 1, dirichlet_in=dirichlet,
                                        dirichlet_out=dirichlet, transpose=True)
                - apply_mass_matrix(res, s, k - 1, dirichlet=dirichlet))

    return solve_saddle_point_minres(
        stiffness_matvec=lambda x: stiffness_on(seq, x),
        derivative_matvec=lambda s: apply_derivative_matrix(
            seq, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: apply_derivative_matrix(
            seq, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: apply_mass_matrix(
            seq, s, k - 1, dirichlet=dirichlet),
        b_upper=rhs,
        n_upper=seq.n(k, dirichlet),
        n_lower=seq.n(k - 1, dirichlet),
        precond_upper=_laplacian_atom(operators, k, dirichlet).apply,
        precond_lower=_mass_atom(operators, k - 1, dirichlet).apply,
        mass_upper_matvec=lambda x: apply_mass_matrix(
            seq, x, k, dirichlet=dirichlet),
        vs_upper=vs_upper,
        x0_upper=guess,
        x0_lower=sigma_guess,
        tol=tol,
        maxiter=maxiter,
        saddle_res=saddle_res,
        norm_upper=_dual_norm(operators, k, dirichlet),
        norm_lower=_dual_norm(operators, k - 1, dirichlet),
        inner_tol=inner, inner_dtype=seq.dtype,
    )


def apply_inverse_shifted_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                    eps: float, dirichlet: bool = True, guess=None,
                                    tol: Optional[float] = None,
                                    maxiter: Optional[int] = None,
                                    return_info: bool = False):
    """Solve with the inverse of the shifted Hodge Laplacian ``L_k + eps M_k``.

    ``k = 0``: deflated PCG with the k=0 Laplacian atom (measured in its
    favour against the shifted diagonal on the shifted operator); at ``eps
    = 0`` the harmonic forms are deflated and the mass term is not applied.
    ``k >= 1``: the saddle-point MINRES of
    :func:`apply_inverse_laplacian_saddle`, ``u`` only.
    """
    operators = _require_bundle(operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    rhs, guess = _plain(seq, rhs, guess)

    if k == 0:
        res, inner = _outer(seq, tol)

        def A_on(s, x):
            if eps == 0:
                return apply_stiffness(s, x, 0, dirichlet=dirichlet)
            return (apply_stiffness(s, x, 0, dirichlet=dirichlet)
                    + eps * apply_mass_matrix(s, x, 0, dirichlet=dirichlet))

        if eps == 0:
            vs = _nullspace_vectors(operators, 0, dirichlet)

            def mass_matvec(x):
                return apply_mass_matrix(seq, x, 0, dirichlet=dirichlet)
        else:
            vs = jnp.zeros((0, rhs.shape[0]), dtype=rhs.dtype)
            mass_matvec = None
        u, info = solve_singular_cg(
            lambda x: A_on(seq, x), rhs, mass_matvec=mass_matvec,
            precond_matvec=_laplacian_atom(operators, 0, dirichlet).apply,
            x0=guess, vs=vs, tol=tol, maxiter=maxiter,
            A_res=lambda x: A_on(res, x),
            norm=_dual_norm(operators, 0, dirichlet), inner_tol=inner, inner_dtype=seq.dtype)
        u = _out(seq, u)
        return (u, info) if return_info else u

    u, _, info = apply_inverse_laplacian_saddle(
        seq, operators, rhs, k, eps, dirichlet=dirichlet, guess=guess,
        tol=tol, maxiter=maxiter)
    u = _out(seq, u)
    return (u, info) if return_info else u


def apply_inverse_mass_plus_eps_laplace_matrix(seq, operators: SequenceOperators, rhs, k: int,
                                               eps: float, dirichlet: bool = True, guess=None,
                                               tol: Optional[float] = None,
                                               maxiter: Optional[int] = None,
                                               return_info: bool = False):
    """Solve ``(M_k + eps L_k) x = rhs`` as two SPD solves.

    With ``S_k = D_k^T M_{k+1} D_k`` the strong stiffness and
    ``L_k = S_k + M_k D_{k-1} M_{k-1}^{-1} D_{k-1}^T M_k`` the Hodge
    Laplacian, ``D_k D_{k-1} = 0`` gives, exactly,

        (M_k + eps L_k)^{-1} = (M_k + eps S_k)^{-1}
                               - eps D_{k-1} (M_{k-1} + eps S_{k-1})^{-1} D_{k-1}^T

    (multiply out: ``(M_k + eps S_k) D_{k-1} = M_k D_{k-1}``, and the cross
    terms cancel through ``(M_{k-1} + eps S_{k-1}) M_{k-1}^{-1} = I + eps
    S_{k-1} M_{k-1}^{-1}``). Each system is a mass plus a semidefinite
    stiffness -- SPD, matrix-free, PCG -- so there is no saddle system, no
    inner mass solve and nothing to deflate. ``k = 0`` is the first solve
    alone; ``k = 3`` has ``S_3 = 0``. Each level is preconditioned by its
    shifted-stiffness atom (:func:`_shifted_atom_apply`), ``(M^ + eps
    S^)^-1`` from the Laplacian atom on the bundle.

    Measured on li383 p=3 at the velocity-smoothing ``eps = 0.064 / n_r^2``,
    both solves together, against the saddle MINRES with the mass atom this
    replaced (same tolerance): 145 / 249 iterations at (8,16,8) /
    (12,24,12) against 2134 / 8478; with the mass atom on the split instead
    330 / 772 / 1326 at (8,16,8) / (12,24,12) / (16,32,16) against 2134 /
    8478 / 20362. The solutions agree to ``3 tol`` in the mass norm
    (``docs/research/shifted_split_2026-09-02.md``).

    ``eps`` may be a traced scalar: the relaxation's resistive step passes
    ``dt * eta`` from inside a jitted step. Nothing here branches on its
    value, so ``eps = 0`` runs two mass solves rather than dispatching to
    :func:`apply_inverse_mass_matrix` -- a caller with ``eps = 0`` wants
    that function and should say so. ``info`` is the summed signed
    iteration count of the two solves, negative when both converged.
    """
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    rhs, guess = _plain(seq, rhs, guess)
    on, inner = _outer(seq, tol)

    def A_on(s, j, x):
        return (apply_mass_matrix(s, x, j, dirichlet=dirichlet)
                + eps * apply_stiffness(s, x, j, dirichlet=dirichlet))

    def shifted_cg(j, b, tol, x0=None, on=None):
        # PCG on M_j + eps S_j with the shifted atom: under the outer loop
        # measuring on ``on``, or on its own criterion at ``tol`` (an inner
        # solve of the split, whose outer loop is the pair loop).
        return solve_singular_cg(
            lambda x: A_on(seq, j, x),
            b,
            jnp.zeros((0, b.shape[0]), dtype=b.dtype),
            precond_matvec=_shifted_atom_apply(operators, j, dirichlet, eps),
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            A_res=None if on is None else (lambda x: A_on(on, j, x)),
            norm=_dual_norm(operators, j, dirichlet),
            inner_tol=inner, inner_dtype=seq.dtype,
        )

    if k == 0:
        x, info = shifted_cg(0, rhs, tol, x0=guess, on=on)
        x = _out(seq, x)
        return (x, info) if return_info else x

    def split(b):
        # z IS M_{k-1}^-1 D^T M x of the returned x (D^T M x_1 = D^T b since
        # D^T S_k = 0, and (M + eps S) z = D^T b): the pair the outer loop carries.
        x, info = shifted_cg(k, b, inner)
        z, info_lower = shifted_cg(
            k - 1,
            apply_incidence_matrix(seq, b, k - 1, dirichlet, dirichlet, transpose=True),
            inner)
        x = x - eps * apply_incidence_matrix(seq, z, k - 1, dirichlet, dirichlet)
        total = jnp.abs(info) + jnp.abs(info_lower)
        return x, z, jnp.where((info <= 0) & (info_lower <= 0), -total, total)

    # The composite under the outer loop on the pair (x, z) of
    # (M_k + eps L_k) x = b, i.e. (1/eps) M x + L x = b / eps (:func:`_pair_loop`);
    # M_k + eps L_k is SPD, there is no kernel to deflate.
    x, info = _pair_loop(seq, operators, on, k, dirichlet, 1.0 / eps, tol, maxiter,
                         lambda r: split(eps * r.astype(seq.dtype)), rhs / eps, guess,
                         jnp.zeros((0, seq.n(k, dirichlet))))
    x = _out(seq, x)
    return (x, info) if return_info else x


def _laplacian_apply(seq, v, k: int, dirichlet: bool, minv):
    """``S_k v + D_{k-1} minv(D_{k-1}^T v, k - 1)``: the Hodge Laplacian with
    ``minv(w, j)`` standing in for ``M_j^{-1} w`` (``S_3 = 0``, ``L_0 = S_0``)."""
    if k not in (0, 1, 2, 3):
        raise ValueError("k must be 0, 1, 2 or 3")
    strong = apply_stiffness(seq, v, k, dirichlet=dirichlet)
    if k == 0:
        return strong
    Dt_v = apply_derivative_matrix(seq, v, k - 1, dirichlet_in=dirichlet,
                                   dirichlet_out=dirichlet, transpose=True)
    return strong + apply_derivative_matrix(seq, minv(Dt_v, k - 1), k - 1,
                                            dirichlet_in=dirichlet, dirichlet_out=dirichlet)


def apply_laplacian(seq, operators: SequenceOperators, v, k: int,
                    dirichlet: bool = True, guess=None,
                    tol: Optional[float] = None,
                    maxiter: Optional[int] = None):
    """Apply the Hodge Laplacian ``L_k = S_k + D M_{k-1}^{-1} D^T``, the weak
    half through a mass solve (``guess``, ``tol``, ``maxiter`` are its)."""
    return _laplacian_apply(
        seq, v, k, dirichlet,
        lambda w, j: apply_inverse_mass_matrix(seq, operators, w, j, dirichlet=dirichlet,
                                               guess=guess, tol=tol, maxiter=maxiter))


def apply_laplacian_approx(seq, operators: SequenceOperators, v, k: int,
                           dirichlet: bool = True):
    """Linear approximation of the Hodge Laplacian apply.

    Replaces the exact ``M_{k-1}^{-1}`` in the Schur term of ``L_k`` with one
    apply of the mass atom. The result is a fully linear SPD matvec: safe to
    nest inside Krylov iterations and to use as a preconditioner or a
    diagnostic ``L_k``-apply. It is not exactly ``L_k`` unless the metric is
    tensor-separable on the reference domain.
    """
    return _laplacian_apply(
        seq, v, k, dirichlet,
        lambda w, j: apply_mass_matrix_preconditioner(seq, operators, w, j, dirichlet=dirichlet))
