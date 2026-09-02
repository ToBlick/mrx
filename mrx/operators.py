"""Matrix-free operator bundle (:class:`SequenceOperators`), assembly of its fields, and the operator applies and solves."""
from __future__ import annotations

from typing import Optional, Sequence
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.extraction_operators import MatrixFreeExtraction
import numpy as np

from mrx.preconditioners import (
    MassPreconditionerSpec,
    SchurPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    _symmetrize,
    default_mass_preconditioner,
)
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg
import mrx
def _nullspace_vectors(operators, k: int, dirichlet: bool):
    """Return the stacked nullspace array for ``(k, dirichlet)``."""
    from mrx.nullspace import get_nullspace
    return get_nullspace(operators, k, dirichlet)


def _saddle_nullspaces(seq, operators, k: int, dirichlet: bool):
    """Return upper/lower nullspace arrays for the saddle-point system."""
    from mrx.nullspace import get_saddle_point_nullspaces
    return get_saddle_point_nullspaces(seq, operators, k, dirichlet)


class SequenceOperators(eqx.Module):
    """Everything built FROM a geometry: preconditioners and harmonic forms.

    The sequence itself is static (bases, extraction, incidence); this bundle
    holds every factorisation of the installed metric -- Jacobi diagonals,
    metric-lumped mass and Laplacian atoms, probed Schur diagonals -- and the
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
    # The Jacobi option (``build_preconditioners(jacobi=True)``): the inverse
    # diagonals of ``E M_k E^T``, of ``L_k`` (as ``apply_laplacian_approx``
    # applies it) and of the approximate Schur operator of the saddle solves,
    # all by one-hot probes of the applies themselves, keyed ``(k, dirichlet)``.
    mass_jacobi: Optional[dict] = None
    laplacian_jacobi: Optional[dict] = None
    schur_jacobi: Optional[dict] = None
    # Harmonic forms of the k-form Laplacians, keyed ``(k, dirichlet)``: an
    # array ``(n_vectors, n_k)``, one nullspace vector per row. The shapes are
    # topological (Betti numbers); the values belong to the geometry. Zero
    # until computed, so deflation is a no-op.
    nullspaces: Optional[dict] = None


def new_operators(seq) -> SequenceOperators:
    """The empty bundle for ``seq``: zero nullspaces, no preconditioners."""
    from mrx.nullspace import init_nullspaces  # noqa: PLC0415
    return init_nullspaces(seq, SequenceOperators(mass_lumping={}, laplacian_lumping={},
                                                  mass_jacobi={}, laplacian_jacobi={},
                                                  schur_jacobi={}, nullspaces={}))


def _require_bundle(operators):
    if operators is None:
        raise ValueError(
            "no operator bundle: call seq.build_preconditioners() after set_map")
    return operators


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _assemble_weighted_1d_stiffness(
        primal_basis: jnp.ndarray,
        derivative_basis: jnp.ndarray,
        weights: jnp.ndarray,
        incidence: jnp.ndarray) -> jnp.ndarray:
    mass_d = _assemble_weighted_1d_mass(derivative_basis, weights)
    stiffness = incidence.T @ (mass_d @ incidence)
    return _symmetrize(stiffness)


def _materialize_default_mass_preconditioner(
        seq, operators: SequenceOperators, *, k: int):
    # The `_tensor_available` gate here was a leftover from when
    # `default_mass_preconditioner()` meant kind='tensor', which DOES need an
    # eager assembly and so needed a fallback. It has meant 'metric_lumping'
    # since 2026-08-22, and that is always buildable -- so the gate was
    # silently downgrading the saddle solve's LOWER block to a per-DoF
    # diagonal whenever the tensor factors happened not to be assembled,
    # which is the normal case.
    #
    # MEASURED (2026-08-24), same operator, same block-Jacobi upper block,
    # toroid p=3 k=2 free: 84 iterations with block_jacobi below, 9612 with
    # the jacobi diagonal. The k>=1 saddle solves were not "badly
    # conditioned"; they were running without the mass preconditioner.
    del seq, operators, k
    return default_mass_preconditioner()


def _materialize_default_saddle_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        coupled_preconditioner: bool = False):
    """The k>=1 saddle default: metric_lumping mass, metric_lumping outer.

    You get what you built. The atom when it has been assembled for this
    ``(k, BC)``, and ``'none'`` otherwise -- never a substitute.

    This used to drop to the per-DoF diagonal with a RuntimeWarning, which is
    how the relaxation loop came to run its innermost solve on the diagonal
    without anyone noticing. Probe-building a jacobi diagonal as a soft fallback
    is not a service: it silently swaps in a different, worse preconditioner and
    the solve merely gets slower, which is invisible. Running unpreconditioned
    is visible -- the solve stalls or fails, and the cause is the missing
    assembly.

    Preconditioners are built explicitly, by the caller, against a known
    geometry -- see
    :meth:`~mrx.derham_sequence.DeRhamSequence.set_map_and_preconditioners`.
    ``set_geometry`` drops the atoms, so "assembled" always means "assembled for
    the geometry now installed".

    ``schur.inner`` is metric_lumping. Under ``outer='metric_lumping'`` the
    atom IS the upper-block inverse and the inner slot does no work at all.
    """
    outer = ('metric_lumping' if _metric_lumping_available(operators, k, dirichlet) else 'none')
    return SaddlePointPreconditionerSpec(
        mass=_materialize_default_mass_preconditioner(seq, operators, k=k - 1),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='metric_lumping'),
            outer=MassPreconditionerSpec(kind=outer),
        ),
        coupled=coupled_preconditioner,
    )


def _materialize_default_scalar_hodge_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool = True,
        eps: float = 0.0):
    """The scalar (k=0) Laplacian default: the block-Jacobi atom, required.

    Same rule as the k>=1 saddle default, and for the same reason: an
    availability test cannot tell an atom built for THIS geometry from one left
    over from the last, so it is not asked. Build it explicitly.

    ``eps > 0`` gets nothing: the atom approximates ``L_k``, not
    ``L_k + eps M_k``, and its fit to the shifted operator is unmeasured (audit
    item 3.2). Pass an explicit kind there if you want one.
    """
    if eps != 0.0:
        return MassPreconditionerSpec(kind='none')
    if _metric_lumping_available(operators, k, dirichlet):
        return MassPreconditionerSpec(kind='metric_lumping')
    return MassPreconditionerSpec(kind='none')


def _coerce_diffusion_preconditioner_spec(
        seq, operators: SequenceOperators, *, k: int, preconditioner):
    if preconditioner is None or preconditioner == 'auto':
        # 'auto' is the shifted-stiffness atom, (M^_j + eps S^_j)^-1 built
        # from the metric-lumped Laplacian atom of level j (kind
        # 'metric_lumping' in this slot); see
        # MetricLumpingLaplacian.shifted_stiffness_apply.
        del seq, operators, k
        return default_mass_preconditioner()
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        return MassPreconditionerSpec(kind=preconditioner)
    raise TypeError(
        'diffusion preconditioner must be a kind string or MassPreconditionerSpec')


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
        return jnp.zeros((n_out, n0)).at[j, j].set(-1.0).at[j, j + 1].set(1.0)
    if typ == 'periodic':
        j = jnp.arange(n0)
        return jnp.zeros((n0, n0)).at[j, j].set(-1.0).at[j, (j + 1) % n0].set(1.0)
    if typ == 'constant':
        return jnp.zeros((n0, n0))
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
    if seq.mass_apply is None:
        raise ValueError("no geometry installed: call seq.set_map first")
    return seq.mass_apply[k]


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

    def operator(self, shape):
        """Return the collected triplets as a :class:`MatrixFreeExtraction`.

        Duplicates are summed on the host first so the device arrays hold one
        entry per nonzero.
        """
        import scipy.sparse as _sps
        coo = _sps.coo_matrix(
            (np.concatenate(self.data),
             (np.concatenate(self.rows).astype(np.int32),
              np.concatenate(self.cols).astype(np.int32))),
            shape=shape).tocsr().tocoo()
        return MatrixFreeExtraction.from_coo(coo.row, coo.col, coo.data, shape)


def build_grad_stencil_g0(seq, xi, dirichlet_in: bool, dirichlet_out: bool):
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
    return out.operator((n1, n0))


def build_curl_stencil_g1(seq, xi, dirichlet_in: bool, dirichlet_out: bool):
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
    return out.operator((n2, n1))


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


# Row/column spaces of the projection masses ``P_{k_in k_out}``: rows are the
# space of ``e_out`` and columns the space of ``e_in`` in
# :func:`_projection_extraction`.
_PROJECTION_SPACES = {(2, 1): (1, 2), (1, 2): (2, 1), (0, 3): (0, 3), (3, 0): (3, 0)}


def projection_core_apply(seq, k_in: int, k_out: int):
    """The raw-DOF apply of the projection mass ``P_{k_in k_out}`` (built by ``set_geometry``)."""
    try:
        k_row, k_col = _PROJECTION_SPACES[(k_in, k_out)]
    except KeyError:
        raise ValueError(
            "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
        ) from None
    if seq.projection_apply is None:
        raise ValueError("no geometry installed: call seq.set_map first")
    return seq.projection_apply[(k_row, k_col)]


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
    try:
        k_row, k_col = _PROJECTION_SPACES[(k_in, k_out)]
    except KeyError:
        raise ValueError(
            "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
        ) from None
    # Rows of P_{k_in k_out} live in the space of ``e_out`` (``k_row``) and
    # its columns in the space of ``e_in`` (``k_col``).
    e_in = extraction(seq, k_col, dirichlet_in)
    e_out = extraction(seq, k_row, dirichlet_out)
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
                                     dirichlet: bool = True,
                                     kind: str = 'auto'):
    """Apply a mass-matrix preconditioner from an explicit operator bundle.

    Parameters
    ----------
    kind : {'auto', 'jacobi', 'metric_lumping'}
        Which preconditioner to use.
    """
    apply = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        preconditioner=kind,
        allow_none=False,
    )
    return apply(v)


def apply_inverse_mass_matrix(seq, operators: SequenceOperators, rhs, k: int,
                              dirichlet: bool = True, guess=None,
                              tol: Optional[float] = None,
                              maxiter: Optional[int] = None,
                              preconditioner='auto',
                              return_info: bool = False):
    """Solve with the inverse mass matrix from an explicit operator bundle.

    ``preconditioner`` accepts a kind string or a
    :class:`MassPreconditionerSpec`. When omitted (``'auto'``) it resolves to
    :func:`~mrx.preconditioners.default_mass_preconditioner` (metric_lumping).
    """
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    precond_apply = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        preconditioner=preconditioner,
        allow_none=True,
    )
    x, info = solve_singular_cg(
        lambda x: apply_mass_matrix(seq, x, k, dirichlet=dirichlet),
        rhs,
        jnp.zeros((0, rhs.shape[0]), dtype=rhs.dtype),
        mass_matvec=lambda x: apply_mass_matrix(
            seq, x, k, dirichlet=dirichlet),
        precond_matvec=precond_apply,
        x0=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (x, info) if return_info else x


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


def _coerce_mass_preconditioner_spec(preconditioner):
    if preconditioner is None:
        return default_mass_preconditioner()
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        return MassPreconditionerSpec(kind=preconditioner)
    raise TypeError(
        "mass preconditioner must be a kind string or MassPreconditionerSpec")


def _mass_metric_lumping_for(seq, operators, k: int, dirichlet: bool):
    """The metric-lumped MASS preconditioner for ``(k, dirichlet)`` from the bundle.

    This is the default mass preconditioner (:func:`~mrx.preconditioners.
    default_mass_preconditioner`). It is never built here: a missing atom is
    a missing :meth:`~mrx.derham_sequence.DeRhamSequence.build_preconditioners`.
    """
    del seq
    atoms = _require_bundle(operators).mass_lumping or {}
    try:
        return atoms[(int(k), bool(dirichlet))]
    except KeyError:
        raise ValueError(
            f"metric_lumping mass preconditioner for k={k}, dirichlet={dirichlet} is "
            "not built; seq.build_preconditioners() builds it for the installed "
            "geometry") from None


def _diagonal_from_matvec(operator_apply, size: int):
    """Probe ``diag(A)`` by one-hot vectors, 16 columns per ``lax.map`` batch.

    A full ``vmap`` over chunks of canonical basis vectors fuses into one
    large transpose that spills registers and crashes ptxas (measured
    2026-08-17); ``batch_size=16`` keeps each kernel small and is 1.3-2x
    faster than the fully sequential map.
    """
    batched = jax.jit(jax.vmap(operator_apply))
    out = []
    for start in range(0, size, 16):
        idx = jnp.arange(start, min(start + 16, size))
        basis = jax.nn.one_hot(idx, size, dtype=mrx.DTYPE)
        out.append(batched(basis)[jnp.arange(idx.shape[0]), idx])
    return jnp.concatenate(out)


def _invert_diagonal(diagonal):
    diagonal = jnp.asarray(diagonal, dtype=mrx.DTYPE)
    return jnp.where(diagonal != 0.0, 1.0 / diagonal, 0.0)


def _jacobi_entry(operators, slot: str, k: int, dirichlet: bool):
    table = getattr(_require_bundle(operators), slot) or {}
    try:
        return table[(int(k), bool(dirichlet))]
    except KeyError:
        raise ValueError(
            f"{slot} for k={k}, dirichlet={dirichlet} is not on the bundle; "
            "seq.build_preconditioners(jacobi=True) probes it") from None


def _mass_diaginv(seq, operators, k: int, dirichlet: bool):
    """``1/diag(E M_k E^T)`` from the bundle."""
    del seq
    return _jacobi_entry(operators, "mass_jacobi", k, dirichlet)


def _laplacian_diaginv(seq, operators, k: int, dirichlet: bool):
    """``1/diag(L_k)`` from the bundle, ``L_k`` as :func:`apply_laplacian_approx` applies it."""
    del seq
    return _jacobi_entry(operators, "laplacian_jacobi", k, dirichlet)


def _schur_diaginv(seq, operators, k: int, dirichlet: bool):
    """``1/diag(S_k + D B D^T)`` from the bundle, the approximate Schur operator of the saddle solve."""
    del seq
    return _jacobi_entry(operators, "schur_jacobi", k, dirichlet)


def assemble_jacobi_preconditioners(seq, operators: SequenceOperators, *,
                                    ks: Sequence[int] = (0, 1, 2, 3),
                                    dirichlets: Sequence[bool] = (False, True)):
    """Probe the Jacobi diagonals for the given degrees onto the bundle.

    One-hot probes of the applies themselves -- ``O(n_k)`` applies per
    ``(k, dirichlet)`` -- so the stored diagonal is that of the operator as
    it is really applied: ``E M_k E^T``, ``L_k`` through
    :func:`apply_laplacian_approx` (its weak half through the metric-lumped
    mass atom), and for ``k >= 1`` the approximate Schur operator
    ``S_k + D_{k-1} B_{k-1} D_{k-1}^T`` that ``schur.outer='jacobi'``
    preconditions. Needs the metric-lumped mass atoms of the bundle.
    """
    operators = _require_bundle(operators)
    mass, lap, schur = (dict(operators.mass_jacobi or {}), dict(operators.laplacian_jacobi or {}),
                        dict(operators.schur_jacobi or {}))
    schur_spec = SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='metric_lumping'),
        schur=SchurPreconditionerSpec(inner=MassPreconditionerSpec(kind='metric_lumping'),
                                      outer=MassPreconditionerSpec(kind='none')))
    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError("k must be 0, 1, 2 or 3")
        for dirichlet in dirichlets:
            key, n = (int(k), bool(dirichlet)), int(seq.n(k, dirichlet))
            mass[key] = _invert_diagonal(_diagonal_from_matvec(
                lambda x, k=k, d=dirichlet: apply_mass_matrix(seq, x, k, dirichlet=d), n))
            lap[key] = _invert_diagonal(_diagonal_from_matvec(
                lambda x, k=k, d=dirichlet: apply_laplacian_approx(seq, operators, x, k, dirichlet=d), n))
            if k >= 1:
                schur_apply = _build_schur_apply_from_saddle_preconditioner(
                    seq, operators, k=k, dirichlet=dirichlet, eps=0.0,
                    saddle_preconditioner=schur_spec)
                schur[key] = _invert_diagonal(_diagonal_from_matvec(schur_apply, n))
    return eqx.tree_at(lambda o: (o.mass_jacobi, o.laplacian_jacobi, o.schur_jacobi), operators,
                       (mass, lap, schur), is_leaf=lambda x: x is None or isinstance(x, dict))


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


def _resolve_mass_preconditioner(preconditioner):
    if isinstance(preconditioner, str) and preconditioner == 'auto':
        # 'auto' is the production default, always buildable on demand.
        return default_mass_preconditioner()
    return _coerce_mass_preconditioner_spec(preconditioner)


def _build_operator_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
    operator_apply, preconditioner, allow_none: bool = True):
    spec = _resolve_mass_preconditioner(preconditioner)
    valid_kinds = ('none', 'jacobi', 'metric_lumping')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.kind == 'metric_lumping':
        # Separable Kronecker bulk plus a sandwich, with the polar CORE probed
        # and inverted densely. Never splits the space.
        pre = _mass_metric_lumping_for(seq, operators, k, dirichlet)
        return lambda x, pre=pre: pre.apply(x)
    if spec.kind == 'jacobi':
        diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        return lambda x, diaginv=diaginv: diaginv * x
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    raise ValueError(f"unsupported mass preconditioner kind {spec.kind!r}")


def _build_mass_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
    preconditioner, allow_none: bool = True):
    def operator_apply(x):
        return apply_mass_matrix(seq, x, k, dirichlet=dirichlet)

    return _build_operator_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        operator_apply=operator_apply,
        preconditioner=preconditioner,
        allow_none=allow_none,
    )


def _build_schur_operator_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, inner_preconditioner_apply):
    def apply(x):
        d_t_x = apply_derivative_matrix(
            seq, x,
            k - 1,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
            transpose=True,
        )
        inner_d_t_x = inner_preconditioner_apply(d_t_x)
        schur = apply_derivative_matrix(
            seq, inner_d_t_x,
            k - 1,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
        return apply_stiffness(seq, x, k, dirichlet=dirichlet) \
            + eps * apply_mass_matrix(seq, x, k, dirichlet=dirichlet) \
            + schur

    return apply


def _build_schur_apply_from_saddle_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, saddle_preconditioner: SaddlePointPreconditionerSpec):
    schur_inner_spec = saddle_preconditioner.schur.inner
    if schur_inner_spec.kind != 'metric_lumping':
        raise ValueError(
            "schur.inner supports kind='metric_lumping' only "
            f"(got {schur_inner_spec.kind!r})"
        )

    # The weak term B_{k-1} standing in for M_{k-1}^{-1}. Builds on demand, so
    # this cannot fail for want of a prior assemble_*.
    pre = _mass_metric_lumping_for(seq, operators, k - 1, dirichlet)

    def schur_inner(x, pre=pre):
        return pre.apply(x)

    return _build_schur_operator_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        inner_preconditioner_apply=schur_inner,
    )


def _coerce_scalar_hodge_preconditioner(
        seq, operators: SequenceOperators, *, k: int, preconditioner,
        dirichlet: bool = True, eps: float = 0.0):
    """The k=0 Laplacian slot: ``None``/``'auto'`` is the eps-aware default,
    anything else goes through :func:`_coerce_mass_preconditioner_spec`."""
    if preconditioner is None or preconditioner == 'auto':
        return _materialize_default_scalar_hodge_preconditioner(
            seq, operators, k=k, dirichlet=dirichlet, eps=eps)
    return _coerce_mass_preconditioner_spec(preconditioner)


def _coerce_saddle_preconditioner_spec(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        preconditioner) -> SaddlePointPreconditionerSpec:
    if preconditioner is None or preconditioner == 'auto':
        return _materialize_default_saddle_preconditioner(
            seq, operators, k=k, dirichlet=dirichlet)
    if isinstance(preconditioner, SaddlePointPreconditionerSpec):
        # 'metric_lumping' added 2026-08-24. The Schur complement of this saddle system
        # IS L_k = S_k + D M^-1 D^T, which is exactly what the block-Jacobi
        # atom preconditions, so it belongs here -- and MINRES needs its
        # preconditioner SPD, which the atom is (test_preconditioner_is_spd).
        # Until now the only outer option was the per-DoF diagonal, whose weak
        # half is itself a Kronecker mass MODEL, i.e. doubly approximate.
        valid_outer_kinds = ('none', 'jacobi', 'metric_lumping')
        if preconditioner.schur.outer.kind not in valid_outer_kinds:
            raise ValueError(
                "schur.outer kind must be one of "
                f"{valid_outer_kinds} (got {preconditioner.schur.outer.kind!r})"
            )
        return preconditioner
    if isinstance(preconditioner, str):
        valid_outer_kinds = ('none', 'jacobi', 'metric_lumping')
        if preconditioner not in valid_outer_kinds:
            raise ValueError(
                "saddle outer kind must be one of "
                f"{valid_outer_kinds} (got {preconditioner!r})"
            )
        lower = default_mass_preconditioner()
        return SaddlePointPreconditionerSpec(
            mass=lower,
            schur=SchurPreconditionerSpec(
                inner=MassPreconditionerSpec(kind='metric_lumping'),
                outer=MassPreconditionerSpec(kind=preconditioner),
            ),
        )
    raise TypeError(
        'saddle preconditioner must be a kind string or '
        'SaddlePointPreconditionerSpec')


def _build_diffusion_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, preconditioner, allow_none: bool = True):
    spec = _coerce_diffusion_preconditioner_spec(
        seq,
        operators,
        k=k,
        preconditioner=preconditioner,
    )
    # The accept list names exactly what is dispatched below.
    valid_kinds = ('none', 'jacobi', 'metric_lumping')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.kind == 'jacobi':
        # The shifted diagonal 1 / (diag(M) + eps diag(L)): valid for every eps.
        mass_diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        stiffness_diaginv = _laplacian_diaginv(seq, operators, k, dirichlet)
        shifted = 1.0 / (1.0 / mass_diaginv + eps / stiffness_diaginv)
        return lambda x, d=shifted: d * x
    if spec.kind == 'metric_lumping':
        # The shifted-stiffness atom: the strong-half (primal-axis) Kronecker
        # terms of the metric-lumped Laplacian atom, divided by
        # 1 + eps lambda in their eigenbasis, i.e. (M^ + eps S^)^-1 for the
        # atom's own separable mass, plus the dense (M + eps S)^-1 on the
        # core rows. Measured on li383 p=3 at the smoothing eps against the
        # mass atom (CG iterations on M_k + eps S_k): k=2 69 vs 153 at
        # (8,16,8), 117 vs 371 at (12,24,12); k=1 74 vs 181 and 128 vs 422.
        # Two "consistent" factorisations with the Jacobian in the 1-D masses
        # were measured and lost to this plain shift (~200 at (12,24,12)).
        # Its implied mass is a worse M than the mass atom's, so below
        # eps n_r^2 ~ 0.006 (the resistive step's eta dt) the mass atom
        # wins by up to 2x on ~100 iterations; the smoothing eps is 10x
        # above the crossover. docs/research/shifted_split_2026-09-02.md.
        if not _metric_lumping_available(operators, k, dirichlet):
            raise ValueError(
                f"metric_lumping Laplacian atom not assembled for k={k}, "
                f"dirichlet={dirichlet}; seq.build_preconditioners() builds it")
        return operators.laplacian_lumping[(int(k), bool(dirichlet))].shifted_stiffness_apply(eps)
    raise ValueError(
        f"unsupported diffusion preconditioner kind {spec.kind!r}")


def _build_scalar_hodge_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, preconditioner, allow_none: bool = True):
    spec = _coerce_mass_preconditioner_spec(preconditioner)
    valid_kinds = ('none', 'jacobi', 'metric_lumping')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.kind == 'jacobi':
        # The shifted diagonal 1 / (diag(L) + eps diag(M)).
        stiffness_diaginv = _laplacian_diaginv(seq, operators, k, dirichlet)
        if eps == 0.0:
            return lambda x, d=stiffness_diaginv: d * x
        mass_diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        shifted = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv)
        return lambda x, d=shifted: d * x
    if spec.kind == 'metric_lumping':
        # The atom approximates L_k; on L_k + eps M_k it was measured 6/6 in
        # its favour against the shifted diagonal.
        if not _metric_lumping_available(operators, k, dirichlet):
            raise ValueError(
                f"scalar preconditioner kind='metric_lumping' needs the metric_lumping "
                f"Laplacian atom for k={k}, dirichlet={dirichlet}; "
                "seq.build_preconditioners() builds it")
        return lambda x: apply_laplacian_preconditioner(
            seq, operators, x, k, dirichlet=dirichlet, kind='metric_lumping')
    raise ValueError(f"unsupported scalar preconditioner kind {spec.kind!r}")


def _build_coupled_saddle_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        upper_preconditioner, lower_preconditioner):
    n_upper = seq.n(k, dirichlet)

    def apply(x):
        u = x[:n_upper]
        s = x[n_upper:]
        m_inv_s = lower_preconditioner(s)
        w_u = u + apply_derivative_matrix(
            seq, m_inv_s, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        y_u = upper_preconditioner(w_u)
        d_t_y_u = apply_derivative_matrix(
            seq, y_u, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
        z_s = m_inv_s + lower_preconditioner(d_t_y_u)
        return jnp.concatenate([y_u, z_s])

    return apply


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


def _metric_lumping_available(operators, k: int, dirichlet: bool) -> bool:
    """True iff the metric-lumped Laplacian atom for ``(k, dirichlet)`` is on the bundle."""
    atoms = operators.laplacian_lumping if operators is not None else None
    return bool(atoms) and (int(k), bool(dirichlet)) in atoms


def apply_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                   dirichlet: bool = True,
                                   kind: str = 'auto'):
    """Apply the Laplacian preconditioner of the bundle to ``v``.

    ``kind``: ``'metric_lumping'`` (the metric-lumped atom, k = 0..3, free and
    Dirichlet -- the production preconditioner), ``'jacobi'`` (the probed
    ``1/diag(L_k)``, ``build_preconditioners(jacobi=True)``), ``'none'``
    (identity), or ``'auto'``: the atom when it is on the bundle for this
    ``(k, BC)``, otherwise a warning and the identity.
    """
    if kind not in ('auto', 'none', 'jacobi', 'metric_lumping'):
        raise ValueError(
            f"kind must be 'auto', 'none', 'jacobi' or 'metric_lumping' (got {kind!r})")
    if kind == 'jacobi':
        return _laplacian_diaginv(seq, operators, k, dirichlet) * v
    available = _metric_lumping_available(operators, k, dirichlet)
    if kind == 'auto':
        if not available:
            warnings.warn(
                f"no metric_lumping Laplacian atom for k={k}, dirichlet={dirichlet} "
                "on the bundle; solving unpreconditioned", stacklevel=2)
            return v
        kind = 'metric_lumping'
    if kind == 'none':
        return v
    if not available:
        raise ValueError(
            f"metric_lumping Laplacian preconditioner not assembled for "
            f"k={k}, dirichlet={dirichlet}; seq.build_preconditioners() builds it")
    return operators.laplacian_lumping[(int(k), bool(dirichlet))].apply(v)


def apply_inverse_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                  dirichlet: bool = True, guess=None,
                                  tol: Optional[float] = None,
                                  maxiter: Optional[int] = None,
                                  preconditioner='auto',
                                  return_info: bool = False):
    """Solve with the inverse of the unshifted Hodge Laplacian ``L_k``.

    For ``k = 0`` this uses the dedicated singular scalar-Laplacian solve
    directly rather than routing through the shifted ``eps = 0`` path.
    For ``k >= 1`` the saddle-point implementation remains shared with the
    shifted solve because the only difference is the absent mass shift.
    """
    operators = _require_bundle(operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if k == 0:
        # The UNSHIFTED path: this is L_0 itself, so eps = 0 and the block atom
        # is admissible (it approximates L_k, not L_k + eps M_k).
        selected_preconditioner = _coerce_scalar_hodge_preconditioner(
            seq, operators, k=k, preconditioner=preconditioner,
            dirichlet=dirichlet, eps=0.0)

        precond_upper = _build_scalar_hodge_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=0.0,
            preconditioner=selected_preconditioner,
            allow_none=True,
        )

        vs = _nullspace_vectors(operators, 0, dirichlet)
        u, info = solve_singular_cg(
            lambda x: apply_stiffness(
                seq, x, 0, dirichlet=dirichlet),
            rhs,
            mass_matvec=lambda x: apply_mass_matrix(
                seq, x, 0, dirichlet=dirichlet),
            precond_matvec=precond_upper,
            x0=guess,
            vs=vs,
            tol=tol,
            maxiter=maxiter,
        )
        return (u, info) if return_info else u

    return apply_inverse_shifted_laplacian(
        seq,
        operators,
        rhs,
        k,
        0.0,
        dirichlet=dirichlet,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        return_info=return_info,
    )


def apply_inverse_shifted_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                          eps: float, dirichlet: bool = True, guess=None,
                                          tol: Optional[float] = None,
                                          maxiter: Optional[int] = None,
                                          preconditioner='auto',
                                          return_info: bool = False):
    """Solve with the inverse of the shifted Hodge Laplacian ``L_k + eps M_k``.

    For ``k >= 1`` the interface is ``preconditioner``, a structured
    saddle-point preconditioner spec with a lower mass block, a Schur-inner
    mass inverse, a Schur-outer preconditioner, and an optional coupled
    completion. Kind strings are accepted as convenience shorthands.
    """
    operators = _require_bundle(operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if k == 0:
        selected_preconditioner = _coerce_scalar_hodge_preconditioner(
            seq, operators, k=k, preconditioner=preconditioner,
            dirichlet=dirichlet, eps=eps)

        precond_upper = _build_scalar_hodge_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            preconditioner=selected_preconditioner,
            allow_none=True,
        )

        vs = _nullspace_vectors(
            operators, 0, dirichlet) if eps == 0 else jnp.zeros((0, rhs.shape[0]))
        u, info = solve_singular_cg(
            lambda x: apply_stiffness(
                seq, x, 0, dirichlet=dirichlet)
            + eps * apply_mass_matrix(seq, x, 0, dirichlet=dirichlet),
            rhs,
            mass_matvec=(
                lambda x: apply_mass_matrix(
                    seq, x, 0, dirichlet=dirichlet)
            ) if eps == 0 else None,
            precond_matvec=precond_upper,
            x0=guess,
            vs=vs,
            tol=tol,
            maxiter=maxiter,
        )
        return (u, info) if return_info else u

    vs_upper, vs_lower = _saddle_nullspaces(
        seq, operators, k, dirichlet) if eps == 0 else (
            jnp.zeros((0, rhs.shape[0])), jnp.zeros((0, 0)))
    n_upper = seq.n(k, dirichlet)
    n_lower = seq.n(k-1, dirichlet)
    saddle_preconditioner = _coerce_saddle_preconditioner_spec(
        seq, operators, k=k, dirichlet=dirichlet, preconditioner=preconditioner)

    if saddle_preconditioner.schur.inner.kind == 'none':
        raise ValueError("schur.inner cannot use kind='none'")
    precond_lower = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k - 1,
        dirichlet=dirichlet,
        preconditioner=saddle_preconditioner.mass,
        allow_none=True,
    )
    # The Schur apply is built only in the `else` branch (outer='none'): with
    # outer='metric_lumping' the atom IS the upper-block inverse and with
    # outer='jacobi' the probed Schur diagonal is on the bundle; neither needs
    # the Schur operator or schur.inner here.
    outer_spec = saddle_preconditioner.schur.outer
    if outer_spec.kind == 'metric_lumping':
        if not _metric_lumping_available(operators, k, dirichlet):
            raise ValueError(
                "schur.outer kind='metric_lumping' needs the metric_lumping Laplacian "
                f"atom for k={k}, dirichlet={dirichlet}; seq.build_preconditioners() "
                "builds it")

        def precond_upper(x, _k=k, _d=dirichlet):
            return apply_laplacian_preconditioner(
                seq, operators, x, _k, dirichlet=_d, kind='metric_lumping')
    elif outer_spec.kind == 'jacobi':
        schur_diaginv = _schur_diaginv(seq, operators, k, dirichlet)

        def precond_upper(x, d=schur_diaginv):
            return d * x
    else:
        schur_apply = _build_schur_apply_from_saddle_preconditioner(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            saddle_preconditioner=saddle_preconditioner,
        )
        precond_upper = _build_operator_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            operator_apply=schur_apply,
            preconditioner=outer_spec,
            allow_none=True,
        )
    precond_matvec = (
        _build_coupled_saddle_preconditioner(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            upper_preconditioner=precond_upper,
            lower_preconditioner=precond_lower,
        )
        if saddle_preconditioner.coupled
        else None
    )

    u, sigma, info = solve_saddle_point_minres(
        stiffness_matvec=lambda x: apply_stiffness(
            seq, x, k, dirichlet=dirichlet)
        + eps * apply_mass_matrix(seq, x, k, dirichlet=dirichlet),
        derivative_matvec=lambda s: apply_derivative_matrix(
            seq, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: apply_derivative_matrix(
            seq, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: apply_mass_matrix(
            seq, s, k - 1, dirichlet=dirichlet),
        b_upper=rhs,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_matvec=precond_matvec,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        mass_upper_matvec=lambda x: apply_mass_matrix(
            seq, x, k, dirichlet=dirichlet),
        vs_upper=vs_upper,
        vs_lower=vs_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (u, info) if return_info else u


def apply_inverse_mass_plus_eps_laplace_matrix(seq, operators: SequenceOperators, rhs, k: int,
                                               eps: float, dirichlet: bool = True, guess=None,
                                               tol: Optional[float] = None,
                                               maxiter: Optional[int] = None,
                                               preconditioner='auto',
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
    alone; ``k = 3`` has ``S_3 = 0``. ``'auto'`` is the shifted-stiffness
    atom of each level (:meth:`~mrx.metric_lumping_laplacian.
    MetricLumpingLaplacian.shifted_stiffness_apply`), ``(M^ + eps S^)^-1``
    from the Laplacian atom already on the bundle.

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

    def shifted_solve(j, b, x0):
        precond_apply = _build_diffusion_preconditioner_apply(
            seq,
            operators,
            k=j,
            dirichlet=dirichlet,
            eps=eps,
            preconditioner=preconditioner,
            allow_none=True,
        )
        return solve_singular_cg(
            lambda x: apply_mass_matrix(seq, x, j, dirichlet=dirichlet)
            + eps * apply_stiffness(seq, x, j, dirichlet=dirichlet),
            b,
            jnp.zeros((0, b.shape[0]), dtype=b.dtype),
            precond_matvec=precond_apply,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
        )

    x, info = shifted_solve(k, rhs, guess)
    if k == 0:
        return (x, info) if return_info else x

    z, info_lower = shifted_solve(
        k - 1,
        apply_incidence_matrix(seq, rhs, k - 1, dirichlet, dirichlet, transpose=True),
        None)
    x = x - eps * apply_incidence_matrix(seq, z, k - 1, dirichlet, dirichlet)
    total = jnp.abs(info) + jnp.abs(info_lower)
    info = jnp.where((info <= 0) & (info_lower <= 0), -total, total)
    return (x, info) if return_info else x


def apply_laplacian(seq, operators: SequenceOperators, v, k: int,
                          dirichlet: bool = True, guess=None,
                          tol: Optional[float] = None,
                          maxiter: Optional[int] = None):
    """Apply the Hodge Laplacian using explicit operator data.

    This uses bundled mass, weak derivative, and stiffness operators.
    """
    match k:
        case 0:
            return apply_stiffness(seq, v, 0, dirichlet=dirichlet)
        case 1:
            Dt_v = apply_derivative_matrix(
                seq, v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 0, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, Minv_Dt_v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 1, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, Minv_Dt_v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 2, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_derivative_matrix(
                seq, Minv_Dt_v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")


def apply_laplacian_approx(seq, operators: SequenceOperators, v, k: int,
                                 dirichlet: bool = True):
    """Linear approximation of the Hodge Laplacian apply.

    Replaces the exact ``M_{k-1}^{-1}`` in the Schur term of ``L_k`` with one
    apply of the configured mass preconditioner. The result is a fully linear SPD
    matvec: safe to nest inside Krylov iterations and to use as a
    preconditioner or a diagnostic ``L_k``-apply.  It is not exactly
    ``L_k`` unless the metric is tensor-separable on the reference domain.
    """
    match k:
        case 0:
            return apply_stiffness(seq, v, 0, dirichlet=dirichlet)
        case 1:
            Dt_v = apply_derivative_matrix(
                seq, v, 0,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 0, dirichlet=dirichlet, kind='auto')
            return apply_stiffness(seq, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, Minv_Dt_v, 0,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, v, 1,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 1, dirichlet=dirichlet, kind='auto')
            return apply_stiffness(seq, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, Minv_Dt_v, 1,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 2, dirichlet=dirichlet, kind='auto')
            return apply_derivative_matrix(
                seq, Minv_Dt_v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")


# ---------------------------------------------------------------------------
