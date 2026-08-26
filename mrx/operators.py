"""Matrix-free operator bundle (:class:`SequenceOperators`), assembly of its fields, and the operator applies and solves."""
from __future__ import annotations

from typing import Optional, Sequence
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from mrx.extraction_operators import MatrixFreeExtraction, get_xi
import numpy as np

from mrx.local_assembly import (build_matrixfree_mass_apply,
                                build_matrixfree_projection_apply)
from mrx.preconditioners import (
    BoundaryConditionPair,
    MassPreconditioners,
    MassPreconditionerSpec,
    SchurPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    _bulk_tensor_shape,
    # Re-export, not a local use: two debug scripts import _core_size FROM
    # mrx.operators (mrx/experimental/k0_core_schur.py did too until it was
    # deleted with the tensor-Hodge path it was built on). `ruff --fix` will
    # happily delete this as F401 and break them at import time.
    _core_size,  # noqa: F401
    _symmetrize,
    default_mass_preconditioner,
    get_mass_jacobi_diaginv,
    set_mass_jacobi_pair,
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


def _shifted_harmonic_coarse_vector(
        seq, operators: SequenceOperators, k: int, dirichlet: bool):
    """Return the stored M_k-normalised coarse vector for shifted solves."""
    n_dof = getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")
    vs = _nullspace_vectors(operators, k, dirichlet)
    if vs.shape[0] == 0:
        return jnp.zeros(n_dof)
    stored = vs[0]
    stored_norm = seq.l2_norm(stored, k, dirichlet=dirichlet)
    return stored / jnp.where(stored_norm > 0, stored_norm, 1.0)


def _shifted_harmonic_coarse_ready(
        seq, operators: SequenceOperators, k: int, dirichlet: bool) -> bool:
    """True iff a nonzero stored harmonic coarse vector is available."""
    vs = _nullspace_vectors(operators, k, dirichlet)
    if vs.shape[0] == 0:
        return jnp.asarray(False)
    stored = vs[0]
    stored_norm = seq.l2_norm(stored, k, dirichlet=dirichlet)
    return stored_norm > 0


def _wrap_shifted_harmonic_coarse_correction(
        seq, operators: SequenceOperators, base_precond, eps: float,
        k: int, dirichlet: bool):
    """Add an exact ``1/eps`` coarse correction on the stored harmonic mode."""
    z = _shifted_harmonic_coarse_vector(seq, operators, k, dirichlet)
    mz = apply_mass_matrix(seq, operators, z, k, dirichlet=dirichlet)

    def precond(x):
        alpha = z @ x
        x_perp = x - alpha * mz
        y_perp = base_precond(x_perp)
        beta = z @ apply_mass_matrix(
            seq, operators, y_perp, k, dirichlet=dirichlet)
        return y_perp - beta * z + (alpha / eps) * z

    return precond


class SequenceOperators(eqx.Module):
    """Dynamic operator bundle for a de Rham sequence.

    Stores geometry-dependent operator data explicitly so it can be carried
    through JAX transforms while the sequence object remains a static topology
    shell.
    """

    k0_tensor_hodge_precond: Optional[BoundaryConditionPair] = None
    k1_tensor_stiff_model: Optional[K1TensorCurlCurlForwardModel] = None
    k2_tensor_stiff_model: Optional[K2TensorDivDivForwardModel] = None
    k1_tensor_stiff_precond: Optional[BoundaryConditionPair] = None
    k2_tensor_stiff_precond: Optional[BoundaryConditionPair] = None
    e0: Optional[MatrixFreeExtraction] = None
    e0_T: Optional[MatrixFreeExtraction] = None
    e0_dbc: Optional[MatrixFreeExtraction] = None
    e0_dbc_T: Optional[MatrixFreeExtraction] = None
    e0_bc: Optional[MatrixFreeExtraction] = None
    e0_bc_T: Optional[MatrixFreeExtraction] = None
    e1: Optional[MatrixFreeExtraction] = None
    e1_T: Optional[MatrixFreeExtraction] = None
    e1_dbc: Optional[MatrixFreeExtraction] = None
    e1_dbc_T: Optional[MatrixFreeExtraction] = None
    e1_bc: Optional[MatrixFreeExtraction] = None
    e1_bc_T: Optional[MatrixFreeExtraction] = None
    e2: Optional[MatrixFreeExtraction] = None
    e2_T: Optional[MatrixFreeExtraction] = None
    e2_dbc: Optional[MatrixFreeExtraction] = None
    e2_dbc_T: Optional[MatrixFreeExtraction] = None
    e2_bc: Optional[MatrixFreeExtraction] = None
    e2_bc_T: Optional[MatrixFreeExtraction] = None
    e3: Optional[MatrixFreeExtraction] = None
    e3_T: Optional[MatrixFreeExtraction] = None
    e3_dbc: Optional[MatrixFreeExtraction] = None
    e3_dbc_T: Optional[MatrixFreeExtraction] = None
    e3_bc: Optional[MatrixFreeExtraction] = None
    e3_bc_T: Optional[MatrixFreeExtraction] = None
    mass_preconds: Optional[MassPreconditioners] = None
    # Topological exterior-derivative incidence matrices on the full
    # pre-extraction DoF grid. Entries are in {-1, 0, +1}; they encode the
    # discrete de Rham complex structure and are geometry-independent. The
    # strong derivatives ``apply_strong_{grad,curl,div}`` multiply by these
    # directly (no mass solve). Stored as :class:`_MatrixFreeIncidence`
    # (difference stencils); no matrix is ever materialised.
    g0: Optional[_MatrixFreeIncidence] = None
    g0_T: Optional[_MatrixFreeIncidence] = None
    g1: Optional[_MatrixFreeIncidence] = None
    g1_T: Optional[_MatrixFreeIncidence] = None
    g2: Optional[_MatrixFreeIncidence] = None
    g2_T: Optional[_MatrixFreeIncidence] = None
    # Analytic inverse-free polar grad G_0 (V0->V1) on extracted DoFs, built
    # from the incidence pattern + polar coefficients xi alone. On the polar
    # axis the extraction is non-unitary, so the raw ``E_out^T sp E_in`` is not
    # the topological d; these stencils are its closed-form correction
    # ``Gram_{k+1}^{-1} (E^T sp E)``. Stored per (dirichlet_in, dirichlet_out)
    # BC pair, forward + transpose, as indexed gather/scatter operators.
    # ``None`` on non-polar sequences -> apply uses the raw incidence path.
    g0_grad_00: Optional[MatrixFreeExtraction] = None
    g0_grad_00_T: Optional[MatrixFreeExtraction] = None
    g0_grad_01: Optional[MatrixFreeExtraction] = None
    g0_grad_01_T: Optional[MatrixFreeExtraction] = None
    g0_grad_10: Optional[MatrixFreeExtraction] = None
    g0_grad_10_T: Optional[MatrixFreeExtraction] = None
    g0_grad_11: Optional[MatrixFreeExtraction] = None
    g0_grad_11_T: Optional[MatrixFreeExtraction] = None
    # Analytic inverse-free polar curl G_1 (V1->V2), same construction one degree
    # up. ``None`` on non-polar -> raw incidence fallback. Div (V2->V3) needs
    # no stencil: the V3 extraction is a 0/1 selection.
    g1_curl_00: Optional[MatrixFreeExtraction] = None
    g1_curl_00_T: Optional[MatrixFreeExtraction] = None
    g1_curl_01: Optional[MatrixFreeExtraction] = None
    g1_curl_01_T: Optional[MatrixFreeExtraction] = None
    g1_curl_10: Optional[MatrixFreeExtraction] = None
    g1_curl_10_T: Optional[MatrixFreeExtraction] = None
    g1_curl_11: Optional[MatrixFreeExtraction] = None
    g1_curl_11_T: Optional[MatrixFreeExtraction] = None
    dd0_diaginv: Optional[object] = None
    dd1_diaginv: Optional[object] = None
    dd2_diaginv: Optional[object] = None
    dd3_diaginv: Optional[object] = None
    dd0_diaginv_dbc: Optional[object] = None
    dd1_diaginv_dbc: Optional[object] = None
    dd2_diaginv_dbc: Optional[object] = None
    dd3_diaginv_dbc: Optional[object] = None

    # Pre-probed diagonal inverses of the approximate Schur operator
    # S_k + D_{k-1} M_tensor^{-1}_{k-1} D_{k-1}^T.  Built at assembly time
    # by assemble_schur_jacobi_preconditioner; used as a cheap multiply in
    # the saddle-point Schur-outer Jacobi preconditioner instead of probing
    # at solve time.
    schur_diaginv_k1: Optional[jnp.ndarray] = None
    schur_diaginv_k1_dbc: Optional[jnp.ndarray] = None
    schur_diaginv_k2: Optional[jnp.ndarray] = None
    schur_diaginv_k2_dbc: Optional[jnp.ndarray] = None
    schur_diaginv_k3: Optional[jnp.ndarray] = None
    schur_diaginv_k3_dbc: Optional[jnp.ndarray] = None
    schur_diaginv_mode_k1: Optional[str] = None
    schur_diaginv_mode_k1_dbc: Optional[str] = None
    schur_diaginv_mode_k2: Optional[str] = None
    schur_diaginv_mode_k2_dbc: Optional[str] = None
    schur_diaginv_mode_k3: Optional[str] = None
    schur_diaginv_mode_k3_dbc: Optional[str] = None

    # Harmonic nullspaces of the k-form Laplacians. Each field, when set, holds
    # a stacked array of shape ``(n_vectors, n_k)`` with one nullspace basis
    # vector per row. Shapes are topology-determined (from the Betti numbers);
    # the DoFs are dynamic and may be overwritten when the geometry changes.
    null_0: Optional[jnp.ndarray] = None
    null_1: Optional[jnp.ndarray] = None
    null_2: Optional[jnp.ndarray] = None
    null_3: Optional[jnp.ndarray] = None
    null_0_dbc: Optional[jnp.ndarray] = None
    null_1_dbc: Optional[jnp.ndarray] = None
    null_2_dbc: Optional[jnp.ndarray] = None
    null_3_dbc: Optional[jnp.ndarray] = None

    def _laplacian_diaginv_field_name(self, k: int, dirichlet: bool) -> str:
        if k not in (0, 1, 2, 3):
            raise ValueError("k must be 0, 1, 2, or 3")
        suffix = "_dbc" if dirichlet else ""
        return f"dd{k}_diaginv{suffix}"

    def get_laplacian_diaginv(self, k: int, dirichlet: bool = True):
        """Return the stored Jacobi inverse diagonal for ``L_k`` if available."""
        return getattr(self, self._laplacian_diaginv_field_name(k, dirichlet))

    def with_laplacian_diaginv(self, k: int, value, dirichlet: bool = True):
        """Return a copy with updated stored Jacobi inverse diagonal for ``L_k``."""
        field_name = self._laplacian_diaginv_field_name(k, dirichlet)
        return eqx.tree_at(
            lambda ops: getattr(ops, field_name),
            self,
            value,
            is_leaf=lambda x: x is None,
        )

    @property
    def laplacian0_diaginv(self):
        return self.dd0_diaginv

    @property
    def laplacian1_diaginv(self):
        return self.dd1_diaginv

    @property
    def laplacian2_diaginv(self):
        return self.dd2_diaginv

    @property
    def laplacian3_diaginv(self):
        return self.dd3_diaginv

    @property
    def laplacian0_diaginv_dbc(self):
        return self.dd0_diaginv_dbc

    @property
    def laplacian1_diaginv_dbc(self):
        return self.dd1_diaginv_dbc

    @property
    def laplacian2_diaginv_dbc(self):
        return self.dd2_diaginv_dbc

    @property
    def laplacian3_diaginv_dbc(self):
        return self.dd3_diaginv_dbc


_EXTRACTION_OPERATOR_NAMES = (
    'e0', 'e0_T', 'e0_dbc', 'e0_dbc_T', 'e0_bc', 'e0_bc_T',
    'e1', 'e1_T', 'e1_dbc', 'e1_dbc_T', 'e1_bc', 'e1_bc_T',
    'e2', 'e2_T', 'e2_dbc', 'e2_dbc_T', 'e2_bc', 'e2_bc_T',
    'e3', 'e3_T', 'e3_dbc', 'e3_dbc_T', 'e3_bc', 'e3_bc_T',
)


def _ensure_extraction_operators(seq, operators: Optional[SequenceOperators]):
    if operators is None:
        operators = SequenceOperators()
    current = seq.get_operators() if hasattr(seq, 'get_operators') else None
    if current is not None:
        replacements = {
            name: getattr(current, name)
            for name in _EXTRACTION_OPERATOR_NAMES
            if getattr(operators, name, None) is None and getattr(current, name, None) is not None
        }
        if replacements:
            operators = eqx.tree_at(
                lambda ops: tuple(getattr(ops, name) for name in replacements),
                operators,
                tuple(replacements.values()),
                is_leaf=lambda x: x is None,
            )

    if getattr(operators, 'null_0', None) is None:
        from mrx.nullspace import init_nullspaces
        operators = init_nullspaces(
            seq,
            operators,
            betti_numbers=getattr(seq, 'betti_numbers', None),
        )

    return operators


def _reshape_quadrature_scalar_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    # NOTE: returns (ny, nx, nz) = (theta, r, zeta) -- theta-major, NOT
    # (r, theta, zeta). The flat quad ordering comes from QuadratureRule's
    # meshgrid with default indexing='xy' (axis swap); see the TODO in
    # mrx/quadrature.py. Transpose (1, 0, 2) for (r, theta, zeta) fields.
    return jnp.asarray(values).reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz)


def _reshape_quadrature_matrix_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    # NOTE: (ny, nx, nz, ...) = (theta, r, zeta, ...); see note above.
    field = jnp.asarray(values)
    return field.reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz, *field.shape[1:])


class K2TensorDivDivForwardModel(eqx.Module):
    r_shape: tuple[int, int, int] = eqx.field(static=True)
    theta_shape: tuple[int, int, int] = eqx.field(static=True)
    zeta_shape: tuple[int, int, int] = eqx.field(static=True)
    scalar_shape: tuple[int, int, int] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    g_r: jnp.ndarray
    g_t: jnp.ndarray
    g_z: jnp.ndarray
    mass_r_terms: tuple[jnp.ndarray, ...] = ()
    mass_t_terms: tuple[jnp.ndarray, ...] = ()
    mass_z_terms: tuple[jnp.ndarray, ...] = ()
    component_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    component_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    component_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None


class K1TensorCurlCurlForwardModel(eqx.Module):
    r_shape: tuple[int, int, int] = eqx.field(static=True)
    theta_shape: tuple[int, int, int] = eqx.field(static=True)
    zeta_shape: tuple[int, int, int] = eqx.field(static=True)
    curl_r_shape: tuple[int, int, int] = eqx.field(static=True)
    curl_theta_shape: tuple[int, int, int] = eqx.field(static=True)
    curl_zeta_shape: tuple[int, int, int] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    g_r: jnp.ndarray
    g_t: jnp.ndarray
    g_z: jnp.ndarray
    rr_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    rr_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    rr_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    tt_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    tt_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    tt_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    zz_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    zz_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    zz_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _assemble_unweighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return _symmetrize(_assemble_weighted_1d_mass(B, weights))


def _assemble_weighted_1d_stiffness(
        primal_basis: jnp.ndarray,
        derivative_basis: jnp.ndarray,
        weights: jnp.ndarray,
        incidence: jnp.ndarray) -> jnp.ndarray:
    mass_d = _assemble_weighted_1d_mass(derivative_basis, weights)
    stiffness = incidence.T @ (mass_d @ incidence)
    return _symmetrize(stiffness)


def _restrict_radial_window(raw_matrix: jnp.ndarray, radial_start: int,
                            nr: int) -> jnp.ndarray:
    radial_stop = radial_start + nr
    return raw_matrix[radial_start:radial_stop, radial_start:radial_stop]


def _assemble_dense_from_apply(apply, size: int, *, sequential: bool = False) -> jnp.ndarray:
    def column(j):
        return apply(jnp.zeros(size, dtype=mrx.DTYPE).at[j].set(1.0))

    if sequential:
        # Probe a few columns at a time. ``vmap`` batches every transient of
        # the probed apply by ``size``; for a matrix-free (sum-factorized)
        # operator the per-apply transient is a dense ``(ne, q, q, q)`` tensor,
        # so the fully batched peak is ``size``x larger and overflows device
        # memory at high resolution. ``lax.map`` with a small batch keeps the
        # peak to a few applies at the cost of a mostly serial build.
        #
        # One eager warmup call first: ``apply`` may build host-side static
        # state lazily (e.g. the matrix-free mass index plan, which converts
        # basis arrays via ``np.asarray``). Under ``lax.map`` the body is traced
        # as a ``scan``, so any such build would see tracers and raise
        # ``TracerArrayConversionError``. The eager call builds and caches that
        # state with concrete arrays, after which ``lax.map`` only re-invokes
        # the already-jitted apply.
        column(0)
        cols = jax.lax.map(column, jnp.arange(size), batch_size=16)
        return cols.T
    return jax.vmap(column, out_axes=1)(jnp.arange(size))


def _normalize_cp_term_signs(
        scale: jnp.ndarray,
        factor_theta: jnp.ndarray,
        factor_r: jnp.ndarray,
        factor_z: jnp.ndarray):
    if jnp.mean(factor_theta) < 0:
        factor_theta = -factor_theta
        scale = -scale
    if jnp.mean(factor_r) < 0:
        factor_r = -factor_r
        scale = -scale
    if jnp.mean(factor_z) < 0:
        factor_z = -factor_z
        scale = -scale
    if scale < 0:
        factor_r = -factor_r
        scale = -scale
    return scale, factor_theta, factor_r, factor_z


def _k0_bundled_axis_profiles(seq):
    """Per-axis quad-point profiles of the BUNDLED weight g^{aa} J: the
    quad-weighted mean over the other two axes. Bundling keeps the g-J
    correlation inside the average (g^tt J ~ 1/r instead of the divergent
    bare g^tt ~ 1/r^2), so the radial integration of the angular means only
    needs to skip the polar-surgery element [0, xi_1] (core DOFs are handled
    exactly by the Schur envelope)."""
    minv = jnp.transpose(
        _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl),
        (1, 0, 2, 3, 4))
    jacq = jnp.transpose(
        _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j), (1, 0, 2))
    w00, w11, w22 = (minv[..., a, a] * jacq for a in range(3))
    xi1 = jnp.asarray(seq.basis_0.Λ[0].T)[seq.ps[0] + 1]
    wx_cut = seq.quad.w_x * (jnp.asarray(seq.quad.x_x) >= xi1)
    wy, wz = seq.quad.w_y, seq.quad.w_z
    sy, sz, sxc = jnp.sum(wy), jnp.sum(wz), jnp.sum(wx_cut)
    pr = jnp.einsum('qrs,r,s->q', w00, wy, wz) / (sy * sz)
    pt = jnp.einsum('qrs,q,s->r', w11, wx_cut, wz) / (sxc * sz)
    pz = jnp.einsum('qrs,q,r->s', w22, wx_cut, wy) / (sxc * sy)

    def clip(v):
        return jnp.maximum(v, mrx.sqrt_eps(0.67) * jnp.abs(jnp.median(v)))

    return clip(pr), clip(pt), clip(pz)


def _assemble_k0_greville_bulk_factors(seq, *, dirichlet: bool):
    """k=0 stiffness bulk factors: the "fd" atom (exact additive FD inverse).

    Per-axis 1D stiffnesses WEIGHTED by the bundled profiles <g^{aa} J> of
    the other two axes, D = 1, alpha = 1 (adopted 2026-08-13, see
    docs/research/handoff_2026-08-13_gpu_cluster.md). Keeping the g-J
    correlation inside the per-axis averages beat the pre-2026-08
    collocated variant on every geometry tested (W7-X 16,32,32: dbc
    80->62, free 117->85 CG its at equal ms/it). The collocated atom was
    DELETED 2026-08-14 when the core-Schur rebuild switched to exact bulk
    solves -- its last remaining role was the one-sided Schur probe (see
    _assemble_k0_tensor_hodge_preconditioner).
    """
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk, nt, nz = (int(s) for s in bulk_shape)
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    pr, pt, pz = _k0_bundled_axis_profiles(seq)
    kw_x, kw_y, kw_z = seq.quad.w_x * pr, seq.quad.w_y * pt, seq.quad.w_z * pz

    M0_r = _restrict_radial_window(_assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), 2, nr_bulk)
    M0_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M0_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    K0_r = _restrict_radial_window(_assemble_weighted_1d_stiffness(seq.basis_r_jk, seq.d_basis_r_jk, kw_x, g_r), 2, nr_bulk)
    K0_t = _assemble_weighted_1d_stiffness(seq.basis_t_jk, seq.d_basis_t_jk, kw_y, g_t)
    K0_z = _assemble_weighted_1d_stiffness(seq.basis_z_jk, seq.d_basis_z_jk, kw_z, g_z)
    V_r, lam_r = _assemble_1d_fd_eigendecomp(M0_r, K0_r)
    V_t, lam_t = _assemble_1d_fd_eigendecomp(M0_t, K0_t)
    V_z, lam_z = _assemble_1d_fd_eigendecomp(M0_z, K0_z)

    # J lives inside the bundled 1D weights; no collocated diagonal.
    return {
        "bulk_shape": bulk_shape,
        "bulk_V_r": V_r, "bulk_V_t": V_t, "bulk_V_z": V_z,
        "bulk_lam_r": lam_r, "bulk_lam_t": lam_t, "bulk_lam_z": lam_z,
        "bulk_alpha": jnp.ones((3,), dtype=mrx.DTYPE),
        "bulk_greville_inv_sqrt_D": jnp.ones(bulk_shape, dtype=mrx.DTYPE),
    }


def assemble_mass_jacobi_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble/store Jacobi mass diagonals eagerly for requested degrees.

    Probes ``diag(E M_k E^T)`` through the SAME matrix-free extracted mass
    apply the solvers use (via :func:`mrx.preconditioners.build_mass_jacobi_pair`),
    so the stored diagonal is consistent with the runtime operator by
    construction. The previous sparse-block route (a scatter plan over the
    assembled mass entries) produced inconsistent diagonals on the polar
    extracted spaces k=0,1,2 (CG with the "preconditioner" stalled at
    maxiter; caught 2026-08-14 when the test suite first exercised it).
    """
    from mrx.preconditioners import build_mass_jacobi_pair  # noqa: PLC0415

    operators = _ensure_extraction_operators(seq, operators)
    preconds = operators.mass_preconds

    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError("Mass Jacobi assembly only supports k=0,1,2,3")

        mass_apply = build_matrixfree_mass_apply(seq, k)
        pair = build_mass_jacobi_pair(seq, mass_apply, k)
        preconds = set_mass_jacobi_pair(preconds, k, pair)

    return eqx.tree_at(
        lambda ops: ops.mass_preconds,
        operators,
        preconds,
        is_leaf=lambda x: x is None,
    )


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

    ``schur.inner`` is metric_lumping. It was raw_kron until 2026-08-25,
    justified here by "the Schur operator is still built before the branch, so
    the factors are needed" -- which stopped being true at 31ef58f, when the
    Schur apply was moved into the only branch that consumes it. Under
    ``outer='metric_lumping'`` the atom IS the upper-block inverse and the inner slot
    does no work at all.
    """
    outer = ('metric_lumping' if _metric_lumping_available(seq, k, dirichlet) else 'none')
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
    del operators
    if eps != 0.0:
        return MassPreconditionerSpec(kind='none')
    if _metric_lumping_available(seq, k, dirichlet):
        return MassPreconditionerSpec(kind='metric_lumping')
    return MassPreconditionerSpec(kind='none')


def _coerce_diffusion_preconditioner_spec(
        seq, operators: SequenceOperators, *, k: int, preconditioner):
    if preconditioner is None or preconditioner == 'auto':
        # Audit item 3.1, resolved 2026-08-25.  This used to return plain
        # 'jacobi' -- diag(M)^-1, with eps ignored entirely -- on the grounds
        # that "the block atom approximates L_k, not L_k + eps M_k, so it is
        # not a drop-in here".
        #
        # That reasoning is correct about the LAPLACIAN atom and incomplete
        # about the operator.  M + eps L is MASS-DOMINATED in the regime this
        # solve is actually used: the relaxation's hyperregularisation runs at
        # mu = 1e-3, where eps * lambda_max(M^-1 L) ~ 0.26 at ns=(8,16,8)
        # (lambda_max ~ n_theta^2).  The MASS preconditioner approximates the
        # term that dominates, and it is the production one -- measured 0.70
        # to 0.83x raw_kron's iterations at k=1,2.  So the right object was
        # available the whole time; it just was not the one the comment
        # considered.
        #
        # 'jacobi' remains available and is now the correctly SHIFTED diagonal
        # (see _build_diffusion_preconditioner_apply), which is the robust
        # choice when eps * lambda_max is not small.
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


def _assemble_1d_fd_eigendecomp(M: jnp.ndarray, K: jnp.ndarray):
    """Reduce ``K v = λ M v`` to a standard eigenproblem via Cholesky.

    Returns ``(V, lam)`` where ``V`` columns are ``M``-orthonormal
    eigenvectors and ``lam`` the eigenvalues.
    """
    L = jnp.linalg.cholesky(M)
    # K_tilde = L^{-1} K L^{-T}
    Y = jsp.linalg.solve_triangular(L, K, lower=True)
    K_tilde = jsp.linalg.solve_triangular(L, Y.T, lower=True).T
    K_tilde = 0.5 * (K_tilde + K_tilde.T)
    lam, W = jnp.linalg.eigh(K_tilde)
    # V = L^{-T} W satisfies V^T M V = I and K V = M V diag(lam).
    V = jsp.linalg.solve_triangular(L.T, W, lower=False)
    return V, lam


# TODO: remove (deprecated no-op shim)
def _fd_apply_3d(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, x, eps: float = 0.0):
    """Apply ``(L + eps M)^{-1}`` via fast diagonalisation on a 3-tensor ``x``."""
    # Forward transform: y = V^T x (in all three axes).
    y = jnp.einsum('ji,jkl->ikl', V_r, x)
    y = jnp.einsum('ji,kjl->kil', V_t, y)
    y = jnp.einsum('ji,klj->kli', V_z, y)
    # Diagonal solve in the eigenbasis.
    denom = (alpha[0] * lam_r[:, None, None]
             + alpha[1] * lam_t[None, :, None]
             + alpha[2] * lam_z[None, None, :]) + eps
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
    # Back transform: x_out = V y (in all three axes).
    y = jnp.einsum('ij,jkl->ikl', V_r, y)
    y = jnp.einsum('ij,kjl->kil', V_t, y)
    y = jnp.einsum('ij,klj->kli', V_z, y)
    return y


def mass_core_apply(seq, operators: SequenceOperators, k: int):
    """Return a raw-DOF-space callable ``x -> M_k @ x``.

    The returned callable acts in the unextracted tensor-product DOF space and
    is evaluated matrix-free: the sum-factorized kernel never materializes
    ``M_k``, removing the high-(n, p) storage bottleneck (notably for M1). The
    element plan is built once per geometry and cached on ``seq``.
    """
    del operators  # the apply is built from the geometry attached to seq
    return _matrixfree_mass_apply_cached(seq, k)


def _matrixfree_mass_apply_cached(seq, k: int):
    """Build (and cache on ``seq``) the matrix-free ``M_k`` apply.

    The element plan inside :func:`build_matrixfree_mass_apply` is host-built
    and reused across matvecs, so it must be constructed once rather than per
    apply. The cache is keyed by the current geometry object so that re-mapping
    the sequence (``set_map``) transparently rebuilds the plan.
    """
    geometry = seq.geometry
    cache = getattr(seq, "_matrixfree_mass_apply_cache", None)
    if cache is None:
        cache = {}
        seq._matrixfree_mass_apply_cache = cache
    entry = cache.get(k)
    if entry is not None and entry[0] is geometry:
        return entry[1]
    apply = build_matrixfree_mass_apply(seq, k, geometry)
    cache[k] = (geometry, apply)
    return apply


def _mass_diaginv(seq, operators: SequenceOperators, k: int, dirichlet: bool):
    del seq
    try:
        selected = get_mass_jacobi_diaginv(operators.mass_preconds, k, dirichlet)
    except ValueError:
        selected = None
    if selected is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(
            f"Jacobi mass diagonal for k={k} ({side}) is not assembled. "
            "Call assemble_mass_jacobi_preconditioner(...) during operator assembly."
        )
    return selected


def _laplacian_diaginv(seq, operators: SequenceOperators, k: int, dirichlet: bool):
    selected = operators.get_laplacian_diaginv(k, dirichlet)
    if selected is None:
        if k in (0, 1, 2, 3):
            # The Laplacian Jacobi diagonal is not assembled eagerly: with the
            # incidence ``G_k`` matrix-free and the mass ``M_{k+1}`` stored
            # nowhere, the direct entry-scatter form is unavailable. It is
            # built here in closed form instead -- no probe at any k. Returns
            # the inverse diagonal.
            if k == 0:
                # L_0 = S_0 (no lower term), and diag(E S_0 E^T) is the energy
                # of the extracted basis functions -- closed form, O(N), no
                # applies at all. Verified against this probe to <1e-15.
                from mrx.local_assembly import (  # noqa: PLC0415
                    build_extracted_stiffness_diagonal_k0)
                selected = _invert_diagonal(
                    build_extracted_stiffness_diagonal_k0(seq, dirichlet))
            elif os.environ.get("MRX_LAPLACIAN_DIAG_PROBE", "0") == "1":
                # A/B escape hatch: the exact but O(N)-applies probe that the
                # closed form below replaced.
                suffix = "_dbc" if dirichlet else ""
                size = int(getattr(seq, f"n{k}{suffix}"))
                diag = _diagonal_from_matvec(
                    lambda x: apply_hodge_laplacian_approx(
                        seq, operators, x, k, dirichlet=dirichlet),
                    size,
                )
                selected = _invert_diagonal(diag)
            else:
                # k>=1 also carries the weak term D B D^T. Both halves are
                # closed form in the raw DOF space and only the O(n_polar n_z)
                # coupled rows need an apply -- see
                # ``build_extracted_laplacian_diagonal``.
                from mrx.preconditioners import (  # noqa: PLC0415
                    build_extracted_laplacian_diagonal)
                selected = _invert_diagonal(
                    build_extracted_laplacian_diagonal(
                        seq, operators, k, dirichlet=dirichlet))
        else:
            raise ValueError(f"Laplacian preconditioner k={k} is not assembled")
    return selected


def update_derivative_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Ensure the k-th incidence ``G_k`` is present (``D_k = M_{k+1} G_k`` is applied lazily)."""
    del geometry  # unused
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    operators = _ensure_extraction_operators(seq, operators)
    if _incidence_components(operators, k)[0] is None:
        operators = update_incidence_operator(seq, operators, k)
    return operators


def assemble_derivative_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                                  ks: Sequence[int] = (0, 1, 2)):
    """Assemble weak derivative operators for the requested form degrees."""
    for k in ks:
        operators = update_derivative_operator(seq, geometry, operators, k)
    return operators


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


def _build_matrixfree_incidence(seq, k: int):
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


def update_incidence_operator(seq, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th topological incidence updated."""
    sp, sp_T = _build_matrixfree_incidence(seq, k)
    operators = _ensure_extraction_operators(seq, operators)

    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.g0, ops.g0_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.g1, ops.g1_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.g2, ops.g2_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1 or 2")


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

    n0 = int(seq.n0_dbc if dirichlet_in else seq.n0)
    n1 = int(seq.n1_dbc if dirichlet_out else seq.n1)
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

    n1 = int(seq.n1_dbc if dirichlet_in else seq.n1)
    n2 = int(seq.n2_dbc if dirichlet_out else seq.n2)
    return out.operator((n2, n1))


def _grad_stencil(operators: SequenceOperators, dirichlet_in: bool,
                  dirichlet_out: bool, transpose: bool):
    """Look up the analytic inverse-free polar grad ``G_0`` (or None on non-polar)."""
    name = f"g0_grad_{int(dirichlet_in)}{int(dirichlet_out)}"
    if transpose:
        name += "_T"
    return getattr(operators, name, None)


def _curl_stencil(operators: SequenceOperators, dirichlet_in: bool,
                  dirichlet_out: bool, transpose: bool):
    """Look up the analytic inverse-free polar curl ``G_1`` (or None on non-polar)."""
    name = f"g1_curl_{int(dirichlet_in)}{int(dirichlet_out)}"
    if transpose:
        name += "_T"
    return getattr(operators, name, None)


def _extraction_is_polar(operators: SequenceOperators, space: int) -> bool:
    """True iff the extraction of ``space`` is non-unitary (polar axis fusion).

    Tests ``E E^T x != x`` on one probe (``E E^T = I`` on the 0/1
    non-polar/unitary extractions).
    """
    e, e_T = _mass_extraction(operators, space, False)
    if e is None or e_T is None:
        return False
    n_ext = int(e.shape[0])
    x = jax.random.normal(jax.random.PRNGKey(0), (n_ext,), dtype=mrx.DTYPE)
    return bool(jnp.max(jnp.abs(e @ (e_T @ x) - x)) > mrx.sqrt_eps(1e-2))


def assemble_incidence_operators(seq, operators: Optional[SequenceOperators] = None,
                                 ks: Sequence[int] = (0, 1, 2)):
    """Assemble topological incidence operators for the requested degrees.

    On polar sequences also builds the analytic grad/curl stencils so
    :func:`apply_incidence_matrix` returns the true strong derivative (exact
    ``d.d = 0`` on extracted DoFs); elsewhere the raw incidence is exact.
    """
    for k in ks:
        operators = update_incidence_operator(seq, operators, k)
    operators = _ensure_extraction_operators(seq, operators)

    # Analytic inverse-free polar grad G_0, built when grad is requested (0 in
    # ks) on a polar sequence (V1 extraction non-unitary). Stored per BC pair,
    # forward + transpose.
    polar = _extraction_is_polar(operators, 1)
    # The analytic grad stencil encodes the C¹ polar surgery structure; a
    # polar_order=2 sequence has a different 0-form layout (6 polar
    # functions, rings 0-2) and its weak-form k=0 pipeline never uses the
    # stencil (apply_stiffness sandwiches the TENSOR incidence) — skip.
    if (getattr(seq, "polar_order", 1) == 1
            and 0 in ks and polar and operators.g0_grad_00 is None):
        xi = get_xi(seq.ns[1])
        gfields, gvals = [], []
        for din in (False, True):
            for dout in (False, True):
                g0 = build_grad_stencil_g0(seq, xi, din, dout)
                base = f"g0_grad_{int(din)}{int(dout)}"
                gfields += [base, base + "_T"]
                gvals += [g0, g0.T]
        operators = eqx.tree_at(
            lambda o: tuple(getattr(o, f) for f in gfields),
            operators, tuple(gvals),
            is_leaf=lambda x: x is None,
        )

    # Analytic inverse-free polar curl G_1, built when curl is requested (1 in
    # ks) on a polar sequence. Div (k=2, output V3) needs no stencil: the V3
    # extraction is a 0/1 selection, so apply_incidence(., 2) is already the
    # true div.
    polar2 = _extraction_is_polar(operators, 2)
    if 1 in ks and polar2 and operators.g1_curl_00 is None:
        xi = get_xi(seq.ns[1])
        cfields, cvals = [], []
        for din in (False, True):
            for dout in (False, True):
                g1 = build_curl_stencil_g1(seq, xi, din, dout)
                base = f"g1_curl_{int(din)}{int(dout)}"
                cfields += [base, base + "_T"]
                cvals += [g1, g1.T]
        operators = eqx.tree_at(
            lambda o: tuple(getattr(o, f) for f in cfields),
            operators, tuple(cvals),
            is_leaf=lambda x: x is None,
        )
    return operators


def _incidence_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.g0, operators.g0_T
        case 1:
            return operators.g1, operators.g1_T
        case 2:
            return operators.g2, operators.g2_T
    raise ValueError("k must be 0, 1 or 2")


def apply_incidence_matrix(seq, operators: SequenceOperators, v, k: int,
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
        g0 = _grad_stencil(operators, dirichlet_in, dirichlet_out, transpose)
        if g0 is not None:
            return g0 @ v
    if k == 1:
        g1 = _curl_stencil(operators, dirichlet_in, dirichlet_out, transpose)
        if g1 is not None:
            return g1 @ v

    sp, sp_T = _incidence_components(operators, k)
    if sp is None or sp_T is None:
        raise ValueError(f"Incidence operator k={k} is not assembled")
    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        operators, k, dirichlet_in, dirichlet_out)
    if transpose:
        return e_in @ (sp_T @ (e_out_T @ v))
    return e_out @ (sp @ (e_in_T @ v))


# Row/column spaces of the projection masses ``P_{k_in k_out}``: rows are the
# space of ``e_out`` and columns the space of ``e_in`` in
# :func:`_projection_extraction`.
_PROJECTION_SPACES = {(2, 1): (1, 2), (1, 2): (2, 1), (0, 3): (0, 3), (3, 0): (3, 0)}


def _matrixfree_projection_apply_cached(seq, k_in: int, k_out: int):
    """Build (and cache on ``seq``) the raw-DOF apply of the projection mass.

    Same memoisation as :func:`_matrixfree_mass_apply_cached`: keyed by the
    pair and the geometry object so a re-mapped sequence rebuilds the plan.
    """
    try:
        k_row, k_col = _PROJECTION_SPACES[(k_in, k_out)]
    except KeyError:
        raise ValueError(
            "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
        ) from None
    geometry = seq.geometry
    cache = getattr(seq, "_matrixfree_projection_apply_cache", None)
    if cache is None:
        cache = {}
        seq._matrixfree_projection_apply_cache = cache
    entry = cache.get((k_row, k_col))
    if entry is not None and entry[0] is geometry:
        return entry[1]
    apply = build_matrixfree_projection_apply(seq, k_row, k_col)
    cache[(k_row, k_col)] = (geometry, apply)
    return apply


def update_hodge_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Ensure the incidence ``G_k`` behind the k-th Laplacian is present.

    Stiffness matrices satisfy ``K_k = G_k^T M_{k+1} G_k`` and are never
    materialised: :func:`apply_stiffness` / :func:`apply_hodge_laplacian`
    compose the matrix-free incidence and mass applies. The Jacobi diagonals
    ``dd{k}_diaginv`` are built lazily by :func:`_laplacian_diaginv`.
    """
    del geometry  # unused
    if k not in (0, 1, 2, 3):
        raise ValueError("k must be 0, 1, 2, or 3")
    operators = _ensure_extraction_operators(seq, operators)
    if k in (0, 1, 2) and _incidence_components(operators, k)[0] is None:
        operators = update_incidence_operator(seq, operators, k)
    return operators


def assemble_hodge_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                             ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble Hodge/stiffness operators for the requested form degrees."""
    for k in ks:
        operators = update_hodge_operator(seq, geometry, operators, k)
    return operators


def assemble_laplacian_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                                 ks: Sequence[int] = (0, 1, 2, 3)):
    """Alias of assemble_hodge_operators using Laplacian naming."""
    return assemble_hodge_operators(seq, geometry, operators=operators, ks=ks)


def assemble_all_operators(seq, geometry,
                           operators: Optional[SequenceOperators] = None):
    """Assemble the incidence operators (and polar stencils) for ``seq``.

    Masses, projections, derivatives and Laplacians are all applied
    matrix-free from the geometry attached to ``seq``, so the incidence is the
    only operator data built here.
    """
    operators = assemble_incidence_operators(seq, operators=operators)
    operators = assemble_derivative_operators(
        seq, geometry, operators=operators)
    operators = assemble_laplacian_operators(seq, geometry, operators=operators)
    return operators


def _mass_extraction(operators: SequenceOperators, k: int, dirichlet: bool):
    match k:
        case 0:
            return (operators.e0_dbc, operators.e0_dbc_T) if dirichlet else (operators.e0, operators.e0_T)
        case 1:
            return (operators.e1_dbc, operators.e1_dbc_T) if dirichlet else (operators.e1, operators.e1_T)
        case 2:
            return (operators.e2_dbc, operators.e2_dbc_T) if dirichlet else (operators.e2, operators.e2_T)
        case 3:
            return (operators.e3_dbc, operators.e3_dbc_T) if dirichlet else (operators.e3, operators.e3_T)
    raise ValueError("k must be 0, 1, 2 or 3")


def _derivative_extraction(operators: SequenceOperators, k: int,
                           dirichlet_in: bool, dirichlet_out: bool):
    match k:
        case 0:
            e_in = operators.e0_dbc if dirichlet_in else operators.e0
            e_in_T = operators.e0_dbc_T if dirichlet_in else operators.e0_T
            e_out = operators.e1_dbc if dirichlet_out else operators.e1
            e_out_T = operators.e1_dbc_T if dirichlet_out else operators.e1_T
        case 1:
            e_in = operators.e1_dbc if dirichlet_in else operators.e1
            e_in_T = operators.e1_dbc_T if dirichlet_in else operators.e1_T
            e_out = operators.e2_dbc if dirichlet_out else operators.e2
            e_out_T = operators.e2_dbc_T if dirichlet_out else operators.e2_T
        case 2:
            e_in = operators.e2_dbc if dirichlet_in else operators.e2
            e_in_T = operators.e2_dbc_T if dirichlet_in else operators.e2_T
            e_out = operators.e3_dbc if dirichlet_out else operators.e3
            e_out_T = operators.e3_dbc_T if dirichlet_out else operators.e3_T
        case _:
            raise ValueError("k must be 0, 1 or 2")
    return e_in, e_in_T, e_out, e_out_T


def _projection_extraction(operators: SequenceOperators,
                           k_in: int, k_out: int,
                           dirichlet_in: bool, dirichlet_out: bool):
    match (k_in, k_out):
        case (2, 1):
            e_in = operators.e2_dbc if dirichlet_in else operators.e2
            e_in_T = operators.e2_dbc_T if dirichlet_in else operators.e2_T
            e_out = operators.e1_dbc if dirichlet_out else operators.e1
        case (1, 2):
            e_in = operators.e1_dbc if dirichlet_in else operators.e1
            e_in_T = operators.e1_dbc_T if dirichlet_in else operators.e1_T
            e_out = operators.e2_dbc if dirichlet_out else operators.e2
        case (0, 3):
            e_in = operators.e3_dbc if dirichlet_in else operators.e3
            e_in_T = operators.e3_dbc_T if dirichlet_in else operators.e3_T
            e_out = operators.e0_dbc if dirichlet_out else operators.e0
        case (3, 0):
            e_in = operators.e0_dbc if dirichlet_in else operators.e0
            e_in_T = operators.e0_dbc_T if dirichlet_in else operators.e0_T
            e_out = operators.e3_dbc if dirichlet_out else operators.e3
        case _:
            raise ValueError(
                "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
            )
    return e_in, e_in_T, e_out


def apply_mass_matrix(seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """Apply a mass matrix from an explicit operator bundle."""
    core = mass_core_apply(seq, operators, k)
    e, e_T = _mass_extraction(operators, k, dirichlet)
    return e @ core(e_T @ v)


def apply_projection_matrix(seq, operators: SequenceOperators, v,
                            k_in: int, k_out: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True):
    """Apply the projection mass ``P_{k_in k_out}`` (matrix-free, memoised on ``seq``)."""
    core = _matrixfree_projection_apply_cached(seq, k_in, k_out)
    e_in, e_in_T, e_out = _projection_extraction(
        operators, k_in, k_out, dirichlet_in, dirichlet_out)
    return e_out @ core(e_in_T @ v)


def apply_derivative_matrix(seq, operators: SequenceOperators, v, k: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True,
                            transpose: bool = False):
    """Apply a weak derivative matrix from an explicit operator bundle.

    ``D_k = M_{k+1} G_k`` is applied as a composition of matrix-free applies;
    the full ``D_k`` is never materialised.
    """
    g_sp, g_sp_T = _incidence_components(operators, k)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply D{k}")
    m_apply = mass_core_apply(seq, operators, k + 1)

    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        operators, k, dirichlet_in, dirichlet_out)

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
        Which preconditioner to use. The retired ``'tensor'`` and
        ``'raw_kron'`` kinds were deleted on 2026-08-25.
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
    :func:`~mrx.preconditioners.default_mass_preconditioner`, i.e. block_jacobi
    since 2026-08-22. (It read "tensor when assembled and Jacobi otherwise"
    until 2026-08-24; that has not been true since the 2026-08-17 pivot.)
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
        lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        rhs,
        mass_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet),
        precond_matvec=precond_apply,
        x0=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (x, info) if return_info else x


def apply_stiffness(seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """Apply a stiffness matrix from an explicit operator bundle.

    ``K_k = G_k^T M_{k+1} G_k`` is applied as a composition of matrix-free
    applies; the full ``K_k`` is never materialised.
    """
    if k == 3:
        return jnp.zeros_like(v)
    g_sp, g_sp_T = _incidence_components(operators, k)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply K{k}")
    m_apply = mass_core_apply(seq, operators, k + 1)

    e, e_T = _mass_extraction(operators, k, dirichlet)
    return e @ (g_sp_T @ m_apply(g_sp @ (e_T @ v)))


def _diagonal_from_matvec(operator_apply, size: int):
    """Probe ``diag(A)`` 16 columns at a time via ``jax.lax.map``.

    DO NOT batch this with a full ``vmap`` over chunks of canonical basis
    vectors, the way :func:`mrx.preconditioners.diag_matvec` does. It was
    tried on 2026-08-17 and crashes the CUDA toolchain: the batched kernel
    fuses into a large transpose that spills registers and ptxas exits with an
    internal compiler error (``ptxas fatal: Internal compiler error``, 94 test
    errors, all inside the ``lax.while_loop`` in
    ``nullspace.find_nullspace_vectors``). ``batch_size=16`` keeps each kernel
    small enough to compile (clean on the nullspace tests 2026-08-26) and is
    1.3-2x faster than the fully sequential map.
    """
    # Eager warmup: operator_apply may lazily build host-side static state
    # (e.g. matrix-free mass index plans that call np.asarray internally).
    # Under lax.map the body is traced as a scan, so those calls would see
    # tracers and raise TracerArrayConversionError.  One concrete call first
    # forces that state to be built and cached before the traced loop runs.
    operator_apply(jnp.zeros(size, dtype=mrx.DTYPE))

    def entry(i):
        basis = jnp.zeros(size, dtype=mrx.DTYPE).at[i].set(1.0)
        return operator_apply(basis)[i]

    return jax.lax.map(entry, jnp.arange(size), batch_size=16)


def _invert_diagonal(diagonal):
    diagonal = jnp.asarray(diagonal, dtype=mrx.DTYPE)
    return jnp.where(diagonal != 0.0, 1.0 / diagonal, 0.0)


def _get_schur_diaginv(operators: SequenceOperators, k: int, dirichlet: bool, mode: str):
    """Return stored Schur diaginv for ``(k, dirichlet, mode)``, or ``None``."""
    suffix = '_dbc' if dirichlet else ''
    diaginv = getattr(operators, f'schur_diaginv_k{k}{suffix}', None)
    mode_stored = getattr(operators, f'schur_diaginv_mode_k{k}{suffix}', None)
    if diaginv is None:
        return None
    if mode_stored == mode:
        return diaginv
    return None


def _set_schur_diaginv(operators: SequenceOperators, k: int, dirichlet: bool, diaginv, mode: str):
    """Return operators with Schur diaginv + mode tag for ``(k, dirichlet)`` updated."""
    suffix = '_dbc' if dirichlet else ''
    field = f'schur_diaginv_k{k}{suffix}'
    mode_field = f'schur_diaginv_mode_k{k}{suffix}'
    return eqx.tree_at(
        lambda ops: (getattr(ops, field), getattr(ops, mode_field)),
        operators,
        (diaginv, mode),
        is_leaf=lambda x: x is None,
    )


_SCHUR_DIAG_MODES = ('metric_lumping_probe',)


def _coerce_schur_diag_mode(spec: MassPreconditionerSpec, *, context: str) -> str:
    mode = spec.schur_diag_mode
    if mode not in _SCHUR_DIAG_MODES:
        raise ValueError(
            f"{context} schur_diag_mode must be one of {_SCHUR_DIAG_MODES} "
            f"(got {mode!r})"
        )
    return mode


def _build_schur_probe_apply(
        seq, operators: SequenceOperators, *,
        k: int, dirichlet: bool, eps: float,
        mode: str,
        saddle_preconditioner: SaddlePointPreconditionerSpec):
    if mode not in _SCHUR_DIAG_MODES:
        raise ValueError(
            f"Unsupported Schur diagonal probe mode {mode!r}; "
            f"expected one of {_SCHUR_DIAG_MODES}"
        )
    return _build_schur_apply_from_saddle_preconditioner(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        saddle_preconditioner=saddle_preconditioner,
    )


def _build_schur_outer_jacobi_diaginv(
        seq, operators: SequenceOperators, *,
        k: int, dirichlet: bool, eps: float,
        outer_spec: MassPreconditionerSpec,
        saddle_preconditioner: SaddlePointPreconditionerSpec,
        allow_stored_tensor_diaginv: bool):
    mode = _coerce_schur_diag_mode(
        outer_spec,
        context=f"schur.outer kind={outer_spec.kind!r}",
    )
    if allow_stored_tensor_diaginv:
        stored_diaginv = _get_schur_diaginv(operators, k, dirichlet, mode)
        if stored_diaginv is not None:
            return stored_diaginv

    probe_apply = _build_schur_probe_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        mode=mode,
        saddle_preconditioner=saddle_preconditioner,
    )
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    diagonal = _diagonal_from_matvec(probe_apply, size)
    return _invert_diagonal(diagonal)


def assemble_schur_jacobi_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (1, 2, 3),
        dirichlet_variants: Optional[Sequence[bool]] = None,
    eps: float = 0.0,
    schur_diag_mode: str = 'metric_lumping_probe') -> SequenceOperators:
    """Probe and store the approximate Schur diagonal at assembly time.

    For each (k, dirichlet) pair, builds the approximate Schur operator

        A_k(x) = S_k x + D_{k-1} B_{k-1} D_{k-1}^T x

    and probes its diagonal by O(n_k) matrix-vector products.  The
    resulting ``1/diag(A_k)`` is stored on the operator bundle so that
    the saddle-point Schur-outer Jacobi preconditioner is a cheap
    multiply at solve time rather than an O(n_k) probing scan.

    Parameters
    ----------
    seq : DeRhamSequence
    operators : SequenceOperators, optional
    ks : sequence of int
        Form degrees to assemble (must be in 1, 2, 3).
    dirichlet_variants : sequence of bool, optional
        Boundary condition variants to assemble.  Defaults to (True, False).
    eps : float
        Shift for the stiffness term; 0 gives the unshifted Schur.

    where ``B_{k-1}`` is selected by ``schur_diag_mode``. There is exactly ONE
    mode, ``'metric_lumping_probe'`` -- the metric_lumping schur.inner inverse,
    which needs no prior assembly. It was raw_kron-backed until 2026-08-25; the
    switch was forced by that deletion and measured first, see
    docs/research/result_2026-08-25_schur_probe_ab.md.

    This docstring used to advertise four (``'tensor_probe'``, ``'exact_probe'``
    and ``'diag'`` as well) against a guard that accepted two, so two of them
    were rejected by this function's own ValueError and a third pointed at the
    tensor path deleted on 2026-08-25. Callers still passing ``'tensor_probe'``
    (six sites across scripts/benchmark and scripts/debug) therefore raise; what
    they should point at instead is a MEASUREMENT decision, not a repoint, and
    is open -- benchmark_overnight_sweep.py builds its cross-k baseline on that
    mode and switching it silently redefines what the baseline is.
    """
    if dirichlet_variants is None:
        dirichlet_variants = (True, False)
    operators = _ensure_extraction_operators(seq, operators)
    if schur_diag_mode not in _SCHUR_DIAG_MODES:
        raise ValueError(
            "assemble_schur_jacobi_preconditioner schur_diag_mode must be one "
            f"of {_SCHUR_DIAG_MODES} (got {schur_diag_mode!r})"
        )
    dummy_spec = SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='metric_lumping'),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='metric_lumping'),
            outer=MassPreconditionerSpec(kind='none'),
        ),
    )
    for k in ks:
        if k not in (1, 2, 3):
            raise ValueError(
                f"assemble_schur_jacobi_preconditioner: k must be 1, 2, or 3 (got {k})")
        for dirichlet in dirichlet_variants:
            schur_apply = _build_schur_probe_apply(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                eps=eps,
                mode=schur_diag_mode,
                saddle_preconditioner=dummy_spec,
            )
            suffix = '_dbc' if dirichlet else ''
            n = getattr(seq, f'n{k}{suffix}')
            diagonal = _diagonal_from_matvec(schur_apply, n)
            diaginv = _invert_diagonal(diagonal)
            operators = _set_schur_diaginv(
                operators,
                k,
                dirichlet,
                diaginv,
                mode=schur_diag_mode,
            )
    return operators


def _coerce_mass_preconditioner_spec(preconditioner):
    if preconditioner is None:
        return default_mass_preconditioner()
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        return MassPreconditionerSpec(kind=preconditioner)
    raise TypeError(
        "mass preconditioner must be a kind string or MassPreconditionerSpec")


def _validate_public_mass_preconditioner_spec(spec: MassPreconditionerSpec):
    """The whole of the public mass-spec contract, now that it is small.

    Was three per-degree validators built entirely around ``surgery_schur`` and
    ``kind='tensor'``: which combinations took an inner smoother, which were
    legacy aliases, which were disabled. Both are gone, so what is left is that
    the terminal kinds take no smoother.
    """
    if spec.smoother is not None:
        raise ValueError(
            f"kind={spec.kind!r} does not accept an inner smoother")


def _validate_public_k0_mass_preconditioner_spec(spec: MassPreconditionerSpec):
    _validate_public_mass_preconditioner_spec(spec)


def _validate_public_k1_mass_preconditioner_spec(spec: MassPreconditionerSpec):
    _validate_public_mass_preconditioner_spec(spec)


def _validate_public_k2_mass_preconditioner_spec(spec: MassPreconditionerSpec):
    _validate_public_mass_preconditioner_spec(spec)


def _mass_metric_lumping_for(seq, operators, k: int, dirichlet: bool, **kwargs):
    """Return the block-Jacobi MASS preconditioner for ``(k, dirichlet)``.

    Lazily built and memoised
    on the sequence, keyed on GEOMETRY IDENTITY so ``set_map`` /
    ``set_spline_map`` invalidate it. That is what lets a kind be the *default*
    without every call site having to assemble it first.

    THIS IS THE DEFAULT, since 2026-08-22 -- see
    :func:`~mrx.preconditioners.default_mass_preconditioner`. (This line read
    "NOT YET THE DEFAULT" until 2026-08-24; the swap had already happened.)
    """
    from mrx.metric_lumping_laplacian import (  # noqa: PLC0415
        MetricLumpingMass,
    )
    geometry = seq.geometry
    cache = getattr(seq, '_mass_metric_lumping_cache', None)
    if cache is None or cache.get('geometry') is not geometry:
        cache = {'geometry': geometry, 'factors': {}}
        seq._mass_metric_lumping_cache = cache
    key = (int(k), bool(dirichlet))
    if key not in cache['factors']:
        cache['factors'][key] = MetricLumpingMass(
            seq, operators, int(k), bool(dirichlet), **kwargs)
    return cache['factors'][key]


def assemble_mass_metric_lumping_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (0, 1, 2, 3),
        dirichlet_variants: Optional[Sequence[bool]] = None,
        **kwargs) -> SequenceOperators:
    """Eagerly build the block-Jacobi mass preconditioner for the given degrees.

    Optional -- :func:`_mass_metric_lumping_for` builds on demand -- but a
    MetricLumpingMass build probes a dense core, so it is far from free and is
    usually worth doing up front rather than inside the first solve.
    """
    operators = _ensure_extraction_operators(seq, operators)
    if dirichlet_variants is None:
        dirichlet_variants = (True, False)
    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError(
                "metric_lumping mass preconditioner supports k=0..3")
        for dirichlet in dirichlet_variants:
            _mass_metric_lumping_for(seq, operators, k, dirichlet, **kwargs)
    return operators


def _resolve_legacy_mass_preconditioner(seq, operators, k: int, preconditioner):
    if isinstance(preconditioner, str) and preconditioner == 'auto':
        # 'auto' resolves to default_mass_preconditioner() UNCONDITIONALLY --
        # metric_lumping since 2026-08-22. Always buildable on demand
        # (_mass_metric_lumping_for),
        # which is what makes the unconditional resolve safe. The tensor and
        # jacobi paths remain reachable by explicit spec.
        #
        # NOTE: this unconditional resolve is why the mass preconditioner
        # reached solves through apply_mass_matrix_preconditioner while
        # _materialize_default_mass_preconditioner's _tensor_available gate
        # silently disabled it for every k>=1 saddle solve until 2026-08-24.
        return default_mass_preconditioner()
    return _coerce_mass_preconditioner_spec(preconditioner)


def _normalize_mass_preconditioner_spec_for_degree(
        spec: MassPreconditionerSpec, *, k: int):
    if k != 3:
        return spec

    inner_spec = spec.smoother
    if inner_spec is not None:
        return _normalize_mass_preconditioner_spec_for_degree(inner_spec, k=k)

    if not spec.surgery_schur:
        return spec

    return MassPreconditionerSpec(kind=spec.kind)


def _build_operator_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
    operator_apply, preconditioner, allow_none: bool = True,
    orthogonal_vectors=None):
    del orthogonal_vectors  # retained for call-site compatibility
    spec = _resolve_legacy_mass_preconditioner(seq, operators, k, preconditioner)
    spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=k)
    if k == 0:
        _validate_public_k0_mass_preconditioner_spec(spec)
    if k == 1:
        _validate_public_k1_mass_preconditioner_spec(spec)
    if k == 2:
        _validate_public_k2_mass_preconditioner_spec(spec)
    valid_kinds = ('none', 'jacobi', 'metric_lumping')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.kind == 'metric_lumping':
        # Separable Kronecker bulk plus a sandwich, with the polar CORE probed
        # and inverted densely. Never splits the space.
        if spec.surgery_schur:
            raise ValueError(
                "kind='metric_lumping' does not split the space, so "
                "surgery_schur=True is meaningless; drop it")
        if spec.smoother is not None:
            raise ValueError("kind='metric_lumping' does not support a smoother")
        pre = _mass_metric_lumping_for(seq, operators, k, dirichlet)
        return lambda x, pre=pre: pre.apply(x)
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.kind == 'jacobi':
        diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        return lambda x, diaginv=diaginv: diaginv * x
    raise ValueError(f"unsupported mass preconditioner kind {spec.kind!r}")


def _build_mass_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
    preconditioner, allow_none: bool = True):
    def operator_apply(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

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
            seq,
            operators,
            x,
            k - 1,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
            transpose=True,
        )
        inner_d_t_x = inner_preconditioner_apply(d_t_x)
        schur = apply_derivative_matrix(
            seq,
            operators,
            inner_d_t_x,
            k - 1,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
        return apply_stiffness(seq, operators, x, k, dirichlet=dirichlet) \
            + eps * apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet) \
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
    if schur_inner_spec.surgery_schur or schur_inner_spec.smoother is not None:
        raise ValueError(
            "schur.inner must be a terminal preconditioner"
        )

    # The weak term B_{k-1} standing in for M_{k-1}^{-1}. Builds on demand, so
    # this cannot fail for want of a prior assemble_*.
    pre = _mass_metric_lumping_for(seq, operators, k - 1, dirichlet)
    schur_inner = (lambda x, pre=pre: pre.apply(x))
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
    if preconditioner is None or preconditioner == 'auto':
        return _materialize_default_scalar_hodge_preconditioner(
            seq, operators, k=k, dirichlet=dirichlet, eps=eps)
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        return preconditioner
    raise TypeError(
        'scalar Hodge preconditioner must be a kind string or '
        'MassPreconditionerSpec')


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
        _coerce_schur_diag_mode(
            preconditioner.schur.outer,
            context=f"schur.outer kind={preconditioner.schur.outer.kind!r}",
        )
        return preconditioner
    if isinstance(preconditioner, str):
        lower_kind = 'jacobi'
        valid_outer_kinds = ('none', 'jacobi', 'metric_lumping')
        if preconditioner not in valid_outer_kinds:
            raise ValueError(
                "saddle outer kind must be one of "
                f"{valid_outer_kinds} (got {preconditioner!r})"
            )
        lower = MassPreconditionerSpec(kind=lower_kind)
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
    spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=k)
    # 'tensor' used to sit in this list while only 'none' and 'jacobi' were
    # dispatched, so asking for it passed validation and then hit the trailing
    # raise saying it was unsupported.  Same accept-list/dispatch mismatch that
    # has produced several instances in this file; the accept list now names
    # exactly what is implemented.
    #
    # MERGE 2026-08-26: greville-prod fixed the same mismatch by DROPPING
    # 'tensor' and leaving ('none', 'jacobi').  That is the right fix for the
    # accept list it had; this branch additionally IMPLEMENTED the production
    # kind below, so the list names three things and dispatches three things.
    # Both sides agree on the rule -- accept exactly what is dispatched -- and
    # this is the superset.
    #
    # Named through default_mass_preconditioner() rather than spelled out, so
    # this survives the production atom being renamed -- it has been
    # 'raw_kron', then 'block_jacobi', then 'metric_lumping' inside a month.
    # What this branch wants is "whatever the production mass preconditioner
    # currently is", and saying that literally is both more honest and
    # rename-proof.
    production_kind = default_mass_preconditioner().kind
    valid_kinds = ('none', 'jacobi', production_kind)
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.surgery_schur:
        raise ValueError(
            "diffusion upper-block preconditioners do not support surgery_schur"
        )
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.kind == 'jacobi':
        # THE SHIFTED diagonal, 1 / (diag(M) + eps diag(S)).
        #
        # This used to be diag(M)^-1 with eps discarded, which is only a
        # preconditioner for M and degrades as eps lambda_max(M^-1 L) passes
        # 1.  Inverses do not superpose -- (M + eps L)^-1 is NOT M^-1 +
        # eps L^-1 -- but the approximate OPERATORS do, and for a diagonal
        # approximation adding them is exact and free.  This mirrors
        # _build_scalar_hodge_preconditioner_apply's jacobi branch, which has
        # done the same thing for the sibling operator L + eps M all along.
        mass_diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        if eps == 0.0:
            shifted_diaginv = mass_diaginv
        else:
            stiffness_diaginv = _laplacian_diaginv(seq, operators, k, dirichlet)
            shifted_diaginv = 1.0 / (
                1.0 / mass_diaginv + eps / stiffness_diaginv)
        return lambda x, diaginv=shifted_diaginv: diaginv * x
    if spec.kind == production_kind:
        # The production MASS preconditioner.  It approximates M_k and knows
        # nothing about eps L_k, so it is admissible exactly while the
        # operator is mass-dominated, i.e. eps * lambda_max(M^-1 L) << 1
        # (lambda_max ~ h^-2, so ~ n^2 on these grids).  That is the regime
        # the relaxation's hyperregularisation uses.
        #
        # NOT gated on eps here, unlike kind='block' in the scalar Hodge
        # builder, and the difference is deliberate: 'block' approximates
        # L_k and becomes WRONG IN THE DOMINANT TERM the moment the operator
        # is not Laplacian-dominated, whereas this one degrades smoothly
        # towards "a very good preconditioner for the part of the operator
        # that still dominates".  If you push eps until it does not, use
        # kind='jacobi', which stays valid for every eps.
        #
        # NOT DONE, BUT AVAILABLE IF THIS EVER NEEDS TO REACH LARGER eps.
        # Expand the inverse in eps and keep the first order:
        #
        #     (M + eps L)^-1 = M^-1 - eps M^-1 L M^-1 + O(eps^2)
        #     P_1 = P_M - theta eps P_M L P_M      (2 applies + 1 matvec)
        #
        # Note the MINUS.  Accuracy is not the issue -- a preconditioner may
        # truncate freely -- but SPD is: P_1 goes indefinite once
        # theta eps lambda_max(P_M^1/2 L P_M^1/2) > 1, and MINRES on an
        # indefinite preconditioner returns noise rather than converging
        # slowly (see mrx/solvers.py's deliberate refusal to abs() the
        # initial inner product).  Damping with theta < 1 keeps it SPD
        # without needing lambda_max.  Judged not worth it at the mu this
        # is used at: eps lambda_max ~ 0.26 there, so the first-order term
        # moves the spread ~1.26 -> ~1.07, against block_jacobi's 0.70-0.83x
        # on iterations outright.
        return _build_mass_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            preconditioner=default_mass_preconditioner(),
            allow_none=allow_none,
        )
    raise ValueError(
        f"unsupported diffusion preconditioner kind {spec.kind!r} "
        "(richardson/chebyshev removed 2026-08-14, see mrx/experimental/chebyshev.py)"
    )


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
    if spec.kind == 'metric_lumping':
        # The block-Jacobi atom, i.e. the same production Laplacian
        # preconditioner k >= 1 gets through schur.outer. It approximates
        # L_k, so it is admissible only for the UNSHIFTED operator; there is
        # no fallback here on purpose, because a shifted solve quietly
        # dropping to the diagonal is the exact failure this stack has already
        # shipped twice.
        if eps != 0.0:
            raise ValueError(
                "scalar preconditioner kind='metric_lumping' approximates L_k, not "
                f"L_k + eps M_k (got eps={eps!r}); how the atom fits the "
                "shifted operator is unmeasured -- see audit item 3.2")
        if not _metric_lumping_available(seq, k, dirichlet):
            raise ValueError(
                f"scalar preconditioner kind='metric_lumping' needs the metric_lumping "
                f"Laplacian assembled for k={k}, dirichlet={dirichlet}; call "
                "assemble_metric_lumping_laplacian_preconditioner first")
        return lambda x: apply_hodge_laplacian_preconditioner(
            seq, operators, x, k, dirichlet=dirichlet, kind='metric_lumping')
    if spec.kind == 'jacobi':
        stiffness_diaginv = _laplacian_diaginv(seq, operators, k, dirichlet)
        if eps == 0.0:
            shifted_diaginv = stiffness_diaginv
        else:
            mass_diaginv_k = _mass_diaginv(seq, operators, k, dirichlet)
            shifted_diaginv = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv_k)
        return lambda x, diaginv=shifted_diaginv: diaginv * x
    raise ValueError(
        f"unsupported scalar Hodge preconditioner kind {spec.kind!r} "
        "(richardson/chebyshev removed 2026-08-14, see mrx/experimental/chebyshev.py)"
    )


def _build_coupled_saddle_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        upper_preconditioner, lower_preconditioner):
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")

    def apply(x):
        u = x[:n_upper]
        s = x[n_upper:]
        m_inv_s = lower_preconditioner(s)
        w_u = u + apply_derivative_matrix(
            seq, operators, m_inv_s, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        y_u = upper_preconditioner(w_u)
        d_t_y_u = apply_derivative_matrix(
            seq, operators, y_u, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
        z_s = m_inv_s + lower_preconditioner(d_t_y_u)
        return jnp.concatenate([y_u, z_s])

    return apply


METRIC_LUMPING_CACHE_ATTR = "_metric_lumping_laplacian"


def assemble_metric_lumping_laplacian_preconditioner(
        seq, operators: SequenceOperators, ks=(0, 1, 2, 3),
        dirichlets=(False, True), **kwargs):
    """Build the tensor block-Jacobi Laplacian preconditioner for ``L_k``.

    This is the production Laplacian preconditioner for k = 0..3. It replaces
    the per-DoF ``'jacobi'`` diagonal, which is what ``k >= 1`` silently fell
    back to; see ``docs/research/production_simplification_plan.md``.

    Build ONCE per (k, BC) -- the atom is a factorisation, not a per-apply
    computation, so it is cached on ``seq`` rather than rebuilt in the apply.
    It is deliberately NOT stored on ``operators``: that is an ``eqx.Module``,
    and a dict of preconditioner objects is neither a sensible pytree leaf nor
    a hashable static field, so parking it there would risk ``filter_jit``.

    ``kwargs`` go to :class:`MetricLumpingLaplacian`. The defaults are already the
    production configuration -- pass nothing.

    NEEDS ``n >= p + 2``. ``component_factors`` forms ``A^-1 M`` per axis
    (``A`` a 1-D mass weighted by the stiffness profile) and takes its mean
    eigenvalue as a scale; below that the solve goes non-finite and numpy
    raises ``LinAlgError: Array must not contain infs or NaNs`` from inside
    ``eigvals``. ``n - p`` is the number of radial elements, so ``n = 4`` at
    ``p = 3`` is a ONE-element radial mesh. Measured on a toroid
    (``scripts/debug/atom_coarse_grid.py``): at ``p = 3``, ``n = 4`` fails for
    k = 0, 1, 2 in both BCs and ``n = 5, 6, 8, 12`` all build; k = 3 builds even
    at ``n = 4``. The geometry is healthy throughout, so this is the 1-D
    factorisation, not the map.

    Returns ``operators`` unchanged, for symmetry with the other assemble_*
    helpers.
    """
    # Deferred: the experimental module imports back from mrx.operators.
    from mrx.metric_lumping_laplacian import (  # noqa: PLC0415
        MetricLumpingLaplacian,
    )
    cache = dict(getattr(seq, METRIC_LUMPING_CACHE_ATTR, None) or {})
    for k in ks:
        for dbc in dirichlets:
            cache[(int(k), bool(dbc))] = MetricLumpingLaplacian(
                seq, operators, int(k), bool(dbc), **kwargs)
    setattr(seq, METRIC_LUMPING_CACHE_ATTR, cache)
    return operators


PROBED_DIAG_CACHE_ATTR = "_probed_laplacian_diag"


def _probed_laplacian_diaginv(seq, operators: SequenceOperators, k: int,
                              dirichlet: bool):
    """``1 / diag(L_k)`` taken EXACTLY, by one apply per DOF. The REFERENCE.

    ``kind='jacobi'`` is not this. For k >= 1 it uses
    :func:`~mrx.preconditioners.build_extracted_laplacian_diagonal`, whose weak
    half is a closed form under the KRONECKER MASS MODEL -- a model of
    ``D M^-1 D^T``, not the thing the operator actually applies. This probes
    ``apply_hodge_laplacian_approx`` itself, which uses the production mass
    preconditioner as the inner inverse, so it is the exact diagonal of the
    operator as it is really applied.

    That makes it the honest baseline to measure a preconditioner against, and
    the reason to prefer it over ``'jacobi'``: any gap between the two is the
    mass model's error, which has nothing to do with the preconditioner under
    test.

    O(N) applies, so it is cached on the sequence and keyed on geometry
    identity (``set_map`` invalidates it). Far too expensive to rebuild per
    apply, and expensive enough that it is a REFERENCE, not a production
    candidate.
    """
    geometry = seq.geometry
    cache = getattr(seq, PROBED_DIAG_CACHE_ATTR, None)
    if cache is None or cache.get('geometry') is not geometry:
        cache = {'geometry': geometry, 'diag': {}}
        setattr(seq, PROBED_DIAG_CACHE_ATTR, cache)
    key = (int(k), bool(dirichlet))
    if key not in cache['diag']:
        size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
        cache['diag'][key] = _invert_diagonal(_diagonal_from_matvec(
            lambda x: apply_hodge_laplacian_approx(
                seq, operators, x, k, dirichlet=dirichlet),
            size))
    return cache['diag'][key]


def warm_mass_preconditioner_cache(seq, operators: SequenceOperators,
                                   ks=(0, 1, 2, 3), dirichlets=(False, True)):
    """Build every lazily-cached mass preconditioner OUTSIDE any trace.

    The mass factors are built on first use and memoised on the sequence. That
    is fine as long as the first use is not inside a ``jax.lax.while_loop`` --
    the BUILD is host-side numpy (mass diagonals, 1-D inverses, and for
    block_jacobi a dense core probe), so a cold cache inside a traced body dies
    with TracerArrayConversionError.

    It used to be warm by luck: the main mass preconditioner and ``schur.inner``
    were the same kind, so the main path populated the cache before any solve
    entered its loop. Since 2026-08-25 there is only one mass kind, so this
    warms it for every ``(k, BC)`` before the loop. Cheap and idempotent --
    the builder memoises.
    """
    for k in [int(v) for v in ks if 0 <= int(v) <= 3]:
        for dirichlet in dirichlets:
            try:
                _mass_metric_lumping_for(seq, operators, k, dirichlet)
            except Exception:                # noqa: BLE001
                # A degree/BC this kind does not support is not an error here
                # -- the real call site will raise with context.
                # FOLLOW-UP: with one kind left this swallow hides genuine
                # build failures, which is the failure mode the no-defensive-
                # code rule exists to prevent. Removing it would turn an
                # unsupported (k, BC) into a hard failure during WARMING
                # rather than at the real call site, so it is a behaviour
                # change and is flagged rather than folded into this commit.
                pass


def _metric_lumping_available(seq, k: int, dirichlet: bool) -> bool:
    cache = getattr(seq, METRIC_LUMPING_CACHE_ATTR, None)
    return bool(cache) and (int(k), bool(dirichlet)) in cache


def apply_hodge_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                         dirichlet: bool = True,
                                         kind: str = 'auto'):
    """Apply the Hodge-Laplacian preconditioner from an operator bundle.

    ``kind`` options:

    * ``'none'`` — identity (no preconditioning).
    * ``'jacobi'`` — per-DoF diagonal of ``L_k``, always available, but for
      k >= 1 the weak half is a MODEL (the Kronecker mass model), not the
      operator's own ``D M^-1 D^T``.
    * ``'metric_lumping'`` — the metric-lumped atom, k = 0..3, free and Dirichlet.
      Requires :func:`assemble_metric_lumping_laplacian_preconditioner` first.
    * ``'auto'`` — ``'metric_lumping'`` when it has been assembled for this ``(k, BC)``,
      otherwise ``'jacobi'``.

    (``'auto'`` previously resolved to ``'jacobi'`` unconditionally while this
    docstring claimed it preferred ``'tensor'`` at k = 0. It did not. The
    ``'tensor'`` kind itself -- the surgery-plus-Schur k = 0 model -- was
    accepted here until 2026-08-25 with no dispatch branch of its own, so it
    fell through to the ``unreachable`` assertion below.)
    """
    if kind not in ('auto', 'none', 'jacobi', 'metric_lumping'):
        raise ValueError(
            "kind must be 'auto', 'none', 'jacobi' or 'metric_lumping' "
            f"(got {kind!r})")
    if kind == 'auto':
        kind = 'metric_lumping' if _metric_lumping_available(seq, k, dirichlet) else 'jacobi'
    if kind == 'none':
        return v
    if kind == 'jacobi':
        return _laplacian_diaginv(seq, operators, k, dirichlet) * v
    if kind == 'metric_lumping':
        if not _metric_lumping_available(seq, k, dirichlet):
            raise ValueError(
                f"metric_lumping Laplacian preconditioner not assembled for "
                f"k={k}, dirichlet={dirichlet}; call "
                "assemble_metric_lumping_laplacian_preconditioner first")
        cache = getattr(seq, METRIC_LUMPING_CACHE_ATTR)
        return cache[(int(k), bool(dirichlet))].apply(v)
    raise AssertionError("unreachable")


def apply_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                   dirichlet: bool = True,
                                   kind: str = 'auto'):
    """Alias of apply_hodge_laplacian_preconditioner using Laplacian naming."""
    return apply_hodge_laplacian_preconditioner(
        seq, operators, v, k, dirichlet=dirichlet, kind=kind)


def apply_inverse_hodge_laplacian(seq, operators: SequenceOperators, rhs, k: int,
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
    operators = _ensure_extraction_operators(seq, operators)
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
                seq, operators, x, 0, dirichlet=dirichlet),
            rhs,
            mass_matvec=lambda x: apply_mass_matrix(
                seq, operators, x, 0, dirichlet=dirichlet),
            precond_matvec=precond_upper,
            x0=guess,
            vs=vs,
            tol=tol,
            maxiter=maxiter,
        )
        return (u, info) if return_info else u

    return apply_inverse_shifted_hodge_laplacian(
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
        use_harmonic_coarse=None,
        return_info=return_info,
    )


def apply_inverse_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                            dirichlet: bool = True, guess=None,
                            tol: Optional[float] = None,
                            maxiter: Optional[int] = None,
                            preconditioner='auto',
                            return_info: bool = False):
    """Alias of apply_inverse_hodge_laplacian using Laplacian naming."""
    return apply_inverse_hodge_laplacian(
        seq,
        operators,
        rhs,
        k,
        dirichlet=dirichlet,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        return_info=return_info,
    )


def apply_inverse_shifted_hodge_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                          eps: float, dirichlet: bool = True, guess=None,
                                          tol: Optional[float] = None,
                                          maxiter: Optional[int] = None,
                                          preconditioner='auto',
                                          use_harmonic_coarse: Optional[bool] = None,
                                          return_info: bool = False):
    """Solve with the inverse of the shifted Hodge Laplacian ``L_k + eps M_k``.

    For ``k >= 1`` the interface is ``preconditioner``, a structured
    saddle-point preconditioner spec with a lower mass block, a Schur-inner
    mass inverse, a Schur-outer preconditioner, and an optional coupled
    completion. Kind strings are accepted as convenience shorthands.
    """
    operators = _ensure_extraction_operators(seq, operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if k == 0:
        if use_harmonic_coarse is None:
            use_harmonic_coarse = eps > 0 and not dirichlet

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

        if use_harmonic_coarse:
            precond_upper = _wrap_shifted_harmonic_coarse_correction(
                seq, operators, precond_upper, eps, k, dirichlet)

        vs = _nullspace_vectors(
            operators, 0, dirichlet) if eps == 0 else jnp.zeros((0, rhs.shape[0]))
        u, info = solve_singular_cg(
            lambda x: apply_stiffness(
                seq, operators, x, 0, dirichlet=dirichlet)
            + eps * apply_mass_matrix(seq, operators,
                                      x, 0, dirichlet=dirichlet),
            rhs,
            mass_matvec=(
                lambda x: apply_mass_matrix(
                    seq, operators, x, 0, dirichlet=dirichlet)
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
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")
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
    # NOTE the Schur apply is built inside the `else` branch below, not here.
    # It is the ONLY consumer. Building it up front cost a full
    # schur.inner construction -- the whole atom build -- on every
    # production solve, and then discarded it, because `outer='metric_lumping'` uses the
    # atom as the upper-block inverse directly and `outer='jacobi'` builds its
    # own through _build_schur_probe_apply.
    outer_spec = saddle_preconditioner.schur.outer
    if outer_spec.kind == 'metric_lumping':
        # The atom approximates L_k directly, so it needs neither the
        # Schur probe nor schur.inner -- it IS the upper-block inverse.
        if not _metric_lumping_available(seq, k, dirichlet):
            raise ValueError(
                "schur.outer kind='metric_lumping' needs the block-Jacobi Laplacian "
                f"assembled for k={k}, dirichlet={dirichlet}; call "
                "assemble_metric_lumping_laplacian_preconditioner first")

        def precond_upper(x, _k=k, _d=dirichlet):
            return apply_hodge_laplacian_preconditioner(
                seq, operators, x, _k, dirichlet=_d, kind='metric_lumping')
    elif outer_spec.kind == 'jacobi' and outer_spec.smoother is None:
        schur_diaginv = _build_schur_outer_jacobi_diaginv(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            outer_spec=outer_spec,
            saddle_preconditioner=saddle_preconditioner,
            # If a mode-matched Schur Jacobi diagonal was preassembled,
            # reuse it to avoid repeated probe builds.
            allow_stored_tensor_diaginv=True,
        )
        precond_upper = lambda x, d=schur_diaginv: d * x
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
            orthogonal_vectors=vs_upper if eps == 0.0 else None,
        )
    # Apply 1/eps coarse correction on the harmonic upper-block mode, mirroring
    # the k=0 treatment.  For k>=1 the DBC nullspace is always empty on this
    # topology, so only the NBC case is relevant.
    if use_harmonic_coarse is None:
        use_harmonic_coarse = eps > 0 and not dirichlet
    if use_harmonic_coarse and eps > 0:
        # _shifted_harmonic_coarse_ready may return a traced bool when this
        # function is called inside a jax.lax.while_loop body.  Use
        # jax.lax.cond so the selection is JAX-traceable, mirroring the
        # tensor/jacobi fallback in the k=0 path.
        coarse_ready = _shifted_harmonic_coarse_ready(seq, operators, k, dirichlet)
        precond_with_coarse = _wrap_shifted_harmonic_coarse_correction(
            seq, operators, precond_upper, eps, k, dirichlet)
        precond_no_coarse = precond_upper
        precond_upper = lambda x, r=coarse_ready, a=precond_with_coarse, b=precond_no_coarse: (
            jax.lax.cond(r, a, b, x))

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
            seq, operators, x, k, dirichlet=dirichlet)
        + eps * apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        derivative_matvec=lambda s: apply_derivative_matrix(
            seq, operators, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: apply_derivative_matrix(
            seq, operators, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: apply_mass_matrix(
            seq, operators, s, k - 1, dirichlet=dirichlet),
        b_upper=rhs,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_matvec=precond_matvec,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        mass_upper_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet),
        vs_upper=vs_upper,
        vs_lower=vs_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (u, info) if return_info else u


def apply_inverse_shifted_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                    eps: float, dirichlet: bool = True, guess=None,
                                    tol: Optional[float] = None,
                                    maxiter: Optional[int] = None,
                                    preconditioner='auto',
                                    use_harmonic_coarse: Optional[bool] = None,
                                    return_info: bool = False):
    """Alias of apply_inverse_shifted_hodge_laplacian using Laplacian naming."""
    return apply_inverse_shifted_hodge_laplacian(
        seq,
        operators,
        rhs,
        k,
        eps,
        dirichlet=dirichlet,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        use_harmonic_coarse=use_harmonic_coarse,
        return_info=return_info,
    )


def apply_inverse_mass_plus_eps_laplace_matrix(seq, operators: SequenceOperators, rhs, k: int,
                                               eps: float, dirichlet: bool = True, guess=None,
                                               tol: Optional[float] = None,
                                               maxiter: Optional[int] = None,
                                               preconditioner='auto',
                                               return_info: bool = False):
    """Solve with the inverse of M_k + eps L_k using an explicit operator bundle."""
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if eps < 0.0:
        raise ValueError("eps must be nonnegative")

    if eps == 0.0:
        return apply_inverse_mass_matrix(
            seq,
            operators,
            rhs,
            k,
            dirichlet=dirichlet,
            guess=guess,
            tol=tol,
            maxiter=maxiter,
            preconditioner=preconditioner,
            return_info=return_info,
        )

    if k == 0:
        def operator_apply(x):
            return apply_mass_matrix(
                seq, operators, x, 0, dirichlet=dirichlet) + eps * apply_stiffness(
                    seq, operators, x, 0, dirichlet=dirichlet)

        precond_apply = _build_diffusion_preconditioner_apply(
            seq,
            operators,
            k=0,
            dirichlet=dirichlet,
            eps=eps,
            preconditioner=preconditioner,
            allow_none=True,
        )
        x, info = solve_singular_cg(
            operator_apply,
            rhs,
            precond_matvec=precond_apply,
            x0=guess,
            tol=tol,
            maxiter=maxiter,
        )
        return (x, info) if return_info else x

    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")

    def upper_operator_apply(x):
        return apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet) + eps * apply_stiffness(
                seq, operators, x, k, dirichlet=dirichlet)

    upper_preconditioner = _build_diffusion_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        preconditioner=preconditioner,
        allow_none=True,
    )
    lower_preconditioner_apply = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k - 1,
        dirichlet=dirichlet,
        preconditioner=preconditioner,
        allow_none=True,
    )

    def precond_lower(x):
        return (1.0 / eps) * lower_preconditioner_apply(x)

    def precond_upper(x):
        return upper_preconditioner(x)

    u, sigma, info = solve_saddle_point_minres(
        stiffness_matvec=upper_operator_apply,
        derivative_matvec=lambda s: eps * apply_derivative_matrix(
            seq, operators, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: eps * apply_derivative_matrix(
            seq, operators, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: eps * apply_mass_matrix(
            seq, operators, s, k - 1, dirichlet=dirichlet),
        b_upper=rhs,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (u, info) if return_info else u


def apply_hodge_laplacian(seq, operators: SequenceOperators, v, k: int,
                          dirichlet: bool = True, guess=None,
                          tol: Optional[float] = None,
                          maxiter: Optional[int] = None):
    """Apply the Hodge Laplacian using explicit operator data.

    This uses bundled mass, weak derivative, and stiffness operators.
    """
    match k:
        case 0:
            return apply_stiffness(seq, operators, v, 0, dirichlet=dirichlet)
        case 1:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 0, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, operators, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators,
                    Minv_Dt_v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 1, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, operators, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators,
                    Minv_Dt_v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 2, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_derivative_matrix(
                seq, operators,
                Minv_Dt_v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")


def apply_laplacian(seq, operators: SequenceOperators, v, k: int,
                    dirichlet: bool = True, guess=None,
                    tol: Optional[float] = None,
                    maxiter: Optional[int] = None):
    """Alias of apply_hodge_laplacian using Laplacian naming."""
    return apply_hodge_laplacian(
        seq,
        operators,
        v,
        k,
        dirichlet=dirichlet,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
    )


def apply_hodge_laplacian_approx(seq, operators: SequenceOperators, v, k: int,
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
            return apply_stiffness(seq, operators, v, 0, dirichlet=dirichlet)
        case 1:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 0,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 0, dirichlet=dirichlet, kind='auto')
            return apply_stiffness(seq, operators, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators, Minv_Dt_v, 0,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 1,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 1, dirichlet=dirichlet, kind='auto')
            return apply_stiffness(seq, operators, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators, Minv_Dt_v, 1,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 2, dirichlet=dirichlet, kind='auto')
            return apply_derivative_matrix(
                seq, operators, Minv_Dt_v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")


def apply_laplacian_approx(seq, operators: SequenceOperators, v, k: int,
                           dirichlet: bool = True):
    """Alias of apply_hodge_laplacian_approx using Laplacian naming."""
    return apply_hodge_laplacian_approx(
        seq,
        operators,
        v,
        k,
        dirichlet=dirichlet,
    )


# ---------------------------------------------------------------------------
