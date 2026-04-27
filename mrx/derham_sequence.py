from typing import Any, Callable

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

import mrx
from mrx.assembly import (assemble_leray_projection, eval_basis_0_ijk,
                          eval_basis_1_ijk, eval_basis_2_ijk, eval_basis_3_ijk,
                          eval_d_basis_0_ijk, eval_d_basis_1_ijk,
                          eval_d_basis_2_ijk, grad_1d)
from mrx.differential_forms import DifferentialForm
from mrx.extraction_operators import (BoundaryOperator,
                                      PolarExtractionOperator,
                                      bc_extraction_op, get_xi)
from mrx.nullspace import (compute_nullspaces, compute_nullspaces_iterative,
                           find_nullspace_vectors, get_nullspace,
                           get_saddle_point_nullspaces, init_nullspaces)
from mrx.operators import \
    apply_derivative_matrix as apply_derivative_matrix_ops
from mrx.operators import apply_hodge_laplacian as apply_hodge_laplacian_ops
from mrx.operators import \
    apply_hodge_laplacian_approx as apply_hodge_laplacian_approx_ops
from mrx.operators import \
    apply_hodge_laplacian_preconditioner as \
    apply_hodge_laplacian_preconditioner_ops
from mrx.operators import apply_incidence_matrix as apply_incidence_matrix_ops
from mrx.operators import \
    apply_inverse_mass_plus_eps_laplace_matrix as \
    apply_inverse_mass_plus_eps_laplace_matrix_ops
from mrx.operators import \
    apply_inverse_hodge_laplacian as apply_inverse_hodge_laplacian_ops
from mrx.operators import \
    apply_inverse_mass_matrix as apply_inverse_mass_matrix_ops
from mrx.operators import \
    apply_inverse_shifted_hodge_laplacian as \
    apply_inverse_shifted_hodge_laplacian_ops
from mrx.operators import apply_mass_matrix as apply_mass_matrix_ops
from mrx.operators import \
    apply_mass_matrix_preconditioner as apply_mass_matrix_preconditioner_ops
from mrx.operators import \
    apply_projection_matrix as apply_projection_matrix_ops
from mrx.operators import apply_stiffness as apply_stiffness_ops
from mrx.operators import (SequenceOperators,
                           assemble_all_operators,
                           assemble_derivative_operators,
                           assemble_hodge_operators,
                           assemble_incidence_operators,
                           assemble_mass_operators,
                           assemble_projection_operators)
from mrx.projectors import Projector
from mrx.quadrature import QuadratureRule
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg
from mrx.utils import (evaluate_at_xq_deprecated, extract_diag_vector,
                       integrate_against_deprecated, inv33,
                       jacobian_determinant, square_sparse)


def compute_geometry_terms(map: Callable, quad_x: jnp.ndarray):
    """Compute metric and Jacobian terms for a map on the quadrature grid."""

    def G(x):
        DF = jax.jacfwd(map)(x)
        return DF.T @ DF

    metric_jkl = jax.lax.map(G, quad_x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    metric_inv_jkl = jax.lax.map(
        inv33, metric_jkl, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    jacobian_j = jax.lax.map(jacobian_determinant(
        map), quad_x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return metric_jkl, metric_inv_jkl, jacobian_j


class SequenceGeometry(eqx.Module):
    """Geometry data attached to a de Rham sequence.

    An ``eqx.Module`` so that the three quadrature-grid arrays
    (``metric_jkl``, ``metric_inv_jkl``, ``jacobian_j``) are dynamic
    pytree leaves and can flow through ``jit`` / ``grad``. ``map`` is
    kept as a normal field so that if it is itself a pytree (e.g. a
    :class:`SplineMap`), its coefficient leaves are tracked; plain
    ``Callable`` maps are treated as opaque leaves.
    """

    map: Any
    metric_jkl: jnp.ndarray = None
    metric_inv_jkl: jnp.ndarray = None
    jacobian_j: jnp.ndarray = None

    @classmethod
    def from_map(cls, map: Callable, quad_x: jnp.ndarray) -> "SequenceGeometry":
        """Build geometry data by evaluating a map on the quadrature grid."""
        metric_jkl, metric_inv_jkl, jacobian_j = compute_geometry_terms(
            map, quad_x)
        return cls(map, metric_jkl, metric_inv_jkl, jacobian_j)

    @classmethod
    def from_spline_map(cls, spline_map, seq) -> "SequenceGeometry":
        """Sum-factorized geometry builder for tensor-product spline maps.

        Requires that ``seq.evaluate_1d()`` has already been called (so
        ``seq.basis_{r,t,z}_jk`` / ``seq.d_basis_{r,t,z}_jk`` are
        populated) and that ``spline_map.extraction_T`` is set.
        """
        from mrx.spline_geometry import compute_geometry_terms_from_spline

        if spline_map.extraction_T is None:
            raise ValueError(
                "SplineMap.extraction_T must be set for the sum-factorized "
                "geometry path; construct the map via "
                "seq.build_spline_map(coefficients) or pass seq.e0_T.")
        if not hasattr(seq, "basis_r_jk"):
            raise ValueError(
                "Call seq.evaluate_1d() before constructing a "
                "SequenceGeometry from a SplineMap.")
        metric_jkl, metric_inv_jkl, jacobian_j = \
            compute_geometry_terms_from_spline(
                spline_map.coefficients, spline_map.extraction_T, seq)
        return cls(spline_map, metric_jkl, metric_inv_jkl, jacobian_j)


_EXTRACTION_OPERATOR_NAMES = (
    'e0', 'e0_T', 'e0_dbc', 'e0_dbc_T', 'e0_bc', 'e0_bc_T',
    'e1', 'e1_T', 'e1_dbc', 'e1_dbc_T', 'e1_bc', 'e1_bc_T',
    'e2', 'e2_T', 'e2_dbc', 'e2_dbc_T', 'e2_bc', 'e2_bc_T',
    'e3', 'e3_T', 'e3_dbc', 'e3_dbc_T', 'e3_bc', 'e3_bc_T',
)


def _operator_bundle_field_property(name: str):
    def getter(self):
        return getattr(self._require_operators(), name)

    def setter(self, value):
        operators = self.get_operators()
        if operators is None:
            operators = SequenceOperators()
        operators = eqx.tree_at(
            lambda ops: getattr(ops, name),
            operators,
            value,
            is_leaf=lambda x: x is None,
        )
        self.operators = operators

    return property(getter, setter)


class DeRhamSequence():
    """Discrete de Rham sequence on a mapped 3-D domain.

    Holds four ``DifferentialForm`` objects (``basis_0`` … ``basis_3``),
    a ``QuadratureRule``, a ``SequenceGeometry``, and extraction/boundary
    operators for each form degree.  After calling :meth:`assemble_all_sparse`
    (or the individual ``assemble_*`` methods), operator application methods
    become available.

    Attributes
    ----------
    ns : tuple of int
        Number of basis functions in each direction (``n_r``, ``n_θ``, ``n_ζ``).
    ps : tuple of int
        Polynomial degree in each direction (``p_r``, ``p_θ``, ``p_ζ``).
    basis_0, basis_1, basis_2, basis_3 : DifferentialForm
        Spline bases for 0-, 1-, 2-, and 3-forms respectively.
    quad : QuadratureRule
        Tensor-product Gauss quadrature rule used for assembly.
    geometry : SequenceGeometry
        Metric and Jacobian data derived from the logical-to-physical map.
    e0, e1, e2, e3 : jsparse.BCSR
        Extraction operators mapping constrained DOF vectors to the full
        spline basis for each form degree (no Dirichlet BCs).
    e0_dbc, e1_dbc, e2_dbc, e3_dbc : jsparse.BCSR
        Extraction operators with homogeneous Dirichlet BCs applied at
        the radial boundary (or axis in polar coordinates).
    basis_r_jk : jnp.ndarray
        Radial 0-form basis splines evaluated at radial quadrature points.
        Shape ``(n_qr, n_r)``.  Populated by :meth:`evaluate_1d`.
    basis_t_jk : jnp.ndarray
        Poloidal 0-form basis splines evaluated at poloidal quadrature
        points.  Shape ``(n_qθ, n_θ)``.  Populated by :meth:`evaluate_1d`.
    basis_z_jk : jnp.ndarray
        Toroidal 0-form basis splines evaluated at toroidal quadrature
        points.  Shape ``(n_qζ, n_ζ)``.  Populated by :meth:`evaluate_1d`.
    d_basis_r_jk : jnp.ndarray
        Radial derivative splines evaluated at radial quadrature points.
        Shape ``(n_qr, n_r)``.  Populated by :meth:`evaluate_1d`.
    d_basis_t_jk : jnp.ndarray
        Poloidal derivative splines evaluated at poloidal quadrature
        points.  Shape ``(n_qθ, n_θ)``.  Populated by :meth:`evaluate_1d`.
    d_basis_z_jk : jnp.ndarray
        Toroidal derivative splines evaluated at toroidal quadrature
        points.  Shape ``(n_qζ, n_ζ)``.  Populated by :meth:`evaluate_1d`.
    """
    ns: tuple[int, int, int]
    ps: tuple[int, int, int]
    basis_0: DifferentialForm
    basis_1: DifferentialForm
    basis_2: DifferentialForm
    basis_3: DifferentialForm
    quad: QuadratureRule
    geometry: SequenceGeometry
    e0: jsparse.BCOO
    e1: jsparse.BCOO
    e2: jsparse.BCOO
    e3: jsparse.BCOO
    e0_dbc: jsparse.BCOO
    e1_dbc: jsparse.BCOO
    e2_dbc: jsparse.BCOO
    e3_dbc: jsparse.BCOO
    basis_r_jk: jnp.ndarray
    basis_t_jk: jnp.ndarray
    basis_z_jk: jnp.ndarray
    d_basis_r_jk: jnp.ndarray
    d_basis_t_jk: jnp.ndarray
    d_basis_z_jk: jnp.ndarray

    e0 = _operator_bundle_field_property('e0')
    e0_T = _operator_bundle_field_property('e0_T')
    e0_dbc = _operator_bundle_field_property('e0_dbc')
    e0_dbc_T = _operator_bundle_field_property('e0_dbc_T')
    e0_bc = _operator_bundle_field_property('e0_bc')
    e0_bc_T = _operator_bundle_field_property('e0_bc_T')
    e1 = _operator_bundle_field_property('e1')
    e1_T = _operator_bundle_field_property('e1_T')
    e1_dbc = _operator_bundle_field_property('e1_dbc')
    e1_dbc_T = _operator_bundle_field_property('e1_dbc_T')
    e1_bc = _operator_bundle_field_property('e1_bc')
    e1_bc_T = _operator_bundle_field_property('e1_bc_T')
    e2 = _operator_bundle_field_property('e2')
    e2_T = _operator_bundle_field_property('e2_T')
    e2_dbc = _operator_bundle_field_property('e2_dbc')
    e2_dbc_T = _operator_bundle_field_property('e2_dbc_T')
    e2_bc = _operator_bundle_field_property('e2_bc')
    e2_bc_T = _operator_bundle_field_property('e2_bc_T')
    e3 = _operator_bundle_field_property('e3')
    e3_T = _operator_bundle_field_property('e3_T')
    e3_dbc = _operator_bundle_field_property('e3_dbc')
    e3_dbc_T = _operator_bundle_field_property('e3_dbc_T')
    e3_bc = _operator_bundle_field_property('e3_bc')
    e3_bc_T = _operator_bundle_field_property('e3_bc_T')

    def __init__(self, ns, ps, q, types, map, polar, tol=1e-12, maxiter=10_000,
                 r_scale=1.0, n_inner=5, betti_numbers=(1, 1, 0, 0)):
        """Construct a de Rham sequence.

        Parameters
        ----------
        ns : list of int
            Number of basis functions ``[n_r, n_θ, n_ζ]`` for each direction.
        ps : list of int
            Polynomial degree ``[p_r, p_θ, p_ζ]`` of the spline basis.
        q : int
            Number of quadrature points per direction.
        types : list of str
            Boundary-condition type string per direction, e.g.
            ``['periodic', 'periodic', 'periodic']``.
        map : callable
            Logical-to-physical coordinate map ``F: [0,1]³ → ℝ³``.
        polar : bool
            If ``True``, apply polar extraction operators that enforce
            regularity at the magnetic axis.
        tol : float, optional
            Convergence tolerance for iterative solvers.
        maxiter : int, optional
            Maximum iteration count for iterative solvers.
        r_scale : float, optional
            Exponent used to cluster radial knots toward the axis
            (knot spacing proportional to ``r**r_scale``).
        n_inner : int, optional
            Number of inner CG iterations used by block preconditioners.
        betti_numbers : tuple of 4 ints, optional
            ``(b0, b1, b2, b3)`` for the physical domain. Determines how
            many harmonic ``k``-forms each Hodge Laplacian has, and hence
            the shapes of the nullspace arrays stored on
            :class:`SequenceOperators`. Defaults to ``(1, 1, 0, 0)`` which
            matches a solid torus.
        """
        self.ns = tuple(ns)
        self.ps = tuple(ps)
        self.tol = tol
        self.maxiter = maxiter
        self.n_inner = n_inner
        assert len(betti_numbers) == 4, "betti_numbers must have length 4"
        self.betti_numbers = tuple(betti_numbers)
        if not polar:
            Ts = [None] * 3
        else:
            Tr = jnp.concatenate([
                jnp.zeros(ps[0]),
                jnp.linspace(0, 1, ns[0]-ps[0]+1)**r_scale,
                jnp.ones(ps[0])
            ])
            Ts = [Tr, None, None]

        self.basis_0, self.basis_1, self.basis_2, self.basis_3 = [
            DifferentialForm(i, ns, ps, types, Ts) for i in range(0, 4)
        ]
        self.quad = QuadratureRule(self.basis_0, q)
        self.set_map(map)

        if polar:
            xi = get_xi(ns[1])
            e0, e1, e2, e3 = [
                PolarExtractionOperator(Λ, xi, False)
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
            ]
            e0_dbc, e1_dbc, e2_dbc, e3_dbc = [
                PolarExtractionOperator(Λ, xi, True)
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
            ]

        else:
            # TODO: right now, we only support dirichlet BCs in r
            e0, e1, e2, e3 = [
                BoundaryOperator(
                    Λ, ('none', 'none', 'none'))
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
            ]
            e0_dbc, e1_dbc, e2_dbc, e3_dbc = [
                BoundaryOperator(
                    Λ, ('dirichlet', 'none', 'none'))
                for Λ in [self.basis_0, self.basis_1, self.basis_2, self.basis_3]
            ]

        def _to_bcsr_pair(bcoo):
            return jsparse.BCSR.from_bcoo(bcoo), jsparse.BCSR.from_bcoo(bcoo.T)

        e0_bcoo = e0.assemble_sparse()
        e0_dbc_bcoo = e0_dbc.assemble_sparse()
        self.e0, self.e0_T = _to_bcsr_pair(e0_bcoo)
        self.e0_dbc, self.e0_dbc_T = _to_bcsr_pair(e0_dbc_bcoo)
        self.e0_bc, self.e0_bc_T = _to_bcsr_pair(
            bc_extraction_op(e0_bcoo, e0_dbc_bcoo, self.basis_0.n))
        self.n0 = e0.n
        self.n0_dbc = e0_dbc.n
        self.n0_bc = e0.n - e0_dbc.n
        self.n0_1, self.n0_2, self.n0_3 = e0.n, 0, 0
        self.n0_1_dbc, self.n0_2_dbc, self.n0_3_dbc = e0_dbc.n, 0, 0

        e1_bcoo = e1.assemble_sparse()
        e1_dbc_bcoo = e1_dbc.assemble_sparse()
        self.e1, self.e1_T = _to_bcsr_pair(e1_bcoo)
        self.e1_dbc, self.e1_dbc_T = _to_bcsr_pair(e1_dbc_bcoo)
        self.e1_bc, self.e1_bc_T = _to_bcsr_pair(
            bc_extraction_op(e1_bcoo, e1_dbc_bcoo, self.basis_1.n))
        self.n1 = e1.n
        self.n1_dbc = e1_dbc.n
        self.n1_bc = e1.n - e1_dbc.n
        self.n1_1, self.n1_2, self.n1_3 = e1.n1, e1.n2, e1.n3
        self.n1_1_dbc, self.n1_2_dbc, self.n1_3_dbc = e1_dbc.n1, e1_dbc.n2, e1_dbc.n3

        e2_bcoo = e2.assemble_sparse()
        e2_dbc_bcoo = e2_dbc.assemble_sparse()
        self.e2, self.e2_T = _to_bcsr_pair(e2_bcoo)
        self.e2_dbc, self.e2_dbc_T = _to_bcsr_pair(e2_dbc_bcoo)
        self.e2_bc, self.e2_bc_T = _to_bcsr_pair(
            bc_extraction_op(e2_bcoo, e2_dbc_bcoo, self.basis_2.n))
        self.n2 = e2.n
        self.n2_dbc = e2_dbc.n
        self.n2_bc = e2.n - e2_dbc.n
        self.n2_1, self.n2_2, self.n2_3 = e2.n1, e2.n2, e2.n3
        self.n2_1_dbc, self.n2_2_dbc, self.n2_3_dbc = e2_dbc.n1, e2_dbc.n2, e2_dbc.n3

        e3_bcoo = e3.assemble_sparse()
        e3_dbc_bcoo = e3_dbc.assemble_sparse()
        self.e3, self.e3_T = _to_bcsr_pair(e3_bcoo)
        self.e3_dbc, self.e3_dbc_T = _to_bcsr_pair(e3_dbc_bcoo)
        self.e3_bc, self.e3_bc_T = _to_bcsr_pair(
            bc_extraction_op(e3_bcoo, e3_dbc_bcoo, self.basis_3.n))
        self.n3 = e3.n
        self.n3_dbc = e3_dbc.n
        self.n3_bc = e3.n - e3_dbc.n
        self.n3_1, self.n3_2, self.n3_3 = e3.n1, e3.n2, e3.n3
        self.n3_1_dbc, self.n3_2_dbc, self.n3_3_dbc = e3_dbc.n1, e3_dbc.n2, e3_dbc.n3

        self.p0, self.p1, self.p2, self.p3 = [
            Projector(self, k, False) for k in range(4)
        ]

        self.p0_dbc, self.p1_dbc, self.p2_dbc, self.p3_dbc = [
            Projector(self, k, True) for k in range(4)
        ]

        self.p0_bc, self.p1_bc, self.p2_bc, self.p3_bc = [
            Projector(self, k, bc=True) for k in range(4)
        ]

    @property
    def map(self):
        return self.geometry.map

    @property
    def metric_jkl(self):
        return self.geometry.metric_jkl

    @property
    def metric_inv_jkl(self):
        return self.geometry.metric_inv_jkl

    @property
    def jacobian_j(self):
        return self.geometry.jacobian_j

    @property
    def null_0(self):
        return get_nullspace(self._require_operators(), 0, False)

    @property
    def null_1(self):
        return get_nullspace(self._require_operators(), 1, False)

    @property
    def null_2(self):
        return get_nullspace(self._require_operators(), 2, False)

    @property
    def null_3(self):
        return get_nullspace(self._require_operators(), 3, False)

    @property
    def null_0_dbc(self):
        return get_nullspace(self._require_operators(), 0, True)

    @property
    def null_1_dbc(self):
        return get_nullspace(self._require_operators(), 1, True)

    @property
    def null_2_dbc(self):
        return get_nullspace(self._require_operators(), 2, True)

    @property
    def null_3_dbc(self):
        return get_nullspace(self._require_operators(), 3, True)

    def set_geometry(self, geometry: SequenceGeometry):
        """Replace the geometry attached to this sequence."""
        self.geometry = geometry

    def _sync_operators(self):
        """Mirror bundled operators onto legacy fields during the transition."""
        if not hasattr(self, 'operators') or self.operators is None:
            return
        for k in range(4):
            setattr(self, f'm{k}_sp', getattr(self.operators, f'm{k}_sp'))
            setattr(self, f'm{k}_sp_diaginv',
                    getattr(self.operators, f'm{k}_sp_diaginv'))
            setattr(self, f'm{k}_sp_diaginv_dbc',
                    getattr(self.operators, f'm{k}_sp_diaginv_dbc'))
            setattr(self, f'dd{k}_sp_diaginv',
                    getattr(self.operators, f'dd{k}_sp_diaginv'))
            setattr(self, f'dd{k}_sp_diaginv_dbc',
                    getattr(self.operators, f'dd{k}_sp_diaginv_dbc'))
        for k in range(3):
            setattr(self, f'd{k}_sp', getattr(self.operators, f'd{k}_sp'))
            setattr(self, f'd{k}_sp_T', getattr(self.operators, f'd{k}_sp_T'))
        self.grad_grad_sp = self.operators.grad_grad_sp
        self.curl_curl_sp = self.operators.curl_curl_sp
        self.div_div_sp = self.operators.div_div_sp

    def get_operators(self):
        """Return the cached operator bundle, if one is attached."""
        return getattr(self, 'operators', None)

    def set_operators(self, operators, sync_legacy=True):
        """Attach an operator bundle to the sequence and optionally mirror legacy fields.

        If ``operators`` has no nullspace arrays yet, they are initialised to
        zeros with shapes derived from ``self.betti_numbers``.
        """
        current = self.get_operators()
        if operators is not None and current is not None:
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
        if operators is not None and getattr(operators, 'null_0', None) is None:
            operators = init_nullspaces(self, operators,
                                        betti_numbers=self.betti_numbers)
        self.operators = operators
        if sync_legacy and operators is not None:
            self._sync_operators()
        return operators

    def _resolve_operators(self, operators=None):
        """Use an explicit operator bundle when provided, else fall back to the cache."""
        if operators is not None:
            return operators
        return self.get_operators()

    def _require_operators(self, operators=None):
        """Return an explicit operator bundle or raise when none is available."""
        operators = self._resolve_operators(operators)
        if operators is None:
            raise ValueError(
                'Assemble operators first, for example with assemble_all_sparse().')
        return operators

    def set_geometry_terms(self, metric_jkl, metric_inv_jkl, jacobian_j):
        """Replace the geometry tensors used by mapped assembly and operators."""
        self.set_geometry(SequenceGeometry(
            self.map, metric_jkl, metric_inv_jkl, jacobian_j))

    def set_map(self, map):
        """Update the active logical-to-physical map and derived geometry terms."""
        self.set_geometry(SequenceGeometry.from_map(map, self.quad.x))

    def build_spline_map(self, coefficients, extraction=None):
        """Build a spline map using the sequence's scalar spline basis."""
        from mrx.mappings import SplineMap

        if extraction is None:
            extraction = self.e0
            extraction_T = self.e0_T
        else:
            extraction_T = None
        return SplineMap(
            coefficients=coefficients,
            extraction=extraction,
            extraction_T=extraction_T,
            basis_0=self.basis_0,
        )

    def geometry_from_spline_map(self, coefficients, extraction=None):
        """Construct geometry data from spline map coefficients.

        Uses the sum-factorized path when the extraction operator is the
        sequence's own ``e0`` (so we have a precomputed transpose and 1D
        basis evaluations); otherwise falls back to the generic
        ``SequenceGeometry.from_map``.
        """
        spline_map = self.build_spline_map(coefficients, extraction=extraction)
        if spline_map.extraction_T is not None and hasattr(self, "basis_r_jk"):
            return SequenceGeometry.from_spline_map(spline_map, self)
        return SequenceGeometry.from_map(spline_map, self.quad.x)

    def set_spline_map(self, coefficients, extraction=None):
        """Update the sequence geometry from spline map coefficients."""
        self.set_geometry(self.geometry_from_spline_map(
            coefficients, extraction=extraction))

    def _require_reference_mass_matrix(self):
        """Raise if the reference-domain mass matrix has not been assembled."""
        if not hasattr(self, 'reference_m0_sp'):
            raise ValueError(
                'Call assemble_reference_mass_matrix() before using reference 0-form operators.')

    def _apply_reference_mass_matrix(self, v, dirichlet=True):
        """Apply the reference-domain 0-form mass matrix to ``v``."""
        self._require_reference_mass_matrix()
        e = self.e0_dbc if dirichlet else self.e0
        e_T = self.e0_dbc_T if dirichlet else self.e0_T
        return e @ (self.reference_m0_sp @ (e_T @ v))

    def _apply_reference_mass_matrix_preconditioner(self, v, dirichlet=True):
        """Apply the diagonal (Jacobi) preconditioner for the reference mass matrix."""
        self._require_reference_mass_matrix()
        if dirichlet:
            return self.reference_m0_sp_diaginv_dbc * v
        return self.reference_m0_sp_diaginv * v

    def assemble_reference_mass_matrix(self):
        """Assemble and cache the 0-form mass matrix on the reference domain."""
        from mrx.assembly import assemble_scalar_tp
        from mrx.utils import diag_EAET

        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        sp = assemble_scalar_tp(
            self.basis_r_jk, self.basis_t_jk, self.basis_z_jk,
            self.basis_r_jk, self.basis_t_jk, self.basis_z_jk,
            self.quad.w, quad_shape, self.basis_0.shape[0],
            self.basis_0.pr, self.basis_0.pt, self.basis_0.pz)
        self.reference_m0_sp = jsparse.BCSR.from_bcoo(sp)
        self.reference_m0_sp_diaginv = 1.0 / diag_EAET(
            self.e0, self.reference_m0_sp, self.e0_T)
        self.reference_m0_sp_diaginv_dbc = 1.0 / diag_EAET(
            self.e0_dbc, self.reference_m0_sp, self.e0_dbc_T)

    def apply_inverse_reference_mass_matrix(self, rhs, dirichlet=True, guess=None,
                                            tol=None, maxiter=None):
        """Apply the inverse of the cached reference-domain 0-form mass matrix."""
        self._require_reference_mass_matrix()
        return solve_singular_cg(
            lambda x: self._apply_reference_mass_matrix(
                x, dirichlet=dirichlet),
            rhs,
            mass_matvec=lambda x: self._apply_reference_mass_matrix(
                x, dirichlet=dirichlet),
            precond_matvec=lambda x: self._apply_reference_mass_matrix_preconditioner(
                x, dirichlet=dirichlet),
            x0=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter)[0]

    def bc_lift(self, g: jnp.ndarray, k: int) -> jnp.ndarray:
        """Embed boundary DOF values into the full spline basis space.

        Parameters
        ----------
        g : array of shape (n_k_bc,)
            DOF values at the Dirichlet boundary nodes.
        k : int
            Form degree (0, 1, 2, 3).

        Returns
        -------
        array of shape (basis_k.n,)
            Full spline vector with g placed at the BC positions,
            zeros everywhere else.  Multiply any full-spline-space
            operator by this vector to compute the BC contribution.
        """
        e_bc_T = getattr(self, f'e{k}_bc_T')
        return e_bc_T @ g

    def apply_bc_mass_correction(self, g: jnp.ndarray, k: int) -> jnp.ndarray:
        """Compute the DBC-space RHS correction for a non-zero Dirichlet BC.

        For a k-form mass-matrix system  M_dbc @ u = rhs  where the
        boundary DOFs are prescribed as g, the corrected right-hand side is::

            rhs_corrected = rhs - seq.apply_bc_mass_correction(g, k)

        The correction is  E_dbc @ M_full @ E_bc^T @ g, i.e. the
        DBC-space projection of the mass matrix applied to the BC lift.

        Requires ``assemble_all_sparse()`` (or the relevant
        ``assemble_M{k}`` call) to have been called first.

        Parameters
        ----------
        g : array of shape (n_k_bc,)
        k : int

        Returns
        -------
        array of shape (n_k_dbc,)
        """
        m_sp = getattr(self, f'm{k}_sp')
        e_dbc = getattr(self, f'e{k}_dbc')
        e_bc_T = getattr(self, f'e{k}_bc_T')
        return e_dbc @ (m_sp @ (e_bc_T @ g))

    def evaluate_1d(self):
        """Precompute 1-D spline and derivative values at quadrature points.

        Populates ``basis_{r,t,z}_jk`` and ``d_basis_{r,t,z}_jk`` on
        ``self``.  These arrays drive the sum-factorized assembly and
        evaluation routines, and are required by
        :meth:`geometry_from_spline_map` when using the fast spline-geometry
        path.
        """
        # TODO: This should really be fine as double vmap since it is all 1D.
        # Consider replacing with jax.lax.map if we ever run into memory issues.
        self.basis_r_jk = jax.vmap(jax.vmap(self.basis_0.Λ[0], (0, None)),
                                   (None, 0))(self.quad.x_x, self.basis_0.Λ[0].ns)
        self.basis_t_jk = jax.vmap(jax.vmap(self.basis_0.Λ[1], (0, None)),
                                   (None, 0))(self.quad.x_y, self.basis_0.Λ[1].ns)
        self.basis_z_jk = jax.vmap(jax.vmap(self.basis_0.Λ[2], (0, None)),
                                   (None, 0))(self.quad.x_z, self.basis_0.Λ[2].ns)
        self.d_basis_r_jk = jax.vmap(jax.vmap(self.basis_0.dΛ[0], (0, None)),
                                     (None, 0))(self.quad.x_x, self.basis_0.dΛ[0].ns)
        self.d_basis_t_jk = jax.vmap(jax.vmap(self.basis_0.dΛ[1], (0, None)),
                                     (None, 0))(self.quad.x_y, self.basis_0.dΛ[1].ns)
        self.d_basis_z_jk = jax.vmap(jax.vmap(self.basis_0.dΛ[2], (0, None)),
                                     (None, 0))(self.quad.x_z, self.basis_0.dΛ[2].ns)

    def _form_comp_info(self, k):
        """Return component metadata for tensor-product evaluation of the k-th form.

        Returns
        -------
        comp_info : list of tuple
            Each entry ``(output_dim, R_jk, T_jk, Z_jk)`` describes one
            component: the physical vector index and the three 1-D basis
            arrays (one differentiated per form degree).
        comp_shapes : list of int
            Number of DOFs for each component block.
        """
        match k:
            case 0:
                return (
                    [(0, self.basis_r_jk, self.basis_t_jk, self.basis_z_jk)],
                    list(self.basis_0.shape),
                )
            case 1:
                return (
                    [(0, self.d_basis_r_jk, self.basis_t_jk, self.basis_z_jk),
                     (1, self.basis_r_jk, self.d_basis_t_jk, self.basis_z_jk),
                     (2, self.basis_r_jk, self.basis_t_jk, self.d_basis_z_jk)],
                    list(self.basis_1.shape),
                )
            case 2:
                return (
                    [(0, self.basis_r_jk, self.d_basis_t_jk, self.d_basis_z_jk),
                     (1, self.d_basis_r_jk, self.basis_t_jk, self.d_basis_z_jk),
                     (2, self.d_basis_r_jk, self.d_basis_t_jk, self.basis_z_jk)],
                    list(self.basis_2.shape),
                )
            case 3:
                return (
                    [(0, self.d_basis_r_jk, self.d_basis_t_jk, self.d_basis_z_jk)],
                    list(self.basis_3.shape),
                )
            case _:
                raise ValueError("k must be 0, 1, 2, or 3")

    def eval_basis_0_ijk(self, i, j, k):
        """Evaluate the (i, j, k)-th 0-form basis function at all quadrature points."""
        return eval_basis_0_ijk(self, i, j, k)

    def eval_d_basis_0_ijk(self, i, j, k):
        """Evaluate the gradient of the (i, j, k)-th 0-form basis at all quadrature points."""
        return eval_d_basis_0_ijk(self, i, j, k)

    def eval_basis_1_ijk(self, i, j, k):
        """Evaluate the (i, j, k)-th 1-form basis function at all quadrature points."""
        return eval_basis_1_ijk(self, i, j, k)

    def eval_d_basis_1_ijk(self, i, j, k):
        """Evaluate the curl of the (i, j, k)-th 1-form basis at all quadrature points."""
        return eval_d_basis_1_ijk(self, i, j, k)

    def eval_basis_2_ijk(self, i, j, k):
        """Evaluate the (i, j, k)-th 2-form basis function at all quadrature points."""
        return eval_basis_2_ijk(self, i, j, k)

    def eval_d_basis_2_ijk(self, i, j, k):
        """Evaluate the divergence of the (i, j, k)-th 2-form basis at all quadrature points."""
        return eval_d_basis_2_ijk(self, i, j, k)

    def eval_basis_3_ijk(self, i, j, k):
        """Evaluate the (i, j, k)-th 3-form basis function at all quadrature points."""
        return eval_basis_3_ijk(self, i, j, k)

    def l2_norm_sq(self, v, k, dirichlet=True):
        """Return the squared L² norm of a k-form DOF vector ``v``."""
        return v @ self.apply_mass_matrix(v, k, dirichlet=dirichlet)

    def l2_norm(self, v, k, dirichlet=True):
        """Return the L² norm of a k-form DOF vector ``v``."""
        return jnp.sqrt(self.l2_norm_sq(v, k, dirichlet=dirichlet))

    def assemble_all_sparse(self):
        """Assemble and cache all sparse operator matrices.

        Builds mass matrices, derivative matrices, stiffness matrices, and
        Hodge-Laplacian operators for all form degrees, storing the result in
        ``self.operators`` and mirroring legacy fields.  Returns the operator
        bundle.
        """
        operators = assemble_all_operators(
            self, self.geometry, operators=self.get_operators())
        self.set_operators(operators)
        return operators

    def assemble_mass_matrix(self, k):
        """Assemble and cache the mass matrix for k-forms.

        Parameters
        ----------
        k : int
            Form degree (0, 1, 2, or 3).
        """
        return self.set_operators(assemble_mass_operators(
            self, self.geometry,
            operators=self.get_operators(),
            ks=(k,),
        ))

    def assemble_projection_matrix(self, k_from, k_to):
        """Assemble and cache the L²-projection matrix from k_from-forms to k_to-forms.

        Parameters
        ----------
        k_from : int
            Source form degree.
        k_to : int
            Target form degree.
        """
        return self.set_operators(assemble_projection_operators(
            self,
            operators=self.get_operators(),
            pairs=((k_from, k_to),),
        ))

    def assemble_derivative_matrix(self, k):
        """Assemble and cache the weak derivative matrix mapping k-forms to (k+1)-forms.

        Parameters
        ----------
        k : int
            Form degree of the *input* form (0, 1, or 2).
        """
        return self.set_operators(assemble_derivative_operators(
            self, self.geometry,
            operators=self.get_operators(),
            ks=(k,),
        ))

    def _grad_1d(self, d_basis, boundary_type):
        """Return the 1-D gradient matrix for the given derivative basis and BC type."""
        return grad_1d(d_basis, boundary_type)

    def assemble_hodge_laplacian(self, k):
        """Assemble and cache the Hodge-Laplacian stiffness matrix for k-forms.

        Parameters
        ----------
        k : int
            Form degree (0, 1, 2, or 3).
        """
        return self.set_operators(assemble_hodge_operators(
            self, self.geometry,
            operators=self.get_operators(),
            ks=(k,),
        ))

    def assemble_leray_projection(self):
        """Assemble the auxiliary operators required by :meth:`apply_leray_projection`."""
        assemble_leray_projection(self)

    def assemble_incidence_matrix(self, k):
        """Assemble and cache the topological incidence matrix Gk.

        Parameters
        ----------
        k : int
            Form degree of the *input* form (0, 1, or 2).
        """
        return self.set_operators(assemble_incidence_operators(
            self,
            operators=self.get_operators(),
            ks=(k,),
        ))

    def apply_incidence_matrix(self, v, k, dirichlet_in=True, dirichlet_out=True,
                               transpose=False, operators=None):
        """Apply the topological exterior-derivative incidence Gk to ``v``.

        Gk has entries in {-1, 0, +1} and is geometry-independent. On DoF
        spaces where the extraction operators are "unitary" (``e @ e^T = I``),
        this equals ``M_{k+1}^{-1} @ apply_derivative_matrix``. For non-unitary
        extractions (e.g. polar axis gluing) the two differ; in that regime
        :meth:`apply_strong_grad` / curl / div remain the mass-projected form
        and should be preferred when exact d∘d = 0 on extracted DoFs is
        required.
        """
        operators = self._require_operators(operators)
        return apply_incidence_matrix_ops(
            self, operators, v, k,
            dirichlet_in=dirichlet_in,
            dirichlet_out=dirichlet_out,
            transpose=transpose,
        )

    # TODO: Cache the extracted strong derivatives S_k = M_ext^{-1} D_ext as a
    # sparse-plus-low-rank operator and use it here instead of running CG on
    # every call. Decomposition (exact, no thresholding):
    #
    #     S_k = G_ext  +  C_tilde @ P_K^T,
    #
    # where G_ext = E_{k+1} G^k E_k^T is the topological ±1 incidence on the
    # extracted DoFs (sparse), P_K picks the K polar-fused output DoFs (small,
    # ~3 n_z), and C_tilde ∈ R^{n_{k+1} × K} is dense and built once via K CG
    # solves against M_{k+1,ext} on the residual columns
    # R = D_ext - M_{k+1,ext} G_ext (which has only K nonzero columns by
    # construction, since (I - E^T E) vanishes off the polar fusion set).
    # Apply cost then drops from one CG solve per call to one sparse + one
    # K-wide dense matvec. Requires exposing the polar-fused DoF indices from
    # PolarExtractionOperator.
    def apply_strong_grad(self, v, dirichlet_in=True, dirichlet_out=True):
        """Apply the strong gradient M1⁻¹ D0 to a 0-form DOF vector ``v``."""
        dv_dual = self.apply_derivative_matrix(
            v, 0, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out)
        return self.apply_inverse_mass_matrix(dv_dual, 1, dirichlet=dirichlet_out)

    def apply_strong_curl(self, v, dirichlet_in=True, dirichlet_out=True):
        """Apply the strong curl M2⁻¹ D1 to a 1-form DOF vector ``v``."""
        dv_dual = self.apply_derivative_matrix(
            v, 1, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out)
        return self.apply_inverse_mass_matrix(dv_dual, 2, dirichlet=dirichlet_out)

    def apply_strong_div(self, v, dirichlet_in=True, dirichlet_out=True):
        """Apply the strong divergence M3⁻¹ D2 to a 2-form DOF vector ``v``."""
        dv_dual = self.apply_derivative_matrix(
            v, 2, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out)
        return self.apply_inverse_mass_matrix(dv_dual, 3, dirichlet=dirichlet_out)

    def _add_boundary_dual(self, dv_dual, boundary_dual, operator_name):
        """Add a prescribed boundary functional in the operator's dual target space."""
        if boundary_dual is None:
            return dv_dual
        if boundary_dual.shape != dv_dual.shape:
            raise ValueError(
                f"{operator_name}: boundary_dual shape {boundary_dual.shape} does not match dual shape {dv_dual.shape}"
            )
        return dv_dual + boundary_dual

    def apply_weak_grad(self, v, dirichlet_in=True, dirichlet_out=True, boundary_dual=None):
        """
        Apply the weak gradient operator to a vector v.

        This returns ``M2^{-1} (-D2.T v + boundary_dual)`` where
        ``boundary_dual`` is an optional prescribed boundary functional in the
        dual 2-form space.
        """
        dv_dual = -self.apply_derivative_matrix(
            v, 2, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        dv_dual = self._add_boundary_dual(
            dv_dual, boundary_dual, "apply_weak_grad")
        return self.apply_inverse_mass_matrix(dv_dual, 2, dirichlet=dirichlet_out)

    def apply_weak_curl(self, v, dirichlet_in=True, dirichlet_out=True, boundary_dual=None):
        """
        Apply the weak curl operator to a vector v.

        This returns ``M1^{-1} (D1.T v + boundary_dual)`` where
        ``boundary_dual`` is an optional prescribed boundary functional in the
        dual 1-form space.
        """
        dv_dual = self.apply_derivative_matrix(
            v, 1, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        dv_dual = self._add_boundary_dual(
            dv_dual, boundary_dual, "apply_weak_curl")
        return self.apply_inverse_mass_matrix(dv_dual, 1, dirichlet=dirichlet_out)

    def apply_weak_div(self, v, dirichlet_in=True, dirichlet_out=True, boundary_dual=None):
        """
        Apply the weak divergence operator to a vector v.

        This returns ``M0^{-1} (-D0.T v + boundary_dual)`` where
        ``boundary_dual`` is an optional prescribed boundary functional in the
        dual 0-form space.
        """
        dv_dual = -self.apply_derivative_matrix(
            v, 0, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        dv_dual = self._add_boundary_dual(
            dv_dual, boundary_dual, "apply_weak_div")
        return self.apply_inverse_mass_matrix(dv_dual, 0, dirichlet=dirichlet_out)

    def apply_mass_matrix_preconditioner(self, v, k, dirichlet=True,
                                         operators=None, kind='auto'):
        """
        Apply a configured mass-matrix preconditioner for Mk to a vector v.
        """
        operators = self._require_operators(operators)
        return apply_mass_matrix_preconditioner_ops(
            self, operators, v, k, dirichlet=dirichlet, kind=kind)

    def apply_inverse_mass_matrix(self, rhs, k, dirichlet=True, guess=None,
                                  operators=None, tol=None, maxiter=None,
                                  preconditioner='auto',
                                  return_info=False):
        """
        Apply the inverse of the sparse mass matrix Mk⁻¹ for k-forms to a right-hand side,
        solved via CG with a structured mass preconditioner. An optional initial
        guess can be provided to warm-start the solver.
        """
        operators = self._require_operators(operators)
        return apply_inverse_mass_matrix_ops(
            self, operators, rhs, k,
            dirichlet=dirichlet, guess=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter,
            preconditioner=preconditioner,
            return_info=return_info)

    def apply_mass_matrix(self, v, k, dirichlet=True, operators=None):
        """
        Apply the sparse mass matrix Mk for k-forms to a vector v:
            k=0: M0_ij = ∫ Λ0_i Λ0_j det DF dx
            k=1: M1_ij = ∫ Λ1_i · G⁻¹ Λ1_j det DF dx
            k=2: M2_ij = ∫ Λ2_i · G Λ2_j (det DF)⁻¹ dx
            k=3: M3_ij = ∫ Λ3_i Λ3_j (det DF)⁻¹ dx
        """
        operators = self._require_operators(operators)
        return apply_mass_matrix_ops(
            self, operators, v, k, dirichlet=dirichlet)

    def apply_projection_matrix(self, v, k_in, k_out, dirichlet_in=True, dirichlet_out=True,
                                operators=None):
        """
        Apply the sparse projection matrix Pk_in_k_out to a vector v.
        """
        operators = self._require_operators(operators)
        return apply_projection_matrix_ops(
            self, operators, v, k_in, k_out,
            dirichlet_in=dirichlet_in,
            dirichlet_out=dirichlet_out,
        )

    def apply_derivative_matrix(self, v, k, dirichlet_in=True, dirichlet_out=True,
                                transpose=False, operators=None):
        """
        Apply the derivative matrix Dk (mapping k-forms to (k+1)-forms) to a vector v:
            k=0: D0_ij = ∫ Λ1_i · G⁻¹ grad Λ0_j det DF dx  (grad)
            k=1: D1_ij = ∫ Λ2_i · G curl Λ1_j (det DF)⁻¹ dx  (curl)
            k=2: D2_ij = ∫ Λ3_i div Λ2_j (det DF)⁻¹ dx  (div)
        If transpose=True, apply Dk.T instead (mapping (k+1)-forms to k-forms).
        """
        operators = self._require_operators(operators)
        return apply_derivative_matrix_ops(
            self, operators, v, k,
            dirichlet_in=dirichlet_in,
            dirichlet_out=dirichlet_out,
            transpose=transpose,
        )

    def apply_hodge_laplacian(self, v, k, dirichlet=True, operators=None):
        """
        Apply the k-th Hodge Laplacian (δd) to a vector v.

        For k ≥ 1, applied via the saddle-point Schur complement:

        | stiffness_k   d_{k-1}  |   | u (k-form)      |
        | d_{k-1}^T    -M_{k-1}  |   | s ((k-1)-form)  |

        where stiffness_k_ij = ∫ d Λ^k_i · d Λ^k_j dx (assembled directly),
        and d_{k-1} is the discrete exterior derivative from (k-1)- to k-forms.
        Eliminating s = M_{k-1}^{-1} d_{k-1}^T u gives the Schur complement:

            (stiffness_k + d_{k-1} M_{k-1}^{-1} d_{k-1}^T) u

        Concretely:
            k=0: stiffness_0 = grad_grad_ij = ∫ ∇Λ0_i · G⁻¹ ∇Λ0_j det DF dx
            k=1: stiffness_1 = curl_curl_ij = ∫ curl Λ1_i · G curl Λ1_j (det DF)⁻¹ dx,  d_0 = grad
            k=2: stiffness_2 = div_div_ij   = ∫ div  Λ2_i div  Λ2_j (det DF)⁻¹ dx,  d_1 = curl
            k=3: stiffness_3 = 0,  d_2 = div

        The inner M_{k-1}^{-1} solves use CG with Jacobi preconditioning
        to full solver tolerance.
        """
        operators = self._require_operators(operators)
        return apply_hodge_laplacian_ops(
            self, operators, v, k, dirichlet=dirichlet,
            tol=self.tol, maxiter=self.maxiter)

    def apply_hodge_laplacian_approx(self, v, k, dirichlet=True, operators=None):
        """Linear approximate Hodge-Laplacian apply.

        Replaces ``M_{k-1}^{-1}`` in the Schur term with a single configured
        mass-preconditioner apply. Linear, SPD, safe to nest inside Krylov
        solvers and to use as a preconditioner.  Not exact unless the metric
        is tensor-separable on the reference domain.
        """
        operators = self._require_operators(operators)
        return apply_hodge_laplacian_approx_ops(
            self, operators, v, k, dirichlet=dirichlet)

    def apply_mass_plus_eps_laplace_matrix(self, v, k, eps, dirichlet=True, operators=None):
        """Apply ``(M_k + eps * L_k)`` to a k-form vector."""
        return self.apply_mass_matrix(
            v, k, dirichlet=dirichlet, operators=operators) \
            + eps * self.apply_hodge_laplacian(
                v, k, dirichlet=dirichlet, operators=operators)

    def apply_stiffness(self, v, k, dirichlet=True, operators=None):
        """
        Apply the stiffness matrix S_k to a k-form vector v.

            k=0: grad_grad
            k=1: curl_curl
            k=2: div_div
            k=3: 0 (no stiffness)
        """
        operators = self._require_operators(operators)
        return apply_stiffness_ops(
            self, operators, v, k, dirichlet=dirichlet)

    def _get_nullspace(self, k, dirichlet):
        """Return the nullspace basis for the k-th Hodge Laplacian."""
        return get_nullspace(self._require_operators(), k, dirichlet)

    def _get_saddle_point_nullspaces(self, k, dirichlet):
        """Return the pair of nullspace bases for the k-th saddle-point system."""
        return get_saddle_point_nullspaces(
            self, self._require_operators(), k, dirichlet)

    def apply_inverse_hodge_laplacian(self, rhs, k, dirichlet=True, guess=None,
                                      operators=None, tol=None, maxiter=None,
                                      preconditioner='auto',
                                      return_info=False):
        """Apply the inverse of the k-th Hodge Laplacian (δd)⁻¹ to a right-hand side."""
        operators = self._require_operators(operators)
        return apply_inverse_hodge_laplacian_ops(
            self, operators, rhs, k,
            dirichlet=dirichlet, guess=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter,
            preconditioner=preconditioner,
            return_info=return_info)

    def apply_inverse_shifted_hodge_laplacian(self, rhs, k, eps, dirichlet=True, guess=None,
                                              operators=None, tol=None, maxiter=None,
                                              preconditioner='auto',
                                              use_harmonic_coarse=None,
                                              return_info=False):
        """
        Solve (L_k + eps * M_k) x = rhs for the k-form x.

        For eps=0 this reduces to the Hodge Laplacian solve; the system may be
        singular and nullspace deflation is applied automatically.
        For eps > 0 the system is nonsingular (shift-invert for L_k u = λ M_k u).
        The shifted solve itself does not require precomputed nullspace data;
        any harmonic coarse correction is optional and should stay disabled
        while inverse iteration is still constructing those vectors.

        For k=0: solved with CG on ``(S_0 + eps M_0) u = rhs``.
        For k>=1: MINRES on the symmetric saddle-point form of L_k + eps M_k:

            | S_k + eps*M_k    D_{k-1}   | | u |   | rhs |
            | D_{k-1}^T       -M_{k-1}   | | σ | = | 0 |
        """
        operators = self._require_operators(operators)
        return apply_inverse_shifted_hodge_laplacian_ops(
            self, operators, rhs, k, eps,
            dirichlet=dirichlet, guess=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter,
            preconditioner=preconditioner,
            use_harmonic_coarse=use_harmonic_coarse,
            return_info=return_info)

    def apply_inverse_mass_plus_eps_laplace_matrix(self, rhs, k, eps, dirichlet=True, guess=None,
                                                   operators=None, tol=None, maxiter=None,
                                                   preconditioner='auto',
                                                   return_info=False):
        """
        Solve (M_k + eps * L_k) x = rhs for the k-form x.

        For k=0: (M_0 + eps * S_0) is SPD, solved with CG.
        For k>=1: uses MINRES on the symmetric saddle-point system:

            | M_k + eps*S_k    eps*D_{k-1}   | | u |   | rhs |
            | eps*D_{k-1}^T   -eps*M_{k-1}   | | σ | = | 0 |

        The system is nonsingular (no nullspace) since M_k + eps*L_k is SPD.
        Out-of-the-box diffusion preconditioners currently use the same mass-side
        defaults as the other inverse paths: Jacobi, tensor, and Chebyshev.
        """
        operators = self._require_operators(operators)
        return apply_inverse_mass_plus_eps_laplace_matrix_ops(
            self, operators, rhs, k, eps,
            dirichlet=dirichlet, guess=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter,
            preconditioner=preconditioner,
            return_info=return_info)

    def apply_hodge_laplacian_preconditioner(self, v, k, dirichlet=True,
                                             operators=None, kind='auto'):
        """
        Apply a preconditioner for the k-th Hodge Laplacian to a vector ``v``.

        ``kind`` selects between ``'none'`` (identity), ``'jacobi'`` (per-DoF
        diagonal) and ``'tensor'`` (tensorized auxiliary-space
        preconditioner; available for ``k = 0`` when the tensor Hodge data are
        assembled, and for ``k = 3`` via the tensor round-trip path).
        ``'auto'`` (the default) uses ``'tensor'`` when available and falls
        back to ``'jacobi'`` otherwise.
        """
        operators = self._require_operators(operators)
        return apply_hodge_laplacian_preconditioner_ops(
            self, operators, v, k, dirichlet=dirichlet, kind=kind)

    def _compute_nullspaces(self, betti_numbers=None, eps=1e-6):
        """Iteratively compute harmonic forms and store them on ``self.operators``.

        ``betti_numbers`` defaults to ``self.betti_numbers``. Returns the
        info dict from :func:`compute_nullspaces_iterative`.
        """
        operators, info = compute_nullspaces_iterative(
            self, self._require_operators(),
            betti_numbers=betti_numbers, eps=eps)
        self.operators = operators
        return info

    def _find_nullspace_vectors(self, k, n_vectors, eps, dirichlet=True):
        """Find ``n_vectors`` nullspace vectors of the k-th Hodge Laplacian via inverse iteration."""
        return find_nullspace_vectors(
            self, self._require_operators(), k, n_vectors, eps, dirichlet)

    def compute_nullspaces(self):
        """Compute and cache the harmonic forms for all form degrees (closed-form)."""
        self.operators = compute_nullspaces(self, self._require_operators())
        return self.operators

    def init_nullspaces(self, betti_numbers=None):
        """Initialise zero-valued nullspace arrays on ``self.operators``.

        Shapes are derived from ``betti_numbers`` (or ``self.betti_numbers``).
        """
        self.operators = init_nullspaces(
            self, self._require_operators(), betti_numbers=betti_numbers)
        return self.operators

    # def cross_product_projection_deprecated(
    #     self, w, u, n, m, k,
    #     dirichlet_n=True,
    #     dirichlet_m=True,
    #     dirichlet_k=True
    # ):
    #     """
    #     Evaluate the projection of the cross product of the m-form w
    #     and the k-form u onto an n-form.
    #     TODO: Tobi please add a description of these projections.

    #     Args:
    #         w (array): m-form dofs
    #         u (array): k-form dofs
    #         n, m, k (ints): degree of the forms
    #         dirichlet_n: boundary condtions on the n-form (default True)
    #         dirichlet_m: boundary condtions on the m-form (default True)
    #         dirichlet_k: boundary condtions on the k-form (default True)

    #     Returns:
    #         array: ∫ (wₕ × uₕ) · Λn[i] dx for all i

    #     """
    #     match n:
    #         case 1:
    #             if dirichlet_n:
    #                 en = self.e1_dbc
    #             else:
    #                 en = self.e1
    #             eval_basis_n_ijk = self.eval_basis_1_ijk
    #             nn = self.basis_1.n
    #         case 2:
    #             if dirichlet_n:
    #                 en = self.e2_dbc
    #             else:
    #                 en = self.e2
    #             eval_basis_n_ijk = self.eval_basis_2_ijk
    #             nn = self.basis_2.n
    #         case _:
    #             raise ValueError("n must be 1 or 2")
    #     match m:
    #         case 1:
    #             if dirichlet_m:
    #                 em_T = self.e1_dbc_T
    #             else:
    #                 em_T = self.e1_T
    #             eval_basis_m_ijk = self.eval_basis_1_ijk
    #         case 2:
    #             if dirichlet_m:
    #                 em_T = self.e2_dbc_T
    #             else:
    #                 em_T = self.e2_T
    #             eval_basis_m_ijk = self.eval_basis_2_ijk
    #         case _:
    #             raise ValueError("m must be 1 or 2")
    #     match k:
    #         case 1:
    #             if dirichlet_k:
    #                 ek_T = self.e1_dbc_T
    #             else:
    #                 ek_T = self.e1_T
    #             eval_basis_k_ijk = self.eval_basis_1_ijk
    #         case 2:
    #             if dirichlet_k:
    #                 ek_T = self.e2_dbc_T
    #             else:
    #                 ek_T = self.e2_T
    #             eval_basis_k_ijk = self.eval_basis_2_ijk
    #         case _:
    #             raise ValueError("k must be 1 or 2")

    #     # w and u evaluated at quadrature points: shape: n_q x 3
    #     w_jk = evaluate_at_xq_deprecated(eval_basis_m_ijk,
    #                                      em_T @ w, self.quad.n, 3)
    #     u_jk = evaluate_at_xq_deprecated(eval_basis_k_ijk,
    #                                      ek_T @ u, self.quad.n, 3)

    #     # now, we compute
    #     # ∑ Λn[i](x_j)_a w(x_j)_b u(x_j)_c ) t(x_j)_abc
    #     # where t is some transformation depending on n,m,k and the metric
    #     # and we sum over j (quadrature points) and b,c (dimensions)
    #     # To avoid assembling the huge Λn[i](x_j)_a tensor, we scan over i.
    #     if n == 1 and m == 2 and k == 1:
    #         # ∫ Λ[i] (Gw x u) / J dx
    #         Gw_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_jk)
    #         Gw_x_u_jk = jnp.cross(Gw_jk, u_jk, axis=1)
    #         f_jk = Gw_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
    #     elif n == 1 and m == 1 and k == 1:
    #         # ∫ Λ[i] (w x u) dx
    #         w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
    #         f_jk = w_x_u_jk * (self.quad.w)[:, None]
    #     elif n == 2 and m == 1 and k == 1:
    #         # ∫ Λ[i] G(w x u) / J dx
    #         w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
    #         G_wxu_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_x_u_jk)
    #         f_jk = G_wxu_jk * (self.quad.w / self.jacobian_j)[:, None]
    #     elif n == 2 and m == 2 and k == 1:
    #         # ∫ Λ[i] (w x G_inv u) dx
    #         Ginvu_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, u_jk)
    #         w_x_Ginvu_jk = jnp.cross(w_jk, Ginvu_jk, axis=1)
    #         f_jk = w_x_Ginvu_jk * (self.quad.w)[:, None]
    #     elif n == 1 and m == 2 and k == 2:
    #         # ∫ Λ[i] G_inv(w x u) dx
    #         w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
    #         Ginv_wxu_jk = jnp.einsum(
    #             'jkl,jk->jl', self.metric_inv_jkl, w_x_u_jk)
    #         f_jk = Ginv_wxu_jk * (self.quad.w)[:, None]
    #     elif n == 2 and m == 1 and k == 2:
    #         # ∫ Λ[i] (G_inv w x u) dx
    #         Ginvw_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, w_jk)
    #         Ginvw_x_u_jk = jnp.cross(Ginvw_jk, u_jk, axis=1)
    #         f_jk = Ginvw_x_u_jk * (self.quad.w)[:, None]
    #     elif n == 2 and m == 2 and k == 2:
    #         # ∫ Λ[i] (w x u) / J dx
    #         w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
    #         f_jk = w_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
    #     else:
    #         raise ValueError("Not yet implemented")
    #     return en @ integrate_against_deprecated(eval_basis_n_ijk, f_jk, nn)

    def cross_product_projection(
        self, w, u, n, m, k,
        dirichlet_n=True,
        dirichlet_m=True,
        dirichlet_k=True
    ):
        """Project a cross product of two differential forms onto an n-form.

        Computes the n-form dual DOF vector

            ``v_i = ∫ Λⁿ_i · (w × u) dx``

        with appropriate metric contractions depending on the form degrees
        ``n``, ``m``, ``k``.  Uses the tensor-product structure for efficient
        evaluation and integration.

        Parameters
        ----------
        w : array
            DOF vector of the m-form.
        u : array
            DOF vector of the k-form.
        n : int
            Form degree of the output (1 or 2).
        m : int
            Form degree of the first input (1 or 2).
        k : int
            Form degree of the second input (1 or 2).
        dirichlet_n : bool, optional
            Use Dirichlet-constrained extraction for the output n-form.
        dirichlet_m : bool, optional
            Use Dirichlet-constrained extraction for the input m-form.
        dirichlet_k : bool, optional
            Use Dirichlet-constrained extraction for the input k-form.

        Returns
        -------
        array
            n-form dual DOF vector (apply ``M_n⁻¹`` to obtain primal DOFs).
        """
        from mrx.utils import evaluate_at_xq, integrate_against
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)

        # Extraction matrices and comp_info for output form n
        match n:
            case 1:
                en = self.e1_dbc if dirichlet_n else self.e1
                comp_info_n, comp_shapes_n = self._form_comp_info(1)
                nn = self.basis_1.n
            case 2:
                en = self.e2_dbc if dirichlet_n else self.e2
                comp_info_n, comp_shapes_n = self._form_comp_info(2)
                nn = self.basis_2.n
            case _:
                raise ValueError("n must be 1 or 2")

        # Extraction matrices and comp_info for input form m
        match m:
            case 1:
                em_T = self.e1_dbc_T if dirichlet_m else self.e1_T
                comp_info_m, comp_shapes_m = self._form_comp_info(1)
            case 2:
                em_T = self.e2_dbc_T if dirichlet_m else self.e2_T
                comp_info_m, comp_shapes_m = self._form_comp_info(2)
            case _:
                raise ValueError("m must be 1 or 2")

        # Extraction matrices and comp_info for input form k
        match k:
            case 1:
                ek_T = self.e1_dbc_T if dirichlet_k else self.e1_T
                comp_info_k, comp_shapes_k = self._form_comp_info(1)
            case 2:
                ek_T = self.e2_dbc_T if dirichlet_k else self.e2_T
                comp_info_k, comp_shapes_k = self._form_comp_info(2)
            case _:
                raise ValueError("k must be 1 or 2")

        # TP evaluation at quadrature points
        w_jk = evaluate_at_xq(em_T @ w, comp_info_m, comp_shapes_m,
                              quad_shape, 3)
        u_jk = evaluate_at_xq(ek_T @ u, comp_info_k, comp_shapes_k,
                              quad_shape, 3)

        # Nonlinear part (same as original)
        if n == 1 and m == 2 and k == 1:
            Gw_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_jk)
            Gw_x_u_jk = jnp.cross(Gw_jk, u_jk, axis=1)
            f_jk = Gw_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
        elif n == 1 and m == 1 and k == 1:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 1 and k == 1:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            G_wxu_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, w_x_u_jk)
            f_jk = G_wxu_jk * (self.quad.w / self.jacobian_j)[:, None]
        elif n == 2 and m == 2 and k == 1:
            Ginvu_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, u_jk)
            w_x_Ginvu_jk = jnp.cross(w_jk, Ginvu_jk, axis=1)
            f_jk = w_x_Ginvu_jk * (self.quad.w)[:, None]
        elif n == 1 and m == 2 and k == 2:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            Ginv_wxu_jk = jnp.einsum(
                'jkl,jk->jl', self.metric_inv_jkl, w_x_u_jk)
            f_jk = Ginv_wxu_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 1 and k == 2:
            Ginvw_jk = jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, w_jk)
            Ginvw_x_u_jk = jnp.cross(Ginvw_jk, u_jk, axis=1)
            f_jk = Ginvw_x_u_jk * (self.quad.w)[:, None]
        elif n == 2 and m == 2 and k == 2:
            w_x_u_jk = jnp.cross(w_jk, u_jk, axis=1)
            f_jk = w_x_u_jk * (self.quad.w / self.jacobian_j)[:, None]
        else:
            raise ValueError("Not yet implemented")

        # TP integration
        return en @ integrate_against(
            f_jk, comp_info_n, comp_shapes_n, quad_shape)

    def pressure_projection(
        self, p, u, gamma,
        dirichlet_p=True,
        dirichlet_u=True,
    ):
        """Evaluate the pressure projection -(grad p · u + γ p div u).

        Computes the 0-form dual DOF vector:

            q_i = ∫ Λ⁰_i (−∇p · u − γ p div u) w dx

        The 0-form mass matrix weight J cancels with the 1/J from the
        wedge product (1-form · 2-form) and from div = (1/J) div_logical,
        so the integrand has no metric or Jacobian — only quad weights.

        Parameters
        ----------
        p : array  –  0-form DOFs
        u : array  –  2-form DOFs
        gamma : float  –  adiabatic exponent
        dirichlet_p : bool  –  Dirichlet BCs on p
        dirichlet_u : bool  –  Dirichlet BCs on u

        Returns
        -------
        q_dual : array  –  0-form dual DOFs (apply M0⁻¹ to get primal DOFs)
        """
        from mrx.utils import evaluate_at_xq, integrate_against
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)

        types = self.basis_0.types
        grad_r = self._grad_1d(self.d_basis_r_jk, types[0])
        grad_t = self._grad_1d(self.d_basis_t_jk, types[1])
        grad_z = self._grad_1d(self.d_basis_z_jk, types[2])

        # --- evaluate p at quad points (0-form, 1 component) ---
        ep_T = self.e0_dbc_T if dirichlet_p else self.e0_T
        comp_info_0, comp_shapes_0 = self._form_comp_info(0)
        p_jk = evaluate_at_xq(ep_T @ p, comp_info_0, comp_shapes_0,
                              quad_shape, 1)  # (n_q, 1)

        # --- evaluate grad(p) at quad points (3 components) ---
        s0 = list(self.basis_0.shape)[0]
        d0_comp_info = [
            (0, grad_r, self.basis_t_jk, self.basis_z_jk),
            (1, self.basis_r_jk, grad_t, self.basis_z_jk),
            (2, self.basis_r_jk, self.basis_t_jk, grad_z),
        ]
        d0_comp_shapes = [s0, s0, s0]
        grad_p_jk = evaluate_at_xq(
            jnp.tile(ep_T @ p, 3), d0_comp_info, d0_comp_shapes,
            quad_shape, 3)  # (n_q, 3)

        # --- evaluate u at quad points (2-form, 3 components) ---
        eu_T = self.e2_dbc_T if dirichlet_u else self.e2_T
        comp_info_2, comp_shapes_2 = self._form_comp_info(2)
        u_jk = evaluate_at_xq(eu_T @ u, comp_info_2, comp_shapes_2,
                              quad_shape, 3)  # (n_q, 3)

        # --- evaluate div_logical(u) at quad points (scalar) ---
        s2 = list(self.basis_2.shape)
        div_comp_info = [
            (0, grad_r, self.d_basis_t_jk, self.d_basis_z_jk),
            (0, self.d_basis_r_jk, grad_t, self.d_basis_z_jk),
            (0, self.d_basis_r_jk, self.d_basis_t_jk, grad_z),
        ]
        div_comp_shapes = [s2[0], s2[1], s2[2]]
        div_u_jk = evaluate_at_xq(eu_T @ u, div_comp_info, div_comp_shapes,
                                  quad_shape, 1)  # (n_q, 1)

        # --- combine: q = -(grad_p · u) - γ p div_logical(u) ---
        grad_p_dot_u = jnp.sum(grad_p_jk * u_jk, axis=1, keepdims=True)
        q_jk = -(grad_p_dot_u + gamma * p_jk * div_u_jk)  # (n_q, 1)

        # Weight by quadrature weights only (J from M0 cancels 1/J in formula)
        f_jk = q_jk * self.quad.w[:, None]

        # Integrate against 0-form basis
        e0 = self.e0_dbc if dirichlet_p else self.e0
        return e0 @ integrate_against(f_jk, comp_info_0, comp_shapes_0,
                                      quad_shape)

    def apply_leray_projection(self, v, k=2, p_guess=None):
        """
        Apply the Leray projection to a 1 or 2-form v.

        When k = 2:
            Solves the system (k=3 Hodge Laplacian):
            div v = div σ
            (σ, ω) = -(p, div ω) ∀ω 2-forms
            -> div(v - σ) = 0 and σ.n = 0 on the boundary.
        When k = 1:
            Solves the k=0 Hodge Laplacian:
            (grad p, grad ω) = (v, grad ω) ∀ω 0-forms
            -> div(v - grad p) = 0 and p = 0 on the boundary.

        Parameters
        ----------
        v : jnp.ndarray 
            The vector form DoFs
        k : int
            The degree of the vector form
        p_guess : jnp.ndarray 
            Guess for pressure form DoFs

        Returns
        -------
        v_out : jnp.ndarray 
            divergence-cleaned v
        p : jnp.ndarray 
            The pressure form DoFs

        """
        if k == 2:
            p_guess = jnp.zeros(self.n3_dbc) if p_guess is None else p_guess
            # Assumes dirichlet == True on all spaces.
            div_v = self.apply_derivative_matrix(
                v, 2, dirichlet_in=True, dirichlet_out=True)
            p = self.apply_inverse_hodge_laplacian(
                div_v, 3, dirichlet=True, guess=p_guess)
            σ = -self.apply_weak_grad(p, True, True)
            return v - σ, p
        elif k == 1:
            # Assumes dirichlet == False on all spaces.
            p_guess = jnp.zeros(self.n0) if p_guess is None else p_guess
            div_v = -self.apply_derivative_matrix(
                v, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
            p = self.apply_inverse_hodge_laplacian(
                div_v, 0, dirichlet=False, guess=p_guess)
            σ = -self.apply_strong_grad(p, False, False)
            return v - σ, p
        elif k == 1:
            # Assumes dirichlet == False on all spaces.
            p_guess = jnp.zeros(self.n0) if p_guess is None else p_guess
            div_v = -self.apply_derivative_matrix(
                v, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
            p = self.apply_inverse_hodge_laplacian(
                div_v, 0, dirichlet=False, guess=p_guess)
            σ = -self.apply_strong_grad(p, False, False)
            return v - σ, p
            return v - σ, p
