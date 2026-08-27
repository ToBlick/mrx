"""The :class:`DeRhamSequence`: spline spaces, extraction, geometry, and every operator apply and solve.

Design
------
The sequence separates what never changes from what a map changes.

**Static (on the sequence, built once in** ``__init__`` **):** the 1-D spline bases and their quadrature tables, the DoF counts, the
polar weights ``xi``, the extraction operators ``e0 .. e3`` (free, Dirichlet
and boundary flavours), and the topological incidence stencils (``g0``,
``g1``, ``g2`` and the polar grad/curl corrections). All of this depends on
``(ns, ps, types, polar)`` only. The sequence is a plain Python object; the
solvers close over it, and ``jax.jit(..., static_argnames=["seq"])`` hashes
it by identity.

**Geometry (on** ``seq.geometry`` **, installed by** ``set_map`` **):** the
metric, its inverse and the Jacobian at the quadrature points, plus the
matrix-free mass and projection applies built from them. Masses,
derivatives, stiffness and Laplacian applies need nothing else.

**Preconditioners and harmonic forms (on** ``seq.operators`` **, a**
:class:`~mrx.operators.SequenceOperators` **pytree, built by**
:meth:`~DeRhamSequence.build_preconditioners` **):** every factorisation of
the installed metric -- mass and Laplacian atoms, Jacobi diagonals, probed
Schur diagonals -- and the nullspace vectors. Nothing on the bundle is built
on first use: it is built by that one call, against the geometry installed
at that moment, and a new geometry means calling it again. That is the
contract for an outer loop over geometries (stellarator optimisation:
relaxation inside, the map outside).

The rule for where a new thing belongs: changes with the map, or is a solve
of it -> the bundle; depends on the bases only -> the sequence.

The ``apply_*`` methods are forwarders to the free functions in
:mod:`mrx.operators` with ``self`` and ``self.operators`` filled in.
"""
import jax
import jax.numpy as jnp

import mrx
from mrx.differential_forms import DifferentialForm
from mrx.extraction_operators import (BoundaryOperator,
                                      PolarExtractionOperator,
                                      bc_extraction_op, get_xi)
from mrx.nullspace import (compute_nullspaces, compute_nullspaces_iterative,
                           find_nullspace_vectors, get_nullspace,
                           get_saddle_point_nullspaces, init_nullspaces)
import mrx.operators as op
from mrx.local_assembly import (_second_derivative_tables,
                                build_matrixfree_mass_apply,
                                build_matrixfree_projection_apply)
from mrx.projectors import greville_axes, load as _load, interpolate as _interpolate
from mrx.quadrature import QuadratureRule
from mrx.geometry import SequenceGeometry, grad_1d





class DeRhamSequence():
    """Discrete de Rham sequence on a mapped 3-D domain.

    Holds four ``DifferentialForm`` objects (``basis_0`` … ``basis_3``),
    a ``QuadratureRule``, the extraction and incidence operators of every
    form degree (all static), the ``SequenceGeometry`` installed by
    :meth:`set_map`, and the :class:`~mrx.operators.SequenceOperators`
    bundle built by :meth:`build_preconditioners`.

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
    xi : jnp.ndarray or None
        Polar extraction weights ``(3, 2, n_θ)`` (:func:`~mrx.extraction_operators.get_xi`);
        ``None`` on a non-polar sequence.
    e0, e1, e2, e3 : MatrixFreeExtraction
        Extraction operators mapping constrained DOF vectors to the full
        spline basis for each form degree (no Dirichlet BCs). ``.T`` is the
        transpose.
    e0_dbc, e1_dbc, e2_dbc, e3_dbc : MatrixFreeExtraction
        Extraction operators with homogeneous Dirichlet BCs applied at
        the radial boundary (or axis in polar coordinates).
    e0_bc, e1_bc, e2_bc, e3_bc : MatrixFreeExtraction
        Extraction of the boundary DOFs (those in ``e_k`` but not ``e_k_dbc``).
    g0, g1, g2 : _MatrixFreeIncidence
        Raw {-1, 0, +1} incidence stencils (grad, curl, div) and their
        transposes ``g*_T``.
    g0_grad, g1_curl : dict or None
        Analytic polar grad/curl stencils on extracted DoFs, keyed
        ``(dirichlet_in, dirichlet_out)``; ``None`` on a non-polar sequence.
    geometry : SequenceGeometry or None
        Metric and Jacobian data of the installed map (:meth:`set_map`).
    mass_apply, projection_apply : dict or None
        Matrix-free raw-DOF applies of ``M_k`` and ``P_{k_in k_out}``, built
        with the geometry.
    operators : SequenceOperators or None
        Preconditioners and harmonic forms of the installed geometry
        (:meth:`build_preconditioners`).
    basis_r_jk, basis_t_jk, basis_z_jk : jnp.ndarray
        0-form basis splines at the quadrature points of each direction,
        ``(n_q, n)``.
    d_basis_r_jk, d_basis_t_jk, d_basis_z_jk : jnp.ndarray
        Derivative-basis splines at the quadrature points, ``(n_q, n)``.
    dd_basis_jk : tuple of jnp.ndarray
        Second derivatives of the derivative-basis splines at the
        quadrature points, per direction (the metric-lumping stiffness
        tables).
    greville : tuple of _GrevilleAxis
        Greville points, collocation/histopolation matrices and span
        quadrature per direction (:func:`mrx.projectors.greville_axes`).
    """
    ns: tuple[int, int, int]
    ps: tuple[int, int, int]
    basis_0: DifferentialForm
    basis_1: DifferentialForm
    basis_2: DifferentialForm
    basis_3: DifferentialForm
    quad: QuadratureRule
    geometry: SequenceGeometry
    basis_r_jk: jnp.ndarray
    basis_t_jk: jnp.ndarray
    basis_z_jk: jnp.ndarray
    d_basis_r_jk: jnp.ndarray
    d_basis_t_jk: jnp.ndarray
    d_basis_z_jk: jnp.ndarray

    def __init__(self, ns, ps, q, types, *, polar,
                 tol=None, maxiter=10_000,
                 r_scale=1.0, knots=None,
                 n_inner=5, betti_numbers=(1, 1, 0, 0)):
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
        polar : bool
            If ``True``, apply polar extraction operators that enforce
            regularity at the magnetic axis.
        tol : float, optional
            Relative residual tolerance of every iterative solve that goes
            through the sequence. ``None`` (default) selects
            :func:`mrx.precision.sqrt_eps` for the working precision
            (1.5e-8 in float64, 3.5e-4 in float32); an explicit value is used
            as given.
        maxiter : int, optional
            Maximum iteration count for iterative solvers.
        r_scale : float, optional
            Exponent used to cluster radial knots toward the axis
            (knot spacing proportional to ``r**r_scale``).
        knots : tuple of 3 optional array_like, optional
            Explicit FULL knot vectors per direction ``(T_r, T_θ, T_ζ)``;
            any entry ``None`` falls back to the default for that axis
            (radial: clamped-open padding of the ``r_scale``-graded
            breakpoints; angular: the :class:`SplineBasis` default for the
            axis type). The breakpoint/padding structure is implied by the
            axis regularity, so callers usually build e.g.
            ``T_r = [0]*p + breakpoints + [1]*p``. Lets multigrid coarse
            levels ANCHOR the first radial breakpoint (identical polar-core
            footprint across levels) instead of re-grading.
        n_inner : int, optional
            Number of inner CG iterations used by block preconditioners.
        betti_numbers : tuple of 4 ints, optional
            ``(b0, b1, b2, b3)`` for the physical domain. Determines how
            many harmonic ``k``-forms each Hodge Laplacian has, and hence
            the shapes of the nullspace arrays stored on
            :class:`SequenceOperators`. Defaults to ``(1, 1, 0, 0)`` which
            matches a solid torus.

        Notes
        -----
        Everything static is built here. The geometry is installed by
        :meth:`set_map` or :meth:`set_spline_map`, the preconditioners by
        :meth:`build_preconditioners`.
        """
        self.ns = tuple(ns)
        self.ps = tuple(ps)
        self.polar = bool(polar)
        self.tol = mrx.sqrt_eps() if tol is None else tol
        self.maxiter = maxiter
        self.n_inner = n_inner
        self.geometry = None
        self.mass_apply = None
        self.projection_apply = None
        self.operators = None
        assert len(betti_numbers) == 4, "betti_numbers must have length 4"
        self.betti_numbers = tuple(betti_numbers)
        Ts = list(knots) if knots is not None else [None] * 3
        if len(Ts) != 3:
            raise ValueError(f"knots must have 3 entries (got {len(Ts)})")
        for ax, T in enumerate(Ts):
            if T is not None:
                T = jnp.asarray(T, dtype=mrx.DTYPE)
                n_expected = ns[ax] + ps[ax] + 1
                if types[ax] == "clamped" and T.shape != (n_expected,):
                    raise ValueError(
                        f"knot vector for axis {ax} must have n+p+1 = "
                        f"{n_expected} entries (got shape {T.shape})")
                Ts[ax] = T
        if polar and Ts[0] is None:
            bp = jnp.linspace(0, 1, ns[0]-ps[0]+1)**r_scale
            Ts[0] = jnp.concatenate([jnp.zeros(ps[0]), bp, jnp.ones(ps[0])])

        self.basis_0, self.basis_1, self.basis_2, self.basis_3 = [
            DifferentialForm(i, ns, ps, types, Ts) for i in range(0, 4)
        ]
        self.quad = QuadratureRule(self.basis_0, q)

        bases = (self.basis_0, self.basis_1, self.basis_2, self.basis_3)
        self.xi = get_xi(ns[1]) if polar else None
        if polar:
            raw = [PolarExtractionOperator(L, self.xi, False) for L in bases]
            raw_dbc = [PolarExtractionOperator(L, self.xi, True) for L in bases]
        else:
            # Dirichlet conditions are supported in r only.
            raw = [BoundaryOperator(L, ('none', 'none', 'none')) for L in bases]
            raw_dbc = [BoundaryOperator(L, ('dirichlet', 'none', 'none')) for L in bases]
        for k in range(4):
            e, e_dbc = raw[k].build_extraction(), raw_dbc[k].build_extraction()
            setattr(self, f"e{k}", e)
            setattr(self, f"e{k}_dbc", e_dbc)
            setattr(self, f"e{k}_bc", bc_extraction_op(e, e_dbc, bases[k].n))
            setattr(self, f"n{k}", raw[k].n)
            setattr(self, f"n{k}_dbc", raw_dbc[k].n)
            setattr(self, f"n{k}_bc", raw[k].n - raw_dbc[k].n)
            for c in (1, 2, 3):
                setattr(self, f"n{k}_{c}", getattr(raw[k], f"n{c}"))
                setattr(self, f"n{k}_{c}_dbc", getattr(raw_dbc[k], f"n{c}"))

        # 1-D basis tables at the quadrature points, and the Greville data.
        for name, funcs, x in (("basis_r_jk", self.basis_0.Λ, self.quad.x_x),
                               ("basis_t_jk", self.basis_0.Λ, self.quad.x_y),
                               ("basis_z_jk", self.basis_0.Λ, self.quad.x_z),
                               ("d_basis_r_jk", self.basis_0.dΛ, self.quad.x_x),
                               ("d_basis_t_jk", self.basis_0.dΛ, self.quad.x_y),
                               ("d_basis_z_jk", self.basis_0.dΛ, self.quad.x_z)):
            f = funcs[("r", "t", "z").index(name.split("_")[-2][-1])]
            setattr(self, name, jax.vmap(jax.vmap(f, (0, None)), (None, 0))(x, f.ns))
        self.dd_basis_jk = _second_derivative_tables(self)
        self.greville = greville_axes(self)

        # Topological incidence: the raw stencils and, on polar sequences, the
        # analytic grad/curl corrections that make d.d = 0 exact on extracted
        # DoFs (div needs none: the V3 extraction is a 0/1 selection).
        for k in range(3):
            g, g_T = op.build_matrixfree_incidence(self, k)
            setattr(self, f"g{k}", g)
            setattr(self, f"g{k}_T", g_T)
        self.g0_grad = self.g1_curl = None
        if polar:
            pairs = [(din, dout) for din in (False, True) for dout in (False, True)]
            self.g0_grad = {pr: op.build_grad_stencil_g0(self, self.xi, *pr) for pr in pairs}
            self.g1_curl = {pr: op.build_curl_stencil_g1(self, self.xi, *pr) for pr in pairs}

    def load(self, f, k: int, dirichlet: bool = False, bc: bool = False,
             frame: str = 'phys'):
        """Assemble the dual k-form load vector  v_i = ∫ Λ^k_i · f(ξ) w(ξ) dξ.

        Parameters
        ----------
        f : callable
        k : int  Form degree (0, 1, 2, 3).
        dirichlet : bool  Use Dirichlet-constrained DOFs.
        bc : bool  Use boundary-trace DOFs (takes precedence over dirichlet).
        frame : {'phys', 'ref'}  Passed to :func:`mrx.projectors.load`.
        """
        return _load(self, f, k, dirichlet=dirichlet, bc=bc, frame=frame)

    def interpolate(self, f, k: int, dirichlet: bool = False,
                    frame: str = 'phys'):
        """Compute primal DOFs by Greville interpolation (k=0) or histopolation (k=1,2,3).

        Parameters
        ----------
        f : callable
        k : int  Form degree (0, 1, 2, 3).
        dirichlet : bool  Use Dirichlet-constrained DOFs.
        frame : {'phys', 'ref'}  Passed to :func:`mrx.projectors.interpolate`.
        """
        return _interpolate(self, f, k, dirichlet=dirichlet, frame=frame)

    @property
    def map(self):
        """The logical-to-physical map ``F`` of the installed geometry."""
        return self._require_geometry().map

    @property
    def metric_jkl(self):
        """Metric ``G = DF^T DF`` at the quadrature points, ``(n_q, 3, 3)``."""
        return self._require_geometry().metric_jkl

    @property
    def metric_inv_jkl(self):
        """Inverse metric at the quadrature points, ``(n_q, 3, 3)``."""
        return self._require_geometry().metric_inv_jkl

    @property
    def jacobian_j(self):
        """``det DF`` at the quadrature points, ``(n_q,)``."""
        return self._require_geometry().jacobian_j

    @property
    def null_0(self):
        """Harmonic 0-forms of the free space, ``(n_vectors, n0)``."""
        return get_nullspace(self._require_operators(), 0, False)

    @property
    def null_1(self):
        """Harmonic 1-forms of the free space, ``(n_vectors, n1)``."""
        return get_nullspace(self._require_operators(), 1, False)

    @property
    def null_2(self):
        """Harmonic 2-forms of the free space, ``(n_vectors, n2)``."""
        return get_nullspace(self._require_operators(), 2, False)

    @property
    def null_3(self):
        """Harmonic 3-forms of the free space, ``(n_vectors, n3)``."""
        return get_nullspace(self._require_operators(), 3, False)

    @property
    def null_0_dbc(self):
        """Harmonic 0-forms of the Dirichlet space, ``(n_vectors, n0_dbc)``."""
        return get_nullspace(self._require_operators(), 0, True)

    @property
    def null_1_dbc(self):
        """Harmonic 1-forms of the Dirichlet space, ``(n_vectors, n1_dbc)``."""
        return get_nullspace(self._require_operators(), 1, True)

    @property
    def null_2_dbc(self):
        """Harmonic 2-forms of the Dirichlet space, ``(n_vectors, n2_dbc)``."""
        return get_nullspace(self._require_operators(), 2, True)

    @property
    def null_3_dbc(self):
        """Harmonic 3-forms of the Dirichlet space, ``(n_vectors, n3_dbc)``."""
        return get_nullspace(self._require_operators(), 3, True)

    def set_geometry(self, geometry: SequenceGeometry):
        """Install a geometry and build the matrix-free mass and projection applies from it.

        Drops the operator bundle: every preconditioner and harmonic form on
        it was built for the previous metric, and a stale one would
        precondition the wrong operator silently (slow convergence, nothing
        else). Call :meth:`build_preconditioners` again.
        """
        self.geometry = geometry
        self.mass_apply = {k: build_matrixfree_mass_apply(self, k, geometry)
                           for k in range(4)}
        self.projection_apply = {pair: build_matrixfree_projection_apply(self, *pair)
                                 for pair in ((1, 2), (2, 1), (0, 3), (3, 0))}
        self.operators = None

    def build_preconditioners(self, *, ks=(0, 1, 2, 3), dirichlets=(False, True),
                              schur_jacobi=False):
        """Build every preconditioner of the installed geometry; install and return the bundle.

        A fresh :class:`~mrx.operators.SequenceOperators` with, for each
        ``k`` in ``ks`` and each BC in ``dirichlets``: the Jacobi mass
        diagonal, the metric-lumped mass atom and the metric-lumped Laplacian
        atom (the production preconditioners of every solve through the
        sequence), plus the closed-form ``k = 0`` Jacobi Laplacian diagonal
        (O(N); the shifted scalar solve of the nullspace iteration uses it).
        ``schur_jacobi=True`` also probes the Schur diagonals that
        ``schur.outer='jacobi'`` needs (the comparison baseline and the
        shift-and-invert nullspace route; O(n_k) applies per pair, so off by
        default). The ``k >= 1`` Jacobi Laplacian diagonals are a comparison
        baseline production never applies; build them with
        :func:`~mrx.operators.assemble_laplacian_jacobi_preconditioner`.

        Nothing on the bundle is built anywhere else, and nothing on it
        survives a geometry change: after :meth:`set_map` call this again, and
        recompute the harmonic forms (:meth:`compute_nullspaces`), which live
        on the bundle too. That is the contract for an outer loop over
        geometries.

        Building a sequence WITHOUT preconditioners is a first-class path:
        for purely geometrical work ``set_map`` alone is the whole setup.

        NEEDS ``n >= p + 2`` for the Laplacian atoms (see
        :func:`~mrx.operators.assemble_metric_lumping_laplacian_preconditioner`).
        """
        self._require_geometry()
        ks = tuple(int(v) for v in ks)
        dirichlets = tuple(bool(v) for v in dirichlets)
        ops = op.new_operators(self)
        ops = op.assemble_mass_jacobi_preconditioner(self, ops, ks=ks)
        ops = op.assemble_mass_metric_lumping_preconditioner(
            self, ops, ks=ks, dirichlet_variants=dirichlets)
        ops = op.assemble_laplacian_jacobi_preconditioner(
            self, ops, ks=tuple(k for k in ks if k == 0), dirichlets=dirichlets)
        ops = op.assemble_metric_lumping_laplacian_preconditioner(
            self, ops, ks=ks, dirichlets=dirichlets)
        if schur_jacobi:
            ops = op.assemble_schur_jacobi_preconditioner(
                self, ops, ks=tuple(k for k in ks if k >= 1),
                dirichlet_variants=dirichlets)
        self.operators = ops
        return ops

    def set_map_and_preconditioners(self, map, *, ks=(0, 1, 2, 3),
                                    dirichlets=(False, True), schur_jacobi=False):
        """:meth:`set_map` followed by :meth:`build_preconditioners`, nothing else."""
        self.set_map(map)
        return self.build_preconditioners(ks=ks, dirichlets=dirichlets,
                                          schur_jacobi=schur_jacobi)

    def _require_geometry(self):
        """Return the attached geometry or raise when none is installed."""
        geometry = getattr(self, 'geometry', None)
        if geometry is None:
            raise ValueError(
                'Set the geometry first, for example with seq.set_map(...) '
                'or seq.set_spline_map(...).')
        return geometry

    def get_operators(self):
        """The installed operator bundle, or ``None``."""
        return self.operators

    def set_operators(self, operators):
        """Install an operator bundle (for example one with nullspaces computed on it)."""
        self.operators = operators
        return operators

    def _require_operators(self, operators=None):
        """``operators`` if given, else the installed bundle; raises when there is none."""
        if operators is not None:
            return operators
        if self.operators is None:
            raise ValueError(
                'no operator bundle: call seq.build_preconditioners() after set_map')
        return self.operators

    def set_map(self, map):
        """Update the active logical-to-physical map and derived geometry terms."""
        self.set_geometry(SequenceGeometry.from_map(map, self.quad.x))

    def build_spline_map(self, coefficients, extraction=None):
        """Build a spline map using the sequence's scalar spline basis."""
        from mrx.mappings import SplineMap

        if extraction is None:
            extraction = self.e0
            extraction_T = self.e0.T
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

        Requires the sequence's extraction operators (or the relevant
        ``assemble_M{k}`` call) to have been called first.

        Parameters
        ----------
        g : array of shape (n_k_bc,)
        k : int

        Returns
        -------
        array of shape (n_k_dbc,)
        """
        m_sp = getattr(self, f'm{k}')
        e_dbc = getattr(self, f'e{k}_dbc')
        e_bc_T = getattr(self, f'e{k}_bc_T')
        return e_dbc @ (m_sp @ (e_bc_T @ g))

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

    def l2_norm_sq(self, v, k, dirichlet=True):
        """Return the squared L² norm of a k-form DOF vector ``v``."""
        return v @ self.apply_mass_matrix(v, k, dirichlet=dirichlet)

    def l2_norm(self, v, k, dirichlet=True):
        """Return the L² norm of a k-form DOF vector ``v``."""
        return jnp.sqrt(self.l2_norm_sq(v, k, dirichlet=dirichlet))

    def apply_incidence_matrix(self, v, k, dirichlet_in=True, dirichlet_out=True,
                               transpose=False):
        """Apply the topological exterior-derivative incidence Gk to ``v``.

        Gk has entries in {-1, 0, +1} and is geometry-independent. On DoF
        spaces where the extraction operators are "unitary" (``e @ e^T = I``),
        this equals ``M_{k+1}^{-1} @ apply_derivative_matrix``. For non-unitary
        extractions (e.g. polar axis gluing) the two differ.

        PREFER THIS FORM. Until 2026-08-25 this docstring said the
        mass-projected :meth:`apply_strong_grad` / curl / div "should be
        preferred when exact d∘d = 0 on extracted DoFs is required". That is
        stale: :func:`~mrx.operators.apply_incidence_matrix` applies the cached
        coefficient-Gram correction ``G = Gram_{k+1}^{-1} (E_out^T sp E_in)``,
        which makes this form the true strong derivative on polar sequences
        too. Measured on quasr44970 ns=(8,16,8) p=3:

            div.curl, mass-projected   1.261e-10
            div.curl, incidence        8.641e-16   (machine zero)
            curl agreement between them 1.025e-12   (so the swap is free)

        The mass-projected path also costs a Krylov solve per apply. The stale
        advice cost a relaxation study a spurious 1e-10 "floor" on div B that
        was read as a property of the discretisation rather than of the
        operator choice.
        """
        return op.apply_incidence_matrix(
            self, v, k,
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
        return op.apply_mass_matrix_preconditioner(
            self, operators, v, k, dirichlet=dirichlet, kind=kind)

    def apply_inverse_mass_matrix(self, rhs, k, dirichlet=True, guess=None,
                                  operators=None, tol=None, maxiter=None,
                                  preconditioner='auto',
                                  return_info=False):
        """
        Apply the inverse mass matrix Mk⁻¹ for k-forms to a right-hand side,
        solved via CG with a structured mass preconditioner. An optional initial
        guess can be provided to warm-start the solver.
        """
        operators = self._require_operators(operators)
        return op.apply_inverse_mass_matrix(
            self, operators, rhs, k,
            dirichlet=dirichlet, guess=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter,
            preconditioner=preconditioner,
            return_info=return_info)

    def apply_mass_matrix(self, v, k, dirichlet=True):
        """
        Apply the (matrix-free) mass matrix Mk for k-forms to a vector v:
            k=0: M0_ij = ∫ Λ0_i Λ0_j det DF dx
            k=1: M1_ij = ∫ Λ1_i · G⁻¹ Λ1_j det DF dx
            k=2: M2_ij = ∫ Λ2_i · G Λ2_j (det DF)⁻¹ dx
            k=3: M3_ij = ∫ Λ3_i Λ3_j (det DF)⁻¹ dx
        """
        return op.apply_mass_matrix(
            self, v, k, dirichlet=dirichlet)

    def apply_projection_matrix(self, v, k_in, k_out, dirichlet_in=True, dirichlet_out=True):
        """
        Apply the (matrix-free) projection mass Pk_in_k_out to a vector v.
        """
        return op.apply_projection_matrix(
            self, v, k_in, k_out,
            dirichlet_in=dirichlet_in,
            dirichlet_out=dirichlet_out,
        )

    def apply_derivative_matrix(self, v, k, dirichlet_in=True, dirichlet_out=True,
                                transpose=False):
        """Apply the weak derivative ``D_k`` (k-forms to (k+1)-forms) to ``v``::

            k=0: D0_ij = ∫ Λ1_i · G⁻¹ grad Λ0_j det DF dx  (grad)
            k=1: D1_ij = ∫ Λ2_i · G curl Λ1_j (det DF)⁻¹ dx  (curl)
            k=2: D2_ij = ∫ Λ3_i div Λ2_j (det DF)⁻¹ dx  (div)

        If ``transpose=True``, apply ``D_k^T`` instead ((k+1)-forms to k-forms).
        """
        return op.apply_derivative_matrix(
            self, v, k,
            dirichlet_in=dirichlet_in,
            dirichlet_out=dirichlet_out,
            transpose=transpose,
        )

    def apply_laplacian(self, v, k, dirichlet=True, operators=None):
        """Apply the k-form Laplacian ``L_k`` to a vector ``v``.

        Naming and structure used throughout MRX:

        - ``S_k`` is the k-form stiffness block,
        - ``L_k = S_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T``,
        - equivalently
          ``L_k = G_k^T M_{k+1} G_k + M_k G_{k-1} M_{k-1}^{-1} G_{k-1}^T M_k``.

        For k >= 1 this is applied through the Schur form above. For k = 0,
        ``L_0 = S_0``.
        """
        operators = self._require_operators(operators)
        return op.apply_laplacian(
            self, operators, v, k, dirichlet=dirichlet,
            tol=self.tol, maxiter=self.maxiter)

    def apply_laplacian_approx(self, v, k, dirichlet=True, operators=None):
        """Linear approximate Laplacian apply.

        Replaces ``M_{k-1}^{-1}`` in the Schur term with a single configured
        mass-preconditioner apply. Linear, SPD, safe to nest inside Krylov
        solvers and to use as a preconditioner.  Not exact unless the metric
        is tensor-separable on the reference domain.
        """
        operators = self._require_operators(operators)
        return op.apply_laplacian_approx(
            self, operators, v, k, dirichlet=dirichlet)

    def apply_mass_plus_eps_laplace_matrix(self, v, k, eps, dirichlet=True, operators=None):
        """Apply ``(M_k + eps * L_k)`` to a k-form vector."""
        return self.apply_mass_matrix(
            v, k, dirichlet=dirichlet) \
            + eps * self.apply_laplacian(
                v, k, dirichlet=dirichlet, operators=operators)

    def apply_stiffness(self, v, k, dirichlet=True):
        """
        Apply the stiffness matrix S_k to a k-form vector v.

            k=0: grad-grad
            k=1: curl-curl
            k=2: div-div
            k=3: 0 (no stiffness)
        """
        return op.apply_stiffness(
            self, v, k, dirichlet=dirichlet)

    def _get_nullspace(self, k, dirichlet):
        """Return the nullspace basis for the k-form Laplacian."""
        return get_nullspace(self._require_operators(), k, dirichlet)

    def _get_saddle_point_nullspaces(self, k, dirichlet):
        """Return the pair of nullspace bases for the k-th saddle-point system."""
        return get_saddle_point_nullspaces(
            self, self._require_operators(), k, dirichlet)

    def apply_inverse_laplacian(self, rhs, k, dirichlet=True, guess=None,
                                operators=None, tol=None, maxiter=None,
                                preconditioner='auto',
                                return_info=False):
        """Apply the inverse of the k-form Laplacian to a right-hand side."""
        operators = self._require_operators(operators)
        return op.apply_inverse_laplacian(
            self, operators, rhs, k,
            dirichlet=dirichlet, guess=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter,
            preconditioner=preconditioner,
            return_info=return_info)

    def apply_inverse_shifted_laplacian(self, rhs, k, eps, dirichlet=True, guess=None,
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
        return op.apply_inverse_shifted_laplacian(
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
        defaults as the other inverse paths (``'auto'`` resolves to the
        production metric-lumping kind, ``'jacobi'`` is the fallback).
        """
        operators = self._require_operators(operators)
        return op.apply_inverse_mass_plus_eps_laplace_matrix(
            self, operators, rhs, k, eps,
            dirichlet=dirichlet, guess=guess,
            tol=self.tol if tol is None else tol,
            maxiter=self.maxiter if maxiter is None else maxiter,
            preconditioner=preconditioner,
            return_info=return_info)

    def apply_laplacian_preconditioner(self, v, k, dirichlet=True,
                                       operators=None, kind='auto'):
        """
        Apply a preconditioner for the k-form Laplacian to a vector ``v``.

        ``kind`` selects between ``'none'`` (identity), ``'jacobi'`` (per-DoF
        diagonal; for k >= 1 its weak half is a Kronecker mass MODEL),
        and ``'metric_lumping'`` (the metric-lumped block-Jacobi atom, k = 0..3, free
        and Dirichlet — the production preconditioner; call
        :func:`~mrx.operators.assemble_metric_lumping_laplacian_preconditioner`
        first).

        ``'auto'`` (the default) uses ``'metric_lumping'`` when it has been assembled
        for this ``(k, BC)`` and falls back to ``'jacobi'`` otherwise. It used
        to resolve to ``'jacobi'`` unconditionally while claiming to prefer
        ``'tensor'`` at k = 0; ``'tensor'`` itself was deleted 2026-08-25.
        """
        operators = self._require_operators(operators)
        return op.apply_laplacian_preconditioner(
            self, operators, v, k, dirichlet=dirichlet, kind=kind)

    def _compute_nullspaces(self, betti_numbers=None, eps=None, direct=False,
                            **kwargs):
        """Compute harmonic forms and store them on ``self.operators``.

        ``betti_numbers`` defaults to ``self.betti_numbers``.

        ``direct=True`` selects the Hodge-decomposition construction
        (:func:`~mrx.nullspace.compute_nullspaces`): a fixed pair of Hodge
        solves per form, with no shift, no outer iteration and no spectral-gap
        assumption.  It is only self-sufficient when ``b2 == 0`` -- which
        covers the toroidal geometries we actually run -- and raises with an
        explanation otherwise, because at ``b2 > 0`` the two constructions
        each need the other's kernel for deflation.  ``eps`` is unused in that
        mode.

        ``direct=False`` (default) uses shift-and-invert inverse iteration
        (:func:`~mrx.nullspace.compute_nullspaces_iterative`), which works for
        any topology because the shift removes the singularity.

        Returns the info dict for the iterative route, or ``None`` for the
        direct one (which has no per-vector iteration counts to report).
        """
        if direct:
            self.operators = compute_nullspaces(
                self, self._require_operators(), betti_numbers=betti_numbers)
            return None
        if eps is not None:
            kwargs["eps"] = eps
        operators, info = compute_nullspaces_iterative(
            self, self._require_operators(),
            betti_numbers=betti_numbers, **kwargs)
        self.operators = operators
        return info

    def _find_nullspace_vectors(self, k, n_vectors, eps, dirichlet=True):
        """Find ``n_vectors`` nullspace vectors of the k-form Laplacian via inverse iteration."""
        return find_nullspace_vectors(
            self, self._require_operators(), k, n_vectors, eps, dirichlet)

    def compute_nullspaces(self, betti_numbers=None):
        """Cache the harmonic forms via the direct Hodge-decomposition route.

        Thin wrapper over :meth:`_compute_nullspaces` with ``direct=True``;
        raises on topologies where that route is circular (``b2 > 0``).
        """
        self.operators = compute_nullspaces(
            self, self._require_operators(), betti_numbers=betti_numbers)
        return self.operators

    def init_nullspaces(self, betti_numbers=None):
        """Initialise zero-valued nullspace arrays on ``self.operators``.

        Shapes are derived from ``betti_numbers`` (or ``self.betti_numbers``).
        """
        self.operators = init_nullspaces(
            self, self._require_operators(), betti_numbers=betti_numbers)
        return self.operators

    def evaluate_at_quadrature(self, dofs, k, dirichlet=True):
        """Evaluate a 1- or 2-form at the quadrature points.

        Args:
            dofs: DOF vector of the k-form.
            k: Form degree, 1 or 2.
            dirichlet: Use the Dirichlet-constrained extraction.

        Returns:
            Array of shape ``(n_q, 3)``: the reference components at every
            quadrature point, in the sequence's flat quadrature order.
        """
        from mrx.quadrature import evaluate_at_xq
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        match k:
            case 1:
                e_T = self.e1_dbc.T if dirichlet else self.e1.T
            case 2:
                e_T = self.e2_dbc.T if dirichlet else self.e2.T
            case _:
                raise ValueError("k must be 1 or 2")
        comp_info, comp_shapes = self._form_comp_info(k)
        return evaluate_at_xq(e_T @ dofs, comp_info, comp_shapes, quad_shape, 3)

    def cross_product_load(
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
        evaluation and integration.  This is
        :meth:`evaluate_at_quadrature` on both inputs followed by
        :meth:`cross_product_load_values`; call those directly to reuse the
        quadrature values of an input.

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
        w_jk = self.evaluate_at_quadrature(w, m, dirichlet_m)
        u_jk = self.evaluate_at_quadrature(u, k, dirichlet_k)
        return self.cross_product_load_values(w_jk, u_jk, n, m, k, dirichlet_n)

    def cross_product_load_values(self, w_jk, u_jk, n, m, k, dirichlet_n=True):
        """Integrate ``Λⁿ_i · (w × u)`` from quadrature values of ``w`` and ``u``.

        Args:
            w_jk: Reference components of the m-form at the quadrature points,
                shape ``(n_q, 3)`` (see :meth:`evaluate_at_quadrature`).
            u_jk: The same for the k-form.
            n: Form degree of the output (1 or 2).
            m: Form degree of ``w`` (1 or 2).
            k: Form degree of ``u`` (1 or 2).
            dirichlet_n: Use the Dirichlet-constrained extraction for the
                output.

        Returns:
            The n-form dual DOF vector.
        """
        from mrx.quadrature import integrate_against
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        match n:
            case 1:
                en = self.e1_dbc if dirichlet_n else self.e1
            case 2:
                en = self.e2_dbc if dirichlet_n else self.e2
            case _:
                raise ValueError("n must be 1 or 2")
        comp_info_n, comp_shapes_n = self._form_comp_info(n)

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

        return en @ integrate_against(
            f_jk, comp_info_n, comp_shapes_n, quad_shape)

    def pressure_load(
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
        from mrx.quadrature import evaluate_at_xq, integrate_against
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)

        types = self.basis_0.types
        grad_r = grad_1d(self.d_basis_r_jk, types[0])
        grad_t = grad_1d(self.d_basis_t_jk, types[1])
        grad_z = grad_1d(self.d_basis_z_jk, types[2])

        # --- evaluate p at quad points (0-form, 1 component) ---
        ep_T = self.e0_dbc.T if dirichlet_p else self.e0.T
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
        eu_T = self.e2_dbc.T if dirichlet_u else self.e2.T
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

    def apply_leray_projection(self, v, k=2, p_guess=None, dirichlet_p=False):
        """
        Apply the Leray projection to a 1 or 2-form v.

        When k = 2 (Dirichlet complex, ``v`` in the k=2 space with
        ``v . n = 0``):
            Solves the system (k=3 Hodge Laplacian):
            div v = div σ
            (σ, ω) = -(p, div ω) ∀ω 2-forms
            -> div(v - σ) = 0 and σ.n = 0 on the boundary.
        When k = 1 (``v`` in the natural 1-form space, no boundary
        condition):
            Solves the k=0 Laplacian
            (grad p, grad ω) = (v, grad ω) ∀ω in the scalar space,
            so ``v - grad p`` is weakly divergence-free. The scalar space
            fixes the boundary condition of the decomposition:
            ``dirichlet_p=False``: ``p`` in the natural k=0 space, the
            constants deflated; ``(v - grad p) . n = 0`` on the boundary,
            i.e. ``dp/dn = v . n`` (the Neumann problem).
            ``dirichlet_p=True``: ``p`` in the Dirichlet k=0 space,
            ``p = 0`` on the boundary; ``v - grad p`` keeps its normal
            trace (the Dirichlet problem). This is the weak pressure of
            :func:`mrx.relaxation.weak_pressure`.

        Parameters
        ----------
        v : jnp.ndarray 
            The vector form DoFs
        k : int
            The degree of the vector form
        p_guess : jnp.ndarray 
            Guess for pressure form DoFs
        dirichlet_p : bool
            k = 1 only: solve the multiplier in the Dirichlet k=0 space.
            k = 2 raises: its multiplier is the 3-form of the Dirichlet
            complex, and there is no natural variant.

        Returns
        -------
        v_out : jnp.ndarray 
            divergence-cleaned v
        p : jnp.ndarray 
            The pressure form DoFs

        """
        # SIGN CONVENTION (fixed 2026-08-14): both branches remove the
        # gradient part as sigma = -grad(q) with q the solved multiplier, so
        # q = -(physical pressure) + gauge. The RETURNED p is negated to be
        # the physical pressure multiplier (v_out = v - grad p; at MHD
        # equilibrium J x B = grad p this recovers +p, verified against the
        # analytic z-pinch, 2026-08-14). Warm starts arrive in
        # the returned (physical) convention and are negated back on entry.
        # The gauge of p is solver-defined (a constant offset) for the k=2
        # and the k=1 natural branches; the k=1 Dirichlet branch has none.
        if k == 2:
            if dirichlet_p:
                raise ValueError("dirichlet_p selects the k=1 scalar space; "
                                 "the k=2 multiplier is always the Dirichlet 3-form")
            p_guess = jnp.zeros(self.n3_dbc) if p_guess is None else p_guess
            # Assumes dirichlet == True on all spaces.
            div_v = self.apply_derivative_matrix(
                v, 2, dirichlet_in=True, dirichlet_out=True)
            q = self.apply_inverse_laplacian(
                div_v, 3, dirichlet=True, guess=-p_guess)
            σ = -self.apply_weak_grad(q, True, True)
            return v - σ, -q
        elif k == 1:
            # v lives in the natural 1-form space; only the scalar space
            # (test functions AND multiplier) carries the boundary condition.
            n_p = self.n0_dbc if dirichlet_p else self.n0
            p_guess = jnp.zeros(n_p) if p_guess is None else p_guess
            div_v = -self.apply_derivative_matrix(
                v, 0, dirichlet_in=dirichlet_p, dirichlet_out=False, transpose=True)
            q = self.apply_inverse_laplacian(
                div_v, 0, dirichlet=dirichlet_p, guess=-p_guess)
            σ = -self.apply_strong_grad(q, dirichlet_p, False)
            return v - σ, -q
