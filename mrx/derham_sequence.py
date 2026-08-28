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
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.differential_forms import DifferentialForm
from mrx.extraction_operators import (PolarExtractionOperator,
                                      bc_extraction_op, get_xi)
from mrx.nullspace import (compute_nullspaces, compute_nullspaces_iterative,
                           get_nullspace)
import mrx.operators as op
from mrx.mass import build_matrixfree_mass_apply, build_matrixfree_projection_apply
from mrx.projectors import greville_axes, load as _load, interpolate as _interpolate
from mrx.quadrature import QuadratureRule
from mrx.spline_bases import basis_derivative_table, basis_table
from mrx.geometry import SequenceGeometry





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
    xi : jnp.ndarray
        Polar extraction weights ``(3, 2, n_θ)`` (:func:`~mrx.extraction_operators.get_xi`).
    extraction : dict
        ``E(k, dirichlet)`` -- the extraction operators mapping constrained
        DoF vectors to the full spline basis, free and with homogeneous
        Dirichlet conditions at the radial boundary (``.T`` is the
        transpose); ``E_bc(k)`` the extraction of the boundary DoFs;
        ``n(k, dirichlet)`` / ``n_bc(k)`` the DoF counts.
    g0, g1, g2 : _MatrixFreeIncidence
        Raw {-1, 0, +1} incidence stencils (grad, curl, div) and their
        transposes ``g*_T``.
    g0_grad, g1_curl : dict
        Analytic polar grad/curl stencils on extracted DoFs, keyed
        ``(dirichlet_in, dirichlet_out)``.
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
            Must be ``True``: the sequence is built for a geometry with a
            polar axis at ``r = 0`` (C¹ polar extraction there). A
            non-polar tensor-product sequence is not supported.
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
        if not polar:
            raise ValueError(
                "polar=False is not supported: every MRX geometry has a polar axis "
                "at r = 0 and the extraction, incidence and preconditioners are "
                "built for it. A tensor-product (non-polar) sequence would need "
                "its own selection extraction; nothing in the code base uses one.")
        self.ns = tuple(ns)
        self.ps = tuple(ps)
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
        self.xi = get_xi(ns[1])
        raw = [PolarExtractionOperator(L, self.xi, False) for L in bases]
        raw_dbc = [PolarExtractionOperator(L, self.xi, True) for L in bases]
        self.extraction, self.boundary_extraction = {}, {}
        self.n_dofs, self.n_boundary = {}, {}
        for k in range(4):
            e, e_dbc = raw[k].build_extraction(), raw_dbc[k].build_extraction()
            self.extraction[(k, False)], self.extraction[(k, True)] = e, e_dbc
            self.boundary_extraction[k] = bc_extraction_op(e, e_dbc, bases[k].n)
            self.n_dofs[(k, False)], self.n_dofs[(k, True)] = raw[k].n, raw_dbc[k].n
            self.n_boundary[k] = raw[k].n - raw_dbc[k].n

        # 1-D basis tables at the quadrature points, and the Greville data.
        for name, funcs, x in (("basis_r_jk", self.basis_0.Λ, self.quad.x_x),
                               ("basis_t_jk", self.basis_0.Λ, self.quad.x_y),
                               ("basis_z_jk", self.basis_0.Λ, self.quad.x_z),
                               ("d_basis_r_jk", self.basis_0.dΛ, self.quad.x_x),
                               ("d_basis_t_jk", self.basis_0.dΛ, self.quad.x_y),
                               ("d_basis_z_jk", self.basis_0.dΛ, self.quad.x_z)):
            f = funcs[("r", "t", "z").index(name.split("_")[-2][-1])]
            setattr(self, name, basis_table(f, x))
        # d/dx of the DERIVATIVE-basis tables (the metric-lumping stiffness
        # profiles): one order deeper than anything else the sequence tabulates.
        self.dd_basis_jk = tuple(basis_derivative_table(self.basis_0.dΛ[a], x)
                                 for a, x in enumerate((self.quad.x_x, self.quad.x_y, self.quad.x_z)))
        self.greville = greville_axes(self)

        # Topological incidence: the raw stencils and the analytic polar
        # grad/curl corrections that make d.d = 0 exact on extracted DoFs
        # (div needs none: the V3 extraction is a 0/1 selection).
        for k in range(3):
            g, g_T = op.build_matrixfree_incidence(self, k)
            setattr(self, f"g{k}", g)
            setattr(self, f"g{k}_T", g_T)
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

    def E(self, k, dirichlet=False):
        """The extraction of the free (default) or Dirichlet ``k``-form space; ``.T`` is its transpose."""
        return self.extraction[(int(k), bool(dirichlet))]

    def E_bc(self, k):
        """The extraction of the boundary DoFs of the ``k``-form space (in ``E(k)``, not in ``E(k, True)``)."""
        return self.boundary_extraction[int(k)]

    def n(self, k, dirichlet=False):
        """Number of DoFs of the free (default) or Dirichlet ``k``-form space."""
        return self.n_dofs[(int(k), bool(dirichlet))]

    def n_bc(self, k):
        """Number of boundary DoFs of the ``k``-form space."""
        return self.n_boundary[int(k)]

    def nullspace(self, k, dirichlet=False):
        """Harmonic ``k``-forms of the free or Dirichlet space, ``(n_vectors, n_k)``."""
        return get_nullspace(self._require_operators(), k, dirichlet)

    def set_geometry(self, geometry: SequenceGeometry):
        """Install a geometry and build the matrix-free mass and projection applies from it.

        Refuses a folded map (``det DF <= 0`` or non-finite at any quadrature
        point). Drops the operator bundle: every preconditioner and harmonic form on
        it was built for the previous metric, and a stale one would
        precondition the wrong operator silently (slow convergence, nothing
        else). Call :meth:`build_preconditioners` again.
        """
        jac = np.asarray(geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0.0:
            raise ValueError(
                f"the map folds: det DF at the quadrature points spans "
                f"[{jac.min():.3e}, {jac.max():.3e}] and must be positive")
        self.geometry = geometry
        self.mass_apply = {k: build_matrixfree_mass_apply(self, k, geometry)
                           for k in range(4)}
        self.projection_apply = {pair: build_matrixfree_projection_apply(self, *pair)
                                 for pair in ((1, 2), (2, 1), (0, 3), (3, 0))}
        self.operators = None

    def build_preconditioners(self, *, ks=(0, 1, 2, 3), dirichlets=(False, True),
                              jacobi=False, bc_scale=None):
        """Build the preconditioners of the installed geometry; install and return the bundle.

        A fresh :class:`~mrx.operators.SequenceOperators` with, for each
        ``k`` in ``ks`` and each BC in ``dirichlets``, the metric-lumped mass
        atom and the metric-lumped Laplacian atom -- the preconditioners of
        every solve through the sequence (``kind='auto'`` resolves to them
        everywhere: mass, Laplacian, saddle, diffusion and the shifted solves
        of the nullspace iteration) -- and zero nullspaces. ``jacobi=True``
        also probes the Jacobi option onto the bundle -- ``1/diag`` of
        ``E M_k E^T``, of ``L_k`` and of the saddle solves' approximate Schur
        operator, ``O(n_k)`` applies each -- for ``kind='jacobi'`` and
        ``schur.outer='jacobi'``.

        Nothing on the bundle is built anywhere else, and nothing on it
        survives a geometry change: after :meth:`set_map` call this again, and
        recompute the harmonic forms (:meth:`compute_nullspaces`), which live
        on the bundle too. That is the contract for an outer loop over
        geometries.

        ``bc_scale`` overrides the natural-BC penalty scale of the Laplacian
        atoms (``metric_lumping_laplacian.PRODUCTION_BC_SCALE = 3.0``, a
        measured balance point); ``None`` keeps it.

        Building a sequence WITHOUT preconditioners is a first-class path:
        for purely geometrical work ``set_map`` alone is the whole setup.

        NEEDS ``n >= p + 2`` for the Laplacian atoms (see
        :func:`~mrx.operators.assemble_metric_lumping_laplacian_preconditioner`).
        """
        self._require_geometry()
        ks = tuple(int(v) for v in ks)
        dirichlets = tuple(bool(v) for v in dirichlets)
        ops = op.new_operators(self)
        ops = op.assemble_mass_metric_lumping_preconditioner(
            self, ops, ks=ks, dirichlet_variants=dirichlets)
        ops = op.assemble_metric_lumping_laplacian_preconditioner(
            self, ops, ks=ks, dirichlets=dirichlets,
            **({} if bc_scale is None else {"bc_scale": bc_scale}))
        if jacobi:
            ops = op.assemble_jacobi_preconditioners(self, ops, ks=ks, dirichlets=dirichlets)
        self.operators = ops
        return ops

    def set_map_and_preconditioners(self, map, *, ks=(0, 1, 2, 3), dirichlets=(False, True),
                                    jacobi=False):
        """:meth:`set_map` followed by :meth:`build_preconditioners`, nothing else."""
        self.set_map(map)
        return self.build_preconditioners(ks=ks, dirichlets=dirichlets, jacobi=jacobi)

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

    def build_spline_map(self, coefficients):
        """A :class:`~mrx.mappings.SplineMap` in the sequence's scalar spline basis."""
        from mrx.mappings import SplineMap

        return SplineMap(
            coefficients=coefficients,
            extraction=self.E(0),
            extraction_T=self.E(0).T,
            basis_0=self.basis_0,
        )

    def geometry_from_spline_map(self, coefficients):
        """Geometry data from spline map coefficients, by the sum-factorised path."""
        return SequenceGeometry.from_spline_map(self.build_spline_map(coefficients), self)

    def set_spline_map(self, coefficients):
        """Install the geometry of a spline map given by its coefficients."""
        self.set_geometry(self.geometry_from_spline_map(coefficients))

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
        e_bc_T = self.E_bc(k).T
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
        e_dbc = self.E(k, True)
        e_bc_T = self.E_bc(k).T
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
        """The exterior derivative on coefficients: ``G_k`` (k-forms to (k+1)-forms).

        For grad and curl the analytic polar stencils, the exact strong
        derivative on extracted DoFs (``d.d = 0`` to round-off); for div the
        {-1, 0, +1} difference stencil through the (0/1) V3 extraction. No
        geometry and no solve. ``transpose=True`` applies ``G_k^T``.
        """
        return op.apply_incidence_matrix(
            self, v, k,
            dirichlet_in=dirichlet_in,
            dirichlet_out=dirichlet_out,
            transpose=transpose,
        )

    def apply_strong_grad(self, v, dirichlet_in=True, dirichlet_out=True):
        """The strong gradient of a 0-form: the incidence ``G_0``, exact, no solve."""
        return self.apply_incidence_matrix(v, 0, dirichlet_in=dirichlet_in,
                                           dirichlet_out=dirichlet_out)

    def apply_strong_curl(self, v, dirichlet_in=True, dirichlet_out=True):
        """The strong curl of a 1-form: the incidence ``G_1``, exact, no solve."""
        return self.apply_incidence_matrix(v, 1, dirichlet_in=dirichlet_in,
                                           dirichlet_out=dirichlet_out)

    def apply_strong_div(self, v, dirichlet_in=True, dirichlet_out=True):
        """The strong divergence of a 2-form: the incidence ``G_2``, exact, no solve."""
        return self.apply_incidence_matrix(v, 2, dirichlet_in=dirichlet_in,
                                           dirichlet_out=dirichlet_out)

    def apply_weak_grad(self, v, dirichlet_in=True, dirichlet_out=True):
        """The weak gradient of a 3-form: ``-M_2^{-1} D_2^T v`` (the codifferential; one mass solve)."""
        dv_dual = -self.apply_derivative_matrix(
            v, 2, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        return self.apply_inverse_mass_matrix(dv_dual, 2, dirichlet=dirichlet_out)

    def apply_weak_curl(self, v, dirichlet_in=True, dirichlet_out=True):
        """The weak curl of a 2-form: ``M_1^{-1} D_1^T v`` (the codifferential; one mass solve)."""
        dv_dual = self.apply_derivative_matrix(
            v, 1, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
        return self.apply_inverse_mass_matrix(dv_dual, 1, dirichlet=dirichlet_out)

    def apply_weak_div(self, v, dirichlet_in=True, dirichlet_out=True):
        """The weak divergence of a 1-form: ``-M_0^{-1} D_0^T v`` (the codifferential; one mass solve)."""
        dv_dual = -self.apply_derivative_matrix(
            v, 0, dirichlet_in=dirichlet_in, dirichlet_out=dirichlet_out, transpose=True)
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
        """The dual derivative ``D_k = M_{k+1} G_k`` (k-forms to dual (k+1)-forms).

        The incidence followed by the mass, i.e. the exterior derivative
        tested against the (k+1)-form basis::

            k=0: D0_ij = ∫ Λ1_i · G⁻¹ grad Λ0_j det DF dx  (grad)
            k=1: D1_ij = ∫ Λ2_i · G curl Λ1_j (det DF)⁻¹ dx  (curl)
            k=2: D2_ij = ∫ Λ3_i div Λ2_j (det DF)⁻¹ dx  (div)

        No solve. ``transpose=True`` applies ``D_k^T = G_k^T M_{k+1}`` (dual
        (k+1)-forms to dual k-forms), the building block of the weak
        derivatives and of the Laplacian.
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
        """The linear approximation of ``L_k`` with ``M_{k-1}^{-1}`` replaced by one mass-preconditioner apply.

        Linear and SPD, so it can sit inside a Krylov solve: it is the
        operator the metric-lumped Laplacian atoms probe their polar core
        from (:class:`~mrx.metric_lumping_laplacian.MetricLumpingLaplacian`).
        Not ``L_k`` itself unless the metric is tensor-separable.
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

    def apply_inverse_laplacian(self, rhs, k, dirichlet=True, guess=None,
                                operators=None, tol=None, maxiter=None,
                                preconditioner='auto',
                                return_info=False):
        """Solve ``L_k x = rhs`` for the k-form ``x``.

        ``k = 0``: ``L_0 = S_0`` is SPD up to its harmonic forms; deflated
        CG (:func:`~mrx.solvers.solve_singular_cg`), the harmonic forms of
        the bundle projected out, the metric-lumped k=0 atom as
        preconditioner.

        ``k >= 1``: the symmetric saddle system in ``(x, sigma)``::

            | S_k        D_{k-1} | | x     |   | rhs |
            | D_{k-1}^T  -M_{k-1} | | sigma | = | 0   |

        whose Schur complement is ``L_k``, by MINRES
        (:func:`~mrx.solvers.solve_saddle_point_minres`) with the block
        preconditioner ``'auto'``: the metric-lumped Laplacian atom on the
        upper block and the metric-lumped mass atom on the lower one,
        harmonic forms deflated. No Krylov solve nests inside another.
        """
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

        This is the operator of the shift-and-invert inverse iteration that
        :func:`~mrx.nullspace.find_nullspace_vectors` /
        :func:`~mrx.nullspace.compute_nullspaces_iterative` run to find the
        harmonic forms: for ``eps > 0`` the system is nonsingular, so it
        needs no nullspace data, and the same solvers and preconditioners as
        :meth:`apply_inverse_laplacian` apply (the metric-lumped atoms, on
        the shifted operator). ``eps = 0`` is the Laplacian solve with
        deflation. The optional harmonic coarse correction stays off while
        the vectors are still being constructed.

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
        ``'auto'`` resolves to the metric-lumped mass atom, which
        preconditions the dominant (mass) term in the regime this solve is
        used in (``eps * lambda_max(M^-1 L) << 1``).
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

        ``kind``: ``'metric_lumping'`` (the metric-lumped atom, k = 0..3, free
        and Dirichlet -- the production preconditioner, built by
        :meth:`build_preconditioners`), ``'none'`` (identity), or ``'auto'``
        (the default): the atom when the bundle has it for this ``(k, BC)``,
        otherwise a warning and the identity.
        """
        operators = self._require_operators(operators)
        return op.apply_laplacian_preconditioner(
            self, operators, v, k, dirichlet=dirichlet, kind=kind)

    def compute_nullspaces(self, betti_numbers=None, *, direct=True, **kwargs):
        """Compute the harmonic forms and store them on ``self.operators``.

        ``direct=True`` (the default) is the Hodge-decomposition construction
        (:func:`~mrx.nullspace.compute_nullspaces`, ``kwargs`` such as
        ``gap_sweeps`` and ``verbose``): a fixed pair of production solves
        per form, no shift and no outer iteration; it is self-sufficient when
        ``b2 == 0``, i.e. on the solid torus, and raises otherwise. Every
        form is reported with its Rayleigh quotient against ``lambda_1``.
        ``direct=False`` is shift-and-invert inverse iteration
        (:func:`~mrx.nullspace.compute_nullspaces_iterative`, ``kwargs`` such
        as ``eps``, ``abs_tol``, ``inner_tol``, ``maxiter``), which works for
        any Betti numbers and returns its per-vector iteration counts.
        ``betti_numbers`` defaults to ``self.betti_numbers``.
        """
        if direct:
            self.operators = compute_nullspaces(
                self, self._require_operators(), betti_numbers=betti_numbers,
                **kwargs)
            return None
        operators, info = compute_nullspaces_iterative(
            self, self._require_operators(), betti_numbers=betti_numbers, **kwargs)
        self.operators = operators
        return info

    def evaluate_at_quadrature(self, dofs, k, dirichlet=True):
        """Evaluate a k-form at the quadrature points.

        Args:
            dofs: DOF vector of the k-form.
            k: Form degree, 0..3.
            dirichlet: Use the Dirichlet-constrained extraction.

        Returns:
            Array of shape ``(n_q, 3)`` for k = 1, 2 and ``(n_q, 1)`` for
            k = 0, 3: the reference components at every quadrature point,
            in the sequence's flat quadrature order.
        """
        from mrx.quadrature import evaluate_at_xq
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        comp_info, comp_shapes = self._form_comp_info(k)
        ncomp = 3 if k in (1, 2) else 1
        return evaluate_at_xq(self.E(k, dirichlet).T @ dofs, comp_info, comp_shapes,
                              quad_shape, ncomp)

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
        if n not in (1, 2):
            raise ValueError("n must be 1 or 2")
        en = self.E(n, dirichlet_n)
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

    def magnitude_squared_load(self, B, dirichlet=True):
        """The 0-form dual vector of ``|B|^2`` for a 2-form ``B``: ``v_i = ∫ Λ⁰_i |B|² det DF dx``.

        ``|B|^2 = B^T G B / J^2`` for the 2-form proxy, so the integrand is
        ``Λ⁰_i B^T G B / J``. ``M_0^{-1} v`` is the L2 projection of the
        energy density ``|B|^2`` onto the 0-forms (free space).
        """
        from mrx.quadrature import integrate_against
        quad_shape = (self.quad.ny, self.quad.nx, self.quad.nz)
        B_jk = self.evaluate_at_quadrature(B, 2, dirichlet)
        GB_jk = jnp.einsum('jkl,jk->jl', self.metric_jkl, B_jk)
        f_jk = (jnp.sum(B_jk * GB_jk, axis=1) * self.quad.w / self.jacobian_j)[:, None]
        comp_info, comp_shapes = self._form_comp_info(0)
        return self.E(0) @ integrate_against(f_jk, comp_info, comp_shapes, quad_shape)

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
            p_guess = jnp.zeros(self.n(3, True)) if p_guess is None else p_guess
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
            n_p = self.n(0, True) if dirichlet_p else self.n(0)
            p_guess = jnp.zeros(n_p) if p_guess is None else p_guess
            div_v = -self.apply_derivative_matrix(
                v, 0, dirichlet_in=dirichlet_p, dirichlet_out=False, transpose=True)
            q = self.apply_inverse_laplacian(
                div_v, 0, dirichlet=dirichlet_p, guess=-p_guess)
            σ = -self.apply_strong_grad(q, dirichlet_p, False)
            return v - σ, -q
