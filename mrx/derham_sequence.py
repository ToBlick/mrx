"""The :class:`DeRhamSequence`, the discrete de Rham complex that every MRX computation runs on.

A sequence holds the fixed discretisation: the spline spaces of 0-, 1-, 2- and 3-forms on the
logical domain ``(r, theta, zeta)``, their resolution, degree and knots, the number of field
periods and the symmetry of the map, the quadrature rule and the treatment of the polar axis
``r = 0``. None of this depends on the shape of the domain.

Boundary conditions belong to the spaces. ``seq`` itself has the Dirichlet spaces, the working
spaces of ``B``, ``J``, ``E`` and the force: on the wall ``r = 1`` a 0-form vanishes, a 1-form has
no tangential part and a 2-form no normal part. The view ``seq.free`` has the free spaces without a
boundary condition. On a
stellarator-symmetric sequence the views ``seq.odd`` and ``seq.even`` have only the fields of one
parity, and ``seq.odd.free`` combines both. Views share all arrays with the sequence, and taking one
is free. :meth:`~DeRhamSequence.restrict` drops the wall DoFs of a free form.

The matrices of the complex are operator objects indexed by the form degree and applied with
``@``. Each acts on the spaces of the sequence or view it came from::

    B = seq.G[1] @ A                  # curl, a Dirichlet 1-form to a Dirichlet 2-form
    b = seq.D[1].T @ B                # the weak curl of B tested against the 1-forms
    J = seq.M[1].solve(b)             # mass solve
    a = seq.L[1].solve(b)             # Hodge Laplacian solve
    x = seq.shifted(2, eps).solve(r)  # (M_2 + eps L_2)^-1 r
    a = seq.S[1].solve(b)             # S_1 a = b for b = G_1^T w, a up to a closed form
    Bf = seq.free.G[1] @ seq.free.interpolate(A_ref, 1)   # curl of a free 1-form (logical components)
    B = seq.restrict(Bf, 2)                                # dropped to the Dirichlet 2-forms

``M`` is the mass matrix, ``L`` the Hodge Laplacian, ``S`` the stiffness, ``G`` the exterior
derivative (grad, curl, div), ``D = M G`` the weak derivative and ``P[k_in, k_out]`` the metric-free
pairing of two degrees. :mod:`mrx.operators` describes them. Other methods bring a function into a
space (:meth:`~DeRhamSequence.load`, :meth:`~DeRhamSequence.interpolate`), measure a form
(:meth:`~DeRhamSequence.l2_norm`), split a field (:meth:`~DeRhamSequence.leray`) and form products
of fields such as ``J x B`` (:meth:`~DeRhamSequence.evaluate_at_quadrature` with the
``*_load_values`` methods).

The geometry (metric and Jacobian of the map at the quadrature points) and the preconditioners are
data installed on the sequence. :meth:`~DeRhamSequence.set_geometry_from_map` installs a new map and discards the
preconditioners, :meth:`~DeRhamSequence.build_preconditioners` builds them again, and
:func:`mrx.nullspace.compute_nullspaces` adds the harmonic forms. :func:`mrx.geometry.build_sequence`
builds a sequence from a geometry file with its map and preconditioners installed. A free view
copies what is installed when it is taken, so take it anew after any of these calls.

The sequence is a JAX pytree. Under ``eqx.filter_jit`` its arrays, among them the geometry and the
preconditioners, are traced inputs, and everything else (resolution, degree, symmetry, parity, the
boundary condition of a view, tolerances) is compiled in. A new map of the same resolution
therefore reuses the compiled programs, while changing a static attribute during a run recompiles
them. To change the resolution, build a new sequence.
"""
import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mrx
import mrx.operators as op
from mrx.differential_forms import DifferentialForm
from mrx.extraction_operators import build_extraction, conforming_restriction, dirichlet_dofs, get_xi, gram_blocks
from mrx.geometry import SYMMETRIES, SequenceGeometry, map_jacobian_at
from mrx.incidence import build_matrixfree_incidence, build_polar_derivative
from mrx.mass import attach_weights, mass_plan, projection_plan
from mrx.metric_lumping import ReducedAtom
from mrx.precision import SOLVE_TOL
from mrx.projectors import greville_axes, interpolate as _interpolate
from mrx.pytree import register_arrays
from mrx.quadrature import QuadratureRule, evaluate_at_xq, integrate_against
from mrx.spline_bases import basis_derivative_table, basis_table
from mrx.symmetry import (FreeProjector, is_uniform_periodic, parity_extraction, reduce_operator,
                          reduced_dirichlet_dofs, reflection_plan)


#: The boundary type of each logical axis ``(r, theta, zeta)``: clamped in ``r``, periodic in both angles.
AXIS_TYPES = ("clamped", "periodic", "periodic")
#: The Betti numbers ``(b0, b1, b2, b3)`` of the logical domain, a solid torus.
SOLID_TORUS_BETTI = (1, 1, 0, 0)


class DeRhamSequence():
    """Discrete de Rham sequence on a mapped solid torus with a polar axis at ``r = 0``.

    The logical axes ``(r, theta, zeta)`` are clamped in ``r`` and periodic in ``theta`` and ``zeta``.
    All three use splines of degree ``p``. The quadrature takes ``p + 1`` Gauss points per knot span in ``r`` and
    ``p`` in ``theta`` and ``zeta`` (at least 2) unless ``quad_points`` says otherwise.

    Args:
        ns: ``(n_r, n_theta, n_zeta)``, the number of splines per axis.
        p: the spline degree.
        nfp: the number of field periods spanned by ``zeta`` in ``[0, 1]``.
        symmetry: the symmetry of the map, one of :data:`mrx.geometry.SYMMETRIES`. ``"none"`` needs
            ``nfp = 1``. With ``"stellarator"`` and uniform angular knots the sequence integrates over
            half the zeta period only (:attr:`half_period`), which needs an even number of zeta cells.
        knots: full knot vectors ``(T_r, T_theta, T_zeta)``, ``None`` entries uniform.
        theta_counts: the theta count of every primal radial basis function of a graded space
            (:func:`~mrx.extraction_operators.graded_theta_counts`), or ``None`` for ``n_theta`` on
            every ring. Each count is ``n_theta`` divided by a power of two, the counts do not
            decrease outward and the three innermost are equal. Needs uniform theta knots.
        equilibrium: the parsed geometry file (:func:`mrx.equilibria.read_equilibrium`), kept as
            :attr:`equilibrium` for the initial conditions, or ``None``.
        tol: relative residual of every solve (:data:`mrx.precision.SOLVE_TOL` by default).
        maxiter: iteration limit of every solve.
        quad_points: the Gauss points per knot span ``(q_r, q_theta, q_zeta)``, by default
            ``(p + 1, p, p)``. Against ``p + 1`` everywhere this cuts the quadrature grid, and with it the cost
            of every operator apply, by ``(p / (p + 1))^2``. One point per span would make the periodic masses
            singular, so the angles take at least 2. The radial axis keeps ``p + 1`` for the ``r`` and
            ``1 / r`` factors of the polar metric. See ``docs/source/concepts/mass.md``.

    The sequence has the Dirichlet spaces, :attr:`free` the free ones. The constructor also builds
    the parity views :attr:`even` and :attr:`odd` of a half-period sequence.

    Attributes:
        basis_0 .. basis_3: the k-form bases (:class:`~mrx.differential_forms.DifferentialForm`).
        quad: the :class:`~mrx.quadrature.QuadratureRule`.
        half_period: whether the quadrature covers only half the zeta period (:mod:`mrx.symmetry`).
        geometry: the installed geometry (:meth:`set_geometry_from_map`), ``None`` before the first map.
        map_theta: the theta cells of the installed map, the same on every ring (``n_theta`` unless
            :func:`mrx.geometry.build_sequence` was given another).
        operators: the preconditioners and harmonic forms (:meth:`build_preconditioners`),
            ``None`` after every :meth:`set_geometry_from_map`.
    """

    def __init__(self, ns, p, *, nfp, symmetry, knots=None, theta_counts=None, equilibrium=None, tol=None,
                 maxiter=10_000, quad_points=None):
        if symmetry not in SYMMETRIES:
            raise ValueError(f"symmetry must be one of {SYMMETRIES}, got {symmetry!r}")
        if symmetry == "none" and int(nfp) != 1:
            raise ValueError(f"symmetry 'none' means zeta in [0, 1] is the whole torus, but nfp = {nfp}")
        self.ns = tuple(int(n) for n in ns)
        self.p = int(p)
        self.nfp = int(nfp)
        self.symmetry = symmetry
        self.tol = SOLVE_TOL if tol is None else tol
        self.maxiter = maxiter
        self.geometry = None
        #: the theta cells of the installed map on every ring, set by :func:`mrx.geometry.build_sequence`
        self.map_theta = self.ns[1]
        self.operators = None
        self.betti_numbers = SOLID_TORUS_BETTI
        Ts = [None if T is None else jnp.asarray(T, dtype=jnp.float64)
              for T in (knots if knots is not None else (None,) * 3)]
        #: the theta count of every primal radial basis function (graded space), or ``None`` (uniform)
        self.theta_counts = None if theta_counts is None else tuple(int(m) for m in theta_counts)
        if self.theta_counts is not None:
            m, nt = np.asarray(self.theta_counts), self.ns[1]
            ratio = nt // m
            if (len(m) != self.ns[0] or Ts[1] is not None or np.any(nt % m) or np.any(ratio & (ratio - 1))
                    or np.any(np.diff(m) < 0) or len(set(m[:3])) != 1 or m[0] <= self.p):
                raise ValueError(f"theta_counts {self.theta_counts} do not grade n_theta = {nt} over "
                                 f"{self.ns[0]} radial functions (see DeRhamSequence)")
        self.basis_0, self.basis_1, self.basis_2, self.basis_3 = [
            DifferentialForm(i, self.ns, (self.p,) * 3, AXIS_TYPES, Ts) for i in range(0, 4)
        ]
        # the reflection is an index permutation only on uniform periodic angular bases
        self.half_period = symmetry == "stellarator" and all(
            is_uniform_periodic(self.basis_0.Lambda[a]) for a in (1, 2))
        angles = max(self.p, 2)
        self.quad_points = ((self.p + 1, angles, angles) if quad_points is None
                            else tuple(int(q) for q in quad_points))
        self.quad = QuadratureRule(self.basis_0, self.quad_points, half_zeta=self.half_period)
        #: the reflection of every raw DoF grid on a half-period sequence
        self.reflection_plan = ({k: reflection_plan(self, k) for k in range(4)}
                                if self.half_period else None)
        #: ``None``, or ``+1`` / ``-1`` on the reduced views :attr:`even` / :attr:`odd`
        self.parity = None
        #: ``True`` on the Dirichlet spaces, ``False`` on the view :attr:`free`
        self.dirichlet = True
        self._base = None
        #: on a reduced view, the expansion ``X`` (free <- reduced) per ``(k, dirichlet)``
        self.reduction = None
        self.extraction, self.core = self._extractions()
        # the position of each Dirichlet DoF among the free DoFs, per degree (restrict and extend)
        forms = (self.basis_0, self.basis_1, self.basis_2, self.basis_3)
        self._dirichlet_dofs = {k: jnp.asarray(dirichlet_dofs(forms[k], self.theta_counts, self.extraction[(k, False)],
                                                              self.extraction[(k, True)]))
                                for k in range(4)}
        self._gram = {(k, d): gram_blocks(forms[k], d, self.theta_counts) for k in range(4) for d in (False, True)}

        lam, dlam = self.basis_0.Lambda, self.basis_0.dLambda
        x = (self.quad.x_x, self.quad.x_y, self.quad.x_z)
        #: per axis (r, theta, zeta) at its quadrature points, each ``(n, n_q)``: the splines, the derivative
        #: splines and the derivatives of the derivative splines
        self.basis_jk = tuple(basis_table(b, xa) for b, xa in zip(lam, x))
        self.d_basis_jk = tuple(basis_table(b, xa) for b, xa in zip(dlam, x))
        self.dd_basis_jk = tuple(basis_derivative_table(b, xa) for b, xa in zip(dlam, x))
        self.greville = greville_axes(self)

        #: grad, curl, div on the raw tensor-product coefficients (:mod:`mrx.incidence`)
        self.incidence = tuple(build_matrixfree_incidence(self, k) for k in range(3))
        # grad, curl and div on the extracted DoFs of the free (False) and Dirichlet (True) spaces
        self.g0_grad, self.g1_curl, self.g2_div = ({d: build_polar_derivative(self, k, d) for d in (False, True)}
                                                   for k in range(3))
        self.mass_plan = {k: mass_plan(self, k) for k in range(4)}
        self.projection_plan = {pair: projection_plan(self, *pair)
                                for pair in ((1, 2), (2, 1), (0, 3), (3, 0))}
        self.equilibrium = equilibrium
        self._free_projectors = ({(k, d): FreeProjector(self, k, d) for k in range(4) for d in (False, True)}
                                 if self.half_period else {})
        # The parity views get each new geometry from set_geometry, so the programs compiled against
        # them survive a new map.
        self._parity_views = {}
        if self.half_period:
            self._parity_views = {s: self._parity_view(s) for s in (1, -1)}

    def _xi(self):
        """The polar weights on the coarsest theta count."""
        return get_xi(self.ns[1] if self.theta_counts is None else self.theta_counts[0], self.p)

    def _extractions(self):
        """The extraction operators and their axis rows for every ``(k, dirichlet)``."""
        xi = self._xi()
        built = {(k, d): build_extraction(form, xi, d, self.theta_counts)
                 for k, form in enumerate((self.basis_0, self.basis_1, self.basis_2, self.basis_3))
                 for d in (False, True)}
        return {key: e for key, (e, _) in built.items()}, {key: c for key, (_, c) in built.items()}

    def load(self, f, k: int):
        """The vector of integrals of ``f`` against the k-form basis functions, the right-hand side of an L2
        projection. ``f`` takes a logical point and returns a scalar (k = 0, 3) or three physical components
        (k = 1, 2), which are pulled back here."""
        f_jk = jax.lax.map(f, self.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        if k in (0, 3):
            return self._scalar_load_values(jnp.reshape(f_jk, (-1,)), k)
        # covariant components DPhi^T f. DPhi is not stored on the geometry, so it is recomputed here
        return self.vector_load_values(jnp.einsum('qji,qj->qi', map_jacobian_at(self.map, self.quad.x), f_jk), 1, k)

    def interpolate(self, f, k: int):
        """The k-form DoFs of ``f``, which returns the logical components of the form at a logical point, by
        interpolation at the Greville points (k = 0) or by integrals over edges, faces and cells (k = 1, 2, 3),
        see :func:`mrx.projectors.interpolate`."""
        return _interpolate(self, f, k)

    @property
    def map(self):
        """The logical-to-physical map ``Phi`` of the installed geometry."""
        return self._require_geometry().map

    @property
    def metric_jkl(self):
        """Metric ``G = DPhi^T DPhi`` at the quadrature points, ``(n_q, 3, 3)``."""
        return self._require_geometry().metric_jkl

    @property
    def metric_inv_jkl(self):
        """Inverse metric at the quadrature points, ``(n_q, 3, 3)``."""
        return self._require_geometry().metric_inv_jkl

    @property
    def jacobian_j(self):
        """``det DPhi`` at the quadrature points, ``(n_q,)``."""
        return self._require_geometry().jacobian_j

    def E(self, k):
        """The extraction operator of the ``k``-form space. It maps the tensor-product spline
        coefficients to the DoFs of the space, and ``.T`` maps DoFs back to tensor-product
        coefficients."""
        return self.extraction[(int(k), self.dirichlet)]

    def n(self, k):
        """The number of DoFs of the ``k``-form space."""
        return int(self.E(k).forward_shape[0])

    def conforming(self, c, k):
        """The ``k``-form DoFs whose tensor-product coefficients are closest (least squares) to ``c``,
        ``(E E^T)^-1 E c`` (:func:`~mrx.extraction_operators.conforming_restriction`). On a parity view
        this is ``X^T`` of the full space's, since ``E E^T`` commutes with the reflection."""
        key = (int(k), self.dirichlet)
        base = self if self.parity is None else self._base
        v = conforming_restriction(base.extraction[key], c, base.core[key], base._gram[key])
        return v if self.parity is None else self.reduction[key].T @ v

    def core_rows(self, k):
        """The rows of ``E(k)`` that combine several tensor-product coefficients near the axis. On
        all other rows ``E E^T`` is the identity."""
        return self.core[(int(k), self.dirichlet)]

    def nullspace(self, k):
        """The harmonic ``k``-forms of the space, shape ``(n_vectors, n_k)``. They are zero until
        :func:`mrx.nullspace.compute_nullspaces` has run."""
        return self._require_operators().nullspaces[(int(k), self.dirichlet)]

    @property
    def free(self):
        """The view of this sequence with the free spaces, which have no boundary condition at the
        wall ``r = 1``. It shares all arrays, the geometry and the preconditioners with this
        sequence. The sequence itself has the Dirichlet spaces (``B . n = 0``, ``E x n = 0``,
        ``p = 0`` on the wall), the working spaces of ``B``, ``J``, ``E`` and the force. Taking the
        view is free, also inside a jitted function. ``seq.free.odd`` and ``seq.odd.free`` are the
        same space. The view copies the geometry and preconditioners installed when it is taken, so
        take it anew after :meth:`set_geometry_from_map`, :meth:`build_preconditioners` or
        :func:`mrx.nullspace.compute_nullspaces`."""
        if not self.dirichlet:
            return self
        view = copy.copy(self)
        view.dirichlet = False
        return view

    def restrict(self, v, k):
        """The Dirichlet ``k``-form with the DoFs of the free ``k``-form ``v`` away from the wall. The
        DoFs on the wall are dropped."""
        return v[self._dirichlet_dofs[int(k)]]

    # --- the matrices of the complex (mrx.operators) ------------------------------------
    #
    # Each acts on the spaces of this sequence, the Dirichlet ones on seq and the free ones on
    # seq.free. The solves and preconditioners need build_preconditioners.

    @property
    def M(self):
        """The mass matrices: ``seq.M[k] @ v``, ``seq.M[k].solve(b)``, ``seq.M[k].precondition(r)``."""
        return op.OperatorFamily(self, op.MassMatrix, (0, 1, 2, 3))

    @property
    def L(self):
        """The Hodge Laplacians: ``seq.L[k] @ v``, ``seq.L[k].solve(b)``, ``seq.L[k].precondition(r)``."""
        return op.OperatorFamily(self, op.Laplacian, (0, 1, 2, 3))

    @property
    def S(self):
        """The stiffness matrices ``S_k = G_k^T M_{k+1} G_k``: ``seq.S[k] @ v``, ``seq.S[k].solve(b)``."""
        return op.OperatorFamily(self, op.Stiffness, (0, 1, 2, 3))

    @property
    def G(self):
        """The exterior derivatives grad, curl, div (k = 0, 1, 2): ``seq.G[k] @ v``, ``seq.G[k].T @ w``."""
        return op.OperatorFamily(self, op.ExteriorDerivative, (0, 1, 2))

    @property
    def D(self):
        """The weak derivatives ``D_k = M_{k+1} G_k``: ``seq.D[k] @ v``, ``seq.D[k].T @ w``."""
        return op.OperatorFamily(self, op.WeakDerivative, (0, 1, 2))

    @property
    def P(self):
        """The metric-free pairings ``seq.P[k_in, k_out] @ v`` of a ``k_in``-form with the
        ``k_out``-forms, for (1, 2), (2, 1), (0, 3) and (3, 0)."""
        return op.OperatorFamily(self, op.Projection, ((1, 2), (2, 1), (0, 3), (3, 0)))

    def shifted(self, k, eps):
        """The shifted Laplacian ``M_k + eps L_k``, solved by ``seq.shifted(k, eps).solve(b)``. Pass
        ``eps`` as a JAX array, otherwise every new value recompiles."""
        return op.ShiftedLaplacian(self, int(k), eps)

    def shifted_stiffness(self, k, eps):
        """The shifted stiffness ``M_k + eps S_k``, solved by ``seq.shifted_stiffness(k, eps).solve(b)``. Pass
        ``eps`` as a JAX array, otherwise every new value recompiles."""
        return op.ShiftedStiffness(self, int(k), eps)

    def with_geometry(self, geometry: SequenceGeometry):
        """A copy of this sequence and of its parity views with ``geometry`` installed. Unlike
        :meth:`set_geometry` it changes nothing in place and can be used inside ``jax.grad`` and ``jit``, so the
        copy is differentiable in ``geometry``. It keeps the preconditioners and harmonic forms of this
        sequence, which belong to the old geometry."""
        out = copy.copy(self)
        out.geometry = attach_weights(self, geometry)
        out._parity_views = {}
        for s, view in self._parity_views.items():
            v = copy.copy(view)
            v.geometry, v._base = out.geometry, out
            out._parity_views[s] = v
        return out

    def set_geometry_from_map(self, Phi, keep_preconditioners=False):
        """Evaluate the geometry of the logical-to-physical map ``Phi`` at the quadrature points and
        install it (:meth:`set_geometry`)."""
        self.set_geometry(SequenceGeometry.from_map(Phi, self.quad.x), keep_preconditioners)

    def set_geometry(self, geometry: SequenceGeometry, keep_preconditioners=False):
        """Install ``geometry`` on this sequence and its parity views. Raises
        ``ValueError`` if ``det DPhi`` is not positive at every quadrature point.

        The preconditioners and the harmonic forms belong to the old geometry and are discarded: call
        :meth:`build_preconditioners` and :func:`mrx.nullspace.compute_nullspaces` again. With
        ``keep_preconditioners`` the preconditioners stay. They are those of the old geometry, which
        is still a valid preconditioner of the new operators and a good one after a small change of
        the map. The harmonic forms are set to zero either way, because a form of the old geometry
        is not in the kernel of the new Laplacian: compute them again before a solve that removes
        them."""
        jac = np.asarray(geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0.0:
            raise ValueError(
                f"the map folds: det DPhi at the quadrature points spans "
                f"[{jac.min():.3e}, {jac.max():.3e}] and must be positive")
        def stale(seq):
            # the bundle of the old geometry with its harmonic forms zeroed, or nothing
            if not keep_preconditioners or seq.operators is None:
                return None
            return op.init_nullspaces(seq, seq.operators)

        self.geometry = attach_weights(self, geometry)
        self.operators = stale(self)
        for view in self._parity_views.values():
            view.geometry, view.operators = view._base.geometry, stale(view)

    @property
    def odd(self):
        """The sequence reduced to the fields odd under the stellarator rotation (``B``, ``A``, ``J``, ``E``, ``H``)."""
        return self.parity_view(-1)

    @property
    def even(self):
        """The sequence reduced to the fields even under the stellarator rotation (``u``, ``F``, ``p``)."""
        return self.parity_view(1)

    def parity_view(self, parity):
        """The view of a half-period sequence whose spaces contain only the fields of the given
        parity (``+1`` even, ``-1`` odd). It shares the geometry and quadrature with this
        sequence, has about half the DoFs and its own preconditioners. A full-period sequence is
        its own view. In compiled code take both views from the full sequence. A view reaches the
        other one through its static base, so that view's arrays would be compiled in as
        constants, which a later :meth:`set_geometry_from_map` does not reach."""
        s = int(parity)
        if not self.half_period or s == self.parity:
            return self
        view = (self._parity_views if self.parity is None else self._base._parity_views)[s]
        return view if self.dirichlet else view.free

    def _parity_view(self, s):
        """Build the view of parity ``s`` behind :meth:`parity_view`."""
        view = copy.copy(self)
        view.parity, view._base, view._parity_views, view.operators = s, self, {}, None
        view._free_projectors = {}
        # the harmonic forms are odd, the constants (k=0 free, k=3 Dirichlet) even
        b0, b1, b2, b3 = self.betti_numbers
        view.betti_numbers = (0, b1, b2, b3) if s == -1 else (b0, 0, 0, 0)
        view.extraction, view.reduction, view.core = {}, {}, {}
        for key in self.extraction:
            view.extraction[key], view.reduction[key], view.core[key] = parity_extraction(self, *key, s)
        X = view.reduction
        view.g0_grad = {d: reduce_operator(X[(1, d)], g, X[(0, d)]) for d, g in self.g0_grad.items()}
        view.g1_curl = {d: reduce_operator(X[(2, d)], g, X[(1, d)]) for d, g in self.g1_curl.items()}
        view.g2_div = {d: reduce_operator(X[(3, d)], g, X[(2, d)]) for d, g in self.g2_div.items()}
        view._dirichlet_dofs = {
            k: jnp.asarray(reduced_dirichlet_dofs(X[(k, False)], X[(k, True)], np.asarray(index)))
            for k, index in self._dirichlet_dofs.items()}
        return view

    def build_preconditioners(self, mass_atom="slab", laplacian_atom="lumping"):
        """Build the mass and Laplacian preconditioners of every space for the installed geometry,
        install them as :attr:`operators` and return them. ``mass_atom`` selects the mass
        preconditioner of ``k = 1, 2``, one of :data:`mrx.operators.MASS_ATOMS`, and ``laplacian_atom``
        whether the co-closed 2-form solves use the multigrid, one of :data:`mrx.operators.LAPLACIAN_ATOMS`. The harmonic forms stay zero until
        :func:`mrx.nullspace.compute_nullspaces` runs. Needs ``n_r >= p + 2``."""
        self._require_geometry()
        if self.parity is not None:
            # a parity view reuses the preconditioners of the full sequence, restricted to its space
            base = self._base.operators
            if base is None:
                base = self._base.build_preconditioners(mass_atom, laplacian_atom)
            ops = op.new_operators(self)
            ops = eqx.tree_at(lambda o: (o.mass_lumping, o.laplacian_lumping, o.mass_model, o.stiffness), ops,
                              tuple({key: ReducedAtom(atom, self.reduction[key]) for key, atom in atoms.items()}
                                    for atoms in (base.mass_lumping, base.laplacian_lumping, base.mass_model,
                                                  base.stiffness)),
                              is_leaf=lambda x: x is None or isinstance(x, dict))
            self.operators = ops
            return ops
        ops = op.assemble_preconditioners(self, op.new_operators(self), mass_atom, laplacian_atom)
        self.operators = ops
        if self.half_period:
            for view in (self.odd, self.even):
                view.build_preconditioners()
        return ops

    def _require_geometry(self):
        if self.geometry is None:
            raise ValueError('Set the geometry first, for example with seq.set_geometry_from_map(...).')
        return self.geometry

    def _require_operators(self):
        if self.operators is None:
            raise ValueError('no operator bundle: call seq.build_preconditioners() after set_geometry_from_map')
        return self.operators

    def _form_comp_info(self, k):
        """The 1-D basis tables at the quadrature points for each component of the k-forms."""
        form = (self.basis_0, self.basis_1, self.basis_2, self.basis_3)[k]
        return [(c, *((self.d_basis_jk if a in form.derivative_axes(c) else self.basis_jk)[a] for a in range(3)))
                for c in range(len(form.shape))]

    def _integrate(self, f_jk, n):
        """The integrals of the quadrature values ``f_jk`` (weights included) against the n-form basis. A
        half-period sequence integrates only over half the period, so a field must be on its parity view."""
        if self.half_period and self.parity is None:
            raise ValueError("on a half-period sequence, loads and products are formed on seq.odd or seq.even")
        return self.E(n) @ integrate_against(f_jk, self._form_comp_info(n))

    def free_projector(self, k):
        """The :class:`mrx.symmetry.FreeProjector` of the ``k``-form space. ``None`` on a
        full-period sequence and on a parity view."""
        return self._free_projectors.get((int(k), self.dirichlet))

    def l2_norm_sq(self, v, k):
        """The squared L2 norm ``v^T M_k v`` of the k-form ``v``."""
        return v @ (self.M[k] @ v)

    def l2_norm(self, v, k):
        """The L2 norm ``sqrt(v^T M_k v)`` of the k-form ``v``."""
        return jnp.sqrt(self.l2_norm_sq(v, k))

    def leray(self, v, k=2, p_guess=None, sigma_guess=None):
        """``(v_out, p)``: the weakly divergence-free part ``v_out = v - grad p`` of a 1- or 2-form.

        For ``k = 2``, ``p`` is a 3-form from the k = 3 Hodge split, whose second unknown is
        ``sigma = v - v_out``. On the Dirichlet sequence ``v . n = 0`` and ``p`` is the pressure of the
        force. ``p_guess`` and ``sigma_guess`` (the previous ``v - v_out``) warm-start the solve. For ``k = 1``, ``p`` is a 0-form from the k = 0
        Laplacian. On ``seq.free`` it satisfies ``dp/dn = v . n`` up to a constant, on the Dirichlet
        sequence ``p = 0`` on the wall (the weak pressure).
        """
        # The solves return q = -p (sigma = -grad q). p and p_guess carry the physical sign.
        if k == 2:
            p_guess = jnp.zeros(self.n(3)) if p_guess is None else p_guess
            q, sigma, _ = op._hodge_split(self, self.D[2] @ v, 3, -p_guess, guess_lower=sigma_guess)
            return v - sigma, -q
        p_guess = jnp.zeros(self.n(0)) if p_guess is None else p_guess
        q = self.L[0].solve(-(self.D[0].T @ v), guess=-p_guess)
        return v + self.G[0] @ q, -q

    # --- evaluation and products of fields ------------------------------------------
    #
    # Reference components: a 1-form is covariant (DPhi^T a), a 2-form a contravariant
    # density (J DPhi^-1 b), a 3-form J times its value. Testing a 1-form against the 1-forms
    # carries the weight G^-1 J, a 2-form against the 2-forms G / J, and a value against the
    # 0-forms J. A 1-form against the 2-forms, a 2-form against the 1-forms and a value
    # against the 3-forms carry no metric.

    def evaluate_at_quadrature(self, dofs, k):
        """The reference components of the k-form ``dofs`` at the quadrature points, shape
        ``(n_q, 3)`` for ``k = 1, 2`` and ``(n_q, 1)`` for ``k = 0, 3``. These are the inputs of
        the ``*_load_values`` methods."""
        form = (self.basis_0, self.basis_1, self.basis_2, self.basis_3)[k]
        return evaluate_at_xq(self.E(k).T @ dofs, self._form_comp_info(k), list(form.shape),
                              self.quad.shape, 3 if k in (1, 2) else 1)

    def cross_product_load_values(self, w_jk, u_jk, n, m, k):
        """The integrals of ``w x u`` against the n-form basis, from the quadrature values
        (:meth:`evaluate_at_quadrature`) of the m-form ``w`` and the k-form ``u``, with n, m, k in
        {1, 2}. On a half-period sequence it is formed on the view of the product's parity.
        """
        # (A a) x (A b) = det(A) A^-T (a x b): the product is formed in the components that need
        # the least metric when tested against the n-forms
        def contract(A, x):
            return jnp.einsum('jkl,jk->jl', A, x)

        G, G_inv, J = self.metric_jkl, self.metric_inv_jkl, self.jacobian_j[:, None]
        if m == 1 and k == 1:
            c, rep = jnp.cross(w_jk, u_jk, axis=1), 2
        elif m == 2 and k == 2:
            wxu = jnp.cross(w_jk, u_jk, axis=1)
            c, rep = (wxu / J, 1) if n == 2 else (contract(G_inv, wxu), 2)
        elif n == 2:      # covariant: G^-1 on the 1-form factor
            c = (jnp.cross(contract(G_inv, w_jk), u_jk, axis=1) if m == 1
                 else jnp.cross(w_jk, contract(G_inv, u_jk), axis=1))
            rep = 1
        else:             # density: G on the 2-form factor, over J
            c = (jnp.cross(contract(G, w_jk), u_jk, axis=1) if m == 2
                 else jnp.cross(w_jk, contract(G, u_jk), axis=1)) / J
            rep = 2
        return self.vector_load_values(c, rep, n)

    def _scalar_load_values(self, s_jk, n):
        """The integrals of the scalar ``s_jk`` (values at the quadrature points) against the n-form basis, n = 0, 3."""
        weight = self.quad.w * self.jacobian_j if n == 0 else self.quad.w
        return self._integrate((s_jk * weight)[:, None], n)

    def vector_load_values(self, c_jk, rep, n):
        """The integrals of a vector field against the n-form basis (n = 1, 2), from its
        covariant (``rep = 1``) or contravariant-density (``rep = 2``) reference components
        ``c_jk`` at the quadrature points."""
        if n == 1 and rep == 1:
            f_jk = (jnp.einsum('jkl,jk->jl', self.metric_inv_jkl, c_jk)
                    * (self.quad.w * self.jacobian_j)[:, None])
        elif n == 2 and rep == 2:
            f_jk = (jnp.einsum('jkl,jk->jl', self.metric_jkl, c_jk)
                    * (self.quad.w / self.jacobian_j)[:, None])
        else:
            f_jk = c_jk * self.quad.w[:, None]
        return self._integrate(f_jk, n)

    def dot_product_load_values(self, w_jk, u_jk, n, m, k):
        """The integrals of ``w . u`` against the n-form basis (n = 0, 3), from the quadrature
        values of the m-form ``w`` and the k-form ``u`` (m, k in {1, 2})."""
        if m == 1 and k == 1:
            s_jk = jnp.einsum('jk,jkl,jl->j', w_jk, self.metric_inv_jkl, u_jk)
        elif m == 2 and k == 2:
            s_jk = jnp.einsum('jk,jkl,jl->j', w_jk, self.metric_jkl, u_jk) / self.jacobian_j ** 2
        else:
            s_jk = jnp.sum(w_jk * u_jk, axis=1) / self.jacobian_j
        return self._scalar_load_values(s_jk, n)


# ``_base`` (the full sequence behind a parity view) is static. As a pytree child it would
# form a cycle with the full sequence's ``_parity_views``.
register_arrays(DeRhamSequence, static=("equilibrium", "_base"))
