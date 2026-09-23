"""Shape derivatives of the vacuum field by reverse-mode automatic differentiation.

The variable is the boundary of a spline map, the state the harmonic 2-form
``h`` of the Dirichlet complex (the vacuum field with one unit of toroidal
flux, :func:`mrx.nullspace.compute_nullspaces`), the objective any function
of ``h`` -- here the rotational transform. ``jax.grad`` of such an objective
costs one forward and one adjoint solve.

**The state is one linear solve.** ``h = s - G_1 a`` with ``S_1 a = G_1^T M_2 s``,
``s`` the histopolated flux seed (:func:`flux_seed`, geometry-free) and ``S_1
= G_1^T M_2 G_1`` the curl-curl stiffness of the Dirichlet 1-forms. The
geometry enters only through the element weights ``G / J`` of ``M_2``,
elementwise functions of the map coefficients. :func:`vacuum_two_form` wraps
the solve in :func:`jax.lax.custom_linear_solve`, so its derivative is the
implicit one: ``S_1 da = G_1^T dM_2 h``, and in reverse mode

    dJ/dalpha = (G_1 lambda)^T (dM_2/dalpha) h,   S_1 lambda = -G_1^T dJ/dh.

Every right-hand side is annihilated by ``G_0^T`` (``G_1 G_0 = 0``), so the
singular ``S_1`` is safe; the production solve (the Hodge split of ``L_1``)
returns the weakly divergence-free solution, and ``G_1`` removes the
exact part a gauge would leave. ``custom_linear_solve`` differentiates only
what the ``matvec`` closes over and drops the tangents of what the ``solve``
closes over: the preconditioners -- host-built, not differentiable -- stay
FROZEN at the geometry they were built for (:func:`with_geometry`), which
changes the iteration count of the solves and nothing else.

**Parameters** (:class:`BoundaryShape`): the outermost ring of the raw
``R`` and ``Z`` coefficients of the map, ``(n_theta, n_zeta)`` each,
extended inwards by a radial ramp that vanishes on the two innermost rings,
so the C1 polar structure and the coordinate axis of the start map are kept
(the magnetic axis is free to move). The perturbation is projected onto
stellarator symmetry (``R`` even, ``Z`` odd), and the coefficients are
rescaled so that the volume stays that of the start map exactly (a uniform
scaling leaves the vacuum rotational transform unchanged). Interior
coefficients are not free: they would let the logical surfaces bend and
fool :func:`flux_ratio_iota`.

**The objective** (:func:`flux_ratio_iota`): ``iota = nfp <B^theta> /
<B^zeta>`` on the logical surfaces, the averages over ``(theta, zeta)``
taken exactly from the DoFs. It is the rotational transform wherever the
logical surfaces are flux surfaces; :func:`normal_field_fraction` measures
how far they are from that.

On a half-period sequence (:mod:`mrx.symmetry`) every field here is odd, the
geometry even, and the parametrisation stellarator-symmetric by
construction, so the half-period quadrature is exact for the objective and
for its gradient with respect to the parameters.
"""
import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mrx.operators as op
from mrx.differential_forms import DifferentialForm, inv33
from mrx.geometry import SequenceGeometry, _tp_evaluate, grad_1d
from mrx.mappings import stellarator_symmetric_scalar
from mrx.mass import attach_weights
from mrx.quadrature import composite_quad
from mrx.spline_bases import DerivativeSpline, basis_derivative_table, basis_table


def with_geometry(seq, geometry):
    """``seq`` with ``geometry`` installed, as a shallow copy that keeps the
    operator bundle of ``seq``: the traceable counterpart of
    :meth:`~mrx.derham_sequence.DeRhamSequence.set_geometry`, which reads the
    Jacobian on the host and drops the bundle.

    The mass and projection weights are attached from ``geometry`` (they are
    differentiable in it); the preconditioners on the bundle stay those of
    the geometry they were built for. The float64 view of a refined
    sequence is dropped, so the residual of a solve is measured on the new
    geometry.
    """
    out = copy.copy(seq)
    out.geometry = attach_weights(seq, geometry)
    out._residual = None
    return out


def cylindrical_geometry(seq, raw_R, raw_Z, nfp, sign):
    """The :class:`~mrx.geometry.SequenceGeometry` of the map ``F = (R cos
    phi, sign R sin phi, Z)``, ``phi = 2 pi zeta / nfp``, whose ``R`` and ``Z``
    are the scalar splines of ``seq.basis_0`` with the raw ``(n_r, n_t,
    n_z)`` coefficients ``raw_R`` and ``raw_Z`` (the map of
    :func:`mrx.gvec.build_gvec_map`). Sum-factorised on the quadrature grid
    like :meth:`~mrx.geometry.SequenceGeometry.from_spline_map`, from the
    first derivatives of ``R`` and ``Z`` (no ``jacfwd``):

        G_ij = d_i R d_j R + d_i Z d_j Z + delta_{i zeta} delta_{j zeta} (2 pi R / nfp)^2,
        J = sign (2 pi / nfp) R (R_theta Z_r - R_r Z_theta).

    Differentiable in the coefficients. The geometry carries no map (the
    solves never evaluate it).
    """
    Br, Bt, Bz = seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk
    Dr, Dt, Dz = (grad_1d(d, t) for d, t in zip(
        (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk), seq.basis_0.types))
    C = jnp.stack([raw_R, raw_Z])
    R = _tp_evaluate(C[:1], Br, Bt, Bz)[0].reshape(-1)
    d = jnp.stack([_tp_evaluate(C, Dr, Bt, Bz), _tp_evaluate(C, Br, Dt, Bz),
                   _tp_evaluate(C, Br, Bt, Dz)], axis=-1)            # (2, nqr, nqt, nqz, 3)
    dR, dZ = d[0].reshape(-1, 3), d[1].reshape(-1, 3)
    a = 2.0 * np.pi / nfp
    G = dR[:, :, None] * dR[:, None, :] + dZ[:, :, None] * dZ[:, None, :]
    G = G.at[:, 2, 2].add((a * R) ** 2)
    J = sign * a * R * (dR[:, 1] * dZ[:, 0] - dR[:, 0] * dZ[:, 1])
    return SequenceGeometry(None, G, jax.vmap(inv33)(G), J)


class BoundaryShape(eqx.Module):
    """The map's ``R`` and ``Z`` coefficients as a function of the boundary
    variables ``beta``, an array ``(2, n_theta, n_zeta)``: ``beta[0]`` the
    change of the outermost ring of the ``R`` coefficients, ``beta[1]`` of
    the ``Z`` ones.

    The change is projected onto stellarator symmetry (``R`` even, ``Z``
    odd, :meth:`perturbation`) and extended inwards by ``ramp``, ``w_i =
    ((i - 1) / (n_r - 2))^2`` on ring ``i >= 1``: zero on the two innermost
    rings, which carry the C1 polar structure and the coordinate axis, one
    on the boundary. :meth:`geometry` rescales the map so that its volume
    is ``volume``, the start map's, exactly.
    """

    raw_R: jnp.ndarray
    raw_Z: jnp.ndarray
    ramp: jnp.ndarray
    volume: jnp.ndarray
    nfp: int = eqx.field(static=True)
    sign: float = eqx.field(static=True)
    basis_0: DifferentialForm = eqx.field(static=True)

    @classmethod
    def from_coefficients(cls, seq, raw_R, raw_Z, nfp, sign):
        """The parametrisation about the map with raw coefficients ``raw_R``,
        ``raw_Z`` (``nfp`` and handedness ``sign`` as in
        :func:`cylindrical_geometry`); its volume is the one kept."""
        n_r = raw_R.shape[0]
        ramp = jnp.asarray((np.maximum(np.arange(n_r) - 1, 0) / (n_r - 2)) ** 2, dtype=raw_R.dtype)
        volume = jnp.sum(seq.quad.w * cylindrical_geometry(seq, raw_R, raw_Z, nfp, sign).jacobian_j)
        return cls(raw_R, raw_Z, ramp, volume, int(nfp), float(sign), seq.basis_0)

    def perturbation(self, beta):
        """``(b_R, b_Z)``, the boundary change ``beta`` projected onto ``R``
        even and ``Z`` odd under ``(theta, zeta) -> (-theta, -zeta)``."""
        return (stellarator_symmetric_scalar(beta[0][None], self.basis_0, even=True)[0],
                stellarator_symmetric_scalar(beta[1][None], self.basis_0, even=False)[0])

    def coefficients(self, beta, scale=1.0):
        """The raw ``(R, Z)`` coefficients at ``beta``, times ``scale``."""
        b_R, b_Z = self.perturbation(beta)
        w = self.ramp[:, None, None]
        return scale * (self.raw_R + w * b_R), scale * (self.raw_Z + w * b_Z)

    def geometry(self, seq, beta):
        """``(geometry, scale)``: the geometry of the map at ``beta`` scaled
        by ``scale = (volume / V)^(1/3)``, ``V`` the quadrature volume of the
        unscaled map, so that the volume is ``volume`` at every ``beta``
        (``G`` scales with ``scale^2``, ``J`` with ``scale^3``)."""
        g = cylindrical_geometry(seq, *self.coefficients(beta), self.nfp, self.sign)
        scale = (self.volume / jnp.sum(seq.quad.w * g.jacobian_j)) ** (1.0 / 3.0)
        return SequenceGeometry(None, scale ** 2 * g.metric_jkl, g.metric_inv_jkl / scale ** 2,
                                scale ** 3 * g.jacobian_j), scale


def _periodic_grams(basis):
    """``int B_i^(d) B_j^(d)`` over one period of the periodic spline
    ``basis`` for ``d = 0, 1, 2``, ``(3, n, n)``, by Gauss quadrature on its
    knot spans (exact). ``B'`` comes from the derivative basis
    (``B'_l = D_{l-1} - D_l``, :func:`mrx.geometry.grad_1d`), ``B''`` from
    its derivative."""
    T = np.asarray(basis.T)
    x, w = composite_quad(jnp.asarray(np.unique(T[(T >= 0.0) & (T <= 1.0)])), basis.p + 1)
    d = DerivativeSpline(basis)
    tables = (basis_table(basis, x), grad_1d(basis_table(d, x), basis.type),
              grad_1d(basis_derivative_table(d, x), basis.type))
    return jnp.stack([(B * w) @ B.T for B in tables])


class BoundaryH2(eqx.Module):
    """The squared H^2 seminorm of a boundary change ``b`` (raw coefficients
    ``(n_theta, n_zeta)`` on the angular splines of the map) on the
    ``(theta, zeta)`` torus in arc length, ``x = length_t theta`` and ``y =
    length_z zeta``, plus an L2 term at the scale ``length_l2``:

        int b_xx^2 + 2 b_xy^2 + b_yy^2 + b^2 / length_l2^4  dx dy,

    the exact quadratic form ``b^T (K2 x M + 2 K1 x K1 + M x K2 + M x M /
    length_l2^4) b`` of the 1-D periodic masses ``M`` and stiffnesses ``K1 =
    int B' B'``, ``K2 = int B'' B''`` in those lengths.
    """

    gram_t: jnp.ndarray
    gram_z: jnp.ndarray
    length_t: float = eqx.field(static=True)
    length_z: float = eqx.field(static=True)
    length_l2: float = eqx.field(static=True)

    @classmethod
    def build(cls, seq, length_t, length_z, length_l2):
        """The form on the angular bases of ``seq.basis_0``."""
        return cls(_periodic_grams(seq.basis_0.Λ[1]), _periodic_grams(seq.basis_0.Λ[2]),
                   float(length_t), float(length_z), float(length_l2))

    def __call__(self, b):
        (m_t, k1_t, k2_t), (m_z, k1_z, k2_z) = self.gram_t, self.gram_z
        lt, lz = self.length_t, self.length_z

        def pair(A, B):
            return jnp.sum((A @ b @ B) * b)
        return lt * lz * (pair(k2_t, m_z) / lt ** 4 + 2.0 * pair(k1_t, k1_z) / (lt * lz) ** 2
                          + pair(m_t, k2_z) / lz ** 4 + pair(m_t, m_z) / self.length_l2 ** 4)


def flux_seed(seq):
    """The histopolated flux 2-form ``dr ^ dchi`` (reference proxy ``(0, 0,
    1)``) of the Dirichlet space: closed and geometry-free, the seed of
    :func:`vacuum_two_form`, as in :func:`mrx.nullspace.compute_nullspaces`."""
    flux = jnp.asarray((0.0, 0.0, 1.0), dtype=seq.dtype)
    return seq.interpolate(lambda x_hat: flux, 2, dirichlet=True, frame='ref')


def vacuum_two_form(seq, seed):
    """``(h, info)``: the harmonic 2-form ``h = seed - G_1 a`` of the
    Dirichlet complex, ``S_1 a = G_1^T M_2 seed``, on ``seq`` (a
    :func:`with_geometry` copy), differentiable in the geometry.

    The solve is the production ``L_1`` solve with the bundle's
    preconditioners inside :func:`jax.lax.custom_linear_solve` with the
    matvec ``S_1``: the derivative is ``S_1^+`` of the tangent right-hand
    side ``G_1^T dM_2 h``, and reverse mode is one more solve (``S_1`` is
    symmetric). ``h`` is not normalised: it carries the seed's toroidal
    flux. ``info`` is the forward solve's signed iteration count
    (:func:`mrx.solvers.refine`).
    """
    b = op.apply_derivative_matrix(seq, seed, 1, dirichlet_in=True, dirichlet_out=True,
                                   transpose=True)

    def matvec(x):
        return op.apply_stiffness(seq, x, 1, dirichlet=True)

    def solve(_, r):
        return op.apply_inverse_laplacian(seq, seq.operators, r, 1, dirichlet=True,
                                          return_info=True)

    a, info = jax.lax.custom_linear_solve(matvec, b, solve, symmetric=True, has_aux=True)
    return seed - op.apply_incidence_matrix(seq, a, 1, dirichlet_in=True, dirichlet_out=True), info


def _radial_moments(seq, h):
    """``(A, C)``: the ``(theta, zeta)`` averages of the reference densities
    ``B^theta`` and ``B^zeta`` of the 2-form ``h`` as coefficient vectors on
    the radial derivative splines, ``<B^theta>(r) = sum_i A_i D_i(r)``.
    The theta component lives on ``D_r x B_theta x D_zeta``, the zeta
    component on ``D_r x D_theta x B_zeta``; a derivative spline integrates
    to one, a periodic spline to its support over ``p + 1``."""
    _, (n_rt, n_tt, n_zt), (n_rz, n_tz, n_zz) = seq.basis_2.shape
    raw = seq.E(2, True).T @ h
    n0, n1 = seq.basis_2.n1, seq.basis_2.n2
    lt, lz = seq.basis_0.Λ[1], seq.basis_0.Λ[2]

    def integrals(b):
        return (b.T[b.p + 1:b.p + 1 + b.n] - b.T[:b.n]) / (b.p + 1)
    A = jnp.einsum("ijk,j->i", raw[n0:n0 + n1].reshape(n_rt, n_tt, n_zt), integrals(lt))
    C = jnp.einsum("ijk,k->i", raw[n0 + n1:].reshape(n_rz, n_tz, n_zz), integrals(lz))
    return A, C


def _flux_tables(seq, C, rho):
    """``(s, ds/drho)`` at the logical radii ``rho``: the normalised toroidal
    flux ``s = int_0^rho <B^zeta> / int_0^1 <B^zeta>`` from the moments
    ``C`` (the antiderivative of ``sum_i C_i D_i`` is ``sum_l B_l(rho)
    sum_{i < l} C_i``, ``B'_l = D_{l-1} - D_l``) and its density."""
    flux = jnp.concatenate([jnp.zeros(1, C.dtype), jnp.cumsum(C)])
    return ((flux @ basis_table(seq.basis_0.Λ[0], rho)) / flux[-1],
            (C @ basis_table(seq.basis_0.dΛ[0], rho)) / flux[-1])


def flux_ratio_iota(seq, h, rho):
    """``(iota, s)`` at the logical radii ``rho`` of the 2-form ``h``:
    ``iota = nfp <B^theta> / <B^zeta>``, the averages over the logical
    surface (one unit of zeta is one field period), and the normalised
    toroidal flux ``s`` inside it, both exact from the DoFs. The rotational
    transform where the logical surfaces are flux surfaces
    (:func:`normal_field_fraction`); its sign is that of the logical
    orientation."""
    A, C = _radial_moments(seq, h)
    D = basis_table(seq.basis_0.dΛ[0], rho)
    return seq.nfp * (A @ D) / (C @ D), _flux_tables(seq, C, rho)[0]


def flux_surface_radius(seq, h, s, newton_steps=8):
    """The logical radii of the flux labels ``s`` (normalised toroidal flux
    of ``h``, :func:`flux_ratio_iota`): Newton on ``s(rho)`` from ``rho =
    sqrt(s)`` with the gradient stopped, then one differentiable step, whose
    derivative at the root is the implicit one, ``-(ds/dalpha) / (ds/drho)``."""
    _, C = _radial_moments(seq, h)
    C0 = jax.lax.stop_gradient(C)
    rho = jnp.sqrt(s)
    for _ in range(newton_steps):
        val, slope = _flux_tables(seq, C0, rho)
        rho = rho - (val - s) / slope
    val, slope = _flux_tables(seq, C, rho)
    return rho - (val - s) / slope


def normal_field_fraction(seq, h):
    """``int (B . n)^2 / int |B|^2`` over the volume, ``n`` the unit normal of
    the logical surfaces: ``(B . n)^2 J = (B^r)^2 / (J g^rr)`` and ``|B|^2 J =
    B^T G B / J`` for the reference densities ``B``. Zero when the logical
    surfaces are flux surfaces of ``h``."""
    b = seq.evaluate_at_quadrature(h, 2, True)
    w, J = seq.quad.w, seq.jacobian_j
    normal = jnp.sum(w * b[:, 0] ** 2 / (J * seq.metric_inv_jkl[:, 0, 0]))
    return normal / jnp.sum(w * jnp.einsum("qi,qij,qj->q", b, seq.metric_jkl, b) / J)
