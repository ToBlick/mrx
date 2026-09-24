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
(the magnetic axis is free to move); for large changes, harmonically in
the logical disc, poloidal mode by mode, which moves the coordinate axis
with the mean of the change. The perturbation is projected onto
stellarator symmetry (``R`` even, ``Z`` odd), and the coefficients are
rescaled so that the volume stays that of the start map exactly (a uniform
scaling leaves the vacuum rotational transform unchanged). Optionally the
aspect ratio is held exactly too, by scaling every cross-section about the
coordinate axis first (:meth:`BoundaryShape.map_coefficients`). Or every
coefficient is free (``free="all"``): the extended boundary change plus a
change of every ring inside, the axis (ring 0) included, with ring 1 kept
pure ``m = 1`` about it (C1). Then the logical surfaces bend with the
variables, which can fool :func:`flux_ratio_iota` unless
:func:`normal_field_fraction` holds them to the flux surfaces.

**The objectives** (:func:`flux_ratio_iota`): ``iota = nfp <B^theta> /
<B^zeta>`` on the logical surfaces, the averages over ``(theta, zeta)``
taken exactly from the DoFs. It is the rotational transform wherever the
logical surfaces are flux surfaces; :func:`normal_field_fraction` measures
how far they are from that. :func:`quasisymmetry_residual` is the two-term
quasi-axisymmetry residual on the same surfaces, from the spline
derivatives of the field and of the map at the quadrature points, and
:func:`edge_quasisymmetry_residual` the same on the boundary alone, which is
a flux surface exactly.

On a half-period sequence (:mod:`mrx.symmetry`) every field here is odd, the
geometry even, and the parametrisation stellarator-symmetric by
construction, so the half-period quadrature is exact for the objective and
for its gradient with respect to the parameters.
"""
import copy
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mrx.operators as op
from mrx.differential_forms import DifferentialForm, inv33
from mrx.geometry import SequenceGeometry, _tp_evaluate, grad_1d
from mrx.mappings import stellarator_symmetric_scalar
from mrx.mass import attach_weights
from mrx.quadrature import composite_quad, evaluate_at_xq
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
    R, dR, dZ = _cylindrical_fields(seq, raw_R, raw_Z)
    a = 2.0 * np.pi / nfp
    G = dR[:, :, None] * dR[:, None, :] + dZ[:, :, None] * dZ[:, None, :]
    G = G.at[:, 2, 2].add((a * R) ** 2)
    J = sign * a * R * (dR[:, 1] * dZ[:, 0] - dR[:, 0] * dZ[:, 1])
    return SequenceGeometry(None, G, jax.vmap(inv33)(G), J)


def _cylindrical_fields(seq, raw_R, raw_Z):
    """``(R, dR, dZ)`` at the quadrature points: ``R`` ``(n_q,)`` and the
    logical gradients of ``R`` and ``Z`` ``(n_q, 3)``, sum-factorised."""
    Br, Bt, Bz = seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk
    Dr, Dt, Dz = (grad_1d(d, t) for d, t in zip(
        (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk), seq.basis_0.types))
    C = jnp.stack([raw_R, raw_Z])
    R = _tp_evaluate(C[:1], Br, Bt, Bz)[0].reshape(-1)
    d = jnp.stack([_tp_evaluate(C, Dr, Bt, Bz), _tp_evaluate(C, Br, Dt, Bz),
                   _tp_evaluate(C, Br, Bt, Dz)], axis=-1)            # (2, nqr, nqt, nqz, 3)
    return R, d[0].reshape(-1, 3), d[1].reshape(-1, 3)


def section_moments(seq, raw_R, raw_Z, nfp, sign):
    """``(V, V_axis, S)`` of the map of :func:`cylindrical_geometry`, over
    one field period by the quadrature: the volume ``V = int J``, its part
    ``V_axis = int R_axis J / R`` (``R_axis(zeta)`` the coordinate axis, the
    ring-0 coefficients), and the mean cross-section area ``S = int J / (2
    pi R / nfp)`` (``dR dZ = J / (2 pi R / nfp) dr dtheta`` at fixed
    ``zeta``). ``J / R`` is the in-plane Jacobian, so a scaling of every
    cross-section by ``mu`` about the axis takes ``S`` to ``mu^2 S`` and
    ``V`` to ``mu^2 V_axis + mu^3 (V - V_axis)`` exactly, quadrature
    included."""
    R, dR, dZ = _cylindrical_fields(seq, raw_R, raw_Z)
    a = 2.0 * np.pi / nfp
    J_over_R = sign * a * (dR[:, 1] * dZ[:, 0] - dR[:, 0] * dZ[:, 1])
    axis = jnp.broadcast_to(raw_R[0, 0] @ seq.basis_z_jk, seq.quad.shape).reshape(-1)
    w = seq.quad.w
    return jnp.sum(w * R * J_over_R), jnp.sum(w * axis * J_over_R), jnp.sum(w * J_over_R) / a


def aspect_ratio(V, S, nfp):
    """VMEC's aspect ratio ``R_major / a_minor = V_torus / (2 sqrt(pi)
    S^(3/2))`` from the volume ``V`` of one field period and the mean
    cross-section area ``S`` (:func:`section_moments`), ``a_minor =
    sqrt(S / pi)``, ``R_major = V_torus / (2 pi S)``."""
    return nfp * V / (2.0 * np.sqrt(np.pi) * S ** 1.5)


class BoundaryShape(eqx.Module):
    """The map's ``R`` and ``Z`` coefficients as a function of the variables
    ``beta``, an array ``(2, n_theta, n_zeta)``: ``beta[0]`` the change of
    the outermost ring of the ``R`` coefficients, ``beta[1]`` of the ``Z``
    ones.

    The change is projected onto stellarator symmetry (``R`` even, ``Z``
    odd, :meth:`perturbation`) and extended inwards ring by ring, poloidal
    mode by poloidal mode (:meth:`extension_weights`): the ``ramp``, ``w_i =
    ((i - 1) / (n_r - 2))^2`` on ring ``i >= 1`` for every mode, zero on the
    two innermost rings, which carry the C1 polar structure and the
    coordinate axis; or the ``harmonic`` extension, ``rho_i^|m|`` on mode
    ``m`` of ring ``i`` (``rho_i`` the ring's Greville abscissa), the
    extension of ``exp(2 pi i m theta)`` harmonic in the logical disc, which
    moves every ring, and the coordinate axis with them, by the ``m = 0``
    part of the change (ring 1 keeps only ``|m| <= 1``: it stays pure ``m =
    1`` about the axis). One on the boundary in both; ``none`` is the
    boundary ring alone.

    With ``free="all"`` every coefficient is a variable: ``beta`` is
    ``(2, n_r, n_theta, n_zeta)``, its outermost ring the boundary change,
    extended as above, and every ring inside a change of its own on top
    (``interior``, :meth:`interior_modes`): ``m = 0`` on ring 0, the axis;
    ``|m| = 1`` on ring 1, whose ``m = 0`` part is ring 0's, so it stays pure
    ``m = 1`` about the moved axis. :meth:`map_coefficients` holds the
    volume, and with ``aspect`` the aspect ratio, exactly.
    """

    raw_R: jnp.ndarray
    raw_Z: jnp.ndarray
    extension: jnp.ndarray
    volume: jnp.ndarray
    nfp: int = eqx.field(static=True)
    sign: float = eqx.field(static=True)
    basis_0: DifferentialForm = eqx.field(static=True)
    aspect: Optional[float] = eqx.field(static=True, default=None)
    interior: Optional[jnp.ndarray] = None

    @classmethod
    def from_coefficients(cls, seq, raw_R, raw_Z, nfp, sign, volume=None, aspect=None, extension="ramp",
                          free="boundary"):
        """The parametrisation about the map with raw coefficients ``raw_R``,
        ``raw_Z`` (``nfp`` and handedness ``sign`` as in
        :func:`cylindrical_geometry`). ``volume`` (of one field period) is
        the one kept, that map's by default; ``aspect`` the aspect ratio
        held (:func:`aspect_ratio`), none by default; ``extension`` the
        interior extension of the boundary change, ``"ramp"``,
        ``"harmonic"`` or ``"none"``; ``free`` the variables, the boundary ring
        (``"boundary"``) or every ring (``"all"``)."""
        if volume is None:
            volume = section_moments(seq, raw_R, raw_Z, nfp, sign)[0]
        weights = cls.extension_weights(seq.basis_0, extension)
        interior = jnp.asarray(cls.interior_modes(seq.basis_0), dtype=raw_R.dtype) if free == "all" else None
        return cls(raw_R, raw_Z, jnp.asarray(weights, dtype=raw_R.dtype), jnp.asarray(volume), int(nfp),
                   float(sign), seq.basis_0, None if aspect is None else float(aspect), interior)

    @property
    def free(self):
        """``"boundary"`` or ``"all"``: the variables."""
        return "boundary" if self.interior is None else "all"

    @staticmethod
    def extension_weights(basis_0, extension):
        """``(n_r, n_theta)``: the weight of poloidal mode ``m`` (FFT order over
        the ring's theta coefficients) of the boundary change on ring ``i``."""
        n_r, n_t = basis_0.Λ[0].n, basis_0.Λ[1].n
        m = np.abs(np.fft.fftfreq(n_t, 1.0 / n_t))
        if extension == "ramp":
            return np.broadcast_to(((np.maximum(np.arange(n_r) - 1, 0) / (n_r - 2)) ** 2)[:, None], (n_r, n_t))
        if extension == "none":
            return np.concatenate([np.zeros((n_r - 1, n_t)), np.ones((1, n_t))])
        if extension != "harmonic":
            raise ValueError(f"extension must be 'ramp', 'harmonic' or 'none', got {extension!r}")
        rho = np.asarray(basis_0.Λ[0].greville_points(), dtype=np.float64)
        weights = rho[:, None] ** m[None, :]
        weights[1, m >= 2] = 0.0
        return weights

    @staticmethod
    def interior_modes(basis_0):
        """``(n_r, n_theta)``: the poloidal modes of ring ``i`` free on their
        own with ``free="all"``, ``m = 0`` on ring 0, ``|m| = 1`` on ring 1
        (its ``m = 0`` part is ring 0's), all on the rings inside, none on the
        boundary ring (its change is the extended one)."""
        n_r, n_t = basis_0.Λ[0].n, basis_0.Λ[1].n
        m = np.abs(np.fft.fftfreq(n_t, 1.0 / n_t))
        modes = np.ones((n_r, n_t))
        modes[0], modes[1], modes[-1] = m == 0, m == 1, 0.0
        return modes

    def perturbation(self, beta):
        """``(b_R, b_Z)``, the change ``beta`` projected onto ``R`` even and
        ``Z`` odd under ``(theta, zeta) -> (-theta, -zeta)``, of the boundary
        ring or of every ring."""
        if self.interior is not None:
            return (stellarator_symmetric_scalar(beta[0], self.basis_0, even=True),
                    stellarator_symmetric_scalar(beta[1], self.basis_0, even=False))
        return (stellarator_symmetric_scalar(beta[0][None], self.basis_0, even=True)[0],
                stellarator_symmetric_scalar(beta[1][None], self.basis_0, even=False)[0])

    def change(self, beta):
        """``(2, n_r, n_theta, n_zeta)``: the change of the raw ``R`` and ``Z``
        coefficients at ``beta``, before the scalings: the extended boundary
        change, plus with ``free="all"`` the :meth:`interior_change`."""
        b = jnp.stack(self.perturbation(beta))
        if self.interior is None:
            spectrum = jnp.fft.fft(b, axis=1)                                            # (2, n_t, n_z)
            return jnp.fft.ifft(self.extension[None, :, :, None] * spectrum[:, None], axis=2).real
        boundary = jnp.fft.fft(b[:, -1:], axis=2)                                         # (2, 1, n_t, n_z)
        return jnp.fft.ifft(self.extension[None, :, :, None] * boundary, axis=2).real + self.interior_change(beta)

    def interior_change(self, beta):
        """``(2, n_r, n_theta, n_zeta)``: the inner rings' own change at
        ``beta`` (``free="all"``), the change less the extended boundary
        change; zero on the boundary ring."""
        spectrum = jnp.fft.fft(jnp.stack(self.perturbation(beta)), axis=2)              # (2, n_r, n_t, n_z)
        own = (self.interior[None, :, :, None] * spectrum).at[:, 1, 0].set(spectrum[:, 0, 0])
        return jnp.fft.ifft(own, axis=2).real

    def map_coefficients(self, seq, beta):
        """``(R, Z, mu, scale)``: the raw coefficients of the map at ``beta``
        and the two scalings that made them. With ``aspect``, every
        cross-section is scaled by ``mu`` about the coordinate axis, ``(R,
        Z) -> axis + mu ((R, Z) - axis)`` on every ring (ring 0 stays the
        axis, ring 1 stays pure ``m = 1``), which leaves ``V_axis / mu + V -
        V_axis`` (:func:`section_moments`) proportional to the aspect ratio,
        so ``mu = V_axis / (2 sqrt(pi) S^(3/2) aspect / nfp - V + V_axis)``
        in closed form; ``mu = 1`` without. Then the whole map is scaled by
        ``scale = (volume / V(mu))^(1/3)``, which keeps the aspect ratio."""
        rings = self.change(beta)
        R, Z = self.raw_R + rings[0], self.raw_Z + rings[1]
        V, V_axis, S = section_moments(seq, R, Z, self.nfp, self.sign)
        mu = jnp.ones_like(V)
        if self.aspect is not None:
            mu = V_axis / (2.0 * np.sqrt(np.pi) * S ** 1.5 * self.aspect / self.nfp - V + V_axis)
            R = R[:1, :1] + mu * (R - R[:1, :1])
            Z = Z[:1, :1] + mu * (Z - Z[:1, :1])
            V = mu ** 2 * V_axis + mu ** 3 * (V - V_axis)
        scale = (self.volume / V) ** (1.0 / 3.0)
        return scale * R, scale * Z, mu, scale

    def geometry(self, seq, beta):
        """The geometry of the map at ``beta`` (:meth:`map_coefficients`)."""
        R, Z, _, _ = self.map_coefficients(seq, beta)
        return cylindrical_geometry(seq, R, Z, self.nfp, self.sign)


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

    On a half-period sequence every right-hand side is projected onto the
    odd dual vectors, the parity of ``h``. The forward one is odd already.
    The adjoint one, ``-G_1^T dJ/dh``, is not wherever the half-period
    quadrature of the objective is not exact for it (the products of ``h``
    with a direction of the other parity): that part pairs to zero with
    every odd tangent, and left in, it decides the parity the solver reads
    off its right-hand side once the odd part is small (near an optimum of
    a quadratic objective such as :func:`quasisymmetry_residual`), which
    then solves for the wrong one.
    """
    b = op.apply_derivative_matrix(seq, seed, 1, dirichlet_in=True, dirichlet_out=True,
                                   transpose=True)
    parity = seq.free_projector(1, True)

    def matvec(x):
        return op.apply_stiffness(seq, x, 1, dirichlet=True)

    def solve(_, r):
        if parity is not None:
            r = parity.dual(r, -1.0)
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


def mean_iota(seq, h):
    """``int_0^1 iota ds`` of the flux-ratio iota (:func:`flux_ratio_iota`)
    over the normalised toroidal flux: ``ds = <B^zeta> dr / sum C`` turns it
    into ``nfp sum_i A_i / sum_i C_i`` (the derivative splines integrate to
    one), the ratio of the poloidal to the toroidal flux per field period
    times ``nfp``. simsopt's ``mean_iota`` of a VMEC equilibrium."""
    A, C = _radial_moments(seq, h)
    return seq.nfp * jnp.sum(A) / jnp.sum(C)


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


def _tables(seq, radii=None):
    """``(prim, der, dprim, dder, ddprim, shape)``: per axis the 1-D tables
    ``(n, n_points)`` of the 0-form splines, of their derivative splines, of
    the derivatives of both (``B'_l = D_{l-1} - D_l``, the tabulated ``D'``)
    and the second derivatives of the 0-form splines, on the tensor grid of
    the volume quadrature points, or with ``radii`` of those radii and the
    angular quadrature points (a surface rule); ``shape`` is the grid's."""
    types = seq.basis_0.types
    if radii is None:
        r_prim, r_der, r_dder = seq.basis_r_jk, seq.d_basis_r_jk, seq.dd_basis_jk[0]
    else:
        radii = jnp.asarray(radii)
        r_prim, r_der = basis_table(seq.basis_0.Λ[0], radii), basis_table(seq.basis_0.dΛ[0], radii)
        r_dder = basis_derivative_table(seq.basis_0.dΛ[0], radii)
    prim = (r_prim, seq.basis_t_jk, seq.basis_z_jk)
    der = (r_der, seq.d_basis_t_jk, seq.d_basis_z_jk)
    dder = (r_dder, seq.dd_basis_jk[1], seq.dd_basis_jk[2])
    dprim = tuple(grad_1d(d, t) for d, t in zip(der, types))
    ddprim = tuple(grad_1d(dd, t) for dd, t in zip(dder, types))
    return prim, der, dprim, dder, ddprim, (int(r_prim.shape[1]), seq.quad.shape[1], seq.quad.shape[2])


def _two_form_derivatives(seq, h, tables):
    """``(b, db)``: the reference components of the Dirichlet 2-form ``h`` on
    the grid of ``tables`` (:func:`_tables`), ``(n_q, 3)``, and their logical
    derivatives, ``(3, n_q, 3)`` with ``db[a, q, i] = d_a b^i``. Component
    ``c`` lives on the primal splines on axis ``c`` and on the derivative
    splines on the others, and ``d_a`` replaces the axis-``a`` table by its
    derivative."""
    prim, der, dprim, dder, _, grid = tables
    shapes = [tuple(int(v) for v in sh) for sh in seq.basis_2.shape]
    raw = seq.E(2, True).T @ h

    def info(a):
        out = []
        for c in range(3):
            tabs = [prim[i] if i == c else der[i] for i in range(3)]
            if a is not None:
                tabs[a] = dprim[a] if a == c else dder[a]
            out.append((c, *tabs))
        return out
    b = evaluate_at_xq(raw, info(None), shapes, grid, 3)
    return b, jnp.stack([evaluate_at_xq(raw, info(a), shapes, grid, 3) for a in range(3)])


def _cylindrical_derivatives(raw_R, raw_Z, tables):
    """``(v, d1, d2)``: ``(R, Z)`` of the raw coefficients on the grid of
    ``tables`` (:func:`_tables`), ``(2, n_q)``, their first logical
    derivatives ``d1[q, c, i] = d_i c`` ``(n_q, 2, 3)``, and their second
    ones ``d2[q, c, i, j]`` ``(n_q, 2, 3, 3)``, sum-factorised."""
    prim, _, dprim, _, ddprim, _ = tables
    tabs = (prim, dprim, ddprim)
    C = jnp.stack([raw_R, raw_Z])

    def ev(orders):
        return _tp_evaluate(C, *(tabs[o][a] for a, o in enumerate(orders))).reshape(2, -1)
    unit = np.eye(3, dtype=int)
    d1 = jnp.stack([ev(unit[i]) for i in range(3)], -1)
    d2 = jnp.stack([jnp.stack([ev(unit[i] + unit[j]) for j in range(3)], -1) for i in range(3)], -2)
    return ev((0, 0, 0)), jnp.moveaxis(d1, 1, 0), jnp.moveaxis(d2, 1, 0)


def _residual_terms(seq, h, raw_R, raw_Z, nfp, sign, tables):
    """``(B, parallel, cross, B_cov, J)`` on the grid of ``tables``: ``|B|``,
    ``B . grad|B|``, ``(B x grad r) . grad|B|``, the covariant components of
    ``B`` and ``det DF``, from the spline derivatives of ``h`` and of the
    map (:func:`quasisymmetry_residual`)."""
    a = 2.0 * np.pi / nfp
    (R, _), d1, d2 = _cylindrical_derivatives(raw_R, raw_Z, tables)
    dR, dZ, ddR, ddZ = d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1]
    G = jnp.einsum("qi,qj->qij", dR, dR) + jnp.einsum("qi,qj->qij", dZ, dZ)
    G = G.at[:, 2, 2].add((a * R) ** 2)
    dG = (jnp.einsum("qik,qj->qkij", ddR, dR) + jnp.einsum("qi,qjk->qkij", dR, ddR)
          + jnp.einsum("qik,qj->qkij", ddZ, dZ) + jnp.einsum("qi,qjk->qkij", dZ, ddZ))
    dG = dG.at[:, :, 2, 2].add(2.0 * a ** 2 * R[:, None] * dR)
    cross_rz = dR[:, 1] * dZ[:, 0] - dR[:, 0] * dZ[:, 1]
    J = sign * a * R * cross_rz
    dJ = sign * a * (dR * cross_rz[:, None] + R[:, None] * (
        ddR[:, 1] * dZ[:, 0:1] + dR[:, 1:2] * ddZ[:, 0] - ddR[:, 0] * dZ[:, 1:2] - dR[:, 0:1] * ddZ[:, 1]))

    b, db = _two_form_derivatives(seq, h, tables)
    Gb = jnp.einsum("qij,qj->qi", G, b)
    B2 = jnp.sum(b * Gb, -1) / J ** 2
    dB2 = ((2.0 * jnp.einsum("qi,kqi->qk", Gb, db) + jnp.einsum("qi,qkij,qj->qk", b, dG, b)) / J[:, None] ** 2
           - 2.0 * B2[:, None] * dJ / J[:, None])
    B = jnp.sqrt(B2)
    dB = dB2 / (2.0 * B[:, None])
    B_cov = Gb / J[:, None]
    return (B, jnp.sum(b * dB, -1) / J, (dB[:, 1] * B_cov[:, 2] - dB[:, 2] * B_cov[:, 1]) / J, B_cov, J)


def quasisymmetry_residual(seq, h, raw_R, raw_Z, nfp, sign, r_min):
    """``(F, F_parallel)``: the volume average over ``r >= r_min`` of the
    squared two-term quasi-axisymmetry residual of the vacuum 2-form ``h`` on
    the map of the raw coefficients ``raw_R``, ``raw_Z``
    (:func:`cylindrical_geometry`),

        f = (G B . grad|B| - iota (B x grad psi) . grad|B|) / |B|^3,

    zero where ``|B|`` does not depend on the Boozer toroidal angle (the
    toroidal current ``I`` vanishes in a vacuum). ``psi`` is the toroidal
    flux over ``2 pi`` with the logical surfaces as its level sets, so
    ``grad psi = psi'(r) grad r``, ``iota`` the flux ratio
    (:func:`flux_ratio_iota`, signed), ``G`` the circulation of ``B`` once
    around the torus over ``2 pi``, the mean of the covariant ``B_zeta`` over
    the logical domain times ``nfp / 2 pi``. With the reference densities
    ``b`` and ``J = det DF``: ``B . grad|B| = b^k d_k|B| / J``,
    ``(B x grad psi) . grad|B| = psi' (d_theta|B| B_zeta - d_zeta|B|
    B_theta) / J``, ``B_i = g_ij b^j / J``, and ``d_k|B|`` from the spline
    derivatives of ``b`` and of the map. Exact where the logical surfaces are
    flux surfaces (:func:`normal_field_fraction`), like the flux ratio.
    ``F_parallel`` is the same average of the first term alone, the scale the
    two terms cancel from."""
    B, parallel, cross, B_cov, J = _residual_terms(seq, h, raw_R, raw_Z, nfp, sign, _tables(seq))
    A, C = _radial_moments(seq, h)
    D = basis_table(seq.basis_0.dΛ[0], seq.quad.x_x)
    radial = seq.quad.shape[1] * seq.quad.shape[2]
    iota = jnp.repeat(nfp * (A @ D) / (C @ D), radial)
    psi_prime = jnp.repeat((C @ D) / (2.0 * np.pi), radial)
    w = seq.quad.w
    G_circ = nfp / (2.0 * np.pi) * jnp.sum(w * B_cov[:, 2]) / jnp.sum(w)
    f = (G_circ * parallel - iota * psi_prime * cross) / B ** 3
    mask = jnp.repeat(seq.quad.x_x >= r_min, radial)
    weight = jnp.where(mask, w * J, 0.0)
    return (jnp.sum(weight * f ** 2) / jnp.sum(weight),
            jnp.sum(weight * (G_circ * parallel / B ** 3) ** 2) / jnp.sum(weight))


def edge_quasisymmetry_residual(seq, h, raw_R, raw_Z, nfp, sign, eps=1e-6):
    """``<f^2>`` on the logical surface ``r = 1 - eps``, the residual ``f`` of
    :func:`quasisymmetry_residual` (the same ``G``, from the volume), averaged
    with the volume Jacobian, ``int f^2 J dtheta dzeta / int J dtheta
    dzeta``, by the Gauss rule of the angular quadrature points: the
    thin-shell limit of the volume average. The boundary ``r = 1`` is a flux
    surface of ``h`` exactly (``B . n = 0``), so it needs no alignment of the
    logical surfaces; ``eps > 0`` because the end-point derivative tables of
    the clamped radial splines are one-sided at ``r = 1`` exactly."""
    b = seq.evaluate_at_quadrature(h, 2, True)
    w = seq.quad.w
    B_zeta = jnp.einsum("qj,qj->q", seq.metric_jkl[:, 2], b) / seq.jacobian_j
    G_circ = nfp / (2.0 * np.pi) * jnp.sum(w * B_zeta) / jnp.sum(w)
    B, parallel, cross, _, J = _residual_terms(seq, h, raw_R, raw_Z, nfp, sign, _tables(seq, [1.0 - eps]))
    A, _ = _radial_moments(seq, h)
    D = basis_table(seq.basis_0.dΛ[0], jnp.asarray([1.0 - eps]))[:, 0]
    # iota psi' = nfp <B^theta> / 2 pi: the flux ratio times the toroidal flux density
    f = (G_circ * parallel - nfp * (A @ D) / (2.0 * np.pi) * cross) / B ** 3
    weight = (seq.quad.w_y[:, None] * seq.quad.w_z[None, :]).reshape(-1) * J
    return jnp.sum(weight * f ** 2) / jnp.sum(weight)
