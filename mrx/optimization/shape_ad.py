"""Shape optimisation of a vacuum field by automatic differentiation.

This module makes the vacuum magnetic field of a stellarator a differentiable function of the shape of its
domain, so that ``jax.grad`` of an objective such as a quasi-symmetry residual gives the derivative with
respect to the shape. The shape is the map Phi from logical to physical coordinates, a
:class:`~mrx.equilibria.FrameMap` ``Phi = R_z(phi) (a + X1 N + X2 B + X3 T)``, ``phi = 2 pi zeta / nfp``, with
``X1``, ``X2``, ``X3`` scalar splines and the frame ``(a, N, B)`` a Fourier series in ``zeta``. Every coefficient of
the map, of ``X1``, ``X2``, ``X3`` and of the frame, is a variable. The shapes of :class:`BoundaryShape` keep ``X3``
of their start, zero for a map read from a file. In the cylindrical frame ``X1``, ``X2`` are ``R``, ``Z``.

The vacuum field is the harmonic 2-form ``h = s - G_1 a`` of the complex with Dirichlet boundary conditions.
Here ``s`` is a fixed seed that carries the toroidal flux, ``G_1`` the discrete curl, ``M_2`` the 2-form mass
matrix, and ``a`` solves ``S_1 a = G_1^T M_2 s`` with ``S_1 = G_1^T M_2 G_1``. A gradient costs one forward
and one adjoint solve of this system.

A typical use: build a :class:`BoundaryShape` about a starting map, turn its variables into map coefficients
(:meth:`BoundaryShape.map_at`) and a geometry (:func:`frame_geometry`), install it with
:meth:`~mrx.derham_sequence.DeRhamSequence.with_geometry`, compute the field with :func:`flux_seed` and :func:`vacuum_two_form`, and evaluate an
objective: :func:`mean_iota`, :func:`normal_field_fraction` or :func:`quasisymmetry_residual`. The objectives treat the logical surfaces as flux surfaces, and
:func:`normal_field_fraction` measures how far that is from true. The preconditioners of the solves are
those of the sequence's own geometry and are not rebuilt as the shape changes.
"""
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DifferentialForm, inv33
from mrx.equilibria import frame_derivatives
from mrx.geometry import SequenceGeometry
from mrx.incidence import diff_1d
from mrx.symmetry import stellarator_symmetric_scalar
from mrx.quadrature import evaluate_at_xq
from mrx.spline_bases import basis_table


def _grad_1d(d_basis, boundary_type):
    """The derivatives of the splines of one axis, ``(n, nq)``, from the values ``d_basis`` of the
    derivative splines ``D_l`` of that axis: ``B'_l = D_{l-1} - D_l``, the transposed 1-D incidence.
    ``boundary_type`` is ``'clamped'`` or ``'periodic'``."""
    n = d_basis.shape[0] + (boundary_type == 'clamped')
    return jnp.asarray(diff_1d(n, boundary_type).toarray()).T @ d_basis


def _tp_evaluate(C_raw, M1, M2, M3):
    """A tensor-product spline ``sum_abc C_raw[i, a, b, c] M1[a, I] M2[b, J] M3[c, K]`` evaluated on a grid,
    ``(len(C_raw), nqr, nqt, nqz)``."""
    T1 = jnp.einsum("iabc,aI->iIbc", C_raw, M1)
    T2 = jnp.einsum("iIbc,bJ->iIJc", T1, M2)
    return jnp.einsum("iIJc,cK->iIJK", T2, M3)


def _coordinate_derivatives(seq, Phi, second=False):
    """``(X, d1, d2)``: ``(X1, X2, X3)`` of the map ``Phi`` at the quadrature points, ``(n_q, 3)``, their first
    logical derivatives ``d1[q, c, i]``, ``(n_q, 3, 3)``, and with ``second`` their second ones ``d2[q, c, i, j]``,
    ``(n_q, 3, 3, 3)`` (else ``None``)."""
    prim, _, dprim, _, ddprim, _ = _tables(seq)
    tabs = (prim, dprim, ddprim)
    C = jnp.stack([Phi.raw_X1, Phi.raw_X2, Phi.raw_X3])

    def ev(orders):
        return _tp_evaluate(C, *(tabs[o][a] for a, o in enumerate(orders))).reshape(3, -1).T
    unit = np.eye(3, dtype=int)
    d1 = jnp.stack([ev(unit[i]) for i in range(3)], -1)
    d2 = (jnp.stack([jnp.stack([ev(unit[i] + unit[j]) for j in range(3)], -1) for i in range(3)], -2)
          if second else None)
    return ev((0, 0, 0)), d1, d2


def _turn(v):
    """``e_z x v`` for the rows of ``v``, ``(n_q, 3)``."""
    return jnp.stack([-v[:, 1], v[:, 0], jnp.zeros_like(v[:, 0])], -1)


def _frame_vectors(seq, Phi):
    """``(a, E)``, each with its first and second ``zeta`` derivatives: the axis ``(n_q, 3)`` and the directions of
    ``X1``, ``X2``, ``X3`` at the quadrature points, ``E[q, c]`` = ``N``, ``B``, ``T = N x B``, ``(n_q, 3, 3)``."""
    F = jax.vmap(lambda z: frame_derivatives(Phi.frame_cos, Phi.frame_sin, Phi.frame_n, z))(seq.quad.x_z)
    (a, N, B), (da, dN, dB), (dda, ddN, ddB) = (
        tuple(jnp.broadcast_to(f, seq.quad.shape + (3, 3)).reshape(-1, 3, 3)[:, r] for r in range(3)) for f in F)
    T = jnp.cross(N, B)
    dT = jnp.cross(dN, B) + jnp.cross(N, dB)
    ddT = jnp.cross(ddN, B) + 2.0 * jnp.cross(dN, dB) + jnp.cross(N, ddB)
    return (a, da, dda), tuple(jnp.stack(v, 1) for v in ((N, B, T), (dN, dB, dT), (ddN, ddB, ddT)))


def map_jacobian(seq, Phi, second=False):
    """``(D, dD, E)`` of the frame map ``Phi`` at the quadrature points, in the frame that turns with the field
    periods. ``D = R_z(-phi) DPhi``, ``(n_q, 3, 3)`` with ``D[q, :, i]`` the derivative in the logical direction
    ``i``, has the metric and the determinant of ``DPhi``. With ``second``, ``dD[q, k]`` are its logical derivatives
    (else ``None``). ``E[q, c]`` are the directions ``N``, ``B``, ``T = N x B`` of ``X1``, ``X2``, ``X3``, ``(n_q, 3,
    3)``: a change ``w`` of the coordinates moves a point by ``sum_c w_c E_c``. With ``P = a + X1 N + X2 B + X3 T``,
    ``D[:, i] = d_i X1 N + d_i X2 B + d_i X3 T`` plus, along ``zeta``, ``a' + X1 N' + X2 B' + X3 T' + (2 pi / nfp)
    e_z x P``, the change of the frame and its turn with the field period."""
    X, d1, d2 = _coordinate_derivatives(seq, Phi, second)
    (a, da, dda), (E, dE, ddE) = _frame_vectors(seq, Phi)
    c = 2.0 * np.pi / Phi.nfp
    P = a + jnp.einsum("qc,qcj->qj", X, E)
    dP = da + jnp.einsum("qc,qcj->qj", X, dE)                       # the change of P along zeta at fixed X
    D = jnp.einsum("qcj,qci->qji", E, d1)
    D = D.at[:, :, 2].add(dP + c * _turn(P))
    if not second:
        return D, None, E
    # d_k D[:, i] = sum_c d_k d_i X_c E_c, plus for k = zeta sum_c d_i X_c E_c', plus in the column i = zeta the
    # derivative of dP + c e_z x P at fixed d X
    dD = jnp.einsum("qcj,qcik->qkji", E, d2)                          # (n_q, k, 3, i)
    dD = dD.at[:, 2].add(jnp.einsum("qcj,qci->qji", dE, d1))
    for k in range(3):
        dPk = jnp.einsum("qcj,qc->qj", E, d1[:, :, k])
        col = jnp.einsum("qcj,qc->qj", dE, d1[:, :, k])
        if k == 2:
            dPk = dPk + dP
            col = col + dda + jnp.einsum("qc,qcj->qj", X, ddE)
        dD = dD.at[:, k, :, 2].add(col + c * _turn(dPk))
    return D, dD, E


def frame_geometry(seq, Phi):
    """The :class:`~mrx.geometry.SequenceGeometry` of the frame map ``Phi`` (:class:`mrx.equilibria.FrameMap`) on
    the quadrature points of ``seq``, differentiable in every coefficient of the map: those of ``X1``, ``X2``,
    ``X3`` (``(n_r, n_theta, n_zeta)`` in ``seq.basis_0``, the form of :func:`mrx.equilibria.build_map`) and those of
    the frame. It is :meth:`~mrx.geometry.SequenceGeometry.from_map` of ``Phi``, evaluated on the spline tables
    instead of by a Jacobian per point."""
    D, _, _ = map_jacobian(seq, Phi)
    G = jnp.einsum("qki,qkj->qij", D, D)
    return SequenceGeometry(Phi, G, jax.vmap(inv33)(G), jnp.linalg.det(D))


def section_moments(seq, Phi):
    """``(V, V_axis, S)`` for the map ``Phi``, over one field period.

    ``V`` is the volume and ``S`` the mean area of the map's cross-sections ``zeta = const``: the planes ``phi =
    const`` in the cylindrical frame, the planes normal to the axis in a G-frame, where the same torus has a smaller
    ``S``. Scaling every cross-section by ``mu`` about the
    coordinate axis (``X1``, ``X2``, ``X3`` of the innermost ring of coefficients) scales the logical ``r`` and
    ``theta`` columns of ``DPhi`` by ``mu`` and makes the ``zeta`` column affine in ``mu``. ``V_axis`` is the
    volume with that column taken at the coordinate axis, so that the scaling takes ``S`` to ``mu^2 S`` and
    ``V`` to ``mu^2 V_axis + mu^3 (V - V_axis)`` exactly."""
    D, _, _ = map_jacobian(seq, Phi)
    axis = Phi.with_coordinates(*(jnp.broadcast_to(raw[:1, :1], raw.shape) for raw in (Phi.raw_X1, Phi.raw_X2,
                                                                                     Phi.raw_X3)))
    D_axis, _, _ = map_jacobian(seq, axis)
    J = jnp.linalg.det(D)
    J_axis = jnp.linalg.det(D.at[:, :, 2].set(D_axis[:, :, 2]))
    # the area element of the cross-section, |d_r P x d_theta P|, both in the plane of the frame
    area = jnp.linalg.norm(jnp.cross(D[:, :, 0], D[:, :, 1]), axis=-1)
    w = seq.quad.w
    return jnp.sum(w * J), jnp.sum(w * J_axis), jnp.sum(w * area)


class BoundaryShape(eqx.Module):
    """A parametrisation of the coefficients of ``X1`` and ``X2`` of a frame map by the optimisation variables
    ``beta``, the frame and ``X3`` held.

    Build it with :meth:`from_map` about a starting map. Every shape it produces is stellarator symmetric
    (``X1`` even and ``X2`` odd under ``(theta, zeta) -> (-theta, -zeta)``) and has the volume of the starting
    map, and also its aspect ratio if ``aspect`` is given (:meth:`map_at`).

    ``beta`` has shape ``(2, n_r, n_theta, n_zeta)``. Its outermost ring is the change of the ``X1`` and ``X2``
    coefficients on the boundary, carried into the interior with the weight ``r_i^|m|`` on ring ``i`` and
    poloidal mode ``m`` (``r_i`` the radius of the ring), except that ring 1 only takes ``|m| <= 1``. The
    ``m = 0`` part then moves the axis as well. The inner rings add their own independent change
    (:meth:`interior_modes`).
    """

    start: object
    extension: jnp.ndarray
    volume: jnp.ndarray
    basis_0: DifferentialForm = eqx.field(static=True)
    interior: jnp.ndarray
    aspect: Optional[float] = eqx.field(static=True, default=None)

    @classmethod
    def from_map(cls, seq, Phi, volume=None, aspect=None):
        """The parametrisation about the frame map ``Phi`` (:class:`mrx.equilibria.FrameMap`). ``volume`` is the
        volume of one field period to hold, by default that of the starting map. ``aspect`` is the aspect ratio
        to hold, by default none: ``nfp V / (2 sqrt(pi) S^(3/2))`` of :func:`section_moments`, measured in the
        cross-sections of the map, so the same torus has another value in another frame. With these defaults, or with
        the starting map's own aspect ratio, ``beta = 0`` gives back the starting map."""
        if volume is None:
            volume = section_moments(seq, Phi)[0]
        weights = cls.extension_weights(seq.basis_0)
        interior = jnp.asarray(cls.interior_modes(seq.basis_0), dtype=Phi.raw_X1.dtype)
        return cls(Phi, jnp.asarray(weights, dtype=Phi.raw_X1.dtype), jnp.asarray(volume), seq.basis_0, interior,
                   None if aspect is None else float(aspect))

    @staticmethod
    def extension_weights(basis_0):
        """The ``(n_r, n_theta)`` weights with which the boundary change enters ring ``i``, per poloidal mode
        ``m`` in FFT order (see the class docstring)."""
        n_t = basis_0.Lambda[1].n
        m = np.abs(np.fft.fftfreq(n_t, 1.0 / n_t))
        r = np.asarray(basis_0.Lambda[0].greville_points(), dtype=np.float64)
        weights = r[:, None] ** m[None, :]
        weights[1, m >= 2] = 0.0
        return weights

    @staticmethod
    def interior_modes(basis_0):
        """A ``(n_r, n_theta)`` mask of the poloidal modes each ring may change on its own:
        ``m = 0`` on ring 0, ``|m| = 1`` on ring 1, every mode on the rings in between and none on the boundary
        ring. Ring 1 takes its ``m = 0`` change from ring 0."""
        n_r, n_t = basis_0.Lambda[0].n, basis_0.Lambda[1].n
        m = np.abs(np.fft.fftfreq(n_t, 1.0 / n_t))
        modes = np.ones((n_r, n_t))
        modes[0], modes[1], modes[-1] = m == 0, m == 1, 0.0
        return modes

    def perturbation(self, beta):
        """``(b_1, b_2)``: the stellarator-symmetric part of ``beta``, ``X1`` even and ``X2`` odd under
        ``(theta, zeta) -> (-theta, -zeta)``."""
        return (stellarator_symmetric_scalar(beta[0], self.basis_0, even=True),
                stellarator_symmetric_scalar(beta[1], self.basis_0, even=False))

    def change(self, beta):
        """The change ``(2, n_r, n_theta, n_zeta)`` of the ``X1`` and ``X2`` coefficients at ``beta``, before the
        volume and aspect-ratio scalings: the boundary ring carried inwards plus the inner rings' own change."""
        spectrum = jnp.fft.fft(jnp.stack(self.perturbation(beta)), axis=2)              # (2, n_r, n_t, n_z)
        boundary = self.extension[None, :, :, None] * spectrum[:, -1:]
        own = (self.interior[None, :, :, None] * spectrum).at[:, 1, 0].set(spectrum[:, 0, 0])
        return jnp.fft.ifft(boundary + own, axis=2).real

    def map_at(self, seq, beta):
        """The frame map at ``beta``. With ``aspect``, every cross-section is first scaled by ``mu`` about the
        coordinate axis so that the aspect ratio is ``aspect``. Then the whole map, the frame's axis with it, is
        scaled by ``scale = (volume / V)^(1/3)`` so that the volume is ``volume``. Both factors are exact."""
        rings = self.change(beta)
        X = (self.start.raw_X1 + rings[0], self.start.raw_X2 + rings[1], self.start.raw_X3)
        V, V_axis, S = section_moments(seq, self.start.with_coordinates(*X))
        if self.aspect is not None:
            # the aspect ratio nfp V / (2 sqrt(pi) S^(3/2)) does not change under the scaling by `scale`
            mu = V_axis / (2.0 * np.sqrt(np.pi) * S ** 1.5 * self.aspect / self.start.nfp - V + V_axis)
            X = tuple(c[:1, :1] + mu * (c - c[:1, :1]) for c in X)
            V = mu ** 2 * V_axis + mu ** 3 * (V - V_axis)
        scale = (self.volume / V) ** (1.0 / 3.0)
        return self.start.with_coordinates(*X).scaled(scale)


def flux_seed(seq):
    """The seed ``s`` of :func:`vacuum_two_form`: the 2-form ``dr ^ dtheta`` interpolated into the Dirichlet
    2-form space of the odd view ``seq.odd`` (the field is odd). It carries the toroidal flux, is closed and
    does not depend on the geometry, so it can be computed once and reused for every shape."""
    flux = jnp.asarray((0.0, 0.0, 1.0))
    return seq.odd.interpolate(lambda x_hat: flux, 2)


def vacuum_two_form(seq, seed, tol=None):
    """``(h, info)``: the vacuum field, the harmonic 2-form ``h = seed - G_1 a`` with ``S_1 a = G_1^T M_2
    seed`` (see the module docstring), on the odd view of ``seq`` (a
    :meth:`~mrx.derham_sequence.DeRhamSequence.with_geometry` copy). ``h`` is differentiable in the geometry and
    has the toroidal flux of ``seed``, without normalisation. ``info`` is the signed iteration count of the
    forward solve. ``tol`` replaces ``seq.tol`` in the forward and the adjoint solve."""
    odd = seq.odd
    b = odd.D[1].T @ seed

    def matvec(x):
        return odd.S[1] @ x

    def solve(_, r):
        # both right-hand sides are G_1^T of something: the forward one by construction, the adjoint one
        # because a enters h only through G_1 a
        return odd.S[1].solve(r, tol=tol, return_info=True)

    a, info = jax.lax.custom_linear_solve(matvec, b, solve, symmetric=True, has_aux=True)
    return seed - odd.G[1] @ a, info


def _radial_moments(seq, h):
    """``(A, C)``: the ``(theta, zeta)`` averages of the logical densities ``B^theta`` and ``B^zeta`` of ``h`` as
    radial splines, ``<B^theta>(r) = sum_i A_i D_i(r)`` with ``D_i`` the radial derivative splines."""
    _, (n_rt, n_tt, n_zt), (n_rz, n_tz, n_zz) = seq.basis_2.shape
    raw = seq.odd.E(2).T @ h
    n0, n1 = seq.basis_2.n1, seq.basis_2.n2
    lt, lz = seq.basis_0.Lambda[1], seq.basis_0.Lambda[2]

    def integrals(b):
        # a periodic spline integrates to its support over p + 1
        return (b.T[b.p + 1:b.p + 1 + b.n] - b.T[:b.n]) / (b.p + 1)
    A = jnp.einsum("ijk,j->i", raw[n0:n0 + n1].reshape(n_rt, n_tt, n_zt), integrals(lt))
    C = jnp.einsum("ijk,k->i", raw[n0 + n1:].reshape(n_rz, n_tz, n_zz), integrals(lz))
    return A, C


def mean_iota(seq, h):
    """The mean ``int_0^1 iota ds`` over the normalised toroidal flux ``s`` of the flux-ratio rotational transform
    ``iota = nfp <B^theta> / <B^zeta>`` (averages over the logical surfaces). It equals ``nfp`` times the ratio of
    the poloidal to the toroidal flux per field period, the analogue of simsopt's ``mean_iota``."""
    A, C = _radial_moments(seq, h)
    return seq.nfp * jnp.sum(A) / jnp.sum(C)


def normal_field_fraction(seq, h):
    """``int (B . n)^2 / int |B|^2`` over the volume, with ``n`` the unit normal of the logical surfaces. It is
    zero when the logical surfaces are flux surfaces of ``h``."""
    b = seq.odd.evaluate_at_quadrature(h, 2)
    w, J = seq.quad.w, seq.jacobian_j
    # (B . n)^2 J = (B^r)^2 / (J g^rr) and |B|^2 J = B^T G B / J for the logical densities B, J = det DPhi
    normal = jnp.sum(w * b[:, 0] ** 2 / (J * seq.metric_inv_jkl[:, 0, 0]))
    return normal / jnp.sum(w * jnp.einsum("qi,qij,qj->q", b, seq.metric_jkl, b) / J)


def _tables(seq):
    """The 1-D spline tables at the quadrature points that :func:`quasisymmetry_residual` needs, and the
    quadrature grid shape."""
    types = seq.basis_0.types
    prim, der, dder = seq.basis_jk, seq.d_basis_jk, seq.dd_basis_jk
    dprim = tuple(_grad_1d(d, t) for d, t in zip(der, types))
    ddprim = tuple(_grad_1d(dd, t) for dd, t in zip(dder, types))
    return prim, der, dprim, dder, ddprim, (int(prim[0].shape[1]), seq.quad.shape[1], seq.quad.shape[2])


def _two_form_derivatives(seq, h, tables):
    """``(b, db)``: the logical components ``b`` of the 2-form ``h`` at the quadrature points, ``(n_q, 3)``, and
    their logical derivatives ``db[a, q, i] = d_a b^i``, ``(3, n_q, 3)``."""
    prim, der, dprim, dder, _, grid = tables
    shapes = [tuple(int(v) for v in sh) for sh in seq.basis_2.shape]
    raw = seq.odd.E(2).T @ h

    def info(a):
        # Component c uses the 0-form splines on axis c and the derivative splines on the others. d_a
        # differentiates the table of axis a.
        out = []
        for c in range(3):
            tabs = [prim[i] if i == c else der[i] for i in range(3)]
            if a is not None:
                tabs[a] = dprim[a] if a == c else dder[a]
            out.append((c, *tabs))
        return out
    b = evaluate_at_xq(raw, info(None), shapes, grid, 3)
    return b, jnp.stack([evaluate_at_xq(raw, info(a), shapes, grid, 3) for a in range(3)])


def _residual_terms(seq, h, Phi, tables):
    """``(B, parallel, cross, B_cov, J)`` at the quadrature points: ``|B|``, ``B . grad|B|``,
    ``(B x grad r) . grad|B|``, the covariant components of ``B`` and ``det DPhi``."""
    D, dD, _ = map_jacobian(seq, Phi, second=True)
    G = jnp.einsum("qki,qkj->qij", D, D)
    dG = jnp.einsum("qkmi,qmj->qkij", dD, D) + jnp.einsum("qmi,qkmj->qkij", D, dD)
    J = jnp.linalg.det(D)
    dJ = J[:, None] * jnp.einsum("qij,qkji->qk", jax.vmap(jnp.linalg.inv)(D), dD)

    b, db = _two_form_derivatives(seq, h, tables)
    Gb = jnp.einsum("qij,qj->qi", G, b)
    B2 = jnp.sum(b * Gb, -1) / J ** 2
    dB2 = ((2.0 * jnp.einsum("qi,kqi->qk", Gb, db) + jnp.einsum("qi,qkij,qj->qk", b, dG, b)) / J[:, None] ** 2
           - 2.0 * B2[:, None] * dJ / J[:, None])
    B = jnp.sqrt(B2)
    dB = dB2 / (2.0 * B[:, None])
    B_cov = Gb / J[:, None]
    return (B, jnp.sum(b * dB, -1) / J, (dB[:, 1] * B_cov[:, 2] - dB[:, 2] * B_cov[:, 1]) / J, B_cov, J)


def quasisymmetry_residual(seq, h, Phi, r_min):
    """``(F, F_parallel)``: the volume average over ``r >= r_min`` of the squared two-term quasi-axisymmetry
    residual of the vacuum field ``h`` on the frame map ``Phi`` (:func:`frame_geometry`),

        f = (G B . grad|B| - iota (B x grad Psi) . grad|B| / (2 pi)) / |B|^3.

    Here ``Psi`` is the toroidal flux, taken constant on the logical surfaces, ``iota = nfp <B^theta> / <B^zeta>`` the signed flux
    ratio (averages over the logical surfaces), and ``G`` the mean of the covariant ``B_zeta`` over the quadrature points times
    ``nfp / 2 pi``. The residual is exact where the logical surfaces are flux surfaces. ``F_parallel`` is the
    same average of the first term alone."""
    nfp = Phi.nfp
    B, parallel, cross, B_cov, J = _residual_terms(seq, h, Phi, _tables(seq))
    A, C = _radial_moments(seq, h)
    D = basis_table(seq.basis_0.dLambda[0], seq.quad.x_x)
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
