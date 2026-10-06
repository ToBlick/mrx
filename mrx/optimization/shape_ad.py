"""Shape optimisation of a vacuum field by automatic differentiation.

This module makes the vacuum magnetic field of a stellarator a differentiable function of the shape of its
domain, so that ``jax.grad`` of an objective such as a quasi-symmetry residual gives the derivative with
respect to the shape. The shape is the map Phi from logical to physical coordinates, written in cylindrical
form ``Phi = (R cos phi, R sin phi, Z)``, ``phi = 2 pi zeta / nfp``, with ``R`` and ``Z`` scalar
splines. Their coefficients are the variables.

The vacuum field is the harmonic 2-form ``h = s - G_1 a`` of the complex with Dirichlet boundary conditions.
Here ``s`` is a fixed seed that carries the toroidal flux, ``G_1`` the discrete curl, ``M_2`` the 2-form mass
matrix, and ``a`` solves ``S_1 a = G_1^T M_2 s`` with ``S_1 = G_1^T M_2 G_1``. A gradient costs one forward
and one adjoint solve of this system.

A typical use: build a :class:`BoundaryShape` about a starting map, turn its variables into a geometry
(:meth:`BoundaryShape.geometry`), install it with :func:`with_geometry`, compute the field with
:func:`flux_seed` and :func:`vacuum_two_form`, and evaluate an objective: :func:`flux_ratio_iota`,
:func:`mean_iota`, :func:`flux_surface_radius`, :func:`normal_field_fraction` or
:func:`quasisymmetry_residual`. The objectives treat the logical surfaces as flux surfaces, and
:func:`normal_field_fraction` measures how far that is from true. The preconditioners of the solves are
those of the sequence's own geometry and are not rebuilt as the shape changes.
"""
import copy
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import solvax

from mrx.differential_forms import DifferentialForm, inv33
from mrx.equilibria import CylindricalMap
from mrx.geometry import SequenceGeometry, _tp_evaluate, grad_1d
from mrx.mappings import stellarator_symmetric_scalar
from mrx.mass import attach_weights
from mrx.precision import RESIDUAL_DTYPE, cast_arrays
from mrx.quadrature import evaluate_at_xq
from mrx.spline_bases import basis_table


def with_geometry(seq, geometry):
    """A copy of ``seq`` with ``geometry`` installed, differentiable in ``geometry``. Unlike
    :meth:`~mrx.derham_sequence.DeRhamSequence.set_geometry` it can be used inside ``jax.grad`` and ``jit``.
    ``seq`` itself is not changed, and the copy keeps the preconditioners of ``seq``'s geometry."""
    out = copy.copy(seq)
    out.geometry = attach_weights(seq, geometry)
    if seq.residual is not None:          # the float64 copy of the mixed configuration gets the geometry too
        out.residual = copy.copy(seq.residual)
        out.residual.geometry = cast_arrays(out.geometry, RESIDUAL_DTYPE)
    return out


def _cylindrical_metric(R, dR, dZ, nfp):
    """``(G, cross)``: the metric of the cylindrical map Phi and ``cross = R_theta Z_r - R_r Z_theta``, so that
    ``det DPhi = (2 pi / nfp) R cross``."""
    G = dR[:, :, None] * dR[:, None, :] + dZ[:, :, None] * dZ[:, None, :]
    G = G.at[:, 2, 2].add((2.0 * np.pi / nfp * R) ** 2)
    return G, dR[:, 1] * dZ[:, 0] - dR[:, 0] * dZ[:, 1]


def cylindrical_geometry(seq, raw_R, raw_Z, nfp):
    """The :class:`~mrx.geometry.SequenceGeometry` of the map ``Phi = (R cos phi, R sin phi, Z)``,
    ``phi = 2 pi zeta / nfp``, differentiable in the coefficients.

    ``R`` and ``Z`` are scalar splines in ``seq.basis_0`` with coefficient arrays ``raw_R`` and ``raw_Z`` of
    shape ``(n_r, n_theta, n_zeta)``, the same form as the map of :func:`mrx.equilibria.build_map`. The result
    holds the map (:class:`mrx.equilibria.CylindricalMap`), the metric and ``det DPhi`` at the quadrature
    points."""
    R, dR, dZ = _cylindrical_fields(seq, raw_R, raw_Z)
    G, cross = _cylindrical_metric(R, dR, dZ, nfp)
    J = (2.0 * np.pi / nfp) * R * cross
    return SequenceGeometry(CylindricalMap(raw_R, raw_Z, seq.basis_0.bases[0], nfp), G, jax.vmap(inv33)(G), J)


def _cylindrical_fields(seq, raw_R, raw_Z):
    """``R`` and the logical gradients of ``R`` and ``Z`` at the quadrature points, shapes ``(n_q,)`` and
    ``(n_q, 3)``."""
    (Br, Bt, Bz), _, (Dr, Dt, Dz), *_ = _tables(seq)
    C = jnp.stack([raw_R, raw_Z])
    R = _tp_evaluate(C[:1], Br, Bt, Bz)[0].reshape(-1)
    d = jnp.stack([_tp_evaluate(C, Dr, Bt, Bz), _tp_evaluate(C, Br, Dt, Bz),
                   _tp_evaluate(C, Br, Bt, Dz)], axis=-1)            # (2, nqr, nqt, nqz, 3)
    return R, d[0].reshape(-1, 3), d[1].reshape(-1, 3)


def section_moments(seq, raw_R, raw_Z, nfp):
    """``(V, V_axis, S)`` for the map of :func:`cylindrical_geometry`, over one field period.

    ``V`` is the volume, ``S`` the mean cross-section area, and ``V_axis`` is the volume integral of ``R_axis /
    R``, where ``R_axis`` is the ``R`` of the coordinate axis (the innermost ring of coefficients). They are
    defined so that scaling every cross-section by ``mu`` about the axis takes ``S`` to ``mu^2 S`` and ``V`` to
    ``mu^2 V_axis + mu^3 (V - V_axis)`` exactly."""
    R, dR, dZ = _cylindrical_fields(seq, raw_R, raw_Z)
    a = 2.0 * np.pi / nfp
    J_over_R = a * (dR[:, 1] * dZ[:, 0] - dR[:, 0] * dZ[:, 1])
    axis = jnp.broadcast_to(raw_R[0, 0] @ seq.basis_z_jk, seq.quad.shape).reshape(-1)
    w = seq.quad.w
    return jnp.sum(w * R * J_over_R), jnp.sum(w * axis * J_over_R), jnp.sum(w * J_over_R) / a


class BoundaryShape(eqx.Module):
    """A parametrisation of the map's ``R`` and ``Z`` coefficients by the optimisation variables ``beta``.

    Build it with :meth:`from_coefficients` about a starting map. Every shape it produces is stellarator
    symmetric (``R`` even and ``Z`` odd under ``(theta, zeta) -> (-theta, -zeta)``) and has the volume of the
    starting map, and also its aspect ratio if ``aspect`` is given (:meth:`map_coefficients`).

    With ``free="boundary"``, ``beta`` has shape ``(2, n_theta, n_zeta)`` and is the change of the ``R`` and
    ``Z`` coefficients on the outermost ring. The change is carried into the interior with a weight per ring
    and poloidal mode ``m``, chosen by ``extension``:

    * ``"ramp"``: weight ``((i - 1) / (n_r - 2))^2`` on ring ``i``, and zero on the two innermost rings, so the
      magnetic axis does not move.
    * ``"harmonic"``: weight ``r_i^|m|``, with ``r_i`` the radius of ring ``i``, except that ring 1 only takes
      ``|m| <= 1``. The ``m = 0`` part then moves the axis as well.

    With ``free="all"``, ``beta`` has shape ``(2, n_r, n_theta, n_zeta)``. Its outermost ring is extended
    inwards as above, and the inner rings add their own independent change (:meth:`interior_modes`).
    """

    raw_R: jnp.ndarray
    raw_Z: jnp.ndarray
    extension: jnp.ndarray
    volume: jnp.ndarray
    nfp: int = eqx.field(static=True)
    basis_0: DifferentialForm = eqx.field(static=True)
    aspect: Optional[float] = eqx.field(static=True, default=None)
    interior: Optional[jnp.ndarray] = None

    @classmethod
    def from_coefficients(cls, seq, raw_R, raw_Z, nfp, volume=None, aspect=None, extension="ramp",
                          free="boundary"):
        """The parametrisation about the map with coefficients ``raw_R``, ``raw_Z`` (``nfp`` as in
        :func:`cylindrical_geometry`). ``volume`` is the volume of one field period to hold, by default that of
        the starting map. ``aspect`` is the aspect ratio to hold, by default none. With these defaults
        ``beta = 0`` gives back the starting map."""
        if volume is None:
            volume = section_moments(seq, raw_R, raw_Z, nfp)[0]
        weights = cls.extension_weights(seq.basis_0, extension)
        interior = jnp.asarray(cls.interior_modes(seq.basis_0), dtype=raw_R.dtype) if free == "all" else None
        return cls(raw_R, raw_Z, jnp.asarray(weights, dtype=raw_R.dtype), jnp.asarray(volume), int(nfp),
                   seq.basis_0, None if aspect is None else float(aspect), interior)

    @staticmethod
    def extension_weights(basis_0, extension):
        """The ``(n_r, n_theta)`` weights with which the boundary change enters ring ``i``, per poloidal mode
        ``m`` in FFT order (see the class docstring)."""
        n_r, n_t = basis_0.Lambda[0].n, basis_0.Lambda[1].n
        m = np.abs(np.fft.fftfreq(n_t, 1.0 / n_t))
        if extension == "ramp":
            return np.broadcast_to(((np.maximum(np.arange(n_r) - 1, 0) / (n_r - 2)) ** 2)[:, None], (n_r, n_t))
        r = np.asarray(basis_0.Lambda[0].greville_points(), dtype=np.float64)
        weights = r[:, None] ** m[None, :]
        weights[1, m >= 2] = 0.0
        return weights

    @staticmethod
    def interior_modes(basis_0):
        """A ``(n_r, n_theta)`` mask of the poloidal modes each ring may change on its own with ``free="all"``:
        ``m = 0`` on ring 0, ``|m| = 1`` on ring 1, every mode on the rings in between and none on the boundary
        ring. Ring 1 takes its ``m = 0`` change from ring 0."""
        n_r, n_t = basis_0.Lambda[0].n, basis_0.Lambda[1].n
        m = np.abs(np.fft.fftfreq(n_t, 1.0 / n_t))
        modes = np.ones((n_r, n_t))
        modes[0], modes[1], modes[-1] = m == 0, m == 1, 0.0
        return modes

    def perturbation(self, beta):
        """``(b_R, b_Z)``: the stellarator-symmetric part of ``beta``, ``R`` even and ``Z`` odd under
        ``(theta, zeta) -> (-theta, -zeta)``."""
        if self.interior is not None:
            return (stellarator_symmetric_scalar(beta[0], self.basis_0, even=True),
                    stellarator_symmetric_scalar(beta[1], self.basis_0, even=False))
        return (stellarator_symmetric_scalar(beta[0][None], self.basis_0, even=True)[0],
                stellarator_symmetric_scalar(beta[1][None], self.basis_0, even=False)[0])

    def change(self, beta):
        """The change ``(2, n_r, n_theta, n_zeta)`` of the ``R`` and ``Z`` coefficients at ``beta``, before the
        volume and aspect-ratio scalings."""
        b = jnp.stack(self.perturbation(beta))
        if self.interior is None:
            spectrum = jnp.fft.fft(b, axis=1)                                            # (2, n_t, n_z)
            return jnp.fft.ifft(self.extension[None, :, :, None] * spectrum[:, None], axis=2).real
        boundary = jnp.fft.fft(b[:, -1:], axis=2)                                         # (2, 1, n_t, n_z)
        return jnp.fft.ifft(self.extension[None, :, :, None] * boundary, axis=2).real + self.interior_change(beta)

    def interior_change(self, beta):
        """The independent change of the inner rings at ``beta`` with ``free="all"``, shape ``(2, n_r, n_theta,
        n_zeta)``. It is zero on the boundary ring."""
        spectrum = jnp.fft.fft(jnp.stack(self.perturbation(beta)), axis=2)              # (2, n_r, n_t, n_z)
        own = (self.interior[None, :, :, None] * spectrum).at[:, 1, 0].set(spectrum[:, 0, 0])
        return jnp.fft.ifft(own, axis=2).real

    def map_coefficients(self, seq, beta):
        """``(R, Z, mu, scale)``: the coefficients of the map at ``beta`` and the two scale factors applied to
        them. With ``aspect``, every cross-section is first scaled by ``mu`` about the coordinate axis so that
        the aspect ratio is ``aspect`` (``mu = 1`` without it). Then the whole map is scaled by
        ``scale = (volume / V)^(1/3)`` so that the volume is ``volume``. Both factors are exact."""
        rings = self.change(beta)
        R, Z = self.raw_R + rings[0], self.raw_Z + rings[1]
        V, V_axis, S = section_moments(seq, R, Z, self.nfp)
        mu = jnp.ones_like(V)
        if self.aspect is not None:
            # the aspect ratio nfp V / (2 sqrt(pi) S^(3/2)) does not change under the scaling by `scale`
            mu = V_axis / (2.0 * np.sqrt(np.pi) * S ** 1.5 * self.aspect / self.nfp - V + V_axis)
            R = R[:1, :1] + mu * (R - R[:1, :1])
            Z = Z[:1, :1] + mu * (Z - Z[:1, :1])
            V = mu ** 2 * V_axis + mu ** 3 * (V - V_axis)
        scale = (self.volume / V) ** (1.0 / 3.0)
        return scale * R, scale * Z, mu, scale

    def geometry(self, seq, beta):
        """The differentiable geometry (:func:`cylindrical_geometry`) of the map at ``beta``."""
        R, Z, _, _ = self.map_coefficients(seq, beta)
        return cylindrical_geometry(seq, R, Z, self.nfp)


def flux_seed(seq):
    """The seed ``s`` of :func:`vacuum_two_form`: the 2-form ``dr ^ dtheta`` interpolated into the Dirichlet
    2-form space. It carries the toroidal flux, is closed and does not depend on the geometry, so it can be
    computed once and reused for every shape."""
    flux = jnp.asarray((0.0, 0.0, 1.0), dtype=seq.dtype)
    return seq.interpolate(lambda x_hat: flux, 2, frame='logical')


def vacuum_two_form(seq, seed):
    """``(h, info)``: the vacuum field, the harmonic 2-form ``h = seed - G_1 a`` with ``S_1 a = G_1^T M_2
    seed`` (see the module docstring), on ``seq`` (a :func:`with_geometry` copy). ``h`` is differentiable in
    the geometry and has the toroidal flux of ``seed``, without normalisation. ``info`` is the signed
    iteration count of the forward solve.

    The right-hand side lies in the range of ``G_1^T``, so the exact part of a Hodge-Laplacian solve is zero
    and one preconditioned conjugate-gradient solve on ``S_1`` (:func:`solvax.pcg`, with the preconditioner of
    ``L_1``) gives ``a``. The forward and the adjoint solve are both this solve, so the derivative is that of
    the converged equation. The residual is measured in the Euclidean norm, two digits below ``seq.tol``."""
    b = seq.D[1].T @ seed
    parity = seq.free_projector(1)

    def matvec(x):
        return seq.S[1] @ x

    def solve(_, r):
        if parity is not None:
            # h is odd. The adjoint right-hand side has an even part, which pairs to zero with every odd
            # tangent. Removing it keeps both solves in the odd space.
            r = parity.dual(r, -1.0)
        sol = solvax.pcg(matvec, r, precond=seq.L[1].precondition, rtol=1e-2 * seq.tol, max_steps=seq.maxiter)
        return sol.x, jnp.where(sol.converged, sol.iterations, -sol.iterations)

    a, info = jax.lax.custom_linear_solve(matvec, b, solve, symmetric=True, has_aux=True)
    return seed - seq.G[1] @ a, info


def _radial_moments(seq, h):
    """``(A, C)``: the ``(theta, zeta)`` averages of the logical densities ``B^theta`` and ``B^zeta`` of ``h`` as
    radial splines, ``<B^theta>(r) = sum_i A_i D_i(r)`` with ``D_i`` the radial derivative splines."""
    _, (n_rt, n_tt, n_zt), (n_rz, n_tz, n_zz) = seq.basis_2.shape
    raw = seq.E(2).T @ h
    n0, n1 = seq.basis_2.n1, seq.basis_2.n2
    lt, lz = seq.basis_0.Lambda[1], seq.basis_0.Lambda[2]

    def integrals(b):
        # a periodic spline integrates to its support over p + 1
        return (b.T[b.p + 1:b.p + 1 + b.n] - b.T[:b.n]) / (b.p + 1)
    A = jnp.einsum("ijk,j->i", raw[n0:n0 + n1].reshape(n_rt, n_tt, n_zt), integrals(lt))
    C = jnp.einsum("ijk,k->i", raw[n0 + n1:].reshape(n_rz, n_tz, n_zz), integrals(lz))
    return A, C


def _flux_tables(seq, C, r):
    """``(s, ds/dr)`` at the logical radii ``r``, with ``s = Psi(r) / Psi_edge`` the normalised toroidal flux
    computed from the moments ``C`` of :func:`_radial_moments`."""
    # the antiderivative of sum_i C_i D_i is sum_l B_l(r) sum_{i < l} C_i (B'_l = D_{l-1} - D_l)
    flux = jnp.concatenate([jnp.zeros(1, C.dtype), jnp.cumsum(C)])
    return ((flux @ basis_table(seq.basis_0.Lambda[0], r)) / flux[-1],
            (C @ basis_table(seq.basis_0.dLambda[0], r)) / flux[-1])


def flux_ratio_iota(seq, h, r):
    """``(iota, s)`` at the logical radii ``r`` for the field ``h``. ``iota = nfp <B^theta> / <B^zeta>`` is the
    ratio of the averages over the logical surface, and ``s = Psi / Psi_edge`` the normalised toroidal flux
    inside it. Both are exact functions of the coefficients of ``h``. ``iota`` is the rotational transform
    where the logical surfaces are flux surfaces, and its sign follows the logical orientation."""
    A, C = _radial_moments(seq, h)
    D = basis_table(seq.basis_0.dLambda[0], r)
    return seq.nfp * (A @ D) / (C @ D), _flux_tables(seq, C, r)[0]


def mean_iota(seq, h):
    """The mean ``int_0^1 iota ds`` of :func:`flux_ratio_iota` over the normalised toroidal flux ``s``. It
    equals ``nfp`` times the ratio of the poloidal to the toroidal flux per field period, the analogue of
    simsopt's ``mean_iota``."""
    A, C = _radial_moments(seq, h)
    return seq.nfp * jnp.sum(A) / jnp.sum(C)


def flux_surface_radius(seq, h, s, newton_steps=8):
    """The logical radii at which the normalised toroidal flux of :func:`flux_ratio_iota` takes the values
    ``s``, found by ``newton_steps`` Newton steps from ``r = sqrt(s)``. The derivative with respect to the
    shape is the exact derivative of the root."""
    # Newton runs with the gradient stopped. One final differentiable step gives the implicit derivative.
    _, C = _radial_moments(seq, h)
    C0 = jax.lax.stop_gradient(C)
    r = jnp.sqrt(s)
    for _ in range(newton_steps):
        val, slope = _flux_tables(seq, C0, r)
        r = r - (val - s) / slope
    val, slope = _flux_tables(seq, C, r)
    return r - (val - s) / slope


def normal_field_fraction(seq, h):
    """``int (B . n)^2 / int |B|^2`` over the volume, with ``n`` the unit normal of the logical surfaces. It is
    zero when the logical surfaces are flux surfaces of ``h``."""
    b = seq.evaluate_at_quadrature(h, 2)
    w, J = seq.quad.w, seq.jacobian_j
    # (B . n)^2 J = (B^r)^2 / (J g^rr) and |B|^2 J = B^T G B / J for the logical densities B, J = det DPhi
    normal = jnp.sum(w * b[:, 0] ** 2 / (J * seq.metric_inv_jkl[:, 0, 0]))
    return normal / jnp.sum(w * jnp.einsum("qi,qij,qj->q", b, seq.metric_jkl, b) / J)


def _tables(seq):
    """The 1-D spline tables at the quadrature points that :func:`quasisymmetry_residual` needs, and the
    quadrature grid shape."""
    types = seq.basis_0.types
    prim = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    der = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    dder = seq.dd_basis_jk
    dprim = tuple(grad_1d(d, t) for d, t in zip(der, types))
    ddprim = tuple(grad_1d(dd, t) for dd, t in zip(dder, types))
    return prim, der, dprim, dder, ddprim, (int(prim[0].shape[1]), seq.quad.shape[1], seq.quad.shape[2])


def _two_form_derivatives(seq, h, tables):
    """``(b, db)``: the logical components ``b`` of the 2-form ``h`` at the quadrature points, ``(n_q, 3)``, and
    their logical derivatives ``db[a, q, i] = d_a b^i``, ``(3, n_q, 3)``."""
    prim, der, dprim, dder, _, grid = tables
    shapes = [tuple(int(v) for v in sh) for sh in seq.basis_2.shape]
    raw = seq.E(2).T @ h

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


def _cylindrical_derivatives(raw_R, raw_Z, tables):
    """``(v, d1, d2)``: ``(R, Z)`` at the quadrature points, ``(2, n_q)``, their first logical derivatives
    ``d1[q, c, i]``, ``(n_q, 2, 3)``, and their second ones ``d2[q, c, i, j]``, ``(n_q, 2, 3, 3)``."""
    prim, _, dprim, _, ddprim, _ = tables
    tabs = (prim, dprim, ddprim)
    C = jnp.stack([raw_R, raw_Z])

    def ev(orders):
        return _tp_evaluate(C, *(tabs[o][a] for a, o in enumerate(orders))).reshape(2, -1)
    unit = np.eye(3, dtype=int)
    d1 = jnp.stack([ev(unit[i]) for i in range(3)], -1)
    d2 = jnp.stack([jnp.stack([ev(unit[i] + unit[j]) for j in range(3)], -1) for i in range(3)], -2)
    return ev((0, 0, 0)), jnp.moveaxis(d1, 1, 0), jnp.moveaxis(d2, 1, 0)


def _residual_terms(seq, h, raw_R, raw_Z, nfp, tables):
    """``(B, parallel, cross, B_cov, J)`` at the quadrature points: ``|B|``, ``B . grad|B|``,
    ``(B x grad r) . grad|B|``, the covariant components of ``B`` and ``det DPhi``."""
    a = 2.0 * np.pi / nfp
    (R, _), d1, d2 = _cylindrical_derivatives(raw_R, raw_Z, tables)
    dR, dZ, ddR, ddZ = d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1]
    G, cross_rz = _cylindrical_metric(R, dR, dZ, nfp)
    dG = (jnp.einsum("qik,qj->qkij", ddR, dR) + jnp.einsum("qi,qjk->qkij", dR, ddR)
          + jnp.einsum("qik,qj->qkij", ddZ, dZ) + jnp.einsum("qi,qjk->qkij", dZ, ddZ))
    dG = dG.at[:, :, 2, 2].add(2.0 * a ** 2 * R[:, None] * dR)
    J = a * R * cross_rz
    dJ = a * (dR * cross_rz[:, None] + R[:, None] * (
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


def quasisymmetry_residual(seq, h, raw_R, raw_Z, nfp, r_min):
    """``(F, F_parallel)``: the volume average over ``r >= r_min`` of the squared two-term quasi-axisymmetry
    residual of the vacuum field ``h`` on the map with coefficients ``raw_R``, ``raw_Z``
    (:func:`cylindrical_geometry`),

        f = (G B . grad|B| - iota (B x grad Psi) . grad|B| / (2 pi)) / |B|^3.

    Here ``Psi`` is the toroidal flux, taken constant on the logical surfaces, ``iota`` the signed flux ratio
    of :func:`flux_ratio_iota`, and ``G`` the mean of the covariant ``B_zeta`` over the quadrature points times
    ``nfp / 2 pi``. The residual is exact where the logical surfaces are flux surfaces. ``F_parallel`` is the
    same average of the first term alone."""
    B, parallel, cross, B_cov, J = _residual_terms(seq, h, raw_R, raw_Z, nfp, _tables(seq))
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
