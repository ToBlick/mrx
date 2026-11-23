"""Twist moves as Newton unknowns: poloidal flux slips between the flux surfaces, which set the net toroidal current
inside every surface, and the outer loop that sets that current from a target rotational transform.

The move is ``dB = curl(f grad zeta) = grad f x grad zeta`` with ``f = F(s)`` a function of the flux label ``s``
(:mod:`mrx.flux_label`, the normalised toroidal flux inside a level set of the label's temperature ``T``) and
``zeta`` the logical toroidal angle, one field period per unit. The sections ``zeta = const`` need not be planes.
``dB`` is divergence-free, tangent to the level sets of ``T`` and has no ``zeta`` component, so every flux surface
stays a flux surface and keeps its toroidal flux. It adds ``-F(s)`` to the poloidal flux per unit ``zeta`` inside the
surface ``s``, so the rotational transform changes by ``-nfp dF/dPhi``. Where ``T`` is flat (an island chain or a
chaotic band) ``f`` is flat and nothing is added. The net toroidal current inside each surface changes with
``dF/ds``.

As unknowns of every Newton step (:class:`Slips`) the moves set the current: the relaxation minimises

    W = int B^2/2 dV - sum_j d_j g_j

over the ideal moves and the twists ``F = sum_j d_j F_j``, ``j = 0 .. K - 2``, with ``F_j = sum_{k > j} phi_k`` a
smoothed step across the spline ``j`` of the profile space (``phi_k`` its ``K`` cubic splines in ``s``,
:func:`mrx.relaxation.shells.profile_basis`, the same space as the shells'). The derivative of ``F_j`` in ``s`` is
``M_j``, the unit-integral derivative spline, so the energy's derivative along the twist ``j`` is ``-Z mu_j`` with
``mu_j = int M_j(s) mu0 I(s) ds`` the local mean of the enclosed current, ``Z`` the ``zeta`` extent of the integrals.
With ``g_j = -Z mu*_j``, the target's mean ``mu*_j``, a stationary point has the target's current in every moment.
``W`` is quadratic in the ``d_j`` with the Hessian ``(T_i, T_j)_M``, ``T_j`` the field change of ``F_j``, so it is
convex along the slips. A constant ``F`` is no move: ``F(0) = 0`` is the gauge. Every ``F_j`` is a spline in ``s``, so
next to the magnetic axis, where ``s`` goes as the squared distance, it is a smooth potential: the first step goes as
``s`` there, and the current next to the axis is a target like any other. (Steps of splines in ``sqrt(s)`` would be
cones at the axis.) ``F(1)`` is free, so the total current is a target. The current itself has no gauge, it vanishes
on the axis.

A target transform is not a control of ``W``. It sets the current target in an outer loop: :func:`iota_newton`
measures the transform's weak residual on the profile space (:func:`iota_residual`) and takes a Newton step on the
current target's moments, with the derivative of the equilibrium along the target from the implicit function theorem
and the slips the only part of a step that changes the transform at fixed toroidal flux (:func:`slip_transform`).
The step is taken before every chunk of Newton steps. The chunk then runs with the slips of that target
(:func:`build_slips`), as with any other.

- :func:`build_slips`, :func:`slip_changes` and :func:`current_moments` are the move as Newton
  unknowns, :func:`current_on_label` the moments of a field, :func:`current_target` those of a target and
  :func:`current_error` the current's error against them.
- :func:`iota_residual` is the weak residual of a target transform on the profile space, :func:`slip_transform`
  its change along the slips and :func:`iota_newton` the outer Newton step with them.
- :func:`iota_fit` is the transform as a spline in ``s``, the L2 projection of its weak moments, and
  :func:`enclosed_current` the current inside the label's level sets.
- :func:`load_iota_target` reads a transform from a wout or a VMEC input namelist with ``NCURR = 0``,
  :func:`load_current_target` a current from a namelist with ``NCURR = 1``.

The potential is ``F(1) dzeta + a``, ``a`` the L2 projection of ``(F(s) - F(1)) dzeta`` onto the 1-forms ``a dzeta``
with zero tangential trace (:func:`slip_changes`). ``dzeta`` is closed, so the field change is ``curl a``, that of ``F dzeta``,
with no flux through the wall. ``F(1) != 0`` changes the poloidal flux inside the wall and with it the net toroidal
current.
"""
from __future__ import annotations


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.flux_label import ANISOTROPY, enclosed_flux, flux_label
from mrx.operators import _mass, _mass_model
from mrx.precision import DTYPE, to_working
from mrx.relaxation.physics import current as weak_current
from mrx.relaxation.shells import LABEL_TOL, PROFILE_DEGREE, profile_basis, profile_cells
from mrx.solvers import solve_singular_cg
from mrx.spline_bases import DerivativeSpline, basis_table

#: The transform is at its target when every local mean of its miss, the weak residual of :func:`iota_residual`, is
#: below this fraction of the target's largest magnitude: ``relax.py --seed`` waits for it with ``--iota.target``.
AT_TARGET = 1e-2


def _axis_angle_gradient(seq, B):
    """``(d_r, d_t, d_z)`` at the quadrature points: the logical components of ``grad theta_a``, ``theta_a`` the
    angle about the magnetic axis of ``B`` (:func:`mrx.relaxation.mesh_motion.magnetic_axis`, a curve from a weak
    residual) over 2 pi, so that ``J B . grad theta_a`` is ``B^r d_r + B^theta d_t + B^zeta d_z`` in the 2-form's
    components. Summed over the region inside a level set of the label, ``J B . grad theta_a`` is the poloidal flux
    enclosed by that surface, for any position of the logical axis. The logical ``theta`` would not do: its cut ends
    on the logical axis, so it counts no flux inside a surface that does not enclose the logical axis and misses the
    flux between the two axes in every other. The axis and its ``zeta`` derivative come from its Fourier series at the
    quadrature planes."""
    from mrx.relaxation.mesh_motion import _series, magnetic_axis
    c, _ = magnetic_axis(seq, B)
    V, dV = _series(c.shape[2] - 1, np.asarray(seq.quad.x_z, dtype=np.float64))
    (xa, ya), (dxa, dya) = np.einsum("ckn,qkn->cq", c, V), np.einsum("ckn,qkn->cq", c, dV)
    xq = np.asarray(seq.quad.x)
    r, th = xq[:, 0], 2 * np.pi * xq[:, 1]
    k = np.broadcast_to(np.arange(xa.size), seq.quad.shape).ravel()
    u, v = r * np.cos(th) - xa[k], r * np.sin(th) - ya[k]
    rho2 = u * u + v * v
    # d theta_a = (u dv - v du) / (2 pi rho^2), dx = cos dr - 2 pi r sin d theta, dy = sin dr + 2 pi r cos d theta
    d_r = (u * np.sin(th) - v * np.cos(th)) / (2 * np.pi * rho2)
    d_t = r * (u * np.cos(th) + v * np.sin(th)) / rho2
    d_z = (-u * dya[k] + v * dxa[k]) / (2 * np.pi * rho2)
    return d_r, d_t, d_z


def iota_fit(seq, B, s_q, n_mu=None):
    """The transform per full turn of ``B`` as a spline in the label ``s`` (``s_q`` at the quadrature points), with the
    sign of the field's transform: the L2 projection in ``s`` onto the ``n_mu`` derivative splines ``M_j`` of the
    profile space (of :func:`~mrx.relaxation.shells.profile_cells` of ``seq`` unless given), the space the iota loop
    controls, from the weak moments ``int M_j iota ds``, the weak residual of the zero target (:func:`iota_residual`).
    The moments are moments of the field against functions of the label: no sorting of points, no differentiation of
    sampled fluxes, and one coefficient per spline, no free degree. A ``scipy.interpolate.BSpline`` of degree
    :data:`~mrx.relaxation.shells.PROFILE_DEGREE` ``- 1``, whose ``derivative()`` is ``d iota / ds``."""
    from scipy.interpolate import BSpline
    n_mu = profile_cells(seq) + PROFILE_DEGREE - 1 if n_mu is None else n_mu
    r = np.asarray(iota_residual(seq, B, s_q, np.zeros_like, n_mu)(B)[0], dtype=np.float64)
    d = _derivatives(n_mu)
    T = np.asarray(d.T, dtype=np.float64)
    # the Gram matrix int M_i M_j ds, exact by Gauss-Legendre with p + 1 points on every cell of the knots
    knots = np.unique(T)
    g, wg = np.polynomial.legendre.leggauss(d.p + 1)
    h = np.diff(knots)
    x = (knots[:-1, None] + 0.5 * h[:, None] * (g[None, :] + 1.0)).ravel()
    w = (0.5 * h[:, None] * wg[None, :]).ravel()
    M = np.asarray(basis_table(d, jnp.asarray(x)), dtype=np.float64)
    c = np.linalg.solve((M * w) @ M.T, r)
    scale = (d.p + 1) / (T[d.p + 1:d.p + 1 + d.n] - T[:d.n])      # M_j is the B-spline N_j times this
    return BSpline(T, c * scale, d.p)


def enclosed_current(seq, B, s_q, edges):
    """``mu0 I`` (T m) through the sections ``zeta = const`` inside the level sets ``s < edges``, the weak current's
    flux averaged over ``zeta``, positive along ``+zeta``."""
    J = seq.odd.evaluate_at_quadrature(weak_current(B, seq), 1)
    Jz = np.asarray(seq.jacobian_j * jnp.einsum("qj,qj->q", seq.metric_inv_jkl[:, 2, :], J))
    w, s = np.asarray(seq.quad.w), np.asarray(s_q)
    return np.array([np.sum((s < e) * w * Jz) for e in edges]) / np.sum(np.asarray(seq.quad.w_z))


def load_iota_target(path):
    """The target transform in a file, a function of ``s``: the transform of a wout (``.nc``, its
    ``iotaf``) or of a VMEC input namelist with ``NCURR = 0``. A namelist with ``NCURR = 1`` prescribes the current,
    a target of the relaxation (:func:`load_current_target`)."""
    if path.endswith(".nc"):
        from mrx.equilibria.vmec import read_wout
        f = read_wout(path)["profiles"]["iota"]                          # a function of r = sqrt(s)
        return lambda s: np.asarray(f(np.sqrt(np.asarray(s))))
    from mrx.equilibria.indata import read_indata
    nl = read_indata(path)
    if nl.ncurr == 1:
        raise ValueError(f"{path}: NCURR = 1 prescribes the current, a target of the relaxation (--current.target), "
                         "not a transform")
    return nl.iota


class Slips(eqx.Module):
    """The frozen part of the slip Newton step: the field changes ``T`` ``(K - 1, n_2)`` of the twists ``F_j`` on the
    label of one field (:func:`slip_changes`, 2-forms of ``seq.odd``, one row per slip, as the shells hold their
    velocities), the target's part ``g`` of the right-hand side and ``D_inv``, the inverse of the slip block
    ``(T_i, T_j)_M``. Arrays only, so a rebuild does not recompile the step."""
    T: jnp.ndarray
    g: jnp.ndarray
    D_inv: jnp.ndarray

    def maps(self, seq):
        """``(move, move_t)``: the map ``d -> T d`` to the field change of the twist ``F = sum_j d_j F_j`` and its
        transpose. MINRES calls the step's operator in float32, the products are taken in the dtype of ``T``."""
        return (lambda d: d.astype(self.T.dtype) @ self.T), (lambda y: self.T @ y.astype(self.T.dtype))


def _label_of(seq, B, T):
    """The label ``s`` at the quadrature points of the field ``B`` with the label temperature ``T``."""
    return enclosed_flux(seq, T, seq.odd.evaluate_at_quadrature(B, 2))


def _zeta_dofs(odd):
    """The 1-form DoFs of ``odd`` whose tensor-product coefficients are all in the ``zeta`` component, a 0/1 vector:
    the space of ``a_zeta dzeta``. The polar extraction keeps that component apart (its axis rows combine rings of the
    ``zeta`` component alone), so a DoF with any ``r`` or ``theta`` coefficient is found by one product of ``E`` with
    a vector that is generic on those components and zero on ``zeta``."""
    sizes = [int(np.prod(sh)) for sh in odd.basis_1.shape]
    generic = np.random.default_rng(0).uniform(1.0, 2.0, sizes[0] + sizes[1])
    probe = jnp.asarray(np.concatenate([generic, np.zeros(sizes[2])]), dtype=DTYPE)
    return (odd.E(1) @ probe == 0.0).astype(DTYPE)


@eqx.filter_jit
def slip_changes(seq, s_q, n):
    """``T`` ``(n - 1, n_2)``: the field changes of the ``n - 1`` twists ``F_j = sum_{k > j} phi_k(s)`` (:func:`_steps`)
    on the label ``s_q`` at the quadrature points, 2-forms of ``seq.odd``. The twist is ``curl(F_j dzeta)``. ``dzeta``
    is a closed 1-form of the space, so ``F_j(1) dzeta`` adds no field and the move is ``curl a_j``, ``a_j`` the L2
    projection of ``(F_j(s) - F_j(1)) dzeta`` onto the 1-forms ``a dzeta`` with zero tangential trace
    (:func:`_zeta_dofs`): one mass solve of that block with the load of its quadrature values, no point value off the
    quadrature and no interpolation of the label. ``curl(a dzeta)`` has no ``zeta`` component, so the twist keeps the
    toroidal flux of every flux surface exactly. The projection onto all the 1-forms would not: the mass couples the
    components, and its ``r`` and ``theta`` parts give the curl a ``zeta`` component (1e-3 of the largest on W7-X,
    2026-11-21). ``F_j(1) = 1`` for every ``j``. The field change has no flux through the wall."""
    odd = seq.odd
    values, indices = jax.vmap(profile_basis(n).evaluate_local)(s_q)
    S = _steps(n, values.dtype)
    zeros = jnp.zeros((s_q.size, 2), values.dtype)
    m = _zeta_dofs(odd)
    model = _mass_model(odd, 1)

    def one(column):
        f = jnp.sum(values * column[indices], axis=1) - 1.0
        load = odd.vector_load_values(jnp.concatenate([zeros, f[:, None]], axis=1), 1, 1)
        a, _ = solve_singular_cg(lambda x: _mass(odd, x * m, 1) * m, load * m, jnp.zeros((0, load.size)),
                                 precond_matvec=lambda r: model.apply(r * m) * m, tol=odd.tol, maxiter=odd.maxiter)
        return to_working(odd.G[1] @ to_working(a))
    return jax.lax.map(one, S.T)


def _steps(K, dtype):
    """``S`` ``(K, K - 1)``, ``S_kj = 1`` for ``k > j``, ``j = 0 .. K - 2``: the steps ``F_j = sum_{k > j} phi_k`` of
    the ``K`` splines of the profile space, whose derivatives in ``s`` are the unit-integral derivative splines
    ``M_j``. They vanish on the axis and go as ``s`` there."""
    return jnp.asarray(np.tril(np.ones((K, K - 1)), -1), dtype=dtype)


def current_moments(seq, B, move_t):
    """``mu_j = -(B, T_j)_M / Z``, the local means ``int M_j(s) mu0 I(s) ds`` of the net toroidal current of ``B``
    inside the label's surfaces (T m, along ``+zeta``), with ``move_t`` the transpose of the slips' field change
    (:meth:`Slips.maps`) and ``Z`` the ``zeta`` extent of the integrals."""
    return -move_t(seq.odd.M[2] @ B) / jnp.sum(seq.quad.w_z)


@eqx.filter_jit
def _slip_parts(seq, B, T_slip):
    M2T = jax.lax.map(lambda t: seq.odd.M[2] @ t, T_slip)
    D = T_slip @ M2T.T
    return -(T_slip @ (seq.odd.M[2] @ B)) / jnp.sum(seq.quad.w_z), 0.5 * (D + D.T)


def build_slips(seq, B, T, mu_star):
    """``(slips, mu)``: the :class:`Slips` at the field ``B`` with the label temperature ``T`` and the target moments
    ``mu_star`` (:func:`current_target`, one per slip, on the profile space of ``mu_star.size - 2`` cells), and the
    moments ``mu`` of ``B`` (:func:`current_moments`). The field changes cost
    one mass solve per slip (:func:`slip_changes`). The target's part is ``g = -Z mu_star``."""
    T_slip = slip_changes(seq, _label_of(seq, B, T), int(mu_star.size) + 1)
    mu, D = _slip_parts(seq, B, T_slip)
    g = -jnp.sum(seq.quad.w_z) * mu_star
    # inverted on the host in float64, as the shell block is
    return Slips(T_slip, g, jnp.asarray(np.linalg.inv(np.asarray(D, dtype=np.float64)))), mu


def iota_residual(seq, B0, s_q, iota, n_mu):
    """The weak residual of the target transform ``iota(s)`` (per full turn, its magnitude taken with the sign of the
    field's transform) on the flux label ``s_q`` (at the quadrature points) of the field ``B0``: the function
    ``B -> (r, Phi)`` of a 2-form ``B`` with

        r_j = int_0^1 (iota_B - iota*) M_j ds = int M_j(s) (nfp B . grad theta_a - iota*(s) B . grad zeta) dV / (Z Phi),

    ``M_j`` the ``n_mu`` unit-integral derivative splines of the profile space, ``theta_a`` the angle about the
    magnetic axis of ``B0`` (:func:`_axis_angle_gradient`), ``Phi`` the toroidal flux per unit ``zeta`` inside the label's
    outermost level set and ``Z`` the ``zeta`` extent of the integrals.

    ``r_j`` is the local mean of the transform's miss over the support of ``M_j``. The ``M_j`` span the changes of
    the transform that the slips make (a twist ``F`` changes it by ``-nfp F'(s) / Phi``, :func:`slip_transform`), so
    ``r = 0`` is the Galerkin condition: the target's transform projected in L2 in ``s`` onto what the slips reach.
    By the coarea formula the volume integral is the integral of ``M_j`` against ``nfp dpsi - iota* dPhi`` over the
    level sets of the label: it needs neither the points sorted by ``s`` nor a derivative of the sampled flux nor a
    division by ``s``. ``B`` enters linearly through the fluxes, ``1 / Phi`` aside, and the label and the axis are
    those of ``B0``, held, so a derivative in ``B`` is the change of the fluxes through the label's level sets. In an
    island band those level sets cross the islands: the residual there is that of the flux through them."""
    s = jnp.asarray(s_q)
    d_r, d_t, d_z = (jnp.asarray(c) for c in _axis_angle_gradient(seq, B0))
    M = jnp.asarray(basis_table(_derivatives(n_mu), s))
    i_star = jnp.asarray(np.abs(np.asarray(iota(np.asarray(s_q)), dtype=float)))
    w = jnp.asarray(np.asarray(seq.quad.w) / np.sum(np.asarray(seq.quad.w_z)))

    def residual(B):
        Bq = seq.odd.evaluate_at_quadrature(B, 2)
        dpsi = w * (Bq[:, 0] * d_r + Bq[:, 1] * d_t + Bq[:, 2] * d_z)
        dphi = w * Bq[:, 2]
        phi = jnp.sum(dphi)
        sign = jax.lax.stop_gradient(jnp.sign(jnp.sum(dpsi) / phi))
        return M @ (seq.nfp * dpsi - sign * i_star * dphi) / phi, phi
    return residual


def slip_transform(residual, B, change, n):
    """``R`` ``(n, n)``: the change of the weak residual along the slips with the rest of the field frozen, the
    derivatives of ``residual`` (the function of :func:`iota_residual`, label held) at the field ``B`` along the field
    changes ``change(e_j)`` of the ``n`` slips. An ideal move and a move of the mesh at fixed DoFs keep the poloidal
    flux of every flux surface as a function of its toroidal flux, so in an equilibrium's response the slips alone
    change the residual.

    An exact twist ``F = sum_k d_k F_k`` changes the transform by ``-nfp F'(s) / Phi``, so ``R`` would be
    ``-nfp / Phi int M_j M_k ds``. The discrete twist is the projection of ``F`` on the mesh, which resolves the
    innermost steps poorly: on W7-X 25 kA (33, 64, 36) the derivative of the first two slips of the histopolated twist
    was 48 % and 51 % off the exact one, of the others 1.3 to 3 %. So ``R`` is taken from the discrete twists, ``n`` derivatives in ``B``."""
    eye = jnp.eye(n, dtype=B.dtype)
    return np.stack([np.asarray(jax.jvp(residual, (B,), (change(eye[j]),))[1][0], dtype=np.float64)
                     for j in range(n)], 1)


@eqx.filter_jit
def _slip_responses(ts, B, slips):
    """``(d, C, info)``: the slips ``d`` of the Gauss-Newton direction of the stepper ``ts`` (its harmonic flows,
    shells and Newton settings) with the ``slips`` at the field ``B``, and ``C`` ``(L, L)``, the slips of its
    derivatives along the target's moments, with the signed MINRES counts. The target enters only the slip rows'
    right-hand side ``g - T_t M B``, ``g = -Z mu*``, so the column ``j`` is the solution for the right-hand side
    ``-Z e_j`` in the slip rows and zero elsewhere."""
    from mrx.relaxation.newton import target_direction
    seq = ts.seq
    _, load = ts._lorentz(B, None)
    a0 = jnp.zeros(seq.even.n(1), B.dtype)

    def solve(load_, shells, sl):
        out = target_direction(seq, B, load_, a0, ts.harmonic_flows, shells, sl, ts.newton_penalty, ts.newton_tol,
                               ts.newton_maxiter, ts.newton_mass_solve_tol)
        return out[5], out[2]
    d, info = solve(load, ts.shells, slips)
    shells0 = None if ts.shells is None else eqx.tree_at(lambda x: x.g_p, ts.shells, jnp.zeros_like(ts.shells.g_p))
    TtMB = slips.maps(seq)[1](seq.odd.M[2] @ B)
    Z = jnp.sum(seq.quad.w_z)

    def column(e):
        return solve(jnp.zeros_like(load), shells0, eqx.tree_at(lambda x: x.g, slips, TtMB - Z * e))
    C, infos = jax.lax.map(column, jnp.eye(slips.g.size, dtype=B.dtype))
    return d, C.T, jnp.concatenate([info[None], infos])


def iota_newton(seq, B, T, s_q, iota, ts, mu_star):
    """``(slips, mu_new, record)``: the outer loop's Newton step on the current target's moments ``mu_star`` for the
    target transform ``iota(s)`` at the field ``B`` (label temperature ``T``, label ``s_q`` at the quadrature points),
    for the stepper ``ts`` (its Newton settings and shells), and the slips of the new target for the next chunk
    (:func:`build_slips`).

    The miss is the weak residual ``r`` on the profile space (:func:`iota_residual`). The target enters only the slip
    rows, so the derivative of the state along the target is ``-A^-1`` times the slip rows' derivative ``Z e_j``,
    ``A`` the Gauss-Newton matrix of the step (:func:`_slip_responses`, one MINRES solve per moment, ``C`` their
    slips). Ideal moves keep the poloidal flux of every flux surface as a function of its toroidal flux, so only the
    slips change ``r``, by ``R d`` (:func:`slip_transform`). Before every chunk the field is still on its way to the
    equilibrium of the old target, hence the Newton direction's slips in the step. The step is the joint Newton step
    on the field and the target, ``R C delta = -(r + R d_N)``, ``d_N`` the slips of the Newton direction at
    ``mu_star``: a square system on the profile space, with no least-squares fit and nothing carried from one chunk to
    the next but the target. The Gauss-Newton matrix leaves out the second variation's terms in the current, so near a
    current-carrying equilibrium the step is a quasi-Newton step.

    ``record`` holds ``r`` (local means of ``iota - iota*`` with the field's sign), its largest magnitude and rms in
    ``s``, ``mu`` (the moments of ``B``), ``lam`` (the new target), ``delta``, ``r_eq = r + R d_N``, the MINRES
    counts."""
    mu_star = jnp.asarray(mu_star, dtype=T.dtype)
    L = int(mu_star.size)
    slips, mu = build_slips(seq, B, T, mu_star)
    residual = iota_residual(seq, B, s_q, iota, L)
    r = np.asarray(residual(B)[0], dtype=np.float64)
    R = slip_transform(residual, B, slips.maps(seq)[0], L)
    d_N, C, info = (np.asarray(v) for v in _slip_responses(ts, B, slips))
    r_eq = r + R @ d_N.astype(np.float64)
    delta = np.linalg.solve(R @ C.astype(np.float64), -r_eq)
    lam = np.asarray(mu_star, dtype=np.float64) + delta
    w = moment_widths(L)
    rec = dict(r=r.tolist(), r_max=float(np.abs(r).max()), r_rms=float(np.sqrt(np.sum(w * r ** 2) / np.sum(w))),
               mu=np.asarray(mu).tolist(), lam=lam.tolist(), delta=delta.tolist(), r_eq=r_eq.tolist(),
               minres=info.tolist())
    # the field changes hold the label alone: the new target changes only the right-hand side g = -Z mu*
    g = -jnp.sum(seq.quad.w_z) * jnp.asarray(lam, dtype=slips.g.dtype)
    return eqx.tree_at(lambda x: x.g, slips, g), lam, rec


@eqx.filter_jit
def current_on_label(seq, B, T, cells=None):
    """The moments ``mu`` (:func:`current_moments`) of the field ``B`` on the label of the temperature ``T``, on the
    profile space of ``cells`` cells (:func:`~mrx.relaxation.shells.profile_cells` of ``seq`` unless given)."""
    cells = profile_cells(seq) if cells is None else cells
    return _slip_parts(seq, B, slip_changes(seq, _label_of(seq, B, T), cells + PROFILE_DEGREE))[0]


def _derivatives(n_mu):
    """The ``n_mu`` unit-integral derivative splines ``M_j`` of the profile space with ``n_mu + 1`` splines."""
    return DerivativeSpline(profile_basis(n_mu + 1))


def current_target(current, cells, samples=20001):
    """The moments ``mu*_j = int M_j(s) mu0 I*(s) ds`` of the target ``current`` (``mu0 I*(s)`` in T m along
    ``+zeta``, a function of ``s``) on the profile space of ``cells`` cells, by the trapezoidal rule on ``samples``
    points of ``sqrt(s)``. They do not depend on the field: a rebuild on a new label keeps them."""
    x = np.linspace(0.0, 1.0, samples)
    M = np.asarray(basis_table(_derivatives(cells + PROFILE_DEGREE - 1), jnp.asarray(x ** 2)))
    w = np.full(samples, x[1]) * 2.0 * x                       # ds = 2 x dx
    w[[0, -1]] *= 0.5
    return jnp.asarray(M @ (w * np.asarray(current(x ** 2), dtype=float)))


def moment_widths(n_mu):
    """The widths in ``s`` of the supports of the ``n_mu`` splines ``M_j``: the weights of the L2 norm in ``s`` of the
    moments, which are local means of ``mu0 I(s)``."""
    d = _derivatives(n_mu)
    knots = np.asarray(d.T)
    return knots[d.p + 1:d.p + 1 + d.n] - knots[:d.n]


def current_error(mu, mu_star):
    """The error of the current's moments ``mu`` against the target's ``mu_star``, each weighted by the width in ``s``
    of the support of its spline ``M_j`` (:func:`moment_widths`): an L2 norm in ``s`` of the local means, relative to
    the target's. A zero target (no net current on any surface) has no size to compare with: its error is the absolute
    one, the weighted root mean square of ``mu`` in T m. The current vanishes on the axis, so there is no constant to
    fix."""
    w = moment_widths(len(mu_star))
    mu_star = np.asarray(mu_star, dtype=float)
    e = np.asarray(mu, dtype=float) - mu_star
    size = np.sum(w * mu_star ** 2)
    return float(np.sqrt(np.sum(w * e ** 2) / (size if size > 0.0 else np.sum(w))))


def spline_centres(n_mu):
    """The centres ``sqrt(int s M_j(s) ds)`` of the ``n_mu`` splines ``M_j``, where the moments sit, in
    ``sqrt(s)``."""
    d = _derivatives(n_mu)
    knots = np.asarray(d.T)
    return np.sqrt(np.array([knots[j:j + d.p + 2].mean() for j in range(d.n)]))


def load_current_target(seq, spec, B_file, scale=1.0, T=None, anisotropy=ANISOTROPY, label_tol=LABEL_TOL,
                        cells=None):
    """The moments ``mu*`` (:func:`current_target`) on the profile space of ``cells`` cells
    (:func:`~mrx.relaxation.shells.profile_cells` of ``seq`` unless given) of the current target
    ``spec``, times ``scale``: ``"file"``, the
    current of the geometry file's field ``B_file`` on its flux label (temperature ``T``, solved if not given with
    ``anisotropy`` to ``label_tol``), or a VMEC input namelist with ``NCURR = 1`` (its ``mu0 I(s)`` along ``phi``,
    :meth:`mrx.equilibria.indata.Indata.current`). The namelist's current is turned along the field with the sign of
    ``PHIEDGE``, then along ``+zeta`` with the sign of the toroidal flux of ``B_file``."""
    B_jk = seq.odd.evaluate_at_quadrature(B_file, 2)
    if spec == "file":
        if T is None:
            T = flux_label(seq, B_jk, anisotropy=anisotropy, tol=label_tol)[0]
        return scale * current_on_label(seq, B_file, T, cells)
    from mrx.equilibria.indata import read_indata
    nl = read_indata(spec)
    if nl.ncurr == 0:
        raise ValueError(f"{spec}: NCURR = 0 prescribes iota, a target of the twist (--iota.target), not a current")
    sign = float(np.sign(nl.phiedge) * np.sign(jnp.sum(seq.quad.w * B_jk[:, 2])))
    return current_target(lambda s: scale * sign * nl.current(s), profile_cells(seq) if cells is None else cells)
