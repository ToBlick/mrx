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

A target transform is not a control of ``W``. It sets the current target in an outer loop: before every chunk of
Newton steps :func:`iota_current` measures the field's poloidal flux on its label, asks for the slips that close the
miss and takes as the current target the current the field carries once it has followed them. The chunk then runs
with the slips of that target (:func:`build_slips`), as with any other.

- :func:`build_slips`, :func:`slip_table`, :func:`slip_maps` and :func:`current_moments` are the move as Newton
  unknowns, :func:`current_on_label` the moments of a field, :func:`current_target` those of a target and
  :func:`current_error` the current's error against them.
- :func:`iota_current` is the outer loop's update of the current target from a target transform,
  :func:`slip_request` its request.
- :func:`iota_fit` and :func:`enclosed_current` measure the transform and the current on the label's level sets.
- :func:`load_iota_target` reads a transform from a wout or a VMEC input namelist with ``NCURR = 0``,
  :func:`load_current_target` a current from a namelist with ``NCURR = 1``.

The potential ``f dzeta`` lives in the free 1-forms. On the wall ``T = 0`` and ``f = F(1)``, so its trace there is
the closed form ``F(1) dzeta`` and ``dB`` has no flux through the wall. ``F(1) != 0`` changes the poloidal flux inside
the wall and with it the net toroidal current.
"""
from __future__ import annotations


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.flux_label import ANISOTROPY, enclosed_flux_map, flux_label
from mrx.relaxation.physics import current as weak_current
from mrx.relaxation.shells import LABEL_TOL, PROFILE_CELLS, PROFILE_DEGREE, profile_basis
from mrx.spline_bases import DerivativeSpline, basis_table

#: The cumulative sums :func:`iota_fit` fits, sampled at this many points.
FIT_SAMPLES = 1024
#: The transform is at its target when the label fit misses it by less than this fraction of the target's largest
#: magnitude: ``relax.py --seed`` waits for it with ``--iota.target``.
AT_TARGET = 1e-2


def label_axis(seq, T, n_r=48, n_t=64, newton_steps=8, h=2e-3):
    """``(x_a, y_a, dx_a, dy_a)``, each ``(n_zeta_q,)``: the magnetic axis on every quadrature plane ``zeta = const``
    in the logical disc coordinates ``x + i y = r exp(2 pi i theta)``, and its derivative in ``zeta``. It is the
    maximum of the label's temperature ``T``, where ``s = 0``: the label's own axis, wherever it sits on the mesh.

    The start is the largest sample of ``T`` on a polar grid of the whole disc (``n_r`` by ``n_t``), then Newton on
    the gradient of ``T``, with the step halved until ``T`` rises. Gradient and Hessian in ``(x, y)`` are central
    differences of the values of ``T`` at the distance ``h``, since derivatives through the chart ``(x, y) -> (r,
    theta)`` cancel large terms next to the logical axis and lose all digits there in float32. The derivative of the
    axis is that of the stationary point, ``-H^-1 d grad T / d zeta``, with the ``zeta`` derivative of ``T`` exact.
    The planes are the quadrature's: the poloidal flux density (:func:`poloidal_density`) is needed there."""
    import mrx
    T_fn = DiscreteFunction(T, seq.basis_0, seq.even.E(0))

    def T_at(p, z):
        return T_fn(jnp.stack([jnp.hypot(p[0], p[1]), jnp.arctan2(p[1], p[0]) / (2 * jnp.pi) % 1.0, z]))[0]
    T_z = jax.grad(T_at, argnums=1)
    ex, ey = jnp.asarray([h, 0.0], T.dtype), jnp.asarray([0.0, h], T.dtype)

    def model(p, z):
        """The gradient and the Hessian of T at p by central differences."""
        t0, xp, xm, yp, ym = (T_at(p + d, z) for d in (0.0 * ex, ex, -ex, ey, -ey))
        pp, pm, mp, mm = (T_at(p + a + b, z) for a, b in ((ex, ey), (ex, -ey), (-ex, ey), (-ex, -ey)))
        g = jnp.stack([xp - xm, yp - ym]) / (2 * h)
        hxy = (pp - pm - mp + mm) / (4 * h * h)
        H = jnp.stack([jnp.stack([xp - 2 * t0 + xm, hxy * h * h]), jnp.stack([hxy * h * h, yp - 2 * t0 + ym])]) / (h * h)
        return g, H

    zz = jnp.asarray(seq.quad.x_z)
    r = (np.arange(n_r) + 0.5) / n_r
    th = np.arange(n_t) / n_t
    rr, tt = np.meshgrid(r, th, indexing="ij")
    disc = jnp.asarray(np.stack([rr * np.cos(2 * np.pi * tt), rr * np.sin(2 * np.pi * tt)], -1).reshape(-1, 2))
    vals = jax.lax.map(lambda z: jax.vmap(lambda p: T_at(p, z))(disc), zz, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    p = disc[jnp.argmax(vals, axis=1)]
    T_v = jax.jit(jax.vmap(T_at))

    @jax.jit
    def step(p):
        def one(q, z):
            g, H = model(q, z)
            return -jnp.linalg.solve(H, g)
        return jax.vmap(one)(p, zz)
    for _ in range(newton_steps):
        t0, d = T_v(p, zz), step(p)
        lam = jnp.ones(zz.shape, T.dtype)
        for _ in range(6):
            lam = jnp.where(T_v(p + lam[:, None] * d, zz) >= t0, lam, 0.5 * lam)
        p = jnp.where((T_v(p + lam[:, None] * d, zz) >= t0)[:, None], p + lam[:, None] * d, p)

    @jax.jit
    def slope(p):
        def one(q, z):
            gz = jnp.stack([T_z(q + ex, z) - T_z(q - ex, z), T_z(q + ey, z) - T_z(q - ey, z)]) / (2 * h)
            return -jnp.linalg.solve(model(q, z)[1], gz)
        return jax.vmap(one)(p, zz)
    dp = slope(p)
    p, dp = np.asarray(p), np.asarray(dp)
    return p[:, 0], p[:, 1], dp[:, 0], dp[:, 1]


def poloidal_density(seq, B, T):
    """The density of the poloidal flux at the quadrature points: ``J B . grad theta_a`` with ``theta_a`` the angle
    about the label's own axis (:func:`label_axis`), from the 2-form's components. Summed over the region inside a
    level set of the label it is the poloidal flux enclosed by that surface, for any position of the logical axis.
    The logical ``theta`` would not do: its cut ends on the logical axis, so it counts no flux inside a surface that
    does not enclose the logical axis and misses the flux between the two axes in every other. Where the magnetic axis
    is the logical one ``theta_a = theta``."""
    Bq = np.asarray(seq.odd.evaluate_at_quadrature(B, 2))
    xq = np.asarray(seq.quad.x)
    r, th = xq[:, 0], 2 * np.pi * xq[:, 1]
    xa, ya, dxa, dya = label_axis(seq, T)
    k = np.broadcast_to(np.arange(xa.size), seq.quad.shape).ravel()
    u, v = r * np.cos(th) - xa[k], r * np.sin(th) - ya[k]
    rho2 = u * u + v * v
    # d theta_a = (u dv - v du) / (2 pi rho^2), dx = cos dr - 2 pi r sin d theta, dy = sin dr + 2 pi r cos d theta
    d_r = (u * np.sin(th) - v * np.cos(th)) / (2 * np.pi * rho2)
    d_t = r * (u * np.cos(th) + v * np.sin(th)) / rho2
    d_z = (-u * dya[k] + v * dxa[k]) / (2 * np.pi * rho2)
    return Bq[:, 0] * d_r + Bq[:, 1] * d_t + Bq[:, 2] * d_z, Bq


def iota_fit(seq, B, T, s_q, degree):
    """The transform per full turn as a smooth function of the label ``s``: the poloidal flux enclosed by the label's
    level sets (:func:`poloidal_density`) against the toroidal flux they enclose, cumulative volume sums over the
    points sorted by ``s``, fitted by a polynomial of ``degree + 1`` without constant in the enclosed toroidal flux
    fraction and differentiated."""
    dens, Bq = poloidal_density(seq, B, T)
    w = np.asarray(seq.quad.w)
    order = np.argsort(np.asarray(s_q))
    psi = np.cumsum((w * dens)[order])
    phi = np.cumsum((w * Bq[:, 2])[order])
    pick = np.linspace(0, psi.size - 1, FIT_SAMPLES).astype(int)
    x = phi[pick] / phi[-1]
    c = np.linalg.lstsq(np.stack([x ** k for k in range(1, degree + 2)], axis=1), psi[pick] / phi[-1], rcond=None)[0]
    return seq.nfp * np.polynomial.Polynomial(np.concatenate([[0.0], c])).deriv()


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
    """The frozen part of the slip Newton step: the splines of the profile space at the points of the histopolation of
    the twist's potential (``values``, ``indices`` of :func:`slip_table`, on the label of one field), the steps ``S``
    ``(K, K - 1)`` with ``F_j = sum_k S_kj phi_k``, the target's part ``g`` of the right-hand side and ``D_inv``, the
    inverse of the slip block ``(T_i, T_j)_M``. Nothing in it holds a vector per slip, so it grows as the mesh does.
    Arrays only, so a rebuild does not recompile the step."""
    values: jnp.ndarray
    indices: jnp.ndarray
    S: jnp.ndarray
    g: jnp.ndarray
    D_inv: jnp.ndarray

    def maps(self, seq):
        """The maps of :func:`slip_maps` of these slips."""
        return slip_maps(seq, self.values, self.indices, self.S)


def _zeta_rule(seq):
    """``(x, w)``: the points ``(n_r, n_theta, n_zeta, q, 3)`` and the zeta weights ``(n_zeta, q)`` at which
    :func:`mrx.projectors.interpolate` histopolates the ``zeta`` part of a 1-form: the Greville points in ``r`` and
    ``theta`` and a Gauss rule on each Greville cell in ``zeta``, periodic coordinates wrapped into ``[0, 1)``."""
    from mrx.projectors import _wrap_periodic_point
    axes = seq.greville
    x_r, x_t = axes[0].point_rule[0][:, 0], axes[1].point_rule[0][:, 0]
    xz, wz = axes[2].span_rule
    x = jnp.stack(jnp.meshgrid(x_r, x_t, xz.ravel(), indexing="ij"), axis=-1)
    x = jax.vmap(lambda y: _wrap_periodic_point(seq, y))(x.reshape(-1, 3))
    return x.reshape(x_r.size, x_t.size, xz.shape[0], xz.shape[1], 3), wz


def slip_table(seq, T, B_jk, n):
    """``(values, indices)``, each ``(n_x, 4)``: the ``n`` splines of the profile space that can be nonzero at the
    label ``s`` on the points of :func:`_zeta_rule`, ``s`` that of the temperature ``T`` for the field with the values
    ``B_jk`` at the quadrature points (:func:`mrx.flux_label.enclosed_flux_map`)."""
    import mrx
    x, _ = _zeta_rule(seq)
    T_fn = DiscreteFunction(T, seq.basis_0, seq.even.E(0))
    s_of = enclosed_flux_map(seq, T, B_jk)
    basis = profile_basis(n)
    return jax.lax.map(lambda y: basis.evaluate_local(s_of(T_fn(y)[0])), x.reshape(-1, 3),
                       batch_size=mrx.MAP_BATCH_SIZE_INNER)


def slip_maps(seq, values, indices, S):
    """``(move, move_t)``: the map ``d -> T d`` to the field change (a 2-form of ``seq.odd``) of the twist
    ``F = sum_j d_j F_j``, ``F_j = sum_k S_kj phi_k(s)``, and its transpose. ``move`` is the twist ``curl(F grad zeta)``
    with the label's splines at the histopolation points taken from :func:`slip_table`: the values of ``F`` there,
    their zeta integrals over the Greville cells, the three 1-D histopolation and collocation solves, the restriction
    to the polar space and the curl. All of it is linear in ``d`` and explicit, and jax transposes it."""
    from mrx.projectors import _restrict, _solve_tensor_collocation_axis
    free = seq.odd.free
    axes = seq.greville
    _, wz = _zeta_rule(seq)
    n_r, n_t = axes[0].point_rule[0].shape[0], axes[1].point_rule[0].shape[0]
    zeros = jnp.zeros(sum(int(np.prod(sh)) for sh in seq.basis_1.shape[:2]), values.dtype)

    def move(d):
        f = jnp.sum(values * (S @ d.astype(S.dtype))[indices], axis=1).reshape((n_r, n_t) + wz.shape)
        m = jnp.einsum("ijlq,lq->ijl", f, wz)
        for j, matrix in enumerate((axes[0].coll, axes[1].coll, axes[2].hist)):
            m = _solve_tensor_collocation_axis(matrix, m, axis=j)
        return seq.odd.restrict(free.G[1] @ _restrict(free, jnp.concatenate([zeros, m.ravel()]), 1), 2)
    zero = jnp.zeros(S.shape[1], S.dtype)
    # through the linearisation: the scatter of the polar restriction has a transpose only in its JVP form
    move_t = jax.linear_transpose(jax.linearize(move, zero)[1], zero)
    # MINRES calls the step's operator in float32, the transpose takes only the dtype it was built for
    return move, lambda y: move_t(y.astype(values.dtype))[0]


def _steps(K, dtype):
    """``S`` ``(K, K - 1)``, ``S_kj = 1`` for ``k > j``, ``j = 0 .. K - 2``: the steps ``F_j = sum_{k > j} phi_k`` of
    the ``K`` splines of the profile space, whose derivatives in ``s`` are the unit-integral derivative splines
    ``M_j``. They vanish on the axis and go as ``s`` there."""
    return jnp.asarray(np.tril(np.ones((K, K - 1)), -1), dtype=dtype)


def current_moments(seq, B, move_t):
    """``mu_j = -(B, T_j)_M / Z``, the local means ``int M_j(s) mu0 I(s) ds`` of the net toroidal current of ``B``
    inside the label's surfaces (T m, along ``+zeta``), with ``move_t`` the transpose of the slips' field change
    (:func:`slip_maps`) and ``Z`` the ``zeta`` extent of the integrals."""
    return -move_t(seq.odd.M[2] @ B) / jnp.sum(seq.quad.w_z)


@eqx.filter_jit
def _slip_parts(seq, B, T, cells):
    K = cells + PROFILE_DEGREE
    values, indices = slip_table(seq, T, seq.odd.evaluate_at_quadrature(B, 2), K)
    S = _steps(K, T.dtype)
    move, move_t = slip_maps(seq, values, indices, S)
    M2 = seq.odd.M[2]
    D = jax.lax.map(lambda e: move_t(M2 @ move(e)), jnp.eye(S.shape[1], dtype=S.dtype))
    return values, indices, S, current_moments(seq, B, move_t), 0.5 * (D + D.T)


def build_slips(seq, B, T, mu_star):
    """``(slips, mu)``: the :class:`Slips` at the field ``B`` with the label temperature ``T`` and the target moments
    ``mu_star`` (:func:`current_target`, one per slip, on the profile space of ``mu_star.size - 2`` cells), and the
    moments ``mu`` of ``B`` (:func:`current_moments`). The block costs two applies of the field change per slip,
    without a solve. The target's part is ``g = -Z mu_star``."""
    values, indices, S, mu, D = _slip_parts(seq, B, T, int(mu_star.size) + 1 - PROFILE_DEGREE)
    g = -jnp.sum(seq.quad.w_z) * mu_star
    # inverted on the host in float64, as the shell block is
    return Slips(values, indices, S, g, jnp.asarray(np.linalg.inv(np.asarray(D, dtype=np.float64)))), mu


def slip_request(seq, B, T, s_q, iota, cells=PROFILE_CELLS, degree=6, samples=FIT_SAMPLES):
    """``(d, fit)``: the slips whose twist, with the rest of the field frozen, takes the poloidal flux of ``B`` inside
    the level sets of the label ``s_q`` (temperature ``T``) to that of the target transform ``iota(s)`` (per full turn,
    its magnitude taken with the sign of the field's transform), and the field's transform ``fit`` (:func:`iota_fit`).

    The poloidal and toroidal fluxes per unit ``zeta`` inside the level sets, ``psi(s)`` and ``Phi s``, are the
    cumulative volume sums of the poloidal flux density about the label's axis (:func:`poloidal_density`) and of the
    2-form's ``zeta`` density over the points sorted by ``s``, as in :func:`iota_fit`, at ``samples`` points: functions
    of ``s`` alone. The target's is ``psi*(s) = Phi / nfp int_0^s iota*``. The twist ``F`` adds ``-F(s)`` to ``psi``,
    so ``d`` is the least-squares fit of ``F = sum_j d_j F_j`` to ``psi - psi*`` at the samples, relative to ``s``: the
    miss of the mean transform inside each surface. The fluxes themselves, not the derivative of the transform, set
    ``d``: the slip at the wall moves only the flux next to the wall and the total, which a fit of the transform, or a
    polynomial fit of the miss, does not see."""
    dens, Bq = poloidal_density(seq, B, T)
    w = np.asarray(seq.quad.w) / np.sum(np.asarray(seq.quad.w_z))
    order = np.argsort(np.asarray(s_q))
    psi = np.cumsum((w * dens)[order])
    phi = np.cumsum((w * Bq[:, 2])[order])
    pick = np.linspace(0, psi.size - 1, samples).astype(int)
    s = phi[pick] / phi[-1]
    fit = iota_fit(seq, B, T, s_q, degree)
    sign = float(np.sign(psi[-1] / phi[-1]))
    s_fine = np.linspace(0.0, 1.0, 4001)
    i_fine = sign * np.abs(np.asarray(iota(s_fine)))
    psi_star = phi[-1] / seq.nfp * np.concatenate([[0.0], np.cumsum(0.5 * (i_fine[1:] + i_fine[:-1]) * np.diff(s_fine))])
    miss = psi[pick] - np.interp(s, s_fine, psi_star)
    K = cells + PROFILE_DEGREE
    A = np.asarray(basis_table(profile_basis(K), jnp.asarray(s))).T @ np.asarray(_steps(K, Bq.dtype))
    # each row divided by s: the samples equal in volume, the flux next to the axis goes as s
    return np.linalg.lstsq(A / s[:, None], miss / s, rcond=None)[0], fit


@eqx.filter_jit
def _relaxed_moments(ts, B, values, indices, S, d):
    """The moments of ``B + T d + Q u``, ``u`` the Newton velocity (with the shells of ``ts``) of the step with the
    slips held at ``d``: the current the field carries once the ideal part has followed the slips, to first order. Also
    returns the signed MINRES count of that solve."""
    from mrx.relaxation.newton import gauss_newton_parts, target_direction
    seq = ts.seq
    move, move_t = slip_maps(seq, values, indices, S)
    dB = move(d.astype(S.dtype))
    change, change_t, _ = gauss_newton_parts(seq, B, None, ts.newton_mass_solve_tol)
    _, load = ts._lorentz(B, None)
    # the slip column moved to the right-hand side: the ideal and shell rows of target_direction's system
    u, _, info, _, _, _ = target_direction(seq, B, load - change_t(dB), jnp.zeros(seq.even.n(1)), ts.shells, None,
                                           ts.newton_penalty, ts.newton_tol, ts.newton_maxiter,
                                           ts.newton_mass_solve_tol)
    return current_moments(seq, B + dB + change(seq.even.evaluate_at_quadrature(u, 2)), move_t), info


def iota_current(seq, B, T, s_q, iota, ts, cells=PROFILE_CELLS, degree=6):
    """``(lam, record)``: the current target ``lambda`` (moments as those of :func:`current_target`, T m) that is the
    multiplier of the target transform ``iota(s)`` at the field ``B`` (label temperature ``T``, label ``s_q`` at the
    quadrature points), for the stepper ``ts`` (its Newton settings and shells), on the profile space of ``cells``
    cells. The chunk runs with ``build_slips(seq, B, T, lam)``.

    The energy is ``W = int B^2/2 dV - int p* dV - int lambda dpsi_pol``, the work of the unknown current on the
    slipped flux. Its stationary points in the slips have the moments ``mu = lambda``. ``lambda`` is set from the field
    as it is, nothing is carried over: ``d_req`` are the slips that close the miss of the poloidal flux with the rest
    frozen (:func:`slip_request`), and ``lambda`` the moments of the Newton step with the slips held at ``d_req``
    (:func:`_relaxed_moments`), the current the field carries once the ideal part and the shells have followed the
    slips. The first Newton step at this ``lambda`` slips by ``d_req``. The slip block ``H`` alone,
    ``lambda = mu - H d_req / Z``, overshoots: the response with the ideal part relaxed is ``S`` with
    ``(d, S d) / (d, H d)`` 0.6 to 0.9 along the request on li383, and the update then oscillates.

    ``record`` holds ``mu`` (the moments of ``B``), ``lambda``, ``d_req``, its norm ``|d_req|_H``, the signed MINRES
    count of the Newton solve and the miss of the transform (:func:`iota_fit`) on ``s = 0.05 .. 0.95`` and at
    ``s = 0.01``."""
    values, indices, S, mu, D = _slip_parts(seq, B, T, cells)
    d, fit = slip_request(seq, B, T, s_q, iota, cells, degree)
    D = np.asarray(D, dtype=np.float64)
    lam, info = _relaxed_moments(ts, B, values, indices, S, jnp.asarray(d))
    lam = np.asarray(lam, dtype=np.float64)
    s = np.linspace(0.05, 0.95, 19)
    miss = np.abs(np.abs(fit(s)) - np.abs(np.asarray(iota(s))))
    rec = dict(mu=np.asarray(mu).tolist(), lam=lam.tolist(), d_req=d.tolist(), d_req_H=float(np.sqrt(d @ D @ d)),
               response_it=int(info), miss_max=float(miss.max()), miss_median=float(np.median(miss)),
               miss_axis=float(abs(abs(fit(0.01)) - abs(float(iota(0.01))))))
    return jnp.asarray(lam, dtype=T.dtype), rec


@eqx.filter_jit
def current_on_label(seq, B, T, cells=PROFILE_CELLS):
    """The moments ``mu`` (:func:`current_moments`) of the field ``B`` on the label of the temperature ``T``, on the
    profile space of ``cells`` cells."""
    K = cells + PROFILE_DEGREE
    values, indices = slip_table(seq, T, seq.odd.evaluate_at_quadrature(B, 2), K)
    return current_moments(seq, B, slip_maps(seq, values, indices, _steps(K, T.dtype))[1])


def _derivatives(n_mu):
    """The ``n_mu`` unit-integral derivative splines ``M_j`` of the profile space with ``n_mu + 1`` splines."""
    return DerivativeSpline(profile_basis(n_mu + 1))


def current_target(current, cells=PROFILE_CELLS, samples=20001):
    """The moments ``mu*_j = int M_j(s) mu0 I*(s) ds`` of the target ``current`` (``mu0 I*(s)`` in T m along
    ``+zeta``, a function of ``s``) on the profile space of ``cells`` cells, by the trapezoidal rule on ``samples``
    points of ``sqrt(s)``. They do not depend on the field: a rebuild on a new label keeps them."""
    x = np.linspace(0.0, 1.0, samples)
    M = np.asarray(basis_table(_derivatives(cells + PROFILE_DEGREE - 1), jnp.asarray(x ** 2)))
    w = np.full(samples, x[1]) * 2.0 * x                       # ds = 2 x dx
    w[[0, -1]] *= 0.5
    return jnp.asarray(M @ (w * np.asarray(current(x ** 2), dtype=float)))


def current_error(mu, mu_star):
    """The error of the current's moments ``mu`` against the target's ``mu_star``, each weighted by the width in ``s``
    of the support of its spline ``M_j``: an L2 norm in ``s`` of the local means, relative to the target's. A zero
    target (no net current on any surface) has no size to compare with: its error is the absolute one, the weighted
    root mean square of ``mu`` in T m. The current vanishes on the axis, so there is no constant to fix."""
    d = _derivatives(len(mu_star))
    knots = np.asarray(d.T)
    w = knots[d.p + 1:d.p + 1 + d.n] - knots[:d.n]
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
                        cells=PROFILE_CELLS):
    """The moments ``mu*`` (:func:`current_target`) on the profile space of ``cells`` cells of the current target
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
    return current_target(lambda s: scale * sign * nl.current(s), cells)
