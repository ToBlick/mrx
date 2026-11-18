"""Prescribed pressure: the volumes of the flux shells as unknowns of the Newton step.

The incompressible relaxation keeps the volume of every flux shell, and the pressure it ends with is the multiplier
of those volumes. To prescribe a pressure ``p*(s)`` the shells must change their volumes, and only their volumes.
The velocity of a step is then

    u = curl a + sum_k c_k w_k,        div w_k ~ phi_k(s) - <phi_k>,

with ``phi_k`` the radial 0-form B-splines of the mesh as functions of ``x = sqrt(s)``, ``s`` the flux label of the
field (:mod:`mrx.flux_label`). The space of these velocities depends only on the divergences ``f_k`` of the ``w_k``,
since two velocities with one divergence differ by a divergence-free one, which ``curl a`` already holds. So each
``w_k`` is an explicit velocity with that divergence (:func:`radial_velocity`): the radial flux summed along ``r``
from the axis, which needs no storage per shell. The divergence ``f_k`` is the projection of ``phi_k - <phi_k>``
onto the 3-forms. A step needs the divergence of a combination of the shells only, one 3-form mass solve
(:func:`shell_maps`). The energy is the magnetic energy less the work of the target pressure carried with the
shells (gamma = 0),

    W = int B^2/2 dV - int p*(s) dV,

whose derivative along ``w_k`` at a relaxed field is ``int (p - p*) f_k dxi``, with ``p`` the multiplier of the
divergence of the relaxation's 2-form velocity. The Newton step (:func:`mrx.relaxation.newton.shell_direction`)
solves the Gauss-Newton system of ``W`` on this enlarged velocity space. The pressure term's own second variation is
left out: it is the term that makes the step indefinite on compressible velocities (finite gamma), and on the
shells the magnetic part alone is a good model.

A :class:`Shells` holds the frozen part of one step: the label at the quadrature points, the pressure part of the
right-hand side and the inverse of the shell block. It depends on the flux label of the field, so
:func:`build_shells` rebuilds it every few steps (the relaxation's chunk). The splines sum to one, so the sum of the
shell divergences is zero: the shells kept are the combinations whose coefficients sum to zero, since a uniform
pressure cannot move a fixed wall. The pressure's constant, which no move changes, is the gauge ``<p> = <p*>``
(:func:`pressure_error`): with it all ``K`` moments of ``p - p*`` vanish at an equilibrium.

The targets: a profile ``p*(s)`` (mu0 p in T^2, from the geometry file or a measured profile on flux surfaces,
:class:`PressureTarget`), made constant across seeded island chains (:func:`plateau_intervals`,
:func:`flattened`), or pressure data at points in space (:class:`PressureData`), refitted to a profile of the current
labels at every rebuild.
"""
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.flux_label import enclosed_flux, enclosed_flux_map, flux_label
from mrx.operators import _mass_solve
from mrx.relaxation.newton import atom_profiles, parallel_penalty_profile, second_variation
from mrx.spline_bases import basis_table


#: The relative tolerance of the shells' label solve. Solved cold on the fields of a run, the label ``s`` is then
#: within 3.1e-6 of the converged one (1e-9) at less than half its iterations: 754 to 861 against 1795 to 1909 on W7-X
#: graded (24,48,28), 846 to 1322 against 2428 to 2746 on li383 graded (48,64,24) and 1077 to 1216 against 3014 to 3188
#: on W7-X graded (32,64,36). On the last two the sequence's 1e-8 ran into the solve's 2000 (2026-11-16).
LABEL_TOL = 1e-5


@eqx.filter_jit
def label(seq, B, anisotropy, guess=None, tol=LABEL_TOL):
    """``(s_q, T, info)``: the flux label at the quadrature points, its temperature (:func:`mrx.flux_label.flux_label`
    to the tolerance ``tol``, warm-started from ``guess``) and the signed count of its solve."""
    B_jk = seq.odd.evaluate_at_quadrature(B, 2)
    T, info = flux_label(seq, B_jk, anisotropy=anisotropy, guess=guess, tol=tol)
    return enclosed_flux(seq, T, B_jk), T, info


def shell_table(seq, x_q):
    """``(values, indices)``, each ``(n_q, p + 1)``: the radial 0-form B-splines of the mesh (degree ``p``, its graded
    knots, the axis and wall splines included) that can be nonzero at ``x = sqrt(s)``, at the quadrature points. On the
    surfaces the mesh was cut along, ``x`` is the logical radius, so the shells follow the mesh's grading. They sum to
    one."""
    return jax.vmap(seq.basis_0.Lambda[0].evaluate_local)(x_q)


def zero_total(f, e, m):
    """The 3-form ``f`` less the multiple of ``e`` (the 3-form of the constant 1) that takes its integral ``m . f`` to
    zero, with ``m`` the integrals of the basis functions. Only a 3-form of zero integral is the divergence of a
    velocity with ``u . n = 0``."""
    return f - e * (m @ f) / (m @ e)


def zero_total_t(y, e, m):
    """The transpose of :func:`zero_total`."""
    return y - m * (e @ y) / (m @ e)


def _column_weights(seq):
    """The radial weights ``omega`` (one per raw radial 3-form ring) of the tangential part of
    :func:`radial_velocity`: the logical widths of the rings with the full theta count, normalised to sum to one."""
    d = seq.basis_0.dLambda[0]
    T, q = np.asarray(d.T), d.p
    width = (T[q + 1:q + 1 + d.n] - T[:d.n]) / (q + 1)       # the coefficients of 1 in the unit-integral splines
    nt = seq.ns[1]
    counts = np.full(seq.ns[0], nt) if seq.theta_counts is None else np.asarray(seq.theta_counts)
    full = np.arange(d.n) >= 1                                # ring 0 is not in the space (the axis)
    full &= counts[1:d.n + 1] == nt                           # 3-form ring a has the count of primal ring a + 1
    omega = np.where(full, width, 0.0)
    return omega / omega.sum()


def radial_velocity(seq, f):
    """An even 2-form ``w`` with ``div w = f`` and ``w . n = 0`` on the wall, for an even 3-form ``f`` with zero
    integral, without a solve.

    On the tensor-product coefficients the divergence is ``d_r w^r + d_theta w^theta + d_zeta w^zeta`` with the
    coefficient differences ``d``. The radial flux ``w^r`` is the sum of ``f`` along ``r`` from the axis, so it
    vanishes on the wall when every column ``(theta, zeta)`` of ``f`` sums to zero. A column's sum ``h`` is moved
    across the columns by a tangential flux instead, spread over the rings with the full theta count by
    :func:`_column_weights`: ``w^theta`` sums ``h`` less its theta mean along theta, ``w^zeta`` sums the theta mean
    along zeta. Both sums are periodic because ``h`` and its theta mean sum to zero. The rings near the axis carry
    fewer theta cells (graded meshes), and a radial sum stays in the coarser theta space of each ring, so ``w`` lies in
    the extracted space, and the reflection of the stellarator symmetry maps it onto itself. Linear in ``f``."""
    even = seq.even
    shape = tuple(int(v) for v in seq.basis_3.shape[0])
    F = (even.E(3).T @ f).reshape(shape)
    h = jnp.sum(F, axis=0)
    omega = jnp.asarray(_column_weights(seq))[:, None, None]
    F = F - omega * h
    w_r = jnp.concatenate([jnp.zeros((1,) + shape[1:]), jnp.cumsum(F, axis=0)])
    h_t = jnp.mean(h, axis=0)

    def periodic_sum(x, axis):
        # y with y[j + 1] - y[j] = x[j] (periodic, x sums to zero), its mean removed
        y = jnp.cumsum(x, axis=axis) - x
        return y - jnp.mean(y, axis=axis, keepdims=True)
    w_t = omega * periodic_sum(h - h_t, 0)
    w_z = omega * jnp.broadcast_to(periodic_sum(h_t, 0), shape[1:])
    return even.conforming(jnp.concatenate([w_r.ravel(), w_t.ravel(), w_z.ravel()]), 2)


def shell_maps(seq, x, Q, e, m, mass_tol):
    """``(div, div_t, velocity, velocity_t)``: the maps ``c -> sum_k c_k f_k`` (3-forms) and ``c -> sum_k c_k w_k``
    (2-forms, :func:`radial_velocity`) of the coefficients ``c`` of the zero-sum combinations ``Q`` of the shells, at
    the label ``x = sqrt(s)`` (quadrature points), and their transposes.

    The divergence of the combination ``g = sum_k (Q c)_k phi_k`` is the projection of ``g - <g>`` onto the 3-forms,
    ``M_3^-1 load(g)`` less its mean (:func:`zero_total`). Since everything is linear in ``c``, one 3-form mass solve
    gives the divergence of the combination, and the transpose costs one more, to the relative tolerance
    ``mass_tol``. ``<g>`` is the mean with ``e`` as the volume element. The load and :func:`radial_velocity` are
    explicit and transposed by jax, the mass solve is its own transpose up to its tolerance, so ``velocity_t`` is the
    transpose of ``velocity`` to that tolerance. The Newton step passes the tolerance of the Hessian's 1-form mass
    solves. At 1e-6 the asymmetry of the bordered operator is within a factor of five of the Hessian's own (1.2e-5
    against 2.3e-6 on li383 graded (16,32,12), 2.4e-6 against 6.9e-6 on (48,64,24), 3.4e-6 against 2.0e-5 on
    (64,80,32), the largest of four random pairs), the 3-form solve takes 2 to 3 iterations, and MINRES takes the same
    count at the step's tolerance 0.1 for any 3-form tolerance from 1e-3 to 1e-10 (2026-11-17)."""
    table = shell_table(seq, x)
    even = seq.even
    zero = jnp.zeros(Q.shape[1], Q.dtype)

    def load(c):
        values, indices = table
        return even._scalar_load_values(jnp.sum(values * (Q @ c)[indices], axis=1), 3)
    # through the linearisation: the scatter of the polar restriction has a transpose only in its JVP form
    load_t = jax.linear_transpose(jax.linearize(load, zero)[1], zero)
    f0 = jnp.zeros(e.shape, e.dtype)
    R_t = jax.linear_transpose(jax.linearize(lambda f: radial_velocity(seq, f), f0)[1], f0)

    def div(c):
        return zero_total(_mass_solve(even, load(c), 3, tol=mass_tol), e, m)

    def div_t(y):
        return load_t(_mass_solve(even, zero_total_t(y, e, m), 3, tol=mass_tol))[0]

    def velocity(c):
        return radial_velocity(seq, div(c))

    def velocity_t(y):
        return div_t(R_t(y)[0])
    return div, div_t, velocity, velocity_t


class Shells(eqx.Module):
    """The frozen part of the shell-volume Newton step: the label ``x = sqrt(s)`` at the quadrature points, the
    orthonormal combinations ``Q`` ``(K, K - 1)`` of the splines whose coefficients sum to zero (on which the block is
    positive definite, a singular block would make the step's preconditioner singular), the 3-form ``e`` of the
    constant 1 and the integrals ``m`` of the 3-form basis functions (:func:`zero_total`), the pressure part ``g_p`` of
    the right-hand side and ``D_inv``, the inverse of the Gauss-Newton block of the shell velocities. Nothing in it
    holds a vector per shell, so it grows as the mesh does. Arrays only, so a rebuild does not recompile the step."""
    x: jnp.ndarray
    Q: jnp.ndarray
    e: jnp.ndarray
    m: jnp.ndarray
    g_p: jnp.ndarray
    D_inv: jnp.ndarray

    def maps(self, seq, mass_tol):
        """The maps of :func:`shell_maps` of these shells, with their 3-form mass solves to ``mass_tol``."""
        return shell_maps(seq, self.x, self.Q, self.e, self.m, mass_tol)


@eqx.filter_jit
def _shell_parts(seq, B, x, Q, e, m, p_q, kappa, mass_tol):
    _, div_t, velocity, velocity_t = shell_maps(seq, x, Q, e, m, mass_tol)
    n = Q.shape[1]
    g_p = div_t(seq.even._scalar_load_values(p_q, 3))
    penalty = parallel_penalty_profile(atom_profiles(seq, B), kappa)
    Hs = second_variation(seq, B, None, penalty, mass_tol)
    D = jax.lax.map(lambda u: velocity_t(Hs(velocity(u))), jnp.eye(n, dtype=Q.dtype))
    return g_p, 0.5 * (D + D.T)


def build_shells(seq, B, s_q, p_q, kappa, mass_tol):
    """The :class:`Shells` at the field ``B``, with the flux label ``s_q`` and the target ``p_q`` (mu0 p) at the
    quadrature points, and the penalty ``kappa`` and mass-solve tolerance of the Newton operator. It costs one Hessian
    apply and two 3-form mass solves per shell for the block, and two 3-form mass solves for ``e`` and ``g_p``. The
    pressure part is ``int p* f_k dxi``, the target's work along ``w_k``."""
    even = seq.even
    x = jnp.sqrt(s_q)                                     # the label is a share of the toroidal flux, in [0, 1]
    K = seq.basis_0.Lambda[0].n
    Q = jnp.asarray(np.linalg.qr(np.eye(K) - 1.0 / K)[0][:, :K - 1])
    m = even._scalar_load_values(jnp.ones_like(s_q), 3)          # the integrals of the basis functions
    e = even.M[3].solve(m)
    g_p, D = _shell_parts(seq, B, x, Q, e, m, p_q, kappa, mass_tol)
    # inverted on the host in float64, as the polar Gram blocks are: the block's condition (2e6 on li383 graded
    # (16,32,12)) is beyond a float32 inverse, which came out indefinite and stopped MINRES on nonpositive curvature
    return Shells(x, Q, e, m, g_p, jnp.asarray(np.linalg.inv(np.asarray(D, dtype=np.float64))))


def plateau_intervals(rows, margin=1.0):
    """The intervals of ``s`` that seeded island chains occupy, from the seeding's estimate (the rows of
    :func:`mrx.relaxation.seeding.energy_seed`: resonant radius ``r`` and estimated full width ``w`` in the logical
    radius, which is ``sqrt(s)`` on the file's surfaces): ``sqrt(s)`` in ``r +- margin w / 2``, overlapping intervals
    merged. Ideal moves keep the toroidal flux of every surface, so the intervals stay where the chains are."""
    iv = sorted((max(row["r"] - 0.5 * margin * row["w"], 0.0) ** 2, min(row["r"] + 0.5 * margin * row["w"], 1.0) ** 2)
                for row in rows)
    merged = []
    for lo, hi in iv:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def flattened(profile, intervals):
    """The profile ``profile(s)`` made constant on each interval of ``s``: ``profile(sigma(s))`` with ``sigma`` the
    label that counts only the flux outside the intervals. The values at the axis and the wall stay, the profile is
    constant and continuous across each interval, and the rest of it gets steeper. The two branches of an island's
    separatrix are one surface with one pressure, so a target that falls across an island cannot be met: the
    relaxation then keeps a pressure that varies along the island's field lines and a residual that does not fall."""
    s = np.linspace(0.0, 1.0, 4001)
    inside = sum(np.clip(s, lo, hi) - lo for lo, hi in intervals)
    total = sum(hi - lo for lo, hi in intervals)
    values = jnp.asarray(profile(jnp.asarray((s - inside) / (1.0 - total))))
    s = jnp.asarray(s)
    return lambda x: jnp.interp(x, s, values)


def fit_profile(seq, s_i, p_i, smoothing=1e-3):
    """The profile ``p*(s)`` through the data ``(s_i, p_i)`` in the radial 0-form splines of ``sqrt(s)``: least
    squares with ``smoothing`` times the squared second differences of the coefficients (relative to the data's
    scale), and ``p*(1) = 0`` (the wall spline's coefficient zero). The edge value is imposed because data rarely reach
    the wall, and the drop between the last datum and the wall sets the level of the relaxation's pressure everywhere
    (it vanishes on the wall), a constant no shell move changes."""
    rb = seq.basis_0.bases[0].bases[0]
    A = np.asarray(basis_table(rb, jnp.sqrt(jnp.asarray(s_i)))).T[:, :-1]
    D2 = np.diff(np.eye(A.shape[1] + 1), 2, axis=0)[:, :-1]
    w = smoothing * np.linalg.norm(A, 2) ** 2
    a = jnp.asarray(np.append(np.linalg.solve(A.T @ A + w * D2.T @ D2, A.T @ np.asarray(p_i)), 0.0))
    return lambda s: basis_table(rb, jnp.sqrt(s)).T @ a


def logical_points(seq, X, newton_steps=8):
    """``(x, miss)``: the logical points of the physical points ``X`` (Cartesian, ``(n, 3)``) and the distance
    ``|Phi(x) - X|`` left. The nearest point of a coarse logical grid, then Newton in the chart ``(r cos 2 pi theta,
    r sin 2 pi theta, zeta)``, which is smooth at the axis."""
    def chart(p):
        return jnp.stack([jnp.hypot(p[0], p[1]), jnp.arctan2(p[1], p[0]) / (2.0 * jnp.pi) % 1.0, p[2]])

    r, t, z = np.meshgrid(np.linspace(0.02, 1.0, 25), np.linspace(0.0, 1.0, 64, endpoint=False),
                          np.linspace(0.0, 1.0, 32, endpoint=False), indexing="ij")
    grid = jnp.asarray(np.stack([r.ravel(), t.ravel(), z.ravel()], axis=1))
    G = jax.lax.map(seq.map, grid, batch_size=4096)
    X = jnp.asarray(X)
    x0 = grid[jnp.argmin(jnp.sum((X[:, None, :] - G[None, :, :]) ** 2, axis=2), axis=1)]

    def invert(x, Xi):
        p = jnp.stack([x[0] * jnp.cos(2.0 * jnp.pi * x[1]), x[0] * jnp.sin(2.0 * jnp.pi * x[1]), x[2]])
        for _ in range(newton_steps):
            p = p - jnp.linalg.solve(jax.jacfwd(lambda q: seq.map(chart(q)))(p), seq.map(chart(p)) - Xi)
        y = chart(p)
        return y, jnp.linalg.norm(seq.map(y) - Xi)
    return jax.vmap(invert)(x0, X)


class PressureTarget(NamedTuple):
    """A pressure target ``p*(s)``, mu0 p in T^2 as a function of the normalised toroidal flux. ``profile`` is fixed,
    or ``None`` when ``data`` (a :class:`PressureData`) refits it at every rebuild."""
    profile: Optional[object] = None
    data: Optional["PressureData"] = None

    def at(self, seq, B, T):
        """The profile for the field ``B`` with the label temperature ``T``."""
        return self.profile if self.data is None else self.data.fit(seq, B, T)


class PressureData(NamedTuple):
    """Pressure data at points in space: the logical points ``x`` (from :func:`logical_points`, once, the mesh does
    not move), the values ``p`` (mu0 p in T^2) and the ``smoothing`` of :func:`fit_profile`."""
    x: jnp.ndarray
    p: np.ndarray
    smoothing: float = 1e-3

    def labels(self, seq, B, T):
        """The flux labels of the points for the field ``B`` with the label temperature ``T``."""
        B_jk = seq.odd.evaluate_at_quadrature(B, 2)
        T_x = jax.vmap(DiscreteFunction(T, seq.basis_0, seq.even.E(0)))(self.x)[:, 0]
        return enclosed_flux_map(seq, T, B_jk)(T_x)

    def fit(self, seq, B, T):
        """The profile through the data at the points' current labels (they move with the surfaces)."""
        return fit_profile(seq, self.labels(seq, B, T), self.p, self.smoothing)


def load_target(seq, spec, scale=1.0, smoothing=1e-3):
    """The :class:`PressureTarget` of ``spec``: ``"file"`` (the geometry file's pressure) or a JSON file of measured
    pressure in pascal, ``{"rho": [...], "p_pa": [...]}`` on flux surfaces (``rho = sqrt(s)``) or
    ``{"points": [[R, Z, phi_deg], ...], "p_pa": [...]}`` at points in space (metres, degrees). Profiles are
    multiplied by ``scale``. The constant of a measured profile is removed so that it vanishes on the wall (a uniform
    pressure does no work on a fixed wall), points are fitted at every rebuild (:class:`PressureData`) with
    ``smoothing``."""
    import json
    mu0 = 4e-7 * np.pi
    if spec == "file":
        p_r = seq.equilibrium["profiles"]["pressure"]               # Pa, a function of r = sqrt(s)
        edge = float(p_r(1.0))
        s = np.linspace(0.0, 1.0, 2001)
        values = jnp.asarray(scale * mu0 * (p_r(np.sqrt(s)) - edge))
        return PressureTarget(profile=lambda x: jnp.interp(x, jnp.asarray(s), values))
    with open(spec) as fh:
        d = json.load(fh)
    p = scale * mu0 * np.asarray(d["p_pa"], dtype=float)
    if "rho" in d:
        rho = np.asarray(d["rho"], dtype=float)
        return PressureTarget(profile=fit_profile(seq, rho ** 2, p - np.interp(1.0, rho, p), smoothing))
    pts = np.asarray(d["points"], dtype=float)
    phi = np.radians(pts[:, 2])
    x, miss = logical_points(seq, np.stack([pts[:, 0] * np.cos(phi), pts[:, 0] * np.sin(phi), pts[:, 1]], axis=1))
    if float(jnp.max(miss)) > 1e-6 * float(jnp.max(jnp.abs(jnp.asarray(pts[:, :2])))):
        raise ValueError(f"{spec}: a point is outside the domain or the map inversion failed "
                         f"(|Phi(x) - X| up to {float(jnp.max(miss)):.1e} m)")
    return PressureTarget(data=PressureData(x, p, smoothing))


def target_beta(seq, B, p_q):
    """The volume beta of the target values ``p_q`` at the quadrature points on the field ``B``,
    ``int p* dV / int B^2/2 dV``. In the gauge ``<p> = <p*>`` (:func:`pressure_error`) it is the beta of the pressure
    of a run that has met its target."""
    wJ = seq.quad.w * seq.jacobian_j
    return float(jnp.sum(wJ * p_q)) / (0.5 * float(seq.odd.l2_norm_sq(B, 2)))


def pressure_error(seq, p_w, s_q, p_q, bins):
    """``(err, constant)``: the pressure against the target values ``p_q`` on ``bins`` bins uniform in ``sqrt(s)``.

    In a fixed boundary the pressure is a multiplier defined up to a constant, and no shell move changes it (their
    sum, a uniform compression, would move the wall). Its constant is fixed by the gauge ``<p> = <p*>`` (volume means),
    with which the shells' ``K - 1`` zero-sum rows and the gauge make all ``K`` moments of ``p - p*`` vanish at an
    equilibrium. The pressure is then ``p_w + C`` with ``p_w`` the weak pressure (a 0-form
    that vanishes on the wall) and ``C = <p* - p_w>``. ``err`` is the volume-weighted error of ``p_w + C`` over the
    target's norm, ``constant`` is ``C`` over the target's maximum."""
    wJ = seq.quad.w * seq.jacobian_j
    b = jnp.minimum((jnp.sqrt(s_q) * bins).astype(jnp.int32), bins - 1)       # s = 1 belongs to the last bin
    V = jnp.zeros(bins).at[b].add(wJ)
    pw_q = seq.even.evaluate_at_quadrature(p_w, 0)[:, 0]
    e = (jnp.zeros(bins).at[b].add(wJ * (pw_q - p_q))) / V
    t = (jnp.zeros(bins).at[b].add(wJ * p_q)) / V
    C = -jnp.sum(V * e) / jnp.sum(V)                     # <p* - p_w>
    return (float(jnp.sqrt(jnp.sum(V * (e + C) ** 2) / jnp.sum(V * t ** 2))),
            float(C / jnp.max(jnp.abs(t))))
