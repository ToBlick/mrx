"""Prescribed pressure: the volumes of the flux shells as unknowns of the Newton step.

The incompressible relaxation keeps the volume of every flux shell, and the pressure it ends with is the multiplier
of those volumes. To prescribe a pressure ``p*(s)`` the shells must change their volumes, and only their volumes.
The velocity of a step is then

    u = curl a + sum_k c_k xi_k,        div xi_k ~ phi_k(s) - <phi_k>,

with ``phi_k`` the splines of the profile space (:func:`profile_basis`) and ``s`` the flux label of the field
(:mod:`mrx.flux_label`). The profile space is its own: cubic B-splines in ``s`` on ``cells`` cells whose breakpoints
are uniform in ``sqrt(s)``. It does not refer to the mesh, which has only to resolve it. A spline in ``s`` is a smooth
function in space next to the magnetic axis, where ``s`` goes as the squared distance, so every profile of the space
is regular there, ``p = p0 + p1 s + ...``. A spline in ``sqrt(s)`` would be a cone at the axis.

The space of these velocities depends only on the divergences ``f_k`` of the ``xi_k``, the projections of
``phi_k - <phi_k>`` onto the 3-forms, since two velocities with one divergence differ by a divergence-free one, which
``curl a`` already holds. Each ``xi_k`` is the gradient with that divergence, ``xi_k = grad chi_k`` with
``div grad chi_k = f_k`` and ``xi_k . n = 0`` on the wall (:func:`shell_gradients`): the 2-form part of the mixed
Poisson problem of the 3-forms, the solve with which the Leray projection forms the gradient of the pressure. It is
orthogonal in L2 to the divergence-free velocities, so the shells add to ``curl a`` only what changes the volumes, and it
refers to nothing but the domain and the label. The energy is the magnetic energy less the work of the target
pressure carried with the shells (gamma = 0),

    W = int B^2/2 dV - int p*(s) dV,

whose derivative along ``xi_k`` at a relaxed field is ``int (p - p*) f_k dxi``, with ``p`` the multiplier of the
divergence of the relaxation's 2-form velocity. The Newton step (:func:`mrx.relaxation.newton.target_direction`)
solves the Gauss-Newton system of ``W`` on this enlarged velocity space. The pressure term's own second variation is
left out: it is the term that makes the step indefinite on compressible velocities (finite gamma), and on the
shells the magnetic part alone is a good model.

A :class:`Shells` holds the frozen part of one step: the shell velocities, the pressure part of the right-hand side
and the inverse of the shell block. It depends on the flux label of the field, so
:func:`build_shells` rebuilds it every few steps (the relaxation's chunk). The splines sum to one, so the sum of the
shell divergences is zero: the shells kept are the combinations whose coefficients sum to zero, since a uniform
pressure cannot move a fixed wall. The pressure's constant, which no move changes, is the gauge ``<p> = <p*>``
(:func:`pressure_error`): with it all ``K`` moments of ``p - p*`` vanish at an equilibrium.

The target is a profile ``p*(s)`` (mu0 p in T^2, from the geometry file or a VMEC input namelist,
:func:`load_target`), made constant across seeded island chains (:func:`plateau_intervals`, :func:`flattened`). A
profile that an outer loop sets from the field, such as a fit of pressure data (``scripts/pressure_fit.py``), takes
its place between chunks (:func:`mrx.relaxation.loop.relax`). The labels of points in space on a field are
:func:`point_labels`, after :func:`logical_points` once.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.flux_label import enclosed_flux, enclosed_flux_map, flux_label
from mrx.operators import _hodge_split
from mrx.precision import to_working
from mrx.relaxation.newton import atom_profiles, parallel_penalty_profile, second_variation
from mrx.spline_bases import SplineBasis, knot_vector


#: The degree of the splines of the profile space (:func:`profile_basis`).
PROFILE_DEGREE = 3
#: The default number of cells of the profile space.
PROFILE_CELLS = 16
#: The number of bins uniform in ``sqrt(s)`` on which :func:`pressure_error` compares the pressure with its target.
ERROR_BINS = 48

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


def profile_knots(n):
    """The knot vector of the ``n`` splines of the profile space: cubic, clamped, on ``n - 3`` cells of ``s`` whose
    breakpoints are uniform in ``sqrt(s)``, ``s_i = (i / cells)^2``. A float64 numpy array."""
    cells = n - PROFILE_DEGREE
    return knot_vector((np.arange(cells + 1) / cells) ** 2, PROFILE_DEGREE, False)


def profile_basis(n):
    """The ``n`` splines of the profile space in ``s`` (:func:`profile_knots`). They sum to one. Every profile of the
    pressure, every shell and every slip of the current is a combination of them, so the controls are functions of the
    flux label alone, the same on every mesh. ``n`` is the number of cells plus three."""
    return SplineBasis(n, PROFILE_DEGREE, "clamped", jnp.asarray(profile_knots(n)))


def shell_table(n, s_q):
    """``(values, indices)``, each ``(n_q, 4)``: the splines of the profile space (``n`` of them) that can be nonzero
    at the labels ``s_q``."""
    return jax.vmap(profile_basis(n).evaluate_local)(s_q)


def zero_total_t(y, e, m):
    """The dual 3-form ``y`` less the multiple of ``m`` (the integrals of the 3-form basis functions) that makes it
    vanish on ``e`` (the 3-form of the constant 1). For ``y = load(g)`` it is ``M_3 f`` with ``f`` the projection of
    ``g - <g>`` onto the 3-forms, ``<g>`` the mean with ``e`` as the volume element. Only a 3-form of zero integral is
    the divergence of a velocity with ``u . n = 0``."""
    return y - m * (e @ y) / (m @ e)


@eqx.filter_jit
def shell_gradients(seq, s, Q, e, m):
    """``(Xi, info)``: the gradient velocities ``xi_j`` (rows of ``Xi``, even 2-forms) of the combinations of the
    shells in the columns of ``Q`` ``(K, n)``, at the label ``s`` (quadrature points), and the signed counts of their
    solves.

    The combination ``g = sum_k Q_kj phi_k`` has the divergence ``f_j``, the projection of ``g - <g>`` onto the
    3-forms (``<g>`` the mean with ``e``, the 3-form of the constant 1, as the volume element). ``xi_j`` is the 2-form
    part ``M_2^-1 D_2^T chi_j`` of the mixed Poisson problem ``L_3 chi_j = f_j`` (:func:`mrx.operators._hodge_split`,
    the solve of the Leray projection), so ``div xi_j = f_j`` to the tolerance of the sequence, ``xi_j . n = 0`` on
    the wall and ``xi_j`` is orthogonal in ``M_2`` to the divergence-free 2-forms. Its right-hand side ``M_3 f_j`` is
    ``load(g)`` less ``m`` times the integral of ``g`` against ``e`` (:func:`zero_total_t`), with no mass solve."""
    table = shell_table(Q.shape[0], s)
    even = seq.even

    def one(q):
        values, indices = table
        rhs = zero_total_t(even._scalar_load_values(jnp.sum(values * q[indices], axis=1), 3), e, m)
        _, xi, info = _hodge_split(even, rhs, 3, None)
        return to_working(xi), info
    return jax.lax.map(one, Q.T)


class Shells(eqx.Module):
    """The frozen part of the shell-volume Newton step: the gradient velocities ``Xi`` (:func:`shell_gradients`) of
    the orthonormal combinations of the ``K`` splines of the profile space whose coefficients sum to zero (on which
    the block is positive definite, a singular block would make the step's preconditioner singular), the pressure
    part ``g_p`` of the right-hand side and ``D_inv``, the inverse of the Gauss-Newton block of the shell velocities.
    It holds one velocity per shell, ``K - 1`` 2-forms. Arrays only, so a rebuild does not recompile the step."""
    Xi: jnp.ndarray
    g_p: jnp.ndarray
    D_inv: jnp.ndarray

    def velocity(self, c):
        """The velocity ``sum_j c_j xi_j`` of the shell coefficients ``c``."""
        return c @ self.Xi

    def velocity_t(self, y):
        """The transpose of :meth:`velocity`, ``(xi_j . y)_j`` for a dual 2-form ``y``."""
        return self.Xi @ y


@eqx.filter_jit
def _shell_parts(seq, B, Xi, p_q, kappa, mass_tol):
    # the target's work along xi_j, int p* div xi_j dV, with the divergence of the xi_j themselves
    g_p = Xi @ (seq.even.G[2].T @ seq.even._scalar_load_values(p_q, 3))
    penalty = parallel_penalty_profile(atom_profiles(seq, B), kappa)
    Hs = second_variation(seq, B, None, penalty, mass_tol)
    D = jax.lax.map(lambda xi: Xi @ Hs(xi), Xi)
    return g_p, 0.5 * (D + D.T)


def build_shells(seq, B, s_q, p_q, kappa, mass_tol, cells=PROFILE_CELLS):
    """``(shells, info)``: the :class:`Shells` at the field ``B``, with the flux label ``s_q`` and the target ``p_q``
    (mu0 p) at the quadrature points, the penalty ``kappa`` and mass-solve tolerance of the Newton operator, on the
    profile space of ``cells`` cells, and the signed counts of the ``K - 1`` Poisson solves of
    :func:`shell_gradients`. Those solves are most of the cost, 200 to 420 conjugate gradients each, then one
    Hessian apply per shell for the block. The pressure part is ``int p* div xi_k dV``, the target's work along
    ``xi_k``."""
    even = seq.even
    K = cells + PROFILE_DEGREE
    Q = jnp.asarray(np.linalg.qr(np.eye(K) - 1.0 / K)[0][:, :K - 1])
    m = even._scalar_load_values(jnp.ones_like(s_q), 3)          # the integrals of the basis functions
    e = even.M[3].solve(m)
    Xi, info = shell_gradients(seq, s_q, Q, e, m)
    g_p, D = _shell_parts(seq, B, Xi, p_q, kappa, mass_tol)
    # inverted on the host in float64, as the polar Gram blocks are: the block's condition (2e6 on li383 graded
    # (16,32,12)) is beyond a float32 inverse, which came out indefinite and stopped MINRES on nonpositive curvature
    return Shells(Xi, g_p, jnp.asarray(np.linalg.inv(np.asarray(D, dtype=np.float64)))), info


def plateau_intervals(rows, margin=1.0):
    """The intervals of ``s`` that seeded island chains occupy, from the seeding's estimate (the rows of
    :func:`mrx.relaxation.seeding.energy_seed`: the resonant surface ``x = sqrt(s)`` and the estimated full width
    ``w`` in ``x``): ``sqrt(s)`` in ``x +- margin w / 2``, overlapping intervals merged. Ideal moves keep the toroidal
    flux of every surface, so the intervals stay where the chains are."""
    iv = sorted((max(row["x"] - 0.5 * margin * row["w"], 0.0) ** 2, min(row["x"] + 0.5 * margin * row["w"], 1.0) ** 2)
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


def point_labels(seq, B, T, x):
    """The flux labels ``s`` at the logical points ``x`` (``(n, 3)``, from :func:`logical_points`) for the field
    ``B`` with the label temperature ``T`` (:func:`label`): the toroidal flux fraction inside the level set of ``T``
    through each point. The points stay, the labels move with the surfaces."""
    B_jk = seq.odd.evaluate_at_quadrature(B, 2)
    T_x = jax.vmap(DiscreteFunction(T, seq.basis_0, seq.even.E(0)))(x)[:, 0]
    return enclosed_flux_map(seq, T, B_jk)(T_x)


def load_target(seq, spec, scale=1.0):
    """The target profile ``p*(s)`` of ``spec``, mu0 p in T^2 as a function of the normalised toroidal flux times
    ``scale``: ``"file"`` (the geometry file's pressure) or a VMEC input namelist (its pressure profile,
    :meth:`mrx.equilibria.indata.Indata.pressure`). The constant of a profile is removed so that it vanishes on the wall
    (a uniform pressure does no work on a fixed wall)."""
    from mrx.equilibria.indata import read_indata
    s = np.linspace(0.0, 1.0, 2001)
    if spec == "file":
        p_r = seq.equilibrium["profiles"]["pressure"]               # Pa, a function of r = sqrt(s)
        values = jnp.asarray(scale * 4e-7 * np.pi * (p_r(np.sqrt(s)) - float(p_r(1.0))))
    else:
        indata = read_indata(spec)                                  # mu0 p in T^2
        values = jnp.asarray(scale * (indata.pressure(s) - indata.pressure(1.0)))
    return lambda x: jnp.interp(x, jnp.asarray(s), values)


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
