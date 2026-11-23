"""Motion of the mesh: its axis onto the magnetic axis, and its toroidal coordinate.

A mesh filled from the boundary alone (:mod:`mrx.equilibria.fill`) or read from a file has its axis ``r = 0``, the
pole, where the fill's quality guess or the file put it, not on the magnetic axis of the field it carries, and the
relaxation moves the magnetic axis further, by the Shafranov shift of the pressure among others. The field next to an
axis off the pole is resolved poorly: the polar core's transform drifts step by step. So the pole is kept on the
magnetic axis. The surfaces ``r = const`` between the axis and the wall are not moved towards the flux surfaces. The
wall stays where it is.

Every point moves within its surface ``zeta = const``: a move is a change of ``X1``, ``X2`` of the
:class:`~mrx.equilibria.FrameMap` (``R`` and ``Z`` in the cylindrical frame), and ``X3`` follows the surface to first
order (:func:`tangent`, zero on planar sections). One move of the axis (:func:`axis_move`) of the field ``B`` on the
current map:

1. The magnetic axis (:func:`magnetic_axis`), the field line that closes after one field period, as a whole curve: a
   Fourier series in ``zeta`` of its point in the chart ``(r cos 2 pi theta, r sin 2 pi theta)`` of each logical plane,
   whose Galerkin residual of the field-line equation vanishes.
2. Its in-plane distance ``d(zeta)`` from the pole, projected in L2 onto the ``zeta`` splines
   (:func:`axis_coefficients`), moves every point by ``d(zeta) (1 - r^2)`` (:func:`axis_step`): the pole goes onto the
   magnetic axis, the wall
   stays. The move is halved until no cell loses more than half its volume (:data:`MIN_DET_RATIO` of ``det DPhi``).
3. The physical field is carried onto the moved map (:func:`transfer`) by the relaxation's own induction with the
   mesh's velocity, which keeps ``div B = 0``, ``B . n = 0`` and the toroidal flux exactly. Kept coefficients would
   move the field with the mesh instead, an ideal transport that kicks the equilibrium: on W7-X P_M_p2 a move of 0.19
   first cells raised the force residual from 7e-10 to 5.2e-5 and took 33 steps back below 1e-8, the transfer 5.9e-8
   and 2 steps, to the same end state (2026-11-21).
4. The geometry and the harmonic forms of the new map are installed (:func:`install_map`), the preconditioners of
   the old map are kept. The map is data of the sequence, so the compiled relaxation is reused.

``scripts/relax.py --mesh.align`` moves the pole at a chunk boundary where the field is at an equilibrium and its
axis is off the pole by more than a tolerance, and builds the preconditioners again when the Newton solve says that
those of an older map no longer fit. A run from the vacuum field of a namelist starts on a
mesh whose pole is on the vacuum field's axis (:func:`vacuum_start`): there the mesh moves first and the vacuum field
is solved on it, so the first move, the largest, perturbs nothing. A moved map goes into the checkpoints
(:func:`write_map`, :func:`install_checkpoint_map`).

The toroidal coordinate is moved from another misfit: ``q``, with ``zeta + q`` the harmonic angle of the map, the
potential of its vacuum field (:func:`mrx.nullspace.harmonic_angle`). Every point slides along its line of constant
``r`` and ``theta`` by the Newton step of ``q = 0`` (:func:`harmonic_displacement`), and the moves repeat on the moved
map until ``zeta`` is harmonic (:func:`harmonic_build`). ``--geometry.zeta harmonic`` does this to the first map,
before the field exists, so a move is only a change of the map.

A third choice of the toroidal coordinate is a cut, not a move: the planes normal to the axis of the map
(:func:`axis_normal_map`), in the frame of the axis, GVEC's G-frame, the default of
:func:`mrx.geometry.build_sequence` for every file. Every point slides along its line of constant ``r`` and
``theta`` into the plane of its ``zeta`` at once, by Newton's method along the line, and the map is in the new frame
with ``X3 = 0``. The shift of ``zeta`` the cut returns carries a field given in the old logical coordinates, a file's
field above all.

The moves keep the preconditioners of the map they start from (:func:`install_map`). Those of the final map are built
by :func:`mrx.geometry.build_sequence` after the harmonic moves, by :func:`vacuum_start` after its moves, and by
``scripts/relax.py`` when the Newton solve asks for them.
"""
import time
from functools import partial

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.diagnostics.poincare import logical_field, to_polar
from mrx.differential_forms import det33, inv33
from mrx.equilibria import FrameMap, frame_derivatives, polar_projection
from mrx.geometry import SequenceGeometry, map_jacobian_at
from mrx.nullspace import compute_nullspaces
from mrx.optimization.shape_ad import frame_geometry
from mrx.precision import DTYPE, to_working
from mrx.relaxation.loop import CFL, logical_cfl_weights
from mrx.relaxation.physics import electric_field
from mrx.symmetry import stellarator_symmetric_scalar, stellarator_symmetry_defect

#: The smallest ratio of ``det DPhi`` after and before a move at any quadrature point.
MIN_DET_RATIO = 0.5
#: The Fourier modes per spline of ``zeta`` of the axis curve of :func:`magnetic_axis`.
AXIS_MODES_PER_SPLINE = 1
#: The largest number of Newton steps of :func:`magnetic_axis`, and the size of a Newton step, in first radial cells
#: of the logical chart, below which it stops.
AXIS_NEWTON_STEPS, AXIS_NEWTON_TOL = 20, 1e-10
#: The distance |grad zeta - h| / |h| (:func:`mrx.nullspace.zeta_harmonic_distance`) below which the moves of
#: :func:`harmonic_build` stop. From the planes phi = const of the boundary meshes it fell 0.29 -> 1.6e-2 -> 1.6e-5 ->
#: 2.5e-7 (li383 (48,64,24)), 0.35 -> 3.1e-2 -> 9.5e-5 -> 2.3e-7 (W7-X (24,48,28)), 0.65 -> 9.3e-2 -> 2.4e-3 -> 5.3e-6
#: (HSX (32,64,36)), three moves each (2026-11-19).
HARMONIC_TOL = 1e-5

def extended(Phi, x):
    """The map ``Phi`` at a logical point whose ``zeta`` may leave ``[0, 1)``: one field period on is the point turned
    by ``2 pi / nfp`` about the ``z`` axis."""
    k = jnp.floor(x[2])
    c, s = jnp.cos(2.0 * jnp.pi * k / Phi.nfp), jnp.sin(2.0 * jnp.pi * k / Phi.nfp)
    P = Phi(x.at[2].add(-k))
    return jnp.array([c * P[0] - s * P[1], s * P[0] + c * P[1], P[2]])


def _planes(seq):
    """The Greville points of the ``zeta`` splines, the planes on which the axis offset is recorded."""
    return np.asarray(seq.basis_0.Lambda[2].greville_points(), dtype=np.float64)


def first_cell(seq, Phi, zetas):
    """The mean distance in the planes ``zetas`` from the axis to the first radial breakpoint, the size of the
    innermost radial cell."""
    r1 = _first_knot(seq)
    th = np.arange(16) / 16
    x = jnp.asarray(np.stack(np.broadcast_arrays(r1, th[None, :], zetas[:, None]), -1).reshape(-1, 3))
    ax = jnp.asarray(np.stack(np.broadcast_arrays(0.0, 0.0, zetas[:, None]), -1).reshape(-1, 3))
    S = jax.vmap(Phi.section)
    return float(jnp.mean(jnp.linalg.norm(S(x).reshape(len(zetas), 16, 2) - S(ax).reshape(len(zetas), 1, 2), axis=-1)))


def _first_knot(seq):
    """The first radial breakpoint of ``seq`` after the axis: the logical size of the first radial cell."""
    T_r = np.asarray(seq.basis_0.Lambda[0].T, dtype=np.float64)
    return float(np.unique(T_r[(T_r > 0) & (T_r <= 1)])[0])


def _series(modes, z):
    """The values and the ``zeta`` derivatives ``(len(z), 2, modes + 1)`` of ``cos(2 pi k zeta)`` and
    ``sin(2 pi k zeta)``, ``k = 0 .. modes``, at ``z``."""
    k = 2.0 * np.pi * np.arange(modes + 1)
    a = np.outer(z, k)
    return (np.stack([np.cos(a), np.sin(a)], 1), np.stack([-k * np.sin(a), k * np.cos(a)], 1))


def _zeta_quadrature(seq, modes):
    """Gauss points and weights in ``zeta`` over one field period, on every cell of the ``zeta`` splines, enough of
    them for the integrand of a spline field along a curve of ``modes`` Fourier modes."""
    bp = np.unique(np.asarray(seq.basis_0.Lambda[2].T, dtype=np.float64).clip(0.0, 1.0))
    g = seq.p + 1 + 4 * int(np.ceil(modes / (bp.size - 1)))
    x, w = np.polynomial.legendre.leggauss(g)
    a, b = bp[:-1, None], bp[1:, None]
    return (0.5 * (b - a) * x + 0.5 * (a + b)).ravel(), (0.5 * (b - a) * w).ravel()


def _axis_free(seq, modes):
    """``(free, test)``: the flat indices into ``(2, 2, modes + 1)`` (component ``u``, ``v`` of the chart, cosine or
    sine, mode) of the free coefficients of an axis curve and of the test functions of its residual. On a
    stellarator-symmetric sequence ``u`` is even in ``zeta`` and ``v`` odd, and the residual ``y' - f`` the other way
    round, so its ``u`` part is tested with the sines, its ``v`` part with the cosines: the other parts vanish."""
    free = np.zeros((2, 2, modes + 1), dtype=bool)
    if seq.symmetry != "stellarator":
        free[:, 0, :], free[:, 1, 1:] = True, True
        return np.flatnonzero(free), np.flatnonzero(free)
    test = free.copy()
    free[0, 0, :], free[1, 1, 1:] = True, True
    test[0, 1, 1:], test[1, 0, :] = True, True
    return np.flatnonzero(free), np.flatnonzero(test)


def _axis_residual(field, coef, x, idx, tidx, z, w, V, dV):
    """The Galerkin residual of :func:`magnetic_axis` at the free coefficients ``x`` (flat indices ``idx``), tested
    with the functions of the flat indices ``tidx``. The integrand is ``Bt^zeta y' - (Bt^u, Bt^v)``, ``Bt`` the
    field's density in the chart ``(u, v, zeta)``: the 2-form's logical components ``Bh`` carried to the chart and
    divided by its Jacobian ``2 pi r``, finite and nonzero on the pole, where ``Bh`` vanishes with ``J``."""
    c = jnp.zeros(2 * V.shape[1] * V.shape[2], dtype=x.dtype).at[idx].set(x).reshape(2, V.shape[1], V.shape[2])
    y, dy = jnp.einsum("ckn,qkn->qc", c, V), jnp.einsum("ckn,qkn->qc", c, dV)

    def density(zq, yq, dyq):
        r, theta = to_polar(yq)
        b = field.at(jnp.array([r, theta, zq % 1.0]), coef)
        cs, sn = jnp.cos(2.0 * jnp.pi * theta), jnp.sin(2.0 * jnp.pi * theta)
        uv = jnp.array([cs * b[0] - 2.0 * jnp.pi * r * sn * b[1], sn * b[0] + 2.0 * jnp.pi * r * cs * b[1]])
        return (b[2] * dyq - uv) / (2.0 * jnp.pi * r)
    return jnp.einsum("q,qkn,qc->ckn", w, V, jax.vmap(density)(z, y, dy)).reshape(-1)[tidx]


@partial(jax.jit, static_argnames=("field",))
def _axis_newton(field, coef, x, idx, tidx, z, w, V, dV):
    """The Newton step of :func:`magnetic_axis` at ``x`` and the norm of the residual there."""
    F = partial(_axis_residual, field, coef, idx=idx, tidx=tidx, z=z, w=w, V=V, dV=dV)
    R = F(x)
    return -jnp.linalg.solve(jax.jacfwd(F)(x), R), jnp.linalg.norm(R)


@partial(jax.jit, static_argnames=("field",))
def _axis_residual_norm(field, coef, x, idx, tidx, z, w, V, dV):
    return jnp.linalg.norm(_axis_residual(field, coef, x, idx, tidx, z, w, V, dV))


def magnetic_axis(seq, B, modes=None):
    """``(c, info)``: the magnetic axis of the 2-form ``B`` on the installed map, the field line that closes after one
    field period, as a curve in logical coordinates.

    The curve is ``y(zeta) = sum_k c[:, 0, k] cos(2 pi k zeta) + c[:, 1, k] sin(2 pi k zeta)``, ``k = 0 .. modes``,
    in the chart ``y = (r cos 2 pi theta, r sin 2 pi theta)`` of each logical plane ``zeta``, which is smooth at the
    pole (``modes`` :data:`AXIS_MODES_PER_SPLINE` per spline of ``zeta`` unless given). On a stellarator-symmetric
    sequence its first component is a cosine series, its second a sine series. The field line satisfies
    ``Bt^zeta dy/dzeta = (Bt^u, Bt^v)``, ``Bt`` the field's density in the chart (:func:`_axis_residual`), with no
    division by the field's values and no bound on ``r``. ``c`` makes the Galerkin residual
    ``int_0^1 phi(zeta) (Bt^zeta dy/dzeta - (Bt^u, Bt^v))(zeta, y(zeta)) dzeta`` vanish for every ``phi`` of the series (on a symmetric
    sequence those of the residual's parity, the other way round from the curve's), integrated by Gauss points on the cells of the ``zeta`` splines: the residual is weak in ``zeta``, and the whole
    curve is found at once. Newton's method with the exact Jacobian, from the curve half a first radial cell off the
    pole (the chart's derivatives do not exist at the pole itself), each step halved until it lowers the norm of the
    residual, until a step is below :data:`AXIS_NEWTON_TOL` first radial cells or after :data:`AXIS_NEWTON_STEPS`
    steps. The solution is the axis of an elliptic field line,
    whose period map has no eigenvalue 1.

    ``info`` holds the number of modes, the Newton steps, the halvings of all steps, the norms of the residual at the
    start and the end, the last step in first radial cells and the wall time."""
    t0 = time.perf_counter()
    modes = AXIS_MODES_PER_SPLINE * seq.basis_0.Lambda[2].n if modes is None else int(modes)
    field = logical_field(seq, B)
    coef = field.coefficients(B)
    z, w = _zeta_quadrature(seq, modes)
    V, dV = _series(modes, z)
    idx, tidx = _axis_free(seq, modes)
    r1 = _first_knot(seq)
    x = jnp.zeros(idx.size, dtype=DTYPE).at[0].set(0.5 * r1)        # idx[0] is the mean of u
    args = (jnp.asarray(idx), jnp.asarray(tidx), jnp.asarray(z, dtype=DTYPE), jnp.asarray(w, dtype=DTYPE), jnp.asarray(V, dtype=DTYPE),
            jnp.asarray(dV, dtype=DTYPE))
    resid, halved = [], 0
    for k in range(AXIS_NEWTON_STEPS):
        dx, R = _axis_newton(field, coef, x, *args)
        resid.append(float(R))
        step = float(jnp.max(jnp.abs(dx))) / r1
        if step < AXIS_NEWTON_TOL:
            x = x + dx
            break
        # a full step far from the axis may leave the domain, where the field-line equation is not defined
        while (float(jnp.max(jnp.abs(dx))) / r1 > AXIS_NEWTON_TOL
               and not float(_axis_residual_norm(field, coef, x + dx, *args)) < resid[-1]):
            dx, halved = 0.5 * dx, halved + 1
        x = x + dx
    c = np.zeros(2 * 2 * (modes + 1))
    c[idx] = np.asarray(x)
    return c.reshape(2, 2, modes + 1), dict(modes=modes, newton=k + 1, halved=halved, resid_start=resid[0],
                                            resid_end=resid[-1], step_cells=step, seconds=time.perf_counter() - t0)


def axis_points(c, zetas):
    """The logical points ``(len(zetas), 3)`` of the axis curve ``c`` of :func:`magnetic_axis` on the planes
    ``zetas``."""
    zetas = np.asarray(zetas, dtype=np.float64)
    y = np.einsum("ckn,qkn->qc", c, _series(c.shape[2] - 1, zetas)[0])
    return np.stack([np.hypot(y[:, 0], y[:, 1]), np.arctan2(y[:, 1], y[:, 0]) / (2.0 * np.pi) % 1.0, zetas], 1)


def _axis_profile(seq):
    """The radial spline coefficients of ``1 - r^2``, interpolated (exactly for degree 2 and up). The first is 1."""
    br = seq.basis_0.Lambda[0]
    gr = np.asarray(br.greville_points(), dtype=np.float64)
    return np.linalg.solve(np.asarray(br.collocation_matrix(), dtype=np.float64), 1.0 - gr ** 2)


def axis_step(seq, dz):
    """The coefficients ``(2, n_r, n_theta, n_zeta)`` of ``X1``, ``X2`` that move every point by ``d(zeta) (1 - r^2)``,
    ``d`` the in-plane displacement with the ``zeta`` spline coefficients ``dz`` ``(n_z, 2)``. Constant in ``theta``,
    so polar-regular. On a stellarator-symmetric sequence ``X1``'s part is projected on the even, ``X2``'s on the odd
    functions."""
    n_t = seq.basis_0.Lambda[1].n
    raw = jnp.asarray(np.einsum("i,j,kc->cijk", _axis_profile(seq), np.ones(n_t), dz), dtype=seq.map.raw_X1.dtype)
    if seq.symmetry == "stellarator":
        raw = jnp.stack([stellarator_symmetric_scalar(raw[0], seq.basis_0, even=True),
                         stellarator_symmetric_scalar(raw[1], seq.basis_0, even=False)])
    return raw


def axis_coefficients(seq, c):
    """The ``zeta`` spline coefficients ``(n_z, 2)`` of the in-plane displacement from the pole to the axis curve
    ``c`` of :func:`magnetic_axis`: its L2 projection onto the ``zeta`` splines, one mass solve of the splines with the
    displacement at Gauss points of their cells."""
    bz = seq.basis_0.Lambda[2]
    z, w = _zeta_quadrature(seq, c.shape[2] - 1)
    N = np.asarray(bz.collocation_matrix(jnp.asarray(z)), dtype=np.float64)              # (n_q, n_z)
    S = jax.vmap(seq.map.section)
    pole = jnp.asarray(np.stack(np.broadcast_arrays(0.0, 0.0, z[:, None]), -1).reshape(-1, 3))
    d = np.asarray(S(jnp.asarray(axis_points(c, z))) - S(pole), dtype=np.float64)
    return np.linalg.solve((N * w[:, None]).T @ N, (N * w[:, None]).T @ d)


def moved(Phi, delta):
    """The map ``Phi`` with ``delta`` ``(3, ...)`` added to the coefficients of ``X1``, ``X2`` and ``X3``."""
    return Phi.with_coordinates(Phi.raw_X1 + delta[0], Phi.raw_X2 + delta[1], Phi.raw_X3 + delta[2])


@eqx.filter_jit
def tangent(seq, delta):
    """``delta`` ``(2, ...)``, a change of ``X1`` and ``X2``, completed by the change of ``X3`` that keeps every point
    on its surface ``zeta = const`` to first order: ``d_r X3 xi_r + d_theta X3 xi_theta`` with ``xi`` the logical
    displacement of the in-plane change, projected onto the odd Dirichlet 0-forms. Zero on a map whose surfaces
    ``zeta = const`` are planes (``X3 = 0``)."""
    Phi, odd = seq.map, seq.odd
    D = jax.vmap(jax.jacfwd(lambda y: jnp.array([Phi.basis.contract(c, y) for c in (Phi.raw_X1, Phi.raw_X2,
                                                                                     Phi.raw_X3)])))(seq.quad.x)
    d = _section_values(seq, Phi, jnp.concatenate([delta, jnp.zeros_like(delta[:1])]))
    xi = jnp.linalg.solve(D[:, :2, :2], d[..., None])[..., 0]
    dX3 = jnp.sum(D[:, 2, :2] * xi, axis=1)
    c = odd.M[0].solve(odd._scalar_load_values(dX3, 0))
    return jnp.concatenate([delta, (odd.E(0).T @ c).reshape((1,) + delta.shape[1:])])


@eqx.filter_jit
def _section_values(seq, Phi, delta):
    """The change ``delta`` ``(3, ...)`` of ``X1``, ``X2`` at the quadrature points, ``(n_q, 2)``."""
    D = Phi.with_coordinates(delta[0], delta[1], delta[2])
    return jax.vmap(D.section)(seq.quad.x)


@eqx.filter_jit
def _det_ratio(seq, Phi, reference=None):
    """``det DPhi`` of ``Phi`` over that of the installed map at the quadrature points, or over ``reference``, the
    ``det DPhi`` of another map there."""
    return frame_geometry(seq, Phi).jacobian_j / (seq.jacobian_j if reference is None else reference)


def _backtrack(seq, base, delta, reference=None):
    """``(lam, ratio)``: the largest ``lam`` of ``1, 1/2, ..., 2^-8`` for which no cell of ``base + lam delta``
    loses more than :data:`MIN_DET_RATIO` of its volume, against the installed map or the ``det DPhi`` ``reference``
    of another, or 0, and the ratio of ``det DPhi`` there."""
    lam = 1.0
    for _ in range(9):
        ratio = _det_ratio(seq, moved(seq.map, base + lam * delta), reference)
        if float(jnp.min(ratio)) >= MIN_DET_RATIO:
            return lam, ratio
        lam *= 0.5
    return 0.0, _det_ratio(seq, moved(seq.map, base), reference)


def install_map(seq, Phi, keep_preconditioners=False):
    """Install the map ``Phi`` on ``seq`` with its harmonic forms, and with its own preconditioners unless
    ``keep_preconditioners``. A move keeps those of the map before it. On the boundary meshes of li383 (48,64,24),
    W7-X (24,48,28) and HSX (32,64,36), after three, three and one moves of the whole mesh, the start map's
    preconditioners took the same MINRES and label CG counts as those of the moved map, but the mass solves inside
    every step took more iterations (the 1-form solve of the Newton operator 15, 15, 19 against 11, 11, 16), and a
    chunk of ten steps with both targets took 26, 23 and 12 % longer. A build costs 57-72 s on an H200 (host
    assembly), the price of 25 to 50 such chunks, so it is done only when the Newton solve asks for it (2026-11-19).
    A move of the pole alone changes the map less."""
    seq.set_geometry(frame_geometry(seq, Phi), keep_preconditioners)
    if not keep_preconditioners:
        seq.build_preconditioners()
    compute_nullspaces(seq, verbose=False)


def axis_sections(seq, c):
    """``(y, h_1)``: the in-plane coordinates ``y`` ``(n_zeta, 2)`` of the axis curve ``c`` of :func:`magnetic_axis`
    on the planes of :func:`_planes`, and the size ``h_1`` of the first radial cell (:func:`first_cell`)."""
    Phi, zetas = seq.map, _planes(seq)
    return np.asarray(jax.vmap(Phi.section)(jnp.asarray(axis_points(c, zetas)))), first_cell(seq, Phi, zetas)


def axis_offset(seq, c):
    """``(d, h_1)``: the in-plane displacement ``d`` ``(n_zeta, 2)`` from the pole to the axis curve ``c`` of
    :func:`magnetic_axis` on the planes of :func:`_planes`, and the size ``h_1`` of the first radial cell."""
    y, h1 = axis_sections(seq, c)
    zetas = _planes(seq)
    ax = jnp.asarray(np.stack(np.broadcast_arrays(0.0, 0.0, zetas[:, None]), -1).reshape(-1, 3))
    return y - np.asarray(jax.vmap(seq.map.section)(ax)), h1


def _displacement(seq, dz, d, h1, reference=None):
    """``(Phi_new, record)``: the map of the move of the pole of ``seq`` by the in-plane displacement with the ``zeta``
    spline coefficients ``dz`` (step 2 of the module docstring), halved against folds. ``d`` are the offsets on the
    planes of :func:`_planes`. ``record`` holds their largest (``axis_m`` in metres, ``axis_cells`` in first radial
    cells of size ``h1``), the backtracking factor ``lam``, the largest move in metres and the range of the ratio of
    ``det DPhi`` after and before, or after and of ``reference``, the ``det DPhi`` the volume bound holds against."""
    delta = tangent(seq, axis_step(seq, dz))
    lam, ratio = _backtrack(seq, jnp.zeros_like(delta), delta, reference)
    off = float(np.max(np.linalg.norm(d, axis=1)))
    return moved(seq.map, lam * delta), dict(
        axis_m=off, axis_cells=off / h1, h1_m=h1, lam=lam,
        move_m=float(jnp.max(jnp.linalg.norm(_section_values(seq, seq.map, lam * delta), axis=1))),
        det_ratio_min=float(jnp.min(ratio)), det_ratio_max=float(jnp.max(ratio)))


def axis_displacement(seq, B):
    """``(Phi_new, record)``: the map of one move of the pole of ``seq`` onto the magnetic axis of ``B``, steps 1 and 2
    of the module docstring. ``seq`` is not changed. ``record`` is that of :func:`_displacement` with the axis
    finder's record (``axis``)."""
    c, found = magnetic_axis(seq, B)
    Phi, record = _displacement(seq, axis_coefficients(seq, c), *axis_offset(seq, c))
    record.update(axis=found)
    return Phi, record


def axis_move(seq, B):
    """Move the pole of ``seq`` onto the magnetic axis of ``B`` (:func:`axis_displacement`), carry the field onto the
    new map (:func:`transfer`), install the map with the preconditioners kept and return ``(B', record)``. The record
    holds that of the move and of the transfer, the relative change of the energy, the relative change of the field's
    coefficients against keeping them (``dB_rel``, in the new map's norm) and the wall time."""
    t0 = time.perf_counter()
    Phi_new, record = axis_displacement(seq, B)
    E0 = float(seq.odd.l2_norm_sq(B, 2))
    B_new, carried = transfer(seq, B, Phi_new)
    install_map(seq, Phi_new, keep_preconditioners=True)
    record.update(carried, dE_rel=float(seq.odd.l2_norm_sq(B_new, 2)) / E0 - 1.0,
                  dB_rel=float(seq.odd.l2_norm(B_new - B, 2) / seq.odd.l2_norm(B_new, 2)),
                  wall=time.perf_counter() - t0)
    return B_new, record


def _start(seq, offset, recut):
    """Move the pole of ``seq`` in rounds onto the curve whose offsets ``offset(seq) = (dz, d, h_1, info)`` gives on the
    installed map (:func:`_displacement`), with no field carried, and return the record. The rounds repeat while the
    distance falls by half or more, or the last move was halved against a fold, so a move of any size is made in
    rounds. They stop when a move is halved to nothing. The volume bound of :data:`MIN_DET_RATIO` holds against the map
    the rounds start from, for the rounds together. Per round it compounds: on li383 (24,48,24) twenty rounds towards
    the vacuum's axis, halved from the first, left ``det DPhi`` at 2e-4 at the quadrature points, the Newton solves unconverged and the run at NaN (2026-11-23). What is left of the way is
    the moves' at the bar. With ``recut`` the moved map is cut again by the planes normal to its axis
    (:func:`axis_normal_map`), which the pole stays on. The final map is installed with its preconditioners and
    harmonic forms. The record holds the offset in first radial cells, the backtracking factor,
    the largest move, the smallest ratio of ``det DPhi`` to the start map's and ``info`` of every round, the offset
    on the final map and the wall time."""
    t0 = time.perf_counter()
    rounds = []
    # the volume bound of a move holds against the map the rounds start from: per round it would compound
    J0 = seq.jacobian_j
    while True:
        dz, d, h1, info = offset(seq)
        Phi, rec = _displacement(seq, dz, d, h1, J0)
        rounds.append(dict(axis_cells=rec["axis_cells"], lam=rec["lam"], move_m=rec["move_m"],
                           det_ratio_min=rec["det_ratio_min"], axis=info))
        print(f"[mesh] start round {len(rounds)}: axis {rec['axis_cells']:.3e} first cells off the pole, "
              f"move {rec['move_m']:.3e} m x{rec['lam']:g}, {info}", flush=True)
        # a move halved to nothing against a fold changes nothing: the next round would find the same offset
        if rec["lam"] == 0.0 or (len(rounds) > 1 and rounds[-2]["lam"] == 1.0
                                 and rec["axis_cells"] > 0.5 * rounds[-2]["axis_cells"]):
            break
        install_map(seq, Phi, keep_preconditioners=True)
    record = dict(rounds=rounds)
    if recut:
        Phi, _, record["cut"] = axis_normal_map(seq, seq.map)
        install_map(seq, Phi, keep_preconditioners=True)
    install_map(seq, seq.map)
    _, d, h1, _ = offset(seq)
    record.update(axis_cells=float(np.max(np.linalg.norm(d, axis=1))) / h1, seconds=time.perf_counter() - t0)
    return record


def vacuum_start(seq, vacuum, recut):
    """Move the pole of ``seq`` onto the axis of its vacuum field before a run starts from it, and return the record
    of :func:`_start`. ``vacuum(seq)`` is the vacuum field on the installed map: every round finds its axis
    (:func:`magnetic_axis`) on the map the round before made, with the vacuum field solved again there."""
    def offset(sq):
        c, found = magnetic_axis(sq, vacuum(sq))
        return (axis_coefficients(sq, c), *axis_offset(sq, c), found)
    return _start(seq, offset, recut)


def _angle_space(seq):
    """The free 0-forms on which the harmonic angle of ``seq`` is solved: those of the odd view on a half-period
    sequence, where ``zeta`` and ``q`` are odd."""
    return (seq.odd if seq.half_period else seq).free


@eqx.filter_jit
def _resection(seq, q, dq):
    """The change of ``X1``, ``X2``, ``X3`` ``(3, n_q)`` at the quadrature points when every point moves along its
    line of constant ``r`` and ``theta`` to the logical ``zeta - q / (1 + d_zeta q)`` of the installed map: the
    coordinates of the moved point in the frame of its own plane ``zeta`` less its present ones. ``q`` and
    ``dq = d_zeta q`` are given at the quadrature points. ``1 + dq > 0``: the harmonic angle increases along every
    ``zeta`` line of a map that does not fold."""
    Phi, x = seq.map, seq.quad.x
    o, e1, e2 = jax.vmap(Phi.plane)(x[:, 2])
    frame = jnp.stack([e1, e2, jnp.cross(e1, e2)], 1)

    def coordinates(y):
        return jnp.einsum("qij,qj->iq", frame, jax.vmap(lambda p: extended(Phi, p))(y) - o)
    return coordinates(x.at[:, 2].add(-q / (1.0 + dq))) - coordinates(x)


def harmonic_displacement(seq, q):
    """``(Phi_new, record)``: the map of one move of the mesh of ``seq`` towards the harmonic toroidal angle, from
    its misfit ``q``, the 0-form of :func:`_angle_space` with ``zeta + q`` the harmonic angle of the installed map
    (:func:`mrx.nullspace.harmonic_angle`). ``seq`` is not changed.

    Every point moves along its line of constant ``r`` and ``theta`` by the Newton step of ``q = 0`` at that point,
    the logical shift ``-q / (1 + d_zeta q)``, to the position the installed map has there (:func:`_resection`).
    The surfaces ``r = const`` and the poloidal angle stay, and the wall slides along itself. The change of ``X1``
    (even), ``X2`` and ``X3`` (odd) is the L2 projection onto the free 0-forms of the sequence, with the wall's values
    free. The move is halved until no cell folds
    (:func:`_backtrack`). ``record`` holds the largest shift in logical ``zeta``, the backtracking factor ``lam`` and
    the range of the ratio of ``det DPhi`` after and before."""
    free = _angle_space(seq)
    q_q = free.evaluate_at_quadrature(q, 0)[:, 0]
    dq_q = free.evaluate_at_quadrature(free.G[0] @ q, 1)[:, 2]
    dX = _resection(seq, q_q, dq_q)
    views = (seq.even.free, seq.odd.free, seq.odd.free) if seq.half_period else (seq.free,) * 3
    shape = seq.basis_0.shape[0]
    delta = jnp.stack([(v.E(0).T @ v.M[0].solve(v._scalar_load_values(d, 0))).reshape(shape)
                       for v, d in zip(views, dX)])
    lam, ratio = _backtrack(seq, jnp.zeros_like(delta), delta)
    return moved(seq.map, lam * delta), dict(shift_max=float(jnp.max(jnp.abs(q_q / (1.0 + dq_q)))), lam=lam,
                                             det_ratio_min=float(jnp.min(ratio)), det_ratio_max=float(jnp.max(ratio)))


def harmonic_build(seq):
    """Move the map installed on ``seq`` until its toroidal coordinate is harmonic, before a field exists, and keep in
    ``seq.equilibrium["harmonic"]`` the distance :func:`mrx.nullspace.zeta_harmonic_distance` before each move and at
    the end, the moves' backtracking factors and the wall time.

    Each round solves for the misfit ``q`` (one ``L_0`` solve), stops once the distance is below
    :data:`HARMONIC_TOL` or has not fallen by half, and otherwise installs :func:`harmonic_displacement`. Without a
    field a move is a change of the map and nothing is carried. The solves need the 0-form preconditioners only, which
    the caller builds before (:func:`mrx.geometry.build_sequence` builds them on the first map, the moves keep them,
    and all of them are built once on the final map). The phi map's preconditioners kept on the harmonic map for the
    relaxation slowed every step by 7-14 % (li383, W7-X, HSX boundary meshes, 2026-11-19)."""
    from mrx.nullspace import harmonic_angle, zeta_harmonic_distance
    t0 = time.perf_counter()
    rec = dict(drift=[], lam=[])
    while True:
        q = to_working(harmonic_angle(_angle_space(seq)))
        rec["drift"].append(zeta_harmonic_distance(seq, q)[0])
        if rec["drift"][-1] < HARMONIC_TOL or (len(rec["drift"]) > 1 and rec["drift"][-1] > 0.5 * rec["drift"][-2]):
            break
        Phi, r = harmonic_displacement(seq, q)
        seq.set_geometry(frame_geometry(seq, Phi), keep_preconditioners=True)
        rec["lam"].append(r["lam"])
    rec["seconds"] = time.perf_counter() - t0
    seq.equilibrium["harmonic"] = rec
    print(f"[geom] harmonic zeta in {len(rec['lam'])} moves: |grad zeta - h| / |h| "
          + " -> ".join(f"{d:.2e}" for d in rec["drift"]) + f", {rec['seconds']:.0f} s", flush=True)


#: The samples of the mesh axis per spline of ``zeta`` from which :func:`axis_normal_map` fits its curve. The curve's
#: series has half as many modes as the map has splines of ``zeta``, what the splines resolve: a spline axis also
#: carries the kinks of its pieces, at the cell frequency, and a series that keeps them gives the frame's vectors
#: wiggles there (HSX (24,48,48) with 48 modes: 1.8e-4 in ``N`` at mode 47, and a vacuum transform 3e-3 low, 2026-11-20).
AXIS_SAMPLES_PER_SPLINE = 4
#: The Newton steps of the cut of :func:`axis_normal_map` along every line ``(r, theta)``.
CUT_NEWTON_STEPS = 12


def _fourier_fit(zetas, values, n_modes):
    """``(cos, sin, rms)``: the least-squares series ``sum_k cos_k cos(-2 pi k zeta) + sin_k sin(-2 pi k zeta)``,
    ``k = 0 .. n_modes``, of the ``values`` ``(K, c)`` at the logical ``zetas``, ``(c, n_modes + 1)`` each, and the rms
    of its residual at the samples."""
    arg = -2.0 * np.pi * np.outer(zetas, np.arange(n_modes + 1))
    A = np.concatenate([np.cos(arg), np.sin(arg[:, 1:])], axis=1)
    coef = np.linalg.lstsq(A, values, rcond=None)[0]
    rms = float(np.sqrt(np.mean(np.sum((A @ coef - values) ** 2, axis=1))))
    sin = np.concatenate([np.zeros((1, values.shape[1])), coef[n_modes + 1:]])
    return coef[:n_modes + 1].T, sin.T, rms


def curve_frame(zetas, points, nfp, n_modes):
    """``(frame, rms)``: the frame of the planes normal to a closed curve, in the form of a state's ``frame``
    (:func:`mrx.equilibria.gvec.read_frame`), and the rms distance (in the map's units) of the curve's series from
    its samples.

    ``points`` ``(K, 3)`` are Cartesian points of the curve at the logical ``zetas`` of one field period. Turned back
    by ``2 pi zeta / nfp`` about the ``z`` axis they are periodic in ``zeta``, and their least-squares series with
    ``n_modes`` modes per field period is the frame's axis ``a``. Its unit tangent ``t`` in space is ``a'`` plus the
    turn ``(2 pi / nfp) e_z x a``. ``N`` is the radial unit vector ``e_R`` made normal to ``t``, ``B = N x t``, so
    ``N x B = -t``, the orientation of the cylindrical frame (``N = e_R``, ``B = e_z``), and ``X1``, ``X2`` are
    about ``R``, ``Z`` about the axis. ``N`` and ``B`` are the series with ``2 n_modes`` modes that interpolate them
    at ``4 n_modes + 1`` angles. A curve symmetric under ``(x, y, z) -> (x, -y, -z)`` at ``-zeta`` gives a frame with
    that symmetry, to round-off."""
    omega = 2.0 * np.pi / nfp
    c, s = np.cos(-omega * zetas), np.sin(-omega * zetas)
    p = np.asarray(points, dtype=np.float64)
    turned = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], 1)
    a_cos, a_sin, rms = _fourier_fit(np.asarray(zetas, dtype=np.float64), turned, n_modes)
    m = 2 * n_modes
    cos, sin = np.zeros((3, 3, m + 1)), np.zeros((3, 3, m + 1))
    cos[0, :, :n_modes + 1], sin[0, :, :n_modes + 1] = a_cos, a_sin
    z = np.arange(2 * m + 1) / (2 * m + 1)
    n_per = jnp.arange(m + 1, dtype=float)
    a, da = (np.asarray(v[:, 0]) for v in jax.vmap(lambda q: frame_derivatives(jnp.asarray(cos), jnp.asarray(sin),
                                                                                n_per, q)[:2])(jnp.asarray(z)))
    t = da + omega * np.stack([-a[:, 1], a[:, 0], np.zeros_like(a[:, 0])], 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    N = np.array([1.0, 0.0, 0.0]) - t[:, :1] * t
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    NB_cos, NB_sin, _ = _fourier_fit(z, np.concatenate([N, np.cross(N, t)], 1), m)
    cos[1:], sin[1:] = NB_cos.reshape(2, 3, -1), NB_sin.reshape(2, 3, -1)
    return {"n": nfp * np.arange(m + 1), "cos": cos, "sin": sin}, rms


def section_tilt(Phi, zetas):
    """The angle in degrees between the tangent of the axis ``r = 0`` of the map ``Phi`` and the normal of its surface
    ``zeta = const`` at the axis, on the planes ``zetas``. At the axis of a polar map the radial derivatives at
    ``theta = 0`` and ``1/4`` span the tangent plane of that surface."""
    def tilt(z):
        D0 = jax.jacfwd(Phi)(jnp.array([0.0, 0.0, z]))
        n = jnp.cross(D0[:, 0], jax.jacfwd(Phi)(jnp.array([0.0, 0.25, z]))[:, 0])
        return jnp.arctan2(jnp.linalg.norm(jnp.cross(n, D0[:, 2])), jnp.abs(n @ D0[:, 2]))
    return np.degrees(np.asarray(jax.vmap(tilt)(jnp.asarray(zetas))))


@eqx.filter_jit
def _cut(position, plane_map, x):
    """``(X1, X2, f, miss)`` at the logical points ``x``: the point of ``position`` on the line ``(r, theta)`` of ``x``
    in the plane of the frame of ``plane_map`` at ``zeta`` of ``x``, its coordinates along ``N`` and ``B`` there,
    the shift ``f = zeta_c - zeta`` of the line's logical ``zeta_c`` at that point, and its distance from the plane.
    ``zeta_c`` may leave ``[0, 1)``: one field period on is the point turned by ``2 pi / nfp`` about the ``z`` axis."""
    def at(y):
        k = jnp.floor(y[2])
        P = position(y.at[2].add(-k))
        a = 2.0 * jnp.pi * k / plane_map.nfp
        return jnp.array([jnp.cos(a) * P[0] - jnp.sin(a) * P[1], jnp.sin(a) * P[0] + jnp.cos(a) * P[1], P[2]])

    def one(xq):
        o, e1, e2 = plane_map.plane(xq[2])
        n = jnp.cross(e1, e2)

        def newton(z, _):
            P, dP = jax.jvp(lambda w: at(xq.at[2].set(w)), (z,), (jnp.ones_like(z),))
            return z - (P - o) @ n / (dP @ n), None
        z = jax.lax.scan(newton, xq[2], None, length=CUT_NEWTON_STEPS)[0]
        d = at(xq.at[2].set(z)) - o
        return jnp.stack([d @ e1, d @ e2, z - xq[2], jnp.abs(d @ n)])
    return jax.lax.map(one, x, batch_size=4096)


def _map_space(seq):
    """The sequence of the map of ``seq``: ``seq`` itself, or on a graded space the same mesh with every ring's theta
    count, on which :func:`mrx.geometry.build_sequence` builds the map."""
    if seq.theta_counts is None:
        return seq
    from mrx.derham_sequence import DeRhamSequence

    def knots(b):
        # uniform angular knots are passed as None, which keeps the half-period quadrature
        bp = np.unique(np.asarray(b.T, dtype=np.float64).clip(0.0, 1.0))
        return None if b.type == "periodic" and np.allclose(np.diff(bp), bp[1] - bp[0]) else b.T
    return DeRhamSequence(seq.ns, seq.p, nfp=seq.nfp, symmetry=seq.symmetry,
                          knots=[knots(b) for b in seq.basis_0.Lambda], quad_points=seq.quad_points)


def axis_normal_map(seq, position):
    """``(Phi, shift, info)``: the map of ``position`` re-sectioned so that its surfaces ``zeta = const`` are the
    planes normal to its axis, in the frame of that axis (GVEC's G-frame), and the change of the toroidal coordinate.

    ``position`` is a map from the logical points of ``seq`` (one field period of ``seq.nfp``) to Cartesian points:
    a :class:`~mrx.equilibria.FrameMap` on the 0-form splines of ``seq``, in any frame and with any ``X3``, or the
    exact map of a file's state (:func:`mrx.equilibria.state_position`). The curve is its axis ``position(0, theta,
    zeta)``, sampled at :data:`AXIS_SAMPLES_PER_SPLINE` angles per spline of ``zeta``. Its frame is :func:`curve_frame`, with half as many modes as
    the map has splines of ``zeta``, what they resolve.

    Every point of ``Phi`` is the point of ``position`` on the same line ``(r, theta)`` that lies in the frame's plane
    at its ``zeta``: ``position(r, theta, zeta + f)``, with ``f`` found by Newton's method along the line. So every
    surface ``r = const``, the wall among them, only slides along itself, and the poloidal angle stays. ``X1`` and
    ``X2`` are the point's coordinates in the plane. They and ``f`` are interpolated at the Greville points of the
    tensor splines and projected onto the polar 0-forms with the full theta count on every ring, the map's space
    (:func:`mrx.equilibria.polar_projection`), on a half-period sequence also onto ``X1`` even and ``X2``, ``f``
    odd. ``X3`` is zero: the surfaces ``zeta = const`` of ``Phi`` are the frame's planes exactly, its surfaces
    ``r = const`` those of ``position`` to the interpolation error. The cut of a spline map is only C1 where a line
    crosses a knot of the old map, so the exact map of a file, where there is one, is cut with a smaller error
    (li383 (48,64,24): 0.08 mm from the exact cut against 0.24 mm, W7-X (24,48,28) 0.13 against 0.28 mm, 2026-11-20). The planes of neighbouring ``zeta``
    meet one radius of curvature of the curve away from it, so the map folds if the domain reaches that far:
    ``det DPhi`` at the quadrature points of ``seq`` must be positive (raises otherwise).

    ``shift`` holds the tensor coefficients of ``f``: a logical point ``x`` of ``Phi`` is the point ``x + f(x)
    e_zeta`` of ``position``. A field given in the logical coordinates of ``position`` is carried onto ``Phi`` with
    it (:mod:`mrx.relaxation.initial_conditions` does so for a file's field).

    ``info`` holds the number of frame modes and the curve's fit error (``frame_modes``, ``curve_rms``), the tilt of
    the surfaces ``zeta = const`` against the axis before and after, in degrees (:func:`section_tilt`, the largest
    over the Greville planes), the largest ``|f|`` in field periods, the largest distance of a cut point from its
    plane, the symmetry defect (:func:`mrx.symmetry.stellarator_symmetry_defect`), the range of ``det DPhi`` at the
    sample points of :func:`mrx.equilibria.build_map` and its smallest value at the quadrature points, and the wall
    time."""
    t0 = time.perf_counter()
    bases = seq.basis_0.Lambda
    g = [np.asarray(b.greville_points(), dtype=np.float64) for b in bases]
    n_z = bases[2].n
    zetas = np.arange(AXIS_SAMPLES_PER_SPLINE * n_z + 1) / (AXIS_SAMPLES_PER_SPLINE * n_z + 1)
    x_a = np.stack(np.broadcast_arrays(0.0, 0.0, zetas), 1)
    n_modes = min(n_z // 2, (zetas.size - 1) // 2)
    frame, rms = curve_frame(zetas, np.asarray(jax.vmap(position)(jnp.asarray(x_a, dtype=float))), seq.nfp, n_modes)
    zeros = jnp.zeros(seq.basis_0.shape[0], dtype=float)
    plane_map = FrameMap(zeros, zeros, zeros, jnp.asarray(frame["cos"], dtype=float),
                         jnp.asarray(frame["sin"], dtype=float), jnp.asarray(frame["n"] / seq.nfp, dtype=float),
                         seq.basis_0.bases[0], seq.nfp)
    x = np.stack(np.meshgrid(*g, indexing="ij"), -1)
    v = np.asarray(_cut(position, plane_map, jnp.asarray(x.reshape(-1, 3), dtype=float)), dtype=np.float64)
    v = v.reshape(*x.shape[:-1], 4)
    inv = [np.linalg.inv(np.asarray(b.collocation_matrix(), dtype=np.float64)) for b in bases]
    polar = polar_projection(_map_space(seq))
    raw = [polar(np.einsum("ia,jb,kc,abc->ijk", *inv, v[..., c], optimize=True)) for c in range(3)]
    if seq.half_period:
        raw = [stellarator_symmetric_scalar(jnp.asarray(r), seq.basis_0, even=(c == 0)) for c, r in enumerate(raw)]
    X1, X2, shift = (jnp.asarray(r, dtype=DTYPE) for r in raw)
    Phi = plane_map.with_coordinates(X1, X2, jnp.zeros_like(X1))
    J = np.asarray(frame_geometry(seq, Phi).jacobian_j)
    planes = g[2]
    rng = np.random.default_rng(0)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.15, 1.0, 64), rng.uniform(0.0, 1.0, 64),
                                       rng.uniform(0.0, 1.0, 64)]))
    d = np.asarray(jax.vmap(lambda y: det33(jax.jacfwd(Phi)(y)))(pts))
    info = dict(frame_modes=int(frame["n"].size), curve_rms=rms,
                tilt_before=float(section_tilt(position, planes).max()),
                tilt_after=float(section_tilt(Phi, planes).max()), shift_max=float(np.abs(v[..., 2]).max()),
                plane_miss=float(v[..., 3].max()), symmetry_defect=float(stellarator_symmetry_defect(Phi, pts)),
                det_range=(float(d.min()), float(d.max())), det_quadrature_min=float(J.min()),
                seconds=time.perf_counter() - t0)
    if not J.min() > 0:
        raise RuntimeError(f"det DPhi of the map cut by the planes normal to its axis in [{J.min():.3e}, "
                           f"{J.max():.3e}]: the planes meet inside the domain")
    return Phi, shift, info


def write_map(path, Phi):
    """Add the map ``Phi``, the coefficients of ``X1``, ``X2``, ``X3`` and its frame, to the checkpoint ``path``, so a
    run on a moved or re-sectioned mesh restarts and is traced on it (:func:`install_checkpoint_map`)."""
    with h5py.File(path, "a") as fh:
        for name in MAP_ARRAYS:
            fh.create_dataset(f"map_{name}", data=np.asarray(getattr(Phi, name)))


#: The arrays of a map that a checkpoint holds (:func:`write_map`).
MAP_ARRAYS = ("raw_X1", "raw_X2", "raw_X3", "frame_cos", "frame_sin", "frame_n")


def checkpoint_map(path):
    """The arrays of the map stored in the checkpoint ``path`` by :func:`write_map`, a dict by the names of
    :data:`MAP_ARRAYS`, or ``None`` for a checkpoint without one (written on the file's sections)."""
    with h5py.File(path, "r") as fh:
        if "map_raw_X1" not in fh:
            return None
        return {name: np.asarray(fh[f"map_{name}"]) for name in MAP_ARRAYS}


def install_checkpoint_map(seq, path):
    """Install the map of the checkpoint ``path`` (:func:`checkpoint_map`) on ``seq`` and return ``True``, or return
    ``False`` if it has none. The coefficients come with the frame they belong to."""
    arrays = checkpoint_map(path)
    if arrays is None:
        return False
    Phi = seq.map.with_coordinates(*(jnp.asarray(arrays[k], dtype=seq.map.raw_X1.dtype)
                                     for k in ("raw_X1", "raw_X2", "raw_X3")))
    Phi = Phi.with_frame(*(jnp.asarray(arrays[k], dtype=seq.map.frame_cos.dtype)
                           for k in ("frame_cos", "frame_sin", "frame_n")))
    install_map(seq, Phi)
    return True


class _Path(eqx.Module):
    """The maps ``Phi_tau = Phi0 + tau (Phi1 - Phi0)`` from the installed map to a new one, the straight paths of the
    points, so ``DPhi_tau = D0 + tau (D1 - D0)`` exactly, whatever the frames of the two maps. ``D0``, ``D1`` are the
    Jacobians of the two ends at the quadrature points and ``d_q = Phi_1 - Phi_0`` there, the velocity of the mesh per
    unit ``tau``. The geometry of ``Phi_tau`` keeps ``Phi0`` as its map: the steps use only its metric."""
    Phi0: object
    D0: jnp.ndarray
    D1: jnp.ndarray
    d_q: jnp.ndarray

    def at(self, seq, tau):
        """``(seq_tau, D)``: ``seq`` with the geometry of ``Phi_tau`` and the Jacobian ``D`` of ``Phi_tau``."""
        D = self.D0 + tau * (self.D1 - self.D0)
        G = jnp.einsum("qki,qkj->qij", D, D)
        return seq.with_geometry(SequenceGeometry(self.Phi0, G, jax.vmap(inv33)(G), jnp.linalg.det(D))), D


def _rate(seq, path, tau, b):
    """``db / dtau`` on the map ``Phi_tau``: the relaxation's induction ``curl E``, ``E = Pi_1(u x H)``
    (:func:`mrx.relaxation.physics.electric_field`), on that map with the velocity ``u = -J xi`` of the mesh's
    relative motion, ``xi = DPhi_tau^-1 d`` the logical velocity of the mesh and ``J xi`` the reference components of
    a 2-form."""
    seq_t, D = path.at(seq, tau)
    w_q = seq_t.jacobian_j[:, None] * jnp.linalg.solve(D, path.d_q[..., None])[..., 0]
    return -(seq_t.odd.G[1] @ electric_field(w_q, b, seq_t))


@eqx.filter_jit
def _substep(seq, path, tau0, tau1, b):
    """One SSP-RK3 step of :func:`_rate` from ``tau0`` to ``tau1``."""
    h = tau1 - tau0
    b1 = b + h * _rate(seq, path, tau0, b)
    b2 = 0.75 * b + 0.25 * (b1 + h * _rate(seq, path, tau0 + h, b1))
    return b / 3.0 + 2.0 / 3.0 * (b2 + h * _rate(seq, path, tau0 + 0.5 * h, b2))


def transfer(seq, B, Phi_new):
    """``(B', record)``: the coefficients on the map ``Phi_new`` of the physical field of the 2-form ``B`` on the
    installed map, with ``seq`` unchanged.

    Kept coefficients on a new map move the physical field with the mesh. The relative motion puts it back: on the
    maps ``Phi_tau = Phi + tau (Phi_new - Phi)`` the coefficients of a fixed physical field change as the relaxation's
    induction with the mesh's velocity in place of the flow's (:func:`_rate`), integrated from ``tau = 0`` to 1 in
    substeps that cross at most :data:`~mrx.relaxation.loop.CFL` logical cells, the cap of the relaxation's substeps
    (:func:`~mrx.relaxation.loop.logical_cfl_weights`). Unlike a relaxation step, whose error the next Newton step
    removes, this advection is not corrected afterwards, so the substeps are SSP-RK3, not forward Euler: forward Euler
    substeps blew up over repeated moves at every cap down to 0.05 cells on W7-X (2026-11-19). Every increment is a
    curl of a Dirichlet 1-form, so ``div B = 0``, ``B . n = 0`` and the toroidal flux hold exactly. ``record`` holds the
    number of substeps and the cells the move crosses."""
    x, Phi = seq.quad.x, seq.map
    D0, D1 = map_jacobian_at(Phi, x), map_jacobian_at(Phi_new, x)
    d_q = jax.vmap(Phi_new)(x) - jax.vmap(Phi)(x)
    path = _Path(Phi, D0, D1, d_q)
    weights = logical_cfl_weights(seq)
    cells = max(float(jnp.max(jnp.abs(seq.jacobian_j[:, None]
                                      * jnp.linalg.solve(D0 + t * (D1 - D0), d_q[..., None])[..., 0]) * weights))
                for t in (0.0, 0.5, 1.0))
    n = max(1, int(np.ceil(cells / CFL)))
    b = B
    for i in range(n):
        b = _substep(seq, path, jnp.asarray(i / n), jnp.asarray((i + 1) / n), b)
    return b, dict(substeps=n, cells=cells)
