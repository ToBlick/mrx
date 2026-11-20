"""Motion of the mesh towards the flux surfaces of the field it carries.

A mesh filled from the boundary alone (:mod:`mrx.equilibria.fill`) has its logical surfaces ``r = const`` where the
fill puts them, not on the flux surfaces of the field, and its axis ``r = 0`` where the fill's quality guess puts it,
not on the magnetic axis. The relaxation converges more slowly on such a mesh. A move changes the map ``Phi`` so that
the flux label ``s`` of the field (:mod:`mrx.flux_label`) becomes closer to a function of ``r`` alone. It is a
property of the mesh only: the field is carried so that the physical field stays what it is, and the profile control,
a function of the flux label, does not see it.

Every point moves within its surface ``zeta = const``: the move is given by the change of ``X1``, ``X2`` of the
:class:`~mrx.equilibria.FrameMap` (``R`` and ``Z`` in the cylindrical frame), and ``X3`` follows the surface to first
order (:func:`tangent`, zero on planar sections). So a harmonic ``zeta`` (:func:`harmonic_build`) stays harmonic
while the wall is fixed, which :func:`mrx.nullspace.zeta_harmonic_distance` measures. The wall stays where it is. One move
(:func:`move`) from the field ``B`` and the temperature ``T`` of its flux label on the current map:

1. The axis step. In each plane the magnetic axis is the maximum of ``T`` (:func:`label_maximum`). Its distance
   ``d(zeta)`` from the mesh axis moves every point by ``d(zeta) (1 - r^2)``, the shift with which the fill moves the
   centre of its boundary onto its axis. The alignment below cannot move the axis: its source vanishes where the
   label has its maximum.
2. The alignment error. At every quadrature point of the map moved by the axis step, the label ``s`` and its
   gradient ``g`` in the plane, at the old logical point of the moved point (:func:`locate`). The target of the
   logical surface ``r`` is ``S(r)``, the volume mean of ``s`` on the quadrature layer ``r``, the target that moves
   the mesh least. The error is ``e = s - S(r)``.
3. The velocity ``v`` in the plane minimises ``int |g|^2 |v + kappa e g / |g|^2|^2 + alpha |grad v|^2 dV``, one
   weighted Helmholtz solve ``(M_{|g|^2} + alpha S_0) v_c = -kappa L(e g_c)`` per component ``c``. ``v_1`` is an even,
   ``v_2`` an odd Dirichlet 0-form of the sequence, so the wall stays and the map stays polar-regular and
   stellarator-symmetric. ``kappa`` is :data:`GAIN`, ``alpha = SMOOTHING_CELLS^2 h_r^2 <|g|^2>`` with ``h_r`` the
   radial cell size (:func:`mrx.relaxation.loop.radial_cell_sq`), a smoothing over a fixed number of radial cells.
   Where the label is flat (the axis, an island) the source vanishes and the smoothing carries the mesh. The
   smoothing also moves the axis, so ``v`` less its axis value carried out by ``1 - r^2`` is taken (:func:`pinned`),
   and the axis goes where the axis step puts it.
4. Folds. The axis step and then the velocity are halved until no cell loses more than half its volume
   (:data:`MIN_DET_RATIO` of ``det DPhi``).
5. The field (:func:`undrag`). The coefficients are kept, which drags the physical field with the mesh and keeps
   every flux through every face of the mesh, and the drag is undone by the relaxation's ideal update with the
   mesh's velocity, in SSP-RK3 substeps of at most the relaxation's substep of CFL cells. ``div B = 0``, ``B . n = 0`` and the
   toroidal flux hold exactly. One move changes the physical field by the spatial error of the transport (2.6e-3 on
   W7-X graded (24,48,28) for a 9.9 cm move). Moves there and back return the field to the time error (2.5e-5 after
   twelve on W7-X), and a seeded island chain on li383 keeps its width (2026-11-18).
6. The geometry and the harmonic forms of the new map are installed (:func:`install_map`), the preconditioners of
   the old map are kept. The map is data of the sequence, so the compiled relaxation is reused. Steps 5 and 6 are
   :func:`carry`.

The relaxation removes the error of the carried field. So a few moves with relaxation between them, until the
alignment error is below a tolerance in ``s`` and the axis within :data:`AXIS_TOL` of a radial cell of the label's
maximum (:func:`aligned`), then the run only relaxes. ``scripts/relax.py --mesh.align`` does this between its chunks.
A moved map goes into the checkpoints (:func:`write_map`, :func:`install_checkpoint_map`).

The toroidal coordinate is moved the same way, from a second misfit: ``q``, with ``zeta + q`` the harmonic angle of the
map, the potential of its vacuum field (:func:`mrx.nullspace.harmonic_angle`). Every point slides along its line of
constant ``r`` and ``theta`` by the Newton step of ``q = 0`` (:func:`harmonic_displacement`), and the moves repeat on
the moved map until ``zeta`` is harmonic (:func:`harmonic_build`). ``--geometry.zeta harmonic`` does this to the first
map, before the field exists, so a move is only a change of the map. A move of a field ends in :func:`carry` whichever
misfit gave it.

A third choice of the toroidal coordinate is a cut, not a move: the planes normal to the axis of the map
(:func:`axis_normal_map`), in the frame of the axis, GVEC's G-frame, the default of
:func:`mrx.geometry.build_sequence` for every file. Every point slides along its line of constant ``r`` and
``theta`` into the plane of its ``zeta`` at once, by Newton's method along the line, and the map is in the new frame
with ``X3 = 0``. With the temperature of a flux label the planes are those normal to the magnetic axis instead. The
shift of ``zeta`` the cut returns carries a field given in the old logical coordinates, a file's field above all.

The moves of either misfit keep the preconditioners of the map they start from (:func:`install_map`), and those of the
final map are built once: by :func:`mrx.geometry.build_sequence` after the harmonic moves, by ``scripts/relax.py``
once the mesh is aligned or out of moves.
"""
import time

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction, Pushforward, det33, inv33
from mrx.equilibria import FrameMap, frame_derivatives, polar_projection
from mrx.flux_label import enclosed_flux_map
from mrx.geometry import SequenceGeometry, map_jacobian_at
from mrx.nullspace import compute_nullspaces
from mrx.optimization.shape_ad import frame_geometry
from mrx.precision import DTYPE, lift, to_working
from mrx.relaxation.initial_conditions import toroidal_flux
from mrx.relaxation.loop import CFL, logical_cfl_weights, radial_cell_sq
from mrx.relaxation.physics import compute_divergence_norm
from mrx.solvers import preconditioned_cg
from mrx.symmetry import stellarator_symmetric_scalar, stellarator_symmetry_defect

#: The fraction ``kappa`` of the linearised alignment error a move removes.
GAIN = 1.0
#: The smoothing length of the velocity in radial cells: ``alpha = SMOOTHING_CELLS^2 h_r^2 <|g|^2>``. On W7-X graded
#: (24,48,28) from the plain fill the rms of ``s - S(r)`` fell 6.3 % -> 1.9 -> 1.1 -> 0.83 % in three moves with 2 cells,
#: 6.3 % -> 2.8 -> 2.0 -> 1.6 % with 4 (2026-11-18).
SMOOTHING_CELLS = 2.0
#: The smallest ratio of ``det DPhi`` after and before a move at any quadrature point.
MIN_DET_RATIO = 0.5
#: The distance of the mesh axis from the label's maximum, in first radial cells, below which the axis is aligned.
AXIS_TOL = 0.25
#: The relative tolerance of the two Helmholtz solves of the velocity.
VELOCITY_TOL = 1e-6
#: The distance |grad zeta - h| / |h| (:func:`mrx.nullspace.zeta_harmonic_distance`) below which the moves of
#: :func:`harmonic_build` stop. From the planes phi = const of the boundary meshes it fell 0.29 -> 1.6e-2 -> 1.6e-5 ->
#: 2.5e-7 (li383 (48,64,24)), 0.35 -> 3.1e-2 -> 9.5e-5 -> 2.3e-7 (W7-X (24,48,28)), 0.65 -> 9.3e-2 -> 2.4e-3 -> 5.3e-6
#: (HSX (32,64,36)), three moves each, and the moves of the flux alignment let it drift to 7e-4 (2026-11-19).
HARMONIC_TOL = 1e-5
#: The Newton steps of :func:`locate`.
NEWTON_STEPS = 10


def _chart(p, z):
    """The logical point of the chart point ``p = (r cos 2 pi theta, r sin 2 pi theta)`` on the surface ``zeta = z``."""
    return jnp.stack([jnp.hypot(p[0], p[1]), jnp.arctan2(p[1], p[0]) / (2.0 * jnp.pi) % 1.0, z])


def _unchart(x):
    return jnp.stack([x[0] * jnp.cos(2.0 * jnp.pi * x[1]), x[0] * jnp.sin(2.0 * jnp.pi * x[1])])


def extended(Phi, x):
    """The map ``Phi`` at a logical point whose ``zeta`` may leave ``[0, 1)``: one field period on is the point turned
    by ``2 pi / nfp`` about the ``z`` axis."""
    k = jnp.floor(x[2])
    c, s = jnp.cos(2.0 * jnp.pi * k / Phi.nfp), jnp.sin(2.0 * jnp.pi * k / Phi.nfp)
    P = Phi(x.at[2].add(-k))
    return jnp.array([c * P[0] - s * P[1], s * P[0] + c * P[1], P[2]])


@eqx.filter_jit
def locate(Phi, X, x):
    """``(x', miss)``: the logical points ``x'`` with ``Phi(x') = X`` (Cartesian, ``(n, 3)``) and ``|Phi(x') - X|``.
    Newton from the logical points ``x`` in the chart ``(r cos 2 pi theta, r sin 2 pi theta, zeta)``, which is
    smooth at the axis. ``zeta`` of ``x'`` may leave ``[0, 1)`` by the move of a point out of its section."""
    def one(xq, Xq):
        F = lambda p: extended(Phi, _chart(p[:2], p[2])) - Xq   # noqa: E731
        p = jnp.concatenate([_unchart(xq), xq[2:]])
        for _ in range(NEWTON_STEPS):
            p = p - jnp.linalg.solve(jax.jacfwd(F)(p), F(p))
        return _chart(p[:2], p[2]), jnp.linalg.norm(F(p))
    return jax.vmap(one)(x, X)


@eqx.filter_jit
def _temperature(seq, T, x):
    """``T`` at the logical points ``x`` and its derivatives in ``r``, ``theta``, ``zeta``."""
    f = DiscreteFunction(T, seq.basis_0, seq.even.E(0))
    return jax.vmap(jax.value_and_grad(lambda y: f(y)[0]))(x)


def label_maximum(seq, T, zetas):
    """The logical points ``(len(zetas), 3)`` of the maximum of the label's temperature ``T`` on the planes
    ``zetas``: where the label puts the magnetic axis. The largest value on a polar grid of each plane, then three
    rounds of the stationary point of a quadratic fitted to ``T`` on a 7 x 7 grid about the last point in the chart
    ``(r cos 2 pi theta, r sin 2 pi theta)``, the grid four times finer each round."""
    zetas = np.asarray(zetas, dtype=np.float64)
    nz = zetas.size
    r, th = np.meshgrid(np.linspace(0.0, 0.9, 37), np.arange(48) / 48, indexing="ij")
    pr = np.stack([(r * np.cos(2 * np.pi * th)).ravel(), (r * np.sin(2 * np.pi * th)).ravel()], 1)

    def values(p, z):
        x = jax.vmap(_chart)(jnp.asarray(p), jnp.asarray(z))
        return np.asarray(_temperature(seq, T, x)[0])
    v = values(np.tile(pr, (nz, 1)), np.repeat(zetas, len(pr))).reshape(nz, -1)
    c = pr[np.argmax(v, axis=1)]
    a = np.linspace(-3.0, 3.0, 7)
    du, dw = (q.ravel() for q in np.meshgrid(a, a, indexing="ij"))
    h = 0.025
    for _ in range(3):
        p = c[:, None, :] + h * np.stack([du, dw], 1)[None]
        v = values(p.reshape(-1, 2), np.repeat(zetas, du.size)).reshape(nz, -1)
        A = np.stack([np.ones_like(du), du, dw, du * du, du * dw, dw * dw], 1)
        k = np.linalg.lstsq(A, v.T, rcond=None)[0]                              # (6, nz)
        H = np.stack([np.stack([2 * k[3], k[4]], -1), np.stack([k[4], 2 * k[5]], -1)], -2)
        c = c + h * np.linalg.solve(H, -np.stack([k[1], k[2]], -1)[..., None])[..., 0]
        h /= 4.0
    return np.asarray(jax.vmap(_chart)(jnp.asarray(c), jnp.asarray(zetas)))


def _planes(seq):
    """The Greville points of the ``zeta`` splines, on which the axis step is interpolated."""
    return np.asarray(seq.basis_0.Lambda[2].greville_points(), dtype=np.float64)


def first_cell(seq, Phi, zetas):
    """The mean distance in the planes ``zetas`` from the axis to the first radial breakpoint, the size of the
    innermost radial cell."""
    T_r = np.asarray(seq.basis_0.Lambda[0].T, dtype=np.float64)
    r1 = np.unique(T_r[(T_r > 0) & (T_r <= 1)])[0]
    th = np.arange(16) / 16
    x = jnp.asarray(np.stack(np.broadcast_arrays(r1, th[None, :], zetas[:, None]), -1).reshape(-1, 3))
    ax = jnp.asarray(np.stack(np.broadcast_arrays(0.0, 0.0, zetas[:, None]), -1).reshape(-1, 3))
    S = jax.vmap(Phi.section)
    return float(jnp.mean(jnp.linalg.norm(S(x).reshape(len(zetas), 16, 2) - S(ax).reshape(len(zetas), 1, 2), axis=-1)))


def axis_offset(seq, T):
    """``(d, h_1)``: the in-plane displacement ``d`` ``(n_zeta, 2)`` from the mesh axis to the label's maximum on the
    planes of :func:`_planes`, and the size ``h_1`` of the first radial cell (:func:`first_cell`)."""
    Phi, zetas = seq.map, _planes(seq)
    xa = label_maximum(seq, T, zetas)
    ax = jnp.asarray(np.stack(np.broadcast_arrays(0.0, 0.0, zetas[:, None]), -1).reshape(-1, 3))
    S = jax.vmap(Phi.section)
    return np.asarray(S(jnp.asarray(xa)) - S(ax)), first_cell(seq, Phi, zetas)


def _axis_profile(seq):
    """The radial spline coefficients of ``1 - r^2``, interpolated (exactly for degree 2 and up). The first is 1."""
    br = seq.basis_0.Lambda[0]
    gr = np.asarray(br.greville_points(), dtype=np.float64)
    return np.linalg.solve(np.asarray(br.collocation_matrix(), dtype=np.float64), 1.0 - gr ** 2)


def axis_step(seq, d):
    """The coefficients ``(2, n_r, n_theta, n_zeta)`` of ``X1``, ``X2`` that move every point by ``d(zeta) (1 - r^2)``,
    ``d`` given on the planes of :func:`_planes` and interpolated by the ``zeta`` splines. Constant in ``theta``, so
    polar-regular. On a stellarator-symmetric sequence ``X1``'s part is projected on the even, ``X2``'s on the odd
    functions."""
    bz = seq.basis_0.Lambda[2]
    dz = np.linalg.solve(np.asarray(bz.collocation_matrix(), dtype=np.float64), d)               # (n_z, 2)
    n_t = seq.basis_0.Lambda[1].n
    raw = jnp.asarray(np.einsum("i,j,kc->cijk", _axis_profile(seq), np.ones(n_t), dz), dtype=seq.map.raw_X1.dtype)
    if seq.symmetry == "stellarator":
        raw = jnp.stack([stellarator_symmetric_scalar(raw[0], seq.basis_0, even=True),
                         stellarator_symmetric_scalar(raw[1], seq.basis_0, even=False)])
    return raw


def pinned(seq, v):
    """The coefficients ``v`` ``(2, n_r, n_theta, n_zeta)`` less their axis values carried out by ``1 - r^2``: zero on
    the axis, so they leave the axis where the axis step puts it. The axis value of a polar-regular spline is its ring 0
    coefficient, the same for every ``theta``."""
    return v - jnp.einsum("i,ck->cik", jnp.asarray(_axis_profile(seq), dtype=v.dtype), v[:, 0, 0, :])[:, :, None, :]


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
def _det_ratio(seq, Phi):
    """``det DPhi`` of ``Phi`` over that of the installed map at the quadrature points."""
    return frame_geometry(seq, Phi).jacobian_j / seq.jacobian_j


def _backtrack(seq, base, delta):
    """``(lam, ratio)``: the largest ``lam`` of ``1, 1/2, ..., 2^-8`` for which no cell of ``base + lam delta``
    loses more than :data:`MIN_DET_RATIO` of its volume, or 0, and the ratio of ``det DPhi`` there."""
    lam = 1.0
    for _ in range(9):
        ratio = _det_ratio(seq, moved(seq.map, base + lam * delta))
        if float(jnp.min(ratio)) >= MIN_DET_RATIO:
            return lam, ratio
        lam *= 0.5
    return 0.0, _det_ratio(seq, moved(seq.map, base))


@eqx.filter_jit
def _label_at(seq, T, B_jk, x):
    """``(s, g)`` at the logical points ``x``: the label and its gradient in the plane, ``(n, 2)``, with respect to
    the in-plane coordinates of the installed map."""
    smap = enclosed_flux_map(seq, T, B_jk)
    Tx, dT = _temperature(seq, T, x)
    J2 = jax.vmap(lambda y: jax.jacfwd(seq.map.section)(y)[:, :2])(x)
    dsdT = jax.vmap(jax.grad(smap))(Tx)
    g = jnp.linalg.solve(jnp.swapaxes(J2, 1, 2), dT[:, :2, None])[..., 0]
    return smap(Tx), dsdT[:, None] * g


def alignment(seq, s_q):
    """``(e, e_rms, e_max)``: the alignment error ``e = s - S(r)`` of the label ``s_q`` at the quadrature points,
    ``S`` the volume mean of ``s`` on each quadrature layer ``r``, and its volume rms and largest size."""
    shape = tuple(int(n) for n in seq.quad.shape)
    w = (seq.quad.w * seq.jacobian_j).reshape(shape)
    s = jnp.asarray(s_q).reshape(shape)
    e = (s - (w * s).sum((1, 2), keepdims=True) / w.sum((1, 2), keepdims=True)).reshape(-1)
    w = w.reshape(-1)
    return e, float(jnp.sqrt(jnp.sum(w * e ** 2) / jnp.sum(w))), float(jnp.max(jnp.abs(e)))


@eqx.filter_jit
def _velocity(seq, e, g, alpha, kappa):
    """The coefficients ``(2, n_r, n_theta, n_zeta)`` of the velocity of step 3 of the module docstring and the
    signed counts of its two solves."""
    g2 = jnp.sum(g ** 2, axis=1)
    shape = seq.basis_0.shape[0]

    def solve(view, comp):
        def A(v):
            v = to_working(v)
            return lift(view._scalar_load_values(g2 * view.evaluate_at_quadrature(v, 0)[:, 0], 0)
                        + alpha * (view.S[0] @ v))
        b = view._scalar_load_values(-kappa * e * comp, 0)
        v, info = preconditioned_cg(A, b, M=lambda x: view.L[0].precondition(x) / alpha, tol=VELOCITY_TOL,
                                    maxiter=seq.maxiter)
        return (view.E(0).T @ to_working(v)).reshape(shape), info
    v1, i1 = solve(seq.even, g[:, 0])
    v2, i2 = solve(seq.odd, g[:, 1])
    return jnp.stack([v1, v2]), i1, i2


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
    """``(db / dtau, info)`` on the map ``Phi_tau``: ``-curl E`` with ``E`` the projection of ``xi x b`` onto the
    Dirichlet 1-forms in that map's ``M_1``, ``xi = DPhi_tau^-1 d`` the logical velocity of the mesh. ``J xi`` is the
    reference velocity of a 2-form, so this is the ideal increment ``curl(u x B)`` of a relaxation step with the
    velocity ``u = -J xi``. ``info`` is the signed count of the mass solve."""
    seq_t, D = path.at(seq, tau)
    odd = seq_t.odd
    w_q = seq_t.jacobian_j[:, None] * jnp.linalg.solve(D, path.d_q[..., None])[..., 0]
    E, info = odd.M[1].solve(odd.cross_product_load_values(w_q, odd.evaluate_at_quadrature(b, 2), 1, 2, 2),
                             return_info=True)
    return -(odd.G[1] @ E), info


@eqx.filter_jit
def _substep(seq, path, tau0, tau1, b):
    """One SSP-RK3 step of :func:`_rate` from ``tau0`` to ``tau1`` and the signed counts of its three mass solves."""
    h = tau1 - tau0
    k1, i1 = _rate(seq, path, tau0, b)
    b1 = b + h * k1
    k2, i2 = _rate(seq, path, tau0 + h, b1)
    b2 = 0.75 * b + 0.25 * (b1 + h * k2)
    k3, i3 = _rate(seq, path, tau0 + 0.5 * h, b2)
    return b / 3.0 + 2.0 / 3.0 * (b2 + h * k3), jnp.stack([i1, i2, i3])


def undrag(seq, B, Phi_new):
    """``(B', record)``: the coefficients on the map ``Phi_new`` of the physical field of the 2-form ``B`` on the
    installed map.

    Keeping the coefficients on a new map moves the physical field with the mesh, which keeps every flux through every
    face of the mesh. The relative motion puts it back. On the maps ``Phi_tau = Phi + tau (Phi_new - Phi)`` the
    coefficients of a fixed physical field change as ``db / dtau = d(i_xi b) = -curl(xi x b)``, ``xi`` the logical
    velocity of the mesh: the ideal update of the relaxation with the mesh's velocity in place of the flow's
    (:func:`_rate`, on the map of each stage). Its eigenvalues lie near the imaginary axis, at most 1.41 times the
    logical cells crossed per unit ``tau`` (li383 (8..16, 16..32, 8), dense, 2026-11-18). Unlike a relaxation step,
    whose error the next Newton step removes, this advection is not corrected afterwards: forward Euler and the
    two-stage methods amplify every such mode (forward Euler substeps blew up over repeated moves at every cap down to
    0.05 cells on W7-X, 2026-11-19), SSP-RK3 is stable up to ``|z| = 3^(1/2)``. So it is integrated from ``tau = 0``
    to 1 by SSP-RK3 in substeps that cross at most :data:`~mrx.relaxation.loop.CFL` logical cells
    (:func:`~mrx.relaxation.loop.logical_cfl_weights`), the cap of the relaxation's substeps. Every increment is a
    curl of a Dirichlet 1-form, so ``div B = 0``, ``B . n = 0`` and the toroidal flux hold exactly. The fluxes of the
    other surfaces and the topology are kept to the accuracy of the transport, as by the relaxation's own steps.
    ``record`` holds the number of substeps, the cells the move crosses, the number of mass solves and the number of
    them that did not converge."""
    x, Phi = seq.quad.x, seq.map
    D0, D1 = map_jacobian_at(Phi, x), map_jacobian_at(Phi_new, x)
    d_q = jax.vmap(Phi_new)(x) - jax.vmap(Phi)(x)
    path = _Path(Phi, D0, D1, d_q)
    weights = logical_cfl_weights(seq)
    cells = max(float(jnp.max(jnp.abs(seq.jacobian_j[:, None]
                                      * jnp.linalg.solve(D0 + t * (D1 - D0), d_q[..., None])[..., 0]) * weights))
                for t in (0.0, 0.5, 1.0))
    n = max(1, int(np.ceil(cells / CFL)))
    b, failed = B, 0
    for i in range(n):
        b, info = _substep(seq, path, jnp.asarray(i / n), jnp.asarray((i + 1) / n), b)
        failed += int(jnp.sum(info < 0))
    return b, dict(substeps=n, cells=cells, mass_solves=3 * n, mass_cg_failed=failed)


def install_map(seq, Phi, keep_preconditioners=False):
    """Install the map ``Phi`` on ``seq`` with its harmonic forms, and with its own preconditioners unless
    ``keep_preconditioners``. A move keeps those of the map before it. On the boundary meshes of li383 (48,64,24),
    W7-X (24,48,28) and HSX (32,64,36), aligned after three, three and one moves, the start map's preconditioners take
    the same MINRES and label CG counts as those of the moved map, but the mass solves inside every step take more
    iterations (the 1-form solve of the Newton operator 15, 15, 19 against 11, 11, 16), and a chunk of ten steps with
    both targets takes 26, 23 and 12 % longer. A build costs 57-72 s on an H200 (host assembly), the price of 25 to
    50 such chunks, so it is done once, on the final map (2026-11-19)."""
    seq.set_geometry(frame_geometry(seq, Phi), keep_preconditioners)
    if not keep_preconditioners:
        seq.build_preconditioners()
    compute_nullspaces(seq, verbose=False)


def _field_sampler(seq, n=(8, 32, 8)):
    """A function giving the physical field of a 2-form, on the map installed when it is called, at the physical
    points of a fixed logical grid of the map installed now (``r`` in 0.1 .. 0.9), with the largest miss of their
    inversion."""
    r, t, z = np.meshgrid(np.linspace(0.1, 0.9, n[0]), np.arange(n[1]) / n[1], np.arange(n[2]) / n[2],
                          indexing="ij")
    x = jnp.asarray(np.stack([r.ravel(), t.ravel(), z.ravel()], 1))
    X = jax.vmap(seq.map)(x)

    def B_at(B):
        xn, miss = locate(seq.map, X, x)
        f = Pushforward(DiscreteFunction(B, seq.basis_2, seq.odd.E(2)), seq.map, 2)
        return np.asarray(jax.vmap(f)(xn)), float(jnp.max(miss))
    return B_at


def aligned(e_rms, d, h1, tol):
    """Whether a mesh is aligned: the alignment error's rms below ``tol`` and the axis within :data:`AXIS_TOL` of
    the first radial cell ``h1`` of the label's maximum (``d`` from :func:`axis_offset`)."""
    return e_rms < tol and float(np.max(np.linalg.norm(d, axis=1))) < AXIS_TOL * h1


def displacement(seq, B, T, velocity=True):
    """``(Phi_new, record)``: the map of one move of the mesh of ``seq`` towards the flux surfaces of ``B``, whose label
    has the temperature ``T``, steps 1 to 4 of the module docstring. ``seq`` is not changed. Without ``velocity`` only
    the axis step moves the mesh.

    ``record`` holds the alignment before the move (``e_rms``, ``e_max``), the axis offset (``axis_m``, the largest
    distance in metres, and ``axis_cells`` in first radial cells), the backtracking factors of the axis step and the
    velocity (``lam_axis``, ``lam_v``), the largest move in metres, the range of the ratio of ``det DPhi`` after and
    before and the counts of the two Helmholtz solves."""
    B_jk = seq.odd.evaluate_at_quadrature(B, 2)
    Phi = seq.map
    d, h1 = axis_offset(seq, T)
    s_q = enclosed_flux_map(seq, T, B_jk)(seq.even.evaluate_at_quadrature(T, 0)[:, 0])
    _, e_rms, e_max = alignment(seq, s_q)
    # 1. the axis step, halved until no cell folds
    delta_a = tangent(seq, axis_step(seq, d))
    lam_a, _ = _backtrack(seq, jnp.zeros_like(delta_a), delta_a)
    delta_a = lam_a * delta_a
    # 2. the label and its gradient at the points moved by the axis step, found in the old map
    x_old, miss = locate(Phi, jax.vmap(moved(Phi, delta_a))(seq.quad.x), seq.quad.x)
    if velocity:
        s, g = _label_at(seq, T, B_jk, x_old)
        e, _, _ = alignment(seq, s)
        # 3. the velocity
        wJ = seq.quad.w * seq.jacobian_j
        alpha = SMOOTHING_CELLS ** 2 * radial_cell_sq(seq) * jnp.sum(wJ * jnp.sum(g ** 2, axis=1)) / jnp.sum(wJ)
        v, i1, i2 = _velocity(seq, e, g, alpha, jnp.asarray(GAIN, dtype=e.dtype))
        v = tangent(seq, pinned(seq, v))
    else:
        v, i1, i2, alpha = jnp.zeros_like(delta_a), 0, 0, 0.0
    # 4. the velocity halved until no cell folds
    lam_v, ratio = _backtrack(seq, delta_a, v)
    delta = delta_a + lam_v * v
    record = dict(e_rms=e_rms, e_max=e_max, axis_m=float(np.max(np.linalg.norm(d, axis=1))),
                  axis_cells=float(np.max(np.linalg.norm(d, axis=1)) / h1), h1_m=h1, lam_axis=lam_a, lam_v=lam_v,
                  move_m=float(jnp.max(jnp.linalg.norm(_section_values(seq, Phi, delta), axis=1))),
                  alpha=float(alpha), locate_miss=float(jnp.max(miss)), det_ratio_min=float(jnp.min(ratio)),
                  det_ratio_max=float(jnp.max(ratio)), velocity_cg=[int(i1), int(i2)])
    return moved(Phi, delta), record


def move(seq, B, T, velocity=True):
    """Move the mesh of ``seq`` once towards the flux surfaces of ``B``, whose label has the temperature ``T``
    (:func:`displacement`, without ``velocity`` only the axis step), carry ``B`` onto the new map (:func:`carry`) and
    return ``(B', record)``, with the records of both and the wall time of the whole move."""
    t0 = time.perf_counter()
    Phi_new, record = displacement(seq, B, T, velocity)
    B_new, carried = carry(seq, B, Phi_new)
    record.update(carried, wall=time.perf_counter() - t0)
    return B_new, record


def carry(seq, B, Phi_new):
    """Install the map ``Phi_new`` on ``seq`` and return ``(B', record)``: the field ``B`` carried onto it
    (:func:`undrag`) and what the move measured. Every move of a field ends here, whichever misfit gave its map.

    ``record`` holds that of :func:`undrag`, the range of ``det DPhi`` after the move, the change of the physical
    field at fixed physical points (relative L2, ``dB_phys``, with the largest miss of their inversion), the relative
    changes of the energy and of the toroidal flux, ``||div B|| / ||B||`` and the wall time."""
    t0 = time.perf_counter()
    odd = seq.odd
    B_at = _field_sampler(seq)
    before, _ = B_at(B)
    E0, F0 = 0.5 * float(odd.l2_norm_sq(B, 2)), float(toroidal_flux(seq, B))
    B_new, record = undrag(seq, B, Phi_new)
    install_map(seq, Phi_new, keep_preconditioners=True)
    after, miss_new = B_at(B_new)
    J = np.asarray(seq.jacobian_j)
    record.update(det_min=float(J.min()), det_max=float(J.max()),
                  dB_phys=float(np.sqrt(np.sum((after - before) ** 2) / np.sum(before ** 2))), dB_miss=miss_new,
                  dE_rel=(0.5 * float(odd.l2_norm_sq(B_new, 2)) - E0) / E0,
                  flux_rel=(float(toroidal_flux(seq, B_new)) - F0) / F0,
                  div_rel=float(compute_divergence_norm(B_new, seq) / odd.l2_norm(B_new, 2)),
                  wall=time.perf_counter() - t0)
    return B_new, record


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
    (even), ``X2`` and ``X3`` (odd) is the L2 projection onto the free 0-forms of the sequence, the space of the
    flux alignment's velocity with the wall's values free. The move is halved until no cell folds
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


def axis_normal_map(seq, position, T=None):
    """``(Phi, shift, info)``: the map of ``position`` re-sectioned so that its surfaces ``zeta = const`` are the
    planes normal to its axis, in the frame of that axis (GVEC's G-frame), and the change of the toroidal coordinate.

    ``position`` is a map from the logical points of ``seq`` (one field period of ``seq.nfp``) to Cartesian points:
    a :class:`~mrx.equilibria.FrameMap` on the 0-form splines of ``seq``, in any frame and with any ``X3``, or the
    exact map of a file's state (:func:`mrx.equilibria.state_position`). The curve is its axis ``position(0, theta,
    zeta)``, sampled at :data:`AXIS_SAMPLES_PER_SPLINE` angles per spline of ``zeta``, or with the temperature ``T``
    of a flux label on the installed map of ``seq`` the label's maximum on the Greville planes of the ``zeta``
    splines (:func:`label_maximum`), the magnetic axis. Its frame is :func:`curve_frame`, with half as many modes as
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
    it (:mod:`mrx.relaxation.initial_conditions` does so for a file's field). A discrete field of a map installed on
    ``seq`` is carried by :func:`carry`.

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
    if T is None:
        zetas = np.arange(AXIS_SAMPLES_PER_SPLINE * n_z + 1) / (AXIS_SAMPLES_PER_SPLINE * n_z + 1)
        x_a = np.stack(np.broadcast_arrays(0.0, 0.0, zetas), 1)
    else:
        zetas = g[2]
        x_a = label_maximum(seq, T, zetas)
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
    :data:`MAP_ARRAYS`, or ``None`` for a checkpoint without one (written on the file's sections). A checkpoint written
    before the frame was stored holds the coefficients alone, on the frame of the file."""
    with h5py.File(path, "r") as fh:
        if "map_raw_X1" not in fh:
            return None
        return {name: np.asarray(fh[f"map_{name}"]) for name in MAP_ARRAYS if f"map_{name}" in fh}


def install_checkpoint_map(seq, path):
    """Install the map of the checkpoint ``path`` (:func:`checkpoint_map`) on ``seq`` and return ``True``, or return
    ``False`` if it has none. The coefficients come with the frame they belong to, unless the checkpoint predates the
    stored frame."""
    arrays = checkpoint_map(path)
    if arrays is None:
        return False
    Phi = seq.map.with_coordinates(*(jnp.asarray(arrays[k], dtype=seq.map.raw_X1.dtype)
                                     for k in ("raw_X1", "raw_X2", "raw_X3")))
    if "frame_cos" in arrays:
        Phi = Phi.with_frame(*(jnp.asarray(arrays[k], dtype=seq.map.frame_cos.dtype)
                               for k in ("frame_cos", "frame_sin", "frame_n")))
    install_map(seq, Phi)
    return True
