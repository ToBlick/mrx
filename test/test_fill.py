"""The map from a boundary alone (:mod:`mrx.equilibria.fill`).

The boundary is that of shearedE in its namelist, with no axis, so the reader fills it about the axis
:func:`~mrx.equilibria.fill.guess_axis` places. The fill reaches the boundary and the axis exactly, and the guessed axis
is a maximum of its objective and leaves the fill unfolded. The fill's spline map on the tiny shearedE spaces, cut by the
planes normal to its axis as every map of a sequence is, meets the boundary to the projection error with a positive
Jacobian.
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.equilibria import read_equilibrium, state_position
from mrx.equilibria.fill import fill_rows, guess_axis
from mrx.equilibria.indata import read_indata
from mrx.relaxation.mesh_motion import axis_normal_map
from test.conftest import LENGTH, minor_radius

#: The move of the guessed axis point that must lower the guess's objective, as a fraction of the minor radius.
AXIS_STEP = 0.03


def _section_jacobian(boundary, axis_point, v, n_rho=200, n_u=400):
    """``J / rho`` of the fill in the plane ``v`` about the point ``axis_point`` ``(R, Z)`` of that plane, by centred
    differences on an ``n_rho`` x ``n_u`` grid (an evaluation independent of the library's closed-form derivatives).
    ``J = R_rho Z_u - R_u Z_rho`` with the sign that makes it positive for an unfolded fill."""
    m, n, Rb, Zb = (boundary[:, i] for i in range(4))
    rho = np.linspace(0.0, 1.0, n_rho + 1)[1:-1, None, None]
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)[None, :, None]
    shape = np.where(m == 0, rho ** 2, rho ** m)
    R = axis_point[0] * (1.0 - rho[..., 0] ** 2) + np.sum(shape * Rb * np.cos(m * u - n * v), axis=-1)
    Z = axis_point[1] * (1.0 - rho[..., 0] ** 2) + np.sum(shape * Zb * np.sin(m * u - n * v), axis=-1)
    dr, du = 1.0 / n_rho, 2.0 * np.pi / n_u
    R_r, Z_r = (np.gradient(X, dr, axis=0) for X in (R, Z))
    R_u, Z_u = ((np.roll(X, -1, axis=1) - np.roll(X, 1, axis=1)) / (2 * du) for X in (R, Z))
    J = (R_r * Z_u - R_u * Z_r) / rho[..., 0]
    return J * np.sign(np.mean(J))


def _rows(sheared):
    """The namelist's boundary rows ``(m, n, R_mn, Z_mn)``, ``n`` times nfp, and nfp."""
    nl = read_indata(sheared[2])
    return np.asarray(nl.boundary_rows()[0]), nl.nfp


def test_fill_meets_boundary_and_axis(sheared):
    """The fill is the axis at rho = 0 and the boundary at rho = 1 to round-off, mode by mode (numpy, milliseconds)."""
    rows, nfp = _rows(sheared)
    axis = np.array([[0, 0.75, 0.0], [nfp, 0.01, -0.005]])
    m, n, R, Z = fill_rows(rows, axis, np.array([0.0, 1.0]))
    on_axis = np.where(m == 0, 1.0, 0.0)
    ax_R = on_axis * np.array([dict((int(k), r) for k, r, _ in axis).get(int(k), 0.0) for k in n])
    ax_Z = on_axis * np.array([dict((int(k), z) for k, _, z in axis).get(int(k), 0.0) for k in n])
    table = {(int(a), int(b)): (r, z) for a, b, r, z in rows}
    eps = np.finfo(np.float64).eps
    assert np.max(np.abs(R[0] - ax_R)) <= 4 * eps and np.max(np.abs(Z[0] - ax_Z)) <= 4 * eps
    assert np.max(np.abs(R[1] - [table.get((int(a), int(b)), (0.0, 0.0))[0] for a, b in zip(m, n)])) <= 4 * eps
    assert np.max(np.abs(Z[1] - [table.get((int(a), int(b)), (0.0, 0.0))[1] for a, b in zip(m, n)])) <= 4 * eps


def test_guessed_axis_is_the_best_unfolded_fill(sheared):
    """In each of four planes per field period the fill about the guessed axis is unfolded (``J / rho`` positive on a
    fine grid), and moving the axis point by :data:`AXIS_STEP` of the minor radius in R or Z lowers the mean of
    ``log(J / rho)``, the objective the guess maximises (the axis of the Fourier fit is within the grid search's
    resolution of the planes' maxima). Numpy."""
    rows, nfp = _rows(sheared)
    step = AXIS_STEP * minor_radius(rows, nfp)
    axis = guess_axis(rows, nfp)
    for v in 2.0 * np.pi / nfp * np.arange(4) / 4:
        point = np.array([np.cos(axis[:, 0] * v) @ axis[:, 1], -np.sin(axis[:, 0] * v) @ axis[:, 2]])
        J = _section_jacobian(rows, point, v)
        assert J.min() > 0.0
        best = np.mean(np.log(J))
        for d in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            J_d = _section_jacobian(rows, point + np.array(d), v)
            assert J_d.min() <= 0.0 or np.mean(np.log(J_d)) < best


def test_the_fill_map_meets_the_boundary(sheared):
    """The namelist's exact fill (:func:`mrx.equilibria.state_position`) cut by the planes normal to its axis on the
    tiny shearedE spaces (:func:`mrx.relaxation.mesh_motion.axis_normal_map`, the map of every sequence) has a positive
    Jacobian at every quadrature point, and at ``r = 1`` lies on the namelist's boundary at its own ``phi`` within
    :data:`~test.conftest.LENGTH` of the minor radius. A point of the wall keeps its ``theta`` and slides along the
    boundary. 0.63 % measured on (12,16,12), PENDING on the tiny mesh."""
    boundary, nfp = _rows(sheared)
    seq = sheared[3]
    eq = read_equilibrium(sheared[2])
    Phi, _, cut = axis_normal_map(seq, state_position(eq))
    t, z = np.meshgrid(np.arange(64) / 64, np.arange(12) / 12, indexing="ij")
    X = np.asarray(jax.vmap(Phi)(jnp.asarray(np.stack([np.ones(t.size), t.ravel(), z.ravel()], axis=1))))
    u = 2.0 * np.pi * t.ravel() * (-1.0 if eq["theta_reversed"] else 1.0)
    v = np.arctan2(X[:, 1], X[:, 0])
    arg = boundary[:, 0][None, :] * u[:, None] - boundary[:, 1][None, :] * v[:, None]
    miss = np.hypot(np.hypot(X[:, 0], X[:, 1]) - np.cos(arg) @ boundary[:, 2], X[:, 2] - np.sin(arg) @ boundary[:, 3])
    a = minor_radius(boundary, nfp)
    print(f"\n  min det DPhi {cut['det_range'][0]:.3e}, max distance from the boundary at r = 1 {miss.max():.2e} m = "
          f"{miss.max() / a:.2%} of the minor radius {a:.3f} m")
    assert cut["det_range"][0] > 0.0
    assert miss.max() < LENGTH * a
