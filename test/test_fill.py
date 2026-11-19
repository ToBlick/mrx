"""The map from a boundary alone (:mod:`mrx.equilibria.fill`) and the vacuum start on it.

The boundary is that of the tracked li383 wout, written as a VMEC input namelist with no axis, so the reader fills it
about the axis :func:`~mrx.equilibria.fill.guess_axis` places. The fill reaches the boundary and the axis exactly,
the guessed axis is a maximum of its objective and leaves the fill unfolded, the spline map built from it meets the
boundary and has a positive Jacobian at every quadrature point, and the vacuum field has the namelist's toroidal flux
through every cross-section, no divergence and no current.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import netcdf_file

from mrx.equilibria.fill import fill_rows, guess_axis
from mrx.equilibria.indata import read_indata
from mrx.paths import data_file
from mrx.precision import eps

WOUT = "data/wout_li383_low_res_reference.nc"
NS, P = (16, 32, 12), 2


def _write_namelist(tmp_path):
    """The boundary of :data:`WOUT` and its total toroidal flux as a namelist ``input.li383_boundary``, without an axis.
    Returns its path and the boundary rows ``(m, n, R_mn, Z_mn)`` with ``n`` in the wout's convention (times nfp)."""
    with netcdf_file(data_file(WOUT), "r", mmap=False) as f:
        nfp = int(f.variables["nfp"].data)
        xm, xn = f.variables["xm"].data.astype(int), f.variables["xn"].data.astype(int)
        R, Z = f.variables["rmnc"].data[-1].copy(), f.variables["zmns"].data[-1].copy()
        phiedge = float(f.variables["phi"].data[-1])
    lines = [f"NFP = {nfp}", f"PHIEDGE = {phiedge!r}", "AM = 0.0"]
    lines += [f"RBC({n // nfp},{m}) = {float(r)!r}  ZBS({n // nfp},{m}) = {float(z)!r}" for m, n, r, z in zip(xm, xn, R, Z)]
    path = tmp_path / "input.li383_boundary"
    path.write_text("&INDATA\n" + "\n".join(lines) + "\n/\n")
    return str(path), np.stack([xm, xn, R, Z], axis=1), nfp, phiedge


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


def test_fill_meets_boundary_and_axis(tmp_path):
    """The namelist's rows read back as the wout's boundary, and the fill is the axis at rho = 0 and the boundary at
    rho = 1 to round-off, mode by mode (numpy, milliseconds)."""
    path, boundary, nfp, _ = _write_namelist(tmp_path)
    rows, axis = read_indata(path).boundary_rows()
    rows = np.asarray(rows)
    assert axis == []
    order = np.lexsort((boundary[:, 1], boundary[:, 0]))
    assert np.array_equal(rows[:, :2], boundary[order, :2])
    assert np.max(np.abs(rows[:, 2:] - boundary[order, 2:])) == 0.0
    axis = np.array([[0, 1.45, 0.0], [nfp, 0.05, -0.02]])
    m, n, R, Z = fill_rows(rows, axis, np.array([0.0, 1.0]))
    on_axis = np.where(m == 0, 1.0, 0.0)
    ax_R = on_axis * np.array([dict((int(k), r) for k, r, _ in axis).get(int(k), 0.0) for k in n])
    ax_Z = on_axis * np.array([dict((int(k), z) for k, _, z in axis).get(int(k), 0.0) for k in n])
    table = {(int(a), int(b)): (r, z) for a, b, r, z in rows}
    eps = np.finfo(np.float64).eps
    assert np.max(np.abs(R[0] - ax_R)) <= 4 * eps and np.max(np.abs(Z[0] - ax_Z)) <= 4 * eps
    assert np.max(np.abs(R[1] - [table.get((int(a), int(b)), (0.0, 0.0))[0] for a, b in zip(m, n)])) <= 4 * eps
    assert np.max(np.abs(Z[1] - [table.get((int(a), int(b)), (0.0, 0.0))[1] for a, b in zip(m, n)])) <= 4 * eps


def test_guessed_axis_is_the_best_unfolded_fill(tmp_path):
    """In each of four planes per field period the fill about the guessed axis is unfolded (``J / rho`` positive on a
    fine grid), and moving the axis point by 1 cm in R or Z lowers the mean of ``log(J / rho)``, the objective the
    guess maximises (li383, minor radius about 0.3 m: the axis of the Fourier fit is within the grid search's
    resolution of the planes' maxima). 11 s, numpy."""
    path, _, nfp, _ = _write_namelist(tmp_path)
    rows = np.asarray(read_indata(path).boundary_rows()[0])
    axis = guess_axis(rows, nfp)
    for v in 2.0 * np.pi / nfp * np.arange(4) / 4:
        point = np.array([np.cos(axis[:, 0] * v) @ axis[:, 1], -np.sin(axis[:, 0] * v) @ axis[:, 2]])
        J = _section_jacobian(rows, point, v)
        assert J.min() > 0.0
        best = np.mean(np.log(J))
        for d in ((0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01)):
            J_d = _section_jacobian(rows, point + np.array(d), v)
            assert J_d.min() <= 0.0 or np.mean(np.log(J_d)) < best


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    """The sequence of the namelist (the fill about the guessed axis) with its harmonic forms, the boundary rows, nfp
    and PHIEDGE. Its build costs 118 s in float64 and 165 s in float32 on four CPU cores, the rest of the module
    15 s."""
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    path, boundary, nfp, phiedge = _write_namelist(tmp_path_factory.mktemp("fill"))
    seq = build_sequence(path, NS, P, graded=True)
    compute_nullspaces(seq, verbose=False)
    return seq, boundary, nfp, phiedge


def test_filled_map_meets_the_boundary_with_positive_jacobian(filled):
    """The spline map of the fill has a positive Jacobian at every quadrature point (smallest det DPhi 1.6e-2), and at
    r = 1 it lies on the namelist's boundary up to the projection onto the (16, 32, 12) degree-2 splines: 6.5e-4 m at
    most on li383 in both precisions (bound 1e-3 m, minor radius about 0.3 m)."""
    seq, boundary, nfp, _ = filled
    assert float(jnp.min(seq.jacobian_j)) > 0.0
    t, z = np.meshgrid(np.arange(64) / 64, np.arange(12) / 12, indexing="ij")
    x = jnp.asarray(np.stack([np.ones(t.size), t.ravel(), z.ravel()], axis=1))
    RZ = np.asarray(jax.vmap(seq.map.section)(x))
    u = 2.0 * np.pi * t.ravel() * (-1.0 if seq.equilibrium["theta_reversed"] else 1.0)
    v = 2.0 * np.pi * z.ravel() / nfp
    arg = boundary[:, 0][None, :] * u[:, None] - boundary[:, 1][None, :] * v[:, None]
    R_b, Z_b = np.cos(arg) @ boundary[:, 2], np.sin(arg) @ boundary[:, 3]
    miss = float(np.max(np.hypot(RZ[:, 0] - R_b, RZ[:, 1] - Z_b)))
    print(f"\n  min det DPhi {float(jnp.min(seq.jacobian_j)):.3e}, max distance from the boundary at r = 1 {miss:.2e} m")
    assert miss < 1e-3


def test_vacuum_start_has_the_flux_and_no_divergence_or_current(filled):
    """The start field of the namelist is the harmonic 2-form: its flux through every quadrature plane
    ``zeta = const`` is PHIEDGE to round-off (the planes' fluxes agree because div B = 0 and B . n = 0), its divergence
    is round-off, and its weak current ``M_1^-1 D_1^T M_2 B`` is at the solver tolerance or the round-off of the
    precision. Measured: the planes' fluxes 1.9 eps off PHIEDGE in both precisions (bound 10 eps), ``||div B|| / ||B||``
    11.5 eps in float64 and 1.1 eps in float32 (bound 100 eps), ``h_r ||J|| / ||B||`` 6.6e-10 in float64 (bound the
    solve tolerance 1e-8) and 5.4e-7 = 4.5 eps in float32 (bound 100 eps)."""
    from mrx.relaxation.initial_conditions import initial_field
    from mrx.relaxation.loop import radial_cell_sq
    from mrx.relaxation.physics import current

    seq, _, _, phiedge = filled
    B, info = initial_field(seq)
    assert info["kind"] == "indata"
    q = seq.quad
    density = seq.odd.evaluate_at_quadrature(B, 2)[:, 2].reshape(q.shape)
    planes = jnp.einsum("i,j,ijk->k", q.w_x, q.w_y, density)
    flux = float(jnp.max(jnp.abs(planes - phiedge))) / abs(phiedge)
    div = info["div"] / float(seq.odd.l2_norm(B, 2))
    J = current(B, seq)
    ratio = float(jnp.sqrt(radial_cell_sq(seq)) * seq.odd.l2_norm(J, 1) / seq.odd.l2_norm(B, 2))
    print(f"\n  flux per plane {flux:.2e} off PHIEDGE ({flux / eps():.1f} eps), div {div:.2e} ({div / eps():.1f} eps), "
          f"h_r |J| / |B| {ratio:.2e} (solve tol {seq.tol:.0e})")
    assert flux < eps(10) and div < eps(100) and ratio < max(seq.tol, eps(100))
