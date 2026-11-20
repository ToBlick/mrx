"""The map from a boundary alone (:mod:`mrx.equilibria.fill`) and the vacuum start on it.

The boundary is that of the tracked li383 wout, written as a VMEC input namelist with no axis, so the reader fills it
about the axis :func:`~mrx.equilibria.fill.guess_axis` places. The fill reaches the boundary and the axis exactly, and
the guessed axis is a maximum of its objective and leaves the fill unfolded. The sequence of the fill and its vacuum
start are checked in test_relax_run.py, whose run starts there.
"""
import numpy as np
from scipy.io import netcdf_file

from mrx.equilibria.fill import fill_rows, guess_axis
from mrx.equilibria.indata import read_indata
from mrx.paths import data_file

WOUT = "data/wout_li383_low_res_reference.nc"
NS, P = (16, 32, 12), 2


def _write_namelist(tmp_path, profiles=("AM = 0.0",), name="input.li383_boundary"):
    """The boundary of :data:`WOUT` and its total toroidal flux as a namelist ``name``, without an axis, with the lines
    ``profiles``. Returns its path and the boundary rows ``(m, n, R_mn, Z_mn)`` with ``n`` in the wout's convention
    (times nfp), nfp and PHIEDGE."""
    with netcdf_file(data_file(WOUT), "r", mmap=False) as f:
        nfp = int(f.variables["nfp"].data)
        xm, xn = f.variables["xm"].data.astype(int), f.variables["xn"].data.astype(int)
        R, Z = f.variables["rmnc"].data[-1].copy(), f.variables["zmns"].data[-1].copy()
        phiedge = float(f.variables["phi"].data[-1])
    lines = [f"NFP = {nfp}", f"PHIEDGE = {phiedge!r}", *profiles]
    lines += [f"RBC({n // nfp},{m}) = {float(r)!r}  ZBS({n // nfp},{m}) = {float(z)!r}" for m, n, r, z in zip(xm, xn, R, Z)]
    path = tmp_path / name
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
