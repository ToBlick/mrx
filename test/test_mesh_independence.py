"""The profile control does not depend on the mesh.

The pressure, the current and the transform the relaxation is steered to are functions of the flux label ``s`` alone,
in a profile space of their own (:func:`mrx.relaxation.shells.profile_basis`). So two meshes of one domain must end
at the same field when they start from the same field and are given the same targets. The second mesh here is the
fixture's wout with its magnetic axis moved 4 cm inwards, the shift tapered to zero at the wall: the same domain,
with logical surfaces that are not the field's surfaces and a logical axis 4 cm off the magnetic one. Both start from
the vacuum field of the domain with the file's toroidal flux and are given the file's net toroidal current inside
every surface as the target.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import netcdf_file

from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.flux_label import ANISOTROPY
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.precision import DTYPE
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.shells import label, logical_points
from mrx.relaxation.twist import build_slips, current_error, current_on_label, iota_fit
from test.conftest import NS, P
from test.test_twist import CELLS

#: The Newton steps, in chunks with the label and the slips rebuilt before each.
STEPS, CHUNK = 20, 10
#: The axis shift in metres.
SHIFT = 0.04
#: The measured differences in the test's docstring, with their bounds.
IOTA_DIFF, B_DIFF, CURRENT_ERROR = 1e-2, 1.5e-2, 1e-3


def _shifted_wout(src, dst, dR):
    """The wout ``src`` with the magnetic axis moved by ``dR`` in R, tapered as ``1 - s`` so that the boundary stays,
    written to ``dst`` with what MRX reads."""
    v = netcdf_file(src, "r", mmap=False).variables
    names = ("ns", "nfp", "mnmax", "lasym__logical__", "version_", "xm", "xn", "rmnc", "zmns", "lmns", "phi", "iotaf",
             "presf")
    v = {k: np.array(v[k][()]) for k in names}
    ns = int(v["ns"])
    i = np.nonzero((v["xm"] == 0) & (v["xn"] == 0))[0][0]
    v["rmnc"][:, i] += (1.0 - np.arange(ns) / (ns - 1)) * dR
    with netcdf_file(dst, "w") as f:
        f.createDimension("radius", ns)
        f.createDimension("mn_mode", int(v["mnmax"]))
        for k in ("ns", "nfp", "mnmax", "lasym__logical__"):
            f.createVariable(k, "i4", ())[()] = int(v[k])
        f.createVariable("version_", "f8", ())[()] = float(v["version_"])
        for k in ("xm", "xn"):
            f.createVariable(k, "f8", ("mn_mode",))[:] = v[k]
        for k in ("rmnc", "zmns", "lmns"):
            f.createVariable(k, "f8", ("radius", "mn_mode"))[:] = v[k]
        for k in ("phi", "iotaf", "presf"):
            f.createVariable(k, "f8", ("radius",))[:] = v[k]


def _vacuum(seq, flux):
    """The vacuum field of the domain of ``seq`` with the toroidal flux ``flux`` per unit ``zeta``."""
    h = seq.odd.nullspace(2)[0]
    return h * (flux / jnp.sum(seq.quad.w * seq.odd.evaluate_at_quadrature(h, 2)[:, 2]))


def _relax_to_current(seq, B, mu_star):
    """``(B, T, s_q)`` after STEPS Newton steps with the current target ``mu_star``, the label rebuilt per chunk."""
    ts, T = TimeStepper(seq=seq, newton=True), None
    state = initial_state(B, ts)
    for _ in range(STEPS // CHUNK):
        _, T, _ = label(seq, state.B_n, ANISOTROPY, T)
        state = relax(state, ts.with_slips(build_slips(seq, state.B_n, T, mu_star)[0]), steps=CHUNK, chunk=CHUNK,
                      verbose=False).state
    s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
    return state.B_n, T, s_q


def test_the_same_current_gives_the_same_field_on_two_meshes(seq, b0, tmp_path):
    """The current target alone from the vacuum, on the fixture's mesh and on the mesh with the axis moved 4 cm
    inwards. After twenty Newton steps each meets the target (8.3e-5 and 8.2e-5) and the two transforms on their own
    labels agree on ``s`` in 0.05 .. 0.95 to 3.0e-3, the fields at common physical points to 6.5e-3, the size of the
    two meshes' discretisation difference at this resolution. The bounds are CURRENT_ERROR, IOTA_DIFF and B_DIFF. On
    CELLS cells of the profile space. 1278 s on four CPU cores after the fixture (2026-11-19). Skipped in float32,
    where a Newton step costs three times as much on the CPU."""
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    path = str(tmp_path / "wout_shifted.nc")
    _shifted_wout(seq.equilibrium["path"], path, -SHIFT)
    seq2 = build_sequence(path, NS, P, symmetry="stellarator", graded=True)
    compute_nullspaces(seq2, verbose=False)
    flux = float(jnp.sum(seq.quad.w * seq.odd.evaluate_at_quadrature(b0, 2)[:, 2]))
    _, T0, _ = label(seq, b0, ANISOTROPY)
    mu_star = current_on_label(seq, b0, T0, CELLS)

    s = np.linspace(0.05, 0.95, 19)
    out = []
    for sq in (seq, seq2):
        B, T, s_q = _relax_to_current(sq, _vacuum(sq, flux), mu_star)
        err = current_error(current_on_label(sq, B, T, CELLS), mu_star)
        out.append((sq, B, err, np.abs(iota_fit(sq, B, T, np.asarray(s_q), 6)(s))))
    (_, B1, err1, i1), (_, B2, err2, i2) = out
    # the fields at the points of a logical grid of the first mesh, located in the second
    r, t, z = np.meshgrid(np.linspace(0.05, 0.95, 10), np.arange(16) / 16, np.arange(8) / 8, indexing="ij")
    x1 = jnp.asarray(np.stack([r.ravel(), t.ravel(), z.ravel()], axis=1))
    x2, miss = logical_points(seq2, jax.vmap(seq.map)(x1))

    def physical(sq, B, x):
        return jax.vmap(Pushforward(DiscreteFunction(B, sq.odd.basis_2, sq.odd.E(2)), sq.map, 2))(x)
    F1, F2 = np.asarray(physical(seq, B1, x1)), np.asarray(physical(seq2, B2, x2))
    b_diff = float(np.sqrt(np.sum((F1 - F2) ** 2) / np.sum(F1 ** 2)))
    i_diff = float(np.abs(i1 - i2).max())
    print(f"\n  current error {err1:.2e} aligned, {err2:.2e} shifted. iota on the labels differs by {i_diff:.2e} "
          f"(aligned {i1.min():.4f} .. {i1.max():.4f}). B at common points {b_diff:.2e}, location miss "
          f"{float(jnp.max(miss)):.1e} m")
    assert err1 < CURRENT_ERROR and err2 < CURRENT_ERROR
    assert i_diff < IOTA_DIFF
    assert b_diff < B_DIFF
