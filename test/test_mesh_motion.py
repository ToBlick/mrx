"""The motion of the mesh towards the flux surfaces changes the map and not the physical field.

On the li383 fixture mesh (16,32,12) the file's field is the VMEC equilibrium, whose logical surfaces are its flux
surfaces and whose magnetic axis is the mesh axis: a move hardly moves that mesh. The same mesh with its axis displaced
by :data:`SHIFT` (``d (1 - r^2)``, the fill's axis shift) carries the same physical field, transported onto it
(:func:`mrx.relaxation.mesh_motion.undrag`). One move puts the mesh axis back on the label's maximum, which is the
file's axis, keeps ``div B`` and the toroidal flux to round-off (``B . n = 0`` holds by the space, the field is a
Dirichlet 2-form) and leaves the physical field at fixed physical points within the transport's error. A relaxation
from that mesh with the motion ends where the relaxation on the file's mesh ends. The moves change the sequence, so the
module builds its own.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test.conftest import GEOMETRY, NS, P
from mrx.flux_label import ANISOTROPY
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces, zeta_harmonic_distance
from mrx.precision import DTYPE, eps
from mrx.relaxation import mesh_motion as mm
from mrx.relaxation.initial_conditions import initial_field, toroidal_flux
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import compute_divergence_norm
from mrx.relaxation.shells import label

#: The displacement of the axis of the test mesh along X1, in metres (the first radial cell is 2.9 cm).
SHIFT = 0.04
#: The change of the physical field by one move at fixed physical points: 1.0e-2 measured, near the difference of the
#: vacuum fields of the two maps there, 7.0e-3 (2026-11-18).
DB_PHYS = 1.5e-2
#: The label's maximum after one move against the file's axis, in first radial cells: 0.005 measured (2026-11-18).
AXIS_FILE = 0.05
#: The Newton steps of the run test, and the agreement of its two end fields and residuals: 1.5e-4 and a factor 0.92
#: measured (2026-11-18).
STEPS, SAME_STATE, RESID_FACTOR = 10, 1e-3, 2.0
#: |grad zeta - h| / |h| on the planes phi = const at least, and after the harmonic moves and a flux move at most, and
#: the wall's move by the harmonic moves in metres: 0.30, 1.3e-6 and 3.2e-6 in float64, 2.9e-6 and 4.1e-6 in float32
#: (four moves), 3.0e-3 m (a tenth of the 2.9 cm first radial cell, the spline of the wall on 16 radial splines)
#: measured (2026-11-19).
PHI_DRIFT, HARMONIC_DRIFT, HARMONIC_WALL = 0.1, 1e-5, 1e-2
#: The field after an undrag move there and back against the start: 3.7e-6 measured in float64, 4.0e-6 in float32
#: (SSP-RK3 at the relaxation's CFL cap, 36 substeps each way, li383 (16,32,12) with the axis displaced by 4 cm, div B
#: 2.6e-15 and 1.4e-6 relative, 2026-11-18).
UNDRAG_ROUND_TRIP = 1e-5


@pytest.fixture(scope="module")
def mesh():
    """``(seq, Phi, B)``: a sequence of the fixture's discretisation of its own, which the moves change, its map and
    the file's field."""
    seq = build_sequence(GEOMETRY, NS, P, symmetry="stellarator", graded=True)
    compute_nullspaces(seq, verbose=False)
    return seq, seq.map, initial_field(seq)[0]


def _label(seq, B, guess=None):
    return label(seq, B, ANISOTROPY, guess)


def _displaced(seq, Phi, B):
    """Install the map ``Phi`` with its axis displaced by :data:`SHIFT` and return the field ``B`` of ``Phi`` on it."""
    d = np.zeros((mm._planes(seq).size, 2))
    d[:, 0] = SHIFT
    shifted = mm.moved(Phi, mm.tangent(seq, mm.axis_step(seq, d)))
    B_shifted, _ = mm.undrag(seq, B, shifted)
    mm.install_map(seq, shifted)
    return B_shifted


def test_an_aligned_mesh_does_not_move(mesh):
    """The VMEC field on its own mesh: the largest move is 0.46 % of the first radial cell (0.14 mm against 2.9 cm,
    2026-11-18), the label's maximum is 0.5 % of that cell off the mesh axis, against the bounds 2 % and
    :data:`~mrx.relaxation.mesh_motion.AXIS_TOL`."""
    seq, Phi, B = mesh
    mm.install_map(seq, Phi)
    _, T, _ = _label(seq, B)
    _, rec = mm.move(seq, B, T)
    mm.install_map(seq, Phi)
    print(f"\n  largest move {rec['move_m']:.2e} m, first cell {rec['h1_m']:.2e} m, axis {rec['axis_cells']:.2e} cells")
    assert rec["move_m"] < 2e-2 * rec["h1_m"]
    assert rec["axis_cells"] < mm.AXIS_TOL


@pytest.fixture(scope="module")
def displaced(mesh):
    """One move of the file's field on the mesh with the displaced axis: ``(B, rec)``, the field before the move and
    the move's record with ``axis_cells_after``, the label's maximum after it against the mesh axis, and
    ``axis_file_m``, against the file's axis."""
    seq, Phi, B0 = mesh
    mm.install_map(seq, Phi)
    zetas = mm._planes(seq)
    axis_file = np.asarray(jax.vmap(Phi.section)(jnp.asarray(np.stack([0 * zetas, 0 * zetas, zetas], 1))))
    B = _displaced(seq, Phi, B0)
    _, T, _ = _label(seq, B)
    B1, rec = mm.move(seq, B, T)
    T1 = _label(seq, B1, T)[1]
    d1, h1 = mm.axis_offset(seq, T1)
    axis = np.asarray(jax.vmap(seq.map.section)(jnp.asarray(mm.label_maximum(seq, T1, zetas))))
    rec.update(axis_cells_after=float(np.linalg.norm(d1, axis=1).max() / h1),
               axis_file_m=float(np.linalg.norm(axis - axis_file, axis=1).max()))
    mm.install_map(seq, Phi)
    return B, rec


def test_a_move_keeps_the_invariants_and_the_field(displaced):
    """``||div B|| / ||B||`` and the toroidal flux of the moved field to round-off, and the physical field at fixed physical
    points within :data:`DB_PHYS`."""
    _, rec = displaced
    print(f"\n  div {rec['div_rel']:.1e}, flux {rec['flux_rel']:+.1e}, physical field {rec['dB_phys']:.2e}, energy "
          f"{rec['dE_rel']:+.2e}")
    assert rec["div_rel"] < 1e2 * eps()
    assert abs(rec["flux_rel"]) < 1e2 * eps()
    assert rec["dB_phys"] < DB_PHYS


def test_the_axis_step_finds_the_label_maximum(displaced):
    """Before the move the label's maximum is more than a radial cell off the mesh axis. After it, the maximum is on
    the mesh axis to :data:`~mrx.relaxation.mesh_motion.AXIS_TOL` of the first radial cell, and within
    :data:`AXIS_FILE` of that cell of the file's axis."""
    _, rec = displaced
    print(f"\n  axis {rec['axis_cells']:.2f} cells off before, {rec['axis_cells_after']:.3f} after, "
          f"{rec['axis_file_m']:.2e} m from the file's axis")
    assert rec["axis_cells"] > 1.0
    assert rec["axis_cells_after"] < mm.AXIS_TOL
    assert rec["axis_file_m"] < AXIS_FILE * rec["h1_m"]


def test_a_displaced_mesh_relaxes_to_the_aligned_state(mesh):
    """The run of ``scripts/relax.py --mesh.align --mesh.start 0`` from the mesh with the displaced axis: a move,
    then :data:`STEPS` Newton steps, against the same steps on the file's mesh. The two end fields agree at fixed
    physical points to :data:`SAME_STATE`, and the residuals to a factor :data:`RESID_FACTOR`. Skipped in
    float32, where a Newton step costs three times as much on the CPU."""
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    seq, Phi, B0 = mesh
    mm.install_map(seq, Phi)
    B_at = mm._field_sampler(seq)
    ts = TimeStepper(seq=seq, newton=True)
    aligned = relax(initial_state(B0, ts), ts, steps=STEPS, chunk=STEPS, verbose=False)
    reference, _ = B_at(aligned.state.B_n)
    B = _displaced(seq, Phi, B0)
    B, _ = mm.move(seq, B, _label(seq, B)[1])
    ts = TimeStepper(seq=seq, newton=True)
    moved = relax(initial_state(B, ts), ts, steps=STEPS, chunk=STEPS, verbose=False)
    end, _ = B_at(moved.state.B_n)
    mm.install_map(seq, Phi)
    diff = float(np.sqrt(np.sum((end - reference) ** 2) / np.sum(reference ** 2)))
    r_aligned, r_moved = float(aligned.trace["resid"][-1]), float(moved.trace["resid"][-1])
    print(f"\n  end fields differ by {diff:.2e}, residual {r_aligned:.2e} aligned, {r_moved:.2e} displaced and moved")
    assert diff < SAME_STATE
    assert r_moved < RESID_FACTOR * r_aligned


def test_harmonic_moves(mesh):
    """``--geometry.zeta harmonic``: the moves of the fixture's map towards the harmonic toroidal angle
    (:func:`~mrx.relaxation.mesh_motion.harmonic_build`) take ``|grad zeta - h| / |h|`` from :data:`PHI_DRIFT` or
    more on the planes ``phi = const`` below :data:`HARMONIC_DRIFT`, move the wall by less than :data:`HARMONIC_WALL`
    and keep ``det DPhi > 0`` (``set_geometry`` raises otherwise). A flux alignment move of the file's field on the
    new map, tangent to the surfaces ``zeta = const``, keeps the drift below :data:`HARMONIC_DRIFT`."""
    seq, Phi, B0 = mesh
    mm.install_map(seq, Phi)
    before = zeta_harmonic_distance(seq)[0]
    mm.harmonic_build(seq)
    new, drift = seq.map, seq.equilibrium["harmonic"]["drift"]
    t, z = np.meshgrid(np.arange(64) / 64, np.arange(16) / 16, indexing="ij")
    w = jnp.asarray(np.stack([np.ones(t.size), t.ravel(), z.ravel()], 1))
    xw, _ = mm.locate(Phi, jax.vmap(new)(w), w)
    wall = float(jnp.max(jnp.linalg.norm(jax.vmap(Phi)(xw.at[:, 0].set(1.0)) - jax.vmap(new)(w), axis=1)))
    mm.install_map(seq, Phi)
    B, _ = mm.undrag(seq, B0, new)
    mm.install_map(seq, new)
    after = zeta_harmonic_distance(seq)[0]
    _, rec = mm.move(seq, B, _label(seq, B)[1])
    moved = zeta_harmonic_distance(seq)[0]
    mm.install_map(seq, Phi)
    print(f"\n  |grad zeta - h| / |h| {before:.2e} on the phi planes, " + " -> ".join(f"{d:.2e}" for d in drift)
          + f" in {len(drift) - 1} moves, {after:.2e} installed, {moved:.2e} after a flux move of {rec['move_m']:.1e} m, "
          f"wall moved by {wall:.1e} m")
    assert before > PHI_DRIFT
    assert after < HARMONIC_DRIFT and moved < HARMONIC_DRIFT
    assert wall < HARMONIC_WALL


def test_undrag_there_and_back(mesh):
    """The field of the mesh with the displaced axis carried by :func:`~mrx.relaxation.mesh_motion.undrag` onto the
    map of one move and back: each way ``||div B|| / ||B||`` and the toroidal flux hold to round-off, and the field
    returns to its coefficients within :data:`UNDRAG_ROUND_TRIP`, the time error of the transport. The two ways run
    along the same maps in opposite directions."""
    seq, Phi, B0 = mesh
    mm.install_map(seq, Phi)
    B = _displaced(seq, Phi, B0)
    start, flux = seq.map, float(toroidal_flux(seq, B))
    new, _ = mm.displacement(seq, B, _label(seq, B)[1])
    B1, rec = mm.undrag(seq, B, new)
    mm.install_map(seq, new, keep_preconditioners=True)
    div1 = float(compute_divergence_norm(B1, seq) / seq.odd.l2_norm(B1, 2))
    flux1 = float(toroidal_flux(seq, B1)) / flux - 1.0
    B2, _ = mm.undrag(seq, B1, start)
    mm.install_map(seq, start, keep_preconditioners=True)
    back = float(seq.odd.l2_norm(B2 - B, 2) / seq.odd.l2_norm(B, 2))
    mm.install_map(seq, Phi)
    print(f"\n  {rec['substeps']} substeps, div {div1:.1e}, flux {flux1:+.1e}, there and back {back:.2e}")
    assert div1 < 1e2 * eps()
    assert abs(flux1) < 1e2 * eps()
    assert back < UNDRAG_ROUND_TRIP
