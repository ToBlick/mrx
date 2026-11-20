"""The motion of the mesh towards the flux surfaces changes the map and not the physical field.

On the li383 fixture mesh (16,32,12) the file's field is the VMEC equilibrium, whose logical surfaces are its flux
surfaces and whose magnetic axis is the mesh axis: a move hardly moves that mesh. On the same mesh with its axis
displaced by :data:`SHIFT` (``d (1 - r^2)``, the fill's axis shift) the undo transport
(:func:`mrx.relaxation.mesh_motion.undrag`) onto a moved map and back returns the field to its time error, with
``div B`` and the toroidal flux at round-off each way. The harmonic section brings ``zeta`` onto the harmonic angle and
a flux move keeps it there. The moves of a run, their invariants and the axis they find are checked on the run of
test_relax_run.py. The moves change the sequence, so the module builds its own.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test.conftest import GEOMETRY, NS, P
from mrx.flux_label import ANISOTROPY
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces, zeta_harmonic_distance
from mrx.precision import eps
from mrx.relaxation import mesh_motion as mm
from mrx.relaxation.initial_conditions import initial_field, toroidal_flux
from mrx.relaxation.physics import compute_divergence_norm
from mrx.relaxation.shells import label

#: The displacement of the axis of the test mesh along X1, in metres (the first radial cell is 2.9 cm).
SHIFT = 0.04
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
