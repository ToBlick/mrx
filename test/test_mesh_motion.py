"""The motion of the mesh towards the flux surfaces changes the map and not the physical field.

On the li383 fixture mesh (16,32,12) the file's field is the VMEC equilibrium, whose logical surfaces are its flux
surfaces and whose magnetic axis is the mesh axis: a move hardly moves that mesh. On the same mesh with its axis
displaced by :data:`SHIFT` (``d (1 - r^2)``, the fill's axis shift) the undo transport
(:func:`mrx.relaxation.mesh_motion.undrag`) onto a moved map and back returns the field to its time error, with
``div B`` and the toroidal flux at round-off each way. The harmonic section brings ``zeta`` onto the harmonic angle and
a flux move keeps it there. The cut by the planes normal to the axis keeps the file's surfaces and field. The moves of a run, their invariants and the axis they find are checked on the run of
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
from mrx.relaxation.initial_conditions import _clebsch_potential, initial_field, toroidal_flux
from mrx.relaxation.physics import compute_divergence_norm
from mrx.relaxation.shells import label

#: The displacement of the axis of the test mesh along X1, in metres (the first radial cell is 2.9 cm).
SHIFT = 0.04
#: |grad zeta - h| / |h| on the planes phi = const at least, and after the harmonic moves and a flux move at most, and
#: the wall's move by the harmonic moves in metres: 0.30, 1.3e-6 and 3.2e-6 in float64, 2.9e-6 and 4.1e-6 in float32
#: (four moves), 3.0e-3 m (a tenth of the 2.9 cm first radial cell, the spline of the wall on 16 radial splines)
#: measured (2026-11-19).
PHI_DRIFT, HARMONIC_DRIFT, HARMONIC_WALL = 0.1, 1e-5, 1e-2
#: The axis-normal cut of the fixture's map: the largest tilt of the sections against the axis before (at least) and
#: after (at most), in degrees, the largest move of a point off its surface ``r = const`` of the file's map (in ``r``,
#: the first radial cell is 0.07) and the relative rms over random points of the file's field on the cut map against
#: that on the file's map, carried (at most) and dragged with the points (at least, the contrast): 10.7, 6.4e-3,
#: 7.8e-3, 5.9e-3 and 2.4e-2 measured in float64 (2026-11-20).
CUT_TILT_BEFORE, CUT_TILT_AFTER, CUT_SURFACE, CUT_CARRIED, CUT_DRAGGED = 5.0, 2e-2, 2e-2, 1.2e-2, 1.5e-2
#: The field after an undrag move there and back against the start: 3.7e-6 measured in float64, 4.0e-6 in float32
#: (SSP-RK3 at the relaxation's CFL cap, 36 substeps each way, li383 (16,32,12) with the axis displaced by 4 cm, div B
#: 2.6e-15 and 1.4e-6 relative, 2026-11-18).
UNDRAG_ROUND_TRIP = 1e-5


@pytest.fixture(scope="module")
def mesh():
    """``(seq, Phi, B)``: a sequence of the fixture's discretisation of its own on the file's sections, which the
    moves change, its map and the file's field. The harmonic moves start from the file's sections, and the
    axis-normal cut is checked against them."""
    seq = build_sequence(GEOMETRY, NS, P, symmetry="stellarator", graded=True, zeta="phi")
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


def _file_field(st, Phi):
    """The file's field ``B = DPhi curl A' / det DPhi`` of the state ``st`` on the map ``Phi``, the curl of its
    potential (:func:`mrx.relaxation.initial_conditions._clebsch_potential`) taken exactly, at logical points."""
    A = _clebsch_potential(st, Phi.nfp, Phi.basis)

    def B(y):
        D = jax.jacfwd(A)(y)
        DP = jax.jacfwd(Phi)(y)
        return DP @ jnp.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / jnp.linalg.det(DP)
    return jax.vmap(B)


def test_axis_normal_sections(mesh):
    """The fixture's map cut by the planes normal to its axis (:func:`~mrx.relaxation.mesh_motion.axis_normal_map`):
    the tilt of the sections at the axis falls from :data:`CUT_TILT_BEFORE` or more to :data:`CUT_TILT_AFTER`, every
    point stays on its surface ``r = const`` of the file's map to :data:`CUT_SURFACE` in ``r`` (the error of the
    splines of the cut), ``det DPhi > 0`` at the quadrature points, and the file's field on the cut map with the
    shift of ``zeta`` is the field on the file's map at the same points to :data:`CUT_CARRIED`. Without the shift it is
    off by :data:`CUT_DRAGGED` or more: the points slide, the field must not."""
    seq, Phi, _ = mesh
    new, shift, info = mm.axis_normal_map(seq, Phi)
    rng = np.random.default_rng(0)
    x = jnp.asarray(np.column_stack([rng.uniform(0.05, 1.0, 256), rng.uniform(0.0, 1.0, 256),
                                     rng.uniform(0.0, 1.0, 256)]))
    f = jax.vmap(lambda y: new.basis.contract(shift, y))(x)
    x_old, _ = mm.locate(Phi, jax.vmap(new)(x), x.at[:, 2].add(f))
    surface = float(jnp.max(jnp.abs(x_old[:, 0] - x[:, 0])))
    st = seq.equilibrium
    B = _file_field(st, Phi)(x_old)

    def rms(C):
        return float(jnp.sqrt(jnp.mean(jnp.sum((C - B) ** 2, axis=1)) / jnp.mean(jnp.sum(B ** 2, axis=1))))
    carried = rms(_file_field(dict(st, zeta_shift=shift), new)(x))
    dragged = rms(_file_field(st, new)(x))
    print(f"\n  tilt {info['tilt_before']:.2f} -> {info['tilt_after']:.1e} deg, off the surfaces {surface:.1e} in r, "
          f"det DPhi >= {info['det_range'][0]:.2e}, field carried {carried:.1e}, dragged {dragged:.1e}")
    assert info["tilt_before"] > CUT_TILT_BEFORE and info["tilt_after"] < CUT_TILT_AFTER
    assert surface < CUT_SURFACE
    assert info["det_range"][0] > 0
    assert carried < CUT_CARRIED and dragged > CUT_DRAGGED
