"""The motion of the mesh's pole onto the magnetic axis changes the map, and the field is carried onto it.

On the fixture the field is the projected exact equilibrium shearedE, whose magnetic axis is the mesh axis: the axis
finder finds it on the pole and on the exact axis, and a move hardly moves that mesh. The field carried by the relaxation's induction
(:func:`~mrx.relaxation.mesh_motion.transfer`) onto the map with its pole displaced by :data:`SHIFT` and back keeps
``div B`` and the toroidal flux at round-off and returns to the time error of its substeps. On the displaced map the
carried field's axis is off the pole, and one move (:func:`~mrx.relaxation.mesh_motion.axis_move`) puts the pole back
on it. The cut by the planes normal to the axis keeps the file's surfaces and field. Every map is installed on the
fixture with its preconditioners (:func:`test.conftest.map_installed`).
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.equilibria import build_map
from mrx.precision import eps
from mrx.relaxation import mesh_motion as mm
from mrx.relaxation.initial_conditions import _clebsch_potential, toroidal_flux
from mrx.relaxation.physics import compute_divergence_norm
from test.conftest import LENGTH, map_installed

#: The displacement of the pole of the test map along X1, a fraction of the minor radius, about one first radial cell.
SHIFT = 0.12
#: The field after a transfer there and back against the start: 1.14e-6 measured (2026-11-22), 1.24e-6 in float64 with the induction of
#: h = Pi_1 B (36 substeps, div B 2.3e-15, flux 1.6e-15 relative, 2026-11-21), 4.0e-6 in float32 with the old
#: induction (2026-11-18).
ROUND_TRIP = 1e-5
#: The axis-normal cut of the fixture's map: the largest tilt of the sections against the axis before (at least) and
#: after (at most), in degrees, the largest move of a point off its surface ``r = const`` of the file's map (in ``r``,
#: the first radial cell is 0.07) and the relative rms over random points of the file's field on the cut map against
#: that on the file's map, carried (at most) and dragged with the points (at least, the contrast): 10.7, 6.4e-3,
#: 7.8e-3, 5.9e-3 and 2.4e-2 measured in float64 (2026-11-20).
CUT_TILT_BEFORE, CUT_TILT_AFTER, CUT_SURFACE, CUT_CARRIED, CUT_DRAGGED = 5.0, 2e-2, 2e-2, 1.2e-2, 1.5e-2
#: The distance of the file's axis from its pole in first radial cells.
AXIS_ON_POLE = 2e-2
#: The distance of the magnetic axis from the pole after one move from :data:`SHIFT`, in first radial cells: 7.1e-3
#: measured in float64, from 1.35 before (2026-11-22).
AXIS_AFTER_MOVE = 1e-2


def _locate(Phi, X, x, steps=10):
    """``(x', miss)``: the logical points ``x'`` with ``Phi(x') = X`` (Cartesian, ``(n, 3)``) and ``|Phi(x') - X|``,
    by Newton from the logical points ``x`` in the chart ``(r cos 2 pi theta, r sin 2 pi theta, zeta)``, which is
    smooth at the axis. ``zeta`` of ``x'`` may leave ``[0, 1)`` by the move of a point out of its section."""
    def one(xq, Xq):
        F = lambda p: mm.extended(Phi, mm._chart(p[:2], p[2])) - Xq   # noqa: E731
        t = 2.0 * jnp.pi * xq[1]
        p = jnp.stack([xq[0] * jnp.cos(t), xq[0] * jnp.sin(t), xq[2]])
        for _ in range(steps):
            p = p - jnp.linalg.solve(jax.jacfwd(F)(p), F(p))
        return mm._chart(p[:2], p[2]), jnp.linalg.norm(F(p))
    return jax.jit(jax.vmap(one))(x, X)


def _shifted(seq, a):
    """The fixture's map with its pole displaced by :data:`SHIFT` times the minor radius ``a`` along ``X1`` (a
    constant offset, so its ``zeta`` spline coefficients are that offset)."""
    d = np.zeros((seq.basis_0.Lambda[2].n, 2))
    d[:, 0] = SHIFT * a
    return mm.moved(seq.map, mm.tangent(seq, mm.axis_step(seq, d)))


def _axis_cells(seq, B):
    """The largest distance of the magnetic axis of ``B`` from the pole, in first radial cells."""
    d, h1 = mm.axis_offset(seq, mm.magnetic_axis(seq, B)[0])
    return float(np.max(np.linalg.norm(d, axis=1))) / h1


def test_the_axis_is_the_exact_axis_on_the_pole(eq, seq, b0, minor):
    """The projected exact field on the fixture: the axis finder (:func:`~mrx.relaxation.mesh_motion.magnetic_axis`)
    puts the magnetic axis within :data:`AXIS_ON_POLE` first radial cells of the pole, a move moves the mesh by less
    than that, and the axis curve is the exact axis ``k = 0`` within :data:`~test.conftest.LENGTH` of the minor
    radius."""
    _, rec = mm.axis_displacement(seq, b0)
    zetas, points = mm.axis_curve(seq, b0)
    # the exact axis in the plane of each point's own azimuth: the cut map's zeta is not the azimuth
    off = float(np.max(np.linalg.norm(points - eq.axis(np.arctan2(points[:, 1], points[:, 0])), axis=1))) / minor
    print(f"\n  axis {rec['axis_cells']:.2e} first cells off the pole ({rec['axis']['newton']} Newton steps), largest "
          f"move {rec['move_m']:.2e} m, first cell {rec['h1_m']:.2e} m, {off:.2e} of the minor radius off the exact axis")
    assert rec["axis_cells"] < AXIS_ON_POLE
    assert rec["move_m"] < AXIS_ON_POLE * rec["h1_m"]
    assert off < LENGTH


def test_transfer_there_and_back_and_a_move_onto_the_axis(seq, b0, minor):
    """The file's field carried by :func:`~mrx.relaxation.mesh_motion.transfer` onto the map with the pole displaced
    by :data:`SHIFT` times the minor radius: ``||div B|| / ||B||`` and the toroidal flux hold to round-off, and carried
    back the field returns to its coefficients within :data:`ROUND_TRIP`, the time error of the transport (the two ways
    run along the same maps in opposite directions). On the displaced map the carried field's axis is off the pole by
    about the shift, and one move (:func:`~mrx.relaxation.mesh_motion.axis_move`) puts the pole within
    :data:`AXIS_AFTER_MOVE` first radial cells of the axis of the field it carries, with ``div B`` at round-off."""
    Phi = seq.map
    flux = float(toroidal_flux(seq, b0))
    new = _shifted(seq, minor)
    B1, rec = mm.transfer(seq, b0, new)
    with map_installed(seq, new, harmonic=True):
        div1 = float(compute_divergence_norm(B1, seq) / seq.odd.l2_norm(B1, 2))
        flux1 = float(toroidal_flux(seq, B1)) / flux - 1.0
        B2, _ = mm.transfer(seq, B1, Phi)
        before = _axis_cells(seq, B1)
        B3, moved = mm.axis_move(seq, B1)
        after = _axis_cells(seq, B3)
        div3 = float(compute_divergence_norm(B3, seq) / seq.odd.l2_norm(B3, 2))
    back = float(seq.odd.l2_norm(B2 - b0, 2) / seq.odd.l2_norm(b0, 2))
    print(f"\n  {rec['substeps']} substeps ({rec['cells']:.2f} cells), div {div1:.1e}, flux {flux1:+.1e}, there and "
          f"back {back:.2e}\n  axis {before:.3e} first cells off the pole on the displaced map, {after:.3e} after the "
          f"move (x{moved['lam']:g}, {moved['substeps']} substeps), div {div3:.1e}")
    assert div1 < 1e2 * eps() and abs(flux1) < 1e2 * eps()
    assert back < ROUND_TRIP
    assert after < AXIS_AFTER_MOVE < before
    assert div3 < 1e2 * eps()


def _file_field(st, Phi):
    """The file's field ``B = DPhi curl A' / det DPhi`` of the state ``st`` on the map ``Phi``, the curl of its
    potential (:func:`mrx.relaxation.initial_conditions._clebsch_potential`) taken exactly, at logical points."""
    A = _clebsch_potential(st, Phi.nfp, Phi.basis)

    def B(y):
        D = jax.jacfwd(A)(y)
        DP = jax.jacfwd(Phi)(y)
        return DP @ jnp.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / jnp.linalg.det(DP)
    return jax.vmap(B)


def test_axis_normal_sections(seq):
    """The fixture's map cut by the planes normal to its axis (:func:`~mrx.relaxation.mesh_motion.axis_normal_map`):
    the tilt of the sections at the axis falls from :data:`CUT_TILT_BEFORE` or more to :data:`CUT_TILT_AFTER`, every
    point stays on its surface ``r = const`` of the file's map to :data:`CUT_SURFACE` in ``r`` (the error of the
    splines of the cut), ``det DPhi > 0`` at the quadrature points, and the file's field on the cut map with the
    shift of ``zeta`` is the field on the file's map at the same points to :data:`CUT_CARRIED`. Without the shift it is
    off by :data:`CUT_DRAGGED` or more: the points slide, the field must not."""
    Phi = build_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)[0]
    new, shift, info = mm.axis_normal_map(seq, Phi)
    rng = np.random.default_rng(0)
    x = jnp.asarray(np.column_stack([rng.uniform(0.05, 1.0, 256), rng.uniform(0.0, 1.0, 256),
                                     rng.uniform(0.0, 1.0, 256)]))
    f = jax.vmap(lambda y: new.basis.contract(shift, y))(x)
    x_old, _ = _locate(Phi, jax.vmap(new)(x), x.at[:, 2].add(f))
    surface = float(jnp.max(jnp.abs(x_old[:, 0] - x[:, 0])))
    st = {k: v for k, v in seq.equilibrium.items() if k != "zeta_shift"}
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
