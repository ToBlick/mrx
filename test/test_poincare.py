"""The field-line tracer of :mod:`mrx.diagnostics.poincare`.

``b0`` (:func:`mrx.relaxation.initial_conditions.potential_two_form`) is the curl of a vector potential whose
``theta`` and ``zeta`` components depend on ``r`` alone, so ``B^r = 0`` exactly: its field lines stay on their
logical surfaces and wind with the rotational transform of the exact Solov'ev equilibrium. The traced iota of
every regular line matches the exact transform on its surface within a measured band, the discretisation error of the
field, and halving the step moves the lines by less than a measured drift, also with ``b0`` written as a free
2-form (no flux through the wall). A closed-form field, whose lines are
known exactly, pins the crossings forwards and backwards and the cut at the boundary. The sections ``phi = const``
are the sections ``zeta = const`` on a cylindrical map, and on the fixture's map, whose sections are normal to the
axis, they are those of the same field on the file's map. CI runs the suite in float64 and float32, and the
tolerances scale with the precision (:func:`_tol`).
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.diagnostics.poincare import poincare

#: ``max |iota_traced - iota| / max iota`` against the exact transform, 8 lines over 60 periods: 8.3e-4 measured
#: (tokamak (12,16,6), float64, 2026-11-22). A tracer that missed the factor nfp or a wrong winding would be off by O(1).
IOTA_BAND = 1.2e-3
#: The step-halving drift of the same trace in logical radius: 2.9e-5 measured.
DRIFT_BAND = 6e-5


#: The file's field on the fixture's map (sections normal to the axis) and on the file's map, traced from the same
#: lines and cut at the planes ``phi = const``: the largest difference of a regular line's iota, and the largest
#: distance of its first ten crossings as a fraction of the minor radius. 2.4e-5 and 2.5e-2 m = 7.7 % measured in
#: float64, 3.6e-5 and 3.8e-2 m = 11.7 % in float32 (2026-11-22): the transforms agree, the crossings drift apart along
#: the surfaces by the small difference of the two discrete transforms over ten transits.
PHI_IOTA_BAND, PHI_CROSSING_BAND = 5e-5, 0.2


def _tol(f64, f32):
    """The tolerance ``f64`` in float64 and ``f32`` in float32 (CI runs both)."""
    from mrx.precision import DTYPE
    return f64 if DTYPE == np.float64 else f32


def test_traced_iota_is_the_exact_iota(eq, seq, b0):
    """The traced iota of the regular lines is the exact transform of shearedE on their surface (the logical radius
    of the wout is ``sqrt(s)``, which the cut by the planes normal to the axis keeps), and the step is resolved. The field
    is ``b0`` as a free 2-form, the space of fields with a flux through the wall, here zero. The padding lines,
    traced forwards and backwards for more periods than the main lines, stay on their seed's logical surface both
    ways."""
    free = seq.odd.free
    B = jnp.zeros(free.n(2)).at[seq.odd._dirichlet_dofs[2]].set(b0)      # restrict(B) = b0, the wall DoFs zero
    res = poincare(seq, B, lines=8, periods=60, planes=1, padding=(4, 64, 0.8))
    shown = np.asarray(res["shown"])
    r = np.asarray(res["seed_r"])[shown]
    want = eq.iota(r ** 2)
    err = float(np.abs(np.asarray(res["iota"])[shown] - want).max() / want.max())
    pad = res["padding"]
    logr = next(iter(pad["sections"].values()))["logr"]
    off = float(np.nanmax(np.abs(logr - np.asarray(pad["seed_r"])[:, None])))
    print(f"\n  {int(shown.sum())} of {shown.size} lines regular, iota {want.min():.4f}..{want.max():.4f}, "
          f"relative error {err:.2e}, drift {float(res['drift']):.2e}, padding off its surface {off:.1e}")
    assert shown.all()                  # the field is integrable: no line is chaotic or leaves the domain
    assert err < IOTA_BAND
    assert float(res["drift"]) < _tol(DRIFT_BAND, 1e-3)       # float32: 2.7e-4 measured here, 5.2e-4 on CI
    assert pad["shown"].all() and np.isfinite(logr[:, 1:]).all()      # column 0: the seed, where on the plane
    assert off < _tol(1e-9, 1e-4)     # B^r = 0 exactly: measured 6.2e-10 on the tokamak (float64), 1.2e-5 (float32)


#: The poloidal rate of the closed-form field, in turns per period.
C = 0.3


def _closed_form(x, dof):
    """The logical field ``(dof[0], C, 1)``: with ``dof[0] = 0`` the line from ``(r0, theta0)`` is ``r = r0``,
    ``theta = theta0 + C zeta``."""
    return jnp.array([dof[0], C, 1.0])


def test_crossings_of_a_closed_form_field(seq):
    """The batched trace of :func:`_closed_form`, lines forwards and backwards. Without a radial component, the
    crossing ``n`` of the plane ``zeta = p`` is at ``zeta = n + p`` forwards and ``p - n`` backwards, on the exact
    line, to the integrator's error. With ``B^r = 0.02`` per period the line from ``r0 = 0.9`` reaches the boundary
    after 5 periods, its crossings from then on are NaN and the earlier ones finite, and the other lines stay in."""
    from mrx.diagnostics.poincare import (_batch_sections, _record_offsets, _reduced_trace, _step_end_crossings,
                                          steps_for, to_uv)
    planes, periods = (0.0, 0.13), 12
    S = steps_for(planes)
    offsets = _record_offsets(planes, S)
    seeds = jnp.array([[0.9, 0.0], [0.6, 0.25], [0.3, 0.5]])
    uv = jnp.broadcast_to(to_uv(seeds[:, 0], seeds[:, 1]), (2, 2, 3, 2)).reshape(2, 6, 2)
    direction = jnp.array([1.0] * 3 + [-1.0] * 3)
    rec, _, esc = _reduced_trace(_closed_form, jnp.array([[0.0], [0.02]]), uv, direction, jnp.zeros((2, S, 2)),
                                 jnp.full(6, periods * S + 1), periods, S, offsets)
    esc = np.asarray(esc)
    for lines, sign in ((slice(0, 3), 1), (slice(3, 6), -1)):
        for plane, (got, z) in _step_end_crossings(rec[0, lines], S, planes, offsets, periods - 1, sign).items():
            n = np.arange(1, periods)
            assert np.allclose(np.asarray(z), sign * n + plane)
            want = to_uv(seeds[:, :1], seeds[:, 1:] + C * (sign * n + plane))
            assert np.abs(np.asarray(got) - np.asarray(want)).max() < _tol(1e-9, 1e-4), (sign, plane)
    assert (esc[0] == periods * S + 1).all()
    assert 0.0 <= esc[1, 0] / S - 5.0 <= 3.0 / S and (esc[1, 1:] == periods * S + 1).all()
    crossings = _step_end_crossings(rec[1, :3], S, (0.13,), offsets, periods - 1, 1)
    R = _batch_sections(seq, "zeta", crossings, esc[1, :3], S, 1)[0.13]["R"]
    assert np.array_equal(np.flatnonzero(np.isnan(R[0])), np.arange(4, periods - 1))   # zeta = 5.13 and later
    assert np.isfinite(R[1:]).all()


def test_phi_sections_are_zeta_sections_on_a_cylindrical_map(seq, b0):
    """On the file's own map the planes ``phi = const`` are the sections ``zeta = const``. The crossings that
    the Newton path of the sections ``"phi"`` finds in the steps the batched trace kept (:func:`_refine_crossings`)
    are then the states at the ends of steps, forwards and backwards, to round-off. The fixture's map has the
    sections normal to the axis, so the file's map (:func:`mrx.equilibria.build_map`) stands in for it, with the
    fixture's logical field."""
    from mrx.diagnostics.poincare import (_record_offsets, _reduced_trace, _refine_crossings, _step_end_crossings,
                                          logical_field, steps_for, to_uv)
    from mrx.equilibria import build_map
    Phi = build_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)[0]
    planes, periods = (0.0, 0.25), 6
    S = steps_for(planes)
    offsets = _record_offsets(planes, S)
    field = logical_field(seq, b0)
    coef = field.coefficients(jnp.asarray(b0))
    seeds = jnp.array([[0.3, 0.1], [0.6, 0.7], [0.9, 0.3]] * 2)
    direction = jnp.array([1.0] * 3 + [-1.0] * 3)
    rec, _, _, brackets, reached = _reduced_trace(
        field, jax.tree.map(lambda a: a[None], coef), to_uv(seeds[:, 0], seeds[:, 1])[None], direction,
        jnp.zeros((1, S, 2)), jnp.full(6, periods * S + 1), periods, S, offsets, Phi=Phi, planes=planes,
        n_cross=periods - 1)
    uv, z = _refine_crossings(field, coef, Phi, brackets[0], reached[0], direction, S, planes)
    for lines, sign in ((slice(0, 3), 1), (slice(3, 6), -1)):
        want = _step_end_crossings(rec[0, lines], S, planes, offsets, periods - 1, sign)
        for j, plane in enumerate(planes):
            assert np.abs(np.asarray(uv[lines, j]) - np.asarray(want[plane][0])).max() < _tol(1e-10, 1e-5)
            assert np.abs(np.asarray(z[lines, j]) - np.asarray(want[plane][1])).max() < _tol(1e-12, 1e-5)


def test_phi_sections_of_an_axis_normal_map(seq, b0, minor):
    """The fixture's map has its sections normal to the axis, so its planes ``zeta = const`` are not the planes
    ``phi = const``, and the file's field on it is carried to the new coordinates
    (:mod:`mrx.relaxation.initial_conditions`). Cut at the planes ``phi = const`` (``sections="phi"``, the default),
    its lines are those of the same field on the file's map (:func:`mrx.equilibria.build_map`, the field without the
    shift of ``zeta``): the same iota per line and the same crossings in cylindrical ``(R, Z)``, to the
    discretisation error of the two fields (:data:`PHI_IOTA_BAND`, :data:`PHI_CROSSING_BAND`). The seeds lie in the
    plane ``zeta = 0``, where a stellarator-symmetric cut leaves every point where it is, so the two traces start
    from the same physical points."""
    from mrx.equilibria import build_map
    from mrx.geometry import SequenceGeometry
    from mrx.relaxation.initial_conditions import _clebsch_potential
    st = {k: v for k, v in seq.equilibrium.items() if k != "zeta_shift"}
    free = seq.odd.free
    A = free.interpolate(_clebsch_potential(st, seq.nfp, seq.basis_0.bases[0]), 1)
    B_file = seq.odd.restrict(free.G[1] @ A, 2)
    Phi = build_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)[0]
    on_file = seq.with_geometry(SequenceGeometry.from_map(Phi, seq.quad.x))
    kw = dict(lines=8, periods=40, planes=(0.0, 0.25), sections="phi")
    a, b = poincare(seq, b0, **kw), poincare(on_file, B_file, **kw)
    shown = np.asarray(a["shown"]) & np.asarray(b["shown"])
    d_iota = float(np.max(np.abs(np.asarray(a["iota"]) - np.asarray(b["iota"]))[shown]))
    d_x = max(float(np.nanmax(np.hypot(a["sections"][p]["R"][shown, :10] - b["sections"][p]["R"][shown, :10],
                                       a["sections"][p]["Z"][shown, :10] - b["sections"][p]["Z"][shown, :10])))
              for p in (0.0, 0.25))
    print(f"\n  {shown.sum()} regular lines, iota apart by {d_iota:.1e}, first ten crossings by {d_x:.1e} m = {d_x / minor:.1%} "
          f"of the minor radius")
    assert shown.sum() >= 6
    assert d_iota < PHI_IOTA_BAND and d_x < PHI_CROSSING_BAND * minor
