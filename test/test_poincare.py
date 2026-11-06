"""The field-line tracer of :mod:`mrx.diagnostics.poincare`.

``b0`` (:func:`mrx.relaxation.initial_conditions.potential_two_form`) is the curl of a vector potential whose
``theta`` and ``zeta`` components depend on ``r`` alone, so ``B^r = 0`` exactly: its field lines stay on their
logical surfaces and wind with the rotational transform of the file. The traced iota of every regular line
matches ``|iota(r)|`` of the file at the line's radius within a measured band, the discretisation error of the
field, and halving the step moves the lines by less than a measured drift. A closed-form field, whose lines are
known exactly, pins the crossings forwards and backwards and the cut at the boundary. The sections ``phi = const``
are the sections ``zeta = const`` on a cylindrical map. CI runs the suite in float64 and float32, and the
tolerances scale with the precision (:func:`_tol`).
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.diagnostics.poincare import poincare

#: 1.25x ``max |iota_traced - |iota_file|| / max |iota_file|`` on li383 (16, 32, 12) p=2, 8 lines over 60 periods,
#: measured 1.06e-3 (2026-10-20). A tracer that missed the factor nfp or a wrong winding would be off by O(1).
IOTA_BAND = 1.33e-3
#: 1.25x the step-halving drift of the same trace in logical radius, measured 3.23e-4.
DRIFT_BAND = 4.1e-4


def _tol(f64, f32):
    """The tolerance ``f64`` in float64 and ``f32`` in float32 (CI runs both)."""
    from mrx.precision import DTYPE
    return f64 if DTYPE == np.float64 else f32


def test_traced_iota_is_the_file_iota(seq, b0):
    """The traced iota of the regular lines is the file's iota at their radius, and the step is resolved. The
    padding lines, traced forwards and backwards for more periods than the main lines, stay on their seed's logical
    surface both ways."""
    res = poincare(seq, b0, lines=8, periods=60, planes=1, padding=(4, 64, 0.8))
    shown = np.asarray(res["shown"])
    r = np.asarray(res["seed_r"])[shown]
    want = np.abs(np.asarray(seq.equilibrium["profiles"]["iota"](r)))
    err = float(np.abs(np.asarray(res["iota"])[shown] - want).max() / want.max())
    pad = res["padding"]
    logr = next(iter(pad["sections"].values()))["logr"]
    off = float(np.nanmax(np.abs(logr - np.asarray(pad["seed_r"])[:, None])))
    print(f"\n  {int(shown.sum())} of {shown.size} lines regular, iota {want.min():.4f}..{want.max():.4f}, "
          f"relative error {err:.2e}, drift {float(res['drift']):.2e}, padding off its surface {off:.1e}")
    assert shown.all()                  # the field is integrable: no line is chaotic or leaves the domain
    assert err < IOTA_BAND
    assert float(res["drift"]) < DRIFT_BAND
    assert pad["shown"].all() and np.isfinite(logr[:, 1:]).all()      # column 0: the seed, where on the plane
    assert off < _tol(1e-10, 1e-4)     # B^r = 0 exactly: measured 3.3e-13 (float64), 1.2e-5 (float32), 2026-11-06


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
    """On li383 the planes ``phi = const`` are the sections ``zeta = const``. The crossings that the Newton path
    of the sections ``"phi"`` finds in the steps the batched trace kept (:func:`_refine_crossings`) are then the
    states at the ends of steps, forwards and backwards, to round-off."""
    from mrx.diagnostics.poincare import (_record_offsets, _reduced_trace, _refine_crossings, _step_end_crossings,
                                          logical_field, steps_for, to_uv)
    planes, periods = (0.0, 0.25), 6
    S = steps_for(planes)
    offsets = _record_offsets(planes, S)
    field = logical_field(seq)
    coef = field.coefficients(jnp.asarray(b0))
    seeds = jnp.array([[0.3, 0.1], [0.6, 0.7], [0.9, 0.3]] * 2)
    direction = jnp.array([1.0] * 3 + [-1.0] * 3)
    rec, _, _, brackets, reached = _reduced_trace(
        field, jax.tree.map(lambda a: a[None], coef), to_uv(seeds[:, 0], seeds[:, 1])[None], direction,
        jnp.zeros((1, S, 2)), jnp.full(6, periods * S + 1), periods, S, offsets, Phi=seq.map, planes=planes,
        n_cross=periods - 1)
    uv, z = _refine_crossings(field, coef, seq.map, brackets[0], reached[0], direction, S, planes)
    for lines, sign in ((slice(0, 3), 1), (slice(3, 6), -1)):
        want = _step_end_crossings(rec[0, lines], S, planes, offsets, periods - 1, sign)
        for j, plane in enumerate(planes):
            assert np.abs(np.asarray(uv[lines, j]) - np.asarray(want[plane][0])).max() < _tol(1e-10, 1e-5)
            assert np.abs(np.asarray(z[lines, j]) - np.asarray(want[plane][1])).max() < _tol(1e-12, 1e-5)
