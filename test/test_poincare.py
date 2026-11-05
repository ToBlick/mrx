"""The field-line tracer of :mod:`mrx.diagnostics.poincare`, on the equilibrium's own field.

``b0`` (:func:`mrx.relaxation.initial_conditions.potential_two_form`) is the curl of a vector potential whose
``theta`` and ``zeta`` components depend on ``r`` alone, so ``B^r = 0`` exactly: its field lines stay on their
logical surfaces and wind with the rotational transform of the file. The traced iota of every regular line
matches ``|iota(r)|`` of the file at the line's radius within a measured band, the discretisation error of the
field, and halving the step moves the lines by less than a measured drift. The sections are cut at the physical
planes ``phi = const``, also off the tracer's step grid and on maps whose sections ``zeta = const`` are not those
planes (a GVEC G-frame state). The batched trace (:func:`mrx.diagnostics.poincare.poincare_many`) is checked against
the stepwise one, which saves every step. CI runs the suite in float64 and float32: the comparisons scale with the
precision (:func:`_tol`).
"""
import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.diagnostics.poincare import poincare
from mrx.equilibria import read_equilibrium, state_position
from mrx.relaxation.initial_conditions import _clebsch_potential

#: 1.25x ``max |iota_traced - |iota_file|| / max |iota_file|`` on li383 (16, 32, 12) p=2, 8 lines over 60 periods,
#: measured 1.06e-3 (2026-10-20). A tracer that missed the factor nfp or a wrong winding would be off by O(1).
IOTA_BAND = 1.33e-3
#: 1.25x the step-halving drift of the same trace in logical radius, measured 3.23e-4.
DRIFT_BAND = 4.1e-4
#: The iota of the batched trace against the stepwise one on the same lines, whose windings are counted about two
#: estimates of the same axis (the axis probe's mean and the main trace's).
IOTA_AXIS_BAND = 1e-4


def _tol(f64, f32):
    """The tolerance ``f64`` in float64 and ``f32`` in float32 (CI runs both)."""
    from mrx.precision import DTYPE
    return f64 if DTYPE == np.float64 else f32


def test_traced_iota_is_the_file_iota(seq, b0):
    """The traced iota of the regular lines is the file's iota at their radius, and the step is resolved."""
    res = poincare(seq, b0, lines=8, periods=60, planes=1)
    shown = np.asarray(res["shown"])
    r = np.asarray(res["seed_r"])[shown]
    want = np.abs(np.asarray(seq.equilibrium["profiles"]["iota"](r)))
    err = float(np.abs(np.asarray(res["iota"])[shown] - want).max() / want.max())
    print(f"\n  {int(shown.sum())} of {shown.size} lines regular, iota {want.min():.4f}..{want.max():.4f}, "
          f"relative error {err:.2e}, drift {float(res['drift']):.2e}")
    assert shown.all()                  # the field is integrable: no line is chaotic or leaves the domain
    assert err < IOTA_BAND
    assert float(res["drift"]) < DRIFT_BAND

def test_poincare_sections_are_planes_phi_const(seq):
    """Field lines of the closed-form field of the production sequence's state cut by
    :func:`mrx.diagnostics.poincare.poincare`'s crossings at the plane ``phi = 2 pi 0.13 / nfp``, which is off the
    tracer's step grid. Every crossing lies on that plane to round-off, and on the file's flux surface of its line
    there (the closed-form field has no radial component, so a line keeps its ``r``) to the accuracy of the spline
    map (tolerance 1e-3 m)."""
    from mrx.diagnostics.poincare import _section_RZ, steps_for, trace
    st = seq.equilibrium
    A = _clebsch_potential(st, st["nfp"])

    def field(x, dof):
        dA = jax.jacfwd(A)(x)
        return jnp.array([dA[2, 1] - dA[1, 2], dA[0, 2] - dA[2, 0], dA[1, 0] - dA[0, 1]])
    plane, dof = 0.13, jnp.zeros(1)
    steps = steps_for((plane,))
    ys = trace(field, dof, jnp.array([[0.3, 0.0], [0.6, 0.0], [0.9, 0.0]]), 4, steps)
    sec = _section_RZ(seq, field, dof, ys, ys[0], steps, plane)
    x = jnp.stack([jnp.asarray(sec["logr"]).ravel(), jnp.asarray(sec["logth"]).ravel(),
                   jnp.asarray(sec["logz"]).ravel()], 1)
    P = jax.vmap(seq.map)(x)
    phi = np.asarray(jnp.arctan2(P[:, 1], P[:, 0])) * seq.nfp / (2.0 * np.pi)
    off_plane = np.abs((phi - plane + 0.5) % 1.0 - 0.5).max()
    file_position = state_position(read_equilibrium(st["path"]))
    R, Z = np.asarray(sec["R"]).ravel(), np.asarray(sec["Z"]).ravel()
    off_surface = []
    for r, Ri, Zi in zip(np.asarray(sec["logr"]).ravel(), R, Z):
        def dist2(th):
            q = file_position(jnp.array([r, th, plane]))
            return (jnp.hypot(q[0], q[1]) - Ri) ** 2 + (q[2] - Zi) ** 2
        th = jnp.linspace(0.0, 1.0, 257)[:-1]
        t = th[jnp.argmin(jax.vmap(dist2)(th))]
        for _ in range(8):
            t = t - jax.grad(dist2)(t) / jax.hessian(dist2)(t)
        off_surface.append(float(jnp.sqrt(dist2(t))))
    assert off_plane <= mrx.eps(1e3)
    assert max(off_surface) <= 1e-3


def test_lost_lines_are_cut_at_the_boundary_and_reseeded(seq):
    """A closed-form field with a constant outward ``B^r`` (``0.02`` per period) carries every line to the boundary
    in ``(1 - r0) / 0.02`` periods. The line from ``r0 = 0.9`` reaches it after 5 periods and is frozen there:
    :func:`_escape_step` finds that step, the section's crossings of that line from then on are NaN and the earlier
    ones finite, and the lines from 0.3 and 0.6 (12 periods traced) are untouched. :func:`reseed_lost` then traces
    two more rounds on the lost radius, each with as many lines as the first round, and none when nothing is
    lost."""
    from mrx.diagnostics.poincare import _escape_step, _escaped_mask, _section_RZ, reseed_lost, steps_for, trace

    def field(x, dof):
        return jnp.array([0.02, 0.3, 1.0])
    plane, dof, periods = 0.13, jnp.zeros(1), 12
    steps = steps_for((plane,))
    seeds = jnp.array([[0.01, 0.0], [0.9, 0.0], [0.6, 0.25], [0.3, 0.5]])
    ys = trace(field, dof, seeds, periods, steps)
    esc = np.asarray(_escape_step(ys))
    assert np.asarray(_escaped_mask(ys)).tolist() == [False, True, False, False]
    assert 0.0 <= esc[1] / steps - 5.0 <= 3.0 / steps
    assert (esc[[0, 2, 3]] == ys.shape[1]).all()
    sec = _section_RZ(seq, field, dof, ys[1:], ys[0], steps, plane, esc[1:] / steps)
    R = np.asarray(sec["R"])
    gone = np.isnan(R[0])
    assert gone.any() and not gone.all() and (np.flatnonzero(gone) >= 4).all() and gone[-1]
    assert np.isfinite(R[1:]).all() and np.isfinite(sec["axisR"]).all()
    more = reseed_lost(field, dof, seeds, _escaped_mask(ys), periods, steps, 2)
    assert len(more) == 2
    for new, y in more:
        assert new.shape == (3, 2) and np.allclose(np.asarray(new[:, 0]), 0.9) and y.shape == ys[1:].shape
        assert np.asarray(_escaped_mask(y)).all()
    stay = jnp.array([0, 2, 3])
    assert reseed_lost(field, dof, seeds[stay], _escaped_mask(ys[stay]), periods, steps, 2) == []


def test_padding_lines_are_archived_with_their_own_period_count(seq, b0):
    """With ``padding = (n, periods, r_min)`` :func:`poincare` returns ``n`` further lines seeded over
    ``r_min < r < R_EDGE`` and traced for that many periods: their sections have ``periods - 1`` crossings, their
    seeds fill the annulus, and on the integrable field ``b0`` they all stay in and are regular."""
    from mrx.diagnostics.poincare import R_EDGE
    res = poincare(seq, b0, lines=4, periods=20, planes=1, padding=(12, 8, 0.7))
    pad = res["padding"]
    assert pad["keep"].shape == (12,) and pad["keep"].all() and not pad["chaotic"].any()
    r = np.asarray(pad["seed_r"])
    assert 0.7 < r.min() < 0.75 and 0.95 < r.max() <= R_EDGE and np.all(np.diff(r) > 0)
    sec = next(iter(pad["sections"].values()))
    assert sec["R"].shape == (12, 1 + 7 + 7)                  # the seed, 7 forward and 7 backward crossings
    assert np.isfinite(sec["R"][:, 1:]).all()
    assert next(iter(res["sections"].values()))["R"].shape == (4, 19)
    # b0 has no radial component: the backward crossings lie on the seed's logical surface like the forward ones
    assert np.allclose(sec["logr"][:, 8:], sec["logr"][:, 1:8].mean(1, keepdims=True), atol=2e-3)
    # padding lines traced longer than the main lines (the edge mapped over more periods): until 2026-11-05 the copy
    # of the main trace's axis line into the padding's broke on the length
    long = poincare(seq, b0, lines=4, periods=8, planes=1, padding=(6, 12, 0.7))["padding"]
    assert next(iter(long["sections"].values()))["R"].shape == (6, 1 + 11 + 11) and long["keep"].all()


def test_boozer_line_spectrum_recovers_the_modes():
    """Samples along one field line of a field given in Boozer angles (``iota = 1.05``, ``nfp = 4``): ``|B| = 1 + 0.1
    cos(theta_B - 4 zeta_B) + 0.02 cos(2 theta_B - 8 zeta_B) + 0.01 cos(4 zeta_B)`` on ``theta_B = iota zeta_B +
    0.3``, the arc length ``dl = (G + iota I) d zeta_B / B`` and a toroidal angle ``phi = zeta_B + nu`` with a periodic
    ``nu``. The fit recovers the three amplitudes to 1e-3 and the quasi-helical error of the (m, m) family (0.01)."""
    from mrx.diagnostics.poincare import boozer_line_spectrum
    iota, nfp, N = 1.05, 4, 60
    zB = np.linspace(0.0, 2 * np.pi * N, 60 * 400 + 1)
    thB = iota * zB + 0.3
    phi = zB + 0.2 * np.sin(zB) + 0.1 * np.sin(thB)
    for sign in (1, -1):                       # the line may wind either way poloidally; iota is given unsigned
        thB = sign * iota * zB + 0.3
        B = 1.0 + 0.1 * np.cos(thB - 4 * zB) + 0.02 * np.cos(2 * thB - 8 * zB) + 0.01 * np.cos(4 * zB)
        dl = 3.0 * np.diff(zB) / (0.5 * (B[1:] + B[:-1]))
        modes, amps, f_qa, f_qh, resid = boozer_line_spectrum(B, dl, phi, iota, nfp)
        want = {(0, 0): 1.0, (1, sign): 0.1, (2, 2 * sign): 0.02, (0, 1): 0.01}       # the winding sign flips n
        for (m, n), a in zip(map(tuple, modes), amps):
            assert abs(a - want.get((m, n), 0.0)) < 1e-3, ((m, n), a)
        assert resid < 2e-3
        assert abs(f_qh - 0.01) < 1e-3 and abs(f_qa - np.hypot(np.hypot(0.1, 0.02), 0.01)) < 1e-3


def test_boozer_spectra_of_the_file_field(seq, b0):
    """With ``boozer = (m_max, n_max)`` :func:`poincare` fits a spectrum on every regular line: one row per regular
    line, ``B_00`` close to the field's mean |B| and the errors between 0 and 1."""
    res = poincare(seq, b0, lines=4, periods=20, planes=1, boozer=(3, 3))
    bz = res["boozer"]
    n = int(res["shown"].sum())
    assert bz["amps"].shape == (n, bz["modes"].shape[0]) and bz["f_qa"].shape == (n,)
    b00 = bz["amps"][:, (bz["modes"][:, 0] == 0) & (bz["modes"][:, 1] == 0)][:, 0]
    assert np.all(b00 > 0) and np.all((0 <= bz["f_qa"]) & (bz["f_qa"] <= 1)) and np.all(bz["f_qh"] <= bz["f_qa"] + 1e-12 + 1)


def test_backward_trace_retraces_the_forward_one(seq, b0):
    """Traced forward for two periods and then backward from the end point, a line returns to its seed (to the
    integrator's error), and a backward trace from the seed visits the forward one's points in reverse once the
    field is reversed: on the integrable b0, the lines also keep their logical radius both ways."""
    from mrx.diagnostics.poincare import logical_field, to_polar, to_uv, trace
    field, dof = logical_field(seq), jnp.asarray(b0)
    seeds = jnp.array([[0.3, 0.1], [0.6, 0.4]])
    fwd = trace(field, dof, seeds, 2, 96)
    end = jnp.stack(to_polar(fwd[:, -1]), axis=1)
    back = trace(field, dof, end, 2, 96, direction=-1)
    # the backward tracer starts at zeta = 0 and steps to negative zeta; the field has period 1 in zeta, so starting
    # at the forward end (zeta = 2) is the same as starting at zeta = 0
    err = float(jnp.abs(back[:, -1] - to_uv(seeds[:, 0], seeds[:, 1])).max())
    r_b = jnp.sqrt(jnp.sum(back ** 2, axis=-1))
    print(f"\n  return error after 2 periods out and back: {err:.1e}; radius spread {float(jnp.ptp(r_b, axis=1).max()):.1e}")
    assert err < _tol(1e-6, 1e-4)
    assert float(jnp.ptp(r_b, axis=1).max()) < 2e-3


def test_trace_archive_holds_every_field_with_its_pressure(seq, b0):
    """:func:`trace_archive` (the content of ``trace.npz``): per field the per-line entries, the sections of every
    plane with the logical crossings, and the weak pressure at each crossing, finite where the line is; a padding
    trace under ``<name>_pad_``."""
    from mrx.diagnostics.poincare import trace_archive
    arch, res = trace_archive(seq, {"b0": (b0, 3), "half": (0.5 * b0, 4)}, lines=4, periods=20, planes=(0.0,),
                              source="li383 test", padding=(6, 8, 0.7))
    assert list(arch["fields"]) == ["b0", "half"] and set(res) == {"b0", "half"}
    assert str(arch["b0_label"]) == "b0 (step 3)" and int(arch["half_step"]) == 4
    assert abs(float(arch["half_bsq"]) / float(arch["b0_bsq"]) - 0.25) < _tol(1e-12, 1e-6)
    for key in ("iota", "seed_r", "keep", "chaotic", "shown", "round", "escape_period"):
        assert arch[f"b0_{key}"].shape == (4,), key
    for key in ("R", "Z", "logr", "logth", "logz", "pressure"):
        assert arch[f"b0_zeta0_{key}"].shape == (4, 19), key
    assert arch["b0_pad_zeta0_R"].shape == (6, 15) and "b0_pad_iota" in arch
    p = arch["b0_zeta0_pressure"]
    assert np.isfinite(p).all() and float(np.abs(p).max()) > 0
    # the weak pressure is quadratic in B: the half field has a quarter of it
    assert np.allclose(arch["half_zeta0_pressure"], 0.25 * p, rtol=_tol(1e-6, 1e-3),
                       atol=_tol(1e-9, 1e-5) * float(np.abs(p).max()))


def test_the_batched_trace_keeps_the_stepwise_states(seq, b0):
    """:func:`_reduced_trace` (two copies of the field in one batch, lines forwards and backwards) keeps the states
    that :func:`trace` saves at the same steps and the same escape step, and with the stepwise axis its running fits
    are :func:`rotational_transform`'s and :func:`_iota_convergence`'s; :func:`_drift_many` is
    :func:`_step_convergence`."""
    from mrx.diagnostics.poincare import (_drift_many, _escape_step, _iota_convergence, _record_offsets,
                                          _reduced_trace, _step_convergence, axis_track, logical_field,
                                          rotational_transform, steps_for, to_uv, trace)
    tol = _tol(1e-11, 2e-4)
    planes = (0.0, 0.25)
    S, periods = steps_for(planes), 12
    offsets = _record_offsets(planes, S)
    seeds = jnp.array([[0.01, 0.0], [0.3, 0.1], [0.6, 0.7], [0.9, 0.3]])
    field = logical_field(seq)
    fwd, back = trace(field, b0, seeds, periods, S), trace(field, b0, seeds, periods, S, direction=-1)
    centre = axis_track(fwd, S)
    coefs = jax.tree.map(lambda a: jnp.stack([a, a]), field.coefficients(jnp.asarray(b0)))
    both = jnp.concatenate([seeds, seeds])
    uv = jnp.broadcast_to(to_uv(both[:, 0], both[:, 1]), (2, 8, 2))
    direction = jnp.array([1.0] * 4 + [-1.0] * 4)
    n_saves = jnp.full(8, periods * S + 1)
    rec, rates, esc = _reduced_trace(field, coefs, uv, direction, jnp.stack([centre[:S]] * 2), n_saves, periods, S,
                                     offsets)
    for k, o in enumerate(offsets):
        steps = np.arange(periods) * S + o
        assert np.allclose(np.asarray(rec[0, :4, :, k]), np.asarray(fwd[:, steps]), rtol=0, atol=tol)
        assert np.allclose(np.asarray(rec[0, 4:, :, k]), np.asarray(back[:, steps]), rtol=0, atol=tol)
    assert np.array_equal(np.asarray(rec[0]), np.asarray(rec[1]))
    assert np.array_equal(np.asarray(esc[0, :4]), np.minimum(np.asarray(_escape_step(fwd)), periods * S + 1))
    nfp = seq.nfp
    iota = np.abs(np.asarray(rates[0, :4, 0])) * nfp
    conv = np.abs(np.abs(np.asarray(rates[0, :4, 1])) - np.abs(np.asarray(rates[0, :4, 2]))) * nfp
    assert np.allclose(iota, np.asarray(rotational_transform(fwd, S, nfp, centre)), rtol=tol, atol=tol)
    assert np.allclose(conv, np.asarray(_iota_convergence(fwd, S, nfp, centre)), rtol=0, atol=100 * tol)
    d_many = np.asarray(_drift_many(field, coefs, jnp.broadcast_to(to_uv(seeds[1:, 0], seeds[1:, 1]), (2, 3, 2)),
                                    4, S))
    d_one = _step_convergence(field, b0, seeds[1:], fwd[1:, :4 * S + 1], 4, S)
    assert np.allclose(d_many, d_one, rtol=_tol(1e-6, 1e-2), atol=tol)


def _compare_with_stepwise(seq, b, kw, tol):
    """:func:`poincare_many` (two copies of the field in one batch) against :func:`_poincare_stepwise`: the same
    flags, rounds and seeds, sections, axis and drift, and iota within :data:`IOTA_AXIS_BAND`."""
    from mrx.diagnostics.poincare import _poincare_stepwise, poincare_many
    old = _poincare_stepwise(seq, b, **kw)
    new, again = poincare_many(seq, [b, b], **kw)
    for key in ("keep", "chaotic", "shown", "round"):
        assert np.array_equal(np.asarray(new[key]), np.asarray(old[key])), key
    assert np.allclose(new["seed_r"], old["seed_r"], rtol=0, atol=tol)
    print(f"\n  iota batched - stepwise: max {np.abs(new['iota'] - old['iota']).max():.2e}; drift "
          f"{new['drift']:.3e} / {old['drift']:.3e}")
    # the innermost line starts 2 % of the way out from the axis, where its winding over a short trace follows the
    # axis estimate (the axis probe's mean in the batch, the main trace's in the stepwise trace) more closely
    assert np.allclose(new["iota"][1:], old["iota"][1:], rtol=0, atol=IOTA_AXIS_BAND)
    assert abs(new["iota"][0] - old["iota"][0]) < 10 * IOTA_AXIS_BAND
    for plane in old["sections"]:
        for key in ("R", "Z", "logr", "logz", "axisR", "axisZ"):
            assert np.allclose(new["sections"][plane][key], old["sections"][plane][key], rtol=0, atol=100 * tol,
                               equal_nan=True), (plane, key)
        for key in ("R", "Z", "logr"):
            assert np.allclose(new["padding"]["sections"][plane][key], old["padding"]["sections"][plane][key],
                               rtol=0, atol=100 * tol, equal_nan=True), ("padding", plane, key)
    assert np.array_equal(new["padding"]["keep"], old["padding"]["keep"])
    assert np.isclose(new["drift"], old["drift"], rtol=_tol(1e-6, 1e-2), atol=tol)
    for key in ("iota", "keep"):
        assert np.array_equal(np.asarray(new[key]), np.asarray(again[key]))


def test_the_batched_sections_are_the_stepwise_ones(seq, b0):
    """On the planes ``phi = const`` of the production sequence's cylindrical map (:func:`_planar_sections`) the
    crossings are the states at the ends of steps in both traces."""
    from mrx.diagnostics.poincare import _planar_sections
    assert _planar_sections(seq)
    _compare_with_stepwise(seq, b0, dict(lines=6, periods=30, planes=(0.0, 0.25), padding=(5, 10, 0.7)),
                           _tol(1e-11, 2e-5))


def test_the_batched_sections_on_a_map_without_planar_sections():
    """On a GVEC G-frame state (the vacuum N2-12 of GVEC's tests, whose sections ``zeta = const`` are the planes of a
    turning frame) :func:`poincare_many` finds the crossings in the steps its trace kept (:func:`_refine_crossings`)
    and gives the stepwise trace's sections, axis, flags and drift, with the padding traced both ways."""
    from mrx.diagnostics.poincare import _planar_sections
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation.initial_conditions import potential_two_form
    s = build_sequence("data/gvec_gframe/N2-12/gframe_N2-12_State_final.dat", (8, 16, 12), 2,
                       symmetry="field-period", graded=True)
    compute_nullspaces(s, verbose=False)
    assert not _planar_sections(s)
    b = potential_two_form(s)[0]
    _compare_with_stepwise(s, b, dict(lines=6, periods=16, planes=(0.0, 0.25), padding=(4, 6, 0.7)),
                           _tol(1e-11, 2e-5))
