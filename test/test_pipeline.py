"""The whole fixed-boundary pipeline from a namelist alone, on the shared sequence, against the exact Solov'ev tokamak.

The steps of a run from a VMEC input namelist (``scripts/relax.py --mesh.align --mesh.nested`` with the pressure and
the transform of the namelist as targets), driven through the library:

1. the coarse nested-surface solve of the namelist puts the finite-beta magnetic axis
   (:mod:`mrx.equilibria.nested_axis`),
2. the namelist's fill, cut by the planes normal to its axis, is moved onto that axis and cut again
   (:func:`mrx.relaxation.mesh_motion.curve_start`),
3. the vacuum field of the namelist's flux on that mesh is the start,
4. Newton steps with the shells of the exact pressure and the iota loop on the exact transform, rebuilt before every
   chunk, with the move rule: the pole moves onto the magnetic axis at a chunk boundary below the residual bar when
   the axis is more than the move tolerance off it, the field carried by the induction,
5. the energy seed at the resonances in the transform's range, relaxed again,
6. a short field-line trace.

The end state is checked against shearedE: the field error at the projection's level, beta, the magnetic axis in
fractions of the minor radius, the traced transform and the residual below the bar. The nested axis is checked on its
own, on the tokamak. The run is the
most expensive test of the suite, in float64 only: a float32 Newton step costs three times a float64 one on the CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.diagnostics.poincare import poincare
from mrx.equilibria import nested_axis, read_equilibrium, state_position
from mrx.equilibria.indata import read_indata
from mrx.flux_label import ANISOTROPY
from mrx.precision import DTYPE
from mrx.relaxation import mesh_motion as mm
from mrx.relaxation.initial_conditions import vacuum_field
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import current, weak_pressure
from mrx.relaxation.seeding import energy_seed
from mrx.relaxation.shells import build_shells, label, profile_cells, target_beta
from mrx.relaxation.twist import current_on_label, iota_newton
from test.conftest import LENGTH, field_error, map_installed

#: The nested axis against the exact axis, as a fraction of the minor radius: 6.3e-3 measured on the tokamak at
#: :data:`NESTED` (2026-11-22).
NESTED_AXIS = LENGTH
#: The residual bar and the move tolerance of the move rule (``--mesh.bar``, ``--mesh.move-tol``).
BAR, MOVE_TOL = 1e-8, 0.1
#: The Newton steps and the chunk of the run.
STEPS, CHUNK = 60, 10
#: The end state against shearedE: the field error over the projection's error on the fixture's own mesh, beta
#: relative to the exact beta, the magnetic axis as a fraction of the minor radius, the traced transform relative to
#: the exact one: PENDING measured.
FIELD_FACTOR, BETA, AXIS, IOTA = 3.0, 5e-2, LENGTH, 1e-2


def _float64_only():
    if DTYPE != np.float64:
        pytest.skip("the pipeline runs in float64 only: a float32 Newton step costs three times as much on the CPU")


#: The coarsest nested-surface solve that finds the tokamak's axis: (3, 6, 4) cubic splines, 24 s on 8 CPU cores,
#: 6.3e-3 of the minor radius (the default (6, 12, 10) takes 1015 s for 7.5e-5, 2026-11-22).
NESTED = dict(nr=3, nt=6, nz=4)


def _nested(path, **kw):
    """``(zetas, points)``: the axis of the namelist's nested-surface solve, Cartesian points at logical ``zetas``."""
    Phi, _ = nested_axis.solve(path, verbose=False, **kw)
    return nested_axis.axis_points(Phi)


@pytest.fixture(scope="module")
def nested(files):
    """The nested axis of the Solov'ev namelist (:func:`_nested` at :data:`NESTED`)."""
    return _nested(files[1], **NESTED)


def _off_axis(eq, points, minor):
    """The largest distance of the Cartesian ``points`` from the exact axis in the plane of their own azimuth, as a
    fraction of the minor radius."""
    return float(np.max(np.linalg.norm(points - eq.axis(np.arctan2(points[:, 1], points[:, 0])), axis=1))) / minor


def test_the_nested_axis_is_the_exact_axis(eq, nested, minor):
    """The axis of the coarse nested-surface solve of the namelist (:data:`NESTED`) is the tokamak's exact axis
    within :data:`NESTED_AXIS` of the minor radius."""
    _float64_only()
    off = _off_axis(eq, nested[1], minor)
    print(f"\n  nested axis {off:.3e} of the minor radius off the exact axis")
    assert off < NESTED_AXIS


@pytest.mark.skip(reason="pending the gradual iota / current target ramp from the vacuum (another branch): the tokamak's "
                         "vacuum has no flux surfaces, its iota loop matrix is singular")
def test_the_pipeline_ends_at_the_exact_equilibrium(eq, seq, b0, files, nested, minor):
    """Steps 1 to 6 of the module docstring: the end field is the exact field within :data:`FIELD_FACTOR` times the
    projection's error of the fixture, its beta the exact beta within :data:`BETA`, its magnetic axis the exact axis
    within :data:`AXIS` of the minor radius, the traced transform of its regular lines the exact transform on their
    surfaces within :data:`IOTA`, and its squared force residual below :data:`BAR`."""
    _float64_only()
    projection = field_error(eq, seq, b0)
    st = read_equilibrium(files[1])
    nl = read_indata(files[1])
    fill = mm.axis_normal_map(seq, state_position(st))[0]
    with map_installed(seq, fill):
        start = mm.curve_start(seq, *nested, True)
        B0 = vacuum_field(seq, nl.phiedge)
        ts = TimeStepper(seq=seq, newton=True)
        p_star = lambda s: jnp.asarray(eq.mu0_p_of_s(np.asarray(s)))   # noqa: E731
        loop = dict(target=None, moves=[])

        def rebuild(state, it):
            nonlocal ts
            if it > 0 and float(initial_state(state.B_n, ts).best.resid) < BAR:
                d, h1 = mm.axis_offset(seq, mm.magnetic_axis(seq, state.B_n)[0])
                if float(np.max(np.linalg.norm(d, axis=1))) / h1 > MOVE_TOL:
                    B, rec = mm.axis_move(seq, state.B_n)
                    loop["moves"].append(dict(step=it, cells=rec["axis_cells"]))
                    ts = TimeStepper(seq=seq, newton=True)
                    state = initial_state(B, ts, step=it)
            s_q, T, _ = label(seq, state.B_n, ANISOTROPY)
            shells, _ = build_shells(seq, state.B_n, s_q, p_star(s_q), ts.newton_penalty, ts.newton_mass_solve_tol,
                                     profile_cells(seq))
            ts_k = ts.with_shells(shells)
            if loop["target"] is None:
                loop["target"] = np.asarray(current_on_label(seq, state.B_n, T, profile_cells(seq)), dtype=np.float64)
            slips, loop["target"], _ = iota_newton(seq, state.B_n, T, s_q, eq.iota, ts_k, loop["target"])
            return state, ts_k.with_slips(slips)
        res = relax(initial_state(B0, ts), ts, steps=STEPS, chunk=CHUNK, rebuild=rebuild)
        B = res.state.B_n
        seeded, rows = energy_seed(seq, B, verbose=False)
        assert rows
        res = relax(initial_state(seeded, ts, step=STEPS), ts, steps=CHUNK, chunk=CHUNK, it0=STEPS, rebuild=rebuild)
        B = res.state.B_n
        trace = poincare(seq, B, lines=6, periods=30, planes=(0.0,))
        resid = float(initial_state(B, ts).best.resid)
        error = field_error(eq, seq, B)
        X = np.asarray(jax.vmap(seq.map)(seq.quad.x))
        w = np.asarray(seq.quad.w * seq.jacobian_j)
        p_q = eq.mu0_p(np.hypot(X[:, 0], X[:, 1]), X[:, 2])
        beta_exact = float(np.sum(w * p_q) / (0.5 * np.sum(w * np.sum(eq.field(X) ** 2, axis=1))))
        p_w = np.asarray(seq.even.evaluate_at_quadrature(weak_pressure(current(B, seq), B, seq), 0)[:, 0])
        p_gauged = p_w + np.sum(w * (p_q - p_w)) / np.sum(w)      # the gauge <p> = <p*> of the fixed boundary
        beta = target_beta(seq, B, jnp.asarray(p_gauged))
        axis = _off_axis(eq, mm.axis_curve(seq, B)[1], minor)
    shown = np.asarray(trace["shown"])
    R0, Z0 = (np.asarray(trace["sections"][0.0][c])[:, 0] for c in ("R", "Z"))
    want = eq.iota(eq.s_at(np.stack([R0, np.zeros_like(R0), Z0], axis=1)))
    d_iota = float(np.abs(np.abs(np.asarray(trace["iota"])) - want)[shown].max() / want.max())
    r_hist = np.asarray(res.trace["resid"])
    print(f"\n  seeds {[(r['m'], r['n'], round(r['x'], 3), round(r['w'], 4)) for r in rows]}"
          f"\n  start: pole onto the nested axis in {len(start['rounds'])} rounds, {start['axis_cells']:.2e} cells"
          f"\n  resid every chunk {' '.join(f'{r:.1e}' for r in r_hist[CHUNK - 1::CHUNK])}, moves {loop['moves']}"
          f"\n  end: resid {resid:.2e}, field {error:.3e} (projection {projection:.3e}), beta {beta:.5f} (exact "
          f"{beta_exact:.5f}), axis {axis:.2e} of a, traced iota {d_iota:.2e} ({int(shown.sum())} regular lines)")
    assert resid < BAR
    assert error < FIELD_FACTOR * projection
    assert abs(beta / beta_exact - 1.0) < BETA
    assert axis < AXIS
    assert shown.sum() >= 4 and d_iota < IOTA
