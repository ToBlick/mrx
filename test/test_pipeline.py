"""The whole fixed-boundary pipeline from a namelist alone, on the shared sequence, against the exact Solov'ev tokamak.

The steps of a run from a VMEC input namelist (``scripts/relax.py --mesh.align`` with the pressure and the transform
of the namelist as targets), driven through the library:

1. the exact field with its poloidal part scaled by 0.8 (:meth:`test.solovev.Solovev.start`) is the start, as the
   vacuum field is for a stellarator: the tokamak's vacuum has no transform and no flux label,
2. the namelist's fill, cut by the planes normal to its axis, is moved onto the start field's magnetic axis in rounds,
   with the start field solved again on every round's map, and cut again
   (:func:`mrx.relaxation.mesh_motion.vacuum_start`),
3. Newton steps with the shells of the exact pressure and the iota loop on the exact transform, rebuilt before every
   chunk, with the move rule: the pole moves onto the magnetic axis at a chunk boundary below the residual bar when
   the axis is more than the move tolerance off it, the field carried by the induction,
4. the energy seed at the resonances in the transform's range, relaxed again,
5. a short field-line trace.

The end state is checked against the exact equilibrium: the field error at the projection's level, beta, the magnetic
axis in fractions of the minor radius, the traced transform and the residual below the bar. The run is the most
expensive test of the suite, in float64 only: a float32 Newton step costs three times a float64 one on the CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.diagnostics.poincare import poincare
from mrx.equilibria import read_equilibrium, state_position
from mrx.flux_label import ANISOTROPY, LABEL_MARGIN, label_condition
from mrx.precision import DTYPE
from mrx.relaxation import mesh_motion as mm
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import current, weak_pressure
from mrx.relaxation.seeding import energy_seed
from mrx.relaxation.shells import build_shells, label, profile_cells, target_beta
from mrx.relaxation.twist import current_on_label, iota_newton
from test.conftest import LENGTH, axis_error, field_error, map_installed

#: The residual bar and the move tolerance of the move rule (``--mesh.bar``, ``--mesh.move-tol``).
BAR, MOVE_TOL = 1e-8, 0.1
#: The Newton steps and the chunk of the run.
STEPS, CHUNK = 60, 10
#: The end state against the exact equilibrium: the field error over the projection's error on the fixture's own mesh, beta
#: relative to the exact beta, the magnetic axis as a fraction of the minor radius, the traced transform relative to
#: the exact one: 0.87, beta to 4 digits, 6.5e-5 and 6.6e-4 measured with the pole on the start field's axis (float64,
#: 2026-11-23).
FIELD_FACTOR, BETA, AXIS, IOTA = 3.0, 5e-2, LENGTH, 1e-2


def _float64_only():
    if DTYPE != np.float64:
        pytest.skip("the pipeline runs in float64 only: a float32 Newton step costs three times as much on the CPU")


def test_the_pipeline_ends_at_the_exact_equilibrium(eq, seq, b0, files, minor):
    """Steps 1 to 5 of the module docstring: the end field is the exact field within :data:`FIELD_FACTOR` times the
    projection's error of the fixture, its beta the exact beta within :data:`BETA`, its magnetic axis the exact axis
    within :data:`AXIS` of the minor radius, the traced transform of its regular lines the exact transform on their
    surfaces within :data:`IOTA`, and its squared force residual below :data:`BAR`."""
    _float64_only()
    projection = field_error(eq, seq, b0)
    st = read_equilibrium(files[1])
    fill = mm.axis_normal_map(seq, state_position(st))[0]
    # the start field is solved on the fill with its harmonic field, as every round's map has it
    with map_installed(seq, fill, harmonic=True):
        start = mm.vacuum_start(seq, eq.start, True)
        B0 = eq.start(seq)
        iota_m, a_m, R_m = label_condition(seq, seq.odd.evaluate_at_quadrature(B0, 2))
        margin = (iota_m * a_m / R_m) ** 2 / ANISOTROPY
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
        axis = axis_error(eq, seq, B, minor)
    shown = np.asarray(trace["shown"])
    R0, Z0 = (np.asarray(trace["sections"][0.0][c])[:, 0] for c in ("R", "Z"))
    want = eq.iota(eq.s_at(np.stack([R0, np.zeros_like(R0), Z0], axis=1)))
    d_iota = float(np.abs(np.abs(np.asarray(trace["iota"])) - want)[shown].max() / want.max())
    r_hist = np.asarray(res.trace["resid"])
    print(f"\n  seeds {[(r['m'], r['n'], round(r['x'], 3), round(r['w'], 4)) for r in rows]}"
          f"\n  start: pole onto the start field's axis in {len(start['rounds'])} rounds, {start['axis_cells']:.2e} "
          f"cells, label's condition {margin:.3g} kperp"
          f"\n  resid every chunk {' '.join(f'{r:.1e}' for r in r_hist[CHUNK - 1::CHUNK])}, moves {loop['moves']}"
          f"\n  end: resid {resid:.2e}, field {error:.3e} (projection {projection:.3e}), beta {beta:.5f} (exact "
          f"{beta_exact:.5f}), axis {axis:.2e} of a, traced iota {d_iota:.2e} ({int(shown.sum())} regular lines)")
    assert margin > LABEL_MARGIN
    assert resid < BAR
    assert error < FIELD_FACTOR * projection
    assert abs(beta / beta_exact - 1.0) < BETA
    assert axis < AXIS
    assert shown.sum() >= 4 and d_iota < IOTA
