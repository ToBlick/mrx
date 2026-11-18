"""The twist move changes the rotational transform by ``nfp dF/dPhi`` and nothing else.

On the li383 fixture field (the VMEC equilibrium, whose logical surfaces ``r = sqrt(s)`` are its flux surfaces)
:func:`mrx.relaxation.twist.retarget_iota` twists the transform to 0.9 times the file's. The change is divergence-free
to round-off, has no flux through the wall and none through the sections ``zeta = const`` (no ``zeta`` component), and
the transform of the logical layers, measured apart from the label's bins the twist is built on, meets the target.
Where the twist's profile is flat it vanishes, as across an island, where the label is flat. The test takes 20 s on
one H200 next to the suite's fixtures (their setup 170 s), 11 s on four CPU cores.

A user's run with ``--iota.target`` and ``--pressure.target`` from one VMEC input namelist ends at both targets. On
the fixture a transform and a pressure a few percent off the file's are met, and a seed on the twisted field sits at
the resonances of its new transform.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.flux_label import ANISOTROPY
from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import current, weak_pressure
from mrx.relaxation.seeding import energy_seed
from mrx.relaxation.shells import build_shells, label, load_target, pressure_error
from mrx.relaxation.twist import iota_fit, layer_iota, load_iota_target, retarget_iota, twist_field

#: The Newton steps after the twist in the user-facing test, in chunks of relax.py's default length with the label and
#: the shells rebuilt before each.
STEPS, CHUNK = 20, 10
#: The interior of the label where the user-facing test compares transforms.
S_TEST = np.linspace(0.05, 0.95, 19)
#: The transform's largest miss after the twist and after the steps, and the pressure's error after the steps:
#: 1.38e-3, 4.01e-3 and 7.59e-3 in float64, 4.14e-3 and 7.58e-3 after the steps in float32 (2026-11-18).
MISS_TWIST, MISS_END, PRESSURE_ERROR = 3e-3, 8e-3, 1.5e-2

def test_retarget_iota(seq, b0):
    iota_file = seq.equilibrium["profiles"]["iota"]           # a function of r = sqrt(s)

    def target(s):
        return 0.9 * np.abs(iota_file(np.sqrt(s)))
    B1, rec = retarget_iota(seq, b0, iota=target, bins=12, corrections=1)
    dB = B1 - b0
    assert abs(abs(rec["gain"]) - 1.0) < 1e-2                  # nfp dF/dPhi per unit twist, 0.9994 measured
    assert rec["div_dB_rel"] < 1e3 * eps()                     # 5.0e-15 measured in float64
    assert rec["wall_flux_dB"] < 1e3 * eps() * float(jnp.abs(b0).max())     # 1.1e-16 measured in float64
    dBz = seq.odd.evaluate_at_quadrature(dB, 2)[:, 2]
    assert float(jnp.abs(dBz).max()) < 1e3 * eps() * float(jnp.abs(seq.odd.evaluate_at_quadrature(b0, 2)).max())
    r, iota_r = layer_iota(seq, B1)
    i0 = layer_iota(seq, b0)[1]
    miss = np.abs(np.abs(iota_r) - target(r ** 2))
    # the layers' miss against the target over the largest change: 0.094 measured (6.2e-3 next to the axis, where the
    # label resolves s least, 0.04 to 0.066 change), the label's own fit misses by 8.4e-4
    assert miss[1:-1].max() < 0.12 * np.abs(np.abs(i0) - target(r ** 2)).max()

    # F flat on 0.2 < s < 0.6, C2 at the band's ends: no twist inside the band (the label here is s = r^2)
    T = seq.even.interpolate(lambda x: jnp.atleast_1d(1.0 - x[0] ** 2), 0)

    def F(s):
        return jnp.where(s < 0.2, -(0.2 - s) ** 3, jnp.where(s > 0.6, (s - 0.6) ** 3, 0.0))
    dB = twist_field(seq, T, lambda T0: 1.0 - T0, F)[1]           # the free 2-form
    dB_q = np.asarray(seq.odd.free.evaluate_at_quadrature(dB, 2))
    x = np.asarray(seq.quad.x)[:, 0]
    band = (x ** 2 > 0.25) & (x ** 2 < 0.55)
    # 8.7e-4 measured: the histopolation's tail of the twist outside the band
    assert np.abs(dB_q[band]).max() < 2e-3 * np.abs(dB_q).max()


def test_iota_and_pressure_targets_from_one_namelist(seq, b0, tmp_path):
    """The run of ``scripts/relax.py --iota.target input.x --pressure.target input.x --seed`` with ONE VMEC input
    namelist: the transform 0.95 times the file's and the pressure the file's reshaped by ``1 + 0.05 (1 - 2 s)``.

    The label is solved once for the twist and the seed. The twist meets the transform (largest miss on the label
    1.38e-3 against a change of 3.29e-2). The seed on the twisted field puts each of its 7 chains within 1.9e-3 of the
    target's resonant radius, against at least 6.1e-2 from the file's. Twenty Newton steps in two chunks, with the
    label and the shells rebuilt before each, take positive steps and lower ``W`` at every step. At the end the
    transform misses by 4.01e-3 (0.12 of the change) and the pressure by 7.59e-3, against 2.93e-2 from the file's
    pressure (0.26), in float64. float32 gives 4.14e-3 and 7.58e-3. So a run that ignored either target fails. The
    bounds are MISS_TWIST, MISS_END, PRESSURE_ERROR, 1e-2 for the seeds, a third of the transform's change, half the pressure's error
    against the file's profile, and half the smallest distance to the file's resonances. The pressure target moves
    the transform no more than the plain relaxation does here (4.28e-3 after ten plain Newton steps). 227 s on four
    CPU cores after the fixture (2026-11-18). Skipped in float32, where a Newton step costs three times as much on
    the CPU.
    """
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    iota_file, p_file = seq.equilibrium["profiles"]["iota"], seq.equilibrium["profiles"]["pressure"]
    s = np.linspace(0.0, 1.0, 101)
    row = lambda x: " ".join(f"{v:.17g}" for v in x)  # noqa: E731
    path = tmp_path / "input.target"
    path.write_text(f"&INDATA\n  NFP = {seq.nfp}, NCURR = 0, PIOTA_TYPE = 'cubic_spline'\n"
                    f"  AI_AUX_S = {row(s)}\n  AI_AUX_F = {row(0.95 * np.abs(iota_file(np.sqrt(s))))}\n"
                    f"  PMASS_TYPE = 'cubic_spline'\n  AM_AUX_S = {row(s)}\n"
                    f"  AM_AUX_F = {row(p_file(np.sqrt(s)) * (1.0 + 0.05 * (1.0 - 2.0 * s)))}\n/\n")
    iota_t = load_iota_target(str(path))["iota"]
    p_t, p_f = load_target(seq, str(path)).profile, load_target(seq, "file").profile
    want = np.abs(np.asarray(iota_t(S_TEST)))
    change_iota = np.abs(np.abs(iota_file(np.sqrt(S_TEST))) - want).max()

    def iota_miss(B, T):
        s_q, T, _ = label(seq, B, ANISOTROPY, T)
        return np.abs(np.abs(iota_fit(seq, B, np.asarray(s_q), 6)(S_TEST)) - want).max(), s_q, T

    _, T, _ = label(seq, b0, ANISOTROPY)
    B1, _ = retarget_iota(seq, b0, iota=iota_t, T=T, verbose=False)
    miss_twist = iota_miss(B1, T)[0]

    # the seed on the twisted field, as relax.py --seed with the same label: each chain at the target's resonant radius
    rows = energy_seed(seq, B1, T=T, verbose=False)[1]
    r_fine = np.linspace(0.02, 0.99, 20001)
    off_new, off_file = [], []
    for r in rows:
        res = seq.nfp * r["n"] / r["m"]
        off_new.append(abs(r["r"] - r_fine[np.argmin(np.abs(np.abs(np.asarray(iota_t(r_fine ** 2))) - res))]))
        off_file.append(abs(r["r"] - r_fine[np.argmin(np.abs(np.abs(iota_file(r_fine)) - res))]))

    ts = TimeStepper(seq=seq, newton=True)
    state, dW, dt_star, E0 = initial_state(B1, ts), [], [], None
    for _ in range(STEPS // CHUNK):
        s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
        ts = ts.with_shells(build_shells(seq, state.B_n, s_q, p_t(s_q), ts.newton_penalty, ts.newton_mass_solve_tol))
        res = relax(state, ts, steps=CHUNK, chunk=CHUNK, verbose=False)
        state, E0 = res.state, res.E0 if E0 is None else E0
        dW += list(np.asarray(res.trace["dW"], dtype=float))
        dt_star += list(np.asarray(res.trace["dt_star"], dtype=float))
    miss_end, s_q, T = iota_miss(state.B_n, T)
    p_w = weak_pressure(current(state.B_n, seq), state.B_n, seq)
    err, _ = pressure_error(seq, p_w, s_q, p_t(s_q), int(seq.ns[0]))
    err_file, _ = pressure_error(seq, p_w, s_q, p_f(s_q), int(seq.ns[0]))
    print(f"\n  iota miss {miss_twist:.3e} after the twist, {miss_end:.3e} at the end, change {change_iota:.3e}. "
          f"{len(rows)} seeded chains, off the target's resonance {max(off_new):.2e}, off the file's {min(off_file):.2e}. "
          f"|p - p*| / |p*| {err:.3e}, against the file's profile {err_file:.3e}. max dW / E0 {max(dW) / E0:+.1e}")
    assert miss_twist < MISS_TWIST
    assert rows and max(off_new) < 1e-2 and max(off_new) < 0.5 * min(off_file)
    assert min(dt_star) > 0.0
    assert max(dW) < eps() * E0, f"W not monotone: {dW}"
    assert miss_end < MISS_END and miss_end < change_iota / 3
    assert err < PRESSURE_ERROR and err < 0.5 * err_file
