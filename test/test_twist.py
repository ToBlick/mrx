"""The twist move changes the rotational transform by ``nfp dF/dPhi`` and nothing else.

On the li383 fixture field (the VMEC equilibrium, whose logical surfaces ``r = sqrt(s)`` are its flux surfaces)
:func:`mrx.relaxation.twist.retarget_iota` twists the transform to 0.9 times the file's. The change is divergence-free
to round-off, has no flux through the wall and none through the sections ``zeta = const`` (no ``zeta`` component), and
the transform of the logical layers, measured apart from the label's bins the twist is built on, meets the target.
Where the twist's profile is flat it vanishes, as across an island, where the label is flat. The test takes 20 s on
one H200 next to the suite's fixtures (their setup 170 s), 11 s on four CPU cores.

A user's run with ``--iota.target`` ends at the target transform. On the fixture a target a few percent off the
file's transform is met by the twist and held by the Newton steps that follow it.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.flux_label import ANISOTROPY
from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.shells import label
from mrx.relaxation.twist import iota_fit, layer_iota, retarget_iota, twist_field

#: The Newton steps after the twist in the user-facing test, as in test_newton.py.
STEPS, CHUNK = 10, 5
#: The interior of the label where the user-facing test compares transforms.
S_TEST = np.linspace(0.05, 0.95, 19)
#: The transform's largest miss after the twist and after the steps: 1.38e-3 and 4.28e-3 in float64 (2026-11-17).
MISS_TWIST, MISS_HELD = 3e-3, 1e-2


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


def test_iota_target_reproduced_and_held(seq, b0):
    """The run of ``scripts/relax.py --iota.target``: the file's transform lowered to 0.95 times itself by
    :func:`mrx.relaxation.twist.retarget_iota`, then ten Newton steps. The transform of the field on its own flux label
    (:func:`mrx.relaxation.twist.iota_fit`, at 0.05 <= s <= 0.95) meets the target after the twist and is still held
    after the steps.

    The largest miss is 1.38e-3 after the twist and 4.28e-3 after the steps in float64 (1.38e-3 and 4.27e-3 in
    float32), against a change of 3.29e-2, so a run that ignored the target would miss it by eight times the measured
    error. The bounds are 3e-3, 1e-2 and a third of the change (measured 0.13 of it). 106 s on four CPU cores after
    the fixture (2026-11-17). Skipped in float32, where it meets the same numbers but takes 280 s.
    """
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    iota_file = seq.equilibrium["profiles"]["iota"]           # a function of r = sqrt(s)

    def target(s):
        return 0.95 * np.abs(iota_file(np.sqrt(s)))
    B1, _ = retarget_iota(seq, b0, iota=target, verbose=False)
    ts = TimeStepper(seq=seq, newton=True)
    res = relax(initial_state(B1, ts), ts, steps=STEPS, chunk=CHUNK, verbose=False)

    def iota_of(B):
        s_q = np.asarray(label(seq, B, ANISOTROPY)[0])
        return np.abs(iota_fit(seq, B, s_q, 6)(S_TEST))
    want = target(S_TEST)
    change = np.abs(np.abs(iota_file(np.sqrt(S_TEST))) - want).max()
    miss_start, miss_end = np.abs(iota_of(B1) - want).max(), np.abs(iota_of(res.state.B_n) - want).max()
    print(f"\n  change {change:.3e}, miss after the twist {miss_start:.3e}, after {STEPS} steps {miss_end:.3e}")
    assert miss_start < MISS_TWIST
    assert miss_end < MISS_HELD and miss_end < change / 3
