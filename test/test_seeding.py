"""Island seeds (:mod:`mrx.relaxation.seeding`) on the exact Solov'ev tokamak.

The transform falls from 0.8 to 0.65, so the surface ``iota = 3/4`` carries the resonance ``(m, n) = (4, 3)``. The
energy seed there, scaled to a finite resonant amplitude, opens a chain: field lines started on the resonant surface
are locked to the rational transform.
"""
import jax.numpy as jnp
import numpy as np

from mrx.diagnostics.poincare import logical_field, trace
from mrx.relaxation.seeding import energy_seed

#: The resonance of the seed.
IOTA = 3.0 / 4.0
#: Lines on the resonant surface, their periods and steps per period, and the transform's lock: PENDING.
PHASES, PERIODS, STEPS, LOCKED = 8, 300, 100, 1e-5
#: The resonant normal field of the brute-force seed, ``|dB^r| / |B^zeta|`` on the resonant surface: an island about
#: 0.065 wide in ``sqrt(s)`` (0.029 at 1e-3 left the nearest line, 0.011 off the resonance, at 0.7478).
AMPLITUDE = 5e-3


def _birkhoff_iota(seq, B, seeds, periods, steps):
    """The transform per full turn of the lines from the logical ``seeds`` ``(r, theta)`` at ``zeta = 0``: the
    weighted Birkhoff average of the poloidal angle's advance per field period about the pole, which is the magnetic
    axis here."""
    uv = np.asarray(trace(logical_field(seq, B), B, jnp.asarray(seeds), periods, steps))
    theta = np.unwrap(np.arctan2(uv[..., 1], uv[..., 0]), axis=1) / (2 * np.pi)
    d = np.diff(theta[:, ::steps], axis=1)
    t = (np.arange(periods) + 0.5) / periods
    w = np.exp(-1.0 / (t * (1.0 - t)))
    return np.abs(d @ w / w.sum()) * seq.nfp


def test_a_resonant_seed_opens_a_chain(seq, b0):
    """The seed at :data:`IOTA` scaled to the resonant amplitude :data:`AMPLITUDE`: lines started on the resonant
    surface at :data:`PHASES` phases over a period of the chain are locked to it, their Birkhoff transform over
    :data:`PERIODS` periods is :data:`IOTA` within :data:`LOCKED` for at least half of them (the others start near
    the X-points), and the same lines of the unseeded field are not. Measured PENDING."""
    seeded, rows = energy_seed(seq, b0, iotas=[IOTA], amplitudes=[AMPLITUDE], verbose=False)
    (row,) = rows
    seeds = np.stack([np.full(PHASES, row["x"]), np.arange(PHASES) / (PHASES * row["m"])], axis=1)
    miss = np.abs(_birkhoff_iota(seq, seeded, seeds, PERIODS, STEPS) - IOTA)
    miss_0 = np.abs(_birkhoff_iota(seq, b0, seeds, PERIODS, STEPS) - IOTA)
    print(f"\n  chain at x {row['x']:.3f}, width {row['w']:.4f}: |iota - 3/4| seeded {' '.join(f'{v:.1e}' for v in miss)}"
          f", unseeded {' '.join(f'{v:.1e}' for v in miss_0)}")
    assert np.sum(miss < LOCKED) >= PHASES // 2
    assert np.all(miss_0 > LOCKED)
