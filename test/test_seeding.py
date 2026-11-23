"""Island seeds (:mod:`mrx.relaxation.seeding`) run on the exact Solov'ev tokamak.

The transform falls from 0.8 to 0.65, so the surface ``iota = 3/4`` carries the resonance ``(m, n) = (4, 3)``. The
energy seed there, energy-optimal and at a fixed resonant amplitude, finds that chain and returns a finite 2-form
whose divergence stays at round-off (the seed is a curl). Optimality and the chain's opening return with a 3D shared
sequence (test_suite/PLAN.md, decision 2026-11-22).
"""
import numpy as np

from mrx.precision import eps
from mrx.relaxation.physics import compute_divergence_norm
from mrx.relaxation.seeding import energy_seed

#: The resonance of the seed and the fixed resonant amplitude ``|dB^r| / |B^zeta|``.
IOTA, AMPLITUDE = 3.0 / 4.0, 5e-3


def test_the_seed_runs(seq, b0):
    """The energy seed at :data:`IOTA`, energy-optimal and at :data:`AMPLITUDE`: the (4, 3) chain is found, and the
    seeded field is a finite 2-form of the field's space with ``||div B|| / ||B||`` at round-off."""
    for amplitudes in (None, [AMPLITUDE]):
        seeded, rows = energy_seed(seq, b0, iotas=[IOTA], amplitudes=amplitudes, verbose=False)
        div = float(compute_divergence_norm(seeded, seq) / seq.odd.l2_norm(seeded, 2))
        print(f"\n  amplitudes {amplitudes}: chains {[(r['m'], r['n']) for r in rows]}, div {div:.1e}")
        assert [(r["m"], r["n"]) for r in rows] == [(4, 3)]
        assert seeded.shape == b0.shape and np.all(np.isfinite(np.asarray(seeded)))
        assert div < 1e3 * eps()
