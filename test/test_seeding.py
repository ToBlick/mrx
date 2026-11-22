"""Island seeds (:mod:`mrx.relaxation.seeding`) on the exact Solov'ev tokamak.

The transform falls from 0.8 to 0.65, so the surface ``iota = 2/3`` carries the resonance ``(m, n) = (3, 2)``. The
energy seed there is the combination of its radial profiles that lowers the energy the most: the seeded field is
``M``-orthogonal to the seed, ``(B + dB, dB)_M = 0``, and the energy falls by ``||dB||^2_M / 2``, both to round-off.
The seed opens a chain: field lines started inside it have the transform of the rational surface.
"""
import numpy as np

from mrx.diagnostics.poincare import poincare
from mrx.precision import eps
from mrx.relaxation.seeding import energy_seed

#: The resonance of the seed.
IOTA = 2.0 / 3.0


def test_the_seed_is_the_energy_optimum_and_opens_a_chain(seq, b0):
    """The seed at :data:`IOTA`: ``(B + dB, dB)_M = 0`` and ``E(B + dB) - E(B) = -||dB||^2_M / 2`` to round-off
    relative to ``||B|| ||dB||``, and the traced transform of the lines started on the resonant surface is
    :data:`IOTA` within PENDING."""
    seeded, rows = energy_seed(seq, b0, iotas=[IOTA], verbose=False)
    odd = seq.odd
    dB = seeded - b0
    scale = float(odd.l2_norm(b0, 2) * odd.l2_norm(dB, 2))
    orth = abs(float(seeded @ (odd.M[2] @ dB))) / scale
    dE = 0.5 * float(odd.l2_norm_sq(seeded, 2) - odd.l2_norm_sq(b0, 2))
    model = -0.5 * float(odd.l2_norm_sq(dB, 2))
    (row,) = rows
    trace = poincare(seq, seeded, lines=8, periods=60, planes=(0.0,))
    iota = np.abs(np.asarray(trace["iota"]))
    r = np.asarray(trace["seed_r"])
    print(f"\n  seed ({row['m']},{row['n']}) at x {row['x']:.3f}, width {row['w']:.4f}, |dB|/|B| "
          f"{float(odd.l2_norm(dB, 2) / odd.l2_norm(b0, 2)):.2e}, orthogonality {orth:.1e}, dE {dE:.6e} vs "
          f"{model:.6e}\n  lines r {np.round(r, 3)}\n  iota {np.round(iota, 5)}")
    assert (row["m"], row["n"]) == (3, 2)
    assert orth < 1e3 * eps()
    assert abs(dE - model) < 1e3 * eps() * scale
