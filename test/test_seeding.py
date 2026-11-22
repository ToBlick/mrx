"""Island seeds (:mod:`mrx.relaxation.seeding`) on the exact Solov'ev tokamak.

The transform falls from 0.8 to 0.65, so the surface ``iota = 3/4`` carries the resonance ``(m, n) = (4, 3)``. The
energy seed there is the combination of its radial profiles that lowers the energy the most: the seeded field is
``M``-orthogonal to the seed, ``(B + dB, dB)_M = 0``, and the energy falls by ``||dB||^2_M / 2``, both to the
tolerance of the solves.
The seed scaled to a finite amplitude opens a chain: the field line started inside it has the transform of the
rational surface.
"""
import numpy as np

from mrx.diagnostics.poincare import poincare
from mrx.precision import eps
from mrx.relaxation.seeding import energy_seed

#: The resonance of the seed.
IOTA = 3.0 / 4.0
#: The resonant normal field of the brute-force seed, ``|dB^r| / |B^zeta|`` on the resonant surface: an island about
#: 0.065 wide in ``sqrt(s)`` (0.029 at 1e-3 left the nearest line, 0.011 off the resonance, at 0.7478).
AMPLITUDE = 5e-3


def test_the_energy_seed_is_the_optimum_in_its_span(seq, b0):
    """The energy seed at :data:`IOTA`: ``(B + dB, dB)_M = 0`` and ``E(B + dB) - E(B) = -||dB||^2_M / 2``, relative to
    ``||B|| ||dB||``, to the tolerance of the solves that make the seed. On an exact equilibrium the optimum is
    small: ``||dB|| / ||B||`` 7.8e-10, the orthogonality 1.4e-9 measured in float64 (2026-11-22)."""
    seeded, rows = energy_seed(seq, b0, iotas=[IOTA], verbose=False)
    odd = seq.odd
    dB = seeded - b0
    scale = float(odd.l2_norm(b0, 2) * odd.l2_norm(dB, 2))
    orth = abs(float(seeded @ (odd.M[2] @ dB))) / scale
    dE = 0.5 * float(odd.l2_norm_sq(seeded, 2) - odd.l2_norm_sq(b0, 2))
    model = -0.5 * float(odd.l2_norm_sq(dB, 2))
    (row,) = rows
    print(f"\n  seed ({row['m']},{row['n']}) at x {row['x']:.3f}, |dB|/|B| "
          f"{float(odd.l2_norm(dB, 2) / odd.l2_norm(b0, 2)):.2e}, orthogonality {orth:.1e}, dE {dE:.3e} vs {model:.3e}")
    assert (row["m"], row["n"]) == (4, 3)
    assert orth < 1e3 * max(seq.tol, eps())
    assert abs(dE - model) < 1e3 * max(seq.tol, eps()) * scale


def test_a_resonant_seed_opens_a_chain(seq, b0):
    """The seed at :data:`IOTA` scaled to the resonant amplitude :data:`AMPLITUDE`: the traced transform of the line
    started inside the chain is :data:`IOTA`, locked to it, closer than the same line of the unseeded field. PENDING
    measured."""
    seeded, rows = energy_seed(seq, b0, iotas=[IOTA], amplitudes=[AMPLITUDE], verbose=False)
    (row,) = rows
    kw = dict(lines=8, periods=200, planes=(0.0,))
    iota_0 = np.abs(np.asarray(poincare(seq, b0, **kw)["iota"]))
    trace = poincare(seq, seeded, **kw)
    iota = np.abs(np.asarray(trace["iota"]))
    r = np.asarray(trace["seed_r"])
    k = int(np.argmin(np.abs(r - row["x"])))
    print(f"\n  chain at x {row['x']:.3f}, width {row['w']:.4f}, line {k} at r {r[k]:.3f}: iota {iota[k]:.6f} seeded, "
          f"{iota_0[k]:.6f} unseeded\n  iota {np.round(iota, 5)}")
    assert abs(r[k] - row["x"]) < row["w"] / 2
    assert abs(iota[k] - IOTA) < 0.1 * abs(iota_0[k] - IOTA)
