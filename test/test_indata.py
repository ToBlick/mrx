"""The profiles of a VMEC input namelist (:mod:`mrx.equilibria.indata`).

Every profile type is checked against a closed form or an independent computation: types that describe one function
in two ways against each other, end values the type fixes, and the current profiles that integrate a density against
that density integrated by ``scipy.integrate.quad``. The current is scaled to ``CURTOR`` at the edge, the pressure is
``mu0 p`` in T^2.
"""
import numpy as np
from scipy.integrate import quad

from mrx.equilibria.indata import MU0, PCURR_TYPES, profile, read_indata

X = np.linspace(0.0, 1.0, 41)
EPS = np.finfo(np.float64).eps


def _write(tmp_path, body, name="input.test"):
    path = tmp_path / name
    path.write_text(f"&INDATA\n{body}\n/\n")
    return str(path)


def test_pmass_and_piota_types():
    """Each pmass and piota type in closed form, to round-off. two_power ``(1, 2, 2)`` is ``(1 - x^2)^2``, rational
    ``(1 - x^2) / (1 + x)`` is ``1 - x``, two_lorentz, gauss_trunc and pmass's pedestal fall from their axis value to
    zero at the edge, sum_atan reaches its edge value, the splines reproduce a quadratic (akima_spline up to the
    second-last knot, VMEC's right end) and line_segment a line (10 ms on the CPU, no compile)."""
    def pm(kind, c=(), s=(), f=()):
        return profile(kind, "pmass", c, s, f, X)

    def close(a, b):
        return np.max(np.abs(a - b)) < 50 * EPS * max(1.0, np.max(np.abs(b)))

    assert close(pm("two_power", (1.0, 2.0, 2.0)), pm("power_series", (1.0, 0.0, -2.0, 0.0, 1.0)))
    peak = 1.0 + 0.3 * np.exp(-((X - 0.4) / 0.1) ** 2)
    assert close(pm("two_power_gs", (2.0, 1.0, 1.0, 0.3, 0.4, 0.1)), 2.0 * (1.0 - X) * peak)
    rational = np.zeros(21)
    rational[[0, 2, 10, 11]] = 1.0, -1.0, 1.0, 1.0
    assert close(pm("rational", rational), 1.0 - X)
    for kind, c in (("two_lorentz", (3.0, 0.4, 0.5, 1.5, 2.0, 0.8, 1.0, 1.0)), ("gauss_trunc", (3.0, 0.6))):
        ends = pm(kind, c)[[0, -1]]
        assert close(ends, np.array([3.0, 0.0]))
    gauss = pm("gauss_trunc", (3.0, 0.6))
    edge = np.exp(-1.0 / 0.36)
    assert close(gauss, 3.0 * (np.exp(-(X / 0.6) ** 2) - edge) / (1.0 - edge))
    pedestal = np.zeros(21)
    pedestal[[0, 1, 17, 18, 19]] = 1.0, -1.0, 0.5, 0.8, 0.1
    assert close(pm("pedestal", pedestal)[[0, -1]], np.array([1.5, 0.0]))
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    quadratic = lambda x: 1.0 - 0.5 * x + 2.0 * x ** 2  # noqa: E731
    assert close(pm("cubic_spline", s=knots, f=quadratic(knots)), quadratic(X))
    inner = X <= knots[-2]                               # VMEC's akima knots beyond the right end are off the parabola
    assert close(pm("akima_spline", s=knots, f=quadratic(knots))[inner], quadratic(X[inner]))
    assert close(pm("line_segment", s=knots, f=1.0 - knots), 1.0 - X)

    def pi(kind, c=(), s=(), f=()):
        return profile(kind, "piota", c, s, f, X)

    assert close(pi("power_series", (0.4, 0.2)), 0.4 + 0.2 * X)
    assert close(pi("nice_quadratic", (0.4, 0.6, 0.1)), 0.4 + 0.2 * X + 0.4 * X * (1.0 - X))
    assert close(pi("rational", rational), 1.0 - X)
    atan = np.zeros(21)
    atan[:5] = 0.3, 0.2, 2.0, 1.0, 1.0
    y = X[:-1]
    assert close(pi("sum_atan", atan)[:-1], 0.3 + 0.4 / np.pi * np.arctan(2.0 * y / (1.0 - y)))
    assert close(pi("sum_atan", atan)[-1], 0.5)
    assert close(pi("cubic_spline", s=knots, f=quadratic(knots)), quadratic(X))
    assert close(pi("akima_spline", s=knots, f=quadratic(knots))[inner], quadratic(X[inner]))


def test_pcurr_types_integrate_their_density():
    """Each pcurr type that integrates a current density ``I'(s)`` against that density integrated by ``quad``, to
    ``1e-12`` of ``I(1)`` (1.1e-14 measured), and ``I(0) = 0``. two_power, two_power_gs and gauss_trunc are
    integrated by VMEC's 10-point Gauss rule on ``[0, s]``, part of their definition, which is exact for two_power
    with integer powers and within 2e-13 of the integral for the smooth densities here (a peak of width 0.1 is off
    by 1e-3, as in VMEC). The types that give ``I`` itself (power_series_i, sum_atan, rational, the ``_i`` splines)
    are the functions above (0.9 s on the CPU, no compile)."""
    knots = np.array([0.0, 0.2, 0.45, 0.7, 1.0])
    values = np.array([1.0, 0.8, 1.3, 0.4, 0.1])
    pedestal = np.zeros(21)
    pedestal[[0, 2, 8, 10, 11, 13, 15, 16, 17, 19, 20]] = 1.0, -0.5, 0.3, 0.6, 0.2, 0.2, 0.5, 0.05, 0.1, 0.8, 0.1
    a8, a12 = 0.05, 0.1
    amp = 1.0 / (np.tanh(2 * 0.6 / 0.2) - np.tanh(2 * (0.6 - 1.0) / 0.2))

    def sech2(y):
        return 1.0 / np.cosh(y) ** 2

    cases = {
        "power_series": ((1.0, -0.5, 0.3), lambda x: 1.0 - 0.5 * x + 0.3 * x ** 2),
        "two_power": ((2.0, 2.0, 2.0), lambda x: 2.0 * (1.0 - x ** 2) ** 2),
        "two_power_gs": ((2.0, 1.0, 1.0, 0.3, 0.4, 0.5),
                         lambda x: 2.0 * (1.0 - x) * (1.0 + 0.3 * np.exp(-((x - 0.4) / 0.5) ** 2))),
        "gauss_trunc": ((2.0, 0.5), lambda x: 2.0 * (np.exp(-(x / 0.5) ** 2) - np.exp(-4.0))),
        "sum_cossq_s": ((3.0, 1.0, 0.5, 2.0),
                        lambda x: sum(c * np.cos(np.pi * (x - xi) / 1.0) ** 2 * (abs(x - xi) < 0.5)
                                      for c, xi in ((1.0, 0.0), (0.5, 0.5), (2.0, 1.0)))),
        "sum_cossq_sqrts": ((3.0, 1.0, 0.5, 2.0),
                            lambda x: sum(0.5 * c * np.cos(np.pi * (np.sqrt(x) - xi)) ** 2 * (abs(np.sqrt(x) - xi) < 0.5)
                                          for c, xi in ((1.0, 0.0), (0.5, 0.5), (2.0, 1.0)))),
        "sum_cossq_s_free": ((1.0, 0.3, 0.2, 0.5, 0.7, 0.3),
                             lambda x: (np.cos(np.pi * (x - 0.3) / 0.4) ** 2 * (abs(x - 0.3) < 0.2)
                                        + 0.5 * np.cos(np.pi * (x - 0.7) / 0.6) ** 2 * (abs(x - 0.7) < 0.3))),
        "pedestal": (pedestal, lambda x: (1.0 - 0.5 * x ** 2
                                          + amp * 0.3 * sech2(2 * (0.6 - np.sqrt(x)) / 0.2) * 2 / 0.2 / (2 * np.sqrt(x))
                                          + 0.2 * sech2((x - 0.5) / a8) / a8 + 0.1 * sech2((x - 0.8) / a12) / a12)),
        "cubic_spline_ip": (None, None),
        "akima_spline_ip": (None, None),
        "line_segment_ip": (None, lambda x: np.interp(x, knots, values)),
    }
    for kind, (c, density) in cases.items():
        if density is None:          # the spline itself, the pmass type of the same name, is the density
            def density(x, kind=kind):
                return profile(kind[:-3], "pmass", (), knots, values, np.array([x]))[0]
        current = profile(kind, "pcurr", c, knots, values, X)
        breaks = [0.04, 0.2, 0.25, 0.3, 0.36, 0.45, 0.5, 0.7, 0.8]
        ref = np.array([quad(density, 0.0, x, points=[b for b in breaks if b < x] or None, limit=200,
                             epsabs=1e-15, epsrel=1e-13)[0] for x in X])
        err = np.max(np.abs(current - ref)) / abs(ref[-1])
        assert current[0] == 0.0 and err < 1e-12, (kind, err)


def test_namelist_profiles(tmp_path):
    """A namelist read back: ``mu0 p`` in T^2 (``PRES_SCALE = 1 / 2 mu0`` Pa on a unit profile is ``B^2 / 2`` of a
    field of 1 T), held beyond ``SPRES_PED``, its argument stretched by ``BLOAT``, the current ``mu0 I`` with
    ``I(1) = CURTOR`` for every pcurr type, the transform negated in MRX's angle unless VMEC flips the boundary, and
    ``NFP`` and the signed ``PHIEDGE`` (10 ms on the CPU, no compile)."""
    body = """NFP = 3, PHIEDGE = -0.5, NCURR = 1, CURTOR = -1.5D5, GAMMA = 0.0
  PMASS_TYPE = 'power_series', AM = 1.0 -1.0, PRES_SCALE = {scale}, SPRES_PED = 0.6, BLOAT = 1.0
  PIOTA_TYPE = 'power_series', AI = 0.4 0.2
  RBC(0,0) = 1.5, RBC(0,1) = 0.3, ZBS(0,1) = 0.3 ! counter-clockwise, as VMEC wants it
"""
    nl = read_indata(_write(tmp_path, body.format(scale=1.0 / (2.0 * MU0))))
    assert nl.nfp == 3 and nl.phiedge == -0.5 and nl.ncurr == 1 and not nl.flip
    p = nl.pressure(X)
    assert abs(p[0] - 0.5) < 4 * EPS
    assert np.max(np.abs(p - 0.5 * (1.0 - np.minimum(X, 0.6)))) < 4 * EPS
    stretched = read_indata(_write(tmp_path, body.format(scale=1.0).replace("BLOAT = 1.0", "BLOAT = 1.25"), "in2"))
    assert np.allclose(stretched.pressure(X), MU0 * (1.0 - np.minimum(1.25 * np.minimum(X, 0.6), 1.0)), rtol=0, atol=4 * EPS * MU0)
    assert np.max(np.abs(nl.iota(X) + 0.4 + 0.2 * X)) < 4 * EPS
    clockwise = read_indata(_write(tmp_path, body.format(scale=1.0).replace("ZBS(0,1) = 0.3", "ZBS(0,1) = -0.3"), "in3"))
    assert clockwise.flip and np.max(np.abs(clockwise.iota(X) - 0.4 - 0.2 * X)) < 4 * EPS
    knots = np.array([0.0, 0.2, 0.45, 0.7, 1.0])
    rng = np.random.default_rng(0)
    for kind in PCURR_TYPES:
        ac = rng.uniform(0.5, 1.5, 21)
        if kind in ("sum_atan", "rational"):
            ac[0] = 0.0                                  # these give I itself, and I(0) = AC(0)
        if kind in ("sum_cossq_s", "sum_cossq_sqrts"):
            ac[0] = 4.0                                  # the number of windows
        aux = "AC_AUX_S = " + " ".join(map(str, knots)) + ", AC_AUX_F = " + " ".join(map(str, knots * (2.0 - knots)))
        current = read_indata(_write(tmp_path, body.format(scale=1.0) + f"PCURR_TYPE = '{kind}', AC = "
                                     + " ".join(map(str, ac)) + ", " + aux, "in4")).current(np.array([0.0, 1.0]))
        assert abs(current[0]) < 8 * EPS * MU0 * 1.5e5 and abs(current[1] / (MU0 * -1.5e5) - 1.0) < 8 * EPS, kind

