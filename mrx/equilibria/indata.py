"""The profiles of a VMEC input namelist (INDATA) as functions of the normalised toroidal flux.

:func:`read_indata` reads the ``&INDATA`` namelist of a VMEC input file (``input.<name>``) into an :class:`Indata`.
It holds the profiles a fixed-boundary run takes from the namelist, and what their meaning needs:

- ``NFP`` and ``PHIEDGE``, the toroidal flux inside the boundary in Wb with its sign. A positive one is a field along
  the geometric toroidal angle ``phi``, as in VMEC.
- The pressure ``PMASS_TYPE``, ``AM``, ``AM_AUX_S``, ``AM_AUX_F``, ``PRES_SCALE`` (Pa), ``SPRES_PED`` and ``BLOAT``
  (:meth:`Indata.pressure`).
- With ``NCURR = 0`` the transform ``PIOTA_TYPE``, ``AI``, ``AI_AUX_S``, ``AI_AUX_F`` (:meth:`Indata.iota`).
- With ``NCURR = 1`` the net toroidal current inside each surface ``PCURR_TYPE``, ``AC``, ``AC_AUX_S``, ``AC_AUX_F``,
  scaled so that ``I(1) = CURTOR`` (A, positive along ``phi``, :meth:`Indata.current`).
- The orientation of the boundary: the sums of the ``m = 1`` coefficients of ``RBC`` and ``ZBS``, which fix the sign
  of the transform in MRX's poloidal angle. The boundary itself is not read. MRX takes its geometry from an
  equilibrium file.

The profiles are VMEC's (``profile_functions.f``), the types in :data:`PMASS_TYPES`, :data:`PIOTA_TYPES` and
:data:`PCURR_TYPES`. Their argument is the normalised toroidal flux ``s``, so VMEC's ``APHI``, which only places
VMEC's radial grid, does not enter. ``BLOAT`` stretches the argument of the pressure and the current, as in VMEC.

Not supported, and an error when the namelist asks for it: ``LASYM = T`` (MRX is stellarator symmetric), ``LRFP = T``
(the poloidal flux as the radial label), ``NCURR`` other than 0 or 1, and ``GAMMA != 0``. With ``GAMMA != 0`` VMEC's
profile is the mass ``M(s)`` of an adiabatic law and the pressure is ``M(s) / V'(s)^GAMMA``, with ``V'`` the volume
derivative of the equilibrium VMEC finds. That pressure is not a profile of the namelist alone. VMEC's numerical
controls (``MPOL``, ``NTOR``, ``NS_ARRAY``, ``FTOL_ARRAY``, ``NITER_ARRAY``, ``DELT``, ``NSTEP``, ``APHI``, the
preconditioner and the output flags) are not read, and neither is anything else in the namelist.

The angles. VMEC's poloidal angle runs counter-clockwise in the ``(R, Z)`` plane (it flips a clockwise boundary, then
negates ``AI``), MRX's clockwise, so that ``(r, theta, zeta)`` are right-handed. :meth:`Indata.iota` is the
transform in MRX's angle. The current does not depend on the poloidal angle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
TWO_PI = 2.0 * np.pi
#: VMEC's vacuum permeability, ``4 pi 1e-7`` exactly (``stel_constants``), which converts Pa to T^2 and A to T m.
MU0 = 4e-7 * np.pi

#: ``PMASS_TYPE`` values, the pressure ``p(s)``.
PMASS_TYPES = ("power_series", "two_power", "two_power_gs", "two_lorentz", "gauss_trunc", "pedestal", "rational",
               "cubic_spline", "akima_spline", "line_segment")
#: ``PIOTA_TYPE`` values, the transform ``iota(s)``.
PIOTA_TYPES = ("power_series", "sum_atan", "nice_quadratic", "rational", "cubic_spline", "akima_spline",
               "line_segment")
#: ``PCURR_TYPE`` values, the current ``I(s)`` (or its derivative ``I'(s)``, integrated).
PCURR_TYPES = ("power_series", "power_series_i", "two_power", "two_power_gs", "gauss_trunc", "sum_atan", "rational",
               "sum_cossq_s", "sum_cossq_sqrts", "sum_cossq_s_free", "pedestal", "cubic_spline_i",
               "cubic_spline_ip", "akima_spline_i", "akima_spline_ip", "line_segment_i", "line_segment_ip")

# VMEC's 10-point Gauss-Legendre rule on [0, 1] (profile_functions.f, glx and glw), with which it integrates the
# current densities I'(s) of two_power, two_power_gs and gauss_trunc: part of those profiles' definition
_GL_X = np.array([0.01304673574141414, 0.06746831665550774, 0.1602952158504878, 0.2833023029353764,
                  0.4255628305091844, 0.5744371694908156, 0.7166976970646236, 0.8397047841495122,
                  0.9325316833444923, 0.9869532642585859])
_GL_W = np.array([0.03333567215434407, 0.0747256745752903, 0.1095431812579910, 0.1346333596549982,
                  0.1477621123573764, 0.1477621123573764, 0.1346333596549982, 0.1095431812579910,
                  0.0747256745752903, 0.03333567215434407])


# --- the namelist ----------------------------------------------------------------------------------------------------

_HEAD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)=]*\))?\s*=")


def _strip_comments(text):
    """``text`` without its ``!`` comments (outside quotes)."""
    out = []
    for line in text.splitlines():
        quote = None
        for i, ch in enumerate(line):
            if quote:
                quote = None if ch == quote else quote
            elif ch in "'\"":
                quote = ch
            elif ch == "!":
                line = line[:i]
                break
        out.append(line)
    return "\n".join(out)


def _value(token):
    """One namelist value: a string, a logical, an integer or a real (Fortran ``d`` exponents too)."""
    if token[0] in "'\"":
        return token[1:-1]
    low = token.lower()
    if low in ("t", ".t.", ".true.", "true"):
        return True
    if low in ("f", ".f.", ".false.", "false"):
        return False
    low = low.replace("d", "e")
    try:
        return int(low)
    except ValueError:
        return float(low)


def _values(chunk):
    """The values of one assignment, repeat counts ``n*v`` expanded."""
    tokens = re.findall(r"'[^']*'|\"[^\"]*\"|[^\s,]+", chunk)
    out = []
    for tok in tokens:
        rep = re.fullmatch(r"(\d+)\*(.+)", tok)
        out += [_value(rep.group(2))] * int(rep.group(1)) if rep else [_value(tok)]
    return out


def parse_namelist(text, group="indata"):
    """The assignments of the namelist ``&group`` in ``text``: a dict from the upper-case name to a dict from the
    index tuple (empty for an unindexed name) to the list of values, which fill the array from that index on."""
    text = _strip_comments(text)
    start = re.search(rf"&{group}\b", text, flags=re.IGNORECASE)
    if start is None:
        raise ValueError(f"no &{group.upper()} namelist")
    body = text[start.end():]
    end = re.search(r"(^|\s)/\s*($|\n)|&end\b", body, flags=re.IGNORECASE | re.MULTILINE)
    body = body if end is None else body[:end.start()]
    heads = list(_HEAD.finditer(body))
    out = {}
    for h, nxt in zip(heads, heads[1:] + [None]):
        chunk = body[h.end():nxt.start() if nxt else len(body)]
        idx = () if h.group(2) is None else tuple(int(i) for i in h.group(2)[1:-1].split(","))
        out.setdefault(h.group(1).upper(), {})[idx] = _values(chunk)
    return out


def _scalar(nl, name, default):
    entry = nl.get(name)
    if entry is None:
        return default
    if list(entry) != [()] or len(entry[()]) != 1:
        raise ValueError(f"{name} is a scalar, the namelist gives {entry}")
    return entry[()][0]


def _vector(nl, name, lower):
    """The 1D array ``name`` with first index ``lower``: unindexed values fill it from ``lower`` on, ``NAME(i) = ...``
    from ``i`` on. Unset entries are zero."""
    entry = nl.get(name, {})
    pos = {}
    for idx, vals in entry.items():
        first = lower if idx == () else idx[0]
        for k, v in enumerate(vals):
            pos[first + k] = float(v)
    if not pos:
        return np.zeros(0)
    if min(pos) < lower:
        raise ValueError(f"{name}({min(pos)}): the index starts at {lower}")
    out = np.zeros(max(pos) - lower + 1)
    for i, v in pos.items():
        out[i - lower] = v
    return out


def _aux(nl, name):
    """A knot array ``*_AUX_S`` or ``*_AUX_F``. VMEC takes the knots up to the first that does not increase."""
    return _vector(nl, name, 1)


def _knots(s, f):
    """The knots of a tabulated profile: the leading strictly increasing part of ``s``, ``f`` cut to it."""
    n = min(s.size, f.size)
    cut = next((i for i in range(1, n) if s[i] <= s[i - 1]), n)
    return s[:cut], f[:cut]


# --- the profile functions (profile_functions.f) ------------------------------------------------------------------


def _padded(c, n):
    c = np.asarray(c, dtype=np.float64)
    return np.pad(c, (0, max(0, n - c.size)))


def _poly(c, x):
    """``sum_i c[i] x^i`` by Horner."""
    y = np.zeros_like(x)
    for ci in np.asarray(c, dtype=np.float64)[::-1]:
        y = y * x + ci
    return y


def _two_power(c, x):
    c = _padded(c, 3)
    return c[0] * np.maximum(1.0 - x ** c[1], 0.0) ** c[2]


def _two_power_gs(c, x):
    """two_power times ``1 + sum c[i] exp(-((x - c[i+1]) / c[i+2])^2)``, ``i = 3, 6, .., 18``. A peak of zero amplitude
    is no peak and is skipped (VMEC evaluates its ``0 / 0`` width at the axis)."""
    c = _padded(c, 21)
    peaks = np.ones_like(x)
    for i in range(3, 19, 3):
        if c[i] != 0.0:
            peaks = peaks + c[i] * np.exp(-((x - c[i + 1]) / c[i + 2]) ** 2)
    return _two_power(c, x) * peaks


def _two_lorentz(c, x):
    c = _padded(c, 8)

    def term(width, power, exponent):
        shape = 1.0 / (1.0 + (x / width ** 2) ** power) ** exponent
        edge = 1.0 / (1.0 + (1.0 / width ** 2) ** power) ** exponent
        return (shape - edge) / (1.0 - edge)
    return c[0] * (c[1] * term(c[2], c[3], c[4]) + (1.0 - c[1]) * term(c[5], c[6], c[7]))


def _gauss_trunc(c, x):
    """The truncated Gaussian, ``c0 / (1 - E) (exp(-(x / c1)^2) - E)``, ``E = exp(-1 / c1^2)``: ``c0`` at the axis, zero
    at the edge."""
    c = _padded(c, 2)
    edge = np.exp(-(1.0 / c[1]) ** 2)
    return c[0] / (1.0 - edge) * (np.exp(-(x / c[1]) ** 2) - edge)


def _gauss_trunc_current(c, x):
    """pcurr's gauss_trunc density, ``c0 (exp(-(x / c1)^2) - exp(-1 / c1^2))``, not normalised (unlike pmass's)."""
    c = _padded(c, 2)
    return c[0] * (np.exp(-(x / c[1]) ** 2) - np.exp(-(1.0 / c[1]) ** 2))


def _rational(c, x):
    """``c[0:10]`` over ``c[10:21]``, both polynomials in ``x``."""
    c = _padded(c, 21)
    return _poly(c[:10], x) / _poly(c[10:21], x)


def _nice_quadratic(c, x):
    c = _padded(c, 3)
    return c[0] * (1.0 - x) + c[1] * x + 4.0 * c[2] * x * (1.0 - x)


def _sum_atan(c, x):
    """``c0 + 2 / pi sum_k c[1+4k] atan(c[2+4k] x^c[3+4k] / (1 - x)^c[4+4k])``, ``k = 0..4``, and at ``x = 1`` VMEC's
    edge value ``c0 + c1 + c5 + c9 + c13 + c17``."""
    c = _padded(c, 21)
    inner = x < 1.0
    xi = np.where(inner, x, 0.5)
    total = np.zeros_like(x)
    for k in range(5):
        if c[1 + 4 * k] != 0.0:
            total = total + c[1 + 4 * k] * np.arctan(c[2 + 4 * k] * xi ** c[3 + 4 * k] / (1.0 - xi) ** c[4 + 4 * k])
    return np.where(inner, c[0] + 2.0 / np.pi * total, c[0] + c[1] + c[5] + c[9] + c[13] + c[17])


def _pedestal(c, x):
    """pmass's pedestal: the degree-15 series ``c[0:16]`` plus ``A c17 (tanh(2 (c18 - sqrt x) / c19) - tanh(2 (c18 - 1)
    / c19))``, ``A = 1 / (tanh(2 c18 / c19) - tanh(2 (c18 - 1) / c19))``, the pedestal only for ``c19 > 0``."""
    c = _padded(c, 21)
    y = _poly(c[:16], x)
    if c[19] <= 0.0:
        return y
    amp = 1.0 / (np.tanh(2.0 * c[18] / c[19]) - np.tanh(2.0 * (c[18] - 1.0) / c[19]))
    return y + amp * c[17] * (np.tanh(2.0 * (c[18] - np.sqrt(x)) / c[19]) - np.tanh(2.0 * (c[18] - 1.0) / c[19]))


def _pedestal_current(c, x):
    """pcurr's pedestal, ``I(x)``: ``sum_{i<8} c[i] x^(i+1) / (i+1)``, a tanh pedestal ``c8`` in ``sqrt x`` (for
    ``c11 > 0``), and two tanh steps ``c13`` and ``c17`` in ``x``, their widths at least 0.01."""
    c = _padded(c, 21)
    y = x * _poly(c[:8] / np.arange(1, 9), x)
    if c[11] > 0.0:
        amp = 1.0 / (np.tanh(2.0 * c[10] / c[11]) - np.tanh(2.0 * (c[10] - 1.0) / c[11]))
        y = y + amp * c[8] * (np.tanh(2.0 * c[10] / c[11]) - np.tanh(2.0 * (c[10] - np.sqrt(x)) / c[11]))
    a8, a12 = max(c[16], 0.01), max(c[20], 0.01)
    y = y + c[13] * (np.tanh((x - c[15]) / a8) - np.tanh(-c[15] / a8))
    return y + c[17] * (np.tanh((x - c[19]) / a12) - np.tanh(-c[19] / a12))


def _cossq_count(c):
    n = int(c[0]) if len(c) else 0
    if not 1 <= n <= 20:
        raise ValueError(f"the sum_cossq wave count AC(0) = {n} is outside VMEC's 1..20")
    return n


def _sum_cossq_s(c, x):
    """``I(x)`` of ``c0`` cos^2 windows of the current density evenly placed in ``s`` (J. Geiger), wave ``i`` peaked
    at ``(i - 1) dx``, ``dx = 1 / (c0 - 1)``, integrated in closed form."""
    n = _cossq_count(c)
    c = _padded(c, max(21, n + 1))
    dx = 1.0 / (n - 1.0) if n > 1 else np.inf
    reached = np.zeros_like(x)                  # no window reached at the axis, I(0) = 0
    win = []
    for i in range(1, n + 1):
        xi = (i - 1) * dx
        within = (x > max(0.0, xi - dx)) & (x <= min(xi + dx, 1.0))
        win.append((within, xi))
        reached = np.where(within, float(i), reached)
    total = np.zeros_like(x)
    for i, (within, xi) in enumerate(win, start=1):
        wave = np.sin(np.pi * (x - xi) / dx) * dx / np.pi
        partial = 0.5 * c[i] * (x + wave) if i == 1 else 0.5 * c[i] * (x - xi + dx + wave)
        complete = 0.5 * c[i] * dx if i == 1 else c[i] * dx
        total = total + np.where(i <= reached, np.where(within, partial, complete), 0.0)
    return total


def _sum_cossq_sqrts(c, x):
    """The windows of :func:`_sum_cossq_s` placed in ``sqrt(s)``."""
    n = _cossq_count(c)
    c = _padded(c, max(21, n + 1))
    root = np.sqrt(x)
    dx = 1.0 / (n - 1.0) if n > 1 else np.inf
    dx2, pi2 = dx * dx, np.pi * np.pi
    reached = np.zeros_like(x)                  # no window reached at the axis, I(0) = 0
    win = []
    for i in range(1, n + 1):
        xi = (i - 1) * dx
        start = max(0.0, xi - dx)
        within = (root > start) & (root <= min(xi + dx, 1.0))
        win.append((within, xi, start))
        reached = np.where(within, float(i), reached)
    total = np.zeros_like(x)
    for i, (within, xi, start) in enumerate(win, start=1):
        phase = np.pi * (root - xi) / dx
        if i == 1:
            partial = c[i] * (0.25 * x + dx / TWO_PI * root * np.sin(phase) + dx2 / (2.0 * pi2) * (np.cos(phase) - 1.0))
            complete = c[i] * (0.25 - 1.0 / pi2) * dx2
        else:
            partial = c[i] * (0.25 * x + dx / TWO_PI * root * np.sin(phase) + dx2 / (2.0 * pi2) * np.cos(phase)
                              - 0.25 * start ** 2 + dx2 / (2.0 * pi2))
            complete = c[i] * dx * xi
        total = total + np.where(i <= reached, np.where(within, partial, complete), 0.0)
    return total


def _sum_cossq_s_free(c, x):
    """Seven freely placed cos^2 windows, ``c[3i:3i+3]`` the amplitude, centre and half width of wave ``i``, a zero
    amplitude no wave."""
    c = _padded(c, 21)
    total = np.zeros_like(x)
    for i in range(7):
        amp, xi, w = c[3 * i:3 * i + 3]
        if amp == 0.0:
            continue
        start, end = max(0.0, xi - w), min(xi + w, 1.0)
        s0 = np.sin(np.pi * (start - xi) / w)
        partial = 0.5 * amp * ((x - start) + w / np.pi * (np.sin(np.pi * (x - xi) / w) - s0))
        complete = 0.5 * amp * ((end - start) + w / np.pi * (np.sin(np.pi * (end - xi) / w) - s0))
        total = total + np.where((x > start) & (x <= end), partial, np.where(x > end, complete, 0.0))
    return total


def _integrated(density):
    """``I(x) = int_0^x density(c, t) dt`` by VMEC's 10-point rule on ``[0, x]``."""
    def current(c, x):
        return x * (density(c, x[..., None] * _GL_X) @ _GL_W)
    return current


def _power_series_current(c, x):
    """pcurr's power_series: ``I'(x) = sum_i c[i] x^i`` integrated exactly."""
    c = np.asarray(c, dtype=np.float64)
    return x * _poly(c / np.arange(1, c.size + 1), x)


def _power_series_i(c, x):
    return x * _poly(c, x)


def _cubic_spline(s, f, x, integrate):
    """VMEC's ``spline_cubic`` through the knots: the cubic spline with the end slopes of the parabolas through the
    three end knots, or its integral from ``s[0]``. ``x`` is clipped to the knots."""
    from scipy.interpolate import CubicSpline  # noqa: PLC0415
    if s.size == 2:
        slope = (f[1] - f[0]) / (s[1] - s[0])
        ends = (slope, slope)
    else:
        def end(i0, i1, i2):
            c = ((f[i2] - f[i0]) / (s[i2] - s[i0]) - (f[i1] - f[i0]) / (s[i1] - s[i0])) / (s[i2] - s[i1])
            return (f[i1] - f[i0]) / (s[i1] - s[i0]) - c * (s[i1] - s[i0])
        ends = (end(0, 1, 2), end(-1, -2, -3))
    spline = CubicSpline(s, f, bc_type=((1, ends[0]), (1, ends[1])))
    xc = np.clip(x, s[0], s[-1])
    return spline.antiderivative()(xc) if integrate else spline(xc)


def _line_segment(s, f, x, integrate):
    xc = np.clip(x, s[0], s[-1])
    if not integrate:
        return np.interp(xc, s, f)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(s) * (f[1:] + f[:-1]))])
    i = np.clip(np.searchsorted(s, xc, side="right"), 1, s.size - 1) - 1
    dx = xc - s[i]
    return cum[i] + f[i] * dx + 0.5 * (f[i + 1] - f[i]) / (s[i + 1] - s[i]) * dx * dx


def _akima_spline(s, f, x, integrate):
    """VMEC's ``spline_akima`` (LIBSTELL's): Akima's tangents with two knots added at each end, or its integral from
    ``s[0]``. The knots added at the left end lie on the parabola through the three end knots. Those at the right end
    are VMEC's, the same construction with the sign of the curvature reversed and the slope of the middle knot, so a
    quadratic is reproduced up to the second-last knot only. VMEC stops for fewer than four knots."""
    n = s.size
    if n < 4:
        raise ValueError("an akima_spline profile needs at least four knots (VMEC stops on fewer)")
    xl = np.zeros(n + 4)
    yl = np.zeros(n + 4)
    xl[2:n + 2], yl[2:n + 2] = s, f
    xl[0], xl[1] = 2 * xl[2] - xl[4], xl[2] + xl[3] - xl[4]
    xl[n + 3], xl[n + 2] = 2 * xl[n + 1] - xl[n - 1], xl[n + 1] + xl[n] - xl[n - 1]
    m = np.zeros(n + 3)
    m[2:n + 1] = np.diff(yl[2:n + 2]) / np.diff(xl[2:n + 2])
    cl = (m[3] - m[2]) / (xl[4] - xl[2])
    bl = m[2] - cl * (xl[3] - xl[2])
    cr = (m[n - 1] - m[n]) / (xl[n + 1] - xl[n - 1])
    br = m[n - 1] - cr * (xl[n] - xl[n - 1])
    for j in (1, 0):
        yl[j] = yl[2] + bl * (xl[j] - xl[2]) + cl * (xl[j] - xl[2]) ** 2
    for j in (n + 2, n + 3):
        yl[j] = yl[n + 1] + br * (xl[j] - xl[n + 1]) + cr * (xl[j] - xl[n + 1]) ** 2
    for j in (0, 1, n + 1, n + 2):
        m[j] = (yl[j + 1] - yl[j]) / (xl[j + 1] - xl[j])
    dm = np.abs(np.diff(m))
    t = np.empty(n)
    for k, i in enumerate(range(1, n + 1)):
        den = dm[i + 1] + dm[i - 1]
        t[k] = 0.5 * (m[i] + m[i + 1]) if den == 0.0 else (dm[i + 1] * m[i] + dm[i - 1] * m[i + 1]) / den
    h = np.diff(s)
    slope = np.diff(f) / h
    a, b = f[:-1], t[:-1]
    c = (3.0 * slope - t[1:] - 2.0 * t[:-1]) / h
    d = (t[1:] + t[:-1] - 2.0 * slope) / h ** 2
    xc = np.clip(x, s[0], s[-1])
    i = np.clip(np.searchsorted(s, xc, side="right"), 1, n - 1) - 1
    dx = xc - s[i]
    if not integrate:
        return a[i] + dx * (b[i] + dx * (c[i] + d[i] * dx))
    full = np.concatenate([[0.0], np.cumsum(a * h + b * h ** 2 / 2 + c * h ** 3 / 3 + d * h ** 4 / 4)])
    return full[i] + dx * (a[i] + dx * (b[i] / 2 + dx * (c[i] / 3 + d[i] * dx / 4)))


_TABULATED = {"cubic_spline": _cubic_spline, "akima_spline": _akima_spline, "line_segment": _line_segment}
_PMASS = {"power_series": _poly, "two_power": _two_power, "two_power_gs": _two_power_gs,
          "two_lorentz": _two_lorentz, "gauss_trunc": _gauss_trunc, "pedestal": _pedestal, "rational": _rational}
_PIOTA = {"power_series": _poly, "sum_atan": _sum_atan, "nice_quadratic": _nice_quadratic, "rational": _rational}
_PCURR = {"power_series": _power_series_current, "power_series_i": _power_series_i,
          "two_power": _integrated(_two_power), "two_power_gs": _integrated(_two_power_gs),
          "gauss_trunc": _integrated(_gauss_trunc_current), "sum_atan": _sum_atan, "rational": _rational,
          "sum_cossq_s": _sum_cossq_s, "sum_cossq_sqrts": _sum_cossq_sqrts, "sum_cossq_s_free": _sum_cossq_s_free,
          "pedestal": _pedestal_current}


def profile(kind, table, c, aux_s, aux_f, x):
    """The profile ``kind`` of ``table`` (``"pmass"``, ``"piota"`` or ``"pcurr"``) at the points ``x`` of ``[0, 1]``,
    from the coefficients ``c`` or the knots ``aux_s``, ``aux_f``: VMEC's ``pmass``, ``piota`` or ``pcurr`` without
    their scale factors (``pcurr`` is ``I(x)`` up to its scale, the ``_ip`` types and the I' types integrated)."""
    kinds = dict(pmass=PMASS_TYPES, piota=PIOTA_TYPES, pcurr=PCURR_TYPES)[table]
    key = str(kind).strip().lower()
    if key not in kinds:
        raise ValueError(f"{table} type {kind!r} is not supported: one of {kinds}")
    x = np.asarray(x, dtype=np.float64)
    base = key.rsplit("_", 1)[0] if table == "pcurr" and key.endswith(("_i", "_ip")) else key
    if base in _TABULATED:
        s, f = _knots(np.asarray(aux_s, dtype=np.float64), np.asarray(aux_f, dtype=np.float64))
        if s.size < 2:
            raise ValueError(f"{table} type {kind!r} needs at least two knots in its _AUX_S and _AUX_F")
        return _TABULATED[base](s, f, x, key.endswith("_ip"))
    return dict(pmass=_PMASS, piota=_PIOTA, pcurr=_PCURR)[table][key](np.asarray(c, dtype=np.float64), x)


# --- the namelist's profiles ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Indata:
    """The profiles of a VMEC input namelist (:func:`read_indata`). Fields are the namelist's values in VMEC's units,
    arrays from VMEC's first index (``AM(0)``, ``AM_AUX_S(1)``). ``flip`` says whether VMEC flips the namelist's
    poloidal angle: its boundary runs clockwise (``readin.f``: the product of the sums of the ``m = 1`` coefficients
    of ``RBC`` and ``ZBS`` is negative)."""
    path: str
    nfp: int
    phiedge: float
    flip: bool
    pmass_type: str
    am: np.ndarray
    am_aux_s: np.ndarray
    am_aux_f: np.ndarray
    pres_scale: float
    spres_ped: float
    bloat: float
    ncurr: int
    pcurr_type: str
    ac: np.ndarray
    ac_aux_s: np.ndarray
    ac_aux_f: np.ndarray
    curtor: float
    piota_type: str
    ai: np.ndarray
    ai_aux_s: np.ndarray
    ai_aux_f: np.ndarray

    def _x(self, s):
        """VMEC's profile argument ``min(|s bloat|, 1)``."""
        return np.minimum(np.abs(np.asarray(s, dtype=np.float64) * self.bloat), 1.0)

    def pressure(self, s):
        """``mu0 p(s)`` in T^2, VMEC's ``mu0 PRES_SCALE pmass`` with ``mu0 = 4 pi 1e-7``, held at its value at
        ``SPRES_PED`` beyond it. ``mu0 p = B^2 / 2`` is the pressure of a field of ``B`` T at beta one."""
        s = np.asarray(s, dtype=np.float64)

        def p(x):
            return MU0 * self.pres_scale * profile(self.pmass_type, "pmass", self.am, self.am_aux_s, self.am_aux_f, x)
        ped = abs(self.spres_ped)
        out = p(self._x(s))
        return np.where(s > ped, p(self._x(ped)), out) if ped < 1.0 else out

    def current(self, s):
        """``mu0 I(s)`` in T m, the net toroidal current inside the surface ``s`` along ``phi``: VMEC's ``pcurr``
        scaled to ``CURTOR`` at the edge."""
        if self.curtor == 0.0:
            return np.zeros_like(np.asarray(s, dtype=np.float64))
        shape = profile(self.pcurr_type, "pcurr", self.ac, self.ac_aux_s, self.ac_aux_f, self._x(s))
        edge = float(profile(self.pcurr_type, "pcurr", self.ac, self.ac_aux_s, self.ac_aux_f, self._x(1.0)))
        if edge == 0.0:
            raise ValueError(f"{self.path}: the current profile {self.pcurr_type} AC = {self.ac} has no net current, "
                             f"so CURTOR cannot scale it")
        return MU0 * self.curtor * shape / edge

    def iota(self, s):
        """The rotational transform in MRX's poloidal angle (clockwise), VMEC's ``piota`` negated unless VMEC flips
        the namelist's angle."""
        value = profile(self.piota_type, "piota", self.ai, self.ai_aux_s, self.ai_aux_f, np.asarray(s, np.float64))
        return value if self.flip else -value


def read_indata(path):
    """Read the ``&INDATA`` namelist of the VMEC input file ``path`` (:class:`Indata`)."""
    with open(path) as fh:
        nl = parse_namelist(fh.read())
    if _scalar(nl, "LASYM", False):
        raise ValueError(f"{path}: LASYM = T, but MRX is stellarator symmetric (R a cosine and Z a sine series)")
    if _scalar(nl, "LRFP", False):
        raise ValueError(f"{path}: LRFP = T (the poloidal flux as the radial label) is not supported, MRX's label is the "
                         f"toroidal flux")
    ncurr = int(_scalar(nl, "NCURR", 0))
    if ncurr not in (0, 1):
        raise ValueError(f"{path}: NCURR = {ncurr}, VMEC's NCURR is 0 (iota) or 1 (current)")

    if float(_scalar(nl, "GAMMA", 0.0)) != 0.0:
        raise ValueError(f"{path}: GAMMA != 0, VMEC's pressure is then a mass over V'(s)^GAMMA of the equilibrium it "
                         f"finds, not a profile of the namelist")

    def m1_sum(name):
        """The sum of the ``m = 1`` coefficients ``NAME(n, 1)``."""
        return sum(float(vals[0]) for idx, vals in nl.get(name, {}).items() if idx[1:] == (1,))

    return Indata(
        path=str(path), nfp=int(_scalar(nl, "NFP", 1)), phiedge=float(_scalar(nl, "PHIEDGE", 1.0)),
        flip=m1_sum("RBC") * m1_sum("ZBS") < 0.0,
        pmass_type=str(_scalar(nl, "PMASS_TYPE", "power_series")), am=_vector(nl, "AM", 0),
        am_aux_s=_aux(nl, "AM_AUX_S"), am_aux_f=_aux(nl, "AM_AUX_F"),
        pres_scale=float(_scalar(nl, "PRES_SCALE", 1.0)), spres_ped=float(_scalar(nl, "SPRES_PED", 1.0)),
        bloat=float(_scalar(nl, "BLOAT", 1.0)), ncurr=ncurr,
        pcurr_type=str(_scalar(nl, "PCURR_TYPE", "power_series")), ac=_vector(nl, "AC", 0),
        ac_aux_s=_aux(nl, "AC_AUX_S"), ac_aux_f=_aux(nl, "AC_AUX_F"), curtor=float(_scalar(nl, "CURTOR", 0.0)),
        piota_type=str(_scalar(nl, "PIOTA_TYPE", "power_series")), ai=_vector(nl, "AI", 0),
        ai_aux_s=_aux(nl, "AI_AUX_S"), ai_aux_f=_aux(nl, "AI_AUX_F"))
