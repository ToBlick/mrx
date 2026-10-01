"""Using GVEC's ``GVEC_State_*.dat`` files as input.

The state file holds:
- the radial B-spline basis (degree ``deg`` on the element grid ``sp``)
- Fourier modes ``(m, n)`` (``n`` multiplied by ``nfp``)
- for every mode, the radial B-spline coefficients of
    - ``X1 = R`` (cosine series)
    - ``X2 = Z`` (sine series)
    - ``LA = lambda`` (sine series, in radians)
  or, without stellarator symmetry (``sin_cos = 3``), of both series, the sine modes listed first
- the profiles ``Psi``, ``chi``, ``iota``, ``p`` at the interpolation points of the ``X1`` basis
- ``a_minor``, ``r_major``, ``volume``
The series have argument ``m theta - n zeta`` in radians, ``zeta`` the full-turn toroidal angle.
GVEC's radial label is MRX' ``r`` (``Psi = Psi_edge r^2``). The fields keep GVEC's radial B-splines
unchanged. Each profile is interpolated by a spline on the knots of the ``X1`` basis.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from mrx.geometry import knot_vector


def _numbers(line):
    return [float(v) for v in line.replace(",", " ").split()]


def read_state(path):
    """Read a GVEC state file and return its state (:mod:`mrx.equilibria`). The state also holds the
    poloidal flux profile ``chi`` and GVEC's minor and major radius ``a_minor``, ``r_major``."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    heads = [i for i, ln in enumerate(lines) if ln.startswith("##")]
    blocks = []                                     # (header text, data lines)
    for j, i in enumerate(heads):
        end = heads[j + 1] if j + 1 < len(heads) else len(lines)
        blocks.append((lines[i][2:].strip(" #"), [ln for ln in lines[i + 1:end] if ln.strip()]))

    def block(prefix):
        for head, data in blocks:
            if head.startswith(prefix):
                return data
        raise ValueError(f"{path}: no '## {prefix}' block")

    n_elems = int(_numbers(block("grid: nElems")[0])[0])
    sp = np.array(_numbers(block("grid: sp")[0]))[: n_elems + 1]
    sp[0] = 0.0                                     # GVEC writes the axis of a stretched grid with round-off
    st = {"nfp": int(_numbers(block("global")[0])[0])}
    for name in ("X1", "X2", "LA"):
        n_base, deg, _, n_modes, sin_cos, _ = (int(v) for v in _numbers(block(f"{name}_base")[0]))
        rows = np.array([_numbers(ln) for ln in block(f"{name}:")])
        if rows.shape != (n_modes, 2 + n_base):
            raise ValueError(f"{path}: {name} block is {rows.shape}, expected {(n_modes, 2 + n_base)}")
        mn = rows[:, :2].astype(int)
        # With sin_cos = 3 the sine modes come first (m = 0 with n = 1..n_max, then m = 1..m_max with
        # n = -n_max..n_max), followed by the cosine modes (the same modes and m = n = 0).
        m_max, n_max = mn[:, 0].max(), np.abs(mn[:, 1]).max() // st["nfp"]
        n_sin = {1: n_modes, 2: 0, 3: n_max + m_max * (2 * n_max + 1)}[sin_cos]
        modes = mn[n_sin:] if n_sin < n_modes else mn
        index = {tuple(k): i for i, k in enumerate(modes)}
        sin = np.zeros((len(modes), n_base))
        sin[[index[tuple(k)] for k in mn[:n_sin]]] = rows[:n_sin, 2:]
        cos = rows[n_sin:, 2:] if n_sin < n_modes else np.zeros((len(modes), n_base))
        st[name] = dict(m=modes[:, 0], n=modes[:, 1], cos=cos, sin=sin, deg=deg, T=knot_vector(sp, deg, False))
    T, deg = st["X1"]["T"], st["X1"]["deg"]
    prof = np.array([_numbers(ln) for ln in block("at X1_base IP point positions")]).T   # s, phi, chi, iota, p
    A = BSpline.design_matrix(prof[0], T, deg).toarray()
    st["profiles"] = {name: BSpline(T, np.linalg.solve(A, values), deg)
                      for name, values in zip(("phi", "chi", "iota", "pressure"), prof[1:])}
    st["a_minor"], st["r_major"], _ = _numbers(block("a_minor,r_major,volume")[0])
    return st
