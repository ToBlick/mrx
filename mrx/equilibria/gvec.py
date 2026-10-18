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

The state file also names GVEC's map ``hmap`` from ``(X1, X2, zeta)`` to space. MRX reads two of them.
``hmap = 1`` is cylindrical, ``X1 = R`` and ``X2 = Z`` at the toroidal angle ``zeta``. ``hmap = 21`` is the
G-frame of an axis-following frame,

    x = a(zeta) + X1 N(zeta) + X2 B(zeta),

with the curve ``a`` and the orthonormal vectors ``N``, ``B`` sampled in a netCDF file (GVEC's
``hmap_axisNB``). The state file does not hold that file's name, the GVEC parameter file in the same
directory does (``hmap_ncfile``). :func:`read_frame` reads it into the state's ``frame``.
"""
from __future__ import annotations

import glob
import os

import h5py
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
    nfp, _, _, _, hmap = (int(v) for v in _numbers(block("global")[0]))
    if hmap not in (1, 21):
        raise ValueError(f"{path}: GVEC map hmap = {hmap}, MRX reads hmap = 1 (cylindrical) and 21 (G-frame)")
    st = {"nfp": nfp, "hmap": hmap}
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
    if hmap == 21:
        st["frame"] = read_frame(frame_file(path), nfp)
    return st


def frame_file(path):
    """The G-frame netCDF file of the GVEC state ``path``, from the ``hmap_ncfile`` line of the GVEC parameter
    file (``*.ini``) next to it. A relative name is relative to the parameter file."""
    found = set()
    for ini in glob.glob(os.path.join(os.path.dirname(os.path.abspath(path)), "*.ini")):
        with open(ini) as fh:
            for ln in fh:
                key, _, value = ln.split("!")[0].partition("=")
                if key.strip() == "hmap_ncfile":
                    found.add(os.path.normpath(os.path.join(os.path.dirname(ini), value.strip().strip("'\""))))
    if len(found) != 1:
        raise ValueError(f"{path}: hmap = 21 needs one hmap_ncfile in a GVEC parameter file (*.ini) next to the "
                         f"state, found {sorted(found) or 'none'}")
    return found.pop()


def read_frame(path, nfp):
    """Read the G-frame file ``path`` and return the frame as series in the frame that rotates with the field
    periods.

    The file samples the axis ``a`` and the vectors ``N``, ``B`` in Cartesian coordinates at ``nzeta`` angles
    per field period, over the full turn. A field period is the rotation ``R_z(2 pi / nfp)`` about the ``z``
    axis, so ``R_z(-zeta) a(zeta)``, ``R_z(-zeta) N(zeta)`` and ``R_z(-zeta) B(zeta)`` are periodic in one
    field period. Their trigonometric series, as GVEC builds them (modes up to ``(nzeta - 1) / 2`` per
    field period, interpolation for an odd ``nzeta``), are the frame: a dict with the mode numbers ``n``
    (multiples of ``nfp``) and the coefficients ``cos``, ``sin`` of shape ``(3, 3, n_modes)`` (vector
    ``a, N, B``, Cartesian component, mode) of ``cos(-n zeta)`` and ``sin(-n zeta)``, as in a state block.
    """
    with h5py.File(path, "r") as fh:
        if int(fh["NFP"][()]) != nfp:
            raise ValueError(f"{path}: NFP = {int(fh['NFP'][()])}, the state has nfp = {nfp}")
        zeta = fh["axis/zeta(:)"][:]
        full = np.stack([fh[f"axis/{name}(::)"][:] for name in ("xyz", "Nxyz", "Bxyz")])   # (vector, 3, nfp n_z)
    n_z = len(zeta)
    turned = np.einsum("ij,vjk->vik", _rotation(2.0 * np.pi / nfp), full[:, :, :-n_z])
    if np.abs(turned - full[:, :, n_z:]).max() > 1e-10 * np.abs(full).max():
        raise ValueError(f"{path}: the frame is not periodic under the rotation by 2 pi / nfp about z")
    co = np.einsum("ijz,vjz->viz", _rotation(-zeta), full[:, :, :n_z])        # (vector, component, zeta)
    # GVEC's series: modes up to (n_z - 1) / 2 per field period, the discrete L2 projection of the samples
    # (interpolation for an odd n_z, an even n_z loses its top mode)
    k = np.arange((n_z - 1) // 2 + 1)
    arg = -nfp * np.outer(zeta, k)                                            # (zeta, mode)
    coef = np.linalg.lstsq(np.concatenate([np.cos(arg), np.sin(arg[:, 1:])], axis=1),
                           co.reshape(9, n_z).T, rcond=None)[0].T             # (9, 2 len(k) - 1)
    sin = np.concatenate([np.zeros((9, 1)), coef[:, len(k):]], axis=1)
    return {"n": nfp * k, "cos": coef[:, :len(k)].reshape(3, 3, -1), "sin": sin.reshape(3, 3, -1)}


def _rotation(angle):
    """The rotations about the ``z`` axis by ``angle`` (a number or an array), ``(3, 3, ...)``."""
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.array([[c, -s, zero], [s, c, zero], [zero, zero, one]])
