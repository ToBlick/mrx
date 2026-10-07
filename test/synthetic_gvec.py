"""A GVEC state file of an analytic circular torus, for tests.

:func:`write_synthetic_state` writes the ``GVEC_State_*.dat`` layout that
:func:`mrx.equilibria.gvec.read_state` parses, filled from closed formulas
(``r`` GVEC's radial label, the square root of the normalised flux):

    R = R0 + a r cos(theta_G),   Z = a r sin(theta_G)
    Psi(r)  = Psi_edge r^2
    iota(r) = iota0 + iota1 r^2                    (per full toroidal turn)
    chi(r)  = Psi_edge (iota0 r^2 + iota1 r^4 / 2) = int_0^r iota Psi'
    LA      = lam_amplitude r sin(theta_G) (1 + LA_ZETA_MODULATION cos(nfp zeta_G))
    p(r)    = p0 (1 - r^2),  p0 = beta B0^2 / (2 mu0),  B0 = Psi_edge / (pi a^2)

with the radian angles ``theta_G = 2 pi theta``, ``zeta_G = 2 pi zeta / nfp``.
Every radial function of the map and of lambda is ``1`` or ``r`` and the
profiles are stored at the Greville points, so the degree-5 splines represent
all of it exactly and a correct parser reproduces the formulas to round-off.
The poloidal angle runs counter-clockwise, so the angles are left-handed as in a
VMEC file and :func:`mrx.equilibria.read_equilibrium` reverses theta. :func:`evaluate` evaluates a block
independently of the parser. ``shift = (c, d)`` writes the same fields at
``(theta_G + c, zeta_G + d)``, without stellarator symmetry.

``frame=True`` writes the same torus in GVEC's G-frame (``hmap = 21``), with the frame file ``frame.nc``
and a parameter file ``parameter.ini`` naming it next to the state. The axis is the circle ``R = R0``,
and the frame turns once per field period about it,

    N = cos(alpha) e_R - sin(alpha) e_Z,   B = -sin(alpha) e_R - cos(alpha) e_Z,   alpha = -nfp zeta_G,

so ``(T, N, B)`` is right-handed, ``X1 = a r cos(theta_G - nfp zeta_G)`` and
``X2 = -a r sin(theta_G - nfp zeta_G)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

#: Relative amplitude of the ``cos(nfp zeta_G)`` modulation of lambda.
LA_ZETA_MODULATION = 0.3

MU0 = 4e-7 * np.pi
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class SyntheticTorus:
    """The closed formulas behind one synthetic state, in the logical
    ``(r, theta, zeta)`` in [0, 1] (zeta per field period), for scalars or arrays."""
    R0: float
    a: float
    nfp: int
    Psi_edge: float
    iota0: float
    iota1: float
    lam_amplitude: float
    beta: float

    def R(self, r, theta):
        return self.R0 + self.a * r * jnp.cos(TWO_PI * theta)

    def Z(self, r, theta):
        return self.a * r * jnp.sin(TWO_PI * theta)

    def Psi(self, r):
        return self.Psi_edge * r ** 2

    def dPsi_dr(self, r):
        return 2.0 * self.Psi_edge * r

    def iota(self, r):
        """Rotational transform per full toroidal turn, ``chi' / Psi'``."""
        return self.iota0 + self.iota1 * r ** 2

    def chi(self, r):
        return self.Psi_edge * (self.iota0 * r ** 2 + 0.5 * self.iota1 * r ** 4)

    def LA(self, r, theta, zeta):
        return (self.lam_amplitude * r * jnp.sin(TWO_PI * theta)
                * (1.0 + LA_ZETA_MODULATION * jnp.cos(TWO_PI * zeta)))

    @property
    def B0(self):
        """Mean toroidal field ``Psi_edge / (pi a^2)``."""
        return self.Psi_edge / (np.pi * self.a ** 2)

    def pressure(self, r):
        p0 = self.beta * self.B0 ** 2 / (2.0 * MU0)
        return p0 * (1.0 - r ** 2)


def greville(sp, deg):
    """Greville abscissae of the clamped degree-``deg`` basis on the element
    grid ``sp``: GVEC's radial interpolation points, and the B-spline
    coefficients of the function ``r`` itself."""
    T = np.concatenate([np.full(deg, sp[0]), sp, np.full(deg, sp[-1])])
    n_base = len(sp) - 1 + deg
    return np.array([T[i + 1:i + deg + 1].mean() for i in range(n_base)])


def evaluate(block, r, theta, zeta):
    """Evaluate a state block on the tensor grid ``r x theta x zeta``, independently of the parser. The
    angles are in radians, ``zeta`` is the full-turn toroidal angle."""
    D = BSpline.design_matrix(np.asarray(r, dtype=np.float64), block["T"], block["deg"]).toarray()
    arg = (np.outer(block["m"], theta)[:, :, None]
           - np.outer(block["n"], zeta)[:, None, :])                      # (n_modes, n_t, n_z)
    return (np.einsum("sk,ktz->stz", D @ block["cos"].T, np.cos(arg))
            + np.einsum("sk,ktz->stz", D @ block["sin"].T, np.sin(arg)))


def _row(values):
    return ", ".join(f"{float(v): .15E}" for v in values)


def _sincos(modes, sin_cos, nfp, c, d):
    """Rewrite the ``modes`` of one parity at the shifted angles as sine and cosine rows in GVEC's mode order."""
    coef = {(m, n): cf for m, n, cf in modes}
    m_max, n_max = max(m for m, _ in coef), max(abs(n) for _, n in coef) // nfp
    order = [(0, k * nfp) for k in range(1, n_max + 1)]
    order += [(m, k * nfp) for m in range(1, m_max + 1) for k in range(-n_max, n_max + 1)]
    zero = np.zeros_like(next(iter(coef.values())))
    rows = {1: [], 2: []}
    for m, n in [(0, 0)] + order:
        cf, phase = coef.get((m, n), zero), m * c - n * d
        # cos(A + phase) = cos A cos phase - sin A sin phase, sin(A + phase) = sin A cos phase + cos A sin phase
        same, other = cf * np.cos(phase), cf * np.sin(phase) * (-1.0 if sin_cos == 2 else 1.0)
        for parity, value in ((sin_cos, same), (3 - sin_cos, other)):
            if (m, n) != (0, 0) or parity == 2:
                rows[parity].append((m, n, value))
    return rows[1] + rows[2]


def write_synthetic_state(path, *, R0, a, nfp, iota, Psi_edge, lam_amplitude,
                          beta, n_elems=10, deg=5, shift=None, frame=False):
    """Write the synthetic state to ``path`` and return its :class:`SyntheticTorus`.

    Args:
        path: output file (overwritten).
        R0, a: major and minor radius of the circular torus.
        nfp: number of field periods. The logical zeta spans one of them.
        iota: ``(iota0, iota1)`` of ``iota(r) = iota0 + iota1 r^2`` per
            full toroidal turn (negative for W7-X).
        Psi_edge: toroidal flux at ``r = 1``, so that ``Psi = Psi_edge r^2``.
        lam_amplitude: amplitude of ``LA`` in radians. Zero switches lambda off.
        beta: on-axis ``2 mu0 p0 / B0^2`` of the stored pressure profile.
        n_elems, deg: number of uniform radial elements and the B-spline degree
            (GVEC's defaults for W7-X).
        shift: ``(c, d)`` writes the fields at ``(theta_G + c, zeta_G + d)``. The state is then not
            stellarator-symmetric and every block carries both the sine and the cosine series.
        frame: write the torus in the G-frame, with ``frame.nc`` and ``parameter.ini`` next to ``path``.
    """
    iota0, iota1 = (float(v) for v in iota)
    torus = SyntheticTorus(float(R0), float(a), int(nfp), float(Psi_edge),
                           iota0, iota1, float(lam_amplitude), float(beta))
    sp = np.linspace(0.0, 1.0, n_elems + 1)
    g = greville(sp, deg)                       # the coefficients of r, also GVEC's interpolation points
    one = np.ones_like(g)
    half = 0.5 * LA_ZETA_MODULATION * lam_amplitude
    blocks = {                                  # name: (sin_cos, [(m, n, coef)])
        "X1": (2, [(0, 0, R0 * one), (1, 0, a * g)]),
        "X2": (1, [(1, 0, a * g)]),
        "LA": (1, [(1, 0, lam_amplitude * g), (1, nfp, half * g), (1, -nfp, half * g)]),
    }
    if frame:
        blocks["X1"], blocks["X2"] = (2, [(1, nfp, a * g)]), (1, [(1, nfp, -a * g)])
        _write_frame(os.path.dirname(os.path.abspath(path)), R0, nfp)
    if shift is not None:
        blocks = {name: (3, _sincos(modes, sin_cos, nfp, *shift)) for name, (sin_cos, modes) in blocks.items()}
    rule = "#" * 60
    lines = ["## MHD3D Solution... outputLevel and fileID:", "0001,00000000",
             f"## grid: nElems, gridType {rule}", f"{n_elems:8d},{0:8d}",
             "## grid: sp(0:nElems)", _row(sp),
             f"## global: nfp,degGP,mn_nyq(2),hmap {rule}",
             f"{nfp:8d},{deg + 2:8d},{4:8d},{4:8d},{21 if frame else 1:8d}"]
    for name, (sin_cos, modes) in blocks.items():
        lines.append(f"## {name}_base: s%nbase,s%deg,s%continuity,f%modes,f%sin_cos,f%excl_mn_zero {rule}")
        lines.append(f"{len(g):8d},{deg:8d},{deg - 1:8d},{len(modes):8d},{sin_cos:8d},{0:8d}")
    for name, (_, modes) in blocks.items():
        lines.append(f"## {name}: m,n,{name}(1:nbase,iMode) {rule}")
        for m, n, coef in modes:
            lines.append(f"{m:8d},{n:8d}, " + _row(coef))
    lines.append(f"## at X1_base IP point positions (size nBase): spos,phi,chi,iota,pressure  {rule}")
    for s in g:
        lines.append(_row([s, torus.Psi(s), torus.chi(s), torus.iota(s), torus.pressure(s)]))
    lines.append(f"## a_minor,r_major,volume  {rule}")
    lines.append(_row([a, R0, 2.0 * np.pi ** 2 * R0 * a ** 2]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return torus


def _write_frame(directory, R0, nfp, n_zeta=7):
    """Write the G-frame file ``frame.nc`` of the synthetic torus and a ``parameter.ini`` naming it into
    ``directory``, with ``n_zeta`` samples per field period over the full turn."""
    zeta = (np.arange(nfp * n_zeta) + 0.5) / n_zeta * TWO_PI / nfp
    alpha = -nfp * zeta
    e_R = np.array([np.cos(zeta), np.sin(zeta), np.zeros_like(zeta)])
    e_Z = np.array([np.zeros_like(zeta), np.zeros_like(zeta), np.ones_like(zeta)])
    vectors = {"xyz": R0 * e_R, "Nxyz": np.cos(alpha) * e_R - np.sin(alpha) * e_Z,
               "Bxyz": -np.sin(alpha) * e_R - np.cos(alpha) * e_Z}
    with h5py.File(os.path.join(directory, "frame.nc"), "w") as fh:
        fh["NFP"] = nfp
        fh["axis/zeta(:)"] = zeta[:n_zeta]
        for name, v in vectors.items():
            fh[f"axis/{name}(::)"] = v
    with open(os.path.join(directory, "parameter.ini"), "w") as fh:
        fh.write("which_hmap = 21\nhmap_ncfile = frame.nc\n")
