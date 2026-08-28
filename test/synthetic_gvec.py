"""A GVEC state file of an analytic circular torus, for tests.

:func:`write_synthetic_state` writes the ``GVEC_State_*.dat`` layout that
:func:`mrx.gvec.read_state` parses (the inverse of that parser, block for
block: element grid, mode tables, radial B-spline coefficients per mode,
profiles at the radial interpolation points), filled from closed formulas:
the map is the circular torus

    R = R0 + a rho cos(theta_G),   Z = a rho sin(theta_G)

and the Clebsch scalars are

    Phi(rho)  = Phi_edge rho^2                         (rho = sqrt(s))
    iota(rho) = iota0 + iota1 rho^2                    (per full toroidal turn)
    chi(rho)  = Phi_edge (iota0 rho^2 + iota1 rho^4 / 2) = int_0^rho iota Phi'
    LA        = lam_amplitude rho sin(theta_G) (1 + LA_ZETA_MODULATION cos(nfp zeta_G))
    p(rho)    = p0 (1 - rho^2),  p0 = beta B0^2 / (2 mu0),  B0 = Phi_edge / (pi a^2)

in GVEC's units: ``theta_G = 2 pi theta`` and ``zeta_G = 2 pi zeta / nfp``
are the radian angles of the series ``sum f_mn(s) trig(m theta_G - n
zeta_G)`` with ``n`` a multiple of ``nfp``, and ``s = rho`` is GVEC's radial
label. ``lambda`` is stellarator-symmetric (odd under ``(theta, zeta) ->
(-theta, -zeta)``), ``rho^1`` for its ``m = 1`` regularity at the axis, and
carries one ``n = nfp`` modulation so both angular derivatives are
exercised: ``sin(theta_G) cos(nfp zeta_G)`` is the pair of modes
``(m, n) = (1, +-nfp)`` at half the amplitude each.

Every radial function of the map and of lambda is ``1`` or ``rho``, which
any clamped B-spline basis represents exactly (the coefficients of ``rho``
are the Greville abscissae); the profiles are stored as values at the
Greville points, GVEC's interpolation points, through which the degree-5
profile splines reproduce ``rho^2`` and ``rho^4`` exactly. So a reader that
parses the file correctly reproduces the formulas to round-off.

Conventions that match GVEC's: theta runs counter-clockwise in the
``(R, Z)`` plane from the outboard midplane, so with ``det DF > 0`` the map
is ``Y = -R sin(2 pi zeta / nfp)``, the handedness of
``mrx.mappings.toroid_map``, which :func:`mrx.gvec.build_gvec_map` measures
rather than assumes.

The field on concentric circular surfaces is an equilibrium only in the
large-aspect-ratio, low-beta limit (the analytic map has no Shafranov
shift), so keep ``beta`` small and read the force residual of the projected
field as a property of the choice, not of the code. The pressure is stored
for the format (``load_clebsch`` returns it); no solver in MRX consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

#: Relative amplitude of the ``cos(nfp zeta_G)`` modulation of lambda.
LA_ZETA_MODULATION = 0.3

MU0 = 4e-7 * np.pi
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class SyntheticTorus:
    """The closed formulas behind one synthetic state, in normalised
    coordinates ``(rho, theta, zeta)`` in [0, 1] (zeta per field period).

    Every method accepts scalars or arrays and is differentiable with
    ``jax.grad``. Angular derivatives of ``LA`` with respect to GVEC's radian
    angles are ``d/dtheta_G = (1 / 2 pi) d/dtheta`` and
    ``d/dzeta_G = (nfp / 2 pi) d/dzeta``.
    """
    R0: float
    a: float
    nfp: int
    Phi_edge: float
    iota0: float
    iota1: float
    lam_amplitude: float
    beta: float

    def R(self, rho, theta):
        return self.R0 + self.a * rho * jnp.cos(TWO_PI * theta)

    def Z(self, rho, theta):
        return self.a * rho * jnp.sin(TWO_PI * theta)

    def Phi(self, rho):
        return self.Phi_edge * rho ** 2

    def dPhi_dr(self, rho):
        return 2.0 * self.Phi_edge * rho

    def iota(self, rho):
        """Rotational transform per full toroidal turn, ``chi' / Phi'``."""
        return self.iota0 + self.iota1 * rho ** 2

    def chi(self, rho):
        return self.Phi_edge * (self.iota0 * rho ** 2 + 0.5 * self.iota1 * rho ** 4)

    def dchi_dr(self, rho):
        return self.iota(rho) * self.dPhi_dr(rho)

    def LA(self, rho, theta, zeta):
        return (self.lam_amplitude * rho * jnp.sin(TWO_PI * theta)
                * (1.0 + LA_ZETA_MODULATION * jnp.cos(TWO_PI * zeta)))

    @property
    def B0(self):
        """Mean toroidal field ``Phi_edge / (pi a^2)``."""
        return self.Phi_edge / (np.pi * self.a ** 2)

    def pressure(self, rho):
        p0 = self.beta * self.B0 ** 2 / (2.0 * MU0)
        return p0 * (1.0 - rho ** 2)


def greville(sp, deg):
    """Greville abscissae of the clamped degree-``deg`` basis on the element
    grid ``sp``: GVEC's radial interpolation points, and the B-spline
    coefficients of the function ``s`` itself."""
    T = np.concatenate([np.full(deg, sp[0]), sp, np.full(deg, sp[-1])])
    n_base = len(sp) - 1 + deg
    return np.array([T[i + 1:i + deg + 1].mean() for i in range(n_base)])


def _row(values):
    return ", ".join(f"{float(v): .15E}" for v in values)


def write_synthetic_state(path, *, R0, a, nfp, iota, Phi_edge, lam_amplitude,
                          beta, n_elems=10, deg=5):
    """Write the synthetic state to ``path``; returns its :class:`SyntheticTorus`.

    Args:
        path: output file (overwritten).
        R0, a: major and minor radius of the circular torus.
        nfp: field periods; zeta spans one of them.
        iota: ``(iota0, iota1)`` of ``iota(rho) = iota0 + iota1 rho^2`` per
            full toroidal turn; negative on W7-X.
        Phi_edge: toroidal flux at ``rho = 1``; ``Phi = Phi_edge rho^2``.
        lam_amplitude: amplitude of ``LA`` (radians); zero switches lambda off.
        beta: on-axis ``2 mu0 p0 / B0^2`` of the stored pressure profile.
        n_elems, deg: the radial element grid (uniform) and B-spline degree;
            GVEC's defaults for W7-X.
    """
    iota0, iota1 = (float(v) for v in iota)
    torus = SyntheticTorus(float(R0), float(a), int(nfp), float(Phi_edge),
                           iota0, iota1, float(lam_amplitude), float(beta))
    sp = np.linspace(0.0, 1.0, n_elems + 1)
    g = greville(sp, deg)                       # coefficients of s; IP points
    one = np.ones_like(g)
    half = 0.5 * LA_ZETA_MODULATION * lam_amplitude
    blocks = {                                  # name: (sin_cos, [(m, n, coef)])
        "X1": (2, [(0, 0, R0 * one), (1, 0, a * g)]),
        "X2": (1, [(1, 0, a * g)]),
        "LA": (1, [(1, 0, lam_amplitude * g), (1, nfp, half * g), (1, -nfp, half * g)]),
    }
    rule = "#" * 60
    lines = ["## MHD3D Solution... outputLevel and fileID:", "0001,00000000",
             f"## grid: nElems, gridType {rule}", f"{n_elems:8d},{0:8d}",
             "## grid: sp(0:nElems)", _row(sp),
             f"## global: nfp,degGP,mn_nyq(2),hmap {rule}",
             f"{nfp:8d},{deg + 2:8d},{4:8d},{4:8d},{1:8d}"]
    for name, (sin_cos, modes) in blocks.items():
        lines.append(f"## {name}_base: s%nbase,s%deg,s%continuity,f%modes,f%sin_cos,f%excl_mn_zero {rule}")
        lines.append(f"{len(g):8d},{deg:8d},{deg - 1:8d},{len(modes):8d},{sin_cos:8d},{0:8d}")
    for name, (_, modes) in blocks.items():
        lines.append(f"## {name}: m,n,{name}(1:nbase,iMode) {rule}")
        for m, n, coef in modes:
            lines.append(f"{m:8d},{n:8d}, " + _row(coef))
    lines.append(f"## at X1_base IP point positions (size nBase): spos,phi,chi,iota,pressure  {rule}")
    for s in g:
        lines.append(_row([s, torus.Phi(s), torus.chi(s), torus.iota(s), torus.pressure(s)]))
    lines.append(f"## a_minor,r_major,volume  {rule}")
    lines.append(_row([a, R0, 2.0 * np.pi ** 2 * R0 * a ** 2]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return torus
