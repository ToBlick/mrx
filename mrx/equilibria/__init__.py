"""Equilibria computed by other codes (GVEC, VMEC, DESC) as input to MRX.

:func:`read_equilibrium` reads a GVEC ``.dat`` (:mod:`mrx.equilibria.gvec`), a VMEC ``.nc``
(:mod:`mrx.equilibria.vmec`) or a DESC ``.h5`` (:mod:`mrx.equilibria.desc`) file and returns the same
dict for every code, called the *state*. :func:`build_map` turns a state into the spline map ``Phi`` of an
MRX de Rham sequence. :class:`StateField` evaluates one field of the state, for example lambda, as a JAX
function of the logical point.

The state holds the number of field periods ``nfp``, the reader ``kind`` (``"gvec"``, ``"vmec"`` or
``"desc"``), the file ``path``, the radial profiles and three fields, ``X1 = R``, ``X2 = Z`` and
``LA = lambda`` (lambda in radians). Each field is a Fourier series
``sum_mn c_mn(r) cos(m theta - n zeta) + s_mn(r) sin(m theta - n zeta)`` with the angles in radians,
``zeta`` the toroidal angle over the full torus and ``n`` a multiple of ``nfp``. The dict of a field holds
the mode numbers ``m`` and ``n`` and the radial functions ``c_mn``, ``s_mn`` as B-splines: the coefficient
arrays ``cos`` and ``sin`` of shape ``(n_modes, n_base)``, the degree ``deg`` and the clamped knot vector
``T``. The radial label ``r`` is the square root of the normalised toroidal flux. ``profiles`` holds
``phi`` (the toroidal flux divided by ``2 pi``), ``iota`` (per full turn) and ``pressure`` as scipy
``BSpline``\\ s in ``r``. A stellarator-symmetric state (:func:`is_stellarator_symmetric`) has ``R`` as a
cosine series and ``Z``, lambda as sine series.

The logical coordinates ``(r, theta, zeta)`` of a state are right-handed: the map
``(R cos phi, R sin phi, Z)`` has a positive Jacobian. A file whose angles are left-handed, as in a VMEC
wout, is read with the poloidal angle reversed, ``theta -> -theta``, as DESC does. Lambda and iota then
change sign, and ``theta_reversed`` in the state says so.
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

from mrx.differential_forms import det33
from mrx.equilibria.desc import read_desc
from mrx.equilibria.gvec import read_state
from mrx.equilibria.vmec import read_wout
from mrx.mappings import stellarator_symmetric_scalar, stellarator_symmetry_defect
from mrx.precision import DTYPE
from mrx.extraction_operators import conforming_restriction
from mrx.spline_bases import SplineBasis

TWO_PI = 2.0 * np.pi

#: The readers, by file extension.
READERS = {".dat": ("gvec", read_state), ".nc": ("vmec", read_wout), ".h5": ("desc", read_desc)}


def read_equilibrium(path):
    """Read the equilibrium file ``path`` and return its state. The file extension selects the reader
    (:data:`READERS`)."""
    ext = os.path.splitext(path)[1]
    if ext not in READERS:
        raise ValueError(f"{path}: not an equilibrium file, MRX reads GVEC state files (.dat), "
                         "VMEC wout files (.nc) and DESC output files (.h5)")
    kind, read = READERS[ext]
    st = dict(read(path), kind=kind, path=path)
    reverse = _left_handed(st)
    return dict(_reverse_theta(st) if reverse else st, theta_reversed=reverse)


def _left_handed(st):
    """Whether the angles of ``st`` are left-handed, that is whether ``(R cos phi, R sin phi, Z)`` has a
    negative Jacobian. It is checked at mid-radius, where it must have one sign."""
    R, Z = StateField(st["X1"], st["nfp"]), StateField(st["X2"], st["nfp"])

    def jacobian_sign(x):
        # det DPhi = (2 pi / nfp) R (R_theta Z_r - R_r Z_theta)
        dR, dZ = jax.grad(R)(x), jax.grad(Z)(x)
        return R(x) * (dR[1] * dZ[0] - dR[0] * dZ[1])

    t, z = np.meshgrid(np.arange(16) / 16, np.arange(8) / 8, indexing="ij")
    x = jnp.stack([jnp.full(t.size, 0.5), jnp.asarray(t.ravel()), jnp.asarray(z.ravel())], axis=1)
    d = np.asarray(jax.vmap(jacobian_sign)(x))
    if not ((d > 0).all() or (d < 0).all()):
        raise ValueError(f"{st['path']}: the Jacobian of the map changes sign at r = 0.5")
    return bool(d[0] < 0)


def _negated(f):
    return BSpline(f.t, -f.c, f.k)


def _reverse_theta(st):
    """``st`` in the poloidal angle ``-theta``. ``R`` and ``Z`` are the same functions of the new angle, lambda
    is ``-lambda``, since ``theta + lambda`` is the straight-field-line angle, and iota and chi change sign."""
    st = dict(st)
    # cos(m theta - n zeta) = cos(m (-theta) + n zeta) and sin(m theta - n zeta) = -sin(m (-theta) + n zeta)
    for key, cos_sign, sin_sign in (("X1", 1.0, -1.0), ("X2", 1.0, -1.0), ("LA", -1.0, 1.0)):
        block = dict(st[key])
        block["n"], block["cos"], block["sin"] = -block["n"], cos_sign * block["cos"], sin_sign * block["sin"]
        st[key] = block
    st["profiles"] = {name: _negated(f) if name in ("iota", "chi") else f for name, f in st["profiles"].items()}
    return st


def is_stellarator_symmetric(st):
    """Whether the state ``st`` is stellarator-symmetric, that is ``R`` is even and ``Z``, lambda are odd
    under ``(theta, zeta) -> (-theta, -zeta)``. This holds when ``R`` has no sine and ``Z``, lambda have
    no cosine coefficients."""
    return not (st["X1"]["sin"].any() or st["X2"]["cos"].any() or st["LA"]["cos"].any())


class StateField:
    """One field of the state (``block``, for example ``st["LA"]``) as a JAX function of the logical point
    ``(r, theta, zeta)``. Here the angles are in turns (``[0, 1)``) and ``zeta`` covers one of the ``nfp``
    field periods, as in the logical domain of the map."""

    def __init__(self, block, nfp):
        self.basis = SplineBasis(block["cos"].shape[1], block["deg"], "clamped", T=jnp.asarray(block["T"]))
        self.C_cos = jnp.asarray(block["cos"])                           # (n_modes, n_base)
        self.C_sin = jnp.asarray(block["sin"])
        self.m = jnp.asarray(block["m"], dtype=DTYPE)
        self.n_per = jnp.asarray(block["n"], dtype=DTYPE) / nfp          # per field period

    def __call__(self, x):
        # r is not clipped to [0, 1]. A clip halves the autodiff radial derivative at r = 1 (JAX splits
        # the gradient of a tie). The local evaluator continues the end pieces instead.
        vals, idx = self.basis.evaluate_local(x[0])
        arg = 2.0 * jnp.pi * (self.m * x[1] - self.n_per * x[2])
        return jnp.cos(arg) @ (self.C_cos[:, idx] @ vals) + jnp.sin(arg) @ (self.C_sin[:, idx] @ vals)


def _cell_gauss(bp, q):
    """``q`` Gauss-Legendre points and weights per cell of the breakpoints ``bp``, float64."""
    xi, wi = np.polynomial.legendre.leggauss(q)
    lo, hi = bp[:-1], bp[1:]
    pts = (0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * xi[None, :]).ravel()
    return pts, (0.5 * (hi - lo)[:, None] * wi[None, :]).ravel()


def _angular_coefficients(basis, freqs):
    """The L2 projections of ``exp(2 pi i f theta)`` onto the periodic ``basis``, one row per frequency."""
    freqs = np.asarray(freqs, dtype=np.float64)
    T = np.asarray(basis.T, dtype=np.float64)
    bp = np.unique(T[(T >= 0.0) & (T <= 1.0)])
    # enough points for the highest mode's phase across the widest cell
    pts, w = _cell_gauss(bp, basis.p + 7 + int(np.ceil(TWO_PI * np.abs(freqs).max() * np.diff(bp).max())))
    B = np.asarray(basis.collocation_matrix(jnp.asarray(pts)), dtype=np.float64)
    moments = B.T @ (w[:, None] * np.exp(1j * TWO_PI * np.outer(pts, freqs)))   # (N, n_modes)
    return np.linalg.solve(B.T @ (w[:, None] * B), moments).T


def _radial_coefficients(block, C, basis_r):
    """The exact L2 projections of the radial splines ``C`` of a block onto ``basis_r``, one column per mode."""
    # Gauss quadrature on the union of both knot vectors integrates the products exactly
    T_s, deg_s = np.asarray(block["T"]), block["deg"]
    T_r, p_r = np.asarray(basis_r.T, dtype=np.float64), basis_r.p
    pts, w = _cell_gauss(np.unique(np.concatenate([T_s, T_r])), (deg_s + p_r) // 2 + 1)
    Br = BSpline.design_matrix(pts, T_r, p_r).toarray()
    Bs = BSpline.design_matrix(pts, T_s, deg_s).toarray()
    return np.linalg.solve(Br.T @ (w[:, None] * Br), Br.T @ (w[:, None] * (Bs @ C.T)))


class CylindricalMap(eqx.Module):
    """The map ``Phi = (R cos phi, R sin phi, Z)``, ``phi = 2 pi zeta / nfp``, of the scalar splines ``R`` and
    ``Z`` with the tensor-product coefficients ``raw_R``, ``raw_Z`` ``(n_r, n_theta, n_zeta)`` in ``basis``.

    It is a pytree: under ``jax.jit`` the coefficients are traced inputs, so a new map of the same mesh reuses
    every compiled function of the sequence it is installed on."""

    raw_R: jnp.ndarray
    raw_Z: jnp.ndarray
    basis: object = eqx.field(static=True)
    nfp: int = eqx.field(static=True)

    def __call__(self, x):
        ang = TWO_PI / self.nfp * x[2]
        R = self.basis.contract(self.raw_R, x)
        return jnp.array([R * jnp.cos(ang), R * jnp.sin(ang), self.basis.contract(self.raw_Z, x)])


def build_map(st, seq, nfp=None, stellarator_symmetric=False):
    """Return ``(Phi, info)``, the spline map of the state ``st`` in the 0-form space of ``seq``, a
    :class:`CylindricalMap`.

    The map is ``Phi = (R cos phi, R sin phi, Z)`` with ``phi = 2 pi zeta / nfp``, so logical ``zeta``
    covers one field period. ``nfp`` is the file's unless given. ``R`` and ``Z`` are the L2 projections of
    the file's Fourier series onto the 0-form spline space, which is smooth (C1) at the axis. With
    ``stellarator_symmetric`` they are further projected onto ``R`` even and ``Z`` odd under
    ``(theta, zeta) -> (-theta, -zeta)``. This needs uniform angular knots and a stellarator-symmetric
    state.

    ``info`` holds ``nfp``, the range ``det_range`` of ``det DPhi`` at the sample points, the
    ``symmetry_defect`` (:func:`mrx.mappings.stellarator_symmetry_defect`) and the tensor-product spline
    coefficients ``raw_R``, ``raw_Z`` of shape ``(n_r, n_theta, n_zeta)`` that ``Phi`` evaluates. Raises if
    ``det DPhi`` is not positive, which :func:`read_equilibrium` guarantees for a map that does not fold.
    """
    if stellarator_symmetric and not is_stellarator_symmetric(st):
        raise ValueError("the state is not stellarator-symmetric: the projection would drop its asymmetric part")
    nfp = st["nfp"] if nfp is None else int(nfp)
    br, bt, bz = seq.basis_0.Lambda
    free = seq.free                   # the map's coefficients are not zero on the wall
    E = free.E(0)

    def raw(block):
        # the map's nfp, not the file's: with an override logical zeta spans nfp_file / nfp periods
        n_per = block["n"] / nfp
        if np.abs(n_per - np.round(n_per)).max() > 0:
            raise ValueError("toroidal mode numbers are not multiples of nfp")
        # the tensor coefficients: sum over modes of radial x angular coefficients
        modes = (_angular_coefficients(bt, block["m"].astype(np.float64))[:, :, None]
                 * np.conj(_angular_coefficients(bz, n_per))[:, None, :])   # exp(2 pi i (m theta - n zeta))
        C = (np.einsum("ik,kjl->ijl", _radial_coefficients(block, block["cos"], br), modes.real)
             + np.einsum("ik,kjl->ijl", _radial_coefficients(block, block["sin"], br), modes.imag))
        dofs = conforming_restriction(E, jnp.asarray(C.reshape(-1)), free.core_rows(0)).astype(E.dtype)
        return (E.T @ dofs).reshape(seq.basis_0.shape[0])

    raw_R, raw_Z = raw(st["X1"]), raw(st["X2"])
    if stellarator_symmetric:
        raw_R = stellarator_symmetric_scalar(raw_R, seq.basis_0, even=True)
        raw_Z = stellarator_symmetric_scalar(raw_Z, seq.basis_0, even=False)
    # sample points away from the axis, where det DPhi = 0
    rng = np.random.default_rng(0)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.15, 1.0, 64), rng.uniform(0.0, 1.0, 64),
                                       rng.uniform(0.0, 1.0, 64)]))
    Phi = CylindricalMap(raw_R, raw_Z, seq.basis_0.bases[0], nfp)
    d = np.asarray(jax.vmap(lambda x: det33(jax.jacfwd(Phi)(x)))(pts))
    if not (np.isfinite(d).all() and d.min() > 0):
        raise RuntimeError(f"{st['path']}: det DPhi of the spline map in [{d.min():.3e}, {d.max():.3e}] is not "
                           "positive (a map that folds, or a resolution too coarse for the file)")
    return Phi, {"nfp": nfp, "det_range": (float(d.min()), float(d.max())),
                 "symmetry_defect": float(stellarator_symmetry_defect(Phi, pts)), "raw_R": raw_R, "raw_Z": raw_Z}
