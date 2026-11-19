"""The harmonic flow as an unknown of the Newton step.

The incompressible flows tangent to the wall are ``curl a + c h``, with ``h`` the harmonic 2-form of the domain. The
curls are ``M``-orthogonal to ``h``, so a step ``curl a`` cannot move the field along ``h``, and the derivative of the
energy along it, ``h . load(J x B)``, is the harmonic part of the force. With stellarator symmetry ``h`` is odd and the
force even, so that part vanishes, and the even flows of a half-period sequence hold no ``h``. Without the symmetry
the Newton step has the amplitude ``c`` as one more unknown (:func:`mrx.relaxation.newton.target_direction`).

The asymmetric fixture is the li383 wout with ``Z += 0.03 s cos(2 theta)`` on every surface (the ``zmnc`` of
``(m, n) = (2, 0)``, a third of the boundary's ``(2, 0)`` ``RBC``), on the fixture's mesh in one field period without
the reflection, with the file's field. The bordered Gauss-Newton system solved tightly zeroes its harmonic row, the
curls alone leave it. On the fixture's li383 in one field period without the reflection the harmonic part of the
force vanishes to the solves' tolerance and the step takes no harmonic flow.
"""
import jax.numpy as jnp
import numpy as np
from scipy.io import netcdf_file

from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.precision import eps
from mrx.relaxation.initial_conditions import potential_two_form
from mrx.relaxation.loop import TimeStepper
from mrx.relaxation.newton import (NEWTON_MASS_SOLVE_TOL, NEWTON_PENALTY, atom_profiles, newton_direction,
                                   parallel_penalty_profile, second_variation, target_direction)
from test.conftest import NS, P

#: The up-down asymmetric perturbation of the fixture, in metres.
ASYMMETRY = 0.03
#: The tolerance of the tight Newton solves.
TIGHT = 1e-4
#: The harmonic part of the force at the start of the asymmetric fixture, relative to the force: 5.6e-3 in float64
#: and float32 (2026-11-19).
HARMONIC_SHARE = 1e-3


def _asymmetric_wout(src, dst):
    """The wout ``src`` with ``Z += ASYMMETRY s cos(2 theta)`` on every surface, written to ``dst`` with what MRX reads."""
    v = netcdf_file(src, "r", mmap=False).variables
    names = ("ns", "nfp", "mnmax", "version_", "xm", "xn", "rmnc", "zmns", "lmns", "phi", "iotaf", "presf")
    v = {k: np.array(v[k][()]) for k in names}
    ns = int(v["ns"])
    zmnc = np.zeros_like(v["rmnc"])
    zmnc[:, np.nonzero((v["xm"] == 2) & (v["xn"] == 0))[0][0]] = ASYMMETRY * np.arange(ns) / (ns - 1)
    with netcdf_file(dst, "w") as f:
        f.createDimension("radius", ns)
        f.createDimension("mn_mode", int(v["mnmax"]))
        for k in ("ns", "nfp", "mnmax"):
            f.createVariable(k, "i4", ())[()] = int(v[k])
        f.createVariable("lasym__logical__", "i4", ())[()] = 1
        f.createVariable("version_", "f8", ())[()] = float(v["version_"])
        for k in ("xm", "xn"):
            f.createVariable(k, "f8", ("mn_mode",))[:] = v[k]
        for k, a in (("rmnc", v["rmnc"]), ("zmns", v["zmns"]), ("lmns", v["lmns"]), ("rmns", 0 * zmnc),
                     ("zmnc", zmnc), ("lmnc", 0 * zmnc)):
            f.createVariable(k, "f8", ("radius", "mn_mode"))[:] = a
        for k in ("phi", "iotaf", "presf"):
            f.createVariable(k, "f8", ("radius",))[:] = v[k]


def _harmonic_part(ts, y):
    """``|h . y| / |h|_M`` of the dual 2-form ``y``: the norm of its harmonic part."""
    return float(jnp.linalg.norm(ts.harmonic_flows @ y / jnp.sqrt(ts.harmonic_norm_sq)))


def _bordered_step(seq, B, ts, load, tol):
    """``(u, |c h|_M / |u|_M)``: the bordered Gauss-Newton direction at ``B`` solved to ``tol`` and the share of its
    harmonic flow."""
    u, _, _, c_h, _, _, _ = target_direction(seq, B, load, jnp.zeros(seq.even.n(1)), ts.harmonic_flows, tol=tol)
    return u, float(jnp.linalg.norm(c_h * jnp.sqrt(ts.harmonic_norm_sq)) / seq.even.l2_norm(u, 2))


def test_bordered_step_removes_the_harmonic_force(seq, tmp_path):
    """On the asymmetric fixture the force has a harmonic part (2.82e-3 of the force's 0.500, the bound is
    HARMONIC_SHARE of it). The harmonic row of the Gauss-Newton model, ``h . (load - H u)``, after the tight curl step
    is 2.69e-3, 95 % of the harmonic part, after the tight bordered step 3.0e-6 (3.1e-6 in float32), 11 times the
    solve's tolerance TIGHT relative to the harmonic part. The bordered step takes 3.8e-2 of its norm along ``h``.
    147 s on four CPU cores in float64, 356 s in float32 (2026-11-19)."""
    path = str(tmp_path / "wout_asym.nc")
    _asymmetric_wout(seq.equilibrium["path"], path)
    seq2 = build_sequence(path, NS, P, symmetry="field-period", graded=True)
    compute_nullspaces(seq2, verbose=False)
    ts = TimeStepper(seq=seq2, newton=True)
    B = potential_two_form(seq2)[0]
    _, load = ts._lorentz(B, None)
    F, MF, _ = ts._force(load, jnp.zeros(seq2.even.n(1)))
    H = second_variation(seq2, B, None, parallel_penalty_profile(atom_profiles(seq2, B), NEWTON_PENALTY))
    u_curl = newton_direction(seq2, B, load, jnp.zeros(seq2.even.n(1)), tol=TIGHT)[0]
    u, share = _bordered_step(seq2, B, ts, load, TIGHT)
    F_h, after_curl, after = (_harmonic_part(ts, y) for y in (load, load - H(u_curl), load - H(u)))
    F_norm = float(jnp.sqrt(F @ MF))
    print(f"\n  F_h {F_h:.4e} of |F| {F_norm:.4e}, harmonic row after the curl step {after_curl:.4e}, after the bordered "
          f"step {after:.4e}, |c h| / |u| {share:.3e}")
    assert F_h > HARMONIC_SHARE * F_norm
    assert after_curl > 0.5 * F_h
    assert after < 30 * TIGHT * F_h


def test_no_harmonic_force_with_stellarator_symmetry(domains):
    """The half-period flows hold no harmonic form, so the step has no harmonic unknown. On li383 in one field
    period without the reflection the harmonic part of the force is 1.9e-12 of the force's 0.139 in float64 (0 in
    float32), below ten solve tolerances, and the production step takes 7.4e-8 of its norm along ``h`` (5.7e-6 in
    float32), the parity defect of the 1-form mass solves inside the Hessian, below ten of their tolerance. 54 s and
    94 s on four CPU cores after the fixture (2026-11-19)."""
    assert TimeStepper(seq=domains("half"), newton=True).harmonic_flows is None
    seq = domains("period")
    ts = TimeStepper(seq=seq, newton=True)
    B = potential_two_form(seq)[0]
    _, load = ts._lorentz(B, None)
    F, MF, _ = ts._force(load, jnp.zeros(seq.even.n(1)))
    F_norm, F_h = float(jnp.sqrt(F @ MF)), _harmonic_part(ts, load)
    _, share = _bordered_step(seq, B, ts, load, ts.newton_tol)
    print(f"\n  F_h {F_h:.3e} of |F| {F_norm:.4e}, |c h| / |u| {share:.3e}")
    assert F_h < 10 * max(seq.tol, eps()) * F_norm
    assert share < 10 * NEWTON_MASS_SOLVE_TOL
