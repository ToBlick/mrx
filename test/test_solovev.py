"""MRX against an exact equilibrium: the Solov'ev tokamak of :mod:`test.solovev`.

The wout holds the closed-form surfaces of the Solov'ev equilibrium (A = 0) in their elliptic angle, its transform
and its pressure, so every quantity MRX computes from it has an exact counterpart. On one graded mesh at p = 2 with 6
cells in zeta (the field does not depend on zeta, and the slab mass model needs five):

- the start field, the 2-form of the file's fluxes, is the Solov'ev field to the discretisation,
- its weak current is the projection of the Solov'ev current onto the Dirichlet 1-forms to the discretisation,
- the flux label is the normalised toroidal flux, and the transform of its weak moments is the Solov'ev transform,
- ten Newton steps from the start field lower the energy at every step, take the force residual to the round-off of
  the precision and leave the field as close to the Solov'ev field as the start was: the discrete equilibrium
  approximates the exact one as well as the projection does,
- with the exact pressure and the start's current as targets the steps lower ``W`` at every step and end at both
  targets, next to the Solov'ev field.

This is the suite's relaxation run. The errors are relative L2 errors over the plasma volume at the quadrature
points, measured 2026-11-21 in float64 unless stated. The start field's error sits next to the axis (3.1e-3 for
r < 0.3 against 5e-4 outside) and falls there at order 1.8 only, from the elliptic angle's lambda next to the axis,
not from the wout's radial resolution (801 surfaces give the same errors as 201). The orders of convergence need a
second mesh and are measured outside the suite (2.09 for B, 1.10 for J, 2.06 for the transform between (14,24,6) and
(18,32,6)).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.flux_label import ANISOTROPY
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.precision import eps
from mrx.relaxation.initial_conditions import initial_field
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import current, weak_pressure
from mrx.relaxation.shells import ERROR_BINS, build_shells, label, pressure_error
from mrx.relaxation.twist import build_slips, current_error, current_on_label, iota_projection
from test.solovev import Solovev

#: The mesh per field period (one field period, the whole torus), p = 2, graded.
MESH = (14, 24, 6)
#: 1.5x the errors measured in float64: the start field 1.09e-3, the weak current against the projection of the
#: Solov'ev current 9.90e-3, the flux label (rms of s) 4.91e-4 with the label spline, the transform of the weak
#: moments (largest error on s in 0.05 .. 0.95 over the largest transform) 1.35e-2 (2026-11-22). The degree-6 fit of
#: the label's fluxes missed by 1.42e-2 here, the field-line tracer is within 4e-4 (2026-11-21): the label's transform
#: carries an error of its own.
BANDS = {"B": 1.6e-3, "J": 1.5e-2, "s": 7.4e-4, "iota": 2.0e-2}
STEPS = 10
#: The cells of the profile space of the targets: the mesh's 12 radial cells resolve 8 (test_twist.py).
CELLS = 8
#: The end of the run with targets: the pressure against the exact pressure and the current against the start's,
#: 7.4e-3 and 3.5e-6 measured in float64 (2026-11-22).
TARGET_BANDS = {"p": 1.1e-2, "I": 7e-6}


@pytest.fixture(scope="module")
def solovev(tmp_path_factory):
    """``(eq, seq, B)``: the equilibrium, the sequence of its wout with the harmonic forms and the start field."""
    eq = Solovev()
    path = eq.write_wout(str(tmp_path_factory.mktemp("solovev") / "wout_solovev.nc"), ns=201, mpol=40)
    seq = build_sequence(path, MESH, 2, symmetry="stellarator", graded=True)
    compute_nullspaces(seq, verbose=False)
    return eq, seq, initial_field(seq)[0]


def _points(seq):
    """The Cartesian quadrature points and their volume weights."""
    return np.asarray(jax.vmap(seq.map)(seq.quad.x)), np.asarray(seq.quad.w * seq.jacobian_j)


def _field_error(eq, seq, B):
    """The relative L2 error of the 2-form ``B`` against the Solov'ev field."""
    X, w = _points(seq)
    Bh = np.asarray(jax.vmap(Pushforward(DiscreteFunction(B, seq.odd.basis_2, seq.odd.E(2)), seq.map, 2))(seq.quad.x))
    Be = eq.field(X)
    return float(np.sqrt(np.sum(w[:, None] * (Bh - Be) ** 2) / np.sum(w[:, None] * Be ** 2)))


def _check(name, error):
    print(f"\n  {name}: {error:.3e} on {MESH} (band {BANDS[name]:.1e})")
    assert error < BANDS[name]


def test_the_start_field_is_the_solovev_field(solovev):
    """``||B_h - B|| / ||B||`` within its band."""
    eq, seq, B = solovev
    _check("B", _field_error(eq, seq, B))


def test_the_current_is_solovevs(solovev):
    """The weak current ``M_1 J = D_1^T B`` against the L2 projection of the Solov'ev ``mu0 J`` onto the
    Dirichlet 1-forms, in the mass norm. The weak current has no tangential part on the wall, where the Solov'ev
    current has one, so it is compared with the projection onto its own space, not with the current itself."""
    eq, seq, B = solovev
    odd = seq.odd

    def J_exact(x):
        X = seq.map(x)
        R, Z = jnp.hypot(X[0], X[1]), X[2]
        cos, sin = X[0] / R, X[1] / R
        psi_R = 2.0 * eq.a * R * Z ** 2 + 4.0 * eq.b * R * (R ** 2 - 1.0)
        psi_Z = 2.0 * eq.a * R ** 2 * Z + 2.0 * eq.d * Z
        psi = eq.a * R ** 2 * Z ** 2 + eq.b * (R ** 2 - 1.0) ** 2 + eq.d * Z ** 2
        dF = -2.0 * eq.d / jnp.sqrt(eq.F0 ** 2 - 4.0 * eq.d * psi)
        J_R, J_phi, J_Z = -dF * psi_Z / R, -(eq.c * R + 2.0 * eq.d / R), dF * psi_R / R
        return jnp.array([J_R * cos - J_phi * sin, J_R * sin + J_phi * cos, J_Z])
    J_proj = odd.M[1].solve(odd.load(J_exact, 1))
    _check("J", float(odd.l2_norm(current(B, seq) - J_proj, 1) / odd.l2_norm(J_proj, 1)))


def test_the_label_is_the_toroidal_flux_and_its_transform_is_solovevs(solovev):
    """The flux label of the start field (anisotropy :data:`~mrx.flux_label.ANISOTROPY`, solved to 1e-9) against the
    normalised toroidal flux of the Solov'ev surface through each quadrature point, in the root mean square, and the
    transform of the label's weak moments (:func:`mrx.relaxation.twist.iota_projection`) against the Solov'ev
    transform on ``s`` in 0.05 .. 0.95. MRX reads the wout's angles reversed, so its transform is the negative of the
    file's. The field is tangent to the logical surfaces, so the temperature is a function of ``r`` alone and the
    anisotropy costs nothing."""
    eq, seq, B = solovev
    S = np.linspace(0.05, 0.95, 19)
    X, w = _points(seq)
    s_q, T, info = label(seq, B, ANISOTROPY, None, 1e-9)
    assert int(info) > 0
    _check("s", float(np.sqrt(np.sum(w * (np.asarray(s_q) - eq.s_at(X)) ** 2) / np.sum(w))))
    iota = -iota_projection(seq, B, s_q)[0](S)
    print("\n  iota error on s = " + " ".join(f"{x:.2f}: {e:+.1e}" for x, e in zip(S, iota - eq.iota(S))))
    _check("iota", float(np.abs(iota - eq.iota(S)).max() / eq.iota(S).max()))


def _descends(res, floor):
    """Every step taken above the residual ``floor`` moves along its descent direction (a step with ``dt* <= 0`` is
    clipped to no step), and ``W`` (the energy less the targets' work) never rises by more than its round-off."""
    resid = np.asarray(res.trace["resid"], dtype=float)
    dt_star = np.asarray(res.trace["dt_star"], dtype=float)
    dW = np.asarray(res.trace["dW" if "dW" in res.trace else "dE"], dtype=float)
    print(f"\n  resid {' '.join(f'{r:.1e}' for r in resid)}\n  dt* {' '.join(f'{d:.2f}' for d in dt_star)}"
          f"\n  max dW / E0 {dW.max() / res.E0:+.1e}")
    assert np.all(dW < eps() * res.E0)
    assert np.all(dt_star[resid > floor] > 0.0)
    return resid


def test_newton_steps_reach_the_discrete_equilibrium_next_to_solovevs(solovev):
    """Ten Newton steps from the start field descend (:func:`_descends`) above the round-off floor ``1e3 eps^2`` of the
    squared force residual, the residual falls from 3.1e-7 to 2.4e-16 in float64 and to 3.1e-13 in float32 (bound: by
    1e6 or to that floor), and the end field is as close
    to the Solov'ev field as the start (1.06e-3 against 1.09e-3 in both precisions, bound 1.05 times the start's
    error)."""
    eq, seq, B = solovev
    ts = TimeStepper(seq=seq, newton=True)
    res = relax(initial_state(B, ts), ts, steps=STEPS, chunk=STEPS, verbose=False)
    resid = _descends(res, 1e3 * eps() ** 2)
    start, end = _field_error(eq, seq, B), _field_error(eq, seq, res.state.B_n)
    print(f"  error against Solov'ev {start:.3e} -> {end:.3e}")
    assert resid[-1] < max(1e-6 * resid[0], 1e3 * eps() ** 2)
    assert end < 1.05 * start


def test_newton_steps_with_the_exact_pressure_and_current_as_targets(solovev):
    """Ten Newton steps in two chunks with the shells of the exact pressure ``mu0 p = c (psi_b - psi(s))`` and the
    slips of the start's current on :data:`CELLS` cells as targets, both rebuilt on the field's label before each
    chunk (as ``scripts/relax.py`` does): they descend (:func:`_descends`) until the residual is 1e-5 of the start's,
    then it stays at the floor of the targets (2.4e-13 from 3.1e-7 in float64, 2026-11-22), they end at the pressure
    and the current within :data:`TARGET_BANDS` and leave the field as close to the Solov'ev field as the start (bound 1.05 times its
    error)."""
    eq, seq, B = solovev
    ts = TimeStepper(seq=seq, newton=True)

    def p_star(s):
        return jnp.asarray(eq.c * (eq.psi_b - eq.psi_of_s(np.asarray(s))), dtype=B.dtype)
    _, T0, _ = label(seq, B, ANISOTROPY)
    mu_star = current_on_label(seq, B, T0, CELLS)

    def rebuild(state, it):
        s_q, T, _ = label(seq, state.B_n, ANISOTROPY)
        shells, _ = build_shells(seq, state.B_n, s_q, p_star(s_q), ts.newton_penalty, ts.newton_mass_solve_tol, CELLS)
        return state, ts.with_shells(shells).with_slips(build_slips(seq, state.B_n, T, mu_star)[0])
    res = relax(initial_state(B, ts), ts, steps=STEPS, chunk=STEPS // 2, rebuild=rebuild, verbose=False)
    resid = _descends(res, 1e-5 * res.trace["resid"][0])
    assert resid[-1] < 1e-5 * resid[0]
    B_end = res.state.B_n
    s_q, T, _ = label(seq, B_end, ANISOTROPY)
    p_err, _ = pressure_error(seq, weak_pressure(current(B_end, seq), B_end, seq), s_q, p_star(s_q), ERROR_BINS)
    I_err = current_error(current_on_label(seq, B_end, T, CELLS), mu_star)
    start, end = _field_error(eq, seq, B), _field_error(eq, seq, B_end)
    print(f"  pressure error {p_err:.3e}, current error {I_err:.3e}, error against Solov'ev {start:.3e} -> {end:.3e}")
    assert p_err < TARGET_BANDS["p"] and I_err < TARGET_BANDS["I"]
    assert end < 1.05 * start
