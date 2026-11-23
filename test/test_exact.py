"""MRX against the exact Solov'ev tokamak on the shared sequence.

- the projections: the field ``b0`` of the wout against the exact ``B``, its weak current against the projection of
  the exact ``mu0 J`` onto the Dirichlet 1-forms, its weak pressure against the exact ``p``,
- the observation error of the transform: the weak residual ``r_j = int (iota - iota*) M_j ds`` of ``b0`` against
  the exact transform,
- Newton steps from a perturbed exact state return to the discrete equilibrium next to the exact one,
- with the exact pressure as the target of the shells and the exact transform as the target of the iota loop, the
  relaxation from ``b0`` stays next to the exact equilibrium and meets both targets.

The errors are relative to the exact norms over the plasma, at the quadrature points. The relaxation runs are in
float64 only: a float32 Newton step costs three times a float64 one on the CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.flux_label import ANISOTROPY
from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.physics import current, weak_pressure
from mrx.relaxation.shells import ERROR_BINS, build_shells, label, pressure_error
from mrx.relaxation.twist import current_on_label, iota_fit, iota_newton, iota_residual
from test.conftest import field_error

#: The cells of the profile space of the targets on the shared sequence's 10 radial cells.
CELLS = 6
#: The projections: the field, the weak current against the projected exact current, the weak pressure against the
#: exact pressure, and the largest local mean of the transform's miss over the largest transform: 2.3e-3, 1.95e-2,
#: 9.7e-3 and 2.4e-4 measured (tokamak (12,16,6), float64, 2026-11-22).
BANDS = dict(B=3e-3, J=2.5e-2, p=1.3e-2, iota=3.5e-4, iota_fit=1e-3)
#: The end of the run with the exact targets: the pressure and the transform's weak residual, 9.2e-3 and 1.1e-4
#: measured after 20 steps (tokamak (12,16,6), float64, 2026-11-22).
TARGET_BANDS = dict(p=1.3e-2, iota=1.6e-4)
#: The size of the smooth perturbation of the Newton run, relative to the field.
PERTURBATION = 1e-3
#: Newton steps of the runs: the perturbed field is below the bar after 3 (9.3e-9), the exact targets after 3 with one
#: more rebuild of the loop.
STEPS = 5
#: The squared force residual an equilibrium is below (the production bar of the mesh moves).
BAR = 1e-8


def _float64_only():
    if DTYPE != np.float64:
        pytest.skip("the relaxation runs in float64 only: a float32 Newton step costs three times as much on the CPU")


def _points(seq):
    return np.asarray(jax.vmap(seq.map)(seq.quad.x)), np.asarray(seq.quad.w * seq.jacobian_j)


def _mu0_p(eq, X):
    """The exact ``mu0 p`` at the Cartesian points ``X``."""
    return eq.mu0_p(np.hypot(X[:, 0], X[:, 1]), X[:, 2])


def _p_star(eq):
    """The exact ``mu0 p`` as a function of the label ``s``."""
    return lambda s: jnp.asarray(eq.mu0_p_of_s(np.asarray(s)))


def test_projections(eq, seq, b0):
    """The field ``b0`` against the exact field, its weak current ``M_1 J = D_1^T B`` against the L2 projection of
    the exact ``mu0 J`` onto the Dirichlet 1-forms (the weak current has no tangential part on the wall), and its weak
    pressure, gauged by its mean (:func:`mrx.relaxation.shells.pressure_error`), against the exact ``mu0 p`` at the
    quadrature points, within :data:`BANDS`."""
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
    J_h = current(b0, seq)
    e_B = field_error(eq, seq, b0)
    e_J = float(odd.l2_norm(J_h - J_proj, 1) / odd.l2_norm(J_proj, 1))
    X, _ = _points(seq)
    s_q, _, _ = label(seq, b0, ANISOTROPY)
    e_p, _ = pressure_error(seq, weak_pressure(J_h, b0, seq), s_q, jnp.asarray(_mu0_p(eq, X)), ERROR_BINS)
    print(f"\n  B {e_B:.3e}, J {e_J:.3e}, p {e_p:.3e}")
    assert e_B < BANDS["B"] and e_J < BANDS["J"] and e_p < BANDS["p"]


def test_observation_error_of_the_transform(eq, seq, b0):
    """The weak residual of ``b0`` against the exact transform on its label (:func:`mrx.relaxation.twist
    .iota_residual`, :data:`CELLS` cells): every local mean of the miss is within :data:`BANDS` of the largest
    transform, the error with which the iota loop sees the exact transform on this mesh, and the degree-6 polynomial
    fitted to the weak moments (:func:`mrx.relaxation.twist.iota_fit`) is the exact transform on ``s`` in 0.05 .. 0.95
    within ``BANDS["iota_fit"]``."""
    s_q, _, _ = label(seq, b0, ANISOTROPY)
    r = np.asarray(iota_residual(seq, b0, s_q, eq.iota, CELLS + 2)(b0)[0])
    miss = float(np.abs(r).max() / eq.iota(0.0))
    S = np.linspace(0.05, 0.95, 19)
    fit = float(np.abs(np.abs(iota_fit(seq, b0, s_q, 6)(S)) - eq.iota(S)).max() / eq.iota(0.0))
    print(f"\n  weak residual {' '.join(f'{v:+.1e}' for v in r)}, largest {miss:.3e} of iota, the degree-6 fit of the "
          f"weak moments {fit:.3e}")
    assert miss < BANDS["iota"] and fit < BANDS["iota_fit"]


def _descends(res, floor):
    """Every step taken above the squared residual ``floor`` moves along its descent direction (a step with
    ``dt* <= 0`` is clipped to no step), and ``W`` (the energy less the targets' work) never rises by more than its
    round-off. Returns the residuals."""
    resid = np.asarray(res.trace["resid"], dtype=float)
    dt_star = np.asarray(res.trace["dt_star"], dtype=float)
    dW = np.asarray(res.trace["dW" if "dW" in res.trace else "dE"], dtype=float)
    print(f"\n  resid {' '.join(f'{r:.1e}' for r in resid)}\n  dt* {' '.join(f'{d:.2f}' for d in dt_star)}"
          f"\n  max dW / E0 {dW.max() / res.E0:+.1e}")
    assert np.all(dW < eps() * res.E0)
    assert np.all(dt_star[resid > floor] > 0.0)
    return resid


def test_newton_returns_to_the_discrete_equilibrium(eq, seq, b0):
    """Newton steps from ``b0`` plus a smooth curl of :data:`PERTURBATION` of its norm descend, take the squared
    force residual below :data:`BAR` and end as close to the exact field as the projection: within 1.1 times the
    error of ``b0``. Measured (tokamak (12,16,6), float64, 2026-11-22): the residual 1.1e-5 -> 6.9e-10 after 5 steps, the error 2.3e-3 -> 1.9e-3."""
    _float64_only()
    odd = seq.odd

    def A(x):
        t = 2 * jnp.pi * x[1]
        return jnp.array([0.0, 0.0, x[0] * (1.0 - x[0] ** 2) * jnp.cos(t - 2 * jnp.pi * x[2])])
    curl = odd.G[1] @ odd.interpolate(jax.jit(A), 1)
    B = b0 + PERTURBATION * curl * odd.l2_norm(b0, 2) / odd.l2_norm(curl, 2)
    ts = TimeStepper(seq=seq, newton=True)
    res = relax(initial_state(B, ts), ts, steps=STEPS, chunk=STEPS, verbose=False)
    resid = _descends(res, 1e3 * eps() ** 2)
    start, perturbed, end = field_error(eq, seq, b0), field_error(eq, seq, B), field_error(eq, seq, res.state.B_n)
    print(f"  error against the exact field: projection {start:.3e}, perturbed {perturbed:.3e}, end {end:.3e}")
    assert resid[-1] < BAR
    assert end < 1.1 * start


def test_the_exact_targets_are_a_fixed_point(eq, seq, b0):
    """Newton steps from ``b0`` in two chunks with the exact pressure as the shells' target and the exact transform
    as the target of the iota loop (:func:`mrx.relaxation.twist.iota_newton` before each chunk, from the current
    ``b0`` carries), both on the field's label rebuilt before each chunk: they descend, end below :data:`BAR`, at the
    pressure and the transform within :data:`TARGET_BANDS` and as close to the exact field as the projection, within
    1.1 times the error of ``b0``: residual 7e-13 after 20 steps, error 2.3e-3 -> 1.9e-3 (tokamak (12,16,6), float64, 2026-11-22)."""
    _float64_only()
    ts = TimeStepper(seq=seq, newton=True)
    p_star = _p_star(eq)
    _, T0, _ = label(seq, b0, ANISOTROPY)
    target = [np.asarray(current_on_label(seq, b0, T0, CELLS), dtype=np.float64)]

    def rebuild(state, it):
        s_q, T, _ = label(seq, state.B_n, ANISOTROPY)
        shells, _ = build_shells(seq, state.B_n, s_q, p_star(s_q), ts.newton_penalty, ts.newton_mass_solve_tol, CELLS)
        ts_k = ts.with_shells(shells)
        slips, target[0], rec = iota_newton(seq, state.B_n, T, s_q, eq.iota, ts_k, target[0])
        return state, ts_k.with_slips(slips)
    res = relax(initial_state(b0, ts), ts, steps=2 * STEPS, chunk=STEPS, rebuild=rebuild, verbose=False)
    resid = _descends(res, 1e3 * eps() ** 2)
    B = res.state.B_n
    s_q, _, _ = label(seq, B, ANISOTROPY)
    X, _ = _points(seq)
    e_p, _ = pressure_error(seq, weak_pressure(current(B, seq), B, seq), s_q, jnp.asarray(_mu0_p(eq, X)), ERROR_BINS)
    r = np.asarray(iota_residual(seq, B, s_q, eq.iota, CELLS + 2)(B)[0])
    miss = float(np.abs(r).max() / eq.iota(0.0))
    start, end = field_error(eq, seq, b0), field_error(eq, seq, B)
    print(f"  pressure {e_p:.3e}, transform's weak residual {miss:.3e}, field {start:.3e} -> {end:.3e}")
    assert resid[-1] < BAR
    assert e_p < TARGET_BANDS["p"] and miss < TARGET_BANDS["iota"]
    assert end < 1.1 * start
