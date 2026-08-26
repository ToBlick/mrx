"""One relaxation step on ``tiny_seq`` (ci tier).

1. ``test_resistive_step`` -- at eta = 0 the step is the ideal one,
   ``B_n + dt curl E``, with the resistive solve skipped; at eta > 0 the
   backward-Euler defect solve satisfies its equation, lowers the energy
   below the ideal step's, keeps div B at solver tolerance, and
   ``eta_every`` defers it.
2. ``test_cfl_cap`` -- a non-binding cap reproduces the ``cfl = inf`` step
   to round-off; a binding one takes ``cfl / cfl_max`` and still lowers the
   energy.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import mrx
from mrx.relaxation import TimeStepper, initial_state


@pytest.fixture(scope="module")
def B0(tiny_seq):
    """A random divergence-free 2-form with ``B . n = 0``: the topological curl of a random Dirichlet 1-form."""
    A = jax.random.normal(jax.random.PRNGKey(3), (tiny_seq.n1_dbc,), dtype=mrx.DTYPE)
    B = tiny_seq.apply_incidence_matrix(A, 1, dirichlet_in=True, dirichlet_out=True)
    return B / tiny_seq.l2_norm(B, 2)


def _energy(seq, B):
    return float(0.5 * seq.l2_norm_sq(B, 2))


def _div(seq, B):
    return float(seq.l2_norm(seq.apply_incidence_matrix(
        B, 2, dirichlet_in=True, dirichlet_out=True), 3))


def _step(ts):
    return jax.jit(lambda s: ts.relaxation_step(s, s.key))


@pytest.fixture(scope="module")
def free_stepper(tiny_seq, B0):
    """``TimeStepper(cfl=inf, resistive=True)``, its compiled step and the
    initial state at ``B0``. The resistive and the CFL tests both start from
    exactly this stepper, so it is compiled once rather than once per test
    (a step compile is ~7 s on four CPU cores, ~12 s on the GPU). The CFL
    test's other two steppers are ideal-only (``resistive=False``): their
    steps must agree with this one's ``eta = 0`` branch to round-off."""
    ts = TimeStepper(seq=tiny_seq, cfl=float("inf"), resistive=True)
    return ts, _step(ts), initial_state(B0, ts, dt=1.0)


def test_resistive_step(tiny_seq, B0, free_stepper):
    seq = tiny_seq
    ts, step, state0 = free_stepper
    E0 = _energy(seq, B0)
    scale = float(jnp.max(jnp.abs(B0)))

    # eta = 0: the solve is skipped and B_{n+1} = B_n + dt curl E.
    ideal = step(state0)
    assert int(ideal.resistive_info) == 0 and float(ideal.resistive_delta) == 0.0
    # Reconstructed outside the jitted step, so the two agree to round-off
    # (measured 2 ulps), not to the bit.
    curl_E = seq.apply_incidence_matrix(ideal.E, 1, dirichlet_in=True, dirichlet_out=True)
    B_ideal = state0.B_n + ideal.dt * curl_E
    assert float(jnp.max(jnp.abs(ideal.B_nplus1 - B_ideal))) <= 32 * mrx.eps() * scale
    E_ideal = _energy(seq, ideal.B_nplus1)
    assert E_ideal < E0
    assert _div(seq, ideal.B_nplus1) <= 100 * mrx.eps() * scale

    # eta > 0: the same ideal step, then (M_2 + eps L_2) delta = -eps L_2 B_ideal.
    eta = 1e-2
    res = step(eqx.tree_at(lambda s: s.eta, state0, eta))
    assert abs(float(res.dt) - float(ideal.dt)) <= 100 * mrx.eps() * float(ideal.dt)
    assert int(res.resistive_info) < 0, f"resistive MINRES did not converge: {int(res.resistive_info)}"
    assert float(res.resistive_delta) > 0 and int(res.resistive_count) == 0
    eps = float(res.dt) * eta
    lhs = seq.apply_mass_matrix(res.B_nplus1, 2) + eps * seq.apply_laplacian(res.B_nplus1, 2)
    rhs = seq.apply_mass_matrix(B_ideal, 2)
    rel = float(jnp.linalg.norm(lhs - rhs) / jnp.linalg.norm(rhs))
    E_res = _energy(seq, res.B_nplus1)
    div_res = _div(seq, res.B_nplus1)
    print(f"\n  resistive step: MINRES {-int(res.resistive_info)} iterations, "
          f"||delta||/||B|| {float(res.resistive_delta):.2e}, equation residual {rel:.2e}, "
          f"div B {div_res:.2e}, dE ideal {E_ideal - E0:+.3e} resistive {E_res - E_ideal:+.3e}")
    assert rel <= 10 * mrx.sqrt_eps()
    assert E_res < E_ideal
    assert div_res <= 100 * mrx.sqrt_eps() * float(res.resistive_delta) * scale

    # eta_every = 2: the first step is not due; it accumulates dt and skips.
    every2 = TimeStepper(seq=seq, cfl=float("inf"), eta_every=2, resistive=True)
    s2 = _step(every2)(eqx.tree_at(lambda s: s.eta, state0, eta))
    assert int(s2.resistive_info) == 0 and int(s2.resistive_count) == 1
    assert float(s2.resistive_time) == float(s2.dt)
    assert float(jnp.max(jnp.abs(s2.B_nplus1 - ideal.B_nplus1))) <= 32 * mrx.eps() * scale


def test_cfl_cap(tiny_seq, B0, free_stepper):
    seq = tiny_seq
    _, step_free, state0 = free_stepper
    s_free = step_free(state0)
    assert float(s_free.dt) == float(s_free.dt_star) > 0
    assert float(s_free.cfl_max) > 0
    taken = float(s_free.dt_star * s_free.cfl_max)

    # A cap above the step taken does not bind and changes nothing.  (Two
    # steppers are two executables; on the GPU those agree to an ulp, not
    # to the bit -- measured 0.08405602786284262 against ...264.)
    loose = TimeStepper(seq=seq, cfl=2 * taken)
    s_loose = _step(loose)(state0)
    assert float(s_loose.dt) == float(s_loose.dt_star)
    assert abs(float(s_loose.dt) - float(s_free.dt)) <= 8 * mrx.eps() * float(s_free.dt)
    scale = float(jnp.max(jnp.abs(s_free.B_nplus1)))
    assert float(jnp.max(jnp.abs(s_loose.B_nplus1 - s_free.B_nplus1))) <= 32 * mrx.eps() * scale

    # A cap below it binds: dt = cfl / cfl_max, and the energy still falls.
    C = 0.1 * taken
    s_cap = _step(TimeStepper(seq=seq, cfl=C))(state0)
    assert float(s_cap.dt) < float(s_cap.dt_star)
    assert abs(float(s_cap.dt) - C / float(s_cap.cfl_max)) <= 100 * mrx.eps() * float(s_cap.dt)
    assert _energy(seq, s_cap.B_nplus1) < _energy(seq, B0)


# ---------------------------------------------------------------------------
# Twenty production steps through relaxation_loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def B_analytic(tiny_seq):
    """The analytic initial condition on the toroid: prescribed profiles,
    projected and Leray-cleaned (test_initial_conditions.analytic_ic)."""
    from test.test_initial_conditions import analytic_ic
    return analytic_ic(tiny_seq)


# Relative helicity drift over the sixteen ideal (eta = 0) steps: 4.88e-3
# measured 2026-08-26 on tiny_seq in float64 (see the print) at linesearch
# steps dt ~ 1, which is what the toroid's flat energy landscape gives. The
# band is 2x that and an order below the change one resistive step makes
# (+3.8e-2 -> +1.3e-3).
HELICITY_DRIFT = 1e-2


def test_relaxation_loop(tiny_seq, B_analytic):
    """``relaxation_loop`` with the production stepper (CG descent, analytic
    linesearch, ``cfl = 0.5``): five blocks of four steps, the last one
    resistive through ``resistivity_schedule``.

    Every block lowers the energy; ``div B`` stays at the initial condition's
    (the ideal step adds ``dt curl E``, whose divergence is exactly zero);
    the helicity drifts by less than ``HELICITY_DRIFT`` while ``eta = 0`` and
    the resistive solve runs only in the last block; the CFL statistics on
    the final state satisfy ``dt = min(dt_star, cfl / cfl_max)``.
    """
    from mrx.relaxation import DescentMethod, relaxation_loop

    seq = tiny_seq
    ts = TimeStepper(seq=seq, descent_method=DescentMethod.CONJUGATE_GRADIENT,
                     resistive=True)
    outer, inner, eta = 5, 4, 1e-2
    state, traces = relaxation_loop(
        B_analytic, ts, outer, inner,
        resistivity_schedule=lambda i: eta if i == outer else 0.0)

    E = [float(e) for e in traces["energy"]]
    H = [float(h) for h in traces["helicity"]]
    div = [float(d) for d in traces["divergence_B"]]
    F = [float(f) for f in traces["force_norm"]]
    res_info = [int(i) for i in traces["resistive_info"]]
    drift = max(abs(h - H[0]) for h in H[:outer]) / abs(H[0])
    dt, dt_star, cfl_max = float(state.dt), float(state.dt_star), float(state.cfl_max)
    print(f"\n  E {E[0]:.6f} -> {E[-1]:.6f}, |F| {F[0]:.3e} -> {F[-1]:.3e}, "
          f"helicity {H[0]:+.5e} drift {drift:.2e} (ideal) -> {H[-1]:+.5e} (resistive), "
          f"div B max {max(div):.2e}, resistive MINRES {-res_info[-1]} it, "
          f"dt {dt:.3e} dt* {dt_star:.3e} cfl_max {cfl_max:.3e} "
          f"cap {'binds' if dt < dt_star else 'inactive'}")
    assert len(E) == outer + 1
    assert all(E[i + 1] < E[i] for i in range(outer)), E
    assert max(div) <= 10 * seq.tol
    assert drift < HELICITY_DRIFT
    assert res_info[:-1] == [0] * outer, res_info
    assert res_info[-1] < 0, f"resistive MINRES did not converge: {res_info[-1]}"
    assert cfl_max > 0 and dt_star > 0
    assert abs(dt - min(dt_star, ts.cfl / cfl_max)) <= 100 * mrx.eps() * dt
