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


def test_resistive_step(tiny_seq, B0):
    seq = tiny_seq
    ts = TimeStepper(seq=seq, cfl=float("inf"))
    step = _step(ts)
    state0 = initial_state(B0, ts, dt=1.0)
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
    every2 = TimeStepper(seq=seq, cfl=float("inf"), eta_every=2)
    s2 = _step(every2)(eqx.tree_at(lambda s: s.eta, state0, eta))
    assert int(s2.resistive_info) == 0 and int(s2.resistive_count) == 1
    assert float(s2.resistive_time) == float(s2.dt)
    assert float(jnp.max(jnp.abs(s2.B_nplus1 - ideal.B_nplus1))) <= 32 * mrx.eps() * scale


def test_cfl_cap(tiny_seq, B0):
    seq = tiny_seq
    free = TimeStepper(seq=seq, cfl=float("inf"))
    state0 = initial_state(B0, free, dt=1.0)
    s_free = _step(free)(state0)
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
