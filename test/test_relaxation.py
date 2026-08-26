"""One production relaxation run on ``tiny_seq``: ``mrx.relaxation`` end to end.

The most general stepper -- CG descent with the analytic linesearch, the
CFL cap, the ``(1 + mu L)^-1`` hyperregularisation (``gamma = 1``) and
the backward-Euler resistive solve live from the first step -- is
compiled ONCE and driven for ``STEPS`` steps from the logical-profile
initial condition. Along the trajectory:

* the energy falls at every step, and the measured drop agrees with the
  linesearch prediction ``dE = -dt (F, u)_M (1 - dt / 2 dt*)`` for the
  ideal part of the step (exact for the quadratic energy, up to the mass
  solves at ``seq.tol``); the resistive part lowers it further;
* ``div B`` stays at the initial condition's, i.e. at solver tolerance;
* ``dt = min(dt*, cfl / cfl_max)`` -- the CFL cap invariant;
* ``eta = 0`` reproduces the ideal step ``B_n + dt curl E`` and skips the
  solve; at ``eta > 0`` the solve satisfies its equation;
* the helicity rate of the resistive step. With ``E = -u x B + eta J`` the
  identity is ``dH/dt = -2 eta <J, B>``; for the backward-Euler step
  ``delta = -eps curl J_{n+1}`` (``eps = eta dt``) the polarised form
  ``H(B_{n+1}) - H(B_ideal) = -eps (<J_{n+1}, B_ideal> + <J_{n+1}, B_{n+1}>)``
  is exact for the quadratic helicity, and either single-time form
  ``-2 eps <J, B>`` is off by ``O(eps^2)`` -- the one at ``B_{n+1}`` with
  a far smaller constant, since the implicit step's current IS ``J_{n+1}``.
  Halving ``eta`` at the same state halves ``eps`` with the ideal part of
  the step unchanged, which is the ``dt -> dt/2`` refinement of the
  resistive substep without a second compiled stepper; the single-time
  error must fall by ~4x.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

import mrx
from mrx.relaxation import (
    DescentMethod,
    TimeStepper,
    compute_helicity,
    initial_state,
)
from test.test_initial_conditions import logical_ic

ETA, MU, CFL = 1e-3, 1e-2, 0.5
STEPS = 12
CHECK = 6           # the step at which the resistive identities are measured


def test_relaxation(tiny_seq):
    seq = tiny_seq
    B0 = logical_ic(seq)
    ts = TimeStepper(seq=seq, descent_method=DescentMethod.CONJUGATE_GRADIENT,
                     resistive=True, gamma=1, mu=MU, cfl=CFL)

    @jax.jit
    def step(state):
        s = ts.relaxation_step(state, state.key)
        return eqx.tree_at(lambda t: t.B_n, s, s.B_nplus1)

    def energy(B):
        return 0.5 * seq.l2_norm_sq(B, 2)

    @jax.jit
    def probe(B_prev, state):
        """The ideal step reconstructed from the post-step state, its energy
        prediction, and the diagnostics of the full step."""
        curl_E = seq.apply_incidence_matrix(state.E, 1, dirichlet_in=True, dirichlet_out=True)
        B_ideal = B_prev + state.dt * curl_E
        Fu = state.F_prev @ seq.apply_mass_matrix(state.v, 2)
        dE_pred = -state.dt * Fu * (1.0 - 0.5 * state.dt / state.dt_star)
        div = seq.l2_norm(seq.apply_incidence_matrix(
            state.B_n, 2, dirichlet_in=True, dirichlet_out=True), 3)
        return B_ideal, energy(B_ideal), energy(state.B_n), dE_pred, div

    @jax.jit
    def current_pairing(B_J, B):
        """``<J, B>`` with ``J = curl B_J``: the L2 pairing of the Dirichlet
        1-form with the 2-form through the mixed mass."""
        J = seq.apply_weak_curl(B_J, dirichlet_in=True, dirichlet_out=True)
        return J @ seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True)

    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])

    state = eqx.tree_at(lambda s: s.eta, initial_state(B0, ts), ETA)
    A = jnp.zeros(seq.n1_dbc)
    E0 = float(energy(B0))
    H0, A = get_helicity(B0, seq, A)
    E_prev, F_prev = E0, float(state.F_norm)
    scale = float(jnp.max(jnp.abs(B0)))
    ratios_ideal, ratios_full, divs = [], [], []
    for n in range(1, STEPS + 1):
        if n == CHECK:
            pre = state          # the pre-step state the resistive checks restart from
        B_prev = state.B_n
        state = step(state)
        B_ideal, E_ideal, E_new, dE_pred, div = (probe(B_prev, state))
        E_ideal, E_new, dE_pred, div = (float(x) for x in (E_ideal, E_new, dE_pred, div))
        dt, dt_star, cfl_max = float(state.dt), float(state.dt_star), float(state.cfl_max)
        assert int(state.resistive_info) < 0, (
            f"step {n}: resistive MINRES did not converge ({int(state.resistive_info)})")
        assert E_ideal < E_prev and E_new < E_ideal, (n, E_prev, E_ideal, E_new)
        assert cfl_max > 0 and dt_star > 0
        assert abs(dt - min(dt_star, CFL / cfl_max)) <= 100 * mrx.eps() * dt
        ratios_ideal.append((E_ideal - E_prev) / dE_pred)
        ratios_full.append((E_new - E_prev) / dE_pred)
        divs.append(div)
        if n == CHECK:
            check_state, check_B_ideal = state, B_ideal
        E_prev = E_new
    print(f"\n  {STEPS} steps: E {E0:.6f} -> {E_prev:.6f}, |F| {F_prev:.3e} -> "
          f"{float(state.F_norm):.3e}, dt {dt:.3e} dt* {dt_star:.3e} "
          f"cap {'binds' if dt < dt_star else 'inactive'}, div B max {max(divs):.2e}, "
          f"dE_meas/dE_pred ideal [{min(ratios_ideal):.6f}, {max(ratios_ideal):.6f}] "
          f"full [{min(ratios_full):.4f}, {max(ratios_full):.4f}]")

    # --- the resistive step at CHECK, from the same pre-step state --------
    # eta = 0: the ideal step alone, the solve skipped.
    ideal = step(eqx.tree_at(lambda s: s.eta, pre, 0.0))
    assert int(ideal.resistive_info) == 0 and float(ideal.resistive_delta) == 0.0
    # Two executables (the step and the probe) apply the incidence operator
    # -- a dense polar-core Gram solve inside -- in different fusion orders:
    # measured 2.4e-14 absolute at max |B| = 3.6e-2 on the H100 (3e3 eps),
    # below 32 eps on the CPU.
    assert float(jnp.max(jnp.abs(ideal.B_n - check_B_ideal))) <= mrx.eps(1e4) * scale
    # eta > 0: (M_2 + eps L_2) B_{n+1} = M_2 B_ideal.
    B1 = check_state.B_n
    eps = float(check_state.dt) * ETA
    lhs = seq.apply_mass_matrix(B1, 2) + eps * seq.apply_laplacian(B1, 2)
    rhs = seq.apply_mass_matrix(check_B_ideal, 2)
    rel = float(jnp.linalg.norm(lhs - rhs) / jnp.linalg.norm(rhs))
    assert rel <= 10 * mrx.sqrt_eps(), rel

    # The helicity rate, at eps and eps/2.
    half = step(eqx.tree_at(lambda s: s.eta, pre, 0.5 * ETA))
    assert abs(float(half.dt) - float(check_state.dt)) <= 8 * mrx.eps() * float(check_state.dt)
    H_ideal, A = get_helicity(check_B_ideal, seq, A)
    results = {}
    for label, B_new, e in (("eps", B1, eps), ("eps/2", half.B_n, 0.5 * eps)):
        H_new, A = get_helicity(B_new, seq, A)
        dH = float(H_new - H_ideal)
        pred_n = -2.0 * e * float(current_pairing(check_B_ideal, check_B_ideal))
        pred_n1 = -2.0 * e * float(current_pairing(B_new, B_new))
        pred_mid = -e * float(current_pairing(B_new, check_B_ideal)
                              + current_pairing(B_new, B_new))
        results[label] = (dH, pred_n, pred_n1, pred_mid)
        print(f"  helicity rate at {label} = {e:.3e}: dH {dH:+.6e}, "
              f"-2 eps <J,B> at B_n {pred_n:+.6e} (err {dH - pred_n:+.2e}), "
              f"at B_n+1 {pred_n1:+.6e} (err {dH - pred_n1:+.2e}), "
              f"polarised {pred_mid:+.6e} (err {dH - pred_mid:+.2e})")
    (dH, pn, pn1, pm), (dHh, pnh, _, _) = results["eps"], results["eps/2"]
    print(f"  H {float(H0):+.5e} -> {float(get_helicity(state.B_n, seq, A)[0]):+.5e} "
          f"over {STEPS} resistive steps")

    # The ideal step's energy is a quadratic polynomial in dt, so the
    # prediction is exact up to the mass solves behind F, u and E at
    # seq.tol: measured 2026-08-26 (see the print) 1 +- 1e-6 in float64.
    # The full step adds the resistive drop -eps ||J||^2 on top, which does
    # not shrink with |F|: the full ratio ran from 1.35 (step 1) to 17.0
    # (step 12) at eta = 1e-3, so only its lower bound is a statement.
    assert all(abs(r - 1.0) <= 1e3 * seq.tol for r in ratios_ideal), ratios_ideal
    assert all(r >= 1.0 - 1e3 * seq.tol for r in ratios_full), ratios_full
    assert max(divs) <= 10 * seq.tol
    # Measured 2026-08-26 in float64 at eps = 1.174e-3 (dt = 1.17, see the
    # print): dH = -1.602378e-3; the polarised form is off by 1.1e-11
    # (7e-9 relative, the helicity solves); -2 eps <J, B> at B_{n+1} by
    # -8.2e-7 and at B_n by +1.45e-4, and at eps/2 by -2.3e-7 and +3.9e-5:
    # ratios 0.275 and 0.267, the O(eps^2) quartering. So the backward-Euler
    # step makes the rate at B_{n+1} the accurate one (175x smaller
    # constant), and the exact statement is the polarised one. The
    # 1e2 tol |dH| terms are the helicity solves' resolution of dH (1.2e-5
    # in float32, where they carry the assertion).
    floor = 1e2 * seq.tol * abs(dH)
    assert abs(dH - pm) <= floor
    assert abs(dH - pn1) <= 0.01 * abs(dH - pn) + floor
    assert abs(dHh - pnh) <= 0.35 * abs(dH - pn) + floor
