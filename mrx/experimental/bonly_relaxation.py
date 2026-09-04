"""Energy descent WITHOUT the auxiliary 1-form H: the cross products are formed
from the 2-form B directly, J x B for the force and u x B for the electric field.

The production step (:class:`mrx.relaxation.TimeStepper`) projects B onto the
1-forms, ``H = M_1^-1 P B``, and uses H in both cross products. Helicity is
conserved because its discrete rate, the dual E paired with H, is the
integral of ``(u x H) . H``, zero at every quadrature point (when E and H
share a space, see ``TimeStepper._midpoint_solve``). This variant keeps the
energy identity, ``(u x B) . J = -u . (J x B)`` pointwise with the same J on
both sides, but the helicity rate becomes the integral of ``(u x B) . H_h``,
and H_h differs from B_h by the projection error. It exists to show that
drift; ``scripts/relax.py --stepper bonly`` runs it, ``--scheme midpoint``
included, where the drift is that projection error alone.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.relaxation import (PICARD_BLOWUP, DescentMethod, Increment, State, TimeStepper)


def compute_force_bonly(
    B: jnp.ndarray,
    seq: DeRhamSequence,
    dirichlet_H: bool = False,
    p_guess: jnp.ndarray | None = None,
    H_guess: jnp.ndarray | None = None,
    JxH_guess: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The Leray-projected ``J x B`` with ``J`` the weak curl of ``B``.

    Same return tuple as :func:`mrx.relaxation.compute_force` ``(F, p, J, H,
    JxB)`` so the driver's pressure diagnostics, which read ``(J, H)``, keep
    working; H is only computed for them, the force never sees it.
    """
    J = seq.apply_weak_curl(B, dirichlet=True)
    JxB_dual = seq.cross_product_load(J, B, 2, 1, 2, True, True, True)
    JxB = seq.apply_inverse_mass_matrix(JxB_dual, 2, guess=JxH_guess)
    F, p = seq.apply_leray_projection(JxB, k=2, p_guess=p_guess)
    H_dual = seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=dirichlet_H)
    H = seq.apply_inverse_mass_matrix(H_dual, 1, dirichlet=dirichlet_H, guess=H_guess)
    return F, p, J, H, JxB


class BOnlyTimeStepper(TimeStepper):
    """:class:`mrx.relaxation.TimeStepper` with ``J x B`` and ``u x B`` in
    place of ``J x H`` and ``u x H``. The step is otherwise the production
    one (L-BFGS or gradient direction, smoothing, Leray projection, analytic
    line search, CFL cap, backward-Euler resistivity, and the midpoint
    scheme); ``state.H`` is left untouched and ``state.JxH`` carries the
    ``J x B`` warm start.

    Under ``IntegrationScheme.IMPLICIT_MIDPOINT`` the induction is
    ``B_{n+1} = B_n + dt curl(u x B_mid)`` with the 2-form ``B_mid`` itself at
    the quadrature nodes, no proxy: the fixed point costs one k=1 mass
    solve (for ``E``) and a curl per sweep, and the helicity rate is
    ``E^T P B_mid = int (u x B_mid) . H_dir(B_mid)`` with ``H_dir`` the
    Dirichlet 1-form projection of ``B_mid``, which is the projection error
    of the grid and nothing else: the time-integration error is gone, so
    this arm isolates the grid error of the helicity from the time error.
    """

    def _ideal_increment(self, B, state, p_guess, p_v_guess, H_guess, JxH_guess, E_guess):
        seq = self.seq
        J = seq.apply_weak_curl(B, dirichlet=True)
        B_jk = seq.evaluate_at_quadrature(B, 2, True)
        J_jk = seq.evaluate_at_quadrature(J, 1, True)
        JxB_dual = seq.cross_product_load_values(J_jk, B_jk, 2, 1, 2, True)
        JxB = seq.apply_inverse_mass_matrix(JxB_dual, 2, guess=JxH_guess)
        F, p = seq.apply_leray_projection(JxB, k=2, p_guess=p_guess)
        MF = seq.apply_mass_matrix(F, 2)

        lbfgs = self.descent_method == DescentMethod.LBFGS
        s_hist, Ms_hist = state.s_history, state.Ms_history
        y_hist, My_hist = state.y_history, state.My_history
        if lbfgs:
            y_hist = jnp.roll(y_hist, 1, axis=0).at[0].set(state.F_prev - F)
            My_hist = jnp.roll(My_hist, 1, axis=0).at[0].set(state.MF_prev - MF)

        sy = jnp.array(0.0)
        if lbfgs:
            u, sy = self._lbfgs_direction(F, s_hist, y_hist, Ms_hist, My_hist)
        elif self.descent_method == DescentMethod.GRADIENT:
            u = F
        else:
            raise ValueError(f"Unknown descent_method: {self.descent_method}.")
        u = self.smooth_velocity(u)
        u, p_v = seq.apply_leray_projection(u, k=2, p_guess=p_v_guess)
        Mu = seq.apply_mass_matrix(u, 2)

        u_jk = seq.evaluate_at_quadrature(u, 2, True)
        E_dual = seq.cross_product_load_values(u_jk, B_jk, 1, 2, 2, True)
        E = seq.apply_inverse_mass_matrix(E_dual, 1, guess=E_guess)
        cfl_max = jnp.max(jnp.abs(u_jk) * self.cfl_weights)
        dB = seq.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
        return Increment(dB, u, Mu, F, MF, p, p_v, H_guess, JxB, E, cfl_max, sy, y_hist, My_hist)

    def _midpoint_solve(self, state: State):
        """``x = dt curl(u x B(B_n + x / 2))``, Picard as in the production
        solve (same tolerance, blow-up and halving rules), one k=1 mass
        solve per sweep."""
        B_n = state.B_n
        seq = self.seq
        inc0 = self._ideal_increment(B_n, state, state.p, state.p_v, state.H, state.JxH, state.E)
        dt0, dt_star = self._step_size(inc0, state)
        dB0 = inc0.dB
        dB0_norm = seq.l2_norm(dB0, 2)
        u_jk = seq.evaluate_at_quadrature(inc0.u, 2, True)
        one = jnp.ones((), B_n.dtype)

        def sweep(carry):
            k, n_eval, restarts, dt, x, E, resid = carry
            B_jk = seq.evaluate_at_quadrature(B_n + 0.5 * x, 2, True)
            E_dual = seq.cross_product_load_values(u_jk, B_jk, 1, 2, 2, True)
            E = seq.apply_inverse_mass_matrix(E_dual, 1, guess=E)
            g = dt * seq.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
            resid = seq.l2_norm(g - x, 2) / (dt * dB0_norm)
            k, n_eval = k + 1, n_eval + 1
            converged = resid <= self.picard_tol
            restart = (~converged & (~(resid < PICARD_BLOWUP) | (k >= self.picard_max))
                       & (restarts < self.picard_restarts))
            dt = jnp.where(restart, 0.5 * dt, dt)
            x = jnp.where(restart, dt * dB0, g)
            k = jnp.where(restart, 0, k)
            resid = jnp.where(restart, one, resid)
            restarts = restarts + restart.astype(jnp.int32)
            return k, n_eval, restarts, dt, x, E, resid

        def unconverged(carry):
            k, _, _, _, _, _, resid = carry
            return ~(resid <= self.picard_tol) & (resid < PICARD_BLOWUP) & (k < self.picard_max)

        carry = (jnp.int32(0), jnp.int32(1), jnp.int32(0), dt0, dt0 * dB0, inc0.E, one)
        _, n_eval, restarts, dt, x, E, resid = jax.lax.while_loop(unconverged, sweep, carry)
        return inc0._replace(E=E), dt, dt_star, B_n + x, n_eval, restarts, resid


def initial_state_bonly(B_dof: jnp.ndarray, ts: BOnlyTimeStepper, dt: float = 1.0) -> State:
    """:func:`mrx.relaxation.initial_state` seeded from :func:`compute_force_bonly`,
    so the first secant pairs with the force the step actually uses."""
    seq = ts.seq
    n = seq.n(2, True)
    m = ts.history_size if ts.descent_method == DescentMethod.LBFGS else 0
    F0, p0, _, H0, JxB0 = compute_force_bonly(B_dof, seq, dirichlet_H=ts.dirichlet_H)
    MF0 = seq.apply_mass_matrix(F0, 2)
    return State(
        B_n=B_dof,
        dt=dt,
        dt_star=dt,
        resistive_info=jnp.int32(0),
        resistive_count=jnp.int32(0),
        v=jnp.zeros(n),
        p=p0,
        p_v=jnp.zeros(seq.n(3, True)),
        H=H0,
        JxH=JxB0,
        E=jnp.zeros(seq.n(1, True)),
        A=jnp.zeros(seq.n(1, True)),
        F_prev=F0,
        MF_prev=MF0,
        F_norm=jnp.sqrt(F0 @ MF0),
        s_history=jnp.zeros((m, n)),
        y_history=jnp.zeros((m, n)),
        Ms_history=jnp.zeros((m, n)),
        My_history=jnp.zeros((m, n)),
    )
