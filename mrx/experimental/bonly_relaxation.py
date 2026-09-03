"""Energy descent WITHOUT the auxiliary 1-form H: the cross products are formed
from the 2-form B directly, J x B for the force and u x B for the electric field.

The production step (:class:`mrx.relaxation.TimeStepper`) projects B onto the
1-forms, ``H = M_1^-1 P B``, and uses H in both cross products. Helicity is
conserved because its discrete rate, the dual E paired with H, is the
integral of ``(u x H) . H``, zero at every quadrature point. This variant
keeps the energy identity, ``(u x B) . J = -u . (J x B)`` pointwise with the
same J on both sides, but the helicity rate becomes the integral of
``(u x B) . H_h``, and H_h differs from B_h by the projection error. It exists
to show that drift; ``scripts/relax.py --stepper bonly`` runs it.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.relaxation import (DescentMethod, State, TimeStepChoice, TimeStepper,
                            resistive_step)


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
    line search, CFL cap, backward-Euler resistivity); ``state.H`` is left
    untouched and ``state.JxH`` carries the ``J x B`` warm start.
    """

    def relaxation_step(self, state: State) -> State:
        B_n = state.B_n
        seq = self.seq
        J = seq.apply_weak_curl(B_n, dirichlet=True)
        B_jk = seq.evaluate_at_quadrature(B_n, 2, True)
        J_jk = seq.evaluate_at_quadrature(J, 1, True)
        JxB_dual = seq.cross_product_load_values(J_jk, B_jk, 2, 1, 2, True)
        JxB = seq.apply_inverse_mass_matrix(JxB_dual, 2, guess=state.JxH)
        F, p = seq.apply_leray_projection(JxB, k=2, p_guess=state.p)
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
        u, p_v = seq.apply_leray_projection(u, k=2, p_guess=state.p_v)
        Mu = seq.apply_mass_matrix(u, 2)

        u_jk = seq.evaluate_at_quadrature(u, 2, True)
        E_dual = seq.cross_product_load_values(u_jk, B_jk, 1, 2, 2, True)
        E = seq.apply_inverse_mass_matrix(E_dual, 1, guess=state.E)
        cfl_max = jnp.max(jnp.abs(u_jk) * self.cfl_weights)

        dB = seq.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
        if self.dt_mode == TimeStepChoice.FIXED:
            dt_star = state.dt
        elif self.dt_mode == TimeStepChoice.ANALYTIC_LINESEARCH:
            dt_star = F @ Mu / seq.l2_norm_sq(dB, 2)
        else:
            raise ValueError(f"Unknown dt_mode: {self.dt_mode}.")
        dt = jnp.minimum(dt_star, self.cfl / cfl_max)
        B_ideal = B_n + dt * dB

        resistive_time = state.resistive_time + dt
        resistive_count = state.resistive_count + 1
        if self.resistive:
            due = (state.eta > 0) & (resistive_count >= self.eta_every)

            def resistive(B):
                return resistive_step(B, seq, resistive_time * state.eta)

            def skip(B):
                return B, jnp.int32(0), jnp.zeros((), B.dtype)

            B_nplus1, resistive_info, resistive_delta = jax.lax.cond(due, resistive, skip, B_ideal)
            resistive_time = jnp.where(due, 0.0, resistive_time)
            resistive_count = jnp.where(due, 0, resistive_count)
        else:
            B_nplus1 = B_ideal
            resistive_info = jnp.int32(0)
            resistive_delta = jnp.zeros((), B_ideal.dtype)

        if lbfgs:
            s_hist = jnp.roll(s_hist, 1, axis=0).at[0].set(dt * u)
            Ms_hist = jnp.roll(Ms_hist, 1, axis=0).at[0].set(dt * Mu)

        return eqx.tree_at(
            lambda s: (s.B_nplus1, s.v, s.p, s.p_v, s.JxH, s.E,
                       s.F_prev, s.MF_prev, s.F_norm, s.v_norm, s.lbfgs_sy,
                       s.dt, s.dt_star, s.cfl_max, s.resistive_info,
                       s.resistive_delta, s.resistive_count, s.resistive_time,
                       s.s_history, s.y_history, s.Ms_history, s.My_history),
            state,
            (B_nplus1, u, p, p_v, JxB, E,
             F, MF, jnp.sqrt(F @ MF), jnp.sqrt(u @ Mu), sy,
             dt, dt_star, cfl_max, resistive_info,
             resistive_delta, resistive_count, resistive_time,
             s_hist, y_hist, Ms_hist, My_hist))


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
