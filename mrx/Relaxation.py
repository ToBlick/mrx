# %%
import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.DifferentialForms import DiscreteFunction


class MRXHessian:

    def __init__(self, Seq):
        self.Seq = Seq

    def δB(self, B, u):
        H = self.Seq.P12 @ B
        uxH = jnp.linalg.solve(self.Seq.M1, self.Seq.P2x1_to_1(u, H))
        return self.Seq.strong_curl @ uxH

    def uxJ(self, B, u):
        J = self.Seq.weak_curl @ B
        return jnp.linalg.solve(self.Seq.M1, self.Seq.PP2x1_to_1(u, J))

    def assemble(self, B):
        X = jnp.eye(B.shape[0])
        δBᵢ = jax.vmap(self.δB, in_axes=(None, 1), out_axes=1)(B, X)
        ΛxJᵢ = jax.vmap(self.uxJ, in_axes=(None, 1), out_axes=1)(B, X)
        H = (δBᵢ.T @ self.Seq.M2 @ δBᵢ
             + ΛxJᵢ.T @ self.Seq.M12 @ self.Seq.M2 @ δBᵢ)
        return (H + H.T) / 2


class MRXDiagnostics:

    def __init__(self, Seq, force_free=False):
        self.Seq = Seq
        self.force_free = force_free

    def energy(self, B):
        return 0.5 * B.T @ self.Seq.M2 @ B

    def helicity(self, B):
        A = jnp.linalg.solve(self.Seq.dd1, self.Seq.weak_curl @ B)
        B_harm = B - self.Seq.strong_curl @ A
        return A.T @ self.Seq.M1 @ self.Seq.P12 @ (B + B_harm)

    def harmonic_component(self, B):
        A = jnp.linalg.solve(self.Seq.dd1, self.Seq.weak_curl @ B)
        B_harm = B - self.Seq.strong_curl @ A
        return B_harm

    def divergence_norm(self, B):
        return ((self.Seq.strong_div @ B) @ self.Seq.M3 @ (self.Seq.strong_div @ B))**0.5

    def pressure(self, B_hat):
        if not self.force_free:
            J_hat = self.Seq.weak_curl @ B_hat
            H_hat = self.Seq.P12 @ B_hat
            JxH_hat = jnp.linalg.solve(
                self.Seq.M2, self.Seq.P1x1_to_2(J_hat, H_hat))
            return -jnp.linalg.solve(self.Seq.dd0, self.Seq.P03 @ self.Seq.strong_div @ JxH_hat)
        else:
            # Compute p(x) = J · B / |B|²
            B_h = DiscreteFunction(B_hat, self.Seq.Λ2, self.Seq.E2)
            J_hat = self.Seq.weak_curl @ B_hat
            J_h = DiscreteFunction(J_hat, self.Seq.Λ1, self.Seq.E1)

            def lmbda(x):
                DFx = jax.jacfwd(self.Seq.F)(x)
                Bx = B_h(x)
                return (J_h(x) @ Bx) / ((DFx @ Bx) @ DFx @ Bx) * jnp.linalg.det(DFx) * jnp.ones(1)
            return jnp.linalg.solve(self.Seq.M0, self.Seq.P0(lmbda))


class State(eqx.Module):
    B_n: jax.Array
    B_guess: jax.Array
    dt: float
    eta: float
    Hessian: jax.Array
    picard_iterations: int
    picard_residuum: float
    force_norm: float
    velocity_norm: float


class TimeStepper:

    def __init__(self, Seq, gamma=0, newton=False, picard_tol=1e-12, picard_maxit=20, force_free=False):
        self.Seq = Seq
        self.gamma = gamma
        self.newton = newton
        self.id2 = jnp.eye(Seq.M2.shape[0])
        self.picard_tol = picard_tol
        self.picard_maxit = picard_maxit
        self.force_free = force_free

    def init_state(self, B, dt, eta, Hessian):
        return self.State(B, B, dt, eta, Hessian)

    def norm_2(self, B):
        return (B @ self.Seq.M2 @ B)**0.5

    def midpoint_residuum(self, state):
        dt = state.dt
        B_guess = state.B_guess
        B_n = state.B_n
        B_mid = (B_guess + B_n) / 2

        J_hat = self.Seq.weak_curl @ B_mid
        H_hat = self.Seq.P12 @ B_mid
        JxH_hat = jnp.linalg.solve(
            self.Seq.M2, self.Seq.P1x1_to_2(J_hat, H_hat))
        if not self.force_free:
            f_hat = self.Seq.P_Leray @ JxH_hat
        else:
            f_hat = JxH_hat
        u_hat = f_hat
        for _ in range(self.gamma):
            u_hat = jnp.linalg.solve(self.id2 + self.Seq.dd2, u_hat)
        if self.newton:
            u_hat = self.Seq.P_Leray @ jnp.linalg.lstsq(
                state.Hessian, self.Seq.P_Leray.T @ self.Seq.M2 @ u_hat)[0]
            # P_Leray.T technically not needed because u is already div_free
        E_hat = jnp.linalg.solve(self.Seq.M1,
                                 self.Seq.P2x1_to_1(u_hat, H_hat)) - state.eta * J_hat
        B_nplus1 = B_n + dt * self.Seq.strong_curl @ E_hat
        residuum = self.norm_2(B_nplus1 - B_guess)
        force_norm = self.norm_2(f_hat)
        velocity_norm = self.norm_2(u_hat)
        # update state
        return eqx.tree_at(
            lambda s: (s.B_guess,
                       s.picard_residuum,
                       s.picard_iterations,
                       s.force_norm,
                       s.velocity_norm),
            state,
            (B_nplus1,
             residuum,
             state.picard_iterations + 1,
             force_norm,
             velocity_norm)
        )

    def update_dt(self, state, dt):
        return eqx.tree_at(
            lambda s: s.dt,
            state,
            dt
        )

    def update_hessian(self, state, Hessian):
        return eqx.tree_at(
            lambda s: s.Hessian,
            state,
            Hessian
        )

    def update_B_n(self, state, B):
        return eqx.tree_at(
            lambda s: s.B_n,
            state,
            B
        )

    def update_B_guess(self, state, B):
        return eqx.tree_at(
            lambda s: s.B_guess,
            state,
            B
        )

    def picard_solver(self, state):
        def cond_fun(state):
            # Continue while either residual or change is above tolerance.
            return jnp.logical_and(state.picard_iterations < self.picard_maxit,
                                   jnp.logical_or(state.picard_residuum > self.picard_tol,
                                                  jnp.isnan(state.picard_residuum)))

        def body_fun(state):
            return self.midpoint_residuum(state)

        # initialize residuum and picard_iterations
        state = eqx.tree_at(
            lambda s: (s.picard_residuum, s.picard_iterations),
            state,
            (1.0, 0)
        )
        state = jax.lax.while_loop(cond_fun, body_fun, state)

        return state


def aitken_step(z_prev, z_curr, fz, eps=1e-12, inprod=jnp.vdot):
    d1 = z_curr - z_prev
    d2 = fz - z_curr
    num = inprod(d1, d2)
    den = inprod(d2, d2) + eps
    omega = jnp.clip(num / den, 0.0, 1.0)
    z_next = (1.0 - omega) * z_curr + omega * fz
    return z_next, omega
