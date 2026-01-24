# %%
from enum import Enum
from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction


class MRXHessian:

    def __init__(self, Seq: DeRhamSequence):
        """ 
        Initialize the MRX Hessian class.

        Parameters
        ----------
        Seq : DeRham sequence
            The de Rham sequence to use.
        """
        self.Seq = Seq

    def δB(self, B: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the δB operator. TODO: Add definition and formula of δB.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.
        u : jnp.ndarray
            The velocity field.

        Returns
        -------
        δB : jnp.ndarray
            The δB operator.
        """
        H = self.Seq.P12 @ B
        uxH = jnp.linalg.solve(self.Seq.M1, self.Seq.P2x1_to_1(u, H))
        return self.Seq.strong_curl @ uxH

    def uxJ(self, B: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the uxJ operator. TODO: Add formula of uxJ.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.
        u : jnp.ndarray
            The velocity field.

        Returns
        -------
        uxJ : jnp.ndarray
            The uxJ operator.
        """
        J = self.Seq.weak_curl @ B
        # Assuming that this was supposed to be P2x1_to_1?
        # This was PP2x1_to_1 earlier but maybe a typo?
        return jnp.linalg.solve(self.Seq.M1, self.Seq.P2x1_to_1(u, J))

    def assemble(self, B: jnp.ndarray) -> jnp.ndarray:
        """
        Assemble the MRX Hessian.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.

        Returns
        -------
        H : jnp.ndarray
            The assembled and symmetrized MRX Hessian.
        """
        X = jnp.eye(B.shape[0])
        δBᵢ = jax.vmap(self.δB, in_axes=(None, 1), out_axes=1)(B, X)
        ΛxJᵢ = jax.vmap(self.uxJ, in_axes=(None, 1), out_axes=1)(B, X)
        H = (δBᵢ.T @ self.Seq.M2 @ δBᵢ
             + ΛxJᵢ.T @ self.Seq.M12 @ self.Seq.M2 @ δBᵢ)
        return (H + H.T) / 2


class MRXDiagnostics:
    """
    A class to compute various important quantities and diagnostics of the MRX relaxation.
    """

    def __init__(self, Seq: DeRhamSequence, force_free: bool = False):
        """
        Initialize the MRX Diagnostics class.

        Parameters
        ----------
        Seq : DeRham sequence
            The de Rham sequence to use.
        force_free : bool, default=False
            Whether the problem has grad(p) = 0.
        """
        self.Seq = Seq
        self.force_free = force_free

    def energy(self, B: jnp.ndarray) -> float:
        """
        Compute the magnetic field energy: E = 1/2 |B|^2.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.

        Returns
        -------
        energy : float
            The magnetic field energy.
        """
        return 0.5 * B.T @ self.Seq.M2 @ B

    def helicity(self, B: jnp.ndarray) -> float:
        """
        Compute the magnetic helicity: H = ∫ A · B dx.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.

        Returns
        -------
        helicity : float
            The magnetic helicity.
        """
        A = jnp.linalg.solve(self.Seq.dd1, self.Seq.weak_curl @ B)
        B_harm = B - self.Seq.strong_curl @ A
        return A.T @ self.Seq.M1 @ self.Seq.P12 @ (B + B_harm)

    def harmonic_component(self, B: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the harmonic component of the magnetic field.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.

        Returns
        -------
        B_harm : jnp.ndarray
            The harmonic component of the magnetic field.
        """
        A = jnp.linalg.solve(self.Seq.dd1, self.Seq.weak_curl @ B)
        B_harm = B - self.Seq.strong_curl @ A
        return B_harm

    def divergence_norm(self, B: jnp.ndarray) -> float:
        """
        Compute the norm of the divergence of the magnetic field. 
        Should be at machine precision if the right forms are being used.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.

        Returns
        -------
        divergence_norm : float
            The norm of the divergence of the magnetic field.
        """
        return ((self.Seq.strong_div @ B) @ self.Seq.M3 @ (self.Seq.strong_div @ B))**0.5

    def pressure(self, B_hat: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the pressure. TODO: Add formula of how the pressure is defined when not force-free.

        Parameters
        ----------
        B_hat : jnp.ndarray
            The magnetic field.

        Returns
        -------
        pressure : jnp.ndarray
            The pressure.
        """
        if not self.force_free:
            J_hat = self.Seq.weak_curl @ B_hat
            H_hat = self.Seq.P12 @ B_hat
            JxH_hat = jnp.linalg.solve(
                self.Seq.M2, self.Seq.P1x1_to_2(J_hat, H_hat))
            return -jnp.linalg.solve(self.Seq.dd0, self.Seq.P03 @ self.Seq.strong_div @ JxH_hat)
        else:
            # Compute p(x) = J · B / |B|²
            B_h = DiscreteFunction(B_hat, self.Seq.Lambda_2, self.Seq.E2)
            J_hat = self.Seq.weak_curl @ B_hat
            J_h = DiscreteFunction(J_hat, self.Seq.Lambda_1, self.Seq.E1)

            def lmbda(x):
                DFx = jax.jacfwd(self.Seq.F)(x)
                Bx = B_h(x)
                return (J_h(x) @ Bx) / ((DFx @ Bx) @ DFx @ Bx) * jnp.linalg.det(DFx) * jnp.ones(1)
            return jnp.linalg.solve(self.Seq.M0, self.Seq.P0(lmbda))
    
    


class State(eqx.Module):
    """
    A class to store the state (variables and parameters) of the MRX relaxation.

    Attributes:
    B_n : jnp.ndarray
        The magnetic field at the current time step.
    B_nplus1 : jnp.ndarray (note name change from B_guess to B_nplus1)
        The magnetic field at the next time step.
    v : jnp.ndarray
        The velocity field.
    dt : float
        The time step.
    eta : float
        The resistivity.
    hessian : jnp.ndarray
        The Hessian of the MRX relaxation.
    picard_iterations : int
        The number of Picard iterations.
    picard_residuum : float
        The residuum of the Picard solver.
    F_norm : float (note name change from force_norm to F_norm)
        The norm of the force.
    v_norm : float (note name change from velocity_norm to v_norm)
        The norm of the velocity.
    noise_level : float
        The noise level.
    key : jax.random.PRNGKey
        The random key for noise generation.
    """
    B_n: jnp.ndarray
    B_nplus1: Optional[jnp.ndarray] = None
    v: Optional[jnp.ndarray] = None
    dt: float = 1e-2
    eta: float = 0.0
    hessian: Optional[jnp.ndarray] = None
    picard_iterations: int = 0
    picard_residuum: float = 0.0
    F_norm: float = 0.0
    v_norm: float = 0.0
    noise_level: float = 0.0
    key: jax.Array = eqx.field(default_factory=lambda: jax.random.PRNGKey(67))

    def __post_init__(self):
        if self.B_nplus1 is None:
            object.__setattr__(self, "B_nplus1", self.B_n)

# %%


class IntegrationScheme(Enum):
    EXPLICIT = 0
    IMPLICIT_MIDPOINT = 1


class TimeStepChoice(Enum):
    FIXED = 0
    PICARD_ADAPTIVE = 1
    ANALYTIC_LINESEARCH = 2


class TimeStepper(eqx.Module):
    """
    TimeStepper class.

    Fields
        ----------
    seq : DeRham sequence
            The de Rham sequence to use.
        gamma : int, default=0
        The hyperregularization parameter: v = (I - μ Δ)^{-Ɣ} f
    mu : float, default=0.0
        characteristic length scale for hyperregularization. v = (I - μ Δ)^{-Ɣ} f
        newton : bool, default=False
            Whether to use Newton's method.
    timestep_mode : TimestepMode, default=TimestepMode.EXPLICIT
        The time stepping mode to use.
    picard_tol : float, default=1e-9
            The tolerance for the Picard solver.
    picard_k_restart : int, default=10
        The number of iterations after which to restart the Picard solver.
    picard_k_crit : int, default=4
        If the Picard iterations to convergence is below this value, lower the time step.
    picard_dt_increment : float, default=1.01
        The factor by which to increase/decrease the time step based on Picard iterations.
        force_free : bool, default=False
            Whether the problem has grad(p) = 0.
    conjugate: bool, default=False
        Whether to use the conjugate gradient method.
    b : float, default=1.0
        The parameter for the modified update step:
        v = f + b v⁻ max{ <f, f - v⁻> / <v⁻, v⁻>, 0}.
    dt_mode : DTMode, default=DTMode.ANALYTIC_LS
        The mode for updating the time step:
        - DTMode.FIXED: use the time step stored in the state.
        - DTMode.PICARD_ADAPTIVE: adapt the time step based on Picard iterations taken.
        - DTMode.ANALYTIC_LS: compute the time step using an analytic line search.
    apply_noise : bool, default=False
        Whether to apply noise to the velocity field.
    key: jax.random.PRNGKey = None
        The random key for noise generation.
    """
    seq: DeRhamSequence
    gamma: int = 0
    mu: float = 0.0
    newton: bool = False
    picard_tol: float = 1e-9
    picard_k_restart: int = 20
    picard_k_crit: int = 4
    picard_dt_increment: float = 1.01
    force_free: bool = False
    conjugate: bool = False
    b: float = 1.0
    dt_mode: TimeStepChoice = TimeStepChoice.ANALYTIC_LINESEARCH
    timestep_mode: IntegrationScheme = IntegrationScheme.EXPLICIT
    stochastic: bool = False
    key: Optional[jax.Array] = None

    def norm_2(self, B: jnp.ndarray) -> float:
        """
        Compute the L2 norm of a vector.

        Parameters
        ----------
        B : jnp.ndarray
            The vector we want to compute the norm of.

        Returns
        -------
        norm : float
            The L2 norm of the vector.
        """
        return (B @ self.seq.M2 @ B)**0.5

    def compute_force(self, B: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the force at the current state.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.

        Returns
        -------
        F : jnp.ndarray
            The force at the current state.
        """
        J = self.seq.weak_curl @ B
        H = self.seq.P12 @ B
        JxH = jnp.linalg.solve(
            self.seq.M2, self.seq.P1x1_to_2(J, H))
        if not self.force_free:
            F = self.seq.P_Leray @ JxH
        else:
            F = JxH
        return F, J, H

    def apply_regularization(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """
        Apply hyperregularization to the velocity field.

        Parameters
        ----------
        u_hat : jnp.ndarray
            The velocity field in spectral space.

        Returns
        -------
        u_hat_reg : jnp.ndarray
            The regularized velocity field in spectral space.
        """
        for _ in range(self.gamma):
            u_hat = jnp.linalg.solve(
                jnp.eye(self.seq.M2.shape[0]) + self.mu * self.seq.dd2, u_hat)
        return u_hat

    def apply_inverse_hessian(self, state: State, F: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the inverse Hessian to the velocity field.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        F : jnp.ndarray
            The direction of the force.

        Returns
        -------
        u_hat_inv : jnp.ndarray
            The velocity field after applying the inverse Hessian.
        """
        return -self.seq.P_Leray @ jnp.linalg.lstsq(state.hessian,
                                                    self.seq.P_Leray.T @ self.seq.M2 @ F)[0]
        # P_Leray.T technically not needed because F is already div_free

    def update_field(self, state: State, field_name: Literal['B_n', 'B_nplus1', 'dt', 'eta', 'hessian', 'picard_iterations', 'picard_residuum', 'F_norm', 'v_norm', 'noise_level', 'v'], value) -> State:
        """
        Update any field in the state.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        field_name : str
            The name of the field to update.
        value : any
            The new value for the field.

        Returns
        -------
        state : State
            The updated state of the MRX relaxation.
        """
        return eqx.tree_at(
            lambda s: getattr(s, field_name),
            state,
            value
        )

    def apply_noise(self, v: jnp.ndarray, key: jax.random.PRNGKey, strength: float) -> jnp.ndarray:
        """
        Apply noise to the velocity field.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        v : jnp.ndarray
            The velocity field to which noise will be applied.
        key : jax.random.PRNGKey
            The random key for noise generation.
        strength : float
            The strength of the noise to be applied.

        Returns
        -------
        jnp.ndarray
            The noise applied to the velocity field.
        """
        noise = jnp.linalg.solve(
            self.seq.M2, jax.random.normal(key, v.shape))
        return v + strength * self.seq.apply_leray_projection(noise)

    def midpoint_residuum(self, state: State) -> float:
        """
        Compute the residuum of the Picard solver at the midpoint.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.

        Returns
        -------
        residuum : float
            The residuum of the Picard solver at the midpoint.
        """
        B_guess = state.B_nplus1
        # this updates B_nplus1, force_norm, velocity_norm
        state = self._relaxation_step(state, state.key)
        residuum = self.norm_2(state.B_nplus1 - B_guess)
        return eqx.tree_at(
            lambda s: (s.picard_residuum,
                       s.picard_iterations),
            state,
            (residuum,
             state.picard_iterations + 1,)
        )

    def _relaxation_step(self, state: State, key: jax.random.PRNGKey) -> State:
        """
        Perform a single conjugate gradient step for the MRX relaxation.

        Parameters
        ------------
        state : State
            The state of the MRX relaxation.
        key : jax.random.PRNGKey
            The random key for noise generation.
        Returns
        -------
        state : State
            The updated state of the MRX relaxation. Updates:
            - B_nplus1
            - v
            - F_norm
            - v_norm
            - dt
        """
        if self.timestep_mode == IntegrationScheme.EXPLICIT:
            B_n = state.B_n
        elif self.timestep_mode == IntegrationScheme.IMPLICIT_MIDPOINT:
            B_n = 0.5 * (state.B_n + state.B_nplus1)
        else:
            raise ValueError(
                f"Unknown timestep_mode: {self.timestep_mode}. Supported modes are given by the IntegrationScheme enum.")
        F, J, H = self.compute_force(B_n)
        # TODO: Need to check how Newton interacts with the div = 0 constraint and if another projection is needed.
        if self.newton:
            u = self.apply_inverse_hessian(state, F)
        else:
            u = F
        if self.stochastic:
            u = self.apply_noise(u, key, state.noise_level * self.norm_2(u))
        if self.conjugate:
            v = state.v
            u = u + self.b * v * \
                jnp.maximum((F @ self.seq.M2 @ (u - v)) /
                            (v @ self.seq.M2 @ v), 0.0)
        u = self.apply_regularization(u)
    
        # Project u and H to E1 space using P2x1_to_1
        P2x1_result = self.seq.P2x1_to_1(u, H)
        
        # M1 is (n_E1, n_E1), so P2x1_result should be (n_E1,)
        if P2x1_result.shape[0] != self.seq.M1.shape[0]:
            # Result is in wrong space, project from E2 to E1 using P12
            if P2x1_result.shape[0] == self.seq.M2.shape[0]:
                P2x1_result = self.seq.P12 @ P2x1_result
            else:
                raise ValueError(
                    "Shape mismatch."
                )

        # J should be in E1 space
        if J.shape[0] != self.seq.M1.shape[0]:
            if J.shape[0] == self.seq.M2.shape[0]:
                # J is in E2, need to project to E1
                J = self.seq.P12 @ J
            else:
                raise ValueError("Shape mismatch.")

        solve_result = jnp.linalg.solve(self.seq.M1, P2x1_result)

        # Ensure solve_result and J have matching shapes before calculating E
        if solve_result.shape[0] != J.shape[0]:
            raise ValueError(
                f"Shape mismatch: {solve_result.shape[0]} != {J.shape[0]}"
            )

        E = solve_result - state.eta * J

        dB = self.seq.strong_curl @ E
        if self.dt_mode == TimeStepChoice.FIXED or self.dt_mode == TimeStepChoice.PICARD_ADAPTIVE:
            dt = state.dt
        elif self.dt_mode == TimeStepChoice.ANALYTIC_LINESEARCH:
            dt = F @ self.seq.M2 @ u / (dB @ self.seq.M2 @ dB)
        else:
            raise ValueError(
                f"Unknown dt_mode: {self.dt_mode}. Supported modes are given by the TimeStepChoice enum.")
        B_nplus1 = B_n + dt * dB
        # update state
        return eqx.tree_at(
            lambda s: (s.B_nplus1,
                       s.v,
                       s.F_norm,
                       s.v_norm,
                       s.dt),
            state,
            (B_nplus1,
             u,
             self.norm_2(F),
             self.norm_2(u),
             dt)
        )

    def midpoint_picard_step(self, state: State, key: jax.random.PRNGKey) -> State:
        """
        Picard solver for the MRX relaxation.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        key : jax.random.PRNGKey
            The random key for noise generation.

        Returns
        -------
        state : State
            The updated state of the MRX relaxation.
        """
        def cond_fun(state: State) -> bool:
            """
            Condition function for the while loop of the Picard solver.
            Continue while either residual or change is above tolerance.

            Parameters
            ----------
            state : State
                The state of the MRX relaxation.

            Returns
            -------
            bool : Whether to continue the loop.
            """
            # Continue while either residual or change is above tolerance.
            return jnp.logical_and(state.picard_iterations < self.picard_k_restart,
                                   jnp.logical_or(state.picard_residuum > self.picard_tol,
                                                  jnp.isnan(state.picard_residuum)))

        def body_fun(state: State) -> State:
            """
            Body function for the while loop of the Picard solver.

            Parameters
            ----------
            state : State
                The state of the MRX relaxation.

            Returns
            -------
            state : State
                The updated state of the MRX relaxation.
            """
            return self.midpoint_residuum(state)

        # Initialize residuum, picard_iterations, key, and v (if None)
        if state.v is None:
            # Initialize v to zeros with same shape as B_n
            v_init = jnp.zeros_like(state.B_n)
        else:
            v_init = state.v
            
        state = eqx.tree_at(
            lambda s: (s.picard_residuum, s.picard_iterations, s.key, s.v),
            state,
            (1.0, 0, key, v_init)
        )

        # Finally, perform the while loop in the Picard solver.
        state = jax.lax.while_loop(cond_fun, body_fun, state)
        return state

    def relaxation_step(self, state: State, key: jax.random.PRNGKey) -> State:
        """
        Perform a single relaxation step for the MRX relaxation.

        Parameters
        ------------
        state : State
            The state of the MRX relaxation.
        key : jax.random.PRNGKey
            The random key for noise generation.
        Returns
        -------
        state : State
            The updated state of the MRX relaxation. Updates:
            - B_nplus1
            - v
            - F_norm
            - v_norm
            - dt
        """
        if self.timestep_mode == IntegrationScheme.EXPLICIT:
            return self._relaxation_step(state, key)
        elif self.timestep_mode == IntegrationScheme.IMPLICIT_MIDPOINT:
            return self.midpoint_picard_step(state, key)
        else:
            raise ValueError(
                f"Unknown timestep_mode: {self.timestep_mode}. Supported modes are given by the IntegrationScheme enum.")


def relaxation_loop(B_dof: jnp.ndarray,
                    ts: TimeStepper,
                    num_iters_outer,
                    num_iters_inner=100,
                    dt0=1.0,
                    force_tolerance: float = 1e-6,
                    key: jax.random.PRNGKey = jax.random.PRNGKey(67),
                    noise_schedule: Optional[Callable[[int], float]] = None,
                    resistivity_schedule: Optional[Callable[[
                        int], float]] = None,
                    callback: Optional[Callable[[State, int], State]] = None,
                    ) -> State:
    """
    Perform multiple relaxation steps for the MRX relaxation.

    Parameters
    ------------
    B_dof : jnp.ndarray
        The initial magnetic field degrees of freedom.
    timestepper : TimeStepper
        The TimeStepper object to use for the relaxation.
    num_iters_outer : int
        The number of outer relaxation steps to perform (using a python for loop).
    num_iters_inner : int, default=100
        The number of inner relaxation steps to perform (using jax.lax.scan).
    dt0 : float, default=1.0
        The initial time step to use.
    force_tolerance : float, default=1e-6
        The tolerance for the force norm to consider the relaxation converged.
    key : jax.random.PRNGKey, default=jax.random.PRNGKey(67)
        The random key for noise generation.
    noise_schedule : Optional[Callable[[int], float]], default=None
        An optional noise schedule to use for the relaxation. This is function that takes the outer iteration index and returns the noise level to be used.
    resistivity_schedule : Optional[Callable[[int], float]], default=None
        An optional resistivity schedule to use for the relaxation. This is function that takes the outer iteration index and returns the resistivity to be used.
    callback : Optional[Callable[[State, int], State]], default=None
        A callback function to be called after each outer iteration. The signature is callback(state: State, iteration: int) -> State and it overrides the current state.

    Returns
    -------
    state : State
        The final state of the MRX relaxation.
    traces : dict
        A dictionary containing traces of various quantities during the relaxation: 
        - force_norm
        - helicity
        - timestep
        - energy
        - picard_residua
        - picard_iterations
        - velocity_norm
    """
    seq = ts.seq
    diagnostics = MRXDiagnostics(seq, ts.force_free)
    get_helicity = jax.jit(diagnostics.helicity)
    if ts.newton:
        hessian_assembler = MRXHessian(seq)
        assemble_hessian = jax.jit(hessian_assembler.assemble)
        H = assemble_hessian(B_dof)
    else:
        H = None
    F, _, _ = ts.compute_force(B_dof)
    state = State(B_n=B_dof,
                  dt=dt0,
                  hessian=H,
                  v=F,
                  F_norm=ts.norm_2(F))
    if ts.newton:
        v = ts.apply_inverse_hessian(state, F)
    else:
        v = F
    state = ts.update_field(state, "v", v)
    state = ts.update_field(state, "v_norm", ts.norm_2(v))
    # ---- diagnostics ----
    force_norm_trace = [state.F_norm]
    helicity_trace = [get_helicity(state.B_n)]
    timesteps = [state.dt]
    energy_trace = [state.B_n @ seq.M2 @ state.B_n/2]
    picard_residua = [state.picard_residuum]
    picard_iterations = [state.picard_iterations]
    velocity_norm_trace = [state.v_norm]
    divergence_B_trace = [diagnostics.divergence_norm(state.B_n)]

    traces = {
        "force_norm": force_norm_trace,
        "helicity": helicity_trace,
        "timestep": timesteps,
        "energy": energy_trace,
        "picard_residua": picard_residua,
        "picard_iterations": picard_iterations,
        "velocity_norm": velocity_norm_trace,
        "divergence_B": divergence_B_trace,
    }
    # -----------------------
    print(
        f"Initial condition: \nforce norm: {state.F_norm:.2e} \nhelicity: {helicity_trace[-1]:.2e} \nenergy: {energy_trace[-1]:.2e} \n------------------------")

    def body_fn(state, key):
        # ---- one state update ----
        state = ts.relaxation_step(state, key)
        failed = (state.picard_residuum > ts.picard_tol) | (
            ~jnp.isfinite(state.picard_residuum))

        def on_fail(state):
            state = ts.update_field(state, "dt", state.dt / 2)
            state = ts.update_field(
                state, "B_nplus1", state.B_n)  # restart with halved dt
            return state

        def on_success(state):
            state = ts.update_field(state, "B_n", state.B_nplus1)
            if ts.dt_mode == 'picard_adaptive':
                dt_new = jnp.where(state.picard_iterations < ts.picard_k_crit,
                                   state.dt * ts.picard_dt_increment,   # few iterations → increase dt
                                   state.dt / ts.picard_dt_increment)   # many iterations → decrease dt
                state = ts.update_field(state, "dt", dt_new)
            return state
        state = jax.lax.cond(failed, on_fail, on_success, state)
        return state, None

    for i in range(num_iters_outer + 1):
        key, _ = jax.random.split(key)
        if noise_schedule is not None:
            state = ts.update_field(state, "noise_level", noise_schedule(i))
        if resistivity_schedule is not None:
            state = ts.update_field(
                state, "eta", resistivity_schedule(i))
        if ts.newton:
            H = assemble_hessian(state.B_n)
            state = ts.update_field(state, "hessian", H)
        state, _ = jax.lax.scan(
            body_fn, state, jax.random.split(key, num_iters_inner))
        # ---- diagnostics ----
        force_norm_trace.append(state.F_norm)
        helicity_trace.append(get_helicity(state.B_n))
        timesteps.append(state.dt)
        energy_trace.append(state.B_n @ seq.M2 @ state.B_n/2)
        picard_residua.append(state.picard_residuum)
        picard_iterations.append(state.picard_iterations)
        velocity_norm_trace.append(state.v_norm)
        divergence_B_trace.append(diagnostics.divergence_norm(state.B_n))
        # -----------------------
        if callback is not None:
            state = callback(state, i * num_iters_inner)
        print(
            f"Iteration {i * num_iters_inner}: \nforce norm: {state.F_norm:.2e} \nrelative helicity change: {jnp.abs(helicity_trace[0] - helicity_trace[-1])/helicity_trace[0]:.2e} \ndt: {state.dt:.2e} \nrelative energy change: {(energy_trace[0] - energy_trace[-1])/energy_trace[0]:.2e} \npicard iterations: {state.picard_iterations} \npicard residuum: {state.picard_residuum:.2e} \n------------------------")
        if state.F_norm < force_tolerance:
            break

    return state, traces

# %%
