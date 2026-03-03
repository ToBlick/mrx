# %%
from enum import Enum
from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from mrx.derham_sequence import DeRhamSequence


class MRXDiagnostics:
    """
    A class to compute various important quantities and diagnostics of the MRX relaxation.
    """

    def __init__(self, seq: DeRhamSequence, tol=1e-9, maxiter=1000):
        """
        Initialize the MRX Diagnostics class.

        Parameters
        ----------
        Seq : DeRham sequence
            The de Rham sequence to use.
        """
        self.seq = seq
        self.tol = tol
        self.maxiter = maxiter

    def energy(self, B: jnp.ndarray) -> float:
        return 0.5 * B @ self.seq.apply_m2_sparse(B)

    def helicity(self, B: jnp.ndarray) -> float:
        """
        Compute the magnetic helicity: H = ∫ A · B dx.
        """
        A = cg(self.seq.apply_dd1_sparse, self.seq.apply_d1t_sparse(B), M=self.seq.apply_dd1_precond,
               tol=self.tol, maxiter=self.maxiter)[0]
        curlA = cg(self.seq.apply_m2_sparse, self.seq.apply_d1_sparse(
            A), tol=self.tol, M=self.seq.apply_m2_precond, maxiter=self.maxiter)[0]
        return A @ self.seq.apply_m12_sparse @ (2 * B - curlA)

    def harmonic_component(self, B: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the harmonic component of the magnetic field.
        """
        A = cg(self.seq.apply_dd1_sparse, self.seq.apply_d1t_sparse(B), M=self.seq.apply_dd1_precond,
               tol=self.tol, maxiter=self.maxiter)[0]
        curlA = self.seq.apply_strong_curl(A)
        return B - curlA

    def divergence_norm(self, B: jnp.ndarray) -> float:
        """
        Compute the norm of the divergence of the magnetic field. 
        """
        return (self.seq.apply_d2_sparse(B) @ self.seq.apply_strong_div(B))**0.5

    def pressure(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the pressure.
        """
        # j = curl b
        j = self.seq.apply_weak_curl(b)
        # h = P1 b
        h = cg(self.seq.apply_m1_sparse, self.seq.apply_m12_sparse(b), M=self.seq.apply_m1_precond,
               tol=self.tol, maxiter=self.maxiter)[0]
        jxb = cg(self.seq.apply_m1_sparse, self.seq.P1x1_to_1(j, h), M=self.seq.apply_m1_precond,
                 tol=self.tol, maxiter=self.maxiter)[0]
        return cg(self.seq.apply_dd0_sparse, self.seq.apply_d0t_sparse(jxb), M=self.seq.apply_dd0_precond,
                  tol=self.tol, maxiter=self.maxiter)[0]


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


class DescentMethod(Enum):
    GRADIENT = 0
    CONJUGATE_GRADIENT = 1


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
    preconditioner : Preconditioner, default=Preconditioner.NONE
        The preconditioner to use.
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
    descent_method: DescentMethod = DescentMethod.GRADIENT
    dt_mode: TimeStepChoice = TimeStepChoice.ANALYTIC_LINESEARCH
    timestep_mode: IntegrationScheme = IntegrationScheme.EXPLICIT
    picard_tol: float = 1e-9
    picard_k_restart: int = 20
    picard_k_crit: int = 4
    picard_dt_increment: float = 1.01
    force_free: bool = False
    stochastic: bool = False
    key: Optional[jax.Array] = None

    def norm_2(self, b: jnp.ndarray) -> float:
        """
        Compute the L2 norm of a 2-form.
        """
        return (b @ self.seq.apply_m2_sparse(b))**0.5

    def compute_force(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the force at the current state.
        """
        # j = curl b
        j = self.seq.apply_weak_curl(b)
        # h = P1 b
        h = cg(self.seq.apply_m1_sparse, self.seq.apply_m12_sparse(b), M=self.seq.apply_m1_precond,
               tol=self.tol, maxiter=self.maxiter)[0]
        jxh = cg(self.seq.apply_m2_sparse, self.seq.P1x1_to_2(j, h), M=self.seq.apply_m2_precond,
                 tol=self.tol, maxiter=self.maxiter)[0]
        f = self.seq.apply_leray_projection(
            jxh, nullspace_vectors=self.seq.null_3, k=2)
        return f, j, h

    def regularization_operator(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.seq.apply_m2_sparse(u) + self.mu * self.seq.apply_dd2_sparse(u)

    def apply_regularization(self, u: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.gamma):
            u = cg(self.regularization_operator, u, M=self.seq.apply_m2_precond,
                   tol=self.tol, maxiter=self.maxiter)[0]
        return u

    def update_field(self, state: State, field_name: Literal['B_n', 'B_nplus1', 'dt', 'eta', 'hessian', 'picard_iterations', 'picard_residuum', 'F_norm', 'v_norm', 'noise_level', 'v'], value) -> State:
        return eqx.tree_at(
            lambda s: getattr(s, field_name),
            state,
            value
        )

    def apply_noise(self, v: jnp.ndarray, key: jax.random.PRNGKey, strength: float) -> jnp.ndarray:
        """
        Apply noise to the velocity field.
        """
        noise = cg(self.seq.apply_m2_sparse, jax.random.normal(key, v.shape),
                   M=self.seq.apply_m2_precond, tol=self.tol, maxiter=self.maxiter)[0]
        return v + strength * self.seq.apply_leray_projection(noise)

    def midpoint_residuum(self, state: State) -> float:
        """
        Compute the residuum of the Picard solver at the midpoint.
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
        if self.descent_method == DescentMethod.CONJUGATE_GRADIENT:
            u = F
            v = state.v
            u += v * jnp.maximum((u @ self.seq.m2 @ (u - v)) /
                                 (v @ self.seq.m2 @ v), 0.0)
        elif self.descent_method == DescentMethod.GRADIENT:
            u = F
        else:
            raise ValueError(
                f"Unknown descent_method: {self.descent_method}. Supported methods are given by the DescentMethod enum.")
        if self.stochastic:
            u = self.apply_noise(u, key, state.noise_level * self.norm_2(u))
        u = self.apply_regularization(u)
        u = self.seq.apply_leray_projection(u)

        uxH = self.seq.P2x1_to_1(u, H)

        E = cg(self.seq.apply_m1_sparse, uxH,
               M=self.seq.apply_m1_precond, tol=self.tol, maxiter=self.maxiter)[0]
        E = E - state.eta * J

        dB = self.seq.apply_strong_curl(E)
        if self.dt_mode == TimeStepChoice.FIXED or self.dt_mode == TimeStepChoice.PICARD_ADAPTIVE:
            dt = state.dt
        elif self.dt_mode == TimeStepChoice.ANALYTIC_LINESEARCH:
            dt = F @ self.seq.apply_m2_sparse(u) / \
                (dB @ self.seq.apply_m2_sparse(dB))
        else:
            raise ValueError(
                f"Unknown dt_mode: {self.dt_mode}. Supported modes are given by the TimeStepChoice enum.")
        B_nplus1 = B_n + dt * dB

        H_new = state.hessian

        # update state
        return eqx.tree_at(
            lambda s: (s.B_nplus1,
                       s.v,
                       s.F_norm,
                       s.v_norm,
                       s.dt,
                       s.hessian),
            state,
            (B_nplus1,
             u,
             self.norm_2(F),
             self.norm_2(u),
             dt,
             H_new,
             )
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
    H = None  # No Newton and BFGS for now
    F, _, _ = ts.compute_force(B_dof)
    state = State(B_n=B_dof,
                  dt=dt0,
                  hessian=H,
                  v=F,
                  F_norm=ts.norm_2(F))
    v = F
    state = ts.update_field(state, "v", v)
    state = ts.update_field(state, "v_norm", ts.norm_2(v))
    # ---- diagnostics ----
    force_norm_trace = [state.F_norm]
    helicity_trace = [get_helicity(state.B_n)]
    timesteps = []
    energy_trace = [0.5 * state.B_n @ seq.apply_m2_sparse(state.B_n)]
    picard_residua = []
    picard_iterations = []
    velocity_norm_trace = [state.v_norm]
    divergence_B_trace = [diagnostics.divergence_norm(state.B_n)]
    eta_trace = [state.eta]
    iterations = [0]

    traces = {
        "force_norm": force_norm_trace,
        "helicity": helicity_trace,
        "timestep": timesteps,
        "energy": energy_trace,
        "picard_residua": picard_residua,
        "picard_iterations": picard_iterations,
        "velocity_norm": velocity_norm_trace,
        "divergence_B": divergence_B_trace,
        "eta": eta_trace,
        "iteration": iterations
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

    for i in range(1, num_iters_outer + 1):
        key, _ = jax.random.split(key)
        if noise_schedule is not None:
            state = ts.update_field(state, "noise_level", noise_schedule(i))
        if resistivity_schedule is not None:
            state = ts.update_field(
                state, "eta", resistivity_schedule(i))
        state, _ = jax.lax.scan(
            body_fn, state, jax.random.split(key, num_iters_inner))
        # ---- diagnostics ----
        force_norm_trace.append(state.F_norm)
        helicity_trace.append(get_helicity(state.B_n))
        timesteps.append(state.dt)
        energy_trace.append(0.5 * state.B_n @ seq.apply_m2_sparse(state.B_n))
        picard_residua.append(state.picard_residuum)
        picard_iterations.append(state.picard_iterations)
        velocity_norm_trace.append(state.v_norm)
        divergence_B_trace.append(diagnostics.divergence_norm(state.B_n))
        eta_trace.append(state.eta)
        iterations.append(i * num_iters_inner)
        # -----------------------
        if callback is not None:
            state = callback(state, i)
        print(
            f"Iteration {iterations[-1]}: \nforce norm: {state.F_norm:.2e} \nrelative helicity change: {jnp.abs(helicity_trace[0] - helicity_trace[-1])/helicity_trace[0]:.2e} \ndt: {state.dt:.2e} \nrelative energy change: {(energy_trace[0] - energy_trace[-1])/energy_trace[0]:.2e} \npicard iterations: {state.picard_iterations} \npicard residuum: {state.picard_residuum:.2e} \n------------------------")
        if state.F_norm < force_tolerance:
            break

    return state, traces
