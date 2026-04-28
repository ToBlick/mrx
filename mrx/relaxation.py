# %%
from enum import Enum
from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.preconditioners import get_mass_jacobi_diaginv
from mrx.solvers import solve_singular_cg


def apply_diffusion(B: jnp.ndarray, seq: DeRhamSequence, eta: float, dirichlet: bool = True, B_guess: jnp.ndarray | None = None) -> jnp.ndarray:

    B_guess = B if B_guess is None else B_guess

    operators = seq.get_operators()
    if operators is None:
        raise ValueError("Assemble operators first before calling apply_diffusion")
    m_diaginv = get_mass_jacobi_diaginv(operators.mass_preconds, 2, dirichlet)
    dd_diaginv = operators.dd2_diaginv_dbc if dirichlet else operators.dd2_diaginv
    combined_diaginv = 1.0 / (1.0 / m_diaginv + eta / dd_diaginv)

    def apply_A(x):
        return seq.apply_mass_matrix(x, 2, dirichlet) \
            + eta * seq.apply_hodge_laplacian(x, 2, dirichlet)

    def apply_Ainv(x, x0=None):
        return solve_singular_cg(
            apply_A,
            x,
            mass_matvec=lambda x: seq.apply_mass_matrix(x, 2, dirichlet),
            precond_matvec=lambda x: combined_diaginv * x,
            x0=x0,
            maxiter=seq.maxiter, tol=seq.tol
        )[0]

    return apply_Ainv(seq.apply_mass_matrix(B, 2, dirichlet), x0=B_guess)


def compute_helicity(B: jnp.ndarray, seq: DeRhamSequence, A_guess: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    A = seq.apply_inverse_hodge_laplacian(
        seq.apply_weak_curl(B), 1, guess=A_guess)
    B_harm = B - seq.apply_strong_curl(A)
    # <A, B + B_harm>_{L^2} via the 1->2 projection matrix
    helicity = A @ seq.apply_projection_matrix(
        B + B_harm, 2, 1, True, dirichlet_out=True)
    return helicity, A


def compute_divergence_norm(B: jnp.ndarray, seq: DeRhamSequence) -> float:
    # hard-coded dirichlet=True for now
    div_B = seq.apply_strong_div(B)
    return seq.l2_norm_sq(div_B, 3)**0.5

# %%


def compute_force(
    B: jnp.ndarray,
    seq: DeRhamSequence,
    dirichlet_H: bool = False,
    p_guess: jnp.ndarray | None = None,
    H_guess: jnp.ndarray | None = None,
    JxH_guess: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    H_dual = seq.apply_projection_matrix(
        B, 2, 1, True, dirichlet_out=dirichlet_H)
    H = seq.apply_inverse_mass_matrix(
        H_dual, 1, dirichlet=dirichlet_H, guess=H_guess)
    # J = seq.apply_strong_curl(H, dirichlet_in=dirichlet_H, dirichlet_out=True)
    # JxH_dual = seq.cross_product_projection(J, H, 2, 2, 1, True, True, dirichlet_H)
    J = seq.apply_weak_curl(B, dirichlet_in=True, dirichlet_out=True)
    JxH_dual = seq.cross_product_projection(
        J, H, 2, 1, 1, True, True, dirichlet_H)
    JxH = seq.apply_inverse_mass_matrix(JxH_dual, 2, guess=JxH_guess)
    F, p = seq.apply_leray_projection(JxH, k=2, p_guess=p_guess)
    return F, p, J, H, JxH

# %%


class State(eqx.Module):
    """
    A class to store the state (variables and parameters) of the MRX relaxation.

    Attributes:
    B_n : jnp.ndarray
        The magnetic field at the current time step.
    B_nplus1 : jnp.ndarray
        The magnetic field at the next time step.
    v : jnp.ndarray (optional)
        The velocity field.
    p : jnp.ndarray (optional)
        The pressure.
    A : jnp.ndarray (optional)
        The vector potential.
    dt : float
        The time step.
    eta : float
        The resistivity.
    F_prev : jnp.ndarray (optional)
        The force from the previous time step (for L-BFGS y computation).
    s_history : jnp.ndarray (optional)
        History of iterate differences s_k = B_{k+1} - B_k (for L-BFGS).
    y_history : jnp.ndarray (optional)
        History of L^2-gradient differences y_k = grad_M E_{k+1} - grad_M E_k
        = F_k - F_{k+1} (for L-BFGS).  Here grad_M E = -F is the Riesz
        representative of dE w.r.t. the M2 inner product.
    picard_iterations : int
        The number of Picard iterations.
    picard_residuum : float
        The residuum of the Picard solver.
    F_norm : float
        The norm of the force.
    v_norm : float
        The norm of the velocity.
    noise_level : float
        The noise level.
    key : jax.random.PRNGKey
        The random key for noise generation.
    """
    B_n: jnp.ndarray
    B_nplus1: Optional[jnp.ndarray] = None
    p: Optional[jnp.ndarray] = None
    p_v: Optional[jnp.ndarray] = None
    v: Optional[jnp.ndarray] = None
    H: Optional[jnp.ndarray] = None
    JxH: Optional[jnp.ndarray] = None
    E: Optional[jnp.ndarray] = None
    F_prev: Optional[jnp.ndarray] = None
    s_history: Optional[jnp.ndarray] = None
    y_history: Optional[jnp.ndarray] = None
    A: Optional[jnp.ndarray] = None
    dt: float = 1e-2
    eta: float = 0.0
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
    LBFGS = 2


class TimeStepper(eqx.Module):
    """
    TimeStepper for MRX relaxation.

    Fields
    ----------
    seq : DeRhamSequence
        The de Rham sequence.
    gamma : int
        Hyperregularization exponent: v = (I - mu * Δ)^{-gamma} f.
    mu : float
        Length scale for hyperregularization.
    descent_method : DescentMethod
        GRADIENT, CONJUGATE_GRADIENT, or LBFGS.
    dt_mode : TimeStepChoice
        FIXED, PICARD_ADAPTIVE, or ANALYTIC_LINESEARCH.
    timestep_mode : IntegrationScheme
        EXPLICIT or IMPLICIT_MIDPOINT.
    picard_tol : float
        Tolerance for implicit midpoint Picard solver.
    picard_k_restart : int
        Max Picard iterations before restart.
    picard_k_crit : int
        Threshold: fewer iterations -> increase dt, more -> decrease.
    picard_dt_increment : float
        Factor for adaptive dt adjustment.
    stochastic : bool
        Whether to add noise to the velocity.
    history_size : int
        Number of stored pairs for CG / L-BFGS.
    dirichlet_H : bool
        Dirichlet BC on H.
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
    stochastic: bool = False
    history_size: int = 1
    dirichlet_H: bool = False

    def __post_init__(self):
        if self.descent_method in (DescentMethod.CONJUGATE_GRADIENT, DescentMethod.LBFGS) and self.history_size < 1:
            raise ValueError(
                "history_size must be at least 1 when using CG or L-BFGS.")

    def _lbfgs_direction(self, F: jnp.ndarray, state: State) -> jnp.ndarray:
        """
        Compute the L-BFGS descent direction v = H_k F using the two-loop recursion.

        The L^2 (M2) inner product is used both to identify the gradient via
        the Riesz map (dℰ[v] = -(F, v)_{L^2} = <-F, v>_M  =>  grad_M E = -F)
        and inside the two-loop recursion (<a, b>_M = a^T M b).

        Stored histories:
            s_k = B_{k+1} - B_k            (iterate differences)
            y_k = grad_M ℰ_{k+1} - grad_M ℰ_k = F_k - F_{k+1}  (gradient differences)

        Falls back to steepest descent (F) when all history entries are zero.
        """
        m = self.history_size
        def apply_M(x): return self.seq.apply_mass_matrix(x, 2)

        s = state.s_history  # (m, n)
        y = state.y_history  # (m, n)

        # --- two-loop recursion ---
        q = F.copy()
        alpha = jnp.zeros(m)

        # first loop: newest (i=0) to oldest (i=m-1)
        for i in range(m):
            rho_i = 1.0 / (s[i] @ apply_M(y[i]) + 1e-30)
            alpha_i = rho_i * (s[i] @ apply_M(q))
            alpha = alpha.at[i].set(alpha_i)
            q = q - alpha_i * y[i]

        # initial Hessian scaling: gamma = (s_0^T M y_0) / (y_0^T M y_0)
        sy = s[0] @ apply_M(y[0])
        yy = y[0] @ apply_M(y[0])
        gamma = jnp.where(yy > 1e-30, sy / yy, 1.0)
        gamma = jnp.maximum(gamma, 1e-30)
        r = gamma * q

        # second loop: oldest (i=m-1) to newest (i=0)
        for i in range(m - 1, -1, -1):
            rho_i = 1.0 / (s[i] @ apply_M(y[i]) + 1e-30)
            beta_i = rho_i * (y[i] @ apply_M(r))
            r = r + (alpha[i] - beta_i) * s[i]

        return r

    def apply_regularization(self, u: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.gamma):
            u = apply_diffusion(u, self.seq, self.mu)
        return u

    def update_field(self, state: State, field_name: Literal['B_n', 'B_nplus1', 'v', 'p_v', 'H', 'JxH', 'E', 's_history', 'y_history', 'F_prev', 'A', 'dt', 'eta', 'picard_iterations', 'picard_residuum', 'F_norm', 'v_norm', 'noise_level', 'key'], value) -> State:  # noqa: E501
        return eqx.tree_at(
            lambda s: getattr(s, field_name),
            state,
            value
        )

    def apply_noise(self, v: jnp.ndarray, key: jax.random.PRNGKey, strength: float) -> jnp.ndarray:
        """
        Apply noise to the velocity field.
        """
        noise = self.seq.apply_inverse_mass_matrix(
            jax.random.normal(key, v.shape), 2)
        return v + strength * self.seq.apply_leray_projection(noise, k=2)[0]

    def midpoint_residuum(self, state: State) -> State:
        B_guess = state.B_nplus1
        # this updates B_nplus1, force_norm, velocity_norm
        state = self._relaxation_step(state, state.key)
        residuum = self.seq.l2_norm(state.B_nplus1 - B_guess, 2)
        return eqx.tree_at(
            lambda s: (s.picard_residuum,
                       s.picard_iterations),
            state,
            (residuum,
             state.picard_iterations + 1,)
        )

    def _relaxation_step(self, state: State, key: jax.random.PRNGKey) -> State:
        if self.timestep_mode == IntegrationScheme.EXPLICIT:
            B_n = state.B_n
        elif self.timestep_mode == IntegrationScheme.IMPLICIT_MIDPOINT:
            B_n = 0.5 * (state.B_n + state.B_nplus1)
        else:
            raise ValueError(
                f"Unknown timestep_mode: {self.timestep_mode}. Supported modes are given by the IntegrationScheme enum.")
        F, p, J, H, JxH = compute_force(
            B_n, self.seq, dirichlet_H=self.dirichlet_H,
            p_guess=state.p, H_guess=state.H, JxH_guess=state.JxH)
        if self.descent_method == DescentMethod.LBFGS:
            u = self._lbfgs_direction(F, state)
        elif self.descent_method == DescentMethod.CONJUGATE_GRADIENT:
            u = F
            v = state.v
            v_norm_sq = self.seq.l2_norm_sq(v, 2)
            beta = jnp.where(
                v_norm_sq > 0,
                jnp.maximum(
                    (u @ self.seq.apply_mass_matrix(u - v, 2)) / v_norm_sq,
                    0.0),
                0.0)
            u = u + beta * v
        elif self.descent_method == DescentMethod.GRADIENT:
            u = F
        else:
            raise ValueError(
                f"Unknown descent_method: {self.descent_method}. Supported methods are given by the DescentMethod enum.")
        if self.stochastic:
            u = self.apply_noise(
                u, key, state.noise_level * self.seq.l2_norm(u, 2))
        u = self.apply_regularization(u)
        u, p_v = self.seq.apply_leray_projection(u, k=2, p_guess=state.p_v)

        E_dual = self.seq.cross_product_projection(
            u, H, 1, 2, 1, True, True, self.dirichlet_H)
        E = self.seq.apply_inverse_mass_matrix(E_dual, 1, guess=state.E)
        E = E - state.eta * J

        dB = self.seq.apply_strong_curl(E)
        if self.dt_mode == TimeStepChoice.FIXED or self.dt_mode == TimeStepChoice.PICARD_ADAPTIVE:
            dt = state.dt
        elif self.dt_mode == TimeStepChoice.ANALYTIC_LINESEARCH:
            dt = F @ self.seq.apply_mass_matrix(u, 2) / \
                self.seq.l2_norm_sq(dB, 2)
        else:
            raise ValueError(
                f"Unknown dt_mode: {self.dt_mode}. Supported modes are given by the TimeStepChoice enum.")
        B_nplus1 = B_n + dt * dB

        # update histories: s = iterate difference, y = L^2-gradient difference
        s_new = B_nplus1 - state.B_n
        y_new = state.F_prev - F  # grad_M E = -F, so y = F_old - F_new
        s_history = jnp.roll(state.s_history, 1, axis=0).at[0].set(s_new)
        y_history = jnp.roll(state.y_history, 1, axis=0).at[0].set(y_new)

        return eqx.tree_at(
            lambda s: (s.B_nplus1, s.v, s.p, s.p_v, s.H, s.JxH, s.E,
                       s.F_prev, s.F_norm, s.v_norm, s.dt,
                       s.s_history, s.y_history),
            state,
            (B_nplus1, u, p, p_v, H, JxH, E,
             F, self.seq.l2_norm(F, 2), self.seq.l2_norm(u, 2), dt,
             s_history, y_history))

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
            return jnp.logical_and(
                state.picard_iterations < self.picard_k_restart,
                jnp.logical_or(state.picard_residuum > self.picard_tol,
                               jnp.isnan(state.picard_residuum)))

        def body_fun(state: State) -> State:
            return self.midpoint_residuum(state)

        state = eqx.tree_at(
            lambda s: (s.picard_residuum, s.picard_iterations, s.key),
            state,
            (1.0, 0, key)
        )

        # While loop in the Picard solver.
        state = jax.lax.while_loop(cond_fun, body_fun, state)
        return state

    def relaxation_step(self, state: State, key: jax.random.PRNGKey) -> State:
        if self.timestep_mode == IntegrationScheme.EXPLICIT:
            return self._relaxation_step(state, key)
        elif self.timestep_mode == IntegrationScheme.IMPLICIT_MIDPOINT:
            return self.midpoint_picard_step(state, key)
        else:
            raise ValueError(
                f"Unknown timestep_mode: {self.timestep_mode}. Supported modes are given by the IntegrationScheme enum.")


def initial_state(B_dof: jnp.ndarray, ts: TimeStepper, dt: float = 1.0) -> State:
    n = ts.seq.n2_dbc
    n1 = ts.seq.n1 if not ts.dirichlet_H else ts.seq.n1_dbc
    m = ts.history_size
    return State(
        B_n=B_dof,
        dt=dt,
        v=jnp.zeros(n),
        p=jnp.zeros(ts.seq.n3_dbc),
        p_v=jnp.zeros(ts.seq.n3_dbc),
        H=jnp.zeros(n1),
        JxH=jnp.zeros(n),
        E=jnp.zeros(ts.seq.n1_dbc),
        A=jnp.zeros(ts.seq.n1_dbc),
        F_prev=jnp.zeros(n),
        s_history=jnp.zeros((m, n)),
        y_history=jnp.zeros((m, n)),
    )


def relaxation_loop(B_dof: jnp.ndarray,
                    ts: TimeStepper,
                    num_iters_outer: int,
                    num_iters_inner: int = 100,
                    dt0: float = 1.0,
                    force_tolerance: float = 1e-6,
                    key: jax.random.PRNGKey = jax.random.PRNGKey(67),
                    noise_schedule: Optional[Callable[[int], float]] = None,
                    resistivity_schedule: Optional[Callable[[
                        int], float]] = None,
                    callback: Optional[Callable[[State, int], State]] = None,
                    ) -> tuple[State, dict]:
    """
    Perform multiple relaxation steps for the MRX relaxation.

    The outer loop is a Python for-loop (for diagnostics / callbacks),
    the inner loop is compiled via jax.lax.scan.

    Returns
    -------
    state : State
    traces : dict  with keys: force_norm, helicity, timestep, energy,
             picard_residua, picard_iterations, velocity_norm, divergence_B, eta, iteration
    """
    seq = ts.seq
    state = initial_state(B_dof, ts, dt0)

    def body_fn(state, key):
        state = ts.relaxation_step(state, key)
        failed = (state.picard_residuum > ts.picard_tol) | (
            ~jnp.isfinite(state.picard_residuum))

        def on_fail(s):
            return eqx.tree_at(
                lambda s: (s.dt, s.B_nplus1),
                s, (s.dt / 2, s.B_n))

        def on_success(s):
            s = eqx.tree_at(lambda s: s.B_n, s, s.B_nplus1)
            if ts.dt_mode == TimeStepChoice.PICARD_ADAPTIVE:
                dt_new = jnp.where(
                    s.picard_iterations < ts.picard_k_crit,
                    s.dt * ts.picard_dt_increment,
                    s.dt / ts.picard_dt_increment)
                s = eqx.tree_at(lambda s: s.dt, s, dt_new)
            return s

        state = jax.lax.cond(failed, on_fail, on_success, state)
        return state, None

    @jax.jit
    def _run_scan(state, keys):
        return jax.lax.scan(body_fn, state, keys)

    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])
    get_energy = jax.jit(lambda B: 0.5 * seq.l2_norm_sq(B, 2))
    get_div_norm = jax.jit(compute_divergence_norm, static_argnames=["seq"])

    traces = {k: [] for k in (
        "force_norm", "helicity", "timestep", "energy",
        "picard_residua", "picard_iterations",
        "velocity_norm", "divergence_B", "eta", "iteration")}

    def record(state, iteration):
        traces["force_norm"].append(state.F_norm)
        h, A_new = get_helicity(state.B_n, seq, state.A)
        traces["helicity"].append(h)
        traces["timestep"].append(state.dt)
        traces["energy"].append(get_energy(state.B_n))
        traces["picard_residua"].append(state.picard_residuum)
        traces["picard_iterations"].append(state.picard_iterations)
        traces["velocity_norm"].append(state.v_norm)
        traces["divergence_B"].append(get_div_norm(state.B_n, seq))
        traces["eta"].append(state.eta)
        traces["iteration"].append(iteration)
        return eqx.tree_at(lambda s: s.A, state, A_new)

    F0, p0, _, H0, JxH0 = compute_force(
        state.B_n, seq, dirichlet_H=ts.dirichlet_H, p_guess=state.p)
    state = eqx.tree_at(
        lambda s: (s.F_norm, s.F_prev, s.p, s.H, s.JxH),
        state,
        (seq.l2_norm(F0, 2), F0, p0, H0, JxH0))
    state = record(state, 0)
    print(f"Initial: |F|={state.F_norm:.2e}  "
          f"H={traces['helicity'][-1]:.2e}  "
          f"E={traces['energy'][-1]:.2e}")

    for i in range(1, num_iters_outer + 1):
        key, subkey = jax.random.split(key)
        if noise_schedule is not None:
            state = eqx.tree_at(lambda s: s.noise_level,
                                state, noise_schedule(i))
        if resistivity_schedule is not None:
            state = eqx.tree_at(lambda s: s.eta, state,
                                resistivity_schedule(i))

        state, _ = _run_scan(state, jax.random.split(subkey, num_iters_inner))

        state = record(state, i * num_iters_inner)
        if callback is not None:
            state = callback(state, i)

        print(f"Iter {traces['iteration'][-1]:>6d}: "
              f"|F|={state.F_norm:.2e}  "
              f"dH/H={(traces['helicity'][0] - traces['helicity'][-1]) / (abs(traces['helicity'][0]) + 1e-30):.2e}  "
              f"dt={state.dt:.2e}  "
              f"dE/E={(traces['energy'][0] - traces['energy'][-1]) / (abs(traces['energy'][0]) + 1e-30):.2e}")

        if state.F_norm < force_tolerance:
            break

    return state, traces
