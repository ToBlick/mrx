# %%
from enum import Enum
from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence


def compute_helicity(B: jnp.ndarray, seq: DeRhamSequence, A_guess: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    # The rhs must be the DUAL 1-form D_1^T B, not the primal weak curl.
    #
    # apply_inverse_hodge_laplacian solves the saddle form
    #     | S   D   | | A |   | f |
    #     | D^T -M  | | s | = | 0 |
    # in which f is a dual k-form; apply_leray_projection, solving the same
    # kind of system, correspondingly feeds it apply_derivative_matrix (dual)
    # and not apply_strong_div (primal).  This function used to pass
    # apply_weak_curl(B) = M_1^-1 D_1^T B, i.e. one mass inverse too many, and
    # nothing complained: the solve CONVERGES (measured info = -468, i.e. 468
    # MINRES iterations to tolerance), it just converges to the solution of a
    # different problem.
    #
    # The gate that catches it is an identity, not an error estimate.  In the
    # Dirichlet complex b_2 = 1 (relative cohomology: b_k^rel = b_{3-k}^abs,
    # and a solid torus has b_1^abs = 1), so B_harm is a genuine harmonic
    # remainder and MUST satisfy ||B_harm|| <= ||B||.  Measured on the
    # logical-profile IC at quasr44970 ns=(8,16,8) p=3:
    #
    #     primal rhs (old):  ||B - curl A|| / ||B|| = 8.56e+01   H = +1.99e+01
    #     dual rhs   (new):  ||B - curl A|| / ||B|| = 9.74e-01   H = +1.73e-02
    #
    # 85x is not a fraction of anything.  0.974 is, and it is the right size:
    # the IC is dominated by net toroidal flux, which IS the harmonic mode.
    # See docs/research/handoff_2026-08-25_relaxation_prelim.md.
    A = seq.apply_inverse_hodge_laplacian(
        seq.apply_derivative_matrix(
            B, 1, dirichlet_in=True, dirichlet_out=True, transpose=True),
        1, guess=A_guess)
    B_harm = B - seq.apply_incidence_matrix(
        A, 1, dirichlet_in=True, dirichlet_out=True)
    # <A, B + B_harm>_{L^2} via the 1->2 projection matrix
    helicity = A @ seq.apply_projection_matrix(
        B + B_harm, 2, 1, True, dirichlet_out=True)
    return helicity, A


def compute_divergence_norm(B: jnp.ndarray, seq: DeRhamSequence) -> float:
    # hard-coded dirichlet=True for now
    # Incidence, so this measures the field's divergence and not the
    # mass solver's residual -- see _relaxation_step.
    div_B = seq.apply_incidence_matrix(
        B, 2, dirichlet_in=True, dirichlet_out=True)
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
    # JxH_dual = seq.cross_product_load(J, H, 2, 2, 1, True, True, dirichlet_H)
    J = seq.apply_weak_curl(B, dirichlet_in=True, dirichlet_out=True)
    JxH_dual = seq.cross_product_load(
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
    MF_prev : jnp.ndarray (optional)
        ``M_2 F_prev``.  Carried so that the secant ``M y = M F_prev - M F``
        and the CG beta cost no mass apply of their own.
    s_history : jnp.ndarray (optional)
        History of steps in the descent variable, s_k = dt_k u_k (for L-BFGS).
    y_history : jnp.ndarray (optional)
        History of L^2-gradient differences y_k = grad_M E_{k+1} - grad_M E_k
        = F_k - F_{k+1} (for L-BFGS).  Here grad_M E = -F is the Riesz
        representative of dE w.r.t. the M2 inner product.
    Ms_history, My_history : jnp.ndarray (optional)
        ``M_2 s_k`` and ``M_2 y_k``, row-aligned with the histories above.
        Every M-inner product the two-loop recursion takes is against a
        stored vector, so with these in hand it applies M zero times.
    picard_iterations : int
        The number of Picard iterations.
    picard_residuum : float
        The residuum of the Picard solver.
    F_norm : float
        The norm of the force.
    v_norm : float
        The norm of the velocity.
    lbfgs_sy : float
        The curvature <s_k, y_k>_M of the NEWEST L-BFGS pair, as it was
        actually used by the two-loop recursion.  Reported, never clamped: a
        negative value means the stored pair is not a descent pair and the
        approximate inverse Hessian it builds is indefinite.
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
    MF_prev: Optional[jnp.ndarray] = None
    s_history: Optional[jnp.ndarray] = None
    y_history: Optional[jnp.ndarray] = None
    Ms_history: Optional[jnp.ndarray] = None
    My_history: Optional[jnp.ndarray] = None
    A: Optional[jnp.ndarray] = None
    dt: float = 1e-2
    eta: float = 0.0
    picard_iterations: int = 0
    picard_residuum: float = 0.0
    F_norm: float = 0.0
    v_norm: float = 0.0
    lbfgs_sy: float = 0.0
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

    def _lbfgs_direction(self, F: jnp.ndarray, s: jnp.ndarray, y: jnp.ndarray,
                         Ms: jnp.ndarray, My: jnp.ndarray
                         ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the L-BFGS descent direction v = H_k F using the two-loop recursion.

        The L^2 (M2) inner product is used both to identify the gradient via
        the Riesz map (dℰ[v] = -(F, v)_{L^2} = <-F, v>_M  =>  grad_M E = -F)
        and inside the two-loop recursion (<a, b>_M = a^T M b).

        NOTE that ``grad_M E = -F`` is a derivative with respect to the
        VELOCITY u, not with respect to B: the admissible variation is
        ``dB = curl(u x H)``, and ``dE[dB] = -(F, u)_M``.  So both members of
        an L-BFGS pair have to live in velocity space; see
        ``_relaxation_step`` for how that is arranged and what happens when
        it is not.

        Parameters
        ----------
        F : the force / negative gradient at the current iterate.
        s : (m, n) newest-first step history in the descent variable.
        y : (m, n) newest-first gradient-difference history, ALIGNED with s.
        Ms, My : (m, n) ``M s`` and ``M y``, row by row.  Every inner product
            the recursion takes is against a STORED vector, and M is
            symmetric, so ``<s_i, q>_M = (M s_i)^T q`` and
            ``<y_i, r>_M = (M y_i)^T r``: this function applies M zero times.
            (It used to apply it 4m + 2 times per step, re-forming ``M y_i``
            in both loops and ``M q``, ``M r`` once per pair.)

        Returns
        -------
        r : the direction ``H_k F``.
        sy0 : ``<s_0, y_0>_M``, the curvature of the newest pair -- returned
            so the caller can record it.  It is NOT used to reject the pair.

        Falls back to steepest descent (F) when all history entries are zero.
        """
        m = self.history_size

        # <s_i, y_i>_M for every stored pair.  An EMPTY slot -- s_i identically
        # zero, as the history is before it fills -- has sy_i = 0 exactly and
        # contributes nothing to either loop; rho_i = 0 states that.  (The
        # old ``1 / (sy + 1e-30)`` gave the same result only because
        # 1e30 * 0 happens to be 0.)  A NEGATIVE sy_i is a non-descent pair;
        # it is used as stored and reported through ``sy`` below, not
        # repaired here.
        sy_all = jnp.einsum('in,in->i', s, My)
        filled = jnp.any(s != 0, axis=1)
        rho = jnp.where(filled, 1.0 / jnp.where(filled, sy_all, 1.0), 0.0)

        # --- two-loop recursion ---
        # first loop: newest (i=0) to oldest (i=m-1)
        q = F
        alpha = []
        for i in range(m):
            alpha_i = rho[i] * (Ms[i] @ q)
            alpha.append(alpha_i)
            q = q - alpha_i * y[i]

        # Initial Hessian scaling: gamma = (s_0^T M y_0) / (y_0^T M y_0).
        #
        # The branch is on `sy > 0`, i.e. on whether a USABLE curvature pair
        # exists at all, and NOT on `yy > 1e-30`.  On the first step the
        # history is zero, so sy = 0 exactly and the recursion must fall back
        # to steepest descent (gamma = 1, r = q = F).  Keying on yy instead
        # gets that wrong: y_0 = F_prev - F is not exactly zero even on the
        # first step, because F_prev comes from a separate compute_force call
        # whose warm-started solves land at tolerance rather than bit-identity
        # (yy ~ 1e-24).  The yy branch is then not taken, gamma = 0/yy = 0,
        # the direction is exactly zero, dB is exactly zero, and the
        # linesearch dt = (F,u)/||dB||^2 is 0/0 = NaN on step one.  Measured.
        #
        # There is deliberately NO `maximum(gamma, 1e-30)` floor.  Flooring
        # does not repair negative curvature: it silently annihilates q, i.e.
        # the gradient's entire contribution to the direction, leaving only
        # the stored-s combination from the second loop.  That is how this
        # failed for months -- the returned "descent direction" came out
        # orthogonal to F, ||dB|| collapsed to solver noise, and dt exploded
        # to compensate.  sy is returned so the caller records it instead.
        sy = sy_all[0]
        yy = y[0] @ My[0]
        gamma = jnp.where(sy > 0, sy / jnp.where(yy > 0, yy, 1.0), 1.0)
        r = gamma * q

        # second loop: oldest (i=m-1) to newest (i=0)
        for i in range(m - 1, -1, -1):
            beta_i = rho[i] * (My[i] @ r)
            r = r + (alpha[i] - beta_i) * s[i]

        return r, sy

    def apply_regularization(self, u: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.gamma):
            rhs = self.seq.apply_mass_matrix(u, 2, True)
            u = self.seq.apply_inverse_mass_plus_eps_laplace_matrix(
                rhs, 2, self.mu, dirichlet=True, guess=u)
        return u

    def update_field(self, state: State, field_name: Literal['B_n', 'B_nplus1', 'v', 'p_v', 'H', 'JxH', 'E', 's_history', 'y_history', 'F_prev', 'MF_prev', 'Ms_history', 'My_history', 'A', 'dt', 'eta', 'picard_iterations', 'picard_residuum', 'F_norm', 'v_norm', 'lbfgs_sy', 'noise_level', 'key'], value) -> State:  # noqa: E501
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
        # M F ONCE.  It serves ||F||_M, the CG beta and the L-BFGS secant
        # M y = M F_prev - M F; the step applies M_2 three times in total
        # (M F, M u, M dB) whichever method is running -- L-BFGS used to
        # apply it 4m + 6 times.
        MF = self.seq.apply_mass_matrix(F, 2)

        # --- history bookkeeping, part 1: push y BEFORE the direction -------
        # y_{k-1} = grad_M E_k - grad_M E_{k-1} = F_prev - F is a difference
        # over the step that ALREADY happened, so it pairs with s_{k-1}, which
        # the end of the previous step put in s_history[0].  Pushing it HERE,
        # rather than next to the brand-new s_k at the end of this step, is
        # what keeps (s_i, y_i) aligned.  Pushing it at the end instead leaves
        # y lagging its paired s by exactly one step.
        y_new = state.F_prev - F
        My_new = state.MF_prev - MF
        s_hist = state.s_history
        Ms_hist = state.Ms_history
        y_hist = jnp.roll(state.y_history, 1, axis=0).at[0].set(y_new)
        My_hist = jnp.roll(state.My_history, 1, axis=0).at[0].set(My_new)

        sy = jnp.array(0.0)
        if self.descent_method == DescentMethod.LBFGS:
            u, sy = self._lbfgs_direction(F, s_hist, y_hist, Ms_hist, My_hist)
        elif self.descent_method == DescentMethod.CONJUGATE_GRADIENT:
            u = F
            # Polak-Ribiere.  The previous GRADIENT is -F_prev, which is why
            # F_prev and not the previous search DIRECTION v belongs here;
            # the two coincide only on the first step, where beta is 0 anyway.
            # <F, F - F_prev>_M = F^T M F - F^T (M F_prev) by symmetry of M.
            g_prev_norm_sq = state.F_prev @ state.MF_prev
            beta = jnp.where(
                g_prev_norm_sq > 0,
                jnp.maximum(
                    (F @ MF - F @ state.MF_prev) / g_prev_norm_sq, 0.0),
                0.0)
            u = u + beta * state.v
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
        # M u once: the linesearch numerator, ||u||_M and the stored M s.
        Mu = self.seq.apply_mass_matrix(u, 2)

        E_dual = self.seq.cross_product_load(
            u, H, 1, 2, 1, True, True, self.dirichlet_H)
        E = self.seq.apply_inverse_mass_matrix(E_dual, 1, guess=state.E)
        E = E - state.eta * J

        # The TOPOLOGICAL curl, not M_2^-1 D_1.  Three reasons, all measured
        # on quasr44970 ns=(8,16,8) p=3:
        #   * div . curl is 8.6e-16 this way against 1.3e-10 for the
        #     mass-projected form, so div B is conserved EXACTLY along the
        #     trajectory instead of to the mass solver's tolerance;
        #   * it is matrix-free, so it removes one Krylov solve per step from
        #     the hot path;
        #   * the two curls agree to 1.0e-12, so the swap does not move the
        #     trajectory -- it only removes an error that had no business
        #     being there.
        # The Gram correction inside `mrx.operators.apply_incidence_matrix`
        # -- G = Gram_{k+1}^-1 (E_out^T sp E_in) -- is what makes the incidence
        # form exact at the polar axis. (This comment used to warn that
        # `DeRhamSequence.apply_incidence_matrix`'s docstring still recommended
        # the mass-projected form; that docstring was corrected on 2026-08-25,
        # so the warning described a contradiction that no longer existed. It
        # also cited a bare line number, which had drifted by three within a
        # day -- cite the SYMBOL, not the line.)
        dB = self.seq.apply_incidence_matrix(
            E, 1, dirichlet_in=True, dirichlet_out=True)
        if self.dt_mode == TimeStepChoice.FIXED or self.dt_mode == TimeStepChoice.PICARD_ADAPTIVE:
            dt = state.dt
        elif self.dt_mode == TimeStepChoice.ANALYTIC_LINESEARCH:
            dt = F @ Mu / self.seq.l2_norm_sq(dB, 2)
        else:
            raise ValueError(
                f"Unknown dt_mode: {self.dt_mode}. Supported modes are given by the TimeStepChoice enum.")
        B_nplus1 = B_n + dt * dB

        # --- history bookkeeping, part 2: push the step just taken ----------
        # The descent variable is the VELOCITY u, not B: grad_M E = -F is the
        # derivative of E with respect to u (dE = -(F, u)_M), and the direction
        # the recursion returns is consumed as a velocity.  So the step in the
        # descent variable is dt*u.  B moves by dt*curl(u x H) instead, which
        # is a different vector in a different space; storing THAT as s pairs
        # it with a y that is a secant of a different map entirely, and the
        # curvature <s, y>_M goes negative on a third to a half of all steps.
        # See docs/research/handoff_2026-08-25_relaxation_prelim.md.
        s_new = dt * u
        s_history = jnp.roll(s_hist, 1, axis=0).at[0].set(s_new)
        Ms_history = jnp.roll(Ms_hist, 1, axis=0).at[0].set(dt * Mu)
        y_history = y_hist
        My_history = My_hist

        return eqx.tree_at(
            lambda s: (s.B_nplus1, s.v, s.p, s.p_v, s.H, s.JxH, s.E,
                       s.F_prev, s.MF_prev, s.F_norm, s.v_norm, s.lbfgs_sy,
                       s.dt, s.s_history, s.y_history, s.Ms_history,
                       s.My_history),
            state,
            (B_nplus1, u, p, p_v, H, JxH, E,
             F, MF, jnp.sqrt(F @ MF), jnp.sqrt(u @ Mu), sy,
             dt, s_history, y_history, Ms_history, My_history))

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
    """Build the state at ``B_dof`` with its force already evaluated.

    ``F_prev``, ``MF_prev``, ``F_norm`` and the warm-start guesses ``p``,
    ``H``, ``JxH`` are seeded from one ``compute_force`` here, so the first
    step's secant ``y = F_prev - F`` and CG beta see the true previous
    gradient.  Callers used to repeat this seeding by hand (and could not
    have seeded ``MF_prev``, which did not exist).
    """
    seq = ts.seq
    n = seq.n2_dbc
    m = ts.history_size
    F0, p0, _, H0, JxH0 = compute_force(B_dof, seq, dirichlet_H=ts.dirichlet_H)
    MF0 = seq.apply_mass_matrix(F0, 2)
    return State(
        B_n=B_dof,
        dt=dt,
        v=jnp.zeros(n),
        p=p0,
        p_v=jnp.zeros(seq.n3_dbc),
        H=H0,
        JxH=JxH0,
        E=jnp.zeros(seq.n1_dbc),
        A=jnp.zeros(seq.n1_dbc),
        F_prev=F0,
        MF_prev=MF0,
        F_norm=jnp.sqrt(F0 @ MF0),
        s_history=jnp.zeros((m, n)),
        y_history=jnp.zeros((m, n)),
        Ms_history=jnp.zeros((m, n)),
        My_history=jnp.zeros((m, n)),
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


# ---------------------------------------------------------------------------
# Trace dictionary helpers (legacy relaxation loop)
# ---------------------------------------------------------------------------

default_trace_dict = {
    "iterations": [],
    "force_trace": [],
    "energy_trace": [],
    "helicity_trace": [],
    "divergence_trace": [],
    "picard_iterations": [],
    "picard_errors": [],
    "timesteps": [],
    "velocity_trace": [],
    "wall_time_trace": [],
    "B_fields": [],
    "p_fields": [],
    "start_time": None,
    "end_time": None,
}


