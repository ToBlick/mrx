# %%
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
        # Assuming that this was suppose to be P2x1_to_1?
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
        Compute the magnetic field energy.

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
        Compute the magnetic helicity.

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
    B_guess : jnp.ndarray
        The magnetic field at the previous time step.
    dt : float
        The time step.
    eta : float
        The resistivity.
    Hessian : jnp.ndarray
        The Hessian of the MRX relaxation.
    picard_iterations : int
        The number of Picard iterations.
    picard_residuum : float
        The residuum of the Picard solver.
    force_norm : float
        The norm of the force.
    velocity_norm : float
        The norm of the velocity.
    """
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

    def __init__(
            self, Seq: DeRhamSequence, gamma: int = 0, newton: bool = False,
            picard_tol: float = 1e-12, picard_maxit: int = 20, force_free: bool = False):
        """
        Initialize the TimeStepper class.

        Parameters
        ----------
        Seq : DeRham sequence
            The de Rham sequence to use.
        gamma : int, default=0
            The regularization parameter.
        newton : bool, default=False
            Whether to use Newton's method.
        picard_tol : float, default=1e-12
            The tolerance for the Picard solver.
        picard_maxit : int, default=20
            The maximum number of iterations for the Picard solver.
        force_free : bool, default=False
            Whether the problem has grad(p) = 0.
        """
        self.Seq = Seq
        self.gamma = gamma
        self.newton = newton
        self.id2 = jnp.eye(Seq.M2.shape[0])
        self.picard_tol = picard_tol
        self.picard_maxit = picard_maxit
        self.force_free = force_free

    def init_state(self, B: jnp.ndarray, dt: float, eta: float, Hessian: jnp.ndarray) -> State:
        """
        Initialize the state of the MRX relaxation.

        Parameters
        ----------
        B : jnp.ndarray
            The magnetic field.
        dt : float
            The time step.
        eta : float
            The resistivity.
        Hessian : jnp.ndarray
            The Hessian of the MRX relaxation.

        Returns
        -------
        state : State
            The state of the MRX relaxation.
        """
        return self.State(B, B, dt, eta, Hessian)

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
        return (B @ self.Seq.M2 @ B)**0.5

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

    def update_dt(self, state: State, dt: float) -> State:
        """
        Update the time step.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        dt : float
            The new time step.

        Returns
        -------
        state : State
            The updated state of the MRX relaxation.
        """
        return eqx.tree_at(
            lambda s: s.dt,
            state,
            dt
        )

    def update_hessian(self, state: State, Hessian: jnp.ndarray) -> State:
        """
        Update the Hessian of the MRX relaxation.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        Hessian : jnp.ndarray
            The new Hessian.

        Returns
        -------
        state : State
            The updated state of the MRX relaxation.
        """
        return eqx.tree_at(
            lambda s: s.Hessian,
            state,
            Hessian
        )

    def update_B_n(self, state: State, B: jnp.ndarray) -> State:
        """
        Update the magnetic field at the current time step.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        B : jnp.ndarray
            The new magnetic field.

        Returns
        -------
        state : State
            The updated state of the MRX relaxation.
        """
        return eqx.tree_at(
            lambda s: s.B_n,
            state,
            B
        )

    def update_B_guess(self, state: State, B: jnp.ndarray) -> State:
        """
        Update the magnetic field at the previous time step.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.
        B : jnp.ndarray
            The new magnetic field.

        Returns
        -------
        state : State
            The updated state of the MRX relaxation.
        """
        return eqx.tree_at(
            lambda s: s.B_guess,
            state,
            B
        )

    def picard_solver(self, state: State) -> State:
        """
        Picard solver for the MRX relaxation.

        Parameters
        ----------
        state : State
            The state of the MRX relaxation.

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
            return jnp.logical_and(state.picard_iterations < self.picard_maxit,
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

        # initialize residuum and picard_iterations
        state = eqx.tree_at(
            lambda s: (s.picard_residuum, s.picard_iterations),
            state,
            (1.0, 0)
        )

        # Finally, perform the while loop in the Picard solver.
        state = jax.lax.while_loop(cond_fun, body_fun, state)
        return state
