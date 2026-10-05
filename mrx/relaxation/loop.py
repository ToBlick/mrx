"""The relaxation loop, which lowers the magnetic energy of ``B`` towards an MHD equilibrium.

Each step moves the field as ``B <- B + dt curl(u x B)``. This keeps ``div B = 0`` and the helicity fixed, so
only the energy changes. The velocity ``u`` is either a Newton direction or a smoothed Lorentz force.

A typical run builds a :class:`TimeStepper` for a sequence, makes the start state with :func:`initial_state`
and calls :func:`relax`, which runs the steps in compiled chunks and prints diagnostics after every chunk.
:func:`write_checkpoint` and :func:`read_checkpoint` save and restore the state for a restart.

With ``compressible=True`` the velocity is not projected onto divergence-free fields. A prescribed pressure ``p``
then moves with the flow, ``p <- p - dt u . grad p``, and the step lowers ``int B^2/2 - p dV`` instead of the
magnetic energy (see :mod:`mrx.relaxation.newton`). Its fixed point is ``J x B = grad p`` with that pressure.

The sequence and the stepper's switches (Newton or not, compressible or not, resistive or not) are compiled in. Changing them
recompiles. A new geometry or a new field on the same sequence does not.
"""
from typing import Callable, NamedTuple, Optional

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.relaxation.newton import (NEWTON_MAXITER, NEWTON_PENALTY, NEWTON_TOL, compressible_newton_direction,
                                   newton_direction)
from mrx.relaxation.physics import (advection, compute_divergence_norm, compute_force, compute_helicity,
                                   parallel_smoothing, pressure_gradient, pressure_integral, resistive_step,
                                   weak_pressure, beta_vol)
from mrx.precision import DTYPE, RESIDUAL_DTYPE


def knot_spacing(seq: DeRhamSequence) -> np.ndarray:
    """The smallest knot spacing of each logical direction, shape ``(3,)``."""
    h = []
    for b in seq.basis_0.bases[0].bases:
        knots = np.asarray(b.T)
        interior = knots[b.p:-b.p] if b.type in ('clamped', 'periodic') else knots
        h.append(np.diff(interior).min())
    return np.array(h)


def radial_cell_sq(seq: DeRhamSequence) -> jnp.ndarray:
    """Return the squared physical size of a radial cell, ``h_r^2 = <g_rr>_V dr^2``.

    ``dr`` is the smallest radial knot spacing and ``<g_rr>_V`` the volume average of the radial metric
    coefficient. It is the length unit of the velocity smoothing and of the resistive doses.
    """
    wJ = seq.quad.w * seq.jacobian_j
    return knot_spacing(seq)[0] ** 2 * jnp.sum(wJ * seq.metric_jkl[:, 0, 0]) / jnp.sum(wJ)


def toroidal_cell_sq(seq: DeRhamSequence) -> jnp.ndarray:
    """Return the squared physical size of a toroidal cell, ``h_zeta^2 = <g_zz>_V dzeta^2``, the length unit of
    the parallel smoothing of an advected pressure (field lines run mostly along ``zeta``)."""
    wJ = seq.quad.w * seq.jacobian_j
    return knot_spacing(seq)[2] ** 2 * jnp.sum(wJ * seq.metric_jkl[:, 2, 2]) / jnp.sum(wJ)


def logical_cfl_weights(seq: DeRhamSequence) -> jnp.ndarray:
    """Return the weights ``1 / (J h_i)``, shape ``(n_q, 3)``, with ``J`` the Jacobian determinant and ``h_i``
    the knot spacing of direction ``i``.

    Multiplied with the components of a 2-form velocity at the quadrature points they give the number of
    logical cells crossed per unit time, the CFL number of each direction.
    """
    h = jnp.asarray(knot_spacing(seq), dtype=DTYPE)
    weights = 1.0 / (seq.jacobian_j[:, None] * h[None, :])
    # the theta cell degenerates in the first radial span
    return weights.at[:, 1].multiply(seq.quad.x[:, 0] >= h[0])


class WarmStarts(eqx.Module):
    """The solutions of the last step's linear solves, kept as starting guesses for the next step.

    ``p`` is the pressure, ``JxB`` the Lorentz force before projection, ``J`` the current, ``E = u x B`` the
    electric field, ``a`` the vector potential of the step's direction, ``a_F`` the vector potential of the
    force (the same as ``a`` for gradient descent), ``A`` the vector potential of the helicity (updated only
    when diagnostics are sampled), ``resistive_delta`` the last resistive increment and ``dp`` the last rate of
    change of the advected pressure.
    """
    p: jnp.ndarray
    JxB: jnp.ndarray
    J: jnp.ndarray
    E: jnp.ndarray
    a: jnp.ndarray
    a_F: jnp.ndarray
    A: jnp.ndarray
    resistive_delta: jnp.ndarray
    dp: jnp.ndarray


class LastStep(eqx.Module):
    """What the last step computed: the force ``F`` at the start of the step and its norm, the velocity ``v``
    and its norm, the iteration counts of the Newton and resistive solves (positive when converged, negative
    when not), the relative size ``||delta|| / ||B||`` of the resistive increment and the iteration count of the
    parallel smoothing of the advected pressure."""
    F: jnp.ndarray
    F_norm: jnp.ndarray
    v: jnp.ndarray
    v_norm: jnp.ndarray
    newton_it: jnp.ndarray
    resistive_it: jnp.ndarray
    resistive_moved: jnp.ndarray
    smoothing_it: jnp.ndarray


class BestState(eqx.Module):
    """The field (and advected pressure) with the lowest force residual seen so far, that residual and the step
    it was reached at."""
    B: jnp.ndarray
    p: jnp.ndarray
    resid: jnp.ndarray
    step: jnp.ndarray


class State(eqx.Module):
    """Everything the relaxation carries from one step to the next. Make one with :func:`initial_state`.

    ``B_n`` is the current field and ``p_n`` the advected pressure (a free 0-form, zero and unused unless the
    stepper is compressible). ``dt`` is the step length taken, ``min(dt_star, CFL / cfl_max)``, where
    ``dt_star`` minimises the energy along the step and ``cfl_max`` is the largest CFL number of the velocity.
    All leaves are arrays, so the state can be passed through compiled code.
    """
    B_n: jnp.ndarray
    B_nplus1: jnp.ndarray
    p_n: jnp.ndarray
    p_nplus1: jnp.ndarray
    dt: jnp.ndarray
    dt_star: jnp.ndarray
    cfl_max: jnp.ndarray
    warm: WarmStarts
    last: LastStep
    best: BestState


class Increment(NamedTuple):
    """The ideal increment ``dB = curl(u x B)`` at one field and ``dp = -u . grad p`` of the advected pressure,
    with the intermediate results of the step."""
    dB: jnp.ndarray
    dp: jnp.ndarray
    u: jnp.ndarray
    Mu: jnp.ndarray
    F: jnp.ndarray
    MF: jnp.ndarray
    p: jnp.ndarray
    JxB: jnp.ndarray
    J: jnp.ndarray
    E: jnp.ndarray
    cfl_max: jnp.ndarray
    a: jnp.ndarray
    a_F: jnp.ndarray
    newton_it: jnp.ndarray


#: The step cap in logical cells: ``dt = min(dt_star, CFL / cfl_max)``.
CFL = 0.5
#: The velocity smoothing scale in squared radial cells, ``mu = SMOOTHING_C h_r^2`` (:func:`radial_cell_sq`).
#: It damps the shortest radial mode (two cells) by ``1 / (1 + SMOOTHING_C pi^2)`` on any mesh.
SMOOTHING_C = 0.075
#: The parallel smoothing of an advected pressure in squared toroidal cells, ``eps = PRESSURE_SMOOTHING h_zeta^2``
#: (:func:`toroidal_cell_sq`). At 1 the variation of p along the field still grows on li383 and W7-X, at 10 it holds.
PRESSURE_SMOOTHING = 10.0


class TimeStepper(eqx.Module):
    """One step of the energy descent, ``B_{n+1} = B_n + dt curl(u x B)`` (forward Euler).

    The force is ``F = curl a + c h``, the divergence-free part of ``J x B`` computed from a vector potential
    ``a`` (one 1-form Laplacian solve) and the harmonic 2-forms ``h``, with no pressure solve.
    With ``newton=True`` the velocity ``u`` is the Newton direction of
    :func:`mrx.relaxation.newton.newton_direction`, controlled by ``newton_penalty``, ``newton_tol`` and
    ``newton_maxiter``. Otherwise ``u`` is the smoothed force ``(M_2 + mu L_2)^-1 M_2 F`` with
    ``mu = SMOOTHING_C h_r^2``, where ``M_2`` is the 2-form mass matrix and ``L_2`` the 2-form Laplacian. The
    step length ``dt`` minimises the energy along the increment, capped by the CFL limit and, for Newton, by 1.

    With ``compressible=True`` the force is ``F = M_2^-1 load(J x B - grad p)`` with the advected pressure ``p``,
    not projected, and the velocity is its smoothed version or the Newton direction of
    :func:`mrx.relaxation.newton.compressible_newton_direction`. ``p`` moves with the same step,
    ``p_{n+1} = p_n - dt u . grad p_n``. A nonzero ``pressure_smoothing`` ``C`` then diffuses ``p`` along the new
    field by :func:`~mrx.relaxation.physics.parallel_smoothing` with ``eps = C h_zeta^2``
    (:func:`toroidal_cell_sq`), which removes the variation along field lines that the discrete advection creates
    and keeps ``int p dV``.

    A nonzero ``resistivity`` (the dose ``eta dt`` per step, a length squared) adds a
    :func:`~mrx.relaxation.physics.resistive_step` towards ``resistive_reference`` after every ideal step.
    The remaining fields are computed from the sequence at construction and should not be passed.
    """
    seq: DeRhamSequence
    newton: bool = False
    newton_penalty: float = NEWTON_PENALTY
    newton_tol: float = NEWTON_TOL
    newton_maxiter: int = NEWTON_MAXITER
    compressible: bool = False
    pressure_smoothing: float = PRESSURE_SMOOTHING
    resistivity: float = 0.0
    resistive_reference: Optional[jnp.ndarray] = None
    velocity_smoothing_scale: float = None
    cfl_weights: jnp.ndarray = None
    harmonic: jnp.ndarray = None
    harmonic_norm_sq: jnp.ndarray = None
    resistive: bool = eqx.field(static=True, default=False)
    smooth_pressure: bool = eqx.field(static=True, default=False)

    def __post_init__(self):
        # with stellarator symmetry the harmonic form is odd and the force even, so this is empty
        even = self.seq.even
        hs = even.nullspace(2)
        self.harmonic = hs
        self.harmonic_norm_sq = jnp.asarray([h @ (even.M[2] @ h) for h in hs], dtype=hs.dtype)
        # device scalars, not Python floats: a new geometry's h_r must not recompile the chunk
        self.velocity_smoothing_scale = SMOOTHING_C * radial_cell_sq(self.seq)
        self.resistive = bool(self.resistivity != 0)
        self.resistivity = jnp.asarray(self.resistivity, dtype=DTYPE)
        self.smooth_pressure = bool(self.compressible and self.pressure_smoothing != 0)
        self.pressure_smoothing = jnp.asarray(self.pressure_smoothing * toroidal_cell_sq(self.seq), dtype=DTYPE)
        self.cfl_weights = logical_cfl_weights(self.seq)

    def _lorentz(self, B: jnp.ndarray, J_guess: jnp.ndarray):
        """Return ``(J, (J x B)_dual)``: the current and the Lorentz force as a dual 2-form (tested against the
        2-form basis), not projected."""
        odd, even = self.seq.odd, self.seq.even
        J = odd.weak_curl(B, guess=J_guess)
        return J, even.cross_product_load_values(odd.evaluate_at_quadrature(J, 1),
                                                 odd.evaluate_at_quadrature(B, 2), 2, 1, 2)

    def _potential_force(self, B: jnp.ndarray, a_guess: jnp.ndarray, J_guess: jnp.ndarray, smooth: bool = True):
        """Return ``(F, M F, F_s, J, a)``: the divergence-free force ``F = curl a + c h`` and, with ``smooth``,
        its smoothed version ``F_s`` (else ``None``).

        ``a`` is a vector potential of the force and ``c h`` its harmonic part. Smoothing ``a`` with
        ``(M_1 + mu L_1)^-1 M_1`` and taking the curl equals smoothing ``curl a`` with ``(M_2 + mu L_2)^-1 M_2``.
        """
        seq = self.seq.even
        J, JxB_dual = self._lorentz(B, J_guess)
        a = seq.L[1].solve(seq.G[1].T @ JxB_dual, guess=a_guess)
        ch = self.harmonic.T @ ((self.harmonic @ JxB_dual) / self.harmonic_norm_sq)   # zero on the even view
        F = seq.G[1] @ a + ch
        if not smooth:
            return F, seq.M[2] @ F, None, J, a
        a_s = seq.shifted(1, self.velocity_smoothing_scale).solve(seq.M[1] @ a, guess=a)
        Fs = seq.G[1] @ a_s + ch
        return F, seq.M[2] @ F, Fs, J, a

    def _compressible_force(self, B: jnp.ndarray, p_adv: jnp.ndarray, J_guess: jnp.ndarray,
                            F_guess: jnp.ndarray):
        """Return ``(F, M F, J, grad p)``: the force ``J x B - grad p`` of the advected pressure ``p_adv``, not
        projected, as a 2-form and as a dual 2-form, the current and the quadrature values of ``grad p``."""
        even = self.seq.even
        J, JxB_dual = self._lorentz(B, J_guess)
        grad_p = pressure_gradient(p_adv, self.seq)
        MF = JxB_dual - even.vector_load_values(grad_p, 1, 2)
        return even.M[2].solve(MF, guess=F_guess), MF, J, grad_p

    def _ideal_increment(self, B: jnp.ndarray, p_adv: jnp.ndarray, state: State) -> Increment:
        """The ideal increment at ``(B, p_adv)``, every Krylov solve warm-started from ``state``."""
        seq = self.seq
        odd, even = seq.odd, seq.even
        w = state.warm
        newton_it = jnp.int32(0)
        p, JxB = w.p, w.JxB                                           # no pressure solve on either route
        if self.compressible:
            F, MF, J, grad_p = self._compressible_force(B, p_adv, w.J, state.last.F)
            if self.newton:
                u, newton_it = compressible_newton_direction(seq, B, J, p_adv, MF, state.last.v,
                                                             self.newton_penalty, self.newton_tol,
                                                             self.newton_maxiter)
            else:
                u = even.shifted(2, self.velocity_smoothing_scale).solve(MF, guess=state.last.v)
            a, a_F = w.a, w.a_F
        elif self.newton:
            # the Newton system and dt* see the force only through curl^T, which annihilates the gradient and
            # harmonic parts exactly, so they take the unprojected load. F is formed for the residual only.
            J, JxB_dual = self._lorentz(B, w.J)
            u, a, newton_it = newton_direction(seq, B, J, JxB_dual, w.a, self.newton_penalty, self.newton_tol,
                                               self.newton_maxiter)
            F, MF, _, _, a_F = self._potential_force(B, w.a_F, J, smooth=False)
        else:
            F, MF, u, J, a = self._potential_force(B, w.a, w.J)       # u the smoothed force
            a_F = a
        Mu = even.M[2] @ u
        u_jk = even.evaluate_at_quadrature(u, 2)
        E_dual = odd.cross_product_load_values(u_jk, odd.evaluate_at_quadrature(B, 2), 1, 2, 2)
        E = odd.M[1].solve(E_dual, guess=w.E)
        cfl_max = jnp.max(jnp.abs(u_jk) * self.cfl_weights)
        # the topological curl: div B is conserved exactly, no mass solve
        dB = odd.G[1] @ E
        dp = advection(grad_p, u_jk, seq, guess=w.dp) if self.compressible else w.dp
        return Increment(dB, dp, u, Mu, F, MF, p, JxB, J, E, cfl_max, a, a_F, newton_it)

    def _step_size(self, inc: Increment) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(dt, dt_star)``. ``dt_star = <F, u> / ||dB||^2`` minimises the energy along the increment (in
        the compressible case ``int B^2/2 - p dV``, whose pressure part is linear in ``dt``) and
        ``dt`` is ``dt_star`` after the caps."""
        dt_star = (inc.F @ inc.Mu) / self.seq.odd.l2_norm_sq(inc.dB, 2)
        # a non-positive dt* is no step: a negative one would climb the energy
        dt = jnp.minimum(jnp.maximum(dt_star, 0.0), CFL / inc.cfl_max)
        if self.newton:
            dt = jnp.minimum(dt, 1.0)              # the Newton length
        return dt, dt_star

    def relaxation_step(self, state: State) -> State:
        """Advance ``state.B_n`` by one step into ``state.B_nplus1``."""
        B_n = state.B_n
        inc = self._ideal_increment(B_n, state.p_n, state)
        dt, dt_star = self._step_size(inc)
        B_nplus1 = B_n + dt * inc.dB
        p_nplus1 = state.p_n + dt * inc.dp if self.compressible else state.p_n
        smoothing_it = state.last.smoothing_it
        if self.smooth_pressure:
            p_nplus1, smoothing_it = parallel_smoothing(p_nplus1, B_nplus1, self.seq, self.pressure_smoothing)

        res_delta, res_it, res_moved = state.warm.resistive_delta, state.last.resistive_it, state.last.resistive_moved
        if self.resistive:
            B_ideal = B_nplus1
            B_nplus1, res_it, res_moved = resistive_step(B_ideal, self.seq, self.resistivity,
                                                         self.resistive_reference, guess=res_delta)
            res_delta = B_nplus1 - B_ideal
            res_moved = res_moved.astype(state.last.resistive_moved.dtype)

        warm = WarmStarts(p=inc.p, JxB=inc.JxB, J=inc.J, E=inc.E, a=inc.a, a_F=inc.a_F, A=state.warm.A,
                          resistive_delta=res_delta, dp=inc.dp)
        last = LastStep(F=inc.F, F_norm=jnp.sqrt(inc.F @ inc.MF), v=inc.u, v_norm=jnp.sqrt(inc.u @ inc.Mu),
                        newton_it=inc.newton_it, resistive_it=res_it, resistive_moved=res_moved,
                        smoothing_it=jnp.asarray(smoothing_it, jnp.int32))
        return eqx.tree_at(lambda s: (s.B_nplus1, s.p_nplus1, s.dt, s.dt_star, s.cfl_max, s.warm, s.last), state,
                           (B_nplus1, p_nplus1, dt, dt_star, inc.cfl_max, warm, last))


def initial_state(B_dof: jnp.ndarray, ts: TimeStepper, step: int = 0, p: Optional[jnp.ndarray] = None) -> State:
    """Return the start state for the field ``B_dof`` and, for a compressible stepper, the advected pressure ``p``
    (see :func:`mrx.relaxation.physics.initial_pressure`). ``step`` is the step number the field is at (nonzero
    for a restart). This evaluates the force once."""
    seq = ts.seq
    odd, even = seq.odd, seq.even
    if ts.compressible == (p is None):
        raise ValueError("a compressible stepper needs the advected pressure p, and only it takes one")
    p = jnp.zeros(even.free.n(0), dtype=DTYPE) if p is None else p.astype(DTYPE)
    F0, MF0, _, J0, a_F0 = ts._potential_force(B_dof, jnp.zeros(even.n(1), dtype=DTYPE), None, smooth=False)
    if ts.compressible:
        F0, MF0, J0, _ = ts._compressible_force(B_dof, p, None, None)
    # the pressure warm starts only feed the sampler's diagnostic Leray solve
    p0, JxB0 = jnp.zeros(even.n(3), dtype=DTYPE), jnp.zeros(even.n(2), dtype=DTYPE)
    resid0 = (jnp.sqrt(F0 @ MF0) / force_scale_jit(seq, B_dof)) ** 2
    zero = jnp.zeros((), dtype=DTYPE)       # scalars as dtype arrays: one carry signature for the scan
    zeros1_odd = jnp.zeros(odd.n(1), dtype=DTYPE)
    return State(
        B_n=B_dof, B_nplus1=B_dof, p_n=p, p_nplus1=p, dt=jnp.ones((), dtype=DTYPE), dt_star=jnp.ones((), dtype=DTYPE), cfl_max=zero,
        warm=WarmStarts(p=p0, JxB=JxB0, J=J0, E=zeros1_odd, a=jnp.zeros(even.n(1), dtype=DTYPE),
                        a_F=a_F0,
                        A=zeros1_odd, resistive_delta=jnp.zeros(odd.n(2), dtype=DTYPE), dp=jnp.zeros_like(p)),
        last=LastStep(F=F0, F_norm=jnp.sqrt(F0 @ MF0), v=jnp.zeros(even.n(2), dtype=DTYPE), v_norm=zero,
                      newton_it=jnp.int32(0), resistive_it=jnp.int32(0), resistive_moved=zero,
                      smoothing_it=jnp.int32(0)),
        best=BestState(B=B_dof, p=p, resid=jnp.asarray(resid0, dtype=DTYPE), step=jnp.int32(step)),
    )


def chunk_runner(ts: TimeStepper, n_chunk: int) -> Callable[[State, int], tuple[State, dict]]:
    """Return a function ``run(state, it0) -> (state, trace)`` that takes ``n_chunk`` steps in one compiled loop.

    ``it0`` is the step number before the chunk. ``trace[name]`` is an array with one value per step:
    ``dE`` (the change of the energy the step lowers, ``int B^2/2 - p dV`` when compressible), ``F`` and ``v`` (the norms of force and velocity), ``dt``, ``dt_star``, ``cfl``,
    ``div`` (``||div B||``), ``Fu`` (``<F, u>``), ``newton_it``, ``res_it``, ``res_moved`` and ``resid``, the
    squared normalised force residual ``||F||^2 / ||grad(B^2/2)||^2``. A field with a lower ``resid`` than
    ``state.best`` replaces it. Runners of the same stepper share the compiled code. Only a new ``n_chunk``
    compiles again.
    """
    return lambda state, it0: _run_chunk(ts, state, jnp.asarray(it0), n_chunk)


def _chunk_body(ts, state, it):
    seq = ts.seq
    state = ts.relaxation_step(state)
    B_n, B_new = state.B_n, state.B_nplus1
    dE = 0.5 * ((B_new - B_n) @ (seq.odd.M[2] @ (B_new + B_n)))
    if ts.compressible:
        dE = dE - pressure_integral(state.p_nplus1 - state.p_n, seq)
    resid = (state.last.F_norm / force_scale(seq, B_new)) ** 2
    better = resid < state.best.resid
    best = BestState(B=jnp.where(better, B_n, state.best.B), p=jnp.where(better, state.p_n, state.best.p),
                     resid=jnp.where(better, resid, state.best.resid).astype(state.best.resid.dtype),
                     step=jnp.where(better, it - 1, state.best.step).astype(state.best.step.dtype))
    state = eqx.tree_at(lambda s: (s.B_n, s.p_n, s.best), state, (B_new, state.p_nplus1, best))
    trace = dict(
        dE=dE, F=state.last.F_norm, v=state.last.v_norm,
        dt=state.dt, dt_star=state.dt_star, cfl=state.cfl_max,
        div=compute_divergence_norm(state.B_n, seq),
        Fu=state.last.F @ (seq.even.M[2] @ state.last.v),
        newton_it=state.last.newton_it, res_it=state.last.resistive_it, res_moved=state.last.resistive_moved,
        smoothing_it=state.last.smoothing_it,
        resid=resid)
    return state, trace


# At module level with ts as an argument, so all runners on the same sequence share one compiled program. The
# step index is passed as an array because a Python int would be static and recompile every chunk.
@eqx.filter_jit
def _run_chunk(ts, state, it0, n_chunk):
    return jax.lax.scan(lambda st, it: _chunk_body(ts, st, it), state, it0 + jnp.arange(1, n_chunk + 1))


def force_scale(seq: DeRhamSequence, B: jnp.ndarray) -> jnp.ndarray:
    """Return ``||grad(B^2/2)||``, the scale the force residual is divided by. It keeps the residual of order
    one at any beta."""
    B_jk = seq.odd.evaluate_at_quadrature(B, 2)
    even = seq.even.free                                   # |B|^2 is even and free on the wall
    q = 0.5 * even.dot_product_load_values(B_jk, B_jk, 0, 2, 2)
    return even.l2_norm(even.G[0] @ even.M[0].solve(q), 1)


#: :func:`force_scale`, jit-compiled once per sequence.
force_scale_jit = eqx.filter_jit(force_scale)


def make_sampler(ts: TimeStepper):
    """Return a function ``sample(state, pw_guess) -> (state, p_w, scalars)`` that computes the diagnostics of
    the current field.

    ``scalars`` is a dict of Python floats: the energy ``E`` (computed in the residual precision), the
    ``helicity``, ``JoverB`` (``||J|| / ||B||``), ``JB`` (``int J . B``) and ``beta_vol``. For a compressible
    stepper ``E`` is ``int B^2/2 - p dV``, and ``p_int`` (``int p dV`` of the advected pressure) and ``beta_p``
    (``int p dV / int B^2/2 dV``) are added. ``p_w`` is the weak
    pressure, to be passed back as ``pw_guess`` next time. The returned state has refreshed starting guesses.
    """
    seq = ts.seq
    odd = seq.odd
    on = odd if odd.residual is None else odd.residual

    def sample(state: State, pw_guess: jnp.ndarray):
        f = _probe_jit
        w = state.warm
        p, JxB, J, A, p_w, h, JoverB, JB, beta = f(seq, state.B_n, w.p, w.JxB, w.J, state.last.F, pw_guess, w.A)
        state = eqx.tree_at(lambda s: (s.warm.p, s.warm.JxB, s.warm.J, s.warm.A), state, (p, JxB, J, A))
        E = 0.5 * float(on.l2_norm_sq(state.B_n.astype(RESIDUAL_DTYPE), 2))
        scalars = dict(E=E, helicity=float(h), JoverB=float(JoverB), JB=float(JB), beta_vol=float(beta))
        if ts.compressible:
            p_int = float(pressure_integral(state.p_n, seq))
            scalars.update(E=E - p_int, p_int=p_int, beta_p=p_int / E)
        return state, p_w, scalars

    return sample


def _probe(seq, B, p, JxB, J, F_prev, pw_guess, A):
    """The diagnostics of :func:`make_sampler` at the field ``B``."""
    F, p, J, JxB = compute_force(B, seq, p, JxB, J, F_prev)
    p_w = weak_pressure(J, B, seq, p_guess=pw_guess)
    h, A_new = compute_helicity(B, seq, A)
    odd = seq.odd
    JoverB = odd.l2_norm(J, 1) / odd.l2_norm(B, 2)
    JB = J @ (odd.P[2, 1] @ B)
    return p, JxB, J, A_new, p_w, h, JoverB, JB, beta_vol(B, p_w, seq)


#: :func:`_probe`, jit-compiled once per sequence.
_probe_jit = eqx.filter_jit(_probe)


def write_checkpoint(path: str, state: State, step: int, seq: DeRhamSequence) -> None:
    """Write the state to one HDF5 file, with every array stored under its name (``B_n``, ``warm.p``, ...).

    The step number and the discretisation (resolution, degree, ``nfp``, symmetry, precision and the knots) are
    stored as attributes, so :func:`checkpoint_attrs` can rebuild the matching sequence. The attribute
    ``angles = "right-handed"`` marks the convention of :func:`mrx.equilibria.read_equilibrium`, see
    :func:`check_checkpoint`.
    """
    import h5py  # noqa: PLC0415
    from mrx.precision import current_precision  # noqa: PLC0415
    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    axes = seq.basis_0.bases[0].bases
    with h5py.File(path, "w") as fh:
        fh.attrs["step"] = int(step)
        fh.attrs["resolution"] = np.asarray(seq.ns, dtype=np.int64)
        fh.attrs["degree"] = int(axes[0].p)
        fh.attrs["nfp"] = int(seq.nfp)
        fh.attrs["symmetry"] = str(seq.symmetry)
        fh.attrs["precision"] = current_precision()
        fh.attrs["angles"] = "right-handed"
        for name, basis in zip(("r", "theta", "zeta"), axes):
            T = np.asarray(basis.T, dtype=np.float64)
            fh.attrs[f"knots_{name}"] = np.unique(T[(T >= 0.0) & (T <= 1.0)])
        for keypath, leaf in leaves:
            fh.create_dataset(jax.tree_util.keystr(keypath).lstrip("."), data=np.asarray(leaf))


def checkpoint_attrs(path: str) -> dict:
    """Return the discretisation stored in a checkpoint as :func:`mrx.geometry.build_sequence` keywords (``ns``,
    ``p``, ``knots``, ``symmetry``), together with ``nfp``, ``precision`` and ``step``."""
    import h5py  # noqa: PLC0415
    with h5py.File(path, "r") as fh:
        a = dict(fh.attrs)

    def breakpoints(name):
        # a uniform axis returns None, because explicit angular knots would switch the half-period reduction off
        bp = np.asarray(a[f"knots_{name}"], dtype=np.float64)
        return None if np.allclose(bp, np.linspace(0.0, 1.0, bp.size)) else [float(v) for v in bp]

    return dict(ns=tuple(int(v) for v in a["resolution"]), p=int(a["degree"]),
                knots=[breakpoints(name) for name in ("r", "theta", "zeta")],
                symmetry=str(a["symmetry"]), nfp=int(a["nfp"]), precision=str(a["precision"]),
                step=int(a["step"]))


def check_checkpoint(path: str, seq: DeRhamSequence) -> None:
    """Raise if the checkpoint ``path`` does not belong to ``seq``: another resolution or degree, or angles of
    another orientation. Checkpoints written before the logical angles were made right-handed (2026-09-30)
    have no ``angles`` attribute. On a file read with its poloidal angle reversed (a VMEC wout) their
    coefficients describe the mirror image of the field."""
    import h5py  # noqa: PLC0415
    with h5py.File(path, "r") as fh:
        a = dict(fh.attrs)
    ns, p = tuple(int(v) for v in a["resolution"]), int(a["degree"])
    if ns != tuple(seq.ns) or p != seq.p:
        raise ValueError(f"{path}: resolution {ns} p={p}, the sequence is {tuple(seq.ns)} p={seq.p}")
    if "angles" not in a and seq.equilibrium is not None and seq.equilibrium["theta_reversed"]:
        raise ValueError(f"{path}: written in the left-handed angles of MRX before 2026-09-30, while "
                         f"{seq.equilibrium['path']} is now read with theta reversed. Relax the field again.")


def read_checkpoint(path: str, ts: TimeStepper) -> tuple[State, int]:
    """Return the ``(state, step)`` saved by :func:`write_checkpoint`. ``ts`` must be built on a sequence with
    the same discretisation. Arrays missing from the file are taken from :func:`initial_state` of the stored
    field."""
    import h5py  # noqa: PLC0415
    check_checkpoint(path, ts.seq)
    with h5py.File(path, "r") as fh:
        step = int(fh.attrs["step"])
        data = {k: np.asarray(v) for k, v in fh.items()}
    p = jnp.asarray(data["p_n"]) if ts.compressible else None
    skeleton = initial_state(jnp.asarray(data["B_n"]), ts, step=step, p=p)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(skeleton)
    new = []
    for keypath, leaf in leaves:
        v = data.get(jax.tree_util.keystr(keypath).lstrip("."))
        new.append(leaf if v is None else
                   jnp.asarray(v, dtype=DTYPE if np.issubdtype(v.dtype, np.floating) else v.dtype))
    return jax.tree_util.tree_unflatten(treedef, new), step


class RelaxResult(NamedTuple):
    """What :func:`relax` returns and hands to ``on_chunk``.

    ``steps`` is the number of steps done in this run, so ``it0 + steps`` is the current step number. ``stop``
    is ``"steps"``, ``"floor"`` or ``"running"``. ``wall`` is the time in seconds spent in the steps themselves.
    ``trace`` holds the per-step values of :func:`chunk_runner` except ``v`` and ``Fu``, plus ``dE_ls`` (the
    energy change the line search predicts, ``-dt <F, u> (1 - dt / 2 dt_star)``) and ``cos`` (the cosine between
    force and velocity). ``qoi`` holds the diagnostics of :func:`make_sampler` once per chunk, the first entry
    being the start of the run. ``E0`` is the energy at the start.
    """
    state: State
    steps: int
    stop: str
    wall: float
    trace: dict
    qoi: dict
    chunk: int
    E0: float


def relax(state: State, ts: TimeStepper, steps: int, chunk: int = 500, it0: int = 0,
          floor_tol: float = 0.0,
          on_chunk: Optional[Callable[[RelaxResult], None]] = None,
          verbose: bool = True) -> RelaxResult:
    """Run up to ``steps`` relaxation steps in compiled chunks of ``chunk`` steps and return a :class:`RelaxResult`.

    Diagnostics are computed and printed after every chunk. The run stops early when the mean force residual
    of a chunk falls below ``floor_tol``. ``on_chunk`` (for example a checkpoint writer) is called after every
    chunk. ``it0`` is the step number of ``state``, nonzero for a restart. ``steps`` must be a multiple of
    ``chunk``.
    """
    if chunk < 1 or steps % chunk:
        raise ValueError("steps must be a positive multiple of chunk")
    seq = ts.seq
    run = chunk_runner(ts, chunk)
    sample = make_sampler(ts)

    trace: dict = {k: [] for k in ("dE_ls", "cos")}    # the per-step scalars are added at the first chunk
    qoi: dict = {}

    def result(n_done, stop, wall):
        return RelaxResult(state, n_done, stop, wall, trace, qoi, chunk, E0)

    def record(it, wall, scalars):
        row = dict(it=it, wall=wall, F=float(state.last.F_norm),
                   resid=float((state.last.F_norm / force_scale_jit(seq, state.B_n)) ** 2), **scalars)
        for k, v in row.items():
            qoi.setdefault(k, []).append(v)

    t_arm = time.perf_counter()
    t_out = 0.0     # time spent in samples and callbacks, which wall excludes
    pw = jnp.zeros(seq.even.n(0), dtype=DTYPE)
    tq = time.perf_counter()
    state, pw, scalars = sample(state, pw)   # the start of THIS run
    E0, h0 = scalars["E"], scalars["helicity"]
    record(it0, 0.0, scalars)
    if verbose:
        print(f"[start] it {it0}  E={E0:.8e}  |F|={float(state.last.F_norm):.4e}  "
              f"resid={qoi['resid'][-1]:.4e}  H={h0:+.6e}  J/B={scalars['JoverB']:.4f}  "
              f"beta_vol={scalars['beta_vol']:.3e}"
              + (f"  beta_p={scalars['beta_p']:.4e}" if ts.compressible else ""), flush=True)
    t_out += time.perf_counter() - tq

    n_done, stop = 0, "running"
    for _ in range(steps // chunk):
        state, ch = run(state, it0 + n_done)
        ch = {k: np.asarray(v) for k, v in ch.items()}
        n_done += chunk
        it = it0 + n_done
        with np.errstate(invalid="ignore", divide="ignore"):   # a zero velocity has no cosine
            cos = ch["Fu"] / (ch["F"] * ch["v"])
            trace["cos"].extend(cos.tolist())
            trace["dE_ls"].extend((-ch["dt"] * ch["Fu"] * (1.0 - 0.5 * ch["dt"] / ch["dt_star"])).tolist())
        for k, v in ch.items():
            if k not in ("v", "Fu"):        # those only feed cos and dE_ls
                trace.setdefault(k, []).extend(v.tolist())
        resid_now = float(ch["resid"].mean())

        tq = time.perf_counter()
        wall = tq - t_arm - t_out
        state, pw, scalars = sample(state, pw)
        record(it, wall, scalars)
        if verbose:
            print(f"  it {it:>5d}  E_0-E={E0 - scalars['E']:.4e}  |F|={ch['F'][-1]:.4e}  "
                  f"resid={resid_now:.3e} (chunk mean)  H={scalars['helicity']:+.6e}  "
                  f"dH={scalars['helicity'] - h0:+.3e}  dt={ch['dt'].mean():+.3e}  "
                  f"cos min={np.nanmin(cos):+.4f}  divB={ch['div'].max():.2e}  beta_vol={scalars['beta_vol']:.3e}  "
                  + (f"beta_p={scalars['beta_p']:.4e}  " if ts.compressible else "")
                  + f"[{wall:.0f}s steps +{t_out:.0f}s other]"
                  + (f"\n           newton: MINRES it mean {np.abs(ch['newton_it']).mean():.0f} max "
                     f"{np.abs(ch['newton_it']).max()}, unconverged {int((ch['newton_it'] < 0).sum())}, "
                     f"dt* mean {ch['dt_star'].mean():.3e}" if ts.newton else "")
                  + (f"\n           resistive: eps {ts.resistivity:.3e} per step, CG it mean "
                     f"{np.abs(ch['res_it']).mean():.0f} max {np.abs(ch['res_it']).max()}, "
                     f"||delta||/||B|| mean {ch['res_moved'].mean():.2e}" if ts.resistivity else ""),
                  flush=True)
        if resid_now < floor_tol:
            stop = "floor"
        elif n_done == steps:
            stop = "steps"
        if on_chunk is not None:
            on_chunk(result(n_done, stop, wall))
        if stop != "running":
            if verbose and stop == "floor":
                print(f"  [floor] chunk mean of the force residual {resid_now:.3e} below {floor_tol:.1e} at it={it}",
                      flush=True)
            t_out += time.perf_counter() - tq
            break
        t_out += time.perf_counter() - tq

    res = result(n_done, stop, time.perf_counter() - t_arm - t_out)
    if verbose:
        print_summary(res, ts)
    return res


def print_summary(res: RelaxResult, ts: TimeStepper) -> None:
    """The end-of-run summary of :func:`relax`."""
    tr, q = res.trace, res.qoi
    n = res.steps
    E0 = res.E0
    dE, dE_ls = np.array(tr["dE"]), np.array(tr["dE_ls"])
    removed = E0 - q["E"][-1]
    ident = np.abs(dE - dE_ls) / E0
    resid = np.array(tr["resid"])
    print(f"\n--- {n} steps in {res.wall:.1f}s ({res.wall / max(n, 1):.2f} s/step), stopped on: {res.stop}")
    print(f"    E_0 {E0:.8e}, E_0 - E {removed:.4e}  ({removed / E0:.4%} of the initial energy removed)")
    print(f"    residual {resid[0]:.4e} -> {resid[-1]:.4e}  (mean over the last chunk of "
          f"{res.chunk} steps {resid[-res.chunk:].mean():.4e}, min {resid.min():.4e})")
    print(f"    best state: step {int(res.state.best.step)}, residual {float(res.state.best.resid):.4e}")
    print(f"    |dE - dE_ls| / E0 (the velocity's gradient part against grad p): median {np.median(ident):.3e}"
          f"  max {ident.max():.3e}")
    print(f"    energy increases on {int((dE > 0).sum())}/{n} steps,  ||div B|| max {max(tr['div']):.3e},  "
          f"||J||/||B|| {q['JoverB'][0]:.4e} -> {q['JoverB'][-1]:.4e}")
    h = np.array(q["helicity"])
    print(f"    helicity {h[0]:+.6e} -> {h[-1]:+.6e}  drift {h[-1] - h[0]:+.3e}"
          f"  relative {(h[-1] - h[0]) / abs(h[0]):+.3e}")
    print(f"    beta_vol {q['beta_vol'][0]:.4e} -> {q['beta_vol'][-1]:.4e}")
    if ts.compressible:
        print(f"    advected pressure: int p dV {q['p_int'][0]:.6e} -> {q['p_int'][-1]:.6e},  "
              f"beta_p {q['beta_p'][0]:.4e} -> {q['beta_p'][-1]:.4e}"
              + (f",  parallel smoothing CG it mean {np.abs(np.array(tr['smoothing_it'])).mean():.0f}, "
                 f"unconverged {int((np.array(tr['smoothing_it']) < 0).sum())}" if ts.smooth_pressure else ""))
    dts, dt_star = np.array(tr["dt"]), np.array(tr["dt_star"])
    print(f"    CFL cap (C={CFL}) bound on {int((dts < dt_star).sum())}/{n} steps,  "
          f"dt/dt* min {(dts / dt_star).min():.3f} mean {(dts / dt_star).mean():.3f},  "
          f"CFL number taken max {(dts * np.array(tr['cfl'])).max():.3f}")
    if ts.newton:
        nit = np.abs(np.array(tr["newton_it"]))
        print(f"    newton: MINRES iterations mean {nit.mean():.1f}  max {nit.max()}  "
              f"unconverged on {int((np.array(tr['newton_it']) < 0).sum())}/{n} steps,  "
              f"dt* mean {dt_star.mean():.3e}", flush=True)
