"""The relaxation loop, which lowers the magnetic energy of ``B`` towards an MHD equilibrium.

Each step moves the field as ``B <- B + dt curl(u x B)``. This keeps ``div B = 0`` and the helicity fixed, so
only the energy changes. The velocity ``u`` is either a Newton direction or a smoothed Lorentz force.

A typical run builds a :class:`TimeStepper` for a sequence, makes the start state with :func:`initial_state`
and calls :func:`relax`, which runs the steps in compiled chunks and prints diagnostics after every chunk.
:func:`write_checkpoint` and :func:`read_checkpoint` save and restore the state for a restart.

The sequence and the stepper's switches (Newton or not, resistive or not) are compiled in. Changing them
recompiles. A new geometry or a new field on the same sequence does not.
"""
from typing import Callable, NamedTuple, Optional

import time

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.relaxation.newton import (NEWTON_MASS_TOL, NEWTON_MAXITER, NEWTON_PENALTY, NEWTON_TOL, MassChebyshev,
                                   newton_direction)
from mrx.relaxation.physics import beta_vol, compute_divergence_norm, compute_helicity, resistive_step, weak_pressure


def knot_spacing(seq: DeRhamSequence) -> np.ndarray:
    """The smallest knot spacing of each logical direction, shape ``(3,)``."""
    h = []
    for b in seq.basis_0.bases[0].bases:
        h.append(np.diff(np.asarray(b.T)[b.p:-b.p]).min())
    return np.array(h)


def radial_cell_sq(seq: DeRhamSequence) -> jnp.ndarray:
    """Return the squared physical size of a radial cell, ``h_r^2 = <g_rr>_V dr^2``.

    ``dr`` is the smallest radial knot spacing and ``<g_rr>_V`` the volume average of the radial metric
    coefficient. It is the length unit of the velocity smoothing and of the resistive doses.
    """
    wJ = seq.quad.w * seq.jacobian_j
    return knot_spacing(seq)[0] ** 2 * jnp.sum(wJ * seq.metric_jkl[:, 0, 0]) / jnp.sum(wJ)


def logical_cfl_weights(seq: DeRhamSequence) -> jnp.ndarray:
    """Return the weights ``1 / (J h_i)``, shape ``(n_q, 3)``, with ``J`` the Jacobian determinant and ``h_i``
    the knot spacing of direction ``i``.

    Multiplied with the components of a 2-form velocity at the quadrature points they give the number of
    logical cells crossed per unit time, the CFL number of each direction.
    """
    h = jnp.asarray(knot_spacing(seq))
    weights = 1.0 / (seq.jacobian_j[:, None] * h[None, :])
    # the theta cell degenerates in the first radial span
    return weights.at[:, 1].multiply(seq.quad.x[:, 0] >= h[0])


class WarmStarts(eqx.Module):
    """The solutions of the last step's linear solves, kept as starting guesses for the next step.

    ``J`` is the current, ``E = u x B`` the electric field, ``a`` the vector potential of the step's direction,
    ``a_F`` the vector potential of the force (the same as ``a`` for gradient descent), ``A`` the vector potential
    of the helicity (updated only when diagnostics are sampled) and ``resistive_a`` the 1-form whose curl was the
    last resistive increment.
    """
    J: jnp.ndarray
    E: jnp.ndarray
    a: jnp.ndarray
    a_F: jnp.ndarray
    A: jnp.ndarray
    resistive_a: jnp.ndarray


class BestState(eqx.Module):
    """The field with the lowest force residual seen so far, that residual and the step it was reached at."""
    B: jnp.ndarray
    resid: jnp.ndarray
    step: jnp.ndarray


class State(eqx.Module):
    """Everything the relaxation carries from one step to the next: the current field ``B_n``, the warm starts of
    its solves and the best field so far. Make one with :func:`initial_state`. All leaves are arrays, so the
    state can be passed through compiled code. What a step measures goes into the trace of :func:`relax`.
    """
    B_n: jnp.ndarray
    warm: WarmStarts
    best: BestState


class Increment(NamedTuple):
    """The ideal increment ``dB = curl(u x B)`` at one field, with the intermediate results of the step."""
    dB: jnp.ndarray
    u: jnp.ndarray
    Mu: jnp.ndarray
    F: jnp.ndarray
    MF: jnp.ndarray
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


class TimeStepper(eqx.Module):
    """One step of the energy descent, ``B_{n+1} = B_n + dt curl(u x B)`` (forward Euler).

    The force is ``F = curl a + c h``, the divergence-free part of ``J x B`` computed from a vector potential
    ``a`` (one 1-form Laplacian solve) and the harmonic 2-forms ``h``, with no pressure solve.
    With ``newton=True`` the velocity ``u`` is the Newton direction of
    :func:`mrx.relaxation.newton.newton_direction`, controlled by ``newton_penalty``, ``newton_tol`` and
    ``newton_maxiter``. The mass solves inside the Hessian are replaced by a
    :class:`~mrx.relaxation.newton.MassChebyshev` of accuracy ``newton_mass_tol``, built once at construction
    (``None`` keeps the solves). Otherwise ``u``
    is the smoothed force ``(M_2 + mu L_2)^-1 M_2 F`` with ``mu = SMOOTHING_C h_r^2``, where ``M_2`` is the
    2-form mass matrix and ``L_2`` the 2-form Laplacian. The step length ``dt`` minimises the energy along the increment, capped by the CFL limit and, for Newton, by 1.

    A nonzero ``resistivity`` (the dose ``eta dt`` per step, a length squared) adds a
    :func:`~mrx.relaxation.physics.resistive_step` towards ``resistive_reference`` after every ideal step.
    The remaining fields are computed from the sequence at construction and should not be passed.
    """
    seq: DeRhamSequence
    newton: bool = False
    newton_penalty: float = NEWTON_PENALTY
    newton_tol: float = NEWTON_TOL
    newton_maxiter: int = NEWTON_MAXITER
    newton_mass_tol: Optional[float] = eqx.field(static=True, default=NEWTON_MASS_TOL)
    resistivity: float = 0.0
    resistive_reference: Optional[jnp.ndarray] = None
    velocity_smoothing_scale: float = None
    cfl_weights: jnp.ndarray = None
    harmonic: jnp.ndarray = None
    harmonic_norm_sq: jnp.ndarray = None
    mass_inverse: Optional[MassChebyshev] = None
    resistive: bool = eqx.field(static=True, default=False)

    def __post_init__(self):
        # with stellarator symmetry the harmonic form is odd and the force even, so this is empty
        even = self.seq.even
        hs = even.nullspace(2)
        self.harmonic = hs
        self.harmonic_norm_sq = jnp.asarray([h @ (even.M[2] @ h) for h in hs], dtype=jnp.float64)
        if not isinstance(self.harmonic_norm_sq, jax.core.Tracer) and bool((self.harmonic_norm_sq <= 0).any()):
            raise ValueError("the harmonic 2-forms of the sequence are zero: run mrx.nullspace.compute_nullspaces "
                             "before building the TimeStepper")
        # device scalars, not Python floats: a new geometry's h_r must not recompile the chunk
        self.velocity_smoothing_scale = SMOOTHING_C * radial_cell_sq(self.seq)
        self.resistive = bool(self.resistivity != 0)
        self.resistivity = jnp.asarray(self.resistivity, dtype=jnp.float64)
        self.cfl_weights = logical_cfl_weights(self.seq)
        self.mass_inverse = (MassChebyshev.build(self.seq, self.newton_mass_tol)
                             if self.newton and self.newton_mass_tol is not None else None)

    def _lorentz(self, B: jnp.ndarray, J_guess: jnp.ndarray):
        """Return ``(J, (J x B)_dual)``: the current and the Lorentz force as a dual 2-form (tested against the
        2-form basis), not projected."""
        odd, even = self.seq.odd, self.seq.even
        J = odd.M[1].solve(odd.D[1].T @ B, guess=J_guess)      # the weak curl of B
        return J, even.cross_product_load_values(odd.evaluate_at_quadrature(J, 1),
                                                 odd.evaluate_at_quadrature(B, 2), 2, 1, 2)

    def _force(self, JxB_dual: jnp.ndarray, a_guess: jnp.ndarray):
        """Return ``(F, M F, a)``: the divergence-free force ``F = curl a + c h`` of the Lorentz force
        ``JxB_dual``, with ``a`` a vector potential of the force and ``c h`` its harmonic part."""
        seq = self.seq.even
        # only curl a enters, so a keeps a gradient that nothing controls
        a = seq.S[1].solve(seq.G[1].T @ JxB_dual, guess=a_guess)
        ch = self.harmonic.T @ ((self.harmonic @ JxB_dual) / self.harmonic_norm_sq)   # zero on the even view
        F = seq.G[1] @ a + ch
        return F, seq.M[2] @ F, a

    def _smooth(self, F: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """The smoothed force ``(M_2 + mu L_2)^-1 M_2 F`` of the force ``F = curl a + c h`` of :meth:`_force`.

        Smoothing ``a`` with ``(M_1 + mu L_1)^-1 M_1`` and taking the curl equals smoothing ``curl a``, the
        harmonic part is unchanged, and ``(M + mu L)^-1`` and ``(M + mu S)^-1`` differ by an exact form, which the
        curl removes: one solve.
        """
        seq = self.seq.even
        a_s = seq.shifted_stiffness(1, self.velocity_smoothing_scale).solve(seq.M[1] @ a, guess=a)
        return F + seq.G[1] @ (a_s - a)

    def _ideal_increment(self, B: jnp.ndarray, w: WarmStarts) -> Increment:
        """The ideal increment at ``B``, every Krylov solve warm-started from ``w``."""
        seq = self.seq
        odd, even = seq.odd, seq.even
        J, JxB_dual = self._lorentz(B, w.J)
        F, MF, a_F = self._force(JxB_dual, w.a_F)
        if self.newton:
            # the Newton system and dt* see the force only through curl^T, which annihilates the gradient and
            # harmonic parts exactly, so they take the unprojected load. F is formed for the residual only.
            u, a, newton_it = newton_direction(seq, B, J, JxB_dual, w.a, self.newton_penalty, self.newton_tol,
                                               self.newton_maxiter, self.mass_inverse)
        else:
            u, a, newton_it = self._smooth(F, a_F), a_F, jnp.int32(0)    # u the smoothed force
        Mu = even.M[2] @ u
        u_jk = even.evaluate_at_quadrature(u, 2)
        E_dual = odd.cross_product_load_values(u_jk, odd.evaluate_at_quadrature(B, 2), 1, 2, 2)
        E = odd.M[1].solve(E_dual, guess=w.E)
        cfl_max = jnp.max(jnp.abs(u_jk) * self.cfl_weights)
        # the topological curl: div B is conserved exactly, no mass solve
        dB = odd.G[1] @ E
        return Increment(dB, u, Mu, F, MF, J, E, cfl_max, a, a_F, newton_it)

    def _step_size(self, inc: Increment) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(dt, dt_star)``. ``dt_star = <F, u> / ||dB||^2`` minimises the energy along the increment and
        ``dt`` is ``dt_star`` after the caps."""
        dt_star = (inc.F @ inc.Mu) / self.seq.odd.l2_norm_sq(inc.dB, 2)
        # a non-positive dt* is no step: a negative one would climb the energy
        dt = jnp.minimum(jnp.maximum(dt_star, 0.0), CFL / inc.cfl_max)
        if self.newton:
            dt = jnp.minimum(dt, 1.0)              # the Newton length
        return dt, dt_star

    def relaxation_step(self, state: State) -> tuple[jnp.ndarray, WarmStarts, dict]:
        """One step from ``state.B_n``. Returns ``(B_{n+1}, warm starts, scalars)``, where ``scalars`` holds what
        the step measured at ``B_n``: ``F`` and ``v`` (the norms of force and velocity), ``Fu`` (``<F, u>``),
        ``dt``, ``dt_star``, ``cfl`` (the largest CFL number of the velocity), ``newton_it``, ``res_it`` and
        ``res_moved`` (see :func:`relax`)."""
        w = state.warm
        inc = self._ideal_increment(state.B_n, w)
        dt, dt_star = self._step_size(inc)
        B_new = state.B_n + dt * inc.dB
        res_a, res_it, res_moved = w.resistive_a, jnp.int32(0), jnp.zeros(())
        if self.resistive:
            B_new, res_it, res_moved, res_a = resistive_step(B_new, self.seq, self.resistivity,
                                                             self.resistive_reference, guess=res_a)
        warm = WarmStarts(J=inc.J, E=inc.E, a=inc.a, a_F=inc.a_F, A=w.A, resistive_a=res_a)
        scalars = dict(F=jnp.sqrt(inc.F @ inc.MF), v=jnp.sqrt(inc.u @ inc.Mu), Fu=inc.F @ inc.Mu, dt=dt,
                       dt_star=dt_star, cfl=inc.cfl_max, newton_it=inc.newton_it, res_it=res_it,
                       res_moved=res_moved)
        return B_new, warm, scalars


def initial_state(B_dof: jnp.ndarray, ts: TimeStepper, step: int = 0) -> State:
    """Return the start state for the field ``B_dof``. ``step`` is the step number the field is at (nonzero for
    a restart). This evaluates the force once."""
    seq = ts.seq
    odd, even = seq.odd, seq.even
    J0, JxB_dual = ts._lorentz(B_dof, None)
    F0, MF0, a_F0 = ts._force(JxB_dual, jnp.zeros(even.n(1)))
    resid0 = (jnp.sqrt(F0 @ MF0) / force_scale_jit(seq, B_dof)) ** 2
    zeros1_odd = jnp.zeros(odd.n(1))
    return State(
        B_n=B_dof,
        warm=WarmStarts(J=J0, E=zeros1_odd, a=jnp.zeros(even.n(1)), a_F=a_F0, A=zeros1_odd,
                        resistive_a=zeros1_odd),
        best=BestState(B=B_dof, resid=jnp.asarray(resid0, dtype=jnp.float64), step=jnp.int32(step)),
    )


def _chunk_body(ts, state, it):
    seq = ts.seq
    B_n = state.B_n
    B_new, warm, scalars = ts.relaxation_step(state)
    dE = 0.5 * ((B_new - B_n) @ (seq.odd.M[2] @ (B_new + B_n)))
    # the force of this step is that of B_n, so the scale is too: best.resid is the residual of best.B
    resid = (scalars["F"] / force_scale(seq, B_n)) ** 2
    better = resid < state.best.resid
    best = BestState(B=jnp.where(better, B_n, state.best.B),
                     resid=jnp.where(better, resid, state.best.resid).astype(state.best.resid.dtype),
                     step=jnp.where(better, it - 1, state.best.step).astype(state.best.step.dtype))
    trace = dict(scalars, dE=dE, div=compute_divergence_norm(B_new, seq), resid=resid)
    return State(B_n=B_new, warm=warm, best=best), trace


# At module level with ts as an argument, so all chunks on the same sequence share one compiled program. The
# step index is passed as an array because a Python int would be static and recompile every chunk.
@eqx.filter_jit
def _run_chunk(ts, state, it0, n_chunk):
    """Take ``n_chunk`` steps from ``state`` in one compiled loop and return ``(state, trace)``.

    ``it0`` is the step number before the chunk. ``trace[name]`` is an array with one value per step: the
    scalars of :meth:`TimeStepper.relaxation_step` and ``dE`` (the energy change), ``div`` (``||div B||``) and
    ``resid``, the squared normalised force residual ``||F||^2 / ||grad(B^2/2)||^2``. A field with a lower
    ``resid`` than ``state.best`` replaces it. Only a new ``n_chunk`` compiles again.
    """
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


def sample(ts: TimeStepper, state: State, pw_guess: jnp.ndarray) -> tuple[State, jnp.ndarray, dict]:
    """Return ``(state, p_w, scalars)``, the diagnostics of the current field ``state.B_n``.

    ``scalars`` is a dict of Python floats: the energy ``E``, the force norm ``F`` and residual ``resid`` (as in
    the trace of :func:`relax`), the ``helicity``, ``JoverB`` (``||J|| / ||B||``), ``JB`` (``int J . B``) and
    ``beta_vol``. ``p_w`` is the weak pressure, to be passed back as ``pw_guess`` next time. The returned state
    has refreshed warm starts.
    """
    state, p_w, scalars = _sample(ts, state, pw_guess)
    return state, p_w, {k: float(v) for k, v in scalars.items()}


@eqx.filter_jit
def _sample(ts, state, pw_guess):
    seq = ts.seq
    odd = seq.odd
    B, w = state.B_n, state.warm
    J, JxB_dual = ts._lorentz(B, w.J)
    F, MF, a_F = ts._force(JxB_dual, w.a_F)
    p_w = weak_pressure(J, B, seq, p_guess=pw_guess)
    h, A = compute_helicity(B, seq, w.A)
    F_norm = jnp.sqrt(F @ MF)
    scalars = dict(E=0.5 * odd.l2_norm_sq(B, 2), F=F_norm, resid=(F_norm / force_scale(seq, B)) ** 2, helicity=h,
                   JoverB=odd.l2_norm(J, 1) / odd.l2_norm(B, 2), JB=J @ (odd.P[2, 1] @ B),
                   beta_vol=beta_vol(B, p_w, seq))
    state = eqx.tree_at(lambda s: (s.warm.J, s.warm.a_F, s.warm.A), state, (J, a_F, A))
    return state, p_w, scalars


def write_checkpoint(path: str, state: State, step: int, seq: DeRhamSequence) -> None:
    """Write the state to one HDF5 file, with every array stored under its name (``B_n``, ``warm.J``, ...).

    The step number and the discretisation (resolution, degree, ``nfp``, symmetry and the knots) are
    stored as attributes, so :func:`checkpoint_attrs` can rebuild the matching sequence.
    """
    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    with h5py.File(path, "w") as fh:
        fh.attrs["step"] = int(step)
        fh.attrs["resolution"] = np.asarray(seq.ns, dtype=np.int64)
        fh.attrs["degree"] = int(seq.p)
        fh.attrs["nfp"] = int(seq.nfp)
        fh.attrs["symmetry"] = str(seq.symmetry)
        for name, bp in zip(_AXES, _breakpoints(seq)):
            fh.attrs[f"knots_{name}"] = bp
        for keypath, leaf in leaves:
            fh.create_dataset(jax.tree_util.keystr(keypath).lstrip("."), data=np.asarray(leaf))


_AXES = ("r", "theta", "zeta")


def _breakpoints(seq):
    """The breakpoints of the three axes of ``seq``, as stored in a checkpoint."""
    Ts = (np.asarray(b.T, dtype=np.float64) for b in seq.basis_0.bases[0].bases)
    return [np.unique(T[(T >= 0.0) & (T <= 1.0)]) for T in Ts]


def checkpoint_attrs(path: str) -> dict:
    """Return the discretisation stored in a checkpoint as :func:`mrx.geometry.build_sequence` keywords (``ns``,
    ``p``, ``knots``, ``symmetry``), together with ``nfp`` and ``step``."""
    with h5py.File(path, "r") as fh:
        a = dict(fh.attrs)

    def breakpoints(name):
        # a uniform axis returns None, because explicit angular knots would switch the half-period reduction off
        bp = np.asarray(a[f"knots_{name}"], dtype=np.float64)
        return None if np.allclose(bp, np.linspace(0.0, 1.0, bp.size)) else [float(v) for v in bp]

    return dict(ns=tuple(int(v) for v in a["resolution"]), p=int(a["degree"]),
                knots=[breakpoints(name) for name in _AXES],
                symmetry=str(a["symmetry"]), nfp=int(a["nfp"]), step=int(a["step"]))


def check_checkpoint(path: str, seq: DeRhamSequence) -> None:
    """Raise if the checkpoint ``path`` was written on another discretisation than that of ``seq``: another
    resolution, degree, number of field periods, symmetry or knots."""
    with h5py.File(path, "r") as fh:
        a = dict(fh.attrs)
    stored = (tuple(int(v) for v in a["resolution"]), int(a["degree"]), int(a["nfp"]), str(a["symmetry"]))
    same_knots = all(np.allclose(a[f"knots_{name}"], bp, rtol=0.0, atol=1e-12) if a[f"knots_{name}"].shape == bp.shape
                     else False for name, bp in zip(_AXES, _breakpoints(seq)))
    if stored != (tuple(seq.ns), seq.p, seq.nfp, seq.symmetry) or not same_knots:
        raise ValueError(f"{path}: written on {stored[0]} p={stored[1]} nfp={stored[2]} {stored[3]}"
                         f"{'' if same_knots else ' with other knots'}, the sequence is {tuple(seq.ns)} p={seq.p} "
                         f"nfp={seq.nfp} {seq.symmetry}")


def read_checkpoint(path: str, ts: TimeStepper) -> tuple[State, int]:
    """Return the ``(state, step)`` saved by :func:`write_checkpoint`. ``ts`` must be built on a sequence with
    the same discretisation. Arrays missing from the file are taken from :func:`initial_state` of the stored
    field."""
    check_checkpoint(path, ts.seq)
    with h5py.File(path, "r") as fh:
        step = int(fh.attrs["step"])
        data = {k: np.asarray(v) for k, v in fh.items()}
    skeleton = initial_state(jnp.asarray(data["B_n"], dtype=jnp.float64), ts, step=step)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(skeleton)
    new = []
    for keypath, leaf in leaves:
        v = data.get(jax.tree_util.keystr(keypath).lstrip("."))
        new.append(leaf if v is None else
                   jnp.asarray(v, dtype=jnp.float64 if np.issubdtype(v.dtype, np.floating) else v.dtype))
    return jax.tree_util.tree_unflatten(treedef, new), step


class RelaxResult(NamedTuple):
    """What :func:`relax` returns and hands to ``on_chunk``.

    ``steps`` is the number of steps done in this run, so ``it0 + steps`` is the current step number. ``stop``
    is ``"steps"``, ``"floor"`` or ``"running"``. ``wall`` is the time in seconds spent in the steps themselves.
    ``trace`` holds the per-step values of :func:`_run_chunk` except ``v`` and ``Fu``, plus ``dE_ls`` (the
    energy change the line search predicts, ``-dt <F, u> (1 - dt / 2 dt_star)``) and ``cos`` (the cosine between
    force and velocity). ``qoi`` holds the diagnostics of :func:`sample` once per chunk, the first entry
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
    trace: dict = {k: [] for k in ("dE_ls", "cos")}    # the per-step scalars are added at the first chunk
    qoi: dict = {}

    def result(n_done, stop, wall):
        return RelaxResult(state, n_done, stop, wall, trace, qoi, chunk, E0)

    def record(it, wall, scalars):
        for k, v in dict(it=it, wall=wall, **scalars).items():
            qoi.setdefault(k, []).append(v)

    t_arm = time.perf_counter()
    t_out = 0.0     # time spent in samples and callbacks, which wall excludes
    pw = jnp.zeros(seq.even.n(0))
    tq = time.perf_counter()
    state, pw, scalars = sample(ts, state, pw)   # the start of THIS run
    E0, h0 = scalars["E"], scalars["helicity"]
    record(it0, 0.0, scalars)
    if verbose:
        print(f"[start] it {it0}  E={E0:.8e}  |F|={scalars['F']:.4e}  "
              f"resid={scalars['resid']:.4e}  H={h0:+.6e}  J/B={scalars['JoverB']:.4f}  "
              f"beta_vol={scalars['beta_vol']:.3e}", flush=True)
    t_out += time.perf_counter() - tq

    n_done, stop = 0, "running"
    for _ in range(steps // chunk):
        state, ch = _run_chunk(ts, state, jnp.asarray(it0 + n_done), chunk)
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
        state, pw, scalars = sample(ts, state, pw)
        record(it, wall, scalars)
        if verbose:
            print(f"  it {it:>5d}  E_0-E={E0 - scalars['E']:.4e}  |F|={ch['F'][-1]:.4e}  "
                  f"resid={resid_now:.3e} (chunk mean)  H={scalars['helicity']:+.6e}  "
                  f"dH={scalars['helicity'] - h0:+.3e}  dt={ch['dt'].mean():+.3e}  "
                  f"cos min={np.nanmin(cos):+.4f}  divB={ch['div'].max():.2e}  beta_vol={scalars['beta_vol']:.3e}  "
                  f"[{wall:.0f}s steps +{t_out:.0f}s other]"
                  + (f"\n           newton: MINRES it mean {np.abs(ch['newton_it']).mean():.0f} max "
                     f"{np.abs(ch['newton_it']).max()}, unconverged {int((ch['newton_it'] < 0).sum())}, "
                     f"dt* mean {ch['dt_star'].mean():.3e}" if ts.newton else "")
                  + (f"\n           resistive: eps {ts.resistivity:.3e} per step, CG it mean "
                     f"{np.abs(ch['res_it']).mean():.0f} max {np.abs(ch['res_it']).max()}, "
                     f"||delta||/||B|| mean {ch['res_moved'].mean():.2e}" if ts.resistive else ""),
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
    dts, dt_star = np.array(tr["dt"]), np.array(tr["dt_star"])
    print(f"    CFL cap (C={CFL}) bound on {int((dts < dt_star).sum())}/{n} steps,  "
          f"dt/dt* min {(dts / dt_star).min():.3f} mean {(dts / dt_star).mean():.3f},  "
          f"CFL number taken max {(dts * np.array(tr['cfl'])).max():.3f}")
    if ts.newton:
        nit = np.abs(np.array(tr["newton_it"]))
        print(f"    newton: MINRES iterations mean {nit.mean():.1f}  max {nit.max()}  "
              f"unconverged on {int((np.array(tr['newton_it']) < 0).sum())}/{n} steps,  "
              f"dt* mean {dt_star.mean():.3e}", flush=True)
