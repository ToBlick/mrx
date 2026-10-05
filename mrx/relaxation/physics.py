"""Physical quantities of a magnetic field ``B``, given as the coefficients of a 2-form on a
:class:`~mrx.derham_sequence.DeRhamSequence`.

- :func:`compute_force` returns the Lorentz force ``J x B`` with its pressure-gradient part removed, together
  with the pressure and the current. This is the force the relaxation drives to zero.
- :func:`weak_pressure` and :func:`beta_vol` give a pressure that vanishes on the wall and the volume-averaged beta.
- :func:`compute_helicity` and :func:`compute_divergence_norm` monitor the two quantities an ideal relaxation
  should conserve.
- :func:`resistive_step` takes one implicit step of resistive diffusion.
- :func:`initial_pressure`, :func:`pressure_gradient`, :func:`advection` and :func:`pressure_integral` serve the
  compressible relaxation, in which a prescribed pressure is carried along by the flow instead of being the
  multiplier of a divergence-free velocity.

All functions are jit-compiled with the sequence as an argument. The first call on a sequence compiles, later
calls on the same sequence (also with a new geometry) reuse the compiled code.
"""
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.precision import DTYPE, RESIDUAL_DTYPE

# With stellarator symmetry, B, A, E and J live on the odd view ``seq.odd`` and the velocity, the force and the
# pressures on the even view ``seq.even``. A product is assembled on its own view from factors on theirs.
# The advected pressure is a free 0-form (``seq.even.free``). The constant is in that space, so the discrete
# advection changes int p dV by exactly -int u . grad p dV.


@eqx.filter_jit
def compute_helicity(B: jnp.ndarray, seq: DeRhamSequence, A_guess: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    """Return ``(H, A)``, the magnetic helicity ``H = <A, B + B_harm>`` and the vector potential ``A``.

    ``B`` is split as ``B = curl A + B_harm`` with ``B_harm`` harmonic. ``A_guess`` (for example the ``A`` of the
    previous call) is the starting guess of the solve for ``A``.
    """
    seq = seq.odd
    # the saddle solve takes the DUAL 1-form D_1^T B, not the weak curl M_1^-1 D_1^T B
    A = seq.L[1].solve(seq.D[1].T @ B, guess=A_guess)
    B_harm = B - seq.G[1] @ A
    helicity = A @ (seq.P[2, 1] @ (B + B_harm))
    return helicity, A


@eqx.filter_jit
def compute_divergence_norm(B: jnp.ndarray, seq: DeRhamSequence) -> float:
    """Return the L2 norm of ``div B``. It is cheap (no linear solve) and should stay at round-off level."""
    seq = seq.odd
    return seq.l2_norm_sq(seq.G[2] @ B, 3) ** 0.5


@eqx.filter_jit
def compute_force(B: jnp.ndarray, seq: DeRhamSequence, p_guess: jnp.ndarray | None = None,
                  JxB_guess: jnp.ndarray | None = None, J_guess: jnp.ndarray | None = None,
                  F_guess: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(F, p, J, JxB)``. ``JxB`` is the Lorentz force, ``F`` is ``JxB`` with its gradient part ``grad p``
    removed (the divergence-free force), ``p`` is that pressure (a 3-form) and ``J = curl B`` the current (a
    1-form, computed weakly).

    The optional guesses are the results of a previous call and only speed up the linear solves.
    """
    odd, even = seq.odd, seq.even
    J = odd.weak_curl(B, guess=J_guess)
    JxB_dual = even.cross_product_load_values(odd.evaluate_at_quadrature(J, 1),
                                              odd.evaluate_at_quadrature(B, 2), 2, 1, 2)
    # in the residual precision, because F = JxB - grad p is a small difference of large fields
    JxB = even.M[2].solve(JxB_dual, guess=JxB_guess, dtype=RESIDUAL_DTYPE)
    sigma_guess = None if F_guess is None else JxB_guess - F_guess
    F, p = even.leray(JxB, k=2, p_guess=p_guess, sigma_guess=sigma_guess)
    return F, p, J, JxB.astype(DTYPE)


@eqx.filter_jit
def weak_pressure(J: jnp.ndarray, B: jnp.ndarray, seq: DeRhamSequence,
                  p_guess: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return the weak pressure ``p_w``, a 0-form that vanishes on the wall.

    It is the gradient part of the Helmholtz split ``J x B = F_w + grad p_w`` of the force as a 1-form. Unlike the
    pressure of :func:`compute_force` it does not absorb the force normal to the wall. Pass the ``J`` returned by
    :func:`compute_force`.
    """
    odd, even = seq.odd, seq.even
    # the Dirichlet 1-forms suffice: p_w only sees J x B tested against gradients of 0-forms that vanish on the wall
    v_dual = even.cross_product_load_values(odd.evaluate_at_quadrature(J, 1),
                                            odd.evaluate_at_quadrature(B, 2), 1, 1, 2)
    return even.leray(even.M[1].solve(v_dual), k=1, p_guess=p_guess)[1]


@eqx.filter_jit
def beta_vol(B: jnp.ndarray, p_w: jnp.ndarray, seq: DeRhamSequence) -> jnp.ndarray:
    """Return the volume beta ``int p_w dV / int B^2/2 dV``, with ``p_w`` from :func:`weak_pressure`."""
    even = seq.even
    wJ = even.quad.w * even.jacobian_j
    pw_q = even.evaluate_at_quadrature(p_w, 0)[:, 0]
    return jnp.sum(wJ * pw_q) / (0.5 * seq.odd.l2_norm_sq(B, 2))


@eqx.filter_jit
def resistive_step(B: jnp.ndarray, seq: DeRhamSequence, eps, B_ref: Optional[jnp.ndarray] = None,
                   guess: Optional[jnp.ndarray] = None):
    """Take one backward-Euler step of resistive diffusion ``dB/dt = -eta curl (curl B - J_ref)``.

    ``eps = eta dt`` is the step's dose (a length squared). ``J_ref`` is the current of ``B_ref``, or zero if
    ``B_ref`` is not given, so the field diffuses towards ``B_ref`` rather than towards a vacuum field. The step
    solves ``(M_2 + eps L_2) delta = -eps L_2 (B - B_ref)``, with ``M_2`` the 2-form mass matrix and ``L_2`` the
    2-form Laplacian. Returns ``(B + delta, info, ||delta|| / ||B||)``, where ``info`` is the iteration count of
    the solve (positive when converged, negative when not). ``guess`` is a starting guess for ``delta``.
    """
    seq = seq.odd
    # solve for the small increment, not for B itself, so that float32 keeps its accuracy
    rhs = -eps * (seq.L[2] @ (B if B_ref is None else B - B_ref))
    delta, info = seq.shifted(2, eps).solve(rhs, guess=guess, return_info=True)
    rel = seq.l2_norm(delta, 2) / seq.l2_norm(B, 2)
    return B + delta, info.astype(jnp.int32), rel


def initial_pressure(seq: DeRhamSequence, B: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Return the advected pressure at the start, a free 0-form on the even view.

    It has the shape of the equilibrium file's pressure profile as a function of the logical radius ``r``, so it
    is constant on the flux surfaces of the file's field, and is scaled to the volume beta
    ``int p dV / int B^2/2 dV = beta`` of the field ``B``."""
    r = np.linspace(0.0, 1.0, 2001)
    values = seq.equilibrium["profiles"]["pressure"](r)
    if not np.any(values):
        raise ValueError(f"{seq.equilibrium['path']} has no pressure profile to prescribe")
    r, values = jnp.asarray(r), jnp.asarray(values / np.abs(values).max())
    p = seq.even.free.interpolate(lambda x: jnp.interp(x[0], r, values), 0, frame='logical')
    return p * (beta * 0.5 * seq.odd.l2_norm_sq(B, 2) / pressure_integral(p, seq))


@eqx.filter_jit
def pressure_integral(p: jnp.ndarray, seq: DeRhamSequence) -> jnp.ndarray:
    """Return ``int p dV`` of an advected pressure ``p``."""
    even = seq.even
    return jnp.sum(even.quad.w * even.jacobian_j * even.free.evaluate_at_quadrature(p, 0)[:, 0])


def pressure_gradient(p: jnp.ndarray, seq: DeRhamSequence) -> jnp.ndarray:
    """Return the covariant components of ``grad p`` at the quadrature points, for an advected pressure ``p``.

    For the result ``g``, ``seq.even.vector_load_values(g, 1, 2)`` is the force ``grad p`` tested against the
    2-forms. It carries no metric."""
    free = seq.even.free
    return free.evaluate_at_quadrature(free.G[0] @ p, 1)


def advection(grad_p: jnp.ndarray, u_jk: jnp.ndarray, seq: DeRhamSequence,
              guess: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return ``dp/dt = -u . grad p``, the rate of change of the advected pressure under the velocity ``u``.

    ``grad_p`` is the output of :func:`pressure_gradient` and ``u_jk`` the quadrature values of the velocity
    2-form. The result is the L2 projection onto the free 0-forms, one mass solve warm-started from ``guess``."""
    free = seq.even.free
    return -free.M[0].solve(free.dot_product_load_values(u_jk, grad_p, 0, 2, 1), guess=guess)


def parallel_smoothing(p: jnp.ndarray, B: jnp.ndarray, seq: DeRhamSequence, eps) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(p + delta, info)``: one backward-Euler step of diffusion of the advected pressure along ``B``,
    ``(M_0 + eps K) delta = -eps K p``, with ``K`` the form ``int (b . grad p)(b . grad w) dV`` and ``b = B/|B|``.

    It damps the variation of ``p`` along the field lines that the discrete advection creates and leaves its
    variation across them alone. ``eps`` is a length squared. On the free 0-forms the constant is in the kernel of
    ``K``, so ``int p dV`` does not change. ``info`` is the signed iteration count of the conjugate-gradient solve,
    preconditioned by the 0-form mass atom.
    """
    from mrx.precision import default_tol  # noqa: PLC0415
    from mrx.solvers import preconditioned_cg  # noqa: PLC0415
    free = seq.even.free
    B_jk = seq.odd.evaluate_at_quadrature(B, 2)
    # the parallel conductivity J b b^T maps grad p (covariant) to a contravariant density, with no metric on top
    K_q = (jnp.einsum('qi,qj->qij', B_jk, B_jk) * seq.jacobian_j[:, None, None]
           / jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk)[:, None, None])

    def K(q):
        flux = jnp.einsum('qij,qj->qi', K_q, free.evaluate_at_quadrature(free.G[0] @ q, 1))
        return free.G[0].T @ free.vector_load_values(flux, 2, 1)

    # solved for the small increment, so that float32 keeps its accuracy
    delta, info = preconditioned_cg(lambda x: free.M[0] @ x + eps * K(x), -eps * K(p), M=free.M[0].precondition,
                                    tol=default_tol(free.dtype, refine=False), maxiter=seq.maxiter)
    return p + delta, info
