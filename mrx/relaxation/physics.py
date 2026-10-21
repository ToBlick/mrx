"""Physical quantities of a magnetic field ``B``, given as the coefficients of a 2-form on a
:class:`~mrx.derham_sequence.DeRhamSequence`.

- :func:`compute_force` returns the Lorentz force ``J x B`` with its pressure-gradient part removed, together
  with the pressure and the current. This is the force the relaxation drives to zero.
- :func:`weak_pressure` and :func:`beta_vol` give a pressure that vanishes on the wall and the volume-averaged beta.
- :func:`compute_helicity` and :func:`compute_divergence_norm` monitor the two quantities an ideal relaxation
  should conserve.
- :func:`resistive_step` takes one implicit step of resistive diffusion.

All functions are jit-compiled with the sequence as an argument. The first call on a sequence compiles, later
calls on the same sequence (also with a new geometry) reuse the compiled code.
"""
from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.precision import lift, to_working

# With stellarator symmetry, B, A, E and J live on the odd view ``seq.odd`` and the velocity, the force and the
# pressures on the even view ``seq.even``. A product is assembled on its own view from factors on theirs.


@eqx.filter_jit
def compute_helicity(B: jnp.ndarray, seq: DeRhamSequence, A_guess: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    """Return ``(H, A)``, the magnetic helicity ``H = <A, B + B_harm>`` and the vector potential ``A``.

    ``B`` is split as ``B = curl A + B_harm`` with ``B_harm`` harmonic. ``A_guess`` (for example the ``A`` of the
    previous call) is the starting guess of the solve for ``A``.
    """
    seq = seq.odd
    # A enters the helicity up to a gradient, which pairs to zero with the divergence-free B + B_harm
    # the right-hand side is formed in the arithmetic of the solve, so that it lies in the range of G^T to its accuracy
    A = seq.S[1].solve(seq.G[1].T @ (seq.M[2] @ lift(B)), guess=A_guess)
    B_harm = B - seq.G[1] @ A
    helicity = A @ (seq.P[2, 1] @ (B + B_harm))
    return helicity, A


@eqx.filter_jit
def compute_divergence_norm(B: jnp.ndarray, seq: DeRhamSequence) -> float:
    """Return the L2 norm of ``div B``. It is cheap (no linear solve) and should stay at round-off level."""
    seq = seq.odd
    return seq.l2_norm_sq(seq.G[2] @ B, 3) ** 0.5


@eqx.filter_jit
def current(B: jnp.ndarray, seq: DeRhamSequence, guess: jnp.ndarray | None = None) -> jnp.ndarray:
    """The current ``J = curl B`` of the odd 2-form ``B``, a 1-form computed weakly: ``M_1 J = D_1^T B``."""
    odd = seq.odd
    return odd.M[1].solve(odd.D[1].T @ B, guess=guess)


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
    J = current(B, seq, J_guess)
    JxB_dual = even.cross_product_load_values(odd.evaluate_at_quadrature(J, 1),
                                              odd.evaluate_at_quadrature(B, 2), 2, 1, 2)
    # kept as the solve accumulated it, because F = JxB - grad p is a small difference of large fields
    JxB = even.M[2].solve(JxB_dual, guess=JxB_guess, accurate=True)
    sigma_guess = None if F_guess is None else JxB_guess - F_guess
    F, p = even.leray(JxB, k=2, p_guess=p_guess, sigma_guess=sigma_guess)
    return F, p, J, to_working(JxB)


@eqx.filter_jit
def weak_pressure(J: jnp.ndarray, B: jnp.ndarray, seq: DeRhamSequence,
                  p_guess: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return the weak pressure ``p_w``, a 0-form that vanishes on the wall.

    It is the gradient part of the Helmholtz split ``J x B = F_w + grad p_w`` of the force as a 1-form. Unlike the
    pressure of :func:`compute_force` it does not absorb the force normal to the wall. ``J`` is the current of
    ``B`` (:func:`current`).
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
    is ``(M_2 + eps L_2) delta = -eps L_2 (B - B_ref)``, with ``M_2`` the 2-form mass matrix and ``L_2`` the
    2-form Laplacian. For a divergence-free ``B - B_ref`` its solution is a curl, ``delta = G_1 a`` with

        (M_1 + eps S_1) a = -eps G_1^T M_2 (B - B_ref),

    because ``L_2 (B - B_ref) = M_2 G_1 M_1^{-1} G_1^T M_2 (B - B_ref)`` and
    ``(M_2 + eps L_2) G_1 a = M_2 G_1 M_1^{-1} (M_1 + eps S_1) a``. That is one 1-form solve
    (``seq.shifted_stiffness(1, eps).solve(...)``), and ``div B`` stays at round-off, since ``G_2 G_1 = 0``.

    Returns ``(B + delta, info, ||delta|| / ||B||, a)``, where ``info`` is the iteration count of the solve
    (positive when converged, negative when not). ``guess`` is a starting guess for ``a``.
    """
    seq = seq.odd
    # solve for the small increment, not for B itself
    rhs = -eps * (seq.G[1].T @ (seq.M[2] @ (B if B_ref is None else B - B_ref)))
    a, info = seq.shifted_stiffness(1, eps).solve(rhs, guess=guess, return_info=True)
    delta = seq.G[1] @ a
    rel = seq.l2_norm(delta, 2) / seq.l2_norm(B, 2)
    return B + delta, info.astype(jnp.int32), rel, a
