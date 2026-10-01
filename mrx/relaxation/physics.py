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
from mrx.precision import DTYPE, RESIDUAL_DTYPE

# With stellarator symmetry, B, A, E and J live on the odd view ``seq.odd`` and the velocity, the force and the
# pressures on the even view ``seq.even``. A product is assembled on its own view from factors on theirs.


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
