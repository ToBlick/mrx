"""The midpoint-implicit induction, retired from the descent loop 2026-09-20.

``B_{n+1} = B_n + dt curl(u x X_mid)`` at the midpoint field ``(B_n +
B_{n+1}) / 2`` with the explicit predictor's velocity ``u`` and ``dt``, a
linear fixed point solved by Picard iteration. With the auxiliary field
(``X_mid = H_mid = M_1^-1 P B_mid``, the Dirichlet proxy) it conserved the
discrete helicity ``<A, B + B_harm>`` exactly; the helicity correction
(:attr:`mrx.relaxation.TimeStepper.helicity_correction`) does the same with
``H`` natural at one scalar per step, which is why the scheme left the loop
(``docs/research/implicit_midpoint_2026-09-04.md`` has the measurements).

TO USE IT IN THE OPTIMISER AGAIN: :class:`mrx.relaxation.State` needs its
Picard fields back (``picard_iterations``, ``picard_restarts``,
``picard_residual``, int32 / int32 / working-dtype scalars, seeded to 0 by
``initial_state`` and written by ``relaxation_step``, traced as
``picard_it`` / ``picard_resid``), :class:`~mrx.relaxation.TimeStepper` a
scheme switch that calls :func:`midpoint_solve` in place of the explicit
increment-and-step, and ``TimeStepper._helicity_lambda`` its ``dt``-less
branch ``<E, P B> / <H_D, P B>`` (the midpoint pairing, inlined below).
"""
import jax
import jax.numpy as jnp

from mrx.precision import RESIDUAL_DTYPE, eps
from mrx.relaxation import State, TimeStepper, dirichlet_proxy

#: A midpoint sweep whose defect exceeds this many times the predictor's
#: increment is not contracting: halve ``dt`` and start again.
PICARD_BLOWUP = 1e3
#: Sweeps at one ``dt`` before the midpoint solve halves ``dt`` and restarts
#: from the predictor.
PICARD_MAX = 20
#: Halvings allowed per step; after the last one the step goes out
#: unconverged, the residual above the tolerance.
PICARD_RESTARTS = 4
#: The Picard tolerance in units of ``seq.tol``: the inner solves define the
#: map, so a tighter fixed point means nothing.
PICARD_TOL_FACTOR = 10.0
#: ... plus this many roundoffs of the working dtype: the defect is formed in
#: the stored precision and floors there (11 eps measured on li383 (8,12,12)
#: p=2 in float32, 2026-09-05; 4e-15 in float64, inert).
PICARD_EPS_FACTOR = 20.0


def picard_tol(ts: TimeStepper) -> float:
    """``PICARD_TOL_FACTOR seq.tol + PICARD_EPS_FACTOR eps``."""
    return PICARD_TOL_FACTOR * ts.seq.tol + PICARD_EPS_FACTOR * eps()


def midpoint_solve(ts: TimeStepper, state: State):
    """Midpoint-implicit induction with the explicit descent velocity.

    The step is ``B_{n+1} = B_n + dt curl(u x X_mid)`` with ``u`` the
    descent velocity of the explicit predictor at ``B_n`` (direction,
    smoothing, line-search ``dt``, CFL cap: all of
    ``TimeStepper._ideal_increment``) and ``X_mid`` the MIDPOINT field
    ``(B_n + B_{n+1}) / 2`` itself or, with ``auxiliary_B_field``, its 1-form
    proxy ``H_mid = M_1^-1 P (B_n + B_{n+1}) / 2``: the auxiliary-variable
    scheme.

    WHY IT CONSERVES HELICITY.  The pairing of the 2-form ``B`` with a
    discrete 1-form ``E`` goes through the proxy ``H = M_1^-1 P B``::

        E^T P B = E^T M_1 H = H^T load(u x H) = int H_h . (u_h x H_h) = 0

    at every quadrature node, for ANY ``u``.  With ``B = D_1 A + B_harm`` and
    the exact discrete Stokes identity ``<A, D_1 E> = <D_1 A, E>``::

        d/dt <A, B + B_harm> = 2 <A, D_1 E> + 2 <E, B_harm> = 2 <B, E> = 0,

    so the semi-discrete flow conserves the discrete helicity exactly; it
    is a quadratic form ``Q(B)`` and evaluating ``E`` at the midpoint field
    keeps it exactly, ``Q(B_{n+1}) - Q(B_n) = 2 dt <B_mid, E> = 0``. The
    one condition: ``E`` and ``H`` in the SAME (Dirichlet) space; with a
    natural ``H`` both schemes leak through the wall layer alike (li383
    (8,16,16) p=2, float64, 1000 steps: -5.5e-7 explicit, -6.6e-7 midpoint;
    with the Dirichlet ``H`` +5e-12 against +2.2e-7).

    WHY THE VELOCITY STAYS EXPLICIT.  ``u`` at the midpoint makes the step a
    nonlinear fixed point through the force, whose linearisation is the
    descent operator ``|H|^2 curl curl``; the line-search ``dt`` sits 35x
    above the Picard contraction limit (li383 (8,16,16) p=2: blow-up in six
    sweeps; a Laplacian preconditioner only flips the spectrum; Newton is a
    Krylov solve inside a Krylov solve). With ``u`` frozen the map ``x ->
    dt curl(u x H(B_n + x / 2))`` is LINEAR in the increment with contraction
    constant ``dt |u| / (2 h)``, small because ``u`` is the force, so plain
    Picard converges in a few sweeps (one k=1 mass solve for ``H_mid``, one
    for ``E``, the topological curl, warm-started). On a blow-up
    (``PICARD_BLOWUP``) or ``PICARD_MAX`` sweeps ``dt`` is halved and the
    solve restarts from the predictor, at most ``PICARD_RESTARTS`` times.
    Convergence is judged on ``||g(x) - x||_M / ||dt dB(B_n)||_M``.

    Returns ``(inc, dt, dt_star, B_{n+1}, evaluations, restarts, residual,
    lambda)``: ``inc`` the predictor's increment with ``H`` and ``E``
    replaced by the midpoint's (the next step's warm starts), ``lambda``
    the helicity correction of the last sweep (0 without it).
    """
    B_n = state.B_n
    seq = ts.seq
    tol = picard_tol(ts)
    inc0 = ts._ideal_increment(B_n, state, state.warm.p, state.warm.H, state.warm.JxH, state.warm.J, state.warm.E)
    dt0, dt_star = ts._step_size(inc0)
    dB0 = inc0.dB
    dB0_norm = seq.l2_norm(dB0, 2)
    u_jk = seq.evaluate_at_quadrature(inc0.u, 2, True)
    one = jnp.ones((), B_n.dtype)

    def sweep(carry):
        k, n_eval, restarts, dt, x, H, E, resid, lam = carry
        B_mid = B_n + 0.5 * x
        if ts.auxiliary_B_field:
            _, H = dirichlet_proxy(seq, B_mid, H)
        E = ts._induction_field(u_jk, H if ts.auxiliary_B_field else B_mid, E)
        if ts.helicity_correction:
            PB, H = ts._helicity_proxy(B_mid, H, H)
            # the midpoint pairing: <E, P B_mid> / <H_D, P B_mid>, in the residual precision
            E64, PB64, H64 = (v.astype(RESIDUAL_DTYPE) for v in (E, PB, H))
            lam = ((E64 @ PB64) / (H64 @ PB64)).astype(B_n.dtype)
            E = E - lam.astype(E.dtype) * H
        g = dt * seq.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
        resid = seq.l2_norm(g - x, 2) / (dt * dB0_norm)
        k, n_eval = k + 1, n_eval + 1
        converged = resid <= tol
        restart = (~converged & (~(resid < PICARD_BLOWUP) | (k >= PICARD_MAX))
                   & (restarts < PICARD_RESTARTS))
        dt = jnp.where(restart, 0.5 * dt, dt)
        x = jnp.where(restart, dt * dB0, g)
        k = jnp.where(restart, 0, k)
        resid = jnp.where(restart, one, resid)
        restarts = restarts + restart.astype(jnp.int32)
        return k, n_eval, restarts, dt, x, H, E, resid, lam

    def unconverged(carry):
        k, _, _, _, _, _, _, resid, _ = carry
        return ~(resid <= tol) & (resid < PICARD_BLOWUP) & (k < PICARD_MAX)

    carry = (jnp.int32(0), jnp.int32(1), jnp.int32(0), dt0, dt0 * dB0, inc0.H, inc0.E, one,
             jnp.zeros((), B_n.dtype))
    _, n_eval, restarts, dt, x, H, E, resid, lam = jax.lax.while_loop(unconverged, sweep, carry)
    return inc0._replace(H=H, E=E), dt, dt_star, B_n + x, n_eval, restarts, resid, lam
