"""Energy-descent relaxation of a 2-form magnetic field at fixed helicity: force, time stepper, and diagnostics."""
# %%
from typing import Callable, NamedTuple, Optional

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.hessian import NEWTON_MAXITER, NEWTON_PASSES, NEWTON_PENALTY, NEWTON_TOL, newton_direction
from mrx.precision import DTYPE, RESIDUAL_DTYPE


def compute_helicity(B: jnp.ndarray, seq: DeRhamSequence, A_guess: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    # The rhs must be the DUAL 1-form D_1^T B, not the primal weak curl.
    #
    # apply_inverse_laplacian solves the saddle form
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
    # analytic-profile IC at quasr44970 ns=(8,16,8) p=3:
    #
    #     primal rhs (old):  ||B - curl A|| / ||B|| = 8.56e+01   H = +1.99e+01
    #     dual rhs   (new):  ||B - curl A|| / ||B|| = 9.74e-01   H = +1.73e-02
    #
    # 85x is not a fraction of anything.  0.974 is, and it is the right size:
    # the IC is dominated by net toroidal flux, which IS the harmonic mode.
    # See docs/research/handoff_2026-08-25_relaxation_prelim.md.
    A = seq.apply_inverse_laplacian(
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
    # mass solver's residual -- see TimeStepper.relaxation_step.
    div_B = seq.apply_incidence_matrix(
        B, 2, dirichlet_in=True, dirichlet_out=True)
    return seq.l2_norm_sq(div_B, 3)**0.5

# %%


def dirichlet_proxy(seq: DeRhamSequence, B: jnp.ndarray, guess: jnp.ndarray | None = None):
    """``(P B, H_D)``: the dual Dirichlet 1-form of the 2-form ``B`` and its
    Dirichlet proxy ``H_D = M_1^-1 P B`` (the auxiliary field, the field of
    the helicity correction), the mass solve
    warm-started from ``guess``."""
    PB = seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True)
    return PB, seq.apply_inverse_mass_matrix(PB, 1, dirichlet=True, guess=guess)


def compute_force(
    B: jnp.ndarray,
    seq: DeRhamSequence,
    auxiliary_B_field: bool = False,
    p_guess: jnp.ndarray | None = None,
    H_guess: jnp.ndarray | None = None,
    JxH_guess: jnp.ndarray | None = None,
    J_guess: jnp.ndarray | None = None,
    F_guess: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The Leray-projected Lorentz force at ``B`` and what it was built from.

    ``J`` is the weak curl of ``B`` (a Dirichlet 1-form). Without the
    auxiliary field the force is ``J x B`` with the 2-form ``B`` itself;
    with it, ``J x H`` where ``H = M_1^-1 P B`` is the Dirichlet 1-form
    proxy of ``B``. Returns ``(F,
    p, J, X, JxX)``: ``p`` the Leray multiplier, ``X`` the field the cross
    products read (``H``, or ``B`` itself) and ``JxX`` the unprojected
    force. The guesses are the previous call's ``p``, ``X``, ``JxX`` and
    ``J``, and ``F_guess`` its force: with ``JxH_guess`` it gives the
    previous gradient part ``JxX - F``, which warm-starts the lower block
    of the Leray saddle solve next to ``p_guess`` on its upper one.
    """
    J = seq.apply_weak_curl(B, dirichlet=True, guess=J_guess)
    if auxiliary_B_field:
        _, X = dirichlet_proxy(seq, B, H_guess)
        JxX_dual = seq.cross_product_load(J, X, 2, 1, 1, True, True, True)
    else:
        X = B
        JxX_dual = seq.cross_product_load(J, B, 2, 1, 2, True, True, True)
    # JxX in the residual precision: the Leray projection forms the force
    # as JxX - sigma, the small difference of two large fields, before it
    # rounds to the working dtype.
    JxX = seq.apply_inverse_mass_matrix(JxX_dual, 2, guess=JxH_guess, dtype=RESIDUAL_DTYPE)
    sigma_guess = None if F_guess is None else JxH_guess - F_guess
    F, p = seq.apply_leray_projection(JxX, k=2, p_guess=p_guess, sigma_guess=sigma_guess)
    return F, p, J, X, JxX.astype(DTYPE)


def weak_pressure(
    J: jnp.ndarray,
    X: jnp.ndarray,
    seq: DeRhamSequence,
    auxiliary_B_field: bool = False,
    p_guess: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The weak pressure ``p_w`` of the Lorentz force ``J x X`` and the weak force residual.

    ``compute_force`` projects ``J x H`` onto the Dirichlet 2-form space,
    which discards the wall-normal force ``(J x H) . n``, and its Leray
    multiplier ``p`` inherits ``dp/dn = 0`` on the wall. The weak pressure
    keeps that component: ``v = M_1^{-1} load(J x H)`` in the NATURAL
    1-form space (no boundary condition), then the Helmholtz decomposition
    ``v = F_w + grad p_w`` with ``p_w`` in the Dirichlet 0-form space,
    ``p_w = 0`` on the wall (``apply_leray_projection(k=1,
    dirichlet_p=True)``: one k=0 Dirichlet Laplacian solve). ``F_w`` is
    weakly divergence-free in the interior and keeps its normal trace, so
    on the wall ``(J x H) . n = dp_w/dn + F_w . n``; at a fixed point of the
    relaxation, where ``J x H`` is a gradient, ``F_w`` vanishes and
    ``dp_w/dn`` is the wall force.

    ``J`` and ``X`` are ``compute_force``'s (``J`` a Dirichlet 1-form, ``X``
    the 1-form ``H`` with ``auxiliary_B_field`` and the 2-form ``B``
    without), so the current is not recomputed. Costs two natural k=1 mass
    solves and the k=0 solve.

    Returns:
        ``(p_w, F_w, v)``: the Dirichlet 0-form DoFs of the weak pressure,
        the weak force residual and the natural 1-form projection of
        ``J x H``, both in the natural 1-form space.
    """
    v_dual = seq.cross_product_load(J, X, 1, 1, 1 if auxiliary_B_field else 2, False, True, True)
    v = seq.apply_inverse_mass_matrix(v_dual, 1, dirichlet=False)
    F_w, p_w = seq.apply_leray_projection(v, k=1, p_guess=p_guess, dirichlet_p=True)
    return p_w, F_w, v


def _wall_normal_component(a_w: jnp.ndarray, G_inv: jnp.ndarray) -> jnp.ndarray:
    """``a . n`` of covariant components ``a_w`` at wall points with inverse metric ``G_inv``.

    The unit normal of the surface ``r = const`` is ``n^i = g^{ir} / sqrt(g^{rr})``.
    """
    return jnp.einsum('qij,qj->qi', G_inv, a_w)[:, 0] / jnp.sqrt(G_inv[:, 0, 0])


def pressure_diagnostics(
    B: jnp.ndarray,
    p: jnp.ndarray,
    p_w: jnp.ndarray,
    F_w: jnp.ndarray,
    v: jnp.ndarray,
    seq: DeRhamSequence,
) -> dict[str, jnp.ndarray]:
    """Scalars comparing the strong pressure ``p`` with the weak one ``p_w``, and the plasma beta.

    ``p`` is ``compute_force``'s 3-form multiplier, ``(p_w, F_w, v)`` are
    :func:`weak_pressure`'s. Every entry is a scalar:

    - ``gradp_cmp``: ``||Pi_2 grad p_w - grad_w p||_{M_2} / ||Pi_2 grad p_w||_{M_2}``,
      gauge-free. ``grad_w p`` is the weak gradient of the 3-form in the
      Dirichlet 2-form space, the ``sigma`` the Leray step subtracts: the
      L2 projection of the true gradient onto that space, so its normal
      trace is zero whatever ``dp/dn`` is. ``grad p_w`` is the exact strong
      gradient of the 0-form (incidence matrix, natural 1-form space),
      projected onto the same space, ``Pi_2 = M_2^{-1} P_{12}``, so that both
      sides lose the same normal trace and the ratio compares the pressures,
      not the projection. (Comparing against ``grad p_w`` unprojected reads
      0.6 for IDENTICAL pressures on the (4,6,4) torus: the wall layer.)
    - ``p_cmp``: ``||(p/J - <p/J>) - (p_w - <p_w>)||_{L2} / ||p_w - <p_w>||_{L2}``
      at the quadrature points, ``<.>`` the volume mean: the pressures as
      functions, the strong one's gauge removed.
    - ``weak_resid``: ``||F_w||_{M_1} / ||v||_{M_1}``, the part of ``J x H``
      that is not a gradient of a function vanishing on the wall.
    - ``dpdn_wall``: ``max |dp_w/dn|`` over the wall, sampled at the angular
      quadrature points of ``r = 1``, relative to ``max |grad p_w|`` over
      the quadrature points. ``p_w = 0`` on the wall, so its gradient there
      is purely normal.
    - ``JxBn_wall``: ``max |(J x H) . n|`` on the same wall points, from
      ``v``, relative to the same ``max |grad p_w|``: the wall force.
      ``(J x H) . n = dp_w/dn + F_w . n`` pointwise.
    - ``beta_vol``: ``<p_w, 1>_{M_0} / E`` with ``E = B^T M_2 B / 2 = int B^2/2 dV``,
      the magnetic energy; code units, magnetic pressure ``B^2/2``, so
      ``beta = int p dV / int B^2/2 dV``.
    - ``beta_axis``: ``<p_w> / <|B|^2/2>`` on the COORDINATE axis, logical
      ``r = 0``: both averaged over the innermost radial quadrature layer
      (``r = x_r[0]``, a few percent of the first knot span, all theta and
      zeta), where the mass matrix reads the field. The 2-form's magnitude
      ``B_ref^T G B_ref / J^2`` is 0/0 on the polar axis itself, and the
      polar 2-form space does not pin ``B_ref(0)`` to zero, so a limit
      ``r -> 0`` reads the solver's residual there.
    """
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import map_jacobian_at

    wJ = seq.quad.w * seq.jacobian_j

    # (a) the gauge-free comparisons: gradients in the Dirichlet 2-form
    # space, and the functions at the quadrature points with the means removed.
    gpw = seq.apply_incidence_matrix(p_w, 0, dirichlet_in=True, dirichlet_out=False)
    gpw2 = seq.apply_inverse_mass_matrix(
        seq.apply_projection_matrix(gpw, 1, 2, dirichlet_in=False, dirichlet_out=True), 2)
    gp = seq.apply_weak_grad(p, True)
    gradp_cmp = seq.l2_norm(gpw2 - gp, 2) / seq.l2_norm(gpw2, 2)
    pw_q = seq.evaluate_at_quadrature(p_w, 0, True)[:, 0]
    p_q = seq.evaluate_at_quadrature(p, 3, True)[:, 0] / seq.jacobian_j
    pw_c = pw_q - jnp.sum(wJ * pw_q) / jnp.sum(wJ)
    p_c = p_q - jnp.sum(wJ * p_q) / jnp.sum(wJ)
    p_cmp = jnp.sqrt(jnp.sum(wJ * (p_c - pw_c) ** 2) / jnp.sum(wJ * pw_c ** 2))
    weak_resid = seq.l2_norm(F_w, 1, dirichlet=False) / seq.l2_norm(v, 1, dirichlet=False)

    # (b) the wall: |grad p_w| over the quadrature points, the normal
    # components at r = 1 over the angular quadrature points.
    gpw_q = seq.evaluate_at_quadrature(gpw, 1, False)
    grad_max = jnp.sqrt(jnp.max(jnp.einsum('qi,qij,qj->q', gpw_q, seq.metric_inv_jkl, gpw_q)))
    th, ze = jnp.meshgrid(seq.quad.x_y, seq.quad.x_z, indexing='ij')
    x_wall = jnp.stack([jnp.ones_like(th).ravel(), th.ravel(), ze.ravel()], axis=1)
    DF_w = map_jacobian_at(seq.map, x_wall)
    G_inv_w = jnp.linalg.inv(jnp.einsum('qki,qkj->qij', DF_w, DF_w))
    gpw_w = jax.vmap(DiscreteFunction(gpw, seq.basis_1, seq.E(1)))(x_wall)
    v_w = jax.vmap(DiscreteFunction(v, seq.basis_1, seq.E(1)))(x_wall)
    dpdn_wall = jnp.max(jnp.abs(_wall_normal_component(gpw_w, G_inv_w))) / grad_max
    JxBn_wall = jnp.max(jnp.abs(_wall_normal_component(v_w, G_inv_w))) / grad_max

    # (c) beta_vol: <p_w, 1>_{M_0} is the quadrature sum of p_w J.
    energy = 0.5 * seq.l2_norm_sq(B, 2)
    beta_vol = jnp.sum(wJ * pw_q) / energy

    # (d) beta_axis: the innermost radial quadrature layer, theta- and
    # zeta-averaged with the quadrature weights (the layer's own measure).
    B_q = seq.evaluate_at_quadrature(B, 2, True)
    Bsq_q = jnp.einsum('qi,qij,qj->q', B_q, seq.metric_jkl, B_q) / seq.jacobian_j ** 2
    axis = seq.quad.x[:, 0] == seq.quad.x_x[0]
    w_axis = jnp.where(axis, wJ, 0.0)
    beta_axis = jnp.sum(w_axis * pw_q) / jnp.sum(w_axis * 0.5 * Bsq_q)

    return dict(gradp_cmp=gradp_cmp, p_cmp=p_cmp, weak_resid=weak_resid, dpdn_wall=dpdn_wall,
                JxBn_wall=JxBn_wall, beta_vol=beta_vol, beta_axis=beta_axis)


def resistive_step(B: jnp.ndarray, seq: DeRhamSequence, eps, B_ref: Optional[jnp.ndarray] = None,
                   guess: Optional[jnp.ndarray] = None):
    """One backward-Euler step of ``dB/dt = -eta curl curl B`` over ``dt``,
    ``eps = eta dt``, in defect form: ``(M_2 + eps L_2) delta = -eps L_2 B``
    and ``B + delta``. With ``B_ref`` the step diffuses ``B - B_ref`` only,
    ``dB/dt = -eta curl (curl B - J_ref)``: Ohm's law with the source current
    of ``B_ref``. Solving for the increment keeps the step meaningful in
    float32 (the solution is ``B`` plus something small, not something that
    happens to be close to ``B``). Returns ``(B + delta, info, ||delta||_M
    / ||B||_M)`` with ``info`` the solver's signed iteration count.
    :func:`relax` applies this solve between chunks (``reconnect_every``),
    a dose ``eps`` per reconnection; a :class:`TimeStepper` with
    ``resistivity`` after every step."""
    rhs = -eps * seq.apply_laplacian(B if B_ref is None else B - B_ref, 2, dirichlet=True)
    delta, info = seq.apply_inverse_mass_plus_eps_laplace_matrix(
        rhs, 2, eps, dirichlet=True, guess=guess, return_info=True)
    rel = seq.l2_norm(delta, 2) / seq.l2_norm(B, 2)
    return B + delta, info.astype(jnp.int32), rel


def knot_spacing(seq: DeRhamSequence) -> np.ndarray:
    """The smallest knot spacing of each logical direction, shape ``(3,)``."""
    h = []
    for b in seq.basis_0.bases[0].bases:
        knots = np.asarray(b.T)
        interior = knots[b.p:-b.p] if b.type in ('clamped', 'periodic') else knots
        h.append(np.diff(interior).min())
    return np.array(h)


def radial_cell_sq(seq: DeRhamSequence) -> float:
    """The squared physical radial cell, ``h_r^2 = <g_rr>_V dr^2``: the
    radial knot spacing ``dr`` times the volume mean of the covariant
    metric's ``rr`` entry (``a^2`` for a circular torus of minor radius
    ``a``). The unit of the resistive doses: ``eps = C h_r^2`` damps the
    radial two-cell mode by ``1 / (1 + C pi^2)`` on any geometry; a bare
    ``C / n_r^2`` would carry the device's ``a^2`` (0.106 m^2 on li383)."""
    wJ = np.asarray(seq.quad.w * seq.jacobian_j)
    g_rr = np.asarray(seq.metric_jkl)[:, 0, 0]
    return float(knot_spacing(seq)[0] ** 2 * np.sum(wJ * g_rr) / np.sum(wJ))


def logical_cfl_weights(seq: DeRhamSequence) -> jnp.ndarray:
    """Weights ``1 / (J h_i)`` turning 2-form values at the quadrature points into logical CFL numbers.

    A 2-form velocity has reference components ``u_ref^i = J xi_dot^i``, so
    ``|u_ref^i| / (J h_i)`` is the number of logical cells of width ``h_i``
    (the knot spacing of direction ``i``) the flow crosses per unit time.
    The theta weight is zero inside the first radial span: the theta cell
    degenerates at the polar axis, where the polar space resolves nothing
    in theta. Returns an array of shape ``(n_q, 3)``; ``TimeStepper`` builds
    it once at construction (everything it reads is fixed by the sequence),
    so it is never traced.
    """
    h = knot_spacing(seq)
    weights = 1.0 / (np.asarray(seq.jacobian_j)[:, None] * h[None, :])
    weights[:, 1] *= np.asarray(seq.quad.x[:, 0]) >= h[0]
    return jnp.asarray(weights, dtype=DTYPE)


# %%


class WarmStarts(eqx.Module):
    """The solutions of one step's Krylov solves, the warm starts of the next.

    ``p``: the Leray multiplier (the strong pressure); ``H``: the auxiliary
    1-form ``M_1^-1 P B`` (zeros without ``auxiliary_B_field``; the Dirichlet
    proxy ``H_D`` under the helicity correction); ``JxH``: the unprojected
    force; ``J``: the weak curl; ``E``: the induction field; ``a``: the
    Dirichlet 1-form potential of the Newton direction (``u = curl a``) or
    of the force on the potential route (``F = curl a + c h``), zeros on the
    plain Leray route; ``A``: the potential of the helicity, refreshed by the
    sampler; ``resistive_delta``: the last resistive increment
    (``TimeStepper.resistivity``), zeros without.
    """
    p: jnp.ndarray
    H: jnp.ndarray
    JxH: jnp.ndarray
    J: jnp.ndarray
    E: jnp.ndarray
    a: jnp.ndarray
    A: jnp.ndarray
    resistive_delta: jnp.ndarray


class LastStep(eqx.Module):
    """What the last step computed, for the trace and the next step's force.

    ``F``, ``F_norm``: the Leray-projected force at the step's start field
    and ``||F||_M`` (the warm start of the Leray saddle's lower block, the
    ``Fu`` pairing of the trace); ``v``, ``v_norm``: the velocity and
    ``||u||_M``; ``newton_it``: the signed MINRES count of the Newton solve
    (negative when it met ``newton_tol``; 0 without ``newton``);
    ``helicity_lambda``: the multiple of ``H_D`` the helicity correction
    removed from ``E`` (0 without); ``resistive_it``, ``resistive_moved``:
    the resistive solve's signed iteration count and ``||delta||_M /
    ||B||_M`` (in mixed precision an increment near 1e-7 of the field is
    rounded away in ``B + delta``).
    """
    F: jnp.ndarray
    F_norm: jnp.ndarray
    v: jnp.ndarray
    v_norm: jnp.ndarray
    newton_it: jnp.ndarray
    helicity_lambda: jnp.ndarray
    resistive_it: jnp.ndarray
    resistive_moved: jnp.ndarray


class BestState(eqx.Module):
    """The field with the lowest squared normalised force residual the run
    has seen, that residual, and the absolute step it was at (the start
    field until a step beats it). A run past its floor (the residual is not
    monotone, and past the resolved floor the ideal descent raises it)
    returns this as its answer; ``chunk_runner`` keeps it,
    ``scripts/relax.py`` writes it as ``checkpoints/best.h5``."""
    B: jnp.ndarray
    resid: jnp.ndarray
    step: jnp.ndarray


class State(eqx.Module):
    """The descent state: the carry of :func:`chunk_runner`'s scan, one
    checkpoint file (:func:`write_checkpoint`, every leaf a dataset named
    ``B_n``, ``warm.p``, ``last.F``, ``best.B``, ...).

    ``B_n``, ``B_nplus1``: the field at the current and the next step;
    ``dt``: the step taken, ``min(dt_star, cfl / cfl_max)``; ``dt_star``: the
    uncapped step, the line-search minimiser; ``cfl_max``: the largest
    logical CFL number of the velocity, ``max_i max_q |u_ref^i| / (J h_i)``
    (:func:`logical_cfl_weights`); ``warm``: :class:`WarmStarts`; ``last``:
    :class:`LastStep`; ``best``: :class:`BestState`. Every leaf is an array
    of the working dtype (:func:`initial_state`). A scheme with a state of
    its own (the Picard iteration of :mod:`mrx.experimental.midpoint`) adds
    a subtree.
    """
    B_n: jnp.ndarray
    B_nplus1: jnp.ndarray
    dt: jnp.ndarray
    dt_star: jnp.ndarray
    cfl_max: jnp.ndarray
    warm: WarmStarts
    last: LastStep
    best: BestState


class Increment(NamedTuple):
    """The ideal increment ``dB = curl(u x H)`` at one field, with what the
    step computed on the way: the force and direction it stores, and the
    solutions that warm-start the next evaluation's five Krylov solves."""
    dB: jnp.ndarray
    u: jnp.ndarray
    Mu: jnp.ndarray
    F: jnp.ndarray
    MF: jnp.ndarray
    p: jnp.ndarray
    H: jnp.ndarray
    JxH: jnp.ndarray
    J: jnp.ndarray
    E: jnp.ndarray
    cfl_max: jnp.ndarray
    a: jnp.ndarray
    newton_it: jnp.ndarray


#: The velocity smoothing scale in squared physical radial cells:
#: ``mu = SMOOTHING_C h_r^2`` (:func:`radial_cell_sq`) damps a mode of
#: wavenumber ``k`` by ``1 / (1 + mu k^2)``, the radial two-cell mode by
#: ``1 / (1 + SMOOTHING_C pi^2)`` = 1/1.7 on any device. Swept 2026-09-05 on
#: li383 (16,32,32) p=2 in mixed precision, 5000 steps, as ``c / n_r^2``
#: with c in {0.0064, 0.02, 0.064, 0.2, 0.64} against no smoothing
#: (``outputs/mu_sweep``): the residual per step has a flat optimum over
#: 0.02-0.064, per wall second 0.02 is the cheapest of them (the shifted
#: solve's cost grows with the scale), the helicity drift is the same for
#: every smoothed arm (it is the time discretisation's, not the smoother's),
#: and the unsmoothed descent is a factor 2 behind at equal wall time. The
#: bare ``c / n_r^2`` carried li383's metric: 0.02 / n_r^2 is 0.074 h_r^2
#: there (<g_rr> = 0.208 m^2, knot spacing 1 / (n_r - p)), rounded to 0.075
#: (2026-09-18); the logical sweep's other values are 3.68 c.
SMOOTHING_C = 0.075


def smoothing_scale(seq) -> float:
    """``SMOOTHING_C h_r^2``: the smoothing scale of the sequence's mesh."""
    return SMOOTHING_C * radial_cell_sq(seq)


class TimeStepper(eqx.Module):
    """One step of the energy descent, ``B_{n+1} = B_n + dt curl(u x X)``.

    Force and descent direction (the smoothed force, or Newton's), velocity
    smoothing, the analytic line search with its CFL cap, and the
    induction, forward Euler. The step is ideal unless ``resistivity``
    adds a :func:`resistive_step` after it; reconnection between chunks is
    :func:`relax`'s (``reconnect_every``).

    Attributes:
        seq: The de Rham sequence.
        auxiliary_B_field: False (the default) reads the 2-form ``B`` itself
            in both cross products, ``J x B`` and ``u x B``. True routes
            them through the auxiliary Dirichlet 1-form ``H = M_1^-1 P B``,
            ``J x H`` and ``u x H``, at one extra k=1 mass solve per force
            evaluation and ``H_t = 0`` on the wall (the variable of the
            midpoint scheme, :mod:`mrx.experimental.midpoint` since 2026-09-20).
        velocity_smoothing_order: Number of smoothing solves applied to the
            descent direction, ``v = (I - scale * Laplacian)^-order F``;
            1 is the default. 0 leaves the direction as it is -- and is fragile:
            the explicit step with the unsmoothed velocity stops conserving
            helicity after ~1e4 steps at (16,32,32) p=2 on li383 (the drift
            grows a hundredfold, the field reconnects numerically, the
            energy release accelerates and the force residual climbs), in
            float64 as in float32; order 1 holds the drift at 5e-8 over the
            same run (docs/research/floor_study_2026-09-05.md). Use order 1
            for any long ideal run; order 0 only for short smoke runs.
        velocity_smoothing_scale: Length scale of the smoothing,
            the ``mu`` in ``(M_2 + mu L_2)^-1 M_2``; ``None`` (the default)
            is :func:`smoothing_scale`, ``SMOOTHING_C h_r^2``.
        cfl: Cap on the step: ``dt = min(dt_star, cfl / cfl_max)`` with
            ``cfl_max`` the largest logical CFL number of the velocity. The
            linesearch minimiser cannot raise the energy, but a large step
            leaves the ideal-induction flow (frozen-in topology violated at
            O(dt^2)) and diverges when ``||dB||`` collapses. ``inf`` disables
            the cap and leaves the trajectory untouched.
        potential_velocity: Compute the projected force as ``F = curl a +
            c h`` instead of by the Leray saddle solve: ``a`` from the k=1
            Hodge Laplacian solve of ``curl^T load(J x B)`` (the curl-curl
            equation in the Coulomb gauge; the gradient part of ``J x B`` is
            annihilated by ``curl^T`` exactly) and ``c = (J x B, h) / (h,
            h)`` on the harmonic 2-form ``h``. ``F`` is divergence-free to
            roundoff. ``None`` (the default) is ``True`` unless ``newton`` or
            ``auxiliary_B_field`` is set, which have their own routes;
            ``True`` with either raises. The velocity smoothing acts on the potential through
            the k=1 shifted solve, ``curl (M_1 + mu L_1)^-1 M_1 a =
            (M_2 + mu L_2)^-1 M_2 curl a`` exactly. No pressure comes out of it; ``State.p`` keeps the
            sampler's. Excludes ``newton`` and the auxiliary field.
        newton: Replace the smoothed-force direction by the Newton direction
            of the second variation, ``u = curl a`` with ``curl^T H curl a =
            curl^T M_2 F`` solved by Newton-MR (:mod:`mrx.hessian`: MINRES
            with the harmonic atom of the current field, the parallel-flow
            penalty in the operator, the nonpositive-curvature exit). The
            Hessian reads the 2-form ``B``; the force and the induction
            follow ``auxiliary_B_field`` as usual. The line search along the
            direction is capped at the Newton length ``dt = 1``: a truncated
            direction mixes resolved modes, whose energy minimum is at the
            Newton step, with unresolved flat ones that want a longer step,
            and the exact line search settles near 2, where the resolved
            modes' residual is reflected rather than removed (measured 2026-09).
        newton_penalty: ``kappa`` of the parallel-flow penalty, ``kappa``
            times the strain along the field (:func:`mrx.hessian.parallel_penalty_profile`):
            the one number of the Newton configuration, 3 on every case
            measured (docs/research/hessian_spectrum_2026-09-17.md).
        newton_tol: The forcing term of the Newton solve: the residual of the
            Newton system, in the residual precision and the mass-atom norm,
            below ``newton_tol`` of the right-hand side ends it.
        newton_maxiter: MINRES iterations per pass of the Newton solve.
        newton_passes: Passes of ``newton_maxiter`` iterations at most; the
            solve is inexact by design (one pass of 200 gives the same
            relaxation as any tighter solve, 2026-09-18).
        helicity_correction: Remove from the induction field ``E`` the
            one component that changes the discrete helicity. Over one
            step ``B_{n+1} = B_n + dt curl E`` the helicity ``<A, B +
            B_harm>`` of :func:`compute_helicity` changes by exactly ``2 dt
            <E, P B_n> + dt^2 <E, P curl E>`` (the discrete Stokes identity
            is exact with ``A`` and ``E`` Dirichlet, ``B_harm`` does not
            move), and ``<E, P B> = <E - u x B, H_w>`` is the pairing of
            the projection residual of ``u x B`` with the tangential wall
            DoFs of the natural proxy of ``B``: the leak of the plain-``B``
            route (``docs/research/implicit_midpoint_2026-09-04.md``). With
            ``E - lambda H_D``, ``H_D = M_1^-1 P B`` the Dirichlet proxy,
            one scalar zeroes the change, the root near zero of the
            quadratic in ``lambda``. The pairings are formed
            in the residual precision (a small total of large terms). The
            helicity is then flat to the solves and the field's own
            rounding, with ``H`` natural and no wall layer; the induction
            picks up ``-lambda curl H_D``, of the size of the leak, and the
            energy decrease is perturbed by ``lambda`` times the ``J . B``
            pairing: not variational. ``State.helicity_lambda`` records it.
            Not covered by the test suite on purpose (2026-09-20): the
            option is off by default and its 20-step run was 130 s of
            compile per configuration; ``docs/research/helicity_correction_2026-09-11.md``
            has the measurements.
        resistivity: The resistive dose per step, ``eps = eta dt`` (a length
            squared; ``scripts/relax.py --resistivity C`` gives ``C h_r^2``,
            :func:`radial_cell_sq`), 0 for the ideal descent. After the ideal
            step, one backward-Euler step of ``dB/dt = -eta curl (J - J_ref)``,
            ``(M_2 + eps L_2) delta = -eps L_2 (B - B_ref)``: two SPD PCG solves
            with the shifted-stiffness atoms
            (:meth:`DeRhamSequence.apply_inverse_mass_plus_eps_laplace_matrix`;
            the right-hand side is a curl, so the gradient solve is trivial),
            warm-started from the previous step's increment. Every step, not
            between chunks: the fixed point is the resistive steady state (the
            descent flow balancing the diffusion, a force residual of order
            ``eps``) instead of the sawtooth of solves between blocks of ideal
            steps. Switching it off afterwards relaxes that state ideally, the
            islands it opened frozen in.
        resistive_reference: ``B_ref``, the field whose current is the source
            ``J_ref`` (``None``: no source, ``J_ref = 0``). It must not carry the
            rational-surface sheets of an ideal equilibrium, or they are
            sustained and a run started from it is a fixed point.
        cfl_weights: ``logical_cfl_weights(seq)``, built by ``__post_init__``.
    """
    seq: DeRhamSequence
    auxiliary_B_field: bool = False
    velocity_smoothing_order: int = 1
    velocity_smoothing_scale: float = None
    cfl: float = 0.5
    potential_velocity: bool = None
    newton: bool = False
    newton_penalty: float = NEWTON_PENALTY
    newton_tol: float = NEWTON_TOL
    newton_maxiter: int = NEWTON_MAXITER
    newton_passes: int = NEWTON_PASSES
    helicity_correction: bool = False
    resistivity: float = 0.0
    resistive_reference: Optional[jnp.ndarray] = None
    cfl_weights: jnp.ndarray = None
    harmonic: jnp.ndarray = None
    harmonic_norm_sq: jnp.ndarray = None

    def __post_init__(self):
        if self.potential_velocity is None:
            self.potential_velocity = not (self.newton or self.auxiliary_B_field)
        if self.potential_velocity and (self.newton or self.auxiliary_B_field):
            raise ValueError("potential_velocity is the Leray route's replacement on the 2-form B: "
                             "it excludes newton and the auxiliary field.")
        if self.potential_velocity:
            h = self.seq.nullspace(2, True)[0]
            self.harmonic = h
            self.harmonic_norm_sq = h @ self.seq.apply_mass_matrix(h, 2)
        if self.velocity_smoothing_scale is None:
            self.velocity_smoothing_scale = smoothing_scale(self.seq)
        self.cfl_weights = logical_cfl_weights(self.seq)

    def smooth_velocity(self, u: jnp.ndarray) -> jnp.ndarray:
        """Apply ``(M_2 + scale L_2)^-1 M_2`` to ``u`` ``velocity_smoothing_order`` times."""
        for _ in range(self.velocity_smoothing_order):
            rhs = self.seq.apply_mass_matrix(u, 2, True)
            u = self.seq.apply_inverse_mass_plus_eps_laplace_matrix(
                rhs, 2, self.velocity_smoothing_scale, dirichlet=True, guess=u)
        return u

    def _helicity_proxy(self, B: jnp.ndarray, X: jnp.ndarray, H_guess: jnp.ndarray):
        """``(P B, H_D)`` of :func:`dirichlet_proxy`, ``H_D`` being ``X`` itself
        on the auxiliary route (no solve)."""
        if self.auxiliary_B_field:
            return self.seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True), X
        return dirichlet_proxy(self.seq, B, H_guess)

    def _helicity_lambda(self, E: jnp.ndarray, PB: jnp.ndarray, H_D: jnp.ndarray, dt) -> jnp.ndarray:
        """The multiple of ``H_D`` whose removal from ``E`` zeroes the step's
        helicity change (``helicity_correction``): the root near zero of ``2
        <E_l, P B> + dt <E_l, P curl E_l> = 0``, ``E_l = E - lambda H_D``.
        Every pairing in the residual precision."""
        on = self.seq if self.seq.residual is None else self.seq.residual
        E, PB, H_D = (x.astype(RESIDUAL_DTYPE) for x in (E, PB, H_D))

        def P_curl(y):
            return on.apply_projection_matrix(
                on.apply_incidence_matrix(y, 1, dirichlet_in=True, dirichlet_out=True),
                2, 1, True, dirichlet_out=True)
        PDE, PDH = P_curl(E), P_curl(H_D)
        a = dt * (H_D @ PDH)
        b = -2.0 * (H_D @ PB) - 2.0 * dt * (E @ PDH)
        c = 2.0 * (E @ PB) + dt * (E @ PDE)
        return 2.0 * c / (-b + jnp.sqrt(b * b - 4.0 * a * c))

    def _induction_field(self, u_jk: jnp.ndarray, X: jnp.ndarray, E_guess: jnp.ndarray) -> jnp.ndarray:
        """``E = M_1^-1 load(u x X)`` with ``u`` at the quadrature points and
        ``X`` the auxiliary 1-form ``H`` or the 2-form ``B`` itself."""
        seq = self.seq
        k = 1 if self.auxiliary_B_field else 2
        X_jk = seq.evaluate_at_quadrature(X, k, True)
        E_dual = seq.cross_product_load_values(u_jk, X_jk, 1, 2, k, True, parity=-1)
        return seq.apply_inverse_mass_matrix(E_dual, 1, guess=E_guess)

    def _potential_force(self, B: jnp.ndarray, a_guess: jnp.ndarray, J_guess: jnp.ndarray):
        """The projected force as ``F = curl a + c h`` and its smoothed version.

        ``a`` solves ``L_1 a = curl^T load(J x B)`` (the k=1 Hodge split,
        warm-started from ``a_guess``): the right-hand side is orthogonal
        to gradients, so ``a`` is in the Coulomb gauge and ``curl a`` is the
        exact part of the Leray projection of ``J x B``; ``c h`` is its
        harmonic part. The smoothing solves ``(M_1 + mu L_1) a_s = M_1 a``
        ``velocity_smoothing_order`` times. Returns ``(F, M F, F_s, J, a)``.
        """
        seq = self.seq
        J = seq.apply_weak_curl(B, dirichlet=True, guess=J_guess)
        JxB_dual = seq.cross_product_load(J, B, 2, 1, 2, True, True, True)
        rhs = seq.apply_incidence_matrix(JxB_dual, 1, dirichlet_in=True, dirichlet_out=True,
                                         transpose=True)
        a = seq.apply_inverse_laplacian(rhs, 1, dirichlet=True, guess=a_guess)
        ch = ((self.harmonic @ JxB_dual) / self.harmonic_norm_sq) * self.harmonic
        F = seq.apply_incidence_matrix(a, 1, dirichlet_in=True, dirichlet_out=True) + ch
        a_s = a
        for _ in range(self.velocity_smoothing_order):
            a_s = seq.apply_inverse_mass_plus_eps_laplace_matrix(
                seq.apply_mass_matrix(a_s, 1, True), 1, self.velocity_smoothing_scale,
                dirichlet=True, guess=a_s)
        Fs = seq.apply_incidence_matrix(a_s, 1, dirichlet_in=True, dirichlet_out=True) + ch
        return F, seq.apply_mass_matrix(F, 2), Fs, J, a

    def _ideal_increment(self, B: jnp.ndarray, state: State,
                         p_guess: jnp.ndarray,
                         H_guess: jnp.ndarray, JxH_guess: jnp.ndarray,
                         J_guess: jnp.ndarray, E_guess: jnp.ndarray) -> Increment:
        """The ideal increment ``dB = curl(u x X)`` evaluated at the field ``B``.

        Force, descent direction (the smoothed force, or Newton's), velocity
        smoothing, the cross product and the topological curl. The velocity
        is divergence-free without a projection of its own: the force is
        Leray-projected and the smoothing ``(M + mu L)^-1 M`` commutes with
        the divergence; a second Leray projection of the velocity was
        measured to change nothing in float64 and in mixed precision and
        to cost 1.5-4x the step (``docs/research/velocity_leray_ab_2026-09-04.md``;
        in float32 with a solve tolerance relative to ``J x B`` it was what
        kept the step a descent). The five
        guesses, and ``state.last.F`` for the gradient part of the force,
        warm-start the Krylov solves;
        they come from ``state`` (the previous step). Without the auxiliary
        field ``H_guess`` passes through untouched.
        """
        seq = self.seq
        newton_it = jnp.int32(0)
        if self.potential_velocity:
            F, MF, u, J, a = self._potential_force(B, state.warm.a, J_guess)   # u the smoothed force
            p, X, JxX = p_guess, B, JxH_guess     # not computed on this route
        else:
            F, p, J, X, JxX = compute_force(
                B, seq, self.auxiliary_B_field,
                p_guess=p_guess, H_guess=H_guess, JxH_guess=JxH_guess,
                J_guess=J_guess, F_guess=state.last.F)
            # M F once: ||F||_M and the Newton right-hand side; the increment
            # applies M_2 twice in total (M F, M u).
            MF = seq.apply_mass_matrix(F, 2)
            if self.newton:
                u, a, newton_it = newton_direction(seq, B, J, MF, state.warm.a, self.newton_penalty,
                                                   self.newton_tol, self.newton_maxiter, self.newton_passes)
            else:
                # gradient descent on the smoothed force
                u, a = self.smooth_velocity(F), state.warm.a
        # M u once: the linesearch numerator and ||u||_M.
        Mu = seq.apply_mass_matrix(u, 2)

        # u at the quadrature points once: the cross product and the CFL
        # number both read it.
        u_jk = seq.evaluate_at_quadrature(u, 2, True)
        E = self._induction_field(u_jk, X, E_guess)
        cfl_max = jnp.max(jnp.abs(u_jk) * self.cfl_weights)

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
        dB = seq.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
        H = X if self.auxiliary_B_field else H_guess
        return Increment(dB, u, Mu, F, MF, p, H, JxX, J, E, cfl_max, a, newton_it)

    def _step_size(self, inc: Increment) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(dt, dt_star)``: the line-search step at ``inc``, its CFL cap and, under
        Newton, the cap at the Newton length.

        ``dt_star = <F, u>_M / ||dB||_M^2`` minimises the quadratic energy
        along the increment exactly (``dE = -dt <F, u>_M + dt^2 ||dB||^2 / 2``).
        The cap: ``cfl = inf`` gives ``min(dt_star, inf) = dt_star`` exactly.
        """
        slope, curvature = inc.F @ inc.Mu, self.seq.l2_norm_sq(inc.dB, 2)
        dt_star = slope / curvature
        # a non-positive dt* is no step: the energy does not decrease along
        # the increment; a negative step would climb it
        dt = jnp.minimum(jnp.maximum(dt_star, 0.0), self.cfl / inc.cfl_max)
        if self.newton:
            dt = jnp.minimum(dt, 1.0)              # the Newton length
        return dt, dt_star

    def relaxation_step(self, state: State) -> State:
        """Advance ``state.B_n`` by one ideal step into ``state.B_nplus1``:
        forward Euler on the descent velocity, ``B_n + dt curl(u x X)``."""
        B_n = state.B_n
        lam = jnp.zeros((), B_n.dtype)
        inc = self._ideal_increment(B_n, state, state.warm.p, state.warm.H, state.warm.JxH,
                                    state.warm.J, state.warm.E)
        dt, dt_star = self._step_size(inc)
        if self.helicity_correction:
            PB, H_D = self._helicity_proxy(B_n, inc.H, state.warm.H)
            lam = self._helicity_lambda(inc.E, PB, H_D, dt).astype(B_n.dtype)
            E = inc.E - lam * H_D
            inc = inc._replace(E=E, H=H_D, dB=self.seq.apply_incidence_matrix(
                E, 1, dirichlet_in=True, dirichlet_out=True))
        B_nplus1 = B_n + dt * inc.dB

        res_delta, res_it, res_moved = state.warm.resistive_delta, state.last.resistive_it, state.last.resistive_moved
        if self.resistivity:
            B_ideal = B_nplus1
            B_nplus1, res_it, res_moved = resistive_step(B_ideal, self.seq, self.resistivity,
                                                 self.resistive_reference, guess=state.warm.resistive_delta)
            res_delta = B_nplus1 - B_ideal
            res_moved = res_moved.astype(state.last.resistive_moved.dtype)

        # The descent variable is the VELOCITY u, not B: grad_M E = -F is the
        # derivative of E with respect to u (dE = -(F, u)_M) and the line
        # search minimises along dt*u; B moves by dt*curl(u x H), a different
        # vector in a different space.
        warm = WarmStarts(p=inc.p, H=inc.H, JxH=inc.JxH, J=inc.J, E=inc.E, a=inc.a, A=state.warm.A,
                          resistive_delta=res_delta)
        last = LastStep(F=inc.F, F_norm=jnp.sqrt(inc.F @ inc.MF), v=inc.u, v_norm=jnp.sqrt(inc.u @ inc.Mu),
                        newton_it=inc.newton_it, helicity_lambda=lam, resistive_it=res_it,
                        resistive_moved=res_moved)
        return eqx.tree_at(lambda s: (s.B_nplus1, s.dt, s.dt_star, s.cfl_max, s.warm, s.last), state,
                           (B_nplus1, dt, dt_star, inc.cfl_max, warm, last))


def initial_state(B_dof: jnp.ndarray, ts: TimeStepper, dt: float = 1.0, step: int = 0) -> State:
    """Build the state at ``B_dof`` with its force already evaluated.

    ``last.F``, ``last.F_norm`` and the warm starts ``p``,
    ``H``, ``JxH``, ``J`` are seeded from one ``compute_force`` here, so the
    first step's solves start from the true previous force.
    Every leaf is an array of the working dtype, the scalars included: the
    state is the carry of :func:`chunk_runner`'s scan, and a Python-float
    leaf here against a float32 array out of the scan gave the scan two
    carry signatures, i.e. a second compile at the second chunk of every run
    (30-55 s, measured 2026-09-05). ``step`` is the absolute step the
    field is at (a restart's), the label of the best state until a step
    beats it.
    """
    seq = ts.seq
    n = seq.n(2, True)
    F0, p0, J0, X0, JxX0 = compute_force(B_dof, seq, ts.auxiliary_B_field)
    MF0 = seq.apply_mass_matrix(F0, 2)
    resid0 = (jnp.sqrt(F0 @ MF0) / force_scale_jit(seq, B_dof)) ** 2
    n1 = seq.n(1, True)
    zeros1 = jnp.zeros(n1, dtype=DTYPE)
    return State(
        B_n=B_dof, B_nplus1=B_dof,
        dt=jnp.asarray(dt, dtype=DTYPE), dt_star=jnp.asarray(dt, dtype=DTYPE),
        cfl_max=jnp.zeros((), dtype=DTYPE),
        warm=WarmStarts(p=p0, H=X0 if ts.auxiliary_B_field else zeros1, JxH=JxX0, J=J0, E=zeros1, a=zeros1,
                        A=zeros1, resistive_delta=jnp.zeros(n, dtype=DTYPE)),
        last=LastStep(F=F0, F_norm=jnp.sqrt(F0 @ MF0), v=jnp.zeros(n, dtype=DTYPE),
                      v_norm=jnp.zeros((), dtype=DTYPE), newton_it=jnp.int32(0),
                      helicity_lambda=jnp.zeros((), dtype=DTYPE), resistive_it=jnp.int32(0),
                      resistive_moved=jnp.zeros((), dtype=DTYPE)),
        best=BestState(B=B_dof, resid=jnp.asarray(resid0, dtype=DTYPE), step=jnp.int32(step)),
    )


#: The per-step scalars of :func:`chunk_runner`'s trace that :func:`relax`
#: keeps (``v`` and ``Fu`` only feed the derived ``cos``, ``gain`` and ``dE_ls``).
TRACE_COLUMNS = ("dE", "F", "resid", "dt", "dt_star", "cfl", "div", "newton_it", "hcorr", "res_it", "res_moved")


def chunk_runner(ts: TimeStepper, n_chunk: int) -> Callable[[State, int], tuple[State, dict]]:
    """``run(state, it0) -> (state, trace)``, jit-compiled: ``n_chunk``
    relaxation steps as one ``lax.scan``.

    The state (B, the force, the warm-start guesses) is the carry and
    comes out once; the per-step scalars are the scan's stacked output,
    ``trace[name]`` an array of length ``n_chunk`` over the steps
    ``it0 + 1 .. it0 + n_chunk``: ``dE`` (the step's change of the energy
    ``||B||_M^2 / 2``, exactly: ``<B_{n+1} - B_n, M (B_{n+1} + B_n)> / 2``,
    a small increment against an O(1) field at the increment's own
    precision -- the energy itself has none at the level of one step in
    float32), ``F`` (``||F||_M``), ``v`` (``||u||_M``), ``dt``,
    ``dt_star``, ``cfl`` (the velocity's largest logical CFL number),
    ``div`` (``||div B||``), ``Fu`` (``<F_prev, u>_M``: the line search
    predicts ``dE = -dt Fu (1 - dt / 2 dt_star)``),
    ``newton_it`` (the Newton solve's signed MINRES count; 0 without
    ``newton``),
    ``hcorr`` (the helicity correction's ``lambda``; 0 without it),
    ``resid`` (the squared normalised force residual ``||F||_M^2 /
    ||grad(B^2/2)||^2``, :func:`force_scale`, the force being the step's
    start field's and the scale the end field's). The body also keeps the best state: the start
    field of any step whose ``resid`` is below ``state.best.resid`` replaces
    ``state.best.B`` with its residual and absolute step.

    Compile time is the body's whatever ``n_chunk`` (a ``While`` trip
    count); the chunk is the cadence at which the host sees the trace and
    may act on the state.

    A PURE function of the stepper: ``ts`` (and through it the sequence,
    a pytree, :mod:`mrx.pytree`) is an argument of the jitted function, so
    the geometry, the element weights, the extraction tables and the atoms
    reach the program as device inputs, not as constants captured by a
    closure (3.5 GB of them at (48,96,96), 10.6 GB for the whole torus,
    constant-folded and held on the host through the compile). The step
    index is an array for the same reason: a Python int is static under
    ``filter_jit`` and would recompile every chunk.
    """
    def body(ts, state, it):
        seq = ts.seq
        state = ts.relaxation_step(state)
        B_n, B_new = state.B_n, state.B_nplus1
        dE = 0.5 * ((B_new - B_n) @ seq.apply_mass_matrix(B_new + B_n, 2))
        resid = (state.last.F_norm / force_scale(seq, B_new)) ** 2
        better = resid < state.best.resid
        best = BestState(B=jnp.where(better, B_n, state.best.B),
                         resid=jnp.where(better, resid, state.best.resid).astype(state.best.resid.dtype),
                         step=jnp.where(better, it - 1, state.best.step).astype(state.best.step.dtype))
        state = eqx.tree_at(lambda s: (s.B_n, s.best), state, (B_new, best))
        trace = dict(
            dE=dE, F=state.last.F_norm, v=state.last.v_norm,
            dt=state.dt, dt_star=state.dt_star, cfl=state.cfl_max,
            div=compute_divergence_norm(state.B_n, seq),
            Fu=state.last.F @ seq.apply_mass_matrix(state.last.v, 2),
            newton_it=state.last.newton_it, res_it=state.last.resistive_it, res_moved=state.last.resistive_moved,
            hcorr=state.last.helicity_lambda, resid=resid)
        return state, trace

    @eqx.filter_jit
    def run(ts, state, it0):
        return jax.lax.scan(lambda st, it: body(ts, st, it), state, it0 + jnp.arange(1, n_chunk + 1))

    return lambda state, it0: run(ts, state, jnp.asarray(it0))


# ---------------------------------------------------------------------------
# The run: the residual scale, the diagnostics sampler, checkpoints, the loop
# ---------------------------------------------------------------------------

def force_scale(seq: DeRhamSequence, B: jnp.ndarray) -> jnp.ndarray:
    """``||grad(B^2/2)||_L2`` of the 2-form DoFs ``B``: the scale the force
    residual is measured against (:func:`force_scale_jit` is its compiled form).

    ``grad p`` is a real scale too (the scheme converges to ``J x B = grad
    p``) but vanishes in the low-beta limit; ``grad(B^2/2)`` has the same
    units and stays O(1). Through the sequence: the 0-form load of
    ``B^2/2`` (:meth:`~mrx.derham_sequence.DeRhamSequence.magnitude_squared_load`),
    one natural ``M_0`` solve, the
    strong gradient, its norm.
    """
    q = 0.5 * seq.magnitude_squared_load(B)
    w0 = seq.apply_inverse_mass_matrix(q, 0, dirichlet=False)
    g1 = seq.apply_strong_grad(w0, dirichlet_in=False, dirichlet_out=False)
    return seq.l2_norm(g1, 1, dirichlet=False)


#: :func:`force_scale` compiled, the sequence an argument of the compiled function.
force_scale_jit = eqx.filter_jit(force_scale)


def make_sampler(seq: DeRhamSequence, ts: TimeStepper):
    """``sample(state, pw_guess, eager=False) -> (state, p_w, scalars)``: the
    diagnostics of a state's field.

    The force at the CURRENT field (``state.warm.p``, ``H``, ``JxH``, ``J``,
    ``F_prev`` are the step's values at the previous one; they warm-start it
    and are refreshed from it), the weak pressure and its diagnostics
    (:func:`pressure_diagnostics`), the helicity (``state.warm.A`` refreshed),
    ``||J|| / ||B||``, the pairing ``int J . B`` that sets a reconnection
    dose, and the energy ``E = <B, M B> / 2`` of the stored field in the
    residual precision (exact to the field's own rounding, 3e-8 on E = 0.5
    in float32 storage; the trace's per-step ``dE`` is formed in the
    working precision and its sum over 1e4 steps drifts by that much).
    ``scalars`` are Python floats. The first call of a
    run goes ``eager`` (the 1->2 projection builds a host-side core on
    first use); the loop uses the compiled one.
    """
    aux = ts.auxiliary_B_field
    on = seq if seq.residual is None else seq.residual      # the energy, in the residual precision

    def probe(seq, aux, B, p, H, JxH, J, F_prev, pw_guess, A):
        F, p, J, X, JxX = compute_force(B, seq, aux, p_guess=p, H_guess=H, JxH_guess=JxH,
                                        J_guess=J, F_guess=F_prev)
        p_w, F_w, v = weak_pressure(J, X, seq, aux, p_guess=pw_guess)
        diag = pressure_diagnostics(B, p, p_w, F_w, v, seq)
        h, A_new = compute_helicity(B, seq, A)
        JoverB = seq.l2_norm(J, 1) / seq.l2_norm(B, 2)
        JB = J @ seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True)
        return p, (X if aux else H), JxX, J, A_new, p_w, h, JoverB, JB, diag

    probe_jit = eqx.filter_jit(probe)      # the sequence an argument, not a captured constant

    def sample(state: State, pw_guess: jnp.ndarray, eager: bool = False):
        f = probe if eager else probe_jit
        p, H, JxH, J, A, p_w, h, JoverB, JB, diag = f(
            seq, aux, state.B_n, state.warm.p, state.warm.H, state.warm.JxH, state.warm.J, state.last.F, pw_guess, state.warm.A)
        state = eqx.tree_at(lambda s: (s.warm.p, s.warm.H, s.warm.JxH, s.warm.J, s.warm.A), state, (p, H, JxH, J, A))
        E = 0.5 * float(on.l2_norm_sq(state.B_n.astype(RESIDUAL_DTYPE), 2))
        scalars = dict(E=E, helicity=float(h), JoverB=float(JoverB), JB=float(JB),
                       **{k: float(v) for k, v in diag.items()})
        return state, p_w, scalars

    return sample


def write_checkpoint(path: str, state: State, step: int) -> None:
    """The state at a step as one HDF5 file: every leaf of the pytree as a
    dataset named by its key path (``B_n``, ``warm.p``, ``last.F``, ...), the step
    as an attribute. Nothing else: the run's parameters are the driver's
    ``relax.json``, and the weak pressure is a diagnostic
    (:func:`make_sampler`), not state."""
    import h5py  # noqa: PLC0415
    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    with h5py.File(path, "w") as fh:
        fh.attrs["step"] = int(step)
        for keypath, leaf in leaves:
            fh.create_dataset(jax.tree_util.keystr(keypath).lstrip("."), data=np.asarray(leaf))


def read_checkpoint(path: str, ts: TimeStepper) -> tuple[State, int]:
    """The ``(state, step)`` of :func:`write_checkpoint`, for a stepper of the
    same sequence: the skeleton comes from :func:`initial_state` on the
    stored field (one force evaluation), every leaf is then replaced by
    the stored one. A leaf the file does not have (a diagnostic added
    after the file was written; every warm start and diagnostic of a file
    written before the subtrees of 2026-09-20) keeps the skeleton's value."""
    import h5py  # noqa: PLC0415
    with h5py.File(path, "r") as fh:
        step = int(fh.attrs["step"])
        data = {k: np.asarray(v) for k, v in fh.items()}
    skeleton = initial_state(jnp.asarray(data["B_n"]), ts, step=step)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(skeleton)
    new = []
    for keypath, leaf in leaves:
        name = jax.tree_util.keystr(keypath).lstrip(".")
        if name not in data:
            new.append(leaf)
            continue
        v = data[name]
        new.append(jnp.asarray(v, dtype=DTYPE if np.issubdtype(v.dtype, np.floating) else v.dtype))
    return jax.tree_util.tree_unflatten(treedef, new), step


class RelaxResult(NamedTuple):
    """What :func:`relax` returns and hands to ``on_chunk`` at every chunk.

    ``state`` the descent state, ``steps`` the steps of this run so far
    (``it0 + steps`` is the absolute step), ``stop`` why it ended (``steps``,
    ``floor``, or ``running``), ``wall`` the seconds in the
    compiled steps (sampling and callbacks excluded), ``trace`` the per-step
    scalars (``dE`` the exact energy change of the step, ``dE_ls`` the line
    search's prediction ``-dt <F, u>_M (1 - dt / 2 dt_star)`` -- the two
    differ by ``-dt <u, grad p>_M``, zero for a divergence-free velocity
    --, ``F``, ``resid``, ``dt``, ``dt_star``, ``cfl``, ``div``, ``cos``,
    ``gain``), ``E0`` the energy at the
    start of the run (``E0 + cumsum(dE)`` is the trace's energy after every
    step, to the working precision's rounding per step), ``qoi`` the
    per-chunk samples (``it``, ``wall``, ``E`` the energy of the stored
    field in the residual precision, ``F``, ``resid``, ``helicity``,
    ``JoverB``, ``JB`` and the pressure
    diagnostics; the first entry is the start of the run, a reconnection
    adds a second sample at its step), ``reconnect`` one record per
    reconnection, ``reconnect_every`` the interval actually used (rounded to
    whole chunks), ``chunk`` the chunk length.
    """
    state: State
    steps: int
    stop: str
    wall: float
    trace: dict
    qoi: dict
    reconnect: list
    reconnect_every: int
    chunk: int
    E0: float


def pressure_line(d: dict) -> str:
    """One line of the pressure diagnostics of a sample."""
    return (f"beta_vol={d['beta_vol']:.3e}  beta_axis={d['beta_axis']:.3e}  "
            f"|grad pw - grad p|/|grad pw|={d['gradp_cmp']:.3e}  |pw - p|/|pw|={d['p_cmp']:.3e}  "
            f"weak_resid={d['weak_resid']:.3e}  "
            f"wall dpw/dn={d['dpdn_wall']:.3e}  (JxB).n={d['JxBn_wall']:.3e}")


def relax(state: State, ts: TimeStepper, steps: int, chunk: int = 500, it0: int = 0,
          floor_tol: float = 0.0,
          reconnect_every: int = 0, reconnect_helicity: float = 0.01,
          reconnect_eps: Optional[float] = None, reconnect_window: Optional[tuple] = None,
          on_chunk: Optional[Callable[[RelaxResult], None]] = None,
          verbose: bool = True) -> RelaxResult:
    """The relaxation run: ``steps`` steps in compiled chunks of ``chunk``
    (:func:`chunk_runner`), the diagnostics sampled once per chunk
    (:func:`make_sampler`), the stop tests and the reconnection series.

    Stops on the step count, on ``floor_tol`` (the last chunk's mean of
    the squared normalised force residual ``||F||_M^2 / ||grad(B^2/2)||^2``
    below it; the residual is not monotone, the window mean is the
    quantity); a job's
    time limit is no stop, the checkpoint of every chunk restarts it.
    ``reconnect_every`` (rounded to
    whole chunks, never on the last one) applies one :func:`resistive_step`
    to the field whose dose spends the fraction ``reconnect_helicity`` of
    its helicity, ``eps = X |H| / (2 |int J . B|)`` from ``dH = -2 eps int J
    . B``, or with ``reconnect_eps`` a constant dose ``eps`` per solve (a
    constant resistivity: the ideal relaxation is fast and the diffusion slow,
    so the solves go between blocks of ideal steps and the field is back in
    equilibrium before the next one; the helicity spent is then an outcome),
    only at steps inside ``reconnect_window = (start, stop)`` when given,
    then restarts the optimiser on the diffused field
    (:func:`initial_state`) and samples it again; ``on_chunk`` runs after
    every chunk's sample and BEFORE a reconnection at that step, so what it
    saves is the field the solve starts from. ``it0`` is the absolute step
    the run starts at (a restart); the trace and samples are this run's.
    """
    if chunk < 1 or steps % chunk:
        raise ValueError("steps must be a positive multiple of chunk")
    if reconnect_every:
        reconnect_every = max(1, round(reconnect_every / chunk)) * chunk
    seq = ts.seq
    run = chunk_runner(ts, chunk)
    sample = make_sampler(seq, ts)
    reconnect_jit = eqx.filter_jit(lambda sq, B, eps: resistive_step(B, sq, eps))
    reconnect_fn = lambda B, eps: reconnect_jit(seq, B, jnp.asarray(eps))      # noqa: E731

    trace = {k: [] for k in TRACE_COLUMNS + ("dE_ls", "cos", "gain")}
    qoi: dict = {}
    events: list = []

    def result(n_done, stop, wall):
        return RelaxResult(state, n_done, stop, wall, trace, qoi, events, reconnect_every, chunk, E0)

    def record(it, wall, scalars):
        row = dict(it=it, wall=wall, F=float(state.last.F_norm),
                   resid=float((state.last.F_norm / force_scale_jit(seq, state.B_n)) ** 2), **scalars)
        for k, v in row.items():
            qoi.setdefault(k, []).append(v)

    t_arm = time.perf_counter()
    t_out = 0.0     # time in samples, callbacks and reconnections; wall excludes it
    pw = jnp.zeros(seq.n(0, True), dtype=DTYPE)
    tq = time.perf_counter()
    state, pw, scalars = sample(state, pw, eager=True)   # the start of THIS run
    E0, h0 = scalars["E"], scalars["helicity"]
    record(it0, 0.0, scalars)
    if verbose:
        # The force's gradient-part remnant is the pressure solve's residual,
        # relative to |J x B| while the force is sqrt(resid) times that
        # (resid the squared normalised residual): its energy term is
        # 0.1 tol / resid of the descent (li383, float64,
        # docs/research/velocity_leray_ab_2026-09-04.md), a tenth of it at
        # resid = tol. Reported, not enforced: the tolerance and the floor
        # are the caller's choices.
        print(f"[start] it {it0}  E={E0:.8e}  |F|={float(state.last.F_norm):.4e}  "
              f"resid={qoi['resid'][-1]:.4e}  H={h0:+.6e}  J/B={scalars['JoverB']:.4f}\n"
              f"        {pressure_line(scalars)}\n"
              f"        solve tol {seq.tol:.1e}: the force's gradient-part term is a tenth of the "
              f"descent at the squared residual {seq.tol:.1e} (0.1 tol / resid)", flush=True)
    t_out += time.perf_counter() - tq

    n_done, stop = 0, "running"
    for _ in range(steps // chunk):
        state, ch = run(state, it0 + n_done)
        ch = {k: np.asarray(v) for k, v in ch.items()}
        n_done += chunk
        it = it0 + n_done
        with np.errstate(invalid="ignore"):   # a backward line-search step has no gain
            cos = ch["Fu"] / (ch["F"] * ch["v"])
            trace["cos"].extend(cos.tolist())
            trace["gain"].extend(((ch["Fu"] / ch["dt"]) ** 0.5 / ch["v"]).tolist())
            trace["dE_ls"].extend((-ch["dt"] * ch["Fu"] * (1.0 - 0.5 * ch["dt"] / ch["dt_star"])).tolist())
        for k in TRACE_COLUMNS:
            trace[k].extend(ch[k].tolist())
        resid_now = float(ch["resid"].mean())

        tq = time.perf_counter()
        wall = tq - t_arm - t_out
        state, pw, scalars = sample(state, pw)
        record(it, wall, scalars)
        if verbose:
            print(f"  it {it:>5d}  E_0-E={E0 - scalars['E']:.4e}  |F|={ch['F'][-1]:.4e}  "
                  f"resid={resid_now:.3e} (chunk mean)  H={scalars['helicity']:+.6e}  "
                  f"dH={scalars['helicity'] - h0:+.3e}  dt={ch['dt'].mean():+.3e}  "
                  f"cos min={np.nanmin(cos):+.4f}  divB={ch['div'].max():.2e}  "
                  f"[{wall:.0f}s steps +{t_out:.0f}s other]\n"
                  + (f"           newton: MINRES it mean {np.abs(ch['newton_it']).mean():.0f} max "
                     f"{np.abs(ch['newton_it']).max()}, unconverged {int((ch['newton_it'] > 0).sum())}, "
                     f"dt* mean {ch['dt_star'].mean():.3e}\n"
                     if ts.newton else "")
                  + (f"           resistive: eps {ts.resistivity:.3e} per step, CG it mean "
                     f"{np.abs(ch['res_it']).mean():.0f} max {np.abs(ch['res_it']).max()}, "
                     f"||delta||/||B|| mean {ch['res_moved'].mean():.2e}\n" if ts.resistivity else "")
                  + f"           {pressure_line(scalars)}", flush=True)
        if resid_now < floor_tol:
            stop = "floor"
        elif n_done == steps:
            stop = "steps"
        if on_chunk is not None:
            on_chunk(result(n_done, stop, wall))
        if stop != "running":
            if verbose and stop == "floor":
                print(f"  [floor] chunk mean of the force residual {resid_now:.3e} below {floor_tol:.1e} at it={it}", flush=True)
            t_out += time.perf_counter() - tq
            break
        in_window = reconnect_window is None or reconnect_window[0] <= it <= reconnect_window[1]
        if reconnect_every and n_done % reconnect_every == 0 and in_window:
            k = len(events) + 1
            eps = (reconnect_eps if reconnect_eps is not None
                   else reconnect_helicity * abs(scalars["helicity"]) / (2.0 * abs(scalars["JB"])))
            ev = dict(k=k, it=it, resid=resid_now, eps=eps,
                      helicity_target=None if reconnect_eps is not None else reconnect_helicity,
                      F_before=float(state.last.F_norm), **{f"{kk}_before": v for kk, v in scalars.items()})
            B_new, info, rel = reconnect_fn(state.B_n, eps)
            state = initial_state(B_new, ts, dt=float(state.dt), step=it)
            state, pw, scalars = sample(state, pw)
            record(it, wall, scalars)
            ev.update(solve_it=int(info), moved=float(rel), F_after=float(state.last.F_norm),
                      helicity_spent=(scalars["helicity"] - ev["helicity_before"]) / abs(ev["helicity_before"]),
                      **{f"{kk}_after": v for kk, v in scalars.items()})
            events.append(ev)
            if verbose:
                dose = "constant" if reconnect_eps is not None else f"for {reconnect_helicity:.2%} of H"
                print(f"  [reconnect {k}] at it={it}: eps={eps:.3e} {dose} "
                      f"({int(info)} it, moved {float(rel):.2e}); |F| {ev['F_before']:.3e} -> "
                      f"{ev['F_after']:.3e}, H {ev['helicity_before']:+.6e} -> {ev['helicity_after']:+.6e} "
                      f"({ev['helicity_spent']:+.2%}), J/B {ev['JoverB_before']:.3f} -> "
                      f"{ev['JoverB_after']:.3f}", flush=True)
        t_out += time.perf_counter() - tq

    res = result(n_done, stop, time.perf_counter() - t_arm - t_out)
    if verbose:
        print_summary(res, ts)
    return res


def print_summary(res: RelaxResult, ts: TimeStepper) -> None:
    """The end-of-run summary of :func:`relax`: energy removed, residual,
    the line-search identity, helicity drift, pressures, the CFL cap."""
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
    print(f"    energy increases on {int((dE > 0).sum())}/{n} steps;  ||div B|| max {max(tr['div']):.3e};  "
          f"||J||/||B|| {q['JoverB'][0]:.4e} -> {q['JoverB'][-1]:.4e}")
    h = np.array(q["helicity"])
    print(f"    helicity {h[0]:+.6e} -> {h[-1]:+.6e}  drift {h[-1] - h[0]:+.3e}"
          f"  relative {(h[-1] - h[0]) / abs(h[0]):+.3e}")
    print(f"    pressures at the start: {pressure_line({k: v[0] for k, v in q.items()})}")
    print(f"    pressures at the end:   {pressure_line({k: v[-1] for k, v in q.items()})}")
    dts, dt_star = np.array(tr["dt"]), np.array(tr["dt_star"])
    print(f"    CFL cap (C={ts.cfl}) bound on {int((dts < dt_star).sum())}/{n} steps;  "
          f"dt/dt* min {(dts / dt_star).min():.3f} mean {(dts / dt_star).mean():.3f};  "
          f"CFL number taken max {(dts * np.array(tr['cfl'])).max():.3f}")
    if ts.newton:
        nit = np.abs(np.array(tr["newton_it"]))
        print(f"    newton: MINRES iterations mean {nit.mean():.1f}  max {nit.max()}  "
              f"unconverged on {int((np.array(tr['newton_it']) > 0).sum())}/{n} steps;  "
              f"dt* mean {dt_star.mean():.3e}", flush=True)
