"""Energy-descent relaxation of a 2-form magnetic field at fixed helicity: force, time stepper, and diagnostics."""
# %%
from enum import Enum
from typing import Callable, NamedTuple, Optional

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence


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
    proxy of ``B``, the auxiliary variable that makes the midpoint scheme
    conserve helicity exactly (:class:`IntegrationScheme`). Returns ``(F,
    p, J, X, JxX)``: ``p`` the Leray multiplier, ``X`` the field the cross
    products read (``H``, or ``B`` itself) and ``JxX`` the unprojected
    force. The guesses are the previous call's ``p``, ``X``, ``JxX`` and
    ``J``, and ``F_guess`` its force: with ``JxH_guess`` it gives the
    previous gradient part ``JxX - F``, which warm-starts the lower block
    of the Leray saddle solve next to ``p_guess`` on its upper one.
    """
    J = seq.apply_weak_curl(B, dirichlet=True, guess=J_guess)
    if auxiliary_B_field:
        H_dual = seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True)
        X = seq.apply_inverse_mass_matrix(H_dual, 1, dirichlet=True, guess=H_guess)
        JxX_dual = seq.cross_product_load(J, X, 2, 1, 1, True, True, True)
    else:
        X = B
        JxX_dual = seq.cross_product_load(J, B, 2, 1, 2, True, True, True)
    JxX = seq.apply_inverse_mass_matrix(JxX_dual, 2, guess=JxH_guess)
    sigma_guess = None if F_guess is None else JxH_guess - F_guess
    F, p = seq.apply_leray_projection(JxX, k=2, p_guess=p_guess, sigma_guess=sigma_guess)
    return F, p, J, X, JxX


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
    from mrx.quadrature import evaluate_at_xq

    quad_shape = seq.quad.shape
    wJ = seq.quad.w * seq.jacobian_j

    # (a) the gauge-free comparisons: gradients in the Dirichlet 2-form
    # space, and the functions at the quadrature points with the means removed.
    gpw = seq.apply_incidence_matrix(p_w, 0, dirichlet_in=True, dirichlet_out=False)
    gpw2 = seq.apply_inverse_mass_matrix(
        seq.apply_projection_matrix(gpw, 1, 2, dirichlet_in=False, dirichlet_out=True), 2)
    gp = seq.apply_weak_grad(p, True)
    gradp_cmp = seq.l2_norm(gpw2 - gp, 2) / seq.l2_norm(gpw2, 2)
    ci0, cs0 = seq._form_comp_info(0)
    ci3, cs3 = seq._form_comp_info(3)
    pw_q = evaluate_at_xq(seq.E(0, True).T @ p_w, ci0, cs0, quad_shape, 1)[:, 0]
    p_q = evaluate_at_xq(seq.E(3, True).T @ p, ci3, cs3, quad_shape, 1)[:, 0] / seq.jacobian_j
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


def resistive_step(B: jnp.ndarray, seq: DeRhamSequence, eps):
    """One backward-Euler step of ``dB/dt = -eta curl curl B`` over ``dt``,
    ``eps = eta dt``, in defect form: ``(M_2 + eps L_2) delta = -eps L_2 B``
    and ``B + delta``. Solving for the increment keeps the step meaningful in
    float32 (the solution is ``B`` plus something small, not something that
    happens to be close to ``B``). Returns ``(B + delta, info, ||delta||_M
    / ||B||_M)`` with ``info`` the solver's signed iteration count. The
    descent itself is ideal; ``scripts/relax.py --reconnect-every`` applies
    this solve between chunks, a dose ``eps`` per reconnection."""
    rhs = -eps * seq.apply_laplacian(B, 2, dirichlet=True)
    delta, info = seq.apply_inverse_mass_plus_eps_laplace_matrix(
        rhs, 2, eps, dirichlet=True, return_info=True)
    rel = seq.l2_norm(delta, 2) / seq.l2_norm(B, 2)
    return B + delta, info.astype(jnp.int32), rel


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
    h = []
    for b in seq.basis_0.bases[0].bases:
        knots = np.asarray(b.T)
        interior = knots[b.p:-b.p] if b.type in ('clamped', 'periodic') else knots
        h.append(np.diff(interior).min())
    h = np.array(h)
    weights = 1.0 / (np.asarray(seq.jacobian_j)[:, None] * h[None, :])
    weights[:, 1] *= np.asarray(seq.quad.x[:, 0]) >= h[0]
    return jnp.asarray(weights)


# %%


class State(eqx.Module):
    """
    A class to store the state (variables and parameters) of the MRX relaxation.

    Attributes
    ----------
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
        The time step taken, ``min(dt_star, cfl / cfl_max)``.
    dt_star : float
        The uncapped step, the linesearch minimiser.
    cfl_max : float
        The largest logical CFL number of the velocity, ``max_i max_q
        |u_ref^i| / (J h_i)`` (see ``logical_cfl_weights``).
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
    F_norm : float
        The norm of the force.
    v_norm : float
        The norm of the velocity.
    lbfgs_sy : float
        The curvature <s_k, y_k>_M of the NEWEST L-BFGS pair, as it was
        actually used by the two-loop recursion.  Reported, never clamped: a
        negative value means the stored pair is not a descent pair and the
        approximate inverse Hessian it builds is indefinite.
    picard_iterations : int
        1 under ``EXPLICIT``; the predictor plus every Picard sweep (two
        k=1 mass solves and a curl each) under ``IMPLICIT_MIDPOINT``.
    picard_restarts : int
        Times the midpoint solve halved ``dt`` and restarted (0 explicit).
    picard_residual : float
        Fixed-point defect of the last midpoint sweep, ``||g(x) - x||_M``
        relative to the predictor's increment ``||dt dB(B_n)||_M`` (0
        explicit). Above ``picard_tol`` means the step went out unconverged.
    """
    B_n: jnp.ndarray
    B_nplus1: Optional[jnp.ndarray] = None
    p: Optional[jnp.ndarray] = None
    p_v: Optional[jnp.ndarray] = None
    v: Optional[jnp.ndarray] = None
    H: Optional[jnp.ndarray] = None
    JxH: Optional[jnp.ndarray] = None
    J: Optional[jnp.ndarray] = None
    E: Optional[jnp.ndarray] = None
    F_prev: Optional[jnp.ndarray] = None
    MF_prev: Optional[jnp.ndarray] = None
    s_history: Optional[jnp.ndarray] = None
    y_history: Optional[jnp.ndarray] = None
    Ms_history: Optional[jnp.ndarray] = None
    My_history: Optional[jnp.ndarray] = None
    A: Optional[jnp.ndarray] = None
    dt: float = 1e-2
    dt_star: float = 1e-2
    cfl_max: float = 0.0
    F_norm: float = 0.0
    v_norm: float = 0.0
    lbfgs_sy: float = 0.0
    picard_iterations: int = 0
    picard_restarts: int = 0
    picard_residual: float = 0.0

    def __post_init__(self):
        if self.B_nplus1 is None:
            object.__setattr__(self, "B_nplus1", self.B_n)

# %%


#: A midpoint sweep whose defect exceeds this many times the predictor's
#: increment is not contracting: halve ``dt`` and start again.
PICARD_BLOWUP = 1e3
#: Sweeps at one ``dt`` before the midpoint solve halves ``dt`` and restarts
#: from the predictor.
PICARD_MAX = 20
#: Halvings allowed per step; after the last one the step goes out
#: unconverged, ``state.picard_residual`` above the tolerance.
PICARD_RESTARTS = 4
#: The Picard tolerance in units of ``seq.tol``: the inner solves define the
#: map, so a tighter fixed point means nothing (and in float32 is unreachable).
PICARD_TOL_FACTOR = 10.0


class IntegrationScheme(Enum):
    """``EXPLICIT`` is forward Euler on the descent velocity, ``B_{n+1} =
    B_n + dt curl(u x X_n)``, with the line search choosing ``dt`` and ``X``
    the field the cross products read (``B`` itself, or the auxiliary
    1-form ``H``, see :class:`TimeStepper`). ``IMPLICIT_MIDPOINT`` keeps
    that velocity and ``dt`` and makes the induction midpoint-implicit,
    ``B_{n+1} = B_n + dt curl(u x X_mid)`` at the midpoint field ``(B_n +
    B_{n+1}) / 2``, a linear fixed point solved by Picard iteration
    (``TimeStepper._midpoint_solve``). With the auxiliary field it
    conserves the discrete helicity ``<A, B + B_harm>`` exactly, ``E`` and
    ``H`` sharing the Dirichlet 1-form space; without it what remains is
    the grid's projection error of the pairing.
    """
    EXPLICIT = 0
    IMPLICIT_MIDPOINT = 1


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
    p_v: jnp.ndarray
    H: jnp.ndarray
    JxH: jnp.ndarray
    J: jnp.ndarray
    E: jnp.ndarray
    cfl_max: jnp.ndarray
    sy: jnp.ndarray
    y_history: jnp.ndarray
    My_history: jnp.ndarray


class TimeStepper(eqx.Module):
    """One step of the energy descent, ``B_{n+1} = B_n + dt curl(u x X)``.

    Force and descent direction (L-BFGS on the velocity), velocity
    smoothing, Leray projection, the analytic line search with its CFL cap,
    and the induction, forward Euler or midpoint-implicit
    (:class:`IntegrationScheme`). The step is ideal; reconnection is a
    separate :func:`resistive_step` between chunks.

    Attributes:
        seq: The de Rham sequence.
        auxiliary_B_field: False (the default) reads the 2-form ``B`` itself
            in both cross products, ``J x B`` and ``u x B``. True routes
            them through the auxiliary Dirichlet 1-form ``H = M_1^-1 P B``,
            ``J x H`` and ``u x H``: the variable that makes the midpoint
            scheme conserve helicity exactly, at one extra k=1 mass solve
            per force evaluation and ``H_t = 0`` on the wall.
        velocity_smoothing_order: Number of smoothing solves applied to the
            descent direction, ``v = (I - scale * Laplacian)^-order F``.
            0 (the default) leaves the direction as it is.
        velocity_smoothing_scale: Length scale of the smoothing,
            the ``mu`` in ``(M_2 + mu L_2)^-1 M_2``.
        history_size: Stored secant pairs of the L-BFGS direction. 0 is
            steepest descent, ``u = F``; 1 (the default) is memoryless
            BFGS, which under the exact line search IS Polak-Ribiere CG
            (same direction; docs/research/descent_method_2026-08-26.md,
            the separate CG arm was removed 2026-08-28). Larger values were
            measured to add nothing (li383 note, section 5j).
        cfl: Cap on the step: ``dt = min(dt_star, cfl / cfl_max)`` with
            ``cfl_max`` the largest logical CFL number of the velocity. The
            linesearch minimiser cannot raise the energy, but a large step
            leaves the ideal-induction flow (frozen-in topology violated at
            O(dt^2)) and diverges when ``||dB||`` collapses. ``inf`` disables
            the cap and leaves the trajectory untouched.
        scheme: EXPLICIT (the default) or IMPLICIT_MIDPOINT.
        picard_tol: Convergence tolerance of the midpoint fixed point,
            ``||g(x) - x||_M`` relative to the predictor's increment
            ``||dt dB(B_n)||_M``: ``PICARD_TOL_FACTOR`` times ``seq.tol``,
            set by ``__post_init__``.
        cfl_weights: ``logical_cfl_weights(seq)``, built by ``__post_init__``.
    """
    seq: DeRhamSequence
    auxiliary_B_field: bool = False
    velocity_smoothing_order: int = 0
    velocity_smoothing_scale: float = 0.0
    history_size: int = 1
    cfl: float = 0.5
    scheme: IntegrationScheme = IntegrationScheme.EXPLICIT
    picard_tol: float = None
    cfl_weights: jnp.ndarray = None

    def __post_init__(self):
        if self.history_size < 0:
            raise ValueError("history_size must be non-negative (0 is steepest descent).")
        self.picard_tol = PICARD_TOL_FACTOR * self.seq.tol
        self.cfl_weights = logical_cfl_weights(self.seq)

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
        ``relaxation_step`` for how that is arranged and what happens when
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

        ``history_size = 0`` is steepest descent, ``r = F``, and so is any
        history whose entries are all zero.
        """
        m = self.history_size
        if m == 0:
            return F, jnp.zeros((), F.dtype)

        # <s_i, y_i>_M for every stored pair.  An EMPTY slot -- s_i identically
        # zero, as the history is before it fills -- has sy_i = 0 exactly and
        # contributes nothing to either loop; rho_i = 0 states that.  (The
        # old ``1 / (sy + 1e-30)`` gave the same result only because
        # 1e30 * 0 happens to be 0.)  A NEGATIVE sy_i is a non-descent pair:
        # it is SKIPPED (rho_i = 0, so it contributes nothing and the
        # direction falls back towards F) and reported through ``sy`` below.
        # This is the curvature guard of BFGS and, at m = 1, exactly the
        # Polak-Ribiere+ restart ``beta = max(beta, 0)`` the CG arm had.
        sy_all = jnp.einsum('in,in->i', s, My)
        usable = sy_all > 0
        rho = jnp.where(usable, 1.0 / jnp.where(usable, sy_all, 1.0), 0.0)

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

    def smooth_velocity(self, u: jnp.ndarray) -> jnp.ndarray:
        """Apply ``(M_2 + scale L_2)^-1 M_2`` to ``u`` ``velocity_smoothing_order`` times."""
        for _ in range(self.velocity_smoothing_order):
            rhs = self.seq.apply_mass_matrix(u, 2, True)
            u = self.seq.apply_inverse_mass_plus_eps_laplace_matrix(
                rhs, 2, self.velocity_smoothing_scale, dirichlet=True, guess=u)
        return u

    def _induction_field(self, u_jk: jnp.ndarray, X: jnp.ndarray, E_guess: jnp.ndarray) -> jnp.ndarray:
        """``E = M_1^-1 load(u x X)`` with ``u`` at the quadrature points and
        ``X`` the auxiliary 1-form ``H`` or the 2-form ``B`` itself."""
        seq = self.seq
        k = 1 if self.auxiliary_B_field else 2
        X_jk = seq.evaluate_at_quadrature(X, k, True)
        E_dual = seq.cross_product_load_values(u_jk, X_jk, 1, 2, k, True)
        return seq.apply_inverse_mass_matrix(E_dual, 1, guess=E_guess)

    def _ideal_increment(self, B: jnp.ndarray, state: State,
                         p_guess: jnp.ndarray, p_v_guess: jnp.ndarray,
                         H_guess: jnp.ndarray, JxH_guess: jnp.ndarray,
                         J_guess: jnp.ndarray, E_guess: jnp.ndarray) -> Increment:
        """The ideal increment ``dB = curl(u x X)`` evaluated at the field ``B``.

        Force, descent direction (the L-BFGS secant ``y = F_prev - F`` is
        pushed against ``state``'s history here, see part 1 below), velocity
        smoothing, Leray projection, the cross product and the topological
        curl. The explicit step evaluates it once at ``B_n``; the midpoint
        step's Picard sweeps re-evaluate only the induction at the midpoint
        field (``_midpoint_solve``). The six guesses, and ``state.F_prev``
        for the gradient part of the force, warm-start the Krylov solves;
        they come from ``state`` (the previous step). Without the auxiliary
        field ``H_guess`` passes through untouched.
        """
        seq = self.seq
        F, p, J, X, JxX = compute_force(
            B, seq, self.auxiliary_B_field,
            p_guess=p_guess, H_guess=H_guess, JxH_guess=JxH_guess,
            J_guess=J_guess, F_guess=state.F_prev)
        # M F ONCE.  It serves ||F||_M and the L-BFGS secant
        # M y = M F_prev - M F; the increment applies M_2 three times in total
        # (M F, M u, M dB) whichever method is running -- L-BFGS used to
        # apply it 4m + 6 times.
        MF = seq.apply_mass_matrix(F, 2)

        # The secant history exists only for history_size > 0 (a static
        # branch: steepest descent carries (0, n) arrays and never touches
        # them).
        y_hist, My_hist = state.y_history, state.My_history
        if self.history_size > 0:
            # --- history bookkeeping, part 1: push y BEFORE the direction ---
            # y_{k-1} = grad_M E_k - grad_M E_{k-1} = F_prev - F is a
            # difference over the step that ALREADY happened, so it pairs with
            # s_{k-1}, which the end of the previous step put in s_history[0].
            # Pushing it HERE, rather than next to the brand-new s_k at the end
            # of this step, is what keeps (s_i, y_i) aligned.  Pushing it at
            # the end instead leaves y lagging its paired s by exactly one step.
            # Under the midpoint rule every sweep re-pushes it against the
            # SAME state history, so the pair is (s_{k-1}, F_prev - F(B_mid))
            # of the converged sweep and never an inner iterate's.
            y_hist = jnp.roll(y_hist, 1, axis=0).at[0].set(state.F_prev - F)
            My_hist = jnp.roll(My_hist, 1, axis=0).at[0].set(state.MF_prev - MF)
        u, sy = self._lbfgs_direction(F, state.s_history, y_hist, state.Ms_history, My_hist)
        u = self.smooth_velocity(u)
        u, p_v = seq.apply_leray_projection(u, k=2, p_guess=p_v_guess)
        # M u once: the linesearch numerator, ||u||_M and the stored M s.
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
        return Increment(dB, u, Mu, F, MF, p, p_v, H, JxX, J, E, cfl_max, sy, y_hist, My_hist)

    def _step_size(self, inc: Increment) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(dt, dt_star)``: the line-search step at ``inc`` and its CFL cap.

        ``dt_star = <F, u>_M / ||dB||_M^2`` minimises the quadratic energy
        along the increment exactly (``dE = -dt <F, u>_M + dt^2 ||dB||^2 / 2``).
        The cap: ``cfl = inf`` gives ``min(dt_star, inf) = dt_star`` exactly.
        """
        dt_star = inc.F @ inc.Mu / self.seq.l2_norm_sq(inc.dB, 2)
        return jnp.minimum(dt_star, self.cfl / inc.cfl_max), dt_star

    def _midpoint_solve(self, state: State):
        """Midpoint-implicit induction with the explicit descent velocity.

        The step is ``B_{n+1} = B_n + dt curl(u x X_mid)`` with ``u`` the
        descent velocity of the explicit predictor at ``B_n`` (direction,
        smoothing, Leray projection, line-search ``dt``, CFL cap: all of
        ``_ideal_increment``) and ``X_mid`` the MIDPOINT field ``(B_n +
        B_{n+1}) / 2`` itself or, with ``auxiliary_B_field``, its 1-form
        proxy ``H_mid = M_1^-1 P (B_n + B_{n+1}) / 2``: the
        auxiliary-variable scheme, whose exact helicity conservation is
        derived below. Without the proxy the pairing ``E^T P B_mid`` is the
        grid's projection error and nothing else: the time-integration
        error is gone, so that arm isolates the grid error of the helicity
        from the time error.

        WHY IT CONSERVES HELICITY.  The pairing of the 2-form ``B`` with a
        discrete 1-form ``E`` goes through the proxy ``H = M_1^-1 P B``
        (``P`` the 1-form/2-form pairing matrix)::

            E^T P B = E^T M_1 H = H^T load(u x H) = int H_h . (u_h x H_h) = 0

        at every quadrature node, for ANY ``u``.  With ``B = D_1 A + B_harm``
        (``A`` the Dirichlet potential of ``compute_helicity``) and the exact
        discrete Stokes identity ``<A, D_1 E> = <D_1 A, E>`` (``A``
        tangentially zero)::

            d/dt <A, B + B_harm> = 2 <A, D_1 E> + 2 <E, B_harm> = 2 <B, E> = 0,

        ``B_harm`` being constant (``dB/dt = D_1 E`` lies in ``range D_1``)
        and the gauge freedom in ``dA/dt`` pairing a Dirichlet gradient with
        ``D_2 B_harm = 0``.  So the semi-discrete flow conserves the discrete
        helicity exactly; as ``A`` and ``B_harm`` are linear in ``B`` the
        helicity is a quadratic form ``Q(B)``, and evaluating ``E`` at the
        midpoint field keeps it exactly, ``Q(B_{n+1}) - Q(B_n) = 2 dt <B_mid,
        E> = 0``, whatever ``u`` is.  The explicit scheme's helicity drift is
        entirely the time-integration error of evaluating ``H`` at ``B_n``.

        THE ONE CONDITION.  ``E^T P B = E^T M_1 H`` needs ``E`` and ``H`` in
        the SAME space, which is why the auxiliary ``H`` is a Dirichlet
        1-form like ``E``: with a natural ``H`` the load ``load(u x H)``
        loses its tangential wall DoFs on the way to ``E`` and both schemes
        leak helicity through that wall layer alike (li383 (8,16,16) p=2,
        float64, 1000 steps: -5.5e-7 explicit, -6.6e-7 midpoint); with the
        Dirichlet ``H`` the midpoint scheme is exact to the solves (+5e-12
        against +2.2e-7 explicit). docs/research/implicit_midpoint_2026-09-04.md.
        Energy: ``E_{n+1} - E_n = dt B_mid^T M_2 D_1 E = -dt <u, F_mid>_M``
        with ``F_mid`` the force at the midpoint field, so the step descends
        as long as the predictor's velocity still correlates with the
        midpoint force -- second order in ``dt``, not a guarantee; the line
        search of the predictor sets ``dt``.

        WHY THE VELOCITY STAYS EXPLICIT.  Taking ``u`` at the midpoint too
        makes the step a nonlinear fixed point in ``B`` through the force,
        whose linearisation is the descent operator ``|H|^2 curl curl`` on
        the field: its largest eigenvalue is ``|H|^2 / h^2`` and the
        line-search ``dt`` sits 35x above the Picard contraction limit on
        li383 (8,16,16) p=2 (iterates blow up in six sweeps; a Laplacian
        preconditioner only flips the spectrum, because the operator is
        soft on the force-free perturbations the Laplacian is stiff on;
        Newton is a Krylov solve inside a Krylov solve).  With ``u`` frozen
        the map ``x -> dt curl(u x H(B_n + x / 2))`` is LINEAR in the
        increment ``x`` with contraction constant ``dt |u| / (2 h)``, which
        is small because ``u`` is the force (``dt* |u| ~ h^2 |F| / |H|^2``
        under the line search), so plain Picard converges in a few sweeps,
        each one k=1 mass solve for ``H_mid``, one for ``E`` and the
        topological curl, warm-started from the previous sweep.  Should the
        defect ever blow up (``PICARD_BLOWUP`` times the predictor's
        increment, NaN included) or ``PICARD_MAX`` sweeps not converge,
        ``dt`` is halved and the solve restarts from the predictor, at most
        ``PICARD_RESTARTS`` times, after which the step goes out
        unconverged with ``state.picard_residual`` above ``picard_tol``.
        Convergence is judged on the defect ``||g(x) - x||_M / ||dt
        dB(B_n)||_M``, relative to the predictor's INCREMENT and never to
        ``B`` (the defect form of ``resistive_step``); ``picard_tol`` cannot
        be tighter than the inner solves that define ``g``, hence its
        value of ``PICARD_TOL_FACTOR`` times ``seq.tol``.

        Returns ``(inc, dt, dt_star, B_ideal, evaluations, restarts,
        residual)``: ``inc`` is the predictor's increment with ``H`` and
        ``E`` replaced by the midpoint's, the warm starts of the next step.
        """
        B_n = state.B_n
        seq = self.seq
        inc0 = self._ideal_increment(B_n, state, state.p, state.p_v, state.H, state.JxH,
                                     state.J, state.E)
        dt0, dt_star = self._step_size(inc0)
        dB0 = inc0.dB
        dB0_norm = seq.l2_norm(dB0, 2)
        u_jk = seq.evaluate_at_quadrature(inc0.u, 2, True)
        one = jnp.ones((), B_n.dtype)

        def sweep(carry):
            k, n_eval, restarts, dt, x, H, E, resid = carry
            B_mid = B_n + 0.5 * x
            if self.auxiliary_B_field:
                H_dual = seq.apply_projection_matrix(B_mid, 2, 1, True, dirichlet_out=True)
                H = seq.apply_inverse_mass_matrix(H_dual, 1, dirichlet=True, guess=H)
            E = self._induction_field(u_jk, H if self.auxiliary_B_field else B_mid, E)
            g = dt * seq.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
            resid = seq.l2_norm(g - x, 2) / (dt * dB0_norm)
            k, n_eval = k + 1, n_eval + 1
            converged = resid <= self.picard_tol
            restart = (~converged & (~(resid < PICARD_BLOWUP) | (k >= PICARD_MAX))
                       & (restarts < PICARD_RESTARTS))
            dt = jnp.where(restart, 0.5 * dt, dt)
            x = jnp.where(restart, dt * dB0, g)
            k = jnp.where(restart, 0, k)
            resid = jnp.where(restart, one, resid)
            restarts = restarts + restart.astype(jnp.int32)
            return k, n_eval, restarts, dt, x, H, E, resid

        def unconverged(carry):
            k, _, _, _, _, _, _, resid = carry
            return ~(resid <= self.picard_tol) & (resid < PICARD_BLOWUP) & (k < PICARD_MAX)

        carry = (jnp.int32(0), jnp.int32(1), jnp.int32(0), dt0, dt0 * dB0, inc0.H, inc0.E, one)
        _, n_eval, restarts, dt, x, H, E, resid = jax.lax.while_loop(unconverged, sweep, carry)
        return inc0._replace(H=H, E=E), dt, dt_star, B_n + x, n_eval, restarts, resid

    def relaxation_step(self, state: State) -> State:
        """Advance ``state.B_n`` by one ideal step into ``state.B_nplus1``:
        forward Euler on the descent velocity, ``B_n + dt curl(u x X)``, or
        the implicit midpoint rule (``scheme``)."""
        B_n = state.B_n
        if self.scheme == IntegrationScheme.EXPLICIT:
            inc = self._ideal_increment(B_n, state, state.p, state.p_v, state.H, state.JxH,
                                        state.J, state.E)
            dt, dt_star = self._step_size(inc)
            B_nplus1 = B_n + dt * inc.dB
            n_eval, restarts, resid = jnp.int32(1), jnp.int32(0), jnp.zeros((), B_n.dtype)
        elif self.scheme == IntegrationScheme.IMPLICIT_MIDPOINT:
            # inc is the predictor's (u, F, dt* are the explicit step's);
            # only the induction is implicit.
            inc, dt, dt_star, B_nplus1, n_eval, restarts, resid = self._midpoint_solve(state)
        else:
            raise ValueError(
                f"Unknown scheme: {self.scheme}. Supported schemes are given by the IntegrationScheme enum.")

        # --- history bookkeeping, part 2: push the step just taken ----------
        # The descent variable is the VELOCITY u, not B: grad_M E = -F is the
        # derivative of E with respect to u (dE = -(F, u)_M), and the direction
        # the recursion returns is consumed as a velocity.  So the step in the
        # descent variable is dt*u.  B moves by dt*curl(u x H) instead, which
        # is a different vector in a different space; storing THAT as s pairs
        # it with a y that is a secant of a different map entirely, and the
        # curvature <s, y>_M goes negative on a third to a half of all steps.
        # See docs/research/handoff_2026-08-25_relaxation_prelim.md.
        s_hist, Ms_hist = state.s_history, state.Ms_history
        if self.history_size > 0:
            s_hist = jnp.roll(s_hist, 1, axis=0).at[0].set(dt * inc.u)
            Ms_hist = jnp.roll(Ms_hist, 1, axis=0).at[0].set(dt * inc.Mu)

        return eqx.tree_at(
            lambda s: (s.B_nplus1, s.v, s.p, s.p_v, s.H, s.JxH, s.J, s.E,
                       s.F_prev, s.MF_prev, s.F_norm, s.v_norm, s.lbfgs_sy,
                       s.dt, s.dt_star, s.cfl_max,
                       s.s_history, s.y_history, s.Ms_history, s.My_history,
                       s.picard_iterations, s.picard_restarts, s.picard_residual),
            state,
            (B_nplus1, inc.u, inc.p, inc.p_v, inc.H, inc.JxH, inc.J, inc.E,
             inc.F, inc.MF, jnp.sqrt(inc.F @ inc.MF), jnp.sqrt(inc.u @ inc.Mu), inc.sy,
             dt, dt_star, inc.cfl_max,
             s_hist, inc.y_history, Ms_hist, inc.My_history,
             n_eval, restarts, resid))


def initial_state(B_dof: jnp.ndarray, ts: TimeStepper, dt: float = 1.0) -> State:
    """Build the state at ``B_dof`` with its force already evaluated.

    ``F_prev``, ``MF_prev``, ``F_norm`` and the warm-start guesses ``p``,
    ``H``, ``JxH``, ``J`` are seeded from one ``compute_force`` here, so the
    first step's secant ``y = F_prev - F`` sees the true previous gradient.
    """
    seq = ts.seq
    n = seq.n(2, True)
    m = ts.history_size
    F0, p0, J0, X0, JxX0 = compute_force(B_dof, seq, ts.auxiliary_B_field)
    MF0 = seq.apply_mass_matrix(F0, 2)
    return State(
        B_n=B_dof,
        dt=dt,
        dt_star=dt,
        v=jnp.zeros(n),
        p=p0,
        p_v=jnp.zeros(seq.n(3, True)),
        H=X0 if ts.auxiliary_B_field else jnp.zeros(seq.n(1, True)),
        JxH=JxX0,
        J=J0,
        E=jnp.zeros(seq.n(1, True)),
        A=jnp.zeros(seq.n(1, True)),
        F_prev=F0,
        MF_prev=MF0,
        F_norm=jnp.sqrt(F0 @ MF0),
        s_history=jnp.zeros((m, n)),
        y_history=jnp.zeros((m, n)),
        Ms_history=jnp.zeros((m, n)),
        My_history=jnp.zeros((m, n)),
        picard_iterations=jnp.int32(0),
        picard_restarts=jnp.int32(0),
        picard_residual=jnp.zeros((), B_dof.dtype),
    )


def chunk_runner(ts: TimeStepper, n_chunk: int,
                 extra: Optional[dict[str, Callable[[State], jnp.ndarray]]] = None,
                 ) -> Callable[[State, int], tuple[State, dict]]:
    """``run(state, it0) -> (state, trace)``, jit-compiled: ``n_chunk``
    relaxation steps as one ``lax.scan``.

    The state (B, the L-BFGS pair, the warm-start guesses) is the carry and
    comes out once; the per-step scalars are the scan's stacked output,
    ``trace[name]`` an array of length ``n_chunk`` over the steps
    ``it0 + 1 .. it0 + n_chunk``: ``E`` (``||B||_M^2 / 2``), ``F``
    (``||F||_M``), ``v`` (``||u||_M``), ``dt``, ``dt_star``, ``cfl`` (the
    velocity's largest logical CFL number), ``div`` (``||div B||``), ``Fu``
    (``<F_prev, u>_M``: the line search predicts ``dE = -dt Fu / 2``),
    ``picard_it`` and ``picard_resid`` (the midpoint solve's increment
    evaluations and final defect; 1 and 0 for the explicit step), plus
    ``extra[name](state)`` for every extra probe.

    Compile time is the body's whatever ``n_chunk`` (a ``While`` trip
    count); the chunk is the cadence at which the host sees the trace and
    may act on the state.
    """
    seq = ts.seq
    extra = extra or {}

    def body(state, it):
        state = ts.relaxation_step(state)
        state = eqx.tree_at(lambda s: s.B_n, state, state.B_nplus1)
        trace = dict(
            E=0.5 * seq.l2_norm_sq(state.B_n, 2), F=state.F_norm, v=state.v_norm,
            dt=state.dt, dt_star=state.dt_star, cfl=state.cfl_max,
            div=compute_divergence_norm(state.B_n, seq),
            Fu=state.F_prev @ seq.apply_mass_matrix(state.v, 2),
            picard_it=state.picard_iterations, picard_resid=state.picard_residual,
            **{k: f(state) for k, f in extra.items()})
        return state, trace

    @jax.jit
    def run(state, it0):
        return jax.lax.scan(body, state, it0 + jnp.arange(1, n_chunk + 1))

    return run


# ---------------------------------------------------------------------------
# The run: the residual scale, the diagnostics sampler, checkpoints, the loop
# ---------------------------------------------------------------------------

def force_scale(seq: DeRhamSequence) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """``||grad(B^2/2)||_L2`` as a jitted function of the 2-form DoFs: the
    scale the force residual is measured against.

    ``grad p`` is a real scale too (the scheme converges to ``J x B = grad
    p``) but vanishes in the low-beta limit; ``grad(B^2/2)`` has the same
    units and stays O(1). Through the sequence: the 0-form load of
    ``B^2/2`` (:meth:`dot_product_load`), one natural ``M_0`` solve, the
    strong gradient, its norm.
    """
    @jax.jit
    def scale(B):
        q = 0.5 * seq.dot_product_load(B, B, 0, 2, 2, dirichlet_n=False)
        w0 = seq.apply_inverse_mass_matrix(q, 0, dirichlet=False)
        g1 = seq.apply_strong_grad(w0, dirichlet_in=False, dirichlet_out=False)
        return seq.l2_norm(g1, 1, dirichlet=False)
    return scale


def make_sampler(seq: DeRhamSequence, ts: TimeStepper):
    """``sample(state, pw_guess, eager=False) -> (state, p_w, scalars)``: the
    diagnostics of a state's field.

    The force at the CURRENT field (``state.p``, ``H``, ``JxH``, ``J``,
    ``F_prev`` are the step's values at the previous one; they warm-start it
    and are refreshed from it), the weak pressure and its diagnostics
    (:func:`pressure_diagnostics`), the helicity (``state.A`` refreshed),
    ``||J|| / ||B||`` and the pairing ``int J . B`` that sets a
    reconnection dose. ``scalars`` are Python floats. The first call of a
    run goes ``eager`` (the 1->2 projection builds a host-side core on
    first use); the loop uses the compiled one.
    """
    aux = ts.auxiliary_B_field

    def probe(B, p, H, JxH, J, F_prev, pw_guess, A):
        F, p, J, X, JxX = compute_force(B, seq, aux, p_guess=p, H_guess=H, JxH_guess=JxH,
                                        J_guess=J, F_guess=F_prev)
        p_w, F_w, v = weak_pressure(J, X, seq, aux, p_guess=pw_guess)
        diag = pressure_diagnostics(B, p, p_w, F_w, v, seq)
        h, A_new = compute_helicity(B, seq, A)
        JoverB = seq.l2_norm(J, 1) / seq.l2_norm(B, 2)
        JB = J @ seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=True)
        return p, (X if aux else H), JxX, J, A_new, p_w, h, JoverB, JB, diag

    probe_jit = jax.jit(probe)

    def sample(state: State, pw_guess: jnp.ndarray, eager: bool = False):
        f = probe if eager else probe_jit
        p, H, JxH, J, A, p_w, h, JoverB, JB, diag = f(
            state.B_n, state.p, state.H, state.JxH, state.J, state.F_prev, pw_guess, state.A)
        state = eqx.tree_at(lambda s: (s.p, s.H, s.JxH, s.J, s.A), state, (p, H, JxH, J, A))
        scalars = dict(helicity=float(h), JoverB=float(JoverB), JB=float(JB),
                       **{k: float(v) for k, v in diag.items()})
        return state, p_w, scalars

    return sample


def write_checkpoint(path: str, state: State, step: int) -> None:
    """The state at a step as one HDF5 file: every leaf of the pytree as a
    dataset named by its field (``B_n``, ``p``, ``s_history``, ...), the step
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
    the stored one."""
    import h5py  # noqa: PLC0415
    from mrx.precision import DTYPE  # noqa: PLC0415
    with h5py.File(path, "r") as fh:
        step = int(fh.attrs["step"])
        data = {k: np.asarray(v) for k, v in fh.items()}
    skeleton = initial_state(jnp.asarray(data["B_n"]), ts)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(skeleton)
    new = []
    for keypath, _ in leaves:
        v = data[jax.tree_util.keystr(keypath).lstrip(".")]
        new.append(jnp.asarray(v, dtype=DTYPE if np.issubdtype(v.dtype, np.floating) else v.dtype))
    return jax.tree_util.tree_unflatten(treedef, new), step


class RelaxResult(NamedTuple):
    """What :func:`relax` returns and hands to ``on_chunk`` at every chunk.

    ``state`` the descent state, ``steps`` the steps of this run so far
    (``it0 + steps`` is the absolute step), ``stop`` why it ended (``steps``,
    ``floor``, ``seconds``, or ``running``), ``wall`` the seconds in the
    compiled steps (sampling and callbacks excluded), ``trace`` the per-step
    scalars (``E``, ``F``, ``resid``, ``dt``, ``dt_star``, ``cfl``, ``div``,
    ``cos``, ``gain``, ``picard_it``, ``picard_resid``, ``dE_meas``,
    ``dE_pred``), ``qoi`` the per-chunk samples (``it``, ``wall``,
    ``F``, ``resid``, ``helicity``, ``JoverB``, ``JB`` and the pressure
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


def pressure_line(d: dict) -> str:
    """One line of the pressure diagnostics of a sample."""
    return (f"beta_vol={d['beta_vol']:.3e}  beta_axis={d['beta_axis']:.3e}  "
            f"|grad pw - grad p|/|grad pw|={d['gradp_cmp']:.3e}  |pw - p|/|pw|={d['p_cmp']:.3e}  "
            f"weak_resid={d['weak_resid']:.3e}  "
            f"wall dpw/dn={d['dpdn_wall']:.3e}  (JxB).n={d['JxBn_wall']:.3e}")


def relax(state: State, ts: TimeStepper, steps: int, chunk: int = 500, it0: int = 0,
          floor_tol: float = 0.0, seconds: Optional[float] = None,
          reconnect_every: int = 0, reconnect_helicity: float = 0.01,
          on_chunk: Optional[Callable[[RelaxResult], None]] = None,
          verbose: bool = True) -> RelaxResult:
    """The relaxation run: ``steps`` steps in compiled chunks of ``chunk``
    (:func:`chunk_runner`), the diagnostics sampled once per chunk
    (:func:`make_sampler`), the stop tests and the reconnection series.

    Stops on the step count, on ``floor_tol`` (the last chunk's mean of the
    relative force residual ``||F||_M / ||grad(B^2/2)||`` below it; the
    residual is not monotone, the window mean is the quantity) or on
    ``seconds`` of wall time in the steps. ``reconnect_every`` (rounded to
    whole chunks, never on the last one) applies one :func:`resistive_step`
    to the field whose dose spends the fraction ``reconnect_helicity`` of
    its helicity, ``eps = X |H| / (2 |int J . B|)`` from ``dH = -2 eps int J
    . B``, then restarts the optimiser on the diffused field
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
    scale = force_scale(seq)
    run = chunk_runner(ts, chunk, extra=dict(resid=lambda st: st.F_norm / scale(st.B_n)))
    sample = make_sampler(seq, ts)
    reconnect_fn = jax.jit(lambda B, eps: resistive_step(B, seq, eps))

    trace = {k: [] for k in ("E", "F", "resid", "dt", "dt_star", "cfl", "div", "cos",
                             "gain", "picard_it", "picard_resid", "dE_meas", "dE_pred")}
    qoi: dict = {}
    events: list = []

    def result(n_done, stop, wall):
        return RelaxResult(state, n_done, stop, wall, trace, qoi, events, reconnect_every, chunk)

    def record(it, wall, scalars):
        row = dict(it=it, wall=wall, F=float(state.F_norm),
                   resid=float(state.F_norm / scale(state.B_n)), **scalars)
        for k, v in row.items():
            qoi.setdefault(k, []).append(v)

    t_arm = time.perf_counter()
    t_out = 0.0     # time in samples, callbacks and reconnections; wall excludes it
    E_prev = 0.5 * float(seq.l2_norm_sq(state.B_n, 2))
    pw = jnp.zeros(seq.n(0, True))
    tq = time.perf_counter()
    state, pw, scalars = sample(state, pw, eager=True)   # the start of THIS run
    h0 = scalars["helicity"]
    record(it0, 0.0, scalars)
    if verbose:
        print(f"[start] it {it0}  E={E_prev:.8e}  |F|={float(state.F_norm):.4e}  "
              f"resid={float(state.F_norm / scale(state.B_n)):.4e}  H={h0:+.6e}  J/B={scalars['JoverB']:.4f}\n"
              f"        {pressure_line(scalars)}", flush=True)
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
        trace["dE_meas"].extend(np.diff(ch["E"], prepend=E_prev).tolist())
        trace["dE_pred"].extend((-0.5 * ch["dt"] * ch["Fu"]).tolist())
        for k in ("E", "F", "resid", "dt", "dt_star", "cfl", "div", "picard_it", "picard_resid"):
            trace[k].extend(ch[k].tolist())
        E_prev = float(ch["E"][-1])
        resid_now = float(ch["resid"].mean())

        tq = time.perf_counter()
        wall = tq - t_arm - t_out
        state, pw, scalars = sample(state, pw)
        record(it, wall, scalars)
        if verbose:
            print(f"  it {it:>5d}  E={E_prev:.8e}  |F|={ch['F'][-1]:.4e}  "
                  f"resid={resid_now:.3e} (chunk mean)  H={scalars['helicity']:+.6e}  "
                  f"dH={scalars['helicity'] - h0:+.3e}  dt={ch['dt'].mean():+.3e}  "
                  f"cos min={np.nanmin(cos):+.4f}  divB={ch['div'].max():.2e}  "
                  f"picard max={int(ch['picard_it'].max())}  [{wall:.0f}s steps +{t_out:.0f}s other]\n"
                  f"           {pressure_line(scalars)}", flush=True)
        if resid_now < floor_tol:
            stop = "floor"
        elif seconds is not None and wall > seconds:
            stop = "seconds"
        elif n_done == steps:
            stop = "steps"
        if on_chunk is not None:
            on_chunk(result(n_done, stop, wall))
        if stop != "running":
            if verbose and stop != "steps":
                print(f"  [{stop}] {'chunk mean of the force residual %.3e below %.1e' % (resid_now, floor_tol) if stop == 'floor' else '%.0f s spent' % seconds} at it={it}", flush=True)
            t_out += time.perf_counter() - tq
            break
        if reconnect_every and n_done % reconnect_every == 0:
            k = len(events) + 1
            eps = reconnect_helicity * abs(scalars["helicity"]) / (2.0 * abs(scalars["JB"]))
            ev = dict(k=k, it=it, resid=resid_now, eps=eps, helicity_target=reconnect_helicity,
                      F_before=float(state.F_norm), **{f"{kk}_before": v for kk, v in scalars.items()})
            B_new, info, rel = reconnect_fn(state.B_n, eps)
            state = initial_state(B_new, ts, dt=float(state.dt))
            state, pw, scalars = sample(state, pw)
            record(it, wall, scalars)
            ev.update(solve_it=int(info), moved=float(rel), F_after=float(state.F_norm),
                      helicity_spent=(scalars["helicity"] - ev["helicity_before"]) / abs(ev["helicity_before"]),
                      **{f"{kk}_after": v for kk, v in scalars.items()})
            events.append(ev)
            if verbose:
                print(f"  [reconnect {k}] at it={it}: eps={eps:.3e} for {reconnect_helicity:.2%} of H "
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
    the line-search identity, helicity drift, pressures, the CFL cap, the
    midpoint solve."""
    tr, q = res.trace, res.qoi
    n = res.steps
    E0, E1 = tr["E"][0] - tr["dE_meas"][0], tr["E"][-1]
    dEm, dEp = np.array(tr["dE_meas"]), np.array(tr["dE_pred"])
    ident = np.abs(dEm - dEp) / E0
    resid = np.array(tr["resid"])
    print(f"\n--- {n} steps in {res.wall:.1f}s ({res.wall / max(n, 1):.2f} s/step), stopped on: {res.stop}")
    print(f"    E {E0:.8e} -> {E1:.8e}  ({(E0 - E1) / E0:.4%} of the initial energy removed)")
    print(f"    residual {resid[0]:.4e} -> {resid[-1]:.4e}  (mean over the last chunk of "
          f"{res.chunk} steps {resid[-res.chunk:].mean():.4e}, min {resid.min():.4e})")
    print(f"    linesearch identity |dE_meas - dE_pred| / E0: median {np.median(ident):.3e}  max {ident.max():.3e}"
          + ("  (not an identity under the midpoint scheme)"
             if ts.scheme == IntegrationScheme.IMPLICIT_MIDPOINT else ""))
    print(f"    energy increases on {int((dEm > 0).sum())}/{n} steps;  ||div B|| max {max(tr['div']):.3e};  "
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
    if ts.scheme == IntegrationScheme.IMPLICIT_MIDPOINT:
        pit, pres = np.array(tr["picard_it"]), np.array(tr["picard_resid"])
        print(f"    midpoint solve: increment evaluations mean {pit.mean():.2f}  max {pit.max()};  "
              f"defect max {pres.max():.2e};  unconverged on {int((pres > ts.picard_tol).sum())}/{n} "
              f"steps (tolerance {ts.picard_tol:.1e})", flush=True)
