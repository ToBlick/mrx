"""Energy-descent relaxation of a 2-form magnetic field at fixed helicity: force, time stepper, and diagnostics."""
# %%
from enum import Enum
from typing import Callable, Literal, Optional

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
    J = seq.apply_weak_curl(B, dirichlet=True)
    JxH_dual = seq.cross_product_load(
        J, H, 2, 1, 1, True, True, dirichlet_H)
    JxH = seq.apply_inverse_mass_matrix(JxH_dual, 2, guess=JxH_guess)
    F, p = seq.apply_leray_projection(JxH, k=2, p_guess=p_guess)
    return F, p, J, H, JxH


def weak_pressure(
    J: jnp.ndarray,
    H: jnp.ndarray,
    seq: DeRhamSequence,
    dirichlet_H: bool = False,
    p_guess: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The weak pressure ``p_w`` of ``J x H`` and the weak force residual.

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

    ``J`` and ``H`` are ``compute_force``'s (``J`` a Dirichlet 1-form, ``H``
    a 1-form with ``dirichlet_H``), so the current is not recomputed.
    Costs two natural k=1 mass solves and the k=0 solve.

    Returns:
        ``(p_w, F_w, v)``: the Dirichlet 0-form DoFs of the weak pressure,
        the weak force residual and the natural 1-form projection of
        ``J x H``, both in the natural 1-form space.
    """
    v_dual = seq.cross_product_load(J, H, 1, 1, 1, False, True, dirichlet_H)
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
    happens to be close to ``B``). Returns ``(B + delta, minres_info,
    ||delta||_M / ||B||_M)``. The relaxation's resistive half-step and
    ``scripts/relax.py --presmooth`` both use it."""
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
        The uncapped step: the linesearch minimiser, or ``dt`` in FIXED mode.
    cfl_max : float
        The largest logical CFL number of the velocity, ``max_i max_q
        |u_ref^i| / (J h_i)`` (see ``logical_cfl_weights``).
    eta : float
        The resistivity. The resistive part of the step is backward Euler
        (see ``TimeStepper.relaxation_step``), so it does not restrict
        ``dt``.
    resistive_info : int
        The signed MINRES iteration count of the resistive solve on the last
        step: ``-k`` converged after ``k`` iterations, ``+k`` not; ``0`` when
        the solve was skipped (``eta = 0``, or not due under ``eta_every``).
    resistive_delta : float
        ``||delta||_M / ||B||_M`` of the last resistive solve, 0 when skipped.
    resistive_count : int
        Steps since the last resistive solve.
    resistive_time : float
        Time (sum of ``dt``) since the last resistive solve; the next solve
        diffuses over it.
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
    dt_star: float = 1e-2
    cfl_max: float = 0.0
    eta: float = 0.0
    resistive_info: int = 0
    resistive_delta: float = 0.0
    resistive_count: int = 0
    resistive_time: float = 0.0
    F_norm: float = 0.0
    v_norm: float = 0.0
    lbfgs_sy: float = 0.0

    def __post_init__(self):
        if self.B_nplus1 is None:
            object.__setattr__(self, "B_nplus1", self.B_n)

# %%


class TimeStepChoice(Enum):
    FIXED = 0
    ANALYTIC_LINESEARCH = 1


class DescentMethod(Enum):
    """``GRADIENT`` is steepest descent, ``u = F``. ``LBFGS`` with
    ``history_size = 1`` (the default) is memoryless BFGS, which under the
    exact line search IS Polak-Ribiere CG (same direction; the classical
    identity, and one trajectory to 2e-8 in energy on W7-X, see
    docs/research/descent_method_2026-08-26.md) -- the separate CG arm was
    removed 2026-08-28. Larger ``history_size`` was measured to add nothing.
    """
    GRADIENT = 0
    LBFGS = 1


class TimeStepper(eqx.Module):
    """One explicit step of the energy descent.

    The step is an operator splitting (Lie): ideal transport,
    ``B_ideal = B_n + dt curl E_ideal`` with ``E_ideal = u x H`` and the
    descent velocity ``u``, then implicit (backward-Euler) resistive
    diffusion of ``B_ideal``. The splitting is first order in ``dt``.

    Attributes:
        seq: The de Rham sequence.
        velocity_smoothing_order: Number of smoothing solves applied to the
            descent direction, ``v = (I - scale * Laplacian)^-order F``.
            0 (the default) leaves the direction as it is.
        velocity_smoothing_scale: Length scale of the smoothing,
            the ``mu`` in ``(M_2 + mu L_2)^-1 M_2``.
        descent_method: GRADIENT or LBFGS (see :class:`DescentMethod`).
        dt_mode: FIXED or ANALYTIC_LINESEARCH.
        cfl: Cap on the step: ``dt = min(dt_star, cfl / cfl_max)`` with
            ``cfl_max`` the largest logical CFL number of the velocity. The
            linesearch minimiser cannot raise the energy, but a large step
            leaves the ideal-induction flow (frozen-in topology violated at
            O(dt^2)) and diverges when ``||dB||`` collapses. ``inf`` disables
            the cap and leaves the trajectory untouched.
        cfl_weights: ``logical_cfl_weights(seq)``, built by ``__post_init__``.
        resistive: Compile the resistive solve into the step. False (the
            default) traces the ideal step only: no ``cond``, no MINRES,
            regardless of ``state.eta``. Set it when the run has ``eta > 0``
            anywhere.
        eta_every: Apply the resistive solve every ``eta_every`` steps,
            diffusing over the time accumulated since the last one. In
            float32 a per-step increment of 1-10 ulps of ``B`` is not
            representable, so ``eta ~ 1e-4`` at ``dt ~ 1e-3`` needs 10-100
            here; the diffusive time of even the finest mode,
            ``1 / (eta lambda_max) ~ 1000`` such steps, makes ``<= 100``
            physically harmless.
        history_size: Number of stored secant pairs for L-BFGS (1 = the
            CG-equivalent default). The history arrays exist only under
            L-BFGS; GRADIENT carries none.
        dirichlet_H: Dirichlet BC on H.
    """
    seq: DeRhamSequence
    velocity_smoothing_order: int = 0
    velocity_smoothing_scale: float = 0.0
    descent_method: DescentMethod = DescentMethod.LBFGS
    dt_mode: TimeStepChoice = TimeStepChoice.ANALYTIC_LINESEARCH
    cfl: float = 0.5
    eta_every: int = 1
    resistive: bool = False
    history_size: int = 1
    dirichlet_H: bool = False
    cfl_weights: jnp.ndarray = None

    def __post_init__(self):
        if self.descent_method == DescentMethod.LBFGS and self.history_size < 1:
            raise ValueError("history_size must be at least 1 for L-BFGS.")
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

        Falls back to steepest descent (F) when all history entries are zero.
        """
        m = self.history_size

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

    def update_field(self, state: State, field_name: Literal['B_n', 'B_nplus1', 'v', 'p_v', 'H', 'JxH', 'E', 's_history', 'y_history', 'F_prev', 'MF_prev', 'Ms_history', 'My_history', 'A', 'dt', 'dt_star', 'cfl_max', 'eta', 'resistive_info', 'resistive_delta', 'resistive_count', 'resistive_time', 'F_norm', 'v_norm', 'lbfgs_sy'], value) -> State:  # noqa: E501
        return eqx.tree_at(
            lambda s: getattr(s, field_name),
            state,
            value
        )

    def relaxation_step(self, state: State) -> State:
        """Advance ``state.B_n`` by one step into ``state.B_nplus1``.

        Operator-split (Lie): ideal transport, then implicit resistive
        diffusion. The ideal half is explicit Euler on the descent
        velocity, ``B_ideal = B_n + dt curl(u x H)``; the resistive half is
        backward Euler on ``B_ideal``. The splitting is first order in
        ``dt``; resistivity is not applied to ``B_n``.
        """
        B_n = state.B_n
        F, p, _, H, JxH = compute_force(
            B_n, self.seq, dirichlet_H=self.dirichlet_H,
            p_guess=state.p, H_guess=state.H, JxH_guess=state.JxH)
        # M F ONCE.  It serves ||F||_M and the L-BFGS secant
        # M y = M F_prev - M F; the step applies M_2 three times in total
        # (M F, M u, M dB) whichever method is running -- L-BFGS used to
        # apply it 4m + 6 times.
        MF = self.seq.apply_mass_matrix(F, 2)

        # The secant history exists only under L-BFGS (a static branch: the
        # other methods carry (0, n) arrays and never touch them).
        lbfgs = self.descent_method == DescentMethod.LBFGS
        s_hist, Ms_hist = state.s_history, state.Ms_history
        y_hist, My_hist = state.y_history, state.My_history
        if lbfgs:
            # --- history bookkeeping, part 1: push y BEFORE the direction ---
            # y_{k-1} = grad_M E_k - grad_M E_{k-1} = F_prev - F is a
            # difference over the step that ALREADY happened, so it pairs with
            # s_{k-1}, which the end of the previous step put in s_history[0].
            # Pushing it HERE, rather than next to the brand-new s_k at the end
            # of this step, is what keeps (s_i, y_i) aligned.  Pushing it at
            # the end instead leaves y lagging its paired s by exactly one step.
            y_hist = jnp.roll(y_hist, 1, axis=0).at[0].set(state.F_prev - F)
            My_hist = jnp.roll(My_hist, 1, axis=0).at[0].set(state.MF_prev - MF)

        sy = jnp.array(0.0)
        if lbfgs:
            u, sy = self._lbfgs_direction(F, s_hist, y_hist, Ms_hist, My_hist)
        elif self.descent_method == DescentMethod.GRADIENT:
            u = F
        else:
            raise ValueError(
                f"Unknown descent_method: {self.descent_method}. Supported methods are given by the DescentMethod enum.")
        u = self.smooth_velocity(u)
        u, p_v = self.seq.apply_leray_projection(u, k=2, p_guess=state.p_v)
        # M u once: the linesearch numerator, ||u||_M and the stored M s.
        Mu = self.seq.apply_mass_matrix(u, 2)

        # u at the quadrature points once: the cross product and the CFL
        # number both read it.
        u_jk = self.seq.evaluate_at_quadrature(u, 2, True)
        H_jk = self.seq.evaluate_at_quadrature(H, 1, self.dirichlet_H)
        E_dual = self.seq.cross_product_load_values(u_jk, H_jk, 1, 2, 1, True)
        E = self.seq.apply_inverse_mass_matrix(E_dual, 1, guess=state.E)
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
        dB = self.seq.apply_incidence_matrix(
            E, 1, dirichlet_in=True, dirichlet_out=True)
        if self.dt_mode == TimeStepChoice.FIXED:
            dt_star = state.dt
        elif self.dt_mode == TimeStepChoice.ANALYTIC_LINESEARCH:
            dt_star = F @ Mu / self.seq.l2_norm_sq(dB, 2)
        else:
            raise ValueError(
                f"Unknown dt_mode: {self.dt_mode}. Supported modes are given by the TimeStepChoice enum.")
        # The CFL cap.  cfl = inf gives min(dt_star, inf) = dt_star exactly.
        dt = jnp.minimum(dt_star, self.cfl / cfl_max)
        B_ideal = B_n + dt * dB

        # Resistive diffusion, BACKWARD Euler on top of the ideal step, in
        # Second half of the Lie splitting: backward Euler on B_ideal in
        # DEFECT form:
        #     (M_2 + eps L_2) delta = -eps L_2 B_ideal,   B_{n+1} = B_ideal + delta,
        # eps = eta * (time since the last resistive solve), guess 0.
        # It used to be explicit, E = E_ideal - eta J inside dB, which is
        # stable only for dt eta <~ h^2 -- a limit the linesearch knows
        # nothing about, so eta had to be kept small and scheduled.  The
        # implicit form is unconditionally stable and dissipative:
        # (I + eps M^-1 L)^-1 is an M-contraction, so E(B_{n+1}) <=
        # E(B_ideal) <= E(B_n) with the linesearch dt of the IDEAL step.  It
        # also preserves div B in exact arithmetic (M^-1 L maps ker(div) into
        # itself; the rhs is in range(D_1) since S_2 B_ideal = 0), so the
        # topological curl's div B = 0 survives to the solver's tolerance.
        # The defect form is what makes the solve MEAN something in float32:
        # solving for B itself with a tolerance relative to ||B|| returns
        # B_ideal unchanged when the correction is a few ulps (eps ~ 1e-7),
        # whereas the tolerance here is relative to delta in both precisions.
        # eta_every batches the diffusion over several steps for the same
        # reason.  The cond skips the solve at zero cost when it is not due;
        # a run without resistivity sets ``resistive=False`` and never traces it.
        resistive_time = state.resistive_time + dt
        resistive_count = state.resistive_count + 1
        if self.resistive:
            due = (state.eta > 0) & (resistive_count >= self.eta_every)

            def resistive(B):
                return resistive_step(B, self.seq, resistive_time * state.eta)

            def skip(B):
                return B, jnp.int32(0), jnp.zeros((), B.dtype)

            B_nplus1, resistive_info, resistive_delta = jax.lax.cond(
                due, resistive, skip, B_ideal)
            resistive_time = jnp.where(due, 0.0, resistive_time)
            resistive_count = jnp.where(due, 0, resistive_count)
        else:
            # A non-resistive run: the ideal step is the whole step. Decided in
            # Python, so the trace contains no solve and no branch.
            B_nplus1 = B_ideal
            resistive_info = jnp.int32(0)
            resistive_delta = jnp.zeros((), B_ideal.dtype)

        # --- history bookkeeping, part 2: push the step just taken ----------
        # The descent variable is the VELOCITY u, not B: grad_M E = -F is the
        # derivative of E with respect to u (dE = -(F, u)_M), and the direction
        # the recursion returns is consumed as a velocity.  So the step in the
        # descent variable is dt*u.  B moves by dt*curl(u x H) instead, which
        # is a different vector in a different space; storing THAT as s pairs
        # it with a y that is a secant of a different map entirely, and the
        # curvature <s, y>_M goes negative on a third to a half of all steps.
        # See docs/research/handoff_2026-08-25_relaxation_prelim.md.
        if lbfgs:
            s_hist = jnp.roll(s_hist, 1, axis=0).at[0].set(dt * u)
            Ms_hist = jnp.roll(Ms_hist, 1, axis=0).at[0].set(dt * Mu)

        return eqx.tree_at(
            lambda s: (s.B_nplus1, s.v, s.p, s.p_v, s.H, s.JxH, s.E,
                       s.F_prev, s.MF_prev, s.F_norm, s.v_norm, s.lbfgs_sy,
                       s.dt, s.dt_star, s.cfl_max, s.resistive_info,
                       s.resistive_delta, s.resistive_count, s.resistive_time,
                       s.s_history, s.y_history, s.Ms_history, s.My_history),
            state,
            (B_nplus1, u, p, p_v, H, JxH, E,
             F, MF, jnp.sqrt(F @ MF), jnp.sqrt(u @ Mu), sy,
             dt, dt_star, cfl_max, resistive_info,
             resistive_delta, resistive_count, resistive_time,
             s_hist, y_hist, Ms_hist, My_hist))


def initial_state(B_dof: jnp.ndarray, ts: TimeStepper, dt: float = 1.0) -> State:
    """Build the state at ``B_dof`` with its force already evaluated.

    ``F_prev``, ``MF_prev``, ``F_norm`` and the warm-start guesses ``p``,
    ``H``, ``JxH`` are seeded from one ``compute_force`` here, so the first
    step's secant ``y = F_prev - F`` sees the true previous
    gradient.  Callers used to repeat this seeding by hand (and could not
    have seeded ``MF_prev``, which did not exist).
    """
    seq = ts.seq
    n = seq.n(2, True)
    m = ts.history_size if ts.descent_method == DescentMethod.LBFGS else 0
    F0, p0, _, H0, JxH0 = compute_force(B_dof, seq, dirichlet_H=ts.dirichlet_H)
    MF0 = seq.apply_mass_matrix(F0, 2)
    return State(
        B_n=B_dof,
        dt=dt,
        dt_star=dt,
        resistive_info=jnp.int32(0),
        resistive_count=jnp.int32(0),
        v=jnp.zeros(n),
        p=p0,
        p_v=jnp.zeros(seq.n(3, True)),
        H=H0,
        JxH=JxH0,
        E=jnp.zeros(seq.n(1, True)),
        A=jnp.zeros(seq.n(1, True)),
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
             resistive_info, velocity_norm, divergence_B, eta, iteration
    """
    seq = ts.seq
    state = initial_state(B_dof, ts, dt0)

    def body_fn(state, _):
        state = ts.relaxation_step(state)
        state = eqx.tree_at(lambda s: s.B_n, state, state.B_nplus1)
        return state, None

    @jax.jit
    def _run_scan(state):
        return jax.lax.scan(body_fn, state, None, length=num_iters_inner)

    get_helicity = jax.jit(compute_helicity, static_argnames=["seq"])
    get_energy = jax.jit(lambda B: 0.5 * seq.l2_norm_sq(B, 2))
    get_div_norm = jax.jit(compute_divergence_norm, static_argnames=["seq"])

    traces = {k: [] for k in (
        "force_norm", "helicity", "timestep", "energy", "resistive_info",
        "velocity_norm", "divergence_B", "eta", "iteration")}

    def record(state, iteration):
        traces["force_norm"].append(state.F_norm)
        h, A_new = get_helicity(state.B_n, seq, state.A)
        traces["helicity"].append(h)
        traces["timestep"].append(state.dt)
        traces["energy"].append(get_energy(state.B_n))
        traces["resistive_info"].append(state.resistive_info)
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
        if resistivity_schedule is not None:
            state = eqx.tree_at(lambda s: s.eta, state,
                                resistivity_schedule(i))

        state, _ = _run_scan(state)

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
