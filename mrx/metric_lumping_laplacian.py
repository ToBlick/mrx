"""Block-Jacobi Hodge-Laplacian preconditioner: separable bulk + dense core.

THE PRODUCTION Laplacian and mass preconditioner for k = 0..3, free and
Dirichlet (``docs/PRODUCTION.md``; ``kind='metric_lumping'``).

The shape is the same at every ``k``:

* **bulk** -- one separable atom per component, a three-term Kronecker sum
  ``K_r (x) M_t (x) M_z + M_r (x) K_t (x) M_z + M_r (x) M_t (x) K_z``, inverted
  by fast diagonalisation (:func:`mrx.operators._fd_apply_3d`);
* **core** -- the polar rows, probed and inverted densely;
* the two blocks are NOT coupled: block Jacobi, not a Schur complement.

The one construction that is not obvious is where the third Kronecker term
comes from at ``k = 1`` and ``k = 2``.  Take the ``r`` component of a 1-form,
``dLam_r (x) Lam_t (x) Lam_z``.  Curl-curl differentiates along ``t`` and ``z``,
where the factors are primal splines, so those give honest 1-D stiffnesses.
Div-div acts along ``r``, where the factor is ALREADY a derivative spline --
and it does not differentiate it.  ``delta`` steps BACKWARDS through the
incidence to ``V_0``, applies the inverse mass there, and comes back::

    W_1^{rr} ~ (x)_a [ M_a G_a A_a^-1 G_a^T M_a ] ,  G_r = incidence, G_t = G_z = I

so every axis makes the round trip ``V_1 -> V_0 -> V_1``; only the axis being
differentiated carries the incidence.  The ``r`` factor

    Ktilde_r = M_r^(d) G_r A_r^-1 G_r^T M_r^(d)

is symmetric PSD, built from the derivative-spline mass, the metric-free
incidence and the PRIMAL mass inverse -- no second derivatives and nothing
outside the complex.  ``k = 2`` is the mirror image: div-div gives an honest
stiffness on the component's own (primal) axis, and the curl-curl weak half
gives ``Ktilde`` on the other two.

Conditioning works out the right way round: the curl-curl stiffnesses are
singular (constants), ``Ktilde`` is SPD, and the SUM is nonsingular -- which is
why the two halves are combined into one atom rather than inverted separately
and added.

Approximations, all deliberate:

1. off-diagonal component blocks dropped (metric off-diagonals and the
   cross-component derivative couplings);
2. each bundled 3-D metric weight collapsed to a product of axis profiles;
3. ``M_0`` inside the weak term taken as a pure Kronecker product, which drops
   its ``Lam`` diagonal;
4. ``M A^-1 M -> M`` on the undifferentiated axes, so one mass per axis serves
   all three terms (needed for a shared eigenbasis);
5. no bulk<->core coupling.

The natural-BC boundary term IS carried, by default, as a rank-one update to
the radial stiffness (``bc_entry="ibpd"``; ``"exact"`` is the probed
alternative): under a free condition the weak
block's integration by parts leaves ``int_{r=1} w u_r^2``, which for a tensor
basis is ``alpha (e e^T) (x) M_t (x) M_z`` -- the same shape as the first
Kronecker term, so it merges into ``K_r`` for free.  It is exactly zero under
Dirichlet and at k=0 (no weak block).  Its limit is that folding it into the
sum forces the face weight ``w(1,theta,zeta)`` down to a SCALAR: worth 1.5x ->
3.3x on a toroid, only 1.26x -> 1.56x on W7-X, where the face weight varies too
much for a scalar. Carrying that variation needs an exact outer ring.
"""

from __future__ import annotations

import functools
import os

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.operators import (
    _assemble_weighted_1d_mass,
    _fd_apply_3d,
    _assemble_weighted_1d_stiffness,
    _dense_incidence_1d,
    _reshape_quadrature_matrix_field,
    _reshape_quadrature_scalar_field,
)
from mrx.precision import DTYPE, eps, sqrt_eps
from mrx.preconditioners import _simultaneous_diagonalize_pair

#: Relative cut-off below which an eigenvalue of the probed dense core is
#: treated as exactly zero. ~1e-12 in float64 (the value it was tuned at).
CORE_TOL = eps(4096.0)

# --------------------------------------------------------------------------- #
# Bundled axis profiles                                                        #
# --------------------------------------------------------------------------- #

def _polar_cut_weight(seq):
    """Radial quad weight with the polar-surgery element removed.

    The core DOFs are handled by their own dense block, so they must not
    contribute to the bulk averages -- this is the ``wx_cut`` convention from
    :func:`mrx.operators._k0_bundled_axis_profiles`, carried over unchanged.
    """
    xi1 = jnp.asarray(seq.basis_0.Λ[0].T)[seq.ps[0] + 1]
    return seq.quad.w_x * (jnp.asarray(seq.quad.x_x) >= xi1)


def bundled_axis_profiles(seq, field):
    """Quad-weighted axis means of one BUNDLED weight field.

    ``field`` is a scalar quadrature field already reshaped to ``(qx, qy, qz)``.
    Bundled means the product ``g * J`` is averaged as a unit, never ``g`` and
    ``J`` separately: ``g^tt J ~ 1/r`` stays integrable where the bare
    ``g^tt ~ 1/r^2`` does not.  Arithmetic means, not harmonic -- harmonic
    profiles were measured to degrade badly off-axis (W7-X 88 -> 152 at
    16x32x32).
    """
    wx = _polar_cut_weight(seq)
    wy, wz = seq.quad.w_y, seq.quad.w_z
    sx, sy, sz = jnp.sum(wx), jnp.sum(wy), jnp.sum(wz)
    pr = jnp.einsum('qrs,r,s->q', field, wy, wz) / (sy * sz)
    pt = jnp.einsum('qrs,q,s->r', field, wx, wz) / (sx * sz)
    pz = jnp.einsum('qrs,q,r->s', field, wx, wy) / (sx * sy)
    return pr, pt, pz


def weight_fields(seq):
    """The metric weight families the atoms need, at quadrature points.

    Three distinct families, one per mass appearing in ``L_k``:

    ==========  ===================  ==================================
    symbol      space                weight
    ==========  ===================  ==================================
    ``jac``     0-form mass          ``J``
    ``ginvJ``   1-form mass          ``g^{aa} J``
    ``gJinv``   2-form mass          ``g_{aa} / J``
    ``invjac``  3-form mass          ``1 / J``
    ==========  ===================  ==================================
    """
    jac = jnp.transpose(
        _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j), (1, 0, 2))
    ginv = jnp.transpose(
        _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl),
        (1, 0, 2, 3, 4))
    met = jnp.transpose(
        _reshape_quadrature_matrix_field(seq, seq.geometry.metric_jkl),
        (1, 0, 2, 3, 4))
    return {
        "jac": jac,
        "invjac": 1.0 / jac,
        "ginv_aa": tuple(ginv[..., a, a] for a in range(3)),
        "met_aa": tuple(met[..., a, a] for a in range(3)),
        "ginvJ": tuple(ginv[..., a, a] * jac for a in range(3)),
        # (g^{aa})^2 J -- the weight of the div-div energy when it is written
        # directly as a stiffness of the derivative splines rather than routed
        # through M_0^-1: ||delta u||^2 = int [d_r(J g^rr u_r)]^2 / J
        #                              ~= int (d_r u_r)^2 (g^rr)^2 J.
        "ginv2J": tuple(ginv[..., a, a] ** 2 * jac for a in range(3)),
        "gJinv": tuple(met[..., a, a] / jac for a in range(3)),
    }


# --------------------------------------------------------------------------- #
# 1-D factors                                                                  #
# --------------------------------------------------------------------------- #

def _axis_bases(seq):
    """Bases and the FULL quadrature weights.

    The polar cut belongs to :func:`bundled_axis_profiles` -- it decides which
    elements contribute to an AVERAGE. Assembling the 1-D matrices against a cut
    weight instead makes the radial primal mass singular (the basis functions
    living only in the cut element get zero rows). The core is excluded from the
    bulk atom by RESTRICTING THE RADIAL WINDOW, exactly as the k=0 path does
    with ``_restrict_radial_window``, not by zeroing quadrature weights.
    """
    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)
    return primal, deriv, quad_w
def _ktilde_1d(seq, axis, mass_deriv, profile_primal, window=None):
    """``Ktilde = M^(d) G A^-1 G^T M^(d)`` -- the weak half's 1-D factor.

    This is what replaces "the stiffness of the derivative spline", which does
    not exist: the basis on this axis is already differentiated, so ``delta``
    goes back through the incidence to the primal space, inverts the mass
    there, and returns.  Symmetric PSD by construction.
    """
    primal, _, quad_w = _axis_bases(seq)
    types = seq.basis_0.types
    a_primal = _assemble_weighted_1d_mass(
        primal[axis], quad_w[axis] * profile_primal)
    g = _dense_incidence_1d(int(a_primal.shape[0]), types[axis])
    # Restrict only the V_k (row) side: the round trip still passes through the
    # FULL primal space, which is what keeps A invertible.
    if window is not None:
        g = g[window[0]:window[0] + window[1], :]
    if g.shape[0] != mass_deriv.shape[0]:
        raise ValueError(
            f"axis {axis}: incidence gives {g.shape[0]} rows after the window "
            f"but the derivative mass is {mass_deriv.shape[0]}")
    inner = jnp.linalg.solve(a_primal, g.T)                  # A^-1 G^T
    k = mass_deriv @ (g @ inner) @ mass_deriv
    return 0.5 * (k + k.T)


def _fd_stiffness_degree0(seq, axis, profile):
    """Self-contained p=1 stand-in for the honest derivative-spline stiffness.

    At ``p = 1`` the derivative splines are degree 0 -- piecewise constants --
    so ``int (dLam_i)' (dLam_j)'`` does not exist: the space contains no
    non-constant ``H^1`` function.  The natural discrete Dirichlet energy there
    is the JUMP seminorm over faces,

        sum_faces  t_f (u_{i+1} - u_i)^2 ,   t_f = <w>_f / d_f

    i.e. the DG-0 / finite-volume form, assembled here as ``D^T diag(t) D`` with
    ``D`` the first difference and ``t`` the standard harmonic-mean
    transmissibility.  Non-conforming, which is fine: a preconditioner needs
    spectral equivalence, not conformity.  It has the same constant kernel as
    the ``p >= 2`` honest stiffness, so it drops into the same Kronecker sum.

    Deliberately NOT routed through the round-trip form: that path is a
    different model with its own problems, and a fallback should not inherit
    them.

    Assembled in the D-spline COEFFICIENT basis, which is not the value basis --
    see the ``diag(1/h)`` conjugation below.

    Host numpy: this is knot-vector bookkeeping at p = 1 only, cast to the
    working dtype once at the end.
    """
    lam = seq.basis_0.Λ[axis]
    nodes = np.asarray((seq.quad.x_x, seq.quad.x_y, seq.quad.x_z)[axis])
    edges = np.asarray(lam.greville_points())
    periodic = lam.type == "periodic"
    if periodic:
        edges = np.concatenate([edges, [edges[0] + 1.0]])
    h = np.diff(edges)
    n_cell = h.size

    # Cell-mean weight: bucket the axis quadrature points into the cells.
    prof = np.asarray(profile)
    idx = np.clip(np.searchsorted(edges, nodes, side="right") - 1, 0, n_cell - 1)
    w = np.array([prof[idx == i].mean() if np.any(idx == i) else prof.mean()
                  for i in range(n_cell)])

    centre = 0.5 * (edges[:-1] + edges[1:])
    if periodic:
        pairs = [(i, (i + 1) % n_cell) for i in range(n_cell)]
        dist = np.array([abs(((centre[(i + 1) % n_cell] - centre[i]) + 0.5)
                             % 1.0 - 0.5) for i in range(n_cell)])
    else:
        pairs = [(i, i + 1) for i in range(n_cell - 1)]
        dist = np.diff(centre)

    d = np.zeros((len(pairs), n_cell))
    trans = np.zeros(len(pairs))
    for f, (i, j) in enumerate(pairs):
        d[f, i], d[f, j] = -1.0, 1.0
        trans[f] = 2.0 / (1.0 / w[i] + 1.0 / w[j]) / dist[f]
    k = d.T @ (trans[:, None] * d)

    # COEFFICIENTS ARE NOT VALUES. The jump energy above is a functional of the
    # cell VALUES, but the atom's unknowns are D-spline coefficients, and the
    # derivative basis is normalised to unit INTEGRAL, not unit height:
    # ``D_i = 1_{cell i} / h_i`` at p = 1 (mrx.spline_bases.DerivativeSpline
    # scales by ``(p+1)/(T[i+p+2]-T[i+1])``). So ``value_i = u_i / h_i`` and the
    # jump form has to be conjugated by ``diag(1/h)``.
    #
    # Without it the factor is under-scaled by h^2 -- and h^2, not a constant,
    # so no fixed multiplier repairs it and the damage GROWS with resolution.
    # Against the derivative mass ``M^d_ii = 1/h_i`` the generalized eigenvalues
    # come out O(1) instead of O(1/h^2), i.e. the radial direction of the atom
    # is left with essentially no stiffness at all.
    # The jump form is assembled on VALUES, not coefficients. Assembling it on
    # coefficients under-scales it by h^2 -- no fixed multiplier repairs that,
    # because the damage grows with resolution. The two diagnostic knobs that
    # once selected between the forms are gone; the fix is landed.
    k = k / np.outer(h, h)
    return jnp.asarray(0.5 * (k + k.T), dtype=DTYPE)


def _h_last(seq):
    """Width of the last radial element, from the knot vector."""
    uniq = np.unique(np.asarray(seq.basis_0.Λ[0].T))
    return float(uniq[-1] - uniq[-2])


def _boundary_point(seq):
    """Where the outer face ``r = 1`` is sampled: just inside the last element.

    A clamped spline evaluated AT ``x = 1`` exactly hits the half-open last
    piece and returns the wrong branch (memory: "spline map DF singular at
    r=1"), so the face is sampled at ``1 - delta``. The nudge is GEOMETRIC --
    a fraction of the last knot span ``h``, so it stays inside that element at
    every resolution -- and the fraction is ``sqrt(eps)`` of the working
    dtype: resolvable next to 1.0 in float32 (where ``1 - 1e-8 == 1``) while
    the O(delta / h) change it makes to the basis value stays at roundoff
    level in either precision.
    """
    return 1.0 - sqrt_eps() * _h_last(seq)


def _face_alpha(seq, k, c, lumped):
    r"""``(scalar, amplification)`` for the natural-BC face term.

    The boundary term is a penalty on the trace of the form at ``r = 1``:
    ``u . n`` at ``k=1``, ``w x n`` at ``k=2``, ``omega`` at ``k=3``, squared
    and integrated over the surface.  With ``n = grad r / sqrt(g^rr)`` and
    ``dsigma = J sqrt(g^rr) dtheta dzeta``, the integrand is ``m_k sqrt(g^rr)``
    at every degree, where ``m_k`` is the component's mass weight::

        k=1   (u.n)^2 dsigma  =  g^rr u_r^2        . J sqrt(g^rr)
        k=2   |w x n|^2       =  g_cc w^c w^c/J^2  . J sqrt(g^rr)
        k=3   omega^2 dsigma  =  omega^2/J^2       . J sqrt(g^rr)

    so the coefficient is the face average of ``m_k sqrt(g^rr)`` over
    ``theta, zeta``, and the amplification is a bare ``1/h`` on the last
    radial element.

    Under ``lumped="diag"`` the component factor ``w_comp = m_k/J`` is carried
    outside as the ``D`` sandwich, so it has to come back out here or it is
    counted twice.  That division is also what makes the coefficient
    ``(k,c)``-dependent -- ``m_k`` survives as a WEIGHT on the average rather
    than cancelling -- which is why the scale is degree-dependent and why it is
    exact on a face where ``J`` is constant.  See the paper, natural-BC section.
    """
    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    wy, wz = seq.quad.w_y, seq.quad.w_z
    norm = jnp.sum(wy) * jnp.sum(wz)

    def fm(field):
        return jnp.einsum('rs,r,s->', field[-1], wy, wz) / norm

    m_k = {0: jac, 1: ginv[c] * jac, 2: met[c] / jac, 3: 1.0 / jac}[k]
    div = fm(m_k / jac) if lumped == "diag" else 1.0
    return fm(m_k * jnp.sqrt(ginv[0])) / div, 1.0 / _h_last(seq)


def _edge_vector(seq, axis, window):
    """``e = dLam_axis(1)``, windowed -- the shape every boundary update uses."""
    dlam = seq.basis_0.dΛ[axis]
    end = _boundary_point(seq) if dlam.type != "periodic" else 0.0
    e = jax.vmap(lambda i: jnp.sum(dlam(end, i)))(dlam.ns)
    if window is not None and axis == 0:
        e = e[window[0]:window[0] + window[1]]
    return e


#: The production natural-BC penalty scale.
#: `alpha` as spelled is the surface integral itself and is the best NORM
#: approximation to `L`'s boundary block, but `P` minimises kappa(P^-1 L),
#: which wants a much smaller number. EMPIRICAL, not derived -- see
#: docs/research/natural_bc_coefficient_handoff.md §16 (what to ship) and
#: §17.5 (why a scale is needed at all).
#:
#: 3.0 is the `penalty` value, from the merged phase2+phase3 grids (24 cells
#: that bracket their optimum: 4 geometries x k=1,2,3 x p=2,3,5 x two meshes),
#: ranked by TOTAL iterations -- sum over cells of iterations at fixed s, over
#: the sum of the per-cell optima. Worst-case ranking is the wrong metric here
#: and says otherwise: it lets a 34-iteration cell outvote a 1450-iteration
#: one. Measured 1.062 / 1.060 / 1.066 at s = 2 / 2.828 / 4, so the basin is
#: flat over [2, 4] and 3.0 is a round number inside it, not a fitted optimum.
#:
#: It is also not an independent fit: 2.828 / 0.10 = 28.3, and the derivation's
#: own conversion factor is c(p)/a = 31 at p=3 (measured 28-32). This is the
#: superseded `product` value of 0.10 pushed through §5.2(e).
#:
#: NO p CAVEAT. `product` needed one ("prefer 0.05 at p >= 5") because its
#: mu_0 = c(p)/h carries a c(p) that triples over p=2..5. `penalty` uses a bare
#: 1/h and its best single s is 2 / 2 / 2.83 at p = 2 / 3 / 5 -- a 1.4x drift
#: costing 1.8-3.5%, i.e. nothing. Do not re-add a degree-dependent default.
#:
#: KNOWN GAP: on rot-ellipse at p=5, k=1 and k=2, the sweep grid stopped at
#: s=8 with the optimum still there, so those two cells are not bracketed and
#: are excluded from the 24. Holding s=3 costs ~10% on them (630 vs 574, 635 vs
#: 578), which is the one place `penalty` at a fixed scale loses to `product`.
#: Including them moves the head-to-head from +0.8% to +1.7%. Shipped with that
#: known and accepted; extending that grid is 2 jobs if it ever looks relevant.
PRODUCTION_BC_SCALE = 3.0


def _resolve_bc_scale(bc_scale=None):
    """Resolve the natural-BC penalty scale.

    Precedence, highest first: the explicit ``bc_scale`` argument, then
    ``MRX_BJ_BC_SCALE``, then :data:`PRODUCTION_BC_SCALE`. **An explicit
    argument always wins.**

    The ordering was the other way round until 2026-08-25 -- the environment
    variable overrode the argument -- on the reasoning that the sweep harnesses
    always set the variable, so every recorded arm kept its meaning. The cost
    was that a caller passing ``bc_scale=2.0`` was silently ignored whenever
    the variable happened to be set, including by a leftover export in a shell,
    and nothing reported it. A hidden factor that overrides an explicit one is
    exactly the failure the no-implicit-weights rule exists to prevent.

    The flip changes NO existing caller. Checked repo-wide: nothing sets the
    variable AND passes the argument. The four sweep harnesses
    (``verify_block_jacobi``, ``block_jacobi_spectrum``, ``bench_real_solves``,
    ``bc_schur_effective``) set the variable and pass no argument, so the env
    still supplies their default; ``test_metric_lumping_laplacian``'s fixture
    does the same, and its defaults test pops the variable before passing an
    argument. Recorded arms keep meaning what they meant.
    """
    if bc_scale is not None:
        return float(bc_scale)
    env = os.environ.get("MRX_BJ_BC_SCALE")
    if env is not None:
        return float(env)
    return PRODUCTION_BC_SCALE


def _boundary_entry_direct(seq, axis, weight_field, window, dirichlet,
                           scalar=None, bc_scale=None):
    """The natural-BC boundary term, straight from the surface integral.

    Under a natural (free) condition the weak block's integration by parts
    leaves ``int_{r=1} w . u_r^2``. For a tensor basis that is

        alpha . (e e^T) (x) M_t (x) M_z ,   e = dLam(1),  alpha = <w>_{theta,zeta}(1)

    which has exactly the shape of the FIRST Kronecker-sum term, so it merges
    into ``K_r`` as a rank-one update. Nothing about the sum, the shared
    eigenbasis, the cost or the storage changes.

    ``alpha`` needs no fitting and no reference to the exact factor ``F``: it is
    the same weight family the stiffness already uses, evaluated AT the boundary
    instead of averaged over ``r``. Returns zero under Dirichlet, where the test
    function vanishes on the boundary and the term does not exist.
    """
    if dirichlet:
        return None
    if scalar is not None:
        alpha = scalar
    else:
        wy, wz = seq.quad.w_y, seq.quad.w_z
        # <w> over theta,zeta on the last radial quadrature slice.
        alpha = (jnp.einsum('rs,r,s->', weight_field[-1], wy, wz)
                 / (jnp.sum(wy) * jnp.sum(wz)))

    # Penalty STRENGTH. The natural condition here is u.n = 0 -- an essential
    # condition on the normal trace, which the free-BC weak block enforces by a
    # mesh-dependent penalty rather than by removing a DOF. alpha as assembled
    # is the exact surface integral; this knob asks whether the atom wants the
    # exact penalty or the hard u_r = 0 limit it approximates.
    alpha *= _resolve_bc_scale(bc_scale)

    e = _edge_vector(seq, axis, window)
    return alpha * jnp.outer(e, e)
def component_factors(seq, k, c, window=None, ktilde_mode="honest",
                      lumped="diag", bc_entry="ibpd", dirichlet=False,
                      bc_scale=None):
    """``(masses, stiffnesses)`` per axis for component ``c`` of ``L_k``.

    The component's basis is a derivative spline on the axes it is
    differentiated on by ``d``, and primal on the others.  Whichever axis is
    already a derivative gets ``Ktilde`` from the weak half; the primal axes get
    an honest stiffness from the stiffness half.

    * k=1: derivative on axis ``c``   -> Ktilde on ``c``, K on the other two.
    * k=2: primal on axis ``c``       -> K on ``c``, Ktilde on the other two.

    The MASS is the same on every axis and every term -- taken from the space's
    own mass, ``g^{cc} J`` at k=1 and ``g_{cc} / J`` at k=2 -- because fast
    diagonalisation needs one mass per axis. The stiffnesses keep their own
    weights; the generalized problem ``K v = lam M v`` does not require them to
    agree.
    """
    if k not in (0, 1, 2, 3):
        raise ValueError("component_factors handles k = 0..3")
    primal, deriv, quad_w = _axis_bases(seq)
    fields = weight_fields(seq)

    # The space's own mass weight, and the weight of the stiffness half.
    if ktilde_mode not in ("roundtrip", "honest"):
        raise ValueError(f"unknown ktilde_mode {ktilde_mode!r}")
    degree0 = int(seq.basis_0.Λ[0].p) < 2

    # ONE formula for every degree:
    #
    #     w(k, c, a) = [mass weight of component c at level k] * g^{aa}
    #
    #     k=0  J            -> g^{aa} J              (the validated fd atom)
    #     k=1  g^{cc} J     -> g^{cc} g^{aa} J
    #     k=2  g_cc / J     -> g_cc g^{aa} / J
    #     k=3  1 / J        -> g^{aa} / J
    #
    # Each reproduces the terms derived separately: at k=1, a=c gives
    # (g^{cc})^2 J (div-div) and a!=c gives g_dd/J (curl-curl, _CURL_CONTRIB);
    # at k=2, a=c gives 1/J (_DIV_CONTRIB). Uses g^{aa} = 1/g_aa and
    # J^2 = prod g_aa, i.e. an ORTHOGONAL metric -- exact on the toroid,
    # approximate on W7-X where g_{theta zeta} is the largest off-diagonal.
    #
    # An axis is a DERIVATIVE axis exactly where the component's basis is a
    # derivative spline; those get the honest-K, the primal axes get an
    # ordinary stiffness. k=0 has none, k=3 has all three.
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    mass_weight = {0: jac, 1: ginv[c] * jac,
                   2: met[c] / jac, 3: 1.0 / jac}[k]
    deriv_axes = {0: (), 1: (c,), 3: (0, 1, 2)}.get(
        k, tuple(a for a in range(3) if a != c))
    if lumped == "diag":
        # DIAGONAL lumping. w(c,a) = g^{cc} * (g^{aa}J) factors into a
        # component part and an axis part, so assemble the 1-D factors with the
        # k=0 weights ONLY -- shared by every component and every degree -- and
        # carry g^{cc} as a diagonal sandwich instead of inside the averages:
        #
        #     P_c = D_c^{-1/2} FD^{-1} D_c^{-1/2}
        #
        # Unlike scalar lumping (below, measured 2-8x worse) this keeps the
        # FIELD exactly and gives up only its correlation with the axis
        # averages. Same shape as raw_kron's Lam (A x A x A) Lam sandwich.
        stiff_prof = [bundled_axis_profiles(seq, ginv[a] * jac)[a]
                      for a in range(3)]
    elif lumped:
        # Separate the per-component factor from the k=0 profile as a SCALAR:
        # component-independent factors, but the bundling principle is broken.
        scalar = float(jnp.mean(mass_weight / jac))
        stiff_prof = [scalar * bundled_axis_profiles(seq, ginv[a] * jac)[a]
                      for a in range(3)]
    else:
        stiff_prof = [bundled_axis_profiles(seq, mass_weight * ginv[a])[a]
                      for a in range(3)]

    primal_prof = bundled_axis_profiles(seq, fields["jac"])

    def cut(mat, axis):
        """Radial window: the bulk atom lives on the bulk DOFs only."""
        if axis != 0 or window is None:
            return mat
        lo, n = window
        return mat[lo:lo + n, lo:lo + n]

    masses, stiffs, ratios = [], [], []
    for a in range(3):
        basis = deriv[a] if a in deriv_axes else primal[a]
        # UNWEIGHTED mass, per the validated k=0 fd/fdbund recipe
        # (_assemble_k0_greville_bulk_factors): the bundled metric goes into the
        # STIFFNESS profiles only. In K_r (x) M_t (x) M_z the M's are just
        # "int phi phi in the other directions" -- g^{aa}J has already been
        # folded into K_r by averaging over exactly those directions, so
        # weighting the masses as well double counts it.
        m_full = _assemble_weighted_1d_mass(basis, quad_w[a])
        m = cut(m_full, a)
        masses.append(m)
        # M A^-1 M ~ c M on the axes the weak half does NOT differentiate. The
        # incidence is the identity there, but M_0^-1 is a 3-D inverse, so the
        # factor does not disappear -- it contributes a SCALE. Dropping it
        # mis-weights the two halves of the Kronecker sum by ~g^{cc} per axis
        # (81x on an epsilon=1/3 toroid), which is enough to wreck the atom.
        # c = mean eig(A^-1 M) = tr(A^-1 M) / n, exact when the two weight
        # profiles are proportional. Only the round-trip form uses it; the
        # honest stiffness carries its own weight (alpha is all ones below).
        if a in deriv_axes or ktilde_mode == "honest":
            ratios.append(1.0)
        else:
            a_full = _assemble_weighted_1d_mass(
                primal[a], quad_w[a] * primal_prof[a])
            ratios.append(float(jnp.trace(jnp.linalg.solve(cut(a_full, a), m))
                                / m.shape[0]))
        if a in deriv_axes and ktilde_mode == "honest":
            # The honest thing: the 1-D stiffness OF the derivative splines.
            # With their derivative values tabulated, that is just a weighted
            # mass of the table -- no incidence, no A^-1, and so nothing to
            # mis-scale (the M A^-1 M factor exists only because the round trip
            # drags M_0^-1 in).
            if not degree0 and not hasattr(seq, "_bj_dd_tables"):
                from mrx.local_assembly import (  # noqa: PLC0415
                    _second_derivative_tables)
                seq._bj_dd_tables = _second_derivative_tables(seq)
            prof = stiff_prof[a]
            if degree0:
                # p = 1: the DG-0 jump stand-in, in the same (value-scaled)
                # normalisation as the honest form -- so the natural-BC block
                # below applies to it unchanged. e = dLam(1) is (0,...,1/h) at
                # degree 0, which is that same convention on the face.
                kt = cut(_fd_stiffness_degree0(seq, a, prof), a)
            else:
                kt = cut(_assemble_weighted_1d_mass(
                    seq._bj_dd_tables[a], quad_w[a] * prof), a)
            # a is a derivative axis here; the trace only lives on the
            # RADIAL one (a == 0), the boundary face being r = 1.
            if bc_entry == "ibpd" and a == 0:
                # a == 0 ONLY: the boundary face is r = 1; theta and zeta are
                # periodic and have no boundary, so an entry added there is
                # pure noise. (It was, once: a missing guard put entries on the
                # periodic axes and perturbed the Dirichlet cases where the
                # term must vanish identically.)
                #
                # The coefficient is the face average of the squared trace
                # against the surface element; see _face_alpha.
                #
                # Other spellings of this term were measured and lost; see
                # docs/research/natural_bc_coefficient_handoff.md §9, §12.3 and
                # §14.3. Do not re-add them: the exact 2-D face shape and the
                # cross-term corrections (one of which is INDEFINITE) are both
                # refuted.
                scalar, amp = _face_alpha(seq, k, c, lumped)
                w_face = mass_weight * ginv[a]
                corr = _boundary_entry_direct(
                    seq, a, w_face, window, dirichlet, scalar=scalar,
                    bc_scale=bc_scale)
                if corr is not None:
                    kt = kt + corr * amp
            stiffs.append(kt)
        elif a in deriv_axes:
            stiffs.append(_ktilde_1d(seq, a, m, primal_prof[a],
                                     window=window if a == 0 else None))
        else:
            k_full = _assemble_weighted_1d_stiffness(
                primal[a], deriv[a], quad_w[a] * stiff_prof[a],
                _dense_incidence_1d(int(m_full.shape[0]),
                                    seq.basis_0.types[a]))
            stiffs.append(cut(k_full, a))

    # One alpha per Kronecker term: the weak-half terms carry the scale of the
    # OTHER axes' round trips, the stiffness-half terms carry none.
    # The honest stiffness carries its own weight, so there is no round-trip
    # scale to restore: alpha is all ones.
    alpha = tuple(1.0 for _ in range(3)) if ktilde_mode == "honest" else tuple(
        float(np.prod([ratios[b] for b in range(3) if b != a]))
        if a in deriv_axes else 1.0
        for a in range(3))
    return tuple(masses), tuple(stiffs), alpha


def component_diagonal(seq, k, c, shape):
    """Support-averaged component factor at each DOF: ``D_i``.

    ``D_i = int phi_i^2 (w_comp J) / int phi_i^2 J`` -- the numerator is the
    k-form mass diagonal and the denominator the same basis against the 0-form
    weight, so the ratio is the component factor averaged over each basis
    function's own support. Exact, no fit.
    """
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz

    def rs(field):
        return jnp.asarray(field).reshape(ny, nx, nz).transpose(1, 0, 2)

    fields = weight_fields(seq)
    jac = fields["jac"]
    w_comp = {0: jnp.ones_like(jac), 1: fields["ginv_aa"][c],
              2: fields["met_aa"][c] / jac ** 2, 3: 1.0 / jac ** 2}[k]
    primal, deriv, quad_w = _axis_bases(seq)
    deriv_axes = {0: (), 1: (c,), 3: (0, 1, 2)}.get(
        k, tuple(a for a in range(3) if a != c))
    tabs = [(deriv[a] if a in deriv_axes else primal[a]) ** 2
            for a in range(3)]
    wq = rs(seq.quad.w)

    def contract(field):
        f = wq * field
        t1 = jnp.einsum('ax,xyz->ayz', tabs[0], f)
        t2 = jnp.einsum('by,ayz->abz', tabs[1], t1)
        return jnp.einsum('cz,abz->abc', tabs[2], t2)

    num = contract(w_comp * jac)
    den = contract(jac)
    return (num / den).reshape(shape)


def build_bulk_atom(seq, k, c, window=None, ktilde_mode="honest",
                    lumped="diag", bc_entry="ibpd", dirichlet=False,
                    bc_scale=None):
    """Fast-diagonalisation factors for component ``c`` of ``L_k``.

    Returns ``(V_r, V_t, V_z, lam_r, lam_t, lam_z)`` ready for
    :func:`mrx.operators._fd_apply_3d` with ``alpha = (1, 1, 1)``.
    """
    masses, stiffs, alpha = component_factors(seq, k, c, window=window,
                                             bc_scale=bc_scale,
                                              ktilde_mode=ktilde_mode,
                                              lumped=lumped, bc_entry=bc_entry,
                                              dirichlet=dirichlet)
    vs, lams = [], []
    for a in range(3):
        v, lam = _simultaneous_diagonalize_pair(masses[a], stiffs[a])
        vs.append(v)
        lams.append(lam)
    return tuple(vs), tuple(lams), alpha


# --------------------------------------------------------------------------- #
# Core block: probed and densely inverted                                      #
# --------------------------------------------------------------------------- #

def trace_components(k):
    """Components whose radial axis is a DERIVATIVE axis, i.e. the ones the
    integration by parts touches and the only ones carrying a boundary trace::

        k=0  ()        W_0 = 0
        k=1  (r,)      <u, grad tau>  -> int (u.n) tau
        k=2  (t, z)    <w, curl tau>  -> int (w x n).tau
        k=3  (all)     <om, div tau>  -> int om (tau.n)

    These are also exactly the components whose trace is ESSENTIAL in the
    opposite BC family, which is what makes them the ones to pin: fixing them
    on the face kills the tangential derivatives of that component, and the
    remaining components' natural conditions then collapse onto the scalar
    per-component conditions the atom already imposes.  See
    :meth:`MetricLumpingLaplacian.__init__`.
    """
    return {0: (), 1: (0,), 2: (1, 2), 3: (0, 1, 2)}[k]


def core_rows(seq, k, dirichlet, extra_rings=0, outer_rings=0):
    """Extracted rows handled by the dense core block.

    Always the non-selector rows (the polar ring, which mixes raw DOFs and which
    ``wx_cut`` removes from the bulk averages).  ``extra_rings`` FATTENS the
    core by also taking the first ``n`` radial rings exactly -- they are the ones
    where the axis-averaged profiles are worst, because the innermost element is
    where ``g^{tt}J ~ 1/r`` varies fastest.  ``outer_rings`` does the same at
    the OUTER radial boundary, where ``det(DF) = 0`` at the last knot -- a
    Dirichlet condition drops those rows, which is exactly why free-BC solves lag
    dbc ones by 4-6x at every degree while dbc is uniformly strong.  Cost is one
    probe apply per row added, i.e. ``rings * n_theta * n_zeta``, and a larger
    dense inverse.
    """
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows, cols = np.asarray(e.rows), np.asarray(e.cols)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)
    core = np.flatnonzero(counts > 1)
    inner_rows = outer_rows = np.array([], dtype=int)

    if extra_rings > 0 or outer_rings > 0:
        shapes = [tuple(int(v) for v in sh)
                  for sh in getattr(seq, f"basis_{k}").shape]
        starts = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes])
        single = counts[rows] == 1
        r_s, c_s = rows[single], cols[single]
        comp = np.searchsorted(starts[1:], c_s, side="right")
        loc = c_s - starts[comp]
        nt = np.array([sh[1] for sh in shapes])[comp]
        nz = np.array([sh[2] for sh in shapes])[comp]
        nr = np.array([sh[0] for sh in shapes])[comp]
        i_r = loc // (nt * nz)
        if extra_rings > 0:
            inner_rows = r_s[i_r < extra_rings]
            core = np.union1d(core, inner_rows)
        take = np.zeros(i_r.shape, dtype=bool)
        if outer_rings > 0:
            take |= i_r >= nr - outer_rings
        if take.any():
            outer_rows = r_s[take]
            core = np.union1d(core, outer_rows)

    polar = np.flatnonzero(counts > 1)
    inner = np.setdiff1d(inner_rows, polar)
    outer = np.setdiff1d(outer_rows, polar)
    bulk = np.setdiff1d(np.arange(n_ext), core)
    return core, bulk, e, polar, inner, outer






def _probe_rows(apply, size, rows):
    """Dense ``A`` restricted to ``rows``, by one apply per row, on device.

    A Python loop of ASYNCHRONOUS dispatches: nothing here touches the host
    until the block is used, whereas the previous form copied every column
    back (one sync per row). ``lax.map`` over the rows was measured instead
    on 2026-08-26 and rejected: it compiles a fresh scan per ``(k, BC)`` and
    tripled the k = 0..2 build times at (8,16,8), against a loop whose cost
    is a few hundred dispatches of an already-compiled apply.
    """
    if rows.size == 0:
        return jnp.zeros((0, 0), dtype=DTYPE)
    rows_j = jnp.asarray(rows)
    block = jnp.stack(
        [apply(jnp.zeros(size, dtype=DTYPE).at[int(i)].set(1.0))[rows_j]
         for i in rows], axis=1)
    return 0.5 * (block + block.T)


def _dense_symmetric_inverse(block, tol):
    """Pseudoinverse of a symmetric ``block`` dropping ``|w| <= tol max|w|``."""
    if block.size == 0:
        return block
    w, v = jnp.linalg.eigh(block)
    keep = jnp.abs(w) > tol * jnp.max(jnp.abs(w))
    inv_w = jnp.where(keep, 1.0 / jnp.where(keep, w, 1.0), 0.0)
    return (v * inv_w) @ v.T


def probe_core_block(seq, operators, k, dirichlet, rows):
    """Dense ``L_k`` restricted to the core rows, by one apply per row."""
    from mrx.operators import apply_hodge_laplacian_approx  # noqa: PLC0415

    size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
    return _probe_rows(
        lambda x: apply_hodge_laplacian_approx(seq, operators, x, k,
                                               dirichlet=dirichlet),
        size, rows)
# --------------------------------------------------------------------------- #
# The applied payload, as a pytree                                             #
# --------------------------------------------------------------------------- #
#
# WHY THIS EXISTS. `_build_apply` used to close over its arrays and return
# `jax.jit(m_apply)`, so every array reached jit as a CLOSURE CONSTANT. A new
# preconditioner object was a new closure and therefore a new compilation, no
# matter where the object was stored -- measured 2026-08-25 at ~287 ms of
# recurring compile per payload change, about 4,100 applies' worth of work.
#
# As a pytree the arrays are LEAVES passed as arguments. Two payloads with the
# same shapes share a treedef, so changing the NUMBERS reuses the compiled
# apply. `alpha` in particular was `tuple(float(a) for a in alpha)` -- Python
# floats baked in as constants, and alpha is where bc_scale lands.
#
# STATIC vs LEAF, and getting this wrong is the whole risk: leaves are arrays
# whose VALUES change and whose SHAPES do not; static is anything used in
# Python control flow or as a reshape target. Wrong one way and every change
# retraces and the exercise buys nothing; wrong the other and a traced value
# gets used as a Python bool.


class _LumpBlock(eqx.Module):
    """One component's separable atom. Arrays are leaves; the shape is static.

    ``rows`` and ``vals`` are in TENSOR order -- entry ``j`` is the extracted
    row that owns flat DOF ``j`` of the ``shape`` block -- so the input is one
    gather and no index plan is needed.
    """

    rows: jnp.ndarray            # leaf: gather indices, tensor order
    vals: jnp.ndarray            # leaf: extraction weights, tensor order
    v_r: jnp.ndarray             # leaf: per-axis eigenvectors
    v_t: jnp.ndarray
    v_z: jnp.ndarray
    lam_r: jnp.ndarray           # leaf: per-axis eigenvalues
    lam_t: jnp.ndarray
    lam_z: jnp.ndarray
    alpha: jnp.ndarray           # leaf: was a tuple of Python floats
    dscale: jnp.ndarray          # leaf: ALWAYS an array (None was a treedef split)
    shape: tuple = eqx.field(static=True)     # STATIC: reshape target
    # STATIC: when the block's rows are the contiguous range starting at
    # ``offset`` with unit weights (a pure selector, e.g. every k=3 block)
    # the gather and the multiply are a static slice instead.
    offset: int = eqx.field(static=True)      # -1 when not a selector


class _LumpPayload(eqx.Module):
    """Everything the apply reads. One treedef per (k, BC, discretisation).

    CONCATENATING these leaves was tried on 2026-08-25 and REVERTED. The theory
    was that ~35 array arguments cost ~0.6 us each in dispatch, so folding them
    into nine arrays would recover most of the 21.5 us regression. Measured, the
    concatenated version was SLOWER -- 98.0 us against 91.4 -- so argument count
    is not the mechanism and the extra in-trace slicing cost more than it saved.
    See the note above `_apply_lump_payload`.
    """

    blocks: tuple                # leaves, one _LumpBlock per component
    core: jnp.ndarray            # leaf
    core_inv: jnp.ndarray        # leaf
    perm: jnp.ndarray            # leaf: output gather, see _output_permutation
    has_core: bool = eqx.field(static=True)   # STATIC: guards a branch
    identity_perm: bool = eqx.field(static=True)  # STATIC: skip the gather


def _block_input(b, x):
    """``vals * x[rows]`` as the block tensor; a static slice for a selector."""
    n = int(np.prod(b.shape))
    if b.offset >= 0:
        return x[b.offset:b.offset + n].reshape(b.shape)
    return (b.vals * x[b.rows]).reshape(b.shape)


def _block_output(b, sol):
    return sol.reshape(-1) if b.offset >= 0 else b.vals * sol.reshape(-1)


def _place(payload, parts):
    out = parts[0] if len(parts) == 1 else jnp.concatenate(parts)
    return out if payload.identity_perm else out[payload.perm]


def _apply_lump_payload(payload: _LumpPayload, x):
    """The apply, with the payload as an ARGUMENT rather than a closure.

    No scatters: every block gathers its input in tensor order, and the
    per-block results are concatenated and gathered once through ``perm``.
    The previous form wrote one full-length ``out.at[rows].set`` per block
    plus one for the core -- four scatters per apply. Selector blocks and an
    identity output order are static slices / no-ops, so a k=3 apply is the
    fast-diagonalisation solve and nothing else.
    """
    parts = []
    for b in payload.blocks:
        buf = _block_input(b, x) * b.dscale
        sol = _fd_apply_3d(b.v_r, b.v_t, b.v_z,
                           b.lam_r, b.lam_t, b.lam_z, b.alpha, buf)
        parts.append(_block_output(b, sol * b.dscale))
    if payload.has_core:
        parts.append(payload.core_inv @ x[payload.core])
    return _place(payload, parts)


def _tensor_blocks(seq, k, dirichlet, extra_rings=0, outer_rings=0):
    """Split the extraction into per-component tensor blocks plus the core.

    Returns ``(core, bulk, e, polar, inner, outer, blocks)`` where ``blocks``
    holds, per component, ``None`` or ``(rows, vals, (r0, nr), shape, offset)``
    with ``rows``/``vals`` in TENSOR order over the ``(nr, n_t, n_z)`` block
    and ``offset >= 0`` when the block is a pure selector (rows contiguous
    from ``offset``, unit weights). Raises if a component's bulk DOFs are not
    a full radial slab, since the separable atom does not apply then.
    """
    shapes = [tuple(int(s) for s in sh)
              for sh in getattr(seq, f"basis_{k}").shape]
    starts = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    core, bulk, e, polar, inner, outer = core_rows(
        seq, k, dirichlet, extra_rings=extra_rings, outer_rings=outer_rings)
    rows, cols, vals = (np.asarray(e.rows), np.asarray(e.cols),
                        np.asarray(e.vals))
    keep = np.isin(rows, bulk)
    rows_b, cols_b, vals_b = rows[keep], cols[keep], vals[keep]
    comp = np.searchsorted(starts[1:], cols_b, side="right")
    loc = cols_b - starts[comp]

    blocks = []
    for c, shape in enumerate(shapes):
        sel = comp == c
        if not sel.any():
            blocks.append(None)
            continue
        lidx = loc[sel]
        i_r = lidx // (shape[1] * shape[2])
        r0, r1 = int(i_r.min()), int(i_r.max()) + 1
        nr = r1 - r0
        flat = lidx - r0 * shape[1] * shape[2]
        order = np.argsort(flat)
        if not np.array_equal(flat[order], np.arange(nr * shape[1] * shape[2])):
            raise ValueError(
                f"k={k} component {c}: the {lidx.size} bulk DOFs are not the "
                f"tensor block [{r0},{r1}) x {shape[1]} x {shape[2]}; the "
                "separable atom does not apply")
        rows_t, vals_t = rows_b[sel][order], vals_b[sel][order]
        selector = (np.array_equal(rows_t, rows_t[0] + np.arange(rows_t.size))
                    and np.all(vals_t == 1.0))
        blocks.append((rows_t, vals_t, (r0, nr), (nr, shape[1], shape[2]),
                       int(rows_t[0]) if selector else -1))
    return core, bulk, e, polar, inner, outer, blocks


def _output_permutation(block_rows, core, n_ext):
    """``(perm, identity)``: the gather that puts ``concat(block results...,
    core result)`` into place, and whether it is the identity.

    Every extracted row is owned by exactly one bulk block entry or by the
    core -- checked, since the gather silently mis-places rows otherwise.
    """
    owners = np.concatenate(list(block_rows) + [core])
    if not np.array_equal(np.sort(owners), np.arange(n_ext)):
        raise ValueError(
            f"bulk blocks and core cover {owners.size} rows, not every one of "
            f"the {n_ext} extracted rows exactly once")
    perm = np.argsort(owners)
    return jnp.asarray(perm), bool(np.array_equal(perm, np.arange(n_ext)))


# --------------------------------------------------------------------------- #
# Flatten once, compile once per treedef                                       #
# --------------------------------------------------------------------------- #
#
# `eqx.filter_jit(payload, x)` re-PARTITIONS the module on every call, walking
# ~35 leaves each time. Measured 2026-08-25: that cost 194 us per apply
# (69.9 -> 264.4 us/call), against a 284.5 ms saving per payload change --
# break-even at ~1,460 applies, and a production k>=1 solve runs thousands. The
# first version of this refactor was therefore a NET LOSS on real work despite
# taking recompiles to zero.
#
# The payload does not change between calls, so the flattening is hoisted to
# BUILD time: the leaves are stored as a flat tuple and the jitted function is
# cached on the TREEDEF. Per call there is no tree walk, just a jitted call on
# a tuple of arrays. Two payloads of the same shapes still share one compile
# because the cache key is the treedef -- which is what keeps arms C and D at
# zero recompiles.


#: BOUNDED deliberately. The entries hold COMPILED functions, so an unbounded
#: cache is a slow leak. At a fixed discretisation this holds one or two
#: entries; a sweep over resolutions grows it by one treedef per shape, and 32
#: covers any realistic sweep. Eviction is harmless -- it costs one recompile,
#: which is the 2.8 ms this whole mechanism reduced it to.
@functools.lru_cache(maxsize=32)
def _jitted_for(treedef, impl):
    """One jitted apply per (treedef, impl). Unflattening is inside the trace."""

    def run(leaves, x):
        return impl(jax.tree_util.tree_unflatten(treedef, leaves), x)

    return jax.jit(run)


def _flatten_payload(payload):
    """(leaves, jitted) for a payload, computed once at build time."""
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    impl = (_apply_lump_payload if isinstance(payload, _LumpPayload)
            else _apply_mass_payload)
    return tuple(leaves), _jitted_for(treedef, impl)


class MetricLumpingLaplacian:
    """Bulk FD atoms + a dense core inverse, applied independently.

    Block Jacobi, deliberately: the bulk and core blocks are not coupled, so
    this is NOT the Schur envelope the k=0 thin-core preconditioner uses.

    The bulk atom lives on the bulk DOFs only. Their radial window is derived
    from the extraction rather than hard-coded, and checked to be a full tensor
    product ``{r0..r1} x all theta x all zeta`` -- if it is not, the separable
    atom does not apply to this component and we raise instead of silently
    building the wrong operator.
    """

    def __init__(self, seq, operators, k, dirichlet, *, core_tol=CORE_TOL,
                 ktilde_mode="honest", lumped="diag", extra_rings=0,
                 outer_rings=0, bc_entry="ibpd", bc_scale=None):
        self.k, self.dirichlet = k, dirichlet
        self.ktilde_mode, self.lumped = ktilde_mode, lumped
        self.bc_scale = bc_scale
        self.shapes = [tuple(int(s) for s in sh)
                       for sh in getattr(seq, f"basis_{k}").shape]

        core, bulk, e, _, _, _, tensor_blocks = _tensor_blocks(
            seq, k, dirichlet, extra_rings=extra_rings,
            outer_rings=outer_rings)
        self.core, self.bulk = core, bulk
        self.n_ext = int(e.forward_shape[0])

        self.blocks = []
        for c, blk in enumerate(tensor_blocks):
            if blk is None:
                self.blocks.append(None)
                continue
            rows_t, vals_t, (r0, nr), shape, offset = blk
            atom = build_bulk_atom(
                seq, k, c, window=(r0, nr), ktilde_mode=ktilde_mode,
                lumped=lumped, dirichlet=dirichlet, bc_scale=bc_scale,
                bc_entry=bc_entry)
            # The natural-BC trace exists only on the components the
            # integration by parts actually touches:
            #   k=1  <u, grad tau>   -> +int (u.n) tau        -> NORMAL, c = r
            #   k=2  <w, curl tau>   -> +int (w x n).tau      -> TANGENTIAL, c = t,z
            #   k=3  <om, div tau>   -> +int om (tau.n)       -> the one component
            #   k=0  W_0 = 0                                  -> none
            # Equivalently: wherever the component's RADIAL axis is a derivative
            # axis, which is where delta acts. Building it on w_r at k=2 adds a
            # term the operator does not have.
            dscale = jnp.ones(shape, dtype=DTYPE)
            if lumped == "diag":
                # D_i is a ratio of two positive integrals; no floor.
                d_full = component_diagonal(seq, k, c, self.shapes[c])
                dscale = 1.0 / jnp.sqrt(d_full[r0:r0 + nr, :, :])
            self.blocks.append({
                "rows": rows_t, "vals": vals_t, "shape": shape,
                "offset": offset, "atom": atom, "dscale": dscale})

        # Probe the whole core (polar ring + any extra/outer rings) and invert
        # it exactly. A separable 2-D ring atom was tried instead and dropped:
        # it MATCHES the dense probe on the inner rings at a tenth of the build
        # cost but LOSES badly on the outer ones (toroid k=3 free 65 vs 24,
        # W7-X k=2 free 616 vs 198), because the outer ring earns its keep
        # through radial coupling a separable ring cannot carry -- the
        # Steklov/DtN operator is nonlocal.
        self.probe_rows = core
        self.core_inv = _dense_symmetric_inverse(
            probe_core_block(seq, operators, k, dirichlet, core), core_tol)
        self._flat = _flatten_payload(self._build_payload())

    def _build_payload(self):
        """Pack the factors into the :class:`_LumpPayload` pytree.

        Built EAGERLY, at construction. It used to be memoised on the first
        ``apply``; construction is hoisted above every trace but a first apply
        inside a ``lax`` body stashed tracers on this long-lived object, and
        the failure surfaced as an ``UnexpectedTracerError`` in whatever ran
        next (docs/research/OPEN.md 1.1).

        The result is a PYTREE handed to a module-level jitted apply as an
        argument, rather than closed over by a per-instance `jax.jit`. Two
        payloads with the same shapes share a treedef, so rebuilding one
        reuses the compiled apply instead of paying ~287 ms to compile an
        identical program again.
        """
        blocks = []
        for blk in self.blocks:
            if blk is None:
                continue
            (v_r, v_t, v_z), (l_r, l_t, l_z), alpha = blk["atom"]
            blocks.append(_LumpBlock(
                rows=jnp.asarray(blk["rows"]),
                vals=jnp.asarray(blk["vals"], dtype=DTYPE),
                v_r=v_r, v_t=v_t, v_z=v_z,
                lam_r=l_r, lam_t=l_t, lam_z=l_z,
                alpha=jnp.asarray(alpha, dtype=DTYPE),
                dscale=blk["dscale"],
                shape=blk["shape"],
                offset=blk["offset"],
            ))
        perm, identity = _output_permutation(
            [b["rows"] for b in self.blocks if b is not None],
            self.probe_rows, self.n_ext)
        return _LumpPayload(
            blocks=tuple(blocks),
            core=jnp.asarray(self.probe_rows),
            core_inv=self.core_inv,
            perm=perm,
            has_core=bool(self.probe_rows.size > 0),
            identity_perm=identity,
        )

    def apply(self, x):
        """Apply the preconditioner to an extracted-space vector."""
        leaves, jitted = self._flat
        return jitted(leaves, jnp.asarray(x))


class _MassBlock(eqx.Module):
    """One component of the separable mass inverse. Tensor-ordered rows."""

    rows: jnp.ndarray            # leaf
    vals: jnp.ndarray            # leaf
    inv_r: jnp.ndarray           # leaf: the three 1-D inverses
    inv_t: jnp.ndarray
    inv_z: jnp.ndarray
    lam: jnp.ndarray             # leaf: the diagonal sandwich
    shape: tuple = eqx.field(static=True)     # STATIC: reshape target
    offset: int = eqx.field(static=True)      # STATIC: see _LumpBlock


class _MassPayload(eqx.Module):
    blocks: tuple
    core: jnp.ndarray
    core_inv: jnp.ndarray
    perm: jnp.ndarray
    has_core: bool = eqx.field(static=True)   # STATIC: guards a branch
    identity_perm: bool = eqx.field(static=True)


def _apply_mass_payload(payload: _MassPayload, x):
    parts = []
    for b in payload.blocks:
        buf = _block_input(b, x) / b.lam
        # A mass is a single Kronecker PRODUCT, not a sum, so the bulk inverse
        # is three 1-D solves and no fast diagonalisation is involved.
        for a, inv in enumerate((b.inv_r, b.inv_t, b.inv_z)):
            buf = jnp.moveaxis(jnp.tensordot(inv, buf, axes=([1], [a])), 0, a)
        parts.append(_block_output(b, buf / b.lam))
    if payload.has_core:
        parts.append(payload.core_inv @ x[payload.core])
    return _place(payload, parts)


class MetricLumpingMass:
    """``M_k^-1`` as a separable bulk plus a densely-probed core.

    raw_kron is already half of this shape -- ``M ~ Lam (A_r x A_t x A_z) Lam``
    is "separable bracket + diagonal sandwich", the same structure the Laplacian
    atom uses.  What changes is the CORE: raw_kron reaches the polar rows
    through the ``E+`` pseudoinverse, whose "both sides must carry the full
    ``(EE^T)^-1``" requirement its own docstring calls the single easiest thing
    to get wrong (the pow0/pow1/pow2 ablation cost 2.3x at k=1).  Here the core
    rows are probed and inverted densely instead, so there is no pseudoinverse
    anywhere, and ``extra_rings`` can widen the exact region exactly as it does
    for the Laplacians.

    A mass is structurally easier than a Laplacian: it is a single Kronecker
    product rather than a sum, so the bulk inverse is three 1-D solves and no
    fast diagonalisation is involved.

    NOTE this is not only a preconditioner. ``apply_hodge_laplacian_approx``
    uses the mass preconditioner as the inner inverse of the weak term, so
    swapping it changes the OPERATOR ``L_k`` at k>=1, not just the solve.
    """

    def __init__(self, seq, operators, k, dirichlet, *, extra_rings=0,
                 core_tol=CORE_TOL):
        from mrx.operators import apply_mass_matrix  # noqa: PLC0415
        from mrx.preconditioners import _kron_mass_model_1d  # noqa: PLC0415

        self.k, self.dirichlet = k, dirichlet
        shapes, mass_1d, lam = _kron_mass_model_1d(seq, k)
        self.shapes = [tuple(int(v) for v in sh) for sh in shapes]

        core, bulk, e, _, _, _, tensor_blocks = _tensor_blocks(
            seq, k, dirichlet, extra_rings=extra_rings)
        self.core, self.bulk = core, bulk
        self.n_ext = int(e.forward_shape[0])

        self.blocks = []
        for c, blk in enumerate(tensor_blocks):
            if blk is None:
                self.blocks.append(None)
                continue
            rows_t, vals_t, (r0, nr), shape, offset = blk
            inv = [jnp.linalg.inv(m[r0:r0 + nr, r0:r0 + nr] if a == 0 else m)
                   for a, m in enumerate(mass_1d[c])]
            self.blocks.append({
                "rows": rows_t, "vals": vals_t, "shape": shape, "inv": inv,
                "offset": offset, "lam": lam[c][r0:r0 + nr, :, :]})

        size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
        self.core_inv = _dense_symmetric_inverse(_probe_rows(
            lambda x: apply_mass_matrix(seq, operators, x, k,
                                        dirichlet=dirichlet),
            size, core), core_tol)
        self._flat = _flatten_payload(self._build_payload())

    def _build_payload(self):
        """Pack the factors into the :class:`_MassPayload` pytree.

        Eager, at construction, for the reason given at
        :meth:`MetricLumpingLaplacian._build_payload`. The apply is jitted
        and device-only because the mass preconditioner runs INSIDE
        ``solve_singular_cg``'s ``jax.lax.while_loop``.
        """
        blocks = []
        for blk in self.blocks:
            if blk is None:
                continue
            inv_r, inv_t, inv_z = blk["inv"]
            blocks.append(_MassBlock(
                rows=jnp.asarray(blk["rows"]),
                vals=jnp.asarray(blk["vals"], dtype=DTYPE),
                inv_r=inv_r, inv_t=inv_t, inv_z=inv_z,
                lam=blk["lam"],
                shape=blk["shape"],
                offset=blk["offset"],
            ))
        perm, identity = _output_permutation(
            [b["rows"] for b in self.blocks if b is not None],
            self.core, self.n_ext)
        return _MassPayload(
            blocks=tuple(blocks),
            core=jnp.asarray(self.core),
            core_inv=self.core_inv,
            perm=perm,
            has_core=bool(self.core.size > 0),
            identity_perm=identity,
        )

    def apply(self, x):
        """Apply the preconditioner to an extracted-space vector."""
        leaves, jitted = self._flat
        return jitted(leaves, jnp.asarray(x))
