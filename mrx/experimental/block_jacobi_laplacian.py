"""Block-Jacobi Hodge-Laplacian preconditioner: separable bulk + dense core.

NOT production. The production k>=1 Laplacian preconditioner is the shifted
Jacobi with the closed-form diagonal (``mrx.preconditioners``).

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
the radial stiffness (``bc_entry="exact"``): under a free condition the weak
block's integration by parts leaves ``int_{r=1} w u_r^2``, which for a tensor
basis is ``alpha (e e^T) (x) M_t (x) M_z`` -- the same shape as the first
Kronecker term, so it merges into ``K_r`` for free.  It is exactly zero under
Dirichlet and at k=0 (no weak block).  Its limit is that folding it into the
sum forces the face weight ``w(1,theta,zeta)`` down to a SCALAR: worth 1.5x ->
3.3x on a toroid, only 1.26x -> 1.56x on W7-X, where the face weight varies too
much for a scalar. Carrying that variation needs an exact outer ring.
"""

from __future__ import annotations

import os

import numpy as np

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
from mrx.preconditioners import _simultaneous_diagonalize_pair


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

    def clip(v):
        return jnp.maximum(v, 1e-8 * jnp.abs(jnp.median(v)))

    return clip(pr), clip(pt), clip(pz)


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
    a_primal = np.asarray(_assemble_weighted_1d_mass(
        primal[axis], quad_w[axis] * profile_primal))
    g = np.asarray(_dense_incidence_1d(int(a_primal.shape[0]), types[axis]))
    # Restrict only the V_k (row) side: the round trip still passes through the
    # FULL primal space, which is what keeps A invertible.
    if window is not None:
        g = g[window[0]:window[0] + window[1], :]
    if g.shape[0] != mass_deriv.shape[0]:
        raise ValueError(
            f"axis {axis}: incidence gives {g.shape[0]} rows after the window "
            f"but the derivative mass is {mass_deriv.shape[0]}")
    inner = np.linalg.solve(a_primal, g.T)                  # A^-1 G^T
    k = np.asarray(mass_deriv) @ (g @ inner) @ np.asarray(mass_deriv)
    return jnp.asarray(0.5 * (k + k.T))


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
    dist = np.maximum(dist, 1e-14)

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
    return jnp.asarray(0.5 * (k + k.T))


def _weak_inverse_amplification(seq, k, c):
    """``(M_r^{(k-1)})^{-1}[last, last]`` -- the 1/h the surface term picks up.

    The natural-BC term is not the bare surface integral. It enters ``W_k`` as
    ``E^T M_{k-1}^{-1} E`` with ``E`` the pairing of the V_k boundary trace
    against V_{k-1}::

        radial:  dLam_i(1) Lam_j(1)      angular: M_t (x) M_z

    and the angular masses cancel against the same factors in ``M_{k-1}^{-1}``,
    leaving one scalar: the last diagonal entry of the INVERSE radial mass of
    V_{k-1}.  That entry scales like ``1/h_last``, which is what the measured
    optimum does (toroid k=1 free: best scale 3 / 4.5 / 6.5 at n_r = 6 / 8 / 12,
    i.e. ~0.55/h) -- and it differs by degree through the V_{k-1} mass weight,
    which is why k=1 wanted ~4x and k=2 ~100x.

    WHICH component of V_{k-1} it pairs with is not always ``c``. At k=2 the
    trace is ``int (w x n).tau`` and the cross product SWAPS the tangential
    components: ``w_theta`` pairs with ``tau_zeta``. The bases confirm it --
    V_2 at c=theta has angular bases (theta primal, zeta derivative), which are
    V_1's at c=ZETA, not V_1's at c=theta -- and the angular cancellation above
    only holds for that partner. On the toroid the two weights differ by
    ``(R/a)^2``, so getting this wrong is not a small perturbation.

        k=1  c=r      -> V_0, the only component
        k=2  c=theta  -> V_1 c=zeta ,  c=zeta -> V_1 c=theta   (3 - c)
        k=3  c=r      -> V_2 c=r  (the normal component, int om (tau.n))

    The V_{k-1} radial basis is PRIMAL for every one of those: V_0 always, V_1
    component c is a derivative only on axis c (and the partner is never r when
    k=2 has a trace), V_2 component r is a derivative on theta and zeta only.
    """
    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    partner = 3 - c if k == 2 else c
    weight = {0: jac, 1: ginv[partner] * jac, 2: met[partner] / jac}[k - 1]
    prof = bundled_axis_profiles(seq, weight)[0]
    primal, _, quad_w = _axis_bases(seq)
    m_r = np.asarray(_assemble_weighted_1d_mass(primal[0], quad_w[0] * prof))
    # Lam(1) = e_last for a clamped spline, so this is (M_r^-1)[last, last];
    # a solve rather than a full inverse.
    rhs = np.zeros(m_r.shape[0])
    rhs[-1] = 1.0
    return float(np.linalg.solve(m_r, rhs)[-1])


def _mesh_amplification(seq):
    """``(M_r^{logical})^{-1}[last, last]`` -- the ``1/h``, with NO metric.

    The counterpart to :func:`_weak_inverse_amplification`, and the reason that
    one is wrong.  Reducing ``E^T M_{k-1}^{-1} E`` to a scalar means taking the
    weight of each of the three matrices at ``r = 1``, averaging over
    ``theta, zeta``, and multiplying::

        E            ->  m_k          := mass weight of V_k component c
        M_{k-1}^{-1} ->  1 / m_{k-1}  := INVERSE mass weight of the partner
        E^T          ->  m_k

        w_face = m_k^2 / m_{k-1}

    and with ``J^2 = prod g_aa`` that product is ``m_k g^rr`` for EVERY degree:

        k=1  (g^rr J)^2 / J          = (g^rr)^2 J        = m_1 g^rr
        k=2  (g_tt/J)^2 / (g^zz J)   = g_tt/(g_rr J)     = m_2 g^rr
        k=3  (1/J)^2 / (g_rr/J)      = 1/(g_rr J)        = m_3 g^rr

    So the face weight is ``mass_weight * g^rr`` -- the ORIGINAL ``direct``
    weight -- and what is left of ``M_{k-1}^{-1}`` is a purely LOGICAL
    quantity: the last diagonal entry of the inverse of the unweighted radial
    mass, ``~ c(p)/h_last``.  No metric, no ``k``, no component, no geometry:
    one number per (radial mesh, degree).  The partner's radial basis is primal
    for every degree (see :func:`_weak_inverse_amplification`), so it really is
    the same matrix in all eight cases.

    ``bc_entry="exact"`` instead put ``m_{k-1}`` INSIDE this mass and then
    substituted ``g^rr -> sqrt(g^rr)`` in the face weight to compensate; the
    two errors do not cancel and the net result is short by one surface
    element, ``J sqrt(g^rr)|_{r=1}`` (~13x on the epsilon=1/3 toroid).
    """
    primal, _, quad_w = _axis_bases(seq)
    m_r = np.asarray(_assemble_weighted_1d_mass(primal[0], quad_w[0]))
    rhs = np.zeros(m_r.shape[0])
    rhs[-1] = 1.0
    return float(np.linalg.solve(m_r, rhs)[-1])


def _face_metric_scalar(seq, k, c, lumped, separate=False):
    r"""ALL the metric in one number: pullback x measure at r=1, over theta,zeta.

    Everything except the ``1/h`` (:func:`_mesh_amplification`) is geometry,
    and it collapses to a single scalar. With ``m_k`` the component's mass
    weight, ``m_{k-1}`` the partner's and ``w_comp = m_k/J`` the factor the
    diag lumping carries outside as the ``D`` sandwich::

        no lumping   m_k^2 / m_{k-1}              = m_k g^rr
        diag lumping m_k^2 / (m_{k-1} w_comp)     = m_k J / m_{k-1} = J g^rr

    and ``J g^rr`` is the SAME for k=1,2,3 (use ``J^2 = prod g_aa``): the whole
    per-degree spread was the double-counted ``w_comp``. It factors into the
    two geometric ingredients, and nothing else::

        J g^rr = (J sqrt(g^rr))  x  (sqrt(g^rr))
                  surface element    pullback of the normal component

    ``separate`` averages those factors INDEPENDENTLY over theta,zeta and then
    multiplies, instead of averaging their product. They differ by the
    covariance, so they agree exactly wherever ``g^rr`` is constant on the face
    (the toroid -- a control), and ``<S><P> <= <SP>`` by Chebyshev whenever the
    two are positively correlated, which is the case on a shaped boundary.

    NOTE this is the opposite convention to :func:`bundled_axis_profiles`,
    deliberately: bundling `g * J` as a unit is there to keep `g^tt J ~ 1/r`
    integrable toward the AXIS. On the r=1 face there is no such singularity,
    so the argument does not carry over.
    """
    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    m_k = {0: jac, 1: ginv[c] * jac, 2: met[c] / jac, 3: 1.0 / jac}[k]
    surf = jac * jnp.sqrt(ginv[0])          # J sqrt(g^rr), the surface element
    pull = jnp.sqrt(ginv[0])                # sqrt(g^rr), the normal pullback
    comp = None if lumped == "diag" else m_k / jac

    wy, wz = seq.quad.w_y, seq.quad.w_z
    norm = jnp.sum(wy) * jnp.sum(wz)

    def fm(field):
        return float(jnp.einsum('rs,r,s->', jnp.asarray(field)[-1], wy, wz)
                     / norm)

    if separate:
        val = fm(surf) * fm(pull)
        return val if comp is None else val * fm(comp)
    return fm(surf * pull if comp is None else surf * pull * comp)


def _edge_vector(seq, axis, window):
    """``e = dLam_axis(1)``, windowed -- the shape every boundary update uses."""
    dlam = seq.basis_0.dΛ[axis]
    end = 1.0 - 1e-8 if dlam.type != "periodic" else 0.0
    e = np.asarray(jax.vmap(lambda i: jnp.sum(dlam(end, i)))(dlam.ns))
    if window is not None and axis == 0:
        e = e[window[0]:window[0] + window[1]]
    return e


#: The production natural-BC penalty scale. `alpha` as derived is the exact
#: surface integral and is the best NORM approximation to `L`'s boundary block,
#: but `P` minimises kappa(P^-1 L), which wants a much smaller number. 0.10 is
#: the minimax over 82 measured cells (4 geometries x k=0..3 x n=8..32 x
#: p=2..5): worst case 1.11x each cell's own optimum on the shaped geometries.
#: EMPIRICAL, not derived -- see docs/research/natural_bc_coefficient_handoff.md
#: §16 (what to ship) and §17.5 (why a scale is needed at all). The optimum
#: drifts DOWN with n and p; at p >= 5 or very high resolution prefer 0.05.
PRODUCTION_BC_SCALE = 0.10


def _resolve_bc_scale(bc_scale=None):
    """Resolve the natural-BC penalty scale.

    ``MRX_BJ_BC_SCALE`` OVERRIDES the argument when set. That ordering is
    deliberate: the sweep harnesses (``verify_block_jacobi.py``,
    ``block_jacobi_spectrum.py``) always set the variable -- to "1.0" when the
    arm names no scale -- so every recorded arm keeps meaning exactly what it
    meant before this default existed.
    """
    env = os.environ.get("MRX_BJ_BC_SCALE")
    if env is not None:
        return float(env)
    return PRODUCTION_BC_SCALE if bc_scale is None else float(bc_scale)


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
        alpha = float(scalar)
    else:
        wy, wz = seq.quad.w_y, seq.quad.w_z
        # <w> over theta,zeta on the last radial quadrature slice.
        alpha = float(jnp.einsum('rs,r,s->',
                                 jnp.asarray(weight_field)[-1], wy, wz)
                      / (jnp.sum(wy) * jnp.sum(wz)))

    # Penalty STRENGTH. The natural condition here is u.n = 0 -- an essential
    # condition on the normal trace, which the free-BC weak block enforces by a
    # mesh-dependent penalty rather than by removing a DOF. alpha as assembled
    # is the exact surface integral; this knob asks whether the atom wants the
    # exact penalty or the hard u_r = 0 limit it approximates.
    alpha *= _resolve_bc_scale(bc_scale)

    e = _edge_vector(seq, axis, window)
    return alpha * np.outer(e, e)
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
        # c = mean eig(A^-1 M), exact when the two weight profiles are
        # proportional.
        a_full = _assemble_weighted_1d_mass(primal[a], quad_w[a] * primal_prof[a])
        if a in deriv_axes:
            ratios.append(1.0)
        else:
            ratios.append(float(np.mean(np.real(np.linalg.eigvals(
                np.linalg.solve(np.asarray(cut(a_full, a)), np.asarray(m)))))))
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
                # The face weight keeps its FULL g^rr -- E^T M_{k-1}^-1 E
                # carries two powers of the surface weight and one inverse
                # mass, and that combination collapses to m_k g^rr at every
                # degree -- while the amplification stays metric-FREE (see
                # _mesh_amplification). Under `lumped="diag"` the component
                # factor w_comp is carried outside as the D sandwich, so
                # _face_metric_scalar drops it here.
                #
                # Ten other spellings of this term were measured and lost; see
                # docs/research/natural_bc_coefficient_handoff.md §9, §12.3 and
                # §14.3. Do not re-add them: the exact 2-D face shape, the
                # cross-term corrections (one of which is INDEFINITE), and the
                # "exact" sqrt(g^rr) form are all refuted, the last being worse
                # than no boundary term at all at k=1/2 free.
                scalar = _face_metric_scalar(seq, k, c, lumped)
                w_face = mass_weight * ginv[a]
                corr = _boundary_entry_direct(
                    seq, a, w_face, window, dirichlet, scalar=scalar,
                    bc_scale=bc_scale)
                if corr is not None:
                    kt = kt + jnp.asarray(corr * _mesh_amplification(seq))
            stiffs.append(kt)
        elif a in deriv_axes:
            stiffs.append(_ktilde_1d(seq, a, m, primal_prof[a],
                                     window=window if a == 0 else None))
        else:
            k_full = _assemble_weighted_1d_stiffness(
                primal[a], deriv[a], quad_w[a] * stiff_prof[a],
                jnp.asarray(_dense_incidence_1d(int(m_full.shape[0])
                                                if a != 0 or window is None
                                                else int(m_full.shape[0]),
                                                seq.basis_0.types[a])))
            kc = cut(k_full, a)
            stiffs.append(kc)

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
    tabs = [np.asarray(deriv[a] if a in deriv_axes else primal[a]) ** 2
            for a in range(3)]
    wq = rs(seq.quad.w)

    def contract(field):
        f = wq * field
        t1 = jnp.einsum('ax,xyz->ayz', tabs[0], f)
        t2 = jnp.einsum('by,ayz->abz', tabs[1], t1)
        return jnp.einsum('cz,abz->abc', tabs[2], t2)

    num = contract(w_comp * jac)
    den = contract(jac)
    return np.asarray(num / den).reshape(shape)


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
    :meth:`BlockJacobiLaplacian.__init__`.
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


def coarse_ring_basis(seq, k, dirichlet, rings, m_max, n_max, comps=None,
                      exclude=None):
    """Orthonormal columns spanning the outer rings x TRUNCATED Fourier modes.

    The dense core block is already an in-preconditioner coarse correction --
    `core_inv = (R L R^T)^-1` with `R` a SELECTION of the ring's rows, costing
    `n_t n_z` probe applies and an `(n_t n_z)^2` block. This generalises `R` to
    a RESTRICTION onto `|m| <= m_max, |n| <= n_max`, so the cost is one apply
    per coarse VECTOR: `(2 m_max+1)(2 n_max+1)` per component-ring, and it stops
    growing as the mesh refines.

    Justified by measurement: the outliers of `P L` sit on the outer rings
    (energy fraction 0.79-0.91) with LOW mode content (`|m|` 1.2-2.3 and,
    across a 6,12,6 -> 8,16,8 refinement, 1.39 -> 1.38 -- it does not drift).
    The face weight is banded the same way (99% of its energy inside
    `|m|<=3, |n|<=2` on W7-X).

    ``exclude`` drops rows already handled by REPLACEMENT (the dense core), so
    the correction stays additive without double-counting them.
    """
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows, cols = np.asarray(e.rows), np.asarray(e.cols)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)
    shapes = [tuple(int(v) for v in sh)
              for sh in getattr(seq, f"basis_{k}").shape]
    starts = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes])
    single = counts[rows] == 1
    r_s, c_s = rows[single], cols[single]
    comp = np.searchsorted(starts[1:], c_s, side="right")
    loc = c_s - starts[comp]

    cols_out = []
    for c, shape in enumerate(shapes):
        if comps is not None and c not in comps:
            continue
        nr, nt, nz = shape
        sel = comp == c
        if not sel.any():
            continue
        lidx, rid = loc[sel], r_s[sel]
        i_r = lidx // (nt * nz)
        i_t = (lidx // nz) % nt
        i_z = lidx % nz
        js, ks = np.arange(nt), np.arange(nz)
        for ring in range(max(0, nr - rings), nr):
            take = i_r == ring
            if not take.any():
                continue
            rr, tt, zz = rid[take], i_t[take], i_z[take]
            for m in range(0, m_max + 1):
                for n in range(-n_max, n_max + 1):
                    ph = 2.0 * np.pi * (m * js[:, None] / nt
                                        + n * ks[None, :] / nz)
                    for f in (np.cos, np.sin):
                        v = np.zeros(n_ext)
                        v[rr] = f(ph)[tt, zz]
                        cols_out.append(v)
    if not cols_out:
        return np.zeros((n_ext, 0))
    v_mat = np.stack(cols_out, axis=1)
    if exclude is not None and len(exclude):
        v_mat[np.asarray(exclude), :] = 0.0
    # cos/sin over the full (m, n) box is redundant by construction (m=0 pairs
    # n with -n); a pivoted QR drops the dependents and leaves an orthonormal
    # basis, which is what keeps the Galerkin block well conditioned.
    q_mat, r_mat = np.linalg.qr(v_mat)
    keep = np.abs(np.diag(r_mat)) > 1e-10 * np.abs(np.diag(r_mat)).max()
    return q_mat[:, keep]


def coarse_correction(seq, operators, k, dirichlet, v_mat, tol=1e-12,
                      trunc_rows=None):
    """``V (V^T L V)^-1 V^T`` and ``L V`` -- one apply per coarse column.

    MEASURED (rot-ellipse k=1 free, 6,12,6): used ADDITIVELY this raises
    `lambda_min` (2.52e-2 -> 3.55e-2) and removes low outliers (33 -> 29) but
    leaves the HIGH outliers untouched (9 -> 10) and `lambda_max` slightly
    worse (21.05 -> 21.90). That is structural, not a bug: a high outlier means
    `P` is too LARGE there (the atom is too soft), and `P + V A_0^-1 V^T` only
    makes it larger. Additive two-level Schwarz assumes the local part
    UNDER-resolves the coarse modes; this atom OVER-resolves them.

    Hence the hybrid (balancing) form, which REMOVES the atom's action on the
    coarse space instead of adding to it, and is still entirely inside `P`::

        P = Q + (I - Q L) M (I - L Q) ,    Q = V A_0^-1 V^T
    """
    from mrx.operators import apply_hodge_laplacian_approx  # noqa: PLC0415

    size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
    if v_mat.shape[1] == 0:
        return None
    apply_hodge_laplacian_approx(seq, operators, jnp.zeros(size), k,
                                 dirichlet=dirichlet)
    lv = np.stack([np.asarray(apply_hodge_laplacian_approx(
        seq, operators, jnp.asarray(v_mat[:, j]), k, dirichlet=dirichlet))
        for j in range(v_mat.shape[1])], axis=1)
    a0 = v_mat.T @ lv
    a0 = 0.5 * (a0 + a0.T)
    w, u = np.linalg.eigh(a0)
    keep = np.abs(w) > tol * np.abs(w).max()
    # ``lv`` is kept: the hybrid form needs ``L Q`` every apply, and
    # ``L Q x = (L V) A_0^-1 (V^T x)`` is a dense matvec against it -- so the
    # correction costs NO extra operator apply.
    if trunc_rows is not None:
        # MEASURED: |LV|^2 is 99.5% on the outer ring itself and 0.5% one ring
        # in (`fm_cost.py`, rot-ellipse n=12 k=1), so both V and LV live on a
        # thin slab. A_0 = V^T (L V) samples LV only on V's support, so the
        # Galerkin block is UNCHANGED by this; only the (I - L Q) factors move,
        # by 0.5%, inside a preconditioner.
        return (v_mat[trunc_rows], (u[:, keep] / w[keep]) @ u[:, keep].T,
                lv[trunc_rows], trunc_rows)
    return v_mat, (u[:, keep] / w[keep]) @ u[:, keep].T, lv, None


def probe_core_block(seq, operators, k, dirichlet, rows):
    """Dense ``L_k`` restricted to the core rows, by one apply per row."""
    from mrx.operators import apply_hodge_laplacian_approx  # noqa: PLC0415

    size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
    if rows.size == 0:
        return np.zeros((0, 0))
    # Warm the apply outside any trace: the matrix-free mass plan is host-built.
    apply_hodge_laplacian_approx(seq, operators, jnp.zeros(size), k,
                                 dirichlet=dirichlet)
    cols = []
    for i in rows:
        e_i = jnp.zeros(size).at[int(i)].set(1.0)
        col = apply_hodge_laplacian_approx(seq, operators, e_i, k,
                                           dirichlet=dirichlet)
        cols.append(np.asarray(col)[rows])
    block = np.stack(cols, axis=1)
    return 0.5 * (block + block.T)
class BlockJacobiLaplacian:
    """Bulk FD atoms + a dense core inverse, applied independently.

    Block Jacobi, deliberately: the bulk and core blocks are not coupled, so
    this is NOT the Schur envelope the k=0 thin-core preconditioner uses.

    The bulk atom lives on the bulk DOFs only. Their radial window is derived
    from the extraction rather than hard-coded, and checked to be a full tensor
    product ``{r0..r1} x all theta x all zeta`` -- if it is not, the separable
    atom does not apply to this component and we raise instead of silently
    building the wrong operator.
    """

    def __init__(self, seq, operators, k, dirichlet, *, core_tol=1e-12,
                 ktilde_mode="honest", lumped="diag", extra_rings=0,
                 outer_rings=0, bc_entry="ibpd",
                 bc_scale=None,
                 coarse_rings=0, coarse_modes=(3, 3),
                 coarse_set="all", coarse_mode="hybrid",
                 coarse_trunc=0):
        self.k, self.dirichlet = k, dirichlet
        self.ktilde_mode, self.lumped = ktilde_mode, lumped
        self.bc_scale = bc_scale
        self.shapes = [tuple(int(s) for s in sh)
                       for sh in getattr(seq, f"basis_{k}").shape]
        starts = np.cumsum([0] + [int(np.prod(s)) for s in self.shapes])

        core, bulk, e, polar, inner_rings, outer_ring_rows = core_rows(
            seq, k, dirichlet, extra_rings=extra_rings,
            outer_rings=outer_rings)
        self.core, self.bulk = core, bulk
        self.n_ext = int(e.forward_shape[0])
        rows, cols, vals = (np.asarray(e.rows), np.asarray(e.cols),
                            np.asarray(e.vals))
        keep = np.isin(rows, bulk)
        rows_b, cols_b, vals_b = rows[keep], cols[keep], vals[keep]

        comp = np.searchsorted(starts[1:], cols_b, side="right")
        loc = cols_b - starts[comp]

        self.blocks = []
        for c, shape in enumerate(self.shapes):
            sel = comp == c
            if not sel.any():
                self.blocks.append(None)
                continue
            lidx = loc[sel]
            i_r = lidx // (shape[1] * shape[2])
            i_t = (lidx // shape[2]) % shape[1]
            i_z = lidx % shape[2]
            r0, r1 = int(i_r.min()), int(i_r.max()) + 1
            nr = r1 - r0
            expected = nr * shape[1] * shape[2]
            if lidx.size != expected or len(set(map(int, i_r))) != nr:
                raise ValueError(
                    f"k={k} component {c}: the {lidx.size} bulk DOFs are not the "
                    f"tensor block [{r0},{r1}) x {shape[1]} x {shape[2]} "
                    f"({expected} entries); the separable atom does not apply")
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
            dscale = None
            if lumped == "diag":
                d_full = component_diagonal(seq, k, c, shape)
                dscale = 1.0 / np.sqrt(np.maximum(
                    d_full[r0:r0 + nr, :, :], 1e-300))
            self.blocks.append({
                "rows": rows_b[sel], "vals": vals_b[sel],
                "idx": (i_r - r0, i_t, i_z), "shape": (nr, shape[1], shape[2]),
                "atom": atom, "dscale": dscale})

        # Probe the whole core (polar ring + any extra/outer rings) and invert
        # it exactly. A separable 2-D ring atom was tried instead and dropped:
        # it MATCHES the dense probe on the inner rings at a tenth of the build
        # cost but LOSES badly on the outer ones (toroid k=3 free 65 vs 24,
        # W7-X k=2 free 616 vs 198), because the outer ring earns its keep
        # through radial coupling a separable ring cannot carry -- the
        # Steklov/DtN operator is nonlocal.
        probe_rows = core
        self.probe_rows = probe_rows

        block = probe_core_block(seq, operators, k, dirichlet, probe_rows)
        if block.size:
            w, v = np.linalg.eigh(block)
            keep_w = np.abs(w) > core_tol * np.abs(w).max()
            self.core_inv = (v[:, keep_w] / w[keep_w]) @ v[:, keep_w].T
        else:
            self.core_inv = np.zeros((0, 0))

        # ADDITIVE truncated-Fourier coarse correction on the outer rings.
        # Unlike the dense core -- which REPLACES its rows, valid only because
        # `R` is a selection and those rows were removed from the bulk -- the
        # truncated basis does not span its rows, so it must be added on top of
        # the atom. Two-level additive Schwarz; P stays SPD as a sum of SPD
        # pieces and CG is untouched.
        self.coarse = None
        if coarse_rings > 0 and not (dirichlet and coarse_set == "trace"):
            _tr = trace_components(k)
            _ot = tuple(c for c in range(len(self.shapes)) if c not in _tr)
            cset = {"all": None, "trace": _tr, "other": _ot}[coarse_set]
            v_mat = coarse_ring_basis(
                seq, k, dirichlet, coarse_rings, int(coarse_modes[0]),
                int(coarse_modes[1]), comps=cset, exclude=probe_rows)
            trunc_rows = None
            if coarse_trunc:
                slab = coarse_ring_basis(
                    seq, k, dirichlet, coarse_rings + int(coarse_trunc),
                    0, 0, comps=cset, exclude=None)
                trunc_rows = np.flatnonzero(
                    np.abs(slab).sum(axis=1) > 0) if slab.size else None
            self.coarse = coarse_correction(seq, operators, k, dirichlet,
                                            v_mat, trunc_rows=trunc_rows)
            self.coarse_mode = coarse_mode
            self.n_coarse = int(v_mat.shape[1])

    def _build_apply(self):
        """Compile the apply. Everything here is on-device and jitted.

        Build-time work (eigendecompositions, the dense core inverse, the face
        operator) is host-side numpy and belongs there -- it happens once. The
        APPLY runs once per CG iteration, thousands of times per solve, so it
        must not round-trip through the host: the scatter/gather use flat index
        arrays computed once, and _fd_apply_3d is inside the jit rather than
        called per component from Python.
        """
        blocks = []
        for blk in self.blocks:
            if blk is None:
                continue
            nr, nt, nz = blk["shape"]
            ir, it, iz = blk["idx"]
            flat = jnp.asarray((ir * nt + it) * nz + iz)
            (v_r, v_t, v_z), (l_r, l_t, l_z), alpha = blk["atom"]
            blocks.append({
                "rows": jnp.asarray(blk["rows"]),
                "vals": jnp.asarray(blk["vals"]),
                "flat": flat, "shape": (nr, nt, nz),
                "v": (v_r, v_t, v_z), "lam": (l_r, l_t, l_z),
                "alpha": tuple(float(a) for a in alpha),
                "dscale": (None if blk["dscale"] is None
                           else jnp.asarray(blk["dscale"])),
            })
        core = jnp.asarray(self.probe_rows)
        core_inv = jnp.asarray(self.core_inv)
        has_core = np.asarray(self.probe_rows).size > 0
        coarse = (None if getattr(self, "coarse", None) is None else
                  (jnp.asarray(self.coarse[0]), jnp.asarray(self.coarse[1]),
                   jnp.asarray(self.coarse[2])))
        coarse_rows = (None if getattr(self, "coarse", None) is None
                       or self.coarse[3] is None
                       else jnp.asarray(self.coarse[3]))
        coarse_hybrid = getattr(self, "coarse_mode", "hybrid") == "hybrid"

        def m_apply(x):
            out = jnp.zeros_like(x)
            for b in blocks:
                buf = jnp.zeros(int(np.prod(b["shape"]))).at[b["flat"]].set(
                    b["vals"] * x[b["rows"]]).reshape(b["shape"])
                if b["dscale"] is not None:
                    buf = buf * b["dscale"]
                sol = _fd_apply_3d(*b["v"], *b["lam"], b["alpha"], buf)
                if b["dscale"] is not None:
                    sol = sol * b["dscale"]
                out = out.at[b["rows"]].set(b["vals"] * sol.reshape(-1)[b["flat"]])
            if has_core:
                out = out.at[core].set(core_inv @ x[core])
            return out

        def impl(x):
            if coarse is None:
                return m_apply(x)
            v_mat, a0inv, lv = coarse
            if coarse_rows is not None:
                # V and LV are held only on the slab; gather once, scatter back.
                xs = x[coarse_rows]
                vtx = v_mat.T @ xs
                z = jnp.zeros_like(x).at[coarse_rows].set(v_mat @ (a0inv @ vtx))
                if not coarse_hybrid:
                    return m_apply(x) + z
                y = x.at[coarse_rows].add(-(lv @ (a0inv @ vtx)))
                w = m_apply(y)
                return z - jnp.zeros_like(x).at[coarse_rows].set(
                    v_mat @ (a0inv @ (lv.T @ w[coarse_rows]))) + w
            vtx = v_mat.T @ x
            z = v_mat @ (a0inv @ vtx)
            if not coarse_hybrid:
                # ADDITIVE -- kept as a diagnostic only. It cannot cure a HIGH
                # outlier (P already too large there); measured in
                # `coarse_correction`.
                return m_apply(x) + z
            # HYBRID / balancing: P = Q + (I - Q L) M (I - L Q). Removes the
            # atom's action on the coarse space instead of adding to it, stays
            # symmetric, and costs NO extra operator apply because L Q uses the
            # stored L V.
            y = x - lv @ (a0inv @ vtx)
            w = m_apply(y)
            return z + w - v_mat @ (a0inv @ (lv.T @ w))

        return jax.jit(impl)

    def apply(self, x):
        """Apply the preconditioner to an extracted-space vector."""
        if getattr(self, "_jit", None) is None:
            self._jit = self._build_apply()
        return self._jit(jnp.asarray(x))
def mixed_mass_1d(seq, axis):
    """``P[j, i] = int Lam0_j dLam_i dxi`` on one axis -- METRIC FREE.

    The V_0 / V_3 pairing carries no geometry at all: a 3-form's physical proxy
    is ``phi/J`` and the 0-form measure is ``J dxi``, so the Jacobian cancels
    and what is left is a knot-determined banded matrix.  The 3-D transfer is
    therefore a pure Kronecker product of three of these.
    """
    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)[axis]
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)[axis]
    w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)[axis]
    return np.asarray((jnp.asarray(primal) * w[None, :]) @ jnp.asarray(deriv).T)
def radial_profiles(seq, field):
    """Quad-weighted mean of a weight field over theta and zeta -- a RADIAL
    profile, keeping the full radial dependence.

    :func:`bundled_axis_profiles` averages each weight over the other two axes,
    which throws the radial dependence away on the theta and zeta terms -- and
    ``g^{tt}J ~ 1/r``, so that is exactly where it hurts. The k=0 campaign
    measured the difference as mesh-DEPENDENT vs mesh-independent (toroid
    22/32/40 -> 13/13/14 over three refinements).
    """
    wy, wz = seq.quad.w_y, seq.quad.w_z
    return jnp.einsum('qrs,r,s->q', field, wy, wz) / (jnp.sum(wy) * jnp.sum(wz))
class BlockJacobiMass:
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
                 core_tol=1e-12):
        from mrx.preconditioners import _kron_mass_model_1d  # noqa: PLC0415

        self.k, self.dirichlet = k, dirichlet
        shapes, mass_1d, lam = _kron_mass_model_1d(seq, k)
        self.shapes = [tuple(int(v) for v in sh) for sh in shapes]
        starts = np.cumsum([0] + [int(np.prod(s)) for s in self.shapes])

        core, bulk, e = core_rows(seq, k, dirichlet,
                                 extra_rings=extra_rings)[:3]
        self.core, self.bulk = core, bulk
        rows, cols, vals = (np.asarray(e.rows), np.asarray(e.cols),
                            np.asarray(e.vals))
        keep = np.isin(rows, bulk)
        rows_b, cols_b, vals_b = rows[keep], cols[keep], vals[keep]
        comp = np.searchsorted(starts[1:], cols_b, side="right")
        loc = cols_b - starts[comp]

        self.blocks = []
        for c, shape in enumerate(self.shapes):
            sel = comp == c
            if not sel.any():
                self.blocks.append(None)
                continue
            lidx = loc[sel]
            i_r = lidx // (shape[1] * shape[2])
            i_t = (lidx // shape[2]) % shape[1]
            i_z = lidx % shape[2]
            r0, nr = int(i_r.min()), int(i_r.max()) - int(i_r.min()) + 1
            if lidx.size != nr * shape[1] * shape[2]:
                raise ValueError(
                    f"mass k={k} component {c}: bulk DOFs are not a tensor "
                    f"block [{r0},{r0 + nr}) x {shape[1]} x {shape[2]}")
            inv = []
            for a in range(3):
                m = np.asarray(mass_1d[c][a])
                if a == 0:
                    m = m[r0:r0 + nr, r0:r0 + nr]
                inv.append(np.linalg.inv(m))
            self.blocks.append({
                "rows": rows_b[sel], "vals": vals_b[sel],
                "idx": (i_r - r0, i_t, i_z), "shape": (nr, shape[1], shape[2]),
                "inv": inv,
                "lam": np.asarray(lam[c])[r0:r0 + nr, :, :]})

        self.core_inv = np.zeros((0, 0))
        if core.size:
            block = self._probe_core(seq, operators, core)
            w, v = np.linalg.eigh(block)
            keep_w = np.abs(w) > core_tol * np.abs(w).max()
            self.core_inv = (v[:, keep_w] / w[keep_w]) @ v[:, keep_w].T

    def _probe_core(self, seq, operators, rows):
        from mrx.operators import apply_mass_matrix  # noqa: PLC0415

        size = int(getattr(seq, f"n{self.k}_dbc" if self.dirichlet
                           else f"n{self.k}"))
        apply_mass_matrix(seq, operators, jnp.zeros(size), self.k,
                          dirichlet=self.dirichlet)
        cols = [np.asarray(apply_mass_matrix(
            seq, operators, jnp.zeros(size).at[int(i)].set(1.0), self.k,
            dirichlet=self.dirichlet))[rows] for i in rows]
        b = np.stack(cols, axis=1)
        return 0.5 * (b + b.T)

    def apply(self, x):
        x = np.asarray(x)
        out = np.zeros_like(x)
        for blk in self.blocks:
            if blk is None:
                continue
            buf = np.zeros(blk["shape"])
            ir, it, iz = blk["idx"]
            buf[ir, it, iz] = blk["vals"] * x[blk["rows"]]
            buf = buf / blk["lam"]
            for a in range(3):
                buf = np.moveaxis(np.tensordot(blk["inv"][a], buf,
                                               axes=([1], [a])), 0, a)
            buf = buf / blk["lam"]
            out[blk["rows"]] = blk["vals"] * buf[ir, it, iz]
        if self.core.size:
            out[self.core] = self.core_inv @ x[self.core]
        return jnp.asarray(out)


# --------------------------------------------------------------------------- #
# Natural BC by capacitance (Woodbury) -- no operator probes                    #
# --------------------------------------------------------------------------- #

def face_operator(seq, k, c, window, corrected=True):
    """The natural-BC face operator ``B`` on the outer radial ring.

    The boundary term is a surface integral on ``r = 1``, so it is assembled by
    2-D QUADRATURE over the face -- no operator applies at all, which is the
    whole point next to probing the ring.  It keeps the full angular dependence
    of ``w(1, theta, zeta)``: nothing is lumped to a scalar.

    Returned in the physical ``(theta, zeta)`` face basis, shape
    ``(n_t n_z, n_t n_z)``, already carrying the radial basis value at the
    boundary so that ``E B E^T`` is the term as it acts on the full grid.
    """
    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    mass_weight = {0: jac, 1: ginv[c] * jac, 2: met[c] / jac, 3: 1.0 / jac}[k]
    # Same weight as the scalar route (§6.1), but NOT collapsed: this is the
    # only object that can carry the theta,zeta dependence the scalar drops.
    #   corrected = True   J g^rr, and x mu_0 -- the diag-lumped coefficient
    #                      with the full angular profile kept
    #   corrected = False  the historical m_k g^rr with no amplification, i.e.
    #                      w_comp double counted and the 1/h missing
    w_face = np.asarray((jac if corrected else mass_weight) * ginv[0])[-1]

    # The ANGULAR factors must be the component's own bases, not always the
    # primal ones: at k=2 one angular axis is a derivative axis (w_t has zeta
    # differentiated, w_z has theta) and at k=3 both are. A derivative basis is
    # O(1/h) larger, so each wrong axis costs (1/h)^2 -- measured as 41x, 174x
    # and 7100x against the k=1 reference, with 41 * 174 = 7134 confirming that
    # k=3 is just the two single-axis errors compounding.
    primal, deriv, _ = _axis_bases(seq)
    deriv_axes = {0: (), 1: (c,), 3: (0, 1, 2)}.get(
        k, tuple(a for a in range(3) if a != c))
    t_tab = np.asarray(deriv[1] if 1 in deriv_axes else primal[1])
    z_tab = np.asarray(deriv[2] if 2 in deriv_axes else primal[2])
    wy, wz = np.asarray(seq.quad.w_y), np.asarray(seq.quad.w_z)

    # sum-factorised: theta mass per zeta quad point, then contract zeta
    b_t = np.einsum('t,tz,jt,Jt->jJz', wy, w_face, t_tab, t_tab)
    b4 = np.einsum('jJz,z,kz,Kz->jkJK', b_t, wz, z_tab, z_tab)
    n_t, n_z = t_tab.shape[0], z_tab.shape[0]
    b = b4.reshape(n_t * n_z, n_t * n_z)

    dlam = seq.basis_0.dΛ[0]
    end = 1.0 - 1e-8 if dlam.type != "periodic" else 0.0
    e_r = np.asarray(jax.vmap(lambda i: jnp.sum(dlam(end, i)))(dlam.ns))
    if window is not None:
        e_r = e_r[window[0]:window[0] + window[1]]
    amp = _mesh_amplification(seq) if corrected else 1.0
    return 0.5 * (b + b.T) * float(e_r[-1]) ** 2 * amp