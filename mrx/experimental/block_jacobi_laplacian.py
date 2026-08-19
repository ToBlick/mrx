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

    The V_{k-1} radial basis is PRIMAL for every matching component: V_0 always,
    V_1 component c is a derivative only on axis c (and c != r whenever k = 2
    has a trace), V_2 component r is a derivative on theta and zeta only.
    """
    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    weight = {0: jac, 1: ginv[c] * jac, 2: met[c] / jac}[k - 1]
    prof = bundled_axis_profiles(seq, weight)[0]
    primal, _, quad_w = _axis_bases(seq)
    m_r = np.asarray(_assemble_weighted_1d_mass(primal[0], quad_w[0] * prof))
    return float(np.linalg.inv(m_r)[-1, -1])


def _boundary_entry_direct(seq, axis, weight_field, window, dirichlet):
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
    wy, wz = seq.quad.w_y, seq.quad.w_z
    # <w> over theta,zeta on the last radial quadrature slice.
    alpha = float(jnp.einsum('rs,r,s->', jnp.asarray(weight_field)[-1], wy, wz)
                  / (jnp.sum(wy) * jnp.sum(wz)))

    # Penalty STRENGTH. The natural condition here is u.n = 0 -- an essential
    # condition on the normal trace, which the free-BC weak block enforces by a
    # mesh-dependent penalty rather than by removing a DOF. alpha as assembled
    # is the exact surface integral; this knob asks whether the atom wants the
    # exact penalty or the hard u_r = 0 limit it approximates.
    alpha *= float(os.environ.get("MRX_BJ_BC_SCALE", "1.0"))

    dlam = seq.basis_0.dΛ[axis]
    end = 1.0 - 1e-8 if dlam.type != "periodic" else 0.0
    e = np.asarray(jax.vmap(lambda i: jnp.sum(dlam(end, i)))(dlam.ns))
    if window is not None and axis == 0:
        e = e[window[0]:window[0] + window[1]]
    return alpha * np.outer(e, e)


def _boundary_entry(seq, k, c, axis, ktilde, m_d, prof_primal, window,
                    dirichlet=False):
    """The natural-BC boundary trace, as a rank-one update to the 1-D stiffness.

    The weak block contains the ADJOINT derivative, so integrating by parts
    leaves a boundary term. Under Dirichlet the test function vanishes on the
    boundary and it dies; under the free (natural) condition it does not, and
    the honest stiffness of the derivative splines omits it entirely.

    The exact radial factor is ``F = M^d G A^-1 G^T M^d`` -- built from the
    LOWER space's mass and incidence, so it carries the boundary condition
    automatically. The correction is therefore ``F - Ktilde``, which
    integration by parts says is supported on the ``r = 1`` face and hence rank
    one. Returned as-is rather than truncated, so the caller can check.
    """
    from mrx.operators import _dense_incidence_1d  # noqa: PLC0415

    primal, _, quad_w = _axis_bases(seq)
    a_mat = np.asarray(_assemble_weighted_1d_mass(
        primal[axis], quad_w[axis] * prof_primal))
    g = np.asarray(_dense_incidence_1d(int(a_mat.shape[0]),
                                       seq.basis_0.types[axis]))
    if dirichlet:
        # The lower space loses its outer boundary DOF, which is precisely why
        # the trace term dies under an essential condition. Measured: the
        # correction shrinks 6-15x when this column goes.
        g = g[:, :-1]
        a_mat = a_mat[:-1, :-1]
    if window is not None and axis == 0:
        g = g[window[0]:window[0] + window[1], :]
    m = np.asarray(m_d)
    f = m @ (g @ np.linalg.solve(a_mat, g.T)) @ m
    return f - np.asarray(ktilde)


def component_factors(seq, k, c, window=None, ktilde_mode="roundtrip",
                      lumped=False, bc_entry="exact", dirichlet=False):
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
                stiffs.append(cut(_fd_stiffness_degree0(seq, a, prof), a))
            else:
                kt = cut(_assemble_weighted_1d_mass(
                    seq._bj_dd_tables[a], quad_w[a] * prof), a)
                # a is a derivative axis here; the trace only lives on the
                # RADIAL one (a == 0), the boundary face being r = 1.
                if bc_entry in ("direct", "face", "exact") and a == 0:
                    # "face": the REAL surface element. On r = const,
                    #   sqrt(g_tt g_zz - g_tz^2) = J / sqrt(g_rr) = J sqrt(g^rr),
                    # exact whenever the radial direction is orthogonal to the
                    # surface (g_rt = g_rz = 0) -- no assumption on g_tz, which
                    # is the large off-diagonal on W7-X. Carrying the physical
                    # component factor through each degree turns the whole
                    # correction into ONE substitution: g^rr -> sqrt(g^rr),
                    # at k=1 (normal trace), k=2 (tangential) and k=3 alike.
                    w_face = (mass_weight * jnp.sqrt(ginv[a])
                              if bc_entry in ("face", "exact")
                              else mass_weight * ginv[a])
                    corr = _boundary_entry_direct(
                        seq, a, w_face, window, dirichlet)
                    if bc_entry == "exact" and corr is not None:
                        corr = corr * _weak_inverse_amplification(seq, k, c)
                    if corr is not None:
                        kt = kt + jnp.asarray(corr)
                elif bc_entry and a == 0:
                    # a == 0 ONLY: the boundary face is r = 1. theta and zeta
                    # are periodic and have no boundary, so any entry added
                    # there is pure noise -- it was measurably harmful (k=3 free
                    # 97 vs 87 for no correction at all, and it perturbed the
                    # DIRICHLET cases where the term must vanish identically).
                    corr = _boundary_entry(seq, k, c, a, kt, m,
                                           primal_prof[a], window,
                                           dirichlet=dirichlet)
                    u, sv, _ = np.linalg.svd(corr)
                    # rank-one truncation from F - Ktilde; contaminated by the
                    # product-vs-sum shape mismatch, see _boundary_entry_direct.
                    kt = kt + sv[0] * np.outer(u[:, 0], u[:, 0])
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


def build_bulk_atom(seq, k, c, window=None, ktilde_mode="roundtrip",
                    lumped=False, bc_entry="exact", dirichlet=False):
    """Fast-diagonalisation factors for component ``c`` of ``L_k``.

    Returns ``(V_r, V_t, V_z, lam_r, lam_t, lam_z)`` ready for
    :func:`mrx.operators._fd_apply_3d` with ``alpha = (1, 1, 1)``.
    """
    masses, stiffs, alpha = component_factors(seq, k, c, window=window,
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
        if outer_rings > 0:
            outer_rows = r_s[i_r >= nr - outer_rings]
            core = np.union1d(core, outer_rows)

    polar = np.flatnonzero(counts > 1)
    inner = np.setdiff1d(inner_rows, polar)
    outer = np.setdiff1d(outer_rows, polar)
    bulk = np.setdiff1d(np.arange(n_ext), core)
    return core, bulk, e, polar, inner, outer


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


def build_ring_atom(seq, k, c, ring_index, radial_pair, alpha,
                    ktilde_mode="honest"):
    """2-D atom for ONE radial ring: `(theta, zeta)` fast diagonalisation.

    Restricting the three-term Kronecker sum to a single radial index leaves a
    2-D Kronecker sum plus a shift::

        K_r[i,i] M_t (x) M_z  +  M_r[i,i] ( K_t (x) M_z + M_t (x) K_z )

    so one pair of small eigendecompositions inverts it, with denominator
    ``alpha_r K_r[i,i] + M_r[i,i] (alpha_t lam_t + alpha_z lam_z)``.

    The point is NOT the storage (though ``O(n_t^2 + n_z^2)`` against
    ``(n_t n_z)^2`` and no probe applies are both welcome).  A ring atom built
    from the BULK atom's 1-D factors would reproduce the bulk atom's error on
    those rows and gain exactly nothing.  What it must carry instead is the
    information the bulk atom throws away: the angular weight profiles here are
    averaged over the OTHER angular axis only, with the radial direction
    collapsed by the ring's OWN basis function rather than by a global radial
    mean.  On the innermost rings ``g^tt J ~ 1/r`` varies fastest, and on the
    outermost ``det(DF) -> 0``; both are precisely where a global radial average
    is worst.
    """
    primal, deriv, quad_w = _axis_bases(seq)
    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]
    deriv_axes = {0: (), 1: (c,), 3: (0, 1, 2)}.get(
        k, tuple(a for a in range(3) if a != c))

    # Collapse the radial direction with THIS ring's own basis function, not
    # with a global mean: w_r(q) = phi_i(q)^2 * wq.
    tab_r = np.asarray(deriv[0] if 0 in deriv_axes else primal[0])
    w_r = jnp.asarray(tab_r[ring_index] ** 2) * quad_w[0]
    s_r = jnp.sum(w_r)

    def ring_profile(field, axis):
        wy, wz = quad_w[1], quad_w[2]
        if axis == 1:
            p = jnp.einsum('qrs,q,s->r', field, w_r, wz) / (s_r * jnp.sum(wz))
        else:
            p = jnp.einsum('qrs,q,r->s', field, w_r, wy) / (s_r * jnp.sum(wy))
        return jnp.maximum(p, 1e-8 * jnp.abs(jnp.median(p)))

    def angular(axis):
        # k=0 weights only: the component factor g^{cc} rides on the diagonal
        # sandwich, exactly as in the bulk atom's "diag" lumping.
        prof = ring_profile(ginv[axis] * jac, axis)
        basis = deriv[axis] if axis in deriv_axes else primal[axis]
        m = _assemble_weighted_1d_mass(basis, quad_w[axis])
        if axis in deriv_axes and ktilde_mode == "honest":
            if int(seq.basis_0.Λ[0].p) < 2:
                kk = _fd_stiffness_degree0(seq, axis, prof)
            else:
                if not hasattr(seq, "_bj_dd_tables"):
                    from mrx.local_assembly import (  # noqa: PLC0415
                        _second_derivative_tables)
                    seq._bj_dd_tables = _second_derivative_tables(seq)
                kk = _assemble_weighted_1d_mass(
                    seq._bj_dd_tables[axis], quad_w[axis] * prof)
        elif axis in deriv_axes:
            kk = _ktilde_1d(seq, axis, m,
                            bundled_axis_profiles(seq, jac)[axis])
        else:
            kk = _assemble_weighted_1d_stiffness(
                primal[axis], deriv[axis], quad_w[axis] * prof,
                jnp.asarray(_dense_incidence_1d(int(m.shape[0]),
                                                seq.basis_0.types[axis])))
        return _simultaneous_diagonalize_pair(m, kk)

    v_t, lam_t = angular(1)
    v_z, lam_z = angular(2)
    k_rr, m_rr = radial_pair
    den = (alpha[0] * k_rr
           + m_rr * (alpha[1] * lam_t[:, None] + alpha[2] * lam_z[None, :]))
    dmax = jnp.max(jnp.abs(den))
    den = jnp.where(jnp.abs(den) < 1e-10 * dmax, jnp.inf, den)
    return {"v_t": v_t, "v_z": v_z, "inv_den": 1.0 / den}


def apply_ring_atom(fac, x2d):
    y = fac["v_t"].T @ jnp.asarray(x2d) @ fac["v_z"]
    y = y * fac["inv_den"]
    return fac["v_t"] @ y @ fac["v_z"].T


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
                 ktilde_mode="roundtrip", lumped=False, extra_rings=0,
                 outer_rings=0, radial="averaged", bc_entry="exact",
                 core_mode="dense"):
        self.k, self.dirichlet = k, dirichlet
        self.ktilde_mode, self.lumped = ktilde_mode, lumped
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
            if radial == "modal":
                atom = ("modal", build_modal_radial_atom(
                    seq, k, c, (r0, nr), ktilde_mode=ktilde_mode))
            else:
                atom = build_bulk_atom(
                    seq, k, c, window=(r0, nr), ktilde_mode=ktilde_mode,
                    lumped=lumped, dirichlet=dirichlet,
                    bc_entry=(False if bc_entry in ("woodbury", "wdiag")
                              else bc_entry))
            cap = None
            # The natural-BC trace exists only on the components the
            # integration by parts actually touches:
            #   k=1  <u, grad tau>   -> +int (u.n) tau        -> NORMAL, c = r
            #   k=2  <w, curl tau>   -> +int (w x n).tau      -> TANGENTIAL, c = t,z
            #   k=3  <om, div tau>   -> +int om (tau.n)       -> the one component
            #   k=0  W_0 = 0                                  -> none
            # Equivalently: wherever the component's RADIAL axis is a derivative
            # axis, which is where delta acts. Building it on w_r at k=2 adds a
            # term the operator does not have.
            has_trace = k > 0 and (0 in {0: (), 1: (c,), 3: (0, 1, 2)}.get(
                k, tuple(a for a in range(3) if a != c)))
            if (bc_entry in ("woodbury", "wdiag") and not dirichlet
                    and has_trace):
                # Natural BC by capacitance: the term is assembled on the face
                # by quadrature (no probes) and folded in exactly through
                # Woodbury, so nothing is lumped to a scalar.
                cap = capacitance_correction(
                    atom, face_operator(seq, k, c, (r0, nr)),
                    diagonal_only=(bc_entry == "wdiag"))
            dscale = None
            if lumped == "diag":
                d_full = component_diagonal(seq, k, c, shape)
                dscale = 1.0 / np.sqrt(np.maximum(
                    d_full[r0:r0 + nr, :, :], 1e-300))
            self.blocks.append({
                "rows": rows_b[sel], "vals": vals_b[sel],
                "idx": (i_r - r0, i_t, i_z), "shape": (nr, shape[1], shape[2]),
                "atom": atom, "dscale": dscale, "cap": cap})

        # "dense": probe the whole core (polar + rings) and invert it exactly.
        # "atom2d": probe only the POLAR rows -- whose extraction genuinely
        # mixes a ring of raw DOFs -- and give each added radial ring a
        # separable 2-D atom instead. No probe applies for the rings, and
        # O(n_t^2 + n_z^2) storage rather than (n_t n_z)^2.
        # The split is not cosmetic: measured, the 2-D atom MATCHES the dense
        # probe on the inner rings at a tenth of the build cost, and LOSES
        # badly on the outer ones (toroid k=3 free 65 vs 24, W7-X k=2 free 616
        # vs 198).  The outer ring earns its keep through the radial coupling a
        # separable ring cannot carry -- the Steklov/DtN operator is nonlocal --
        # while the inner rings only ever fixed a bad radial AVERAGE, which is
        # exactly what a ring-local profile fixes for free.
        self.core_mode = core_mode
        probe_rows = (np.union1d(polar, outer_ring_rows)
                      if core_mode == "atom2d" else core)
        self.probe_rows = probe_rows
        block = probe_core_block(seq, operators, k, dirichlet, probe_rows)
        if block.size:
            w, v = np.linalg.eigh(block)
            keep_w = np.abs(w) > core_tol * np.abs(w).max()
            self.core_inv = (v[:, keep_w] / w[keep_w]) @ v[:, keep_w].T
        else:
            self.core_inv = np.zeros((0, 0))

        self.rings = []
        if core_mode == "atom2d" and inner_rings.size:
            self.rings = self._build_ring_blocks(seq, k, dirichlet, inner_rings)

    def _build_ring_blocks(self, seq, k, dirichlet, ring_rows):
        """One 2-D atom per (component, radial ring) among the added rings."""
        e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
        rows, cols, vals = (np.asarray(e.rows), np.asarray(e.cols),
                            np.asarray(e.vals))
        keep = np.isin(rows, ring_rows)
        rows_r, cols_r, vals_r = rows[keep], cols[keep], vals[keep]
        starts = np.cumsum([0] + [int(np.prod(s_)) for s_ in self.shapes])
        comp = np.searchsorted(starts[1:], cols_r, side="right")
        loc = cols_r - starts[comp]

        out = []
        for c, shape in enumerate(self.shapes):
            sel = comp == c
            if not sel.any():
                continue
            # Radial factors only for their DIAGONAL entries: restricted to
            # one ring, the radial term is the scalar K_r[i,i], and the mass
            # scale multiplying the angular part is M_r[i,i].
            masses, stiffs, alpha = component_factors(
                seq, k, c, window=None, ktilde_mode=self.ktilde_mode,
                lumped=self.lumped, bc_entry=False, dirichlet=dirichlet)
            k_r = np.asarray(stiffs[0])
            m_r = np.asarray(masses[0])
            lidx = loc[sel]
            i_r = lidx // (shape[1] * shape[2])
            i_t = (lidx // shape[2]) % shape[1]
            i_z = lidx % shape[2]
            dfull = (component_diagonal(seq, k, c, shape)
                     if self.lumped == "diag" else np.ones(shape))
            for ridx in np.unique(i_r):
                m = i_r == ridx
                d = np.maximum(dfull[int(ridx)], 1e-300)
                out.append({
                    "rows": jnp.asarray(rows_r[sel][m]),
                    "vals": jnp.asarray(vals_r[sel][m]),
                    "flat": jnp.asarray(i_t[m] * shape[2] + i_z[m]),
                    "shape2": (shape[1], shape[2]),
                    # the ring's own metric, kept EXACTLY as a diagonal -- this
                    # is the information the axis-averaged bulk atom discards
                    # and the only reason a separable ring can compete with an
                    # exact one.
                    "dscale": jnp.asarray(1.0 / np.sqrt(d)),
                    "atom": build_ring_atom(
                        seq, k, c, int(ridx),
                        (float(k_r[int(ridx), int(ridx)]),
                         float(m_r[int(ridx), int(ridx)])),
                        alpha, ktilde_mode=self.ktilde_mode)})
        return out

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
                "cap": (None if blk.get("cap") is None
                        else _cap_arrays(blk["cap"])),
            })
        core = jnp.asarray(self.probe_rows)
        core_inv = jnp.asarray(self.core_inv)
        has_core = np.asarray(self.probe_rows).size > 0
        rings = self.rings

        def impl(x):
            out = jnp.zeros_like(x)
            for b in blocks:
                buf = jnp.zeros(int(np.prod(b["shape"]))).at[b["flat"]].set(
                    b["vals"] * x[b["rows"]]).reshape(b["shape"])
                if b["dscale"] is not None:
                    buf = buf * b["dscale"]
                sol = _fd_apply_3d(*b["v"], *b["lam"], b["alpha"], buf)
                if b["cap"] is not None:
                    ring = sol[-1].reshape(-1)
                    upd = jnp.zeros(b["shape"]).at[-1].set(
                        _apply_cap_jax(b["cap"], ring).reshape(
                            b["shape"][1], b["shape"][2]))
                    sol = sol - _fd_apply_3d(*b["v"], *b["lam"], b["alpha"], upd)
                if b["dscale"] is not None:
                    sol = sol * b["dscale"]
                out = out.at[b["rows"]].set(b["vals"] * sol.reshape(-1)[b["flat"]])
            if has_core:
                out = out.at[core].set(core_inv @ x[core])
            for r in rings:
                buf = jnp.zeros(int(np.prod(r["shape2"]))).at[r["flat"]].set(
                    r["vals"] * x[r["rows"]]).reshape(r["shape2"])
                buf = buf * r["dscale"]
                sol = apply_ring_atom(r["atom"], buf) * r["dscale"]
                out = out.at[r["rows"]].set(
                    r["vals"] * sol.reshape(-1)[r["flat"]])
            return out

        return jax.jit(impl)

    def apply(self, x):
        """Apply the preconditioner to an extracted-space vector."""
        if getattr(self, "_jit", None) is None:
            self._jit = self._build_apply()
        return self._jit(jnp.asarray(x))


def _cap_arrays(cap):
    if cap["mode"] == "diag":
        return ("diag", jnp.asarray(cap["q"]), jnp.asarray(cap["coef"]))
    return ("dense", jnp.asarray(cap["c"]))


def _apply_cap_jax(cap, ring):
    if cap[0] == "diag":
        _, q, coef = cap
        return q @ (coef * (q.T @ ring))
    return cap[1] @ ring


# --------------------------------------------------------------------------- #
# k=3 by transfer into the k=0 atom                                            #
# --------------------------------------------------------------------------- #

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


class TransferK3Preconditioner:
    """``L_3`` preconditioned through the k=0 space.

    ``S_3 = 0``, so ``L_3`` IS the weak term, and a weak term's energy is a norm
    on the space below: ``u^T W_3 u = ||delta u||^2`` with ``delta u`` in
    ``V_2``.  Hodge duality takes that all the way to ``V_0`` -- ``L_3 ~ L_0``
    with the BOUNDARY CONDITIONS FLIPPED, so k=3 free pairs with k=0 dbc and
    k=3 dbc with k=0 free.  Since ``W_3 = M_3 (D M_2^-1 D^T) M_3``, the bracket
    is what corresponds to ``S_0``, and the masses have to be put back::

        P_3 = M_3^-1 Q P_0 Q^T M_3^-1

    ``Q`` is the metric-free separable transfer (:func:`mixed_mass_1d`) and
    ``P_0`` the k=0 block-Jacobi atom -- which is the strongest thing we have
    (4.8-6.6x over Jacobi), and the reason to expect more from this than the
    earlier attempt that used the k=0 Jacobi DIAGONAL as ``P_0``.
    """

    def __init__(self, seq, operators, dirichlet, **kwargs):
        self.inner = BlockJacobiLaplacian(seq, operators, 0, not dirichlet,
                                          **kwargs)          # <- the BC flip
        q_raw = np.kron(np.kron(mixed_mass_1d(seq, 0).T, mixed_mass_1d(seq, 1).T),
                        mixed_mass_1d(seq, 2).T)             # (n3_raw, n0_raw)

        def dense(e, ncol):
            m = np.zeros((int(e.forward_shape[0]), ncol))
            m[np.asarray(e.rows), np.asarray(e.cols)] = np.asarray(e.vals)
            return m

        e3 = getattr(seq, "e3_dbc" if dirichlet else "e3")
        e0 = getattr(seq, "e0" if dirichlet else "e0_dbc")
        self.q = (dense(e3, q_raw.shape[0]) @ q_raw
                  @ dense(e0, q_raw.shape[1]).T)              # (n3ext, n0ext)
        from mrx.local_assembly import build_mass_diagonal  # noqa: PLC0415
        raw_d = np.asarray(build_mass_diagonal(seq, 3))
        rows, cols, vals = (np.asarray(e3.rows), np.asarray(e3.cols),
                            np.asarray(e3.vals))
        d = np.zeros(int(e3.forward_shape[0]))
        np.add.at(d, rows, vals ** 2 * raw_d[cols])
        self.m3inv = 1.0 / np.maximum(d, 1e-300)

    def apply(self, x):
        w = self.m3inv * np.asarray(x)
        return jnp.asarray(self.m3inv * (self.q @ np.asarray(
            self.inner.apply(jnp.asarray(self.q.T @ w)))))


# --------------------------------------------------------------------------- #
# Modal-radial atom: no radial averaging                                       #
# --------------------------------------------------------------------------- #

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


def build_modal_radial_atom(seq, k, c, window, ktilde_mode="honest"):
    """Diagonalise theta/zeta, solve the radial direction EXACTLY.

    Mirrors :func:`mrx.operators._assemble_k0_modal_radial_bulk_factors`: the
    metric lives entirely in the RADIAL profiles and the angular factors are
    unweighted, so no radial average is taken anywhere. Per ``(j,k)`` angular
    mode the radial operator is ``K_r + mu_j M_b + nu_k M_c``; ``M_c ~ kappa
    M_a`` collapses that to one parameter, so it costs ``n_zeta`` small radial
    eigendecompositions rather than ``n_theta * n_zeta``.
    """
    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]
    deriv_axes = {0: (), 1: (c,), 3: (0, 1, 2)}.get(
        k, tuple(a for a in range(3) if a != c))
    # k=0 WEIGHTS ONLY. The component factor g^{cc} is carried by the diagonal
    # sandwich in BlockJacobiLaplacian; folding it in here as well applies it
    # twice. That bug is invisible at k=0 (the factor is 1 there) and cost 5x
    # at k=1 -- exactly the observed pattern.
    rad = [radial_profiles(seq, ginv[a] * jac) for a in range(3)]

    primal, deriv, quad_w = _axis_bases(seq)
    types = seq.basis_0.types
    lo, nr = window

    def cut(m):
        return m[lo:lo + nr, lo:lo + nr]

    def mass(a, w):
        return _assemble_weighted_1d_mass(
            deriv[a] if a in deriv_axes else primal[a], w)

    def stiff(a, w):
        if a in deriv_axes:
            if int(seq.basis_0.Λ[0].p) < 2:
                return _fd_stiffness_degree0(seq, a, w / quad_w[a])
            if not hasattr(seq, "_bj_dd_tables"):
                from mrx.local_assembly import (  # noqa: PLC0415
                    _second_derivative_tables)
                seq._bj_dd_tables = _second_derivative_tables(seq)
            return _assemble_weighted_1d_mass(seq._bj_dd_tables[a], w)
        n_in = int(mass(a, quad_w[a]).shape[0])
        return _assemble_weighted_1d_stiffness(
            primal[a], deriv[a], w,
            jnp.asarray(_dense_incidence_1d(n_in, types[a])))

    k_r = cut(stiff(0, quad_w[0] * rad[0]))
    m_a = cut(mass(0, quad_w[0] * rad[0]))
    m_b = cut(mass(0, quad_w[0] * rad[1]))
    m_c = cut(mass(0, quad_w[0] * rad[2]))
    m_t, k_t = mass(1, quad_w[1]), stiff(1, quad_w[1])
    m_z, k_z = mass(2, quad_w[2]), stiff(2, quad_w[2])

    v_t, mu = _simultaneous_diagonalize_pair(m_t, k_t)
    v_z, nu = _simultaneous_diagonalize_pair(m_z, k_z)
    kappa = float(jnp.sum(m_c * m_a) / jnp.sum(m_a * m_a))
    ws, ds = [], []
    for j in range(int(nu.shape[0])):
        w_j, d_j = _simultaneous_diagonalize_pair(
            m_b, k_r + (kappa * float(nu[j])) * m_a)
        ws.append(w_j)
        ds.append(d_j)
    return {"V_t": v_t, "V_z": v_z, "mu": mu,
            "W": jnp.stack(ws), "d": jnp.stack(ds)}


def apply_modal_radial(fac, x):
    """theta/zeta transforms, an exact radial solve per mode, and back."""
    y = jnp.einsum('tj,rtz->rjz', fac["V_t"], jnp.asarray(x))
    y = jnp.einsum('zk,rjz->rjk', fac["V_z"], y)
    y = jnp.einsum('krs,rjk->sjk', fac["W"], y)
    den = fac["d"].T[:, None, :] + fac["mu"][None, :, None]
    den_max = jnp.max(jnp.abs(den))
    null = jnp.abs(den) < 1e-10 * den_max
    y = jnp.where(null, 0.0, y / jnp.where(null, 1.0, den))
    y = jnp.einsum('krs,sjk->rjk', fac["W"], y)
    y = jnp.einsum('zk,rjk->rjz', fac["V_z"], y)
    return jnp.einsum('tj,rjz->rtz', fac["V_t"], y)


# --------------------------------------------------------------------------- #
# Block-Jacobi MASS preconditioner                                             #
# --------------------------------------------------------------------------- #

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

def face_operator(seq, k, c, window):
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
    w_face = np.asarray(mass_weight * ginv[0])[-1]          # (n_qt, n_qz)

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
    return 0.5 * (b + b.T) * float(e_r[-1]) ** 2


def capacitance_correction(atom, b_face, diagonal_only=False):
    """Woodbury data for ``(A + E B E^T)^-1``.

    ``E`` injects onto the LAST radial index, so ``E^T A^-1 E`` never mixes
    angular modes: in the angular eigenbasis it is exactly

        d_jk = sum_i V_r[last,i]^2 / (a0 lam_r(i) + a1 lam_t(j) + a2 lam_z(k))

    -- closed form, one O(n_r n_t n_z) pass, no solves and no probes.  What is
    left is a purely 2-D system of size ``n_t n_z``.

    ``diagonal_only`` keeps only ``B``'s diagonal in the angular eigenbasis: no
    dense 2-D solve at all, and strictly better than collapsing the face weight
    to a scalar.
    """
    (v_r, v_t, v_z), (l_r, l_t, l_z), alpha = atom
    v_r, v_t, v_z = (np.asarray(v) for v in (v_r, v_t, v_z))
    l_r, l_t, l_z = (np.asarray(lam) for lam in (l_r, l_t, l_z))

    denom = (alpha[0] * l_r[:, None, None] + alpha[1] * l_t[None, :, None]
             + alpha[2] * l_z[None, None, :])
    dmax = np.abs(denom).max()
    safe = np.where(np.abs(denom) < 1e-10 * dmax, np.inf, denom)
    d = np.einsum('i,ijk->jk', v_r[-1] ** 2, 1.0 / safe).reshape(-1)

    q = np.kron(v_t, v_z)                       # angular eigenbasis
    if diagonal_only:
        b_hat_diag = np.einsum('ij,jk,ki->i', q.T, b_face, q)
        return {"mode": "diag", "q": q, "coef": b_hat_diag / (1.0 + b_hat_diag * d)}
    d_phys = q @ (d[:, None] * q.T)
    n = b_face.shape[0]
    c = np.linalg.solve(np.eye(n) + b_face @ d_phys, b_face)
    return {"mode": "dense", "c": c}


def apply_capacitance(corr, ring):
    """Apply the 2-D correction to one ring slice (flattened ``(n_t, n_z)``)."""
    if corr["mode"] == "diag":
        q = corr["q"]
        return q @ (corr["coef"] * (q.T @ ring))
    return corr["c"] @ ring
