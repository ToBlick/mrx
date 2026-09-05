"""The metric-lumping preconditioners: ``M_k^-1`` and ``L_k^-1`` as a separable bulk plus a dense polar core.

One atom per ``(k, dirichlet)`` (``docs/source/concepts/preconditioning.md``):
:class:`MetricLumpingMass` for the mass and :class:`MetricLumpingLaplacian`
for the Hodge Laplacian, built by ``DeRhamSequence.build_preconditioners``
onto the operator bundle.

The shape is the same at every ``k``, block Jacobi over two blocks that are
NOT coupled (no Schur complement is formed):

* **bulk** -- the tensor-product DoFs. For the mass, one Kronecker product
  ``M_r (x) M_t (x) M_z`` per component, inverted by three 1-D solves. For
  the Laplacian, one Kronecker SUM per component,
  ``K_r (x) M_t (x) M_z + M_r (x) K_t (x) M_z + M_r (x) M_t (x) K_z``,
  inverted by fast diagonalisation (:func:`mrx.operators._fd_apply_3d`);
* **core** -- the polar rows (the extraction's fused rows), probed through
  the operator itself and inverted densely.

**Metric lumping**, which names the module: the 3-D weight of each 1-D
factor is a *bundled* axis mean of the metric field, ``<g^{aa} J>`` taken as
one product over the other two directions (:func:`bundled_axis_profiles`;
``g^{tt} J ~ 1/r`` is integrable where ``g^{tt}`` alone is not), with the
component's own factor ``g^{cc}`` kept exactly as a diagonal sandwich
``D_c^{-1/2} (.) D_c^{-1/2}`` around the separable inverse (the scalar and
unlumped variants were measured and lost). The 1-D masses are unweighted:
the metric goes into the
stiffness profiles only.

The Laplacian's 1-D stiffness on each axis depends on whether the
component's basis is a derivative spline there (:func:`component_factors`):
on the primal axes it is the ordinary weighted stiffness of the primal
splines; on the derivative axes -- where ``L_k``'s weak half
``D M^{-1} D^T`` acts -- it is the stiffness OF the derivative splines
themselves, from their tabulated derivatives (``seq.dd_basis_jk``). The
round-trip alternative, ``M G A^{-1} G^T M`` through ``V_0``, lost every A/B
row and is gone. Conditioning works
out the right way round: the curl-curl stiffnesses are singular (constants),
the derivative-axis term is SPD, and the sum is nonsingular -- which is why
the two halves form ONE atom.

Approximations, all deliberate: off-diagonal component blocks dropped (metric
off-diagonals and the cross-component derivative couplings); each bundled
3-D weight collapsed to a product of axis profiles; no bulk<->core coupling.

The natural-BC boundary term IS carried, on the components whose radial
axis is a derivative axis, as a rank-one update to the radial stiffness:
under a free condition the weak block's integration by parts leaves
``int_{r=1} w u_r^2``, which for a tensor basis is
``alpha (e e^T) (x) M_t (x) M_z`` -- the shape of the first Kronecker term,
so it merges into ``K_r``. ``alpha`` is the face average of the weight
(:func:`_face_alpha`) times :data:`PRODUCTION_BC_SCALE`, a measured fit
(``docs/research/natural_bc_coefficient_handoff.md``). It vanishes under
Dirichlet and at k=0.

Every atom is a pytree payload with one jitted apply per tree structure, built
at construction (never on the first apply), so a rebuild for a new geometry
does not recompile.
"""

from __future__ import annotations

import functools

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.operators import (
    _fd_apply_3d,
    _fd_apply_3d_shifted,
    _assemble_weighted_1d_stiffness,
    _dense_incidence_1d,
)
from mrx.precision import DTYPE, RESIDUAL_DTYPE, sqrt_eps
from mrx.preconditioners import _assemble_weighted_1d_mass, _simultaneous_diagonalize_pair

#: Relative cut-off below which an eigenvalue of the probed dense core is
#: treated as exactly zero: 4096 machine epsilons of the RESIDUAL precision,
#: ~1e-12 in float64, the precision the cores are probed and inverted in
#: (see :func:`_probe_rows`). In the working precision it was 5e-4 in a
#: float32 process, which zeroed real modes of the k=1 Laplacian core on
#: li383 (12,24,24) p=3: the preconditioner was singular on them, the CG's
#: preconditioned criterion blind to their residual, and the k=1 Dirichlet
#: solve reported convergence with a true residual of 2e-6 (2026-09-05).
CORE_TOL = 4096.0 * float(jnp.finfo(RESIDUAL_DTYPE).eps)

# --------------------------------------------------------------------------- #
# Bundled axis profiles                                                        #
# --------------------------------------------------------------------------- #

def _polar_cut_weight(seq):
    """Radial quad weight with the polar-surgery element removed: the core
    DOFs are handled by their own dense block, so they must not contribute
    to the bulk averages."""
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
    """The metric weight fields the atoms reduce to axis profiles, at quadrature points.

    ``jac`` = ``J``, ``ginv_aa`` = ``(g^{rr}, g^{tt}, g^{zz})``, ``met_aa`` =
    ``(g_{rr}, g_{tt}, g_{zz})``.  Products such as ``g^{aa} J`` are formed by
    the consumer on the fly and reduced immediately (ten more full-size
    arrays here OOMed at ``(68, 136, 68)`` p=4).
    """
    shape = seq.quad.shape
    jac = jnp.asarray(seq.geometry.jacobian_j).reshape(shape)
    ginv = jnp.asarray(seq.geometry.metric_inv_jkl).reshape(*shape, 3, 3)
    met = jnp.asarray(seq.geometry.metric_jkl).reshape(*shape, 3, 3)
    return {
        "jac": jac,
        "ginv_aa": tuple(ginv[..., a, a] for a in range(3)),
        "met_aa": tuple(met[..., a, a] for a in range(3)),
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
    bulk atom by RESTRICTING THE RADIAL WINDOW (``cut`` below), not by
    zeroing quadrature weights.
    """
    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)
    return primal, deriv, quad_w
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
    # jump form has to be conjugated by ``diag(1/h)``. Without it the factor
    # is under-scaled by h^2, which no fixed multiplier repairs: the radial
    # direction of the atom is left with essentially no stiffness.
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


def _face_alpha(seq, k, c):
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

    The component factor ``w_comp = m_k/J`` is carried outside as the ``D``
    sandwich, so it has to come back out here or it is counted twice.  That division is also what makes the coefficient
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

    m_k = {1: ginv[c] * jac, 2: met[c] / jac, 3: 1.0 / jac}[k]
    return fm(m_k * jnp.sqrt(ginv[0])) / fm(m_k / jac), 1.0 / _h_last(seq)


#: The natural-BC penalty scale on the face coefficient of :func:`_face_alpha`.
#: ``alpha`` as derived is the surface integral itself, the best NORM
#: approximation to ``L``'s boundary block; the preconditioner wants a larger
#: number, and 3.0 is inside the flat optimum [2, 4] of a 24-cell sweep (4
#: geometries x k=1,2,3 x p=2,3,5 x two meshes, ranked by total iterations;
#: docs/research/natural_bc_coefficient_handoff.md). Not degree-dependent:
#: the best single scale drifts 2 -> 2.83 over p=2..5 at a cost of 2-3%.
PRODUCTION_BC_SCALE = 3.0


def _boundary_entry(seq, window, alpha):
    """The natural-BC boundary term as a rank-one update to ``K_r``.

    Under a free condition the weak block's integration by parts leaves
    ``int_{r=1} w . u_r^2``; for a tensor basis that is ``alpha (e e^T) (x)
    M_t (x) M_z`` with ``e = dLam(1)`` on the bulk radial ``window``, the
    shape of the first Kronecker-sum term. ``alpha`` is the face coefficient
    of :func:`_face_alpha`, scaled by :data:`PRODUCTION_BC_SCALE`: the exact
    surface integral is a penalty on the normal trace, and the atom wants it
    closer to the hard ``u_r = 0`` limit.
    """
    dlam = seq.basis_0.dΛ[0]
    e = jax.vmap(lambda i: jnp.sum(dlam(_boundary_point(seq), i)))(dlam.ns)
    e = e[window[0]:window[0] + window[1]]
    return PRODUCTION_BC_SCALE * alpha * jnp.outer(e, e)


def component_factors(seq, k, c, window, dirichlet):
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
    primal, deriv, quad_w = _axis_bases(seq)
    fields = weight_fields(seq)
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
    ginv, jac = fields["ginv_aa"], fields["jac"]
    deriv_axes = getattr(seq, f"basis_{k}").derivative_axes(c)
    # DIAGONAL lumping. w(c,a) = g^{cc} * (g^{aa}J) factors into a component
    # part and an axis part, so assemble the 1-D factors with the k=0 weights
    # ONLY -- shared by every component and every degree -- and carry g^{cc}
    # as a diagonal sandwich instead of inside the averages:
    #
    #     P_c = D_c^{-1/2} FD^{-1} D_c^{-1/2}
    #
    # Unlike scalar lumping (measured 2-8x worse, deleted) this keeps the
    # FIELD exactly and gives up only its correlation with the axis averages.
    stiff_prof = [bundled_axis_profiles(seq, ginv[a] * jac)[a]
                  for a in range(3)]

    def cut(mat, axis):
        """Radial window: the bulk atom lives on the bulk DOFs only."""
        if axis != 0:
            return mat
        lo, n = window
        return mat[lo:lo + n, lo:lo + n]

    masses, stiffs = [], []
    for a in range(3):
        basis = deriv[a] if a in deriv_axes else primal[a]
        # UNWEIGHTED mass, per the validated k=0 fd/fdbund recipe (adopted
        # 2026-08-13, since folded into this atom): the bundled metric goes into the
        # STIFFNESS profiles only. In K_r (x) M_t (x) M_z the M's are just
        # "int phi phi in the other directions" -- g^{aa}J has already been
        # folded into K_r by averaging over exactly those directions, so
        # weighting the masses as well double counts it.
        m_full = _assemble_weighted_1d_mass(basis, quad_w[a])
        masses.append(cut(m_full, a))
        if a in deriv_axes:
            # The honest thing: the 1-D stiffness OF the derivative splines.
            # With their derivative values tabulated, that is just a weighted
            # mass of the table -- no incidence, no A^-1, and so nothing to
            # mis-scale (the M A^-1 M factor of the deleted round-trip form
            # existed only because that form dragged M_0^-1 in).
            prof = stiff_prof[a]
            if degree0:
                # p = 1: the DG-0 jump stand-in, in the same (value-scaled)
                # normalisation as the honest form -- so the natural-BC block
                # below applies to it unchanged. e = dLam(1) is (0,...,1/h) at
                # degree 0, which is that same convention on the face.
                kt = cut(_fd_stiffness_degree0(seq, a, prof), a)
            else:
                kt = cut(_assemble_weighted_1d_mass(
                    seq.dd_basis_jk[a], quad_w[a] * prof), a)
            # The natural-BC trace lives on the RADIAL derivative axis only:
            # the boundary face is r = 1, theta and zeta are periodic. Other
            # spellings of the term (the exact 2-D face shape, cross-term
            # corrections) were measured and lost;
            # docs/research/natural_bc_coefficient_handoff.md §9, §12.3, §14.3.
            if a == 0 and not dirichlet:
                scalar, amp = _face_alpha(seq, k, c)
                kt = kt + _boundary_entry(seq, window, scalar) * amp
            stiffs.append(kt)
        else:
            k_full = _assemble_weighted_1d_stiffness(
                primal[a], deriv[a], quad_w[a] * stiff_prof[a],
                _dense_incidence_1d(int(m_full.shape[0]),
                                    seq.basis_0.types[a]))
            stiffs.append(cut(k_full, a))
    return tuple(masses), tuple(stiffs)


def component_diagonal(seq, k, c, shape):
    """Support-averaged component factor at each DOF: ``D_i``.

    ``D_i = int phi_i^2 (w_comp J) / int phi_i^2 J`` -- the numerator is the
    k-form mass diagonal and the denominator the same basis against the 0-form
    weight, so the ratio is the component factor averaged over each basis
    function's own support. Exact, no fit.
    """
    fields = weight_fields(seq)
    jac = fields["jac"]
    w_comp = {0: jnp.ones_like(jac), 1: fields["ginv_aa"][c],
              2: fields["met_aa"][c] / jac ** 2, 3: 1.0 / jac ** 2}[k]
    primal, deriv, quad_w = _axis_bases(seq)
    deriv_axes = getattr(seq, f"basis_{k}").derivative_axes(c)
    tabs = [(deriv[a] if a in deriv_axes else primal[a]) ** 2
            for a in range(3)]
    wq = seq.quad.w.reshape(seq.quad.shape)

    def contract(field):
        f = wq * field
        t1 = jnp.einsum('ax,xyz->ayz', tabs[0], f)
        t2 = jnp.einsum('by,ayz->abz', tabs[1], t1)
        return jnp.einsum('cz,abz->abc', tabs[2], t2)

    num = contract(w_comp * jac)
    den = contract(jac)
    return (num / den).reshape(shape)


def build_bulk_atom(seq, k, c, window, dirichlet):
    """Fast-diagonalisation factors for component ``c`` of ``L_k``:
    ``((V_r, V_t, V_z), (lam_r, lam_t, lam_z))`` for
    :func:`mrx.operators._fd_apply_3d`."""
    masses, stiffs = component_factors(seq, k, c, window, dirichlet)
    vs, lams = [], []
    for a in range(3):
        v, lam = _simultaneous_diagonalize_pair(masses[a], stiffs[a])
        vs.append(v)
        lams.append(lam)
    return tuple(vs), tuple(lams)


# --------------------------------------------------------------------------- #
# Core block: probed and densely inverted                                      #
# --------------------------------------------------------------------------- #

def core_rows(seq, k, dirichlet):
    """``(core, bulk, e)``: the extracted rows handled by the dense core block
    (the polar ring, where the extraction fuses raw DOFs -- the rows
    ``wx_cut`` removes from the bulk averages), the rest, and the extraction.

    Fattening the core with the innermost or outermost radial rings was
    measured (dense outer-ring probes) and lost; the core is the polar ring
    only.
    """
    e = seq.E(k, dirichlet)
    rows = np.asarray(e.rows)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)
    core = np.flatnonzero(counts > 1)
    bulk = np.setdiff1d(np.arange(n_ext), core)
    return core, bulk, e


def _probe_rows(apply, size, rows, dtype=DTYPE):
    """Dense ``A`` restricted to ``rows``, by one apply per row, on device,
    in ``dtype``: the cores are probed on the residual-precision sequence
    (:func:`_probing_sequence`) so that their inversion at :data:`CORE_TOL`
    drops the kernel and nothing else.

    A Python loop of asynchronous dispatches of an already-compiled apply
    (``lax.map`` over the rows compiles a fresh scan per ``(k, BC)`` and was
    measured at three times the build time).
    """
    if rows.size == 0:
        return jnp.zeros((0, 0), dtype=dtype)
    rows_j = jnp.asarray(rows)
    block = jnp.stack(
        [apply(jnp.zeros(size, dtype=dtype).at[int(i)].set(1.0))[rows_j]
         for i in rows], axis=1)
    return 0.5 * (block + block.T)


def _probing_sequence(seq):
    """The sequence the dense cores are probed on: the float64 view, or
    ``seq`` itself in the residual precision."""
    return seq if seq.residual is None else seq.residual


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
    from mrx.operators import apply_laplacian_approx  # noqa: PLC0415

    size = int(seq.n(k, dirichlet))
    return _probe_rows(
        lambda x: apply_laplacian_approx(seq, operators, x, k,
                                               dirichlet=dirichlet),
        size, rows, dtype=seq.dtype)
# --------------------------------------------------------------------------- #
# The applied payload, as a pytree                                             #
# --------------------------------------------------------------------------- #
#
# The arrays of an atom are LEAVES of a pytree passed to one jitted apply, so
# two payloads with the same shapes share a treedef and a compiled program
# (an apply that closed over its arrays recompiled per object, ~287 ms).
# Leaves are arrays whose values change and whose shapes do not; static is
# anything used in Python control flow or as a reshape target.


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
    alpha: jnp.ndarray           # leaf: the weight of each Kronecker term
    dscale: jnp.ndarray          # leaf: the diagonal sandwich
    shape: tuple = eqx.field(static=True)     # STATIC: reshape target
    # STATIC: when the block's rows are the contiguous range starting at
    # ``offset`` with unit weights (a pure selector, e.g. every k=3 block)
    # the gather and the multiply are a static slice instead.
    offset: int = eqx.field(static=True)      # -1 when not a selector


class _LumpPayload(eqx.Module):
    """Everything the apply reads. One treedef per (k, BC, discretisation)."""

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
    Selector blocks and an identity output order are static slices / no-ops,
    so a k=3 apply is the fast-diagonalisation solve and nothing else.
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


class _ShiftedPayload(eqx.Module):
    """The shifted-stiffness atom's leaves: the Laplacian blocks with their
    strong-half mask as ``alpha``, and the core's ``(M, S)`` pair
    diagonalised (``V^T M V = I``, ``V^T S V = diag(mu)``), from which the
    core inverse ``V diag(1 / (1 + eps mu)) V^T`` is two matmuls per solve."""

    blocks: tuple                # leaves, one _LumpBlock per component
    core: jnp.ndarray            # leaf
    core_V: jnp.ndarray          # leaf: the M-orthonormal eigenvectors of (S, M) on the core
    core_mu: jnp.ndarray         # leaf: their generalised eigenvalues
    perm: jnp.ndarray            # leaf
    has_core: bool = eqx.field(static=True)
    identity_perm: bool = eqx.field(static=True)


def _apply_shifted_payload(payload: _ShiftedPayload, core_inv, inv_eps, x):
    """``(M^ + eps S^)^-1 x``: per block ``(1/eps) D^{-1/2} FD(alpha_strong,
    shift 1/eps) D^{-1/2}``, and ``core_inv`` on the core rows."""
    parts = []
    for b in payload.blocks:
        buf = _block_input(b, x) * b.dscale
        sol = _fd_apply_3d_shifted(b.v_r, b.v_t, b.v_z,
                                   b.lam_r, b.lam_t, b.lam_z, b.alpha, buf, inv_eps)
        parts.append(inv_eps * _block_output(b, sol * b.dscale))
    if payload.has_core:
        parts.append(core_inv @ x[payload.core])
    return _place(payload, parts)


def _tensor_blocks(seq, k, dirichlet):
    """Split the extraction into per-component tensor blocks plus the core.

    Returns ``(core, e, blocks)`` where ``blocks``
    holds, per component, ``None`` or ``(rows, vals, (r0, nr), shape, offset)``
    with ``rows``/``vals`` in TENSOR order over the ``(nr, n_t, n_z)`` block
    and ``offset >= 0`` when the block is a pure selector (rows contiguous
    from ``offset``, unit weights). Raises if a component's bulk DOFs are not
    a full radial slab, since the separable atom does not apply then.
    """
    shapes = [tuple(int(s) for s in sh)
              for sh in getattr(seq, f"basis_{k}").shape]
    starts = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    core, bulk, e = core_rows(seq, k, dirichlet)
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
    return core, e, blocks


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
# The payload is flattened once at build time and the jitted apply cached on
# its treedef, so a call is one jitted call on a tuple of arrays with no tree
# walk (``eqx.filter_jit`` re-partitions the module per call, measured at
# 194 us per apply). The cache is bounded because its entries are compiled
# functions; a sweep over resolutions adds one treedef per shape.
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

    def __init__(self, seq, operators, k, dirichlet):
        form = getattr(seq, f"basis_{k}")
        self.shapes = [tuple(int(s) for s in sh) for sh in form.shape]

        core, e, tensor_blocks = _tensor_blocks(seq, k, dirichlet)
        self.core = core
        self.n_ext = int(e.forward_shape[0])

        self.blocks = []
        for c, blk in enumerate(tensor_blocks):
            if blk is None:
                self.blocks.append(None)
                continue
            rows_t, vals_t, (r0, nr), shape, offset = blk
            atom = build_bulk_atom(seq, k, c, (r0, nr), dirichlet)
            # D_i is a ratio of two positive integrals; no floor.
            d_full = component_diagonal(seq, k, c, self.shapes[c])
            dscale = 1.0 / jnp.sqrt(d_full[r0:r0 + nr, :, :])
            # The strong half S_k lives on the primal axes of the component
            # (component_factors); this mask selects its Kronecker terms for
            # the shifted-stiffness atom.
            alpha_strong = tuple(
                0.0 if a in form.derivative_axes(c) else 1.0 for a in range(3))
            self.blocks.append({
                "rows": rows_t, "vals": vals_t, "shape": shape,
                "offset": offset, "atom": atom, "dscale": dscale,
                "alpha_strong": alpha_strong})
        self.perm, self.identity_perm = _output_permutation(
            [b["rows"] for b in self.blocks if b is not None], core, self.n_ext)

        # Probe the whole core and invert it exactly (a separable 2-D ring
        # atom matches the dense probe on the inner rings and loses badly on
        # the outer ones: the Steklov/DtN operator is nonlocal).
        on = _probing_sequence(seq)
        self.core_inv = _dense_symmetric_inverse(
            probe_core_block(on, operators, k, dirichlet, core), CORE_TOL)
        # M_k and S_k on the core rows, diagonalised together ONCE: the
        # shifted-stiffness atom's core block is (M + eps S)^-1 = V diag(1 /
        # (1 + eps mu)) V^T with V^T M V = I and V^T S V = diag(mu), and eps
        # is only known at solve time.
        from mrx.operators import apply_mass_matrix, apply_stiffness  # noqa: PLC0415
        size = int(seq.n(k, dirichlet))
        mass_core = _probe_rows(
            lambda x: apply_mass_matrix(on, x, k, dirichlet=dirichlet), size, core, dtype=on.dtype)
        stiffness_core = _probe_rows(
            lambda x: apply_stiffness(on, x, k, dirichlet=dirichlet), size, core, dtype=on.dtype)
        if core.size > 0:
            self.core_V, self.core_mu = _simultaneous_diagonalize_pair(mass_core, stiffness_core)
        else:
            self.core_V, self.core_mu = mass_core, jnp.zeros(0, dtype=DTYPE)
        self._flat = _flatten_payload(self._build_payload())
        self._shifted = self._build_shifted_payload()

    def _build_payload(self):
        """Pack the factors into the :class:`_LumpPayload` pytree, eagerly
        (a first apply inside a ``lax`` body would stash tracers on this
        long-lived object). All three Kronecker terms weigh one: the
        stiffnesses carry their own weights."""
        return _LumpPayload(
            blocks=tuple(self._pack_blocks(lambda blk: (1.0, 1.0, 1.0))),
            core=jnp.asarray(self.core),
            core_inv=self.core_inv,
            perm=self.perm,
            has_core=bool(self.core.size > 0),
            identity_perm=self.identity_perm,
        )

    def apply(self, x):
        """Apply the preconditioner to an extracted-space vector."""
        leaves, jitted = self._flat
        return jitted(leaves, jnp.asarray(x))

    def _pack_blocks(self, alpha_of):
        """The component blocks as :class:`_LumpBlock` leaves with ``alpha =
        alpha_of(block)``."""
        blocks = []
        for blk in self.blocks:
            if blk is None:
                continue
            (v_r, v_t, v_z), (l_r, l_t, l_z) = blk["atom"]
            blocks.append(_LumpBlock(
                rows=jnp.asarray(blk["rows"]),
                vals=jnp.asarray(blk["vals"], dtype=DTYPE),
                v_r=v_r, v_t=v_t, v_z=v_z,
                lam_r=l_r, lam_t=l_t, lam_z=l_z,
                alpha=jnp.asarray(alpha_of(blk), dtype=DTYPE),
                dscale=blk["dscale"],
                shape=blk["shape"],
                offset=blk["offset"],
            ))
        return blocks

    def _build_shifted_payload(self):
        """The Laplacian blocks with ``alpha`` = the strong-half mask."""
        return _ShiftedPayload(
            blocks=tuple(self._pack_blocks(lambda blk: blk["alpha_strong"])),
            core=jnp.asarray(self.core),
            core_V=self.core_V,
            core_mu=self.core_mu,
            perm=self.perm,
            has_core=bool(self.core.size > 0),
            identity_perm=self.identity_perm,
        )

    def shifted_stiffness_apply(self, eps):
        """``x -> (M^_k + eps S^_k)^-1 x``, the preconditioner of ``M_k + eps S_k``.

        Per component the strong-half (primal-axis) Kronecker terms of this
        atom, divided by ``1 + eps lambda`` in their eigenbasis: exactly
        ``(M^ + eps S^)^-1`` for the atom's own separable mass
        ``D_c^{1/2} (m_r x m_t x m_z) D_c^{1/2}`` (unweighted 1-D masses,
        the component factor as the sandwich). It tends to ``M^-1`` as
        ``eps -> 0`` and to ``(1/eps) S^-1`` as ``eps -> inf``. The core
        rows get the dense ``(M + eps S)^-1 = V diag(1 / (1 + eps mu)) V^T``
        from the pair diagonalised at build: ``eps`` enters the diagonal
        only, so it may be traced, and the block costs two small matmuls
        here, hoisted out of the solve. Both split systems of
        :func:`~mrx.operators.apply_inverse_mass_plus_eps_laplace_matrix`
        use it.
        """
        payload = self._shifted
        core_inv = (payload.core_V / (1.0 + eps * payload.core_mu)) @ payload.core_V.T
        inv_eps = 1.0 / eps

        def apply(x):
            return _apply_shifted_payload(payload, core_inv, inv_eps, jnp.asarray(x))
        return apply


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

    A mass is a single Kronecker product, so the bulk inverse is three 1-D
    solves (:func:`mrx.preconditioners._kron_mass_model_1d`) inside the
    diagonal sandwich ``Lam``; the polar rows are probed through
    ``apply_mass_matrix`` and inverted densely, so there is no pseudoinverse
    of the extraction anywhere.

    Not only a preconditioner: ``apply_laplacian_approx`` uses it as the
    inner inverse of the weak term, so swapping it changes the OPERATOR
    ``L_k`` at k>=1, not just the solve.
    """

    def __init__(self, seq, operators, k, dirichlet):
        from mrx.operators import apply_mass_matrix  # noqa: PLC0415
        from mrx.preconditioners import _kron_mass_model_1d  # noqa: PLC0415

        _, mass_1d, lam = _kron_mass_model_1d(seq, k)
        core, e, tensor_blocks = _tensor_blocks(seq, k, dirichlet)
        self.core = core
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

        size = int(seq.n(k, dirichlet))
        on = _probing_sequence(seq)
        self.core_inv = _dense_symmetric_inverse(_probe_rows(
            lambda x: apply_mass_matrix(on, x, k, dirichlet=dirichlet),
            size, core, dtype=on.dtype), CORE_TOL)
        self._flat = _flatten_payload(self._build_payload())
        self._apply_in = {}

    def _build_payload(self):
        """Pack the factors into the :class:`_MassPayload` pytree, eagerly
        (:meth:`MetricLumpingLaplacian._build_payload`)."""
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

    def apply_in(self, dtype):
        """``apply`` with the payload in ``dtype``: the atom as part of an
        OPERATOR of that precision -- the weak term of the hat Laplacian
        (:func:`mrx.operators._hat_solve`) and of the approximate Laplacian
        the Laplacian atoms' cores are probed with, on the float64 view --
        rather than as a preconditioner, whose own precision is immaterial.
        One cast per dtype, memoised in closures (which the bundle's cast
        to the working dtype leaves alone)."""
        dtype = jnp.dtype(dtype)
        try:
            return self._apply_in[dtype]
        except KeyError:
            leaves, jitted = self._flat
            leaves = tuple(leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
                           for leaf in leaves)
            self._apply_in[dtype] = apply = lambda x: jitted(leaves, jnp.asarray(x, dtype))
            return apply
