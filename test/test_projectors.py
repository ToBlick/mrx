"""Tests for mrx.projectors: L2 projection and Greville/histopolation.

All tests use a shared module-scoped rotating-ellipse polar sequence
(``polar=True``, ``("clamped", "periodic", "periodic")``, ``ns=(4, 4, 4)``,
p=2; the odd-parity identity tests build the p=3 twin). This is a genuinely
3D polar sequence.  All test functions vanish at the polar axis r=0
(ξ[0] = 0).

L2 errors are measured in the logical frame using the sequence's own
Gauss quadrature.  For 0-forms and 1-forms the Jacobian weight is included;
for 2- and 3-forms it is not (the projection already absorbs the geometry).

Mathematical properties checked
--------------------------------
* k=0, 1: the L2 projection and the Greville interpolation / histopolation
  errors are below MEASURED bands, and the projection is the best
  approximation (its error ≤ the interpolation's).
* k=2, 3: the L2 projection error is below a measured band.
"""

import jax
import jax.numpy as jnp
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import rotating_ellipse_map
from mrx.projectors import _oneform_pullback, _twoform_pullback


# ---------------------------------------------------------------------------
# Shared module-scoped polar sequence on a rotating-ellipse torus.
# Genuinely 3D: clamped × periodic × periodic, ns=(6,6,6).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def proj_seq():
    """(4, 4, 4) p=2 polar rotating ellipse: the accuracy, finiteness, pullback
    and load tests, and the even-p identity tests, on ONE sequence -- the
    first ``interpolate`` per (sequence, k) pays the histopolation setup
    (8-17 s on the CPU), so every k is set up once here."""
    return _build_identity_seq(2)


# ---------------------------------------------------------------------------
# Analytic test functions.  All vanish at ξ[0] = 0 (polar axis r = 0).
# All arguments are logical coordinates ξ ∈ [0,1]^3.
# ---------------------------------------------------------------------------

def _f0(xi):
    """Smooth scalar 0-form, vanishes at r=0."""
    return xi[0] * jnp.sin(2 * jnp.pi * xi[1]) * jnp.cos(2 * jnp.pi * xi[2]) * jnp.ones(1)


def _v1(xi):
    """Smooth vector field for 1-form projection (logical frame), vanishes at r=0."""
    return jnp.array([
        xi[0] * jnp.sin(2 * jnp.pi * xi[1]),
        xi[0] * jnp.cos(2 * jnp.pi * xi[2]),
        xi[0] * jnp.sin(2 * jnp.pi * xi[1] + 2 * jnp.pi * xi[2]),
    ])


def _v2(xi):
    """Smooth vector field for 2-form projection (logical frame), vanishes at r=0."""
    return jnp.array([
        xi[0] * jnp.cos(2 * jnp.pi * xi[1]) * jnp.sin(2 * jnp.pi * xi[2]),
        xi[0] * jnp.sin(2 * jnp.pi * xi[1]),
        xi[0] * jnp.cos(2 * jnp.pi * xi[2]),
    ])


def _f3(xi):
    """Smooth scalar 3-form (volume density), vanishes at r=0."""
    return xi[0] * jnp.cos(2 * jnp.pi * xi[1]) * jnp.ones(1)


# ---------------------------------------------------------------------------
# Unified physical-space L2 error helper.
#
# Uses Pushforward to map the discrete form to physical xyz components at
# each logical quadrature point, then compares against f_ref (which is
# defined as a function of logical ξ returning physical xyz components).
# Weight = jacobian_j * quad.w for all k  (physical volume measure).
# ---------------------------------------------------------------------------

_BASIS_ATTR = {0: "basis_0", 1: "basis_1", 2: "basis_2", 3: "basis_3"}

# Exact identities (a projector applied to its own range, the pullback
# inverting the pushforward) hold to the roundoff of the collocation and
# histopolation solves: 1e4 eps = 2.2e-12 f64 / 1.2e-3 f32.
IDENT = mrx.eps(1e4)


def _phys_l2_rel_error(seq, dofs, e, k, f_ref):
    """Relative physical L2 error of a k-form.

    ``sqrt(int |F_* omega_h - f_ref|^2 J) / sqrt(int |f_ref|^2 J)``, with the
    pushforward ``F_*`` of ``differential_forms.Pushforward`` (k=1
    ``DF^-T omega``, k=2 ``DF omega / J``, k=3 ``omega / J``) applied at the
    quadrature points to the tensor-product evaluation of the form --
    ``DiscreteFunction`` evaluates every basis function per point and cost
    30 s of CPU per run here for the same number.
    """
    from mrx.quadrature import evaluate_at_xq

    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    if k in (0, 3):
        comp_info, comp_shapes = seq._form_comp_info(k)
        vals = evaluate_at_xq(e.T @ dofs, comp_info, comp_shapes, quad_shape, 1)
    else:
        vals = seq.evaluate_at_quadrature(dofs, k, dirichlet=False)
    J = seq.jacobian_j
    if k == 1:
        DF = jax.vmap(jax.jacfwd(seq.map))(seq.quad.x)
        vals = jnp.linalg.solve(jnp.swapaxes(DF, 1, 2), vals[..., None])[..., 0]
    elif k == 2:
        DF = jax.vmap(jax.jacfwd(seq.map))(seq.quad.x)
        vals = jnp.einsum("qij,qj->qi", DF, vals) / J[:, None]
    elif k == 3:
        vals = vals / J[:, None]
    w = J * seq.quad.w
    ref_vals = jax.vmap(f_ref)(seq.quad.x)
    diff_vals = vals - ref_vals
    num = float(jnp.einsum("qi,qi,q->", diff_vals, diff_vals, w))
    den = float(jnp.einsum("qi,qi,q->", ref_vals, ref_vals, w))
    return (num / den) ** 0.5


# ---------------------------------------------------------------------------
# Accuracy at (4, 4, 4) p=2
#
# Relative physical-L2 errors measured 2026-08-26 in float64 (see the
# prints): projection 5.024e-2 / 4.537e-1 / 4.249e-1 / 1.563e-1 for
# k = 0..3, Greville interpolation 7.119e-2 (k=0), histopolation 4.649e-1
# (k=1); the bands are 1.25x the measurement. These are approximation errors of the p=2 space, orders
# above the float32 roundoff, so the bands hold in either precision. A
# wrong pullback, a wrong extraction row or a missing metric factor moves
# an error by a factor, not by percent.
# ---------------------------------------------------------------------------

PROJ_BAND = {0: 6.3e-2, 1: 5.7e-1, 2: 5.3e-1, 3: 1.95e-1}
INTERP_BAND = {0: 8.9e-2, 1: 5.8e-1}

_ACCURACY = {0: (_f0, "e0"), 1: (_v1, "e1"), 2: (_v2, "e2"), 3: (_f3, "e3")}


def _projection_error(seq, k):
    f, e_name = _ACCURACY[k]
    dofs = seq.apply_inverse_mass_matrix(seq.load(f, k), k, dirichlet=False)
    return _phys_l2_rel_error(seq, dofs, getattr(seq, e_name), k, f)


@pytest.mark.parametrize("k", [0, 1])
def test_l2_projection_is_the_best_approximation(proj_seq, k):
    """Both errors below their measured bands, and projection ≤ interpolation
    (k=0 Greville collocation, k=1 histopolation)."""
    f, e_name = _ACCURACY[k]
    err_proj = _projection_error(proj_seq, k)
    err_interp = _phys_l2_rel_error(
        proj_seq, proj_seq.interpolate(f, k), getattr(proj_seq, e_name), k, f)
    print(f"\n  k={k} L2 projection {err_proj:.3e}  interpolation {err_interp:.3e}")
    assert err_proj < PROJ_BAND[k]
    assert err_interp < INTERP_BAND[k]
    assert err_proj <= err_interp + mrx.eps(100), (
        f"L2 projection error {err_proj:.3e} > interpolation error {err_interp:.3e}"
    )


@pytest.mark.parametrize("k", [2, 3])
def test_l2_projection_error(proj_seq, k):
    err = _projection_error(proj_seq, k)
    print(f"\n  k={k} L2 projection relative error: {err:.3e}")
    assert err < PROJ_BAND[k]


# ---------------------------------------------------------------------------
# The property that makes "histopolate on the full tensor space, then restrict
# with one extraction apply" legitimate.
#
# That composition is Pi_Z = P_Z . Pi_W of Guclu & Campos Pinto
# (arXiv:2505.15996).  For it to be a PROJECTOR, interpolating a function that
# ALREADY lives in the target space must return that function's own DOFs:
#
#     e @ Pi_full(e^T a)  ==  a
#
# This is the test that decides whether MRX's own extraction is the conforming
# projection, on the polar sequence and on the Dirichlet subspace -- exactly
# the two cases the removed guard used to reject.
#
# The function must be wrapped in a plain lambda: handing a DiscreteFunction
# straight to interpolate() short-circuits through _matching_discrete_dofs and
# returns the DOFs untouched, which would pass without testing anything.
# ---------------------------------------------------------------------------

def _build_identity_seq(deg):
    """Small polar sequence for the EXACT-IDENTITY tests, at EVEN and ODD p.

    The degree is parametrised because it is the discriminating variable for
    histopolation exactness, not an incidental setting.

    A Greville point of degree ``p`` is the mean of ``p`` consecutive knots.  On
    a uniform knot vector that lands ON a knot for ODD p and exactly HALFWAY
    BETWEEN two knots for EVEN p (measured offset 0.500h for p = 2, 4, 6 and
    0.000h for p = 1, 3, 5).  So for even p EVERY Greville histopolation span
    straddles an interior knot -- and a Gauss rule spanning a knot integrates
    across a derivative jump, which is not exact at any quadrature order,
    because a spline is only PIECEWISE polynomial.

    That straddling turned out to be a moment-ACCURACY defect only (fixed by
    splitting spans at knots).  The parity effect on the IDENTITY was a second
    consequence of the same half-knot offset: on a PERIODIC axis the last
    sorted span is ``[1 - h/2, 1 + h/2]`` and crosses the seam, and
    ``histopolation_matrix`` evaluated the basis unwrapped there, where
    ``SplineBasis.evaluate`` is not periodic.  The moments wrapped their points,
    so H and the moments disagreed on the integrand at even p only.  Keeping
    both degrees in the suite means a fix has to hold for both rather than
    being tuned to one.

    Idempotency holds at any resolution, so these tests do not need a fine
    mesh -- and they are quadratically expensive in it.  (The accuracy tests
    used to run on a (6, 6, 6) twin; its errors are the same to within the
    band and the second sequence cost 45 s of CPU per run.)  The round-trip feeds
    ``lambda x: discrete(x)`` into the quadrature and
    ``DiscreteFunction.__call__`` evaluates ALL ``n`` basis functions per point,
    so the cost is ``O(n^2 q^d)`` with ``d`` the number of histopolated axes.

    On the (6,6,6) fixture that made a single k=2 round-trip run past TEN
    MINUTES and a full pass unable to finish inside a 90-minute job -- and it
    was mistaken for a hang by a separate full-suite gate.  (4,4,4) cuts ``n``
    by ~3.4x, hence the wall time by ~11x, while testing the identical
    identity.
    """
    seq = DeRhamSequence(
        (4, 4, 4), (deg,) * 3, deg + 1, ("clamped", "periodic", "periodic"),
        polar=True, maxiter=200, betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2))
    return seq


@pytest.fixture(scope="module")
def identity_seq_p2(proj_seq):
    return proj_seq


@pytest.fixture(scope="module")
def identity_seq_p3():
    return _build_identity_seq(3)


# Every (k, BC) pair at even p; at odd p only k = 1, 2. The parity effect
# lives in the periodic histopolation seam, which k=0 (pure collocation)
# never touches, and k=3 (every axis histopolated) exercises no differently
# from k=1 and k=2 together -- at more than twice their cost.
# Odd p at k=1 only: it is the histopolated PERIODIC axis that carries the
# parity effect, and every k=1 component has one; the k=2 case at p=3 cost
# 9 s of CPU for the same seam.
_ROUNDTRIP_CASES = [
    pytest.param(2, k, d, id=f"p2-k{k}-{'dbc' if d else 'free'}")
    for k in (0, 1, 2, 3) for d in (False, True)
] + [
    pytest.param(3, 1, d, id=f"p3-k1-{'dbc' if d else 'free'}")
    for d in (False, True)
]


@pytest.mark.parametrize("p, k, dirichlet", _ROUNDTRIP_CASES)
def test_interpolation_reproduces_its_own_space(request, p, k, dirichlet):
    """Interpolating a function already in the space returns its own DOFs."""
    proj_seq = request.getfixturevalue(f"identity_seq_p{p}")
    basis = getattr(proj_seq, _BASIS_ATTR[k])
    e = getattr(proj_seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    n = int(getattr(proj_seq, f"n{k}_dbc" if dirichlet else f"n{k}"))

    a = jax.random.normal(jax.random.PRNGKey(11 * k + dirichlet), (n,))
    discrete = DiscreteFunction(a, basis, e)
    # plain lambda: defeats the _matching_discrete_dofs short-circuit
    kwargs = {} if k in (0, 3) else {"frame": "ref"}
    got = proj_seq.interpolate(
        lambda x: discrete(x), k, dirichlet=dirichlet, **kwargs)

    err = float(jnp.linalg.norm(got - a) / jnp.linalg.norm(a))
    print(f"\n  p={p} k={k} dirichlet={dirichlet} round-trip relative error: {err:.3e}")
    assert err < IDENT, (
        f"k={k} dirichlet={dirichlet}: interpolation is not a projector onto "
        f"its own space, relative error {err:.3e}. The extraction is then not "
        f"the conforming projection P_Z and needs the explicit local rules of "
        f"arXiv:2505.15996."
    )


# The finiteness of the k=0 and k=2 interpolations at the polar axis (the
# clamped Greville points include rho = 0 exactly; only the k=1 pullback
# inverts DF there) is implied by the round trips above passing on the same
# sequence, so it is not asserted on its own.


@pytest.mark.parametrize("k", [1, 2])
def test_phys_pullback_inverts_pushforward(proj_seq, k):
    """interpolate's physical pullback must be the INVERSE of Pushforward.

    Pushforward (differential_forms.py:301) is the authority on the convention:

        k=1  F_* omega = (DF^T)^-1 omega   =>   omega = DF^T   v_phys
        k=2  F_* omega = DF omega / J      =>   omega = adj(DF) v_phys

    Before 2026-08-25 the histopolation paths used load's DUAL pullbacks instead
    -- DF^-1 at k=1 (that is the k=-1 VECTOR-FIELD rule, off by G^-1) and DF^T at
    k=2 (off by g/J).  load is right to use them: it builds a dual vector and
    M_k^{-1} converts back to primal.  Histopolation has no mass solve, so
    reusing them silently returned the wrong object -- and it PASSED every
    structural gate, because being off by a metric factor is still finite,
    smooth and divergence-free-looking.

    Tested on the pullbacks directly, at INTERIOR points only, on purpose:
    routing this through histopolation would conflate the convention with the
    polar-axis singularity, since Pushforward at k=1 is itself unbounded at
    rho = 0 where det DF -> 0.
    """
    basis = getattr(proj_seq, _BASIS_ATTR[k])
    e = getattr(proj_seq, f"e{k}")
    n = int(getattr(proj_seq, f"n{k}"))
    a = jax.random.normal(jax.random.PRNGKey(7 * k), (n,))
    omega = DiscreteFunction(a, basis, e)
    pushed = Pushforward(omega, proj_seq.map, k)
    pull = (_oneform_pullback if k == 1 else _twoform_pullback)(proj_seq, pushed)

    # interior only: rho in [0.2, 0.9], away from the degenerate axis
    key = jax.random.PRNGKey(101)
    pts = jax.random.uniform(key, (24, 3))
    pts = pts.at[:, 0].set(0.2 + 0.7 * pts[:, 0])

    got = jax.vmap(pull)(pts)
    want = jax.vmap(omega)(pts)
    err = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
    print(f"\n  k={k} phys pullback vs Pushforward inverse: {err:.3e}")
    assert err < IDENT, (
        f"k={k} physical pullback does not invert Pushforward (rel {err:.3e}). "
        f"Expected omega = {'DF^T v' if k == 1 else 'adj(DF) v'}."
    )


# ---------------------------------------------------------------------------
# Pi_full in ISOLATION.
#
# The restriction (E E^T)^-1 E makes the round-trip exact BY CONSTRUCTION -- but
# only if Pi_full, the tensor-product projector it composes with, is itself
# idempotent.  That is a separate claim and deserves a separate test.
#
# A NON-POLAR, NON-DIRICHLET sequence has BoundaryOperator(('none',)*3), so its
# extraction is the identity and `interpolate` IS Pi_full with nothing else in
# the loop.  If these fail while the polar round-trips pass, the fault is the
# tensor-product projector; if they pass while polar fails, it is the extraction.
#
# The degrees exercise different axes, so a failure localises:
#   k=0  coll_r, coll_t, coll_z              pure collocation, no histopolation
#   k=1  hist_r / hist_t / hist_z            one histopolated axis per component
#   k=2  two histopolated axes per component
#   k=3  hist_r, hist_t, hist_z              all three
# theta and zeta are PERIODIC here, so k>=1 exercises periodic histopolation --
# which `interpolate` only reaches at all because a clamped-only guard was
# removed on the grounds that greville_spans supports periodic spans.  Producing
# spans is a weaker claim than the resulting DOFs being unisolvent.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tensor_seq():
    """Full tensor-product sequence: no polar surgery, no Dirichlet rows."""
    seq = DeRhamSequence(
        (4, 4, 4), (2, 2, 2), 4, ("clamped", "periodic", "periodic"),
        polar=False, maxiter=200, betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2))
    return seq


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_pi_full_is_idempotent(tensor_seq, k):
    """Pi_full must reproduce a function already in the full tensor space."""
    basis = getattr(tensor_seq, _BASIS_ATTR[k])
    e = getattr(tensor_seq, f"e{k}")
    n = int(getattr(tensor_seq, f"n{k}"))
    assert n == int(basis.n), (
        f"k={k}: expected an identity extraction on a non-polar free sequence, "
        f"got {n} extracted of {int(basis.n)} raw -- this test no longer "
        f"isolates Pi_full"
    )

    a = jax.random.normal(jax.random.PRNGKey(23 * k + 5), (n,))
    discrete = DiscreteFunction(a, basis, e)
    kwargs = {} if k in (0, 3) else {"frame": "ref"}
    got = tensor_seq.interpolate(lambda x: discrete(x), k, **kwargs)

    err = float(jnp.linalg.norm(got - a) / jnp.linalg.norm(a))
    print(f"\n  k={k} Pi_full idempotency relative error: {err:.3e}")
    assert err < IDENT, (
        f"k={k}: Pi_full is not idempotent (rel {err:.3e}) on the FULL tensor "
        f"space, with no extraction involved. No restriction can repair this."
    )


# ---------------------------------------------------------------------------
# Physical-frame loads.
#
# ``load(frame='phys')`` and ``io.load_grid_field(frame='phys')`` are the only
# consumers of the raw map Jacobian, which the geometry does not store; both
# recompute ``DF`` at the quadrature points on demand.  These tests pin the two
# entry points against the reference-frame path, which never touches ``DF``:
# the dual pullbacks are ``DF^-1 v`` at k=1 and ``DF^T v`` at k=2, so loading
# the pulled-back field with ``frame='ref'`` must give the same dual vector.
# ---------------------------------------------------------------------------

def _v_phys(xi):
    """Smooth physical vector field; no axis condition is needed (quadrature
    never samples r=0 and the pullbacks are evaluated at quadrature only)."""
    return jnp.array([
        0.3 + xi[0] ** 2 * jnp.cos(2 * jnp.pi * xi[1]),
        xi[0] * jnp.sin(2 * jnp.pi * xi[2]),
        0.5 - xi[0] ** 3,
    ])


def _pulled_back(seq, k):
    """``f_ref(xi)`` such that ``load(f_ref, k, 'ref') == load(_v_phys, k, 'phys')``."""
    from mrx.differential_forms import inv33

    DF = jax.jacfwd(seq.map)
    if k == 1:
        return lambda xi: inv33(DF(xi)) @ _v_phys(xi)
    return lambda xi: DF(xi).T @ _v_phys(xi)


@pytest.mark.parametrize("k", [1, 2])
def test_load_phys_frame_matches_ref_frame_of_pulled_back_field(proj_seq, k):
    dual_phys = proj_seq.load(_v_phys, k, frame='phys')
    dual_ref = proj_seq.load(_pulled_back(proj_seq, k), k, frame='ref')
    err = float(jnp.linalg.norm(dual_phys - dual_ref) / jnp.linalg.norm(dual_ref))
    print(f"\n  k={k} load phys-vs-ref relative difference: {err:.3e}")
    assert err < mrx.eps(1e4), (
        f"k={k}: load(frame='phys') disagrees with load(frame='ref') of the "
        f"pulled-back field (rel {err:.3e}); the on-demand DF pullback is wrong."
    )


@pytest.mark.parametrize("k", [1, 2])
def test_load_grid_field_phys_frame_matches_pointwise_load(proj_seq, k):
    """Grid-sampled physical data must load like the analytic field.

    The sampled field is a cubic in r and constant in the angles, which the
    degree-3 interpolatory fit reproduces exactly, so the only thing left to
    differ is the pullback -- done by ``load_grid_field`` from ``DF`` at the
    quadrature points, exactly as ``load`` does it.
    """
    from mrx.differential_forms import DifferentialForm
    from mrx.io import load_grid_field

    def v(xi):
        return jnp.array([0.3 + xi[0] ** 2, 0.1 * xi[0], 0.5 - xi[0] ** 3])

    n = (8, 6, 6)
    fit = DifferentialForm(0, n, (3, 3, 3), proj_seq.basis_0.types)
    axes = (fit.Λ[0].greville_points(),
            jnp.linspace(0.0, 1.0, n[1], endpoint=False),
            jnp.linspace(0.0, 1.0, n[2], endpoint=False))
    grid = jnp.stack(jnp.meshgrid(*axes, indexing='ij'), axis=-1)   # (n1,n2,n3,3)
    values = jax.vmap(v)(grid.reshape(-1, 3)).reshape(*n, 3)

    dual_grid = load_grid_field(axes, values, proj_seq, k, frame='phys')
    dual_load = proj_seq.load(v, k, frame='phys')
    err = float(jnp.linalg.norm(dual_grid - dual_load) / jnp.linalg.norm(dual_load))
    print(f"\n  k={k} load_grid_field phys vs load phys relative difference: {err:.3e}")
    assert err < mrx.eps(1e5), (
        f"k={k}: load_grid_field(frame='phys') disagrees with load(frame='phys') "
        f"on a field the fit reproduces exactly (rel {err:.3e})."
    )
