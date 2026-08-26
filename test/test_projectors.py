"""Tests for mrx.projectors: L2 projection and Greville/histopolation.

All tests use a shared module-scoped rotating-ellipse polar sequence
(``polar=True``, ``("clamped", "periodic", "periodic")``, ``ns=(6, 6, 6)``).
This is a genuinely 3D polar sequence.  All test functions vanish at the
polar axis r=0 (ξ[0] = 0).

L2 errors are measured in the logical frame using the sequence's own
Gauss quadrature.  For 0-forms and 1-forms the Jacobian weight is included;
for 2- and 3-forms it is not (the projection already absorbs the geometry).

Mathematical properties checked
--------------------------------
* k=0: L2 projection relative error is small.
* k=0: Greville interpolation relative error is small.
* k=0: L2 projection error ≤ interpolation error (best-approximation).
* k=1: L2 projection relative error is small.
* k=1: Histopolation relative error is small.
* k=1: L2 projection error ≤ histopolation error (best-approximation).
* k=2: L2 projection relative error is small.
* k=3: L2 projection relative error is small.

Tolerances are set to ``< 1.0`` (trivially pass) on first commit.
Run with ``-s`` to read the actual errors, then tighten them.
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
    """(6, 6, 6) p=2 polar rotating ellipse: the ACCURACY fixture.

    Resolution is the point of the error tests, so they belong to the ``gpu``
    tier; the identity tests below build (4, 4, 4) sequences of their own.
    """
    seq = DeRhamSequence(
        (6, 6, 6), (2, 2, 2), 4, ("clamped", "periodic", "periodic"),
        polar=True, maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2))
    # projection tests need operators, not preconditioners; skipping the eager payloads avoids the CP/NTF fits
    # and the core-Schur build, which production no longer uses.
    seq.assemble_all_sparse()
    return seq


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
    """Relative physical L2 error for a k-form.

    Computes sqrt( ∫ |Φ_*(ω_h) - f_ref|² J dξ ) / sqrt( ∫ |f_ref|² J dξ )
    where Φ_* is the k-form pushforward from logical to physical space.
    """
    basis = getattr(seq, _BASIS_ATTR[k])
    discrete = DiscreteFunction(dofs, basis, e)
    pushed = Pushforward(discrete, seq.map, k)
    w = seq.jacobian_j * seq.quad.w

    diff_vals = jax.lax.map(
        lambda x: pushed(x) - f_ref(x), seq.quad.x, batch_size=20_000)
    ref_vals = jax.lax.map(f_ref, seq.quad.x, batch_size=20_000)
    num = float(jnp.einsum("qi,qi,q->", diff_vals, diff_vals, w))
    den = float(jnp.einsum("qi,qi,q->", ref_vals, ref_vals, w))
    return (num / max(den, 1e-30)) ** 0.5


# ---------------------------------------------------------------------------
# k=0 tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_k0_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_f0, 0)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 0, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e0, 0, _f0)
    print(f"\n  k=0 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=0 L2 projection error unreasonably large: {err:.3e}"


@pytest.mark.gpu
def test_k0_greville_interpolation_error_is_small(proj_seq):
    dofs = proj_seq.interpolate(_f0, 0)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e0, 0, _f0)
    print(f"\n  k=0 Greville interpolation relative error: {err:.3e}")
    assert err < 1.0, f"k=0 Greville interpolation error unreasonably large: {err:.3e}"


@pytest.mark.gpu
def test_k0_l2_projection_leq_interpolation(proj_seq):
    """L2 projection is best-approximation: its error ≤ interpolation error."""
    dofs_proj = proj_seq.apply_inverse_mass_matrix(proj_seq.load(_f0, 0), 0, dirichlet=False)
    dofs_interp = proj_seq.interpolate(_f0, 0)
    err_proj = _phys_l2_rel_error(proj_seq, dofs_proj, proj_seq.e0, 0, _f0)
    err_interp = _phys_l2_rel_error(proj_seq, dofs_interp, proj_seq.e0, 0, _f0)
    print(f"\n  k=0 proj={err_proj:.3e}  interp={err_interp:.3e}")
    assert err_proj <= err_interp + mrx.eps(100), (
        f"L2 projection error {err_proj:.3e} > interpolation error {err_interp:.3e}"
    )


# ---------------------------------------------------------------------------
# k=1 tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_k1_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_v1, 1)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 1, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e1, 1, _v1)
    print(f"\n  k=1 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=1 L2 projection error unreasonably large: {err:.3e}"


@pytest.mark.gpu
def test_k1_histopolation_error_is_small(proj_seq):
    dofs = proj_seq.interpolate(_v1, 1)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e1, 1, _v1)
    print(f"\n  k=1 histopolation relative error: {err:.3e}")
    assert err < 1.0, f"k=1 histopolation error unreasonably large: {err:.3e}"


@pytest.mark.gpu
def test_k1_l2_projection_leq_histopolation(proj_seq):
    """L2 projection is best-approximation: its error ≤ histopolation error."""
    dofs_proj = proj_seq.apply_inverse_mass_matrix(proj_seq.load(_v1, 1), 1, dirichlet=False)
    dofs_hist = proj_seq.interpolate(_v1, 1)
    err_proj = _phys_l2_rel_error(proj_seq, dofs_proj, proj_seq.e1, 1, _v1)
    err_hist = _phys_l2_rel_error(proj_seq, dofs_hist, proj_seq.e1, 1, _v1)
    print(f"\n  k=1 proj={err_proj:.3e}  hist={err_hist:.3e}")
    assert err_proj <= err_hist + mrx.eps(100), (
        f"L2 projection error {err_proj:.3e} > histopolation error {err_hist:.3e}"
    )


# ---------------------------------------------------------------------------
# k=2 and k=3 L2 projection
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_k2_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_v2, 2)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 2, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e2, 2, _v2)
    print(f"\n  k=2 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=2 L2 projection error unreasonably large: {err:.3e}"


@pytest.mark.gpu
def test_k3_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_f3, 3)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 3, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e3, 3, _f3)
    print(f"\n  k=3 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=3 L2 projection error unreasonably large: {err:.3e}"


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

@pytest.fixture(scope="module", params=[2, 3], ids=["p2", "p3"])
def identity_seq(request):
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
    mesh -- and they are quadratically expensive in it.  The round-trip feeds
    ``lambda x: discrete(x)`` into the quadrature and
    ``DiscreteFunction.__call__`` evaluates ALL ``n`` basis functions per point,
    so the cost is ``O(n^2 q^d)`` with ``d`` the number of histopolated axes.

    On the (6,6,6) fixture that made a single k=2 round-trip run past TEN
    MINUTES and a full pass unable to finish inside a 90-minute job -- and it
    was mistaken for a hang by a separate full-suite gate.  (4,4,4) cuts ``n``
    by ~3.4x, hence the wall time by ~11x, while testing the identical
    identity.  Accuracy tests stay on ``proj_seq``, where resolution is the
    point.
    """
    deg = request.param
    seq = DeRhamSequence(
        (4, 4, 4), (deg,) * 3, deg + 1, ("clamped", "periodic", "periodic"),
        polar=True, maxiter=200, betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2))
    seq.assemble_all_sparse()
    return seq


_ROUNDTRIP_CASES = [
    pytest.param(0, False, id="k0-free"),
    pytest.param(0, True, id="k0-dbc"),
    pytest.param(1, False, id="k1-free"),
    pytest.param(1, True, id="k1-dbc"),
    pytest.param(2, False, id="k2-free"),
    pytest.param(2, True, id="k2-dbc"),
    pytest.param(3, False, id="k3-free"),
    pytest.param(3, True, id="k3-dbc"),
]


@pytest.mark.parametrize("k, dirichlet", _ROUNDTRIP_CASES)
def test_interpolation_reproduces_its_own_space(identity_seq, k, dirichlet):
    """Interpolating a function already in the space returns its own DOFs."""
    proj_seq = identity_seq
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
    print(f"\n  k={k} dirichlet={dirichlet} round-trip relative error: {err:.3e}")
    assert err < IDENT, (
        f"k={k} dirichlet={dirichlet}: interpolation is not a projector onto "
        f"its own space, relative error {err:.3e}. The extraction is then not "
        f"the conforming projection P_Z and needs the explicit local rules of "
        f"arXiv:2505.15996."
    )


@pytest.mark.gpu
def test_k2_histopolation_is_finite(proj_seq):
    """Isolates the polar-axis singularity to the k=1 physical pullback.

    The three degrees differ in exactly one way at the axis:

        k=0  _interpolate_0form     no pullback at all (scalar)
        k=1  _oneform_pullback      inv33(DF(x)) @ v(x)   <- INVERTS DF
        k=2  _histopolate_2form     DF(x).T @ v(x)        <- no inverse

    ``det DF -> 0`` on the polar axis, and the clamped radial Greville points
    include rho = 0 EXACTLY (quadrature never samples the endpoint, which is why
    this class of bug survives quad-point checks -- see the note on det(DF) = 0
    at the outer knot in docs/research).  So only k=1 evaluates an inverse at the
    singular point.

    This asserts FINITENESS rather than accuracy on purpose: with
    ``frame='phys'`` the k=2 histopolation returns coefficients of ``g omega/J``
    rather than ``omega`` (``M_k`` carries the ``g/J`` weight and there is no
    mass solve here to undo it), so an accuracy assertion would fail for a
    reason unrelated to the axis.  Finite-vs-nan is the discriminator.
    """
    dofs = proj_seq.interpolate(_v2, 2)
    n_bad = int(jnp.sum(~jnp.isfinite(dofs)))
    print(f"\n  k=2 histopolation non-finite DOFs: {n_bad} of {dofs.size}")
    assert n_bad == 0, (
        f"k=2 histopolation produced {n_bad} non-finite DOFs. Its pullback has "
        f"no inverse, so if this fails the polar-axis explanation for the k=1 "
        f"nan is WRONG and something shared by both degrees is at fault."
    )


@pytest.mark.gpu
def test_k0_interpolation_is_finite(proj_seq):
    """Companion to the k=2 finiteness test: k=0 has no pullback at all."""
    dofs = proj_seq.interpolate(_f0, 0)
    n_bad = int(jnp.sum(~jnp.isfinite(dofs)))
    print(f"\n  k=0 interpolation non-finite DOFs: {n_bad} of {dofs.size}")
    assert n_bad == 0


@pytest.mark.gpu
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
    seq.evaluate_1d()
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2))
    seq.assemble_all_sparse()
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
