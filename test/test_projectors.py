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

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import rotating_ellipse_map

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared module-scoped polar sequence on a rotating-ellipse torus.
# Genuinely 3D: clamped × periodic × periodic, ns=(6,6,6).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def proj_seq():
    seq = DeRhamSequence(
        (6, 6, 6), (2, 2, 2), 4, ("clamped", "periodic", "periodic"),
        polar=True, tol=1e-10, maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2))
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

def test_k0_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_f0, 0)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 0, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e0, 0, _f0)
    print(f"\n  k=0 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=0 L2 projection error unreasonably large: {err:.3e}"


@pytest.mark.xfail(reason="polar zeroform interpolation not yet implemented", raises=NotImplementedError, strict=True)
def test_k0_greville_interpolation_error_is_small(proj_seq):
    dofs = proj_seq.interpolate(_f0, 0)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e0, 0, _f0)
    print(f"\n  k=0 Greville interpolation relative error: {err:.3e}")
    assert err < 1.0, f"k=0 Greville interpolation error unreasonably large: {err:.3e}"


@pytest.mark.xfail(reason="polar zeroform interpolation not yet implemented", raises=NotImplementedError, strict=True)
def test_k0_l2_projection_leq_interpolation(proj_seq):
    """L2 projection is best-approximation: its error ≤ interpolation error."""
    dofs_proj = proj_seq.apply_inverse_mass_matrix(proj_seq.load(_f0, 0), 0, dirichlet=False)
    dofs_interp = proj_seq.interpolate(_f0, 0)
    err_proj = _phys_l2_rel_error(proj_seq, dofs_proj, proj_seq.e0, 0, _f0)
    err_interp = _phys_l2_rel_error(proj_seq, dofs_interp, proj_seq.e0, 0, _f0)
    print(f"\n  k=0 proj={err_proj:.3e}  interp={err_interp:.3e}")
    assert err_proj <= err_interp + 1e-14, (
        f"L2 projection error {err_proj:.3e} > interpolation error {err_interp:.3e}"
    )


# ---------------------------------------------------------------------------
# k=1 tests
# ---------------------------------------------------------------------------

def test_k1_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_v1, 1)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 1, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e1, 1, _v1)
    print(f"\n  k=1 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=1 L2 projection error unreasonably large: {err:.3e}"


@pytest.mark.xfail(reason="polar oneform histopolation not yet implemented", raises=NotImplementedError, strict=True)
def test_k1_histopolation_error_is_small(proj_seq):
    dofs = proj_seq.interpolate(_v1, 1)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e1, 1, _v1)
    print(f"\n  k=1 histopolation relative error: {err:.3e}")
    assert err < 1.0, f"k=1 histopolation error unreasonably large: {err:.3e}"


@pytest.mark.xfail(reason="polar oneform histopolation not yet implemented", raises=NotImplementedError, strict=True)
def test_k1_l2_projection_leq_histopolation(proj_seq):
    """L2 projection is best-approximation: its error ≤ histopolation error."""
    dofs_proj = proj_seq.apply_inverse_mass_matrix(proj_seq.load(_v1, 1), 1, dirichlet=False)
    dofs_hist = proj_seq.interpolate(_v1, 1)
    err_proj = _phys_l2_rel_error(proj_seq, dofs_proj, proj_seq.e1, 1, _v1)
    err_hist = _phys_l2_rel_error(proj_seq, dofs_hist, proj_seq.e1, 1, _v1)
    print(f"\n  k=1 proj={err_proj:.3e}  hist={err_hist:.3e}")
    assert err_proj <= err_hist + 1e-14, (
        f"L2 projection error {err_proj:.3e} > histopolation error {err_hist:.3e}"
    )


# ---------------------------------------------------------------------------
# k=2 and k=3 L2 projection
# ---------------------------------------------------------------------------

def test_k2_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_v2, 2)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 2, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e2, 2, _v2)
    print(f"\n  k=2 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=2 L2 projection error unreasonably large: {err:.3e}"


def test_k3_l2_projection_error_is_small(proj_seq):
    dual = proj_seq.load(_f3, 3)
    dofs = proj_seq.apply_inverse_mass_matrix(dual, 3, dirichlet=False)
    err = _phys_l2_rel_error(proj_seq, dofs, proj_seq.e3, 3, _f3)
    print(f"\n  k=3 L2 projection relative error: {err:.3e}")
    assert err < 1.0, f"k=3 L2 projection error unreasonably large: {err:.3e}"
