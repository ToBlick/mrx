# %%
# test_mappings.py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from mrx.mappings import (
    lcfs_fit,
    get_lcfs_F,
    cerfon_map,
    helical_map,
    rotating_ellipse_map,
    toroid_map,
    polar_map,
    cylinder_map,
    drumshape_map,
)
from mrx.differential_forms import DifferentialForm, DiscreteFunction
from mrx.quadrature import QuadratureRule

jax.config.update("jax_enable_x64", True)


# Tests for toroid_map
def test_toroid_map_basic():
    """Test basic toroid_map functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Test at origin
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"
    npt.assert_allclose(y, jnp.array([2.0, 0.0, 0.0]), atol=1e-10, err_msg="Origin should map to (R0, 0, 0)")


def test_toroid_map_origin():
    """Test toroid_map at origin."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    # At r=0, should be at major radius
    expected = jnp.array([2.0, 0.0, 0.0])
    npt.assert_allclose(y, expected, atol=1e-10, err_msg="Origin should map to major radius")


def test_toroid_map_parameters():
    """Test toroid_map with different parameters."""
    for epsilon in [0.1, 0.5, 1.0]:
        for R0 in [1.0, 2.0, 5.0]:
            F = toroid_map(epsilon=epsilon, R0=R0)
            x = jnp.array([0.0, 0.0, 0.0])
            y = F(x)
            expected = jnp.array([R0, 0.0, 0.0])
            npt.assert_allclose(y, expected, atol=1e-10, 
                              err_msg=f"Failed for epsilon={epsilon}, R0={R0}")


def test_toroid_map_periodicity():
    """Test toroid_map periodicity in z direction."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    x1 = jnp.array([0.5, 0.0, 0.0])
    x2 = jnp.array([0.5, 0.0, 1.0])
    y1 = F(x1)
    y2 = F(x2)
    # Should be periodic in z (ζ) direction
    npt.assert_allclose(y1, y2, atol=1e-10, err_msg="Should be periodic in z direction")


def test_toroid_map_shape():
    """Test toroid_map output shape for multiple points."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    x = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    y = jax.vmap(F)(x)
    assert y.shape == (3, 3), "vmap should produce shape (n_points, 3)"


# Tests for rotating_ellipse_map
def test_rotating_ellipse_map_basic():
    """Test basic rotating_ellipse_map functionality."""
    F = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=3)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"


def test_rotating_ellipse_map_origin():
    """Test rotating_ellipse_map at origin."""
    F = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=3)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    # At r=0, should be at R=1
    expected = jnp.array([1.0, 0.0, 0.0])
    npt.assert_allclose(y, expected, atol=1e-10, err_msg="Origin should map to (1, 0, 0)")


def test_rotating_ellipse_map_parameters():
    """Test rotating_ellipse_map with different parameters."""
    for eps in [0.1, 0.5, 1.0]:
        for kappa in [1.0, 1.2, 2.0]:
            for nfp in [1, 3, 5]:
                F = rotating_ellipse_map(eps=eps, kappa=kappa, nfp=nfp)
                x = jnp.array([0.0, 0.0, 0.0])
                y = F(x)
                assert y.shape == (3,), f"Failed for eps={eps}, kappa={kappa}, nfp={nfp}"


def test_rotating_ellipse_map_nfp_zero():
    """Test rotating_ellipse_map with nfp=0."""
    with pytest.raises(ValueError, match="nfp must be a positive integer, got 0"):
        _ = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=0)
    with pytest.raises(ValueError, match="nfp must be a positive integer, got -1"):
        _ = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=-1)

def test_rotating_ellipse_map_eps_zero():
    """Test rotating_ellipse_map with eps=0."""
    with pytest.raises(ValueError, match="eps must be a positive number, got 0"):
        _ = rotating_ellipse_map(eps=0, kappa=1.2, nfp=3)
    with pytest.raises(ValueError, match="eps must be a positive number, got -0.5"):
        _ = rotating_ellipse_map(eps=-0.5, kappa=1.2, nfp=3)

def test_rotating_ellipse_map_zeta_range():
    """Test that when nfp != 0, the zeta angle has the correct range of values."""
    # When nfp > 0, zeta is divided by nfp to model only one field period
    # So ζ_effective = ζ / nfp, and the effective zeta range is [0, 1/nfp] when input is [0, 1]
    # The mapping is periodic with period nfp in the input zeta: F(r, θ, ζ) ≈ F(r, θ, ζ + nfp)
    for nfp in [1, 3, 5]:
        F = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=nfp)
        
        r, θ = 0.5, 0.25
        
        # Test that F(r, θ, 0) equals F(r, θ, nfp)
        # This verifies that the mapping is periodic with period nfp in the input zeta
        zeta1 = 0.0
        zeta2 = float(nfp)
        
        x1 = jnp.array([r, θ, zeta1])
        x2 = jnp.array([r, θ, zeta2])
        
        y1 = F(x1)
        y2 = F(x2)
        
        # Should be periodic with period nfp
        npt.assert_allclose(y1, y2, atol=1e-6, 
                          err_msg=f"Should be periodic with period nfp for nfp={nfp}")
        
        # Test that the effective zeta range is [0, 1/nfp] when input is [0, 1]
        # When ζ = 0, ζ_effective = 0
        # When ζ = 1, ζ_effective = 1/nfp
        # So F(r, θ, 0) and F(r, θ, 1) should be different (unless nfp=1)
        zeta3 = 1.0
        x3 = jnp.array([r, θ, zeta3])
        y3 = F(x3)
        
        if nfp == 1:
            # When nfp=1, ζ=1 maps to ζ_effective=1, which should equal ζ_effective=0 (periodic)
            npt.assert_allclose(y1, y3, atol=1e-6,
                              err_msg="Should be periodic with period 1 for nfp=1")
        else:
            # When nfp>1, ζ=1 maps to ζ_effective=1/nfp, which is not equal to ζ_effective=0
            assert not jnp.allclose(y1, y3, atol=1e-6), \
                f"Points at zeta=0 and zeta=1 should be different for nfp={nfp} (unless nfp=1)"
        
        # Test that intermediate zeta values produce different results
        # F(r, θ, 0) should be different from F(r, θ, 0.5/nfp)
        zeta4 = 0.5 / nfp
        x4 = jnp.array([r, θ, zeta4])
        y4 = F(x4)
        
        # Should be different from y1 (not at the boundary)
        assert not jnp.allclose(y1, y4, atol=1e-6), \
            f"Points at zeta=0 and zeta=0.5/nfp should be different for nfp={nfp}"


# Tests for cerfon_map
def test_cerfon_map_basic():
    """Test basic cerfon_map functionality."""
    F = cerfon_map(epsilon=0.5, kappa=1.2, alpha=0.1, R0=1.0)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"


def test_cerfon_map_parameters():
    """Test cerfon_map with different parameters."""
    for epsilon in [0.1, 0.5, 1.0]:
        for kappa in [1.0, 1.2, 2.0]:
            for alpha in [0.0, 0.1, 0.5]:
                F = cerfon_map(epsilon=epsilon, kappa=kappa, alpha=alpha, R0=1.0)
                x = jnp.array([0.0, 0.0, 0.0])
                y = F(x)
                assert y.shape == (3,), f"Failed for epsilon={epsilon}, kappa={kappa}, alpha={alpha}"


def test_cerfon_map_periodicity():
    """Test cerfon_map periodicity."""
    F = cerfon_map(epsilon=0.5, kappa=1.2, alpha=0.1, R0=1.0)
    x1 = jnp.array([0.5, 0.0, 0.0])
    x2 = jnp.array([0.5, 1.0, 0.0])
    y1 = F(x1)
    y2 = F(x2)
    # Should be periodic in t direction
    npt.assert_allclose(y1, y2, atol=1e-10, err_msg="Should be periodic in t direction")


# Tests for helical_map
def test_helical_map_basic():
    """Test basic helical_map functionality."""
    F = helical_map(epsilon=0.33, h=0.25, n_turns=3)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"


def test_helical_map_parameters():
    """Test helical_map with different parameters."""
    for epsilon in [0.1, 0.33, 0.5]:
        for h in [0.1, 0.25, 0.5]:
            for n_turns in [1, 3, 5]:
                F = helical_map(epsilon=epsilon, h=h, n_turns=n_turns)
                x = jnp.array([0.0, 0.0, 0.0])
                y = F(x)
                assert y.shape == (3,), f"Failed for epsilon={epsilon}, h={h}, n_turns={n_turns}"


def test_helical_map_origin():
    """Test helical_map at origin."""
    F = helical_map(epsilon=0.33, h=0.25, n_turns=3)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    # At r=0, should be on the helix centerline
    assert y.shape == (3,), "Output should have shape (3,)"


# Tests for polar_map
def test_polar_map_basic():
    """Test basic polar_map functionality."""
    F = polar_map()
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"


def test_polar_map_origin():
    """Test polar_map at origin."""
    F = polar_map()
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    expected = jnp.array([0.0, 0.0, 0.0])
    npt.assert_allclose(y, expected, atol=1e-10, err_msg="Origin should map to origin")


def test_polar_map_periodicity():
    """Test polar_map periodicity in θ direction."""
    F = polar_map()
    x1 = jnp.array([0.5, 0.0, 0.0])
    x2 = jnp.array([0.5, 1.0, 0.0])
    y1 = F(x1)
    y2 = F(x2)
    npt.assert_allclose(y1, y2, atol=1e-10, err_msg="Should be periodic in θ direction")


# Tests for cylinder_map
def test_cylinder_map_basic():
    """Test basic cylinder_map functionality."""
    F = cylinder_map(a=1.0, h=2.0)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"


def test_cylinder_map_origin():
    """Test cylinder_map at origin."""
    F = cylinder_map(a=1.0, h=2.0)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    expected = jnp.array([0.0, 0.0, 0.0])
    npt.assert_allclose(y, expected, atol=1e-10, err_msg="Origin should map to origin")


def test_cylinder_map_parameters():
    """Test cylinder_map with different parameters."""
    for a in [0.5, 1.0, 2.0]:
        for h in [1.0, 2.0, 5.0]:
            F = cylinder_map(a=a, h=h)
            x = jnp.array([0.0, 0.0, 0.0])
            y = F(x)
            expected = jnp.array([0.0, 0.0, 0.0])
            npt.assert_allclose(y, expected, atol=1e-10, 
                              err_msg=f"Failed for a={a}, h={h}")


def test_cylinder_map_radius():
    """Test cylinder_map radius scaling."""
    F = cylinder_map(a=2.0, h=1.0)
    x = jnp.array([1.0, 0.0, 0.0])  # r=1, χ=0
    y = F(x)
    # At r=1, χ=0, should be at radius a
    expected_radius = 2.0
    actual_radius = jnp.sqrt(y[0]**2 + y[1]**2)
    npt.assert_allclose(actual_radius, expected_radius, atol=1e-10, 
                       err_msg="Radius should scale with parameter a")


def test_cylinder_map_height():
    """Test cylinder_map height scaling."""
    F = cylinder_map(a=1.0, h=3.0)
    x = jnp.array([0.0, 0.0, 1.0])  # z=1
    y = F(x)
    # At z=1, should be at height h
    npt.assert_allclose(y[2], 3.0, atol=1e-10, err_msg="Height should scale with parameter h")


# Tests for drumshape_map
def test_drumshape_map_basic():
    """Test basic drumshape_map functionality."""
    def a_h(χ):
        return 1.0 + 0.1 * jnp.cos(2 * jnp.pi * χ)
    
    F = drumshape_map(a_h)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"


def test_drumshape_map_origin():
    """Test drumshape_map at origin."""
    def a_h(χ):
        return 1.0 + 0.1 * jnp.cos(2 * jnp.pi * χ)
    
    F = drumshape_map(a_h)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    expected = jnp.array([0.0, 0.0, 0.0])
    # Actually at r=0, should be (0, 0, 0)
    npt.assert_allclose(y, expected, atol=1e-10, 
                       err_msg="Origin should map to origin")


def test_drumshape_map_with_function():
    """Test drumshape_map with different radius functions."""
    def a_h1(χ):
        return 1.0
    
    def a_h2(χ):
        return 1.0 + 0.5 * jnp.cos(2 * jnp.pi * χ)
    
    F1 = drumshape_map(a_h1)
    F2 = drumshape_map(a_h2)
    
    x = jnp.array([0.5, 0.0, 0.0])
    y1 = F1(x)
    y2 = F2(x)
    
    assert y1.shape == (3,), "Output should have shape (3,)"
    assert y2.shape == (3,), "Output should have shape (3,)"


# Tests for lcfs_fit
def test_lcfs_fit_basic():
    """Test basic lcfs_fit functionality."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    a_hat = lcfs_fit(n_map, p_map, q_map, R0, k0, q0, aR, 
                     atol=1e-4, rtol=1e-4, maxiter=1000)
    
    assert a_hat.shape == (n_map,), "Output should have shape (n_map,)"
    assert jnp.all(jnp.isfinite(a_hat)), "Output should be finite"


def test_lcfs_fit_parameters():
    """Test lcfs_fit with different parameters."""
    for n_map in [3, 5, 8]:
        for p_map in [1, 2]:
            # p_map must be < n_map for valid spline basis
            if p_map < n_map:
                a_hat = lcfs_fit(n_map, p_map, 4, 1.0, 1.0, 1.0, 0.5,
                               atol=1e-3, rtol=1e-3, maxiter=500)
                assert a_hat.shape == (n_map,), f"Failed for n_map={n_map}, p_map={p_map}"
                assert jnp.all(jnp.isfinite(a_hat)), f"Output should be finite for n_map={n_map}, p_map={p_map}"


def test_lcfs_fit_different_R0():
    """Test lcfs_fit with different major radii."""
    n_map, p_map, q_map = 5, 2, 4
    for R0 in [1.0, 2.0, 3.0]:
        a_hat = lcfs_fit(n_map, p_map, q_map, R0, 1.0, 1.0, 0.5,
                        atol=1e-3, rtol=1e-3, maxiter=500)
        assert a_hat.shape == (n_map,), f"Failed for R0={R0}"
        assert jnp.all(jnp.isfinite(a_hat)), f"Output should be finite for R0={R0}"


def test_lcfs_fit_different_aR():
    """Test lcfs_fit with different radial positions."""
    n_map, p_map, q_map = 5, 2, 4
    for aR in [0.3, 0.5, 0.7]:
        a_hat = lcfs_fit(n_map, p_map, q_map, 1.0, 1.0, 1.0, aR,
                        atol=1e-3, rtol=1e-3, maxiter=500)
        assert a_hat.shape == (n_map,), f"Failed for aR={aR}"
        assert jnp.all(jnp.isfinite(a_hat)), f"Output should be finite for aR={aR}"


def test_lcfs_fit_dtype():
    """Test that lcfs_fit returns correct dtype."""
    n_map, p_map, q_map = 5, 2, 4
    a_hat = lcfs_fit(n_map, p_map, q_map, 1.0, 1.0, 1.0, 0.5,
                    atol=1e-4, rtol=1e-4, maxiter=1000)
    assert a_hat.dtype in [jnp.float32, jnp.float64], "Output should be float"


def _compute_lcfs_loss(a_hat, n_map, p_map, q_map, R0, k0, q0, aR):
    """Helper function to compute the LCFS loss function."""
    def psi(R, Z):
        def _psi(R, Z):
            return (k0**2/4*(R**2 - R0**2)**2 + R**2*Z**2) / (2 * R0**2 * k0 * q0)
        return _psi(R, Z) - _psi(R0 + aR, 0)
    
    π = jnp.pi
    Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                             ("periodic", "constant", "constant"))
    Q_map = QuadratureRule(Λ_map, q_map)
    a_h = DiscreteFunction(a_hat, Λ_map)
    
    def _psi_eval(x):
        a = a_h(x)[0]
        χ = x[0]
        R = R0 + a * jnp.cos(2 * π * χ)
        Z = a * jnp.sin(2 * π * χ)
        return psi(R, Z)**2
    return Q_map.w @ jax.vmap(_psi_eval)(Q_map.x)


def _evaluate_psi_on_surface(a_hat, n_map, p_map, R0, k0, q0, aR, χ_values):
    """Helper function to evaluate psi on the fitted surface at given χ values."""
    def psi(R, Z):
        def _psi(R, Z):
            return (k0**2/4*(R**2 - R0**2)**2 + R**2*Z**2) / (2 * R0**2 * k0 * q0)
        return _psi(R, Z) - _psi(R0 + aR, 0)
    
    π = jnp.pi
    Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                             ("periodic", "constant", "constant"))
    a_h = DiscreteFunction(a_hat, Λ_map)
    
    def eval_at_chi(χ):
        x = jnp.array([χ, 0.0, 0.0])
        a = a_h(x)[0]
        R = R0 + a * jnp.cos(2 * π * χ)
        Z = a * jnp.sin(2 * π * χ)
        return psi(R, Z)
    
    return jax.vmap(eval_at_chi)(χ_values)


def test_lcfs_fit_loss_reduction():
    """Test that lcfs_fit reduces the loss function."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    # Use a poor initial guess (much larger than aR)
    a_hat_initial = jnp.ones(n_map) * (aR * 1.5)
    loss_initial = _compute_lcfs_loss(a_hat_initial, n_map, p_map, q_map, R0, k0, q0, aR)
    
    # Optimized solution
    a_hat_final = lcfs_fit(n_map, p_map, q_map, R0, k0, q0, aR,
                          atol=1e-4, rtol=1e-4, maxiter=1000)
    loss_final = _compute_lcfs_loss(a_hat_final, n_map, p_map, q_map, R0, k0, q0, aR)
    
    # Loss should be reduced
    assert loss_final <= loss_initial, (
        f"Loss should be reduced: initial={loss_initial:.2e}, final={loss_final:.2e}"
    )
    # Loss should be significantly reduced (at least 2x)
    assert loss_final < loss_initial / 2, (
        f"Loss should be significantly reduced: initial={loss_initial:.2e}, final={loss_final:.2e}"
    )


def test_lcfs_fit_psi_agreement():
    """Test that the fitted surface agrees with exact psi to some tolerance."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    # Fit the surface
    a_hat = lcfs_fit(n_map, p_map, q_map, R0, k0, q0, aR,
                    atol=1e-4, rtol=1e-4, maxiter=1000)
    
    # Evaluate psi at several points on the fitted surface
    χ_values = jnp.linspace(0.0, 1.0, 20, endpoint=False)
    psi_values = _evaluate_psi_on_surface(a_hat, n_map, p_map, R0, k0, q0, aR, χ_values)
    
    # Psi should be close to 0 (the target value) on the fitted surface
    # Use a reasonable tolerance - the optimization minimizes the squared loss,
    # so individual psi values might not be exactly 0, but should be small
    max_psi_error = jnp.max(jnp.abs(psi_values))
    assert max_psi_error < 0.2, (
        f"Psi should be reasonably close to 0 on fitted surface. Max error: {max_psi_error:.2e}"
    )
    # Most points should have smaller error
    mean_psi_error = jnp.mean(jnp.abs(psi_values))
    assert mean_psi_error < 0.1, (
        f"Mean psi error should be reasonably small. Mean error: {mean_psi_error:.2e}"
    )
    
    # Also check that the loss is small
    loss = _compute_lcfs_loss(a_hat, n_map, p_map, q_map, R0, k0, q0, aR)
    assert loss < 1e-2, (
        f"Loss should be small after optimization. Loss: {loss:.2e}"
    )


def test_lcfs_fit_psi_agreement_strict():
    """Test that the fitted surface agrees with exact psi with stricter tolerance."""
    n_map, p_map, q_map = 8, 3, 6
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    # Fit the surface with stricter tolerances
    a_hat = lcfs_fit(n_map, p_map, q_map, R0, k0, q0, aR,
                    atol=1e-6, rtol=1e-6, maxiter=2000)
    
    # Evaluate psi at many points on the fitted surface
    χ_values = jnp.linspace(0.0, 1.0, 50, endpoint=False)
    psi_values = _evaluate_psi_on_surface(a_hat, n_map, p_map, R0, k0, q0, aR, χ_values)
    
    # With stricter tolerances and more basis functions, psi should be closer to 0
    max_psi_error = jnp.max(jnp.abs(psi_values))
    assert max_psi_error < 0.1, (
        f"With strict tolerances, psi should be closer to 0. Max error: {max_psi_error:.2e}"
    )
    mean_psi_error = jnp.mean(jnp.abs(psi_values))
    assert mean_psi_error < 0.05, (
        f"Mean psi error should be smaller. Mean error: {mean_psi_error:.2e}"
    )
    
    # Also check that the loss is smaller with stricter tolerances
    loss = _compute_lcfs_loss(a_hat, n_map, p_map, q_map, R0, k0, q0, aR)
    assert loss < 2e-3, (
        f"Loss should be smaller with strict tolerances. Loss: {loss:.2e}"
    )


def test_lcfs_fit_loss_reduction_different_params():
    """Test loss reduction for different parameter combinations."""
    n_map, p_map, q_map = 5, 2, 4
    
    for R0 in [1.0, 2.0]:
        for aR in [0.3, 0.5]:
            # Use a poor initial guess (much larger than aR)
            a_hat_initial = jnp.ones(n_map) * (aR * 1.5)
            loss_initial = _compute_lcfs_loss(a_hat_initial, n_map, p_map, q_map, R0, 1.0, 1.0, aR)
            
            # Optimized solution
            a_hat_final = lcfs_fit(n_map, p_map, q_map, R0, 1.0, 1.0, aR,
                                  atol=1e-3, rtol=1e-3, maxiter=500)
            loss_final = _compute_lcfs_loss(a_hat_final, n_map, p_map, q_map, R0, 1.0, 1.0, aR)
            
            # Loss should be reduced
            assert loss_final <= loss_initial, (
                f"Loss should be reduced for R0={R0}, aR={aR}: "
                f"initial={loss_initial:.2e}, final={loss_final:.2e}"
            )
            # Loss should be significantly reduced (at least 1.5x)
            assert loss_final < loss_initial / 1.5, (
                f"Loss should be significantly reduced for R0={R0}, aR={aR}: "
                f"initial={loss_initial:.2e}, final={loss_final:.2e}"
            )


# Tests for get_lcfs_F
def test_get_lcfs_F_basic():
    """Test basic get_lcfs_F functionality."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    F = get_lcfs_F(n_map, p_map, q_map, R0, k0, q0, aR,
                   atol=1e-4, rtol=1e-4, maxiter=1000)
    
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Output should have shape (3,)"


def test_get_lcfs_F_origin():
    """Test get_lcfs_F at origin."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 2.0, 1.0, 1.0, 0.5
    
    F = get_lcfs_F(n_map, p_map, q_map, R0, k0, q0, aR,
                   atol=1e-4, rtol=1e-4, maxiter=1000)
    
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    # At r=0, should be at major radius
    assert y.shape == (3,), "Output should have shape (3,)"
    # At r=0, χ=0, z=0, we expect R ≈ R0, Z ≈ 0
    npt.assert_allclose(y[0]**2 + y[1]**2, R0**2, rtol=1e-1)
    npt.assert_allclose(y[2], 0.0, atol=1e-1)


def test_get_lcfs_F_parameters():
    """Test get_lcfs_F with different parameters."""
    for R0 in [1.0, 2.0]:
        for aR in [0.3, 0.5]:
            F = get_lcfs_F(5, 2, 4, R0, 1.0, 1.0, aR,
                         atol=1e-3, rtol=1e-3, maxiter=500)
            x = jnp.array([0.0, 0.0, 0.0])
            y = F(x)
            assert y.shape == (3,), f"Failed for R0={R0}, aR={aR}"


def test_get_lcfs_F_vmap():
    """Test get_lcfs_F with vmap."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    F = get_lcfs_F(n_map, p_map, q_map, R0, k0, q0, aR,
                   atol=1e-4, rtol=1e-4, maxiter=1000)
    
    # Test with multiple points
    xs = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ys = jax.vmap(F)(xs)
    
    assert ys.shape == (3, 3), "Output should have shape (3, 3)"
    assert jnp.all(jnp.isfinite(ys)), "All outputs should be finite"


def test_get_lcfs_F_continuity():
    """Test that get_lcfs_F is continuous."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    F = get_lcfs_F(n_map, p_map, q_map, R0, k0, q0, aR,
                   atol=1e-4, rtol=1e-4, maxiter=1000)
    
    # Test continuity at r=0 boundary
    x1 = jnp.array([0.0, 0.0, 0.0])
    x2 = jnp.array([1e-6, 0.0, 0.0])
    y1 = F(x1)
    y2 = F(x2)
    
    # Should be close for small r
    npt.assert_allclose(y1, y2, rtol=1e-1, atol=1e-1)


def test_get_lcfs_F_different_R0():
    """Test get_lcfs_F with different major radii."""
    n_map, p_map, q_map = 5, 2, 4
    for R0 in [1.0, 2.0]:
        F = get_lcfs_F(n_map, p_map, q_map, R0, 1.0, 1.0, 0.5,
                      atol=1e-3, rtol=1e-3, maxiter=500)
        x = jnp.array([0.0, 0.0, 0.0])
        y = F(x)
        assert y.shape == (3,), f"Failed for R0={R0}"
        assert jnp.all(jnp.isfinite(y)), f"Output should be finite for R0={R0}"


def test_get_lcfs_F_jacobian():
    """Test that get_lcfs_F has a valid Jacobian."""
    n_map, p_map, q_map = 5, 2, 4
    R0, k0, q0, aR = 1.0, 1.0, 1.0, 0.5
    
    F = get_lcfs_F(n_map, p_map, q_map, R0, k0, q0, aR,
                   atol=1e-4, rtol=1e-4, maxiter=1000)
    
    x = jnp.array([0.5, 0.0, 0.0])
    DF = jax.jacfwd(F)(x)
    
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"
    assert jnp.all(jnp.isfinite(DF)), "Jacobian should be finite"


# Tests for mapping function properties
def test_toroid_map_jacobian():
    """Test that toroid_map produces valid Jacobian."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    x = jnp.array([0.5, 0.5, 0.5])
    
    # Compute Jacobian
    DF = jax.jacfwd(F)(x)
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"


def test_rotating_ellipse_map_jacobian():
    """Test that rotating_ellipse_map produces valid Jacobian."""
    F = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=3)
    x = jnp.array([0.5, 0.5, 0.5])
    
    DF = jax.jacfwd(F)(x)
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"


def test_cerfon_map_jacobian():
    """Test that cerfon_map produces valid Jacobian."""
    F = cerfon_map(epsilon=0.5, kappa=1.2, alpha=0.1, R0=1.0)
    x = jnp.array([0.5, 0.5, 0.5])
    
    DF = jax.jacfwd(F)(x)
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"


def test_helical_map_jacobian():
    """Test that helical_map produces valid Jacobian."""
    F = helical_map(epsilon=0.33, h=0.25, n_turns=3)
    x = jnp.array([0.5, 0.5, 0.5])
    
    DF = jax.jacfwd(F)(x)
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"


def test_polar_map_jacobian():
    """Test that polar_map produces valid Jacobian."""
    F = polar_map()
    x = jnp.array([0.5, 0.5, 0.5])
    
    DF = jax.jacfwd(F)(x)
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"


def test_cylinder_map_jacobian():
    """Test that cylinder_map produces valid Jacobian."""
    F = cylinder_map(a=1.0, h=2.0)
    x = jnp.array([0.5, 0.5, 0.5])
    
    DF = jax.jacfwd(F)(x)
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"


def test_drumshape_map_jacobian():
    """Test that drumshape_map produces valid Jacobian."""
    def a_h(χ):
        return 1.0 + 0.1 * jnp.cos(2 * jnp.pi * χ)
    
    F = drumshape_map(a_h)
    x = jnp.array([0.5, 0.5, 0.5])
    
    DF = jax.jacfwd(F)(x)
    assert DF.shape == (3, 3), "Jacobian should have shape (3, 3)"


# Tests for mapping function continuity
def test_toroid_map_continuity():
    """Test that toroid_map is continuous."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Test continuity at boundaries
    x1 = jnp.array([0.0, 0.0, 0.0])
    x2 = jnp.array([0.0, 0.0, 1.0])
    y1 = F(x1)
    y2 = F(x2)
    # Should be continuous (periodic)
    npt.assert_allclose(y1, y2, atol=1e-10, err_msg="Should be continuous at z=0 and z=1")


def test_rotating_ellipse_map_continuity():
    """Test that rotating_ellipse_map is continuous."""
    F = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=3)
    
    x1 = jnp.array([0.5, 0.0, 0.0])
    x2 = jnp.array([0.5, 1.0, 0.0])
    y1 = F(x1)
    y2 = F(x2)
    # Should be periodic in θ
    npt.assert_allclose(y1, y2, atol=1e-10, err_msg="Should be periodic in θ direction")


# Tests for mapping function with vmap
def test_toroid_map_vmap():
    """Test toroid_map with vmap."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    x = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    y = jax.vmap(F)(x)
    assert y.shape == (3, 3), "vmap should produce shape (n_points, 3)"


def test_rotating_ellipse_map_vmap():
    """Test rotating_ellipse_map with vmap."""
    F = rotating_ellipse_map(eps=0.5, kappa=1.2, nfp=3)
    x = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    y = jax.vmap(F)(x)
    assert y.shape == (2, 3), "vmap should produce shape (n_points, 3)"


# Tests for edge cases
def test_toroid_map_boundary():
    """Test toroid_map at boundaries."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Test at r=1 boundary
    x = jnp.array([1.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Should work at r=1 boundary"
    
    # Test at r=0
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Should work at r=0"


def test_cylinder_map_boundary():
    """Test cylinder_map at boundaries."""
    F = cylinder_map(a=1.0, h=2.0)
    
    # Test at r=1
    x = jnp.array([1.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), "Should work at r=1"
    
    # Test at z=1
    x = jnp.array([0.0, 0.0, 1.0])
    y = F(x)
    assert y.shape == (3,), "Should work at z=1"


# Tests for parameter validation
@pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("R0", [1.0, 2.0, 5.0])
def test_toroid_map_parametrized(epsilon, R0):
    """Test toroid_map with parametrized inputs."""
    F = toroid_map(epsilon=epsilon, R0=R0)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    expected = jnp.array([R0, 0.0, 0.0])
    npt.assert_allclose(y, expected, atol=1e-10,
                       err_msg=f"Failed for epsilon={epsilon}, R0={R0}")


@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("kappa", [1.0, 1.2, 2.0])
@pytest.mark.parametrize("nfp", [1, 3, 5])
def test_rotating_ellipse_map_parametrized(eps, kappa, nfp):
    """Test rotating_ellipse_map with parametrized inputs."""
    F = rotating_ellipse_map(eps=eps, kappa=kappa, nfp=nfp)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    assert y.shape == (3,), f"Failed for eps={eps}, kappa={kappa}, nfp={nfp}"


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("h", [1.0, 2.0, 5.0])
def test_cylinder_map_parametrized(a, h):
    """Test cylinder_map with parametrized inputs."""
    F = cylinder_map(a=a, h=h)
    x = jnp.array([0.0, 0.0, 0.0])
    y = F(x)
    expected = jnp.array([0.0, 0.0, 0.0])
    npt.assert_allclose(y, expected, atol=1e-10, err_msg=f"Failed for a={a}, h={h}")

