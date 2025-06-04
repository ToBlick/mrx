"""
Unit tests for the derivatives implemented in Differential Forms.
Tests include:
- Musical operators (♭, ♯) in cylindrical and Cartesian coordinates
- Vector calculus identities 
"""

import unittest
from jax import numpy as jnp
import numpy as np
from mrx.DifferentialForms import Pushforward, Pullback, Flat, Sharp
from mrx.Utils import grad as Grad
from mrx.Utils import curl as Curl
from mrx.Utils import div as Div

# Mapping from cylindrical coordinates to Cartesian coordinates

def _X(r, χ):
        """Compute the X coordinate."""
        return jnp.ones(1) *r *jnp.cos(2 * jnp.pi * χ)

def _Y(r, χ):
        """Compute the Y coordinate."""
        return jnp.ones(1) *  r * jnp.sin(2 * jnp.pi * χ)

def F(x):
        """Cylindrical to cartesian mapping function."""
        r, χ, z = x
        return jnp.ravel(jnp.array([_X(r, χ), _Y(r, χ), jnp.ones(1) * z]))


# Mapping from Cartesian coordinates to cylindrical coordinates
def _R(x_1, x_2):
        """Compute the R coordinate."""
        return jnp.ones(1) * jnp.sqrt((x_1**2) +(x_2**2))
def _χ(x_1, x_2):
        """Compute the χ coordinate."""
        return jnp.ones(1) * jnp.arctan2(x_2, x_1) / (2 * jnp.pi)

def G(x):
        """Cartesian to cylindrical mapping function."""
        x_1, x_2, x_3 = x
        return jnp.ravel(jnp.array([_R(x_1, x_2), _χ(x_1, x_2), jnp.ones(1) * x_3]))

# 0-form
def p_hat(ζ):
    """Scalar field in cylindrical coordinates."""
    r, phi, z = ζ
    return r**2 + z**2

def p(x):
    """Scalar field in Cartesian coordinates."""
    x_1, x_2, x_3 = x
    return x_1**2 + x_2**2 + x_3**2

# 1-form
def A_hat(ζ):
    """Vector field in cylindrical coordinates."""
    r, phi, z = ζ
    return jnp.array([r**2, r*phi, z])

def A(x):
    """Vector field in Cartesian coordinates."""
    x_1, x_2, x_3 = x
    r = jnp.sqrt(x_1**2 + x_2**2)
    phi = jnp.arctan2(x_2, x_1)
    return jnp.array([r*jnp.cos(phi), r*jnp.sin(phi), x_3])

#2-form 
def B_hat(ζ):
    """2-form in cylindrical coordinates."""
    r, phi, z = ζ
    return jnp.array([r*jnp.sin(phi), r*jnp.cos(phi), z])

def B(x):
    """2-form in Cartesian coordinates."""
    x_1, x_2, x_3 = x
    return jnp.array([x_2, x_1, x_3])

# Pullback and pushforward operations
def pullback_p_hat(x):
    return Pullback(p_hat, G, 0).__call__(x)

def pushforward_p(ζ):
    return Pushforward(p, F, 0).__call__(ζ)

def pullback_A_hat(x):
    return Pullback(A_hat, G, 1).__call__(x)

def pushforward_A(ζ):
    return Pushforward(A, F, 1).__call__(ζ)

def pullback_B_hat(x):
    return Pullback(B_hat, G, 2).__call__(x)

def pushforward_B(ζ):
    return Pushforward(B, F, 2).__call__(ζ)


# Musical operators

def flat_A(x):
    """Flat operator applied to vector field in cartesian coordinates, returns the same thing since tensor is identity."""
    x_1,x_2,x_3 = x
    return A(x)

def sharp_A(x):
    """Sharp operator applied to vector field in cartesian coordinates, returns the same thing since tensor is identity."""
    x_1, x_2, x_3 = x
    return A(x)
    
def flat_A_hat(ζ):
    """Flat operator applied to A_hat."""
    r, phi, z = ζ
    return Flat(A_hat,G).__call__(ζ)
    
def sharp_A_hat(ζ):
    """Sharp operator applied to A_hat."""
    r, phi, z = ζ
    
    return Sharp(A_hat,G).__call__(ζ)

class TestDerivatives(unittest.TestCase):
    def test_musical_operators(self):
        """Test flat and sharp operators."""

        # Test point in cylindrical coordinates

        ζ_star = jnp.array([1.0, jnp.pi / 4, 2.0])  
        # Test (♭) ∘ (♯) =  I in cylindrical coordinates
        flat_sharp_A_hat = flat_A_hat(sharp_A_hat(ζ_star))
        np.testing.assert_allclose(
            flat_sharp_A_hat,
            A_hat(ζ_star),
            atol=1.0e-5,
            err_msg="Flat composed with sharp returned incorrect map"
        )

        # Test point in cartesian coordinates
        x_star = jnp.array([1.0, 2.0, 3.0])
        # Test (♯) ∘ (♭) = I in Cartesian coordinates
        sharp_flat_A = sharp_A(flat_A(x_star))
        np.testing.assert_allclose(
            sharp_flat_A,
            A(x_star),
            atol=1.0e-5,
            err_msg="Sharp composed with flat returned incorrect map"
        )


    def test_vector_calculus_identities_cartesian(self):
        """Test vector calculus identities in Cartesian coordinates."""


        # Test point in Cartesian coordinates
        x_star = jnp.array([1.0, 2.0, 3.0])
        # Test that the curl of gradient is zero vector
        curl_grad = Curl(Grad(p))(x_star)
        np.testing.assert_allclose(
            curl_grad,
            jnp.zeros(3),
            atol=1.0e-7,
            err_msg="Curl of gradient should be the zero vector"
        )

        # Test divergence of curl is zero
        div_curl = Div(Curl(A))(x_star)
        np.testing.assert_allclose(
            div_curl,
            0.0,
            atol=1.0e-7,
            err_msg="Divergence of curl should be zero"
        )


    def test_vector_calculus_identities_cylindrical(self):
        """Test vector calculus identities in cylindrical coordinates."""

        # Test point in cylindrical coordinates
        ζ_star = jnp.array([1.0, jnp.pi / 4, 2.0])
        # Test curl of gradient is zero in cylindrical coordinates
        curl_grad_hat = Curl(Grad(p_hat))(ζ_star)
        np.testing.assert_allclose(
            curl_grad_hat,
            jnp.zeros(3),
            atol=1.0e-7,
            err_msg="Curl of gradient in cylindrical coordinates should be the zero vector"
        )

        # Test divergence of curl is zero in cylindrical coordinates
        div_curl_hat = Div(Curl(A_hat))(ζ_star)
        np.testing.assert_allclose(
            div_curl_hat,
            0.0,
            atol=1.0e-7,
            err_msg="Divergence of curl in cylindrical coordinates should be zero"
        )

if __name__ == '__main__':
    unittest.main()