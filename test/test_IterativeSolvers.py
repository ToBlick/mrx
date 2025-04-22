from mrx.IterativeSolvers import *

import jax.numpy as jnp

def test_picard_solver():
    # Test with a quadratic function f(z) = cos z
    f = lambda z: jnp.cos(z)
    z_init = jnp.array([-1.0])
    tol = 1e-6
    expected_solution = jnp.array([0.739085133215160641655312087673])  # Dottie number
    z_star = picard_solver(f, z_init, tol)
    assert jnp.isclose(z_star, expected_solution, atol=tol), f"Expected {expected_solution}, got {z_star}"
    z_star = newton_solver(f, z_init, tol)
    assert jnp.isclose(z_star, expected_solution, atol=tol), f"Expected {expected_solution}, got {z_star}"
