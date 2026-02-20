"""
Testing that the numerical solution of the cylinder vector Poisson problem converges to the exact solution with reasonable error.
"""
import sys
sys.path.insert(0, '/Users/juliannestratton/mrx')

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import cylinder_map

jax.config.update("jax_enable_x64", True)


def test_cylinder_poisson():
    """
    Testing the cylinder Poisson problem with n=6, p=2 and asserts error is below 2e-1.
    """
    n = 6
    p = 2
    error = get_err(n, p)

    error_val = float(error)

    assert error_val < 2e-1, f"Error {error_val:.2e} exceeds 2e-1."

    return error_val


def get_err(n, p):
    # Set up finite element spaces 
    q = 2 * p
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("clamped", "periodic", "periodic")

    # Domain parameters
    a = 1
    h = 1
    π = jnp.pi
    F = cylinder_map(a=a, h=h)

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        u_theta = r**2 * (1 - r)**2 * jnp.cos(2*π*z)
        return jnp.array([0, u_theta, 0])

    def f(x):
        """Source term of the Poisson problem."""
        r, χ, z = x
        f_theta = (r**2 * (1 - r)**2 * 4*π**2 - (3 - 16*r + 15*r**2)) * jnp.cos(2*π*z)
        return jnp.array([0, f_theta, 0])

    # Create DeRham sequence
    derham = DeRhamSequence(ns, ps, q, types, F, polar=False, dirichlet=True)
    derham.evaluate_1d()
    derham.assemble_M0()
    derham.assemble_M1()
    derham.assemble_M2()

    # Assemble curl operators
    derham.assemble_d1()
    
    # Assemble the vector Laplacian operator on 2-forms
    derham.assemble_dd2()  
    
    # Mass matrix for 2-forms
    M2 = derham.M2

    # Project source term onto 2-form space
    f_proj = derham.P2(f)
    
    # Solve 
    L = M2 @ derham.dd2
    
    # Solve using pseudo-inverse 
    tol = 1e-12
    eigvals, eigvecs = jnp.linalg.eigh(L)
    
    inv_eigvals = jnp.where(
        jnp.abs(eigvals) > tol,
        1.0 / eigvals,
        0.0
    )
    L_pinv = (eigvecs * inv_eigvals) @ eigvecs.T
    u_hat = L_pinv @ f_proj
    
    # Check residual
    residual = L @ u_hat - f_proj
    residual_norm = jnp.linalg.norm(residual) / jnp.linalg.norm(f_proj)
    print(f" Relative residual: {residual_norm:.2e}")

    # Project exact solution onto 2-form space 
    u_proj = derham.P2(u)
    u_hat_analytic = jnp.linalg.solve(M2, u_proj)
    
    error = ((u_hat - u_hat_analytic) @ M2 @ (u_hat - u_hat_analytic) /
             (u_hat_analytic @ M2 @ u_hat_analytic))**0.5
    return error

    print(error)


if __name__ == "__main__":
    test_cylinder_poisson()
    print("Test passed.")
