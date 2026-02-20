"""
Testing that the numerical solution of the cylinder vector Poisson problem converges to the exact solution with reasonable error, using projection.
"""

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

    # Assemble curl 
    derham.assemble_d1()
    
    # Assemble the vector Laplacian operator on 2-forms
    derham.assemble_dd2()  
    
    
    # Mass matrix for 2-forms
    M2 = derham.M2

    # Project source term onto 2-form space
    f_proj = derham.P2(f)

    # Solve the system
    L = M2 @ derham.dd2
    
    # Generalized eigendecomposition gives Lv = λ M2 v, where singular vector satisfies L v = 0.
    # We transform to standard form: M2^(1/2) L M2^(-1/2) w = λ w where w = M2^(1/2) v
    
   # Cholesky decomposition of M2 with small regularization
    L_chol = jnp.linalg.cholesky(M2 + jnp.eye(M2.shape[0]) * 1e-12)

    # Invert L_chol (cholesky factor of M2)
    L_chol_inv = jnp.linalg.inv(L_chol)

    # Transform to standard 
    L_transformed = L_chol_inv @ L @ L_chol_inv.T

    # Solve standard eigenvalue problem 
    eigvals, eigvecs_transformed = jnp.linalg.eigh(L_transformed)

    # Transform eigenvectors back: v = M2^(-1/2)  w = L_chol_inv.T  w
    eigvecs = L_chol_inv.T @ eigvecs_transformed

    # Find the smallest eigenvalue and eigenvector (this is the singular vector)
    min_idx = jnp.argmin(jnp.abs(eigvals))
    v = eigvecs[:, min_idx]
    #λ_min = eigvals[min_idx]
    
    # Evaluate projection matrix: P = I - M2 v v.T / (v.T M2 v)
    v_Mv = v @ M2 @ v
    P = jnp.eye(L.shape[0]) - jnp.outer(M2 @ v, v) / v_Mv
    
    # Project the system: P L P @ x = P f_proj
    L_proj = P @ L @ P
    f_proj_proj = P @ f_proj
    
    # Solve with least squares
    u_hat, residuals, rank, s = jnp.linalg.lstsq(L_proj, f_proj_proj)
    
    
    # Project exact solution onto 2-form space 
    u_proj = derham.P2(u)

    # Full exact solution (coefficient vector)
    u_hat_analytic_full = jnp.linalg.solve(M2, u_proj)
    
    # Project to remove null space  (in case)
    u_hat_analytic = P @ u_hat_analytic_full
    
    error = ((u_hat - u_hat_analytic) @ M2 @ (u_hat - u_hat_analytic) /
             (u_hat_analytic @ M2 @ u_hat_analytic))**0.5

    print(error)
    return error


if __name__ == "__main__":
    test_cylinder_poisson()
    print("Test passed.")
