import numpy as np
import jax.numpy as jnp
import jax
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Utils import grad,div,curl
from mrx.Quadrature import QuadratureRule
from mrx.SplineBases import SplineBasis, TensorBasis
import scipy.linalg

# Florian thesis,  Eq 4.1. In slab geometry, with cartesian coordinates x,y,z.

# Relevant Parameters/constants
R0 = 3.0
a = 1.0
B0 = 1.0
q0 = 1.05 # Figure 4.1
q1 = 1.85 # Figure 4.1
β = 0.0
gamma = 1

# Create basis
x_basis = SplineBasis(n=2, p=1, type='periodic')  # 2 basis functions, degree 1
y_basis = SplineBasis(n=2, p=1, type='periodic')  # 2 basis functions, degree 1
# Create a Fourier basis for z (toroidal direction)
z_basis = SplineBasis(n=3, p=1, type='fourier')  # 3 Fourier basis functions for z (1,sin(2piz),cos(2piz)))

# Create  tensor basis
xyz_basis = TensorBasis([x_basis, y_basis, z_basis])

# Use F as identity mapping
def F(x):
    return x

# Create differential forms
# Λ0 is for quadrature and pressure
Λ0 = DifferentialForm(0, (x_basis.n, y_basis.n, z_basis.n), 
                     (x_basis.p, y_basis.p, z_basis.p), 
                     (x_basis.type, y_basis.type, z_basis.type))
# Λ2 is for the magnetic field B
Λ2 = DifferentialForm(2, (x_basis.n, y_basis.n, z_basis.n), 
                     (x_basis.p, y_basis.p, z_basis.p), 
                     (x_basis.type, y_basis.type, z_basis.type))

# Use the identity matrix as the evaluation matrix with correct size
E0 = jnp.eye(Λ0.n)
E2 = jnp.eye(Λ2.n)  

# Retrieve Quadrature rule
Q = QuadratureRule(Λ0, 5)

# Get Mass matrix (not sure if this is needed)
# M2 = LazyMassMatrix(Λ2, Q, F, E2).M

def get_hessian(B_hat, p_hat):
    # Create discrete functions for each component of B since B is a vector field
    B_h = [DiscreteFunction(B_hat[i], Λ2, E2) for i in range(3)]
    p_h = DiscreteFunction(p_hat, Λ0, E0)

    # B field evaluations at points
    def get_B_at_points(x):
        return jnp.array([B_h[i](x) for i in range(3)])

    # Pressure evaluations at points
    def get_p_at_points(x):
        return jnp.real(p_h(x)[0])

    # Basis function evaluations at points 
    def get_phi_at_points(x, i):
        basis_val = xyz_basis(x, i)
        # The tensor product basis now includes Fourier basis in z-direction
        return jnp.array([basis_val, basis_val, basis_val])

    # Compute mass matrix C based on inner products of basis functions
    def compute_mass_entry(i, j):
        def integrand(x):
            phi_i = get_phi_at_points(x, i)
            phi_j = get_phi_at_points(x, j)

            # For complex vectors, we need to use conjugate of one of them
            # Also ensure we're taking the real part of the final result
            inner_prod = jnp.real(jnp.sum(jnp.conj(phi_i) * phi_j))
            # Add small regularization to diagonal entries for numerical stability
            return inner_prod 
        
        integrand_values = jax.vmap(integrand)(Q.x)
        weights = Q.w.reshape(-1, 1, 1)
        return jnp.sum(integrand_values * weights)

    # Compute the full mass matrix
    total_basis = Λ2.n
    i_indices = jnp.arange(total_basis)
    j_indices = jnp.arange(total_basis)
    II, J = jnp.meshgrid(i_indices, j_indices, indexing='ij')
    C = jax.vmap(lambda i, j: compute_mass_entry(i, j))(II.flatten(), J.flatten()).reshape(total_basis, total_basis)

    # Curl of B at points
    def get_curl_B(x):
        return curl(get_B_at_points)(x)

    # Gradient of pressure at points
    def get_grad_p(x):
        return grad(get_p_at_points)(x)

    def integrand(i, j, x):
        # Get pre-computed values
        p = get_p_at_points(x)
        phi_i_val = get_phi_at_points(x, i)
        phi_j_val = get_phi_at_points(x, j)
        curl_B = get_curl_B(x)
        grad_p = get_grad_p(x)

        # Compute cross products
        def phi_i_cross_B(y): 
            return jnp.cross(get_phi_at_points(y, i), get_B_at_points(y))
        
        def phi_j_cross_B(y): 
            return jnp.cross(get_phi_at_points(y, j), get_B_at_points(y))
        
        # First term
        curl_i = curl(phi_i_cross_B)(x)
        curl_j = curl(phi_j_cross_B)(x)
        term1 = (jnp.dot(curl_i, curl_j) + jnp.dot(curl_j, curl_i))/2

        # Second term
        div_i = div(lambda y: get_phi_at_points(y, i))(x)
        div_j = div(lambda y: get_phi_at_points(y, j))(x)
        term2 = gamma * p * div_i * div_j
        
        # Third term
        term3 = -(jnp.dot(phi_i_val, jnp.cross(curl_B, curl_j)) + 
                 jnp.dot(phi_j_val, jnp.cross(curl_B, curl_i)))/2
        
        # Fourth term 
        def phi_i_dot_grad_p(y):
            return jnp.real(jnp.dot(get_phi_at_points(y, i), grad_p))
        
        def phi_j_dot_grad_p(y):
            return jnp.real(jnp.dot(get_phi_at_points(y, j), grad_p))
        
        grad_i = jax.grad(phi_i_dot_grad_p)(x)  # Use i for i
        grad_j = jax.grad(phi_j_dot_grad_p)(x)  # Use j for j
        term4 = -(jnp.dot(phi_i_val, grad_j) + jnp.dot(phi_j_val, grad_i))/2
        
        return (term1 + term2 + term3 + term4)

    def compute_hessian_entry(i, j):
        integrand_values = jax.vmap(lambda x: integrand(i, j, x))(Q.x)
        weights = Q.w.reshape(-1, 1, 1)
        return jnp.sum(integrand_values * weights)

    # Now compute the full Hessian
    total_basis = Λ2.n
    i_indices = jnp.arange(total_basis)
    j_indices = jnp.arange(total_basis)
    II, J = jnp.meshgrid(i_indices, j_indices, indexing='ij')
    H = jax.vmap(lambda i, j: compute_hessian_entry(i, j))(II.flatten(), J.flatten()).reshape(total_basis, total_basis)
    
    return H, C

# JIT compile the function 
get_hessian = jax.jit(get_hessian)

def compute_term_hessian(term_func, B_hat, p_hat):
    # Create discrete functions
    B_h = [DiscreteFunction(B_hat[i], Λ2, E2) for i in range(3)]
    p_h = DiscreteFunction(p_hat, Λ0, E0)

    # B field evaluations at points
    def get_B_at_points(x):
        return jnp.array([B_h[i](x) for i in range(3)])

    # Pressure evaluations at points
    def get_p_at_points(x):
        return jnp.real(p_h(x)[0])

    # Basis function evaluations at points
    def get_phi_at_points(x, i):
        basis_val = xyz_basis(x, i)
        return jnp.array([basis_val, basis_val, basis_val])

    # Curl of B at points
    def get_curl_B(x):
        return curl(get_B_at_points)(x)

    # Gradient of pressure at points
    def get_grad_p(x):
        return grad(get_p_at_points)(x)

    total_basis = Λ2.n
    i_indices = jnp.arange(total_basis)
    j_indices = jnp.arange(total_basis)
    II, J = jnp.meshgrid(i_indices, j_indices, indexing='ij')
    
    def compute_term_entry(i, j):
        integrand_values = jax.vmap(lambda x: term_func(i, j, x, get_B_at_points, get_p_at_points, get_phi_at_points, get_curl_B, get_grad_p))(Q.x)
        weights = Q.w.reshape(-1, 1, 1)
        return jnp.sum(integrand_values * weights)
    
    H_term = jax.vmap(lambda i, j: compute_term_entry(i, j))(II.flatten(), J.flatten()).reshape(total_basis, total_basis)
    return H_term

if __name__ == "__main__":
    # Print information about the Fourier basis
    print("Fourier basis structure in z-direction:")
    print(f"  Number of basis functions: {z_basis.n}")
    n_cos = (z_basis.n + 1) // 2  # Integer division for cosine terms (including constant)
    n_sin = z_basis.n - n_cos     # Remaining terms are sine terms
    print(f"  Cosine terms (including constant): {n_cos}")
    print(f"  Sine terms: {n_sin}")
    print("  Basis functions:")
    print("    Basis 0: constant (1)")
    if n_cos > 1:
        for i in range(1, n_cos):
            print(f"    Basis {i}: cos(2π*{i}*z)")
    for i in range(n_sin):
        print(f"    Basis {i + n_cos}: sin(2π*{i + 1}*z)")
    print()
    
    # Test case from FLorian Thesis, 4.1
    def q(x):
        return q0 + (q1-q0)*(x**2)/(a**2)

    def B_hat(x, y, z):
        """Equilibrium B field in slab geometry"""
        Bx = 0.0
        Bz = B0
        By = (B0*a/q(x))
        
        return jnp.array([Bx, By, Bz])

    def p_hat(x, y, z):
        """Pressure p field in slab geometry"""
        return (β*B0**2)/(2)*(1+(a**2/((q(x))**2)*(R0**2))) + ((B0**2)*(a**2)/R0**2)*((1/(q0**2))-(1/q(x)**2))

    # Evaluate the discrete B field on the quadrature points
    B_components = jax.vmap(B_hat, in_axes=(0, 0, 0))(Q.x[:, 0], Q.x[:, 1], Q.x[:, 2])

    # Evaluate the discrete pressure on the quadrature points
    p_components = jax.vmap(p_hat, in_axes=(0, 0, 0))(Q.x[:, 0], Q.x[:, 1], Q.x[:, 2])

    # Reshape to match the basis size
    B_hat = [B_components[:, i].reshape(-1)[:Λ2.n] for i in range(3)]
    p_hat = p_components.reshape(-1)[:Λ0.n]

    print("Computing Hessian for an equilibrium B field in slab geometry...")
    print(f"Using Fourier basis in z-direction with {z_basis.n} basis functions")


    # Compute the Hessian
    H, C = get_hessian(B_hat, p_hat)
    
    # Basic checks
    print("\nHessian shape:", H.shape)
    assert H.shape == (Λ2.n, Λ2.n), f"Hessian has wrong shape: {H.shape}, expected ({Λ2.n}, {Λ2.n})"
    print("Is Hessian zero?", jnp.allclose(H, 0, atol=1e-10))
    print("Is Hessian symmetric?", jnp.allclose(H, H.T, atol=1e-10))

    # Print mass matrix C properties
    print("\nMass matrix C properties:")
    print("Is C zero?", jnp.allclose(C, 0, atol=1e-10))
    print("Is C symmetric?", jnp.allclose(C, C.T, atol=1e-10))
    
    # Compute eigenvalues of C
    C_eigvals = jnp.linalg.eigvals(C)
    print("\nMass matrix C eigenvalue statistics:")
    print("Min eigenvalue:", jnp.min(jnp.real(C_eigvals)))
    print("Max eigenvalue:", jnp.max(jnp.real(C_eigvals)))
    print("Condition number:", jnp.max(jnp.abs(C_eigvals)) / jnp.min(jnp.abs(C_eigvals)))
    
    # Check if C is Hermitian (C = C^H)
    print("\nIs C Hermitian?", jnp.allclose(C, C.T.conj(), atol=1e-10))
    
    # Check if C is positive semidefinite
    print("Is C positive semidefinite?", jnp.all(jnp.real(C_eigvals) >= -1e-10))
    
    # --- Generalized eigenvalue problem Hx = lambda Cx ---
    # Convert JAX arrays to numpy arrays
    H_np = np.array(H)
    C_np = np.array(C)
    
    # Add stronger regularization to ensure C is positive definite
    C_reg = C_np + np.eye(C_np.shape[0]) * 1e-9
    
    # Use Cholesky-based preconditioner

    L = np.linalg.cholesky(C_reg)
    L_inv = np.linalg.inv(L)
    H_precond = L_inv @ H_np @ L_inv.T
    C_precond = np.eye(C_reg.shape[0])  # Identity after preconditioning
    print("\nUsing Cholesky preconditioner")
    
    
    # Compute eigenvalues using scipy's eigh with shift
    H_shifted = H_precond + np.eye(H_precond.shape[0]) * 1e-10

    # Check condition numbers
    H_cond = np.linalg.cond(H_shifted)
    C_cond = np.linalg.cond(C_precond)
    print("\nCondition numbers after preconditioning:")
    print(f"H condition number: {H_cond:.2e}")
    print(f"C condition number: {C_cond:.2e}")

    eigvals_gen, eigvecs_gen = scipy.linalg.eigh(H_shifted, C_precond)
    print("\nUsing scipy.linalg.eigh to compute all eigenvalues")

        
    # Sort eigenvalues
    idx = np.argsort(eigvals_gen)
    eigvals_gen = eigvals_gen[idx]
    eigvecs_gen = eigvecs_gen[:, idx]
    

    print("\nGeneralized eigenvalue statistics (Hx = lambda Cx):")
    print("\n10 smallest generalized eigenvalues:")
    for i in range(min(10, len(eigvals_gen))):
        print(f"  {i}: {eigvals_gen[i]:.6e}")
    
    # Count zero eigenvalues in generalized problem
    tol_gen = 1e-10
    zero_mask_gen = np.abs(eigvals_gen) < tol_gen
    zero_eigvals_gen = np.sum(zero_mask_gen)
    print(f"\nNumber of zero generalized eigenvalues (tol={tol_gen}): {zero_eigvals_gen}")
    
    # Analyze stability
    negative_eigvals = eigvals_gen[eigvals_gen < 0]
    if len(negative_eigvals) > 0:
        print("\nSTABILITY ANALYSIS:")
        print(f"Number of unstable modes: {len(negative_eigvals)}")
        print(f"Most unstable eigenvalue: {np.min(negative_eigvals):.6e}")
    
   