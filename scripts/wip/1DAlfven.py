'''
Hessian calculation and Alfvén wave analysis

This file solves the generalized eigenvalue problem, Hx = λCx, where H is the Hessian of the force operator,
C is a mass matrix, and x is the eigenvector. The equilibrium B-field and equilibrium pressure is from (4.1) in Florian's thesis, and is in a slab geometry 
in (x',y',z') slab coordinates.

The four terms in the Hessian are:
1. Term 1: ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀))  dx
2. Term 2: ∫ γ*p (∇·φ_i)* (∇·φ_j) dx
3. Term 3: ∫ φ_i · [(∇×B₀) × (∇×(φ_j × B₀))] dx
4. Term 4: ∫ φ_i · ∇(φ_j · ∇p) dx

I symmetrize the Hessian after the fact. Everything is in double precision.
'''

import numpy as np
import jax.numpy as jnp
import jax
import jax.lax
import matplotlib
matplotlib.use('Agg') 
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
import scipy.linalg
from mrx.LazyMatrices import LazyMassMatrix, LazyWeightedDoubleDivergenceMatrix, LazyMagneticTensionMatrix, LazyPressureGradientForceMatrix, LazyCurrentDensityMatrix
import time

# Relevant parameters/constants
R0 = 3.0
a = 1.0
B0 = 1.0
q0 = 1.05 # Figure 4.1
q1 = 1.85 # Figure 4.1
β = 0.0  
gamma = 5/3

# Mode number parameters 
n = -1  
m = 1

# Wave numbers 
k_y = m/(2*jnp.pi*a)
k_z = n/(2*jnp.pi*R0)

# q-profile function
def q_profile(x):
    """q-profile as a function of x"""
    return q0 + (q1-q0)*(x**2)/(a**2)  

differential_1form = DifferentialForm( 
    k=1,  # 1-form 
    ns=[10, 2, 2],  # Increased x-resolution from 6 to ensure enough quadrature points
    ps=[4, k_y, k_z], 
    types=['clamped', 'simple_fourier', 'simple_fourier']  
)

# Quadrature rule
Q = QuadratureRule(differential_1form, 7)

# Coordinate transformation
def F(x):
    """Transform coordinates to slab dimensions: Lx = a, Ly = 2πa, Lz = 2πR₀"""
    return jnp.array([a * x[0], 2*jnp.pi*a * x[1], 2*jnp.pi*R0 * x[2]])

def get_hessian_components_fast():
    """
    Fast version: Compute individual matrix components.
    """
    
    # ============================================================================
    # X-DEPENDENT EQUILIBRIUM FIELD FUNCTIONS
    # ============================================================================
    
    # q-profile function is now defined globally
    
    # Pressure function 
    def pressure_func(x_logical):
        """Pressure function that takes logical coordinates and returns pressure"""
        # Only need x-coordinate since p only depends on x
        x_slab = a * x_logical[0]  # Transform only x to slab coordinates
        q_val = q_profile(x_slab)
        p_val = ((β*B0**2)/(2))*(1+(a/(q_val*R0))**2) + ((B0**2)*(a**2)/R0**2)*((1/(q0**2))-(1/q_val**2))
        return p_val

    # Equilibrium B₀ function
    def B0_field(x_logical):
        """Equilibrium B-field function that takes logical coordinates and returns B₀ in LOGICAL coordinates"""
        # Transform to slab coordinates for q-profile calculation (leaving in case we change a in future)
        x_slab = a * x_logical[0]
        q_val = q_profile(x_slab)
        
        # B₀ in slab coordinates
        B0x_slab = 0.0
        B0z_slab = B0
        B0y_slab = (B0*a)/(q_val*R0)
        
        # Transform B₀ back to logical coordinates
        # For slab geometry: B_logical = B_slab / (scale factors)
        B0x_logical = B0x_slab / a
        B0y_logical = B0y_slab / (2*jnp.pi*a)
        B0z_logical = B0z_slab / (2*jnp.pi*R0)
        
        return jnp.array([B0x_logical, B0y_logical, B0z_logical])

    # ============================================================================
    # COMPUTE MATRIX COMPONENTS OF HESSIAN
    # ============================================================================
    
    print("Computing matrix components (fast version)...")
    start_time = time.time()
    
    # Mass matrix
    M_mass = LazyMassMatrix(differential_1form, Q, F=F)
    C = M_mass.M
    print(f"Mass matrix: {C.shape} ({time.time() - start_time:.2f}s)")

    # Term 1:  ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀))  dx
    K_magnetic_tension = LazyMagneticTensionMatrix(differential_1form, Q, B0_field, F=F)    
    print(f"Term 1: {time.time() - start_time:.2f}s")

    # Term 2: ∫ γ*p*(∇·φ_i)*(∇·φ_j)
    D_divdiv_weighted = LazyWeightedDoubleDivergenceMatrix(differential_1form, Q, pressure_func, F=F)
    print(f"Term 2: {time.time() - start_time:.2f}s")

    # Term 3: ∫ φ_i · [(∇×B₀) × (∇×(φ_j × B₀))] dx
    K_current_density = LazyCurrentDensityMatrix(differential_1form, Q, B0_field, F=F)
    print(f"Term 3: {time.time() - start_time:.2f}s")

    # Term 4: ∫ φ_i · ∇(φ_j · ∇p) dx
    K_pressure_gradient_force = LazyPressureGradientForceMatrix(differential_1form, Q, pressure_func, F=F)
    K_pressure_gradient_force = K_pressure_gradient_force.M
    print(f"Term 4: {time.time() - start_time:.2f}s")
    
    
    print(f"Total matrix computation time: {time.time() - start_time:.2f}s")
    
    return K_magnetic_tension, D_divdiv_weighted.M, K_current_density.M, K_pressure_gradient_force, C

# JIT-compiled assembly function
@jax.jit
def assemble_hessian_jitted_fast(K_curlcurl, D_divdiv_weighted, K_current_density, K_pressure_gradient_force, gamma):
    """
    Fast JIT-compiled function to assemble the Hessian.s
    """
    H = K_curlcurl + gamma * D_divdiv_weighted - K_current_density - K_pressure_gradient_force
    H = 0.5*(H + H.T)  # Symmetrize the matrix
    return H


if __name__ == "__main__":
    print("=== ALFVÉN WAVE COMPUTATION ===")
 
 
    # Compute the Hessian using fast approach
    start_total = time.time()
    
    K_curlcurl, D_divdiv_weighted, K_current_density, K_pressure_gradient_force, C_analytical = get_hessian_components_fast()
    
    # Convert lazy matrices to NumPy arrays for JIT compilation
    print("Converting to NumPy arrays...")
    K_curlcurl_np = np.array(K_curlcurl)
    D_divdiv_weighted_np = np.array(D_divdiv_weighted)
    K_current_density_np = np.array(K_current_density)
    K_pressure_gradient_force_np = np.array(K_pressure_gradient_force)
    
    # Use JIT-compiled assembly
    print("Assembling Hessian...")
    H_analytical = assemble_hessian_jitted_fast(K_curlcurl_np, D_divdiv_weighted_np, K_current_density_np, K_pressure_gradient_force_np, gamma)
    
    print(f"Hessian assembly completed: {time.time() - start_total:.2f}s")
    print(f"Hessian shape: {H_analytical.shape}")
    print(f"Mass matrix shape: {C_analytical.shape}")

    # Convert to numpy for analysis
    H_np = np.array(H_analytical)
    C_np = np.array(C_analytical)
    
    
    # Use Cholesky decomposition to transform to regular eigenvalue problem
    print("Using Cholesky decomposition to transform Hx = λCx → H̃y = λy")
    

    #  HESSIAN COMPONENT ANALYSIS
    print("\n=== HESSIAN COMPONENT ANALYSIS ===")
    
    # Collect all matrices for vectorized analysis
    matrices = [K_curlcurl_np, D_divdiv_weighted_np, K_current_density_np, K_pressure_gradient_force_np]
    matrix_names = ["Term 1 (Magnetic tension)", "Term 2 (Pressure gradient)", 
                   "Term 3 (Current density)", "Term 4 (Pressure gradient force)"]
    
    # Analyze each matrix individually 
    matrix_properties = []
    for i, matrix in enumerate(matrices):
        print(f"Analyzing matrix {i+1} ({matrix_names[i]})...")
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Matrix range: [{np.min(matrix):.2e}, {np.max(matrix):.2e}]")
        print(f"  Matrix has NaN: {np.any(np.isnan(matrix))}")
        print(f"  Matrix has Inf: {np.any(np.isinf(matrix))}")
        try:
            # Try to compute condition number 
            try:
                cond_num = np.linalg.cond(matrix)
            except np.linalg.LinAlgError:
                cond_num = float('inf')
                print("  ⚠️  Failed condition number computation")
            
            # Try to compute eigenvalues with error handling
            try:
                eigvals = np.linalg.eigvals(matrix)
                min_eig = np.min(eigvals)
                max_eig = np.max(eigvals)
            except np.linalg.LinAlgError:
                min_eig = float('nan')
                max_eig = float('nan')
                print("  ⚠️  Eigenvalue computation failed")
            
            props = {
                'shape': matrix.shape,
                'symmetric': np.allclose(matrix, matrix.T, atol=1e-7),
                'condition_number': cond_num,
                'min_eigenvalue': min_eig,
                'max_eigenvalue': max_eig,
            }
        except Exception as e:
            print(f"  ⚠️  Error analyzing matrix: {e}")
            props = {
                'shape': matrix.shape,
                'symmetric': False,
                'condition_number': float('inf'),
                'min_eigenvalue': float('nan'),
                'max_eigenvalue': float('nan'),
            }
        matrix_properties.append(props)
    
    # Print results
    for i, name in enumerate(matrix_names):
        props = matrix_properties[i]
        print(f"{name} properties:")
        print(f"  Shape: {props['shape']}")
        print(f"  Symmetric: {props['symmetric']}")
        print(f"  Condition number: {props['condition_number']:.2e}")
        print(f"  Min eigenvalue: {props['min_eigenvalue']:.2e}")
        print(f"  Max eigenvalue: {props['max_eigenvalue']:.2e}")
        print()
    
   
    
    # CHOLESKY DECOMPOSITION APPROACH
    print("\n=== CHOLESKY DECOMPOSITION TRANSFORMATION ===")
    print("Computing Cholesky decomposition: C = L L^T")
    

    C_symmetric = 0.5*(C_np + C_np.T)
    H_symmetric = H_np
    
    
    
    # Compute Cholesky decomposition: C = L L^T
    L = np.linalg.cholesky(C_symmetric)
    print("  ✓ Cholesky decomposition successful")
    
    # Transform the problem: Hx = λCx → H̃y = λy
    # where H̃ = L^(-1) H L^(-T) and y = L^T x
    print("Transforming problem: H̃ = L^(-1) H L^(-T)")
    
    # Check Hessian matrix properties
    print("Checking Hessian matrix properties...")
    print(f"  Hessian matrix shape: {H_symmetric.shape}")
    print(f"  Hessian matrix range: [{np.min(H_symmetric):.2e}, {np.max(H_symmetric):.2e}]")
    print(f"  Hessian matrix has NaN: {np.any(np.isnan(H_symmetric))}")
    print(f"  Hessian matrix has Inf: {np.any(np.isinf(H_symmetric))}")
    
    # Compute L^(-1) H L^(-T) using triangular solves
    Z = scipy.linalg.solve_triangular(L, H_symmetric.T, lower=True).T
    H_tilde = scipy.linalg.solve_triangular(L, Z, lower=True)
    print("  ✓ Triangular solve successful")
    
    # Ensure symmetry by averaging with transpose
    H_tilde = 0.5 * (H_tilde + H_tilde.T)
    
    print("  ✓ Transformation completed")
    print(f"  H̃ matrix condition number: {np.linalg.cond(H_tilde):.2e}")
    print(f"  H̃ symmetric: {np.allclose(H_tilde, H_tilde.T, atol=1e-10)}")
    

    # Solve regular eigenvalue problem: H̃y = λy
    print("\n=== SOLVING REGULAR EIGENVALUE PROBLEM ===")
    print("Computing eigenvalues and eigenvectors of H̃...")
    
    eigvals, eigvecs_y = scipy.linalg.eigh(H_tilde)
    
    # Transform eigenvectors back: x = L^(-T) y
    print("Transforming eigenvectors back: x = L^(-T) y")
    eigvecs_full = scipy.linalg.solve_triangular(L.T, eigvecs_y, lower=False)
    
    print("  ✓ Eigenvalue computation completed")
    print(f"  Number of eigenvalues: {len(eigvals)}")
    print(f"  Eigenvalue range: [{np.min(eigvals):.2e}, {np.max(eigvals):.2e}]")

    # ANALYSIS
    eigvals_full = eigvals[:C_np.shape[0]]
    eigvecs_full = eigvecs_full[:, :C_np.shape[0]]
    alfven_scale = B0**2/R0**2
    eigvals_normalized = eigvals_full / alfven_scale


    # q profile
    def q(x): return q0 + (q1-q0)*(x**2)/(a**2)

    # Leaving this in for sound continuum later
    def B_cont(x):
        """Norm of equilibrium B field for continuum calculation"""
        Bx = 0.0
        Bz = B0
        By = (B0*a)/(q(x)*R0)
        return jnp.sqrt(Bx**2 + By**2 + Bz**2)
    
    def p_cont(x):
        """Pressure p field for continuum calculation"""
        # Pressure terms
        thermal_pressure = (β*B0**2)/(2)*(1+(a**2/((q(x))**2)*(R0**2)))
        magnetic_pressure = ((B0**2)*(a**2)/R0**2)*((1/(q0**2))-(1/q(x)**2))
        
        return thermal_pressure  + magnetic_pressure

    # Normalized Alfvén continuum
    def w_2(x): return B0**2/R0**2 * (n + (m/q(x)))**2
    
    # CONTINUUM EVALUATION
    print("\n===CONTINUUM EVALUATION ===")
    x_vals = np.linspace(0, a, 50) 
    
    q_vals = jax.vmap(q)(x_vals)
    B_cont_vals = jax.vmap(B_cont)(x_vals)
    p_cont_vals = jax.vmap(p_cont)(x_vals)
    w_2_vals = jax.vmap(w_2)(x_vals)
    
    continuum_vals = w_2_vals / alfven_scale
    continuum_min = np.min(continuum_vals)
    continuum_max = np.max(continuum_vals)

    print("\n=== CONTINUUM ANALYSIS ===")
    print(f"Alfvén continuum range: [{continuum_min:.3f}, {continuum_max:.3f}]")
    
    # Find eigenvalues in Alfvén continuum
    continuum_mask = (eigvals_normalized >= continuum_min) & (eigvals_normalized <= continuum_max)
    continuum_indices = np.where(continuum_mask)[0]
    continuum_eigenvalues = eigvals_normalized[continuum_mask]

    print("\n=== RESULTS SUMMARY ===")
    print(f"Found {len(continuum_eigenvalues)} eigenvalues within Alfvén continuum range")
    print(f"Total computation time: {time.time() - start_total:.2f}s")

    # DIRECT DIVERGENCE ANALYSIS
    if len(continuum_eigenvalues) > 0:
        print("\n=== 1D DIVERGENCE ANALYSIS ===")
        
        # Define 1D divergence function 
        def compute_divergence_1d(eigvec):
            """Compute 1D divergence of eigenfunction along x-direction"""
            eigenfunction = DiscreteFunction(eigvec, differential_1form)
            
            # Create evaluation points for divergence computation
            n_div_points = 50
            x_logical = np.linspace(0.0, 1.0, n_div_points)  
            x_physical = x_logical * a
            y_fixed = 0.5 #Using fror now
            z_fixed = 0.5
            eval_points = np.array([[x, y_fixed, z_fixed] for x in x_logical])
            
            # Compute eigenfunction values at evaluation points
            eigenfunction_values = jax.vmap(eigenfunction)(eval_points)  
            
            # Simple finite difference approach for divergence
            # For slab geometry, I focus on radial directionfor now: div = ∂ξ_x/∂x 

            x_components = eigenfunction_values[:, 0] 
            # Compute ∂ξ_x/∂x using finite differences
            dx = x_physical[1] - x_physical[0]
            div_values = np.gradient(x_components, dx)
            
            # Compute L2 norm of divergence
            div_norm_squared = np.sum(div_values**2) * dx
            div_norm = np.sqrt(div_norm_squared)
        
            
            return div_norm, div_values, x_physical

        
        # ANALYZE ALFVÉN CONTINUUM MODES
        if len(continuum_eigenvalues) > 0:
            print("\n=== ALFVÉN CONTINUUM MODES ===")
            print(f"{'Mode':<6} {'Index':<8} {'Eigenvalue':<12} {'Div1D':<12}")
            print("-" * 50)
            
        
            # Analyze all Alfvén continuum modes
            for i, idx in enumerate(continuum_indices):
                eigval = continuum_eigenvalues[i]
                eigvec = eigvecs_full[:, idx]
                
                # Compute 1D divergence
                div_norm_1d, div_values_1d, x_div_1d = compute_divergence_1d(eigvec)
                
                print(f"{i:<6} {idx:<8} {eigval:<12.3f} {div_norm_1d:<12.3f}")
                
                # Classify mode type based on 1D divergence
                if div_norm_1d < 0.3:
                    print("    ✓ Pure Alfvén mode (incompressible)")
                elif div_norm_1d < 0.6:
                    print("    ⚠️  Mixed mode")
                else:
                    print("    ❌ Compressible mode")
                
                # Check for degeneracy with previous mode
                if i > 0:
                    prev_idx = continuum_indices[i-1]
                    prev_eigval = continuum_eigenvalues[i-1]
                    prev_eigvec = eigvecs_full[:, prev_idx]
                    
                    # Check eigenvalue degeneracy
                    eigval_diff = abs(eigval - prev_eigval)
                    if eigval_diff < 1e-10:
                        print("    🔄 DEGENERATE: within e-10 of  previous mode")
                        
                        # Check differences in eigenvectors by taking norms
                        overlap = abs(eigvec.T @ C_np @ prev_eigvec) / (np.sqrt(eigvec.T @ C_np @ eigvec) * np.sqrt(prev_eigvec.T @ C_np @ prev_eigvec))
                        print(f"    Overlap with previous mode: {overlap:.6f}")
                        
                        if overlap > 0.9:
                            print("    ❌  Nearly identical eigenvectors")
                        elif overlap < 0.1:
                            print("    ✓ Orthogonal eigenvectors")
                        else:
                            print("    ⚠️  mixed degeneracy")
                    elif eigval_diff < 1e-6:
                        print(f"    ⚠️  NEAR-DEGENERATE: Very close eigenvalue (diff: {eigval_diff:.2e})")
                
               

 