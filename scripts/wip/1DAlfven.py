'''
Hessian calculation and Alfvén wave analysis

This file solves the generalized eigenvalue problem, Hx = λCx, where H is the Hessian of the force operator,
C is a mass matrix, and x is the eigenvector. The equilibrium B-field and equilibrium pressure is from (4.1) in Florian's thesis, and is in a slab geometry 
in (x',y',z') slab coordinates. The basis functions are now 2 forms.

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
matplotlib.use('TkAgg') 
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
from mrx.BoundaryConditions import LazyBoundaryOperator
import scipy.linalg
from mrx.Utils import inv33,jacobian_determinant
from mrx.LazyMatrices import LazyMassMatrix, LazyMagneticTensionMatrixV2, LazyPressureGradientForceMatrixV2, LazyCurrentDensityMatrixV2, LazyWeightedDoubleDivergenceMatrixV2, LazyDerivativeMatrix
import time
import matplotlib.pyplot as plt

# Relevant parameters/constants
R0 = 3.0
a = 1.5
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

differential_2form = DifferentialForm( 
    k=2,  
    ns=[8,2, 2], 
    ps=[3, k_y, k_z], 
    types=[
        'clamped', 
        'simple_fourier', 
        'simple_fourier'
    ]
)

Q = QuadratureRule(differential_2form, 7)  

def F(x):
    """Transform coordinates to slab dimensions: Lx = a, Ly = 2πa, Lz = 2πR₀"""
    return jnp.array([a * x[0], 2*jnp.pi*a * x[1], 2*jnp.pi*R0 * x[2]])


def DF(x):
    """Jacobian of the coordinate transformation"""
    return jax.jacfwd(F)(x)

def get_hessian_components_fast():
    """
    Fast version: Compute individual matrix components.
    """
    
    # ============================================================================
    # X-DEPENDENT EQUILIBRIUM FIELD FUNCTIONS
    # ============================================================================
    
    # Pressure function 
    def pressure_func(x_log):
        """Pressure function that takes logical coordinates """
        x_coord = x_log[0] if jnp.ndim(x_log) > 0 else x_log # x-coordinate
        q_val = q_profile(x_coord)
        p_val = ((β*B0**2)/(2))*(1+(a/(q_val*R0))**2) + ((B0**2)*(a**2)/R0**2)*((1/(q0**2))-(1/q_val**2))
        return jnp.array([p_val]) 

    # Equilibrium B₀ function with force balance
    def B0_field(x_log):
        """B-field function that takes logical coordinates"""
        x_coord = x_log[0] if jnp.ndim(x_log) > 0 else x_log # x-coordinate
        q_val = q_profile(a*x_coord)
        
       
        B0_x = 0.0  
        B0_y = (B0*a)/(q_val*R0)
        B0_z = B0
        
      
        
        return jnp.array([B0_x, B0_y, B0_z])


    # ============================================================================
    # COMPUTE MATRIX COMPONENTS OF HESSIAN
    # ============================================================================
    
    print("Computing matrix components (fast version)...")
    start_time = time.time()
    
    # Mass matrix
    M_mass = LazyMassMatrix(differential_2form, Q, F=F)
    C = M_mass.M
    print(f"Mass matrix: {C.shape} ({time.time() - start_time:.2f}s)")

    # Term 1:  ∫ (∇ × (φ_i × B₀)) · (∇ × (φ_j × B₀))  dx
    K_magnetic_tension = LazyMagneticTensionMatrixV2(differential_2form, Q, B0_field, F=F)    
    print(f"Term 1: {time.time() - start_time:.2f}s")

    # Term 2: ∫ γ*p*(∇·φ_i)*(∇·φ_j)dx
    D_divdiv_weighted = LazyWeightedDoubleDivergenceMatrixV2(differential_2form, Q, pressure_func, F=F).M
    print(f"Term 2: {time.time() - start_time:.2f}s")

    # Term 3: ∫ φ_i · [(∇×B₀) × (∇×(φ_j × B₀))] dx (note(∇×B₀) is nearly zero)
    K_current_density = LazyCurrentDensityMatrixV2(differential_2form, Q, B0_field, F=F)
    print(f"Term 3: {time.time() - start_time:.2f}s")

    # Term 4: ∫ φ_i · ∇(φ_j · ∇p) dx
    #K_pressure_gradient_force = LazyPressureGradientForceMatrixV2(differential_2, Q, pressure_func, F=F).M
    # Use a simpler approach that doesnt require 3-forms
    K_pressure_gradient_force = LazyPressureGradientForceMatrixV2(differential_2form, Q, pressure_func, F=F).M
    print(f"Term 4: {time.time() - start_time:.2f}s")
    
    
    print(f"Total matrix computation time: {time.time() - start_time:.2f}s")
    
    return K_magnetic_tension, D_divdiv_weighted, K_current_density.M, K_pressure_gradient_force, C

# JIT-compiled assembly function
@jax.jit
def assemble_hessian_jitted_fast(K_curlcurl, D_divdiv_weighted, K_current_density, K_pressure_gradient_force, gamma):
    """
    Fast JIT-compiled function to assemble the Hessian.
    Using magnetic tension (Term 1) and div-div matrix (Term 2) for balanced system.
    """
    # Original weights for all terms
    H = K_curlcurl + gamma * D_divdiv_weighted - K_current_density - K_pressure_gradient_force 
    H = 0.5*(H + H.conj().T)  # Symmetrize the matrix
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
    
    # Apply boundary conditions to enforce zero at x=0,1
    print("\n=== APPLYING BOUNDARY CONDITIONS ===")
    print("Creating boundary operator for zero boundary conditions at x=0,1...")
    
    # Create boundary operator for zero BCs in x-direction
    boundary_op = LazyBoundaryOperator(differential_2form, ('dirichlet', 'none', 'none'))
    B = np.array(boundary_op.M)
    
    print(f"Boundary operator shape: {B.shape}")
    print(f"Full space dimension: {H_np.shape[0]}")
    print(f"Reduced space dimension: {B.shape[0]}")
    print(f"Removed {H_np.shape[0] - B.shape[0]} boundary degrees of freedom")
    
    # Apply boundary conditions to matrices
    H_reduced = B @ H_np @ B.T
    C_reduced = B @ C_np @ B.T
    
    print(f"Reduced Hessian shape: {H_reduced.shape}")
    print(f"Reduced mass matrix shape: {C_reduced.shape}")
    
    # Use reduced matrices for eigenvalue problem
    H_np = H_reduced
    C_np = C_reduced
    
    
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
        
        # Handle condition number formatting
        if np.isscalar(props['condition_number']) and np.isfinite(props['condition_number']):
            print(f"  Condition number: {props['condition_number']:.2e}")
        else:
            print(f"  Condition number: {props['condition_number']}")
        
        # Handle eigenvalue formatting
        if np.isscalar(props['min_eigenvalue']) and np.isfinite(props['min_eigenvalue']):
            print(f"  Min eigenvalue: {props['min_eigenvalue']:.2e}")
        else:
            print(f"  Min eigenvalue: {props['min_eigenvalue']}")
            
        if np.isscalar(props['max_eigenvalue']) and np.isfinite(props['max_eigenvalue']):
            print(f"  Max eigenvalue: {props['max_eigenvalue']:.2e}")
        else:
            print(f"  Max eigenvalue: {props['max_eigenvalue']}")
        print()
    
   
    
    # CHOLESKY DECOMPOSITION APPROACH
    print("\n=== CHOLESKY DECOMPOSITION TRANSFORMATION ===")
    print("Computing Cholesky decomposition: C = L L^T")
    

    C_symmetric = 0.5*(C_np + C_np.T)
    H_symmetric = H_np
    
    
    
    # Compute Cholesky decomposition: C = L L^T
    L = np.linalg.cholesky(C_symmetric)
    print("  ✓ Cholesky decomposition successful")
    
    # Transform the problem: Hx = λCx → H̃ = L^(-1) H L^(-T) and y = L^T x
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
    eigvecs_reduced = scipy.linalg.solve_triangular(L.T, eigvecs_y, lower=False)
    
    print("  ✓ Eigenvalue computation completed")
    print(f"  Number of eigenvalues: {len(eigvals)}")
    print(f"  Eigenvalue range: [{np.min(eigvals):.2e}, {np.max(eigvals):.2e}]")

    # ANALYSIS (Making sure shape is correct)
    eigvals_full = eigvals[:C_np.shape[0]]
    eigvecs_reduced = eigvecs_reduced[:, :C_np.shape[0]]
    alfven_scale = B0**2/R0**2
    eigvals_normalized = eigvals_full / alfven_scale

    # Map eigenvectors back to full space for plotting
    print("Mapping eigenvectors back to full space for plotting...")
    eigvecs_full = B.T @ eigvecs_reduced
    print(f"  Full space eigenvector shape: {eigvecs_full.shape}")
    
    # Use unnormalized eigenfunctions
    eigvecs_normalized = eigvecs_full
    
    # Verify boundary conditions are working
    print("\n=== VERIFYING BOUNDARY CONDITIONS ===")
    print("Checking that eigenvectors are zero at boundaries...")
    
    # Check the first few eigenvectors at boundary indices
    # For clamped B-splines, the first and last basis functions are boundary functions
    boundary_indices = [0, 1, eigvecs_normalized.shape[0]-2, eigvecs_normalized.shape[0]-1]
    
    for i in range(min(5, eigvecs_normalized.shape[1])):
        eigenvector = eigvecs_normalized[:, i]
        boundary_values = eigenvector[boundary_indices]
        max_boundary_val = np.max(np.abs(boundary_values))
        print(f"  Mode {i}: max boundary value = {max_boundary_val:.2e}")
        
        if max_boundary_val > 1e-10:
            print(f"    ⚠️  Warning: Non-zero boundary values detected!")
        else:
            print(f"    ✓ Boundary conditions satisfied")
    
    print("\n=== USING UNNORMALIZED EIGENFUNCTIONS ===")
    print("Eigenfunction norms (unnormalized):")
    for i in range(min(10, eigvecs_normalized.shape[1])):
        norm = np.sqrt(eigvecs_normalized[:, i].T @ C_np @ eigvecs_normalized[:, i])
        print(f"  Mode {i}: {norm:.6f}")
   

    # q profile
    def q(x): return q0 + (q1-q0)*(x**2)/(a**2)


    #  Alfvén continuum
    def w_2(x): return B0**2/R0**2 * (n + (m/q(x)))**2

    def plot_eigenfunction(eigvec, eigval, differential_2form, a, alfven_scale, mode_index, mass_matrix=None):
        """Plot eigenfunction magnitude with proper 2-form to vector transformation."""
        print(f"\n=== PLOTTING EIGENFUNCTION FOR MODE {mode_index} ===")
        
        # Compute both Euclidean norm and L2 norm with respect to mass matrix
        euclidean_norm = np.linalg.norm(eigvec)
        print(f"Euclidean norm: {euclidean_norm:.6f}")
        
        if mass_matrix is not None:
            l2_norm = np.sqrt(eigvec.T @ mass_matrix @ eigvec)
            print(f"L2 norm (with mass matrix): {l2_norm:.6f}")
        else:
            l2_norm = euclidean_norm
            print(f"L2 norm (no mass matrix provided): {l2_norm:.6f}")
            
        print(f"Eigenvector range: [{np.min(eigvec):.6f}, {np.max(eigvec):.6f}]")
        
      
        # Create fine grid for plotting (logical coordinates)
        n_plot = 100  # Increased resolution
        x_min_safe = 0.01  # Avoid x=0 to prevent boundary issues
        x_max_safe = 0.99  # Avoid x=1 to prevent boundary issues
        x_logical_plot = np.linspace(x_min_safe, x_max_safe, n_plot)
        y_fixed = 0.5
        z_fixed = 0.5
        
        # Create eigenfunction from coefficients (2-form)
        eigenfunction_2form = DiscreteFunction(eigvec, differential_2form)
        
        print("\nComputing eigenfunction on full domain x=0 to x=1...")
        
        try:
            # Evaluate eigenfunction at each point
            eigenfunction_values = []
            physical_positions = []
            
            for x_log in x_logical_plot:
                x_logical = jnp.array([x_log, y_fixed, z_fixed])
                
                # Evaluate 2-form at this logical point  
                twoform_values = eigenfunction_2form(x_logical)
                eigenfunction_values.append(twoform_values)
                
                # Get physical position
                x_physical = F(x_logical)
                physical_positions.append(x_physical[0])  # x-coordinate only
            
            eigenfunction_values = np.array(eigenfunction_values)
            physical_positions = np.array(physical_positions)
            
            # Compute magnitude of entire eigenfunction (all components)
            eigenfunction_magnitude = np.sqrt(np.sum(np.abs(eigenfunction_values)**2, axis=1))
            
            # Also extract individual components for comparison
            u_x_plot = np.abs(eigenfunction_values[:, 0])  # Radial component
            u_y_plot = np.abs(eigenfunction_values[:, 1])  # Poloidal component  
            u_z_plot = np.abs(eigenfunction_values[:, 2])  # Toroidal component
            
            # Add diagnostic information about boundary behavior
            print(f"Eigenfunction magnitude near x=0: {eigenfunction_magnitude[:5]}")
            print(f"Eigenfunction magnitude near x=1: {eigenfunction_magnitude[-5:]}")
            print(f"Max eigenfunction magnitude: {np.max(eigenfunction_magnitude):.6f}")
            print(f"Min eigenfunction magnitude: {np.min(eigenfunction_magnitude):.6f}")
            print(f"Component ranges:")
            print(f"  u_x (radial): [{np.min(u_x_plot):.6f}, {np.max(u_x_plot):.6f}]")
            print(f"  u_y (poloidal): [{np.min(u_y_plot):.6f}, {np.max(u_y_plot):.6f}]")
            print(f"  u_z (toroidal): [{np.min(u_z_plot):.6f}, {np.max(u_z_plot):.6f}]")
            
            # Analyze eigenvector coefficients
            print(f"\n=== EIGENVECTOR COEFFICIENT ANALYSIS ===")
            print(f"Eigenvector shape: {eigvec.shape}")
            print(f"Eigenvector range: [{np.min(eigvec):.6f}, {np.max(eigvec):.6f}]")
            
            # Check the first few coefficients (likely to affect x=0 behavior)
            print(f"First 5 coefficients: {eigvec[:5]}")
            print(f"Last 5 coefficients: {eigvec[-5:]}")
            
            # Check if any coefficients are unusually large
            abs_coeffs = np.abs(eigvec)
            max_coeff_idx = np.argmax(abs_coeffs)
            max_coeff_val = abs_coeffs[max_coeff_idx]
            print(f"Largest coefficient: {max_coeff_val:.6f} at index {max_coeff_idx}")
            
            # Check if the first coefficient is the largest
            if max_coeff_idx == 0:
                print("⚠️  WARNING: First coefficient (c₀) is the largest!")
                print("   This explains the shooting up behavior at x=0")
            elif max_coeff_idx < 5:
                print(f"⚠️  WARNING: One of the first {max_coeff_idx+1} coefficients is largest")
                print("   This could cause boundary issues")
            else:
                print("✓ Largest coefficient is not in the first few basis functions")
            
            # Check coefficient distribution
            print(f"Mean coefficient magnitude: {np.mean(abs_coeffs):.6f}")
            print(f"Std coefficient magnitude: {np.std(abs_coeffs):.6f}")
            print(f"Number of coefficients > 1.0: {np.sum(abs_coeffs > 1.0)}")
            print(f"Number of coefficients > 10.0: {np.sum(abs_coeffs > 10.0)}")
            
            # PEAK ANALYSIS
            print(f"\n=== PEAK ANALYSIS ===")
            
            # Find peaks in the eigenfunction magnitude
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(eigenfunction_magnitude, height=0.1*np.max(eigenfunction_magnitude))
            
            print(f"Number of peaks detected: {len(peaks)}")
            if len(peaks) > 0:
                peak_positions = physical_positions[peaks] / a  # Convert to x/a
                peak_heights = eigenfunction_magnitude[peaks]
                print(f"Peak positions (x/a): {peak_positions}")
                print(f"Peak heights: {peak_heights}")
                
                if len(peaks) == 2:
                    print("✓ TWO PEAKS DETECTED - This is the expected behavior!")
                    print("  Possible explanations:")
                    print("    1. Multiple resonant surfaces")
                    print("    2. Higher harmonic mode")
                    print("    3. Boundary effects")
                    print("    4. Mixed mode structure")
            
            # MULTIPLE RESONANT SURFACE ANALYSIS
            print(f"\n=== MULTIPLE RESONANT SURFACE ANALYSIS ===")
            
            # Calculate expected resonant surface
            lambda_scaled = eigval * alfven_scale
            sqrt_lambda_scaled = np.sqrt(lambda_scaled)
            print(f"Eigenvalue λ = {eigval:.6f}")
            print(f"√(λ * alfven_scale) = {sqrt_lambda_scaled:.6f}")
            print(f"This should equal |n + m/q(x_res)| = |{n} + {m}/q(x_res)|")
            
            # Solve for resonant q-value: n + m/q_res = ±sqrt_lambda_scaled
            q_res_positive = m / (sqrt_lambda_scaled - n) if (sqrt_lambda_scaled - n) != 0 else None
            q_res_negative = m / (-sqrt_lambda_scaled - n) if (-sqrt_lambda_scaled - n) != 0 else None
            
            print("Possible resonant q-values:")
            if q_res_positive is not None:
                print(f"  q_res (+): {q_res_positive:.6f}")
            if q_res_negative is not None:
                print(f"  q_res (-): {q_res_negative:.6f}")
            
            # Convert q-values to radial positions
            resonant_positions = []
            for q_res in [q_res_positive, q_res_negative]:
                if q_res is not None and q0 <= q_res <= q1:
                    x_over_a = np.sqrt((q_res - q0) / (q1 - q0))
                    resonant_positions.append(x_over_a)
                    print(f"  Predicted resonant position: x/a = {x_over_a:.6f}")
            
            # Check if we have multiple resonant surfaces
            if len(resonant_positions) == 2:
                print("✓ TWO RESONANT SURFACES PREDICTED!")
                print("  This explains the two peaks in the eigenfunction")
                print("  The mode is localized near both resonant surfaces")
            elif len(resonant_positions) == 1:
                print("⚠️  Only one resonant surface predicted")
                print("  The second peak might be due to:")
                print("    - Higher harmonic structure")
                print("    - Boundary effects")
                print("    - Numerical resolution issues")
            else:
                print("❌ No resonant surfaces predicted in domain")
                print("  Check if q-profile range covers the predicted q-values")
            
            # BOUNDARY EFFECT ANALYSIS
            print(f"\n=== BOUNDARY EFFECT ANALYSIS ===")
            
            # Check if peaks are near boundaries
            boundary_threshold = 0.1  # Within 10% of domain edge
            near_left_boundary = np.any(peak_positions < boundary_threshold) if len(peaks) > 0 else False
            near_right_boundary = np.any(peak_positions > (1 - boundary_threshold)) if len(peaks) > 0 else False
            
            if near_left_boundary:
                print("⚠️  Peak detected near left boundary (x/a ≈ 0)")
                print("  This could be a boundary effect from clamped BCs")
            if near_right_boundary:
                print("⚠️  Peak detected near right boundary (x/a ≈ 1)")
                print("  This could be a boundary effect from clamped BCs")
            
            if not (near_left_boundary or near_right_boundary):
                print("✓ Peaks are not near boundaries")
                print("  This suggests the peaks are physical (resonant surfaces)")
            
            # Create plot for eigenfunction magnitude
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot magnitude of entire eigenfunction
            ax.plot(physical_positions/a, eigenfunction_magnitude, 'k-', linewidth=3, label='|u| (Total magnitude)')
            
            # Mark resonant positions on plot
            for x_res in resonant_positions:
                ax.axvline(x=x_res, color='red', linestyle='--', alpha=0.7, 
                          label=f'Predicted resonance')
                ax.text(x_res, ax.get_ylim()[1] * 0.8, f'x/a = {x_res:.3f}', 
                       rotation=90, ha='right', va='top', fontsize=8, color='red')
            
            # Mark peak positions
            if len(peaks) > 0:
                ax.plot(peak_positions, peak_heights, 'go', markersize=8, label='Detected peaks')
                for i, (x_peak, y_peak) in enumerate(zip(peak_positions, peak_heights)):
                    ax.text(x_peak, y_peak * 1.1, f'Peak {i+1}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Normalized radial position (x/a)')
            ax.set_ylabel('|u| - Total displacement magnitude')
            ax.set_title(f'Mode {mode_index} Total Eigenfunction Magnitude (λ = {eigval:.6f})\n{len(peaks)} peaks detected')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Show the plot
            plt.tight_layout()
            plt.show()
            
         
            
        except Exception as e:
            print(f"❌ Error plotting eigenfunction: {e}")
            import traceback
            traceback.print_exc()
            return None


    # CONTINUUM EVALUATION
    print("\n===CONTINUUM EVALUATION ===")
    x_vals = np.linspace(0, a, 50) 
    
    q_vals = jax.vmap(q)(x_vals)
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
        
       

        differential_3form = DifferentialForm( 
            k=3,  
            ns=[8, 2, 2], 
            ps=[3, k_y, k_z], 
            types=['clamped', 'simple_fourier', 'simple_fourier']  )


        Q_3form = QuadratureRule(differential_3form, 7)  
        
        # Create divergence matrix for 2-forms to 3-forms
        div_matrix = LazyDerivativeMatrix(differential_2form, differential_3form, Q_3form, F=F)
        
        # Define divergence computation function using the matrix
        def compute_divergence_2form(eigvec):
            """Compute divergence of 2-form eigenfunction using LazyDerivativeMatrix"""
            # Apply divergence matrix to eigenfunction coefficients
            div_coeffs = div_matrix.M @ eigvec
            
            # Create discrete function for divergence (3-form)
            div_function = DiscreteFunction(div_coeffs, differential_3form)
            
            # Evaluate divergence at3 quadrature points (correct quadrature)
            div_values = jax.vmap(div_function)(Q_3form.x)
            
            # Compute L2 norm of divergence using 3-form weights and Jacobian
            # For 3-forms: ∫ div² * (1/J) * w dx
            Jj = jax.vmap(jacobian_determinant(F))(Q_3form.x)
            div_norm_squared = jnp.sum(jnp.abs(div_values)**2 * (1/Jj) * Q_3form.w)
            div_norm = jnp.sqrt(div_norm_squared)
            
            
            return div_norm, div_values, Q_3form.x
        
        # ANALYZE ALFVÉN CONTINUUM MODES
        if len(continuum_eigenvalues) > 0:
            print("\n=== ALFVÉN CONTINUUM MODES ===")
            print(f"{'Mode':<6} {'Index':<8} {'Eigenvalue':<12} {'Div2Form':<12}")
            print("-" * 50)
            
            # Track divergence values for all modes
            divergence_values = []
            
            # Analyze all Alfvén continuum modes
            for i, idx in enumerate(continuum_indices):
                eigval = continuum_eigenvalues[i]
                eigvec = eigvecs_normalized[:, idx]
                
                # Compute divergence using 2-form approach
                div_norm_2form, div_values_2form, x_div_2form = compute_divergence_2form(eigvec)
                divergence_values.append(div_norm_2form)
                
                # Check if this is a pure Alfvén mode (incompressible)
                if div_norm_2form < 0.3:
                    mode_type = "Pure Alfvén"
                elif div_norm_2form < 0.6:
                    mode_type = "Mixed"
                else:
                    mode_type = "Compressible"
                
                print(f"{i:<6} {idx:<8} {eigval:<12.3f} {div_norm_2form:<12.3f} {mode_type}")
     
            
            # Find the mode with the lowest divergence and eigenvalue > 0.1
            divergence_values_array = np.array(divergence_values)
            eigenvalue_mask = continuum_eigenvalues > 0.002
            
            if np.any(eigenvalue_mask):
                valid_divergence_values = divergence_values_array[eigenvalue_mask]
                valid_indices = np.where(eigenvalue_mask)[0]
                min_div_idx_in_valid = np.argmin(valid_divergence_values)
                min_div_idx = valid_indices[min_div_idx_in_valid]
                best_idx = continuum_indices[min_div_idx]
                best_eigval = continuum_eigenvalues[min_div_idx]
                best_div = divergence_values[min_div_idx]
                print(f"\n✓ Found mode with lowest divergence and eigenvalue > 0.1: index {best_idx}, eigenvalue {best_eigval:.6f}, divergence {best_div:.6f}")
            else:
                # If no mode has eigenvalue > 0.1, find the mode with lowest divergence overall
                min_div_idx = np.argmin(divergence_values)
                best_idx = continuum_indices[min_div_idx]
                best_eigval = continuum_eigenvalues[min_div_idx]
                best_div = divergence_values[min_div_idx]
                print(f"\n⚠️  No mode has eigenvalue > 0.1. Using mode with lowest divergence: index {best_idx}, eigenvalue {best_eigval:.6f}, divergence {best_div:.6f}")
            
           
            
            # Plot the mode with lowest divergence in continuum range
            print(f"\n=== PLOTTING MODE WITH LOWEST DIVERGENCE ===")
            min_div_idx_in_continuum = np.argmin(divergence_values)
            min_div_mode_idx = continuum_indices[min_div_idx_in_continuum]
            min_div_eigval = continuum_eigenvalues[min_div_idx_in_continuum]
            min_div_value = divergence_values[min_div_idx_in_continuum]
            print(f"Mode with lowest divergence: index {min_div_mode_idx}, eigenvalue {min_div_eigval:.6f}, divergence {min_div_value:.6f}")
            plot_eigenfunction(eigvecs_normalized[:, min_div_mode_idx], min_div_eigval, differential_2form, a, alfven_scale, min_div_mode_idx, C_np)
            
            # FIND FUNDAMENTAL MODE (1 PEAK)
            print(f"\n=== FINDING FUNDAMENTAL MODE (1 PEAK) ===")
            
            def count_peaks(eigvec, differential_2form, Q, F):
                """Count peaks in eigenfunction, excluding boundary effects"""
                # Create eigenfunction
                eigenfunction = DiscreteFunction(eigvec, differential_2form)
                
                # Evaluate on interior points (avoid boundaries)
                x_interior = np.linspace(0.1, 0.9, 50)  # Interior only
                y_fixed, z_fixed = 0.5, 0.5
                eval_points = np.array([[x, y_fixed, z_fixed] for x in x_interior])
                
                try:
                    eigenfunction_values = jax.vmap(eigenfunction)(eval_points)
                    eigenfunction_magnitude = np.sqrt(np.sum(np.abs(eigenfunction_values)**2, axis=1))
                    
                    # Find peaks
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(eigenfunction_magnitude, height=0.1*np.max(eigenfunction_magnitude))
                    
                    return len(peaks)
                except:
                    return -1  # Error
            
            # Check all modes in continuum for fundamental mode
            fundamental_mode_found = False
            fundamental_mode_idx = None
            fundamental_mode_eigval = None
            
            print("Checking all continuum modes for fundamental mode (1 peak):")
            print(f"{'Mode':<6} {'Index':<8} {'Eigenvalue':<12} {'Peaks':<8} {'Status':<15}")
            print("-" * 60)
            
            for i, idx in enumerate(continuum_indices):
                eigval = continuum_eigenvalues[i]
                eigvec = eigvecs_normalized[:, idx]
                
                # Count peaks
                n_peaks = count_peaks(eigvec, differential_2form, Q, F)
                
                if n_peaks == 1:
                    status = "FUNDAMENTAL ✓"
                    if not fundamental_mode_found:
                        fundamental_mode_found = True
                        fundamental_mode_idx = idx
                        fundamental_mode_eigval = eigval
                elif n_peaks == 0:
                    status = "Flat mode"
                elif n_peaks == 2:
                    status = "1st harmonic"
                elif n_peaks == 3:
                    status = "2nd harmonic"
                elif n_peaks == 4:
                    status = "3rd harmonic"
                elif n_peaks == 5:
                    status = "4th harmonic"
                else:
                    status = f"{n_peaks}th harmonic"
                
                print(f"{i:<6} {idx:<8} {eigval:<12.6f} {n_peaks:<8} {status:<15}")
            
            if fundamental_mode_found:
                print(f"\n✓ FUNDAMENTAL MODE FOUND!")
                print(f"  Mode index: {fundamental_mode_idx}")
                print(f"  Eigenvalue: {fundamental_mode_eigval:.6f}")
                print(f"  Plotting fundamental mode...")
                plot_eigenfunction(eigvecs_normalized[:, fundamental_mode_idx], fundamental_mode_eigval, differential_2form, a, alfven_scale, fundamental_mode_idx, C_np)
            else:
                print(f"\n❌ No fundamental mode (1 peak) found in continuum")
                print(f"  All modes are higher harmonics")
                print(f"  Try reducing resolution or changing boundary conditions")
            
            # CHECK LOWER EIGENVALUES (outside continuum)
            print(f"\n=== CHECKING LOWER EIGENVALUES FOR FUNDAMENTAL MODE ===")
            
            # Look at the lowest 10 eigenvalues overall
            lowest_indices = np.argsort(eigvals_normalized)[:10]
            
            print("Checking lowest 10 eigenvalues for fundamental mode:")
            print(f"{'Rank':<6} {'Index':<8} {'Eigenvalue':<12} {'Peaks':<8} {'In Continuum':<12}")
            print("-" * 65)
            
            for rank, idx in enumerate(lowest_indices):
                eigval = eigvals_normalized[idx]
                eigvec = eigvecs_normalized[:, idx]
                
                # Count peaks
                n_peaks = count_peaks(eigvec, differential_2form, Q, F)
                
                # Check if in continuum
                in_continuum = continuum_min <= eigval <= continuum_max
                
                print(f"{rank:<6} {idx:<8} {eigval:<12.6f} {n_peaks:<8} {'Yes' if in_continuum else 'No':<12}")
                
                # If we find a fundamental mode with low eigenvalue, plot it
                if n_peaks == 1 and eigval < 0.05:  # Low eigenvalue fundamental mode
                    print(f"  → Found low-eigenvalue fundamental mode!")
                    print(f"  Plotting this mode...")
                    plot_eigenfunction(eigvecs_normalized[:, idx], eigval, differential_2form, a, alfven_scale, idx, C_np)
                    break
            
            # Plot the Alfvén continuum
            print(f"\n=== PLOTTING ALFVÉN CONTINUUM ===")
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot continuum
            x_plot = np.linspace(0, 1, 100)  # Normalized x/a
            q_plot = q(x_plot * a)  # q-profile
            continuum_plot = (B0**2/R0**2) * (n + m/q_plot)**2 / alfven_scale
            
            ax.plot(x_plot, continuum_plot, color='darkblue', linewidth=2, label='Alfvén Continuum')
            
            # Track which categories we've already added to legend
            legend_added = {'Pure Alfvén': False, 'Mixed': False, 'Compressible': False}
            
            # Mark the eigenvalues in the continuum
            for i, idx in enumerate(continuum_indices):
                eigval = continuum_eigenvalues[i]
                div_val = divergence_values[i]
                if div_val < 0.3:
                    color = 'darkgreen'  # Pure Alfvén
                    label = 'Pure Alfvén'
                elif div_val < 0.6:
                    color = 'darkturquoise'  # Mixed
                    label = 'Mixed'
                else:
                    color = 'magenta'  # Compressible
                    label = 'Compressible'
                
                # Only add label to legend if we haven't added this category yet
                if not legend_added[label]:
                    ax.axhline(y=eigval, color=color, alpha=0.7, linestyle='--', label=label)
                    legend_added[label] = True
                else:
                    ax.axhline(y=eigval, color=color, alpha=0.7, linestyle='--')
            
            ax.set_xlabel('x')
            ax.set_ylabel('Normalized eigenvalue')
            ax.set_title('Alfvén Continuum and Eigenvalues')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            print(f"✓ Analysis complete!")
            
            # Additional diagnostics
            print(f"\n=== MATRIX CONDITIONING DIAGNOSTIC ===")
            print(f"Mass matrix condition number: {np.linalg.cond(C_np):.2e}")
            print(f"Hessian matrix condition number: {np.linalg.cond(H_np):.2e}")
            print(f"H̃ matrix condition number: {np.linalg.cond(H_tilde):.2e}")
            
          
            
                
              