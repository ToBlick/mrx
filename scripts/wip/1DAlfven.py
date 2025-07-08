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
import scipy.linalg
from mrx.Utils import inv33
from mrx.LazyMatrices import LazyMassMatrix, LazyMagneticTensionMatrix, LazyPressureGradientForceMatrix, LazyCurrentDensityMatrix, LazyWeightedDoubleDivergenceMatrix
import time
import matplotlib.pyplot as plt

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

differential_2form = DifferentialForm( 
    k=2,  
    ns=[8, 2, 2], 
    ps=[4, k_y, k_z], 
    types=['clamped', 'simple_fourier', 'simple_fourier']  
)

Q = QuadratureRule(differential_2form, 10)  

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
        """Pressure function that takes logical coordinates (0-form)"""
        x_coord = x_log[0] if jnp.ndim(x_log) > 0 else x_log # x-coordinate
        q_val = q_profile(x_coord)
        p_val = ((β*B0**2)/(2))*(1+(a/(q_val*R0))**2) + ((B0**2)*(a**2)/R0**2)*((1/(q0**2))-(1/q_val**2))
        return p_val

    # Equilibrium B₀ function
    def B0_field(x_log):
        """B-field function that takes logical coordinates (2-form)"""
        x_coord = x_log[0] if jnp.ndim(x_log) > 0 else x_log # x-coordinate
        # For now, we will use B(F(\hat{x})) as the appropriate two form for B
        q_val = q_profile(a*x_coord)

        # B-field
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
    K_magnetic_tension = LazyMagneticTensionMatrix(differential_2form, Q, B0_field, F=F, k_y=k_y, k_z=k_z)    
    print(f"Term 1: {time.time() - start_time:.2f}s")

    # Term 2: ∫ γ*p*(∇·φ_i)*(∇·φ_j)
    D_divdiv_weighted = LazyWeightedDoubleDivergenceMatrix(differential_2form, Q, pressure_func, F=F).M
    print(f"Term 2: {time.time() - start_time:.2f}s")

    # Term 3: ∫ φ_i · [(∇×B₀) × (∇×(φ_j × B₀))] dx (note(∇×B₀) is nearly zero)
    K_current_density = LazyCurrentDensityMatrix(differential_2form, Q, B0_field, F=F, k_y=k_y, k_z=k_z)
    print(f"Term 3: {time.time() - start_time:.2f}s")

    # Term 4: ∫ φ_i · ∇(φ_j · ∇p) dx
    K_pressure_gradient_force = LazyPressureGradientForceMatrix(differential_2form, Q, pressure_func, F=F)
    K_pressure_gradient_force = K_pressure_gradient_force.M
    print(f"Term 4: {time.time() - start_time:.2f}s")
    
    
    print(f"Total matrix computation time: {time.time() - start_time:.2f}s")
    
    return K_magnetic_tension, D_divdiv_weighted, K_current_density.M, K_pressure_gradient_force, C

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

    

    #  Alfvén continuum
    def w_2(x): return B0**2/R0**2 * (n + (m/q(x)))**2

    def plot_eigenfunction(eigvec, eigval, differential_2form, a, alfven_scale, mode_index):
        """Plot eigenfunction magnitude with proper 2-form to vector transformation."""
        print(f"\n=== PLOTTING EIGENFUNCTION FOR MODE {mode_index} ===")
        
        # RESONANCE CONDITION CHECK
        print("\n=== RESONANCE CONDITION CHECK ===")
        print(f"Eigenvalue λ = {eigval:.6f}")
        print(f"Mode numbers: m = {m}, n = {n}")
        
        # For Alfvén continuum: λ = (n + m/q(x_res))**2 / alfven_scale  
        # So: λ * alfven_scale = (n + m/q(x_res))**2
        # This gives: sqrt(λ * alfven_scale) = |n + m/q(x_res)|
        
        lambda_scaled = eigval * alfven_scale
        sqrt_lambda_scaled = np.sqrt(lambda_scaled)
        
        print(f"√(λ * alfven_scale) = {sqrt_lambda_scaled:.6f}")
        print(f"This should equal |n + m/q(x_res)| = |{n} + {m}/q(x_res)|")
        
        # Solve for resonant q-value: n + m/q_res = ±sqrt_lambda_scaled
        # So: m/q_res = ±sqrt_lambda_scaled - n
        # Therefore: q_res = m / (±sqrt_lambda_scaled - n)
        
        # Try both signs
        q_res_positive = m / (sqrt_lambda_scaled - n) if (sqrt_lambda_scaled - n) != 0 else None
        q_res_negative = m / (-sqrt_lambda_scaled - n) if (-sqrt_lambda_scaled - n) != 0 else None
        
        print("Possible resonant q-values:")
        if q_res_positive is not None:
            print(f"  q_res (+): {q_res_positive:.6f}")
        if q_res_negative is not None:
            print(f"  q_res (-): {q_res_negative:.6f}")
        
        # Check which q-value(s) are in the valid range [q0, q1]
        valid_q_values = []
        if q_res_positive is not None and q0 <= q_res_positive <= q1:
            valid_q_values.append(q_res_positive)
        if q_res_negative is not None and q0 <= q_res_negative <= q1:
            valid_q_values.append(q_res_negative)
        
        if len(valid_q_values) == 0:
            print(f"⚠️  No resonant q-value in valid range [{q0:.3f}, {q1:.3f}]")
            resonant_positions = []
        else:
            print(f"Valid resonant q-value(s): {valid_q_values}")
            
            # Convert q-values to radial positions using: q(x) = q0 + (q1-q0)*(x/a)**2
            # So: x/a = sqrt((q - q0)/(q1 - q0))
            resonant_positions = []
            for q_res in valid_q_values:
                x_over_a = np.sqrt((q_res - q0) / (q1 - q0))
                x_physical = x_over_a * a
                resonant_positions.append((x_physical, x_over_a, q_res))
                print(f"  Predicted resonant position: x = {x_physical:.6f}, x/a = {x_over_a:.6f}")
                
                # Verify by evaluating continuum at this position
                w2_at_resonance = w_2(x_physical)
                w2_normalized = w2_at_resonance / alfven_scale
                print(f"  Continuum frequency at x = {x_physical:.6f}: ω²/ω_A² = {w2_normalized:.6f}")
                print(f"  Match with eigenvalue: {np.abs(w2_normalized - eigval):.2e}")
        
        def oneform_to_vector(oneform_values, x_logical):
            """Transform 1-form to vector field using the metric tensor."""
            # Compute Jacobian at this point
            DF = jax.jacfwd(F)(x_logical)
            
            # Metric tensor in logical coordinates: g = DF^T DF
            g = DF.T @ DF
            g_inv = inv33(g)
            
            # Transform 1-form to vector using inverse metric: v = g^(-1) v_α
            vector_logical = g_inv @ oneform_values
            
            # Transform to physical coordinates: v_phys = DF v_logical  
            vector_physical = DF @ vector_logical
            
            return vector_physical

        
        # Create fine grid for plotting (logical coordinates)
        n_plot = 100
        x_min_safe = 0.0   # Include x = 0
        x_max_safe = 1.0   # Include up to x = 1.0
        x_logical_plot = np.linspace(x_min_safe, x_max_safe, n_plot)
        y_fixed = 0.5
        z_fixed = 0.5
        
        # Create eigenfunction from coefficients
        eigenfunction_1form = DiscreteFunction(eigvec, differential_2form)
        
        print("\nComputing eigenfunction on full domain x=0 to x=1...")
        
        try:
            # Evaluate and transform each point
            vector_components = []
            physical_positions = []
            
            for x_log in x_logical_plot:
                x_logical = jnp.array([x_log, y_fixed, z_fixed])
                
                # Evaluate 1-form at this logical point  
                oneform_values = eigenfunction_1form(x_logical)
                
                # Transform to physical vector field
                vector_phys = oneform_to_vector(oneform_values, x_logical)
                vector_components.append(vector_phys)
                
                # Get physical position
                x_physical = F(x_logical)
                physical_positions.append(x_physical[0])  # x-coordinate only
            
            vector_components = np.array(vector_components)
            physical_positions = np.array(physical_positions)
            
            # Extract physical displacement components
            u_x_plot = vector_components[:, 0]  # Radial 
            u_y_plot = vector_components[:, 1]  # Poloidal
            u_z_plot = vector_components[:, 2]  # Toroidal 
            eigenfunction_norm_plot = np.sqrt(np.abs(u_x_plot)**2 + np.abs(u_y_plot)**2 + np.abs(u_z_plot)**2)
            
        
            # Create simple magnitude plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot just the eigenfunction magnitude
            ax.plot(physical_positions/a, eigenfunction_norm_plot, 'k-', linewidth=2, label='|u|')
            
            # Mark predicted resonant positions
            if len(resonant_positions) > 0:
                for i, (x_phys, x_norm, q_res) in enumerate(resonant_positions):
                    ax.axvline(x=x_norm, color='red', linestyle='--', alpha=0.7, 
                              label=f'Predicted resonance {i+1}' if i == 0 else '')
                    # Add text annotation
                    ax.text(x_norm, ax.get_ylim()[1] * 0.8, f'x/a = {x_norm:.3f}\nq = {q_res:.3f}', 
                           rotation=90, ha='right', va='top', fontsize=8, color='red')
    
            ax.set_xlabel('Normalized radial position (x/a)')
            ax.set_ylabel('|u| - Displacement magnitude')
            ax.set_title(f'Mode {mode_index} Eigenfunction Magnitude (λ = {eigval:.6f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'alfven_mode_{mode_index}_eigenfunction.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ Eigenfunction plot saved as 'alfven_mode_{mode_index}_eigenfunction.png'")
            
        except Exception as e:
            print(f"❌ Error plotting eigenfunction: {e}")
            import traceback
            traceback.print_exc()
            return None


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
            eigenfunction = DiscreteFunction(eigvec, differential_2form)
            
            # Create evaluation points for divergence computation
            n_div_points = 50
            x_min_safe = 0.0   
            x_max_safe = 1.0
            x_logical = np.linspace(x_min_safe, x_max_safe, n_div_points)  
            x_physical = x_logical * a
            y_fixed = 0.5 #Using for now
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
            
            # Track divergence values for all modes
            divergence_values = []
            
            # Analyze all Alfvén continuum modes
            for i, idx in enumerate(continuum_indices):
                eigval = continuum_eigenvalues[i]
                eigvec = eigvecs_full[:, idx]
                
                # Compute 1D divergence
                div_norm_1d, div_values_1d, x_div_1d = compute_divergence_1d(eigvec)
                divergence_values.append(div_norm_1d)
                
                # Check if this is a pure Alfvén mode (incompressible)
                if div_norm_1d < 0.3:
                    mode_type = "Pure Alfvén"
                elif div_norm_1d < 0.6:
                    mode_type = "Mixed"
                else:
                    mode_type = "Compressible"
                
                print(f"{i:<6} {idx:<8} {eigval:<12.3f} {div_norm_1d:<12.3f} {mode_type}")
                
                # Check for degeneracy with previous mode
                if i > 0:
                    prev_idx = continuum_indices[i-1]
                    prev_eigval = continuum_eigenvalues[i-1]
                    prev_eigvec = eigvecs_full[:, prev_idx]
                 
            
            # Find the mode with the lowest divergence
            min_div_idx = np.argmin(divergence_values)
            best_idx = continuum_indices[min_div_idx]
            best_eigval = continuum_eigenvalues[min_div_idx]
            best_div = divergence_values[min_div_idx]
            
            print("\n=== PLOTTING LOWEST DIVERGENCE MODE ===")
            print(f"Mode with lowest divergence: index {best_idx}, eigenvalue {best_eigval:.6f}, divergence {best_div:.3f}")
            plot_eigenfunction(eigvecs_full[:, best_idx], best_eigval, differential_2form, a, alfven_scale, best_idx)
                
              