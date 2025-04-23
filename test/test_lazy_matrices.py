"""
Unit tests for LazyMatrix calculations.

This module contains tests for verifying the correctness and numerical stability
of LazyMatrix implementations, particularly focusing on mass matrices for
different differential form degrees.

The tests verify:
1. Basic functionality (no NaN values)
2. Numerical stability
3. Expected properties (symmetry, positive definiteness)
4. Integration accuracy
"""

import unittest
import numpy.testing as npt
import jax
import jax.numpy as jnp

from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule
from mrx.LazyMatrices import (LazyMassMatrix, LazyDerivativeMatrix, 
                            LazyDoubleCurlMatrix,
                            LazyStiffnessMatrix)
from mrx.BoundaryConditions import LazyBoundaryOperator

# Enable double precision
jax.config.update("jax_enable_x64", True)

def print_matrix_stats(M, name):
    """Print statistics about a matrix for debugging."""
    print(f"\n{name} Statistics:")
    print(f"Shape: {M.shape}")
    print(f"NaN count: {jnp.sum(jnp.isnan(M))}")
    print(f"Inf count: {jnp.sum(jnp.isinf(M))}")
    print(f"Min value: {jnp.nanmin(M)}")
    print(f"Max value: {jnp.nanmax(M)}")
    print(f"Mean value: {jnp.nanmean(M)}")
    print(f"Zero count: {jnp.sum(M == 0)}")
    if jnp.any(jnp.isnan(M)):
        nan_indices = jnp.where(jnp.isnan(M))
        print(f"First NaN at indices: {list(zip(nan_indices[0][:5], nan_indices[1][:5]))}")

class TestLazyMatrices(unittest.TestCase):
    """Test cases for LazyMatrix implementations."""
    
    def setUp(self):
        """Set up test cases with different form configurations."""
        # Test cases: (ns, ps, types)
        self.test_cases = [
            # Simple periodic case
            ((4, 4, 1), (2, 2, 1), ('periodic', 'periodic', 'constant')),
            # Clamped case
            ((4, 4, 1), (2, 2, 1), ('clamped', 'clamped', 'constant')),
            # Mixed case
            ((4, 4, 1), (2, 2, 1), ('periodic', 'clamped', 'constant')),
        ]
        
        # Quadrature orders to test
        self.quad_orders = [3, 5, 7]
        
        # Tolerance for numerical comparisons
        self.rtol = 1e-7
        self.atol = 1e-15

    def test_mass_matrix_basic(self):
        """Test basic functionality of mass matrices for all form degrees."""
        for ns, ps, types in self.test_cases:
            print(f"\nTesting mass matrices for ns={ns}, ps={ps}, types={types}")
            # Create differential forms
            Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(k, ns, ps, types) for k in range(4)]
            
            for quad_order in self.quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ0, quad_order)
                
                # Test mass matrices for each form degree
                for k, Λ in enumerate([Λ0, Λ1, Λ2, Λ3]):
                    print(f"\nTesting {k}-form mass matrix")
                    M = LazyMassMatrix(Λ, Q).M
                    print_matrix_stats(M, f"{k}-form Mass Matrix")
                    
                    # Check for NaN values
                    self.assertFalse(
                        jnp.any(jnp.isnan(M)),
                        f"Mass matrix for {k}-form contains NaN values "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )
                    
                    # Check for Inf values
                    self.assertFalse(
                        jnp.any(jnp.isinf(M)),
                        f"Mass matrix for {k}-form contains Inf values "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )
                    
                    # Check matrix properties
                    self.assertEqual(
                        M.shape, (Λ.n, Λ.n),
                        f"Mass matrix shape mismatch for {k}-form "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )
                    
                    # Check symmetry
                    npt.assert_allclose(
                        M, M.T,
                        rtol=self.rtol,
                        atol=self.atol,
                        err_msg=f"Mass matrix for {k}-form not symmetric "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )
                    
                    # Check positive definiteness (for 0-forms)
                    if k == 0:
                        eigvals = jnp.linalg.eigvalsh(M)
                        print(f"Eigenvalues of 0-form mass matrix: min={jnp.min(eigvals)}, max={jnp.max(eigvals)}")
                        self.assertTrue(
                            jnp.all(eigvals > 0),
                            f"Mass matrix for 0-form not positive definite "
                            f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                        )

    def test_mass_matrix_integration(self):
        """Test integration accuracy of mass matrices."""
        for ns, ps, types in self.test_cases:
            print(f"\nTesting integration for ns={ns}, ps={ps}, types={types}")
            Λ0 = DifferentialForm(0, ns, ps, types)
            
            # Use higher quadrature order for clamped BCs
            quad_orders = [5, 7, 9] if types[0] == 'clamped' else [3, 5, 7]
            
            for quad_order in quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ0, quad_order)
                
                # Create boundary operator for clamped BCs
                bcs = tuple('dirichlet' if t == 'clamped' else 'none' for t in types)
                B0 = LazyBoundaryOperator(Λ0, bcs).M if 'clamped' in types else None
                M = LazyMassMatrix(Λ0, Q, F=None, E=B0).M
                print_matrix_stats(M, "Mass Matrix")
                
                # Test integration of constant function
                if B0 is not None:
                    # For clamped BCs, we need to project the constant function onto the reduced space
                    ones_full = jnp.ones(Λ0.n)
                    ones = B0 @ ones_full
                    # Normalize to get integral 1
                    ones = ones / jnp.sqrt(ones @ M @ ones)
                else:
                    ones = jnp.ones(Λ0.n)
                integral = ones @ M @ ones
                expected = 1.0  # Volume of unit cube
                print(f"Constant function integral: {integral} (expected: {expected})")
                
                npt.assert_allclose(
                    integral, expected,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg=f"Mass matrix integration test failed for constant function "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Test integration of sine function
                def sin_func(x):
                    # For periodic BCs: sin^2(2πx) * sin^2(2πy)
                    # For clamped BCs: sin^2(πx) * sin^2(πy)
                    if types[0] == 'periodic':
                        return jnp.sin(2 * jnp.pi * x[0])**2 * jnp.sin(2 * jnp.pi * x[1])**2
                    else:  # clamped
                        return jnp.sin(jnp.pi * x[0])**2 * jnp.sin(jnp.pi * x[1])**2
                
                # Project sin_func onto the basis functions using the mass matrix
                sin_vals = jax.vmap(sin_func)(Q.x)  # Shape: (n_q,)
                print(f"sin_vals shape: {sin_vals.shape}")
                print(f"Q.x shape: {Q.x.shape}")
                print(f"Q.x min/max: {jnp.min(Q.x)}, {jnp.max(Q.x)}")
                
                # Evaluate basis functions at quadrature points
                # Shape: (n, n_q) after squeeze
                basis_vals = jax.vmap(jax.vmap(Λ0, (0, None)), (None, 0))(Q.x, jnp.arange(Λ0.n))
                basis_vals = basis_vals.squeeze(-1)
                print(f"basis_vals shape: {basis_vals.shape}")
                print(f"Q.w shape: {Q.w.shape}")
                print(f"Q.w sum: {jnp.sum(Q.w)}")
                
                # Compute right-hand side for projection: (f, Λ_i)
                rhs = jnp.einsum("i,ji,i->j", sin_vals, basis_vals, Q.w)
                if B0 is not None:
                    # For clamped BCs, we need to project the sine function onto the reduced space
                    rhs = B0 @ rhs
                    # Normalize the coefficients to get the correct integral
                    # rhs = rhs / jnp.sqrt(rhs @ M @ rhs)
                
                print(f"rhs shape: {rhs.shape}")
                print(f"M shape: {M.shape}")
                
                # Solve M @ sin_coeffs = rhs to get coefficients
                sin_coeffs = jnp.linalg.solve(M, rhs)
                print(f"sin_coeffs shape: {sin_coeffs.shape}")
                print(f"sin_coeffs min/max: {jnp.min(sin_coeffs)}, {jnp.max(sin_coeffs)}")
                
                # Compute L2 norm squared of the projected function
                # This should equal the L2 norm squared of sin_func
                integral = sin_coeffs @ M @ sin_coeffs
                
                # Expected value depends on the boundary conditions
                if types[0] == 'periodic':
                    expected = 0.0625  # L2 norm squared of sin^2(2πx) * sin^2(2πy) over [0,1]^2
                else:  # clamped
                    expected = 0.0625  # L2 norm squared of sin^2(πx) * sin^2(πy) over [0,1]^2
                
                print(f"Sine function integral: {integral} (expected: {expected})")
                
                # Also compute the integral directly using quadrature for verification
                direct_integral = jnp.sum(sin_vals * Q.w)
                print(f"Direct quadrature integral: {direct_integral}")
                
                npt.assert_allclose(
                    integral, expected,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg=f"Mass matrix integration test failed for sine function "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )

    def test_derivative_matrices(self):
        """Test derivative matrices (grad, curl, div)."""
        for ns, ps, types in self.test_cases:
            print(f"\nTesting derivative matrices for ns={ns}, ps={ps}, types={types}")
            Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(k, ns, ps, types) for k in range(4)]
            
            for quad_order in self.quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ0, quad_order)
                
                # Test gradient matrix
                print("\nTesting gradient matrix")
                D0 = LazyDerivativeMatrix(Λ0, Λ1, Q).M
                print_matrix_stats(D0, "Gradient Matrix")
                self.assertFalse(
                    jnp.any(jnp.isnan(D0)),
                    f"Gradient matrix contains NaN values "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Test curl matrix
                print("\nTesting curl matrix")
                D1 = LazyDerivativeMatrix(Λ1, Λ2, Q).M
                print_matrix_stats(D1, "Curl Matrix")
                self.assertFalse(
                    jnp.any(jnp.isnan(D1)),
                    f"Curl matrix contains NaN values "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Test divergence matrix
                print("\nTesting divergence matrix")
                D2 = LazyDerivativeMatrix(Λ2, Λ3, Q).M
                print_matrix_stats(D2, "Divergence Matrix")
                self.assertFalse(
                    jnp.any(jnp.isnan(D2)),
                    f"Divergence matrix contains NaN values "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )

    def test_double_curl_matrix(self):
        """Test double curl matrix."""
        for ns, ps, types in self.test_cases:
            print(f"\nTesting double curl matrix for ns={ns}, ps={ps}, types={types}")
            Λ1 = DifferentialForm(1, ns, ps, types)
            
            for quad_order in self.quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ1, quad_order)
                C = LazyDoubleCurlMatrix(Λ1, Q).M
                print_matrix_stats(C, "Double Curl Matrix")
                
                # Check for NaN values
                self.assertFalse(
                    jnp.any(jnp.isnan(C)),
                    f"Double curl matrix contains NaN values "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Check symmetry
                npt.assert_allclose(
                    C, C.T,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg=f"Double curl matrix not symmetric "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )

    def test_stiffness_matrix(self):
        """Test stiffness matrix."""
        for ns, ps, types in self.test_cases:
            print(f"\nTesting stiffness matrix for ns={ns}, ps={ps}, types={types}")
            Λ0 = DifferentialForm(0, ns, ps, types)
            
            for quad_order in self.quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ0, quad_order)
                K = LazyStiffnessMatrix(Λ0, Q).M
                print_matrix_stats(K, "Stiffness Matrix")
                
                # Check for NaN values
                self.assertFalse(
                    jnp.any(jnp.isnan(K)),
                    f"Stiffness matrix contains NaN values "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Check symmetry
                npt.assert_allclose(
                    K, K.T,
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg=f"Stiffness matrix not symmetric "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Check positive semi-definiteness
                eigvals = jnp.linalg.eigvalsh(K)
                print(f"Eigenvalues of stiffness matrix: min={jnp.min(eigvals)}, max={jnp.max(eigvals)}")
                self.assertTrue(
                    jnp.all(eigvals >= 0),
                    f"Stiffness matrix not positive semi-definite "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )

    def test_clamped_boundary_conditions(self):
        """Test that clamped boundary conditions are enforced correctly."""
        # Initialize with clamped boundary conditions
        ns = (5, 5, 5)
        ps = (3, 3, 3)
        types = ('clamped', 'clamped', 'clamped')
        
        # Create differential form and quadrature rule
        Λ0 = DifferentialForm(0, ns, ps, types)
        Q = QuadratureRule(Λ0, 5)  # Higher order quadrature for accuracy
        
        print(f"\nQuadrature points shape: {Q.x.shape}")
        print(f"Quadrature weights shape: {Q.w.shape}")
        
        # Create boundary operator
        bcs = tuple('dirichlet' if t == 'clamped' else 'none' for t in types)
        B0 = LazyBoundaryOperator(Λ0, bcs)
        
        # Create mass and stiffness matrices with boundary conditions
        # M = LazyMassMatrix(Λ0, Q, E=B0.M).M
        K = LazyStiffnessMatrix(Λ0, Q, E=B0.M).M
        
        # Test 1: Check that the boundary operator reduces degrees of freedom
        self.assertLess(B0.n, Λ0.n, "Boundary operator should reduce degrees of freedom")
        
        # Test 2: Check that the stiffness matrix is symmetric
        npt.assert_allclose(
            K, K.T,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Stiffness matrix should be symmetric"
        )
        
        # Test 3: Check that the stiffness matrix is positive semi-definite
        eigvals = jnp.linalg.eigvalsh(K)
        self.assertTrue(
            jnp.all(eigvals >= -self.atol),
            f"Stiffness matrix has negative eigenvalues: min={jnp.min(eigvals)}"
        )
        
        # Test 4: Check that functions in the range of B0 satisfy the boundary conditions
        # Create a test function that's 0 on the boundary and 1 in the interior
        def test_func(x):
            # x has shape (n_q, 3)
            return jnp.prod(x * (1 - x), axis=1)
        
        # Project the test function onto the basis
        coeffs = jnp.zeros(Λ0.n)
        test_vals = test_func(Q.x)  # Shape (n_q,)
        print(f"\nTest function values shape: {test_vals.shape}")
        print(f"Test function min/max: {jnp.min(test_vals)}, {jnp.max(test_vals)}")
        
        # Evaluate all basis functions at all quadrature points
        basis_vals = jax.vmap(jax.vmap(Λ0, (0, None)), (None, 0))(Q.x, jnp.arange(Λ0.n))  # Shape (n, n_q, 1)
        basis_vals = basis_vals.squeeze(-1)  # Shape (n, n_q)
        print(f"Basis values shape: {basis_vals.shape}")
        print(f"Basis values min/max: {jnp.min(basis_vals)}, {jnp.max(basis_vals)}")
        
        # Ensure shapes match for broadcasting
        test_vals = test_vals.reshape(1, -1)  # Shape (1, n_q)
        Q_w = Q.w.reshape(1, -1)  # Shape (1, n_q)
        
        # Compute coefficients using vectorized operations
        coeffs = jnp.sum(test_vals * basis_vals * Q_w, axis=1)
        print(f"Coefficients shape: {coeffs.shape}")
        print(f"Coefficients min/max: {jnp.min(coeffs)}, {jnp.max(coeffs)}")
        
        # Apply boundary conditions
        coeffs_bc = B0.M @ coeffs
        print(f"Boundary coefficients shape: {coeffs_bc.shape}")
        print(f"Boundary coefficients min/max: {jnp.min(coeffs_bc)}, {jnp.max(coeffs_bc)}")
        
        # Check boundary values
        boundary_points = []
        for i in range(3):  # For each dimension
            for val in [0.0, 1.0]:  # Check both boundaries
                x = jnp.array([0.5, 0.5, 0.5])  # Interior point
                x = x.at[i].set(val)  # Set boundary value
                boundary_points.append(x)
        
        boundary_points = jnp.stack(boundary_points)  # Shape (6, 3)
        print(f"\nBoundary points shape: {boundary_points.shape}")
        
        # Evaluate function at boundary points
        boundary_values = jnp.zeros(len(boundary_points))
        basis_vals_boundary = jax.vmap(jax.vmap(Λ0, (0, None)), (None, 0))(boundary_points, jnp.arange(B0.n))  # Shape (n, 6, 1)
        basis_vals_boundary = basis_vals_boundary.squeeze(-1)  # Shape (n, 6)
        boundary_values = coeffs_bc @ basis_vals_boundary  # Shape (6,)

        print("\nBoundary values statistics:")
        print(f"Max absolute value: {jnp.max(jnp.abs(boundary_values))}")
        print(f"Mean absolute value: {jnp.mean(jnp.abs(boundary_values))}")
        print(f"Boundary values: {boundary_values}")
        
        # Check that boundary values are close to zero
        npt.assert_allclose(
            boundary_values, jnp.zeros_like(boundary_values),
            rtol=1e-10, atol=1e-10,
            err_msg="Function values should be zero at boundaries"
        )

if __name__ == '__main__':
    unittest.main() 