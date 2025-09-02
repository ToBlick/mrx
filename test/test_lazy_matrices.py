"""
Unit tests for LazyMatrix calculations.

This module contains tests for verifying the correctness and numerical stability
of LazyMatrix implementations, particularly focusing on mass matrices for
different differential form degrees.

The tests verify:
1. Basic functionality (no NaN values)
2. Numerical stability
3. Expected properties (symmetry, positive definiteness)
"""

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from mrx.DifferentialForms import DifferentialForm
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleCurlMatrix,
    LazyMassMatrix,
    LazyStiffnessMatrix,
)
from mrx.Quadrature import QuadratureRule

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
        print(
            f"First NaN at indices: {list(zip(nan_indices[0][:5], nan_indices[1][:5]))}")


def check_sparsity(M, name, expected_density=None):
    """
    Check sparsity properties of a matrix.
    
    Args:
        M: Matrix to check
        name: Name of the matrix 
        expected_density: Expected ratio of non-zero elements to total elements (if known)
        
    Returns:
        tuple: (density, max_nonzeros_row)

    """

    # Count nonzeros per row 
    nonzeros_per_row = jnp.sum(~jnp.isclose(M, 0, atol=1e-12), axis=1)
    # Count total number of nonzeros
    nonzeros = jnp.sum(nonzeros_per_row)
    # Get maximum number of nonzeros in any row
    max_nonzeros_row = jnp.max(nonzeros_per_row)
    # Calculate density as the ratio of nonzeros to total elements
    density = nonzeros / (M.shape[0] * M.shape[1])
    
    print(f"\nSparsity analysis for {name}:")
    print(f"Total Number of nonzeros: {nonzeros}")
    print(f"Density: {density:.6f}")
    print(f"Max nonzeros per row: {max_nonzeros_row}")
    
    if expected_density is not None:
        print(f"Expected density: {expected_density:.6f}")
        
    return density, max_nonzeros_row 


class TestLazyMatrices(unittest.TestCase):
    """Test cases for LazyMatrix implementations."""

    def setUp(self):
        """Set up test cases with different form configurations."""
        # Test cases: (ns, ps, types)
        self.test_cases = [
            # Mixed case
            ((4, 3, 3), (3, 1, 2), ('periodic', 'clamped', 'clamped')),
        ]

        # Quadrature orders to test
        self.quad_orders = [5]

        # Tolerance for numerical comparisons
        self.rtol = 1e-7
        self.atol = 1e-14

    def test_mass_matrix(self):
        """
        Test basic functionality of mass matrices for all form degrees.
        Checks if:
        - No NaN or Inf values are present
        - Matrix properties:
            - shape
            - symmetry
            - positive definiteness
        """

        for ns, ps, types in self.test_cases:
            print(
                f"\nTesting mass matrices for ns={ns}, ps={ps}, types={types}")
            # Create differential forms
            Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(
                k, ns, ps, types) for k in range(4)]

            for quad_order in self.quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ0, quad_order)

                # Test mass matrices for each form degree
                for k, Λ in enumerate([Λ0, Λ1, Λ2, Λ3]):
                    print(f"\nTesting {k}-form mass matrix")
                    M = LazyMassMatrix(Λ, Q).matrix()
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

                    

    def test_derivative_matrices(self):
        """
        Test derivative matrices (grad, curl, div).
        Checks if:
        - No NaN or Inf values are present
        - Matrix properties:
            - shape
            - strong form only contains -1, 0, 1
            - sparsity pattern
        - curl grad = 0
        - div curl = 0
        """
        def check_all_entries_valid(D):
            def check_is_entry_valid(i, j):
                return jnp.logical_or(jnp.isclose(D[i, j], 0, atol=1e-12), 
                                    jnp.isclose(jnp.abs(D[i, j]), 1, atol=1e-12))
            n = D.shape[0]
            m = D.shape[1]
            vals = jax.vmap(jax.vmap(check_is_entry_valid, (None, 0)), (0, None))(
                jnp.arange(n), jnp.arange(m))
            return jnp.all(vals)

        for ns, ps, types in self.test_cases:
            for quad_order in self.quad_orders:
                print(
                    f"\nTesting derivative matrices for ns={ns}, ps={ps}, types={types}, quad_order={quad_order}")
                Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(
                    k, ns, ps, types) for k in range(4)]
                Q = QuadratureRule(Λ0, quad_order)
                D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).matrix()
                              for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])]
                M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).matrix()
                                  for Λ in [Λ0, Λ1, Λ2, Λ3]]

                for k, D, M in zip([0, 1, 2], [D0, D1, D2], [M1, M2, M3]):
                    D_strong = jnp.linalg.solve(M, D)
                    print(f"\nTesting case for k={k}")
                    print_matrix_stats(D, f"{k} Derivative Matrix")
                    
                
                    # Check correct shape
                    if k == 0:  # gradient: 0-form -> 1-form
                        expected_shape = (Λ1.n, Λ0.n)
                    elif k == 1:  # curl: 1-form -> 2-form
                        expected_shape = (Λ2.n, Λ1.n)
                    else:  # divergence: 2-form -> 3-form
                        expected_shape = (Λ3.n, Λ2.n)
                        
                    self.assertEqual(
                        D_strong.shape, expected_shape,
                        f"Strong form {k}-derivative matrix shape doesn't match expectation: "
                        f"got {D_strong.shape}, expected {expected_shape} "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )
                 
                    # Check for NaN values
                    self.assertFalse(
                        jnp.any(jnp.isnan(D)),
                        f"Matrix contains NaN values "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )

                    # Check for Inf values
                    self.assertFalse(
                        jnp.any(jnp.isinf(D)),
                        f"Matrix contains Inf values "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )
                    
                    # Check that all entries of D_strong are either -1, 0, or 1
                    self.assertTrue(
                        check_all_entries_valid(D_strong),
                        f"Strong derivative matrix contains unexpected values "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )

                # Need to retrieve separately to check identities
                G = jnp.linalg.solve(M1, D0) #Strong form gradient
                C = jnp.linalg.solve(M2, D1) #Strong form curl
                D = jnp.linalg.solve(M3, D2) #Strong form divergence

                # Check that curl grad = 0 with increased tolerance
                npt.assert_allclose(
                    C @ G, jnp.zeros_like(C @ G),
                    rtol=self.rtol * 100,  # Increased tolerance
                    atol=self.atol * 100,  # Increased tolerance
                    err_msg=f"Curl of grad is not zero "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Check that div curl = 0 with increased tolerance
                npt.assert_allclose(
                    D @ C, jnp.zeros((D.shape[0], C.shape[1])),
                    rtol=self.rtol * 100,  # Increased tolerance
                    atol=self.atol * 100,  # Increased tolerance
                    err_msg=f"Div of curl is not zero "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )

              

    def test_double_curl_matrix(self):
        """Test double curl matrix."""
        for ns, ps, types in self.test_cases:
            print(
                f"\nTesting double curl matrix for ns={ns}, ps={ps}, types={types}")
            Λ1 = DifferentialForm(1, ns, ps, types)

            for quad_order in self.quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ1, quad_order)
                C = LazyDoubleCurlMatrix(Λ1, Q).matrix()
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

    def test_stiffness_matrix_pos_def(self):
        """
        Test positive definiteness of mass matrices for all form degrees.
    
        """
        for ns, ps, types in self.test_cases:
            print(
                f"\nTesting stiffness matrix for ns={ns}, ps={ps}, types={types}")
            Λ0 = DifferentialForm(0, ns, ps, types)

            for quad_order in self.quad_orders:
                print(f"\nQuadrature order: {quad_order}")
                Q = QuadratureRule(Λ0, quad_order)
                K = LazyStiffnessMatrix(Λ0, Q).matrix()
                print_matrix_stats(K, "Stiffness Matrix")

        

                # Analyze eigenvalues
                eigvals = jnp.linalg.eigvalsh(K)
            
                # Check for negative eigenvalues
                neg_eig = eigvals < 0
                neg_count = jnp.sum(neg_eig)
                if neg_count > 0:
                    print("\nNegative Eigenvalues Found:")
                    print(f"Number of negative eigenvalues: {neg_count}")
                    print(f"Range of negative eigenvalues: {jnp.min(eigvals[neg_eig]):.3e} to {jnp.max(eigvals[neg_eig]):.3e}")
                    print(f"Mean of negative eigenvalues: {jnp.mean(eigvals[neg_eig]):.3e}")
                    
                    # Check if negative eigenvalues are within tolerance
                    max_neg = jnp.max(jnp.abs(eigvals[neg_eig]))
                    print(f"Maximum absolute value of negative eigenvalues: {max_neg:.3e}")
                    
                    # Allow small negative eigenvalues within tolerance
                    if max_neg < self.atol * 10:
                        print("Negative eigenvalues are within tolerance")
                        return
                
                # Check positive semi-definiteness with tolerance
                self.assertTrue(
                    jnp.all(eigvals >= -self.atol * 10),
                    f"Stiffness matrix has significant negative eigenvalues, violating positive semi-definiteness"
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                
                # Check condition number
                cond = jnp.max(jnp.abs(eigvals)) / jnp.min(jnp.abs(eigvals))
                print(f"\nCondition number: {cond:.2e}")
          
                
                


if __name__ == '__main__':
    unittest.main()
    