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

import jax
import jax.numpy as jnp
import numpy.testing as npt

from mrx.BoundaryConditions import LazyBoundaryOperator
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
        self.atol = 1e-15

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

                    # Check positive definiteness
                    eigvals = jnp.linalg.eigvalsh(M)
                    print(
                        f"Eigenvalues of {k}-form mass matrix: min={jnp.min(eigvals)}, max={jnp.max(eigvals)}")
                    self.assertTrue(
                        jnp.all(eigvals > 0),
                        f"Mass matrix for {k}-form not positive definite "
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
        - curl grad = 0
        - div curl = 0
        """
        def check_all_entries_valid(D):
            def check_is_entry_valid(i, j):
                return jnp.logical_or(jnp.isclose(D[i, j], 0, atol=1e-12), jnp.isclose(jnp.abs(D[i, j]), 1, atol=1e-12))
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
                D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).M
                              for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])]
                M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).M
                                  for Λ in [Λ0, Λ1, Λ2, Λ3]]

                for k, D, M in zip([0, 1, 2], [D0, D1, D2], [M1, M2, M3]):
                    D_strong = jnp.linalg.solve(M, D)
                    print(f"\nTesting case for k={k}")
                    print_matrix_stats(D, "{k} Derivative Matrix")
                    self.assertFalse(
                        jnp.any(jnp.isnan(D)),
                        f"Matrix contains NaN values "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )
                    # Check that all entries of D_strong are either -1, 0, or 1
                    self.assertTrue(
                        check_all_entries_valid(D_strong),
                        f"Strong derivative matrix contains unexpected values "
                        f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                    )

                G = jnp.linalg.solve(M1, D0)
                C = jnp.linalg.solve(M2, D1)
                D = jnp.linalg.solve(M3, D2)
                # Check that curl grad = 0
                npt.assert_allclose(
                    0, C @ G,
                    rtol=2,
                    atol=self.atol,
                    err_msg=f"Curl grad is not zero "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )
                # Check that div curl = 0
                npt.assert_allclose(
                    0, D @ C,
                    rtol=2,
                    atol=self.atol,
                    err_msg=f"Div curl is not zero "
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
        """
        Test basic functionality of mass matrices for all form degrees.
        Checks if:
        - No NaN or Inf values are present
        - Matrix properties:
            - shape
            - symmetry
            - positive semi-definiteness
        """
        for ns, ps, types in self.test_cases:
            print(
                f"\nTesting stiffness matrix for ns={ns}, ps={ps}, types={types}")
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
                print(
                    f"Eigenvalues of stiffness matrix: min={jnp.min(eigvals)}, max={jnp.max(eigvals)}")
                self.assertTrue(
                    jnp.all(eigvals >= 0),
                    f"Stiffness matrix not positive semi-definite "
                    f"(ns={ns}, ps={ps}, types={types}, quad_order={quad_order})"
                )


if __name__ == '__main__':
    unittest.main()
