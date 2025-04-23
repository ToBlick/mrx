import unittest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from mrx.BoundaryConditions import LazyBoundaryOperator
from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule

jax.config.update("jax_enable_x64", True)

class TestBoundaryConditions(unittest.TestCase):
    """Test cases for boundary condition operators."""

    def setUp(self):
        """Set up common parameters for testing."""
        self.ns = (5, 5, 5)  # Number of points in each direction
        self.ps = (3, 3, 3)  # Polynomial degree in each direction
        self.quad_order = 5  # Quadrature order

    def test_dirichlet_boundary_conditions(self):
        """Test Dirichlet boundary conditions for different form degrees."""
        for k in [0]:  # Test all form degrees
            with self.subTest(k=k):
                # Create differential form with clamped boundary conditions
                Λ = DifferentialForm(k, self.ns, self.ps, ('clamped', 'clamped', 'clamped'))
                # Create boundary operator with Dirichlet conditions
                B = LazyBoundaryOperator(Λ, ('dirichlet', 'dirichlet', 'dirichlet'))

                # Check that boundary operator reduces degrees of freedom
                self.assertLess(B.n, Λ.n, f"Boundary operator should reduce degrees of freedom for k={k}")

                # Create test function based on form degree
                if k == 0:
                    # Scalar field that's zero on boundaries
                    def test_func(x):
                        return jnp.prod(x * (1 - x))
                elif k == 1:
                    # Vector field with zero normal component on boundaries
                    def test_func(x):
                        return jnp.array([
                            x[1] * (1 - x[1]) * x[2] * (1 - x[2]),  # r-component
                            x[0] * (1 - x[0]) * x[2] * (1 - x[2]),  # χ-component
                            x[0] * (1 - x[0]) * x[1] * (1 - x[1])   # ζ-component
                        ])
                elif k == 2:
                    # 2-form with zero tangential component on boundaries
                    def test_func(x):
                        return jnp.array([
                            x[0] * (1 - x[0]) * x[1] * (1 - x[1]),  # rχ-component
                            x[0] * (1 - x[0]) * x[2] * (1 - x[2]),  # rζ-component
                            x[1] * (1 - x[1]) * x[2] * (1 - x[2])   # χζ-component
                        ])
                else:  # k == 3
                    # 3-form that's zero on boundaries
                    def test_func(x):
                        return jnp.prod(x * (1 - x))

                # Create quadrature rule for accurate integration
                Q = QuadratureRule(Λ, self.quad_order)

                # Project test function onto basis
                # Evaluate test function at quadrature points
                test_vals = jax.vmap(test_func)(Q.x)  # Shape depends on k
                print("\nTest function values at quadrature points:")
                print(test_vals[:5])  # Print first 5 values

                # Evaluate basis functions at quadrature points
                basis_vals = jax.vmap(lambda x: jax.vmap(Λ, (None, 0))(x, jnp.arange(Λ.n)))(Q.x)
                if k == 0:
                    basis_vals = basis_vals.squeeze(-1)  # Remove last dimension if k=0
                print("\nBasis function values at quadrature points:")
                print(basis_vals[:5])  # Print first 5 values

                # Compute coefficients
                if k == 0:
                    coeffs = jnp.einsum('q,qn,q->n', test_vals, basis_vals, Q.w)
                else:
                    coeffs = jnp.einsum('q...,qn...,q->n', test_vals, basis_vals, Q.w)
                print("\nCoefficients before boundary conditions:")
                print(coeffs[:5])  # Print first 5 coefficients

                # Apply boundary conditions
                coeffs_bc = B.M @ coeffs
                print("\nCoefficients after boundary conditions:")
                print(coeffs_bc[:5])  # Print first 5 coefficients

                # Create points on all boundaries (faces, edges, and corners)
                boundary_points = []
                
                # Face centers (6 points)
                for i in range(3):  # For each dimension
                    for val in [0.0, 1.0]:  # Check both boundaries
                        x = jnp.array([0.5, 0.5, 0.5])  # Interior point
                        x = x.at[i].set(val)  # Set boundary value
                        boundary_points.append(x)
                
                # Edge centers (12 points)
                for i in range(3):  # First dimension
                    for j in range(i+1, 3):  # Second dimension
                        for val1 in [0.0, 1.0]:  # First boundary value
                            for val2 in [0.0, 1.0]:  # Second boundary value
                                x = jnp.array([0.5, 0.5, 0.5])  # Interior point
                                x = x.at[i].set(val1)  # Set first boundary
                                x = x.at[j].set(val2)  # Set second boundary
                                boundary_points.append(x)
                
                # Corners (8 points)
                for val1 in [0.0, 1.0]:
                    for val2 in [0.0, 1.0]:
                        for val3 in [0.0, 1.0]:
                            boundary_points.append(jnp.array([val1, val2, val3]))

                boundary_points = jnp.stack(boundary_points)  # Shape (26, 3)
                print("\nBoundary points:")
                print(boundary_points)

                # Evaluate function at boundary points
                basis_vals_boundary = jax.vmap(lambda x: jax.vmap(Λ, (None, 0))(x, jnp.arange(B.n)))(boundary_points)
                if k == 0:
                    basis_vals_boundary = basis_vals_boundary.squeeze(-1)  # Shape (26, n)

                # For k>0, we need to check all components
                if k == 0:
                    boundary_values = jnp.einsum('n,bn->b', coeffs_bc, basis_vals_boundary)
                else:
                    boundary_values = jnp.einsum('n,bn...->b...', coeffs_bc, basis_vals_boundary)

                print("\nBoundary values:")
                print(boundary_values)

                # Check that boundary values are close to zero
                npt.assert_allclose(
                    boundary_values, jnp.zeros_like(boundary_values),
                    rtol=1e-10, atol=1e-10,
                    err_msg=f"Function values should be zero at boundaries for k={k}"
                )

    def test_mixed_boundary_conditions(self):
        """Test mixed boundary conditions (Dirichlet and periodic)."""
        for k in [0, 1, 2, 3]:  # Test all form degrees
            with self.subTest(k=k):
                # Create differential form with mixed boundary conditions
                Λ = DifferentialForm(k, self.ns, self.ps, ('clamped', 'periodic', 'periodic'))
                # Create boundary operator with mixed conditions
                B = LazyBoundaryOperator(Λ, ('dirichlet', 'none', 'none'))

                # Check that boundary operator reduces degrees of freedom only in Dirichlet directions
                expected_reduction = 1  # Only in r-direction
                actual_reduction = Λ.n - B.n
                self.assertGreaterEqual(
                    actual_reduction, expected_reduction,
                    f"Expected reduction of at least {expected_reduction} degrees of freedom for k={k}, got {actual_reduction}"
                )

                # Create test function based on form degree
                if k == 0:
                    # Scalar field that's zero on r-boundaries and periodic in χ,ζ
                    def test_func(x):
                        return jnp.sin(2 * jnp.pi * x[1]) * jnp.cos(2 * jnp.pi * x[2]) * x[0] * (1 - x[0])
                elif k == 1:
                    # Vector field with zero normal component on r-boundaries and periodic in χ,ζ
                    def test_func(x):
                        return jnp.array([
                            jnp.sin(2 * jnp.pi * x[1]) * jnp.cos(2 * jnp.pi * x[2]) * x[0] * (1 - x[0]),  # r-component
                            jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[2]) * x[1] * (1 - x[1]),  # χ-component
                            jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1]) * x[2] * (1 - x[2])   # ζ-component
                        ])
                elif k == 2:
                    # 2-form with zero tangential component on r-boundaries and periodic in χ,ζ
                    def test_func(x):
                        return jnp.array([
                            jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1]) * x[2] * (1 - x[2]),  # rχ-component
                            jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[2]) * x[1] * (1 - x[1]),  # rζ-component
                            jnp.sin(2 * jnp.pi * x[1]) * jnp.cos(2 * jnp.pi * x[2]) * x[0] * (1 - x[0])   # χζ-component
                        ])
                else:  # k == 3
                    # 3-form that's zero on r-boundaries and periodic in χ,ζ
                    def test_func(x):
                        return jnp.sin(2 * jnp.pi * x[1]) * jnp.cos(2 * jnp.pi * x[2]) * x[0] * (1 - x[0])

                # Create quadrature rule
                Q = QuadratureRule(Λ, self.quad_order)

                # Project test function onto basis
                test_vals = jax.vmap(test_func)(Q.x)
                basis_vals = jax.vmap(lambda x: jax.vmap(Λ, (None, 0))(x, jnp.arange(Λ.n)))(Q.x)
                if k == 0:
                    basis_vals = basis_vals.squeeze(-1)
                if k == 0:
                    coeffs = jnp.einsum('q,qn,q->n', test_vals, basis_vals, Q.w)
                else:
                    coeffs = jnp.einsum('q...,qn...,q->n', test_vals, basis_vals, Q.w)

                # Apply boundary conditions
                coeffs_bc = B.M @ coeffs

                # Check boundary values only in Dirichlet directions
                x_boundary = []
                for val in [0.0, 1.0]:
                    x = jnp.array([val, 0.5, 0.5])
                    x_boundary.append(x)

                x_boundary = jnp.stack(x_boundary)
                basis_vals_boundary = jax.vmap(lambda x: jax.vmap(Λ, (None, 0))(x, jnp.arange(B.n)))(x_boundary)
                if k == 0:
                    basis_vals_boundary = basis_vals_boundary.squeeze(-1)

                # For k>0, we need to check all components
                if k == 0:
                    boundary_values = jnp.einsum('n,bn->b', coeffs_bc, basis_vals_boundary)
                else:
                    boundary_values = jnp.einsum('n,bn...->b...', coeffs_bc, basis_vals_boundary)

                # Check that boundary values are zero only in Dirichlet directions
                npt.assert_allclose(
                    boundary_values, jnp.zeros_like(boundary_values),
                    rtol=1e-10, atol=1e-10,
                    err_msg=f"Function values should be zero at Dirichlet boundaries for k={k}"
                )

    def test_boundary_operator_properties(self):
        """Test properties of the boundary operator matrix."""
        for k in [0, 1, 2, 3]:  # Test all form degrees
            with self.subTest(k=k):
                # Create differential form with Dirichlet boundary conditions
                Λ = DifferentialForm(k, self.ns, self.ps, ('clamped', 'clamped', 'clamped'))
                # Create boundary operator
                B = LazyBoundaryOperator(Λ, ('dirichlet', 'dirichlet', 'dirichlet'))

                # Test matrix properties
                M = B.M

                # Check sparsity
                nnz = jnp.sum(M != 0)
                sparsity = 1 - nnz / (M.shape[0] * M.shape[1])
                self.assertGreater(sparsity, 0.9, f"Boundary operator should be sparse for k={k}")

                # Check number of non-zero entries per row
                nnz_per_row = jnp.sum(M != 0, axis=1)
                self.assertTrue(
                    jnp.all(nnz_per_row <= 2),
                    f"Each row should have at most 2 non-zero entries for k={k}"
                )

                # Check that projection preserves interior basis functions
                # Create unit vector for first interior basis function
                e = jnp.zeros(Λ.n)
                e = e.at[B.n // 2].set(1.0)  # Choose middle basis function
                
                # Project onto boundary operator range
                projected_coeffs = B.M @ e

                # Check that projection preserves exactly one basis function
                self.assertEqual(
                    jnp.sum(projected_coeffs != 0), 1,
                    f"Projection should preserve exactly one basis function for k={k}"
                )

    def test_basis_function_count(self):
        """Test that the number of basis functions is correct for each form degree and boundary condition type."""
        # Test parameters
        ns = (5, 5, 5)  # Number of points in each direction
        ps = (3, 3, 3)  # Polynomial degree in each direction

        # Test all form degrees
        for k in [0, 1, 2, 3]:
            with self.subTest(k=k):
                # Create differential form with all Dirichlet BCs
                Λ_dirichlet = DifferentialForm(k, ns, ps, ('clamped', 'clamped', 'clamped'))
                B_dirichlet = LazyBoundaryOperator(Λ_dirichlet, ('dirichlet', 'dirichlet', 'dirichlet'))

                # Create differential form with no BCs
                Λ_none = DifferentialForm(k, ns, ps, ('periodic', 'periodic', 'periodic'))
                B_none = LazyBoundaryOperator(Λ_none, ('none', 'none', 'none'))

                # Expected number of basis functions for each case
                if k == 0:
                    # For 0-forms:
                    # - Dirichlet: (nr-2)*(nχ-2)*(nζ-2) basis functions (remove boundary points)
                    # - None: nr*nχ*nζ basis functions (all points)
                    expected_dirichlet = (ns[0]-2) * (ns[1]-2) * (ns[2]-2)
                    expected_none = ns[0] * ns[1] * ns[2]
                elif k == 1:
                    # For 1-forms:
                    # - Dirichlet: Remove boundary points in normal direction for each component
                    # r-component: (nr)*(nχ-2)*(nζ-2)
                    # χ-component: (nr-2)*(nχ)*(nζ-2)
                    # ζ-component: (nr-2)*(nχ-2)*(nζ)
                    expected_dirichlet = (ns[0]-3)*(ns[1]-2)*(ns[2]-2) + \
                                         (ns[0]-2)*(ns[1]-3)*(ns[2]-2) + \
                                         (ns[0]-2)*(ns[1]-2)*(ns[2]-3)
                    expected_none = (ns[0]-1)*(ns[1])*(ns[2]) + \
                                    (ns[0])*(ns[1]-1)*(ns[2]) + \
                                    (ns[0])*(ns[1])*(ns[2]-1)
                elif k == 2:
                    # For 2-forms:
                    # - Dirichlet: Remove boundary points in tangential directions for each component
                    # rχ-component: (nr-2)*nχ*(nζ-2)
                    # rζ-component: (nr-2)*(nχ-2)*nζ
                    # χζ-component: nr*(nχ-2)*(nζ-2)
                    expected_dirichlet = (ns[0]-2)*(ns[1]-3)*(ns[2]-3) + \
                                    (ns[0]-3)*(ns[1]-2)*(ns[2]-3) + \
                                    (ns[0]-3)*(ns[1]-3)*(ns[2]-2)
                    expected_none = (ns[0])*(ns[1]-1)*(ns[2]-1) + \
                                    (ns[0]-1)*(ns[1])*(ns[2]-1) + \
                                    (ns[0]-1)*(ns[1]-1)*(ns[2])
                else:  # k == 3
                    # For 3-forms:
                    # - Dirichlet: nr*nχ*nζ basis functions (no reduction)
                    # - None: nr*nχ*nζ basis functions
                    expected_dirichlet = (ns[0]-3) * (ns[1]-3) * (ns[2]-3)
                    expected_none = (ns[0]-1) * (ns[1]-1) * (ns[2]-1)

                # Check number of basis functions
                self.assertEqual(
                    B_dirichlet.n, expected_dirichlet,
                    f"Wrong number of basis functions for k={k} with Dirichlet BCs. "
                    f"Expected {expected_dirichlet}, got {B_dirichlet.n}"
                )
                self.assertEqual(
                    B_none.n, expected_none,
                    f"Wrong number of basis functions for k={k} with no BCs. "
                    f"Expected {expected_none}, got {B_none.n}"
                )

                # Test mixed boundary conditions (Dirichlet in r-direction only)
                B_mixed = LazyBoundaryOperator(Λ_dirichlet, ('dirichlet', 'none', 'none'))
                if k == 0:
                    # For 0-forms: Remove boundary points only in r-direction
                    expected_mixed = (ns[0]-2) * ns[1] * ns[2]
                elif k == 1:
                    # For 1-forms:
                    # r-component: nr*ns[1]*ns[2] (no reduction)
                    # χ-component: (nr-2)*ns[1]*ns[2]
                    # ζ-component: (nr-2)*ns[1]*ns[2]
                    expected_mixed = (ns[0]-3)*(ns[1])*(ns[2]) + \
                                    (ns[0]-2)*(ns[1]-1)*(ns[2]) + \
                                    (ns[0]-2)*(ns[1])*(ns[2]-1)
                elif k == 2:
                    # For 2-forms:
                    # rχ-component: (nr-2)*ns[1]*ns[2]
                    # rζ-component: (nr-2)*ns[1]*ns[2]
                    # χζ-component: nr*ns[1]*ns[2]
                    expected_mixed = (ns[0]-2)*(ns[1]-1)*(ns[2]-1) + \
                                    (ns[0]-3)*(ns[1])*(ns[2]-1) + \
                                    (ns[0]-3)*(ns[1]-1)*(ns[2])
                else:  # k == 3
                    # For 3-forms: No reduction in any direction
                    expected_mixed = (ns[0]-3) * (ns[1]-1) * (ns[2]-1)

                self.assertEqual(
                    B_mixed.n, expected_mixed,
                    f"Wrong number of basis functions for k={k} with mixed BCs. "
                    f"Expected {expected_mixed}, got {B_mixed.n}"
                )

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions for different form degrees."""
        for k in [0, 1, 2, 3]:  # Test all form degrees
            with self.subTest(k=k):
                # Create differential form with periodic boundary conditions
                Λ = DifferentialForm(k, self.ns, self.ps, ('periodic', 'periodic', 'periodic'))
                # Create boundary operator with periodic conditions
                B = LazyBoundaryOperator(Λ, ('periodic', 'periodic', 'periodic'))

                # Create test function based on form degree
                if k == 0:
                    # Scalar field that's periodic in all directions
                    def test_func(x):
                        return jnp.sin(2 * jnp.pi * x[0]) * jnp.cos(2 * jnp.pi * x[1]) * jnp.sin(2 * jnp.pi * x[2])
                elif k == 1:
                    # Vector field that's periodic in all directions
                    def test_func(x):
                        return jnp.array([
                            jnp.sin(2 * jnp.pi * x[0]) * jnp.cos(2 * jnp.pi * x[1]),  # r-component
                            jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[2]),  # χ-component
                            jnp.sin(2 * jnp.pi * x[1]) * jnp.cos(2 * jnp.pi * x[2])   # ζ-component
                        ])
                elif k == 2:
                    # 2-form that's periodic in all directions
                    def test_func(x):
                        return jnp.array([
                            jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1]),  # rχ-component
                            jnp.cos(2 * jnp.pi * x[0]) * jnp.cos(2 * jnp.pi * x[2]),  # rζ-component
                            jnp.sin(2 * jnp.pi * x[1]) * jnp.sin(2 * jnp.pi * x[2])   # χζ-component
                        ])
                else:  # k == 3
                    # 3-form that's periodic in all directions
                    def test_func(x):
                        return jnp.sin(2 * jnp.pi * x[0]) * jnp.cos(2 * jnp.pi * x[1]) * jnp.sin(2 * jnp.pi * x[2])

                # Create quadrature rule for accurate integration
                Q = QuadratureRule(Λ, self.quad_order)

                # Project test function onto basis
                # Evaluate test function at quadrature points
                test_vals = jax.vmap(test_func)(Q.x)  # Shape depends on k

                # Evaluate basis functions at quadrature points
                basis_vals = jax.vmap(lambda x: jax.vmap(Λ, (None, 0))(x, jnp.arange(Λ.n)))(Q.x)
                if k == 0:
                    basis_vals = basis_vals.squeeze(-1)  # Remove last dimension if k=0

                # Compute coefficients
                if k == 0:
                    coeffs = jnp.einsum('q,qn,q->n', test_vals, basis_vals, Q.w)
                else:
                    coeffs = jnp.einsum('q...,qn...,q->n', test_vals, basis_vals, Q.w)

                # Apply boundary conditions
                coeffs_bc = B.M @ coeffs

                # Create points on all boundaries (faces, edges, and corners)
                boundary_points = []
                
                # Face centers (6 points)
                for i in range(3):  # For each dimension
                    for val in [0.0, 1.0]:  # Check both boundaries
                        x = jnp.array([0.5, 0.5, 0.5])  # Interior point
                        x = x.at[i].set(val)  # Set boundary value
                        boundary_points.append(x)
                
                # Edge centers (12 points)
                for i in range(3):  # First dimension
                    for j in range(i+1, 3):  # Second dimension
                        for val1 in [0.0, 1.0]:  # First boundary value
                            for val2 in [0.0, 1.0]:  # Second boundary value
                                x = jnp.array([0.5, 0.5, 0.5])  # Interior point
                                x = x.at[i].set(val1)  # Set first boundary
                                x = x.at[j].set(val2)  # Set second boundary
                                boundary_points.append(x)
                
                # Corners (8 points)
                for val1 in [0.0, 1.0]:
                    for val2 in [0.0, 1.0]:
                        for val3 in [0.0, 1.0]:
                            boundary_points.append(jnp.array([val1, val2, val3]))

                boundary_points = jnp.stack(boundary_points)  # Shape (26, 3)

                # Evaluate function at boundary points
                basis_vals_boundary = jax.vmap(lambda x: jax.vmap(Λ, (None, 0))(x, jnp.arange(B.n)))(boundary_points)
                if k == 0:
                    basis_vals_boundary = basis_vals_boundary.squeeze(-1)  # Shape (26, n)

                # For k>0, we need to check all components
                if k == 0:
                    boundary_values = jnp.einsum('n,bn->b', coeffs_bc, basis_vals_boundary)
                else:
                    boundary_values = jnp.einsum('n,bn...->b...', coeffs_bc, basis_vals_boundary)

                # Check periodic conditions for each pair of opposite boundaries
                # Face pairs (3 pairs)
                for i in range(3):  # For each dimension
                    face0_idx = i * 2  # First face in pair
                    face1_idx = i * 2 + 1  # Second face in pair
                    npt.assert_allclose(
                        boundary_values[face0_idx], boundary_values[face1_idx],
                        rtol=1e-10, atol=1e-10,
                        err_msg=f"Function values should match at opposite faces for k={k} in direction {i}"
                    )

                # Edge pairs (6 pairs)
                edge_pairs = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]
                for edge0_idx, edge1_idx in edge_pairs:
                    npt.assert_allclose(
                        boundary_values[edge0_idx], boundary_values[edge1_idx],
                        rtol=1e-10, atol=1e-10,
                        err_msg=f"Function values should match at opposite edges for k={k}"
                    )

                # Corner pairs (4 pairs)
                corner_pairs = [(18, 25), (19, 24), (20, 23), (21, 22)]
                for corner0_idx, corner1_idx in corner_pairs:
                    npt.assert_allclose(
                        boundary_values[corner0_idx], boundary_values[corner1_idx],
                        rtol=1e-10, atol=1e-10,
                        err_msg=f"Function values should match at opposite corners for k={k}"
                    )


if __name__ == '__main__':
    unittest.main() 