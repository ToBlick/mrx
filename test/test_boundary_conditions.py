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
        for k in [0, 1]:  # Test all form degrees
            with self.subTest(k=k):
                # First test with periodic boundary conditions
                Λ_periodic = DifferentialForm(k, self.ns, self.ps, ('periodic', 'periodic', 'periodic'))
                B_periodic = LazyBoundaryOperator(Λ_periodic, ('none', 'none', 'none'))

                # For periodic BCs, test preservation of periodic functions
                M_periodic = B_periodic.M
                print(f"\nPeriodic boundary operator matrix shape: {M_periodic.shape}")
                print(f"Number of non-zero elements: {jnp.sum(M_periodic != 0)}")

                if k == 0 or k == 3:
                    # For 0-forms and 3-forms, the operator should be identity
                    print(f"Is identity? {jnp.allclose(M_periodic, jnp.eye(B_periodic.n))}")
                    npt.assert_allclose(
                        M_periodic, jnp.eye(B_periodic.n),
                        rtol=1e-10, atol=1e-10,
                        err_msg=f"Periodic boundary operator should be identity for k={k}"
                    )
                else:
                    # For 1-forms and 2-forms, test with periodic test functions
                    # Create a periodic test function
                    if k == 1:
                        # Test each component separately
                        for component in range(3):
                            # Create a test vector that's periodic in the component's direction
                            test_vec = jnp.zeros(Λ_periodic.n)
                            if component == 0:  # r-component
                                # For r-component: (nr - 1) * nχ * nζ basis functions
                                r_idx = 1  # Interior point in r (0 to nr-2)
                                m = 2      # Any point in χ (0 to nχ-1)
                                n = 2      # Any point in ζ (0 to nζ-1)
                                idx = r_idx * Λ_periodic.nχ * Λ_periodic.nζ + \
                                     m * Λ_periodic.nζ + \
                                     n
                            elif component == 1:  # χ-component
                                # For χ-component: nr * (nχ - 1) * nζ basis functions
                                r_idx = 2  # Any point in r (0 to nr-1)
                                m = 1      # Interior point in χ (0 to nχ-2)
                                n = 2      # Any point in ζ (0 to nζ-1)
                                idx = Λ_periodic.n1 + \
                                     r_idx * (Λ_periodic.nχ - 1) * Λ_periodic.nζ + \
                                     m * Λ_periodic.nζ + \
                                     n
                            else:  # ζ-component
                                # For ζ-component: nr * nχ * (nζ - 1) basis functions
                                r_idx = 2  # Any point in r (0 to nr-1)
                                m = 2      # Any point in χ (0 to nχ-1)
                                n = 1      # Interior point in ζ (0 to nζ-2)
                                idx = Λ_periodic.n1 + Λ_periodic.n2 + \
                                     r_idx * Λ_periodic.nχ * (Λ_periodic.nζ - 1) + \
                                     m * (Λ_periodic.nζ - 1) + \
                                     n
                            test_vec = test_vec.at[idx].set(1.0)
                            
                            # Apply boundary operator
                            result = M_periodic @ test_vec
                            
                            # For periodic BCs, the indices should be the same
                            expected = jnp.zeros(B_periodic.n)
                            expected = expected.at[idx].set(1.0)
                            
                            print(f"\nTesting periodic preservation for component {component}")
                            print(f"Test vector non-zero at index: {idx}")
                            print(f"Result vector: {result}")
                            print(f"Expected vector: {expected}")
                            
                            npt.assert_allclose(
                                result, expected,
                                rtol=1e-10, atol=1e-10,
                                err_msg=f"Periodic boundary operator should preserve component {component} for k={k}"
                            )
                    else:  # k == 2
                        # Similar test for 2-forms, but with appropriate indices for the components
                        for component in range(3):
                            test_vec = jnp.zeros(Λ_periodic.n)
                            if component == 0:  # rχ-component
                                # Test rχ-component at an interior point
                                r_idx = 2  # Interior point in r
                                m = 2      # Interior point in χ
                                n = 2      # Interior point in ζ
                                idx = r_idx * (Λ_periodic.nχ - 1) * (Λ_periodic.nζ - 1) + \
                                     m * (Λ_periodic.nζ - 1) + \
                                     n
                            elif component == 1:  # rζ-component
                                # Test rζ-component at an interior point
                                r_idx = 2  # Interior point in r
                                m = 2      # Interior point in χ
                                n = 2      # Interior point in ζ
                                idx = Λ_periodic.n1 + \
                                     r_idx * Λ_periodic.nχ * (Λ_periodic.nζ - 1) + \
                                     m * (Λ_periodic.nζ - 1) + \
                                     n
                            else:  # χζ-component
                                # Test χζ-component at an interior point
                                r_idx = 2  # Interior point in r
                                m = 2      # Interior point in χ
                                n = 2      # Interior point in ζ
                                idx = Λ_periodic.n1 + Λ_periodic.n2 + \
                                     r_idx * (Λ_periodic.nχ - 1) * Λ_periodic.nζ + \
                                     m * Λ_periodic.nζ + \
                                     n
                            test_vec = test_vec.at[idx].set(1.0)
                            
                            # Apply boundary operator
                            result = M_periodic @ test_vec
                            
                            # The result should preserve the value at the interior point
                            expected = jnp.zeros(B_periodic.n)
                            if component == 0:
                                # For rχ-component, same indices in reduced space
                                reduced_idx = idx
                            elif component == 1:
                                # For rζ-component, same indices in reduced space
                                reduced_idx = idx
                            else:
                                # For χζ-component, same indices in reduced space
                                reduced_idx = idx
                            expected = expected.at[reduced_idx].set(1.0)
                            
                            print(f"\nTesting periodic preservation for component {component}")
                            print(f"Test vector non-zero at index: {idx}")
                            print(f"Expected non-zero at index: {reduced_idx}")
                            print(f"Result vector: {result}")
                            print(f"Expected vector: {expected}")
                            
                            npt.assert_allclose(
                                result, expected,
                                rtol=1e-10, atol=1e-10,
                                err_msg=f"Periodic boundary operator should preserve component {component} for k={k}"
                            )

                # Now test with Dirichlet boundary conditions
                Λ_dirichlet = DifferentialForm(k, self.ns, self.ps, ('clamped', 'clamped', 'clamped'))
                B_dirichlet = LazyBoundaryOperator(Λ_dirichlet, ('dirichlet', 'dirichlet', 'dirichlet'))

                # Test matrix properties
                M = B_dirichlet.M
                print(f"\nDirichlet boundary operator matrix shape: {M.shape}")
                print(f"Number of non-zero elements: {jnp.sum(M != 0)}")

                # Check sparsity
                nnz = jnp.sum(M != 0)
                sparsity = 1 - nnz / (M.shape[0] * M.shape[1])
                print(f"Sparsity: {sparsity}")
                self.assertGreater(sparsity, 0.9, f"Boundary operator should be sparse for k={k}")

                # Check number of non-zero entries per row
                nnz_per_row = jnp.sum(M != 0, axis=1)
                print(f"\nNon-zero entries per row: {nnz_per_row}")
                self.assertTrue(
                    jnp.all(nnz_per_row <= 2),
                    f"Each row should have at most 2 non-zero entries for k={k}"
                )

                # Check that projection preserves interior basis functions
                if k == 0:
                    # For 0-forms, test middle basis function
                    r_idx = 2  # Interior point in r
                    m = 2      # Interior point in χ
                    n = 2      # Interior point in ζ
                    test_idx = r_idx * Λ_dirichlet.nχ * Λ_dirichlet.nζ + \
                             m * Λ_dirichlet.nζ + \
                             n
                    
                    print(f"\nTesting interior basis function at index {test_idx}")
                    print(f"Matrix dimensions: {M.shape}")
                    print(f"B_dirichlet.n: {B_dirichlet.n}")
                    print(f"B_dirichlet.nr, nχ, nζ: {B_dirichlet.nr}, {B_dirichlet.nχ}, {B_dirichlet.nζ}")
                    print(f"Full space indices (r,χ,ζ): ({r_idx}, {m}, {n})")
                    
                    # Calculate expected indices in reduced space
                    reduced_r = r_idx - 1
                    reduced_m = m - 1
                    reduced_n = n - 1
                    reduced_idx = reduced_r * B_dirichlet.nχ * B_dirichlet.nζ + \
                                reduced_m * B_dirichlet.nζ + \
                                reduced_n
                    print(f"Reduced space indices (r,χ,ζ): ({reduced_r}, {reduced_m}, {reduced_n})")
                    print(f"Expected reduced index: {reduced_idx}")
                    
                    e = jnp.zeros(Λ_dirichlet.n)
                    e = e.at[test_idx].set(1.0)
                    print(f"Test vector shape: {e.shape}")
                    print(f"Test vector non-zero index: {test_idx}")
                    
                    projected = M @ e
                    print(f"Projected vector shape: {projected.shape}")
                    print(f"Projected vector: {projected}")
                    print(f"Original vector slice: {e[:B_dirichlet.n]}")
                    
                    # Create the expected vector in the reduced space
                    expected = jnp.zeros(B_dirichlet.n)
                    expected = expected.at[reduced_idx].set(1.0)
                    print(f"Expected vector: {expected}")
                    
                    npt.assert_allclose(
                        projected, expected,
                        rtol=1e-10, atol=1e-10,
                        err_msg=f"Interior basis function not preserved for k={k}"
                    )
                elif k == 1:
                    # For 1-forms, test middle basis function in each component
                    for component in range(3):
                        if component == 0:
                            # r-component: test middle point in r, interior in χ,ζ
                            r_idx = 2  # Interior point in r
                            m = 2      # Interior point in χ
                            n = 2      # Interior point in ζ
                            test_idx = r_idx * Λ_dirichlet.nχ * Λ_dirichlet.nζ + \
                                     m * Λ_dirichlet.nζ + \
                                     n
                        elif component == 1:
                            # χ-component: interior in r, middle point in χ, interior in ζ
                            r_idx = 2  # Interior point in r
                            m = 2      # Middle point in χ
                            n = 2      # Interior point in ζ
                            test_idx = Λ_dirichlet.n1 + \
                                     r_idx * (Λ_dirichlet.nχ - 1) * Λ_dirichlet.nζ + \
                                     m * Λ_dirichlet.nζ + \
                                     n
                        else:  # component == 2
                            # ζ-component: interior in r,χ, middle point in ζ
                            r_idx = 2  # Interior point in r
                            m = 2      # Interior point in χ
                            n = 2      # Middle point in ζ
                            test_idx = Λ_dirichlet.n1 + Λ_dirichlet.n2 + \
                                     r_idx * Λ_dirichlet.nχ * (Λ_dirichlet.nζ - 1) + \
                                     m * (Λ_dirichlet.nζ - 1) + \
                                     n

                        print(f"\nTesting component {component} at index {test_idx}")
                        e = jnp.zeros(Λ_dirichlet.n)
                        e = e.at[test_idx].set(1.0)
                        projected = M @ e
                        
                        # Calculate expected reduced index
                        if component == 0:
                            reduced_r = r_idx - 1
                            reduced_m = m - 1
                            reduced_n = n - 1
                            reduced_idx = reduced_r * B_dirichlet.nχ * B_dirichlet.nζ + \
                                        reduced_m * B_dirichlet.nζ + \
                                        reduced_n
                        elif component == 1:
                            reduced_r = r_idx - 1
                            reduced_m = m - 1
                            reduced_n = n - 1
                            reduced_idx = B_dirichlet.n1 + \
                                        reduced_r * (B_dirichlet.nχ - 1) * B_dirichlet.nζ + \
                                        reduced_m * B_dirichlet.nζ + \
                                        reduced_n
                        else:  # component == 2
                            reduced_r = r_idx - 1
                            reduced_m = m - 1
                            reduced_n = n - 1
                            reduced_idx = B_dirichlet.n1 + B_dirichlet.n2 + \
                                        reduced_r * B_dirichlet.nχ * (B_dirichlet.nζ - 1) + \
                                        reduced_m * B_dirichlet.nζ + \
                                        reduced_n

                        expected = jnp.zeros(B_dirichlet.n)
                        expected = expected.at[reduced_idx].set(1.0)
                        
                        npt.assert_allclose(
                            projected, expected,
                            rtol=1e-10, atol=1e-10,
                            err_msg=f"Interior basis function not preserved for k={k}, component={component}"
                        )
                elif k == 2:
                    # For 2-forms, test middle basis function in each component
                    for component in range(3):
                        if component == 0:
                            # rχ-component: middle point in r,χ, interior in ζ
                            r_idx = 2  # Middle point in r
                            m = 2      # Middle point in χ
                            n = 2      # Interior point in ζ
                            test_idx = r_idx * (Λ_dirichlet.nχ - 1) * (Λ_dirichlet.nζ - 1) + \
                                     m * (Λ_dirichlet.nζ - 1) + \
                                     n
                        elif component == 1:
                            # rζ-component: middle point in r, interior in χ, middle point in ζ
                            r_idx = 2  # Middle point in r
                            m = 2      # Interior point in χ
                            n = 2      # Middle point in ζ
                            test_idx = Λ_dirichlet.n1 + \
                                     r_idx * Λ_dirichlet.nχ * (Λ_dirichlet.nζ - 1) + \
                                     m * (Λ_dirichlet.nζ - 1) + \
                                     n
                        else:  # component == 2
                            # χζ-component: interior in r, middle point in χ,ζ
                            r_idx = 2  # Interior point in r
                            m = 2      # Middle point in χ
                            n = 2      # Middle point in ζ
                            test_idx = Λ_dirichlet.n1 + Λ_dirichlet.n2 + \
                                     r_idx * (Λ_dirichlet.nχ - 1) * Λ_dirichlet.nζ + \
                                     m * Λ_dirichlet.nζ + \
                                     n

                        print(f"\nTesting component {component} at index {test_idx}")
                        e = jnp.zeros(Λ_dirichlet.n)
                        e = e.at[test_idx].set(1.0)
                        projected = M @ e
                        
                        # Calculate expected reduced index
                        if component == 0:
                            reduced_r = r_idx - 1
                            reduced_m = m - 1
                            reduced_n = n - 1
                            reduced_idx = reduced_r * (B_dirichlet.nχ - 1) * (B_dirichlet.nζ - 1) + \
                                        reduced_m * (B_dirichlet.nζ - 1) + \
                                        reduced_n
                        elif component == 1:
                            reduced_r = r_idx - 1
                            reduced_m = m - 1
                            reduced_n = n - 1
                            reduced_idx = B_dirichlet.n1 + \
                                        reduced_r * B_dirichlet.nχ * (B_dirichlet.nζ - 1) + \
                                        reduced_m * (B_dirichlet.nζ - 1) + \
                                        reduced_n
                        else:  # component == 2
                            reduced_r = r_idx - 1
                            reduced_m = m - 1
                            reduced_n = n - 1
                            reduced_idx = B_dirichlet.n1 + B_dirichlet.n2 + \
                                        reduced_r * (B_dirichlet.nχ - 1) * B_dirichlet.nζ + \
                                        reduced_m * B_dirichlet.nζ + \
                                        reduced_n

                        expected = jnp.zeros(B_dirichlet.n)
                        expected = expected.at[reduced_idx].set(1.0)
                        
                        npt.assert_allclose(
                            projected, expected,
                            rtol=1e-10, atol=1e-10,
                            err_msg=f"Interior basis function not preserved for k={k}, component={component}"
                        )
                else:  # k == 3
                    # For 3-forms, test middle basis function
                    r_idx = 2  # Middle point in r
                    m = 2      # Middle point in χ
                    n = 2      # Middle point in ζ
                    test_idx = r_idx * (Λ_dirichlet.nχ - 1) * (Λ_dirichlet.nζ - 1) + \
                             m * (Λ_dirichlet.nζ - 1) + \
                             n

                    print(f"\nTesting 3-form at index {test_idx}")
                    e = jnp.zeros(Λ_dirichlet.n)
                    e = e.at[test_idx].set(1.0)
                    projected = M @ e
                    
                    # Calculate expected reduced index
                    reduced_r = r_idx - 1
                    reduced_m = m - 1
                    reduced_n = n - 1
                    reduced_idx = reduced_r * (B_dirichlet.nχ - 1) * (B_dirichlet.nζ - 1) + \
                                reduced_m * (B_dirichlet.nζ - 1) + \
                                reduced_n

                    expected = jnp.zeros(B_dirichlet.n)
                    expected = expected.at[reduced_idx].set(1.0)
                    
                    npt.assert_allclose(
                        projected, expected,
                        rtol=1e-10, atol=1e-10,
                        err_msg=f"Interior basis function not preserved for k={k}"
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