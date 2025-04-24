"""Object-Oriented Splines and Differential Forms Demo

This script demonstrates the use of spline bases and differential forms for numerical computations
in a mapped geometry. It includes:

1. Spline basis construction and visualization
2. Differential form operations (grad, curl, div)
3. Mapping between reference and physical domains
4. Error analysis of various numerical approximations
5. Visualization of fields in the mapped domain

The script uses periodic and clamped boundary conditions in different directions
and demonstrates both scalar and vector field operations.

Key components:
- SplineBasis: For function approximation
- DifferentialForm: For handling differential geometry operations
- LazyExtractionOperator: For efficient operator computation
- Projector: For L2 projection of functions onto spline spaces

Author: [Your Name]
Date: [Current Date]
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Callable, Any

from mrx.SplineBases import SplineBasis, DerivativeSpline
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix
from mrx.Utils import div, curl, grad

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)

# Create output directory if it doesn't exist
output_dir = Path("scripts_output")
output_dir.mkdir(parents=True, exist_ok=True)

# Define mesh parameters
ns = (8, 8, 1)  # Number of elements in each direction (r, χ, z)
ps = (3, 3, 0)  # Polynomial degrees
types = ('clamped', 'periodic', 'constant')  # Boundary condition types

# Define knot vectors for spline bases
_T = jnp.array([0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
# Add periodic extensions for the χ direction
T = jnp.concatenate([
    _T[-(ps[1]+1):-1] - 1,  # Left periodic extension
    _T,                      # Main knot vector
    _T[1:(ps[1]+1)] + 1     # Right periodic extension
])

# Define radial knot vector with clamped ends
Tr = jnp.array([0, 0.3, 0.6, 0.8, 0.9, 1.0])
Tr = jnp.concatenate([
    jnp.zeros(ps[0]),    # Clamped at r=0
    Tr,                  # Interior knots
    jnp.ones(ps[0])     # Clamped at r=1
])

# Combine knot vectors for all directions
Ts = (Tr, T, None)  # None for z direction (constant)

# Create spline basis and its derivative
s = SplineBasis(ns[1], ps[1], 'periodic', T)
d = DerivativeSpline(s)
x = jnp.linspace(0, 1, 1000)  # Points for visualization

# Plot basis functions
plt.figure(figsize=(10, 5))
plt.plot(x, jax.vmap(lambda x: s(jnp.array(x), 0))(x))
plt.title('Single Basis Function')
plt.xlabel('x')
plt.ylabel('Value')
plt.savefig(output_dir / 'single_basis.png')

plt.figure(figsize=(10, 5))
plt.plot(x, jax.vmap(lambda x: d(jnp.array(x), 0))(x))
plt.title('Derivative of Basis Function')
plt.xlabel('x')
plt.ylabel('Value')
plt.savefig(output_dir / 'basis_derivative.png')

# Plot all basis functions and their derivatives
plt.figure(figsize=(12, 6))
for i in range(ns[1]):
    plt.plot(x, jax.vmap(lambda x: s(jnp.array(x), i))(x))
plt.title('All Basis Functions')
plt.xlabel('x')
plt.ylabel('Value')
plt.savefig(output_dir / 'all_basis_functions.png')

plt.figure(figsize=(12, 6))
for i in range(ns[1]):
    plt.plot(x, jax.vmap(lambda x: d(jnp.array(x), i))(x))
plt.title('All Basis Function Derivatives')
plt.xlabel('x')
plt.ylabel('Value')
plt.savefig(output_dir / 'all_basis_derivatives.png')

# Create differential forms of different degrees
Λ0 = DifferentialForm(0, ns, ps, types, Ts)  # 0-forms (scalar fields)
Λ1 = DifferentialForm(1, ns, ps, types, Ts)  # 1-forms (vector fields)
Λ2 = DifferentialForm(2, ns, ps, types, Ts)  # 2-forms (vector fields)
Λ3 = DifferentialForm(3, ns, ps, types, Ts)  # 3-forms (scalar fields)
Q = QuadratureRule(Λ0, 10)  # Quadrature rule for integration

# Define mapping parameters
a = 1    # Deformation amplitude
R0 = 3.0  # Major radius
Y0 = 0.0  # Vertical shift


def θ(x):
    """Compute the poloidal angle in the mapped geometry.

    Args:
        x: Array of coordinates (r, χ, z)
    Returns:
        The poloidal angle θ
    """
    r, χ, z = x
    return 2 * jnp.atan(jnp.sqrt((1 + a*r/R0)/(1 - a*r/R0)) * jnp.tan(jnp.pi * χ))


def _R(r, χ):
    """Compute the major radius in the mapped geometry.

    Args:
        r: Radial coordinate
        χ: Poloidal coordinate
    Returns:
        The major radius R
    """
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))


def _Y(r, χ):
    """Compute the vertical coordinate in the mapped geometry.

    Args:
        r: Radial coordinate
        χ: Poloidal coordinate
    Returns:
        The vertical coordinate Y
    """
    return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))


def F(x):
    """Map from reference coordinates to physical space.

    Args:
        x: Array of reference coordinates (r, χ, z)
    Returns:
        Array of physical coordinates (R, Y, Z)
    """
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ) * jnp.cos(2 * jnp.pi * z),
                               _Y(r, χ),
                               _R(r, χ) * jnp.sin(2 * jnp.pi * z)]))

# def F(x):
#     r, χ, z = x
#     return jnp.ravel(jnp.array([
#         _R(r, χ) ,
#         _Y(r, χ),
#         jnp.ones(1) * z]))


ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0)
# %%
E0, E1, E2, E3 = [LazyExtractionOperator(Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
P0, P1, P2, P3 = [Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]

# E0 = LazyExtractionOperator(Λ0, ξ, True).M
# M0 = LazyMassMatrix(Λ0, Q, F, E0).M
# P0 = Projector(Λ0, Q, F, E0)
# %%


def l2_product(f: Callable[[jnp.ndarray], jnp.ndarray],
               g: Callable[[jnp.ndarray], jnp.ndarray],
               Q: Any) -> jnp.ndarray:
    """Compute the L2 inner product of two functions.

    Args:
        f: First function
        g: Second function
        Q: Quadrature rule
    Returns:
        The L2 inner product (f,g)
    """
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)


# %%
ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
_nx = 16
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.zeros(1)
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)


def compute_f(x):
    """Test function for scalar field.

    Args:
        x: Array of coordinates (r, χ, z)
    Returns:
        Scalar value representing a test function with radial and angular dependence.
        The function vanishes at r=0 and r=1 and has sinusoidal variation in χ.
    """
    r, χ, z = x
    return jnp.ones(1) * r**2 * jnp.sin(4 * jnp.pi * χ) * (1 - r)**2


# Initialize and visualize a discrete scalar field
f_hat = jnp.zeros(E0.shape[0]).at[35:43:2].set(1)  # Set specific coefficients to 1
f_h = DiscreteFunction(f_hat, Λ0, E0)  # Create discrete function
_z1 = jax.vmap(Pullback(f_h, F, 0))(_x).reshape(nx, nx, 1)  # Pull back to reference domain
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm)
plt.colorbar(label='Magnitude')
plt.title('Discrete Scalar Field')
plt.xlabel('R')
plt.ylabel('Y')
plt.savefig(output_dir / 'discrete_scalar.png')
plt.close()

# Project continuous function onto discrete space and compute error
f_hat = jnp.linalg.solve(M0, P0(compute_f))  # L2 projection
f_h = DiscreteFunction(f_hat, Λ0, E0)


def compute_f_error(x):
    """Compute pointwise error between exact and discrete scalar fields.

    Args:
        x: Coordinate point
    Returns:
        Difference between exact and discrete functions at x
    """
    return compute_f(x) - f_h(x)


# Compute relative L2 error for scalar field
scalar_error = (l2_product(compute_f_error, compute_f_error, Q) /
                l2_product(compute_f, compute_f, Q))**0.5

# Compute gradient field and its discrete approximation
A = grad(compute_f)  # Exact gradient
F_A = Pullback(A, F, 1)  # Pulled back gradient
A_hat = jnp.linalg.solve(M1, P1(A))  # L2 projection of gradient
A_h = DiscreteFunction(A_hat, Λ1, E1)  # Discrete gradient field


def compute_A_error(x):
    """Compute pointwise error between exact and discrete gradient fields.

    Args:
        x: Coordinate point
    Returns:
        Difference between exact and discrete gradient fields at x
    """
    return A(x) - A_h(x)


# Compute relative L2 error for gradient field
gradient_error = (l2_product(compute_A_error, compute_A_error, Q) /
                  l2_product(A, A, Q))**0.5

# Compute discrete gradient and compare with exact gradient
grad_fh = jax.grad(lambda x: (f_h)(x).sum())  # Gradient of discrete function
grad_f_hat = jnp.linalg.solve(M1, P1(grad_fh))  # Project discrete gradient
gradf_h = DiscreteFunction(grad_f_hat, Λ1, E1)  # Create discrete gradient field


def compute_error_1(x: jnp.ndarray) -> jnp.ndarray:
    """Compute error between discrete gradient and gradient of discrete function.

    Args:
        x: Coordinate point
    Returns:
        Difference between discrete gradient approaches at x
    """
    return grad_fh(x) - gradf_h(x)


# Compute relative L2 error between gradient approaches
gradient_consistency_error = (l2_product(compute_error_1, compute_error_1, Q) /
                              l2_product(grad_fh, grad_fh, Q))**0.5

# Initialize and visualize a 2-form field
B_hat = jnp.ones(E2.shape[0])  # Uniform coefficients
B_h = DiscreteFunction(B_hat, Λ2, E2)  # Create discrete 2-form
F_B_h = Pullback(B_h, F, 2)  # Pull back to reference domain

# Visualize 2-form field
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar(label='Magnitude')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
plt.title('2-Form Field')
plt.xlabel('R')
plt.ylabel('Y')
plt.savefig(output_dir / 'two_form_field.png')
plt.close()


def B(x):
    """Create a test 2-form by permuting components of gradient field.

    Args:
        x: Coordinate point
    Returns:
        Permuted vector field at x
    """
    v = A(x)
    return jnp.array([v[1], v[0], v[2]])


# Project and discretize the 2-form
B_hat = jnp.linalg.solve(M2, P2(B))  # L2 projection
B_h = DiscreteFunction(B_hat, Λ2, E2)  # Create discrete 2-form


def err(x):
    """Compute pointwise error in 2-form approximation.

    Args:
        x: Coordinate point
    Returns:
        Difference between exact and discrete 2-forms at x
    """
    return B(x) - B_h(x)


# Compute relative L2 error for 2-form
two_form_error = (l2_product(err, err, Q) / l2_product(B, B, Q))**0.5

# Pull back fields for visualization
F_B = Pullback(B, F, 2)  # Pull back exact 2-form
F_B_h = Pullback(B_h, F, 2)  # Pull back discrete 2-form

# Visualize exact vs discrete 2-form fields
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)

plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar(label='Discrete Field Magnitude')
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k', label='Exact Field')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
plt.title('Comparison of Exact and Discrete 2-Form Fields')
plt.xlabel('R')
plt.ylabel('Y')
plt.legend()
plt.savefig(output_dir / 'two_form_comparison.png')
plt.close()

# Compute curl of vector field
curl_Ah = curl(A_h)  # Curl of discrete vector field
curl_A_hat = jnp.linalg.solve(M2, P2(curl_Ah))  # Project curl
curlA_h = DiscreteFunction(curl_A_hat, Λ2, E2)  # Create discrete curl field


def compute_error_2(x: jnp.ndarray) -> jnp.ndarray:
    """Compute error between exact and discrete curl.

    Args:
        x: Coordinate point
    Returns:
        Difference between curl of exact and discrete vector fields at x
    """
    return curl(A)(x) - curl(A_h)(x)


# Compute relative L2 error for curl
curl_error = (l2_product(compute_error_2, compute_error_2, Q) /
              l2_product(curl(A), curl(A), Q))**0.5

# Define and project a 3-form
g = div(B)  # Divergence of 2-form field
g_hat = jnp.linalg.solve(M3, P3(g))  # L2 projection
g_h = DiscreteFunction(g_hat, Λ3, E3)  # Create discrete 3-form


def compute_error_4(x: jnp.ndarray) -> jnp.ndarray:
    """Compute error in 3-form approximation.

    Args:
        x: Coordinate point
    Returns:
        Difference between exact and discrete 3-forms at x
    """
    return g(x) - g_h(x)


# Compute relative L2 error for 3-form
three_form_error = (l2_product(compute_error_4, compute_error_4, Q) /
                    l2_product(g, g, Q))**0.5

# Pull back 3-forms for visualization
F_g = Pullback(g, F, 3)
F_g_h = Pullback(g_h, F, 3)

# Visualize 3-form fields
_z1 = jax.vmap(F_g_h)(_x).reshape(nx, nx, 1)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_g)(_x).reshape(nx, nx, 1)
_z2_norm = jnp.linalg.norm(_z2, axis=2)

plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar(label='Discrete Field Magnitude')
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k', label='Exact Field')
plt.title('Comparison of Exact and Discrete 3-Form Fields')
plt.xlabel('R')
plt.ylabel('Y')
plt.legend()
plt.savefig(output_dir / 'three_form_comparison.png')
plt.close()

# Compute divergence of vector field
div_Bh = div(B_h)  # Divergence of discrete 2-form
div_B_hat = jnp.linalg.solve(M3, P3(div_Bh))  # Project divergence
divB_h = DiscreteFunction(div_B_hat, Λ3, E3)  # Create discrete divergence field


def compute_error_3(x: jnp.ndarray) -> jnp.ndarray:
    """Compute error between exact and discrete divergence.

    Args:
        x: Coordinate point
    Returns:
        Difference between divergence of exact and discrete vector fields at x
    """
    return div(A)(x) - div(A_h)(x)


# Compute relative L2 error for divergence
div_error = (l2_product(compute_error_3, compute_error_3, Q) /
             l2_product(div(A), div(A), Q))**0.5

# Print error summary
print("\nNumerical Errors:")
print("-----------------")
print(f"Scalar field L2 error: {scalar_error:.2e}")
print(f"Gradient field L2 error: {gradient_error:.2e}")
print(f"Gradient consistency error: {gradient_consistency_error:.2e}")
print(f"2-form field L2 error: {two_form_error:.2e}")
print(f"Curl L2 error: {curl_error:.2e}")
print(f"3-form field L2 error: {three_form_error:.2e}")
print(f"Divergence L2 error: {div_error:.2e}")

# %%
plt.scatter(R_hat, Y_hat, s=5)
plt.scatter([τ + R0, R0 - τ/2, R0 - τ/2], [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2], s=10, c='k')
plt.plot([τ + R0, R0 - τ/2, R0 - τ/2, τ + R0],
         [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2, 0],
         'k:')
# %%

# %%


@jax.jit
def f():
    """Test function to verify index mapping consistency."""
    return [jnp.all(jax.vmap(lambda i: jax.jit(Λ._ravel_index)(*jax.jit(Λ._unravel_index)(i)))(jnp.arange(Λ.n)) == jnp.arange(Λ.n)) for Λ in [Λ0, Λ1, Λ2, Λ3]]


f()
# %%


@jax.jit
def test():
    """Test function for basic differential form operations."""
    ns = (4, 8, 1)
    ps = (2, 3, 0)
    types = ('clamped', 'periodic', 'constant')
    Λ0 = DifferentialForm(0, ns, ps, types)
    return Λ0[0](jnp.array([0.5, 0.5, 0.5]))


test()


def test_function(x: jnp.ndarray) -> jnp.ndarray:
    """Test function for numerical integration.

    Args:
        x: Array of coordinates
    Returns:
        Function value at x
    """
    return jnp.ones(1) * jnp.sqrt(2 * x[0]**3 * jnp.sin(jnp.pi * x[1]) * jnp.pi)


print(jnp.einsum("ij,ij,i->", jax.vmap(test_function)(Q.x), jax.vmap(test_function)(Q.x), Q.w))

# %%


@jax.jit
def get_err():
    """Compute L2 error for a test case.

    Returns:
        L2 error between exact and approximated function
    """
    ns = (4, 8, 1)
    ps = (2, 3, 0)
    types = ('clamped', 'periodic', 'constant')

    Λ0 = DifferentialForm(0, ns, ps, types)
    Q = QuadratureRule(Λ0, 6)

    def f(x):
        return jnp.ones(1) * jnp.sqrt(2 * x[0]**3 * jnp.sin(jnp.pi * x[1]) * jnp.pi)

    P0 = Projector(Λ0, Q, F)
    M0 = LazyMassMatrix(Λ0, Q, F).M

    f_hat = jnp.linalg.solve(M0, P0(f))
    f_h = DiscreteFunction(f_hat, Λ0)
    def err(x): return f(x) - f_h(x)
    return l2_product(err, err, Q)


# %%
start = time.time()
get_err()
print(time.time() - start)
start = time.time()
get_err()
print(time.time() - start)
for _ in range(100):
    get_err()
print((time.time() - start)/100)

# Save final visualization
plt.figure(figsize=(10, 8))
plt.scatter(R_hat, Y_hat, s=5, label='Grid Points')
plt.scatter([τ + R0, R0 - τ/2, R0 - τ/2],
            [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2],
            s=10, c='k', label='Reference Points')
plt.plot([τ + R0, R0 - τ/2, R0 - τ/2, τ + R0],
         [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2, 0],
         'k:', label='Reference Triangle')
plt.title('Domain Mapping Visualization')
plt.xlabel('R')
plt.ylabel('Y')
plt.legend()
plt.savefig(output_dir / 'domain_mapping.png')

# Show all plots at the end
plt.show()

print("\nScript completed successfully!")
print(f"Output files saved to: {output_dir}")
