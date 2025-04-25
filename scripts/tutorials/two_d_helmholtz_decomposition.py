"""Two-dimensional Helmholtz decomposition using finite element methods.

This script demonstrates the Helmholtz decomposition of a vector field into its
irrotational (gradient) and solenoidal (curl) components using finite element
methods. The decomposition is performed using the Leray projector.

The script:
1. Sets up the problem with a test vector field
2. Performs the Helmholtz decomposition
3. Computes and plots the results
4. Analyzes the accuracy of the decomposition

Example usage:
    python two_d_helmholtz_decomposition.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import curl, div, grad, l2_product

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)

# Create output directory if it doesn't exist
output_dir = Path("script_outputs/helmholtz")
output_dir.mkdir(parents=True, exist_ok=True)


def setup_problem(n=8, p=3):
    """Set up the finite element problem for Helmholtz decomposition.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree of the basis functions

    Returns:
        tuple: (Λ0, Λ2, Λ3, Q, D, M2, K, P2) - The differential forms and operators
    """
    # Define the domain and basis functions
    ns = (n, n, 1)  # Number of elements in each direction
    ps = (p, p, 0)  # Polynomial degree in each direction
    types = ('clamped', 'clamped', 'constant')  # Boundary conditions

    # Create differential forms for 0-forms, 2-forms, and 3-forms
    Λ0 = DifferentialForm(0, ns, ps, types)
    Λ2 = DifferentialForm(2, ns, ps, types)
    Λ3 = DifferentialForm(3, ns, ps, types)

    # Set up quadrature and operators
    Q = QuadratureRule(Λ0, 4)
    D = LazyDerivativeMatrix(Λ2, Λ3, Q).M
    M2 = LazyMassMatrix(Λ2, Q).M
    K = D @ jnp.linalg.solve(M2, D.T)
    P2 = Projector(Λ2, Q)

    return Λ0, Λ2, Λ3, Q, D, M2, K, P2


def define_test_functions():
    """Define the test functions for the Helmholtz decomposition.

    Returns:
        tuple: (q, w, u) - The scalar potential, vector potential, and total field
    """
    def q(x):
        """Scalar potential function."""
        r, χ, z = x
        v = ((0.5 - r)**2 + (χ - 0.5)**2)**0.5
        return jnp.ones(1) * jnp.sin(jnp.pi * r) * jnp.sin(jnp.pi * χ) * jnp.exp(-v**2)

    def _w(x):
        """Vector potential function."""
        r, χ, z = x
        v = ((0.5 - r)**2 + (χ - 0.5)**2)**0.5
        return 10 * jnp.array([0, 0, 1]) * jnp.exp(-v**2)

    w = curl(_w)  # Solenoidal component of _w
    def u(x): return grad(q)(x) + w(x)  # Total field

    return q, w, u


def perform_decomposition(Λ2, M2, P2, D, K, u, grad_q, w):
    """Perform the Helmholtz decomposition using the Leray projector.

    Args:
        Λ2: The 2-form differential form
        M2: The mass matrix for 2-forms
        P2: The projection operator
        D: The derivative matrix
        K: The stiffness matrix
        u: The total field
        grad_q: The gradient component
        w: The curl component

    Returns:
        tuple: (u_hat, grad_q_hat, w_hat) - The decomposed components
    """
    # Project the total field
    u_hat = jnp.linalg.solve(M2, P2(u))

    # Compute the Leray projector
    𝚷_Leray = jnp.eye(Λ2.n) - jnp.linalg.solve(M2,
                                               D.T @ jnp.linalg.solve(K, D))

    # Decompose the field
    w_hat = 𝚷_Leray @ u_hat
    grad_q_hat = u_hat - w_hat

    return u_hat, grad_q_hat, w_hat


def compute_errors(u, grad_q, w, u_h, grad_q_h, w_h, Q, u_h_proj, grad_q_h_proj, w_h_proj):
    """Compute the L2 errors of the decomposition.

    Args:
        u, grad_q, w: The exact fields
        u_h, grad_q_h, w_h: The computed fields
        Q: The quadrature rule

    Returns:
        dict: The relative L2 errors for each component
    """
    def err(err, val):
        return (l2_product(err, err, Q) / l2_product(val, val, Q))**0.5

    def err_u(x): return u(x) - u_h(x)
    def err_u_proj(x): return u(x) - u_h_proj(x)
    def err_grad_q(x): return grad(q)(x) - grad_q_h(x)
    def err_grad_q_proj(x): return grad(q)(x) - grad_q_h_proj(x)
    def err_w(x): return w(x) - w_h(x)
    def err_w_proj(x): return w(x) - w_h_proj(x)

    return {
        'u': err(err_u, u),
        'grad_q': err(err_grad_q, grad_q),
        'w': err(err_w, w),
        'u (projection)': err(err_u_proj, u),
        'grad_q (projection)': err(err_grad_q_proj, grad_q),
        'w (projection)': err(err_w_proj, w),
        'div_w': l2_product(div(w_h), div(w_h), Q)**0.5,
        'curl_grad_q': l2_product(curl(grad_q_h), curl(grad_q_h), Q)**0.5
    }


def plot_field(x, y, field, title, filename):
    """Plot a vector field with streamlines and quivers.

    Args:
        x: The x-coordinates
        y: The y-coordinates
        field: The vector field to plot
        title: The plot title
        filename: The output filename
    """
    plt.figure(figsize=(8, 6))
    field_norm = jnp.linalg.norm(field, axis=2)
    plt.contourf(x, y, field_norm)
    plt.colorbar(label='Magnitude')
    plt.contour(x, y, field_norm, colors='k', alpha=0.5)
    plt.quiver(x[::4, ::4], y[::4, ::4],
               field[::4, ::4, 0], field[::4, ::4, 1],
               color='w', scale=20)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(output_dir / filename)


# Set up the problem
Λ0, Λ2, Λ3, Q, D, M2, K, P2 = setup_problem()
q, w, u = define_test_functions()

# Perform the decomposition
u_hat, grad_q_hat, w_hat = perform_decomposition(
    Λ2, M2, P2, D, K, u, grad(q), w)

# Create discrete functions
u_h = DiscreteFunction(u_hat, Λ2)
grad_q_h = DiscreteFunction(grad_q_hat, Λ2)
w_h = DiscreteFunction(w_hat, Λ2)
grad_q_hat_proj = jnp.linalg.solve(M2, P2(grad(q)))
w_hat_proj = jnp.linalg.solve(M2, P2(w))
w_h_proj = DiscreteFunction(w_hat_proj, Λ2)
grad_q_h_proj = DiscreteFunction(grad_q_hat_proj, Λ2)
u_h_proj = DiscreteFunction(u_hat, Λ2)

# Compute errors
errors = compute_errors(u, grad(q), w, u_h, grad_q_h,
                        w_h, Q, u_h_proj, grad_q_h_proj, w_h_proj)
print("\nRelative L2 errors:")
for component, error in errors.items():
    print(f"{component}: {error:.2e}")

# Set up plotting grid
nx = 64
x1 = jnp.linspace(1e-5, 1-1e-5, nx)
x2 = jnp.linspace(1e-5, 1-1e-5, nx)
x3 = jnp.ones(1)/2
X = jnp.array(jnp.meshgrid(x1, x2, x3))
X = X.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)

# Plot the fields


def F(x): return x  # Identity mapping


F_u = Pullback(u, F, 2)
F_u_h = Pullback(u_h, F, 2)
F_grad_q = Pullback(grad(q), F, 2)
F_grad_q_h = Pullback(grad_q_h, F, 2)
F_w = Pullback(w, F, 2)
F_w_h = Pullback(w_h, F, 2)

# Evaluate fields on grid
u_exact = jax.vmap(F_u)(X).reshape(nx, nx, 3)
u_approx = jax.vmap(F_u_h)(X).reshape(nx, nx, 3)
grad_q_exact = jax.vmap(F_grad_q)(X).reshape(nx, nx, 3)
grad_q_approx = jax.vmap(F_grad_q_h)(X).reshape(nx, nx, 3)
w_exact = jax.vmap(F_w)(X).reshape(nx, nx, 3)
w_approx = jax.vmap(F_w_h)(X).reshape(nx, nx, 3)
Y = jax.vmap(F)(X)
Y1 = Y[:, 0].reshape(nx, nx)
Y2 = Y[:, 1].reshape(nx, nx)

# Create plots
plot_field(Y1, Y2, u_exact, "Exact Total Field", "total_field_exact.png")
plot_field(Y1, Y2, u_approx, "Approximate Total Field",
           "total_field_approx.png")
plot_field(Y1, Y2, grad_q_exact,
           "Exact Gradient Component", "gradient_exact.png")
plot_field(Y1, Y2, grad_q_approx,
           "Approximate Gradient Component", "gradient_approx.png")
plot_field(Y1, Y2, w_exact, "Exact Curl Component", "curl_exact.png")
plot_field(Y1, Y2, w_approx, "Approximate Curl Component", "curl_approx.png")

# Plot singular values of Leray projector
𝚷_Leray = jnp.eye(Λ2.n) - jnp.linalg.solve(M2, D.T @ jnp.linalg.solve(K, D))
U, S, Vh = jnp.linalg.svd(𝚷_Leray)

plt.figure(figsize=(8, 6))
plt.plot(S / S[0])
plt.yscale('log')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.vlines(S.shape[0] - Λ3.n, ymax=2, ymin=1e-8, color='k', linestyle='--')
plt.title('Singular Values of Leray Projector')
plt.savefig(output_dir / "leray_singular_values.png")
plt.show()
