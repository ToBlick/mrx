"""
Beltrami Field Analysis with Homogeneous Dirichlet Boundary Conditions

This script analyzes Beltrami fields in a 3D domain with homogeneous Dirichlet boundary conditions.
It computes the magnetic helicity and performs error analysis for different modes (m,n).

The script includes:
1. Definition of Beltrami field components
2. Computation of magnetic helicity
3. Error analysis for different modes
4. Visualization of results
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule

# Create output directory if it doesn't exist
os.makedirs('script_outputs', exist_ok=True)


def mu(m: int, n: int) -> jnp.ndarray:
    """
    Compute the eigenvalue for the Beltrami field.

    Args:
        m: First mode number
        n: Second mode number

    Returns:
        jnp.ndarray: The eigenvalue mu(m,n)
    """
    return jnp.pi * jnp.sqrt(m**2 + n**2)


def u(A_0: float, x: jnp.ndarray, m: int, n: int) -> jnp.ndarray:
    """
    Compute the Beltrami field components.

    Args:
        A_0: Amplitude factor
        x: Position vector (x1, x2, x3)
        m: First mode number
        n: Second mode number

    Returns:
        jnp.ndarray: Vector field components [u1, u2, u3]
    """
    x_1, x_2, x_3 = x
    return jnp.array([
        ((A_0 * n) / (jnp.sqrt(m**2 + n**2))) * jnp.sin(jnp.pi * m * x_1) * jnp.cos(jnp.pi * n * x_2),
        ((A_0 * m * -1) / (jnp.sqrt(m**2 + n**2))) * jnp.cos(jnp.pi * m * x_1) * jnp.sin(jnp.pi * n * x_2),
        jnp.sin(jnp.pi * m * x_1) * jnp.sin(jnp.pi * n * x_2)
    ])


def eta(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the weight function for the domain.

    Args:
        x: Position vector (x1, x2, x3)

    Returns:
        jnp.ndarray: Weight value at position x
    """
    x_1, x_2, x_3 = x
    return ((x_1**2) * ((1-x_1))**2) * ((x_2**2) * ((1-x_2))**2) * ((x_3**2) * ((1-x_3))**2)


def integrand(m: int, n: int, x: jnp.ndarray, A_0: float) -> jnp.ndarray:
    """
    Compute the integrand for magnetic helicity calculation.

    Args:
        m: First mode number
        n: Second mode number
        x: Position vector
        A_0: Amplitude factor

    Returns:
        jnp.ndarray: Value of the integrand
    """
    field = u(A_0, x, m, n)
    # The magnetic helicity is the integral of A·B where B = curl A
    # For Beltrami fields, B = mu × A
    mu_val = mu(m, n)
    return eta(x) * jnp.dot(field, field) * mu_val


def compute_helicity(m: int, n: int, A_0: float, Q: QuadratureRule) -> jnp.ndarray:
    """
    Compute the magnetic helicity for given modes.

    Args:
        m: First mode number
        n: Second mode number
        A_0: Amplitude factor
        Q: Quadrature rule for integration

    Returns:
        jnp.ndarray: Magnetic helicity value
    """
    # Compute integrand at quadrature points
    integrand_values = jnp.array([integrand(m, n, x, A_0) for x in Q.x])

    # Compute integral using quadrature weights
    integral = jnp.sum(integrand_values * Q.w)

    # Multiply by eigenvalue
    return integral * mu(m, n)


def plot_field_components(m: int, n: int, A_0: float, nx: int = 100) -> None:
    """
    Plot the Beltrami field components and save to file.

    Args:
        m: First mode number
        n: Second mode number
        A_0: Amplitude factor
        nx: Number of points in each direction
    """
    x: jnp.ndarray = jnp.linspace(0, 1, nx)
    y: jnp.ndarray = jnp.linspace(0, 1, nx)
    X, Y = jnp.meshgrid(x, y)
    z: float = 0.5  # Fixed z value for 2D slice

    # Initialize arrays
    U: jnp.ndarray = jnp.zeros((nx, nx))
    V: jnp.ndarray = jnp.zeros((nx, nx))
    W: jnp.ndarray = jnp.zeros((nx, nx))

    # Compute field components using JAX's immutable operations
    for i in range(nx):
        for j in range(nx):
            field = u(A_0, jnp.array([X[i, j], Y[i, j], z]), m, n)
            U = U.at[i, j].set(field[0])
            V = V.at[i, j].set(field[1])
            W = W.at[i, j].set(field[2])

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot u1 component
    im1 = ax1.contourf(X, Y, U)
    ax1.set_title('u1 Component')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)

    # Plot u2 component
    im2 = ax2.contourf(X, Y, V)
    ax2.set_title('u2 Component')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)

    # Plot u3 component
    im3 = ax3.contourf(X, Y, W)
    ax3.set_title('u3 Component')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3)

    plt.suptitle(f'Beltrami Field Components (m={m}, n={n})')
    plt.tight_layout()

    # Save figure
    save_path = os.path.join('script_outputs', f'beltrami_components_m{m}_n{n}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved field components plot to {save_path}")

    # Show figure
    plt.show()


def analyze_convergence(m_range: List[int], n_range: List[int], A_0: float, Q: QuadratureRule) -> None:
    """
    Analyze convergence of magnetic helicity for different modes and save plot to file.

    Args:
        m_range: List of m values to analyze
        n_range: List of n values to analyze
        A_0: Amplitude factor
        Q: Quadrature rule for integration
    """
    helicity_values: List[float] = []
    modes: List[str] = []

    for m in m_range:
        for n in n_range:
            H = compute_helicity(m, n, A_0, Q)
            helicity_values.append(float(H))  # Convert jnp.ndarray to float for plotting
            modes.append(f'({m},{n})')

    # Plot helicity values
    plt.figure(figsize=(10, 6))
    plt.bar(modes, helicity_values)
    plt.xlabel('Mode (m,n)')
    plt.ylabel('Magnetic Helicity')
    plt.title('Magnetic Helicity for Different Modes')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join('script_outputs', 'beltrami_helicity.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved helicity plot to {save_path}")

    # Show figure
    plt.show()


def main() -> None:
    """Main function to run the Beltrami field analysis."""
    # Set parameters
    n: int = 5
    p: int = 3
    ns: Tuple[int, int, int] = (n, n, n)
    ps: Tuple[int, int, int] = (p, p, p)
    types: Tuple[str, str, str] = ('clamped', 'clamped', 'constant')

    # Initialize differential form and quadrature rule
    Λ0 = DifferentialForm(0, ns, ps, types)
    Q = QuadratureRule(Λ0, 15)

    # Set amplitude
    A_0: float = 1.0

    # Analyze different modes
    m_range: List[int] = [1, 2, 3]
    n_range: List[int] = [1, 2, 3]

    # Plot field components for first mode
    plot_field_components(m_range[0], n_range[0], A_0)

    # Analyze convergence
    analyze_convergence(m_range, n_range, A_0, Q)

    # Print helicity values
    print("\nMagnetic Helicity Values:")
    for m in m_range:
        for n in n_range:
            H = compute_helicity(m, n, A_0, Q)
            print(f"Mode ({m},{n}): H = {float(H):.6f}")  # Convert jnp.ndarray to float for printing


if __name__ == "__main__":
    main()
