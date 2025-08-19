# %%
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pushforward
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
import matplotlib.gridspec as gridspec

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def generalized_eigh(A, B):
    """Solve the generalized eigenvalue problem A*v = lambda*B*v."""
    # Add a small identity matrix for numerical stability during Cholesky decomposition
    L = jnp.linalg.cholesky(B + jnp.eye(B.shape[0]) * 1e-12)
    L_inv = jnp.linalg.inv(L)
    # Transform to a standard eigenvalue problem: C*v' = lambda*v'
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    # Transform eigenvectors back to the original basis
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original

# %%
@partial(jax.jit, static_argnames=["n", "p", "n_map"])
def get_evs(a_hat, n, p, n_map):
    """
    Computes the eigenvalues and eigenvectors for a drum shape defined by a_hat.
    
    Args:
        a_hat: Discrete representation of the radius function r(χ).
        n: Number of elements.
        p: Polynomial degree.
        
    Returns:
        A tuple (eigenvalues, eigenvectors).
    """
    # Define the mapping from the parameter χ to the radius function
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p, 1, 1),
                            ('periodic', 'constant', 'constant'))
    _a_h = DiscreteFunction(a_hat, Λmap)

    def a_h(x):
        _x = jnp.array([x, 0, 0])
        return _a_h(_x)

    # Define the polar mapping from logical (r, χ) to physical (X, Y) coordinates
    def _R(r, χ):
        return a_h(χ) * r * jnp.cos(2 * jnp.pi * χ) * jnp.ones(1)

    def _Y(r, χ):
        return a_h(χ) * r * jnp.sin(2 * jnp.pi * χ) * jnp.ones(1)

    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array([_R(r, χ), _Y(r, χ), z * jnp.ones(1)]))

    # Set up finite element spaces
    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ('clamped', 'periodic', 'constant')

    # Set up differential forms and quadrature
    Λ0 = DifferentialForm(0, ns, ps, types)
    Λ2 = DifferentialForm(2, ns, ps, types)
    Λ3 = DifferentialForm(3, ns, ps, types)
    Q = QuadratureRule(Λ0, 3*p)
    ξ, _, _, _, _ = get_xi(_R, _Y, Λ0, Q)
    
    # Set up operators
    E2 = LazyExtractionOperator(Λ2, ξ, False).M
    E3 = LazyExtractionOperator(Λ3, ξ, False).M
    D = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
    M2 = LazyMassMatrix(Λ2, Q, F, E2).M
    M3 = LazyMassMatrix(Λ3, Q, F, E3).M
    
    # Assemble and solve the system K*u = lambda*M*u
    K = D @ jnp.linalg.solve(M2, D.T)
    evs, evecs = generalized_eigh(K, M3)
    
    return evs, evecs

# %%
def setup_target_shape(n_map, p, a, e):
    """
    Computes the discrete representation of an elliptical target shape.
    
    Returns:
        (a_target, radius_func): A tuple containing the discrete parameters
                                 and the analytical radius function.
    """
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p, 1, 1),
                            ('periodic', 'constant', 'constant'))
    Q = QuadratureRule(Λmap, 3*p)
    P_0 = Projector(Λmap, Q)
    M0 = LazyMassMatrix(Λmap, Q).M

    def radius_func(chi):
        b = a * e
        return a * b / (b**2 * jnp.cos(2 * jnp.pi * chi)**2 + a**2 * jnp.sin(2 * jnp.pi * chi)**2)**0.5

    def _radius_func_wrapped(x):
        return radius_func(x[0]) * jnp.ones(1)

    a_target = jnp.linalg.solve(M0, P_0(_radius_func_wrapped))
    return a_target, radius_func


def plot_reconstruction(a_hat, target_radius_func, target_evs, n, p, n_map, iter_num, output_dir):
    """
    Generates and saves a three-panel plot showing the current fitted radius,
    the first eigenfunction, and the eigenvalue spectrum.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.618])
    ax2 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 5))
    fig.suptitle(f'Iteration {iter_num}', fontsize=14)

    # --- Panel 1: Radius Plot ---
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p, 1, 1), ('periodic', 'constant', 'constant'))
    _radius_h_discrete = DiscreteFunction(a_hat, Λmap)

    def radius_h_func(x):
        return _radius_h_discrete(jnp.array([x, 0, 0]))

    chi_plot = jnp.linspace(0, 1, 200)
    ax1.plot(chi_plot, jax.vmap(radius_h_func)(chi_plot), label='Fitted Radius', color='purple')
    ax1.plot(chi_plot, jax.vmap(target_radius_func)(chi_plot),
             '--', label='Target Radius', color='k')
    ax1.set_xlabel(r'$\chi$')
    ax1.set_ylabel(r'$r(\chi)$')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 2: First Eigenfunction Plot ---
    evs, evecs = get_evs(a_hat, n, p, n_map)
    first_evec = evecs[:, 5]

    # Recreate mapping helpers for the current a_hat
    def a_h(x):
        _x = jnp.array([x, 0, 0])
        return _radius_h_discrete(_x)
    def _R(r, χ): return a_h(χ) * r * jnp.cos(2 * jnp.pi * χ) * jnp.ones(1)
    def _Y(r, χ): return a_h(χ) * r * jnp.sin(2 * jnp.pi * χ) * jnp.ones(1)
    def F(x): return jnp.ravel(jnp.array([_R(x[0], x[1]), _Y(x[0], x[1]), x[2] * jnp.ones(1)]))
    
    # Create grid in logical coordinates
    nx = 64
    eps = 1e-6
    r_coords = jnp.linspace(eps, 1.0 - eps, nx)
    chi_coords = jnp.linspace(0, 1.0, nx)
    z_coords = jnp.zeros(1)
    grid_logical = jnp.array(jnp.meshgrid(r_coords, chi_coords, z_coords))
    grid_logical = grid_logical.transpose(1, 2, 3, 0).reshape(nx * nx, 3)
    
    # Map grid to physical coordinates
    grid_physical = jax.vmap(F)(grid_logical)
    y1 = grid_physical[:, 0].reshape(nx, nx)
    y2 = grid_physical[:, 1].reshape(nx, nx)
    
    # Evaluate the eigenfunction on the grid
    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ('clamped', 'periodic', 'constant')
    Λ3 = DifferentialForm(3, ns, ps, types)
    Q_dummy = QuadratureRule(Λ3, 1) # Dummy Q for ξ
    ξ, _, _, _, _ = get_xi(_R, _Y, DifferentialForm(0, ns, ps, types), Q_dummy)
    E3 = LazyExtractionOperator(Λ3, ξ, False).M
    u_h = Pushforward(DiscreteFunction(first_evec, Λ3, E3), F, 3)
    
    # Normalize the sign of the eigenfunction for consistent plotting
    if u_h(jnp.array([0.0, 0, 0])) < 0:
        u_h_vals = -jax.vmap(u_h)(grid_logical).reshape(nx, nx)
    else:
        u_h_vals = jax.vmap(u_h)(grid_logical).reshape(nx, nx)
    
    ax2.contourf(y1, y2, u_h_vals, levels=15, cmap='plasma')
    x_mean = jnp.mean(y1)
    y_mean = jnp.mean(y2)
    ax2.set_xlim(x_mean-1, x_mean+1)
    ax2.set_ylim(y_mean-1, y_mean+1)
    ax2.set_aspect('equal', 'box')
    ax2.axis('off')

    # --- Panel 3: Eigenvalue Spectrum Plot ---
    k_evs = len(target_evs)
    current_evs = evs[:k_evs]
    k_plot = jnp.arange(1, k_evs + 1)
    
    ax3.plot(k_plot, target_evs, 'x--', color='k', label='Target Spectrum')
    ax3.plot(k_plot, current_evs, '^-', color='purple', label='Fitted Spectrum')
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$k$')
    ax3.set_ylabel(r'$\lambda_k$')
    ax3.grid(True, which="both", linestyle='--', alpha=0.6)
    
    # --- Panel 4: Difference in Eigenvalues Plot ---
    ax4.plot(k_plot, (current_evs - target_evs) / target_evs, 'o-', color='purple', label='Relative Error')
    ax4.set_xlabel(r'$k$')
    ax4.set_ylabel(r'$\Delta \lambda_k / \lambda_k$')
    ax4.grid(True, which="both", linestyle='--', alpha=0.6)
    
    # --- Save and close ---
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"solution_iter_{iter_num:04d}.pdf")
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved reconstruction plot to {filepath}")

# %%

def main():
    # --- Configuration ---
    N_PARAMS = 8
    N_MAP = 4
    POLY_DEGREE = 3
    K_EVS = (N_PARAMS - 2) * (N_PARAMS) # Number of eigenvalues to use in the loss function
    LEARNING_RATE = 5e-1
    NUM_STEPS = 500
    PLOT_EVERY = 10
    OUTPUT_DIR = "scripts/interactive/script_outputs/drumshape"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Problem Setup ---
    # Define a target ellipse shape and get its discrete representation
    a_target, target_radius_func = setup_target_shape(
        n_map=N_MAP, p=POLY_DEGREE, a=1.0, e=2/3
    )
    
    # Get the target eigenvalue spectrum
    target_evs_full, _ = get_evs(a_target, N_PARAMS, POLY_DEGREE, N_MAP)
    target_evs = target_evs_full[:K_EVS]

    # --- Loss Function ---
    def fit_evs(a_hat):
        """Computes the squared error between current and target spectra."""
        evs, _ = get_evs(a_hat, N_PARAMS, POLY_DEGREE, N_MAP)
        valid_evs = evs[:K_EVS]

        # Normalize by k to weigh smaller eigenvalues more
        k_norm = jnp.arange(1, 1+K_EVS)
        return jnp.sum(((valid_evs) - (target_evs))**2 * 1/k_norm**2)
    
    # --- Optimization ---
    # JIT-compile the function that computes both loss and gradient
    value_and_grad_fn = jax.jit(jax.value_and_grad(fit_evs))
    
    # Initialize parameters with a random perturbation around a circle
    key = jax.random.PRNGKey(1)
    a_hat = jnp.ones(N_MAP) * 0.8 + jax.random.normal(key, (N_MAP,)) * 0.2
    
    # Set up the optimizer
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    
    opt_state = optimizer.init(a_hat)

    print("--- Starting Shape Optimization ---")
    # Plot initial state before starting
    print("Plotting initial state (Iteration 0)...")
    plot_reconstruction(a_hat, target_radius_func, target_evs, N_PARAMS, POLY_DEGREE, N_MAP, 0, OUTPUT_DIR)

    for i in range(NUM_STEPS):
        value, grads = value_and_grad_fn(a_hat)
        
        updates, opt_state = optimizer.update(grads, opt_state, a_hat)
        a_hat = optax.apply_updates(a_hat, updates)
        
        # Plot and save the current solution periodically
        if (i + 1) % PLOT_EVERY == 0:
            print(f"Step {i+1:4d}/{NUM_STEPS}, Loss: {value:.6E}")
            plot_reconstruction(a_hat, target_radius_func, target_evs, N_PARAMS, POLY_DEGREE, N_MAP, i + 1, OUTPUT_DIR)

    print("\n--- Optimization Finished ---")
    
    # --- Final Analysis and Plotting ---
    print("Plotting final results...")
    
    # Plot final reconstructed shape and eigenfunction
    plot_reconstruction(a_hat, target_radius_func, target_evs, N_PARAMS, POLY_DEGREE, N_MAP, NUM_STEPS, OUTPUT_DIR)

    # Plot a comparison of the eigenvalue spectra
    plt.figure(figsize=(8, 6))
    k_plot = jnp.arange(K_EVS)
    circle_evs, _ = get_evs(jnp.ones(N_MAP), N_PARAMS, POLY_DEGREE, N_MAP)
    fit_evs_final, _ = get_evs(a_hat, N_PARAMS, POLY_DEGREE, N_MAP)
    
    plt.plot(k_plot, circle_evs[:K_EVS], 'o-', label='Circle (Initial Guess Basis)')
    plt.plot(k_plot, target_evs, 's-', label='Ellipse (Target)')
    plt.plot(k_plot, fit_evs_final[:K_EVS], '^-', label='Fitted Shape (Final)')
    plt.xlabel(r'Eigenvalue index $k$')
    plt.ylabel(r'Eigenvalue $\lambda_k$')
    plt.title('Comparison of Eigenvalue Spectra')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_spectra_comparison.pdf"))
    plt.show()

if __name__ == '__main__':
    main()
# %%