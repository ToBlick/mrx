# %%
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pushforward
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
import matplotlib.gridspec as gridspec

from mrx.Utils import inv33, jacobian_determinant

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# %%

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
@partial(jax.jit, static_argnames=["n_map", "p_map", "Seq"])
def get_evs(a_hat, n_map, p_map, Seq):
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
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                            ('periodic', 'constant', 'constant'))
    _a_h = DiscreteFunction(a_hat, Λmap)

    def a_h(x):
        _x = jnp.array([x, 0, 0])
        return _a_h(_x)

    def F(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.array([a_h(χ)[0] * r * jnp.cos(2 * jnp.pi * χ),
                          -z,
                          a_h(χ)[0] * r * jnp.sin(2 * jnp.pi * χ)])
    
    # We now assemble the matrices by hand 
    # TODO: Make the deRhamSequence class compatible with jax transformations
    # by extending NamedTuple
      
    def G(x):
        return jax.jacfwd(F)(x).T @ jax.jacfwd(F)(x)
    
    G_jkl = jax.vmap(G)(Seq.Q.x)
    G_inv_jkl = jax.vmap(inv33)(G_jkl)
    J_j = jax.vmap(jacobian_determinant(F))(Seq.Q.x)

    K = jnp.einsum("ijk,jkl,qjl,j,j->iq", Seq.dΛ0_ijk,
                        G_inv_jkl, Seq.dΛ0_ijk, J_j, Seq.Q.w)
    K = Seq.E0.matrix() @ K @ Seq.E0.matrix().T

    M0 = jnp.einsum("ijk,ljk,j,j->il", Seq.Λ0_ijk,
                        Seq.Λ0_ijk, J_j, Seq.Q.w)
    M0 = Seq.E0.matrix() @ M0 @ Seq.E0.matrix().T
    
    # Solve the system
    evs, evecs = generalized_eigh(K, M0)
    # finite_indices = evs > 0
    # jnp.ispositive
    # evs = evs[finite_indices]
    return evs, evecs

# %%
def setup_target_shape(n_map, p_map, a, e):
    """
    Computes the discrete representation of an elliptical target shape.
    
    Returns:
        (a_target, radius_func): A tuple containing the discrete parameters
                                 and the analytical radius function.
    """
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 1, 1),
                            ('periodic', 'constant', 'constant'))
    Q = QuadratureRule(Λmap, 3*p_map)
    P_0 = Projector(Λmap, Q)
    M0 = LazyMassMatrix(Λmap, Q).matrix()

    def radius_func(chi):
        b = a * e
        return a * b / (b**2 * jnp.cos(2 * jnp.pi * chi)**2 + a**2 * jnp.sin(2 * jnp.pi * chi)**2)**0.5

    def _radius_func_wrapped(x):
        return radius_func(x[0]) * jnp.ones(1)

    a_target = jnp.linalg.solve(M0, P_0(_radius_func_wrapped))
    return a_target, radius_func

# %%
def plot_reconstruction(a_hat, 
                        target_radius_func, 
                        target_evs, 
                        Seq, 
                        n_map,
                        p_map,
                        iter_num, 
                        output_dir,
                        loss_history=None,
                        max_iters=None):
    """
    Generates and saves a three-panel plot showing the current fitted radius,
    the first eigenfunction, and the eigenvalue spectrum. Also includes a
    small panel that tracks the loss over iterations (bottom-left).
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.618])

    # Left column: contour occupies first two rows, loss history bottom row
    ax2 = fig.add_subplot(gs[0:2, 0])  # contour (spans rows 0 and 1)
    ax_err = fig.add_subplot(gs[2, 0])  # loss history (bottom-left)

    # Right column: top to bottom
    ax1 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    
    fig.suptitle(f'Iteration {iter_num}', fontsize=14)

    # --- Panel 1 (right-top): Radius Plot ---
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 1, 1), ('periodic', 'constant', 'constant'))
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

    # --- Panel 2 (left big): First Eigenfunction Contour ---
    evs, evecs = get_evs(a_hat, n_map, p_map, Seq)
    first_evec = evecs[:, 0]

    # Recreate mapping helpers for the current a_hat
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 1, 1),
                            ('periodic', 'constant', 'constant'))
    _a_h = DiscreteFunction(a_hat, Λmap)

    def a_h(x):
        _x = jnp.array([x, 0, 0])
        return _a_h(_x)

    def F(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.array([a_h(χ)[0] * r * jnp.cos(2 * jnp.pi * χ),
                          -z,
                          a_h(χ)[0] * r * jnp.sin(2 * jnp.pi * χ)])
        
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
    y2 = grid_physical[:, 2].reshape(nx, nx)
    
    # Evaluate the eigenfunction on the grid
    u_h = Pushforward(
        DiscreteFunction(
            first_evec, Seq.Λ0, Seq.E0.matrix()), 
        F, 
        0)
    
    # Fix the sign of the eigenfunction for consistent plotting
    if u_h(jnp.array([0.0, 0, 0])) < 0:
        u_h_vals = -jax.vmap(u_h)(grid_logical).reshape(nx, nx)
    else:
        u_h_vals = jax.vmap(u_h)(grid_logical).reshape(nx, nx)
    
    ax2.contourf(y2, y1, u_h_vals, levels=15, cmap='plasma')
    x_mean = jnp.mean(y1)
    y_mean = jnp.mean(y2)
    ax2.set_ylim(x_mean-1, x_mean+1)
    ax2.set_xlim(y_mean-1, y_mean+1)
    ax2.set_aspect('equal', 'box')
    ax2.axis('off')

    # --- Panel 3 (right-middle): Eigenvalue Spectrum Plot ---
    k_evs = len(target_evs)
    current_evs = evs[:k_evs]
    k_plot = jnp.arange(1, k_evs + 1)
    
    ax3.plot(k_plot, target_evs, 'x--', color='k', label='Target Spectrum')
    ax3.plot(k_plot, current_evs, '^-', color='purple', label='Fitted Spectrum')
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$k$')
    ax3.set_ylabel(r'$\lambda_k$')
    ax3.grid(True, which="both", linestyle='--', alpha=0.6)
    
    # --- Panel 4 (right-bottom): Eigenvalue Difference (log scale, abs rel diff) ---
    # Use absolute relative difference and plot on a log scale
    rel_diff = jnp.abs((current_evs - target_evs) / target_evs)
    # Convert to python lists for matplotlib
    k_plot_list = list(range(1, k_evs + 1))
    rel_diff_list = jnp.asarray(rel_diff).tolist()
    ax4.plot(k_plot_list, rel_diff_list, 'o-', color='purple', label='Absolute Relative Error')
    ax4.set_yscale('log')
    ax4.set_xlabel(r'$k$')
    ax4.set_ylabel(r'$|\Delta \lambda_k| / \lambda_k$')
    ax4.grid(True, which='both', linestyle='--', alpha=0.6)
    ax4.set_ylim(1e-5, 1.0)
    
    # --- Panel err (bottom-left): Loss history over iterations ---
    if loss_history is not None:
        xs = list(range(len(loss_history)))
        ys = [float(v) for v in loss_history]
        # plot loss in log-scale and use purple to match other plots
        ax_err.plot(xs, ys, '-', color='purple', linewidth=2)
        ax_err.set_xlabel('Iteration')
        ax_err.set_ylabel('Loss')
        ax_err.set_yscale('log')
        ax_err.grid(True, which="both", linestyle='--', alpha=0.6)
        ax_err.set_ylim(1e-5, 1.1 * loss_history[0])
        # Fix x-axis length if requested
        if max_iters is not None:
            ax_err.set_xlim(0, int(max_iters))
    else:
        # If no loss history is provided, hide axis
        ax_err.axis('off')
    
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
    N_MAP = 6
    P_MAP = 3
    POLY_DEGREE = 3
    K_EVS = (N_PARAMS - 3) * (N_PARAMS) # Number of eigenvalues to use in the loss function
    LEARNING_RATE = 5e-2
    NUM_STEPS = 300
    PLOT_EVERY = 1
    OUTPUT_DIR = "scripts/interactive/script_outputs/drumshape"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Problem Setup ---
    # Define a target ellipse shape and get its discrete representation
    a_target, target_radius_func = setup_target_shape(
        n_map=N_MAP, p_map=P_MAP, a=1.0, e=0.6
    )
    
    # Set up finite element spaces
    ns = (N_PARAMS, N_PARAMS, 1)
    ps = (POLY_DEGREE, POLY_DEGREE, 0)
    q = 3
    types = ("clamped", "periodic", "constant")
    bcs = ("dirichlet", "none", "none")
    
    def F_default(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.array([r * jnp.cos(2 * jnp.pi * χ),
                          -z,
                          r * jnp.sin(2 * jnp.pi * χ)])

    Seq = DeRhamSequence(ns, ps, q, types, bcs, F_default, polar=True)
    
    # Get the target eigenvalue spectrum
    target_evs_full, _ = get_evs(a_target, N_MAP, P_MAP, Seq)
    target_evs = target_evs_full[:K_EVS]
    k_norm = jnp.arange(0, K_EVS)
    
    # --- Loss Function ---
    def fit_evs(a_hat):
        """Computes the squared error between current and target spectra."""
        evs, _ = get_evs(a_hat, N_MAP, P_MAP, Seq)
        valid_evs = evs[:K_EVS]

        # Normalize by k to weigh smaller eigenvalues more
        return jnp.sum(((valid_evs) - (target_evs))**2 * jnp.exp(-0.5*k_norm)) \
            + 0.01 * jnp.sum((a_hat - 0.8)**2) / N_PARAMS 
    
    # --- Optimization ---
    # JIT-compile the function that computes both loss and gradient
    value_and_grad_fn = jax.jit(jax.value_and_grad(fit_evs))
    
    # Initialize parameters with a random perturbation around a circle
    key = jax.random.PRNGKey(123)
    a_hat = jnp.ones(N_MAP) * 0.8 + jax.random.normal(key, (N_MAP,)) * 0.5
    
    # Set up the optimizer
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    
    opt_state = optimizer.init(a_hat)

    print("--- Starting Shape Optimization ---")
    # Plot initial state before starting
    print("Plotting initial state (Iteration 0)...")
    # evaluate initial loss and start loss history
    value0, _ = value_and_grad_fn(a_hat)
    losses = [float(value0)]
    plot_reconstruction(a_hat, 
                        target_radius_func,
                        target_evs,
                        Seq,
                        n_map=N_MAP,
                        p_map=P_MAP,
                        iter_num=0,
                        output_dir=OUTPUT_DIR,
                        loss_history=losses,
                        max_iters=NUM_STEPS
                    )

    for i in range(NUM_STEPS):
        value, grads = value_and_grad_fn(a_hat)
        
        updates, opt_state = optimizer.update(grads, opt_state, a_hat)
        a_hat = optax.apply_updates(a_hat, updates)
        
        # record loss
        losses.append(float(value))
        
        # Plot and save the current solution periodically
        if (i + 1) % PLOT_EVERY == 0:
            print(f"Step {i+1:4d}/{NUM_STEPS}, Loss: {value:.6E}")
            plot_reconstruction(a_hat, 
                        target_radius_func,
                        target_evs,
                        Seq,
                        n_map=N_MAP,
                        p_map=P_MAP,
                        iter_num=i+1,
                        output_dir=OUTPUT_DIR,
                        loss_history=losses,
                        max_iters=NUM_STEPS
                    )

    print("\n--- Optimization Finished ---")
    
    # --- Final Analysis and Plotting ---
    print("Plotting final results...")
    
    # Plot final reconstructed shape and eigenfunction
    plot_reconstruction(a_hat, 
                        target_radius_func,
                        target_evs,
                        Seq,
                        n_map=N_MAP,
                        p_map=P_MAP,
                        iter_num=NUM_STEPS,
                        output_dir=OUTPUT_DIR,
                        loss_history=losses,
                        max_iters=NUM_STEPS
                    )

    # Plot a comparison of the eigenvalue spectra
    plt.figure(figsize=(8, 6))
    k_plot = jnp.arange(K_EVS)
    circle_evs, _ = get_evs(jnp.ones(N_MAP),N_MAP, P_MAP, Seq)
    fit_evs_final, _ = get_evs(a_hat, N_MAP, P_MAP, Seq)
    
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