# %%
"""
Interactive script to optimize the shape of a drum (a poloidal domain) to match a target eigenvalue spectrum.

This is "hearing" the shape of a drum by specifying the eigenvalues and figuring out the shape using 
inverse optimization.
"""

import os
import time
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import optax

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DifferentialForm, DiscreteFunction, Pushforward
from mrx.mappings import drumshape_map
from mrx.quadrature import QuadratureRule
from mrx.utils import assemble, integrate_against, inv33, jacobian_determinant

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)
script_dir = Path(__file__).parent / 'script_outputs'
script_dir.mkdir(parents=True, exist_ok=True)


# %%
def generalized_eigh(A: jnp.ndarray, B: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the generalized eigenvalue problem A*v = lambda*B*v.

    Args:
        A : jnp.ndarray
            Matrix appearing in the generalized eigenvalue problem A*v = lambda*B*v.
        B : jnp.ndarray
            Matrix appearing in the generalized eigenvalue problem A*v = lambda*B*v.

    Returns:
        eigenvalues : jnp.ndarray
            Eigenvalues
        eigenvectors_original : jnp.ndarray
            Eigenvectors in the original basis before Cholesky decomposition
    """
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
def get_evs(a_hat: jnp.ndarray, n_map: int, p_map: int, Seq: DeRhamSequence) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the eigenvalues and eigenvectors for a drum shape defined by a_hat.

    Args:
        a_hat: jnp.ndarray
            Discrete representation of the radius function r(χ).
        n_map: int 
            Number of elements in the map.
        p_map: int
            Polynomial degree in the map.
        Seq: DeRhamSequence
            DeRham sequence.

    Returns:
        eigenvalues : jnp.ndarray
            Eigenvalues
        eigenvectors : jnp.ndarray
            Eigenvectors
    """
    # Define the mapping from the parameter χ to the radius function
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                            ('periodic', 'constant', 'constant'))
    _a_h = DiscreteFunction(a_hat, Λmap)

    def a_h(x):
        _x = jnp.array([x, 0, 0])
        return _a_h(_x)

    F = drumshape_map(a_h=lambda χ: a_h(χ)[0])

    # We now assemble the matrices by hand
    # TODO: Make the deRhamSequence class compatible with jax transformations
    # by extending NamedTuple

    def G(x: jnp.ndarray) -> jnp.ndarray:
        """Metric tensor from the coordinate transformation. Formula is:

        G = F.T @ F

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            G: Metric tensor.
        """
        return jax.jacfwd(F)(x).T @ jax.jacfwd(F)(x)

    G_jkl = jax.vmap(G)(Seq.Q.x)
    G_inv_jkl = jax.vmap(inv33)(G_jkl)
    J_j = jax.vmap(jacobian_determinant(F))(Seq.Q.x)

    K = assemble(Seq.get_d_Lambda_0_ijk,
                 Seq.get_d_Lambda_0_ijk,
                 G_inv_jkl * J_j[:, None, None] * Seq.Q.w[:, None, None],
                 Seq.Lambda_0.n,
                 Seq.Lambda_0.n)
    K = Seq.E0 @ K @ Seq.E0.T

    M = assemble(Seq.get_Lambda_0_ijk,
                 Seq.get_Lambda_0_ijk,
                 J_j[:, None, None] * Seq.Q.w[:, None, None],
                 Seq.Lambda_0.n,
                 Seq.Lambda_0.n)
    M = Seq.E0 @ M @ Seq.E0.T

    evs, evecs = generalized_eigh(K, M)
    return evs, evecs

# %%


def setup_target_shape(n_map: int, p_map: int, a: float, e: float) -> tuple[jnp.ndarray, Callable]:
    """
    Computes the discrete representation of an elliptical target shape.

    Args:
        n_map: int
            Number of elements in the map.
        p_map: int
            Polynomial degree in the map.
        a: float
            Radius of the drum.
        e: float
            Eccentricity of the target shape.

    Returns:
        (a_target, radius_func): A tuple containing the discrete parameters
                                 and the analytical radius function.
    """
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                            ('periodic', 'constant', 'constant'))
    Q = QuadratureRule(Λmap, 3*p_map)

    def get_Λmap_ijk(a: int, j: int, k: int) -> float:
        """
        Gets the value of the map at a given point.

        Args:
            a: Index of the map.
            j: Index of the quadrature point.
            k: Index of the component of the map.
        """
        return Λmap[a](Q.x[j])[k]

    M0 = assemble(
        get_Λmap_ijk,
        get_Λmap_ijk,
        Q.w[:, None, None],
        Λmap.n,
        Λmap.n,
    )

    def radius_func(x: jnp.ndarray) -> jnp.ndarray:
        """Elliptical radius function. Formula is:

        r(θ) = a * b / (b**2 * cos(2πθ)**2 + a**2 * sin(2πθ)**2)**0.5

        Args:
            x: Input logical coordinates (θ, 0, 0)

        Returns:
            r: Radius function.
        """
        if jnp.size(x) > 1:
            θ = x[0]
        else:
            θ = x
        b = a * e
        return a * b / (b**2 * jnp.cos(2 * jnp.pi * θ)**2 + a**2 * jnp.sin(2 * jnp.pi * θ)**2)**0.5 * jnp.ones(1)

    rad_fct_jk = jax.vmap(radius_func)(Q.x) * Q.w[:, None]  # (n_q, 1)
    a_target = jnp.linalg.solve(
        M0, integrate_against(get_Λmap_ijk, rad_fct_jk, Λmap.n))
    return a_target, radius_func

# %%


def plot_reconstruction(a_hat: jnp.ndarray,
                        target_radius_func: Callable,
                        target_evs: jnp.ndarray,
                        Seq: DeRhamSequence,
                        n_map: int,
                        p_map: int,
                        iter_num: int,
                        output_dir: Path,
                        loss_history: list[float] = None,
                        max_iters: int = None,
                        legends: bool = True) -> None:
    """
    Generates and saves a three-panel plot showing the current fitted radius,
    the first eigenfunction, and the eigenvalue spectrum. Also includes a
    small panel that tracks the loss over iterations (bottom-left).

    Args:
        a_hat: jnp.ndarray
            Discrete representation of the radius function r(χ).
        target_radius_func: Callable
            Target radius function.
        target_evs: jnp.ndarray
            Target eigenvalues.
        Seq: DeRhamSequence
            DeRham sequence.
        n_map: int
            Number of elements in the map.
        p_map: int
            Polynomial degree in the map.
        iter_num: int
            Iteration number.
        output_dir: Path
            Output directory.
        loss_history: list[float]
            Loss history. Defaults to None.
        max_iters: int
            Maximum number of iterations. Defaults to None.
        legends: bool
            Whether to show legends. Defaults to True.
    """
    # Set some plotting variables
    LABEL_SIZE = 18
    TICK_SIZE = 16
    LINE_WIDTH = 3
    LEGEND_SIZE = 18

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.618])

    # Left column: contour occupies first two rows, loss history bottom row
    ax2 = fig.add_subplot(gs[0:2, 0])  # contour (spans rows 0 and 1)
    ax_err = fig.add_subplot(gs[2, 0])  # loss history (bottom-left)

    # Right column: top to bottom
    ax1 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])

    # --- Panel 1 (right-top): Radius Plot ---
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 1, 1),
                            ('periodic', 'constant', 'constant'))
    _radius_h_discrete = DiscreteFunction(a_hat, Λmap)

    def radius_h_func(x: jnp.ndarray) -> jnp.ndarray:
        """Wrapper for the discrete radius function."""
        return _radius_h_discrete(jnp.array([x, 0, 0]))

    θ_plot = jnp.linspace(0, 1, 200)
    ax1.plot(θ_plot, jax.vmap(radius_h_func)(θ_plot),
             label=r'Fitted Radius', color='purple', linewidth=LINE_WIDTH)
    ax1.plot(θ_plot, jax.vmap(target_radius_func)(θ_plot),
             ':', label='Target Radius', color='k', linewidth=LINE_WIDTH)
    ax1.set_xlabel(r'$\theta$', fontsize=LABEL_SIZE)
    ax1.set_ylabel(r'$a(\theta)$', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    if legends:
        ax1.legend(fontsize=LEGEND_SIZE)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 2 (left big): First Eigenfunction Contour ---
    evs, evecs = get_evs(a_hat, n_map, p_map, Seq)
    first_evec = evecs[:, 0]

    # Recreate mapping helpers for the current a_hat
    Λmap = DifferentialForm(0, (n_map, 1, 1), (p_map, 1, 1),
                            ('periodic', 'constant', 'constant'))
    _a_h = DiscreteFunction(a_hat, Λmap)

    def a_h(x: jnp.ndarray) -> jnp.ndarray:
        """Wrapper for the a_h(χ) function."""
        _x = jnp.array([x, 0, 0])
        return _a_h(_x)

    def F(x: jnp.ndarray) -> jnp.ndarray:
        """Polar coordinate mapping function. Formula is:

        F(r, χ, z) = (a_h(χ) r cos(2πχ), -z, a_h(χ) r sin(2πχ))

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            F: Coordinate mapping function (a_h(χ) r cos(2πχ), -z, a_h(χ) r sin(2πχ))
        """
        r, χ, z = x
        return jnp.array([a_h(χ)[0] * r * jnp.cos(2 * jnp.pi * χ),
                          -z,
                          a_h(χ)[0] * r * jnp.sin(2 * jnp.pi * χ)])

    # Create grid in logical coordinates
    nx = 64
    eps = 1e-6
    r_coords = jnp.linspace(eps, 1.0 - eps, nx)
    θ_coords = jnp.linspace(0, 1.0, nx)
    z_coords = jnp.zeros(1)
    grid_logical = jnp.array(jnp.meshgrid(r_coords, θ_coords, z_coords))
    grid_logical = grid_logical.transpose(1, 2, 3, 0).reshape(nx * nx, 3)

    # Map grid to physical coordinates
    grid_physical = jax.vmap(F)(grid_logical)
    y1 = grid_physical[:, 0].reshape(nx, nx)
    y2 = grid_physical[:, 2].reshape(nx, nx)

    # Evaluate the eigenfunction on the grid
    u_h = Pushforward(DiscreteFunction(first_evec, Seq.Lambda_0, Seq.E0), F, 0)

    # Fix the sign of the eigenfunction for consistent plotting
    if u_h(jnp.array([0.0, 0, 0])) < 0:
        u_h_vals = -jax.vmap(u_h)(grid_logical).reshape(nx, nx)
    else:
        u_h_vals = jax.vmap(u_h)(grid_logical).reshape(nx, nx)

    ax2.contourf(y1, y2, u_h_vals, levels=15, cmap='plasma')
    # x_mean = jnp.mean(y1)
    # y_mean = jnp.mean(y2)
    # ax2.set_xlim(x_mean-1, x_mean+1)
    # ax2.set_ylim(y_mean-1, y_mean+1)
    ax2.set_aspect('equal', 'box')
    ax2.axis('off')

    # --- Panel 3 (right-middle): Eigenvalue Spectrum Plot ---
    k_evs = len(target_evs)
    current_evs = evs[:k_evs]
    k_plot = jnp.arange(1, k_evs + 1)

    ax3.plot(k_plot, target_evs, ':', color='k',
             label='target spectrum', linewidth=LINE_WIDTH)
    ax3.plot(k_plot, current_evs, '-',
             color='purple', label='fitted spectrum', linewidth=LINE_WIDTH)
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$k$', fontsize=LABEL_SIZE)
    ax3.set_ylabel(r'$\lambda_k$', fontsize=LABEL_SIZE)
    ax3.tick_params(axis='y', labelsize=TICK_SIZE)
    ax3.tick_params(axis='x', labelsize=TICK_SIZE)
    if legends:
        ax3.legend(fontsize=LEGEND_SIZE)
    ax3.grid(True, which="both", linestyle='--', alpha=0.6)

    # --- Panel 4 (right-bottom): Eigenvalue Difference (log scale, abs rel diff) ---
    # Use absolute relative difference and plot on a log scale
    rel_diff = jnp.abs((current_evs - target_evs) / target_evs)
    # Convert to python lists for matplotlib
    k_plot_list = list(range(1, k_evs + 1))
    rel_diff_list = jnp.asarray(rel_diff).tolist()
    ax4.plot(k_plot_list, rel_diff_list, 'o',
             color='purple', label='relative Error')
    ax4.set_yscale('log')
    ax4.set_xlabel(r'$k$', fontsize=LABEL_SIZE)
    ax4.set_ylabel(r'$|\lambda_k - \lambda_k^*| / \lambda_k$',
                   fontsize=LABEL_SIZE)
    ax4.tick_params(axis='y', labelsize=TICK_SIZE)
    ax4.tick_params(axis='x', labelsize=TICK_SIZE)
    ax4.grid(True, which='both', linestyle='--', alpha=0.6)
    ax4.set_ylim(1e-6, 1.0)

    # --- Panel err (bottom-left): Loss history over iterations ---
    if loss_history is not None:
        xs = list(range(len(loss_history)))
        ys = [float(v) for v in loss_history]
        # plot loss in log-scale and use purple to match other plots
        ax_err.plot(xs, ys, '-', color='purple', linewidth=LINE_WIDTH)
        ax_err.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
        ax_err.set_ylabel(
            r'$\sum_k ( |\lambda_k - \lambda_k^*| / \lambda_k^* )^2$', fontsize=LABEL_SIZE)
        ax_err.set_yscale('log')
        ax_err.tick_params(axis='y', labelsize=TICK_SIZE)
        ax_err.tick_params(axis='x', labelsize=TICK_SIZE)
        ax_err.grid(True, which="both", linestyle='--', alpha=0.6)
        ax_err.set_ylim(1e-7, 1.1 * loss_history[0])
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
    # Numerical parameters to use
    N_PARAMS = 8
    N_MAP = 8
    P_MAP = 3
    POLY_DEGREE = 3

    # max. Number of eigenvalues to use in the loss function
    # and other hyperparameters of the optimization and plotting
    K_EVS = 100
    LEARNING_RATE = 1e-1
    NUM_STEPS = 500
    PLOT_EVERY = 10

    # Set up finite element spaces
    ns = (N_PARAMS, N_PARAMS, 1)
    ps = (POLY_DEGREE, POLY_DEGREE, 0)
    q = 2 * POLY_DEGREE
    types = ("clamped", "periodic", "constant")

    F_default = drumshape_map(a_h=lambda χ: jnp.ones(1)[0])
    Seq = DeRhamSequence(ns, ps, q, types, F_default,
                         polar=True, dirichlet=True)
    Seq.evaluate_1d()

    # --- Problem Setup ---
    # Define a target ellipse shape and get its discrete representation
    a_target, target_radius_func = setup_target_shape(
        n_map=N_MAP, p_map=P_MAP, a=1.0, e=0.6
    )

    # Get the target eigenvalue spectrum
    target_evs_full, _ = get_evs(a_target, N_MAP, P_MAP, Seq)
    k_max = jnp.minimum(K_EVS, len(target_evs_full))
    target_evs = target_evs_full[:k_max]

    # --- Loss Function ---
    def fit_evs(a_hat: jnp.ndarray) -> float:
        """Computes the squared error between current and target spectra. Formula is:

        loss = sum_k ( (lambda_k - lambda_k^*) / lambda_k^* )^2

        Args:
            a_hat: Discrete representation of the radius function r(χ).

        Returns:
            loss: Squared error between current and target spectra.
        """
        evs, _ = get_evs(a_hat, N_MAP, P_MAP, Seq)
        valid_evs = evs[:k_max]

        return jnp.sum(((valid_evs) - (target_evs))**2 / target_evs**2) \
            + 0.0 * jnp.sum((a_hat)**2)

    # --- Optimization ---
    # JIT-compile the function that computes both loss and gradient
    value_and_grad_fn = jax.jit(jax.value_and_grad(fit_evs))
    value_fun = jax.jit(fit_evs)

    # Initialize parameters with a random perturbation around a circle
    key = jax.random.PRNGKey(1)
    a_hat = jnp.maximum(jnp.ones(N_MAP) + 0.5 *
                        jax.random.normal(key, (N_MAP,)), 0.01)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(a_hat)

    print("--- Starting Shape Optimization ---")
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
                        output_dir=script_dir,
                        loss_history=losses,
                        max_iters=NUM_STEPS
                        )

    # Start the optimization loop
    t1 = time.time()
    for i in range(NUM_STEPS):
        value, grad = value_and_grad_fn(a_hat)

        updates, opt_state = optimizer.update(
            grad, opt_state, a_hat, value=value, grad=grad, value_fn=value_fun)
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
                                output_dir=script_dir,
                                loss_history=losses,
                                max_iters=NUM_STEPS,
                                legends=False
                                )

    print("\n--- Optimization Finished ---")
    t2 = time.time()
    print(f"Total time for {NUM_STEPS} steps: {t2 - t1:.2f} seconds")
    print(f"Final Loss: {value:.6E}")


if __name__ == '__main__':
    main()
# %%
