# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx
import scipy as sp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.plotting import get_2d_grids, plot_scalar_fct_physical_logical

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

Ip = 1.95
It = 2.46
Is = jnp.array([Ip, It])
p = 2
q = p + 2
ɛ = 0.2
π = jnp.pi
μ0 = 1.0

# %%
r_inner = 0.33  # Inner normalized radius
r_outer = 1.0  # Outer normalized radius


@jax.jit
def F(x):
    """Hollow toroid. Formula is:
    F(r, θ, ζ) = (R cos(2πζ), -R sin(2πζ), ɛ (r + 1)/2 sin(2πθ))
    where R = 1 + ɛ (r + 1)/2 cos(2πθ) is the radial coordinate.
    Args: 
        x: (r, θ, ζ) in logical coordinates
    Returns:
        F: (x, y, z) in physical coordinates
    """
    r, θ, ζ = x
    r_phys = r_inner + (r_outer - r_inner) * (r + 1)/2
    R = 1 + ɛ * r_phys * jnp.cos(2 * π * θ)
    return jnp.array([R * jnp.cos(2 * π * ζ),
                      -R * jnp.sin(2 * π * ζ),
                      ɛ * r_phys * jnp.sin(2 * π * θ)])


# Set up finite element spaces
ns = (6, 8, 1)
ps = (p, p, 0)
types = ("clamped", "periodic", "constant")
Seq = DeRhamSequence(ns, ps, q, types, F, polar=False, dirichlet=False)
Seq.evaluate_1d()
Seq.assemble_all()
evs, evecs = sp.linalg.eigh(Seq.M1 @ Seq.dd1, Seq.M1)

# %%
# Assemble 0-form Laplacian
L0 = Seq.M0 @ Seq.dd0

mode1 = evecs[:, 0] / (evecs[:, 0] @ Seq.M1 @ evecs[:, 0])**0.5
mode2 = evecs[:, 1] / (evecs[:, 1] @ Seq.M1 @ evecs[:, 1])**0.5
# %%

inner_ring_quad = get_2d_grids(F, cut_axis=0, cut_value=0, ny=16, nz=1)[0]
R_inner = 1 + ɛ * (inner_ring_quad[:, 0] + 1) / \
    2 * jnp.cos(2 * π * inner_ring_quad[:, 1])
# Guess for boundary values
v_guess_full = 0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, ns[1] * ns[2]))
# Use reduced parameterization: optimize over n-1 values, last is determined by sum=0 constraint
v_guess_reduced = v_guess_full[:-1]


def reconstruct_v(v_reduced):
    """Reconstruct full v vector with zero-sum constraint."""
    return jnp.concatenate([v_reduced, jnp.array([-jnp.sum(v_reduced)])])


# Regularization parameter
λ_reg = 1e-3


@jax.jit
def B_sq_mismatch(v_guess):
    # In MRX's ordering, the outer boundary DOFs are the last n ones.
    g_hat = jnp.concatenate(
        [jnp.zeros(Seq.Lambda_0.n - ns[1] * ns[2]), v_guess])
    f_hat = jnp.linalg.lstsq(L0, g_hat)[0]
    B_hat = 0.0 * mode1 + 0.0 * mode2 + Seq.strong_grad @ f_hat
    B_h = DiscreteFunction(B_hat, Seq.Lambda_1)
    B_h_xyz = Pushforward(B_h, F, 1)
    B_sq_plasma = (1/R_inner)**2
    B_sq_vacuum = jnp.sum(jax.vmap(B_h_xyz)(inner_ring_quad)**2, axis=1)
    mismatch = 0 * jnp.sum((B_sq_vacuum - B_sq_plasma)**2)
    regularization = λ_reg * jnp.sum(g_hat**2)
    return mismatch + regularization, B_hat


# %%
# Optimize boundary values using BFGS with zero-sum constraint
# Loss function that operates on reduced parameters
def loss_fn(v_reduced, args):
    v_full = reconstruct_v(v_reduced)
    return B_sq_mismatch(v_full)[0]


# Set up BFGS optimizer
solver = optx.BFGS(rtol=1e-9, atol=1e-9)

# Run optimization on reduced parameters
initial_loss = loss_fn(v_guess_reduced, None)
result = optx.minimise(loss_fn, solver, v_guess_reduced, max_steps=10_000)
v_optimal_reduced = result.value
v_optimal = reconstruct_v(v_optimal_reduced)
loss_optimal = loss_fn(v_optimal_reduced, None)

print(f"Initial loss: {initial_loss:.6e}")
print(f"Final loss: {loss_optimal:.6e}")
print(f"Sum of v_optimal: {jnp.sum(v_optimal):.6e}")
print(f"Converged: {result.result == optx.RESULTS.successful}")

# Get the optimized B field
_, B_hat_optimal = B_sq_mismatch(v_optimal)
print(f"Converged: {result.result == optx.RESULTS.successful}")

# Get the optimized B field
_, B_hat_optimal = B_sq_mismatch(v_optimal)
# %%
B_h = DiscreteFunction(B_hat_optimal, Seq.Lambda_1)
B_h_xyz = Pushforward(B_h, F, 1)
# %%
B_sq_vacuum = jnp.sum(jax.vmap(B_h_xyz)(inner_ring_quad)**2, axis=1)
# %%
plt.plot(B_sq_vacuum, label='Vacuum B²', marker='o', linestyle='', alpha=0.7)
plt.plot(1/R_inner**2, label='Plasma B²', marker='d', linestyle='', alpha=0.7)
plt.grid(True)
plt.xlabel('Collocation point along inner ring (θ)')
plt.legend()
# %%


def B_h_r(x):
    return B_h(x)[0]


def B_h_theta(x):
    return B_h(x)[1]


def B_h_zeta(x):
    return B_h(x)[2]


def mod_B(x):
    return jnp.sum(B_h_xyz(x)**2)**0.5


# %%
g_hat = jnp.concatenate(
    [jnp.zeros(Seq.Lambda_0.n - ns[1] * ns[2]), v_optimal])
f_hat = jnp.linalg.lstsq(L0, g_hat)[0]
f_h = DiscreteFunction(f_hat, Seq.Lambda_0)
plot_scalar_fct_physical_logical(f_h, F)

# %%
plot_scalar_fct_physical_logical(mod_B, F)
# %%
