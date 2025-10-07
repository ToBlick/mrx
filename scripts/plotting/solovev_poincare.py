# %%
import os

import diffrax as dfx
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

from mrx.BoundaryFitting import cerfon_map, helical_map, rotating_ellipse_map
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Plotting import (
    get_2d_grids,
    plot_crossections,
    plot_crossections_separate,
    plot_torus,
    set_axes_equal,
)

# %%
name = "9zrUjmG9"
with h5py.File("../../script_outputs/solovev/" + name + ".h5", "r") as f:
    B_hat = f["B_final"][:]
    p_hat = f["p_final"][:]
    helicity_trace = f["helicity_trace"][:]
    energy_trace = f["energy_trace"][:]
    force_trace = f["force_trace"][:]

    CONFIG = {k: v for k, v in f["config"].attrs.items()}
    # decode strings back if needed
    CONFIG = {k: v.decode() if isinstance(v, bytes)
              else v for k, v in CONFIG.items()}
# %%
# Get the map and sequences back:
kappa = CONFIG["kappa"]
eps = CONFIG["eps"]
alpha = jnp.arcsin(CONFIG["delta"])

if CONFIG["type"] == "tokamak":
    F = cerfon_map(eps, kappa, alpha)
elif CONFIG["type"] == "helix":
    F = helical_map(epsilon=CONFIG["eps"], h=CONFIG["h_helix"],
                    n_turns=CONFIG["m_helix"], kappa=CONFIG["kappa"], alpha=alpha)
elif CONFIG["type"] == "rotating_ellipse":
    F = rotating_ellipse_map(eps, CONFIG["kappa"], CONFIG["m_rot"])
else:
    raise ValueError("Unknown configuration type.")
ns = (CONFIG["n_r"], CONFIG["n_theta"], CONFIG["n_zeta"])
ps = (CONFIG["p_r"], CONFIG["p_theta"], 0
      if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
q = max(ps)
types = ("clamped", "periodic",
         "constant" if CONFIG["n_zeta"] == 1 else "periodic")
print("Setting up FEM spaces...")
Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)

assert jnp.min(Seq.J_j) > 0, "Mapping is singular!"
# %%
F = jax.jit(F)


@jax.jit
def F_cyl_signed(x):
    """Cylindrical coords with signed R."""
    R = jnp.sqrt(x[0]**2 + x[1]**2) * \
        jnp.sign(jnp.sin(jnp.arctan2(x[1], x[0])))
    phi = jnp.arctan2(x[1], x[0])
    z = x[2]
    return jnp.array([R, phi, z])


p_h = DiscreteFunction(p_hat, Seq.Λ0, Seq.E0)
B_h = (DiscreteFunction(B_hat, Seq.Λ2, Seq.E2))
# %%
# at zeta = 0, we are in the (R, z) plane, so we sample there:
n_lines = 60  # even numbers only
r_min, r_max = 0.05, 1.0
p = 1.0
_r = np.linspace(r_min, r_max, n_lines)
x0s = np.vstack(
    (np.hstack((_r[::3], _r[1::3], _r[2::3])),                              # between 0 and 1 - samples along x
     # half go to theta=0 and half to theta=pi
     np.hstack((0.31 * np.ones(n_lines//3),
                0.43 * np.ones(n_lines//3),
                0.76 * np.ones(n_lines//3))),
     0.743 * np.ones(n_lines))
).T
# x0s = np.vstack(
#     (_r,
#      0.0 * np.ones(n_lines),
#      0.0 * np.ones(n_lines))
# ).T

# %%


def integrate_field_line_diffrax(B_h, x0, F, N, phi_target):
    """
    Integrate a field line x'(t) = v(x)/||v(x)|| using Diffrax, 
    detecting crossings where F(x)[1] passes target values.

    Args:
        v: Callable(x) -> vector field (R^3 -> R^3)
        F: Callable(x) -> map R^3 -> R^2 or higher (used for event detection)
        x0: initial position array (3,)
        targets: sequence of float target values to cross in F(x)[1]
        N: number of intersections to record
        t_span: (t0, tf)
        max_step: maximum step size
        rtol, atol: solver tolerances
    Returns:
        crossings: array of shape (N, 3) with intersection points
    """

    @jax.jit
    def vector_field(t, x, args):
        """Return the norm of B at x=(r,theta,zeta) in [0,1]^3."""
        x = x % 1.0
        r, θ, z = x
        r = jnp.clip(r, 1e-6, 1)  # avoid r=0 in polar coords
        x = jnp.array([r, θ, z])
        Bx = B_h(jnp.array(x))
        DFx = jax.jacfwd(F)(jnp.array(x))
        return Bx / jnp.linalg.norm(DFx @ Bx)

    term = dfx.ODETerm(vector_field)
    solver = dfx.Dopri5()

    def cond_fn(t, y, args, **kwargs):
        x = F(y)  # cartesian coords
        phi = (jnp.arctan2(x[1], x[0]) / jnp.pi + 1) / 2  # in [0, 1]
        return jnp.sin(2 * jnp.pi * (phi - phi_target))

    t0 = 0
    t1 = jnp.inf
    dt0 = 0.1
    term = dfx.ODETerm(vector_field)
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = dfx.Event(cond_fn, root_finder)
    solver = dfx.Tsit5()

    crossings = jnp.zeros((N, 3))

    for i in range(N):
        sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, x0, event=event)
        if sol.ys.size == 0:
            break
        x_cross = sol.ys[0]
        crossings = crossings.at[i, :].set(x_cross)
        # restart
        # step a bit forward to avoid finding the same root again
        x0 = (sol.ys[0] + 1e-6 * vector_field(0, sol.ys[0], None)) % 1.0

    return crossings


# %%
crossings = jax.vmap(lambda x0: integrate_field_line_diffrax(
    # m x N x 3
    B_h, x0, F, N=5000, phi_target=0.2, rtol=1e-6, atol=1e-9))(x0s)

# %%
crossings_xyz = jax.vmap(jax.vmap(F))(crossings)
crossings_Rphiz = jax.vmap(jax.vmap(F_cyl_signed))(crossings_xyz)
# %%

# ================== PLOT CONFIGURATION ==================
dot_width = 0.5
tick_label_size = 20
axis_label_size = 22
# --------------------------------------------------------

# --- Gap and Limit Detection ---
all_R_values = crossings_Rphiz[:, :, 0].flatten()
positive_R = all_R_values[all_R_values > 0]
negative_R = all_R_values[all_R_values < 0]

gap_left = 0.9 * np.max(negative_R) if len(negative_R) > 0 else -0.1
gap_right = 0.9 * np.min(positive_R) if len(positive_R) > 0 else 0.1

Rmin = -0.05 + np.min(all_R_values)
Rmax = 0.05 + np.max(all_R_values)
Zmin = -0.05 + np.min(crossings_Rphiz[:, :, 2])
Zmax = 0.05 + np.max(crossings_Rphiz[:, :, 2])

# 1. Find out which plot's data range is wider.
x_span_left = gap_left - Rmin
x_span_right = Rmax - gap_right
max_x_span = max(x_span_left, x_span_right)  # The width needed for the box

# 2. Find the height of the data.
y_span = Zmax - Zmin

# 3. Create two identical plot boxes
fig = plt.figure(figsize=(24, 12))
gs = gridspec.GridSpec(1, 2, wspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)

# =============================================================================

# --- Plotting Loop ---
colors = ['purple', 'black', 'teal']
for i, curve in enumerate(crossings_Rphiz):
    ax1.scatter(curve[curve[:, 0] < gap_left, 0], curve[curve[:, 0] < gap_left, 2],
                s=dot_width, color=colors[i % len(colors)], rasterized=True)
    ax2.scatter(curve[curve[:, 0] > gap_right, 0], curve[curve[:, 0] > gap_right, 2],
                s=dot_width, color=colors[i % len(colors)], rasterized=True)

# --- Set limits, aspect ratio, and styling ---
# Enforce strict axis bounds and disable autoscaling so Matplotlib doesn't
# expand the right-hand panel beyond the requested limits.
ax1.set_xlim(gap_left - max_x_span, gap_left, auto=False)
ax2.set_xlim(gap_right, gap_right + max_x_span, auto=False)
ax1.set_ylim(Zmin, Zmax, auto=False)
# # --- Ticks and Labels ---
ax1.xaxis.set_major_locator(MultipleLocator(0.2))
ax2.xaxis.set_major_locator(MultipleLocator(0.2))
ax1.yaxis.set_major_locator(MultipleLocator(0.2))

ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
ax2.tick_params(axis='x', which='major', labelsize=tick_label_size)
ax1.grid(True, linestyle="--", lw=0.5)
ax2.grid(True, linestyle="--", lw=0.5)

plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(axis='y', which='both', length=0)

ax1.set_ylabel(r"$z$", fontsize=axis_label_size)
fig.supxlabel(r"$\pm R$", fontsize=axis_label_size)

# # Use tight_layout for final margin adjustments
plt.tight_layout()
plt.subplots_adjust(bottom=0.08)  # Add a bit more space for the supxlabel

# --- Save and Show ---
# Create directory if it doesn't exist
output_dir = os.path.join("script_outputs", "solovev")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "helix_poincare.pdf"),
            dpi=400, bbox_inches=None)

plt.show()
# %%
# get iota profile

# integrate field line a bunch
n_lines = 24  # even numbers only
r_min, r_max = 0.05, 0.99
_r = np.linspace(r_min, r_max, n_lines)
x0s = np.vstack((_r, 0.325 * jnp.ones(n_lines), 0.413 * jnp.ones(n_lines))).T
# %%
p_h = DiscreteFunction(p_hat, Seq.Λ0, Seq.E0)
B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2)
# %%
crossings = jax.vmap(lambda x0: integrate_field_line_diffrax(
    # m x N x 3
    B_h, x0, F, N=1000, phi_target=0.2))(x0s)
# %%
crossings_xyz = jax.vmap(jax.vmap(F))(crossings)
crossings_Rphiz = jax.vmap(jax.vmap(F_cyl_signed))(crossings_xyz)
# %%


def get_iota(c):
    # crossings_Rphiz:M x 3
    _crossings = c[c[:, 1] > 0]
    # compute center
    center = jnp.mean(_crossings, axis=0)
    # compute angles
    angles = jnp.arctan2(_crossings[:, 2] - center[2],
                         _crossings[:, 0] - center[0])
    # unwrap
    angles = jnp.unwrap(angles)
    # fit line
    A = jnp.vstack([jnp.arange(len(angles)), jnp.ones(len(angles))]).T
    m, c = jnp.linalg.lstsq(A, angles, rcond=None)[0]
    return m / (2 * jnp.pi)


iotas = [get_iota(crossing) for crossing in crossings_Rphiz]

# %%
poincare = crossings_Rphiz

R = poincare[:, :, 0]
phi = poincare[:, :, 1]
Z = poincare[:, :, 2]

# Flatten arrays
R_flat = R.flatten()
phi_flat = phi.flatten()
Z_flat = Z.flatten()
iota_flat = np.repeat(iotas, poincare.shape[1]) * 2 * np.pi

# Keep only positive phi crossings
mask = phi_flat > 0
R_pos = R_flat[mask]
Z_pos = Z_flat[mask]
iota_pos = iota_flat[mask]

plt.figure(figsize=(7, 7))
sc = plt.scatter(R_pos, Z_pos, c=iota_pos, s=8,
                 cmap='plasma', edgecolor='none')

plt.xlabel(r"$R$", fontsize=16)
plt.ylabel(r"$Z$", fontsize=16)
plt.axis("equal")
sc = plt.scatter(R_pos, Z_pos, c=iota_pos, s=8,
                 cmap='plasma', edgecolor='none')
cbar = plt.colorbar(sc)

# Set the label and its font size
cbar.set_label(r"$2 \pi \iota$", fontsize=16)

# Set the tick label size
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.show()

# %%
