# %%
import os

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import cerfon_map, helical_map, rotating_ellipse_map
from mrx.plotting import get_2d_grids, plot_crossections_separate, plot_torus

# %%
name = "helix/helix_qstar_3.0_m_2_h_0.05_16x16x8"
with h5py.File("script_outputs/" + name + ".h5", "r") as f:
    CONFIG = {k: v for k, v in f["config"].attrs.items()}
    # decode strings back if needed
    CONFIG = {k: v.decode() if isinstance(v, bytes)
              else v for k, v in CONFIG.items()}
    B_final = f["B_final"][:]
    p_final = f["p_final"][:]
    iterations = f["iterations"][:]
    force_trace = f["force_trace"][:]
    velocity_trace = f["velocity_trace"][:]
    helicity_trace = f["helicity_trace"][:]
    energy_trace = f["energy_trace"][:]
    divergence_B_trace = f["divergence_B_trace"][:]
    picard_iterations = f["picard_iterations"][:]
    picard_errors = f["picard_errors"][:]
    timesteps = f["timesteps"][:]
    total_time = f["total_time"][0]
    time_setup = f["time_setup"][0]
    time_solve = f["time_solve"][0]
    wall_time_trace = f["wall_time_trace"][:]
    if CONFIG["save_B"]:
        # B_fields = f["B_fields"][:]
        p_fields = f["p_fields"][:]
        B_fields = f["B_fields"][:]
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
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

FIG_SIZE = (8, 6)
LABEL_SIZE = 20
TICK_SIZE = 16
LINE_WIDTH = 2.5
LEGEND_SIZE = 16

color1 = 'purple'   # Force trace
color2 = 'black'    # Helicity change
color3 = 'teal'  # Divergence B
color4 = 'orange'

xticks = jnp.arange(0, jnp.max(iterations) + 1, 5000)

# --- Data ---
helicity_change = jnp.abs(jnp.array(helicity_trace - helicity_trace[0]))
divB_change = jnp.abs(jnp.array(divergence_B_trace - divergence_B_trace[0]))
energy_change = jnp.abs(jnp.array(energy_trace - energy_trace[0]))
all_timesteps = jnp.concatenate((jnp.ones(1) * CONFIG["dt"], timesteps))

# === Create two vertically stacked axes (sharing x) ===
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=FIG_SIZE,
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05}
)

# --- Define ranges ---
cut_low, cut_high = 2.0*jnp.maximum(jnp.max(helicity_change),
                                    jnp.max(divergence_B_trace)), 0.9*jnp.min(force_trace)

# === Top axis ===
ax_top.set_yscale('log')
# from high values down to cut
ax_top.set_ylim(cut_high, 1.5*jnp.max(force_trace))
ax_top.tick_params(axis='y', labelsize=TICK_SIZE)
ax_top.grid(which="both", linestyle="--", linewidth=0.5)

# === Bottom axis ===
ax_bottom.set_yscale('log')
ax_bottom.set_ylim(1e-16, cut_low)  # from cut down to small values
ax_bottom.tick_params(axis='x', labelsize=TICK_SIZE)
ax_bottom.tick_params(axis='y', labelsize=TICK_SIZE)
ax_bottom.grid(which="both", linestyle="--", linewidth=0.5)

# === Plot data on both ===
for ax in [ax_top, ax_bottom]:
    ax.plot(iterations, force_trace,
            label=r'$\| \, J \times B - \mathrm{grad} \, p \| / \| \mathrm{grad} \, p \|$',
            color=color1, lw=LINE_WIDTH, linestyle='-')
    ax.plot(iterations, helicity_change,
            label=r'$| \mathcal{H} - \mathcal{H}^0 |$', color=color2, lw=LINE_WIDTH, linestyle='--')
    ax.plot(iterations, divB_change,
            label=r'$\| \mathrm{div} \, B \|$', color=color3, lw=LINE_WIDTH, linestyle=':')
    # ax.plot(iterations, all_timesteps,
    #         label=r'$\delta t$', color=color4, lw=LINE_WIDTH, linestyle='-')
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks / 1e3).astype(int))
    ax.grid(which="both", linestyle="--", linewidth=0.5)


# === Labels and legend ===
ax_bottom.set_xlabel(r'Iteration $(×10^3)$', fontsize=LABEL_SIZE)
ax_bottom.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
ax_bottom.xaxis.get_offset_text().set_x(0.95)
lines, labels = ax_top.get_legend_handles_labels()
ax_top.legend(lines, labels, loc='upper right',
              fontsize=LEGEND_SIZE, frameon=True)

fig.tight_layout()

plt.savefig(os.path.join("script_outputs", f"{name}_trace_plot.pdf"))

plt.show()
# %%
B_hat = B_fields[-1]
p_hat = p_fields[-1]
# %%
# eps = CONFIG["eps"]
# dR = 0.0
# kappa = 1.0 # CONFIG["kappa"]


# def du(p):
#     r, θ, ζ = p

#     r_star = CONFIG["pert_radial_loc"]

#     def rad(θ):
#         a = jnp.cos(2 * jnp.pi * θ)**2 + kappa**2 * jnp.sin(2 * jnp.pi * θ)**2
#         return r_star + dR * jnp.cos(2 * jnp.pi * θ) / a

#     def phi(θ):
#         c = jnp.cos(2 * jnp.pi * θ)
#         s = jnp.sin(2 * jnp.pi * θ)
#         rr = rad(θ)
#         return jnp.arctan2(kappa * rr * s, rr * c - dR)

#     def a(r, θ):
#         return jnp.exp(- (r - rad(θ))**2 / (2 * CONFIG["pert_radial_width"]**2))
#     B_rad = a(r, θ) * jnp.sin(phi(θ) * CONFIG["pert_pol_mode"]) * \
#         jnp.sin(2 * jnp.pi * ζ * CONFIG["pert_tor_mode"])
#     return B_rad * jnp.ones(1)

# Seq.evaluate_1d()
# Seq.assemble_M0()
# p_hat = jnp.linalg.solve(Seq.M0, Seq.P0(du))
# %%
# F = helical_map(epsilon=0.33, h=0.2, n_turns=3, kappa=1.0, alpha=0.0)
# %%
p_h = Pushforward(DiscreteFunction(p_hat, Seq.Lambda_0, Seq.E0), F, 0)
# %%
cuts = jnp.linspace(0, 1, 9, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0,
                            ny=128, nz=128, z_min=0, z_max=1, invert_z=True)
fig, ax = plot_torus(p_h, grids_pol, grid_surface,
                     gridlinewidth=1, cstride=8, noaxes=False, elev=15, azim=40)
plt.savefig(os.path.join("script_outputs", f"{name}_3d_plot.pdf"))

# %%
cuts = jnp.linspace(0, 1/5, 6, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
plot_crossections_separate(p_h, grids_pol, cuts, plot_centerline=True)
# %%
Seq.evaluate_1d()
p_avg = p_hat @ Seq.P0(lambda x: jnp.ones(1)) / (Seq.J_j @ Seq.Q.w)
print(f"p_avg = {p_avg:.3e}")

# %%
beta = p_avg / energy_trace[-1]
print(f"Beta = {beta:.3e}")
# %%
print("Final |JxB - grad p| / |grad p| =", force_trace[-1])
print("Initial |JxB - grad p| / |grad p| =", force_trace[0])


# %%
