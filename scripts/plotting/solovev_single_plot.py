# %%
import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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
name = "helix"
with h5py.File("script_outputs/solovev/" + name + ".h5", "r") as f:
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
delta = CONFIG["delta"]
kappa = CONFIG["kappa"]
q_star = CONFIG["q_star"]
eps = CONFIG["eps"]
alpha = jnp.arcsin(delta)
tau = q_star * eps * kappa * (1 + kappa**2) / (kappa + 1)
π = jnp.pi
gamma = CONFIG["gamma"]

if CONFIG["type"] == "tokamak":
    F = cerfon_map(eps, kappa, alpha)
elif CONFIG["type"] == "helix":
    F = helical_map(eps, CONFIG["h_helix"], CONFIG["m_helix"])
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
Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

# %%
p_h = Pushforward(DiscreteFunction(p_hat, Seq.Λ0, Seq.E0_0.matrix()), F, 0)
# %%
cuts = jnp.linspace(0, 1, 8, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0,
                            ny=128, nz=128, z_min=0, z_max=1, invert_z=True)
# %%
fig, ax = plot_torus(p_h, grids_pol, grid_surface,
                     gridlinewidth=1, cstride=8)
# plt.savefig(os.path.join("script_outputs",
#             "solovev", "rotating_ellipse_3d.pdf"))
# %%
plot_crossections_separate(p_h, grids_pol, cuts, plot_centerline=True)
# plt.savefig(os.path.join("script_outputs",
#             "solovev", "rotating_ellipse_cuts.pdf"))
# %%
p_avg = p_hat @ Seq.P0_0(lambda x: jnp.ones(1)) / (Seq.J_j @ Seq.Q.w)
beta = p_avg / energy_trace[-1]
print(f"Beta = {beta:.3e}")
# %%
print("Final |JxB - grad p| / |grad p| =", force_trace[-1])
print("Initial |JxB - grad p| / |grad p| =", force_trace[0])
# %%
