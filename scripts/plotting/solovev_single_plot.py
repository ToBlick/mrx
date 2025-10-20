# %%
import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.BoundaryFitting import cerfon_map, helical_map, rotating_ellipse_map
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Plotting import get_2d_grids, plot_crossections_separate, plot_torus

# %%
# %%
name = "FV2QFGrX"
with h5py.File("script_outputs/solovev/" + name + ".h5", "r") as f:
    B_hat = f["B_final"][:]
    p_hat = f["p_final"][:]
    helicity_trace = f["helicity_trace"][:]
    energy_trace = f["energy_trace"][:]
    force_trace = f["force_trace"][:]

    B_fields = f["B_fields"][:] if "B_fields" in f else None
    p_fields = f["p_fields"][:] if "p_fields" in f else None

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
B_hat = B_fields[4]
p_hat = p_fields[4]
# %%
eps = CONFIG["eps"]
dR = 0.2
kappa = CONFIG["kappa"]


def du(p):
    r, θ, ζ = p

    r_star = CONFIG["pert_radial_loc"]

    def rad(θ):
        a = jnp.cos(2 * jnp.pi * θ)**2 + kappa**2 * jnp.sin(2 * jnp.pi * θ)**2
        return r_star + dR * jnp.cos(2 * jnp.pi * θ) / a

    def phi(θ):
        c = jnp.cos(2 * jnp.pi * θ)
        s = jnp.sin(2 * jnp.pi * θ)
        rr = rad(θ)
        return jnp.arctan2(kappa * rr * s, rr * c - dR)

    def a(r, θ):
        return jnp.exp(- (r - rad(θ))**2 / (2 * CONFIG["pert_radial_width"]**2))
    B_rad = a(r, θ) * jnp.sin(phi(θ) * CONFIG["pert_pol_mode"]) * \
        jnp.sin(2 * jnp.pi * ζ * CONFIG["pert_tor_mode"])
    return B_rad * jnp.ones(1)


p_hat = jnp.linalg.solve(Seq.M0, Seq.P0(du))


p_h = Pushforward(DiscreteFunction(p_hat, Seq.Λ0, Seq.E0), F, 0)
# %%
cuts = jnp.linspace(0, 1, 5, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0,
                            ny=128, nz=128, z_min=0, z_max=1, invert_z=True)
# %%
fig, ax = plot_torus(p_h, grids_pol, grid_surface,
                     gridlinewidth=1, cstride=8, noaxes=True)
# plt.savefig(os.path.join("script_outputs", "solovev", "rotating_ellipse_3d.pdf"))
# %%
cuts = jnp.linspace(0, 1/5, 6, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
plot_crossections_separate(p_h, grids_pol, cuts, plot_centerline=True)
# plt.savefig("rotating_ellipse_cuts.pdf")
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
