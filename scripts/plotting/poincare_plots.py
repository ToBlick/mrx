# %%
"""
Interactive script to load and visualize results from relaxation simulations.

This script can handle both single runs and multirun (sweep) outputs from Hydra.
It loads the HDF5 results files and uses the trace_plot function from mrx.plotting.
"""
from pathlib import Path
from omegaconf import OmegaConf
import yaml
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import tqdm

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import stellarator_map
from mrx.plotting import get_iota_log, get_periodic_intersections, integrate_fieldlines, poincare_plot

import matplotlib.pyplot as plt

# %%
# Configuration - set your output directory here
# For a single run:
# base_dir = Path("out/gvec_relaxation/20260203_120000")
# For a multirun:
# base_dir = Path("out/gvec_relaxation/multirun/20260203_120000")

n_scan = 2
n_vmap = 16
T = 500 * jnp.pi * 2 * 3
N = int(T)
n_traj = n_scan * n_vmap
plot_pressure = True
ks_thresh = 10
markersize = 0.005
zeta_values = [0.33]
dpi = 150
denom_max = 10
axis_margin=0.02

# %%
base_dir = Path("out/gvec_relaxation/20260206_072421")
outdir = base_dir / "poincare_plots"
outdir.mkdir(exist_ok=True)

# Load run_name from config.yaml in .hydra folder
config_path = base_dir / ".hydra" / "config.yaml"
with open(config_path, "r") as f:
    cfg = OmegaConf.create(yaml.safe_load(f))
run_name = cfg.get("run_name")
if run_name is None:
    raise KeyError("run_name not found in config.yaml")

results = {}
# read results file
results_file = base_dir / f"{run_name}.h5"
with h5py.File(results_file, "r") as f:
    for key in f.keys():
        results[key] = np.array(f[key])


# Iterate over intermediate_states.h5 one group at a time
trace_file = base_dir / "intermediate_states.h5"
def iter_traces(trace_file):
    with h5py.File(trace_file, "r") as f:
        for group_name in f:
            group = f[group_name]
            B_dof = np.array(group["B_dof"])
            p_dof = np.array(group["p_dof"])
            iteration = group.attrs["iteration"]
            force_norm = group.attrs["force_norm"]
            yield {
                "B_dof": B_dof,
                "p_dof": p_dof,
                "iteration": iteration,
                "force_norm": force_norm
            }
        
# %%
# build the map
# Map interpolation
ns_map = (cfg.map.ns_r, cfg.map.ns_theta, cfg.map.ns_zeta)
ps_map = (cfg.map.ps_r, cfg.map.ps_theta, cfg.map.ps_zeta)
quad_map = cfg.map.quad_order
# %%
map_seq = DeRhamSequence(ns_map, ps_map, quad_map, ("clamped", "periodic", "periodic"), 
                             lambda x: x, polar=False, dirichlet=False)
# %%
# get the map
nfp = cfg.nfp
R_dof = results["R_dof"]
Z_dof = results["Z_dof"]
X1_h = DiscreteFunction(R_dof, map_seq.Lambda_0, map_seq.E0)
X2_h = DiscreteFunction(Z_dof, map_seq.Lambda_0, map_seq.E0)
map = jax.jit(stellarator_map(X1_h, X2_h, nfp=nfp, flip_zeta=False))
# %%
# get the full sequence
ns = (cfg.fem.ns_r, cfg.fem.ns_theta, cfg.fem.ns_zeta)
ps = (cfg.fem.ps_r, cfg.fem.ps_theta, cfg.fem.ps_zeta)
seq = DeRhamSequence(
        ns, ps, cfg.fem.quad_order, 
        ("clamped", "periodic", "periodic"),
        map, polar=True, dirichlet=True
    )
# %%
integrate_fieldlines_jit = jax.jit(integrate_fieldlines, static_argnames=["T", "N", "seq"]) 
get_periodic_intersections_jit = jax.jit(get_periodic_intersections, static_argnames=["max_intersections"])

r_vals = jnp.linspace(axis_margin, 0.99, n_traj)
x0s = jnp.stack(
    [r_vals, 0.5 * jnp.ones_like(r_vals), 0.5 * jnp.ones_like(r_vals)], axis=1
)
# Reshape to (n_scan, n_vmap, 3)
x0_grid = x0s.reshape((n_scan, n_vmap, 3))

plot_data = []
for trace in tqdm.tqdm(iter_traces(trace_file)):
    B_dof = trace["B_dof"]
    p_dof = trace["p_dof"]
    iteration = trace["iteration"]
    force_norm = trace["force_norm"]
    print(f"Iteration {iteration}, force norm: {force_norm}")
    
    logical_trajectories, p_values = integrate_fieldlines_jit(x0_grid, B_dof, p_dof, seq, T, N)

    logical_trajectories = logical_trajectories.reshape((n_traj, N, 3))
    p_values = p_values.reshape((n_traj, N))

    iotas, flags, ks = jax.vmap(lambda c: get_iota_log(c, nfp, ks_thresh=ks_thresh))(
        logical_trajectories
    )
    
    logical_intersections, p_at_intersections, counts = jax.vmap(lambda t, p: get_periodic_intersections_jit(
        t,
        plane_normal=jnp.array([0.0, 0.0, 1.0]),
        plane_point=jnp.array([0.0, 0.0, 0.33]),
        p_values=p,
        max_intersections=int(T // 6)
    ))(logical_trajectories, p_values)

    physical_intersections = jax.vmap(map)(logical_intersections.reshape(-1, 3)).reshape(logical_intersections.shape)
    
    plot_data.append({
        "iteration": iteration,
        "force_norm": force_norm,
        "physical_intersections": physical_intersections,
        "logical_intersections": logical_intersections,
        "p_at_intersections": p_at_intersections,
        "iotas": iotas,
        "ks": ks,
        "counts": counts
    })
# %%
p_min = 0.0
p_max = jnp.nanmax(jnp.array([d["p_at_intersections"] for d in plot_data]))
iota_min = jnp.nanmin(jnp.array([d["iotas"] for d in plot_data]))
iota_max = jnp.nanmax(jnp.array([d["iotas"] for d in plot_data]))

# %%
figs = []
for i in range(len(plot_data)):
    phys_intersec = plot_data[i]["physical_intersections"]
    log_intersec = plot_data[i]["logical_intersections"]
    iotas = plot_data[i]["iotas"]
    p_at_intersections = plot_data[i]["p_at_intersections"]
    
    # Convert physical_intersections to cylindrical coordinates for plotting
    R_vals = (phys_intersec[:, :, 0] ** 2 + phys_intersec[:, :, 1] ** 2) ** 0.5
    phi_vals = jnp.arctan2(phys_intersec[:, :, 1], phys_intersec[:, :, 0])
    z_vals = phys_intersec[:, :, 2]
    cyl_intersections = jnp.stack([R_vals, phi_vals, z_vals], axis=-1)
    iota_values = jnp.broadcast_to(iotas[:, None], phys_intersec[:, :, 0].shape)
    fig, axes = poincare_plot(log_intersec,
                  cyl_intersections,
                  p_at_intersections,
                  iota_values,
                  nfp,
                  Rlim=(jnp.nanmin(R_vals), jnp.nanmax(R_vals)),
                  zlim=(jnp.nanmin(z_vals), jnp.nanmax(z_vals)),
                  p_lim = (p_min, p_max),
                  iota_lim = (iota_min, iota_max)
    )
    figs.append(fig)
    plt.savefig(f"{outdir}/poincare_plot_{plot_data[i]['iteration']}.pdf", dpi=dpi)
    plt.show()
# %%

in_data = {}
# read results file
results_file = "data/desc_heliotron_8_8_8.h5"
with h5py.File(results_file, "r") as f:
    for key in f.keys():
        in_data[key] = np.array(f[key])
# %%
in_data["B"].shape # (N, 3)
in_data["R"].shape # (N,)
in_data["Z"].shape # (N,)
in_data["eval_points"].shape # (N, 3)
# %%
# Load GVEC file and bring it to the same format
gvec_data = {}
# read results file
results_file = "data/gvec_stellarator.h5"
with h5py.File(results_file, "r") as f:
    for key in f.keys():
        gvec_data[key] = np.array(f[key])
# %%
nfs_file = "data/gvec_rotating_ellipse.h5"
print(f"Loading NFS data from {nfs_file}...")
nfs_data = {}
# read results file
with h5py.File(nfs_file, "r") as f:
    pts =    jnp.array(f["eval_points"])
    R =      jnp.array(f["R"])
    Z =      jnp.array(f["Z"])
    B_vals = jnp.array(f["B"])

# %%
nfs_data.keys()
# %%

# %%
