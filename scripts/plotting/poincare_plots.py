# %%
"""
Interactive script to load and visualize results from relaxation simulations.

This script can handle both single runs and multirun (sweep) outputs from Hydra.
It loads the HDF5 results files and uses the trace_plot function from mrx.plotting.
"""
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
import yaml
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
import tqdm

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import stellarator_map
from mrx.plotting import get_iota_log

import matplotlib.pyplot as plt



# %%
# Configuration - set your output directory here
# For a single run:
# base_dir = Path("out/gvec_relaxation/20260203_120000")
# For a multirun:
# base_dir = Path("out/gvec_relaxation/multirun/20260203_120000")

T = 500
n_traj = 8
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
@partial(jax.jit, static_argnames=["T", "n_scan", "n_vmap", "N"])
def integrate_fieldlines_hybrid(B_dof, p_dof, n_scan, n_vmap, T, N):
    B_h = DiscreteFunction(B_dof, seq.Lambda_2, seq.E2)
    p_h = DiscreteFunction(p_dof, seq.Lambda_0, seq.E0)
    
    def vector_field(t, x, args):
        x %= 1.0
        Bx = B_h(x)
        DFx = jax.jacfwd(seq.F)(x)
        return Bx / jnp.linalg.norm(DFx @ Bx)

    def integrate_fieldline(x0):
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(vector_field),
            solver=dfx.Dopri8(),
            t0=0.0,
            t1=T,
            dt0=1.0,
            y0=x0,
            saveat=dfx.SaveAt(ts=jnp.linspace(0.0, T, N)),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=100_000,
        )
        # Apply periodicity to the final trajectory
        traj = sol.ys % 1.0
        # Evaluate scalar field along the trajectory
        p_vals = jax.vmap(p_h)(traj)
        return traj, p_vals

    # 1. Prepare the initial conditions grid
    n_traj = n_scan * n_vmap
    r_vals = jnp.linspace(axis_margin, 0.99, n_traj)
    x0s = jnp.stack(
        [r_vals, 0.5 * jnp.ones_like(r_vals), 0.5 * jnp.ones_like(r_vals)], axis=1
    )
    
    # Reshape to (n_scan, n_vmap, 3)
    x0_grid = x0s.reshape((n_scan, n_vmap, 3))

    # 2. Define the vectorized inner function
    # This will process 'n_vmap' trajectories in parallel on the GPU
    vmapped_integrate = jax.vmap(integrate_fieldline)

    # 3. Define the scan function
    # This will loop over 'n_scan' batches sequentially to save memory
    def scan_fn(carry, x0_batch):
        trajs, ps = vmapped_integrate(x0_batch)
        return carry, (trajs, ps)
    
    _, (logical_trajectories, p_values) = jax.lax.scan(scan_fn, None, x0_grid)
    
    # 4. Reshape back to flat trajectory lists: (n_traj, N, 3) and (n_traj, N)
    logical_trajectories = logical_trajectories.reshape((n_traj, N, 3))
    p_values = p_values.reshape((n_traj, N))
    
    return logical_trajectories, p_values


# %%   
@partial(jax.jit, static_argnames=["max_intersections"])
def get_periodic_intersections_jit(
    field_line, 
    p_values, 
    plane_normal, 
    plane_point, 
    max_intersections=100
):
    # --- Helpers: Padding and Masking ---
    def pad_axis0(arr):
        pads = ((1, 1),) + ((0, 0),) * (arr.ndim - 1)
        return jnp.pad(arr, pads, mode='edge')

    def apply_mask(data, mask):
        mask_reshaped = mask.reshape(mask.shape + (1,) * (data.ndim - 1))
        return jnp.where(mask_reshaped, data, jnp.nan)

    # 1. Periodic-Aware Distance Calculation
    # Calculate current distances
    diffs = field_line - plane_point
    dists = jnp.dot(diffs, plane_normal)
    
    # Calculate "Virtual" Next Distances to handle periodic jumps
    pos_curr = field_line[:-1]
    pos_next_raw = field_line[1:]
    
    # Unwrap the step: find shortest vector in periodic [0, 1] space
    delta = pos_next_raw - pos_curr
    delta_unwrapped = delta - jnp.round(delta)
    pos_next_virt_detect = pos_curr + delta_unwrapped
    
    # Distance at the "virtual" next point
    dist_next_virt = jnp.dot(pos_next_virt_detect - plane_point, plane_normal)
    
    # 2. Detect Crossings
    # A crossing exists if the line segment (curr -> next_virt) crosses zero
    d_curr_detect = dists[:-1]
    valid_crossing = (d_curr_detect * dist_next_virt) < 0.0
    
    # 3. Prepare for Interpolation
    # Pad arrays to allow (idx-1, idx, idx+1) access in the kernel
    d_pad = pad_axis0(dists)
    pos_pad = pad_axis0(field_line)
    p_pad = pad_axis0(p_values)
    
    crossing_indices = jnp.nonzero(
        valid_crossing, 
        size=max_intersections, 
        fill_value=0
    )[0]
    
    count = jnp.sum(valid_crossing)
    # +1 because we padded the start of the array
    centers = crossing_indices + 1 

    # 4. Interpolation Kernel (Quadratic)
    def interpolate_single(idx):
        pos_prev_raw = pos_pad[idx - 1]
        pos_curr     = pos_pad[idx]
        pos_next_raw = pos_pad[idx + 1]
        
        # Unwrap Geometry for the local neighborhood
        # This ensures the quadratic fit sees a continuous line
        delta_prev = pos_prev_raw - pos_curr
        pos_prev_v = pos_curr + (delta_prev - jnp.round(delta_prev))
        
        delta_next = pos_next_raw - pos_curr
        pos_next_v = pos_curr + (delta_next - jnp.round(delta_next))
        
        # Recalculate distances on unwrapped local points
        d_prev = jnp.dot(pos_prev_v - plane_point, plane_normal)
        d_curr = jnp.dot(pos_curr - plane_point, plane_normal)
        d_next = jnp.dot(pos_next_v - plane_point, plane_normal)
        
        # Quadratic coefficients: f(t) = at^2 + bt + c, where t=0 is pos_curr
        c = d_curr
        a = 0.5 * (d_prev - 2.0 * d_curr + d_next)
        b = 0.5 * (d_next - d_prev)
        
        # Solve at^2 + bt + c = 0
        discriminant = b**2 - 4*a*c
        sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        
        div = 2.0 * a + 1e-15
        t1 = (-b + sqrt_disc) / div
        t2 = (-b - sqrt_disc) / div
        t_lin = -c / (b + 1e-15)
        
        # Choose the root that lies within the segment range [-0.5, 0.5] 
        # (or 0 to 1 depending on your specific center definition)
        # Here we use the logic consistent with your original range:
        t_quad = jnp.where((t1 >= 0.0) & (t1 <= 1.0), t1, t2)
        t = jnp.where(jnp.abs(a) > 1e-10, t_quad, t_lin)
        t = jnp.clip(t, 0.0, 1.0)
        
        # Interpolate Position
        A_pos = 0.5 * (pos_prev_v - 2.0 * pos_curr + pos_next_v)
        B_pos = 0.5 * (pos_next_v - pos_prev_v)
        final_pos = A_pos * (t**2) + B_pos * t + pos_curr
        
        # Re-apply periodicity to the result
        final_pos = final_pos - jnp.floor(final_pos)
        
        # Interpolate Scalar Field p
        v0, v1, v2 = p_pad[idx-1], p_pad[idx], p_pad[idx+1]
        Ap = 0.5 * (v0 - 2.0 * v1 + v2)
        Bp = 0.5 * (v2 - v0)
        final_p = Ap * (t**2) + Bp * t + v1

        return final_pos, final_p

    # Vectorize and Mask
    intersections, inter_p = jax.vmap(interpolate_single)(centers)
    
    _idx = jnp.arange(max_intersections)
    mask = _idx < count
    
    intersections = apply_mask(intersections, mask)
    inter_p = apply_mask(inter_p, mask)
    
    return intersections, inter_p, count


# %%
n_scan = 2
n_vmap = 32
T = 1000 * jnp.pi * 2 * 3
N = int(T)

plot_data = []
for trace in tqdm.tqdm(iter_traces(trace_file)):
    B_dof = trace["B_dof"]
    p_dof = trace["p_dof"]
    iteration = trace["iteration"]
    force_norm = trace["force_norm"]
    print(f"Iteration {iteration}, force norm: {force_norm}")
    
    logical_trajectories, p_values = integrate_fieldlines_hybrid(B_dof, p_dof, n_scan, n_vmap, T, N)

    iotas, flags, ks = jax.vmap(lambda c: get_iota_log(c, nfp, ks_thresh=ks_thresh))(
        logical_trajectories
    )
    
    logical_intersections, p_at_intersections, counts = jax.vmap(lambda t, p: get_periodic_intersections_jit(
        t,
        plane_normal=jnp.array([0.0, 0.0, 1.0]),
        plane_point=jnp.array([0.0, 0.0, 0.5]),
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
for i in range(len(plot_data)):
    plt.figure(figsize=(6, 4), dpi=dpi)
    iotas_expanded = jnp.broadcast_to(plot_data[i]["iotas"][:, None], plot_data[i]["physical_intersections"][:, :, 0].shape)
    plt.scatter(
        plot_data[i]["physical_intersections"][:, :, 0].flatten(),
        plot_data[i]["physical_intersections"][:, :, 2].flatten(),
        s=markersize,
        # c=plot_data[i]["p_at_intersections"].flatten(), 
        c=iotas_expanded,
        cmap="nipy_spectral",
    )
    plt.colorbar()
    plt.xlabel("R")
    plt.ylabel("z")
    plt.show()
# %%
