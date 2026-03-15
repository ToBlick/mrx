# %%
"""
Relaxation of the Hegna nfp=3 stellarator equilibrium.

Loads the GVEC equilibrium data from an HDF5 file (80³ regular grid in
Clebsch coordinates), interpolates the coordinate map R(ρ,θ,ζ) and
Z(ρ,θ,ζ) onto B-splines, builds a de Rham sequence on the resulting
stellarator geometry, L²-projects the sampled B-field via
``project_sampled_field``, and runs the MRX relaxation loop.

Usage
-----
    python scripts/relax_hegna.py
"""
import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.io import project_sampled_field
from mrx.mappings import interpolate_map, toroid_map
from mrx.relaxation import (DescentMethod, IntegrationScheme, TimeStepChoice,
                            TimeStepper, relaxation_loop)

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
DATA_FILE = "data/gvec_nfp3_hegna_80cubed_clebsch.h5"
NFP = 3

# FEM resolution
NS = (6, 6, 6)
PS = (2, 2, 2)
QUAD_ORDER = 4

# Relaxation
NUM_ITERS_OUTER = 10
NUM_ITERS_INNER = 10
DT0 = 1.0
FORCE_TOLERANCE = 1e-9

# %%
# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
print("Loading data...")
with h5py.File(DATA_FILE, "r") as f:
    pts = jnp.array(f["eval_points"])   # (512000, 3) — (ρ, θ, ζ) ∈ [0,1]³
    R_vals = jnp.array(f["R"])          # (512000,)
    Z_vals = jnp.array(f["Z"])          # (512000,)
    B_vals = jnp.array(f["B"])          # (512000, 3)

# Extract 1-D grid axes (80 per direction)
rho = jnp.array(np.unique(np.asarray(pts[:, 0])))
theta = jnp.array(np.unique(np.asarray(pts[:, 1])))
zeta = jnp.array(np.unique(np.asarray(pts[:, 2])))
n_rho, n_theta, n_zeta = len(rho), len(theta), len(zeta)
print(f"Grid: {n_rho} × {n_theta} × {n_zeta}")

# ---------------------------------------------------------------
# Interpolate the coordinate map
# ---------------------------------------------------------------
print("Building map interpolation sequence...")
map_seq = DeRhamSequence(
    NS, PS, QUAD_ORDER,
    ("clamped", "periodic", "periodic"),
    lambda x: x, polar=True,
)
map_seq.evaluate_1d()
map_seq.assemble_mass_matrix(0)

print("Interpolating coordinate map (r, θ, ζ) -> (x, y, z)...")
t0 = time.time()
R_grid = R_vals.reshape(n_rho, n_theta, n_zeta)
Z_grid = Z_vals.reshape(n_rho, n_theta, n_zeta)
map_func = interpolate_map((rho, theta, zeta), R_grid, Z_grid, NFP, map_seq)
map_func = jax.jit(map_func)
# Warm-up JIT
_ = map_func(jnp.array([0.5, 0.5, 0.5]))
print(f"Map interpolation done in {time.time() - t0:.1f}s")

# %%
# ---------------------------------------------------------------
# Build the FEM de Rham sequence
# ---------------------------------------------------------------
print(f"Building FEM sequence (ns={NS}, ps={PS})...")
t0 = time.time()
seq = DeRhamSequence(
    NS, PS, QUAD_ORDER,
    ("clamped", "periodic", "periodic"),
    map_func, polar=True,
)

assert jnp.min(seq.jacobian_j) > 0, "Negative Jacobian detected!"
print(f"Jacobian range: [{float(jnp.min(seq.jacobian_j)):.2e}, "
      f"{float(jnp.max(seq.jacobian_j)):.2e}]")

seq.evaluate_1d()
seq.assemble_all_sparse()
seq.assemble_leray_projection()
print(f"FEM setup done in {time.time() - t0:.1f}s")

# %%
# ---------------------------------------------------------------
# Project B-field onto 2-form basis
# ---------------------------------------------------------------
print("Projecting B-field via project_sampled_field (k=2)...")
t0 = time.time()

# Reshape B to (n_rho, n_theta, n_zeta, 3) grid — data is already
# ordered by (ρ, θ, ζ) with ζ varying fastest, then θ, then ρ.
B_grid = B_vals.reshape(n_rho, n_theta, n_zeta, 3)

B_dof_0 = project_sampled_field(
    axes=(rho, theta, zeta),
    values=B_grid,
    seq=seq,
    k=2,
    dirichlet=True,
)
print(f"B projection done in {time.time() - t0:.1f}s")

# Divergence cleaning and normalisation
div_B = float(seq.l2_norm_sq(seq.apply_strong_div(B_dof_0), 3) ** 0.5)
print(f"div(B) before Leray projection: {div_B:.2e}")

B_dof_0, _ = seq.apply_leray_projection(B_dof_0, k=2)
energy_0 = float(0.5 * seq.l2_norm_sq(B_dof_0, 2))
B_dof_0 = B_dof_0 / seq.l2_norm_sq(B_dof_0, 2) ** 0.5
print(
    f"Initial energy (after normalisation): {float(0.5 * seq.l2_norm_sq(B_dof_0, 2)):.6f}")

# %%
# ---------------------------------------------------------------
# Set up the time stepper and run relaxation
# ---------------------------------------------------------------
print("Setting up time stepper...")
ts = TimeStepper(
    seq=seq,
    descent_method=DescentMethod.GRADIENT,
    dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
    timestep_mode=IntegrationScheme.EXPLICIT,
)

print(
    f"Running relaxation: {NUM_ITERS_OUTER} outer × {NUM_ITERS_INNER} inner...")
t0 = time.time()
final_state, traces = relaxation_loop(
    B_dof_0,
    ts,
    num_iters_outer=NUM_ITERS_OUTER,
    num_iters_inner=NUM_ITERS_INNER,
    dt0=DT0,
    force_tolerance=FORCE_TOLERANCE,
    key=jax.random.PRNGKey(42),
)
relax_time = time.time() - t0

# %%
# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
final_force = float(traces["force_norm"][-1])
final_energy = float(traces["energy"][-1])
final_helicity = float(traces["helicity"][-1])
init_helicity = float(traces["helicity"][0])
helicity_change = abs((final_helicity - init_helicity) /
                      (abs(init_helicity) + 1e-30))

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Final force norm:          {final_force:.2e}")
print(f"Final energy:              {final_energy:.6f}")
print(f"Relative helicity change:  {helicity_change:.2e}")
print(f"Relaxation time:           {relax_time:.1f}s")
print("=" * 60)

# %%
# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

iterations = np.array(traces["iteration"])
force_norm = np.array(traces["force_norm"])
energy = np.array(traces["energy"])
helicity = np.array(traces["helicity"])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Force norm
axes[0].semilogy(iterations, force_norm)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel(r"$\|F\|$")
axes[0].set_title("Force norm")

# Energy change
dE = (energy - energy[0]) / abs(energy[0])
axes[1].plot(iterations, dE)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel(r"$\Delta E / |E_0|$")
axes[1].set_title("Relative energy change")

# Helicity change
dH = (helicity - helicity[0]) / (abs(helicity[0]) + 1e-30)
axes[2].plot(iterations, dH)
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel(r"$\Delta H / |H_0|$")
axes[2].set_title("Relative helicity change")

fig.tight_layout()
plt.savefig("relax_hegna_traces.png", dpi=150)
plt.show()
