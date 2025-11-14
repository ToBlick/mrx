# %%
import os
import time

import jax
import jax.numpy as jnp

from mrx.mappings import rotating_ellipse_map
from mrx.derham_sequence import DeRhamSequence
from mrx.io import parse_args, unique_id
from mrx.plotting import trace_plot
from mrx.relaxation import MRXDiagnostics, State
from mrx.utils import update_config, default_trace_dict, save_trace_dict_to_hdf5, norm_2, run_relaxation_loop, DEFAULT_CONFIG

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/stell/"
os.makedirs(outdir, exist_ok=True)

def main():
    # Get user input
    params = parse_args()
    CONFIG = DEFAULT_CONFIG.copy()
    # Specific configuration parameters for the simulation that are not the default configuration
    # NOTE: these can be overridden by the user-supplied parameters from the command line.
    CONFIG["boundary_type"] = "rotating_ellipse"
    CONFIG["eps"] = 0.33
    CONFIG["kappa"] = 1.1
    CONFIG["q_star"] = 1.54
    CONFIG["n_fp"] = 3
    CONFIG["n_r"] = 8
    CONFIG["n_theta"] = 8
    CONFIG["n_zeta"] = 4
    CONFIG["dt"] = 1e-4
    CONFIG = update_config(params, CONFIG)
    run(CONFIG)


def run(CONFIG):
    run_name = CONFIG["run_name"]
    if run_name == "":
        run_name = unique_id(8)

    print("Running simulation " + run_name + "...")

    start_time = time.time()
    # Initialize the trace dictionary
    trace_dict = default_trace_dict.copy()
    trace_dict["start_time"] = start_time

    F = rotating_ellipse_map(CONFIG["eps"], CONFIG["kappa"], CONFIG["n_fp"])

    ns = (CONFIG["n_r"], CONFIG["n_theta"], CONFIG["n_zeta"])
    ps = (CONFIG["p_r"], CONFIG["p_theta"], 0
          if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
    q = max(ps)
    types = ("clamped", "periodic",
             "constant" if CONFIG["n_zeta"] == 1 else "periodic")
    tau = CONFIG["q_star"]

    print("Setting up FEM spaces...")
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)
    assert jnp.min(Seq.J_j) > 0, "Mapping is singular!"

    # Assemble the FEM spaces and build the projections
    Seq.evaluate_1d()
    Seq.assemble_all()
    Seq.build_crossproduct_projections()
    Seq.assemble_leray_projection()    

    # Define the initial magnetic field in physical space
    def B_xyz(p):
        x, y, z = F(p)
        R = (x**2 + y**2)**0.5
        phi = jnp.arctan2(y, x)
        BR = z * R
        Bphi = tau / R
        Bz = - (1 / 2 * (R**2 - 1**2) + z**2)
        Bx = BR * jnp.cos(phi) - Bphi * jnp.sin(phi)
        By = BR * jnp.sin(phi) + Bphi * jnp.cos(phi)
        return jnp.array([Bx, By, Bz])

    # Define the perturbation to the initial magnetic field in physical space
    def dB_xyz(p):
        r, θ, ζ = p
        DFx = jax.jacfwd(F)(p)

        def a(r):
            return jnp.exp(- (r - CONFIG["pert_radial_loc"])**2 / (2 * CONFIG["pert_radial_width"]**2))
        B_rad = a(r) * jnp.sin(2 * jnp.pi * θ * CONFIG["pert_pol_mode"]) * \
            jnp.sin(2 * jnp.pi * ζ * CONFIG["pert_tor_mode"]) * DFx[:, 0]
        return B_rad

    # Solve for the magnetic field in the FEM space
    B_hat = jnp.linalg.solve(Seq.M2, Seq.P2(B_xyz))
    B_hat = Seq.P_Leray @ B_hat
    B_hat /= norm_2(B_hat, Seq)

    # Apply a perturbation to the initial magnetic field if specified
    if CONFIG["apply_pert_after"] == 0 and CONFIG["pert_strength"] > 0:
        print("Applying perturbation to initial condition...")
        dB_hat = jnp.linalg.solve(Seq.M2, Seq.P2(dB_xyz))
        dB_hat = Seq.P_Leray @ dB_hat
        dB_hat /= norm_2(dB_hat, Seq)
        B_hat += CONFIG["pert_strength"] * dB_hat

    # Initialize the state and diagnostics of the simulation
    diagnostics = MRXDiagnostics(Seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], Seq.M2, 0, 0, 0, 0)

# %%
    # Perform the magnetic relaxation solve
    CONFIG["dB_xyz"] = dB_xyz  # Needed for the perturbation to be applied
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)
# %%
    final_time = time.time()
    trace_dict["end_time"] = final_time
    print(
        f"Main loop took {final_time - trace_dict["setup_done_time"]:.2e} ",
        f"seconds for {trace_dict["iterations"][-1]} steps, avg.", 
        f"{(final_time - trace_dict["setup_done_time"])/trace_dict["iterations"][-1]:.2e} s/step."
    )

    # Post-processing
    B_hat = state.B_n
    trace_dict["B_final"] = B_hat
    print("Simulation finished, post-processing...")
    get_pressure = jax.jit(diagnostics.pressure)
    if CONFIG["save_B"]:
        trace_dict["p_fields"] = [get_pressure(B) if B is not None else None for B in trace_dict["B_fields"]]
    p_hat = get_pressure(B_hat)
    trace_dict["p_final"] = p_hat

    print("Saving to hdf5...")
    save_trace_dict_to_hdf5(trace_dict, diagnostics, outdir + run_name, CONFIG)
    print(f"Data saved to {outdir + run_name + '.h5'}")

    # Plot all traces
    print("Generating plots...")
    trace_plot(trace_dict, filename=outdir + "force_trace.pdf")

if __name__ == "__main__":
    main()
# %%
