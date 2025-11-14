# %%
import os
import time

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.io import parse_args, unique_id
from mrx.mappings import cerfon_map, helical_map, rotating_ellipse_map
from mrx.plotting import generate_solovev_plots
from mrx.relaxation import MRXDiagnostics, State
from mrx.utils import run_relaxation_loop, update_config, DEFAULT_CONFIG
from mrx.utils import default_trace_dict, save_trace_dict_to_hdf5, norm_2

jax.config.update("jax_enable_x64", True)

def main():
    """
    Runs a magnetic relaxation simulation for a Solovev configuration.
    Usage: python solovev.py <parameter_name>=<parameter_value>
    where <parameter_name> is one of the parameters in DEFAULT_CONFIG and <parameter_value> is the value to use.
    
    For example:
    python solovev.py run_name=test_run boundary_type=rotating_ellipse n_r=16 n_theta=16 n_zeta=8 p_r=3 p_theta=3 p_zeta=3
    
    will run a simulation with 16 radial, 16 poloidal, and 8 toroidal splines, 
    with radial and poloidal splines of degree 3 and toroidal splines of degree 3, on the CPU.
    """
    # Get user input
    params = parse_args()
    CONFIG = update_config(params, DEFAULT_CONFIG)
    run(CONFIG)


def run(CONFIG):
    run_name = CONFIG["run_name"]
    if run_name == "":
        run_name = unique_id(8)
    outdir = "script_outputs/solovev/" + run_name + "/"
    os.makedirs(outdir, exist_ok=True)
    
    print("Running simulation " + run_name + "...")
    start_time = time.time()
    # Initialize the trace dictionary
    trace_dict = default_trace_dict.copy()
    trace_dict["start_time"] = start_time

    if CONFIG["boundary_type"] == "tokamak":
        F = cerfon_map(CONFIG["eps"], CONFIG["kappa"], jnp.arcsin(CONFIG["delta"]))
    elif CONFIG["boundary_type"] == "helix":
        F = helical_map(epsilon=CONFIG["eps"], h=CONFIG["h_helix"],
                        nfp=CONFIG["nfp"], kappa=CONFIG["kappa"], alpha=jnp.arcsin(CONFIG["delta"]))
    elif CONFIG["boundary_type"] == "rotating_ellipse":
        F = rotating_ellipse_map(
            eps=CONFIG["eps"], kappa=CONFIG["kappa"], nfp=CONFIG["nfp"])
    else:
        raise ValueError("Unknown boundary type.")

    ns = (CONFIG["n_r"], CONFIG["n_theta"], CONFIG["n_zeta"])
    ps = (CONFIG["p_r"], CONFIG["p_theta"], 0
          if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
    q = max(ps)
    types = ("clamped", "periodic",
             "constant" if CONFIG["n_zeta"] == 1 else "periodic")

    print("Setting up FEM spaces...")
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)
    assert jnp.min(Seq.J_j) > 0, "Mapping is singular!"

    # Assemble the FEM spaces and build the projections
    Seq.evaluate_1d()
    Seq.assemble_all()
    Seq.build_crossproduct_projections()
    Seq.assemble_leray_projection()

    # Initialize initial magnetic field guess
    B_harm = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)[1][:, 0]
    B_hat = B_harm / norm_2(B_harm, Seq)

    # Initialize the state of the simulation
    diagnostics = MRXDiagnostics(Seq, CONFIG["force_free"])
    state = State(B_hat, B_hat, CONFIG["dt"], CONFIG["eta"], Seq.M2, 0, 0, 0, 0)
# %%
    # Perform the magnetic relaxation solve
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
    print("Simulation finished, post-processing...")
    B_hat = state.B_n
    get_pressure = jax.jit(diagnostics.pressure)
    if CONFIG["save_B"]:
        trace_dict["p_fields"] = [get_pressure(B) for B in trace_dict["B_fields"]]
    p_hat = get_pressure(B_hat)
    trace_dict["p_final"] = p_hat
    trace_dict["B_final"] = B_hat

    # Write to HDF5
    print("Saving to hdf5...")
    save_trace_dict_to_hdf5(trace_dict, diagnostics, outdir + run_name, CONFIG)

    print(f"Data saved to {outdir + run_name + '.h5'}.")

    # Plot all traces
    print("Generating plots...")
    generate_solovev_plots(outdir + run_name + ".h5")

# %%
if __name__ == "__main__":
    main()
