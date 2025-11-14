# %%
import os
import time

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.io import parse_args, unique_id
from mrx.mappings import cerfon_map
from mrx.relaxation import MRXDiagnostics, State
from mrx.utils import update_config, DEFAULT_CONFIG, run_relaxation_loop
from mrx.utils import default_trace_dict, save_trace_dict_to_hdf5, norm_2
from mrx.plotting import trace_plot

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/iter/"
os.makedirs(outdir, exist_ok=True)


def main():
    # Get user input
    params = parse_args()
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["boundary_type"] = "tokamak"
    CONFIG["eps"] = 0.33
    CONFIG["kappa"] = 1.7
    CONFIG["delta"] = 0.33
    CONFIG["delta_B"] = 0.2
    CONFIG["q_star"] = 1.54
    CONFIG["dt"] = 1e-4
    CONFIG["pert_strength"] = 2e-5
    CONFIG["save_every"] = 10
    CONFIG = update_config(params, CONFIG)
    run(CONFIG)


def run(CONFIG):
    run_name = CONFIG["run_name"]
    if run_name == "":
        run_name = unique_id(8)

    print("Running simulation " + run_name + "...")

    kappa = CONFIG["kappa"]
    eps = CONFIG["eps"]
    alpha = jnp.arcsin(CONFIG["delta"])

    start_time = time.time()

    F = cerfon_map(eps, kappa, alpha)

    ns = (CONFIG["n_r"], CONFIG["n_theta"], CONFIG["n_zeta"])
    ps = (CONFIG["p_r"], CONFIG["p_theta"], 0
          if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
    q = max(ps)
    types = ("clamped", "periodic",
             "constant" if CONFIG["n_zeta"] == 1 else "periodic")
    tau = CONFIG["q_star"] * kappa * (1 + kappa**2) / (kappa + 1)
    print("Setting up FEM spaces...")

    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)

    assert jnp.min(Seq.J_j) > 0, "Mapping is singular!"

    Seq.evaluate_1d()
    Seq.assemble_all()
    Seq.build_crossproduct_projections()
    Seq.assemble_leray_projection()

    trace_dict = default_trace_dict.copy()
    trace_dict["start_time"] = start_time

    def B_0(p):
        x, y, z = F(p)
        R = (x**2 + y**2)**0.5
        phi = jnp.arctan2(y, x)
        BR = z * R
        Bphi = tau / R
        Bz = - (kappa**2 / 2 * (R**2 - 1**2) + z**2)
        Bx = BR * jnp.cos(phi) - Bphi * jnp.sin(phi)
        By = BR * jnp.sin(phi) + Bphi * jnp.cos(phi)
        return jnp.array([Bx, By, Bz])

    def dB_xyz(p):
        r, θ, ζ = p
        DFx = jax.jacfwd(F)(p)

        def a(r):
            return jnp.exp(- (r - CONFIG["pert_radial_loc"])**2 / (2 * CONFIG["pert_radial_width"]**2))
        B_rad = a(r) * jnp.sin(2 * jnp.pi * θ * CONFIG["pert_pol_mode"]) * \
            jnp.sin(2 * jnp.pi * ζ * CONFIG["pert_tor_mode"]) * DFx[:, 0]
        return B_rad

    # Initialize the initial magnetic field in the FEM space
    B_dof = jnp.linalg.solve(Seq.M2, Seq.P2(B_0))
    B_dof = Seq.P_Leray @ B_dof
    B_dof /= norm_2(B_dof, Seq)

    # Apply a perturbation to the initial magnetic field if specified
    if CONFIG["apply_pert_after"] == 0 and CONFIG["pert_strength"] > 0:
        print("Applying perturbation to initial condition...")
        dB_dof = jnp.linalg.solve(Seq.M2, Seq.P2(dB_xyz))
        dB_dof = Seq.P_Leray @ dB_dof
        dB_dof /= norm_2(dB_dof, Seq)
        B_dof += CONFIG["pert_strength"] * dB_dof

    diagnostics = MRXDiagnostics(Seq, CONFIG["force_free"])
    state = State(B_dof, B_dof, CONFIG["dt"],
                  CONFIG["eta"], Seq.M2, 0, 0, 0, 0)

# %%
    # Perform the magnetic relaxation solve
    CONFIG["dB_xyz"] = dB_xyz  # Needed for the perturbation to be applied
    run_relaxation_loop(CONFIG, trace_dict, state, diagnostics)

# %%
    final_time = time.time()
    trace_dict["end_time"] = final_time
    print(
        f"Main loop took {final_time - trace_dict['setup_done_time']:.2e} ",
        f"seconds for {trace_dict['iterations'][-1]} steps, avg.", 
        f"{(final_time - trace_dict['setup_done_time'])/trace_dict['iterations'][-1]:.2e} s/step.")

    # Post-processing
    B_dof = state.B_n
    get_pressure = jax.jit(diagnostics.pressure)
    print("Simulation finished, post-processing...")
    if CONFIG["save_B"]:
        trace_dict["p_fields"] = [get_pressure(B) if B is not None else None for B in trace_dict["B_fields"]]
    p_dof = get_pressure(B_dof)
    trace_dict["B_final"] = B_dof
    trace_dict["p_final"] = p_dof

    print("Saving to hdf5...")
    save_trace_dict_to_hdf5(trace_dict, diagnostics, outdir + run_name, CONFIG)
    print(f"Data saved to {outdir + run_name + '.h5'}.")

    # Plot all traces
    print("Generating plots...")
    trace_plot(trace_dict, filename=outdir + "force_trace.pdf")

if __name__ == "__main__":
    main()
# %%
