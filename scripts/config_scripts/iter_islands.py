# %%
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.mappings import cerfon_map, helical_map, rotating_ellipse_map
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.io import parse_args, unique_id
from mrx.relaxation import MRXDiagnostics, MRXHessian, State, TimeStepper
from mrx.utils import inv33

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/iter/"
os.makedirs(outdir, exist_ok=True)


CONFIG = {
    "run_name": "",  # Name for the run. If empty, a hash will be created

    # Type of configuration: "tokamak" or "helix" or "rotating_ellipse"
    "type": "tokamak",

    ###
    # Parameters describing the domain.
    ###
    "eps":      0.33,  # aspect ratio
    "kappa":    1.7,   # Elongation parameter
    "delta":    0.33,   # triangularity
    "delta_B":   0.2,   # poloidal field strength relative to harmonic one
    "q_star":   1.54,

    ###
    # Discretization
    ###
    "n_r": 8,       # Number of radial splines
    "n_theta": 8,   # Number of poloidal splines
    "n_zeta": 4,    # Number of toroidal splines
    "p_r": 3,       # Degree of radial splines
    "p_theta": 3,     # Degree of poloidal splines
    "p_zeta": 3,    # Degree of toroidal splines

    ###
    # Hyperparameters for the relaxation
    ###
    "maxit":                 5_000,   # max. Number of time steps
    "precond":               False,     # Use preconditioner
    "precond_compute_every": 1000,       # Recompute preconditioner every n iterations
    # Regularization, u = (-Δ)⁻ᵞ (J x B - grad p)
    "gamma":                 0,
    "dt":                    1e-4,      # initial time step
    # time-steps are increased by this factor and decreased by its square
    "dt_factor":             1.01,
    # Convergence tolerance for |JxB - grad p| (or |JxB| if force_free)
    "force_tol":             1e-15,
    "eta":                   0.0,       # Resistivity
    # If True, solve for JxB = 0. If False, JxB = grad p
    "force_free":            False,

    ###
    # Hyperparameters pertaining to island seeding
    ###
    "pert_strength":       2e-5,  # strength of perturbation
    "pert_pol_mode":          2,  # poloidal mode number of perturbation
    "pert_tor_mode":          1,  # toroidal mode number of perturbation
    "pert_radial_loc":      1/2,  # radial location of perturbation
    "pert_radial_width":    0.07,  # radial width of perturbation
    # apply perturbation after n steps (0 = to initial condition)
    "apply_pert_after":     2000,

    ###
    # Solver hyperparameters
    ###
    # Picard solver
    "solver_maxit": 20,    # Maximum number of iterations before Picard solver gives up
    # If Picard solver converges in less than this number of iterations, increase time step
    "solver_critit": 4,
    "solver_tol": 1e-12,   # Tolerance for convergence
    "verbose": False,      # If False, prints only force every 'print_every'
    "print_every": 500,    # Print every n iterations
    "save_every": 10,     # Save intermediate results every n iterations
    "save_B": True,       # Save intermediate B fields to file
    "save_B_every": 500,   # Save full B every n iterations
}


def main():
    # Get user input
    params = parse_args()

    for k, v in params.items():
        if k in CONFIG:
            CONFIG[k] = v
        elif k != "device":
            print(f"Unknown parameter '{k}' - ignoring.")

    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

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

    iterations = [0]
    force_trace = []
    energy_trace = []
    helicity_trace = []
    divergence_trace = []
    picard_iterations = []
    picard_errors = []
    timesteps = []
    velocity_trace = []
    wall_time_trace = []
    if CONFIG["save_B"]:
        B_fields = []

    def append_all(i, f, E, H, dvg, v, p_i, e, dt, B=None):
        iterations.append(i)
        force_trace.append(f)
        energy_trace.append(E)
        helicity_trace.append(H)
        divergence_trace.append(dvg)
        velocity_trace.append(v)
        picard_iterations.append(p_i)
        picard_errors.append(e)
        timesteps.append(dt)
        wall_time_trace.append(time.time() - start_time)
        if CONFIG["save_B"]:
            B_fields.append(B)

    def norm_2(u):
        return (u @ Seq.M2 @ u)**0.5

    def norm_1(u):
        return (u @ Seq.M1 @ u)**0.5

    def B_xyz(p):
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

    # def A_xyz(p):
    #     x, y, z = F(p)
    #     r, θ, ζ = p
    #     # seeded island at r = 1/3 in toroidal direction
    #     DFx = jax.jacfwd(F)(p)
    #     J = jnp.linalg.det(DFx)
    #     A_ζ = jnp.sin(2 * jnp.pi * ζ) * jnp.exp(-100 *
    #                                             (r - 1 / 3)**2) * jnp.sin(6 * jnp.pi * θ)
    #     return inv33(DFx.T)[:, 2] * A_ζ / J

    B_hat = jnp.linalg.solve(Seq.M2, Seq.P2(B_xyz))
    B_hat = Seq.P_Leray @ B_hat
    B_hat /= norm_2(B_hat)

    # B_harm = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)[1][:, 0]
    # B_harm = B_harm / norm_2(B_harm)

    # B_hat = B_harm + CONFIG["delta_B"] * B_hat
    # B_hat /= norm_2(B_hat)

    if CONFIG["apply_pert_after"] == 0 and CONFIG["pert_strength"] > 0:
        print("Applying perturbation to initial condition...")
        dB_hat = jnp.linalg.solve(Seq.M2, Seq.P2(dB_xyz))
        dB_hat = Seq.P_Leray @ dB_hat
        dB_hat /= norm_2(dB_hat)
        B_hat += CONFIG["pert_strength"] * dB_hat

    diagnostics = MRXDiagnostics(Seq, CONFIG["force_free"])

    get_pressure = jax.jit(diagnostics.pressure)
    get_energy = jax.jit(diagnostics.energy)
    get_helicity = jax.jit(diagnostics.helicity)
    get_divergence_B = jax.jit(diagnostics.divergence_norm)

    p_hat = get_pressure(B_hat)

    state = State(B_hat, B_hat, CONFIG["dt"],
                  CONFIG["eta"], Seq.M2, 0, 0, 0, 0)
    # initial Hessian is identity

    timestepper = TimeStepper(Seq,
                              gamma=CONFIG["gamma"],
                              newton=CONFIG["precond"],
                              force_free=CONFIG["force_free"],
                              picard_tol=CONFIG["solver_tol"],
                              picard_maxit=CONFIG["solver_maxit"])

    compute_hessian = jax.jit(MRXHessian(Seq).assemble)
    step = jax.jit(timestepper.picard_solver)

    dry_run = step(state)  # compile

    force_trace.append(dry_run.force_norm)
    velocity_trace.append(dry_run.velocity_norm)
    energy_trace.append(get_energy(state.B_n))
    helicity_trace.append(get_helicity(state.B_n))
    divergence_trace.append(get_divergence_B(state.B_n))
    if CONFIG["save_B"]:
        B_fields.append(B_hat)

    print(f"Initial force error: {force_trace[-1]:.2e}")
    print(f"Initial energy: {energy_trace[-1]:.2e}")
    print(f"Initial helicity: {helicity_trace[-1]:.2e}")
    print(f"Initial ||div B||: {divergence_trace[-1]:.2e}")

    setup_done_time = time.time()
    wall_time_trace.append(setup_done_time - start_time)
    print(f"Setup took {setup_done_time - start_time:.2e} seconds.")
    print("Starting relaxation loop...")
# %%
    for i in range(1, CONFIG["maxit"] + 1):

        state = step(state)
        if (state.picard_residuum > CONFIG["solver_tol"]
                or ~jnp.isfinite(state.picard_residuum)):
            # half time step and try again
            state = timestepper.update_dt(state, state.dt / 2)
            state = timestepper.update_B_guess(state, state.B_n)
            continue
        # otherwise, we converged - proceed
        state = timestepper.update_B_n(state, state.B_guess)

        if i == CONFIG["apply_pert_after"] and CONFIG["pert_strength"] > 0:
            print(f"Applying perturbation after {i} steps...")
            dB_hat = jnp.linalg.solve(Seq.M2, Seq.P2(dB_xyz))
            dB_hat = Seq.P_Leray @ dB_hat
            dB_hat /= norm_2(dB_hat)
            B_new = state.B_n + CONFIG["pert_strength"] * dB_hat
            state = timestepper.update_B_n(state, B_new)

        if CONFIG["precond"] and (i % CONFIG["precond_compute_every"] == 0):
            state = timestepper.update_hessian(
                state, compute_hessian(state.B_n))

        if state.picard_iterations < CONFIG["solver_critit"]:
            dt_new = state.dt * CONFIG["dt_factor"]
        else:
            dt_new = state.dt / (CONFIG["dt_factor"])**2
        state = timestepper.update_dt(state, dt_new)

        if i % CONFIG["save_every"] == 0 or i == CONFIG["maxit"]:
            append_all(i,
                       state.force_norm,
                       get_energy(state.B_n),
                       get_helicity(state.B_n),
                       get_divergence_B(state.B_n),
                       state.velocity_norm,
                       state.picard_iterations,
                       state.picard_residuum,
                       state.dt,
                       state.B_n if (CONFIG["save_B"] and i % CONFIG["save_B_every"] == 0) else None)

        if i % CONFIG["print_every"] == 0:
            print(
                f"Iteration {i}, u norm: {state.velocity_norm:.2e}, force norm: {state.force_norm:.2e}")
            if CONFIG["verbose"]:
                print(
                    f"   dt: {dt_new:.2e}, picard iters: {state.picard_iterations:.2e}, picard err: {state.picard_residuum:.2e}")
        if force_trace[-1] < CONFIG["force_tol"]:
            print(
                f"Converged to force tolerance {CONFIG['force_tol']} after {i} steps.")
            break
# %%
    final_time = time.time()
    print(
        f"Main loop took {final_time - setup_done_time:.2e} seconds for {i} steps, avg. {(final_time - setup_done_time)/i:.2e} s/step.")

    B_fields = [x for x in B_fields if x is not None]
    ###
    # Post-processing
    ###
    B_hat = state.B_n
    print("Simulation finished, post-processing...")
    if CONFIG["save_B"]:
        p_fields = [get_pressure(
            B) if B is not None else None for B in B_fields]
    p_hat = get_pressure(B_hat)

    ###
    # Save stuff
    ###
    print("Saving to hdf5...")

    # Write to HDF5
    with h5py.File(outdir + run_name + ".h5", "w") as f:
        # Store arrays
        f.create_dataset("iterations", data=jnp.array(iterations))
        f.create_dataset("force_trace", data=jnp.array(force_trace))
        f.create_dataset("B_final", data=B_hat)
        f.create_dataset("p_final", data=p_hat)
        f.create_dataset("energy_trace", data=jnp.array(energy_trace))
        f.create_dataset("helicity_trace", data=jnp.array(helicity_trace))
        f.create_dataset("divergence_B_trace",
                         data=jnp.array(divergence_trace))
        f.create_dataset("velocity_trace", data=jnp.array(velocity_trace))
        f.create_dataset("picard_iterations",
                         data=jnp.array(picard_iterations))
        f.create_dataset("picard_errors", data=jnp.array(picard_errors))
        f.create_dataset("timesteps", data=jnp.array(timesteps))
        f.create_dataset("harmonic_norm", data=jnp.array(
            [norm_2(diagnostics.harmonic_component(B_hat))]))
        f.create_dataset("total_time", data=jnp.array(
            [final_time - start_time]))
        f.create_dataset("time_setup", data=jnp.array(
            [setup_done_time - start_time]))
        f.create_dataset("time_solve", data=jnp.array(
            [final_time - setup_done_time]))
        f.create_dataset("wall_time_trace", data=jnp.array(
            wall_time_trace) - start_time)
        if CONFIG["save_B"]:
            f.create_dataset("B_fields", data=jnp.array(B_fields))
            f.create_dataset("p_fields", data=jnp.array(p_fields))
        # Store config variables in a group
        cfg_group = f.create_group("config")
        for key, val in CONFIG.items():
            if isinstance(val, str):
                # Strings need special handling
                cfg_group.attrs[key] = np.bytes_(val)
            else:
                cfg_group.attrs[key] = val

    print(f"Data saved to {outdir + run_name + '.h5'}.")

if __name__ == "__main__":
    main()
# %%
