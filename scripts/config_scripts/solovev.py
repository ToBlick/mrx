# %%
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.mappings import cerfon_map, helical_map, rotating_ellipse_map
from mrx.derham_sequence import DeRhamSequence
from mrx.io import parse_args, unique_id
from mrx.relaxation import MRXDiagnostics, MRXHessian, State, TimeStepper

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/solovev/"
os.makedirs(outdir, exist_ok=True)


###
# Default configuration
###
DEVICE_PRESETS = {
    "ITER":  {"eps": 0.32, "kappa": 1.7, "delta": 0.33, "q_star": 1.57, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "NSTX":  {"eps": 0.78, "kappa": 2.0, "delta": 0.35, "q_star": 2.0, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "SPHERO": {"eps": 0.95, "kappa": 1.0, "delta": 0.2,  "q_star": 0.0, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "ROT_ELL": {"a": 0.1, "b": 0.025, "m_rot": 5, "q_star": 1.6, "type": "rotating_ellipse"},
    "HELIX": {"eps": 0.33, "h_helix": 0.20, "kappa": 1.7, "delta": 0.33, "m_helix": 3, "q_star": 2.0, "type": "helix"},
}

CONFIG = {
    "run_name": "",  # Name for the run. If empty, a hash will be created

    # Type of configuration: "tokamak" or "helix" or "rotating_ellipse"
    "type": "helix",

    ###
    # Parameters describing the domain.
    ###
    "eps":      0.2,  # aspect ratio
    "kappa":    1.7,   # Elongation parameter
    "q_star":   1.57,   # toroidal field strength
    "delta": 0.0,   # triangularity
    "m_helix":  3,     # poloidal mode number of helix
    "h_helix":  0,   # radius of helix turns
    "m_rot":    5,      # mode number of rotating ellipse
    "a": 0.1,      # amplitude of rotating ellipse deformation
    "b": 0.025,    # amplitude of rotating ellipse deformation

    ###
    # Discretization
    ###
    "n_r": 8,       # Number of radial splines
    "n_theta": 8,   # Number of poloidal splines
    "n_zeta": 6,    # Number of toroidal splines
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
    "dt":                    1e-6,      # initial time step
    # time-steps are increased by this factor and decreased by its square
    "dt_factor":             1.01,
    # Convergence tolerance for |JxB - grad p| (or |JxB| if force_free)
    "force_tol":             1e-15,
    "eta":                   0.0,       # Resistivity
    # If True, solve for JxB = 0. If False, JxB = grad p
    "force_free":            False,

    ###
    # Solver hyperparameters
    ###
    # Picard solver
    "solver_maxit": 20,    # Maximum number of iterations before Picard solver gives up
    # If Picard solver converges in less than this number of iterations, increase time step
    "solver_critit": 4,
    "solver_tol": 1e-12,   # Tolerance for convergence
    "verbose": False,      # If False, prints only force every 'print_every'
    "print_every": 1000,    # Print every n iterations
    "save_every": 100,     # Save intermediate results every n iterations
    "save_B": False,       # Save intermediate B fields to file
}


def main():
    # Get user input
    params = parse_args()

    # Step 1: If device specified, apply defaults
    device_name = params.get("device")
    if device_name:
        preset = DEVICE_PRESETS.get(device_name.upper())
        if preset:
            for k, v in preset.items():
                CONFIG[k] = v
        else:
            print(f"Unknown device '{device_name}' - ignoring.")

    # Step 2: Override with user-supplied parameters
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
    if CONFIG["type"] == "tokamak":
        tau = CONFIG["q_star"] * kappa * (1 + kappa**2) / (kappa + 1)
    else:
        tau = CONFIG["q_star"]

    start_time = time.time()

    if CONFIG["type"] == "tokamak":
        F = cerfon_map(eps, kappa, alpha)
    elif CONFIG["type"] == "helix":
        F = helical_map(epsilon=CONFIG["eps"], h=CONFIG["h_helix"],
                        n_turns=CONFIG["m_helix"], kappa=CONFIG["kappa"], alpha=alpha)
    elif CONFIG["type"] == "rotating_ellipse":
        F = rotating_ellipse_map(
            a=CONFIG["a"], b=CONFIG["b"], m=CONFIG["m_rot"])
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

    Seq.evaluate_all()
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

    def B_xyz(p):
        x, y, z = F(p)
        R = (x**2 + y**2)**0.5
        phi = jnp.arctan2(y, x)

        # if CONFIG["type"] == "tokamak":
        #     BR = z * R
        #     Bphi = tau / R
        #     Bz = - (kappa**2 / 2 * (R**2 - 1**2) + z**2)
        #     Bx = BR * jnp.cos(phi) - Bphi * jnp.sin(phi)
        #     By = BR * jnp.sin(phi) + Bphi * jnp.cos(phi)
        #     return jnp.array([Bx, By, Bz])
        # else:
        #     BR = z * R
        #     Bphi = tau / R
        #     Bz = - (1**2 / 2 * (R**2 - 1**2) + z**2)
        #     Bx = BR * jnp.cos(phi) - Bphi * jnp.sin(phi)
        #     By = BR * jnp.sin(phi) + Bphi * jnp.cos(phi)
        #     return jnp.array([Bx, By, Bz])
        DFx = jax.jacfwd(F)(p)
        # purely poloidal
        return DFx[:, 1]

    B_hat = jnp.linalg.solve(Seq.M2, Seq.P2(B_xyz))
    B_hat = Seq.P_Leray @ B_hat
    B_hat /= norm_2(B_hat)

    B_harm = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)[1][:, 0]
    B_harm = B_harm / norm_2(B_harm)

    # if CONFIG["type"] != "tokamak":
    # for ITER, B_harm + 2 B_hat does the trick (unnormalized),
    # also for ROT_ELL (or 1 for rot_ell, 8x8x5 - ~ 7% normalized)
    B_hat = B_harm + 0 * B_hat
    B_hat /= norm_2(B_hat)

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
    for i in range(1, 5000 + 1):

        state = step(state)
        if (state.picard_residuum > CONFIG["solver_tol"]
                or ~jnp.isfinite(state.picard_residuum)):
            # half time step and try again
            state = timestepper.update_dt(state, state.dt / 2)
            state = timestepper.update_B_guess(state, state.B_n)
            continue
        # otherwise, we converged - proceed
        state = timestepper.update_B_n(state, state.B_guess)

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
                       state.B_n if CONFIG["save_B"] else None)

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

    ###
    # Post-processing
    ###
    B_hat = state.B_n
    print("Simulation finished, post-processing...")
    if CONFIG["save_B"]:
        p_fields = [get_pressure(B) for B in B_fields]
    p_hat = get_pressure(B_hat)

    # # save final state on a grid in physical space
    # grid_3d = get_3d_grids(F, x_min=1e-3,
    #                        nx=ps[0]*ns[0]*2,
    #                        ny=ps[1]*ns[1]*2,
    #                        nz=ps[2]*ns[2]*2 if ns[2] > 1 else 1)

    # B_final = Pushforward(DiscreteFunction(
    #     B_hat, Seq.Λ2, Seq.E2), F, 2)
    # B_final_values = jax.vmap(B_final)(grid_3d[0])
    # p_final = Pushforward(DiscreteFunction(
    #     p_hat, Seq.Λ0, Seq.E0), F, 0)
    # p_final_values = jax.vmap(p_final)(grid_3d[0])
    # grid_points = grid_3d[1]

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
        # f.create_dataset("B_final_values", data=B_final_values)
        # f.create_dataset("p_final_values", data=p_final_values)
        # f.create_dataset("grid_points", data=grid_points)
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

    # alternative ICs
    # DFp = jax.jacfwd(F)(p)
    # J = jnp.linalg.det(DFp)
    # Br = J / jnp.linalg.norm(DFp[:, 1]) * 0.0
    # Btheta = J / jnp.linalg.norm(DFp[:, 1]) * eps * p[0]
    # Bzeta = J / jnp.linalg.norm(DFp[:, 2]) * R * tau
    # return DFp @ jnp.array([Br, Btheta, Bzeta]) / J
# %%

    # def anisotropic_diffusion_tensor(B_hat):
    #     B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())

    #     def B_tensor(x):
    #         Bx = B_h(x)
    #         Dfx = jax.jacfwd(F)(x)
    #         B_phys = Dfx @ Bx
    #         B_norm = (B_phys @ B_phys)**0.5
    #         return jnp.outer(Bx, Bx) / (B_norm**2)
    #     BB_jmn = jax.vmap(B_tensor)(Seq.Q.x)  # Q x 3 x 3
    #     M = jnp.einsum('ajm,jmn,bjn,j->ab', Seq.dΛ0_ijk,
    #                    BB_jmn, Seq.dΛ0_ijk, Seq.J_j)
    #     return Seq.E0_0.matrix() @ M @ Seq.E0_0.matrix().T

# %%
    # eps_diff = 0.0
    # Diff = eps_diff * laplace_0 + \
    #     (1-eps_diff) * anisotropic_diffusion_tensor(B_hat)

    # def localized_source(x, x0, sigma=0.05):
    #     return jnp.exp(-((x - x0) @ (x - x0)) / (2 * sigma**2)) * jnp.ones(1) / (2 * jnp.pi * sigma**2)**(3/2)
    # %%
    # T_hat = jnp.linalg.solve(Diff, Seq.P0_0(lambda x: localized_source(x, jnp.array([0, 0.0, 0.0]), 0.02)))
    # T_hat = jnp.linalg.solve(Diff, Seq.P0_0(lambda x: jnp.ones(1)))
    # T_h = Pushforward(DiscreteFunction(T_hat, Seq.Λ0, Seq.E0_0.matrix()), F, 0)
    # T_hat_iso = jnp.linalg.solve(laplace_0, Seq.P0_0(lambda x: jnp.ones(1)))
    # T_h_iso = Pushforward(DiscreteFunction(T_hat_iso, Seq.Λ0, Seq.E0_0.matrix()), F, 0)

#     T_hat = jnp.linalg.solve(Diff, M0 @ p_hat)
#     T_h = Pushforward(DiscreteFunction(T_hat, Seq.Λ0, Seq.E0_0.matrix()), F, 0)
#     T_hat_iso = jnp.linalg.solve(laplace_0, M0 @ p_hat)
#     T_h_iso = Pushforward(DiscreteFunction(
#         T_hat_iso, Seq.Λ0, Seq.E0_0.matrix()), F, 0)


# # %%
#     grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=32, ny=32,)
#                  for v in jnp.linspace(0, 1, 16, endpoint=False)]
    # %%
    # fig, ax = plot_crossections_separate(T_h, grids_pol)
    # # %%
    # fig, ax = plot_crossections_separate(T_h_iso, grids_pol)
    # # %%
    # fig, ax = plot_crossections_separate(p_h, grids_pol)
# %%


if __name__ == "__main__":
    main()
# %%
