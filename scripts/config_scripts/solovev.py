# %%
import copy
import os
import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.BoundaryFitting import cerfon_map, helical_map, rotating_ellipse_map, w7x_map
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.InputOutput import parse_args, unique_id
from mrx.IterativeSolvers import picard_solver
from mrx.Nonlinearities import CrossProductProjection
from mrx.Plotting import get_2d_grids, get_3d_grids

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/solovev/"
os.makedirs(outdir, exist_ok=True)


###
# Default configuration
###
DEVICE_PRESETS = {
    "ITER":  {"eps": 0.32, "kappa": 1.7,  "delta": 0.33,
              "q_star": 1.57, "type": "tokamak",
              "n_zeta": 1, "p_zeta": 0},
    "NSTX":  {"eps": 0.78, "kappa": 2.0, "delta": 0.35,
              "q_star": 2.0, "type": "tokamak",
              "n_zeta": 1, "p_zeta": 0},
    "SPHERO": {"eps": 0.95, "kappa": 1.0, "delta": 0.2,
               "q_star": 0.0, "type": "tokamak",
               "n_zeta": 1, "p_zeta": 0},
    "ROT_ELL": {"eps": 0.1, "kappa": 4, "m_rot": 5,
                "q_star": 0.0, "type": "rotating_ellipse"},
    "HELIX": {"eps": 0.33, "kappa": 1.7, "delta": 0.33, "m_delta": 3,
              "h_helix": 0.20,  "m_helix": 3,
              "q_star": 2.0,  "type": "helix"},
    "W7X": {"eps": 0.2, "q_star": 0.0,  "type": "w7x"},
}

CONFIG = {
    "run_name": "",  # Name for the run. If empty, a hash will be created

    "type": "w7x",  # Type of configuration: "tokamak" or "helix" or "rotating_ellipse"

    ###
    # Parameters describing the domain.
    ###
    "eps":      0.33,  # aspect ratio
    "kappa":    1.0,   # Elongation parameter
    "q_star":   2.0,  # toroidal field strength
    "delta":    0.0,  # triangularity
    "m_helix":  3,     # poloidal mode number of helix
    "h_helix":  0.25,  # radius of helix turns
    "m_rot":   2,      # mode number of rotating ellipse
    "m_delta": 3,    # mode number of variation of triangularity in helix

    ###
    # Discretization
    ###
    "n_r": 16,       # Number of radial splines
    "n_theta": 8,   # Number of poloidal splines
    "n_zeta": 4,    # Number of toroidal splines
    "p_r": 3,       # Degree of radial splines
    "p_theta": 3,     # Degree of poloidal splines
    "p_zeta": 3,    # Degree of toroidal splines

    ###
    # Hyperparameters for the relaxation
    ###
    "precond":               False,     # Use preconditioner
    "precond_compute_every": 1000,       # Recompute preconditioner every n iterations
    # Regularization, u = (-Δ)⁻ᵞ (J x B - grad p)
    "gamma":                 0,
    "dt":                    1e-6,      # initial time step
    # time-steps are increased by this factor and decreased by its square
    "dt_factor":             1.01,
    "maxit":                 10_000,   # max. Number of time steps
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


def run(CONFIG):
    run_name = CONFIG["run_name"]
    if run_name == "":
        run_name = unique_id(8)

    print("Running simulation " + run_name + "...")

    delta = CONFIG["delta"]
    kappa = CONFIG["kappa"]
    q_star = CONFIG["q_star"]
    eps = CONFIG["eps"]
    alpha = jnp.arcsin(delta)
    tau = q_star * eps * kappa * (1 + kappa**2) / (kappa + 1)
    π = jnp.pi

    gamma = CONFIG["gamma"]
    force_free = CONFIG["force_free"]
    η = CONFIG["eta"]

    start_time = time.time()

    if CONFIG["type"] == "tokamak":
        F = cerfon_map(eps, kappa, alpha)
    elif CONFIG["type"] == "helix":
        F = helical_map(eps, CONFIG["h_helix"], CONFIG["m_helix"])
    elif CONFIG["type"] == "rotating_ellipse":
        F = rotating_ellipse_map(eps, CONFIG["kappa"], CONFIG["m_rot"])
    elif CONFIG["type"] == "w7x":
        F = w7x_map()
    else:
        raise ValueError("Unknown configuration type.")

    F = jax.jit(F)

    ns = (CONFIG["n_r"], CONFIG["n_theta"], CONFIG["n_zeta"])
    ps = (CONFIG["p_r"], CONFIG["p_theta"], 0
          if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
    q = max(ps)
    types = ("clamped", "periodic",
             "constant" if CONFIG["n_zeta"] == 1 else "periodic")

    print("Setting up FEM spaces...")

    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

    assert jnp.min(Seq.J_j) > 0, "Mapping is singular!"

    M0 = Seq.assemble_M0_0()
    M1 = Seq.assemble_M1_0()
    M2 = Seq.assemble_M2_0()
    M3 = Seq.assemble_M3_0()
    M12 = Seq.assemble_M12_0()
    M03 = Seq.assemble_M03_0()

    ###
    # Operators
    ###

    grad = jnp.linalg.solve(M1, Seq.assemble_grad_0())
    curl = jnp.linalg.solve(M2, Seq.assemble_curl_0())
    dvg = jnp.linalg.solve(M3, Seq.assemble_dvg_0())
    weak_grad = -jnp.linalg.solve(M2, Seq.assemble_dvg_0().T)
    weak_curl = jnp.linalg.solve(M1, Seq.assemble_curl_0().T)
    weak_dvg = -jnp.linalg.solve(M0, Seq.assemble_grad_0().T)

    laplace_0 = Seq.assemble_gradgrad_0()                         # dim ker = 0
    laplace_1 = Seq.assemble_curlcurl_0() - M1 @ grad @ weak_dvg  # dim ker = 0 (no voids)
    laplace_2 = M2 @ curl @ weak_curl + \
        Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)
    laplace_3 = - M3 @ dvg @ weak_grad  # dim ker = 1 (constants)

    P_Leray = jnp.eye(M2.shape[0]) + \
        weak_grad @ jnp.linalg.pinv(laplace_3) @ M3 @ dvg

    P_2to1 = jnp.linalg.solve(M1, M12)

    P_1x1to2 = CrossProductProjection(
        Seq.Λ2, Seq.Λ1, Seq.Λ1, Seq.Q, Seq.F,
        En=Seq.E2_0, Em=Seq.E1_0, Ek=Seq.E1_0,
        Λn_ijk=Seq.Λ2_ijk, Λm_ijk=Seq.Λ1_ijk, Λk_ijk=Seq.Λ1_ijk,
        J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
    P_2x1to1 = CrossProductProjection(
        Seq.Λ1, Seq.Λ2, Seq.Λ1, Seq.Q, Seq.F,
        En=Seq.E1_0, Em=Seq.E2_0, Ek=Seq.E1_0,
        Λn_ijk=Seq.Λ1_ijk, Λm_ijk=Seq.Λ2_ijk, Λk_ijk=Seq.Λ1_ijk,
        J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)

    def δB(B, u):
        H = P_2to1 @ B
        uxH = jnp.linalg.solve(M1, P_2x1to1(u, H))
        return curl @ uxH

    def uxJ(B, u):
        J = weak_curl @ B
        return jnp.linalg.solve(M1, P_2x1to1(u, J))

    def δδE(B):
        X = jnp.eye(B.shape[0])
        δBᵢ = jax.vmap(δB, in_axes=(None, 1), out_axes=1)(B, X)
        # shape is (n2, n2)
        ΛxJᵢ = jax.vmap(uxJ, in_axes=(None, 1), out_axes=1)(B, X)
        # shape is (n2, n1)
        H = (δBᵢ.T @ M2 @ δBᵢ + ΛxJᵢ.T @ M12 @ δBᵢ)
        return (H + H.T) / 2

    iterations = [0]
    force_trace = []
    energy_trace = []
    energy_diff_trace = []
    helicity_trace = []
    divergence_trace = []
    picard_iterations = []
    picard_errors = []
    timesteps = []
    velocity_trace = []
    wall_time_trace = []
    if CONFIG["save_B"]:
        B_fields = []

    def append_all(i, f, E, dE, H, dvg, v, p_i, e, dt, B=None):
        iterations.append(i)
        force_trace.append(f)
        energy_trace.append(E)
        energy_diff_trace.append(dE)
        helicity_trace.append(H)
        divergence_trace.append(dvg)
        velocity_trace.append(v)
        picard_iterations.append(p_i)
        picard_errors.append(e)
        timesteps.append(dt)
        wall_time_trace.append(time.time() - start_time)
        if CONFIG["save_B"]:
            B_fields.append(B)

    @jax.jit
    def compute_diagnostics(B_hat):
        A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
        B_harm_hat = B_hat - curl @ A_hat
        dvg_B = (dvg @ B_hat @ M3 @ dvg @ B_hat)**0.5
        return B_hat @ M2 @ B_hat / 2, A_hat @ M12 @ (B_hat + B_harm_hat), dvg_B

    @jax.jit
    def compute_pressure(B_hat):
        if not force_free:
            J_hat = weak_curl @ B_hat
            H_hat = P_2to1 @ B_hat
            JxH_hat = jnp.linalg.solve(M2, P_1x1to2(J_hat, H_hat))
            return -jnp.linalg.solve(laplace_0, M03 @ dvg @ JxH_hat)
        else:
            # Compute p(x) = J · B / |B|²
            B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
            J_hat = weak_curl @ B_hat
            J_h = DiscreteFunction(J_hat, Seq.Λ1, Seq.E1_0.matrix())

            def lmbda(x):
                DFx = jax.jacfwd(F)(x)
                Bx = B_h(x)
                return (J_h(x) @ Bx) / ((DFx @ Bx) @ DFx @ Bx) * jnp.linalg.det(DFx) * jnp.ones(1)
            return jnp.linalg.solve(M0, Seq.P0_0(lmbda))

    # State is given by x = (B˖, (B, dt, |JxB - grad p|, |u|, Hess))
    @jax.jit
    def implicit_update(x):
        dt = x[1][1]
        B_nplus1 = x[0]
        B_n = x[1][0]
        B_mid = (B_nplus1 + B_n) / 2
        J_hat = weak_curl @ B_mid
        H_hat = P_2to1 @ B_mid
        JxH_hat = jnp.linalg.solve(M2, P_1x1to2(J_hat, H_hat))
        if not force_free:
            f_hat = P_Leray @ JxH_hat
            gradp_hat = -f_hat + JxH_hat
            norm = (gradp_hat @ M2 @ gradp_hat)**0.5
        else:
            f_hat = JxH_hat
            norm = (B_hat @ M2 @ B_hat)**0.5
        f_norm = (f_hat @ M2 @ f_hat)**0.5 / norm
        u_hat = f_hat
        for _ in range(gamma):
            u_hat = jnp.linalg.solve(M2 + laplace_2, M2 @ u_hat)
        if CONFIG["precond"]:
            Hessian = x[1][4]
            u_hat = P_Leray @ jnp.linalg.lstsq(
                Hessian, P_Leray.T @ M2 @ u_hat)[0]
            # P_Leray.T technically not needed because u is already div_free
        else:
            Hessian = None
        u_norm = (u_hat @ M2 @ u_hat)**0.5
        E_hat = jnp.linalg.solve(M1, P_2x1to1(u_hat, H_hat)) - η * J_hat
        return (B_n + dt * curl @ E_hat, (B_n, dt, f_norm, u_norm, Hessian))

    @jax.jit
    def update(x):
        _x, error, iters = picard_solver(
            implicit_update,
            x,
            tol=CONFIG["solver_tol"],
            norm=lambda B: (B @ M2 @ B)**0.5,
            inprod=lambda u, v: u @ M2 @ v,
            max_iter=CONFIG["solver_maxit"],
        )
        # _x has the form (B˖, (B, dt, |JxB - grad p|, |u|, Hess))
        # for next iterate, update B to B˖
        x = (_x[0], (_x[0], *_x[1][1:]))
        return x, error, iters

    def B_xyz(p):
        x, y, z = F(p)
        R = (x**2 + y**2)**0.5
        phi = jnp.arctan2(y, x) / (2 * π)

        BR = z * R
        Bphi = tau / R
        if CONFIG["type"] == "tokamak":
            Bz = - (kappa**2 / 2 * (R**2 - 1**2) + z**2)
        else:
            Bz = - (1 / 2 * (R**2 - 1**2) + z**2)

        Bx = BR * jnp.cos(2 * π * phi) - Bphi * jnp.sin(2 * π * phi)
        By = BR * jnp.sin(2 * π * phi) + Bphi * jnp.cos(2 * π * phi)
        return jnp.array([Bx, By, Bz])
        # DFp = jax.jacfwd(F)(p)
        # J = jnp.linalg.det(DFp)
        # Br = J / jnp.linalg.norm(DFp[:, 1]) * 0.0
        # Btheta = J / jnp.linalg.norm(DFp[:, 1]) * (-p[0])
        # Bzeta = J / jnp.linalg.norm(DFp[:, 2]) * (-4 * R)
        # return DFp @ jnp.array([Br, Btheta, Bzeta]) / J

    # Set up inital condition
    B_harm_hat = jnp.linalg.eigh(laplace_2)[1][:, 0]
    B_harm_hat /= (B_harm_hat @ M2 @ B_harm_hat)**0.5

    # One step of resisitive relaxation to get J x n = 0 on ∂Ω
    B_hat = jnp.linalg.solve(
        jnp.eye(M2.shape[0]) + laplace_2, B_harm_hat)
    B_hat = B_harm_hat
    # B_hat = jnp.linalg.solve(M2, Seq.P2_0(B_xyz))
    # # if CONFIG["type"] == "w7x":
    # #     B_hat /= (B_hat @ M2 @ B_hat)**0.5
    # #     B_hat = B_hat * 0.05 + B_harm_hat * 0.95
    # B_hat = P_Leray @ B_hat

    B_hat /= (B_hat @ M2 @ B_hat)**0.5
    p_hat = compute_pressure(B_hat)
    # %%
    eps_precond = jnp.linalg.eigvalsh(
        P_Leray.T @ δδE(B_hat) @ P_Leray)[-1] * 1e-2 if CONFIG["precond"] else 0.0

    @jax.jit
    def compute_Hessian(B_hat):
        return P_Leray.T @ (M2 * eps_precond + δδE(B_hat)) @ P_Leray if CONFIG["precond"] else None

    x = (B_hat, (B_hat, CONFIG["dt"], 0.0, 0.0,
         M2 if CONFIG["precond"] else None))
    # initial state: (B, (B_old, dt, |JxB - grad p|, |u|, Hess))

    __x, _, _ = update(x)  # also doing the compilation here

    force_norm, velocity_norm = __x[1][2], __x[1][3]

    force_trace.append(force_norm)
    velocity_trace.append(velocity_norm)

    energy, helicity, divergence_B = compute_diagnostics(B_hat)
    energy_trace.append(energy)
    helicity_trace.append(helicity)
    divergence_trace.append(divergence_B)
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
    dt = CONFIG["dt"]
    for i in range(1, 1 + int(CONFIG["maxit"])):
        x_old = copy.deepcopy(x)

        x, picard_err, picard_it = update(x)
        if picard_err > CONFIG["solver_tol"] or jnp.isnan(picard_err) or jnp.isinf(picard_err):
            # halve time step and try again
            x = (x_old[0], (x_old[0], dt/2, x_old[1]
                 [2], x_old[1][3], x_old[1][4]))
            continue
        # otherwise, we converged - proceed
        _, dt, force_norm, velocity_norm, invHessian = x[1]

        if CONFIG["precond"] and (i % CONFIG["precond_compute_every"] == 0):
            invHessian = compute_Hessian(x[0])

        if picard_it <= CONFIG["solver_critit"]:
            dt_new = dt * CONFIG["dt_factor"]
        else:
            dt_new = dt / (CONFIG["dt_factor"])**2

        # collect everything
        x = (x[0], (x[0], dt_new, force_norm, velocity_norm, invHessian))

        if i % CONFIG["save_every"] == 0 or i == CONFIG["maxit"]:
            energy, helicity, divergence_B = compute_diagnostics(x[0])
            append_all(i,
                       force_norm,
                       energy,
                       (x_old[0] @ M2 @ x_old[0] - x[0] @ M2 @ x[0]) / 2,
                       helicity,
                       divergence_B,
                       velocity_norm,
                       picard_it,
                       picard_err,
                       dt_new,
                       x[0] if CONFIG["save_B"] else None)

        if i % CONFIG["print_every"] == 0:
            print(
                f"Iteration {i}, u norm: {velocity_norm:.2e}, force norm: {force_norm:.2e}")
            if CONFIG["verbose"]:
                print(
                    f"   dt: {dt_new:.2e}, picard iters: {picard_it:.2e}, picard err: {picard_err:.2e}")
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
    B_hat = x[0]
    print("Simulation finished, post-processing...")
    if CONFIG["save_B"]:
        p_fields = [compute_pressure(B) for B in B_fields]
    p_hat = compute_pressure(B_hat)
    p_h = Pushforward(DiscreteFunction(p_hat, Seq.Λ0, Seq.E0_0.matrix()), F, 0)

    # save final state on a grid in physical space
    grid_3d = get_3d_grids(F, x_min=1e-3,
                           nx=ps[0]*ns[0]*2,
                           ny=ps[1]*ns[1]*2,
                           nz=ps[2]*ns[2]*2 if ns[2] > 1 else 1)

    B_final = Pushforward(DiscreteFunction(
        B_hat, Seq.Λ2, Seq.E2_0.matrix()), F, 2)
    B_final_values = jax.vmap(B_final)(grid_3d[0])
    p_final = Pushforward(DiscreteFunction(
        p_hat, Seq.Λ0, Seq.E0_0.matrix()), F, 0)
    p_final_values = jax.vmap(p_final)(grid_3d[0])
    grid_points = grid_3d[1]

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
        f.create_dataset("B_final_values", data=B_final_values)
        f.create_dataset("p_final_values", data=p_final_values)
        f.create_dataset("grid_points", data=grid_points)
        f.create_dataset("energy_trace", data=jnp.array(energy_trace))
        f.create_dataset("energy_diff_trace",
                         data=jnp.array(energy_diff_trace))
        f.create_dataset("helicity_trace", data=jnp.array(helicity_trace))
        f.create_dataset("divergence_B_trace",
                         data=jnp.array(divergence_trace))
        f.create_dataset("velocity_trace", data=jnp.array(velocity_trace))
        f.create_dataset("picard_iterations",
                         data=jnp.array(picard_iterations))
        f.create_dataset("picard_errors", data=jnp.array(picard_errors))
        f.create_dataset("timesteps", data=jnp.array(timesteps))
        f.create_dataset("harmonic_norm", data=jnp.array(
            [(B_harm_hat @ M2 @ B_harm_hat)**0.5]))
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


if __name__ == "__main__":
    main()
# %%
