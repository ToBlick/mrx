# %%
import copy
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.InputOutput import parse_args, unique_id
from mrx.IterativeSolvers import picard_solver
from mrx.Nonlinearities import CrossProductProjection

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/solovev/"
os.makedirs(outdir, exist_ok=True)


CONFIG = {
    "run_name": "",  # Name for the run. If empty, a hash will be created

    ###
    # Parameters describing the domain. Default is ITER-like
    ###
    "eps":      0.32,  # aspect ratio
    "q_star":   1.57,  # toroidal field strength
    "R_0":      1.0,   # Major radius

    ###
    # Discretization
    ###
    "n_r": 5,       # Number of radial splines
    "n_theta": 5,   # Number of poloidal splines
    "n_zeta": 4,    # Number of toroidal splines
    "p_r": 2,       # Degree of radial splines
    "p_theta": 2,     # Degree of poloidal splines
    "p_zeta": 2,    # Degree of toroidal splines

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
    "maxit":                 50_000,   # max. Number of time steps
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

    q_star = CONFIG["q_star"]
    ɛ = CONFIG["eps"]
    R0 = CONFIG["R_0"]
    tau = q_star * ɛ
    π = jnp.pi

    gamma = CONFIG["gamma"]
    force_free = CONFIG["force_free"]
    η = CONFIG["eta"]

    start_time = time.time()

    n_turns = 3  # Number of helix turns
    h = 1/4      # radius of helix

    def X(ζ):
        return jnp.array([
            (1 + h * jnp.sin(2 * π * n_turns * ζ)) * jnp.sin(2 * π * ζ),
            (1 + h * jnp.sin(2 * π * n_turns * ζ)) * jnp.cos(2 * π * ζ),
            h * jnp.cos(2 * π * n_turns * ζ)
        ])

    def get_frame(ζ):
        dX = jax.jacrev(X)(ζ)
        τ = dX / jnp.linalg.norm(dX)  # Tangent vector

        e = jnp.array([0.0, 0.0, 1.0])
        ν1 = (e - jnp.dot(e, τ) * τ)
        ν1 = ν1 / jnp.linalg.norm(ν1)  # First normal vector
        ν2 = jnp.cross(τ, ν1)         # Second normal vector
        return τ, ν1, ν2

    def F(x):
        """Helical coordinate mapping function."""
        r, θ, ζ = x
        τ, ν1, ν2 = get_frame(ζ)
        return (X(ζ) + ɛ * r * jnp.cos(2 * π * θ) * ν1
                + ɛ * r * jnp.sin(2 * π * θ) * ν2)

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
        # ΛxJᵢ = jax.vmap(uxJ, in_axes=(None, 1), out_axes=1)(B, X)
        # shape is (n2, n1)
        return δBᵢ.T @ M2 @ δBᵢ  # + ΛxJᵢ.T @ M12 @ δBᵢ

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
            u_hat = -jnp.linalg.solve(Hessian, M2 @ u_hat)
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
        Bz = - 1**2 / 2 * (R**2 - R0**2) - z**2

        Bx = BR * jnp.cos(2 * π * phi) - Bphi * jnp.sin(2 * π * phi)
        By = BR * jnp.sin(2 * π * phi) + Bphi * jnp.cos(2 * π * phi)

        return jnp.array([Bx, By, Bz])

    # Set up inital condition
    B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_xyz))
    # One step of resisitive relaxation to get J x n = 0 on ∂Ω
    # B_hat = jnp.linalg.solve(jnp.eye(M2.shape[0]) + 1e-2 * curl @ weak_curl, B_hat)
    B_hat /= (B_hat @ M2 @ B_hat)**0.5  # normalize

    # %%
    eps_precond = jnp.linalg.eigh(
        δδE(B_hat))[0][-1] * 1e-6 if CONFIG["precond"] else 0.0

    @jax.jit
    def compute_Hessian(B_hat):
        return M2 * eps_precond + δδE(B_hat) if CONFIG["precond"] else None

    x = (B_hat, (B_hat, CONFIG["dt"], 0.0, 0.0, compute_Hessian(B_hat)))
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
    dt = CONFIG["dt"]
# %%

    for i in range(1, 1 + int(CONFIG["maxit"])):
        x_old = copy.deepcopy(x)

        x, picard_err, picard_it = update(x)
        if picard_err > CONFIG["solver_tol"] or jnp.isnan(picard_err) or jnp.isinf(picard_err):
            # halve time step and try again
            x = (x_old[0], (x_old[0], dt/2, x_old[1]
                 [2], x_old[1][3], x_old[1][4]))
            continue
        # otherwise, we converged - proceed
        _, dt, force_norm, velocity_norm, Hessian = x[1]

        if CONFIG["precond"] and (i % CONFIG["precond_compute_every"] == 0):
            Hessian = compute_Hessian(x[0])

        if picard_it <= CONFIG["solver_critit"]:
            dt_new = dt * CONFIG["dt_factor"]
        else:
            dt_new = dt / (CONFIG["dt_factor"])**2

        # collect everything
        x = (x[0], (x[0], dt_new, force_norm, velocity_norm, Hessian))

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
                f"Iteration {i}, u norm: {velocity_norm:.2e}, force norm: {force_norm:.2e}, energy: {energy:.2e}, helicity: {helicity:.2e}, ||div B||: {divergence_B:.2e}")
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

    print("Simulation finished, post-processing...")
    if CONFIG["save_B"]:
        p_fields = [compute_pressure(B) for B in B_fields]
    p_hat = compute_pressure(B_hat)
    p_h = DiscreteFunction(p_hat, Seq.Λ0, Seq.E0_0.matrix())
    u_h = p_h
    B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
    # %%

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


if __name__ == "__main__":
    main()
# %%

# # %%
    # Force_op = δδE(B_hat)

    # evd = jnp.linalg.eigh(Force_op)
    # plt.semilogy(evd[0])
    # plt.semilogy(jnp.abs(evd[0]))

    # dominant_evec = evd[1][:, -1]
    # v_h = Pushforward(DiscreteFunction(dominant_evec, Seq.Λ2, Seq.E2_0.matrix()), Seq.F, 2)

    # key = jax.random.PRNGKey(0)
    # _x = jax.random.uniform(key, shape=(2000, 3)).at[:, -1].set(0.0)
    # _y = jax.vmap(F)(_x)
    # vals = jax.vmap(v_h)(_x)
    # val_3 = vals[:, 2]  #jnp.linalg.norm(vals, axis=1)
    # colors = plt.cm.viridis((val_3 - val_3.min()) / (val_3.max() - val_3.min()))
    # plt.scatter(_y[:, 0], _y[:, 2], c=colors, s=5)
    # plt.quiver(_y[:, 0], _y[:, 2], vals[:, 0], vals[:, 2], color="black")
    # plt.xlabel("R")
    # plt.ylabel("z")

    # def angle(u):
    #     return jnp.abs(u[1]) / ((u[0]**2 + u[2]**2)**0.5 + 1e-16)

    # angles = jax.vmap(angle)(vals)
    # colors = plt.cm.viridis((angles - angles.min()) / (angles.max() - angles.min()))
    # plt.scatter(_y[:, 0], _y[:, 2], c=colors, s=5)
    # plt.colorbar()
    # plt.xlabel("R")
    # plt.ylabel("z")
    # # plt.axis("equal")
