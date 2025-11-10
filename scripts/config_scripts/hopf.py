# %%
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.io import parse_args, unique_id
from mrx.iterative_solvers import picard_solver
from mrx.nonlinearities import CrossProductProjection

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/hopf/"
os.makedirs(outdir, exist_ok=True)

###
# Default configuration
###
CONFIG = {

    "run_name": "",  # Name for the run. If empty, a hash will be created

    ###
    # Parameters describing the domain and initial psi
    ###
    "omega_1": 3,  # first winding number
    "omega_2": 2,  # second winding number
    "s": 1,        # scale parameter

    ###
    # Discretization
    ###
    "n_r": 6,       # Number of radial splines
    "n_chi": 6,     # Number of poloidal splines
    "n_zeta": 6,    # Number of toroidal splines
    "p_r": 3,       # Degree of radial splines
    "p_chi": 3,     # Degree of poloidal splines
    "p_zeta": 3,    # Degree of toroidal splines

    ###
    # Hyperparameters for the relaxation
    ###
    "gamma": 0,                  # Regularization, u = (-Δ)⁻ᵞ (J x B - grad p)
    "eps": 1e-2,                 # Regularization for the initial condition
    "dt": 10.0,                  # Time step
    "dt_max": 1000.0,             # max. time step
    "n_steps": 50_000,           # max. Number of time steps
    # Stop if || JxB - grad p || < this or (force-free) || JxB || < this
    "force_tol": 1e-12,
    "eta": 0.0,                  # Resistivity
    "force_free": False,         # If True, solve for JxB = 0. If False, JxB = grad p

    ###
    # Solver hyperparameters
    ###
    "max_iter": 100,              # Maximum number of iterations
    "solver_tol": 1e-9,           # Tolerance for convergence
}


def run(CONFIG):
    run_name = CONFIG["run_name"]
    if run_name == "":
        run_name = unique_id(8)

    print("Running simulation " + run_name + "...")

    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    π = jnp.pi
    ω1 = CONFIG["omega_1"]
    ω2 = CONFIG["omega_2"]
    s = CONFIG["s"]

    eps = CONFIG["eps"]

    gamma = CONFIG["gamma"]
    force_free = CONFIG["force_free"]
    eta = CONFIG["eta"]
    dt0 = CONFIG["dt"]
    dt_max = CONFIG["dt_max"]
    n_steps = int(CONFIG["n_steps"])
    force_tol = CONFIG["force_tol"]

    solver_tol = CONFIG["solver_tol"]
    max_iter = int(CONFIG["max_iter"])

    def F(x):
        return jnp.array([x[0] * 8 - 4, x[1] * 8 - 4, x[2] * 20 - 10])

    ns = (CONFIG["n_r"], CONFIG["n_chi"], CONFIG["n_zeta"])
    ps = (CONFIG["p_r"], CONFIG["p_chi"], CONFIG["p_zeta"])
    q = max(ps)
    types = ("clamped", "clamped", "clamped")

    Seq = DeRhamSequence(ns, ps, q, types, F, polar=False)

    M0 = Seq.assemble_M0_0()
    M1 = Seq.assemble_M1_0()
    M2 = Seq.assemble_M2_0()
    M3 = Seq.assemble_M3_0()

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

    M12 = Seq.assemble_M12_0()
    M03 = Seq.assemble_M03_0()

    def B_xyz(p):
        x, y, z = F(p)
        rsq = (x**2 + y**2 + z**2)
        return 4 * jnp.sqrt(s) / (π * (1 + rsq)**3 * (ω1**2 + ω2**2)**0.5) * jnp.array([
            2 * ω2 * y - 2 * ω1 * x * z,
            - 2 * ω2 * x - 2 * ω1 * y * z,
            ω1 * (x**2 + y**2 - z**2 - 1)
        ])

    P_JxH = CrossProductProjection(
        Seq.Lambda_2, Seq.Lambda_1, Seq.Lambda_1, Seq.Q, Seq.F,
        En=Seq.E2_0, Em=Seq.E1_0, Ek=Seq.E1_0,
        Λn_ijk=Seq.Λ2_ijk, Λm_ijk=Seq.Λ1_ijk, Λk_ijk=Seq.Λ1_ijk,
        J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
    P_uxH = CrossProductProjection(
        Seq.Lambda_1, Seq.Lambda_2, Seq.Lambda_1, Seq.Q, Seq.F,
        En=Seq.E1_0, Em=Seq.E2_0, Ek=Seq.E1_0,
        Λn_ijk=Seq.Λ1_ijk, Λm_ijk=Seq.Λ2_ijk, Λk_ijk=Seq.Λ1_ijk,
        J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)

    P_Leray = jnp.eye(M2.shape[0]) + \
        weak_grad @ jnp.linalg.pinv(laplace_3) @ M3 @ dvg

    # Set up inital condition
    B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_xyz))
    # One step of resisitive relaxation to get J x n = 0 on ∂Ω
    B_hat = jnp.linalg.solve(
        jnp.eye(M2.shape[0]) + eps * curl @ weak_curl, B_hat)
    A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
    B_harm_hat = B_hat - curl @ A_hat

    force_trace = []
    E_trace = []
    H_trace = []
    dvg_trace = []
    iters = []
    errs = []

    @jax.jit
    def L2norm(x):
        dB, _ = jnp.split(x, 2)
        return (dB @ M2 @ dB)**0.5

    @jax.jit
    def implicit_update(x):
        B_nplus1, B_n = jnp.split(x, 2)
        B_mid = (B_nplus1 + B_n) / 2
        J_hat = weak_curl @ B_mid
        H_hat = jnp.linalg.solve(M1, M12 @ B_mid)
        JxH_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
        if force_free:
            u_hat = JxH_hat
        else:
            u_hat = P_Leray @ JxH_hat
        for _ in range(gamma):
            u_hat = jnp.linalg.inv(M2 + laplace_2) @ M2 @ u_hat
        u_norm = (u_hat @ M2 @ u_hat)**0.5
        dt = jnp.minimum(dt0 / u_norm, dt_max)
        E_hat = jnp.linalg.solve(M1, P_uxH(u_hat, H_hat)) - eta * J_hat
        return jnp.concatenate([B_n + dt * curl @ E_hat, B_n])

    @jax.jit
    def compute_force_norm(B):
        J_hat = weak_curl @ B
        H_hat = jnp.linalg.solve(M1, M12 @ B)
        JxH_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
        if force_free:
            u_hat = JxH_hat
        else:
            u_hat = P_Leray @ JxH_hat
        return (u_hat @ M2 @ u_hat)**0.5

    @jax.jit
    def update(x):
        return picard_solver(implicit_update, x, tol=solver_tol, norm=L2norm, max_iter=max_iter)

    for i in range(n_steps):
        x = jnp.concatenate([B_hat, B_hat])
        x, err, it = update(x)

        B_hat, _ = jnp.split(x, 2)
        A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)

        iters.append(it)
        errs.append(err)
        force_trace.append(compute_force_norm(B_hat))
        E_trace.append(B_hat @ M2 @ B_hat / 2)
        H_trace.append(A_hat @ M12 @ (B_hat + B_harm_hat))
        dvg_trace.append((dvg @ B_hat @ M3 @ dvg @ B_hat)**0.5)
        if iters[-1] == max_iter and err > solver_tol:
            print(
                f"Picard solver did not converge in {max_iter} iterations (err={err:.2e})")
            break
        if i % 1000 == 0:
            print(f"Iteration {i}, u norm: {force_trace[-1]}")
        if force_trace[-1] < force_tol:
            print(
                f"Converged to force tolerance {force_tol} after {i} steps.")
            break

    ###
    # Post-processing
    ###

    print("Simulation finished, post-processing...")

    if not force_free:
        # Compute pressure
        J_hat = weak_curl @ B_hat
        H_hat = jnp.linalg.solve(M1, M12 @ B_hat)
        JxH_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
        p_hat = -jnp.linalg.solve(laplace_0, M03 @ dvg @ JxH_hat)
    else:
        # Compute p(x) = J · B / |B|²
        B_h = DiscreteFunction(B_hat, Seq.Lambda_2, Seq.E2_0.matrix())
        J_hat = weak_curl @ B_hat
        J_h = DiscreteFunction(J_hat, Seq.Lambda_1, Seq.E1_0.matrix())

        @jax.jit
        def lmbda(x):
            DFx = jax.jacfwd(F)(x)
            Bx = B_h(x)
            return (J_h(x) @ Bx) / ((DFx @ Bx) @ DFx @ Bx) * jnp.linalg.det(DFx) * jnp.ones(1)
        p_hat = jnp.linalg.solve(M0, Seq.P0_0(lmbda))

    ###
    # Save stuff
    ###

    # Write to HDF5
    with h5py.File(outdir + run_name + ".h5", "w") as f:
        # Store arrays
        f.create_dataset("B_hat", data=B_hat)
        f.create_dataset("p_hat", data=p_hat)
        f.create_dataset("energy_trace", data=jnp.array(E_trace))
        f.create_dataset("helicity_trace", data=jnp.array(H_trace))
        f.create_dataset("divergence_B_trace", data=jnp.array(dvg_trace))
        f.create_dataset("force_trace", data=jnp.array(force_trace))
        f.create_dataset("iters", data=jnp.array(iters))
        f.create_dataset("errs", data=jnp.array(errs))
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


def main():
    # Get user input
    params = parse_args()
    # replace defaults with user input
    for k, v in params.items():
        if k in CONFIG:
            CONFIG[k] = v
        else:
            print(f"Unknown parameter '{k}' - ignoring.")

    run(CONFIG)


if __name__ == "__main__":
    main()
# %%
