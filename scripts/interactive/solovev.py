# %%
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.BoundaryFitting import cerfon_map
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction
from mrx.InputOutput import parse_args, unique_id
from mrx.IterativeSolvers import picard_solver
from mrx.Nonlinearities import CrossProductProjection
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Plotting import get_1d_grids, get_2d_grids
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/solovev/"
os.makedirs(outdir, exist_ok=True)


###
# Default configuration
###
CONFIG = {

    "run_name": "",  # Name for the run. If empty, a hash will be created

    ###
    # Parameters describing the domain
    ###
    "eps":      0.32,   # aspect ratio
    "kappa":    1.7,   # Elongation parameter
    "q_star":   1.57,  # toroidal field strength
    "delta":    0.33,
    
    ###
    # ITER: eps=0.32, kappa=1.7, delta=0.33, q_star=1.57
    # NSTX: eps=0.78, kappa=2, delta=0.35, q_star=2
    # SPHERO: eps=0.95, kappa=1, delta=0.2, q_star=0.0
    ###

    ###
    # Discretization
    ###
    "n_r": 8,       # Number of radial splines
    "n_chi": 8,     # Number of poloidal splines
    "n_zeta": 1,    # Number of toroidal splines
    "p_r": 3,       # Degree of radial splines
    "p_chi": 3,     # Degree of poloidal splines
    "p_zeta": 0,    # Degree of toroidal splines

    ###
    # Hyperparameters for the relaxation
    ###
    "gamma":    0,                  # Regularization, u = (-Δ)⁻ᵞ (J x B - grad p)
    "dt":       1e-4,                  # Time step
    "dt_max":   1e-1,                # max. time step
    "n_steps":  100_000,           # max. Number of time steps
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

    R0 = 1.0
    π = jnp.pi
    delta = CONFIG["delta"]
    kappa = CONFIG["kappa"]
    q_star = CONFIG["q_star"]
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
    
    alpha = jnp.arcsin(delta)
    tau = q_star * eps * kappa * (1 + kappa**2) / (kappa + 1)
    F = cerfon_map(eps, kappa, alpha, R0)
    
    time0 = time.time()
        
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(
        F, zeta=0, nx=64, tol=1e-2)
        
        # %%

    ns = (CONFIG["n_r"], CONFIG["n_chi"], CONFIG["n_zeta"])
    ps = (CONFIG["p_r"], CONFIG["p_chi"], 0
          if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
    q = max(ps)
    types = ("clamped", "periodic",
             "constant" if CONFIG["n_zeta"] == 1 else "periodic")

    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

    assert jnp.min(Seq.J_j) > 0, "Mapping is singular!"

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

# %%
    def B_xyz(p):
        x, y, z = F(p)
        R = (x**2 + y**2)**0.5
        phi = jnp.arctan2(y, x) / (2 * π)

        BR = z * R
        Bz = - kappa**2 / 2 * (R**2 - R0**2) - z**2
        Bphi = tau / R

        Bx = BR * jnp.cos(2 * π * phi) - Bphi * jnp.sin(2 * π * phi)
        By = BR * jnp.sin(2 * π * phi) + Bphi * jnp.cos(2 * π * phi)

        return jnp.array([Bx, By, Bz])

    P_JxH = CrossProductProjection(
        Seq.Λ2, Seq.Λ1, Seq.Λ1, Seq.Q, Seq.F,
        En=Seq.E2_0, Em=Seq.E1_0, Ek=Seq.E1_0,
        Λn_ijk=Seq.Λ2_ijk, Λm_ijk=Seq.Λ1_ijk, Λk_ijk=Seq.Λ1_ijk,
        J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
    P_uxH = CrossProductProjection(
        Seq.Λ1, Seq.Λ2, Seq.Λ1, Seq.Q, Seq.F,
        En=Seq.E1_0, Em=Seq.E2_0, Ek=Seq.E1_0,
        Λn_ijk=Seq.Λ1_ijk, Λm_ijk=Seq.Λ2_ijk, Λk_ijk=Seq.Λ1_ijk,
        J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)

    P_Leray = jnp.eye(M2.shape[0]) + \
        weak_grad @ jnp.linalg.pinv(laplace_3) @ M3 @ dvg

# %%
    FIG_SIZE = (12, 6)
    FIG_SIZE_SQUARE = (8, 8)
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LINE_WIDTH = 2.5
    LEGEND_SIZE = 16

# %%
    # Set up inital condition
    B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_xyz))
    # One step of resisitive relaxation to get J x n = 0 on ∂Ω
    # B_hat = jnp.linalg.solve(
    #     jnp.eye(M2.shape[0]) + eps_init * curl @ weak_curl, B_hat)
    
    B_hat /= (B_hat @ M2 @ B_hat)**0.5  # normalize
    A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
    B_harm_hat = B_hat - curl @ A_hat
    
    print(f"Initial energy: {B_hat @ M2 @ B_hat / 2:.5f}")
    print(f"Initial helicity: {A_hat @ M12 @ (B_hat + B_harm_hat):.5f}")
    print(f"Initial ||div B||: {((dvg @ B_hat) @ M3 @ dvg @ B_hat)**0.5:.2e}")

    force_trace = []
    E_trace = []
    H_trace = []
    dvg_trace = []
    iters = []
    errs = []
    dts = []

    dim0 = M0.shape[0]
    dim1 = M1.shape[0]
    dim2 = M2.shape[0]
    dim3 = M3.shape[0]

    # State is given by x = (B˖, B, dt, |JxB - grad p|) 
    #
    @jax.jit
    def L2norm(x):
        dB = x[:dim2]
        return (dB @ M2 @ dB)**0.5

    @jax.jit
    def implicit_update(x):
        B_nplus1, B_n, _, _ = jnp.split(x, [dim2, 2*dim2, 2*dim2+1])
        B_mid = (B_nplus1 + B_n) / 2
        J_hat = weak_curl @ B_mid
        H_hat = jnp.linalg.solve(M1, M12 @ B_mid)
        JxH_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
        u_hat = JxH_hat if force_free else P_Leray @ JxH_hat
        f_norm = (u_hat @ M2 @ u_hat)**0.5
        for _ in range(gamma):
            u_hat = jnp.linalg.inv(M2 + laplace_2) @ M2 @ u_hat
        dt = jnp.minimum(dt0 / f_norm, dt_max)
        E_hat = jnp.linalg.solve(M1, P_uxH(u_hat, H_hat)) - eta * J_hat
        return jnp.concatenate((B_n + dt * curl @ E_hat, 
                                B_n, 
                                jnp.ones(1) * dt, 
                                jnp.ones(1) * f_norm))

    @jax.jit
    def update(x):
        return picard_solver(implicit_update, x, tol=solver_tol, norm=L2norm, max_iter=max_iter)


    # x = jnp.concatenate((B_hat, B_hat, jnp.ones(1) * dt0, jnp.zeros(1)))
    dt = dt0
    force_err = 0
    
    x = jnp.concatenate((B_hat, B_hat, jnp.ones(1) * dt, jnp.ones(1) * force_err))
    _, _, _ = update(x)  # compile
    print(f"Initial force error: {update(x)[0][-1]:.2e}")
    # %%
    time1 = time.time()
    print(f"Setup took {time1 - time0:.2f} seconds.")
    print("Starting main loop...")
    for i in range(1,n_steps+1):
        x = jnp.concatenate((B_hat, B_hat, jnp.ones(1) * dt, jnp.ones(1) * force_err))
        x, picard_err, it = update(x)
        
        force_err = x[-1]
        dt = x[-2]
        B_hat = x[:dim2]
        
        A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
        dts.append(dt)
        iters.append(it)
        errs.append(picard_err)
        force_trace.append(force_err)
        E_trace.append(B_hat @ M2 @ B_hat / 2)
        H_trace.append(A_hat @ M12 @ (B_hat + B_harm_hat))
        dvg_trace.append((dvg @ B_hat @ M3 @ dvg @ B_hat)**0.5)
        if iters[-1] == max_iter and picard_err > solver_tol:
            print(
                f"Picard solver did not converge in {max_iter} iterations (err={picard_err:.2e})")
            break
        if i % 100 == 0:
            print(f"Iteration {i}, u norm: {force_trace[-1]}")
        if force_trace[-1] < force_tol:
            print(
                f"Converged to force tolerance {force_tol} after {i} steps.")
            break
# %%

    time2 = time.time()
    print(f"Main loop took {time2 - time1:.2f} seconds for {i} steps, avg. { (time2 - time1)/i:.5f} s/step.")
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
        B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
        J_hat = weak_curl @ B_hat
        J_h = DiscreteFunction(J_hat, Seq.Λ1, Seq.E1_0.matrix())

        @jax.jit
        def lmbda(x):
            DFx = jax.jacfwd(F)(x)
            Bx = B_h(x)
            return (J_h(x) @ Bx) / ((DFx @ Bx) @ DFx @ Bx) * jnp.linalg.det(DFx) * jnp.ones(1)
        p_hat = jnp.linalg.solve(M0, Seq.P0_0(lmbda))


# %%
    energy_trace = E_trace
    helicity_trace = H_trace


    
    FIG_SIZE = (12, 6)
    SQUARE_FIG_SIZE = (8, 8)
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LINE_WIDTH = 2.5
    LEGEND_SIZE = 16
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(
        F, zeta=0, nx=64, tol=1e-2)
    print("Generating pressure plot...")
    # Plot number one: pressure contour plot
    p_h = DiscreteFunction(p_hat, Seq.Λ0, Seq.E0_0.matrix())
    p_h_xyz = Pushforward(p_h, F, 0)

    _s = jax.vmap(F)(jnp.vstack(
        [jnp.ones(256), jnp.linspace(0, 1, 256), jnp.zeros(256)]).T)

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    # Plot the line first
    ax.plot(_s[:, 0], _s[:, 2], 'k--',
            linewidth=LINE_WIDTH, label="trajectory")

    # Evaluate Z values for contour
    Z = jax.vmap(p_h_xyz)(_x).reshape(_y1.shape)

    # Filled contours for nicer visualization
    cf = ax.contourf(_y1, _y3, Z, levels=20, cmap="plasma", alpha=0.8)

    # Contour lines on top
    # cs = ax.contour(_y1, _y3, Z, levels=10, colors="k", linewidths=LINE_WIDTH)
    # ax.clabel(cs, fmt="%.2f", fontsize=0.5 * LABEL_SIZE)

    spacing = 0.05
    ax.set_xlim(jnp.min(_s[:, 0]) - spacing, jnp.max(_s[:, 0]) + spacing)
    ax.set_ylim(jnp.min(_s[:, 2]) - spacing, jnp.max(_s[:, 2]) + spacing)
    ax.set_aspect('equal')
    ax.set_xlabel("R", fontsize=LABEL_SIZE)
    ax.set_ylabel("z", fontsize=LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"p", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    # Save
    plt.tight_layout()

# %%
    print("Generating convergence plot...")

    # Figure 2: Energy and Force

    fig1, ax2 = plt.subplots(figsize=FIG_SIZE)
    ax1 = ax2.twinx()
    

    # Plot Energy on the left y-axis (ax1)
    color1 = 'purple'
    ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
    ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$',
                   color=color1, fontsize=LABEL_SIZE)
    ax1.plot(jnp.array(energy_trace),
             label=r'$\frac{1}{2} \| B \|^2$', color=color1, linestyle='-.', lw=LINE_WIDTH)
    # ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size

    helicity_change = jnp.abs(
        jnp.array(jnp.array(helicity_trace) - helicity_trace[0])) / helicity_trace[0]
    # Plot Force on the right y-axis (ax2)
    color2 = 'black'
    ax2.set_ylabel(r'$\|J \times B - \nabla p\|, \quad | H - H_0 | / | H_0 |$',
                   color=color2, fontsize=LABEL_SIZE)
    ax2.plot(force_trace, label=r'$\|J \times B - \nabla p \|^2$',
             color=color2, lw=LINE_WIDTH)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
    # Set y-limits for better visibility
    ax2.set_ylim(0.5 * min(min(force_trace), 0.1 * max(helicity_change)),
                 2 * max(max(force_trace), max(helicity_change)))
    ax2.set_yscale('log')

    ax2.plot(helicity_change, label=r'$| H - H_0 |$',
             color='darkgray', linestyle='--', lw=LINE_WIDTH)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=LEGEND_SIZE)
    # ax1.grid(which="major", linestyle="-", color=color1, linewidth=0.5)
    ax2.grid(which="both", linestyle="--", linewidth=0.5)
    fig1.tight_layout()

# %%
    
    ###
    # Save stuff
    ###
    print("Saving to hdf5...")

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
        f.create_dataset("dts", data=jnp.array(dts))
        f.create_dataset("total_time", data=jnp.array([time2 - time0]))
        f.create_dataset("time_setup", data=jnp.array([time1 - time0]))
        f.create_dataset("time_solve", data=jnp.array([time2 - time1]))
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
    # replace defaults with user input
    for k, v in params.items():
        if k in CONFIG:
            CONFIG[k] = v
        else:
            print(f"Unknown parameter '{k}' - ignoring.")

    run(CONFIG)


if __name__ == "__main__":
    main()