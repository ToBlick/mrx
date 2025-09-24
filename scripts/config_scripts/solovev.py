# %%
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.BoundaryFitting import get_lcfs_F
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction
from mrx.InputOutput import parse_args, unique_id
from mrx.IterativeSolvers import picard_solver
from mrx.Nonlinearities import CrossProductProjection

jax.config.update("jax_enable_x64", True)

outdir = "script_outputs/solovev/"
os.makedirs(outdir, exist_ok=True)


###
# Default configuration
###
CONFIG = {

    "run_name": "",  # Name for the run. If empty, a hash will be created

    ###
    # Parameters describing the domain and initial psi
    ###
    "R_0": 3.0,  # Major radius
    "k_0": 1.5,  # Elongation parameter
    "q_0": 1.5,  # Safety factor (?)
    "F_0": 0.5,  # toroidal field strength
    "a_R": 1.2,  # LCFS is at R_0 + a_R
    # If True, use circular cross section instead of Solovev
    "circular_cross_section": False,

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
    "gamma": 0,                  # Regularization, u = (-Δ)⁻ᵞ (J x B - grad p)
    "eps": 1e-2,                 # Regularization for the initial condition
    "dt": 1e-3,                  # Time step
    "dt_max": 1.0,               # max. time step
    "n_steps": 20_000,           # max. Number of time steps
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

    R0 = CONFIG["R_0"]
    π = jnp.pi
    k0 = CONFIG["k_0"]
    q0 = CONFIG["q_0"]
    F0 = CONFIG["F_0"]
    aR = CONFIG["a_R"]

    p_map = CONFIG["p_chi"]
    n_map = CONFIG["n_chi"]
    q_map = 2 * p_map

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

    if CONFIG["circular_cross_section"]:
        def F(x):
            r, χ, z = x
            return jnp.ravel(jnp.array(
                [(R0 + aR * r * jnp.cos(2 * π * χ)) * jnp.cos(2 * π * z),
                 -(R0 + aR * r * jnp.cos(2 * π * χ)) * jnp.sin(2 * π * z),
                 aR * r * jnp.sin(2 * π * χ)]))
    else:
        F = get_lcfs_F(n_map, p_map, q_map, R0, k0, q0, aR)

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

    def B_xyz(p):
        x, y, z = F(p)
        R = (x**2 + y**2)**0.5
        phi = jnp.arctan2(y, x) / (2 * π)

        BR = - R * z / (R0**2 * k0 * q0)
        Bz = (k0**2 * (R**2 - R0**2) + 2*z**2) / (2 * R0**2 * k0 * q0)
        BPhi = F0 / R

        Bx = BR * jnp.cos(2 * π * phi) - BPhi * jnp.sin(2 * π * phi)
        By = BR * jnp.sin(2 * π * phi) + BPhi * jnp.cos(2 * π * phi)

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

    # jax.lax.scan ? jax.lax.fori?
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
        B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
        J_hat = weak_curl @ B_hat
        J_h = DiscreteFunction(J_hat, Seq.Λ1, Seq.E1_0.matrix())

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
# # Poincare plot
# B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
# B_h_xyz = Pushforward(B_h, F, 2)


# @jax.jit
# def vector_field(t, p, args):
#     r, χ, z = p
#     r = jnp.clip(r, 1e-6, 1)
#     χ = χ % 1.0
#     z = z % 1.0
#     x = jnp.array([r, χ, z])
#     DFx = jax.jacfwd(F)(x)
#     norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
#     return B_h(x) / (norm + 1e-9)


# # %%
# t1 = 2_000.0
# n_saves = 20_000
# term = diffrax.ODETerm(vector_field)
# solver = diffrax.Dopri5()
# saveat = diffrax.SaveAt(ts=jnp.linspace(0, t1, n_saves))
# stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-7)

# n_loop = 5
# n_batch = 5
# key = jax.random.PRNGKey(123)
# x0s = jax.random.uniform(key, (n_loop, n_batch, 3), minval=0.05, maxval=0.95)
# x0s = x0s.at[:, :, 2].set(0.0)

# # trajectories = jax.vmap(lambda x0: diffeqsolve(term, solver,
# #                             t0=0, t1=t1, dt0=None,
# #                             y0=x0,
# #                             max_steps=2**14,
# #                             saveat=saveat, stepsize_controller=stepsize_controller).ys)(x0s)

# trajectories = []
# for x0 in x0s:
#     trajectories.append(jax.vmap(lambda x0: diffrax.diffeqsolve(term, solver,
#                                                                 t0=0, t1=t1, dt0=None,
#                                                                 y0=x0,
#                                                                 max_steps=2**15,
#                                                                 saveat=saveat, stepsize_controller=stepsize_controller).ys)(x0))

# trajectories = jnp.array(trajectories) % 1
# trajectories = trajectories.reshape(n_batch * n_loop, n_saves, 3)
# trajectories.shape

# physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
# physical_trajectories = physical_trajectories.reshape(
#     trajectories.shape[0], trajectories.shape[1], 3)
# # %%
# cmap = plt.cm.plasma

# colors = [cmap(v) for v in np.linspace(0, 1, 10)]

# crossings = []
# for i, t in enumerate(trajectories):
#     cross = []
#     for j in range(t.shape[0] - 1):
#         if (t[j, 2] - 0.5) * (t[j+1, 2] - 0.5) < 0:
#             # determine intersection by linear interpolation
#             alpha = (0.5 - t[j, 2]) / (t[j+1, 2] - t[j, 2])
#             intersection = t[j] + alpha * (t[j+1] - t[j])
#             cross.append(intersection)
#     crossings.append(jnp.array(cross))

# physical_crossings = [jax.vmap(F)(c) for c in crossings]
# # %%
# # --- PLOT SETTINGS ---
# FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
# TITLE_SIZE = 20         # Font size for the plot title
# LABEL_SIZE = 20         # Font size for x and y axis labels
# TICK_SIZE = 16          # Font size for x and y tick labels
# # Font size for the legend (not directly used here, but good to keep)
# LEGEND_SIZE = 16
# LINE_WIDTH = 2.5        # Width of the plot lines (not directly used here)
# # ---------------------------------
# # %%
# # Create a figure with two subplots next to each other
# fig = plt.figure(figsize=FIG_SIZE)
# gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

# # Left subplot: x < -2
# ax1 = fig.add_subplot(gs[0])
# # Right subplot: x > 2
# ax2 = fig.add_subplot(gs[1], sharey=ax1)

# # Turn off tick labels on right of left plot and left of right plot
# ax1.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.yaxis.tick_right()
# ax1.yaxis.tick_left()
# ax2.tick_params(labelleft=False)

# for i, t in enumerate(physical_crossings):
#     x = np.array(t[:, 0])
#     z = np.array(t[:, 2])

#     mask_left = x < -R0 + 1.1
#     mask_right = x > R0 - 1.1

#     current_color = colors[i % len(colors)]  # Cycle through the defined colors

#     if np.any(mask_left):
#         ax1.scatter(x[mask_left], z[mask_left], s=0.5,
#                     alpha=1, color=current_color)

#     if np.any(mask_right):
#         ax2.scatter(x[mask_right], z[mask_right], s=0.5,
#                     alpha=1, color=current_color)

# # Set labels and titles with specified font sizes
# ax1.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
# ax2.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
# ax1.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
# # fig.suptitle(r'Field line intersections', fontsize=TITLE_SIZE)

# # Set x limits for both subplots
# ax1.set_xlim(-R0 - 1.1, -R0 + 1.1)
# ax2.set_xlim(R0 - 1.1, R0 + 1.1)

# # Set tick parameters for both axes
# ax1.tick_params(axis='x', labelsize=TICK_SIZE)
# ax1.tick_params(axis='y', labelsize=TICK_SIZE)
# ax2.tick_params(axis='x', labelsize=TICK_SIZE)
# # Although labelleft=False, still good to set size for potential future use
# ax2.tick_params(axis='y', labelsize=TICK_SIZE)

# # Adjust layout
# # Adjust rect to prevent suptitle overlap
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# # fig.savefig('poincare_solovev_physical.png', bbox_inches='tight', dpi=800)

# # %%
# plt.figure(figsize=FIG_SIZE)

# for i, t in enumerate(crossings):
#     current_color = colors[i % len(colors)]  # Cycle through the defined colors
#     plt.scatter(t[:, 0], t[:, 1] % 1, s=0.1,
#                 # alpha=np.exp(-np.array(t[:, 2] % 1)**2 / 0.01),
#                 color=current_color)

# # plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
# plt.xlabel(r'$r$', fontsize=LABEL_SIZE)
# plt.ylabel(r'$\chi$', fontsize=LABEL_SIZE)

# # Set tick parameters
# plt.xticks(fontsize=TICK_SIZE)
# plt.yticks(fontsize=TICK_SIZE)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# # %% Figure 1: Energy and Force
# fig1, ax2 = plt.subplots(figsize=FIG_SIZE)
# ax1 = ax2.twinx()

# # Plot Energy on the left y-axis (ax1)
# color1 = 'purple'
# ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
# ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$', color=color1, fontsize=LABEL_SIZE)
# ax1.plot(jnp.array(E_trace),
#          label=r'$\frac{1}{2} \| B \|^2$', color=color1, linestyle='-.', lw=LINE_WIDTH)
# # ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
# ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
# ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size

# relative_helicity_change = jnp.abs(jnp.array(jnp.array(H_trace) - H_trace[0]))
# # Plot Force on the right y-axis (ax2)
# color2 = 'black'
# ax2.set_ylabel(r'$\|J \times B - \nabla p\|, \quad | H - H_0 |$',
#                color=color2, fontsize=LABEL_SIZE)
# ax2.plot(u_trace, label=r'$\|J \times B - \nabla p \|^2$',
#          color=color2, lw=LINE_WIDTH)
# ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
# # Set y-limits for better visibility
# ax2.set_ylim(0.5 * min(min(u_trace), min(relative_helicity_change[10:])),
#              2 * max(max(u_trace), max(relative_helicity_change)))
# ax2.set_yscale('log')


# ax2.plot(relative_helicity_change, label=r'$| H - H_0 |$',
#          color='darkgray', linestyle='--', lw=LINE_WIDTH)
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2,
#            loc='upper right', fontsize=LEGEND_SIZE)
# # ax1.grid(which="major", linestyle="-", color=color1, linewidth=0.5)
# ax2.grid(which="both", linestyle="--", linewidth=0.5)
# fig1.tight_layout()
# plt.show()

# # fig1.savefig('solovev_force.pdf', bbox_inches='tight')
# # %%
# print(f"B squared norm: {B_hat @ M2 @ B_hat}")
# print(f"B_harmonic squared norm: {B_harm_hat @ M2 @ B_harm_hat}")
# # %%
# H_h = DiscreteFunction(jnp.linalg.solve(
#     M1, M12 @ B_harm_hat), Seq.Λ1, Seq.E1_0.matrix())
# # %%

# %%
