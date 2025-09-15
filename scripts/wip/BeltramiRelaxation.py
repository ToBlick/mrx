"""
Full MHD Evolution for Beltrami Fields
"""

import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Nonlinearities import CrossProductProjection

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
os.makedirs("script_outputs", exist_ok=True)


# Set up exact solution components (for now mode numbers (1,1))
A_0 = 10000
m_mode = 1
n_mode = 1
mu_target = jnp.pi * jnp.sqrt(m_mode**2 + n_mode**2)

def B_exact(x: jnp.ndarray) -> jnp.ndarray:
    """Analytical Beltrami magnetic field components."""
    x_1, x_2, x_3 = x
    return jnp.array([
        ((A_0 * n_mode) / (jnp.sqrt(m_mode**2 + n_mode**2))) * jnp.sin(jnp.pi * m_mode * x_1) * jnp.cos(jnp.pi * n_mode * x_2),
        ((A_0 * m_mode * -1) / (jnp.sqrt(m_mode**2 + n_mode**2))) * jnp.cos(jnp.pi * m_mode * x_1) * jnp.sin(jnp.pi * n_mode * x_2),
        A_0*jnp.sin(jnp.pi * m_mode * x_1) * jnp.sin(jnp.pi * n_mode * x_2)
    ])



def eta(x: jnp.ndarray) -> float:
    x_1, x_2, x_3 = x
    return (x_1)**2 * (1 - x_1)**2 * (x_2)**2 * (1 - x_2)**2 * (x_3)**2 * (1 - x_3)**2



def mu(m:float, n:float) -> float:
    return jnp.pi * jnp.sqrt(m**2 + n**2)

def grad_eta(x: jnp.ndarray) -> jnp.ndarray:
    return jax.grad(lambda t: eta(t))(x)


def grad_eta_cross_B(x: jnp.ndarray) -> jnp.ndarray:
    B_x, B_y, B_z = B_exact(x)
    eta_x, eta_y, eta_z = grad_eta(x)
    return jnp.array([
        B_z*eta_y - B_y*eta_z,
        B_x*eta_z - B_z*eta_x,
        B_y*eta_x - B_x*eta_y
    ])


def B_local(x: jnp.ndarray) -> jnp.ndarray:
    """Creating an interesting initial condition"""
    x_1, x_2, x_3 = x
    return grad_eta_cross_B(x) + mu(1,1)*B_exact(x)*eta(x)

# %%
def F(x): # Identity maps
    return x

#Set up
p = 3
q = 3*p
ns = (8, 8, 8)  
ps = (1, 1, 1)  
types = ("periodic", "periodic", "periodic")
bcs = ("dirichlet", "dirichlet", "dirichlet")

# %%

# De Rham sequence
Seq = DeRhamSequence(ns, ps, q, types, F, polar=False)

# Mass matrices 
M0 = Seq.assemble_M0_0()
M1 = Seq.assemble_M1_0()
M2 = Seq.assemble_M2_0()
M3 = Seq.assemble_M3_0()

# Operators
grad = jnp.linalg.solve(M1, Seq.assemble_grad_0())
curl = jnp.linalg.solve(M2, Seq.assemble_curl_0())
dvg = jnp.linalg.solve(M3, Seq.assemble_dvg_0())

# Weak operators 
weak_grad = -jnp.linalg.solve(M2, Seq.assemble_dvg_0().T)
weak_curl = jnp.linalg.solve(M1, Seq.assemble_curl_0().T)
weak_dvg = -jnp.linalg.solve(M0, Seq.assemble_grad_0().T)

curlcurl = jnp.linalg.solve(M1, Seq.assemble_curlcurl_0())
graddiv = - jnp.linalg.solve(M2, Seq.assemble_divdiv_0())

# Laplacian operators
laplace_0 = Seq.assemble_gradgrad_0()  # dim ker = 0
laplace_1 = Seq.assemble_curlcurl_0() - M1 @ grad @ weak_dvg  # dim ker = 0 (no voids)
laplace_2 = M2 @ curl @ weak_curl + \
    Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)
laplace_3 = - M3 @ dvg @ weak_grad  # dim ker = 1 (constants)

P1 = jnp.linalg.solve(Seq.assemble_M1(), Seq.assemble_P2().T)

M12 = Seq.assemble_M12_0()


# %%
P_JxH = CrossProductProjection(
    Seq.Λ2, Seq.Λ1, Seq.Λ1, Seq.Q, Seq.F,
    En=Seq.E2_0, Em=Seq.E1_0, Ek=Seq.E1,
    Λn_ijk=Seq.Λ2_ijk, Λm_ijk=Seq.Λ1_ijk, Λk_ijk=Seq.Λ1_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
P_JxB = CrossProductProjection(
    Seq.Λ2, Seq.Λ1, Seq.Λ2, Seq.Q, Seq.F,
    En=Seq.E2_0, Em=Seq.E1_0, Ek=Seq.E2_0,
    Λn_ijk=Seq.Λ2_ijk, Λm_ijk=Seq.Λ1_ijk, Λk_ijk=Seq.Λ2_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
P_uxH = CrossProductProjection(
    Seq.Λ1, Seq.Λ2, Seq.Λ1, Seq.Q, Seq.F,
    En=Seq.E1_0, Em=Seq.E2_0, Ek=Seq.E1,
    Λn_ijk=Seq.Λ1_ijk, Λm_ijk=Seq.Λ2_ijk, Λk_ijk=Seq.Λ1_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
P_uxB = CrossProductProjection(
    Seq.Λ1, Seq.Λ2, Seq.Λ2, Seq.Q, Seq.F,
    En=Seq.E1_0, Em=Seq.E2_0, Ek=Seq.E2_0,
    Λn_ijk=Seq.Λ1_ijk, Λm_ijk=Seq.Λ2_ijk, Λk_ijk=Seq.Λ2_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
# %%
P_Leray = jnp.eye(M2.shape[0]) + \
    weak_grad @ jnp.linalg.pinv(laplace_3) @ M3 @ dvg



# %%


# Initial magnetic field 
B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_local))


# One step of resistive relaxation 
B_hat = jnp.linalg.solve(jnp.eye(M2.shape[0]) + 1e-2 * curl @ weak_curl, B_hat)


# A_hat = L_vec_pinv @ M1 @ weak_curl @ B_hat
A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
B_harm_hat = B_hat - curl @ A_hat
print(f"|div B_harm|^2: {(dvg @ B_harm_hat).T @ M3 @ dvg @ B_harm_hat}")
print(
    f"|curl B_harm|^2: {(weak_curl @ B_harm_hat) @ M1 @ (weak_curl @ B_harm_hat)}")

u_trace = []
E_trace = []
H_trace = []
dvg_trace = []

dt = 0.001
eta  = 0.0

@jax.jit
def update(B_hat):
    J_hat = weak_curl @ B_hat
    JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, B_hat))
    u_hat = P_Leray @ JxB_hat
    E_hat = jnp.linalg.solve(M1, P_uxB(u_hat, B_hat)) - eta * J_hat
    B_hat += dt * curl @ E_hat
    return B_hat, J_hat, u_hat


@jax.jit
def implicit_update(B_hat_guess, B_hat_0, dt, eta):
    B_hat_star = (B_hat_guess + B_hat_0) / 2
    J_hat = weak_curl @ B_hat_star

    H_hat = P1 @ B_hat_star
    JxH_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
    u_hat = P_Leray @ JxH_hat
    E_hat = jnp.linalg.solve(M1, P_uxH(u_hat, H_hat)) - eta * J_hat

    B_hat_1 = B_hat_0 + dt * curl @ E_hat
    return B_hat_1, J_hat, u_hat


def picard_loop(B_hat, dt, eta, tol):
    B_hat_0 = B_hat
    B_hat_guess = B_hat
    B_hat_1, J_hat, u_hat = implicit_update(
        B_hat_guess, B_hat_0, dt, eta)
    delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    while delta > tol:
        B_hat_guess = B_hat_1
        B_hat_1, J_hat, u_hat = implicit_update(
            B_hat_guess, B_hat_0, dt, eta)
        delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    return B_hat_1, J_hat, u_hat


# %%

for i in range(200):  # Fewer iterations 
    # B_hat, J_hat, u_hat = update(B_hat)
    B_hat, J_hat, u_hat = picard_loop(B_hat, dt=0.001, eta=0.0, tol=1e-13)  # no dissipation
    A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
    u_trace.append(u_hat @ M2 @ u_hat)
    E_trace.append(B_hat @ M2 @ B_hat / 2)
    H_trace.append(A_hat @ M12 @ (B_hat + B_harm_hat))
    dvg_trace.append(dvg @ B_hat @ M3 @ dvg @ B_hat)
    if i % 10 == 0:
        print(f"Iteration {i:3d}, u norm: {jnp.sqrt(u_trace[-1]):.2e}, energy: {E_trace[-1]:.6f}")

# %%
# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Velocity norm evolution
axes[0, 0].plot(jnp.sqrt(jnp.array(u_trace)), 'b-', linewidth=2)
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_ylabel("||u_h||")
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("Velocity Evolution")
axes[0, 0].grid(True, alpha=0.3)

# Magnetic energy evolution  
axes[0, 1].plot(E_trace, 'r-', linewidth=2)
axes[0, 1].set_xlabel("Iteration")
axes[0, 1].set_ylabel("½||B_h||²")
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Magnetic Energy")
axes[0, 1].grid(True, alpha=0.3)

# Helicity evolution
axes[1, 0].plot(jnp.array(H_trace), 'g-', linewidth=2)
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].set_ylabel("Helicity - H(0)")
axes[1, 0].set_title("Helicity Evolution")
axes[1, 0].grid(True, alpha=0.3)

# Divergence error
axes[1, 1].plot(jnp.sqrt(jnp.array(dvg_trace)), 'm-', linewidth=2)
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("||div B_h||")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("Divergence Error")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('script_outputs/beltrami_mhd_evolution.png', dpi=150, bbox_inches='tight')
plt.show()




# # %%
# # Poincare plot

# B_full = Seq.E2_0.matrix().T @ B_hat
# B_h = DiscreteFunction(B_full, Seq.Λ2)
# B_h_xyz = Pushforward(B_h, F, 2)

# def rk4(x0, f, dt):
#     k1 = f(x0)
#     k2 = f(x0 + dt/2 * k1)
#     k3 = f(x0 + dt/2 * k2)
#     k4 = f(x0 + dt * k3)
#     return x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


# @partial(jax.jit, static_argnames=['B_h', 'n_steps'])
# def fieldline(x0, B_h, dt, n_steps):
#     def step(current_x, _):
#         next_x = rk4(current_x, B_h, dt)
#         next_x = next_x.at[0].set(jnp.clip(next_x[0], 0, 1))
#         next_x = next_x.at[1].set(jnp.mod(next_x[1], 1))
#         next_x = next_x.at[2].set(jnp.mod(next_x[2], 1))
#         return next_x, next_x
#     final_x, xs = jax.lax.scan(step, x0, None, length=n_steps)
#     return xs


# def vector_field(x):
#     DFx = jax.jacfwd(F)(x)
#     norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5 / (jnp.linalg.det(DFx) + 1e-12)
#     return B_h(x) / (jnp.linalg.det(DFx) * norm + 1e-12)


# # %%
# x0 = jnp.linspace(5e-2, 1-5e-2, 20)
# x0_1 = jnp.array([x0, jnp.zeros_like(x0), jnp.ones_like(x0)/7]).T
# x0_2 = jnp.array([x0, jnp.ones_like(x0)/4, jnp.zeros_like(x0)]).T
# x0_3 = jnp.array([x0, jnp.ones_like(x0)/2, jnp.zeros_like(x0)]).T
# x0_4 = jnp.array([x0, 3*jnp.ones_like(x0)/4, jnp.zeros_like(x0)]).T
# # x0 = jnp.concatenate([x0_1, x0_2, x0_3, x0_4], axis=0)
# x0 = x0_1
# # %%
# trajectories = jax.vmap(lambda x: fieldline(
#     x, vector_field, 0.1, 2000))(x0)
# # %%
# physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
# physical_trajectories = physical_trajectories.reshape(
#     trajectories.shape[0], trajectories.shape[1], 3)
# # %%

# import matplotlib.gridspec as gridspec

# # Dummy data for demonstration if 'trajectories' and 'physical_trajectories' are not defined
# # In a real scenario, these would come from your calculations.
# if 'trajectories' not in locals():
#     num_points = 500
#     trajectories = [np.random.rand(num_points, 3) * 10 for _ in range(3)]
#     trajectories[0][:, 0] = trajectories[0][:, 0] - 5 # Example x values
#     trajectories[1][:, 0] = trajectories[1][:, 0] # Example x values
#     trajectories[2][:, 0] = trajectories[2][:, 0] + 5 # Example x values


# if 'physical_trajectories' not in locals():
#     R0 = 5 # Example value for R0
#     num_physical_points = 1000
#     physical_trajectories = []
#     # Create some dummy data that crosses the R0 threshold
#     t1 = np.random.rand(num_physical_points // 2, 3) * 2
#     t1[:, 0] = t1[:, 0] - (R0 + 1) # Left side
#     physical_trajectories.append(t1)

#     t2 = np.random.rand(num_physical_points // 2, 3) * 2
#     t2[:, 0] = t2[:, 0] + (R0 + 1) # Right side
#     physical_trajectories.append(t2)


# # --- PLOT SETTINGS FOR SLIDES ---
# FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
# TITLE_SIZE = 20         # Font size for the plot title
# LABEL_SIZE = 20         # Font size for x and y axis labels
# TICK_SIZE = 16          # Font size for x and y tick labels
# LEGEND_SIZE = 16        # Font size for the legend (not directly used here, but good to keep)
# LINE_WIDTH = 2.5        # Width of the plot lines (not directly used here)
# # ---------------------------------

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

# # Define primary and secondary colors for the scatter plots
# primary_color = 'purple'
# secondary_color = 'teal'
# tertiary_color = 'orange'
# quaternary_color = 'black'

# colors = [primary_color, secondary_color, tertiary_color, quaternary_color]

# for i, t in enumerate(physical_trajectories):
#     x = np.array(t[:, 0])
#     z = np.array(t[:, 2])
#     alpha = np.exp(-np.array(t[:, 1])**2 / 0.02)

#     mask_left = x < -R0 + 1
#     mask_right = x > R0 - 1

#     current_color = colors[i % len(colors)] # Cycle through the defined colors

#     if np.any(mask_left):
#         ax1.scatter(x[mask_left], z[mask_left], s=0.1, alpha=alpha[mask_left], color=current_color)

#     if np.any(mask_right):
#         ax2.scatter(x[mask_right], z[mask_right], s=0.1, alpha=alpha[mask_right], color=current_color)

# # Set labels and titles with specified font sizes
# ax1.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
# ax2.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
# ax1.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
# # fig.suptitle(r'Field line intersections', fontsize=TITLE_SIZE)

# # Set x limits for both subplots
# ax1.set_xlim(-R0 - 0.9, -R0 + 0.9)
# ax2.set_xlim(R0 - 0.9, R0 + 0.9)

# # Set tick parameters for both axes
# ax1.tick_params(axis='x', labelsize=TICK_SIZE)
# ax1.tick_params(axis='y', labelsize=TICK_SIZE)
# ax2.tick_params(axis='x', labelsize=TICK_SIZE)
# ax2.tick_params(axis='y', labelsize=TICK_SIZE) # Although labelleft=False, still good to set size for potential future use

# # Adjust layout
# fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap

# # fig.savefig('poincare_solovev_physical.png', bbox_inches='tight', dpi=800)

# # %%
# plt.figure(figsize=FIG_SIZE)

# for i, t in enumerate(trajectories):
#     current_color = colors[i % len(colors)] # Cycle through the defined colors
#     plt.scatter(t[:, 0], t[:, 1], s=0.1, alpha=jnp.exp(-t[:, 2]**2/0.02), color=current_color)

# # plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
# plt.xlabel(r'$r$', fontsize=LABEL_SIZE)
# plt.ylabel(r'$\chi$', fontsize=LABEL_SIZE)

# # Set tick parameters
# plt.xticks(fontsize=TICK_SIZE)
# plt.yticks(fontsize=TICK_SIZE)

# plt.tight_layout() # Adjust layout to prevent labels from overlapping

# # plt.savefig('poincare_solovev_logical.png', bbox_inches='tight', dpi=800)

# # %% Figure 1: Energy and Force
# fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

# # Plot Energy on the left y-axis (ax1)
# color1 = 'purple'
# ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
# ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$', color=color1, fontsize=LABEL_SIZE)
# ax1.plot(jnp.array(E_trace), label=r'$\frac{1}{2} \| B \|^2$', color=color1, lw=LINE_WIDTH)
# # ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
# ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
# ax1.tick_params(axis='x', labelsize=TICK_SIZE) # Set x-tick size

# # Create a second y-axis that shares the same x-axis
# ax2 = ax1.twinx()

# # Plot Force on the right y-axis (ax2)
# color2 = 'black'
# ax2.set_ylabel(r'$\|J \times B - \nabla p\|^2, \quad | H - H_0 |$', color=color2, fontsize=LABEL_SIZE)
# ax2.plot(u_trace, label=r'$\|J \times B - \nabla p \|^2$', color=color2, lw=LINE_WIDTH)
# ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
# ax2.set_ylim(0.5 * min(u_trace), 2 * max(u_trace))  # Set y-limits for better visibility
# ax2.set_yscale('log')

# relative_helicity_change = jnp.abs(jnp.array(jnp.array(H_trace) - H_trace[0]))
# ax2.plot(relative_helicity_change, label=r'$| H - H_0 |$', color='darkgray', linestyle='--', lw=LINE_WIDTH)
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=LEGEND_SIZE)

# fig1.tight_layout()
# plt.show()

# # fig1.savefig('solovev_force.pdf', bbox_inches='tight')
# # %%