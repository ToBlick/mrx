# %%
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from mrx.BoundaryFitting import solovev_lcfs_fit
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pushforward
from mrx.Nonlinearities import CrossProductProjection
from mrx.Plotting import get_1d_grids, get_2d_grids

jax.config.update("jax_enable_x64", True)

# %%
R0 = 3.0
μ0 = 1.0
π = jnp.pi
k0 = 1.5
q0 = 1.5
F0 = 0.5
aR = 1.0

###
# ψ(R, Z) =  (¼ k₀² (R² - R₀²)² + R²Z² ) / (2 R₀² k₀ q₀)
###

# p_map = 3
# n_map = 8
# q_map = 2 * p_map

# a_hat = solovev_lcfs_fit(n_map, p_map, q_map, R0, a=0.6, k0=k0, q0=q0)

# Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
#                          ("periodic", "constant", "constant"))
# a_h = DiscreteFunction(a_hat, Λ_map)


# def a(χ):
#     """Radius as a function of chi."""
#     return a_h(jnp.array([χ, 0, 0]))[0]

def a(χ):
    return 1.0


_x = jnp.linspace(0, 1, 1024)
plt.plot(_x, jax.vmap(a)(_x))

# %%
p = 3
q = 2*p
ns = (8, 8, 1)
ps = (3, 3, 0)
types = ("clamped", "periodic", "constant")


def _R(r, χ):
    return jnp.ones(1) * (R0 + a(χ) * r * jnp.cos(2 * π * χ))


def _Z(r, χ):
    return jnp.ones(1) * a(χ) * r * jnp.sin(2 * π * χ)


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array(
        [_R(r, χ) * jnp.cos(2 * π * z),
         -_R(r, χ) * jnp.sin(2 * π * z),
         _Z(r, χ)]))

# %%


def psi(p):
    x, y, z = F(p)
    R = (x**2 + y**2)**0.5
    Z = z

    def _psi(R, Z):
        return (k0**2/4*(R**2 - R0**2)**2 + R**2*Z**2) / (2 * R0**2 * k0 * q0)
    return _psi(R, Z) - _psi(R0 + a(0), 0)


# %%
_x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(F, zeta=0, nx=64)
_x_1d, _y_1d, (_y1_1d, _y2_1d, _y3_1d), (_x1_1d, _x2_1d,
                                         _x3_1d) = get_1d_grids(F, zeta=0, chi=0, nx=128)

# %%
plt.contourf(_y1, _y3, jax.vmap(psi)(_x).reshape(_y1.shape), levels=20)
vals = jax.vmap(psi)(_x_1d)
plt.plot(_y1_1d, vals - vals[0], 'k', label=r'$\psi(r, 0, 0)$')
plt.plot(_y1_1d, jnp.zeros_like(_y2_1d), ':k')
plt.axis('equal')
plt.legend()
plt.colorbar()
plt.xlabel("R")
plt.ylabel("Z")


# %%
# Set up finite element spaces
bcs = ('dirichlet', 'none', 'none')

Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

# %%

M0 = Seq.assemble_M0_0()
M1 = Seq.assemble_M1_0()
M2 = Seq.assemble_M2_0()
M3 = Seq.assemble_M3_0()

M1_dual = Seq.assemble_M1()

###
# Operators
###

grad = jnp.linalg.solve(M1, Seq.assemble_grad_0())
curl = jnp.linalg.solve(M2, Seq.assemble_curl_0())
dvg = jnp.linalg.solve(M3, Seq.assemble_dvg_0())
weak_grad = -jnp.linalg.solve(M2, Seq.assemble_dvg_0().T)
weak_curl = jnp.linalg.solve(M1, Seq.assemble_curl_0().T)
weak_dvg = -jnp.linalg.solve(M0, Seq.assemble_grad_0().T)

curlcurl = jnp.linalg.solve(M1, Seq.assemble_curlcurl_0())
graddiv = - jnp.linalg.solve(M2, Seq.assemble_divdiv_0())

laplace_0 = Seq.assemble_gradgrad_0()                        # dim ker = 0
laplace_1 = Seq.assemble_curlcurl_0() - M1 @ grad @ weak_dvg  # dim ker = 0 (no voids)
laplace_2 = M2 @ curl @ weak_curl + \
    Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)
laplace_3 = - M3 @ dvg @ weak_grad  # dim ker = 1 (constants)

# from H₀(div) to H(curl)
P1 = jnp.linalg.solve(Seq.assemble_M1(), Seq.assemble_P2().T)

M12 = Seq.assemble_M12_0()


# %%

def p_phys(x):
    return - (k0**2 + 1)/(R0**2 * k0 * q0) * psi(x) * jnp.ones(1)


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
# Set up inital condition
B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_xyz))
# One step of resisitive relaxation to get J x n = 0 on ∂Ω
B_hat = jnp.linalg.solve(jnp.eye(M2.shape[0]) + 1e-2 * curl @ weak_curl, B_hat)

A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
B_harm_hat = B_hat - curl @ A_hat
print(f"|div B_harm|^2: {(dvg @ B_harm_hat).T @ M3 @ dvg @ B_harm_hat}")
print(
    f"|curl B_harm|^2: {(weak_curl @ B_harm_hat) @ M1 @ (weak_curl @ B_harm_hat)}")

u_trace = []
E_trace = []
H_trace = []
dvg_trace = []

dt = 1.0
eta = 0.00

# %%
gamma = 1


@jax.jit
def update(B_hat):
    J_hat = weak_curl @ B_hat
    JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, B_hat))
    u_hat = P_Leray @ JxB_hat
    for _ in range(gamma):
        u_hat = jnp.linalg.inv(M2 + laplace_2) @ M2 @ u_hat
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
    for _ in range(gamma):
        u_hat = jnp.linalg.inv(M2 + laplace_2) @ M2 @ u_hat

    E_hat = jnp.linalg.solve(M1, P_uxH(u_hat, H_hat)) - eta * J_hat
    B_hat_1 = B_hat_0 + dt * curl @ E_hat
    return B_hat_1, J_hat, u_hat


def picard_loop(B_hat, dt, eta, tol):
    B_hat_0 = B_hat
    B_hat_guess = B_hat
    B_hat_1, J_hat, u_hat = implicit_update(
        B_hat_guess, B_hat_0, dt, eta)
    delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    while delta > tol**2:
        B_hat_guess = B_hat_1
        B_hat_1, J_hat, u_hat = implicit_update(
            B_hat_guess, B_hat_0, dt, eta)
        delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    return B_hat_1, J_hat, u_hat


# %%
for i in range(10_000):
    # B_hat, J_hat, u_hat = update(B_hat)
    B_hat, J_hat, u_hat = picard_loop(B_hat, dt=dt, eta=0.00, tol=1e-9)
    A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
    u_trace.append((u_hat @ M2 @ u_hat)**0.5)
    E_trace.append(B_hat @ M2 @ B_hat / 2)
    H_trace.append(A_hat @ M12 @ (B_hat + B_harm_hat))
    dvg_trace.append((dvg @ B_hat @ M3 @ dvg @ B_hat)**0.5)
    if i % 100 == 0:
        print(f"Iteration {i}, u norm: {u_trace[-1]}")

# %%
# Poincare plot
B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
B_h_xyz = Pushforward(B_h, F, 2)


@jax.jit
def vector_field(t, p, args):
    r, χ, z = p
    r = jnp.clip(r, 1e-6, 1)
    χ = χ % 1.0
    z = z % 1.0
    x = jnp.array([r, χ, z])
    DFx = jax.jacfwd(F)(x)
    norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
    return B_h(x) / (norm + 1e-9)


# %%
key = jax.random.PRNGKey(0)
x0s = jax.random.uniform(key, (20, 3), maxval=1, minval=0)
x0s = x0s.at[:, 0].set(x0s[:, 0]**0.5)  # Bias towards r=0

t1 = 2_000.0
n_saves = 20_000
term = diffrax.ODETerm(vector_field)
solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(ts=jnp.linspace(0, t1, n_saves))
stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

trajectories = jax.vmap(lambda x0: diffrax.diffeqsolve(term, solver,
                                                       t0=0, t1=t1, dt0=None,
                                                       y0=x0,
                                                       max_steps=2**14,
                                                       saveat=saveat, stepsize_controller=stepsize_controller).ys)(x0s)

# %%
physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
physical_trajectories = physical_trajectories.reshape(
    trajectories.shape[0], trajectories.shape[1], 3)
# %%

# --- PLOT SETTINGS FOR SLIDES ---
FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
# Font size for the legend (not directly used here, but good to keep)
LEGEND_SIZE = 16
LINE_WIDTH = 2.5        # Width of the plot lines (not directly used here)
# ---------------------------------
# %%
# Create a figure with two subplots next to each other
fig = plt.figure(figsize=FIG_SIZE)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

# Left subplot: x < -2
ax1 = fig.add_subplot(gs[0])
# Right subplot: x > 2
ax2 = fig.add_subplot(gs[1], sharey=ax1)

# Turn off tick labels on right of left plot and left of right plot
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()
ax1.yaxis.tick_left()
ax2.tick_params(labelleft=False)

cmap = plt.cm.plasma

colors = [cmap(v) for v in np.linspace(0, 1, 10)]

for i, t in enumerate(physical_trajectories):
    x = np.array(t[:, 0])
    z = np.array(t[:, 2])
    alpha = np.exp(-np.array(t[:, 1])**2 / 0.05)

    mask_left = x < -R0 + 1.1
    mask_right = x > R0 - 1.1

    current_color = colors[i % len(colors)]  # Cycle through the defined colors

    if np.any(mask_left):
        ax1.scatter(x[mask_left], z[mask_left], s=0.1,
                    alpha=alpha[mask_left], color=current_color)

    if np.any(mask_right):
        ax2.scatter(x[mask_right], z[mask_right], s=0.1,
                    alpha=alpha[mask_right], color=current_color)

# Set labels and titles with specified font sizes
ax1.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax2.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
# fig.suptitle(r'Field line intersections', fontsize=TITLE_SIZE)

# Set x limits for both subplots
ax1.set_xlim(-R0 - 1.1, -R0 + 1.1)
ax2.set_xlim(R0 - 1.1, R0 + 1.1)

# Set tick parameters for both axes
ax1.tick_params(axis='x', labelsize=TICK_SIZE)
ax1.tick_params(axis='y', labelsize=TICK_SIZE)
ax2.tick_params(axis='x', labelsize=TICK_SIZE)
# Although labelleft=False, still good to set size for potential future use
ax2.tick_params(axis='y', labelsize=TICK_SIZE)

# Adjust layout
# Adjust rect to prevent suptitle overlap
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# fig.savefig('poincare_solovev_physical.png', bbox_inches='tight', dpi=800)

# %%
plt.figure(figsize=FIG_SIZE)

for i, t in enumerate(trajectories):
    current_color = colors[i % len(colors)]  # Cycle through the defined colors
    plt.scatter(t[:, 0], t[:, 1] % 1, s=0.1,
                alpha=np.exp(-np.array(t[:, 2] % 1)**2 / 0.05),
                color=current_color)

# plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
plt.xlabel(r'$r$', fontsize=LABEL_SIZE)
plt.ylabel(r'$\chi$', fontsize=LABEL_SIZE)

# Set tick parameters
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# plt.savefig('poincare_solovev_logical.png', bbox_inches='tight', dpi=800)

# %% Figure 1: Energy and Force
fig1, ax2 = plt.subplots(figsize=FIG_SIZE)
ax1 = ax2.twinx()

# Plot Energy on the left y-axis (ax1)
color1 = 'purple'
ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$', color=color1, fontsize=LABEL_SIZE)
ax1.plot(jnp.array(E_trace),
         label=r'$\frac{1}{2} \| B \|^2$', color=color1, linestyle='-.', lw=LINE_WIDTH)
# ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size


# Plot Force on the right y-axis (ax2)
color2 = 'black'
ax2.set_ylabel(r'$\|J \times B - \nabla p\|, \quad | H - H_0 |$',
               color=color2, fontsize=LABEL_SIZE)
ax2.plot(u_trace, label=r'$\|J \times B - \nabla p \|^2$',
         color=color2, lw=LINE_WIDTH)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
# Set y-limits for better visibility
ax2.set_ylim(0.5 * min(u_trace), 2 * max(u_trace))
ax2.set_yscale('log')

relative_helicity_change = jnp.abs(jnp.array(jnp.array(H_trace) - H_trace[0]))
ax2.plot(relative_helicity_change, label=r'$| H - H_0 |$',
         color='darkgray', linestyle='--', lw=LINE_WIDTH)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           loc='upper right', fontsize=LEGEND_SIZE)
# ax1.grid(which="major", linestyle="-", color=color1, linewidth=0.5)
ax2.grid(which="both", linestyle="--", linewidth=0.5)
fig1.tight_layout()
plt.show()

# fig1.savefig('solovev_force.pdf', bbox_inches='tight')
# %%
print(f"B squared norm: {B_hat @ M2 @ B_hat}")
print(f"B_harmonic squared norm: {B_harm_hat @ M2 @ B_harm_hat}")
# %%
