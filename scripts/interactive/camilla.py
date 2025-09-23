# %%
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Dopri5, Kvaerno3, ODETerm, PIDController, SaveAt, diffeqsolve

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Nonlinearities import CrossProductProjection
from mrx.Plotting import get_1d_grids, get_2d_grids

jax.config.update("jax_enable_x64", True)

# %%
π = jnp.pi
p = 3
q = 3*p
ns = (6, 6, 6)
ps = (3, 3, 3)
types = ("clamped", "clamped", "clamped")


def F(x):
    return x


# %%

n_mu = 1
m_mu = 1
mu = π * (n_mu**2 + m_mu**2)**0.5
s = 1e4


def B(p):
    x, y, z = F(p)
    return mu * jnp.array([
        n_mu / (n_mu**2 + m_mu**2) *
        jnp.sin(m_mu * π * x) * jnp.cos(n_mu * π * y),
        - m_mu / (n_mu**2 + m_mu**2) *
        jnp.cos(m_mu * π * x) * jnp.sin(n_mu * π * y),
        jnp.sin(m_mu * π * x) * jnp.sin(n_mu * π * y)
    ])


# %%
_x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(F, zeta=0.5, nx=64)
_x_1d, _y_1d, (_y1_1d, _y2_1d, _y3_1d), (_x1_1d, _x2_1d,
                                         _x3_1d) = get_1d_grids(F, zeta=0.5, chi=0.5, nx=128)

# %%
# Set up finite element spaces
bcs = ('dirichlet', 'dirichlet', 'dirichlet')

Seq = DeRhamSequence(ns, ps, q, types, F, polar=False)

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
B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B))
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

dt = 1e-3
eta = 0.00

# %%
gamma = 0


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

    def is_not_converged(x):
        B, B_guess, _, _ = x
        delta = (B - B_guess) @ M2 @ (B - B_guess)
        return delta > tol**2

    def update(x):
        B, B_guess, _, _ = x
        B_new, J, u = implicit_update(B_guess, B, dt, eta)
        return B, B_new, J, u

    B_hat_1, J_hat, u_hat = implicit_update(
        B_hat, B_hat, dt, eta)
    x = (B_hat, B_hat_1, J_hat, u_hat)
    x = jax.lax.while_loop(is_not_converged, update, x)
    B_hat, B_hat_1, J_hat, u_hat = x
    return B_hat_1, J_hat, u_hat, dt


# %%
for i in range(200):
    # B_hat, J_hat, u_hat = update(B_hat)
    B_hat, J_hat, u_hat = picard_loop(B_hat, dt=dt, eta=0.0, tol=1e-9)
    A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
    u_trace.append((u_hat @ M2 @ u_hat)**0.5)
    E_trace.append(B_hat @ M2 @ B_hat / 2)
    H_trace.append(A_hat @ M12 @ (B_hat + B_harm_hat))
    dvg_trace.append(((dvg @ B_hat) @ M3 @ dvg @ B_hat)**0.5)
    if i % 100 == 0:
        print(f"Iteration {i}, u norm: {u_trace[-1]}")

# %%
# pressure computation

J_hat = weak_curl @ B_hat
JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, B_hat))
p_hat = jnp.linalg.pinv(laplace_3) @ M3 @ dvg @ JxB_hat

p_h = DiscreteFunction(p_hat, Seq.Λ3, Seq.E3_0.matrix())
p_h_xyz = jax.jit(lambda x: Pushforward(p_h, F, 3)(x))

# %%
J_h = DiscreteFunction(J_hat, Seq.Λ1, Seq.E1_0.matrix())
J_h_xyz = Pushforward(J_h, F, 1)

B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
B_h_xyz = Pushforward(B_h, F, 2)
# %%

plt.contourf(_x1, _x2, jax.vmap(p_h_xyz)(_x).reshape(
    _x1.shape[0], _x2.shape[0]), levels=30)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$p$ at $z=0.5$')
plt.show()
# %%
plt.contourf(_y1, _y2, jnp.linalg.norm(jax.vmap(J_h_xyz)(_x), axis=-1).reshape(
    _y1.shape[0], _y2.shape[0]), levels=30)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$\|J\|$ at $z=0.5$')
plt.show()
# %%
plt.contourf(_y1, _y2, jnp.linalg.norm(jax.vmap(B_h_xyz)(_x), axis=-1).reshape(
    _y1.shape[0], _y2.shape[0]), levels=30)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$\|B\|$ at $z=0.5$')
plt.show()

# %%


@jax.jit
def vector_field(t, x, args):
    DFx = jax.jacfwd(F)(x)
    norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
    return B_h(x) / (norm + 1e-9)


# %%
key = jax.random.PRNGKey(123)
x0s = jax.random.uniform(key, (100, 3), minval=0.05, maxval=0.95)

t1 = 1_000.0
n_saves = 10_000
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(ts=jnp.linspace(0, t1, n_saves))
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

trajectories = jax.vmap(lambda x0: diffeqsolve(term, solver,
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
FIG_SIZE_SQUARE = (8, 8)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
# Font size for the legend (not directly used here, but good to keep)
LEGEND_SIZE = 16
LINE_WIDTH = 2.5        # Width of the plot lines (not directly used here)
# ---------------------------------


colors = [
    "black",
    "purple",
    "teal",
    "orange",
    # "grey",
    # "blue",
    # "red",
    # "pink",
    # "green",
    # "gold"
]

# %%
plt.figure(figsize=FIG_SIZE_SQUARE)

for i, t in enumerate(trajectories):
    current_color = colors[i % len(colors)]  # Cycle through the defined colors
    plt.scatter(t[:, 0], t[:, 2], s=1,
                alpha=jnp.exp(-(t[:, 1] - 0.5)**2/0.0001),
                color=current_color)

# plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
plt.xlabel(r'$x$', fontsize=LABEL_SIZE)
plt.ylabel(r'$z$', fontsize=LABEL_SIZE)

# Set tick parameters
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# plt.savefig('poincare_solovev_logical.png', bbox_inches='tight', dpi=800)


# %%

fig = plt.figure(figsize=FIG_SIZE_SQUARE)
ax = fig.add_subplot(projection='3d')

for i, t in enumerate(trajectories[1:4]):
    current_color = colors[i % len(colors)]  # Cycle through the defined colors
    ax.plot(t[:, 0], t[:, 1], t[:, 2],
            # alpha=jnp.exp(-(t[:, 1] - 0.5)**2/0.01),
            color=current_color,
            alpha=0.8)

# plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
ax.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
ax.set_zlabel(r'$y$', fontsize=LABEL_SIZE)

# Set tick parameters
ax.tick_params(axis='x', labelsize=TICK_SIZE)
ax.tick_params(axis='y', labelsize=TICK_SIZE)
ax.tick_params(axis='z', labelsize=TICK_SIZE)

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

fig1.savefig('hopf_force.pdf', bbox_inches='tight')
# %%
print(f"B squared norm: {B_hat @ M2 @ B_hat}")
print(f"B_harmonic squared norm: {B_harm_hat @ M2 @ B_harm_hat}")
# %%
