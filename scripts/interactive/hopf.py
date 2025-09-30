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


π = jnp.pi
p = 3
q = 2*p
ns = (4, 4, 10)
ps = (3, 3, 3)
types = ("clamped", "clamped", "clamped")


def F(x):
    r, χ, z = x
    return jnp.array([r * 8 - 4, χ * 8 - 4, z * 20 - 10])


# %%
s = 1
ω1 = 3
ω2 = 2


def B(p):
    x, y, z = F(p)
    rsq = (x**2 + y**2 + z**2)
    return 4 * jnp.sqrt(s) / (π * (1 + rsq)**3 * (ω1**2 + ω2**2)**0.5) * jnp.array([
        2 * ω2 * y - 2 * ω1 * x * z,
        - 2 * ω2 * x - 2 * ω1 * y * z,
        ω1 * (x**2 + y**2 - z**2 - 1)
    ])


# %%
_x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(
    F, cut_value=0.5, nx=64)
_x_1d, _y_1d, (_y1_1d, _y2_1d, _y3_1d), (_x1_1d, _x2_1d,
                                         _x3_1d) = get_1d_grids(F, zeta=0.5, chi=0.5, nx=128)

# %%
Seq = DeRhamSequence(ns, ps, q, types, F, polar=False)

# %%
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

curlcurl = jnp.linalg.solve(M1, Seq.assemble_curlcurl_0())
graddiv = - jnp.linalg.solve(M2, Seq.assemble_divdiv_0())

laplace_0 = Seq.assemble_gradgrad_0()                        # dim ker = 0
laplace_1 = Seq.assemble_curlcurl_0() - M1 @ grad @ weak_dvg  # dim ker = 0 (no voids)
laplace_2 = M2 @ curl @ weak_curl + \
    Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)
laplace_3 = - M3 @ dvg @ weak_grad  # dim ker = 1 (constants)

M12 = Seq.assemble_M12_0()

# %%
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

dt0 = 10
dt_max = 1_000
eta = 0.00
# %%
gamma = 0


@jax.jit
def implicit_update(B_hat_guess, B_hat_0, dt, eta):
    B_hat_star = (B_hat_guess + B_hat_0) / 2
    J_hat = weak_curl @ B_hat_star
    H_hat = jnp.linalg.solve(M1, M12 @ B_hat_star)
    JxH_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
    u_hat = P_Leray @ JxH_hat
    for _ in range(gamma):
        u_hat = jnp.linalg.inv(M2 + laplace_2) @ M2 @ u_hat
    u_norm = (u_hat @ M2 @ u_hat)**0.5
    dt = jnp.minimum(dt0 / u_norm, dt_max)
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
n_iters = 20_000

for i in range(n_iters):
    # B_hat, J_hat, u_hat = update(B_hat)
    B_hat, J_hat, u_hat = picard_loop(B_hat, dt=dt0, eta=0.0, tol=1e-9)
    # u_stoch = u_stoch - 1e-2 * dt * u_stoch \
    # + jnp.sqrt(dt) * (u_hat @ M2 @ u_hat) * jax.random.normal(key, (M2.shape[0],))
    A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
    u_trace.append((u_hat @ M2 @ u_hat)**0.5)
    E_trace.append(B_hat @ M2 @ B_hat / 2)
    H_trace.append(A_hat @ M12 @ (B_hat + B_harm_hat))
    dvg_trace.append(((dvg @ B_hat) @ M3 @ dvg @ B_hat)**0.5)
    if i % 100 == 0:
        print(f"Iteration {i}, u norm: {u_trace[-1]}")

# %%
J_h = DiscreteFunction(J_hat, Seq.Λ1, Seq.E1_0.matrix())
J_h_xyz = Pushforward(J_h, F, 1)

B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
B_h_xyz = Pushforward(B_h, F, 2)
# %%
plt.contourf(_y1, _y2, jnp.linalg.norm(jax.vmap(B_h_xyz)(_x), axis=-1).reshape(
    _y1.shape[0], _y2.shape[0]), levels=30)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$\|B\|$ at $z=0.5$')
plt.show()

# %%
B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
B_h_xyz = Pushforward(B_h, F, 2)


@jax.jit
def vector_field(t, x, args):
    DFx = jax.jacfwd(F)(x)
    norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
    return B_h(x) / (norm + 1e-9)

# %%


t1 = 10_000.0
n_saves = 10_000
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(ts=jnp.linspace(0, t1, n_saves))
stepsize_controller = PIDController(rtol=1e-7, atol=1e-7)

n_loop = 5
n_batch = 10
key = jax.random.PRNGKey(123)
x0s = jnp.vstack(
    (jnp.linspace(0.05, 0.95, n_loop * n_batch),
     jnp.ones(n_loop * n_batch) * 0.5,
     jnp.ones(n_loop * n_batch) * 0.5)
)

x0s = x0s.T.reshape(n_batch, n_loop, 3)

trajectories = []
for x0 in x0s:
    trajectories.append(jax.vmap(lambda x0: diffeqsolve(term, solver,
                                                        t0=0, t1=t1, dt0=None,
                                                        y0=x0,
                                                        max_steps=2**17,
                                                        saveat=saveat, stepsize_controller=stepsize_controller).ys)(x0))

trajectories = jnp.array(trajectories).reshape(n_batch * n_loop, n_saves, 3)
trajectories.shape

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

n_cols = trajectories.shape[0]
cm = plt.cm.plasma
vals = jnp.linspace(0, 1, n_cols)

# Interleave from start and end
order = jnp.ravel(jnp.column_stack(
    [jnp.arange(n_cols//2), n_cols-1-jnp.arange(n_cols//2)]))
if n_cols % 2 == 1:
    order = jnp.append(order, n_cols//2)

colors = cm(vals[order])

# %%


def trajectory_plane_intersections(trajectories, plane_val=0.5, axis=1):
    """
    Vectorized + jittable intersection with plane x_axis = plane_val.

    Parameters
    ----------
    trajectories : array (N, T, D)
    plane_val    : float
    axis         : int, which coordinate axis (default=1 for x_2).

    Returns
    -------
    intersections : array (N, T-1, D)
        Intersection points for each segment. Non-crossings are filled with NaN.
    mask : bool array (N, T-1)
        True if the corresponding segment contains an intersection.
    """
    x = trajectories[..., axis]                           # (N, T)
    diff = x - plane_val

    # crossings: sign change or exact hit
    mask = (diff[..., :-1] * diff[..., 1:] <= 0)

    # interpolation fraction t in [0,1]
    denom = diff[..., :-1] - diff[..., 1:]
    t = jnp.where(mask, diff[..., :-1] / denom, jnp.nan)  # (N, T-1)

    # shape to broadcast into (N, T-1, 1)
    t = t[..., None]

    # segment start + t * (segment end - start)
    intersections = trajectories[:, :-1, :] + t * \
        (trajectories[:, 1:, :] - trajectories[:, :-1, :])

    return intersections, mask


def collapse_to_ragged(intersections, mask):
    """
    Collapse dense representation into ragged Python lists.
    """
    N = intersections.shape[0]
    result = []
    for n in range(N):
        pts = intersections[n][mask[n]]
        result.append(jnp.array(pts))
    return result


# %%
intersections, mask = trajectory_plane_intersections(
    physical_trajectories, plane_val=0.5, axis=1)

# intersections_ragged = collapse_to_ragged(intersections, mask)
# %%
plt.figure(figsize=FIG_SIZE_SQUARE)

for i, t in enumerate(intersections):
    current_color = colors[i % len(colors)]  # Cycle through the defined colors
    plt.scatter(t[:, 0], t[:, 2], s=1,
                color=current_color)

# plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
plt.xlabel(r'$x$', fontsize=LABEL_SIZE)
plt.ylabel(r'$z$', fontsize=LABEL_SIZE)

# Set tick parameters
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# plt.savefig('poincare_hopf.png', bbox_inches='tight', dpi=800)


# %%

fig = plt.figure(figsize=FIG_SIZE_SQUARE)
ax = fig.add_subplot(projection='3d')

for i, t in enumerate(trajectories[4:5]):
    current_color = colors[i % len(colors)]  # Cycle through the defined colors
    ax.plot(t[:, 0], t[:, 1], t[:, 2],
            # alpha=jnp.exp(-(t[:, 1] - 0.5)**2/0.01),
            color=current_color,
            alpha=0.5)

# plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
ax.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
ax.set_zlabel(r'$y$', fontsize=LABEL_SIZE)

# Set tick parameters
ax.tick_params(axis='x', labelsize=TICK_SIZE)
ax.tick_params(axis='y', labelsize=TICK_SIZE)
ax.tick_params(axis='z', labelsize=TICK_SIZE)

plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# plt.savefig('trajectories_hopf.png', bbox_inches='tight', dpi=800)


# %% Figure 1: Energy and Force

fig1, ax2 = plt.subplots(figsize=FIG_SIZE)
ax1 = ax2.twinx()
# Plot Energy on the left y-axis (ax1)
color1 = 'purple'
ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$',
               color=color1, fontsize=LABEL_SIZE)
ax1.plot(jnp.array(E_trace),
         label=r'$\frac{1}{2} \| B \|^2$', color=color1, linestyle='-.', lw=LINE_WIDTH)
# ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size
helicity_change = jnp.abs(
    jnp.array(jnp.array(H_trace) - H_trace[0]))
# Plot Force on the right y-axis (ax2)
color2 = 'black'
ax2.set_ylabel(r'$\|J \times B - \nabla p\|, \quad | H - H_0 |$',
               color=color2, fontsize=LABEL_SIZE)
ax2.plot(u_trace, label=r'$\|J \times B - \nabla p \|^2$',
         color=color2, lw=LINE_WIDTH)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
# Set y-limits for better visibility
ax2.set_ylim(0.5 * min(min(u_trace), 0.1 * max(helicity_change)),
             2 * max(max(u_trace), max(helicity_change)))
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
print(f"B squared norm: {B_hat @ M2 @ B_hat}")
print(f"B_harmonic squared norm: {B_harm_hat @ M2 @ B_harm_hat}")
# %%
