# %%
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Dopri5, Kvaerno3, ODETerm, PIDController, SaveAt, diffeqsolve

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.IterativeSolvers import picard_solver
from mrx.Nonlinearities import CrossProductProjection
from mrx.Plotting import get_1d_grids, get_2d_grids, poincare_plot

jax.config.update("jax_enable_x64", True)

# %%
π = jnp.pi
p = 3
q = 2*p
ns = (8, 8, 8)
ps = (3, 3, 3)
types = ("clamped", "clamped", "clamped")


def F(x):
    return x


# %%

n_mu = 1
m_mu = 1
mu = π * (n_mu**2 + m_mu**2)**0.5
s = 1e4


def B_beltrami(p):
    x, y, z = F(p)
    return mu * jnp.array([
        n_mu / (n_mu**2 + m_mu**2) *
        jnp.sin(m_mu * π * x) * jnp.cos(n_mu * π * y),
        - m_mu / (n_mu**2 + m_mu**2) *
        jnp.cos(m_mu * π * x) * jnp.sin(n_mu * π * y),
        jnp.sin(m_mu * π * x) * jnp.sin(n_mu * π * y)
    ])


# %%
_x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(
    F, cut_value=0.5, nx=64)
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

P_Leray = jnp.eye(M2.shape[0]) + \
    weak_grad @ jnp.linalg.pinv(laplace_3) @ M3 @ dvg

# %%

dt_max = 2e-3
dt0 = 1e-3

# Set up inital condition
B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_beltrami))
# One step of resisitive relaxation to get J x n = 0 on ∂Ω
B_hat = jnp.linalg.solve(
    jnp.eye(M2.shape[0]) + 1e-2 * curl @ weak_curl, B_hat)
A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
B_harm_hat = B_hat - curl @ A_hat
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
    u_hat = P_Leray @ JxH_hat
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


# %%
force_err = 0
max_iter = 100
solver_tol = 1e-9
gamma = 0
eta = 0
# %%
n_steps = 15_000

# %%
for i in range(n_steps):
    x = jnp.concatenate(
        (B_hat, B_hat, jnp.ones(1) * dt, jnp.ones(1) * force_err))
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
    if iters[-1] == max_iter and picard_err > 1e-9:
        print(
            f"Picard solver did not converge in {max_iter} iterations (err={picard_err:.2e})")
        break
    if i % 100 == 0:
        print(f"Iteration {i}, u norm: {force_trace[-1]}")


# %%
plt.plot(jnp.array(E_trace))
plt.yscale('log')
# %%
# %%
plt.plot(jnp.array(force_trace))
plt.yscale('log')
# %%
B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())
B_h_xyz = Pushforward(B_h, F, 2)
# %%


@jax.jit
def vector_field(t, x, args):
    DFx = jax.jacfwd(F)(x)
    norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
    return B_h(x) / (norm + 1e-9)


# %%
n_loop = 5
n_batch = 10

x0s = jnp.vstack(
    (jnp.linspace(0.05, 0.95, n_loop * n_batch),
     jnp.ones(n_loop * n_batch)/2,
     jnp.ones(n_loop * n_batch)/2)
)

n_cols = x0s.shape[1]
cm = plt.cm.nipy_spectral
vals = jnp.linspace(0, 1, n_cols + 2)[:-1]

# shuffle them
order = jax.random.permutation(jax.random.PRNGKey(0), n_cols)
colors = cm(vals[order])

for (i, c) in enumerate(colors):
    plt.plot(i, i, color=c, marker='o', markersize=5)

x0s = x0s.T.reshape(n_batch, n_loop, 3)

poincare_plot("/scratch/tblickhan/mrx/script_outputs/beltrami/", vector_field, F, x0s,
              n_loop, n_batch, colors,
              name="final",
              plane_val=0.5, axis=1,
              final_time=2_000, n_saves=100_000, cylindrical=False,
              r_tol=1e-8, a_tol=1e-8)
# %%
# Figure 2: Energy and Force

energy_trace = E_trace
helicity_trace = H_trace


FIG_SIZE = (12, 6)
SQUARE_FIG_SIZE = (8, 8)
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 16
LINE_WIDTH = 2.5
LEGEND_SIZE = 16

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
    jnp.array(jnp.array(helicity_trace) - helicity_trace[0]))
# Plot Force on the right y-axis (ax2)
color2 = 'black'
ax2.set_ylabel(r'$\|J \times B - \nabla p\|, \quad | H - H_0 |$',
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
plt.savefig("/scratch/tblickhan/mrx/script_outputs/beltrami/energy_force.pdf",
            bbox_inches='tight')
# %%
plt.close('all')
# %%
