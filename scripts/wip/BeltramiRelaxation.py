"""
Full MHD Evolution for Beltrami Fields
"""

import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

from mrx.DeRhamSequence import DeRhamSequence
from mrx.Nonlinearities import CrossProductProjection

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
os.makedirs("script_outputs", exist_ok=True)


# Set up exact solution components (for now mode numbers (1,1))
A_0 = 1.0
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

# %%
def F(x): # Identity maps
    return x

#Set up
p = 3
q = 3*p
ns = (3, 3, 2)  
ps = (2, 2, 1)  
types = ("periodic", "periodic", "periodic")
bcs = ("periodic", "periodic", "periodic")

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
B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_exact))


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
    B_hat, J_hat, u_hat = picard_loop(B_hat, dt=0.001, eta=0.0, tol=1e-12)  # no dissipation
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


# %%