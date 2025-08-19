# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.BoundaryConditions import LazyBoundaryOperator
from mrx.DifferentialForms import (
    DifferentialForm,
    DiscreteFunction,
    Pullback,
    Pushforward,
)
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleCurlMatrix,
    LazyMassMatrix,
    LazyProjectionMatrix,
)
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import grad, inv33, jacobian_determinant, l2_product

jax.config.update("jax_enable_x64", True)
# %%
eps = 1/5
a = 1
π = jnp.pi
γ = 5/3
R0 = a / eps

p = 3
q = 10
ns = (6, 6, 4)
ps = (3, 3, 3)
types = ("clamped", "periodic", "periodic")


def _X(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))


def _Z(r, χ):
    return jnp.ones(1) * a * r * jnp.sin(2 * π * χ)


def _Y(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array([_X(r, χ) * jnp.cos(2 * π * z),
                                -_Y(r, χ) * jnp.sin(2 * π * z),
                                _Z(r, χ)]))

Λ0, Λ1, Λ2, Λ3 = [
    DifferentialForm(i, ns, ps, types) for i in range(0, 4)
]  # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, q)
ξ = get_xi(_X, _Y, Λ0, Q)[0]

E0, E1, E2, E3 = [
    LazyExtractionOperator(Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]
]

P0, P1, P2, P3 = [
    Projector(Λ, Q, F, E=E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]

M1, M2, M3 = [
    LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ1, Λ2, Λ3], [E1, E2, E3])
]

D1 = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
D2 = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M

M21 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M

CC_pinv = jnp.linalg.pinv(LazyDoubleCurlMatrix(Λ1, Q, F, E1).M)
L_pinv = jnp.linalg.pinv(D2 @ jnp.linalg.solve(M2, D2.T))


# %%
ɛ = 1e-6
nx = 64
_x1 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x2 = jnp.zeros(1) / 2
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * 1, 3)


# %%
Bz0 = 1.0
mu_0 = 1.0
p_avg = 0.1 * Bz0**2 / 2
p_max = 140/81 * p_avg
alpha_z = 1.1
alpha_p = 2 * p_avg / Bz0**2
beta = p_avg / (Bz0**2 / 2)

def B_phys(x):
    r, χ, z = x
    Br = 0
    Bχ = Bz0 * (alpha_p/9 * (35 * r**6 - 40 * r**12 + 14 * r**18) 
                + alpha_z/15 * (30 * r**2 - 20 * (alpha_z * 2 + 1) * r**4 
                                + 45 * alpha_z * r**6 - 12 * alpha_z * r**8))**0.5
    Bz = Bz0 * (1 - 2 * alpha_z * r**2 + alpha_z * r**4)
    return jnp.array([Br, Bχ, Bz])

def B_analytic(x):
    Br, Bχ, Bz = B_phys(x)
    
    DF = jax.jacfwd(F)
    G = DF(x).T @ DF(x)
    
    return jnp.array([Br * G[0,0]**0.5,
                      Bχ * G[1,1]**0.5, 
                      Bz * G[2,2]**0.5])
    
def p_analytic(x):
    r, χ, z = x
    return p_max * (1 - r**6)**3 * jnp.ones(1)

# %%
class CrossProductProjection:
    """
    Given bases Λn, Λm, Λk, constructs an operator to evaluate
    (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
    and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
    with coordinate transformation F.
    """

    def __init__(self, Λn, Λm, Λk, Q, F=None, En=None, Em=None, Ek=None):
        """
        Given bases Λn, Λm, Λk, constructs an operator to evaluate
        (w, u) -> ∫ (wₕ × uₕ) · Λn[i] dx for all i, where Λn[i] is the i-th basis function of Λn
        and wₕ = ∑ w[i] Λm[i], uₕ = ∑ u[i] Λk[i] are discrete functions
        with coordinate transformation F.

        Args:
            Λn: Basis for n-forms (n can be 1 or 2)
            Λm: Basis for m-forms (m can be 1 or 2)
            Λk: Basis for k-forms (k can be 1 or 2)
            Q: Quadrature rule for numerical integration
            F (callable, optional): Coordinate transformation function.
                                    Defaults to identity mapping.
            Ek, Ev, En (array, optional): Extraction operator matrix for Λn, Λm, Λk.
                                Defaults to identity matrix.
        """
        self.Λn = Λn
        self.Λm = Λm
        self.Λk = Λk
        self.Q = Q
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        self.En = En if En is not None else None
        self.Em = Em if Em is not None else None
        self.Ek = Ek if Ek is not None else None
        self.M = En

    def __call__(self, w, u):
        """
        evaluates ∫ (wₕ × uₕ) · Λn[i] dx for all i 
        and collects the values in a vector.

        Args:
            w (array): m-form dofs
            u (array): k-form dofs

        Returns:
            array: ∫ (wₕ × uₕ) · Λn[i] dx for all i 
        """
        return self.M @ self.projection(w, u)

    def projection(self, w, u):

        DF = jax.jacfwd(self.F)

        w_h = DiscreteFunction(w, self.Λm, self.Em)
        u_h = DiscreteFunction(u, self.Λk, self.Ek)

        if self.Λn.k == 1 and self.Λm.k == 2 and self.Λk.k == 1:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return jnp.cross(G @ w_h(x), u_h(x))
        elif self.Λn.k == 2 and self.Λm.k == 1 and self.Λk.k == 1:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return G @ jnp.cross(w_h(x), u_h(x))
        elif self.Λn.k == 2 and self.Λm.k == 2 and self.Λk.k == 1:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return G @ jnp.cross(G @ w_h(x), u_h(x))
        elif self.Λn.k == 1 and self.Λm.k == 2 and self.Λk.k == 2:
            def v(x):
                G = DF(x).T @ DF(x)
                return inv33(G) @ jnp.cross(w_h(x), u_h(x))
        elif self.Λn.k == 2 and self.Λm.k == 1 and self.Λk.k == 2:
            def v(x):
                G = DF(x).T @ DF(x) / jnp.linalg.det(DF(x))
                return G @ jnp.cross(w_h(x), G @ u_h(x))
        else:
            raise ValueError("Not yet implemented")

        # Compute projections
        vjk = jax.vmap(v)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(self.Λn, (0, None)), (None, 0))(
            self.Q.x, self.Λn.ns)  # n x n_q x d
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j->i", Λijk, vjk, wj)


# %%

# P_uxH = CrossProductProjection(Λ1, Λ2, Λ1, Q, F, En=E1, Em=E2, Ek=E1)
# P_JxH = CrossProductProjection(Λ2, Λ1, Λ1, Q, F, En=E2, Em=E1, Ek=E1)

P_uxH = CrossProductProjection(Λ1, Λ2, Λ2, Q, F, En=E1, Em=E2, Ek=E2)
P_JxH = CrossProductProjection(Λ2, Λ1, Λ2, Q, F, En=E2, Em=E1, Ek=E2)

@jax.jit
def step(B_guess, B_prev, dt):
    B = (B_guess + B_prev) / 2
    # H = jnp.linalg.solve(M1, M21.T @ B)
    H = B
    J = jnp.linalg.solve(M1, D1.T @ B)
    v = jnp.linalg.solve(M2, P_JxH(J, H))
    q = - L_pinv @ D2 @ v
    u = v + jnp.linalg.solve(M2, D2.T @ q)
    E = jnp.linalg.solve(M1, P_uxH(u, H))
    delta_B = jnp.linalg.solve(M2, D1 @ E)
    return B_prev + dt * delta_B, u, q

def update(B_hat, dt, tol):
    B_guess, _, _ = step(B_hat, B_hat, dt)
    err = 1
    while err > tol:
        B_next, u, q = step(B_guess, B_hat, dt)
        err = ((B_next - B_guess) @ M2 @
               (B_next - B_guess) / (B_guess @ M2 @ B_guess))
        # print("Error: ", err)
        B_guess = B_next
    return B_next, u, q
# %%
dt = 1e-3

B_hat = jnp.linalg.solve(M2, P2(B_analytic))
B_hat = B_hat - jnp.linalg.solve(M2, D2.T @ L_pinv @ D2 @ B_hat)
H_hat = jnp.linalg.solve(M1, M21.T @ B_hat)
J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
A_hat = CC_pinv @ D1.T @ B_hat

E_mag_0 = B_hat @ M2 @ B_hat / 2
Helicity_0 = B_hat @ M21 @ A_hat
print("magnetic energy: ", E_mag_0)
print("helicity: ", Helicity_0)
print("div B: ", (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
# %%
B_next, u_hat, q_hat = update(B_hat, dt, 1e-9)
# %%
p_hat = jnp.linalg.solve(M3, P3(p_analytic))

(p_hat - q_hat) @ M3 @ (p_hat - q_hat) / (p_hat @ M3 @ p_hat)
# %%
A_next = CC_pinv @ D1.T @ B_next
curlA_next = jnp.linalg.solve(M2, D1 @ A_next)
H_next = jnp.linalg.solve(M1, M21.T @ B_next)
E_mag_1 = B_next @ M2 @ B_next / 2
Helicity_1 = B_next @ M21 @ A_next

print("magnetic energy: ", E_mag_1)
print("helicity: ", Helicity_1)
print("div B: ", (D2 @ B_next) @ jnp.linalg.solve(M3, D2 @ B_next))
print("force squared: ", u_hat @ M2 @ u_hat)

# %%
print("Change in magnetic energy: ", (E_mag_1 - E_mag_0))
print("Change in helicity: ", (Helicity_1 - Helicity_0) )

# %%
B_hat = jnp.linalg.solve(M2, P2(B_analytic))
B_hat = B_hat - jnp.linalg.solve(M2, D2.T @ L_pinv @ D2 @ B_hat)
A_hat = CC_pinv @ D1.T @ B_hat

Energy_trace = [ B_hat @ M2 @ B_hat / 2 ]
Helicity_trace = [ B_hat @ M21 @ A_hat ]
force_trace = [ ]
divergence_trace = [D2 @ B_hat @ jnp.linalg.solve(M3, D2 @ B_hat)]

# %%
dt = 1e-2
for iter in range(1000):
    B_hat, u_hat, q_hat = update(B_hat, dt, 1e-14)
    A_hat = CC_pinv @ D1.T @ B_hat
    Energy_trace.append(B_hat @ M2 @ B_hat / 2)
    Helicity_trace.append(B_hat @ M21 @ A_hat)
    force_trace.append(u_hat @ M2 @ u_hat)
    divergence_trace.append(D2 @ B_hat @ jnp.linalg.solve(M3, D2 @ B_hat))
    if iter % 10 == 0:
        print(
            f"Iteration {iter}, Energy: {Energy_trace[-1]}, Helicity: {Helicity_trace[-1]/Helicity_trace[0] - 1}, Force: {force_trace[-1]}, Divergence: {divergence_trace[-1]}")
# %%
# %%
# --- PLOT SETTINGS FOR SLIDES ---
# You can easily adjust these values
FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
LEGEND_SIZE = 16        # Font size for the legend
LINE_WIDTH = 2.5        # Width of the plot lines
# ---------------------------------


# %% Figure 1: Energy and Force
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

# Plot Energy on the left y-axis (ax1)
color1 = 'purple'
ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2, \quad \pi \, (A, B)$', color=color1, fontsize=LABEL_SIZE)
ax1.plot(jnp.array(Energy_trace), label=r'$\frac{1}{2} \| B \|^2$', color=color1, lw=LINE_WIDTH)
ax1.plot(jnp.pi * jnp.array(Helicity_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE) # Set x-tick size

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot Force on the right y-axis (ax2)
color2 = 'black'
ax2.set_ylabel(r'$\|J \times B \|^2$', color=color2, fontsize=LABEL_SIZE)
ax2.plot(force_trace, label=r'$\|J \times B \|^2$', color=color2, lw=LINE_WIDTH)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
ax2.set_ylim(0.5 * min(force_trace), 2 * max(force_trace))  # Set y-limits for better visibility
ax2.set_yscale('log')

# Add grid and title
# ax1.grid(True, which="both", linestyle='--', color=color1, alpha=0.6)
# ax2.grid(True, which="both", linestyle='--', alpha=0.6)
# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=LEGEND_SIZE)

fig1.tight_layout()
plt.show()

fig1.savefig('torus_rfp_force.pdf', bbox_inches='tight')


# %% Figure 2: Divergence and Helicity
fig2, ax3 = plt.subplots(figsize=FIG_SIZE)

# Plot Divergence on the left y-axis (ax3)
color3 = 'purple'
ax3.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
ax3.set_ylabel(r'$\| \nabla \cdot B \|^2$', color=color3, fontsize=LABEL_SIZE)
ax3.plot(jnp.array(divergence_trace), label='Divergence', color=color3, lw=LINE_WIDTH)
ax3.tick_params(axis='y', labelcolor=color3, labelsize=TICK_SIZE)
ax3.tick_params(axis='x', labelsize=TICK_SIZE) # Set x-tick size
offset_text3 = ax3.yaxis.get_offset_text()
offset_text3.set_size(TICK_SIZE)
# Create a second y-axis
ax4 = ax3.twinx()

# Calculate and plot relative Helicity change on the right y-axis (ax4)
relative_helicity_change = jnp.array(jnp.array(Helicity_trace) - Helicity_trace[0])
color4 = 'black'
ax4.set_ylabel(r'$(B, A) - (B_0, A_0)$', color=color4, fontsize=LABEL_SIZE)
ax4.plot(relative_helicity_change, label='Relative Helicity Change', color=color4, lw=LINE_WIDTH)
ax4.tick_params(axis='y', labelcolor=color4, labelsize=TICK_SIZE)
offset_text4 = ax4.yaxis.get_offset_text()
offset_text4.set_size(TICK_SIZE)
# Add grid and title
ax3.grid(True, which="both", linestyle='--', alpha=0.6)

# Combine legends from both axes
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
# ax4.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=LEGEND_SIZE)

fig2.tight_layout()
plt.show()
fig2.savefig('torus_rfp_invariants.pdf', bbox_inches='tight')

# %%
J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
B_h = Pushforward(DiscreteFunction(B_hat, Λ2, E2), F, 2)
J_h = Pushforward(DiscreteFunction(J_hat, Λ1, E1), F, 1)
p_h = Pushforward(DiscreteFunction(p_hat, Λ3, E3), F, 3)
q_h = Pushforward(DiscreteFunction(q_hat, Λ3, E3), F, 3)
# %%
tol = 1e-6
nx = 64
_nx = 16
_x1 = jnp.linspace(tol, 1 - tol, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * nx * 1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
_y3 = _y[:, 2].reshape(nx, nx)

plt.contourf(_y1, _y3, jax.vmap(q_h)(_x).reshape(nx, nx), levels=25, cmap='plasma')
plt.colorbar()
# indicate origin
plt.scatter(_y1[0, 0], _y3[0, 0], color='k', s=2,)
plt.scatter(_y1[0, 0] + 1/3, _y3[0, 0], color='k', s=2,)
plt.scatter(_y1[0, 0] - 1/3, _y3[0, 0], color='k', s=2,)

# # %%
# color1 = 'purple'
# color2 = 'black'
# color3 = 'grey'
# fig, ax = plt.subplots(figsize=(8, 6)) #is normally 12, 6

# ax.set_xlabel(r'$r$', fontsize=LABEL_SIZE)
# ax.plot(_x1, 1/beta * (jax.vmap(q_h)(_x) - q_h(_x[-1])), label=r'$p / \beta$', color='grey', linestyle="-", lw=LINE_WIDTH)
# ax.plot(_x1, jax.vmap(B_h)(_x)[:, 1], label=r'$B_\chi$', color='purple', linestyle="--", lw=LINE_WIDTH)
# ax.plot(_x1, jax.vmap(B_h)(_x)[:, 2], label=r'$B_z$', color='purple', linestyle="-", lw=LINE_WIDTH)
# ax.plot(_x1, jax.vmap(J_h)(_x)[:, 1], label=r'$J_\chi$', color='black', linestyle="--", lw=LINE_WIDTH)
# ax.plot(_x1, jax.vmap(J_h)(_x)[:, 2], label=r'$J_z$', color='black', linestyle="-", lw=LINE_WIDTH)

# ax.tick_params(axis='y', labelsize=TICK_SIZE)
# ax.tick_params(axis='x', labelsize=TICK_SIZE)
# ax.legend(fontsize=LEGEND_SIZE)
# ax.grid(True, which="both", linestyle='--', alpha=0.6)

# plt.tight_layout()
# plt.show()
# # %%
# fig.savefig('torus_rfp_fields.pdf', bbox_inches='tight')
# %%


ns = (6, 1, 1)
ps = (3, 0, 0)
types = ("clamped", "constant", "constant")

Λ = DifferentialForm(0, (6, 1, 1), (3, 0, 0), ("clamped", "constant", "constant"))
# %%
nx = 128
_x1 = jnp.linspace(0, 1, nx)
_x2 = jnp.zeros(1) / 2
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx, 3)

plt.plot(_x1, jax.vmap(Λ[0])(_x))
plt.plot(_x1, jax.vmap(Λ[1])(_x))
plt.plot(_x1, jax.vmap(Λ[2])(_x))
plt.plot(_x1, jax.vmap(Λ[3])(_x))
plt.plot(_x1, jax.vmap(Λ[4])(_x))
plt.plot(_x1, jax.vmap(Λ[5])(_x))
plt.xlabel(r'$x$')
plt.ylabel(r'$\phi_i(x)$')
# %%
fig2, ax3 = plt.subplots(figsize=FIG_SIZE)
colors = ['purple', 'black', 'grey', 'orange', 'teal', 'violet']
# Plot Divergence on the left y-axis (ax3)
color3 = 'purple'
ax3.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax3.set_ylabel(r'$\phi_i(x)$', fontsize=LABEL_SIZE)
for i, c in enumerate(colors):
    ax3.plot(_x1, jax.vmap(Λ[i])(_x), label=f'$\phi_{i}(x)$', lw=LINE_WIDTH, color=c)
ax3.tick_params(axis='y', labelsize=TICK_SIZE)
ax3.tick_params(axis='x', labelsize=TICK_SIZE) # Set x-tick size
ax3.grid(True, which="both", linestyle='--', alpha=0.6)
plt.savefig('spline_basis.pdf', bbox_inches='tight')
# %%
