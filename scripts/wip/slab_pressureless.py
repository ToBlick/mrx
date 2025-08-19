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

p = 3
q = 10
ns = (8, 8, 8)
ps = (3, 3, 3)
types = ("clamped", "clamped", "clamped")

def F(x):
    return x


Λ0, Λ1, Λ2, Λ3 = [
    DifferentialForm(i, ns, ps, types) for i in range(0, 4)
]  # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, q)

E0, E1, E2, E3 = [
    LazyBoundaryOperator(Λ, ('dirichlet', 'dirichlet', 'dirichlet')).M for Λ in [Λ0, Λ1, Λ2, Λ3]
]

P0, P1, P2, P3 = [
    Projector(Λ, Q, F, E=E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]

M1, M2, M3 = [
    LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ1, Λ2, Λ3], [E1, E2, E3])
]

D1 = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
D2 = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M

CC = LazyDoubleCurlMatrix(Λ1, Q, F, E1).M
U, S, Vh = jnp.linalg.svd(CC)
S_inv = jnp.where(S > 1e-12, 1/S, 0)
CC_pinv = Vh.T @ jnp.diag(S_inv) @ U.T

M21 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M

L = D2 @ jnp.linalg.solve(M2, D2.T)

# %%

ɛ = 1e-6
nx = 64
_x1 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x2 = jnp.zeros(1) / 2
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * 1, 3)


# %%
gamma = 5/3

def A(r):
    n = 2
    m = 1
    x, y, z = r
    h = x**2 * (1 - x)**2 * y**2 * (1 - y)**2 * z**2 * (1 - z)**2 * 1e3
    Ax = (n / (jnp.sqrt(n**2 + m**2))) * jnp.sin(jnp.pi * m * x) * jnp.cos(jnp.pi * n * y)
    Ay = ((-m) / (jnp.sqrt(n**2 + m**2))) * jnp.cos(jnp.pi * m * x) * jnp.sin(jnp.pi * n * y)
    Az = jnp.sin(jnp.pi * m * x) * jnp.sin(jnp.pi * n * y)
    return jnp.array([Ax, Ay, Az]) * h

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

P_uxH = CrossProductProjection(Λ1, Λ2, Λ1, Q, F, En=E1, Em=E2, Ek=E1)
P_JxH = CrossProductProjection(Λ2, Λ1, Λ1, Q, F, En=E2, Em=E1, Ek=E1)

@jax.jit
def step(B_guess, B_prev, dt):
    B = (B_guess + B_prev) / 2
    H = jnp.linalg.solve(M1, M21.T @ B)
    J = jnp.linalg.solve(M1, D1.T @ B)
    v = jnp.linalg.solve(M2, P_JxH(J, H))
    q = jnp.linalg.solve(L, D2 @ v)
    u = v - jnp.linalg.solve(M2, D2.T @ q)
    E = jnp.linalg.solve(M1, P_uxH(u, H))
    delta_B = jnp.linalg.solve(M2, D1 @ E)
    return B_prev + dt * delta_B, u

def update(B_hat, dt, tol):
    B_guess, _ = step(B_hat, B_hat, dt)
    err = 1
    while err > tol:
        B_next, u = step(B_guess, B_hat, dt)
        err = ((B_next - B_guess) @ M2 @
               (B_next - B_guess) / (B_guess @ M2 @ B_guess))
        # print("Error: ", err)
        B_guess = B_next
    return B_next, u
# %%
dt = 1e-2

B_hat = jnp.linalg.solve(M2, D1 @ jnp.linalg.solve(M1, P1(A)))
H_hat = jnp.linalg.solve(M1, M21.T @ B_hat)
J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
A_hat = CC_pinv @ D1.T @ B_hat

E_mag_0 = B_hat @ M2 @ B_hat / 2
Helicity_0 = B_hat @ M21 @ A_hat
print("magnetic energy: ", E_mag_0)
print("helicity: ", Helicity_0)
print("div B: ", (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
# %%
B_next, u_hat = update(B_hat, dt, 1e-9)
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
B_hat = jnp.linalg.solve(M2, D1 @ jnp.linalg.solve(M1, P1(A)))
A_hat = CC_pinv @ D1.T @ B_hat

Energy_trace = [ B_hat @ M2 @ B_hat / 2 ]
Helicity_trace = [ B_hat @ M21 @ A_hat ]
force_trace = [ ]
divergence_trace = [D2 @ B_hat @ jnp.linalg.solve(M3, D2 @ B_hat)]

# %%
dt = 1e-2
for iter in range(300):
    B_hat, u_hat = update(B_hat, dt, 1e-14)
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

fig1.savefig('slab_vacuum_energy_force_pressureproj.pdf', bbox_inches='tight')


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
fig2.savefig('slab_vacuum_divergence_helicity_pressureproj.pdf', bbox_inches='tight')
# %%
H_hat = jnp.linalg.solve(M1, M21.T @ B_hat)
J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
v_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
p_hat = jnp.linalg.solve(L, D2 @ v_hat)
grad_p_hat = jnp.linalg.solve(M2, D2.T @ p_hat)
B_h = Pushforward(DiscreteFunction(B_hat, Λ2, E2), F, 2)
J_h = Pushforward(DiscreteFunction(J_hat, Λ1, E1), F, 1)
u_h = Pushforward(DiscreteFunction(u_hat, Λ2, E2), F, 2)
p_h = Pushforward(DiscreteFunction(p_hat, Λ3, E3), F, 3)

# %%
print("u norm: ", u_hat @ M2 @ u_hat)
print("grad_p norm: ", grad_p_hat @ M2 @ grad_p_hat)
# %%
ɛ = 1e-6
nx = 64
_x1 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x2 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x3 = jnp.ones(1)/2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * nx * 1, 3)
# %%
plt.contourf(_x1, _x2, jax.vmap(p_h)(_x).reshape(nx, nx), levels=10)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
# %%
