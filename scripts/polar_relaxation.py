# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mrx.SplineBases import SplineBasis, DerivativeSpline, TensorBasis
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback, Pushforward
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector, CurlProjection
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix, LazyStiffnessMatrix
from mrx.Utils import div, curl, inv33, jacobian, grad

jax.config.update("jax_enable_x64", True)

# %%
types = ('clamped', 'periodic', 'constant')
ns = (8, 8, 1)
ps = (3, 3, 0)
Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in range(4)]
Q = QuadratureRule(Λ0, 10)
###
# Mapping definition
###
a = 1
R0 = 3.0
Y0 = 0.0

def θ(x):
    r, χ, z = x
    return 2 * jnp.atan(jnp.sqrt((1 + a*r/R0)/(1 - a*r/R0)) * jnp.tan(jnp.pi * χ))

def _R(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))

def _Y(r, χ):
    return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))

@jax.jit
def F(x):
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ) * jnp.cos(2 * jnp.pi * z),
                                _Y(r, χ),
                                _R(r, χ) * jnp.sin(2 * jnp.pi * z)]))
# %%
ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0)
E0, E1, E2, E3 = [LazyExtractionOperator(Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
P0, P1, P2, P3 = [Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M.T
C = LazyDoubleCurlMatrix(Λ1, Q, F, E1).M
D2 = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
D1 = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
D0 = LazyDerivativeMatrix(Λ0, Λ1, Q, F, E0, E1).M
Pc = CurlProjection(Λ1, Q, F, E1)                      # given A and B, computes (B, A x Λ[i])
# %%
def l2_product(f, g, Q):
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)
# %%
ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.ones(1)/2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:,0].reshape(nx, nx)
_y2 = _y[:,1].reshape(nx, nx)
_nx = 16
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.ones(1)/2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:,0].reshape(_nx, _nx)
__y2 = __y[:,1].reshape(_nx, _nx)

def A(x):
    r, χ, z = x
    a1 = jnp.sin(2 * jnp.pi * χ)
    a2 = 1
    a3 = jnp.cos(2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3]) * jnp.sin(jnp.pi * r)**2 * r
B = curl(A)

# %%
A_hat = jnp.linalg.solve(M1, P1(A))
A_h = DiscreteFunction(A_hat, Λ1, E1)
err = lambda x: A(x) - A_h(x)
(l2_product(err, err, Q) / l2_product(A, A, Q))**0.5

# %%
B0 = curl(A)
B0_hat = jnp.linalg.solve(M2, P2(B0))
B_h = DiscreteFunction(B0_hat, Λ2, E2)
B0_h = DiscreteFunction(B0_hat, Λ2, E2)
err = lambda x: B0(x) - B_h(x)
(l2_product(err, err, Q) / l2_product(B0, B0, Q))**0.5

# %%
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1, 
    __y2,
    __z1[:,:,0], 
    __z1[:,:,1],
    color='w')
plt.xlabel('x')
plt.ylabel('y')

# %%
A_hat @ M12 @ B0_hat
# %%
U, S, Vh = jnp.linalg.svd(C)
S_inv = jnp.where(S/S[0] > 1e-11, 1/S, 0)
A_hat_recon = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B0_hat

# %%
print("error in Helicity:", (A_hat - A_hat_recon) @ M12 @ B0_hat / (A_hat @ M12 @ B0_hat) )
# %%
err = lambda x: curl(A)(x) - curl(A_h)(x)
print("error in curl A:", (l2_product(err, err, Q) / l2_product(curl(A), curl(A), Q))**0.5)
# %%
print("Helicity before perturbation: ", A_hat @ M12 @ B0_hat)
print("Energy before perturbation: ", B0_hat @ M2 @ B0_hat / 2)
# %%
# perturb helicity-preserving
# def u(x):
#     r, χ, z = x
#     a1 = 0
#     a2 = r**2 * (1-r)**2 * 100
#     a3 = 0
#     return jnp.array([a1, a2, a3])
# u_hat = jnp.linalg.solve(M2, P2(u))
# u_h = DiscreteFunction(u_hat, Λ2, E2)

# @jax.jit
# def perturb_B_hat(B_hat, B_hat_0, dt):
#     H_hat_1 = jnp.linalg.solve(M1, M12 @ B_hat)         # H = Proj(B)
#     H_hat_0 = jnp.linalg.solve(M1, M12 @ B_hat_0)
#     H_h = DiscreteFunction((H_hat_0 + H_hat_1)/2, Λ1, E1)           
#     u_h = DiscreteFunction(u_hat, Λ2, E2)
#     E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
#     ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
#     B_hat_1 = B_hat_0 + dt * ẟB_hat
#     err = (B_hat_1 - B_hat) @ M2 @ (B_hat_1 - B_hat)
#     return err, B_hat_1, u_hat

# # %%
# B_hat = B0_hat
# dt = 0.01

# # %%
# for i in range(int(0.1/dt)):
#     err = 1
#     B_hat_1 = B_hat
#     it = 0
#     while err > 1e-20:
#         err, B_hat_1, _u_hat = perturb_B_hat(B_hat_1, B_hat, dt)
#         it += 1
#     # if it < 5:
#     #     dt *= 1.2
#     # else:
#     #     dt *= 0.8
#     B_hat = B_hat_1
#     print("Iteration: ", i+1)
#     print("Magnetic Energy: ", (B_hat @ M2 @ B_hat) / 2)
#     print("Force: ", (_u_hat @ M2 @ _u_hat))
#     A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
#     print("Helicity: ", A_hat @ M12 @ B_hat)
#     print("Div B: ", (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
#     print("Picard iterations: ", it)
#     print("dt: ", dt)

# # %%
# B_h = DiscreteFunction(B_hat, Λ2, E2)

# # %%
# F_B = Pullback(B0, F, 2)
# F_B_h = Pullback(B_h, F, 2)
# _z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
# _z1_norm = jnp.linalg.norm(_z1, axis=2)
# _z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
# _z2_norm = jnp.linalg.norm(_z2, axis=2)
# plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
# plt.colorbar()
# plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
# __z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
# plt.quiver(
#     __y1, 
#     __y2,
#     __z1[:,:,0], 
#     __z1[:,:,1],
#     color='w')
# # %%
# # new helicity
# A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
# print("Helicity after perturbation: ", A_hat @ M12 @ B_hat)
# print("Energy after perturbation: ", B_hat @ M2 @ B_hat / 2)
# %%
# l2_product(div(A), div(A), Q)**0.5
# # %%
# l2_product(div(A_h), div(A_h), Q)**0.5
# BN_hat = B0_hat

# %%
# calculate the force for this state
# F = J x B = (curl B) x B
# i.e.
# F = J x B
# J = curl H
# H = Proj(B)
# %%

@jax.jit
def ẟB_hat(B_hat, B_hat_0, dt):
    H_hat = jnp.linalg.solve(M1, M12 @ (B_hat + B_hat_0))/2     # H = Proj(B)
    J_hat = jnp.linalg.solve(M1, D1.T @ (B_hat + B_hat_0))/2    # J = curl H
    H_h = DiscreteFunction(H_hat, Λ1, E1)           
    J_h = DiscreteFunction(J_hat, Λ1, E1)
    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))           # u = J x H
    u_h = DiscreteFunction(u_hat, Λ2, E2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    B_diff = B_hat_1 - B_hat
    err = (B_diff @ M2 @ B_diff)**0.5
    return B_diff, err, B_hat_1, u_hat

# %%
helicities = []
energies = []
forces = []
critical_as = []
divBs = []
dts = []
B_hat = B0_hat

# %%
dt0 = 1e-4
dt = dt0
B_hat_1 = B_hat
# %%
for i in range(10):
    err = 1
    it = 0
    while err > 1e-12:
        # Picard iteration
        # _, err, B_hat_1, _u_hat = ẟB_hat(B_hat_1, B_hat, dt)
        
        # Newton update step
        B_diff, err, _, _u_hat = ẟB_hat(B_hat_1, B_hat, dt)
        J = jax.jacrev(lambda B: ẟB_hat(B, B_hat, dt)[0])(B_hat_1)
        ẟB = -jnp.linalg.solve(J, B_diff)  
        B_hat_1 += ẟB
        
        it += 1
    dt = 0.01 / (_u_hat @ M2 @ _u_hat)**0.5
    #     if it > 10:
    #         dt *= 0.9
    #         continue
    # if it < 5:
    #     dt *= 1.05
    # else:
    #     dt *= 0.95
    B_hat = B_hat_1
    
    print("Iteration: ", i+1)
    print("Magnetic Energy: ", (B_hat @ M2 @ B_hat) / 2)
    print("Force: ", (_u_hat @ M2 @ _u_hat))
    A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
    print("Helicity: ", A_hat @ M12 @ B_hat)
    print("Div B: ", (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
    print("Iterations: ", it)
    print("dt: ", dt)
    
    # a = dt
    # while ((B_hat + a * _ẟB_hat) @ M2 @ (B_hat + a * _ẟB_hat)) > (B_hat @ M2 @ B_hat):
    #     a *= 0.8
    # a_crit = a
    # print("Critical a: ", a_crit)
    # B_hat = B_hat + a_crit * _ẟB_hat
    
    helicities.append(A_hat @ M12 @ B_hat)
    energies.append((B_hat @ M2 @ B_hat) / 2)
    forces.append((_u_hat @ M2 @ _u_hat))
    critical_as.append(it)
    divBs.append((D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
    dts.append(dt)
    
# %%
nt = 100_000
nr = 8
nχ = 4
_x1 = jnp.linspace(1/nr, 1-1/nr, nr)
_x2 = jnp.linspace(0, 1-1/nχ, nχ)
_x3 = jnp.ones(1)/2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nr*nχ*1, 3)

# %%
B_h = DiscreteFunction(B_hat, Λ2, E2)
dt = 1e-3
_B = jax.jit(B_h)

def rk4(y):
    k1 = dt * _B(y)
    k2 = dt * _B(y + 0.5 * k1)
    k3 = dt * _B(y + 0.5 * k2)
    k4 = dt * _B(y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

@jax.jit
def integrate(x):
    x_traj = [ x ]
    def step(x, _):
        x = jnp.mod(rk4(x), 1)
        return x, x
    _, x_traj = jax.lax.scan(step, x, None, length=nt)
    return jnp.array(x_traj)

x_traj = jax.vmap(integrate)(_x)
# %%
x_plot = jax.vmap(lambda i: jax.vmap(F)(x_traj[i]))(jnp.arange(x_traj.shape[0]))
# %%
colors = plt.cm.viridis(np.linspace(0, 1, len(x_plot)))
for i in range(len(x_plot)):
    _x_plot = x_plot[i, jnp.where(jnp.abs(x_traj[i, :, -1] < 1e-2))]
    plt.scatter(_x_plot[0, :, 0], _x_plot[0, :, 1], s=0.1, color=colors[i], alpha=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(2, 4)
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('y')
# %%
plt.plot(np.array(energies), label='E(0) - Energy')
plt.xlabel('Iteration')
plt.yscale('log')
plt.legend()

# %%
plt.plot(np.abs(np.array(helicities) - helicities[0]), label='|Helicity - H(0)|')
plt.xlabel('Iteration')
plt.legend()
plt.yscale('log')

# %%
plt.plot(np.array(forces)/np.array(energies), label='force/energy')
plt.xlabel('Iteration')
plt.yscale('log')
plt.legend()

# %%
plt.plot(np.abs(np.array(divBs) - divBs[0]), label='div B')
plt.xlabel('Iteration')
plt.legend()

# %%
plt.plot(critical_as, label='Picard iterations')
plt.xlabel('Iteration')
plt.yscale('log')
plt.legend()

# %%
plt.plot(dts, label='adaptive time-step')
plt.xlabel('Iteration')
plt.yscale('log')
plt.legend()

# %%
# F_B = Pullback(B0_h, F, 2)
# _z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
# _z2_norm = jnp.linalg.norm(_z2, axis=2)
# B_h = DiscreteFunction(B_hat, Λ2, E2)
# F_A_h = Pullback(B_h, F, 2)
# _z1 = jax.vmap(F_A_h)(_x).reshape(nx, nx, 3)
# _z1_norm = jnp.linalg.norm(_z1, axis=2)
# plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
# plt.colorbar()
# plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
# __z1 = jax.vmap(F_A_h)(__x).reshape(_nx, _nx, 3)
# plt.quiver(
#     __y1, 
#     __y2,
#     __z1[:,:,0], 
#     __z1[:,:,1],
#     color='w')
# %%
