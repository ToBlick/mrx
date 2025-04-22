# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector, CurlProjection
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
from mrx.Utils import curl
from mrx.IterativeSolvers import picard_solver
jax.config.update("jax_enable_x64", True)
# %%
# %%
ns = (10, 10, 1)
ps = (3, 3, 1)
types = ('periodic', 'periodic', 'constant')

Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in range(4)] # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, 4)              # Quadrature
F = lambda x: x                         # identity mapping
# %%
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).M for Λ in [Λ0, Λ1, Λ2, Λ3]] # assembled mass matries
P0, P1, P2, P3 = [Projector(Λ, Q) for Λ in [Λ0, Λ1, Λ2, Λ3] ]      # L2 projectors
Pc = CurlProjection(Λ1, Q)                      # given A and B, computes (B, A x Λ[i])
D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).M
    for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])] # grad, curl, div
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F).M.T      # L2 projection from H(curl) to H(div)
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F).M.T      # L2 projection from H1 to L2
C = LazyDoubleCurlMatrix(Λ1, Q).M               # bilinear form (A, E) → (curl A, curl E)
# K = LazyStiffnessMatrix(Λ0, Q).M                # bilinear form (q, p) → (grad q, grad p)

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
# %%
def l2_product(f, g, Q):
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)
# %%

def E(x, m, n):
    r, χ, z = x
    h = (1 + 0.0 * jnp.exp(-((r - 0.5)**2 + (χ - 0.5)**2) / 0.3**2))
    a1 =  jnp.sin(m * jnp.pi * r) * jnp.cos(n * jnp.pi * χ) * jnp.sqrt(n**2/(n**2 + m**2))
    a2 = -jnp.cos(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ) * jnp.sqrt(m**2/(n**2 + m**2))
    a3 = jnp.sin(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ)
    return jnp.array([a1, a2, a3]) * h
A = lambda x: E(x, 2, 2)

B0 = curl(A)
B0_hat = jnp.linalg.solve(M2, P2(B0))

# %%
U, S, Vh = jnp.linalg.svd(C)
S_inv = jnp.where(S > 1e-12 * S[0] * S.shape[0], 1/S, 0)
C_inv = Vh.T @ jnp.diag(S_inv) @ U.T
A_hat_recon = C_inv @ D1.T @ B0_hat

# %%
print("Helicity before perturbation: ", A_hat_recon @ M12 @ B0_hat)
print("Energy before perturbation: ", B0_hat @ M2 @ B0_hat / 2)
# %%
# perturb helicity-preserving
def u(x):
    r, χ, z = x
    a1 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    a2 = jnp.cos(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)
    a3 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3])
u_hat = jnp.linalg.solve(M2, P2(u))
u_h = DiscreteFunction(u_hat, Λ2)

B_hat = B0_hat

def twoformnorm(B):
    return (B @ M2 @ B)**0.5

@jax.jit
def force(B_n, B_0):
    H_hat = jnp.linalg.solve(M1, M12 @ (B_n + B_0)/2)
    J_hat = jnp.linalg.solve(M1, D1.T @ (B_n + B_0)/2)
    J_h = DiscreteFunction(J_hat, Λ1)
    H_h = DiscreteFunction(H_hat, Λ1)
    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))           # u = J x H
    return u_hat

@jax.jit
def force_residual(B_hat):
    u_hat = force(B_hat, B_hat)
    return (u_hat @ M2 @ u_hat)**0.5

def divergence_residual(B_hat):
    divB = ((D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))**0.5
    return divB

# coupled u
def ẟB(B_guess, B_n, u_n):
    H_hat = jnp.linalg.solve(M1, M12 @ (B_guess + B_n)/2)  # H = Proj(B)
    H_h = DiscreteFunction(H_hat, Λ1)
    u_h = DiscreteFunction(u_n, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB = jnp.linalg.solve(M2, D1 @ E_hat)               # ẟB = curl E
    return ẟB

def evolve_B(B_n, u, dt): 
    # input u is not used, can be previous u for momentum/conjugate descent
    def f(B):
        # coupled u
        # u_n = force(B, B_n)
        # decoupled u
        u_n = force(B_n, B_n)
        return B_n + dt * ẟB(B, B_n, u_n)
    # # midpoint
    return picard_solver(f, B_n, tol=1e-8, norm=twoformnorm)
    # # explicit Euler
    # return f(B_n)

def advect_B(B_n, u_n, dt):
    def f(B):
        return B_n + dt * ẟB(B, B_n, u_n)
    # # midpoint
    return picard_solver(f, B_n, tol=1e-8, norm=twoformnorm)
    # # explicit Euler
    # return f(B_n)

def f_relax(B_n, dt):
    B_n = evolve_B(B_n, None, dt)
    helicity = (C_inv @ D1.T @ B_n) @ M12 @ B_n
    energy = B_n @ M2 @ B_n / 2
    divB = divergence_residual(B_n)
    normF_n = force_residual(B_n)
    return B_n, (helicity, energy, divB, normF_n)

def f_loop_relax(i, B_hat):
    dt = dt_n[i]
    B_hat = evolve_B(B_hat, None, dt)
    normF = force_residual(B_hat)
    jax.debug.print("Force residual: {normF}", normF=normF)
    return B_hat

def f_perturb(B_hat, key):
    u_hat = jax.random.normal(key, shape=B_hat.shape)
    B_hat = advect_B(B_hat, u_hat, 1e-4)
    helicity = (C_inv @ D1.T @ B_hat) @ M12 @ B_hat
    energy = B_hat @ M2 @ B_hat / 2
    divB = divergence_residual(B_hat)
    normF = force_residual(B_hat)
    # jax.debug.print("Energy: {energy}", energy=energy)
    # jax.debug.print("Helicity: {helicity}", helicity=helicity)
    # jax.debug.print("Div B: {divB}", divB=divB)
    # jax.debug.print("Force residual: {normF}", normF=normF)
    return B_hat, (helicity, energy, divB, normF)

def f_loop_perturb(i, B_hat):
    key = keys[i]
    u_hat = jax.random.normal(key, shape=B_hat.shape)
    B_hat = advect_B(B_hat, u_hat, 1e-4)
    return B_hat
# %%
# # Check that Beltrami field is indeed force-free:
# print("Force residual: ", force_residual(B0_hat))
B_hat = BT_hat
B_h = DiscreteFunction(B_hat, Λ2)

# %%
# Check stability
@jax.jit
def integrand(i, j, x):
    B = B_h(x)
    Du_i = jax.jacfwd(Λ2[i])(x)
    Du_j = jax.jacfwd(Λ2[j])(x)
    term1 = (Du_i @ B) @ (Du_j @ B)
    term2 = - B @ (Du_i @ B) * jnp.trace(Du_j) - B @ (Du_j @ B) * jnp.trace(Du_i)
    term3 = 0.5 * B @ B * jnp.trace(Du_i) * jnp.trace(Du_j)
    term4 = 0.5 * B @ B * jnp.trace(Du_j @ Du_i)
    return term1 + term2 + term3 + term4

def lazy_stab_mat(i, j):
    return jax.vmap(integrand, (None, None, 0))(i, j, Q.x).T @ Q.w

# %%
stab_mat = jax.vmap(jax.vmap(lazy_stab_mat, (0, None)), (None, 0))(Λ2.ns, Λ2.ns)

# %%
jnp.max(stab_mat - stab_mat.T)

# %%
evd = jnp.linalg.eigh(stab_mat)

plt.plot(evd[0])

# %%
ker1 = evd[1][:, 0]
ker2 = evd[1][:, 1]

ker3 = evd[1][:, -1]
# %%
B_h = DiscreteFunction(ker1, Λ2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1, 
    __y2,
    __z1[:,:,0], 
    __z1[:,:,1],
    color='w')
# %%

# # %%
# _nx = 50
# # __x1 = jnp.linspace(0, 1-1/_nx, _nx)
# # __x2 = jnp.linspace(0, 1-1/_nx, _nx)
# # __x3 = jnp.array([0.5])
# # __x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
# # __x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
# __x = jax.random.uniform(jax.random.PRNGKey(3), shape=(_nx, 3), minval=0, maxval=1)
# __y = jax.vmap(F)(__x)

# def RK4(x, f, dt):
#     k1 = f(x)
#     k2 = f(x + dt / 2 * k1)
#     k3 = f(x + dt / 2 * k2)
#     k4 = f(x + dt * k3)
#     return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# @jax.jit
# def integrate(B_hat, x0, dt):
#     B_h = DiscreteFunction(B_hat, Λ2)
#     F_B_h = Pullback(B_h, F, 2)
    
#     def f(x, trace):
#         xplus1 = jnp.mod(RK4(x, F_B_h, dt), 1)
#         return xplus1, xplus1
    
#     _, x = jax.lax.scan(f, x0, jnp.arange(10_000))
#     return x
# # %%
# # __y = jax.random.uniform(jax.random.PRNGKey(3), shape=(9, 3), minval=0, maxval=1)
# x_test = jax.vmap(integrate, (None, 0, None))(ker1, __y, 1e-2)
# # %%
# for i in range(x_test.shape[0]):
#     plt.scatter(x_test[i,:,0], x_test[i,:,1], s=np.float64(np.abs(x_test[i,:,2] - 0.5) < 1e-2), alpha=0.75)

# # %%
# ax = plt.figure().add_subplot(projection='3d')
# for i in range(x_test.shape[0]):
#     ax.scatter(x_test[i,:,0], x_test[i,:,1], x_test[i,:,2], s=0.1, alpha=0.5)
# plt.show()
# %%
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 10)
traces = []
BN_hat = B0_hat
B_hat = B0_hat
# %%
# for key in jax.random.split(key, 50):
#     BN_hat, trace = f_perturb(BN_hat, key)
#     traces.append(trace)

for key in jax.random.split(key, 5):
    BN_hat, trace = jax.lax.scan(f_perturb, BN_hat, jax.random.split(key, 10))
    traces.append(trace)
# BN_hat = jax.lax.fori_loop(0, keys.shape[0], f_loop_perturb, BN_hat)  
  
# for i in range(keys.shape[0]):
#     BN_hat = f_loop_perturb(i, BN_hat)
# %%
print("Helicity after perturbation: ", (C_inv @ D1.T @ BN_hat) @ M12 @ BN_hat)
print("Energy after perturbation: ", BN_hat @ M2 @ BN_hat / 2)
print("Div B after perturbation: ", divergence_residual(BN_hat))
print("Force residual after perturbation: ", force_residual(BN_hat))

# %%
trace_array = jnp.hstack(jnp.array(traces))
helicity, energy, divB, normF = trace_array
base_energy = B0_hat @ M2 @ B0_hat / 2
plt.plot(energy - base_energy, label='Energy')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(helicity - helicity[0], label='Helicity')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(divB - divB[0], label='|Div B|')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(normF, label='|F|')
plt.xlabel('Iteration')
plt.yscale('log')
plt.legend()
# %%
# non-equispaced timesteps
N = 10
k = N//3
j = jnp.arange(N+1)
nu = k * j
x_n = 1 + jnp.cos((2 * nu - 1) * jnp.pi / (2 * N) )
dt_0 = 1e-5
dt_n = jnp.array([dt_0] * N)
# dt_n = dt_0 / (x_n + 1/N**2)

# plt.plot(dt_n)
# plt.yscale('log')
# %%
BT_hat = BN_hat
traces = []

# %%
for i in range(500):
    BT_hat, trace = jax.lax.scan(f_relax, BT_hat, dt_n)
    traces.append(trace)
    print("Iteration: ", i * 10)
print("Force residual: ", traces[-1][-1][-1])
# %%
# BT_hat, trace = jax.lax.scan(f_relax, BT_hat, dt_n)

# BT_hat = jax.lax.fori_loop(0, dt_n.shape[0], f_loop_relax, BT_hat)

# for i in range(dt_n.shape[0]):
    # BT_hat = f_loop_relax(i, BT_hat)
# %%
# for dt in dt_n:
#     B_hat, trace = f_relax(B_hat, dt)
    
#     normF = trace[-1]
#     if normF < 1e-12:
#         break
#     jax.debug.print("Force residual: {normF}", normF=normF)
#     traces.append(trace)
# %%

# %%
trace_array = jnp.hstack(jnp.array(traces))
print("Helicity after relaxation: ", trace_array[0,-1])
print("Energy after relaxation: ", trace_array[1,-1])
print("Div B after relaxation: ", trace_array[2,-1])
print("Force residual after relaxation: ", trace_array[3,-1])
# %%
helicity, energy, divB, normF = trace_array
base_energy = B0_hat @ M2 @ B0_hat / 2
plt.plot(energy, label='Energy')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(np.abs(helicity - helicity[0]), label='|Helicity - Helicity(0)|')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(divB - divB[0], label='|Div B|')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(normF, label='|F|')
plt.xlabel('Iteration')
plt.yscale('log')
plt.legend()

# %%
print("B(0) - B(1): ", ((B0_hat - BN_hat) @ M2 @ (B0_hat - BN_hat) / ( B0_hat @ M2 @ B0_hat))**0.5)
print("B(0) - B(T): ", ((B0_hat - BT_hat) @ M2 @ (B0_hat - BT_hat) / ( B0_hat @ M2 @ B0_hat))**0.5)

# %%
print("F(0): ", force_residual(B0_hat))
print("F(1): ", force_residual(BN_hat))
print("F(T): ", force_residual(BT_hat))

# %%
print("E(0): ", B0_hat @ M2 @ B0_hat / 2)
print("E(1): ", BN_hat @ M2 @ BN_hat / 2)
print("E(T): ", BT_hat @ M2 @ BT_hat / 2)
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

# %%
B_h = DiscreteFunction(B0_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.clim(0, 20)
plt.colorbar()
# plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1, 
    __y2,
    __z1[:,:,0], 
    __z1[:,:,1],
    color='w')

# %%
B_h = DiscreteFunction(BN_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.clim(0, 20)
plt.colorbar()
# plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1, 
    __y2,
    __z1[:,:,0], 
    __z1[:,:,1],
    color='w')

# %%
B_h = DiscreteFunction(BT_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.clim(0, 20)
# plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1, 
    __y2,
    __z1[:,:,0], 
    __z1[:,:,1],
    color='w')
# %%
u_hat = force(BT_hat, BT_hat)
B_h = DiscreteFunction(u_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
# plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1, 
    __y2,
    __z1[:,:,0], 
    __z1[:,:,1],
    color='w')
# %%
_nx = 7
__x1 = jnp.linspace(0, 1-1/_nx, _nx)
__x2 = jnp.linspace(0, 1-1/_nx, _nx)
__x3 = jnp.array([0.5])
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)

def RK4(x, f, dt):
    k1 = f(x)
    k2 = f(x + dt / 2 * k1)
    k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

@jax.jit
def integrate(B_hat, x0, dt):
    B_h = DiscreteFunction(B_hat, Λ2)
    F_B_h = Pullback(B_h, F, 2)
    
    def f(x, trace):
        xplus1 = jnp.mod(RK4(x, F_B_h, dt), 1)
        return xplus1, xplus1
    
    _, x = jax.lax.scan(f, x0, jnp.arange(10_000))
    return x
# %%
# __y = jax.random.uniform(jax.random.PRNGKey(3), shape=(9, 3), minval=0, maxval=1)
x_test = jax.vmap(integrate, (None, 0, None))(BT_hat, __y, 1e-3)

# %%
# ax = plt.figure().add_subplot(projection='3d')
# for i in range(x_test.shape[0]):
#     ax.scatter(x_test[i,:,0], x_test[i,:,1], x_test[i,:,2], s=0.1, alpha=0.5)
# plt.show()

# %%
for i in range(x_test.shape[0]):
    plt.scatter(x_test[i,:,0], x_test[i,:,1], s=np.float64(np.abs(x_test[i,:,2] - 0.5) < 1e-2), alpha=0.5)

# %%
plt.plot(x_test[0,:,0])

# %%
# b = 0.998
# ds = 1e-6
# a = 1.0

# @jax.jit
# def f_relax(x, key):
    
#     B_n, u_nminus1, normF_nminus1 = x
    
#     F_n = force(B_n)
#     normF_n = F_n @ M2 @ F_n

#     u_n = F_n + b * normF_n / normF_nminus1 * u_nminus1

#     B_s = B_n + ds * ẟB(B_n, B_n, u_n) # calling this with 2x B_n just does an explicit Euler step
#     F_s = force(B_s)
#     ẟW_s = F_s @ M2 @ u_n
#     ẟW_n = F_n @ M2 @ u_n
    
#     dt = - ds * a * ẟW_n / (ẟW_s - ẟW_n)
    
#     jax.debug.print("dt: {dt}", dt=dt)
    
#     B_n = advect_B(B_n, u_n, dt)
    
#     helicity = (C_inv @ D1.T @ B_n) @ M12 @ B_n
#     energy = B_n @ M2 @ B_n / 2
#     divB = divergence_residual(B_n)
    
#     x = B_n, u_n, normF_n

#     return x, (helicity, energy, divB, normF_n)

# # %%
# # %%
# key = jax.random.PRNGKey(0)
# B_hat = BN_hat
# traces = []
# # %%
# for key in jax.random.split(key, 50):
#     x = B_hat, jnp.zeros_like(B_hat), 1.0
#     x, trace = f_relax(x, key)
#     # BN_hat, trace = jax.lax.scan(f, BN_hat, jax.random.split(key, 10))
#     normF = trace[-1]
#     if normF < 1e-12:
#         break
#     jax.debug.print("Force residual: {normF}", normF=normF)
#     traces.append(trace)

# %%
#scheme where u is held fixed
# @jax.jit
# def ẟB_hat(B_guess, B_hat, u_hat):
#     H_hat = jnp.linalg.solve(M1, M12 @ (B_guess + B_hat)/2)
#     H_h = DiscreteFunction(H_hat, Λ1)
#     u_h = DiscreteFunction(u_hat, Λ2)
#     E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))  # E = u x H
#     ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)   # ẟB = curl E
#     return ẟB_hat
    

    
# @jax.jit
# def update_B_hat(B_n, u_n_minus1):
#     F_n = force(B_n)
#     u_n = α * F_n
#     B_s = B_n + ds * ẟB_hat(B_n, B_n, u_n) # calling this with 2x B_n just does an explicit Euler step
#     F_s = force(B_s)
#     ẟW_s = F_s @ M2 @ u_n
#     ẟW_n = F_n @ M2 @ u_n
    
#     dt = - ds * a * ẟW_n / (ẟW_s - ẟW_n)
    
#     B_nplus1 = B_n + dt * ẟB_hat(B_n, B_n, u_n) # explicit Euler step
    
#     def f(B):
#         return B_n + dt * ẟB_hat(B, B_n, u_n)   # fixed point: B_n+1 = B_n + dt * ẟB_hat(B_n+1, B_n, u_n)
    
#     jax.debug.print("dt: {dt}", dt=dt)
    
#     B_nplus1 = newton_solver(f, B_n, tol=1e-12, norm=twoformnorm)
    
#     return B_hat

# # %%
# B_hat = BN_hat
# # %%
# B_hat = update_B_hat(B_hat, None)
# # %%
# traces = []
# # %%
# for i in range(10):
#     helicity = (C_inv @ D1.T @ B_hat) @ M12 @ B_hat
#     energy = B_hat @ M2 @ B_hat / 2
#     divB = divergence_residual(B_hat)
#     normF = force_residual(B_hat)
#     traces.append((helicity, energy, divB, normF))
#     jax.debug.print("Force residual: {normF}", normF=normF)
#     B_hat = update_B_hat(B_hat, None)
# # %%
# trace = jnp.vstack(jnp.array(traces))
# __helicity, __energy, __divB, __force_res = trace.T

# # %%
# plt.plot(__energy)
# plt.xlabel('Iteration')
# plt.ylabel('Energy - Energy(0)')
# plt.yscale('log')
# plt.legend()
# # %%
# plt.plot(__helicity - __helicity[0])
# plt.xlabel('Iteration')
# plt.ylabel('Helicity - Helicity(0)')
# plt.legend()
# # %%
# plt.plot(__divB)
# plt.xlabel('Iteration')
# plt.ylabel('|Div B|')
# plt.legend()

# # %%
# plt.plot(__force_res )
# plt.xlabel('Iteration')
# plt.ylabel('| J x B |')
# plt.yscale('log')
# plt.legend()

# %%
