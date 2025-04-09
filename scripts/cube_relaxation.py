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
ns = (8, 8, 1)
ps = (3, 3, 1)
types = ('periodic', 'periodic', 'constant')

Λ0 = DifferentialForm(0, ns, ps, types) # functions in H1
Λ1 = DifferentialForm(1, ns, ps, types) # vector fields in H(curl)
Λ2 = DifferentialForm(2, ns, ps, types) # vector fields in H(div)
Λ3 = DifferentialForm(3, ns, ps, types) # densities in L2
Q = QuadratureRule(Λ0, 10)              # Quadrature
F = lambda x: x                         # identity mapping
# %%
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).M 
    for Λ in [Λ0, Λ1, Λ2, Λ3]]                  # assembled mass matries
P0, P1, P2, P3 = [ Projector(Λ, Q) 
    for Λ in [Λ0, Λ1, Λ2, Λ3] ]                 # L2 projectors
Pc = CurlProjection(Λ1, Q)                      # given A and B, computes (B, A x Λ[i])
D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).M
    for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])] # grad, curl, div
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F).M      # L2 projection from H(curl) to H(div)
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F).M      # L2 projection from H1 to L2
C = LazyDoubleCurlMatrix(Λ1, Q).M               # bilinear form (A, E) → (curl A, curl E)
K = LazyStiffnessMatrix(Λ0, Q).M                # bilinear form (q, p) → (grad q, grad p)

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
S_inv = jnp.where(S > 1e-6 * S[0] * S.shape[0], 1/S, 0)
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

@jax.jit
def _perturb_B_hat(B_guess, B_hat_0, dt):
    H_hat = jnp.linalg.solve(M1, M12 @ (B_guess + B_hat_0)/2) # H = Proj(B)
    H_h = DiscreteFunction(H_hat, Λ1)           
    u_h = DiscreteFunction(u_hat, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    return B_hat_1

@jax.jit
def perturb_B_hat(B_hat_0, dt):
    def cond_fun(B_guess):
        B_hat_1 = _perturb_B_hat(B_guess, B_hat_0, dt)
        err = ((B_hat_1 - B_guess) @ M2 @ (B_hat_1 - B_guess))**0.5
        return err > 1e-12
    def body_fun(B_guess):
        B_hat_1 = _perturb_B_hat(B_guess, B_hat_0, dt)
        return B_hat_1
    B_hat = jax.lax.while_loop(cond_fun, body_fun, B_hat_0)
    return B_hat

# %%
@jax.jit
def f(B_hat, x):
    B_hat = perturb_B_hat(B_hat, 1e-3)
    helicity = (C_inv @ D1.T @ B_hat) @ M12 @ B_hat
    energy = B_hat @ M2 @ B_hat / 2
    divB = ((D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))**0.5
    return B_hat, (helicity, energy, divB)

# %%
BN_hat, trace = jax.lax.scan(f, B0_hat, None, length=25)

# %%
helicity, energy, divB = trace
plt.plot(energy - energy[0], label='Energy')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(helicity - helicity[0], label='Helicity')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(divB - divB[0], label='Div B')
plt.xlabel('Iteration')
plt.legend()

# %%
def ẟB_hat(B_guess, B_hat_0, dt):
    H_hat = jnp.linalg.solve(M1, M12 @ (B_guess + B_hat_0)/2)  # H = Proj(B)
    J_hat = jnp.linalg.solve(M1, D1.T @ (B_guess + B_hat_0)/2) # J = curl H
    J_h = DiscreteFunction(J_hat, Λ1)
    H_h = DiscreteFunction(H_hat, Λ1)
    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))           # u = J x H          
    u_h = DiscreteFunction(u_hat, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    B_diff = B_hat_1 - B_guess
    return B_diff

def force_residual(B_hat):
    H_hat = jnp.linalg.solve(M1, M12 @ B_hat)
    J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
    J_h = DiscreteFunction(J_hat, Λ1)
    H_h = DiscreteFunction(H_hat, Λ1)
    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))           # u = J x H
    return (u_hat @ M2 @ u_hat)**0.5

def update_B_hat(B_hat_0, dt):
    def ẟB(B_guess):
        return ẟB_hat(B_guess, B_hat_0, dt)
    def cond_fun(B_guess):
        B_diff = ẟB(B_guess)
        err = (B_diff @ M2 @ B_diff)**0.5
        jax.debug.print("Residual: {err}", err=err)
        return err > 1e-12
    def body_fun(B_guess):
        B_diff = ẟB(B_guess)
        ### Picard method
        return B_guess + B_diff
        ### Newton method
        # J = jax.jacrev(ẟB)(B_guess)
        # return B_guess - jnp.linalg.solve(J, B_diff)
    B_hat = jax.lax.while_loop(cond_fun, body_fun, B_hat_0)
    return B_hat

@jax.jit
def f(B_hat, x):
    helicity = (C_inv @ D1.T @ B_hat) @ M12 @ B_hat
    energy = B_hat @ M2 @ B_hat / 2
    divB = ((D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))**0.5
    force = force_residual(B_hat)
    dt = 1e-4 / force
        
    jax.debug.print("Energy: {energy}", energy=energy)
    jax.debug.print("Helicity: {helicity}", helicity=helicity)
    jax.debug.print("Div B: {divB}", divB=divB)
    jax.debug.print("Force residual: {force}", force=force)
    jax.debug.print("time step: {dt}", dt=dt)
    B_hat = update_B_hat(B_hat, dt)
    return B_hat, (helicity, energy, divB)

# %%
B_hat, trace = jax.lax.scan(f, BN_hat, None, length=100)

# %%
_helicity, _energy, _divB = trace
# %%
plt.plot(energy, label='Energy')
plt.plot(_energy, label='Energy')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(helicity, label='Helicity')
plt.plot(_helicity, label='Helicity')
plt.xlabel('Iteration')
plt.legend()
# %%
plt.plot(divB, label='Div B')
plt.plot(_divB, label='Div B')
plt.xlabel('Iteration')
plt.legend()

# %%
print("B(0) - B(1): ", ((B0_hat - BN_hat) @ M2 @ (B0_hat - BN_hat) / ( B0_hat @ M2 @ B0_hat))**0.5)
print("B(0) - B(T): ", ((B0_hat - B_hat) @ M2 @ (B0_hat - B_hat) / ( B0_hat @ M2 @ B0_hat))**0.5)

# %%
print("F(0): ", force_residual(B0_hat))
print("F(1): ", force_residual(BN_hat))
print("F(T): ", force_residual(B_hat))
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
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
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
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1, 
    __y2,
    __z1[:,:,0], 
    __z1[:,:,1],
    color='w')

# %%
B_h = DiscreteFunction(B_hat, Λ2)
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
# %%
