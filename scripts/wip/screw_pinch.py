# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Flat, Pushforward
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleCurlMatrix,
    LazyMassMatrix,
    LazyProjectionMatrix,
)
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import inv33, jacobian_determinant

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

a = 1
B0 = 1.0
q0 = 0.8
q1 = q0  # 1.85
R0 = 0
π = jnp.pi
γ = 5/3
alpha = 4

n = 5
p = 3
q = 3*p
ns = (8, 4, 1)
ps = (p, p, 0)
types = ("clamped", "periodic", "fourier")


def _X(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))


def _Y(r, χ):
    return jnp.ones(1) * a * r * jnp.sin(2 * π * χ)


def _Z(r, χ):
    return jnp.ones(1)


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array([_X(r, χ),
                                _Y(r, χ),
                                _Z(r, χ) * z]))


# Set up plotting grid
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
__x1 = jnp.linspace(tol, 1 - tol, _nx)
__x2 = jnp.linspace(0, 1, _nx)
__x3 = jnp.zeros(1) / 2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx * _nx * 1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)
__nx = 64
___x1 = jnp.linspace(tol, 1 - tol, nx)
___x2 = jnp.zeros(1)
___x3 = jnp.zeros(1)
___x = jnp.array(jnp.meshgrid(___x1, ___x2, ___x3))
___x = ___x.transpose(1, 2, 3, 0).reshape(__nx * 1 * 1, 3)
___y = jax.vmap(F)(___x)
___y1 = ___y[:, 0]



# %%
Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(k, ns, ps, types) for k in range(4)]
Q = QuadratureRule(Λ0, q)
ξ = get_xi(_X, _Y, Λ0, Q)[0]

boundary_operator_1_dbc = LazyExtractionOperator(Λ1, ξ, True).M
boundary_operator_2_dbc = LazyExtractionOperator(Λ2, ξ, True).M
boundary_operator_1 = LazyExtractionOperator(Λ1, ξ, False).M
boundary_operator_2 = LazyExtractionOperator(Λ2, ξ, False).M
boundary_operator_3 = LazyExtractionOperator(Λ3, ξ, False).M

mass_matrix_1_cart = LazyMassMatrix(Λ1, Q, F, None).M
mass_matrix_1 = boundary_operator_1 @ mass_matrix_1_cart @ boundary_operator_1.T
mass_matrix_1_dbc = boundary_operator_1_dbc @ mass_matrix_1_cart @ boundary_operator_1_dbc.T
dirichlet_operator_1 = jnp.linalg.solve(mass_matrix_1_dbc,
                                        boundary_operator_1_dbc @ mass_matrix_1_cart @ boundary_operator_1.T
                                        )

mass_matrix_2_cart = LazyMassMatrix(Λ2, Q, F, None).M
mass_matrix_2 = boundary_operator_2 @ mass_matrix_2_cart @ boundary_operator_2.T
mass_matrix_2_dbc = boundary_operator_2_dbc @ mass_matrix_2_cart @ boundary_operator_2_dbc.T
dirichlet_operator_2 = jnp.linalg.solve(mass_matrix_2_dbc,
                                        boundary_operator_2_dbc @ mass_matrix_2_cart @ boundary_operator_2.T
                                        )

mass_matrix_3 = LazyMassMatrix(Λ3, Q, F, boundary_operator_3).M

projector_1_dbc = Projector(Λ1, Q, F, boundary_operator_1_dbc)
projector_2_dbc = Projector(Λ2, Q, F, boundary_operator_2_dbc)
projector_1 = Projector(Λ1, Q, F, boundary_operator_1)
projector_2 = Projector(Λ2, Q, F, boundary_operator_2)
projector_3 = Projector(Λ3, Q, F, boundary_operator_3)


curl_matrix_cart = LazyDerivativeMatrix(Λ1, Λ2, Q, F, None, None).M
curl_matrix_dbc = boundary_operator_2_dbc @ curl_matrix_cart @ boundary_operator_1_dbc.T
curl_matrix = boundary_operator_2 @ curl_matrix_cart @ boundary_operator_1.T

curl_curl_matrix = LazyDoubleCurlMatrix(Λ1, Q, F, boundary_operator_1_dbc).M
U, S, Vh = jnp.linalg.svd(curl_curl_matrix)
S_inv = jnp.where(S > 1e-12, 1/S, 0)
curl_curl_matrix_pinv = Vh.T @ jnp.diag(S_inv) @ U.T

divergence_matrix_cart = LazyDerivativeMatrix(Λ2, Λ3, Q, F, None, None).M
divergence_matrix_dbc = boundary_operator_3 @ divergence_matrix_cart @ boundary_operator_2_dbc.T
divergence_matrix = boundary_operator_3 @ divergence_matrix_cart @ boundary_operator_2.T

laplacian_matrix_dbc = divergence_matrix_dbc @ mass_matrix_2_dbc @ divergence_matrix_dbc.T

projection_matrix_12 = jnp.linalg.solve(mass_matrix_1,
                                        LazyProjectionMatrix(
                                            Λ1, Λ2, Q, F, boundary_operator_1, boundary_operator_2_dbc
                                        ).M.T)
mass_matrix_12 = LazyProjectionMatrix(
    Λ1, Λ2, Q, F, boundary_operator_1_dbc, boundary_operator_2_dbc).M
# %%


class uxB:
    """
    Given 2-form u and 2-form B, computes ∫ (u × B) · Λ[i] dx for all i,
    where Λ[i] is the i-th basis function of the 1-form space.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E

    def __call__(self, u, B):
        return self.M @ self.projection(u, B)

    def projection(self, u, B):
        DF = jax.jacfwd(self.F)

        def _A(x):
            DFx_inv = inv33(DF(x))
            return DFx_inv @ DFx_inv.T @ jnp.cross(u(x), B(x))

        def _Λ(x, i):
            # Note: cross products of one-forms transform like two-forms
            return self.Λ(x, i)

        # Compute projections
        Ajk = jax.vmap(_A)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        # Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j->i", Λijk, Ajk, wj)


class JxB:
    """
    Given 2-form J and 2-form B, computes ∫ (J × B) · Λ[i] dx for all i,
    where Λ[i] is the i-th basis function of the 2-form space.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E

    def __call__(self, J, B):
        return self.M @ self.projection(J, B)

    def projection(self, J, B):
        def _A(x):
            return jnp.cross(J(x), B(x))

        def _Λ(x, i):
            return self.Λ(x, i)

        # Compute projections
        Ajk = jax.vmap(_A)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Ajk, 1/Jj, wj)


class dp_plus_gu:
    """
    Given 2-forms u, g and 3-forms d, p, computes ∫ (dp + g·u ) Λ[i] dx for all i,
    where Λ[i] is the i-th basis function of the 3-form space.
    """

    def __init__(self, Λ, Q, F=None, E=None):
        self.Λ = Λ
        self.Q = Q
        self.n = Λ.n
        self.ns = Λ.ns
        if F is None:
            self.F = lambda x: x
        else:
            self.F = F
        if E is None:
            self.M = jnp.eye(self.n)
        else:
            self.M = E

    def __call__(self, d, p, g, u):
        return self.M @ self.projection(d, p, g, u)

    def projection(self, d, p, g, u):
        DF = jax.jacfwd(self.F)

        def _A(x):
            DFx = DF(x)
            return d(x) * p(x) + g(x) @ DFx.T @ DFx @ u(x)

        def _Λ(x, i):
            return self.Λ(x, i)

        # Compute projections
        Ajk = jax.vmap(_A)(self.Q.x)  # n_q x d
        Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
            self.Q.x, jnp.arange(self.Λ.n))  # n x n_q x d
        Jj = jax.vmap(jacobian_determinant(self.F))(self.Q.x)  # n_q x 1
        wj = self.Q.w
        return jnp.einsum("ijk,jk,j,j->i", Λijk, Ajk, 1/Jj**2, wj)

# %%


def p_analytic(x):
    r, χ, z = x
    return (alpha + 1) / (2 * alpha) * (1 - r**(2*alpha)) * jnp.ones(1)
    # qr = q0 + (q1 - q0) * r**2
    # return (1 - r**2) * jnp.ones(1)  # * (1 - 0.2 * jnp.cos(2 * jnp.pi * r))
    # return - B0**2 * (r**2 - 1) / (q0**2 * R0**2) * jnp.ones(1)
    # return q0 * a**2 * B0**2 / (2 * R0**2 * (q1 - q0)) * (1/qr**2 - 1/q1**2) * jnp.ones(1)
    # return (B0**2 / 48) * (5 - 12*r**2 + 9*r**4 - 2*r**6) * jnp.ones(1)


def B_analytic(x):
    r, χ, z = x
    return jnp.array([0, r**alpha * 2 * π * r, 1])
    # qr = q0 + (q1 - q0) * r**2
    # Btheta = r
    # Bz = 0.2 * (1 - r**2)
    # Btheta = r * a / R0 / qr
    # Btheta = r * 0.5 * (1 - 0.5 * r**2)
    # return jnp.array([0, Btheta, Bz]) * B0

# def p_0(x):
#     r, χ, z = x
#     return (γ / (1 + γ))**γ * jnp.ones(1)


B_0 = B_analytic

p_0 = p_analytic
# def p_0(x):
#     r, χ, z = x
#     delta_rho = 0.05 * jnp.sin(2 * π * r) / r
#     return (p_analytic(x)**(1/γ) + delta_rho)**γ


# %%
H_analytic = B_0
p_hat = jnp.linalg.solve(mass_matrix_3, projector_3(p_0))
B_hat = jnp.linalg.solve(mass_matrix_2_dbc, projector_2_dbc(H_analytic))
o_hat = jnp.linalg.solve(mass_matrix_3, projector_3(lambda x: jnp.ones(1)))
# %%
A_hat = curl_curl_matrix_pinv @ curl_matrix_dbc.T @ B_hat
H_0 = B_hat @ mass_matrix_12 @ A_hat
H_0 / π 
# %%
@jax.jit
def mass(p_hat):
    p_h = DiscreteFunction(p_hat, Λ3, boundary_operator_3)
    J = jax.vmap(jacobian_determinant(F))(Q.x)  # n_q x 1
    return jnp.sum(jax.vmap(p_h)(Q.x)[:, 0]**(1/γ) * J**(1 - 1/γ) * Q.w)

# %%
mass(p_hat) / jnp.pi
# %%
PuxB = uxB(Λ1, Q, F, boundary_operator_1_dbc)
PJxB = JxB(Λ2, Q, F, boundary_operator_2_dbc)
Pdpgu = dp_plus_gu(Λ3, Q, F, boundary_operator_3)
# %%


def u_pert(x):
    r, χ, z = x
    return jnp.array([jnp.cos(4 * jnp.pi * χ), 1.0, -1.0]) * r**2 * (1 - r)**2


@jax.jit
def perturb(B_hat, p_hat, dt):
    B_h = DiscreteFunction(B_hat, Λ2, boundary_operator_2_dbc)
    p_h = DiscreteFunction(p_hat, Λ3, boundary_operator_3)
    # H, J, g functions
    g_hat = - jnp.linalg.solve(mass_matrix_2_dbc,
                               divergence_matrix_dbc.T @ p_hat)
    g_h = DiscreteFunction(g_hat, Λ2, boundary_operator_2_dbc)
    # perturbation is a poloidal flow
    u_hat = jnp.linalg.solve(
        mass_matrix_2_dbc, projector_2_dbc(Flat(u_pert, F)))
    u_h = DiscreteFunction(u_hat, Λ2, boundary_operator_2_dbc)
    # evolve B
    E_hat = jnp.linalg.solve(mass_matrix_1_dbc, PuxB(u_h, B_h))
    dB_hat = jnp.linalg.solve(mass_matrix_2_dbc, curl_matrix_dbc @ E_hat)
    # evolve p
    d_hat = γ * jnp.linalg.solve(mass_matrix_3, divergence_matrix_dbc @ u_hat)
    d_h = DiscreteFunction(d_hat, Λ3, boundary_operator_3)
    dp_hat = - Pdpgu(d_h, p_h, g_h, u_h)

    return B_hat + dt * dB_hat, p_hat + dt * dp_hat, u_hat
# %%


@jax.jit
def update(B_hat, p_hat, dt):
    B_h = DiscreteFunction(B_hat, Λ2, boundary_operator_2_dbc)
    p_h = DiscreteFunction(p_hat, Λ3, boundary_operator_3)
    # H, J, g functions
    H_hat = projection_matrix_12 @ B_hat
    J_hat = jnp.linalg.solve(mass_matrix_2, curl_matrix @ H_hat)
    J_h = DiscreteFunction(J_hat, Λ2, boundary_operator_2)
    g_hat = - jnp.linalg.solve(mass_matrix_2_dbc,
                               divergence_matrix_dbc.T @ p_hat)
    g_h = DiscreteFunction(g_hat, Λ2, boundary_operator_2_dbc)
    # compute force
    u_hat = jnp.linalg.solve(mass_matrix_2_dbc, PJxB(J_h, B_h)) - g_hat
    u_h = DiscreteFunction(u_hat, Λ2, boundary_operator_2_dbc)
    # evolve B
    E_hat = jnp.linalg.solve(mass_matrix_1_dbc, PuxB(u_h, B_h))
    dB_hat = jnp.linalg.solve(mass_matrix_2_dbc, curl_matrix_dbc @ E_hat)
    # evolve p
    d_hat = γ * jnp.linalg.solve(mass_matrix_3, divergence_matrix_dbc @ u_hat)
    d_h = DiscreteFunction(d_hat, Λ3, boundary_operator_3)
    dp_hat = - Pdpgu(d_h, p_h, g_h, u_h)

    return B_hat + dt * dB_hat, p_hat + dt * dp_hat, u_hat


# %%
# for _ in range(10):
#     B_hat, p_hat, u_hat = perturb(B_hat, p_hat, 1e-3)

p_hat = jnp.linalg.solve(mass_matrix_3, projector_3(p_0))
B_hat = jnp.linalg.solve(mass_matrix_2_dbc, projector_2_dbc(H_analytic))
# %%
_, _, u_hat = update(B_hat, p_hat, 1e-3)
# d_hat = jnp.linalg.solve(mass_matrix_3, divergence_matrix_dbc @ u_hat)
# c_hat = jnp.linalg.solve(mass_matrix_1_dbc, curl_matrix_dbc.T @ u_hat)
print("|u|**2: ", (u_hat @ mass_matrix_2_dbc @ u_hat))
trace_u = []
trace_E = []
trace_H = []
trace_mass = []

# %%
for _ in range(50):
    B_hat, p_hat, u_hat = update(B_hat, p_hat, 5e-4)
    A_hat = curl_curl_matrix_pinv @ curl_matrix_dbc.T @ B_hat
    # print("|u|: ", (u_hat @ mass_matrix_2_dbc @ u_hat)**0.5)
    trace_u.append(u_hat @ mass_matrix_2_dbc @ u_hat)
    trace_E.append(0.5 * B_hat @ mass_matrix_2_dbc @ B_hat
                   + jnp.sum(p_hat)/(γ - 1))
    trace_H.append(B_hat @ mass_matrix_12 @ A_hat)
    trace_mass.append(mass(p_hat))
    print("|u|: ", trace_u[-1])

# %%
plt.plot(trace_u)
plt.yscale("log")
# %%
plt.plot(np.array(trace_E))
# %%
plt.plot(np.array(trace_H)/trace_H[0] - 1, label='H')
plt.plot(np.array(trace_mass)/trace_mass[0] - 1, label='mass')
plt.legend()
# %%
u_h = DiscreteFunction(u_hat, Λ2, boundary_operator_2_dbc)
B_h = DiscreteFunction(B_hat, Λ2, boundary_operator_2_dbc)
p_h = DiscreteFunction(p_hat, Λ3, boundary_operator_3)
# %%
F_u_h = Pushforward(u_h, F, 2)
_z1 = jax.vmap(F_u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
F_p_h = Pushforward(p_h, F, 3)
_z1 = jax.vmap(F_p_h)(_x).reshape(nx, nx)
plt.contourf(_y1, _y2, _z1)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Z')
# %%
F_B_h = Pushforward(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
_z1 = jax.vmap(p_h)(___x)[:, 0] / ___x[:, 0] / 2 / jnp.pi
_z3 = jax.vmap(B_h)(___x)[:, 0] / ___x[:, 0] / 2 / jnp.pi
_z4 = jax.vmap(B_h)(___x)[:, 1]
_z5 = jax.vmap(B_h)(___x)[:, 2] / ___x[:, 0] / 2 / jnp.pi
_z6 = jax.vmap(p_analytic)(___x)
_z7 = jax.vmap(B_analytic)(___x)
_z8 = jax.vmap(p_0)(___x)
_z9 = jax.vmap(B_0)(___x)

# %%
plt.plot(___y1, _z1, label='pressure')
plt.plot(___y1, _z3, label='B r-component')
plt.plot(___y1, _z4, label='B chi-component')
plt.plot(___y1, _z5, label='B z-component')
plt.plot(___y1, _z6, label='p analytic', linestyle='--')
plt.plot(___y1, _z7[:, 0], label='Br analytic', linestyle='--')
plt.plot(___y1, _z7[:, 1] / (2 * π * ___x[:, 0]), label='Bchi analytic', linestyle='--')
plt.plot(___y1, _z7[:, 2], label='Bz analytic', linestyle='--')
plt.plot(___y1, _z8, label='p_0 analytic', linestyle='--')
# plt.plot(___y1, _z9[:, 0], label='Br_0 analytic', linestyle='--')
# plt.plot(___y1, _z9[:, 1], label='Bchi_0 analytic', linestyle='--')
# plt.plot(___y1, _z9[:, 2], label='Bz_0 analytic', linestyle='--')
plt.legend()
plt.xlabel('r')
plt.ylabel('p')
# %%
