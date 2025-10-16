
# %%
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from jax.numpy import cos, pi, sin

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Plotting import (
    get_2d_grids,
    get_3d_grids,
    plot_crossections_separate,
    plot_torus,
)

jax.config.update("jax_enable_x64", True)
# %%
# Mapping:


def helical_map(epsilon=0.2, h=0.2, n_turns=3, kappa=1.0, alpha=0.0):
    π = jnp.pi

    def X(ζ):
        return jnp.array([
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.cos(2 * π * ζ),
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.sin(2 * π * ζ),
            h * jnp.sin(2 * π * n_turns * ζ)
        ])

    def get_frame(ζ):
        dX = jax.jacrev(X)
        τ = dX(ζ) / jnp.linalg.norm(dX(ζ))  # Tangent vector
        dτ = jax.jacfwd(dX)(ζ)
        ν1 = dτ / jnp.linalg.norm(dτ)
        ν1 = ν1 / jnp.linalg.norm(ν1)  # First normal vector
        ν2 = jnp.cross(τ, ν1)         # Second normal vector
        return τ, ν1, ν2

    def x_t(t):
        return epsilon * jnp.cos(2 * π * t + alpha * jnp.sin(2 * π * t))

    def y_t(t):
        return epsilon * kappa * jnp.sin(2 * π * t)

    def _s_from_t(t):
        return jnp.arctan2(y_t(t), x_t(t))

    def s_from_t(t):
        return jnp.where(t > 0.5, _s_from_t(t) + 2 * π, _s_from_t(t))

    def a_from_t(t):
        return jnp.sqrt(x_t(t)**2 + y_t(t)**2)

    def F(x):
        """Helical coordinate mapping function."""
        r, t, ζ = x
        _, ν1, ν2 = get_frame(ζ)
        return (X(ζ)
                + a_from_t(t) * r * jnp.cos(s_from_t(t)) * ν1
                + a_from_t(t) * r * jnp.sin(s_from_t(t)) * ν2)

    return F


def siesta_map():
    Np = 3

    def R0(z):
        return 2.9

    def Z0(z):
        return 0.0

    def Rb(t, z):
        t = 2 * pi * t
        z = 2 * pi * z * Np
        return (R0(z)
                + cos(t)
                - 0.51 * cos(t + z)
                - 0.01 * cos(4 * t + 2 * z)
                - 0.01 * cos(6 * t + 2 * z)
                )

    def Zb(t, z):
        t = 2 * pi * t
        z = 2 * pi * z * Np
        return (Z0(z)
                + sin(t)
                + 0.51 * cos(t + z)
                + 0.01 * cos(4 * t + 2 * z)
                - 0.01 * cos(6 * t + 2 * z)
                )

    def F(x):
        r, χ, z = x
        z /= Np
        dR = r * (Rb(χ, z) - R0(z))
        dZ = r * (Zb(χ, z) - Z0(z))
        return jnp.array([(R0(z) + dR) * jnp.cos(2 * pi * z),
                          -(R0(z) + dR) * jnp.sin(2 * pi * z),
                          Z0(z) + dZ])

    return F


def w7x_map():
    π = jnp.pi
    Nfp = 5
    R0 = 5.5586

    m_vals = jnp.arange(0, rbc.shape[0])        # [0..6]
    n_vals = jnp.arange(-6, 7)                  # [-6..6]

    def R_axis(ɸ):
        return R0 + 0.35209 * jnp.cos(ɸ)

    def Z_axis(ɸ):
        return -0.29578 * jnp.sin(ɸ)

    # Fourier sums
    def eval_boundary(theta, phi):
        phases = (m_vals[:, None] * theta) - (n_vals[None, :] * Nfp * phi)
        cosvals = jnp.cos(phases)
        sinvals = jnp.sin(phases)
        Rb = jnp.sum(rbc * cosvals)
        Zb = jnp.sum(zbc * sinvals)
        return Rb, Zb

    def F(p):
        r, theta, zeta = p
        phi = -zeta * 2 * π
        theta *= 2 * π

        Rb, Zb = eval_boundary(theta, phi)
        Rc, Zc = R_axis(phi), Z_axis(phi)
        # eps = 0.5
        # # "smooth" the boundary by increasing two mode numbers
        # Rb = Rb + eps * (Rc + jnp.cos(theta))
        # Zb = Zb + eps * (Zc + jnp.sin(theta))

        # interpolate from magnetic axis to outer boundary
        R_pt = Rc + r * (Rb - Rc)
        Z_pt = Zc + r * (Zb - Zc)

        x = R_pt * jnp.cos(phi)
        y = R_pt * jnp.sin(phi)
        z = Z_pt

        return jnp.array([x, y, z]) / R0

    return F


def spec_map(a=0.1, b=0.025, m=5):
    π = jnp.pi

    def Rb(θ, ζ):
        return 1 + a * jnp.cos(2 * π * θ) + b * jnp.cos(2 * π * θ - 2 * π * m * ζ)

    def Zb(θ, ζ):
        return -a * jnp.sin(2 * π * θ) + b * jnp.sin(2 * π * θ - 2 * π * m * ζ)

    def F(x):
        r, θ, ζ = x
        ζ /= m
        R_val = 1 + r * (Rb(θ, ζ) - 1)
        Z_val = r * Zb(θ, ζ)
        return jnp.ravel(jnp.array([
            R_val * jnp.cos(2 * π * ζ),
            R_val * jnp.sin(2 * π * ζ),
            Z_val
        ]))

    return F


rbc = jnp.array([  # rows are m, cols are n (-6 to 6), modes are cos(m theta - n Nfp phi)
    # m = 0
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.5586, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # m = 1
    [5.8671e-4, -0.0020214, -0.0032499, -0.0048626, 0.0099524, 0.033555, 0.3 + 0.49093,  # !!!
     -0.25107, 0.0051767, 0.0048235, 3.2065e-4, 1.959e-4, -0.0013768],
    # m = 2
    [5.0917e-4, -6.6977e-4, -6.7572e-4, -0.0014368, 0.0020763, 0.014863, 0.038447,
     0.039485, 0.063998, -0.0079348, 0.0026108, 0.0020567, 5.4194e-4],
    # m = 3
    [-2.123e-4, 4.3695e-5, 3.3644e-4, 6.4011e-4, 0.0014828, 0.001668, -0.007879,
     -0.016151, -0.017821, -0.010231, 0.0020548, -0.0014357, -6.283e-4],
    # m = 4
    [1.6361e-4, -1.6965e-4, 1.6596e-4, 1.9015e-5, -1.6802e-4, -0.0018659, 0.003313,
     -0.0010438, 0.0092226, 7.7604e-4, 5.9064e-5, -5.2259e-5, 3.0372e-4],
    # m = 5
    [-9.7626e-5, -2.1519e-5, 4.0222e-5, -2.9516e-4, -2.6001e-4, 0.001071, 7.9054e-4,
     0.0043576, -8.9611e-4, -6.8915e-4, 0.0016336, -1.3732e-4, 6.2276e-5],
    # m = 6
    [4.932e-5, -1.0468e-4, 3.3192e-5, 1.7365e-4, 3.8511e-4, 1.0762e-4, 2.6971e-5,
     -0.003413, -0.0016615, -0.0010591, -5.3034e-4, -6.5265e-4, 9.7208e-5]
])

zbc = jnp.array([
    # m = 0
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.23754, -
        0.0068221, 0.0031833, 0.0017251, 5.7592e-5, 2.643e-4],
    # m = 1
    [0.0013323, -0.0016138, -0.0042108, -0.0070096, 0.0096053, 0.036669,
     0.3 + 0.61965, 0.17897, 0.0011535, 0.011208, 0.0042871, 0.0022236, -3.1109e-4],  # !!!
    # m = 2
    [-8.5478e-4, 3.8773e-4, -4.941e-4, 1.4908e-4, 0.0013789, 0.011065, -0.0089934,
        0.023219, -0.055638, 0.010888, -5.4241e-4, -0.0010494, -6.6246e-5],
    # m = 3
    [1.434e-4, -6.3872e-6, -1.558e-4, 1.7715e-4, 3.9966e-4, -0.0021165, -
        4.2775e-4, 6.5673e-4, 0.0075688, 0.0094628, -0.0025063, 7.8944e-4, 3.7802e-4],
    # m = 4
    [-1.0181e-4, 1.2157e-4, 4.6167e-5, -2.4091e-4, -1.4719e-4, -4.1381e-4,
        4.3528e-4, 0.0016804, 0.0072835, -0.0060814, 1.6546e-5, 9.278e-4, -2.6709e-4],
    # m = 5
    [-7.3965e-5, 1.3889e-4, -2.0578e-4, 1.2722e-4, 2.1614e-4, 0.0010369,
        0.0015842, -0.0038666, -0.0024331, 6.5004e-4, 5.165e-4, -3.3282e-4, -1.9109e-4],
    # m = 6
    [1.7171e-6, -5.2834e-5, 7.6033e-5, 1.1611e-4, 5.5193e-5, -1.5852e-5, -
        5.8329e-4, -2.3786e-4, -4.2717e-4, -8.1209e-4, -4.0144e-4, 2.4175e-4, -1.894e-4]
])


def torus_map():
    def F(x):
        r, θ, ζ = x
        R = 1 + 0.3 * r * cos(2 * pi * θ)
        Z = 0.3 * r * sin(2 * pi * θ)
        return jnp.array([R * cos(2 * pi * ζ),
                          -R * sin(2 * pi * ζ),
                          Z])
    return F


# F = helical_map(h = 0.0)
F = spec_map(a=0.1, b=0.025, m=5)
# F = w7x_map()
# F = siesta_map()
# F = torus_map()

grid = get_3d_grids(F, nx=8, ny=17, nz=17, x_min=1e-6)


def f_test(p):
    x1, x2, x3 = F(p)
    # return x[0] * x[1] * x[2] * jnp.ones(1)
    # return sin(x[0] * 2 * jnp.pi) * sin(x[1] * 2 * jnp.pi) * sin(x[2] * 2 * jnp.pi) * jnp.ones(1)
    R = (x1**2 + x2**2)**0.5
    phi = jnp.arctan2(x2, x1)
    r = p[0]
    return R * jnp.cos(phi) * jnp.ones(1) * (1-r**2)


def E_test(p):
    x1, x2, x3 = F(p)
    r = p[0]
    R = (x1**2 + x2**2)**0.5
    phi = jnp.arctan2(x2, x1)
    return jnp.array([0, R * jnp.cos(phi), 0]) * (1-r**2)
# %%
# Seq = DeRhamSequence((6, 6, 6), (3, 3, 3), 3,
#                      ("clamped", "periodic", "periodic"), F, polar=True, dirichlet=True)


# start = time.time()
# Seq.evaluate_1d()
# Seq.assemble_all()
# end = time.time()
# print("Assembly time:", end - start, "s")

# # %%
# print("first Evs of dd0:", jnp.linalg.eigh(Seq.M0 @ Seq.dd0)[0][:3])
# print("first Evs of dd1:", jnp.linalg.eigh(Seq.M1 @ Seq.dd1)[0][:3])
# print("first Evs of dd2:", jnp.linalg.eigh(Seq.M2 @ Seq.dd2)[0][:3])
# print("first Evs of dd3:", jnp.linalg.eigh(Seq.M3 @ Seq.dd3)[0][:3])
# print("curl grad", jnp.max(jnp.abs(Seq.strong_curl @ Seq.strong_grad)))
# print("div curl:", jnp.max(jnp.abs(Seq.strong_div @ Seq.strong_curl)))
# %%
# jnp.max(Seq.strong_curl @ Seq.strong_grad), jnp.max(Seq.strong_div @ Seq.strong_curl)
# # %%
# Seq.evaluate_all()
# # %%
# _i, _j, _k = Seq.dΛ0_ijk.shape
# d0 = jax.vmap(jax.vmap(jax.vmap(Seq.get_dΛ0_ijk, in_axes=(None, None, 0)), in_axes=(
#     None, 0, None)), in_axes=(0, None, None))(jnp.arange(_i), jnp.arange(_j), jnp.arange(_k))
# jnp.max(jnp.abs(d0 - Seq.dΛ0_ijk))
# # %%
# _i, _j, _k = Seq.dΛ1_ijk.shape
# d1 = jax.vmap(jax.vmap(jax.vmap(Seq.get_dΛ1_ijk, in_axes=(None, None, 0)), in_axes=(
#     None, 0, None)), in_axes=(0, None, None))(jnp.arange(_i), jnp.arange(_j), jnp.arange(_k))
# jnp.max(jnp.abs(d1 - Seq.dΛ1_ijk))
# # %%
# _i, _j, _k = Seq.dΛ2_ijk.shape
# d2 = jax.vmap(jax.vmap(jax.vmap(Seq.get_dΛ2_ijk, in_axes=(None, None, 0)), in_axes=(
#     None, 0, None)), in_axes=(0, None, None))(jnp.arange(_i), jnp.arange(_j), jnp.arange(_k))
# jnp.max(jnp.abs(d2 - Seq.dΛ2_ijk))
# # %%
# Seq.evaluate_2()
# M2 = jnp.einsum("ijk,jkl,qjl,j,j->iq", Seq.Λ2_ijk,
#                 Seq.G_jkl, Seq.Λ2_ijk, 1/Seq.J_j, Seq.Q.w)
# M2 = Seq.E2 @ M2 @ Seq.E2.T
# jnp.max(jnp.abs(M2 - Seq.M2))
# # %%
# Seq.evaluate_1()
# M1 = jnp.einsum("ijk,jkl,qjl,j,j->iq", Seq.Λ1_ijk,
#                 Seq.G_inv_jkl, Seq.Λ1_ijk, Seq.J_j, Seq.Q.w)
# M1 = Seq.E1 @ M1 @ Seq.E1.T
# jnp.max(jnp.abs(M1 - Seq.M1))

# # %%
# Seq.evaluate_0()
# M0 = jnp.einsum("ijk,qjk,j,j->iq", Seq.Λ0_ijk, Seq.Λ0_ijk, Seq.J_j, Seq.Q.w)
# M0 = Seq.E0 @ M0 @ Seq.E0.T
# jnp.max(jnp.abs(M0 - Seq.M0))

# # %%
# Seq.evaluate_3()
# M3 = jnp.einsum("ijk,qjk,j,j->iq", Seq.Λ3_ijk, Seq.Λ3_ijk, 1/Seq.J_j, Seq.Q. w)
# M3 = Seq.E3 @ M3 @ Seq.E3.T
# jnp.max(jnp.abs(M3 - Seq.M3))
# # %%
# Seq.get_Λ0_ijk(10, 4, 0), Seq.Λ0_ijk[10, 4, 0]
# # %%
# Seq.get_dΛ0_ijk(10, 4, 0), Seq.dΛ0_ijk[10, 4, 0]
# %%
# MapSeq = DeRhamSequence((6, 6, 6), (3, 3, 3), 3,
#                         ("clamped", "periodic", "periodic"), F, polar=True, dirichlet=False)
# MapSeq.evaluate_1d()
# MapSeq.assemble_M0()

# F_x_hat = jnp.linalg.solve(MapSeq.M0, MapSeq.P0(lambda x: F(x)[0:1]))
# F_x_h = DiscreteFunction(F_x_hat, MapSeq.Λ0, MapSeq.E0)
# F_y_hat = jnp.linalg.solve(MapSeq.M0, MapSeq.P0(lambda x: F(x)[1:2]))
# F_y_h = DiscreteFunction(F_y_hat, MapSeq.Λ0, MapSeq.E0)
# F_z_hat = jnp.linalg.solve(MapSeq.M0, MapSeq.P0(lambda x: F(x)[2:3]))
# F_z_h = DiscreteFunction(F_z_hat, MapSeq.Λ0, MapSeq.E0)


# @jax.jit
# def F_h(x):
#     return jnp.ravel(jnp.array([F_x_h(x), F_y_h(x), F_z_h(x)]))

# %%


@partial(jax.jit, static_argnames=["n", "p", "q"])
def proj_error(n, p, q):
    Seq = DeRhamSequence((n, n, n), (p, p, p), q,
                         ("clamped", "periodic", "periodic"), F, polar=True, dirichlet=True)
    Seq.evaluate_1d()
    Seq.assemble_M2()

    f_hat = jnp.linalg.solve(Seq.M2, Seq.P2(E_test))
    f_h = Pushforward(DiscreteFunction(f_hat, Seq.Λ2, Seq.E2), F, 2)

    Seq2 = DeRhamSequence((n, n, n), (p, p, p), 2*q,
                          ("clamped", "periodic", "periodic"), F, polar=True, dirichlet=True)

    df = jax.vmap(lambda x: E_test(x) - f_h(x))(Seq2.Q.x)
    L2_df = jnp.einsum('ik,ik,i,i->', df, df, Seq2.J_j, Seq2.Q.w)**0.5
    L2_f = jnp.einsum('ik,ik,i,i->',
                      jax.vmap(E_test)(Seq2.Q.x), jax.vmap(E_test)(Seq2.Q.x),
                      Seq2.J_j, Seq2.Q.w)**0.5
    return L2_df / L2_f


# %%
errs = []
ns = np.arange(4, 9, 1)
for n in ns:
    print(f"n = {n}")
    errs.append(proj_error(n, 3, 3))
    print("err =", errs[-1])

# %%
plt.plot(ns, errs, marker='o', label="L2 projection error")
plt.plot(ns, (0.3 * ns)**-2, marker='o', label=r"$\mathcal{O}(h^2)$")
plt.yscale("log")
plt.xscale("log")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.xlabel("n")
plt.legend()
plt.show()

# %%
Seq = DeRhamSequence((8, 8, 6), (3, 3, 3), 3,
                     ("clamped", "periodic", "periodic"), F, polar=True, dirichlet=True)
Seq.evaluate_1d()
Seq.assemble_all()

assert jnp.min(Seq.J_j) > 0, "Mapping is not orientation-preserving!"
# %%
print("first Evs of dd0:", jnp.linalg.eigvalsh(Seq.M0 @ Seq.dd0)[:3])
print("first Evs of dd1:", jnp.linalg.eigvalsh(Seq.M1 @ Seq.dd1)[:3])
print("first Evs of dd2:", jnp.linalg.eigvalsh(Seq.M2 @ Seq.dd2)[:3])
print("first Evs of dd3:", jnp.linalg.eigvalsh(Seq.M3 @ Seq.dd3)[:3])
print("curl grad", jnp.max(jnp.abs(Seq.strong_curl @ Seq.strong_grad)))
print("div curl:", jnp.max(jnp.abs(Seq.strong_div @ Seq.strong_curl)))
# %%

u_hat = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)[1][:, 0]
u_h = DiscreteFunction(u_hat, Seq.Λ2, Seq.E2)


def norm_u_h(x):
    DFx = jax.jacfwd(F)(x)
    u_h_x = u_h(x)
    J = jnp.linalg.det(DFx)
    return ((DFx @ u_h_x) @ (DFx @ u_h_x))**0.5 / J


f_hat = jnp.linalg.solve(Seq.M0, Seq.P0(f_test))
f_h = Pushforward(DiscreteFunction(f_hat, Seq.Λ0, Seq.E0), F, 0)
# %%
cuts = jnp.linspace(0, 1/5, 5, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0,
                            ny=128, nz=128, z_min=0, z_max=1/3, invert_z=True)
# %%
fig, ax = plot_torus(norm_u_h, grids_pol, grid_surface,
                     gridlinewidth=1, cstride=8, elev=10,
                     azim=-70)
# %%
cuts = jnp.linspace(0, 1/3, 6, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
plot_crossections_separate(norm_u_h, grids_pol, cuts, plot_centerline=True)
# %%
