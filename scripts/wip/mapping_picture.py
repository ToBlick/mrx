# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.BoundaryFitting import solovev_lcfs_fit
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Plotting import get_1d_grids, get_2d_grids, get_3d_grids

jax.config.update("jax_enable_x64", True)

# %%
R0 = 3.0
μ0 = 1.0
π = jnp.pi
k0 = 1.3
q0 = 1.5
F0 = 3
aR = 0.66

###
# ψ(R, Z) =  (¼ k₀² (R² - R₀²)² + R²Z² ) / (2 R₀² k₀ q₀)
###

p_map = 3
n_map = 8
q_map = 2 * p_map

a_hat = solovev_lcfs_fit(n_map, p_map, q_map, R0, a=0.6, k0=k0, q0=q0)

Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                         ("periodic", "constant", "constant"))
a_h = DiscreteFunction(a_hat, Λ_map)


def a(χ):
    """Radius as a function of chi."""
    return a_h(jnp.array([χ, 0, 0]))[0]


_x = jnp.linspace(0, 1, 1024)
plt.plot(_x, jax.vmap(a)(_x))

# %%
γ = 5/3

p = 3
q = 3*p
ns = (8, 8, 1)
ps = (3, 3, 0)
types = ("clamped", "periodic", "constant")


def _R(r, χ):
    return jnp.ones(1) * (R0 + a(χ) * r * jnp.cos(2 * π * χ))


def _Z(r, χ):
    return jnp.ones(1) * a(χ) * r * jnp.sin(2 * π * χ)


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array(
        [_R(r, χ) * jnp.cos(2 * π * z),
         -_R(r, χ) * jnp.sin(2 * π * z),
         _Z(r, χ)]))

# %%


_x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(F, zeta=0, nx=64)
_x_1d, _y_1d, (_y1_1d, _y2_1d, _y3_1d), (_x1_1d, _x2_1d,
                                         _x3_1d) = get_1d_grids(F, zeta=0, chi=0, nx=128)

# %%
nx = 8
ny = 8
nz = 8

_x, _F_x, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_3d_grids(
    F, x_min=0.1, x_max=0.6, y_min=0.0, y_max=0.3, z_min=0.5, z_max=0.55,
    nx=nx, ny=ny, nz=nz)

# %%
_F_x = _F_x.reshape(nx, ny, nz, 3)
_x = _x.reshape(nx, ny, nz, 3)
# %%

ax = plt.figure().add_subplot(projection='3d')
for j in range(nz):
    for i in range(nx):
        for y in [_F_x]:
            ax.plot(y[:, i, j, 0], y[:, i, j, 1], y[:, i, j, 2],
                    alpha=0.4, color='k', linewidth=1)
            ax.plot(y[i, :, j, 0], y[i, :, j, 1], y[i, :, j, 2],
                    alpha=0.4, color='k', linewidth=1)
            ax.plot(y[i, j, :, 0], y[i, j, :, 1], y[i, j, :, 2],
                    alpha=0.4, color='k', linewidth=1)

ax.set_axis_off()

x_val = (_F_x[-1, -1, 0, 0])
y_val = (_F_x[-1, -1, 0, 1])
z_val = (_F_x[-1, -1, 0, 2])
ax.margins(0)
ax.scatter([x_val], [y_val], [z_val], color="purple", s=20)

# add a label "(x, y, z)" near the point
ax.text(x_val + 0.1, y_val, z_val + 0.001, r"$\Phi(r, \chi, \zeta)$",
        fontsize=14, ha="left", va="bottom", color="purple")

ax.view_init(elev=30, azim=120, roll=0)

plt.savefig("script_outputs/physical_grid.pdf", bbox_inches='tight')
# %%
ax = plt.figure().add_subplot(projection='3d')
nj = 8
for j in range(nz):
    for i in range(nx):
        for y in [_x]:
            ax.plot(y[:, i, j, 0], y[:, i, j, 1], y[:, i, j, 2],
                    alpha=0.4, color='k', linewidth=1)
            ax.plot(y[i, :, j, 0], y[i, :, j, 1], y[i, :, j, 2],
                    alpha=0.4, color='k', linewidth=1)
            ax.plot(y[i, j, :, 0], y[i, j, :, 1], y[i, j, :, 2],
                    alpha=0.4, color='k', linewidth=1)

ax.set_axis_off()

x_val = (_x[0, -1, -1, 0])
y_val = (_x[0, -1, -1, 1])
z_val = (_x[0, -1, -1, 2])

ax.scatter([x_val], [y_val], [z_val], color="purple", s=20)
ax.margins(0)
# add a label "(x, y, z)" near the point
ax.text(x_val + 0.1, y_val, z_val + 0.001, r"$(r, \chi, \zeta)$",
        fontsize=10, ha="left", va="bottom", color="purple")

ax.view_init(elev=30, azim=120, roll=0)
plt.savefig("script_outputs/logical_grid.pdf", bbox_inches='tight')
# %%
