# %%
import os
import time
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from mrx.BoundaryFitting import helical_map
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Plotting import get_2d_grids, get_3d_grids, trajectory_plane_intersections

# --- Figure settings ---
FIG_SIZE = (12, 6)
FIG_SIZE_SQUARE = (8, 8)
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 16
LINE_WIDTH = 2.5
LEGEND_SIZE = 16

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

p = 2
n = 4

# Set up finite element spaces
q = 2*p
ns = (n, n, n)
ps = (p, p, p)
types = ("clamped", "periodic", "periodic")  # Types
# Domain parameters
π = jnp.pi
ɛ = 1/3
n_turns = 4  # Number of helix turns
h = 1/4      # radius of helix

# For the current construction to work, the tangent line must be parallel to Y at zeta = 0.
# Helical curve


F = helical_map(ɛ, h, n_turns)

# %%


def f(x):
    return jnp.ones(1)


# Create DeRham sequence
Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)
# Get stiffness matrix and projector

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
laplace_0 = Seq.assemble_gradgrad_0()                         # dim ker = 0
laplace_1 = Seq.assemble_curlcurl_0() - M1 @ grad @ weak_dvg  # dim ker = 0 (no voids)
laplace_2 = M2 @ curl @ weak_curl + \
    Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)
laplace_3 = - M3 @ dvg @ weak_grad  # dim ker = 1 (constants)

P0 = Seq.P0_0  # Projector for 0-forms


# %%
print("Smallest 3 eigenvalues of Laplace operators:")
print("0-forms (dim ker = 0):", jnp.linalg.eigvalsh(laplace_0)[:3])
print("1-forms (dim ker = 0):", jnp.linalg.eigvalsh(laplace_1)[:3])
print("2-forms (dim ker = 1):", jnp.linalg.eigvalsh(laplace_2)[:3])
print("3-forms (dim ker = 1):", jnp.linalg.eigvalsh(laplace_3)[:3])
# %%
# Solve the system
u_hat = jnp.linalg.solve(laplace_0, P0(f))
p_h = DiscreteFunction(u_hat, Seq.Λ0, Seq.E0_0.matrix())

# Plot solution
grids = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=32)
         for v in jnp.linspace(0, 1, 32, endpoint=False)]
# grids[i] = _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)
# %%


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    X_limits = ax.get_xlim3d()
    Y_limits = ax.get_ylim3d()
    Z_limits = ax.get_zlim3d()

    X_range = X_limits[1] - X_limits[0]
    Y_range = Y_limits[1] - Y_limits[0]
    Z_range = Z_limits[1] - Z_limits[0]
    max_range = max(X_range, Y_range, Z_range)

    X_mid = np.mean(X_limits)
    Y_mid = np.mean(Y_limits)
    Z_mid = np.mean(Z_limits)

    ax.set_xlim3d([X_mid - max_range/2, X_mid + max_range/2])
    ax.set_ylim3d([Y_mid - max_range/2, Y_mid + max_range/2])
    ax.set_zlim3d([Z_mid - max_range/2, Z_mid + max_range/2])


def plot_crossections(f, grids):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for grid in grids:
        X = grid[2][0]
        Y = grid[2][1]
        Z = grid[2][2]
        vals = jax.vmap(f)(grid[0]).reshape(X.shape)

        ax.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(
            (vals - vals.min())/(vals.max()-vals.min())), rstride=1, cstride=1, shade=False)
    set_axes_equal(ax)
    plt.tight_layout()
    return fig, ax


# %%
# fig, ax = plot_crossections(u_h, grids)
# %%
cuts = jnp.linspace(0, 1, 7)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v,
                          nx=16, ny=16, nz=1) for v in cuts]
grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0,
                            ny=64, nz=64, z_min=cuts[0], z_max=cuts[-1], invert_z=True)


def plot_torus(f, grids_pol, grid_surface):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    X = grid_surface[2][0]
    Y = grid_surface[2][1]
    Z = grid_surface[2][2]
    vals = jax.vmap(f)(grid_surface[0]).reshape(X.shape)
    colors = plt.cm.plasma(vals)
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, shade=False,
                    alpha=0.0, linewidth=0.05,)

    for grid in grids_pol:
        X = grid[2][0]
        Y = grid[2][1]
        Z = grid[2][2]
        vals = jax.vmap(f)(grid[0]).reshape(X.shape)
        colors = plt.cm.plasma((vals - vals.min()) / (vals.max() - vals.min()))
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1,
                        cstride=1, shade=False, zsort='min', linewidth=0)

    set_axes_equal(ax)
    plt.tight_layout()
    ax.view_init(elev=25, azim=130)
    ax.set_axis_off()
    return fig, ax


def plot_crossections_separate(f, grids):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()  # make it easy to index

    for i, (ax, grid) in enumerate(zip(axes, grids)):
        R = jnp.sqrt(grid[2][0]**2 + grid[2][1]**2)
        Z = grid[2][2]

        vals = jax.vmap(f)(grid[0]).reshape(R.shape)

        c = ax.contourf(R, Z, vals, 20, cmap='plasma')
        fig.colorbar(c, ax=ax)  # optional colorbar per panel
        ax.set_aspect('equal')  # 2D "equal axes"

    plt.tight_layout()
    return fig, axes


# %%
fig, ax = plot_torus(p_h, grids_pol, grid_surface)
# %%
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=32, ny=32,)
             for v in jnp.linspace(0, 1, 9, endpoint=False)]

# %%
fig, ax = plot_crossections_separate(p_h, grids_pol)
# %%


def trajectory_plane_intersections_list(trajectory, plane_point, plane_normal):
    """
    Compute intersections of a 3D trajectory with a general plane.

    Returns a list of intersection points (no NaNs, no masks).

    Parameters
    ----------
    trajectory : ndarray (T, 3)
    plane_point : ndarray (3,)
    plane_normal : ndarray (3,)

    Returns
    -------
    intersections : list of ndarray, each of shape (3,)
    """
    trajectory = np.asarray(trajectory)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)

    intersections = []

    for i in range(len(trajectory)-1):
        seg_start = trajectory[i]
        seg_end = trajectory[i+1]
        seg_vec = seg_end - seg_start

        denom = np.dot(plane_normal, seg_vec)
        if np.abs(denom) < 1e-12:  # parallel segment
            continue

        t = np.dot(plane_normal, plane_point - seg_start) / denom

        if 0 <= t <= 1:
            intersections.append(seg_start + t * seg_vec)

    return intersections

# %%


# @jax.jit
# def vector_field(t, p, args):
#     r, χ, z = p
#     r = jnp.clip(r, 1e-6, 1)
#     χ = χ % 1.0
#     z = z % 1.0
#     x = jnp.array([r, χ, z])
#     DFx = jax.jacfwd(F)(x)
#     norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
#     return B_h(x) / (norm + 1e-9)


# _x3d, _y3d, (_y3d_1, _y3d_2, _y3d_3), (_x3d_1, _x3d_2,
#                                        _x3d_3) = get_3d_grids(F, nx=64, ny=64, nz=64, x_min=1e-3)

# # Evaluate pushforward on the logical grid points (returns shape (N,3)).
# B_vals = jax.vmap(Pushforward(B_h, F, 2))(_x3d)
# B_vals = B_vals / jnp.linalg.norm(B_vals, axis=1, keepdims=True)


# def build_B_interpolator(_y1, _y2, _y3, B_vals_np, tol=1e-12, fill_value=0.0):
#     """Try to build a RegularGridInterpolator when physical coords are separable;
#     otherwise fall back to a scattered LinearNDInterpolator.

#     Returns (interp_callable, kind, axes) where kind is 'regular' or 'scattered'.
#     """
#     _y1 = np.asarray(_y1)
#     _y2 = np.asarray(_y2)
#     _y3 = np.asarray(_y3)
#     B_np = np.asarray(B_vals_np)

#     nx, ny, nz = _y1.shape

#     # separability checks: each slice along two axes must be constant along the third
#     ok1 = np.allclose(_y1, _y1[:, 0, 0][:, None, None], atol=tol)
#     ok2 = np.allclose(_y2, _y2[0, :, 0][None, :, None], atol=tol)
#     ok3 = np.allclose(_y3, _y3[0, 0, :][None, None, :], atol=tol)

#     if ok1 and ok2 and ok3:
#         # extract 1D axes from the 3D physical arrays
#         x_coords = _y1[:, 0, 0].copy()
#         y_coords = _y2[0, :, 0].copy()
#         z_coords = _y3[0, 0, :].copy()

#         # reshape values into (nx, ny, nz, 3)
#         vals = B_np.reshape(nx, ny, nz, 3).copy()

#         # ensure ascending order on each axis; flip values if necessary
#         if x_coords[0] > x_coords[-1]:
#             x_coords = x_coords[::-1]
#             vals = vals[::-1, :, :, :]
#         if y_coords[0] > y_coords[-1]:
#             y_coords = y_coords[::-1]
#             vals = vals[:, ::-1, :, :]
#         if z_coords[0] > z_coords[-1]:
#             z_coords = z_coords[::-1]
#             vals = vals[:, :, ::-1, :]

#         # build component-wise RegularGridInterpolator
#         Bx_interp = RegularGridInterpolator((x_coords, y_coords, z_coords), vals[..., 0],
#                                             bounds_error=False, fill_value=fill_value)
#         By_interp = RegularGridInterpolator((x_coords, y_coords, z_coords), vals[..., 1],
#                                             bounds_error=False, fill_value=fill_value)
#         Bz_interp = RegularGridInterpolator((x_coords, y_coords, z_coords), vals[..., 2],
#                                             bounds_error=False, fill_value=fill_value)

#         def B_interp_reg(pt):
#             return np.asarray([Bx_interp(pt), By_interp(pt), Bz_interp(pt)])

#         return B_interp_reg, 'regular', (x_coords, y_coords, z_coords)

#     # fallback: scattered interpolator on physical points
#     phys_pts = np.column_stack([_y1.ravel(), _y2.ravel(), _y3.ravel()])
#     B_interp_nd = LinearNDInterpolator(phys_pts, B_np, fill_value=fill_value)

#     def B_interp_scat(pt):
#         val = B_interp_nd(pt)
#         if val is None:
#             return np.zeros(3)
#         return np.asarray(val)

#     return B_interp_scat, 'scattered', None


# # Build interpolator and print which type was chosen. Use the numpy arrays
# B_interp_callable, interp_kind, interp_axes = build_B_interpolator(
#     _y3d_1, _y3d_2, _y3d_3, B_vals)
# print(f"Built B interpolator of kind='{interp_kind}'")

# # Quick validation: check interpolation reproduces a few grid values
# B_np = np.asarray(B_vals)
# N = B_np.shape[0]
# for test_idx in (0, N//2, N-1):
#     pt = np.asarray([_y3d_1.ravel()[test_idx], _y3d_2.ravel()
#                     [test_idx], _y3d_3.ravel()[test_idx]])
#     interp_val = B_interp_callable(pt)
#     stored = B_np[test_idx]
#     err = np.linalg.norm(interp_val - stored)
#     print(f"interp residual at sample index {test_idx}: {err:.3e}")


# def B_interp(x):
#     # wrapper keeping the same name used elsewhere; returns numpy array
#     return B_interp_callable(np.asarray(x))

# # parametrize a poloidal line:


# def poloidal_line(p1, p2, n=10, safety=1e-3):
#     # p1 is on the outside, p2 is halfway inside
#     t = jnp.linspace(0, 4 - 2 * safety, n)
#     return p1[None, :] + t[:, None] * (p2 - p1)[None, :]


# n_batch = 3
# n_total = n_batch
# safety = 1e-3
# x0s = poloidal_line(F(jnp.array([1-safety, 0.0, 0.0])),
#                     F(jnp.array([0.5, 0.0, 0.0])), n=n_total, safety=safety)

# n_cols = x0s.shape[1]
# cm = plt.cm.nipy_spectral
# vals = jnp.linspace(0, 1, n_cols + 2)[:-2]
# # Interleave from start and end
# # order = jnp.ravel(jnp.column_stack(
# #     [jnp.arange(n_cols//2), n_cols-1-jnp.arange(n_cols//2)]))
# # if n_cols % 2 == 1:
# #     order = jnp.append(order, n_cols//2)
# colors = cm(vals)


# def integrate_field_line(f, x0, t_span=(0, 10), dt=0.01, method='RK45', **kwargs):
#     x0 = np.asarray(x0)

#     def fun(t, x):
#         return np.asarray(f(x))

#     t_eval = np.arange(t_span[0], t_span[1], dt)

#     sol = solve_ivp(fun, t_span, x0, t_eval=t_eval, method=method, **kwargs)

#     return sol.y.T, sol.t


# field_lines = [integrate_field_line(B_interp, x0, t_span=(
#     0, 100 * 2 * jnp.pi), dt=0.001)[0] for x0 in x0s]
# intersections = [trajectory_plane_intersections_list(t, plane_point=np.array(
#     [0.0, 0.0, 0.1]), plane_normal=np.array([0.0, 1.0, 0.0])) for t in field_lines]


# def F_cyl(p):
#     x, y, z = F(p)
#     r = jnp.sqrt(x**2 + y**2)
#     phi = jnp.arctan2(y, x)
#     return jnp.array([r, phi, z])


# intersections_cylindrical = []
# for t in intersections:
#     if len(t) == 0:
#         continue
#     t = np.array(t)
#     intersections_cylindrical.append(jax.vmap(F_cyl)(t))

# fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
# for i, t in enumerate(intersections):
#     t = np.array(t)
#     if len(t) == 0:
#         continue
#     # Cycle through the defined colors
#     current_color = colors[i % len(colors)]

#     ax1.scatter(t[:, 0], t[:, 2], s=1, color=current_color)
#     ax1.set_xlabel(r'$x_1$', fontsize=LABEL_SIZE)
#     ax1.set_ylabel(r'$x_3$', fontsize=LABEL_SIZE)
# ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
# ax1.grid(True, linestyle="--", alpha=0.5)
# ax1.set_aspect('equal')
# plt.tight_layout()

# # %%


# intersections, mask = trajectory_plane_intersections(
#     field_lines, plane_val=0.1, axis=0)
# physical_intersections = jax.vmap(F)(intersections.reshape(-1, 3))
# physical_intersections = physical_intersections.reshape(
#     intersections.shape[0], intersections.shape[1], 3)
# fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
# for i, t in enumerate(physical_intersections):
#     # Cycle through the defined colors
#     current_color = colors[i % len(colors)]
#     ax1.scatter(t[:, 0], t[:, 2], s=1, color=current_color)
#     ax1.set_xlabel(r'$x_1$', fontsize=LABEL_SIZE)
#     ax1.set_ylabel(r'$x_3$', fontsize=LABEL_SIZE)
# ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
# ax1.grid(True, linestyle="--", alpha=0.5)
# ax1.set_aspect('equal')
# plt.tight_layout()


# # %%


# # --- Figure settings ---
# FIG_SIZE = (12, 6)
# FIG_SIZE_SQUARE = (8, 8)
# TITLE_SIZE = 20
# LABEL_SIZE = 20
# TICK_SIZE = 16
# LINE_WIDTH = 2.5
# LEGEND_SIZE = 16

# final_time = 100 * 2 * jnp.pi
# n_saves = int(final_time * 100)
# max_steps = 100_000
# r_tol = 1e-7
# a_tol = 1e-7

# term = diffrax.ODETerm(vector_field)
# solver = diffrax.Dopri5()
# saveat = diffrax.SaveAt(ts=jnp.linspace(0, final_time, n_saves))
# stepsize_controller = diffrax.PIDController(rtol=r_tol, atol=a_tol)
# trajectories = []


# # Compute trajectories
# print("Integrating field lines...")
# for x in x0s:
#     trajectories.append(jax.vmap(lambda x0: diffrax.diffeqsolve(term, solver,
#                                                                 t0=0, t1=final_time, dt0=None,
#                                                                 y0=x0,
#                                                                 max_steps=max_steps,
#                                                                 saveat=saveat, stepsize_controller=stepsize_controller).ys)(x))
# trajectories = jnp.array(trajectories).reshape(
#     n_batch * n_loop, n_saves, 3) % 1
# physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
# physical_trajectories = physical_trajectories.reshape(
#     trajectories.shape[0], trajectories.shape[1], 3)


# def F_cyl(p):
#     x, y, z = F(p)
#     r = jnp.sqrt(x**2 + y**2)
#     phi = jnp.arctan2(y, x) / (2 * jnp.pi) + 0.5
#     return jnp.array([r, phi, z])


# cylindrical_trajectories = jax.vmap(F_cyl)(trajectories.reshape(-1, 3))
# cylindrical_trajectories = cylindrical_trajectories.reshape(
#     trajectories.shape[0], trajectories.shape[1], 3)

# # %%
# for plane_val in [0.5]:
#     intersections, mask = trajectory_plane_intersections(
#         trajectories, plane_val=plane_val, axis=2)
#     physical_intersections = jax.vmap(F)(intersections.reshape(-1, 3))
#     physical_intersections = physical_intersections.reshape(
#         intersections.shape[0], intersections.shape[1], 3)
#     fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
#     for i, t in enumerate(physical_intersections):
#         # Cycle through the defined colors
#         current_color = colors[i % len(colors)]
#         ax1.scatter(t[:, 0], t[:, 2], s=1, color=current_color)
#         ax1.set_xlabel(r'$x_1$', fontsize=LABEL_SIZE)
#         ax1.set_ylabel(r'$x_3$', fontsize=LABEL_SIZE)
#     ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
#     ax1.grid(True, linestyle="--", alpha=0.5)
#     ax1.set_aspect('equal')
#     plt.tight_layout()
# # %%
# for plane_val in [0.2, 0.4, 0.6, 0.8]:
#     intersections, mask = trajectory_plane_intersections(
#         trajectories, plane_val=plane_val, axis=2)
#     fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
#     for i, t in enumerate(intersections):
#         # Cycle through the defined colors
#         current_color = colors[i % len(colors)]
#         ax1.scatter(t[:, 0], t[:, 1], s=1, color=current_color)
#         ax1.set_xlabel(r'$r$', fontsize=LABEL_SIZE)
#         ax1.set_ylabel(r'$\theta$', fontsize=LABEL_SIZE)
#     ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
#     ax1.grid(True, linestyle="--", alpha=0.5)
#     ax1.set_aspect('equal')
#     plt.tight_layout()
# # %%
# for plane_val in [0.2, 0.4, 0.6, 0.8]:
#     intersections, mask = trajectory_plane_intersections(
#         cylindrical_trajectories, plane_val=plane_val, axis=1)
#     fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
#     for i, t in enumerate(intersections):
#         # Cycle through the defined colors
#         current_color = colors[i % len(colors)]
#         ax1.scatter(t[:, 0] - jnp.where(t[:, 0] < 0, 1, 0),
#                     t[:, 2], s=1, color=current_color)
#         ax1.set_xlabel(r'$R$', fontsize=LABEL_SIZE)
#         ax1.set_ylabel(r'$Z$', fontsize=LABEL_SIZE)
#     ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
#     ax1.grid(True, linestyle="--", alpha=0.5)
#     ax1.set_aspect('equal')
#     plt.tight_layout()
# # %%
# t = field_lines[5][0]
# fig = plt.figure(figsize=FIG_SIZE_SQUARE)
# ax = fig.add_subplot(projection='3d')
# ax.plot(t[:, 0], t[:, 1], t[:, 2],
#         color="purple",
#         alpha=1)
# ax.set_axis_off()
# plt.tight_layout()

# # %%
