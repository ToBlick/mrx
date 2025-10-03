# %%
import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from mrx.BoundaryFitting import cerfon_map, helical_map, rotating_ellipse_map
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward
from mrx.Plotting import (
    get_2d_grids,
    plot_crossections,
    plot_crossections_separate,
    plot_torus,
    set_axes_equal,
)

# %%
name = "HELIX_precond"
with h5py.File("../../script_outputs/solovev/" + name + ".h5", "r") as f:
    B_hat = f["B_final"][:]
    p_hat = f["p_final"][:]
    helicity_trace = f["helicity_trace"][:]
    energy_trace = f["energy_trace"][:]
    force_trace = f["force_trace"][:]

    CONFIG = {k: v for k, v in f["config"].attrs.items()}
    # decode strings back if needed
    CONFIG = {k: v.decode() if isinstance(v, bytes)
              else v for k, v in CONFIG.items()}
# %%
# Get the map and sequences back:
delta = CONFIG["delta"]
kappa = CONFIG["kappa"]
q_star = CONFIG["q_star"]
eps = CONFIG["eps"]
alpha = jnp.arcsin(delta)
tau = q_star * eps * kappa * (1 + kappa**2) / (kappa + 1)
π = jnp.pi
gamma = CONFIG["gamma"]

if CONFIG["type"] == "tokamak":
    F = cerfon_map(eps, kappa, alpha)
elif CONFIG["type"] == "helix":
    F = helical_map(eps, CONFIG["h_helix"], CONFIG["m_helix"])
elif CONFIG["type"] == "rotating_ellipse":
    F = rotating_ellipse_map(eps, CONFIG["kappa"], CONFIG["m_rot"])
else:
    raise ValueError("Unknown configuration type.")
ns = (CONFIG["n_r"], CONFIG["n_theta"], CONFIG["n_zeta"])
ps = (CONFIG["p_r"], CONFIG["p_theta"], 0
      if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
q = max(ps)
types = ("clamped", "periodic",
         "constant" if CONFIG["n_zeta"] == 1 else "periodic")
print("Setting up FEM spaces...")
Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

# %%
B_h = jax.jit(DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix()))
# %%
# at zeta = 0, we are in the (R, z) plane, so we sample there:
n_lines = 16  # even numbers only
_r = np.linspace(0.05, 0.95, n_lines)
x0s = np.vstack(
    (np.hstack((_r[::2], _r[-1::-2])),                              # between 0 and 1
     # half go to theta=0 and half to theta=pi
     np.hstack((np.zeros(n_lines//2), 0.5 * np.ones(n_lines//2))),
     np.zeros(n_lines))
).T
# %%


def vector_field(x):
    """Return the norm of B at x=(r,theta,zeta) in [0,1]^3."""
    r, θ, z = x
    r = jnp.clip(r, 1e-6, 1)  # avoid r=0 in polar coords
    θ = θ % 1.0
    z = z % 1.0
    x = jnp.array([r, θ, z])
    Bx = B_h(jnp.array(x))
    DFx = jax.jacfwd(F)(jnp.array(x))
    return Bx / jnp.linalg.norm(DFx @ Bx)


def integrate_field_line(v, x0, t_span=(0, 1000), n_saves=10000, max_step=0.05, rtol=1e-6, atol=1e-9):

    if n_saves is None:
        n_saves = 100 * t_span[1]
    t_eval = np.linspace(t_span[0], t_span[1], n_saves)

    def rhs(t, x):
        # apply periodicity
        r, θ, z = x
        r = jnp.clip(r, 1e-6, 1)  # avoid r=0 in polar coords
        θ = θ % 1.0
        z = z % 1.0

        vec = v(np.array([r, θ, z]))
        return vec / jnp.linalg.norm(vec)

    sol = solve_ivp(
        rhs, t_span, np.array(x0, dtype=float),
        method="RK45", max_step=max_step, rtol=rtol, atol=atol, t_eval=t_eval
    )
    return sol.y % 1


# %%
field_lines = [integrate_field_line(
    vector_field, x0, t_span=(0, 100), n_saves=None) for x0 in x0s]
# %%
physical_field_lines = [
    np.array(jax.vmap(F, in_axes=1, out_axes=1)(x)) for x in field_lines]
# %%


def fieldline_phi_plane_crossings(ys, phi0):
    """
    Find points where a field line crosses the toroidal plane phi = phi0.

    Parameters
    ----------
    ys : np.ndarray
        Array of shape (3, N) representing the field line in Cartesian coordinates (x, y, z).
    phi0 : float
        The toroidal angle (in radians) of the plane to intersect.

    Returns
    -------
    crossings : list of (R, z)
        List of (R, z) points where the field line intersects the plane.
    """
    x, y, z = ys

    # Compute toroidal angle phi along trajectory (principal value in [-pi, pi))
    phi = np.arctan2(y, x)

    # Unwrap phi to a continuous curve to avoid branch-cut artifacts near ±pi.
    phi_unwrapped = np.unwrap(phi)

    # Look for crossings of phi = phi0 + 2π*k. Compute how many integer
    # multiples of 2π the unwrapped angle has passed between consecutive
    # samples. Each integer jump corresponds to a crossing of the target
    # toroidal plane; interpolate each such crossing accurately.
    q = (phi_unwrapped - phi0) / (2 * np.pi)
    floor_q = np.floor(q).astype(int)
    dq = floor_q[1:] - floor_q[:-1]

    crossings = []
    for i, delta in enumerate(dq):
        if delta == 0:
            # Check exact sample-on-plane (rare) at index i
            if np.isclose(phi_unwrapped[i] - phi0 - 2 * np.pi * floor_q[i], 0.0, atol=1e-12):
                R_c = np.hypot(x[i], y[i])
                crossings.append((R_c, z[i]))
            continue

        # One or more integer crossings occurred between i and i+1.
        # Handle forward (delta>0) and backward (delta<0) motion.
        step = 1 if delta > 0 else -1
        for s in range(abs(delta)):
            target_k = floor_q[i] + (s + 1) * step
            target_phi = phi0 + 2 * np.pi * target_k

            denom = (phi_unwrapped[i + 1] - phi_unwrapped[i])
            if np.isclose(denom, 0.0):
                # Degenerate step — fall back to midpoint
                t = 0.5
            else:
                t = (target_phi - phi_unwrapped[i]) / denom

            # Clamp interpolation parameter to [0,1]
            t = max(0.0, min(1.0, t))

            x_c = x[i] + t * (x[i + 1] - x[i])
            y_c = y[i] + t * (y[i + 1] - y[i])
            z_c = z[i] + t * (z[i + 1] - z[i])

            R_c = np.hypot(x_c, y_c)
            crossings.append((R_c, z_c))

    return np.array(crossings)


# %%
phi_0 = 0.35  # toroidal angle of the plane to intersect
crossings = [fieldline_phi_plane_crossings(
    x, phi0=phi_0) for x in physical_field_lines]

fig, ax = plt.subplots()
for x in crossings:
    if x.size == 0:
        continue
    ax.scatter(x[:, 0], x[:, 1], lw=1, s=1)
ax.set_title(f"poincare plot at phi = {phi_0}")
ax.set_xlabel("R")
ax.set_ylabel("z")
ax.set_aspect('equal', adjustable='box')
plt.show()
# %%
# Plot a field line in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for x in physical_field_lines[3:6]:
    ax.plot(*x, lw=0.1)
set_axes_equal(ax)
ax.set_title("Magnetic field lines")
plt.show()
# %%
plt.plot(field_lines[0][:, 0], label="r")
plt.plot(field_lines[0][:, 1], label="θ")
plt.plot(field_lines[0][:, 2], label="z")
plt.legend()
