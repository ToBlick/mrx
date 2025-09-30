# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale so spheres look like spheres, etc."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# Helical curve
R, r, n = 1.0, 0.2, 3  # example
theta = np.linspace(0, 2*np.pi, 1000)
x = (R + r*np.cos(n*theta))*np.cos(theta)
y = (R + r*np.cos(n*theta))*np.sin(theta)
z = r*np.sin(n*theta)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
set_axes_equal(ax)
plt.show()

# %%


# centerline: replace with your X(theta)
def X(theta, R=1.0, r=0.2, n=3):
    return np.vstack([
        (R + r*np.cos(n*theta))*np.cos(theta),
        (R + r*np.cos(n*theta))*np.sin(theta),
        r*np.sin(n*theta)
    ]).T


def compute_frenet_frame(X_pts, theta):
    # numeric derivatives (periodic)
    d = np.gradient(X_pts, theta, axis=0)
    dd = np.gradient(d, theta, axis=0)
    T = d / np.linalg.norm(d, axis=1)[:, None]
    # Frenet normal (avoid divide-by-zero if curvature ~ 0)
    # N ~ normalized projection of dd onto normal plane
    curv_vec = dd
    N = curv_vec - (np.sum(curv_vec * T, axis=1)[:, None]) * T
    N /= np.linalg.norm(N, axis=1)[:, None] + 1e-12
    B = np.cross(T, N)
    # ensure right-handed and normalized
    B /= np.linalg.norm(B, axis=1)[:, None] + 1e-12
    return T, N, B


def make_tube(R=1.0, r=0.2, n=3, rho=0.25, Nth=400, Nph=80):
    theta = np.linspace(0, 2*np.pi, Nth, endpoint=False)
    X_pts = X(theta, R=R, r=r, n=n)
    T, N, B = compute_frenet_frame(X_pts, theta)
    phi = np.linspace(0, 2*np.pi, Nph, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    # Parametric surface
    surf = (X_pts[:, None, :]
            + rho*(N[:, None, :]*np.cos(PH[:, :, None]) + B[:, None, :]*np.sin(PH[:, :, None])))
    return surf, theta, phi


# Example plot
surf, theta, phi = make_tube(R=1.0, r=0.2, n=3, rho=0.18, Nth=400, Nph=80)
Xsurf = surf[:, :, 0]
Ysurf = surf[:, :, 1]
Zsurf = surf[:, :, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xsurf, Ysurf, Zsurf, rstride=1,
                cstride=1, linewidth=0, antialiased=False)
ax.set_box_aspect([1, 1, 1])
plt.show()

# %%
