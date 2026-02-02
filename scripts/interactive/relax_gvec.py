# %%
import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.gvec_interface import interpolate_B_from_GVEC, interpolate_map_from_GVEC
from mrx.plotting import (
    get_2d_grids,
    plot_scalar_fct_physical_logical,
    plot_torus,
    plot_twin_axis,
)
from mrx.relaxation import (
    IntegrationScheme,
    MRXDiagnostics,
    TimeStepChoice,
    TimeStepper,
    relaxation_loop,
)

jax.config.update("jax_enable_x64", True)

# %%
gvec_eq = xr.open_dataset("../../data/gvec_stellarator.h5", engine="h5netcdf")
nfp = 3
# ---------- GVEC mapping ----------
resolutions = (4, 8, 4)
spline_degrees = (3, 3, 3)
quad_order = 5
mapSeq = DeRhamSequence(resolutions, spline_degrees, quad_order, ("clamped",
                        "periodic", "periodic"), lambda x: x, polar=False, dirichlet=False)
Phi, X1, X2 = interpolate_map_from_GVEC(gvec_eq, nfp, mapSeq)
Phi, X1, X2 = jax.jit(Phi), jax.jit(X1), jax.jit(X2)
# ---------- deRham sequence ----------
seq = DeRhamSequence(resolutions, spline_degrees, quad_order, ("clamped",
                     "periodic", "periodic"), Phi, polar=True, dirichlet=True)
seq.evaluate_1d()
seq.assemble_all()
seq.build_crossproduct_projections()
seq.assemble_leray_projection()
# ---------- Initial condition ----------
B_dof, residuals = interpolate_B_from_GVEC(gvec_eq, seq, Phi, nfp=nfp)
print(f"Interpolation residual: {residuals[0]:.2e}")
print(
    f"div B after interpolation: {((seq.strong_div @ B_dof) @ seq.M3 @ (seq.strong_div @ B_dof))**0.5: .2e}")
B_dof = seq.P_Leray @ B_dof
B_dof /= (B_dof @ seq.M2 @ B_dof)**0.5
# %%
# ---------- Relaxation ----------
num_iters_inner = 10
num_iters_outer = 50
ts = TimeStepper(seq=seq,
                 conjugate=False,
                 dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
                 timestep_mode=IntegrationScheme.EXPLICIT,
                 newton=False,
                 )

final_state, traces = relaxation_loop(B_dof,
                                      ts,
                                      num_iters_inner=num_iters_inner,
                                      num_iters_outer=num_iters_outer,
                                      dt0=1.0,
                                      )

# %%
fig = plot_twin_axis(
    left_y=jnp.array(traces["force_norm"]),
    right_y=jnp.abs((jnp.array(traces["helicity"]) -
                    traces["helicity"][0]) / traces["helicity"][0]),
    left_label='|| JxB - grad p || / || grad p ||',
    right_label='relative helicity change',
    left_log=True,
    right_log=False,
    num_iters_inner=num_iters_inner,
    left_marker='',
    right_marker='',
)
plt.show()

# %%
fig = plot_twin_axis(
    left_y=jnp.array(traces["force_norm"]),
    right_y=jnp.array(traces["timestep"]),
    left_label='|| JxB - grad p || / || grad p ||',
    right_label='dt',
    left_log=True,
    right_log=True,
    num_iters_inner=num_iters_inner,
    left_marker='',
    right_marker='',
)
plt.show()


# %%
fig = plot_twin_axis(
    left_y=jnp.array(traces["force_norm"]),
    right_y=(traces["energy"])[0] - jnp.array(traces["energy"]),
    left_label='|| JxB - grad p || / || grad p ||',
    right_label='Energy change',
    left_log=True,
    right_log=True,
    num_iters_inner=num_iters_inner,
    left_marker='',
    right_marker='',
)
plt.show()

# %%
cuts = jnp.linspace(0, 1, 4, endpoint=True)
grids_pol = [get_2d_grids(Phi, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
grid_surface = get_2d_grids(Phi, cut_axis=0, cut_value=1.0,
                            ny=128, nz=128, z_min=0, z_max=1, invert_z=True)
fig, ax = plot_torus(lambda x: 1-x[0], grids_pol, grid_surface,
                     gridlinewidth=1, cstride=8, noaxes=False, elev=90, azim=20)

# %%
diagnostics = MRXDiagnostics(seq)
state = final_state
B_dof = state.B_n
get_pressure = jax.jit(diagnostics.pressure)
p_dof = get_pressure(B_dof)
p_h = jax.jit(DiscreteFunction(p_dof, seq.Lambda_0, seq.E0))
fig = plot_scalar_fct_physical_logical(
    p_h, Phi, n_vis=64, logical_plane='r_theta', cbar_label="$p (\\times 10^2)$", cmap="berlin", fixed_zeta=0.3)

# %%


@jax.jit
def Phi_full_fp(x):
    r, θ, ζ = x  # now ζ ∈ [0, 1] should cover the FULL device
    π_nfp = 2 * jnp.pi / nfp
    # Decompose ζ into wedge index and local wedge coordinate
    ξ = ζ * nfp                 # in [0, nfp]
    k = jnp.floor(ξ)            # 0, 1, ..., nfp-1
    ζ_loc = ξ - k               # in [0, 1)

    # Build local x inside the wedge
    x_loc = jnp.array([r, θ, ζ_loc])

    # Geometry for a single wedge
    R = X1(x_loc)[0]
    Z = X2(x_loc)[0]

    # Local angle within wedge
    φ_wedge = π_nfp * ζ_loc     # 0 → 2π/nfp

    # Additional rotation to place the correct wedge
    φ_shift = 2 * jnp.pi * k / nfp

    φ = φ_wedge + φ_shift       # total toroidal angle

    return jnp.array([
        R * jnp.cos(φ),
        -R * jnp.sin(φ),
        Z
    ])


# %%
B_h = jax.jit(DiscreteFunction(state.B_n, seq.Lambda_2, seq.E2))


@jax.jit
def B_norm(x):
    x %= 1.0
    Bx = B_h(x)
    DFx = jax.jacfwd(Phi)(x)
    return jnp.linalg.norm(DFx @ Bx) / jnp.linalg.det(DFx)


# %%
fig = plot_scalar_fct_physical_logical(
    B_norm, Phi_full_fp, n_vis=64, logical_plane='tz', cbar_label="$|B|(r = 0.99)$", cmap="berlin", fixed_r=0.99)

# %%
cuts = jnp.linspace(0, 1, 4, endpoint=True)
grids_pol = [get_2d_grids(Phi_full_fp, cut_axis=2, cut_value=v,
                          nx=32, ny=32, nz=1) for v in cuts]
grid_surface = get_2d_grids(Phi_full_fp, cut_axis=0, cut_value=1.0,
                            ny=128, nz=128, z_min=0, z_max=1, invert_z=True)
fig, ax = plot_torus(p_h, grids_pol, grid_surface,
                     gridlinewidth=1, cstride=8, noaxes=False, elev=90, azim=40)
# %%


def vector_field(t, x, args):
    x %= 1.0
    Bx = B_h(x)
    DFx = jax.jacfwd(Phi)(x)
    return Bx / jnp.linalg.norm(DFx @ Bx)


def integrate_fieldline(f, x0, N, t1):
    t0 = 0.0
    sol = dfx.diffeqsolve(
        terms=dfx.ODETerm(f),
        solver=dfx.Dopri8(),
        t0=t0,
        t1=t1,
        dt0=0.05,
        y0=x0,
        saveat=dfx.SaveAt(ts=jnp.linspace(t0, t1, N)),
        stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-4),
        max_steps=100_000)
    return sol.ys


# %%
T = 1000
N = 10 * T
n_traj = 25
r_vals = jnp.linspace(0.01, 0.99, n_traj)
x0s = jnp.stack([r_vals,
                 0.5 * jnp.ones_like(r_vals),
                 0.5 * jnp.ones_like(r_vals)], axis=1)
# x0s_theta0 = jnp.stack([r_vals, jnp.zeros_like(r_vals),
#                        jnp.zeros_like(r_vals)], axis=1)
# x0s_theta05 = jnp.stack(
#     [r_vals, 0.5 * jnp.ones_like(r_vals), jnp.zeros_like(r_vals)], axis=1)
# x0s = jnp.concatenate([x0s_theta0, x0s_theta05], axis=0)
logical_trajectories = jax.vmap(
    lambda x0: integrate_fieldline(vector_field, x0, N, T)
)(x0s) % 1.0

# %%
physical_trajectories = jax.vmap(Phi_full_fp)(
    logical_trajectories.reshape(-1, 3)).reshape(n_traj, N, 3)

# %%
# 3D plot of a few field lines
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# for i in range(0, n_traj, 3):
#     traj = physical_trajectories[i]
#     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_zlim(-4, 4)
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# plt.show()

# %%


def intersect_with_plane_logical_periodic(traj, zeta_value, deg=2):
    """
    Plane defined by: ζ = constant, where ζ is periodic on [0,1]

    Parameters
    ----------
    traj : (N,3)
        Trajectory points in 3D.
    zeta_value : float
        ζ value of the plane (should be in [0,1]).
    deg : int
        Polynomial interpolation degree.
    """
    N = traj.shape[0]
    pad_size = N
    half = deg // 2

    zeta_points = traj[:, 2]

    # Handle periodic wrapping: compute shortest distance considering periodicity
    def periodic_distance(z1, z2):
        """Compute shortest distance between two points on [0,1] torus"""
        return (z2 - z1 + zeta_value + 0.1) % 1.0 - (zeta_value + 0.1) % 1.0

    # Compute signed distances considering periodicity
    s = periodic_distance(zeta_value, zeta_points)

    # Find sign changes (true crossings)
    flip_mask = s[:-1] * s[1:] < 0

    # Additional check: ensure we're not crossing due to large jumps
    zeta_diff = periodic_distance(zeta_points[:-1], zeta_points[1:])
    large_jump_mask = jnp.abs(zeta_diff) > 0.5  # Detect wrapping

    # Only count as crossing if not a large jump
    valid_flip_mask = flip_mask & ~large_jump_mask

    idxs = jnp.where(valid_flip_mask, jnp.arange(N - 1), N)
    idxs = jnp.sort(idxs)
    idxs = jnp.pad(idxs, (0, jnp.maximum(0, pad_size - idxs.size)),
                   constant_values=N)[:pad_size]

    def interp(i):
        valid = (i >= half) & (i < N - half)
        offset = jnp.arange(-half, deg - half + 1)
        idxs_local = jnp.clip(i + offset, 0, N - 1)
        pts_seg = traj[idxs_local]
        s_seg = s[idxs_local]
        t = jnp.arange(deg + 1, dtype=float)

        # fit polynomial s(t) ~ a t^deg + b t^{deg-1} + ... + c
        coeffs_s = jnp.polyfit(t, s_seg, deg=deg)
        roots = jnp.roots(coeffs_s, strip_zeros=False)
        roots_real = jnp.real(roots)
        cond = (jnp.abs(jnp.imag(roots)) <
                1e-8) & (roots_real > 0.0) & (roots_real < deg)
        t_cross = jnp.nanmin(jnp.where(cond, roots_real, jnp.nan))

        # fit each coordinate & evaluate at t_cross
        def eval_coord(y_seg):
            coeffs = jnp.polyfit(t, y_seg, deg=deg)
            return jnp.polyval(coeffs, t_cross)

        pt = jax.vmap(eval_coord)(pts_seg.T)
        return jnp.where(valid, pt, jnp.nan)

    pts = jax.vmap(interp)(idxs)
    return pts, idxs


# native Python loop - lax.scan not possible for now as padding width is a runtime value
res = [intersect_with_plane_logical_periodic(
    traj, zeta_value=0.5, deg=3) for traj in logical_trajectories]
logical_intersections = jnp.array([r[0] for r in res])
idxs = jnp.array([r[1] for r in res])
# %%


def toroidal_unwrapped(phi):
    phi_unwrapped = jnp.unwrap(phi)
    total_angle = phi_unwrapped[-1] - phi_unwrapped[0]
    return total_angle / (2 * jnp.pi)


def poloidal_unwrapped(R, Z, R_center=1.0, Z_center=0.0):
    θ = jnp.arctan2(Z - Z_center, R - R_center)
    θ_unwrapped = jnp.unwrap(θ)
    total_angle = θ_unwrapped[-1] - θ_unwrapped[0]
    return total_angle / (2 * jnp.pi)


@jax.jit
def get_iota(c, nfp):
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    R = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    r_mean = jnp.mean(R)
    z_mean = jnp.mean(z)
    m = poloidal_unwrapped(R, z, R_center=r_mean, Z_center=z_mean)
    n = toroidal_unwrapped(phi) / nfp
    return jnp.abs(m / n)

# %%


def uniformity_score(t_mod):
    # sort values
    ts = jnp.sort(t_mod)
    n = t_mod.size

    # empirical CDF minus ideal uniform CDF
    ecdf = jnp.arange(1, n+1) / n
    ucdf = ts  # uniform CDF value at ts

    # KS statistic
    ks = jnp.max(jnp.abs(ecdf - ucdf)) * (n**0.5)
    return ks


def classify_uniformity(t, ks_thresh=0.05):
    t_mod = (t % 1.0)
    ks = uniformity_score(t_mod)
    well_winding = ks < ks_thresh
    return well_winding, ks


@jax.jit
def get_iota_log(c, nfp, ks_thresh=0.05):
    t = jnp.unwrap(c[:, 1], period=1.0)
    z = jnp.unwrap(c[:, 2], period=1.0)
    total_t_angle = t[-1] - t[0]
    total_z_angle = z[-1] - z[0]
    iota = jnp.abs(total_t_angle / total_z_angle * nfp)
    # Uniformity test
    well_winding, ks = classify_uniformity(t, ks_thresh=ks_thresh)
    # set bad values to nan
    iota = jnp.where(well_winding, iota, jnp.nan)
    return iota, well_winding, ks


# %%
iotas, flags, ks = jax.vmap(lambda c: get_iota_log(
    c, nfp, ks_thresh=10.0))(logical_trajectories)
# iotas = jax.vmap(lambda c: get_iota(c, nfp))(physical_trajectories)
# iotas = jnp.clip(iotas, 0.0, 1.0)
# %%
mask = (~jnp.isnan(logical_intersections[..., 0])) & (
    logical_intersections[..., 2] < 0.5)
pts_log = logical_intersections[mask] % 1.0
pts_phys = jax.vmap(Phi)(pts_log)
p_vals = jax.vmap(p_h)(pts_log)
# Create iota_vals array where each point gets the iota value of its trajectory
traj_indices = jnp.arange(logical_intersections.shape[0])[:, None]
traj_indices_expanded = jnp.broadcast_to(
    traj_indices, logical_intersections.shape[:2])
iota_vals = iotas[traj_indices_expanded][mask]

# Separate points based on whether iota is NaN
valid_mask = ~jnp.isnan(iota_vals)
nan_mask = jnp.isnan(iota_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

# Plot valid points with color mapping
if jnp.any(valid_mask):
    s1 = ax1.scatter(  # physical
        (pts_phys[valid_mask, 0]**2 + pts_phys[valid_mask, 1]**2)**0.5,
        pts_phys[valid_mask, 2],
        c=iota_vals[valid_mask],
        cmap="berlin",
        s=0.25,
    )
    s2 = ax2.scatter(  # logical
        pts_log[valid_mask, 0], pts_log[valid_mask, 1],
        c=iota_vals[valid_mask],
        cmap="berlin",
        s=0.25,
    )

# Plot NaN points in grey
if jnp.any(nan_mask):
    ax1.scatter(  # physical
        (pts_phys[nan_mask, 0]**2 + pts_phys[nan_mask, 1]**2)**0.5,
        pts_phys[nan_mask, 2],
        c='grey',
        s=0.25,
    )
    ax2.scatter(  # logical
        pts_log[nan_mask, 0], pts_log[nan_mask, 1],
        c='grey',
        s=0.25,
    )

ax1.set(xlabel="R", ylabel="z", aspect="equal")
ax2.set(xlabel="r", ylabel="θ", aspect="equal")

# Only add colorbar if there are valid points
if jnp.any(valid_mask):
    cbar2 = fig.colorbar(s2, ax=ax2, label="iota", shrink=0.9)

    # Automatically determine rational ticks based on nfp and clipped iota range
    iota_min, iota_max = jnp.nanmin(
        iota_vals[valid_mask]), jnp.nanmax(iota_vals[valid_mask])
    rational_ticks = []
    rational_labels = []
    seen_rationals = set()
    for m in range(1, 20//nfp + 1):
        m_scaled = m * nfp
        for n in range(1, 20):  # reasonable range for denominators
            rational = m_scaled / n
            if iota_min <= rational <= iota_max and rational not in seen_rationals:
                rational_ticks.append(rational)
                g = jnp.gcd(m_scaled, n)
                rational_labels.append(f'{int(m_scaled//g)}/{int(n//g)}')
                seen_rationals.add(rational)
    if rational_ticks:
        cbar2.set_ticks(rational_ticks)
        cbar2.set_ticklabels(rational_labels)

plt.show()

# %%
p_avg = p_dof @ seq.P0(lambda x: jnp.ones(1)) / (seq.J_j @ seq.Q.w)
B_dof = state.B_n
beta = 2 * p_avg / (B_dof @ seq.M2 @ B_dof)
print(f"Beta = {beta:.3e}")
# %%
