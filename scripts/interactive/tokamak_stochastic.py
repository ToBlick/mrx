# %%
import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.gvec_interface import interpolate_B_from_GVEC, interpolate_map_from_GVEC
from mrx.mappings import approx_inverse_map, invert_map, rotating_ellipse_map
from mrx.plotting import (
    get_2d_grids,
    intersect_with_plane,
    plot_scalar_fct_physical_logical,
    plot_torus,
    plot_twin_axis,
)
from mrx.relaxation import (
    IntegrationScheme,
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
    f"div B after interpolation: {((seq.strong_div @ B_dof) @ seq.M3 @ (seq.strong_div @ B_dof))**0.5:.2e}")
B_dof = seq.P_Leray @ B_dof
B_dof /= (B_dof @ seq.M2 @ B_dof)**0.5
# %%
# ---------- Relaxation ----------
num_iters_inner = 100
num_iters_outer = 100
timestepper = TimeStepper(seq=seq,
                          conjugate=True,
                          dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
                          timestep_mode=IntegrationScheme.EXPLICIT,
                          )

final_state, traces = relaxation_loop(B_dof,
                                      timestepper,
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
    right_log=False,
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
# F, _, _ = timestepper.compute_force(B_dof)
# state = State(B_n=B_dof,
#               dt=1.0,
#               v=F,
#               F_norm=timestepper.norm_2(F),
#               v_norm=timestepper.norm_2(F))
# force_trace = [state.F_norm]
# helicity_trace = [get_helicity(state.B_n)]
# timesteps = [state.dt]
# energy_trace = [state.B_n @ seq.M2 @ state.B_n/2]
# print(f"Initial force error: {force_trace[-1]:.2e}")
# print(f"Initial helicity: {helicity_trace[-1]:.2e}")
# print(f"Initial timestep: {timesteps[-1]:.2e}")
# # %%
# p_dof = get_pressure(B_dof)
# p_avg = p_dof @ seq.P0(lambda x: jnp.ones(1)) / (seq.J_j @ seq.Q.w)
# beta = 2 * p_avg / (B_dof @ seq.M2 @ B_dof)
# print(f"Beta = {beta:.3e}")


# def body_fn(state, key):
#     # ---- one state update ----
#     state = timestepper.relaxation_step(state, key)
#     failed = (state.picard_residuum > timestepper.picard_tol) | (
#         ~jnp.isfinite(state.picard_residuum))

#     def on_fail(state):
#         state = timestepper.update_field(state, "dt", state.dt / 2)
#         state = timestepper.update_field(
#             state, "B_nplus1", state.B_n)  # restart with halved dt
#         return state

#     def on_success(state):
#         state = timestepper.update_field(state, "B_n", state.B_nplus1)
#         if timestepper.dt_mode == 'from_state':
#             dt_new = jnp.where(state.picard_iterations < 4,
#                                state.dt * 1.01,   # few iterations → increase dt
#                                state.dt / 1.01)   # many iterations → decrease dt
#             state = timestepper.update_field(state, "dt", dt_new)
#         return state
#     state = jax.lax.cond(failed, on_fail, on_success, state)
#     return state, None


# # %%
# num_iters_inner = 10
# num_iters_outer = 10
# # %%
# NOISE_AMP = 0.5
# NOISE_MAX = 0.33
# NOISE_DECAY = 0.33


# def noise_level(i):
#     tau = i / num_iters_outer
#     return NOISE_AMP * tau / NOISE_MAX * \
#         jnp.exp(1 - tau / (NOISE_DECAY * NOISE_MAX))


# noise_schedule = [0.0]
# for i in range(num_iters_outer + 1):
#     key, _ = jax.random.split(key)
#     state = timestepper.update_field(state, "noise_level", noise_level(i))
#     state, _ = jax.lax.scan(
#         body_fn, state, jax.random.split(key, num_iters_inner))

#     # state = jax.lax.fori_loop(0, num_iters_inner, wrap_body_fn, state)
#     force_trace.append(state.F_norm)
#     helicity_trace.append(get_helicity(state.B_n))
#     timesteps.append(state.dt)
#     noise_schedule.append(state.noise_level)
#     energy_trace.append(state.B_n @ seq.M2 @ state.B_n/2)
#     print(
#         f"Iteration {i * num_iters_inner}, force norm: {state.F_norm:.2e}")

# %%
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


T = 1000
N = 10 * T
n_traj = 16
r_vals = jnp.linspace(0.01, 0.99, n_traj)
x0s = jnp.stack([r_vals, jnp.zeros_like(r_vals),
                 jnp.zeros_like(r_vals)], axis=1)
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
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
for i in range(0, n_traj, 3):
    traj = physical_trajectories[i]
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(-4, 4)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
plt.show()

# %%
# TODO: Compute intersections with logical plane


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
        diff = z2 - z1
        # Wrap to [-0.5, 0.5] range
        return jnp.where(jnp.abs(diff) > 0.5,
                         diff - jnp.sign(diff),
                         diff)

    # Compute signed distances considering periodicity
    s = jnp.array([periodic_distance(zeta_value, z) for z in zeta_points])

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


logical_intersections, idxs = jax.vmap(lambda t: intersect_with_plane_logical_periodic(
    t, zeta_value=0.33, deg=5))(logical_trajectories)

# %%
# logical_intersections = jax.vmap(
#     lambda pt: invert_map(
#         Phi_full_fp, pt, lambda x: approx_inverse_map(x, eps=0.5, R0=3.0))
# )(intersections.reshape(-1, 3)).reshape(intersections.shape) % 1

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
    x = c[:, 0]
    y = c[:, 1]
    z = c[:, 2]
    R = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    r_mean = jnp.mean(R)
    z_mean = jnp.mean(z)
    m = poloidal_unwrapped(R, z, R_center=r_mean, Z_center=z_mean)
    n = toroidal_unwrapped(phi) / nfp
    return jnp.abs(m / n), m, n


@jax.jit
def get_iota_log(c, nfp):
    t = jnp.unwrap(c[:, 1], period=1.0)
    total_t_angle = t[-1] - t[0]
    z = jnp.unwrap(c[:, 2], period=1.0)
    total_z_angle = z[-1] - z[0]
    return jnp.abs(total_t_angle / total_z_angle * nfp)


iotas = jax.vmap(lambda c: get_iota_log(c, nfp))(logical_trajectories)

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
# Clip iota to 5-95 percentiles so outliers get boundary colors
p5, p95 = jnp.percentile(iota_vals, jnp.array([5.0, 95.0]))
# avoid degenerate vmin==vmax
p5 = float(p5)
p95 = float(p95)
if p95 <= p5:
    p5 -= 1e-8
    p95 += 1e-8

iota_clipped = jnp.clip(iota_vals, p5, p95)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
s1 = ax1.scatter(  # physical
    (pts_phys[:, 0]**2 + pts_phys[:, 1]**2)**0.5, pts_phys[:, 2],
    c=iota_clipped,
    cmap="berlin",
    s=0.25,
)
s2 = ax2.scatter(  # logical
    pts_log[:, 0], pts_log[:, 1],
    c=iota_clipped,
    cmap="berlin",
    norm=plt.Normalize(vmin=p5, vmax=p95),
    s=0.25,
)
ax1.set(xlabel="R", ylabel="z", aspect="equal")
ax2.set(xlabel="r", ylabel="θ", aspect="equal")
# fig.colorbar(s1, ax=ax1, label="p (×100)", shrink=0.9)
cbar2 = fig.colorbar(s2, ax=ax2, label="iota", shrink=0.9)

# Automatically determine rational ticks based on nfp and clipped iota range
iota_min, iota_max = p5, p95
rational_ticks = []
rational_labels = []
seen_rationals = set()
for m in range(1, 16//nfp + 1):
    m_scaled = m * nfp
    for n in range(1, 16):  # reasonable range for denominators
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
print(f"p_avg = {p_avg:.3e}")

# %%
B_dof = state.B_n
beta = 2 * p_avg / (B_dof @ seq.M2 @ B_dof)
print(f"Beta = {beta:.3e}")
# %%
