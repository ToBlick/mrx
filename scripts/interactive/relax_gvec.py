# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr

# %%
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.gvec_interface import load_and_reshape_GVEC
from mrx.io import interpolate_B, interpolate_map_from_points
from mrx.mappings import extend_map_nfp
from mrx.plotting import (get_2d_grids, integrate_fieldline,
                          plot_scalar_fct_physical_logical, plot_torus,
                          plot_twin_axis, poincare_plot)
from mrx.relaxation import (DescentMethod, IntegrationScheme, MRXDiagnostics,
                            TimeStepChoice, TimeStepper, relaxation_loop)

jax.config.update("jax_enable_x64", True)

# %%
gvec_eq = xr.open_dataset("data/gvec_stellarator.h5", engine="h5netcdf")
nfp = 3
# %%
pts, R, Z, B_vals = load_and_reshape_GVEC(gvec_eq, nfp)

ns_map = (4, 4, 4)
ps_map = (3, 3, 3)
quad_order_map = 3

ns = (8, 16, 8)
ps = (6, 6, 6)
quad_order = 5

map, R_dof, Z_dof, resid_R, resid_Z = interpolate_map_from_points(
    pts, R, Z, nfp, ns=ns_map, ps=ps_map, quad_order=quad_order_map)
map = jax.jit(map)
print(f"Map interpolation residuals: R={resid_R:.2e}, Z={resid_Z:.2e}")
seq = DeRhamSequence(ns, ps, quad_order, ("clamped", "periodic", "periodic"),
                     map, polar=True, dirichlet=True)
seq.evaluate_1d()
seq.assemble_all()
seq.build_crossproduct_projections()
seq.assemble_leray_projection()
# %%
# set aside  points for validation using strided sampling
val_stride = 3
exclude_axis_tol = 1e-3
val_mask = (jnp.arange(pts.shape[0]) % val_stride == 0) & (
    pts[:, 0] > exclude_axis_tol) & (pts[:, 0] < 1 - exclude_axis_tol)
train_mask = ~val_mask
val_pts = pts[val_mask]
train_pts = pts[train_mask]
train_B_vals = B_vals[train_mask]
B_dof_0, resid_B = interpolate_B(
    train_pts, train_B_vals, seq, exclude_axis_tol=exclude_axis_tol)
print(f"B-field interpolation residual (train): {resid_B[0]:.2e}")
# %%
# Validate interpolation
B_h = jax.jit(Pushforward(DiscreteFunction(
    B_dof_0, seq.basis_2, seq.e2), seq.map, 2))
B_val_interp = jax.vmap(B_h)(val_pts)
val_error = jnp.linalg.norm(B_vals[val_mask] - B_val_interp, axis=1)
val_rel_error = val_error / jnp.linalg.norm(B_vals[val_mask], axis=1)
print(
    f"B-field interpolation relative error (validation): mean={jnp.mean(val_rel_error):.2e}, max={jnp.max(val_rel_error):.2e}")
print(
    f"div B after interpolation: {((seq.strong_div @ B_dof_0) @ seq.m3 @ (seq.strong_div @ B_dof_0))**0.5: .2e}")

# %%
B_dof_0 = seq.P_Leray @ B_dof_0
B_dof_0 /= (B_dof_0 @ seq.m2 @ B_dof_0)**0.5
# %%
B_dof = B_dof_0.copy()
# %%
# ---------- Relaxation ----------
num_iters_inner = 10
num_iters_outer = 100
ts = TimeStepper(
    seq=seq,
    descent_method=DescentMethod.CONJUGATE_GRADIENT,
    dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
    timestep_mode=IntegrationScheme.EXPLICIT,
)

# %%


def eta_schedule(iter_outer):
    return 1e-6 * 0.5 * (1 - jnp.tanh(4 * jnp.pi * (iter_outer / num_iters_outer - 0.5)))


plt.plot(jnp.array([eta_schedule(i) for i in range(num_iters_outer)]))
plt.xlabel("Outer iteration")
plt.ylabel(r"$\eta$ schedule")
plt.title("Resistivity schedule")
plt.grid()
# %%

final_state, traces = relaxation_loop(
    B_dof,
    ts,
    num_iters_inner=num_iters_inner,
    num_iters_outer=num_iters_outer,
    dt0=1.0,
    force_tolerance=1e-9,
    resistivity_schedule=eta_schedule,
)


# %%
# ---------- Plots ----------
fig = plot_twin_axis(
    left_y=jnp.array(traces["force_norm"]),
    right_y=jnp.abs(
        (jnp.array(traces["helicity"]) -
         traces["helicity"][0]) / traces["helicity"][0]
    ),
    left_label="|| JxB - grad p || / || grad p ||",
    right_label="relative helicity change",
    left_log=True,
    right_log=False,
    num_iters_inner=num_iters_inner,
    left_marker="",
    right_marker="",
)
plt.show()

# %%
fig = plot_twin_axis(
    left_y=jnp.array(traces["force_norm"]),
    right_y=jnp.array(traces["timestep"]),
    left_label="|| JxB - grad p || / || grad p ||",
    right_label="dt",
    left_log=True,
    right_log=False,
    num_iters_inner=num_iters_inner,
    left_marker="",
    right_marker="",
)
plt.show()

# %%
fig = plot_twin_axis(
    left_y=jnp.array(traces["force_norm"]),
    right_y=(traces["energy"])[0] - jnp.array(traces["energy"]),
    left_label="|| JxB - grad p || / || grad p ||",
    right_label="Energy change",
    left_log=True,
    right_log=True,
    num_iters_inner=num_iters_inner,
    left_marker="",
    right_marker="",
)
plt.show()

# %%
fig = plot_twin_axis(
    left_y=jnp.array(traces["force_norm"]),
    right_y=jnp.array([eta_schedule(i) for i in range(num_iters_outer)]),
    left_label="|| JxB - grad p || / || grad p ||",
    right_label="Resistivity",
    left_log=True,
    right_log=True,
    num_iters_inner=num_iters_inner,
    left_marker="",
    right_marker="",
)
plt.show()

# %%
diagnostics = MRXDiagnostics(seq)
# state = final_state
B_dof = final_state.B_n
get_pressure = jax.jit(diagnostics.pressure)
p_dof = get_pressure(B_dof)
p_h = jax.jit(Pushforward(DiscreteFunction(
    p_dof, seq.basis_0, seq.e0), seq.map, 0))

# %%
# J_hat = seq.weak_curl @ B_dof
# H_hat = seq.P12 @ B_dof
# JxH_hat = jnp.linalg.solve(
#                 seq.M2, seq.P1x1_to_2(J_hat, H_hat))
# p_dof = -jnp.linalg.solve(seq.dd3, seq.strong_div @ JxH_hat)
# p_h = jax.jit(Pushforward(DiscreteFunction(p_dof, seq.Lambda_3, seq.E3), seq.F, 3))
# %%
fig = plot_scalar_fct_physical_logical(
    p_h,
    seq.map,
    n_vis=64,
    logical_plane="r_theta",
    cbar_label="$p$",
    cmap="berlin",
    fixed_zeta=0.33,
    avoid_boundary=1e-2
)

# %%
J_xyz = jax.jit(Pushforward(DiscreteFunction(
    seq.weak_curl @ B_dof, seq.basis_1, seq.e1), seq.map, 1))


def J_norm(x):
    return jnp.linalg.norm(J_xyz(x))


# %%
fig = plot_scalar_fct_physical_logical(
    J_norm,
    seq.map,
    n_vis=64,
    logical_plane="r_theta",
    cbar_label=r"$| J |$",
    cmap="berlin",
    fixed_zeta=0.0,
    avoid_boundary=1e-2
)

# %%
cuts = jnp.linspace(0, 1, 4, endpoint=True)
grids_pol = [
    get_2d_grids(seq.map, cut_axis=2, cut_value=v, nx=32, ny=32, nz=1) for v in cuts
]
grid_surface = get_2d_grids(
    seq.map, cut_axis=0, cut_value=1.0, ny=128, nz=128, z_min=0, z_max=1, invert_z=True
)
# %%
fig, ax = plot_torus(
    p_h,
    grids_pol,
    grid_surface,
    gridlinewidth=1,
    cstride=8,
    noaxes=True,
    elev=30,
    azim=30,
)


plt.savefig("relaxed_pressure_torus.pdf", dpi=300)
# %%
Phi_full_fp = jax.jit(extend_map_nfp(seq.map, nfp))
# %%
B_h = jax.jit(DiscreteFunction(B_dof, seq.basis_2, seq.e2))


@jax.jit
def B_norm(x):
    x %= 1.0
    Bx = B_h(x)
    DFx = jax.jacfwd(seq.map)(x)
    return jnp.linalg.norm(DFx @ Bx) / jnp.linalg.det(DFx)


# %%
fig = plot_scalar_fct_physical_logical(
    B_norm,
    Phi_full_fp,
    n_vis=64,
    logical_plane="tz",
    cbar_label="$|B|(r = 0.99)$",
    cmap="berlin",
    fixed_r=0.99,
)

# %%
zmax = nfp / nfp

cuts = jnp.linspace(0, zmax, int(
    1.618033988749894 * nfp * zmax), endpoint=False)
grids_pol = [
    get_2d_grids(Phi_full_fp, cut_axis=2, cut_value=v, nx=32, ny=32, nz=1, z_max=zmax) for v in cuts
]
grid_surface = get_2d_grids(
    Phi_full_fp,
    cut_axis=0,
    cut_value=1.0,
    ny=128,
    nz=128,
    z_min=0,
    z_max=zmax,
    invert_z=True,
)
fig, ax = plot_torus(
    B_norm,
    grids_pol,
    grid_surface,
    gridlinewidth=1,
    cstride=8,
    noaxes=False,
    elev=33,
    azim=33,
)
# %%
B_h = jax.jit(DiscreteFunction(B_dof, seq.basis_2, seq.e2))
logical_trajectories, physical_trajectories = integrate_fieldline(
    B_h, seq.map, nfp, T=10_000.0, n_traj=50
)
# %%
for zeta in jnp.linspace(0.1, 0.5, 5, endpoint=False):
    fig, _ = poincare_plot(
        logical_trajectories,
        seq.map,
        nfp,
        p_h=None,
        zeta_value=zeta,
        show=True,
        ks_thresh=10,
        denom_max=15,
        markersize=0.01
    )

# %%
p_avg = p_dof @ seq.p0(lambda x: jnp.ones(1)) / (seq.jacobian_j @ seq.quad.w)
beta = 2 * p_avg / (B_dof @ seq.m2 @ B_dof)
print(f"Beta = {beta:.3e}")

# %%
