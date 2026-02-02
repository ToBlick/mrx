# %%
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.desc_interface import project_desc_equilibrium
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.io import load_desc
from mrx.mappings import extend_map_nfp
from mrx.plotting import (
    get_2d_grids,
    integrate_fieldline,
    plot_scalar_fct_physical_logical,
    plot_torus,
    plot_twin_axis,
    poincare_plot,
)
from mrx.relaxation import (
    IntegrationScheme,
    MRXDiagnostics,
    TimeStepChoice,
    TimeStepper,
    relaxation_loop,
)
from mrx.utils import interpolate_B

jax.config.update("jax_enable_x64", True)

jax.config.update("jax_compilation_cache_dir", "../../jax-caches")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# %%
DESC_PATH = os.path.join(os.path.dirname(
    __file__), "../", "data", "desc_heliotron.h5")
# %%
result = project_desc_equilibrium(DESC_PATH, ns=(6, 10, 5), ps=(3, 3, 3))
B0 = result['B_h'].dof
nfp = result['nfp']
seq = result['seq']
seq.assemble_all()
seq.build_crossproduct_projections()
seq.assemble_leray_projection()

B0 = seq.P_Leray @ B0
B0 /= (B0 @ seq.M2 @ B0) ** 0.5
# %%
# ---------- Relaxation ----------
num_iters_inner = 100
num_iters_outer = 10
ts = TimeStepper(
    seq=seq,
    conjugate=True,
    dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
    timestep_mode=IntegrationScheme.EXPLICIT,
    newton=False,
)

final_state, traces = relaxation_loop(
    B0,
    ts,
    num_iters_inner=num_iters_inner,
    num_iters_outer=num_iters_outer,
    dt0=1.0,
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
diagnostics = MRXDiagnostics(seq)
# state = final_state
B_dof = B0  # state.B_n
get_pressure = jax.jit(diagnostics.pressure)
p_dof = get_pressure(B_dof)
p_h = jax.jit(DiscreteFunction(p_dof, seq.Lambda_0, seq.E0))
# %%
fig = plot_scalar_fct_physical_logical(
    p_h,
    seq.F,
    n_vis=64,
    logical_plane="r_theta",
    cbar_label="$p$",
    cmap="berlin",
    fixed_zeta=0.5,
    avoid_boundary=1e-3
)

# %%
J_xyz = jax.jit(Pushforward(DiscreteFunction(
    seq.weak_curl @ B_dof, seq.Lambda_1, seq.E1), seq.F, 1))


def J_norm(x):
    return jnp.linalg.norm(J_xyz(x))


# %%
fig = plot_scalar_fct_physical_logical(
    J_norm,
    seq.F,
    n_vis=64,
    logical_plane="r_theta",
    cbar_label=r"$| J |$",
    cmap="berlin",
    fixed_zeta=0.0,
    avoid_boundary=1e-3
)

# %%
cuts = jnp.linspace(0, 1, 4, endpoint=True)
grids_pol = [
    get_2d_grids(seq.F, cut_axis=2, cut_value=v, nx=32, ny=32, nz=1) for v in cuts
]
grid_surface = get_2d_grids(
    seq.F, cut_axis=0, cut_value=1.0, ny=128, nz=128, z_min=0, z_max=1, invert_z=True
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
Phi_full_fp = jax.jit(extend_map_nfp(seq.F, nfp))
# %%
B_h = jax.jit(DiscreteFunction(B_dof, seq.Lambda_2, seq.E2))


@jax.jit
def B_norm(x):
    x %= 1.0
    Bx = B_h(x)
    DFx = jax.jacfwd(seq.F)(x)
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
    p_h,
    grids_pol,
    grid_surface,
    gridlinewidth=1,
    cstride=8,
    noaxes=False,
    elev=33,
    azim=33,
)
# %%
logical_trajectories, physical_trajectories = integrate_fieldline(
    B_h, seq.F, nfp, T=1000.0
)
# %%
for zeta in jnp.linspace(0.1, 0.5, 5, endpoint=False):
    fig, _ = poincare_plot(
        logical_trajectories,
        seq.F,
        nfp,
        zeta_value=zeta,
        show=True,
        cmap='turbo',
    )

# %%
p_avg = p_dof @ seq.P0(lambda x: jnp.ones(1)) / (seq.J_j @ seq.Q.w)
B_dof = B_dof
beta = 2 * p_avg / (B_dof @ seq.M2 @ B_dof)
print(f"Beta = {beta:.3e}")

# %%
