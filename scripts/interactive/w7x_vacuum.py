# %%
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.io import interpolate_map_from_points
from mrx.plotting import (
        get_iota_log,
        get_periodic_intersections,
        integrate_fieldlines,
        poincare_plot,
)

mrx.MAP_BATCH_SIZE = 20_000

# %%

jax.config.update("jax_enable_x64", True)


with h5py.File("data/gvec_w7x.h5", "r") as f:
        pts = jnp.array(f["eval_points"])
        R = jnp.array(f["R"])
        Z = jnp.array(f["Z"])
nfp = 5
# %%
# Map interpolation
ns = (8, 12, 8)
ps = (6,6,6)
quad_order = max(ps)
# Validate that ps_map_x <= ns_map_x - 1
ps = tuple(min(p, n - 1) for p, n in zip(ps, ns))
nfp = 5
print("Interpolating map...")
map_func, R_dof, Z_dof, map_resid = interpolate_map_from_points(
    pts, R, Z, nfp, ns=ns, ps=ps,
    quad_order=quad_order, flip_zeta=False
)
map_func = jax.jit(map_func)
print(
    f"Map interpolation residuals: {map_resid[0]:.2e}, {map_resid[1]:.2e}")
# Setup FEM spaces
print(f"Setting up FEM spaces with ns={ns}, ps={ps}...")
seq = DeRhamSequence(
    ns, ps, quad_order,
    ("clamped", "periodic", "periodic"),
    map_func, polar=True, dirichlet=True
)
assert jnp.min(seq.J_j) > 0, "Negative Jacobian!"
# %%
seq.evaluate_1d()
seq.assemble_M3()
seq.assemble_M2()
seq.assemble_M1()
seq.assemble_d1()
seq.assemble_d2()
seq.assemble_dd2()
# %%
evs, evecs = sp.linalg.eigh(seq.M2 @ seq.dd2, seq.M2)

B_vac_dof = evecs[:, 0]
B_vac = DiscreteFunction(B_vac_dof, seq.Lambda_2, seq.E2)

assert jnp.all(jnp.abs(seq.strong_div @ B_vac_dof) < 1e-8)
assert jnp.all(jnp.abs(seq.weak_curl @ B_vac_dof) < 1e-8)

# %%
# Generate Poincare plot
axis_margin = 1e-3
n_scan = 1
n_vmap = 64
n_traj = n_scan * n_vmap
T = 300 * jnp.pi * 2 * nfp
N = int(T)

integrate_fieldlines_jit = jax.jit(
        integrate_fieldlines, static_argnames=["T", "N", "seq"]
    )
get_periodic_intersections_jit = jax.jit(
        get_periodic_intersections, static_argnames=["max_intersections"]
    )

r_vals = jnp.linspace(axis_margin, 0.99, n_traj)
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)
x2_vals = jax.random.uniform(key1, shape=(n_traj,), minval=0.0, maxval=1.0)
x3_vals = jax.random.uniform(key2, shape=(n_traj,), minval=0.0, maxval=1.0)
x0s = jnp.stack([r_vals, x2_vals, x3_vals], axis=1)
x0_grid = x0s.reshape((n_scan, n_vmap, 3))
# %%

# ── Trace fieldlines for each intermediate state ─────────────────────
ks_thresh = 4
zeta_values = jnp.linspace(0.1, 0.9, 9)
max_intersections = int(T // (2 * nfp))
plot_data = []
print(f"\nIntegrating fieldlines (n_traj={n_traj}, T={float(T):.0f}, N={N}) ...")

logical_trajectories, p_values = integrate_fieldlines_jit(
        x0_grid, B_vac_dof, jnp.zeros(seq.E0.shape[0]), seq, T, N
    )
logical_trajectories = logical_trajectories.reshape((n_traj, N, 3))
p_values = p_values.reshape((n_traj, N))
iotas, flags, ks = jax.vmap(
    lambda c: get_iota_log(c, nfp, ks_thresh=ks_thresh)
)(logical_trajectories)
for zeta_val in zeta_values:
    logical_intersections, p_at_intersections, counts = jax.vmap(
        lambda t, p: get_periodic_intersections_jit(
            t,
            plane_normal=jnp.array([0.0, 0.0, 1.0]),
            plane_point=jnp.array([0.0, 0.0, zeta_val]),
            p_values=p,
            max_intersections=max_intersections,
        )
    )(logical_trajectories, p_values)
    physical_intersections = jax.vmap(map_func)(
        logical_intersections.reshape(-1, 3)
    ).reshape(logical_intersections.shape)
    plot_data.append({
        "zeta": zeta_val,
        "physical_intersections": physical_intersections,
        "logical_intersections": logical_intersections,
        "p_at_intersections": p_at_intersections,
        "iotas": iotas,
        "ks": ks,
        "counts": counts,
    })

# %%
cmap_iota = "nipy_spectral"
# ─ Compute global colour limits ─────────────────────────────────────
# Set iotas where ks > ks_thresh to NaN
all_iotas = jnp.array([d["iotas"] for d in plot_data])
all_ks = jnp.array([d["ks"] for d in plot_data])
all_iotas_filtered = jnp.where(all_ks > ks_thresh, jnp.nan, all_iotas)
iota_min = float(jnp.nanmin(all_iotas_filtered))
iota_max = float(jnp.nanmax(all_iotas_filtered))
# ── Generate and save plots ──────────────────────────────────────────
dpi = 150
markersize = 0.01
denom_max = 40
fig_format = "pdf"
print(f"\nGenerating {len(plot_data)} Poincaré plots ...")
for i, data in enumerate(tqdm(plot_data)):
    phys_intersec = data["physical_intersections"]
    log_intersec = data["logical_intersections"]
    iotas = data["iotas"]
    p_at_intersections = data["p_at_intersections"]
    # Convert to cylindrical coordinates
    R_vals = (phys_intersec[:, :, 0] ** 2 + phys_intersec[:, :, 1] ** 2) ** 0.5
    phi_vals = jnp.arctan2(phys_intersec[:, :, 1], phys_intersec[:, :, 0])
    z_vals = phys_intersec[:, :, 2]
    cyl_intersections = jnp.stack([R_vals, phi_vals, z_vals], axis=-1)
    iota_values = jnp.broadcast_to(iotas[:, None], phys_intersec[:, :, 0].shape)
    auto_Rlim = (float(jnp.nanmin(R_vals)), float(jnp.nanmax(R_vals)))
    auto_zlim = (float(jnp.nanmin(z_vals)), float(jnp.nanmax(z_vals)))
    fig, axes = poincare_plot(
        log_intersec,
        cyl_intersections,
        None,
        iota_values,
        nfp,
        cmap_iota=cmap_iota,
        cmap_p="plasma",
        markersize=markersize,
        denom_max=denom_max,
        Rlim=auto_Rlim,
        zlim=auto_zlim,
        p_lim=(0, 1),
        iota_lim=(iota_min, iota_max),
    )
    zeta_tag = f"_zeta{data['zeta']:.2f}" if len(zeta_values) > 1 else ""
    filename = f"poincare_plot_{zeta_tag}.{fig_format}"
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
print(f"\nDone! {len(plot_data)} plots saved.")
# %%
