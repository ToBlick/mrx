# %%
import os

import diffrax as dfx
import h5py
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

from mrx.mappings import cerfon_map, helical_map, rotating_ellipse_map
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction

from mrx.io import parse_args

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    
    # Get user input
    params = parse_args()
    name = params["run_name"]
# %%
    with h5py.File("script_outputs/helix/" + name + ".h5", "r") as f:
        B_hat = f["B_final"][:]
        p_hat = f["p_final"][:]
        helicity_trace = f["helicity_trace"][:]
        energy_trace = f["energy_trace"][:]
        force_trace = f["force_trace"][:]

        B_fields = f["B_fields"][:] if "B_fields" in f else None
        p_fields = f["p_fields"][:] if "p_fields" in f else None

        CONFIG = {k: v for k, v in f["config"].attrs.items()}
        # decode strings back if needed
        CONFIG = {k: v.decode() if isinstance(v, bytes)
                else v for k, v in CONFIG.items()}
    # %%
    # Get the map and sequences back:
    kappa = CONFIG["kappa"]
    eps = CONFIG["eps"]
    alpha = jnp.arcsin(CONFIG["delta"])

    if CONFIG["type"] == "tokamak":
        F = cerfon_map(eps, kappa, alpha)
    elif CONFIG["type"] == "helix":
        F = helical_map(epsilon=CONFIG["eps"], h=CONFIG["h_helix"],
                        n_turns=CONFIG["m_helix"], kappa=CONFIG["kappa"], alpha=alpha)
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
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)

    assert jnp.min(Seq.J_j) > 0, "Mapping is singular!"

    def integrate_fieldline(B_h, x0, F, N, t1=10):
        @jax.jit
        def vector_field(t, x, args):
            """Return the norm of B at x=(r,theta,zeta) in [0,1]^3."""
            x = x % 1.0
            r, θ, z = x
            r = jnp.clip(r, 1e-16, 1.0)
            x = jnp.array([r, θ, z])
            Bx = B_h(jnp.array(x))
            DFx = jax.jacfwd(F)(jnp.array(x))
            return Bx / jnp.linalg.norm(DFx @ Bx)

        term = dfx.ODETerm(vector_field)
        solver = dfx.Dopri5()

        t0 = 0
        dt0 = 0.05
        term = dfx.ODETerm(vector_field)
        solver = dfx.Dopri5()
        saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, N))
        stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-8)

        sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, x0,
                            saveat=saveat,
                            stepsize_controller=stepsize_controller,
                            max_steps=100_000)
        return sol.ys

    def get_crossings(B_h, x0, F, N, phi_targets):
        @jax.jit
        def vector_field(t, x, args):
            """Return the norm of B at x=(r,theta,zeta) in [0,1]^3."""
            x = x % 1.0
            r, θ, z = x
            r = jnp.clip(r, 1e-16, 1.0)
            x = jnp.array([r, θ, z])
            Bx = B_h(jnp.array(x))
            DFx = jax.jacfwd(F)(jnp.array(x))
            return Bx / jnp.linalg.norm(DFx @ Bx)

        term = dfx.ODETerm(vector_field)
        solver = dfx.Dopri5()

        def make_cond_fn(phi_target):
            def cond_fn(t, y, args, **kwargs):
                x = F(y)  # cartesian coords
                phi = (jnp.arctan2(x[1], x[0]) / jnp.pi + 1) / 2  # in [0, 1]
                return jnp.sin(2 * jnp.pi * (phi - phi_target))
            return cond_fn

        t0 = 0
        t1 = jnp.inf
        dt0 = 0.05
        term = dfx.ODETerm(vector_field)
        root_finder = dfx.VeryChord(rtol=1e-3, atol=1e-3)
        event = dfx.Event([make_cond_fn(phi) for phi in phi_targets], root_finder)
        solver = dfx.Tsit5()

        crossings = jnp.zeros((N, 3))

        for i in range(N):
            sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, x0, event=event,
                                max_steps=20_000, throw=False)
            if sol.ys.size == 0:
                break
            x_cross = sol.ys[0]
            crossings = crossings.at[i, :].set((x_cross))
            # restart
            # step a bit forward to avoid finding the same root again
            x0 = (sol.ys[0] + 1e-3 * vector_field(0, sol.ys[0], None))

        return crossings

    F = jax.jit(F)

    @jax.jit
    def F_cyl_signed(x):
        """Cylindrical coords with signed R."""
        R = jnp.sqrt(x[0]**2 + x[1]**2) * \
            jnp.sign(jnp.sin(1e-16 + jnp.arctan2(x[1], x[0])))
        phi = jnp.arctan2(x[1], x[0])
        z = x[2]
        return jnp.array([R, phi, z])

    # %%
    n_lines = 100
    r_min, r_max = 0.01, 0.99
    p = 1.0
    _r = np.linspace(r_min, r_max, n_lines)
    # x0s = np.vstack(
    #     (np.hstack((_r[::3], _r[1::3], _r[2::3])),                              # between 0 and 1 - samples along x
    #     # half go to theta=0 and half to theta=pi
    #     np.hstack((0.33 * np.ones(n_lines//3),
    #                 0.66 * np.ones(n_lines//3),
    #                 0.99 * np.ones(n_lines//3))),
    #     0.0 * np.ones(n_lines))
    # ).T
    x0s = np.vstack(
        (np.hstack((_r[::4], _r[1::4], _r[2::4], _r[3::4])),                              # between 0 and 1 - samples along x
        # half go to theta=0 and half to theta=pi
        np.hstack(( 0.027345 * np.ones(n_lines//4),
                    0.246452 * np.ones(n_lines//4),
                    0.515341 * np.ones(n_lines//4),
                    0.749154 * np.ones(n_lines//4)
                )),
        0.174 * np.ones(n_lines))
    ).T
    
    N_cross = 5_000

    key = x0s[:, 0] * np.cos(x0s[:, 1])

    # Get sorted indices
    idx = np.argsort(key)

    # Apply sorting
    x0s_sorted = x0s[idx]

    # %%
    output_dir = os.path.join("script_outputs", "helix", name, "poincare")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    print(B_fields.shape, p_fields.shape)
    for plt_nr in range(B_fields.shape[0]):
        print(f"--- Starting step {plt_nr} ---", flush=True)
        B_hat = B_fields[plt_nr]
        p_hat = p_fields[plt_nr]

        p_h = DiscreteFunction(p_hat, Seq.Λ0, Seq.E0)
        B_h = (DiscreteFunction(B_hat, Seq.Λ2, Seq.E2))

        crossings = jax.vmap(lambda x0: get_crossings(
            # m x N x 3
            B_h, x0, F, N=N_cross, phi_targets=[0.25]))(x0s_sorted)

        crossings_xyz = jax.vmap(jax.vmap(F))(crossings)
        crossings_Rphiz = jax.vmap(jax.vmap(F_cyl_signed))(crossings_xyz)


        crossings_to_plot = crossings_Rphiz
        crossings_p = crossings % 1

        # ================== PLOT CONFIGURATION ==================
        dot_width = 1.0
        tick_label_size = 20
        axis_label_size = 22

        # --- Gap and Limit Detection ---
        all_R_values = crossings_to_plot[:, :, 0].flatten()
        positive_R = all_R_values[all_R_values > 0]
        negative_R = all_R_values[all_R_values < 0]

        gap_left = 0.95 * np.max(negative_R) if len(negative_R) > 0 else -0.1
        gap_right = 0.95 * np.min(positive_R) if len(positive_R) > 0 else 0.1

        Rmin = -0.01 + np.min(all_R_values)
        Rmax = 0.01 + np.max(all_R_values)
        Zmin = -0.01 + np.min(crossings_to_plot[:, :, 2])
        Zmax = 0.01 + np.max(crossings_to_plot[:, :, 2])

        # 1. Determine max spans for symmetric panels
        x_span_left = gap_left - Rmin
        x_span_right = Rmax - gap_right
        max_x_span = max(x_span_left, x_span_right)
        y_span = Zmax - Zmin

        # --- Colormaps ---
        def truncate_colormap(cmap_name="turbo", minval=0.0, maxval=0.85, n=256, reverse=False):
            cmap = plt.get_cmap(cmap_name, n)
            new_colors = cmap(np.linspace(minval, maxval, n))
            if reverse:
                new_colors = new_colors[::-1]
            return mpl.colors.ListedColormap(new_colors)

        # iota → turbo, p → plasma
        cmap_p = truncate_colormap("plasma", 0.0, 0.9)

        # compute p-values per crossing
        p_values = 100 * (jnp.array([jax.vmap(p_h)(curve) for curve in crossings_p]))[:, :, 0]
        norm_p = mpl.colors.Normalize(vmin=0.0, vmax=np.max(p_values))

        # --- Figure setup ---
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(1, 2, wspace=0.01)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)

        # =============================================================================
        # --- Plotting Loop ---
        for i, curve in enumerate(crossings_to_plot):
            
            # shape of curve is (N_cross, 3)
            if jnp.min(crossings[i, :, 0]) <= 1e-12:
                continue
            # --- Left panel (negative-R side): color by p ---
            mask_left = curve[:, 0] < gap_left
            if np.any(mask_left):
                ax1.scatter(curve[mask_left, 0],
                            curve[mask_left, 2],
                            s=dot_width,
                            color=cmap_p(norm_p(p_values[i][mask_left])),
                            rasterized=True)

            # --- Right panel (positive-R side): color by iota ---
            mask_right = curve[:, 0] > gap_right
            if np.any(mask_right):
                ax2.scatter(curve[mask_right, 0],
                            curve[mask_right, 2],
                            s=dot_width,
                            color=cmap_p(norm_p(p_values[i][mask_right])),
                            rasterized=True)

        # =============================================================================
        # --- Styling, limits, and axes ---
        ax1.set_xlim(gap_left - max_x_span, gap_left, auto=False)
        ax2.set_xlim(gap_right, gap_right + max_x_span, auto=False)
        ax1.set_ylim(Zmin, Zmax, auto=False)

        ax1.xaxis.set_major_locator(MultipleLocator(0.1))
        ax2.xaxis.set_major_locator(MultipleLocator(0.1))
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))

        ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax2.tick_params(axis='x', which='major', labelsize=tick_label_size)
        ax1.grid(True, linestyle="--", lw=0.5)
        ax2.grid(True, linestyle="--", lw=0.5)

        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.tick_params(axis='y', which='both', length=0)

        ax1.set_ylabel(r"$z$", fontsize=axis_label_size)
        ax1.set_xlabel(r"$-R$", fontsize=axis_label_size)
        ax2.set_xlabel(r"$R$", fontsize=axis_label_size)

       # Equal aspect and synced limits
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlim(gap_right, gap_right + max_x_span)

        # Adjust layout for right-side colorbar
        fig.subplots_adjust(bottom=0.08, wspace=0.02, right=0.90)

        # --- External colorbar ---
        sm_p = mpl.cm.ScalarMappable(norm=norm_p, cmap=cmap_p)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar1 = fig.colorbar(sm_p, cax=cbar_ax)
        cbar1.set_label(r"$p \; (\times 10^2)$", fontsize=axis_label_size)
        cbar1.ax.tick_params(labelsize=tick_label_size)

        # --- Save and Show ---
        plt.savefig(os.path.join(output_dir, f"step_{plt_nr}.pdf"),
                    dpi=200, bbox_inches=None)
        print(f"Saved step {plt_nr} poincare plot.")
        plt.close()

# %%



# # %%
# crossings_to_plot = crossings_Rphiz[:, :, :]

# # ================== PLOT CONFIGURATION ==================
# dot_width = 1.0
# tick_label_size = 20
# axis_label_size = 22
# # --------------------------------------------------------

# # --- Gap and Limit Detection ---
# all_R_values = crossings_to_plot[:, :, 0].flatten()
# positive_R = all_R_values[all_R_values > 0]
# negative_R = all_R_values[all_R_values < 0]

# gap_left = 0.95 * np.max(negative_R) if len(negative_R) > 0 else -0.1
# gap_right = 0.95 * np.min(positive_R) if len(positive_R) > 0 else 0.1

# Rmin = -0.01 + np.min(all_R_values)
# Rmax = 0.01 + np.max(all_R_values)
# Zmin = -0.01 + np.min(crossings_to_plot[:, :, 2])
# Zmax = 0.01 + np.max(crossings_to_plot[:, :, 2])

# # 1. Find out which plot's data range is wider.
# x_span_left = gap_left - Rmin
# x_span_right = Rmax - gap_right
# max_x_span = max(x_span_left, x_span_right)  # The width needed for the box

# # 2. Find the height of the data.
# y_span = Zmax - Zmin

# # 3. Create two identical plot boxes
# fig = plt.figure(figsize=(24, 12))
# gs = gridspec.GridSpec(1, 2, wspace=0.05)
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1], sharey=ax1)

# # =============================================================================

# # --- Plotting Loop ---
# def truncate_colormap(cmap_name="turbo", minval=0.0, maxval=0.85, n=256):
#     cmap = plt.get_cmap(cmap_name, n)
#     new_colors = cmap(np.linspace(minval, maxval, n))
#     return mpl.colors.ListedColormap(new_colors)

# cmap_p = truncate_colormap("plasma", 0.0, 0.9)
# cmap = truncate_colormap("turbo", 0.0, 0.85)
# norm = mpl.colors.Normalize(vmin=np.min(iotas), vmax=np.max(iotas))
# sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

# for i, curve in enumerate(crossings_to_plot):
#     color = cmap(norm(iotas[i]))  # color based on iota
#     ax1.scatter(curve[curve[:, 0] < gap_left, 0],
#                 curve[curve[:, 0] < gap_left, 2],
#                 s=dot_width, color=color, rasterized=True)
#     ax2.scatter(curve[curve[:, 0] > gap_right, 0],
#                 curve[curve[:, 0] > gap_right, 2],
#                 s=dot_width, color=color, rasterized=True)

# # --- Set limits, aspect ratio, and styling ---
# # Enforce strict axis bounds and disable autoscaling so Matplotlib doesn't
# # expand the right-hand panel beyond the requested limits.
# ax1.set_xlim(gap_left - max_x_span, gap_left, auto=False)
# ax2.set_xlim(gap_right, gap_right + max_x_span, auto=False)
# ax1.set_ylim(Zmin, Zmax, auto=False)
# # # --- Ticks and Labels ---
# ax1.xaxis.set_major_locator(MultipleLocator(0.1))
# ax2.xaxis.set_major_locator(MultipleLocator(0.1))
# ax1.yaxis.set_major_locator(MultipleLocator(0.1))

# ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
# ax2.tick_params(axis='x', which='major', labelsize=tick_label_size)
# ax1.grid(True, linestyle="--", lw=0.5)
# ax2.grid(True, linestyle="--", lw=0.5)

# plt.setp(ax2.get_yticklabels(), visible=False)
# ax2.tick_params(axis='y', which='both', length=0)

# ax1.set_ylabel(r"$z$", fontsize=axis_label_size)
# ax1.set_xlabel(r"$-R$", fontsize=axis_label_size)
# ax2.set_xlabel(r"$R$", fontsize=axis_label_size)

# # # Use tight_layout for final margin adjustments
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.08)  # Add a bit more space for the supxlabel

# # Add a taller colorbar on the right side of the figure
# cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=1.0, pad=0.02, aspect=25)
# cbar.set_label(r"$\iota$", fontsize=axis_label_size)
# cbar.ax.tick_params(labelsize=tick_label_size)

# # ##### INSET PLOT #####
# # # Parameters controlling the inset position and size
# # # horizontal position in figure fraction (0 → left, 1 → right)
# # inset_x = 0.6
# # inset_y = 0.6     # vertical position in figure fraction (0 → bottom, 1 → top)
# # inset_width = 0.25
# # inset_height = 0.25

# # # Create inset axis (attached to the first panel, but could be fig.add_axes instead)
# # ax_inset = fig.add_axes([inset_x, inset_y, inset_width, inset_height])

# # # Plot the iota curve
# # ax_inset.plot(iota_s, color='black', lw=2)

# # # Optional cosmetics
# # ax_inset.set_title(r"$\iota(\zeta)$", fontsize=axis_label_size - 2)
# # ax_inset.tick_params(labelsize=tick_label_size - 4)
# # ax_inset.grid(True, ls='--', lw=0.5)

# # --- Save and Show ---
# # Create directory if it doesn't exist
# # output_dir = os.path.join("script_outputs", "solovev")
# # os.makedirs(output_dir, exist_ok=True)
# # plt.savefig(os.path.join(output_dir, "helix_poincare.pdf"),
# #             dpi=400, bbox_inches=None)

# plt.show()


# #################################

# # %%
# crossings_to_plot = crossings_Rphiz
# crossings_p = crossings % 1

# # ================== CONFIG ==================
# dot_width = 1.0
# tick_label_size = 16
# axis_label_size = 18

# # --- Limits ---
# all_R_values = crossings_to_plot[:, :, 0].flatten()  # exclude zeros
# all_Z_values = (crossings_to_plot[:, :, 2].flatten())
# Rmin, Rmax = np.min(all_R_values[all_R_values > 0]), np.max(
#     all_R_values[all_R_values > 0])
# Zmin, Zmax = np.min(all_Z_values[all_R_values > 0]), np.max(
#     all_Z_values[all_R_values > 0])

# # --- Colormaps ---


# def truncate_colormap(cmap_name="plasma", minval=0.0, maxval=0.85, n=256, reverse=False):
#     cmap = plt.get_cmap(cmap_name, n)
#     new_colors = cmap(np.linspace(minval, maxval, n))
#     if reverse:
#         new_colors = new_colors[::-1]
#     return mpl.colors.ListedColormap(new_colors)


# cmap_iota = truncate_colormap("turbo", 0.0, 1.0)
# cmap_p = truncate_colormap("plasma", 0.0, 0.9)

# norm_iota = mpl.colors.Normalize(vmin=np.min(iotas), vmax=np.max(iotas))

# # if p varies per curve
# p_values = (jnp.array([jax.vmap(p_h)(curve)
#             for curve in crossings_p]))[:, :, 0]
# norm_p = mpl.colors.Normalize(vmin=np.min(p_values), vmax=np.max(p_values))

# # --- Figure ---
# fig, ax = plt.subplots(figsize=(10, 8))

# # --- Plotting loop ---
# for i, curve in enumerate(crossings_to_plot):
#     z = curve[:, 2]
#     R = curve[:, 0]

#     # mask z>0 and z<0
#     mask_pos = (z > 0) & (R > 0)
#     mask_neg = (z < 0) & (R > 0)

#     # color by iota (for z>0)
#     if np.any(mask_pos):
#         ax.scatter((R[mask_pos]), (z[mask_pos]),
#                    s=dot_width,
#                    color=cmap_iota(norm_iota(iotas[i])),
#                    rasterized=True)

#     # color by p (for z<0)
#     if np.any(mask_neg):
#         ax.scatter((R[mask_neg]), (z[mask_neg]),
#                    s=dot_width,
#                    color=cmap_p(norm_p(p_values[i][mask_neg])),
#                    rasterized=True)

# # --- Style ---
# ax.set_xlim(Rmin - 0.01, Rmax + 0.01)
# ax.set_ylim(Zmin - 0.01, Zmax + 0.01)
# ax.set_aspect("equal", adjustable="box")

# ax.xaxis.set_major_locator(MultipleLocator(0.1))
# ax.yaxis.set_major_locator(MultipleLocator(0.1))
# ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
# # ax.grid(True, linestyle="--", lw=0.5)

# ax.set_xlabel(r"$R$", fontsize=axis_label_size)
# ax.set_ylabel(r"$z$", fontsize=axis_label_size)


# # ============================================================
# # Colorbars stacked vertically on the right
# # ============================================================
# sm_iota = mpl.cm.ScalarMappable(norm=norm_iota, cmap=cmap_iota)
# sm_p = mpl.cm.ScalarMappable(norm=norm_p, cmap=cmap_p)

# # Main plot area adjustment to make space for colorbars
# plt.subplots_adjust(right=0.85)

# # Position colorbars manually using add_axes
# cbar_height = 0.33   # relative height of each bar
# cbar_width = 0.05
# x0 = 0.82           # x-position for both bars

# # iota (top)
# cax1 = fig.add_axes([x0, 0.55, cbar_width, cbar_height])
# cbar1 = fig.colorbar(sm_iota, cax=cax1)
# cbar1.set_label(r"$\iota$", fontsize=axis_label_size)
# cbar1.ax.tick_params(labelsize=tick_label_size)

# # p (bottom)
# cax2 = fig.add_axes([x0, 0.1, cbar_width, cbar_height])
# cbar2 = fig.colorbar(
#     sm_p, cax=cax2, format=mpl.ticker.ScalarFormatter(useMathText=True))
# cbar2.formatter.set_powerlimits((0, 0))
# cbar2.set_label(r"$p$", fontsize=axis_label_size)
# cbar2.ax.tick_params(labelsize=tick_label_size)
# cbar2.update_ticks()
# offset_text = cbar2.ax.yaxis.get_offset_text()
# offset_text.set_fontsize(tick_label_size)
# offset_text.set_x(1.5)

# # plt.savefig("ROT_ELL_double_poincare.pdf", dpi=400)

# plt.show()

# %%
