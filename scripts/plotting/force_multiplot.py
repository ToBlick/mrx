# %%
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from mrx.io import load_sweep
# %%
# --- Figure settings ---
FIG_SIZE = (12, 6)
SQUARE_FIG_SIZE = (8, 8)
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 16
LINE_WIDTH = 2.5
LEGEND_SIZE = 16

def sci_fmt(x):
    """Format numbers in scientific notation, 1 digit."""
    return f"{x:.1e}" if isinstance(x, (int, float, np.floating)) else str(x)
def plot_traces(cfgs, traces, iters, sweep_params, FIG_SIZE=(6,4),
                LABEL_SIZE=14, TICK_SIZE=12, LEGEND_SIZE=12, LINE_WIDTH=2.0,
                extra_labels=None, ylabel=r'$\| J \times B - \nabla p \|$'):
    """
    Plot force traces with up to 2 sweep parameters.
    Param 1 → color, Param 2 → linestyle.
    extra_labels: list of (name, value) tuples for fixed parameters 
                  to display in the legend.
    """

    if len(sweep_params) > 2:
        raise ValueError("plot_forces only supports up to 2 sweep parameters.")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Style cycles
    color_cycle = plt.cm.plasma(
        jnp.linspace(0, 1, 1 + len(set(cfg[sweep_params[0]] for cfg in cfgs)))
    )
    linestyle_cycle = ['-', '--', '-.', ':', (0, (8,1))]

    style_map = {}
    legend_handles = []

    # Color mapping
    vals = sorted(set(cfg[sweep_params[0]] for cfg in cfgs))
    style_map[sweep_params[0]] = {val: color_cycle[i] for i, val in enumerate(vals)}
    legend_handles.append(plt.Line2D([0], [0], color="none", label=f"{sweep_params[0]}:"))
    legend_handles.extend([
        plt.Line2D([0], [0], color=color_cycle[i], lw=LINE_WIDTH,
                   label=sci_fmt(val))
        for i, val in enumerate(vals)
    ])

    # Linestyle mapping (only if 2nd parameter)
    if len(sweep_params) == 2:
        vals = sorted(set(cfg[sweep_params[1]] for cfg in cfgs))
        style_map[sweep_params[1]] = {val: linestyle_cycle[i % len(linestyle_cycle)] for i, val in enumerate(vals)}
        legend_handles.append(plt.Line2D([0], [0], color="none", label=f"{sweep_params[1]}:"))
        legend_handles.extend([
            plt.Line2D([0], [0], color="black", linestyle=linestyle_cycle[i % len(linestyle_cycle)], lw=LINE_WIDTH,
                       label=sci_fmt(val))
            for i, val in enumerate(vals)
        ])

    # Extra fixed labels (non-swept): we check they are constant
    if extra_labels is not None:
        legend_handles.append(plt.Line2D([0], [0], color="none", label="Fixed params:"))
        for key in extra_labels:
            vals = {cfg[key] for cfg in cfgs}
            if len(vals) != 1:
                raise ValueError(f"Parameter {key} is not constant across cfgs: {vals}")
            val = vals.pop()
            legend_handles.append(
                plt.Line2D([0], [0], color="none", label=f"{key} = {val}")
            )

    # Labels
    ax.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)

    # Plot
    for cfg, trace, iter in zip(cfgs, traces, iters):
        color = style_map[sweep_params[0]][cfg[sweep_params[0]]]
        linestyle = style_map[sweep_params[1]][cfg[sweep_params[1]]] if len(sweep_params) == 2 else '-'

        ax.semilogy(
            iter, trace,
            linestyle=linestyle, color=color, linewidth=LINE_WIDTH
        )

    # Legend
    ax.legend(handles=legend_handles, fontsize=LEGEND_SIZE,
              title="Sweep parameters", title_fontsize=LEGEND_SIZE,
              ncol=2, loc="best")

    ax.grid(which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    return fig, ax



# %%
print(os.getcwd())
# %%
path = "script_outputs/solovev/ITER_sweep_1/"
outdir = path + "iter_sweep/"
os.makedirs(outdir, exist_ok=True)

reference = path + "ITER_6x6_dt_1e-6_dtmax_1e0_gamma0.h5"
sweep_params = ["dt_max", "n_r", "n_chi", "dt"]  
# whatever parameters you expect to vary
# up to 3 params to map to [color, marker, linestyle]

# %%
for (qoi, label, diff) in ( 
                            ["force_trace", r'$\| J \times B - \nabla p \|$', False],
                            ["helicity_trace", r'$|H - H_0|$', True],
                            ["energy_diff_trace", r'energy dissipation violation', False],
                            ["picard_errors", r'Picard residuals', False],
                            ["picard_iterations", r'Picard iterations', False],
                            ["timesteps", r'timestep sizes', False], 
                            ):
    
    
    cfgs, trace, iters = load_sweep(path, reference, qoi, sweep_params)

    trace = np.array(trace)
    # TODO: fix iteration plots
    iters = [c["save_every"] * np.arange(len(t)) for t, c in zip(trace, cfgs)]
    # TODO:
    if qoi == "energy_diff_trace":
        trace = np.abs(trace)
    if diff:
        trace = np.abs(trace - trace[0])
    # plot_params = ["dt_max", "dt"]
    # for n in range(6, 19, 2):
    #     indices = [i for i, cfg in enumerate(cfgs) if cfg["n_r"] == n and cfg["n_chi"] == n]
    #     if indices == []:
    #         continue
    #     cfg_subset = [cfgs[i] for i in indices]
    #     f_subset = [trace[i] for i in indices]
    #     i_subset = [iters[i] for i in indices]
    #     fig, ax = plot_traces(cfg_subset, f_subset, i_subset, plot_params, FIG_SIZE=FIG_SIZE,
    #                     LABEL_SIZE=LABEL_SIZE, TICK_SIZE=TICK_SIZE,
    #                     LEGEND_SIZE=LEGEND_SIZE, LINE_WIDTH=LINE_WIDTH,
    #                     extra_labels=["n_r", "n_chi"], ylabel=label)

    plot_params = ["n_r", "dt_max"]
    for dt in [1e-6, 1e-5, 1e-4]:
        indices = [i for i, cfg in enumerate(cfgs) if cfg["dt"] == dt]
        cfg_subset = [cfgs[i] for i in indices]
        f_subset = [trace[i] for i in indices]
        i_subset = [iters[i] for i in indices]
        fig, ax = plot_traces(cfg_subset, f_subset, i_subset, plot_params, FIG_SIZE=FIG_SIZE,
                        LABEL_SIZE=LABEL_SIZE, TICK_SIZE=TICK_SIZE,
                        LEGEND_SIZE=LEGEND_SIZE, LINE_WIDTH=LINE_WIDTH,
                        extra_labels=["dt"], ylabel=label)
        fig.savefig(outdir + f"multiplot_{qoi}_dt_{dt}.png", dpi=300)
        plt.close(fig)
# %%
