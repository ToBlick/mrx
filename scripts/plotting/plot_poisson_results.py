# %%
"""Scan multirun result.json files and create convergence/timing plots."""
import glob
import json
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})

# %%

# ---------- collect results ----------
results = defaultdict(dict)  # results[p][n] = {...}
for path in glob.glob("/scratch/tblickhan/mrx/multirun/2026-03-05/13-08-53/**/result.json", recursive=True):
    with open(path) as f:
        r = json.load(f)
    results[r["p"]][r["n"]] = r

ps = sorted(results.keys())
markers = "os^Dv<>ph"

# Plasma colormap, truncated to drop the bright yellow end
cmap = matplotlib.colormaps["plasma"]
colors = [cmap(x) for x in np.linspace(0.0, 0.78, max(len(ps), 1))]

# ---------- define panels ----------
# Panel kind: "scalar" | "nnz"
panels = [
    ("scalar", r"Relative $L^2$ error",                       lambda r: r["error"]),
    ("scalar", r"Assembly time ($M^0$)",                       lambda r: r["timings"]["assemble_m0_sparse"]),
    ("scalar", r"Assembly time ($D^\top D^0$)",                lambda r: r["timings"]["assemble_dd0_sparse"]),
    ("scalar", r"CG solve time",                               lambda r: r["timings"]["cg_solve"]),
    ("scalar", r"DeRhamSequence init",                         lambda r: r["timings"]["DeRhamSequence.__init__"]),
    ("scalar", r"1-D basis evaluation",                        lambda r: r["timings"]["evaluate_1d"]),
    ("nnz",    r"$M^0$ nnz / $n^{2d}$ (stored vs actual)",    None),
    ("nnz",    r"$D^\top D^0$ nnz / $n^{2d}$ (stored vs actual)", None),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.ravel()

for ax, (kind, title, getter) in zip(axes, panels):
    if kind == "scalar":
        for i, p in enumerate(ps):
            ns = sorted(results[p].keys())
            vals = [getter(results[p][n]) for n in ns]
            ax.loglog(ns, vals, f"-{markers[i % len(markers)]}",
                      color=colors[i], label=rf"$p={p}$",
                      linewidth=2.2, markersize=7)
        ax.legend(fontsize=9)

    elif kind == "nnz":
        prefix = "m0" if "M^0" in title else "dd0"
        for i, p in enumerate(ps):
            ns = sorted(results[p].keys())
            stored = np.array([results[p][n]["sparsity"][f"{prefix}_nnz_stored"] for n in ns])
            actual = np.array([results[p][n]["sparsity"][f"{prefix}_nnz_actual"] for n in ns])
            ns_arr = np.array(ns)
            m = markers[i % len(markers)]
            ax.loglog(ns_arr, stored / ns_arr**6, f"--{m}", color=colors[i],
                      label=rf"$p={p}$ stored", linewidth=1.8, markersize=6, alpha=0.6)
            ax.loglog(ns_arr, actual / ns_arr**6, f"-{m}", color=colors[i],
                      label=rf"$p={p}$ actual", linewidth=2.2, markersize=7)
        ax.legend(fontsize=7, ncol=2)

    ax.set_xlabel(r"$n$")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.3)

fig.tight_layout()
fig.savefig("/scratch/tblickhan/mrx/scripts/plotting/multirun_summary.png", dpi=200)
print("Saved multirun_summary.png")
plt.show()

# %%
