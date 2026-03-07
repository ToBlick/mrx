# %%
"""Scan multirun result.json files and create convergence/timing plots."""
import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# %%

# ---------- collect results ----------
results = defaultdict(dict)  # results[p][n] = {...}
for path in glob.glob("/scratch/tblickhan/mrx/multirun/2026-03-05/13-08-53/**/result.json", recursive=True):
    with open(path) as f:
        r = json.load(f)
    results[r["p"]][r["n"]] = r

ps = sorted(results.keys())
markers = "os^Dv<>ph"

# ---------- define panels ----------
# Panel kind: "scalar" | "nnz" | "cond"
panels = [
    ("scalar", "Relative $L^2$ error",                  lambda r: r["error"]),
    ("scalar", "Assembly time (m0)",                     lambda r: r["timings"]["assemble_m0_sparse"]),
    ("scalar", "Assembly time (dd0)",                    lambda r: r["timings"]["assemble_dd0_sparse"]),
    ("scalar", "CG solve time",                         lambda r: r["timings"]["cg_solve"]),
    ("scalar", "DeRhamSequence.__init__",               lambda r: r["timings"]["DeRhamSequence.__init__"]),
    ("scalar", "evaluate\\_1d",                          lambda r: r["timings"]["evaluate_1d"]),
    ("nnz",    "m0 nnz / $n^{2d}$ (stored vs actual)",  None),
    ("nnz",    "dd0 nnz / $n^{2d}$ (stored vs actual)", None),
    ("cond",   "Condition number (raw vs precond.)",     None),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
axes = axes.ravel()

for ax, (kind, title, getter) in zip(axes, panels):
    if kind == "scalar":
        for i, p in enumerate(ps):
            ns = sorted(results[p].keys())
            vals = [getter(results[p][n]) for n in ns]
            ax.loglog(ns, vals, f"-{markers[i % len(markers)]}",
                      label=f"$p={p}$", linewidth=1.4, markersize=6)
        ax.legend(fontsize=8)

    elif kind == "nnz":
        prefix = "m0" if "m0" in title else "dd0"
        for i, p in enumerate(ps):
            ns = sorted(results[p].keys())
            stored = np.array([results[p][n]["sparsity"][f"{prefix}_nnz_stored"] for n in ns])
            actual = np.array([results[p][n]["sparsity"][f"{prefix}_nnz_actual"] for n in ns])
            ns_arr = np.array(ns)
            m = markers[i % len(markers)]
            ax.loglog(ns_arr, stored / ns_arr**6, f"--{m}", color=f"C{i}",
                      label=f"$p={p}$ stored", linewidth=1.2, markersize=5, alpha=0.6)
            ax.loglog(ns_arr, actual / ns_arr**6, f"-{m}", color=f"C{i}",
                      label=f"$p={p}$ actual", linewidth=1.4, markersize=6)
        ax.legend(fontsize=7, ncol=2)

    elif kind == "cond":
        any_data = False
        for i, p in enumerate(ps):
            ns_with_cond = [n for n in sorted(results[p].keys())
                            if "cond" in results[p][n]]
            if not ns_with_cond:
                continue
            any_data = True
            ns_arr = np.array(ns_with_cond)
            raw    = np.array([results[p][n]["cond"] for n in ns_with_cond])
            pre    = np.array([results[p][n]["cond_precond"] for n in ns_with_cond])
            m = markers[i % len(markers)]
            ax.loglog(ns_arr, raw, f"--{m}", color=f"C{i}",
                      label=f"$p={p}$ raw", linewidth=1.2, markersize=5, alpha=0.6)
            ax.loglog(ns_arr, pre, f"-{m}", color=f"C{i}",
                      label=f"$p={p}$ precond.", linewidth=1.4, markersize=6)
        if any_data:
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.text(0.5, 0.5, "no cond. data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    ax.set_xlabel("$n$")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.3)

fig.tight_layout()
fig.savefig("/scratch/tblickhan/mrx/scripts/plotting/multirun_summary.png", dpi=200)
print("Saved multirun_summary.png")
plt.show()

# %%
