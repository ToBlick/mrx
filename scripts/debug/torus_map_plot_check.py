"""Diagnostic: render the SAME plot_torus figure for the analytic toroid map.

If the poloidal cut planes star/pinwheel here too (circular cross-section ->
should be clean filled discs), the bug is in get_2d_grids / plot_torus index
ordering, not in the W7-X map. Analytic + tiny -> runs locally, no GPU/solves.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from mrx.differential_forms import jacobian_determinant  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.plotting import get_2d_grids, plot_torus  # noqa: E402

outdir = os.path.join("outputs", "torus_map_plot_check")
os.makedirs(outdir, exist_ok=True)

F = toroid_map(epsilon=1 / 3, kappa=1.0)
p_h = jacobian_determinant(F)

cuts = jnp.linspace(0.0, 1.0, 6, endpoint=False)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=float(v),
                          nx=28, ny=48, nz=1) for v in cuts]
grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0, ny=96, nz=96)

fig, ax = plot_torus(p_h, grids_pol, grid_surface,
                     gridlinewidth=1, cstride=6, elev=22, azim=40)
ax.set_title("Analytic toroid — surface $\\rho{=}1$, colour = Jacobian $J$")
out = os.path.join(outdir, "torus_field.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")

# Also dump the raw index ordering of one poloidal cut for inspection.
g = grids_pol[0]
xlog = g[0]            # (N,3) logical points fed to p_h
X = g[2][0]           # (n1,n2) physical x reshaped
print("poloidal-cut logical pts shape:", xlog.shape, " physical X shape:", X.shape)
print("first 6 logical (r,theta,zeta):")
for row in xlog[:6]:
    print("  ", [float(v) for v in row])
