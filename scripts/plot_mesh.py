"""Draw the spline mesh in the mapped geometry, to show how strongly shaped the domain is and where the knots land.

    python scripts/plot_mesh.py --geometry data/wout_li383_1.4m.nc --resolution 16 32 32 --out DIR
    python scripts/plot_mesh.py --geometry data/wout_li383_1.4m.nc --resolution 16 32 32 --view 3d --out DIR

``--view 2d`` writes ``mesh_2d.png``, the poloidal cross-section at every plane of ``--planes``. The radial
breakpoints are drawn as closed curves and the poloidal knots as spokes. The two innermost rings are left out, so
the polar patch at the axis shows as one region. ``--view 3d`` writes ``mesh_3d.png``, the boundary surface over
``nfp - 1`` field periods with one period cut away to look inside, and its poloidal and toroidal knot lines. Only
the map Phi is built, no operators, so the script is quick.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Optional

import tyro


class View(StrEnum):
    TWO_D = "2d"
    THREE_D = "3d"


@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class PlotMesh:
    """Draw the spline mesh in the mapped geometry."""
    geometry: str
    """The file that defines the map, a VMEC wout (.nc), GVEC state (.dat) or DESC output (.h5)."""
    out: str
    """The directory for the figure."""
    resolution: tuple[int, int, int] = (16, 32, 32)
    """The spline resolution (r, theta, zeta)."""
    spline_degree: int = 2
    """The spline degree."""
    knots_r: Optional[tuple[float, ...]] = None
    """The radial breakpoints, increasing from 0 to 1, instead of the uniform grid. They set n_r."""
    view: View = View.TWO_D
    """2d draws the poloidal cross-sections, 3d the cut-open torus."""
    planes: tuple[float, ...] = (0.0, 0.5)
    """The logical toroidal planes of the cross-sections (2d only)."""
    precision: Literal["float32", "float64"] = "float32"
    """The floating-point precision of the map."""


def main(cli):
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.derham_sequence import DeRhamSequence
    from mrx.geometry import knot_vector
    from mrx.equilibria import build_map, read_equilibrium
    from mrx.diagnostics.plotstyle import LEFT, house_style, save_figure

    black, grey = LEFT["color"], "0.55"
    p, ns = cli.spline_degree, tuple(cli.resolution)
    bp = cli.knots_r
    if bp is not None:
        ns = (len(bp) - 1 + p,) + ns[1:]
    T = knot_vector(bp if bp is not None else np.linspace(0, 1, ns[0] - p + 1), p, False)
    eq = read_equilibrium(cli.geometry)
    seq = DeRhamSequence(ns, p, nfp=eq["nfp"], symmetry="field-period", knots=(T, None, None))
    Phi, info = build_map(eq, seq)
    Phi_v = jax.jit(jax.vmap(Phi))
    nfp, breakpoints = int(info["nfp"]), np.unique(np.asarray(T))
    label = f"({ns[0]}, {ns[1]}, {ns[2]})" + (" refined" if bp is not None else "")
    print(f"[mesh] {label}: nfp={nfp} radial cells {ns[0] - p}, "
          f"breakpoints {'given' if bp is not None else 'uniform'}", flush=True)

    section_v = jax.jit(jax.vmap(Phi.section))

    def RZ(r, th, ze):
        # the coordinates in the cross-section: (R, Z), or (X1, X2) in the plane of a G-frame
        return np.asarray(section_v(jnp.stack([jnp.asarray(r), jnp.asarray(th), jnp.asarray(ze)], axis=-1))).T

    os.makedirs(cli.out, exist_ok=True)
    with house_style():
        if cli.view == View.TWO_D:
            planes = cli.planes
            n_t, n_r = 400, 120
            fig, axes = plt.subplots(1, len(planes), figsize=(4.2 * len(planes), 4.4), squeeze=False,
                                     constrained_layout=True)
            # The polar extraction (mrx.extraction_operators) merges the two innermost radial rings into three
            # C1 basis functions, so the patch near the axis is one region, not nested rings. Those inner
            # breakpoint circles are dropped and the poloidal spokes start at the first remaining ring.
            ring_depth = 2
            r_inner = float(breakpoints[min(1 + ring_depth, len(breakpoints) - 1)])
            th = np.linspace(0.0, 1.0, n_t)
            for ax, ze in zip(axes[0], planes):
                for r in breakpoints[1 + ring_depth:]:
                    ax.plot(*RZ(np.full(n_t, min(r, 1.0 - 1e-6)), th, np.full(n_t, ze)), color=black, lw=0.5)
                rr = np.linspace(r_inner, 1.0 - 1e-6, n_r)
                for j in range(ns[1]):
                    ax.plot(*RZ(rr, np.full(n_r, j / ns[1]), np.full(n_r, ze)), color=grey, lw=0.3)
                ax.set_aspect("equal")
                ax.set_axis_off()
                ax.set_title(f"{label}, $\\zeta = {ze:g}$")
            name = "mesh_2d.png"
        else:
            # The boundary over nfp - 1 field periods, with one period cut away to look inside.
            n_line, shown = 200, max(nfp - 1, 1)
            fig = plt.figure(figsize=(8.0, 6.0))
            ax = fig.add_subplot(111, projection="3d")
            th = np.linspace(0.0, 1.0, 4 * ns[1] + 1)
            ze = np.linspace(0.0, shown, 4 * ns[2] * shown + 1)
            TH, ZE = np.meshgrid(th, ze, indexing="ij")
            pts = jnp.stack([jnp.full(TH.size, 1.0 - 1e-6), jnp.asarray(TH.ravel()), jnp.asarray(ZE.ravel())], axis=-1)
            y = np.asarray(Phi_v(pts)).reshape(*TH.shape, 3)
            ax.plot_surface(y[..., 0], y[..., 1], y[..., 2], color="0.85", alpha=0.35, linewidth=0,
                            antialiased=False, rasterized=True, shade=True)
            for j in range(ns[1]):      # poloidal knot lines (theta = const) along the torus
                zz = np.linspace(0.0, shown, n_line * shown)
                q = np.asarray(Phi_v(jnp.stack([jnp.full(zz.size, 1.0 - 1e-6), jnp.full(zz.size, j / ns[1]), jnp.asarray(zz)], -1)))
                ax.plot(q[:, 0], q[:, 1], q[:, 2], color=black, lw=0.35)
            for j in range(ns[2] * shown + 1):   # toroidal knot lines (zeta = const) around the cross-section
                tt = np.linspace(0.0, 1.0, n_line)
                q = np.asarray(Phi_v(jnp.stack([jnp.full(tt.size, 1.0 - 1e-6), jnp.asarray(tt), jnp.full(tt.size, j / ns[2])], -1)))
                ax.plot(q[:, 0], q[:, 1], q[:, 2], color=grey, lw=0.3)
            lo, hi = y.reshape(-1, 3).min(0), y.reshape(-1, 3).max(0)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo, zoom=1.35)     # equal scaling, with the torus filling the frame
            ax.view_init(elev=25, azim=40)
            ax.set_axis_off()
            ax.set_position([0.0, 0.0, 1.0, 0.95])
            ax.set_title(f"boundary of {label}: {ns[1]} poloidal and {ns[2]} toroidal knot lines per period, "
                         f"{shown} of {nfp} periods", y=0.98)
            name = "mesh_3d.png"
        save_figure(fig, os.path.join(cli.out, name))
        plt.close(fig)
    print(f"  -> {cli.out}/{name} (+ pgf/)", flush=True)


if __name__ == "__main__":
    # The precision must be in the environment before mrx is imported. The full parse follows.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--precision", default="float32", choices=("float32", "float64"))
    os.environ["MRX_DTYPE"] = pre.parse_known_args()[0].precision
    main(tyro.cli(PlotMesh, description=__doc__))
