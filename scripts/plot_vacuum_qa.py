"""The vacuum-convergence test case as a picture: ``|B|`` on the device boundary.

``LandremanPaul2021_QA`` is a vacuum equilibrium, so its field is the harmonic
2-form of the Dirichlet de Rham complex (curl-free, div-free, ``B . n = 0``) --
the same field ``scripts/analytic_vacuum.py`` / ``scripts/vacuum_convergence.py``
converge to. This builds that field once (a Hodge decomposition, no eigen-
iteration) and draws its magnitude on the boundary surface of the whole device,
one colour scale, styled like the ``mesh_3d`` boundary render: full torus, no
axes, ``plasma`` by ``|B|``. Writes ``vacuum_qa_Bmag.png`` (+ ``pgf/``).

    python scripts/plot_vacuum_qa.py --geometry data/wout_LandremanPaul2021_QA_lowres.nc --out DIR

Options
    --geometry PATH      VMEC wout .nc or GVEC .dat (``mrx.geometry`` names)
    --ns N_R,N_T,N_Z     resolution [12,24,12]
    --p P                spline degree [3]
    --out DIR            figure directory
    --precision {float32,float64}
"""
from __future__ import annotations

import argparse
import os


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    return ap.parse_args(argv)


def main(cli):
    os.environ["MRX_DTYPE"] = cli.precision
    import jax
    import jax.numpy as jnp
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    import mrx
    from mrx.differential_forms import DiscreteFunction, Pushforward
    from mrx.geometry import build_sequence, geometry_nfp
    from mrx.nullspace import compute_nullspaces, get_nullspace
    from mrx.plotstyle import FIELD_CMAP, house_style
    from mrx.plotting import get_2d_grids, save_figure, set_axes_equal

    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)
    nfp = geometry_nfp(cli.geometry)
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    compute_nullspaces(seq)

    # The vacuum field: the harmonic 2-form, L2-normalised, pushed forward by the
    # Piola map so ``B_mag`` is the physical |B| at a logical point (see tutorial 2).
    B = get_nullspace(seq.get_operators(), 2, True)[0]
    B = B / float(seq.l2_norm(B, 2))
    B_phys = Pushforward(DiscreteFunction(B, seq.basis_2, seq.E(2, True)), seq.map, 2)

    def B_mag(x):
        return jnp.linalg.norm(B_phys(x))

    # One field period of the boundary surface (rho = 1), coloured by |B|; the
    # whole device is that period rotated nfp times about z (sign -1 for the
    # (R cos, -R sin) GVEC convention, as in mrx.plotting.plot_torus).
    with house_style():
        g = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                         ny=4 * ns[1], nz=4 * ns[2], invert_z=True)
        vals = np.asarray(jax.lax.map(B_mag, g[0], batch_size=mrx.MAP_BATCH_SIZE_INNER)
                          ).reshape(g[2][0].shape)
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(FIELD_CMAP)
        colours = cmap(norm(vals))
        print(f"[vacuum] {ns} p={cli.p} nfp={nfp}: |B| on boundary in [{vmin:.3f}, {vmax:.3f}]", flush=True)

        fig = plt.figure(figsize=(9.0, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        pts = []
        for j in range(nfp):
            angle = -2.0 * np.pi * j / nfp
            X, Y, Z = g[2]
            c, s = np.cos(angle), np.sin(angle)
            Xr, Yr = c * X - s * Y, s * X + c * Y
            ax.plot_surface(Xr, Yr, Z, facecolors=colours, rstride=1, cstride=1,
                            shade=False, linewidth=0, antialiased=False, zsort="min")
            pts.append(np.stack([Xr, Yr, Z], axis=-1).reshape(-1, 3))

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(vals)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(r"$|B|$")

        lo, hi = np.concatenate(pts).min(0), np.concatenate(pts).max(0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.3)
        ax.view_init(elev=25, azim=40)
        ax.set_axis_off()
        ax.set_title(f"QA vacuum field: $|B|$ on the boundary  ({ns[0]}, {ns[1]}, {ns[2]}), $p = {cli.p}$", y=0.97)
        save_figure(fig, os.path.join(cli.out, "vacuum_qa_Bmag.png"))
        plt.close(fig)
    print(f"  -> {cli.out}/vacuum_qa_Bmag.png (+ pgf/)", flush=True)


if __name__ == "__main__":
    main(parse_args())
