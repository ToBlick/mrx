"""The spline meshes in the mapped geometry: how strongly shaped the domain is
and where the knots land.

    python scripts/plot_mesh.py --geometry data/wout_li383_1.4m.nc \\
        --meshes "16,32,32;32,32,32;32,32,32|0.47:0.62:6,0.68:0.94:15" --out DIR

Writes ``mesh_2d.png``: one column per mesh, one row per plane of ``--planes``,
the poloidal cross-section with the radial breakpoints as closed curves (a
``--r-refine`` window shows as denser lines, see ``mrx.geometry.radial_knots``)
and the poloidal knots as spokes. With ``--sections`` a panel is split at the
magnetic axis like the section pages: the grid above, the Poincaré crossings
of that mesh's field below, coloured by iota (``scripts/poincare_relax.py``'s
``sections.npz``; the plane must be one it traced). ``mesh_3d.png``: the boundary surface
of the first mesh over the full torus with its poloidal and toroidal knot
lines. Both also as ``pgf/*.pgf``. Only the map is built (no preconditioners).

Options
    --geometry PATH      GVEC .dat or VMEC wout .nc (``mrx.geometry.build_sequence`` names)
    --meshes SPEC        ``n_r,n_t,n_z[|a:b:m,...]`` per mesh, ``;``-separated
    --p P                spline degree [2]
    --planes Z,...       logical toroidal planes of the cross-sections [0,0.5]
    --sections S;...     per mesh ``path/sections.npz[:tag]`` (tag ``final`` by
                         default) or ``-`` for none; one entry serves every mesh
    --nfp N              override the file's nfp
    --out DIR            figure directory
    --precision {float32,float64}
"""
from __future__ import annotations

import argparse
import os


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--planes", default="0,0.5")
    ap.add_argument("--sections", default="")
    ap.add_argument("--nfp", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    return ap.parse_args(argv)


def main(cli):
    os.environ["MRX_DTYPE"] = cli.precision
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.derham_sequence import DeRhamSequence
    from mrx.geometry import parse_r_refine, radial_knots
    from mrx.gvec import build_gvec_map, read_equilibrium
    from mrx.plotstyle import LEFT, SECTION_CMAP, house_style
    from mrx.plotting import save_figure

    black, grey = LEFT["color"], "0.55"
    planes = [float(v) for v in cli.planes.split(",")]
    sections = [w for w in cli.sections.split(";") if w]
    meshes = []
    for spec in cli.meshes.split(";"):
        ns_spec, _, refine = spec.partition("|")
        ns = tuple(int(v) for v in ns_spec.split(","))
        windows = parse_r_refine(refine)
        T = radial_knots(ns[0], cli.p, windows)
        seq = DeRhamSequence(ns, (cli.p,) * 3, cli.p + 1, ("clamped", "periodic", "periodic"),
                             polar=True, knots=(T, None, None) if windows else None)
        F, info = build_gvec_map(read_equilibrium(cli.geometry), seq, nfp=cli.nfp)
        label = f"({ns[0]}, {ns[1]}, {ns[2]})" + (" refined" if windows else "")
        print(f"[mesh] {label}: nfp={info['nfp']} radial cells {ns[0] - cli.p}, windows {windows}", flush=True)
        meshes.append((label, ns, np.unique(np.asarray(T)), windows, jax.jit(jax.vmap(F)), info["nfp"]))
    if len(sections) == 1:
        sections = sections * len(meshes)
    assert not sections or len(sections) == len(meshes), "--sections: one entry per mesh (or one for all)"
    loaded = []
    for w in sections:
        if w == "-":
            loaded.append(None)
            continue
        path, _, tag = w.partition(":")
        loaded.append((np.load(path, allow_pickle=True), tag or "final"))
    iota_lim = None
    if any(loaded):
        iotas = np.concatenate([z[f"{t}_iota"][z[f"{t}_shown"]] for z, t in filter(None, loaded)])
        iota_lim = (float(iotas.min()), float(iotas.max()))

    def RZ(F, r, th, ze):
        pts = jnp.stack([jnp.asarray(r), jnp.asarray(th), jnp.asarray(ze)], axis=-1)
        y = np.asarray(F(pts))
        return np.hypot(y[:, 0], y[:, 1]), y[:, 2]

    os.makedirs(cli.out, exist_ok=True)
    with house_style():
        # --- 2-D: cross-sections, one column per mesh, one row per plane ------
        n_t, n_r = 400, 120
        fig, axes = plt.subplots(len(planes), len(meshes), figsize=(4.2 * len(meshes), 4.4 * len(planes)),
                                 squeeze=False, constrained_layout=True)
        for i, ze in enumerate(planes):
            for k, (label, ns, bp, windows, F, nfp) in enumerate(meshes):
                ax = axes[i, k]
                sec = loaded[k] if loaded else None
                z_split = None
                if sec is not None:
                    z, tag = sec
                    key = f"{tag}_zeta{ze:g}"
                    z_split = float(np.mean(z[f"{key}_axisZ"]))
                    shown = z[f"{tag}_shown"]
                    Rs, Zs = z[f"{key}_R"][shown], z[f"{key}_Z"][shown]
                    col = np.broadcast_to(z[f"{tag}_iota"][shown][:, None], Rs.shape)
                    low = Zs < z_split
                    ax.scatter(Rs[low], Zs[low], c=col[low], s=0.4, vmin=iota_lim[0], vmax=iota_lim[1],
                               cmap=SECTION_CMAP, linewidths=0, rasterized=True)

                def half(R, Z):   # the grid above the axis when the lower half is the section
                    if z_split is None:
                        return R, Z
                    return np.where(Z >= z_split, R, np.nan), np.where(Z >= z_split, Z, np.nan)

                th = np.linspace(0.0, 1.0, n_t)
                for r in bp[1:]:
                    R, Z = half(*RZ(F, np.full(n_t, min(r, 1.0 - 1e-6)), th, np.full(n_t, ze)))
                    ax.plot(R, Z, color=black, lw=0.5)
                rr = np.linspace(0.0, 1.0 - 1e-6, n_r)
                for j in range(ns[1]):
                    R, Z = half(*RZ(F, rr, np.full(n_r, j / ns[1]), np.full(n_r, ze)))
                    ax.plot(R, Z, color=grey, lw=0.3)
                if z_split is not None:
                    ax.axhline(z_split, color=black, lw=0.4, ls=":")
                ax.set_aspect("equal")
                ax.set_xlabel(r"$R$")
                ax.set_ylabel(r"$Z$")
                ax.set_title(f"{label}, $\\zeta = {ze:g}$")
        save_figure(fig, os.path.join(cli.out, "mesh_2d.png"))
        plt.close(fig)

        # --- 3-D: the boundary of the first mesh over the full torus -----------
        label, ns, bp, windows, F, nfp = meshes[0]
        n_line = 200
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        th = np.linspace(0.0, 1.0, 4 * ns[1] + 1)
        ze = np.linspace(0.0, nfp, 4 * ns[2] * nfp + 1)
        TH, ZE = np.meshgrid(th, ze, indexing="ij")
        pts = jnp.stack([jnp.full(TH.size, 1.0 - 1e-6), jnp.asarray(TH.ravel()), jnp.asarray(ZE.ravel())], axis=-1)
        y = np.asarray(F(pts)).reshape(*TH.shape, 3)
        ax.plot_surface(y[..., 0], y[..., 1], y[..., 2], color="0.85", alpha=0.35, linewidth=0,
                        antialiased=False, rasterized=True, shade=True)
        for j in range(ns[1]):      # poloidal knot lines (theta = const) around the torus
            zz = np.linspace(0.0, nfp, n_line * nfp)
            p = np.asarray(F(jnp.stack([jnp.full(zz.size, 1.0 - 1e-6), jnp.full(zz.size, j / ns[1]), jnp.asarray(zz)], -1)))
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=black, lw=0.35)
        for j in range(ns[2] * nfp):   # toroidal knot lines (zeta = const) around the cross-section
            tt = np.linspace(0.0, 1.0, n_line)
            p = np.asarray(F(jnp.stack([jnp.full(tt.size, 1.0 - 1e-6), jnp.asarray(tt), jnp.full(tt.size, j / ns[2])], -1)))
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=grey, lw=0.3)
        lo, hi = y.reshape(-1, 3).min(0), y.reshape(-1, 3).max(0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.35)     # equal scaling, the torus filling the frame
        ax.view_init(elev=25, azim=40)
        ax.set_axis_off()
        ax.set_position([0.0, 0.0, 1.0, 0.95])
        ax.set_title(f"boundary of {label}: {ns[1]} poloidal and {ns[2] * nfp} toroidal knot lines", y=0.98)
        save_figure(fig, os.path.join(cli.out, "mesh_3d.png"))
        plt.close(fig)
    print(f"  -> {cli.out}/mesh_2d.png, mesh_3d.png (+ pgf/)", flush=True)


if __name__ == "__main__":
    main(parse_args())
