"""The NCSX boundary mesh in 3-D, cut open at two toroidal planes that face the
viewer, with a Poincaré section drawn on each cut face.

The camera looks along the toroidal unit vector at the front plane, so the
front cut face (logical zeta = 0, physical angle 0) and the back one (half a
device further, angle pi, which is the traced plane zeta = 0.5 of the next
field period) are both seen face-on; the near half of the torus is removed
so nothing hides them. The surface and the knot lines are those of
``scripts/plot_mesh.py``'s ``mesh_3d.png``; the crossings come from a
``sections.npz`` of ``scripts/poincare_relax.py`` (physical R, Z per plane),
coloured by the line's rotational transform on the section colour scale.

    python mesh_sections_3d.py --geometry data/wout_li383_1.4m.nc --ns 16,32,32 --p 2 \\
        --sections outputs/reconnect_2026-09/ladder/h16_p2/poincare/sections.npz[:final] --out DIR
"""
from __future__ import annotations

import argparse
import os


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--ns", default="16,32,32")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--sections", required=True, help="path/sections.npz[:tag], tag 'final' by default")
    ap.add_argument("--front", type=float, default=0.0, help="logical zeta of the front face (traced plane)")
    ap.add_argument("--back", type=float, default=0.5, help="logical zeta of the back face (traced plane)")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--colour", default="iota,p", help="what colours the front and the back face: iota or p")
    ap.add_argument("--line-stride", type=int, default=4, help="keep every k-th traced line (160 traced; 4 -> 40)")
    ap.add_argument("--crossings", type=int, default=400, help="crossings kept per line")
    ap.add_argument("--nfp", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    return ap.parse_args(argv)


def main(cli):
    os.environ["MRX_DTYPE"] = cli.precision
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.derham_sequence import DeRhamSequence
    from mrx.gvec import build_gvec_map, read_equilibrium
    from mrx.plotstyle import LEFT, PRESSURE_CMAP, SECTION_CMAP, house_style
    from mrx.plotting import save_figure

    black, grey = LEFT["color"], "0.55"
    ns = tuple(int(v) for v in cli.ns.split(","))
    seq = DeRhamSequence(ns, (cli.p,) * 3, cli.p + 1, ("clamped", "periodic", "periodic"), polar=True)
    F0, info = build_gvec_map(read_equilibrium(cli.geometry), seq, nfp=cli.nfp)
    F = jax.jit(jax.vmap(F0))
    nfp = info["nfp"]
    print(f"[mesh] ({ns[0]}, {ns[1]}, {ns[2]}) p={cli.p}, nfp={nfp}", flush=True)

    def xyz(r, th, ze):
        pts = jnp.stack([jnp.asarray(r), jnp.asarray(th), jnp.asarray(ze)], axis=-1)
        return np.asarray(F(pts))

    # The map's toroidal angle is sign * 2 pi zeta / nfp; a logical zeta in [0, nfp) covers the device.
    # The camera sits on the toroidal unit vector of the front face; the far half of the device,
    # the half whose points lie behind the plane through the axis of symmetry and the front face,
    # is the half-device zeta in [front, front + nfp/2] (either handedness: it is the half the
    # front face's tangent points into).
    z_lo, z_hi = cli.front, cli.front + nfp / 2.0
    probe = xyz(np.full(2, 1.0 - 1e-6), np.zeros(2), np.array([z_lo, z_lo + 0.05]))
    t = probe[1, :2] - probe[0, :2]                     # the tube's direction at the front face, in the plane
    t /= np.linalg.norm(t)
    azim = float(np.degrees(np.arctan2(-t[1], -t[0])))  # camera opposite to where the tube goes: it goes away from us
    print(f"[view] tube direction at the front face {t}, camera azimuth {azim:.1f} deg", flush=True)

    # the sections: physical (R, Z) of the crossings on the two planes, and the angle of each face
    path, _, tag = cli.sections.partition(":")
    z = np.load(path, allow_pickle=True)
    tag = tag or "final"
    shown = np.nonzero(z[f"{tag}_shown"])[0][::cli.line_stride]     # a thinned set of the traced lines
    iota = z[f"{tag}_iota"][shown]
    nc = cli.crossings
    print(f"[lines] {len(shown)} of {int(z[f'{tag}_shown'].sum())} traced lines, {nc} crossings each", flush=True)
    colours = [c.strip() for c in cli.colour.split(",")]
    keys = [f"{tag}_zeta{cli.front % 1.0:g}", f"{tag}_zeta{cli.back % 1.0:g}"]
    # one scale per quantity over both faces: iota from the fit, p from the crossings (x100)
    p_all = np.concatenate([z[f"{k}_pressure"][shown][:, :nc].ravel() for k in keys])
    p_all = p_all[np.isfinite(p_all)]
    scales = {"iota": (float(iota.min()), float(iota.max()), SECTION_CMAP, r"$\iota$"),
              "p": (float(100 * np.nanmin(p_all)), float(100 * np.nanmax(p_all)), PRESSURE_CMAP, r"$p \times 100$")}

    def face(key, zeta_device, what):
        R, Z = z[f"{key}_R"][shown][:, :nc], z[f"{key}_Z"][shown][:, :nc]
        # the physical angle of this face from the map itself (its axis point at theta = 0, r -> 0)
        a = xyz(np.full(1, 1e-3), np.zeros(1), np.array([zeta_device]))[0]
        phi = float(np.arctan2(a[1], a[0]))
        col = np.broadcast_to(iota[:, None], R.shape) if what == "iota" else 100 * z[f"{key}_pressure"][shown][:, :nc]
        return R * np.cos(phi), R * np.sin(phi), Z, col, what

    faces = [face(keys[0], z_lo, colours[0]), face(keys[1], z_hi, colours[1])]

    os.makedirs(cli.out, exist_ok=True)
    for rasterized, suffix in ((True, ""), (False, "_vector")):
        render(cli, faces, scales, xyz, ns, nfp, z_lo, z_hi, azim, black, grey, rasterized, suffix,
               house_style, plt, np, save_figure)


def render(cli, faces, scales, xyz, ns, nfp, z_lo, z_hi, azim, black, grey, rasterized, suffix,
           house_style, plt, np, save_figure):
    """One drawing: the far half's knot lines, the two cut faces, the sections as rasterized
    dots (``suffix`` '') or as vector points ('_vector'); PNG with inline bars, PDF of the
    drawing alone, and one PDF per colour bar."""
    import os
    with house_style():
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        th = np.linspace(0.0, 1.0, 4 * ns[1] + 1)
        n_line = 200                                      # no surface: the outermost grid and the sections only
        for j in range(ns[1]):                            # poloidal knot lines along the far half
            zz = np.linspace(z_lo, z_hi, n_line)
            p = xyz(np.full(zz.size, 1.0 - 1e-6), np.full(zz.size, j / ns[1]), zz)
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=black, lw=0.35)
        kn = np.arange(np.ceil(z_lo * ns[2]), np.floor(z_hi * ns[2]) + 1) / ns[2]
        for zk in kn:                                     # toroidal knot lines on the far half
            tt = np.linspace(0.0, 1.0, n_line)
            p = xyz(np.full(tt.size, 1.0 - 1e-6), tt, np.full(tt.size, zk))
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=grey, lw=0.3)
        for zk in (z_lo, z_hi):                           # the two cut faces' outlines
            tt = np.linspace(0.0, 1.0, 2 * n_line)
            p = xyz(np.full(tt.size, 1.0 - 1e-6), tt, np.full(tt.size, zk))
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=black, lw=1.0)
        bars = {}
        for X, Y, Z, col, what in faces:                  # the sections on the cut faces
            vmin, vmax, cmap, label = scales[what]
            bars[what] = (ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=col.ravel(), s=0.8, vmin=vmin, vmax=vmax,
                                     cmap=cmap, linewidths=0, rasterized=rasterized, depthshade=False), label)
        order = list(bars)                                 # front face first, then back
        # full-device extent for equal scaling (both halves), so the frame is the whole torus
        zf = np.linspace(0.0, nfp, 4 * ns[2] * nfp + 1)
        THf, ZEf = np.meshgrid(th, zf, indexing="ij")
        yf = xyz(np.full(THf.size, 1.0 - 1e-6), THf.ravel(), ZEf.ravel())
        lo, hi = yf.min(0), yf.max(0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.12)                # inside the square 3-D drawing box; the PDF is trimmed to the artists
        ax.view_init(elev=cli.elev, azim=azim)
        ax.set_proj_type("ortho")                         # no perspective: both faces at their true size
        ax.set_axis_off()
        ax.set_position([0.0, 0.0, 0.88, 1.0])
        n_bars = len(bars)
        caxes = []
        for i, (sc, label) in enumerate(bars.values()):   # stacked on the right, top to bottom (the PNG only)
            top, height = 0.86 - i * (0.72 / n_bars), 0.72 / n_bars - 0.08
            cax = fig.add_axes([0.9, top - height, 0.015, height])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label(label)
            caxes.append(cax)
        save_figure(fig, os.path.join(cli.out, f"mesh_sections_3d{suffix}.png"), pgf=rasterized)
        for cax in caxes:                                 # the PDF: the drawing alone, page fitted to it
            cax.remove()
        fig.savefig(os.path.join(cli.out, f"mesh_sections_3d{suffix}.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        if not rasterized:
            return
        for what in order:                                # each colour bar its own PDF, ticks only, no label
            sc, label = bars[what]
            fb = plt.figure(figsize=(0.5, 3.0))
            cax = fb.add_axes([0.05, 0.03, 0.3, 0.94])
            fb.colorbar(sc, cax=cax)
            fb.savefig(os.path.join(cli.out, f"cbar_{what}.pdf"), bbox_inches="tight", pad_inches=0.02)
            plt.close(fb)
    print(f"  -> {cli.out}/mesh_sections_3d{{,_vector}}.png (bars inline), .pdf (drawing only), "
          f"cbar_{order[0]}.pdf, cbar_{order[1]}.pdf (+ pgf/)", flush=True)


if __name__ == "__main__":
    main(parse_args())
