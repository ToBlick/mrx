#!/usr/bin/env python
"""Figure 1: the device's boundary mesh in 3-D, cut open at two toroidal planes that face the viewer, with the
Poincare section of a relaxed run drawn on each cut face (GPU: it rebuilds the run's map).

    python paper/figure1_mesh3d.py --run RUN --archive RUN/trace.npz --pressure-factor F --out DIR

The camera looks along the toroidal unit vector at the front plane (logical zeta = 0), so the front face and the
back face half a device further (the traced plane zeta = 0.5 of the next field period) are both seen face-on; the
near half of the torus is removed. The front face is coloured by the rotational transform of each line, the back one
by its normalised pressure p_norm = factor * p, on the scales of scripts/poincare_plot.py over every plane of the
archive (iota over the shown lines, p over the kept lines), so the colours match the paper's Poincare pages. Every
kept line is drawn, chaotic ones included, thinned by --line-stride. Writes the drawing alone (mesh_sections_3d.pdf)
and one tick-only PDF per colour bar (cbar_iota.pdf, cbar_pnorm.pdf); figure1_standalone.tex composes them.
"""
import argparse
import json
import os

import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the run directory (relax.json: geometry, ns, p, symmetry)")
    ap.add_argument("--archive", required=True, help="its trace.npz (one field)")
    ap.add_argument("--pressure-factor", type=float, required=True, help="p_norm = factor * p (poincare_pages.py)")
    ap.add_argument("--front", type=float, default=0.0, help="logical zeta of the front face (a traced plane)")
    ap.add_argument("--back", type=float, default=0.5, help="logical zeta of the back face (a traced plane)")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--line-stride", type=int, default=6, help="draw every k-th kept line")
    ap.add_argument("--crossings", type=int, default=400, help="crossings drawn per line")
    ap.add_argument("--dot-size", type=float, default=0.4, help="section marker area, pt^2")
    ap.add_argument("--opaque", action="store_true",
                    help="a covering surface and caps on the cut faces, and only the knot lines facing the camera")
    ap.add_argument("--surface-color", default="0.95", help="colour of the covering surface and the caps")
    ap.add_argument("--mesh-lw", type=float, default=0.25, help="width of every mesh line, pt")
    ap.add_argument("--knot-stride", type=int, default=2, help="draw every k-th knot line of the run's mesh")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(cli):
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.geometry import build_sequence
    from mrx.plotstyle import LEFT, PRESSURE_CMAP, SECTION_CMAP, house_style

    attrs = json.load(open(os.path.join(cli.run, "relax.json")))["params"]
    ns = tuple(int(v) for v in attrs["ns"])
    seq, _ = build_sequence(str(attrs["geometry_path"]), ns, int(attrs["p"]), nfp=attrs.get("nfp"),
                            knots=attrs.get("knots"), symmetry=attrs.get("symmetry", "stellarator"))
    F = jax.jit(jax.vmap(seq.map))
    nfp = int(seq.nfp)
    print(f"[mesh] {os.path.basename(str(attrs['geometry_path']))} {ns} p={attrs['p']}, nfp={nfp}", flush=True)

    def xyz(r, th, ze):
        return np.asarray(F(jnp.stack([jnp.asarray(r), jnp.asarray(th), jnp.asarray(ze)], axis=-1)))

    # The far half of the device, zeta in [front, front + nfp/2], is the half the front face's tangent points into;
    # the camera sits opposite to it.
    z_lo, z_hi = cli.front, cli.front + nfp / 2.0
    probe = xyz(np.full(2, 1.0 - 1e-6), np.zeros(2), np.array([z_lo, z_lo + 0.05]))
    t = probe[1, :2] - probe[0, :2]
    t /= np.linalg.norm(t)
    azim = float(np.degrees(np.arctan2(-t[1], -t[0])))

    z = np.load(cli.archive)
    (tag,) = [str(v) for v in z["fields"]]
    keep, shown, iota_all = z[f"{tag}_keep"], z[f"{tag}_shown"], z[f"{tag}_iota"]
    # the scales of scripts/poincare_plot.py over every plane: iota over the shown lines, the weak pressure (gauge 0)
    # over the kept lines, x 100, padded 5 %
    i_lo, i_hi = float(iota_all[shown].min()), float(iota_all[shown].max())
    ps = [100.0 * cli.pressure_factor * z[f"{tag}_zeta{pl:g}_pressure"][keep] for pl in z["planes"]]
    p0, p1 = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
    scales = {"iota": (i_lo, i_hi, SECTION_CMAP), "p": (p0 - 0.05 * (p1 - p0), p1 + 0.05 * (p1 - p0), PRESSURE_CMAP)}
    lines = np.nonzero(keep)[0][::cli.line_stride]
    nc = cli.crossings
    print(f"[lines] {len(lines)} of {int(keep.sum())} kept lines, {nc} crossings each; iota {i_lo:.4f}..{i_hi:.4f}, "
          f"p_norm x 100 {scales['p'][0]:.3g}..{scales['p'][1]:.3g}", flush=True)

    def face(plane, zeta_device, what):
        key = f"{tag}_zeta{plane % 1.0:g}"
        R, Z = z[f"{key}_R"][lines][:, :nc], z[f"{key}_Z"][lines][:, :nc]
        a = xyz(np.full(1, 1e-3), np.zeros(1), np.array([zeta_device]))[0]      # the face's angle, from its axis point
        phi = float(np.arctan2(a[1], a[0]))
        col = (np.broadcast_to(iota_all[lines][:, None], R.shape) if what == "iota"
               else 100.0 * cli.pressure_factor * z[f"{key}_pressure"][lines][:, :nc])
        return R * np.cos(phi), R * np.sin(phi), Z, col, what

    faces = [face(cli.front, z_lo, "iota"), face(cli.back, z_hi, "p")]
    os.makedirs(cli.out, exist_ok=True)
    # one thin solid style for every mesh line (the house property cycle would dash them)
    mesh = dict(color=LEFT["color"], lw=cli.mesh_lw, ls="-", zorder=2)
    r1 = 1.0 - 1e-6
    elev, az = np.radians(cli.elev), np.radians(azim)
    eye = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])    # toward the viewer

    def normals(th, ze):
        """The boundary's normal d_theta x d_zeta at (1, th, ze), and the vector from the axis to the point."""
        d = 1e-4
        pt, pz = xyz(np.full(th.size, r1), th + d, ze), xyz(np.full(th.size, r1), th - d, ze)
        qt, qz = xyz(np.full(th.size, r1), th, ze + d), xyz(np.full(th.size, r1), th, ze - d)
        out = xyz(np.full(th.size, r1), th, ze) - xyz(np.full(th.size, 1e-3), np.zeros(th.size), ze)
        return np.cross(pt - pz, qt - qz), out

    # The parametrization orients every normal alike, so its sign is fixed once over the whole far half, from the
    # majority pointing away from the axis (a per-point test flips them in the bean's concave dent).
    gt, gz = (v.ravel() for v in np.meshgrid(np.linspace(0.0, 1.0, 64), np.linspace(z_lo, z_hi, 64), indexing="ij"))
    n_g, out_g = normals(gt, gz)
    orient = float(np.sign(np.median(np.einsum("ij,ij->i", n_g, out_g))))

    def facing(th, ze):
        """True where the boundary's outward normal faces the camera (orthographic, equal box aspect)."""
        return orient * (normals(th, ze)[0] @ eye) > 0.0

    def line(th, ze):
        p = xyz(np.full(th.size, r1), th, ze)
        if not cli.opaque:
            ax.plot(p[:, 0], p[:, 1], p[:, 2], **mesh)
            return
        vis = facing(th, ze)                                                      # the visible runs only
        edges = np.flatnonzero(np.diff(np.r_[0, vis.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > 1:
                ax.plot(p[a:b, 0], p[a:b, 1], p[a:b, 2], **mesh)

    with house_style():
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False                                                # surface, then lines, then sections
        n_line = 200
        if cli.opaque:                                                            # the far half's boundary, covering
            TH, ZE = np.meshgrid(np.linspace(0.0, 1.0, 4 * ns[1] + 1),
                                 np.linspace(z_lo, z_hi, int(2 * ns[2] * nfp) + 1), indexing="ij")
            S = xyz(np.full(TH.size, r1), TH.ravel(), ZE.ravel()).reshape(TH.shape + (3,))
            ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], color=cli.surface_color, shade=False, linewidth=0,
                            antialiased=False, zorder=1)
            for zk in (z_lo, z_hi):                                               # caps on the cut faces: over the
                RR, TT = np.meshgrid(np.linspace(1e-3, r1, 48), np.linspace(0.0, 1.0, 4 * ns[1] + 1), indexing="ij")
                C = xyz(RR.ravel(), TT.ravel(), np.full(RR.size, zk)).reshape(RR.shape + (3,))
                ax.plot_surface(C[..., 0], C[..., 1], C[..., 2], color=cli.surface_color, shade=False, linewidth=0,
                                antialiased=False, zorder=2.5)                  # grid lines, under the sections
        for j in range(0, ns[1], cli.knot_stride):                               # poloidal knot lines, far half
            zz = np.linspace(z_lo, z_hi, n_line)
            line(np.full(zz.size, j / ns[1]), zz)
        kn = np.arange(np.ceil(z_lo * ns[2]), np.floor(z_hi * ns[2]) + 1, cli.knot_stride) / ns[2]
        for zk in kn:                                                             # toroidal knot lines, far half
            tt = np.linspace(0.0, 1.0, n_line)
            line(tt, np.full(tt.size, zk))
        for zk in (z_lo, z_hi):                                                   # the two cut faces' outlines, whole
            tt = np.linspace(0.0, 1.0, 2 * n_line)
            p = xyz(np.full(tt.size, r1), tt, np.full(tt.size, zk))
            ax.plot(p[:, 0], p[:, 1], p[:, 2], **{**mesh, "zorder": 2.6})     # over the caps
        bars = {}
        for X, Y, Zc, col, what in faces:
            vmin, vmax, cmap = scales[what]
            bars[what] = ax.scatter(X.ravel(), Y.ravel(), Zc.ravel(), c=col.ravel(), s=cli.dot_size, vmin=vmin, vmax=vmax,
                                    cmap=cmap, linewidths=0, rasterized=True, depthshade=False, zorder=3)
        th = np.linspace(0.0, 1.0, 4 * ns[1] + 1)                                 # the whole device's extent
        THf, ZEf = np.meshgrid(th, np.linspace(0.0, nfp, 4 * ns[2] * nfp + 1), indexing="ij")
        yf = xyz(np.full(THf.size, 1.0 - 1e-6), THf.ravel(), ZEf.ravel())
        lo, hi = yf.min(0), yf.max(0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.12)
        ax.view_init(elev=cli.elev, azim=azim)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        fig.savefig(os.path.join(cli.out, "mesh_sections_3d.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(os.path.join(cli.out, "mesh_sections_3d.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        for what, name in (("iota", "cbar_iota"), ("p", "cbar_pnorm")):          # tick-only bars, labels in LaTeX
            fb = plt.figure(figsize=(0.5, 3.0))
            cax = fb.add_axes([0.05, 0.03, 0.3, 0.94])
            fb.colorbar(bars[what], cax=cax)
            fb.savefig(os.path.join(cli.out, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
            plt.close(fb)
    print(f"  -> {cli.out}/mesh_sections_3d.{{pdf,png}}, cbar_iota.pdf, cbar_pnorm.pdf", flush=True)


if __name__ == "__main__":
    main(parse_args())
