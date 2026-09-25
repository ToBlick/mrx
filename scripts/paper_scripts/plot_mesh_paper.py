"""The spline meshes in the mapped geometry: how strongly shaped the domain is
and where the knots land.

    python scripts/plot_mesh.py --geometry data/wout_li383_1.4m.nc \\
        --meshes "16,32,32;32,32,32;16,32,32|0,0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1" --out DIR

Writes ``mesh_2d.png``: one column per mesh, one row per plane of ``--planes``,
the poloidal cross-section with the radial breakpoints as closed curves
(breakpoints given after ``|`` show as denser lines where they crowd) and the
poloidal knots as spokes. Panels carry no axes or frame; the two
innermost rings are omitted and the spokes start at the first surviving ring,
so the near-axis polar patch (the ``ring_depth=2`` -> three C1 basis functions
surgery of ``mrx.extraction_operators``) reads as one region. With ``--sections`` a panel is split at the
magnetic axis like the section pages: the grid above, the Poincaré crossings
of that mesh's field below, coloured by iota (``scripts/poincare_trace.py``'s
``trace.npz``; the plane must be one it traced). ``mesh_3d.png``: the boundary surface
of the first mesh over the full torus with its poloidal and toroidal knot
lines. Both also as ``pgf/*.pgf``. Only the map is built (no preconditioners).

Options
    --geometry PATH      GVEC .dat or VMEC wout .nc (``mrx.geometry.build_sequence`` names)
    --meshes SPEC        ``n_r,n_t,n_z[|breakpoints]`` per mesh, ``;``-separated;
                         radial breakpoints (comma list, 0 to 1) replace the
                         uniform grid and set ``n_r`` (cells + p)
    --p P                spline degree [2]
    --planes Z,...       logical toroidal planes of the cross-sections [0,0.5]
    --sections S;...     per mesh ``path/trace.npz[:tag]`` (tag ``final`` by
                         default) or ``-`` for none; one entry serves every mesh
    --nfp N              override the file's nfp
    --coefficients C;... per mesh ``path.npz[:start|end]`` or ``-``: the map from the raw ``R``, ``Z``
                         coefficients of a record (``raw_R0``/``raw_Z0`` at the start, ``raw_R1``/``raw_Z1``
                         at the end [end], on that mesh's spline space) instead of --geometry's; --geometry
                         still gives nfp and the handedness. One entry serves every mesh
    --out DIR            figure directory
    --precision {float32,float64}
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class PlotMesh:
    """The spline meshes in the mapped geometry."""
    geometry: str = field(metadata=dict(help="GVEC .dat or VMEC wout .nc (mrx.geometry.build_sequence names)"))
    meshes: str = field(metadata=dict(
        help="n_r,n_t,n_z[|breakpoints] per mesh, ;-separated; radial breakpoints (comma list, 0 to 1) replace "
             "the uniform grid and set n_r (cells + p)"))
    out: str = field(metadata=dict(help="figure directory"))
    p: int = field(default=2, metadata=dict(help="spline degree"))
    planes: str = field(default="0,0.5", metadata=dict(help="logical toroidal planes of the cross-sections"))
    sections: str = field(default="", metadata=dict(
        help="per mesh path/trace.npz[:tag] (tag final by default) or - for none; one entry serves every mesh"))
    nfp: Optional[int] = field(default=None, metadata=dict(help="override the file's nfp"))
    coefficients: str = field(default="", metadata=dict(
        help="per mesh path.npz[:start|end] or -: the map from the raw R, Z coefficients of a record instead of "
             "--geometry's; one entry serves every mesh"))
    precision: Literal["float32", "float64"] = field(default="float32", metadata=dict(help="the map's precision"))


def main(cli):
    os.environ["MRX_DTYPE"] = cli.precision
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.derham_sequence import DeRhamSequence
    from mrx.geometry import knot_vector, parse_knots
    from mrx.gvec import build_gvec_map, read_equilibrium
    from mrx.plotstyle import LEFT, SECTION_CMAP, house_style
    from mrx.plotting import save_figure

    black, grey = LEFT["color"], "0.55"
    planes = [float(v) for v in cli.planes.split(",")]
    sections = [w for w in cli.sections.split(";") if w]
    coefficients = [w for w in cli.coefficients.split(";") if w]
    meshes = []
    for spec in cli.meshes.split(";"):
        ns_spec, _, refine = spec.partition("|")
        ns = tuple(int(v) for v in ns_spec.split(","))
        bp = parse_knots(refine)
        if bp is not None:
            ns = (len(bp) - 1 + cli.p,) + ns[1:]
        T = knot_vector(bp if bp is not None else np.linspace(0, 1, ns[0] - cli.p + 1), cli.p, False)
        seq = DeRhamSequence(ns, (cli.p,) * 3, cli.p + 1, ("clamped", "periodic", "periodic"),
                             polar=True, knots=(T, None, None))
        F, info = build_gvec_map(read_equilibrium(cli.geometry), seq, nfp=cli.nfp)
        entry = coefficients[min(len(meshes), len(coefficients) - 1)] if coefficients else "-"
        if entry != "-":
            path, _, when = entry.partition(":")
            rec = np.load(path)
            k = "0" if when == "start" else "1"
            raw_R, raw_Z = jnp.asarray(rec[f"raw_R{k}"]), jnp.asarray(rec[f"raw_Z{k}"])
            basis, a, sgn = seq.basis_0.bases[0], 2.0 * np.pi / info["nfp"], info["sign"]

            def F(x, raw_R=raw_R, raw_Z=raw_Z, basis=basis, a=a, sgn=sgn):
                r = basis.contract(raw_R, x)
                return jnp.array([r * jnp.cos(a * x[2]), sgn * r * jnp.sin(a * x[2]), basis.contract(raw_Z, x)])
        label = f"({ns[0]}, {ns[1]}, {ns[2]})" + (" refined" if bp is not None else "")
        print(f"[mesh] {label}: nfp={info['nfp']} radial cells {ns[0] - cli.p}, "
              f"breakpoints {'given' if bp is not None else 'uniform'}", flush=True)
        meshes.append((label, ns, np.unique(np.asarray(T)), jax.jit(jax.vmap(F)), info["nfp"]))
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
            for k, (label, ns, bp, F, nfp) in enumerate(meshes):
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

                # The polar surgery (mrx.extraction_operators) fuses the two
                # innermost radial rings into three C1 basis functions, so the
                # patch near the axis is one region, not nested rings: drop those
                # inner breakpoint circles and start the poloidal spokes at the
                # first surviving ring rather than fanning them into the patch.
                ring_depth = 2
                r_inner = float(bp[min(1 + ring_depth, len(bp) - 1)])
                th = np.linspace(0.0, 1.0, n_t)
                for r in bp[1 + ring_depth:]:
                    R, Z = half(*RZ(F, np.full(n_t, min(r, 1.0 - 1e-6)), th, np.full(n_t, ze)))
                    ax.plot(R, Z, color=black, lw=0.5)
                rr = np.linspace(r_inner, 1.0 - 1e-6, n_r)
                for j in range(ns[1]):
                    R, Z = half(*RZ(F, rr, np.full(n_r, j / ns[1]), np.full(n_r, ze)))
                    ax.plot(R, Z, color=grey, lw=0.3)
                if z_split is not None:
                    ax.axhline(z_split, color=black, lw=0.4, ls=":")
                ax.set_aspect("equal")
                ax.set_axis_off()
                ax.set_title(f"{label}, $\\zeta = {ze:g}$")
        save_figure(fig, os.path.join(cli.out, "mesh_2d.png"))
        plt.close(fig)

        # --- 3-D: the boundary of the first mesh over the full torus -----------
        label, ns, bp, F, nfp = meshes[0]
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
    # the precision must be in the environment before mrx is imported; the full parse then follows
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--precision", default="float32", choices=("float32", "float64"))
    os.environ["MRX_DTYPE"] = pre.parse_known_args()[0].precision
    from mrx.cli import parse
    main(parse(PlotMesh, description=__doc__))
