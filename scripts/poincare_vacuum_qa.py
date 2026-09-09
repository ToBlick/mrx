"""Poincaré sections of the QA vacuum solution -- the harmonic 2-form, no pressure.

The vacuum field of ``LandremanPaul2021_QA`` is the harmonic 2-form of the
Dirichlet de Rham complex (built here as in ``scripts/tutorials/2_qa_vacuum_field.py``:
``compute_nullspaces`` + ``get_nullspace``). This traces its field lines once and
renders the five toroidal planes with ``mrx.plotting.render_section`` -- the same
plotter ``scripts/poincare_relax.py`` drives -- but with ``pressure=None`` (a
vacuum carries no pressure, so the pressure colour split, the below-axis scatter
and the pressure profile are all omitted), no iota +-band ribbon and no title.
One iota colour scale across all five planes. The house font hierarchy (label >
tick > legend) is rescaled so the axis labels are ``--label-size`` when the figure
is one page wide. Writes ``poincare_zeta*.pdf`` (and a ``.png`` for viewing) into ``--out``.

    python scripts/poincare_vacuum_qa.py --geometry data/wout_LandremanPaul2021_QA_lowres.nc --out DIR

Options
    --geometry PATH      VMEC wout .nc or GVEC .dat (``mrx.geometry`` names)
    --ns N_R,N_T,N_Z     resolution [12,24,12]
    --p P                spline degree [3]
    --seeds N            field lines seeded from the axis [40]
    --rays N             seed rays [4]
    --periods N          field periods traced per line [400]
    --steps N            integration steps per period [24]
    --saves N            saved crossings per period [8]
    --r-max R            outermost seed radius [0.97]
    --profile-rays N     golden-angle poloidal rays on the profile panel [3]
    --label-size F       axis-label pt at --page-width; ticks/legend proportional [6]
    --page-width W       authored figure width in inches, 'one page wide' [6.5]
    --out DIR            figure directory
    --precision {float64,float32}
"""
from __future__ import annotations

import argparse
import os


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--rays", type=int, default=4)
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--saves", type=int, default=8)
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--profile-rays", type=int, default=3)
    ap.add_argument("--label-size", type=float, default=6.0,
                    help="axis-label size in pt at --page-width; ticks and the legend "
                         "keep the house hierarchy, scaled to this [6]")
    ap.add_argument("--page-width", type=float, default=6.5,
                    help="authored figure width in inches ('one page wide'); the fonts "
                         "read at their pt size when the figure is included at this width [6.5]")
    ap.add_argument("--dpi", type=int, default=600, help="rasterised crossing-scatter resolution [600]")
    ap.add_argument("--field-npz", default=None,
                    help="load the harmonic 2-form DOF vector from this .npz instead of "
                         "solving (e.g. a vacuum_convergence rung's fields.npz); --geometry/--ns/--p must match")
    ap.add_argument("--field-key", default="h_dof", help="array name in --field-npz [h_dof]")
    ap.add_argument("--drop-inner", type=int, default=0,
                    help="omit the N innermost radial lines (near-axis outliers) entirely, "
                         "from every panel and the colour scale [0]")
    ap.add_argument("--cbar-ticks", type=int, default=10,
                    help="max intervals for the numeric iota colour-bar ticks (MaxNLocator) [10]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    return ap.parse_args(argv)


def main(cli):
    os.environ["MRX_DTYPE"] = cli.precision
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    from mrx.geometry import build_sequence, geometry_nfp
    from mrx.nullspace import compute_nullspaces, get_nullspace
    from mrx.plotting import render_section
    from mrx.plotstyle import FS, SectionLimits
    from mrx.poincare import (logical_field, require_zeta_parameterisation, seed_from_axis,
                              section_RZ, surface_label, trace_and_classify)

    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)
    nfp = geometry_nfp(cli.geometry)
    seq, _ = build_sequence(cli.geometry, ns, cli.p)

    # The vacuum field: the harmonic 2-form, L2-normalised (as in tutorial 2).
    # --field-npz loads a stored harmonic DOF vector (e.g. a vacuum_convergence
    # rung's fields.npz 'h_dof') so the ~9 min Hodge solve is skipped; --geometry,
    # --ns and --p must then match the run that produced it.
    if cli.field_npz:
        B = jnp.asarray(np.load(cli.field_npz)[cli.field_key])
        assert B.shape == (seq.n(2, True),), (B.shape, seq.n(2, True))
        print(f"[vacuum] loaded {cli.field_key} from {cli.field_npz}", flush=True)
    else:
        compute_nullspaces(seq)
        B = get_nullspace(seq.get_operators(), 2, True)[0]
    B = B / float(seq.l2_norm(B, 2))

    # Trace the field lines once; each plane is a different cut of the same lines.
    field = logical_field(seq, jnp.asarray(B), 2, True)
    info = require_zeta_parameterisation(field, name="vacuum")
    seeds = seed_from_axis(field, cli.seeds, cli.saves, r_edge=cli.r_max, n_rays=cli.rays,
                           steps_per_period=cli.steps)
    res = trace_and_classify(field, seeds, nfp, n_periods=cli.periods,
                             steps_per_period=cli.steps, saves_per_period=cli.saves)
    keep = np.asarray(~(res["escaped"] | ~res["ok"]))
    # Omit the N innermost radial lines ENTIRELY (near-axis outliers): drop them
    # from every per-line array so they never appear -- not marked "lost".
    line = np.ones(keep.shape[0], dtype=bool)
    if cli.drop_inner > 0:
        order = np.argsort(np.asarray(res["seeds"][:, 0]))
        line[order[:cli.drop_inner]] = False
        print(f"[vacuum] omitting {cli.drop_inner} innermost line(s) up to r = "
              f"{float(np.asarray(res['seeds'][:, 0])[order[cli.drop_inner - 1]]):.4f}", flush=True)
    iota_l = np.asarray(res["iota"])[line]
    iota_err_l = np.asarray(res["iota_err"])[line]
    iota_scat_l = np.asarray(res["iota_scatter"])[line]
    seed_r_l = np.asarray(res["seeds"][:, 0])[line]
    keep_l = keep[line]
    shown_l = keep_l & ~np.asarray(res["chaotic"])[line]
    print(f"[vacuum] {ns} p={cli.p} nfp={nfp}: {int((~keep_l).sum())}/{keep_l.size} lost, "
          f"{int((keep_l & np.asarray(res['chaotic'])[line]).sum())} chaotic, drift {res['drift']:.2e}, "
          f"iota in [{float(iota_l[shown_l].min()):.4f}, {float(iota_l[shown_l].max()):.4f}]; "
          f"B^zeta/|B| in [{info['bz_over_b_min']:+.2e}, {info['bz_over_b_max']:+.2e}]", flush=True)

    # The house font hierarchy (label > tick > annotation/legend), rescaled so the
    # axis labels are --label-size at one page wide; the rest stay proportional.
    scale = cli.label_size / FS.label
    label_sz, tick_sz, annot_sz = FS.label * scale, FS.tick * scale, FS.annot * scale

    # ONE iota colour + profile scale across all five planes (never rescaled).
    lo, hi = float(iota_l[shown_l].min()), float(iota_l[shown_l].max())
    for plane in (0.0, 0.125, 0.25, 0.375, 0.5):
        R, Z, aR, aZ, _cR, _cZ, lr, lth = section_RZ(seq, res["ys"], res["axis"], cli.saves, plane)
        R, Z, lr, lth = np.asarray(R)[line], np.asarray(Z)[line], np.asarray(lr)[line], np.asarray(lth)[line]
        a_eff, xlabel = surface_label(R, Z, aR, aZ)
        fig, axes = render_section(
            R, Z, iota_l, iota_err_l, seed_r_l, keep_l,
            pressure=None, title=None, draw_ribbon=False,   # no title, no iota +-band
            axis_RZ=None,                                   # no magnetic-axis '+' marker
            profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp,
            logical=(lr, lth), limits=SectionLimits(iota=(lo, hi)),
            iota_scatter=iota_scat_l, profile_rays=cli.profile_rays,
            legend_fontsize=annot_sz)
        # render_section is @house_style-decorated (tick/label sizes come from the
        # house mplstyle), so scale the fonts AFTER it returns. Author the figure at
        # one page wide so the sizes above are read as-is when included at \linewidth.
        w0, h0 = fig.get_size_inches()
        fig.set_size_inches(cli.page_width, cli.page_width * h0 / w0)
        for a in fig.axes:
            a.tick_params(labelsize=tick_sz)
            a.xaxis.label.set_size(label_sz)
            a.yaxis.label.set_size(label_sz)
            for t in a.texts:                       # Farey labels, in-axes notes
                t.set_fontsize(annot_sz)
            leg = a.get_legend()
            if leg is None:
                continue
            labels = [t.get_text() for t in leg.get_texts()]
            if any("theta" in lab for lab in labels):
                # At one page wide the 3-across "upper center" theta legend runs
                # over the iota curve; stack it in the empty lower-left corner
                # (the curve sits high on the left, the Farey labels on the right).
                handles = getattr(leg, "legend_handles", None) or leg.legendHandles
                leg.remove()
                a.legend(handles, labels, loc="lower left", ncol=1,
                         fontsize=annot_sz, handlelength=1.4, handletextpad=0.4,
                         labelspacing=0.25, borderpad=0.3, borderaxespad=0.4,
                         framealpha=0.85)
            else:
                for txt in leg.get_texts():
                    txt.set_fontsize(annot_sz)
        # Denser, numeric ticks on the iota colour bar (render_section labels it
        # only at the resonant rationals, which are sparse over a narrow range).
        cbar = axes["ax"].collections[0].colorbar
        if cbar is not None:
            ticks = MaxNLocator(nbins=cli.cbar_ticks).tick_values(lo, hi)
            ticks = [float(t) for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]  # keep within the colour range
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.3f}" for t in ticks])
            cbar.ax.tick_params(labelsize=tick_sz)
        stem = os.path.join(cli.out, f"poincare_zeta{plane:g}")
        fig.savefig(stem + ".pdf", dpi=cli.dpi)     # dpi sets the rasterised crossing scatter
        fig.savefig(stem + ".png", dpi=cli.dpi)     # for quick viewing; the PDF is the deliverable
        plt.close(fig)
        print(f"  -> {stem}.pdf", flush=True)


if __name__ == "__main__":
    main(parse_args())
