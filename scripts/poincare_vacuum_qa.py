"""Poincaré sections of the QA vacuum solution -- the harmonic 2-form, no pressure.

The vacuum field of ``LandremanPaul2021_QA`` is the harmonic 2-form of the
Dirichlet de Rham complex (built here as in ``scripts/tutorials/2_qa_vacuum_field.py``:
``compute_nullspaces`` + ``get_nullspace``). This traces its field lines once and
renders the five toroidal planes with ``mrx.plotting.render_section`` -- the same
plotter ``scripts/poincare_relax.py`` drives -- but with ``pressure=None`` (a
vacuum carries no pressure, so the pressure colour split, the below-axis scatter
and the pressure profile are all omitted), no iota +-band ribbon and no title.
One iota colour scale across all five planes. The fonts are scaled so ticks,
labels and the legend read at ``--font-size`` when the figure is one page wide.
Writes ``poincare_zeta*.pdf`` (and a ``.png`` for quick viewing) into ``--out``.

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
    --font-size F        ticks/labels/legend size in pt at --page-width [10]
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
    ap.add_argument("--font-size", type=float, default=10.0,
                    help="all ticks, labels and the legend, in pt, at --page-width [10]")
    ap.add_argument("--page-width", type=float, default=6.5,
                    help="authored figure width in inches ('one page wide'); the fonts "
                         "read at --font-size when the figure is included at this width [6.5]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    return ap.parse_args(argv)


def main(cli):
    os.environ["MRX_DTYPE"] = cli.precision
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from mrx.geometry import build_sequence, geometry_nfp
    from mrx.nullspace import compute_nullspaces, get_nullspace
    from mrx.plotting import render_section
    from mrx.plotstyle import SectionLimits
    from mrx.poincare import (logical_field, require_zeta_parameterisation, seed_from_axis,
                              section_RZ, surface_label, trace_and_classify)

    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)
    nfp = geometry_nfp(cli.geometry)
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    compute_nullspaces(seq)

    # The vacuum field: the harmonic 2-form, L2-normalised (as in tutorial 2).
    B = get_nullspace(seq.get_operators(), 2, True)[0]
    B = B / float(seq.l2_norm(B, 2))

    # Trace the field lines once; each plane is a different cut of the same lines.
    field = logical_field(seq, jnp.asarray(B), 2, True)
    info = require_zeta_parameterisation(field, name="vacuum")
    seeds = seed_from_axis(field, cli.seeds, cli.saves, r_edge=cli.r_max, n_rays=cli.rays,
                           steps_per_period=cli.steps)
    res = trace_and_classify(field, seeds, nfp, n_periods=cli.periods,
                             steps_per_period=cli.steps, saves_per_period=cli.saves)
    keep = ~(res["escaped"] | ~res["ok"])
    shown = keep & ~res["chaotic"]
    print(f"[vacuum] {ns} p={cli.p} nfp={nfp}: {int((~keep).sum())}/{keep.size} lost, "
          f"{int((keep & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, "
          f"iota in [{float(res['iota'][shown].min()):.4f}, {float(res['iota'][shown].max()):.4f}]; "
          f"B^zeta/|B| in [{info['bz_over_b_min']:+.2e}, {info['bz_over_b_max']:+.2e}]", flush=True)

    # ONE iota colour + profile scale across all five planes (never rescaled).
    lo, hi = float(res["iota"][shown].min()), float(res["iota"][shown].max())
    for plane in (0.0, 0.125, 0.25, 0.375, 0.5):
        R, Z, aR, aZ, _cR, _cZ, lr, lth = section_RZ(seq, res["ys"], res["axis"], cli.saves, plane)
        a_eff, xlabel = surface_label(R, Z, aR, aZ)
        fig, _ = render_section(
            R, Z, res["iota"], res["iota_err"], res["seeds"][:, 0], keep,
            pressure=None, title=None, draw_ribbon=False,   # no title, no iota +-band
            axis_RZ=(aR, aZ), profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp,
            logical=(lr, lth), limits=SectionLimits(iota=(lo, hi)),
            iota_scatter=res["iota_scatter"], profile_rays=cli.profile_rays,
            legend_fontsize=cli.font_size)
        # render_section is @house_style-decorated (tick/label sizes come from the
        # house mplstyle), so scale the fonts AFTER it returns. Author the figure at
        # one page wide with every text at --font-size, so it reads at that size when
        # included at \linewidth.
        w0, h0 = fig.get_size_inches()
        fig.set_size_inches(cli.page_width, cli.page_width * h0 / w0)
        for a in fig.axes:
            a.tick_params(labelsize=cli.font_size)
            a.xaxis.label.set_size(cli.font_size)
            a.yaxis.label.set_size(cli.font_size)
            for t in a.texts:                       # Farey labels, in-axes notes
                t.set_fontsize(cli.font_size)
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
                         fontsize=cli.font_size, handlelength=1.4, handletextpad=0.4,
                         labelspacing=0.25, borderpad=0.3, borderaxespad=0.4,
                         framealpha=0.85)
            else:
                for txt in leg.get_texts():
                    txt.set_fontsize(cli.font_size)
        stem = os.path.join(cli.out, f"poincare_zeta{plane:g}")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=200)         # for quick viewing; the PDF is the deliverable
        plt.close(fig)
        print(f"  -> {stem}.pdf", flush=True)


if __name__ == "__main__":
    main(parse_args())
