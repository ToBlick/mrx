r"""Render the Poincare sections of a trace archive -- the cheap half, no GPU.

Reads the ``trace.npz`` written by ``scripts/poincare_trace.py`` (next to its run
or field file), renders every field and plane it holds -- or ``--fields``, a
subset -- with ONE iota and ONE pressure scale, and writes the pages to
``--out`` [``<archive dir>/poincare``]. Plain matplotlib: it runs on the login
node in seconds per page, so the trace is never repeated for a change to the
figure. Every rendering choice is made here, from the archived trace results:
which lines are drawn, the surface label, the pressure gauge, the rational
ticks, the layout.

    python scripts/poincare_plot.py outputs/run                     # diagnostic look
    python scripts/poincare_plot.py outputs/run/trace.npz --paper   # for the paper

Flags (defaults in brackets):
    archive                the trace.npz, or the directory holding it (positional)
    --fields F             comma-separated subset of the archive's fields [all]
    --out DIR              [<archive dir>/poincare]
    --no-pressure          do not draw the archived pressure (below the axis in
                           the section and the chart, and as the profile's right
                           axis); it is drawn whenever the archive holds one
    --inner-cells C        lines seeded closer to the axis than C radial cells
                           (r < C / n_r) are not drawn: the field is not resolved
                           there (the polar patch) and their iota fit is biased
                           high (QA 32x64x32 p3: +1.6e-3 at r = 0.019, on trend
                           from r ~ 1.5 h). They stay in the archive. [1.5]
    --min-sep F            resonant-rational iota ticks: a rational is labelled
                           only if it is at least this fraction of the iota
                           range from every lower-order label already placed --
                           the spacing decides, not the denominator [0.12]
    --denom-max N          candidate pool for the rational ticks, large enough
                           that --min-sep is the rule that stops them [300]
    --profile-coord C      profile abscissa: logical r on golden-spaced rays
                           [logical], or physical R on the midplane through the axis
    --profile-rays N       golden-angle poloidal rays on the logical profile,
                           marked on both section panels [3]
    --dot-scale F          crossing-marker size relative to the house rule (which
                           sets it from the point count); a third, for a dense
                           section on a page [0.33]
    --paper                publication layout: no title/subtitle, no axis marker,
                           the house font hierarchy at --label-size for a
                           --page-width figure, PDF + PNG at --dpi. Default is the
                           diagnostic look: titled PNG at 200 dpi plus a
                           presentation .pgf (see --no-pgf)
    --label-size F         (--paper) axis-label pt at --page-width; ticks and
                           legends keep the house hierarchy [6]
    --page-width W         (--paper) authored width in inches, 'one page wide' [6.5]
    --dpi N                (--paper) rasterised crossing-scatter resolution [600]
    --no-pgf               skip the .pgf beside each PNG (needs xelatex on PATH)
    --fly                  a movie along zeta: one frame per plane of each field,
                           numbered in plane order (trace with --saves N
                           --planes k/N, k = 0..N-1)
    --window R0,R1,Z0,Z1   pin the section box to this window on every page;
                           give a time movie and the fly through its final
                           state the same one so the two cut together

Pages: ``poincare_<field>_zeta<plane>`` per field and plane --
``poincare_zeta<plane>`` when the archive holds one field -- each section in
the box that fits it, the iota and p colour scales shared across the call.
Movies hold the R/Z window and the profile abscissa fixed across their frames
too, from the union over all fields and planes, for ``ffmpeg -framerate 4 -i
<pattern> -c:v mpeg4 -q:v 2 movie.mp4``: a snapshots archive writes
``frame_zeta<plane>_<i>.png`` (along time, the split line pinned to the first
frame's axis), ``--fly`` writes ``fly_<field>_<k>.png`` (along zeta, the split
following the axis). The ``.pgf`` is the same figure through the pgf
backend (vector LaTeX labels, the scatter as a high-dpi ``-img*.png`` beside it)
under ``pgf/``; the including document needs ``\usepackage[strings]{underscore}``
and ``\providecommand{\mathdefault}[1]{#1}``.
"""
import argparse
import os
import sys
import numpy as np

#: Panel labels. Both read plain $p$: the weak pressure is zero on the wall by
#: construction; the strong (Leray) multiplier is defined up to a constant and
#: is drawn shifted so its lowest kept line reads zero, but a constant offset
#: is not worth a label (Tobias, 2026-09-09: "I do not want p - min p").
PRESSURE_LABELS = {"strong": r"$p$", "weak": r"$p$"}

#: Resolution of the raster layers embedded in the presentation ``.pgf`` (the
#: scatter of ~10^4 crossings). Higher than the PNG's screen dpi: the .pgf goes
#: into slides where the section is enlarged.
PGF_DPI = 300


def save_section(fig, png_path, *, want_pgf):
    """Save the section as a PNG and, when ``want_pgf``, a presentation PGF.

    The PGF keeps every line, axis and label as vector LaTeX -- editable in the
    ``.pgf`` without re-tracing -- while the rasterized scatter is written as a
    high-dpi PNG beside it. Both go under ``pgf/`` next to the PNG pages. It
    needs ``xelatex`` on PATH; without one the PNG is still written and the PGF
    is skipped with a message rather than aborting (the trace is the expensive
    half, and it is not this script's).
    """
    fig.savefig(png_path, dpi=200)
    print(f"  -> {png_path}", flush=True)
    if not want_pgf:
        return
    import matplotlib as mpl
    pgf_dir = os.path.join(os.path.dirname(png_path), "pgf")
    os.makedirs(pgf_dir, exist_ok=True)
    pgf_path = os.path.join(pgf_dir, os.path.splitext(os.path.basename(png_path))[0] + ".pgf")
    try:
        with mpl.rc_context({"pgf.preamble": r"\usepackage[strings]{underscore}\providecommand{\mathdefault}[1]{#1}"}):
            fig.savefig(pgf_path, backend="pgf", dpi=PGF_DPI)
        print(f"  -> {pgf_path}", flush=True)
    except Exception as exc:      # noqa: BLE001 -- the .pgf is an optional artifact
        if os.path.exists(pgf_path):
            os.remove(pgf_path)   # a half-written .pgf is not a usable file
        print(f"  (pgf skipped -- needs xelatex on PATH: "
              f"{type(exc).__name__}: {exc})", flush=True)


def pressure_gauge(kind, presses, keep):
    """The shift subtracted from the drawn pressure: ``min p`` over the kept
    lines' crossings on every plane for the strong pressure, 0 for the weak
    one; None without a pressure."""
    vals = [pv[keep] for pv in presses.values() if pv is not None]
    if not vals:
        return None
    return 0.0 if kind == "weak" else float(min(np.min(v) for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", help="trace.npz, or the directory holding it")
    ap.add_argument("--fields", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-pressure", dest="pressure", action="store_false")
    ap.add_argument("--inner-cells", type=float, default=1.5)
    ap.add_argument("--min-sep", type=float, default=0.12)
    ap.add_argument("--denom-max", type=int, default=300)
    ap.add_argument("--profile-coord", default="logical", choices=("logical", "physical"))
    ap.add_argument("--profile-rays", type=int, default=3)
    ap.add_argument("--dot-scale", type=float, default=0.33)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--label-size", type=float, default=6.0)
    ap.add_argument("--page-width", type=float, default=6.5)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--no-pgf", dest="pgf", action="store_false")
    ap.add_argument("--fly", action="store_true")
    ap.add_argument("--window", default=None,
                    help="Rmin,Rmax,Zmin,Zmax: pin the section box to this window on every page "
                         "(e.g. the same window for a time movie and the fly that follows it)")
    cli = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.plotting import paper_fonts, render_section
    from mrx.plotstyle import SectionLimits
    from mrx.poincare import surface_label

    path = cli.archive if cli.archive.endswith(".npz") else os.path.join(cli.archive, "trace.npz")
    z = np.load(path)
    sec = {k: z[k] for k in z.files}
    out = cli.out or os.path.join(os.path.dirname(os.path.abspath(path)), "poincare")
    os.makedirs(out, exist_ok=True)
    all_fields = [str(f) for f in sec["fields"]]
    which = [w.strip() for w in cli.fields.split(",")] if cli.fields else all_fields
    for n in which:
        assert n in all_fields, (n, all_fields)
    planes = [float(v) for v in sec["planes"]]
    ns, nfp = tuple(int(v) for v in sec["ns"]), int(sec["nfp"])
    source, movie = str(sec["source"]), bool(sec["movie"])
    kind = str(sec["pressure_kind"]) if cli.pressure else "none"
    print(f"[plot] {path}: {source}; fields {which}, planes {planes}, pressure {kind}"
          + (", movie" if movie else ""), flush=True)

    # Lines seeded within --inner-cells radial cells of the axis are archived but
    # not DRAWN: dropped from every per-line array and from both scales, so they
    # never appear -- not marked "lost".
    r_min = cli.inner_cells / ns[0]
    drawn = {n: np.asarray(sec[f"{n}_seed_r"]) >= r_min for n in which}
    per = {n: {k: np.asarray(sec[f"{n}_{k}"])[drawn[n]]
               for k in ("iota", "iota_err", "iota_scatter", "seed_r", "keep", "chaotic", "shown")}
           for n in which}
    for n in which:
        print(f"[{n}] drawing r >= {r_min:.4f} ({cli.inner_cells:g} radial cells): "
              f"{int((~drawn[n]).sum())} inner line(s) not drawn; "
              f"{int((~per[n]['keep']).sum())}/{per[n]['keep'].size} lost, "
              f"{int((per[n]['keep'] & per[n]['chaotic']).sum())} chaotic", flush=True)
    # ONE iota scale and ONE p scale across every field and every plane: ic,
    # final, a reconnection series and the planes are then comparable at a glance.
    lo = min(float(per[n]["iota"][per[n]["shown"]].min()) for n in which if per[n]["shown"].any())
    hi = max(float(per[n]["iota"][per[n]["shown"]].max()) for n in which if per[n]["shown"].any())
    cuts = {(n, pl): tuple(np.asarray(sec[f"{n}_zeta{pl:g}_{k}"])
                           for k in ("R", "Z", "axisR", "axisZ", "logr", "logth"))
            for n in which for pl in planes}
    # The archive holds the raw pressure at every crossing: gauge it here (min
    # over the drawn kept lines on every plane for the strong multiplier, 0 for
    # the weak pressure) and pin one p range.
    presses = {n: {pl: (np.asarray(sec[f"{n}_zeta{pl:g}_pressure"])[drawn[n]]
                        if kind != "none" else None) for pl in planes} for n in which}
    p_min = {n: pressure_gauge(kind, presses[n], per[n]["keep"]) for n in which}
    ps = [100.0 * (presses[n][pl] - p_min[n])[per[n]["keep"]]
          for n in which for pl in planes if presses[n][pl] is not None]
    limits = {}
    if ps:
        lo_p, hi_p = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
        limits = {pl: {"p": (lo_p - 0.05 * (hi_p - lo_p), hi_p + 0.05 * (hi_p - lo_p))}
                  for pl in planes}
    # Pages stand alone: each section gets the box that fits it (equal aspect).
    # A MOVIE holds every axis fixed across its frames instead -- the section
    # window and the profile abscissa from the union over every field AND plane
    # of the call (the iota and p limits already are) -- so the eye can follow
    # a surface from frame to frame: along time at one plane (a snapshots
    # archive), or along zeta for one state (--fly). The split line is pinned
    # to the FIRST field's axis only along time: along zeta the axis moves with
    # the plane, and the split follows it.
    if movie or cli.fly:
        def kept(n, pl, i):
            return cuts[n, pl][i][drawn[n]][per[n]["keep"]]
        Rs = np.concatenate([kept(n, pl, 0).ravel() for n in which for pl in planes])
        Zs = np.concatenate([kept(n, pl, 1).ravel() for n in which for pl in planes])
        span = np.ptp(Rs)
        for pl in planes:
            limits.setdefault(pl, {}).update({
                "RZ": ((Rs.min() - 0.06 * span, Rs.max() + 0.06 * span),
                       (Zs.min() - 0.06 * span, Zs.max() + 0.06 * span))})
            if movie:
                limits[pl]["z_split"] = float(np.mean(cuts[which[0], pl][3]))
        if cli.profile_coord == "physical":
            # The logical profile's abscissa is r in [0, 1] on every frame already;
            # only the physical one (R on the midplane) varies with the plane.
            xs = np.concatenate([
                surface_label(cuts[n, pl][0][drawn[n]], cuts[n, pl][1][drawn[n]],
                              cuts[n, pl][2], cuts[n, pl][3])[0][per[n]["keep"]].ravel()
                for n in which for pl in planes])
            for pl in planes:
                limits[pl]["x"] = (np.nanmin(xs), np.nanmax(xs))
    if cli.window:
        # An explicit box beats the union: two calls (a time movie, then the fly
        # through its final state) cut together only if they are given the SAME
        # window; each call's own union differs.
        r0, r1, z0, z1 = (float(v) for v in cli.window.split(","))
        for pl in planes:
            limits.setdefault(pl, {})["RZ"] = ((r0, r1), (z0, z1))
    for frame, n in enumerate(which):
        for k, pl in enumerate(planes):
            R, Z, aR, aZ, lr, lth = cuts[n, pl]
            R, Z, lr, lth = R[drawn[n]], Z[drawn[n]], lr[drawn[n]], lth[drawn[n]]
            a_eff, xlabel = surface_label(R, Z, aR, aZ)
            press = None if presses[n][pl] is None else presses[n][pl] - p_min[n]
            fig, _ = render_section(
                R, Z, per[n]["iota"], per[n]["iota_err"], per[n]["seed_r"], per[n]["keep"],
                pressure=press, pressure_label=PRESSURE_LABELS.get(kind),
                title=None if cli.paper else
                      f"{source}  |  {n}  |  $\\zeta = {pl:g}$\n"
                      f"{str(sec[f'{n}_label'])} -- {R.shape[1]} crossings/line",
                subtitle=None if cli.paper else
                         f"nfp = {nfp}   |   h/2 drift {float(sec[f'{n}_drift']):.1e}   |   "
                         f"traced in {str(sec['trace_precision'])}",
                axis_RZ=(aR, aZ), axis_marker=not cli.paper, dot_scale=cli.dot_scale,
                profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp, logical=(lr, lth),
                denom_max=cli.denom_max, min_sep=cli.min_sep,
                limits=SectionLimits(iota=(lo, hi), **limits.get(pl, {})),
                iota_scatter=per[n]["iota_scatter"],
                profile_coord=cli.profile_coord, profile_rays=cli.profile_rays)
            infix = "" if len(all_fields) == 1 else f"_{n}"
            stem = os.path.join(out, f"frame_zeta{pl:g}_{frame:04d}" if movie      # along time
                                else f"fly{infix}_{k:04d}" if cli.fly              # along zeta
                                else f"poincare{infix}_zeta{pl:g}")
            if cli.paper:
                paper_fonts(fig, label_size=cli.label_size, page_width=cli.page_width)
                fig.savefig(stem + ".pdf", dpi=cli.dpi)     # dpi sets the rasterised scatter
                fig.savefig(stem + ".png", dpi=cli.dpi)     # for viewing; the PDF is the deliverable
                print(f"  -> {stem}.pdf", flush=True)
            else:
                # A movie's frames are for ffmpeg, not slides: no .pgf per frame.
                save_section(fig, stem + ".png", want_pgf=cli.pgf and not (movie or cli.fly))
            plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
