r"""Poincare sections of a ``scripts/relax.py`` run.

Reads a run directory: ``relax.json`` for the parameters (geometry path,
mesh, degree, the ``nfp`` override, radial refinement) and
``checkpoints/state_<step>.h5`` for the fields (:func:`mrx.relaxation.read_checkpoint`
layout: the state's ``B_n`` and its strong pressure ``p``), rebuilds the
sequence with :func:`mrx.geometry.build_sequence`, traces the requested
fields with :mod:`mrx.poincare`, and renders one section per requested plane.

Usage:
    python -u scripts/poincare_relax.py outputs/run --periods 400 --out outputs/run/poincare

Flags (defaults in brackets):
    run                    the run directory (positional)
    --fields F             comma-separated subset of ic,final [ic,final]:
                           ic is checkpoints/state_000000.h5, final the
                           highest step; `reconnect` expands to one field
                           reconnect<k> per record of relax.json's
                           ``reconnect`` list, the checkpoint at its step
                           (the field before the solve), traced in the same
                           call as ic and final so all of them share one
                           iota and one p colour scale; or `snapshots`: one
                           frame per checkpoint (relax.py --chunk) with every
                           axis held fixed, written as
                           frame_zeta<plane>_<i>.png for ffmpeg
    --snapshot-steps S     subset of the stored steps to render, ranges
                           start:stop:stride separated by commas [all]
    --seeds N              field lines per ray and field [40]
    --rays N               poloidal seed rays, golden-angle spaced [4]
    --periods N            toroidal periods per line [400]
    --steps N              integration steps per period [24]
    --saves N              sections saved per period [8]
    --planes LIST          zeta planes in [0,1) as fractions of a period [0]
    --r-max R              outermost seed radius [0.97]
    --batch-size N         lines integrated per batch [all]
    --precision P          tracing precision float64|float32 [float64]
    --pressure {weak,strong}  which pressure to draw [weak]
    --out DIR              output directory [<run>/poincare]
    --from-npz             re-render from ``<out>/sections.npz`` without tracing

The selected pressure is evaluated at every crossing and drawn below the axis
in the section and as a profile on the right axis of the iota-profile panel;
on a flux surface it is constant, so the width of each stripe is the
diagnostic. ``--pressure weak`` (the default) computes the weak pressure of
each field (:func:`mrx.relaxation.weak_pressure`, two solves per field): a
0-form, so its physical value is the spline evaluation itself (no ``det
DF``), and it is zero on the wall by construction (Dirichlet 0-form space),
so no gauge shift is applied. ``--pressure strong`` reads the checkpoint's
``p``, the Leray multiplier of the relaxation: a 3-form, evaluated as ``p /
det DF``, and defined up to an
additive constant, so the displayed value is ``p - min p``, the minimum taken
over the crossings of the kept lines of that field on every requested plane:
the profile is >= 0 and its lowest surface reads zero. See "Two pressures" in
docs/source/concepts/relaxation.md.

Output: ``poincare_<field>_zeta<plane>.png`` per field and plane -- and, unless
``--no-pgf``, a presentation-ready ``pgf/poincare_<field>_zeta<plane>.pgf``
for each: the same figure through matplotlib's ``pgf`` backend, so every line,
axis and label is vector LaTeX (the labels are editable in the ``.pgf`` without
re-tracing) while the scatter layers are embedded as a high-dpi
``pgf/poincare_<field>_zeta<plane>-img*.png``. The ``.pgf`` needs ``xelatex`` on
PATH (``module load texlive`` is NOT enough on this cluster -- its binaries are
in the ``bin/x86_64-linux`` subdir the module does not add); without it the PNG
is still written and the PGF is skipped with a message. The document that
``\input``s the ``.pgf`` must ``\usepackage[strings]{underscore}`` -- matplotlib
writes plain-text underscores (e.g. a geometry name) raw, not escaped. Plus ``sections.npz``
under ``--out`` with, per field, the crossing coordinates of every plane plus
``iota``, ``iota_err`` (the fit uncertainty drawn as the profile's ribbon, see
``trace_and_classify``), ``seed_r``, ``keep``, ``chaotic``, the step drift and
``pressure_kind``, so a section can be re-rendered (``--from-npz``) without the
5-minute sequence build and trace.
Runtime: ~5 min per field at (8,16,8) p=3 on one H100 (sequence setup
dominates; the trace is ~1 min).
"""
import argparse
import glob
import os
import sys
import numpy as np

#: Panel labels. The strong (Leray) multiplier is gauged so that its lowest
#: kept line reads zero; the weak pressure is zero on the wall by construction.
PRESSURE_LABELS = {"strong": r"$p - \min p$", "weak": r"$p$"}

#: Resolution of the raster layers embedded in the presentation ``.pgf`` (the
#: scatter of ~10^4 crossings). Higher than the PNG's screen dpi: the .pgf goes
#: into slides where the section is enlarged.
PGF_DPI = 300


def save_section(fig, png_path, *, want_pgf):
    """Save the section as a PNG and, when ``want_pgf``, a presentation PGF.

    The PGF is the same figure through matplotlib's ``pgf`` backend: every
    line, axis and label stays vector LaTeX -- so the labels are editable in
    the ``.pgf`` (or its preamble) without re-tracing -- while the rasterized
    layers (the scatter of ~10^4 crossings, ``rasterized=True`` in
    :func:`mrx.plotting.render_section`) are written as a high-dpi PNG beside
    it and pulled in with ``\\includegraphics``. Both go under ``pgf/`` next
    to the PNG pages, so the output directory holds one file per page. It
    needs ``xelatex`` on PATH;
    without one the PNG is still written and the PGF is skipped with a message
    rather than aborting the run (the trace is the expensive half).

    The including document must load ``underscore``: matplotlib's pgf backend
    writes plain-text ``_`` (as in a geometry name like ``wout_li383_1.4m.nc``)
    raw, not escaped, so without it the ``.pgf`` fails to compile with a
    "Missing $ inserted". We inject ``\\usepackage[strings]{underscore}`` into
    the pgf preamble so it is listed in the file's own "required packages"
    header; a document that \\input's the ``.pgf`` still needs that line, and
    ``\\providecommand{\\mathdefault}[1]{#1}`` for any log-axis tick label.
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
    ap.add_argument("run", help="a scripts/relax.py run directory (relax.json + checkpoints/)")
    ap.add_argument("--fields", default="ic,final")
    ap.add_argument("--snapshot-steps", default=None,
                    help="with --fields snapshots: which stored steps to render, as "
                         "comma-separated start:stop:stride ranges, e.g. 0:500:2,500:2501:8; "
                         "default all")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--rays", type=int, default=4)
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--saves", type=int, default=8)
    ap.add_argument("--planes", default="0")
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    ap.add_argument("--pressure", default="weak", choices=("weak", "strong"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-npz", action="store_true")
    ap.add_argument("--no-pgf", dest="pgf", action="store_false",
                    help="skip the presentation .pgf written next to each PNG "
                         "(needs xelatex on PATH)")
    ap.add_argument("--profile-coord", default="logical", choices=("logical", "physical"),
                    help="profile abscissa: logical r on golden-spaced rays [default], "
                         "or physical R on the midplane through the axis")
    ap.add_argument("--profile-rays", type=int, default=3,
                    help="number of golden-angle-spaced poloidal rays for the logical "
                         "profile, marked on both section panels [3]")
    cli = ap.parse_args()
    os.environ["MRX_DTYPE"] = cli.precision

    import h5py
    import json
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import build_sequence, parse_r_refine, geometry_nfp
    from mrx.geometry import map_jacobian_at
    from mrx.plotting import render_section
    from mrx.poincare import (logical_field, seed_from_axis,
                              section_RZ, surface_label, trace_and_classify,
                              require_zeta_parameterisation)

    run_dir = os.path.abspath(cli.run)
    with open(os.path.join(run_dir, "relax.json")) as fh:
        results = json.load(fh)
    attrs = results["params"]
    ckpts = {int(os.path.basename(f)[6:12]): f
             for f in glob.glob(os.path.join(run_dir, "checkpoints", "state_*.h5"))}
    fields = [w.strip() for w in cli.fields.split(",")]
    labels = {"ic": f"initial condition ({attrs.get('ic', '?')})", "final": "relaxed field"}
    steps_of = {"ic": min(ckpts), "final": max(ckpts)}
    if "reconnect" in fields:
        # The field before each reconnection: the checkpoint at the record's step.
        ks = []
        for ev in results.get("reconnect", []):
            k = int(ev["k"])
            ks.append(k)
            steps_of[f"reconnect{k}"] = int(ev["it"])
            labels[f"reconnect{k}"] = f"before reconnection {k} (step {int(ev['it'])})"
        fields = [n for w in fields for n in ([f"reconnect{k}" for k in ks] if w == "reconnect" else [w])]
    movie = fields == ["snapshots"]
    if movie:
        # One frame per checkpoint (relax.py --chunk), named by step.
        steps = sorted(ckpts)
        if cli.snapshot_steps:
            wanted = set()
            for rng in cli.snapshot_steps.split(","):
                a, b, c = (int(v) for v in rng.split(":"))
                wanted.update(range(a, b, c))
            keep_steps = [k for k in steps if k in wanted or k == steps[-1]]
        else:
            keep_steps = steps
        fields = [f"step{k:05d}" for k in keep_steps]
        for k in keep_steps:
            steps_of[f"step{k:05d}"] = k
            labels[f"step{k:05d}"] = f"step {k}"
    dofs = {}
    for name in fields:
        with h5py.File(ckpts[steps_of[name]], "r") as fh:
            dofs["B_" + name] = np.asarray(fh["B_n"], dtype=np.float64)
            dofs["p_" + name] = np.asarray(fh["p"], dtype=np.float64)
    geometry = str(attrs["geometry_path"])
    label = os.path.basename(geometry)
    ns = tuple(int(v) for v in attrs["ns"])
    p = int(attrs["p"])
    nfp_override = None if attrs.get("nfp") is None else int(attrs["nfp"])
    nfp = geometry_nfp(geometry, nfp_override)
    out = cli.out or os.path.join(run_dir, "poincare")
    os.makedirs(out, exist_ok=True)
    planes = [float(v) for v in cli.planes.split(",")]
    which = fields
    print(f"[run] {run_dir}: {geometry} ns={ns} p={p} nfp={nfp} "
          f"relaxed in {attrs.get('precision')} for {attrs.get('steps')} steps "
          f"({attrs.get('scheme')}, auxiliary B field {attrs.get('auxiliary_B_field')}); "
          f"tracing in {cli.precision}",
          flush=True)

    if cli.from_npz:
        z = np.load(os.path.join(out, "sections.npz"))
        lo = min(float(z[f"{n}_iota"][z[f"{n}_shown"]].min()) for n in which)
        hi = max(float(z[f"{n}_iota"][z[f"{n}_shown"]].max()) for n in which)
        # Per-field pressure gauge, then the global p range: pin the pressure
        # ordinate across fields and planes exactly like iota_lim above, so a
        # re-render is comparable frame to frame.
        presses_by, pmin_by, all_p = {}, {}, []
        for name in which:
            presses_by[name] = {plane: z[f"{name}_zeta{plane:g}_pressure"]
                                if f"{name}_zeta{plane:g}_pressure" in z else None
                                for plane in planes}
            pmin_by[name] = pressure_gauge(str(z["pressure_kind"]), presses_by[name],
                                           z[f"{name}_keep"])
            for plane in planes:
                pv = presses_by[name][plane]
                if pv is not None:
                    all_p.append(100.0 * (pv - pmin_by[name])[z[f"{name}_keep"]])
        p_lim = None
        if all_p:
            lo_p, hi_p = min(float(np.nanmin(v)) for v in all_p), max(float(np.nanmax(v)) for v in all_p)
            p_lim = (lo_p - 0.05 * (hi_p - lo_p), hi_p + 0.05 * (hi_p - lo_p))
        for name in which:
            presses = presses_by[name]
            p_min = pmin_by[name]
            for plane in planes:
                tag = f"{name}_zeta{plane:g}"
                # The label is a rendering choice, not a trace result: recompute
                # it so a re-render picks up the current definition.
                a_eff, xlabel = surface_label(z[f"{tag}_R"], z[f"{tag}_Z"],
                                              z[f"{tag}_axisR"], z[f"{tag}_axisZ"])
                fig, _ = render_section(
                    z[f"{tag}_R"], z[f"{tag}_Z"], z[f"{name}_iota"], z[f"{name}_iota_err"],
                    z[f"{name}_seed_r"], z[f"{name}_keep"],
                    title=f"{label} {ns} p={p}  |  {name}  |  $\\zeta = {plane:g}$\n"
                          f"{labels.get(name, name)}, relaxed in {attrs.get('precision')} "
                          f"-- {z[f'{tag}_R'].shape[1]} crossings/line",
                    subtitle=f"nfp = {nfp}   |   h/2 drift {float(z[f'{name}_drift']):.1e}   |   re-rendered from sections.npz",
                    axis_RZ=(z[f"{tag}_axisR"], z[f"{tag}_axisZ"]),
                    profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp,
                    logical=(z[f"{tag}_logr"], z[f"{tag}_logth"]),
                    pressure=None if presses[plane] is None else presses[plane] - p_min,
                    pressure_label=PRESSURE_LABELS[str(z["pressure_kind"])], iota_lim=(lo, hi),
                    limits=None if p_lim is None else {"p": p_lim},
                    iota_scatter=z[f"{name}_iota_scatter"] if f"{name}_iota_scatter" in z else None,
                    profile_coord=cli.profile_coord, profile_rays=cli.profile_rays)
                path = os.path.join(out, f"poincare_{name}_zeta{plane:g}.png")
                save_section(fig, path, want_pgf=cli.pgf)
                plt.close(fig)
        return

    seq, _ = build_sequence(geometry, ns, p, nfp=nfp_override,
                            r_windows=parse_r_refine(str(attrs.get("r_refine", ""))))
    if cli.pressure == "weak":
        # The weak pressure is a diagnostic of the field, not state: two solves per field.
        from mrx.relaxation import compute_force, weak_pressure
        aux = bool(attrs.get("auxiliary_B_field", False))
        for name in fields:
            _, _, J, X, _ = compute_force(jnp.asarray(dofs["B_" + name]), seq, aux)
            dofs["pw_" + name] = np.asarray(weak_pressure(J, X, seq, aux)[0], dtype=np.float64)

    def physical_pressure(name, lr, lth, zeta):
        """The selected pressure at logical ``(lr, lth, zeta)``.

        Weak: the 0-form's value. Strong: the 3-form's ``p / det DF``.
        """
        key = ("pw_" if cli.pressure == "weak" else "p_") + name
        pd = jnp.asarray(dofs[key])
        x = jnp.stack([jnp.asarray(lr).ravel(), jnp.asarray(lth).ravel(),
                       jnp.broadcast_to(jnp.asarray(zeta), lr.shape).ravel()], axis=1)
        if cli.pressure == "weak":
            val = jax.vmap(DiscreteFunction(pd, seq.basis_0, seq.E(0, True)))(x)[:, 0]
        else:
            e3 = seq.E(3, True) if pd.shape[0] == int(seq.n(3, True)) else seq.E(3)
            val = jax.vmap(DiscreteFunction(pd, seq.basis_3, e3))(x)[:, 0]
            val = val / jnp.linalg.det(map_jacobian_at(seq.map, x))
        return np.asarray(val).reshape(lr.shape)

    traced = {}
    for name in which:
        B = dofs["B_" + name]
        assert B.shape == (seq.n(2, True),), (B.shape, seq.n(2, True))
        field = logical_field(seq, jnp.asarray(B), 2, True)
        info = require_zeta_parameterisation(field, name=name)
        print(f"[zeta] {name}: B^zeta/|B| in [{info['bz_over_b_min']:+.3e}, "
              f"{info['bz_over_b_max']:+.3e}]", flush=True)
        seeds = seed_from_axis(field, cli.seeds, cli.saves, r_edge=cli.r_max, n_rays=cli.rays,
                               steps_per_period=cli.steps)
        res = trace_and_classify(field, seeds, nfp, n_periods=cli.periods,
                                 steps_per_period=cli.steps,
                                 saves_per_period=cli.saves,
                                 batch_size=cli.batch_size)
        keep = ~(res["escaped"] | ~res["ok"])
        shown = keep & ~res["chaotic"]
        span = (f"iota {float(res['iota'][shown].min()):.4f}.."
                f"{float(res['iota'][shown].max()):.4f}" if shown.any()
                else "no line converged")
        print(f"[{name}] {res['walltime']:.1f}s, {int((~keep).sum())}/{keep.size} lost, "
              f"{int((keep & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}",
              flush=True)
        traced[name] = (res, keep, shown)
    lo = min(float(t[0]["iota"][t[2]].min()) for t in traced.values() if t[2].any())
    hi = max(float(t[0]["iota"][t[2]].max()) for t in traced.values() if t[2].any())
    sections = {}
    all_cuts, all_presses, all_pmin = {}, {}, {}
    for name, (res, keep, shown) in traced.items():
        all_cuts[name] = {plane: section_RZ(seq, res["ys"], res["axis"], cli.saves, plane)
                          for plane in planes}
        all_presses[name] = {plane: physical_pressure(name, all_cuts[name][plane][6],
                                                      all_cuts[name][plane][7], plane)
                             for plane in planes}
        all_pmin[name] = pressure_gauge(cli.pressure, all_presses[name], keep)
    # ONE pressure scale across every rendered field and every plane, for the
    # same reason iota_lim is one: ic, final, the reconnection series and the
    # planes are then comparable at a glance.
    limits = {}
    ps = [100.0 * (all_presses[n][plane] - all_pmin[n])[traced[n][1]]
          for n in traced for plane in planes if all_presses[n][plane] is not None]
    if ps:
        lo_p, hi_p = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
        limits = {plane: {"p": (lo_p - 0.05 * (hi_p - lo_p), hi_p + 0.05 * (hi_p - lo_p))}
                  for plane in planes}
    # A movie holds EVERY other axis fixed across frames too: the section
    # window, the split line (the FIRST frame's axis) and the profile abscissa,
    # all from the union over frames; iota_lim already is.
    if movie:
        first = which[0]
        for plane in planes:
            Rs = np.concatenate([np.asarray(all_cuts[n][plane][0])[traced[n][1]].ravel() for n in traced])
            Zs = np.concatenate([np.asarray(all_cuts[n][plane][1])[traced[n][1]].ravel() for n in traced])
            xs = np.concatenate([np.asarray(surface_label(*all_cuts[n][plane][:4])[0])[traced[n][1]].ravel()
                                 for n in traced])
            span = np.ptp(Rs)
            limits.setdefault(plane, {}).update({
                "RZ": ((Rs.min() - 0.06 * span, Rs.max() + 0.06 * span),
                       (Zs.min() - 0.06 * span, Zs.max() + 0.06 * span)),
                "z_split": float(np.mean(all_cuts[first][plane][3])),
                "x": (np.nanmin(xs), np.nanmax(xs))})
    for frame, (name, (res, keep, shown)) in enumerate(traced.items()):
        cuts, presses, p_min = all_cuts[name], all_presses[name], all_pmin[name]
        for plane in planes:
            R, Z, aR, aZ, cR, cZ, lr, lth = cuts[plane]
            a_eff, xlabel = surface_label(R, Z, aR, aZ)
            press = None if presses[plane] is None else presses[plane] - p_min
            fig, _ = render_section(
                R, Z, res["iota"], res["iota_err"], res["seeds"][:, 0], keep,
                pressure=press, pressure_label=PRESSURE_LABELS[cli.pressure],
                title=f"{label} {ns} p={p}  |  {name}  |  $\\zeta = {plane:g}$\n"
                      f"{labels.get(name, name)}, relaxed in {attrs.get('precision')} "
                      f"-- {R.shape[1]} crossings/line",
                subtitle=f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e}   |   "
                         f"traced in {cli.precision}",
                axis_RZ=(aR, aZ), profile_x=a_eff,
                profile_xlabel=xlabel, nfp=nfp, logical=(lr, lth),
                iota_lim=(lo, hi), limits=limits.get(plane),
                iota_scatter=res["iota_scatter"],
                profile_coord=cli.profile_coord, profile_rays=cli.profile_rays)
            path = os.path.join(out, (f"frame_zeta{plane:g}_{frame:04d}.png" if movie
                                      else f"poincare_{name}_zeta{plane:g}.png"))
            # A movie's frames are for ffmpeg, not slides: no .pgf per frame.
            save_section(fig, path, want_pgf=cli.pgf and not movie)
            plt.close(fig)
            tag = f"{name}_zeta{plane:g}"
            for key, arr in zip(("R", "Z", "axisR", "axisZ", "logr", "logth", "a_eff"),
                                (R, Z, aR, aZ, lr, lth, a_eff)):
                sections[f"{tag}_{key}"] = np.asarray(arr)
            if press is not None:
                sections[f"{tag}_pressure"] = press
            sections[f"{tag}_xlabel"] = np.array(xlabel)
        for key, arr in (("iota", res["iota"]), ("iota_err", res["iota_err"]),
                         ("iota_scatter", res["iota_scatter"]), ("seed_r", res["seeds"][:, 0]),
                         ("keep", keep), ("chaotic", res["chaotic"]), ("shown", shown),
                         ("drift", np.array(res["drift"]))):
            sections[f"{name}_{key}"] = np.asarray(arr)
    sections["pressure_kind"] = np.array(cli.pressure)
    np.savez_compressed(os.path.join(out, "sections.npz"), **sections)


if __name__ == "__main__":
    sys.exit(main())
