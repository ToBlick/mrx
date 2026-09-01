"""Poincare sections of a ``scripts/relax.py`` run.

Reads the ``B.h5`` a relaxation wrote (datasets ``B_ic`` and ``B_final``,
run parameters as root attributes), rebuilds the sequence with
:func:`mrx.geometry.build_sequence` from the ``geometry_path`` attribute
(the resolved path of the GVEC export, or the analytic name) and the ``nfp``
override the run used, traces both fields with :mod:`mrx.poincare`, and
renders one section per requested plane.

Usage:
    python -u scripts/poincare_relax.py outputs/run/B.h5 --periods 400 --out outputs/run/poincare

Flags (defaults in brackets):
    state                  path to B.h5 (positional)
    --fields F             comma-separated subset of ic,final [ic,final]
                           or `snapshots`: one frame per stored step (relax.py
                           --save-every) with every axis held fixed, written as
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
    --pressure {weak,strong}  which pressure of the state file to draw [weak]
    --out DIR              output directory [<state dir>/poincare]
    --from-npz             re-render from ``<out>/sections.npz`` without tracing

If the state file carries the pressures ``scripts/relax.py`` writes, the
selected one is evaluated at every crossing and drawn below the axis in the
section and as a profile on the right axis of the iota-profile panel; on a
flux surface it is constant, so the width of each stripe is the diagnostic. ``--pressure weak`` (the default) reads
``pw_ic`` / ``pw_final``, the weak pressure: a 0-form, so its physical value
is the spline evaluation itself (no ``det DF``), and it is zero on the wall by
construction (Dirichlet 0-form space), so no gauge shift is applied.
``--pressure strong`` reads ``p_ic`` / ``p_final``, the Leray multiplier of
the relaxation: a 3-form, evaluated as ``p / det DF``, and defined up to an
additive constant, so the displayed value is ``p - min p``, the minimum taken
over the crossings of the kept lines of that field on every requested plane:
the profile is >= 0 and its lowest surface reads zero. See "Two pressures" in
docs/source/concepts/relaxation.md.

Output: ``poincare_<field>_zeta<plane>.png`` per field and plane, and
``sections.npz`` under ``--out`` with, per field, the crossing coordinates of
every plane plus ``iota``, ``iota_err`` (the fit uncertainty drawn as the
profile's ribbon, see ``trace_and_classify``), ``seed_r``, ``keep``,
``chaotic``, the step drift and ``pressure_kind``, so a section can be re-rendered
(``--from-npz``) without the 5-minute sequence build and trace.
Runtime: ~5 min per field at (8,16,8) p=3 on one H100 (sequence setup
dominates; the trace is ~1 min).
"""
import argparse
import os
import sys
import numpy as np

#: Panel labels. The strong (Leray) multiplier is gauged so that its lowest
#: kept line reads zero; the weak pressure is zero on the wall by construction.
PRESSURE_LABELS = {"strong": r"$p - \min p$", "weak": r"$p_w$"}


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
    ap.add_argument("state")
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
    cli = ap.parse_args()
    os.environ["MRX_DTYPE"] = cli.precision

    import h5py
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import build_sequence, geometry_nfp
    from mrx.geometry import map_jacobian_at
    from mrx.plotting import render_section
    from mrx.poincare import (logical_field, seed_from_axis,
                              section_RZ, surface_label, trace_and_classify,
                              require_zeta_parameterisation)

    with h5py.File(cli.state, "r") as fh:
        attrs = dict(fh.attrs)
        dofs = {k: np.asarray(fh[k], dtype=np.float64) for k in fh.keys()}
    movie = cli.fields.strip() == "snapshots"
    if movie:
        # One frame per stored snapshot (relax.py --save-every), named by step.
        steps = [int(v) for v in dofs["snapshot_steps"]]
        if cli.snapshot_steps:
            wanted = set()
            for rng in cli.snapshot_steps.split(","):
                a, b, c = (int(v) for v in rng.split(":"))
                wanted.update(range(a, b, c))
            keep_steps = [k for k in steps if k in wanted or k == steps[-1]]
        else:
            keep_steps = steps
        cli.fields = ",".join(f"step{k:05d}" for k in keep_steps)
        for i, k in enumerate(steps):
            dofs[f"B_step{k:05d}"] = dofs["B_snapshots"][i]
            dofs[f"pw_step{k:05d}"] = dofs["pw_snapshots"][i]
    geometry = str(attrs["geometry_path"])
    label = os.path.basename(geometry)
    ns = tuple(int(v) for v in attrs["ns"])
    p = int(attrs["p"])
    nfp_override = None if str(attrs["nfp"]) == "" else int(attrs["nfp"])
    nfp = geometry_nfp(geometry, nfp_override)
    out = cli.out or os.path.join(os.path.dirname(os.path.abspath(cli.state)), "poincare")
    os.makedirs(out, exist_ok=True)
    planes = [float(v) for v in cli.planes.split(",")]
    which = [w.strip() for w in cli.fields.split(",")]
    print(f"[state] {cli.state}: {geometry} ns={ns} p={p} nfp={nfp} "
          f"relaxed in {attrs.get('precision')} for {attrs.get('steps')} steps "
          f"({attrs.get('method')}, eta_max={attrs.get('eta_max')}); tracing in {cli.precision}",
          flush=True)

    labels = {"ic": f"initial condition (--ic {attrs.get('ic', '?')})",
              **{n: f"step {int(n[4:])}" for n in cli.fields.split(",") if n.startswith("step")},
              "final": "relaxed field"}
    if cli.from_npz:
        z = np.load(os.path.join(out, "sections.npz"))
        lo = min(float(z[f"{n}_iota"][z[f"{n}_shown"]].min()) for n in which)
        hi = max(float(z[f"{n}_iota"][z[f"{n}_shown"]].max()) for n in which)
        for name in which:
            presses = {plane: z[f"{name}_zeta{plane:g}_pressure"]
                       if f"{name}_zeta{plane:g}_pressure" in z else None for plane in planes}
            p_min = pressure_gauge(str(z["pressure_kind"]), presses, z[f"{name}_keep"])
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
                    pressure_label=PRESSURE_LABELS[str(z["pressure_kind"])], iota_lim=(lo, hi))
                path = os.path.join(out, f"poincare_{name}_zeta{plane:g}.png")
                fig.savefig(path, dpi=200)
                plt.close(fig)
                print(f"  -> {path}", flush=True)
        return

    seq, _ = build_sequence(geometry, ns, p, int(attrs["maxiter"]), nfp=nfp_override)

    def physical_pressure(name, lr, lth, zeta):
        """The selected pressure at logical ``(lr, lth, zeta)``, or None without it.

        Weak: the 0-form's value. Strong: the 3-form's ``p / det DF``.
        """
        key = ("pw_" if cli.pressure == "weak" else "p_") + name
        if key not in dofs:
            return None
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
    # A movie holds every axis fixed across frames: the section window, the
    # split line (the FIRST frame's axis), the profile abscissa and the
    # pressure ordinate, all from the union over frames; iota_lim already is.
    limits = {}
    if movie:
        first = cli.fields.split(",")[0]
        for plane in planes:
            Rs = np.concatenate([np.asarray(all_cuts[n][plane][0])[traced[n][1]].ravel() for n in traced])
            Zs = np.concatenate([np.asarray(all_cuts[n][plane][1])[traced[n][1]].ravel() for n in traced])
            xs = np.concatenate([np.asarray(surface_label(*all_cuts[n][plane][:4])[0])[traced[n][1]].ravel()
                                 for n in traced])
            ps = [100.0 * (all_presses[n][plane] - all_pmin[n])[traced[n][1]] for n in traced
                  if all_presses[n][plane] is not None]
            span = np.ptp(Rs)
            lim = {"RZ": ((Rs.min() - 0.06 * span, Rs.max() + 0.06 * span),
                          (Zs.min() - 0.06 * span, Zs.max() + 0.06 * span)),
                   "z_split": float(np.mean(all_cuts[first][plane][3])),
                   "x": (np.nanmin(xs), np.nanmax(xs))}
            if ps:
                lo_p, hi_p = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
                lim["p"] = (lo_p - 0.05 * (hi_p - lo_p), hi_p + 0.05 * (hi_p - lo_p))
            limits[plane] = lim
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
                iota_lim=(lo, hi), limits=limits.get(plane))
            path = os.path.join(out, (f"frame_zeta{plane:g}_{frame:04d}.png" if movie
                                      else f"poincare_{name}_zeta{plane:g}.png"))
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"  -> {path}", flush=True)
            tag = f"{name}_zeta{plane:g}"
            for key, arr in zip(("R", "Z", "axisR", "axisZ", "logr", "logth", "a_eff"),
                                (R, Z, aR, aZ, lr, lth, a_eff)):
                sections[f"{tag}_{key}"] = np.asarray(arr)
            if press is not None:
                sections[f"{tag}_pressure"] = press
            sections[f"{tag}_xlabel"] = np.array(xlabel)
        for key, arr in (("iota", res["iota"]), ("iota_err", res["iota_err"]), ("seed_r", res["seeds"][:, 0]),
                         ("keep", keep), ("chaotic", res["chaotic"]), ("shown", shown),
                         ("drift", np.array(res["drift"]))):
            sections[f"{name}_{key}"] = np.asarray(arr)
    sections["pressure_kind"] = np.array(cli.pressure)
    np.savez_compressed(os.path.join(out, "sections.npz"), **sections)


if __name__ == "__main__":
    sys.exit(main())
