r"""Poincare sections of a ``scripts/relax.py`` run.

Reads a run directory: ``relax.json`` for the parameters (geometry path,
mesh, degree, the ``nfp`` override, radial refinement) and
``checkpoints/state_<step>.h5`` for the fields (:func:`mrx.relaxation.read_checkpoint`
layout: the state's ``B_n`` and its strong pressure ``p``), rebuilds the
sequence with :func:`mrx.geometry.build_sequence`, traces the requested
fields with :mod:`mrx.poincare`, archives the crossings of every requested
plane in ``<out>/sections.npz`` and renders one section per field and plane
from that archive (``--from-npz`` re-renders it without the sequence build
and the trace).

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
    --precision P          tracing precision float64|float32 [float64]
    --pressure {weak,strong}  which pressure to draw [weak]
    --out DIR              output directory [<run>/poincare]
    --from-npz             re-render from ``<out>/sections.npz`` without tracing
    --no-pgf               skip the presentation .pgf next to each PNG

The selected pressure is evaluated at every crossing and drawn below the axis
in the section and as a profile on the right axis of the iota-profile panel;
on a flux surface it is constant, so the width of each stripe is the
diagnostic. ``--pressure weak`` (the default) computes the weak pressure of
each field (:func:`mrx.relaxation.weak_pressure`, two solves per field): a
0-form, so its physical value is the spline evaluation itself (no ``det
DF``), and it is zero on the wall by construction (Dirichlet 0-form space),
so no gauge shift is applied. ``--pressure strong`` reads the checkpoint's
``p``, the Leray multiplier of the relaxation: a 3-form, evaluated as ``p /
det DF``, and defined up to an additive constant, so the displayed value is
``p - min p``, the minimum taken over the crossings of the kept lines of that
field on every requested plane. See "Two pressures" in
docs/source/concepts/relaxation.md.

Output: ``poincare_<field>_zeta<plane>.png`` per field and plane and, unless
``--no-pgf``, ``pgf/poincare_<field>_zeta<plane>.pgf`` next to it
(:func:`mrx.plotting.save_figure`: vector LaTeX for lines and text, the
scatter as a high-dpi raster; needs ``xelatex`` on PATH), plus
``sections.npz`` with, per field, the crossing coordinates and pressure of
every plane, ``iota``, ``iota_err``, ``iota_scatter``, ``keep``,
``chaotic`` and the step drift.
"""
import argparse
import glob
import os
import sys
import numpy as np

#: Panel labels. The strong (Leray) multiplier is gauged so that its lowest
#: kept line reads zero; the weak pressure is zero on the wall by construction.
PRESSURE_LABELS = {"strong": r"$p - \min p$", "weak": r"$p$"}


def pressure_gauge(kind, presses, keep):
    """The shift subtracted from the drawn pressure: ``min p`` over the kept
    lines' crossings on every plane for the strong pressure, 0 for the weak
    one."""
    return 0.0 if kind == "weak" else float(min(np.min(pv[keep]) for pv in presses))


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
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    ap.add_argument("--pressure", default="weak", choices=("weak", "strong"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-npz", action="store_true")
    ap.add_argument("--no-pgf", dest="pgf", action="store_false",
                    help="skip the presentation .pgf written next to each PNG "
                         "(needs xelatex on PATH)")
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
    from mrx.geometry import build_sequence, geometry_nfp, map_jacobian_at, parse_r_refine
    from mrx.plotting import PRESSURE_SCALE, render_section, save_figure
    from mrx.plotstyle import SectionLimits
    from mrx.poincare import section_RZ, trace_sections

    run_dir = os.path.abspath(cli.run)
    with open(os.path.join(run_dir, "relax.json")) as fh:
        results = json.load(fh)
    attrs = results["params"]
    ckpts = {int(os.path.basename(f)[6:12]): f
             for f in glob.glob(os.path.join(run_dir, "checkpoints", "state_*.h5"))}
    fields = [w.strip() for w in cli.fields.split(",")]
    labels = {"ic": f"initial condition ({attrs['ic']})", "final": "relaxed field"}
    steps_of = {"ic": min(ckpts), "final": max(ckpts)}
    if "reconnect" in fields:
        # The field before each reconnection: the checkpoint at the record's step.
        ks = []
        for ev in results["reconnect"]:
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
            steps = [k for k in steps if k in wanted or k == steps[-1]]
        fields = [f"step{k:05d}" for k in steps]
        for k in steps:
            steps_of[f"step{k:05d}"] = k
            labels[f"step{k:05d}"] = f"step {k}"
    geometry, ns, p = attrs["geometry_path"], tuple(attrs["ns"]), int(attrs["p"])
    label = os.path.basename(geometry)
    nfp = geometry_nfp(geometry, attrs["nfp"])
    out = cli.out or os.path.join(run_dir, "poincare")
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, "sections.npz")
    planes = [float(v) for v in cli.planes.split(",")]
    print(f"[run] {run_dir}: {geometry} ns={ns} p={p} nfp={nfp} "
          f"relaxed in {attrs['precision']} for {attrs['steps']} steps "
          f"({attrs['scheme']}, auxiliary B field {attrs['auxiliary_B_field']}); "
          f"tracing in {cli.precision}", flush=True)

    if cli.from_npz:
        z = dict(np.load(npz))
        source = "re-rendered from sections.npz"
    else:
        seq, _ = build_sequence(geometry, ns, p, nfp=attrs["nfp"],
                                r_windows=parse_r_refine(attrs["r_refine"]))
        dofs = {}
        for name in fields:
            with h5py.File(ckpts[steps_of[name]], "r") as fh:
                dofs["B_" + name] = np.asarray(fh["B_n"], dtype=np.float64)
                if cli.pressure == "strong":
                    dofs["p_" + name] = np.asarray(fh["p"], dtype=np.float64)
        if cli.pressure == "weak":
            # The weak pressure is a diagnostic of the field, not state: two solves per field.
            from mrx.relaxation import compute_force, weak_pressure
            aux = bool(attrs["auxiliary_B_field"])
            for name in fields:
                _, _, J, X, _ = compute_force(jnp.asarray(dofs["B_" + name]), seq, aux)
                dofs["p_" + name] = np.asarray(weak_pressure(J, X, seq, aux)[0], dtype=np.float64)

        def physical_pressure(name, lr, lth, zeta):
            """The selected pressure at logical ``(lr, lth, zeta)``: the weak
            0-form's value, or the strong 3-form's ``p / det DF``."""
            pd = jnp.asarray(dofs["p_" + name])
            x = jnp.stack([jnp.asarray(lr).ravel(), jnp.asarray(lth).ravel(),
                           jnp.broadcast_to(jnp.asarray(zeta), lr.shape).ravel()], axis=1)
            if cli.pressure == "weak":
                val = jax.vmap(DiscreteFunction(pd, seq.basis_0, seq.E(0, True)))(x)[:, 0]
            else:
                val = jax.vmap(DiscreteFunction(pd, seq.basis_3, seq.E(3, True)))(x)[:, 0]
                val = val / jnp.linalg.det(map_jacobian_at(seq.map, x))
            return np.asarray(val).reshape(lr.shape)

        z = {"pressure_kind": np.array(cli.pressure)}
        for name in fields:
            res, info = trace_sections(
                seq, dofs["B_" + name], nfp, n_seeds=cli.seeds, n_periods=cli.periods,
                n_rays=cli.rays, steps_per_period=cli.steps, saves_per_period=cli.saves,
                r_edge=cli.r_max, name=name)
            keep = ~(res["escaped"] | ~res["ok"])
            shown = keep & ~res["chaotic"]
            span = (f"iota {float(res['iota'][shown].min()):.4f}.."
                    f"{float(res['iota'][shown].max()):.4f}" if shown.any()
                    else "no line converged")
            print(f"[{name}] B^zeta/|B| in [{info['bz_over_b_min']:+.3e}, {info['bz_over_b_max']:+.3e}]; "
                  f"{res['walltime']:.1f}s, {int((~keep).sum())}/{keep.size} lost, "
                  f"{int((keep & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}",
                  flush=True)
            for key in ("iota", "iota_err", "iota_scatter", "chaotic", "drift"):
                z[f"{name}_{key}"] = np.asarray(res[key])
            z[f"{name}_keep"], z[f"{name}_shown"] = keep, shown
            for plane in planes:
                tag = f"{name}_zeta{plane:g}"
                R, Z, aR, aZ, lr, lth = section_RZ(seq, res["ys"], res["axis"], cli.saves, plane)
                for key, arr in zip(("R", "Z", "axisR", "axisZ", "logr", "logth"),
                                    (R, Z, aR, aZ, lr, lth)):
                    z[f"{tag}_{key}"] = np.asarray(arr)
                z[f"{tag}_pressure"] = physical_pressure(name, lr, lth, plane)
        np.savez_compressed(npz, **z)
        source = f"traced in {cli.precision}"

    # ---- render from the archive: one path for a fresh trace and a re-render.
    kind = str(z["pressure_kind"])
    lo = min(float(z[f"{n}_iota"][z[f"{n}_shown"]].min()) for n in fields if z[f"{n}_shown"].any())
    hi = max(float(z[f"{n}_iota"][z[f"{n}_shown"]].max()) for n in fields if z[f"{n}_shown"].any())
    # ONE pressure scale across every rendered field and every plane, for the
    # same reason the iota limits are one: ic, final, the reconnection series
    # and the planes are then comparable at a glance.
    p_min = {n: pressure_gauge(kind, [z[f"{n}_zeta{plane:g}_pressure"] for plane in planes],
                               z[f"{n}_keep"]) for n in fields}
    ps = [PRESSURE_SCALE * (z[f"{n}_zeta{plane:g}_pressure"] - p_min[n])[z[f"{n}_keep"]]
          for n in fields for plane in planes]
    lo_p, hi_p = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
    limits = {plane: {"p": (lo_p - 0.05 * (hi_p - lo_p), hi_p + 0.05 * (hi_p - lo_p))}
              for plane in planes}
    # A movie holds EVERY other axis fixed across frames too: the section
    # window, the split line (the FIRST frame's axis) and the profile abscissa.
    if movie:
        for plane in planes:
            Rs = np.concatenate([z[f"{n}_zeta{plane:g}_R"][z[f"{n}_keep"]].ravel() for n in fields])
            Zs = np.concatenate([z[f"{n}_zeta{plane:g}_Z"][z[f"{n}_keep"]].ravel() for n in fields])
            span = np.ptp(Rs)
            limits[plane].update(
                RZ=((Rs.min() - 0.06 * span, Rs.max() + 0.06 * span),
                    (Zs.min() - 0.06 * span, Zs.max() + 0.06 * span)),
                z_split=float(np.mean(z[f"{fields[0]}_zeta{plane:g}_axisZ"])),
                x=(0.0, 1.0))
    for frame, name in enumerate(fields):
        for plane in planes:
            tag = f"{name}_zeta{plane:g}"
            fig, _ = render_section(
                z[f"{tag}_R"], z[f"{tag}_Z"], z[f"{name}_iota"], z[f"{name}_iota_err"],
                z[f"{name}_keep"],
                title=f"{label} {ns} p={p}  |  {name}  |  $\\zeta = {plane:g}$\n"
                      f"{labels[name]}, relaxed in {attrs['precision']} "
                      f"-- {z[f'{tag}_R'].shape[1]} crossings/line",
                subtitle=f"nfp = {nfp}   |   h/2 drift {float(z[f'{name}_drift']):.1e}   |   {source}",
                axis_RZ=(z[f"{tag}_axisR"], z[f"{tag}_axisZ"]), nfp=nfp,
                logical=(z[f"{tag}_logr"], z[f"{tag}_logth"]),
                pressure=z[f"{tag}_pressure"] - p_min[name], pressure_label=PRESSURE_LABELS[kind],
                limits=SectionLimits(iota=(lo, hi), **limits[plane]),
                iota_scatter=z[f"{name}_iota_scatter"])
            path = os.path.join(out, (f"frame_zeta{plane:g}_{frame:04d}.png" if movie
                                      else f"poincare_{name}_zeta{plane:g}.png"))
            # A movie's frames are for ffmpeg, not slides: no .pgf per frame.
            pgf = cli.pgf and not movie
            save_figure(fig, path, pgf=pgf, dpi=300 if pgf else 200)
            plt.close(fig)
            print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
