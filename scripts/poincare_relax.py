"""Poincare sections of a ``scripts/relax.py`` run.

Reads the ``B.h5`` a relaxation wrote (datasets ``B_ic`` and ``B_final``,
run parameters as root attributes), rebuilds the sequence with
:func:`mrx.geometries.build_sequence`, traces both fields with
:mod:`mrx.poincare`, and renders one section per requested plane.

Usage:
    python -u scripts/poincare_relax.py outputs/run/B.h5 --periods 400 --out outputs/run/poincare

Flags (defaults in brackets):
    state                  path to B.h5 (positional)
    --fields F             comma-separated subset of ic,final [ic,final]
    --seeds N              field lines per field [40]
    --periods N            toroidal periods per line [400]
    --steps N              integration steps per period [24]
    --saves N              sections saved per period [8]
    --planes LIST          zeta planes in [0,1) as fractions of a period [0]
    --r-max R              outermost seed radius [0.97]
    --batch-size N         lines integrated per batch [all]
    --precision P          tracing precision float64|float32 [float64]
    --out DIR              output directory [<state dir>/poincare]
    --from-npz             re-render from ``<out>/sections.npz`` without tracing

If the state file carries the Leray pressures ``p_ic`` / ``p_final`` (3-form
DoFs, written by ``scripts/relax.py``), the physical pressure ``p / det DF`` is
evaluated at every crossing and drawn below the axis in the section and as a
profile; on a flux surface it is constant, so the width of each stripe is the
diagnostic.

Output: ``poincare_<field>_zeta<plane>.png`` per field and plane, and
``sections.npz`` under ``--out`` with, per field, the crossing coordinates of
every plane plus ``iota``, ``resid``, ``seed_r``, ``keep``, ``chaotic`` and the
step drift, so a section can be re-rendered (``--from-npz``) without the
5-minute sequence build and trace.
Runtime: ~5 min per field at (8,16,8) p=3 on one H100 (sequence setup
dominates; the trace is ~1 min).
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--fields", default="ic,final")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--saves", type=int, default=8)
    ap.add_argument("--planes", default="0")
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
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
    import numpy as np
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometries import build_sequence, geometry_nfp
    from mrx.geometry import map_jacobian_at
    from mrx.poincare import (logical_field, render_section, seed_from_axis,
                              section_RZ, surface_label, trace_and_classify,
                              require_zeta_parameterisation)

    with h5py.File(cli.state, "r") as fh:
        attrs = dict(fh.attrs)
        dofs = {k: np.asarray(fh[k], dtype=np.float64) for k in fh.keys()}
    geometry = str(attrs["geometry"])
    ns = tuple(int(v) for v in attrs["ns"])
    p = int(attrs["p"])
    nfp = geometry_nfp(geometry)
    out = cli.out or os.path.join(os.path.dirname(os.path.abspath(cli.state)), "poincare")
    os.makedirs(out, exist_ok=True)
    planes = [float(v) for v in cli.planes.split(",")]
    which = [w.strip() for w in cli.fields.split(",")]
    print(f"[state] {cli.state}: {geometry} ns={ns} p={p} nfp={nfp} "
          f"relaxed in {attrs.get('precision')} for {attrs.get('steps')} steps "
          f"({attrs.get('method')}, eta_max={attrs.get('eta_max')}); tracing in {cli.precision}",
          flush=True)

    labels = {"ic": "initial condition (after Leray projection)",
              "final": "relaxed field"}
    if cli.from_npz:
        z = np.load(os.path.join(out, "sections.npz"))
        lo = min(float(z[f"{n}_iota"][z[f"{n}_shown"]].min()) for n in which)
        hi = max(float(z[f"{n}_iota"][z[f"{n}_shown"]].max()) for n in which)
        for name in which:
            for plane in planes:
                tag = f"{name}_zeta{plane:g}"
                fig = render_section(
                    z[f"{tag}_R"], z[f"{tag}_Z"], z[f"{name}_iota"], z[f"{name}_resid"],
                    z[f"{name}_seed_r"], z[f"{name}_keep"],
                    title=f"{geometry} {ns} p={p}  |  {name}  |  $\\zeta = {plane:g}$\n"
                          f"{labels.get(name, name)}, relaxed in {attrs.get('precision')} "
                          f"-- {z[f'{tag}_R'].shape[1]} crossings/line",
                    subtitle=f"nfp = {nfp}   |   h/2 drift {float(z[f'{name}_drift']):.1e}   |   re-rendered from sections.npz",
                    axis_RZ=(z[f"{tag}_axisR"], z[f"{tag}_axisZ"]), path=None,
                    profile_x=z[f"{tag}_a_eff"], profile_xlabel=str(z[f"{tag}_xlabel"]), nfp=nfp,
                    logical=(z[f"{tag}_logr"], z[f"{tag}_logth"]), chaotic=z[f"{name}_chaotic"],
                    pressure=z[f"{tag}_pressure"] if f"{tag}_pressure" in z else None,
                    split_iota_p=f"{tag}_pressure" in z, iota_lim=(lo, hi))
                path = os.path.join(out, f"poincare_{name}_zeta{plane:g}.png")
                fig.savefig(path, dpi=200)
                plt.close(fig)
                print(f"  -> {path}", flush=True)
        return

    seq, _ = build_sequence(geometry, ns, p, int(attrs.get("maxiter", 10_000)))

    def pressure_at(name, lr, lth, plane):
        """Physical pressure p / det DF at the crossings, or None without p."""
        key = "p_" + name
        if key not in dofs:
            return None
        pd = dofs[key]
        e3 = seq.e3_dbc if pd.shape[0] == int(seq.n3_dbc) else seq.e3
        p_h = DiscreteFunction(jnp.asarray(pd), seq.basis_3, e3)
        x = jnp.stack([jnp.asarray(lr).ravel(), jnp.asarray(lth).ravel(),
                       jnp.full(lr.size, plane)], axis=1)
        val = jax.vmap(p_h)(x)[:, 0]
        det = jnp.linalg.det(map_jacobian_at(seq.map, x))
        return np.asarray(val / det).reshape(lr.shape)
    traced = {}
    for name in which:
        B = dofs["B_" + name]
        assert B.shape == (seq.n2_dbc,), (B.shape, seq.n2_dbc)
        field = logical_field(seq, jnp.asarray(B), 2, True)
        info = require_zeta_parameterisation(field, name=name)
        print(f"[zeta] {name}: B^zeta/|B| in [{info['bz_over_b_min']:+.3e}, "
              f"{info['bz_over_b_max']:+.3e}]", flush=True)
        seeds = seed_from_axis(field, cli.seeds, cli.saves, r_edge=cli.r_max,
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
        print(f"[{name}] {res['walltime']:.1f}s, {int((~keep).sum())}/{cli.seeds} lost, "
              f"{int((keep & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}",
              flush=True)
        traced[name] = (res, keep, shown)
    lo = min(float(t[0]["iota"][t[2]].min()) for t in traced.values() if t[2].any())
    hi = max(float(t[0]["iota"][t[2]].max()) for t in traced.values() if t[2].any())
    sections = {}
    for name, (res, keep, shown) in traced.items():
        for plane in planes:
            R, Z, aR, aZ, cR, cZ, lr, lth = section_RZ(
                seq, res["ys"], res["axis"], cli.saves, plane)
            a_eff, xlabel = surface_label("midplane", R, Z, aR, aZ, res["seeds"][:, 0])
            press = pressure_at(name, lr, lth, plane)
            fig = render_section(
                R, Z, res["iota"], res["resid"], res["seeds"][:, 0], keep,
                pressure=press, split_iota_p=press is not None,
                title=f"{geometry} {ns} p={p}  |  {name}  |  $\\zeta = {plane:g}$\n"
                      f"{labels.get(name, name)}, relaxed in {attrs.get('precision')} "
                      f"-- {R.shape[1]} crossings/line",
                subtitle=f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e}   |   "
                         f"traced in {cli.precision}",
                axis_RZ=(aR, aZ), path=None, profile_x=a_eff,
                profile_xlabel=xlabel, nfp=nfp, logical=(lr, lth),
                chaotic=res["chaotic"], iota_lim=(lo, hi))
            path = os.path.join(out, f"poincare_{name}_zeta{plane:g}.png")
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
        for key, arr in (("iota", res["iota"]), ("resid", res["resid"]), ("seed_r", res["seeds"][:, 0]),
                         ("keep", keep), ("chaotic", res["chaotic"]), ("shown", shown),
                         ("drift", np.array(res["drift"]))):
            sections[f"{name}_{key}"] = np.asarray(arr)
    np.savez_compressed(os.path.join(out, "sections.npz"), **sections)


if __name__ == "__main__":
    sys.exit(main())
