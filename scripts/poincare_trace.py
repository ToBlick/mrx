r"""Trace field lines and archive their Poincare sections -- the expensive half.

Sources (exactly one):
    --run DIR          a scripts/relax.py run (relax.json + checkpoints/state_<step>.h5);
                       --fields picks the states, and the selected pressure
                       (--pressure) is evaluated at every crossing
    --field-npz PATH   a stored k=2 Dirichlet DOF vector (e.g. a vacuum_convergence
                       rung's fields.npz 'h_dof'); --geometry/--ns/--p must be the
                       mesh it was computed on; no pressure

The archive ``trace.npz`` is written NEXT TO the source -- ``<run>/trace.npz`` or
``<dir of --field-npz>/trace.npz`` -- because the trace is a result of that run
and lives with it. ``scripts/poincare_plot.py`` renders it: plain matplotlib, no
GPU, on the login node, so a change to the figure never repeats the trace.
Every state of one run is traced in ONE call so the plotter can put them on
one iota and one pressure scale.

    python -u scripts/poincare_trace.py --run outputs/run --fields ic,final
    python -u scripts/poincare_trace.py --field-npz outputs/qa_vacuum_highres/rung_32x64x32_p3/fields.npz \
        --geometry data/wout_LandremanPaul2021_QA_highres.nc --ns 32,64,32 --p 3

Flags (defaults in brackets):
    --fields F             comma-separated subset of ic,final [ic,final]: ic is
                           checkpoints/state_000000.h5, final the highest step;
                           `reconnect` expands to one field reconnect<k> per
                           record of relax.json's ``reconnect`` list (the
                           checkpoint at its step, the field before the solve);
                           `snapshots`: one field per checkpoint (relax.py
                           --chunk), rendered by the plotter as movie frames
    --snapshot-steps S     with snapshots: subset of the stored steps, ranges
                           start:stop:stride separated by commas [all]
    --pressure {weak,strong}  which pressure to evaluate at the crossings [weak]:
                           weak is mrx.relaxation.weak_pressure of each field (a
                           0-form, two solves per field), strong the checkpoint's
                           Leray multiplier ``p`` (a 3-form, ``p / det DF``,
                           defined up to a constant -- the plotter gauges it)
    --field-key K          array name in --field-npz [h_dof]
    --geometry PATH        the mesh of --field-npz: VMEC wout .nc or GVEC .dat
    --ns N_R,N_T,N_Z       ... its resolution
    --p P                  ... its spline degree
    --seeds N              field lines per ray and field [40]
    --rays N               poloidal seed rays, golden-angle spaced; every line
                           has its OWN radius (one radial ladder of rays*seeds
                           is dealt round-robin to the rays) [4]
    --periods N            toroidal periods per line [400]
    --steps N              integration steps per period [24]
    --saves N              sections saved per period [8]
    --planes LIST          zeta planes in [0,1) as fractions of a period, each a
                           multiple of 1/--saves (a plane is one of the saved
                           crossings per period; a fly-along-zeta movie wants
                           --saves 64 --planes 0,0.015625,...)
                           [0,0.125,0.25,0.375,0.5 -- half a period; the other
                           half is stellarator-symmetric]
    --r-max R              outermost seed radius [0.97]
    --batch-size N         lines integrated per batch [all]
    --precision P          tracing precision float64|float32 [float64]
    --out PATH             the archive path [next to the source]

Archive (numpy .npz): ``fields`` (names, in order), ``planes``, ``ns``, ``p``,
``nfp``, ``source`` (one line naming the run or field), ``pressure_kind``
(weak|strong|none), ``trace_precision``, ``movie``; per field ``<f>_label``,
``<f>_iota``, ``<f>_iota_err``, ``<f>_iota_scatter`` (the window std, the
profile ribbon), ``<f>_seed_r``, ``<f>_keep``, ``<f>_chaotic``, ``<f>_shown``,
``<f>_drift``; per field and plane ``<f>_zeta<plane>_{R,Z,axisR,axisZ,logr,logth}``
and, with a pressure, ``<f>_zeta<plane>_pressure``: the RAW value at every
crossing (weak: the 0-form's value; strong: ``p / det DF``). Trace RESULTS
only -- every rendering choice is the plotter's.
Runtime: sequence build 1-3 min, then ~0.5-1 min per traced field at (16,32,32)
on one H100 (the weak pressure adds two solves per field).
"""
import argparse
import glob
import os
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="a scripts/relax.py run directory")
    ap.add_argument("--fields", default="ic,final")
    ap.add_argument("--snapshot-steps", default=None)
    ap.add_argument("--pressure", default="weak", choices=("weak", "strong"))
    ap.add_argument("--field-npz", default=None, help="a stored k=2 DOF vector (.npz)")
    ap.add_argument("--field-key", default="h_dof")
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--ns", default=None)
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--rays", type=int, default=4)
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--saves", type=int, default=8)
    ap.add_argument("--planes", default="0,0.125,0.25,0.375,0.5")
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    ap.add_argument("--out", default=None, help="archive path [<source dir>/trace.npz]")
    cli = ap.parse_args()
    if (cli.run is None) == (cli.field_npz is None):
        ap.error("exactly one of --run and --field-npz")
    if cli.field_npz and not (cli.geometry and cli.ns and cli.p):
        ap.error("--field-npz needs --geometry, --ns and --p")
    os.environ["MRX_DTYPE"] = cli.precision

    import h5py
    import json
    import jax
    import jax.numpy as jnp
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import build_sequence, geometry_nfp, map_jacobian_at, parse_r_refine
    from mrx.poincare import (logical_field, require_zeta_parameterisation, seed_from_axis,
                              section_RZ, trace_and_classify)

    planes = [float(v) for v in cli.planes.split(",")]
    dofs, labels = {}, {}
    if cli.run:
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
        for name in fields:
            with h5py.File(ckpts[steps_of[name]], "r") as fh:
                dofs["B_" + name] = np.asarray(fh["B_n"], dtype=np.float64)
                dofs["p_" + name] = np.asarray(fh["p"], dtype=np.float64)
        geometry = str(attrs["geometry_path"])
        ns = tuple(int(v) for v in attrs["ns"])
        p = int(attrs["p"])
        nfp_override = None if attrs.get("nfp") is None else int(attrs["nfp"])
        r_refine = str(attrs.get("r_refine", ""))
        aux = bool(attrs.get("auxiliary_B_field", False))
        pressure_kind = cli.pressure
        source = (f"{os.path.basename(geometry)} {ns} p={p}, relaxed in {attrs.get('precision')} "
                  f"for {attrs.get('steps')} steps")
        archive = cli.out or os.path.join(run_dir, "trace.npz")
    else:
        geometry, p = cli.geometry, cli.p
        ns = tuple(int(v) for v in cli.ns.split(","))
        nfp_override, r_refine, aux, movie = None, "", False, False
        pressure_kind = "none"
        name = cli.field_key
        fields = [name]
        dofs["B_" + name] = np.asarray(np.load(cli.field_npz)[cli.field_key], dtype=np.float64)
        labels[name] = f"{cli.field_key} of {os.path.basename(cli.field_npz)}"
        source = f"{os.path.basename(geometry)} {ns} p={p}, {cli.field_key} of {cli.field_npz}"
        archive = cli.out or os.path.join(os.path.dirname(os.path.abspath(cli.field_npz)), "trace.npz")
    nfp = geometry_nfp(geometry, nfp_override)
    print(f"[trace] {source}: nfp={nfp}, fields {fields}, planes {planes}, "
          f"pressure {pressure_kind}, tracing in {cli.precision}", flush=True)

    seq, _ = build_sequence(geometry, ns, p, nfp=nfp_override, r_windows=parse_r_refine(r_refine))
    if pressure_kind == "weak":
        # The weak pressure is a diagnostic of the field, not state: two solves per field.
        from mrx.relaxation import compute_force, weak_pressure
        for name in fields:
            _, _, J, X, _ = compute_force(jnp.asarray(dofs["B_" + name]), seq, aux)
            dofs["pw_" + name] = np.asarray(weak_pressure(J, X, seq, aux)[0], dtype=np.float64)

    def physical_pressure(name, lr, lth, zeta):
        """The selected pressure at logical ``(lr, lth, zeta)``.

        Weak: the 0-form's value. Strong: the 3-form's ``p / det DF``.
        """
        key = ("pw_" if pressure_kind == "weak" else "p_") + name
        pd = jnp.asarray(dofs[key])
        x = jnp.stack([jnp.asarray(lr).ravel(), jnp.asarray(lth).ravel(),
                       jnp.broadcast_to(jnp.asarray(zeta), lr.shape).ravel()], axis=1)
        if pressure_kind == "weak":
            val = jax.vmap(DiscreteFunction(pd, seq.basis_0, seq.E(0, True)))(x)[:, 0]
        else:
            e3 = seq.E(3, True) if pd.shape[0] == int(seq.n(3, True)) else seq.E(3)
            val = jax.vmap(DiscreteFunction(pd, seq.basis_3, e3))(x)[:, 0]
            val = val / jnp.linalg.det(map_jacobian_at(seq.map, x))
        return np.asarray(val).reshape(lr.shape)

    sections = {"fields": np.array(fields), "planes": np.array(planes), "ns": np.array(ns),
                "p": p, "nfp": nfp, "source": np.array(source),
                "pressure_kind": np.array(pressure_kind),
                "trace_precision": np.array(cli.precision), "movie": movie}
    for name in fields:
        B = dofs["B_" + name]
        assert B.shape == (seq.n(2, True),), (B.shape, seq.n(2, True))
        field = logical_field(seq, jnp.asarray(B), 2, True)
        info = require_zeta_parameterisation(field, name=name)
        print(f"[zeta] {name}: B^zeta/|B| in [{info['bz_over_b_min']:+.3e}, "
              f"{info['bz_over_b_max']:+.3e}]", flush=True)
        seeds = seed_from_axis(field, cli.seeds, cli.saves, r_edge=cli.r_max, n_rays=cli.rays,
                               steps_per_period=cli.steps)
        res = trace_and_classify(field, seeds, nfp, n_periods=cli.periods,
                                 steps_per_period=cli.steps, saves_per_period=cli.saves,
                                 batch_size=cli.batch_size)
        keep = ~(res["escaped"] | ~res["ok"])
        shown = keep & ~res["chaotic"]
        span = (f"iota {float(res['iota'][shown].min()):.4f}.."
                f"{float(res['iota'][shown].max()):.4f}" if shown.any()
                else "no line converged")
        print(f"[{name}] {res['walltime']:.1f}s, {int((~keep).sum())}/{keep.size} lost, "
              f"{int((keep & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}",
              flush=True)
        sections[f"{name}_label"] = np.array(labels[name])
        for plane in planes:
            R, Z, aR, aZ, _cR, _cZ, lr, lth = section_RZ(seq, res["ys"], res["axis"], cli.saves, plane)
            tag = f"{name}_zeta{plane:g}"
            for key, arr in zip(("R", "Z", "axisR", "axisZ", "logr", "logth"),
                                (R, Z, aR, aZ, lr, lth)):
                sections[f"{tag}_{key}"] = np.asarray(arr)
            if pressure_kind != "none":
                sections[f"{tag}_pressure"] = physical_pressure(name, lr, lth, plane)
        for key, arr in (("iota", res["iota"]), ("iota_err", res["iota_err"]),
                         ("iota_scatter", res["iota_scatter"]), ("seed_r", res["seeds"][:, 0]),
                         ("keep", keep), ("chaotic", res["chaotic"]), ("shown", shown),
                         ("drift", np.array(res["drift"]))):
            sections[f"{name}_{key}"] = np.asarray(arr)
    os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
    np.savez_compressed(archive, **sections)
    print(f"  -> {archive}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
