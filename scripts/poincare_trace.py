r"""Trace field lines and archive their Poincare sections -- the expensive half.

Reads a scripts/relax.py run (relax.json + checkpoints/state_<step>.h5), traces
the chosen states with ``mrx.poincare.poincare`` and evaluates the selected
pressure at every crossing. The archive ``trace.npz`` is written NEXT TO the
run -- the trace is a result of that run and lives with it.
``scripts/poincare_plot.py`` renders it: plain matplotlib, no GPU, on the login
node, so a change to the figure never repeats the trace. Every state of one
run is traced in ONE call so the plotter can put them on one iota and one
pressure scale.

    python -u scripts/poincare_trace.py --run outputs/run --fields ic,final

Flags (defaults in brackets):
    --run DIR              the run directory
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
    --geometry PATH        overrides the run's recorded geometry path (a run
                           relaxed in a since-deleted worktree)
    --lines N              field lines per field, each at its own radius from
                           the magnetic axis to the edge, at a random poloidal
                           angle [160]
    --periods N            toroidal periods per line [400]
    --planes N|LIST        N zeta planes equispaced over what the map's
                           symmetry leaves distinct (half a period for a
                           stellarator-symmetric map, the whole period
                           otherwise), or the planes themselves as fractions
                           of a period; the steps per period follow from them
                           (every plane a step endpoint, at least 24; a
                           fly-along-zeta movie wants --planes 64 on a
                           field-period-symmetric run) [5]
    --seed N               the random seed of the poloidal angles [0]
    --precision P          tracing precision float64|float32 [float64]
    --out PATH             the archive path [<run>/trace.npz]

Archive (numpy .npz): ``fields`` (names, in order), ``planes``, ``ns``, ``p``,
``nfp``, ``symmetry``, ``steps`` (per period), ``source`` (one line naming the run),
``pressure_kind`` (weak|strong), ``trace_precision``, ``movie``; per field
``<f>_label``, ``<f>_iota``, ``<f>_iota_err``, ``<f>_iota_scatter`` (the window
std, the profile ribbon), ``<f>_seed_r``, ``<f>_keep``, ``<f>_chaotic``,
``<f>_shown``, ``<f>_drift``; per field and plane
``<f>_zeta<plane>_{R,Z,axisR,axisZ,logr,logth}`` and ``<f>_zeta<plane>_pressure``:
the RAW value at every crossing (weak: the 0-form's value; strong:
``p / det DF``). Trace RESULTS only -- every rendering choice is the plotter's.
A movie archive is rewritten every 25 fields, so a time-out keeps the frames
traced so far.
Runtime: sequence build 1-3 min, then ~40 s per traced field at (16,32,32) on
one H100 (the weak pressure adds two solves per field). The integrator is
compiled once per mesh and schedule (the field's coefficients are an input),
so a movie pays the compile on its first frame only.
"""
import argparse
import glob
import os
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a scripts/relax.py run directory")
    ap.add_argument("--fields", default="ic,final")
    ap.add_argument("--snapshot-steps", default=None)
    ap.add_argument("--pressure", default="weak", choices=("weak", "strong"))
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--lines", type=int, default=160)
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--planes", default="5", help="a count, or a comma-separated list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="float64", choices=("float64", "float32"))
    ap.add_argument("--out", default=None, help="archive path [<run>/trace.npz]")
    cli = ap.parse_args()
    os.environ["MRX_DTYPE"] = cli.precision

    import h5py
    import json
    import time
    import jax
    import jax.numpy as jnp
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import build_sequence, map_jacobian_at
    from mrx.poincare import planes_for, poincare, steps_for

    planes = int(cli.planes) if cli.planes.isdigit() else [float(v) for v in cli.planes.split(",")]
    run_dir = os.path.abspath(cli.run)
    with open(os.path.join(run_dir, "relax.json")) as fh:
        results = json.load(fh)
    attrs = results["params"]
    ckpts = {int(os.path.basename(f)[6:12]): f
             for f in glob.glob(os.path.join(run_dir, "checkpoints", "state_[0-9]*.h5"))}
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
    geometry = cli.geometry or str(attrs["geometry_path"])
    ns = tuple(int(v) for v in attrs["ns"])
    p = int(attrs["p"])
    nfp_override = None if attrs.get("nfp") is None else int(attrs["nfp"])
    symmetry = attrs.get("symmetry", "stellarator")
    knots = attrs.get("knots")
    aux = bool(attrs.get("auxiliary_B_field", False))
    source = (f"{os.path.basename(geometry)} {ns} p={p}, relaxed in {attrs.get('precision')} "
              f"for {attrs.get('steps')} steps")
    archive = cli.out or os.path.join(run_dir, "trace.npz")
    print(f"[trace] {source}: fields {fields}, pressure {cli.pressure}, tracing in {cli.precision}",
          flush=True)

    seq, _ = build_sequence(geometry, ns, p, nfp=nfp_override, knots=knots, symmetry=symmetry)
    planes = planes_for(seq, planes)
    print(f"[trace] nfp={seq.nfp}, symmetry {seq.symmetry}, planes {[f'{v:g}' for v in planes]} "
          f"({steps_for(planes)} steps per period)", flush=True)
    if cli.pressure == "weak":
        # The weak pressure is a diagnostic of the field, not state: two solves per field.
        from mrx.relaxation import compute_force, weak_pressure
        for name in fields:
            _, _, J, X, _ = compute_force(jnp.asarray(dofs["B_" + name]), seq, aux)
            dofs["pw_" + name] = np.asarray(weak_pressure(J, X, seq, aux)[0], dtype=np.float64)

    @jax.jit
    def weak_at(pd, x):
        return jax.vmap(DiscreteFunction(pd, seq.basis_0, seq.E(0, True)))(x)[:, 0]

    @jax.jit
    def strong_at(pd, x):
        e3 = seq.E(3, True) if pd.shape[0] == int(seq.n(3, True)) else seq.E(3)
        val = jax.vmap(DiscreteFunction(pd, seq.basis_3, e3))(x)[:, 0]
        return val / jnp.linalg.det(map_jacobian_at(seq.map, x))

    def physical_pressure(name, lr, lth, zeta):
        """The selected pressure at logical ``(lr, lth, zeta)``.

        Weak: the 0-form's value. Strong: the 3-form's ``p / det DF``.
        """
        key = ("pw_" if cli.pressure == "weak" else "p_") + name
        pd = jnp.asarray(dofs[key])
        x = jnp.stack([jnp.asarray(lr).ravel(), jnp.asarray(lth).ravel(),
                       jnp.broadcast_to(jnp.asarray(zeta), lr.shape).ravel()], axis=1)
        val = weak_at(pd, x) if cli.pressure == "weak" else strong_at(pd, x)
        return np.asarray(val).reshape(lr.shape)

    sections = {"fields": np.array(fields), "planes": np.array(planes), "ns": np.array(ns),
                "p": p, "nfp": seq.nfp, "symmetry": np.array(seq.symmetry),
                "steps": steps_for(planes), "source": np.array(source),
                "pressure_kind": np.array(cli.pressure),
                "trace_precision": np.array(cli.precision), "movie": movie}
    for i, name in enumerate(fields):
        t_field = time.perf_counter()
        B = dofs["B_" + name]
        assert B.shape == (seq.n(2, True),), (B.shape, seq.n(2, True))
        res = poincare(seq, B, lines=cli.lines, periods=cli.periods, planes=planes,
                       seed=cli.seed, name=name)
        sections[f"{name}_label"] = np.array(labels[name])
        for key in ("iota", "iota_err", "iota_scatter", "seed_r", "keep", "chaotic", "shown", "drift"):
            sections[f"{name}_{key}"] = np.asarray(res[key])
        for plane, sec in res["sections"].items():
            tag = f"{name}_zeta{plane:g}"
            for key, arr in sec.items():
                sections[f"{tag}_{key}"] = arr
            sections[f"{tag}_pressure"] = physical_pressure(name, sec["logr"], sec["logth"], plane)
        shown = res["shown"]
        span = (f"iota {float(res['iota'][shown].min()):.4f}..{float(res['iota'][shown].max()):.4f}"
                if shown.any() else "no line converged")
        print(f"[{name}] B^zeta/|B| in [{res['bz_over_b'][0]:+.3e}, {res['bz_over_b'][1]:+.3e}]; "
              f"trace {res['walltime']:.1f}s, field {time.perf_counter() - t_field:.1f}s, "
              f"{int((~res['keep']).sum())}/{res['keep'].size} lost, "
              f"{int((res['keep'] & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}",
              flush=True)
        if movie and (i + 1) % 25 == 0 and i + 1 < len(fields):
            os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
            np.savez_compressed(archive, **dict(sections, fields=np.array(fields[:i + 1])))
            print(f"  -> {archive} (partial, {i + 1} of {len(fields)} fields)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
    np.savez_compressed(archive, **sections)
    print(f"  -> {archive}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
