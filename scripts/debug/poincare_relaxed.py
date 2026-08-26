"""Poincaré sections of a RELAXED field, straight from a relaxation state file.

``poincare_vacuum.py`` traces harmonic nullspace fields, which it computes.
This traces the field a relaxation *produced*, which it reads: the state file
written by ``relax_from_nfs.py`` carries the 2-form DOFs ``B_dof_final`` and
``B_dof_initial``, the map DOFs ``R_dof``/``Z_dof``, and the run's own config in
a root attribute.  Everything needed to rebuild the discrete field is therefore
in one file, and nothing here solves anything.

Both fields are traced by default, and that pair is the point.  ``initial`` is
the interpolated input field after the Leray projection; ``final`` is what the
relaxation made of it.  Rendered on shared axes they answer the question the
run was asked -- did the surfaces survive, did islands open, did the transform
move -- and a single section of ``final`` alone cannot.

Four gates run before any line is traced, because each one fails as a plausible
picture rather than as an error:

1. **The map round-trips.**  The state file names the point cloud it was fitted
   to (``nfs_file``); when that file is present, ``F(rho, theta, zeta)`` is
   compared against the sampled ``(R, Z)``.  A map rebuilt with the wrong
   ``nfp``, degree or zeta flip still plots -- as a differently-shaped device.
2. **The Jacobian keeps one sign.**  Same reason the relaxation asserts it.
3. **The DOF vector fits the space.**  ``len(B_dof)`` against ``seq.n2_dbc``:
   a mismatch means the FEM resolution in the config is not the one the field
   was computed on, and a silently reshaped vector is noise with the right norm.
4. **The field is still discretely divergence-free.**  ``D2 B`` sits at the
   Leray projection's own solve tolerance, and the relaxation preserves that
   exactly because every update is a curl.  It is the one gate that tests
   which space the DOFs actually live in rather than just how many there are:
   a different radial grading or pole extraction can have the same dimension
   and scores O(1) here.

Then :func:`mrx.poincare.require_zeta_parameterisation` gates the tracer's
change of variables, and tracing is :mod:`mrx.poincare` throughout.

    python scripts/debug/poincare_relaxed.py data/w7x_fmm002_relaxed_100.h5 \
        --n-planes 4 --periods 200 --seeds 48
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import jax


import matplotlib  # noqa: E402

matplotlib.use("Agg")

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.differential_forms import DiscreteFunction  # noqa: E402
from mrx.io_nfs_map import evaluate_map_rz_residual_stats  # noqa: E402
from mrx.mappings import stellarator_map  # noqa: E402
from mrx.poincare import (  # noqa: E402
    CHAOS_TOL, SURFACE_LABELS, logical_field, render_section,
    require_zeta_parameterisation, section_RZ, seed_from_axis, seed_line,
    surface_label, trace_and_classify,
)

#: The two states every relaxation run stores, and what each one IS -- the key
#: names the figure, the value describes it on the second title line.
FIELDS = {"initial": "the input field, interpolated and Leray-projected",
          "final": "after relaxation"}

#: Relative ``|D2 B|`` above which the field is not the divergence-free 2-form
#: it is supposed to be.  This gate separates two regimes six orders apart, so
#: where it sits between them does not matter: the Leray projection is an
#: iterative solve and leaves its own tolerance behind, while a DOF vector read
#: into the wrong space -- a different radial grading, a different pole
#: extraction, the same dimension by coincidence -- scores O(1).  It is not an
#: audit of the projection's accuracy and must not be tightened into one.
DIV_TOL = 1e-6


def read_state(path):
    """Every array in the state file, plus the run's config.

    The config is a YAML string in a root attribute rather than a sidecar
    directory, which is what makes the file self-contained -- and what lets a
    section be produced from a file someone copied out of its run directory,
    which is how this one arrived.
    """
    with h5py.File(path, "r") as f:
        data = {k: np.asarray(f[k]) for k in f.keys()}
        if "config" not in f.attrs:
            raise KeyError(
                f"{path} has no 'config' root attribute: the FEM resolution "
                "and nfp cannot be recovered from the DOF vectors alone")
        cfg = yaml.safe_load(f.attrs["config"])
    return data, cfg


def _ns_ps(section, cfg):
    """``(ns, ps)`` for a config section, with the degree clamp the fits use.

    ``ps_x <= ns_x - 1`` is imposed by the map fit and by the relaxation's own
    space construction.  Reapplying it here rather than trusting the config is
    the difference between rebuilding the space that was used and the space
    that was requested.
    """
    s = cfg[section]
    ns = (s["ns_r"], s["ns_theta"], s["ns_zeta"])
    ps = tuple(min(p, n - 1)
               for p, n in zip((s["ps_r"], s["ps_theta"], s["ps_zeta"]), ns))
    return ns, ps


def build_map(cfg, R_dof, Z_dof):
    """The stellarator map, from the stored 0-form DOFs.

    Refitting it from the point cloud would be the expensive half of this
    script and would also be a *different* map whenever the fit changes.  The
    DOFs are the map; this only rebuilds the basis they live in.
    """
    ns, ps = _ns_ps("map", cfg)
    seq = DeRhamSequence(ns, ps, cfg["map"]["quad_order"],
                         ("clamped", "periodic", "periodic"), polar=False)
    R_h = DiscreteFunction(jnp.asarray(R_dof), seq.basis_0, seq.e0)
    Z_h = DiscreteFunction(jnp.asarray(Z_dof), seq.basis_0, seq.e0)
    return stellarator_map(R_h, Z_h, nfp=cfg["nfp"],
                           flip_zeta=cfg["map"]["flip_zeta"]), ns, ps


def check_map(cfg, map_func, root):
    """Round-trip the rebuilt map against the point cloud it was fitted to.

    Returns ``None`` when the source file is not reachable from ``root`` -- a
    state file is portable and its source cloud need not travel with it -- and
    says so, rather than reporting a check that did not run as a pass.
    """
    nfs = os.path.join(root, cfg["nfs_file"])
    if not os.path.exists(nfs):
        print(f"[map] source cloud {nfs} not present: round-trip NOT checked",
              flush=True)
        return None
    with h5py.File(nfs, "r") as f:
        pts, R, Z = np.asarray(f["eval_points"]), np.asarray(f["R"]), np.asarray(f["Z"])
    stats = evaluate_map_rz_residual_stats(
        map_func, pts, R, Z,
        exclude_axis_tol=cfg["interpolation"]["exclude_axis_tol"])
    stats = {k: float(v) for k, v in stats.items()}
    print(f"[map] round-trip vs {os.path.basename(nfs)}: "
          f"max |dR| {stats['max_R']:.3e} m, max |dZ| {stats['max_Z']:.3e} m, "
          f"rms {stats['rms_R']:.3e}/{stats['rms_Z']:.3e} over "
          f"{int(stats['n_points'])} points", flush=True)
    return stats


def build_sequence(cfg, map_func):
    """The polar sequence the relaxation ran on, with the geometry installed."""
    ns, ps = _ns_ps("fem", cfg)
    seq = DeRhamSequence(ns, ps, cfg["fem"]["quad_order"],
                         ("clamped", "periodic", "periodic"), polar=True)
    seq.set_map(map_func)
    jac = seq.jacobian_j
    if float(jnp.min(jac)) <= 0.0:
        raise RuntimeError(
            f"map Jacobian is not positive: min {float(jnp.min(jac)):.3e}, "
            f"max {float(jnp.max(jac)):.3e}. The rebuilt map is orientation-"
            "reversed relative to the one the field was computed on, and every "
            "contravariant component below carries that sign")
    print(f"[geom] ns={ns} ps={ps} n2_dbc={seq.n2_dbc}  "
          f"Jacobian {float(jnp.min(jac)):.3e} .. {float(jnp.max(jac)):.3e}",
          flush=True)
    return seq, ns, ps


def divergence_check(seq, B_dof, name):
    """``|D2 B| / |D2|`` relative to ``|B|`` -- is this still a 2-form here?

    The weak divergence, not the strong one: ``M3^-1`` would need a solve and
    would not sharpen the answer, since the question is whether the DOFs pair
    with this basis at all, and the answer separates by fifteen orders.  The
    operator scale comes from a random vector so that the ratio is
    dimensionless and does not move with the resolution.
    """
    d = seq.apply_derivative_matrix(jnp.asarray(B_dof), 2)
    w = jax.random.normal(jax.random.PRNGKey(3), (seq.n2_dbc,))
    scale = jnp.linalg.norm(seq.apply_derivative_matrix(w, 2)) / jnp.linalg.norm(w)
    rel = float(jnp.linalg.norm(d) / (scale * jnp.linalg.norm(jnp.asarray(B_dof))))
    print(f"[div] {name}: |D2 B| / (|D2| |B|) = {rel:.3e}", flush=True)
    if rel > DIV_TOL:
        raise RuntimeError(
            f"{name}: relative weak divergence {rel:.3e} exceeds {DIV_TOL:g}. "
            "The Leray projection makes this zero and the relaxation preserves "
            "it, so the DOF vector is not being read into the space it was "
            "computed in -- check fem.ns/ps and the polar/dirichlet flags")
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", help="relaxation state HDF5 (B_dof_*, R_dof, Z_dof)")
    ap.add_argument("--fields", default="initial,final",
                    help="comma separated subset of " + ",".join(FIELDS))
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--r-min", type=float, default=0.03)
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--steps", type=int, default=24,
                    help="prescribed steps per field period")
    ap.add_argument("--saves", type=int, default=8,
                    help="samples kept per period; must divide --steps")
    ap.add_argument("--planes", default=None,
                    help="logical zeta of each section, comma separated. "
                         "Mutually exclusive with --n-planes")
    ap.add_argument("--n-planes", type=int, default=None,
                    help="number of EVENLY SPACED sections over one field "
                         "period: zeta = k/N for k in 0..N-1")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--drift-periods", type=int, default=64)
    ap.add_argument("--profile-x", default="midplane", choices=SURFACE_LABELS,
                    help="surface label on the iota profile")
    ap.add_argument("--seed-from", default="axis", choices=("coord", "axis"),
                    help="'axis' walks out from the MAGNETIC axis, 'coord' "
                         "from the logical r=0. A relaxed finite-beta state "
                         "moves its axis away from r=0, which leaves a hole in "
                         "the middle of a 'coord' section, so 'axis' is the "
                         "default here and is not in the vacuum script")
    ap.add_argument("--formats", default="png",
                    help="comma separated figure formats. More than one is "
                         "cheap -- the trace is the expensive part and the "
                         "markers are rasterised, so a pdf stays small")
    ap.add_argument("--root", default=".",
                    help="directory the config's nfs_file path is relative to")
    ap.add_argument("--out", default="outputs/poincare_relaxed")
    cli = ap.parse_args()

    if cli.planes is not None and cli.n_planes is not None:
        ap.error("--planes and --n-planes are mutually exclusive: pass the "
                 "positions or the count, not both")
    if cli.n_planes is not None:
        if cli.n_planes < 1:
            ap.error(f"--n-planes must be >= 1 (got {cli.n_planes})")
        planes = [k / cli.n_planes for k in range(cli.n_planes)]
    else:
        planes = [float(v) for v in (cli.planes or "0").split(",")]

    formats = [f.strip().lstrip(".") for f in cli.formats.split(",")]
    which = [w.strip() for w in cli.fields.split(",")]
    unknown = [w for w in which if w not in FIELDS]
    if unknown:
        ap.error(f"unknown field(s) {unknown}: choose from {sorted(FIELDS)}")

    os.makedirs(cli.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(cli.state))[0]

    data, cfg = read_state(cli.state)
    nfp = int(cfg["nfp"])
    print(f"[state] {cli.state}  run={cfg['run_name']}  nfp={nfp}  "
          f"{cfg['relaxation']['num_iters_outer']}x"
          f"{cfg['relaxation']['num_iters_inner']} iterations", flush=True)
    if "force_trace" in data:
        ft = data["force_trace"]
        print(f"[state] force norm {float(ft[0]):.4e} -> {float(ft[-1]):.4e}"
              + (f",  energy {float(data['energy_trace'][0]):.6e} -> "
                 f"{float(data['energy_trace'][-1]):.6e}"
                 if "energy_trace" in data else ""), flush=True)

    map_func = jax.jit(build_map(cfg, data["R_dof"], data["Z_dof"])[0])
    map_stats = check_map(cfg, map_func, cli.root)
    seq, ns, ps = build_sequence(cfg, map_func)
    seq.evaluate_1d()
    seq.assemble_derivative_matrix(2)

    summary = {"state": os.path.abspath(cli.state), "run_name": cfg["run_name"],
               "nfp": nfp, "ns": list(ns), "ps": list(ps), "planes": planes,
               "periods": cli.periods, "steps": cli.steps, "saves": cli.saves,
               "seeds": cli.seeds, "map_roundtrip": map_stats, "fields": {}}

    # Shared colour limits across the states: the whole point of drawing
    # `initial` beside `final` is that the two iota scales are the same one.
    # Rendering happens after every field is traced for exactly that reason.
    traced = {}
    for name in which:
        B_dof = data[f"B_dof_{name}"]
        if B_dof.shape != (seq.n2_dbc,):
            raise RuntimeError(
                f"B_dof_{name} has shape {B_dof.shape}, but the sequence this "
                f"config describes has n2_dbc={seq.n2_dbc}. The FEM resolution "
                "in the stored config is not the one the field was computed on")
        div = divergence_check(seq, B_dof, name)

        field = logical_field(seq, B_dof, 2, True)
        info = require_zeta_parameterisation(field, name=name)
        # The worst LOCATION is printed, not just the worst value: a dip near
        # r = 0 sits under the seeds that define the axis, and a dip at the edge
        # does not, so the same number means different things at the two ends.
        wx = info["worst_x"]
        print(f"[zeta] {name}: B^zeta/|B| in "
              f"[{info['bz_over_b_min']:+.3e}, {info['bz_over_b_max']:+.3e}]  "
              f"(closest approach to zero {info['bz_over_b_absmin']:.3e} at "
              f"logical r,theta,zeta = {wx[0]:.3f},{wx[1]:.3f},{wx[2]:.3f}; "
              f"tol {info['tol']:g})", flush=True)

        if cli.seed_from == "axis":
            seeds = seed_from_axis(field, cli.seeds, cli.saves,
                                   r_edge=cli.r_max, steps_per_period=cli.steps)
        else:
            seeds = seed_line(cli.seeds, r_min=cli.r_min, r_max=cli.r_max)

        res = trace_and_classify(
            field, seeds, nfp, n_periods=cli.periods,
            steps_per_period=cli.steps, saves_per_period=cli.saves,
            batch_size=cli.batch_size, drift_periods=cli.drift_periods)
        keep = ~(res["escaped"] | ~res["ok"])
        shown = keep & ~res["chaotic"]
        span = (f"iota {float(res['iota'][shown].min()):.4f}.."
                f"{float(res['iota'][shown].max()):.4f}" if shown.any()
                else "iota: no line converged")
        print(f"[{name}] {res['walltime']:.1f}s trace, "
              f"{int((~keep).sum())}/{cli.seeds} lost, "
              f"{int((keep & res['chaotic']).sum())} chaotic, "
              f"step drift {res['drift']:.2e}, {span}", flush=True)
        traced[name] = (res, keep, shown, div, info)

    if not any(t[2].any() for t in traced.values()):
        raise RuntimeError(
            "no line in any field has a converged rotational transform. "
            "iota_convergence compares the two halves of the trace against "
            f"CHAOS_TOL={CHAOS_TOL:g}, a threshold calibrated on long traces; "
            f"at --periods {cli.periods} each half is only {cli.periods // 2} "
            "periods and the estimate has not settled, which classifies clean "
            "surfaces as chaotic. Raise --periods before reading this as a "
            "chaotic field.")
    lo = min(float(t[0]["iota"][t[2]].min()) for t in traced.values() if t[2].any())
    hi = max(float(t[0]["iota"][t[2]].max()) for t in traced.values() if t[2].any())

    for name, (res, keep, shown, div, info) in traced.items():
        sections, offsets = {}, {}
        for plane in planes:
            R, Z, aR, aZ, cR, cZ, lr, lth = section_RZ(
                seq, res["ys"], res["axis"], cli.saves, plane)
            a_eff, xlabel = surface_label(cli.profile_x, R, Z, aR, aZ,
                                          res["seeds"][:, 0])
            offset = float(np.hypot(aR.mean() - cR, aZ.mean() - cZ))
            base = os.path.join(cli.out, f"poincare_{stem}_{name}_zeta{plane:g}")
            # Rendered once and saved per format: render_section closes the
            # figure when it is given a path, so the extra formats have to come
            # off the returned figure rather than from a second call that would
            # redo the whole scatter.
            fig = render_section(
                R, Z, res["iota"], res["resid"], res["seeds"][:, 0], keep,
                # The state's long description goes on the SECOND line. On the
                # first it pushed the plane out past the panel, and a section
                # labelled with the wrong zeta is worse than one not labelled.
                title=f"{cfg['run_name']}  |  {name}  |  "
                      f"$\\zeta = {plane:g}$\n"
                      f"{FIELDS[name]}  --  {R.shape[1]} crossings/line",
                # The divergence and the B^zeta fraction are gates, not
                # findings: they either raised or they did not, and they are in
                # the log and the summary. Spending subtitle width on them
                # pushed the line past the panel.
                subtitle=f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e} "
                         f"({res['drift_lines']} regular lines)   |   "
                         f"axis offset {offset:.2e} m",
                axis_RZ=(aR, aZ), path=None, profile_x=a_eff,
                profile_xlabel=xlabel, nfp=nfp, logical=(lr, lth),
                chaotic=res["chaotic"], iota_lim=(lo, hi))
            for fmt in formats:
                fig.savefig(f"{base}.{fmt}", dpi=200)
            plt.close(fig)
            offsets[f"zeta{plane:g}"] = offset
            print(f"        -> {base}.{{{','.join(formats)}}}  "
                  f"(axis offset {offset:.3e} m)", flush=True)
            for key, arr in zip(("R", "Z", "axisR", "axisZ", "logr", "logth"),
                                (R, Z, aR, aZ, lr, lth)):
                sections[f"{key}_zeta{plane:g}"] = arr
            sections[f"a_mid_zeta{plane:g}"] = a_eff
            sections[f"coordaxis_zeta{plane:g}"] = np.array([cR, cZ])

        np.savez_compressed(
            os.path.join(cli.out, f"trace_{stem}_{name}.npz"),
            ys=res["ys"], iota=res["iota"], resid=res["resid"],
            seeds=res["seeds"], escaped=res["escaped"], ok=res["ok"],
            axis=res["axis"], chaotic=res["chaotic"], nfp=nfp,
            saves_per_period=cli.saves, label=FIELDS[name], **sections)
        summary["fields"][name] = {
            "walltime_s": res["walltime"], "step_drift": res["drift"],
            "step_drift_lines": res["drift_lines"],
            "lost": int((~keep).sum()),
            "chaotic": int((keep & res["chaotic"]).sum()),
            "iota_min": float(res["iota"][shown].min()) if shown.any() else None,
            "iota_max": float(res["iota"][shown].max()) if shown.any() else None,
            "rel_weak_div": div, "axis_offset_m": offsets,
            "zeta_component": {k: float(v) for k, v in info.items()
                               if isinstance(v, (int, float))},
        }

    with open(os.path.join(cli.out, f"summary_{stem}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {cli.out}", flush=True)


if __name__ == "__main__":
    main()
