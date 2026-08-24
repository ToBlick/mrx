"""Poincaré sections of the harmonic (vacuum) field on the production geometries.

The field comes from the nullspace computers, in both of the forms that carry
it.  For a solid torus the harmonic space is one-dimensional at either end of
the sequence:

* ``k = 2`` with essential BCs -- the Neumann harmonic 2-form, ``n . B = 0``;
* ``k = 1`` with natural BCs   -- the absolute harmonic 1-form, ``n _| A = 0``.

Both are ``d``- and ``delta``-closed and boundary-tangent, so up to sign and
normalisation they are the *same physical field*, computed by two different
solve chains.  The script reports the angle between them before plotting
anything: that number is a free correctness check on both nullspace routes, and
the two Poincaré plots should be indistinguishable.

Tracing is :mod:`mrx.poincare` -- toroidal-angle parameterisation, prescribed
step schedule, Cartesian cross-section chart.  Colour is the rotational
transform measured about the magnetic axis.

    python scripts/debug/poincare_vacuum.py --geometry w7x --periods 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.poincare import (  # noqa: E402
    escaped_mask, logical_field, render_section, rotational_transform,
    seed_line, step_convergence, to_RZ, trace,
)
from verify_block_jacobi import build_sequence  # noqa: E402

#: Field periods spanned by logical zeta in [0, 1].  The stellarator maps model
#: ONE field period (``phi = 2 pi zeta / nfp``); the axisymmetric maps model the
#: whole torus, so their nfp is 1 by this definition.  Iota is reported per
#: toroidal turn, which is where the factor enters.
NFP = {
    "toroid": 1, "cylinder": 1, "rot-ellipse": 3, "w7x": 5,
    "quasr9983": 2, "quasr44970": 3, "w7x-gvec": 5, "hegna": 3,
}

#: (k, dirichlet, label) for the two harmonic fields.
FIELDS = {"k2": (2, True, "k=2, essential BC"),
          "k1": (1, False, "k=1, natural BC")}


def field_agreement(seq, ops, n=512):
    """Max angle between the two harmonic fields, over random logical points.

    Both are boundary-tangent harmonic fields on a one-dimensional harmonic
    space, so the angle is zero up to discretisation and solve tolerance.
    """
    f2 = logical_field(seq, ops.null_2_dbc[0], 2, True)
    f1 = logical_field(seq, ops.null_1[0], 1, False)
    key = jax.random.PRNGKey(7)
    x = jax.random.uniform(key, (n, 3)).at[:, 0].multiply(0.95).at[:, 0].add(0.02)
    v2 = jax.vmap(f2)(x)
    v1 = jax.vmap(f1)(x)
    v2 = v2 / jnp.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / jnp.linalg.norm(v1, axis=1, keepdims=True)
    cos = jnp.abs(jnp.sum(v2 * v1, axis=1))
    return float(jnp.max(jnp.arccos(jnp.clip(cos, -1.0, 1.0))))


def bench(field, seeds, cli):
    """Time the prescribed schedule against the adaptive one it replaces.

    The third arm is the point: chunking an adaptive batch does not isolate a
    pathological seed, it only bounds how many healthy seeds each one holds up.
    A prescribed schedule has no such coupling, so its cost per seed is flat.
    """
    arms = [("prescribed, vmap", dict(adaptive=False, batch_size=None)),
            ("adaptive, vmap", dict(adaptive=True, batch_size=None)),
            ("adaptive, chunk 8", dict(adaptive=True, batch_size=8))]
    out = {}
    for name, kw in arms:
        t0 = time.perf_counter()
        ys, _ = trace(field, seeds, cli.bench_periods, cli.steps, cli.saves,
                      **kw)
        jnp.asarray(ys).block_until_ready()
        out[name] = time.perf_counter() - t0
        print(f"[bench] {name:20s} {out[name]:8.2f}s", flush=True)
    return out


def run_field(seq, dof, k, dirichlet, nfp, cli):
    seeds = seed_line(cli.seeds, r_min=cli.r_min, r_max=cli.r_max)
    field = logical_field(seq, dof, k, dirichlet)

    t0 = time.perf_counter()
    ys, ok = trace(field, seeds, cli.periods, cli.steps, cli.saves,
                   batch_size=cli.batch_size)
    ys = jnp.asarray(ys).block_until_ready()
    walltime = time.perf_counter() - t0

    escaped = escaped_mask(ys)
    iota, resid = rotational_transform(ys, cli.saves, nfp)

    drift = step_convergence(field, seeds[:: max(1, cli.seeds // 8)],
                             min(cli.periods, 32), cli.steps, cli.saves)

    # Seed 0 is the axis probe: it defines the centre, so its own winding is
    # the difference of two identical numbers.  Keep its orbit for the plot,
    # drop it from everything that is reported.
    return {"ys": np.asarray(ys[1:]), "ok": np.asarray(ok[1:]),
            "escaped": np.asarray(escaped[1:]), "iota": np.asarray(iota[1:]),
            "resid": np.asarray(resid[1:]), "seeds": np.asarray(seeds[1:]),
            "axis": np.asarray(ys[0]),
            "walltime": walltime, "drift": drift}


def section_RZ(seq, res, plane):
    """(R, Z) of the crossings and of the axis, at one logical zeta plane."""
    saves = res["saves_per_period"]
    off = int(round(plane * saves))
    R, Z = to_RZ(seq, jnp.asarray(res["ys"][:, off::saves, :]), plane)
    aR, aZ = to_RZ(seq, jnp.asarray(res["axis"][off::saves, :]), plane)
    return (np.asarray(R), np.asarray(Z),
            np.asarray(aR), np.asarray(aZ))


def plot(res, geometry, label, plane, nfp, RZ, path):
    R, Z, aR, aZ = RZ
    keep = ~(res["escaped"] | ~res["ok"])
    render_section(
        R, Z, res["iota"], res["resid"], res["seeds"][:, 0], keep,
        title=f"{geometry}  |  {label}  |  $\\zeta = {plane:g}$\n"
              f"{R.shape[1]} crossings/line",
        subtitle=f"nfp = {nfp}   |   h/2 step drift {res['drift']:.1e}",
        axis_RZ=(aR, aZ), path=path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x", choices=sorted(NFP))
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--fields", default="k2,k1")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--r-min", type=float, default=0.03)
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--periods", type=int, default=150)
    ap.add_argument("--steps", type=int, default=24,
                    help="prescribed steps per field period")
    ap.add_argument("--saves", type=int, default=8,
                    help="samples kept per period; must divide --steps")
    ap.add_argument("--planes", default="0",
                    help="logical zeta of each section, comma separated")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--bench", action="store_true",
                    help="time the prescribed schedule against the adaptive one")
    ap.add_argument("--bench-periods", type=int, default=20)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--out", default="outputs/poincare")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    nfp = NFP[cli.geometry]
    os.makedirs(cli.out, exist_ok=True)

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    angle = field_agreement(seq, ops)
    print(f"[check] max angle between the k=2 and k=1 harmonic fields: "
          f"{angle:.3e} rad", flush=True)

    summary = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "nfp": nfp, "periods": cli.periods, "steps": cli.steps,
               "saves": cli.saves, "seeds": cli.seeds,
               "field_angle_rad": angle, "fields": {}}

    for name in cli.fields.split(","):
        k, dirichlet, label = FIELDS[name]
        dof = ops.null_2_dbc[0] if k == 2 else ops.null_1[0]
        if cli.bench:
            summary["bench_" + name] = bench(
                logical_field(seq, dof, k, dirichlet),
                seed_line(cli.seeds, cli.r_min, cli.r_max), cli)

        res = run_field(seq, dof, k, dirichlet, nfp, cli)
        res["saves_per_period"] = cli.saves

        keep = ~(res["escaped"] | ~res["ok"])
        iota = res["iota"][keep]
        print(f"[{name}] {label}: {res['walltime']:.1f}s trace, "
              f"{int((~keep).sum())}/{cli.seeds} lost, "
              f"step drift {res['drift']:.2e}, "
              f"iota {float(iota.min()):.4f}..{float(iota.max()):.4f}",
              flush=True)

        # (R, Z) goes into the archive alongside (u, v): re-deriving it needs
        # the map, and rebuilding the map is the expensive half of this script.
        sections = {}
        for plane in (float(v) for v in cli.planes.split(",")):
            path = os.path.join(
                cli.out, f"poincare_{cli.geometry}_{name}_zeta{plane:g}.png")
            RZ = section_RZ(seq, res, plane)
            plot(res, cli.geometry, label, plane, nfp, RZ, path)
            print(f"        -> {path}", flush=True)
            for key, arr in zip(("R", "Z", "axisR", "axisZ"), RZ):
                sections[f"{key}_zeta{plane:g}"] = arr

        np.savez_compressed(
            os.path.join(cli.out, f"trace_{cli.geometry}_{name}.npz"),
            ys=res["ys"], iota=res["iota"], resid=res["resid"],
            seeds=res["seeds"], escaped=res["escaped"], ok=res["ok"],
            axis=res["axis"], **sections)
        summary["fields"][name] = {
            "walltime_s": res["walltime"], "step_drift": res["drift"],
            "lost": int((~keep).sum()),
            "iota_min": float(iota.min()), "iota_max": float(iota.max()),
            "resid_max": float(res["resid"][keep].max()),
        }

    with open(os.path.join(cli.out, f"summary_{cli.geometry}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
