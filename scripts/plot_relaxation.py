"""Figures of a ``scripts/relax.py`` run: the weak pressure on the torus and
in poloidal cuts, and the force residual against the energy.

Reads ``B.h5`` (the ``pw_ic``/``pw_final`` weak-pressure 0-forms and the
run's attributes) and ``relax.json`` (the per-step trace) from the run
directory and rebuilds the sequence from ``geometry_path`` like
``scripts/poincare_relax.py`` does.

    python -u scripts/plot_relaxation.py outputs/run/B.h5 --cuts 6

Options
    state                path to B.h5 (positional)
    --out DIR            figure directory [<run>/figures]
    --fields ic,final    which pressures to draw [final]
    --cuts N             poloidal cuts per field period [6]
    --n N                points per cut side [48]
    --geometry PATH      override the state's geometry_path (e.g. after a move)
    --precision {float32,float64}

Writes ``torus_<name>.png``, ``crossections_<name>.png`` and ``trace.png``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("state")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", default="final")
    ap.add_argument("--cuts", type=int, default=6)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    cli = ap.parse_args()

    os.environ["MRX_DTYPE"] = cli.precision
    import h5py
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import build_sequence, parse_r_refine
    from mrx.plotting import (get_2d_grids, plot_crossections_separate, plot_torus,
                              plot_twin_axis)

    run = os.path.dirname(os.path.abspath(cli.state))
    out = cli.out or os.path.join(run, "figures")
    os.makedirs(out, exist_ok=True)

    with h5py.File(cli.state, "r") as fh:
        attrs = dict(fh.attrs)
        dofs = {k: np.asarray(fh[k], dtype=np.float64) for k in fh.keys()}
    geometry = cli.geometry or str(attrs["geometry_path"])
    ns = tuple(int(v) for v in attrs["ns"])
    p = int(attrs["p"])
    nfp_override = None if str(attrs["nfp"]) == "" else int(attrs["nfp"])
    print(f"[state] {cli.state}: {geometry} ns={ns} p={p}", flush=True)
    seq, _ = build_sequence(geometry, ns, p, int(attrs["maxiter"]), nfp=nfp_override,
                            r_windows=parse_r_refine(str(attrs.get("r_refine", ""))))

    # The cuts span one field period: zeta in [0, 1) is the logical toroidal
    # angle of the map. The surface sample runs zeta backwards so its normal
    # points outward for the wireframe shading.
    zetas = np.arange(cli.cuts) / cli.cuts
    grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=cli.n, ny=cli.n, nz=1)
                 for z in zetas]
    grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                                ny=4 * cli.n, nz=4 * cli.n, invert_z=True)

    for name in (w.strip() for w in cli.fields.split(",")):
        key = "pw_" + name
        if key not in dofs:
            raise SystemExit(f"{cli.state} has no {key}; run scripts/relax.py first")
        pw = DiscreteFunction(jnp.asarray(dofs[key]), seq.basis_0, seq.E(0, True))

        def p_h(x, pw=pw):
            return pw(x)[0]

        fig, _ = plot_torus(p_h, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                            elev=25, azim=40, cbar_label=r"$p_w$")
        path = os.path.join(out, f"torus_{name}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  -> {path}", flush=True)

        fig, _ = plot_crossections_separate(p_h, grids_pol, zetas)
        path = os.path.join(out, f"crossections_{name}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  -> {path}", flush=True)

    with open(os.path.join(run, "relax.json")) as fh:
        trace = json.load(fh)["trace"]
    fig, _ = plot_twin_axis(trace["F"], trace["E"], left_label=r"$\|F\|_{M}$",
                            right_label=r"$E$", left_marker="", right_marker="")
    path = os.path.join(out, "trace.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
