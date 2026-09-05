"""Figures of a ``scripts/relax.py`` run: the weak pressure on the torus and
in poloidal cuts, and the force residual against the energy removed.

Reads the run directory (``relax.json`` for the parameters and the trace,
the ``ic`` / ``final`` checkpoints for the fields), rebuilds the sequence
from ``geometry_path`` and computes the weak pressure of each field.

    python -u scripts/plot_relaxation.py outputs/run --cuts 6

Options
    run                  the run directory (positional)
    --out DIR            figure directory [<run>/figures]
    --fields F           comma-separated subset of ic,final [final]
    --cuts N             poloidal cuts per field period [6]
    --n N                points per cut side [48]
    --precision {float32,float64}

Writes ``torus_<name>.png``, ``crossections_<name>.png`` and ``trace.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a scripts/relax.py run directory (relax.json + checkpoints/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", default="final")
    ap.add_argument("--cuts", type=int, default=6)
    ap.add_argument("--n", type=int, default=48)
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
    from mrx.plotting import plot_crossections_separate, plot_torus, plot_twin_axis, torus_grids
    from mrx.relaxation import compute_force, weak_pressure

    run = os.path.abspath(cli.run)
    out = cli.out or os.path.join(run, "figures")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(run, "relax.json")) as fh:
        results = json.load(fh)
    attrs = results["params"]
    ckpts = {int(os.path.basename(f)[6:12]): f
             for f in glob.glob(os.path.join(run, "checkpoints", "state_*.h5"))}
    geometry, ns, p = attrs["geometry_path"], tuple(attrs["ns"]), int(attrs["p"])
    print(f"[run] {run}: {geometry} ns={ns} p={p}", flush=True)
    seq, _ = build_sequence(geometry, ns, p, nfp=attrs["nfp"],
                            r_windows=parse_r_refine(attrs["r_refine"]))
    aux = bool(attrs["auxiliary_B_field"])
    zetas, grids_pol, grid_surface = torus_grids(seq.map, cli.cuts, cli.n)

    for name in (w.strip() for w in cli.fields.split(",")):
        step = {"ic": min(ckpts), "final": max(ckpts)}[name]
        with h5py.File(ckpts[step], "r") as fh:
            B = jnp.asarray(np.asarray(fh["B_n"], dtype=np.float64))
        _, _, J, X, _ = compute_force(B, seq, aux)
        pw = DiscreteFunction(weak_pressure(J, X, seq, aux)[0], seq.basis_0, seq.E(0, True))

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

    trace = results["trace"]
    fig, _ = plot_twin_axis(trace["F"], np.cumsum(-np.asarray(trace["dE"], dtype=float)),
                            left_label=r"$\|F\|_{M}$",
                            right_label=r"$E_0 - E$", left_plot_kwargs=dict(marker=""),
                            right_plot_kwargs=dict(marker=""))
    path = os.path.join(out, "trace.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
