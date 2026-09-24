"""Part B of the Landreman verification: a scripts/relax.py run from the exact field, checked against the closed form.

Rebuilds the run's sequence (relax.json ``params``) and, for every checkpoint ``checkpoints/state_<step>.h5``, puts
the stored field back in physical units with the IC's norm (relax.json ``ic.B_norm_raw``) and compares it with the
closed form (:func:`landreman_verify.compare`: ``B_err``, the flux-ratio iota per layer, with the weak pressure
``p_err``), and its ``beta_vol`` and the force residual. Writes ``<run>/landreman_relaxed.json``. With
``--poincare`` the same job then traces ``ic,final`` on 5 planes (scripts/poincare_trace.py, defaults).
Usage (a GPU job)::

    python -u scripts/landreman_relaxed.py --run OUT/partB_sheared_16 --case sheared --poincare
    python scripts/landreman_relaxed.py --plot RUN1 RUN2 ... OUT.png      # login node, matplotlib
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def plot(runs, out):
    """Login-node figure of part B: the per-step squared force residual and energy change of every run, and the
    per-checkpoint ``B_err``, ``beta_vol / beta_V - 1`` from its ``landreman_relaxed.json``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for run in runs:
        rj = json.load(open(os.path.join(run, "relax.json")))
        lr = json.load(open(os.path.join(run, "landreman_relaxed.json")))
        label = lr["case"]
        tr = rj["trace"]
        steps = np.arange(1, len(tr["resid"]) + 1)
        axs[0].semilogy(steps, tr["resid"], label=label)
        axs[1].semilogy(steps, -np.cumsum(tr["dE"]) / rj["summary"]["E0"], label=label)
        st = [r["step"] for r in lr["rows"]]
        axs[2].semilogy(st, [r["B_err"] for r in lr["rows"]], "o-", label=f"{label}: B_err")
        axs[2].semilogy(st, [abs(r["beta_vol"] / lr["exact"]["beta_V"] - 1) for r in lr["rows"]], "s--",
                        label=f"{label}: |beta/beta_V - 1|")
    axs[0].set(xlabel="Newton step", title=r"$\|F\|^2_{\rm norm}$")
    axs[1].set(xlabel="Newton step", title=r"$(E_0 - E)/E_0$")
    axs[2].set(xlabel="step", title="against the exact state")
    for ax in axs:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--plot":
        return plot(sys.argv[2:-1], sys.argv[-1])
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case", required=True, choices=("iota2", "sheared", "shearedA"))
    ap.add_argument("--poincare", action="store_true")
    ap.add_argument("--map-batch", type=int, default=8192, help="mrx.MAP_BATCH_SIZE_INNER (landreman_verify.py)")
    cli = ap.parse_args(argv)

    import h5py
    import jax.numpy as jnp

    import landreman_check
    import mrx
    import landreman_equilibria as le
    from landreman_verify import _log, compare
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces
    from mrx.precision import DTYPE
    from mrx.relaxation import compute_force, force_scale_jit, pressure_diagnostics, weak_pressure

    mrx.MAP_BATCH_SIZE_INNER = cli.map_batch

    run = json.load(open(os.path.join(cli.run, "relax.json")))
    prm = run["params"]
    case = le.CASES[cli.case]
    chk = landreman_check.check(case)
    exact = dict(beta_V=chk["beta_V"], volume=chk["volume"])
    seq, _ = build_sequence(prm["geometry_path"], tuple(prm["ns"]), prm["p"], symmetry=prm["symmetry"])
    compute_nullspaces(seq)
    norm = run["ic"]["B_norm_raw"]
    rows = []
    for f in sorted(glob.glob(os.path.join(cli.run, "checkpoints", "state_*.h5"))):
        with h5py.File(f, "r") as fh:
            step = int(fh.attrs["step"])
            B = jnp.asarray(fh["B_n"][()], dtype=DTYPE)
        F, pr, J, X, _ = compute_force(B, seq, False)
        F_norm = float(jnp.sqrt(F @ seq.even.apply_mass_matrix(F, 2)))
        p_w, F_w, v = weak_pressure(J, X, seq, False)
        diag = pressure_diagnostics(B, pr, p_w, F_w, v, seq)
        r = dict(step=step, resid=float((F_norm / force_scale_jit(seq, B)) ** 2),
                 E=0.5 * float(seq.odd.l2_norm_sq(B, 2)), beta_vol=float(diag["beta_vol"]),
                 JoverB=float(seq.odd.l2_norm(J, 1) / seq.odd.l2_norm(B, 2)))
        r.update(compare(seq, B, norm, case, exact, p_w=p_w, p_strong=pr))
        if not rows:
            B0 = B
        r["B_change"] = float(seq.odd.l2_norm(B - B0, 2) / seq.odd.l2_norm(B0, 2))
        rows.append(r)
        _log(f"step {step}: resid {r['resid']:.3e}  E {r['E']:.10f}  beta_vol {r['beta_vol']:.5e} "
             f"(exact {exact['beta_V']:.5e})  B_err {r['B_err']:.3e}  |B-B0|/|B0| {r['B_change']:.3e}  p_err {r['p_err']:.3e} (centered {r['p_err_centered']:.2e}, strong {r['p_strong_err_centered']:.2e})  "
             f"iota {min(r['iota_h']):.6f}..{max(r['iota_h']):.6f}")
    with open(os.path.join(cli.run, "landreman_relaxed.json"), "w") as fh:
        json.dump(dict(case=cli.case, exact=exact, rows=rows), fh, indent=1)
    if cli.poincare:
        subprocess.run([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            "poincare_trace.py"),
                        "--run", cli.run, "--fields", "ic,final"], check=True)


if __name__ == "__main__":
    main()
