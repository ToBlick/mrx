"""Tutorial 4: a short relaxation run on the W7-X export.

The initial condition is the export's own equilibrium field, ``B = dA'``
from the histopolated Clebsch potential (exactly divergence-free, tangential
to the wall, nested surfaces). The relaxation is the energy descent of
``mrx.relaxation`` with the defaults of ``scripts/relax.py``: conjugate-
gradient direction, analytic line search under a CFL cap, no resistivity,
no velocity smoothing. It conserves helicity and lowers the magnetic energy
until ``J x B = grad p`` in the weak sense; ``p`` is not prescribed, it is
the Lagrange multiplier the descent finds.

This script runs a few hundred steps, prints the traces, draws ``||F||``
against the energy, the weak pressure on the torus, and the Poincare
sections before and after. ``scripts/relax.py`` is the production driver
(archives, QoIs, snapshots, resistivity, smoothing, seeds).

    python -u scripts/tutorials/w7x_relaxation.py
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", default="data/GVEC_State_final.dat",
                    help="a GVEC state file or a flat-schema export (.h5)")
    ap.add_argument("--ns", default="12,24,24")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--outer", type=int, default=10, help="outer (recorded) iterations")
    ap.add_argument("--inner", type=int, default=30, help="compiled steps per outer iteration")
    ap.add_argument("--cuts", type=int, default=6)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--out", default="outputs/tutorials/w7x_relaxation")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometries import build_sequence, geometry_nfp
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import clebsch_potential_form, divergence_norm, potential_two_form
    from mrx.nullspace import compute_nullspaces
    from mrx.plotting import get_2d_grids, plot_torus, plot_twin_axis
    from mrx.poincare import section_figure
    from mrx.relaxation import (DescentMethod, TimeStepChoice, TimeStepper, compute_force,
                                relaxation_loop, weak_pressure)

    nfp = geometry_nfp(cli.geometry)
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    seq.set_operators(compute_nullspaces(seq, ops))

    # --- the initial condition: the export's field as B = dA' -------------------
    cb = load_clebsch(cli.geometry, seq.basis_0.types)
    B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
    print(f"[ic] ||B||_M before normalisation {norm:.4e}, ||div B|| {divergence_norm(seq, B0):.2e}, "
          f"wall-normal part {wall:.1e}")

    # --- the descent with scripts/relax.py's defaults ---------------------------
    ts = TimeStepper(seq=seq, descent_method=DescentMethod.CONJUGATE_GRADIENT,
                     dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                     eta_every=1, resistive=False, history_size=3,
                     velocity_smoothing_order=0, velocity_smoothing_scale=0.0)
    state, traces = relaxation_loop(B0, ts, num_iters_outer=cli.outer,
                                    num_iters_inner=cli.inner, dt0=1.0)
    F = np.asarray(traces["force_norm"], dtype=float)
    E = np.asarray(traces["energy"], dtype=float)
    H = np.asarray(traces["helicity"], dtype=float)
    steps = np.asarray(traces["iteration"])
    print(f"[relax] {steps[-1]} steps: ||F|| {F[0]:.3e} -> {F[-1]:.3e}, E_0 - E = {E[0] - E[-1]:.3e}, "
          f"H {H[0]:+.3e} -> {H[-1]:+.3e} (dH = {H[-1] - H[0]:+.1e}), "
          f"||div B|| {float(traces['divergence_B'][-1]):.1e}")
    B = state.B_n

    fig, _ = plot_twin_axis(F, E, left_label=r"$\|F\|_M$", right_label=r"$E$",
                            num_iters_inner=cli.inner, left_marker="o", right_marker="s")
    path = os.path.join(cli.out, "trace.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")

    # --- the weak pressure of the relaxed field ------------------------------
    _, _, J, Hf, _ = compute_force(B, seq)
    p_w, _, _ = weak_pressure(J, Hf, seq)
    pw = DiscreteFunction(p_w, seq.basis_0, seq.E(0, True))

    def p_h(x):
        return pw(x)[0]

    print(f"[relax] weak pressure on the axis {float(p_h(jnp.array([0.0, 0.0, 0.0]))):.4e} "
          f"(||B||_M = 1 units)")
    zetas = np.arange(cli.cuts) / cli.cuts
    n = 48
    grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
                 for z in zetas]
    grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                                ny=4 * n, nz=4 * n, invert_z=True)
    fig, _ = plot_torus(p_h, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                        elev=25, azim=40, cbar_label=r"$p_w$")
    path = os.path.join(cli.out, "torus_pw.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")

    # --- sections before and after ---------------------------------------------
    for name, field in (("ic", B0), ("final", B)):
        fig, res = section_figure(seq, field, nfp, plane=0.0, n_periods=cli.periods,
                                  title=f"{os.path.basename(cli.geometry)} {ns} p={cli.p}  |  {name}")
        path = os.path.join(cli.out, f"poincare_{name}_zeta0.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        keep = ~(res["escaped"] | ~res["ok"] | res["chaotic"])
        print(f"[{name}] {int(keep.sum())}/{keep.size} regular lines, iota in "
              f"[{float(res['iota'][keep].min()):.4f}, {float(res['iota'][keep].max()):.4f}]  -> {path}")


if __name__ == "__main__":
    main()
