"""Tutorial 2: the vacuum field of the QA domain -- its own equilibrium.

Inside a perfectly conducting wall the current-free, divergence-free field
with ``B . n = 0`` and one unit of toroidal flux is the harmonic 2-form of
the Dirichlet de Rham complex: ``curl B = 0``, ``div B = 0``, tangential to
the wall. QA (``LandremanPaul2021_QA``) is a **vacuum** equilibrium -- its
pressure is zero and its current vanishes -- so this harmonic 2-form *is*
the equilibrium field of the loaded state, reconstructed here from the
bounded geometry alone, with no reference to the field the file stored.

MRX builds it by a direct Hodge decomposition of a seed field
(``mrx.nullspace.compute_nullspaces``) -- two Hodge solves, no eigenvalue
iteration -- and stores it on the sequence's operators, where the Leray
projection and the helicity of the relaxation are deflated against it.

The harmonic-form ratio ``||curl B|| / ||B||`` and the Rayleigh quotient of
the Hodge Laplacian must reach round-off (~1e-10), which needs float64: run
this script with ``MRX_DTYPE=float64`` (the package default), never float32.

This script builds the field, verifies ``div``, ``curl`` and the Rayleigh
quotient, draws ``|B|`` on the torus (the 2-form pushed forward by Piola),
and traces a Poincare section with its rotational transform.

    MRX_DTYPE=float64 python -u scripts/tutorials/qa_vacuum_field.py
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc",
                    help="a VMEC wout (.nc) or a GVEC state file (.dat)")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--cuts", type=int, default=6)
    ap.add_argument("--periods", type=int, default=200, help="field periods per traced line")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--out", default="outputs/tutorials/qa_vacuum_field")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.differential_forms import DiscreteFunction, Pushforward
    from mrx.geometry import build_sequence, geometry_nfp
    from mrx.initial_conditions import divergence_norm
    from mrx.nullspace import compute_nullspaces, get_nullspace, harmonic_rayleigh
    from mrx.plotting import get_2d_grids, plot_crossections_separate, plot_torus
    from mrx.poincare import section_figure
    from mrx.relaxation import compute_force

    nfp = geometry_nfp(cli.geometry)
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    seq.set_operators(compute_nullspaces(seq, ops))

    # --- the harmonic 2-form ----------------------------------------------------
    B = get_nullspace(seq.get_operators(), 2, True)[0]
    B = B / float(seq.l2_norm(B, 2))
    _, _, J, _, _ = compute_force(B, seq)
    ratio = float(seq.l2_norm(J, 1))
    rayleigh = float(harmonic_rayleigh(seq, B, 2))
    print(f"[vacuum] ||div B|| = {divergence_norm(seq, B):.2e}, "
          f"||curl B|| / ||B|| = {ratio:.2e}, "
          f"Rayleigh quotient of the Hodge Laplacian = {rayleigh:.2e}")
    if ratio > 1e-8 or rayleigh > 1e-8:
        print("[vacuum] WARNING: harmonic-form residual is not at round-off -- "
              "run in float64 (MRX_DTYPE=float64)")

    # --- |B| on the torus -----------------------------------------------------
    B_phys = Pushforward(DiscreteFunction(B, seq.basis_2, seq.E(2, True)), seq.map, 2)

    def B_mag(x):
        return jnp.linalg.norm(B_phys(x))

    # Probe next to the axis, not on it: the map is singular at rho = 0 and
    # the Piola factor DF / det DF is 0 / 0 there.
    print(f"[vacuum] |B| near the axis {float(B_mag(jnp.array([0.02, 0.0, 0.0]))):.4f}, "
          f"inboard/outboard midplane at zeta = 0: "
          f"{float(B_mag(jnp.array([0.99, 0.5, 0.0]))):.4f} / {float(B_mag(jnp.array([0.99, 0.0, 0.0]))):.4f} "
          f"(||B||_M = 1)")
    zetas = np.arange(cli.cuts) / cli.cuts
    n = 48
    grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
                 for z in zetas]
    grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                                ny=4 * n, nz=4 * n, invert_z=True)
    fig, _ = plot_torus(B_mag, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                        elev=25, azim=40, cbar_label=r"$|B|$")
    path = os.path.join(cli.out, "torus_Bmag.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")
    fig, _ = plot_crossections_separate(B_mag, grids_pol, zetas)
    path = os.path.join(cli.out, "crossections_Bmag.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")

    # --- field lines --------------------------------------------------------------
    fig, res = section_figure(seq, B, nfp, plane=0.0, n_seeds=cli.seeds,
                              n_periods=cli.periods,
                              title=f"vacuum field of {os.path.basename(cli.geometry)} {ns} p={cli.p}")
    path = os.path.join(cli.out, "poincare_zeta0.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    keep = ~(res["escaped"] | ~res["ok"] | res["chaotic"])
    r, iota = res["seeds"][:, 0][keep], res["iota"][keep]
    print(f"[vacuum] {int(keep.sum())}/{keep.size} regular lines, iota from "
          f"{float(iota[np.argmin(r)]):.4f} (r = {float(r.min()):.2f}) to "
          f"{float(iota[np.argmax(r)]):.4f} (r = {float(r.max()):.2f}); h/2 drift {res['drift']:.1e}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
