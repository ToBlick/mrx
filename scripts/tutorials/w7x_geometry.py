"""Tutorial 1: load a GVEC state of W7-X and look at it.

The file is GVEC's own state (``GVEC_State_*.dat``, see
``docs/source/concepts/gvec_mrx_interface.md``): ``R``, ``Z`` and the stream
function ``lambda`` as radial B-splines times Fourier series in the angles
``(theta, zeta)`` -- ``zeta`` spans ONE field period, ``nfp`` completes
the torus -- and the profiles ``Phi``, ``chi``, ``iota``, ``p`` at the
radial interpolation points. Closed form: MRX evaluates the series
wherever it needs a value (``mrx.gvec.StateField``), there is no grid in
between. A VMEC ``wout_*.nc`` is read the same way (``mrx.vmec``).

``build_sequence`` turns it into an MRX domain: it builds the spline
coefficients of ``R`` and ``Z`` on the map's own spline space (resolution
``ns``, degree ``p``, polar at the axis, periodic in both angles) from the
series coefficients, mode by mode, measures the toroidal handedness so that ``det DF > 0``, installs the map
``F(rho, theta, zeta) = (R cos phi, +-R sin phi, Z)`` with
``phi = 2 pi zeta / nfp``, and assembles the operators and preconditioners
of the de Rham sequence on it. Everything else in MRX -- the Poisson solve,
the vacuum field, the relaxation -- starts from this ``seq``.

This script prints what the file holds, builds the sequence, and draws the
state's pressure on the torus and in poloidal cuts (``mrx.plotting``).

    python -u scripts/tutorials/w7x_geometry.py
    python -u scripts/tutorials/w7x_geometry.py --geometry data/GVEC_State_final.dat --ns 12,24,24 --p 3
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", default="data/GVEC_State_final.dat",
                    help="a GVEC state file (.dat) or a VMEC wout (.nc)")
    ap.add_argument("--ns", default="12,24,24", help="map and sequence resolution")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--cuts", type=int, default=6, help="poloidal cuts per field period")
    ap.add_argument("--out", default="outputs/tutorials/w7x_geometry")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.geometry import build_sequence
    from mrx.gvec import load_clebsch, read_equilibrium
    from mrx.plotting import get_2d_grids, plot_crossections_separate, plot_torus

    # --- 1. what the file holds --------------------------------------------
    # R, Z, lambda as radial B-splines x Fourier series; the profiles at the
    # radial interpolation points (a wout arrives refit into the same blocks).
    st = read_equilibrium(cli.geometry)
    nfp = st["nfp"]
    X1 = st["X1"]
    print(f"[file] {cli.geometry}: nfp = {nfp}, radial B-splines of degree {st['deg']} "
          f"({X1['coef'].shape[1]} functions), {len(X1['m'])} Fourier modes with "
          f"m <= {X1['m'].max()}, |n| <= {abs(X1['n']).max()}")
    # load_clebsch tabulates the profile splines on 401 uniform radii: the
    # flux derivative dPhi (GVEC units, per radian of toroidal angle), chi'
    # = iota Phi' and the pressure. The same dict feeds the relaxation's
    # initial condition (tutorial 4).
    cb = load_clebsch(cli.geometry)
    rho_t, p_t = jnp.asarray(cb["rho"]), jnp.asarray(cb["p"])
    iota = cb["dchi"][1:] / cb["dPhi"][1:]
    print(f"[file] profiles: Phi_edge = {2 * np.pi * st['profiles']['phi'][-1]:.4f} Wb, "
          f"iota {iota[0]:+.4f} -> {iota[-1]:+.4f} (per full turn), "
          f"p_axis = {cb['p'][0]:.4g} Pa, p_edge = {cb['p'][-1]:.4g} Pa")
    if "a_minor" in st:
        print(f"[file] a = {st['a_minor']:.3f} m, R0 = {st['r_major']:.3f} m")

    def p_fit(x):
        return jnp.interp(x[0], rho_t, p_t)

    # --- 2. the sequence on the file's geometry -----------------------------
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    jac = np.asarray(seq.geometry.jacobian_j)
    print(f"[seq] ns = {ns}, p = {cli.p}: {seq.n(0)} 0-form, {seq.n(1)} 1-form, {seq.n(2)} 2-form, "
          f"{seq.n(3)} 3-form DoFs (free spaces); det DF at the quadrature points in "
          f"[{jac.min():.3e}, {jac.max():.3e}]")
    x = jnp.array([0.5, 0.25, 0.0])
    print(f"[seq] the map at logical {np.asarray(x)}: physical {np.asarray(seq.map(x)).round(4)} m")

    # --- 3. the pressure on the torus ----------------------------------------
    def p_h(x):
        return p_fit(x)

    zetas = np.arange(cli.cuts) / cli.cuts
    n = 48
    grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
                 for z in zetas]
    grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                                ny=4 * n, nz=4 * n, invert_z=True)
    fig, _ = plot_torus(p_h, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                        elev=25, azim=40, cbar_label=r"$p$ [Pa]")
    path = os.path.join(cli.out, "torus_pressure.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")
    fig, _ = plot_crossections_separate(p_h, grids_pol, zetas)
    path = os.path.join(cli.out, "crossections_pressure.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
