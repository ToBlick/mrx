"""Tutorial 1: load a GVEC export of W7-X and look at it.

The file is a flat-schema GVEC export (``docs/source/concepts/gvec_mrx_interface.md``): a
tensor grid ``eval_points`` of logical ``(rho, theta, zeta)`` in ``[0, 1]^3``
-- ``zeta`` spans ONE field period -- with the cylindrical ``R``, ``Z`` of
every point, the ``pressure``, and the three Clebsch scalars
``clebsch/{dPhi_dr, dchi_dr, LA}`` that describe the equilibrium field.

``build_sequence`` turns it into an MRX domain: it fits ``R`` and ``Z`` on
the map's own spline space (resolution ``ns``, degree ``p``, polar at the
axis, periodic in both angles), measures the toroidal handedness so that
``det DF > 0``, installs the map ``F(rho, theta, zeta) = (R cos phi, +-R sin
phi, Z)`` with ``phi = 2 pi zeta / nfp``, and assembles the operators and
preconditioners of the de Rham sequence on it. Everything else in MRX --
the Poisson solve, the vacuum field, the relaxation -- starts from this
``seq``.

This script prints what the file holds, builds the sequence, and draws the
file's pressure on the torus and in poloidal cuts (``mrx.plotting``).

    python -u scripts/tutorials/w7x_geometry.py
    python -u scripts/tutorials/w7x_geometry.py --geometry data/w7x_example.h5 --ns 12,24,24 --p 3
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", default="data/GVEC_State_final.dat",
                    help="a GVEC state file or a flat-schema export (.h5)")
    ap.add_argument("--ns", default="12,24,24", help="map and sequence resolution")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--cuts", type=int, default=6, help="poloidal cuts per field period")
    ap.add_argument("--out", default="outputs/tutorials/w7x_geometry")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)

    import h5py
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.geometry import build_sequence
    from mrx.gvec import fit_scalar_spline
    from mrx.plotting import get_2d_grids, plot_crossections_separate, plot_torus

    # --- 1. what the file holds --------------------------------------------
    if cli.geometry.endswith(".dat"):
        # GVEC's own state: R, Z, lambda as radial B-splines x Fourier series,
        # the profiles at the radial interpolation points. Closed form: MRX
        # evaluates it wherever it needs a value.
        from mrx.gvec import profile_spline, read_state
        st = read_state(cli.geometry)
        nfp = st["nfp"]
        X1 = st["X1"]
        print(f"[file] {cli.geometry}: GVEC state, nfp = {nfp}, radial B-splines of degree "
              f"{st['deg']} on {len(st['sp']) - 1} elements ({X1['coef'].shape[1]} functions), "
              f"{len(X1['m'])} Fourier modes with m <= {X1['m'].max()}, |n| <= {abs(X1['n']).max()}")
        prof = st["profiles"]
        print(f"[file] profiles at {len(prof['s'])} radial points: Phi_edge = {prof['phi'][-1]:.4f}, "
              f"iota {prof['iota'][0]:+.4f} -> {prof['iota'][-1]:+.4f}, p_axis = {prof['pressure'][0]:.4g} Pa, "
              f"p_edge = {prof['pressure'][-1]:.4g} Pa; a = {st['a_minor']:.3f} m, R0 = {st['r_major']:.3f} m")
        rho_t = np.linspace(0.0, 1.0, 401)
        p_t = jnp.asarray(profile_spline(st, "pressure")(rho_t))
        rho_t = jnp.asarray(rho_t)

        def p_fit(x):
            return jnp.interp(x[0], rho_t, p_t)
    else:
        with h5py.File(cli.geometry, "r") as f:
            shape = tuple(int(f.attrs[k]) for k in ("n_rho", "n_theta", "n_zeta"))
            nfp = int(f.attrs["nfp"])
            ep = np.asarray(f["eval_points"])
            pressure = np.asarray(f["pressure"]).reshape(shape)
            R = np.asarray(f["R"]).reshape(shape)
            Z = np.asarray(f["Z"]).reshape(shape)
            print(f"[file] {cli.geometry}: grid {shape}, nfp = {nfp}")
            print(f"[file] datasets: {sorted(k for k in f.keys())} + clebsch/{sorted(f['clebsch'].keys())}")
        axes = [np.unique(ep[:, i]) for i in range(3)]
        print(f"[file] rho in [{axes[0][0]:.3f}, {axes[0][-1]:.3f}], theta and zeta on [0, 1) "
              f"(zeta is one field period = {360 / nfp:.0f} degrees)")
        print(f"[file] R in [{R.min():.3f}, {R.max():.3f}] m, Z in [{Z.min():.3f}, {Z.max():.3f}] m, "
              f"axis at R = {R[0].mean():.3f} m; p_axis = {pressure[0].mean():.4g} Pa, "
              f"p_edge = {pressure[-1].mean():.4g} Pa")
        # The same data-node spline fit ``load_clebsch`` uses for lambda: knots
        # at the sample points, so the fit interpolates the file exactly.
        p_fit = jax.jit(fit_scalar_spline(axes, pressure, ("clamped", "periodic", "periodic")))

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
