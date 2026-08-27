"""Tutorial 2: a Poisson problem on the W7-X domain.

Solve ``-Delta u = f`` with ``u = 0`` on the wall (the last closed flux
surface of the export) and the heat source ``f = 1 - rho^2`` in the logical
radius, using 0-forms. On a mapped domain the Laplacian carries the metric
of the map: ``-Delta u = f`` in the physical volume is the weak form
``int grad u . grad v dV = int f v dV`` assembled with the spline metric,
and the solve is the preconditioned CG of ``seq.apply_inverse_laplacian``.

    python -u scripts/tutorials/w7x_poisson.py
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
    ap.add_argument("--cuts", type=int, default=6)
    ap.add_argument("--out", default="outputs/tutorials/w7x_poisson")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    os.makedirs(cli.out, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.differential_forms import DiscreteFunction
    from mrx.geometry import build_sequence
    from mrx.plotting import get_2d_grids, plot_crossections_separate, plot_torus

    seq, _ = build_sequence(cli.geometry, ns, cli.p)

    # --- the source, a function of the logical point -------------------------
    def f(x):
        return (1.0 - x[0] ** 2) * jnp.ones(1)

    # ``load`` integrates f against the 0-form basis (the right-hand side of
    # the weak form); ``dirichlet=True`` restricts to the functions that
    # vanish on the wall.
    rhs = seq.load(f, 0, dirichlet=True)
    u_hat, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=True, return_info=True)
    resid = float(jnp.linalg.norm(seq.apply_laplacian(u_hat, 0, dirichlet=True) - rhs)
                  / jnp.linalg.norm(rhs))
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.E(0, True))

    # The energy identity int |grad u|^2 = int f u, both sides from the
    # discrete operators, and the value on the axis.
    energy = float(u_hat @ seq.apply_laplacian(u_hat, 0, dirichlet=True))
    work = float(u_hat @ rhs)
    u_axis = float(u_h(jnp.array([0.0, 0.0, 0.0]))[0])
    print(f"[poisson] CG iterations {abs(int(info))}, relative residual {resid:.2e}")
    print(f"[poisson] int |grad u|^2 = {energy:.6e}, int f u = {work:.6e}, "
          f"u on the axis = {u_axis:.4e}, max u = {float(jax.vmap(u_h)(seq.quad.x).max()):.4e}")

    def u_scalar(x):
        return u_h(x)[0]

    zetas = np.arange(cli.cuts) / cli.cuts
    n = 48
    grids_pol = [get_2d_grids(seq.map, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
                 for z in zetas]
    grid_surface = get_2d_grids(seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                                ny=4 * n, nz=4 * n, invert_z=True)
    fig, _ = plot_torus(u_scalar, grids_pol, grid_surface, cstride=8, gridlinewidth=0.3,
                        elev=25, azim=40, cbar_label=r"$u$")
    path = os.path.join(cli.out, "torus_u.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")
    fig, _ = plot_crossections_separate(u_scalar, grids_pol, zetas)
    path = os.path.join(cli.out, "crossections_u.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
