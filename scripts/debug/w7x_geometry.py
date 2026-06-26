"""Build a W7-X stellarator spline map from data/W7-X.h5 and sanity-check it.

W7-X.h5 provides R(rho,theta,zeta) and Z(rho,theta,zeta) on a 50^3 grid in
logical coordinates rho in [0,1], theta in [0,2pi), zeta in [0,2pi/nfp) with
nfp=5. We bridge the grid through a (jax) RegularGridInterpolator to get
pointwise R_fn/Z_fn, GREVILLE-interpolate each as a scalar 0-form onto a spline
basis (the recommended path -- see test/test_geometry.py::
test_greville_interpolation_R_Z), and wrap them in mrx.mappings.stellarator_map.

Run directly for sanity checks:
  1) interpolation accuracy of R_h/Z_h vs the data;
  2) is the metric g mostly diagonal?
  3) is the Jacobian > 0 at every quadrature point?
"""
from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.interpolate import RegularGridInterpolator  # noqa: E402

import mrx  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.differential_forms import DiscreteFunction  # noqa: E402
from mrx.mappings import stellarator_map  # noqa: E402

NFP_W7X = 5
H5_DEFAULT = "data/W7-X.h5"
TWO_PI = 2.0 * np.pi


def _load_w7x_grids(h5_path=H5_DEFAULT):
    """Return logical axes (r,theta,zeta in [0,1]) and R,Z grids, periodic-padded."""
    with h5py.File(h5_path, "r") as f:
        rho = np.asarray(f["rho"], dtype=np.float64)        # [0,1]
        theta = np.asarray(f["theta"], dtype=np.float64)    # [0,2pi)
        zeta = np.asarray(f["zeta"], dtype=np.float64)      # [0,2pi/nfp)
        R = np.asarray(f["R"], dtype=np.float64)            # (nr,nt,nz)
        Z = np.asarray(f["Z"], dtype=np.float64)

    # Normalize to logical [0,1]; theta,zeta are periodic -> append wrap point.
    r_ax = rho
    t_ax = np.concatenate([theta / TWO_PI, [1.0]])
    z_ax = np.concatenate([zeta * NFP_W7X / TWO_PI, [1.0]])

    def _pad(grid):
        grid = np.concatenate([grid, grid[:, :1, :]], axis=1)   # theta wrap
        grid = np.concatenate([grid, grid[:, :, :1]], axis=2)   # zeta wrap
        return grid

    return (r_ax, t_ax, z_ax), _pad(R), _pad(Z)


def _rgi_fn(axes, grid):
    """jax RGI bridge; returns f(xi:(3,)) -> (1,) for greville collocation."""
    pts = (jnp.asarray(axes[0]), jnp.asarray(axes[1]), jnp.asarray(axes[2]))
    interp = RegularGridInterpolator(
        pts, jnp.asarray(grid), method="linear",
        bounds_error=False, fill_value=None)   # fill_value=None -> extrapolate

    def f(xi):
        return interp(xi.reshape(1, 3))[0:1]   # (1,)
    return f


def build_w7x_map(map_ns=(12, 24, 24), p=3, h5_path=H5_DEFAULT):
    """Build the W7-X stellarator map plus the R/Z spline functions and bridges."""
    axes, R_grid, Z_grid = _load_w7x_grids(h5_path)
    R_fn = _rgi_fn(axes, R_grid)
    Z_fn = _rgi_fn(axes, Z_grid)

    map_seq = DeRhamSequence(
        map_ns, (p, p, p), 2 * p, ("clamped", "periodic", "periodic"),
        polar=False)
    map_seq.evaluate_1d()

    R_dof = map_seq.interpolate(R_fn, 0)
    Z_dof = map_seq.interpolate(Z_fn, 0)
    R_h = DiscreteFunction(R_dof, map_seq.basis_0, map_seq.e0)
    Z_h = DiscreteFunction(Z_dof, map_seq.basis_0, map_seq.e0)

    map_func = stellarator_map(R_h, Z_h, nfp=NFP_W7X)
    return map_func, {"R_h": R_h, "Z_h": Z_h, "R_fn": R_fn, "Z_fn": Z_fn,
                      "axes": axes, "map_seq": map_seq}


def _interp_accuracy(info, n=400, seed=0):
    """Max/RMS error of the spline R_h/Z_h vs the RGI bridge at random points."""
    rng = np.random.default_rng(seed)
    # sample away from the exact axis to avoid the rho=0 extrapolation corner
    xs = jnp.asarray(np.column_stack([
        rng.uniform(0.02, 0.98, n), rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n)]))
    R_h, Z_h, R_fn, Z_fn = info["R_h"], info["Z_h"], info["R_fn"], info["Z_fn"]
    Rh = jax.vmap(lambda x: R_h(x)[0])(xs)
    Zh = jax.vmap(lambda x: Z_h(x)[0])(xs)
    Rf = jax.vmap(lambda x: R_fn(x)[0])(xs)
    Zf = jax.vmap(lambda x: Z_fn(x)[0])(xs)
    eR = np.asarray(jnp.abs(Rh - Rf)); eZ = np.asarray(jnp.abs(Zh - Zf))
    return dict(R_max=eR.max(), R_rms=np.sqrt((eR**2).mean()),
               Z_max=eZ.max(), Z_rms=np.sqrt((eZ**2).mean()))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="8,12,6",
                    help="Resolution used for BOTH the map interpolation and the "
                         "polar solve seq (same resolution, per design).")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="jax.lax.map batch size for the geometry-term (jacfwd) "
                         "computation; keeps GPU memory bounded (avoids OOM from "
                         "full-vmap over all quad points).")
    ap.add_argument("--h5", type=str, default=H5_DEFAULT)
    cli = ap.parse_args()
    ns = tuple(int(x) for x in cli.ns.split(","))

    # Bound the geometry jacfwd memory: batch the lax.map instead of full vmap.
    mrx.MAP_BATCH_SIZE_INNER = int(cli.batch_size)
    print(f"[w7x] MAP_BATCH_SIZE_INNER={mrx.MAP_BATCH_SIZE_INNER}", flush=True)

    print(f"[w7x] building stellarator map (nfp={NFP_W7X}) ns={ns} p={cli.p}",
          flush=True)
    map_func, info = build_w7x_map(map_ns=ns, p=cli.p, h5_path=cli.h5)

    acc = _interp_accuracy(info)
    print(f"[check 0] spline-vs-data interpolation error: "
          f"R max={acc['R_max']:.2e} rms={acc['R_rms']:.2e}  "
          f"Z max={acc['Z_max']:.2e} rms={acc['Z_rms']:.2e}", flush=True)

    # Build a polar solve sequence and set the map -> triggers geometry terms.
    print(f"[w7x] building polar solve seq ns={ns} p={cli.p} and setting map ...",
          flush=True)
    seq = DeRhamSequence(
        ns, (cli.p, cli.p, cli.p), 2 * cli.p,
        ("clamped", "periodic", "periodic"), polar=True,
        betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(map_func)

    g = np.asarray(seq.metric_jkl)          # (n_q, 3, 3)
    jac = np.asarray(seq.jacobian_j)        # (n_q,)
    n_q = g.shape[0]

    # --- check 1: is g mostly diagonal? ---
    diag = np.sqrt((np.diagonal(g, axis1=1, axis2=2) ** 2).sum(axis=1))
    off = np.sqrt((g ** 2).sum(axis=(1, 2)) - (np.diagonal(g, axis1=1, axis2=2) ** 2).sum(axis=1))
    ratio = off / np.maximum(diag, 1e-300)
    # which off-diagonal pair dominates
    pair_names = [("rt", 0, 1), ("rz", 0, 2), ("tz", 1, 2)]
    pair_frac = {nm: float(np.mean(np.abs(g[:, i, j]) / np.maximum(diag, 1e-300)))
                 for nm, i, j in pair_names}
    print(f"[check 1] g diagonality over {n_q} quad pts: "
          f"||offdiag||/||diag|| mean={ratio.mean():.3e} max={ratio.max():.3e}",
          flush=True)
    print(f"          mean |g_ij|/||diag|| by pair: "
          + "  ".join(f"{k}={v:.3e}" for k, v in pair_frac.items()), flush=True)
    print(f"          verdict: {'MOSTLY DIAGONAL' if ratio.mean() < 0.1 else 'NOT diagonal'}",
          flush=True)

    # --- check 2: Jacobian > 0 everywhere ---
    print(f"[check 2] Jacobian over {n_q} quad pts: "
          f"min={jac.min():.4e} max={jac.max():.4e} "
          f"(#<=0: {int((jac <= 0).sum())})", flush=True)
    print(f"          verdict: {'POSITIVE everywhere' if jac.min() > 0 else 'HAS NON-POSITIVE Jacobian'}",
          flush=True)


if __name__ == "__main__":
    main()
