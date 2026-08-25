"""Build a stellarator spline map from any GVEC-style flat-schema HDF5.

`data/W7-X.h5` has its own loader (`w7x_geometry.py`) because it stores 3-D
grids plus explicit `rho`/`theta`/`zeta` axes IN RADIANS. The GVEC exports --
`quasr_0009983.h5` (nfp=2), `quasr_0044970.h5` (nfp=3),
`w7x_vacuum_co_contra.h5` (nfp=5) and `gvec_nfp3_hegna_80cubed_clebsch.h5`
(nfp=3, 80^3) -- all share a DIFFERENT schema: flat `R`/`Z` of length
n_rho*n_theta*n_zeta, an `eval_points` (N,3) of (rho, theta, zeta) ALREADY
NORMALISED to [0,1], and `nfp`/`n_rho`/`n_theta`/`n_zeta` in attrs. One loader
covers all four.

Two things differ from the W7-X path and both bite:

1. **Handedness.** `mrx.mappings.stellarator_map` hardcodes
   `Y = -R sin(2 pi zeta/nfp)`, which matches `data/W7-X.h5` but MIRRORS raw
   GVEC data (standard cylindrical is `+sin`), giving `det DF < 0` -- and a
   negative Jacobian is not something the preconditioner should be handed.
   `scripts/debug/quasr_covcontra_verify.py` documents the same thing. The sign
   is therefore MEASURED here, not assumed: build, check `det DF` at interior
   points, flip if needed, and say which was used.

2. **Open vs closed periodic axes.** The quasr files sample theta,zeta on
   [0,1) (50 points, step 1/50) and need a wrap point appended; the hegna file
   samples [0,1] closed (80 points, step 1/79) and must NOT be padded. Detected
   from the spacing rather than hardcoded per file.

Interpolation goes through the same linear-RGI + Greville bridge the W7-X map
uses. That bridge has a known ~3.4% non-converging bias against a data-node
collocation spline (docs/research, "interpolatory spline vs linear RGI") -- it
matters for projecting FIELDS and does not matter here, where the only
requirement is a valid, well-conditioned shaped map to precondition on.

Run directly for the sanity checks:
    python scripts/debug/gvec_geometry.py --h5 data/quasr_0044970.h5 --ns 12,24,12
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

TWO_PI = 2.0 * np.pi

#: Geometry name -> h5, for the solve harnesses. Names are what `--geometry`
#: takes; every one of these is a genuinely shaped, non-axisymmetric map.
GVEC_GEOMETRIES = {
    "quasr9983": "data/quasr_0009983.h5",       # nfp=2, 50^3
    "quasr44970": "data/quasr_0044970.h5",      # nfp=3, 50^3
    "quasr65530": "data/quasr0065530_gvec_mrx_nr50_nt50_nz50.h5",  # nfp=4
    "quasr65575": "data/quasr0065575_gvec_mrx_nr50_nt50_nz50.h5",  # nfp=4
    "w7x-gvec": "data/w7x_vacuum_co_contra.h5",  # nfp=5, 50^3 -- a second,
    #   independent W7-X source; a cross-check on the `w7x` map, not a new device
    "w7x-ini": "data/w7x_ini_mrx.h5",           # nfp=5, 50^3, FINITE BETA
    #   (beta_volume_mean 4.2%): a different equilibrium of the same device, not
    #   a second vacuum source.  Its `axis_radial_index = 49` attribute is wrong
    #   -- the data has the axis at rho[0] (mean theta-extent 3.4e-3 there
    #   against 1.8 at rho=1), so no radial reversal is applied.
    "hegna": "data/gvec_nfp3_hegna_80cubed_clebsch.h5",   # nfp=3, 80^3
    #: The 8^3-ish quasr44970 baseline and its two interior perturbations.  All
    #: three share one 8x16x8 grid, so they are directly differenceable.
    "quasr44970-c": "data/quasr0044970_gvec_nr8_nt16_nz8.h5",     # nfp=3
    "pert-axis": "data/axis_pert_dR5e-05_dZ3.75e-05.h5",
    "pert-interior": "data/interior_pert_dR5e-05_dZ3.75e-05.h5",
}

#: nfp that the file gets WRONG, keyed by geometry name.
#:
#: The two perturbed files declare ``nfp = 2``.  Their R/Z data is
#: `quasr0044970_gvec_nr8_nt16_nz8.h5` shifted by exactly the amplitudes in
#: their own filenames (max|dR| 5.000e-05, max|dZ| 3.750e-05), and that device
#: is nfp=3; against quasr0009983, which their `dof_npy` and `perturb_source_h5`
#: attributes name, they differ by 0.15, i.e. a different machine.  Their
#: `geometry_source` and `template_h5` attributes both say quasr0044970 and
#: agree with the measurement, so `nfp = 2` travelled in with the stale
#: quasr0009983 paths.
#:
#: This is not cosmetic.  ``_map_with_sign`` builds
#: ``F = (R cos(2 pi zeta/nfp), +- R sin(2 pi zeta/nfp), Z)``, so nfp=2 would
#: wrap one field period of quasr44970 cross-sections through 180 degrees
#: instead of 120 -- a different domain, and a different iota, with a perfectly
#: healthy positive Jacobian to hide it.
GVEC_NFP_OVERRIDE = {"pert-axis": 3, "pert-interior": 3}


def _take(grid, axis, sl):
    idx = [slice(None)] * grid.ndim
    idx[axis] = sl
    return grid[tuple(idx)]


def _periodic_axis(vals, grid, axis, stride=1):
    """Normalise one PERIODIC axis to a half-open [0,1) sample, then wrap-pad.

    The files disagree about the last point: quasr samples half-open
    (0, 1/n, ..., (n-1)/n) while hegna samples closed (0, ..., 1), where the
    endpoint duplicates the first (verified: they agree to 9e-16). Normalising
    to half-open FIRST and padding unconditionally makes both paths identical
    and, more importantly, makes ``stride`` safe -- subsampling a closed axis
    directly would drop the endpoint for any stride that does not divide n-1
    (79 is prime, so every stride on hegna) and leave a short final cell.
    """
    v = np.asarray(vals, dtype=np.float64)
    if v.max() > 1.5:                      # radians -> [0,1]
        v = v / (v.max() + (v[1] - v[0]))
    step = v[1] - v[0]
    layout = "half-open"
    if abs(v[-1] - 1.0) < 0.5 * step:      # closed: drop the duplicate endpoint
        v, grid, layout = v[:-1], _take(grid, axis, slice(0, -1)), "closed"
    v, grid = v[::stride], _take(grid, axis, slice(None, None, stride))
    pad = np.concatenate([grid, _take(grid, axis, slice(0, 1))], axis=axis)
    return np.concatenate([v, [1.0]]), pad, layout


def _radial_axis(vals, grid, axis, stride=1):
    """Subsample the CLAMPED radial axis, always keeping the last point.

    rho is not periodic, and dropping rho=1 would turn every near-boundary
    evaluation into an extrapolation -- which is exactly where the natural-BC
    term this whole exercise is about lives.
    """
    v = np.asarray(vals, dtype=np.float64)
    keep = np.unique(np.r_[np.arange(0, len(v), stride), len(v) - 1])
    return v[keep], _take(grid, axis, keep)


def load_gvec_grids(h5_path, stride=1, nfp=None):
    """Return ``(axes, R_grid, Z_grid, nfp, layout)`` from a flat-schema file.

    ``nfp`` overrides the file's own attribute; see ``GVEC_NFP_OVERRIDE``.

    ``stride`` subsamples the DATA grid per axis (it stays a tensor grid).
    Default 1, i.e. all of it, and that is the right default: the data grid is
    read once and sampled only at the map's Greville points, so it is not a
    solve cost. MEASURED on this harness -- geometry build 88 s at 50^3 and
    136 s at 80^3, peak RSS 3.9 / 4.1 GB, against solve jobs whose nullspace
    step alone is 177 s and whose CG sweeps run for hours. Subsampling buys
    tens of seconds and costs fit accuracy: the RGI bridge is O(h^2), so
    stride 2 roughly quadruples the R/Z error that is currently 1e-4..5e-3.
    The knob exists for cheap smoke tests, not for production runs.
    (Not to be confused with ``quasr_covcontra_verify.py``'s ``EVAL_STRIDE``,
    which is load-bearing there because that script evaluates the map at every
    one of the 125k data points; this one never does.)
    """
    with h5py.File(h5_path, "r") as f:
        ep = np.asarray(f["eval_points"], dtype=np.float64)
        # Newer exports carry only `precomputed_*`; older ones carry both.
        nr, nt, nz = (int(f.attrs[k] if k in f.attrs else f.attrs[alt])
                      for k, alt in (("n_rho", "precomputed_nr"),
                                     ("n_theta", "precomputed_ntheta"),
                                     ("n_zeta", "precomputed_nzeta")))
        file_nfp = int(f.attrs["nfp"])
        R = np.asarray(f["R"], dtype=np.float64).reshape(nr, nt, nz)
        Z = np.asarray(f["Z"], dtype=np.float64).reshape(nr, nt, nz)
    nfp = file_nfp if nfp is None else int(nfp)

    r_raw = ep[:, 0].reshape(nr, nt, nz)[:, 0, 0]
    t_raw = ep[:, 1].reshape(nr, nt, nz)[0, :, 0]
    z_raw = ep[:, 2].reshape(nr, nt, nz)[0, 0, :]

    r_ax, R = _radial_axis(r_raw, R, 0, stride)
    _, Z = _radial_axis(r_raw, Z, 0, stride)
    t_ax, R, lay_t = _periodic_axis(t_raw, R, 1, stride)
    _, Z, _ = _periodic_axis(t_raw, Z, 1, stride)
    z_ax, R, lay_z = _periodic_axis(z_raw, R, 2, stride)
    _, Z, _ = _periodic_axis(z_raw, Z, 2, stride)
    return (r_ax, t_ax, z_ax), R, Z, nfp, (lay_t, lay_z)


def _rgi_fn(axes, grid):
    """jax RGI bridge; returns ``f(xi:(3,)) -> (1,)`` for greville collocation."""
    pts = tuple(jnp.asarray(a) for a in axes)
    interp = RegularGridInterpolator(
        pts, jnp.asarray(grid), method="linear",
        bounds_error=False, fill_value=None)   # extrapolate (rho < rho[0])

    def f(xi):
        return interp(xi.reshape(1, 3))[0:1]
    return f


def _map_with_sign(R_h, Z_h, nfp, sign):
    a = TWO_PI / nfp

    def F(x):
        ang = a * x[2]
        r = R_h(x)[0]
        return jnp.array([r * jnp.cos(ang), sign * r * jnp.sin(ang), Z_h(x)[0]])
    return F


def _det_DF(map_func, n=64, seed=0):
    """Sign-safe det(DF) sample, away from the axis and the r=1 knot.

    `rho=1` is the outer knot where `det DF` is exactly 0 for a spline map
    (docs/research, "spline map DF singular at r=1"), so sample strictly
    inside or every arm looks degenerate.
    """
    rng = np.random.default_rng(seed)
    xs = jnp.asarray(np.column_stack([
        rng.uniform(0.15, 0.95, n), rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n)]))
    dets = jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(map_func)(x)))(xs)
    return np.asarray(dets)


def build_gvec_map(h5_path, map_ns=(12, 24, 12), p=3, sign=None, stride=1,
                   nfp=None):
    """Build the stellarator map for one flat-schema file.

    ``sign`` is the toroidal handedness ``Y = sign * R sin(2 pi zeta/nfp)``.
    Left as ``None`` it is MEASURED: whichever sign makes ``det DF`` positive
    wins, and a file that is degenerate under both raises.
    """
    axes, R_grid, Z_grid, nfp, layout = load_gvec_grids(
        h5_path, stride=stride, nfp=nfp)
    R_fn, Z_fn = _rgi_fn(axes, R_grid), _rgi_fn(axes, Z_grid)

    map_seq = DeRhamSequence(map_ns, (p, p, p), 2 * p,
                             ("clamped", "periodic", "periodic"), polar=False)
    map_seq.evaluate_1d()
    R_h = DiscreteFunction(map_seq.interpolate(R_fn, 0), map_seq.basis_0, map_seq.e0)
    Z_h = DiscreteFunction(map_seq.interpolate(Z_fn, 0), map_seq.basis_0, map_seq.e0)

    tried = {}
    for s in ((sign,) if sign is not None else (1.0, -1.0)):
        F = _map_with_sign(R_h, Z_h, nfp, s)
        d = _det_DF(F)
        tried[s] = (float(d.min()), float(d.max()))
        if np.isfinite(d).all() and d.min() > 0:
            return F, {"R_h": R_h, "Z_h": Z_h, "R_fn": R_fn, "Z_fn": Z_fn,
                       "axes": axes, "map_seq": map_seq, "nfp": nfp,
                       "sign": s, "layout": layout, "det_range": tried[s],
                       "grid": R_grid.shape, "stride": stride}
    raise RuntimeError(f"{h5_path}: no handedness gives det DF > 0; "
                       f"sampled ranges {tried}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="data/quasr_0044970.h5")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample the DATA grid per axis; see load_gvec_grids "
                         "-- 1 (all of it) is the right default, this is for "
                         "smoke tests")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    mrx.MAP_BATCH_SIZE_INNER = int(cli.batch_size)

    axes, R, Z, nfp, layout = load_gvec_grids(cli.h5, stride=cli.stride)
    print(f"[load] {cli.h5} nfp={nfp} grid={R.shape} stride={cli.stride} "
          f"theta={layout[0]} zeta={layout[1]} "
          f"rho[{axes[0][0]:.4f},{axes[0][-1]:.4f}]")

    F, info = build_gvec_map(cli.h5, map_ns=ns, p=cli.p, stride=cli.stride)
    print(f"[map ] sign={info['sign']:+.0f} det DF in "
          f"[{info['det_range'][0]:.3e}, {info['det_range'][1]:.3e}]")

    rng = np.random.default_rng(0)
    xs = jnp.asarray(np.column_stack([rng.uniform(0.02, 0.98, 400),
                                      rng.uniform(0.0, 1.0, 400),
                                      rng.uniform(0.0, 1.0, 400)]))
    for nm, h, fn in (("R", info["R_h"], info["R_fn"]),
                      ("Z", info["Z_h"], info["Z_fn"])):
        e = np.asarray(jnp.abs(jax.vmap(lambda x, h=h: h(x)[0])(xs)
                               - jax.vmap(lambda x, f=fn: f(x)[0])(xs)))
        print(f"[fit ] {nm}: max={e.max():.2e} rms={np.sqrt((e**2).mean()):.2e}")

    seq = DeRhamSequence(ns, (cli.p,) * 3, 2 * cli.p,
                         ("clamped", "periodic", "periodic"), polar=True,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.set_map(F)
    jac = np.asarray(seq.geometry.jacobian_j)
    print(f"[quad] J over quadrature points: min={jac.min():.3e} "
          f"max={jac.max():.3e} finite={bool(np.isfinite(jac).all())}")
    if not np.isfinite(jac).all() or jac.min() <= 0:
        raise SystemExit("DEGENERATE geometry")
    print("[ok  ] usable as a solve geometry")


if __name__ == "__main__":
    main()
