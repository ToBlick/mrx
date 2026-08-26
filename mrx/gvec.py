"""Stellarator spline maps and Clebsch data from GVEC-derived HDF5 files.

Two file schemas are read here:

* The GVEC flat schema (``quasr_*.h5``, ``w7x_*_mrx.h5``, the hegna export):
  flat ``R``/``Z`` of length ``n_rho*n_theta*n_zeta``, ``eval_points`` of
  shape ``(N, 3)`` in ``(rho, theta, zeta)`` already normalised to ``[0, 1]``,
  and ``nfp``/``n_rho``/``n_theta``/``n_zeta`` in the attributes. Some of
  these files also carry ``clebsch/dPhi_dr``, ``clebsch/dchi_dr``,
  ``clebsch/LA`` and ``pressure``; :func:`load_clebsch` reads those.
* ``data/W7-X.h5``: 3-D ``R``/``Z`` grids with explicit ``rho``/``theta``/
  ``zeta`` axes in radians. :func:`build_w7x_map` reads it.

Both routes interpolate the grid with a linear RegularGridInterpolator,
Greville-interpolate the result as a spline 0-form on a separate map sequence
and wrap the two scalars in a cylindrical map. The linear bridge has a known
non-converging ~3.4% bias against a data-node collocation spline; that matters
for projecting fields and not for a map to precondition on.

Two traps that the flat schema carries:

* **Handedness.** ``mrx.mappings.stellarator_map`` uses
  ``Y = -R sin(2 pi zeta/nfp)``, which matches ``W7-X.h5`` and mirrors raw GVEC
  data (``det DF < 0``). :func:`build_gvec_map` measures the sign instead of
  assuming it.
* **Open versus closed periodic axes.** The quasr files sample the angles on
  ``[0, 1)`` and need a wrap point; the hegna file samples ``[0, 1]`` closed
  and must not be padded. Both are detected from the spacing.

Data paths are relative to ``MRX_DATA`` (default ``data``).
"""
from __future__ import annotations

import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DifferentialForm, DiscreteFunction
from mrx.mappings import stellarator_map
from mrx.projectors import _solve_tensor_collocation_axis

TWO_PI = 2.0 * np.pi

#: Directory the file names below are relative to.
DATA_DIR = os.environ.get("MRX_DATA", "data")

#: Geometry name -> flat-schema file. Every entry is a genuinely shaped,
#: non-axisymmetric map.
GVEC_GEOMETRIES = {
    "quasr9983": "quasr_0009983.h5",       # nfp=2, 50^3
    "quasr44970": "quasr_0044970.h5",      # nfp=3, 50^3
    "quasr65530": "quasr0065530_gvec_mrx_nr50_nt50_nz50.h5",  # nfp=4
    "quasr65575": "quasr0065575_gvec_mrx_nr50_nt50_nz50.h5",  # nfp=4
    # A second, independent W7-X vacuum source; a cross-check on the `w7x`
    # map of build_w7x_map, not a new device.
    "w7x-gvec": "w7x_vacuum_co_contra.h5",  # nfp=5, 50^3
    # Finite beta (beta_volume_mean 4.2%). Its `axis_radial_index = 49`
    # attribute is wrong: the axis is at rho[0], so no radial reversal.
    "w7x-ini": "w7x_ini_mrx.h5",           # nfp=5, 50^3
    "hegna": "gvec_nfp3_hegna_80cubed_clebsch.h5",   # nfp=3, 80^3
    # Finite-beta W7-X exports carrying the clebsch/* ingredients, nfp=5,
    # 50^3, angles sampled HALF-OPEN (0 .. 0.98). Both carry the same wrong
    # `axis_radial_index = 49`.
    "w7x-ini-clebsch": "w7x_ini_00000000_clebsch_mrx.h5",   # beta_mean 5.8%
    "w7x-fmm002": "w7x_fmm002_clebsch_mrx.h5",              # beta_mean 1.8%
    # State_0000_00020000.dat of the same GVEC run as w7x-ini-clebsch
    # (which is the iteration-0 initial guess): the converged equilibrium,
    # and the one to prefer when comparing against the file's pressure.
    "w7x-ini-conv": "w7x_ini_conv_mrx.h5",                  # beta_mean 6.6%
    # The 8x16x8 quasr44970 baseline and its two interior perturbations;
    # all three share one grid and are directly differenceable.
    "quasr44970-c": "quasr0044970_gvec_nr8_nt16_nz8.h5",     # nfp=3
    "pert-axis": "axis_pert_dR5e-05_dZ3.75e-05.h5",
    "pert-interior": "interior_pert_dR5e-05_dZ3.75e-05.h5",
}

#: nfp that the file gets WRONG, keyed by geometry name. The two perturbed
#: files declare ``nfp = 2``; their R/Z data is quasr44970 (nfp=3) shifted by
#: the amplitudes in their file names. nfp enters the map as
#: ``F = (R cos(2 pi zeta/nfp), +-R sin(2 pi zeta/nfp), Z)``, so the wrong
#: value wraps one field period through 180 instead of 120 degrees with a
#: healthy Jacobian to hide it.
GVEC_NFP_OVERRIDE = {"pert-axis": 3, "pert-interior": 3}


def gvec_path(geometry: str) -> str:
    """Return the file path of a named flat-schema geometry."""
    return os.path.join(DATA_DIR, GVEC_GEOMETRIES[geometry])


def _take(grid, axis, sl):
    idx = [slice(None)] * grid.ndim
    idx[axis] = sl
    return grid[tuple(idx)]


def _periodic_axis(vals, grid, axis, stride=1):
    """Normalise one periodic axis to a half-open [0,1) sample, then wrap-pad.

    quasr samples half-open (0, 1/n, ..., (n-1)/n); hegna samples closed
    (0, ..., 1) with the endpoint duplicating the first point. Normalising to
    half-open first and padding unconditionally makes both paths identical and
    makes ``stride`` safe on the closed layout (79 is prime, so every stride
    would otherwise leave a short final cell).
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
    """Subsample the clamped radial axis, always keeping the last point.

    Dropping rho=1 would turn every near-boundary evaluation into an
    extrapolation, which is where the natural-BC term lives.
    """
    v = np.asarray(vals, dtype=np.float64)
    keep = np.unique(np.r_[np.arange(0, len(v), stride), len(v) - 1])
    return v[keep], _take(grid, axis, keep)


def load_gvec_grids(h5_path, stride=1, nfp=None):
    """Return ``(axes, R_grid, Z_grid, nfp, layout)`` from a flat-schema file.

    ``nfp`` overrides the file's attribute (see :data:`GVEC_NFP_OVERRIDE`).
    ``stride`` subsamples the data grid per axis. Default 1 is the right
    default: the grid is read once and sampled only at the map's Greville
    points (geometry build 88 s at 50^3, 136 s at 80^3, peak RSS ~4 GB), and
    stride 2 roughly quadruples the O(h^2) fit error. The knob exists for
    smoke tests.
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
    """Linear grid bridge; returns ``f(xi:(3,)) -> (1,)`` for Greville collocation."""
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
    """Sample det(DF) away from the axis and from the r=1 knot, where a
    spline map has det DF = 0 exactly."""
    rng = np.random.default_rng(seed)
    xs = jnp.asarray(np.column_stack([
        rng.uniform(0.15, 0.95, n), rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n)]))
    dets = jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(map_func)(x)))(xs)
    return np.asarray(dets)


def _spline_scalars(axes, R_grid, Z_grid, map_ns, p):
    R_fn, Z_fn = _rgi_fn(axes, R_grid), _rgi_fn(axes, Z_grid)
    map_seq = DeRhamSequence(map_ns, (p, p, p), 2 * p,
                             ("clamped", "periodic", "periodic"), polar=False)
    map_seq.evaluate_1d()
    R_h = DiscreteFunction(map_seq.interpolate(R_fn, 0), map_seq.basis_0, map_seq.e0)
    Z_h = DiscreteFunction(map_seq.interpolate(Z_fn, 0), map_seq.basis_0, map_seq.e0)
    return R_h, Z_h, R_fn, Z_fn, map_seq


def build_gvec_map(h5_path, map_ns=(12, 24, 12), p=3, sign=None, stride=1,
                   nfp=None):
    """Build the stellarator map of one flat-schema file.

    Returns ``(F, info)``. ``sign`` is the toroidal handedness
    ``Y = sign * R sin(2 pi zeta/nfp)``; left ``None`` it is measured, and a
    file that is degenerate under both signs raises.
    """
    axes, R_grid, Z_grid, nfp, layout = load_gvec_grids(
        h5_path, stride=stride, nfp=nfp)
    R_h, Z_h, R_fn, Z_fn, map_seq = _spline_scalars(axes, R_grid, Z_grid, map_ns, p)

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


# ---------------------------------------------------------------------------
# W7-X.h5: 3-D grids with radian axes
# ---------------------------------------------------------------------------

NFP_W7X = 5
W7X_FILE = "W7-X.h5"


def load_w7x_grids(h5_path=None):
    """Return logical axes in [0,1] and the periodic-padded R, Z grids."""
    h5_path = os.path.join(DATA_DIR, W7X_FILE) if h5_path is None else h5_path
    with h5py.File(h5_path, "r") as f:
        rho = np.asarray(f["rho"], dtype=np.float64)        # [0,1]
        theta = np.asarray(f["theta"], dtype=np.float64)    # [0,2pi)
        zeta = np.asarray(f["zeta"], dtype=np.float64)      # [0,2pi/nfp)
        R = np.asarray(f["R"], dtype=np.float64)            # (nr,nt,nz)
        Z = np.asarray(f["Z"], dtype=np.float64)
    t_ax = np.concatenate([theta / TWO_PI, [1.0]])
    z_ax = np.concatenate([zeta * NFP_W7X / TWO_PI, [1.0]])

    def _pad(grid):
        grid = np.concatenate([grid, grid[:, :1, :]], axis=1)   # theta wrap
        grid = np.concatenate([grid, grid[:, :, :1]], axis=2)   # zeta wrap
        return grid

    return (rho, t_ax, z_ax), _pad(R), _pad(Z)


def build_w7x_map(map_ns=(12, 24, 24), p=3, h5_path=None):
    """Build the W7-X map from ``W7-X.h5``; returns ``(F, info)``."""
    axes, R_grid, Z_grid = load_w7x_grids(h5_path)
    R_h, Z_h, R_fn, Z_fn, map_seq = _spline_scalars(axes, R_grid, Z_grid, map_ns, p)
    map_func = stellarator_map(R_h, Z_h, nfp=NFP_W7X)
    return map_func, {"R_h": R_h, "Z_h": Z_h, "R_fn": R_fn, "Z_fn": Z_fn,
                      "axes": axes, "map_seq": map_seq, "nfp": NFP_W7X}


# ---------------------------------------------------------------------------
# Clebsch ingredients: the equilibrium field as three scalars
# ---------------------------------------------------------------------------

def fit_scalar_spline(axes, values, types, degree=3):
    """Interpolatory tensor-product spline through grid data, as a callable.

    ``n_basis = n_data`` per axis, one square collocation solve each (the fit
    ``mrx.io.load_grid_field`` step 1 does), kept as a function so its
    derivatives can be taken exactly. Evaluation is three 1-D contractions:
    the hegna fit has ~5e5 basis functions, and a ``DiscreteFunction`` would
    evaluate all of them per point.
    """
    n = tuple(len(a) for a in axes)
    fit = DifferentialForm(0, n, (degree,) * 3, types)
    C = jnp.asarray(values).reshape(n)
    for a, (basis, x) in enumerate(zip(fit.Λ, axes)):
        C = _solve_tensor_collocation_axis(
            basis.collocation_matrix(jnp.asarray(x)), C, axis=a)

    br, bt, bz = fit.Λ

    def evaluate(x):
        vr = jax.vmap(lambda i: br(x[0], i))(br.ns)
        vt = jax.vmap(lambda i: bt(x[1], i))(bt.ns)
        vz = jax.vmap(lambda i: bz(x[2], i))(bz.ns)
        return jnp.einsum('ijk,i,j,k->', C, vr, vt, vz)

    return evaluate


def load_clebsch(path, types=("clamped", "periodic", "periodic")):
    """Read the radial profiles, a lambda callable and p(rho) from a file.

    The reference 2-form components of ``mrx.initial_conditions`` are exactly
    GVEC's ``sqrt(g) B^i``, verified against the file's own B:
    ``sqrt(g) B^theta = dchi_dr - dPhi_dr dLA_dz`` and
    ``sqrt(g) B^zeta = dPhi_dr (1 + dLA_dt)``, in GVEC's units (derivatives
    with respect to radian angles). The caller converts with
    ``Phi' = 2 pi dPhi_dr``, ``iota = dchi_dr / (nfp dPhi_dr)`` and
    ``lambda = LA / 2 pi``.

    lambda is fitted as the scalar and differentiated, never read as two
    derivatives: ``div B = 0`` rests on the mixed partials cancelling, which
    holds only when both come from one interpolant. The duplicate endpoint of
    a closed periodic sample is dropped before the fit; whether an axis is
    closed is decided from the data (a duplicate agrees to round-off, the
    half-open files differ by ~7e-2), and the decision is returned as
    ``closed_axes``.

    Returns a dict with ``nfp``, ``rho``, ``dPhi``, ``dchi``, ``p`` (surface
    means, arrays on ``rho``), ``iota_spread`` (max angular departure of
    dchi/dPhi from a flux function at mid-radius) and ``lam_h``.
    """
    with h5py.File(path, "r") as h:
        shape = (int(h.attrs["n_rho"]), int(h.attrs["n_theta"]),
                 int(h.attrs["n_zeta"]))
        c = h["clebsch"]
        dPhi = np.asarray(c["dPhi_dr"]).reshape(shape)
        dchi = np.asarray(c["dchi_dr"]).reshape(shape)
        LA = np.asarray(c["LA"]).reshape(shape)
        pres = np.asarray(h["pressure"]).reshape(shape)
        ep = np.asarray(h["eval_points"])
        nfp = int(h.attrs["nfp"])

    axes = [np.unique(ep[:, i]) for i in range(3)]
    if not all(len(a) == n for a, n in zip(axes, shape)):
        raise RuntimeError(f"eval_points axes {[len(a) for a in axes]} do not "
                           f"match declared shape {shape}")

    nr = shape[0]
    prof_dPhi = dPhi.mean(axis=(1, 2))
    prof_dchi = dchi.mean(axis=(1, 2))
    prof_p = pres.mean(axis=(1, 2))
    spread = float(np.nanmax(
        np.abs(dchi / dPhi - (prof_dchi / prof_dPhi)[:, None, None])
        [nr // 4:3 * nr // 4]))

    fit_axes, LA_fit, closed = list(axes), LA, []
    span = float(np.abs(LA).max()) or 1.0
    for a, kind in enumerate(types):
        if kind != 'periodic':
            continue
        gap = float(np.abs(np.take(LA_fit, 0, axis=a)
                           - np.take(LA_fit, -1, axis=a)).max())
        if gap <= 1e-8 * span:
            fit_axes[a] = fit_axes[a][:-1]
            LA_fit = np.take(LA_fit, np.arange(len(fit_axes[a])), axis=a)
            closed.append(a)
    lam_h = fit_scalar_spline(fit_axes, LA_fit, types)

    return dict(nfp=nfp, rho=axes[0], dPhi=prof_dPhi, dchi=prof_dchi,
                p=prof_p, iota_spread=spread, lam_h=lam_h, closed_axes=closed)
