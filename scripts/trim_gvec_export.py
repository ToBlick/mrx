"""Reduce a GVEC flat-schema export to what MRX reads, optionally on a new grid.

MRX reads ``eval_points``, ``R``, ``Z``, ``pressure`` and
``clebsch/{dPhi_dr, dchi_dr, LA}`` plus the attributes ``n_rho``,
``n_theta``, ``n_zeta``, ``nfp`` (``docs/source/concepts/gvec_mrx_interface.md``); a full
export also carries ``B``, ``beta``, ``Phi``, ``chi``, the lambda
derivatives and the gradients, which is most of its size. This writes only
the read set, gzip-compressed, which the readers decompress transparently.

With ``--n-rho/--n-theta/--n-zeta`` every field is first interpolated by the
same data-node spline fit ``load_clebsch`` uses for lambda
(:func:`mrx.gvec.fit_scalar_spline`) and then sampled on the new grid.
``--radial edge`` refines the radial sample toward the edge,
``rho = (3u - u^2) / 2`` for uniform ``u`` (cells 3x smaller at the wall than
at the axis); the data-placed knots of the fit make that a valid input.

    python -u scripts/trim_gvec_export.py data/w7x_fmm002_clebsch_mrx.h5 out.h5
    python -u scripts/trim_gvec_export.py data/w7x_fmm002_clebsch_mrx.h5 out20.h5 \
        --n-rho 20 --n-theta 20 --n-zeta 20 --radial edge
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

READ_SET = ("R", "Z", "pressure", "clebsch/dPhi_dr", "clebsch/dchi_dr", "clebsch/LA")
DOC_ATTRS = ("clebsch_contract", "gvec_source", "angle_units", "zeta_convention",
             "radial_label")


def radial_axis(n, kind):
    """``n`` radial samples in ``(0, 1]``: the first point ``0.1 / (n-1)`` off
    the axis as GVEC exports it, the last at 1."""
    u = np.arange(n, dtype=np.float64) / (n - 1)
    rho = u if kind == "uniform" else (3.0 * u - u ** 2) / 2.0
    rho[0] = 0.1 / (n - 1)
    return rho


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n-rho", type=int, default=None)
    ap.add_argument("--n-theta", type=int, default=None)
    ap.add_argument("--n-zeta", type=int, default=None)
    ap.add_argument("--radial", default="uniform", choices=("uniform", "edge"))
    cli = ap.parse_args()

    if cli.src.endswith(".dat"):
        raise SystemExit(f"{cli.src} is a GVEC state file; the solvers read it directly "
                         "(closed form, mrx.gvec), nothing to trim")
    with h5py.File(cli.src, "r") as f:
        shape = tuple(int(f.attrs[k]) for k in ("n_rho", "n_theta", "n_zeta"))
        nfp = int(f.attrs["nfp"])
        ep = np.asarray(f["eval_points"], dtype=np.float64)
        fields = {k: np.asarray(f[k], dtype=np.float64).reshape(shape) for k in READ_SET}
        doc = {k: f.attrs[k] for k in DOC_ATTRS if k in f.attrs}
    axes = [np.unique(ep[:, i]) for i in range(3)]
    if tuple(len(a) for a in axes) != shape:
        raise SystemExit(f"{cli.src}: eval_points axes do not match {shape}")

    new_shape = tuple(n or old for n, old in zip((cli.n_rho, cli.n_theta, cli.n_zeta), shape))
    if new_shape != shape or cli.radial != "uniform":
        from mrx.gvec import fit_scalar_spline
        import jax
        import jax.numpy as jnp
        new_axes = [radial_axis(new_shape[0], cli.radial),
                    np.arange(new_shape[1]) / new_shape[1],
                    np.arange(new_shape[2]) / new_shape[2]]
        pts = np.stack(np.meshgrid(*new_axes, indexing="ij"), axis=-1).reshape(-1, 3)
        types = ("clamped", "periodic", "periodic")
        for k in READ_SET:
            fit = fit_scalar_spline(axes, fields[k], types)
            fields[k] = np.asarray(jax.jit(jax.vmap(fit))(jnp.asarray(pts))).reshape(new_shape)
            print(f"  {k}: resampled {shape} -> {new_shape}", flush=True)
        axes = new_axes
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    with h5py.File(cli.dst, "w") as h:
        h.attrs["n_rho"], h.attrs["n_theta"], h.attrs["n_zeta"] = new_shape
        h.attrs["nfp"] = nfp
        h.attrs["source"] = os.path.basename(cli.src)
        h.attrs["trimmed_by"] = "scripts/trim_gvec_export.py"
        h.attrs["radial_sampling"] = cli.radial
        h.attrs.update(doc)
        h.create_dataset("eval_points", data=pts, compression="gzip", shuffle=True)
        for k, v in fields.items():
            h.create_dataset(k, data=np.ascontiguousarray(v.ravel()),
                             compression="gzip", compression_opts=9, shuffle=True)
    print(f"wrote {cli.dst}: grid {new_shape}, {os.path.getsize(cli.dst) / 1e6:.2f} MB",
          flush=True)


if __name__ == "__main__":
    main()
