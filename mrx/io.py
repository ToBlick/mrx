import argparse
import os
import random
import string
import time

import desc
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.mappings import stellarator_map


def parse_args() -> dict:
    """
    Parse key=value arguments from command line.

    Returns
    -------
    kwargs : dict
        Dictionary of key=value arguments.
    """
    parser = argparse.ArgumentParser(
        description="Example script with key=value inputs")

    # Catch all unrecognized arguments as strings
    parser.add_argument('kv', nargs='*', help="key=value arguments")

    args = parser.parse_args()

    # Convert 'key=value' strings to dictionary
    kwargs = {}
    for kv in args.kv:
        if '=' not in kv:
            # Ignore arguments that don't match key=value format
            # This allows scripts to run with default values even when called
            # with incompatible arguments (e.g., in CI fallback scenarios)
            continue
        key, value = kv.split('=', 1)
        # try to convert to int/float/bool
        if value.isdigit():
            value = int(value)
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        kwargs[key] = value
    return kwargs


def unique_id(n: int) -> str:
    """
    Create a unique alphanumeric ID with low collision probability.

    Parameters
    ----------
    n : int
        Length of the ID.
    """
    chars = string.ascii_letters + string.digits  # 64 choices
    id_str = "".join(random.choice(chars) for _ in range(n))
    return id_str


def epoch_time(decimals=0) -> int:
    """
    Get the current epoch time in seconds.

    Parameters
    ----------
    decimals (int): Number of decimal places to round to. Default is 0.

    Returns
    -------
    time : int
        Current epoch time in seconds.
    """
    return int(time.time() * (10 ** (decimals)))


def load_sweep(
    path,
    reference_file,
    QOI,
    sweep_params
):
    """
    Load force traces, iter counts, and configs from HDF5 files in `path`
    that differ from `reference_file` only in `sweep_params` (plus 'run_name').

    Parameters
    ----------
    path : str
        Directory containing .h5 files.
    reference_file : str
        Path to the reference .h5 file.
    QOI : str
        Key of the quantity of interest to load.
    sweep_params : list[str]
        List of config keys that are allowed to differ (in addition to 'run_name').

    Returns
    -------
    cfgs : list[dict]
        List of configurations.
    forces : list[np.ndarray]
        List of force traces.
    iter_counts : list[np.ndarray]
        List of iteration counts.
    """

    # --- load reference config ---
    with h5py.File(reference_file, "r") as f:
        ref_cfg = {k: v for k, v in f["config"].attrs.items()}
        ref_cfg = {k: v.decode() if isinstance(v, bytes) else v
                   for k, v in ref_cfg.items()}

    # extend sweep params with run_name
    allowed_params = set(sweep_params) | {"run_name"}

    cfgs, forces, iter_counts = [], [], []

    for fname in os.listdir(path):
        if not fname.endswith(".h5"):
            continue

        full_path = os.path.join(path, fname)

        try:
            with h5py.File(full_path, "r") as f:
                force_trace = f[QOI][:]

                cfg = {k: v for k, v in f["config"].attrs.items()}
                cfg = {k: v.decode() if isinstance(v, bytes) else v
                       for k, v in cfg.items()}

                iter_count = np.arange(0, cfg["maxit"], cfg["save_every"])
        except Exception as e:
            print(f"Could not open {fname}: {e}")
            continue

        # --- check if cfg matches reference except allowed_params ---
        diffs = {k: (ref_cfg.get(k), v) for k, v in cfg.items()
                 if ref_cfg.get(k) != v}

        unexpected_diffs = {k: v for k,
                            v in diffs.items() if k not in allowed_params}

        if not unexpected_diffs:  # all diffs allowed
            cfgs.append(cfg)
            forces.append(force_trace)
            iter_counts.append(iter_count)
            print(f"Loaded {fname}")
        else:
            print(
                f"Skipped {fname} (unexpected diffs: {list(unexpected_diffs.keys())})")

    return cfgs, forces, iter_counts


def load_desc(path, map_seq, nr=None, ntheta=None, nzeta=None):
    eq_fam = desc.io.load(path)
    eq = eq_fam[-1]
    nfp = eq.NFP
    if nr is None:
        nr = map_seq.Q.nx
    if ntheta is None:
        ntheta = map_seq.Q.ny
    if nzeta is None:
        nzeta = map_seq.Q.nz
    grid = desc.grid.LinearGrid(L=nr, M=ntheta, N=nzeta, NFP=nfp)
    pts = grid.nodes.copy()
    pts[:, 1] /= 2 * jnp.pi
    pts[:, 2] /= 2 * jnp.pi / nfp
    vals = eq.compute(["R", "Z", "B"], grid=grid, basis="xyz")

    R = vals["R"]
    Z = vals["Z"]
    B_vals = vals["B"]
    eval_points = pts

    def body_fun(_, i):
        # Evaluate Λ0[i](x) for all points
        return None, jax.vmap(lambda x: map_seq.Lambda_0[i](x)[0])(pts)

    _, M = jax.lax.scan(body_fun, None, map_seq.Lambda_0.ns)  # Λ0[i](x_hat_j)
    y = jnp.stack([R.ravel(), Z.ravel()], axis=1)  # X_α(x'_j)
    c, mapresid, _, _ = jnp.linalg.lstsq(M.T, y, rcond=None)

    X1_h = DiscreteFunction(c[:, 0], map_seq.Lambda_0, map_seq.E0)
    X2_h = DiscreteFunction(c[:, 1], map_seq.Lambda_0, map_seq.E0)

    Phi, X1, X2 = stellarator_map(
        X1_h, X2_h, nfp=nfp, flip_zeta=True), X1_h, X2_h

    desc_import = {
        'X1': X1,
        'X2': X2,
        'Phi': Phi,
        'nfp': nfp,
        'eval_points': eval_points,
        'B_vals': B_vals,
        'map_interpolation_residual': mapresid
    }

    return desc_import
