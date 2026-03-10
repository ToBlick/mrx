# %%
import argparse
import os
import random
import string
import time

import h5py
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.differential_forms import Pushforward



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


def interpolate_scalar_function(x, f_vals, seq, weights = None, rcond=None):
    """
    Least-squares interpolation of a scalar field onto a 0-form FEM basis.

    Given evaluations *f_vals* of a scalar function at points *x*, find the
    coefficient vector ``c`` such that ``∑_i c_i Λ0[i](x_j) ≈ f(x_j)``
    in the least-squares sense.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points, shape ``(n_pts, 3)``.
    f_vals : jnp.ndarray
        Scalar values at evaluation points, shape ``(n_pts,)``.
    seq : DeRhamSequence
        Pre-built DeRham sequence to use for the basis.
    weights : jnp.ndarray, optional
        Weights for the least-squares problem, shape ``(n_pts,)``.
    rcond : float, optional
        Relative condition number cutoff for small singular values in lstsq.

    Returns
    -------
    dict
        Dictionary containing 'dof' (the coefficient vector) and lstsq diagnostics.
    """
    if weights is None:
        weights = jnp.ones_like(f_vals)
    
    M = mrx.double_map(
        lambda i, pt: seq.basis_0[i](pt)[0],
        seq.basis_0.ns, x,
    )  # shape (n_dof, n_pts)
    
    A = jnp.einsum('ij,j,jk->ik', M, weights, M.T)  # shape (n_dof, n_dof)
    rhs = jnp.einsum('ij,j,j->i', M, weights, f_vals)  # shape (n_dof,)

    c, residual, rank, s = jnp.linalg.lstsq(A, rhs, rcond=rcond)
    return {
        "dof" : c,
        "residual" : residual,
        "rank" : rank,
        "singular_values" : s,
    }


def interpolate_B(x, B_vals, seq, weights = None, rcond=None):
    """
    Interpolate B-field onto FEM basis given evaluations at points x.
    """

    def Λ2_phys(i, x):
        return Pushforward(lambda x: seq.Lambda_2[i](x), seq.F, 2)(x)

    M = mrx.double_map(Λ2_phys, seq.Lambda_2.ns, x)
    M = jnp.einsum('il,ljk->ijk', seq.E2, M)    # Λ2[i](x_hat_j)_k
    y = B[valid_pts].reshape(-1, 3)              # B(x'_j)_k

    # Solve least squares interpolation:
    # ∑ c_ik Λ2[i](x_j) ≈ B_k(x_j) ∀j,k
    # i.e. M @ C ≈ B where M is (num_basis, num_pts, 3) and B is (num_pts, 3)

    A = M.reshape(M.shape[0], -1).T         # reshape to (num_pts*3, num_basis)
    b = y.ravel()                           # reshape to (num_pts*3,)

    # Solve least squares
    B_dof, residuals, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
    return B_dof, residuals