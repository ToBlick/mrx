# %%
import argparse
import os
import random
import string
import time

# Removed desc import
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
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
    """
    Load DESC equilibrium from HDF5 file.

    This function now checks if the HDF5 file contains precomputed R, Z, B data, to avoid relative import of DESC.
    If that data is not available, it falls back to using the DESC library (will fail if DESC is not installed).


    Parameters
    ----------
    path : str
        Path to DESC HDF5 file.
    map_seq : DeRhamSequence
        DeRham sequence for mapping interpolation.
    nr, ntheta, nzeta : int, optional
        Grid resolution. 

    Returns
    -------
    dict
        Dictionary containing 'X1', 'X2', 'Phi', 'nfp', 'eval_points', 'R', 'Z', 'B_vals',
        and 'map_interpolation_residual'.
    """
    if nr is None:  # Defaults to map_seq.Q dimensions
        nr = map_seq.Q.nx
    if ntheta is None:
        ntheta = map_seq.Q.ny
    if nzeta is None:
        nzeta = map_seq.Q.nz

    # Read precomputed data from HDF5 file
    with h5py.File(path, 'r') as f:
        # Check if precomputed data exists
        if 'R' in f and 'Z' in f and 'B' in f and 'eval_points' in f:
            R = jnp.array(f['R'][:])
            Z = jnp.array(f['Z'][:])
            B_vals = jnp.array(f['B'][:])
            eval_points = jnp.array(f['eval_points'][:])

            # Try to get nfp
            if 'nfp' in f.attrs:
                nfp = int(f.attrs['nfp'])
            else:
                raise KeyError(
                    "nfp not found in HDF5 file attributes. "
                )

            # Use the precomputed data
            pts = eval_points
        else:
            raise KeyError(
                "Precomputed R, Z, B data not found in HDF5 file. "
            )

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
        'R': R,  # adding R,Z,B
        'Z': Z,
        'B_vals': B_vals,
        'map_interpolation_residual': mapresid
    }

    return desc_import


def interpolate_map_from_points(x, R, Z, nfp, ns=(6, 6, 6), ps=(3, 3, 3), quad_order=3, flip_zeta=False):
    """
    Given evaluations of a function R(x), Z(x) at some points x, interpolate

    Parameters
    ----------
    x : jnp.ndarray
        Points at which R, Z are evaluated (n_pts, 3).
    R : jnp.ndarray
        R evaluations at points (n_pts,).
    Z : jnp.ndarray
        Z evaluations at points (n_pts,).
    nfp : int
        Number of field periods.

    Returns
    -------
    """
    # Set up DeRham sequence for interpolation
    map_seq = DeRhamSequence(ns, ps, quad_order, ("clamped", "periodic", "periodic"),
                             lambda x: x, polar=False, dirichlet=False)

    # Set up the interpolation problem:
    # ∑ c_ki Λ0[i](x_j) ≈ Xk(x_j) ∀j
    def body_fun(_, i):
        # Evaluate Λ0[i](x) for all points
        return None, jax.vmap(lambda x: map_seq.basis_0[i](x)[0])(x)

    _, M = jax.lax.scan(body_fun, None, map_seq.basis_0.ns)  # Λ0[i](x_j)

    y = jnp.stack([R, Z], axis=1)  # X_α(x'_j)
    c, resid, _, _ = jnp.linalg.lstsq(M.T, y, rcond=None)

    R_dof = c[:, 0]
    Z_dof = c[:, 1]

    X1_h = DiscreteFunction(R_dof, map_seq.basis_0, map_seq.e0)
    X2_h = DiscreteFunction(Z_dof, map_seq.basis_0, map_seq.e0)
    return stellarator_map(X1_h, X2_h, nfp=nfp, flip_zeta=flip_zeta), R_dof, Z_dof, resid


def interpolate_B(x, B, seq, exclude_axis_tol=1e-3):
    """
    Interpolate B-field onto FEM basis given evaluations at points x.
    """

    # valid interpolation points (avoid axis and exact boundary)
    valid_pts = (x[:, 0] > exclude_axis_tol) & (
        x[:, 0] < 1 - exclude_axis_tol)

    def Λ2_phys(i, x):
        return Pushforward(lambda x: seq.Lambda_2[i](x), seq.F, 2)(x)

    def body_fun(_, i):
        # Evaluate Λ2_phys(i, x) for all points (vectorized over x)
        return None, jax.vmap(lambda x: Λ2_phys(i, x))(x[valid_pts])

    _, M = jax.lax.scan(body_fun, None, seq.Lambda_2.ns)
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
# %%
# %%
# %%
# %%
