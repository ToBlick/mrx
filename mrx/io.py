import argparse
import os
import random
import string
import time
from typing import Literal

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

import mrx


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


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse a comma-separated string of integers, e.g. ``'1,2,3'`` → ``(1, 2, 3)``."""
    return tuple(int(s.strip()) for s in text.split(",") if s.strip())


def parse_ns(text: str) -> tuple[int, int, int]:
    """Parse a resolution triple ``'nr,nt,nz'`` into a 3-tuple of ints."""
    parts = parse_int_list(text)
    if len(parts) != 3:
        raise ValueError(f"Expected ns as 'nr,nt,nz', got {text!r}")
    return parts  # type: ignore[return-value]


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


def interpolate_scalar_function(x, f_vals, seq, weights=None, rcond=None):
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
        "dof": c,
        "residual": residual,
        "rank": rank,
        "singular_values": s,
    }


def project_sampled_field(
    axes,
    values,
    seq,
    k: Literal[0, 1, 2, 3],
    dirichlet: bool = True,
    reference_domain: bool = False,
):
    """L2-project a field sampled on a regular grid onto a k-form FEM basis.

    The sampled data is interpolated at the quadrature points in one
    vectorized call, the coordinate pullback is applied via precomputed
    quantities on *seq*, and the result is integrated against the TP basis
    directly — avoiding any point-by-point ``lax.map``.

    Parameters
    ----------
    axes : tuple of 1-D arrays
        Grid axes ``(x1, x2, x3)`` spanning the logical domain, each of
        shape ``(n1,)``, ``(n2,)``, ``(n3,)``.
    values : jnp.ndarray
        Field values on the grid.  For scalar forms (k=0, k=3) the shape
        must be ``(n1*n2*n3,)`` or ``(n1, n2, n3)``.  For vector forms
        (k=1, k=2) the shape must be ``(n1*n2*n3, 3)`` or ``(n1, n2, n3, 3)``.
    seq : DeRhamSequence
        Pre-built de Rham sequence (must have ``evaluate_1d`` and the
        relevant mass matrix assembled).
    k : {0, 1, 2, 3}
        Degree of the differential form.
    dirichlet : bool, optional
        Whether to use Dirichlet boundary conditions (default ``True``).
    reference_domain : bool, optional
        For ``k=0``, use the cached reference-domain mass matrix and
        quadrature weights instead of the active mapped geometry.

    Returns
    -------
    dof : jnp.ndarray
        Coefficient vector in the k-form FEM space.
    """
    from mrx.quadrature import integrate_against

    if reference_domain and k != 0:
        raise ValueError("reference_domain is only supported for k=0")

    x1, x2, x3 = axes
    n1, n2, n3 = len(x1), len(x2), len(x3)
    xq = seq.quad.x  # (n_q, 3)

    comp_info, comp_shapes = seq._form_comp_info(k)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)

    if k in (0, 3):
        # --- scalar field ---
        grid = values.reshape(n1, n2, n3)
        interp = RegularGridInterpolator(
            points=(x1, x2, x3), values=grid, method='linear')
        f_q = interp(xq)[:, None]                     # (n_q, 1)

        if k == 0:
            if reference_domain:
                w_jk = f_q * seq.quad.w[:, None]
            else:
                w_jk = f_q * (seq.quad.w * seq.jacobian_j)[:, None]
        else:  # k == 3
            w_jk = f_q * seq.quad.w[:, None]

    else:
        # --- vector field ---
        grid = values.reshape(n1, n2, n3, 3)
        v_q = jnp.stack([
            RegularGridInterpolator(
                points=(x1, x2, x3), values=grid[..., i],
                method='linear')(xq)
            for i in range(3)
        ], axis=-1)                                    # (n_q, 3)

        if k == 1:
            # 1-form RHS_i = ∫ Λ^1_i · (DF^{-1} v) J w dξ
            #               = ∫ Λ^1_i · G^{-1} (DF^T v) w dξ
            # using DF^{-1} = G^{-1} DF^T  (since G = DF^T DF).
            DF_q = jax.lax.map(
                jax.jacfwd(seq.map), xq,
                batch_size=mrx.MAP_BATCH_SIZE_INNER)  # (n_q, 3, 3)
            DFt_v = jnp.einsum('qji,qj->qi', DF_q, v_q)  # DF^T @ v
            Ginv_DFt_v = jnp.einsum('qij,qj->qi',
                                    seq.metric_inv_jkl, DFt_v)
            w_jk = Ginv_DFt_v * (seq.quad.w * seq.jacobian_j)[:, None]

        else:  # k == 2
            # 2-form pullback: DF^T v,  weighted by w  (no J)
            DF_q = jax.lax.map(
                jax.jacfwd(seq.map), xq,
                batch_size=mrx.MAP_BATCH_SIZE_INNER)  # (n_q, 3, 3)
            DFt_v = jnp.einsum('qji,qj->qi', DF_q, v_q)  # DF^T @ v
            w_jk = DFt_v * seq.quad.w[:, None]

    # Extraction operator
    match k:
        case 0: e = seq.e0_dbc if dirichlet else seq.e0
        case 1: e = seq.e1_dbc if dirichlet else seq.e1
        case 2: e = seq.e2_dbc if dirichlet else seq.e2
        case 3: e = seq.e3_dbc if dirichlet else seq.e3

    rhs = e @ integrate_against(w_jk, comp_info, comp_shapes, quad_shape)
    if reference_domain:
        return seq.apply_inverse_reference_mass_matrix(rhs, dirichlet=dirichlet)
    return seq.apply_inverse_mass_matrix(rhs, k=k, dirichlet=dirichlet)


def load_and_reshape_GVEC(gvec_eq, nfp):
    """
    Load GVEC equilibrium data and reshape it.

    Parameters
    ----------
    gvec_eq : xarray.Dataset
        GVEC equilibrium dataset.

    Returns
    -------
    x : jnp.ndarray
        evaluation points reshaped to (n_pts, 3).
    R : jnp.ndarray
        R coordinates reshaped to (n_pts,).
    Z : jnp.ndarray
        Z coordinates reshaped to (n_pts,).
    B : jnp.ndarray
        B field reshaped to (n_pts, 3).
    """
    _ρ = gvec_eq["rho"].values      # shape (mρ,)
    _θ = gvec_eq["theta"].values    # shape (mθ,)
    _ζ = gvec_eq["zeta"].values     # shape (mζ,)
    R = gvec_eq["X1"].values       # shape (mρ, mθ, mζ)
    Z = gvec_eq["X2"].values       # shape (mρ, mθ, mζ)
    B = gvec_eq["B"].values        # shape (mρ, mθ, mζ, 3)

    ρ, θ, ζ = jnp.meshgrid(_ρ, _θ, _ζ, indexing="ij")
    # θ_star = jnp.asarray(θ_star)
    pts = jnp.stack([ρ.ravel(),
                    θ.ravel() / (2 * jnp.pi),
                    ζ.ravel() / (2 * jnp.pi) * nfp], axis=1)  # x_hat_js, shape (mρ mθ mζ, 3)
    return (pts.reshape(-1, 3),
            R.ravel(),
            Z.ravel(),
            B.reshape(-1, 3))
