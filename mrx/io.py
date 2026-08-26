"""Command-line parsing, sweep loading, and projection of gridded data onto spline spaces."""
import argparse
import os
import random
import string
import h5py
import jax.numpy as jnp
import numpy as np



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


def load_grid_field(axes, values, seq, k, *, dirichlet=False, frame='ref',
                    degree=3):
    """Factorized dual load of a field sampled on a regular logical grid.

    Interpolatory-spline analogue of the pointwise ``seq.load(callable)`` for
    grid-sampled data.  Steps, all sum-factorized (no pointwise ``lax.map`` and
    no per-quad-point basis sweep):

    1. fit an interpolatory tensor-product B-spline to ``values`` — one square
       collocation solve per axis (``n_basis = n_data``);
    2. evaluate it at ``seq``'s quadrature grid via :func:`_tp_evaluate` — three
       1D contractions, ``O(N_q (n1+n2+n3))`` instead of ``O(N_q·n1·n2·n3)``;
    3. apply the k-form frame pullback and quadrature weight (mirrors
       :func:`mrx.projectors.load`; ``frame='phys'`` recomputes ``DF`` at the
       quadrature points with :func:`mrx.geometry.map_jacobian_at` -- this
       and ``load`` are the only consumers of ``DF``, which the geometry does
       not store);
    4. integrate against the k-form basis and extract.

    Returns the **dual load vector** (same as ``seq.load``); pass it to
    ``seq.apply_inverse_mass_matrix`` for the projected DOFs.

    Parameters
    ----------
    axes : tuple of 1-D arrays ``(x1, x2, x3)``  logical grid nodes per axis.
    values : array  ``(n1,n2,n3)`` for k=0,3;  ``(n1,n2,n3,3)`` for k=1,2
        (flattened variants accepted).
    seq : DeRhamSequence  target sequence (``evaluate_1d`` already called).
    k : {0,1,2,3}  form degree.
    dirichlet : bool  use Dirichlet-constrained DOFs.
    frame : {'ref','phys'}  interpretation of ``values`` (see
        :func:`mrx.projectors.load`).
    degree : int  spline degree of the interpolatory fit.
    """
    from mrx.differential_forms import DifferentialForm
    from mrx.geometry import _tp_evaluate, map_jacobian_at
    from mrx.projectors import _solve_tensor_collocation_axis
    from mrx.quadrature import integrate_against

    if frame not in ('ref', 'phys'):
        raise ValueError(f"frame must be 'ref' or 'phys', got {frame!r}")

    x1, x2, x3 = (jnp.asarray(a) for a in axes)
    n1, n2, n3 = len(x1), len(x2), len(x3)
    ncomp = 1 if k in (0, 3) else 3
    C = jnp.asarray(values).reshape(n1, n2, n3) if ncomp == 1 \
        else jnp.asarray(values).reshape(n1, n2, n3, 3).transpose(3, 0, 1, 2)
    C = C.reshape(ncomp, n1, n2, n3)

    # 1. interpolatory fit basis (n_basis = n_data per axis, seq's BC types)
    fit = DifferentialForm(0, (n1, n2, n3), (degree,) * 3, seq.basis_0.types)
    br, bt, bz = fit.Λ
    solve = (br.collocation_matrix(x1), bt.collocation_matrix(x2),
             bz.collocation_matrix(x3))
    for a in range(3):
        C = _solve_tensor_collocation_axis(solve[a], C, axis=a + 1)  # comp is axis 0

    # 2. factorized evaluation at seq's quadrature grid.  M_axis = fit basis at
    #    seq's per-axis 1D quad points (r<->x_x, t<->x_y, z<->x_z, per evaluate_1d).
    Mr = br.collocation_matrix(seq.quad.x_x).T          # (n1, nqr)
    Mt = bt.collocation_matrix(seq.quad.x_y).T          # (n2, nqt)
    Mz = bz.collocation_matrix(seq.quad.x_z).T          # (n3, nqz)
    f = _tp_evaluate(C, Mr, Mt, Mz)                     # (ncomp, nqr, nqt, nqz)
    # flatten with the same (0,2,1,3) transpose the geometry path uses, so the
    # per-quad-point order matches seq.quad.x / seq.quad.w (meshgrid 'xy': t,r,z).
    f_q = f.transpose(0, 2, 1, 3).reshape(ncomp, -1).T  # (n_q, ncomp)

    # 3. frame pullback + quadrature weight (mirrors mrx.projectors.load)
    w = seq.quad.w
    if k == 0:
        w_jk = f_q * (w * seq.jacobian_j)[:, None]
    elif k == 1:
        if frame == 'phys':                             # DF^-1 f = G^-1 DF^T f
            DF_q = map_jacobian_at(seq.map, seq.quad.x)
            DFt = jnp.einsum('qji,qj->qi', DF_q, f_q)
            f_q = jnp.einsum('qij,qj->qi', seq.metric_inv_jkl, DFt)
        w_jk = f_q * (w * seq.jacobian_j)[:, None]
    elif k == 2:
        if frame == 'phys':                             # DF^T f
            DF_q = map_jacobian_at(seq.map, seq.quad.x)
            f_q = jnp.einsum('qji,qj->qi', DF_q, f_q)
        w_jk = f_q * w[:, None]
    else:  # k == 3
        w_jk = f_q * (w if frame == 'phys' else w / seq.jacobian_j)[:, None]

    # 4. integrate against the k-form basis + extraction
    comp_info, comp_shapes = seq._form_comp_info(k)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    match k:
        case 0: e = seq.e0_dbc if dirichlet else seq.e0
        case 1: e = seq.e1_dbc if dirichlet else seq.e1
        case 2: e = seq.e2_dbc if dirichlet else seq.e2
        case 3: e = seq.e3_dbc if dirichlet else seq.e3
    return e @ integrate_against(w_jk, comp_info, comp_shapes, quad_shape)


