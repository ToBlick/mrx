# %%
import os
from typing import Any, Callable, Optional

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

import mrx


def is_running_in_github_actions():
    """
    Checks if the current Python script is running within a GitHub Actions environment.
    """
    return os.getenv("GITHUB_ACTIONS") == "true"


def double_map(f, xs, ys):
    """Apply f(x, y) over all combinations of xs and ys using nested jax.lax.map.

    Batch sizes are controlled by mrx.MAP_BATCH_SIZE_OUTER (outer loop)
    and mrx.MAP_BATCH_SIZE_INNER (inner loop).

    Args:
        f: Function (x, y) -> array.
        xs: Outer loop values.
        ys: Inner loop values.

    Returns:
        Array of shape (len(xs), len(ys), ...).
    """
    def outer(x):
        return jax.lax.map(lambda y: f(x, y), ys, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return jax.lax.map(outer, xs, batch_size=mrx.MAP_BATCH_SIZE_OUTER)


def norm_2(u: jnp.ndarray, Seq) -> float:
    """Compute the L2 norm of a vector field.

    Args:
        u: Vector field.
        Seq: DeRham sequence object.

    Returns:
        L2 norm of the vector field.
    """
    return (u @ Seq.M2 @ u)**0.5


def jacobian_determinant(f: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the determinant of the Jacobian matrix for a given function.

    Args:
        f: Function mapping from R^n to R^n for which to compute the Jacobian determinant

    Returns:
        Function that computes the Jacobian determinant at a given point
    """
    return lambda x: jnp.linalg.det(jax.jacfwd(f)(x))


def det33(mat: jnp.ndarray) -> jnp.ndarray:
    """Compute the determinant of a 3x3 matrix using explicit formula.

    This function computes the determinant using the rule of Sarrus, which is
    more efficient than general determinant computation for 3x3 matrices.

    Args:
        mat: 3x3 matrix for which to compute the determinant
    Returns:
        The determinant of the input matrix
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    return m1 * (m5 * m9 - m6 * m8) - m2 * (m4 * m9 - m6 * m7) + m3 * (m4 * m8 - m5 * m7)


def inv33(mat: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse of a 3x3 matrix using explicit formula.

    This function computes the inverse using the adjugate matrix formula,
    which is more efficient than general matrix inversion for 3x3 matrices.

    Args:
        mat: 3x3 matrix to invert

    Returns:
        The inverse of the input matrix
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * \
        (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    # Return zero matrix if determinant is zero
    return jnp.where(
        jnp.abs(det) < 1e-10,  # Careful with this tolerance
        jnp.zeros((3, 3)),
        jnp.array([
            [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
            [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
            [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
        ]) / det
    )


def div(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the divergence of a vector field.

    Args:
        F: Vector field function for which to compute the divergence

    Returns:
        Function that computes the divergence at a given point
    """
    def div_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.trace(DF) * jnp.ones(1)
    return div_F


def curl(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the curl of a vector field in 3D.

    Args:
        F: Vector field function for which to compute the curl

    Returns:
        Function that computes the curl at a given point
    """
    def curl_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.array([DF[2, 1] - DF[1, 2],
                         DF[0, 2] - DF[2, 0],
                         DF[1, 0] - DF[0, 1]])
    return curl_F


def grad(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the gradient of a scalar field.

    Args:
        F: Scalar field function for which to compute the gradient

    Returns:
        Function that computes the gradient at a given point
    """
    def grad_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.ravel(DF)
    return grad_F


def l2_product(f: Callable[[jnp.ndarray], jnp.ndarray],
               g: Callable[[jnp.ndarray], jnp.ndarray],
               Q: Any,
               F: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x) -> jnp.ndarray:
    """Compute the L2 inner product of two functions over a domain.

    Computes the integral of f·g over the domain defined by the quadrature rule Q,
    with optional coordinate transformation F.

    Args:
        f: First function in the inner product
        g: Second function in the inner product
        Q: Quadrature rule object with attributes x (points) and w (weights)
        F: Optional coordinate transformation function (default is identity)

    Returns:
        The L2 inner product value
    """
    J_i = jax.lax.map(jacobian_determinant(F), Q.x,
                      batch_size=mrx.MAP_BATCH_SIZE_INNER)
    f_ij = jax.lax.map(f, Q.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    g_ij = jax.lax.map(g, Q.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return jnp.einsum("ij,ij,i,i->", f_ij, g_ij, J_i, Q.w)


def evaluate_at_xq_deprecated(getter, dofs, n_q, d):
    """
    Evaluate a finite element function at quadrature points.

    Parameters
    ----------
    getter : callable
        Function (i, j, k) -> scalar. kth component of form i evaluated at quadrature point j.
    dofs : jnp.ndarray, shape (m,)
        Degrees of freedom of the finite element function, already contracted with extraction matrices
    n_q : int
        Number of quadrature points.
    d : int
        Number of dimensions.

    Returns
    -------
    f_h_jk : jnp.ndarray, shape (n_q, d)
        Function values at quadrature points.
    """
    # Evaluate the finite element function at quadrature points
    get_f_k = jax.vmap(getter, in_axes=(None, None, 0))  # over k (dimensions)

    def get_f_jk(i, js, ks):
        return jax.lax.map(lambda j: get_f_k(i, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    v_i = dofs  # shape (n_i,)

    def body_fun(carry, i):
        L_i = get_f_jk(i, jnp.arange(n_q), jnp.arange(d))  # shape (n_q, d)
        # broadcast scalar over last axis (dimesions)
        carry += L_i * v_i[i]
        return carry, None

    R_init = jnp.zeros_like(
        get_f_jk(0, jnp.arange(n_q), jnp.arange(d)))  # shape (n_q, d)
    R, _ = jax.lax.scan(body_fun, R_init, jnp.arange(v_i.shape[0]))
    return R


def integrate_against_deprecated(getter, w_jk, n):
    """
    Integrate a function represented at quadrature points against a set of basis functions.

    Args:
        getter (callable): Function (i, j, k) -> scalar. kth component of form i evaluated at quadrature point j.
        w_jk (jnp.ndarray): Function values at quadrature points, shape (n_q, d).
        n (int): Number of basis functions.

    Returns:
        jnp.ndarray: Integrated values, shape (n,). Entries are given by
        ∑_{j,k} Λ[i,j,k] * w[j,k]
    """
    n_q, d = w_jk.shape
    get_f_k = jax.vmap(getter, in_axes=(None, None, 0))  # over k (dimensions)

    def get_f_jk(i, js, ks):
        return jax.lax.map(lambda j: get_f_k(i, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    def body_fun(carry, i):
        L_i = get_f_jk(i, jnp.arange(n_q), jnp.arange(d))  # shape (n_q, d)
        return None, jnp.sum(L_i * w_jk)

    _, R = jax.lax.scan(body_fun, None, jnp.arange(n))
    return R


def evaluate_at_xq(dofs, comp_info, comp_shapes, quad_shape, d):
    """Evaluate a k-form at quadrature points using tensor-product structure.

    Parameters
    ----------
    dofs : array, shape (n_total,)
        DOF vector (internal, already contracted with extraction matrices).
    comp_info : list of (output_dim, R, T, Z)
        For each component c, the output dimension and 1D basis arrays.
        R shape (s1_c, nq_r), T shape (s2_c, nq_t), Z shape (s3_c, nq_z).
    comp_shapes : list of tuples (s1_c, s2_c, s3_c)
        DOF grid shape per component.
    quad_shape : tuple (nq_t, nq_r, nq_z)
    d : int
        Number of output dimensions.

    Returns
    -------
    f_jk : array, shape (n_q, d)
    """
    f = jnp.zeros((d,) + quad_shape)

    offset = 0
    for c, (out_dim, R, T, Z) in enumerate(comp_info):
        s = comp_shapes[c]
        n_c = s[0] * s[1] * s[2]
        V = dofs[offset:offset + n_c].reshape(s)
        # V[i,j,k], R[i,a], T[j,b], Z[k,c] -> f[b,a,c]
        # quad_shape = (nq_t, nq_r, nq_z) -> output indices (b, a, c)
        val = jnp.einsum('ijk,ia,jb,kc->bac', V, R, T, Z)
        f = f.at[out_dim].add(val)
        offset += n_c

    return f.transpose(1, 2, 3, 0).reshape(-1, d)


def integrate_against(f_jk, comp_info, comp_shapes, quad_shape):
    """Integrate quad-point values against k-form basis using TP structure.

    Parameters
    ----------
    f_jk : array, shape (n_q, d)
        Values at quadrature points (already weighted).
    comp_info : list of (input_dim, R, T, Z)
        For each component c, the input dimension and 1D basis arrays.
    comp_shapes : list of tuples (s1_c, s2_c, s3_c)
    quad_shape : tuple (nq_t, nq_r, nq_z)

    Returns
    -------
    result : array, shape (n_total,)
    """
    d = f_jk.shape[1]
    # Reshape to (d, nq_t, nq_r, nq_z)
    f = f_jk.reshape(quad_shape + (d,)).transpose(3, 0, 1, 2)

    parts = []
    for c, (in_dim, R, T, Z) in enumerate(comp_info):
        # f[in_dim] has shape (nq_t, nq_r, nq_z)
        # R[i,a], T[j,b], Z[k,c], f[b,a,c] -> result[i,j,k]
        val = jnp.einsum('ia,jb,kc,bac->ijk', R, T, Z, f[in_dim])
        parts.append(val.ravel())

    return jnp.concatenate(parts)


# Default trace dictionary for the relaxation loop
default_trace_dict = {
    "iterations": [],
    "force_trace": [],
    "energy_trace": [],
    "helicity_trace": [],
    "divergence_trace": [],
    "picard_iterations": [],
    "picard_errors": [],
    "timesteps": [],
    "velocity_trace": [],
    "wall_time_trace": [],
    "B_fields": [],
    "p_fields": [],
    "start_time": None,
    "end_time": None,
}


def append_to_trace_dict(
        trace_dict: dict, i: int, f: float, E: float,
        H: float, dvg: float, v: float, p_i: int,
        e: float, dt: float, end_time: float,
        B: Optional[jnp.ndarray] = None) -> dict:
    """
    Append values to the trace dictionary.

    Args:
        trace_dict: Dictionary to append values to.
        i: Iteration number.
        f: Force norm.
        E: Energy.
        H: Helicity.
        dvg: Divergence norm.
        v: Velocity norm.
        p_i: Picard iterations.
        e: Picard error.
        dt: Time step.
        end_time: End time.
        B: Magnetic field.

    Returns:
        trace_dict: Dictionary with appended values.
    """
    trace_dict["iterations"].append(i)
    trace_dict["force_trace"].append(f)
    trace_dict["energy_trace"].append(E)
    trace_dict["helicity_trace"].append(H)
    trace_dict["divergence_trace"].append(dvg)
    trace_dict["velocity_trace"].append(v)
    trace_dict["picard_iterations"].append(p_i)
    trace_dict["picard_errors"].append(e)
    trace_dict["timesteps"].append(dt)
    trace_dict["wall_time_trace"].append(end_time - trace_dict["start_time"])
    if B is not None:
        trace_dict["B_fields"].append(B)
    return trace_dict


def save_trace_dict_to_hdf5(trace_dict: dict, diagnostics, filename: str, CONFIG: dict):
    """
    Save the trace dictionary to an HDF5 file.

    Args:
        trace_dict: Trace dictionary.
        diagnostics: MRXDiagnostics object.
        filename: Name of the file to save the trace dictionary to.
        CONFIG: Configuration dictionary.

    Returns:
        None
    """
    import h5py
    import numpy as np
    Seq = diagnostics.Seq
    print(filename)
    with h5py.File(filename + ".h5", "w") as f:
        f.create_dataset("iterations", data=jnp.array(
            trace_dict["iterations"]))
        f.create_dataset("force_trace", data=jnp.array(
            trace_dict["force_trace"]))
        f.create_dataset("B_final", data=trace_dict["B_final"])
        f.create_dataset("p_final", data=trace_dict["p_final"])
        f.create_dataset("energy_trace", data=jnp.array(
            trace_dict["energy_trace"]))
        f.create_dataset("helicity_trace", data=jnp.array(
            trace_dict["helicity_trace"]))
        f.create_dataset("divergence_trace", data=jnp.array(
            trace_dict["divergence_trace"]))
        f.create_dataset("velocity_trace", data=jnp.array(
            trace_dict["velocity_trace"]))
        f.create_dataset("picard_iterations", data=jnp.array(
            trace_dict["picard_iterations"]))
        f.create_dataset("picard_errors", data=jnp.array(
            trace_dict["picard_errors"]))
        f.create_dataset("timesteps", data=jnp.array(trace_dict["timesteps"]))
        f.create_dataset("harmonic_norm", data=jnp.array(
            [norm_2(diagnostics.harmonic_component(trace_dict["B_final"]), Seq)]))
        f.create_dataset("total_time", data=jnp.array(
            [trace_dict["end_time"] - trace_dict["start_time"]]))
        f.create_dataset("time_setup", data=jnp.array(
            [trace_dict["setup_done_time"] - trace_dict["start_time"]]))
        f.create_dataset("time_solve", data=jnp.array(
            [trace_dict["end_time"] - trace_dict["setup_done_time"]]))
        f.create_dataset("wall_time_trace", data=jnp.array(
            trace_dict["wall_time_trace"]))
        if CONFIG["save_B"]:
            f.create_dataset("B_fields", data=jnp.array(
                trace_dict["B_fields"]))
            f.create_dataset("p_fields", data=jnp.array(
                trace_dict["p_fields"]))
        # Store config variables in a group
        cfg_group = f.create_group("config")
        for key, val in CONFIG.items():
            # Skip callable objects (functions) and other non-serializable types
            if callable(val):
                continue
            # Skip numpy arrays with object dtype
            if isinstance(val, np.ndarray) and val.dtype == object:
                continue
            # Try to convert to numpy array and check for object dtype
            try:
                val_array = np.asarray(val)
                if val_array.dtype == object:
                    continue
            except (ValueError, TypeError):
                pass

            if isinstance(val, str):
                # Strings need special handling
                cfg_group.attrs[key] = np.bytes_(val)
            else:
                try:
                    cfg_group.attrs[key] = val
                except (TypeError, ValueError):
                    # Skip values that can't be serialized to HDF5
                    continue


def run_relaxation_loop(CONFIG, trace_dict, state, diagnostics):
    """
    Run the relaxation loop.

    Args:
        CONFIG: Configuration dictionary.
        trace_dict: Trace dictionary.
        state: State object.
        diagnostics: MRXDiagnostics object.
    """
    import time

    from mrx.relaxation_deprecated import (DescentMethod, MRXHessian,
                                           TimeStepper)

    # Construct the time stepper and the Hessian
    Seq = diagnostics.Seq
    timestepper = TimeStepper(Seq,
                              gamma=CONFIG["gamma"],
                              descent_method=DescentMethod.NEWTON if CONFIG[
                                  "precond"] else DescentMethod.GRADIENT,
                              force_free=CONFIG["force_free"],
                              picard_tol=CONFIG["solver_tol"],
                              picard_k_restart=CONFIG["solver_maxit"])

    compute_hessian = jax.jit(MRXHessian(Seq).assemble)  # defaults to identity
    step = jax.jit(lambda state, key: timestepper.relaxation_step(state, key))
    B_hat = state.B_n
    get_energy = jax.jit(diagnostics.energy)
    get_helicity = jax.jit(diagnostics.helicity)
    get_divergence_B = jax.jit(diagnostics.divergence_norm)

    # Compile and record initial values
    dry_run = step(state, state.key)
    trace_dict = append_to_trace_dict(trace_dict, 0,
                                      dry_run.F_norm,
                                      get_energy(state.B_n),
                                      get_helicity(state.B_n),
                                      get_divergence_B(state.B_n),
                                      dry_run.v_norm,
                                      0,
                                      0,
                                      dry_run.dt,
                                      time.time(),
                                      B_hat if CONFIG["save_B"] else None)

    print(f"Initial force error: {trace_dict['force_trace'][-1]:.2e}")
    print(f"Initial energy: {trace_dict['energy_trace'][-1]:.2e}")
    print(f"Initial helicity: {trace_dict['helicity_trace'][-1]:.2e}")
    print(f"Initial ||div B||: {trace_dict['divergence_trace'][-1]:.2e}")

    setup_done_time = time.time()
    trace_dict["setup_done_time"] = setup_done_time
    print(
        f"Setup took {setup_done_time - trace_dict['start_time']:.2e} seconds.")

    print("Starting relaxation loop...")
    for i in range(1, CONFIG["maxit"] + 1):

        state = step(state, state.key)
        if (state.picard_residuum > CONFIG["solver_tol"]
                or ~jnp.isfinite(state.picard_residuum)):
            # half time step and try again
            state = timestepper.update_field(state, "dt", state.dt / 2)
            state = timestepper.update_field(state, "B_nplus1", state.B_n)
            continue
        # otherwise, we converged - proceed
        state = timestepper.update_field(state, "B_n", state.B_nplus1)

        if i == CONFIG["apply_pert_after"] and CONFIG["pert_strength"] > 0:
            print(f"Applying perturbation after {i} steps...")
            dB_hat = jnp.linalg.solve(Seq.M2, Seq.P2(CONFIG["dB_xyz"]))
            dB_hat = Seq.P_Leray @ dB_hat
            dB_hat /= norm_2(dB_hat, Seq)
            B_new = state.B_n + CONFIG["pert_strength"] * dB_hat
            state = timestepper.update_field(state, "B_n", B_new)

        if CONFIG["precond"] and (i % CONFIG["precond_compute_every"] == 0):
            state = timestepper.update_field(
                state, "hessian", compute_hessian(state.B_n))

        if state.picard_iterations < CONFIG["solver_critit"]:
            dt_new = state.dt * CONFIG["dt_factor"]
        else:
            dt_new = state.dt / (CONFIG["dt_factor"])**2
        state = timestepper.update_field(state, "dt", dt_new)

        if i % CONFIG["save_every"] == 0 or i == CONFIG["maxit"]:
            trace_dict = append_to_trace_dict(trace_dict, i,
                                              state.F_norm,
                                              get_energy(state.B_n),
                                              get_helicity(state.B_n),
                                              get_divergence_B(state.B_n),
                                              state.v_norm,
                                              state.picard_iterations,
                                              state.picard_residuum,
                                              state.dt,
                                              time.time(),
                                              state.B_n if CONFIG["save_B"] else None)

        if i % CONFIG["print_every"] == 0:
            print(
                f"Iteration {i}, u norm: {state.v_norm:.2e}, force norm: {state.F_norm:.2e}")
            if CONFIG["verbose"]:
                print(
                    f"   dt: {dt_new:.2e}, picard iters: {state.picard_iterations:.2e}, picard err: {state.picard_residuum:.2e}")
        if trace_dict["force_trace"][-1] < CONFIG["force_tol"]:
            print(
                f"Converged to force tolerance {CONFIG['force_tol']} after {i} steps.")
            break


def update_config(params: dict, CONFIG: dict):
    """
    Get the configuration from parameters specified on the command line.

    Args:
        params: Parameters dictionary.
        CONFIG: Configuration dictionary.

    Returns:
        CONFIG: Updated configuration dictionary.
    """
    # Step 1: If device specified, apply defaults
    device_name = params.get("device")
    if device_name:
        preset = DEVICE_PRESETS.get(device_name.upper())
        if preset:
            for k, v in preset.items():
                CONFIG[k] = v
        else:
            print(f"Unknown device '{device_name}' - ignoring.")

    # Step 2: Override with user-supplied parameters
    for k, v in params.items():
        if k in CONFIG:
            CONFIG[k] = v
        elif k != "device":
            print(f"Unknown parameter '{k}' - ignoring.")

    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    return CONFIG


def _bcsr_to_coo_indices(mat: jsparse.BCSR):
    """Expand BCSR indptr to COO-style (row, col) index array."""
    nse = mat.data.shape[0]
    lengths = mat.indptr[1:] - mat.indptr[:-1]
    rows = jnp.repeat(jnp.arange(mat.shape[0]), lengths,
                      total_repeat_length=nse)
    return jnp.stack([rows, mat.indices], axis=1)


def extract_diag_vector(mat) -> jnp.ndarray:
    """Extracts the main diagonal of a sparse matrix as a 1D array."""
    n = mat.shape[0]
    if isinstance(mat, jsparse.BCSR):
        indices = _bcsr_to_coo_indices(mat)
        rows, cols = indices[:, 0], indices[:, 1]
    else:
        rows = mat.indices[:, 0]
        cols = mat.indices[:, 1]
    is_diag = rows == cols
    diag_data = jnp.where(is_diag, mat.data, 0.0)
    return jnp.zeros(n, dtype=mat.dtype).at[rows].add(diag_data)

def square_sparse(mat) -> jsparse.BCOO:
    """Squares the non-zero elements of a sparse matrix. Always returns BCOO."""
    if isinstance(mat, jsparse.BCSR):
        indices = _bcsr_to_coo_indices(mat)
        return jsparse.BCOO((mat.data**2, indices), shape=mat.shape)
    return jsparse.BCOO((mat.data**2, mat.indices), shape=mat.shape)


# backward compat alias
square_bcoo = square_sparse


def diag_EAET(E, A, E_T=None):
    """Compute the diagonal of E @ A @ E^T via mapped matvecs.

    Uses diag(E A E^T)_i = v_i^T A v_i  where v_i = E^T e_i (row i of E).
    This avoids forming any dense matrices.
    """
    n = E.shape[0]
    if E_T is None:
        if isinstance(E, jsparse.BCSR):
            coo_idx = _bcsr_to_coo_indices(E)
            E_T = jsparse.BCOO((E.data, coo_idx), shape=E.shape).T
        else:
            E_T = E.T

    def entry(i):
        e_i = jnp.zeros(n).at[i].set(1.0)
        v = E_T @ e_i
        return v @ (A @ v)

    return jax.lax.map(entry, jnp.arange(n), batch_size=mrx.MAP_BATCH_SIZE_OUTER)


def diag_schur_complement(apply_DT, diag_inv, n):
    """Compute diag(D @ diag(diag_inv) @ D^T) via mapped matvecs.

    For each row i, computes e_i^T D diag(diag_inv) D^T e_i
    = ||diag_inv^{1/2} D^T e_i||^2, using jax.lax.map over i.

    Args:
        apply_DT: callable, v -> D^T v (maps k-form DOFs to (k-1)-form DOFs)
        diag_inv: 1D array, diagonal approximation of M_{k-1}^{-1}
        n: int, number of k-form DOFs (rows of D)
    """
    def entry(i):
        e_i = jnp.zeros(n).at[i].set(1.0)
        Dt_ei = apply_DT(e_i)
        return jnp.dot(Dt_ei, diag_inv * Dt_ei)
    return jax.lax.map(entry, jnp.arange(n), batch_size=mrx.MAP_BATCH_SIZE_OUTER)

# %%

# def solve_singular_cg(A_matvec, b, mass_matvec=None, precond_matvec=lambda x: x, x0=None, vs=[], maxiter=1000, tol=1e-6):

#    if mass_matvec is None:
#        mass_matvec = lambda x: x

#     # --- 1. Your Exact Projection Logic ---
#     def inner_product(x, y):
#         return jnp.dot(x, mass_matvec(y))

#     def project_primal(x):
#         for v in vs:
#             x = x - inner_product(v, x) * v
#         return x

#     def project_dual(f):
#         for v in vs:
#             f = f - jnp.dot(v, f) * mass_matvec(v)
#         return f

#     # --- 2. Initial Setup ---
#     b_proj = project_dual(b)
#     if x0 is None:
#         x0 = jnp.zeros_like(b_proj)
#     else:
#         x0 = project_primal(x0)

#     # Initial residual (Dual)
#     Ax0 = project_dual(A_matvec(x0))
#     r0 = b_proj - Ax0
#     r0 = project_dual(r0)  # Clean the initial residual

#     # Initial preconditioned residual (Primal)
#     z0 = precond_matvec(r0)
#     z0 = project_primal(z0) # Clean the preconditioner output

#     p0 = z0

#     # State: (iteration, x, r, p, z, r_dot_z, residual_norm)
#     r_dot_z_0 = jnp.vdot(r0, z0)
#     init_state = (0, x0, r0, p0, z0, r_dot_z_0, jnp.linalg.norm(r0))

#     # --- 3. The CG Loop ---
#     def cond_fun(state):
#         i, _, _, _, _, _, r_norm = state
#         return (i < maxiter) & (r_norm > tol)

#     def body_fun(state):
#         i, x, r, p, z, r_dot_z, _ = state

#         # Matrix-vector multiply (output is Dual)
#         Ap = A_matvec(p)
#         Ap = project_dual(Ap) # Safe evaluation

#         # Step size
#         p_dot_Ap = jnp.vdot(p, Ap)
#         alpha = r_dot_z / p_dot_Ap

#         # Update solution and residual
#         x_next = x + alpha * p
#         r_next = r - alpha * Ap

#         r_next = project_dual(r_next)

#         # Apply and project preconditioner
#         z_next = precond_matvec(r_next)
#         z_next = project_primal(z_next) # PREVENTS PRECONDITIONER DRIFT

#         # Update search direction
#         r_dot_z_next = jnp.vdot(r_next, z_next)
#         beta = r_dot_z_next / r_dot_z
#         p_next = z_next + beta * p

#         # Keep p strictly in the Primal subspace
#         p_next = project_primal(p_next)

#         return (i + 1, x_next, r_next, p_next, z_next, r_dot_z_next, jnp.linalg.norm(r_next))

#     # --- 4. Execute ---
#     final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

#     iters, x_final, _, _, _, _, info_norm = final_state

#     # Final safety projection
#     return project_primal(x_final), {"iterations": iters, "residual_norm": info_norm}


def get_smallest_ev_pair(A_matvec, mass_matvec, x0, precond_matvec=lambda x: x, vs=[], shift=1e-9, maxiter=20, tol=1e-6):
    """
    Finds the generalized eigenvector using shifted inverse iteration. 
    """
    def inner_product(x, y):
        return jnp.dot(x, mass_matvec(y))

    def normalize(x):
        return x / jnp.sqrt(inner_product(x, x))

    def project_primal(x):
        # These are DoF vectors
        for v in vs:
            x = x - inner_product(v, x) * v
        return x

    def project_dual(f):
        # These are bilinear form outputs
        for v in vs:
            f = f - jnp.dot(v, f) * mass_matvec(v)
        return f

    def A_shifted(x):
        x = project_primal(x)
        # (outputs are Dual vectors)
        Ax = A_matvec(x) + shift * mass_matvec(x)
        return project_dual(Ax)

    x0 = normalize(project_primal(x0))

    def cond_fun(val):
        i, x, x_prev = val
        # Check both signs
        diff = jnp.minimum(jnp.linalg.norm(x - x_prev),
                           jnp.linalg.norm(x + x_prev))
        return jnp.logical_and(i < maxiter, diff > tol)

    def body_fun(val):
        i, x, _ = val
        rhs = mass_matvec(x)
        rhs = project_dual(rhs)
        y, _ = cg(A_shifted, rhs, x0=jnp.zeros_like(
            x), M=precond_matvec, tol=tol, maxiter=maxiter)
        y = project_primal(y)
        x_next = normalize(y)
        return (i + 1, x_next, x)

    init_val = (0, x0, jnp.zeros_like(x0))
    _, v, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    # Rayleigh quotient
    lmbda = jnp.dot(v, A_matvec(v))

    return v, lmbda
    return v, lmbda
