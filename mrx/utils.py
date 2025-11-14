from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = ['jacobian_determinant', 'inv33',
           'div', 'curl', 'grad', 'l2_product', 'DEVICE_PRESETS', 'DEFAULT_CONFIG',
           'append_to_trace_dict', 'default_trace_dict', 'update_config']

def norm_2(u : jnp.ndarray, Seq) -> float:
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
    Jj = jax.vmap(jacobian_determinant(F))(Q.x)
    return jnp.einsum("ij,ij,i,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Jj, Q.w)


def assemble(getter_1, getter_2, W, n1, n2):    
    """
    Assemble a matrix M[a, b] = Σ_{a,j,k} Λ1[a,j,i] * W[j,i,k] * Λ2[b,j,k]

    Parameters
    ----------
    getter_1 : callable
        Function (a, j, k) -> scalar. kth component of form a evaluated at quadrature point j.
    getter_2 : callable
        Function (b, j, k) -> scalar. kth component of form b evaluated at quadrature point j.
    W : jnp.ndarray, shape (n_q, 3, 3)
        Weight tensor combining metric, Jacobian, and quadrature weights.
        (For example: G_inv[q, ...] * J[q] * w[q])
    n1 : int
        Number of row basis functions.
    n2 : int
        Number of column basis functions.

    Returns
    -------
    M : jnp.ndarray, shape (n1, n2)
        The assembled matrix.
    """

    n_q, d, _ = W.shape

    get_A_jk = jax.vmap(
        jax.vmap(getter_1, in_axes=(None, None, 0)),  # over k (dimensions)
        # over j (quadrature points)
        in_axes=(None, 0, None)
    )

    get_B_jk = jax.vmap(
        jax.vmap(getter_2, in_axes=(None, None, 0)),
        in_axes=(None, 0, None)
    )

    def body_fun(carry, i):
        ΛA_i = get_A_jk(i, jnp.arange(n_q), jnp.arange(d))

        def compute_row(m):
            ΛB_m = get_B_jk(m, jnp.arange(n_q), jnp.arange(d))
            return jnp.einsum("jk,jkm,jm->", ΛA_i, W, ΛB_m)

        M_row = jax.vmap(compute_row)(jnp.arange(n2))
        return None, M_row

    _, M = jax.lax.scan(body_fun, None, jnp.arange(n1))
    return M


def evaluate_at_xq(getter, dofs, n_q, d):
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
    get_f_jk = jax.vmap(
        jax.vmap(getter, in_axes=(None, None, 0)),  # over k (dimensions)
        # over j (quadrature points)
        in_axes=(None, 0, None)
    )

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


def integrate_against(getter, w_jk, n):
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
    get_f_jk = jax.vmap(
        jax.vmap(getter, in_axes=(None, None, 0)),  # over k (dimensions)
        # over j (quadrature points)
        in_axes=(None, 0, None)
    )

    def body_fun(carry, i):
        L_i = get_f_jk(i, jnp.arange(n_q), jnp.arange(d))  # shape (n_q, d)
        return None, jnp.sum(L_i * w_jk)

    _, R = jax.lax.scan(body_fun, None, jnp.arange(n))
    return R


# Device-specific parameter presets for the relaxation
DEVICE_PRESETS = {
    "ITER":  {"eps": 0.32, "kappa": 1.7, "delta": 0.33, "q_star": 1.57, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "NSTX":  {"eps": 0.78, "kappa": 2.0, "delta": 0.35, "q_star": 2.0, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "SPHERO": {"eps": 0.95, "kappa": 1.0, "delta": 0.2,  "q_star": 0.0, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "ROT_ELL": {"a": 0.1, "b": 0.025, "m_rot": 5, "q_star": 1.6, "type": "rotating_ellipse"},
    "HELIX": {"eps": 0.33, "h_helix": 0.20, "kappa": 1.7, "delta": 0.33, "m_helix": 3, "q_star": 2.0, "type": "helix"},
}

# Default configuration parameters for the relaxation
DEFAULT_CONFIG = {
    # Run parameters
    "run_name": "",
    "boundary_type": "rotating_ellipse", # Type of boundary: "tokamak" or "helix" or "rotating_ellipse"

    # Parameters describing the domain. Some of these parameters are ignored for certain domain shapes.
    "eps":      0.2,  # aspect ratio
    "kappa":    1.7,   # Elongation parameter
    "q_star":   1.57,   # toroidal field strength
    "delta": 0.0,   # triangularity
    "nfp":  3,     # poloidal mode number of helix (number of field periods)
    "h_helix":  0,   # radius of helix turns
    "R_0": 1.0,   # major radius of the domain

    # Discretization parameters for the finite element space
    "n_r": 8,       # Number of radial splines
    "n_theta": 8,   # Number of poloidal splines
    "n_zeta": 6,    # Number of toroidal splines
    "p_r": 3,       # Degree of radial splines
    "p_theta": 3,     # Degree of poloidal splines
    "p_zeta": 3,    # Degree of toroidal splines

    # Hyperparameters for the outer loop of the magnetic relaxation solver
    "maxit":                 5_000,   # max. Number of time steps
    "precond":               False,     # Use preconditioner
    "precond_compute_every": 1000,       # Recompute preconditioner every n iterations
    "gamma":                 0, # Regularization, u = (-Δ)⁻ᵞ (J x B - grad p)
    "dt":                    1e-6,      # initial time step
    "dt_factor":             1.01,  # time-steps are increased by this factor and decreased by its square
    "force_tol":             1e-15,  # Convergence tolerance for |JxB - grad p| (or |JxB| if force_free)
    "eta":                   0.0,       # Resistivity
    "force_free":            False, # If True, solve for JxB = 0. If False, JxB = grad p

    # Solver hyperparameters for the inner loop of the magnetic relaxation solver
    "solver_maxit": 20,    # Maximum number of iterations before Picard solver gives up
    "solver_critit": 4, # If Picard solver converges in less than this number of iterations, increase time step
    "solver_tol": 1e-12,   # Tolerance for convergence
    "verbose": False,      # If False, prints only force every 'print_every'
    "print_every": 1000,    # Print every n iterations
    "save_every": 100,     # Save intermediate results every n iterations
    "save_B": False,       # Save intermediate B fields to file
    "save_B_every": 500,   # Save full B every n iterations

    # Hyperparameters pertaining to island seeding
    "pert_strength":       0.0,  # strength of perturbation
    "pert_pol_mode":          2,  # poloidal mode number of perturbation
    "pert_tor_mode":          1,  # toroidal mode number of perturbation
    "pert_radial_loc":      1/2,  # radial location of perturbation
    "pert_radial_width":    0.07,  # radial width of perturbation
    "apply_pert_after":     2000,  # apply perturbation after n steps (0 = to initial condition)
}


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
    trace_dict : dict, i : int, f : float, E : float, 
    H : float, dvg : float, v : float, p_i : int, 
    e : float, dt : float, end_time : float, 
    B : jnp.ndarray | None = None) -> dict:
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
    
def save_trace_dict_to_hdf5(trace_dict : dict, diagnostics, filename: str, CONFIG: dict):
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
            f.create_dataset("iterations", data=jnp.array(trace_dict["iterations"]))
            f.create_dataset("force_trace", data=jnp.array(trace_dict["force_trace"]))
            f.create_dataset("B_final", data=trace_dict["B_final"])
            f.create_dataset("p_final", data=trace_dict["p_final"])
            f.create_dataset("energy_trace", data=jnp.array(trace_dict["energy_trace"]))
            f.create_dataset("helicity_trace", data=jnp.array(trace_dict["helicity_trace"]))
            f.create_dataset("divergence_trace", data=jnp.array(trace_dict["divergence_trace"]))
            f.create_dataset("velocity_trace", data=jnp.array(trace_dict["velocity_trace"]))
            f.create_dataset("picard_iterations", data=jnp.array(trace_dict["picard_iterations"]))
            f.create_dataset("picard_errors", data=jnp.array(trace_dict["picard_errors"]))
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
                f.create_dataset("B_fields", data=jnp.array(trace_dict["B_fields"]))
                f.create_dataset("p_fields", data=jnp.array(trace_dict["p_fields"]))
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
    from mrx.relaxation import TimeStepper, MRXHessian

    # Construct the time stepper and the Hessian
    Seq = diagnostics.Seq
    timestepper = TimeStepper(Seq,
                              gamma=CONFIG["gamma"],
                              newton=CONFIG["precond"],
                              force_free=CONFIG["force_free"],
                              picard_tol=CONFIG["solver_tol"],
                              picard_maxit=CONFIG["solver_maxit"])

    compute_hessian = jax.jit(MRXHessian(Seq).assemble)  # defaults to identity
    step = jax.jit(timestepper.picard_solver)
    B_hat = state.B_n
    get_energy = jax.jit(diagnostics.energy)
    get_helicity = jax.jit(diagnostics.helicity)
    get_divergence_B = jax.jit(diagnostics.divergence_norm)

    # Compile and record initial values
    dry_run = step(state)
    trace_dict = append_to_trace_dict(trace_dict, 0,
                        dry_run.force_norm,
                        get_energy(state.B_n),
                        get_helicity(state.B_n),
                        get_divergence_B(state.B_n),
                        dry_run.velocity_norm,
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
    print(f"Setup took {setup_done_time - trace_dict['start_time']:.2e} seconds.")

    print("Starting relaxation loop...")
    for i in range(1, CONFIG["maxit"] + 1):

        state = step(state)
        if (state.picard_residuum > CONFIG["solver_tol"]
                or ~jnp.isfinite(state.picard_residuum)):
            # half time step and try again
            state = timestepper.update_dt(state, state.dt / 2)
            state = timestepper.update_B_guess(state, state.B_n)
            continue
        # otherwise, we converged - proceed
        state = timestepper.update_B_n(state, state.B_guess)

        if i == CONFIG["apply_pert_after"] and CONFIG["pert_strength"] > 0:
            print(f"Applying perturbation after {i} steps...")
            dB_hat = jnp.linalg.solve(Seq.M2, Seq.P2(CONFIG["dB_xyz"]))
            dB_hat = Seq.P_Leray @ dB_hat
            dB_hat /= norm_2(dB_hat, Seq)
            B_new = state.B_n + CONFIG["pert_strength"] * dB_hat
            state = timestepper.update_B_n(state, B_new)

        if CONFIG["precond"] and (i % CONFIG["precond_compute_every"] == 0):
            state = timestepper.update_hessian(
                state, compute_hessian(state.B_n))

        if state.picard_iterations < CONFIG["solver_critit"]:
            dt_new = state.dt * CONFIG["dt_factor"]
        else:
            dt_new = state.dt / (CONFIG["dt_factor"])**2
        state = timestepper.update_dt(state, dt_new)

        if i % CONFIG["save_every"] == 0 or i == CONFIG["maxit"]:
            trace_dict = append_to_trace_dict(trace_dict, i,
                        state.force_norm,
                        get_energy(state.B_n),
                        get_helicity(state.B_n),
                        get_divergence_B(state.B_n),
                        state.velocity_norm,
                        state.picard_iterations,
                        state.picard_residuum,
                        state.dt,
                        time.time(),
                        state.B_n if CONFIG["save_B"] else None)

        if i % CONFIG["print_every"] == 0:
            print(
                f"Iteration {i}, u norm: {state.velocity_norm:.2e}, force norm: {state.force_norm:.2e}")
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