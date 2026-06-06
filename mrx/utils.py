# %%
import os
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

import mrx

# Re-export symbols that have moved to other modules.
# This keeps old import sites (scripts, deprecated tests) working.
from mrx.differential_forms import (  # noqa: F401
    curl, det33, div, double_map, grad, inv33, jacobian_determinant, l2_product,
)
from mrx.quadrature import evaluate_at_xq, integrate_against  # noqa: F401
from mrx.solvers import get_smallest_ev_pair  # noqa: F401

# Diagonal utilities have moved to mrx.preconditioners.
from mrx.preconditioners import (  # noqa: F401
    _bcsr_to_coo_indices, _build_diag_EAET_plan, _coo_host, _coo_indices_host,
    diag_EAET, diag_EAET_direct, diag_EAET_matvec, diag_EGtMGEt_direct,
    diag_matvec, diag_schur_complement, extract_diag_vector,
)


def is_running_in_github_actions():
    """
    Checks if the current Python script is running within a GitHub Actions environment.
    """
    return os.getenv("GITHUB_ACTIONS") == "true"

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


# Default trace dictionary for the relaxation loop

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

    from mrx.relaxation import append_to_trace_dict
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
