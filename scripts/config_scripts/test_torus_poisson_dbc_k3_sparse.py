"""Convergence study for the k=3 Hodge Laplacian with Dirichlet conditions on a torus.

Solve ``-Δ₃ ω = f`` on the axisymmetric toroid map with Dirichlet
conditions, one resolution after another, and record the relative physical
L² error (proxy-scalar comparison), the MINRES iteration count and every
timing. The manufactured solution and source are

    ω₃ = cos(2πζ) · J dr ∧ dχ ∧ dζ    (proxy scalar cos(2πζ))
    f₃ = same proxy scalar as f₀ = cos(2πζ)/R².

k=3 DBC has a one-dimensional harmonic nullspace spanned by the constant
volume form ``J dr ∧ dχ ∧ dζ`` (Hodge dual of the constant 0-form), which
the saddle-point MINRES solve deflates. ``load_frame='ref'`` passes ``f₃``
as the coefficient ``A = f₀ J`` of ``A dr ∧ dχ ∧ dζ``; ``load_frame='phys'``
passes the physical scalar ``f₀`` (the k=0 source).

Diagnostics logged per resolution:

- relative physical L² error,
- MINRES iteration count and convergence flag,
- nullspace residual ``||L₃ h||₂``,
- ``||D₂ᵀ h||₂`` (co-derivative, expected ≈ 0); no curl at k=3.

Configuration:
    Hydra config ``conf/config_poisson_test.yaml``, schema
    ``mrx.config.PoissonTestConfig``. Override any key as ``key=value``.

    n (list[int] | int): Radial resolutions, run one after another; the
        grid is ``ns = (n, 2n, n)``. An int runs a single resolution.
        Default ``[8, 12, 16, 24, 32, 48, 64]``.
    p (int): Spline degree in every direction. Default 3.
    epsilon (float): Minor radius of ``toroid_map`` (major radius 1).
        Default 1/3.
    quad_order (int | None): Gauss quadrature order per direction. ``None``
        selects ``p + 1 + quad_order_offset``. Default ``None``.
    quad_order_offset (int): Offset on ``p + 1``. Dataclass default 4; the
        yaml sets 0.
    cg_maxiter (int): Iteration cap of the Laplacian solve. Dataclass
        default 100000; the yaml sets 50000.
    solver_tol (float | None): Relative residual tolerance of every
        iterative solve in the sequence. ``None`` selects ``sqrt(eps)`` of
        the working precision; the yaml sets 1e-9.
    precision (str): ``float64`` (default) or ``float32``. Read from argv
        and exported as ``MRX_DTYPE`` before ``mrx`` is imported.
    map_batch_size_inner (int): ``mrx.MAP_BATCH_SIZE_INNER``; 0 means
        ``vmap``. Default 0.
    map_batch_size_outer (int | None): ``mrx.MAP_BATCH_SIZE_OUTER``;
        ``None`` means no batching. Default ``None``.
    load_frame (str): ``'ref'`` passes the reference components of the
        source, ``'phys'`` the physical field (see ``mrx.projectors.load``).
        Default ``'ref'``.

Usage:
    Single run, all listed n in one process::

        python -u scripts/config_scripts/test_torus_poisson_dbc_k3_sparse.py p=3
        python -u scripts/config_scripts/test_torus_poisson_dbc_k3_sparse.py p=2 n=16 precision=float32

    Single GPU job through ``slurm/run.sh``::

        SCRIPT=scripts/config_scripts/test_torus_poisson_dbc_k3_sparse.py ARGS="p=3 n=16" \
            JOB_NAME=pois_dbc_k3 MEM_GB=80 TIMEOUT_MIN=120 bash slurm/run.sh

    Multirun, one submitit job per (p, n) pair. Needs ``SLURM_ACCOUNT``,
    ``SLURM_PARTITION`` and ``MRX_ROOT`` exported; the launcher allots one
    GPU, 80 GB and 120 min per job::

        python scripts/config_scripts/test_torus_poisson_dbc_k3_sparse.py -m p=2,3 n=8,16

Runtime:
    Not measured. The multirun launcher allots one GPU, 80 GB and 120 min
    per job.

Output:
    Single run: ``outputs/<date>/<time>/result.json``, a list with one entry
    per n, rewritten after every n so an OOM at a later n keeps the earlier
    results. Multirun: ``multirun/<date>/<time>/<job>/result.json``. Through
    ``slurm/run.sh`` the stdout log is
    ``outputs/<JOB_NAME>/<date>/<time>/<JOB_NAME>.log``.
"""
import json
import os
import time

import sys
# The working precision is chosen before mrx is imported; hydra only hands
# the config over inside main(), so the override is read from argv here.
os.environ["MRX_DTYPE"] = next(
    (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("precision=")),
    os.environ.get("MRX_DTYPE", "float64"))

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import mrx
import mrx.config  # noqa: F401 — register structured configs in ConfigStore
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.nullspace import get_nullspace, init_nullspaces, _set_null
from mrx.operators import (
    assemble_incidence_operators,
    assemble_metric_lumping_laplacian_preconditioner,
)
from mrx.quadrature import evaluate_at_xq


# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
π = jnp.pi
BETTI = (1, 1, 0, 0)
K = 3
DIRICHLET = True


# ---------------------------------------------------------------------------
# Source functions
# ---------------------------------------------------------------------------
def make_f3_phys(a: float):
    """Physical proxy scalar f₀ = cos(2πζ)/R².  Passed with frame='phys'."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return jnp.cos(2 * π * z) / R**2 * jnp.ones(1)
    return f


def make_f3_ref(a: float, F):
    """Ref coefficient A = f₀·J in  A dr∧dχ∧dζ.  Passed with frame='ref'."""
    DF = jax.jacfwd(F)
    f_phys = make_f3_phys(a)
    def f(x):
        J = jnp.linalg.det(DF(x))
        return f_phys(x) * J
    return f


# ---------------------------------------------------------------------------
# Exact solution (proxy scalar)
# ---------------------------------------------------------------------------
def u3_exact(x):
    """Exact 3-form proxy scalar: cos(2πζ)."""
    return jnp.cos(2 * π * x[2]) * jnp.ones(1)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_error(n: int, p: int, epsilon: float,
                  solver_tol: float, cg_maxiter: int,
                  quad_order, quad_order_offset: int,
                  load_frame: str):
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = p + 1 + quad_order_offset if quad_order is None else quad_order

    F = toroid_map(epsilon=epsilon)
    f3 = make_f3_ref(epsilon, F) if load_frame == 'ref' else make_f3_phys(epsilon)

    # --- Sequence setup ------------------------------------------------
    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=solver_tol, maxiter=cg_maxiter,
        betti_numbers=BETTI,
    )
    seq.set_map(F)
    timings["init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0

    # --- Assembly (compile pass) ----------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_metric_lumping_laplacian_preconditioner(seq, ops, ks=(3,), dirichlets=(True,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    # --- Assembly (exec pass) ------------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_metric_lumping_laplacian_preconditioner(seq, ops, ks=(3,), dirichlets=(True,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_exec"] = time.perf_counter() - t0

    # --- Nullspace: k=3 DBC constant volume form (set analytically) ----
    # The constant primal 3-form lives in ker(K₃^DBC) exactly;
    # compute M₃⁻¹·1 using the assembled tensor preconditioner and normalise.
    t0 = time.perf_counter()
    ops = init_nullspaces(seq, seq.get_operators(), BETTI)
    ones3 = jnp.ones(seq.n3_dbc)
    v3 = seq.apply_inverse_mass_matrix(ones3, 3, dirichlet=True)
    norm3 = seq.l2_norm(v3, 3, dirichlet=True)
    ops = _set_null(ops, 3, True, (v3 / norm3)[None, :])
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    # Diagnostics
    h = get_nullspace(seq.get_operators(), K, DIRICHLET)[0]
    Lh = seq.apply_laplacian(h, K, dirichlet=DIRICHLET)
    null_residual = float(jnp.linalg.norm(Lh))
    div_val = seq.apply_derivative_matrix(
        h, K - 1, dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)
    null_div_norm = float(jnp.linalg.norm(div_val))
    timings["nullspace"] = time.perf_counter() - t0

    # --- RHS -----------------------------------------------------------
    t0 = time.perf_counter()
    rhs = seq.load(f3, K, dirichlet=DIRICHLET, frame=load_frame)
    jax.block_until_ready(rhs)
    timings["load_rhs"] = time.perf_counter() - t0

    # --- Solve (compile + exec) ----------------------------------------
    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(
        rhs, K, dirichlet=DIRICHLET, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(
        rhs, K, dirichlet=DIRICHLET, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_exec"] = time.perf_counter() - t0

    iters = abs(int(info))
    converged = int(info) < 0

    # --- Error: physical scalar via proxy (ref coeff / J) ---------------
    t0 = time.perf_counter()
    comp_info, comp_shapes = seq._form_comp_info(K)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    u_h_jk = evaluate_at_xq(seq.e3_dbc_T @ u_hat, comp_info, comp_shapes, quad_shape, 1)
    u_h_phys = u_h_jk / seq.jacobian_j[:, None]  # divide by J to get proxy scalar
    u_ex = jax.vmap(u3_exact)(seq.quad.x)
    diff = u_h_phys - u_ex
    L2_diff = jnp.einsum("ik,ik,i,i->", diff, diff, seq.jacobian_j, seq.quad.w)
    L2_norm = jnp.einsum("ik,ik,i,i->", u_ex, u_ex, seq.jacobian_j, seq.quad.w)
    jax.block_until_ready(L2_norm)
    timings["error"] = time.perf_counter() - t0

    error = float(jnp.sqrt(L2_diff / L2_norm))
    timings["TOTAL"] = sum(timings.values())

    return {
        "n": n, "p": p, "q": q,
        "error": error,
        "iters": iters,
        "converged": converged,
        "null_residual": null_residual,
        "null_div_norm": null_div_norm,
        "load_frame": load_frame,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    print(f"precision: {mrx.DTYPE}  solver_tol: {cfg.solver_tol}")
    if cfg.precision != str(mrx.DTYPE):
        raise ValueError(f"precision={cfg.precision} but mrx runs in {mrx.DTYPE}; "
                         "MRX_DTYPE was not set before import")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    p, load_frame = cfg.p, cfg.load_frame
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"k=3 DBC Poisson | frame={load_frame} | n={ns} p={p} ε={cfg.epsilon}")
    print(f"JAX devices: {jax.devices()}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")

    results = []
    for n in ns:
        print(f"\n{'='*60}\n  n={n}, p={p}\n{'='*60}")
        result = compute_error(
            n, p, cfg.epsilon, cfg.solver_tol, cfg.cg_maxiter,
            cfg.quad_order, cfg.quad_order_offset, load_frame,
        )
        results.append(result)
        print(f"\n  --- Timings (n={n}, p={p}) ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<32s} {dt:8.3f}s")
        print(f"\n  Relative L2 error     : {result['error']:.6e}")
        print(f"  MINRES iters          : {result['iters']}  converged={result['converged']}")
        print(f"  Nullspace residual    : {result['null_residual']:.3e}  (||L₃ h||)")
        print(f"  Nullspace div  norm   : {result['null_div_norm']:.3e}  (||D₂ᵀ h||)")
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Saved → {outfile}")

    print(f"\n{'='*60}\n  Summary (p={p}, frame={load_frame})\n{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'iters':>6s}  {'||div||':>10s}")
    for r in results:
        print(f"  {r['n']:5d}  {r['error']:12.6e}  {r['iters']:6d}  {r['null_div_norm']:10.3e}")


if __name__ == "__main__":
    main()
