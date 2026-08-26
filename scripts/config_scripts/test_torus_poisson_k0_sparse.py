"""Convergence study for the k=0 Hodge Laplacian with Dirichlet conditions on a torus.

Solve ``-Δ₀ u = f`` on the axisymmetric toroid map with homogeneous
Dirichlet boundary conditions, one resolution after another, and record the
relative L2 error, the CG iteration count, the operator sparsity and every
assembly and solve timing. The manufactured solution is

    u(r, χ, z) = 1/4 (r² - r⁴) cos(2πz)

with source, for minor radius ``a = epsilon`` and ``R = 1 + a r cos(2πχ)``,

    f = cos(2πz) [ -(1 - 4r²)/a² - (r/2 - r³) cos(2πχ)/(aR) + (r² - r⁴)/(4R²) ].

The CG solve runs on the k=0 stiffness with the metric-lumping Laplacian
preconditioner. k=0 DBC has no harmonic forms, so nothing is deflated.
``quad_order`` below ``2*p`` raises ``ValueError``.

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
        selects ``2*p + quad_order_offset``. Default ``None``.
    quad_order_offset (int): Offset on ``2*p``. Dataclass default 4; the
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
    load_frame (str): Present in the config, not read by this script.

Usage:
    Single run, all listed n in one process::

        python -u scripts/config_scripts/test_torus_poisson_k0_sparse.py p=3
        python -u scripts/config_scripts/test_torus_poisson_k0_sparse.py p=2 n=16 precision=float32

    Single GPU job through ``slurm/run.sh``::

        SCRIPT=scripts/config_scripts/test_torus_poisson_k0_sparse.py ARGS="p=3 n=16" \
            JOB_NAME=pois_k0 MEM_GB=80 TIMEOUT_MIN=120 bash slurm/run.sh

    Multirun, one submitit job per (p, n) pair. Needs ``SLURM_ACCOUNT``,
    ``SLURM_PARTITION`` and ``MRX_ROOT`` exported; the launcher allots one
    GPU, 80 GB and 120 min per job::

        python scripts/config_scripts/test_torus_poisson_k0_sparse.py -m p=2,3 n=8,16

Runtime:
    One GPU, p=3, ``logs/mergepois_16819379.out`` (2026-08): TOTAL 116 s at
    n=6, 152 s at n=8, 162 s at n=10 per resolution, JIT included. Memory
    not measured; the multirun launcher allots 80 GB.

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
from mrx.operators import (
    assemble_incidence_operators,
    assemble_metric_lumping_laplacian_preconditioner,
    assemble_projection_operators,
)
from mrx.quadrature import evaluate_at_xq


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
π = jnp.pi


def u(x: jnp.ndarray) -> jnp.ndarray:
    """Exact solution: u(r,χ,z) = 1/4 (r² - r⁴) cos(2πz)."""
    r, χ, z = x
    return 1 / 4 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def make_f(a: float):
    """Return the source term for minor radius *a*."""
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, χ, z = x
        R = 1 + a * r * jnp.cos(2 * jnp.pi * χ)
        return (
            jnp.cos(2 * jnp.pi * z)
            * (
                -1 / a**2 * (1 - 4 * r**2)
                - 1 / (a * R) * (r / 2 - r**3) * jnp.cos(2 * jnp.pi * χ)
                + 1 / 4 * (r**2 - r**4) / R**2
            )
            * jnp.ones(1)
        )
    return f


def exact_u_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    """Evaluate the exact scalar solution on the quadrature grid cheaply."""
    u_r = 0.25 * (seq.quad.x_x**2 - seq.quad.x_x**4)
    u_z = jnp.cos(2 * π * seq.quad.x_z)
    values = jnp.ones((seq.quad.ny, 1, 1)) * \
        u_r[None, :, None] * u_z[None, None, :]
    return values.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_error(n: int, p: int, epsilon: float,
                  solver_tol: float, cg_maxiter: int,
                  quad_order: int | None,
                  quad_order_offset: int):
    """Run the sparse Poisson solve and return (error, timings dict).

    Resolution convention: ``ns = (n, 2*n, n)`` (the toroidal direction
    carries twice the angular resolution of the radial / vertical
    directions).
    """
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order
    if q < 2 * p:
        raise ValueError(
            f"quad_order must satisfy q >= 2*p; got q={q}, p={p}"
        )
    F = toroid_map(epsilon=epsilon)
    f = make_f(epsilon)

    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=solver_tol, maxiter=cg_maxiter,
    )
    seq.set_map(F)
    timings["DeRhamSequence.__init__"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0

    # Each assembly / solve is timed twice: the first call includes XLA
    # compilation (``_compile``), the second is execution-only on the cached
    # compiled kernels (``_exec``).

    # K0 = G0^T M1 G0 applies M1 matrix-free (sum factorization); M1 is never
    # assembled or stored. The block-Jacobi Laplacian atom below is the ONLY
    # preconditioner built for the solve; it does NOT consume any mass
    # surgery/block data, so no mass preconditioner is assembled here.
    #
    # The incidence operators ARE required, and were the k=0 study's failure:
    # the atom's constructor probes its core block via probe_core_block ->
    # apply_hodge_laplacian_approx -> apply_stiffness, which raises
    # "Incidence operator G0 is required to apply K0" without them. The
    # retired tensor preconditioner this script used before 7cead35 did not
    # take that path, so the swap introduced the dependency silently. This is
    # the only one of the nine poisson scripts that had omitted the call.
    #
    # NOTE the atom also NEEDS n >= p + 2 (see
    # assemble_metric_lumping_laplacian_preconditioner's docstring): below that
    # component_factors goes non-finite and numpy raises LinAlgError from
    # inside eigvals. Keep the configured sweep at or above that floor.
    #
    # `assemble_hodge_laplacian` is intentionally not called here -- operator
    # assembly is decoupled from preconditioner construction.
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = seq.set_operators(
        assemble_metric_lumping_laplacian_preconditioner(
            seq, ops, ks=(0,), dirichlets=(True, False),
        )
    )
    jax.block_until_ready(ops)
    timings["build_hodge_preconditioners_0_compile"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = seq.set_operators(
        assemble_metric_lumping_laplacian_preconditioner(
            seq, ops, ks=(0,), dirichlets=(True, False),
        )
    )
    jax.block_until_ready(ops)
    timings["build_hodge_preconditioners_0_exec"] = time.perf_counter() - t0

    # Sparsity diagnostics. K0 = G0^T M1 G0 is never materialised (the solve
    # applies K via composed matvecs, with M1 applied matrix-free), and M0 is
    # also applied matrix-free, so no mass matrix is stored.
    sparsity = {}

    t0 = time.perf_counter()
    rhs = seq.load(f, 0, dirichlet=True)
    jax.block_until_ready(rhs)
    timings["P0_dbc(f)"] = time.perf_counter() - t0

    # k=0 with DBC has no nullspace (default betti_numbers=(1,1,0,0))
    t0 = time.perf_counter()
    u_hat, cg_info = seq.apply_inverse_laplacian(
        rhs, 0, dirichlet=True, return_info=True)
    jax.block_until_ready(u_hat)
    timings["inverse_hodge_laplacian_compile"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    u_hat, cg_info = seq.apply_inverse_laplacian(
        rhs, 0, dirichlet=True, return_info=True)
    jax.block_until_ready(u_hat)
    timings["inverse_hodge_laplacian_exec"] = time.perf_counter() - t0

    cg_info_int = int(cg_info)
    cg_iters = abs(cg_info_int)
    cg_converged = cg_info_int < 0
    residual = seq.apply_laplacian(u_hat, 0, dirichlet=True) - rhs
    final_rel_residual = float(
        jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))

    t0 = time.perf_counter()
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    comp_info_0, comp_shapes_0 = seq._form_comp_info(0)
    u_h_jk = evaluate_at_xq(seq.e0_dbc_T @ u_hat, comp_info_0, comp_shapes_0,
                            quad_shape, 1)
    u_i = exact_u_at_quad(seq)
    df = u_i - u_h_jk
    L2_df = jnp.einsum("ik,ik,i,i->", df, df, seq.jacobian_j, seq.quad.w)
    L2_f = jnp.einsum("ik,ik,i,i->", u_i, u_i, seq.jacobian_j, seq.quad.w)
    jax.block_until_ready(L2_f)
    timings["error_computation"] = time.perf_counter() - t0

    error = float((L2_df / L2_f) ** 0.5)

    timings["TOTAL"] = sum(timings.values())
    return {
        "n": n,
        "p": p,
        "q": q,
        "error": error,
        "cg_iters": cg_iters,
        "cg_converged": cg_converged,
        "final_rel_residual": final_rel_residual,
        "timings": timings,
        "sparsity": sparsity,
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
    print(f"epsilon type: {type(cfg.epsilon).__name__} value: {cfg.epsilon!r}")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    print(f"n: {ns!r}")
    print(f"solver_tol type: {type(cfg.solver_tol).__name__} value: {cfg.solver_tol!r}")

    p = cfg.p
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"Running sparse Poisson solve: n={ns}, p={p}")
    if cfg.quad_order is None:
        print(f"Quadrature order: q = 2*p + {cfg.quad_order_offset}")
    else:
        print(f"Quadrature order: q = {cfg.quad_order}")
    print(f"JAX devices: {jax.devices()}")
    print(
        f"Batch sizes: inner={mrx.MAP_BATCH_SIZE_INNER}, outer={mrx.MAP_BATCH_SIZE_OUTER}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")

    results = []
    for n in ns:
        print(f"\n{'='*60}")
        print(f"  n={n}, p={p}")
        print(f"{'='*60}")

        result = compute_error(
            n,
            p,
            cfg.epsilon,
            cfg.solver_tol,
            cfg.cg_maxiter,
            cfg.quad_order,
            cfg.quad_order_offset,
        )
        results.append(result)

        print(f"\n  --- Timings (n={n}, p={p}) ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<30s} {dt:8.3f}s")
        print("\n  --- Sparsity ---")
        for label, val in result["sparsity"].items():
            print(f"  {label:.<30s} {val}")
        print(f"\n  Relative L2 error: {result['error']:.6e}")
        print(f"  CG iters: {result['cg_iters']}  converged: {result['cg_converged']}"
              f"  final ||K0 u - b||/||b||: {result['final_rel_residual']:.3e}")

        # Write incrementally so results are not lost if a later n OOMs
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Results saved to {outfile}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Summary (p={p})")
    print(f"{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'total_time':>10s}")
    for r in results:
        print(
            f"  {r['n']:5d}  {r['error']:12.6e}  {r['timings']['TOTAL']:10.3f}s")


if __name__ == "__main__":
    main()
