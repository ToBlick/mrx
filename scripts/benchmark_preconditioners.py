"""Benchmark Jacobi vs. Kronecker mass preconditioners.

Builds a moderately-resolved DeRham sequence on the donut-shaped solid torus,
projects N random right-hand sides through ``M_k^{-1}`` with each
preconditioner, and reports the average and standard deviation of CG iteration
counts and wall times for k = 0, 1, 2, 3 (with and without the Dirichlet
extraction).

Run from the project root:

    .venv/bin/python scripts/benchmark_mass_preconditioner.py
"""
# %%
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mrx.operators import (apply_hodge_laplacian_preconditioner,
                           apply_mass_matrix, apply_mass_matrix_preconditioner,
                           assemble_all_operators)

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.io import project_sampled_field  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.nullspace import get_nullspace  # noqa: E402
from mrx.operators import apply_hodge_laplacian  # noqa: E402
from mrx.solvers import solve_singular_cg  # noqa: E402

# ---------------------------------------------------------------------------
# Sequence setup (mirrors test/conftest.py but lives outside pytest).
# ---------------------------------------------------------------------------

N = 5
P = 3
TYPES = ("clamped", "periodic", "periodic")
TORUS_EPSILON = 1 / 3
TORUS_R0 = 1.0
BETTI = (1, 1, 0, 0)
NUM_RHS = 10
TOL = 1e-9
MAXITER = 2000

N_MODES = 8


def build_sequence():
    ns = (N, N, N)
    ps = (P, P, P)
    seq = DeRhamSequence(
        ns, ps, 2 * P, TYPES, lambda x: x, polar=True,
        tol=TOL, maxiter=MAXITER, betti_numbers=BETTI,
    )
    seq = DeRhamSequence(
        ns, ps, 2 * P, TYPES, polar=True,
        tol=TOL, maxiter=MAXITER, betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()

    F_ana = toroid_map(epsilon=TORUS_EPSILON, R0=TORUS_R0)
    n_sample = 40
    r = jnp.linspace(0.0, 1.0, n_sample)
    theta = jnp.linspace(0.0, 1.0, n_sample)
    zeta = jnp.linspace(0.0, 1.0, n_sample)
    ri, thetai, zetai = jnp.meshgrid(r, theta, zeta, indexing="ij")
    pts = jnp.stack([ri.ravel(), thetai.ravel(), zetai.ravel()], axis=1)
    samples = jax.vmap(F_ana)(pts)
    coeffs = jnp.stack([
        project_sampled_field(
            (r, theta, zeta), samples[:, i], seq, k=0,
            dirichlet=False, reference_domain=True,
        )
        for i in range(3)
    ], axis=0)
    seq.set_spline_map(coeffs)
    operators = assemble_all_operators(seq, seq.geometry)
    seq.operators = operators
    seq._compute_nullspaces(BETTI, eps=seq.tol**0.5)
    operators = seq.operators
    return seq, operators


# ---------------------------------------------------------------------------
# Solver wrapper that exposes the CG iteration count.
# ---------------------------------------------------------------------------

def make_solve(seq, operators, k, dirichlet, kind):
    def A_mv(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    if kind == "none":
        def M_mv(x):
            return x
    else:
        def M_mv(x):
            return apply_mass_matrix_preconditioner(
                seq, operators, x, k, dirichlet=dirichlet, kind=kind)

    @jax.jit
    def solve(b):
        x, info = solve_singular_cg(
            A_mv, b, mass_matvec=A_mv, precond_matvec=M_mv,
            tol=TOL, maxiter=MAXITER,
        )
        # info < 0 ⇒ converged, |info| = iterations.
        return x, jnp.abs(info)
    return solve


def dof_size(seq, k, dirichlet):
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    return e.shape[0]


# ---------------------------------------------------------------------------
# Benchmark loop.
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_time_ms: float
    std_time_ms: float
    avg_residual: float = float('nan')


def time_solve(solve, rhs_batch, residual_fn=None):
    # Warm-up + JIT compile.
    x, it = solve(rhs_batch[0])
    jax.block_until_ready(x)

    iters = []
    times = []
    residuals = []
    for b in rhs_batch:
        t0 = time.perf_counter()
        x, it = solve(b)
        jax.block_until_ready(x)
        dt = time.perf_counter() - t0
        iters.append(int(it))
        times.append(dt * 1e3)
        if residual_fn is not None:
            residuals.append(float(residual_fn(x, b)))
    iters = jnp.array(iters)
    times = jnp.array(times)
    return Stats(
        avg_iters=float(iters.mean()),
        std_iters=float(iters.std()),
        max_iters=int(iters.max()),
        avg_time_ms=float(times.mean()),
        std_time_ms=float(times.std()),
        avg_residual=(
            float(jnp.mean(jnp.asarray(residuals))) if residuals else float('nan')
        ),
    )


def benchmark_mass(seq, operators):
    key = jax.random.PRNGKey(0)
    print(f"\nResolution: N={N}, P={P}; {NUM_RHS} RHS per case; "
          f"tol={TOL}, maxiter={MAXITER}\n")
    header = (f"{'k':>2}  {'dbc':>5}  {'n_dof':>6}  {'precond':>10}  "
              f"{'avg_it':>8}  {'std_it':>8}  {'max_it':>7}  "
              f"{'avg_ms':>9}  {'std_ms':>8}  {'avg_resM':>10}  {'speedup':>8}")
    print(header)
    print("-" * len(header))

    for k in (0, 1, 2, 3):
        for dirichlet in (False, True):
            n = dof_size(seq, k, dirichlet)
            key, sub = jax.random.split(key)
            rhs_batch = jax.random.normal(sub, (NUM_RHS, n))

            def A_residual(x, b, k=k, dirichlet=dirichlet):
                A = lambda v: apply_mass_matrix(seq, operators, v, k, dirichlet=dirichlet)
                r = A(x) - b
                rn = seq.l2_norm(r, k, dirichlet=dirichlet)
                bn = seq.l2_norm(b, k, dirichlet=dirichlet)
                return rn / jnp.where(bn > 0.0, bn, 1.0)

            results = {}
            for kind in ("jacobi", "kronecker"):
                solve = make_solve(seq, operators, k, dirichlet, kind)
                results[kind] = time_solve(solve, rhs_batch, residual_fn=A_residual)

            jac = results["jacobi"]
            kro = results["kronecker"]
            speedup = jac.avg_time_ms / \
                kro.avg_time_ms if kro.avg_time_ms > 0 else float('nan')
            for kind, s in results.items():
                if kind == "jacobi":
                    speed_col = "-"
                elif kind == "kronecker":
                    speed_col = f"x{speedup:5.2f}"
                else:
                    speed_col = ""
                print(
                    f"{k:>2}  {str(dirichlet):>5}  {n:>6}  {kind:>10}  "
                    f"{s.avg_iters:>8.1f}  {s.std_iters:>8.2f}  "
                    f"{s.max_iters:>7d}  {s.avg_time_ms:>9.2f}  "
                    f"{s.std_time_ms:>8.2f}  {s.avg_residual:>10.2e}  "
                    f"{speed_col:>8}"
                )
            print()

# %%


def main():
    print("Building sequence and operators...")
    t0 = time.perf_counter()
    seq, operators = build_sequence()
    print(f"  done in {time.perf_counter() - t0:.1f} s")
    # benchmark_mass(seq, operators)
    benchmark_hodge(seq, operators)

# %%

# %%

# ---------------------------------------------------------------------------
# Hodge-Laplacian preconditioner benchmark.
#
# Delegates to ``seq.apply_inverse_hodge_laplacian``, which uses CG for
# k = 0 and saddle-point MINRES for k >= 1 internally, varying only the
# upper-block preconditioner (``'none' | 'jacobi' | 'tensor'``).
# On the closed torus k = 3 with and without Dirichlet b.c. are identical,
# so we only benchmark ``dirichlet=True`` for k = 3.
# ---------------------------------------------------------------------------


def make_hodge_solve(seq, k, dirichlet, kind):
    @jax.jit
    def solve(b):
        u, info = seq.apply_inverse_hodge_laplacian(
            b, k, dirichlet=dirichlet, preconditioner=kind, return_info=True)
        return u, jnp.abs(info)
    return solve


def _smooth_scalar_rhs_batch(seq, key, k, dirichlet, num_rhs, n_modes=3):
    """Project ``num_rhs`` smooth random scalar functions onto the k-form space.

    The test function is a random low-mode expansion

        f(r, theta, zeta) = sum_{i,j,l} a_{ijl} sin(pi*i*r) * phi_j(theta) * phi_l(zeta)

    with ``phi_m`` a random cos/sin Fourier mode of frequency ``m``.  The
    ``sin(pi*i*r)`` factor vanishes at ``r = 0, 1``, matching the radial
    Dirichlet condition used on the torus.  Applying the projector with
    the correct extraction operator then enforces the remaining DoF-level
    BCs.  Using smooth RHSs (vs. pure random DoFs) is a fairer probe of
    preconditioner quality, since realistic right-hand sides in
    downstream applications are not white-noise in DoF space.
    """
    if k == 0:
        proj = lambda f: seq.load(f, 0, dirichlet=dirichlet)
    elif k == 3:
        proj = lambda f: seq.load(f, 3, dirichlet=dirichlet)
    else:
        raise ValueError(
            f"smooth RHS generator only supports k in (0, 3), got {k}")
    mr = jnp.arange(1, n_modes + 1, dtype=jnp.float64)           # 1..n_modes
    mt = jnp.arange(n_modes, dtype=jnp.float64)                  # 0..n_modes-1
    mz = jnp.arange(n_modes, dtype=jnp.float64)
    # coeff shape: (n_modes, n_modes, n_modes, 4) for (cos/sin)_theta x (cos/sin)_zeta
    shape = (num_rhs, n_modes, n_modes, n_modes, 4)
    all_coeffs = jax.random.normal(key, shape)

    def make_f(c):
        def f(x):
            r, th, ze = x[0], x[1], x[2]
            br = jnp.sin(jnp.pi * mr * r)                        # (n_modes,)
            bt = jnp.stack([jnp.cos(2 * jnp.pi * mt * th),
                            jnp.sin(2 * jnp.pi * mt * th)], axis=0)  # (2, n_modes)
            bz = jnp.stack([jnp.cos(2 * jnp.pi * mz * ze),
                            jnp.sin(2 * jnp.pi * mz * ze)], axis=0)  # (2, n_modes)
            # c[i,j,l, s] with s in {cc, cs, sc, ss}
            cc = jnp.einsum('ijl,i,j,l->', c[..., 0], br, bt[0], bz[0])
            cs = jnp.einsum('ijl,i,j,l->', c[..., 1], br, bt[0], bz[1])
            sc = jnp.einsum('ijl,i,j,l->', c[..., 2], br, bt[1], bz[0])
            ss = jnp.einsum('ijl,i,j,l->', c[..., 3], br, bt[1], bz[1])
            return cc + cs + sc + ss
        return f

    rhs_list = [proj(make_f(all_coeffs[i])) for i in range(num_rhs)]
    return jnp.stack(rhs_list)


def benchmark_hodge(seq, operators):
    print("\n--- Hodge Laplacian preconditioner ---")
    print("k = 0 uses CG internally; k = 3 uses saddle-point MINRES.")
    print(f"Resolution: N={N}, P={P}; {NUM_RHS} RHS per case; "
          f"tol={TOL}, maxiter={MAXITER}\n")
    print("RHS: smooth random functions projected onto the form space.\n")
    header = (f"{'k':>2}  {'dbc':>5}  {'n_dof':>6}  {'precond':>10}  "
              f"{'avg_it':>8}  {'std_it':>8}  {'max_it':>7}  "
              f"{'avg_ms':>9}  {'std_ms':>8}  {'avg_resM':>10}  {'speedup':>8}")
    print(header)
    print("-" * len(header))

    key = jax.random.PRNGKey(1)
    cases = [(0, False), (0, True), (3, False), (3, True)]
    for k, dirichlet in cases:
        n = dof_size(seq, k, dirichlet)
        key, sub = jax.random.split(key)
        rhs_batch = _smooth_scalar_rhs_batch(
            seq, sub, k, dirichlet, NUM_RHS, N_MODES)
        # Project out the nullspace component so the RHS lies in range(L_k).
        vs = get_nullspace(operators, k, dirichlet)
        if vs.shape[0] > 0:
            def _deflate(b):
                for v in vs:
                    b = b - jnp.dot(v, b) * apply_mass_matrix(
                        seq, operators, v, k, dirichlet=dirichlet)
                return b
            rhs_batch = jax.vmap(_deflate)(rhs_batch)

        def L_residual(x, b, k=k, dirichlet=dirichlet):
            A = lambda v: apply_hodge_laplacian(
                seq, operators, v, k, dirichlet=dirichlet, tol=TOL, maxiter=MAXITER,
            )
            r = A(x) - b
            rn = seq.l2_norm(r, k, dirichlet=dirichlet)
            bn = seq.l2_norm(b, k, dirichlet=dirichlet)
            return rn / jnp.where(bn > 0.0, bn, 1.0)

        results = {}
        kinds = ("jacobi", "tensor")
        for kind in kinds:
            solve = make_hodge_solve(seq, k, dirichlet, kind)
            try:
                results[kind] = time_solve(solve, rhs_batch, residual_fn=L_residual)
            except (ValueError, NotImplementedError) as exc:
                results[kind] = exc

        jac = results["jacobi"]
        tensor_result = results["tensor"]
        if isinstance(jac, Exception) or isinstance(tensor_result, Exception):
            speedup = float('nan')
        else:
            speedup = jac.avg_time_ms / \
                tensor_result.avg_time_ms if tensor_result.avg_time_ms > 0 else float('nan')
        for kind, s in results.items():
            if isinstance(s, Exception):
                print(
                    f"{k:>2}  {str(dirichlet):>5}  {n:>6}  {kind:>10}  "
                    f"-- not available: {s} --"
                )
                continue
            if kind == "jacobi":
                speed_col = "-"
            elif kind == "tensor":
                speed_col = f"x{speedup:5.2f}"
            else:
                speed_col = ""
            print(
                f"{k:>2}  {str(dirichlet):>5}  {n:>6}  {kind:>10}  "
                f"{s.avg_iters:>8.1f}  {s.std_iters:>8.2f}  "
                f"{s.max_iters:>7d}  {s.avg_time_ms:>9.2f}  "
                f"{s.std_time_ms:>8.2f}  {s.avg_residual:>10.2e}  "
                f"{speed_col:>8}"
            )
        print()


if __name__ == "__main__":
    main()
