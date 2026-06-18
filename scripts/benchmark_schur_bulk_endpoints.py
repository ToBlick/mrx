from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    MassPreconditionerSpec,
    _build_chebyshev_apply_preconditioner,
    _build_mass_surgery_wrapped_preconditioner_apply,
    _estimate_chebyshev_lanczos_bounds_apply,
    _hodge_diaginv,
    _mass_diaginv,
    _apply_k0_tensor_hodge_bulk_inverse,
    _apply_k0_tensor_hodge_bulk_to_surgery_coupling,
    _apply_k0_tensor_hodge_preconditioner,
    _apply_k0_tensor_hodge_surgery_to_bulk_coupling,
    apply_mass_tensor_preconditioner_ops,
    apply_mass_matrix,
    apply_stiffness,
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
)
from mrx.preconditioners import (
    select_boundary_data,
)
from mrx.solvers import solve_singular_cg
from test.random_fields import build_random_besov_rhs_batch

jax.config.update("jax_enable_x64", True)


NS = (8, 16, 8)
P = 3
TYPES = ("clamped", "periodic", "periodic")
ROTATING_ELLIPSE_EPS = 0.33
ROTATING_ELLIPSE_KAPPA = 1.4
ROTATING_ELLIPSE_R0 = 1.0
ROTATING_ELLIPSE_NFP = 3
BETTI = (1, 1, 0, 0)
DIRICHLET = True
NUM_RHS = 8
TOL = 1e-9
MAXITER = 1000
CHEB_STEPS = (1, 2, 4)
BENCHMARK_WHOLE_CHEB_STEPS = tuple(step for step in CHEB_STEPS if step != 1)
TENSOR_CP_KWARGS = {"tol": 1e-9, "maxiter": 100}
TENSOR_CP_KWARGS_INNER_SCHUR_OFF = {
    **TENSOR_CP_KWARGS,
    "k1_inner_schur": False,
    "k2_inner_schur": False,
}
BESOV_RHS_KWARGS = {
    "s": 1.0,
    "upper_limit": 24,
    "num_modes": 64,
    "scale": 1.0,
    "smoothness_margin": 0.0,
    "normalization_samples": 256,
}


@dataclass(frozen=True)
class BenchmarkRow:
    problem: str
    case: str
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_ms: float
    std_ms: float
    max_ms: float


def build_sequence():
    seq = DeRhamSequence(
        NS,
        (P, P, P),
        2 * P,
        TYPES,
        polar=True,
        tol=TOL,
        maxiter=MAXITER,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(
        rotating_ellipse_map(
            eps=ROTATING_ELLIPSE_EPS,
            kappa=ROTATING_ELLIPSE_KAPPA,
            R0=ROTATING_ELLIPSE_R0,
            nfp=ROTATING_ELLIPSE_NFP,
        )
    )

    operators = None
    operators = assemble_mass_operators(seq, seq.geometry, operators=operators, ks=(0, 1, 2, 3))
    tensor_inner_schur_on = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1, 2, 3),
        rank=1,
        cp_kwargs=TENSOR_CP_KWARGS,
    )
    tensor_inner_schur_off = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1, 2, 3),
        rank=1,
        cp_kwargs=TENSOR_CP_KWARGS_INNER_SCHUR_OFF,
    )
    operators = assemble_incidence_operators(seq, operators=tensor_inner_schur_on, ks=(0,))
    operators = assemble_laplacian_operators(seq, seq.geometry, operators=operators, ks=(0,))
    seq.operators = operators
    return seq, {
        "default": operators,
        "tensor_inner_schur_on": operators,
        "tensor_inner_schur_off": tensor_inner_schur_off,
    }


def build_chebyshev_apply(operator_apply, smoother_apply, size: int, *, steps: int, seed: int):
    spec = MassPreconditionerSpec(kind="chebyshev", steps=steps)
    lambda_min, lambda_max = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=seed,
    )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=steps,
        min_eig=lambda_min,
        max_eig=lambda_max,
    )


def build_mass_whole_jacobi(seq, operators, k: int):
    diaginv = _mass_diaginv(operators, k, DIRICHLET)
    return lambda rhs, inv=diaginv: inv * rhs


def build_mass_whole_cheb_jacobi(seq, operators, k: int, steps: int):
    operator_apply = lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=DIRICHLET)
    diaginv = _mass_diaginv(operators, k, DIRICHLET)
    smoother_apply = lambda rhs, inv=diaginv: inv * rhs
    size = getattr(seq, f"n{k}_dbc")
    return build_chebyshev_apply(
        operator_apply,
        smoother_apply,
        size,
        steps=steps,
        seed=10_000 + 100 * k + steps,
    )


def build_mass_schur_tensor(seq, operators, k: int):
    spec = MassPreconditionerSpec(kind="tensor", surgery_schur=True)
    return _build_mass_surgery_wrapped_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=DIRICHLET,
        spec=spec,
    )


def build_mass_schur_cheb_tensor(seq, operators, k: int, steps: int):
    spec = MassPreconditionerSpec(
        kind="chebyshev",
        steps=steps,
        surgery_schur=True,
        smoother=MassPreconditionerSpec(kind="tensor"),
    )
    return _build_mass_surgery_wrapped_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=DIRICHLET,
        spec=spec,
    )


def build_mass_tensor(seq, operators, k: int):
    return lambda rhs: apply_mass_tensor_preconditioner_ops(
        seq,
        operators,
        rhs,
        k,
        dirichlet=DIRICHLET,
    )


def build_mass_cheb_tensor(seq, operators, k: int, steps: int):
    operator_apply = lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=DIRICHLET)
    smoother_apply = lambda rhs: apply_mass_tensor_preconditioner_ops(
        seq,
        operators,
        rhs,
        k,
        dirichlet=DIRICHLET,
    )
    size = getattr(seq, f"n{k}_dbc")
    return build_chebyshev_apply(
        operator_apply,
        smoother_apply,
        size,
        steps=steps,
        seed=40_000 + 100 * k + steps,
    )


def mass_tensor_case_factories(seq, operator_bundles, k: int):
    operators_default = operator_bundles["default"]
    factories = [("jacobi", build_mass_whole_jacobi(seq, operators_default, k))]
    factories.extend(
        (
            f"whole-cheb{steps}-jacobi",
            build_mass_whole_cheb_jacobi(seq, operators_default, k, steps),
        )
        for steps in BENCHMARK_WHOLE_CHEB_STEPS
    )

    if k == 3:
        factories.append(("tensor", build_mass_tensor(seq, operators_default, k)))
        return factories

    if k == 0:
        factories.append(("schur-tensor", build_mass_schur_tensor(seq, operators_default, k)))
        return factories

    for coupling_label, operators_tensor in (
        ("inner-schur-on", operator_bundles["tensor_inner_schur_on"]),
        ("inner-schur-off", operator_bundles["tensor_inner_schur_off"]),
    ):
        factories.append(
            (
                f"schur-tensor-{coupling_label}",
                build_mass_schur_tensor(seq, operators_tensor, k),
            )
        )
    return factories


def build_k0_laplacian_whole_jacobi(seq, operators):
    diaginv = _hodge_diaginv(operators, 0, DIRICHLET)
    return lambda rhs, inv=diaginv: inv * rhs


def build_k0_laplacian_whole_cheb_jacobi(seq, operators, steps: int):
    operator_apply = lambda x: apply_stiffness(seq, operators, x, 0, dirichlet=DIRICHLET)
    diaginv = _hodge_diaginv(operators, 0, DIRICHLET)
    smoother_apply = lambda rhs, inv=diaginv: inv * rhs
    return build_chebyshev_apply(
        operator_apply,
        smoother_apply,
        seq.n0_dbc,
        steps=steps,
        seed=30_000 + steps,
    )


def build_k0_laplacian_schur_tensor(seq, operators):
    return lambda rhs: _apply_k0_tensor_hodge_preconditioner(seq, operators, rhs, dirichlet=DIRICHLET)


def _wrap_k0_hodge_schur_apply(seq, operators, bulk_apply):
    factors = select_boundary_data(operators.k0_tensor_hodge_precond, DIRICHLET, "Tensor Hodge k=0")

    def apply(rhs):
        core_size = factors.core_size
        rhs_c = rhs[:core_size]
        rhs_b = rhs[core_size:]
        y = bulk_apply(rhs_b)
        schur_rhs = rhs_c - _apply_k0_tensor_hodge_bulk_to_surgery_coupling(
            seq,
            operators,
            core_size,
            y,
            dirichlet=DIRICHLET,
        )
        z = factors.schur_inv @ schur_rhs
        x_b = y - bulk_apply(
            _apply_k0_tensor_hodge_surgery_to_bulk_coupling(
                seq,
                operators,
                core_size,
                z,
                dirichlet=DIRICHLET,
            )
        )
        return jnp.concatenate([z, x_b])

    return apply


def build_k0_laplacian_schur_cheb_tensor(seq, operators, steps: int):
    factors = select_boundary_data(operators.k0_tensor_hodge_precond, DIRICHLET, "Tensor Hodge k=0")
    core_size = factors.core_size
    bulk_size = seq.n0_dbc - core_size

    def bulk_operator_apply(x):
        full = jnp.zeros((seq.n0_dbc,), dtype=x.dtype)
        full = full.at[core_size:].set(x)
        return apply_stiffness(seq, operators, full, 0, dirichlet=DIRICHLET)[core_size:]

    bulk_apply = build_chebyshev_apply(
        bulk_operator_apply,
        lambda rhs: _apply_k0_tensor_hodge_bulk_inverse(factors, rhs),
        bulk_size,
        steps=steps,
        seed=32_000 + steps,
    )
    return _wrap_k0_hodge_schur_apply(seq, operators, bulk_apply)


def time_solve(solve, rhs_batch):
    x, it = solve(rhs_batch[0])
    jax.block_until_ready(x)

    iters = []
    times_ms = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        x, it = solve(rhs)
        jax.block_until_ready(x)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iters.append(int(it))

    iters = jnp.asarray(iters)
    times_ms = jnp.asarray(times_ms)
    return {
        "avg_iters": float(jnp.mean(iters)),
        "std_iters": float(jnp.std(iters)),
        "max_iters": int(jnp.max(iters)),
        "avg_ms": float(jnp.mean(times_ms)),
        "std_ms": float(jnp.std(times_ms)),
        "max_ms": float(jnp.max(times_ms)),
    }


def benchmark_problem(problem: str, operator_apply, preconditioner_factories, rhs_batch, *, mass_matvec):
    rows = []
    for label, preconditioner_apply in preconditioner_factories:
        @jax.jit
        def solve(rhs, precond=preconditioner_apply):
            x, info = solve_singular_cg(
                operator_apply,
                rhs,
                mass_matvec=mass_matvec,
                precond_matvec=precond,
                tol=TOL,
                maxiter=MAXITER,
            )
            return x, jnp.abs(info)

        stats = time_solve(solve, rhs_batch)
        rows.append(BenchmarkRow(problem=problem, case=label, **stats))
    return rows


def build_mass_rhs_batch(seq, k: int, *, seed: int):
    return build_random_besov_rhs_batch(
        seq,
        k,
        dirichlet=DIRICHLET,
        n_rhs=NUM_RHS,
        seed=seed,
        **BESOV_RHS_KWARGS,
    )


def print_table(title: str, rows: list[BenchmarkRow]):
    print()
    print(title)
    jacobi_baseline_ms = {
        row.problem: row.avg_ms
        for row in rows
        if row.case == "jacobi"
    }
    header = (
        f"{'problem':<12} {'case':<28} {'avg_it':>8} {'std_it':>8} {'max_it':>8} "
        f"{'avg_ms':>10} {'std_ms':>10} {'max_ms':>10} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        baseline_ms = jacobi_baseline_ms.get(row.problem)
        speedup = float("nan") if baseline_ms is None or row.avg_ms <= 0 else baseline_ms / row.avg_ms
        print(
            f"{row.problem:<12} {row.case:<28} {row.avg_iters:>8.1f} {row.std_iters:>8.2f} {row.max_iters:>8d} "
            f"{row.avg_ms:>10.2f} {row.std_ms:>10.2f} {row.max_ms:>10.2f} {speedup:>8.2f}"
        )


def main():
    print("Building sequence and operators...")
    t0 = time.perf_counter()
    seq, operator_bundles = build_sequence()
    operators = operator_bundles["default"]
    print(f"built in {time.perf_counter() - t0:.2f} s")
    print(f"resolution={NS}, p={P}, dirichlet={DIRICHLET}, n_rhs={NUM_RHS}")
    print("RHS: random Besov-like functions projected into the FEM space")

    all_rows = []

    for k in (0, 1, 2, 3):
        rhs_batch = build_mass_rhs_batch(seq, k, seed=100 + k)
        operator_apply = lambda x, degree=k: apply_mass_matrix(seq, operators, x, degree, dirichlet=DIRICHLET)
        factories = mass_tensor_case_factories(seq, operator_bundles, k)
        all_rows.extend(
            benchmark_problem(
                f"mass-k{k}",
                operator_apply,
                factories,
                rhs_batch,
                mass_matvec=operator_apply,
            )
        )

    laplace_rhs_batch = build_mass_rhs_batch(seq, 0, seed=500)
    laplace_operator_apply = lambda x: apply_stiffness(seq, operators, x, 0, dirichlet=DIRICHLET)
    laplace_mass_matvec = lambda x: apply_mass_matrix(seq, operators, x, 0, dirichlet=DIRICHLET)
    laplace_factories = [("jacobi", build_k0_laplacian_whole_jacobi(seq, operators))]
    laplace_factories.extend(
        (
            f"whole-cheb{steps}-jacobi",
            build_k0_laplacian_whole_cheb_jacobi(seq, operators, steps),
        )
        for steps in BENCHMARK_WHOLE_CHEB_STEPS
    )
    laplace_factories.append(("schur-tensor", build_k0_laplacian_schur_tensor(seq, operators)))
    all_rows.extend(
        benchmark_problem(
            "laplace-k0",
            laplace_operator_apply,
            laplace_factories,
            laplace_rhs_batch,
            mass_matvec=laplace_mass_matvec,
        )
    )

    print_table("Bulk-Endpoint Benchmark", all_rows)


if __name__ == "__main__":
    main()