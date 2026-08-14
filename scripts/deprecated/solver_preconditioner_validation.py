# %% [markdown]
# # Solver Preconditioner Validation
#
# Standalone interactive validation for the solver-facing mass-preconditioner
# routes touched by the surgery-Schur refactor. This script intentionally does
# not depend on `test_sequence.py` or any pytest fixtures.
#

# It builds a small spline-projected torus case and checks:
#
# 1. `k = 3` inverse Hodge solves with the production Chebyshev Schur outer
#    preconditioner path.
# 2. `k = 0` and `k = 3` diffusion solves with the built-in `jacobi`, `tensor`,
#    and `chebyshev` mass preconditioners.
# 3. `k = 0` surgery-Schur and `k = 3` direct mass solves against the current
#    scalar mass-preconditioner options.

# %%
from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.io import project_sampled_field
from mrx.mappings import toroid_map
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)

jax.config.update("jax_enable_x64", True)


# %% Configuration
BETTI = (1, 1, 0, 0)
NS = (6, 8, 4)
PS = (3, 3, 3)
TYPES = ("clamped", "periodic", "periodic")
TORUS_EPSILON = 1.0 / 3.0
TORUS_R0 = 1.0
NULLSPACE_EPS = 1e-6
DIFFUSION_EPS = 1e-2
N_COMPARE_RHS = 8


# %% Helpers
def _dof(seq: DeRhamSequence, k: int, dirichlet: bool) -> int:
    return getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")


def _relative_residual(residual: jnp.ndarray, rhs: jnp.ndarray) -> float:
    rhs_norm = float(jnp.linalg.norm(rhs))
    if rhs_norm == 0.0:
        rhs_norm = 1.0
    return float(jnp.linalg.norm(residual) / rhs_norm)


def _solver_status(info) -> tuple[str, int]:
    info_int = int(info)
    return ("converged" if info_int <= 0 else "not-converged", abs(info_int))


def _label(preconditioner) -> str:
    if isinstance(preconditioner, str):
        return preconditioner
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner.kind
    return type(preconditioner).__name__


def _format_table(rows, headers):
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    header_line = " | ".join(
        f"{header:<{widths[idx]}}" for idx, header in enumerate(headers)
    )
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(f"{str(value):<{widths[idx]}}" for idx, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _block_until_ready2(x, info):
    jax.block_until_ready(x)
    jax.block_until_ready(info)


def _jit_warmup_solver(solve_fn, rhs):
    solve_jit = jax.jit(solve_fn)
    x, info = solve_jit(rhs)
    _block_until_ready2(x, info)
    return solve_jit


def _scalar_polynomial_mass_preconditioners():
    return [
        (
            "richardson",
            dict(
                steps=4,
                power_iterations=8,
                damping_safety=0.8,
            ),
        ),
        (
            "chebyshev",
            dict(
                steps=4,
                power_iterations=8,
                min_eig_fraction=1e-3,
            ),
        ),
    ]


def _scalar_schur_mass_preconditioner_catalog():
    polynomial_configs = _scalar_polynomial_mass_preconditioners()

    def _entry(*, outer, schur, inner, reason, preconditioner, label=None):
        if label is None:
            if schur:
                label = f"{outer}/schur/{inner}"
            else:
                label = f"{outer}/no-schur"
        return {
            "outer": outer,
            "inner": inner,
            "schur": schur,
            "label": label,
            "preconditioner": preconditioner,
            "reason": reason,
        }

    def _inner_configs():
        return [
            ("tensor", MassPreconditionerSpec(kind="tensor")),
        ]

    def _schur_spec(outer, inner_spec, **kwargs):
        if outer == "none":
            return MassPreconditionerSpec(
                kind="none",
                surgery_schur=True,
                smoother=inner_spec,
            )
        return MassPreconditionerSpec(
            kind=outer,
            surgery_schur=True,
            smoother=inner_spec,
            **kwargs,
        )

    catalog = [
        _entry(
            outer="none",
            schur=False,
            inner="-",
            reason="identity baseline",
            preconditioner="none",
        ),
        _entry(
            outer="jacobi",
            schur=False,
            inner="-",
            reason="plain diagonal inverse",
            preconditioner=MassPreconditionerSpec(kind="jacobi", surgery_schur=False),
        ),
    ]

    for inner_label, inner_spec in _inner_configs():
        reason = "legacy 'tensor' path"
        catalog.append(
            _entry(
                outer="none",
                schur=True,
                inner=inner_label,
                reason=reason,
                preconditioner=_schur_spec("none", inner_spec),
            )
        )

    for kind, kwargs in polynomial_configs:
        catalog.append(
            _entry(
                outer=kind,
                schur=False,
                inner="-",
                reason="supported polynomial bulk model",
                preconditioner=MassPreconditionerSpec(
                    kind=kind,
                    surgery_schur=False,
                    **kwargs,
                ),
            )
        )
        if kind != "richardson":
            continue
        for inner_label, inner_spec in _inner_configs():
            catalog.append(
                _entry(
                    outer=kind,
                    schur=True,
                    inner=inner_label,
                    reason="supported Schur wrapper with tensor inner smoother",
                    preconditioner=_schur_spec(kind, inner_spec, **kwargs),
                )
            )
    return catalog


def _k0_mass_preconditioner_catalog():
    return _scalar_schur_mass_preconditioner_catalog()


def _k1_mass_preconditioner_catalog():
    return _scalar_schur_mass_preconditioner_catalog()


def _k2_mass_preconditioner_catalog():
    return _scalar_schur_mass_preconditioner_catalog()


def _k3_mass_preconditioner_catalog():
    catalog = [
        {
            "label": "none",
            "preconditioner": "none",
            "reason": "identity baseline",
        },
        {
            "label": "jacobi",
            "preconditioner": MassPreconditionerSpec(kind="jacobi"),
            "reason": "plain diagonal inverse",
        },
        {
            "label": "tensor",
            "preconditioner": MassPreconditionerSpec(kind="tensor"),
            "reason": "direct scalar tensor inverse",
        },
    ]
    for kind, kwargs in _scalar_polynomial_mass_preconditioners():
        catalog.append(
            {
                "label": kind,
                "preconditioner": MassPreconditionerSpec(kind=kind, **kwargs),
                "reason": "polynomial iteration on the direct scalar mass operator",
            }
        )
    return catalog


def _k0_mass_preconditioner_admissibility_rows():
    return [
        (
            "*/schur/*",
            "*",
            "True",
            "*",
            "yes",
            "cartesian product of the Schur models",
        ),
        (
            "*/no-schur/-",
            "*",
            "False",
            "-",
            "yes",
            "supported bulk smoothers none/jacobi/richardson/chebyshev",
        ),
        (
            "tensor/no-schur/*",
            "tensor",
            "False",
            "*",
            "no",
            "this gets dispatched to none/schur/tensor",
        ),
        (
            "*/no-schur/*",
            "*",
            "no-schur",
            "*",
            "no",
            "when Schur is off there is no inner smoother",
        ),
        (
            "jacobi/schur/*",
            "jacobi",
            "True",
            "*",
            "no",
            "outer Jacobi on the Schur-preconditioned operator is disabled",
        ),
        (
            "chebyshev/schur/*",
            "chebyshev",
            "True",
            "*",
            "no",
            "outer Chebyshev is disabled on the public Schur interface",
        ),
        (
            "none/no-schur",
            "none",
            "False",
            "-",
            "yes",
            "identity baseline",
        ),
        (
            "jacobi/no-schur",
            "jacobi",
            "False",
            "-",
            "yes",
            "plain diagonal inverse",
        ),
        (
            "none/schur/jacobi",
            "none",
            "True",
            "jacobi",
            "no",
            "Jacobi is disabled as an inner Schur smoother",
        ),
        (
            "richardson/schur/jacobi",
            "richardson",
            "True",
            "jacobi",
            "no",
            "Jacobi is disabled as an inner Schur smoother",
        ),
        (
            "none/schur/*(not jac)",
            "none",
            "True",
            "tensor|richardson|chebyshev",
            "yes",
            "allowed inner-only Schur family",
        ),
        (
            "richardson/schur/*(not jac)",
            "richardson",
            "True",
            "tensor|richardson|chebyshev",
            "yes",
            "allowed outer Richardson Schur family",
        ),
    ]


def print_k0_mass_preconditioner_admissibility_table():
    print("k=0 mass preconditioner admissibility")
    rows = _k0_mass_preconditioner_admissibility_rows()
    headers = ("label", "outer", "schur", "inner", "admissible", "reason")
    print(_format_table(rows, headers))


def build_case() -> DeRhamSequence:
    seq = DeRhamSequence(
        NS,
        PS,
        2 * PS[0],
        TYPES,
        polar=True,
        tol=1e-12,
        maxiter=1000,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.set_map(toroid_map(epsilon=TORUS_EPSILON, R0=TORUS_R0))
    seq.assemble_all_sparse()
    # seq._compute_nullspaces(BETTI, eps=NULLSPACE_EPS)
    return seq


def validate_k3_hodge_chebyshev(seq: DeRhamSequence):
    print("k=3 inverse Hodge solve validation")
    dirichlet = False
    n = _dof(seq, 3, dirichlet)
    rhs = jax.random.normal(jax.random.PRNGKey(23), (n,), dtype=jnp.float64)

    for coupled_preconditioner in (False, True):
        preconditioner = SaddlePointPreconditionerSpec(
            mass=MassPreconditionerSpec(kind="tensor", surgery_schur=True),
            schur=SchurPreconditionerSpec(
                inner=MassPreconditionerSpec(kind="tensor", surgery_schur=True),
                outer=MassPreconditionerSpec(
                    kind="chebyshev",
                    steps=4,
                    power_iterations=8,
                    min_eig_fraction=1e-3,
                ),
            ),
            coupled=coupled_preconditioner,
        )

        solve_jit = _jit_warmup_solver(
            lambda vec, preconditioner=preconditioner: seq.apply_inverse_laplacian(
                vec,
                k=3,
                dirichlet=dirichlet,
                preconditioner=preconditioner,
                return_info=True,
            ),
            rhs,
        )

        t0 = time.perf_counter()
        x, info = solve_jit(rhs)
        _block_until_ready2(x, info)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        residual = seq.apply_laplacian(x, k=3, dirichlet=dirichlet) - rhs
        relres = _relative_residual(residual, rhs)
        status, iterations = _solver_status(info)
        print(
            f"  coupled={coupled_preconditioner:<5} status={status:<13} "
            f"iters={iterations:>4d} info={int(info):>4d} "
            f"relres={relres:.3e} time_ms={elapsed_ms:.2f}"
        )
        assert int(info) <= 0, (
            "Chebyshev k=3 inverse Hodge solve did not converge "
            f"(coupled={coupled_preconditioner}, info={int(info)})"
        )
        assert relres < 1e-8, (
            "Chebyshev k=3 inverse Hodge solve residual is too large "
            f"(coupled={coupled_preconditioner}, relres={relres:.3e})"
        )


def validate_diffusion_defaults(seq: DeRhamSequence):
    print("diffusion solve validation")
    dirichlet = False
    chebyshev = MassPreconditionerSpec(
        kind="chebyshev",
        steps=4,
        power_iterations=8,
        min_eig_fraction=1e-3,
    )
    cases = [
        (0, "jacobi"),
        (0, "tensor"),
        (0, chebyshev),
        (3, "jacobi"),
        (3, "tensor"),
        (3, chebyshev),
    ]

    for k, preconditioner in cases:
        rhs = jax.random.normal(
            jax.random.PRNGKey(123 + 17 * k + 100 * len(_label(preconditioner))),
            (_dof(seq, k, dirichlet),),
            dtype=jnp.float64,
        )

        solve_jit = _jit_warmup_solver(
            lambda vec, k=k, preconditioner=preconditioner: seq.apply_inverse_mass_plus_eps_laplace_matrix(
                vec,
                k=k,
                eps=DIFFUSION_EPS,
                dirichlet=dirichlet,
                preconditioner=preconditioner,
                return_info=True,
            ),
            rhs,
        )

        t0 = time.perf_counter()
        x, info = solve_jit(rhs)
        _block_until_ready2(x, info)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        residual = seq.apply_mass_plus_eps_laplace_matrix(
            x,
            k=k,
            eps=DIFFUSION_EPS,
            dirichlet=dirichlet,
        ) - rhs
        relres = _relative_residual(residual, rhs)
        label = _label(preconditioner)
        status, iterations = _solver_status(info)
        print(
            f"  k={k} preconditioner={label:<10} status={status:<13} "
            f"iters={iterations:>4d} info={int(info):>4d} "
            f"relres={relres:.3e} time_ms={elapsed_ms:.2f}"
        )
        assert int(info) <= 0, (
            "Diffusion solve did not converge with built-in preconditioner "
            f"{label!r} for k={k} (info={int(info)})"
        )
        assert relres < 1e-8, (
            "Diffusion solve residual is too large "
            f"for preconditioner {label!r} and k={k} (relres={relres:.3e})"
        )


def compare_k0_mass_preconditioners(seq: DeRhamSequence):
    compare_scalar_mass_preconditioners(seq, 0, _k0_mass_preconditioner_catalog())


def compare_scalar_mass_preconditioners(
        seq: DeRhamSequence, k: int, catalog):
    print(f"k={k} mass inverse comparison table")
    dirichlet = False
    n_rhs = N_COMPARE_RHS
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(909 + 1000 * k),
        (_dof(seq, k, dirichlet), n_rhs),
        dtype=jnp.float64,
    ).T
    rows = []

    for entry in catalog:
        outer = entry["outer"]
        inner = entry["inner"]
        surgery_schur = entry["schur"]
        label = entry["label"]
        preconditioner = entry["preconditioner"]
        iterations = []
        times_ms = []
        residuals = []
        converged = 0

        solve_jit = _jit_warmup_solver(
            lambda vec, preconditioner=preconditioner: seq.apply_inverse_mass_matrix(
                vec,
                k,
                dirichlet=dirichlet,
                preconditioner=preconditioner,
                return_info=True,
            ),
            rhs_batch[0],
        )

        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info = solve_jit(rhs)
            _block_until_ready2(x, info)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            status, iters = _solver_status(info)
            iterations.append(iters)
            residual = seq.apply_mass_matrix(x, k, dirichlet=dirichlet) - rhs
            residuals.append(_relative_residual(residual, rhs))
            converged += int(status == "converged")

        rows.append(
            (
                label,
                outer,
                str(surgery_schur),
                inner,
                f"{converged}/{len(rhs_batch)}",
                f"{sum(iterations) / len(iterations):.2f}",
                f"{max(iterations)}",
                f"{sum(residuals) / len(residuals):.3e}",
                f"{max(residuals):.3e}",
                f"{sum(times_ms) / len(times_ms):.2f}",
            )
        )
        print(
            f"  {label:<24} converged={converged}/{len(rhs_batch)} "
            f"avg_iters={sum(iterations) / len(iterations):>7.2f} "
            f"max_iters={max(iterations):>4d} "
            f"avg_relres={sum(residuals) / len(residuals):.3e} "
            f"max_relres={max(residuals):.3e} "
            f"avg_ms={sum(times_ms) / len(times_ms):.2f}"
        )

    headers = (
        "label",
        "outer",
        "schur",
        "inner",
        "converged",
        "avg_iters",
        "max_iters",
        "avg_relres",
        "max_relres",
        "avg_ms",
    )
    print(_format_table(rows, headers))


def compare_k1_mass_preconditioners(seq: DeRhamSequence):
    compare_scalar_mass_preconditioners(seq, 1, _k1_mass_preconditioner_catalog())


def compare_k2_mass_preconditioners(seq: DeRhamSequence):
    compare_scalar_mass_preconditioners(seq, 2, _k2_mass_preconditioner_catalog())


def compare_k3_mass_preconditioners(seq: DeRhamSequence):
    print("k=3 mass inverse comparison table")
    dirichlet = False
    n_rhs = N_COMPARE_RHS
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(1909),
        (_dof(seq, 3, dirichlet), n_rhs),
        dtype=jnp.float64,
    ).T
    rows = []

    for entry in _k3_mass_preconditioner_catalog():
        label = entry["label"]
        preconditioner = entry["preconditioner"]
        reason = entry["reason"]
        iterations = []
        times_ms = []
        residuals = []
        converged = 0

        solve_jit = _jit_warmup_solver(
            lambda vec, preconditioner=preconditioner: seq.apply_inverse_mass_matrix(
                vec,
                3,
                dirichlet=dirichlet,
                preconditioner=preconditioner,
                return_info=True,
            ),
            rhs_batch[0],
        )

        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info = solve_jit(rhs)
            _block_until_ready2(x, info)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            status, iters = _solver_status(info)
            iterations.append(iters)
            residual = seq.apply_mass_matrix(x, 3, dirichlet=dirichlet) - rhs
            residuals.append(_relative_residual(residual, rhs))
            converged += int(status == "converged")

        rows.append(
            (
                label,
                reason,
                f"{converged}/{len(rhs_batch)}",
                f"{sum(iterations) / len(iterations):.2f}",
                f"{max(iterations)}",
                f"{sum(residuals) / len(residuals):.3e}",
                f"{max(residuals):.3e}",
                f"{sum(times_ms) / len(times_ms):.2f}",
            )
        )
        print(
            f"  {label:<10} converged={converged}/{len(rhs_batch)} "
            f"avg_iters={sum(iterations) / len(iterations):>7.2f} "
            f"max_iters={max(iterations):>4d} "
            f"avg_relres={sum(residuals) / len(residuals):.3e} "
            f"max_relres={max(residuals):.3e} "
            f"avg_ms={sum(times_ms) / len(times_ms):.2f} "
            f"reason={reason}"
        )

    headers = (
        "label",
        "reason",
        "converged",
        "avg_iters",
        "max_iters",
        "avg_relres",
        "max_relres",
        "avg_ms",
    )
    print(_format_table(rows, headers))


# %% Build the standalone validation case.
SEQ = build_case()


# %% Run the k=3 Hodge validation.
# validate_k3_hodge_chebyshev(SEQ)


# %% Run the diffusion validation.
# validate_diffusion_defaults(SEQ)


# %% Show admissible k=0 mass preconditioner combinations.
print_k0_mass_preconditioner_admissibility_table()


# %% Compare k=0 mass-inverse preconditioners.
compare_k0_mass_preconditioners(SEQ)


# %% Compare k=1 mass-inverse preconditioners.
compare_k1_mass_preconditioners(SEQ)


# %% Compare k=2 mass-inverse preconditioners.
compare_k2_mass_preconditioners(SEQ)


# %% Compare k=3 mass-inverse preconditioners.
compare_k3_mass_preconditioners(SEQ)




# def main():
#     validate_k3_hodge_chebyshev(SEQ)
#     validate_diffusion_defaults(SEQ)
#     print("all solver validations passed")


# if __name__ == "__main__":
#     main()
# %%
