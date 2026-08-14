"""Chebyshev / Richardson / Lanczos relaxation machinery -- NOT production.

Removed from mrx.preconditioners on 2026-08-14: the production decision is
NO Chebyshev (or Richardson) acceleration anywhere -- see
``docs/PRODUCTION.md``. Production preconditioner kinds are
none/jacobi/tensor (plus 'exact_jacobi' as a schur-outer debug arm). This
module keeps the generic spectral-estimation and polynomial-apply builders
for research scripts.

Contents: power-iteration lambda_max estimation, Lanczos bound estimation
for a preconditioned operator pair (S A), the optimal-omega Richardson
estimate, and the fixed-window Chebyshev apply builder.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

def _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply, smoother_apply, size: int, *,
        n_iter: int = 10, seed: int = 0):
    """Estimate the largest Rayleigh quotient of ``S A`` via power iteration."""
    vector = jax.random.normal(
        jax.random.PRNGKey(seed), (size,), dtype=jnp.float64)

    def operator_norm(x):
        ax = operator_apply(x)
        return jnp.sqrt(jnp.abs(jnp.vdot(x, ax).real))

    init_norm = operator_norm(vector)
    vector = vector / jnp.where(init_norm > 0, init_norm, 1.0)

    def body(_, state):
        current, _ = state
        image = smoother_apply(operator_apply(current))
        image_norm = operator_norm(image)
        safe_norm = jnp.where(image_norm > 0, image_norm, 1.0)
        updated = image / safe_norm
        rayleigh = jnp.real(
            jnp.vdot(updated, operator_apply(smoother_apply(operator_apply(updated)))))
        return updated, rayleigh

    _, rayleigh = jax.lax.fori_loop(
        0, n_iter, body, (vector, jnp.asarray(0.0, dtype=jnp.float64)))
    return jnp.maximum(rayleigh, jnp.asarray(0.0, dtype=jnp.float64))


def _project_out_vectors(vector, orthogonal_vectors=None):
    if orthogonal_vectors is None or orthogonal_vectors.shape[0] == 0:
        return vector

    def body(index, projected):
        basis_vector = orthogonal_vectors[index]
        denom = jnp.vdot(basis_vector, basis_vector).real
        coeff = jnp.where(
            denom > 0.0,
            jnp.vdot(basis_vector, projected).real / denom,
            jnp.asarray(0.0, dtype=projected.dtype),
        )
        return projected - coeff * basis_vector

    return jax.lax.fori_loop(0, orthogonal_vectors.shape[0], body, vector)


def _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply, smoother_apply, size: int, *,
        lanczos_iterations: Optional[int] = 16,
        lanczos_max_eig_inflation: float = 1.1,
        lanczos_min_eig_deflation: float = 0.85,
        lanczos_min_eig_floor_fraction: float = 1e-3,
        seed: int = 0,
        orthogonal_vectors=None):
    """Estimate spectral bounds of the preconditioned operator ``S A`` via Lanczos.

    (The former ``spec: MassPreconditionerSpec`` parameter is gone -- the
    lanczos_* fields were removed from the production spec 2026-08-14; pass
    the knobs explicitly.)
    """
    if lanczos_iterations is None or lanczos_iterations < 1:
        raise ValueError("Lanczos iteration count must be positive")

    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)

    def operator_norm(x):
        ax = operator_apply(x)
        return jnp.sqrt(jnp.maximum(jnp.abs(jnp.vdot(x, ax).real), tiny))

    vector = jax.random.normal(
        jax.random.PRNGKey(seed), (size,), dtype=jnp.float64)
    vector = _project_out_vectors(vector, orthogonal_vectors)
    init_norm = operator_norm(vector)
    vector = vector / jnp.where(init_norm > 0, init_norm, 1.0)

    def do_iteration(iteration, state):
        previous, current, beta_prev, alphas, betas, active = state

        def step(active_state):
            previous, current, beta_prev, alphas, betas, _ = active_state
            image = smoother_apply(operator_apply(current))
            alpha = jnp.real(jnp.vdot(current, operator_apply(image)))
            residual = image - alpha * current
            residual = residual - jnp.where(iteration > 0, beta_prev, 0.0) * previous
            residual = _project_out_vectors(residual, orthogonal_vectors)
            beta = operator_norm(residual)

            alphas = alphas.at[iteration].set(alpha)
            continue_iteration = (iteration + 1 < lanczos_iterations) & (beta > tiny)
            betas = betas.at[iteration].set(jnp.where(continue_iteration, beta, 0.0))

            safe_beta = jnp.where(beta > 0.0, beta, 1.0)
            next_current = residual / safe_beta
            previous = jnp.where(continue_iteration, current, previous)
            current = jnp.where(continue_iteration, next_current, current)
            beta_prev = jnp.where(continue_iteration, beta, beta_prev)
            return previous, current, beta_prev, alphas, betas, continue_iteration

        return jax.lax.cond(active, step, lambda inactive_state: inactive_state, state)

    initial_state = (
        jnp.zeros_like(vector),
        vector,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.zeros((lanczos_iterations,), dtype=jnp.float64),
        jnp.zeros((lanczos_iterations,), dtype=jnp.float64),
        jnp.asarray(True),
    )
    _, _, _, alphas, betas, _ = jax.lax.fori_loop(
        0,
        lanczos_iterations,
        do_iteration,
        initial_state,
    )

    tridiagonal = jnp.diag(alphas)
    offdiag = betas[:-1]
    tridiagonal = tridiagonal + jnp.diag(offdiag, k=1) + jnp.diag(offdiag, k=-1)
    ritz_values = jnp.linalg.eigvalsh(tridiagonal)
    max_ritz = jnp.maximum(ritz_values[-1], tiny)
    max_eig = jnp.maximum(
        jnp.asarray(lanczos_max_eig_inflation, dtype=jnp.float64) * max_ritz,
        tiny,
    )
    floor = jnp.asarray(
        lanczos_min_eig_floor_fraction, dtype=jnp.float64
    ) * max_eig
    min_positive_ritz = jnp.min(jnp.where(ritz_values > tiny, ritz_values, jnp.inf))
    guarded_min = jnp.asarray(
        lanczos_min_eig_deflation, dtype=jnp.float64
    ) * min_positive_ritz
    min_eig = jnp.where(
        jnp.isfinite(min_positive_ritz),
        jnp.maximum(floor, guarded_min),
        floor,
    )
    return min_eig, max_eig


def _estimate_richardson_omega_apply(
    operator_apply,
    smoother_apply,
    size: int,
    *,
    lanczos_iterations: int,
    lanczos_max_eig_inflation: float,
    lanczos_min_eig_deflation: float,
    lanczos_min_eig_floor_fraction: float,
    seed: int = 0,
) -> float:
    """Auto-tune the Richardson relaxation parameter via Lanczos.

    Estimates the spectral bounds of the preconditioned operator
    ``S A`` (where ``A = operator_apply`` and ``S = smoother_apply``) and
    returns the optimal Richardson weight ``omega = 2 / (lambda_min + lambda_max)``
    for SPD systems. ``S`` and ``A`` are both required to be SPD so that
    ``S A`` has positive real eigenvalues.
    """
    lambda_min, lambda_max = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        size,
        lanczos_iterations=lanczos_iterations,
        lanczos_max_eig_inflation=lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=lanczos_min_eig_floor_fraction,
        seed=seed,
    )
    denom = jnp.maximum(lambda_min + lambda_max,
                        jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64))
    return float(2.0 / denom)


def _build_chebyshev_apply_preconditioner(
    operator_apply,
    smoother_apply,
    *,
    steps: int,
    min_eig: float,
    max_eig: float,
):
    if steps < 1:
        raise ValueError("Chebyshev step count must be positive")
    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    max_eig = jnp.maximum(jnp.asarray(max_eig, dtype=jnp.float64), tiny)
    min_eig = jnp.clip(jnp.asarray(min_eig, dtype=jnp.float64), tiny, max_eig)

    d = 0.5 * (max_eig + min_eig)
    c = 0.5 * (max_eig - min_eig)

    def apply(rhs):
        alpha0 = jnp.asarray(1.0, dtype=rhs.dtype) / d.astype(rhs.dtype)

        def body(iteration, state):
            x, residual, direction, alpha = state
            correction = smoother_apply(residual)
            beta = (0.5 * c.astype(rhs.dtype) * alpha) ** 2
            new_alpha = jnp.where(
                iteration == 0,
                alpha,
                jnp.asarray(1.0, dtype=rhs.dtype) / (d.astype(rhs.dtype) - beta),
            )
            new_direction = jnp.where(
                iteration == 0,
                correction,
                correction + beta * direction,
            )
            x = x + new_alpha * new_direction
            residual = residual - new_alpha * operator_apply(new_direction)
            return x, residual, new_direction, new_alpha

        x, _, _, _ = jax.lax.fori_loop(
            0,
            steps,
            body,
            (jnp.zeros_like(rhs), rhs, jnp.zeros_like(rhs), alpha0),
        )
        return x

    return apply

