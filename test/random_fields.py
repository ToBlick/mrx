"""Random Besov-like test fields for MRX benchmarks and tests.

Provides sparse random Fourier fields with shell-biased mode sampling and
Sobolev-like amplitude decay, suitable for testing projectors, solvers, and
preconditioners with realistic smooth source terms.

Functions
---------
build_random_besov_function
    Build a single random scalar or vector field callable.
build_random_besov_rhs_batch
    Project a batch of random fields into a k-form FEM space and return
    the stacked RHS vectors.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _sample_sparse_fourier_modes(
        key: jax.Array,
        upper_limit: int,
        num_modes: int) -> tuple[jax.Array, jnp.ndarray]:
    """Sample a sparse subset of nonzero Fourier modes without replacement.

    Sampling is biased toward low and intermediate shells (weight ∝ (1+|k|)^{-2})
    to avoid over-representing the large shells that dominate uniform sampling.
    """
    if upper_limit < 1:
        raise ValueError(f"upper_limit must be at least 1, got {upper_limit}")
    max_unique_modes = (upper_limit + 1) ** 3 - 1
    target_modes = min(num_modes, max_unique_modes)
    if target_modes < 1:
        raise ValueError("num_modes must request at least one nonzero mode")

    axis = jnp.arange(upper_limit + 1, dtype=jnp.int32)
    all_modes = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    nonzero_mask = jnp.any(all_modes != 0, axis=1)
    candidate_modes = all_modes[nonzero_mask]

    shell_norms = jnp.linalg.norm(candidate_modes.astype(jnp.float64), axis=1)
    shell_weights = (1.0 + shell_norms) ** (-2.0)
    shell_weights = shell_weights / jnp.sum(shell_weights)

    current_key, draw_key = jax.random.split(key)
    sampled_indices = jax.random.choice(
        draw_key,
        candidate_modes.shape[0],
        shape=(target_modes,),
        replace=False,
        p=shell_weights,
    )
    return current_key, candidate_modes[sampled_indices]


def build_random_besov_function(
        form_degree: int,
        *,
        key: jax.Array | None = None,
        s: float = 1.0,
        upper_limit: int = 50,
        num_modes: int = 256,
        scale: float = 1.0,
        smoothness_margin: float = 1.0,
        normalization_samples: int = 256) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a sparse random Fourier field with shell-biased mode sampling.

    The field is defined on the logical cube ``[0,1]^3`` as a linear combination
    of cosine modes with Gaussian amplitudes decayed by
    ``(1 + |k|)^{-(s + 1/2 + smoothness_margin)}``.

    Args:
        form_degree: Form degree ``k`` in ``{0, 1, 2, 3}``.
        key: Optional JAX PRNG key (defaults to ``PRNGKey(0)``).
        s: Smoothness exponent.
        upper_limit: Maximum Fourier index per direction.
        num_modes: Number of sparse random modes.
        scale: Target RMS scale of the output field.
        smoothness_margin: Extra decay beyond the borderline exponent.
        normalization_samples: Random sample count used to normalise.

    Returns:
        Callable ``f(x)`` with ``x = (r, θ, ζ)``.  Shape ``(1,)`` for
        ``k ∈ {0, 3}`` and ``(3,)`` for ``k ∈ {1, 2}``.
    """
    if form_degree not in (0, 1, 2, 3):
        raise ValueError(f"form_degree must be 0–3, got {form_degree}")
    if num_modes < 1:
        raise ValueError(f"num_modes must be positive, got {num_modes}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if normalization_samples < 1:
        raise ValueError(f"normalization_samples must be positive, got {normalization_samples}")

    if key is None:
        key = jax.random.PRNGKey(0)

    component_count = 1 if form_degree in (0, 3) else 3
    component_keys = jax.random.split(key, component_count)

    def _build_component(component_key):
        component_key, modes = _sample_sparse_fourier_modes(
            component_key, upper_limit, num_modes)
        component_key, coeff_key, phase_key, sample_key = jax.random.split(component_key, 4)

        mode_norms = jnp.linalg.norm(modes.astype(jnp.float64), axis=1)
        decay = (1.0 + mode_norms) ** (-(s + 0.5 + smoothness_margin))
        random_coefficients = jax.random.normal(coeff_key, shape=(modes.shape[0],), dtype=jnp.float64)
        coefficients = decay * random_coefficients
        coefficient_norm = jnp.linalg.norm(coefficients)
        coefficients = jnp.where(coefficient_norm > 0, coefficients / coefficient_norm, coefficients)
        phases = jax.random.uniform(phase_key, shape=(modes.shape[0],),
                                    minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float64)

        def raw_scalar(x):
            x64 = jnp.asarray(x, dtype=jnp.float64)
            arguments = 2.0 * jnp.pi * (modes @ x64) + phases
            return jnp.dot(coefficients, jnp.cos(arguments))

        sample_points = jax.random.uniform(sample_key, shape=(normalization_samples, 3),
                                            minval=0.0, maxval=1.0, dtype=jnp.float64)
        sample_values = jax.vmap(raw_scalar)(sample_points)
        rms_value = jnp.sqrt(jnp.mean(sample_values ** 2))
        normalization = jnp.where(rms_value > 0,
                                  jnp.asarray(scale, dtype=jnp.float64) / rms_value,
                                  jnp.asarray(scale, dtype=jnp.float64))
        return lambda x, _norm=normalization, _raw=raw_scalar: _norm * _raw(x)

    component_fields = tuple(_build_component(ck) for ck in component_keys)

    def besov_field(x):
        return jnp.asarray([cf(x) for cf in component_fields], dtype=jnp.float64)

    return besov_field


def build_random_besov_rhs_batch(
        seq,
        form_degree: int,
        *,
        dirichlet: bool,
        n_rhs: int,
        seed: int = 0,
        s: float = 1.0,
        upper_limit: int = 24,
        num_modes: int = 64,
        scale: float = 1.0,
        smoothness_margin: float = 0.25,
        normalization_samples: int = 256) -> jnp.ndarray:
    """Project a batch of random Besov-like source fields into a k-form space.

    Args:
        seq: DeRham sequence with an assembled map.
        form_degree: Form degree ``k`` in ``{0, 1, 2, 3}``.
        dirichlet: Use Dirichlet extraction when projecting.
        n_rhs: Number of right-hand-side vectors to generate.
        seed: PRNG seed.
        s, upper_limit, num_modes, scale, smoothness_margin,
        normalization_samples: Passed to :func:`build_random_besov_function`.

    Returns:
        Array of shape ``(n_rhs, n_dofs)``.
    """
    if n_rhs < 1:
        raise ValueError(f"n_rhs must be positive, got {n_rhs}")

    attr = f"p{form_degree}_dbc" if dirichlet else f"p{form_degree}"
    projector = getattr(seq, attr)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_rhs)
    rhs_list = []
    for key in keys:
        source = build_random_besov_function(
            form_degree, key=key, s=s,
            upper_limit=upper_limit, num_modes=num_modes,
            scale=scale, smoothness_margin=smoothness_margin,
            normalization_samples=normalization_samples,
        )
        def projected(x, _src=source, _map=seq.map):
            return _src(_map(x))
        rhs_list.append(projector(projected))
    return jnp.stack(rhs_list, axis=0)
