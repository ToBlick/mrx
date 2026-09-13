"""Krylov solvers on small dense operators: no mesh, no sequence."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mrx.solvers import deflation_projectors, minres, preconditioned_cg


def _random_orthogonal(n: int, seed: int = 3) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return q


def test_preconditioned_cg_solves_an_spd_system() -> None:
    n = 40
    q = _random_orthogonal(n)
    a = jnp.asarray(q @ np.diag(np.linspace(1.0, 50.0, n)) @ q.T)
    b = jnp.asarray(np.random.default_rng(3).standard_normal(n))
    result = preconditioned_cg(lambda v: a @ v, b, tol=1e-12, maxiter=500)
    x = result[0] if isinstance(result, tuple) else result
    rel = float(jnp.linalg.norm(a @ x - b) / jnp.linalg.norm(b))
    assert rel < 1e-8


def test_minres_solves_a_symmetric_indefinite_system() -> None:
    """The case CG cannot handle: eigenvalues spanning ``[-20, 20]``."""
    n = 40
    q = _random_orthogonal(n, seed=4)
    ev = np.concatenate([np.linspace(-20.0, -1.0, n // 2), np.linspace(1.0, 20.0, n // 2)])
    a = jnp.asarray(q @ np.diag(ev) @ q.T)
    b = jnp.asarray(np.random.default_rng(4).standard_normal(n))
    result = minres(lambda v: a @ v, b, tol=1e-12, maxiter=500)
    x = result[0] if isinstance(result, tuple) else result
    rel = float(jnp.linalg.norm(a @ x - b) / jnp.linalg.norm(b))
    assert rel < 1e-6


def test_deflation_projectors_kill_the_kernel_and_reject_a_vector() -> None:
    n = 40
    q, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((n, 2)))
    vs = jnp.asarray(q.T)
    project_primal, _ = deflation_projectors(vs, lambda v: v)
    y = project_primal(jnp.asarray(np.random.default_rng(5).standard_normal(n)))
    assert float(jnp.max(jnp.abs(vs @ y))) < 1e-10
    identity, _ = deflation_projectors(jnp.zeros((0, n)), lambda v: v)
    b = jnp.ones(n)
    np.testing.assert_allclose(np.asarray(identity(b)), np.asarray(b))
    with pytest.raises(ValueError, match="shape"):
        deflation_projectors(jnp.zeros(n), lambda v: v)
