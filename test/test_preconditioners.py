"""Preconditioner tests for ``mrx.preconditioners`` and ``mrx.operators``.

All tests share the session-scoped ``torus_seq`` fixture (full 3D assembly,
built once for the entire session). Pre-JIT preconditioner applies are
provided by the session-scoped ``precond_jit`` fixture — also built once and
warmed up before any test runs.

Tests
-----
1. **Symmetry** — ``|uᵀPv − vᵀPu| < tol`` over random probe pairs.
2. **SPD** — ``vᵀPv > 0`` for random vectors.
3. **CG iteration reduction** — preconditioned solve uses fewer iterations
   than unpreconditioned on the same RHS batch.
4. **Round-trip accuracy** — ``‖M(M⁻¹b) − b‖ < tol`` after a converged
   preconditioned CG solve.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from test.conftest import n_dofs
from mrx.operators import (
    apply_inverse_mass_matrix,
    apply_mass_matrix,
)
from mrx.preconditioners import MassPreconditionerSpec

jax.config.update("jax_enable_x64", True)

_ALL_K = (0, 1, 2, 3)
_ALL_DBC = (False, True)
_N_PROBES = 4

_JACOBI_SPEC = MassPreconditionerSpec(kind="jacobi")
_TENSOR_SPEC = MassPreconditionerSpec(kind="tensor", surgery_schur=True)
_SPECS = {"jacobi": _JACOBI_SPEC, "tensor": _TENSOR_SPEC}


# ---------------------------------------------------------------------------
# 1. Symmetry (random-probe)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
@pytest.mark.parametrize("label", ["jacobi", "tensor"])
def test_preconditioner_is_symmetric(torus_seq, precond_jit, label, k, dbc):
    n = n_dofs(torus_seq, k, dbc)
    rng = np.random.default_rng(seed=1 + 7 * k + 50 * int(dbc))
    P = precond_jit[(label, k, dbc)]
    atol = 1e-12 if label == "jacobi" else 1e-10
    for _ in range(_N_PROBES):
        u = jnp.asarray(rng.standard_normal(n))
        v = jnp.asarray(rng.standard_normal(n))
        Pv = P(v)
        Pu = P(u)
        lhs = float(u @ Pv)
        rhs = float(v @ Pu)
        scale = max(float(jnp.linalg.norm(u)) * float(jnp.linalg.norm(Pv)), 1.0)
        assert abs(lhs - rhs) < atol * scale, (
            f"{label} not symmetric for k={k} dbc={dbc}: uᵀPv={lhs}, vᵀPu={rhs}"
        )


# ---------------------------------------------------------------------------
# 2. SPD (random-probe)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
@pytest.mark.parametrize("label", ["jacobi", "tensor"])
def test_preconditioner_is_spd(torus_seq, precond_jit, label, k, dbc):
    n = n_dofs(torus_seq, k, dbc)
    rng = np.random.default_rng(seed=2 + 11 * k + 50 * int(dbc))
    P = precond_jit[(label, k, dbc)]
    for _ in range(_N_PROBES):
        v = jnp.asarray(rng.standard_normal(n))
        qf = float(v @ P(v))
        assert qf > 0, (
            f"{label} not positive for k={k} dbc={dbc}: vᵀPv={qf:.3e}"
        )


# ---------------------------------------------------------------------------
# 3. CG iteration reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
@pytest.mark.parametrize("label", ["jacobi", "tensor"])
def test_preconditioner_reduces_cg_iterations(torus_seq, label, k, dbc):
    ops = torus_seq.operators
    rng = np.random.default_rng(seed=42 + 13 * k + 100 * int(dbc))
    n = n_dofs(torus_seq, k, dbc)
    rhss = [jnp.asarray(rng.standard_normal(n)) for _ in range(2)]

    none_iters = []
    precond_iters = []
    for rhs in rhss:
        _, none_info = apply_inverse_mass_matrix(
            torus_seq, ops, rhs, k, dirichlet=dbc,
            preconditioner="none", tol=1e-10, maxiter=200, return_info=True,
        )
        _, precond_info = apply_inverse_mass_matrix(
            torus_seq, ops, rhs, k, dirichlet=dbc,
            preconditioner=_SPECS[label], tol=1e-10, maxiter=200, return_info=True,
        )
        none_iters.append(abs(int(none_info)))
        precond_iters.append(abs(int(precond_info)))

    assert np.mean(precond_iters) < np.mean(none_iters), (
        f"{label} did not reduce CG iterations for k={k} dbc={dbc}: "
        f"none={none_iters}, precond={precond_iters}"
    )


# ---------------------------------------------------------------------------
# 4. Round-trip accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
@pytest.mark.parametrize("label", ["jacobi", "tensor"])
def test_inverse_mass_roundtrip(torus_seq, label, k, dbc):
    ops = torus_seq.operators
    rng = np.random.default_rng(seed=7 + 5 * k + 50 * int(dbc))
    n = n_dofs(torus_seq, k, dbc)
    rhs = jnp.asarray(rng.standard_normal(n))

    x, info = apply_inverse_mass_matrix(
        torus_seq, ops, rhs, k, dirichlet=dbc,
        preconditioner=_SPECS[label], tol=1e-10, maxiter=200, return_info=True,
    )
    assert int(info) <= 0, (
        f"{label} k={k} dbc={dbc} did not converge: info={int(info)}"
    )
    residual = np.asarray(
        apply_mass_matrix(torus_seq, ops, x, k, dirichlet=dbc)
    ) - np.asarray(rhs)
    npt.assert_allclose(
        residual, np.zeros_like(residual), atol=1e-6,
        err_msg=f"{label} k={k} dbc={dbc} round-trip M(M⁻¹b) ≠ b",
    )
