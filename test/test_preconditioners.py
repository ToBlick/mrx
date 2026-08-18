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
from mrx.preconditioners import (
    MassPreconditionerSpec,
    _extraction_gram_inverse,
    _extraction_projector_kron_terms,
    _raw_block_starts,
    _weak_term_rows_by_apply,
    build_extracted_laplacian_diagonal,
    build_weak_term_diagonal,
)

jax.config.update("jax_enable_x64", True)

_ALL_K = (0, 1, 2, 3)
_ALL_DBC = (False, True)
_N_PROBES = 4

_JACOBI_SPEC = MassPreconditionerSpec(kind="jacobi")
# The kind="tensor" arms moved to test/experimental/ on 2026-08-18 with the
# surgery/Schur code itself; production is raw_kron + jacobi only.
_SPECS = {"jacobi": _JACOBI_SPEC}


# ---------------------------------------------------------------------------
# 1. Symmetry (random-probe)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
@pytest.mark.parametrize("label", ["jacobi"])
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
@pytest.mark.parametrize("label", ["jacobi"])
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


# The old maxiter=200 cap made this test compare TRUNCATED counts: at the
# fixture size both arms exceed 200 at tol 1e-10 and the strict `<` failed
# on the 200 == 200 tie -- NOT because jacobi is worse. Measured with a
# real budget (polar toroid): jacobi wins STRICTLY in all 8 (k, bc) cases
# -- 7-10x on the vector masses (k=1: 630->85, k=2: 480->63 its), a few
# percent on the scalar spaces (k=0: 59->55, k=3: 39->34, whose diagonal
# is nearly uniform). Strict `<` is legitimate on any CURVED geometry
# (non-constant diagonal); it would tie only on an exactly uniform mesh,
# which this fixture is not.
@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
@pytest.mark.parametrize("label", ["jacobi"])
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
            preconditioner="none", tol=1e-10, maxiter=3000, return_info=True,
        )
        _, precond_info = apply_inverse_mass_matrix(
            torus_seq, ops, rhs, k, dirichlet=dbc,
            preconditioner=_SPECS[label], tol=1e-10, maxiter=3000, return_info=True,
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
@pytest.mark.parametrize("label", ["jacobi"])
def test_inverse_mass_roundtrip(torus_seq, label, k, dbc):
    ops = torus_seq.operators
    rng = np.random.default_rng(seed=7 + 5 * k + 50 * int(dbc))
    n = n_dofs(torus_seq, k, dbc)
    rhs = jnp.asarray(rng.standard_normal(n))

    # Tensor (production) must converge fast; jacobi (fallback) must
    # converge, full stop -- on the polar extracted mass its diagonal
    # spread is ~3 orders (k=1: 1.4e-3..2.3) and CG needs several hundred
    # iterations. The diagonal itself is exact (verified against the dense
    # probe to 1e-16).
    maxiter = 200 if label == "tensor" else 3000
    x, info = apply_inverse_mass_matrix(
        torus_seq, ops, rhs, k, dirichlet=dbc,
        preconditioner=_SPECS[label], tol=1e-10, maxiter=maxiter, return_info=True,
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


# ---------------------------------------------------------------------------
# 5. Closed-form Laplacian Jacobi diagonal (k >= 1)
# ---------------------------------------------------------------------------
#
# The weak term ``D_{k-1} B_{k-1} D_{k-1}^T`` used to be probed at one operator
# apply per extracted row -- O(N) applies, i.e. O(N^2) work, unusable past a
# few thousand DOFs. ``build_weak_term_diagonal`` replaces it with a closed form
# that is exact for the Kronecker mass model and approximate only through that
# model. These tests pin the two things that must not drift: the EXACT parts
# (the Kronecker expansion of the extraction projector) and the size of the
# modelling error against exact rows.

_WEAK_K = (1, 2, 3)


def _apply_pi_terms(terms, shapes, vec):
    """Apply the Kronecker expansion of Pi to a raw lower-space vector."""
    starts = _raw_block_starts(shapes)
    out = [np.zeros(shape) for shape in shapes]
    for (src, dst, factors) in terms:
        block = np.asarray(vec[starts[dst]:starts[dst + 1]]).reshape(shapes[dst])
        block = np.tensordot(factors[0], block, axes=([1], [0]))
        block = np.tensordot(factors[1], block, axes=([1], [1])).transpose(1, 0, 2)
        block = np.tensordot(factors[2], block, axes=([1], [2])).transpose(1, 2, 0)
        out[src] += block
    return np.concatenate([o.reshape(-1) for o in out])


@pytest.mark.parametrize("k", (0, 1, 2))
@pytest.mark.parametrize("dbc", _ALL_DBC)
def test_extraction_projector_kron_expansion_is_exact(torus_seq, k, dbc):
    """Pi = E^T (E E^T)^-1 E, expanded in Kronecker terms, is EXACT.

    Not a modelling step: the ring is radially thin, so the SVD split of its
    block truncates nothing. Getting this wrong is a ~90% error on the
    near-axis rows, so it is checked rather than assumed.
    """
    e = getattr(torus_seq, f"e{k}_dbc" if dbc else f"e{k}")
    shapes = [tuple(int(s) for s in sh) for sh in getattr(torus_seq, f"basis_{k}").shape]
    terms = _extraction_projector_kron_terms(e, shapes)

    rng = np.random.default_rng(seed=11 + k + 7 * int(dbc))
    for _ in range(_N_PROBES):
        v = jnp.asarray(rng.standard_normal(int(e.forward_shape[1])))
        # Pi v = E^T (E E^T)^-1 (E v), with (E E^T)^-1 the raw_kron gram apply.
        coupled, gram_inv, _ = _extraction_gram_inverse(e)
        ev = e @ v
        if coupled is not None:
            ev = ev.at[coupled].set(gram_inv @ ev[coupled])
        expected = np.asarray(e.T @ ev)
        npt.assert_allclose(
            _apply_pi_terms(terms, shapes, v), expected, rtol=0, atol=1e-11,
            err_msg=f"Pi expansion is not exact for k={k} dbc={dbc}",
        )


def test_weak_term_diagonal_matches_exact_rows(torus_seq):
    """The closed-form weak diagonal against exact rows of the same operator.

    Tolerances are the measured Kronecker-model error, not aspirations: on a
    spline toroid the closed form sits at ~2-4% median and ~30% max against the
    exact probe. That is far inside what a Jacobi diagonal tolerates -- a
    diagonal accurate to a factor rho perturbs the preconditioned condition
    number by at most rho^2, so CG iterations by at most rho -- and the A/B in
    ``scripts/debug/laplacian_jacobi_ab.py`` shows the closed form matching the
    exact probe iteration for iteration.
    """
    ops = torus_seq.operators
    rng = np.random.default_rng(seed=3)
    for k in _WEAK_K:
        for dbc in _ALL_DBC:
            got = np.asarray(build_weak_term_diagonal(
                torus_seq, ops, k, dirichlet=dbc))
            assert np.all(got > 0), (
                f"weak-term diagonal is not positive for k={k} dbc={dbc}; "
                "the construction is diag(X A^-1 X^T) and must be SPSD"
            )
            n = n_dofs(torus_seq, k, dbc)
            rows = rng.choice(n, size=12, replace=False)
            exact = _weak_term_rows_by_apply(
                torus_seq, ops, k, dirichlet=dbc, indices=rows)
            rel = np.abs(got[rows] - exact) / np.abs(exact)
            assert np.median(rel) < 0.10, (
                f"weak-term diagonal k={k} dbc={dbc}: median relative error "
                f"{np.median(rel):.3e} exceeds the measured model error"
            )
            assert rel.max() < 0.60, (
                f"weak-term diagonal k={k} dbc={dbc}: max relative error "
                f"{rel.max():.3e} exceeds the measured model error"
            )


def test_laplacian_jacobi_diagonal_is_positive(torus_seq):
    """``diag(E L_k E^T) > 0`` for every k >= 1 and both boundary conditions.

    Jacobi inverts this entrywise, so a non-positive entry is not a quality
    issue but a broken preconditioner.
    """
    ops = torus_seq.operators
    for k in _WEAK_K:
        for dbc in _ALL_DBC:
            diag = np.asarray(build_extracted_laplacian_diagonal(
                torus_seq, ops, k, dirichlet=dbc))
            assert np.all(diag > 0), (
                f"Laplacian Jacobi diagonal has a non-positive entry for "
                f"k={k} dbc={dbc}: min={diag.min():.3e}"
            )
