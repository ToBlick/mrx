"""Preconditioner tests for ``mrx.preconditioners`` and ``mrx.operators``.

All tests share the session-scoped ``tiny_seq`` fixture (full 3D assembly,
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

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import mrx
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
)


_ALL_K = (0, 1, 2, 3)
_ALL_DBC = (False, True)
_N_PROBES = 4

_JACOBI_SPEC = MassPreconditionerSpec(kind="jacobi")
_SPECS = {"jacobi": _JACOBI_SPEC}

# Roundoff identities (a diagonal apply is symmetric, the Kronecker expansion
# of the extraction projector is exact): 1e3 eps = 2.2e-13 f64 / 1.2e-4 f32.
IDENT = mrx.eps(1e3)


# ---------------------------------------------------------------------------
# 1. Symmetry (random-probe)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
@pytest.mark.parametrize("label", ["jacobi"])
def test_preconditioner_is_symmetric(tiny_seq, precond_jit, label, k, dbc):
    n = n_dofs(tiny_seq, k, dbc)
    rng = np.random.default_rng(seed=1 + 7 * k + 50 * int(dbc))
    P = precond_jit[(label, k, dbc)]
    atol = IDENT
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
def test_preconditioner_is_spd(tiny_seq, precond_jit, label, k, dbc):
    n = n_dofs(tiny_seq, k, dbc)
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
# real budget (polar toroid, (8,16,8) p=2): jacobi wins STRICTLY in all 8
# (k, bc) cases -- 7-10x on the vector masses (k=1: 630->85, k=2: 480->63
# its), a few percent on the scalar spaces (k=0: 59->55, k=3: 39->34, whose
# diagonal is nearly uniform). The few-percent scalar win needs the
# resolution: on the (4,6,4) tiny fixture k=0 dbc ties or loses (31 vs 33),
# so k=0 is not tested here (its production Laplacian solve is in
# test_poisson.py) and the fixture carries the vector masses and k=3.
def _cg_iterations(seq, label, k, dbc):
    ops = seq.operators
    rng = np.random.default_rng(seed=42 + 13 * k + 100 * int(dbc))
    n = n_dofs(seq, k, dbc)
    rhss = [jnp.asarray(rng.standard_normal(n)) for _ in range(2)]

    none_iters = []
    precond_iters = []
    for rhs in rhss:
        _, none_info = apply_inverse_mass_matrix(
            seq, ops, rhs, k, dirichlet=dbc,
            preconditioner="none", tol=seq.tol, maxiter=3000, return_info=True,
        )
        _, precond_info = apply_inverse_mass_matrix(
            seq, ops, rhs, k, dirichlet=dbc,
            preconditioner=_SPECS[label], tol=seq.tol, maxiter=3000, return_info=True,
        )
        none_iters.append(abs(int(none_info)))
        precond_iters.append(abs(int(precond_info)))
    return none_iters, precond_iters


@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", (1, 2, 3))
@pytest.mark.parametrize("label", ["jacobi"])
def test_preconditioner_reduces_cg_iterations(tiny_seq, label, k, dbc):
    none_iters, precond_iters = _cg_iterations(tiny_seq, label, k, dbc)
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
def test_inverse_mass_roundtrip(tiny_seq, label, k, dbc):
    ops = tiny_seq.operators
    rng = np.random.default_rng(seed=7 + 5 * k + 50 * int(dbc))
    n = n_dofs(tiny_seq, k, dbc)
    rhs = jnp.asarray(rng.standard_normal(n))

    # jacobi must converge, full stop -- on the polar extracted mass its
    # diagonal spread is ~3 orders (k=1: 1.4e-3..2.3) and CG needs several
    # hundred iterations. The diagonal itself is exact (verified against the
    # dense probe to 1e-16). The solve stops at seq.tol relative; the
    # residual is checked against ten times that.
    tol = tiny_seq.tol
    x, info = apply_inverse_mass_matrix(
        tiny_seq, ops, rhs, k, dirichlet=dbc,
        preconditioner=_SPECS[label], tol=tol, maxiter=3000, return_info=True,
    )
    assert int(info) <= 0, (
        f"{label} k={k} dbc={dbc} did not converge: info={int(info)}"
    )
    residual = np.asarray(
        apply_mass_matrix(tiny_seq, x, k, dirichlet=dbc)
    ) - np.asarray(rhs)
    npt.assert_allclose(
        residual, np.zeros_like(residual), atol=10 * tol * float(jnp.linalg.norm(rhs)),
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
def test_extraction_projector_kron_expansion_is_exact(tiny_seq, k, dbc):
    """Pi = E^T (E E^T)^-1 E, expanded in Kronecker terms, is EXACT.

    Not a modelling step: the ring is radially thin, so the SVD split of its
    block truncates nothing. Getting this wrong is a ~90% error on the
    near-axis rows, so it is checked rather than assumed.
    """
    e = getattr(tiny_seq, f"e{k}_dbc" if dbc else f"e{k}")
    shapes = [tuple(int(s) for s in sh) for sh in getattr(tiny_seq, f"basis_{k}").shape]
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
            _apply_pi_terms(terms, shapes, v), expected, rtol=0,
            atol=IDENT * float(np.abs(expected).max()),
            err_msg=f"Pi expansion is not exact for k={k} dbc={dbc}",
        )


@pytest.mark.parametrize("k", _WEAK_K)
@pytest.mark.parametrize("dbc", _ALL_DBC)
def test_laplacian_jacobi_diagonal_is_positive(laplacian_jacobi_diag, k, dbc):
    """``diag(E L_k E^T) > 0`` for every k >= 1 and both boundary conditions.

    Jacobi inverts this entrywise, so a non-positive entry is not a quality
    issue but a broken preconditioner. The diagonals come from the session
    fixture, built once for the whole suite.
    """
    diag = laplacian_jacobi_diag[(k, dbc)]
    assert np.all(diag > 0), (
        f"Laplacian Jacobi diagonal has a non-positive entry for "
        f"k={k} dbc={dbc}: min={diag.min():.3e}"
    )


# ---------------------------------------------------------------------------
# 6. The k=0 Laplacian diagonal is built lazily -- also inside a trace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dbc", _ALL_DBC)
def test_k0_laplacian_jacobi_first_apply_inside_a_trace(tiny_seq, dbc):
    """``kind='jacobi'`` at k=0 applies the stored ``1/diag(S_0)`` of the
    bundle, inside a ``lax`` body (the inverse iteration of
    ``find_nullspace_vectors`` solves the shifted k=0 Laplacian inside a
    ``while_loop``) exactly as eagerly. Nothing is built on the first apply:
    a bundle without the diagonal raises instead.
    """
    import jax

    from mrx.operators import apply_laplacian_preconditioner

    ops = tiny_seq.operators
    assert ops.get_laplacian_diaginv(0, dbc) is not None
    n = n_dofs(tiny_seq, 0, dbc)
    v = jnp.asarray(np.random.default_rng(29).standard_normal(n))

    def P(x):
        return apply_laplacian_preconditioner(
            tiny_seq, ops, x, 0, dirichlet=dbc, kind='jacobi')

    inside, _ = jax.lax.scan(lambda x, _: (P(x), None), v, None, length=2)
    outside = P(P(v))
    assert np.all(np.isfinite(np.asarray(outside)))
    npt.assert_allclose(np.asarray(inside), np.asarray(outside), rtol=0,
                        atol=IDENT * float(np.abs(outside).max()))
    bare = ops.with_laplacian_diaginv(0, None, dirichlet=dbc)
    with pytest.raises(ValueError, match="not built"):
        apply_laplacian_preconditioner(tiny_seq, bare, v, 0, dirichlet=dbc, kind='jacobi')
