"""Matrix-free mass, Laplacian, and de Rham complex tests (``mrx.operators``).

All tests use a small all-periodic (4,8,4) / p=2 / q=4 sequence with two
geometries:

  **Identity map** (first section) — Jacobian is 1 everywhere, providing
  analytic reference values with no geometry dependence.

  **Rotating-ellipse map** (second section onwards) — nontrivial metric;
  also used for the Laplacian and de Rham complex tests.

Module-level objects are precomputed once for each geometry section so the
heavy JIT cost is paid at import time, not per-test.

**Mass tests (both geometries)**
  Symmetry, positive definiteness via random probes and dense assembly.

**Hodge Laplacian tests (rotating ellipse)**
  Dense Laplacians assembled from first principles::

      L_0 = G_0^T M_1 G_0
      L_k = G_k^T M_{k+1} G_k  +  D_{k-1} M_{k-1}^{-1} D_{k-1}^T   (k=1,2,3)

  Checked: symmetry, PSD, and null-space dimension equal to β_k (free BCs)
  or β_{d-k} (DBC, relative cohomology), with d=3 and β=(1,1,0,0) for a
  solid torus (clamped-r).

**de Rham complex (identity map, non-polar)**
  ``curl(grad f) = 0`` and ``div(curl F) = 0`` via random-probe tests on
  ``_SEQ`` (``polar=False``). Polar extraction is not a 0/1 selection
  matrix, so the algebraic identity only holds on the non-polar sequence.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    apply_derivative_matrix,
    apply_incidence_matrix,
    apply_mass_matrix,
    apply_stiffness,
    assemble_incidence_operators,
    build_matrixfree_mass_apply,
    mass_core_apply,
)

# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

_NR, _NT, _NZ = 4, 8, 4
_P = 2
_Q = 4
_TYPES = ("clamped", "periodic", "periodic")

_SEQ = DeRhamSequence((_NR, _NT, _NZ), (_P, _P, _P), _Q, _TYPES, polar=False)
_SEQ.evaluate_1d()
_SEQ.set_map(lambda x: x)

_APPLIES = {k: build_matrixfree_mass_apply(_SEQ, k) for k in (0, 1, 2, 3)}

_N_DOF = {
    0: int(_SEQ.basis_0.shape[0][0] * _SEQ.basis_0.shape[0][1] * _SEQ.basis_0.shape[0][2]),
    1: sum(int(s[0] * s[1] * s[2]) for s in _SEQ.basis_1.shape),
    2: sum(int(s[0] * s[1] * s[2]) for s in _SEQ.basis_2.shape),
    3: int(_SEQ.basis_3.shape[0][0] * _SEQ.basis_3.shape[0][1] * _SEQ.basis_3.shape[0][2]),
}

_DENSE = {
    k: np.asarray(jax.vmap(_APPLIES[k], in_axes=1, out_axes=1)(jnp.eye(_N_DOF[k], dtype=jnp.float64)))
    for k in (0, 1, 2, 3)
}

_RNG = np.random.default_rng(42)
_N_PROBES = 6


def _random_vecs(k: int, count: int = _N_PROBES) -> list[np.ndarray]:
    return list(_RNG.standard_normal((count, _N_DOF[k])))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_symmetry_probe(k):
    """M_k is symmetric: v^T (M u) = u^T (M v) for random pairs."""
    apply = _APPLIES[k]
    vecs = _random_vecs(k, count=8)
    for u, v in zip(vecs[:4], vecs[4:]):
        Mu = np.asarray(apply(jnp.asarray(u)))
        Mv = np.asarray(apply(jnp.asarray(v)))
        lhs = float(v @ Mu)
        rhs = float(u @ Mv)
        scale = max(np.linalg.norm(v) * np.linalg.norm(Mu), 1.0)
        assert abs(lhs - rhs) < 1e-12 * scale, (
            f"k={k}: symmetry failed  v^T M u={lhs}  u^T M v={rhs}"
        )


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_positive_definite_probe(k):
    """M_k is positive definite: v^T (M v) > 0 for non-zero v."""
    apply = _APPLIES[k]
    for v in _random_vecs(k):
        Mv = np.asarray(apply(jnp.asarray(v)))
        qf = float(v @ Mv)
        assert qf > 1e-12, f"k={k}: x^T M x = {qf} is not positive"


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_dense_is_spd(k):
    """Densified M_k is symmetric and has all positive eigenvalues."""
    M = _DENSE[k]
    npt.assert_allclose(M, M.T, atol=1e-12, err_msg=f"k={k}: dense M not symmetric")
    eigvals = np.linalg.eigvalsh(M)
    assert eigvals.min() > 1e-12, (
        f"k={k}: dense M not SPD, lambda_min={eigvals.min()}"
    )

# ---------------------------------------------------------------------------
# de Rham complex: curl(grad f) = 0  and  div(curl F) = 0
#
# On the non-polar sequence the extraction is a 0/1 selection (E^T E = I), so
# the raw extracted incidence E^T sp E already satisfies G_{k+1} G_k = 0.  On
# polar sequences the axis gluing is non-unitary, so apply_incidence_matrix now
# applies the TRUE strong derivative G = Gram^{-1}(E^T sp E) (cached, mass-free)
# which restores exact d.d = 0 on extracted DoFs.  We test both: _SEQ
# (polar=False) below, and a polar sequence in test_polar_complex_is_exact.
# ---------------------------------------------------------------------------

_SEQ_OPS = assemble_incidence_operators(_SEQ)

# DOF counts for the non-polar sequence.
_N_NP = {
    (k, dbc): getattr(_SEQ, f"n{k}_dbc" if dbc else f"n{k}")
    for k in (0, 1, 2, 3)
    for dbc in (False, True)
}

_COMPLEX_RNG = np.random.default_rng(7)
_N_COMPLEX_PROBES = 10


@pytest.mark.parametrize("dirichlet", (False, True))
def test_curl_of_grad_is_zero(dirichlet):
    """curl(grad f) = 0: G_1^ext (G_0^ext f) = 0 for random 0-forms f."""
    n0 = _N_NP[(0, dirichlet)]
    for _ in range(_N_COMPLEX_PROBES):
        f = jnp.asarray(_COMPLEX_RNG.standard_normal(n0))
        grad_f = apply_incidence_matrix(
            _SEQ, _SEQ_OPS, f, k=0,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet,
        )
        curl_grad_f = apply_incidence_matrix(
            _SEQ, _SEQ_OPS, grad_f, k=1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet,
        )
        norm = float(jnp.linalg.norm(curl_grad_f))
        assert norm < 1e-12, (
            f"dirichlet={dirichlet}: curl(grad f) != 0, ||curl grad f|| = {norm:.3e}"
        )


@pytest.mark.parametrize("dirichlet", (False, True))
def test_div_of_curl_is_zero(dirichlet):
    """div(curl F) = 0: G_2^ext (G_1^ext F) = 0 for random 1-forms F."""
    n1 = _N_NP[(1, dirichlet)]
    for _ in range(_N_COMPLEX_PROBES):
        F = jnp.asarray(_COMPLEX_RNG.standard_normal(n1))
        curl_F = apply_incidence_matrix(
            _SEQ, _SEQ_OPS, F, k=1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet,
        )
        div_curl_F = apply_incidence_matrix(
            _SEQ, _SEQ_OPS, curl_F, k=2,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet,
        )
        norm = float(jnp.linalg.norm(div_curl_F))
        assert norm < 1e-12, (
            f"dirichlet={dirichlet}: div(curl F) != 0, ||div curl F|| = {norm:.3e}"
        )


# Polar sequence: the axis extraction is non-unitary, so the raw incidence is
# NOT nilpotent there.  apply_incidence_matrix now applies the true strong
# derivative G = Gram^{-1}(E^T sp E), which must restore d.d = 0 on extracted
# DoFs.  This is the regression guard for the polar de Rham exactness fix.
_POLAR_SEQ = DeRhamSequence((6, 8, 4), (3, 3, 3), 6, _TYPES, polar=True,
                            betti_numbers=(1, 1, 0, 0))
_POLAR_SEQ.evaluate_1d()
_POLAR_SEQ.assemble_reference_mass_matrix()
_POLAR_SEQ.set_map(rotating_ellipse_map(eps=1.0 / 3.0, kappa=1.2, R0=1.0, nfp=3))
_POLAR_OPS = assemble_incidence_operators(_POLAR_SEQ, ks=(0, 1, 2))


@pytest.mark.parametrize("dirichlet", (False, True))
@pytest.mark.parametrize("k,name", ((0, "curl(grad)"), (1, "div(curl)")))
def test_polar_complex_is_exact(k, name, dirichlet):
    """G_{k+1} G_k = 0 on the POLAR sequence with the true strong derivative."""
    n = int(getattr(_POLAR_SEQ, f"n{k}_dbc" if dirichlet else f"n{k}"))
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(4):
        v = jnp.asarray(rng.standard_normal(n))
        g = apply_incidence_matrix(_POLAR_SEQ, _POLAR_OPS, v, k,
                                   dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        gg = apply_incidence_matrix(_POLAR_SEQ, _POLAR_OPS, g, k + 1,
                                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        rel = float(jnp.linalg.norm(gg)) / max(float(jnp.linalg.norm(g)), 1e-300)
        worst = max(worst, rel)
    assert worst < 1e-10, (
        f"polar dirichlet={dirichlet}: {name} != 0, rel={worst:.3e}"
    )


#
# A fresh sequence is built with polar=True so that axis regularity is
# enforced correctly.  _APPLIES and _DENSE above captured the identity-map
# geometry in their closures and are unaffected.
# ---------------------------------------------------------------------------

_RE_MAP = rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3)
_RE_SEQ = DeRhamSequence((_NR, _NT, _NZ), (_P, _P, _P), _Q, _TYPES, polar=True)
_RE_SEQ.evaluate_1d()
_RE_SEQ.set_map(_RE_MAP)

_RE_APPLIES = {k: build_matrixfree_mass_apply(_RE_SEQ, k) for k in (0, 1, 2, 3)}

_RE_DENSE = {
    k: np.asarray(jax.vmap(_RE_APPLIES[k], in_axes=1, out_axes=1)(jnp.eye(_N_DOF[k], dtype=jnp.float64)))
    for k in (0, 1, 2, 3)
}

_RE_RNG = np.random.default_rng(99)


def _re_random_vecs(k: int, count: int = _N_PROBES) -> list[np.ndarray]:
    return list(_RE_RNG.standard_normal((count, _N_DOF[k])))


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_re_mass_symmetry_probe(k):
    """M_k (rotating ellipse) is symmetric: v^T (M u) = u^T (M v)."""
    apply = _RE_APPLIES[k]
    vecs = _re_random_vecs(k, count=8)
    for u, v in zip(vecs[:4], vecs[4:]):
        Mu = np.asarray(apply(jnp.asarray(u)))
        Mv = np.asarray(apply(jnp.asarray(v)))
        lhs = float(v @ Mu)
        rhs = float(u @ Mv)
        scale = max(np.linalg.norm(v) * np.linalg.norm(Mu), 1.0)
        assert abs(lhs - rhs) < 1e-12 * scale, (
            f"k={k}: symmetry failed  v^T M u={lhs}  u^T M v={rhs}"
        )


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_re_mass_positive_definite_probe(k):
    """M_k (rotating ellipse) is positive definite: v^T (M v) > 0."""
    apply = _RE_APPLIES[k]
    for v in _re_random_vecs(k):
        Mv = np.asarray(apply(jnp.asarray(v)))
        qf = float(v @ Mv)
        assert qf > 1e-12, f"k={k}: x^T M x = {qf} is not positive"


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_re_mass_dense_is_spd(k):
    """Densified M_k (rotating ellipse) is symmetric and SPD."""
    M = _RE_DENSE[k]
    npt.assert_allclose(M, M.T, atol=1e-12, err_msg=f"k={k}: dense M not symmetric")
    eigvals = np.linalg.eigvalsh(M)
    assert eigvals.min() > 1e-12, (
        f"k={k}: dense M not SPD, lambda_min={eigvals.min()}"
    )


# ---------------------------------------------------------------------------
# Hodge Laplacians (rotating ellipse, polar=True)
#
# We need incidence operators for the exterior derivative.  Extraction
# operators are already on _RE_SEQ.get_operators() from __init__; we only
# need to assemble G_0, G_1, G_2.
# ---------------------------------------------------------------------------

_OPS = assemble_incidence_operators(_RE_SEQ)

# Betti numbers of a solid torus (clamped-r, free BCs): β=(1,1,0,0)
_BETTI_FREE = {0: 1, 1: 1, 2: 0, 3: 0}
# Betti numbers for DBC (relative cohomology): β_{d-k}, d=3
_BETTI_DBC = {k: _BETTI_FREE[3 - k] for k in range(4)}

# Extracted DOF counts per (k, dirichlet) pair.
_N = {
    (k, dbc): getattr(_RE_SEQ, f"n{k}_dbc" if dbc else f"n{k}")
    for k in (0, 1, 2, 3)
    for dbc in (False, True)
}


def _dense_op(apply, n: int) -> np.ndarray:
    """Densify a linear map R^n -> R^? by scanning unit vectors."""
    return np.asarray(
        jax.vmap(apply, in_axes=1, out_axes=1)(jnp.eye(n, dtype=jnp.float64))
    )


def _dense_mass_extracted(k: int, dirichlet: bool) -> np.ndarray:
    n = _N[(k, dirichlet)]
    return _dense_op(
        lambda v: apply_mass_matrix(_RE_SEQ, _OPS, v, k, dirichlet=dirichlet), n
    )


def _dense_laplacian(k: int, dirichlet: bool) -> np.ndarray:
    n_k = _N[(k, dirichlet)]
    K = _dense_op(
        lambda v: apply_stiffness(_RE_SEQ, _OPS, v, k, dirichlet=dirichlet), n_k
    )
    if k == 0:
        return K
    n_km1 = _N[(k - 1, dirichlet)]
    D_T = _dense_op(
        lambda v: apply_derivative_matrix(
            _RE_SEQ, _OPS, v, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet,
            transpose=True,
        ),
        n_k,
    )  # shape (n_{k-1}, n_k)
    M_km1_inv = np.linalg.inv(_dense_mass_extracted(k - 1, dirichlet))
    return K + D_T.T @ M_km1_inv @ D_T


_DENSE_LAP = {
    (k, dbc): _dense_laplacian(k, dbc)
    for k in (0, 1, 2, 3)
    for dbc in (False, True)
}

_LAP_PARAMS = [(k, dbc) for k in (0, 1, 2, 3) for dbc in (False, True)]


@pytest.mark.parametrize("k,dirichlet", _LAP_PARAMS)
def test_laplacian_symmetry(k, dirichlet):
    """L_k is symmetric."""
    L = _DENSE_LAP[(k, dirichlet)]
    npt.assert_allclose(
        L, L.T, atol=1e-10,
        err_msg=f"k={k} dirichlet={dirichlet}: Laplacian not symmetric",
    )


@pytest.mark.parametrize("k,dirichlet", _LAP_PARAMS)
def test_laplacian_psd(k, dirichlet):
    """L_k is positive semi-definite."""
    L = _DENSE_LAP[(k, dirichlet)]
    eigvals = np.linalg.eigvalsh(L)
    lam_max = max(float(abs(eigvals).max()), 1.0)
    assert eigvals.min() >= -1e-10 * lam_max, (
        f"k={k} dirichlet={dirichlet}: not PSD, "
        f"lambda_min={eigvals.min():.3e}, lambda_max={eigvals.max():.3e}"
    )


@pytest.mark.parametrize("k,dirichlet", [(k, dbc) for k in (0, 1, 2, 3) for dbc in (False, True)])
def test_laplacian_null_space_dim(k, dirichlet):
    """Null space of L_k has dimension β_k (free BCs) or β_{d-k} (DBC)."""
    L = _DENSE_LAP[(k, dirichlet)]
    eigvals = np.linalg.eigvalsh(L)
    lam_max = max(float(abs(eigvals).max()), 1.0)
    null_dim = int(np.sum(eigvals < 1e-8 * lam_max))
    expected = _BETTI_DBC[k] if dirichlet else _BETTI_FREE[k]
    assert null_dim == expected, (
        f"k={k} dirichlet={dirichlet}: expected null dim {expected}, got {null_dim}; "
        f"smallest eigenvalues: {eigvals[:expected + 3]}"
    )

