"""Matrix-free mass, Laplacian, and de Rham complex tests (``mrx.operators``).

One (4, 6, 4) p=2 module fixture:

  **Polar rotating ellipse** (``re_seq``, nfp=3) -- a genuinely 3-D metric
  and the non-unitary axis gluing. Mass symmetry / positive definiteness by
  random probes and dense assembly; the true strong derivative restores
  ``d.d = 0`` on the extracted DoFs; dense Hodge Laplacians assembled from
  first principles::

      L_0 = G_0^T M_1 G_0
      L_k = G_k^T M_{k+1} G_k  +  D_{k-1} M_{k-1}^{-1} D_{k-1}^T   (k=1,2,3)

  are symmetric, PSD, and have null space of dimension β_k (free BCs) or
  β_{d-k} (DBC, relative cohomology), d=3, β=(1,1,0,0) for the solid torus.
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.local_assembly import build_matrixfree_mass_apply
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    apply_derivative_matrix,
    apply_incidence_matrix,
    apply_mass_matrix,
    apply_stiffness,
    new_operators,
)
from test.dense import dense_from_apply

_NS = (4, 6, 4)
_P = 2
_Q = 3
_TYPES = ("clamped", "periodic", "periodic")
_N_PROBES = 6

# Roundoff identities relative to the size of the quantity: 1e3 eps
# (2.2e-13 f64 / 1.2e-4 f32).
IDENT = mrx.eps(1e3)
# The numerical-zero band of a dense Laplacian, relative to lambda_max:
# 50 eps (1.1e-14 f64 / 6e-6 f32). Measured on the dense Laplacians below
# in float32 (2026-08-26): the harmonic eigenvalues come out at most
# 0.27 eps lambda_max (k=3 dbc, through the dense M_2^-1), the first
# non-harmonic eigenvalue at least 1.7e-4 lambda_max (k=1 free), so a band
# of 50 eps sits a decade inside the gap on both sides in either precision.
# The dense M^-1 leaves an asymmetry of order kappa(M) eps ~ sqrt(eps).
BAND = mrx.eps(50)
ASYM = mrx.sqrt_eps()


def _raw_dofs(seq, k):
    return sum(int(np.prod(s)) for s in getattr(seq, f"basis_{k}").shape)


def _n_ext(seq, k, dbc):
    return int(seq.n(k, dbc))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def re_seq():
    """Polar rotating ellipse (nfp=3), with the incidence operators G_0..G_2."""
    seq = DeRhamSequence(_NS, (_P, _P, _P), _Q, _TYPES, polar=True,
                         betti_numbers=(1, 1, 0, 0))
    seq.set_map(rotating_ellipse_map(eps=1.0 / 3.0, kappa=1.2, R0=1.0, nfp=3))
    return seq, new_operators(seq)


@pytest.fixture(scope="module")
def re_mass(re_seq):
    """Raw-space mass applies and their dense forms on the rotating ellipse."""
    seq, _ = re_seq
    applies = {k: build_matrixfree_mass_apply(seq, k) for k in (0, 1, 2, 3)}
    dense = {k: dense_from_apply(applies[k], _raw_dofs(seq, k)) for k in (0, 1, 2, 3)}
    return applies, dense


def _dense_laplacian(seq, ops, k, dirichlet):
    n_k = _n_ext(seq, k, dirichlet)
    K = dense_from_apply(
        lambda v: apply_stiffness(seq, v, k, dirichlet=dirichlet), n_k)
    if k == 0:
        return K
    D_T = dense_from_apply(
        lambda v: apply_derivative_matrix(
            seq, v, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
            transpose=True),
        n_k,
    )  # shape (n_{k-1}, n_k)
    M_km1 = dense_from_apply(
        lambda v: apply_mass_matrix(seq, v, k - 1, dirichlet=dirichlet),
        _n_ext(seq, k - 1, dirichlet))
    return K + D_T.T @ np.linalg.inv(M_km1) @ D_T


_LAP_PARAMS = [(k, dbc) for k in (0, 1, 2, 3) for dbc in (False, True)]


@pytest.fixture(scope="module")
def re_laplacians(re_seq):
    seq, ops = re_seq
    return {(k, dbc): _dense_laplacian(seq, ops, k, dbc) for k, dbc in _LAP_PARAMS}


# ---------------------------------------------------------------------------
# Masses (rotating ellipse)
# ---------------------------------------------------------------------------

def _random_vecs(rng, n, count=_N_PROBES):
    return list(rng.standard_normal((count, n)))


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_symmetry_probe(re_seq, re_mass, k):
    """M_k is symmetric: v^T (M u) = u^T (M v) for random pairs."""
    seq, _ = re_seq
    apply = re_mass[0][k]
    vecs = _random_vecs(np.random.default_rng(99 + k), _raw_dofs(seq, k), count=8)
    for u, v in zip(vecs[:4], vecs[4:]):
        Mu = np.asarray(apply(jnp.asarray(u)))
        Mv = np.asarray(apply(jnp.asarray(v)))
        lhs = float(v @ Mu)
        rhs = float(u @ Mv)
        scale = max(np.linalg.norm(v) * np.linalg.norm(Mu), 1.0)
        assert abs(lhs - rhs) < IDENT * scale, (
            f"k={k}: symmetry failed  v^T M u={lhs}  u^T M v={rhs}"
        )


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_positive_definite_probe(re_seq, re_mass, k):
    """M_k is positive definite: v^T (M v) > 0 for non-zero v."""
    seq, _ = re_seq
    apply = re_mass[0][k]
    for v in _random_vecs(np.random.default_rng(42 + k), _raw_dofs(seq, k)):
        Mv = np.asarray(apply(jnp.asarray(v)))
        qf = float(v @ Mv)
        assert qf > IDENT * np.linalg.norm(v) * np.linalg.norm(Mv), (
            f"k={k}: x^T M x = {qf} is not positive")


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_dense_is_spd(re_mass, k):
    """Densified M_k is symmetric and has all positive eigenvalues."""
    M = re_mass[1][k]
    npt.assert_allclose(M, M.T, atol=IDENT * np.abs(M).max(),
                        err_msg=f"k={k}: dense M not symmetric")
    eigvals = np.linalg.eigvalsh(M)
    # positive beyond the eigensolver's own resolution, 10 eps lambda_max
    assert eigvals.min() > mrx.eps(10) * eigvals.max(), (
        f"k={k}: dense M not SPD, lambda_min={eigvals.min()}, lambda_max={eigvals.max()}"
    )


# ---------------------------------------------------------------------------
# de Rham complex: curl(grad f) = 0  and  div(curl F) = 0
#
# On the non-polar sequence the extraction is a 0/1 selection (E^T E = I), so
# the raw extracted incidence E^T sp E already satisfies G_{k+1} G_k = 0.  On
# polar sequences the axis gluing is non-unitary, so apply_incidence_matrix
# applies the TRUE strong derivative G = Gram^{-1}(E^T sp E) (cached,
# mass-free), which restores exact d.d = 0 on extracted DoFs.
# ---------------------------------------------------------------------------

_N_COMPLEX_PROBES = 10


@pytest.mark.parametrize("dirichlet", (False, True))
@pytest.mark.parametrize("k,name", ((0, "curl(grad)"), (1, "div(curl)")))
def test_polar_complex_is_exact(re_seq, k, name, dirichlet):
    """G_{k+1} G_k = 0 on the POLAR sequence with the true strong derivative.

    The regression guard for the polar de Rham exactness fix: the raw
    incidence is NOT nilpotent there.
    """
    seq, ops = re_seq
    n = _n_ext(seq, k, dirichlet)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(4):
        v = jnp.asarray(rng.standard_normal(n))
        g = apply_incidence_matrix(seq, v, k,
                                   dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        gg = apply_incidence_matrix(seq, g, k + 1,
                                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        rel = float(jnp.linalg.norm(gg)) / max(float(jnp.linalg.norm(g)), 1e-300)
        worst = max(worst, rel)
    # The polar strong derivative carries the Gram inverse of the axis
    # extraction, hence 1e4 eps rather than 1e3 (2.2e-12 f64 / 1.2e-3 f32).
    assert worst < 10 * IDENT, (
        f"polar dirichlet={dirichlet}: {name} != 0, rel={worst:.3e}"
    )


# ---------------------------------------------------------------------------
# Hodge Laplacians (rotating ellipse, polar=True)
# ---------------------------------------------------------------------------

# Betti numbers of a solid torus (clamped-r, free BCs): β=(1,1,0,0)
_BETTI_FREE = {0: 1, 1: 1, 2: 0, 3: 0}
# Betti numbers for DBC (relative cohomology): β_{d-k}, d=3
_BETTI_DBC = {k: _BETTI_FREE[3 - k] for k in range(4)}


@pytest.mark.parametrize("k,dirichlet", _LAP_PARAMS)
def test_laplacian_symmetry(re_laplacians, k, dirichlet):
    """L_k is symmetric, to the accuracy of the dense M_{k-1}^-1 it carries."""
    L = re_laplacians[(k, dirichlet)]
    npt.assert_allclose(
        L, L.T, atol=ASYM * np.abs(L).max(),
        err_msg=f"k={k} dirichlet={dirichlet}: Laplacian not symmetric",
    )


@pytest.mark.parametrize("k,dirichlet", _LAP_PARAMS)
def test_laplacian_psd(re_laplacians, k, dirichlet):
    """L_k is positive semi-definite: no eigenvalue below the numerical zero."""
    L = re_laplacians[(k, dirichlet)]
    eigvals = np.linalg.eigvalsh(L)
    lam_max = float(abs(eigvals).max())
    assert eigvals.min() >= -BAND * lam_max, (
        f"k={k} dirichlet={dirichlet}: not PSD, "
        f"lambda_min={eigvals.min():.3e}, lambda_max={eigvals.max():.3e}"
    )


@pytest.mark.parametrize("k,dirichlet", _LAP_PARAMS)
def test_laplacian_null_space_dim(re_laplacians, k, dirichlet):
    """Null space of L_k has dimension β_k (free BCs) or β_{d-k} (DBC).

    The count is the number of eigenvalues below ``BAND * lambda_max``. A
    count is only meaningful if the spectrum has a GAP where the band cuts,
    so the last eigenvalue counted must sit a decade inside the band and the
    first one not counted a decade outside it; otherwise the band is cutting
    through a cluster and the Betti number read off it is an artefact of the
    threshold. The gap of 100 is what the measured spectra support in
    float32 (harmonic eigenvalues <= 0.27 eps lambda_max, first non-harmonic
    >= 1.7e-4 lambda_max = 1.4e3 eps); in float64 the same bound leaves
    nine orders of margin on the far side.
    """
    L = re_laplacians[(k, dirichlet)]
    eigvals = np.linalg.eigvalsh(L)
    lam_max = float(abs(eigvals).max())
    band = BAND * lam_max
    null_dim = int(np.sum(eigvals < band))
    expected = _BETTI_DBC[k] if dirichlet else _BETTI_FREE[k]
    detail = (f"smallest eigenvalues: {eigvals[:expected + 3]}, "
              f"lambda_max={lam_max:.3e}, band={band:.3e}")
    assert null_dim == expected, (
        f"k={k} dirichlet={dirichlet}: expected null dim {expected}, got {null_dim}; "
        + detail)
    assert float(eigvals[expected]) > 10 * band, (
        f"k={k} dirichlet={dirichlet}: no spectral gap above the band -- "
        f"lambda_{expected} = {float(eigvals[expected]):.3e}; " + detail)
    if expected:
        assert abs(float(eigvals[expected - 1])) < band / 10, (
            f"k={k} dirichlet={dirichlet}: no spectral gap below the band -- "
            f"lambda_{expected - 1} = {float(eigvals[expected - 1]):.3e}; " + detail)
