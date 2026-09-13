"""The discrete complex is exact and its harmonic forms are what they claim.

``d d = 0`` with the strong derivative on the POLAR sequence -- the raw
incidence is not nilpotent there, the analytic axis stencils are (the
regression guard for the polar exactness fix) -- and the stored harmonic forms
have a Rayleigh quotient at roundoff and an identity mass Gram matrix.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import mrx
from mrx.nullspace import (
    _n_vectors,
    direct_construction_unsupported_reason,
    estimate_spectral_gap,
    get_nullspace,
    harmonic_rayleigh,
    laplacian_pair,
)
from mrx.precision import RESIDUAL_DTYPE, eps

# The polar strong derivative carries the Gram inverse of the axis
# extraction: 1e4 eps (2.2e-12 f64 / 1.2e-3 f32).
EXACT = mrx.eps(1e4)
# Rayleigh quotient v^T L v / v^T M v of a stored harmonic form: quadratic in
# the eigenvector error, which the solve leaves at sqrt(eps) -- so eps-scaled,
# 4.5e3 eps (1e-12 f64 / 5.4e-4 f32).
HARMONIC = mrx.eps(4.5e3)


@pytest.mark.parametrize("dirichlet", (False, True))
@pytest.mark.parametrize("k", (0, 1))
def test_polar_complex_is_exact(seq, k, dirichlet):
    """``D_{k+1} D_k = 0`` on random k-forms, relative to ``||D_k v||``."""
    rng = np.random.default_rng(11)
    v = jnp.asarray(rng.standard_normal(seq.n(k, dirichlet)), dtype=mrx.DTYPE)
    g = seq.apply_incidence_matrix(v, k, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
    gg = seq.apply_incidence_matrix(g, k + 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
    rel = float(jnp.linalg.norm(gg) / jnp.linalg.norm(g))
    assert rel < EXACT, f"k={k} dirichlet={dirichlet}: d d != 0, rel={rel:.3e}"


@pytest.mark.parametrize(("k", "dirichlet"), ((1, False), (2, True)))
def test_harmonic_forms(seq, k, dirichlet):
    """The stored forms of the two non-trivial cohomologies are harmonic to
    roundoff in the Rayleigh quotient, and mass-orthonormal to solver
    tolerance or to roundoff, whichever of the two is looser."""
    vs = get_nullspace(seq.operators, k, dirichlet)
    assert vs.shape[0] == 1, f"k={k} dirichlet={dirichlet}: {vs.shape[0]} harmonic forms"
    mass_vs = jax.vmap(lambda v: seq.apply_mass_matrix(v, k, dirichlet=dirichlet))(vs)
    # The Gram entry is formed in the working dtype: 1 to 2 eps measured in
    # every configuration (2026-09-05), on top of the tolerance of the solves
    # that built the form. Refinement takes seq.tol to 1e-8, below the 1.2e-7
    # that one float32 eps already costs, so the roundoff term is what carries
    # this at float32; at float64 it is 2e-15 and inert.
    npt.assert_allclose(vs @ mass_vs.T, jnp.eye(1), atol=seq.tol + eps(10))
    v = vs[0]
    rayleigh = float(v @ seq.apply_laplacian(v, k, dirichlet=dirichlet)) / float(v @ mass_vs[0])
    assert abs(rayleigh) < HARMONIC, f"k={k} dirichlet={dirichlet}: Rayleigh {rayleigh:.2e}"
    lv, mv = laplacian_pair(seq, v, k, dirichlet=dirichlet)
    residual_rq = harmonic_rayleigh(seq, v, k, dirichlet=dirichlet)
    assert lv.dtype == RESIDUAL_DTYPE
    assert abs(residual_rq) < HARMONIC


def test_n_vectors_is_the_poincare_lefschetz_table() -> None:
    """Worked example in ``_n_vectors``: ``betti = (1, 1, 0, 0)``."""
    betti = (1, 1, 0, 0)
    assert [_n_vectors(betti, k, False) for k in range(4)] == [1, 1, 0, 0]
    assert [_n_vectors(betti, k, True) for k in range(4)] == [0, 0, 1, 1]


def test_direct_construction_reason_names_the_offending_betti() -> None:
    assert direct_construction_unsupported_reason((1, 1, 0, 0)) is None
    assert "b0 = 2" in (direct_construction_unsupported_reason((2, 1, 0, 0)) or "")
    assert "b3 = 1" in (direct_construction_unsupported_reason((1, 1, 0, 1)) or "")
    assert "b2 = 1" in (direct_construction_unsupported_reason((1, 0, 1, 0)) or "")


def test_spectral_gap_is_above_the_harmonic_rayleigh(seq) -> None:
    """One pair only: a handful of inverse-iteration sweeps, not a digit hunt."""
    lam, n_sweeps = estimate_spectral_gap(
        seq, seq.operators, 2, True, maxiter=2)
    vs = get_nullspace(seq.operators, 2, True)
    rq = abs(harmonic_rayleigh(seq, vs[0], 2, True))
    assert n_sweeps >= 1
    assert lam > rq
