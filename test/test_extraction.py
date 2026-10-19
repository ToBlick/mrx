"""The structured apply of the extractions (:class:`mrx.extraction_operators.StructuredExtraction`) is the
extraction.

Every extraction of the sequence, free and Dirichlet, of the full space and of both parity views, has a
structured form, and its apply and transpose apply reproduce the apply from the entries to round-off, on float
arrays, on the columns of a matrix and, in float32, on double-single vectors. The compiled apply holds no gather
and no scatter, which is its purpose on a TPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx.double_single as ds
from mrx import extraction_operators
from mrx.precision import PAIRS, eps

# A row of E holds at most two entries of size one, so the two applies differ by a few roundings of the
# working precision, and in double-single by a few of its 48 bits.
ROUND_OFF = eps(8.0)
DS_ROUND_OFF = 1e-13


def _extractions(seq):
    """Every extraction of ``seq`` with a name: the full space and both parity views, free and Dirichlet."""
    return {f"{name} k={k}{' dirichlet' if d else ''}": view.extraction[(k, d)]
            for name, view in (("full", seq), ("odd", seq.odd), ("even", seq.even))
            for k in range(4) for d in (False, True)}


@pytest.fixture(autouse=True)
def _structured(monkeypatch):
    """The structured apply, which ``MRX_EXTRACTION=auto`` keeps for a TPU."""
    monkeypatch.setattr(extraction_operators, "MODE", "structured")


def _close(new, ref, tol):
    new, ref = np.asarray(new, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return np.abs(new - ref).max() <= tol * np.abs(ref).max()


@pytest.mark.parametrize("view", ("full", "odd", "even"))
def test_structured_apply_is_the_extraction(seq, view):
    """``E x``, ``E^T y`` and ``E X`` for a matrix ``X`` agree with the apply from the entries."""
    rng = np.random.default_rng(0)
    for name, e in _extractions(seq).items():
        if not name.startswith(view):
            continue
        assert e.structured is not None, f"{name}: no structured form"
        for A in (e, e.T):
            x = rng.standard_normal(A.shape[1])
            assert _close(A @ jnp.asarray(x), A.coo_matmul(jnp.asarray(x)), ROUND_OFF), name
            X = jnp.asarray(rng.standard_normal((A.shape[1], 3)))
            assert _close(A @ X, A.coo_matmul(X), ROUND_OFF), name
            if PAIRS:
                new, ref = A @ ds.from_host(x), A.coo_matmul(ds.from_host(x))
                assert isinstance(new, ds.DS)
                assert _close(ds.to_host(new), ds.to_host(ref), DS_ROUND_OFF), name


def test_structured_apply_has_no_gather(seq):
    """The compiled ``E(1) x`` and ``E(1)^T y`` of the odd view hold no gather and no scatter."""
    e = seq.odd.E(1)
    for A in (e, e.T):
        x = ds.lift(jnp.ones(A.shape[1])) if PAIRS else jnp.ones(A.shape[1])
        hlo = jax.jit(lambda v, A=A: A @ v).lower(x).as_text()
        assert "gather" not in hlo and "scatter" not in hlo
