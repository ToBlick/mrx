"""Greville interpolation is a projector on the extracted 0-form space."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mrx.differential_forms import DiscreteFunction
from mrx.precision import eps
from mrx.projectors import greville_axes, interpolate


def test_greville_axes_are_one_per_logical_direction(toroid) -> None:
    axes = greville_axes(toroid)
    assert len(axes) == 3
    for axis in axes:
        assert axis.coll.ndim == 2
        assert axis.hist.ndim == 2


def test_interpolate_is_the_identity_on_its_range(toroid) -> None:
    """A 0-form already in the space is recovered to solver / roundoff tolerance."""
    rng = np.random.default_rng(7)
    dof = jnp.asarray(rng.standard_normal(toroid.n(0, False)))
    disc = DiscreteFunction(dof, toroid.basis_0, toroid.E(0, False))
    got = interpolate(toroid, disc, 0, dirichlet=False)
    rel = float(jnp.linalg.norm(got - dof) / jnp.linalg.norm(dof))
    assert rel < 1e2 * toroid.tol + eps(1e2)


def test_interpolate_rejects_bad_frame_and_k(toroid) -> None:
    with pytest.raises(ValueError, match="frame"):
        interpolate(toroid, lambda x: 1.0, 0, frame="tangent")
    with pytest.raises(ValueError, match="k must be"):
        interpolate(toroid, lambda x: 1.0, 4)
    with pytest.raises(ValueError, match="k=3"):
        interpolate(toroid, lambda x: 1.0, 3, frame="ref")
