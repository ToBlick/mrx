"""Stellarator symmetry of a map ``Phi`` from the logical cube ``(r, theta, zeta)`` to physical space.

:func:`stellarator_symmetric_scalar` makes a map fitted to an equilibrium stellarator symmetric and
:func:`stellarator_symmetry_defect` measures how far a map is from being so.
"""
from typing import Callable

import jax
import jax.numpy as jnp

from mrx.differential_forms import DifferentialForm
from mrx.symmetry import reflection_permutation


# Stellarator symmetry (R, phi, Z) -> (R, -phi, -Z) is (r, theta, zeta) -> (r, -theta, -zeta) in logical
# coordinates, with R even and Z odd. In Cartesian coordinates it is the reflection S = diag(1, -1, -1).
STELLARATOR_REFLECTION = jnp.array([1.0, -1.0, -1.0])


def stellarator_symmetric_scalar(raw: jnp.ndarray, basis_0: DifferentialForm, even: bool) -> jnp.ndarray:
    """Project the ``(n_r, n_t, n_z)`` coefficients of a scalar spline onto its even part (for ``R``) or
    its odd part (for ``Z``) under ``(theta, zeta) -> (-theta, -zeta)``. Both angular axes must be
    uniform and periodic."""
    perm_t = reflection_permutation(basis_0.Lambda[1].n, basis_0.Lambda[1].p)
    perm_z = reflection_permutation(basis_0.Lambda[2].n, basis_0.Lambda[2].p)
    reflected = raw[:, perm_t, :][:, :, perm_z]
    return 0.5 * (raw + reflected) if even else 0.5 * (raw - reflected)


def stellarator_symmetry_defect(Phi: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """``max |Phi(r, -t, -z) - S Phi(r, t, z)|`` over the logical points ``x`` (``(n, 3)``). It is zero
    exactly when ``Phi`` is stellarator symmetric at these points."""
    def one(xi):
        r, t, z = xi
        return jnp.max(jnp.abs(Phi(jnp.array([r, -t, -z])) - STELLARATOR_REFLECTION * Phi(xi)))

    return jnp.max(jax.vmap(one)(jnp.atleast_2d(jnp.asarray(x))))
