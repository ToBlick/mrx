"""The analytic vacuum field as the manufactured solution for every degree.

``B* = grad Psi`` with ``Psi = (x^3 - 3 x y^2) + z (x^2 - y^2)``, two harmonic
polynomials: curl-free, div-free, fully 3-D, single-valued (no harmonic
part). It is written in physical coordinates and evaluated through the map,
so no closed-form metric and no boundary data are needed: the weak boundary
terms of a curl-free, div-free field vanish at every level, and each problem
is posed as the best approximation the vacuum convergence study uses
(``scripts/analytic_vacuum.py``). The error against the exact field comes
from the load,

    ||u_h - B*||_M^2 = <u_h, u_h>_M - 2 u_h . load(B*) + int |B*|^2 dV,

with ``int |B*|^2 dV = sum_q w_q J_q |B*(F(xi_q))|^2``, exact to quadrature.
"""
import jax
import jax.numpy as jnp
import numpy as np

import mrx


def psi(X):
    return X[0] ** 3 - 3.0 * X[0] * X[1] ** 2 + X[2] * (X[0] ** 2 - X[1] ** 2)


def make_b_star(F):
    """``xi -> grad Psi(F(xi))``, the lab-frame vacuum field at a logical point."""
    grad_psi = jax.grad(psi)

    def b_star(xi):
        return grad_psi(F(xi))
    return b_star


def exact_norm_sq(seq, b_star):
    """``int |B*|^2 dV`` by the sequence's quadrature."""
    bq = jax.lax.map(b_star, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER or None)
    return float(jnp.sum(seq.quad.w * seq.jacobian_j * jnp.sum(bq ** 2, axis=1)))


def relative_error(seq, u_h, k, dirichlet, b_star):
    """``||u_h - B*||_M / ||B*||_M`` for a k-form ``u_h`` (k = 1 or 2), from the load."""
    load = seq.load(b_star, k, dirichlet=dirichlet)
    exact_sq = exact_norm_sq(seq, b_star)
    err_sq = float(u_h @ seq.apply_mass_matrix(u_h, k, dirichlet)) \
        - 2.0 * float(u_h @ load) + exact_sq
    return float(np.sqrt(max(err_sq, 0.0)) / np.sqrt(exact_sq))
