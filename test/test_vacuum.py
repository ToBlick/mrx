"""The paper's manufactured vacuum solution on the li383 domain, and the Leray
projections.

An analytic vacuum field is curl- and divergence-free; the discrete one is
recovered from its normal boundary data by the two Hodge decompositions of
the vector-valued ``L^2`` (the paper, "Manufactured vacuum solution"):

* ``k = 1``: ``H = grad f + c h_1`` with ``f`` the k=0 natural Laplacian solve
  of ``<grad f, grad v> = <B*, grad v>`` and ``c`` the M-projection onto the
  harmonic 1-form (the scalar-potential route);
* ``k = 2``: ``B = curl A`` with ``A`` the k=1 natural Hodge-Laplacian solve of
  ``<curl A, curl w> = <B*, curl w>`` (the vector-potential route; ``b_2 = 0``
  on the solid torus, so no harmonic term).

``B* = e_phi / R + lam grad(R^nfp sin(nfp phi))``: the toroidal-field flux
part plus a stellarator-symmetric ripple (``Im (x + i y)^nfp``, harmonic), so
it lives on the half-period sequence with its parity; the paper's ripple has
``nfp = 2`` and ``cos``, li383 has ``nfp = 3``; the solves run on the odd
view (``seq.odd``), the scalar ones below on the even one. The error is the exact
``||B_h - B*||_M^2 = <B_h, B_h>_M - 2 B_h . load(B*) + int |B*|^2 dV``. Each
solve must converge to ``seq.tol`` and land under a measured band of the
relative L2 error (a wrong metric factor, boundary row or extraction moves it
by a factor). The Leray projections (k=2, the relaxation's; k=1, the wall
pressure's and the nullspace's) are divergence-free at solver tolerance,
idempotent and non-expansive, and the descent's potential route gives the
same k=2 projection from one k=1 Hodge solve.

The scalar Laplacians get the same treatment with a scalar ``psi``, ANY
smooth one (nothing harmonic about it; its gradient is an EVEN vector --
``B``'s pattern is the odd one, the gradient of a symmetric scalar has the
velocity's): the k=0 natural solve of ``<grad f,
grad v> = <grad psi, grad v>`` recovers ``psi`` up to the constant, the
gradient of the data loaded against the gradient of the test function; the
k=3 natural solve of ``<delta rho, delta tau> = <grad rho*, delta tau>``
with ``delta = -M_2^-1 D_2^T M_3`` the weak gradient into the FREE 2-forms
recovers ``grad rho*`` for ``rho* = (1 - r^2) psi``: the weak gradient is
the adjoint of the divergence on 2-forms with no normal condition, so it is
the gradient of functions that VANISH on the wall -- the natural k=3
Laplacian is the Dirichlet Laplacian on the density (measured 2026-09-21:
for a psi with boundary values the free k=3 solve recovers 8% of ``grad
psi``, the projection onto those gradients). The Dirichlet 2-forms give
the Neumann one instead.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.precision import RESIDUAL_DTYPE, eps

#: Ripple amplitude of ``B*``.
LAM = 1.0
# Relative L2 error of the two routes on li383 (8, 12, 12) p=2.
# PROVISIONAL (2026-09-20, not yet measured on this domain; the QA device at
# (6, 12, 6) p=2 gave 0.087 and 0.152): to be replaced by 1.25x the measured
# errors once the suite has run.
ERROR_BAND = {1: 0.15, 2: 0.25}
# Relative L2 error of psi (k=0, the constant removed) and of grad rho* (k=3).
# PROVISIONAL likewise.
SCALAR_ERROR_BAND = {0: 0.1, 3: 0.25}


def psi(X):
    """A smooth scalar of the physical point, stellarator symmetric (even in
    ``(y, z) -> (-y, -z)``) and not harmonic: ``x^3 + x y^2 + x z^2 + 3 x y z``."""
    return X[0] ** 3 + X[0] * X[1] ** 2 + X[0] * X[2] ** 2 + 3.0 * X[0] * X[1] * X[2]


def vacuum_field(seq, lam=LAM):
    """``B*(xi)`` as a lab-frame vector at the logical point ``xi``."""
    nfp = seq.nfp

    def ripple(X):                      # R^nfp sin(nfp phi) = Im (x + i y)^nfp
        R2 = X[0] ** 2 + X[1] ** 2
        return R2 ** (nfp / 2) * jnp.sin(nfp * jnp.arctan2(X[1], X[0]))
    grad_ripple = jax.grad(ripple)

    def f(xi):
        X = seq.map(xi)
        tf = jnp.array([-X[1], X[0], 0.0]) / (X[0] ** 2 + X[1] ** 2)    # e_phi / R = grad(phi)
        return tf + lam * grad_ripple(X)
    return f


@pytest.fixture(scope="module")
def bstar(seq):
    """``(B*, int |B*|^2 dV)``."""
    f = vacuum_field(seq)
    Bq = jax.vmap(f)(seq.quad.x)
    return f, float(jnp.sum(seq.quad.w * seq.jacobian_j * jnp.sum(Bq ** 2, axis=1)))


@pytest.mark.parametrize("k", (1, 2))
def test_manufactured_vacuum_solution(seq, bstar, k):
    f, bstar_sq = bstar
    seq = seq.odd                                             # B* is odd
    load = seq.load(f, k, dirichlet=False)                    # int Lambda_i . B*
    if k == 1:
        rhs = seq.apply_incidence_matrix(load, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
        phi, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=False, return_info=True, dtype=RESIDUAL_DTYPE)
        residual = seq.apply_stiffness(phi, 0, dirichlet=False) - rhs
        h = seq.nullspace(1, False)[0]
        Bh = (seq.apply_strong_grad(phi.astype(mrx.DTYPE), dirichlet_in=False, dirichlet_out=False)
              + (float(load @ h) / float(h @ seq.apply_mass_matrix(h, 1, False))) * h)
    else:
        rhs = seq.apply_incidence_matrix(load, 1, dirichlet_in=False, dirichlet_out=False, transpose=True)
        A, info = seq.apply_inverse_laplacian(rhs, 1, dirichlet=False, return_info=True, dtype=RESIDUAL_DTYPE)
        on = seq if seq.residual is None else seq.residual
        residual = seq.__class__.apply_laplacian(on, A, 1, dirichlet=False) - rhs.astype(RESIDUAL_DTYPE)
        Bh = seq.apply_strong_curl(A.astype(mrx.DTYPE), dirichlet_in=False, dirichlet_out=False)
    err_sq = float(Bh @ seq.apply_mass_matrix(Bh, k, False)) - 2.0 * float(Bh @ load) + bstar_sq
    err = float(np.sqrt(max(err_sq, 0.0)) / np.sqrt(bstar_sq))

    def norm(v):   # the stopping criterion's norm: the mass atom of the dual forms of the solve
        kk = 0 if k == 1 else 1
        return float(jnp.sqrt(v @ seq.apply_mass_matrix_preconditioner(v.astype(mrx.DTYPE), kk, False)))
    rel_res = norm(residual) / norm(rhs)
    print(f"\n  k={k}: relative L2 error {err:.3e}, {-int(info)} iterations, residual {rel_res:.2e}")
    assert int(info) < 0, f"k={k} did not converge (info={int(info)})"
    assert rel_res <= 1e2 * seq.tol
    assert err < ERROR_BAND[k]


@pytest.mark.parametrize("k", (2, 1))
def test_leray_projection(seq, k):
    """``P v`` is divergence-free at solver tolerance, ``P P v = P v`` and
    ``||P v||_M <= ||v||_M``. k=2 is the relaxation's projection (Dirichlet
    spaces, k=3 pressure); k=1 the free-space one through the k=0 Laplacian.
    At k=2 the potential route of the descent (``TimeStepper.potential_velocity``:
    ``curl a + c h`` with ``L_1 a = curl^T M_2 v`` in the Coulomb gauge and ``c
    h`` the harmonic part) must give the same projection: one k=1 Hodge solve
    against the saddle solve."""
    dbc = k == 2
    seq = seq.even                                # a velocity is even
    v = jnp.asarray(np.random.default_rng(5 * k).standard_normal(seq.n(k, dbc)), dtype=mrx.DTYPE)
    v = v / seq.l2_norm(v, k, dirichlet=dbc)
    Pv, p = seq.apply_leray_projection(v, k=k)
    PPv, _ = seq.apply_leray_projection(Pv, k=k, p_guess=p)
    if k == 2:
        div = seq.apply_incidence_matrix(Pv, 2, dirichlet_in=True, dirichlet_out=True)
        div_norm = float(seq.l2_norm(div, 3, dirichlet=True))
    else:
        div = seq.apply_derivative_matrix(Pv, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
        div_norm = float(jnp.linalg.norm(div))
    moved = float(seq.l2_norm(PPv - Pv, k, dirichlet=dbc))
    e_v = float(seq.l2_norm_sq(v, k, dirichlet=dbc))
    e_Pv = float(seq.l2_norm_sq(Pv, k, dirichlet=dbc))
    print(f"\n  Leray k={k}: ||div P v|| {div_norm:.2e}, ||P P v - P v|| {moved:.2e}, "
          f"energy {e_v:.4f} -> {e_Pv:.4f}")
    # The projection is stored in the working dtype: the divergence of the
    # rounding alone is a few eps, on top of the solve's tolerance.
    assert div_norm <= 10 * seq.tol + eps(10)
    assert moved <= 10 * seq.tol + eps(10)
    assert e_Pv < e_v
    if k == 2:
        Mv = seq.apply_mass_matrix(v, 2, True)
        a = seq.apply_inverse_laplacian(
            seq.apply_incidence_matrix(Mv, 1, dirichlet_in=True, dirichlet_out=True, transpose=True), 1)
        hs = seq.nullspace(2, True)                  # none on the even view: the harmonic 2-form is odd
        Pv_pot = seq.apply_incidence_matrix(a, 1, dirichlet_in=True, dirichlet_out=True) \
            + hs.T @ ((hs @ Mv) / jnp.asarray([h @ seq.apply_mass_matrix(h, 2, True) for h in hs]))
        rel = float(seq.l2_norm(Pv_pot - Pv, 2) / seq.l2_norm(Pv, 2))
        print(f"  potential route vs Leray: |curl a + c h - P v| / |P v| = {rel:.2e}")
        # both solves stop at seq.tol relative to their right-hand sides, the
        # projection is a fraction of v
        assert rel < 1e2 * seq.tol * float(seq.l2_norm(v, 2) / seq.l2_norm(Pv, 2)) + eps(1e4)


def _volume_integrals(seq, values):
    """``int values dV`` on the quadrature grid, ``values`` per quadrature point."""
    return float(jnp.sum(seq.quad.w * seq.jacobian_j * values))


@pytest.mark.parametrize("k", (0, 3))
def test_manufactured_scalar_solution(seq, k):
    seq = seq.even                                # psi and its gradient are even
    if k == 0:
        grad_psi = jax.grad(psi)
        load = seq.load(lambda xi: grad_psi(seq.map(xi)), 1, dirichlet=False)
        rhs = seq.apply_incidence_matrix(load, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
        f, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=False, return_info=True, dtype=RESIDUAL_DTYPE)
        residual = seq.apply_stiffness(f, 0, dirichlet=False) - rhs
        f = f.astype(mrx.DTYPE)
        # the constant: the 0-forms contain 1 exactly (partition of unity), the
        # all-ones vector of the base; on the even view its reduced coefficients
        # are sqrt(2) on every orbit pair, so reduce it rather than write ones
        psi_q = jax.vmap(lambda xi: psi(seq.map(xi)))(seq.quad.x)
        x = seq.reduction[(0, False)]       # X: reduced -> free, so X^T reduces the free all-ones vector
        one = (x.T @ jnp.ones(x.shape[0], dtype=mrx.DTYPE)).astype(mrx.DTYPE)
        M1 = seq.apply_mass_matrix(one, 0, False)
        f = f + (_volume_integrals(seq, psi_q) - float(one @ seq.apply_mass_matrix(f, 0, False))) / float(one @ M1) * one
        load0 = seq.load(lambda xi: psi(seq.map(xi)), 0, dirichlet=False)
        target_sq = _volume_integrals(seq, psi_q ** 2)
        err_sq = float(f @ seq.apply_mass_matrix(f, 0, False)) - 2.0 * float(f @ load0) + target_sq
        what, k_res, dbc = "psi", 0, False
    else:
        def rho(xi):                        # vanishes on the wall r = 1: in the range of the weak gradient
            return (1.0 - xi[0] ** 2) * psi(seq.map(xi))

        def grad_rho(xi):                   # the physical gradient, DF^-T d rho / d xi
            return jnp.linalg.solve(jax.jacfwd(seq.map)(xi).T, jax.grad(rho)(xi))
        load = seq.load(grad_rho, 2, dirichlet=False)
        # rhs_i = <grad rho*, delta Lambda_i>, delta = -M_2^-1 D_2^T M_3 the weak gradient into the free 2-forms
        w = seq.apply_inverse_mass_matrix(load, 2, dirichlet=False)
        rhs = -seq.apply_mass_matrix(seq.apply_incidence_matrix(w, 2, dirichlet_in=False, dirichlet_out=False), 3, False)
        r3, info = seq.apply_inverse_laplacian(rhs, 3, dirichlet=False, return_info=True, dtype=RESIDUAL_DTYPE)
        on = seq if seq.residual is None else seq.residual
        residual = seq.__class__.apply_laplacian(on, r3, 3, dirichlet=False) - rhs.astype(RESIDUAL_DTYPE)
        g = -seq.apply_inverse_mass_matrix(
            seq.apply_derivative_matrix(r3.astype(mrx.DTYPE), 2, dirichlet_in=False, dirichlet_out=False, transpose=True),
            2, dirichlet=False)
        gq = jax.vmap(grad_rho)(seq.quad.x)
        target_sq = _volume_integrals(seq, jnp.sum(gq ** 2, axis=1))
        err_sq = float(g @ seq.apply_mass_matrix(g, 2, False)) - 2.0 * float(g @ load) + target_sq
        what, k_res, dbc = "grad rho*", 3, False
    err = float(np.sqrt(max(err_sq, 0.0)) / np.sqrt(target_sq))

    def norm(v):
        return float(jnp.sqrt(v @ seq.apply_mass_matrix_preconditioner(v.astype(mrx.DTYPE), k_res, dbc)))
    rel_res = norm(residual) / norm(rhs)
    print(f"\n  k={k}: relative L2 error of {what} {err:.3e}, {-int(info)} iterations, residual {rel_res:.2e}")
    assert int(info) < 0, f"k={k} did not converge (info={int(info)})"
    assert rel_res <= 1e2 * seq.tol
    assert err < SCALAR_ERROR_BAND[k]
