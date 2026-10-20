"""The discrete spaces form a de Rham complex.

``V0 --grad--> V1 --curl--> V2 --div--> V3`` with the polar strong derivatives ``G_k``:

- the complex is exact, ``G_{k+1} G_k = 0``, on the free and the Dirichlet spaces and on both parity views.
- the mass matrices are symmetric positive definite, the weak derivative is ``D_k = M_{k+1} G_k``, and the
  projection matrices of complementary degree are adjoint.
- the harmonic forms span the cohomology of the solid torus, Betti numbers ``(1, 1, 0, 0)``: one free 1-form and
  one Dirichlet 2-form, harmonic to round-off and orthonormal.
- the extracted derivatives are the raw ones of the tensor-product space.
- interpolation commutes with the derivative, ``G_k Pi_k f = Pi_{k+1} df``, for fields with the regularity of a smooth
  form at the axis, axis rows included.

The spaces are the graded polar ones of production.

Every statement is an identity of the discretisation, so it holds to round-off, or to the solver tolerance
where a stored solve result enters.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.nullspace import harmonic_rayleigh
from mrx.precision import eps

# The polar derivative involves an inverse near the axis, which costs accuracy:
# 1e4 eps (2.2e-12).
EXACT = mrx.eps(1e4)
# Rayleigh quotient of a stored harmonic form: round-off is below 1e-9, an unconverged
# solve leaves 1e-3.
HARMONIC = 1e-8


def _random(space, k, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(space.n(k)))


def _views(seq):
    """The spaces a field can live in: both parity views, each with Dirichlet and free boundary conditions."""
    return {f"{name}{' free' if free else ''}": (view.free if free else view)
            for name, view in (("odd", seq.odd), ("even", seq.even)) for free in (False, True)}


@pytest.mark.parametrize("k", (0, 1))
def test_the_complex_is_exact(seq, k):
    """``G_{k+1} G_k v = 0`` for random k-forms ``v``, relative to ``||G_k v||``."""
    for name, space in _views(seq).items():
        g = space.G[k] @ _random(space, k, 11 + k)
        rel = float(jnp.linalg.norm(space.G[k + 1] @ g) / jnp.linalg.norm(g))
        assert rel < EXACT, f"{name}, k={k}: d d != 0 ({rel:.2e})"


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_and_weak_derivative(seq, k):
    """``M_k`` is symmetric positive definite, ``D_k = M_{k+1} G_k`` (k < 3), and ``P_12``, ``P_21`` and
    ``P_03``, ``P_30`` are adjoint, on the odd view."""
    space = seq.odd
    u, v = _random(space, k, 1), _random(space, k, 2)
    uMv, vMu, vMv = float(u @ (space.M[k] @ v)), float(v @ (space.M[k] @ u)), float(v @ (space.M[k] @ v))
    assert abs(uMv - vMu) < EXACT * abs(vMv) and vMv > 0.0
    if k < 3:
        w = _random(space, k + 1, 3)
        Dv = space.D[k] @ v
        assert float(jnp.linalg.norm(Dv - space.M[k + 1] @ (space.G[k] @ v))) < EXACT * float(jnp.linalg.norm(Dv))
        assert abs(float(w @ Dv) - float((space.D[k].T @ w) @ v)) < EXACT * float(jnp.linalg.norm(w) * jnp.linalg.norm(Dv))
    partner = {1: 2, 2: 1, 0: 3, 3: 0}[k]
    w = _random(space, partner, 4)
    lhs, rhs = float(w @ (space.P[k, partner] @ v)), float(v @ (space.P[partner, k] @ w))
    assert abs(lhs - rhs) < EXACT * abs(lhs), f"P_{k}{partner}: {lhs:.6e} vs {rhs:.6e}"


def test_harmonic_forms_span_the_cohomology(seq):
    """One harmonic free 1-form and one harmonic Dirichlet 2-form (``b1 = 1`` and, by duality, the
    Dirichlet ``b2``), none at k = 0 Dirichlet or k = 3 free, each with a Rayleigh quotient at round-off and
    unit mass norm. Under stellarator symmetry both are odd."""
    odd = seq.odd
    for space, k in ((odd.free, 1), (odd, 2)):
        vs = space.nullspace(k)
        assert vs.shape[0] == 1, f"k={k}: {vs.shape[0]} harmonic forms"
        norm = float(vs[0] @ (space.M[k] @ vs[0]))
        rayleigh = harmonic_rayleigh(space, vs[0], k)
        print(f"\n  k={k}: Rayleigh {rayleigh:.2e}, ||h||_M^2 {norm:.8f}")
        assert abs(rayleigh) < HARMONIC
        assert abs(norm - 1.0) < seq.tol + eps(10)
    assert seq.nullspace(0).shape[0] == 0 and seq.free.nullspace(3).shape[0] == 0


def test_the_extracted_derivative_is_the_raw_one(seq):
    """The polar extraction commutes with the derivatives of the graded tensor-product space,
    ``E_{k+1}^T G_k v = G^raw_k E_k^T v`` (k = 0, 1, 2)."""
    for k, raw in enumerate(seq.incidence):
        v = _random(seq, k, 41 + k)
        lhs = seq.E(k + 1).T @ (seq.G[k] @ v)
        rel = float(jnp.linalg.norm(lhs - raw @ (seq.E(k).T @ v)) / jnp.linalg.norm(lhs))
        assert rel < EXACT, f"k={k}: the extracted derivative is not the raw one ({rel:.2e})"


def test_interpolation_commutes_with_the_derivative(seq):
    """``G_k Pi_k f = Pi_{k+1} df`` (k = 0, 1, 2) for ``seq.interpolate`` on the free graded spaces, relative to
    ``||Pi_{k+1} df||``.

    In theta and zeta ``f`` is a random spline on the knots of the sequence and in ``r`` a polynomial, so every
    integral of the histopolation is exact and the identity holds to round-off, axis rows included. The fields do not
    vanish at the axis. They meet only the conditions of a smooth form there: a 0-form and the zeta part of a
    1-form do not depend on theta at ``r = 0``, and the theta part of a 1-form and the ``dtheta dzeta`` part of a
    2-form vanish at ``r = 0``."""
    space = seq.free
    theta, zeta = seq.basis_0.Lambda[1], seq.basis_0.Lambda[2]
    rng = np.random.default_rng(7)
    a_t, a_z = (jnp.asarray(rng.standard_normal((3, b.n))) for b in (theta, zeta))
    a_0 = jnp.asarray(rng.standard_normal(zeta.n))

    def g(x, i):
        (vt, it), (vz, iz) = theta.evaluate_local(x[1]), zeta.evaluate_local(x[2])
        return (vt @ a_t[i, it]) * (vz @ a_z[i, iz])

    def h(x):
        # constant in theta, the value of a 0-form on the axis
        vz, iz = zeta.evaluate_local(x[2])
        return vz @ a_0[iz]

    def q(r):
        return r + r ** 2

    fields = (lambda x: h(x) + q(x[0]) * g(x, 0),
              lambda x: jnp.array([(1 + x[0]) * g(x, 0), q(x[0]) * g(x, 1), h(x) + q(x[0]) * g(x, 2)]),
              lambda x: jnp.array([q(x[0]) * g(x, 0), (1 + x[0]) * g(x, 1), (1 + x[0]) * g(x, 2)]))

    def d(f, k):
        # the exterior derivative in logical components
        if k == 0:
            return jax.grad(f)
        if k == 1:
            def curl(x):
                j = jax.jacfwd(f)(x)            # j[i, a] = d_a f_i
                return jnp.array([j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]])
            return curl
        return lambda x: jnp.trace(jax.jacfwd(f)(x))

    for k, f in enumerate(fields):
        rhs = space.interpolate(d(f, k), k + 1)
        rel = float(jnp.linalg.norm(space.G[k] @ space.interpolate(f, k) - rhs) / jnp.linalg.norm(rhs))
        print(f"\n  k={k}: |G Pi f - Pi df| / |Pi df| = {rel:.1e}")
        assert rel < EXACT, f"k={k}: interpolation does not commute with the derivative ({rel:.2e})"
