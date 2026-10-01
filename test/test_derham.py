"""The discrete spaces form a de Rham complex.

``V0 --grad--> V1 --curl--> V2 --div--> V3`` with the polar strong derivatives ``G_k``:

- the complex is exact, ``G_{k+1} G_k = 0``, on the free and the Dirichlet spaces and on both parity views.
- the projections commute with the derivatives, ``G_k Pi_k f = Pi_{k+1} d f`` for a smooth physical ``f``.
- the mass matrices are symmetric positive definite, the weak derivative is ``D_k = M_{k+1} G_k``, and the
  projection matrices of complementary degree are adjoint.
- the harmonic forms span the cohomology of the solid torus, Betti numbers ``(1, 1, 0, 0)``: one free 1-form and
  one Dirichlet 2-form, harmonic to round-off and orthonormal.
- pushforward inverts pullback.

Every statement is an identity of the discretisation, so it holds to round-off, or to the solver tolerance
where a stored solve result enters.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.differential_forms import Pullback, Pushforward
from mrx.nullspace import harmonic_rayleigh
from mrx.precision import eps

# The polar derivative involves an inverse near the axis, which costs accuracy:
# 1e4 eps (2.2e-12 in float64, 1.2e-3 in float32).
EXACT = mrx.eps(1e4)
# Rayleigh quotient of a stored harmonic form: round-off is below 1e-9 in every configuration, an unconverged
# solve leaves 1e-3.
HARMONIC = 1e-8


def _random(space, k, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(space.n(k)), dtype=mrx.DTYPE)


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


# Logical test forms with the period of the sequence and regular at the axis (each component vanishes there as
# fast as its form requires). The derivative of a logical form is metric-free, so the commuting identity holds
# up to the Gauss quadrature of the trigonometric factors, far below the bands.
TWO_PI = 2.0 * np.pi


def _f0(x):
    return x[0] ** 2 * jnp.cos(TWO_PI * x[1]) + jnp.sin(TWO_PI * x[2])


def _a1(x):
    r, t, z = x
    return jnp.array([r * jnp.sin(TWO_PI * z), r ** 2 * jnp.sin(TWO_PI * z), r ** 2 * jnp.cos(TWO_PI * t)])


def _b2(x):
    r, t, z = x
    return jnp.array([r ** 2 * jnp.sin(TWO_PI * z), r ** 2 * jnp.cos(TWO_PI * z), r * (1.0 + r ** 2 * jnp.cos(TWO_PI * t))])


def _d(f, k):
    """The exterior derivative of a logical k-form, in logical components."""
    def df(x):
        D = jax.jacfwd(f)(x)
        if k == 0:
            return D
        if k == 1:      # curl: the 2-form (d_t a_z - d_z a_t, d_z a_r - d_r a_z, d_r a_t - d_t a_r)
            return jnp.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
        return jnp.trace(D)
    return df


@pytest.mark.parametrize("k", (0, 1, 2))
def test_projections_commute_with_the_derivative(domains, k):
    """``G_k Pi_k f = Pi_{k+1} d f`` on the free spaces of one field period, for a logical k-form ``f`` and its
    exterior derivative ``d f``: gradient (k = 0), curl (k = 1) and divergence (k = 2)."""
    space = domains("period").free
    f = (_f0, _a1, _b2)[k]
    lhs = space.G[k] @ space.interpolate(f, k, frame="logical")
    rhs = space.interpolate(_d(f, k), k + 1, frame="logical")
    rel = float(jnp.linalg.norm(lhs - rhs) / jnp.linalg.norm(rhs))
    print(f"\n  k={k}: |G Pi f - Pi d f| / |Pi d f| = {rel:.2e}")
    assert rel < max(EXACT, 1e-7)


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


def test_pushforward_inverts_pullback(seq):
    """``pushforward(pullback(g))(x) = g(Phi(x))`` for k = 0, 1, 2, 3 and random affine physical forms, at
    random points off the axis."""
    rng = np.random.default_rng(3)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.15, 1.0, 6), rng.uniform(0.0, 1.0, 6),
                                       rng.uniform(0.0, 1.0, 6)]))
    for k in range(4):
        n_comp = 1 if k in (0, 3) else 3
        c = jnp.asarray(rng.standard_normal((n_comp, 4)))

        def g(y, c=c, n_comp=n_comp):
            v = c[:, 0] + c[:, 1:] @ y
            return v[0] if n_comp == 1 else v
        forth = Pushforward(Pullback(g, seq.map, k), seq.map, k)
        got = np.asarray(jnp.stack([forth(x) for x in pts]))
        want = np.asarray(jnp.stack([g(seq.map(x)) for x in pts]))
        err = np.abs(got - want).max() / np.abs(want).max()
        assert err < 1e3 * eps(), f"k={k}: pushforward(pullback(g)) off by {err:.2e}"
