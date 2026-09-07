"""The second variation is the Hessian of the pushed-forward energy, and the
Newton direction it defines is divergence-free and descends.

On the li383 fixture's own field: for a random divergence-free ``u`` the
first derivative of the energy along the flow, ``(B, curl(u x B))_M``, is
minus the force's pairing ``(u, J x B)_M``; the second derivative along the
second-order flow, ``||Q||^2 + (B, curl(u x Q))_M``, is the quadratic form
``(u, H u)``; ``H`` is symmetric. The Newton direction ``curl a`` of a few
MINRES iterations has zero divergence to roundoff and a positive pairing
with the force.
"""
import jax
import jax.numpy as jnp

from mrx.hessian import newton_direction, second_variation
from mrx.precision import DTYPE, eps
from mrx.relaxation import compute_force


def _divergence_free(seq, key):
    w = jax.random.normal(key, (seq.n(2, True),), dtype=DTYPE)
    w, _ = seq.apply_leray_projection(w, k=2)
    return w / seq.l2_norm(w, 2)


def _curl_cross(seq, u, X):
    """``curl(u x X)`` of the 2-forms ``u`` and ``X``: the ideal increment."""
    E = seq.apply_inverse_mass_matrix(seq.cross_product_load(u, X, 1, 2, 2), 1)
    return seq.apply_incidence_matrix(E, 1)


def test_second_variation_is_the_energy_hessian(seq, b0):
    _, _, J, _, JxB = compute_force(b0, seq)
    H = second_variation(seq, b0, J)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    u, v = _divergence_free(seq, k1), _divergence_free(seq, k2)
    Hu, Hv = H(u), H(v)

    Q = _curl_cross(seq, u, b0)
    dE = float(b0 @ seq.apply_mass_matrix(Q, 2))
    force = -float(u @ seq.apply_mass_matrix(JxB, 2))
    R = _curl_cross(seq, u, Q)
    d2E = float(seq.l2_norm_sq(Q, 2) + b0 @ seq.apply_mass_matrix(R, 2))
    quad = float(u @ Hu)
    cross, cross_t = float(u @ Hv), float(v @ Hu)
    band = 1e3 * max(seq.tol, eps())
    print(f"\n  dE {dE:+.6e} vs -(u, J x B) {force:+.6e};  (u, H u) {quad:+.6e} vs "
          f"||Q||^2 + (B, R) {d2E:+.6e};  (u, H v) {cross:+.6e} vs (v, H u) {cross_t:+.6e}")
    assert abs(dE - force) < band * abs(force)
    assert abs(quad - d2E) < band * abs(d2E)
    assert abs(cross - cross_t) < band * abs(cross)


def test_newton_direction_is_divergence_free_and_descends(seq, b0):
    F, _, J, _, _ = compute_force(b0, seq)
    MF = seq.apply_mass_matrix(F, 2)
    u, a, info = newton_direction(seq, b0, J, MF, jnp.zeros(seq.n(1, True), dtype=DTYPE),
                                  tol=1e-2, maxiter=40)
    div = float(seq.l2_norm(seq.apply_incidence_matrix(u, 2), 3))
    cos = float(u @ MF) / float(seq.l2_norm(u, 2) * jnp.sqrt(F @ MF))
    print(f"\n  MINRES info {int(info)}, ||div u|| {div:.2e}, descent cosine {cos:+.4f}")
    assert div < 1e2 * eps() * float(seq.l2_norm(u, 2))
    assert cos > 0.0
