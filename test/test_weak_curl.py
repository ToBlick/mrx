"""
Weak-curl manufactured-solution tests (theta-pinch and z-pinch).

Dense-only: m2/M2, m1/M1, weak_curl, p2/P2, p1/P1. Skips if assembly or any of these are missing.
"""
import pytest
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import cylinder_annulus_map

jax.config.update("jax_enable_x64", True)


def _get(seq, low, high):
    a = getattr(seq, low, None)
    if a is not None:
        return a
    return getattr(seq, high, None)


def make_seq():
    """DeRhamSequence on cylinder annulus. Dense assembly only."""
    seq = DeRhamSequence(
        (4, 4, 4),
        (2, 2, 2),
        5,
        ("clamped", "periodic", "periodic"),
        cylinder_annulus_map(R_min=0.2, R_max=1.0, L=2 * jnp.pi),
        False,
    )
    seq.evaluate_1d()
    assemble = getattr(seq, "assemble_all", None)
    if assemble is None:
        assemble = getattr(seq, "assemble", None)
    if assemble is None:
        return None
    assemble()
    return seq


def _require_dense(seq):
    #I'm adding this because a lot of variable names have changed in the sparse branch
    # so trying to make it flexible.
    
    if seq is None:
        pytest.skip("no assemble_all / assemble (dense-only test)")
    m2 = _get(seq, "m2", "M2")
    m1 = _get(seq, "m1", "M1")
    p2 = _get(seq, "p2", "P2")
    p1 = _get(seq, "p1", "P1")
    weak_curl = getattr(seq, "weak_curl", None)
    if weak_curl is None:
        weak_curl = getattr(seq, "Weak_curl", None)
    if None in (m2, m1, p2, p1, weak_curl):
        missing = [n for n, v in [("m2", m2), ("m1", m1), ("p2", p2), ("p1", p1), ("weak_curl", weak_curl)] if v is None]
        pytest.skip(f"dense API missing: {missing}")
    return m2, m1, p2, p1, weak_curl


def test_weak_curl_theta_pinch_manufactured_solution():
    """Theta-pinch: B = B_z(r) e_z, J = curl B. Compare weak_curl @ B_hat to p1(J) in M1 norm."""
    seq = make_seq()
    m2, m1, p2, p1, weak_curl = _require_dense(seq)

    B0 = 1.0
    R_min, R_max = 0.2, 1.0
    dR = R_max - R_min

    def B_xyz(x):
        r, θ, z = x
        return jnp.array([0.0, 0.0, B0 * (1.0 - r**2)])

    def J_xyz(x):
        r, θ, z = x
        J_θ = 2.0 * B0 * r / dR
        φ = 2.0 * jnp.pi * θ
        return jnp.array([-J_θ * jnp.sin(φ), J_θ * jnp.cos(φ), 0.0])

    B_hat = jnp.linalg.solve(m2, p2(B_xyz))
    J_hat = weak_curl @ B_hat
    J_exact_hat = jnp.linalg.solve(m1, p1(J_xyz))

    diff = J_hat - J_exact_hat
    err_rel = float(jnp.asarray(
        jnp.sqrt(diff @ m1 @ diff) / (jnp.sqrt(J_exact_hat @ m1 @ J_exact_hat) + 1e-14)
    ).item())
    assert err_rel < 0.3, f"weak_curl theta-pinch error too large: {err_rel:.2e}"


def test_weak_curl_z_pinch_manufactured_solution():
    """Z-pinch: B = B_θ(r) e_θ, J = curl B. Compare weak_curl @ B_hat to p1(J) in M1 norm."""
    seq = make_seq()
    m2, m1, p2, p1, weak_curl = _require_dense(seq)

    B0 = 1.0
    R_min, R_max = 0.2, 1.0
    dR = R_max - R_min

    def B_xyz(x):
        r, θ, z = x
        B_θ = B0 * r
        φ = 2.0 * jnp.pi * θ
        return jnp.array([-B_θ * jnp.sin(φ), B_θ * jnp.cos(φ), 0.0])

    def J_xyz(x):
        r, θ, z = x
        R = R_min + r * dR
        J_z = B0 * (R_min + 2.0 * r * dR) / (R * dR)
        return jnp.array([0.0, 0.0, J_z])

    B_hat = jnp.linalg.solve(m2, p2(B_xyz))
    J_hat = weak_curl @ B_hat
    J_exact_hat = jnp.linalg.solve(m1, p1(J_xyz))

    diff = J_hat - J_exact_hat
    err_rel = float(jnp.asarray(
        jnp.sqrt(diff @ m1 @ diff) / (jnp.sqrt(J_exact_hat @ m1 @ J_exact_hat) + 1e-14)
    ).item())
    assert err_rel < 0.3, f"weak_curl z-pinch error too large: {err_rel:.2e}"
