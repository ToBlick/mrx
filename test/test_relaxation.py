"""Integration tests for mrx.relaxation on the z-pinch equilibrium.

Two checks per test run (shared module-scoped assembly):
1. ``test_zpinch_force_balance``  — the Lorentz force is small when B is
   projected onto the equilibrium.
2. ``test_zpinch_pressure_recovery`` — the 3-form pressure returned by
   ``compute_force`` agrees with the known analytical z-pinch pressure up
   to the spline approximation error.
"""

import jax
import jax.numpy as jnp
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.operators import assemble_mass_jacobi_preconditioner
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import cylinder_map
from mrx.relaxation import compute_force


# The two assertions are spline-approximation errors at (6,6,1) p=3 (measured
# 3.4e-2 for the pressure), so the resolution is the point: gpu tier.
pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Analytical z-pinch helpers
# ---------------------------------------------------------------------------
#
# Equilibrium: B_theta(r) = 2r(1 - r^2/2), purely azimuthal.
# Exact pressure: p(r) = 1/3 (5 - 2r^2)(1 - r^2)^2,
#   which satisfies dp/dr = J_z * B_theta and p(1) = 0.


def _B0(x):
    """Physical z-pinch magnetic field at logical point x, Cartesian components."""
    r, theta, _ = x
    pi = jnp.pi
    Btheta = 2 * r * (1 - r ** 2 / 2)
    return jnp.array([-jnp.sin(2 * pi * theta),
                       jnp.cos(2 * pi * theta),
                       0.0]) * Btheta


def _p_exact(x):
    """Exact z-pinch pressure at logical point x (scalar, shape (1,))."""
    r, _, _ = x
    return (1 / 3 * (5 - 2 * r ** 2) * (1 - r ** 2) ** 2) * jnp.ones(1)


# ---------------------------------------------------------------------------
# Fixtures — built once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zpinch_seq():
    """Z-pinch sequence: cylinder geometry, (6,6,1) p=3, no z-variation."""
    F_cyl = cylinder_map(a=1.0, h=1.0)
    seq = DeRhamSequence(
        (6, 6, 1), (3, 3, 0), 6,
        ("clamped", "periodic", "constant"),
        polar=True,
        betti_numbers=(1, 0, 0, 0),
    )
    seq.evaluate_1d()
    seq.set_map(F_cyl)
    # Skip the eager payloads (CP/NTF fits, core Schur) that production no
    # longer uses, but the Leray/force solves DO read the Jacobi mass
    # diagonals, so assemble just those.
    seq.assemble_all_sparse()
    seq.set_operators(assemble_mass_jacobi_preconditioner(
        seq, seq.get_operators(), ks=(0, 1, 2, 3)))
    return seq


@pytest.fixture(scope="module")
def zpinch_B_hat(zpinch_seq):
    """Equilibrium B: loaded as k=2 DBC, L2-projected, then Leray-cleaned."""
    seq = zpinch_seq
    rhs = seq.load(_B0, 2, dirichlet=True)
    B = seq.apply_inverse_mass_matrix(rhs, 2, dirichlet=True)
    B, _ = seq.apply_leray_projection(B, k=2)
    return B


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zpinch_force_balance(zpinch_seq, zpinch_B_hat):
    """Lorentz force is small for the (approximately) equilibrium B."""
    seq = zpinch_seq
    F_force, _, _, _, _ = compute_force(zpinch_B_hat, seq)
    F_norm = seq.l2_norm(F_force, 2, dirichlet=True)
    B_norm = seq.l2_norm(zpinch_B_hat, 2, dirichlet=True)
    rel = float(F_norm / B_norm)
    print(f"\n  z-pinch relative force norm: {rel:.3e}")
    assert rel < 1e-1, (
        f"z-pinch force not small: |F|/|B| = {rel:.3e}")


def test_zpinch_pressure_recovery(zpinch_seq, zpinch_B_hat):
    """Pressure from compute_force matches the exact z-pinch pressure."""
    seq = zpinch_seq
    F_cyl = cylinder_map(a=1.0, h=1.0)

    _, p_dofs, _, _, _ = compute_force(zpinch_B_hat, seq)

    # The Leray projection for k=2 returns a 3-form DBC pressure.
    # Push forward to get the physical scalar: p_phys = p_log / J.
    p_h = DiscreteFunction(p_dofs, seq.basis_3, seq.e3_dbc)
    p_phys = Pushforward(p_h, F_cyl, k=3)

    p_h_vals = jax.lax.map(p_phys, seq.quad.x, batch_size=20_000)
    p_exact_vals = jax.vmap(_p_exact)(seq.quad.x)

    # The Leray multiplier's gauge (additive constant) is solver-defined,
    # while the analytic pressure is gauged by p(1) = 0 -- compare
    # gauge-invariantly by removing the weighted mean from both. Measured
    # ~3.4e-2 at (6,6,1) p=3 (spline approximation error).
    wJ = seq.jacobian_j * seq.quad.w
    mean_h = jnp.einsum("ik,i->", p_h_vals, wJ) / wJ.sum()
    mean_e = jnp.einsum("ik,i->", p_exact_vals, wJ) / wJ.sum()
    diff = (p_h_vals - mean_h) - (p_exact_vals - mean_e)
    ref = p_exact_vals - mean_e
    l2_diff = jnp.einsum("ik,ik,i->", diff, diff, wJ)
    l2_ref = jnp.einsum("ik,ik,i->", ref, ref, wJ)
    rel_err = float(jnp.sqrt(l2_diff / l2_ref))
    print(f"\n  z-pinch pressure relative L2 error (gauge-fixed): {rel_err:.3e}")
    assert rel_err < 1e-1, (
        f"z-pinch pressure L2 error too large: {rel_err:.3e}")

