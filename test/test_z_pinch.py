# %%
# 1D Z pinch from Freidberg

import jax
import jax.numpy as jnp
import numpy.testing as npt

from jax.numpy import cos, sin, pi
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.relaxation import MRXDiagnostics

jax.config.update("jax_enable_x64", True)

# --- helper functions ---
def F(p):
    """Cylindrical coordinates"""
    r, θ, z = p
    return jnp.array([r * cos(2 * pi * θ), r * sin(2 * pi * θ), z])


def B0(p):
    r, θ, z = p
    Bθ = 2 * r * (1 - r**2 / 2)
    return jnp.array([-sin(2 * pi * θ), cos(2 * pi * θ), 0.0]) * Bθ


def p_exact(x):
    r, θ, z = x
    return 1/3 * (5 - 2*r**2) * (1 - r**2)**2 * jnp.ones(1)


# --- pytest test ---
def test_zpinch_pressure_convergence():
    Seq = DeRhamSequence(
        (6, 6, 1),
        (3, 3, 0),
        3,
        ("clamped", "periodic", "constant"),
        F,
        polar=True,
        dirichlet=True
    )

    Seq.evaluate_1d()
    Seq.assemble_all()
    Seq.build_crossproduct_projections()
    Seq.assemble_leray_projection()

    # --- compute pressure approximation ---
    B_hat = jnp.linalg.solve(Seq.M2, Seq.P2(B0))
    p_hat = MRXDiagnostics(Seq).pressure(B_hat)
    p_h = DiscreteFunction(p_hat, Seq.Λ0, Seq.E0)

    # --- error evaluation ---
    def diff_at_x(x):
        return p_exact(x) - p_h(x)

    def body_fun(carry, x):
        return None, diff_at_x(x)

    _, df = jax.lax.scan(body_fun, None, Seq.Q.x)

    L2_dp = jnp.einsum('ik,ik,i,i->', df, df, Seq.J_j, Seq.Q.w)**0.5
    L2_p = jnp.einsum('ik,ik,i,i->',
                      jax.vmap(p_exact)(Seq.Q.x),
                      jax.vmap(p_exact)(Seq.Q.x),
                      Seq.J_j, Seq.Q.w)**0.5

    error = L2_dp / L2_p

    # --- numerical tolerance check ---
    npt.assert_allclose(error, 0.0, atol=5e-2, err_msg=f"L2 error too large: {error}")

# Run via:
#    pytest -v test_zpinch.py
