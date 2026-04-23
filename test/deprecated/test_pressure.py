"""Pressure accuracy test at n=8, degree=2."""
import jax
import jax.numpy as jnp
from jax.numpy import cos, pi, sin

import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.relaxation import compute_force
from mrx.utils import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# Betti numbers for a 2D disk (contractible, polar domain)
BETTI = [1, 0, 0, 0]
DEGREE = 2
N = 8


def F(p):
    r, θ, z = p
    return jnp.array([r * cos(2 * pi * θ), r * sin(2 * pi * θ), z])


def B0(p):
    r, θ, _ = p
    Bθ = 2 * r * (1 - r**2 / 2)
    return jnp.array([-sin(2 * pi * θ), cos(2 * pi * θ), 0.0]) * Bθ


def p_exact(x):
    r, _, _ = x
    return 1 / 3 * (5 - 2 * r**2) * (1 - r**2)**2


@pytest.fixture(scope="module")
def pressure_errors():
    """Compute p0 and p3 relative L2 pressure errors at n=8, degree=2."""
    seq = DeRhamSequence(
        (N, N, 1), (DEGREE, DEGREE, 0), DEGREE + 2,
        ("clamped", "periodic", "constant"), F, polar=True)
    seq.evaluate_1d()
    seq.assemble_all_sparse()
    seq._compute_nullspaces(BETTI)

    rhs = seq.p2_dbc(B0)
    B = seq.apply_inverse_mass_matrix(rhs, 2)
    B, _ = seq.apply_leray_projection(B, k=2)

    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    wJ = seq.jacobian_j * seq.quad.w
    V = jnp.sum(wJ)
    pe = jax.vmap(p_exact)(seq.quad.x)
    mean_pe = jnp.sum(pe * wJ) / V

    # Call compute_force once; reuse J and H for both pressure paths.
    _, p3, J, H, _ = compute_force(B, seq)

    # 0-form pressure (k=1 Leray): JxH projected to 1-form
    JxH1_dual = seq.cross_product_projection(J, H, 1, 1, 1, False, True, False)
    JxH1 = seq.apply_inverse_mass_matrix(JxH1_dual, 1, dirichlet=False)
    _, p0 = seq.apply_leray_projection(JxH1, k=1)

    comp_info_0, comp_shapes_0 = seq._form_comp_info(0)
    p0_jk = -evaluate_at_xq(
        seq.e0_T @ p0, comp_info_0, comp_shapes_0, quad_shape, 1)
    mean_p0 = jnp.einsum("ik,i->", p0_jk, wJ) / V
    d0 = (p0_jk[:, 0] - mean_p0) - (pe - mean_pe)
    err_p0 = float((jnp.sum(d0**2 * wJ) / jnp.sum((pe - mean_pe)**2 * wJ)) ** 0.5)

    # 3-form pressure (k=2 Leray, already computed by compute_force)
    comp_info_3, comp_shapes_3 = seq._form_comp_info(3)
    p3_ref = evaluate_at_xq(
        seq.e3_dbc_T @ p3, comp_info_3, comp_shapes_3, quad_shape, 1)
    p3_phys = -p3_ref / seq.jacobian_j[:, None]
    mean_p3 = jnp.einsum("ik,i->", p3_phys, wJ) / V
    d3 = (p3_phys[:, 0] - mean_p3) - (pe - mean_pe)
    err_p3 = float((jnp.sum(d3**2 * wJ) / jnp.sum((pe - mean_pe)**2 * wJ)) ** 0.5)

    return err_p0, err_p3


class TestPressureAccuracy:
    def test_p0_error(self, pressure_errors):
        err_p0, _ = pressure_errors
        assert err_p0 < 0.002, f"p0 relative L2 error {err_p0:.2e} exceeds 0.002"

    def test_p3_error(self, pressure_errors):
        _, err_p3 = pressure_errors
        assert err_p3 < 0.025, f"p3 relative L2 error {err_p3:.2e} exceeds 0.025"

