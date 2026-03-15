"""Convergence study: pressure error vs resolution."""
import jax
import jax.numpy as jnp
from jax.numpy import cos, pi, sin

from mrx.derham_sequence import DeRhamSequence
from mrx.relaxation import compute_force
from mrx.utils import evaluate_at_xq

jax.config.update("jax_enable_x64", True)


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


def run(nr, nt, degree):
    seq = DeRhamSequence(
        (nr, nt, 1), (degree, degree, 0), degree + 2,
        ("clamped", "periodic", "constant"), F, polar=True)
    seq.evaluate_1d()
    seq.assemble_all_sparse()
    seq.compute_nullspaces()

    rhs = seq.p2_dbc(B0)
    B = seq.apply_inverse_mass_matrix(rhs, 2)
    B, _ = seq.apply_leray_projection(B, k=2)

    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    wJ = seq.jacobian_j * seq.quad.w
    V = jnp.sum(wJ)
    pe = jax.vmap(p_exact)(seq.quad.x)
    mean_pe = jnp.sum(pe * wJ) / V

    # 0-form pressure (k=1 Leray)
    J = seq.apply_weak_curl(B)
    H_dual = seq.apply_projection_matrix(B, 2, 1, True, dirichlet_out=False)
    H = seq.apply_inverse_mass_matrix(H_dual, 1, dirichlet=False)
    JxH_dual = seq.cross_product_projection(J, H, 1, 1, 1, False, True, False)
    JxH = seq.apply_inverse_mass_matrix(JxH_dual, 1, dirichlet=False)
    _, p0 = seq.apply_leray_projection(JxH, k=1)

    comp_info_0, comp_shapes_0 = seq._form_comp_info(0)
    p0_jk = -evaluate_at_xq(
        seq.e0_T @ p0, comp_info_0, comp_shapes_0, quad_shape, 1)
    mean_p0 = jnp.einsum("ik,i->", p0_jk, wJ) / V
    d0 = (p0_jk[:, 0] - mean_p0) - (pe - mean_pe)
    err_p0 = (jnp.sum(d0**2 * wJ) / jnp.sum((pe - mean_pe)**2 * wJ)) ** 0.5

    # 3-form pressure (k=2 Leray via compute_force)
    _, p3, _, _ = compute_force(B, seq)
    comp_info_3, comp_shapes_3 = seq._form_comp_info(3)
    p3_ref = evaluate_at_xq(
        seq.e3_dbc_T @ p3, comp_info_3, comp_shapes_3, quad_shape, 1)
    p3_phys = -p3_ref / seq.jacobian_j[:, None]
    mean_p3 = jnp.einsum("ik,i->", p3_phys, wJ) / V
    d3 = (p3_phys[:, 0] - mean_p3) - (pe - mean_pe)
    err_p3 = (jnp.sum(d3**2 * wJ) / jnp.sum((pe - mean_pe)**2 * wJ)) ** 0.5

    return float(err_p0), float(err_p3)


print(f"{'nr':>4s} {'nt':>4s} {'deg':>4s}  {'err_p0':>10s}  {'err_p3':>10s}")
print("-" * 42)

for nr, nt in [(6, 6), (8, 8), (10, 10)]:
    e0, e3 = run(nr, nt, 2)
    print(f"{nr:4d} {nt:4d} {2:4d}  {e0:10.6f}  {e3:10.6f}")
