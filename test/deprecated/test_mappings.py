# %%
# test_mappings.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence, compute_geometry_terms
from mrx.differential_forms import DiscreteFunction
from mrx.io import project_sampled_field
from mrx.mappings import (one_size_fits_all_map, cylinder_map, helical_map,
                          interpolate_map, polar_map, rotating_ellipse_map,
                          toroid_map)
from mrx.utils import jacobian_determinant

jax.config.update("jax_enable_x64", True)

# build some evaluation points
_x = jnp.linspace(0, 1, 11)[1:-1]
_xi, _xj, _xk = jnp.meshgrid(_x, _x, _x, indexing="ij")
pts = jnp.stack([_xi.ravel(), _xj.ravel(), _xk.ravel()], axis=1)


@pytest.mark.parametrize("map_factory", [
    lambda: one_size_fits_all_map(epsilon=0.5, kappa=1.2, alpha=0.1, R0=1.0),
    lambda: rotating_ellipse_map(eps=0.33, kappa=1.2, nfp=3, R0=1.0),
    lambda: toroid_map(epsilon=1/3, kappa=1.0, R0=1.0),
])
def test_map_origin_and_jacobian(map_factory):
    F = jax.jit(map_factory())
    # Check that the origin is mapped to (1, 0, 0)
    npt.assert_allclose(F(jnp.zeros(3)), jnp.array([1, 0, 0]))
    # Check positive Jacobian at all points
    J = jax.vmap(jacobian_determinant(F))(pts)
    npt.assert_allclose(J > 0, True)


@pytest.fixture
def mapping_seq():
    seq = DeRhamSequence(
        (6, 6, 6), (3, 3, 3), 6,
        ("clamped", "periodic", "periodic"),
        map=lambda x: x, polar=False
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    return seq


@pytest.mark.parametrize("map_factory,nfp", [
    (lambda: one_size_fits_all_map(epsilon=0.5, kappa=1.2, alpha=0.1, R0=1.0), 1),
    (lambda: rotating_ellipse_map(eps=0.33, kappa=1.2, nfp=3, R0=1.0), 3),
    (lambda: toroid_map(epsilon=1/3, kappa=1.0, R0=1.0), 1),
], ids=["cerfon", "rotating_ellipse", "toroid"])
def test_map_interpolation(map_factory, nfp, mapping_seq):
    F = map_factory()
    # Build a regular grid covering the full domain for the L2 projection
    _n = 20
    _r = jnp.linspace(0, 1, _n)
    _t = jnp.linspace(0, 1, _n)
    _z = jnp.linspace(0, 1, _n)
    _ri, _ti, _zi = jnp.meshgrid(_r, _t, _z, indexing="ij")
    grid_pts = jnp.stack([_ri.ravel(), _ti.ravel(), _zi.ravel()], axis=1)
    F_grid = jax.vmap(F)(grid_pts)
    R_grid = (F_grid[:, 0]**2 + F_grid[:, 1]**2)**0.5
    Z_grid = F_grid[:, 2]
    F_h = interpolate_map((_r, _t, _z),
                          R_grid.reshape(_n, _n, _n),
                          Z_grid.reshape(_n, _n, _n),
                          nfp, mapping_seq, flip_zeta=False)
    mapping_seq.set_map(F_h)
    mapping_seq.assemble_mass_matrix(0)
    # Evaluate at the standard test points and compare
    F_vals = jax.vmap(F)(pts)
    F_h_vals = jax.vmap(F_h)(pts)
    # Check that the interpolated map matches the original at the evaluation points
    npt.assert_allclose(F_h_vals, F_vals, atol=1e-2)
    # Check positive Jacobian at all points
    J = jax.vmap(jacobian_determinant(F_h))(pts)
    npt.assert_allclose(J > 0, True)


def test_spline_map_geometry_update(mapping_seq):
    F = toroid_map(epsilon=1/3, kappa=1.0, R0=1.0)
    _n = 16
    _r = jnp.linspace(0, 1, _n)
    _t = jnp.linspace(0, 1, _n)
    _z = jnp.linspace(0, 1, _n)
    _ri, _ti, _zi = jnp.meshgrid(_r, _t, _z, indexing="ij")
    grid_pts = jnp.stack([_ri.ravel(), _ti.ravel(), _zi.ravel()], axis=1)
    F_grid = jax.vmap(F)(grid_pts)

    coeffs = jnp.stack([
        project_sampled_field(
            (_r, _t, _z), F_grid[:, i], mapping_seq,
            k=0, dirichlet=False, reference_domain=True,
        )
        for i in range(3)
    ], axis=0)

    F_spline = mapping_seq.build_spline_map(coeffs)
    F_spline_vals = jax.vmap(F_spline)(pts)
    F_vals = jax.vmap(F)(pts)
    npt.assert_allclose(F_spline_vals, F_vals, atol=2e-2)

    mapping_seq.set_spline_map(coeffs)
    mapping_seq.assemble_mass_matrix(0)
    npt.assert_allclose(mapping_seq.jacobian_j > 0, True)

    jit_jacobian = jax.jit(
        lambda x: compute_geometry_terms(mapping_seq.build_spline_map(x), mapping_seq.quad.x)[2]
    )(coeffs)
    npt.assert_allclose(jit_jacobian, mapping_seq.jacobian_j)


# # %%
# arr = np.loadtxt("/scratch/tblickhan/mrx/data/lambda.txt")
# lam = arr.reshape(50, 50, 50)  # lam[i_rho, i_theta, i_zeta]
# lambda_vals = lam.ravel()
# rho = np.linspace(0, 1, 50)
# theta = np.linspace(0, 1, 50, endpoint=False)
# zeta = np.linspace(0, 1, 50, endpoint=False)

# xi, xj, xk = np.meshgrid(rho, theta, zeta, indexing="ij")
# pts = np.stack([xi.ravel(), xj.ravel(), xk.ravel()], axis=1)

# seq = mapping_seq()

# lambda_interpol = interpolate_scalar_function(
#     pts, lam.ravel(), seq, rcond=None)

# lambda_h = jax.jit(DiscreteFunction(
#     lambda_interpol["dof"], seq.basis_0, seq.e0))

# lambda_h_vals = jnp.squeeze(jax.lax.map(lambda_h, pts, batch_size=100_000))

# print("Resolution:", seq.n0)
# print("Max abs error:", jnp.max(jnp.abs(lambda_vals - lambda_h_vals)))
# print("Max abs error occurs at:", pts[jnp.argmax(
#     jnp.abs(lambda_vals - lambda_h_vals))])
# print("Mean abs error:", jnp.mean(jnp.abs(lambda_vals - lambda_h_vals)))
# print("standard deviation of error:", jnp.std(
#     jnp.abs(lambda_vals - lambda_h_vals)))
