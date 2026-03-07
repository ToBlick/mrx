# %%
# test_mappings.py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import (
    interpolate_map,
    cerfon_map,
    helical_map,
    rotating_ellipse_map,
    toroid_map,
    polar_map,
    cylinder_map
)
from mrx.utils import jacobian_determinant
from mrx.io import interpolate_scalar_function
import numpy as np

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# build some evaluation points
_x = jnp.linspace(0, 1, 11)[1:-1]
_xi, _xj, _xk = jnp.meshgrid(_x, _x, _x, indexing="ij")
pts = jnp.stack([_xi.ravel(), _xj.ravel(), _xk.ravel()], axis=1)


@pytest.mark.parametrize("map_factory", [
    lambda: cerfon_map(epsilon=0.5, kappa=1.2, alpha=0.1, R0=1.0),
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


@pytest.fixture(scope="module")
def mapping_seq():
    return DeRhamSequence(
        (20, 20, 20), (3, 3, 3), 6,
        ("clamped", "periodic", "periodic"),
        map=lambda x: x, polar=False, dirichlet=False
    )


@pytest.mark.parametrize("map_factory,nfp", [
    (lambda: cerfon_map(epsilon=0.5, kappa=1.2, alpha=0.1, R0=1.0), 1),
    (lambda: rotating_ellipse_map(eps=0.33, kappa=1.2, nfp=3, R0=1.0), 3),
    (lambda: toroid_map(epsilon=1/3, kappa=1.0, R0=1.0), 1),
], ids=["cerfon", "rotating_ellipse", "toroid"])
def test_map_interpolation(map_factory, nfp, mapping_seq):
    F = map_factory()
    F_vals = jax.vmap(F)(pts)
    R_vals = (F_vals[:, 0]**2 + F_vals[:, 1]**2)**0.5
    Z_vals = F_vals[:, 2]
    F_h = interpolate_map(pts, R_vals, Z_vals, nfp, mapping_seq,
                          flip_zeta=False, rcond=None)
    F_h_vals = jax.vmap(F_h)(pts)
    # Check that the interpolated map matches the original at the evaluation points
    npt.assert_allclose(jnp.max(jnp.abs(F_vals - F_h_vals)) < 1e-3, True)
    # Check positive Jacobian at all points 
    J = jax.vmap(jacobian_determinant(F_h))(pts)
    npt.assert_allclose(J > 0, True)
    
    
# %%
arr = np.loadtxt("/scratch/tblickhan/mrx/data/lambda.txt")
lam = arr.reshape(50, 50, 50)  # lam[i_rho, i_theta, i_zeta]
lambda_vals = lam.ravel()
rho   = np.linspace(0, 1, 50)
theta = np.linspace(0, 1, 50, endpoint=False)
zeta  = np.linspace(0, 1, 50, endpoint=False)

xi, xj, xk = np.meshgrid(rho, theta, zeta, indexing="ij")
pts = np.stack([xi.ravel(), xj.ravel(), xk.ravel()], axis=1)

seq = mapping_seq()

lambda_interpol = interpolate_scalar_function(pts, lam.ravel(), seq, rcond=None)

lambda_h = jax.jit(DiscreteFunction(lambda_interpol["dof"], seq.basis_0, seq.e0))

lambda_h_vals = jnp.squeeze(jax.lax.map(lambda_h, pts, batch_size=100_000))

print("Resolution:", seq.n0)
print("Max abs error:", jnp.max(jnp.abs(lambda_vals - lambda_h_vals)))
print("Max abs error occurs at:", pts[jnp.argmax(jnp.abs(lambda_vals - lambda_h_vals))])
print("Mean abs error:", jnp.mean(jnp.abs(lambda_vals - lambda_h_vals)))
print("standard deviation of error:", jnp.std(jnp.abs(lambda_vals - lambda_h_vals)))