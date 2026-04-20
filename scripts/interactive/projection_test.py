# %%
from time import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from jax.scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.io import interpolate_scalar_function
from mrx.mappings import (cerfon_map, cylinder_map, helical_map,
                          interpolate_map, polar_map, rotating_ellipse_map,
                          toroid_map)
from mrx.solvers import solve_singular_cg
from mrx.utils import (det33, evaluate_at_xq, integrate_against, inv33,
                       jacobian_determinant)

jax.config.update("jax_enable_x64", True)

nfs_path = "/scratch/tblickhan/mrx/data/gvec_nfp3_hegna_80cubed_clebsch.h5"
nfs_path = '/Users/tobias/Coding/mrx/data/gvec_nfp3_hegna_80cubed_clebsch.h5'


def inspect_h5_item(name, obj):
    """Callback function to print the name and type of each item."""
    if isinstance(obj, h5py.Group):
        print(f"Group:   {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")


with h5py.File(nfs_path, 'r') as f:
    print("--- HDF5 File Structure ---")
    # This visits every node in the file and applies our function
    f.visititems(inspect_h5_item)

with h5py.File(nfs_path, "r") as f:
    if not f.attrs.get("clebsch_ingredients", False):
        # Clebsch ingredients means things I extracted from GVEC run
        print("ERROR: File lacks clebsch_ingredients.")
    for k in ("Phi", "chi", "LA"):
        if f"clebsch/{k}" not in f:
            print(
                f"ERROR: File lacks clebsch/{k}. Export with --export-clebsch-ingredients.")

    pts = jnp.array(f["eval_points"][:])
    phi_vals = jnp.array(f["clebsch/Phi"][:]).ravel()
    chi_vals = jnp.array(f["clebsch/chi"][:]).ravel()
    lambda_vals = jnp.array(f["clebsch/LA"][:]).ravel()
    R_vals = jnp.array(f["R"][:]).ravel()
    Z_vals = jnp.array(f["Z"][:]).ravel()
    B_vals = jnp.array(f["B"][:]).reshape(-1, 3)
    J_vals = jnp.array(f["J"][:]).reshape(-1, 3)
    p_vals = jnp.array(f["pressure"][:]).ravel()
    dlambda_dt_vals = jnp.array(f["clebsch/dLA_dt"][:]).ravel()
    dlambda_dz_vals = jnp.array(f["clebsch/dLA_dz"][:]).ravel()
    dphi_dr_vals = jnp.array(f["clebsch/dPhi_dr"][:]).ravel()
    dchi_dr_vals = jnp.array(f["clebsch/dchi_dr"][:]).ravel()
    grad_rho_vals = jnp.array(f["clebsch/grad_rho"][:]).reshape(-1, 3)
    grad_theta_vals = jnp.array(f["clebsch/grad_theta"][:]).reshape(-1, 3)
    grad_zeta_vals = jnp.array(f["clebsch/grad_zeta"][:]).reshape(-1, 3)

# Extract 1-D grid axes (80 per direction)
rho = jnp.array(np.unique(np.asarray(pts[:, 0])))
theta = jnp.array(np.unique(np.asarray(pts[:, 1])))
zeta = jnp.array(np.unique(np.asarray(pts[:, 2])))
n_rho, n_theta, n_zeta = len(rho), len(theta), len(zeta)
print(f"Grid: {n_rho} × {n_theta} × {n_zeta}")
# %%
# ---------------------------------------------------------------
# Interpolate the coordinate map
# ---------------------------------------------------------------
NS = [8] * 3
PS = [3] * 3
QUAD_ORDER = 2 * PS[0]
NFP = 3

print("Building map interpolation sequence...")
map_seq = DeRhamSequence(
    NS, PS, QUAD_ORDER,
    ("clamped", "periodic", "periodic"),
    lambda x: x, polar=True, tol=1e-9
)
map_seq.evaluate_1d()
map_seq.assemble_mass_matrix(0)

# %%
print("Interpolating coordinate map (r, θ, ζ) -> (x, y, z)...")
R_grid = R_vals.reshape(n_rho, n_theta, n_zeta)
Z_grid = Z_vals.reshape(n_rho, n_theta, n_zeta)
map_func = interpolate_map((rho, theta, zeta), R_grid, Z_grid, NFP, map_seq)
# map_func = jax.jit(map_func)
# # Warm-up JIT
# _ = map_func(jnp.array([0.5, 0.5, 0.5]))

# %%
seq = DeRhamSequence(
    NS, PS, QUAD_ORDER,
    ("clamped", "periodic", "periodic"),
    map=map_func, polar=True, r_scale=1.0
)
seq.evaluate_1d()
seq.assemble_all_sparse()
# %%
seq.compute_nullspaces()

# lsq_weights = jax.vmap(jacobian_determinant(map))(pts)
# lambda_interpol = interpolate_scalar_function(pts, lambda_vals, seq, lsq_weights, rcond=None)
# pressure_interpol = interpolate_scalar_function(pts, p_vals, map_seq, lsq_weights, rcond=None)
# phi_interpol = interpolate_scalar_function(pts, phi_vals, map_seq, lsq_weights, rcond=None)
# chi_interpol = interpolate_scalar_function(pts, chi_vals, map_seq, lsq_weights, rcond=None)

# p_h = jax.jit(DiscreteFunction(pressure_interpol["dof"], map_seq.basis_0, map_seq.e0))
# phi_h = jax.jit(DiscreteFunction(phi_interpol["dof"], map_seq.basis_0, map_seq.e0))
# chi_h = jax.jit(DiscreteFunction(chi_interpol["dof"], map_seq.basis_0, map_seq.e0))
# lambda_h = jax.jit(DiscreteFunction(lambda_interpol["dof"], map_seq.basis_0, map_seq.e0))

# %%
m1_dense = seq.e1_dbc.todense() @ seq.m1_sp.todense() @ seq.e1_dbc_T.todense()
m2_dense = seq.e2_dbc.todense() @ seq.m2_sp.todense() @ seq.e2_dbc_T.todense()
m3_dense = seq.e3_dbc.todense() @ seq.m3_sp.todense() @ seq.e3_dbc_T.todense()
div_dense = seq.e3_dbc.todense() @ seq.d2_sp.todense() @ seq.e2_dbc_T.todense()
curl_dense = seq.e2_dbc.todense() @ seq.d1_sp.todense() @ seq.e1_dbc_T.todense()


blockmatrix = jnp.block([
    [m2_dense, div_dense.T, curl_dense],
    [div_dense, 1e-6 * jnp.eye(seq.n3_dbc, seq.n3_dbc),
     jnp.zeros((seq.n3_dbc, seq.n1_dbc))],
    [curl_dense.T, jnp.zeros((seq.n1_dbc, seq.n3_dbc)),
     1e-6 * jnp.eye(seq.n1_dbc, seq.n1_dbc)],
])


# %%

def interpolate_vector_field(x_q, x_axis, y_axis, z_axis, v_values):
    """
    Interpolates a vector field at logical query points x_q.

    Args:
        x_q: Array of query points in the logical domain, shape (m, 3).
        x_axis, y_axis, z_axis: 1D arrays defining the logical Cartesian grid.
        v_values: Vector field values at the logical grid points, shape (nx ny nz, 3).
            Note: nx, ny, nz must equal len(x_axis), len(y_axis), len(z_axis).

    Returns:
        Interpolated vectors at x_q, shape (m, 3).
    """
    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    v_values = v_values.reshape(nx, ny, nz, 3)  # Reshape to (nx, ny, nz, 3)

    def interpolate_scalar(V_scalar):
        # V_scalar shape is purely (nx, ny, nz)
        interpolator = RegularGridInterpolator(
            points=(x_axis, y_axis, z_axis),
            values=V_scalar,
            method='linear'
        )
        return interpolator(x_q)

    return jax.vmap(interpolate_scalar, in_axes=-1, out_axes=-1)(v_values)


def interpolate_scalar_field(x_q, x_axis, y_axis, z_axis, v_values):
    """
    Interpolates a scalar field at logical query points x_q.

    Args:
        x_q: Array of query points in the logical domain, shape (m, 3).
        x_axis, y_axis, z_axis: 1D arrays defining the logical Cartesian grid.
        v_values: Scalar field values at the logical grid points, shape (nx ny nz,).
            Note: nx, ny, nz must equal len(x_axis), len(y_axis), len(z_axis).

    Returns:
        Interpolated scalars at x_q, shape (m,).
    """
    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    v_values = v_values.reshape(nx, ny, nz)

    interpolator = RegularGridInterpolator(
        points=(x_axis, y_axis, z_axis),
        values=v_values,
        method='linear'
    )

    return interpolator(x_q)


# %%
n_gvec = 80
_r = pts.reshape(n_gvec, n_gvec, n_gvec, 3)[:, 0, 0, 0]
_t = pts.reshape(n_gvec, n_gvec, n_gvec, 3)[0, :, 0, 1]
_z = pts.reshape(n_gvec, n_gvec, n_gvec, 3)[0, 0, :, 2]

# Replace crazy-large J values with nearest-neighbor approximation

J_vals_clean = np.array(J_vals).reshape(n_gvec, n_gvec, n_gvec, 3)
bad_mask = np.any(np.abs(J_vals_clean) > 1e30, axis=-1)  # shape (n, n, n)
print(
    f"Replacing {bad_mask.sum()} / {bad_mask.size} bad J values with nearest-neighbor")
if bad_mask.any():
    _, nearest_idx = distance_transform_edt(
        bad_mask, return_distances=True, return_indices=True)
    for comp in range(3):
        good_vals = J_vals_clean[:, :, :, comp]
        J_vals_clean[:, :, :, comp] = good_vals[tuple(nearest_idx)]
J_vals_clean = jnp.array(J_vals_clean.reshape(-1, 3))

B_at_xq = interpolate_vector_field(seq.quad.x, _r, _t, _z, B_vals)
J_at_xq = interpolate_vector_field(seq.quad.x, _r, _t, _z, J_vals_clean)
lambda_at_xq = interpolate_scalar_field(seq.quad.x, _r, _t, _z, lambda_vals)
phi_at_xq = interpolate_scalar_field(seq.quad.x, _r, _t, _z, phi_vals)
chi_at_xq = interpolate_scalar_field(seq.quad.x, _r, _t, _z, chi_vals)
p_at_xq = interpolate_scalar_field(seq.quad.x, _r, _t, _z, p_vals)
# %%
DF = jax.jacfwd(seq.map)
# shape (n_q, 3, 3)
DF_j = jax.lax.map(DF, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
DF_inv_j = jax.vmap(inv33)(DF_j)
J_j = jax.vmap(jacobian_determinant(seq.map))(seq.quad.x)  # shape (n_q,)
w_jk = jnp.einsum('jlk,jl,j->jk', DF_j, B_at_xq, seq.quad.w)  # shape (n_q, 3)
comp_info_2, comp_shapes_2 = seq._form_comp_info(2)
quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
B_proj = seq.e2_dbc @ integrate_against(w_jk,
                                        comp_info_2, comp_shapes_2, quad_shape)

# %%
mu_0 = 1.25663706127e-6
comp_info_1, comp_shapes_1 = seq._form_comp_info(1)
w_jk = jnp.einsum('jkl,jl,j,j->jk', DF_inv_j, mu_0 * J_at_xq,
                  J_j, seq.quad.w)  # shape (n_q, 3)
J_proj = seq.e1_dbc @ integrate_against(w_jk,
                                        comp_info_1, comp_shapes_1, quad_shape)


# %%
rhs = B_proj + curl_dense @ jnp.linalg.solve(m1_dense, J_proj)

L = m2_dense + curl_dense @ jnp.linalg.solve(
    m1_dense, curl_dense.T) + div_dense.T @ jnp.linalg.solve(m3_dense, div_dense)


B_dof = jnp.linalg.solve(L, rhs)

# %%
plt.plot(B_dof)


# %%
divB_dof = jnp.linalg.solve(m3_dense, div_dense @ B_dof)

print("Divergence of projected B:", (divB_dof @ m3_dense @ divB_dof)**0.5)


# %%
J_hat_weak_dbc = seq.apply_weak_curl(
    B_dof, dirichlet_in=True, dirichlet_out=True)
J_h_weak_dbc = jax.jit(DiscreteFunction(
    J_hat_weak_dbc, seq.basis_1, seq.e1_dbc))
J_h_weak_xyz_dbc = jax.jit(Pushforward(J_h_weak_dbc, seq.map, 1))

__r = jnp.linspace(0.001, 0.999, 1000)
eval_pts = jnp.vstack([__r, jnp.zeros_like(__r), jnp.zeros_like(__r)]).T

J_weak_dbc_xyz_at_r = jax.vmap(J_h_weak_xyz_dbc)(eval_pts)
# %%
plt.plot(__r, J_weak_dbc_xyz_at_r[:, 0], label="J_x (weak, Jxn = 0)", ls=":")
plt.plot(__r, J_weak_dbc_xyz_at_r[:, 1], label="J_y (weak, Jxn = 0)", ls=":")
plt.plot(__r, J_weak_dbc_xyz_at_r[:, 2], label="J_z (weak, Jxn = 0)", ls=":")
plt.xlabel("r")
plt.ylabel("J")
plt.legend()
plt.show()

# %%
B_h_xyz = jax.jit(Pushforward(DiscreteFunction(
    B_dof, seq.basis_2, seq.e2_dbc), seq.map, 2))
B_h = jax.jit(DiscreteFunction(B_dof, seq.basis_2, seq.e2_dbc))
B_xyz_at_rad = jax.vmap(B_h_xyz)(eval_pts)
# %%
plt.plot(__r, B_xyz_at_rad[:, 0], label="B_x")
plt.plot(__r, B_xyz_at_rad[:, 1], label="B_y")
plt.plot(__r, B_xyz_at_rad[:, 2], label="B_z")
# plt.plot(__r, jnp.linalg.norm(B_xyz_at_rad, axis=1), label="|B|")
plt.xlabel("r")
plt.ylabel("B")
plt.legend()
plt.show()
# %%

# %%

# %%

# %%
p_proj = seq.e0 @ integrate_against(seq.eval_basis_0_ijk,
                                    (p_at_xq * J_j * seq.quad.w)[:, None], seq.basis_0.n)
chi_proj = seq.e0 @ integrate_against(seq.eval_basis_0_ijk,
                                      (chi_at_xq * J_j * seq.quad.w)[:, None], seq.basis_0.n)
phi_proj = seq.e0 @ integrate_against(seq.eval_basis_0_ijk,
                                      (phi_at_xq * J_j * seq.quad.w)[:, None], seq.basis_0.n)
lambda_proj = seq.e0 @ integrate_against(
    seq.eval_basis_0_ijk, (lambda_at_xq * J_j * seq.quad.w)[:, None], seq.basis_0.n)
# %%
B_dof_direct = solve_singular_cg(
    seq.apply_m2_sparse,
    B_proj,
    mass_matvec=lambda x: seq.apply_m2_sparse(x, True),
    precond_matvec=lambda x: seq.apply_m2_precond(x, True),
    tol=seq.tol,
    maxiter=seq.maxiter)[0]
p_dof, chi_dof, phi_dof, lambda_dof = [solve_singular_cg(
    lambda x: seq.apply_m0_sparse(x, False),
    x,
    mass_matvec=lambda x: seq.apply_m0_sparse(x, False),
    precond_matvec=lambda x: seq.apply_m0_precond(x, False),
    tol=seq.tol,
    maxiter=seq.maxiter)[0] for x in (p_proj, chi_proj, phi_proj, lambda_proj)]
# %%
valid_indices = (pts[:, 0] < 1.0) & (pts[:, 0] > 0)
# %%
# p_h = jax.jit(DiscreteFunction(p_dof, seq.basis_0, seq.e0))
# p_h_vals = jax.lax.map(p_h, pts[valid_indices][::10], batch_size=mrx.MAP_BATCH_SIZE_INNER)
# p_vals_valid = p_vals[valid_indices][::10, None]
# print(f"--- p (FEM) ---")
# print(f"p={p}, n={n}")
# print("Max abs error:",
#       jnp.max(jnp.linalg.norm(p_vals_valid - p_h_vals, axis=1)) / jnp.mean(jnp.linalg.norm(p_vals_valid, axis=1)))
# print("Max abs error occurs at:",
#       pts[jnp.argmax(jnp.linalg.norm(p_vals_valid - p_h_vals, axis=1))])
# print("Mean abs error:",
#       jnp.mean(jnp.abs(p_vals_valid - p_h_vals))  / jnp.mean(jnp.linalg.norm(p_vals_valid, axis=1)))
# print("standard deviation of error:",
#       jnp.std(jnp.abs(p_vals_valid - p_h_vals)) / jnp.mean(jnp.linalg.norm(p_vals_valid, axis=1)))

# # %%
# lambda_h = jax.jit(DiscreteFunction(lambda_dof, seq.basis_0, seq.e0))
# lambda_h_vals = jax.lax.map(lambda_h, pts[valid_indices][::10], batch_size=mrx.MAP_BATCH_SIZE_INNER)
# lambda_vals_valid = lambda_vals[valid_indices][::10, None]
# print(f"--- lambda (FEM) ---")
# print(f"p = {p}, n={n}")
# print("Max abs error:",
#       jnp.max(jnp.linalg.norm(lambda_vals_valid - lambda_h_vals, axis=1)) / jnp.mean(jnp.linalg.norm(lambda_vals_valid, axis=1)))
# print("Max abs error occurs at:",
#       pts[jnp.argmax(jnp.linalg.norm(lambda_vals_valid - lambda_h_vals, axis=1))])
# print("Mean abs error:",
#       jnp.mean(jnp.abs(lambda_vals_valid - lambda_h_vals))  / jnp.mean(jnp.linalg.norm(lambda_vals_valid, axis=1)))
# print("standard deviation of error:",
#       jnp.std(jnp.abs(lambda_vals_valid - lambda_h_vals)) / jnp.mean(jnp.linalg.norm(lambda_vals_valid, axis=1)))

# # %%
# phi_h = jax.jit(DiscreteFunction(phi_dof, seq.basis_0, seq.e0))
# phi_h_vals = jax.lax.map(phi_h, pts[valid_indices][::10], batch_size=mrx.MAP_BATCH_SIZE_INNER)
# phi_vals_valid = phi_vals[valid_indices][::10, None]
# print(f"--- phi (FEM) ---")
# print(f"p = {p}, n={n}")
# print("Max abs error:",
#       jnp.max(jnp.linalg.norm(phi_vals_valid - phi_h_vals, axis=1)) / jnp.mean(jnp.linalg.norm(phi_vals_valid, axis=1)))
# print("Max abs error occurs at:",
#       pts[jnp.argmax(jnp.linalg.norm(phi_vals_valid - phi_h_vals, axis=1))])
# print("Mean abs error:",
#       jnp.mean(jnp.abs(phi_vals_valid - phi_h_vals))  / jnp.mean(jnp.linalg.norm(phi_vals_valid, axis=1)))
# print("standard deviation of error:",
#       jnp.std(jnp.abs(phi_vals_valid - phi_h_vals)) / jnp.mean(jnp.linalg.norm(phi_vals_valid, axis=1)))

# # %%
# chi_h = jax.jit(DiscreteFunction(chi_dof, seq.basis_0, seq.e0))
# chi_h_vals = jax.lax.map(chi_h, pts[valid_indices][::10], batch_size=mrx.MAP_BATCH_SIZE_INNER)
# chi_vals_valid = chi_vals[valid_indices][::10, None]
# print(f"--- chi (FEM) ---")
# print(f"p = {p}, n={n}")
# print("Max abs error:",
#       jnp.max(jnp.linalg.norm(chi_vals_valid - chi_h_vals, axis=1)) / jnp.mean(jnp.linalg.norm(chi_vals_valid, axis=1)))
# print("Max abs error occurs at:",
#       pts[jnp.argmax(jnp.linalg.norm(chi_vals_valid - chi_h_vals, axis=1))])
# print("Mean abs error:",
#       jnp.mean(jnp.abs(chi_vals_valid - chi_h_vals))  / jnp.mean(jnp.linalg.norm(chi_vals_valid, axis=1)))
# print("standard deviation of error:",
#       jnp.std(jnp.abs(chi_vals_valid - chi_h_vals)) / jnp.mean(jnp.linalg.norm(chi_vals_valid, axis=1)))

# %%


def IC_projection(phi_dof, chi_dof, lambda_dof, k=2):
    # evaluate (logical) grad (phi, chi, lambda) at quad. point j. shape: n_q x 3
    grad_phi = evaluate_at_xq(seq.eval_d_basis_0_ijk,
                              seq.e0.T @ phi_dof, seq.quad.n, 3)
    grad_chi = evaluate_at_xq(seq.eval_d_basis_0_ijk,
                              seq.e0.T @ chi_dof, seq.quad.n, 3)
    grad_lambda = evaluate_at_xq(
        seq.eval_d_basis_0_ijk, seq.e0.T @ lambda_dof, seq.quad.n, 3)
    grad_theta = jnp.array([0, 1, 0])[None, :] * 2 * jnp.pi
    grad_zeta = jnp.array([0, 0, 1])[None, :] * 2 * jnp.pi / nfp

    B_jk = jnp.cross(grad_phi, grad_theta + grad_lambda) - \
        jnp.cross(grad_chi, grad_zeta)
    if k == 2:
        GB_jk = jnp.einsum('jkl,jk,j,j->jl', seq.metric_jkl,
                           B_jk, seq.quad.w, 1 / seq.jacobian_j)
        return seq.e2_dbc @ integrate_against(seq.eval_basis_2_ijk, GB_jk, seq.basis_2.n)
    elif k == 1:
        GB_jk = jnp.einsum('jk,j->jk', B_jk, seq.quad.w)
        return seq.e1 @ integrate_against(seq.eval_basis_1_ijk, GB_jk, seq.basis_1.n)


# %%
B_proj = IC_projection(phi_dof, chi_dof, lambda_dof, k=2)
H_proj = IC_projection(phi_dof, chi_dof, lambda_dof, k=1)
# %%


@jax.jit
def apply_B_projection(B_proj, alpha=0.0):

    B_guess = solve_singular_cg(
        lambda x: seq.apply_m2_sparse(x, True),
        B_proj,
        precond_matvec=lambda x: seq.apply_m2_precond(x, True),
        tol=seq.tol,
        maxiter=seq.maxiter
    )[0]

    def apply_A(x):
        return seq.apply_m2_sparse(x, True) - alpha * seq.apply_dd2_sparse(x, True)

    def apply_Ainv(x, x0=None):
        return solve_singular_cg(
            apply_A,
            x,
            mass_matvec=lambda x: seq.apply_m2_sparse(x, True),
            precond_matvec=lambda x: seq.apply_m2_precond(x, True),
            tol=seq.tol,
            x0=x0,
            maxiter=seq.maxiter
        )[0]

    def apply_D_Ainv_Dt(x, x0=None):
        return seq.apply_d2_sparse(apply_Ainv(seq.apply_d2t_sparse(x, True, True), x0=x0), True, True)

    A_inv_B_proj = apply_Ainv(B_proj, x0=B_guess)

    q_dof = solve_singular_cg(
        A_matvec=apply_D_Ainv_Dt,
        b=seq.apply_d2_sparse(A_inv_B_proj, True, True),
        mass_matvec=lambda x: seq.apply_m3_sparse(x, True),
        vs=seq.null_3_dbc,
        precond_matvec=lambda x: seq.apply_dd3_precond(x, True),
        tol=seq.tol,
        maxiter=seq.maxiter
    )[0]

    B_dof = A_inv_B_proj - apply_Ainv(seq.apply_d2t_sparse(q_dof, True, True))

    return B_dof, q_dof, B_guess
# %%
# B_dof, q_dof, B_guess = apply_B_projection(B_proj, alpha=0.0)


# %%
B_guess = solve_singular_cg(
    lambda x: seq.apply_m2_sparse(x, True),
    B_proj,
    precond_matvec=lambda x: seq.apply_m2_precond(x, True),
    tol=seq.tol,
    maxiter=seq.maxiter
)[0]

H_guess = solve_singular_cg(
    lambda x: seq.apply_m1_sparse(x, False),
    H_proj,
    precond_matvec=lambda x: seq.apply_m1_precond(x, False),
    tol=seq.tol,
    maxiter=seq.maxiter
)[0]

# %%
B_dof = seq.apply_leray_projection(B_guess, k=2)[0]
H_dof = seq.apply_leray_projection(H_guess, k=1)[0]
# %%

# %%
# J_hat_strong_dbc = seq.apply_strong_curl(
#     H_dof, dirichlet_in=False, dirichlet_out=True)
# J_h_strong_dbc = jax.jit(DiscreteFunction(
#     J_hat_strong_dbc, seq.basis_2, seq.e2_dbc))
# J_h_strong_xyz_dbc = jax.jit(Pushforward(J_h_strong_dbc, seq.map, 2))

# J_hat_strong = seq.apply_strong_curl(
#     H_dof, dirichlet_in=False, dirichlet_out=False)
# J_h_strong = jax.jit(DiscreteFunction(J_hat_strong_dbc, seq.basis_2, seq.e2))
# J_h_strong_xyz = jax.jit(Pushforward(J_h_strong_dbc, seq.map, 2))

J_hat_weak_dbc = seq.apply_weak_curl(
    B_dof, dirichlet_in=True, dirichlet_out=True)
J_h_weak_dbc = jax.jit(DiscreteFunction(
    J_hat_weak_dbc, seq.basis_1, seq.e1_dbc))
J_h_weak_xyz_dbc = jax.jit(Pushforward(J_h_weak_dbc, seq.map, 1))

__r = jnp.linspace(0.001, 0.999, 1000)
eval_pts = jnp.vstack([__r, jnp.zeros_like(__r), jnp.zeros_like(__r)]).T

# J_strong_dbc_xyz_at_r = jax.vmap(J_h_strong_xyz_dbc)(eval_pts)
# J_strong_xyz_at_r = jax.vmap(J_h_strong_xyz)(eval_pts)
J_weak_dbc_xyz_at_r = jax.vmap(J_h_weak_xyz_dbc)(eval_pts)
# %%
# plt.plot(__r, J_strong_dbc_xyz_at_r[:, 0],
#          label="J_x (strong, J.n = 0)", ls="--")
# plt.plot(__r, J_strong_dbc_xyz_at_r[:, 1],
#          label="J_y (strong, J.n = 0)", ls="--")
# plt.plot(__r, J_strong_dbc_xyz_at_r[:, 2],
#          label="J_z (strong, J.n = 0)", ls="--")
# plt.plot(__r, J_strong_xyz_at_r[:, 0], label="J_x (strong)", ls="-")
# plt.plot(__r, J_strong_xyz_at_r[:, 1], label="J_y (strong)", ls="-")
# plt.plot(__r, J_strong_xyz_at_r[:, 2], label="J_z (strong)", ls="-")
plt.plot(__r, J_weak_dbc_xyz_at_r[:, 0], label="J_x (weak, Jxn = 0)", ls=":")
plt.plot(__r, J_weak_dbc_xyz_at_r[:, 1], label="J_y (weak, Jxn = 0)", ls=":")
plt.plot(__r, J_weak_dbc_xyz_at_r[:, 2], label="J_z (weak, Jxn = 0)", ls=":")
# plt.plot(__r, J_weak_xyz_at_r[:, 0], label="J_x (weak)", ls="-.")
# plt.plot(__r, J_weak_xyz_at_r[:, 1], label="J_y (weak)", ls="-.")
# plt.plot(__r, J_weak_xyz_at_r[:, 2], label="J_z (weak)", ls="-.")
plt.xlabel("r")
plt.ylabel("J")
plt.legend()
plt.show()

# %%
B_h_xyz = jax.jit(Pushforward(DiscreteFunction(
    B_dof, seq.basis_2, seq.e2_dbc), seq.map, 2))
B_h = jax.jit(DiscreteFunction(B_dof, seq.basis_2, seq.e2_dbc))

# B_h_xyz = jax.jit(Pushforward(DiscreteFunction(
#     H_dof, seq.basis_1, seq.e1), seq.map, 1))
# B_h = jax.jit(DiscreteFunction(H_dof, seq.basis_1, seq.e1))

B_xyz_at_rad = jax.vmap(B_h_xyz)(eval_pts)
# %%
plt.plot(__r, B_xyz_at_rad[:, 0], label="B_x")
plt.plot(__r, B_xyz_at_rad[:, 1], label="B_y")
plt.plot(__r, B_xyz_at_rad[:, 2], label="B_z")
# plt.plot(__r, jnp.linalg.norm(B_xyz_at_rad, axis=1), label="|B|")
plt.xlabel("r")
plt.ylabel("B")
plt.legend()
plt.show()

# %%
# # %%
# # Dissipate some B away:
# alpha = 1e-5

# def apply_A(x):
#     return seq.apply_m2_sparse(x, True) + alpha * seq.apply_dd2_sparse(x, True)

# def apply_Ainv(x, x0=None):
#     return solve_singular_cg(
#         apply_A,
#         x,
#         mass_matvec=lambda x: seq.apply_m2_sparse(x, True),
#         precond_matvec=lambda x: seq.apply_m2_precond(x, True),
#         tol=seq.tol,
#         x0=x0,
#         maxiter=seq.maxiter
#     )[0]


# for _ in range(10):
#     B_dof = apply_Ainv(seq.apply_m2_sparse(B_dof, True), x0=B_dof)
# # %%
# # %%
