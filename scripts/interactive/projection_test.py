# %%
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import (
    interpolate_map,
    cerfon_map,
    helical_map,
    rotating_ellipse_map,
    toroid_map,
    polar_map,
    cylinder_map
)
from mrx.utils import inv33, jacobian_determinant, det33, solve_singular_cg
from mrx.io import interpolate_scalar_function
import numpy as np
import h5py
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

nfs_path = "/scratch/tblickhan/mrx/data/gvec_nfp3_hegna_clebsch.h5"

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
        print("ERROR: File lacks clebsch_ingredients.") # Clebsch ingredients means things I extracted from GVEC run
    for k in ("Phi", "chi", "LA"):
        if f"clebsch/{k}" not in f:
            print(f"ERROR: File lacks clebsch/{k}. Export with --export-clebsch-ingredients.")
    
    pts = jnp.array(f["eval_points"][:])
    phi_vals = jnp.array(f["clebsch/Phi"][:]).ravel()
    chi_vals = jnp.array(f["clebsch/chi"][:]).ravel()
    lambda_vals = jnp.array(f["clebsch/LA"][:]).ravel()
    R_vals = jnp.array(f["R"][:]).ravel()
    Z_vals = jnp.array(f["Z"][:]).ravel()
    B_vals = jnp.array(f["B"][:]).reshape(-1, 3)
    p_vals = jnp.array(f["pressure"][:]).ravel()
    dlambda_dt_vals = jnp.array(f["clebsch/dLA_dt"][:]).ravel()
    dlambda_dz_vals = jnp.array(f["clebsch/dLA_dz"][:]).ravel()
    dphi_dr_vals = jnp.array(f["clebsch/dPhi_dr"][:]).ravel()
    dchi_dr_vals = jnp.array(f["clebsch/dchi_dr"][:]).ravel()
    grad_rho_vals = jnp.array(f["clebsch/grad_rho"][:]).reshape(-1, 3)
    grad_theta_vals = jnp.array(f["clebsch/grad_theta"][:]).reshape(-1, 3)
    grad_zeta_vals = jnp.array(f["clebsch/grad_zeta"][:]).reshape(-1, 3)

# Drop boundary points where rho == 1
mask = pts[:, 0] < 1.0
pts             = pts[mask]
phi_vals        = phi_vals[mask]
chi_vals        = chi_vals[mask]
lambda_vals     = lambda_vals[mask]
R_vals          = R_vals[mask]
Z_vals          = Z_vals[mask]
B_vals          = B_vals[mask]
p_vals          = p_vals[mask]
dlambda_dt_vals = dlambda_dt_vals[mask]
dlambda_dz_vals = dlambda_dz_vals[mask]
dphi_dr_vals    = dphi_dr_vals[mask]
dchi_dr_vals    = dchi_dr_vals[mask]
grad_rho_vals   = grad_rho_vals[mask]
grad_theta_vals = grad_theta_vals[mask]
grad_zeta_vals  = grad_zeta_vals[mask]

# %%
# Interpolate/get the map
map_seq = DeRhamSequence(
        (10, 10, 10), (3, 3, 3), 6,
        ("clamped", "periodic", "periodic"),
        map=lambda x: x, polar=False, dirichlet=False
    )
nfp = 3
map = interpolate_map(pts, R_vals, Z_vals, nfp=nfp, seq=map_seq)

# %%
lambda_interpol = interpolate_scalar_function(pts, lambda_vals, map_seq, rcond=None)
pressure_interpol = interpolate_scalar_function(pts, p_vals, map_seq, rcond=None)
phi_interpol = interpolate_scalar_function(pts, phi_vals, map_seq, rcond=None)
chi_interpol = interpolate_scalar_function(pts, chi_vals, map_seq, rcond=None)
# %%
lambda_h = jax.jit(DiscreteFunction(lambda_interpol["dof"], map_seq.basis_0, map_seq.e0))
p_h = jax.jit(DiscreteFunction(pressure_interpol["dof"], map_seq.basis_0, map_seq.e0))
phi_h = jax.jit(DiscreteFunction(phi_interpol["dof"], map_seq.basis_0, map_seq.e0))
chi_h = jax.jit(DiscreteFunction(chi_interpol["dof"], map_seq.basis_0, map_seq.e0))
@jax.jit
def dlambda_dx(x):
    Dmap = jax.jacfwd(map)(x)
    Dmap_inv = inv33(Dmap)
    dlambda_dxhat = jax.jacfwd(lambda_h)(x).T
    return jnp.squeeze(Dmap_inv @ dlambda_dxhat)
@jax.jit
def dlambda_dt(x):
    return jnp.squeeze(jax.jacfwd(lambda_h)(x))[1] / 2 / jnp.pi
@jax.jit
def dlambda_dz(x):
    return jnp.squeeze(jax.jacfwd(lambda_h)(x))[2] / 2 / jnp.pi * nfp
@jax.jit
def dphi_dr(x):
    return jnp.squeeze(jax.jacfwd(phi_h)(x))[0]
@jax.jit
def dchi_dr(x):
    return jnp.squeeze(jax.jacfwd(chi_h)(x))[0]
@jax.jit
def dx(x):
    Dmap = jax.jacfwd(map)(x)
    Dmap_inv = inv33(Dmap)
    grad_rho, grad_theta, grad_zeta = Dmap_inv[0, :], Dmap_inv[1, :], Dmap_inv[2, :]
    return grad_rho, grad_theta, grad_zeta
# %%
for f, f_vals, name in zip(
        (lambda_h, p_h, phi_h, chi_h, dlambda_dt, dlambda_dz, dphi_dr, dchi_dr), 
        (lambda_vals, p_vals, phi_vals, chi_vals, dlambda_dt_vals, dlambda_dz_vals, dphi_dr_vals, dchi_dr_vals), 
        ("lambda", "pressure", "phi", "chi", "dlambda_dt", "dlambda_dz", "dphi_dr", "dchi_dr")
    ):
    f_h_vals = jnp.squeeze(jax.lax.map(f, pts, batch_size=100_000))
    print(f"--- {name} ---")
    print("Resolution:", map_seq.n0)
    print("Max abs error:", jnp.max(jnp.abs(f_vals - f_h_vals)) / jnp.mean(jnp.abs(f_vals)))
    print("Max abs error occurs at:", pts[jnp.argmax(jnp.abs(f_vals - f_h_vals))])
    print("Mean abs error:", jnp.mean(jnp.abs(f_vals - f_h_vals)) / jnp.mean(jnp.abs(f_vals)))
    print("standard deviation of error:", jnp.std(jnp.abs(f_vals - f_h_vals)) / jnp.mean(jnp.abs(f_vals)))

# %%
# %%
@jax.jit
def B_clebsch(x):
    Dmap = jax.jacfwd(map)(x)
    J = det33(Dmap)
    grad_phi =    jnp.squeeze(jax.jacfwd(phi_h)(x))
    grad_chi =    jnp.squeeze(jax.jacfwd(chi_h)(x))
    grad_lambda = jnp.squeeze(jax.jacfwd(lambda_h)(x))
    return Dmap @ ( jnp.cross(grad_phi, (jnp.array([0,1,0]) * 2 * jnp.pi + grad_lambda)) 
                   + jnp.cross(jnp.array([0,0,1]) * 2 * jnp.pi / nfp, grad_chi) ) / J
    
# %%
B_clebsch_vals = jnp.squeeze(jax.lax.map(B_clebsch, pts, batch_size=100_000))
print(f"--- B (Clebsch) ---")
print("Resolution:", map_seq.n0)
print("Max abs error:", jnp.max(jnp.linalg.norm(B_vals - B_clebsch_vals, axis=1)) / jnp.mean(jnp.linalg.norm(B_vals, axis=1)))
print("Max abs error occurs at:", pts[jnp.argmax(jnp.linalg.norm(B_vals - B_clebsch_vals, axis=1))])
print("Mean abs error:", jnp.mean(jnp.abs(B_vals - B_clebsch_vals))  / jnp.mean(jnp.linalg.norm(B_vals, axis=1)))
print("standard deviation of error:", jnp.std(jnp.abs(B_vals - B_clebsch_vals)) / jnp.mean(jnp.linalg.norm(B_vals, axis=1)))
# %%
seq = DeRhamSequence(
        (10, 10, 10), (3, 3, 3), 6,
        ("clamped", "periodic", "periodic"),
        map=map, polar=True, dirichlet=False
    )
seq.evaluate_1d()
seq.assemble_m2_sparse()
# %%
rhs = seq.P2(B)
B_dof = solve_singular_cg(
    seq.apply_m2_sparse, 
    rhs,
    mass_matvec=seq.apply_m2_sparse,
    precond_matvec=seq.apply_m2_precond,
    tol=1e-12, 
    maxiter=10_000)[0]

# %%
seq.assemble_d2_sparse()
seq.assemble_m3_sparse()
# %%
B_norm = B_dof @ seq.apply_m2_sparse(B_dof)
B_norm
# %%
divB_dof = seq.apply_strong_div(B_dof)
div_B_norm = divB_dof @ seq.apply_d2_sparse(B_dof)
# %%
B_h = jax.jit(Pushforward(DiscreteFunction(B_dof, seq.basis_2, seq.e2), seq.map, 2))

# %%
B_h_vals = jnp.squeeze(jax.lax.map(B_h, pts, batch_size=100_000))
print(f"--- B (FEM) ---")
print("Resolution:", map_seq.n0)
print("Max abs error:", jnp.max(jnp.linalg.norm(B_vals - B_h_vals, axis=1)) / jnp.mean(jnp.linalg.norm(B_vals, axis=1)))
print("Max abs error occurs at:", pts[jnp.argmax(jnp.linalg.norm(B_vals - B_h_vals, axis=1))])
print("Mean abs error:", jnp.mean(jnp.abs(B_vals - B_h_vals))  / jnp.mean(jnp.linalg.norm(B_vals, axis=1)))
print("standard deviation of error:", jnp.std(jnp.abs(B_vals - B_h_vals)) / jnp.mean(jnp.linalg.norm(B_vals, axis=1)))