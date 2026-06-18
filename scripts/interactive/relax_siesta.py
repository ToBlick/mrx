# %%
"""
Relaxation of the Siesta nfp=3 stellarator equilibrium.

Loads the GVEC equilibrium data from an HDF5 file (80³ regular grid in
Clebsch coordinates), interpolates the coordinate map R(ρ,θ,ζ) and
Z(ρ,θ,ζ) onto B-splines, builds a de Rham sequence on the resulting
stellarator geometry, L²-projects the sampled B-field via
``project_sampled_field``, and runs the MRX relaxation loop.

Usage
----- 
    python scripts/relax_siesta.py
"""
import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

import mrx
from mrx.assembly import (assemble_dense_laplacian,
                          assemble_dense_mass_matrix)
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import interpolate_map
from mrx.plotting import get_1d_grids
from mrx.preconditioners import get_mass_jacobi_diaginv
from mrx.relaxation import (DescentMethod, IntegrationScheme, TimeStepChoice,
                            TimeStepper, compute_force,
                            relaxation_loop)
from mrx.quadrature import evaluate_at_xq, integrate_against

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
DATA_FILE = "data/gvec_nfp3_hegna_80cubed_clebsch.h5"
NFP = 3

# FEM resolution
NS = (6, 6, 6)
PS = (2, 2, 2)
QUAD_ORDER = 4

# Relaxation
NUM_ITERS_OUTER = 10
NUM_ITERS_INNER = 100
DT0 = 1.0
FORCE_TOLERANCE = 1e-9

# %%
# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
print("Loading data...")
with h5py.File(DATA_FILE, "r") as f:
    pts = jnp.array(f["eval_points"])   # (512000, 3) — (ρ, θ, ζ) ∈ [0,1]³
    R_vals = jnp.array(f["R"])          # (512000,)
    Z_vals = jnp.array(f["Z"])          # (512000,)
    B_vals = jnp.array(f["B"])          # (512000, 3)
    phi_vals = jnp.array(f["clebsch/Phi"]).ravel()
    chi_vals = jnp.array(f["clebsch/chi"]).ravel()
    lambda_vals = jnp.array(f["clebsch/LA"]).ravel()

# Extract 1-D grid axes (80 per direction)
rho = jnp.array(np.unique(np.asarray(pts[:, 0])))
theta = jnp.array(np.unique(np.asarray(pts[:, 1])))
zeta = jnp.array(np.unique(np.asarray(pts[:, 2])))
n_rho, n_theta, n_zeta = len(rho), len(theta), len(zeta)
print(f"Grid: {n_rho} × {n_theta} × {n_zeta}")

# ---------------------------------------------------------------
# Interpolate the coordinate map
# ---------------------------------------------------------------
print("Building map interpolation sequence...")
map_seq = DeRhamSequence(
    NS, PS, QUAD_ORDER,
    ("clamped", "periodic", "periodic"),
    polar=True, tol=1e-9
)
map_seq.set_map(lambda x: x)
map_seq.evaluate_1d()
map_seq.assemble_mass_matrix(0)

print("Interpolating coordinate map (r, θ, ζ) -> (x, y, z)...")
t0 = time.time()
R_grid = R_vals.reshape(n_rho, n_theta, n_zeta)
Z_grid = Z_vals.reshape(n_rho, n_theta, n_zeta)
map_func = interpolate_map((rho, theta, zeta), R_grid, Z_grid, NFP, map_seq)
# map_func = jax.jit(map_func)
# # Warm-up JIT
# _ = map_func(jnp.array([0.5, 0.5, 0.5]))
print(f"Map interpolation done in {time.time() - t0:.1f}s")

# %%
# ---------------------------------------------------------------
# Build the FEM de Rham sequence
# ---------------------------------------------------------------
mrx.MAP_BATCH_SIZE_INNER = 0
print(f"Building FEM sequence (ns={NS}, ps={PS})...")
t0 = time.time()
seq = DeRhamSequence(
    NS, PS, QUAD_ORDER,
    ("clamped", "periodic", "periodic"),
    polar=True, tol=1e-6, maxiter=1000
)
seq.set_map(map_func)
mrx.MAP_BATCH_SIZE_INNER = 0
assert jnp.min(seq.jacobian_j) > 0, "Negative Jacobian detected!"
print(f"Jacobian range: [{float(jnp.min(seq.jacobian_j)):.2e}, "
      f"{float(jnp.max(seq.jacobian_j)):.2e}]")

seq.evaluate_1d()
seq.assemble_all_sparse()
print(f"FEM setup done in {time.time() - t0:.1f}s")
# %%
t0 = time.time()
print(f"Assembly done, computing nullspaces...")
seq.compute_nullspaces()
print(f"Nullspace computation done in {time.time() - t0:.1f}s")

# %%
# Check nullspaces
print(
    f"0 (no dbc): {jnp.sqrt(seq.null_0[0] @ seq.apply_laplacian(seq.null_0[0], 0, dirichlet=False))}")
print(
    f"1 (no dbc): {jnp.sqrt(seq.null_1[0] @ seq.apply_laplacian(seq.null_1[0], 1, dirichlet=False))}")
print(
    f"2 (dbc): {jnp.sqrt(seq.null_2_dbc[0] @ seq.apply_laplacian(seq.null_2_dbc[0], 2, dirichlet=True))}")
print(
    f"3 (dbc): {jnp.sqrt(seq.null_3_dbc[0] @ seq.apply_laplacian(seq.null_3_dbc[0], 3, dirichlet=True))}")


# %%
# ---------------------------------------------------------------
# Project B-field via Clebsch potentials: B = ∇φ×(∇θ+∇λ) - ∇χ×∇ζ
# ---------------------------------------------------------------
print("Projecting B-field via Clebsch potentials...")
t0 = time.time()


def interpolate_scalar_to_xq(scalar_vals, axes, xq):
    """Interpolate a scalar field from a regular grid onto query points."""
    grid = scalar_vals.reshape(len(axes[0]), len(axes[1]), len(axes[2]))
    interp = RegularGridInterpolator(points=axes, values=grid, method='linear')
    return interp(xq)


# Interpolate Clebsch potentials to FEM quadrature points
phi_at_xq = interpolate_scalar_to_xq(phi_vals, (rho, theta, zeta), seq.quad.x)
chi_at_xq = interpolate_scalar_to_xq(chi_vals, (rho, theta, zeta), seq.quad.x)
lambda_at_xq = interpolate_scalar_to_xq(
    lambda_vals, (rho, theta, zeta), seq.quad.x)

# L2-project scalars onto 0-form basis
comp_info_0, comp_shapes_0 = seq._form_comp_info(0)
quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)


@jax.jit
def project_scalar_to_0form(vals_at_xq):
    """L2-project scalar values at quad points onto 0-form DOFs."""
    rhs = seq.e0 @ integrate_against(
        (vals_at_xq * seq.jacobian_j * seq.quad.w)[:, None],
        comp_info_0, comp_shapes_0, quad_shape)
    return seq.apply_inverse_mass_matrix(rhs, 0, dirichlet=False)


phi_dof = project_scalar_to_0form(phi_at_xq)
chi_dof = project_scalar_to_0form(chi_at_xq)
lambda_dof = project_scalar_to_0form(lambda_at_xq)
print(f"Clebsch scalar projection done in {time.time() - t0:.1f}s")

# %%
# Build B from Clebsch: B = ∇φ×(∇θ+∇λ) - ∇χ×∇ζ
print("Computing B from Clebsch formula...")
t0 = time.time()

# comp_info for gradient of a 0-form (3 output dims, same DOF shape)
types = seq.basis_0.types
grad_r = seq._grad_1d(seq.d_basis_r_jk, types[0])
grad_t = seq._grad_1d(seq.d_basis_t_jk, types[1])
grad_z = seq._grad_1d(seq.d_basis_z_jk, types[2])
d0_comp_info = [
    (0, grad_r, seq.basis_t_jk, seq.basis_z_jk),
    (1, seq.basis_r_jk, grad_t, seq.basis_z_jk),
    (2, seq.basis_r_jk, seq.basis_t_jk, grad_z),
]
s0 = list(seq.basis_0.shape)[0]  # (nr, nt, nz)
d0_comp_shapes = [s0, s0, s0]


def grad_0form_at_xq(dof_0):
    """Evaluate the (logical) gradient of a 0-form at quad points."""
    internal = seq.e0_T @ dof_0
    return evaluate_at_xq(jnp.tile(internal, 3), d0_comp_info, d0_comp_shapes,
                          quad_shape, 3)


grad_phi = grad_0form_at_xq(phi_dof)
grad_chi = grad_0form_at_xq(chi_dof)
grad_lambda = grad_0form_at_xq(lambda_dof)
grad_theta = jnp.array([0, 1, 0])[None, :] * 2 * jnp.pi
grad_zeta = jnp.array([0, 0, 1])[None, :] * 2 * jnp.pi / NFP

B_jk = jnp.cross(grad_phi, grad_theta + grad_lambda) - \
    jnp.cross(grad_chi, grad_zeta)

# Project onto 2-form: ∫ Λ_2 · G B / J w dV
comp_info_2, comp_shapes_2 = seq._form_comp_info(2)
GB_jk = jnp.einsum('jkl,jk,j,j->jl', seq.metric_jkl, B_jk,
                   seq.quad.w, 1 / seq.jacobian_j)
B_proj = seq.e2_dbc @ integrate_against(GB_jk,
                                        comp_info_2, comp_shapes_2, quad_shape)

# Solve M2 B = B_proj
_inv_mass_2 = jax.jit(lambda rhs: seq.apply_inverse_mass_matrix(rhs, 2))
B_dof_0 = _inv_mass_2(B_proj)

# Divergence cleaning
div_B = float(seq.l2_norm(seq.apply_strong_div(B_dof_0), 3))
print(f"div(B) before Leray projection: {div_B:.2e}")

B_dof_0, _ = seq.apply_leray_projection(B_dof_0, k=2)
energy_0 = float(0.5 * seq.l2_norm_sq(B_dof_0, 2))
B_dof_0 = B_dof_0 / seq.l2_norm(B_dof_0, 2)
print(f"Clebsch B projection done in {time.time() - t0:.1f}s")

# %%
# Regularity coeff.
reg_coeff = B_dof_0 @ seq.apply_laplacian(
    B_dof_0, 2) / seq.l2_norm_sq(B_dof_0, 2)
print(f"Regularity coefficient: {reg_coeff:.2e}")
# %%
# for _ in range(5):
#     B_dof_0 = apply_diffusion(B_dof_0, seq, eta=1e-4)
#     reg_coeff = B_dof_0 @ seq.apply_laplacian(B_dof_0, 2) / seq.l2_norm_sq(B_dof_0, 2)
#     print(f"Regularity coefficient: {reg_coeff:.2e}")

# %%
F_force, p_dof, J_dof, H_dof, _ = compute_force(B_dof_0, seq)

# Discrete functions (reference domain)
B_h = DiscreteFunction(B_dof_0, seq.basis_2, seq.e2_dbc)
J_h = DiscreteFunction(J_dof, seq.basis_1, seq.e1_dbc)
p_h = DiscreteFunction(p_dof, seq.basis_3, seq.e3_dbc)

# Pushforward to physical domain
B_phys = Pushforward(B_h, seq.map, k=2)
J_phys = Pushforward(J_h, seq.map, k=1)
p_phys = Pushforward(p_h, seq.map, k=3)

# Radial grid at theta=0, zeta=0
_x, _y, _, (_x1, _, _) = get_1d_grids(seq.map, zeta=0.0, chi=0.0, nx=128)

B_vals_rad = jax.lax.map(B_phys, _x, batch_size=0)
J_vals_rad = jax.lax.map(J_phys, _x, batch_size=0)
p_vals_rad = jax.lax.map(p_phys, _x, batch_size=0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# B components
for i, label in enumerate(["x", "y", "z"]):
    axes[0].plot(_x1, B_vals_rad[:, i], label=rf"$B_{label}$")
axes[0].set_xlabel(r"$\rho$")
axes[0].set_ylabel(r"$B$")
axes[0].set_title(r"$\mathbf{B}$ radial profile")
axes[0].legend()

# J components
for i, label in enumerate(["x", "y", "z"]):
    axes[1].plot(_x1, J_vals_rad[:, i], label=rf"$J_{label}$")
axes[1].set_xlabel(r"$\rho$")
axes[1].set_ylabel(r"$J$")
axes[1].set_title(r"$\mathbf{J}$ radial profile")
axes[1].legend()

# p profile
p_vals_flat = p_vals_rad.flatten()
axes[2].plot(_x1, p_vals_flat)
axes[2].set_xlabel(r"$\rho$")
axes[2].set_ylabel(r"$p$")
axes[2].set_title(r"Pressure radial profile")

fig.tight_layout()
plt.show()

# %%
# ---------------------------------------------------------------
# Set up the time stepper and run relaxation
# ---------------------------------------------------------------
print("Setting up time stepper...")
ts = TimeStepper(
    seq=seq,
    descent_method=DescentMethod.GRADIENT,
    dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
    timestep_mode=IntegrationScheme.EXPLICIT,
)

print(
    f"Running relaxation: {NUM_ITERS_OUTER} outer × {NUM_ITERS_INNER} inner...")
t0 = time.time()
final_state, traces = relaxation_loop(
    B_dof_0,
    ts,
    num_iters_outer=NUM_ITERS_OUTER,
    num_iters_inner=NUM_ITERS_INNER,
    dt0=DT0,
    force_tolerance=FORCE_TOLERANCE,
    key=jax.random.PRNGKey(42),
)
relax_time = time.time() - t0

# %%
# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
final_force = float(traces["force_norm"][-1])
final_energy = float(traces["energy"][-1])
final_helicity = float(traces["helicity"][-1])
init_helicity = float(traces["helicity"][0])
helicity_change = abs((final_helicity - init_helicity) /
                      (abs(init_helicity) + 1e-30))

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Final force norm:          {final_force:.2e}")
print(f"Final energy:              {final_energy:.6f}")
print(f"Relative helicity change:  {helicity_change:.2e}")
print(f"Relaxation time:           {relax_time:.1f}s")
print("=" * 60)

# %%
# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

iterations = np.array(traces["iteration"])
force_norm = np.array(traces["force_norm"])
energy = np.array(traces["energy"])
helicity = np.array(traces["helicity"])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Force norm
axes[0].semilogy(iterations, force_norm)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel(r"$\|F\|$")
axes[0].set_title("Force norm")

# Energy change
dE = (energy - energy[0]) / abs(energy[0])
axes[1].plot(iterations, dE)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel(r"$\Delta E / |E_0|$")
axes[1].set_title("Relative energy change")

# Helicity change
dH = (helicity - helicity[0]) / (abs(helicity[0]) + 1e-30)
axes[2].plot(iterations, dH)
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel(r"$\Delta H / |H_0|$")
axes[2].set_title("Relative helicity change")

fig.tight_layout()
plt.savefig("relax_hegna_traces.png", dpi=150)
plt.show()

# %%
# ---------------------------------------------------------------
# Radial profiles of B, J, p
# ---------------------------------------------------------------
F_force, p_dof, J_dof, H_dof, _ = compute_force(final_state.B_n, seq)

# Discrete functions (reference domain)
B_h = DiscreteFunction(final_state.B_n, seq.basis_2, seq.e2_dbc)
J_h = DiscreteFunction(J_dof, seq.basis_1, seq.e1_dbc)
p_h = DiscreteFunction(p_dof, seq.basis_3, seq.e3_dbc)

# Pushforward to physical domain
B_phys = Pushforward(B_h, seq.map, k=2)
J_phys = Pushforward(J_h, seq.map, k=1)
p_phys = Pushforward(p_h, seq.map, k=3)

# Radial grid at theta=0, zeta=0
_x, _y, _, (_x1, _, _) = get_1d_grids(seq.map, zeta=0.0, chi=0.0, nx=128)

B_vals_rad = jax.lax.map(B_phys, _x, batch_size=0)
J_vals_rad = jax.lax.map(J_phys, _x, batch_size=0)
p_vals_rad = jax.lax.map(p_phys, _x, batch_size=0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# B components
for i, label in enumerate(["x", "y", "z"]):
    axes[0].plot(_x1, B_vals_rad[:, i], label=rf"$B_{label}$")
axes[0].set_xlabel(r"$\rho$")
axes[0].set_ylabel(r"$B$")
axes[0].set_title(r"$\mathbf{B}$ radial profile")
axes[0].legend()

# J components
for i, label in enumerate(["x", "y", "z"]):
    axes[1].plot(_x1, J_vals_rad[:, i], label=rf"$J_{label}$")
axes[1].set_xlabel(r"$\rho$")
axes[1].set_ylabel(r"$J$")
axes[1].set_title(r"$\mathbf{J}$ radial profile")
axes[1].legend()

# p profile
p_vals_flat = p_vals_rad.flatten()
axes[2].plot(_x1, p_vals_flat)
axes[2].set_xlabel(r"$\rho$")
axes[2].set_ylabel(r"$p$")
axes[2].set_title(r"Pressure radial profile")

fig.tight_layout()
plt.show()

# %%
for k in range(4):
    M = assemble_dense_mass_matrix(seq, k)
    operators = seq.get_operators()
    mass_diaginv_dbc = get_mass_jacobi_diaginv(operators.mass_preconds, k, True)
    match k:
        case 0:
            pM = jnp.diag(jnp.sqrt(mass_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(mass_diaginv_dbc))
        case 1:
            pM = jnp.diag(jnp.sqrt(mass_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(mass_diaginv_dbc))
        case 2:
            pM = jnp.diag(jnp.sqrt(mass_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(mass_diaginv_dbc))
        case 3:
            pM = jnp.diag(jnp.sqrt(mass_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(mass_diaginv_dbc))
    print(f"Condition number of M{k}:", jnp.linalg.eigvalsh(
        M)[-1]/jnp.linalg.eigvalsh(M)[0])
    print(f"Condition number of preconditioned M{k}:", jnp.linalg.eigvalsh(
        pM)[-1]/jnp.linalg.eigvalsh(pM)[0])
    print(f"first 3 eigenvalues of M{k}:", jnp.linalg.eigvalsh(M)[:3])
    print(
        f"first 3 eigenvalues of preconditioned M{k}:", jnp.linalg.eigvalsh(pM)[:3])

# %%
for k in range(4):
    M = assemble_dense_laplacian(seq, k)
    operators = seq.get_operators()
    hodge_diaginv_dbc = getattr(operators, f"dd{k}_sp_diaginv_dbc")
    match k:
        case 0:
            pM = jnp.diag(jnp.sqrt(hodge_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(hodge_diaginv_dbc))
            has_null = False
        case 1:
            pM = jnp.diag(jnp.sqrt(hodge_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(hodge_diaginv_dbc))
            has_null = False
        case 2:
            pM = jnp.diag(jnp.sqrt(hodge_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(hodge_diaginv_dbc))
            has_null = True
        case 3:
            pM = jnp.diag(jnp.sqrt(hodge_diaginv_dbc)
                          ) @ M @ jnp.diag(jnp.sqrt(hodge_diaginv_dbc))
            has_null = True
    print(f"Condition number of L{k}:", jnp.linalg.eigvalsh(
        M)[-1]/jnp.linalg.eigvalsh(M)[int(has_null)])
    print(f"Condition number of preconditioned L{k}:", jnp.linalg.eigvalsh(
        pM)[-1]/jnp.linalg.eigvalsh(pM)[int(has_null)])
    print(f"first 3 eigenvalues of L{k}:", jnp.linalg.eigvalsh(M)[:3])
    print(
        f"first 3 eigenvalues of preconditioned L{k}:", jnp.linalg.eigvalsh(pM)[:3])
# %%
