"""Debug script for interpolate_map returning all zeros."""
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.io import project_sampled_field
from mrx.mappings import interpolate_map, stellarator_map, toroid_map
from mrx.utils import integrate_against

jax.config.update("jax_enable_x64", True)


# ── 1. Build the same seq as in the test fixture ──
print("=== Building sequence ===")
seq = DeRhamSequence(
    (6, 6, 6), (3, 3, 3), 6,
    ("clamped", "periodic", "periodic"),
    map=lambda x: x, polar=False,
)
seq.evaluate_1d()
seq.assemble_mass_matrix(0)

# ── 2. Build sampling grid + reference map ──
F = toroid_map(epsilon=1/3, kappa=1.0, R0=1.0)
nfp = 1
_n = 20
_r = jnp.linspace(0, 1, _n)
_t = jnp.linspace(0, 1, _n)
_z = jnp.linspace(0, 1, _n)
_ri, _ti, _zi = jnp.meshgrid(_r, _t, _z, indexing="ij")
grid_pts = jnp.stack([_ri.ravel(), _ti.ravel(), _zi.ravel()], axis=1)

F_grid = jax.vmap(F)(grid_pts)
R_flat = (F_grid[:, 0]**2 + F_grid[:, 1]**2)**0.5
Z_flat = F_grid[:, 2]
R_grid = R_flat.reshape(_n, _n, _n)
Z_grid = Z_flat.reshape(_n, _n, _n)

print(f"R_grid range: [{float(R_grid.min()):.4f}, {float(R_grid.max()):.4f}]")
print(f"Z_grid range: [{float(Z_grid.min()):.4f}, {float(Z_grid.max()):.4f}]")

# ── 3. Check RegularGridInterpolator ──
print("\n=== RegularGridInterpolator check ===")
interp_R = RegularGridInterpolator(
    points=(_r, _t, _z), values=R_grid, method='linear')
xq = seq.quad.x
R_at_quad = interp_R(xq)
print(f"Quad points shape: {xq.shape}")
print(f"Quad points range: [{float(xq.min()):.4f}, {float(xq.max()):.4f}]")
print(
    f"R at quad points range: [{float(R_at_quad.min()):.4f}, {float(R_at_quad.max()):.4f}]")
print(f"R at quad all zero? {bool(jnp.allclose(R_at_quad, 0))}")

# ── 4. Check quadrature weights and Jacobian ──
print("\n=== Quadrature weights & Jacobian ===")
print(
    f"quad.w range: [{float(seq.quad.w.min()):.6f}, {float(seq.quad.w.max()):.6f}]")
print(
    f"jacobian_j range: [{float(seq.jacobian_j.min()):.6f}, {float(seq.jacobian_j.max()):.6f}]")
print(f"quad.w all zero? {bool(jnp.allclose(seq.quad.w, 0))}")
print(f"jacobian_j all zero? {bool(jnp.allclose(seq.jacobian_j, 0))}")

# ── 5. Step through project_sampled_field k=0 ──
print("\n=== project_sampled_field internals (k=0, dirichlet=False) ===")
axes = (_r, _t, _z)
values = R_grid
k = 0
dirichlet = False

x1, x2, x3 = axes
n1, n2, n3 = len(x1), len(x2), len(x3)
comp_info, comp_shapes = seq._form_comp_info(k)
quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
print(f"quad_shape: {quad_shape}")
print(f"comp_shapes: {comp_shapes}")

grid = values.reshape(n1, n2, n3)
interp = RegularGridInterpolator(
    points=(x1, x2, x3), values=grid, method='linear')
f_q = interp(xq)[:, None]
print(
    f"f_q shape: {f_q.shape}, range: [{float(f_q.min()):.4f}, {float(f_q.max()):.4f}]")
print(f"f_q all zero? {bool(jnp.allclose(f_q, 0))}")

w_jk = f_q * (seq.quad.w * seq.jacobian_j)[:, None]
print(
    f"w_jk shape: {w_jk.shape}, range: [{float(w_jk.min()):.6e}, {float(w_jk.max()):.6e}]")
print(f"w_jk all zero? {bool(jnp.allclose(w_jk, 0))}")

# ── 6. integrate_against ──
print("\n=== integrate_against ===")
rhs_raw = integrate_against(w_jk, comp_info, comp_shapes, quad_shape)
print(f"rhs_raw shape: {rhs_raw.shape}")
print(
    f"rhs_raw range: [{float(rhs_raw.min()):.6e}, {float(rhs_raw.max()):.6e}]")
print(f"rhs_raw all zero? {bool(jnp.allclose(rhs_raw, 0))}")
print(f"rhs_raw norm: {float(jnp.linalg.norm(rhs_raw)):.6e}")

# ── 7. Extraction operator ──
print("\n=== Extraction operator ===")
e = seq.e0
print(f"e0 shape: {e.shape}")
rhs = e @ rhs_raw
print(f"rhs (after e@) shape: {rhs.shape}")
print(f"rhs range: [{float(rhs.min()):.6e}, {float(rhs.max()):.6e}]")
print(f"rhs all zero? {bool(jnp.allclose(rhs, 0))}")
print(f"rhs norm: {float(jnp.linalg.norm(rhs)):.6e}")

# ── 8. Inverse mass matrix ──
print("\n=== apply_inverse_mass_matrix ===")
dof = seq.apply_inverse_mass_matrix(rhs, k=0, dirichlet=False)
print(f"dof shape: {dof.shape}")
print(f"dof range: [{float(dof.min()):.6e}, {float(dof.max()):.6e}]")
print(f"dof all zero? {bool(jnp.allclose(dof, 0))}")
print(f"dof norm: {float(jnp.linalg.norm(dof)):.6e}")

# ── 9. Reconstruct DiscreteFunction and evaluate ──
print("\n=== DiscreteFunction evaluation ===")
R_h = DiscreteFunction(dof, seq.basis_0, seq.e0)
test_pt = jnp.array([0.5, 0.25, 0.25])
val = R_h(test_pt)
print(f"R_h(0.5, 0.25, 0.25) = {val}")

# Compare with reference
R_ref = (F(test_pt)[0]**2 + F(test_pt)[1]**2)**0.5
print(f"R_ref(0.5, 0.25, 0.25) = {float(R_ref):.6f}")

# ── 10. Full interpolate_map call ──
print("\n=== Full interpolate_map ===")
F_h = interpolate_map((_r, _t, _z), R_grid, Z_grid, nfp, seq, flip_zeta=False)
test_pts = jnp.array([[0.5, 0.25, 0.25], [0.3, 0.1, 0.1], [0.0, 0.0, 0.0]])
for pt in test_pts:
    fh = F_h(pt)
    fref = F(pt)
    print(f"  pt={pt}  F_h={fh}  F={fref}")
    print(f"  pt={pt}  F_h={fh}  F={fref}")
