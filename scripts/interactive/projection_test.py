# # %%
# """
# Projection test script: Load DESC equilibrium and project R, Z, and B onto FEM spaces.

# This script demonstrates:
# 1. Creating callable wrappers for DESC equilibrium quantities (R, Z, B)
# 2. Projecting scalar functions R(x), Z(x) onto 0-form spaces
# 3. Using the interpolated geometry to project B(x) onto a 2-form space
# """
# import jax
# import jax.numpy as jnp
# import matplotlib.pyplot as plt

# # from mrx.desc_interface import DESCWrapper, project_desc_equilibrium

# jax.config.update("jax_enable_x64", True)


# # %%
# # =============================================================================
# # Test against DESC equilibrium
# # =============================================================================

# def test_desc_projection(desc_path: str, n_resolution: int = 8, p: int = 3):
#     """
#     Test projection by comparing interpolated values against DESC using quadrature.

#     For R and Z: uses standard quadrature on [0,1]^3 (identity map).
#     For B: uses quadrature weighted by the Jacobian of F_h (the interpolated geometry).

#     Args:
#         desc_path: Path to DESC equilibrium file
#         n_resolution: Number of basis functions in each direction
#         p: Polynomial degree

#     Returns:
#         Dictionary with error metrics
#     """
#     # Project the equilibrium
#     result = project_desc_equilibrium(desc_path, n_resolution, p)

#     X1_h = result['X1_h']
#     X2_h = result['X2_h']
#     B_h_xyz = result['B_h_xyz']
#     F_h = result['F_h']
#     map_seq = result['map_seq']
#     seq = result['seq']
#     wrapper = result['wrapper']

#     # ==========================================================================
#     # Compute R, Z errors using quadrature on identity map
#     # ==========================================================================
#     # Get DESC values at quadrature points
#     desc_vals = wrapper.compute_at_points(map_seq.Q.x)
#     R_exact = desc_vals['R']
#     Z_exact = desc_vals['Z']

#     # Get interpolated values at quadrature points
#     R_interp = jax.vmap(X1_h)(map_seq.Q.x).ravel()
#     Z_interp = jax.vmap(X2_h)(map_seq.Q.x).ravel()

#     # L2 error: sqrt(∫|f - f_h|^2 dx / ∫|f|^2 dx)
#     # For identity map, J = 1, so just use weights
#     R_diff_sq = (R_exact - R_interp)**2
#     Z_diff_sq = (Z_exact - Z_interp)**2

#     R_L2_error = jnp.sqrt(jnp.sum(R_diff_sq * map_seq.Q.w) /
#                           jnp.sum(R_exact**2 * map_seq.Q.w))
#     Z_L2_error = jnp.sqrt(jnp.sum(Z_diff_sq * map_seq.Q.w) /
#                           jnp.sum(Z_exact**2 * map_seq.Q.w))

#     # L∞ error: max |f - f_h| / max |f|
#     R_inf_error = jnp.max(jnp.abs(R_exact - R_interp)) / \
#         jnp.max(jnp.abs(R_exact))
#     Z_inf_error = jnp.max(jnp.abs(Z_exact - Z_interp)) / \
#         jnp.max(jnp.abs(Z_exact))

#     # ==========================================================================
#     # Compute B error using quadrature with geometry from F_h
#     # ==========================================================================
#     # Get DESC B values at quadrature points (using seq.Q.x which matches the B projection)
#     desc_vals_B = wrapper.compute_at_points(seq.Q.x)
#     B_exact = desc_vals_B['B']

#     # Get interpolated B values
#     B_interp = jax.vmap(B_h_xyz)(seq.Q.x)

#     J = seq.J_j  # Jacobian at quadrature points

#     # L2 error: sqrt(∫|B - B_h|^2 J dξ / ∫|B|^2 J dξ)
#     B_diff_sq = jnp.sum((B_exact - B_interp)**2, axis=1)
#     B_exact_sq = jnp.sum(B_exact**2, axis=1)

#     B_L2_error = jnp.sqrt(jnp.sum(B_diff_sq * J * seq.Q.w) /
#                           jnp.sum(B_exact_sq * J * seq.Q.w))

#     # L∞ error for B
#     B_diff_norm = jnp.sqrt(B_diff_sq)
#     B_exact_norm = jnp.sqrt(B_exact_sq)
#     B_inf_error = jnp.max(B_diff_norm) / jnp.max(B_exact_norm)

#     return {
#         'R_L2_error': float(R_L2_error),
#         'R_inf_error': float(R_inf_error),
#         'Z_L2_error': float(Z_L2_error),
#         'Z_inf_error': float(Z_inf_error),
#         'B_L2_error': float(B_L2_error),
#         'B_inf_error': float(B_inf_error),
#     }


# # %%
# # =============================================================================
# # Run tests
# # =============================================================================

# DESC_PATH = "../data/desc_heliotron.h5"  # Update this path to your DESC file

# print(f"Testing projection against DESC file: {DESC_PATH}")
# print("=" * 70)

# n_values = [4, 5, 6, 7, 8]
# results = []

# for n in n_values:
#     print(f"\nResolution n = {n}")
#     result = test_desc_projection(DESC_PATH, n_resolution=n, p=3)
#     print(
#         f"  R: L2={result['R_L2_error']:.3e}, L∞={result['R_inf_error']:.3e}")
#     print(
#         f"  Z: L2={result['Z_L2_error']:.3e}, L∞={result['Z_inf_error']:.3e}")
#     print(
#         f"  B: L2={result['B_L2_error']:.3e}, L∞={result['B_inf_error']:.3e}")
#     results.append({'n': n, **result})

# # %%
# # Visualize convergence
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# n_vals = [r['n'] for r in results]

# # L2 errors
# ax = axes[0]
# ax.semilogy(n_vals, [r['R_L2_error']
#             for r in results], 'o-', label='R', linewidth=2)
# ax.semilogy(n_vals, [r['Z_L2_error']
#             for r in results], 's-', label='Z', linewidth=2)
# ax.semilogy(n_vals, [r['B_L2_error']
#             for r in results], '^-', label='B', linewidth=2)
# ax.set_xlabel('n (resolution)', fontsize=12)
# ax.set_ylabel('Relative L2 Error', fontsize=12)
# ax.set_title('L2 Error Convergence', fontsize=13, fontweight='bold')
# ax.legend()
# ax.grid(True, alpha=0.3)

# # L∞ errors
# ax = axes[1]
# ax.semilogy(n_vals, [r['R_inf_error']
#             for r in results], 'o--', label='R', linewidth=2)
# ax.semilogy(n_vals, [r['Z_inf_error']
#             for r in results], 's--', label='Z', linewidth=2)
# ax.semilogy(n_vals, [r['B_inf_error']
#             for r in results], '^--', label='B', linewidth=2)
# ax.set_xlabel('n (resolution)', fontsize=12)
# ax.set_ylabel('Relative L∞ Error', fontsize=12)
# ax.set_title('L∞ Error Convergence', fontsize=13, fontweight='bold')
# ax.legend()
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # %%
# # Print summary
# print("\n" + "=" * 90)
# print("SUMMARY")
# print("=" * 90)
# print(f"{'n':<6} {'R L2':<12} {'R L∞':<12} {'Z L2':<12} {'Z L∞':<12} {'B L2':<12} {'B L∞':<12}")
# print("-" * 90)
# for r in results:
#     print(f"{r['n']:<6} {r['R_L2_error']:<12.3e} {r['R_inf_error']:<12.3e} "
#           f"{r['Z_L2_error']:<12.3e} {r['Z_inf_error']:<12.3e} "
#           f"{r['B_L2_error']:<12.3e} {r['B_inf_error']:<12.3e}")

# # %%
