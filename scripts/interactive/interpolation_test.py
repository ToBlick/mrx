# %%
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import rotating_ellipse_map, stellarator_map

jax.config.update("jax_enable_x64", True)

# Define analytical mapping (Toroid):
ɛ = 1/3
π = jnp.pi
nfp = 1
F = jax.jit(rotating_ellipse_map(ɛ, 1.2, nfp))


def X1_analytic(x):
    # the R coordinate
    y = F(x)
    return (y[0]**2 + y[1]**2)**0.5


def X2_analytic(x):
    # the Z coordinate
    return F(x)[2]


n_interp = 4
n_resolution = 4

# %%


@partial(jax.jit, static_argnums=(0, 1, 2))
def check_interpolation(n_interp, n_resolution, p):

    resolutions = [n_resolution] * 3
    spline_degree = p
    spline_degrees = [spline_degree] * 3
    quad_order = spline_degree - 1
    map_seq = DeRhamSequence(resolutions, spline_degrees, quad_order, ("clamped",
                                                                       "periodic", "periodic"), lambda x: x, polar=False, dirichlet=False)

    # Define interpolation points (uniform grid in logical space)
    _pts = jnp.linspace(1e-3, 1-1e-3, n_interp)
    # Set up the interpolation problem:
    # ∑ c_ki Λ0[i](ρ,θ*(ρ,θ,ζ),ζ)_j ≈ Xk(ρ,θ,ζ)_j ∀j
    # evaluation grid, shape (mρ, mθ, mζ)
    ρ, θ, ζ = jnp.meshgrid(_pts, _pts, _pts, indexing="ij")
    # θ_star = jnp.asarray(θ_star)
    pts = jnp.stack([ρ.ravel(), θ.ravel(), ζ.ravel()],
                    axis=1)  # x_hat_js, shape (mρ mθ mζ, 3)

    X1_eval, X2_eval = jax.vmap(lambda x: (
        X1_analytic(x), X2_analytic(x)))(pts)

    M = jax.vmap(lambda i: jax.vmap(lambda x: map_seq.Lambda_0[i](x)[0])(pts))(
        map_seq.Lambda_0.ns).T  # Λ0[i](x_hat_j)
    y = jnp.stack([X1_eval, X2_eval], axis=1)  # X_α(x'_j)
    c, resid, rank, _ = jnp.linalg.lstsq(M, y, rcond=None)
    X1_h = DiscreteFunction(c[:, 0], map_seq.Lambda_0, map_seq.E0)
    X2_h = DiscreteFunction(c[:, 1], map_seq.Lambda_0, map_seq.E0)

    F_h = jax.jit(stellarator_map(X1_h, X2_h, nfp=nfp, flip_zeta=False))

    # Verify on a new, denser grid
    n_test = int(n_interp * 1.5)
    test_pts = jnp.linspace(1e-3, 1-1e-3, n_test)
    ρ, θ, ζ = jnp.meshgrid(test_pts, test_pts, test_pts, indexing="ij")
    # θ_star = jnp.asarray(θ_star)
    test_pts = jnp.stack([ρ.ravel(), θ.ravel(), ζ.ravel()], axis=1)

    F_exact = jax.vmap(F)(test_pts)
    F_interp = jax.vmap(F_h)(test_pts)

    error = jnp.linalg.norm(F_exact - F_interp, axis=1)
    inf_error = jnp.max(error)
    L2_error = jnp.sqrt(jnp.mean(error**2))

    # Get the mapped sequence
    seq = DeRhamSequence(resolutions, spline_degrees, quad_order, ("clamped",
                                                                   "periodic", "periodic"), F_h, polar=True, dirichlet=True)

    def Λ2_phys(i, x):
        # * jnp.linalg.det(jax.jacfwd(seq.F)(x))
        return Pushforward(lambda x: seq.Lambda_2[i](x), seq.F, 2)(x)

    # def Λ2(i, x):
    #     return seq.Lambda_2[i](x) #* jnp.linalg.det(jax.jacfwd(seq.F)(x))
    # B = 1/R eϕ
    def B_xyz(x):
        y = F(x)
        R = (y[0]**2 + y[1]**2)**0.5
        # * jnp.linalg.det(jax.jacfwd(seq.F)(x))
        return jnp.array([-y[1], y[0], 0.0]) / R**2

    B_xyz_at_pts = jax.vmap(B_xyz)(
        pts)  # * jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(seq.F)(x)))(pts)[:, None]

    def body_fun(_, i):
        # Evaluate Λ2_phys(i, x) for all points (vectorized over x)
        return None, jax.vmap(lambda x: Λ2_phys(i, x))(pts)
        # return None, jax.vmap(lambda x: Λ2(i, x))(pts)

    _, M = jax.lax.scan(body_fun, None, seq.Lambda_2.ns)
    M = jnp.einsum('il,ljk->ijk', seq.E2, M)    # Λ2[i](x_hat_j)_k
    y = B_xyz_at_pts.reshape(-1, 3)              # B(x'_j)_k
    # y = B_at_pts.reshape(-1, 3)              # B(x'_j)_k

    # Solve least squares interpolation:
    # ∑ c_ik Λ2[i](ρ,θ,ζ)_j ≈ B_k(ρ,θ,ζ)_j ∀j,k
    # i.e. M @ C ≈ B where M is (num_basis, num_pts, 3) and B is (num_pts, 3)

    A = M.reshape(M.shape[0], -1).T         # reshape to (num_pts*3, num_basis)
    b = y.ravel()                           # reshape to (num_pts*3,)

    J = jax.vmap(lambda x: jnp.abs(jnp.linalg.det(jax.jacfwd(seq.F)(x))))(pts)
    weights = jnp.sqrt(J)
    # Expand weights to match the 3 components per point
    weights_expanded = jnp.repeat(weights, 3)

    A_weighted = A * weights_expanded[:, None]
    b_weighted = b * weights_expanded

    # Transform to minimize B_dof.T @ M2 @ B_dof instead of |B_dof|^2
    # If M2 = S.T @ S, then substituting z = S @ B_dof gives min |z|^2
    # Solve (A @ S^{-1}) @ z = b, then B_dof = S^{-1} @ z
    seq.evaluate_1d()
    seq.assemble_M2()
    L = jnp.linalg.cholesky(seq.M2)  # M2 = L @ L.T
    S = L.T                           # S.T @ S = M2
    S_inv = jnp.linalg.inv(S)

    A_transformed = A_weighted @ S_inv
    z, B_residuals, B_rank, _ = jnp.linalg.lstsq(
        A_transformed, b_weighted, rcond=None)
    B_dof = S_inv @ z

    B_h = jax.jit(DiscreteFunction(B_dof, seq.Lambda_2, seq.E2))
    B_h_xyz = jax.jit(Pushforward(B_h, seq.F, 2))

    B_exact = jax.vmap(B_xyz)(test_pts)
    B_interp = jax.vmap(B_h_xyz)(test_pts)

    B_error = jnp.linalg.norm(B_exact - B_interp, axis=1) / \
        jnp.linalg.norm(B_exact, axis=1)
    B_inf_error = jnp.max(B_error)
    B_L2_error = jnp.sqrt(jnp.mean(B_error**2))
    # condition number of A and A_weighted
    B_cond_A = jnp.linalg.cond(A)
    B_cond_A_weighted = jnp.linalg.cond(A_weighted)

    return {"map_inf_error": inf_error,
            "map_L2_error": L2_error,
            "map_resid": resid,
            "map_rank": rank,
            "B_inf_error": B_inf_error,
            "B_L2_error": B_L2_error,
            "B_resid": B_residuals[0],
            "B_rank": B_rank,
            "B_cond_A": B_cond_A,
            "B_cond_A_weighted": B_cond_A_weighted,
            }


# %%
# Run parameter sweep with n_interp = n_res
# Define parameter range (n_interp = n_res)
n_values = [4, 5, 6, 7]

# Storage for results
results = []

print("Running parameter sweep (n_interp = n_res)...")
for n in n_values:
    print(f"\nTesting n_interp = n_res = {n}")
    result = check_interpolation(n*2, n, min(6, n-1))
    print("  Geometry interpolation:")
    print(f"    L-infinity error: {result['map_inf_error']:.3e}")
    print(f"    L2 error:         {result['map_L2_error']:.3e}")
    print(f"    Residuals:        {result['map_resid']}")
    print(f"    Rank:             {result['map_rank']}")
    print("  B-field interpolation:")
    print(f"    L-infinity error: {result['B_inf_error']:.3e}")
    print(f"    L2 error:         {result['B_L2_error']:.3e}")
    print(f"    Residuals:        {result['B_resid']}")
    print(f"    Rank:             {result['B_rank']}")
    print(f"    Condition number (unweighted): {result['B_cond_A']:.3e}")
    print(
        f"    Condition number (weighted):   {result['B_cond_A_weighted']:.3e}")
    results.append({
        'n': n,
        'map_inf_error': float(result['map_inf_error']),
        'map_L2_error': float(result['map_L2_error']),
        'map_resid_0': float(result['map_resid'][0]),
        'map_resid_1': float(result['map_resid'][1]),
        'map_rank': int(result['map_rank']),
        'B_inf_error': float(result['B_inf_error']),
        'B_L2_error': float(result['B_L2_error']),
        'B_resid': float(result['B_resid']),
        'B_rank': int(result['B_rank']),
        'B_cond_A': float(result['B_cond_A']),
        'B_cond_A_weighted': float(result['B_cond_A_weighted']),
    })

print("\nParameter sweep complete!")

# %%
# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Extract data
n_vals = [r['n'] for r in results]
map_inf_errors = [r['map_inf_error'] for r in results]
map_L2_errors = [r['map_L2_error'] for r in results]
B_inf_errors = [r['B_inf_error'] for r in results]
B_L2_errors = [r['B_L2_error'] for r in results]
map_resid_0 = [r['map_resid_0'] for r in results]
map_resid_1 = [r['map_resid_1'] for r in results]
B_resid = [r['B_resid'] for r in results]
B_cond_A = [r['B_cond_A'] for r in results]

# Plot 1: Geometry (Map) L2 error vs n
ax = axes[0, 0]
ax.semilogy(n_vals, map_L2_errors, marker='o',
            linewidth=2, markersize=8, color='#1f77b4', label='Map')
ax.set_xlabel('n (= n_interp = n_res)', fontsize=12)
ax.set_ylabel('L2 error', fontsize=12)
ax.set_title('Geometry Interpolation L2 Error', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: B-field L2 error vs n
ax = axes[0, 1]
ax.semilogy(n_vals, B_L2_errors, marker='s',
            linewidth=2, markersize=8, color='#ff7f0e', label='B-field')
ax.set_xlabel('n (= n_interp = n_res)', fontsize=12)
ax.set_ylabel('L2 error', fontsize=12)
ax.set_title('B-field Interpolation L2 Error', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Geometry residuals vs n
ax = axes[1, 0]
ax.semilogy(n_vals, map_resid_0, marker='^', linewidth=2,
            markersize=8, label='X₁ (R)', color='#2ca02c')
ax.semilogy(n_vals, map_resid_1, marker='v', linewidth=2,
            markersize=8, label='X₂ (Z)', color='#d62728')
ax.set_xlabel('n (= n_interp = n_res)', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title('Geometry Least Squares Residuals',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Comparison of all L∞ and L2 errors
ax = axes[1, 1]
ax.semilogy(n_vals, map_inf_errors, marker='o', linewidth=2,
            markersize=7, label='Map L∞', color='#1f77b4', linestyle='--')
ax.semilogy(n_vals, map_L2_errors, marker='o', linewidth=2,
            markersize=7, label='Map L2', color='#1f77b4', linestyle='-')
ax.semilogy(n_vals, B_inf_errors, marker='s', linewidth=2,
            markersize=7, label='B-field L∞', color='#ff7f0e', linestyle='--')
ax.semilogy(n_vals, B_L2_errors, marker='s', linewidth=2,
            markersize=7, label='B-field L2', color='#ff7f0e', linestyle='-')
ax.set_xlabel('n (= n_interp = n_res)', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('Error Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Print summary table
print("\n" + "="*130)
print("SUMMARY TABLE (n_interp = n_res)")
print("="*130)
print(f"{'n':<6} {'Map L∞':<12} {'Map L2':<12} {'Map Res0':<12} {'Map Res1':<12} "
      f"{'Map Rk':<8} {'B L∞':<12} {'B L2':<12} {'B Resid':<12} {'B Rank':<8}")
print("-"*130)
for r in results:
    print(f"{r['n']:<6} {r['map_inf_error']:<12.3e} {r['map_L2_error']:<12.3e} "
          f"{r['map_resid_0']:<12.3e} {r['map_resid_1']:<12.3e} {r['map_rank']:<8} "
          f"{r['B_inf_error']:<12.3e} {r['B_L2_error']:<12.3e} "
          f"{r['B_resid']:<12.3e} {r['B_rank']:<8}")

# %%
# 3D scatter plot of F(pts)
n_plot = 12
_pts = jnp.linspace(1e-3, 1-1e-3, n_plot)
ρ, θ, ζ = jnp.meshgrid(_pts, _pts, _pts, indexing="ij")
pts = jnp.stack([ρ.ravel(), θ.ravel(), ζ.ravel() / nfp], axis=1)

F_pts = jax.vmap(F)(pts)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(F_pts[:, 0], F_pts[:, 1], F_pts[:, 2],
           c=pts[:, 0], cmap='viridis', marker='o', s=2, alpha=0.6)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title(f'F(pts) - {n_plot}³ points in physical space',
             fontsize=13, fontweight='bold')
plt.colorbar(ax.scatter(F_pts[:, 0], F_pts[:, 1], F_pts[:, 2],
                        c=pts[:, 0], cmap='viridis', s=2),
             ax=ax, label='ρ (radial coordinate)')
plt.show()

# %%
