from test.test_preconditioned_solves import _build_dense
from mrx.mappings import toroid_map
from mrx.derham_sequence import DeRhamSequence
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

n, p = 5, 2
types = ('clamped', 'periodic', 'periodic')
F = toroid_map(epsilon=1/3)
s = DeRhamSequence((n, n, n), (p, p, p), 2*p, types,
                   polar=True, tol=1e-12, maxiter=2000)
s.set_map(F)
s.evaluate_1d()
for k in range(4):
    s.assemble_mass_matrix(k)
for k in range(3):
    s.assemble_derivative_matrix(k)
for k in range(4):
    s.assemble_hodge_laplacian(k)
s.compute_nullspaces()

k, alpha, dbc = 2, 1e-2, False
n_dofs = s.n2
b = jax.random.normal(jax.random.PRNGKey(k + 100*dbc + 7), (n_dofs,))

# Inverse
x = s.apply_inverse_diffusion(b, k, alpha, dirichlet=dbc)

# Forward
MLx = s.apply_diffusion(x, k, alpha, dirichlet=dbc)

# Decompose
Mx = s.apply_mass_matrix(x, k, dirichlet=dbc)
Lx = s.apply_hodge_laplacian(x, k, dirichlet=dbc)
Sx = s.apply_stiffness(x, k, dirichlet=dbc)

# Inner mass solve accuracy
Dt_x = s.apply_derivative_matrix(
    x, 1, dirichlet_in=dbc, dirichlet_out=dbc, transpose=True)
Minv_Dt_x = s.apply_inverse_mass_matrix(Dt_x, 1, dirichlet=dbc)
M_Minv_Dt_x = s.apply_mass_matrix(Minv_Dt_x, 1, dirichlet=dbc)
inner_err = jnp.max(jnp.abs(M_Minv_Dt_x - Dt_x))

roundtrip_err = jnp.max(jnp.abs(MLx - b))

print(f"Inner M1 solve error: {inner_err:.2e}")
print(f"Round-trip ||(M+aL)x - b||_inf: {roundtrip_err:.2e}")
print(f"  |Mx|_inf: {jnp.max(jnp.abs(Mx)):.2e}")
print(f"  a|Lx|_inf: {alpha * jnp.max(jnp.abs(Lx)):.2e}")
print(f"  a|Sx|_inf: {alpha * jnp.max(jnp.abs(Sx)):.2e}")
print(f"  ||x||: {jnp.linalg.norm(x):.2e}")
print(f"  ||b||: {jnp.linalg.norm(b):.2e}")

# Also check: what if we use a DIFFERENT x (from dense solve)?
# Build dense (M + aL)

n_u = s.n2
n_s = s.n1
Mk = _build_dense(lambda x: s.apply_mass_matrix(x, 2, dirichlet=dbc), n_u)
Sk = _build_dense(lambda x: s.apply_stiffness(x, 2, dirichlet=dbc), n_u)
D = _build_dense(lambda sig: s.apply_derivative_matrix(
    sig, 1, dirichlet_in=dbc, dirichlet_out=dbc), n_s, n_u)
DT = _build_dense(lambda u: s.apply_derivative_matrix(
    u, 1, dirichlet_in=dbc, dirichlet_out=dbc, transpose=True), n_u, n_s)
Ml = _build_dense(lambda sig: s.apply_mass_matrix(sig, 1, dirichlet=dbc), n_s)

K = jnp.block([[Mk + alpha*Sk, alpha*D], [alpha*DT, -alpha*Ml]])
rhs = jnp.concatenate([b, jnp.zeros(n_s)])
x_direct = jnp.linalg.solve(K, rhs)
u_direct = x_direct[:n_u]

# Forward on dense solution
MLx_dense = s.apply_diffusion(u_direct, k, alpha, dirichlet=dbc)
roundtrip_dense = jnp.max(jnp.abs(MLx_dense - b))
print(f"\nDense x round-trip ||(M+aL)x_dense - b||_inf: {roundtrip_dense:.2e}")
print(f"||x - x_dense||_inf: {jnp.max(jnp.abs(x - u_direct)):.2e}")
print(f"\nDense x round-trip ||(M+aL)x_dense - b||_inf: {roundtrip_dense:.2e}")
print(f"||x - x_dense||_inf: {jnp.max(jnp.abs(x - u_direct)):.2e}")
