"""Check accuracy of inner mass solves used by the forward operator."""
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.solvers import solve_singular_cg

jax.config.update('jax_enable_x64', True)

types = ('clamped', 'periodic', 'periodic')
s = DeRhamSequence((6, 6, 6), (2, 2, 2), 4, types, toroid_map(epsilon=1/3),
                   polar=True, tol=1e-10, maxiter=1000)
s.evaluate_1d()
for k in range(4):
    s.assemble_mass_matrix(k)
for k in range(3):
    s.assemble_derivative_matrix(k)
for k in range(4):
    s.assemble_hodge_laplacian(k)
s.compute_nullspaces()

print("=" * 60)
print("Inner mass solve accuracy (used by apply_hodge_laplacian)")
print("=" * 60)
for k_lower in range(3):
    for dbc in [True, False]:
        n = getattr(s, f'n{k_lower}' + ('_dbc' if dbc else ''))
        b = jax.random.normal(jax.random.PRNGKey(
            42 + k_lower + 10 * dbc), (n,))
        x, info = solve_singular_cg(
            lambda x, kl=k_lower, d=dbc: s.apply_mass_matrix(
                x, kl, dirichlet=d),
            b,
            mass_matvec=lambda x, kl=k_lower, d=dbc: s.apply_mass_matrix(
                x, kl, dirichlet=d),
            precond_matvec=lambda x, kl=k_lower, d=dbc: s.apply_mass_matrix_preconditioner(
                x, kl, dirichlet=d),
            tol=1e-10, maxiter=1000)
        Mx = s.apply_mass_matrix(x, k_lower, dirichlet=dbc)
        err = jnp.max(jnp.abs(Mx - b))
        rel_err = jnp.linalg.norm(Mx - b) / jnp.linalg.norm(b)
        bc = 'dbc' if dbc else 'no_dbc'
        print(
            f'M_{k_lower} {bc:>6s}: info={info}, max|Mx-b|={err:.2e}, rel={rel_err:.2e}, n={n}')

print()
print("=" * 60)
print("Full forward operator round-trip: L^{-1} b -> L(L^{-1} b) =? b")
print("=" * 60)
for k in range(4):
    for dbc in [True, False]:
        n = getattr(s, f'n{k}' + ('_dbc' if dbc else ''))
        vs = s._get_nullspace(k, dbc)
        b = jax.random.normal(jax.random.PRNGKey(k + 100 * dbc), (n,))
        for v in vs:
            b = b - jnp.dot(v, b) * s.apply_mass_matrix(v, k, dirichlet=dbc)
        x = s.apply_inverse_hodge_laplacian(b, k, dirichlet=dbc)
        Lx = s.apply_hodge_laplacian(x, k, dirichlet=dbc)
        residual = Lx - b
        for v in vs:
            residual = residual - \
                jnp.dot(v, residual) * s.apply_mass_matrix(v, k, dirichlet=dbc)
        err = jnp.max(jnp.abs(residual))
        rel = jnp.linalg.norm(residual) / jnp.linalg.norm(b)
        bc = 'dbc' if dbc else 'no_dbc'
        print(f'k={k} {bc:>6s}: max|res|={err:.2e}, rel={rel:.2e}')
        print(f'k={k} {bc:>6s}: max|res|={err:.2e}, rel={rel:.2e}')
