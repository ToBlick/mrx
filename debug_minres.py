"""Verify MINRES saddle-point solve against direct dense solve."""
from mrx.solvers import solve_saddle_point_minres
from mrx.mappings import toroid_map
from mrx.derham_sequence import DeRhamSequence
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


types = ("clamped", "periodic", "periodic")
n, p = 6, 2
F = toroid_map(epsilon=1/3)
s = DeRhamSequence((n, n, n), (p, p, p), 2*p, types,
                   polar=True, tol=1e-10, maxiter=1000)
s.set_map(F)
s.evaluate_1d()
for k in range(4):
    s.assemble_mass_matrix(k)
for k in range(3):
    s.assemble_derivative_matrix(k)
for k in range(4):
    s.assemble_hodge_laplacian(k)


s.compute_nullspaces()


def build_dense(matvec, n_in, n_out=None):
    """Build dense matrix from a matvec by probing with unit vectors."""
    if n_out is None:
        n_out = n_in
    cols = []
    for i in range(n_in):
        e = jnp.zeros(n_in).at[i].set(1.0)
        cols.append(matvec(e))
    return jnp.column_stack(cols)


def test_k(k, dirichlet):
    suffix = "_dbc" if dirichlet else ""
    n_u = getattr(s, f"n{k}{suffix}")
    n_s = getattr(s, f"n{k-1}{suffix}") if k >= 1 else 0
    bc_label = "dbc" if dirichlet else "no_dbc"
    has_nullspace = len(s._get_nullspace(k, dirichlet)) > 0
    print(f"\n{'='*60}")
    print(f"k={k}, {bc_label}: n_upper={n_u}, n_lower={n_s}, nullspace={'yes' if has_nullspace else 'no'}")

    if k == 0:
        # k=0 uses CG on stiffness, just verify that path works
        b = jax.random.normal(jax.random.PRNGKey(k + 100*dirichlet), (n_u,))
        x = s.apply_inverse_hodge_laplacian(b, k, dirichlet=dirichlet)
        Lx = s.apply_hodge_laplacian(x, k, dirichlet=dirichlet)
        # For k=0, apply_hodge_laplacian is just stiffness (no inner CG)
        print(
            f"  CG path: ||Lx - b||/||b|| = {jnp.linalg.norm(Lx - b)/jnp.linalg.norm(b):.2e}")
        return

    # Build dense blocks
    S_dense = build_dense(
        lambda x: s.apply_stiffness(x, k, dirichlet=dirichlet), n_u)
    D_dense = build_dense(
        lambda sig: s.apply_derivative_matrix(
            sig, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet), n_s, n_u)
    DT_dense = build_dense(
        lambda u: s.apply_derivative_matrix(
            u, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True), n_u, n_s)
    M_dense = build_dense(
        lambda sig: s.apply_mass_matrix(sig, k-1, dirichlet=dirichlet), n_s)

    print(
        f"  ||D^T - D_dense^T|| = {jnp.linalg.norm(DT_dense - D_dense.T):.2e}")
    print(f"  ||S - S^T|| = {jnp.linalg.norm(S_dense - S_dense.T):.2e}")
    print(f"  ||M - M^T|| = {jnp.linalg.norm(M_dense - M_dense.T):.2e}")

    # Build full saddle-point matrix
    K_dense = jnp.block([
        [S_dense,   D_dense],
        [DT_dense, -M_dense]
    ])
    print(f"  ||K - K^T|| = {jnp.linalg.norm(K_dense - K_dense.T):.2e}")
    print(f"  cond(K) = {jnp.linalg.cond(K_dense):.2e}")

    # Random RHS
    b = jax.random.normal(jax.random.PRNGKey(k + 100*dirichlet), (n_u,))

    # Project b out of nullspace (so the system is consistent)
    vs = s._get_nullspace(k, dirichlet)
    for v in vs:
        b = b - jnp.dot(v, b) * s.apply_mass_matrix(v, k, dirichlet=dirichlet)

    rhs = jnp.concatenate([b, jnp.zeros(n_s)])

    # Direct dense solve (use pseudoinverse for singular systems)
    if has_nullspace:
        x_direct = jnp.linalg.lstsq(K_dense, rhs, rcond=None)[0]
    else:
        x_direct = jnp.linalg.solve(K_dense, rhs)
    u_direct, s_direct = x_direct[:n_u], x_direct[n_u:]

    # Verify direct solve residual
    res_direct = jnp.linalg.norm(
        K_dense @ x_direct - rhs) / jnp.linalg.norm(rhs)
    print(f"  Direct solve residual: {res_direct:.2e}")

    # MINRES solve via apply_inverse_hodge_laplacian (uses CG-preconditioned MINRES)
    u_mr = s.apply_inverse_hodge_laplacian(b, k, dirichlet=dirichlet)

    # Also get sigma from a direct saddle-point call for full comparison
    from jax.scipy.sparse.linalg import cg as jax_cg
    stiffness_diaginv = getattr(s, f"dd{k}_sp_diaginv{suffix}")
    mass_lower_diaginv = getattr(s, f"m{k-1}_sp_diaginv{suffix}")
    vs_upper, vs_lower = s._get_saddle_point_nullspaces(k, dirichlet)

    n_inner = 10

    def precond_lower(x):
        return jax_cg(
            lambda y: s.apply_mass_matrix(y, k-1, dirichlet=dirichlet),
            x, x0=jnp.zeros_like(x),
            M=lambda y: mass_lower_diaginv * y,
            maxiter=n_inner)[0]

    def approx_schur_matvec(x):
        Dt_x = s.apply_derivative_matrix(x, k-1, dirichlet_in=dirichlet,
                                         dirichlet_out=dirichlet, transpose=True)
        D_Minv_Dt_x = s.apply_derivative_matrix(
            mass_lower_diaginv * Dt_x, k-1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet)
        return s.apply_stiffness(x, k, dirichlet=dirichlet) + D_Minv_Dt_x

    def precond_upper(x):
        return jax_cg(approx_schur_matvec, x, x0=jnp.zeros_like(x),
                      M=lambda y: stiffness_diaginv * y,
                      maxiter=n_inner)[0]

    _, s_mr, info = solve_saddle_point_minres(
        stiffness_matvec=lambda x: s.apply_stiffness(
            x, k, dirichlet=dirichlet),
        derivative_matvec=lambda sig: s.apply_derivative_matrix(
            sig, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: s.apply_derivative_matrix(
            u, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda sig: s.apply_mass_matrix(
            sig, k-1, dirichlet=dirichlet),
        b_upper=b,
        n_upper=n_u,
        n_lower=n_s,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        mass_upper_matvec=lambda x: s.apply_mass_matrix(
            x, k, dirichlet=dirichlet),
        vs_upper=vs_upper,
        vs_lower=vs_lower,
        tol=1e-10, maxiter=5000,
    )

    # Compare against direct solve
    u_diff = u_mr - u_direct
    s_diff = s_mr - s_direct
    # Project out nullspace component (arbitrary for singular systems)
    for v in vs_upper:
        u_diff = u_diff - \
            jnp.dot(v, s.apply_mass_matrix(u_diff, k, dirichlet=dirichlet)) * v
    u_err = jnp.linalg.norm(u_diff) / jnp.linalg.norm(u_direct)
    s_err = jnp.linalg.norm(s_diff) / jnp.linalg.norm(
        s_direct) if jnp.linalg.norm(s_direct) > 0 else jnp.linalg.norm(s_diff)
    sp_res = jnp.linalg.norm(
        K_dense @ jnp.concatenate([u_mr, s_mr]) - rhs) / jnp.linalg.norm(rhs)
    print(f"  MINRES info={info}")
    print(f"  MINRES vs direct: u_err={u_err:.2e}, s_err={s_err:.2e}")
    print(f"  MINRES saddle-pt residual: {sp_res:.2e}")


for k in [1, 2, 3]:
    for dirichlet in [True, False]:
        test_k(k, dirichlet)
        test_k(k, dirichlet)
