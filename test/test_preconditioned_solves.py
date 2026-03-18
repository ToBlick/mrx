"""Tests for preconditioned mass and Hodge-Laplace solves.

Mass solves use CG with Jacobi preconditioner.
Hodge-Laplace solves use MINRES on the full saddle-point system (k>=1)
with CG-based block preconditioner. Verified against dense direct solves.
"""

import time

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.solvers import (preconditioned_cg, solve_saddle_point_minres,
                         solve_singular_cg)

jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")
N_TIMING_SOLVES = 5


@pytest.fixture(scope="module")
def seq():
    """DeRham sequence on a torus, fully assembled with nullspaces."""
    n, p = 5, 2
    F = toroid_map(epsilon=1 / 3)
    s = DeRhamSequence((n, n, n), (p, p, p), 2 * p, types, F,
                       polar=True, tol=1e-12, maxiter=2000)
    s.evaluate_1d()
    for k in range(4):
        s.assemble_mass_matrix(k)
    for k in range(3):
        s.assemble_derivative_matrix(k)
    for k in range(4):
        s.assemble_hodge_laplacian(k)
    s.compute_nullspaces()
    return s


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _random_rhs(key, n):
    """Random RHS vector."""
    return jax.random.normal(key, (n,))


def _get_nullspace(seq, k, dirichlet):
    if dirichlet:
        return getattr(seq, f"null_{k}_dbc")
    else:
        return getattr(seq, f"null_{k}")


def _ndofs(seq, k, dirichlet):
    suffix = "_dbc" if dirichlet else ""
    return getattr(seq, f"n{k}{suffix}")


# --------------------------------------------------------------------------
# Mass matrix solves
# --------------------------------------------------------------------------

class TestMassSolves:
    """Solve M_k x = b with and without preconditioner for all k, both BCs."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_mass_solve_preconditioned_correct(self, seq, k, dirichlet):
        """Preconditioned solve gives correct result: M_k x ≈ b."""
        n = _ndofs(seq, k, dirichlet)
        b = _random_rhs(jax.random.PRNGKey(k + 10 * dirichlet), n)

        x = seq.apply_inverse_mass_matrix(b, k, dirichlet=dirichlet)
        Mx = seq.apply_mass_matrix(x, k, dirichlet=dirichlet)
        npt.assert_allclose(Mx, b, atol=1e-8)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_mass_solve_unpreconditioned_correct(self, seq, k, dirichlet):
        """Unpreconditioned solve also gives correct result."""
        n = _ndofs(seq, k, dirichlet)
        b = _random_rhs(jax.random.PRNGKey(k + 10 * dirichlet), n)

        x, _ = solve_singular_cg(
            lambda x: seq.apply_mass_matrix(x, k, dirichlet=dirichlet),
            b,
            tol=1e-10, maxiter=1000,
        )
        Mx = seq.apply_mass_matrix(x, k, dirichlet=dirichlet)
        npt.assert_allclose(Mx, b, atol=1e-6)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_mass_solve_precond_matches_unprecond(self, seq, k, dirichlet):
        """Preconditioned and unpreconditioned solves agree."""
        n = _ndofs(seq, k, dirichlet)
        b = _random_rhs(jax.random.PRNGKey(k + 10 * dirichlet), n)

        x_precond = seq.apply_inverse_mass_matrix(
            b, k, dirichlet=dirichlet)

        x_plain, _ = solve_singular_cg(
            lambda x: seq.apply_mass_matrix(x, k, dirichlet=dirichlet),
            b,
            tol=1e-10, maxiter=1000,
        )
        npt.assert_allclose(x_precond, x_plain, atol=1e-4)


# --------------------------------------------------------------------------
# Hodge-Laplace solves
# --------------------------------------------------------------------------

class TestHodgeLaplaceSolves:
    """Solve L_k x = b, verify L x ≈ b."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_laplace_round_trip(self, seq, k, dirichlet):
        """Random b → x = L^{-1} b → L x ≈ b (up to nullspace)."""
        n = _ndofs(seq, k, dirichlet)
        vs = _get_nullspace(seq, k, dirichlet)
        b = _random_rhs(jax.random.PRNGKey(k + 100 * dirichlet), n)

        # Project b out of the nullspace
        for v in vs:
            b = b - jnp.dot(v, b) * seq.apply_mass_matrix(
                v, k, dirichlet=dirichlet)

        x = seq.apply_inverse_hodge_laplacian(b, k, dirichlet=dirichlet)
        Lx = seq.apply_hodge_laplacian(x, k, dirichlet=dirichlet)

        # Project residual out of nullspace
        residual = Lx - b
        for v in vs:
            residual = residual - jnp.dot(v, residual) * \
                seq.apply_mass_matrix(v, k, dirichlet=dirichlet)
        npt.assert_allclose(residual, 0.0, atol=2e-8)


# --------------------------------------------------------------------------
# Saddle-point dense verification (MINRES vs direct solve)
# --------------------------------------------------------------------------

def _build_dense(matvec, n_in, n_out=None):
    """Build a dense matrix from a matvec by probing with unit vectors."""
    if n_out is None:
        n_out = n_in
    cols = []
    for i in range(n_in):
        e = jnp.zeros(n_in).at[i].set(1.0)
        cols.append(matvec(e))
    return jnp.column_stack(cols)


class TestSaddlePointDenseVerification:
    """For k>=1, verify MINRES saddle-point solve against dense direct solve.

    Builds the full dense saddle-point matrix K = [[S, D], [D^T, -M]],
    solves K x = [b, 0] directly, and checks that MINRES matches.
    """

    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_minres_matches_direct(self, seq, k, dirichlet):
        """MINRES solution matches dense direct solve to high accuracy."""
        suffix = "_dbc" if dirichlet else ""
        n_u = getattr(seq, f"n{k}{suffix}")
        n_s = getattr(seq, f"n{k-1}{suffix}")
        vs = seq._get_nullspace(k, dirichlet)

        # Build dense blocks
        S = _build_dense(
            lambda x: seq.apply_stiffness(x, k, dirichlet=dirichlet), n_u)
        D = _build_dense(
            lambda s: seq.apply_derivative_matrix(
                s, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
            n_s, n_u)
        DT = _build_dense(
            lambda u: seq.apply_derivative_matrix(
                u, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
                transpose=True),
            n_u, n_s)
        M = _build_dense(
            lambda s: seq.apply_mass_matrix(s, k-1, dirichlet=dirichlet), n_s)

        # Full saddle-point matrix
        K = jnp.block([[S, D], [DT, -M]])

        # Random RHS, projected out of nullspace
        b = _random_rhs(jax.random.PRNGKey(k + 100 * dirichlet + 42), n_u)
        for v in vs:
            b = b - jnp.dot(v, b) * seq.apply_mass_matrix(
                v, k, dirichlet=dirichlet)
        rhs = jnp.concatenate([b, jnp.zeros(n_s)])

        # Direct solve
        has_nullspace = len(vs) > 0
        if has_nullspace:
            x_direct = jnp.linalg.lstsq(K, rhs, rcond=None)[0]
        else:
            x_direct = jnp.linalg.solve(K, rhs)
        u_direct = x_direct[:n_u]
        s_direct = x_direct[n_u:]

        # MINRES solve
        u_mr = seq.apply_inverse_hodge_laplacian(b, k, dirichlet=dirichlet)

        # Compare (project out nullspace component before comparing)
        diff = u_mr - u_direct
        for v in vs:
            diff = diff - jnp.dot(v, seq.apply_mass_matrix(
                diff, k, dirichlet=dirichlet)) * v
        npt.assert_allclose(diff, 0.0, atol=1e-7)

    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_saddle_point_symmetry(self, seq, k, dirichlet):
        """Dense saddle-point matrix K is symmetric: K = K^T."""
        suffix = "_dbc" if dirichlet else ""
        n_u = getattr(seq, f"n{k}{suffix}")
        n_s = getattr(seq, f"n{k-1}{suffix}")

        S = _build_dense(
            lambda x: seq.apply_stiffness(x, k, dirichlet=dirichlet), n_u)
        D = _build_dense(
            lambda s: seq.apply_derivative_matrix(
                s, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
            n_s, n_u)
        DT = _build_dense(
            lambda u: seq.apply_derivative_matrix(
                u, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
                transpose=True),
            n_u, n_s)
        M = _build_dense(
            lambda s: seq.apply_mass_matrix(s, k-1, dirichlet=dirichlet), n_s)

        K = jnp.block([[S, D], [DT, -M]])
        npt.assert_allclose(K, K.T, atol=1e-12)

    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_saddle_point_residual(self, seq, k, dirichlet):
        """Saddle-point residual ||K x - rhs|| / ||rhs|| is small."""
        suffix = "_dbc" if dirichlet else ""
        n_u = getattr(seq, f"n{k}{suffix}")
        n_s = getattr(seq, f"n{k-1}{suffix}")
        vs = seq._get_nullspace(k, dirichlet)
        vs_upper, vs_lower = seq._get_saddle_point_nullspaces(k, dirichlet)

        # Build dense K for residual check
        S = _build_dense(
            lambda x: seq.apply_stiffness(x, k, dirichlet=dirichlet), n_u)
        D = _build_dense(
            lambda s: seq.apply_derivative_matrix(
                s, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
            n_s, n_u)
        DT = _build_dense(
            lambda u: seq.apply_derivative_matrix(
                u, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
                transpose=True),
            n_u, n_s)
        M = _build_dense(
            lambda s: seq.apply_mass_matrix(s, k-1, dirichlet=dirichlet), n_s)
        K = jnp.block([[S, D], [DT, -M]])

        b = _random_rhs(jax.random.PRNGKey(k + 100 * dirichlet + 99), n_u)
        for v in vs:
            b = b - jnp.dot(v, b) * seq.apply_mass_matrix(
                v, k, dirichlet=dirichlet)
        rhs = jnp.concatenate([b, jnp.zeros(n_s)])

        # Get full saddle-point solution (u and sigma)
        from jax.scipy.sparse.linalg import cg as jax_cg
        stiffness_diaginv = getattr(seq, f"dd{k}_sp_diaginv{suffix}")
        mass_lower_diaginv = getattr(seq, f"m{k-1}_sp_diaginv{suffix}")

        def precond_lower(x):
            return jax_cg(
                lambda y: seq.apply_mass_matrix(y, k-1, dirichlet=dirichlet),
                x, x0=jnp.zeros_like(x),
                M=lambda y: mass_lower_diaginv * y,
                maxiter=seq.n_inner)[0]

        def approx_schur_matvec(x):
            Dt_x = seq.apply_derivative_matrix(
                x, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
                transpose=True)
            D_Minv_Dt_x = seq.apply_derivative_matrix(
                mass_lower_diaginv * Dt_x, k-1,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet)
            return seq.apply_stiffness(x, k, dirichlet=dirichlet) + D_Minv_Dt_x

        def precond_upper(x):
            return jax_cg(
                approx_schur_matvec, x, x0=jnp.zeros_like(x),
                M=lambda y: stiffness_diaginv * y,
                maxiter=seq.n_inner)[0]

        u_mr, s_mr, info = solve_saddle_point_minres(
            stiffness_matvec=lambda x: seq.apply_stiffness(
                x, k, dirichlet=dirichlet),
            derivative_matvec=lambda sig: seq.apply_derivative_matrix(
                sig, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
            derivative_T_matvec=lambda u: seq.apply_derivative_matrix(
                u, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
                transpose=True),
            mass_lower_matvec=lambda sig: seq.apply_mass_matrix(
                sig, k-1, dirichlet=dirichlet),
            b_upper=b,
            n_upper=n_u,
            n_lower=n_s,
            precond_upper=precond_upper,
            precond_lower=precond_lower,
            mass_upper_matvec=lambda x: seq.apply_mass_matrix(
                x, k, dirichlet=dirichlet),
            vs_upper=vs_upper,
            vs_lower=vs_lower,
            tol=1e-10, maxiter=5000,
        )

        x_mr = jnp.concatenate([u_mr, s_mr])
        rel_res = jnp.linalg.norm(K @ x_mr - rhs) / jnp.linalg.norm(rhs)
        assert rel_res < 1e-7, f"k={k}, dbc={dirichlet}: rel residual {rel_res:.2e}"


# --------------------------------------------------------------------------
# Diffusion solves: (M_k + alpha * L_k) x = b
# --------------------------------------------------------------------------

class TestDiffusionSolves:
    """Solve (M_k + alpha * L_k) x = b, verify round-trip."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    @pytest.mark.parametrize("alpha", [1e-2, 1e-4])
    def test_diffusion_round_trip(self, seq, k, dirichlet, alpha):
        """Random b → x = (M + αL)^{-1} b → (M + αL) x ≈ b."""
        n = _ndofs(seq, k, dirichlet)
        b = _random_rhs(jax.random.PRNGKey(k + 100 * dirichlet + 7), n)

        x = seq.apply_inverse_diffusion(b, k, alpha, dirichlet=dirichlet)
        MLx = seq.apply_diffusion(x, k, alpha, dirichlet=dirichlet)
        npt.assert_allclose(MLx, b, atol=1e-8)


class TestDiffusionDenseVerification:
    """For k>=1, verify diffusion MINRES solve against dense direct solve."""

    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    @pytest.mark.parametrize("alpha", [1e-2, 1e-4])
    def test_diffusion_matches_direct(self, seq, k, dirichlet, alpha):
        """MINRES diffusion solution matches dense direct solve."""
        suffix = "_dbc" if dirichlet else ""
        n_u = getattr(seq, f"n{k}{suffix}")
        n_s = getattr(seq, f"n{k-1}{suffix}")

        # Build dense blocks
        Mk = _build_dense(
            lambda x: seq.apply_mass_matrix(x, k, dirichlet=dirichlet), n_u)
        S = _build_dense(
            lambda x: seq.apply_stiffness(x, k, dirichlet=dirichlet), n_u)
        D = _build_dense(
            lambda s: seq.apply_derivative_matrix(
                s, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
            n_s, n_u)
        DT = _build_dense(
            lambda u: seq.apply_derivative_matrix(
                u, k-1, dirichlet_in=dirichlet, dirichlet_out=dirichlet,
                transpose=True),
            n_u, n_s)
        Ml = _build_dense(
            lambda s: seq.apply_mass_matrix(s, k-1, dirichlet=dirichlet), n_s)

        # Full saddle-point matrix for diffusion
        K = jnp.block([
            [Mk + alpha * S, alpha * D],
            [alpha * DT,    -alpha * Ml],
        ])

        b = _random_rhs(jax.random.PRNGKey(k + 100 * dirichlet + 55), n_u)
        rhs = jnp.concatenate([b, jnp.zeros(n_s)])

        # Dense direct solve (system is nonsingular)
        x_direct = jnp.linalg.solve(K, rhs)
        u_direct = x_direct[:n_u]

        # MINRES solve
        u_mr = seq.apply_inverse_diffusion(b, k, alpha, dirichlet=dirichlet)

        npt.assert_allclose(u_mr, u_direct, atol=1e-7)


# --------------------------------------------------------------------------
# Timing: preconditioned vs unpreconditioned
# --------------------------------------------------------------------------

def _time_solves(solve_fn, bs):
    """JIT-compile, warm up, then time N_TIMING_SOLVES calls."""
    solve_jit = jax.jit(solve_fn)
    # Warm up (compile)
    x0 = solve_jit(bs[0])
    x0.block_until_ready()
    # Timed runs
    times = []
    for b in bs:
        t0 = time.perf_counter()
        x = solve_jit(b)
        x.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


class TestMassSolveTiming:
    """Preconditioned mass solves should not be slower than unpreconditioned."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_mass_precond_not_slower(self, seq, k, dirichlet):
        n = _ndofs(seq, k, dirichlet)
        keys = jax.random.split(jax.random.PRNGKey(k + 10 * dirichlet + 200),
                                N_TIMING_SOLVES)
        bs = [jax.random.normal(key, (n,)) for key in keys]

        def solve_precond(b):
            return solve_singular_cg(
                lambda x: seq.apply_mass_matrix(x, k, dirichlet=dirichlet),
                b,
                precond_matvec=lambda x: seq.apply_mass_matrix_preconditioner(
                    x, k, dirichlet=dirichlet),
                tol=1e-10, maxiter=1000,
            )[0]

        def solve_plain(b):
            return solve_singular_cg(
                lambda x: seq.apply_mass_matrix(x, k, dirichlet=dirichlet),
                b,
                tol=1e-10, maxiter=1000,
            )[0]

        t_precond = _time_solves(solve_precond, bs)
        t_plain = _time_solves(solve_plain, bs)

        dbc_str = "dbc" if dirichlet else "no_dbc"
        speedup = t_plain / t_precond if t_precond > 0 else float('inf')
        print(f"\n  Mass k={k} {dbc_str}: precond {t_precond*1e3:.2f}ms, "
              f"plain {t_plain*1e3:.2f}ms, speedup {speedup:.2f}x")

        # Allow 2x margin — we just want to catch catastrophic regressions
        assert t_precond < 2 * t_plain, \
            f"Mass k={k} dbc={dirichlet}: precond {t_precond*1e3:.1f}ms " \
            f"vs plain {t_plain*1e3:.1f}ms"


class TestHodgeLaplaceSolveTiming:
    """Preconditioned Hodge-Laplace solves should not be slower than unpreconditioned."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_laplace_precond_not_slower(self, seq, k, dirichlet):
        n = _ndofs(seq, k, dirichlet)
        vs = _get_nullspace(seq, k, dirichlet)
        keys = jax.random.split(jax.random.PRNGKey(k + 100 * dirichlet + 200),
                                N_TIMING_SOLVES)
        bs = [jax.random.normal(key, (n,)) for key in keys]

        def solve_precond(b):
            return solve_singular_cg(
                lambda x: seq.apply_hodge_laplacian(
                    x, k, dirichlet=dirichlet),
                b,
                mass_matvec=lambda x: seq.apply_mass_matrix(
                    x, k, dirichlet=dirichlet),
                precond_matvec=lambda x:
                    seq.apply_hodge_laplacian_preconditioner(
                        x, k, dirichlet=dirichlet),
                vs=vs,
                tol=1e-10, maxiter=1000,
            )[0]

        def solve_plain(b):
            return solve_singular_cg(
                lambda x: seq.apply_hodge_laplacian(
                    x, k, dirichlet=dirichlet),
                b,
                mass_matvec=lambda x: seq.apply_mass_matrix(
                    x, k, dirichlet=dirichlet),
                vs=vs,
                tol=1e-10, maxiter=1000,
            )[0]

        t_precond = _time_solves(solve_precond, bs)
        t_plain = _time_solves(solve_plain, bs)

        assert t_precond < 2 * t_plain, \
            f"Laplace k={k} dbc={dirichlet}: precond {t_precond*1e3:.1f}ms " \
            f"vs plain {t_plain*1e3:.1f}ms"


# --------------------------------------------------------------------------
# Custom CG vs JAX CG comparison
# --------------------------------------------------------------------------

class TestCGComparison:
    """Compare custom preconditioned_cg (M-norm stopping) vs JAX cg (2-norm)."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_cg_accuracy(self, seq, k, dirichlet):
        """Both CG implementations produce accurate mass solves."""
        from jax.scipy.sparse.linalg import cg as jax_cg

        n = _ndofs(seq, k, dirichlet)
        b = _random_rhs(jax.random.PRNGKey(k + 10 * dirichlet + 500), n)
        diaginv = seq.apply_mass_matrix_preconditioner(
            jnp.ones(n), k, dirichlet=dirichlet)

        def A(x): return seq.apply_mass_matrix(x, k, dirichlet=dirichlet)
        def M_fn(x): return diaginv * x

        x_custom, info_custom = preconditioned_cg(
            A, b, M=M_fn, tol=1e-12, maxiter=2000)
        x_jax, info_jax = jax_cg(
            A, b, M=M_fn, tol=1e-12, maxiter=2000)

        # Both should give accurate solutions
        res_custom = jnp.max(jnp.abs(A(x_custom) - b))
        res_jax = jnp.max(jnp.abs(A(x_jax) - b))
        print(f"\n  k={k} dbc={dirichlet}: custom |Ax-b|={res_custom:.2e} "
              f"(info={info_custom}), jax |Ax-b|={res_jax:.2e} "
              f"(info={info_jax})")
        npt.assert_allclose(A(x_custom), b, atol=1e-8)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_cg_solutions_agree(self, seq, k, dirichlet):
        """Custom CG and JAX CG produce the same solution."""
        from jax.scipy.sparse.linalg import cg as jax_cg

        n = _ndofs(seq, k, dirichlet)
        b = _random_rhs(jax.random.PRNGKey(k + 10 * dirichlet + 500), n)
        diaginv = seq.apply_mass_matrix_preconditioner(
            jnp.ones(n), k, dirichlet=dirichlet)

        def A(x): return seq.apply_mass_matrix(x, k, dirichlet=dirichlet)
        def M_fn(x): return diaginv * x

        x_custom, _ = preconditioned_cg(
            A, b, M=M_fn, tol=1e-12, maxiter=2000)
        x_jax, _ = jax_cg(
            A, b, M=M_fn, tol=1e-12, maxiter=2000)

        npt.assert_allclose(x_custom, x_jax, atol=1e-8)

    @pytest.mark.parametrize("k", [0, 1, 2])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_cg_speed(self, seq, k, dirichlet):
        """Custom CG should not be significantly slower than JAX CG."""
        from jax.scipy.sparse.linalg import cg as jax_cg

        n = _ndofs(seq, k, dirichlet)
        diaginv = seq.apply_mass_matrix_preconditioner(
            jnp.ones(n), k, dirichlet=dirichlet)

        def A(x): return seq.apply_mass_matrix(x, k, dirichlet=dirichlet)
        def M_fn(x): return diaginv * x

        keys = jax.random.split(
            jax.random.PRNGKey(k + 10 * dirichlet + 600), N_TIMING_SOLVES)
        bs = [jax.random.normal(key, (n,)) for key in keys]

        def solve_custom(b):
            return preconditioned_cg(
                A, b, M=M_fn, tol=1e-10, maxiter=1000)[0]

        def solve_jax(b):
            return jax_cg(A, b, M=M_fn, tol=1e-10, maxiter=1000)[0]

        t_custom = _time_solves(solve_custom, bs)
        t_jax = _time_solves(solve_jax, bs)

        dbc_str = "dbc" if dirichlet else "no_dbc"
        ratio = t_custom / t_jax if t_jax > 0 else float('inf')
        print(f"\n  CG k={k} {dbc_str}: custom {t_custom*1e3:.2f}ms, "
              f"jax {t_jax*1e3:.2f}ms, ratio {ratio:.2f}x")

        # Custom CG should not be more than 3x slower
        assert t_custom < 3 * t_jax, \
            f"CG k={k} dbc={dirichlet}: custom {t_custom*1e3:.1f}ms " \
            f"vs jax {t_jax*1e3:.1f}ms"
