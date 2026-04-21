# test_derham_sequence_sparse.py
# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy.testing as npt
import pytest

from mrx.assembly import build_neighbors
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.solvers import solve_singular_cg
from mrx.utils import get_smallest_ev_pair

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

types = ("clamped", "periodic", "periodic")
# Betti numbers for a solid torus (rotating ellipse domain): [b0, b1, b2, b3]
BETTI = [1, 1, 0, 0]


@pytest.fixture(scope="module")
def seq_toroid():
    """DeRham sequence on a toroid for the Poisson test (p=2)."""
    p = 2
    n = 4
    ns = (n, n, n)
    ps = (p, p, p)
    a = 1 / 3
    F = toroid_map(epsilon=a)
    seq = DeRhamSequence(ns, ps, 2*p, types, F, polar=True,
                         tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    seq.assemble_mass_matrix(0)
    seq.assemble_hodge_laplacian(0)
    seq.null_0_dbc = []
    return seq


def _build_toroid_seq(n, p):
    """Build a toroid DeRham sequence at given resolution."""
    a = 1 / 3
    F = toroid_map(epsilon=a)
    seq = DeRhamSequence((n, n, n), (p, p, p), 2*p, types, F,
                         polar=True, tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    return seq


@pytest.fixture(
    scope="module",
    params=[(p, d) for p in [1, 2, 3] for d in [False, True]],
    ids=[f"p{p}_{'dbc' if d else 'no_dbc'}" for p in [1, 2, 3]
         for d in [False, True]],
)
def seq_ellipse(request):
    """DeRham sequence on a rotating-ellipse for sparse-vs-dense tests.

    Parametrised over polynomial degree p ∈ {1, 2, 3} and dirichlet ∈ {False, True}.
    Returns a ``(seq, dirichlet)`` pair.
    """
    p, dirichlet = request.param
    n = 4
    ns = (n, n, n)
    ps = (p, p, p)
    F = rotating_ellipse_map(nfp=3)
    seq = DeRhamSequence(ns, ps, 2*p, types, F, polar=True,
                         tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    seq.assemble_all_sparse()
    seq._compute_nullspaces(BETTI)
    return seq, dirichlet


# ---------------------------------------------------------------------------
# 1.  Analytical Poisson test (sparse solver only)
# ---------------------------------------------------------------------------

class TestPoissonToroidSparse:
    """Solve -Δu = f on a toroid using the sparse Hodge-Laplacian and check
    the L2 error against the known analytical solution."""

    @staticmethod
    def _exact_u(x):
        r, χ, z = x
        π = jnp.pi
        return 1 / 4 * (r ** 2 - r ** 4) * jnp.cos(2 * π * z) * jnp.ones(1)

    @staticmethod
    def _source_f(x):
        r, χ, z = x
        π = jnp.pi
        a = 1 / 3
        R = 1 + a * r * jnp.cos(2 * π * χ)
        return (
            jnp.cos(2 * π * z)
            * (
                -1 / a ** 2 * (1 - 4 * r ** 2)
                - 1 / (a * R) * (r / 2 - r ** 3) * jnp.cos(2 * π * χ)
                + 1 / 4 * (r ** 2 - r ** 4) / R ** 2
            )
            * jnp.ones(1)
        )

    @staticmethod
    def _solve_and_error_k0(seq):
        """Solve k=0 Poisson and return relative L2 error."""
        exact_u = TestPoissonToroidSparse._exact_u
        source_f = TestPoissonToroidSparse._source_f

        rhs = seq.p0_dbc(source_f)
        u_hat = seq.apply_inverse_hodge_laplacian(rhs, k=0)
        u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0_dbc)

        diff_vals = jax.lax.map(
            lambda x: exact_u(x) - u_h(x), seq.quad.x, batch_size=20_000
        )
        u_vals = jax.vmap(exact_u)(seq.quad.x)
        L2_diff = jnp.einsum("ik,ik,i,i->", diff_vals,
                             diff_vals, seq.jacobian_j, seq.quad.w)
        L2_u = jnp.einsum("ik,ik,i,i->", u_vals, u_vals,
                          seq.jacobian_j, seq.quad.w)
        return float((L2_diff / L2_u) ** 0.5)

    @staticmethod
    def _solve_and_error_k3(seq):
        """Solve k=3 Poisson (no BCs) and return relative L2 error.

        The k=3 Hodge Laplacian without BCs is equivalent to the k=0
        Laplacian with DBC, so we reuse the same analytic solution.
        The physical comparison uses the Pushforward (divides by det DF).
        """
        exact_u = TestPoissonToroidSparse._exact_u
        source_f = TestPoissonToroidSparse._source_f
        a = 1 / 3
        F = toroid_map(epsilon=a)

        rhs = seq.p3(source_f)
        p_hat = seq.apply_inverse_hodge_laplacian(rhs, k=3, dirichlet=False)

        # Compare in physical space via Pushforward
        p_h_ref = DiscreteFunction(p_hat, seq.basis_3, seq.e3)
        p_h_phys = Pushforward(p_h_ref, F, k=3)

        diff_vals = jax.lax.map(
            lambda x: exact_u(x) - p_h_phys(x), seq.quad.x,
            batch_size=20_000
        )
        u_vals = jax.vmap(exact_u)(seq.quad.x)
        L2_diff = jnp.einsum("ik,ik,i,i->", diff_vals,
                             diff_vals, seq.jacobian_j, seq.quad.w)
        L2_u = jnp.einsum("ik,ik,i,i->", u_vals, u_vals,
                          seq.jacobian_j, seq.quad.w)
        return float((L2_diff / L2_u) ** 0.5)

    def test_poisson_k0(self, seq_toroid):
        """k=0 Poisson solve should achieve < 10% relative error at n=4, p=2."""
        rel_error = self._solve_and_error_k0(seq_toroid)
        assert rel_error < 1e-1, f"Relative L2 error too large: {rel_error:.2e}"

    def test_poisson_k0_convergence(self):
        """k=0 error must decrease when resolution increases (n=4 → n=6)."""
        errors = []
        for n in (4, 6):
            seq = _build_toroid_seq(n, p=2)
            seq.assemble_mass_matrix(0)
            seq.assemble_hodge_laplacian(0)
            seq.null_0_dbc = []
            errors.append(self._solve_and_error_k0(seq))
        assert errors[1] < errors[0], (
            f"k=0 error did not decrease: n=4 → {errors[0]:.4e}, "
            f"n=6 → {errors[1]:.4e}")

    def test_poisson_k3(self):
        """k=3 Poisson (no BCs) should converge — same solution as k=0 with DBC."""
        errors = []
        for n in (4, 6):
            seq = _build_toroid_seq(n, p=2)
            seq.assemble_mass_matrix(2)
            seq.assemble_mass_matrix(3)
            seq.assemble_derivative_matrix(2)
            seq.assemble_hodge_laplacian(3)
            seq.null_3 = []
            errors.append(self._solve_and_error_k3(seq))

        assert errors[0] < 5e-1, (
            f"k=3 error too large at n=4: {errors[0]:.2e}")
        assert errors[1] < errors[0], (
            f"k=3 error did not decrease: n=4 → {errors[0]:.4e}, "
            f"n=6 → {errors[1]:.4e}")


# ---------------------------------------------------------------------------
# 2.  Sparse Poisson solves with random right-hand sides
# ---------------------------------------------------------------------------
# %%

class TestSequenceProperty:
    """Check if curl grad and div grad are identically zero for the sparse operators"""

    def test_curl_grad_zero(self, seq_ellipse):
        seq, _ = seq_ellipse
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (seq.n0,))
        grad_x = seq.apply_strong_grad(
            x, dirichlet_in=False, dirichlet_out=False)
        curl_grad_x = seq.apply_strong_curl(
            grad_x, dirichlet_in=False, dirichlet_out=False)
        npt.assert_allclose(
            curl_grad_x @ seq.apply_mass_matrix(curl_grad_x, k=2, dirichlet=False), 0, atol=1e-12)

    def test_div_grad_zero(self, seq_ellipse):
        seq, _ = seq_ellipse
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (seq.n1,))
        curl_x = seq.apply_strong_curl(
            x, dirichlet_in=False, dirichlet_out=False)
        div_curl_x = seq.apply_strong_div(
            curl_x, dirichlet_in=False, dirichlet_out=False)
        npt.assert_allclose(
            div_curl_x @ seq.apply_mass_matrix(div_curl_x, k=3, dirichlet=False), 0, atol=1e-12)


class TestPoissonSparse:
    """For each k = 0, 1, 2, 3 build a random RHS, solve the Hodge-Laplacian
    Poisson problem with the dense solver and the sparse CG solver, and compare
    the solutions."""

    def test_harmonic_fields(self, seq_ellipse):
        """Find the zero eigenvector of the k=2 Hodge-Laplacian."""
        seq, dirichlet = seq_ellipse

        if dirichlet:
            v2 = seq.null_2_dbc[0]
            v3 = seq.null_3_dbc[0]

            div_v2 = seq.apply_derivative_matrix(
                v2, k=2, dirichlet_in=True, dirichlet_out=True)
            curl_v2 = seq.apply_derivative_matrix(
                v2, k=1, dirichlet_in=True, dirichlet_out=True, transpose=True)
            grad_v3 = seq.apply_derivative_matrix(
                v3, k=2, dirichlet_in=True, dirichlet_out=True, transpose=True)
            assert jnp.max(jnp.abs(curl_v2)) < 1e-9, (
                f"||curl v2|| = {jnp.linalg.norm(curl_v2):.2e}, expected ≈ 0"
            )
            assert jnp.max(jnp.abs(div_v2)) < 1e-9, (
                f"||div v2|| = {jnp.linalg.norm(div_v2):.2e}, expected ≈ 0"
            )
            assert jnp.max(jnp.abs(grad_v3)) < 1e-9, (
                f"||grad v3|| = {jnp.linalg.norm(grad_v3):.2e}, expected ≈ 0"
            )
        else:
            v0 = seq.null_0[0]
            v1 = seq.null_1[0]
            grad_v0 = seq.apply_derivative_matrix(
                v0, k=0, dirichlet_in=False, dirichlet_out=False)
            curl_v1 = seq.apply_derivative_matrix(
                v1, k=1, dirichlet_in=False, dirichlet_out=False)
            div_v1 = seq.apply_derivative_matrix(
                v1, k=0, dirichlet_in=False, dirichlet_out=False, transpose=True)
            assert jnp.max(jnp.abs(grad_v0)) < 1e-9, (
                f"||grad v0|| = {jnp.linalg.norm(grad_v0):.2e}, expected ≈ 0"
            )
            assert jnp.max(jnp.abs(curl_v1)) < 1e-9, (
                f"||curl v1|| = {jnp.linalg.norm(curl_v1):.2e}, expected ≈ 0"
            )
            assert jnp.max(jnp.abs(div_v1)) < 1e-9, (
                f"||div v1|| = {jnp.linalg.norm(div_v1):.2e}, expected ≈ 0"
            )

    def test_poisson_k0(self, seq_ellipse):
        seq, dirichlet = seq_ellipse
        n = seq.n0_dbc if dirichlet else seq.n0
        null = seq.null_0_dbc if dirichlet else seq.null_0
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (n,))

        u_sparse, _ = solve_singular_cg(
            lambda x: seq.apply_hodge_laplacian(x, k=0, dirichlet=dirichlet),
            b,
            mass_matvec=lambda x: seq.apply_mass_matrix(
                x, k=0, dirichlet=dirichlet),
            precond_matvec=lambda x: seq.apply_hodge_laplacian_preconditioner(
                x, k=0, dirichlet=dirichlet),
            x0=jnp.zeros_like(b), vs=null, tol=seq.tol, maxiter=seq.maxiter
        )
        b_proj = b
        for v in null:
            b_proj = b - jnp.dot(v, b) * seq.apply_mass_matrix(v,
                                                               k=0, dirichlet=dirichlet)
            npt.assert_allclose(
                v @ seq.apply_mass_matrix(u_sparse, k=0, dirichlet=dirichlet), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_hodge_laplacian(
            u_sparse, k=0, dirichlet=dirichlet), b_proj, atol=1e-6)

    def test_poisson_k1(self, seq_ellipse):
        seq, dirichlet = seq_ellipse
        n = seq.n1_dbc if dirichlet else seq.n1
        null = seq.null_1_dbc if dirichlet else seq.null_1
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (n,))

        u_sparse, _ = solve_singular_cg(
            lambda x: seq.apply_hodge_laplacian(x, k=1, dirichlet=dirichlet),
            b,
            mass_matvec=lambda x: seq.apply_mass_matrix(
                x, k=1, dirichlet=dirichlet),
            precond_matvec=lambda x: seq.apply_hodge_laplacian_preconditioner(
                x, k=1, dirichlet=dirichlet),
            x0=jnp.zeros_like(b), vs=null, tol=seq.tol, maxiter=seq.maxiter
        )
        b_proj = b
        for v in null:
            b_proj = b - jnp.dot(v, b) * seq.apply_mass_matrix(v,
                                                               k=1, dirichlet=dirichlet)
            npt.assert_allclose(
                v @ seq.apply_mass_matrix(u_sparse, k=1, dirichlet=dirichlet), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_hodge_laplacian(
            u_sparse, k=1, dirichlet=dirichlet), b_proj, atol=1e-6)

    def test_poisson_k2(self, seq_ellipse):
        seq, dirichlet = seq_ellipse
        n = seq.n2_dbc if dirichlet else seq.n2
        null = seq.null_2_dbc if dirichlet else seq.null_2
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (n,))

        u_sparse, _ = solve_singular_cg(
            lambda x: seq.apply_hodge_laplacian(x, k=2, dirichlet=dirichlet),
            b,
            mass_matvec=lambda x: seq.apply_mass_matrix(
                x, k=2, dirichlet=dirichlet),
            precond_matvec=lambda x: seq.apply_hodge_laplacian_preconditioner(
                x, k=2, dirichlet=dirichlet),
            x0=jnp.zeros_like(b), vs=null, tol=seq.tol, maxiter=seq.maxiter
        )
        b_proj = b
        for v in null:
            b_proj = b - jnp.dot(v, b) * seq.apply_mass_matrix(v,
                                                               k=2, dirichlet=dirichlet)
            npt.assert_allclose(
                v @ seq.apply_mass_matrix(u_sparse, k=2, dirichlet=dirichlet), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_hodge_laplacian(
            u_sparse, k=2, dirichlet=dirichlet), b_proj, atol=1e-6)

    def test_poisson_k3(self, seq_ellipse):
        seq, dirichlet = seq_ellipse
        n = seq.n3_dbc if dirichlet else seq.n3
        null = seq.null_3_dbc if dirichlet else seq.null_3
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (n,))

        u_sparse, _ = solve_singular_cg(
            lambda x: seq.apply_hodge_laplacian(x, k=3, dirichlet=dirichlet),
            b,
            mass_matvec=lambda x: seq.apply_mass_matrix(
                x, k=3, dirichlet=dirichlet),
            precond_matvec=lambda x: seq.apply_hodge_laplacian_preconditioner(
                x, k=3, dirichlet=dirichlet),
            x0=jnp.zeros_like(b), vs=null, tol=seq.tol, maxiter=seq.maxiter
        )
        b_proj = b
        for v in null:
            b_proj = b - jnp.dot(v, b) * seq.apply_mass_matrix(v,
                                                               k=3, dirichlet=dirichlet)
            npt.assert_allclose(
                v @ seq.apply_mass_matrix(u_sparse, k=3, dirichlet=dirichlet), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_hodge_laplacian(
            u_sparse, k=3, dirichlet=dirichlet), b_proj, atol=1e-6)


class TestLerayProjection:
    def test_leray_k1(self, seq_ellipse):
        seq, _ = seq_ellipse
        key = jax.random.PRNGKey(1)
        # k=1 Leray projection uses dirichlet=False internally → n1 size
        b = jax.random.normal(key, (seq.n1,))

        v, p = seq.apply_leray_projection(b, k=1)
        div_v = seq.apply_weak_div(v, dirichlet_in=False, dirichlet_out=False)

        npt.assert_allclose(
            div_v @ seq.apply_mass_matrix(div_v, k=0, dirichlet=False), 0, atol=1e-12)

    def test_leray_k2(self, seq_ellipse):
        seq, _ = seq_ellipse
        key = jax.random.PRNGKey(2)
        # k=2 Leray projection uses dirichlet=True internally → n2_dbc size
        b = jax.random.normal(key, (seq.n2_dbc,))

        v, p = seq.apply_leray_projection(b, k=2)
        div_v = seq.apply_strong_div(v, dirichlet_in=True, dirichlet_out=True)

        npt.assert_allclose(
            div_v @ seq.apply_mass_matrix(div_v, k=3, dirichlet=True), 0, atol=1e-12)


# # ---------------------------------------------------------------------------
# # Sparse-vs-dense agreement for every operator
# # ---------------------------------------------------------------------------

# class TestSparseVsDense:
#     """Apply every operator with both the dense matrix and the sparse
#     callable to the same random vector and verify they agree.
#     All comparisons use dirichlet=False since the dense matrices are
#     assembled with the no-DBC extraction operators."""

#     # -- mass matrices -------------------------------------------------------

#     def test_m0(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(10), (seq.n0,))
#         dense = seq.m0 @ x
#         sparse = seq.apply_mass_matrix(x, k=0, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="m0 sparse != dense")

#     def test_m1(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(11), (seq.n1,))
#         dense = seq.m1 @ x
#         sparse = seq.apply_mass_matrix(x, k=1, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="m1 sparse != dense")

#     def test_m2(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(12), (seq.n2,))
#         dense = seq.m2 @ x
#         sparse = seq.apply_mass_matrix(x, k=2, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="m2 sparse != dense")

#     def test_m3(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(13), (seq.n3,))
#         dense = seq.m3 @ x
#         sparse = seq.apply_mass_matrix(x, k=3, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="m3 sparse != dense")

#     # -- differential operators d0, d1, d2 and transposes --------------------

#     def test_d0(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(20), (seq.n0,))
#         dense = seq.d0 @ x
#         sparse = seq.apply_derivative_matrix(
#             x, k=0, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="d0 sparse != dense")

#     def test_d0t(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(21), (seq.n1,))
#         dense = seq.d0.T @ x
#         sparse = seq.apply_derivative_matrix(
#             x, k=0, dirichlet_in=False, dirichlet_out=False, transpose=True)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="d0^T sparse != dense")

#     def test_d1(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(22), (seq.n1,))
#         dense = seq.d1 @ x
#         sparse = seq.apply_derivative_matrix(
#             x, k=1, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="d1 sparse != dense")

#     def test_d1t(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(23), (seq.n2,))
#         dense = seq.d1.T @ x
#         sparse = seq.apply_derivative_matrix(
#             x, k=1, dirichlet_in=False, dirichlet_out=False, transpose=True)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="d1^T sparse != dense")

#     def test_d2(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(24), (seq.n2,))
#         dense = seq.d2 @ x
#         sparse = seq.apply_derivative_matrix(
#             x, k=2, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="d2 sparse != dense")

#     def test_d2t(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(25), (seq.n3,))
#         dense = seq.d2.T @ x
#         sparse = seq.apply_derivative_matrix(
#             x, k=2, dirichlet_in=False, dirichlet_out=False, transpose=True)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="d2^T sparse != dense")

#     # -- Hodge-Laplacians dd0 … dd3 -----------------------------------------

#     def test_dd0(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(30), (seq.n0,))
#         dense = seq.m0 @ seq.dd0 @ x
#         sparse = seq.apply_hodge_laplacian(x, k=0, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="dd0 sparse != dense")

#     def test_dd1(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(31), (seq.n1,))
#         dense = seq.m1 @ seq.dd1 @ x
#         sparse = seq.apply_hodge_laplacian(x, k=1, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="dd1 sparse != dense")

#     def test_dd2(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(32), (seq.n2,))
#         dense = seq.m2 @ seq.dd2 @ x
#         sparse = seq.apply_hodge_laplacian(x, k=2, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="dd2 sparse != dense")

#     def test_dd3(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(33), (seq.n3,))
#         dense = seq.m3 @ seq.dd3 @ x
#         sparse = seq.apply_hodge_laplacian(x, k=3, dirichlet=False)
#         npt.assert_allclose(sparse, dense, atol=1e-12,
#                             err_msg="dd3 sparse != dense")

#     # -- strong grad / curl / div -------------------------------------------

#     def test_strong_grad(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(40), (seq.n0,))
#         dense = seq.strong_grad @ x
#         sparse = seq.apply_strong_grad(
#             x, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-9,
#                             err_msg="strong_grad sparse != dense")

#     def test_strong_curl(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(41), (seq.n1,))
#         dense = seq.strong_curl @ x
#         sparse = seq.apply_strong_curl(
#             x, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-9,
#                             err_msg="strong_curl sparse != dense")

#     def test_strong_div(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(42), (seq.n2,))
#         dense = seq.strong_div @ x
#         sparse = seq.apply_strong_div(
#             x, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-9,
#                             err_msg="strong_div sparse != dense")

#     # -- weak grad / curl / div ---------------------------------------------

#     def test_weak_grad(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(50), (seq.n3,))
#         dense = seq.weak_grad @ x
#         sparse = seq.apply_weak_grad(
#             x, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-9,
#                             err_msg="weak_grad sparse != dense")

#     def test_weak_curl(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(51), (seq.n2,))
#         dense = seq.weak_curl @ x
#         sparse = seq.apply_weak_curl(
#             x, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-9,
#                             err_msg="weak_curl sparse != dense")

#     def test_weak_div(self, seq_ellipse):
#         seq, _ = seq_ellipse
#         x = jax.random.normal(jax.random.PRNGKey(52), (seq.n1,))
#         dense = seq.weak_div @ x
#         sparse = seq.apply_weak_div(x, dirichlet_in=False, dirichlet_out=False)
#         npt.assert_allclose(sparse, dense, atol=1e-9,
#                             err_msg="weak_div sparse != dense")
