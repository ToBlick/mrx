# test_derham_sequence_sparse.py
# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy.testing as npt
import pytest
from jax.scipy.sparse.linalg import cg
from scipy.linalg import eigvalsh

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.utils import get_smallest_ev_pair, solve_singular_cg

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

p = 3
n = 5
q = p
ns = (n, n, n)
ps = (p, p, p)
types = ("clamped", "periodic", "periodic")


@pytest.fixture(scope="module")
def seq_toroid():
    """DeRham sequence on a toroid for the Poisson test."""
    a = 1 / 3
    F = toroid_map(epsilon=a)
    seq = DeRhamSequence(ns, ps, q, types, F, polar=True,
                         dirichlet=True, tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    seq.assemble_m0()
    seq.assemble_dd0()
    seq.assemble_m0_sparse()
    seq.assemble_dd0_sparse()
    return seq


@pytest.fixture(scope="module", params=[False, True], ids=["no_dirichlet", "dirichlet"])
def seq_ellipse(request):
    """DeRham sequence on a rotating-ellipse for sparse-vs-dense tests.

    Parametrised over ``dirichlet=False`` and ``dirichlet=True`` so that
    ``TestSequenceProperty`` and ``TestPoissonSparse`` exercise both cases.
    """
    F = rotating_ellipse_map(nfp=3)
    seq = DeRhamSequence(ns, ps, q, types, F, polar=True,
                         dirichlet=request.param, tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    seq.assemble_all()
    seq.assemble_all_sparse()
    seq.compute_nullspaces()
    return seq


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

    def test_poisson_sparse_solver(self, seq_toroid):
        seq = seq_toroid

        # Project source
        rhs = seq.P0(self._source_f)

        # Sparse CG solve:  grad_grad @ u_hat = rhs
        u_hat_sp, _ = cg(
            seq.apply_dd0_sparse, rhs, tol=seq.tol, M=seq.apply_dd0_precond, x0=jnp.zeros_like(rhs), maxiter=seq.maxiter
        )

        u_h = DiscreteFunction(u_hat_sp, seq.basis_0, seq.e0)

        # L2 error
        diff_vals = jax.lax.map(
            lambda x: self._exact_u(x) - u_h(x), seq.quad.x, batch_size=20_000
        )
        u_vals = jax.vmap(self._exact_u)(seq.quad.x)
        L2_diff = jnp.einsum("ik,ik,i,i->", diff_vals,
                             diff_vals, seq.jacobian_j, seq.quad.w)
        L2_u = jnp.einsum("ik,ik,i,i->", u_vals, u_vals,
                          seq.jacobian_j, seq.quad.w)
        rel_error = (L2_diff / L2_u) ** 0.5

        assert rel_error < 5e-2, f"Relative L2 error too large: {rel_error:.2e}"


# ---------------------------------------------------------------------------
# 2.  Sparse-vs-dense Poisson solves with random right-hand sides
# ---------------------------------------------------------------------------
# %%

class TestSequenceProperty:
    """Check if curl grad and div grad are identically zero for the sparse operators"""

    def test_curl_grad_zero(self, seq_ellipse):
        seq = seq_ellipse
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (seq.n0,))
        grad_x = seq.apply_strong_grad(x)
        # grad_x = cg(seq.apply_m1_sparse, seq.apply_d0_sparse(
        # x), tol=seq.tol, maxiter=seq.maxiter)[0]
        curl_grad_x = seq.apply_strong_curl(grad_x)
        # curl_grad_x = cg(seq.apply_m2_sparse, seq.apply_d1_sparse(
        # grad_x), tol=seq.tol, maxiter=seq.maxiter)[0]
        npt.assert_allclose(
            curl_grad_x @ seq.apply_m2_sparse(curl_grad_x), 0, atol=1e-16)

    def test_div_grad_zero(self, seq_ellipse):
        seq = seq_ellipse
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (seq.n1,))
        curl_x = seq.apply_strong_curl(x)
        div_curl_x = seq.apply_strong_div(curl_x)
        npt.assert_allclose(
            div_curl_x @ seq.apply_m3_sparse(div_curl_x), 0, atol=1e-16)


class TestPoissonSparse:
    """For each k = 0, 1, 2, 3 build a random RHS, solve the Hodge-Laplacian
    Poisson problem with the dense solver and the sparse CG solver, and compare
    the solutions."""

    def test_harmonic_fields(self, seq_ellipse):
        """Find the zero eigenvector of the k=2 Hodge-Laplacian."""
        seq = seq_ellipse

        if seq.dirichlet:
            v2 = seq.null_2[0]
            v3 = seq.null_3[0]

            div_v2 = seq.apply_d2_sparse(v2)
            curl_v2 = seq.apply_d1t_sparse(v2)
            grad_v3 = seq.apply_d2t_sparse(v3)
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
            grad_v0 = seq.apply_d0_sparse(v0)
            curl_v1 = seq.apply_d1_sparse(v1)
            div_v1 = seq.apply_d0t_sparse(v1)
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
        seq = seq_ellipse
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (seq.n0,))

        u_sparse, _ = solve_singular_cg(
            seq.apply_dd0_sparse, seq.apply_m0_sparse, b, precond_matvec=seq.apply_dd0_precond,
            x0=jnp.zeros_like(b), vs=seq.null_0, tol=seq.tol, max_iters=seq.maxiter
        )
        b_proj = b
        for v in seq.null_0:
            b_proj = b - jnp.dot(v, b) * seq.apply_m0_sparse(v)
            npt.assert_allclose(
                v @ seq.apply_m0_sparse(u_sparse), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_dd0_sparse(u_sparse), b_proj, atol=1e-6)

    def test_poisson_k1(self, seq_ellipse):
        seq = seq_ellipse
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (seq.n1,))

        u_sparse, _ = solve_singular_cg(
            seq.apply_dd1_sparse, seq.apply_m1_sparse, b, precond_matvec=seq.apply_dd1_precond,
            x0=jnp.zeros_like(b), vs=seq.null_1, tol=seq.tol, max_iters=seq.maxiter
        )
        b_proj = b
        for v in seq.null_1:
            b_proj = b - jnp.dot(v, b) * seq.apply_m1_sparse(v)
            npt.assert_allclose(
                v @ seq.apply_m1_sparse(u_sparse), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_dd1_sparse(u_sparse), b_proj, atol=1e-6)

    def test_poisson_k2(self, seq_ellipse):
        seq = seq_ellipse
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (seq.n2,))

        u_sparse, _ = solve_singular_cg(
            seq.apply_dd2_sparse, seq.apply_m2_sparse, b, precond_matvec=seq.apply_dd2_precond,
            x0=jnp.zeros_like(b), vs=seq.null_2, tol=seq.tol, max_iters=seq.maxiter
        )
        # check that dd2 @ u_sparse = b_proj and that u_sparse is m2-orthogonal to the nullspace
        b_proj = b
        for v in seq.null_2:
            b_proj = b - jnp.dot(v, b) * seq.apply_m2_sparse(v)
            npt.assert_allclose(
                v @ seq.apply_m2_sparse(u_sparse), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_dd2_sparse(u_sparse), b_proj, atol=1e-6)

    def test_poisson_k3(self, seq_ellipse):
        seq = seq_ellipse
        key = jax.random.PRNGKey(0)
        b = jax.random.normal(key, (seq.n3,))

        u_sparse, _ = solve_singular_cg(
            seq.apply_dd3_sparse, seq.apply_m3_sparse, b, precond_matvec=seq.apply_dd3_precond,
            x0=jnp.zeros_like(b), vs=seq.null_3, tol=seq.tol, max_iters=seq.maxiter
        )
        # check that dd3 @ u_sparse = b_proj and that u_sparse is m3-orthogonal to the nullspace
        b_proj = b
        for v in seq.null_3:
            b_proj = b - jnp.dot(v, b) * seq.apply_m3_sparse(v)
            npt.assert_allclose(
                v @ seq.apply_m3_sparse(u_sparse), 0, atol=1e-12)

        npt.assert_allclose(seq.apply_dd3_sparse(u_sparse), b_proj, atol=1e-6)


@pytest.fixture(scope="module")
def seq_ellipse_no_dirichlet(seq_ellipse):
    """Always the dirichlet=False instance, for tests that require it."""
    if not seq_ellipse.dirichlet:
        pytest.skip("skipping no-BCs variant for Leray projection tests")
    return seq_ellipse


class TestLerayProjection:
    def test_leray_k1(self, seq_ellipse_no_dirichlet):
        seq_ellipse = seq_ellipse_no_dirichlet
        seq = seq_ellipse
        key = jax.random.PRNGKey(1)
        b = jax.random.normal(key, (seq.n1,))

        v, p = seq.apply_leray_projection(
            b, k=1)
        div_v = seq.apply_weak_div(v)

        npt.assert_allclose(div_v @ seq.apply_m0_sparse(div_v), 0, atol=1e-16)

    def test_leray_k2(self, seq_ellipse):
        seq = seq_ellipse
        key = jax.random.PRNGKey(2)
        b = jax.random.normal(key, (seq.n2,))

        v, p = seq.apply_leray_projection(
            b, k=2)

        div_v = seq.apply_strong_div(v)

        npt.assert_allclose(div_v @ seq.apply_m3_sparse(div_v), 0, atol=1e-16)
