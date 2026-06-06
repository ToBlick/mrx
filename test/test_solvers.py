"""Tests for ``mrx.solvers``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from mrx.solvers import (
    minres,
    picard_solver,
    preconditioned_cg,
    solve_saddle_point_minres,
    solve_singular_cg,
)

# ---------------------------------------------------------------------------
# Shared random test matrices (built once at module level)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(17)
_N = 24   # small enough that exact solves via np.linalg.solve are cheap


def _spd_matrix(n, rng, cond=10.0):
    """Random SPD matrix with bounded condition number."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.exp(np.linspace(0, np.log(cond), n))
    return (Q * lam) @ Q.T


def _symmetric_indefinite(n, rng, shift=2.0):
    """SPD matrix shifted to have a few negative eigenvalues."""
    A = _spd_matrix(n, rng, cond=8.0)
    return A - shift * np.eye(n)


_A = _spd_matrix(_N, _RNG)
_b = jnp.asarray(_RNG.standard_normal(_N))
_x_exact = jnp.asarray(np.linalg.solve(_A, np.asarray(_b)))

_A_jax = jnp.asarray(_A)
_A_matvec = lambda x: _A_jax @ x

# Jacobi preconditioner for _A
_M_inv_diag = jnp.asarray(1.0 / np.diag(_A))
_M_inv = lambda x: _M_inv_diag * x

# ---------------------------------------------------------------------------
# preconditioned_cg
# ---------------------------------------------------------------------------

def test_pcg_converges_spd():
    """CG converges on an SPD system; residual below tolerance."""
    tol = 1e-8
    x, info = preconditioned_cg(_A_matvec, _b, tol=tol, maxiter=10 * _N)
    res = float(jnp.linalg.norm(_A_jax @ x - _b))
    bnorm = float(jnp.linalg.norm(_b))
    assert res < tol * bnorm * 10, f"CG residual {res:.3e} too large"


# ---------------------------------------------------------------------------
# solve_singular_cg
# ---------------------------------------------------------------------------

def test_singular_cg_spsd_with_nullspace():
    """solve_singular_cg projects out the nullspace and recovers the
    minimum-norm solution of a rank-deficient SPSD system."""
    rng = np.random.default_rng(99)
    n = 20
    # Build rank-(n-1) SPSD matrix: A = V S V^T with one zero eigenvalue.
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.concatenate([np.zeros(1), np.exp(np.linspace(0, 2, n - 1))])
    A = (Q * lam) @ Q.T
    v0 = jnp.asarray(Q[:, 0])  # nullspace vector (mass-normalised: M=I here)

    b = jnp.asarray(rng.standard_normal(n))
    # project b onto the range of A
    b_proj = b - float(jnp.dot(v0, b)) * v0

    A_jax = jnp.asarray(A)
    x, info = solve_singular_cg(
        lambda x: A_jax @ x, b_proj, vs=[v0], tol=1e-8, maxiter=500)
    res = float(jnp.linalg.norm(A_jax @ x - b_proj))
    assert res < 1e-5, f"singular CG residual {res:.3e}"
    # Solution should have no nullspace component
    null_component = float(jnp.abs(jnp.dot(v0, x)))
    assert null_component < 1e-6, f"nullspace component {null_component:.3e}"


# ---------------------------------------------------------------------------
# minres
# ---------------------------------------------------------------------------

def test_minres_converges_spd():
    """MINRES converges on an SPD system."""
    tol = 1e-8
    x, info = minres(_A_matvec, _b, tol=tol, maxiter=10 * _N)
    res = float(jnp.linalg.norm(_A_jax @ x - _b))
    bnorm = float(jnp.linalg.norm(_b))
    assert res < tol * bnorm * 10, f"MINRES residual {res:.3e} too large"


def test_minres_symmetric_indefinite():
    """MINRES converges on a symmetric indefinite system."""
    rng = np.random.default_rng(42)
    n = 20
    A_indef = jnp.asarray(_symmetric_indefinite(n, rng, shift=1.5))
    b = jnp.asarray(rng.standard_normal(n))
    x_exact = jnp.asarray(np.linalg.solve(np.asarray(A_indef), np.asarray(b)))
    tol = 1e-7
    x, info = minres(lambda v: A_indef @ v, b, tol=tol, maxiter=20 * n)
    res = float(jnp.linalg.norm(A_indef @ x - b))
    bnorm = float(jnp.linalg.norm(b))
    assert res < tol * bnorm * 20, f"indefinite MINRES residual {res:.3e}"


# ---------------------------------------------------------------------------
# solve_saddle_point_minres
# ---------------------------------------------------------------------------

def test_saddle_point_minres_converges():
    """solve_saddle_point_minres solves a random 2-block saddle-point system.

    System:  | S    D  | | u |   | f |
             | D^T -M  | | s | = | 0 |

    where S, M are SPD and D is full-column-rank.  The exact solution is
    obtained by np.linalg.solve on the assembled block matrix.
    """
    rng = np.random.default_rng(55)
    nu, ns = 16, 8

    S = _spd_matrix(nu, rng, cond=5.0)
    M = _spd_matrix(ns, rng, cond=4.0)
    D = rng.standard_normal((nu, ns)) * 0.1  # keep coupling small

    f = rng.standard_normal(nu)

    # Exact solution via dense solve
    block = np.block([[S, D], [D.T, -M]])
    rhs = np.concatenate([f, np.zeros(ns)])
    z_exact = np.linalg.solve(block, rhs)
    u_exact, s_exact = z_exact[:nu], z_exact[nu:]

    S_jax = jnp.asarray(S)
    M_jax = jnp.asarray(M)
    D_jax = jnp.asarray(D)
    f_jax = jnp.asarray(f)

    M_inv_diag = jnp.asarray(1.0 / np.diag(M))

    u, s, info = solve_saddle_point_minres(
        stiffness_matvec=lambda v: S_jax @ v,
        derivative_matvec=lambda v: D_jax @ v,
        derivative_T_matvec=lambda v: D_jax.T @ v,
        mass_lower_matvec=lambda v: M_jax @ v,
        b_upper=f_jax,
        n_upper=nu,
        n_lower=ns,
        precond_lower=lambda v: M_inv_diag * v,
        tol=1e-8,
        maxiter=500,
    )

    npt.assert_allclose(np.asarray(u), u_exact, atol=1e-5,
                        err_msg="saddle-point u solution mismatch")
    npt.assert_allclose(np.asarray(s), s_exact, atol=1e-5,
                        err_msg="saddle-point s solution mismatch")


# ---------------------------------------------------------------------------
# picard_solver
# ---------------------------------------------------------------------------

def test_picard_scalar_contraction():
    """picard_solver converges for a scalar linear contraction x -> 0.5*x + c."""
    c = jnp.asarray(3.0)
    # Fixed point: x* = 2c = 6.0
    def f(z):
        x, aux = z
        return (0.5 * x + c, aux)

    z0 = (jnp.asarray(0.0), jnp.asarray(0.0))
    (x_star, _), res, iters = picard_solver(f, z0, tol=1e-10, max_iter=200)
    assert abs(float(x_star) - 6.0) < 1e-8, f"fixed point {float(x_star):.6f} != 6.0"
    assert res < 1e-9, f"residual {res:.3e}"