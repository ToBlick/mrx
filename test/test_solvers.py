"""The Krylov solvers stop on the norm they are given, measured on the residual they carry.

A dense symmetric system of size 60 (indefinite for MINRES) with an SPD preconditioner and an unrelated SPD norm,
in float64. MINRES and conjugate gradients carry the residual ``b - A x`` by a recurrence. The true residuals of
the first 45 iterates, in the given norm, are recorded by stopping the solver at each iteration. A tolerance just
above the residual of iterate ``it`` then stops the solver at the first iterate whose true residual is below it,
which is ``it`` or earlier (the residual need not fall monotonically in this norm).
With its Lanczos vectors reorthogonalised, MINRES on that system reaches the solution within ``N`` iterations up to
round-off.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.solvers import minres, preconditioned_cg

N = 60


def _system(indefinite):
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    w = rng.uniform(0.1, 50.0, N)
    if indefinite:
        w[:10] = -rng.uniform(0.5, 2.0, 10)
    B, C = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    A, P, G = Q @ np.diag(w) @ Q.T, np.linalg.inv(B @ B.T + N * np.eye(N)), C @ C.T + N * np.eye(N)
    A, P, G, b = (jnp.asarray(v, dtype=jnp.float64) for v in (A, P, G, rng.standard_normal(N)))
    return (lambda v: A @ v), (lambda v: P @ v), (lambda r: jnp.sqrt(r @ (G @ r))), b


@pytest.mark.parametrize("solver", ["minres", "cg"])
def test_the_solver_stops_on_the_given_norm_of_the_true_residual(solver):
    A, P, norm, b = _system(indefinite=solver == "minres")

    def run(tol, maxiter):
        if solver == "minres":
            return minres(A, b, M=P, tol=tol, maxiter=maxiter, norm=norm)
        return preconditioned_cg(A, b, P, tol, maxiter, norm=norm)

    true = [float(norm(b - A(run(0.0, it)[0])) / norm(b)) for it in range(1, 46)]
    for it in (1, 3, 7, 20, 45):
        tol = true[it - 1] * (1 + 1e-9)
        first = next(k for k, r in enumerate(true, start=1) if r < tol)
        assert int(run(tol, 10 * N)[1]) == first


def test_reorthogonalised_minres_terminates():
    """With its Lanczos vectors reorthogonalised, MINRES on a system of size ``N`` reaches the solution within ``N``
    iterations up to round-off, which the three-term recurrence alone does not on a spectrum over several decades.
    Spectrum 1e-2 .. 1e2 with 10 negative eigenvalues, SPD preconditioner. True relative residual measured
    2026-10-14: 1.6e-12 reorthogonalised, 0.24 with the three-term recurrence alone."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    w = np.logspace(-2, 2, N)
    w[:10] = -w[:10]
    B = rng.standard_normal((N, N))
    A, P = (jnp.asarray(v, dtype=jnp.float64) for v in (Q @ np.diag(w) @ Q.T, np.linalg.inv(B @ B.T + N * np.eye(N))))
    b = jnp.asarray(rng.standard_normal(N), dtype=jnp.float64)

    def residual(reorthogonalize):
        x, _ = minres(lambda v: A @ v, b, M=lambda v: P @ v, tol=0.0, maxiter=N,
                      norm=lambda r: jnp.sqrt(r @ (P @ r)), reorthogonalize=reorthogonalize)
        return float(jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b))

    assert residual(True) < 1e-10
    assert residual(False) > 1e-2
