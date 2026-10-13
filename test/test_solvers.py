"""Every solve of a sequence inverts its operator, and the Krylov solvers stop on the norm they are given.

Round trip: a known solution ``x`` is mapped to its right-hand side by the operator's apply, solved, and compared
with ``x`` in the L2 norm of its degree. The cases are every solve of :mod:`mrx.operators` and the Leray projection:

- ``M[k].solve``, k = 0..3.
- ``L[k].solve`` with ``part="full"`` (k = 0..3), ``"coexact"`` (k = 0..2) and ``"exact"`` (k = 1..3).
- ``S[k].solve``, which recovers ``G_k x`` only (k = 1, 2).
- ``shifted(k, eps).solve`` (k = 0..3) and ``shifted_stiffness(k, eps).solve`` (k = 1, 2), which recovers
  ``G_k x`` for any right-hand side, at the ``eps`` of the velocity smoothing. The 1-form solves also at
  ``eps = 1``, where the two-solve split of PR #33 did not converge.
- ``leray(v, k)`` for k = 1, 2: the divergence-free part of ``v`` plus a gradient is that part.

A co-exact solution is ``x_c = M_k^{-1} G_k^T w`` and an exact one ``x_e = G_{k-1} c``. Both are orthogonal to
the harmonic forms, and the full solve recovers their sum. The right-hand side of a co-exact part solve is
``S_k x_c``, which is orthogonal to the closed forms without a mass solve. Every right-hand side is the apply of the
lifted solution, ``A @ lift(x)``, so in float32 it is formed in double-single and lies in the range of its operator
to that accuracy. The shifted stiffness solve of
``(M_k + eps L_k) x`` differs from ``x`` by a closed form, so it is compared after ``G_k``. The cases run on the production
spaces: the Dirichlet odd view of li383 (``B``, ``A``, ``J``), with its harmonic 2-form, and the even view for
the Leray projection (``u``, ``F``). The free spaces are solved by ``test_poisson.py``.

Krylov stopping: a dense symmetric system of size 60 (indefinite for MINRES) with an SPD preconditioner and an
unrelated SPD norm, in float64. MINRES and conjugate gradients carry the residual ``b - A x`` by a recurrence.
The true residuals of the first 45 iterates, in the given norm, are recorded by stopping the solver at each
iteration. A tolerance just above the residual of iterate ``it`` then stops the solver at the first iterate whose
true residual is below it, which is ``it`` or earlier (the residual need not fall monotonically in this norm).
With its Lanczos vectors reorthogonalised, MINRES on that system reaches the solution within ``N`` iterations up to
round-off.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.precision import lift
from mrx.relaxation.loop import SMOOTHING_C, radial_cell_sq
from mrx.solvers import minres, preconditioned_cg

#: The relative error of a round trip, in units of seq.tol. Measured on the graded li383 2026-10-19: at most 9.8
#: (L[1] coexact), except the exact parts, 55 to 617 (L[3] exact), whose two solves of degree k - 1 amplify the
#: solve error.
ROUND_TRIP = 1e3
# the Krylov stopping tests are statements about the recurrences at float64 round-off
FLOAT64 = pytest.mark.skipif(mrx.DTYPE != jnp.float64, reason="needs MRX_DTYPE=float64")
N = 60


def _deflate(space, x, k):
    """``x`` without its part along the harmonic k-forms of ``space``."""
    for h in space.nullspace(k):
        x = x - (h @ (space.M[k] @ x)) / (h @ (space.M[k] @ h)) * h
    return x


def test_every_solve_inverts_its_operator(seq):
    """Each solve recovers the solution its right-hand side was made from, to a multiple of ``seq.tol``."""
    rng = np.random.default_rng(0)
    odd, even = seq.odd, seq.even
    smoothing = SMOOTHING_C * radial_cell_sq(seq)

    def rand(space, k):
        return jnp.asarray(rng.standard_normal(space.n(k)))

    x_c = {k: odd.M[k].solve(odd.G[k].T @ rand(odd, k + 1)) for k in (1, 2)}
    x_c[0] = _deflate(odd, rand(odd, 0), 0)
    x_e = {k: odd.G[k - 1] @ rand(odd, k - 1) for k in (1, 2, 3)}
    x_full = {0: x_c[0], 1: x_c[1] + x_e[1], 2: x_c[2] + x_e[2], 3: _deflate(odd, rand(odd, 3), 3)}

    cases = []          # (name, space, degree of the compared form, solution, recovered, info or None)

    def add(name, space, k, want, got_info, apply=None):
        got, info = got_info
        if apply is not None:
            want, got = apply @ want, apply @ got
        cases.append((name, space, k, want, got, info))

    for k in range(4):
        x = rand(odd, k)
        add(f"M[{k}]", odd, k, x, odd.M[k].solve(odd.M[k] @ lift(x), return_info=True))
    for k in range(4):
        add(f"L[{k}] full", odd, k, x_full[k], odd.L[k].solve(odd.L[k] @ lift(x_full[k]), return_info=True))
    for k in (0, 1, 2):
        add(f"L[{k}] coexact", odd, k, x_c[k],
            odd.L[k].solve(odd.S[k] @ lift(x_c[k]), return_info=True, part="coexact"))
    for k in (1, 2, 3):
        add(f"L[{k}] exact", odd, k, x_e[k],
            odd.L[k].solve(odd.L[k] @ lift(x_e[k]), return_info=True, part="exact"))
    for k in (1, 2):
        add(f"S[{k}]", odd, k + 1, x_c[k], odd.S[k].solve(odd.S[k] @ lift(x_c[k]), return_info=True),
            apply=odd.G[k])
    for k, eps in [(k, smoothing) for k in range(4)] + [(1, jnp.asarray(1.0))]:
        x, at = x_full[k], f"eps {float(eps):.0e}"
        b = odd.M[k] @ lift(x) + (odd.L[k] @ lift(x)) * eps
        add(f"shifted[{k}], {at}", odd, k, x, odd.shifted(k, eps).solve(b, return_info=True))
        if k in (1, 2):
            add(f"shifted_stiffness[{k}], {at}", odd, k + 1, x,
                odd.shifted_stiffness(k, eps).solve(b, return_info=True), apply=odd.G[k])

    # Leray: the M-orthogonal complement of the (weakly) divergence-free k-forms is the (weak) gradients. The
    # gradient part is scaled to the size of the divergence-free one.
    for k, v, grad in ((2, even.G[1] @ rand(even, 1), even.M[2].solve(even.D[2].T @ rand(even, 3))),
                       (1, even.M[1].solve(even.G[1].T @ rand(even, 2)), even.G[0] @ rand(even, 0))):
        grad = grad * even.l2_norm(v, k) / even.l2_norm(grad, k)
        cases.append((f"leray k={k}", even, k, v, even.leray(v + grad, k=k)[0], None))

    failed = []
    print(f"\n  {'case':<32}{'info':>6}  error / tol")
    for name, space, k, want, got, info in cases:
        err = float(space.l2_norm(got - want, k) / space.l2_norm(want, k)) / seq.tol
        print(f"  {name:<32}{'' if info is None else int(info):>6}  {err:.2e}")
        if (info is not None and int(info) <= 0) or not err <= ROUND_TRIP:
            failed.append(f"{name}: info {info}, error {err:.2e} tol")
    assert not failed, "; ".join(failed)


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


@FLOAT64
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


@FLOAT64
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
