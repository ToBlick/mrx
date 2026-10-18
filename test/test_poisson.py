"""The Laplacians recover manufactured solutions on the li383 domain, identically on its three representations.

Each Hodge Laplacian ``L_k`` is solved with the right-hand side of a closed-form solution:

- ``k = 0``: the natural solve of ``<grad f, grad v> = <grad psi, grad v>`` recovers ``psi`` up to a constant,
  measured by the error of its gradient.
- ``k = 1``: the vacuum field ``B* = e_phi / R + grad(R^nfp sin(nfp phi))``, curl- and divergence-free, as
  ``grad f + c h_1`` with ``f`` from the natural k = 0 solve and ``c h_1`` its harmonic part (the scalar-potential
  route).
- ``k = 2``: the same ``B*`` as ``curl A`` with ``A`` from the natural k = 1 Hodge solve (the vector-potential
  route).
- ``k = 3``: the natural solve recovers ``grad rho*`` for ``rho* = (1 - r^2) psi``, which vanishes on the wall.
  The weak gradient of the free 2-forms only represents such functions.

Every solve must converge, and its relative L2 error, computed exactly as
``||u_h - u*||^2 = <u_h, u_h>_M - 2 u_h . load(u*) + int |u*|^2``, must lie below a measured band. A wrong metric
factor, boundary row or extraction moves it by a factor. The three representations of the domain (``half``:
one field period with stellarator symmetry, ``period``: one field period, ``full``: the whole torus) have the
same mesh per period, and ``B*`` and ``psi`` have the period and the parity of li383. So the three must give the
same error to the solver tolerance: this checks the half-period quadrature, the parity views and the field-period
reduction end to end. The Leray projection, the k = 3 saddle solve of the relaxation, is divergence-free,
idempotent and non-expansive. The co-closed solve on ``S_1`` alone gives the same curl as the full ``L_1`` solve,
and the one solve of ``M_1 + eps S_1`` the same curl as the shifted solve ``(M_1 + eps L_1)^-1``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.precision import eps

#: 1.25x the relative L2 errors on li383 (12, 20, 12 per period) p = 2, measured 2026-10-13:
#: 9.431e-3 (k = 0, of grad psi), 1.197e-2, 2.563e-2 and 5.215e-2.
ERROR_BAND = {0: 0.0118, 1: 0.0150, 2: 0.0320, 3: 0.0652}
#: The relative spread of the three representations' errors, which is below 3e-12.
SAME = 1e-3
NAMES = ("half", "period", "full")


def psi(X):
    """``R^2 (1 + Z^2) + Re (x + i y)^3 + Z Im (x + i y)^3``: smooth, not harmonic, with the field period of
    li383 (nfp = 3) and even under the stellarator reflection ``(y, z) -> (-y, -z)``."""
    re, im = X[0] ** 3 - 3.0 * X[0] * X[1] ** 2, 3.0 * X[0] ** 2 * X[1] - X[1] ** 3
    return (X[0] ** 2 + X[1] ** 2) * (1.0 + X[2] ** 2) + re + X[2] * im


def vacuum_field(seq):
    """``B*(x)`` as a physical vector at the logical point ``x``, with the field periods of the file."""
    nfp = seq.equilibrium["nfp"]

    def ripple(X):                      # R^nfp sin(nfp phi) = Im (x + i y)^nfp, harmonic
        return (X[0] ** 2 + X[1] ** 2) ** (nfp / 2) * jnp.sin(nfp * jnp.arctan2(X[1], X[0]))
    grad_ripple = jax.grad(ripple)

    def f(x):
        X = seq.map(x)
        return jnp.array([-X[1], X[0], 0.0]) / (X[0] ** 2 + X[1] ** 2) + grad_ripple(X)
    return f


def _integral(seq, values):
    return float(jnp.sum(seq.quad.w * seq.jacobian_j * values))


def _converged(seq, info, residual, rhs, k_res):
    """The solve's sign and its true residual in the norm of its stopping criterion."""
    def norm(v):
        return float(jnp.sqrt(v @ seq.M[k_res].precondition(v)))
    assert int(info) >= 0, f"did not converge (info {int(info)})"
    assert norm(residual) <= 1e2 * seq.tol * norm(rhs)


def _solve(seq, k):
    """The relative L2 error of the k-th manufactured solution on ``seq``."""
    if k in (1, 2):
        space = seq.odd.free                  # B* is odd, and the solves are natural
        f = vacuum_field(space)
        target_sq = _integral(space, jnp.sum(jax.vmap(f)(space.quad.x) ** 2, axis=1))
        load = space.load(f, k)
        if k == 1:
            rhs = space.G[0].T @ load
            phi, info = space.L[0].solve(rhs, return_info=True)
            _converged(space, info, space.S[0] @ phi - rhs, rhs, 0)
            h = space.nullspace(1)[0]
            uh = space.G[0] @ phi + (float(load @ h) / float(h @ (space.M[1] @ h))) * h
        else:
            rhs = space.G[1].T @ load
            A, info = space.L[1].solve(rhs, return_info=True)
            _converged(space, info, space.L[1] @ A - rhs, rhs, 1)
            uh = space.G[1] @ A
        err_sq = float(uh @ (space.M[k] @ uh)) - 2.0 * float(uh @ load) + target_sq
    elif k == 0:
        space = seq.even.free                 # psi is even
        grad_psi = jax.grad(psi)
        load = space.load(lambda x: grad_psi(space.map(x)), 1)
        rhs = space.G[0].T @ load
        u, info = space.L[0].solve(rhs, return_info=True)
        _converged(space, info, space.S[0] @ u - rhs, rhs, 0)
        # psi is determined up to a constant, so the error is that of its gradient. (The error of psi itself
        # would be a small difference of large numbers, ||psi|| being dominated by its mean.)
        Gu = space.G[0] @ u
        target_sq = _integral(space, jnp.sum(jax.vmap(lambda x: grad_psi(space.map(x)))(space.quad.x) ** 2, axis=1))
        err_sq = float(Gu @ (space.M[1] @ Gu)) - 2.0 * float(Gu @ load) + target_sq
    else:
        space = seq.even.free

        def rho(x):
            return (1.0 - x[0] ** 2) * psi(space.map(x))

        def grad_rho(x):                      # the physical gradient, DPhi^-T d rho / dx
            return jnp.linalg.solve(jax.jacfwd(space.map)(x).T, jax.grad(rho)(x))
        load = space.load(grad_rho, 2)
        rhs = -(space.M[3] @ (space.G[2] @ space.M[2].solve(load)))     # <grad rho*, delta Lambda_i>
        r3, info = space.L[3].solve(rhs, return_info=True)
        _converged(space, info, space.L[3] @ r3 - rhs, rhs, 3)
        g = -space.M[2].solve(space.D[2].T @ r3)
        target_sq = _integral(space, jnp.sum(jax.vmap(grad_rho)(space.quad.x) ** 2, axis=1))
        err_sq = float(g @ (space.M[2] @ g)) - 2.0 * float(g @ load) + target_sq
    return float(np.sqrt(max(err_sq, 0.0) / target_sq))


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_manufactured_solution(domains, k):
    """The k-th manufactured solution is recovered within its band, and with the same error on the three
    representations of the domain."""
    errs = {name: _solve(domains(name), k) for name in NAMES}
    spread = (max(errs.values()) - min(errs.values())) / min(errs.values())
    print(f"\n  k={k}: relative L2 errors " + ", ".join(f"{n} {e:.6e}" for n, e in errs.items())
          + f", spread {spread:.2e}")
    assert max(errs.values()) < ERROR_BAND[k]
    assert spread < SAME


def test_leray_projection(seq):
    """The Leray projection of a random velocity is divergence-free to the solver tolerance, idempotent and
    non-expansive."""
    even = seq.even
    v = jnp.asarray(np.random.default_rng(5).standard_normal(even.n(2)))
    v = v / even.l2_norm(v, 2)
    Pv, p = even.leray(v, k=2)
    PPv, _ = even.leray(Pv, k=2, p_guess=p)
    div = float(even.l2_norm(even.G[2] @ Pv, 3))
    moved = float(even.l2_norm(PPv - Pv, 2))
    print(f"\n  ||div P v|| {div:.2e}, ||P P v - P v|| {moved:.2e}, ||P v|| {float(even.l2_norm(Pv, 2)):.4f}")
    assert div <= 10 * seq.tol + eps(10)
    assert moved <= 10 * seq.tol + eps(10)
    assert float(even.l2_norm(Pv, 2)) < 1.0


def test_stiffness_solve_has_the_curl_of_the_laplacian_solve(seq, b0):
    """For the right-hand side ``G_1^T M_2 B`` of the vector potential, ``S[1].solve`` and ``L[1].solve`` converge
    and differ by a closed form only, so their curls agree. With the gauge solve (``part="coexact"``) they agree
    themselves."""
    odd = seq.odd
    b = odd.G[1].T @ (odd.M[2] @ b0)
    a, info = odd.S[1].solve(b, return_info=True)
    a_g = odd.L[1].solve(b, part="coexact")
    a_L, info_L = odd.L[1].solve(b, return_info=True)
    diff = float(odd.l2_norm(odd.G[1] @ (a - a_L), 2) / odd.l2_norm(odd.G[1] @ a_L, 2))
    diff_g = float(odd.l2_norm(a_g - a_L, 1) / odd.l2_norm(a_L, 1))
    print(f"\n  co-closed {int(info)} iterations, Hodge split {int(info_L)}, ||curl (a - a_L)|| / ||curl a_L|| {diff:.2e}, "
          f"with the gauge solve ||a - a_L|| / ||a_L|| {diff_g:.2e}")
    assert info > 0 and info_L > 0
    assert diff <= 1e3 * seq.tol
    assert diff_g <= 1e3 * seq.tol


@pytest.mark.parametrize("shift", ["smoothing", "one"])
def test_shifted_stiffness_solve_has_the_curl_of_the_shifted_solve(seq, shift):
    """For a right-hand side ``M_1 z`` with every part, ``shifted_stiffness(1, eps).solve`` (one solve of
    ``M_1 + eps S_1``) and the shifted Laplacian solve converge and differ by an exact form, so their curls agree. At
    the velocity smoothing's ``eps`` and at ``eps = 1``, where the two-solve split of PR #33 did not converge."""
    from mrx.relaxation.loop import SMOOTHING_C, radial_cell_sq
    odd = seq.odd
    eps = SMOOTHING_C * radial_cell_sq(seq) if shift == "smoothing" else jnp.asarray(1.0)
    b = odd.M[1] @ jax.random.normal(jax.random.PRNGKey(4), (odd.n(1),), dtype=jnp.float64)
    a, info = odd.shifted_stiffness(1, eps).solve(b, return_info=True)
    a_f, info_f = odd.shifted(1, eps).solve(b, return_info=True)
    diff = float(odd.l2_norm(odd.G[1] @ (a - a_f), 2) / odd.l2_norm(odd.G[1] @ a_f, 2))
    print(f"\n  eps {float(eps):.2e}: one solve {int(info)} iterations, full {int(info_f)}, "
          f"||curl (a - a_f)|| / ||curl a_f|| {diff:.2e}")
    assert info > 0 and info_f > 0
    assert diff <= 1e3 * seq.tol


@pytest.mark.parametrize("k", [1, 2, 3])
def test_parts_of_the_laplacian_solve(domains, k):
    """``L[k].solve(b, part=...)`` on right-hand sides of one kind, ``G_k^T w`` (co-exact) and ``M_k G_{k-1} c``
    (exact): each part converges to the full solve of the same right-hand side, the co-exact solution is
    orthogonal to the exact forms, and the full solve of the sum is the sum of the parts. 3-forms only have the
    exact part."""
    s = domains("period")
    rng = np.random.default_rng(k)
    c = jnp.asarray(rng.standard_normal(s.n(k - 1)))
    b = {"exact": s.M[k] @ (s.G[k - 1] @ c)}
    if k < 3:
        b["coexact"] = s.G[k].T @ jnp.asarray(rng.standard_normal(s.n(k + 1)))
    x = {}
    for part, rhs in b.items():
        x[part], info = s.L[k].solve(rhs, return_info=True, part=part)
        x_full, info_full = s.L[k].solve(rhs, return_info=True)
        diff = float(s.l2_norm(x[part] - x_full, k) / s.l2_norm(x_full, k))
        print(f"\n  L{k} {part}: {int(info)} iterations against {int(info_full)} of the full solve, "
              f"relative difference {diff:.2e}")
        assert info > 0 and info_full > 0
        assert diff <= 1e3 * s.tol
    if k == 3:
        return
    gauge = s.G[k - 1].T @ (s.M[k] @ x["coexact"])
    assert float(jnp.linalg.norm(gauge)) <= 1e3 * s.tol * float(jnp.linalg.norm(s.M[k] @ x["coexact"]))
    x_sum = s.L[k].solve(b["coexact"] + b["exact"])
    assert float(s.l2_norm(x_sum - x["coexact"] - x["exact"], k) / s.l2_norm(x_sum, k)) <= 1e3 * s.tol
