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
idempotent and non-expansive. The co-closed solve on ``S_1`` alone gives the same curl as the full ``L_1`` solve.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.precision import RESIDUAL_DTYPE, eps

#: 1.25x the relative L2 errors on li383 (8, 12, 12 per period) p = 2, measured 2026-09-30 in every precision:
#: 1.825e-2 (k = 0, of grad psi), 1.520e-2, 3.777e-2 and 1.136e-1.
ERROR_BAND = {0: 0.023, 1: 0.019, 2: 0.047, 3: 0.142}
#: The relative spread of the three representations' errors: below 1e-12 in float64 and 4e-4 in plain float32.
SAME = 1e-3
NAMES = ("half", "period", "full")
_ERRORS = {}


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
        return float(jnp.sqrt(v @ seq.M[k_res].precondition(v.astype(mrx.DTYPE))))
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
            phi, info = space.L[0].solve(rhs, return_info=True, dtype=RESIDUAL_DTYPE)
            _converged(space, info, space.S[0] @ phi - rhs, rhs, 0)
            h = space.nullspace(1)[0]
            uh = space.G[0] @ phi.astype(mrx.DTYPE) + (float(load @ h) / float(h @ (space.M[1] @ h))) * h
        else:
            rhs = space.G[1].T @ load
            A, info = space.L[1].solve(rhs, return_info=True, dtype=RESIDUAL_DTYPE)
            on = space if space.residual is None else space.residual
            _converged(space, info, on.L[1] @ A - rhs.astype(RESIDUAL_DTYPE), rhs, 1)
            uh = space.G[1] @ A.astype(mrx.DTYPE)
        err_sq = float(uh @ (space.M[k] @ uh)) - 2.0 * float(uh @ load) + target_sq
    elif k == 0:
        space = seq.even.free                 # psi is even
        grad_psi = jax.grad(psi)
        load = space.load(lambda x: grad_psi(space.map(x)), 1)
        rhs = space.G[0].T @ load
        u, info = space.L[0].solve(rhs, return_info=True, dtype=RESIDUAL_DTYPE)
        _converged(space, info, space.S[0] @ u - rhs, rhs, 0)
        # psi is determined up to a constant, so the error is that of its gradient. (The error of psi itself
        # would be a small difference of large numbers, ||psi|| being dominated by its mean.)
        Gu = space.G[0] @ u.astype(mrx.DTYPE)
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
        r3, info = space.L[3].solve(rhs, return_info=True, dtype=RESIDUAL_DTYPE)
        on = space if space.residual is None else space.residual
        _converged(space, info, on.L[3] @ r3 - rhs.astype(RESIDUAL_DTYPE), rhs, 3)
        g = -space.M[2].solve(space.D[2].T @ r3.astype(mrx.DTYPE))
        target_sq = _integral(space, jnp.sum(jax.vmap(grad_rho)(space.quad.x) ** 2, axis=1))
        err_sq = float(g @ (space.M[2] @ g)) - 2.0 * float(g @ load) + target_sq
    return float(np.sqrt(max(err_sq, 0.0) / target_sq))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_manufactured_solution(domains, k, name):
    """The k-th manufactured solution is recovered within its band, and with the same error on the three
    representations of the domain."""
    err = _solve(domains(name), k)
    _ERRORS[k, name] = err
    print(f"\n  k={k} {name}: relative L2 error {err:.6e}")
    assert err < ERROR_BAND[k]
    if name == NAMES[-1]:
        errs = [_ERRORS[k, n] for n in NAMES if (k, n) in _ERRORS]
        spread = (max(errs) - min(errs)) / min(errs)
        print(f"  k={k}: spread of the errors over {len(errs)} representations {spread:.2e}")
        assert spread < SAME


def test_leray_projection(seq):
    """The Leray projection of a random velocity is divergence-free to the solver tolerance, idempotent and
    non-expansive."""
    even = seq.even
    v = jnp.asarray(np.random.default_rng(5).standard_normal(even.n(2)), dtype=mrx.DTYPE)
    v = v / even.l2_norm(v, 2)
    Pv, p = even.leray(v, k=2)
    PPv, _ = even.leray(Pv, k=2, p_guess=p)
    div = float(even.l2_norm(even.G[2] @ Pv, 3))
    moved = float(even.l2_norm(PPv - Pv, 2))
    print(f"\n  ||div P v|| {div:.2e}, ||P P v - P v|| {moved:.2e}, ||P v|| {float(even.l2_norm(Pv, 2)):.4f}")
    assert div <= 10 * seq.tol + eps(10)
    assert moved <= 10 * seq.tol + eps(10)
    assert float(even.l2_norm(Pv, 2)) < 1.0


def test_a_new_geometry_can_keep_the_preconditioners(domains):
    """``set_geometry_from_map(Phi, keep_preconditioners=True)`` keeps the atoms and zeroes the harmonic forms, so
    a solve runs without a new build. Without it the bundle is gone and a solve raises. The map installed here
    is the sequence's own, and the preconditioners and harmonic forms are built again at the end."""
    from mrx.equilibria import build_map
    from mrx.nullspace import compute_nullspaces

    s = domains("period")
    Phi, _ = build_map(s.equilibrium, s, nfp=s.nfp, stellarator_symmetric=s.half_period)
    b = jnp.asarray(np.random.default_rng(5).standard_normal(s.n(1)), dtype=RESIDUAL_DTYPE)
    x_ref = s.M[1].solve(b, dtype=RESIDUAL_DTYPE)
    Pb = s.M[1].precondition(b)
    try:
        s.set_geometry_from_map(Phi, keep_preconditioners=True)
        assert bool(jnp.array_equal(s.M[1].precondition(b), Pb))
        assert not any(np.any(np.asarray(v)) for v in s.operators.nullspaces.values())
        x, info = s.M[1].solve(b, return_info=True, dtype=RESIDUAL_DTYPE)
        assert info > 0
        assert float(s.l2_norm(x - x_ref, 1) / s.l2_norm(x_ref, 1)) <= 1e3 * s.tol
        s.set_geometry_from_map(Phi)
        assert s.operators is None
        with pytest.raises(ValueError):
            s.M[1].solve(b)
    finally:
        s.build_preconditioners()
        compute_nullspaces(s, verbose=False)


def test_coexact_solve_without_the_gauge_has_the_curl_of_the_laplacian_solve(seq, b0):
    """For the right-hand side ``G_1^T M_2 B`` of the vector potential, ``L[1].solve(part="coexact", gauge=False)``
    and ``L[1].solve`` converge and differ by a closed form only, so their curls agree. With the gauge solve they
    agree themselves."""
    odd = seq.odd
    on = odd if odd.residual is None else odd.residual
    b = on.G[1].T @ (odd.M[2] @ b0)
    a, info = odd.L[1].solve(b, return_info=True, dtype=RESIDUAL_DTYPE, part="coexact", gauge=False)
    a_g = odd.L[1].solve(b, dtype=RESIDUAL_DTYPE, part="coexact")
    a_L, info_L = odd.L[1].solve(b, return_info=True, dtype=RESIDUAL_DTYPE)
    diff = float(odd.l2_norm(odd.G[1] @ (a - a_L), 2) / odd.l2_norm(odd.G[1] @ a_L, 2))
    diff_g = float(odd.l2_norm(a_g - a_L, 1) / odd.l2_norm(a_L, 1))
    print(f"\n  co-closed {int(info)} iterations, Hodge split {int(info_L)}, ||curl (a - a_L)|| / ||curl a_L|| {diff:.2e}, "
          f"with the gauge solve ||a - a_L|| / ||a_L|| {diff_g:.2e}")
    assert info > 0 and info_L > 0
    assert diff <= 1e3 * seq.tol
    assert diff_g <= 1e3 * seq.tol


@pytest.mark.parametrize("k", [1, 2, 3])
def test_parts_of_the_laplacian_solve(domains, k):
    """``L[k].solve(b, part=...)`` on right-hand sides of one kind, ``G_k^T w`` (co-exact) and ``M_k G_{k-1} c``
    (exact): each part converges to the full solve of the same right-hand side, the co-exact solution is
    orthogonal to the exact forms, and the full solve of the sum is the sum of the parts. 3-forms only have the
    exact part."""
    s = domains("period")
    on = s if s.residual is None else s.residual        # the right-hand sides are formed in the residual precision
    rng = np.random.default_rng(k)
    c = jnp.asarray(rng.standard_normal(s.n(k - 1)), dtype=RESIDUAL_DTYPE)
    b = {"exact": on.M[k] @ (on.G[k - 1] @ c)}
    if k < 3:
        b["coexact"] = on.G[k].T @ jnp.asarray(rng.standard_normal(s.n(k + 1)), dtype=RESIDUAL_DTYPE)
    x = {}
    for part, rhs in b.items():
        x[part], info = s.L[k].solve(rhs, return_info=True, dtype=RESIDUAL_DTYPE, part=part)
        x_full, info_full = s.L[k].solve(rhs, return_info=True, dtype=RESIDUAL_DTYPE)
        diff = float(s.l2_norm(x[part] - x_full, k) / s.l2_norm(x_full, k))
        print(f"\n  L{k} {part}: {int(info)} iterations against {int(info_full)} of the full solve, "
              f"relative difference {diff:.2e}")
        assert info > 0 and info_full > 0
        assert diff <= 1e3 * s.tol
    if k == 3:
        return
    gauge = s.G[k - 1].T @ (s.M[k] @ x["coexact"])
    assert float(jnp.linalg.norm(gauge)) <= 1e3 * s.tol * float(jnp.linalg.norm(s.M[k] @ x["coexact"]))
    x_sum = s.L[k].solve(b["coexact"] + b["exact"], dtype=RESIDUAL_DTYPE)
    assert float(s.l2_norm(x_sum - x["coexact"] - x["exact"], k) / s.l2_norm(x_sum, k)) <= 1e3 * s.tol


@pytest.mark.parametrize("k", [1, 2])
def test_the_coexact_solve_without_its_gauge_takes_a_guess(domains, k):
    """Started from its own result, the co-exact solve without the gauge is converged at once."""
    s = domains("period")
    on = s if s.residual is None else s.residual
    b = on.G[k].T @ jnp.asarray(np.random.default_rng(10 + k).standard_normal(s.n(k + 1)), dtype=RESIDUAL_DTYPE)
    y, info = s.L[k].solve(b, return_info=True, dtype=RESIDUAL_DTYPE, part="coexact", gauge=False)
    y2, info2 = s.L[k].solve(b, return_info=True, dtype=RESIDUAL_DTYPE, part="coexact", gauge=False, guess=y)
    assert 0 <= info2 < info
    assert float(s.l2_norm(on.G[k] @ (y2 - y), k + 1) / s.l2_norm(on.G[k] @ y, k + 1)) <= 1e3 * s.tol


def test_the_coexact_part_of_a_0_form_solve_is_the_solve(domains):
    """Every 0-form solution is co-exact, so ``L[0].solve(b, part="coexact")`` is ``L[0].solve(b)``."""
    s = domains("period")
    on = s if s.residual is None else s.residual
    b = on.G[0].T @ jnp.asarray(np.random.default_rng(7).standard_normal(s.n(1)), dtype=RESIDUAL_DTYPE)
    x, info = s.L[0].solve(b, return_info=True, dtype=RESIDUAL_DTYPE, part="coexact")
    x_full = s.L[0].solve(b, dtype=RESIDUAL_DTYPE)
    assert info > 0
    assert float(s.l2_norm(x - x_full, 0) / s.l2_norm(x_full, 0)) <= 1e3 * s.tol


def test_parts_that_do_not_exist_are_refused(domains):
    s = domains("period")
    with pytest.raises(ValueError):
        s.L[0].solve(jnp.zeros(s.n(0)), part="exact")
    with pytest.raises(ValueError):
        s.L[3].solve(jnp.zeros(s.n(3)), part="coexact")
    with pytest.raises(ValueError):
        s.L[1].solve(jnp.zeros(s.n(1)), part="closed")
    with pytest.raises(ValueError):
        s.L[1].solve(jnp.zeros(s.n(1)), guess=jnp.zeros(s.n(1)), part="exact")
    with pytest.raises(ValueError):
        s.L[1].solve(jnp.zeros(s.n(1)), guess=jnp.zeros(s.n(1)), part="coexact")
    with pytest.raises(ValueError):
        s.L[1].solve(jnp.zeros(s.n(1)), gauge=False)
