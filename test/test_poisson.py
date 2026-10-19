"""The Laplacians recover manufactured solutions on the li383 domain, identically on two representations.

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
factor, boundary row or extraction moves it by a factor. The two representations of the domain (``half``:
one field period with stellarator symmetry, ``period``: one field period without the reflection) have the same
mesh, and ``B*`` and ``psi`` have the period and the parity of li383. So the two must give the same error to the
solver tolerance: this checks the half-period quadrature and the parity views end to end. Preconditioners kept
across a change of map still precondition the new operators. The zeta-slab mass model conditions ``M_1`` far better than the metric-lumping atom, on the odd view of the
half-period domain.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

#: 1.25x the relative L2 errors on li383 (12, 20, 12 per period) p = 2, graded, quadrature (p + 1, p, p), measured
#: 2026-10-19: 6.106e-3 (k = 0, of grad psi), 7.557e-3, 1.747e-2 and 3.418e-2 (uniform theta: 5.05e-3 to 2.46e-2).
#: The errors are measured with the sequence's own rule.
ERROR_BAND = {0: 0.00763, 1: 0.00945, 2: 0.0218, 3: 0.0427}
#: The relative spread of the two representations' errors, which is below 3e-12.
SAME = 1e-3
NAMES = ("half", "period")


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
    """The k-th manufactured solution is recovered within its band, and with the same error on the two
    representations of the domain."""
    errs = {name: _solve(domains(name), k) for name in NAMES}
    spread = (max(errs.values()) - min(errs.values())) / min(errs.values())
    print(f"\n  k={k}: relative L2 errors " + ", ".join(f"{n} {e:.6e}" for n, e in errs.items())
          + f", spread {spread:.2e}")
    assert max(errs.values()) < ERROR_BAND[k]
    assert spread < SAME


def test_a_new_geometry_can_keep_the_preconditioners(domains):
    """``set_geometry_from_map(Phi, keep_preconditioners=True)`` keeps the preconditioners of the old map, which
    still precondition the operators of the new one: the mass solve on li383 stretched by 10 % in ``Z`` converges
    to the tolerance, measured with the new mass matrix. The harmonic forms are set to zero. Without
    ``keep_preconditioners`` the bundle is dropped and a solve raises. A new map of the same resolution reuses the
    compiled programs. The geometry, preconditioners and harmonic forms of ``period`` are restored at the end."""
    from mrx.equilibria import build_map
    from mrx.nullspace import compute_nullspaces

    s = domains("period")
    geometry = s.geometry
    Phi, _ = build_map(s.equilibrium, s, stellarator_symmetric=s.half_period)
    stretch = jnp.array([1.0, 1.0, 1.1])
    b = jnp.asarray(np.random.default_rng(5).standard_normal(s.n(1)))
    Pb = s.M[1].precondition(b)
    try:
        s.set_geometry_from_map(lambda x: stretch * Phi(x), keep_preconditioners=True)
        assert bool(jnp.array_equal(s.M[1].precondition(b), Pb))
        assert not any(np.any(np.asarray(v)) for v in s.operators.nullspaces.values())
        x, info = s.M[1].solve(b, return_info=True)
        r = b - s.M[1] @ x

        def norm(v):
            return float(jnp.sqrt(v @ s.M[1].precondition(v)))
        assert info > 0 and norm(r) <= 1e1 * s.tol * norm(b)
        s.set_geometry_from_map(lambda x: stretch * Phi(x))
        assert s.operators is None
        with pytest.raises(ValueError):
            s.M[1].solve(b)
    finally:
        s.set_geometry(geometry)
        s.build_preconditioners()
        compute_nullspaces(s, verbose=False)


def test_slab_mass_model_conditions_the_one_form_mass(seq):
    """kappa of P M_1 on the odd view, P the slab model of mrx.slab_mass. Measured 3.30 on li383 (12, 20, 12)
    (2026-10-16), the metric-lumping atom has 37.8. Band 4."""
    from mrx.metric_lumping import ReducedAtom
    from mrx.slab_mass import SlabMass
    from mrx.solvers import lanczos_bounds
    odd = seq.odd
    P = ReducedAtom(SlabMass(seq, seq.operators, 1), odd.reduction[(1, odd.dirichlet)])
    M = odd.M[1]
    b = M @ jnp.asarray(np.random.default_rng(0).standard_normal(odd.n(1)))
    lo, hi = lanczos_bounds(lambda x: M @ x, b, P.apply, 40)
    assert hi / lo < 4.0
