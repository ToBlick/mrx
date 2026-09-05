"""The eight Hodge-Laplacian solves on the toroid, against manufactured solutions.

Every ``(k, dirichlet)`` pair -- the four singular systems (k0 free, k1
free, k2 dbc, k3 dbc) and the four non-singular ones -- is solved with the
production ``'auto'`` preconditioner: the metric-lumping atom at k=0, the
Hodge split at k=1 and k=2, the saddle MINRES at k=3. Each solve must
converge to ``seq.tol``, land under a measured relative-L2-error band (a
wrong metric factor, boundary row or extraction moves the error by a factor)
and under a measured iteration band (a broken preconditioner multiplies the
count). The manufactured solutions are ``test/manufactured.py``, shared with
``scripts/poisson_study.py``; the Leray projections (k=2, the relaxation's;
k=1) are divergence-free at solver tolerance, idempotent and non-expansive.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from test.conftest import TORUS_EPSILON
from test.manufactured import CASES, case_specs, case_tag, relative_l2_error

# Relative L2 error of each solve on the (8, 12, 12) p=2 spline donut,
# measured 2026-09-02 in float64 (see the print); the band is 1.25x that.
# The float32 solve stops at sqrt(eps) = 3.5e-4, below these discretisation
# errors, so the bands hold in either precision.
ERROR_MEASURED = {
    (0, False): 9.315e-4, (0, True): 1.054e-3,
    (1, False): 1.060e-2, (1, True): 1.067e-2,
    (2, False): 1.430e-2, (2, True): 3.185e-3,
    (3, False): 1.998e-2, (3, True): 1.069e-2,
}
# Iterations of the production solve, measured with the errors above; the
# band is 2x (a count moves by a few percent between precisions and
# compilations).
ITERS_MEASURED = {
    (0, False): 16, (0, True): 13,
    (1, False): 39, (1, True): 25,
    (2, False): 32, (2, True): 32,
    (3, False): 51, (3, True): 42,
}


@pytest.fixture(scope="module")
def specs(torus_map):
    return case_specs(TORUS_EPSILON, torus_map)


@pytest.mark.parametrize("k,dirichlet", CASES, ids=[case_tag(*c) for c in CASES])
def test_manufactured_solution(toroid, specs, k, dirichlet):
    seq = toroid
    case = specs[(k, dirichlet)]
    b = seq.load(case["src_ref"], k, dirichlet=dirichlet, frame='ref')
    u, info = seq.apply_inverse_laplacian(b, k, dirichlet=dirichlet, return_info=True)
    residual = seq.apply_laplacian(u, k, dirichlet=dirichlet) - b
    rel_res = float(jnp.linalg.norm(residual) / jnp.linalg.norm(b))
    err = relative_l2_error(seq, k, dirichlet, u, case["exact"])
    print(f"\n  {case_tag(k, dirichlet)}: relative L2 error {err:.3e}, "
          f"{-int(info)} iterations, residual {rel_res:.2e}")
    assert int(info) < 0, f"{case_tag(k, dirichlet)} did not converge (info={int(info)})"
    # The solve stops at seq.tol on its own preconditioned residual; the
    # plain dual residual is measured through apply_laplacian's own inner
    # mass solve at seq.tol and came out at up to 253 tol (k=1 free, the
    # Hodge split, 2026-09-02).
    assert rel_res <= 1e3 * seq.tol
    assert err < 1.25 * ERROR_MEASURED[(k, dirichlet)]
    assert -int(info) <= 2 * ITERS_MEASURED[(k, dirichlet)]


@pytest.mark.parametrize("k", (2, 1))
def test_leray_projection(toroid, k):
    """``P v`` is divergence-free at solver tolerance, ``P P v = P v`` and
    ``||P v||_M <= ||v||_M``. k=2 is the relaxation's projection (Dirichlet
    spaces, k=3 pressure); k=1 the free-space one through the k=0 Laplacian."""
    seq = toroid
    dbc = k == 2
    v = jnp.asarray(np.random.default_rng(5 * k).standard_normal(seq.n(k, dbc)), dtype=mrx.DTYPE)
    v = v / seq.l2_norm(v, k, dirichlet=dbc)
    Pv, p = seq.apply_leray_projection(v, k=k)
    PPv, _ = seq.apply_leray_projection(Pv, k=k, p_guess=p)
    if k == 2:
        div = seq.apply_incidence_matrix(Pv, 2, dirichlet_in=True, dirichlet_out=True)
        div_norm = float(seq.l2_norm(div, 3, dirichlet=True))
    else:
        div = seq.apply_derivative_matrix(
            Pv, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
        div_norm = float(jnp.linalg.norm(div))
    moved = float(seq.l2_norm(PPv - Pv, k, dirichlet=dbc))
    e_v = float(seq.l2_norm_sq(v, k, dirichlet=dbc))
    e_Pv = float(seq.l2_norm_sq(Pv, k, dirichlet=dbc))
    print(f"\n  Leray k={k}: ||div P v|| {div_norm:.2e}, ||P P v - P v|| {moved:.2e}, "
          f"energy {e_v:.4f} -> {e_Pv:.4f}")
    assert div_norm <= 10 * seq.tol
    assert moved <= 10 * seq.tol
    assert e_Pv < e_v
