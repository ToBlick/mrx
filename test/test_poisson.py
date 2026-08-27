"""The eight Hodge-Laplacian solves on the session torus, against manufactured solutions.

Every ``(k, dirichlet)`` pair -- the four singular systems (k0 free, k1
free, k2 dbc, k3 dbc) and the four non-singular ones -- is solved with the
production ``'auto'`` preconditioner, which the test pins by name: the
metric-lumping atom at k=0 and, for the saddle solves, the metric-lumping
mass with the atom as ``schur.outer``. These eight solves are what
exercises the eight preconditioner pairs ``build_preconditioners``
assembles on ``tiny_seq``. Each solve must converge to ``seq.tol``, land
under a measured relative-L2-error band (a wrong metric factor, boundary
row or extraction moves the error by a factor) and under a measured
iteration band (a broken preconditioner multiplies the count).

The manufactured solutions are ``test/manufactured.py``, shared with
``scripts/poisson_study.py``. The Leray projection at k=2 (the
relaxation's) and k=1 is checked separately: divergence-free at solver
tolerance, idempotent, non-expansive.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.operators import (
    _coerce_saddle_preconditioner_spec,
    _metric_lumping_available,
)
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from test.conftest import TORUS_EPSILON, n_dofs
from test.manufactured import CASES, case_specs, case_tag, relative_l2_error

# The production saddle default spelled out, so the test can prove 'auto' IS it.
_PRODUCTION = SaddlePointPreconditionerSpec(
    mass=MassPreconditionerSpec(kind='metric_lumping'),
    schur=SchurPreconditionerSpec(
        inner=MassPreconditionerSpec(kind='metric_lumping'),
        outer=MassPreconditionerSpec(kind='metric_lumping')),
)

# Relative L2 error of each solve on the (4, 6, 4) p=2 spline donut,
# measured 2026-08-26 in float64 (see the print); the band is 1.25x that.
# The float32 solve stops at sqrt(eps) = 3.5e-4, far below these
# discretisation errors, so the bands hold in either precision.
ERROR_MEASURED = {
    (0, False): 6.950e-2, (0, True): 4.337e-2,
    (1, False): 1.640e-1, (1, True): 1.502e-1,
    (2, False): 1.621e-1, (2, True): 8.172e-2,
    (3, False): 1.846e-1, (3, True): 6.558e-2,
}
# Iterations of the production solve (CG at k=0, MINRES at k>=1), measured
# with the errors above; the band is 2x, since a count moves by a few
# percent between precisions and compilations (~1% noise floor).
ITERS_MEASURED = {
    (0, False): 8, (0, True): 5,
    (1, False): 34, (1, True): 21,
    (2, False): 54, (2, True): 44,
    (3, False): 22, (3, True): 22,
}


@pytest.fixture(scope="module")
def specs(torus_map):
    return case_specs(TORUS_EPSILON, torus_map)


@pytest.mark.parametrize("k,dirichlet", CASES, ids=[case_tag(*c) for c in CASES])
def test_manufactured_solution(tiny_seq, specs, k, dirichlet):
    seq = tiny_seq
    if k == 0:
        assert _metric_lumping_available(seq.operators, 0, dirichlet), (
            "the k=0 metric-lumping atom is not assembled, so 'auto' would "
            "fall back to jacobi")
    else:
        spec = _coerce_saddle_preconditioner_spec(
            seq, seq.operators, k=k, dirichlet=dirichlet, preconditioner='auto')
        assert spec == _PRODUCTION, (
            f"k={k} dbc={dirichlet}: 'auto' resolved to {spec} although the "
            "atom is assembled")

    case = specs[(k, dirichlet)]
    b = seq.load(case["src_ref"], k, dirichlet=dirichlet, frame='ref')
    u, info = seq.apply_inverse_laplacian(
        b, k, dirichlet=dirichlet, preconditioner='auto', return_info=True)
    residual = seq.apply_laplacian(u, k, dirichlet=dirichlet) - b
    rel_res = float(jnp.linalg.norm(residual) / jnp.linalg.norm(b))
    err = relative_l2_error(seq, k, dirichlet, u, case["exact"])
    print(f"\n  {case_tag(k, dirichlet)}: relative L2 error {err:.3e}, "
          f"{-int(info)} iterations, residual {rel_res:.2e}")
    assert int(info) < 0, f"{case_tag(k, dirichlet)} did not converge (info={int(info)})"
    # The solve stops at seq.tol on its own preconditioned residual; MINRES
    # applies L_k through an inner mass solve at seq.tol as well, so the
    # plain dual residual came out at up to 26 tol (k=1 free, 2026-08-26).
    assert rel_res <= 100 * seq.tol
    assert err < 1.25 * ERROR_MEASURED[(k, dirichlet)]
    assert -int(info) <= 2 * ITERS_MEASURED[(k, dirichlet)]


@pytest.mark.parametrize("k", (2, 1))
def test_leray_projection(tiny_seq, k):
    """``P v`` is divergence-free at solver tolerance, ``P P v = P v`` and
    ``||P v||_M <= ||v||_M``.

    k=2 is the relaxation's projection (Dirichlet spaces, k=3 pressure);
    k=1 the free-space one through the k=0 Laplacian.
    """
    seq = tiny_seq
    dbc = k == 2
    n = n_dofs(seq, k, dbc)
    v = jnp.asarray(np.random.default_rng(5 * k).standard_normal(n), dtype=mrx.DTYPE)
    v = v / seq.l2_norm(v, k, dirichlet=dbc)
    Pv, p = seq.apply_leray_projection(v, k=k)
    PPv, _ = seq.apply_leray_projection(Pv, k=k, p_guess=p)

    if k == 2:
        div = seq.apply_incidence_matrix(Pv, 2, dirichlet_in=True, dirichlet_out=True)
        div_norm = float(seq.l2_norm(div, 3, dirichlet=True))
    else:
        # weak divergence of a free 1-form: the dual 0-form D_0^T v
        div = seq.apply_derivative_matrix(
            Pv, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
        div_norm = float(jnp.linalg.norm(div))
    moved = float(seq.l2_norm(PPv - Pv, k, dirichlet=dbc))
    e_v = float(seq.l2_norm_sq(v, k, dirichlet=dbc))
    e_Pv = float(seq.l2_norm_sq(Pv, k, dirichlet=dbc))
    print(f"\n  Leray k={k}: ||div P v|| {div_norm:.2e}, ||P P v - P v|| {moved:.2e}, "
          f"energy {e_v:.4f} -> {e_Pv:.4f}")
    # The multiplier solve stops at seq.tol; both the residual divergence and
    # the second projection's correction are that solve's error.
    assert div_norm <= 10 * seq.tol
    assert moved <= 10 * seq.tol
    assert e_Pv < e_v
