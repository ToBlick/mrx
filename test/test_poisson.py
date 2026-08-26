"""Hodge-Laplacian solves on the session torus (ci tier).

1. k=0 Poisson against the analytic donut solutions of
   ``docs/manufactured_solutions.md`` -- Dirichlet (``u = cos(pi r^2/2)``)
   and free (``u = cos(2 pi zeta)``, the singular system with the constant
   deflated) -- with the error below a measured band, and the L2 order
   between (4, 6, 4) and (8, 12, 8) at p = 2.
2. the k=1 and k=2 saddle solves (both boundary conditions, so both the
   singular and the non-singular systems) with the PRODUCTION preconditioner
   -- ``'auto'`` resolves to the metric-lumping mass and the metric-lumping
   atom as ``schur.outer``, which the test pins by name and by iteration
   count against the explicit spec -- reaching ``seq.tol`` under a measured
   iteration band.
3. the Leray projection at k=2 (the relaxation's) and k=1: the projected
   field is divergence-free at solver tolerance, the projection is
   idempotent and does not increase the energy.

The manufactured solutions are a compact copy of the k=0 generators in
``scripts/config_scripts/test_torus_poisson_all_k_sparse.py``, which sets
``MRX_DTYPE`` and imports hydra on import, so it is not imported here.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.geometry import greville_interpolate_map
from mrx.nullspace import _set_null
from mrx.operators import (_coerce_saddle_preconditioner_spec,
                           assemble_incidence_operators)
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from mrx.quadrature import evaluate_at_xq
from test.conftest import BETTI, P, TORUS_EPSILON, TYPES, n_dofs

PI = jnp.pi
A = TORUS_EPSILON          # minor radius of the donut (major radius 1)
NS_FINE = (8, 12, 8)       # one uniform refinement of NS_TINY


# ---------------------------------------------------------------------------
# k = 0 manufactured solutions on toroid_map(epsilon=A, R0=1)
# ---------------------------------------------------------------------------

def _R(x):
    return 1.0 + A * x[0] * jnp.cos(2 * PI * x[1])


def u_cos(x):
    """Free (natural) BC: ``u = cos(2 pi zeta)``, zero mean, so it is orthogonal
    to the constant that spans the kernel."""
    return jnp.cos(2 * PI * x[2]) * jnp.ones(1)


def f_cos(x):
    """``-Delta cos(2 pi zeta) = cos(2 pi zeta) / R^2``."""
    return jnp.cos(2 * PI * x[2]) / _R(x) ** 2 * jnp.ones(1)


def u_par(x):
    """Dirichlet: ``u = cos(pi r^2 / 2)`` vanishes at r = 1 and is smooth at r = 0."""
    return jnp.cos(0.5 * PI * x[0] ** 2) * jnp.ones(1)


def f_par(x):
    """``-Delta cos(pi r^2/2) = (2 pi s + pi^2 r^2 c) / eps^2 + pi r s cos(2 pi chi) / (eps R)``."""
    r = x[0]
    s = jnp.sin(0.5 * PI * r ** 2)
    c = jnp.cos(0.5 * PI * r ** 2)
    return ((2 * PI * s + PI ** 2 * r ** 2 * c) / A ** 2
            + PI * r * s * jnp.cos(2 * PI * x[1]) / (A * _R(x))) * jnp.ones(1)


def _scalar_rel_l2_error(seq, u_hat, exact, dirichlet):
    comp_info, comp_shapes = seq._form_comp_info(0)
    eT = seq.e0_dbc_T if dirichlet else seq.e0_T
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    u_h = evaluate_at_xq(eT @ u_hat, comp_info, comp_shapes, quad_shape, 1)
    u_ex = jax.vmap(exact)(seq.quad.x)
    wJ = seq.jacobian_j * seq.quad.w
    diff = u_h - u_ex
    num = jnp.einsum("qi,qi,q->", diff, diff, wJ)
    den = jnp.einsum("qi,qi,q->", u_ex, u_ex, wJ)
    return float(jnp.sqrt(num / den))


def _solve_k0(seq, dirichlet, preconditioner='auto'):
    exact, source = (u_par, f_par) if dirichlet else (u_cos, f_cos)
    b = seq.load(source, 0, dirichlet=dirichlet)
    u, info = seq.apply_inverse_laplacian(
        b, 0, dirichlet=dirichlet, preconditioner=preconditioner, return_info=True)
    residual = seq.apply_laplacian(u, 0, dirichlet=dirichlet) - b
    rel_res = float(jnp.linalg.norm(residual) / jnp.linalg.norm(b))
    return _scalar_rel_l2_error(seq, u, exact, dirichlet), int(info), rel_res


# Measured on the (4, 6, 4) p=2 spline donut, float64, 2026-08-26 (see the
# print): relative L2 error 6.950e-2 (free) and 4.337e-2 (dbc). The band
# is 1.25x that, so a wrong metric factor, a wrong boundary row or a
# stalled solve fails it while the float32 solve (tolerance sqrt(eps) =
# 3.5e-4, far below the discretisation error) passes.
K0_ERROR = {False: 8.7e-2, True: 5.4e-2}
# CG iterations of the production k=0 solve (metric-lumping atom): 8 and
# 5 measured; the band is twice that, since a count this small moves by
# one or two with the precision.
K0_ITERS = {False: 16, True: 10}


@pytest.mark.parametrize("dirichlet", (False, True), ids=["free", "dbc"])
def test_k0_poisson_matches_analytic(tiny_seq, dirichlet):
    seq = tiny_seq
    err, info, rel_res = _solve_k0(seq, dirichlet)
    print(f"\n  k=0 {'dbc' if dirichlet else 'free'}: relative L2 error {err:.3e}, "
          f"{-info} iterations, residual {rel_res:.2e}")
    assert info < 0, f"k=0 solve did not converge (info={info})"
    # The solve stops at seq.tol on its own (preconditioned) residual; the
    # plain dual residual is within a small factor of that.
    assert rel_res <= 10 * seq.tol
    assert err < K0_ERROR[dirichlet]
    assert -info <= K0_ITERS[dirichlet]


@pytest.fixture(scope="module")
def fine_seq(torus_map):
    """The same donut at (8, 12, 8) p=2, with only what the k=0 ORDER check
    needs: the incidence operators and the constant as the free-BC kernel
    (closed form, no inverse iteration). No atoms -- the solve below runs
    on the closed-form Jacobi diagonal, since the order of the
    discretisation error does not depend on the preconditioner and the
    k=0 atom build was 50 s of the fixture's cost on the GPU."""
    t0 = time.perf_counter()
    seq = DeRhamSequence(NS_FINE, (P, P, P), P + 1, TYPES, polar=True,
                         maxiter=1000, betti_numbers=BETTI)
    seq.evaluate_1d()
    seq.set_spline_map(greville_interpolate_map(torus_map, seq))
    seq.set_operators(assemble_incidence_operators(seq))
    v0 = jnp.ones(seq.n0)
    v0 = v0 / seq.l2_norm(v0, 0, dirichlet=False)
    seq.set_operators(_set_null(seq.operators, 0, False, v0[None, :]))
    print(f"\n  fine (8,12,8) k=0 build: {time.perf_counter() - t0:.1f} s")
    return seq


@pytest.mark.parametrize("dirichlet", (False, True), ids=["free", "dbc"])
def test_k0_poisson_converges_at_order_p_plus_1(tiny_seq, fine_seq, dirichlet):
    """One uniform refinement at p = 2 cuts the L2 error by ~2^3.

    The map is re-interpolated on each mesh, so the geometry error refines
    with the solution. Measured orders 4.07 (free) and 4.31 (dbc) on
    2026-08-26 -- above p + 1 = 3, which is the bound: the coarse mesh has
    four radial cells, so the rate is pre-asymptotic and generous.
    """
    err_coarse, _, _ = _solve_k0(tiny_seq, dirichlet)
    err_fine, info, _ = _solve_k0(fine_seq, dirichlet, preconditioner='jacobi')
    order = np.log2(err_coarse / err_fine)
    print(f"\n  k=0 {'dbc' if dirichlet else 'free'}: error {err_coarse:.3e} -> "
          f"{err_fine:.3e}, order {order:.2f}, {-info} iterations on the fine mesh")
    assert info < 0
    assert order > 3.0, f"k=0 {'dbc' if dirichlet else 'free'}: observed order {order:.2f}"


# ---------------------------------------------------------------------------
# k = 1, 2 saddle solves with the production preconditioner
# ---------------------------------------------------------------------------

_SADDLE_CASES = [
    pytest.param(1, False, id="k1-free"),   # singular: the toroidal 1-form
    pytest.param(1, True, id="k1-dbc"),
    pytest.param(2, False, id="k2-free"),
    pytest.param(2, True, id="k2-dbc"),     # singular: the toroidal 2-form
]

# The production default spelled out, so the test can prove 'auto' IS it.
_PRODUCTION = SaddlePointPreconditionerSpec(
    mass=MassPreconditionerSpec(kind='metric_lumping'),
    schur=SchurPreconditionerSpec(
        inner=MassPreconditionerSpec(kind='metric_lumping'),
        outer=MassPreconditionerSpec(kind='metric_lumping')),
)

# MINRES iterations of the production solve on tiny_seq, measured 2026-08-26
# in float64 (see the print): 99, 56, 105, 95; the band is ~1.25x that. The
# per-DoF jacobi outer (preconditioner='jacobi') needed 329, 145, 397, 332
# on the same systems -- 2.6-3.8x, not re-measured here because its Schur
# diagonal is an O(N)-apply probe.
SADDLE_ITERS = {(1, False): 125, (1, True): 70, (2, False): 130, (2, True): 120}


def _compatible_rhs(seq, k, dbc, seed):
    """A random dual k-form with its harmonic component removed."""
    n = n_dofs(seq, k, dbc)
    rhs = jnp.asarray(np.random.default_rng(seed).standard_normal(n), dtype=mrx.DTYPE)
    for v in seq._get_nullspace(k, dbc):
        rhs = rhs - (v @ rhs) * seq.apply_mass_matrix(v, k, dirichlet=dbc)
    return rhs


@pytest.mark.parametrize("k,dirichlet", _SADDLE_CASES)
def test_saddle_solve_uses_the_production_preconditioner(tiny_seq, k, dirichlet):
    seq = tiny_seq
    spec = _coerce_saddle_preconditioner_spec(
        seq, seq.operators, k=k, dirichlet=dirichlet, preconditioner='auto')
    assert spec == _PRODUCTION, (
        f"k={k} dbc={dirichlet}: 'auto' resolved to {spec} although the atom "
        "is assembled")

    rhs = _compatible_rhs(seq, k, dirichlet, seed=31 + 7 * k + int(dirichlet))
    u, info = seq.apply_inverse_laplacian(rhs, k, dirichlet=dirichlet, return_info=True)
    residual = seq.apply_laplacian(u, k, dirichlet=dirichlet) - rhs
    rel_res = float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))
    print(f"\n  k={k} dbc={dirichlet}: {-int(info)} MINRES iterations, residual {rel_res:.2e}")
    assert int(info) < 0, f"production saddle solve did not converge (info={int(info)})"
    # MINRES stops at seq.tol in its preconditioned norm and L_k u is applied
    # through an inner mass solve at seq.tol; the plain dual residual came
    # out at up to 26 tol (measured 3.9e-7 at k=1 free).
    assert rel_res <= 100 * seq.tol
    assert -int(info) <= SADDLE_ITERS[(k, dirichlet)]

    if (k, dirichlet) == (1, False):
        # The singular system once more with the spec spelled out: the same
        # count to within the ~1% run-to-run noise of a GPU solve (two
        # compilations of one jaxpr differed by 2 of ~100 iterations at
        # k=2). One case only -- every solve here is a fresh compile.
        _, info_explicit = seq.apply_inverse_laplacian(
            rhs, k, dirichlet=dirichlet, preconditioner=_PRODUCTION, return_info=True)
        assert abs(int(info) - int(info_explicit)) <= 0.05 * -int(info)


# ---------------------------------------------------------------------------
# Leray projection
# ---------------------------------------------------------------------------

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
