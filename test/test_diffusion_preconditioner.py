"""Pin the diffusion solve's preconditioner choice to measured iteration counts.

Nothing anywhere pinned the iteration count of ``M_k + eps L_k``, which is why
a comment claiming ``eps * lambda_max(M^-1 L) ~ 0.26`` -- wrong by two and a
half orders of magnitude -- justified the mass atom for years without anyone
noticing that the velocity-smoothing solve had become 75% of a relaxation step.

These tests are deliberately about *which preconditioner wins where*, not about
an absolute count: the absolute number moves with geometry, tolerance and
dtype, and a test that pins it would be rewritten rather than believed. What
must not drift is the ordering, and the fact that the ordering reverses at a
crossover in ``eps``.

The reference numbers below are li383 at ``ns=(8,16,8)``, ``p=3``, float64,
``tol = sqrt_eps``:

    eps       mass    laplacian
    1e-06      498         3682
    1e-04      750         2919
    1e-03     2130         1655
    1e-02     5774          950
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.operators import (apply_inverse_mass_plus_eps_laplace_matrix,
                           apply_mass_matrix)

# The solve is O(10^3) iterations at the smoothing eps, so one sequence is
# built for the whole module rather than per test.
NS = (8, 16, 8)
P = 3
K = 2
GEOMETRY = "data/wout_li383_low_res_reference.nc"

pytestmark = pytest.mark.skipif(
    not os.path.isfile(GEOMETRY), reason=f"{GEOMETRY} is absent")


@pytest.fixture(scope="module")
def seq_ops():
    """li383 at (8,16,8) p=3 with the preconditioner atoms on the bundle."""
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces
    seq, _ = build_sequence(GEOMETRY, NS, P)
    ops = seq.build_preconditioners()
    seq.set_operators(compute_nullspaces(seq, ops))
    return seq, seq.operators


@pytest.fixture(scope="module")
def rhs(seq_ops):
    """``M_2 u`` for a fixed pseudo-random ``u``, the shape smooth_velocity uses."""
    seq, _ = seq_ops
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.standard_normal(seq.n(K, True)), dtype=mrx.DTYPE)
    return apply_mass_matrix(seq, u, K, dirichlet=True)


def solve_iters(seq, ops, b, eps, kind, maxiter=20000):
    """Solve ``(M + eps L) x = b`` with one preconditioner kind.

    Returns ``(iterations, converged)``. ``info`` is the SIGNED iteration
    count -- negative means converged -- which this file reads correctly
    because reading it as "0 if converged" is a documented past bug in
    mrx/solvers.py.
    """
    _, info = apply_inverse_mass_plus_eps_laplace_matrix(
        seq, ops, b, K, eps, dirichlet=True, preconditioner=kind,
        maxiter=maxiter, return_info=True)
    return abs(int(info)), int(info) <= 0


@pytest.mark.parametrize("eps", [1e-3, 1e-2])
def test_laplacian_kind_wins_above_the_crossover(seq_ops, rhs, eps):
    """At the velocity-smoothing eps, ``(1/eps) P_L`` beats the mass atom.

    ``eps = 1e-3`` is exactly ``0.064/n_r^2`` at ``n_r = 8``, i.e. what
    TimeStepper uses, so this is the production configuration and not a
    contrived one.
    """
    seq, ops = seq_ops
    mass_iters, mass_ok = solve_iters(seq, ops, rhs, eps, 'auto')
    lap_iters, lap_ok = solve_iters(seq, ops, rhs, eps, 'laplacian')
    assert mass_ok and lap_ok, (
        f"both kinds must converge at eps={eps:g} "
        f"(mass {mass_iters}, laplacian {lap_iters})")
    assert lap_iters < mass_iters, (
        f"at eps={eps:g} the laplacian kind took {lap_iters} iterations and "
        f"the mass atom {mass_iters}; the laplacian kind is the default for "
        "velocity smoothing precisely because it should win here")


@pytest.mark.parametrize("eps", [1e-6, 1e-4])
def test_mass_kind_wins_below_the_crossover(seq_ops, rhs, eps):
    """Far from the crossover the ordering reverses, which is why 'auto' is mass.

    The resistive step passes ``dt * eta`` here, which lives in this regime.
    If this ever fails, the ``'laplacian'`` kind has become safe as a global
    default and ``_coerce_diffusion_preconditioner_spec`` should be revisited.
    """
    seq, ops = seq_ops
    mass_iters, mass_ok = solve_iters(seq, ops, rhs, eps, 'auto')
    lap_iters, _ = solve_iters(seq, ops, rhs, eps, 'laplacian')
    assert mass_ok, f"the mass atom must converge at eps={eps:g}"
    assert mass_iters < lap_iters, (
        f"at eps={eps:g} the mass atom took {mass_iters} iterations and the "
        f"laplacian kind {lap_iters}; 'auto' resolves to mass on the strength "
        "of this ordering")


def test_the_smoothing_solve_is_not_quietly_expensive(seq_ops, rhs):
    """A ceiling on the production configuration, loose enough to be believed.

    The point is not the exact count but that it cannot silently grow by an
    order of magnitude again. Measured 1655; the ceiling is 3x that, which
    still catches a regression to the mass atom's 2130 becoming 8000.
    """
    seq, ops = seq_ops
    iters, ok = solve_iters(seq, ops, rhs, 0.064 / NS[0] ** 2, 'laplacian')
    assert ok, f"the production smoothing solve did not converge ({iters} iters)"
    assert iters < 5000, (
        f"the smoothing solve took {iters} iterations, measured 1655 when the "
        "laplacian kind landed; something has regressed in the preconditioner")


def test_unknown_kind_is_rejected(seq_ops, rhs):
    """The kind list is the contract; a typo must not fall back to something."""
    seq, ops = seq_ops
    with pytest.raises(ValueError, match="preconditioner kind must be one of"):
        apply_inverse_mass_plus_eps_laplace_matrix(
            seq, ops, rhs, K, 1e-3, dirichlet=True,
            preconditioner='laplacean', maxiter=1)
