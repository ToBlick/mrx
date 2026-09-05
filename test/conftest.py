"""The lean suite: two sequences, few tests.

Two session fixtures, both at ``(8, 12, 12)`` p=2, built once with their
preconditioners and harmonic forms:

* ``seq``: the li383 equilibrium (``data/wout_li383_low_res_reference.nc``,
  the project's fruit-fly stellarator). The assembly and exactness checks
  probe its operators in place and the relaxation runs on its own field
  (``b0``).
* ``toroid``: the spline-interpolated analytic donut torus, where the eight
  Hodge Laplacians have closed-form manufactured solutions
  (``test/manufactured.py``, all ``(k, dirichlet)`` pairs).

Tests that need no sequence (spline bases, quadrature, precision, the
kind-dispatch audit, the file readers) are the milliseconds around them.

The suite is XLA-compile-bound: every eager solve traces and compiles its own
loop body, so the cost of a test is the number of distinct solves it makes,
not the mesh. Keep it that way -- a new test is the production configuration
plus at most one contrasting case.
"""
import os

# The suite verifies at double precision, whatever the package default is.
os.environ.setdefault("MRX_DTYPE", "float64")

import time

import pytest

#: The wout geometry, tracked in the repository.
GEOMETRY = "data/wout_li383_low_res_reference.nc"
#: Resolution (r, theta, zeta) and degree of both session sequences.
NS, P = (8, 12, 12), 2
TYPES = ("clamped", "periodic", "periodic")
#: Betti numbers of a solid torus (free boundary conditions).
BETTI = (1, 1, 0, 0)
#: Donut-torus parameters of ``toroid_map``.
TORUS_EPSILON = 1 / 3
TORUS_R0 = 1.0


@pytest.fixture(scope="session")
def seq():
    """li383 ``(8, 12, 12)`` p=2 with its metric-lumping atoms and harmonic forms."""
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    t0 = time.perf_counter()
    s, ops = build_sequence(GEOMETRY, NS, P)
    t1 = time.perf_counter()
    compute_nullspaces(s)
    t2 = time.perf_counter()
    print(f"\n  li383 {NS} p={P}: build_sequence {t1 - t0:.0f} s, "
          f"nullspaces {t2 - t1:.0f} s", flush=True)
    return s


@pytest.fixture(scope="session")
def b0(seq):
    """The equilibrium's own field, ``B = dA'`` from the histopolated Clebsch
    potential: exactly divergence-free, tangential to the wall."""
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import clebsch_potential_form, potential_two_form

    B, _, _ = potential_two_form(seq, clebsch_potential_form(load_clebsch(seq.equilibrium)))
    return B


@pytest.fixture(scope="session")
def torus_map():
    """Analytical map of the reference cube onto a donut-shaped solid torus."""
    from mrx.mappings import toroid_map

    return toroid_map(epsilon=TORUS_EPSILON, R0=TORUS_R0)


@pytest.fixture(scope="session")
def toroid(torus_map):
    """The donut torus ``(8, 12, 12)`` p=2, spline-interpolated at the Greville
    points, with its atoms and harmonic forms: the production setup on the
    one geometry whose Hodge Laplacians have closed-form solutions."""
    from mrx.derham_sequence import DeRhamSequence
    from mrx.geometry import greville_interpolate_map

    t0 = time.perf_counter()
    s = DeRhamSequence(NS, (P, P, P), P + 1, TYPES, polar=True, betti_numbers=BETTI)
    s.set_spline_map(greville_interpolate_map(torus_map, s))
    s.build_preconditioners()
    s.compute_nullspaces(BETTI)
    print(f"\n  toroid {NS} p={P}: build {time.perf_counter() - t0:.0f} s", flush=True)
    return s
