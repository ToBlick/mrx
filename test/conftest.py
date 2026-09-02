"""The lean suite: one sequence, one geometry, few tests.

Every solve-based test runs on the li383 equilibrium at ``(8, 12, 12)`` p=3,
the project's fruit-fly stellarator (``docs/source/tutorials.md``), built once
per session with its preconditioners and harmonic forms. There is no second
sequence: the vacuum-field solves are posed on the same geometry with a
physical analytic field (``test/manufactured.py``), the relaxation runs on
the state's own field, and the assembly and exactness checks probe the
operators in place. Tests that need no sequence (spline bases, quadrature,
precision, the kind-dispatch audit, the file readers) are the milliseconds
around it.

The suite is XLA-compile-bound: every eager solve traces and compiles its own
loop body, so the cost of a test is the number of distinct solves it makes,
not the mesh. Keep it that way -- a new test is the production configuration
plus at most one contrasting case.
"""
import pytest

#: The one geometry, tracked in the repository.
GEOMETRY = "data/wout_li383_low_res_reference.nc"
#: Resolution (r, theta, zeta) and degree of the session sequence.
NS, P = (8, 12, 12), 3
#: Betti numbers of a solid torus (free boundary conditions).
BETTI = (1, 1, 0, 0)


@pytest.fixture(scope="session")
def seq():
    """li383 ``(8, 12, 12)`` p=3 with its metric-lumping atoms and harmonic forms."""
    import time

    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    t0 = time.perf_counter()
    s, ops = build_sequence(GEOMETRY, NS, P)
    t1 = time.perf_counter()
    s.set_operators(compute_nullspaces(s, ops))
    t2 = time.perf_counter()
    print(f"\n  li383 {NS} p={P}: build_sequence {t1 - t0:.0f} s, "
          f"nullspaces {t2 - t1:.0f} s", flush=True)
    return s


@pytest.fixture(scope="session")
def b0(seq):
    """The equilibrium's own field, ``B = dA'`` from the histopolated Clebsch
    potential: exactly divergence-free, tangential to the wall (the
    relaxation's initial condition and the Leray test's div-free reference)."""
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import clebsch_potential_form, potential_two_form

    B, _, _ = potential_two_form(seq, clebsch_potential_form(load_clebsch(GEOMETRY)))
    return B
