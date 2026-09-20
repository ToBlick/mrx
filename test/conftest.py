"""The lean suite: one sequence, few tests.

One session fixture, ``seq``: the li383 equilibrium
(``data/wout_li383_low_res_reference.nc``, the project's fruit-fly
stellarator) at ``(8, 12, 12)`` p=2, built once with its preconditioners
and harmonic forms, a half-period sequence (the map is stellarator
symmetric). The assembly and exactness checks probe its operators in
place, the manufactured vacuum solves of the paper run on its domain
(``test_vacuum.py``), the relaxation runs on its own field (``b0``).

Tests that need no sequence (spline bases, quadrature, precision, the file
readers) are the milliseconds around them.

The suite is XLA-compile-bound: every eager solve traces and compiles its own
loop body, so the cost of a test is the number of distinct solves it makes,
not the mesh (measured 2026-09-20: float64 runs faster than float32, and two
50-step relaxations of the same stepper cost the same as one). Keep it that
way -- a new test is the production configuration plus at most one
contrasting case -- and the compiled programs persist on disk between runs
(``outputs/xla_cache``, below).
"""
import os
import time

import jax
import pytest

# The suite is XLA-compile-bound (below), so its programs are cached on disk
# across runs: a rerun whose kernels did not change skips the compiles. The
# key is the compiled HLO, so a stale entry cannot return a wrong result, only
# an old compile time. Shared by the three precision configurations (their
# programs differ). MRX_XLA_CACHE names the directory; empty disables it.
_CACHE = os.environ.get("MRX_XLA_CACHE", os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                       "outputs", "xla_cache"))
if _CACHE:
    jax.config.update("jax_compilation_cache_dir", _CACHE)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)

#: The wout geometry, tracked in the repository.
GEOMETRY = "data/wout_li383_low_res_reference.nc"
#: Resolution (r, theta, zeta) and degree of the session sequence.
NS, P = (8, 12, 12), 2
TYPES = ("clamped", "periodic", "periodic")
#: Betti numbers of a solid torus (free boundary conditions).
BETTI = (1, 1, 0, 0)


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
