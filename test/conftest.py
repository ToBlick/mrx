"""Shared pytest fixtures for the MRX test suite.

One tier: ``pytest`` runs everything on the CPU in under ten minutes on
four cores, in float64 and float32, with no data files. Every property is
checked on ``tiny_seq``, a (4, 6, 4) p=2 spline torus built once per
session with the production setup (spline map interpolated at the
Greville points, ``build_preconditioners``, harmonic forms by the direct
Hodge construction). Tests that need low-level objects with other
parameters (quadrature, spline bases, the evaluator, projector identities)
build their own tiny sequences. Tests that read files outside the
repository carry the ``needs_data`` marker and skip when the file is
absent; ``slurm/README.md`` shows how to run them on a GPU node.
"""

import jax
import jax.numpy as jnp
import pytest

import mrx  # selects the working precision from MRX_DTYPE
from mrx.derham_sequence import DeRhamSequence
from mrx.geometry import greville_interpolate_map
from mrx.mappings import toroid_map

# Betti numbers for a solid torus.
BETTI = (1, 1, 0, 0)

# Resolution (r, chi, zeta) of the session fixture.
NS_TINY = (4, 6, 4)
P = 2   # the matvec is O(N p^4)
TYPES = ("clamped", "periodic", "periodic")

# Donut-torus parameters.
TORUS_EPSILON = 1 / 3
TORUS_R0 = 1.0


@pytest.fixture(scope="session")
def torus_map():
    """Analytical map of the reference cube onto a donut-shaped solid torus."""
    return toroid_map(epsilon=TORUS_EPSILON, R0=TORUS_R0)


def build_torus_sequence(ns, torus_map):
    """The production setup on a spline-interpolated donut torus.

    1. the analytical ``toroid_map`` is interpolated to spline coefficients
       at the Greville points via :func:`greville_interpolate_map` and
       installed with ``set_spline_map``;
    2. ``build_preconditioners`` builds the metric-lumping mass and
       Laplacian atoms for every ``(k, dirichlet)`` pair;
    3. harmonic forms are computed by the direct Hodge-decomposition
       construction (``betti_numbers = (1, 1, 0, 0)``): a fixed pair of
       production saddle solves per form through ``'auto'``, i.e. the
       metric-lumping atoms just built. The shift-and-invert route costs
       an outer iteration per form on top.

    The solver tolerance is the sequence default, ``mrx.sqrt_eps()``, so the
    same fixture is meaningful in both precisions.
    """
    import time
    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, (P, P, P), P + 1, TYPES, polar=True,
        maxiter=1000, betti_numbers=BETTI,
    )
    seq.set_spline_map(greville_interpolate_map(torus_map, seq))
    t1 = time.perf_counter()
    seq.build_preconditioners()
    t2 = time.perf_counter()
    seq._compute_nullspaces(BETTI, direct=True)
    t3 = time.perf_counter()
    print(f"\n  torus {ns}: geometry {t1 - t0:.0f} s, preconditioners {t2 - t1:.0f} s, "
          f"nullspaces {t3 - t2:.0f} s")
    return seq


@pytest.fixture(scope="session")
def tiny_seq(torus_map):
    """The (4, 6, 4) p=2 spline torus every solve-based test runs on."""
    return build_torus_sequence(NS_TINY, torus_map)


# ---------------------------------------------------------------------------
# Small helpers usable from any test
# ---------------------------------------------------------------------------

def n_dofs(seq, k, dirichlet):
    """Return the DOF count for k-forms with the given boundary condition."""
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


@pytest.fixture(scope="session")
def precond_jit(tiny_seq):
    """JIT-compiled and warmed-up mass preconditioner applies on ``tiny_seq``.

    Keyed by ``(label, k, dbc)``; the metric-lumping applies for every
    ``(k, dirichlet)`` pair are compiled once per session so the probe tests
    do not re-JIT.
    """
    from mrx.operators import apply_mass_matrix_preconditioner
    ops = tiny_seq.operators
    jit_dict = {}
    for k in range(4):
        for dbc in (False, True):
            jit_dict[("metric_lumping", k, dbc)] = jax.jit(
                lambda v, k=k, dbc=dbc: apply_mass_matrix_preconditioner(
                    tiny_seq, ops, v, k, dirichlet=dbc, kind="metric_lumping",
                )
            )
    for (_, k, dbc), fn in jit_dict.items():
        dummy = jnp.zeros(n_dofs(tiny_seq, k, dbc), dtype=mrx.DTYPE)
        jax.block_until_ready(fn(dummy))
    return jit_dict
