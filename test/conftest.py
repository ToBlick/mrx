"""Shared pytest fixtures for the MRX test suite.

Two tiers, selected by the ``gpu`` marker (``pyproject.toml`` deselects it
by default, ``pytest -m gpu`` runs it):

* the default tier runs on a 4-core CPU runner in under ten minutes, in
  float64 and float32, with no data files.  Every property that holds at any
  resolution (identities, symmetries, ``d.d = 0``, projector idempotency,
  preconditioner SPD-ness and inertness, solver convergence) is checked here
  on ``tiny_seq``, a (4, 6, 4) p=2 spline torus built once per session;
* the ``gpu`` tier holds the production-resolution fixture ``torus_seq``
  ((8, 16, 8) p=2), the measured iteration-count bands calibrated on it, the
  convergence-order tests and everything that reads data outside the
  repository.

Both fixtures are built by the same function, so the two tiers test the same
production setup (spline map interpolated at the Greville points,
``build_preconditioners``, harmonic forms by inverse iteration) at two
resolutions.  Tests that need different parameters (low-level quadrature and
spline checks, the evaluator, projector identities) build their own tiny
objects on the fly.
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

# Resolutions (r, chi, zeta) of the two session fixtures; the periodic
# directions are finer to resolve the azimuthal variation of the Poisson
# tests on the production-sized one.
NS_TINY = (4, 6, 4)
NS = (8, 16, 8)
P = 2   # the matvec is O(N p^4); convergence-ORDER tests that depend on p
        # parametrise it themselves rather than inherit this default.
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
    2. ``build_preconditioners`` assembles the incidence operators, the
       Jacobi mass diagonals and the metric-lumping Laplacian atoms for
       every ``(k, dirichlet)`` pair;
    3. harmonic forms are computed by the direct Hodge-decomposition
       construction (``betti_numbers = (1, 1, 0, 0)``): a fixed pair of
       production saddle solves per form through ``'auto'``, i.e. the
       metric-lumping atoms just built. The shift-and-invert route costs
       an outer iteration per form and pins ``schur.outer='jacobi'``
       (``nullspace._nullspace_shifted_preconditioner``), which is neither
       the production solve nor cheap.

    The solver tolerance is the sequence default, ``mrx.sqrt_eps()``, so the
    same fixture is meaningful in both precisions.
    """
    import time
    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, (P, P, P), P + 1, TYPES, polar=True,
        maxiter=1000, betti_numbers=BETTI,
    )
    seq.evaluate_1d()
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
    """The (4, 6, 4) p=2 spline torus of the default (CPU) tier."""
    return build_torus_sequence(NS_TINY, torus_map)


@pytest.fixture(scope="session")
def torus_seq(torus_map):
    """The (8, 16, 8) p=2 spline torus of the ``gpu`` tier."""
    return build_torus_sequence(NS, torus_map)


# ---------------------------------------------------------------------------
# Small helpers usable from any test
# ---------------------------------------------------------------------------

def n_dofs(seq, k, dirichlet):
    """Return the DOF count for k-forms with the given boundary condition."""
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


@pytest.fixture(scope="session")
def laplacian_jacobi_diag(tiny_seq):
    """``diag(E L_k E^T)`` for every ``(k, dirichlet)`` on ``tiny_seq``, built once.

    ``kind='jacobi'`` Laplacian applies build this diagonal lazily and store
    it nowhere, so every consumer used to rebuild it (the k >= 1 closed form
    costs a few seconds per pair on the CPU). The inverse diagonals are
    installed on the session bundle, where ``_laplacian_diaginv`` finds them,
    so the positivity test, the dispatch test and the probed-reference test
    share one build. Returns ``{(k, dirichlet): diagonal}`` as numpy arrays.
    """
    import numpy as np

    from mrx.local_assembly import build_extracted_stiffness_diagonal_k0
    from mrx.operators import _invert_diagonal
    from mrx.preconditioners import build_extracted_laplacian_diagonal

    ops = tiny_seq.operators
    diags = {}
    for k in range(4):
        for dbc in (False, True):
            if k == 0:
                diag = build_extracted_stiffness_diagonal_k0(tiny_seq, dbc)
            else:
                diag = build_extracted_laplacian_diagonal(
                    tiny_seq, ops, k, dirichlet=dbc)
            diags[(k, dbc)] = np.asarray(diag)
            ops = ops.with_laplacian_diaginv(k, _invert_diagonal(diag), dirichlet=dbc)
    tiny_seq.set_operators(ops)
    return diags


@pytest.fixture(scope="session")
def precond_jit(tiny_seq):
    """JIT-compiled and warmed-up mass preconditioner applies on ``tiny_seq``.

    Keyed by ``(label, k, dbc)``; the jacobi applies for every
    ``(k, dirichlet)`` pair are compiled once per session so the probe tests
    do not re-JIT.
    """
    from mrx.operators import apply_mass_matrix_preconditioner
    ops = tiny_seq.operators
    jit_dict = {}
    for k in range(4):
        for dbc in (False, True):
            jit_dict[("jacobi", k, dbc)] = jax.jit(
                lambda v, k=k, dbc=dbc: apply_mass_matrix_preconditioner(
                    tiny_seq, ops, v, k, dirichlet=dbc, kind="jacobi",
                )
            )
    for (_, k, dbc), fn in jit_dict.items():
        dummy = jnp.zeros(n_dofs(tiny_seq, k, dbc), dtype=mrx.DTYPE)
        jax.block_until_ready(fn(dummy))
    return jit_dict
