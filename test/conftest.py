"""Shared pytest fixtures for the MRX test suite.

The design goal is **one expensive assembly per pytest session**: a single
moderately-resolved DeRham sequence on a nontrivial toroid is built once and
reused by every test that needs a sequence. Tests that genuinely require
different parameters (low-level quadrature / spline checks, etc.) build
their own tiny objects on the fly.
"""

import jax
import jax.numpy as jnp
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.geometry import greville_interpolate_map
from mrx.mappings import toroid_map
from mrx.operators import (
    assemble_derivative_operators,
    assemble_incidence_operators,
    assemble_mass_jacobi_preconditioner,
)

import mrx  # noqa: F401, E402  (selects the working precision from MRX_DTYPE)

# Betti numbers for a solid torus.
BETTI = (1, 1, 0, 0)

# Shared resolution. (r, chi, zeta) — higher in the periodic directions
# to resolve the azimuthal variation in the Poisson tests.
NS = (8, 16, 8)
N = NS[0]  # kept for legacy references in other tests
P = 2   # was 3: the matvec is O(N p^4), so this is ~5x on every solve in the
        # suite. Convergence-ORDER tests that depend on p must parametrise it
        # themselves rather than inherit this fixture default.
TYPES = ("clamped", "periodic", "periodic")

# Donut-torus parameters.
TORUS_EPSILON = 1 / 3
TORUS_R0 = 1.0


@pytest.fixture(scope="session")
def torus_map():
    """Analytical map of the reference cube onto a donut-shaped solid torus."""
    return toroid_map(epsilon=TORUS_EPSILON, R0=TORUS_R0)


@pytest.fixture(scope="session")
def torus_seq(torus_map):
    """One fully-assembled DeRham sequence on a **spline-interpolated donut torus**.

    Built exactly once per pytest session:

    1. the analytical ``toroid_map`` is interpolated to spline coefficients
       at the Greville points via :func:`greville_interpolate_map` (the
       production route);
    2. the coefficients are installed as the sequence geometry via
       ``set_spline_map``;
    3. the incidence operators are assembled on that spline geometry and
       harmonic nullspaces are populated via inverse iteration with
       ``betti_numbers = (1, 1, 0, 0)``.

    The donut geometry is chosen because the k=0 Poisson problem has a
    known analytical solution on it, so several tests can check convergence
    against it.
    """
    ns = NS
    ps = (P, P, P)

    seq = DeRhamSequence(
        ns, ps, P + 1, TYPES, polar=True,
        tol=1e-12, maxiter=1000,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()

    # Interpolate the analytical donut map at the Greville points and install
    # it as the spline geometry. Masses and projections are matrix-free from
    # that geometry; only the incidence operators are assembled.
    seq.set_spline_map(greville_interpolate_map(torus_map, seq))
    geometry = seq.geometry
    ops = assemble_incidence_operators(seq)           # G0, G1, G2 (matrix-free)
    ops = assemble_derivative_operators(seq, geometry, operators=ops)   # validates G_k
    # NO tensor Laplacian / tensor mass preconditioners here any more: both are
    # retired paths (production = raw_kron masses + Jacobi Laplacians) and their
    # CP/NTF fits plus core-Schur build dominated fixture setup. The tests that
    # still cover them live in test/experimental/ and assemble them there.
    # Jacobi mass diagonals (the production FALLBACK preconditioner). Nothing
    # assembles these implicitly; the jacobi arms in test_preconditioners.py
    # and test_sequence.py error with "Jacobi mass diagonal ... is not
    # assembled" without this. Session-once direct diagonal extraction.
    ops = assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    # No explicit eps: exercise the production shift (1e-4). It is fixed, not
    # mesh-scaled -- the discrete harmonic space sits exactly in ker(L), so the
    # only requirement is eps << lambda_1, which is O(1) here.
    seq._compute_nullspaces(BETTI)
    return seq


# ---------------------------------------------------------------------------
# Small helpers usable from any test
# ---------------------------------------------------------------------------

def n_dofs(seq, k, dirichlet):
    """Return the DOF count for k-forms with the given boundary condition."""
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


@pytest.fixture(scope="session")
def precond_jit(torus_seq):
    """JIT-compiled and warmed-up preconditioner applies, keyed by (label, k, dbc).

    Computes the jacobi applies for every (k, dirichlet) pair once per
    session. Tests that need fast repeated preconditioner calls index into this
    dict rather than re-JITting.
    """
    from mrx.operators import (
        apply_mass_matrix_preconditioner,
    )
    ops = torus_seq.operators
    jit_dict = {}
    for k in range(4):
        for dbc in (False, True):
            jit_dict[("jacobi", k, dbc)] = jax.jit(
                lambda v, k=k, dbc=dbc: apply_mass_matrix_preconditioner(
                    torus_seq, ops, v, k, dirichlet=dbc, kind="jacobi",
                )
            )
    # Warm up: pay all JIT compilation costs once here.
    for (_, k, dbc), fn in jit_dict.items():
        dummy = jnp.zeros(n_dofs(torus_seq, k, dbc), dtype=jnp.float64)
        jax.block_until_ready(fn(dummy))
    return jit_dict
