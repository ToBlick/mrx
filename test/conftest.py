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
from mrx.io import project_sampled_field
from mrx.mappings import toroid_map

jax.config.update("jax_enable_x64", True)

# Betti numbers for a solid torus.
BETTI = (1, 1, 0, 0)

# Shared resolution. Kept modest so the session-scoped assembly is fast but
# still exercises a non-trivial (n, p) combination.
N = 4
P = 3
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
    """One fully-assembled DeRham sequence on a **spline-projected donut torus**.

    Built exactly once per pytest session:

    1. a reference-domain sequence is created and its reference mass matrix
       assembled;
    2. the analytical ``toroid_map`` is sampled on a logical grid and
       projected to spline coefficients via :func:`project_sampled_field`;
    3. the projected coefficients are installed as the sequence geometry
       via ``set_spline_map``;
    4. all sparse operators are assembled on that spline geometry and
       harmonic nullspaces are populated via inverse iteration with
       ``betti_numbers = (1, 1, 0, 0)``.

    The donut geometry is chosen because the k=0 Poisson problem has a
    known analytical solution on it, so several tests can check convergence
    against it.
    """
    ns = (N, N, N)
    ps = (P, P, P)

    # Step 1: reference-domain sequence (identity map).
    seq = DeRhamSequence(
        ns, ps, 2 * P, TYPES, lambda x: x, polar=True,
        tol=1e-12, maxiter=1000,
        betti_numbers=BETTI,
    )
    seq = DeRhamSequence(
        ns, ps, 2 * P, TYPES, polar=True,
        tol=1e-12, maxiter=1000,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()

    # Step 2: sample the analytical donut map and project to splines.
    F_ana = toroid_map(epsilon=TORUS_EPSILON, R0=TORUS_R0)
    n_sample = 40
    r = jnp.linspace(0.0, 1.0, n_sample)
    chi = jnp.linspace(0.0, 1.0, n_sample)
    zeta = jnp.linspace(0.0, 1.0, n_sample)
    ri, chii, zetai = jnp.meshgrid(r, chi, zeta, indexing="ij")
    grid_pts = jnp.stack([ri.ravel(), chii.ravel(), zetai.ravel()], axis=1)
    map_samples = jax.vmap(F_ana)(grid_pts)

    coeffs = jnp.stack([
        project_sampled_field(
            (r, chi, zeta), map_samples[:, i], seq,
            k=0, dirichlet=False, reference_domain=True,
        )
        for i in range(3)
    ], axis=0)

    # Step 3 + 4: install the spline geometry and assemble everything.
    seq.set_spline_map(coeffs)
    seq.assemble_all_sparse()
    seq._compute_nullspaces(BETTI, eps=1e-6)
    return seq


# ---------------------------------------------------------------------------
# Small helpers usable from any test
# ---------------------------------------------------------------------------

def build_dense(matvec, n):
    """Build a dense matrix from a matvec by probing with standard unit vectors."""
    eye = jnp.eye(n)
    return jax.vmap(matvec, in_axes=1, out_axes=1)(eye)
