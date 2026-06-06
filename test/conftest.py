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
from mrx.operators import (
    assemble_derivative_operators,
    assemble_incidence_operators,
    assemble_projection_operators,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
)

jax.config.update("jax_enable_x64", True)

# Betti numbers for a solid torus.
BETTI = (1, 1, 0, 0)

# Shared resolution. (r, chi, zeta) — higher in the periodic directions
# to resolve the azimuthal variation in the Poisson tests.
NS = (8, 16, 8)
N = NS[0]  # kept for legacy references in other tests
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
    ns = NS
    ps = (P, P, P)

    # Step 1: reference-domain sequence (identity map).
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

    # Step 3 + 4: install the spline geometry and assemble operators.
    # We do NOT call assemble_mass_operators: mass_core_apply is entirely
    # matrix-free (del operators) so BCSR mass matrices are never read.
    # We do NOT call assemble_hodge_operators: _assemble_hodge_block returns
    # (None, None, None) — it's a no-op — and its k=0 side-effect of building
    # the tensor Laplacian preconditioner with default params would conflict
    # with the explicit rank-1 build below.
    seq.set_spline_map(coeffs)
    geometry = seq.geometry
    ops = assemble_incidence_operators(seq)           # G0, G1, G2 (matrix-free)
    ops = assemble_derivative_operators(seq, geometry, operators=ops)   # validates G_k
    ops = assemble_projection_operators(seq, operators=ops)             # P21, P12, P03, P30
    # k=0 tensor Laplacian preconditioner (rank-1, same as the Poisson script).
    ops = assemble_tensor_laplacian_preconditioner(
        seq, ops, ks=(0,), rank=1,
        cp_kwargs={"maxiter": 100, "tol": 1e-9, "ridge": 1e-12},
    )
    # Tensor mass preconditioners for k=0–3 (needed by test_preconditioners.py).
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1, 2, 3), rank=3)
    seq.set_operators(ops)
    seq._compute_nullspaces(BETTI, eps=1e-6)

    # Pre-compute and cache stiffness-nullspace bases as attributes so any
    # test can read seq.stiffness_null[(k, dbc)] without repeating the
    # expensive vmap over CG solves.
    from mrx.nullspace import get_stiffness_nullspace
    ops = seq.operators
    seq.stiffness_null = {
        (1, False): get_stiffness_nullspace(seq, ops, 1, False),
        (2, True):  get_stiffness_nullspace(seq, ops, 2, True),
    }
    return seq


# ---------------------------------------------------------------------------
# Small helpers usable from any test
# ---------------------------------------------------------------------------

def build_dense(matvec, n):
    """Build a dense matrix from a matvec by probing with standard unit vectors."""
    eye = jnp.eye(n)
    return jax.vmap(matvec, in_axes=1, out_axes=1)(eye)


def n_dofs(seq, k, dirichlet):
    """Return the DOF count for k-forms with the given boundary condition."""
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


@pytest.fixture(scope="session")
def precond_jit(torus_seq):
    """JIT-compiled and warmed-up preconditioner applies, keyed by (label, k, dbc).

    Computes jacobi and tensor applies for every (k, dirichlet) pair once per
    session. Tests that need fast repeated preconditioner calls index into this
    dict rather than re-JITting.
    """
    from mrx.operators import (
        apply_mass_matrix_preconditioner,
        apply_mass_tensor_preconditioner_ops,
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
            jit_dict[("tensor", k, dbc)] = jax.jit(
                lambda v, k=k, dbc=dbc: apply_mass_tensor_preconditioner_ops(
                    torus_seq, ops, v, k, dirichlet=dbc,
                )
            )
    # Warm up: pay all JIT compilation costs once here.
    for (_, k, dbc), fn in jit_dict.items():
        dummy = jnp.zeros(n_dofs(torus_seq, k, dbc), dtype=jnp.float64)
        jax.block_until_ready(fn(dummy))
    return jit_dict
