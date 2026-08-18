"""Fixtures for the RETIRED preconditioner paths.

These tests cover ``mrx/experimental/`` -- the mass surgery/Schur path
(``kind='tensor'``) and the k=0 tensor-Hodge Laplacian preconditioner. Neither
is in production as of 2026-08-18 (production is raw_kron masses + Jacobi
Laplacians), but both are kept because the Schur path is genuinely better on
axisymmetric geometry, and untested kept code rots.

This directory is in ``norecursedirs``, so a bare ``pytest`` never collects it.
Run it explicitly:

    pytest test/experimental

The heavy assembly (CP/NTF fits, core Schur) lives ONLY here, so the production
suite no longer pays ~80 s of fixture setup to build preconditioners nothing
uses.
"""
import pytest

from mrx.operators import (assemble_tensor_laplacian_preconditioner,
                           assemble_tensor_mass_preconditioner)


@pytest.fixture(scope="session")
def torus_seq_tensor(torus_seq):
    """``torus_seq`` plus the retired tensor preconditioner payloads."""
    ops = torus_seq.get_operators()
    ops = assemble_tensor_laplacian_preconditioner(
        torus_seq, ops, ks=(0,), rank=1,
        cp_kwargs={"maxiter": 100, "tol": 1e-9, "ridge": 1e-12},
    )
    ops = assemble_tensor_mass_preconditioner(torus_seq, ops, ks=(0, 1, 2, 3), rank=3)
    torus_seq.set_operators(ops)
    return torus_seq


@pytest.fixture(scope="session")
def precond_jit_tensor(torus_seq_tensor):
    from mrx.operators import apply_mass_tensor_preconditioner_ops
    ops = torus_seq_tensor.operators
    jit_dict = {}
    for k in range(4):
        for dbc in (False, True):
            import jax
            fn = jax.jit(lambda v, k=k, dbc=dbc: apply_mass_tensor_preconditioner_ops(
                torus_seq_tensor, ops, v, k, dirichlet=dbc))
            jit_dict[("tensor", k, dbc)] = fn
    return jit_dict
