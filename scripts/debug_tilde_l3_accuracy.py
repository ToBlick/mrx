"""Compare the exact and approximate k = 3 Hodge-Laplacian applies.

This script measures how close

    L_3 v = D_2 M_2^{-1} D_2^T v

is to the linear Kronecker-mass approximation

    tilde(L_3) v = D_2 tilde(M_2)^{-1} D_2^T v

on a mapped toroidal geometry. It reports relative operator-application
errors, Rayleigh-quotient discrepancies, and output alignment on both
smooth projected test vectors and white-noise DoF vectors.

Run from the project root:

    .venv/bin/python scripts/debug_tilde_l3_accuracy.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
from benchmark_preconditioners import build_sequence

from mrx.nullspace import get_nullspace
from mrx.operators import apply_hodge_laplacian, apply_hodge_laplacian_approx

sys.path.insert(0, os.path.dirname(__file__))


jax.config.update("jax_enable_x64", True)


NUM_RANDOM = 8
COMPARE_DIRICHLET = (False, True)
SEED = 1234
ORTH_TOL = 1e-10


def _mass_deflate(seq, operators, batch, k, dirichlet):
    vs = get_nullspace(operators, k, dirichlet)
    if vs.shape[0] == 0:
        return batch

    def deflate_one(v):
        out = v
        for u in vs:
            coeff = u @ seq.apply_mass_matrix(
                out, k, dirichlet=dirichlet, operators=operators)
            out = out - coeff * u
        return out

    return jax.vmap(deflate_one)(batch)


def _normalize_l2(seq, batch, k, dirichlet):
    def normalize_one(v):
        nv = seq.l2_norm(v, k, dirichlet=dirichlet)
        safe = jnp.where(nv > 0, nv, 1.0)
        return v / safe

    return jax.vmap(normalize_one)(batch)


def _summarize(name, values):
    values = jnp.asarray(values)
    print(
        f"  {name:22s} mean={float(values.mean()):.6e} "
        f"std={float(values.std()):.3e} min={float(values.min()):.6e} "
        f"max={float(values.max()):.6e}"
    )


def _mass_inner(seq, operators, x, y, k, dirichlet):
    My = seq.apply_mass_matrix(y, k, dirichlet=dirichlet, operators=operators)
    return x @ My


def _mass_orthonormal_basis(seq, operators, batch, dirichlet, tol=ORTH_TOL):
    basis = []
    for vec in batch:
        work = vec
        for u in basis:
            work = work - _mass_inner(seq, operators,
                                      u, work, 3, dirichlet) * u
        norm = seq.l2_norm(work, 3, dirichlet=dirichlet)
        if float(norm) > tol:
            basis.append(work / norm)
    if not basis:
        return jnp.zeros((0, batch.shape[1]))
    return jnp.stack(basis)


def _projected_spectrum(seq, operators, basis, dirichlet, exact_apply, approx_apply):
    if basis.shape[0] == 0:
        return jnp.zeros((0,)), jnp.zeros((0, 0)), jnp.zeros((0, 0))

    exact_basis = jax.vmap(exact_apply)(basis)
    approx_basis = jax.vmap(approx_apply)(basis)

    def dense_from_images(images):
        return jax.vmap(
            lambda q: jax.vmap(lambda img: q @ img)(images)
        )(basis)

    exact_dense = dense_from_images(exact_basis)
    approx_dense = dense_from_images(approx_basis)
    exact_dense = 0.5 * (exact_dense + exact_dense.T)
    approx_dense = 0.5 * (approx_dense + approx_dense.T)

    L = jnp.linalg.cholesky(approx_dense)
    Y = jax.scipy.linalg.solve_triangular(L, exact_dense, lower=True)
    projected = jax.scipy.linalg.solve_triangular(L.T, Y.T, lower=False).T
    projected = 0.5 * (projected + projected.T)
    eigvals = jnp.linalg.eigvalsh(projected)
    return eigvals, exact_dense, approx_dense


def _run_case(seq, operators, batch, dirichlet, label):
    exact_apply = jax.jit(
        lambda v: apply_hodge_laplacian(
            seq, operators, v, 3, dirichlet=dirichlet)
    )
    approx_apply = jax.jit(
        lambda v: apply_hodge_laplacian_approx(
            seq, operators, v, 3, dirichlet=dirichlet)
    )

    exact_batch = jax.vmap(exact_apply)(batch)
    approx_batch = jax.vmap(approx_apply)(batch)

    def rel_output_err(exact, approx):
        denom = seq.l2_norm(exact, 3, dirichlet=dirichlet)
        safe = jnp.where(denom > 0, denom, 1.0)
        return seq.l2_norm(exact - approx, 3, dirichlet=dirichlet) / safe

    def rel_energy_err(v, exact, approx):
        num_exact = jnp.dot(v, exact)
        num_approx = jnp.dot(v, approx)
        denom = jnp.maximum(jnp.abs(num_exact), 1e-30)
        return jnp.abs(num_exact - num_approx) / denom

    def cosine_alignment(exact, approx):
        num = jnp.dot(exact, approx)
        denom = jnp.maximum(jnp.linalg.norm(exact) *
                            jnp.linalg.norm(approx), 1e-30)
        return num / denom

    rel_errs = jax.vmap(rel_output_err)(exact_batch, approx_batch)
    energy_errs = jax.vmap(rel_energy_err)(batch, exact_batch, approx_batch)
    cosines = jax.vmap(cosine_alignment)(exact_batch, approx_batch)
    rayleigh_ratios = jax.vmap(
        lambda v, exact, approx: (v @ approx) / jnp.maximum(v @ exact, 1e-30)
    )(batch, exact_batch, approx_batch)
    basis = _mass_orthonormal_basis(seq, operators, batch, dirichlet)
    spec_eigs, _, _ = _projected_spectrum(
        seq, operators, basis, dirichlet, exact_apply, approx_apply
    )


def main():
    print("Building sequence and operators...")
    seq, operators = build_sequence()
    print(
        f"resolution=(N={seq.ns[0]}, P={seq.ps[0]}) tol={seq.tol:.1e} "
        f"maxiter={seq.maxiter}"
    )

    key = jax.random.PRNGKey(SEED)
    for dirichlet in COMPARE_DIRICHLET:
        n_dof = seq.n3_dbc if dirichlet else seq.n3

        key, random_key = jax.random.split(key, 2)

        random_batch = jax.random.normal(random_key, (NUM_RANDOM, n_dof))
        random_batch = _mass_deflate(
            seq, operators, random_batch, 3, dirichlet)
        random_batch = _normalize_l2(seq, random_batch, 3, dirichlet)

        print()
        print(f"Case dirichlet={dirichlet} n_dof={n_dof}")
        _run_case(seq, operators, random_batch, dirichlet, "random dofs")


if __name__ == "__main__":
    main()
