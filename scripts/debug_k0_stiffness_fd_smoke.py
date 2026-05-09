"""Quick CG-iteration smoke test for the k=0 stiffness FD preconditioner."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.nullspace import _overwrite_nullspace_vector, init_nullspaces
from mrx.operators import (
    apply_inverse_hodge_laplacian,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    update_hodge_operator,
)

jax.config.update("jax_enable_x64", True)


def build_case():
    ns = (6, 12, 4)
    p = 3
    seq = DeRhamSequence(
        ns, (p, p, p), 2 * p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-9, maxiter=2000,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map(eps=0.3, kappa=1.2, R0=1.0, nfp=3))
    return seq


def assemble_for_rank(seq, *, rank: int):
    ops = assemble_mass_operators(seq, seq.geometry, ks=(0, 1))
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))
    ops = assemble_tensor_mass_preconditioner(
        seq, operators=ops, ks=(0,), rank=rank,
    )
    ops = update_hodge_operator(seq, seq.geometry, ops, 0)
    ops = init_nullspaces(seq, ops, betti_numbers=seq.betti_numbers)
    v0 = jnp.ones(seq.n0, dtype=jnp.float64)
    v0_norm = jnp.sqrt(v0 @ seq.apply_mass_matrix(v0, 0, dirichlet=False, operators=ops))
    v0 = v0 / v0_norm
    ops = _overwrite_nullspace_vector(ops, 0, False, 0, v0)
    seq.operators = ops
    return ops


def cg_iters(seq, ops, dirichlet, preconditioner):
    rng = jax.random.PRNGKey(0)
    n = seq.n0_dbc if dirichlet else seq.n0
    rhs = jax.random.normal(rng, (n,), dtype=jnp.float64)
    _, info = apply_inverse_hodge_laplacian(
        seq, ops, rhs, 0, dirichlet=dirichlet,
        preconditioner=preconditioner, return_info=True,
    )
    return int(info)


def main():
    seq = build_case()
    print(f"n0={seq.n0}  n0_dbc={seq.n0_dbc}")
    prev_denom_dbc = None
    prev_rank = None
    for rank in (1, 2, 3, 4):
        ops = assemble_for_rank(seq, rank=rank)
        denom_dbc = ops.k0_tensor_hodge_precond.dbc.bulk_modal_denom
        if prev_denom_dbc is not None:
            diff = jnp.abs(denom_dbc - prev_denom_dbc)
            rel = jnp.linalg.norm(diff) / jnp.maximum(jnp.linalg.norm(prev_denom_dbc), 1.0)
            print(
                f"rank {prev_rank}->{rank} dbc denom: "
                f"max_abs_diff={float(jnp.max(diff)):.6e} "
                f"rel_diff={float(rel):.6e}"
            )
        prev_denom_dbc = denom_dbc
        prev_rank = rank
        for dbc in (True, False):
            none_iter = cg_iters(seq, ops, dbc, "none")
            jac_iter = cg_iters(seq, ops, dbc, "jacobi")
            ten_iter = cg_iters(seq, ops, dbc, "tensor")
            tag = "dbc " if dbc else "free"
            print(
                f"rank={rank} {tag}  none={none_iter:4d}  "
                f"jacobi={jac_iter:4d}  tensor={ten_iter:4d}"
            )


if __name__ == "__main__":
    main()
