"""Quick CG-iteration smoke test for the k=2 stiffness tensor preconditioner."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.nullspace import get_nullspace, get_stiffness_nullspace
from mrx.operators import (
    apply_mass_matrix,
    apply_stiffness,
    apply_stiffness_tensor_preconditioner,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_stiffness_preconditioner,
    dense_stiffness_matrix,
)
from mrx.solvers import solve_singular_cg

jax.config.update("jax_enable_x64", True)

K = 2
BETTI = (1, 1, 0, 0)
DIRICHLET = False
SEED_BY_KIND = {
    "none": 0,
    "jacobi": 1,
    "tensor": 2,
}


def build_case():
    ns = (6, 12, 4)
    p = 3
    seq = DeRhamSequence(
        ns, (p, p, p), 2 * p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-9, maxiter=2000,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map(eps=0.3, kappa=1.2, R0=1.0, nfp=3))
    ops = assemble_mass_operators(seq, seq.geometry, ks=(2, 3))
    ops = assemble_incidence_operators(seq, operators=ops, ks=(1, 2))
    seq.operators = ops
    return seq, seq.operators


def assemble_for_rank(seq, base_ops, *, rank: int):
    ops = assemble_tensor_stiffness_preconditioner(
        seq,
        operators=base_ops,
        ks=(K,),
        rank=rank,
        cp_kwargs={
            "k2_rank": rank,
        },
    )
    seq.operators = ops
    return ops


def stiffness_jacobi_diaginv(seq, ops, *, dirichlet: bool) -> jnp.ndarray:
    matrix = dense_stiffness_matrix(seq, ops, K, dirichlet=dirichlet)
    diagonal = jnp.diag(matrix)
    return jnp.where(jnp.abs(diagonal) > 0.0, 1.0 / diagonal, 0.0)


def cg_iters(seq, ops, *, dirichlet: bool, kind: str) -> int:
    key = jax.random.PRNGKey(200 + 10 * int(dirichlet) + SEED_BY_KIND[kind])
    n = seq.n2_dbc if dirichlet else seq.n2
    trial = jax.random.normal(key, (n,), dtype=jnp.float64)
    rhs = apply_stiffness(seq, ops, trial, K, dirichlet=dirichlet)
    nullspace = get_stiffness_nullspace(seq, ops, K, dirichlet)

    if kind == "none":
        precond_apply = lambda x: x
    elif kind == "jacobi":
        diaginv = stiffness_jacobi_diaginv(seq, ops, dirichlet=dirichlet)
        precond_apply = lambda x, diaginv=diaginv: diaginv * x
    elif kind == "tensor":
        precond_apply = lambda x: apply_stiffness_tensor_preconditioner(
            seq,
            ops,
            x,
            K,
            dirichlet=dirichlet,
        )
    else:
        raise ValueError(f"Unknown preconditioner kind {kind!r}")

    _, info = solve_singular_cg(
        lambda x: apply_stiffness(seq, ops, x, K, dirichlet=dirichlet),
        rhs,
        mass_matvec=lambda x: apply_mass_matrix(seq, ops, x, K, dirichlet=dirichlet),
        precond_matvec=precond_apply,
        vs=nullspace,
        tol=seq.tol,
        maxiter=seq.maxiter,
    )
    return int(jnp.abs(info))


def main():
    seq, base_ops = build_case()
    print(f"n2={seq.n2}  n2_dbc={seq.n2_dbc}")
    harmonic_count = get_nullspace(base_ops, K, DIRICHLET).shape[0]
    if harmonic_count != 0:
        raise ValueError(
            f"This smoke script only supports the no-harmonic case; got {harmonic_count} harmonic k={K} vectors"
        )
    for rank in (1, 2, 3, 4):
        ops = assemble_for_rank(seq, base_ops, rank=rank)
        none_iter = cg_iters(seq, ops, dirichlet=DIRICHLET, kind="none")
        jac_iter = cg_iters(seq, ops, dirichlet=DIRICHLET, kind="jacobi")
        ten_iter = cg_iters(seq, ops, dirichlet=DIRICHLET, kind="tensor")
        print(
            f"rank={rank} free  none={none_iter:4d}  "
            f"jacobi={jac_iter:4d}  tensor={ten_iter:4d}"
        )


if __name__ == "__main__":
    main()