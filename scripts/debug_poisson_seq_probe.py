"""Interactive debug: reproduce the smallest failing Poisson case (n=8, p=1).

The multirun jobs crash immediately at the first n=8 case for ALL p (including
p=1/p=2 which previously succeeded), so the regression is in the
`sequential=True` dense-probe path of the k=0 tensor Laplacian preconditioner,
not the old vmap OOM. The SLURM result pickle swallows the real exception
(AttributeError on a str during unpickle), so run this directly on the GPU node
to see the actual traceback.

Run:
    python scripts/debug_poisson_seq_probe.py
"""
import traceback

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import (
    assemble_tensor_mass_preconditioner,
    assemble_tensor_laplacian_preconditioner,
)

types = ("clamped", "periodic", "periodic")


def main():
    n, p = 8, 1
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p
    print(f"JAX devices: {jax.devices()}")
    print(f"Building DeRhamSequence ns={ns} ps={ps} q={q}")

    seq = DeRhamSequence(ns, ps, q, types, polar=True)
    seq.set_map(toroid_map(epsilon=1 / 3))
    seq.evaluate_1d()

    print("assemble_tensor_mass_preconditioner(ks=(0,), rank=1) ...")
    ops = seq.set_operators(
        assemble_tensor_mass_preconditioner(
            seq, seq.get_operators(), ks=(0,), rank=1
        )
    )
    print("  mass preconditioner OK")
    assert seq.m0 is None, "M0 BCSR should NOT be assembled (matrix-free path)"
    print("  confirmed: no M0 BCSR assembled")

    print("assemble_tensor_laplacian_preconditioner(ks=(0,), rank=1) ...")
    try:
        ops = seq.set_operators(
            assemble_tensor_laplacian_preconditioner(
                seq, seq.get_operators(), ks=(0,), rank=1
            )
        )
        jax.block_until_ready(ops)
        print("  laplacian preconditioner OK")
    except Exception:
        print("  >>> laplacian preconditioner FAILED:")
        traceback.print_exc()
        return

    print("p0_dbc(f) + solve ...")

    def f(x):
        r, chi, z = x
        return jnp.ones(1) * jnp.cos(2 * jnp.pi * z)

    try:
        rhs = seq.p0_dbc(f)
        u_hat, info = seq.apply_inverse_laplacian(
            rhs, 0, dirichlet=True, return_info=True
        )
        jax.block_until_ready(u_hat)
        print("  solve OK, info:", int(info))
    except Exception:
        print("  >>> solve FAILED:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
