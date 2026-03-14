#!/usr/bin/env python3
"""
Simple B interpolation test: analytic B on [0,1]^3, dense solve only.

- Build DeRhamSequence with identity map, call assemble_all().
- B_dof = solve(m2, p2(B_exact)), evaluate at quad points, compute relative L2 error.
"""
import sys
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction

jax.config.update("jax_enable_x64", True)

"""
Analytic B field:
B = [sin(2πr), cos(2πθ), sin(2πζ)]
"""
def B_exact(x):
    r, t, z = x
    return jnp.array([
        jnp.sin(2 * jnp.pi * r),
        jnp.cos(2 * jnp.pi * t),
        jnp.sin(2 * jnp.pi * z),
    ])


def main():
    seq = DeRhamSequence(
        (6, 6, 6),
        (3, 3, 3),
        4,
        ("clamped", "periodic", "periodic"),
        lambda x: x,
        False,
        True,
    )
    seq.evaluate_1d()

    # I'm adding different options to help with compatibility with sparse scripts

    assemble = getattr(seq, "assemble_all", None)
    if assemble is None:
        assemble = getattr(seq, "assemble", None)
    if assemble is None:
        print("Skip: no assemble_all/assemble ", file=sys.stderr)
        sys.exit(0)
    assemble()

    m2 = getattr(seq, "m2", None)
    if m2 is None:
        m2 = getattr(seq, "M2", None)
    p2 = getattr(seq, "p2", None)
    if p2 is None:
        p2 = getattr(seq, "P2", None)
    if m2 is None or p2 is None:
        print("Skip: m2/M2 or p2/P2 not available", file=sys.stderr)
        sys.exit(0)

    rhs = p2(B_exact)
    B_dof = jnp.linalg.solve(m2, rhs)

    quad_rule = getattr(seq, "quad", None)
    if quad_rule is None:
        quad_rule = getattr(seq, "Q", None)
    if quad_rule is None:
        print("Skip: no quad/Q on sequence ", file=sys.stderr)
        sys.exit(0)
    xq = quad_rule.x
    w = quad_rule.w
    J_j = getattr(seq, "J_j", None)
    if J_j is None:
        J_j = getattr(seq, "jacobian_j", None)
    weights = w * jnp.ravel(J_j) if (J_j is not None and jnp.size(J_j) == w.size) else w

    basis_2 = getattr(seq, "basis_2", None)
    if basis_2 is None:
        basis_2 = getattr(seq, "Lambda_2", None)
    e2 = getattr(seq, "e2", None)
    if e2 is None:
        e2 = getattr(seq, "E2", None)
    if basis_2 is None or e2 is None:
        print("Skip: no basis_2/Lambda_2 or e2/E2 on sequence", file=sys.stderr)
        sys.exit(0)
    B_h = DiscreteFunction(B_dof, basis_2, e2)
    B_num = jax.vmap(B_h)(xq)
    B_ref = jax.vmap(B_exact)(xq)

    diff = B_num - B_ref
    num = jnp.sqrt(jnp.einsum("ij,ij,i->", diff, diff, weights))
    den = jnp.sqrt(jnp.einsum("ij,ij,i->", B_ref, B_ref, weights))
    err_rel = float(num / (den + 1e-14))

    print(f"Relative L2 interpolation error (analytic B, direct projection): {err_rel:.3e}")


if __name__ == "__main__":
    main()
