"""Mixed Poisson equation on the unit disk.

Solve ``-Δu = f`` on the unit disk in mixed form: ``u`` is a 3-form, the
flux ``σ = -∇u`` a 2-form, and the system is the 3-form Hodge Laplacian
``div M₂⁻¹ divᵀ``. The natural boundary condition of the 3-form Laplacian
is ``u = 0``, so no Dirichlet space is needed. The exact solution is

    u(r) = -(r⁴/16 - r³/12 + 1/48),      f(r) = r (r - 3/4),

with ``u(1) = 0`` and zero mean source.

Usage:
    python scripts/tutorials/mixed_polar_poisson.py --n 6 8 12 16 --p 1 2 3
"""
import argparse

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.nullspace import init_nullspaces


def disk_map(x):
    """Unit disk in the (x, z) plane, extruded along -y."""
    r, θ, z = x
    return jnp.array([r * jnp.cos(2 * jnp.pi * θ), -z, r * jnp.sin(2 * jnp.pi * θ)])


def u(x):
    r, _, _ = x
    return -jnp.ones(1) * (r**4 / 16 - r**3 / 12 + 1 / 48)


def f(x):
    r, _, _ = x
    return jnp.ones(1) * (r - 3 / 4) * r


def relative_l2_error(seq, u_h, u_exact):
    """||u_h - u|| / ||u|| in the L2 norm of the physical domain."""
    diff = jax.vmap(lambda x: u_exact(x) - u_h(x))(seq.quad.x)
    exact = jax.vmap(u_exact)(seq.quad.x)
    w = seq.quad.w * seq.jacobian_j
    return float(jnp.sqrt((diff**2).sum(1) @ w) / jnp.sqrt((exact**2).sum(1) @ w))


def solve(n, p):
    """Return the relative L2 error and the MINRES iteration count."""
    # A disk has Betti numbers (1, 0, 0, 0): no harmonic 3-forms, nothing to deflate.
    seq = DeRhamSequence((n, n, 1), (p, p, 0), p + 1,
                         ("clamped", "periodic", "constant"), polar=True,
                         betti_numbers=(1, 0, 0, 0))
    seq.evaluate_1d()
    # The k=3 solve is a saddle-point solve with M₂⁻¹ inside: build the
    # k=2 mass and the k=3 Laplacian preconditioners for the free space.
    seq.set_map_and_preconditioners(disk_map, ks=(2, 3), dirichlets=(False,))
    seq.set_operators(init_nullspaces(seq, seq.get_operators()))
    # A 3-form load carries no Jacobian: (rhs)_i = ∫ f Λ³_i dx̂.
    rhs = seq.load(f, 3)
    u_hat, info = seq.apply_inverse_laplacian(rhs, 3, dirichlet=False, return_info=True)
    # The pushforward of a 3-form divides by det DF.
    u_h = Pushforward(DiscreteFunction(u_hat, seq.basis_3, seq.e3), disk_map, 3)
    return relative_l2_error(seq, u_h, u), abs(int(info))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8, 12, 16])
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2, 3])
    cli = ap.parse_args()
    print(f"{'p':>3s} {'n':>4s} {'error':>12s} {'iters':>6s}")
    for p in cli.p:
        for n in cli.n:
            err, iters = solve(n, p)
            print(f"{p:3d} {n:4d} {err:12.4e} {iters:6d}")


if __name__ == "__main__":
    main()
