"""Poisson equation on the unit disk.

Solve ``-Δu = f`` with ``u = 0`` on the boundary of the unit disk using
0-forms on a polar spline space, and report the relative L2 error against
the exact solution

    u(r) = r³ (3 log r - 2) / 27 + 2 / 27,      f(r) = -r log r.

``u`` is in ``H^s`` only for ``s < 4``, which caps the convergence order.

Usage:
    python scripts/tutorials/polar_poisson.py --n 6 8 12 16 --p 1 2 3
"""
import argparse

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction


def disk_map(x):
    """Unit disk in the (x, z) plane, extruded along -y."""
    r, θ, z = x
    return jnp.array([r * jnp.cos(2 * jnp.pi * θ), -z, r * jnp.sin(2 * jnp.pi * θ)])


def u(x):
    r, _, _ = x
    return jnp.ones(1) * (r**3 * (3 * jnp.log(r) - 2) / 27 + 2 / 27)


def f(x):
    r, _, _ = x
    return -jnp.ones(1) * r * jnp.log(r)


def relative_l2_error(seq, u_h, u_exact):
    """||u_h - u|| / ||u|| in the L2 norm of the physical domain."""
    diff = jax.vmap(lambda x: u_exact(x) - u_h(x))(seq.quad.x)
    exact = jax.vmap(u_exact)(seq.quad.x)
    w = seq.quad.w * seq.jacobian_j
    return float(jnp.sqrt((diff**2).sum(1) @ w) / jnp.sqrt((exact**2).sum(1) @ w))


def solve(n, p):
    """Return the relative L2 error and the CG iteration count."""
    seq = DeRhamSequence((n, n, 1), (p, p, 0), p + 1,
                         ("clamped", "periodic", "constant"), polar=True)
    seq.set_map_and_preconditioners(disk_map, ks=(0,), dirichlets=(True,))
    rhs = seq.load(f, 0, dirichlet=True)
    u_hat, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=True, return_info=True)
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0_dbc)
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
