"""Poisson equation on a toroid.

Solve ``-Δu = f`` with ``u = 0`` on the boundary of an axisymmetric toroid
of major radius 1 and minor radius ``ε`` using 0-forms, and report the
relative L2 error against the exact solution

    u(r, θ, ζ) = (r² - r⁴) cos(2πζ) / 4,

whose source is, with ``R = 1 + ε r cos 2πθ``,

    f = cos(2πζ) [ -(1 - 4r²)/ε² - (r/2 - r³) cos(2πθ)/(εR) + (r² - r⁴)/(4R²) ].

This is the geometry of the convergence tests in the MRX paper.

Usage:
    python scripts/tutorials/toroid_poisson.py --n 4 6 8 --p 1 2 3
"""
import argparse

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import toroid_map

ε = 1 / 3
π = jnp.pi


def u(x):
    r, _, ζ = x
    return (r**2 - r**4) * jnp.cos(2 * π * ζ) / 4 * jnp.ones(1)


def f(x):
    r, θ, ζ = x
    R = 1 + ε * r * jnp.cos(2 * π * θ)
    return jnp.cos(2 * π * ζ) * (
        -(1 - 4 * r**2) / ε**2
        - (r / 2 - r**3) * jnp.cos(2 * π * θ) / (ε * R)
        + (r**2 - r**4) / (4 * R**2)
    ) * jnp.ones(1)


def relative_l2_error(seq, u_h, u_exact):
    """||u_h - u|| / ||u|| in the L2 norm of the physical domain."""
    diff = jax.vmap(lambda x: u_exact(x) - u_h(x))(seq.quad.x)
    exact = jax.vmap(u_exact)(seq.quad.x)
    w = seq.quad.w * seq.jacobian_j
    return float(jnp.sqrt((diff**2).sum(1) @ w) / jnp.sqrt((exact**2).sum(1) @ w))


def solve(n, p):
    """Return the relative L2 error and the CG iteration count."""
    seq = DeRhamSequence((n, 2 * n, n), (p, p, p), p + 1,
                         ("clamped", "periodic", "periodic"), polar=True)
    seq.evaluate_1d()
    seq.set_map_and_preconditioners(toroid_map(epsilon=ε), ks=(0,), dirichlets=(True,))
    rhs = seq.load(f, 0, dirichlet=True)
    u_hat, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=True, return_info=True)
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0_dbc)
    return relative_l2_error(seq, u_h, u), abs(int(info))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2, 3])
    cli = ap.parse_args()
    print(f"{'p':>3s} {'n':>4s} {'error':>12s} {'iters':>6s}")
    for p in cli.p:
        for n in cli.n:
            err, iters = solve(n, p)
            print(f"{p:3d} {n:4d} {err:12.4e} {iters:6d}")


if __name__ == "__main__":
    main()
