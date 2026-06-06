"""Validate the direct diagonal helpers against the probe-based versions.

Checks:
  * diag_EAET_direct(E, M)        vs diag_EAET(E, M, E_T)            (mass)
  * diag_EGtMGEt_direct(E, G, M)  vs diag_EAET_matvec(E, K_apply..)  (stiffness)

for both free and Dirichlet extraction, on a small polar toroidal sequence.

Run on GPU:
    python scripts/debug_diag_direct_vs_probe.py
"""
import jax
import jax.numpy as jnp

import mrx.operators as ops
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import (diag_EAET, diag_EAET_direct, diag_EAET_matvec,
                            diag_EGtMGEt_direct)

jax.config.update("jax_enable_x64", True)


def relerr(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    denom = jnp.linalg.norm(b)
    return float(jnp.linalg.norm(a - b) / jnp.where(denom == 0, 1.0, denom))


def main():
    n, p = 8, 2
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p
    types = ("clamped", "periodic", "periodic")

    seq = DeRhamSequence(ns, ps, q, types, polar=True, tol=1e-10, maxiter=100)
    seq.set_map(toroid_map(epsilon=1 / 3))
    seq.evaluate_1d()
    seq.assemble_mass_matrix(0)
    seq.assemble_mass_matrix(1)

    operators = ops.assemble_incidence_operators(
        seq, operators=seq.get_operators())
    g0 = operators.g0
    g0_T = operators.g0_T

    print("=== mass diag(E M0 E^T) ===")
    for tag, E, E_T, ndof in (
        ("free", seq.e0, seq.e0_T, seq.n0),
        ("dbc", seq.e0_dbc, seq.e0_dbc_T, seq.n0_dbc),
    ):
        probe = diag_EAET(E, seq.m0, E_T)
        direct = diag_EAET_direct(E, seq.m0)
        print(f"  {tag:4s}  n={ndof:5d}  relerr={relerr(direct, probe):.3e}  "
              f"max|d|={float(jnp.max(jnp.abs(direct - probe))):.3e}")

    print("=== stiffness diag(E G0^T M1 G0 E^T) ===")
    for tag, E, E_T, ndof in (
        ("free", seq.e0, seq.e0_T, seq.n0),
        ("dbc", seq.e0_dbc, seq.e0_dbc_T, seq.n0_dbc),
    ):
        def K_apply(v):
            return g0_T @ (seq.m1 @ (g0 @ v))
        probe = diag_EAET_matvec(E, K_apply, ndof, E_T)
        direct = diag_EGtMGEt_direct(E, g0, seq.m1)
        print(f"  {tag:4s}  n={ndof:5d}  relerr={relerr(direct, probe):.3e}  "
              f"max|d|={float(jnp.max(jnp.abs(direct - probe))):.3e}")


if __name__ == "__main__":
    main()
