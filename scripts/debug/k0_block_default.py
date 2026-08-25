"""Does the k=0 Laplacian solve actually take the block atom now, and is it better?

Until 2026-08-24 `_materialize_default_scalar_hodge_preconditioner` returned
`jacobi` unconditionally, so the two 'auto' paths for the SAME operator
disagreed: `apply_laplacian_preconditioner(kind='auto')` picked the block atom
at k=0 and the SOLVE path could not reach it. Audit item 3.1/3.7.

The default now consults the atom. This checks the three things that claim
buys, none of which were true before:

1. with no k=0 atom assembled the default is still `jacobi` -- so every
   existing caller is bit-identical and nothing was silently activated;
2. with the atom assembled the default becomes `block` and the solve runs;
3. both arms converge to the SAME vector, and the iteration counts say whether
   assembling a k=0 atom is worth it for a given call site.

(3) is the open question the fix deliberately does not answer for
`compute_nullspaces`: it does exactly ONE k=0 solve, inside
`apply_leray_projection(v, k=1)`, and an atom built for one solve is unlikely
to repay its assembly. That call site is left on `ks=(1,2,3)` until this says
otherwise.

    python scripts/debug/k0_block_default.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def solve(seq, ops, rhs, dirichlet, tol, maxiter):
    t0 = time.perf_counter()
    x, info = op.apply_inverse_hodge_laplacian(
        seq, ops, rhs, 0, dirichlet=dirichlet, tol=tol, maxiter=maxiter,
        return_info=True)
    x.block_until_ready()
    # minres/cg return info = -k converged, +k not (mrx/solvers.py).
    it = int(jnp.abs(jnp.asarray(info)))
    return x, it, int(jnp.asarray(info)) < 0, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--maxiter", type=int, default=200000)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    for dbc in (True, False):
        n = int(getattr(seq, "n0_dbc" if dbc else "n0"))
        rhs = jax.random.normal(jax.random.PRNGKey(3), (n,))

        # (1) no atom assembled -> the default must still be jacobi.
        assert not op._block_jacobi_available(seq, 0, dbc)
        pre = op._materialize_default_scalar_hodge_preconditioner(
            seq, ops, k=0, dirichlet=dbc)
        assert pre.kind == 'jacobi', pre.kind
        x_j, it_j, ok_j, t_j = solve(seq, ops, rhs, dbc, cli.tol, cli.maxiter)

        # (2) assemble it -> the default must flip to block.
        t0 = time.perf_counter()
        ops = op.assemble_block_jacobi_laplacian_preconditioner(
            seq, ops, ks=(0,), dirichlets=(dbc,))
        seq.set_operators(ops)
        t_assemble = time.perf_counter() - t0
        pre = op._materialize_default_scalar_hodge_preconditioner(
            seq, ops, k=0, dirichlet=dbc)
        assert pre.kind == 'block', pre.kind
        x_b, it_b, ok_b, t_b = solve(seq, ops, rhs, dbc, cli.tol, cli.maxiter)

        # (3) same operator, so the same answer.
        rel = float(jnp.linalg.norm(x_b - x_j) / jnp.linalg.norm(x_j))
        side = "dbc " if dbc else "free"
        print(f"[k0 {side}] n={n:7d}  jacobi {it_j:6d} it {t_j:6.2f}s "
              f"(converged {ok_j})   block {it_b:6d} it {t_b:6.2f}s "
              f"(converged {ok_b})   assemble {t_assemble:5.2f}s   "
              f"|dx|/|x| = {rel:.3e}", flush=True)
        if not (ok_j and ok_b):
            raise RuntimeError(f"k=0 {side} did not converge")
        if rel > 1e-8:
            raise RuntimeError(
                f"k=0 {side}: the two preconditioners disagree by {rel:.3e}; "
                "a converged solve must not depend on its preconditioner")
        breakeven = t_assemble / max(t_j - t_b, 1e-12)
        print(f"           break-even after {breakeven:.1f} solves "
              f"(negative means the atom never repays here)", flush=True)

        # Drop it so the next BC starts from the same state.
        if hasattr(seq, op.BLOCK_JACOBI_CACHE_ATTR):
            delattr(seq, op.BLOCK_JACOBI_CACHE_ATTR)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
