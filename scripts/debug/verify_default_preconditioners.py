"""Full verification of the default preconditioners on the UNSHIFTED Laplacian.

Two stages, all eight ``(k, dirichlet)`` cases:

1. **Nullspaces by the DIRECT route** (:func:`mrx.nullspace.compute_nullspaces`
   -- Hodge decomposition, a fixed pair of solves per form, no inverse
   iteration and no spectral-gap dependence). Checked for the expected count,
   for ``||L_k v|| / ||v||`` actually vanishing, and for orthonormality.

2. **Poisson solves with eps = 0**, i.e. ``L_k x = b`` and not
   ``(L_k + eps M_k) x = b``. Four of the eight cases are SINGULAR there --
   with betti ``(1,1,0,0)`` the harmonic slots are ``(0,free)``, ``(1,free)``,
   ``(2,dbc)``, ``(3,dbc)`` -- so the right-hand side has to be projected onto
   the range first. ``L`` is symmetric, so ``range(L) = null(L)^perp`` in the
   Euclidean pairing, which is the projection applied here.

Reports, per case: CG iterations, the true relative residual
``||L x - b|| / ||b||`` recomputed outside the solver, and how much of the
solution leaked into the kernel.

Usage:
    python scripts/debug/verify_default_preconditioners.py --geometry w7x
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.nullspace import compute_nullspaces, get_nullspace  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "w7x":
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
        jac = np.asarray(seq.geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0:
            raise RuntimeError(
                f"W7-X geometry is degenerate: finite={np.isfinite(jac).all()} "
                f"min(jac)={jac.min():.3e}")
    else:
        raise ValueError(geometry)
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


def project_off_kernel(b, vecs):
    """Euclidean projection of ``b`` onto ``span(vecs)^perp``.

    ``L`` is symmetric so ``range(L) = null(L)^perp``; this is exactly the
    compatibility condition for the singular solves. Done through the Gram
    system rather than assuming the stored vectors are Euclidean-orthonormal
    (they are normalised in L2, i.e. against the mass).
    """
    if vecs is None or vecs.shape[0] == 0:
        return b, 0.0
    v = np.asarray(vecs)
    gram = v @ v.T
    coef = np.linalg.solve(gram, v @ np.asarray(b))
    removed = coef @ v
    return jnp.asarray(np.asarray(b) - removed), float(np.linalg.norm(removed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--nrhs", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    # The nullspace construction runs its own inner solves through seq.maxiter,
    # so the budget has to be raised BEFORE compute_nullspaces, not just for the
    # Poisson stage: W7-X k=1 needs >1000 iterations even with an exact Jacobi
    # diagonal, and a truncated inner solve silently poisons the harmonic form.
    seq.maxiter = cli.maxiter
    # NOT cli.tol: that is the OUTER CG tolerance. seq.tol bounds the
    # sequence's INNER solves, and the k=3 dbc harmonic form is a raw
    # M_3^-1 solve -- at tol=1e-10 its ||Lv||/||v|| is 1.3e-5 against
    # 2.6e-8 at 1e-12, since the residual is the solve error times
    # ||L_3||.
    seq.tol = 1e-13
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} betti={seq.betti_numbers} "
          f"tol={cli.tol} UNSHIFTED (eps=0)", flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "eps": 0.0, "tol": cli.tol, "nullspace_route": "direct",
               "nullspaces": [], "solves": []}

    # ---- stage 1: nullspaces, direct route ------------------------------- #
    t0 = time.perf_counter()
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    t_null = time.perf_counter() - t0
    print(f"\ncompute_nullspaces (direct) took {t_null:.1f}s", flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n_null':>7} {'rayleigh q':>13} {'rel L2 err':>12} "
          f"{'orthonormality':>15}", flush=True)

    kernels = {}
    for k in range(4):
        for dbc in (False, True):
            vecs = np.asarray(get_nullspace(ops, k, dbc))
            kernels[(k, dbc)] = vecs if vecs.shape[0] else None
            if vecs.shape[0] == 0:
                print(f"{k:>2} {dbc!s:>5} {0:>7} {'--':>15} {'--':>15}",
                      flush=True)
                results["nullspaces"].append(
                    {"k": k, "dbc": dbc, "n": 0})
                continue
            worst_res, gram = 0.0, vecs @ vecs.T
            for v in vecs:
                lv = op.apply_hodge_laplacian(seq, ops, jnp.asarray(v), k,
                                              dirichlet=dbc)
                mv = op.apply_mass_matrix(seq, ops, jnp.asarray(v), k,
                                          dirichlet=dbc)
                worst_res = max(worst_res, abs(float(v @ lv)) / float(v @ mv))
            orth = float(np.abs(gram / np.diag(gram)[:, None]
                                - np.eye(vecs.shape[0])).max())
            print(f"{k:>2} {dbc!s:>5} {vecs.shape[0]:>7} {worst_res:>15.3e} "
                  f"{orth:>15.3e}", flush=True)
            results["nullspaces"].append(
                {"k": k, "dbc": dbc, "n": int(vecs.shape[0]),
                 "residual": worst_res, "orthonormality": orth})

    # ---- stage 2: unshifted Poisson solves ------------------------------- #
    print(f"\nUnshifted solves, {cli.nrhs} right-hand sides per case",
          flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n':>7} {'sing':>5} {'iters':>7} {'cvg':>4} "
          f"{'rel resid':>12} {'kernel leak':>12} {'s':>7}", flush=True)

    for k in range(4):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            vecs = kernels[(k, dbc)]
            for j in range(cli.nrhs):
                key = jax.random.PRNGKey(1000 * k + 10 * dbc + j)
                b = jax.random.normal(key, (n,))
                b, _ = project_off_kernel(b, vecs)
                nb = float(jnp.linalg.norm(b))

                t0 = time.perf_counter()
                out = op.apply_inverse_hodge_laplacian(
                    seq, ops, b, k, dirichlet=dbc, tol=cli.tol,
                    maxiter=cli.maxiter, return_info=True)
                dt = time.perf_counter() - t0
                x, info = (out if isinstance(out, tuple) else (out, {}))

                # Recompute the residual OUTSIDE the solver: a solver-reported
                # residual can be preconditioned, deflated, or measured on a
                # saddle-point reformulation rather than on L x = b.
                lx = op.apply_hodge_laplacian(seq, ops, x, k, dirichlet=dbc)
                rel = float(jnp.linalg.norm(lx - b)) / nb
                leak = 0.0
                if vecs is not None:
                    leak = float(np.linalg.norm(np.asarray(vecs) @ np.asarray(x))
                                 / (np.linalg.norm(np.asarray(x)) + 1e-300))
                # preconditioned_cg encodes both facts in one scalar:
                # -k when converged, +k when it ran out of iterations.
                raw = int(np.asarray(info).reshape(-1)[0])
                iters, converged = abs(raw), raw < 0
                print(f"{k:>2} {dbc!s:>5} {n:>7} "
                      f"{'yes' if vecs is not None else 'no':>5} {iters:>7} "
                      f"{'y' if converged else 'N':>4} "
                      f"{rel:>12.3e} {leak:>12.3e} {dt:>7.1f}", flush=True)
                results["solves"].append(
                    {"k": k, "dbc": dbc, "n": n, "rhs": j,
                     "singular": vecs is not None, "iters": iters,
                     "converged": converged,
                     "rel_residual": rel, "kernel_leak": leak, "seconds": dt})
                if cli.out:
                    os.makedirs(os.path.dirname(os.path.abspath(cli.out)),
                                exist_ok=True)
                    with open(cli.out, "w") as fh:
                        json.dump(results, fh, indent=2)

    worst = max((r["rel_residual"] for r in results["solves"]), default=0.0)
    print(f"\nworst relative residual over all cases: {worst:.3e}", flush=True)


if __name__ == "__main__":
    main()
