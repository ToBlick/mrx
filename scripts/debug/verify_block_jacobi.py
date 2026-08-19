"""Full 8-case verification with the block-Jacobi Laplacian preconditioner.

Same shape as ``verify_default_preconditioners.py`` -- nullspaces by the DIRECT
route, then UNSHIFTED (``eps = 0``) Poisson solves for all four degrees and both
boundary conditions -- but comparing the production Jacobi diagonal against the
block-Jacobi atom (separable bulk + densely-probed core).

The four singular cases (betti ``(1,1,0,0)`` puts harmonic forms at
``(0,free)``, ``(1,free)``, ``(2,dbc)``, ``(3,dbc)``) are handled by DEFLATION:
the right-hand side is projected onto ``range(L) = null(L)^perp`` and the
residual and preconditioned residual are re-projected every iteration, since
round-off otherwise feeds the kernel back in.

The operator is left alone -- raw_kron is still the weak term's inner inverse --
so these numbers are directly comparable with the earlier run.

Usage:
    python scripts/debug/verify_block_jacobi.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    BlockJacobiLaplacian)
from mrx.mappings import toroid_map  # noqa: E402
from mrx.nullspace import compute_nullspaces, get_nullspace  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p, maxiter, tol):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=tol, maxiter=maxiter,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
        jac = np.asarray(seq.geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0:
            raise RuntimeError("W7-X geometry is degenerate")
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


def make_projector(vecs):
    """Euclidean projector onto ``span(vecs)^perp`` (``L`` symmetric, so that
    is ``range(L)``). Identity when there is no kernel."""
    if vecs is None or vecs.shape[0] == 0:
        return lambda v: v
    v = np.asarray(vecs)
    gram = np.linalg.inv(v @ v.T)

    def proj(x):
        a = np.asarray(x)
        return jnp.asarray(a - (gram @ (v @ a)) @ v)
    return proj


def pcg(a_apply, b, minv, proj, tol=1e-10, maxiter=20000):
    """CG with deflation: the residual is kept in the range every iteration."""
    x = jnp.zeros_like(b)
    r = proj(b)
    z = proj(minv(r))
    p = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    for it in range(1, maxiter + 1):
        ap = proj(a_apply(p))
        pap = float(p @ ap)
        if pap <= 0.0:
            return it, float(jnp.linalg.norm(r)) / nb, False
        alpha = rz / pap
        x = x + alpha * p
        r = r - alpha * ap
        if float(jnp.linalg.norm(r)) <= tol * nb:
            return it, float(jnp.linalg.norm(r)) / nb, True
        z = proj(minv(r))
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter, float(jnp.linalg.norm(r)) / nb, False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--arms", default="jacobi,blockjac_r3")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter, cli.tol)
    arms = cli.arms.split(",")
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} UNSHIFTED tol={cli.tol}",
          flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "eps": 0.0, "tol": cli.tol, "nullspaces": [], "rows": []}

    t0 = time.perf_counter()
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"\ncompute_nullspaces (direct) {time.perf_counter() - t0:.1f}s",
          flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n_null':>7} {'||L v||/||v||':>15}", flush=True)
    kernels = {}
    for k in range(4):
        for dbc in (False, True):
            vecs = np.asarray(get_nullspace(ops, k, dbc))
            kernels[(k, dbc)] = vecs if vecs.shape[0] else None
            worst = 0.0
            for v in vecs:
                lv = op.apply_hodge_laplacian(seq, ops, jnp.asarray(v), k,
                                              dirichlet=dbc)
                worst = max(worst, float(jnp.linalg.norm(lv))
                            / float(np.linalg.norm(v)))
            print(f"{k:>2} {dbc!s:>5} {vecs.shape[0]:>7} "
                  f"{worst if vecs.shape[0] else float('nan'):>15.3e}",
                  flush=True)
            results["nullspaces"].append(
                {"k": k, "dbc": dbc, "n": int(vecs.shape[0]),
                 "residual": worst})

    print("\nUnshifted solves (deflated where singular)", flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n':>7} {'sing':>5} " +
          " ".join(f"{a:>26}" for a in arms), flush=True)
    for k in range(4):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            vecs = kernels[(k, dbc)]
            proj = make_projector(vecs)
            b = proj(jax.random.normal(jax.random.PRNGKey(31 * k + dbc), (n,)))
            record = {"k": k, "dbc": dbc, "n": n,
                      "singular": vecs is not None}
            cells = []

            def a_apply(x, k=k, dbc=dbc):
                return op.apply_hodge_laplacian_approx(seq, ops, x, k,
                                                       dirichlet=dbc)

            for arm in arms:
                t0 = time.perf_counter()
                try:
                    if arm == "jacobi":
                        d = jnp.asarray(op._hodge_diaginv(seq, ops, k, dbc))

                        def minv(v, d=d):
                            return d * v
                    else:
                        m = re.search(r"r(\d+)", arm)
                        o = re.search(r"o(\d+)", arm)
                        pre = BlockJacobiLaplacian(
                            seq, ops, k, dbc, ktilde_mode="honest",
                            lumped="diag",
                            extra_rings=int(m.group(1)) if m else 0,
                            outer_rings=int(o.group(1)) if o else 0,
                            bc_entry=(False if "nobc" in arm else "direct"),
                            radial=("modal" if "modal" in arm else "averaged"))
                        minv = pre.apply
                    t_build = time.perf_counter() - t0
                    it, rel, ok = pcg(a_apply, b, minv, proj, tol=cli.tol,
                                      maxiter=cli.maxiter)
                    cells.append(f"{t_build:6.1f}s {it:6d}it "
                                 f"{'y' if ok else 'N'} {rel:8.1e}")
                    record[arm] = {"build_s": t_build, "iters": it,
                                   "rel": rel, "converged": ok}
                except Exception as exc:  # noqa: BLE001
                    cells.append(f"{type(exc).__name__}: {str(exc)[:30]}")
                    record[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{k:>2} {dbc!s:>5} {n:>7} "
                  f"{'yes' if vecs is not None else 'no':>5} " +
                  " ".join(cells), flush=True)
            results["rows"].append(record)
            if cli.out:
                os.makedirs(os.path.dirname(os.path.abspath(cli.out)),
                            exist_ok=True)
                with open(cli.out, "w") as fh:
                    json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
