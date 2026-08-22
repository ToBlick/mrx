"""A/B the block-Jacobi MASS preconditioner against raw_kron.

Solves ``M_k x = b`` with CG for every ``(k, dirichlet)``.  The mass is SPD at
every degree and both boundary conditions, so unlike the Laplacians there are no
singular cases and no nullspace projection is needed.

Arms:
  none      -- unpreconditioned, for context.
  jacobi    -- 1/diag(M_k), the exact closed-form mass diagonal.
  raw_kron  -- production: Lam (A_r x A_t x A_z) Lam with the E+ pseudoinverse
               moving between raw and extracted coordinates.
  blockjac  -- the same separable bracket, but the polar rows are probed and
               inverted DENSELY instead of going through E+, and ``rN`` widens
               that exact region by N radial rings.

Usage:
    python scripts/debug/block_jacobi_mass_ab.py --geometry w7x --ns 8,16,8
"""
from __future__ import annotations

import argparse
import json
import os
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
from mrx.experimental.block_jacobi_laplacian import BlockJacobiMass  # noqa: E402
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    apply_mass_raw_kron_preconditioner, build_mass_raw_kron_factors)

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p, r_scale=1.0):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000, r_scale=r_scale,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "cylinder":
        seq.set_map(cylinder_map(a=0.33, h=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
        if not np.isfinite(np.asarray(seq.geometry.jacobian_j)).all():
            raise RuntimeError("W7-X geometry is degenerate")
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


def pcg(a_apply, b, minv, tol=1e-8, maxiter=5000):
    x = jnp.zeros_like(b)
    r = b
    z = minv(r)
    p = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    for it in range(1, maxiter + 1):
        ap = a_apply(p)
        pap = float(p @ ap)
        if pap == 0.0:
            return it, float(jnp.linalg.norm(r)) / nb
        alpha = rz / pap
        x = x + alpha * p
        r = r - alpha * ap
        if float(jnp.linalg.norm(r)) <= tol * nb:
            return it, float(jnp.linalg.norm(r)) / nb
        z = minv(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter, float(jnp.linalg.norm(r)) / nb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid",
                    choices=("toroid", "w7x", "cylinder", "rot-ellipse"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--r-scale", type=float, default=1.0)
    ap.add_argument("--arms", default="jacobi,raw_kron,blockjac,blockjac_r3")
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p, r_scale=cli.r_scale)
    arms = cli.arms.split(",")
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} MASS solve tol={cli.tol}",
          flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n':>7} " + " ".join(f"{a:>22}" for a in arms),
          flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "tol": cli.tol, "rows": []}
    for k in (int(v) for v in cli.ks.split(",")):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rhs = jax.random.normal(jax.random.PRNGKey(7 * k + dbc), (n,))
            record = {"k": k, "dbc": dbc, "n": n}
            cells = []

            def a_apply(x, k=k, dbc=dbc):
                return op.apply_mass_matrix(seq, ops, x, k, dirichlet=dbc)

            for arm in arms:
                t0 = time.perf_counter()
                try:
                    if arm == "none":
                        def minv(v):
                            return v
                    elif arm == "jacobi":
                        d = jnp.asarray(op._mass_diaginv(seq, ops, k, dbc))

                        def minv(v, d=d):
                            return d * v
                    elif arm == "raw_kron":
                        fac = build_mass_raw_kron_factors(seq, k, dirichlet=dbc)
                        e = getattr(seq, f"e{k}_dbc" if dbc else f"e{k}")

                        def minv(v, fac=fac, e=e):
                            return apply_mass_raw_kron_preconditioner(fac, e, v)
                    elif arm.startswith("blockjac"):
                        rings = (int(arm.rsplit("r", 1)[1])
                                 if arm.rstrip("0123456789").endswith("r")
                                 else 0)
                        pre = BlockJacobiMass(seq, ops, k, dbc,
                                              extra_rings=rings)
                        minv = pre.apply
                    else:
                        raise ValueError(arm)
                    t_build = time.perf_counter() - t0
                    it, rel = pcg(a_apply, rhs, minv, tol=cli.tol,
                                  maxiter=cli.maxiter)
                    cells.append(f"{t_build:7.1f}s {it:6d}it {rel:8.1e}")
                    record[arm] = {"build_s": t_build, "iters": it, "rel": rel}
                except Exception as exc:  # noqa: BLE001
                    cells.append(f"{type(exc).__name__}: {str(exc)[:38]}")
                    record[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{k:>2} {dbc!s:>5} {n:>7} " + " ".join(cells), flush=True)
            results["rows"].append(record)
            if cli.out:
                os.makedirs(os.path.dirname(os.path.abspath(cli.out)),
                            exist_ok=True)
                with open(cli.out, "w") as fh:
                    json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
