"""A/B the block-Jacobi Laplacian preconditioner against the default Jacobi.

Unshifted (``eps = 0``) and, by default, only on the four NON-SINGULAR cases --
``(0,dbc)``, ``(1,free... )`` see ``--cases``. With betti ``(1,1,0,0)`` the
harmonic slots are ``(0,free)``, ``(1,free)``, ``(2,dbc)``, ``(3,dbc)``, so the
complement is what can be solved without projecting the right-hand side.

Arms:
  jacobi    -- the production closed-form Jacobi diagonal, 1/diag(L_k).
  blockjac  -- mrx.experimental.block_jacobi_laplacian: one separable
               three-term Kronecker-sum atom per component (fast
               diagonalisation) plus a densely-probed core, uncoupled.

Usage:
    python scripts/debug/block_jacobi_ab.py --geometry toroid --cases 1t,2f
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
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    BlockJacobiLaplacian)
from mrx.mappings import toroid_map  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p, r_scale=1.0):
    # r_scale grades the radial breakpoints as linspace(0,1,n)**r_scale.
    # 1.0 (default) is UNIFORM; < 1 fattens the innermost element (0.5 puts the
    # first breakpoint at 0.316 instead of ~0.1), which is the hard case for any
    # near-axis approximation; > 1 clusters toward the axis.
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000, r_scale=r_scale,
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


def pcg(a_apply, b, minv, tol=1e-8, maxiter=20000):
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
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--cases", default="0t,1t,2f,3f",
                    help="comma list of <k><f|t>, default = the non-singular four")
    ap.add_argument("--arms", default="jacobi,blockjac")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--r-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p, r_scale=cli.r_scale)
    arms = cli.arms.split(",")
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} r_scale={cli.r_scale} "
          f"UNSHIFTED tol={cli.tol}",
          flush=True)
    print(f"{'case':>6} {'n':>7} " + " ".join(f"{a:>22}" for a in arms),
          flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "eps": 0.0, "tol": cli.tol, "rows": []}
    for case in cli.cases.split(","):
        k, dbc = int(case[0]), case[1] == "t"
        n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
        rhs = jax.random.normal(jax.random.PRNGKey(41 * k + dbc), (n,))
        record = {"k": k, "dbc": dbc, "n": n}
        cells = []

        def a_apply(x, k=k, dbc=dbc):
            return op.apply_hodge_laplacian_approx(seq, ops, x, k, dirichlet=dbc)

        for arm in arms:
            t0 = time.perf_counter()
            try:
                if arm == "jacobi":
                    d = 1.0 / np.asarray(op._hodge_diaginv(seq, ops, k, dbc))
                    inv = jnp.asarray(1.0 / d)

                    def minv(v, inv=inv):
                        return inv * v
                elif arm == "xfer3":
                    from mrx.experimental.block_jacobi_laplacian import (
                        TransferK3Preconditioner)
                    pre = TransferK3Preconditioner(
                        seq, ops, dbc, ktilde_mode="honest", lumped="diag")
                    minv = pre.apply
                elif arm.startswith("blockjac"):
                    mode = ("honest" if "honest" in arm else "roundtrip")
                    pre = BlockJacobiLaplacian(seq, ops, k, dbc,
                                               ktilde_mode=mode,
                                               lumped=("diag" if "diag" in arm
                                                       else "lumped" in arm),
                                               extra_rings=int(
                                                   arm.split("r")[-1]
                                                   if arm.rstrip("0123456789")
                                                   .endswith("r") else 0),
                                               radial=("modal" if "modal" in arm
                                                       else "averaged"))
                    minv = pre.apply
                else:
                    raise ValueError(arm)
                t_build = time.perf_counter() - t0
                it, rel = pcg(a_apply, rhs, minv, tol=cli.tol,
                              maxiter=cli.maxiter)
                cells.append(f"{t_build:7.1f}s {it:6d}it {rel:8.1e}")
                record[arm] = {"build_s": t_build, "iters": it, "rel": rel}
            except Exception as exc:  # noqa: BLE001
                cells.append(f"{type(exc).__name__}: {str(exc)[:40]}")
                record[arm] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"{case:>6} {n:>7} " + " ".join(cells), flush=True)
        results["rows"].append(record)
        if cli.out:
            os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
            with open(cli.out, "w") as fh:
                json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
