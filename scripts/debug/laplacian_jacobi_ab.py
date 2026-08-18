"""Three-arm A/B for the k>=1 Laplacian Jacobi diagonal.

Arms, all feeding the SAME shifted-Jacobi preconditioner and the same CG:

  none   -- no preconditioner (context: is Jacobi doing anything at all?)
  probe  -- diag(L_k) by n_ext operator applies. Exact, O(N) applies, the
            oracle; unusable at production resolution.
  stiff  -- diag(E S_k E^T) only, i.e. the weak term DROPPED. The cheap
            fallback. Undefined at k=3, where S_3 = 0 and the Laplacian IS the
            weak term.
  closed -- the closed-form diagonal (stiffness closed form + tensorized weak
            term + exact applies on the coupled rows).

Usage:
    python scripts/debug/laplacian_jacobi_ab.py --geometry toroid --ns 8,16,8
    python scripts/debug/laplacian_jacobi_ab.py --geometry w7x    --ns 8,16,8
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

import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.local_assembly import build_stiffness_diagonal  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402

EPS = 1e-4


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


def stiffness_only_diagonal(seq, ops, k, dirichlet):
    """diag(E S_k E^T): bulk rows closed form, coupled rows by exact apply."""
    raw = np.asarray(build_stiffness_diagonal(seq, k))
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows, cols, vals = (np.asarray(e.rows), np.asarray(e.cols), np.asarray(e.vals))
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)
    diag = np.zeros(n_ext)
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * raw[cols[single]]
    coupled = np.flatnonzero(counts > 1)
    if coupled.size:
        size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
        op.apply_stiffness(seq, ops, jnp.zeros(size), k, dirichlet=dirichlet)
        diag[coupled] = np.asarray(jax.lax.map(
            lambda i: op.apply_stiffness(
                seq, ops, jnp.zeros(size).at[i].set(1.0), k, dirichlet=dirichlet)[i],
            jnp.asarray(coupled)))
    return diag


def pcg(a_apply, b, minv, tol=1e-8, maxiter=3000):
    x = jnp.zeros_like(b)
    r = b
    z = minv(r)
    p = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    for it in range(1, maxiter + 1):
        ap = a_apply(p)
        alpha = rz / float(p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        if float(jnp.linalg.norm(r)) <= tol * nb:
            return it
        z = minv(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--arms", default="none,probe,stiff,closed")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--out", default=None,
                    help="write the results to this JSON file")
    ap.add_argument("--build-only", action="store_true",
                    help="time the diagonal builds and compare them; skip CG. "
                         "This is the scaling question: the probe costs one "
                         "apply per extracted row, the closed form does not.")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    arms = cli.arms.split(",")
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} eps={EPS}", flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n':>7} " +
          " ".join(f"{a:>20}" for a in arms), flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "eps": EPS, "arms": arms, "build_only": cli.build_only,
               "rows": []}
    mass_jacobi = {}
    for k in (int(v) for v in cli.ks.split(",")):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rhs = jax.random.normal(jax.random.PRNGKey(97 * k + dbc), (n,))

            def a_apply(x, k=k, dbc=dbc):
                return (op.apply_hodge_laplacian_approx(seq, ops, x, k, dirichlet=dbc)
                        + EPS * op.apply_mass_matrix(seq, ops, x, k, dirichlet=dbc))

            mass_jacobi[(k, dbc)] = 1.0 / np.asarray(
                op._mass_diaginv(seq, ops, k, dbc))
            cells = []
            diags = {}
            record = {"k": k, "dbc": dbc, "n": n}
            for arm in arms:
                if arm == "none" and cli.build_only:
                    continue
                if arm == "stiff" and k == 3:
                    cells.append(f"{'n/a (S_3 = 0)':>20}")
                    record[arm] = None
                    continue
                t0 = time.perf_counter()
                if arm == "none":
                    shifted = jnp.ones(n)
                else:
                    if arm == "stiff":
                        diag = stiffness_only_diagonal(seq, ops, k, dbc)
                    else:
                        os.environ["MRX_LAPLACIAN_DIAG_PROBE"] = (
                            "1" if arm == "probe" else "0")
                        diag = 1.0 / np.asarray(op._hodge_diaginv(seq, ops, k, dbc))
                    shifted = jnp.asarray(1.0 / (diag + EPS * mass_jacobi[(k, dbc)]))

                def minv(v, d=shifted):
                    return d * v

                t_build = time.perf_counter() - t0
                if cli.build_only:
                    diags[arm] = np.asarray(shifted)
                    cells.append(f"{t_build:8.1f}s")
                    record[arm] = {"build_s": t_build}
                    continue
                it = pcg(a_apply, rhs, minv)
                cells.append(f"{t_build:8.1f}s {it:5d} it")
                record[arm] = {"build_s": t_build, "iters": it}
            if cli.build_only and {"probe", "closed"} <= set(diags):
                rel = np.abs(diags["closed"] - diags["probe"]) / np.abs(diags["probe"])
                record["closed_vs_probe"] = {
                    "median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)),
                    "max": float(rel.max())}
                cells.append(f"  relerr med={np.median(rel):.2e} "
                             f"p90={np.percentile(rel, 90):.2e} max={rel.max():.2e}")
            print(f"{k:>2} {dbc!s:>5} {n:>7} " + " ".join(cells), flush=True)
            results["rows"].append(record)
            if cli.out:
                os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
                with open(cli.out, "w") as fh:
                    json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
