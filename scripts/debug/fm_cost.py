"""Storage and per-apply cost of the truncated-Fourier coarse correction.

Times the JITTED apply (after warm-up, with `block_until_ready`) and reports
the bytes actually held, so the `fm` path can be compared with the dense
outer-ring probe (`o1`) on the same footing rather than by flop counting.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax


import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_jacobi_spectrum import (build_sequence,  # noqa: E402
                                   make_preconditioner,
                                   row_labels as _labels)


def precond_bytes(pre):
    tot = 0
    parts = {}
    ci = getattr(pre, "core_inv", None)
    if ci is not None and np.asarray(ci).size:
        parts["core_inv"] = np.asarray(ci).nbytes
    co = getattr(pre, "coarse", None)
    if co is not None:
        parts["V"] = np.asarray(co[0]).nbytes
        parts["A0inv"] = np.asarray(co[1]).nbytes
        parts["LV"] = np.asarray(co[2]).nbytes
    for b in getattr(pre, "blocks", []) or []:
        if b is None:
            continue
        for v in b["atom"][0]:
            tot += np.asarray(v).nbytes
    parts["atom eigvecs"] = tot
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="rot-ellipse",
                    choices=("toroid", "rot-ellipse", "w7x"))
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--arms", default="ibpd_r3,ibpd_r3_fm2,ibpd_r3_fm3,"
                                      "ibpd_r3_fm3_fr2,ibpd_r3_o1")
    ap.add_argument("--reps", type=int, default=50)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    k, dbc = cli.k, False
    n = int(getattr(seq, f"n{k}"))
    x = jax.random.normal(jax.random.PRNGKey(0), (n,))
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} k={k} n={n}", flush=True)
    print(f"\n{'arm':22} {'build':>8} {'q':>5} {'apply':>10} "
          f"{'MB total':>10} {'MB coarse':>10} {'MB core':>10}", flush=True)

    for arm in cli.arms.split(","):
        t0 = time.perf_counter()
        apply, pre = make_preconditioner(seq, ops, k, dbc, arm)
        build = time.perf_counter() - t0
        apply(x).block_until_ready()          # compile
        t0 = time.perf_counter()
        for _ in range(cli.reps):
            y = apply(x)
        y.block_until_ready()
        per = (time.perf_counter() - t0) / cli.reps
        co = getattr(pre, "coarse", None) if pre is not None else None
        if co is not None:
            # Does L V DECAY away from the boundary? If it does, the dominant
            # storage (n x q dense) truncates to the outer few rings and fm
            # drops from O(n q) to O(n^{2/3} q). If it does not, that constant
            # (2.3 KB/DOF at q=147) is irreducible and fm cannot scale.
            lv = np.abs(np.asarray(co[2]))
            i_r, n_r, i_c, i_t, i_z, shapes = _labels(seq, k, dbc, n)
            depth = np.where(i_r < 0, -1, n_r - 1 - i_r)   # 0 == outer ring
            tot = (lv ** 2).sum()
            prof = []
            for d in range(0, 6):
                m = depth == d
                prof.append(float((lv[m] ** 2).sum() / tot) if m.any() else 0.0)
            deep = float((lv[(depth > 5) | (depth < 0)] ** 2).sum() / tot)
            print("    |LV|^2 by depth from the outer ring: "
                  + " ".join(f"d{d}={v:.3f}" for d, v in enumerate(prof))
                  + f"  deeper={deep:.3f}", flush=True)
        parts = precond_bytes(pre) if pre is not None else {}
        mb = sum(parts.values()) / 1e6
        print(f"{arm:22} {build:7.1f}s "
              f"{getattr(pre, 'n_coarse', 0):>5} {per * 1e3:8.2f}ms "
              f"{mb:10.1f} "
              f"{(parts.get('V', 0) + parts.get('LV', 0) + parts.get('A0inv', 0)) / 1e6:10.1f} "
              f"{parts.get('core_inv', 0) / 1e6:10.1f}", flush=True)


if __name__ == "__main__":
    main()
