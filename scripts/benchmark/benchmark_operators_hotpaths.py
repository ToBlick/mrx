"""Wall-time benchmark for the hot paths owned by ``mrx.operators`` and
``mrx.local_assembly``: the matrix-free mass apply (every Krylov iteration),
the analytic polar grad/curl stencil builds, the k=0 extracted stiffness
diagonal (polar-row loop) and the sequential diagonal probe.

    python scripts/benchmark/benchmark_operators_hotpaths.py --what all

Timings are per call after a warm-up; the mass apply is timed over
``--reps`` calls with ``block_until_ready`` on the last one.
"""

import argparse
import time

import jax
import numpy as np

import mrx  # noqa: F401
from mrx.derham_sequence import DeRhamSequence
from mrx.extraction_operators import get_xi
from mrx.local_assembly import (
    build_codifferential_diagonal, build_extracted_stiffness_diagonal_k0,
    build_matrixfree_mass_apply, build_stiffness_diagonal,
)
from mrx.mappings import toroid_map
from mrx.operators import (
    _diagonal_from_matvec, assemble_incidence_operators,
    build_curl_stencil_g1, build_grad_stencil_g0, mass_core_apply,
)

P = 3
CASES = {"small": (8, 16, 8), "large": (16, 32, 16)}


def make_seq(ns, polar):
    seq = DeRhamSequence(ns, (P, P, P), 2 * P, ("clamped", "periodic", "periodic"),
                         polar=polar)
    seq.evaluate_1d()
    seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    return seq


def bench_mass(seq, reps):
    print(f"  {'k':>2} {'n_dof':>8} {'ms/apply':>10}")
    for k in (0, 1, 2, 3):
        apply = build_matrixfree_mass_apply(seq, k)
        n = sum(int(np.prod(s)) for s in getattr(seq, f"basis_{k}").shape)
        x = jax.random.normal(jax.random.PRNGKey(k), (n,), dtype=mrx.DTYPE)
        y = apply(x).block_until_ready()
        y = apply(x).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(reps):
            y = apply(x)
        y.block_until_ready()
        dt = (time.perf_counter() - t0) / reps
        print(f"  {k:>2} {n:>8} {1e3 * dt:>10.3f}")


def bench_stencil(seq):
    xi = get_xi(seq.ns[1])
    for name, fn in (("grad_g0", build_grad_stencil_g0),
                     ("curl_g1", build_curl_stencil_g1)):
        fn(seq, xi, False, False)
        t0 = time.perf_counter()
        for din in (False, True):
            for dout in (False, True):
                m = fn(seq, xi, din, dout)
                jax.block_until_ready(m.data)
        dt = (time.perf_counter() - t0) / 4
        print(f"  {name}: {1e3 * dt:.1f} ms/build  (nnz={int(m.nse)}, "
              f"shape={tuple(int(s) for s in m.shape)})")


def bench_k0_diag(seq):
    for dbc in (False, True):
        t0 = time.perf_counter()
        d = build_extracted_stiffness_diagonal_k0(seq, dbc)
        d.block_until_ready()
        dt = time.perf_counter() - t0
        print(f"  extracted_stiffness_diagonal_k0 dbc={dbc}: {1e3 * dt:.1f} ms "
              f"(n={d.shape[0]})")


def bench_diag_builders(seq):
    for k in (0, 1, 2):
        build_stiffness_diagonal(seq, k).block_until_ready()
        t0 = time.perf_counter()
        build_stiffness_diagonal(seq, k).block_until_ready()
        print(f"  build_stiffness_diagonal k={k}: {1e3 * (time.perf_counter() - t0):.1f} ms")
    build_codifferential_diagonal(seq, 3).block_until_ready()
    t0 = time.perf_counter()
    build_codifferential_diagonal(seq, 3).block_until_ready()
    print(f"  build_codifferential_diagonal k=3: {1e3 * (time.perf_counter() - t0):.1f} ms")


def bench_diag_probe(seq):
    ops = assemble_incidence_operators(seq)
    for k in (0, 1):
        apply = mass_core_apply(seq, ops, k)
        n = sum(int(np.prod(s)) for s in getattr(seq, f"basis_{k}").shape)
        _diagonal_from_matvec(apply, n).block_until_ready()
        t0 = time.perf_counter()
        _diagonal_from_matvec(apply, n).block_until_ready()
        dt = time.perf_counter() - t0
        print(f"  diagonal_from_matvec k={k} n={n}: {1e3 * dt:.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=("all", "mass", "stencil", "k0diag", "diag", "probe"))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cases", default="small,large")
    args = ap.parse_args()
    print("device:", jax.devices()[0], "dtype:", mrx.DTYPE)
    for case in args.cases.split(","):
        ns = CASES[case]
        print(f"\n=== {case} ns={ns} p={P} ===")
        if args.what in ("all", "mass"):
            seq = make_seq(ns, polar=False)
            print("mass apply (raw DOF space):")
            bench_mass(seq, args.reps)
        seq = make_seq(ns, polar=True)
        if args.what in ("all", "stencil"):
            print("polar stencil builds:")
            bench_stencil(seq)
        if args.what in ("all", "k0diag"):
            print("k=0 extracted stiffness diagonal:")
            bench_k0_diag(seq)
        if args.what in ("all", "diag"):
            print("closed-form diagonal builders:")
            bench_diag_builders(seq)
        if args.what in ("all", "probe"):
            print("sequential diagonal probe:")
            bench_diag_probe(seq)


if __name__ == "__main__":
    main()
