"""Per-axis separability diagnostic for the W7-X diagonal-metric weight tensors.

For each vector-mass weight component (M1: alpha_* = J g^{ii};  M2: beta_* =
g_{ii}/J) on the quadrature grid, compute the singular-value decay of each
mode-k unfolding. Mode 0 = theta, 1 = r, 2 = zeta (per _quadrature_tensor_shape).

A low mode-k rank means that axis is near-separable and is the cheap one to
factor out first ( zeta-first / r-first strategy). Tells us whether a single
profile (rank-1) on one axis captures it, leaving the exact 2-term Kronecker
budget for the harder cross-section.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    _k1_diagonal_metric_tensors,
    _k2_diagonal_metric_tensors,
    _mode_unfold_3tensor,
    _reshape_quadrature_scalar_field,
)

args = SimpleNamespace(
    ns=(12, 24, 12), p=3, geometry="w7x",
    cg_tol=1e-10, cg_maxiter=10000,
    epsilon=1.0 / 3.0, kappa=1.0, r0=1.0, nfp=3,
)
print(f"building W7-X seq ns={args.ns} p={args.p} ...", flush=True)
seq = build_sequence(args)
print("built. computing per-axis SVD decay.\n", flush=True)

LABEL = {0: "theta", 1: "r", 2: "zeta"}


def report(tensor, name):
    T = np.asarray(tensor)
    pos = bool((T >= 0).all())
    print(f"{name}   shape={T.shape}   all>=0: {pos}   "
          f"dynamic range max/min = {T.max() / max(T.min(), 1e-300):.2e}")
    for mode in range(3):
        U = _mode_unfold_3tensor(tensor, mode)
        s = np.linalg.svd(np.asarray(U), compute_uv=False)
        s = s / s[0]

        def effrank(thr):
            return int((s > thr).sum())
        first = np.array2string(s[:5], precision=3, suppress_small=True)
        print(f"   mode {mode} ({LABEL[mode]:5}): "
              f"s2/s1={s[1]:.2e}  s3/s1={s[2]:.2e}  "
              f"rank@1%={effrank(1e-2)}  rank@0.1%={effrank(1e-3)}  "
              f"rank@1e-6={effrank(1e-6)}   sv[:5]={first}")
    print(flush=True)


# M0 weight = J (rR-like, polynomial in r); M3 weight = 1/J (rational, r in
# denominator -- need not share M1/M2's low r-rank). Check both directly.
jac = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
report(jac, "M0 J")
report(1.0 / jac, "M3 1/J")

for key, t in _k1_diagonal_metric_tensors(seq).items():
    report(t, f"M1 {key}")
for key, t in _k2_diagonal_metric_tensors(seq).items():
    report(t, f"M2 {key}")
print("done.", flush=True)
