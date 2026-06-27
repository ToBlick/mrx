"""A-vs-B structure comparison for the W7-X mass coefficient fields.

For each diagonal weight, at p in {3,4,5}, compare the relative CP fit residual of
  (A) symmetric CP rank-2:        sum_{k=1,2} a_k(r) b_k(theta) c_k(zeta)
  (B) factor-out-r:               a(r) * [ b_1(t) c_1(z) + b_2(t) c_2(z) ]
i.e. rank-1 in r times a rank-2 theta-zeta cross-section (exactly invertible
with 1-D ops only). Also report the rank-1-in-r residual alone and the 2-D rank
of the cross-section, to see whether rank-2 in the theta-zeta plane suffices.

Tensor axis order (per _quadrature_tensor_shape): mode 0 = theta, 1 = r, 2 = zeta.
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
    _cp_als_3tensor,
    _reshape_quadrature_scalar_field,
)

# hard = grows-with-r (rR or r/R); easy = 1/r-type
KIND = {
    "alpha_rr": "HARD", "alpha_thetatheta": "easy", "alpha_zetazeta": "HARD",
    "beta_rr": "easy", "beta_thetatheta": "HARD", "beta_zetazeta": "easy",
    "M0_J": "HARD", "M3_invJ": "easy",
}


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def compare(tensor, name):
    T = np.asarray(tensor, dtype=np.float64)
    nt, nr, nz = T.shape  # (theta, r, zeta)

    # (A) symmetric CP rank-2 (best unconstrained ALS)
    _, _, relA, _, _ = _cp_als_3tensor(
        tensor, 2, maxiter=800, tol=1e-13, ridge=1e-10)

    # (B) factor-out-r: rank-1 in r, then rank-2 theta-zeta cross-section
    Ur = np.moveaxis(T, 1, 0).reshape(nr, nt * nz)   # (r, theta*zeta)
    ur, sr, vr = np.linalg.svd(Ur, full_matrices=False)
    a_r = ur[:, 0]
    S = (sr[0] * vr[0]).reshape(nt, nz)              # cross-section pattern
    rank1r = np.einsum("r,tz->trz", a_r, S)
    rel_rank1r = rel(rank1r, T)
    # rank-2 cross-section of S
    us, ss, vs = np.linalg.svd(S, full_matrices=False)
    S2 = (us[:, :2] * ss[:2]) @ vs[:2]
    relB = rel(np.einsum("r,tz->trz", a_r, S2), T)
    cross_rk = lambda thr: int((ss / ss[0] > thr).sum())

    print(f"  {name:16} [{KIND.get(name,'?'):4}]  "
          f"A(symCP2)={relA:.2e}  B(r*[2D-rk2])={relB:.2e}  "
          f"(rank1-r={rel_rank1r:.2e}, cross 2D-rank @1%={cross_rk(1e-2)} @0.1%={cross_rk(1e-3)})")


for p in (3, 4, 5):
    args = SimpleNamespace(ns=(12, 24, 12), p=p, geometry="w7x",
                           cg_tol=1e-10, cg_maxiter=10000,
                           epsilon=1.0 / 3.0, kappa=1.0, r0=1.0, nfp=3)
    print(f"\n=== p={p} ===", flush=True)
    seq = build_sequence(args)
    jac = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    compare(jac, "M0_J")
    compare(1.0 / jac, "M3_invJ")
    for k, t in _k1_diagonal_metric_tensors(seq).items():
        compare(t, k)
    for k, t in _k2_diagonal_metric_tensors(seq).items():
        compare(t, k)
print("\ndone.", flush=True)
