"""Phase 3 validation gate for the raw_kron mass preconditioner.

Closes the gates left open in docs/research/mass_preconditioner_pivot.md:

  * **k=2 and k=3** -- previously unmeasured. k=2 has 2 n_z coupled rows and
    ``g_ij/J`` weights; k=3 has zero coupled rows, so its pseudoinverse
    degenerates to ``E^T`` and raw_kron is a plain tensor block.
  * **A stellarator geometry** -- both geometries measured so far are
    axisymmetric. The rotating ellipse needs adequate zeta resolution *and* a
    positive projected Jacobian; section 7.5 records a run where the projected
    geometry folded and the test measured nothing, so the Jacobian sign is
    checked before solving and the case is skipped loudly if it fails.
  * **GPU** -- all previous timings are CPU.

Iteration counts are CG to relative residual 1e-10, which is the currency the
pivot is decided on (section 2: the matrix-free mass apply dominates so heavily
that preconditioner apply cost is not a design variable).

Usage:
    python -u scripts/debug/raw_kron_mass_gate.py [--ns 8 16 8] [--p 3]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.local_assembly import build_mass_diagonal, build_matrixfree_mass_apply
from mrx.mappings import one_size_fits_all_map, rotating_ellipse_map, toroid_map
from mrx.preconditioners import (build_mass_jacobi_pair,
                                 build_mass_raw_kron_preconditioner)

p = argparse.ArgumentParser()
p.add_argument("--ns", type=int, nargs=3, default=(8, 16, 8))
p.add_argument("--p", type=int, default=3)
p.add_argument("--tol", type=float, default=1e-10)
p.add_argument("--maxit", type=int, default=3000)
args = p.parse_args()

NS, P = tuple(args.ns), args.p
print(f"raw_kron mass gate: ns={NS} p={P} tol={args.tol:g}", flush=True)
print(f"backend={jax.default_backend()} devices={jax.devices()}\n", flush=True)


def pcg(A, b, Minv, tol, maxit):
    """Plain PCG returning the iteration count (host-synced; this is a probe)."""
    x = jnp.zeros_like(b)
    r = b - A(x)
    z = Minv(r)
    q = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    for i in range(1, maxit + 1):
        Aq = A(q)
        denom = float(q @ Aq)
        if denom <= 0.0:
            return -i                       # breakdown: operator not SPD here
        a = rz / denom
        x = x + a * q
        r = r - a * Aq
        if float(jnp.linalg.norm(r)) / nb < tol:
            return i
        z = Minv(r)
        rz_new = float(r @ z)
        q = z + (rz_new / rz) * q
        rz = rz_new
    return maxit


GEOMETRIES = [
    ("toroid", lambda: toroid_map(epsilon=1 / 3, R0=1.0)),
    ("elongated k=2", lambda: one_size_fits_all_map(epsilon=0.33, kappa=2.0)),
    ("rot-ellipse nfp3", lambda: rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3)),
]

for geo_name, geo_fn in GEOMETRIES:
    print(f"===== {geo_name} =====", flush=True)
    seq = DeRhamSequence(NS, (P,) * 3, 2 * P, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.set_map(geo_fn())

    # Gate on the Jacobian sign before solving (section 7.5).
    jac = np.asarray(seq.geometry.jacobian_j)
    if jac.min() <= 0.0:
        print(f"  SKIP: det(J) in [{jac.min():.3e}, {jac.max():.3e}] -- the "
              f"geometry folds at this resolution; measurement would be "
              f"meaningless\n", flush=True)
        continue
    print(f"  det(J) in [{jac.min():.3e}, {jac.max():.3e}]", flush=True)

    print(f"  {'k':>2} {'BC':>5} {'n':>7} {'jacobi':>8} {'raw_kron':>8} "
          f"{'ratio':>7} {'setup s':>8} {'apply ms':>9}", flush=True)
    for k in range(4):
        ap = build_matrixfree_mass_apply(seq, k)
        d_raw = build_mass_diagonal(seq, k)
        jac_pair = build_mass_jacobi_pair(seq, ap, k)
        for dirichlet in (False, True):
            e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
            n = int(e.shape[0])
            A = jax.jit(lambda x, e=e, ap=ap: e @ ap(e.T @ x))

            t0 = time.perf_counter()
            p2 = build_mass_raw_kron_preconditioner(
                seq, k, dirichlet=dirichlet, d_raw=d_raw)
            rng = np.random.default_rng(0)
            b = jnp.asarray(rng.standard_normal(n))
            p2(b).block_until_ready()
            t_setup = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(20):
                p2(b).block_until_ready()
            t_apply = (time.perf_counter() - t0) / 20 * 1e3

            dj = jac_pair.dbc if dirichlet else jac_pair.free
            it_j = pcg(A, b, lambda r, dj=dj: dj * r, args.tol, args.maxit)
            it_2 = pcg(A, b, p2, args.tol, args.maxit)
            ratio = (it_j / it_2) if it_2 > 0 else float("nan")
            print(f"  {k:>2} {'dbc' if dirichlet else 'free':>5} {n:>7} "
                  f"{it_j:>8} {it_2:>8} {ratio:>6.1f}x {t_setup:>8.2f} "
                  f"{t_apply:>9.3f}", flush=True)
    print(flush=True)

print("negative iteration counts = CG breakdown (operator not SPD on that arm)")
