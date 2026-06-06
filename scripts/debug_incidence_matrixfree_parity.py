"""Regression + timing: matrix-free incidence ``G_k`` vs the old BCSR matvecs.

The topological incidence operators ``G0`` (grad), ``G1`` (curl), ``G2`` (div)
and their transposes are now applied matrix-free as {-1, 0, +1} difference
stencils (``_MatrixFreeIncidence``) instead of stored ``BCSR`` matrices. This
script checks two things on the GPU:

1. **Parity** — for each ``k`` and each of ``G_k`` / ``G_k^T``, the matrix-free
   apply matches the reference ``BCSR`` apply (rebuilt via
   ``_incidence_forward_bcoo``) to ~1e-13 on random vectors.

2. **Timing** — warm-up then timed loops over the *raw* ``G`` apply only
   (forward and transpose), comparing matrix-free vs BCSR, so we know how much
   wall-clock the change saves. The Jacobi diagonal probe is intentionally NOT
   timed here.

Run:
    python scripts/debug_incidence_matrixfree_parity.py
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import (
    _incidence_components,
    _incidence_forward_bcoo,
    _incidence_shapes,
    assemble_incidence_operators,
)

types = ("clamped", "periodic", "periodic")


def _time_apply(fn, x, *, warmup=5, iters=50):
    """Warm-up then median wall-clock per call (seconds)."""
    for _ in range(warmup):
        jax.block_until_ready(fn(x))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    n, p = 32, 3
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p
    print(f"JAX devices: {jax.devices()}")
    print(f"Building DeRhamSequence ns={ns} ps={ps} q={q}")

    seq = DeRhamSequence(ns, ps, q, types, polar=True)
    seq.set_map(toroid_map(epsilon=1 / 3))
    seq.evaluate_1d()

    ops = assemble_incidence_operators(seq, seq.get_operators(), ks=(0, 1, 2))

    incidence_types = tuple(seq.basis_0.types)
    s0, s1, s2, s3 = _incidence_shapes(seq)

    key = jax.random.PRNGKey(0)
    all_ok = True

    for k in (0, 1, 2):
        g, g_T = _incidence_components(ops, k)
        sp, sp_T = _incidence_forward_bcoo(k, incidence_types, s0, s1, s2, s3)
        sp = sp.to_bcoo()
        sp_T = sp_T.to_bcoo()

        n_out, n_in = g.shape
        key, k_in, k_out = jax.random.split(key, 3)
        x = jax.random.normal(k_in, (n_in,), dtype=jnp.float64)
        y = jax.random.normal(k_out, (n_out,), dtype=jnp.float64)

        fwd_mf = jax.jit(lambda v, g=g: g @ v)
        fwd_sp = jax.jit(lambda v, sp=sp: sp @ v)
        adj_mf = jax.jit(lambda v, g_T=g_T: g_T @ v)
        adj_sp = jax.jit(lambda v, sp_T=sp_T: sp_T @ v)

        err_fwd = float(jnp.max(jnp.abs(fwd_mf(x) - fwd_sp(x))))
        err_adj = float(jnp.max(jnp.abs(adj_mf(y) - adj_sp(y))))
        ok = err_fwd < 1e-12 and err_adj < 1e-12
        all_ok = all_ok and ok
        print(
            f"k={k}: shape G={g.shape}  "
            f"max|G_mf-G_bcsr|={err_fwd:.2e}  "
            f"max|GT_mf-GT_bcsr|={err_adj:.2e}  "
            f"{'OK' if ok else 'MISMATCH'}"
        )

        t_fwd_mf = _time_apply(fwd_mf, x)
        t_fwd_sp = _time_apply(fwd_sp, x)
        t_adj_mf = _time_apply(adj_mf, y)
        t_adj_sp = _time_apply(adj_sp, y)
        print(
            f"      fwd: mf={t_fwd_mf * 1e6:8.1f} us  "
            f"bcsr={t_fwd_sp * 1e6:8.1f} us  "
            f"speedup={t_fwd_sp / t_fwd_mf:5.2f}x"
        )
        print(
            f"      adj: mf={t_adj_mf * 1e6:8.1f} us  "
            f"bcsr={t_adj_sp * 1e6:8.1f} us  "
            f"speedup={t_adj_sp / t_adj_mf:5.2f}x"
        )

    print("\nALL PARITY CHECKS PASSED" if all_ok else "\nPARITY FAILURE")


if __name__ == "__main__":
    main()
