"""Regression + timing: matrix-free extraction ``E_k`` vs rebuilt BCSR matvecs.

The polar/boundary extraction operators ``E_k`` (and their transposes,
Dirichlet ``E_k^{dbc}`` and boundary-complement ``E_k^{bc}`` variants) are now
applied matrix-free as a cached gather/scatter over a static sparsity pattern
(``MatrixFreeExtraction``) instead of stored ``BCSR`` matrices. This script
checks four things on the GPU:

1. **Parity** — for each ``k`` and each of ``E_k`` / ``E_k^T`` (plain and
   ``_dbc``), the matrix-free apply matches the reference ``BCSR`` apply
   (rebuilt from the assembled BCOO) to ~1e-13 on random vectors.

2. **to_bcoo round-trip** — verify ``to_bcoo()`` reproduces the dense operator
   exactly (both orientations).

3. **Restriction** — verify ``restrict_rows`` / ``restrict_cols`` match dense
   row/column sub-selection (the surgery preconditioner path).

4. **Timing** — warm-up then timed loops over the raw ``E`` apply only
   (forward and transpose), matrix-free vs BCSR.

Run:
    python scripts/debug_extraction_matrixfree_parity.py
"""
import time

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map

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

    key = jax.random.PRNGKey(0)
    all_ok = True

    for k in range(4):
        for tag in ("", "_dbc"):
            e = getattr(seq, f"e{k}{tag}")
            e_T = getattr(seq, f"e{k}{tag}_T")

            # Reference BCSR rebuilt from the matrix-free sparsity pattern.
            sp = jsparse.BCSR.from_bcoo(e.to_bcoo())
            sp_T = jsparse.BCSR.from_bcoo(e_T.to_bcoo())

            n_out, n_in = e.shape
            key, k_in, k_out = jax.random.split(key, 3)
            x = jax.random.normal(k_in, (n_in,), dtype=jnp.float64)
            y = jax.random.normal(k_out, (n_out,), dtype=jnp.float64)

            fwd_mf = jax.jit(lambda v, e=e: e @ v)
            fwd_sp = jax.jit(lambda v, sp=sp: sp @ v)
            adj_mf = jax.jit(lambda v, e_T=e_T: e_T @ v)
            adj_sp = jax.jit(lambda v, sp_T=sp_T: sp_T @ v)

            err_fwd = float(jnp.max(jnp.abs(fwd_mf(x) - fwd_sp(x))))
            err_adj = float(jnp.max(jnp.abs(adj_mf(y) - adj_sp(y))))

            # to_bcoo round-trip against a densified reference.
            dense_ref = sp.todense()
            err_dense = float(jnp.max(jnp.abs(e.todense() - dense_ref)))

            # Restriction parity: pick first half of rows/cols.
            n_out, n_in = e.shape
            row_sel = jnp.arange(0, n_out, 2, dtype=jnp.int32)
            col_sel = jnp.arange(0, n_in, 2, dtype=jnp.int32)
            dense = e.todense()
            Er = e.restrict_rows(row_sel)
            xr = jax.random.normal(key, (n_in,), dtype=jnp.float64)
            ref_rows = dense[row_sel, :] @ xr
            err_restrict_rows = float(jnp.max(jnp.abs(Er @ xr - ref_rows)))
            Ec = e.restrict_cols(col_sel)
            yr = jax.random.normal(key, (n_out,), dtype=jnp.float64)
            ref_cols = dense[:, col_sel] @ jax.random.normal(key, (col_sel.shape[0],), dtype=jnp.float64)
            err_restrict_cols = float(jnp.max(jnp.abs(
                Ec @ jax.random.normal(key, (col_sel.shape[0],), dtype=jnp.float64) - ref_cols)))

            ok = (err_fwd < 1e-12 and err_adj < 1e-12 and err_dense < 1e-12
                  and err_restrict_rows < 1e-12 and err_restrict_cols < 1e-12)
            all_ok = all_ok and ok
            print(
                f"k={k}{tag or '   '}: shape E={e.shape}  "
                f"max|E_mf-E_bcsr|={err_fwd:.2e}  "
                f"max|ET_mf-ET_bcsr|={err_adj:.2e}  "
                f"max|dense|={err_dense:.2e}  "
                f"restr_r={err_restrict_rows:.2e}  "
                f"restr_c={err_restrict_cols:.2e}  "
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
                f"   adj: mf={t_adj_mf * 1e6:8.1f} us  "
                f"bcsr={t_adj_sp * 1e6:8.1f} us  "
                f"speedup={t_adj_sp / t_adj_mf:5.2f}x"
            )

    print("\nALL PARITY CHECKS PASSED" if all_ok else "\nPARITY FAILURE")


if __name__ == "__main__":
    main()
