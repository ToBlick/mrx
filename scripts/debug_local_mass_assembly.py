"""Element-local sum-factorized mass assembly for k = 0, 1, 2, 3.

Validates new element-local assemblers (`assemble_m{0,1,2,3}_local`) against
the existing global einsum path (`assemble_scalar`/`assemble_vectorial`) at
polar=False, and benchmarks how local vs global scale with (n, p).

Run interactively in VS Code as a Jupyter file (cells delimited by `# %%`),
or top-to-bottom as a regular script (e.g. via slurm/job_local_mass_assembly.sh).
"""

# %% Imports
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.local_assembly import (
    assemble_m0_local,
    assemble_m1_local,
    assemble_m2_local,
    assemble_m3_local,
)

jax.config.update("jax_enable_x64", True)
print("devices:", jax.devices())


# The element-local assemblers now live in mrx.local_assembly (imported above).
# This script just validates them against the global path and benchmarks them.


# %% Build small reference case and validate
def build_seq(n, p, polar=False):
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p
    types = ("clamped", "periodic", "periodic")
    F = toroid_map(epsilon=1 / 3)
    seq = DeRhamSequence(ns, ps, q, types, polar=polar)
    seq.set_map(F)
    seq.evaluate_1d()
    return seq


# Small validation: polar=False keeps M0 in raw tensor-product DOF space,
# which is what _assemble_mass_block produces. (Polar extraction is applied
# afterward as E M E.T -- it is post-multiplicative and orthogonal to the
# assembly itself.)
print("\n=== Validation: n=8, p=2, polar=False ===")
seq_small = build_seq(n=8, p=2, polar=False)
seq_small.assemble_mass_matrix(0)
M0_ref = jnp.asarray(seq_small.m0.todense())
M0_new = assemble_m0_local(seq_small).todense()
err = float(jnp.linalg.norm(M0_new - M0_ref) / jnp.linalg.norm(M0_ref))
print(f"  shape ref = {M0_ref.shape}, new = {M0_new.shape}")
print(f"  relative Frobenius error: {err:.3e}")
print(f"  max abs error: {float(jnp.max(jnp.abs(M0_new - M0_ref))):.3e}")
assert err < 1e-12, f"validation failed: rel err {err:.3e}"
print("  OK")


# %% Unified validation + timing for k = 0, 1, 2, 3
LOCAL_ASSEMBLERS = {
    0: assemble_m0_local,
    1: assemble_m1_local,
    2: assemble_m2_local,
    3: assemble_m3_local,
}


def _ref_mass_dense(seq, k):
    seq.assemble_mass_matrix(k)
    M = {0: seq.m0, 1: seq.m1, 2: seq.m2, 3: seq.m3}[k]
    return jnp.asarray(M.todense())


def _ref_mass_data(seq, k):
    seq.assemble_mass_matrix(k)
    return {0: seq.m0, 1: seq.m1, 2: seq.m2, 3: seq.m3}[k].data


print("\n" + "=" * 60)
print("VALIDATION: local vs global mass matrices (polar=False)")
print("=" * 60)
for k in (0, 1, 2, 3):
    print(f"\n--- k = {k} ---")
    for n_v, p_v in [(8, 2), (8, 3)]:
        seq_v = build_seq(n=n_v, p=p_v, polar=False)
        M_ref = _ref_mass_dense(seq_v, k)
        M_new = LOCAL_ASSEMBLERS[k](seq_v).todense()
        err = float(jnp.linalg.norm(M_new - M_ref) / jnp.linalg.norm(M_ref))
        max_abs = float(jnp.max(jnp.abs(M_new - M_ref)))
        flag = "  *** MISMATCH ***" if err >= 1e-10 else ""
        print(f"  n={n_v:>2} p={p_v}:  shape={tuple(M_ref.shape)}  "
              f"rel err={err:.3e}  max abs={max_abs:.3e}{flag}")


# %% Timing comparison
def time_one(fn, n_warmup=1, n_repeat=2):
    # One warmup call absorbs JIT compilation so we time runtime, not compile.
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return min(times), out


# Global path is O(n^6) and blows up fast, so only time it on the cheaper
# cases. Local path is timed across the full sweep to expose its O(n^3) scaling.
local_cases = [
    (8, 2), (16, 2), (24, 2), (32, 2), (48, 2),
    (8, 3), (16, 3), (24, 3),
    (8, 4), (16, 4),
]
global_cases = {(8, 2), (16, 2), (24, 2), (8, 3), (16, 3), (8, 4)}

print("\n" + "=" * 60)
print("TIMING: local vs global assembly (warmup + timed; runtime only)")
print("=" * 60)
for k in (0, 1, 2, 3):
    print(f"\n--- k = {k} ---")
    print(f"{'n':>4} {'p':>2}   {'global (s)':>12} {'local (s)':>12} {'speedup':>8}")
    for n_v, p_v in local_cases:
        seq_v = build_seq(n=n_v, p=p_v, polar=False)

        def run_new(_seq=seq_v, _k=k):
            return LOCAL_ASSEMBLERS[_k](_seq).data

        t_ref = float("nan")
        if (n_v, p_v) in global_cases:
            def run_ref(_seq=seq_v, _k=k):
                return _ref_mass_data(_seq, _k)
            try:
                t_ref, _ = time_one(run_ref)
            except Exception as e:
                print(f"{n_v:>4} {p_v:>2}   global FAILED: {type(e).__name__}")
        t_new, _ = time_one(run_new)
        if t_ref == t_ref:  # not NaN
            print(f"{n_v:>4} {p_v:>2}   {t_ref:>12.4f} {t_new:>12.4f} "
                  f"{t_ref / t_new:>8.2f}x")
        else:
            print(f"{n_v:>4} {p_v:>2}   {'--':>12} {t_new:>12.4f} {'--':>8}")


# %% Done
print("\nDone. If all four k validate and local timing scales as O(n^3),"
      " we can roll these into mrx.assembly and switch _assemble_mass_block"
      " to use them.")
