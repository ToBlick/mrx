"""Benchmark CG solve with BCSR vs BCOO vs dense mass matrix."""

import time

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
p = 3
n = 20
ns = (n, n, n)
ps = (p, p, p)
types = ("clamped", "periodic", "periodic")

print(f"Building DeRham sequence  n={n}, p={p} ...")
F = rotating_ellipse_map(nfp=3)
seq = DeRhamSequence(ns, ps, 2 * p, types, F, polar=True,
                     tol=1e-12, maxiter=1000)
seq.evaluate_1d()

print("Assembling m0 (tensor-product) ...")
t0 = time.perf_counter()
seq.assemble_mass_matrix(0)
t1 = time.perf_counter()
print(f"  Assembly: {t1 - t0:.2f}s")

m0_bcsr = seq.m0_sp                          # already BCSR
m0_bcoo = m0_bcsr.to_bcoo()                   # convert to BCOO
m0_dense = m0_bcsr.todense()                  # convert to dense

N = m0_bcsr.shape[0]
print(f"  Matrix size: {N} x {N}")
print(f"  NNZ: {m0_bcsr.data.shape[0]}")
print(f"  Dense would be: {N*N*8 / 1e6:.1f} MB")

e_T = seq.e0_dbc_T
e = seq.e0_dbc

# --------------------------------------------------------------------------
# Build matvec closures
# --------------------------------------------------------------------------


def matvec_bcsr(v):
    return e @ (m0_bcsr @ (e_T @ v))


def matvec_bcoo(v):
    return e @ (m0_bcoo @ (e_T @ v))


def matvec_dense(v):
    return e @ (m0_dense @ (e_T @ v))


matvec_bcsr_jit = jax.jit(matvec_bcsr)
matvec_bcoo_jit = jax.jit(matvec_bcoo)
matvec_dense_jit = jax.jit(matvec_dense)

# --------------------------------------------------------------------------
# CG solves
# --------------------------------------------------------------------------

n_solves = 5
key = jax.random.PRNGKey(42)

# Generate random RHS vectors
keys = jax.random.split(key, n_solves)
n_out = e.shape[0]
bs = [jax.random.normal(k, (n_out,)) for k in keys]

# Warm-up: compile the CG solves (don't time this)
print("\nWarming up JIT (first CG solve with each format) ...")


def cg_solve(matvec_fn, b):
    return cg(matvec_fn, b, tol=1e-10, maxiter=500)


cg_bcsr = jax.jit(lambda b: cg_solve(matvec_bcsr_jit, b))
cg_bcoo = jax.jit(lambda b: cg_solve(matvec_bcoo_jit, b))
cg_dense = jax.jit(lambda b: cg_solve(matvec_dense_jit, b))

# Warm up
x0, _ = cg_bcsr(bs[0])
x0.block_until_ready()
x0, _ = cg_bcoo(bs[0])
x0.block_until_ready()
x0, _ = cg_dense(bs[0])
x0.block_until_ready()
print("  Done.")

# --------------------------------------------------------------------------
# Timed runs
# --------------------------------------------------------------------------
print(f"\nTiming {n_solves} CG solves each ...\n")

for label, cg_fn in [("BCSR ", cg_bcsr),
                     ("BCOO ", cg_bcoo),
                     ("Dense", cg_dense)]:
    times = []
    for b in bs:
        t_start = time.perf_counter()
        x, info = cg_fn(b)
        x.block_until_ready()
        t_end = time.perf_counter()
        times.append(t_end - t_start)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    print(f"  {label}:  avg={avg*1e3:7.2f} ms   "
          f"min={mn*1e3:7.2f} ms   max={mx*1e3:7.2f} ms")

# --------------------------------------------------------------------------
# Also benchmark raw matvec (no CG overhead)
# --------------------------------------------------------------------------
print(f"\nTiming raw matvecs ({n_solves} calls each) ...\n")

for label, fn in [("BCSR ", matvec_bcsr_jit),
                  ("BCOO ", matvec_bcoo_jit),
                  ("Dense", matvec_dense_jit)]:
    # warm up already happened via CG
    times = []
    for b in bs:
        t_start = time.perf_counter()
        y = fn(b)
        y.block_until_ready()
        t_end = time.perf_counter()
        times.append(t_end - t_start)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    print(f"  {label}:  avg={avg*1e3:7.2f} ms   "
          f"min={mn*1e3:7.2f} ms   max={mx*1e3:7.2f} ms")
