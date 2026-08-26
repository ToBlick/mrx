"""Cost of jax_default_matmul_precision='highest' in float32: time the hot
matrix-free kernels of the W7-X sequence under TF32 ('default') and full
float32 ('highest').  Run with MRX_DTYPE=float32."""
import time

import jax
import jax.numpy as jnp
import mrx
from mrx.geometries import build_sequence
from mrx.geometry import SequenceGeometry

print("devices", jax.devices(), "dtype", mrx.DTYPE, flush=True)
seq, ops = build_sequence("w7x-fmm002", (8, 16, 8), 3, 10000, tol=1e-5)
ops = seq.assemble_all_sparse(include_preconditioners=False)
seq.set_operators(ops)
F = seq.geometry.map
key = jax.random.PRNGKey(0)


def bench(name, fn, arg, reps=50):
    out = fn(arg).block_until_ready()          # compile
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(arg)
    out.block_until_ready()
    return (time.perf_counter() - t0) / reps * 1e3


kernels = {}
for k in (1, 2):
    n = getattr(seq, f"n{k}_dbc")
    kernels[f"M{k} matvec (n={n})"] = (
        lambda v, k=k: seq.apply_mass_matrix(v, k, dirichlet=True),
        jax.random.normal(key, (n,)))
n1 = seq.n1_dbc
kernels[f"k=1 Laplacian matvec (n={n1})"] = (
    lambda v: seq.apply_laplacian(v, 1, dirichlet=True), jax.random.normal(key, (n1,)))
n2 = seq.n2_dbc
kernels[f"M2^-1 CG solve tol=1e-5 (n={n2})"] = (
    lambda v: seq.apply_inverse_mass_matrix(v, 2, dirichlet=True),
    seq.apply_mass_matrix(jax.random.normal(key, (n2,)), 2, dirichlet=True))
kernels["DF geometry at quad points"] = (
    lambda x: SequenceGeometry.from_map(F, x).jacobian_j, seq.quad.x)

for prec in ("default", "highest"):
    with jax.default_matmul_precision(prec):
        for name, (fn, arg) in kernels.items():
            f = jax.jit(fn)
            reps = 5 if "CG" in name else 50
            ms = bench(name, f, arg, reps)
            extra = ""
            if "CG" in name:
                v = jax.random.normal(key, (n2,))
                sol = f(seq.apply_mass_matrix(v, 2, dirichlet=True))
                extra = f"  round-trip rel err {float(jnp.linalg.norm(sol - v) / jnp.linalg.norm(v)):.2e}"
            print(f"[{prec:7s}] {name:36s} {ms:8.3f} ms{extra}", flush=True)
