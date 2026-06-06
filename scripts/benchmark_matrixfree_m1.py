"""Benchmark: matrix-free M1 matvec vs. stored BCSR M1 matvec.

The 1-form mass matrix ``M1`` is by far the largest assembled operator
(``~16 bytes * nnz``; tens to hundreds of GB at the high-(n, p) corners). This
script checks whether a *matrix-free* sum-factorized ``M1 @ x`` -- which never
stores the matrix -- is competitive in wall-clock time with JAX's stored
``BCSR`` matvec at moderate resolutions, and confirms the two agree.

Both operate in the raw tensor-product (unextracted / periodic) DOF space, so
the matrix-free apply is compared directly against ``operators.m1 @ x`` with no
polar/boundary extraction in the loop.

Run (from repo root, on a GPU node):
    python scripts/benchmark_matrixfree_m1.py
    python scripts/benchmark_matrixfree_m1.py --ns 8 16 --ps 1 2 3 --reps 20
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import _mass_components
from mrx.local_assembly import (
    _component_axis_bases_k1,
    _elem_counts,
    _split_field,
    evaluate_basis_local,
)

jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")


# --------------------------------------------------------------------------- #
# Matrix-free k=1 mass apply (sum-factorized, never materializes M1)
# --------------------------------------------------------------------------- #
def build_matrixfree_m1_apply(seq):
    """Return a jitted ``x -> M1 @ x`` in the raw tensor-product DOF space.

    Mirrors ``mrx.local_assembly.assemble_m1_local`` but contracts against a
    vector per element instead of forming and storing the element block. Per
    element the work is ``O((p+1)^4 * q)`` via sum factorization; the largest
    transient is ``O(n^3 (p+1)^2 q)``, far below the ``O(n^3 (p+1)^6)`` stored
    matrix.
    """
    form = seq.basis_1
    geometry = seq.geometry
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz

    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)

    # Full 3x3 metric weight (without quadrature weights; folded in per axis).
    weight = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]

    shapes = form.shape                       # per-component DOF-grid shapes
    starts = [0, form.n1, form.n1 + form.n2]  # flat offsets per component

    # Cache 1D basis evaluations (values + global DOF ids) per component.
    eval_cache: dict[int, tuple] = {}

    def local_eval(basis, x_q, q):
        key = id(basis)
        if key not in eval_cache:
            eval_cache[key] = evaluate_basis_local(basis, x_q, q)
        return eval_cache[key]

    comp = []
    for c in range(3):
        b = _component_axis_bases_k1(form, c)
        Bx, gx = local_eval(b[0], seq.quad.x_x, qx)
        By, gy = local_eval(b[1], seq.quad.x_y, qy)
        Bz, gz = local_eval(b[2], seq.quad.x_z, qz)
        comp.append((Bx, gx, By, gy, Bz, gz))

    # Pre-split the (cr, cc) weight fields and fold in the Gauss weights.
    gw = (wx[:, None, None, :, None, None]
          * wy[None, :, None, None, :, None]
          * wz[None, None, :, None, None, :])  # (ne_x,ne_y,ne_z,qx,qy,qz)
    Wcc = {}
    for cr in range(3):
        for cc in range(3):
            Wf = _split_field(weight[:, cr, cc], nx, ny, nz,
                              ne_x, ne_y, ne_z, qx, qy, qz)
            Wcc[(cr, cc)] = Wf * gw

    @jax.jit
    def apply(x):
        # Split the input vector into its three vector components.
        Xc = []
        for c in range(3):
            sl = x[starts[c]:starts[c] + (shapes[c][0] * shapes[c][1] * shapes[c][2])]
            Xc.append(sl.reshape(shapes[c]))

        out_parts = []
        for cr in range(3):
            Bxr, gxr, Byr, gyr, Bzr, gzr = comp[cr]
            Sxr, Syr, Szr = shapes[cr]
            acc = jnp.zeros((Sxr * Syr * Szr,), dtype=x.dtype)
            for cc in range(3):
                Bxc, gxc, Byc, gyc, Bzc, gzc = comp[cc]
                Xcc = Xc[cc]

                # Gather element-local input: (ne_x,ne_y,ne_z,nxc,nyc,nzc)
                x_local = Xcc[
                    gxc[:, None, None, :, None, None],
                    gyc[None, :, None, None, :, None],
                    gzc[None, None, :, None, None, :],
                ]

                # Column bases -> quadrature points (sum factorization).
                t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bxc, x_local)
                t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', Byc, t1)
                u = jnp.einsum('zsf,xyzqrf->xyzqrs', Bzc, t2)

                # Apply the metric weight at the quadrature points.
                u = u * Wcc[(cr, cc)]

                # Row bases <- quadrature points.
                s1 = jnp.einsum('xqa,xyzqrs->xyzars', Bxr, u)
                s2 = jnp.einsum('yrc,xyzars->xyzacs', Byr, s1)
                y_local = jnp.einsum('zse,xyzacs->xyzace', Bzr, s2)

                # Scatter-add into the row component's DOF grid.
                row = ((gxr[:, None, None, :, None, None] * Syr
                        + gyr[None, :, None, None, :, None]) * Szr
                       + gzr[None, None, :, None, None, :])
                row = jnp.broadcast_to(row, y_local.shape).reshape(-1)
                acc = acc + jax.ops.segment_sum(
                    y_local.reshape(-1), row, num_segments=Sxr * Syr * Szr)
            out_parts.append(acc)
        return jnp.concatenate(out_parts)

    return apply


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def time_call(fn, x, reps):
    """Return (compile_s, exec_s) for ``fn(x)`` with block-until-ready."""
    t0 = time.perf_counter()
    y = fn(x)
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(reps):
        y = fn(x)
    jax.block_until_ready(y)
    exec_s = (time.perf_counter() - t0) / reps
    return compile_s, exec_s


def run_case(n, p, epsilon, reps):
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p
    seq = DeRhamSequence(ns, ps, q, types, polar=True)
    seq.set_map(toroid_map(epsilon=epsilon))
    seq.evaluate_1d()
    seq.assemble_mass_matrix(1)

    operators = seq.get_operators()
    m1, _, _ = _mass_components(operators, 1)
    n_dof = m1.shape[0]
    nnz = int(m1.nse)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_dof,), dtype=jnp.float64)

    bcsr_apply = jax.jit(lambda v: m1 @ v)
    mf_apply = build_matrixfree_m1_apply(seq)

    # Correctness.
    y_ref = bcsr_apply(x)
    y_mf = mf_apply(x)
    rel = float(jnp.linalg.norm(y_mf - y_ref) / jnp.linalg.norm(y_ref))

    bc_compile, bc_exec = time_call(bcsr_apply, x, reps)
    mf_compile, mf_exec = time_call(mf_apply, x, reps)

    m1_gb = nnz * 16 / 1e9  # float64 data + int32 (nnz,2) indices
    return {
        "n": n, "p": p, "n_dof": n_dof, "nnz": nnz, "m1_gb": m1_gb,
        "relerr": rel,
        "bcsr_compile": bc_compile, "bcsr_exec": bc_exec,
        "mf_compile": mf_compile, "mf_exec": mf_exec,
        "speedup": bc_exec / mf_exec if mf_exec > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--ps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"reps={args.reps}  epsilon={args.epsilon}")
    header = (f"{'n':>4} {'p':>2} {'n_dof':>10} {'nnz':>13} {'M1[GB]':>8} "
              f"{'relerr':>10} {'bcsr_ex[ms]':>12} {'mf_ex[ms]':>11} "
              f"{'speedup':>8}")
    print(header)
    print("-" * len(header))
    for p in args.ps:
        for n in args.ns:
            try:
                r = run_case(n, p, args.epsilon, args.reps)
            except Exception as exc:  # noqa: BLE001 — report and continue
                print(f"{n:>4} {p:>2}  FAILED: {type(exc).__name__}: {exc}")
                continue
            print(f"{r['n']:>4} {r['p']:>2} {r['n_dof']:>10} {r['nnz']:>13} "
                  f"{r['m1_gb']:>8.3f} {r['relerr']:>10.2e} "
                  f"{r['bcsr_exec'] * 1e3:>12.3f} {r['mf_exec'] * 1e3:>11.3f} "
                  f"{r['speedup']:>8.2f}")


if __name__ == "__main__":
    main()
