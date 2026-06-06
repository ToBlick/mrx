"""SLURM/Hydra benchmark: stored-BCSR vs matrix-free mass-matrix matvec.

Robust, launchable version of ``scripts/benchmark_matrixfree_m1.py``. It times
``M_k @ x`` for ``k`` in ``cfg.ks`` (M0, M1 by default) using:

  * the stored ``BCSR`` matrix (``operators.m{k} @ x``), and
  * a matrix-free sum-factorized apply that never materializes ``M_k``,

over the same resolutions the Poisson convergence study uses
(``ns = (n, 2n, n)``, ``q = 2p + offset``). Each timing uses warmup iterations
plus a median over ``reps`` calls with ``block_until_ready``, so it is far more
robust than the ad-hoc debug script.

Single run (loops over all n for the given p):
    python scripts/config_scripts/benchmark_matvec_sparse.py p=2

Multirun sweep (one SLURM job per p):
    python scripts/config_scripts/benchmark_matvec_sparse.py -m p=1,2,3,4 \
        'n=[8,16,32]' hydra.launcher.timeout_min=240
"""
from __future__ import annotations

import json
import os
import statistics
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import mrx
import mrx.config  # noqa: F401 — register structured configs in ConfigStore
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import _mass_components
from mrx.local_assembly import (
    _component_axis_bases_k1,
    _component_axis_bases_k2,
    _elem_counts,
    _split_field,
    evaluate_basis_local,
)

jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")


# --------------------------------------------------------------------------- #
# Matrix-free mass apply (sum-factorized, never materializes M_k)
# --------------------------------------------------------------------------- #
def _quad_gauss_weight(seq):
    """``(ne_x,ne_y,ne_z,qx,qy,qz)`` outer product of the per-axis Gauss weights."""
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)
    return (wx[:, None, None, :, None, None]
            * wy[None, :, None, None, :, None]
            * wz[None, None, :, None, None, :])


def _bases_for_form(seq, form, comp_bases_fn, n_comp):
    """Evaluate the 1D bases (values + global DOF ids) for each component."""
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    cache: dict[int, tuple] = {}

    def local_eval(basis, x_q, q):
        key = id(basis)
        if key not in cache:
            cache[key] = evaluate_basis_local(basis, x_q, q)
        return cache[key]

    comp = []
    for c in range(n_comp):
        b = comp_bases_fn(form, c)
        Bx, gx = local_eval(b[0], seq.quad.x_x, qx)
        By, gy = local_eval(b[1], seq.quad.x_y, qy)
        Bz, gz = local_eval(b[2], seq.quad.x_z, qz)
        comp.append((Bx, gx, By, gy, Bz, gz))
    return comp


def _flat_dof_plan(gx, gy, gz, shape):
    """Static flat index plan into a component's flattened DOF grid.

    ``gx (ne_x, nloc_x)``, ``gy``, ``gz`` are the per-axis global DOF ids of
    each element's local DOFs. Returns a single ``int32`` array of shape
    ``(ne_x, ne_y, ne_z, nloc_x, nloc_y, nloc_z)`` whose entries are the flat
    indices into a ``shape``-grid reshaped to 1D. Built once on the host so the
    matvec needs no index arithmetic -- just one gather / one ``segment_sum``.
    """
    Sx, Sy, Sz = (int(s) for s in shape)
    gx = np.asarray(gx)
    gy = np.asarray(gy)
    gz = np.asarray(gz)
    idx = (gx[:, None, None, :, None, None] * (Sy * Sz)
           + gy[None, :, None, None, :, None] * Sz
           + gz[None, None, :, None, None, :])
    return jnp.asarray(idx.astype(np.int32))


def _element_apply(Bvals_r, Bvals_c, W, x_flat_c, gather_idx_c):
    """One (row-comp, col-comp) element contraction folded against a vector.

    Mirrors ``local_assembly._elem_block_mixed`` but contracts against the
    gathered input instead of forming the dense element block. The gather uses
    a precomputed flat index plan (``gather_idx_c``); no index arithmetic runs
    in the matvec.
    """
    Bxr, Byr, Bzr = Bvals_r
    Bxc, Byc, Bzc = Bvals_c
    # Gather element-local input for the column component (single gather).
    x_local = x_flat_c[gather_idx_c]  # (ne_x,ne_y,ne_z,nxc,nyc,nzc)

    # Column bases -> quadrature points.
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bxc, x_local)
    t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', Byc, t1)
    u = jnp.einsum('zsf,xyzqrf->xyzqrs', Bzc, t2)

    # Metric weight at the quadrature points (already includes Gauss weights).
    u = u * W

    # Row bases <- quadrature points.
    s1 = jnp.einsum('xqa,xyzqrs->xyzars', Bxr, u)
    s2 = jnp.einsum('yrc,xyzars->xyzacs', Byr, s1)
    y_local = jnp.einsum('zse,xyzacs->xyzace', Bzr, s2)
    return y_local


def build_matrixfree_mass_apply(seq, k):
    """Return a jitted raw-DOF-space ``x -> M_k x`` that never stores ``M_k``."""
    geometry = seq.geometry
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    gw = _quad_gauss_weight(seq)

    if k == 0:
        form = seq.basis_0
        comp = _bases_for_form(seq, form, lambda f, c: [f.Λ[0], f.Λ[1], f.Λ[2]], 1)
        weight = geometry.jacobian_j  # scalar (nquad,)
        pairs = [(0, 0)]
        weight_of = {(0, 0): weight}
        n_comp = 1
    elif k == 3:
        form = seq.basis_3
        comp = _bases_for_form(seq, form, lambda f, c: [f.dΛ[0], f.dΛ[1], f.dΛ[2]], 1)
        weight = 1.0 / geometry.jacobian_j
        pairs = [(0, 0)]
        weight_of = {(0, 0): weight}
        n_comp = 1
    elif k == 1:
        form = seq.basis_1
        comp = _bases_for_form(seq, form, _component_axis_bases_k1, 3)
        metric = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]
        pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
        weight_of = {(cr, cc): metric[:, cr, cc] for cr, cc in pairs}
        n_comp = 3
    elif k == 2:
        form = seq.basis_2
        comp = _bases_for_form(seq, form, _component_axis_bases_k2, 3)
        metric = geometry.metric_jkl * (1.0 / geometry.jacobian_j)[:, None, None]
        pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
        weight_of = {(cr, cc): metric[:, cr, cc] for cr, cc in pairs}
        n_comp = 3
    else:
        raise ValueError("k must be 0, 1, 2 or 3")

    shapes = form.shape
    starts = [0]
    for c in range(n_comp):
        Sx, Sy, Sz = shapes[c]
        starts.append(starts[-1] + Sx * Sy * Sz)

    # Pre-split + fold Gauss weights into each (cr, cc) metric field.
    W_split = {}
    for (cr, cc) in pairs:
        Wf = _split_field(weight_of[(cr, cc)], nx, ny, nz,
                          ne_x, ne_y, ne_z, qx, qy, qz)
        W_split[(cr, cc)] = Wf * gw

    # --- Static element plan (built ONCE, reused for every matvec) -----------
    # Basis VALUES (for the einsums) are separated from the gather/scatter
    # index plans. The index plans -- flat gather indices per column component
    # and flat scatter (segment-id) arrays per row component -- depend only on
    # the mesh topology, so they are precomputed here and passed in as device
    # int32 arrays. The matvec then performs a single gather and a single
    # segment_sum per (cr, cc) pair with NO index arithmetic.
    Bvals = tuple((c[0], c[2], c[4]) for c in comp)          # (Bx, By, Bz)
    gather_idx = tuple(
        _flat_dof_plan(comp[cc][1], comp[cc][3], comp[cc][5], shapes[cc])
        for cc in range(n_comp))
    seg_idx = tuple(
        _flat_dof_plan(comp[cr][1], comp[cr][3], comp[cr][5],
                       shapes[cr]).reshape(-1)
        for cr in range(n_comp))
    nseg = tuple(int(np.prod(shapes[c])) for c in range(n_comp))

    shapes_t = tuple(tuple(int(v) for v in s) for s in shapes)
    starts_t = tuple(int(s) for s in starts)

    @jax.jit
    def _impl(x, Bvals, W_split, gather_idx, seg_idx):
        # Split the input into flattened component DOF vectors.
        Xc = [x[starts_t[c]:starts_t[c + 1]] for c in range(n_comp)]

        out_parts = []
        for cr in range(n_comp):
            acc = jnp.zeros((nseg[cr],), dtype=x.dtype)
            for cc in range(n_comp):
                if (cr, cc) not in W_split:
                    continue
                y_local = _element_apply(
                    Bvals[cr], Bvals[cc], W_split[(cr, cc)],
                    Xc[cc], gather_idx[cc])
                acc = acc + jax.ops.segment_sum(
                    y_local.reshape(-1), seg_idx[cr], num_segments=nseg[cr])
            out_parts.append(acc)
        return jnp.concatenate(out_parts)

    def apply(x):
        return _impl(x, Bvals, W_split, gather_idx, seg_idx)

    return apply


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def time_matvec(fn, x, reps, warmup):
    """Return (compile_s, median_exec_s, min_exec_s) with block-until-ready."""
    t0 = time.perf_counter()
    y = fn(x)
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - t0

    for _ in range(warmup):
        jax.block_until_ready(fn(x))

    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        samples.append(time.perf_counter() - t0)
    return compile_s, statistics.median(samples), min(samples)


def run_case(n, p, k, epsilon, quad_order, quad_order_offset, reps, warmup, seed):
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order
    if q < 2 * p:
        raise ValueError(f"quad_order must satisfy q >= 2*p; got q={q}, p={p}")

    seq = DeRhamSequence(ns, ps, q, types, polar=True)
    seq.set_map(toroid_map(epsilon=epsilon))
    seq.evaluate_1d()

    t0 = time.perf_counter()
    seq.assemble_mass_matrix(k)
    jax.block_until_ready(getattr(seq, f"m{k}").data)
    assemble_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    seq.assemble_mass_matrix(k)
    jax.block_until_ready(getattr(seq, f"m{k}").data)
    assemble_exec = time.perf_counter() - t0

    operators = seq.get_operators()
    m_k, _, _ = _mass_components(operators, k)
    n_dof = int(m_k.shape[0])
    nnz = int(m_k.nse)
    m_gb = nnz * 16 / 1e9  # float64 data + int32 (nnz,2) indices

    x = jax.random.normal(jax.random.PRNGKey(seed), (n_dof,), dtype=jnp.float64)

    bcsr_apply = jax.jit(lambda v: m_k @ v)
    mf_apply = build_matrixfree_mass_apply(seq, k)

    y_ref = bcsr_apply(x)
    y_mf = mf_apply(x)
    relerr = float(jnp.linalg.norm(y_mf - y_ref) / jnp.linalg.norm(y_ref))

    bc_c, bc_med, bc_min = time_matvec(bcsr_apply, x, reps, warmup)
    mf_c, mf_med, mf_min = time_matvec(mf_apply, x, reps, warmup)

    return {
        "n": n, "p": p, "k": k, "n_dof": n_dof, "nnz": nnz, "m_gb": m_gb,
        "relerr": relerr,
        "assemble_compile_s": assemble_compile, "assemble_exec_s": assemble_exec,
        "bcsr_compile_s": bc_c, "bcsr_median_s": bc_med, "bcsr_min_s": bc_min,
        "mf_compile_s": mf_c, "mf_median_s": mf_med, "mf_min_s": mf_min,
        "speedup_median": bc_med / mf_med if mf_med > 0 else float("nan"),
    }


@hydra.main(config_path="../../conf",
            config_name="config_matvec_benchmark", version_base=None)
def main(cfg: DictConfig):
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"JAX devices: {jax.devices()}")
    p = cfg.p
    ns = list(cfg.n)
    ks = list(cfg.ks)
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"Matvec benchmark: n={ns}, p={p}, ks={ks}, reps={cfg.reps}, "
          f"warmup={cfg.warmup}")

    results = []
    header = (f"{'n':>4} {'p':>2} {'k':>2} {'n_dof':>10} {'nnz':>13} "
              f"{'M[GB]':>8} {'relerr':>10} {'bcsr[ms]':>10} {'mf[ms]':>10} "
              f"{'speedup':>8}")
    print(header)
    print("-" * len(header))
    for k in ks:
        for n in ns:
            try:
                r = run_case(n, p, k, cfg.epsilon, cfg.quad_order,
                             cfg.quad_order_offset, cfg.reps, cfg.warmup, cfg.seed)
            except Exception as exc:  # noqa: BLE001 — report and continue
                print(f"{n:>4} {p:>2} {k:>2}  FAILED: {type(exc).__name__}: {exc}")
                results.append({"n": n, "p": p, "k": k,
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            results.append(r)
            print(f"{r['n']:>4} {r['p']:>2} {r['k']:>2} {r['n_dof']:>10} "
                  f"{r['nnz']:>13} {r['m_gb']:>8.3f} {r['relerr']:>10.2e} "
                  f"{r['bcsr_median_s'] * 1e3:>10.3f} "
                  f"{r['mf_median_s'] * 1e3:>10.3f} {r['speedup_median']:>8.2f}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")
    with open(outfile, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results saved to {outfile}")


if __name__ == "__main__":
    main()
