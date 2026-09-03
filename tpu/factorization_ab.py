"""Where the mass kernel's element transform loses its arithmetic.

``roofline.py`` shows ``mass_core_apply`` achieving 41.6 GFLOP/s on a v5e, and
``mxu_occupancy.py`` shows a plain matmul of the same contraction width and
the same batch achieving 349 GFLOP/s on the same chip. The kernel is 8.4x off
what its own shape can do, and that gap is not the contraction width: the
occupancy sweep found cost tracking K above 8, so the MXU is not simply
padding 4 up to 128.

The remaining suspect is layout. ``'xqb,xyzbdf->xyzqdf'`` contracts ``b``,
which is axis 3 of a six-axis tensor with ``d`` and ``f`` trailing, so it is
not a matmul until something transposes it, and the axis being contracted
moves between the three stages.

This prices five ways of doing the same element transform, all timed inside a
jitted scan at the real shapes:

    chain3    the three sequential einsums the kernel runs today, K = nloc
    chain3_bt the same, with the contracted axis explicitly brought to the
              front first, to see whether XLA already does this well
    fold2     the last two axes folded into one contraction, K = nly * nlz
    fold3     all three folded, K = nlx * nly * nlz, one batched matvec
              against a precomputed per-element tensor
    gemm      a plain matmul moving the same bytes, as the ceiling

``fold3`` trades about 4x the arithmetic for one well-shaped contraction and a
large precomputed basis tensor, so it is bandwidth-bound rather than
occupancy-bound; whether that is a win is exactly what this measures.

Usage:
    python tpu/factorization_ab.py --ns 12,24,12 --p 3 --k 2 --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


def parse_args() -> argparse.Namespace:
    """Command line for the factorization A/B."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, default=2, choices=(0, 1, 2, 3))
    ap.add_argument("--component", type=int, default=0,
                    help="which component of the k-form to transform")
    ap.add_argument("--scan-length", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--precision", default=None, help="MRX_DTYPE")
    ap.add_argument("--matmul-precision", default=None)
    ap.add_argument("--map-batch-size", type=int, default=None,
                    help="mrx.MAP_BATCH_SIZE_INNER; jax < 0.9 rejects 0")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    return ap.parse_args()


def timed_scan(step, x0, length: int, repeats: int) -> tuple[float, float]:
    """Median per-apply seconds of ``step`` run inside a jitted ``lax.scan``.

    The scan is the form the relaxation actually pays: one fused XLA program
    per solver body rather than one dispatch per operator. Timing beside a
    scan overstates by up to 6.8x on this hardware.

    Args:
        step: ``carry -> carry``, one application of the transform.
        x0: initial carry.
        length: applies per scan.
        repeats: timed scans after the warm-up.

    Returns:
        ``(median_seconds_per_apply, first_call_seconds)``.
    """
    import jax

    @jax.jit
    def run(x):
        out, _ = jax.lax.scan(lambda c, _: (step(c), None), x, None,
                              length=length)
        return out

    start = time.perf_counter()
    jax.block_until_ready(run(x0))
    first = time.perf_counter() - start

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(run(x0))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) / length, first


def main() -> int:
    """Time the five formulations and report GFLOP/s for each."""
    cli = parse_args()
    if cli.precision:
        os.environ["MRX_DTYPE"] = cli.precision
    os.environ.setdefault("MRX_DTYPE", "float32")

    import jax
    import jax.numpy as jnp
    import numpy as np

    import mrx
    from mrx.geometry import build_sequence
    from mrx.mass import _elem_counts, _form_bases

    if cli.map_batch_size is not None:
        mrx.MAP_BATCH_SIZE_INNER = cli.map_batch_size
    if cli.matmul_precision:
        jax.config.update("jax_default_matmul_precision", cli.matmul_precision)

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    _, comp, _ = _form_bases(seq, cli.k)
    Bx, _, By, _, Bz, _ = comp[cli.component]
    nlx, nly, nlz = Bx.shape[-1], By.shape[-1], Bz.shape[-1]
    n_el = ne_x * ne_y * ne_z

    device = jax.devices()[0]
    backend = {
        "platform": device.platform, "device_kind": device.device_kind,
        "jax_version": jax.__version__, "dtype": str(mrx.DTYPE),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
    }
    print("=" * 76)
    print(f"{backend['device_kind']}  {backend['dtype']}  "
          f"matmul {backend['matmul_precision']}")
    print(f"k={cli.k} component {cli.component}: "
          f"{n_el} elements, nloc=({nlx},{nly},{nlz}) -> quad=({qx},{qy},{qz})")
    print("=" * 76)

    key = jax.random.PRNGKey(0)
    x_local = jax.random.normal(
        key, (ne_x, ne_y, ne_z, nlx, nly, nlz), dtype=mrx.DTYPE)

    # ---- (a) the three sequential einsums the kernel runs today ----------
    def chain3(x):
        t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x)
        t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', By, t1)
        u = jnp.einsum('zsf,xyzqrf->xyzqrs', Bz, t2)
        # back to the input layout so the scan carry keeps its shape
        s1 = jnp.einsum('xqa,xyzqrs->xyzars', Bx, u)
        s2 = jnp.einsum('yrc,xyzars->xyzacs', By, s1)
        return jnp.einsum('zse,xyzacs->xyzace', Bz, s2)

    # ---- (b) same, contracted axis moved next to its element axis --------
    def chain3_bt(x):
        # (x,y,z,b,d,f) -> (x,b,y,d,z,f) so each contraction pairs an element
        # axis with the local index it contracts, instead of reaching across
        # three trailing axes.
        t0 = x.transpose(0, 3, 1, 4, 2, 5)
        t1 = jnp.einsum('xqb,xbydzf->xqydzf', Bx, t0)
        t2 = jnp.einsum('yrd,xqydzf->xqyrzf', By, t1)
        u = jnp.einsum('zsf,xqyrzf->xqyrzs', Bz, t2)
        s1 = jnp.einsum('xqa,xqyrzs->xayrzs', Bx, u)
        s2 = jnp.einsum('yrc,xayrzs->xayczs', By, s1)
        s3 = jnp.einsum('zse,xayczs->xaycze', Bz, s2)
        return s3.transpose(0, 2, 4, 1, 3, 5)

    # ---- (c) fold the last two axes: K = nly * nlz -----------------------
    Byz = jnp.einsum('yrd,zsf->yzrsdf', By, Bz).reshape(
        ne_y, ne_z, qy * qz, nly * nlz)

    def fold2(x):
        t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x)
        t1 = t1.reshape(ne_x, ne_y, ne_z, qx, nly * nlz)
        u = jnp.einsum('yzQD,xyzqD->xyzqQ', Byz, t1)
        s = jnp.einsum('yzQD,xyzqQ->xyzqD', Byz, u)
        s = s.reshape(ne_x, ne_y, ne_z, qx, nly, nlz)
        return jnp.einsum('xqa,xyzqdf->xyzadf', Bx, s)

    # ---- (d) fold all three: one batched matvec, K = nlx * nly * nlz -----
    Bxyz = jnp.einsum('xqa,yrc,zse->xyzqrsace', Bx, By, Bz).reshape(
        ne_x, ne_y, ne_z, qx * qy * qz, nlx * nly * nlz)
    n_loc = nlx * nly * nlz
    n_q = qx * qy * qz

    def fold3(x):
        xf = x.reshape(ne_x, ne_y, ne_z, n_loc)
        u = jnp.einsum('xyzQB,xyzB->xyzQ', Bxyz, xf)
        s = jnp.einsum('xyzQB,xyzQ->xyzB', Bxyz, u)
        return s.reshape(ne_x, ne_y, ne_z, nlx, nly, nlz)

    # ---- (e) a plain matmul over the same basis tensor, as the ceiling ---
    def gemm(x):
        xf = x.reshape(n_el, n_loc)
        big = Bxyz.reshape(n_el, n_q, n_loc)
        u = jnp.einsum('eQB,eB->eQ', big, xf)
        s = jnp.einsum('eQB,eQ->eB', big, u)
        return s.reshape(ne_x, ne_y, ne_z, nlx, nly, nlz)

    # FLOPs of one round trip: each stage is 2 * (output elements) * K, and
    # the round trip does the forward chain and its transpose, so twice that.
    f_chain3 = 2 * 2 * n_el * (qx * nly * nlz * nlx
                               + qx * qy * nlz * nly
                               + qx * qy * qz * nlz)
    f_fold2 = 2 * 2 * n_el * (qx * nly * nlz * nlx
                              + qx * qy * qz * nly * nlz)
    f_fold3 = 2 * 2 * n_el * n_q * n_loc

    variants = [
        ("chain3  (today, K=nloc)", chain3, f_chain3),
        ("chain3_bt (explicit transpose)", chain3_bt, f_chain3),
        (f"fold2   (K={nly * nlz})", fold2, f_fold2),
        (f"fold3   (K={n_loc})", fold3, f_fold3),
        (f"gemm    (K={n_loc}, flat)", gemm, f_fold3),
    ]

    ref = np.asarray(chain3(x_local))
    print(f"\n{'formulation':<32}{'ms/apply':>12}{'GFLOP/s':>11}"
          f"{'MFLOP':>9}{'vs today':>10}")
    print("-" * 76)
    rows = {}
    base_ms = None
    for name, fn, flops in variants:
        try:
            seconds, first = timed_scan(fn, x_local, cli.scan_length,
                                        cli.repeats)
        except Exception as exc:  # noqa: BLE001  (report, do not abort the sweep)
            print(f"{name:<32}{'FAILED':>12}   {type(exc).__name__}")
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        ms = seconds * 1e3
        if base_ms is None:
            base_ms = ms
        gflops = flops / seconds / 1e9
        print(f"{name:<32}{ms:>12.4f}{gflops:>11.1f}{flops / 1e6:>9.2f}"
              f"{base_ms / ms:>9.2f}x")
        rows[name] = {"ms": ms, "gflops": gflops, "flops": flops,
                      "speedup_vs_today": base_ms / ms, "first_s": first}

    print("\nagreement with the current chain (max abs difference)")
    for name, fn, _ in variants[1:]:
        try:
            got = np.asarray(fn(x_local))
        except Exception:  # noqa: BLE001
            continue
        scale = float(np.max(np.abs(ref))) or 1.0
        err = float(np.max(np.abs(got - ref))) / scale
        rows[name]["rel_err"] = err
        print(f"  {name:<32}{err:>12.3e}")

    basis_mb = Bxyz.size * np.dtype(str(mrx.DTYPE)).itemsize / 1e6
    print(f"\nfold3/gemm precompute a per-element basis tensor of "
          f"{basis_mb:.1f} MB for this one component")

    result = {"backend": backend, "args": vars(cli),
              "shape": {"n_el": n_el, "nloc": [nlx, nly, nlz],
                        "quad": [qx, qy, qz], "basis_mb": basis_mb},
              "rows": rows}
    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
