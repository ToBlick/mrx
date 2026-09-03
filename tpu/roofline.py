"""Which roof, if any, the mass kernel is under.

The measured per-apply cost of ``mass_core_apply`` says nothing on its own
about whether the machine is doing badly. This counts the work the kernel
actually asks for -- floating point operations, essential HBM traffic, and the
reduced dimension of every contraction -- so the measured time can be placed
against a roofline instead of against another machine's measured time.

It also reports, for each component, whether the shift plan holds. That
decides whether the production path uses :func:`mrx.mass._structured_gather`
or falls back to the indexed read, which is the difference between a gather
being a live cost and being dead code.

Usage:
    python tpu/roofline.py --ns 12,24,12 --p 3 --k 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    """Command line for the roofline count."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, default=2, choices=(0, 1, 2, 3))
    ap.add_argument("--precision", default=None,
                    help="MRX_DTYPE; default leaves the environment alone")
    ap.add_argument("--measured-ms", type=float, default=None,
                    help="measured per-apply cost, to turn the counts into "
                         "achieved FLOP/s and GB/s")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="device compute peak in TFLOP/s at the working "
                         "precision, for the fraction-of-peak column")
    ap.add_argument("--peak-gbs", type=float, default=None,
                    help="device HBM bandwidth peak in GB/s")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    return ap.parse_args()


def main() -> int:
    """Count the kernel's work and place it against a roofline."""
    cli = parse_args()
    if cli.precision:
        os.environ["MRX_DTYPE"] = cli.precision
    os.environ.setdefault("MRX_DTYPE", "float32")

    import numpy as np

    import mrx
    from mrx.geometry import build_sequence
    from mrx.mass import _elem_counts, _form_bases, _shift_plan

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    k = cli.k
    itemsize = np.dtype(str(mrx.DTYPE)).itemsize

    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    n_el = ne_x * ne_y * ne_z
    n_quad = n_el * qx * qy * qz

    form, comp, n_comp = _form_bases(seq, k)

    print("=" * 72)
    print(f"mass_core_apply k={k}  ns={ns} p={cli.p}  {mrx.DTYPE}")
    print(f"  elements      {ne_x} x {ne_y} x {ne_z} = {n_el}")
    print(f"  quad/element  {qx} x {qy} x {qz} = {qx * qy * qz}")
    print(f"  quad total    {n_quad}")
    print("=" * 72)

    # --- the shift plan, which decides whether any indexed read is live ----
    print("\nshift plan per component (structured read vs indexed gather)")
    shapes = form.shape
    plans_hold = []
    for c in range(n_comp):
        Bx, gx, By, gy, Bz, gz = comp[c]
        shape = shapes[c]
        plan = _shift_plan(gx, gy, gz, shape)
        plans_hold.append(plan is not None)
        nloc = (Bx.shape[-1], By.shape[-1], Bz.shape[-1])
        print(f"  component {c}: nloc={nloc} shape={tuple(shape)} "
              f"-> {'STRUCTURED (no index tensor)' if plan else 'INDEXED GATHER'}")
    structured = all(plans_hold)
    print(f"  => production path is "
          f"{'_structured_gather' if structured else 'x[gather_idx]'}")

    # --- contractions: the reduced dimension is the MXU question ----------
    print("\ncontractions, one per einsum (K is what the MXU sees)")
    rows = []
    flops = 0
    for c in range(n_comp):
        Bx, _, By, _, Bz, _ = comp[c]
        nlx, nly, nlz = Bx.shape[-1], By.shape[-1], Bz.shape[-1]
        # column half: contract the local dof index of each axis in turn
        for axis, (K, out) in enumerate((
                (nlx, n_el * qx * nly * nlz),
                (nly, n_el * qx * qy * nlz),
                (nlz, n_el * qx * qy * qz))):
            rows.append(("to_quadrature", c, "xyz"[axis], K, out))
            flops += 2 * out * K
        # row half: contract the quadrature index of each axis in turn
        for axis, (K, out) in enumerate((
                (qx, n_el * nlx * qy * qz),
                (qy, n_el * nlx * nly * qz),
                (qz, n_el * nlx * nly * nlz))):
            rows.append(("from_quadrature", c, "xyz"[axis], K, out))
            flops += 2 * out * K

    for half in ("to_quadrature", "from_quadrature"):
        ks = sorted({r[3] for r in rows if r[0] == half})
        print(f"  {half:<16} K in {ks}")
    all_k = sorted({r[3] for r in rows})
    print(f"  every contraction has K in {all_k}; an MXU systolic array is "
          f"128 wide")

    # the metric mix at quadrature: one multiply-add per column component
    flops += n_comp * n_quad * (2 * n_comp - 1)

    # --- essential HBM traffic --------------------------------------------
    n_raw = int(sum(np.prod(s) for s in shapes))
    n_weight = (n_comp * (n_comp + 1)) // 2
    bytes_weights = n_weight * n_quad * itemsize
    bytes_io = 2 * n_raw * itemsize
    bytes_min = bytes_weights + bytes_io

    print("\nessential HBM traffic (intermediates assumed to stay on chip)")
    print(f"  read x, write y   {bytes_io / 1e6:8.3f} MB  ({n_raw} dofs)")
    print(f"  metric weights    {bytes_weights / 1e6:8.3f} MB  "
          f"({n_weight} unique blocks x {n_quad} points)")
    print(f"  total             {bytes_min / 1e6:8.3f} MB")
    print(f"\n  {flops / 1e6:.2f} MFLOP against {bytes_min / 1e6:.2f} MB "
          f"= {flops / bytes_min:.2f} FLOP/byte")

    result = {
        "ns": list(ns), "p": cli.p, "k": k, "dtype": str(mrx.DTYPE),
        "n_el": n_el, "n_quad": n_quad, "n_raw": n_raw,
        "structured_gather": structured,
        "contraction_dims": all_k,
        "flops": int(flops), "bytes_min": int(bytes_min),
        "flops_per_byte": flops / bytes_min,
    }

    if cli.measured_ms:
        seconds = cli.measured_ms * 1e-3
        achieved_flops = flops / seconds
        achieved_bw = bytes_min / seconds
        print(f"\nat the measured {cli.measured_ms:.4f} ms per apply")
        print(f"  achieved   {achieved_flops / 1e9:9.1f} GFLOP/s"
              f"   {achieved_bw / 1e9:8.2f} GB/s")
        result["measured_ms"] = cli.measured_ms
        result["achieved_gflops"] = achieved_flops / 1e9
        result["achieved_gbs"] = achieved_bw / 1e9
        if cli.peak_tflops:
            frac = achieved_flops / (cli.peak_tflops * 1e12)
            print(f"  of compute peak  {100 * frac:8.3f} %  "
                  f"({cli.peak_tflops} TFLOP/s)")
            result["frac_compute_peak"] = frac
        if cli.peak_gbs:
            frac = achieved_bw / (cli.peak_gbs * 1e9)
            print(f"  of bandwidth peak{100 * frac:8.3f} %  "
                  f"({cli.peak_gbs} GB/s)")
            result["frac_bandwidth_peak"] = frac

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
