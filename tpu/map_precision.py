#!/usr/bin/env python3
"""What ``jax_default_matmul_precision`` costs the geometry on a TPU MXU.

``mrx/precision.py`` pins the setting to ``'highest'`` for a GPU reason: on
Ampere-and-later, float32 matmuls run in TF32 by default, which made the W7-X
map's ``dR/dtheta`` 19% wrong and drove ``det DF`` negative. The comment
records that float64 is unaffected -- so the line is free on a GPU running
float64 and is a standing tax on a TPU running float32, where the MXU
multiplies bf16 natively and 'highest' means six passes, 'high' three and
'default' one.

The matvec benchmark says that tax is real: on the fixed mass kernel,
``mass_core_apply`` k=1 costs 0.504 ms at 'highest' and 0.324 ms at 'high',
a 1.55x. The question this answers is whether the accuracy the setting buys
is needed globally or only by the map, because a global flag is the only
knob mrx has today.

Reports, for one setting, the ``det DF`` range and the map's own values
against a float64 reference computed in the same process at 'highest'.

    python -u map_precision.py --matmul-precision high
"""
from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--matmul-precision", default=None)
    ap.add_argument("--map-batch-size", type=int, default=None)
    ap.add_argument("--out", default="map_precision.json")
    return ap.parse_args()


def main() -> None:
    cli = parse_args()
    os.environ.setdefault("MRX_DTYPE", "float32")

    import jax
    import jax.numpy as jnp
    import numpy as np

    import mrx
    from mrx.derham_sequence import DeRhamSequence
    from mrx.geometry import map_jacobian_at
    from mrx.gvec import build_gvec_map

    if cli.map_batch_size is not None:
        mrx.MAP_BATCH_SIZE_INNER = cli.map_batch_size

    setting = cli.matmul_precision or "highest"
    dev = jax.devices()[0]
    ns = tuple(int(v) for v in cli.ns.split(","))
    print(f"[env] {dev.device_kind}  mrx {mrx.DTYPE}  matmul {setting}",
          flush=True)

    seq = DeRhamSequence(ns, (cli.p,) * 3, cli.p + 1,
                         ("clamped", "periodic", "periodic"), polar=True)
    map_func, _ = build_gvec_map(cli.geometry, seq)
    quad_x = seq.quad.x

    def measure(precision):
        jax.config.update("jax_default_matmul_precision", precision)
        DF = map_jacobian_at(map_func, quad_x)
        det = jnp.linalg.det(DF)
        return np.asarray(DF), np.asarray(det)

    DF_ref, det_ref = measure("highest")
    DF_got, det_got = measure(setting)

    scale = np.abs(DF_ref).max()
    df_err = float(np.abs(DF_got - DF_ref).max() / scale)
    det_err = float(np.abs(det_got - det_ref).max() / np.abs(det_ref).max())
    out = {
        "device": dev.device_kind, "dtype": str(mrx.DTYPE),
        "matmul_precision": setting, "ns": list(ns), "p": cli.p,
        "det_min": float(det_got.min()), "det_max": float(det_got.max()),
        "det_min_reference": float(det_ref.min()),
        "det_max_reference": float(det_ref.max()),
        "folds": bool(det_got.min() <= 0.0),
        "max_relative_DF_error": df_err,
        "max_relative_det_error": det_err,
    }
    print(f"  det DF          [{out['det_min']:.4e}, {out['det_max']:.4e}]")
    print(f"  reference       [{out['det_min_reference']:.4e}, "
          f"{out['det_max_reference']:.4e}]  (same process, 'highest')")
    print(f"  folds           {out['folds']}")
    print(f"  max rel err DF  {df_err:.3e}")
    print(f"  max rel err det {det_err:.3e}")

    with open(cli.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
