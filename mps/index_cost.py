"""How big the indexed assembly's constant index is, and what it costs to build.

``_element_index`` runs in Python at trace time and the result is baked into
the executable as a constant. This reports, per component of each mass plan,
the bytes of that constant and the time to build it, at two meshes. The
kernel traces it twice per component (the gather and the scatter), so the
resident cost is that many copies until XLA folds duplicates.

    MRX_X64=0 python mps/index_cost.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("MRX_X64", "0")


def report(geometry: str, ns: tuple[int, int, int], p: int) -> None:
    """Build one sequence and price the index of every mass component.

    Args:
        geometry: VMEC wout the sequence is built on.
        ns: Resolution.
        p: Spline degree.
    """
    import jax.numpy as jnp

    from mrx.geometry import build_sequence
    from mrx.mass import _element_index

    t0 = time.perf_counter()
    seq, _ = build_sequence(geometry, ns, p)
    print(f"\n=== {ns} p={p}  built {time.perf_counter() - t0:.0f} s ===")
    total = 0
    for k in range(4):
        plan = seq.mass_plan[k]
        seen: list[tuple] = []
        for component in (*plan.gather_plans, *plan.shift_plans):
            if component in seen:
                continue
            seen.append(component)
            t1 = time.perf_counter()
            idx = _element_index(component)
            build_ms = 1e3 * (time.perf_counter() - t1)
            # The device copy is what the trace actually pays, once per use.
            t2 = time.perf_counter()
            device = jnp.asarray(idx)
            device.block_until_ready()
            copy_ms = 1e3 * (time.perf_counter() - t2)
            nbytes = int(idx.nbytes)
            total += nbytes
            print(f"  k={k} {component}  {idx.shape}  {nbytes / 1e6:6.2f} MB"
                  f"  numpy {build_ms:7.1f} ms  device copy {copy_ms:6.1f} ms")
        print(f"  k={k}: {len(seen)} distinct component plans")
    print(f"  distinct-plan total {total / 1e6:.2f} MB"
          f"  (times two uses, gather and scatter, if not folded)")


def main() -> None:
    """Price the index at the benchmark mesh and one step larger."""
    geometry = "data/wout_li383_low_res_reference.nc"
    report(geometry, (12, 24, 12), 3)
    report(geometry, (16, 32, 16), 3)


if __name__ == "__main__":
    main()
