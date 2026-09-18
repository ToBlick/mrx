"""Is the polar (extracted) space invariant under the reflection? For random
conforming raw vectors x = E^T v, the non-conformity of R x."""
import os
os.environ.setdefault("MRX_DTYPE", "float64")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mrx.geometry import build_sequence  # noqa: E402
from mrx.projectors import _conforming_restriction  # noqa: E402
from mrx.symmetry import reflect  # noqa: E402

half, _ = build_sequence("data/wout_li383_low_res_reference.nc", (8, 12, 12), 2, symmetry="stellarator")
rng = np.random.default_rng(3)
for k in range(4):
    for d in (True, False):
        e = half.E(k, d)
        plan = half.reflection_plan[k]
        v = jnp.asarray(rng.standard_normal(half.n(k, d)))
        x = e.T @ v
        y = reflect(x, plan)
        back = e.T @ _conforming_restriction(e, y)
        nonconf = float(jnp.linalg.norm(back - y) / jnp.linalg.norm(y))
        # where: per component, the max over the raw grid of |back - y|
        where = []
        off = 0
        for perm_t, perm_z, s, shape in plan:
            n_c = int(np.prod(shape))
            dcomp = np.asarray(jnp.abs(back - y)[off:off + n_c]).reshape(shape)
            r_prof = dcomp.max(axis=(1, 2))
            where.append(f"comp shape {shape}: max {dcomp.max():.2e}, by radial index {np.array2string(r_prof[:4], precision=1)}")
            off += n_c
        print(f"k={k} d={d}: |R x - restrict(R x)| / |R x| = {nonconf:.2e}", flush=True)
        for w in where:
            print("    " + w, flush=True)
