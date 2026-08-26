"""How many derivative splines are nonzero at the clamped endpoint?

No geometry, no sequence, no solve -- just the 1-D bases. Answers whether the
trace u_r(1) = sum_i c_i dLam_i(1) is ONE dof (so evicting the last row imposes
it) or several (so it is a rank-one CONSTRAINT and row removal is the wrong
operation).
"""

import jax.numpy as jnp
import numpy as np

from mrx.spline_bases import DerivativeSpline, SplineBasis

for p in (1, 2, 3, 4):
    for n in (8, 12):
        b = SplineBasis(n, p, "clamped")
        d = DerivativeSpline(b)
        for eps in (1e-8, 1e-12):
            x = 1.0 - eps
            e = np.asarray([float(jnp.sum(d(x, i))) for i in range(d.n)])
            v = np.asarray([float(jnp.sum(b(x, i))) for i in range(b.n)])
            nz = np.flatnonzero(np.abs(e) > 1e-10 * (np.abs(e).max() + 1e-300))
            nzv = np.flatnonzero(np.abs(v) > 1e-10 * (np.abs(v).max() + 1e-300))
            print(f"p={p} n={n} eps={eps:g}  dLam: n_d={d.n} "
                  f"nonzero={len(nz)} at {nz.tolist()} "
                  f"vals={np.array2string(e[nz], precision=4)}", flush=True)
            if eps == 1e-8:
                print(f"        Lam (value basis): nonzero={len(nzv)} "
                      f"at {nzv.tolist()} "
                      f"vals={np.array2string(v[nzv], precision=4)}", flush=True)
        # ratio that decides whether row-eviction ~ trace constraint
        e = np.asarray([float(jnp.sum(d(1.0 - 1e-8, i))) for i in range(d.n)])
        srt = np.sort(np.abs(e))
        print(f"        |e|_2nd/|e|_max = {srt[-2] / (srt[-1] + 1e-300):.4e}",
              flush=True)
