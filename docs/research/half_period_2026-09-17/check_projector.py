"""Device free-space projector against the exact host one, and the gram structure of E E^T."""
import os
os.environ.setdefault("MRX_DTYPE", "float64")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from mrx.geometry import build_sequence  # noqa: E402
from mrx.symmetry import _extraction_gram_core, free_projector  # noqa: E402

half, _ = build_sequence("data/wout_li383_low_res_reference.nc", (8, 12, 12), 2, symmetry="stellarator")
rng = np.random.default_rng(2)
for k in range(4):
    for d in (True, False):
        e = half.E(k, d)
        n_free, n_raw = (int(v) for v in e.forward_shape)
        E = sparse.csr_matrix((np.asarray(e.vals, dtype=np.float64), (np.asarray(e.rows), np.asarray(e.cols))),
                              shape=(n_free, n_raw))
        gram = (E @ E.T).tocsr()
        core, inv = _extraction_gram_core(half, k, d)
        bulk = np.setdiff1d(np.arange(n_free), core)
        diag = gram.diagonal()
        off = gram.copy()
        off.setdiag(0)
        off_bulk = np.abs(off[bulk]).max() if bulk.size else 0.0
        cross = np.abs(gram[np.ix_(core, bulk)]).max() if core.size and bulk.size else 0.0
        print(f"k={k} d={d}: n_free {n_free}, core rows {core.size}; bulk gram diag in [{diag[bulk].min():.6f}, {diag[bulk].max():.6f}], "
              f"max off-diagonal on bulk rows {off_bulk:.2e}, core-bulk coupling {cross:.2e}", flush=True)
        project = free_projector(half, k, d)
        for parity in (1, -1):
            y = jnp.asarray(rng.standard_normal(n_free))
            x = half.project_parity(jnp.asarray(rng.standard_normal(n_free)), k, parity, d)
            dev = project.post(y, project.parity(x))
            host = half.project_parity(y, k, parity, d)
            print(f"   parity {parity:+d}: device vs host projector {float(jnp.max(jnp.abs(dev - host)) / jnp.max(jnp.abs(host))):.2e}",
                  flush=True)
