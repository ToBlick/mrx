"""The truncated-Fourier coarse correction (`fm`) — EXPERIMENTAL, opt-in.

This is a real result, not dead code: at the corrected boundary scale it is a
further **1.18x (rot-ellipse) / 1.32x (W7-X)** on TOTAL solve time over the
plain atom, and it beats the dense-probe ring arms outright at roughly a tenth
of their build cost (docs/research/natural_bc_coefficient_handoff.md §14.4,
§15.1).

It is NOT in the production preconditioner, and lives here rather than on
`MetricLumpingLaplacian` for reasons that are all cost rather than correctness:

* five parameters against the production class's zero;
* storage LINEAR in `n_dof` -- 102 MB at n=20 where the atom uses 0.1 MB;
* `m95 ~ n_t/3` (§14.6): the demanded angular cutoff grows with resolution, so
  a FIXED mode box captures a shrinking share and the method is asymptotically
  under-resolved;
* a correctness trap that has already been hit once -- the ADDITIVE form is
  structurally wrong for this atom (a HIGH outlier means `P` is already too
  large there, so adding can only make it worse) and is kept as a diagnostic
  only;
* untested at k=0 and under Dirichlet.

Wrapping rather than fusing costs nothing: with `coarse_rings=0` the atom's own
`apply` IS the `m_apply` this used to call inside its jit, and the hybrid form
needs no extra operator apply because `L Q` reuses the stored `L V`.

Usage::

    pre = CoarseCorrectedMetricLumping(seq, ops, k, dirichlet, coarse_modes=(3, 3))
    pre.apply(x)
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.metric_lumping_laplacian import (
    MetricLumpingLaplacian, trace_components,
)

def coarse_ring_basis(seq, k, dirichlet, rings, m_max, n_max, comps=None,
                      exclude=None):
    """Orthonormal columns spanning the outer rings x TRUNCATED Fourier modes.

    The dense core block is already an in-preconditioner coarse correction --
    `core_inv = (R L R^T)^-1` with `R` a SELECTION of the ring's rows, costing
    `n_t n_z` probe applies and an `(n_t n_z)^2` block. This generalises `R` to
    a RESTRICTION onto `|m| <= m_max, |n| <= n_max`, so the cost is one apply
    per coarse VECTOR: `(2 m_max+1)(2 n_max+1)` per component-ring, and it stops
    growing as the mesh refines.

    Justified by measurement: the outliers of `P L` sit on the outer rings
    (energy fraction 0.79-0.91) with LOW mode content (`|m|` 1.2-2.3 and,
    across a 6,12,6 -> 8,16,8 refinement, 1.39 -> 1.38 -- it does not drift).
    The face weight is banded the same way (99% of its energy inside
    `|m|<=3, |n|<=2` on W7-X).

    ``exclude`` drops rows already handled by REPLACEMENT (the dense core), so
    the correction stays additive without double-counting them.
    """
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows, cols = np.asarray(e.rows), np.asarray(e.cols)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)
    shapes = [tuple(int(v) for v in sh)
              for sh in getattr(seq, f"basis_{k}").shape]
    starts = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes])
    single = counts[rows] == 1
    r_s, c_s = rows[single], cols[single]
    comp = np.searchsorted(starts[1:], c_s, side="right")
    loc = c_s - starts[comp]

    cols_out = []
    for c, shape in enumerate(shapes):
        if comps is not None and c not in comps:
            continue
        nr, nt, nz = shape
        sel = comp == c
        if not sel.any():
            continue
        lidx, rid = loc[sel], r_s[sel]
        i_r = lidx // (nt * nz)
        i_t = (lidx // nz) % nt
        i_z = lidx % nz
        js, ks = np.arange(nt), np.arange(nz)
        for ring in range(max(0, nr - rings), nr):
            take = i_r == ring
            if not take.any():
                continue
            rr, tt, zz = rid[take], i_t[take], i_z[take]
            for m in range(0, m_max + 1):
                for n in range(-n_max, n_max + 1):
                    ph = 2.0 * np.pi * (m * js[:, None] / nt
                                        + n * ks[None, :] / nz)
                    for f in (np.cos, np.sin):
                        v = np.zeros(n_ext)
                        v[rr] = f(ph)[tt, zz]
                        cols_out.append(v)
    if not cols_out:
        return np.zeros((n_ext, 0))
    v_mat = np.stack(cols_out, axis=1)
    if exclude is not None and len(exclude):
        v_mat[np.asarray(exclude), :] = 0.0
    # cos/sin over the full (m, n) box is redundant by construction (m=0 pairs
    # n with -n); a pivoted QR drops the dependents and leaves an orthonormal
    # basis, which is what keeps the Galerkin block well conditioned.
    q_mat, r_mat = np.linalg.qr(v_mat)
    keep = np.abs(np.diag(r_mat)) > 1e-10 * np.abs(np.diag(r_mat)).max()
    return q_mat[:, keep]


def coarse_correction(seq, operators, k, dirichlet, v_mat, tol=1e-12,
                      trunc_rows=None):
    """``V (V^T L V)^-1 V^T`` and ``L V`` -- one apply per coarse column.

    MEASURED (rot-ellipse k=1 free, 6,12,6): used ADDITIVELY this raises
    `lambda_min` (2.52e-2 -> 3.55e-2) and removes low outliers (33 -> 29) but
    leaves the HIGH outliers untouched (9 -> 10) and `lambda_max` slightly
    worse (21.05 -> 21.90). That is structural, not a bug: a high outlier means
    `P` is too LARGE there (the atom is too soft), and `P + V A_0^-1 V^T` only
    makes it larger. Additive two-level Schwarz assumes the local part
    UNDER-resolves the coarse modes; this atom OVER-resolves them.

    Hence the hybrid (balancing) form, which REMOVES the atom's action on the
    coarse space instead of adding to it, and is still entirely inside `P`::

        P = Q + (I - Q L) M (I - L Q) ,    Q = V A_0^-1 V^T
    """
    from mrx.operators import apply_hodge_laplacian_approx  # noqa: PLC0415

    size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
    if v_mat.shape[1] == 0:
        return None
    apply_hodge_laplacian_approx(seq, operators, jnp.zeros(size), k,
                                 dirichlet=dirichlet)
    lv = np.stack([np.asarray(apply_hodge_laplacian_approx(
        seq, operators, jnp.asarray(v_mat[:, j]), k, dirichlet=dirichlet))
        for j in range(v_mat.shape[1])], axis=1)
    a0 = v_mat.T @ lv
    a0 = 0.5 * (a0 + a0.T)
    w, u = np.linalg.eigh(a0)
    keep = np.abs(w) > tol * np.abs(w).max()
    # ``lv`` is kept: the hybrid form needs ``L Q`` every apply, and
    # ``L Q x = (L V) A_0^-1 (V^T x)`` is a dense matvec against it -- so the
    # correction costs NO extra operator apply.
    if trunc_rows is not None:
        # MEASURED: |LV|^2 is 99.5% on the outer ring itself and 0.5% one ring
        # in (`fm_cost.py`, rot-ellipse n=12 k=1), so both V and LV live on a
        # thin slab. A_0 = V^T (L V) samples LV only on V's support, so the
        # Galerkin block is UNCHANGED by this; only the (I - L Q) factors move,
        # by 0.5%, inside a preconditioner.
        return (v_mat[trunc_rows], (u[:, keep] / w[keep]) @ u[:, keep].T,
                lv[trunc_rows], trunc_rows)
    return v_mat, (u[:, keep] / w[keep]) @ u[:, keep].T, lv, None

class CoarseCorrectedMetricLumping:
    """`MetricLumpingLaplacian` plus a truncated-Fourier coarse correction.

    Holds the atom rather than subclassing it, so the production class keeps
    no knowledge of this and the two can be compared directly.
    """

    def __init__(self, seq, operators, k, dirichlet, *, coarse_rings=1,
                 coarse_modes=(3, 3), coarse_set="all", coarse_mode="hybrid",
                 coarse_trunc=0, **atom_kwargs):
        self.atom = MetricLumpingLaplacian(seq, operators, k, dirichlet,
                                         **atom_kwargs)
        self.coarse = None
        self.coarse_mode = coarse_mode
        self.n_coarse = 0
        if coarse_rings > 0 and not (dirichlet and coarse_set == "trace"):
            _tr = trace_components(k)
            _ot = tuple(c for c in range(len(self.atom.shapes))
                        if c not in _tr)
            cset = {"all": None, "trace": _tr, "other": _ot}[coarse_set]
            # `exclude` drops the rows the atom's dense core already REPLACES,
            # so the correction stays additive without double-counting them.
            v_mat = coarse_ring_basis(
                seq, k, dirichlet, coarse_rings, int(coarse_modes[0]),
                int(coarse_modes[1]), comps=cset,
                exclude=self.atom.probe_rows)
            trunc_rows = None
            if coarse_trunc:
                slab = coarse_ring_basis(
                    seq, k, dirichlet, coarse_rings + int(coarse_trunc),
                    0, 0, comps=cset, exclude=None)
                trunc_rows = np.flatnonzero(
                    np.abs(slab).sum(axis=1) > 0) if slab.size else None
            self.coarse = coarse_correction(seq, operators, k, dirichlet,
                                            v_mat, trunc_rows=trunc_rows)
            self.n_coarse = int(v_mat.shape[1])
        self._jit = None

    def _build_apply(self):
        m_apply = self.atom.apply
        if self.coarse is None:
            return m_apply
        v_mat, a0inv, lv = (jnp.asarray(self.coarse[0]),
                            jnp.asarray(self.coarse[1]),
                            jnp.asarray(self.coarse[2]))
        rows = (None if self.coarse[3] is None
                else jnp.asarray(self.coarse[3]))
        hybrid = self.coarse_mode == "hybrid"

        def impl(x):
            if rows is not None:
                # V and LV are held only on the slab; gather once, scatter back.
                xs = x[rows]
                vtx = v_mat.T @ xs
                z = jnp.zeros_like(x).at[rows].set(v_mat @ (a0inv @ vtx))
                if not hybrid:
                    return m_apply(x) + z
                y = x.at[rows].add(-(lv @ (a0inv @ vtx)))
                w = m_apply(y)
                return z - jnp.zeros_like(x).at[rows].set(
                    v_mat @ (a0inv @ (lv.T @ w[rows]))) + w
            vtx = v_mat.T @ x
            z = v_mat @ (a0inv @ vtx)
            if not hybrid:
                # ADDITIVE -- diagnostic only. It cannot cure a HIGH outlier,
                # because P is already too large there; see coarse_correction.
                return m_apply(x) + z
            # HYBRID / balancing: P = Q + (I - Q L) M (I - L Q). Removes the
            # atom's action on the coarse space instead of adding to it, stays
            # symmetric, and costs NO extra operator apply because L Q uses the
            # stored L V.
            y = x - lv @ (a0inv @ vtx)
            w = m_apply(y)
            return z + w - v_mat @ (a0inv @ (lv.T @ w))

        return jax.jit(impl)

    def apply(self, x):
        """Apply the coarse-corrected preconditioner."""
        if self._jit is None:
            self._jit = self._build_apply()
        return self._jit(x)
