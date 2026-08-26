"""
NFS tensor-grid helpers: logical eval-point grids and map residual statistics.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _axis_index_lookup(axis: np.ndarray, x: float, atol: float) -> int:
    """Find *i* such that ``abs(axis[i] - x) <= atol``."""
    j = int(np.searchsorted(axis, x))
    for idx in (j, j - 1):
        if 0 <= idx < len(axis) and abs(float(axis[idx]) - x) <= atol:
            return idx
    raise ValueError(
        f"Coordinate {x!r} not aligned with axis (tol={atol}); "
        f"nearest span [{axis[0]}, {axis[-1]}]."
    )


def mrx_logical_eval_points_index_grid(
    grid_shape: tuple[int, int, int],
    *,
    flip_r: bool = False,
) -> np.ndarray:
    """
    Build MRX logical ``(ρ, θ, ζ) ∈ [0, 1]³`` from tensor indices (no periodic wrap).

    Row-major flatten order matches ``R_3d.ravel()`` / GVEC export layout
  ``(nρ, nθ, nζ)`` with ``indexing='ij'``.  Uses ``θ = j/nθ``, ``ζ = k/nζ`` so the
    poloidal/toroidal **endpoints are not folded** to 0 (avoids duplicate logical
    cells when physical angles include ``2π``).
    """
    nr, nt, nz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    ir = np.arange(nr, dtype=np.float64)
    it = np.arange(nt, dtype=np.float64)
    iz = np.arange(nz, dtype=np.float64)
    Ir, It, Iz = np.meshgrid(ir, it, iz, indexing="ij")
    r_mrx = Ir / max(nr - 1, 1)
    theta_mrx = It / max(nt, 1)
    zeta_mrx = Iz / max(nz, 1)
    if flip_r:
        r_mrx = 1.0 - r_mrx
    return np.stack(
        [r_mrx.ravel(), theta_mrx.ravel(), zeta_mrx.ravel()], axis=1
    )


def repair_tensor_eval_points_if_needed(
    pts: np.ndarray,
    *,
    flip_r: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    If ``pts`` are a perfect cube but unique axes do not multiply to ``N``, rebuild
    logical coordinates from grid indices (GVEC export with ``angle % 1``).

    Returns ``(pts_out, repaired)``; ``repaired`` is ``True`` when coordinates were
    replaced (``R``, ``Z``, ``B`` row order is unchanged).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = int(pts.shape[0])
    side = int(round(n ** (1.0 / 3.0)))
    if side**3 != n:
        return pts, False
    u = np.unique(pts[:, 0])
    v = np.unique(pts[:, 1])
    w = np.unique(pts[:, 2])
    if int(len(u) * len(v) * len(w)) == n:
        return pts, False
    fixed = mrx_logical_eval_points_index_grid((side, side, side), flip_r=flip_r)
    print(
        "WARNING: eval_points are not a full tensor grid after unique-axis check "
        f"(N={n}, nρ={len(u)}, nθ={len(v)}, nζ={len(w)}, product={len(u) * len(v) * len(w)}). "
        "Replacing logical (ρ,θ,ζ) with index-based MRX coordinates "
        f"({side}×{side}×{side}, θ=j/nθ, ζ=k/nζ) so R,Z,B row order is preserved. "
        "Re-export with index-based eval_points to silence this.",
        flush=True,
    )
    return fixed, True


def _tensor_axes_and_grids(
    pts: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    atol: float = 1e-9,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """
    Infer sorted logical axes and 3-D *R*, *Z* grids from points on a full tensor grid.

    Points must be samples on a Cartesian product of one-dimensional coordinate
    lists (Fortran / ``indexing='ij'`` flatten order is *not* required; cells are
    filled by matching coordinates).
    """
    pts = np.asarray(pts, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64).ravel()
    if pts.shape[1] != 3:
        raise ValueError(f"pts must have shape (N, 3); got {pts.shape}")
    n = pts.shape[0]
    if R.shape[0] != n or Z.shape[0] != n:
        raise ValueError("R, Z must have the same length as pts")

    u = np.unique(pts[:, 0])
    v = np.unique(pts[:, 1])
    w = np.unique(pts[:, 2])
    n1, n2, n3 = len(u), len(v), len(w)
    if n1 * n2 * n3 != n:
        raise ValueError(
            f"Data do not form a full tensor grid in (ρ, θ, ζ): N={n} samples, but "
            f"unique coordinates along each axis are nρ={n1}, nθ={n2}, nζ={n3}, "
            f"so a volume product grid would need nρ·nθ·nζ={n1 * n2 * n3} points (not {n}). "
            "This usually means the file is a surface slice, scattered samples, or "
            "non-Cartesian-product points; a precomputed **volume** NFS HDF5 "
            "(e.g. from `precompute_desc_data.py`) has eval_points, R, Z on a full "
            "rho–theta–zeta product grid."
        )

    R_grid = np.full((n1, n2, n3), np.nan, dtype=np.float64)
    Z_grid = np.full((n1, n2, n3), np.nan, dtype=np.float64)
    count = np.zeros((n1, n2, n3), dtype=np.int32)

    for row in range(n):
        i = _axis_index_lookup(u, float(pts[row, 0]), atol)
        j = _axis_index_lookup(v, float(pts[row, 1]), atol)
        k = _axis_index_lookup(w, float(pts[row, 2]), atol)
        R_grid[i, j, k] = R[row]
        Z_grid[i, j, k] = Z[row]
        count[i, j, k] += 1

    if np.any(count != 1):
        raise ValueError(
            "Duplicate or missing grid cells when mapping scattered points to "
            f"the tensor grid (count min={int(count.min())}, max={int(count.max())})."
        )
    return (u, v, w), R_grid, Z_grid


def evaluate_map_rz_residual_stats(
    map_func: Callable,
    pts,
    R,
    Z,
    *,
    exclude_axis_tol: float = 0.0,
    rho_lo: float | None = None,
    rho_hi: float | None = None,
) -> dict[str, float]:
    """
    Pointwise map misfit of fitted ``map_func`` against sampled ``R``, ``Z``.

    Returns max and RMS absolute errors over the evaluation mask.

    Parameters
    ----------
    map_func
        Fitted stellarator map ``(ρ, θ, ζ) -> (X, Y, Z)``.
    pts, R, Z
        Logical grid samples and cylindrical targets (same length).
    exclude_axis_tol
        Exclude points with ``ρ <= tol`` or ``ρ >= 1 - tol``.
    rho_lo, rho_hi
        If set, further restrict to ``rho_lo < ρ < rho_hi`` (interior shell).

    Returns
    -------
    dict
        Keys ``max_R``, ``max_Z``, ``rms_R``, ``rms_Z``, ``n_points``.
    """
    pts_j = jnp.asarray(pts, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64).ravel()
    Z_j = jnp.asarray(Z, dtype=jnp.float64).ravel()
    xyz = jax.vmap(map_func)(pts_j)
    R_pred = jnp.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    Z_pred = xyz[:, 2]
    dR = jnp.abs(R_pred - R_j)
    dZ = jnp.abs(Z_pred - Z_j)

    mask = jnp.ones(pts_j.shape[0], dtype=bool)
    if exclude_axis_tol > 0.0:
        mask = mask & (pts_j[:, 0] > exclude_axis_tol) & (
            pts_j[:, 0] < 1.0 - exclude_axis_tol
        )
    if rho_lo is not None:
        mask = mask & (pts_j[:, 0] > float(rho_lo))
    if rho_hi is not None:
        mask = mask & (pts_j[:, 0] < float(rho_hi))
    if not bool(jnp.any(mask)):
        mask = jnp.ones(pts_j.shape[0], dtype=bool)

    dR_m = dR[mask]
    dZ_m = dZ[mask]
    return {
        "max_R": float(jnp.max(dR_m)),
        "max_Z": float(jnp.max(dZ_m)),
        "rms_R": float(jnp.sqrt(jnp.mean(dR_m ** 2))),
        "rms_Z": float(jnp.sqrt(jnp.mean(dZ_m ** 2))),
        "n_points": float(dR_m.shape[0]),
    }


