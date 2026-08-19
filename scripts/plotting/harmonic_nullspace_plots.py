#!/usr/bin/env python3
"""
Helper for harmonic form plotting. 
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import mrx
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import extend_map_nfp
from mrx.plotting import get_2d_grids, set_axes_equal

try:
    from mrx.plotting import apply_serif_cm_style

    apply_serif_cm_style()
except ImportError:
    pass  # greville-prod plotting lacks apply_serif_cm_style


def _eval_pts(field: Any, pts: jnp.ndarray) -> jnp.ndarray:
    """Batch-evaluate a discrete field / pushforward at logical points."""
    return jax.lax.map(field, pts, batch_size=mrx.MAP_BATCH_SIZE_OUTER)


def _make_full_torus_pushforward(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    nfp: int,
) -> tuple[Pushforward, Callable[[jnp.ndarray], jnp.ndarray]]:
    """Localized 1-form DOFs + ``extend_map_nfp`` (same as QUASR hcurl driver)."""
    disc = DiscreteFunction(v, seq.basis_1, seq.e1)
    nfp_i = int(nfp)

    def localized(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64).reshape(3)
        xi = x[2] * float(nfp_i)
        zloc = xi - jnp.floor(xi)
        return disc(x.at[2].set(zloc))

    full_map = extend_map_nfp(map_raw, nfp_i)
    return Pushforward(localized, full_map, 1), full_map


def _poloidal_slice_grid(
    zeta: float,
    nx: int,
    ny: int,
    *,
    rho_lo: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rho_lo = float(max(rho_lo, 1e-6))
    r_axis = jnp.linspace(rho_lo, 1.0 - 1e-6, nx)
    theta_axis = jnp.linspace(0.0, 1.0, ny, endpoint=False)
    r_grid, theta_grid = jnp.meshgrid(r_axis, theta_axis, indexing="ij")
    zeta_grid = jnp.full_like(r_grid, float(zeta))
    pts = jnp.stack([r_grid, theta_grid, zeta_grid], axis=-1).reshape(-1, 3)
    return pts, r_grid, theta_grid


def _norm_mag_panels(
    magnitudes: list[np.ndarray],
    *,
    low_contrast_frac: float = 0.03,
    pct_lo: float = 2.0,
    pct_hi: float = 98.0,
) -> tuple[float, float]:
    """Min/max for ``Normalize`` across panels; robust to outliers and low contrast."""
    flat = np.concatenate(
        [np.asarray(m, dtype=np.float64).ravel() for m in magnitudes]
    )
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    vmin_raw, vmax_raw = float(flat.min()), float(flat.max())
    lo, hi = float(np.percentile(flat, pct_lo)), float(np.percentile(flat, pct_hi))
    span_raw = vmax_raw - vmin_raw
    span_pct = hi - lo
    use_pct = span_pct > 0.0 and (
        span_raw > 2.5 * span_pct
        or span_raw / (abs(vmax_raw) + 1e-30) < low_contrast_frac
    )
    if use_pct:
        if hi <= lo:
            return vmin_raw, vmax_raw + 1e-30
        return lo, hi
    if span_raw <= 0.0:
        return vmin_raw, vmax_raw + 1e-30
    return vmin_raw, vmax_raw


def _relative_percentile_span(values: np.ndarray, *, pct_lo: float = 2.0, pct_hi: float = 98.0) -> float:
    """Relative (p_hi - p_lo) / median for finite samples."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0
    lo, hi = float(np.percentile(flat, pct_lo)), float(np.percentile(flat, pct_hi))
    med = float(np.median(flat))
    return (hi - lo) / (abs(med) + 1e-30)


def _pick_surface_colormap_magnitude(
    mag_phys: np.ndarray,
    mag_log: np.ndarray,
    *,
    low_contrast_frac: float = 0.08,
) -> tuple[np.ndarray, str]:
    """Pick |u| for outer-surface coloring (logical often shows structure on perturbed maps)."""
    mag_phys = np.asarray(mag_phys, dtype=np.float64)
    mag_log = np.asarray(mag_log, dtype=np.float64)
    mag_phys_safe = np.where(np.isfinite(mag_phys), mag_phys, mag_log)

    rel_phys = _relative_percentile_span(mag_phys_safe)
    rel_log = _relative_percentile_span(mag_log)

    flat_p = mag_phys_safe[np.isfinite(mag_phys_safe)]
    outlier_stretch = False
    if flat_p.size:
        vmin_r, vmax_r = float(flat_p.min()), float(flat_p.max())
        lo, hi = float(np.percentile(flat_p, 2.0)), float(np.percentile(flat_p, 98.0))
        outlier_stretch = (vmax_r - vmin_r) > 2.5 * max(hi - lo, 1e-30)

    use_logical = (
        rel_log > rel_phys * 1.25
        or rel_phys < low_contrast_frac
        or outlier_stretch
    )
    if use_logical:
        reasons: list[str] = []
        if rel_phys < low_contrast_frac:
            reasons.append("low pushforward contrast")
        if outlier_stretch:
            reasons.append("pushforward outliers")
        if rel_log > rel_phys * 1.25:
            reasons.append("logical |u| has more structure")
        print(
            "hcurl plot colormap: using logical |u| "
            f"({', '.join(reasons) or 'auto'})",
            flush=True,
        )
        return mag_log, "logical "

    print(
        "hcurl plot colormap: using pushforward |u| "
        f"(relative 2–98% span {rel_phys:.3g})",
        flush=True,
    )
    return mag_phys_safe, ""


def _surface_colors(
    values: np.ndarray,
    *,
    pct_lo: float = 2.0,
    pct_hi: float = 98.0,
) -> tuple[np.ndarray, float, float]:
    """Map scalar samples to RGBA; ignore NaNs when setting the color scale.

    Uses robust percentile limits when a few outliers would otherwise compress
    the colormap (typical on outer-surface pushforward plots).
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        normalized = np.zeros_like(arr, dtype=np.float64)
        return plt.cm.plasma(normalized), 0.0, 1.0
    flat = arr[finite]
    vmin_raw, vmax_raw = float(flat.min()), float(flat.max())
    lo = float(np.percentile(flat, pct_lo))
    hi = float(np.percentile(flat, pct_hi))
    if hi > lo:
        vmin, vmax = lo, hi
    else:
        vmin, vmax = vmin_raw, vmax_raw
    print(
        f"outer-surface |u| color scale: [{vmin:.4g}, {vmax:.4g}] "
        f"(raw min/max [{vmin_raw:.4g}, {vmax_raw:.4g}])",
        flush=True,
    )
    if vmax <= vmin:
        vmax = vmin + 1e-12
    normalized = np.full(arr.shape, np.nan, dtype=np.float64)
    normalized[finite] = np.clip((arr[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color=(0.12, 0.04, 0.22, 1.0))
    return cmap(normalized), vmin, vmax


def _surface_view_direction(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    return np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ]
    )


def _surface_normals(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, major_radius: float) -> np.ndarray:
    tangent_theta = np.stack(
        [np.gradient(X, axis=0), np.gradient(Y, axis=0), np.gradient(Z, axis=0)],
        axis=-1,
    )
    tangent_zeta = np.stack(
        [np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)],
        axis=-1,
    )
    normals = np.cross(tangent_theta, tangent_zeta)
    nn = np.linalg.norm(normals, axis=-1, keepdims=True)
    nn = np.where(nn > 0.0, nn, 1.0)
    normals = normals / nn
    phi = np.arctan2(-Y, X)
    centerline = np.stack(
        [
            major_radius * np.cos(phi),
            -major_radius * np.sin(phi),
            np.zeros_like(Z),
        ],
        axis=-1,
    )
    outward = np.stack([X, Y, Z], axis=-1) - centerline
    flip_mask = np.sum(normals * outward, axis=-1, keepdims=True) < 0.0
    return np.where(flip_mask, -normals, normals)


def _subplots_with_right_colorbar(
    n_panels: int, *, fig_h: float = 4.8
) -> tuple[Any, np.ndarray, Any]:
    """One row of ``n_panels`` axes plus a dedicated narrow colorbar column on the right."""
    fig_w = 5.0 * n_panels + 0.85
    fig = plt.figure(figsize=(fig_w, fig_h))
    wr = [1.0] * n_panels + [0.075]
    gs = GridSpec(1, n_panels + 1, figure=fig, width_ratios=wr, wspace=0.28)
    axes_arr = np.array([fig.add_subplot(gs[0, i]) for i in range(n_panels)])
    cax = fig.add_subplot(gs[0, n_panels])
    return fig, axes_arr, cax



def _save_row_figure(
    fig: Any,
    path: Path,
    axes_arr: np.ndarray,
    *,
    clean: bool,
    suptitle: str | None = None,
    shared_ylabel: str | None = None,
) -> None:
    """Save a one-row multi-panel figure with room for suptitle and shared y label."""
    if not clean:
        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=12)
        if shared_ylabel is not None:
            fig.supylabel(shared_ylabel, x=0.035)
            for ax in np.atleast_1d(axes_arr).flat:
                ax.set_ylabel("")
        fig.subplots_adjust(top=0.82, left=0.11, right=0.90, wspace=0.28)
        fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.35)
    else:
        fig.subplots_adjust(left=0.06, top=0.98, bottom=0.08)
        fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(path)


def plot_logical_rtheta(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    *,
    zetas: tuple[float, ...],
    path: Path,
    nx: int,
    ny: int,
    quiver_stride_r: int,
    quiver_stride_t: int,
    clean: bool = False,
    magnitude_logical: bool = False,
) -> None:
    """Reference logical slice: $|u|$ on $(\\rho,\\theta)$ at fixed $\\zeta$ (no physical pushforward).

    ``clean=True``: no tick labels/frame or suptitle; colorbar kept.
    """
    disc = DiscreteFunction(v, seq.basis_1, seq.e1)
    fig, axes_arr, cax = _subplots_with_right_colorbar(len(zetas))
    blocks: list[tuple[Any, ...]] = []
    for zc in zetas:
        logical_points, r_grid, theta_grid = _poloidal_slice_grid(zc, nx, ny)
        values = np.asarray(_eval_pts(disc, logical_points)).reshape(nx, ny, 3)
        magnitude = np.linalg.norm(values, axis=-1)
        vr = values[..., 0]
        vtheta = values[..., 1]
        blocks.append((r_grid, theta_grid, magnitude, vr, vtheta))
    vmin, vmax = _norm_mag_panels([b[2] for b in blocks])
    if vmax <= vmin:
        vmax = vmin + 1e-30
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = None
    for i, (ax, zc, (r_grid, theta_grid, magnitude, vr, vtheta)) in enumerate(
        zip(axes_arr.flat, zetas, blocks)
    ):
        mappable = ax.pcolormesh(
            np.asarray(r_grid),
            np.asarray(theta_grid),
            magnitude,
            shading="auto",
            cmap="plasma",
            norm=norm,
            zorder=1,
            rasterized=True,
        )
        ax.quiver(
            np.asarray(r_grid)[::quiver_stride_r, ::quiver_stride_t],
            np.asarray(theta_grid)[::quiver_stride_r, ::quiver_stride_t],
            vr[::quiver_stride_r, ::quiver_stride_t],
            vtheta[::quiver_stride_r, ::quiver_stride_t],
            color="white",
            pivot="mid",
            scale=25,
            linewidth=0.25,
            width=0.002,
            alpha=0.45,
            zorder=2,
        )
        if clean:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_frame_on(False)
        else:
            ax.set_xlabel(r"$\rho_{\mathrm{log}}$")
            ax.set_title(rf"$\zeta={zc:.2f}$")
            if i > 0:
                ax.tick_params(axis="y", labelleft=True)
    assert mappable is not None
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(r"$|u|$")
    _save_row_figure(
        fig,
        path,
        axes_arr,
        clean=clean,
        suptitle=None
        if clean
        else r"Free $k=1$ H(curl) nullspace: $|\mathbf{u}|$ in logical $(\rho,\theta)$",
        shared_ylabel=None if clean else r"$\theta_{\mathrm{log}}$",
    )


def plot_poloidal_RZ(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    zetas: tuple[float, ...],
    path: Path,
    nx: int,
    ny: int,
    quiver_stride_r: int,
    quiver_stride_t: int,
    clean: bool = False,
    magnitude_logical: bool = False,
) -> None:
    """Physical poloidal slice via ``extend_map_nfp`` + ``Pushforward``.

    ``clean=True``: no tick labels/frame or suptitle; colorbar kept.
    """
    fld, full_map = _make_full_torus_pushforward(seq, v, map_raw, nfp)
    map_ev = jax.jit(full_map)
    disc = DiscreteFunction(v, seq.basis_1, seq.e1)
    # Skip rho→0: all poloidal angles collapse to the axis in (R,Z), producing a spurious
    # radial "ray" in pcolormesh on perturbed / coarse maps.
    rho_lo = 0.05
    blocks: list[tuple[Any, ...]] = []
    mag_phys_all: list[np.ndarray] = []
    mag_log_all: list[np.ndarray] = []
    for zc in zetas:
        logical_points, _, _ = _poloidal_slice_grid(zc, nx, ny, rho_lo=rho_lo)
        mapped = np.asarray(_eval_pts(map_ev, logical_points)).reshape(nx, ny, 3)
        uv = np.asarray(_eval_pts(fld, logical_points)).reshape(nx, ny, 3)
        uv_log = np.asarray(_eval_pts(disc, logical_points)).reshape(nx, ny, 3)
        X, Y, Z = mapped[..., 0], mapped[..., 1], mapped[..., 2]
        R = np.sqrt(X**2 + Y**2)
        mag_phys = np.linalg.norm(uv, axis=-1)
        mag_log = np.linalg.norm(uv_log, axis=-1)
        mag_phys_all.append(mag_phys)
        mag_log_all.append(mag_log)
        radial_denom = np.where(R > 0.0, R, 1.0)
        vR = (X * uv[..., 0] + Y * uv[..., 1]) / radial_denom
        vZ = uv[..., 2]
        blocks.append((R, Z, mag_phys, mag_log, vR, vZ))
    stack_phys = np.concatenate([m.ravel() for m in mag_phys_all])
    stack_log = np.concatenate([m.ravel() for m in mag_log_all])
    if magnitude_logical:
        print("hcurl plot colormap: using logical |u| (--plot-magnitude-logical)", flush=True)
        mag_label = "logical "
        use_logical_color = True
    else:
        _, mag_label = _pick_surface_colormap_magnitude(stack_phys, stack_log)
        use_logical_color = bool(mag_label)
    mags_for_norm = mag_log_all if use_logical_color else mag_phys_all
    vmin, vmax = _norm_mag_panels(mags_for_norm)
    if vmax <= vmin:
        vmax = vmin + 1e-30
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes_arr, cax = _subplots_with_right_colorbar(len(zetas))
    mappable = None
    for ax, zc, (R, Z, mag_phys, mag_log, vR, vZ) in zip(axes_arr.flat, zetas, blocks):
        magnitude = mag_log if use_logical_color else mag_phys
        mappable = ax.pcolormesh(
            R,
            Z,
            magnitude,
            shading="auto",
            cmap="plasma",
            norm=norm,
            zorder=1,
            rasterized=True,
        )
        ax.quiver(
            R[::quiver_stride_r, ::quiver_stride_t],
            Z[::quiver_stride_r, ::quiver_stride_t],
            vR[::quiver_stride_r, ::quiver_stride_t],
            vZ[::quiver_stride_r, ::quiver_stride_t],
            color="white",
            pivot="mid",
            scale=25,
            linewidth=0.25,
            width=0.002,
            alpha=0.45,
            zorder=2,
        )
        ax.set_aspect("equal")
        if clean:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_frame_on(False)
        else:
            ax.set_xlabel(r"$R$")
            ax.set_title(rf"$\zeta={zc:.2f}$")
    assert mappable is not None
    cb = fig.colorbar(mappable, cax=cax)
    cbar_label = r"$|u|$ (logical)" if use_logical_color else r"$|u|$"
    cb.set_label(cbar_label)
    poloidal_title = (
        r"Free $k=1$ H(curl) nullspace: "
        + (r"logical $|\mathbf{u}|$ on poloidal $(R,Z)$" if use_logical_color else r"$|\mathbf{u}|$ on poloidal $(R,Z)$")
        + r" (GVEC, full torus)"
    )
    _save_row_figure(
        fig,
        path,
        axes_arr,
        clean=clean,
        suptitle=None if clean else poloidal_title,
        shared_ylabel=None if clean else r"$Z$",
    )


def _longest_contiguous_true_span(ok: np.ndarray) -> tuple[int, int] | None:
    """
    Return ``(start, stop)`` so ``ok[start:stop]`` is the longest all-``True`` run.

    If there are no ``True`` entries, return ``None``.
    """
    n = int(ok.size)
    best_lo, best_hi = 0, 0
    i = 0
    while i < n:
        if not bool(ok[i]):
            i += 1
            continue
        j = i
        while j < n and bool(ok[j]):
            j += 1
        if j - i > best_hi - best_lo:
            best_lo, best_hi = i, j
        i = j
    if best_hi <= best_lo:
        return None
    return best_lo, best_hi


def plot_outer_surface(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    path: Path,
    ntheta: int,
    nzeta: int,
    quiver_stride_theta: int,
    quiver_stride_zeta: int,
    quiver_length: float,
    quiver_offset: float,
    surface_rho: float = 1.0 - 1e-5,
    half_cut: bool = False,
    axis_full: bool = False,
    clean: bool = False,
    magnitude_logical: bool = False,
) -> None:
    """Outer boundary shell colored by :math:`|u|`, optional outward quiver.

    Samples the map at logical ``surface_rho`` (default ``1 - 10^{-5}``).  On perturbed
    HDF5 maps the pushforward ``|u|`` can be non-finite at exactly ``\\rho=1`` or where
    ``det J`` is tiny; those points fall back to the reference-cell ``|u|`` magnitude
    so the colormap remains meaningful.

    With ``axis_full=True`` or ``half_cut=True``, a near-magnetic-axis curve is drawn so it
    remains visible while the shell still reads as an enclosing volume (camera-near segments
    stay sharp, camera-far segments are tinted through a translucent shell so the depth cue
    is preserved). This is implemented by:

    * setting ``computed_zorder=False`` on the 3D axes so explicit ``zorder`` values are
      respected (matplotlib's default mplot3d ordering is centroid-based and unreliable when
      the axis curve and the toroidal surface share roughly the same centroid),
    * drawing the translucent shell **before** the axis segments on the axes so line artists
      are not fully occluded on some matplotlib builds,
    * splitting the axis curve into many short toroidal segments and assigning each a
      per-segment ``zorder`` based on its depth along the camera view direction. Segments
      farther than the surface centroid are drawn first (and over-painted by the shell),
      segments closer than the centroid are drawn last (so they remain visible),
    * making the toroidal shell translucent (per-face alpha applied to the colormap RGBA
      values) whenever that axis overlay runs (``axis_full or half_cut``), so far-side axis
      segments are visible through the shell while the colour structure still conveys an
      enclosing surface. Without either flag the shell stays fully opaque (plain surface plot).

    Notes
    -----
    The depth/zorder logic uses the static ``ax.elev`` / ``ax.azim`` at figure-save time, so
    it produces a correct still image; if the figure is later rotated interactively the
    occlusion ordering is no longer guaranteed.

    Parameters
    ----------
    seq
        FEM sequence used to build the discrete 1-form ``v``.
    v
        :math:`k=1` DOF vector to evaluate on the outer surface.
    map_raw
        Single-period map; ``extend_map_nfp(map_raw, nfp)`` is used internally.
    nfp
        Number of field periods.
    path
        Output PNG (or compatible) file path.
    ntheta, nzeta
        Surface grid resolution.
    quiver_stride_theta, quiver_stride_zeta
        Strides for the outward-normal quiver.
    quiver_length, quiver_offset
        Quiver arrow length / lift-off above the surface.
    surface_rho
        Logical radial coordinate of the plotted shell (``1.0`` is the LCFS; use slightly
        below ``1`` on perturbed maps if the pushforward is ill-conditioned there).
    half_cut
        If True, keep only half of the toroidal extent for a cutaway view. The magnetic-axis
        overlay is drawn in the same ``zeta`` interval as the truncated shell (implemented via
        ``show_axis = axis_full or half_cut``).
    axis_full
        If True, overlay the near-axis curve (also implied by ``half_cut`` alone). The code
        sweeps logical :math:`\\rho` from ``0`` up to about ``0.2`` and several poloidal
        ``\\theta`` values until ``full_map`` yields a long contiguous run of finite Cartesian
        points along ``\\zeta`` (GVEC is often singular at ``\\rho=0`` on the original
        ``\\theta=0`` ray only). Segments are depth-sorted for display.
    clean
        If True, drop labels/title/axis frame (colorbar kept).
    """
    fld, full_map = _make_full_torus_pushforward(seq, v, map_raw, nfp)
    disc = DiscreteFunction(v, seq.basis_1, seq.e1)
    rho_plot = float(np.clip(surface_rho, 1.0e-6, 1.0 - 1.0e-12))
    surface_grid = get_2d_grids(
        jax.jit(full_map),
        cut_axis=0,
        cut_value=rho_plot,
        nx=1,
        ny=ntheta,
        nz=nzeta,
        invert_z=True,
    )
    logical_pts = surface_grid[0]
    grid_shape = surface_grid[2][0].shape
    values_phys = np.asarray(
        _eval_pts(fld, logical_pts), dtype=np.float64
    ).reshape(*grid_shape, 3)
    mag_phys = np.linalg.norm(values_phys, axis=-1)
    mag_log = np.linalg.norm(
        np.asarray(_eval_pts(disc, logical_pts), dtype=np.float64).reshape(
            *grid_shape, 3
        ),
        axis=-1,
    )
    n_bad = int(np.sum(~np.isfinite(mag_phys)))
    if n_bad > 0:
        print(
            f"WARNING: outer-surface plot: {n_bad}/{mag_phys.size} pushforward |u| "
            f"non-finite at rho={rho_plot:.6f}; using logical |u| at those points.",
            flush=True,
        )
    if magnitude_logical:
        print("hcurl plot colormap: using logical |u| (--plot-magnitude-logical)", flush=True)
        magnitude, mag_label = mag_log, "logical "
    else:
        magnitude, mag_label = _pick_surface_colormap_magnitude(mag_phys, mag_log)
    values = values_phys.copy()
    bad_vec = ~np.all(np.isfinite(values_phys), axis=-1)
    values[bad_vec] = 0.0
    colors, vmin, vmax = _surface_colors(magnitude)

    X = np.asarray(surface_grid[2][0])
    Y = np.asarray(surface_grid[2][1])
    Z = np.asarray(surface_grid[2][2])
    U = np.asarray(values[..., 0])
    V = np.asarray(values[..., 1])
    W = np.asarray(values[..., 2])
    # Logical ``zeta`` samples for the toroidal grid dimension (axis 1 of X), used so the
    # magnetic-axis curve matches the same toroidal window as the shell (critical when
    # ``half_cut`` leaves only half a torus: a full 0..1 axis extends outside the mesh and
    # can confuse mplot3d limits / layering so nothing reads visible).
    zeta_line = np.asarray(surface_grid[3][2], dtype=np.float64)

    if half_cut:
        keep = max(2, X.shape[1] // 2)
        zeta_line = zeta_line[:keep]
        X = X[:, :keep]
        Y = Y[:, :keep]
        Z = Z[:, :keep]
        U = U[:, :keep]
        V = V[:, :keep]
        W = W[:, :keep]
        magnitude = magnitude[:, :keep]
        colors, vmin, vmax = _surface_colors(magnitude)

    major_radius = float(np.mean(np.sqrt(X**2 + Y**2)))

    fig = plt.figure(figsize=(10, 8))
    # ``computed_zorder=False`` lets manual z-order beat mplot3d's centroid heuristic.
    try:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    except TypeError:
        ax = fig.add_subplot(111, projection="3d")
    if getattr(ax, "computed_zorder", None) is not None:
        try:
            ax.computed_zorder = False
        except (AttributeError, TypeError):
            pass
    normals = _surface_normals(X, Y, Z, major_radius)
    view_direction = _surface_view_direction(ax.elev, ax.azim)

    qs = (
        slice(None, None, quiver_stride_theta),
        slice(None, None, quiver_stride_zeta),
    )
    Xq, Yq, Zq = X[qs], Y[qs], Z[qs]
    Uq, Vq, Wq = U[qs], V[qs], W[qs]
    Nq = normals[qs]
    visible_mask = np.sum(Nq * view_direction[None, None, :], axis=-1) > 0.0
    Xq = Xq[visible_mask]
    Yq = Yq[visible_mask]
    Zq = Zq[visible_mask]
    Uq = Uq[visible_mask]
    Vq = Vq[visible_mask]
    Wq = Wq[visible_mask]
    Nq = Nq[visible_mask]
    vector_norm = np.sqrt(Uq**2 + Vq**2 + Wq**2)
    vector_norm = np.where(vector_norm > 0.0, vector_norm, 1.0)

    # Pre-compute the surface centroid depth (along the camera view direction) so axis
    # segments can be assigned zorders relative to it. Larger ``depth`` = closer to camera.
    surface_centroid = np.array(
        [float(np.mean(X)), float(np.mean(Y)), float(np.mean(Z))]
    )
    surface_depth = float(np.dot(surface_centroid, view_direction))
    surface_zorder = 50.0
    show_axis = bool(axis_full or half_cut)

    # When the magnetic-axis overlay is requested, drop the shell's per-face alpha so
    # camera-far axis segments (drawn under the surface via per-segment zorder) remain
    # visible through the tint. The shell colour structure still encloses the volume,
    # which preserves the "surface surrounding the axis" reading. ``facecolors`` carries
    # the full RGBA from ``plt.cm.plasma`` so we modify the alpha channel in place.
    surface_alpha_face = 0.55 if show_axis else 1.0
    surf_colors = np.asarray(colors, dtype=np.float64)
    if surf_colors.shape[-1] == 3:
        surf_colors = np.concatenate(
            [surf_colors, np.ones(surf_colors.shape[:-1] + (1,), dtype=np.float64)],
            axis=-1,
        )
    surf_colors = surf_colors.copy()
    surf_colors[..., 3] = surface_alpha_face

    # Draw the shell before the axis polyline: mplot3d is painter-approximate; adding the
    # surface first avoids the axis being fully hidden on some matplotlib builds when it was
    # registered before ``plot_surface``.
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=surf_colors,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0,
        antialiased=False,
    )
    try:
        surf.set_zorder(surface_zorder)
    except (AttributeError, TypeError):
        pass

    # Magnetic axis: split into many short segments so each can be z-ordered separately
    # against the surface (mplot3d has no true Z-buffer; per-segment zorder approximates it).
    if show_axis:
        n_axis_pts = int(max(128, 64 * int(nfp)))
        z_lo, z_hi = float(zeta_line[0]), float(zeta_line[-1])
        if z_hi <= z_lo:
            z_hi = z_lo + 1e-12
        zeta_axis = np.linspace(z_lo, z_hi, n_axis_pts + 1, endpoint=True)
        # Logical rho=0 is the magnetic axis, but GVEC / Clebsch-style maps are singular
        # there; ``full_map`` often returns NaN. Sweep a denser rho ladder plus several
        # poloidal angles (theta=0 alone is not always valid on the mapped grid).
        map_eval = jax.jit(full_map)
        rho_candidates = (
            0.0,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            5e-3,
            1e-2,
            2e-2,
            3e-2,
            5e-2,
            0.1,
            0.15,
            0.2,
        )
        theta_candidates = (0.0, 1e-6, 0.25, 0.5, 0.75)
        min_axis_finite = max(8, n_axis_pts // 8)
        axis_xyz = None
        for th_ax in theta_candidates:
            for rho_ax in rho_candidates:
                axis_pts = np.stack(
                    [
                        np.full(zeta_axis.shape, float(rho_ax), dtype=np.float64),
                        np.full(zeta_axis.shape, float(th_ax), dtype=np.float64),
                        zeta_axis,
                    ],
                    axis=1,
                )
                cand = np.asarray(
                    _eval_pts(map_eval, jnp.asarray(axis_pts))
                ).reshape(-1, 3)
                ok = np.isfinite(cand).all(axis=1)
                span = _longest_contiguous_true_span(ok)
                if span is None:
                    continue
                lo, hi = span
                if hi - lo < min_axis_finite:
                    continue
                axis_xyz = cand[lo:hi]
                break
            if axis_xyz is not None:
                break
        if axis_xyz is None:
            print(
                "WARNING: outer-surface plot: magnetic axis overlay skipped — "
                "full_map returned too few finite points after rho/theta search "
                f"(rho in {rho_candidates[:5]}…{rho_candidates[-3:]}, "
                f"theta in {theta_candidates}).",
                flush=True,
            )
        if axis_xyz is not None:
            seg_starts = axis_xyz[:-1]
            seg_ends = axis_xyz[1:]
            seg_mids = 0.5 * (seg_starts + seg_ends)
            seg_depths = seg_mids @ view_direction
            d_min = float(np.min(seg_depths))
            d_max = float(np.max(seg_depths))
            d_range = max(d_max - d_min, 1e-12)
            seg_zorders = surface_zorder + 50.0 * (seg_depths - surface_depth) / d_range
            for i in range(seg_starts.shape[0]):
                (seg_line,) = ax.plot(
                    [seg_starts[i, 0], seg_ends[i, 0]],
                    [seg_starts[i, 1], seg_ends[i, 1]],
                    [seg_starts[i, 2], seg_ends[i, 2]],
                    color="black",
                    linewidth=2.0,
                    solid_capstyle="round",
                )
                seg_line.set_zorder(float(seg_zorders[i]))

    if Xq.size > 0:
        ax.quiver(
            Xq + quiver_offset * Nq[:, 0],
            Yq + quiver_offset * Nq[:, 1],
            Zq + quiver_offset * Nq[:, 2],
            Uq / vector_norm,
            Vq / vector_norm,
            Wq / vector_norm,
            length=quiver_length,
            normalize=False,
            color="black",
            linewidth=0.8,
            alpha=0.9,
        )
        try:
            ax.collections[-1].set_zorder(surface_zorder + 60.0)
        except (AttributeError, TypeError, IndexError):
            pass
    set_axes_equal(ax)
    if clean:
        ax.set_axis_off()
    else:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        rho_label = r"1" if abs(rho_plot - 1.0) < 1e-8 else f"{rho_plot:.4f}"
        ax.set_title(
            rf"Free $k=1$ H(curl) nullspace: {mag_label}$|\mathbf{{u}}|$ on $\rho_{{\mathrm{{log}}}}={rho_label}$ (outer surface)",
            fontsize=11,
        )
    sm = plt.cm.ScalarMappable(cmap="plasma")
    sm.set_clim(vmin, vmax)
    cbar_label = r"$|u|$ (logical)" if mag_label else r"$|u|$"
    fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.08, label=cbar_label)
    if not clean:
        plt.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
    else:
        fig.subplots_adjust(left=0.0, right=0.92, bottom=0.0, top=1.0)
        fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(path)

# Backward-compatible private aliases (tests / legacy imports from the driver module).
_plot_logical_rtheta = plot_logical_rtheta
_plot_poloidal_RZ = plot_poloidal_RZ
_plot_outer_surface = plot_outer_surface


def _make_full_torus_pushforward_2form(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    nfp: int,
) -> tuple[Pushforward, Callable[[jnp.ndarray], jnp.ndarray]]:
    """One-period discrete 2-form (DBC) + ``extend_map_nfp`` for full-torus evaluation."""
    disc = DiscreteFunction(v, seq.basis_2, seq.e2_dbc)
    nfp_i = int(nfp)

    def localized(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64).reshape(3)
        xi = x[2] * float(nfp_i)
        zloc = xi - jnp.floor(xi)
        return disc(x.at[2].set(zloc))

    full_map = extend_map_nfp(map_raw, nfp_i)
    return Pushforward(localized, full_map, 2), full_map


def plot_logical_rtheta_k2(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    *,
    zetas: tuple[float, ...],
    path: Path,
    nx: int,
    ny: int,
    quiver_stride_r: int,
    quiver_stride_t: int,
    clean: bool = False,
) -> None:
    """Logical $(\\rho,\\theta)$ slice for harmonic k=2 (DBC) null vector."""
    disc = DiscreteFunction(v, seq.basis_2, seq.e2_dbc)
    fig, axes_arr, cax = _subplots_with_right_colorbar(len(zetas))
    blocks: list[tuple[Any, ...]] = []
    for zc in zetas:
        logical_points, r_grid, theta_grid = _poloidal_slice_grid(zc, nx, ny)
        values = np.asarray(_eval_pts(disc, logical_points)).reshape(nx, ny, 3)
        magnitude = np.linalg.norm(values, axis=-1)
        vr = values[..., 0]
        vtheta = values[..., 1]
        blocks.append((r_grid, theta_grid, magnitude, vr, vtheta))
    vmin, vmax = _norm_mag_panels([b[2] for b in blocks])
    if vmax <= vmin:
        vmax = vmin + 1e-30
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = None
    for ax, zc, (r_grid, theta_grid, magnitude, vr, vtheta) in zip(
        axes_arr.flat, zetas, blocks
    ):
        mappable = ax.pcolormesh(
            np.asarray(r_grid),
            np.asarray(theta_grid),
            magnitude,
            shading="auto",
            cmap="plasma",
            norm=norm,
            zorder=1,
            rasterized=True,
        )
        ax.quiver(
            np.asarray(r_grid)[::quiver_stride_r, ::quiver_stride_t],
            np.asarray(theta_grid)[::quiver_stride_r, ::quiver_stride_t],
            vr[::quiver_stride_r, ::quiver_stride_t],
            vtheta[::quiver_stride_r, ::quiver_stride_t],
            color="white",
            pivot="mid",
            scale=25,
            linewidth=0.25,
            width=0.002,
            alpha=0.45,
            zorder=2,
        )
        if clean:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_frame_on(False)
        else:
            ax.set_xlabel(r"$\rho_{\mathrm{log}}$")
            ax.set_title(rf"$\zeta={zc:.2f}$")
    assert mappable is not None
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(r"$|u|$")
    _save_row_figure(
        fig,
        path,
        axes_arr,
        clean=clean,
        suptitle=None
        if clean
        else r"Harmonic $k=2$ (DBC): $|\mathbf{u}|$ in logical $(\rho,\theta)$",
        shared_ylabel=None if clean else r"$\theta_{\mathrm{log}}$",
    )


def plot_poloidal_RZ_k2(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    zetas: tuple[float, ...],
    path: Path,
    nx: int,
    ny: int,
    quiver_stride_r: int,
    quiver_stride_t: int,
    clean: bool = False,
) -> None:
    """Physical poloidal slice for harmonic k=2 pushforward."""
    fld, full_map = _make_full_torus_pushforward_2form(seq, v, map_raw, nfp)
    map_ev = jax.jit(full_map)
    blocks: list[tuple[Any, ...]] = []
    for zc in zetas:
        logical_points, _, _ = _poloidal_slice_grid(zc, nx, ny)
        mapped = np.asarray(_eval_pts(map_ev, logical_points)).reshape(nx, ny, 3)
        uv = np.asarray(_eval_pts(fld, logical_points)).reshape(nx, ny, 3)
        X, Y, Z = mapped[..., 0], mapped[..., 1], mapped[..., 2]
        R = np.sqrt(X**2 + Y**2)
        magnitude = np.linalg.norm(uv, axis=-1)
        radial_denom = np.where(R > 0.0, R, 1.0)
        vR = (X * uv[..., 0] + Y * uv[..., 1]) / radial_denom
        vZ = uv[..., 2]
        blocks.append((R, Z, magnitude, vR, vZ))
    vmin, vmax = _norm_mag_panels([b[2] for b in blocks])
    if vmax <= vmin:
        vmax = vmin + 1e-30
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes_arr, cax = _subplots_with_right_colorbar(len(zetas))
    mappable = None
    for ax, zc, (R, Z, magnitude, vR, vZ) in zip(axes_arr.flat, zetas, blocks):
        mappable = ax.pcolormesh(
            R,
            Z,
            magnitude,
            shading="auto",
            cmap="plasma",
            norm=norm,
            zorder=1,
            rasterized=True,
        )
        ax.quiver(
            R[::quiver_stride_r, ::quiver_stride_t],
            Z[::quiver_stride_r, ::quiver_stride_t],
            vR[::quiver_stride_r, ::quiver_stride_t],
            vZ[::quiver_stride_r, ::quiver_stride_t],
            color="white",
            pivot="mid",
            scale=25,
            linewidth=0.25,
            width=0.002,
            alpha=0.45,
            zorder=2,
        )
        ax.set_aspect("equal")
        if clean:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_frame_on(False)
        else:
            ax.set_xlabel(r"$R$")
            ax.set_title(rf"$\zeta={zc:.2f}$")
    assert mappable is not None
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(r"$|u|$")
    _save_row_figure(
        fig,
        path,
        axes_arr,
        clean=clean,
        suptitle=None
        if clean
        else r"Harmonic $k=2$ (DBC): $|\mathbf{u}|$ on poloidal $(R,Z)$ (full torus map)",
        shared_ylabel=None if clean else r"$Z$",
    )


def plot_outer_surface_k2(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    path: Path,
    ntheta: int,
    nzeta: int,
    quiver_stride_theta: int,
    quiver_stride_zeta: int,
    quiver_length: float,
    quiver_offset: float,
    surface_rho: float = 1.0 - 1e-5,
    half_cut: bool = False,
    axis_full: bool = False,
    clean: bool = False,
) -> None:
    """Outer boundary shell for harmonic k=2 pushforward."""
    fld, full_map = _make_full_torus_pushforward_2form(seq, v, map_raw, nfp)
    rho_plot = float(np.clip(surface_rho, 1.0e-6, 1.0 - 1.0e-12))
    surface_grid = get_2d_grids(
        jax.jit(full_map),
        cut_axis=0,
        cut_value=rho_plot,
        nx=1,
        ny=ntheta,
        nz=nzeta,
        invert_z=True,
    )
    values = np.asarray(
        _eval_pts(fld, surface_grid[0]), dtype=np.float64
    ).reshape(*surface_grid[2][0].shape, 3)
    magnitude = np.linalg.norm(values, axis=-1)
    colors, vmin, vmax = _surface_colors(magnitude)

    X = np.asarray(surface_grid[2][0])
    Y = np.asarray(surface_grid[2][1])
    Z = np.asarray(surface_grid[2][2])
    U = values[..., 0]
    V = values[..., 1]
    W = values[..., 2]
    zeta_line = np.asarray(surface_grid[3][2], dtype=np.float64)

    if half_cut:
        keep = max(2, X.shape[1] // 2)
        zeta_line = zeta_line[:keep]
        X = X[:, :keep]
        Y = Y[:, :keep]
        Z = Z[:, :keep]
        U = U[:, :keep]
        V = V[:, :keep]
        W = W[:, :keep]
        magnitude = magnitude[:, :keep]
        colors, vmin, vmax = _surface_colors(magnitude)

    major_radius = float(np.mean(np.sqrt(X**2 + Y**2)))

    fig = plt.figure(figsize=(10, 8))
    try:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    except TypeError:
        ax = fig.add_subplot(111, projection="3d")
    if getattr(ax, "computed_zorder", None) is not None:
        try:
            ax.computed_zorder = False
        except (AttributeError, TypeError):
            pass
    normals = _surface_normals(X, Y, Z, major_radius)
    view_direction = _surface_view_direction(ax.elev, ax.azim)

    qs = (
        slice(None, None, quiver_stride_theta),
        slice(None, None, quiver_stride_zeta),
    )
    Xq, Yq, Zq = X[qs], Y[qs], Z[qs]
    Uq, Vq, Wq = U[qs], V[qs], W[qs]
    Nq = normals[qs]
    visible_mask = np.sum(Nq * view_direction[None, None, :], axis=-1) > 0.0
    Xq = Xq[visible_mask]
    Yq = Yq[visible_mask]
    Zq = Zq[visible_mask]
    Uq = Uq[visible_mask]
    Vq = Vq[visible_mask]
    Wq = Wq[visible_mask]
    Nq = Nq[visible_mask]
    vector_norm = np.sqrt(Uq**2 + Vq**2 + Wq**2)
    vector_norm = np.where(vector_norm > 0.0, vector_norm, 1.0)

    surface_centroid = np.array(
        [float(np.mean(X)), float(np.mean(Y)), float(np.mean(Z))]
    )
    surface_depth = float(np.dot(surface_centroid, view_direction))
    surface_zorder = 50.0
    show_axis = bool(axis_full or half_cut)

    surface_alpha_face = 0.55 if show_axis else 1.0
    surf_colors = np.asarray(colors, dtype=np.float64)
    if surf_colors.shape[-1] == 3:
        surf_colors = np.concatenate(
            [surf_colors, np.ones(surf_colors.shape[:-1] + (1,), dtype=np.float64)],
            axis=-1,
        )
    surf_colors = surf_colors.copy()
    surf_colors[..., 3] = surface_alpha_face

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=surf_colors,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0,
        antialiased=False,
    )
    try:
        surf.set_zorder(surface_zorder)
    except (AttributeError, TypeError):
        pass

    if show_axis:
        n_axis_pts = int(max(128, 64 * int(nfp)))
        z_lo, z_hi = float(zeta_line[0]), float(zeta_line[-1])
        if z_hi <= z_lo:
            z_hi = z_lo + 1e-12
        zeta_axis = np.linspace(z_lo, z_hi, n_axis_pts + 1, endpoint=True)
        map_eval = jax.jit(full_map)
        rho_candidates = (
            0.0,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            5e-3,
            1e-2,
            2e-2,
            3e-2,
            5e-2,
            0.1,
            0.15,
            0.2,
        )
        theta_candidates = (0.0, 1e-6, 0.25, 0.5, 0.75)
        min_axis_finite = max(8, n_axis_pts // 8)
        axis_xyz = None
        for th_ax in theta_candidates:
            for rho_ax in rho_candidates:
                axis_pts = np.stack(
                    [
                        np.full(zeta_axis.shape, float(rho_ax), dtype=np.float64),
                        np.full(zeta_axis.shape, float(th_ax), dtype=np.float64),
                        zeta_axis,
                    ],
                    axis=1,
                )
                cand = np.asarray(
                    _eval_pts(map_eval, jnp.asarray(axis_pts))
                ).reshape(-1, 3)
                ok = np.isfinite(cand).all(axis=1)
                span = _longest_contiguous_true_span(ok)
                if span is None:
                    continue
                lo, hi = span
                if hi - lo < min_axis_finite:
                    continue
                axis_xyz = cand[lo:hi]
                break
            if axis_xyz is not None:
                break
        if axis_xyz is not None:
            seg_starts = axis_xyz[:-1]
            seg_ends = axis_xyz[1:]
            seg_mids = 0.5 * (seg_starts + seg_ends)
            seg_depths = seg_mids @ view_direction
            d_min = float(np.min(seg_depths))
            d_max = float(np.max(seg_depths))
            d_range = max(d_max - d_min, 1e-12)
            seg_zorders = surface_zorder + 50.0 * (seg_depths - surface_depth) / d_range
            for i in range(seg_starts.shape[0]):
                (seg_line,) = ax.plot(
                    [seg_starts[i, 0], seg_ends[i, 0]],
                    [seg_starts[i, 1], seg_ends[i, 1]],
                    [seg_starts[i, 2], seg_ends[i, 2]],
                    color="black",
                    linewidth=2.0,
                    solid_capstyle="round",
                )
                seg_line.set_zorder(float(seg_zorders[i]))

    if Xq.size > 0:
        ax.quiver(
            Xq + quiver_offset * Nq[:, 0],
            Yq + quiver_offset * Nq[:, 1],
            Zq + quiver_offset * Nq[:, 2],
            Uq / vector_norm,
            Vq / vector_norm,
            Wq / vector_norm,
            length=quiver_length,
            normalize=False,
            color="black",
            linewidth=0.8,
            alpha=0.9,
        )
        try:
            ax.collections[-1].set_zorder(surface_zorder + 60.0)
        except (AttributeError, TypeError, IndexError):
            pass
    set_axes_equal(ax)
    if clean:
        ax.set_axis_off()
    else:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        rho_label = r"1" if abs(rho_plot - 1.0) < 1e-8 else f"{rho_plot:.4f}"
        ax.set_title(
            rf"Harmonic $k=2$ (DBC): $|\mathbf{{u}}|$ on $\rho_{{\mathrm{{log}}}}={rho_label}$ (outer surface)",
            fontsize=11,
        )
    sm = plt.cm.ScalarMappable(cmap="plasma")
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.08, label=r"$|u|$")
    if not clean:
        plt.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
    else:
        fig.subplots_adjust(left=0.0, right=0.92, bottom=0.0, top=1.0)
        fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(path)


def run_k2_nullspace_plots(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    out_dir: Path,
    zetas: tuple[float, ...],
    nx: int,
    ny: int,
    quiver_stride_r: int,
    quiver_stride_t: int,
    surface_ntheta: int,
    surface_nzeta: int,
    surface_quiver_stride_theta: int,
    surface_quiver_stride_zeta: int,
    surface_quiver_length: float,
    surface_quiver_offset: float,
    surface_rho: float,
    half_cut: bool,
    axis_full: bool,
    clean: bool,
    magnitude_logical: bool = False,
) -> None:
    """Write the three standard k=2 nullspace PNGs under ``out_dir``."""
    del magnitude_logical  # k=2 plots use pushforward magnitude only
    out_dir = Path(out_dir)
    plot_logical_rtheta_k2(
        seq,
        v,
        zetas=zetas,
        path=out_dir / "hodge_k2_null_logical_rtheta.png",
        nx=nx,
        ny=ny,
        quiver_stride_r=quiver_stride_r,
        quiver_stride_t=quiver_stride_t,
        clean=clean,
    )
    plot_poloidal_RZ_k2(
        seq,
        v,
        map_raw,
        nfp=nfp,
        zetas=zetas,
        path=out_dir / "hodge_k2_null_poloidal_RZ.png",
        nx=nx,
        ny=ny,
        quiver_stride_r=quiver_stride_r,
        quiver_stride_t=quiver_stride_t,
        clean=clean,
    )
    plot_outer_surface_k2(
        seq,
        v,
        map_raw,
        nfp=nfp,
        path=out_dir / "hodge_k2_null_outer_surface.png",
        ntheta=surface_ntheta,
        nzeta=surface_nzeta,
        quiver_stride_theta=surface_quiver_stride_theta,
        quiver_stride_zeta=surface_quiver_stride_zeta,
        quiver_length=surface_quiver_length,
        quiver_offset=surface_quiver_offset,
        surface_rho=surface_rho,
        half_cut=half_cut,
        axis_full=axis_full,
        clean=clean,
    )


def run_k1_nullspace_plots(
    seq: DeRhamSequence,
    v: jnp.ndarray,
    map_raw: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    out_dir: Path,
    zetas: tuple[float, ...],
    nx: int,
    ny: int,
    quiver_stride_r: int,
    quiver_stride_t: int,
    surface_ntheta: int,
    surface_nzeta: int,
    surface_quiver_stride_theta: int,
    surface_quiver_stride_zeta: int,
    surface_quiver_length: float,
    surface_quiver_offset: float,
    surface_rho: float,
    half_cut: bool,
    axis_full: bool,
    clean: bool,
    magnitude_logical: bool = False,
) -> None:
    """Write the three standard k=1 nullspace PNGs under ``out_dir``."""
    out_dir = Path(out_dir)
    plot_logical_rtheta(
        seq,
        v,
        zetas=zetas,
        path=out_dir / "hcurl_null_logical_rtheta.png",
        nx=nx,
        ny=ny,
        quiver_stride_r=quiver_stride_r,
        quiver_stride_t=quiver_stride_t,
        clean=clean,
        magnitude_logical=magnitude_logical,
    )
    plot_poloidal_RZ(
        seq,
        v,
        map_raw,
        nfp=nfp,
        zetas=zetas,
        path=out_dir / "hcurl_null_poloidal_RZ.png",
        nx=nx,
        ny=ny,
        quiver_stride_r=quiver_stride_r,
        quiver_stride_t=quiver_stride_t,
        clean=clean,
        magnitude_logical=magnitude_logical,
    )
    plot_outer_surface(
        seq,
        v,
        map_raw,
        nfp=nfp,
        path=out_dir / "hcurl_null_outer_surface.png",
        ntheta=surface_ntheta,
        nzeta=surface_nzeta,
        quiver_stride_theta=surface_quiver_stride_theta,
        quiver_stride_zeta=surface_quiver_stride_zeta,
        quiver_length=surface_quiver_length,
        quiver_offset=surface_quiver_offset,
        surface_rho=surface_rho,
        half_cut=half_cut,
        axis_full=axis_full,
        clean=clean,
        magnitude_logical=magnitude_logical,
    )
