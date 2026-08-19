"""Rebuild ``DeRhamSequence`` + map from harmonic nullspace meta JSON (replot path).

Reads the meta JSON written beside harmonic solve outputs and rebuilds the FEM sequence
plus ``map_raw`` for pushforward evaluation. Geometry uses MRX ``stellarator_map``
interpolation from volume HDF5 ``(eval_points, R, Z)`` — not live GVEC finite difference evaluation
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence, SequenceGeometry
from mrx.io import project_sampled_field
from mrx.io_nfs_map import (
    _project_sampled_scalar_0form_dense_rho0_theta_indep,
    _tensor_axes_and_grids,
    interpolate_map_from_points,
)

jax.config.update("jax_enable_x64", True)


def default_meta_path(dof_npy: Path, *, k: int) -> Path:
    """Return sibling meta JSON for a harmonic DOF file."""
    dof_npy = dof_npy.expanduser().resolve()
    name = "hodge_k2_nullspace_meta.json" if int(k) == 2 else "hcurl_nullspace_meta.json"
    return dof_npy.with_name(name)


def load_meta(meta_path: Path | None, dof_npy: Path, *, k: int) -> dict[str, Any]:
    """Load meta JSON; default path from ``dof_npy`` and ``k``."""
    meta_path = (
        meta_path.expanduser().resolve()
        if meta_path is not None
        else default_meta_path(dof_npy, k=k)
    )
    if not meta_path.is_file():
        raise FileNotFoundError(f"Harmonic meta not found: {meta_path}")
    return json.loads(meta_path.read_text())


def infer_form_degree(dof_npy: Path, *, k: int | None = None) -> int:
    """Infer k=1 vs k=2 from explicit ``k`` or DOF filename."""
    if k is not None:
        return int(k)
    stem = dof_npy.name.lower()
    if "hodge_k2" in stem or "k2" in stem:
        return 2
    return 1


def load_map_geometry_h5(path: Path) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Load ``(ρ,θ,ζ)``, cylindrical ``R``, ``Z``, and ``nfp`` from an MRX volume HDF5.

    Expects root datasets ``eval_points`` (N,3), ``R``, ``Z``, and attr ``nfp``.
    """
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "map_mode=h5 requires h5py (e.g. pip install h5py)."
        ) from exc
    path = path.expanduser().resolve()
    with h5py.File(path, "r") as h5:
        missing = [key for key in ("eval_points", "R", "Z") if key not in h5]
        if missing:
            raise SystemExit(
                f"{path}: missing dataset(s) {missing!r}; expected /eval_points, /R, /Z."
            )
        if "nfp" not in h5.attrs:
            raise SystemExit(f"{path}: missing HDF5 root attribute 'nfp'.")
        nfp = int(h5.attrs["nfp"])
        ep = np.asarray(h5["eval_points"][:], dtype=np.float64)
        r1d = np.asarray(h5["R"][:], dtype=np.float64).ravel()
        z1d = np.asarray(h5["Z"][:], dtype=np.float64).ravel()
    if ep.ndim != 2 or ep.shape[1] != 3:
        raise SystemExit(f"{path}: /eval_points must be (N, 3); got shape {ep.shape}.")
    n = ep.shape[0]
    if r1d.shape[0] != n or z1d.shape[0] != n:
        raise SystemExit(
            f"{path}: /R and /Z length {r1d.shape[0]}, {z1d.shape[0]} must match N={n}."
        )
    if not np.isfinite(ep).all() or not np.isfinite(r1d).all() or not np.isfinite(z1d).all():
        raise SystemExit(f"{path}: non-finite values in eval_points, R, or Z.")
    return nfp, jnp.asarray(ep), jnp.asarray(r1d), jnp.asarray(z1d)


def _xyz_grids_stellarator_convention(
    R_grid: jnp.ndarray,
    Z_grid: jnp.ndarray,
    zeta_axis: jnp.ndarray,
    *,
    nfp: int,
    flip_zeta: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cartesian ``X,Y,Z`` tensor grids from cylindrical ``R,Z`` and logical ζ samples."""
    pi_nfp = 2.0 * jnp.pi / float(nfp)
    w_b = jnp.asarray(zeta_axis, dtype=jnp.float64).reshape(1, 1, -1)
    if flip_zeta:
        w_b = 1.0 - w_b
    Rg = jnp.asarray(R_grid, dtype=jnp.float64)
    Zg = jnp.asarray(Z_grid, dtype=jnp.float64)
    Xg = Rg * jnp.cos(pi_nfp * w_b)
    Yg = -Rg * jnp.sin(pi_nfp * w_b)
    return Xg, Yg, Zg


def _warn_if_large_spline_misfit(
    max_abs_dxyz: Sequence[float],
    *,
    warn_tol: float,
    rho0_constrained: bool,
) -> None:
    """Warn when SplineMap pointwise geometry misfit is large."""
    m = max(float(max_abs_dxyz[0]), float(max_abs_dxyz[1]), float(max_abs_dxyz[2]))
    if not math.isfinite(m) or m <= float(warn_tol):
        return
    print(
        f"WARNING: SplineMap max |ΔX|,|ΔY|,|ΔZ| at data grid = {m:.3e} "
        f"(exceeds map-residual-warn={warn_tol:g}).",
        flush=True,
    )
    if rho0_constrained:
        print(
            "  With map-rho0-theta-independent, the inner-ρ poloidal constant "
            "constraint may be too strong for X,Y; try omitting that flag.",
            flush=True,
        )


def _build_analytic_map_raw(
    map_kind: str,
    *,
    nfp: int,
    ellipse_eps: float,
    ellipse_kappa: float,
    ellipse_r0: float,
    cerfon_alpha: float = 0.0,
    shafranov_e0: float = 0.3,
    shafranov_delta0: float = 0.2,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], str]:
    """Lazy analytic map builder compatible with greville-prod ``mrx.mappings``."""
    from mrx import mappings as mp

    kind = str(map_kind).strip().lower()
    eps, kap, r0 = float(ellipse_eps), float(ellipse_kappa), float(ellipse_r0)
    if kind == "toroid":
        fn = mp.toroid_map(eps, kap, r0)
        return fn, f"toroid_map(eps={eps:g}, kappa={kap:g}, R0={r0:g})"
    if kind == "cerfon":
        if hasattr(mp, "cerfon_map"):
            fn = mp.cerfon_map(eps, kap, float(cerfon_alpha), r0)
            label = (
                f"cerfon_map(eps={eps:g}, kappa={kap:g}, "
                f"alpha={cerfon_alpha:g}, R0={r0:g})"
            )
        else:
            fn = mp.one_size_fits_all_map(eps, kap, float(cerfon_alpha), r0)
            label = (
                f"one_size_fits_all_map(eps={eps:g}, kappa={kap:g}, "
                f"alpha={cerfon_alpha:g}, R0={r0:g})"
            )
        return fn, label
    if kind == "rotating_ellipse":
        fn = mp.rotating_ellipse_map(eps, kap, r0, int(nfp))
        return fn, f"rotating_ellipse_map(nfp={int(nfp)}, eps={eps:g}, kappa={kap:g}, R0={r0:g})"
    if kind in ("shafranov_bourne", "shafranov", "bourne_shafranov"):
        if not hasattr(mp, "shafranov_shifted_tokamak_map"):
            raise ValueError(
                f"map_kind={map_kind!r} requires shafranov_shifted_tokamak_map "
                "(not on greville-prod); use map_mode h5 with map_geometry_h5."
            )
        fn = mp.shafranov_shifted_tokamak_map(
            epsilon=eps,
            E0=float(shafranov_e0),
            delta0=float(shafranov_delta0),
            R0=r0,
        )
        return fn, (
            f"shafranov_shifted_tokamak_map(eps={eps:g}, E0={shafranov_e0:g}, "
            f"delta0={shafranov_delta0:g}, R0={r0:g})"
        )
    raise ValueError(
        f"Unknown map_kind {map_kind!r}; expected "
        "('toroid', 'cerfon', 'rotating_ellipse', 'shafranov_bourne')."
    )


def _make_sequence(
    *,
    ns: tuple[int, int, int],
    p: int,
    betti: tuple[int, int, int, int],
    tol: float,
    maxiter: int,
) -> DeRhamSequence:
    """Shared FEM sequence constructor for harmonic replot geometry."""
    return DeRhamSequence(
        ns,
        (p, p, p),
        2 * p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=tol,
        maxiter=maxiter,
        betti_numbers=betti,
    )


def _require_map_geometry_h5(meta: dict[str, Any], *, map_mode: str) -> Path:
    """
    Return ``map_geometry_h5`` from harmonic meta.

    Legacy ``map_mode=gvec`` (live GVEC FD map) is removed; both ``gvec`` and ``h5``
    modes require a pre-exported MRX volume HDF5 with ``eval_points``, ``R``, ``Z``.
    """
    raw = meta.get("map_geometry_h5")
    if raw is None or str(raw).strip() == "":
        raise SystemExit(
            f"map_mode={map_mode!r} requires map_geometry_h5 in meta "
            "(MRX volume H5 with /eval_points, /R, /Z). "
            "Export with export_gvec_state_to_mrx.py and add the path to meta, "
            "or set map_mode=h5. Live GVEC FD maps (gvec_jax_map) are no longer used."
        )
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"map_geometry_h5 not found: {path}")
    return path


def build_sequence_h5(
    *,
    ns: tuple[int, int, int],
    p: int,
    betti: tuple[int, int, int, int],
    tol: float,
    maxiter: int,
    map_geometry_h5: Path,
    nfp: int | None,
    flip_zeta: bool,
    strict_jacobian: bool,
    pushforward_only: bool = True,
    map_rho0_theta_independent: bool = False,
    map_residual_warn: float = 1e-2,
    spline_map_geometry: bool = False,
) -> tuple[DeRhamSequence, Any, None, Callable[[jnp.ndarray], jnp.ndarray]]:
    """Build sequence from MRX volume HDF5 via ``stellarator_map`` R,Z interpolation."""
    h5_path = map_geometry_h5.expanduser().resolve()
    nfp_h5, pts, rvals, zvals = load_map_geometry_h5(h5_path)
    nfp_use = int(nfp if nfp is not None else nfp_h5)
    if int(nfp_h5) != nfp_use:
        print(
            f"WARNING: meta nfp={nfp_use} differs from HDF5 attr nfp={int(nfp_h5)} "
            f"({h5_path.name})",
            file=sys.stderr,
        )

    seq = _make_sequence(ns=ns, p=p, betti=betti, tol=tol, maxiter=maxiter)
    seq.evaluate_1d()

    axes_np, R_np, Z_np = _tensor_axes_and_grids(
        np.asarray(pts, dtype=np.float64),
        np.asarray(rvals, dtype=np.float64),
        np.asarray(zvals, dtype=np.float64),
    )

    if spline_map_geometry:
        seq.assemble_reference_mass_matrix()
        axes = tuple(jnp.asarray(a) for a in axes_np)
        R_grid = jnp.asarray(R_np)
        Z_grid = jnp.asarray(Z_np)
        X_grid, Y_grid, Z_grid = _xyz_grids_stellarator_convention(
            R_grid,
            Z_grid,
            axes[2],
            nfp=nfp_use,
            flip_zeta=bool(flip_zeta),
        )
        rho0 = bool(map_rho0_theta_independent)
        if rho0:
            seq.assemble_mass_matrix(0)

        def _project_one(grid: jnp.ndarray) -> jnp.ndarray:
            if rho0:
                return _project_sampled_scalar_0form_dense_rho0_theta_indep(
                    axes, grid, seq, dirichlet=False
                )
            return project_sampled_field(
                axes, grid, seq, 0, dirichlet=False, reference_domain=True
            )

        coeffs = jnp.stack(
            [_project_one(X_grid), _project_one(Y_grid), _project_one(Z_grid)],
            axis=0,
        )
        seq.set_geometry(seq.geometry_from_spline_map(coeffs))
        pts_j = jnp.asarray(pts)
        xyz_data = jnp.stack(
            [X_grid.reshape(-1), Y_grid.reshape(-1), Z_grid.reshape(-1)],
            axis=-1,
        )
        pred = jax.vmap(seq.map)(pts_j)
        diff = pred - xyz_data
        resid = jnp.nanmax(jnp.abs(diff), axis=0)
        dxyz = [float(resid[0]), float(resid[1]), float(resid[2])]
        print(
            f"SplineMap from {h5_path.name}: max |ΔX|={dxyz[0]:.3e}, "
            f"|ΔY|={dxyz[1]:.3e}, |ΔZ|={dxyz[2]:.3e}",
            flush=True,
        )
        _warn_if_large_spline_misfit(
            dxyz,
            warn_tol=float(map_residual_warn),
            rho0_constrained=rho0,
        )
    else:
        u, v, w = axes_np
        map_ns = (len(u), len(v), len(w))
        map_p = min(2, min(map_ns) - 1)
        map_ps = (map_p, map_p, map_p)
        map_func, _, _, map_resid = interpolate_map_from_points(
            pts,
            rvals,
            zvals,
            nfp_use,
            ns=map_ns,
            ps=map_ps,
            quad_order=max(4, 2 * map_p),
            flip_zeta=bool(flip_zeta),
            rho0_theta_independent=bool(map_rho0_theta_independent),
        )
        map_func = jax.jit(map_func)
        seq.set_geometry(SequenceGeometry.from_map(map_func, seq.quad.x))
        dr, dz = float(map_resid[0]), float(map_resid[1])
        print(
            f"stellarator_map from {h5_path.name}: dR={dr:.3e}, dZ={dz:.3e} "
            f"(map ns={map_ns} ps={map_ps}, flip_zeta={bool(flip_zeta)})",
            flush=True,
        )
        if dr > float(map_residual_warn) or dz > float(map_residual_warn):
            print(
                f"WARNING: stellarator_map R/Z residuals exceed {map_residual_warn:.1e}; "
                "try spline_map_geometry or map_rho0_theta_independent in meta.",
                file=sys.stderr,
            )

    map_raw = seq.map
    map_jit = seq.map
    if pushforward_only:
        print(
            f"pushforward_only: geometry set; skipping detJ audit on "
            f"{seq.quad.x.shape[0]} quadrature points ({h5_path})",
            flush=True,
        )
    else:
        jac = jnp.asarray(seq.jacobian_j)
        jmin, jmax = float(jnp.min(jac)), float(jnp.max(jac))
        n_bad = int(jnp.sum(jac <= 0.0))
        print(f"detJ [{jmin:.3e},{jmax:.3e}]  nonpos {n_bad}/{jac.size}  ({h5_path})")
        if strict_jacobian and n_bad > 0:
            raise ValueError("detJ≤0 — try flip_zeta or a finer map_geometry_h5 grid")
        if n_bad > 0:
            print("WARNING: detJ≤0 somewhere", file=sys.stderr)

    return seq, map_jit, None, map_raw


def build_sequence_analytic(
    *,
    ns: tuple[int, int, int],
    p: int,
    betti: tuple[int, int, int, int],
    tol: float,
    maxiter: int,
    map_kind: str,
    nfp: int,
    ellipse_eps: float,
    ellipse_kappa: float,
    ellipse_r0: float,
    cerfon_alpha: float,
    shafranov_e0: float,
    shafranov_delta0: float,
) -> tuple[DeRhamSequence, Any, None, Callable[[jnp.ndarray], jnp.ndarray]]:
    """Build sequence using an analytic toroidal map (no GVEC state)."""
    seq = _make_sequence(ns=ns, p=p, betti=betti, tol=tol, maxiter=maxiter)
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()

    map_raw, label = _build_analytic_map_raw(
        map_kind,
        nfp=int(nfp),
        ellipse_eps=float(ellipse_eps),
        ellipse_kappa=float(ellipse_kappa),
        ellipse_r0=float(ellipse_r0),
        cerfon_alpha=float(cerfon_alpha),
        shafranov_e0=float(shafranov_e0),
        shafranov_delta0=float(shafranov_delta0),
    )
    map_jit = jax.jit(map_raw)
    seq.set_map(map_jit)
    jac = jnp.asarray(seq.jacobian_j)
    jmin, jmax = float(jnp.min(jac)), float(jnp.max(jac))
    n_bad = int(jnp.sum(jac <= 0.0))
    print(f"detJ [{jmin:.3e},{jmax:.3e}]  nonpos {n_bad}/{jac.size}  (analytic: {label})")
    if n_bad > 0:
        print("WARNING: detJ≤0 somewhere in analytic map", file=sys.stderr)
    return seq, map_jit, None, map_raw


def rebuild_sequence_from_meta(
    meta: dict[str, Any],
    *,
    tol: float = 1e-9,
    maxiter: int = 20,
    strict_jacobian: bool = False,
) -> tuple[DeRhamSequence, Any, Callable[[jnp.ndarray], jnp.ndarray], int]:
    """
    Rebuild FEM sequence and raw map from saved harmonic meta.

    Returns
    -------
    seq, map_jit, map_raw, nfp
    """
    ns = tuple(int(x) for x in meta["ns"])
    p = int(meta["p"])
    betti = tuple(int(x) for x in meta.get("betti", (1, 1, 0, 0)))
    nfp = int(meta["nfp"])
    map_mode = str(meta.get("map_mode", "gvec"))
    flip_zeta = bool(meta.get("flip_zeta", False))
    pushforward_only = True
    h5_kwargs = dict(
        ns=ns,
        p=p,
        betti=betti,
        tol=tol,
        maxiter=maxiter,
        nfp=nfp,
        flip_zeta=flip_zeta,
        strict_jacobian=strict_jacobian,
        pushforward_only=pushforward_only,
        map_rho0_theta_independent=bool(meta.get("map_rho0_theta_independent", False)),
        map_residual_warn=float(meta.get("map_residual_warn", 1e-2)),
        spline_map_geometry=bool(meta.get("spline_map_geometry", False)),
    )

    if map_mode in ("gvec", "h5"):
        h5_path = _require_map_geometry_h5(meta, map_mode=map_mode)
        if map_mode == "gvec":
            print(
                "map_mode=gvec: using map_geometry_h5 + stellarator_map interpolation "
                "(gvec_jax_map removed).",
                flush=True,
            )
        seq, map_jit, _, map_raw = build_sequence_h5(
            map_geometry_h5=h5_path,
            **h5_kwargs,
        )
    elif map_mode == "analytic":
        seq, map_jit, _, map_raw = build_sequence_analytic(
            ns=ns,
            p=p,
            betti=betti,
            tol=tol,
            maxiter=maxiter,
            map_kind=str(meta.get("map_kind", "rotating_ellipse")),
            nfp=nfp,
            ellipse_eps=float(meta.get("ellipse_eps", 1.0 / 3.0)),
            ellipse_kappa=float(meta.get("ellipse_kappa", 1.0)),
            ellipse_r0=float(meta.get("ellipse_r0", 1.0)),
            cerfon_alpha=float(meta.get("cerfon_alpha", 0.0)),
            shafranov_e0=float(meta.get("shafranov_e0", 0.3)),
            shafranov_delta0=float(meta.get("shafranov_delta0", 0.2)),
        )
    else:
        raise ValueError(f"Unsupported map_mode={map_mode!r} in meta")

    return seq, map_jit, map_raw, nfp


def load_dof_vector(dof_npy: Path, seq: DeRhamSequence, *, k: int) -> jnp.ndarray:
    """Load and shape-check a saved nullspace DOF vector."""
    dof_npy = dof_npy.expanduser().resolve()
    v = jnp.asarray(np.load(dof_npy), dtype=jnp.float64).reshape(-1)
    n_expected = int(seq.n2_dbc if int(k) == 2 else seq.n1)
    if int(v.shape[0]) != n_expected:
        raise ValueError(
            f"{dof_npy}: DOF length {int(v.shape[0])} != expected n={n_expected} for k={k}"
        )
    return v
