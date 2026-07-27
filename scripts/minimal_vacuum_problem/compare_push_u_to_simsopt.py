#!/usr/bin/env python3
"""
Compare ``Pushforward(u)`` (or ``curl(u)`` for k=1) to SIMSOPT vacuum **B** (Biot–Savart).

Samples both fields at the **same Cartesian points** built from an MRX-style volume HDF5
(``eval_points``, ``R``, ``Z``) — same convention as
:mod:`scripts.wip.compare_B_mrx_simsopt_quasr`.

**Physics**

* **k=1:** ``Pushforward(u)`` is a **1-form** (not **B**). Prefer ``--compare-curl`` for
  ``strong_curl(u)`` vs SIMSOPT **B**.
* **k=2:** ``Pushforward(u₂)`` is a 2-form in Cartesian clothing — shape diagnostic vs **B**.

Requires: ``simsopt`` env for Biot–Savart; ``mrx`` + a prior nullspace run (``--dof-npy`` +
``hcurl_nullspace_meta.json`` or ``hodge_k2_nullspace_meta.json``).

Examples
--------
::

    conda activate simsopt_vmec  # or env with simsopt + mrx

    # Cluster (Slurm driver: scripts/wip/run_compare_push_u_to_simsopt.slurm):
    SIMSOPT_JSON=~/Downloads/serial0044972.json \\
      VOLUME_H5=/scratch/$USER/mrx/data/quasr_new_0044970_mrx.h5 \\
      DOF_NPY=/path/to/hcurl_k1_nullspace_dof.npy \\
      NFP=3 FLIP_ZETA=1 \\
      OUT_JSON=/scratch/$USER/mrx/scripts/wip/script_outputs/push_u_vs_simsopt.json \\
      sbatch scripts/wip/run_compare_push_u_to_simsopt.slurm

    # k=1: push(u) and curl(u) vs QUASR coils at GVEC volume points
    python scripts/wip/compare_push_u_to_simsopt.py \\
      --json ~/Downloads/serial0044972.json \\
      --volume-h5 /scratch/js11789/mrx/data/w7x_ini_mrx.h5 \\
      --dof-npy /path/to/hcurl_nullspace_dof.npy \\
      --from-saved-meta \\
      --compare-curl \\
      --nfp 5 --flip-zeta \\
      -o push_u_vs_simsopt.json

    # k=2 harmonic u₂
    python scripts/wip/compare_push_u_to_simsopt.py \\
      --json ~/Downloads/serial0044972.json \\
      --volume-h5 /scratch/js11789/mrx/data/quasr0065575_mrx.h5 \\
      --dof-npy /path/to/hodge_k2_dbc_nullspace_dof.npy \\
      --from-saved-meta --k 2 \\
      --nfp 4 -o u2_vs_simsopt.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import h5py
import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mrx.differential_forms import DiscreteFunction, Pushforward  # noqa: E402
from scripts.wip.compare_B_mrx_simsopt_quasr import (  # noqa: E402
    xyz_from_mrx_eval_points,
)

jax.config.update("jax_enable_x64", True)


def _ensure_jax_config_submodule() -> None:
    """SIMSOPT expects ``import jax.config``; register alias on newer JAX releases."""
    if "jax.config" in sys.modules:
        return
    m = types.ModuleType("jax.config")
    m.config = jax.config
    sys.modules["jax.config"] = m


def _pointwise_vector_metrics(
    u_phys: np.ndarray,
    b_ref: np.ndarray,
    *,
    align_pointwise: bool = False,
) -> dict[str, float]:
    """
    RMS vector metrics at sample points (equal weights, Cartesian components).

    ``rel_l2`` uses reference **B** in the denominator. Optional alignment uses a single
    scalar ``s = Σ(u·B) / Σ|u|²`` over all sample points.
    """
    diff = u_phys - b_ref
    l2_vec = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    l2_ref = float(np.sqrt(np.mean(np.sum(b_ref**2, axis=1))))
    l2_u = float(np.sqrt(np.mean(np.sum(u_phys**2, axis=1))))
    rel = l2_vec / l2_ref if l2_ref > 0.0 else float("nan")
    out: dict[str, float] = {
        "l2_diff_vector": l2_vec,
        "h5_B_rms_per_point": l2_ref,
        "u_pushforward_rms_per_point": l2_u,
        "rel_l2": rel,
    }
    if align_pointwise:
        num = float(np.sum(u_phys * b_ref))
        den = float(np.sum(u_phys * u_phys))
        scale = num / den if den > 0.0 else float("nan")
        diff_a = scale * u_phys - b_ref
        l2_vec_a = float(np.sqrt(np.mean(np.sum(diff_a**2, axis=1))))
        rel_a = l2_vec_a / l2_ref if l2_ref > 0.0 else float("nan")
        out["pointwise_u_scale_to_optimal"] = float(scale)
        out["l2_diff_vector_aligned"] = l2_vec_a
        out["rel_l2_aligned"] = rel_a
        u_norm = np.linalg.norm(u_phys, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b_ref, axis=1, keepdims=True)
        floor = 1e-30 * max(float(l2_ref), 1.0)
        u_hat = u_phys / np.maximum(u_norm, floor)
        b_hat = b_ref / np.maximum(b_norm, floor)
        diff_dir = u_hat - b_hat
        l2_dir = float(np.sqrt(np.mean(np.sum(diff_dir**2, axis=1))))
        out["rel_l2_direction"] = l2_dir / np.sqrt(3.0)
    return out


def _load_hcurl_module():
    path = _REPO / "scripts/wip/hcurl_nullspace_gvec_quasr.py"
    spec = importlib.util.spec_from_file_location("hcurl_nullspace_gvec_quasr_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_twoform_module() -> Any:
    """Load the k=2 Hodge-nullspace driver without importing scripts as a package."""
    path = _REPO / "scripts/wip/twoform_hodge_nullspace_gvec_quasr.py"
    spec = importlib.util.spec_from_file_location("twoform_hodge_nullspace_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _normalize_eval_points_gvec(pts: jnp.ndarray, *, file_gvec: bool) -> jnp.ndarray:
    """MRX logical [0,1]³ from GVEC radians if attr set."""
    if not file_gvec:
        return pts
    two_pi = 2.0 * jnp.pi
    pts = pts.at[:, 0].set(jnp.clip(pts[:, 0], 0.0, 1.0))
    pts = pts.at[:, 1].set((pts[:, 1] / two_pi) % 1.0)
    pts = pts.at[:, 2].set((pts[:, 2] / two_pi) % 1.0)
    return pts


def _load_meta_for_dof(
    dof_path: Path,
    *,
    k: int,
    meta_json: Path | None,
) -> dict[str, Any]:
    """Load nullspace meta JSON beside ``dof`` or from explicit path."""
    if meta_json is not None:
        meta_fp = meta_json.expanduser().resolve()
    elif k == 2:
        meta_fp = dof_path.with_name("hodge_k2_nullspace_meta.json")
    else:
        meta_fp = dof_path.with_name("hcurl_nullspace_meta.json")
    if not meta_fp.is_file():
        raise FileNotFoundError(f"Meta JSON not found: {meta_fp}")
    return json.loads(meta_fp.read_text())


def _rebuild_sequence_from_meta(
    meta: dict[str, Any],
    *,
    hcurl_mod: Any,
    pushforward_only: bool = False,
) -> tuple[Any, Callable[[jnp.ndarray], jnp.ndarray], int, bool, Callable[[jnp.ndarray], jnp.ndarray]]:
    """Rebuild ``DeRhamSequence``, JIT map, and single-period ``map_raw`` from meta.

    When ``pushforward_only`` is true, skip reference mass assembly and ``set_map`` on the
    full quadrature grid (only needed for operator solves). Sufficient for
    ``Pushforward(u)`` at ``eval_points``.
    """
    nfp = int(meta["nfp"])
    flip_zeta = bool(meta.get("flip_zeta", False))
    ns = meta["ns"]
    ns_t = (int(ns[0]), int(ns[1]), int(ns[2]))
    p_deg = int(meta["p"])
    betti = tuple(int(x) for x in meta.get("betti", (1, 1, 0, 0)))
    tol = float(meta.get("tol", 1.0e-12))
    maxiter = int(meta.get("maxiter", 50))
    map_mode = str(meta.get("map_mode", "gvec"))

    if map_mode == "gvec":
        gvec_runpath = Path(str(meta["gvec_runpath"]))
        seq, map_jit, _, map_raw = hcurl_mod.build_sequence_gvec(
            ns=ns_t,
            p=p_deg,
            betti=betti,
            tol=tol,
            maxiter=maxiter,
            gvec_runpath=gvec_runpath,
            nfp=nfp,
            flip_zeta=flip_zeta,
            gvec_flip_r=bool(meta.get("gvec_flip_r", False)),
            gvec_fd_eps=float(meta.get("gvec_fd_eps", 1.0e-7)),
            strict_jacobian=False,
            pushforward_only=bool(pushforward_only),
        )
    else:
        seq, map_jit, _, map_raw = hcurl_mod.build_sequence_analytic(
            ns=ns_t,
            p=p_deg,
            betti=betti,
            tol=tol,
            maxiter=maxiter,
            map_kind=str(meta.get("map_kind", "ellipse")),
            nfp=nfp,
            ellipse_eps=float(meta.get("ellipse_eps", 0.0)),
            ellipse_kappa=float(meta.get("ellipse_kappa", 1.0)),
            ellipse_r0=float(meta.get("ellipse_r0", 1.0)),
            cerfon_alpha=float(meta.get("cerfon_alpha", 0.0)),
            shafranov_e0=float(meta.get("shafranov_e0", 0.0)),
            shafranov_delta0=float(meta.get("shafranov_delta0", 0.0)),
        )
    return seq, map_jit, nfp, flip_zeta, map_raw


def _load_simsopt_field(json_path: Path) -> tuple[Any, list[Any]]:
    """Load the SIMSOPT Biot–Savart field and surfaces from QUASR data."""
    _ensure_jax_config_submodule()
    from simsopt._core import load
    from simsopt.field import BiotSavart

    loaded = load(str(json_path))
    if not isinstance(loaded, (list, tuple)) or len(loaded) != 2:
        raise RuntimeError(f"Expected [surfaces, coils] from load(); got {type(loaded)}")
    surfaces, coils = list(loaded[0]), loaded[1]
    return BiotSavart(coils), surfaces


def _evaluate_simsopt_field(field: Any, pts_xyz: np.ndarray) -> np.ndarray:
    """Evaluate a SIMSOPT magnetic field at Cartesian ``(N,3)`` points."""
    field.set_points(np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3))
    return np.asarray(field.B(), dtype=np.float64).reshape(-1, 3)


def _fourier_mode_spectrum(
    values: np.ndarray,
    theta_grid: np.ndarray,
    zeta_grid: np.ndarray,
    *,
    normalization: float,
    max_poloidal_mode: int = 16,
    max_field_period_mode: int = 8,
) -> list[dict[str, float | int]]:
    """Return complex Fourier amplitudes on a normalized field-period grid.

    Parameters
    ----------
    values
        Scalar samples with the same two-dimensional shape as ``theta_grid``.
    theta_grid, zeta_grid
        Poloidal and field-period angles measured in cycles.
    normalization
        Positive scale used to make each reported amplitude dimensionless.
    max_poloidal_mode, max_field_period_mode
        Inclusive nonnegative mode limits.
    """
    scalar = np.asarray(values, dtype=np.float64)
    if scalar.shape != np.asarray(theta_grid).shape or scalar.shape != np.asarray(
        zeta_grid
    ).shape:
        raise ValueError("Fourier values and angle grids must have matching shapes")
    scale = float(normalization)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Fourier normalization must be finite and positive")
    spectrum: list[dict[str, float | int]] = []
    for poloidal_mode in range(int(max_poloidal_mode) + 1):
        for field_period_mode in range(int(max_field_period_mode) + 1):
            phase = np.exp(
                -2.0j
                * np.pi
                * (
                    poloidal_mode * np.asarray(theta_grid)
                    - field_period_mode * np.asarray(zeta_grid)
                )
            )
            coefficient = np.mean(scalar * phase)
            spectrum.append(
                {
                    "poloidal_mode": int(poloidal_mode),
                    "field_period_mode": int(field_period_mode),
                    "amplitude": float(abs(coefficient)),
                    "relative_amplitude": float(abs(coefficient) / scale),
                    "phase_radians": float(np.angle(coefficient)),
                }
            )
    return spectrum


def _normal_flux_summary(
    field: Any,
    points_xyz: np.ndarray,
    unit_normals: np.ndarray,
    theta_grid: np.ndarray,
    zeta_grid: np.ndarray,
    *,
    nfp: int,
    max_poloidal_mode: int = 16,
    max_field_period_mode: int = 8,
) -> dict[str, Any]:
    """Quantify normal Biot–Savart flux and its Fourier content on a surface."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(unit_normals, dtype=np.float64).reshape(-1, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    magnetic_field = _evaluate_simsopt_field(field, points)
    field_norm = np.linalg.norm(magnetic_field, axis=1)
    field_rms = float(np.sqrt(np.mean(field_norm**2)))
    normal_field = np.sum(magnetic_field * normals, axis=1)
    pointwise_ratio = normal_field / np.maximum(field_norm, 1.0e-30)
    shape = np.asarray(theta_grid).shape
    absolute_spectrum = _fourier_mode_spectrum(
        normal_field.reshape(shape),
        theta_grid,
        zeta_grid,
        normalization=field_rms,
        max_poloidal_mode=max_poloidal_mode,
        max_field_period_mode=max_field_period_mode,
    )
    ratio_spectrum = _fourier_mode_spectrum(
        pointwise_ratio.reshape(shape),
        theta_grid,
        zeta_grid,
        normalization=1.0,
        max_poloidal_mode=max_poloidal_mode,
        max_field_period_mode=max_field_period_mode,
    )

    def mode(
        spectrum: list[dict[str, float | int]],
        poloidal_mode: int,
        field_period_mode: int,
    ) -> dict[str, float | int]:
        """Select one mode from a complete nonnegative spectrum."""
        return next(
            item
            for item in spectrum
            if int(item["poloidal_mode"]) == int(poloidal_mode)
            and int(item["field_period_mode"]) == int(field_period_mode)
        )

    ranked = sorted(
        (
            item
            for item in absolute_spectrum
            if int(item["poloidal_mode"]) + int(item["field_period_mode"]) > 0
        ),
        key=lambda item: float(item["relative_amplitude"]),
        reverse=True,
    )
    resonant_absolute = mode(absolute_spectrum, 6, 1)
    resonant_ratio = mode(ratio_spectrum, 6, 1)
    return {
        "samples": int(points.shape[0]),
        "field_rms_tesla": field_rms,
        "normal_field_rms_tesla": float(np.sqrt(np.mean(normal_field**2))),
        "normal_field_max_abs_tesla": float(np.max(np.abs(normal_field))),
        "normal_over_field_rms": float(np.sqrt(np.mean(pointwise_ratio**2))),
        "normal_over_field_max_abs": float(np.max(np.abs(pointwise_ratio))),
        "resonant_mode": {
            "poloidal_mode": 6,
            "toroidal_mode": int(nfp),
            "field_period_mode": 1,
            "normal_field_fourier_tesla": float(
                resonant_absolute["amplitude"]
            ),
            "normal_field_fourier_relative_to_B_rms": float(
                resonant_absolute["relative_amplitude"]
            ),
            "normal_over_field_fourier_amplitude": float(
                resonant_ratio["amplitude"]
            ),
        },
        "largest_nonconstant_modes": ranked[:12],
        "spectrum": absolute_spectrum,
    }


def _boundary_flux_diagnostics(
    simsopt_field: Any,
    quasr_surface: Any,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    ntheta: int = 64,
    nzeta: int = 64,
) -> dict[str, Any]:
    """Compare coil-field normal flux on QUASR and GVEC boundary surfaces."""
    from simsopt.geo import SurfaceXYZTensorFourier

    theta = np.arange(int(ntheta), dtype=np.float64) / int(ntheta)
    zeta = np.arange(int(nzeta), dtype=np.float64) / int(nzeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    quasr_grid = SurfaceXYZTensorFourier(
        nfp=int(quasr_surface.nfp),
        stellsym=bool(quasr_surface.stellsym),
        mpol=int(quasr_surface.mpol),
        ntor=int(quasr_surface.ntor),
        quadpoints_phi=zeta / int(nfp),
        quadpoints_theta=theta,
    )
    quasr_grid.set_dofs(np.asarray(quasr_surface.get_dofs(), dtype=np.float64))
    quasr_points = np.swapaxes(np.asarray(quasr_grid.gamma()), 0, 1)
    quasr_normals = np.swapaxes(np.asarray(quasr_grid.unitnormal()), 0, 1)
    quasr_summary = _normal_flux_summary(
        simsopt_field,
        quasr_points,
        quasr_normals,
        theta_grid,
        zeta_grid,
        nfp=int(nfp),
    )

    logical = np.stack(
        [
            np.ones(theta_grid.size),
            theta_grid.ravel(),
            zeta_grid.ravel(),
        ],
        axis=1,
    )
    logical_jax = jnp.asarray(logical, dtype=jnp.float64)
    mapped = np.asarray(
        jax.lax.map(map_fn, logical_jax, batch_size=256),
        dtype=np.float64,
    )
    jacobian_fn = jax.jit(jax.jacfwd(map_fn))
    jacobians = np.asarray(
        jax.lax.map(jacobian_fn, logical_jax, batch_size=128),
        dtype=np.float64,
    )
    grad_rho = np.linalg.solve(
        np.swapaxes(jacobians, 1, 2),
        np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), mapped.shape)[..., None],
    )[..., 0]
    gvec_summary = _normal_flux_summary(
        simsopt_field,
        mapped,
        grad_rho,
        theta_grid,
        zeta_grid,
        nfp=int(nfp),
    )
    geometry_difference = mapped - quasr_points.reshape(-1, 3)
    return {
        "grid_ntheta": int(ntheta),
        "grid_nzeta": int(nzeta),
        "quasr_surface": quasr_summary,
        "gvec_rho_1_surface": gvec_summary,
        "surface_geometry_rms_difference_m": float(
            np.sqrt(np.mean(np.sum(geometry_difference**2, axis=1)))
        ),
        "surface_geometry_max_difference_m": float(
            np.max(np.linalg.norm(geometry_difference, axis=1))
        ),
    }


def _evaluate_biot_savart_periodic_quadrature(
    coils: list[Any],
    points_xyz: np.ndarray,
    *,
    quadrature_factor: int,
) -> np.ndarray:
    """Evaluate coil fields with independently refined periodic quadrature."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    result = np.zeros_like(points)
    mu0_over_4pi = 1.0e-7
    for coil in coils:
        nquad = int(len(coil.curve.quadpoints)) * int(quadrature_factor)
        parameter = np.arange(nquad, dtype=np.float64) / nquad
        # Initialize SIMSOPT's curve-side caches before requesting arbitrary
        # quadrature points. RotatedCurve delegates this request to its parent.
        coil.curve.gamma()
        gamma = np.empty((nquad, 3), dtype=np.float64)
        coil.curve.gamma_impl(gamma, parameter)
        frequencies = np.fft.fftfreq(nquad, d=1.0 / nquad)
        derivative = np.fft.ifft(
            (2.0j * np.pi * frequencies)[:, None]
            * np.fft.fft(gamma, axis=0),
            axis=0,
        ).real
        current = float(coil.current.get_value())
        for start in range(0, points.shape[0], 256):
            targets = points[start : start + 256]
            displacement = gamma[:, None, :] - targets[None, :, :]
            denominator = np.linalg.norm(displacement, axis=2) ** 3
            integrand = np.cross(displacement, derivative[:, None, :])
            result[start : start + targets.shape[0]] += (
                mu0_over_4pi
                * current
                * np.mean(integrand / denominator[..., None], axis=0)
            )
    return result


def _full_torus_surface(surface: Any) -> Any:
    """Return a full-torus RZ-Fourier copy suitable for signed distances."""
    rz_surface = surface.to_RZFourier()
    return rz_surface.copy(
        nphi=max(4 * int(rz_surface.nfp) * int(rz_surface.ntor) + 1, 81),
        ntheta=max(4 * int(rz_surface.mpol) + 1, 41),
        range="full torus",
    )


def _simsopt_B_at_xyz(json_path: Path, pts_xyz: np.ndarray) -> np.ndarray:
    """Vacuum **B** from QUASR SIMSOPT coils at Cartesian ``(N,3)`` [m]."""
    field, _ = _load_simsopt_field(json_path)
    return _evaluate_simsopt_field(field, pts_xyz)


def _evaluate_pushforward(
    seq: Any,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pts: jnp.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Evaluate ``Pushforward(u)`` at logical ``eval_points``, shape ``(N,3)``."""
    u = jnp.asarray(u, dtype=jnp.float64)
    if k == 1:
        disc = DiscreteFunction(u, seq.basis_1, seq.e1)
    elif k == 2:
        disc = DiscreteFunction(u, seq.basis_2, seq.e2_dbc)
    else:
        raise ValueError(f"k must be 1 or 2; got {k}")
    push = Pushforward(disc, map_fn, k)
    return np.asarray(
        jax.lax.map(
            push,
            jnp.asarray(pts, dtype=jnp.float64).reshape(-1, 3),
            batch_size=512,
        ),
        dtype=np.float64,
    ).reshape(-1, 3)


def _evaluate_curl_u_cartesian(
    seq: Any,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    pts: jnp.ndarray,
) -> np.ndarray:
    """``strong_curl(u)`` pushed to sample points (Cartesian), shape ``(N,3)``."""
    B_dof = seq.apply_strong_curl(jnp.asarray(u), dirichlet_in=False, dirichlet_out=True)
    from scripts.wip.hcurl_nullspace_gvec_quasr import _pick_e2_extractor

    E2 = _pick_e2_extractor(seq, B_dof)
    disc_b = DiscreteFunction(B_dof, seq.basis_2, E2)
    push_b = Pushforward(disc_b, map_fn, 2)
    return np.asarray(
        jax.lax.map(
            push_b,
            jnp.asarray(pts, dtype=jnp.float64).reshape(-1, 3),
            batch_size=512,
        ),
        dtype=np.float64,
    ).reshape(-1, 3)


def _parse_section_values(text: str) -> list[float]:
    """Parse comma-separated normalized section values into ``[0,1)``."""
    values = [float(value.strip()) % 1.0 for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one zeta section value is required")
    return values


def _parse_resolution_values(text: str) -> list[tuple[int, int]]:
    """Parse comma-separated ``nrho x ntheta`` resolution specifications."""
    resolutions: list[tuple[int, int]] = []
    for item in text.split(","):
        if not item.strip():
            continue
        parts = item.lower().strip().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid resolution {item!r}; expected NRHOxNTHETA")
        nrho, ntheta = (int(value) for value in parts)
        if nrho < 2 or ntheta < 3:
            raise ValueError("Resolution sweep requires nrho >= 2 and ntheta >= 3")
        resolutions.append((nrho, ntheta))
    if not resolutions:
        raise ValueError("At least one resolution is required")
    return resolutions


def _parse_fem_resolution_values(text: str) -> list[tuple[int, int, int]]:
    """Parse comma-separated ``nr x ntheta x nzeta`` FEM resolutions."""
    resolutions: list[tuple[int, int, int]] = []
    for item in text.split(","):
        if not item.strip():
            continue
        parts = item.lower().strip().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid FEM resolution {item!r}; expected NRxNTxNZ")
        resolution = tuple(int(value) for value in parts)
        if any(value < 2 for value in resolution):
            raise ValueError("FEM resolutions must be at least 2 in every direction")
        resolutions.append(resolution)
    if not resolutions:
        raise ValueError("At least one FEM resolution is required")
    return resolutions


def _poloidal_slice_points(
    zeta: float,
    nrho: int,
    ntheta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a structured logical ``(rho, theta)`` grid at fixed ``zeta``."""
    if nrho < 2 or ntheta < 3:
        raise ValueError("Slice grids require nrho >= 2 and ntheta >= 3")
    rho = np.linspace(1.0e-5, 1.0 - 1.0e-5, int(nrho))
    theta = np.linspace(0.0, 1.0, int(ntheta), endpoint=False)
    rho_grid, theta_grid = np.meshgrid(rho, theta, indexing="ij")
    points = np.stack(
        [
            rho_grid.ravel(),
            theta_grid.ravel(),
            np.full(rho_grid.size, float(zeta) % 1.0),
        ],
        axis=1,
    )
    return points, rho_grid, theta_grid


def _close_poloidal_slice(
    R: np.ndarray,
    Z: np.ndarray,
    *fields: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Append the first poloidal column so contour plots close periodically."""
    closed: list[np.ndarray] = [
        np.hstack([np.asarray(R), np.asarray(R)[:, :1]]),
        np.hstack([np.asarray(Z), np.asarray(Z)[:, :1]]),
    ]
    for field in fields:
        array = np.asarray(field)
        closed.append(np.concatenate([array, array[:, :1, ...]], axis=1))
    return tuple(closed)


def _evaluate_component_slices(
    seq: Any,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    simsopt_field: Any,
    zeta_values: list[float],
    *,
    nrho: int,
    ntheta: int,
    mrx_scale: float,
) -> list[dict[str, Any]]:
    """Evaluate aligned MRX and SIMSOPT fields on fixed-zeta poloidal grids."""
    slices: list[dict[str, Any]] = []
    for zeta in zeta_values:
        logical, rho_grid, theta_grid = _poloidal_slice_points(zeta, nrho, ntheta)
        mapped = np.asarray(
            jax.lax.map(
                map_fn,
                jnp.asarray(logical, dtype=jnp.float64),
                batch_size=512,
            ),
            dtype=np.float64,
        ).reshape(nrho, ntheta, 3)
        b_mrx_raw = _evaluate_pushforward(
            seq,
            u,
            map_fn,
            jnp.asarray(logical),
            k=2,
        ).reshape(nrho, ntheta, 3)
        b_simsopt = _evaluate_simsopt_field(
            simsopt_field,
            mapped.reshape(-1, 3),
        ).reshape(nrho, ntheta, 3)
        slices.append(
            {
                "zeta": float(zeta),
                "rho": rho_grid,
                "theta": theta_grid,
                "R": np.sqrt(mapped[..., 0] ** 2 + mapped[..., 1] ** 2),
                "Z": mapped[..., 2],
                "B_mrx": float(mrx_scale) * b_mrx_raw,
                "B_simsopt": b_simsopt,
            }
        )
    return slices


def _plot_component_slices(
    slices: list[dict[str, Any]],
    output_dir: Path,
    *,
    mrx_scale: float,
    dpi: int = 150,
) -> list[Path]:
    """Write 3×3 MRX, SIMSOPT, and difference component cross-sections."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    component_labels = ("x", "y", "z")
    for slab in slices:
        R, Z, b_mrx, b_simsopt = _close_poloidal_slice(
            slab["R"],
            slab["Z"],
            slab["B_mrx"],
            slab["B_simsopt"],
        )
        difference = b_mrx - b_simsopt
        fig, axes = plt.subplots(3, 3, figsize=(14, 12), squeeze=False)
        for component, label in enumerate(component_labels):
            mrx_values = b_mrx[..., component]
            simsopt_values = b_simsopt[..., component]
            diff_values = difference[..., component]
            field_limit = max(
                float(np.max(np.abs(mrx_values))),
                float(np.max(np.abs(simsopt_values))),
                1.0e-15,
            )
            diff_limit = max(float(np.max(np.abs(diff_values))), 1.0e-15)
            panels = (
                (
                    mrx_values,
                    rf"$s^* B_{{\mathrm{{MRX}},{label}}}$",
                    field_limit,
                ),
                (
                    simsopt_values,
                    rf"$B_{{\mathrm{{SIMSOPT}},{label}}}$",
                    field_limit,
                ),
                (
                    diff_values,
                    rf"$\Delta B_{label}$",
                    diff_limit,
                ),
            )
            for column, (values, title, limit) in enumerate(panels):
                ax = axes[component, column]
                contour = ax.contourf(
                    R,
                    Z,
                    values,
                    levels=25,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.set_xlabel("$R$ [m]")
                ax.set_ylabel("$Z$ [m]")
                fig.colorbar(contour, ax=ax, label="T")
        zeta = float(slab["zeta"])
        fig.suptitle(
            rf"Cartesian field components at $\zeta={zeta:.3f}$; "
            rf"$s^*={float(mrx_scale):.6g}$",
            y=1.01,
        )
        fig.tight_layout()
        path = output_dir / f"B_components_zeta{zeta:.3f}.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        outputs.append(path.resolve())
    return outputs


def _benchmark_resolution_sweep(
    seq: Any,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    simsopt_field: Any,
    zeta_values: list[float],
    resolutions: list[tuple[int, int]],
    *,
    mrx_scale: float,
) -> tuple[list[dict[str, float | int]], list[dict[str, float]]]:
    """Measure aligned slice error and MRX evaluation time versus grid size."""
    records: list[dict[str, float | int]] = []
    finest_rho = np.empty(0)
    finest_mrx = np.empty((0, 3))
    finest_simsopt = np.empty((0, 3))
    for nrho, ntheta in resolutions:
        mrx_fields: list[np.ndarray] = []
        simsopt_fields: list[np.ndarray] = []
        rho_values: list[np.ndarray] = []
        mrx_seconds = 0.0
        for zeta in zeta_values:
            logical, _, _ = _poloidal_slice_points(zeta, nrho, ntheta)
            rho_values.append(logical[:, 0])
            mapped = np.asarray(
                jax.lax.map(
                    map_fn,
                    jnp.asarray(logical, dtype=jnp.float64),
                    batch_size=512,
                ),
                dtype=np.float64,
            )
            simsopt_fields.append(_evaluate_simsopt_field(simsopt_field, mapped))
            start = time.perf_counter()
            mrx_fields.append(
                _evaluate_pushforward(
                    seq,
                    u,
                    map_fn,
                    jnp.asarray(logical),
                    k=2,
                )
            )
            mrx_seconds += time.perf_counter() - start
        b_mrx = float(mrx_scale) * np.concatenate(mrx_fields, axis=0)
        b_simsopt = np.concatenate(simsopt_fields, axis=0)
        difference_rms = float(
            np.sqrt(np.mean(np.sum((b_mrx - b_simsopt) ** 2, axis=1)))
        )
        reference_rms = float(
            np.sqrt(np.mean(np.sum(b_simsopt**2, axis=1)))
        )
        records.append(
            {
                "nrho": int(nrho),
                "ntheta": int(ntheta),
                "points_per_section": int(nrho * ntheta),
                "total_points": int(nrho * ntheta * len(zeta_values)),
                "aligned_rel_l2": difference_rms / reference_rms,
                "mrx_evaluation_seconds": float(mrx_seconds),
            }
        )
        finest_rho = np.concatenate(rho_values)
        finest_mrx = b_mrx
        finest_simsopt = b_simsopt
    radial_profile = _radial_error_profile(
        finest_rho,
        finest_mrx,
        finest_simsopt,
    )
    return records, radial_profile


def _radial_error_profile(
    rho: np.ndarray,
    b_mrx: np.ndarray,
    b_simsopt: np.ndarray,
) -> list[dict[str, float]]:
    """Compute shell-wise aligned vector error on a structured rho grid."""
    rho_values = np.asarray(rho, dtype=np.float64).reshape(-1)
    mrx = np.asarray(b_mrx, dtype=np.float64).reshape(-1, 3)
    simsopt = np.asarray(b_simsopt, dtype=np.float64).reshape(-1, 3)
    if rho_values.size != mrx.shape[0] or mrx.shape != simsopt.shape:
        raise ValueError("rho and magnetic-field samples must have matching lengths")
    profile: list[dict[str, float]] = []
    for shell in np.unique(rho_values):
        mask = np.isclose(rho_values, shell, rtol=0.0, atol=1.0e-12)
        error_rms = float(
            np.sqrt(np.mean(np.sum((mrx[mask] - simsopt[mask]) ** 2, axis=1)))
        )
        reference_rms = float(
            np.sqrt(np.mean(np.sum(simsopt[mask] ** 2, axis=1)))
        )
        profile.append(
            {
                "rho": float(shell),
                "absolute_error_rms_T": error_rms,
                "relative_error_rms": error_rms / reference_rms,
            }
        )
    return profile


def _plot_resolution_sweep(
    records: list[dict[str, float | int]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> list[Path]:
    """Plot aligned error and measured MRX evaluation time versus resolution."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray([record["points_per_section"] for record in records])
    errors = np.asarray([record["aligned_rel_l2"] for record in records])
    timings = np.asarray([record["mrx_evaluation_seconds"] for record in records])
    labels = [
        f'{record["nrho"]}×{record["ntheta"]}'
        for record in records
    ]
    fig, error_axis = plt.subplots(figsize=(7.5, 5.0))
    timing_axis = error_axis.twinx()
    error_line = error_axis.plot(
        points,
        errors,
        "o-",
        color="tab:blue",
        label="Sampled aligned relative $L^2$ error",
    )[0]
    reference_line = error_axis.axhline(
        float(errors[-1]),
        color="tab:blue",
        linestyle=":",
        linewidth=1.0,
        label="Finest-grid estimate",
    )
    timing_line = timing_axis.plot(
        points,
        timings,
        "s--",
        color="tab:red",
        label="MRX evaluation time",
    )[0]
    error_axis.set_xscale("log")
    error_axis.set_yscale("log")
    error_axis.set_xlabel("Logical points per fixed-$\\zeta$ section")
    error_axis.set_ylabel("Sampled error estimate (fixed FEM solution)")
    timing_axis.set_ylabel("MRX Pushforward wall time [s]")
    error_axis.set_xticks(points, labels, rotation=25)
    error_axis.grid(True, which="both", alpha=0.25)
    error_axis.legend(
        [error_line, reference_line, timing_line],
        [
            error_line.get_label(),
            reference_line.get_label(),
            timing_line.get_label(),
        ],
        loc="best",
    )
    fig.suptitle("Fixed-FEM error estimator and MRX timing versus sample grid")
    fig.tight_layout()
    outputs = [
        (output_dir / "error_and_mrx_time_vs_resolution.png").resolve(),
        (output_dir / "error_and_mrx_time_vs_resolution.pdf").resolve(),
    ]
    for path in outputs:
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return outputs


def _plot_radial_error_profile(
    profile: list[dict[str, float]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> list[Path]:
    """Plot aligned field discrepancy versus normalized radial coordinate."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rho = np.asarray([record["rho"] for record in profile])
    relative_error = np.asarray(
        [record["relative_error_rms"] for record in profile]
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(rho, relative_error, "o-", color="tab:blue")
    axis.set_xlabel(r"Normalized radius $\rho$")
    axis.set_ylabel("Shell-wise aligned relative vector RMS error")
    axis.set_title("Where the fixed-FEM vacuum-field error is concentrated")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    outputs = [
        (output_dir / "error_vs_rho.png").resolve(),
        (output_dir / "error_vs_rho.pdf").resolve(),
    ]
    for path in outputs:
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return outputs


def _load_fem_sweep_records(
    paths: Path | list[Path] | None,
) -> list[dict[str, Any]]:
    """Load completed FEM-resolution records from earlier output JSON files."""
    if paths is None:
        return []
    requested_paths = [paths] if isinstance(paths, Path) else paths
    loaded: list[dict[str, Any]] = []
    for path in requested_paths:
        data = json.loads(path.expanduser().resolve().read_text())
        records = data.get("postprocessing", {}).get("fem_resolution_sweep", [])
        if not isinstance(records, list):
            raise ValueError(f"{path} does not contain a FEM resolution sweep")
        loaded.extend(dict(record) for record in records)
    return loaded


def _compute_null_k2_iterative(
    seq: Any,
    eps: float,
    *,
    initial_dof: jnp.ndarray | None = None,
    inner_tol: float = 1.0e-10,
    maxiter: int = 100,
) -> tuple[jnp.ndarray, list[dict[str, float | int | bool]]]:
    """Compute the k=2 DBC harmonic vector without the cubic dense eigensolve."""
    from mrx.nullspace import find_nullspace_vectors

    n_harm = int(tuple(int(value) for value in seq.betti_numbers)[1])
    if n_harm <= 0:
        raise RuntimeError("k=2 DBC harmonic solve requires b1 > 0")
    initial_guesses = (
        [initial_dof, *([None] * (n_harm - 1))]
        if initial_dof is not None
        else None
    )
    vectors, info = find_nullspace_vectors(
        seq,
        seq._require_operators(),
        2,
        n_harm,
        float(eps),
        dirichlet=True,
        x0s=initial_guesses,
        inner_tol=float(inner_tol),
        maxiter=int(maxiter),
    )
    if int(vectors.shape[0]) == 0:
        raise RuntimeError("iterative k=2 DBC nullspace solve returned no vectors")
    iteration_info = [
        {
            "iterations": int(iterations),
            "residual_norm": float(residual),
            "converged": bool(float(residual) <= float(seq.tol)),
        }
        for iterations, residual in info
    ]
    return vectors[0], iteration_info


def _k2_algebraic_health(
    seq: Any,
    dof: jnp.ndarray,
    *,
    inverse_tol: float = 1.0e-10,
) -> dict[str, float]:
    """Return scale-invariant algebraic health metrics for a DBC 2-form.

    The Hodge-Laplacian residual is a dual 2-form, so its norm is measured
    with ``M2^{-1}``. The divergence and weak-curl terms are reported as
    energy norms relative to the physical ``M2`` norm of the input vector.
    """
    vector = jnp.asarray(dof, dtype=jnp.float64).reshape(-1)
    operators = (
        seq._require_operators() if hasattr(seq, "_require_operators") else None
    )

    def inverse_mass(rhs: jnp.ndarray, degree: int) -> jnp.ndarray:
        """Apply a tightly converged inverse mass solve for diagnostics."""
        if operators is None:
            return seq.apply_inverse_mass_matrix(
                rhs,
                degree,
                dirichlet=True,
                tol=float(inverse_tol),
            )
        from mrx.operators import apply_inverse_mass_matrix

        return apply_inverse_mass_matrix(
            seq,
            operators,
            rhs,
            degree,
            dirichlet=True,
            tol=float(inverse_tol),
            maxiter=max(1000, int(getattr(seq, "maxiter", 50))),
        )

    mass_vector = seq.apply_mass_matrix(vector, 2, dirichlet=True)
    norm_sq = float(vector @ mass_vector)
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("k=2 algebraic health requires a positive M2 norm")

    if operators is None:
        laplacian_vector = seq.apply_hodge_laplacian(
            vector,
            2,
            dirichlet=True,
        )
    else:
        from mrx.operators import apply_hodge_laplacian

        laplacian_vector = apply_hodge_laplacian(
            seq,
            operators,
            vector,
            2,
            dirichlet=True,
            tol=float(inverse_tol),
            maxiter=max(1000, int(getattr(seq, "maxiter", 50))),
        )
    inverse_mass_residual = inverse_mass(laplacian_vector, 2)
    residual_sq = max(float(laplacian_vector @ inverse_mass_residual), 0.0)

    divergence_dual = seq.apply_derivative_matrix(
        vector,
        2,
        dirichlet_in=True,
        dirichlet_out=True,
    )
    strong_divergence = inverse_mass(divergence_dual, 3)
    divergence_sq = max(float(divergence_dual @ strong_divergence), 0.0)

    weak_curl_dual = seq.apply_derivative_matrix(
        vector,
        1,
        dirichlet_in=True,
        dirichlet_out=True,
        transpose=True,
    )
    weak_curl = inverse_mass(weak_curl_dual, 1)
    weak_curl_sq = max(float(weak_curl_dual @ weak_curl), 0.0)
    rayleigh = float(vector @ laplacian_vector) / norm_sq
    return {
        "m2_norm": float(np.sqrt(norm_sq)),
        "rayleigh_quotient": rayleigh,
        "relative_residual_m2_inverse": float(np.sqrt(residual_sq / norm_sq)),
        "relative_divergence_energy": float(
            np.sqrt(divergence_sq / norm_sq)
        ),
        "relative_weak_curl_energy": float(
            np.sqrt(weak_curl_sq / norm_sq)
        ),
        "energy_identity_relative_error": float(
            abs((divergence_sq + weak_curl_sq) - float(vector @ laplacian_vector))
            / max(abs(float(vector @ laplacian_vector)), np.finfo(float).tiny)
        ),
    }


def _dense_verify_k2_nullspace(
    seq: Any,
    twoform_module: Any,
    iterative_dof: jnp.ndarray,
) -> tuple[jnp.ndarray, list[float], dict[str, float]]:
    """Compute and compare the dense k=2 DBC null vector."""
    matrix = np.asarray(
        twoform_module._sym(
            np.asarray(
                twoform_module.dense_hodge_laplacian(
                    seq,
                    seq._require_operators(),
                    2,
                    dirichlet=True,
                )
            )
        )
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    dense = jnp.asarray(eigenvectors[:, 0], dtype=jnp.float64)
    dense = dense / seq.l2_norm(dense, 2, dirichlet=True)
    iterative = jnp.asarray(iterative_dof, dtype=jnp.float64)
    iterative = iterative / seq.l2_norm(iterative, 2, dirichlet=True)
    overlap = float(dense @ seq.apply_mass_matrix(iterative, 2, dirichlet=True))
    if overlap < 0.0:
        dense = -dense
        overlap = -overlap
    distance = float(seq.l2_norm(dense - iterative, 2, dirichlet=True))
    comparison = {
        "m2_overlap_with_iterative": overlap,
        "m2_distance_to_iterative": distance,
    }
    return dense, np.asarray(eigenvalues[:5]).tolist(), comparison


def _run_fem_resolution_sweep(
    meta: dict[str, Any],
    twoform_module: Any,
    hcurl_module: Any,
    simsopt_field: Any,
    zeta_values: list[float],
    resolutions: list[tuple[int, int, int]],
    *,
    base_seq: Any,
    base_dof: jnp.ndarray,
    base_map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    base_dof_path: Path,
    resume_records: list[dict[str, Any]],
    nrho: int,
    ntheta: int,
    time_budget_seconds: float,
    dense_check_max_dofs: int,
    solver_eps: float,
    inner_tol: float,
    iterative_maxiter: int,
    refine_current: bool,
    refine_resolutions: set[tuple[int, int, int]],
    dense_verify_resolutions: set[tuple[int, int, int]],
    compute_health: bool,
    output_dir: Path,
    poincare_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resume or solve k=2 nullspaces and measure error versus FEM resolution."""
    if str(meta.get("map_mode", "gvec")) != "gvec":
        raise ValueError("FEM resolution sweeps currently require a GVEC map")
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_by_ns = {
        tuple(int(value) for value in record["ns"]): record
        for record in resume_records
    }
    base_ns = tuple(int(value) for value in meta["ns"])
    records: list[dict[str, Any]] = []
    budget_exceeded = False
    finest_resolution = max(
        resolutions,
        key=lambda resolution: int(np.prod(resolution)),
    )

    def build_full_sequence(
        resolution: tuple[int, int, int],
    ) -> tuple[Any, Callable[[jnp.ndarray], jnp.ndarray]]:
        """Build one fully assembled sequence for solving or diagnostics."""
        full_seq, full_map, _, _ = twoform_module.build_sequence_gvec(
            ns=tuple(int(value) for value in resolution),
            p=int(meta["p"]),
            betti=tuple(
                int(value) for value in meta.get("betti", (1, 1, 0, 0))
            ),
            tol=float(meta.get("tol", 1.0e-12)),
            maxiter=int(meta.get("maxiter", 50)),
            gvec_runpath=Path(str(meta["gvec_runpath"])),
            nfp=int(meta["nfp"]),
            flip_zeta=bool(meta.get("flip_zeta", False)),
            gvec_flip_r=bool(meta.get("gvec_flip_r", False)),
            gvec_fd_eps=float(meta.get("gvec_fd_eps", 1.0e-7)),
            strict_jacobian=False,
        )
        twoform_module.assemble_ops_k2(full_seq)
        return full_seq, full_map

    for ns in resolutions:
        if any(int(value) <= int(meta["p"]) for value in ns):
            raise ValueError(
                f"FEM resolution {ns} must exceed spline degree p={meta['p']}"
            )
        prior = prior_by_ns.get(ns)
        label = "x".join(str(value) for value in ns)
        refining_base = ns == base_ns and bool(refine_current)
        refining = refining_base or ns in refine_resolutions
        dense_comparison: dict[str, float] | None = None
        if ns == base_ns and not refining:
            if compute_health:
                seq, map_fn = build_full_sequence(ns)
            else:
                seq, map_fn = base_seq, base_map_fn
            harmonic_dof = jnp.asarray(base_dof)
            dof_path = base_dof_path.resolve()
            solve_seconds = (
                float(prior["solve_seconds"])
                if prior is not None and prior.get("solve_seconds") is not None
                else None
            )
            eigenvalues = (
                [
                    float(value)
                    for value in prior.get("smallest_eigenvalues", [])
                ]
                if prior is not None
                else []
            )
            solve_method = (
                str(prior.get("solve_method", "saved-current"))
                if prior is not None
                else "saved-current"
            )
            iterative_info: Any = (
                prior.get("iterative_info") if prior is not None else None
            )
            dense_comparison = (
                prior.get("dense_verification")
                if prior is not None
                else None
            )
        elif (
            prior is not None
            and Path(str(prior["dof_npy"])).is_file()
            and not refining
        ):
            if compute_health:
                seq, map_fn = build_full_sequence(ns)
            else:
                resolution_meta = dict(meta)
                resolution_meta["ns"] = list(ns)
                seq, map_fn, _, _, _ = _rebuild_sequence_from_meta(
                    resolution_meta,
                    hcurl_mod=hcurl_module,
                    pushforward_only=True,
                )
            dof_path = Path(str(prior["dof_npy"])).expanduser().resolve()
            harmonic_dof = jnp.asarray(np.load(dof_path), dtype=jnp.float64)
            solve_seconds = (
                float(prior["solve_seconds"])
                if prior.get("solve_seconds") is not None
                else None
            )
            eigenvalues = [
                float(value) for value in prior.get("smallest_eigenvalues", [])
            ]
            solve_method = str(prior.get("solve_method", "resumed"))
            iterative_info = prior.get("iterative_info")
            dense_comparison = prior.get("dense_verification")
        else:
            if budget_exceeded:
                continue
            start = time.perf_counter()
            seq, map_fn = build_full_sequence(ns)
            initial_dof: jnp.ndarray | None = None
            if prior is not None and Path(str(prior["dof_npy"])).is_file():
                initial_dof = jnp.asarray(
                    np.load(Path(str(prior["dof_npy"])).expanduser().resolve()),
                    dtype=jnp.float64,
                )
            elif refining_base:
                initial_dof = jnp.asarray(base_dof, dtype=jnp.float64)
            if int(seq.n2_dbc) > int(dense_check_max_dofs):
                harmonic_dof, iterative_info = _compute_null_k2_iterative(
                    seq,
                    float(solver_eps),
                    initial_dof=initial_dof,
                    inner_tol=float(inner_tol),
                    maxiter=int(iterative_maxiter),
                )
                eigenvalues = []
                solve_method = (
                    "iterative-warm-start"
                    if initial_dof is not None
                    else "iterative-only"
                )
            else:
                harmonic_dof, smallest = twoform_module.compute_null_k2_dbc(
                    seq,
                    float(solver_eps),
                )
                eigenvalues = np.asarray(smallest).tolist()
                solve_method = "dense-check-plus-iterative"
                iterative_info = None
            dense_comparison: dict[str, float] | None = None
            if ns in dense_verify_resolutions:
                harmonic_dof, eigenvalues, dense_comparison = (
                    _dense_verify_k2_nullspace(
                        seq,
                        twoform_module,
                        harmonic_dof,
                    )
                )
                solve_method = f"{solve_method}+dense-verified"
            solve_seconds = time.perf_counter() - start
            dof_path = (output_dir / f"fem_sweep_{label}_dof.npy").resolve()
            np.save(dof_path, np.asarray(harmonic_dof))
            if solve_seconds > float(time_budget_seconds):
                budget_exceeded = True
        algebraic_health = (
            _k2_algebraic_health(
                seq,
                harmonic_dof,
                inverse_tol=float(inner_tol),
            )
            if compute_health
            else (
                prior.get("algebraic_health")
                if prior is not None
                else None
            )
        )
        mrx_fields: list[np.ndarray] = []
        simsopt_fields: list[np.ndarray] = []
        for zeta in zeta_values:
            logical, _, _ = _poloidal_slice_points(zeta, nrho, ntheta)
            mapped = np.asarray(
                jax.lax.map(
                    map_fn,
                    jnp.asarray(logical, dtype=jnp.float64),
                    batch_size=512,
                )
            )
            mrx_fields.append(
                _evaluate_pushforward(
                    seq,
                    harmonic_dof,
                    map_fn,
                    jnp.asarray(logical),
                    k=2,
                )
            )
            simsopt_fields.append(_evaluate_simsopt_field(simsopt_field, mapped))
        metrics = _pointwise_vector_metrics(
            np.concatenate(mrx_fields),
            np.concatenate(simsopt_fields),
            align_pointwise=True,
        )
        record: dict[str, Any] = {
            "ns": [int(value) for value in ns],
            "p": int(meta["p"]),
            "n2_dbc": int(seq.n2_dbc),
            "solve_seconds": solve_seconds,
            "solve_method": solve_method,
            "smallest_eigenvalues": eigenvalues,
            "iterative_info": iterative_info,
            "dense_verification": dense_comparison,
            "algebraic_health": algebraic_health,
            "pointwise_u_scale_to_optimal": metrics[
                "pointwise_u_scale_to_optimal"
            ],
            "aligned_rel_l2": metrics["rel_l2_aligned"],
            "evaluation_nrho": int(nrho),
            "evaluation_ntheta": int(ntheta),
            "dof_npy": str(dof_path),
        }
        if poincare_config is not None:
            diagnostics_enabled = bool(
                poincare_config.get("island_diagnostics", False)
            )
            prior_poincare = (
                prior is not None
                and prior.get("poincare_file") is not None
                and Path(str(prior["poincare_file"])).is_file()
                and not refining
                and (
                    not diagnostics_enabled
                    or (
                        prior.get("island_diagnostics") is not None
                        and int(
                            prior["island_diagnostics"].get(
                                "diagnostic_version",
                                1,
                            )
                        )
                        >= 2
                        and (
                            not bool(poincare_config.get("island_zoom", False))
                            or prior["island_diagnostics"].get("island_zoom_file")
                            is not None
                        )
                    )
                )
            )
            if prior_poincare:
                if (
                    diagnostics_enabled
                    and prior is not None
                    and prior.get("island_diagnostics") is not None
                    and not prior["island_diagnostics"].get(
                        "resonant_normal_error_radial_profile"
                    )
                ):
                    prior["island_diagnostics"][
                        "resonant_normal_error_radial_profile"
                    ] = _resonant_normal_error_radial_profile(
                        seq,
                        harmonic_dof,
                        map_fn,
                        simsopt_field,
                        np.linspace(0.3, 0.95, 10),
                        mrx_scale=float(
                            metrics["pointwise_u_scale_to_optimal"]
                        ),
                        nfp=int(poincare_config["nfp"]),
                        ntheta=int(poincare_config["fourier_ntheta"]),
                        nzeta=int(poincare_config["fourier_nzeta"]),
                    )
                record["poincare_file"] = str(prior["poincare_file"])
                for key in (
                    "poincare_turns",
                    "mrx_completed_transits",
                    "mrx_intersection_counts",
                    "island_diagnostics",
                ):
                    if key in prior:
                        record[key] = prior[key]
                records.append(record)
                continue
            trace_audit: dict[str, Any] = {}
            if (
                ns == base_ns
                and not refining
                and poincare_config.get("base_mrx_sections") is not None
            ):
                mrx_sections = poincare_config["base_mrx_sections"]
                transit_counts = poincare_config["base_transit_counts"]
                logical_sections = poincare_config.get(
                    "base_logical_sections",
                    {},
                )
            else:
                logical_sections: dict[float, list[np.ndarray]] = {}
                (
                    mrx_sections,
                    _,
                    _,
                    transit_counts,
                ) = _trace_mrx_poincare(
                    seq,
                    harmonic_dof,
                    map_fn,
                    zeta_values,
                    nlines=int(poincare_config["nlines"]),
                    turns=int(poincare_config["turns"]),
                    theta0=float(poincare_config["theta0"]),
                    tol=float(poincare_config["tol"]),
                    logical_sections_out=(
                        logical_sections if diagnostics_enabled else None
                    ),
                    audit_out=trace_audit if diagnostics_enabled else None,
                )
            if diagnostics_enabled and not trace_audit:
                trace_audit = {
                    "method": "RK45",
                    "rtol": float(poincare_config["tol"]),
                    "atol": float(poincare_config["tol"]),
                    "section_crossing_construction": (
                        "exact t_eval because the reparameterized ODE enforces dzeta/dt=1"
                    ),
                    "max_unwrapped_section_coordinate_residual": 0.0,
                    "seed_rho_count": int(
                        np.unique(
                            np.asarray(poincare_config["logical_seeds"])[:, 0]
                        ).size
                    ),
                    "seed_theta_count": int(
                        np.unique(
                            np.asarray(poincare_config["logical_seeds"])[:, 1]
                        ).size
                    ),
                    "total_lines": int(
                        np.asarray(poincare_config["logical_seeds"]).shape[0]
                    ),
                }
            if diagnostics_enabled:
                trace_audit["comparison_plot_rendering"] = {
                    "marker_size_points_squared": 2.0,
                    "shared_axes": True,
                    "axis_limit_source": (
                        "common MRX/SIMSOPT boundary and all plotted points"
                    ),
                    "explicit_data_clipping": False,
                }
            poincare_path = _plot_poincare_comparison(
                mrx_sections,
                poincare_config["simsopt_sections"],
                poincare_config["slices"],
                output_dir,
                filename=f"poincare_mrx_vs_simsopt_fem_{label}.png",
            )
            record["poincare_file"] = str(poincare_path)
            record["poincare_turns"] = int(poincare_config["turns"])
            record["mrx_completed_transits"] = transit_counts
            record["mrx_intersection_counts"] = {
                f"{zeta:.12g}": [int(points.shape[0]) for points in lines]
                for zeta, lines in mrx_sections.items()
            }
            if diagnostics_enabled:
                section_zeta = float(poincare_config["diagnostic_zeta"])
                logical_lines = logical_sections[section_zeta]
                seed_rho = np.asarray(
                    poincare_config["logical_seeds"],
                    dtype=np.float64,
                )[:, 0]
                iota_profile = _rotational_transform_profile(
                    logical_lines,
                    seed_rho,
                    nfp=int(poincare_config["nfp"]),
                )
                resonances = _identify_iota_resonances(
                    iota_profile,
                    nfp=int(poincare_config["nfp"]),
                    rho_min=float(poincare_config["rho_min"]),
                    rho_max=float(poincare_config["rho_max"]),
                    max_poloidal_mode=int(
                        poincare_config["max_poloidal_mode"]
                    ),
                )[: int(poincare_config["max_resonances"])]
                width_profile = _island_width_profile(
                    logical_lines,
                    seed_rho,
                )
                resonant_amplitudes = _resonant_normal_error_amplitudes(
                    seq,
                    harmonic_dof,
                    map_fn,
                    simsopt_field,
                    resonances,
                    mrx_scale=float(
                        metrics["pointwise_u_scale_to_optimal"]
                    ),
                    ntheta=int(poincare_config["fourier_ntheta"]),
                    nzeta=int(poincare_config["fourier_nzeta"]),
                )
                resonant_amplitudes_refined = _resonant_normal_error_amplitudes(
                    seq,
                    harmonic_dof,
                    map_fn,
                    simsopt_field,
                    resonances,
                    mrx_scale=float(
                        metrics["pointwise_u_scale_to_optimal"]
                    ),
                    ntheta=2 * int(poincare_config["fourier_ntheta"]),
                    nzeta=2 * int(poincare_config["fourier_nzeta"]),
                )
                radial_resonant_profile = _resonant_normal_error_radial_profile(
                    seq,
                    harmonic_dof,
                    map_fn,
                    simsopt_field,
                    np.linspace(0.3, 0.95, 10),
                    mrx_scale=float(
                        metrics["pointwise_u_scale_to_optimal"]
                    ),
                    nfp=int(poincare_config["nfp"]),
                    ntheta=int(poincare_config["fourier_ntheta"]),
                    nzeta=int(poincare_config["fourier_nzeta"]),
                )
                resonant_widths: list[dict[str, float | int]] = []
                for resonance in resonances:
                    if not width_profile:
                        continue
                    nearest_width = min(
                        width_profile,
                        key=lambda item: abs(
                            float(item["seed_rho"])
                            - float(resonance["rho"])
                        ),
                    )
                    resonant_widths.append(
                        {
                            **resonance,
                            "sample_seed_rho": float(
                                nearest_width["seed_rho"]
                            ),
                            "width_rho_q05_q95": float(
                                nearest_width["width_rho_q05_q95"]
                            ),
                            "detrended_width_rho_q05_q95": float(
                                nearest_width[
                                    "detrended_width_rho_q05_q95"
                                ]
                            ),
                        }
                    )
                relative_amplitudes = [
                    float(item["normal_error_fourier_relative"])
                    for item in resonant_amplitudes
                ]
                aliasing_control: list[dict[str, float | int]] = []
                for coarse, refined_amplitude in zip(
                    resonant_amplitudes,
                    resonant_amplitudes_refined,
                ):
                    coarse_value = float(
                        coarse["normal_error_fourier_relative"]
                    )
                    refined_value = float(
                        refined_amplitude["normal_error_fourier_relative"]
                    )
                    aliasing_control.append(
                        {
                            "poloidal_mode": int(coarse["poloidal_mode"]),
                            "toroidal_mode": int(coarse["toroidal_mode"]),
                            "coarse_grid_ntheta": int(
                                poincare_config["fourier_ntheta"]
                            ),
                            "coarse_grid_nzeta": int(
                                poincare_config["fourier_nzeta"]
                            ),
                            "coarse_relative_amplitude": coarse_value,
                            "refined_relative_amplitude": refined_value,
                            "relative_change": float(
                                abs(refined_value - coarse_value)
                                / max(abs(refined_value), 1.0e-30)
                            ),
                        }
                    )
                zoom_file: str | None = None
                trapped_summary: dict[str, Any] | None = None
                if bool(poincare_config.get("island_zoom", False)):
                    zoom_rho = np.linspace(
                        float(poincare_config["zoom_rho_min"]),
                        float(poincare_config["zoom_rho_max"]),
                        int(poincare_config["zoom_nrho"]),
                    )
                    base_theta = float(poincare_config["theta0"]) % 1.0
                    zoom_theta = (
                        base_theta
                        + np.arange(int(poincare_config["zoom_phases"]))
                        / (
                            6.0
                            * max(int(poincare_config["zoom_phases"]), 1)
                        )
                    ) % 1.0
                    zoom_logical_sections: dict[float, list[np.ndarray]] = {}
                    _, zoom_seeds, _, _ = _trace_mrx_poincare(
                        seq,
                        harmonic_dof,
                        map_fn,
                        [section_zeta],
                        nlines=int(zoom_rho.size),
                        turns=int(poincare_config["turns"]),
                        theta0=base_theta,
                        tol=float(poincare_config["tol"]),
                        logical_sections_out=zoom_logical_sections,
                        seed_rho_values=zoom_rho,
                        theta0_values=zoom_theta,
                    )
                    zoom_lines = zoom_logical_sections[section_zeta]
                    trapped_summary = _trapped_separatrix_summary(
                        zoom_lines,
                        zoom_seeds,
                        poloidal_mode=6,
                    )
                    zoom_path = _plot_island_zoom(
                        zoom_lines,
                        zoom_seeds,
                        output_dir,
                        label=label,
                        rho_min=float(poincare_config["zoom_rho_min"]),
                        rho_max=float(poincare_config["zoom_rho_max"]),
                    )
                    zoom_file = str(zoom_path)
                integrator_control: dict[str, Any] | None = None
                if ns == finest_resolution:
                    control_rho = np.linspace(
                        float(poincare_config["zoom_rho_min"]),
                        float(poincare_config["zoom_rho_max"]),
                        3,
                    )
                    control_widths: dict[str, list[float]] = {}
                    for control_name, method, control_tol in (
                        (
                            "baseline",
                            "RK45",
                            float(poincare_config["tol"]),
                        ),
                        ("tight_dop853", "DOP853", 1.0e-11),
                    ):
                        control_sections: dict[float, list[np.ndarray]] = {}
                        _trace_mrx_poincare(
                            seq,
                            harmonic_dof,
                            map_fn,
                            [section_zeta],
                            nlines=3,
                            turns=int(poincare_config["turns"]),
                            theta0=float(poincare_config["theta0"]),
                            tol=control_tol,
                            logical_sections_out=control_sections,
                            seed_rho_values=control_rho,
                            method=method,
                        )
                        control_profile = _island_width_profile(
                            control_sections[section_zeta],
                            control_rho,
                        )
                        control_widths[control_name] = [
                            float(item["detrended_width_rho_q05_q95"])
                            for item in control_profile
                        ]
                    baseline_widths = np.asarray(
                        control_widths["baseline"],
                        dtype=np.float64,
                    )
                    tight_widths = np.asarray(
                        control_widths["tight_dop853"],
                        dtype=np.float64,
                    )
                    integrator_control = {
                        "seed_rho": control_rho.tolist(),
                        "baseline_method": "RK45",
                        "baseline_tol": float(poincare_config["tol"]),
                        "tight_method": "DOP853",
                        "tight_tol": 1.0e-11,
                        "baseline_detrended_widths": baseline_widths.tolist(),
                        "tight_detrended_widths": tight_widths.tolist(),
                        "max_relative_width_change": float(
                            np.max(
                                np.abs(tight_widths - baseline_widths)
                                / np.maximum(np.abs(tight_widths), 1.0e-30)
                            )
                        ),
                    }
                corrected_widths = [
                    float(item["detrended_width_rho_q05_q95"])
                    for item in resonant_widths
                ]
                record["island_diagnostics"] = {
                    "diagnostic_version": 2,
                    "iota_profile": iota_profile,
                    "resonances": resonances,
                    "island_width_profile": width_profile,
                    "resonant_island_widths": resonant_widths,
                    "max_island_width_rho": (
                        max(
                            float(item["width_rho_q05_q95"])
                            for item in resonant_widths
                        )
                        if resonant_widths
                        else None
                    ),
                    "max_detrended_island_width_rho": (
                        max(corrected_widths) if corrected_widths else None
                    ),
                    "poincare_trace_audit": trace_audit,
                    "fourier_aliasing_control": aliasing_control,
                    "island_zoom_file": zoom_file,
                    "trapped_separatrix": trapped_summary,
                    "integrator_control": integrator_control,
                    "resonant_normal_error": resonant_amplitudes,
                    "resonant_normal_error_radial_profile": (
                        radial_resonant_profile
                    ),
                    "max_resonant_normal_error_relative": (
                        max(relative_amplitudes)
                        if relative_amplitudes
                        else None
                    ),
                }
        records.append(record)
    return records


def _plot_fem_resolution_sweep(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> list[Path]:
    """Plot actual re-solved FEM error and solve time versus k=2 DOF count."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    dofs = np.asarray([record["n2_dbc"] for record in records], dtype=np.float64)
    errors = np.asarray([record["aligned_rel_l2"] for record in records])
    timings = np.asarray(
        [
            float(record["solve_seconds"])
            if record.get("solve_seconds") is not None
            else np.nan
            for record in records
        ]
    )
    labels = ["×".join(str(value) for value in record["ns"]) for record in records]
    characteristic_n = np.asarray(
        [np.prod(record["ns"]) ** (1.0 / 3.0) for record in records]
    )
    fit_mask = np.isfinite(errors) & (errors > 0.0)
    observed_order = float("nan")
    if np.count_nonzero(fit_mask) >= 2:
        observed_order = -float(
            np.polyfit(
                np.log(characteristic_n[fit_mask]),
                np.log(errors[fit_mask]),
                1,
            )[0]
        )
    fig, error_axis = plt.subplots(figsize=(7.5, 5.0))
    timing_axis = error_axis.twinx()
    error_line = error_axis.plot(
        dofs,
        errors,
        "o-",
        color="tab:blue",
        label="Re-solved aligned relative $L^2$ error",
    )[0]
    timing_line = timing_axis.plot(
        dofs,
        timings,
        "s--",
        color="tab:red",
        label="Sequence assembly + nullspace solve",
    )[0]
    error_axis.set_xscale("log")
    error_axis.set_yscale("log")
    timing_axis.set_yscale("log")
    error_axis.set_xlabel("Dirichlet k=2 DOF count")
    error_axis.set_ylabel("Aligned relative vector $L^2$ error")
    timing_axis.set_ylabel("MRX solve wall time [s]")
    error_axis.set_xticks(dofs, labels, rotation=25)
    error_axis.grid(True, which="both", alpha=0.25)
    if np.isfinite(observed_order):
        error_axis.text(
            0.30,
            0.05,
            rf"Global fit: error $\propto n^{{-{observed_order:.2f}}}$",
            transform=error_axis.transAxes,
        )
    error_axis.legend(
        [error_line, timing_line],
        [error_line.get_label(), timing_line.get_label()],
        loc="best",
    )
    fig.suptitle("Actual FEM convergence: re-solved harmonic 2-form")
    fig.tight_layout()
    outputs = [
        (output_dir / "error_and_solve_time_vs_fem_dofs.png").resolve(),
        (output_dir / "error_and_solve_time_vs_fem_dofs.pdf").resolve(),
    ]
    for path in outputs:
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return outputs


def _plot_island_diagnostics(
    records: list[dict[str, Any]],
    simsopt_iota_profile: list[dict[str, float]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> list[Path]:
    """Plot iota profiles and island/resonant-error convergence diagnostics."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    diagnosed = [
        record for record in records if record.get("island_diagnostics") is not None
    ]
    if not diagnosed:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)

    iota_fig, iota_axis = plt.subplots(figsize=(7.2, 5.0))
    if simsopt_iota_profile:
        iota_axis.plot(
            [item["rho"] for item in simsopt_iota_profile],
            [item["iota"] for item in simsopt_iota_profile],
            "k--",
            linewidth=1.8,
            label="SIMSOPT",
        )
    for record in diagnosed:
        profile = record["island_diagnostics"]["iota_profile"]
        iota_axis.plot(
            [item["rho"] for item in profile],
            [item["iota"] for item in profile],
            "o-",
            markersize=3,
            label="MRX " + "×".join(str(value) for value in record["ns"]),
        )
    iota_axis.set_xlabel(r"Seed radius $\rho$")
    iota_axis.set_ylabel(r"Rotational transform $\iota$")
    iota_axis.set_title(
        "Rotational transform: MRX crossings and SIMSOPT pitch average"
    )
    iota_axis.grid(True, alpha=0.25)
    iota_axis.legend(fontsize="small", ncol=2)
    iota_fig.tight_layout()
    iota_paths = [
        (output_dir / "iota_profile_vs_resolution.png").resolve(),
        (output_dir / "iota_profile_vs_resolution.pdf").resolve(),
    ]
    for path in iota_paths:
        iota_fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(iota_fig)

    labels = ["×".join(str(value) for value in record["ns"]) for record in diagnosed]
    widths = np.asarray(
        [
            (
                record["island_diagnostics"]["max_island_width_rho"]
                if record["island_diagnostics"]["max_island_width_rho"]
                is not None
                else np.nan
            )
            for record in diagnosed
        ],
        dtype=np.float64,
    )
    corrected_widths = np.asarray(
        [
            (
                record["island_diagnostics"][
                    "max_detrended_island_width_rho"
                ]
                if record["island_diagnostics"].get(
                    "max_detrended_island_width_rho"
                )
                is not None
                else np.nan
            )
            for record in diagnosed
        ],
        dtype=np.float64,
    )
    amplitudes = np.asarray(
        [
            (
                record["island_diagnostics"][
                    "max_resonant_normal_error_relative"
                ]
                if record["island_diagnostics"][
                    "max_resonant_normal_error_relative"
                ]
                is not None
                else np.nan
            )
            for record in diagnosed
        ],
        dtype=np.float64,
    )
    errors = np.asarray(
        [record["aligned_rel_l2"] for record in diagnosed],
        dtype=np.float64,
    )
    width_fit_mask = (corrected_widths > 0.0) & (errors > 0.0)
    valid_indices = np.flatnonzero(width_fit_mask)
    asymptotic_indices = valid_indices[-3:]
    width_exponent = float("nan")
    if asymptotic_indices.size >= 2:
        width_exponent = float(
            np.polyfit(
                np.log(errors[asymptotic_indices]),
                np.log(corrected_widths[asymptotic_indices]),
                1,
            )[0]
        )

    summary_fig, width_axis = plt.subplots(figsize=(7.5, 5.0))
    x_values = np.arange(len(diagnosed))
    width_line = width_axis.plot(
        x_values,
        widths,
        "o-",
        color="tab:blue",
        label=r"Raw radial width near $\iota=1/2$",
    )[0]
    corrected_width_line = width_axis.plot(
        x_values,
        corrected_widths,
        "d-",
        color="tab:green",
        label=r"Fourier-detrended six-lobe width",
    )[0]
    width_axis.set_xticks(x_values, labels, rotation=25)
    width_axis.set_xlabel("FEM resolution")
    width_axis.set_ylabel(r"Robust island width $\Delta\rho_{5-95\%}$")
    if np.any(
        (np.isfinite(widths) & (widths > 0.0))
        | (np.isfinite(corrected_widths) & (corrected_widths > 0.0))
    ):
        width_axis.set_yscale("log")
    width_axis.grid(True, which="both", alpha=0.25)
    if np.isfinite(width_exponent):
        width_axis.text(
            0.03,
            0.05,
            rf"Finest 3: width $\propto$ error$^{{{width_exponent:.2f}}}$",
            transform=width_axis.transAxes,
        )
    amplitude_mask = np.isfinite(amplitudes) & (amplitudes > 0.0)
    if np.any(amplitude_mask):
        amplitude_axis = width_axis.twinx()
        amplitude_line = amplitude_axis.plot(
            x_values,
            amplitudes,
            "s--",
            color="tab:red",
            label="Max resonant normal-error amplitude / B rms",
        )[0]
        amplitude_axis.set_ylabel("Relative resonant normal-error amplitude")
        amplitude_axis.set_yscale("log")
        width_axis.legend(
            [width_line, corrected_width_line, amplitude_line],
            [
                width_line.get_label(),
                corrected_width_line.get_label(),
                amplitude_line.get_label(),
            ],
            loc="best",
        )
    else:
        width_axis.legend(
            [width_line, corrected_width_line],
            [width_line.get_label(), corrected_width_line.get_label()],
            loc="best",
        )
    summary_fig.suptitle("Island width and resonant error versus FEM resolution")
    summary_fig.tight_layout()
    summary_paths = [
        (output_dir / "island_width_and_resonant_error.png").resolve(),
        (output_dir / "island_width_and_resonant_error.pdf").resolve(),
    ]
    for path in summary_paths:
        summary_fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(summary_fig)
    return [*iota_paths, *summary_paths]


def _island_scaling_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit global field convergence and finest-three island scalings."""
    diagnosed = [
        record for record in records if record.get("island_diagnostics") is not None
    ]
    if len(diagnosed) < 2:
        return {
            "field_error_order": None,
            "island_width_error_exponent_finest_three": None,
            "detrended_island_width_error_exponent_finest_three": None,
            "resonant_amplitude_error_exponent_finest_three": None,
            "fit_points": len(diagnosed),
        }
    errors = np.asarray([record["aligned_rel_l2"] for record in diagnosed])
    characteristic_n = np.asarray(
        [np.prod(record["ns"]) ** (1.0 / 3.0) for record in diagnosed]
    )
    widths = np.asarray(
        [
            (
                record["island_diagnostics"]["max_island_width_rho"]
                if record["island_diagnostics"]["max_island_width_rho"]
                is not None
                else np.nan
            )
            for record in diagnosed
        ],
        dtype=np.float64,
    )
    corrected_widths = np.asarray(
        [
            (
                record["island_diagnostics"][
                    "max_detrended_island_width_rho"
                ]
                if record["island_diagnostics"].get(
                    "max_detrended_island_width_rho"
                )
                is not None
                else np.nan
            )
            for record in diagnosed
        ],
        dtype=np.float64,
    )
    amplitudes = np.asarray(
        [
            (
                record["island_diagnostics"][
                    "max_resonant_normal_error_relative"
                ]
                if record["island_diagnostics"][
                    "max_resonant_normal_error_relative"
                ]
                is not None
                else np.nan
            )
            for record in diagnosed
        ],
        dtype=np.float64,
    )
    field_order = -float(
        np.polyfit(np.log(characteristic_n), np.log(errors), 1)[0]
    )

    def finest_exponent(values: np.ndarray) -> float | None:
        valid = np.flatnonzero(
            np.isfinite(values) & (values > 0.0) & (errors > 0.0)
        )[-3:]
        if valid.size < 2:
            return None
        return float(
            np.polyfit(np.log(errors[valid]), np.log(values[valid]), 1)[0]
        )

    return {
        "field_error_order": field_order,
        "island_width_error_exponent_finest_three": finest_exponent(widths),
        "detrended_island_width_error_exponent_finest_three": finest_exponent(
            corrected_widths
        ),
        "resonant_amplitude_error_exponent_finest_three": finest_exponent(
            amplitudes
        ),
        "fit_points": len(diagnosed),
    }


def _island_width_consistency_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit ``w=C sqrt(a_63/(m |d iota/d rho|))`` across diagnosed runs."""
    rows: list[dict[str, float | list[int]]] = []
    for record in records:
        diagnostics = record.get("island_diagnostics")
        if not diagnostics:
            continue
        resonances = diagnostics.get("resonances", [])
        target = next(
            (
                item
                for item in resonances
                if int(item["poloidal_mode"]) == 6
                and abs(int(item["toroidal_mode"])) == 3
            ),
            None,
        )
        amplitudes = diagnostics.get("resonant_normal_error", [])
        amplitude = next(
            (
                item
                for item in amplitudes
                if int(item["poloidal_mode"]) == 6
                and abs(int(item["toroidal_mode"])) == 3
            ),
            None,
        )
        width = diagnostics.get("max_detrended_island_width_rho")
        profile = sorted(
            diagnostics.get("iota_profile", []),
            key=lambda item: float(item["rho"]),
        )
        if (
            target is None
            or amplitude is None
            or width is None
            or len(profile) < 3
        ):
            continue
        rho_values = np.asarray([item["rho"] for item in profile], dtype=np.float64)
        iota_values = np.asarray(
            [item["iota"] for item in profile],
            dtype=np.float64,
        )
        resonance_rho = float(target["rho"])
        nearest = int(np.argmin(np.abs(rho_values - resonance_rho)))
        lo = max(0, nearest - 2)
        hi = min(rho_values.size, nearest + 3)
        shear = float(
            np.polyfit(rho_values[lo:hi], iota_values[lo:hi], 1)[0]
        )
        relative_amplitude = float(
            amplitude["normal_error_fourier_relative"]
        )
        predictor = float(
            np.sqrt(
                relative_amplitude
                / (int(target["poloidal_mode"]) * max(abs(shear), 1.0e-14))
            )
        )
        rows.append(
            {
                "ns": [int(value) for value in record["ns"]],
                "resonance_rho": resonance_rho,
                "local_shear_diota_drho": shear,
                "relative_amplitude_a63": relative_amplitude,
                "detrended_width_rho": float(width),
                "unscaled_predictor": predictor,
            }
        )
    if len(rows) < 2:
        return {"fit_points": len(rows), "constant_C": None, "records": rows}
    predictors = np.asarray(
        [item["unscaled_predictor"] for item in rows],
        dtype=np.float64,
    )
    widths = np.asarray(
        [item["detrended_width_rho"] for item in rows],
        dtype=np.float64,
    )
    fit_start = max(0, len(rows) - 3)
    fit_predictors = predictors[fit_start:]
    fit_widths = widths[fit_start:]
    constant = float(
        np.dot(fit_predictors, fit_widths)
        / np.dot(fit_predictors, fit_predictors)
    )
    fitted = constant * predictors
    for index, (item, prediction, residual) in enumerate(
        zip(rows, fitted, widths - fitted)
    ):
        item["predicted_width_rho"] = float(prediction)
        item["relative_residual"] = float(
            residual / max(abs(float(item["detrended_width_rho"])), 1.0e-30)
        )
        item["included_in_finest_three_fit"] = bool(index >= fit_start)
    return {
        "fit_points": len(rows),
        "fit_points_finest_three": int(fit_predictors.size),
        "constant_C": constant,
        "relative_rms_residual_finest_three": float(
            np.sqrt(np.mean((fit_widths - constant * fit_predictors) ** 2))
            / np.sqrt(np.mean(fit_widths**2))
        ),
        "relative_rms_residual_all_resolutions": float(
            np.sqrt(np.mean((widths - fitted) ** 2))
            / np.sqrt(np.mean(widths**2))
        ),
        "records": rows,
    }


def _periodic_section_intersections(
    trajectory: np.ndarray,
    zeta: float,
    *,
    max_intersections: int,
) -> np.ndarray:
    """Find a periodic fixed-zeta section and return logical intersections."""
    from mrx.plotting import get_periodic_intersections

    shifted = np.asarray(trajectory, dtype=np.float64).copy()
    shifted[:, 2] = (shifted[:, 2] - float(zeta) + 0.5) % 1.0
    intersections, _, count = get_periodic_intersections(
        jnp.asarray(shifted),
        jnp.zeros(shifted.shape[0]),
        plane_normal=jnp.asarray([0.0, 0.0, 1.0]),
        plane_point=jnp.asarray([0.0, 0.0, 0.5]),
        max_intersections=int(max_intersections),
    )
    count_int = min(int(count), int(max_intersections))
    result = np.asarray(intersections[:count_int], dtype=np.float64).copy()
    if result.size:
        result[:, 2] = float(zeta) % 1.0
    return result.reshape(-1, 3)


def _exact_section_times(zeta: float, turns: int) -> np.ndarray:
    """Return exact positive crossing times for one normalized-zeta section."""
    if turns < 1:
        raise ValueError("turns must be positive")
    section = float(zeta) % 1.0
    if np.isclose(section, 0.0, atol=1.0e-14):
        return np.arange(1, int(turns) + 1, dtype=np.float64)
    return section + np.arange(int(turns), dtype=np.float64)


def _poincare_chunk_size(seed_count: int) -> int:
    """Choose Diffrax batch size for Poincaré integration.

    Chunk so one pathological seed cannot stall the entire batch.  Default to
    small batches for throughput; set ``MRX_POINCARE_CHUNK=1`` for per-seed
    integration when a grid is known to be pathological (e.g. the truncated
    ``12x24x12`` re-trace).
    """
    import os

    count = max(1, int(seed_count))
    env_chunk = os.environ.get("MRX_POINCARE_CHUNK", "").strip()
    if env_chunk:
        return max(1, min(int(env_chunk), count))
    return min(8, count)


def _trace_mrx_poincare(
    seq: Any,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    zeta_values: list[float],
    *,
    nlines: int,
    turns: int,
    theta0: float,
    tol: float,
    logical_sections_out: dict[float, list[np.ndarray]] | None = None,
    seed_rho_values: np.ndarray | None = None,
    theta0_values: np.ndarray | None = None,
    method: str = "RK45",
    audit_out: dict[str, Any] | None = None,
) -> tuple[
    dict[float, list[np.ndarray]],
    np.ndarray,
    np.ndarray,
    list[int],
]:
    """Trace k=2 DBC field lines and return physical ``(R,Z)`` sections.

    Optional explicit radial and poloidal seed arrays are expanded as a
    Cartesian product. This supports phase-resolved island-chain diagnostics
    while preserving the original uniformly radial default.
    """
    from scipy.integrate import solve_ivp

    if nlines < 1 or turns < 1:
        raise ValueError("Poincare tracing requires positive nlines and turns")
    reference_field = jax.jit(
        DiscreteFunction(jnp.asarray(u), seq.basis_2, seq.e2_dbc)
    )
    seed_rho = (
        np.linspace(0.05, 0.95, int(nlines))
        if seed_rho_values is None
        else np.asarray(seed_rho_values, dtype=np.float64).reshape(-1)
    )
    seed_theta = (
        np.asarray([float(theta0) % 1.0], dtype=np.float64)
        if theta0_values is None
        else np.asarray(theta0_values, dtype=np.float64).reshape(-1) % 1.0
    )
    if seed_rho.size < 1 or seed_theta.size < 1:
        raise ValueError("Poincare seed arrays must be nonempty")
    logical_seeds = np.asarray(
        [
            [rho, theta, 0.0]
            for theta in seed_theta
            for rho in seed_rho
        ],
        dtype=np.float64,
    )
    mapped_seeds = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(logical_seeds),
            batch_size=min(64, logical_seeds.shape[0]),
        )
    )
    physical_seeds = np.stack(
        [
            np.sqrt(mapped_seeds[:, 0] ** 2 + mapped_seeds[:, 1] ** 2),
            mapped_seeds[:, 2],
        ],
        axis=1,
    )
    sections: dict[float, list[np.ndarray]] = {
        float(zeta): [] for zeta in zeta_values
    }
    if logical_sections_out is not None:
        logical_sections_out.clear()
        logical_sections_out.update(
            {float(zeta): [] for zeta in zeta_values}
        )
    transit_counts: list[int] = []
    maximum_section_residual = 0.0

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        logical = np.asarray(
            [
                np.clip(state[0], 1.0e-7, 1.0 - 1.0e-7),
                state[1] % 1.0,
                state[2] % 1.0,
            ]
        )
        field = np.asarray(reference_field(jnp.asarray(logical)), dtype=np.float64)
        toroidal = float(field[2])
        if abs(toroidal) < 1.0e-12:
            toroidal = np.copysign(1.0e-12, toroidal if toroidal != 0.0 else 1.0)
        return np.asarray([field[0] / toroidal, field[1] / toroidal, 1.0])

    method_upper = str(method).upper()
    if method_upper == "DIFFRAX_TSIT5":
        if len(zeta_values) != 1:
            raise ValueError(
                "DIFFRAX_TSIT5 currently supports one Poincare section"
            )
        import diffrax

        seed_count = int(logical_seeds.shape[0])
        chunk_size = _poincare_chunk_size(seed_count)
        reference_field_batch = jax.vmap(reference_field)
        zeta = float(zeta_values[0])
        crossing_times = _exact_section_times(zeta, turns)
        crossing_times_jax = jnp.asarray(crossing_times, dtype=jnp.float64)

        def diffrax_rhs(
            _time: jnp.ndarray,
            state: jnp.ndarray,
            _args: Any,
        ) -> jnp.ndarray:
            logical = jnp.stack(
                [
                    jnp.clip(state[:, 0], 1.0e-7, 1.0 - 1.0e-7),
                    jnp.mod(state[:, 1], 1.0),
                    jnp.mod(state[:, 2], 1.0),
                ],
                axis=1,
            )
            fields = reference_field_batch(logical)
            toroidal_raw = fields[:, 2]
            signs = jnp.where(toroidal_raw < 0.0, -1.0, 1.0)
            toroidal = jnp.where(
                jnp.abs(toroidal_raw) < 1.0e-12,
                signs * 1.0e-12,
                toroidal_raw,
            )
            return jnp.stack(
                [
                    fields[:, 0] / toroidal,
                    fields[:, 1] / toroidal,
                    jnp.ones(state.shape[0], dtype=state.dtype),
                ],
                axis=1,
            )

        for chunk_start in range(0, seed_count, chunk_size):
            chunk_end = min(seed_count, chunk_start + chunk_size)
            chunk_seeds = logical_seeds[chunk_start:chunk_end]
            chunk_count = int(chunk_seeds.shape[0])
            print(
                f"MRX_POINCARE_PROGRESS method=DIFFRAX_TSIT5 "
                f"chunk={chunk_start + 1}-{chunk_end}/{seed_count}",
                flush=True,
            )
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(diffrax_rhs),
                diffrax.Tsit5(),
                t0=0.0,
                t1=float(turns),
                dt0=0.02,
                y0=jnp.asarray(chunk_seeds, dtype=jnp.float64),
                saveat=diffrax.SaveAt(ts=crossing_times_jax),
                stepsize_controller=diffrax.PIDController(
                    rtol=float(tol),
                    atol=float(tol),
                ),
                max_steps=max(1_000_000, int(turns) * 10_000),
                throw=True,
            )
            all_hits = np.array(solution.ys, dtype=np.float64, copy=True)
            section_residual = np.abs(
                all_hits[:, :, 2] - crossing_times[:, None]
            )
            if section_residual.size:
                maximum_section_residual = max(
                    maximum_section_residual,
                    float(np.max(section_residual)),
                )
            all_hits[:, :, 1:3] %= 1.0
            all_hits[:, :, 2] = zeta % 1.0
            flat_hits = all_hits.reshape(-1, 3)
            mapped = np.asarray(
                jax.lax.map(
                    map_fn,
                    jnp.asarray(flat_hits),
                    batch_size=min(256, flat_hits.shape[0]),
                )
            ).reshape(crossing_times.size, chunk_count, 3)
            transit_counts.extend([int(turns)] * chunk_count)
            for seed_index in range(chunk_count):
                seed_mapped = mapped[:, seed_index, :]
                sections[zeta].append(
                    np.stack(
                        [
                            np.sqrt(
                                seed_mapped[:, 0] ** 2
                                + seed_mapped[:, 1] ** 2
                            ),
                            seed_mapped[:, 2],
                        ],
                        axis=1,
                    )
                )
                if logical_sections_out is not None:
                    logical_sections_out[zeta].append(
                        np.asarray(
                            all_hits[:, seed_index, :2],
                            dtype=np.float64,
                        )
                    )
    elif method_upper.startswith("BATCHED_"):
        scipy_method = method_upper.removeprefix("BATCHED_")
        if scipy_method not in {"RK23", "RK45", "DOP853"}:
            raise ValueError(f"unsupported batched integration method {method}")
        seed_count = int(logical_seeds.shape[0])
        reference_field_batch = jax.jit(jax.vmap(reference_field))

        def batched_rhs(_time: float, flat_state: np.ndarray) -> np.ndarray:
            state = np.asarray(flat_state, dtype=np.float64).reshape(seed_count, 3)
            logical = np.column_stack(
                [
                    np.clip(state[:, 0], 1.0e-7, 1.0 - 1.0e-7),
                    state[:, 1] % 1.0,
                    state[:, 2] % 1.0,
                ]
            )
            fields = np.asarray(
                reference_field_batch(jnp.asarray(logical)),
                dtype=np.float64,
            )
            toroidal = fields[:, 2]
            signs = np.where(toroidal < 0.0, -1.0, 1.0)
            toroidal = np.where(
                np.abs(toroidal) < 1.0e-12,
                signs * 1.0e-12,
                toroidal,
            )
            derivative = np.column_stack(
                [
                    fields[:, 0] / toroidal,
                    fields[:, 1] / toroidal,
                    np.ones(seed_count, dtype=np.float64),
                ]
            )
            return derivative.ravel()

        solution = solve_ivp(
            batched_rhs,
            (0.0, float(turns)),
            logical_seeds.ravel(),
            rtol=float(tol),
            atol=float(tol),
            dense_output=True,
            method=scipy_method,
        )
        if not solution.success:
            raise RuntimeError(
                f"batched MRX field-line integration failed: {solution.message}"
            )
        if solution.sol is None:
            raise RuntimeError(
                "batched MRX field-line integration did not provide dense output"
            )
        transit_counts.extend([int(turns)] * seed_count)
        for zeta in zeta_values:
            crossing_times = _exact_section_times(zeta, turns)
            all_hits = np.asarray(solution.sol(crossing_times).T).reshape(
                crossing_times.size,
                seed_count,
                3,
            )
            section_residual = np.abs(
                all_hits[:, :, 2] - crossing_times[:, None]
            )
            if section_residual.size:
                maximum_section_residual = max(
                    maximum_section_residual,
                    float(np.max(section_residual)),
                )
            all_hits[:, :, 1:3] %= 1.0
            all_hits[:, :, 2] = float(zeta) % 1.0
            flat_hits = all_hits.reshape(-1, 3)
            mapped = np.asarray(
                jax.lax.map(
                    map_fn,
                    jnp.asarray(flat_hits),
                    batch_size=min(256, flat_hits.shape[0]),
                )
            ).reshape(crossing_times.size, seed_count, 3)
            for seed_index in range(seed_count):
                seed_mapped = mapped[:, seed_index, :]
                sections[float(zeta)].append(
                    np.stack(
                        [
                            np.sqrt(
                                seed_mapped[:, 0] ** 2
                                + seed_mapped[:, 1] ** 2
                            ),
                            seed_mapped[:, 2],
                        ],
                        axis=1,
                    )
                )
                if logical_sections_out is not None:
                    logical_sections_out[float(zeta)].append(
                        np.asarray(
                            all_hits[:, seed_index, :2],
                            dtype=np.float64,
                        )
                    )
    else:
        seed_total = int(logical_seeds.shape[0])
        for seed_index, seed in enumerate(logical_seeds):
            if seed_total >= 8 and (
                seed_index == 0
                or (seed_index + 1) % max(1, seed_total // 8) == 0
                or seed_index + 1 == seed_total
            ):
                print(
                    f"MRX_POINCARE_PROGRESS method={method} "
                    f"line={seed_index + 1}/{seed_total}",
                    flush=True,
                )
            solution = solve_ivp(
                rhs,
                (0.0, float(turns)),
                seed,
                rtol=float(tol),
                atol=float(tol),
                dense_output=True,
                method=str(method),
            )
            if not solution.success:
                raise RuntimeError(
                    f"MRX field-line integration failed: {solution.message}"
                )
            if solution.sol is None:
                raise RuntimeError(
                    "MRX field-line integration did not provide dense output"
                )
            transit_counts.append(int(turns))
            for zeta in zeta_values:
                crossing_times = _exact_section_times(zeta, turns)
                logical_hits = np.asarray(solution.sol(crossing_times).T)
                section_residual = np.abs(
                    logical_hits[:, 2] - crossing_times
                )
                if section_residual.size:
                    maximum_section_residual = max(
                        maximum_section_residual,
                        float(np.max(section_residual)),
                    )
                logical_hits[:, 1:3] %= 1.0
                logical_hits[:, 2] = float(zeta) % 1.0
                if logical_hits.size:
                    mapped = np.asarray(
                        jax.lax.map(
                            map_fn,
                            jnp.asarray(logical_hits),
                            batch_size=min(128, logical_hits.shape[0]),
                        )
                    )
                    physical = np.stack(
                        [
                            np.sqrt(mapped[:, 0] ** 2 + mapped[:, 1] ** 2),
                            mapped[:, 2],
                        ],
                        axis=1,
                    )
                else:
                    physical = np.empty((0, 2), dtype=np.float64)
                sections[float(zeta)].append(physical)
                if logical_sections_out is not None:
                    logical_sections_out[float(zeta)].append(
                        np.asarray(logical_hits[:, :2], dtype=np.float64)
                    )
    if audit_out is not None:
        audit_out.clear()
        audit_out.update(
            {
                "method": str(method),
                "rtol": float(tol),
                "atol": float(tol),
                "section_crossing_construction": (
                    "exact t_eval because the reparameterized ODE enforces dzeta/dt=1"
                ),
                "max_unwrapped_section_coordinate_residual": float(
                    maximum_section_residual
                ),
                "seed_rho_count": int(seed_rho.size),
                "seed_theta_count": int(seed_theta.size),
                "total_lines": int(logical_seeds.shape[0]),
            }
        )
    return sections, logical_seeds, physical_seeds, transit_counts


def _build_interpolated_simsopt_field(
    simsopt_field: Any,
    boundary_surface: Any,
    *,
    nfp: int,
    interpolation_degree: int,
    interpolation_points: int,
) -> tuple[Any, Any]:
    """Build the SIMSOPT interpolation and boundary classifier used for tracing."""
    from simsopt.field import InterpolatedField, SurfaceClassifier

    gamma = np.asarray(boundary_surface.gamma(), dtype=np.float64)
    surface_R = np.linalg.norm(gamma[..., :2], axis=-1)
    surface_Z = gamma[..., 2]
    grid_n = int(interpolation_points)
    if grid_n < 4:
        raise ValueError("SIMSOPT interpolation requires at least 4 grid cells")
    classifier = SurfaceClassifier(boundary_surface, h=0.03, p=2)

    def skip(rs: np.ndarray, phis: np.ndarray, zs: np.ndarray) -> list[bool]:
        """Skip interpolation cells safely outside the boundary surface."""
        rphiz = np.asarray([rs, phis, zs], dtype=np.float64).T.copy()
        return list((classifier.evaluate_rphiz(rphiz) < -0.05).ravel())

    interpolated_field = InterpolatedField(
        simsopt_field,
        int(interpolation_degree),
        (float(np.min(surface_R)), float(np.max(surface_R)), grid_n),
        (0.0, 2.0 * np.pi / int(nfp), 2 * grid_n),
        (0.0, float(np.max(np.abs(surface_Z))), max(2, grid_n // 2)),
        True,
        nfp=int(nfp),
        stellsym=bool(getattr(boundary_surface, "stellsym", True)),
        skip=skip,
    )
    return interpolated_field, classifier


def _biot_savart_validation(
    simsopt_field: Any,
    boundary_surface: Any,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    nfp: int,
    interpolation_degree: int,
    interpolation_points: int,
) -> dict[str, Any]:
    """Validate coil quadrature and tracing interpolation near the ι=1/2 chain."""
    rho = np.linspace(0.58, 0.82, 4)
    theta = np.arange(6, dtype=np.float64) / 6.0
    zeta = np.arange(6, dtype=np.float64) / 6.0
    rho_grid, theta_grid, zeta_grid = np.meshgrid(
        rho,
        theta,
        zeta,
        indexing="ij",
    )
    logical = np.stack(
        [rho_grid.ravel(), theta_grid.ravel(), zeta_grid.ravel()],
        axis=1,
    )
    points = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(logical, dtype=jnp.float64),
            batch_size=128,
        ),
        dtype=np.float64,
    )
    direct = _evaluate_simsopt_field(simsopt_field, points)
    direct_rms = float(np.sqrt(np.mean(np.sum(direct**2, axis=1))))
    coils = list(getattr(simsopt_field, "_coils"))
    independent_base = _evaluate_biot_savart_periodic_quadrature(
        coils,
        points,
        quadrature_factor=1,
    )
    independent_refined = _evaluate_biot_savart_periodic_quadrature(
        coils,
        points,
        quadrature_factor=2,
    )
    interpolated, _ = _build_interpolated_simsopt_field(
        simsopt_field,
        boundary_surface,
        nfp=int(nfp),
        interpolation_degree=int(interpolation_degree),
        interpolation_points=int(interpolation_points),
    )
    interpolated_values = _evaluate_simsopt_field(interpolated, points)

    def comparison(candidate: np.ndarray) -> dict[str, float]:
        """Return absolute and relative vector errors against direct Biot–Savart."""
        difference = np.asarray(candidate) - direct
        pointwise = np.linalg.norm(difference, axis=1)
        return {
            "relative_vector_rms": float(
                np.sqrt(np.mean(pointwise**2)) / direct_rms
            ),
            "relative_vector_max": float(np.max(pointwise) / direct_rms),
        }

    refinement_difference = independent_refined - independent_base
    refinement_pointwise = np.linalg.norm(refinement_difference, axis=1)
    return {
        "samples_near_half_iota_chain": int(points.shape[0]),
        "direct_field_rms_tesla": direct_rms,
        "original_coil_quadrature_points": sorted(
            {int(len(coil.curve.quadpoints)) for coil in coils}
        ),
        "refined_quadrature_factor": 2,
        "direct_vs_independent_original_quadrature": comparison(independent_base),
        "direct_vs_independent_refined_quadrature": comparison(independent_refined),
        "quadrature_refinement_change": {
            "relative_vector_rms": float(
                np.sqrt(np.mean(refinement_pointwise**2)) / direct_rms
            ),
            "relative_vector_max": float(
                np.max(refinement_pointwise) / direct_rms
            ),
        },
        "interpolated_vs_direct": comparison(interpolated_values),
        "interpolation_degree": int(interpolation_degree),
        "interpolation_points": int(interpolation_points),
    }


def _trace_simsopt_poincare(
    simsopt_field: Any,
    boundary_surface: Any,
    physical_seeds: np.ndarray,
    zeta_values: list[float],
    *,
    nfp: int,
    tol: float,
    tmax: float,
    interpolation_degree: int,
    interpolation_points: int,
) -> tuple[
    dict[float, list[np.ndarray]],
    list[np.ndarray],
    list[float],
    list[bool],
    float,
]:
    """Trace an interpolated SIMSOPT field from matched ``(R,Z)`` seeds."""
    from simsopt.field import LevelsetStoppingCriterion
    from simsopt.field.tracing import compute_fieldlines

    R0 = np.asarray(physical_seeds[:, 0], dtype=np.float64)
    Z0 = np.asarray(physical_seeds[:, 1], dtype=np.float64)
    interpolated_field, classifier = _build_interpolated_simsopt_field(
        simsopt_field,
        nfp=int(nfp),
        boundary_surface=boundary_surface,
        interpolation_degree=int(interpolation_degree),
        interpolation_points=int(interpolation_points),
    )
    phis = [2.0 * np.pi * (float(zeta) % 1.0) / int(nfp) for zeta in zeta_values]
    start = time.perf_counter()
    _, phi_hits = compute_fieldlines(
        interpolated_field,
        R0,
        Z0,
        tmax=float(tmax),
        tol=float(tol),
        phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)],
    )
    trace_seconds = time.perf_counter() - start
    sections: dict[float, list[np.ndarray]] = {
        float(zeta): [] for zeta in zeta_values
    }
    lost: list[bool] = []
    for line_hits in phi_hits:
        hits = np.asarray(line_hits, dtype=np.float64)
        lost.append(bool(hits.size and hits[-1, 1] < 0.0))
    for plane_index, zeta in enumerate(zeta_values):
        for line_hits in phi_hits:
            hits = np.asarray(line_hits, dtype=np.float64)
            if hits.size == 0:
                sections[float(zeta)].append(np.empty((0, 2)))
                continue
            selected = hits[hits[:, 1] == plane_index]
            sections[float(zeta)].append(
                np.stack(
                    [
                        np.sqrt(selected[:, 2] ** 2 + selected[:, 3] ** 2),
                        selected[:, 4],
                    ],
                    axis=1,
                )
                if selected.size
                else np.empty((0, 2))
            )
    return sections, phi_hits, phis, lost, float(trace_seconds)


def _rotational_transform_profile(
    section_lines: list[np.ndarray],
    seed_rho: np.ndarray,
    *,
    nfp: int,
    theta_column: int = 1,
) -> list[dict[str, float]]:
    """Estimate standard rotational transform from ordered section crossings."""
    if len(section_lines) != int(np.asarray(seed_rho).size):
        raise ValueError("section line and seed-rho counts must match")
    profile: list[dict[str, float]] = []
    for rho, line in zip(np.asarray(seed_rho, dtype=np.float64), section_lines):
        points = np.asarray(line, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        theta = np.unwrap(2.0 * np.pi * points[:, int(theta_column)])
        crossings = np.arange(points.shape[0], dtype=np.float64)
        slope, intercept = np.polyfit(crossings, theta / (2.0 * np.pi), 1)
        fitted = slope * crossings + intercept
        profile.append(
            {
                "rho": float(rho),
                "iota": float(int(nfp) * slope),
                "iota_per_field_period": float(slope),
                "fit_rms": float(
                    np.sqrt(np.mean((theta / (2.0 * np.pi) - fitted) ** 2))
                ),
                "crossings": int(points.shape[0]),
            }
        )
    return profile


def _physical_rotational_transform_profile(
    section_lines: list[np.ndarray],
    seed_rho: np.ndarray,
    axis_rz: np.ndarray,
    *,
    nfp: int,
    orientation_sign: float = -1.0,
    logical_slice: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Estimate iota from full-torus physical section points about the axis.

    SIMSOPT returns successive hits of each physical plane after a full
    toroidal transit, unlike MRX's normalized field-period coordinate. The
    default sign maps the geometric ``atan2(Z, R-R_axis)`` orientation to the
    logical poloidal-angle convention.
    """
    angular_lines: list[np.ndarray] = []
    if logical_slice is not None:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        coordinates = np.column_stack(
            [
                np.asarray(logical_slice["R"]).ravel(),
                np.asarray(logical_slice["Z"]).ravel(),
            ]
        )
        theta = np.asarray(logical_slice["theta"]).ravel()
        sin_theta = np.sin(2.0 * np.pi * theta)
        cos_theta = np.cos(2.0 * np.pi * theta)
        sin_linear = LinearNDInterpolator(coordinates, sin_theta)
        cos_linear = LinearNDInterpolator(coordinates, cos_theta)
        sin_nearest = NearestNDInterpolator(coordinates, sin_theta)
        cos_nearest = NearestNDInterpolator(coordinates, cos_theta)
        for line in section_lines:
            points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
            sin_values = np.asarray(sin_linear(points), dtype=np.float64)
            cos_values = np.asarray(cos_linear(points), dtype=np.float64)
            invalid = ~np.isfinite(sin_values) | ~np.isfinite(cos_values)
            if np.any(invalid):
                sin_values[invalid] = sin_nearest(points[invalid])
                cos_values[invalid] = cos_nearest(points[invalid])
            theta_cycles = (
                np.arctan2(sin_values, cos_values) / (2.0 * np.pi)
            ) % 1.0
            angular_lines.append(
                np.column_stack([np.zeros(points.shape[0]), theta_cycles])
            )
        orientation_sign = 1.0
    else:
        center = np.asarray(axis_rz, dtype=np.float64).reshape(2)
        for line in section_lines:
            points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
            angle_cycles = np.arctan2(
                points[:, 1] - center[1],
                points[:, 0] - center[0],
            ) / (2.0 * np.pi)
            angular_lines.append(
                np.column_stack([np.zeros(points.shape[0]), angle_cycles])
            )
    profile = _rotational_transform_profile(
        angular_lines,
        seed_rho,
        nfp=1,
    )
    for item in profile:
        item["iota"] *= float(orientation_sign)
        item["iota_per_field_period"] = item["iota"] / int(nfp)
    return profile


def _physical_sections_to_logical(
    section_lines: list[np.ndarray],
    logical_slice: dict[str, Any],
) -> list[np.ndarray]:
    """Interpolate physical ``(R,Z)`` section hits to logical ``(rho, theta)``.

    Circular interpolation of the poloidal angle avoids a discontinuity at the
    periodic theta seam. Linear interpolation is used inside the sampled
    poloidal slice, with nearest-neighbor fallback only for points that lie
    marginally outside its triangulation.

    Parameters
    ----------
    section_lines:
        One ``(n_hits, 2)`` physical ``(R,Z)`` array per traced field line.
    logical_slice:
        Fixed-zeta mapping sample containing ``R``, ``Z``, ``rho``, and
        periodic ``theta`` arrays of identical shape.

    Returns
    -------
    list[numpy.ndarray]
        Logical section arrays with columns ``rho`` and ``theta`` in cycles.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    coordinates = np.column_stack(
        [
            np.asarray(logical_slice["R"], dtype=np.float64).ravel(),
            np.asarray(logical_slice["Z"], dtype=np.float64).ravel(),
        ]
    )
    rho = np.asarray(logical_slice["rho"], dtype=np.float64).ravel()
    theta = np.asarray(logical_slice["theta"], dtype=np.float64).ravel()
    if coordinates.shape[0] != rho.size or rho.size != theta.size:
        raise ValueError("logical slice coordinate arrays must have equal sizes")

    values = {
        "rho": rho,
        "sin_theta": np.sin(2.0 * np.pi * theta),
        "cos_theta": np.cos(2.0 * np.pi * theta),
    }
    linear = {
        key: LinearNDInterpolator(coordinates, value)
        for key, value in values.items()
    }
    nearest = {
        key: NearestNDInterpolator(coordinates, value)
        for key, value in values.items()
    }
    logical_lines: list[np.ndarray] = []
    for line in section_lines:
        points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        interpolated: dict[str, np.ndarray] = {}
        for key in values:
            result = np.asarray(linear[key](points), dtype=np.float64)
            invalid = ~np.isfinite(result)
            if np.any(invalid):
                result[invalid] = nearest[key](points[invalid])
            interpolated[key] = result
        theta_cycles = (
            np.arctan2(
                interpolated["sin_theta"],
                interpolated["cos_theta"],
            )
            / (2.0 * np.pi)
        ) % 1.0
        logical_lines.append(
            np.column_stack([interpolated["rho"], theta_cycles])
        )
    return logical_lines


def _periodic_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest signed difference on the unit circle, in cycles."""
    return (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5


def _batched_map_fn(
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    logical: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Evaluate a geometry map with one bounded-size JAX compilation."""
    points = np.asarray(logical, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    mapped = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(points, dtype=jnp.float64),
            batch_size=int(batch_size),
        ),
        dtype=np.float64,
    )
    return mapped


def _physical_phi_zero_slice(
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    nrho: int = 48,
    ntheta: int = 96,
) -> dict[str, Any]:
    """Parameterize the cylindrical ``phi=0`` plane in logical coordinates.

    Logical ``zeta=0`` is generally *not* the physical cylindrical plane
    ``phi=0``.  This Newton correction adjusts ``zeta(rho, theta)`` so that
    mapped points satisfy ``atan2(y, x) ≈ 0``, enabling a fair physical
    overlay of SIMSOPT and MRX Poincaré hits.
    """
    logical, rho_grid, theta_grid = _poloidal_slice_points(
        0.0, int(nrho), int(ntheta)
    )
    logical = np.asarray(logical, dtype=np.float64)
    zeta = np.zeros(logical.shape[0], dtype=np.float64)
    epsilon = 1.0e-5
    for _ in range(4):
        points = logical.copy()
        points[:, 2] = zeta
        xyz = _batched_map_fn(map_fn, points)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        shifted = points.copy()
        shifted[:, 2] += epsilon
        xyz_shifted = _batched_map_fn(map_fn, shifted)
        phi_shifted = np.arctan2(xyz_shifted[:, 1], xyz_shifted[:, 0])
        derivative = _periodic_difference(
            phi_shifted / (2.0 * np.pi), phi / (2.0 * np.pi)
        ) * (2.0 * np.pi / epsilon)
        zeta -= phi / derivative
    points = logical.copy()
    points[:, 2] = zeta
    mapped = _batched_map_fn(map_fn, points).reshape(int(nrho), int(ntheta), 3)
    return {
        "zeta": zeta.reshape(int(nrho), int(ntheta)),
        "rho": rho_grid,
        "theta": theta_grid,
        "R": np.linalg.norm(mapped[..., :2], axis=-1),
        "Z": mapped[..., 2],
        "phi": np.arctan2(mapped[..., 1], mapped[..., 0]),
    }


def _logical_to_phi_zero_rz(
    lines: list[np.ndarray],
    physical_slice: dict[str, Any],
) -> list[np.ndarray]:
    """Embed logical ``(rho, theta)`` hits into a common physical ``phi=0`` plane."""
    from scipy.interpolate import RegularGridInterpolator

    rho_axis = np.asarray(physical_slice["rho"])[:, 0]
    theta_axis = np.asarray(physical_slice["theta"])[0, :]
    theta_periodic = np.concatenate([theta_axis, [1.0]])
    interpolators = []
    for key in ("R", "Z"):
        values = np.asarray(physical_slice[key])
        periodic_values = np.concatenate([values, values[:, :1]], axis=1)
        interpolators.append(
            RegularGridInterpolator(
                (rho_axis, theta_periodic),
                periodic_values,
                bounds_error=False,
                fill_value=None,
            )
        )
    physical: list[np.ndarray] = []
    for line in lines:
        points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        query = np.column_stack([points[:, 0], points[:, 1] % 1.0])
        physical.append(
            np.column_stack([interpolator(query) for interpolator in interpolators])
        )
    return physical


def _iota_profile_from_physical_pitch(
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    physical_field: np.ndarray,
    seed_rho: np.ndarray,
    *,
    nfp: int,
    ntheta: int,
    nzeta: int,
) -> list[dict[str, float]]:
    """Estimate iota from flux-surface-averaged logical poloidal pitch.

    ``physical_field`` must already be evaluated on the stacked
    ``(n_rho, n_theta, n_zeta)`` logical grid used below.
    """
    theta = np.arange(int(ntheta), dtype=np.float64) / int(ntheta)
    zeta = np.arange(int(nzeta), dtype=np.float64) / int(nzeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    samples_per_surface = int(theta_grid.size)
    rho_values = np.asarray(seed_rho, dtype=np.float64).reshape(-1)
    field = np.asarray(physical_field, dtype=np.float64).reshape(
        rho_values.size * samples_per_surface, 3
    )
    jacobian_fn = jax.jit(jax.jacfwd(map_fn))
    profile: list[dict[str, float]] = []
    for index, rho in enumerate(rho_values):
        logical = np.stack(
            [
                np.full(samples_per_surface, float(rho)),
                theta_grid.ravel(),
                zeta_grid.ravel(),
            ],
            axis=1,
        )
        logical_jax = jnp.asarray(logical, dtype=jnp.float64)
        jacobians = np.asarray(
            jax.lax.map(jacobian_fn, logical_jax, batch_size=128),
            dtype=np.float64,
        )
        surface_field = field[
            index * samples_per_surface : (index + 1) * samples_per_surface
        ]
        logical_field = np.linalg.solve(
            jacobians,
            surface_field[..., None],
        )[..., 0]
        valid = np.abs(logical_field[:, 2]) > 1.0e-12
        pitch = logical_field[valid, 1] / logical_field[valid, 2]
        profile.append(
            {
                "rho": float(rho),
                "iota": float(int(nfp) * np.mean(pitch)),
                "iota_per_field_period": float(np.mean(pitch)),
                "pitch_std": float(np.std(pitch)),
                "samples": int(pitch.size),
            }
        )
    return profile


def _simsopt_iota_profile_from_pitch(
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    simsopt_field: Any,
    seed_rho: np.ndarray,
    *,
    nfp: int,
    ntheta: int = 16,
    nzeta: int = 16,
) -> list[dict[str, float]]:
    """Estimate SIMSOPT iota from flux-surface-averaged logical field pitch."""
    theta = np.arange(int(ntheta), dtype=np.float64) / int(ntheta)
    zeta = np.arange(int(nzeta), dtype=np.float64) / int(nzeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    rho_values = np.asarray(seed_rho, dtype=np.float64).reshape(-1)
    logical = np.stack(
        [
            np.repeat(rho_values, theta_grid.size),
            np.tile(theta_grid.ravel(), rho_values.size),
            np.tile(zeta_grid.ravel(), rho_values.size),
        ],
        axis=1,
    )
    mapped = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(logical, dtype=jnp.float64),
            batch_size=256,
        ),
        dtype=np.float64,
    )
    physical_field = _evaluate_simsopt_field(simsopt_field, mapped)
    return _iota_profile_from_physical_pitch(
        map_fn,
        physical_field,
        rho_values,
        nfp=int(nfp),
        ntheta=int(ntheta),
        nzeta=int(nzeta),
    )


def _mrx_iota_profile_from_pitch(
    seq: Any,
    dof: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    seed_rho: np.ndarray,
    *,
    nfp: int,
    ntheta: int = 32,
    nzeta: int = 32,
) -> list[dict[str, float]]:
    """Estimate MRX iota from flux-surface-averaged pushforward pitch."""
    theta = np.arange(int(ntheta), dtype=np.float64) / int(ntheta)
    zeta = np.arange(int(nzeta), dtype=np.float64) / int(nzeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    rho_values = np.asarray(seed_rho, dtype=np.float64).reshape(-1)
    logical = np.stack(
        [
            np.repeat(rho_values, theta_grid.size),
            np.tile(theta_grid.ravel(), rho_values.size),
            np.tile(zeta_grid.ravel(), rho_values.size),
        ],
        axis=1,
    )
    physical_field = _evaluate_pushforward(
        seq,
        dof,
        map_fn,
        jnp.asarray(logical, dtype=jnp.float64),
        k=2,
    )
    return _iota_profile_from_physical_pitch(
        map_fn,
        physical_field,
        rho_values,
        nfp=int(nfp),
        ntheta=int(ntheta),
        nzeta=int(nzeta),
    )


def _local_shear_from_iota_profile(
    profile: list[dict[str, float]],
    resonance_rho: float,
    *,
    half_window: int = 4,
) -> float:
    """Central finite-difference / local linear fit for dι/dρ."""
    ordered = sorted(profile, key=lambda item: float(item["rho"]))
    rho = np.asarray([item["rho"] for item in ordered], dtype=np.float64)
    iota = np.asarray([item["iota"] for item in ordered], dtype=np.float64)
    nearest = int(np.argmin(np.abs(rho - float(resonance_rho))))
    lo = max(0, nearest - int(half_window))
    hi = min(rho.size, nearest + int(half_window) + 1)
    if hi - lo < 3:
        raise RuntimeError("insufficient iota samples for local shear")
    return float(np.polyfit(rho[lo:hi], iota[lo:hi], 1)[0])


def pendulum_island_width(
    resonant_pitch_amplitude: float,
    shear: float,
) -> float:
    """Full radial island width from the resonant pitch harmonic.

    With phase ``ψ = 2π(m θ - k ζ)`` on the field-period coordinate and ``ι``
    reported per full toroidal turn, the pendulum separatrix width is

        W = 4 sqrt(|c| / (2π |dι/dρ|)),

    where ``c`` is the complex Fourier coefficient of ``B^ρ/B^ζ``. Using the
    poloidal mode number ``m=6`` in place of ``2π`` is accidentally correct to
    about 2.3 percent because the cycle-to-radian factor, the real-amplitude
    convention ``A = 2|c|``, and the field-period resonance index nearly cancel.
    """
    return float(
        4.0
        * np.sqrt(
            max(float(resonant_pitch_amplitude), 0.0)
            / (2.0 * np.pi * max(abs(float(shear)), 1.0e-14))
        )
    )


def _identify_iota_resonances(
    profile: list[dict[str, float]],
    *,
    nfp: int,
    rho_min: float = 0.3,
    rho_max: float = 0.6,
    max_poloidal_mode: int = 16,
    max_iota_mismatch: float = 0.02,
) -> list[dict[str, float | int]]:
    """Locate stellarator-symmetric low-order rational iota surfaces."""
    selected = sorted(
        (
            (float(item["rho"]), float(item["iota"]))
            for item in profile
            if rho_min <= float(item["rho"]) <= rho_max
        ),
        key=lambda item: item[0],
    )
    if len(selected) < 2:
        return []
    iota_values = np.asarray([item[1] for item in selected])
    iota_lo = float(np.min(iota_values))
    iota_hi = float(np.max(iota_values))
    sign = -1 if 0.5 * (iota_lo + iota_hi) < 0.0 else 1
    candidates: dict[float, tuple[int, int]] = {}
    for poloidal_mode in range(1, int(max_poloidal_mode) + 1):
        for field_period_mode in range(1, int(max_poloidal_mode) + 1):
            toroidal_mode = sign * int(nfp) * field_period_mode
            rational = toroidal_mode / poloidal_mode
            if (
                iota_lo - float(max_iota_mismatch)
                <= rational
                <= iota_hi + float(max_iota_mismatch)
            ):
                key = round(float(rational), 14)
                candidate = (toroidal_mode, poloidal_mode)
                previous = candidates.get(key)
                if previous is None or (
                    abs(candidate[0]) + candidate[1]
                    < abs(previous[0]) + previous[1]
                ):
                    candidates[key] = candidate

    resonances: list[dict[str, float | int]] = []
    for toroidal_mode, poloidal_mode in sorted(
        candidates.values(),
        key=lambda mode: (abs(mode[0]) + mode[1], mode[1]),
    ):
        rational = toroidal_mode / poloidal_mode
        best_rho = selected[int(np.argmin(np.abs(iota_values - rational)))][0]
        mismatch = float(np.min(np.abs(iota_values - rational)))
        if mismatch > float(max_iota_mismatch):
            continue
        for (rho0, iota0), (rho1, iota1) in zip(selected[:-1], selected[1:]):
            if (iota0 - rational) * (iota1 - rational) <= 0.0 and iota1 != iota0:
                fraction = (rational - iota0) / (iota1 - iota0)
                best_rho = rho0 + fraction * (rho1 - rho0)
                mismatch = 0.0
                break
        resonances.append(
            {
                "rho": float(best_rho),
                "iota": float(rational),
                "toroidal_mode": int(toroidal_mode),
                "poloidal_mode": int(poloidal_mode),
                "field_period_mode": int(toroidal_mode // int(nfp)),
                "sample_mismatch": mismatch,
            }
        )
    return resonances


def _island_width_profile(
    logical_section_lines: list[np.ndarray],
    seed_rho: np.ndarray,
    *,
    max_background_mode: int = 4,
) -> list[dict[str, float]]:
    """Measure raw and Fourier-detrended radial Poincare widths.

    The detrended metric removes the constant through
    ``max_background_mode`` poloidal harmonics from ``rho(theta)``. It avoids
    treating smooth surface-label variation as a six-lobed island width.
    """
    if len(logical_section_lines) != int(np.asarray(seed_rho).size):
        raise ValueError("section line and seed-rho counts must match")
    widths: list[dict[str, float]] = []
    for seed, line in zip(
        np.asarray(seed_rho, dtype=np.float64),
        logical_section_lines,
    ):
        points = np.asarray(line, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 4:
            continue
        radial = points[:, 0]
        theta = points[:, 1] % 1.0
        design_columns = [np.ones(theta.size)]
        for mode in range(1, int(max_background_mode) + 1):
            design_columns.extend(
                [
                    np.cos(2.0 * np.pi * mode * theta),
                    np.sin(2.0 * np.pi * mode * theta),
                ]
            )
        design = np.column_stack(design_columns)
        coefficients, _, _, _ = np.linalg.lstsq(design, radial, rcond=None)
        background = design @ coefficients
        residual = radial - background
        lower, median, upper = np.quantile(radial, [0.05, 0.5, 0.95])
        residual_lower, residual_upper = np.quantile(residual, [0.05, 0.95])
        widths.append(
            {
                "seed_rho": float(seed),
                "center_rho": float(median),
                "width_rho_q05_q95": float(upper - lower),
                "radial_std": float(np.std(radial)),
                "background_max_poloidal_mode": int(max_background_mode),
                "detrended_width_rho_q05_q95": float(
                    residual_upper - residual_lower
                ),
                "detrended_radial_std": float(np.std(residual)),
                "background_rms": float(np.sqrt(np.mean(background**2))),
            }
        )
    return widths


def _classify_resonant_orbit(
    logical_section_line: np.ndarray,
    *,
    poloidal_mode: int = 6,
    concentration_threshold: float = 0.45,
) -> dict[str, float | bool | int]:
    """Classify a fixed-section orbit as trapped from its resonant phase.

    At the ι=1/2 resonance in an ``nfp=3`` configuration, the six-island
    resonant phase at a fixed toroidal section is ``6 theta``. Libration
    produces a concentrated circular phase distribution, whereas a passing
    orbit circulates around the unit circle.
    """
    points = np.asarray(logical_section_line, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 4:
        return {
            "trapped": False,
            "phase_concentration": float("nan"),
            "phase_span_cycles_q05_q95": float("nan"),
            "crossings": int(points.shape[0]) if points.ndim == 2 else 0,
        }
    phase = 2.0 * np.pi * int(poloidal_mode) * (points[:, 1] % 1.0)
    mean_phase = np.angle(np.mean(np.exp(1.0j * phase)))
    centered_cycles = np.angle(np.exp(1.0j * (phase - mean_phase))) / (
        2.0 * np.pi
    )
    lower, upper = np.quantile(centered_cycles, [0.05, 0.95])
    concentration = float(abs(np.mean(np.exp(1.0j * phase))))
    return {
        "trapped": bool(concentration >= float(concentration_threshold)),
        "phase_concentration": concentration,
        "phase_span_cycles_q05_q95": float(upper - lower),
        "crossings": int(points.shape[0]),
    }


def _trapped_separatrix_summary(
    logical_section_lines: list[np.ndarray],
    logical_seeds: np.ndarray,
    *,
    poloidal_mode: int = 6,
) -> dict[str, Any]:
    """Estimate chain extent from phase-resolved trapped/passing orbits."""
    seeds = np.asarray(logical_seeds, dtype=np.float64).reshape(-1, 3)
    if len(logical_section_lines) != seeds.shape[0]:
        raise ValueError("zoomed line and seed counts must match")
    classifications: list[dict[str, Any]] = []
    trapped_radial: list[np.ndarray] = []
    for seed, line in zip(seeds, logical_section_lines):
        classification = _classify_resonant_orbit(
            line,
            poloidal_mode=int(poloidal_mode),
        )
        classifications.append(
            {
                "seed_rho": float(seed[0]),
                "seed_theta": float(seed[1]),
                **classification,
            }
        )
        if bool(classification["trapped"]):
            trapped_radial.append(np.asarray(line, dtype=np.float64)[:, 0])
    if trapped_radial:
        radial = np.concatenate(trapped_radial)
        lower, upper = np.quantile(radial, [0.05, 0.95])
        extent = float(upper - lower)
    else:
        lower = upper = extent = float("nan")
    return {
        "poloidal_mode": int(poloidal_mode),
        "trapped_line_count": int(
            sum(bool(item["trapped"]) for item in classifications)
        ),
        "total_line_count": int(len(classifications)),
        "trapped_radial_q05": float(lower),
        "trapped_radial_q95": float(upper),
        "trapped_separatrix_width_rho": float(extent),
        "classifications": classifications,
    }


def _plot_island_zoom(
    logical_section_lines: list[np.ndarray],
    logical_seeds: np.ndarray,
    output_dir: Path,
    *,
    label: str,
    rho_min: float,
    rho_max: float,
    dpi: int = 150,
) -> Path:
    """Plot a phase-resolved logical Poincare zoom around the ι=1/2 chain."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.asarray(logical_seeds, dtype=np.float64).reshape(-1, 3)
    if len(logical_section_lines) != seeds.shape[0]:
        raise ValueError("zoomed line and seed counts must match")
    fig, axis = plt.subplots(figsize=(8.0, 5.2))
    phases = np.unique(seeds[:, 1])
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, phases.size))
    for line, seed in zip(logical_section_lines, seeds):
        points = np.asarray(line, dtype=np.float64)
        phase_index = int(np.argmin(np.abs(phases - seed[1])))
        classification = _classify_resonant_orbit(points)
        axis.scatter(
            points[:, 1] % 1.0,
            points[:, 0],
            s=2.2 if bool(classification["trapped"]) else 1.4,
            alpha=0.75 if bool(classification["trapped"]) else 0.45,
            color=colors[phase_index],
            rasterized=True,
        )
    for phase, color in zip(phases, colors):
        axis.plot([], [], ".", color=color, label=rf"seed $\theta_0={phase:.3f}$")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(float(rho_min), float(rho_max))
    axis.set_xlabel(r"Logical poloidal angle $\theta$ [cycles]")
    axis.set_ylabel(r"Logical radius $\rho$")
    axis.set_title(rf"$\iota=1/2$ island-chain zoom, MRX {label}")
    axis.grid(True, alpha=0.2)
    axis.legend(fontsize="small", ncol=min(3, phases.size))
    fig.tight_layout()
    path = (output_dir / f"poincare_island_zoom_fem_{label}.png").resolve()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_reference_island_zooms(
    simsopt_lines: list[np.ndarray],
    mrx_lines: list[np.ndarray],
    logical_seeds: np.ndarray,
    output_dir: Path,
    *,
    rho_min: float,
    rho_max: float,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Plot targeted SIMSOPT and matched MRX/SIMSOPT logical island zooms.

    Both panels use identical logical axes so the physical reference chain can
    be compared without the scale compression of a full-domain ``(R,Z)`` plot.
    Lines are colored by their resonant seed phase.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    seeds = np.asarray(logical_seeds, dtype=np.float64).reshape(-1, 3)
    if len(simsopt_lines) != seeds.shape[0] or len(mrx_lines) != seeds.shape[0]:
        raise ValueError("reference zoom lines and logical seeds must match")
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = np.unique(seeds[:, 1])
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, phases.size))

    def draw(axis: Any, lines: list[np.ndarray], title: str) -> None:
        """Draw one logical Poincare zoom on a supplied axis."""
        for line, seed in zip(lines, seeds):
            points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
            phase_index = int(np.argmin(np.abs(phases - seed[1])))
            axis.scatter(
                points[:, 1] % 1.0,
                points[:, 0],
                s=0.6,
                alpha=0.65,
                color=colors[phase_index],
                rasterized=True,
            )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(float(rho_min), float(rho_max))
        axis.set_xlabel(r"Logical poloidal angle $\theta$ [cycles]")
        axis.set_ylabel(r"Logical radius $\rho$")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)

    reference_figure, reference_axis = plt.subplots(figsize=(8.0, 5.2))
    draw(reference_axis, simsopt_lines, r"$\iota=1/2$ island-chain zoom, SIMSOPT")
    for phase, color in zip(phases, colors):
        reference_axis.plot(
            [],
            [],
            ".",
            color=color,
            label=rf"seed $\theta_0={phase:.3f}$",
        )
    reference_axis.legend(fontsize="small", ncol=min(3, phases.size))
    reference_figure.tight_layout()
    reference_path = (output_dir / "poincare_island_zoom_simsopt.png").resolve()
    reference_figure.savefig(reference_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(reference_figure)

    comparison_figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    draw(axes[0, 0], simsopt_lines, "SIMSOPT Biot–Savart")
    draw(axes[0, 1], mrx_lines, "MRX finest saved field")
    comparison_figure.suptitle(r"Matched logical $\iota=1/2$ island zoom")
    comparison_figure.tight_layout()
    comparison_path = (
        output_dir / "poincare_island_zoom_simsopt_vs_mrx.png"
    ).resolve()
    comparison_figure.savefig(
        comparison_path,
        dpi=int(dpi),
        bbox_inches="tight",
    )
    plt.close(comparison_figure)
    return reference_path, comparison_path


def _simsopt_island_baseline(
    seq: Any,
    dof: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    simsopt_field: Any,
    boundary_surface: Any,
    logical_slice: dict[str, Any],
    output_dir: Path,
    *,
    nfp: int,
    rho_min: float,
    rho_max: float,
    nrho: int,
    phases: int,
    theta0: float,
    mrx_turns: int,
    mrx_tol: float,
    simsopt_tmax: float,
    simsopt_tol: float,
    interpolation_degree: int,
    interpolation_points: int,
    minimum_intersections: int = 100,
    cache_path: Path | None = None,
    mrx_method: str = "RK45",
) -> dict[str, Any]:
    """Trace and quantify a moderate-cost physical SIMSOPT island baseline.

    A Cartesian product of radial and resonant-phase seeds is mapped to
    physical coordinates, traced with SIMSOPT, transformed back to the logical
    section, and compared with an identically seeded MRX trace. Lines below the
    requested intersection threshold remain visible but are excluded from the
    reference-width estimate.
    """
    if nrho < 2 or phases < 1:
        raise ValueError("SIMSOPT island zoom requires nrho >= 2 and phases >= 1")
    seed_rho = np.linspace(float(rho_min), float(rho_max), int(nrho))
    seed_theta = (
        float(theta0)
        + np.arange(int(phases), dtype=np.float64)
        / (6.0 * int(phases))
    ) % 1.0
    logical_seeds = np.asarray(
        [
            [rho, theta, 0.0]
            for theta in seed_theta
            for rho in seed_rho
        ],
        dtype=np.float64,
    )
    mapped_seeds = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(logical_seeds, dtype=jnp.float64),
            batch_size=min(64, logical_seeds.shape[0]),
        ),
        dtype=np.float64,
    )
    physical_seeds = np.column_stack(
        [
            np.linalg.norm(mapped_seeds[:, :2], axis=1),
            mapped_seeds[:, 2],
        ]
    )
    simsopt_sections, _, _, lost, trace_seconds = _trace_simsopt_poincare(
        simsopt_field,
        boundary_surface,
        physical_seeds,
        [0.0],
        nfp=int(nfp),
        tol=float(simsopt_tol),
        tmax=float(simsopt_tmax),
        interpolation_degree=int(interpolation_degree),
        interpolation_points=int(interpolation_points),
    )
    simsopt_logical = _physical_sections_to_logical(
        simsopt_sections[0.0],
        logical_slice,
    )
    mrx_logical_sections: dict[float, list[np.ndarray]] = {}
    _trace_mrx_poincare(
        seq,
        dof,
        map_fn,
        [0.0],
        nlines=int(nrho),
        turns=int(mrx_turns),
        theta0=float(theta0),
        tol=float(mrx_tol),
        logical_sections_out=mrx_logical_sections,
        seed_rho_values=seed_rho,
        theta0_values=seed_theta,
        method=str(mrx_method),
    )
    mrx_logical = mrx_logical_sections[0.0]
    if cache_path is not None:
        cache_path = cache_path.expanduser().resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload: dict[str, np.ndarray] = {
            "logical_seeds": logical_seeds,
        }
        for index, (logical_line, physical_line) in enumerate(
            zip(simsopt_logical, simsopt_sections[0.0])
        ):
            cache_payload[f"simsopt_physical_line_{index:04d}"] = np.asarray(
                physical_line
            )
            cache_payload[f"simsopt_line_{index:04d}"] = np.asarray(logical_line)
        for index, line in enumerate(mrx_logical):
            cache_payload[f"mrx_line_{index:04d}"] = np.asarray(line)
        np.savez_compressed(cache_path, **cache_payload)
    reference_path, comparison_path = _plot_reference_island_zooms(
        simsopt_logical,
        mrx_logical,
        logical_seeds,
        output_dir,
        rho_min=float(rho_min),
        rho_max=float(rho_max),
    )

    counts = np.asarray([line.shape[0] for line in simsopt_logical], dtype=int)
    valid = (counts >= int(minimum_intersections)) & ~np.asarray(lost, dtype=bool)
    valid_lines = [
        line for line, keep in zip(simsopt_logical, valid) if bool(keep)
    ]
    valid_seeds = logical_seeds[valid]
    width_profile = _island_width_profile(valid_lines, valid_seeds[:, 0])
    trapped_summary = _trapped_separatrix_summary(
        valid_lines,
        valid_seeds,
        poloidal_mode=6,
    )
    corrected_widths = [
        float(item["detrended_width_rho_q05_q95"])
        for item in width_profile
    ]
    return {
        "resonance": {"iota": 0.5, "poloidal_mode": 6, "toroidal_mode": 3},
        "rho_min": float(rho_min),
        "rho_max": float(rho_max),
        "radial_seed_count": int(nrho),
        "phase_seed_count": int(phases),
        "total_line_count": int(logical_seeds.shape[0]),
        "logical_seeds": logical_seeds.tolist(),
        "intersection_counts": counts.tolist(),
        "median_intersection_count": float(np.median(counts)),
        "minimum_intersections_for_width": int(minimum_intersections),
        "width_line_count": int(np.sum(valid)),
        "lost_lines": [bool(value) for value in lost],
        "trace_seconds": float(trace_seconds),
        "simsopt_tmax": float(simsopt_tmax),
        "simsopt_tol": float(simsopt_tol),
        "interpolation_degree": int(interpolation_degree),
        "interpolation_points": int(interpolation_points),
        "island_width_profile": width_profile,
        "max_detrended_island_width_rho": (
            max(corrected_widths) if corrected_widths else None
        ),
        "trapped_separatrix": trapped_summary,
        "reference_zoom_file": str(reference_path),
        "comparison_zoom_file": str(comparison_path),
        "trace_cache_npz": str(cache_path) if cache_path is not None else None,
    }


def _resonant_normal_error_amplitudes(
    seq: Any,
    dof: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    simsopt_field: Any,
    resonances: list[dict[str, float | int]],
    *,
    mrx_scale: float,
    ntheta: int,
    nzeta: int,
) -> list[dict[str, float | int]]:
    """Fourier-analyze the normal MRX-minus-SIMSOPT field error."""
    if ntheta < 4 or nzeta < 4:
        raise ValueError("normal-error Fourier grids require at least 4×4 points")
    theta = np.arange(int(ntheta), dtype=np.float64) / int(ntheta)
    zeta = np.arange(int(nzeta), dtype=np.float64) / int(nzeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    jacobian_fn = jax.jit(jax.jacfwd(map_fn))
    results: list[dict[str, float | int]] = []
    for resonance in resonances:
        rho = float(resonance["rho"])
        logical = np.stack(
            [
                np.full(theta_grid.size, rho),
                theta_grid.ravel(),
                zeta_grid.ravel(),
            ],
            axis=1,
        )
        logical_jax = jnp.asarray(logical, dtype=jnp.float64)
        mapped = np.asarray(
            jax.lax.map(map_fn, logical_jax, batch_size=256),
            dtype=np.float64,
        )
        jacobians = np.asarray(
            jax.lax.map(jacobian_fn, logical_jax, batch_size=128),
            dtype=np.float64,
        )
        grad_rho = np.linalg.solve(
            np.swapaxes(jacobians, 1, 2),
            np.broadcast_to(
                np.asarray([1.0, 0.0, 0.0]),
                mapped.shape,
            )[..., None],
        )[..., 0]
        unit_normal = grad_rho / np.linalg.norm(
            grad_rho,
            axis=1,
            keepdims=True,
        )
        mrx_field = float(mrx_scale) * _evaluate_pushforward(
            seq,
            dof,
            map_fn,
            logical_jax,
            k=2,
        )
        simsopt_values = _evaluate_simsopt_field(simsopt_field, mapped)
        error_field = np.asarray(mrx_field) - simsopt_values
        normal_error = np.sum(error_field * unit_normal, axis=1).reshape(
            theta_grid.shape
        )
        poloidal_mode = int(resonance["poloidal_mode"])
        field_period_mode = int(resonance["field_period_mode"])
        phase = np.exp(
            -2.0j
            * np.pi
            * (
                poloidal_mode * theta_grid
                - field_period_mode * zeta_grid
            )
        )
        amplitude = float(abs(np.mean(normal_error * phase)))
        reference_rms = float(
            np.sqrt(np.mean(np.sum(simsopt_values**2, axis=1)))
        )
        results.append(
            {
                **resonance,
                "normal_error_fourier_tesla": amplitude,
                "normal_error_fourier_relative": float(
                    amplitude / reference_rms
                ),
                "normal_error_rms_relative": float(
                    np.sqrt(np.mean(normal_error**2)) / reference_rms
                ),
            }
        )
    return results


def _resonant_normal_error_radial_profile(
    seq: Any,
    dof: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    simsopt_field: Any,
    rho_values: np.ndarray,
    *,
    mrx_scale: float,
    poloidal_mode: int = 6,
    toroidal_mode: int = 3,
    nfp: int = 3,
    ntheta: int = 32,
    nzeta: int = 32,
) -> list[dict[str, float | int]]:
    """Evaluate one resonant normal-error harmonic across logical radius.

    Parameters are explicit so the helper can diagnose saved FEM vectors
    without retracing field lines. The returned records use the same schema as
    :func:`_resonant_normal_error_amplitudes`.
    """
    if int(toroidal_mode) % int(nfp) != 0:
        raise ValueError("toroidal mode must be divisible by nfp")
    resonances = [
        {
            "rho": float(rho),
            "iota": float(toroidal_mode) / int(poloidal_mode),
            "toroidal_mode": int(toroidal_mode),
            "poloidal_mode": int(poloidal_mode),
            "field_period_mode": int(toroidal_mode) // int(nfp),
            "sample_mismatch": 0.0,
        }
        for rho in np.asarray(rho_values, dtype=np.float64).reshape(-1)
    ]
    return _resonant_normal_error_amplitudes(
        seq,
        dof,
        map_fn,
        simsopt_field,
        resonances,
        mrx_scale=float(mrx_scale),
        ntheta=int(ntheta),
        nzeta=int(nzeta),
    )


def _plot_resonant_error_radial_profiles(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> Path | None:
    """Plot radial ``(6,3)`` normal-error profiles for diagnosed FEM grids."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    diagnosed = [
        record
        for record in records
        if record.get("island_diagnostics", {}).get(
            "resonant_normal_error_radial_profile"
        )
    ]
    if not diagnosed:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    for record in diagnosed:
        profile = record["island_diagnostics"][
            "resonant_normal_error_radial_profile"
        ]
        axis.plot(
            [float(item["rho"]) for item in profile],
            [
                float(item["normal_error_fourier_relative"])
                for item in profile
            ],
            marker="o",
            markersize=3,
            label="×".join(str(value) for value in record["ns"]),
        )
    axis.set_xlabel(r"Logical radius $\rho$")
    axis.set_ylabel(r"Relative normal-error amplitude $a_{6,3}/B_{\rm rms}$")
    axis.set_title(r"Radial structure of the $(m,n)=(6,3)$ error")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend(title="FEM grid", fontsize="small")
    figure.tight_layout()
    path = (output_dir / "resonant_error_a63_vs_rho.png").resolve()
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_simsopt_native_poincare(
    phi_hits: list[np.ndarray],
    phis: list[float],
    boundary_surface: Any,
    output_dir: Path,
    *,
    dpi: int = 150,
    filename: str = "poincare_simsopt_native.png",
) -> Path:
    """Render an independent reference using SIMSOPT's native plot helper."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from simsopt.field.tracing import plot_poincare_data

    output_dir.mkdir(parents=True, exist_ok=True)
    path = (output_dir / filename).resolve()
    plot_poincare_data(
        phi_hits,
        phis,
        str(path),
        mark_lost=True,
        surf=boundary_surface,
        dpi=int(dpi),
        s=2,
    )
    return path


def _plot_poincare_comparison(
    mrx_sections: dict[float, list[np.ndarray]],
    simsopt_sections: dict[float, list[np.ndarray]],
    slices: list[dict[str, Any]],
    output_dir: Path,
    *,
    dpi: int = 150,
    filename: str = "poincare_mrx_vs_simsopt.png",
) -> Path:
    """Plot matched MRX and SIMSOPT Poincare sections with common limits."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    nsections = len(slices)
    fig, axes = plt.subplots(
        nsections,
        2,
        figsize=(10, 4.2 * nsections),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    nlines = max((len(lines) for lines in mrx_sections.values()), default=1)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, nlines))
    all_R: list[np.ndarray] = []
    all_Z: list[np.ndarray] = []
    for row, slab in enumerate(slices):
        zeta = float(slab["zeta"])
        boundary_R = np.append(slab["R"][-1], slab["R"][-1, 0])
        boundary_Z = np.append(slab["Z"][-1], slab["Z"][-1, 0])
        all_R.append(boundary_R)
        all_Z.append(boundary_Z)
        for column, (title, sections) in enumerate(
            (("MRX", mrx_sections), ("SIMSOPT", simsopt_sections))
        ):
            ax = axes[row, column]
            ax.plot(boundary_R, boundary_Z, color="black", linewidth=1.0)
            for line_index, points in enumerate(sections[zeta]):
                points = np.asarray(points)
                if points.size:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=2.0,
                        color=colors[line_index],
                        rasterized=True,
                    )
                    all_R.append(points[:, 0])
                    all_Z.append(points[:, 1])
            ax.set_title(rf"{title}, $\zeta={zeta:.3f}$")
            ax.set_aspect("equal")
            ax.set_xlabel("$R$ [m]")
            ax.set_ylabel("$Z$ [m]")
    if all_R and all_Z:
        rmin = min(float(np.min(values)) for values in all_R if values.size)
        rmax = max(float(np.max(values)) for values in all_R if values.size)
        zmin = min(float(np.min(values)) for values in all_Z if values.size)
        zmax = max(float(np.max(values)) for values in all_Z if values.size)
        rpad = 0.03 * max(rmax - rmin, 1.0e-6)
        zpad = 0.03 * max(zmax - zmin, 1.0e-6)
        for ax in axes.ravel():
            ax.set_xlim(rmin - rpad, rmax + rpad)
            ax.set_ylim(zmin - zpad, zmax + zpad)
    fig.tight_layout()
    path = (output_dir / filename).resolve()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return path


def compare_push_u_to_simsopt(
    seq: Any,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    json_path: Path,
    volume_h5: Path,
    *,
    nfp: int,
    flip_zeta: bool,
    y_sign: float = -1.0,
    k: int = 1,
    compare_curl: bool = False,
    max_points: int | None = None,
    seed: int = 0,
    align_pointwise: bool = True,
) -> dict[str, Any]:
    """
    Compare ``Pushforward(u)`` (and optionally ``curl(u)``) to SIMSOPT **B** on a volume grid.

    Parameters
    ----------
    seq, u, map_fn
        Assembled sequence, DOF vector, JIT map.
    json_path
        QUASR SIMSOPT JSON (coils).
    volume_h5
        MRX volume HDF5 with ``eval_points``, ``R``, ``Z`` (``B`` optional).
    nfp, flip_zeta
        Field period count and ζ mirror for ``xyz_from_mrx_eval_points``.
    y_sign
        Sign in ``Y = y_sign * R * sin(phi)``. Use ``+1`` for GVEC-exported
        geometry and SIMSOPT coils.
    k
        Form degree for ``u`` (1 or 2).
    compare_curl
        If True and ``k=1``, also compare ``curl(u)`` vs SIMSOPT **B**.
    max_points
        Optional subsample count.
    align_pointwise
        Report ``rel_l2_aligned`` (optimal global scale).

    Returns
    -------
    dict
        JSON-serializable metrics.
    """
    with h5py.File(volume_h5, "r") as f:
        if "eval_points" not in f or "R" not in f or "Z" not in f:
            raise ValueError(f"{volume_h5} needs eval_points, R, Z")
        pts = jnp.asarray(f["eval_points"][:], dtype=jnp.float64)
        R = np.asarray(f["R"][:], dtype=np.float64).ravel()
        Z = np.asarray(f["Z"][:], dtype=np.float64).ravel()
        file_gvec = bool(f.attrs.get("eval_points_gvec_convention", False))

    pts = _normalize_eval_points_gvec(pts, file_gvec=file_gvec)
    n = int(pts.shape[0])
    idx = np.arange(n, dtype=np.int64)
    if max_points is not None and max_points < n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(max_points), replace=False))

    pts_s = np.asarray(pts[idx], dtype=np.float64)
    X, Y, Zc = xyz_from_mrx_eval_points(
        R[idx],
        Z[idx],
        pts_s,
        int(nfp),
        flip_zeta=bool(flip_zeta),
        y_sign=float(y_sign),
    )
    pts_xyz = np.stack([X, Y, Zc], axis=1)
    B_bs = _simsopt_B_at_xyz(json_path, pts_xyz)

    u_push = _evaluate_pushforward(seq, u, map_fn, jnp.asarray(pts_s), k=k)
    out: dict[str, Any] = {
        "json": str(json_path.resolve()),
        "volume_h5": str(volume_h5.resolve()),
        "n_compare": int(idx.size),
        "n_total": n,
        "nfp": int(nfp),
        "flip_zeta": bool(flip_zeta),
        "y_sign": float(y_sign),
        "k": int(k),
        "fem_quantity": f"pushforward_u_k{k}_cartesian",
        "positions_source": "eval_points",
        "simsopt_B_rms": float(np.sqrt(np.mean(np.sum(B_bs**2, axis=1)))),
        "compare_align_pointwise": bool(align_pointwise),
        "pushforward_vs_simsopt": _pointwise_vector_metrics(
            u_push, B_bs, align_pointwise=align_pointwise
        ),
    }

    if compare_curl:
        if k != 1:
            raise ValueError("--compare-curl only applies to k=1")
        B_curl = _evaluate_curl_u_cartesian(seq, u, map_fn, jnp.asarray(pts_s))
        out["fem_quantity_curl"] = "strong_curl_u_k1_as_2form_cartesian"
        out["curl_u_vs_simsopt"] = _pointwise_vector_metrics(
            B_curl, B_bs, align_pointwise=align_pointwise
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, required=True, help="QUASR SIMSOPT serial JSON")
    parser.add_argument(
        "--volume-h5",
        type=Path,
        required=True,
        help="MRX volume HDF5 (eval_points, R, Z) defining sample grid",
    )
    parser.add_argument("--dof-npy", type=Path, required=True, help="Nullspace DOF vector (.npy)")
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="Meta JSON (default: hcurl_nullspace_meta.json or hodge_k2_nullspace_meta.json)",
    )
    parser.add_argument(
        "--from-saved-meta",
        action="store_true",
        help="Require meta JSON next to --dof-npy (same as default lookup)",
    )
    parser.add_argument("--k", type=int, default=1, choices=(1, 2), help="Form degree of u")
    parser.add_argument(
        "--compare-curl",
        action="store_true",
        help="Also compare curl(u) vs SIMSOPT B (k=1 only; recommended)",
    )
    parser.add_argument("--nfp", type=int, default=None, help="Override meta/HDF5 nfp")
    parser.add_argument("--flip-zeta", action="store_true", help="Override meta: mirror ζ for XYZ")
    parser.add_argument("--no-flip-zeta", action="store_true", help="Override meta: no ζ flip")
    y_group = parser.add_mutually_exclusive_group()
    y_group.add_argument(
        "--y-plus-rsin",
        action="store_true",
        help="Use the GVEC/SIMSOPT lab frame Y = +R sin(phi)",
    )
    y_group.add_argument(
        "--y-minus-rsin",
        action="store_true",
        help="Use the MRX stellarator-map frame Y = -R sin(phi)",
    )
    parser.add_argument("--max-points", type=int, default=None, help="Subsample for speed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-align-pointwise",
        action="store_true",
        help="Skip rel_l2_aligned / pointwise scale metrics",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Write component cross-sections and Poincare plots to this directory",
    )
    parser.add_argument(
        "--zeta-values",
        type=str,
        default="0,0.25,0.5,0.75",
        help="Comma-separated normalized field-period sections",
    )
    parser.add_argument("--slice-nrho", type=int, default=32)
    parser.add_argument("--slice-ntheta", type=int, default=48)
    parser.add_argument(
        "--resolution-sweep",
        type=str,
        default=None,
        help="Optional comma-separated NRHOxNTHETA grids for error/timing plots",
    )
    parser.add_argument(
        "--fem-resolution-sweep",
        type=str,
        default=None,
        help="Optional comma-separated NRxNTxNZ grids to re-solve for convergence",
    )
    parser.add_argument("--fem-sweep-nrho", type=int, default=12)
    parser.add_argument("--fem-sweep-ntheta", type=int, default=18)
    parser.add_argument("--fem-sweep-time-budget", type=float, default=900.0)
    parser.add_argument("--fem-sweep-dense-max-dofs", type=int, default=1000)
    parser.add_argument("--fem-sweep-eps", type=float, default=1.0e-3)
    parser.add_argument("--fem-sweep-inner-tol", type=float, default=1.0e-10)
    parser.add_argument("--fem-sweep-iterative-maxiter", type=int, default=100)
    parser.add_argument(
        "--fem-sweep-resume-json",
        type=Path,
        action="append",
        default=None,
        help="Prior sweep JSON; repeat to merge corrected resolution records",
    )
    parser.add_argument(
        "--fem-sweep-refine-current",
        action="store_true",
        help="Continue the iterative solve for the current saved FEM vector",
    )
    parser.add_argument(
        "--fem-sweep-refine-resolutions",
        type=str,
        default=None,
        help="Comma-separated FEM grids to warm-start and re-solve",
    )
    parser.add_argument(
        "--fem-sweep-dense-verify",
        type=str,
        default=None,
        help="Comma-separated FEM grids to replace with dense-verified vectors",
    )
    parser.add_argument(
        "--no-fem-sweep-health",
        action="store_true",
        help="Skip Rayleigh, residual, divergence, and weak-curl diagnostics",
    )
    parser.add_argument(
        "--fem-sweep-poincare",
        action="store_true",
        help="Generate matched Poincare plots for every completed FEM resolution",
    )
    parser.add_argument(
        "--fem-sweep-island-diagnostics",
        action="store_true",
        help="Measure iota, resonant normal error, and logical island widths",
    )
    parser.add_argument("--island-rho-min", type=float, default=0.58)
    parser.add_argument("--island-rho-max", type=float, default=0.82)
    parser.add_argument("--island-max-poloidal-mode", type=int, default=16)
    parser.add_argument("--island-max-resonances", type=int, default=4)
    parser.add_argument("--island-fourier-ntheta", type=int, default=32)
    parser.add_argument("--island-fourier-nzeta", type=int, default=32)
    parser.add_argument(
        "--island-zoom",
        action="store_true",
        help="Trace phase-resolved seeds and plot logical rho-theta chain zooms",
    )
    parser.add_argument("--island-zoom-rho-min", type=float, default=0.58)
    parser.add_argument("--island-zoom-rho-max", type=float, default=0.82)
    parser.add_argument("--island-zoom-nrho", type=int, default=8)
    parser.add_argument("--island-zoom-phases", type=int, default=3)
    parser.add_argument("--simsopt-island-zoom-rho-min", type=float, default=0.64)
    parser.add_argument("--simsopt-island-zoom-rho-max", type=float, default=0.78)
    parser.add_argument("--simsopt-island-zoom-nrho", type=int, default=16)
    parser.add_argument("--simsopt-island-zoom-phases", type=int, default=3)
    parser.add_argument(
        "--simsopt-island-min-intersections",
        type=int,
        default=100,
    )
    parser.add_argument("--poincare-nlines", type=int, default=20)
    parser.add_argument("--poincare-turns", type=int, default=300)
    parser.add_argument("--poincare-theta0", type=float, default=0.5)
    parser.add_argument("--poincare-tol", type=float, default=1.0e-9)
    parser.add_argument("--simsopt-tmax", type=float, default=20000.0)
    parser.add_argument("--simsopt-poincare-tol", type=float, default=1.0e-12)
    parser.add_argument("--simsopt-interp-degree", type=int, default=3)
    parser.add_argument("--simsopt-interp-points", type=int, default=16)
    parser.add_argument(
        "--no-component-plots",
        action="store_true",
        help="Skip Cartesian component cross-sections",
    )
    parser.add_argument(
        "--no-poincare",
        action="store_true",
        help="Skip matched MRX/SIMSOPT Poincare plots",
    )
    parser.add_argument("-o", "--output-json", type=Path, default=None)
    args = parser.parse_args()

    json_path = args.json.expanduser().resolve()
    h5_path = args.volume_h5.expanduser().resolve()
    dof_path = args.dof_npy.expanduser().resolve()
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")
    if not h5_path.is_file():
        raise SystemExit(f"volume HDF5 not found: {h5_path}")
    if not dof_path.is_file():
        raise SystemExit(f"DOF npy not found: {dof_path}")

    u = jnp.asarray(np.load(dof_path), dtype=jnp.float64).reshape(-1)
    hcurl_mod = _load_hcurl_module()
    meta = _load_meta_for_dof(dof_path, k=int(args.k), meta_json=args.meta_json)
    seq, map_jit, nfp_meta, flip_meta, _map_raw = _rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=not bool(args.compare_curl),
    )

    if bool(args.compare_curl):
        print("Assembling incidence/derivative operators for strong_curl...", flush=True)
        hcurl_mod.assemble_ops(seq)

    nfp = int(args.nfp) if args.nfp is not None else nfp_meta
    if args.flip_zeta:
        flip_zeta = True
    elif args.no_flip_zeta:
        flip_zeta = False
    else:
        flip_zeta = flip_meta
    y_sign = 1.0 if args.y_plus_rsin else -1.0

    stats = compare_push_u_to_simsopt(
        seq,
        u,
        map_jit,
        json_path,
        h5_path,
        nfp=nfp,
        flip_zeta=flip_zeta,
        y_sign=y_sign,
        k=int(args.k),
        compare_curl=bool(args.compare_curl),
        max_points=args.max_points,
        seed=int(args.seed),
        align_pointwise=not bool(args.no_align_pointwise),
    )
    stats["dof_npy"] = str(dof_path)
    stats["meta_json"] = str(
        args.meta_json.resolve()
        if args.meta_json is not None
        else (
            dof_path.with_name("hodge_k2_nullspace_meta.json")
            if args.k == 2
            else dof_path.with_name("hcurl_nullspace_meta.json")
        )
    )

    if args.plot_dir is not None:
        if int(args.k) != 2:
            raise SystemExit("Post-processing plots currently require --k 2")
        plot_dir = args.plot_dir.expanduser().resolve()
        zeta_values = _parse_section_values(args.zeta_values)
        scale = float(
            stats["pushforward_vs_simsopt"].get(
                "pointwise_u_scale_to_optimal",
                1.0,
            )
        )
        if not np.isfinite(scale):
            scale = 1.0
        simsopt_field, simsopt_surfaces = _load_simsopt_field(json_path)
        if not simsopt_surfaces:
            raise RuntimeError("QUASR serialization contains no boundary surfaces")
        boundary_surface = _full_torus_surface(simsopt_surfaces[-1])
        plot_files: list[str] = []
        boundary_flux_diagnostics: dict[str, Any] | None = None
        biot_savart_validation: dict[str, Any] | None = None
        if bool(args.fem_sweep_island_diagnostics):
            boundary_flux_diagnostics = _boundary_flux_diagnostics(
                simsopt_field,
                simsopt_surfaces[-1],
                map_jit,
                nfp=int(nfp),
                ntheta=64,
                nzeta=64,
            )
            biot_savart_validation = _biot_savart_validation(
                simsopt_field,
                boundary_surface,
                map_jit,
                nfp=int(nfp),
                interpolation_degree=int(args.simsopt_interp_degree),
                interpolation_points=int(args.simsopt_interp_points),
            )
        needs_slices = (
            not bool(args.no_component_plots)
            or not bool(args.no_poincare)
            or bool(args.fem_sweep_poincare)
            or bool(args.fem_sweep_island_diagnostics)
        )
        slices: list[dict[str, Any]] = []
        if needs_slices:
            slices = _evaluate_component_slices(
                seq,
                u,
                map_jit,
                simsopt_field,
                zeta_values,
                nrho=int(args.slice_nrho),
                ntheta=int(args.slice_ntheta),
                mrx_scale=scale,
            )
        if not bool(args.no_component_plots):
            plot_files.extend(
                str(path)
                for path in _plot_component_slices(
                    slices,
                    plot_dir,
                    mrx_scale=scale,
                )
            )

        poincare_stats: dict[str, Any] | None = None
        fem_poincare_config: dict[str, Any] | None = None
        simsopt_island_baseline: dict[str, Any] | None = None
        if (
            not bool(args.no_poincare)
            or bool(args.fem_sweep_poincare)
            or bool(args.fem_sweep_island_diagnostics)
        ):
            base_logical_sections: dict[float, list[np.ndarray]] = {}
            base_trace_audit: dict[str, Any] = {}
            (
                mrx_sections,
                logical_seeds,
                physical_seeds,
                mrx_transit_counts,
            ) = _trace_mrx_poincare(
                seq,
                u,
                map_jit,
                zeta_values,
                nlines=int(args.poincare_nlines),
                turns=int(args.poincare_turns),
                theta0=float(args.poincare_theta0),
                tol=float(args.poincare_tol),
                logical_sections_out=(
                    base_logical_sections
                    if bool(args.fem_sweep_island_diagnostics)
                    else None
                ),
                audit_out=base_trace_audit,
            )
            (
                simsopt_sections,
                simsopt_phi_hits,
                simsopt_phis,
                simsopt_lost,
                simsopt_trace_seconds,
            ) = _trace_simsopt_poincare(
                simsopt_field,
                boundary_surface,
                physical_seeds,
                zeta_values,
                nfp=nfp,
                tol=float(args.simsopt_poincare_tol),
                tmax=float(args.simsopt_tmax),
                interpolation_degree=int(args.simsopt_interp_degree),
                interpolation_points=int(args.simsopt_interp_points),
            )
            native_filename = (
                "poincare_simsopt_native.png"
                if not bool(args.no_poincare)
                else "poincare_simsopt_native_fem_reference.png"
            )
            native_poincare_path = _plot_simsopt_native_poincare(
                simsopt_phi_hits,
                simsopt_phis,
                boundary_surface,
                plot_dir,
                filename=native_filename,
            )
            plot_files.append(str(native_poincare_path))
            if bool(args.island_zoom) and bool(
                args.fem_sweep_island_diagnostics
            ):
                logical_slice = min(
                    slices,
                    key=lambda slab: abs(float(slab["zeta"])),
                )
                simsopt_island_baseline = _simsopt_island_baseline(
                    seq,
                    u,
                    map_jit,
                    simsopt_field,
                    boundary_surface,
                    logical_slice,
                    plot_dir,
                    nfp=int(nfp),
                    rho_min=float(args.simsopt_island_zoom_rho_min),
                    rho_max=float(args.simsopt_island_zoom_rho_max),
                    nrho=int(args.simsopt_island_zoom_nrho),
                    phases=int(args.simsopt_island_zoom_phases),
                    theta0=float(args.poincare_theta0),
                    mrx_turns=int(args.poincare_turns),
                    mrx_tol=float(args.poincare_tol),
                    simsopt_tmax=float(args.simsopt_tmax),
                    simsopt_tol=float(args.simsopt_poincare_tol),
                    interpolation_degree=int(args.simsopt_interp_degree),
                    interpolation_points=int(args.simsopt_interp_points),
                    minimum_intersections=int(
                        args.simsopt_island_min_intersections
                    ),
                )
                plot_files.extend(
                    [
                        str(simsopt_island_baseline["reference_zoom_file"]),
                        str(simsopt_island_baseline["comparison_zoom_file"]),
                    ]
                )
            if not bool(args.no_poincare):
                poincare_path = _plot_poincare_comparison(
                    mrx_sections,
                    simsopt_sections,
                    slices,
                    plot_dir,
                )
                plot_files.append(str(poincare_path))
                poincare_stats = {
                    "nlines": int(args.poincare_nlines),
                    "turns": int(args.poincare_turns),
                    "theta0": float(args.poincare_theta0),
                    "tol": float(args.poincare_tol),
                    "simsopt_tmax": float(args.simsopt_tmax),
                    "simsopt_tol": float(args.simsopt_poincare_tol),
                    "simsopt_interpolation_degree": int(
                        args.simsopt_interp_degree
                    ),
                    "simsopt_interpolation_points": int(
                        args.simsopt_interp_points
                    ),
                    "simsopt_trace_seconds": simsopt_trace_seconds,
                    "simsopt_lost_lines": simsopt_lost,
                    "logical_seeds": logical_seeds.tolist(),
                    "physical_RZ_seeds_m": physical_seeds.tolist(),
                    "mrx_completed_transits": mrx_transit_counts,
                    "mrx_trace_audit": base_trace_audit,
                    "mrx_intersection_counts": {
                        f"{zeta:.12g}": [
                            int(points.shape[0]) for points in lines
                        ]
                        for zeta, lines in mrx_sections.items()
                    },
                    "simsopt_intersection_counts": {
                        f"{zeta:.12g}": [
                            int(points.shape[0]) for points in lines
                        ]
                        for zeta, lines in simsopt_sections.items()
                    },
                }
            if bool(args.fem_sweep_poincare) or bool(
                args.fem_sweep_island_diagnostics
            ):
                diagnostic_zeta = float(zeta_values[0])
                fem_poincare_config = {
                    "nlines": int(args.poincare_nlines),
                    "turns": int(args.poincare_turns),
                    "theta0": float(args.poincare_theta0),
                    "tol": float(args.poincare_tol),
                    "simsopt_sections": simsopt_sections,
                    "slices": slices,
                    "base_mrx_sections": mrx_sections,
                    "base_transit_counts": mrx_transit_counts,
                    "base_logical_sections": base_logical_sections,
                    "logical_seeds": logical_seeds,
                    "nfp": int(nfp),
                    "island_diagnostics": bool(
                        args.fem_sweep_island_diagnostics
                    ),
                    "diagnostic_zeta": diagnostic_zeta,
                    "rho_min": float(args.island_rho_min),
                    "rho_max": float(args.island_rho_max),
                    "max_poloidal_mode": int(
                        args.island_max_poloidal_mode
                    ),
                    "max_resonances": int(args.island_max_resonances),
                    "fourier_ntheta": int(args.island_fourier_ntheta),
                    "fourier_nzeta": int(args.island_fourier_nzeta),
                    "island_zoom": bool(args.island_zoom),
                    "zoom_rho_min": float(args.island_zoom_rho_min),
                    "zoom_rho_max": float(args.island_zoom_rho_max),
                    "zoom_nrho": int(args.island_zoom_nrho),
                    "zoom_phases": int(args.island_zoom_phases),
                    "simsopt_iota_profile": (
                        _simsopt_iota_profile_from_pitch(
                            map_jit,
                            simsopt_field,
                            logical_seeds[:, 0],
                            nfp=int(nfp),
                            ntheta=min(
                                24,
                                int(args.island_fourier_ntheta),
                            ),
                            nzeta=min(
                                24,
                                int(args.island_fourier_nzeta),
                            ),
                        )
                        if bool(args.fem_sweep_island_diagnostics)
                        else None
                    ),
                }
        resolution_records: list[dict[str, float | int]] | None = None
        radial_error_profile: list[dict[str, float]] | None = None
        if args.resolution_sweep is not None:
            resolutions = _parse_resolution_values(args.resolution_sweep)
            resolution_records, radial_error_profile = _benchmark_resolution_sweep(
                seq,
                u,
                map_jit,
                simsopt_field,
                zeta_values,
                resolutions,
                mrx_scale=scale,
            )
            plot_files.extend(
                str(path)
                for path in _plot_resolution_sweep(
                    resolution_records,
                    plot_dir,
                )
            )
            plot_files.extend(
                str(path)
                for path in _plot_radial_error_profile(
                    radial_error_profile,
                    plot_dir,
                )
            )
        fem_resolution_records: list[dict[str, Any]] | None = None
        island_scaling_summary: dict[str, Any] | None = None
        if args.fem_resolution_sweep is not None:
            fem_resolutions = _parse_fem_resolution_values(
                args.fem_resolution_sweep
            )
            fem_resolution_records = _run_fem_resolution_sweep(
                meta,
                _load_twoform_module(),
                hcurl_mod,
                simsopt_field,
                zeta_values,
                fem_resolutions,
                base_seq=seq,
                base_dof=u,
                base_map_fn=map_jit,
                base_dof_path=dof_path,
                resume_records=_load_fem_sweep_records(
                    args.fem_sweep_resume_json
                ),
                nrho=int(args.fem_sweep_nrho),
                ntheta=int(args.fem_sweep_ntheta),
                time_budget_seconds=float(args.fem_sweep_time_budget),
                dense_check_max_dofs=int(args.fem_sweep_dense_max_dofs),
                solver_eps=float(args.fem_sweep_eps),
                inner_tol=float(args.fem_sweep_inner_tol),
                iterative_maxiter=int(args.fem_sweep_iterative_maxiter),
                refine_current=bool(args.fem_sweep_refine_current),
                refine_resolutions=set(
                    _parse_fem_resolution_values(
                        args.fem_sweep_refine_resolutions
                    )
                    if args.fem_sweep_refine_resolutions is not None
                    else []
                ),
                dense_verify_resolutions=set(
                    _parse_fem_resolution_values(args.fem_sweep_dense_verify)
                    if args.fem_sweep_dense_verify is not None
                    else []
                ),
                compute_health=not bool(args.no_fem_sweep_health),
                output_dir=plot_dir,
                poincare_config=fem_poincare_config,
            )
            plot_files.extend(
                str(path)
                for path in _plot_fem_resolution_sweep(
                    fem_resolution_records,
                    plot_dir,
                )
            )
            plot_files.extend(
                str(record["poincare_file"])
                for record in fem_resolution_records
                if record.get("poincare_file") is not None
            )
            if bool(args.fem_sweep_island_diagnostics):
                if fem_poincare_config is None:
                    raise RuntimeError("island diagnostics require Poincare data")
                island_scaling_summary = _island_scaling_summary(
                    fem_resolution_records
                )
                island_scaling_summary["width_model_consistency"] = (
                    _island_width_consistency_summary(
                        fem_resolution_records
                    )
                )
                plot_files.extend(
                    str(path)
                    for path in _plot_island_diagnostics(
                        fem_resolution_records,
                        fem_poincare_config["simsopt_iota_profile"],
                        plot_dir,
                    )
                )
                radial_profile_path = _plot_resonant_error_radial_profiles(
                    fem_resolution_records,
                    plot_dir,
                )
                if radial_profile_path is not None:
                    plot_files.append(str(radial_profile_path))
                plot_files.extend(
                    str(record["island_diagnostics"]["island_zoom_file"])
                    for record in fem_resolution_records
                    if record.get("island_diagnostics") is not None
                    and record["island_diagnostics"].get("island_zoom_file")
                    is not None
                )
        stats["postprocessing"] = {
            "plot_dir": str(plot_dir),
            "zeta_values": zeta_values,
            "slice_nrho": int(args.slice_nrho),
            "slice_ntheta": int(args.slice_ntheta),
            "mrx_component_scale": scale,
            "component_plots": not bool(args.no_component_plots),
            "poincare": poincare_stats,
            "resolution_sweep": resolution_records,
            "radial_error_profile": radial_error_profile,
            "fem_resolution_sweep": fem_resolution_records,
            "simsopt_iota_profile": (
                fem_poincare_config["simsopt_iota_profile"]
                if fem_poincare_config is not None
                and bool(args.fem_sweep_island_diagnostics)
                else None
            ),
            "island_scaling_summary": island_scaling_summary,
            "simsopt_island_baseline": simsopt_island_baseline,
            "boundary_flux_diagnostics": boundary_flux_diagnostics,
            "biot_savart_validation": biot_savart_validation,
            "resolution_timing_scope": (
                "Pushforward(u2) wall time over all requested zeta sections; "
                "includes shape-specific JAX dispatch/compilation"
                if resolution_records is not None
                else None
            ),
            "files": plot_files,
        }

    print("=== Pushforward(u) vs SIMSOPT Biot–Savart ===")
    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for sk, sv in val.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {key}: {val}")

    if args.output_json is not None:
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2))
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
