#!/usr/bin/env python3
"""Lean nθ=18 FEM ladder driver.

- solve: dense-verify nullspace via _run_fem_resolution_sweep (no Poincare)
- diagnostics: L2 + MRX island zoom / trapped width + a63 Fourier (coarse)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.minimal_vacuum_problem import compare_push_u_to_simsopt as cmp  # noqa: E402

MVP = ROOT / "scripts" / "minimal_vacuum_problem"
PLOT_DIR = MVP / "fem_convergence_highres"
BASE_DOF = MVP / "solver_repair_8" / "fem_sweep_8x16x8_dof.npy"
META_JSON = MVP / "hodge_k2_nullspace_meta.json"
JSON_PATH = MVP / "serial0044970.json"


def _log(msg: str, log_path: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with log_path.open("a") as fh:
        fh.write(line + "\n")


def _parse_ns(text: str) -> tuple[int, int, int]:
    parts = tuple(int(x) for x in text.lower().split("x"))
    if len(parts) != 3:
        raise ValueError(f"bad resolution {text}")
    return parts  # type: ignore[return-value]


def _aligned_l2(
    seq,
    dof,
    map_fn,
    simsopt_field,
    zeta_values: list[float],
    nrho: int = 24,
    ntheta: int = 48,
) -> dict:
    mrx_fields: list[np.ndarray] = []
    simsopt_fields: list[np.ndarray] = []
    for zeta in zeta_values:
        logical, _, _ = cmp._poloidal_slice_points(zeta, nrho, ntheta)
        mapped = np.asarray(
            jax.lax.map(
                map_fn,
                jnp.asarray(logical, dtype=jnp.float64),
                batch_size=512,
            )
        )
        mrx_fields.append(
            cmp._evaluate_pushforward(
                seq, dof, map_fn, jnp.asarray(logical), k=2
            )
        )
        simsopt_fields.append(cmp._evaluate_simsopt_field(simsopt_field, mapped))
    return cmp._pointwise_vector_metrics(
        np.concatenate(mrx_fields),
        np.concatenate(simsopt_fields),
        align_pointwise=True,
    )


def _run_diagnostics(
    *,
    ns: tuple[int, int, int],
    prior: dict,
    meta: dict,
    hcurl_mod,
    simsopt_field,
    log_path: Path,
    poincare_turns: int,
    zoom_nrho: int,
    zoom_phases: int,
    fourier_n: int,
) -> dict:
    label = "x".join(str(v) for v in ns)
    dof_path = Path(str(prior["dof_npy"])).expanduser().resolve()
    if not dof_path.is_file():
        raise FileNotFoundError(dof_path)

    _log(f"rebuild pushforward seq for {label}", log_path)
    resolution_meta = dict(meta)
    resolution_meta["ns"] = list(ns)
    seq, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        resolution_meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    dof = jnp.asarray(np.load(dof_path), dtype=jnp.float64).reshape(-1)
    if int(dof.size) != int(seq.n2_dbc):
        raise ValueError(
            f"DOF size {dof.size} != seq.n2_dbc {seq.n2_dbc} for {label}"
        )

    zeta_values = [0.0, 0.25, 0.5, 0.75]
    _log("aligned L2 evaluation", log_path)
    t0 = time.perf_counter()
    metrics = _aligned_l2(seq, dof, map_fn, simsopt_field, zeta_values)
    _log(
        f"L2={metrics['rel_l2_aligned']:.6e} "
        f"scale={metrics['pointwise_u_scale_to_optimal']:.6e} "
        f"({time.perf_counter()-t0:.1f}s)",
        log_path,
    )

    section_zeta = 0.0
    _log(
        f"island zoom Poincare nrho={zoom_nrho} phases={zoom_phases} "
        f"turns={poincare_turns}",
        log_path,
    )
    t0 = time.perf_counter()
    zoom_rho = np.linspace(0.58, 0.82, int(zoom_nrho))
    base_theta = 0.5
    zoom_theta = (
        base_theta
        + np.arange(int(zoom_phases)) / (6.0 * max(int(zoom_phases), 1))
    ) % 1.0
    zoom_logical: dict[float, list[np.ndarray]] = {}
    _, zoom_seeds, _, _ = cmp._trace_mrx_poincare(
        seq,
        dof,
        map_fn,
        [section_zeta],
        nlines=int(zoom_rho.size),
        turns=int(poincare_turns),
        theta0=base_theta,
        tol=1.0e-9,
        logical_sections_out=zoom_logical,
        seed_rho_values=zoom_rho,
        theta0_values=zoom_theta,
    )
    zoom_lines = zoom_logical[section_zeta]
    trapped = cmp._trapped_separatrix_summary(
        zoom_lines, zoom_seeds, poloidal_mode=6
    )
    zoom_path = cmp._plot_island_zoom(
        zoom_lines,
        zoom_seeds,
        PLOT_DIR,
        label=label,
        rho_min=0.58,
        rho_max=0.82,
    )
    _log(
        f"trapped_w={trapped.get('trapped_separatrix_width_rho')} "
        f"zoom={zoom_path.name} ({time.perf_counter()-t0:.1f}s)",
        log_path,
    )

    _log("iota profile on zoom seeds (for resonance location)", log_path)
    seed_rho = np.asarray(zoom_seeds, dtype=np.float64)[:, 0]
    # Prefer unique rho rows at first phase for iota profile.
    unique_rho = np.unique(seed_rho)
    iota_lines = []
    iota_rho = []
    for rho in unique_rho:
        idxs = [i for i, r in enumerate(seed_rho) if abs(r - rho) < 1e-12]
        if idxs:
            iota_lines.append(zoom_lines[idxs[0]])
            iota_rho.append(rho)
    iota_profile = cmp._rotational_transform_profile(
        iota_lines, np.asarray(iota_rho), nfp=3
    )
    resonances = cmp._identify_iota_resonances(
        iota_profile,
        nfp=3,
        rho_min=0.58,
        rho_max=0.82,
        max_poloidal_mode=12,
    )[:8]
    _log(f"found {len(resonances)} resonances near island band", log_path)

    _log(f"a63 Fourier ntheta=nzeta={fourier_n}", log_path)
    t0 = time.perf_counter()
    resonant_amplitudes = cmp._resonant_normal_error_amplitudes(
        seq,
        dof,
        map_fn,
        simsopt_field,
        resonances,
        mrx_scale=float(metrics["pointwise_u_scale_to_optimal"]),
        ntheta=int(fourier_n),
        nzeta=int(fourier_n),
    )
    a63 = None
    for item in resonant_amplitudes:
        if int(item.get("poloidal_mode", -1)) == 6 and int(
            item.get("toroidal_mode", -1)
        ) == 3:
            a63 = float(item["normal_error_fourier_relative"])
            break
    _log(f"a63={a63} ({time.perf_counter()-t0:.1f}s)", log_path)

    # Optional overview Poincare (cheaper than full diagnostics path).
    _log("overview Poincare (8 lines)", log_path)
    t0 = time.perf_counter()
    mrx_sections, _, _, transit_counts = cmp._trace_mrx_poincare(
        seq,
        dof,
        map_fn,
        zeta_values,
        nlines=8,
        turns=int(poincare_turns),
        theta0=0.5,
        tol=1.0e-9,
    )
    # Boundary-only slices for axis limits.
    slices = []
    for zeta in zeta_values:
        logical, _, _ = cmp._poloidal_slice_points(float(zeta), 24, 48)
        mapped = np.asarray(
            jax.lax.map(
                map_fn,
                jnp.asarray(logical, dtype=jnp.float64),
                batch_size=512,
            )
        )
        R = np.sqrt(mapped[..., 0] ** 2 + mapped[..., 1] ** 2).reshape(24, 48)
        Z = mapped[..., 2].reshape(24, 48)
        slices.append({"zeta": float(zeta), "R": R, "Z": Z})
    empty_simsopt = {
        float(z): [np.zeros((0, 2)) for _ in range(8)] for z in zeta_values
    }
    poincare_path = cmp._plot_poincare_comparison(
        mrx_sections,
        empty_simsopt,
        slices,
        PLOT_DIR,
        filename=f"poincare_mrx_vs_simsopt_fem_{label}.png",
    )
    _log(f"poincare={poincare_path.name} ({time.perf_counter()-t0:.1f}s)", log_path)

    width_profile = cmp._island_width_profile(zoom_lines, seed_rho)
    record = {
        "ns": [int(v) for v in ns],
        "p": int(meta["p"]),
        "n2_dbc": int(seq.n2_dbc),
        "solve_seconds": prior.get("solve_seconds"),
        "solve_method": prior.get("solve_method", "resumed"),
        "smallest_eigenvalues": prior.get("smallest_eigenvalues", []),
        "iterative_info": prior.get("iterative_info"),
        "dense_verification": prior.get("dense_verification"),
        "algebraic_health": prior.get("algebraic_health"),
        "pointwise_u_scale_to_optimal": metrics["pointwise_u_scale_to_optimal"],
        "aligned_rel_l2": metrics["rel_l2_aligned"],
        "evaluation_nrho": 24,
        "evaluation_ntheta": 48,
        "dof_npy": str(dof_path),
        "poincare_file": str(poincare_path),
        "poincare_turns": int(poincare_turns),
        "mrx_completed_transits": transit_counts,
        "island_diagnostics": {
            "diagnostic_version": 2,
            "iota_profile": iota_profile,
            "resonances": resonances,
            "island_width_profile": width_profile,
            "resonant_island_widths": [],
            "max_island_width_rho": None,
            "max_detrended_island_width_rho": (
                max(
                    (
                        float(item["detrended_width_rho_q05_q95"])
                        for item in width_profile
                    ),
                    default=None,
                )
            ),
            "island_zoom_file": str(zoom_path),
            "trapped_separatrix": trapped,
            "resonant_normal_error": resonant_amplitudes,
            "max_resonant_normal_error_relative": (
                max(
                    (
                        float(item["normal_error_fourier_relative"])
                        for item in resonant_amplitudes
                    ),
                    default=None,
                )
            ),
            "lean_fast_diagnostics": True,
            "fourier_ntheta": int(fourier_n),
            "fourier_nzeta": int(fourier_n),
        },
    }
    return record


def _run_solve(
    *,
    ns: tuple[int, int, int],
    meta: dict,
    hcurl_mod,
    twoform_mod,
    simsopt_field,
    resume_records: list[dict],
    log_path: Path,
    dense_max_dofs: int,
    time_budget: float,
) -> dict:
    label = "x".join(str(v) for v in ns)
    _log(f"dense-verify solve {label}", log_path)
    base_dof = jnp.asarray(np.load(BASE_DOF), dtype=jnp.float64).reshape(-1)
    seq, map_jit, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    t0 = time.perf_counter()
    records = cmp._run_fem_resolution_sweep(
        meta,
        twoform_mod,
        hcurl_mod,
        simsopt_field,
        [0.0],
        [ns],
        base_seq=seq,
        base_dof=base_dof,
        base_map_fn=map_jit,
        base_dof_path=BASE_DOF,
        resume_records=resume_records,
        nrho=8,
        ntheta=16,
        time_budget_seconds=float(time_budget),
        dense_check_max_dofs=int(dense_max_dofs),
        solver_eps=1.0e-12,
        inner_tol=1.0e-8,
        iterative_maxiter=200,
        refine_current=False,
        refine_resolutions={ns},
        dense_verify_resolutions={ns},
        compute_health=False,
        output_dir=PLOT_DIR,
        poincare_config=None,
    )
    _log(f"solve wall {time.perf_counter()-t0:.1f}s", log_path)
    if not records:
        raise RuntimeError("solve produced no records")
    rec = records[0]
    _log(
        f"solved dofs={rec.get('n2_dbc')} method={rec.get('solve_method')} "
        f"L2={rec.get('aligned_rel_l2')}",
        log_path,
    )
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--mode", choices=("diagnostics", "solve", "solve+diagnostics"), default="diagnostics")
    parser.add_argument("--resume-json", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poincare-turns", type=int, default=300)
    parser.add_argument("--zoom-nrho", type=int, default=8)
    parser.add_argument("--zoom-phases", type=int, default=3)
    parser.add_argument("--fourier-n", type=int, default=32)
    parser.add_argument("--dense-max-dofs", type=int, default=12000)
    parser.add_argument("--time-budget", type=float, default=36000.0)
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ns = _parse_ns(args.target)
    _log(f"start target={args.target} mode={args.mode}", log_path)

    try:
        hcurl_mod = cmp._load_hcurl_module()
        twoform_mod = cmp._load_twoform_module()
        meta = cmp._load_meta_for_dof(BASE_DOF, k=2, meta_json=META_JSON)
        _log(f"base meta ns={meta['ns']}", log_path)
        _log("load SIMSOPT", log_path)
        simsopt_field, _ = cmp._load_simsopt_field(JSON_PATH)

        resume_records = cmp._load_fem_sweep_records(
            [p.expanduser().resolve() for p in args.resume_json]
        )
        prior_by_ns = {
            tuple(int(v) for v in r["ns"]): r for r in resume_records
        }

        record: dict
        if args.mode in ("solve", "solve+diagnostics"):
            record = _run_solve(
                ns=ns,
                meta=meta,
                hcurl_mod=hcurl_mod,
                twoform_mod=twoform_mod,
                simsopt_field=simsopt_field,
                resume_records=resume_records,
                log_path=log_path,
                dense_max_dofs=args.dense_max_dofs,
                time_budget=args.time_budget,
            )
            # Keep solve record as prior for optional diagnostics.
            prior_by_ns[ns] = record
            resume_records = list(prior_by_ns.values())
        else:
            if ns not in prior_by_ns:
                raise RuntimeError(
                    f"no resume record for {ns}; pass --resume-json with DOF"
                )
            record = dict(prior_by_ns[ns])

        if args.mode in ("diagnostics", "solve+diagnostics"):
            prior = prior_by_ns[ns]
            record = _run_diagnostics(
                ns=ns,
                prior=prior,
                meta=meta,
                hcurl_mod=hcurl_mod,
                simsopt_field=simsopt_field,
                log_path=log_path,
                poincare_turns=args.poincare_turns,
                zoom_nrho=args.zoom_nrho,
                zoom_phases=args.zoom_phases,
                fourier_n=args.fourier_n,
            )

        out = {
            "postprocessing": {
                "fem_resolution_sweep": [record],
                "plot_dir": str(PLOT_DIR),
                "lean_driver": {
                    "target": args.target,
                    "mode": args.mode,
                    "fast_diagnostics": True,
                },
            }
        }
        out_path = args.output_json.expanduser().resolve()
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        _log(f"wrote {out_path}", log_path)
        return 0
    except Exception:
        _log("FAILED\n" + traceback.format_exc(), log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
