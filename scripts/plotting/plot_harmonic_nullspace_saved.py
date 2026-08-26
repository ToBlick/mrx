#!/usr/bin/env python3
"""
Main driver for harmonic form plotting. Plots saved harmonic nullspace DOFs (without re-solving).

Inputs (same files the harmonic solve writes):

Examples: 

  hcurl_k1_nullspace_dof.npy  +  hcurl_nullspace_meta.json
  hodge_k2_dbc_nullspace_dof.npy  +  hodge_k2_nullspace_meta.json

Writes three PNGs per form degree (logical slice, poloidal R–Z, outer surface).

Example (k=2):

  python scripts/plotting/plot_harmonic_nullspace_saved.py \\
    --dof-npy .../hodge_k2_dbc_nullspace_dof.npy \\
    --k 2 \\
    -o .../plots_dir
    
    
    Examples so far are to be added like so -- Ju

Example (k=1):

  python scripts/plotting/plot_harmonic_nullspace_saved.py \\
    --dof-npy .../hcurl_k1_nullspace_dof.npy \\
    -o .../plots_dir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.plotting.harmonic_nullspace_geometry import (
    infer_form_degree,
    load_dof_vector,
    load_meta,
    rebuild_sequence_from_meta,
)
from scripts.plotting.harmonic_nullspace_plots import (
    run_k1_nullspace_plots,
    run_k2_nullspace_plots,
)


def _parse_zetas(text: str) -> tuple[float, ...]:
    zetas = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if not zetas:
        raise SystemExit("--plot-zetas produced no values")
    return zetas


def plot_harmonic_nullspace_saved(
    dof_npy: Path,
    output_dir: Path,
    *,
    k: int | None = None,
    meta_json: Path | None = None,
    tol: float = 1e-9,
    maxiter: int = 20,
    strict_jacobian: bool = False,
    zetas: tuple[float, ...] = (0.0, 0.25, 0.5),
    plot_clean: bool = False,
    plot_magnitude_logical: bool = False,
    plot_cut_nx: int = 48,
    plot_cut_ny: int = 64,
    quiver_stride_r: int = 3,
    quiver_stride_t: int = 4,
    surface_ntheta: int = 48,
    surface_nzeta: int = 48,
    surface_quiver_stride_theta: int = 4,
    surface_quiver_stride_zeta: int = 4,
    surface_quiver_length: float = 0.15,
    surface_quiver_offset: float = 0.0,
    surface_rho: float = 1.0 - 1e-5,
    surface_half_cut: bool = False,
    surface_axis_full: bool = False,
    write_meta_copy: bool = True,
) -> dict[str, Any]:
    """
    Rebuild geometry from meta and write standard harmonic nullspace PNGs.

    Returns a small summary dict (paths, k, nfp, m2_norm).
    """
    dof_npy = dof_npy.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    k_eff = infer_form_degree(dof_npy, k=k)
    meta = load_meta(meta_json, dof_npy, k=k_eff)
    print(f"Replot k={k_eff} from {dof_npy.name}  map_mode={meta.get('map_mode')}", flush=True)

    seq, _map_jit, map_raw, nfp = rebuild_sequence_from_meta(
        meta,
        tol=tol,
        maxiter=maxiter,
        strict_jacobian=strict_jacobian,
    )
    v = load_dof_vector(dof_npy, seq, k=k_eff)
    try:
        m2_norm = float(seq.l2_norm(v, k_eff, dirichlet=(k_eff == 2)))
    except ValueError:
        m2_norm = float("nan")

    plot_kwargs = dict(
        seq=seq,
        v=v,
        map_raw=map_raw,
        nfp=int(nfp),
        out_dir=output_dir,
        zetas=zetas,
        nx=int(plot_cut_nx),
        ny=int(plot_cut_ny),
        quiver_stride_r=int(quiver_stride_r),
        quiver_stride_t=int(quiver_stride_t),
        surface_ntheta=int(surface_ntheta),
        surface_nzeta=int(surface_nzeta),
        surface_quiver_stride_theta=int(surface_quiver_stride_theta),
        surface_quiver_stride_zeta=int(surface_quiver_stride_zeta),
        surface_quiver_length=float(surface_quiver_length),
        surface_quiver_offset=float(surface_quiver_offset),
        surface_rho=float(surface_rho),
        half_cut=bool(surface_half_cut),
        axis_full=bool(surface_axis_full),
        clean=bool(plot_clean),
        magnitude_logical=bool(plot_magnitude_logical),
    )

    if k_eff == 2:
        run_k2_nullspace_plots(**plot_kwargs)
        prefix = "hodge_k2_null"
    else:
        run_k1_nullspace_plots(**plot_kwargs)
        prefix = "hcurl_null"

    summary = {
        "dof_npy": str(dof_npy),
        "output_dir": str(output_dir),
        "k": k_eff,
        "nfp": int(nfp),
        "m2_norm": m2_norm,
        "plot_prefix": prefix,
        "map_mode": meta.get("map_mode"),
    }
    if write_meta_copy:
        sidecar = output_dir / "plot_harmonic_nullspace_summary.json"
        sidecar.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {sidecar}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dof-npy", type=Path, required=True, help="Saved harmonic DOF .npy")
    ap.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="Meta JSON (default: hcurl_nullspace_meta.json or hodge_k2_nullspace_meta.json)",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: directory containing --dof-npy)",
    )
    ap.add_argument(
        "--k",
        type=int,
        choices=(1, 2),
        default=None,
        help="Form degree (default: infer from DOF filename)",
    )
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--strict-jacobian", action="store_true")
    ap.add_argument("--plot-clean", action="store_true")
    ap.add_argument("--plot-magnitude-logical", action="store_true")
    ap.add_argument("--plot-zetas", type=str, default="0.0,0.25,0.5")
    ap.add_argument("--plot-cut-nx", type=int, default=48)
    ap.add_argument("--plot-cut-ny", type=int, default=64)
    ap.add_argument("--plot-quiver-stride-r", type=int, default=3)
    ap.add_argument("--plot-quiver-stride-t", type=int, default=4)
    ap.add_argument("--surface-ntheta", type=int, default=48)
    ap.add_argument("--surface-nzeta", type=int, default=48)
    ap.add_argument("--surface-quiver-stride-theta", type=int, default=4)
    ap.add_argument("--surface-quiver-stride-zeta", type=int, default=4)
    ap.add_argument("--surface-quiver-length", type=float, default=0.15)
    ap.add_argument("--surface-quiver-offset", type=float, default=0.0)
    ap.add_argument("--surface-rho", type=float, default=1.0 - 1e-5)
    ap.add_argument("--surface-half-cut", action="store_true")
    ap.add_argument("--surface-axis-full", action="store_true")
    args = ap.parse_args()

    out = args.output_dir
    if out is None:
        out = args.dof_npy.expanduser().resolve().parent

    plot_harmonic_nullspace_saved(
        args.dof_npy,
        out,
        k=args.k,
        meta_json=args.meta_json,
        tol=args.tol,
        maxiter=args.maxiter,
        strict_jacobian=args.strict_jacobian,
        zetas=_parse_zetas(args.plot_zetas),
        plot_clean=args.plot_clean,
        plot_magnitude_logical=args.plot_magnitude_logical,
        plot_cut_nx=args.plot_cut_nx,
        plot_cut_ny=args.plot_cut_ny,
        quiver_stride_r=args.plot_quiver_stride_r,
        quiver_stride_t=args.plot_quiver_stride_t,
        surface_ntheta=args.surface_ntheta,
        surface_nzeta=args.surface_nzeta,
        surface_quiver_stride_theta=args.surface_quiver_stride_theta,
        surface_quiver_stride_zeta=args.surface_quiver_stride_zeta,
        surface_quiver_length=args.surface_quiver_length,
        surface_quiver_offset=args.surface_quiver_offset,
        surface_rho=args.surface_rho,
        surface_half_cut=args.surface_half_cut,
        surface_axis_full=args.surface_axis_full,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
