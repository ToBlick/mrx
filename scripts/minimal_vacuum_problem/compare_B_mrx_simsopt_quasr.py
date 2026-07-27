#!/usr/bin/env python3
"""
Compare magnetic field **B** from a QUASR SIMSOPT vacuum coil setup (Biot–Savart) to **B**
stored in an MRX-style HDF5 volume (e.g. GVEC export or DESC precompute).

We evaluate SIMSOPT ``B`` at the **same physical Cartesian points** as the MRX volume.

**Preferred (default when present):** if the HDF5 has an ``eval_points`` dataset ``(N,3)`` in the
usual MRX convention — ``[ρ, θ/(2π), ζ_rad·nfp/(2π)]`` row-aligned with ``R``, ``Z`` — we build
``(X,Y,Z)`` from logical ζ:

    ζ_logic = eval_points[:,2]   (optionally mirrored if ``flip_zeta``, see below)
    X = R cos((2π/NFP) ζ_logic),   Y = y_sign · R sin((2π/NFP) ζ_logic),   Z = Z.

``y_sign=-1`` matches :func:`mrx.mappings.stellarator_map`. For GVEC Cartesian ``B`` vs
SIMSOPT coils use ``y_sign=+1`` (``Y = +R sin φ``); this is auto-selected when the HDF5
has ``gvec_rz_from_xyz_cylindrical`` (override with ``--y-plus-rsin`` / ``--y-minus-rsin``).

**Fallback:** if there is no ``eval_points`` dataset, reconstruct toroidal angle from the tensor
shape (``precomputed_*`` attrs) and C-order ``(ρ,θ,ζ)`` indexing: ``ζ = (i_z + ½)/n_zeta`` with
``--zeta-centers`` (default), or ``i_z/n_zeta`` with ``--zeta-edges``, optionally spanning one
field period with ``--one-field-period-toroidal``.

Metrics (per component and vector **L²** on discrete points, equal weights):

- ``l2_diff_xyz``: sqrt(mean(sum((B_mrx − B_bs)^2, axis=1)))
- ``rel_l2``: l2_diff / sqrt(mean(sum(B_mrx^2, axis=1)))

**Environment:** SIMSOPT + JAX compat — use an env where ``from simsopt._core import load`` and
``BiotSavart`` work (e.g. ``simsopt_vmec`` or your QUASR env). MRX only needs ``numpy``, ``h5py``.

Examples
--------
::

    conda activate simsopt_vmec  # or env with simsopt + your QUASR JSON

    python scripts/wip/compare_B_mrx_simsopt_quasr.py \\
      --json ~/Downloads/serial0044790.json \\
      --mrx-h5 /scratch/js11789/mrx/scripts/wip/quasr_new_0044970_mrx.h5 \\
      --nfp 3

If ``B`` in the HDF5 is cylindrical ``(B_R, B_phi, B_Z)`` (see ``B_convention`` attr), pass
``--b-is-cylindrical``. GVEC ``export_to_mrx_h5`` writes Cartesian ``(B_x,B_y,B_z)``.

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _infer_grid(h5: Any) -> tuple[int, int, int, int]:
    """Return ``nfp`` and ``(nr, ntheta, nzeta)`` from HDF5 attrs or dataset shapes.

    Do **not** use ``with h5:`` here: an open :class:`h5py.File` would be closed on exit,
    breaking the caller's context manager.
    """
    nfp = int(h5.attrs.get("nfp", h5.attrs.get("NFP", 3)))
    nr = int(h5.attrs.get("precomputed_nr", h5.attrs.get("n_rho", 0)))
    nt = int(h5.attrs.get("precomputed_ntheta", h5.attrs.get("n_theta", 0)))
    nz = int(h5.attrs.get("precomputed_nzeta", h5.attrs.get("n_zeta", 0)))
    if nr <= 0 or nt <= 0 or nz <= 0:
        n = int(h5["R"].shape[0])
        # cubic fallback
        n3 = int(round(n ** (1.0 / 3.0)))
        if n3 * n3 * n3 == n:
            nr = nt = nz = n3
        else:
            raise ValueError(
                "Could not infer (nr,ntheta,nzeta); set precomputed_nr/ntheta/nzeta on HDF5 "
                "or use a full tensor-product export."
            )
    return nfp, nr, nt, nz


def _flat_indices(nr: int, nt: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C-order ravel: zeta (last) varies fastest."""
    N = nr * nt * nz
    idx = np.arange(N, dtype=np.int64)
    iz = idx % nz
    it = (idx // nz) % nt
    ir = idx // (nz * nt)
    return ir, it, iz


def apply_flip_zeta_logical(z_log: np.ndarray, flip_zeta: bool) -> np.ndarray:
    """
    Mirror ``ζ`` the same way as :func:`mrx.mappings.stellarator_map` / ``gvec_jax_map``.

    For third logical coordinate in ``[0, 1]`` (one field period), use ``ζ ← 1 − ζ``.
    For ``[0, NFP]`` (full torus labeling ``ζ_rad·NFP/(2π)``), use ``ζ ← ζ_max + ζ_min − ζ``.
    """
    z_log = np.asarray(z_log, dtype=np.float64)
    if not flip_zeta:
        return z_log
    zmax = float(np.max(z_log))
    zmin = float(np.min(z_log))
    if zmax <= 1.0 + 1e-9:
        return 1.0 - z_log
    return (zmax + zmin) - z_log


def resolve_lab_y_sign(
    *,
    y_sign: float | None = None,
    gvec_rz_from_xyz_cylindrical: bool = False,
    b_convention_attr: str = "",
) -> float:
    """
    Choose the lab-frame ``Y`` sign for ``X = R cos φ``, ``Y = y_sign · R sin φ``.

    Parameters
    ----------
    y_sign
        Explicit override. ``+1`` is the right-handed cylindrical / GVEC / SIMSOPT
        coil frame (``Y = +R sin φ``). ``-1`` matches :func:`mrx.mappings.stellarator_map`
        (``Y = −R sin φ``). ``None`` selects from HDF5 metadata.
    gvec_rz_from_xyz_cylindrical
        HDF5 attr set by GVEC ``export_to_mrx_h5`` when ``R,Z`` come from a
        Cartesian ``(x,y,z)`` with ``y = +R sin φ``.
    b_convention_attr
        Lowercased ``B_convention`` string; GVEC Cartesian tags also imply ``+1``.

    Returns
    -------
    float
        ``+1.0`` or ``-1.0``.
    """
    if y_sign is not None:
        s = float(y_sign)
        if abs(abs(s) - 1.0) > 1e-12:
            raise ValueError(f"y_sign must be ±1, got {y_sign}")
        return 1.0 if s > 0.0 else -1.0
    conv = str(b_convention_attr or "").lower()
    if gvec_rz_from_xyz_cylindrical or "cartesian_bx_by_bz" in conv:
        return 1.0
    return -1.0


def xyz_from_mrx_eval_points(
    R: np.ndarray,
    Z: np.ndarray,
    eval_points: np.ndarray,
    nfp: int,
    *,
    flip_zeta: bool,
    y_sign: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cartesian samples from MRX ``R``, ``Z``, and ``eval_points`` column 2 (logical ζ).

    Row ``i``: major radius ``R[i]``, height ``Z[i]``, third logical ``ζ = eval_points[i, 2]``.

    Lab frame
    ---------
    ``φ = (2π / NFP) · ζ`` (after optional ``flip_zeta``), then

    - ``X = R cos φ``
    - ``Y = y_sign · R sin φ`` with default ``y_sign=-1`` (``stellarator_map``)
    - ``Z`` unchanged

    Use ``y_sign=+1`` when comparing to GVEC-exported Cartesian ``B`` or SIMSOPT
    Biot–Savart on QUASR coils (both use the right-handed ``Y = +R sin φ`` frame).
    See :func:`resolve_lab_y_sign`.
    """
    R = np.asarray(R, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64).ravel()
    ep = np.asarray(eval_points, dtype=np.float64)
    if ep.ndim != 2 or ep.shape[1] < 3:
        raise ValueError(f"eval_points must be (N,3+), got {ep.shape}")
    if ep.shape[0] != R.size or Z.size != R.size:
        raise ValueError(f"eval_points length {ep.shape[0]} vs R,Z length {R.size}")
    y_s = resolve_lab_y_sign(y_sign=y_sign)
    z_log = apply_flip_zeta_logical(ep[:, 2], flip_zeta)
    pi_nfp = 2.0 * np.pi / float(nfp)
    ang = pi_nfp * z_log
    X = R * np.cos(ang)
    Y = y_s * R * np.sin(ang)
    return X, Y, Z


def xyz_from_rz_grid(
    R: np.ndarray,
    Z: np.ndarray,
    nr: int,
    nt: int,
    nz: int,
    nfp: int,
    *,
    zeta_centers: bool,
    one_field_period: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct Cartesian ``X,Y,Z`` from cylindrical ``R,Z`` on a uniform toroidal grid.

    Uses the same lab-frame convention as ``stellarator_map``: ``X = R cos φ``, ``Y = −R sin φ``.

    Toroidal angle ``φ`` for slice index ``iz`` is sampled uniformly:

    - ``one_field_period=False`` (default): ``φ ∈ [0, 2π)`` with ``nz`` points (full torus).
    - ``one_field_period=True``: ``φ ∈ [0, 2π/NFP)`` with ``nz`` points (one field period).
    """
    R = np.asarray(R, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64).ravel()
    N = nr * nt * nz
    if R.size != N or Z.size != N:
        raise ValueError(f"Expected R,Z length {N}, got {R.size}, {Z.size}")

    _, _, iz = _flat_indices(nr, nt, nz)
    span = (2.0 * np.pi / float(nfp)) if one_field_period else (2.0 * np.pi)
    if zeta_centers:
        phi_line = (np.arange(nz, dtype=np.float64) + 0.5) / float(nz) * span
    else:
        phi_line = np.arange(nz, dtype=np.float64) / float(nz) * span
    phi = phi_line[iz]
    X = R * np.cos(phi)
    Y = -R * np.sin(phi)
    return X, Y, Z


def cyl_B_to_xyz(
    BR: np.ndarray,
    Bphi: np.ndarray,
    BZ: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Convert ``(B_R,B_phi,B_Z)`` to ``(B_x,B_y,B_z)`` using φ = atan2(Y,X)."""
    phi = np.arctan2(Y, X)
    Bx = BR * np.cos(phi) - Bphi * np.sin(phi)
    By = BR * np.sin(phi) + Bphi * np.cos(phi)
    return np.stack([Bx, By, BZ], axis=1)


def xyz_B_to_cyl(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Inverse of :func:`cyl_B_to_xyz` at fixed ``(X,Y,Z)`` (φ = atan2(Y,X))."""
    phi = np.arctan2(Y, X)
    BR = Bx * np.cos(phi) + By * np.sin(phi)
    Bphi = -Bx * np.sin(phi) + By * np.cos(phi)
    return np.stack([BR, Bphi, Bz], axis=1)


def load_mrx_b_xyz(
    path: Path, *, b_is_cylindrical: bool | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict[str, Any], np.ndarray | None]:
    """Load ``R,Z,B`` and optional ``eval_points``; meta includes grid and ``flip_zeta`` attr."""
    import h5py

    meta: dict[str, Any] = {}
    ep: np.ndarray | None = None
    with h5py.File(path, "r") as f:
        if "R" not in f or "Z" not in f or "B" not in f:
            raise ValueError(f"{path} needs datasets R, Z, B")
        R = np.asarray(f["R"][:])
        Z = np.asarray(f["Z"][:])
        B = np.asarray(f["B"][:])
        conv = str(f.attrs.get("B_convention", "")).lower()
        gvec_xyz = bool(f.attrs.get("gvec_rz_from_xyz_cylindrical", False))
        meta["B_convention_attr"] = conv or "(unset)"
        meta["gvec_rz_from_xyz_cylindrical"] = gvec_xyz
        meta["flip_zeta_attr"] = bool(f.attrs.get("flip_zeta", False))

        if B.ndim == 1:
            raise ValueError("B must be (N,3)")
        if B.shape[-1] != 3:
            raise ValueError(f"B shape {B.shape} expected (N,3)")

        is_cyl = b_is_cylindrical
        if is_cyl is None:
            is_cyl = "cyl" in conv or "b_r" in conv or "rpz" in conv

        nfp, nr, nt, nz = _infer_grid(f)
        meta.update({"nfp": nfp, "nr": nr, "ntheta": nt, "nzeta": nz})

        if "eval_points" in f:
            ep = np.asarray(f["eval_points"][:])
            meta["has_eval_points"] = True
        else:
            meta["has_eval_points"] = False

    return R, Z, B, bool(is_cyl), meta, ep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, required=True, help="QUASR SIMSOPT serial JSON (surfaces, coils)")
    p.add_argument("--mrx-h5", type=Path, required=True, help="MRX volume HDF5 with R, Z, B, eval_points attrs")
    p.add_argument("--nfp", type=int, default=None, help="Override field periods (default: from HDF5)")
    p.add_argument(
        "--b-is-cylindrical",
        action="store_true",
        help="B is (B_R, B_phi, B_Z). Default: infer from B_convention attr.",
    )
    p.add_argument(
        "--b-is-cartesian",
        action="store_true",
        help="Force B as (Bx, By, Bz) (ignore B_convention heuristic)",
    )
    p.add_argument(
        "--zeta-centers",
        action="store_true",
        default=True,
        help="Use (iz+0.5)/nz for ζ_logical (default)",
    )
    p.add_argument("--zeta-edges", action="store_true", help="Use iz/nz for ζ_logical instead")
    p.add_argument(
        "--one-field-period-toroidal",
        action="store_true",
        help="Assume nz toroidal samples span one field period [0,2π/NFP) instead of full [0,2π)",
    )
    p.add_argument(
        "--force-grid-xyz",
        action="store_true",
        help="Ignore eval_points and reconstruct X,Y,Z from grid indices (fallback path)",
    )
    p.add_argument(
        "--flip-zeta",
        action="store_true",
        help="Apply stellarator_map flip on eval_points[:,2] before trig (overrides HDF5 attr)",
    )
    p.add_argument(
        "--no-flip-zeta",
        action="store_true",
        help="Do not flip ζ for eval_points reconstruction (overrides HDF5 attr)",
    )
    p.add_argument(
        "--y-plus-rsin",
        action="store_true",
        help="Lab frame Y=+R sin φ (GVEC / SIMSOPT coil frame)",
    )
    p.add_argument(
        "--y-minus-rsin",
        action="store_true",
        help="Lab frame Y=-R sin φ (stellarator_map); default unless GVEC attrs say otherwise",
    )
    p.add_argument("-o", "--output-json", type=Path, default=None, help="Write metrics JSON")
    args = p.parse_args()

    json_path = args.json.expanduser().resolve()
    h5_path = args.mrx_h5.expanduser().resolve()
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")
    if not h5_path.is_file():
        raise SystemExit(f"HDF5 not found: {h5_path}")

    if args.flip_zeta and args.no_flip_zeta:
        raise SystemExit("Use only one of --flip-zeta / --no-flip-zeta")
    if args.b_is_cartesian and args.b_is_cylindrical:
        raise SystemExit("Use only one of --b-is-cartesian / --b-is-cylindrical")
    b_cyl: bool | None
    if args.b_is_cartesian:
        b_cyl = False
    elif args.b_is_cylindrical:
        b_cyl = True
    else:
        b_cyl = None

    R, Z, B_raw, inferred_cyl, meta, eval_points = load_mrx_b_xyz(h5_path, b_is_cylindrical=b_cyl)
    if args.nfp is not None:
        meta["nfp"] = int(args.nfp)

    zeta_centers = not bool(args.zeta_edges)

    nfp = int(meta["nfp"])
    nr, nt, nz = int(meta["nr"]), int(meta["ntheta"]), int(meta["nzeta"])

    use_eval_xyz = (
        not bool(args.force_grid_xyz)
        and eval_points is not None
        and int(eval_points.shape[0]) == int(R.shape[0])
    )
    if eval_points is not None and int(eval_points.shape[0]) != int(R.shape[0]):
        raise SystemExit(
            f"eval_points length {eval_points.shape[0]} does not match R length {R.shape[0]}"
        )

    if args.flip_zeta:
        flip_zeta_eff = True
    elif args.no_flip_zeta:
        flip_zeta_eff = False
    else:
        flip_zeta_eff = bool(meta.get("flip_zeta_attr", False))

    if args.y_plus_rsin and args.y_minus_rsin:
        raise SystemExit("Use only one of --y-plus-rsin / --y-minus-rsin")
    y_sign_override: float | None
    if args.y_plus_rsin:
        y_sign_override = 1.0
    elif args.y_minus_rsin:
        y_sign_override = -1.0
    else:
        y_sign_override = None
    y_sign_eff = resolve_lab_y_sign(
        y_sign=y_sign_override,
        gvec_rz_from_xyz_cylindrical=bool(meta.get("gvec_rz_from_xyz_cylindrical", False)),
        b_convention_attr=str(meta.get("B_convention_attr", "")),
    )

    if use_eval_xyz:
        X, Y, Zcoord = xyz_from_mrx_eval_points(
            R, Z, eval_points, nfp, flip_zeta=flip_zeta_eff, y_sign=y_sign_eff
        )
        meta["positions_source"] = "eval_points"
        meta["flip_zeta_used"] = flip_zeta_eff
        meta["y_sign_used"] = y_sign_eff
    else:
        if not use_eval_xyz:
            if bool(args.force_grid_xyz):
                reason = "--force-grid-xyz"
            elif eval_points is None:
                reason = "no eval_points dataset"
            else:
                reason = (
                    f"eval_points length {eval_points.shape[0]} != R length {R.shape[0]}"
                )
            print(
                f"  Note: {reason}; using grid-index φ reconstruction.",
                file=sys.stderr,
            )
        X, Y, Zcoord = xyz_from_rz_grid(
            R,
            Z,
            nr,
            nt,
            nz,
            nfp,
            zeta_centers=zeta_centers,
            one_field_period=bool(args.one_field_period_toroidal),
        )
        meta["positions_source"] = "grid_flat_index"
        meta["flip_zeta_used"] = None

    if inferred_cyl:
        BR, Bphi, BZ = B_raw[:, 0], B_raw[:, 1], B_raw[:, 2]
        B_mrx = cyl_B_to_xyz(BR, Bphi, BZ, X, Y)
        meta["B_interpretation"] = "cylindrical -> Cartesian"
    else:
        B_mrx = np.asarray(B_raw, dtype=np.float64)
        meta["B_interpretation"] = "Cartesian (Bx,By,Bz)"

    meta["zeta_logical"] = "centers" if zeta_centers else "edges"
    meta["one_field_period_toroidal"] = bool(args.one_field_period_toroidal)
    meta["force_grid_xyz"] = bool(args.force_grid_xyz)

    pts = np.stack([X, Y, Zcoord], axis=1)

    from mrx.compat_simsopt_jax import ensure_jax_config_submodule

    ensure_jax_config_submodule()
    from simsopt._core import load
    from simsopt.field import BiotSavart
    loaded = load(str(json_path))
    if not isinstance(loaded, (list, tuple)) or len(loaded) != 2:
        raise RuntimeError(f"Expected [surfaces, coils] from load(); got {type(loaded)}")
    _surfaces, coils = loaded[0], loaded[1]

    bs = BiotSavart(coils)
    bs.set_points(pts)
    B_bs = np.asarray(bs.B(), dtype=np.float64)

    diff = B_mrx - B_bs
    l2_per = np.sqrt(np.mean(diff**2, axis=0))
    l2_vec = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    l2_mrx = float(np.sqrt(np.mean(np.sum(B_mrx**2, axis=1))))
    rel = l2_vec / l2_mrx if l2_mrx > 0 else float("nan")

    print("=== B: MRX vs SIMSOPT BiotSavart (same Cartesian sample points) ===")
    print(f"  JSON:      {json_path}")
    print(f"  MRX H5:    {h5_path}")
    print(f"  N points:  {pts.shape[0]}  grid {nr}x{nt}x{nz}  nfp={nfp}")
    print(f"  XYZ from:  {meta['positions_source']}", end="")
    if meta["positions_source"] == "eval_points":
        print(
            f"  flip_zeta={meta['flip_zeta_used']}  y_sign={meta.get('y_sign_used')} "
            f"(--flip-zeta/--no-flip-zeta, --y-plus-rsin/--y-minus-rsin)"
        )
    else:
        cen = "(iz+0.5)/nz" if zeta_centers else "iz/nz"
        print(
            f"  |  φ: {'one FP [0,2π/NFP)' if args.one_field_period_toroidal else 'full [0,2π)'}  centers={cen}"
        )
    print(f"  B MRX:     {meta['B_interpretation']}  (attr B_convention={meta['B_convention_attr']})")
    print(f"  |B|_mrx  min/max: {np.linalg.norm(B_mrx, axis=1).min():.6g} / {np.linalg.norm(B_mrx, axis=1).max():.6g} T")
    print(f"  |B|_bs   min/max: {np.linalg.norm(B_bs, axis=1).min():.6g} / {np.linalg.norm(B_bs, axis=1).max():.6g} T")
    print(f"  L2 diff (Bx,By,Bz) per component RMS: {l2_per}")
    print(f"  L2 diff ||B_mrx - B_bs||_2 (volume mean sqrt): {l2_vec:.6e} T")
    print(f"  Relative L2 (normalized by ||B_mrx||_2):          {rel:.6e}")

    out: dict[str, Any] = {
        "json": str(json_path),
        "mrx_h5": str(h5_path),
        "n_points": int(pts.shape[0]),
        "grid": {"nr": nr, "ntheta": nt, "nzeta": nz, "nfp": nfp},
        "l2_diff_per_component_rms": l2_per.tolist(),
        "l2_diff_vector": l2_vec,
        "rel_l2": rel,
        "meta": meta,
    }

    if args.output_json:
        outp = args.output_json.expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2))
        print(f"  Wrote {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
