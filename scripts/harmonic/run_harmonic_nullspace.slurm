#!/bin/bash
#SBATCH --job-name=harmonic_null
#SBATCH --output=harmonic_null_%j.out
#SBATCH --error=harmonic_null_%j.err
#SBATCH --account=torch_pr_292_general #Fill in with your specific account 
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#
# Harmonic nullspace solve on greville-prod (H5 or analytic geometry). Self-contained. 
#
# k=2 Hodge 2-form (e.g. QUASR 9983, ns 8×16×8):
#   cd /scratch/$USER/mrx-greville-harmonic
#   export MRX_ROOT=$PWD DEVICE=9983 HARMONIC_K=2
#   sbatch --export=ALL scripts/harmonic/run_harmonic_nullspace.slurm
#
# k=1 H(curl) 1-form:
#   export DEVICE=9983 HARMONIC_K=1 NS_R=8 NS_THETA=16 NS_ZETA=8
#   sbatch --export=ALL scripts/harmonic/run_harmonic_nullspace.slurm
#
# Optional: PLOTS=1  FLIP_ZETA=1  SPARSE=0  (dense k=2)
# JAX: defaults to CPU (JAX_PLATFORMS=cpu).
# =============================================================================

set -euo pipefail

HARMONIC_K="${HARMONIC_K:-2}"
MAP_MODE="${MAP_MODE:-h5}"
MAP_GEOMETRY_H5="${MAP_GEOMETRY_H5:-}"
MAP_KIND="${MAP_KIND:-cerfon}"
NFP="${NFP:-}"
DEVICE="${DEVICE:-}"
OUT_DIR="${OUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NS_R="${NS_R:-}"
NS_THETA="${NS_THETA:-}"
NS_ZETA="${NS_ZETA:-}"
FEM_P="${FEM_P:-3}"
FLIP_ZETA="${FLIP_ZETA:-0}"
SPARSE="${SPARSE:-1}"
PLOTS="${PLOTS:-0}"
BETTI="${BETTI:-1 1 0 0}"
NULLSPACE_EPS="${NULLSPACE_EPS:-1e-4}"
NULLSPACE_MAXITER="${NULLSPACE_MAXITER:-100}"
INNER_TOL="${INNER_TOL:-1e-6}"
TOL="${TOL:-1e-9}"
STALL_RATIO="${STALL_RATIO:-0.9}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# --- MRX repo root ------------------------------------------------------------
MRX_ROOT="${MRX_ROOT:-}"
if [[ -n "${MRX_ROOT}" ]] && [[ -f "${MRX_ROOT}/mrx/nullspace.py" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/mrx/nullspace.py" ]]; then
  MRX_ROOT="${SLURM_SUBMIT_DIR}"
else
  for p in "/scratch/${USER}/mrx-greville-harmonic" "/scratch/${USER}/mrx" "/scratch/js11789/mrx" "${HOME}/mrx"; do
    if [[ -f "${p}/mrx/nullspace.py" ]]; then
      MRX_ROOT="${p}"
      break
    fi
  done
fi
if [[ -z "${MRX_ROOT}" ]] || [[ ! -f "${MRX_ROOT}/mrx/nullspace.py" ]]; then
  echo "ERROR: Set MRX_ROOT to repo root containing mrx/nullspace.py"
  exit 1
fi
export MRX_ROOT

DATA_ROOT="${DATA_ROOT:-${MRX_ROOT}}"

# --- DEVICE presets (QUASR MRX volume H5 paths) -------------------------------
case "${DEVICE}" in
  9983|0009983)
    MAP_GEOMETRY_H5="${MAP_GEOMETRY_H5:-${DATA_ROOT}/data/quasr0009983_mrx_nr8_nt16_nz8.h5}"
    NFP="${NFP:-2}"
    DEVICE_TAG="quasr0009983"
    NS_R="${NS_R:-8}"
    NS_THETA="${NS_THETA:-16}"
    NS_ZETA="${NS_ZETA:-8}"
    ;;
  44970|0044970)
    MAP_GEOMETRY_H5="${MAP_GEOMETRY_H5:-${DATA_ROOT}/data/quasr0044970_mrx_n8_16_8.h5}"
    NFP="${NFP:-3}"
    DEVICE_TAG="quasr0044970"
    NS_R="${NS_R:-8}"
    NS_THETA="${NS_THETA:-16}"
    NS_ZETA="${NS_ZETA:-8}"
    ;;
  40379|0040379)
    MAP_GEOMETRY_H5="${MAP_GEOMETRY_H5:-${DATA_ROOT}/data/quasr0040379_mrx_n8_16_8.h5}"
    NFP="${NFP:-3}"
    DEVICE_TAG="quasr0040379"
    NS_R="${NS_R:-8}"
    NS_THETA="${NS_THETA:-16}"
    NS_ZETA="${NS_ZETA:-8}"
    ;;
  65530|0065530)
    MAP_GEOMETRY_H5="${MAP_GEOMETRY_H5:-${DATA_ROOT}/data/quasr0065530_mrx_n8_16_8.h5}"
    NFP="${NFP:-3}"
    DEVICE_TAG="quasr0065530"
    NS_R="${NS_R:-8}"
    NS_THETA="${NS_THETA:-16}"
    NS_ZETA="${NS_ZETA:-8}"
    ;;
  "")
    DEVICE_TAG=""
    ;;
  *)
    DEVICE_TAG="quasr${DEVICE}"
    ;;
esac

if [[ "${MAP_MODE}" == "h5" ]]; then
  if [[ -z "${MAP_GEOMETRY_H5}" ]]; then
    echo "ERROR: Set MAP_GEOMETRY_H5 or export DEVICE=9983|44970|40379|65530 for defaults."
    exit 1
  fi
  if [[ ! -f "${MAP_GEOMETRY_H5}" ]]; then
    echo "ERROR: MAP_GEOMETRY_H5 not found: ${MAP_GEOMETRY_H5}"
    exit 1
  fi
fi

if [[ -z "${NS_R}" ]] || [[ -z "${NS_THETA}" ]] || [[ -z "${NS_ZETA}" ]]; then
  echo "ERROR: Set NS_R, NS_THETA, NS_ZETA (or use a DEVICE preset)."
  exit 1
fi

if [[ -z "${OUT_DIR}" ]] && [[ -n "${DEVICE_TAG}" ]]; then
  if [[ "${HARMONIC_K}" == "2" ]]; then
    OUT_DIR="${MRX_ROOT}/script_outputs/hodge_k2_${DEVICE_TAG}_ns${NS_R}_${NS_THETA}_${NS_ZETA}_p${FEM_P}"
  else
    OUT_DIR="${MRX_ROOT}/script_outputs/hcurl_k1_${DEVICE_TAG}_ns${NS_R}_${NS_THETA}_${NS_ZETA}_p${FEM_P}"
  fi
fi
export OUT_DIR MAP_MODE MAP_GEOMETRY_H5 MAP_KIND NFP HARMONIC_K NS_R NS_THETA NS_ZETA FEM_P
export FLIP_ZETA SPARSE PLOTS BETTI NULLSPACE_EPS NULLSPACE_MAXITER INNER_TOL TOL STALL_RATIO

# EXTRA_ARGS: --plots and --flip-zeta for backward compatibility
case " ${EXTRA_ARGS} " in
  *" --plots "*) PLOTS=1 ;;
esac

echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "MRX_ROOT:     ${MRX_ROOT}"
echo "HARMONIC_K:   ${HARMONIC_K}"
echo "MAP_MODE:     ${MAP_MODE}"
echo "MAP_GEOMETRY_H5: ${MAP_GEOMETRY_H5:-<none>}"
echo "NFP:          ${NFP:-<from H5>}"
echo "NS:           ${NS_R}x${NS_THETA}x${NS_ZETA}  p=${FEM_P}"
echo "OUT_DIR:      ${OUT_DIR:-<required>}"
echo "SPARSE:       ${SPARSE}"
echo "PLOTS:        ${PLOTS}"
echo "FLIP_ZETA:    ${FLIP_ZETA}"
echo "JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo "============================================================================"

if [[ -z "${OUT_DIR}" ]]; then
  echo "ERROR: Set OUT_DIR or use a DEVICE preset."
  exit 1
fi

# --- conda (NYU Torch style) --------------------------------------------------
module purge
module load anaconda3/2025.06
if [[ -f "/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh" ]]; then
  source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
eval "$(conda shell.bash hook)"

MRX_ENV_PATH=""
for _base in "/scratch/${USER}/.conda/envs" "/scratch/js11789/.conda/envs" "${HOME}/.conda/envs"; do
  if [[ -d "${_base}/mrx" ]]; then
    MRX_ENV_PATH="${_base}/mrx"
    break
  fi
done
if [[ -z "${MRX_ENV_PATH}" ]] || [[ ! -x "${MRX_ENV_PATH}/bin/python" ]]; then
  echo "ERROR: mrx conda env not found."
  conda env list
  exit 1
fi
PYTHON_CMD="${MRX_ENV_PATH}/bin/python"

cd "${MRX_ROOT}"
export PYTHONPATH="${MRX_ROOT}:${PYTHONPATH:-}"

echo "Running inline harmonic nullspace solve via ${PYTHON_CMD}"
exec "${PYTHON_CMD}" <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

repo = Path(os.environ["MRX_ROOT"]).resolve()
sys.path.insert(0, str(repo))

from mrx.nullspace import _initial_guesses, _n_vectors, find_nullspace_vectors
from mrx.operators import (
    assemble_derivative_operators,
    assemble_hodge_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_schur_jacobi_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from scripts.plotting.harmonic_nullspace_geometry import (
    build_sequence_analytic,
    build_sequence_h5,
    load_map_geometry_h5,
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _harmonic_diagnostics(seq, v, *, k: int, dirichlet: bool) -> dict[str, float]:
    lv = seq.apply_hodge_laplacian(v, k, dirichlet=dirichlet)
    mv = seq.apply_mass_matrix(v, k, dirichlet=dirichlet)
    m_inner = float(v @ mv)
    rayleigh = float((v @ lv) / m_inner) if m_inner > 0.0 else float("nan")
    residual = float(seq.l2_norm(lv, k, dirichlet=dirichlet))
    return {
        "rayleigh": rayleigh,
        "rayleigh_abs": abs(rayleigh) if np.isfinite(rayleigh) else float("nan"),
        "hodge_residual_l2": residual,
        "m_norm": float(seq.l2_norm(v, k, dirichlet=dirichlet)),
        "m_inner": m_inner,
    }


k = int(os.environ["HARMONIC_K"])
map_mode = os.environ["MAP_MODE"]
map_kind = os.environ.get("MAP_KIND", "cerfon")
ns = (int(os.environ["NS_R"]), int(os.environ["NS_THETA"]), int(os.environ["NS_ZETA"]))
p = int(os.environ.get("FEM_P", "3"))
betti = tuple(int(x) for x in os.environ.get("BETTI", "1 1 0 0").split())
flip_zeta = _env_bool("FLIP_ZETA")
sparse = _env_bool("SPARSE", default=True) and k == 2
plots = _env_bool("PLOTS")
out_dir = Path(os.environ["OUT_DIR"]).expanduser().resolve()
tol = float(os.environ.get("TOL", "1e-9"))
maxiter = int(os.environ.get("NULLSPACE_MAXITER", "100"))
eps = float(os.environ.get("NULLSPACE_EPS", "1e-4"))
inner_tol = float(os.environ.get("INNER_TOL", "1e-6"))
stall_ratio = float(os.environ.get("STALL_RATIO", "0.9"))
nfp_raw = os.environ.get("NFP", "").strip()
nfp = int(nfp_raw) if nfp_raw else None
map_geometry_h5_raw = os.environ.get("MAP_GEOMETRY_H5", "").strip()
map_geometry_h5 = Path(map_geometry_h5_raw) if map_geometry_h5_raw else None

if map_mode == "h5":
    if map_geometry_h5 is None:
        raise SystemExit("MAP_GEOMETRY_H5 required for map_mode=h5")
    if nfp is None:
        nfp_h5, _, _, _ = load_map_geometry_h5(map_geometry_h5.resolve())
        nfp = int(nfp_h5)
    seq, _, _, map_raw = build_sequence_h5(
        ns=ns,
        p=p,
        betti=betti,
        tol=tol,
        maxiter=maxiter,
        map_geometry_h5=map_geometry_h5,
        nfp=nfp,
        flip_zeta=flip_zeta,
        strict_jacobian=True,
    )
elif map_mode == "analytic":
    if nfp is None:
        raise SystemExit("NFP required for map_mode=analytic")
    seq, _, _, map_raw = build_sequence_analytic(
        ns=ns,
        p=p,
        betti=betti,
        tol=tol,
        maxiter=maxiter,
        map_kind=map_kind,
        nfp=nfp,
    )
else:
    raise SystemExit(f"unsupported MAP_MODE={map_mode!r}")

if k == 1:
    ops = assemble_mass_operators(seq, seq.geometry, ks=(0, 1, 2))
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1))
    ops = assemble_incidence_operators(seq, ops, ks=(0, 1))
    ops = assemble_derivative_operators(seq, seq.geometry, ops, ks=(0, 1))
    ops = assemble_hodge_operators(seq, seq.geometry, ops, ks=(0, 1))
    seq.set_operators(ops)
elif sparse:
    seq.assemble_all_sparse()
else:
    ops = assemble_mass_operators(seq, seq.geometry, ks=(0, 1, 2, 3))
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3))
    ops = assemble_incidence_operators(seq, ops, ks=(0, 1, 2))
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(2,), dirichlet_variants=(True,))
    ops = assemble_derivative_operators(seq, seq.geometry, ops, ks=(0, 1, 2))
    ops = assemble_hodge_operators(seq, seq.geometry, ops, ks=(0, 1, 2))
    seq.set_operators(ops)

dirichlet = k == 2
n_harm = _n_vectors(betti, k, dirichlet)
if n_harm <= 0:
    raise RuntimeError(f"betti={betti} gives zero harmonic count for k={k}")

ops = seq._require_operators()
guesses = _initial_guesses(seq, ops, k, dirichlet, n_harm)
x0s = [guesses[0]] if guesses and guesses[0] is not None else None

t0 = time.perf_counter()
vecs, info = find_nullspace_vectors(
    seq,
    ops,
    k,
    n_harm,
    eps,
    dirichlet=dirichlet,
    x0s=x0s,
    abs_tol=tol,
    inner_tol=inner_tol,
    maxiter=maxiter,
    stall_ratio=stall_ratio,
)
elapsed = time.perf_counter() - t0
if int(vecs.shape[0]) == 0:
    raise RuntimeError(f"no k={k} harmonic vector returned")
v = vecs[0]
if not np.isfinite(np.asarray(v)).all():
    raise RuntimeError("non-finite harmonic DOFs; check detJ / FLIP_ZETA")

diag = _harmonic_diagnostics(seq, v, k=k, dirichlet=dirichlet)
diag["wall_seconds"] = float(elapsed)
n_iters, res, _rq = info[0]
print(
    f"k={k} harmonic: iters={n_iters}  ||L v||={res:.3e}  "
    f"rayleigh={diag['rayleigh']:.3e}  M{k}_norm={diag['m_norm']:.6e}  "
    f"wall={elapsed:.1f}s",
    flush=True,
)

meta: dict[str, Any] = {
    "operator": f"hodge_laplacian_k{k}_{'dirichlet' if k == 2 else 'natural'}",
    "form_degree": k,
    "map_mode": map_mode,
    "map_kind": map_kind,
    "map_geometry_h5": str(map_geometry_h5.resolve()) if map_geometry_h5 else None,
    "nfp": int(nfp),
    "ns": list(ns),
    "p": int(p),
    "betti": list(betti),
    "flip_zeta": bool(flip_zeta),
    "nullspace_eps": float(eps),
    "nullspace_maxiter": int(maxiter),
    "inner_tol": float(inner_tol),
    "stall_ratio": float(stall_ratio),
    "sparse_harmonic": bool(sparse),
    "harmonic_diagnostics": diag,
    "inverse_iteration": [
        {"n_iters": int(n_i), "residual": float(r_i)} for n_i, r_i in info
    ],
}

out_dir.mkdir(parents=True, exist_ok=True)
if k == 2:
    dof_path = out_dir / "hodge_k2_dbc_nullspace_dof.npy"
    meta_path = out_dir / "hodge_k2_nullspace_meta.json"
else:
    dof_path = out_dir / "hcurl_k1_nullspace_dof.npy"
    meta_path = out_dir / "hcurl_nullspace_meta.json"
np.save(dof_path, np.asarray(v))
meta_path.write_text(json.dumps(meta, indent=2))
print(f"Wrote {dof_path}", flush=True)
print(f"Wrote {meta_path}", flush=True)

if plots:
    from scripts.plotting.harmonic_nullspace_plots import (
        run_k1_nullspace_plots,
        run_k2_nullspace_plots,
    )

    plot_kwargs = dict(
        seq=seq,
        v=v,
        map_raw=map_raw,
        nfp=int(nfp),
        out_dir=out_dir,
        zetas=(0.0, 0.25, 0.5),
    )
    if k == 2:
        run_k2_nullspace_plots(**plot_kwargs)
    else:
        run_k1_nullspace_plots(**plot_kwargs)

print(out_dir, flush=True)
PY
