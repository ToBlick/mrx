"""Submit one mass-preconditioner benchmark case through Hydra/Submitit.

Usage from the repo root:

    # Submit a single baseline job through SLURM
    python scripts/config_scripts/mass_preconditioner_submit.py -m sweep_axis=baseline

    # Submit one parallel 1-D kappa sweep
    python scripts/config_scripts/mass_preconditioner_submit.py -m sweep_axis=kappa kappa=1.0,1.2,1.4,1.6
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _serialize_number_list(values) -> str:
    return ",".join(str(value) for value in values)


def _artifact_dir_from_stdout(stdout: str) -> str | None:
    matches = re.findall(r"saved benchmark artifacts to:\s*(.+)", stdout)
    if not matches:
        return None
    return matches[-1].strip()


@hydra.main(config_path="../../conf", config_name="config_mass_preconditioner", version_base=None)
def main(cfg: DictConfig):
    repo_root = _repo_root()
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    env = os.environ.copy()
    case_config = {
        "ns": [int(cfg.n), int(cfg.n), int(cfg.n)],
        "p": int(cfg.p),
        "maxiter": int(cfg.maxiter),
        "map_kind": "rotating_ellipse",
        "rotating_kappa": float(cfg.kappa),
        "rotating_eps": float(cfg.eps),
    }
    env["MRX_MASS_CONFIG_JSON"] = json.dumps(case_config)
    env["MRX_MASS_POLY_STEP_OPTIONS"] = _serialize_number_list(cfg.poly_step_options)
    env["MRX_MASS_TENSOR_RANKS"] = _serialize_number_list(cfg.tensor_ranks)
    env["MRX_MASS_RHS_KIND"] = str(cfg.rhs_kind)
    env["MRX_MASS_RHS_S"] = str(cfg.rhs_s)
    env["MRX_MASS_RHS_UPPER_LIMIT"] = str(cfg.rhs_upper_limit)
    env["MRX_MASS_RHS_NUM_MODES"] = str(cfg.rhs_num_modes)
    env["MRX_MASS_RHS_SCALE"] = str(cfg.rhs_scale)
    env["MRX_MASS_RHS_SMOOTHNESS_MARGIN"] = str(cfg.rhs_smoothness_margin)
    env["MRX_MASS_RHS_NORMALIZATION_SAMPLES"] = str(cfg.rhs_normalization_samples)

    command = [sys.executable, "scripts/interactive/mass_preconditioner_demo.py"]
    print("Submitting mass benchmark case:")
    print(f"  sweep_axis={cfg.sweep_axis}")
    print(f"  n={cfg.n}")
    print(f"  p={cfg.p}")
    print(f"  kappa={cfg.kappa}")
    print(f"  eps={cfg.eps}")
    print(f"  maxiter={cfg.maxiter}")
    print(f"  poly_step_options={list(cfg.poly_step_options)}")
    print(f"  tensor_ranks={list(cfg.tensor_ranks)}")
    print(f"  rhs_kind={cfg.rhs_kind}")
    print(f"  rhs_s={cfg.rhs_s}")
    print(f"  rhs_upper_limit={cfg.rhs_upper_limit}")
    print(f"  rhs_num_modes={cfg.rhs_num_modes}")
    print(f"  rhs_scale={cfg.rhs_scale}")
    print(f"  rhs_smoothness_margin={cfg.rhs_smoothness_margin}")
    print(f"  rhs_normalization_samples={cfg.rhs_normalization_samples}")
    print(f"  hydra_output_dir={output_dir}")

    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()

    result = {
        "sweep_axis": str(cfg.sweep_axis),
        "n": int(cfg.n),
        "p": int(cfg.p),
        "kappa": float(cfg.kappa),
        "eps": float(cfg.eps),
        "maxiter": int(cfg.maxiter),
        "poly_step_options": [int(value) for value in cfg.poly_step_options],
        "tensor_ranks": [int(value) for value in cfg.tensor_ranks],
        "rhs_kind": str(cfg.rhs_kind),
        "rhs_s": float(cfg.rhs_s),
        "rhs_upper_limit": int(cfg.rhs_upper_limit),
        "rhs_num_modes": int(cfg.rhs_num_modes),
        "rhs_scale": float(cfg.rhs_scale),
        "rhs_smoothness_margin": float(cfg.rhs_smoothness_margin),
        "rhs_normalization_samples": int(cfg.rhs_normalization_samples),
        "artifact_dir": _artifact_dir_from_stdout(completed.stdout),
    }
    (output_dir / "launcher_result.json").write_text(json.dumps(result, indent=2))
    print(f"Launcher metadata saved to {output_dir / 'launcher_result.json'}")


if __name__ == "__main__":
    main()