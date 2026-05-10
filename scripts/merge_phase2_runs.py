"""Merge Phase 2 run directories or aggregated CSVs into one normalized CSV.

This handles the schema split between older mass-only Phase 2 outputs and
newer outputs that include explicit ``case`` and ``inner_schur`` columns.
Missing fields are filled with sensible defaults so the merged CSV can be
plotted directly with ``plot_phase2_geometry.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PREFERRED_FIELDS = [
    "k",
    "strategy",
    "method",
    "avg_iters",
    "max_iters",
    "case",
    "min_iters",
    "std_iters",
    "avg_solve_ms",
    "min_solve_ms",
    "max_solve_ms",
    "std_solve_ms",
    "setup_ms",
    "final_residual",
    "rank",
    "cheb_steps",
    "bulk_cheb_steps",
    "inner_schur",
    "n_rhs",
    "seed",
    "n_r",
    "n_t",
    "n_z",
    "p",
    "eps",
    "kappa",
    "nfp",
    "besov_s",
    "dirichlet",
    "hyperparams",
]


def iter_csv_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("cell_*.csv"))
        if not files:
            raise SystemExit(f"No cell_*.csv files found under {path}")
        return files
    if path.is_file():
        return [path]
    raise SystemExit(f"Input not found: {path}")


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    if not normalized.get("case"):
        k = normalized.get("k", "")
        normalized["case"] = f"M{k}" if k not in ("", None) else "unknown"
    if not normalized.get("inner_schur"):
        normalized["inner_schur"] = "False"
    return normalized


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for csv_path in iter_csv_files(path):
            with csv_path.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append(_normalize_row(row))
    return rows


def field_order(rows: list[dict[str, str]]) -> list[str]:
    seen = {field for row in rows for field in row.keys()}
    ordered = [field for field in PREFERRED_FIELDS if field in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def sort_key(row: dict[str, str]):
    strategy_order = {
        "jacobi": 0,
        "chebyshev": 1,
        "tensor": 2,
        "tensor_inner_schur": 3,
    }

    def as_int(key: str, default: int = -1) -> int:
        try:
            return int(float(row.get(key, default)))
        except (TypeError, ValueError):
            return default

    def as_float(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    return (
        as_float("kappa"),
        as_float("eps"),
        as_int("p"),
        (as_int("n_r"), as_int("n_t"), as_int("n_z")),
        as_int("nfp"),
        as_float("besov_s"),
        row.get("case", ""),
        as_int("k"),
        strategy_order.get(row.get("strategy", ""), 99),
        as_int("cheb_steps"),
        as_int("rank"),
        as_int("bulk_cheb_steps"),
        row.get("inner_schur", "False"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Phase 2 run directories with cell_*.csv files or aggregated CSVs.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Merged CSV output path.")
    args = parser.parse_args()

    rows = load_rows([path.resolve() for path in args.inputs])
    if not rows:
        raise SystemExit("No rows found in inputs.")
    rows.sort(key=sort_key)

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = field_order(rows)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()