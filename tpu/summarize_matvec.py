"""Join the per-backend matvec JSON files into one table and compose a step.

The matvec benchmark writes one JSON per backend and precision setting. This
reads several of them, prints the scan-form cost of every operator side by
side, and composes the velocity smoothing solve from those costs so the step
time can be checked against a sum of measured parts rather than back-derived
by dividing a step time by an apply count.

Usage:
    python tpu/summarize_matvec.py LABEL=PATH [LABEL=PATH ...] [--iters N]
"""

from __future__ import annotations

import argparse
import json
import sys

# One MINRES iteration of the velocity smoothing solve, measured by counting
# calls through mrx.operators: the count is backend-independent.
SMOOTHING_ITERATION: tuple[tuple[str, int], ...] = (
    ("apply_stiffness k=2", 1),
    ("apply_mass_matrix k=2", 1),
    ("apply_mass_matrix k=1", 1),
    ("apply_derivative D^T D k=1", 1),
    ("mass atom k=2", 1),
    ("mass atom k=1", 1),
)


def load(path: str) -> dict:
    """Read one benchmark JSON.

    Args:
        path: file written by ``tpu/matvec_bench.py``.

    Returns:
        the parsed object, with ``rows`` mapping an operator name to its
        ``eager``/``jit``/``scan`` seconds.
    """
    with open(path) as handle:
        return json.load(handle)


def best(row: dict) -> float | None:
    """The honest per-apply cost of one operator, in milliseconds.

    Prefers the scan form, which is what the relaxation actually pays: it is
    the only form measured inside a fused XLA graph. Falls back to the jitted
    form for the non-square operators, which cannot be scanned in place.

    Args:
        row: one entry of the ``rows`` mapping.

    Returns:
        milliseconds, or ``None`` if the operator was not timed.
    """
    for key in ("scan", "jit", "eager"):
        if row.get(key) is not None:
            return row[key] * 1e3
    return None


def main() -> None:
    """Print the joined matvec table and the composed smoothing solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="+", help="LABEL=PATH")
    parser.add_argument("--iters", type=int, default=3497,
                        help="MINRES iterations of the smoothing solve")
    cli = parser.parse_args()

    runs = {}
    for pair in cli.pairs:
        if "=" not in pair:
            sys.exit(f"expected LABEL=PATH, got {pair!r}")
        label, path = pair.split("=", 1)
        try:
            runs[label] = load(path)
        except OSError as exc:
            print(f"[skip] {label}: {exc}", file=sys.stderr)

    if not runs:
        sys.exit("no readable inputs")

    labels = list(runs)
    for label in labels:
        backend = runs[label]["backend"]
        print(f"[{label}] {backend['device_kind']} x{backend['device_count']}  "
              f"{runs[label]['mrx_dtype']}  matmul {backend['matmul_precision']}  "
              f"jax {backend['jax_version']}")

    names: list[str] = []
    for run in runs.values():
        for name in run["rows"]:
            if name not in names:
                names.append(name)

    width = max(len(n) for n in names) + 2
    print("\nper-apply cost in ms (scan form where it exists, else jit)\n")
    print("  " + "operator".ljust(width) + "".join(f"{lab:>14}" for lab in labels))
    for name in names:
        cells = []
        for label in labels:
            row = runs[label]["rows"].get(name)
            value = best(row) if row else None
            cells.append(f"{value:14.4f}" if value is not None else f"{'-':>14}")
        print("  " + name.ljust(width) + "".join(cells))

    print(f"\nvelocity smoothing solve composed from those costs, "
          f"{cli.iters} MINRES iterations\n")
    print("  " + "term".ljust(width) + "x" + "".join(f"{lab:>14}" for lab in labels))
    totals = dict.fromkeys(labels, 0.0)
    complete = dict.fromkeys(labels, True)
    for name, multiplicity in SMOOTHING_ITERATION:
        cells = []
        for label in labels:
            row = runs[label]["rows"].get(name)
            value = best(row) if row else None
            if value is None:
                complete[label] = False
                cells.append(f"{'-':>14}")
            else:
                totals[label] += multiplicity * value
                cells.append(f"{multiplicity * value:14.4f}")
        print("  " + name.ljust(width) + str(multiplicity) + "".join(cells))

    print("  " + "per iteration (ms)".ljust(width) + " "
          + "".join(f"{totals[lab]:14.4f}" if complete[lab] else f"{'partial':>14}"
                    for lab in labels))
    print("  " + f"x {cli.iters} iterations (s)".ljust(width) + " "
          + "".join(f"{totals[lab] * cli.iters / 1e3:14.2f}" if complete[lab]
                    else f"{'partial':>14}" for lab in labels))


if __name__ == "__main__":
    main()
