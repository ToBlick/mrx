"""Phase 2 sweep: six independent axes anchored at the Phase 1 baseline.

Holds R0=1 fixed and runs the Phase 1 baseline-vs-tensor protocol along
each axis below, with all *other* parameters held at their baseline value.
The central baseline cell is run exactly once; cells where the axis value
equals its baseline are deduplicated.

Axes:

* ``--kappas``       — geometry aspect ratio
* ``--epses``        — minor radius
* ``--ps``           — polynomial degree (uniform p in 3D)
* ``--ns-list``      — resolution triples (semicolon-separated)
* ``--besov-svals``  — Besov RHS smoothness

Output is one CSV with the Phase 1 schema (already includes ``kappa``,
``eps``, ``p``, ``n_r/n_t/n_z``, ``nfp``, ``besov_s`` columns) so every
cell lives in one tidy file.

The matching SLURM launcher ``slurm/job_phase2_geometry.sh`` performs the
same sweep in parallel.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from dataclasses import asdict
from pathlib import Path

import benchmark_phase1_mass_baseline_vs_tensor as phase1


def _parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(s.strip()) for s in text.split(",") if s.strip())


def _parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(s.strip()) for s in text.split(",") if s.strip())


def _parse_ns_list(text: str) -> tuple[tuple[int, int, int], ...]:
    """Semicolon-separated list of triples, e.g. '8,16,8;16,32,16'."""
    triples: list[tuple[int, int, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        triples.append(phase1._parse_ns(chunk))
    return tuple(triples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Axis values.
    parser.add_argument("--kappas", type=_parse_float_list,
                        default=(1.0, 1.25, 1.5, 1.75))
    parser.add_argument("--epses", type=_parse_float_list,
                        default=(0.1, 0.2, 0.33, 0.5))
    parser.add_argument("--ps", type=_parse_int_list, default=(1, 2, 3, 4))
    parser.add_argument("--ns-list", type=_parse_ns_list,
                        default=((8, 16, 8), (16, 32, 16),
                                 (32, 64, 32), (64, 128, 64)),
                        help="Semicolon-separated 'nr,nt,nz' triples.")
    parser.add_argument("--besov-svals", type=_parse_float_list,
                        default=(0.0, 1.0, 2.0, 3.0))
    # Anchors.
    parser.add_argument("--kappa-baseline", type=float,
                        default=phase1.DEFAULT_KAPPA)
    parser.add_argument("--eps-baseline", type=float,
                        default=phase1.DEFAULT_EPS)
    parser.add_argument("--p-baseline", type=int, default=phase1.DEFAULT_P)
    parser.add_argument("--ns-baseline", type=phase1._parse_ns,
                        default=phase1.DEFAULT_NS)
    parser.add_argument("--nfp-baseline", type=int, default=phase1.DEFAULT_NFP)
    parser.add_argument("--besov-s-baseline", type=float,
                        default=phase1.DEFAULT_BESOV_S)
    # Static phase-1 controls.
    parser.add_argument("--r0", type=float, default=phase1.DEFAULT_R0)
    parser.add_argument("--ks", type=phase1._parse_int_list, default=(0, 1, 2, 3))
    parser.add_argument("--n-rhs", type=int, default=phase1.DEFAULT_N_RHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=phase1.DEFAULT_TOL)
    parser.add_argument("--maxiter", type=int, default=phase1.DEFAULT_MAXITER)
    parser.add_argument("--ranks", type=phase1._parse_int_list, default=(3,))
    parser.add_argument("--bulk-cheb", type=phase1._parse_int_list, default=(0,))
    parser.add_argument("--cheb-baseline", type=phase1._parse_int_list, default=(3,))
    parser.add_argument("--no-jacobi", action="store_true")
    parser.add_argument("--no-chebyshev", action="store_true")
    parser.add_argument("--no-tensor", action="store_true")
    parser.add_argument("--include-k0-stiffness-dbc", action="store_true",
                        help="Append the k=0 Dirichlet stiffness block to each Phase 2 cell.")
    parser.add_argument("--only-k0-stiffness-dbc", action="store_true",
                        help="Run each Phase 2 cell only for the k=0 Dirichlet stiffness block.")
    parser.add_argument(
        "--inner-schur", action="store_true",
        help="Benchmark only the tensor variant with inner RT/zeta Schur on for k=1,2.",
    )
    parser.set_defaults(inner_schur=False)
    parser.add_argument("--out", type=str,
                        default="outputs/phase2_sweep.csv",
                        help="CSV output path.")
    args = parser.parse_args()

    base = {
        "kappa": float(args.kappa_baseline),
        "eps": float(args.eps_baseline),
        "p": int(args.p_baseline),
        "ns": tuple(args.ns_baseline),
        "nfp": int(args.nfp_baseline),
        "besov_s": float(args.besov_s_baseline),
    }

    axes: list[tuple[str, list]] = [
        ("kappa", [float(v) for v in args.kappas]),
        ("eps", [float(v) for v in args.epses]),
        ("p", [int(v) for v in args.ps]),
        ("ns", [tuple(v) for v in args.ns_list]),
        ("besov_s", [float(v) for v in args.besov_svals]),
    ]

    cells: list[tuple[dict, str]] = []
    seen: set[tuple] = set()

    def _add_cell(overrides: dict, axis_label: str) -> None:
        params = {**base, **overrides}
        key = tuple(params[k] for k in ("kappa", "eps", "p", "ns", "nfp", "besov_s"))
        if key in seen:
            return
        seen.add(key)
        cells.append((params, axis_label))

    # Always run the central baseline cell first.
    _add_cell({}, "baseline")
    for axis, values in axes:
        for v in values:
            _add_cell({axis: v}, axis)

    all_rows = []
    for params, axis_label in cells:
        print(f"\n=== axis={axis_label}  {params} ===", flush=True)
        sub_args = copy.copy(args)
        sub_args.kappa = params["kappa"]
        sub_args.eps = params["eps"]
        sub_args.p = params["p"]
        sub_args.ns = params["ns"]
        sub_args.nfp = params["nfp"]
        sub_args.besov_s = params["besov_s"]
        # Drop phase-2-only fields before handing to phase1.run_benchmark.
        for fld in (
            "kappas", "epses", "ps", "ns_list", "besov_svals",
            "kappa_baseline", "eps_baseline", "p_baseline", "ns_baseline",
            "nfp_baseline", "besov_s_baseline", "out",
        ):
            if hasattr(sub_args, fld):
                delattr(sub_args, fld)
        t0 = time.perf_counter()
        rows = phase1.run_benchmark(sub_args)
        dt = time.perf_counter() - t0
        print(f"  {len(rows)} rows in {dt:.1f}s", flush=True)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows produced; check sweep arguments.")

    def _key(row):
        order = {
            "jacobi": 0,
            "chebyshev": 1,
            "tensor": 2,
            "tensor_inner_schur": 3,
        }.get(row.strategy, 99)
        return (
            row.kappa, row.eps, row.p, (row.n_r, row.n_t, row.n_z),
            row.nfp, row.besov_s, row.case, row.k, order,
            row.cheb_steps, row.rank, row.bulk_cheb_steps, row.inner_schur,
        )
    all_rows.sort(key=_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
