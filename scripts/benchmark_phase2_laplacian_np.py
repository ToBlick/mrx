"""Phase 2 compatibility wrapper for the unified preconditioner benchmark driver.

Use scripts/benchmark_preconditioners.py directly for both single-cell
and sweep modes. This wrapper is kept to preserve existing commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_preconditioners import run_phase2_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=str, default="8,12,16,20")
    parser.add_argument("--ps", type=str, default="1,2,3")
    parser.add_argument("--kappas", type=str, default="1.25")
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--dirichlet", choices=("dbc", "nbc"), default="dbc")
    parser.add_argument("--n-rhs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--besov-s", type=float, default=1.0)
    parser.add_argument("--load-frame", choices=("ref", "phys"), default="ref")
    parser.add_argument("--cg-tol", type=float, default=1e-12)
    parser.add_argument("--cg-maxiter", type=int, default=1000)
    parser.add_argument("--laplace-ks", type=str, default="0,1,2,3")
    parser.add_argument("--tensor-ranks", type=str, default="1,2,3")
    parser.add_argument("--cheb-steps", type=str, default="2,3,4")
    parser.add_argument("--mass-ks", type=str, default="0,1,2,3")
    parser.add_argument("--out", type=str, default="outputs/preconditioners/phase2_preconditioners_np.csv")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    # Re-map legacy phase2 flags to unified sweep API.
    unified_args = argparse.Namespace(
        n_list=tuple(int(s.strip()) for s in args.ns.split(",") if s.strip()),
        p_list=tuple(int(s.strip()) for s in args.ps.split(",") if s.strip()),
        kappas=tuple(float(s.strip()) for s in args.kappas.split(",") if s.strip()),
        epsilon=float(args.epsilon),
        r0=float(args.r0),
        nfp=int(args.nfp),
        dirichlet=args.dirichlet,
        n_rhs=int(args.n_rhs),
        seed=int(args.seed),
        besov_s=float(args.besov_s),
        load_frame=args.load_frame,
        cg_tol=float(args.cg_tol),
        cg_maxiter=int(args.cg_maxiter),
        laplace_ks=tuple(int(s.strip()) for s in args.laplace_ks.split(",") if s.strip()),
        tensor_ranks=tuple(int(s.strip()) for s in args.tensor_ranks.split(",") if s.strip()),
        cheb_steps=tuple(int(s.strip()) for s in args.cheb_steps.split(",") if s.strip()),
        mass_ks=tuple(int(s.strip()) for s in args.mass_ks.split(",") if s.strip()),
    )

    summary, rows, cell_summaries = run_phase2_sweep(unified_args)

    if not rows:
        raise SystemExit("No rows produced.")

    import csv
    import json
    from dataclasses import asdict

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"Wrote {out_path}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as fh:
            json.dump({"summary": summary, "cell_summaries": cell_summaries}, fh, indent=2)
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
