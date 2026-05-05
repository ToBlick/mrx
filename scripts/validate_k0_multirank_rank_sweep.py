from __future__ import annotations

import argparse

from debug_tensor_forward_models import evaluate_forward_model


def _parse_ns(text: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in text.split(","))
    if len(parts) != 3:
        raise ValueError(f"Expected ns as 'nr,nt,nz', got {text!r}")
    return parts


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of integers")
    return values


def _print_table(title: str, rows: list[dict[str, float]]) -> None:
    print(title)
    headers = ["rank", "cp_err", "fwd_mean", "fwd_max", "fro_err"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in rows:
        fro_err = row.get("fro_relative_error")
        fro_text = f"{fro_err:12.5e}" if fro_err is not None else f"{'-':>12}"
        print(
            f"{row['rank']:12d}"
            f"{row['cp_relative_error_max']:12.5e}"
            f"{row['forward_relative_error_mean']:12.5e}"
            f"{row['forward_relative_error_max']:12.5e}"
            f"{fro_text}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep tensor rank for a tensor forward-model validation problem."
    )
    parser.add_argument(
        "--problem",
        choices=("k0-stiffness", "mass-k0", "mass-k1", "mass-k2", "mass-k3"),
        default="k0-stiffness",
        help="Tensor forward-model problem to validate",
    )
    parser.add_argument("--ns", type=_parse_ns, default=(4, 8, 4), help="Grid sizes as nr,nt,nz")
    parser.add_argument("--p", type=int, default=3, help="Spline degree in each direction")
    parser.add_argument("--ranks", type=_parse_int_list, default=(1, 2, 3, 4), help="Comma-separated tensor ranks")
    parser.add_argument("--n-vectors", type=int, default=8, help="Number of random test vectors")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--cp-maxiter", type=int, default=100, help="Maximum CP ALS iterations")
    parser.add_argument("--cp-tol", type=float, default=1e-9, help="CP ALS tolerance")
    parser.add_argument("--cp-ridge", type=float, default=1e-12, help="CP ALS ridge regularization")
    parser.add_argument("--map-kind", choices=("rotating_ellipse", "identity"), default="rotating_ellipse")
    parser.add_argument("--rotating-eps", type=float, default=0.33)
    parser.add_argument("--rotating-kappa", type=float, default=1.4)
    parser.add_argument("--rotating-r0", type=float, default=1.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--free", action="store_true", help="Use free extraction space")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense Frobenius checks")
    parser.add_argument(
        "--modes",
        choices=("full", "bulk", "both"),
        default="both",
        help="Which extracted-space slices to validate",
    )
    args = parser.parse_args()

    common_kwargs = {
        "problem": args.problem,
        "ns": args.ns,
        "p": args.p,
        "cp_maxiter": args.cp_maxiter,
        "cp_tol": args.cp_tol,
        "cp_ridge": args.cp_ridge,
        "n_vectors": args.n_vectors,
        "seed": args.seed,
        "dense": not args.skip_dense,
        "free": args.free,
        "map_kind": args.map_kind,
        "rotating_eps": args.rotating_eps,
        "rotating_kappa": args.rotating_kappa,
        "rotating_r0": args.rotating_r0,
        "rotating_nfp": args.rotating_nfp,
        "tol": args.tol,
        "maxiter": args.maxiter,
    }

    if args.modes in ("full", "both"):
        full_rows = [
            evaluate_forward_model(rank=rank, bulk_only=False, **common_kwargs)
            for rank in args.ranks
        ]
        _print_table(f"{args.problem} extracted-space forward validation", full_rows)

    if args.modes in ("bulk", "both"):
        bulk_rows = [
            evaluate_forward_model(rank=rank, bulk_only=True, **common_kwargs)
            for rank in args.ranks
        ]
        _print_table(f"{args.problem} bulk-only forward validation", bulk_rows)


if __name__ == "__main__":
    main()