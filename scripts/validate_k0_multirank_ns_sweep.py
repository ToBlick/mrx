from __future__ import annotations

import argparse

from debug_tensor_forward_models import evaluate_forward_model


def _parse_ns_list(text: str) -> tuple[tuple[int, int, int], ...]:
    values = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = tuple(int(part.strip()) for part in chunk.split(","))
        if len(parts) != 3:
            raise ValueError(f"Expected each ns entry as 'nr,nt,nz', got {chunk!r}")
        values.append(parts)
    if not values:
        raise ValueError("Expected at least one ns triple")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run non-dense tensor forward validation across several resolutions."
    )
    parser.add_argument(
        "--problem",
        choices=("k0-stiffness", "mass-k0", "mass-k1", "mass-k2", "mass-k3"),
        default="k0-stiffness",
        help="Tensor forward-model problem to validate",
    )
    parser.add_argument(
        "--ns-list",
        type=_parse_ns_list,
        default=((4, 8, 4), (6, 12, 6), (8, 16, 8)),
        help="Semicolon-separated list like '4,8,4;6,12,6;8,16,8'",
    )
    parser.add_argument("--p", type=int, default=3, help="Spline degree in each direction")
    parser.add_argument("--rank", type=int, default=2, help="Tensor rank to validate")
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
    parser.add_argument("--bulk-only", action="store_true", help="Restrict the check to the tensor bulk")
    args = parser.parse_args()

    print("          ns       cp_err     fwd_mean      fwd_std      fwd_max")
    for ns in args.ns_list:
        result = evaluate_forward_model(
            problem=args.problem,
            ns=ns,
            p=args.p,
            rank=args.rank,
            cp_maxiter=args.cp_maxiter,
            cp_tol=args.cp_tol,
            cp_ridge=args.cp_ridge,
            n_vectors=args.n_vectors,
            seed=args.seed,
            dense=False,
            free=args.free,
            bulk_only=args.bulk_only,
            map_kind=args.map_kind,
            rotating_eps=args.rotating_eps,
            rotating_kappa=args.rotating_kappa,
            rotating_r0=args.rotating_r0,
            rotating_nfp=args.rotating_nfp,
            tol=args.tol,
            maxiter=args.maxiter,
        )
        ns_label = f"{ns[0]},{ns[1]},{ns[2]}"
        print(
            f"{ns_label:>12}"
            f"{result['cp_relative_error_max']:12.5e}"
            f"{result['forward_relative_error_mean']:12.5e}"
            f"{result['forward_relative_error_std']:12.5e}"
            f"{result['forward_relative_error_max']:12.5e}"
        )


if __name__ == "__main__":
    main()