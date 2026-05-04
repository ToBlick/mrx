from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.utils import build_random_besov_rhs_batch

jax.config.update("jax_enable_x64", True)


def _build_sequence(n: int, p: int) -> DeRhamSequence:
    seq = DeRhamSequence(
        (n, n, n),
        (p, p, p),
        2 * p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-9,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map())
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Besov RHS projection for k=0..3 on a tiny sequence.")
    parser.add_argument("--n", type=int, default=4, help="isotropic spline resolution")
    parser.add_argument("--p", type=int, default=2, help="spline degree")
    parser.add_argument("--n-rhs", type=int, default=2, help="number of RHS samples per form degree")
    parser.add_argument("--dirichlet", action="store_true", help="use Dirichlet extraction instead of the free case")
    parser.add_argument("--s", type=float, default=1.0, help="Besov smoothness exponent")
    parser.add_argument("--upper-limit", type=int, default=8, help="maximum Fourier index per coordinate direction")
    parser.add_argument("--num-modes", type=int, default=8, help="number of sparse modes per RHS")
    parser.add_argument("--scale", type=float, default=1.0, help="target RMS scale before projection")
    parser.add_argument("--smoothness-margin", type=float, default=0.0, help="extra coefficient decay beyond the baseline H^s exponent")
    parser.add_argument("--normalization-samples", type=int, default=32, help="number of random points used for RMS normalization")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    args = parser.parse_args()

    seq = _build_sequence(args.n, args.p)

    print("built sequence:")
    print(f"  ns={(args.n, args.n, args.n)}")
    print(f"  p={args.p}")
    print(f"  dirichlet={args.dirichlet}")

    for k in range(4):
        print(f"\nGenerating Besov RHS batch for k={k}...")
        rhs_batch = build_random_besov_rhs_batch(
            seq,
            k,
            dirichlet=args.dirichlet,
            n_rhs=args.n_rhs,
            seed=args.seed + 1000 * k,
            s=args.s,
            upper_limit=args.upper_limit,
            num_modes=args.num_modes,
            scale=args.scale,
            smoothness_margin=args.smoothness_margin,
            normalization_samples=args.normalization_samples,
        )
        norms = jnp.linalg.norm(rhs_batch, axis=1)
        print(f"k={k}: shape={tuple(rhs_batch.shape)} norms={[float(value) for value in norms]}")

    print("Besov RHS smoke test completed.")


if __name__ == "__main__":
    main()