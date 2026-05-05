from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    _apply_k2_divdiv_extracted_tensor_model,
    _apply_k2_divdiv_regular_forward,
    _apply_k2_divdiv_regular_tensor_model,
    _assemble_dense_from_apply,
    _assemble_k2_divdiv_regular_tensor_dense_matrix,
    _assemble_k2_divdiv_regular_tensor_model,
    apply_stiffness,
    assemble_incidence_operators,
    assemble_mass_operators,
)
from mrx.preconditioners import _tensor_block_indices_k2


jax.config.update("jax_enable_x64", True)


def _identity_map(x):
    return x


def _parse_ns(text: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in text.split(","))
    if len(parts) != 3:
        raise ValueError(f"Expected ns as 'nr,nt,nz', got {text!r}")
    return parts


def _build_map(args):
    if args.map_kind == "identity":
        return _identity_map
    if args.map_kind == "rotating_ellipse":
        return rotating_ellipse_map(
            eps=args.rotating_eps,
            kappa=args.rotating_kappa,
            R0=args.rotating_r0,
            nfp=args.rotating_nfp,
        )
    raise ValueError(f"Unknown map kind {args.map_kind!r}")


def _build_sequence(args):
    seq = DeRhamSequence(
        args.ns,
        (args.p, args.p, args.p),
        2 * args.p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=args.tol,
        maxiter=args.maxiter,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(_build_map(args))

    operators = None
    operators = assemble_mass_operators(seq, seq.geometry, operators=operators, ks=(3,))
    operators = assemble_incidence_operators(seq, operators=operators, ks=(2,))
    return seq, operators


def _relative_error(y_model: jnp.ndarray, y_exact: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(y_model - y_exact) / jnp.maximum(jnp.linalg.norm(y_exact), 1e-30)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the regular-space k=2 div-div tensor forward model."
    )
    parser.add_argument("--ns", type=_parse_ns, default=(4, 8, 4), help="Grid sizes as nr,nt,nz")
    parser.add_argument("--p", type=int, default=3, help="Spline degree in each direction")
    parser.add_argument("--rank", type=int, default=1, help="CP rank for the scalar 1/J fit")
    parser.add_argument("--cp-maxiter", type=int, default=100, help="Maximum CP ALS iterations")
    parser.add_argument("--cp-tol", type=float, default=1e-9, help="CP ALS tolerance")
    parser.add_argument("--cp-ridge", type=float, default=1e-12, help="CP ALS ridge regularization")
    parser.add_argument("--n-vectors", type=int, default=8, help="Number of random test vectors")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for test vectors")
    parser.add_argument("--dense", action="store_true", help="Also compare dense exact/model matrices")
    parser.add_argument("--extracted", action="store_true", help="Benchmark extracted-space k=2 forward apply instead of regular-space")
    parser.add_argument("--free", action="store_true", help="Use free extraction space when --extracted is set")
    parser.add_argument("--bulk-only", action="store_true", help="When --extracted is set, restrict the benchmark to the extracted tensor bulk indices")
    parser.add_argument("--map-kind", choices=("rotating_ellipse", "identity"), default="rotating_ellipse")
    parser.add_argument("--rotating-eps", type=float, default=0.33)
    parser.add_argument("--rotating-kappa", type=float, default=1.4)
    parser.add_argument("--rotating-r0", type=float, default=1.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    args = parser.parse_args()

    seq, operators = _build_sequence(args)
    model = _assemble_k2_divdiv_regular_tensor_model(
        seq,
        rank=args.rank,
        cp_maxiter=args.cp_maxiter,
        cp_tol=args.cp_tol,
        cp_ridge=args.cp_ridge,
    )

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_vectors)
    dirichlet = not args.free

    if args.extracted:
        full_size = seq.n2_dbc if dirichlet else seq.n2
        bulk_indices = None
        if args.bulk_only:
            bulk_indices = _tensor_block_indices_k2(seq, dirichlet)["bulk"]
            size = int(bulk_indices.shape[0])
        else:
            size = full_size

        def exact_apply(x):
            if bulk_indices is None:
                return apply_stiffness(seq, operators, x, 2, dirichlet=dirichlet)
            full = jnp.zeros((full_size,), dtype=x.dtype)
            full = full.at[bulk_indices].set(x)
            return apply_stiffness(seq, operators, full, 2, dirichlet=dirichlet)[bulk_indices]

        def model_apply(x):
            if bulk_indices is None:
                return _apply_k2_divdiv_extracted_tensor_model(
                    operators,
                    model,
                    x,
                    dirichlet=dirichlet,
                )
            full = jnp.zeros((full_size,), dtype=x.dtype)
            full = full.at[bulk_indices].set(x)
            return _apply_k2_divdiv_extracted_tensor_model(
                operators,
                model,
                full,
                dirichlet=dirichlet,
            )[bulk_indices]
    else:
        size = seq.basis_2.n

        def exact_apply(x):
            return _apply_k2_divdiv_regular_forward(operators, x)

        def model_apply(x):
            return _apply_k2_divdiv_regular_tensor_model(model, x)

    def error_for_key(key):
        x = jax.random.normal(key, (size,), dtype=jnp.float64)
        y_exact = exact_apply(x)
        y_model = model_apply(x)
        return _relative_error(y_model, y_exact)

    rel_errors = jax.vmap(error_for_key)(keys)

    if args.extracted:
        space_label = "extracted-dbc" if dirichlet else "extracted-free"
        if args.bulk_only:
            space_label = f"{space_label}-bulk"
    else:
        space_label = "regular"
    print(f"ns={args.ns}, p={args.p}, map_kind={args.map_kind}, rank={args.rank}, space={space_label}")
    print(f"cp_relative_error={float(model.cp_relative_error):.16g}")
    print(f"cp_final_delta={float(model.cp_final_delta):.16g}")
    print(f"forward_relative_error_mean={float(jnp.mean(rel_errors)):.16g}")
    print(f"forward_relative_error_std={float(jnp.std(rel_errors)):.16g}")
    print(f"forward_relative_error_max={float(jnp.max(rel_errors)):.16g}")

    if args.dense:
        exact = _assemble_dense_from_apply(
            exact_apply,
            size,
        )
        if args.extracted:
            approx = _assemble_dense_from_apply(model_apply, size)
        else:
            approx = _assemble_k2_divdiv_regular_tensor_dense_matrix(model)
        fro_rel = jnp.linalg.norm(approx - exact) / jnp.maximum(jnp.linalg.norm(exact), 1e-30)
        print(f"fro_relative_error={float(fro_rel):.16g}")
        print(f"exact_symmetry_defect={float(jnp.linalg.norm(exact - exact.T)):.16g}")
        print(f"model_symmetry_defect={float(jnp.linalg.norm(approx - approx.T)):.16g}")


if __name__ == "__main__":
    main()