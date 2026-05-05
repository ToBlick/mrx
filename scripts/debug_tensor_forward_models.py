from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    _apply_k0_tensor_hodge_forward_model,
    _assemble_dense_from_apply,
    apply_mass_matrix,
    apply_mass_tensor_forward_model_ops,
    apply_stiffness,
    assemble_hodge_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
)
from mrx.preconditioners import _select_mass_surgery_factors, select_boundary_data


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
    return seq


def _relative_error(y_model: jnp.ndarray, y_exact: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(y_model - y_exact) / jnp.maximum(jnp.linalg.norm(y_exact), 1e-30)


def _mass_cp_summary(factors, k: int):
    if k == 0:
        return {
            "cp_relative_error_max": float(factors.bulk.cp_relative_error),
            "cp_final_delta_max": float(factors.bulk.cp_final_delta),
        }
    if k == 1:
        return {
            "cp_relative_error_max": float(max(factors.arr.cp_relative_error, factors.theta.cp_relative_error, factors.zeta.cp_relative_error)),
            "cp_final_delta_max": float(max(factors.arr.cp_final_delta, factors.theta.cp_final_delta, factors.zeta.cp_final_delta)),
        }
    if k == 2:
        return {
            "cp_relative_error_max": float(max(factors.r_bulk.cp_relative_error, factors.theta.cp_relative_error, factors.zeta.cp_relative_error)),
            "cp_final_delta_max": float(max(factors.r_bulk.cp_final_delta, factors.theta.cp_final_delta, factors.zeta.cp_final_delta)),
        }
    return {
        "cp_relative_error_max": float(factors.cp_relative_error),
        "cp_final_delta_max": float(factors.cp_final_delta),
    }


def _build_problem(args, seq):
    dirichlet = not args.free
    cp_kwargs = {
        "maxiter": args.cp_maxiter,
        "tol": args.cp_tol,
        "ridge": args.cp_ridge,
    }

    if args.problem == "k0-stiffness":
        operators = None
        operators = assemble_tensor_mass_preconditioner(
            seq,
            operators=operators,
            ks=(0,),
            rank=args.rank,
            cp_kwargs=cp_kwargs,
        )
        operators = assemble_mass_operators(seq, seq.geometry, operators=operators, ks=(1,))
        operators = assemble_incidence_operators(seq, operators=operators, ks=(0,))
        operators = assemble_hodge_operators(seq, seq.geometry, operators=operators, ks=(0,))
        pair = operators.k0_tensor_hodge_precond
        factors = select_boundary_data(pair, dirichlet, "Tensor Hodge k=0")
        full_size = seq.n0_dbc if dirichlet else seq.n0
        bulk_indices = jnp.arange(factors.core_size, full_size)

        def exact_apply(x):
            return apply_stiffness(seq, operators, x, 0, dirichlet=dirichlet)

        def model_apply(x):
            return _apply_k0_tensor_hodge_forward_model(seq, operators, x, dirichlet=dirichlet)

        cp_summary = {
            "cp_relative_error_max": float(factors.cp_relative_error),
            "cp_final_delta_max": float(factors.cp_final_delta),
        }
        return operators, exact_apply, model_apply, full_size, bulk_indices, cp_summary

    if not args.problem.startswith("mass-k"):
        raise ValueError(f"Unknown problem {args.problem!r}")

    k = int(args.problem[-1])
    operators = None
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(k,),
        rank=args.rank,
        cp_kwargs=cp_kwargs,
    )
    factors = select_boundary_data(getattr(operators.mass_preconds.tensor, f"k{k}"), dirichlet, f"Tensor mass k={k}")
    full_size = getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")

    if k == 0:
        surgery = _select_mass_surgery_factors(operators.mass_preconds, k, dirichlet)
        bulk_indices = jnp.arange(surgery.surgery_size, full_size)
    elif k in (1, 2):
        surgery = _select_mass_surgery_factors(operators.mass_preconds, k, dirichlet)
        bulk_indices = surgery.bulk_indices
    else:
        bulk_indices = None

    def exact_apply(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    def model_apply(x):
        return apply_mass_tensor_forward_model_ops(seq, operators, x, k, dirichlet=dirichlet)

    cp_summary = _mass_cp_summary(factors, k)
    return operators, exact_apply, model_apply, full_size, bulk_indices, cp_summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark exact vs modeled tensor forward applies for k=0 stiffness and mass operators."
    )
    parser.add_argument("--problem", choices=("k0-stiffness", "mass-k0", "mass-k1", "mass-k2", "mass-k3"), required=True)
    parser.add_argument("--ns", type=_parse_ns, default=(4, 8, 4), help="Grid sizes as nr,nt,nz")
    parser.add_argument("--p", type=int, default=3, help="Spline degree in each direction")
    parser.add_argument("--rank", type=int, default=1, help="Tensor model rank")
    parser.add_argument("--cp-maxiter", type=int, default=100, help="Maximum CP ALS iterations")
    parser.add_argument("--cp-tol", type=float, default=1e-9, help="CP ALS tolerance")
    parser.add_argument("--cp-ridge", type=float, default=1e-12, help="CP ALS ridge regularization")
    parser.add_argument("--n-vectors", type=int, default=8, help="Number of random test vectors")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for test vectors")
    parser.add_argument("--dense", action="store_true", help="Also compare dense exact/model matrices")
    parser.add_argument("--free", action="store_true", help="Use free extraction space")
    parser.add_argument("--bulk-only", action="store_true", help="Restrict the check to the tensor bulk rows/cols when available")
    parser.add_argument("--map-kind", choices=("rotating_ellipse", "identity"), default="rotating_ellipse")
    parser.add_argument("--rotating-eps", type=float, default=0.33)
    parser.add_argument("--rotating-kappa", type=float, default=1.4)
    parser.add_argument("--rotating-r0", type=float, default=1.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    args = parser.parse_args()

    seq = _build_sequence(args)
    _, exact_apply_raw, model_apply_raw, full_size, bulk_indices, cp_summary = _build_problem(args, seq)

    if args.bulk_only:
        if bulk_indices is None:
            raise ValueError(f"bulk-only mode is not available for {args.problem}")
        size = int(bulk_indices.shape[0])

        def exact_apply(x):
            full = jnp.zeros((full_size,), dtype=x.dtype)
            full = full.at[bulk_indices].set(x)
            return exact_apply_raw(full)[bulk_indices]

        def model_apply(x):
            full = jnp.zeros((full_size,), dtype=x.dtype)
            full = full.at[bulk_indices].set(x)
            return model_apply_raw(full)[bulk_indices]
    else:
        size = full_size
        exact_apply = exact_apply_raw
        model_apply = model_apply_raw

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_vectors)

    def error_for_key(key):
        x = jax.random.normal(key, (size,), dtype=jnp.float64)
        y_exact = exact_apply(x)
        y_model = model_apply(x)
        return _relative_error(y_model, y_exact)

    rel_errors = jax.vmap(error_for_key)(keys)

    space_label = "extracted-free" if args.free else "extracted-dbc"
    if args.bulk_only:
        space_label = f"{space_label}-bulk"

    print(f"problem={args.problem}, ns={args.ns}, p={args.p}, map_kind={args.map_kind}, rank={args.rank}, space={space_label}")
    print(f"cp_relative_error_max={cp_summary['cp_relative_error_max']:.16g}")
    print(f"cp_final_delta_max={cp_summary['cp_final_delta_max']:.16g}")
    print(f"forward_relative_error_mean={float(jnp.mean(rel_errors)):.16g}")
    print(f"forward_relative_error_std={float(jnp.std(rel_errors)):.16g}")
    print(f"forward_relative_error_max={float(jnp.max(rel_errors)):.16g}")

    if args.dense:
        exact = _assemble_dense_from_apply(exact_apply, size)
        approx = _assemble_dense_from_apply(model_apply, size)
        fro_rel = jnp.linalg.norm(approx - exact) / jnp.maximum(jnp.linalg.norm(exact), 1e-30)
        print(f"fro_relative_error={float(fro_rel):.16g}")
        print(f"exact_symmetry_defect={float(jnp.linalg.norm(exact - exact.T)):.16g}")
        print(f"model_symmetry_defect={float(jnp.linalg.norm(approx - approx.T)):.16g}")


if __name__ == "__main__":
    main()