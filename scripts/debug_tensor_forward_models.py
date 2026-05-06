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
    apply_mass_tensor_preconditioner_ops,
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
    def add_block_metadata(summary: dict[str, float], block_factors, prefix: str = ""):
        summary[f"{prefix}term_count"] = float(len(block_factors.term_r))
        summary[f"{prefix}richardson_steps"] = float(block_factors.richardson_steps)
        summary[f"{prefix}richardson_omega"] = float(block_factors.richardson_omega)
        summary[f"{prefix}has_direct_inv"] = float(block_factors.direct_inv_r is not None)
        summary[f"{prefix}has_modal_basis"] = float(block_factors.modal_basis_r is not None)
        summary[f"{prefix}has_backbone_inv"] = float(block_factors.split_backbone_inv_r is not None)

    def add_split_summary(summary: dict[str, float], block_factors, prefix: str = ""):
        if block_factors.split_backbone_relative_norm is None:
            return
        summary[f"{prefix}split_backbone_relative_norm"] = float(block_factors.split_backbone_relative_norm)
        summary[f"{prefix}split_correction_relative_norm"] = float(block_factors.split_correction_relative_norm)
        summary[f"{prefix}split_correction_over_backbone"] = float(block_factors.split_correction_over_backbone)
        summary[f"{prefix}split_backbone_residual_relative"] = float(block_factors.split_backbone_residual_relative)

    if k == 0:
        summary = {
            "cp_relative_error_max": float(factors.bulk.cp_relative_error),
            "cp_final_delta_max": float(factors.bulk.cp_final_delta),
        }
        add_block_metadata(summary, factors.bulk)
        add_split_summary(summary, factors.bulk)
        return summary
    if k == 1:
        summary = {
            "cp_relative_error_max": float(max(factors.arr.cp_relative_error, factors.theta.cp_relative_error, factors.zeta.cp_relative_error)),
            "cp_final_delta_max": float(max(factors.arr.cp_final_delta, factors.theta.cp_final_delta, factors.zeta.cp_final_delta)),
        }
        add_block_metadata(summary, factors.arr, "arr_")
        add_block_metadata(summary, factors.theta, "theta_")
        add_block_metadata(summary, factors.zeta, "zeta_")
        add_split_summary(summary, factors.arr, "arr_")
        add_split_summary(summary, factors.theta, "theta_")
        add_split_summary(summary, factors.zeta, "zeta_")
        return summary
    if k == 2:
        summary = {
            "cp_relative_error_max": float(max(factors.r_bulk.cp_relative_error, factors.theta.cp_relative_error, factors.zeta.cp_relative_error)),
            "cp_final_delta_max": float(max(factors.r_bulk.cp_final_delta, factors.theta.cp_final_delta, factors.zeta.cp_final_delta)),
        }
        add_block_metadata(summary, factors.r_bulk, "r_bulk_")
        add_block_metadata(summary, factors.theta, "theta_")
        add_block_metadata(summary, factors.zeta, "zeta_")
        add_split_summary(summary, factors.r_bulk, "r_bulk_")
        add_split_summary(summary, factors.theta, "theta_")
        add_split_summary(summary, factors.zeta, "zeta_")
        return summary
    summary = {
        "cp_relative_error_max": float(factors.cp_relative_error),
        "cp_final_delta_max": float(factors.cp_final_delta),
    }
    add_block_metadata(summary, factors)
    add_split_summary(summary, factors)
    return summary


def _build_problem(args, seq):
    dirichlet = not args.free
    cp_kwargs = {
        "maxiter": args.cp_maxiter,
        "tol": args.cp_tol,
        "ridge": args.cp_ridge,
    }
    if args.problem.startswith("mass-k"):
        cp_kwargs["fit_strategy"] = args.fit_strategy
        cp_kwargs["richardson_steps"] = args.richardson_steps
        cp_kwargs["richardson_omega"] = args.richardson_omega

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


def _build_mass_preconditioner_apply(args, seq, *, reference_richardson_steps: int | None = None):
    if not args.problem.startswith("mass-k"):
        raise ValueError("Preconditioner comparison is only available for mass problems")

    dirichlet = not args.free
    cp_kwargs = {
        "maxiter": args.cp_maxiter,
        "tol": args.cp_tol,
        "ridge": args.cp_ridge,
        "fit_strategy": args.fit_strategy,
        "richardson_steps": args.richardson_steps if reference_richardson_steps is None else reference_richardson_steps,
        "richardson_omega": args.richardson_omega,
    }
    k = int(args.problem[-1])
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=None,
        ks=(k,),
        rank=args.rank,
        cp_kwargs=cp_kwargs,
    )
    full_size = getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")

    def preconditioner_apply(x):
        return apply_mass_tensor_preconditioner_ops(seq, operators, x, k, dirichlet=dirichlet)

    return preconditioner_apply, full_size


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
    parser.add_argument("--fit-strategy", choices=("multiplicative", "split"), default="multiplicative", help="Mass-only tensor fit strategy")
    parser.add_argument("--richardson-steps", type=int, default=0, help="Mass-only tensor Richardson steps")
    parser.add_argument("--richardson-omega", type=float, default=1.0, help="Mass-only tensor Richardson damping")
    parser.add_argument("--n-vectors", type=int, default=8, help="Number of random test vectors")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for test vectors")
    parser.add_argument("--dense", action="store_true", help="Also compare dense exact/model matrices")
    parser.add_argument("--compare-preconditioner", action="store_true", help="Also compare tensor preconditioner outputs against a reference Richardson step count")
    parser.add_argument("--reference-richardson-steps", type=int, default=0, help="Reference Richardson steps for preconditioner comparison")
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

    print(f"problem={args.problem}, ns={args.ns}, p={args.p}, map_kind={args.map_kind}, rank={args.rank}, fit_strategy={args.fit_strategy}, richardson_steps={args.richardson_steps}, richardson_omega={args.richardson_omega}, space={space_label}")
    print(f"cp_relative_error_max={cp_summary['cp_relative_error_max']:.16g}")
    print(f"cp_final_delta_max={cp_summary['cp_final_delta_max']:.16g}")
    for key, value in cp_summary.items():
        if key in ("cp_relative_error_max", "cp_final_delta_max"):
            continue
        print(f"{key}={value:.16g}")
    print(f"forward_relative_error_mean={float(jnp.mean(rel_errors)):.16g}")
    print(f"forward_relative_error_std={float(jnp.std(rel_errors)):.16g}")
    print(f"forward_relative_error_max={float(jnp.max(rel_errors)):.16g}")

    if args.compare_preconditioner:
        if not args.problem.startswith("mass-k"):
            raise ValueError("--compare-preconditioner is only available for mass problems")

        preconditioner_apply, preconditioner_size = _build_mass_preconditioner_apply(args, seq)
        reference_apply, reference_size = _build_mass_preconditioner_apply(
            args,
            seq,
            reference_richardson_steps=args.reference_richardson_steps,
        )
        if preconditioner_size != reference_size:
            raise ValueError("Preconditioner comparison built mismatched vector sizes")

        def preconditioner_error_for_key(key):
            x = jax.random.normal(key, (preconditioner_size,), dtype=jnp.float64)
            y = preconditioner_apply(x)
            y_ref = reference_apply(x)
            return _relative_error(y, y_ref)

        preconditioner_rel_errors = jax.vmap(preconditioner_error_for_key)(keys)
        print(f"preconditioner_reference_richardson_steps={args.reference_richardson_steps}")
        print(f"preconditioner_relative_difference_mean={float(jnp.mean(preconditioner_rel_errors)):.16g}")
        print(f"preconditioner_relative_difference_std={float(jnp.std(preconditioner_rel_errors)):.16g}")
        print(f"preconditioner_relative_difference_max={float(jnp.max(preconditioner_rel_errors)):.16g}")

    if args.dense:
        exact = _assemble_dense_from_apply(exact_apply, size)
        approx = _assemble_dense_from_apply(model_apply, size)
        fro_rel = jnp.linalg.norm(approx - exact) / jnp.maximum(jnp.linalg.norm(exact), 1e-30)
        print(f"fro_relative_error={float(fro_rel):.16g}")
        print(f"exact_symmetry_defect={float(jnp.linalg.norm(exact - exact.T)):.16g}")
        print(f"model_symmetry_defect={float(jnp.linalg.norm(approx - approx.T)):.16g}")


if __name__ == "__main__":
    main()