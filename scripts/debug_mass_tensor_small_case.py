from __future__ import annotations

import traceback

import jax

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import assemble_mass_operators, assemble_tensor_mass_preconditioner
from mrx.preconditioners import (
    _arr_shape_k1,
    _component_sizes_k1,
    _component_sizes_k2,
    _r_bulk_shape_k2,
    _tensor_block_indices_k1,
    _tensor_block_indices_k2,
    _theta_bulk_shape_k1,
    _theta_shape_k2,
    _zeta_bulk_shape_k1,
    _zeta_shape_k2,
)


jax.config.update("jax_enable_x64", True)


def build_small_case(ns=(5, 5, 3), p=2):
    seq = DeRhamSequence(
        ns,
        (p, p, p),
        2 * p,
        ("clamped", "periodic", "periodic"),
        lambda x: x,
        polar=True,
        tol=1e-10,
        maxiter=1000,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3))
    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1, 2, 3))
    return seq, operators


def print_k1_geometry(seq):
    print("k=1 component sizes")
    for dirichlet in (False, True):
        print(f"  dirichlet={dirichlet}")
        print(f"    component_sizes={_component_sizes_k1(seq, dirichlet)}")
        block_indices = _tensor_block_indices_k1(seq, dirichlet)
        for key, value in block_indices.items():
            if hasattr(value, "shape"):
                print(f"    {key}: shape={value.shape}")
            else:
                print(f"    {key}: {value}")
        print(f"    arr_shape={_arr_shape_k1(seq, dirichlet)}")
        print(f"    theta_bulk_shape={_theta_bulk_shape_k1(seq, dirichlet)}")
        print(f"    zeta_bulk_shape={_zeta_bulk_shape_k1(seq, dirichlet)}")


def print_k2_geometry(seq):
    print("k=2 component sizes")
    for dirichlet in (False, True):
        print(f"  dirichlet={dirichlet}")
        print(f"    component_sizes={_component_sizes_k2(seq, dirichlet)}")
        block_indices = _tensor_block_indices_k2(seq, dirichlet)
        for key, value in block_indices.items():
            if hasattr(value, "shape"):
                print(f"    {key}: shape={value.shape}")
            else:
                print(f"    {key}: {value}")
        print(f"    r_bulk_shape={_r_bulk_shape_k2(seq, dirichlet)}")
        print(f"    theta_shape={_theta_shape_k2(seq, dirichlet)}")
        print(f"    zeta_shape={_zeta_shape_k2(seq, dirichlet)}")


def try_tensor_assembly(seq, operators):
    print("Trying tensor mass assembly one k at a time")
    for k in (0, 1, 2, 3):
        print(f"\n--- k={k} ---")
        try:
            assemble_tensor_mass_preconditioner(
                seq,
                operators=operators,
                ks=(k,),
                rank=3,
                cp_kwargs={"tol": 1e-8, "maxiter": 200},
            )
            print("success")
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()


def main():
    seq, operators = build_small_case()
    print(f"ns={seq.ns}, ps={seq.ps}")
    print_k1_geometry(seq)
    print()
    print_k2_geometry(seq)
    print()
    try_tensor_assembly(seq, operators)


if __name__ == "__main__":
    main()