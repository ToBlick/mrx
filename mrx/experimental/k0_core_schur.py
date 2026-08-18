"""k=0 Laplacian core-Schur preconditioner -- retired from the production path.

Moved out of mrx/operators.py on 2026-08-18 when the production k=0
Laplacian preconditioner became BLOCK DIAGONAL (dense core inverse + the
modal-radial bulk atom, no core<->bulk coupling).

This is **kept, not deleted**: on axisymmetric geometry the Schur factorization
is genuinely better per solve, and it remains the right choice for anyone who
can amortize its setup.

Measured 2026-08-18, full grid, dbc, 16x32x32, modal-radial atom in both:

    geometry        its (blockdiag / Schur)   solve (bd / Schur)   Schur setup
    toroid                26 / 14               268 / 186 ms          52.5 s
    rot-ellipse           47 / 48               491 / 663 ms          56.2 s

and the coupling's worth across resolutions (block-diag its / Schur its):

    toroid        1.62x -> 1.85x -> 1.86x      (8x16x16 -> 12x24x24 -> 16x32x32)
    rot-ellipse   1.00x -> 0.98x -> 0.98x
    W7-X          0.97x -> 0.94x -> 0.89x

So it pays on the toroid and not at all on stellarators, which is why it left
the default path. Two costs came with it: 3 n_z exact bulk CG solves per BC
at assembly (~55 s at 16x32x32), and core_coupling, a dense
bulk x 3 n_z block -- 780 MB at 64x128x64, the last O(N n_z) term in the
code and the same pathology that retired coupling_sb on the mass side.

Both the additive-FD atom (_assemble_k0_greville_bulk_factors) and the
modal-radial atom remain importable from mrx.operators, and schur_inv is
built from exact bulk solves so it is atom-independent -- either atom can be
dropped in.
"""
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.operators import (K0TensorHodgePreconditionerFactors,  # noqa: F401
                           SequenceOperators,
                           _apply_k0_tensor_hodge_bulk_inverse,
                           _apply_k0_tensor_hodge_core_block,
                           _apply_k0_tensor_hodge_surgery_to_bulk_coupling,
                           _assemble_dense_from_apply,
                           _assemble_k0_greville_bulk_factors,
                           _build_k0_tensor_hodge_preconditioner_factors,
                           _core_size, _symmetrize, apply_stiffness)
from mrx.preconditioners import BoundaryConditionPair, _symmetric_pseudoinverse
from mrx.solvers import solve_singular_cg

__all__ = ["assemble_k0_core_schur_preconditioner"]


def assemble_k0_core_schur_preconditioner(
        seq, operators: SequenceOperators, *,
        precompute_coupling: bool = True,
        dirichlet_flags: tuple[bool, ...] = (False, True)) -> BoundaryConditionPair:
    pair = BoundaryConditionPair()
    core_size = _core_size(seq)

    for dirichlet in dirichlet_flags:
        bulk_data = _assemble_k0_greville_bulk_factors(seq, dirichlet=dirichlet)

        ass = _symmetrize(_assemble_dense_from_apply(
            lambda rhs_c, seq=seq, operators=operators, core_size=core_size, dirichlet=dirichlet:
            _apply_k0_tensor_hodge_core_block(seq, operators, core_size, rhs_c, dirichlet=dirichlet),
            core_size,
            sequential=True,
        ))
        surgery_to_bulk_apply = lambda rhs_c, seq=seq, operators=operators, core_size=core_size, dirichlet=dirichlet: _apply_k0_tensor_hodge_surgery_to_bulk_coupling(seq, operators, core_size, rhs_c, dirichlet=dirichlet)

        # Dense core->bulk coupling block C0 (bulk x core), one matrix-free
        # stiffness apply per core DOF. K_0 is symmetric, so the bulk->core
        # block is exactly C0^T. Always built here (the Schur probe below
        # consumes it); it is only STORED in the factors when
        # precompute_coupling is set.
        core_coupling = _assemble_dense_from_apply(
            surgery_to_bulk_apply,
            core_size,
            sequential=True,
        )

        # Core Schur rebuilt with EXACT bulk solves:
        # schur = ass - C0^T A_bb^{-1} C0, one atom-preconditioned CG per
        # core DOF (3*n_zeta solves, assembly-time only -- no runtime
        # Krylov nesting; the result is a fixed dense matrix). The exact
        # Schur of the SPD stiffness is PSD by construction, which retired
        # the 2026-08-13 collocated-probe one-sidedness rule (and with it
        # the collocated atom itself); truncated CG even errs on the PSD
        # side (partial iterates UNDERestimate <c, A_bb^{-1} c>).
        # Validated 2026-08-14: toroid 22/25 its (vs 22/31 atom-probed),
        # W7-X 12,24,24: 53 dbc / 80 free (vs 56/87) at equal assembly cost.
        runtime_bulk_factors = _build_k0_tensor_hodge_preconditioner_factors(
            core_size=core_size,
            schur_inv=jnp.eye(core_size, dtype=jnp.float64),
            bulk_data=bulk_data,
        )
        atom_apply = lambda rhs_b, f=runtime_bulk_factors: _apply_k0_tensor_hodge_bulk_inverse(f, rhs_b)

        def bulk_operator(x_b, seq=seq, operators=operators,
                          core_size=core_size, dirichlet=dirichlet):
            size = seq.n0_dbc if dirichlet else seq.n0
            full = jnp.zeros((size,), dtype=x_b.dtype)
            full = full.at[core_size:].set(x_b)
            return apply_stiffness(seq, operators, full, 0,
                                   dirichlet=dirichlet)[core_size:]

        bulk_solve = jax.jit(lambda b: solve_singular_cg(
            bulk_operator, b, precond_matvec=atom_apply,
            maxiter=1000, tol=1e-12)[0])
        solve_cols = []
        for i in range(core_size):
            c_i = core_coupling[:, i]
            y_i = bulk_solve(c_i)
            b_norm = float(jnp.linalg.norm(c_i))
            rel_res = float(jnp.linalg.norm(bulk_operator(y_i) - c_i)) / max(b_norm, 1e-300)
            if b_norm > 0.0 and rel_res > 1e-8:
                warnings.warn(
                    f"k=0 core-Schur exact probe: bulk CG for core DOF {i} "
                    f"stalled at rel res {rel_res:.2e} (dirichlet={dirichlet}); "
                    "the rebuilt Schur stays on the PSD side but may lose accuracy.")
            solve_cols.append(y_i)
        bulk_solves = jnp.stack(solve_cols, axis=1)
        schur = _symmetrize(ass - core_coupling.T @ bulk_solves)

        schur_inv = _symmetric_pseudoinverse(schur)

        factors = _build_k0_tensor_hodge_preconditioner_factors(
            core_size=core_size,
            schur_inv=schur_inv,
            bulk_data=bulk_data,
            precompute_coupling=precompute_coupling,
            core_coupling=core_coupling if precompute_coupling else None,
        )
        pair = eqx.tree_at(
            lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
            pair,
            factors,
            is_leaf=lambda x: x is None,
        )
    return pair
