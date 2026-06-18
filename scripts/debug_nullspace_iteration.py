"""Debug the nullspace inverse iteration for the solid torus BETTI=(1,1,0,0).

Replaces the opaque jax.lax.while_loop with a plain Python loop so we can
print the inner MINRES iteration count and outer residual after every step.

Run from the repo root:
    .venv/bin/python scripts/debug_nullspace_iteration.py

Adjust N, P, INNER_TOL, ABS_TOL, EPS at the top to taste.
"""
import sys
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import mrx
mrx.MAP_BATCH_SIZE_INNER = 0
mrx.MAP_BATCH_SIZE_OUTER = None

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.nullspace import (
    _initial_guesses,
    _nullspace_shifted_preconditioner,
    init_nullspaces,
    _set_null,
    _commit,
)
from mrx.operators import (
    assemble_incidence_operators,
    assemble_projection_operators,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
    assemble_schur_jacobi_preconditioner,
)

# ---------------------------------------------------------------------------
# Config — tweak these
# ---------------------------------------------------------------------------
N           = 6       # radial / toroidal cell count
P           = 1       # spline degree
EPSILON     = 1 / 3
BETTI       = (1, 1, 0, 0)
EPS         = 1e-6    # shift for (S_k + eps M_k)^{-1}
ABS_TOL       = 1e-9    # outer convergence target on ||L_k v|| (floor ~2-4e-10)
INNER_TOL     = 1e-6    # MINRES tolerance inside each power-iteration step
MAX_OUTER     = 100     # max outer power-iteration steps (matches production default)
USE_V0_COARSE = False   # store v0 in nullspace slot before iteration (1/eps coarse correction)

# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def run_one_pair(seq, operators, k, dirichlet):
    label = f"k={k} dbc={dirichlet}"
    from mrx.nullspace import _n_vectors, _dof_count
    n_vectors = _n_vectors(BETTI, k, dirichlet)
    if n_vectors == 0:
        log(f"  {label}: no harmonic forms, skipping")
        return operators, []

    n_dof = _dof_count(seq, k, dirichlet)
    log(f"  {label}: n_vectors={n_vectors}  n_dof={n_dof}")

    guesses = _initial_guesses(seq, operators, k, dirichlet, n_vectors)
    shifted_precond = _nullspace_shifted_preconditioner(k)

    found = []
    all_iters = []

    for idx in range(n_vectors):
        log(f"    Vector {idx}/{n_vectors}")
        v = guesses[idx] if (guesses and guesses[idx] is not None) else (
            jax.random.normal(jax.random.PRNGKey(idx), (n_dof,)))

        # M-orthogonalise against already-found vectors
        for u in found:
            v = v - (u @ seq.apply_mass_matrix(v, k, dirichlet=dirichlet,
                                                operators=operators)) * u
        v = v / seq.l2_norm(v, k, dirichlet=dirichlet)

        Lv = seq.apply_laplacian(v, k, dirichlet=dirichlet, operators=operators)
        res0 = seq.l2_norm(Lv, k, dirichlet=dirichlet)
        jax.debug.print("      initial  ||Lv|| = {r:.3e}", r=res0)

        if float(res0) <= ABS_TOL:
            log(f"      accepted initial guess without iteration")
            found.append(v)
            all_iters.append((0, float(res0)))
            continue

        # Optionally seed the nullspace slot with v0 so the production
        # 1/eps coarse correction uses it during the inner MINRES solves.
        if USE_V0_COARSE:
            ops_with_v0 = _commit(seq, _set_null(
                operators, k, dirichlet, jnp.stack([v])))
        else:
            ops_with_v0 = operators
        log(f"      use_v0_coarse={USE_V0_COARSE}")

        # Mimic production: single jax.lax.while_loop compiled once, with
        # jax.debug.print for per-step diagnostics.
        def body_fn(state):
            v, res, i = state
            Mv = seq.apply_mass_matrix(v, k, dirichlet=dirichlet, operators=ops_with_v0)
            w, inner_info = seq.apply_inverse_shifted_laplacian(
                Mv, k, EPS,
                dirichlet=dirichlet,
                guess=v,
                operators=ops_with_v0,
                preconditioner=shifted_precond,
                tol=INNER_TOL,
                use_harmonic_coarse=USE_V0_COARSE,
                return_info=True,
            )
            for u in found:
                w = w - (u @ seq.apply_mass_matrix(w, k, dirichlet=dirichlet,
                                                    operators=ops_with_v0)) * u
            w = w / seq.l2_norm(w, k, dirichlet=dirichlet)
            Lw = seq.apply_laplacian(w, k, dirichlet=dirichlet, operators=ops_with_v0)
            res_new = seq.l2_norm(Lw, k, dirichlet=dirichlet)
            inner_iters = jnp.abs(inner_info)
            inner_conv  = inner_info < 0
            jax.debug.print(
                "      step {i:3d}  ||Lv|| = {r:.3e}  inner_iters={it}  inner_conv={ic}",
                i=i + 1, r=res_new, it=inner_iters, ic=inner_conv)
            return w, res_new, i + 1

        def cond_fn(state):
            _, res, i = state
            return (res > ABS_TOL) & (i < MAX_OUTER)

        v_final, res_final, n_iters = jax.lax.while_loop(
            cond_fn, body_fn, (v, res0, jnp.zeros((), dtype=jnp.int32)))

        if float(res_final) <= ABS_TOL:
            jax.debug.print("      converged at step {i}", i=n_iters)
        else:
            jax.debug.print(
                "      WARNING: did not converge in {m} outer steps  ||Lv||={r:.3e}",
                m=MAX_OUTER, r=res_final)

        found.append(v_final)
        all_iters.append((int(n_iters), float(res_final)))

    # Store on operators
    stacked = jnp.stack(found)
    operators = _commit(seq, _set_null(operators, k, dirichlet, stacked))
    return operators, all_iters


def main():
    ns = (N, 2 * N, N)
    ps = (P, P, P)
    q  = 2 * P
    types = ("clamped", "periodic", "periodic")
    cp_kwargs = {"maxiter": 100, "tol": 1e-9, "ridge": 1e-12}

    log(f"=== Nullspace debug: n={N} p={P} eps={EPS} abs_tol={ABS_TOL} inner_tol={INNER_TOL} ===")

    F = toroid_map(epsilon=EPSILON)

    log("Building sequence...")
    seq = DeRhamSequence(ns, ps, q, types, polar=True,
                         tol=1e-10, maxiter=50000,
                         betti_numbers=BETTI)
    seq.set_map(F)
    seq.evaluate_1d()
    log(f"  n0={seq.n0} n1={seq.n1} n2={seq.n2} n3={seq.n3}")

    log("Assembling preconditioners...")
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1, 2, 3), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_tensor_laplacian_preconditioner(seq, ops, ks=(0,), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(1, 2, 3),
                                               dirichlet_variants=(False, True, True))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    log("  Assembly done")

    ops = init_nullspaces(seq, ops, BETTI)
    ops = seq.set_operators(ops)
    ops_init = ops  # save clean slate to reset between runs

    for use_coarse in (False, True):
        log(f"\n{'='*60}")
        log(f"  USE_V0_COARSE = {use_coarse}")
        log(f"{'='*60}")
        # Reset nullspaces to the clean initial state for each run.
        ops = ops_init
        seq.operators = ops_init

        global USE_V0_COARSE
        USE_V0_COARSE = use_coarse

        for k, dirichlet in [(0, False), (1, False), (2, True), (3, True)]:
            log(f"\n--- {k=}, {dirichlet=} ---")
            t0 = time.perf_counter()
            ops, iters = run_one_pair(seq, ops, k, dirichlet)
            log(f"  Done in {time.perf_counter()-t0:.2f}s  iters={iters}")

    log("\nAll nullspaces computed successfully.")


if __name__ == "__main__":
    main()
