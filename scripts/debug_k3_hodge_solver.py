"""Debug the k = 3 Hodge-Laplacian saddle-point solve.

Layered diagnostics, each isolating one potential failure mode:

  (A) Nullspace accuracy: does ``compute_nullspaces_iterative`` actually
      produce ``v`` with ``||L_3 v|| <= tol``?
  (B) RHS sanity: after nullspace deflation the RHS should be in
      range(L_3); check that ``v^T M_3 b == 0`` for every harmonic ``v``.
  (C) Direct operator application: ``L_3 u`` on a simple test vector.
      Compare saddle-point apply vs. ``D_2 M_2^{-1} D_2^T`` (explicit
      Schur form via CG inner solve).
  (D) Preconditioner sanity for each ``precond_kind``:
        - apply P^{-1} to a random vector, check output is finite and
          has expected norm;
        - check SPD heuristically: ``v^T P^{-1} v > 0``;
        - check the individual building blocks of 'hx'.
  (E) MINRES behaviour: run with increasing ``maxiter``, print
      MINRES ``info``, initial/final residuals, and whether convergence
      actually happened or the solver reported ``0`` because the RHS
      was zero (or tiny) in the preconditioned norm.

Run from the project root:

    .venv/bin/python scripts/debug_k3_hodge_solver.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from mrx.operators import (apply_derivative_matrix,
                           apply_hodge_hx_preconditioner,
                           apply_hodge_kron_preconditioner,
                           apply_hodge_laplacian,
                           apply_hodge_laplacian_preconditioner,
                           apply_inverse_mass_matrix,
                           apply_mass_kron_preconditioner, apply_mass_matrix,
                           apply_projection_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


jax.config.update("jax_enable_x64", True)

from mrx.nullspace import get_nullspace  # noqa: E402
from mrx.operators import _hodge_diaginv  # noqa: E402
from scripts.benchmark_preconditioners import build_sequence  # noqa: E402

K = 3
DBC = True


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fnum(x):
    return f"{float(x):+.6e}"


def section_A_nullspace(seq, ops):
    banner("(A) Nullspace accuracy")
    vs = get_nullspace(ops, K, DBC)
    print(f"  n_harmonic(k={K}, dbc={DBC}) = {vs.shape[0]}")
    for i, v in enumerate(vs):
        Lv = apply_hodge_laplacian(seq, ops, v, K, dirichlet=DBC)
        nv = float(jnp.linalg.norm(v))
        nLv = float(jnp.linalg.norm(Lv))
        nv_l2 = float(seq.l2_norm(v, K, dirichlet=DBC))
        nLv_l2 = float(seq.l2_norm(Lv, K, dirichlet=DBC))
        print(f"  v[{i}]:  ||v||_2={nv:.3e}  ||v||_L2={nv_l2:.3e}  "
              f"||Lv||_2={nLv:.3e}  ||Lv||_L2={nLv_l2:.3e}")
    return vs


def section_B_rhs(seq, ops, vs):
    banner("(B) RHS construction and deflation")
    key = jax.random.PRNGKey(42)
    b_raw = jax.random.normal(key, (seq.n3_dbc,))
    # Deflate against each harmonic (M-orthogonal deflation).
    b = b_raw
    for v in vs:
        Mv = apply_mass_matrix(seq, ops, v, K, dirichlet=DBC)
        b = b - jnp.dot(v, b) * Mv
    print(f"  ||b_raw||_2 = {float(jnp.linalg.norm(b_raw)):.6e}")
    print(f"  ||b_deflated||_2 = {float(jnp.linalg.norm(b)):.6e}")
    # After deflation, <v, b> should be 0 in the l2 sense *and* v^T b should
    # be 0 in the l2 sense (dual pairing).
    for i, v in enumerate(vs):
        print(f"  <v[{i}], b>            = {fnum(jnp.dot(v, b))}")
    return b


def section_C_operator(seq, ops, b):
    banner("(C) Direct L_3 operator action")
    # L_3 b should be in the dual 3-form space and finite.
    Lb = apply_hodge_laplacian(seq, ops, b, K, dirichlet=DBC)
    print(f"  ||b||_2     = {float(jnp.linalg.norm(b)):.6e}")
    print(f"  ||L_3 b||_2 = {float(jnp.linalg.norm(Lb)):.6e}")
    # Sanity: Rayleigh quotient b^T L_3 b / b^T M_3 b > 0
    Mb = apply_mass_matrix(seq, ops, b, K, dirichlet=DBC)
    rq_num = float(jnp.dot(b, Lb))
    rq_den = float(jnp.dot(b, Mb))
    print(f"  b^T L_3 b   = {rq_num:.6e}")
    print(f"  b^T M_3 b   = {rq_den:.6e}")
    print(f"  Rayleigh    = {rq_num / rq_den:.6e}")


def section_D_preconditioners(seq, ops):
    banner("(D) Preconditioner sanity checks (random input)")
    key = jax.random.PRNGKey(7)
    v = jax.random.normal(key, (seq.n3_dbc,))
    v = v / jnp.linalg.norm(v)
    print(f"  input: ||v||_2 = {float(jnp.linalg.norm(v)):.6e}")
    for kind in ("none", "jacobi", "hx"):
        try:
            Pv = apply_hodge_laplacian_preconditioner(
                seq, ops, v, K, dirichlet=DBC, kind=kind)
        except Exception as e:
            print(f"  {kind:>10s}: FAILED -> {e}")
            continue
        vdotPv = float(jnp.dot(v, Pv))
        print(f"  {kind:>10s}: ||P v||_2 = {float(jnp.linalg.norm(Pv)):.3e}  "
              f"v^T P v = {vdotPv:+.3e}  finite={bool(jnp.all(jnp.isfinite(Pv)))}")

    banner("(D') HX preconditioner building blocks")
    # Smoother: diag(L_3)^{-1} v
    dinv = _hodge_diaginv(ops, 3, dirichlet=True)
    print(f"  diag(L_3)^{{-1}} min={float(jnp.min(dinv)):.3e}  "
          f"max={float(jnp.max(dinv)):.3e}  "
          f"any_nonfinite={bool(jnp.any(~jnp.isfinite(dinv)))}")
    smooth = dinv * v
    print(f"  smoother(v): ||.||_2 = {float(jnp.linalg.norm(smooth)):.3e}")

    # M_3_kron_inv(v)
    w0 = apply_mass_kron_preconditioner(seq, ops, v, 3, dirichlet=True)
    print(f"  step 1: tilde(M_3)^-1 v : ||.||_2 = "
          f"{float(jnp.linalg.norm(w0)):.3e}")
    # M_03 (3->0): using apply_projection_matrix(k_in=0, k_out=3, in_dbc=T, out_dbc=F)
    w1 = apply_projection_matrix(
        seq, ops, w0, k_in=0, k_out=3,
        dirichlet_in=True, dirichlet_out=False)
    print(f"  step 2: M_{{03}} (3dbc -> 0nbc): shape={w1.shape}  "
          f"||.||_2 = {float(jnp.linalg.norm(w1)):.3e}")
    # L_0^{-1} via kron
    w2 = apply_hodge_kron_preconditioner(seq, ops, w1, 0, dirichlet=False)
    print(f"  step 3: tilde(L_0)^-1 : ||.||_2 = "
          f"{float(jnp.linalg.norm(w2)):.3e}")
    # M_30
    w3 = apply_projection_matrix(
        seq, ops, w2, k_in=3, k_out=0,
        dirichlet_in=False, dirichlet_out=True)
    print(f"  step 4: M_{{30}} (0nbc -> 3dbc): shape={w3.shape}  "
          f"||.||_2 = {float(jnp.linalg.norm(w3)):.3e}")
    w4 = apply_mass_kron_preconditioner(seq, ops, w3, 3, dirichlet=True)
    print(f"  step 5: tilde(M_3)^-1 : ||.||_2 = "
          f"{float(jnp.linalg.norm(w4)):.3e}")
    full = smooth + w4
    direct = apply_hodge_hx_preconditioner(seq, ops, v, 3, dirichlet=True)
    print(f"  reconstructed HX: ||.||_2 = {float(jnp.linalg.norm(full)):.3e}")
    print(
        f"  direct HX       : ||.||_2 = {float(jnp.linalg.norm(direct)):.3e}")
    print(f"  ||direct - reconstructed|| = "
          f"{float(jnp.linalg.norm(direct - full)):.3e}")


def section_E_minres(seq, ops, b, vs):
    banner("(E) MINRES behaviour per preconditioner")
    from mrx.operators import apply_inverse_shifted_hodge_laplacian

    # Initial residual norms (x0 = 0).
    print(f"  ||b||_2               = {float(jnp.linalg.norm(b)):.6e}")
    # What MINRES uses as its relative-tol denominator:
    # bnorm = sqrt(b^T M^{-1} b) with M = block-diag preconditioner.
    for kind in ("none", "jacobi", "hx"):
        try:
            Pb = apply_hodge_laplacian_preconditioner(
                seq, ops, b, K, dirichlet=DBC, kind=kind)
        except Exception as e:
            print(f"  {kind:>10s}: precond(b) FAILED -> {e}")
            continue
        bnorm_prec = float(jnp.sqrt(jnp.abs(jnp.dot(b, Pb))))
        print(f"  {kind:>10s}: sqrt(b^T P b) = {bnorm_prec:.6e}  "
              f"(MINRES rel-tol denom; 0 here => instant 'convergence')")

    for kind in ("none", "jacobi", "hx"):
        try:
            u, info = apply_inverse_shifted_hodge_laplacian(
                seq, ops, b, K, 0.0, dirichlet=DBC,
                precond_kind=kind, return_info=True)
        except Exception as e:
            print(f"  {kind:>10s}: solve FAILED -> {e}")
            continue
        r = b - apply_hodge_laplacian(seq, ops, u, K, dirichlet=DBC)
        # Deflate the residual against the nullspace (that mode is outside
        # range(L_3) and not meaningful).
        for v in vs:
            Mv = apply_mass_matrix(seq, ops, v, K, dirichlet=DBC)
            r = r - jnp.dot(v, r) * Mv / jnp.dot(v, Mv)
        rel = float(jnp.linalg.norm(r) / jnp.linalg.norm(b))
        iters = int(jnp.abs(info))
        status = "converged" if int(info) <= 0 else "STALLED"
        print(f"  {kind:>10s}: {status}  iters={iters:4d}  "
              f"||u||_2={float(jnp.linalg.norm(u)):.3e}  "
              f"||r||_2/||b||_2 = {rel:.3e}")


def main():
    print("Building sequence...")
    seq, ops = build_sequence()
    print(f"  n3_dbc = {seq.n3_dbc}  n2_dbc = {seq.n2_dbc}  "
          f"n0 = {seq.n0}  tol = {seq.tol}  maxiter = {seq.maxiter}")

    vs = section_A_nullspace(seq, ops)
    b = section_B_rhs(seq, ops, vs)
    section_C_operator(seq, ops, b)
    section_D_preconditioners(seq, ops)
    section_E_minres(seq, ops, b, vs)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
