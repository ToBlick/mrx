"""Wall clock, not iteration count: block-diag vs Schur+coupling, k=0 stiffness.

Iterations are the wrong currency here. The Schur factorization applies the bulk
atom TWICE plus two coupling applies per iteration; block-diagonal applies it
once and adds only a small dense core solve. So a lower iteration count does not
imply a faster solve, and this measures the thing that actually matters.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.experimental.modal_radial import modal_perk_apply, modal_perk_bulk_data
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (_apply_k0_tensor_hodge_core_block,
                           _apply_k0_tensor_hodge_surgery_to_bulk_coupling,
                           _assemble_dense_from_apply, _core_size, _symmetrize,
                           apply_stiffness, assemble_incidence_operators)
from mrx.preconditioners import _symmetric_pseudoinverse
from mrx.solvers import solve_singular_cg

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=int, nargs=3, default=(16, 32, 32))
a = ap.parse_args()
NS, P = tuple(a.ns), 3
mrx.MAP_BATCH_SIZE_INNER = 256
TYPES = ("clamped", "periodic", "periodic")
print(f"timing: ns={NS}\n", flush=True)


def pcg_timed(A, b, M, tol=1e-10, maxit=4000):
    x = jnp.zeros_like(b)
    r = b - A(x)
    z = M(r)
    q = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    jax.block_until_ready(z)
    t0 = time.perf_counter()
    for i in range(1, maxit + 1):
        Aq = A(q)
        den = float(q @ Aq)
        if den <= 0:
            return -i, time.perf_counter() - t0
        al = rz / den
        x = x + al * q
        r = r - al * Aq
        if float(jnp.linalg.norm(r)) / nb < tol:
            jax.block_until_ready(x)
            return i, time.perf_counter() - t0
        z = M(r)
        rzn = float(r @ z)
        q = z + (rzn / rz) * q
        rz = rzn
    return maxit, time.perf_counter() - t0


for gname, mk in (("toroid", lambda: toroid_map(epsilon=1 / 3, R0=1.0)),
                  ("rot-ellipse", lambda: rotating_ellipse_map(
                      eps=0.33, kappa=1.5, nfp=3))):
    seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True, tol=1e-12,
                         maxiter=1000, betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.set_map(mk())
    ops = assemble_incidence_operators(seq)
    seq.set_operators(ops)
    cs = _core_size(seq)
    size = int(seq.n0_dbc)

    def K(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=True)
    K(jnp.zeros(size))

    A_cc = _symmetrize(_assemble_dense_from_apply(
        lambda rc: _apply_k0_tensor_hodge_core_block(seq, ops, cs, rc, dirichlet=True),
        cs, sequential=True))
    A_cc_inv = _symmetric_pseudoinverse(A_cc)
    pk = modal_perk_bulk_data(seq, dirichlet=True)

    def atom(rb):
        return modal_perk_apply(pk, rb)

    def bulk_op(xb):
        return apply_stiffness(seq, ops, jnp.zeros((size,)).at[cs:].set(xb), 0,
                               dirichlet=True)[cs:]

    t0 = time.perf_counter()
    C0 = _assemble_dense_from_apply(
        lambda rc: _apply_k0_tensor_hodge_surgery_to_bulk_coupling(
            seq, ops, cs, rc, dirichlet=True), cs, sequential=True)
    bs = jax.jit(lambda b: solve_singular_cg(
        bulk_op, b, precond_matvec=atom, maxiter=1000, tol=1e-12)[0])
    sol = jnp.stack([bs(C0[:, i]) for i in range(cs)], axis=1)
    schur_inv = _symmetric_pseudoinverse(_symmetrize(A_cc - C0.T @ sol))
    t_schur_setup = time.perf_counter() - t0

    def M_bd(r):
        return jnp.concatenate([A_cc_inv @ r[:cs], atom(r[cs:])])

    def M_cp(r):
        y = atom(r[cs:])
        z = schur_inv @ (r[:cs] - C0.T @ y)
        return jnp.concatenate([z, y - atom(C0 @ z)])

    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal(size))
    print(f"  {gname}: Schur assembly = {t_schur_setup:.1f} s "
          f"(block-diag needs none)", flush=True)
    for nm, M in (("block-diag", M_bd), ("Schur+coupling", M_cp)):
        pcg_timed(K, b, M, maxit=3)
        it, t = pcg_timed(K, b, M)
        print(f"    {nm:15s} its={it:>4}  solve={t * 1e3:8.1f} ms  "
              f"per-it={t / max(it, 1) * 1e3:6.2f} ms", flush=True)
