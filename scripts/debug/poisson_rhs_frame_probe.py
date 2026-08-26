"""Is the k=1 NBC Poisson study's RHS off by one factor of the metric?

``test_torus_poisson_nbc_k1_sparse.py`` returns a relative L2 error that is FLAT
in n -- 3.7256e+01 / 3.7259e+01 / 3.7260e+01 at n = 6/8/10 -- with MINRES
reporting converged=True and a clean harmonic form.  ``test_torus_poisson_
dbc_k2_sparse.py`` does the same at a different constant (1.7818 flat).  Those
two share nothing but the analytic problem (``f0 = cos(2 pi zeta)/R^2``, with
``f1 = d f0`` and ``f2 = *(d f0)``, exact solutions Hodge dual), so a defect
common to both is in what they share.

Three of the four candidates are already eliminated:

* the manufactured pair is CORRECT.  ``omega_1 = -2 pi sin(2 pi zeta) dzeta`` is
  closed, so ``L_1 omega_1 = d(delta omega_1)``, and on the toroid metric
  ``delta omega_1 = cos(2 pi zeta)/R^2 = f0`` exactly, hence
  ``L_1 omega_1 = d f0 = f1``.
* the error metric is self-consistent: computed and exact are pushed forward by
  the same ``DF G^-1``, which is the correct 1-form pushforward
  (``DF G^-1 = DF^-T``).
* the discrete space really is covariant -- ``M_1 = int Lambda^T G^-1 Lambda J``,
  so ``M_1^-1 load`` returns covariant DOFs, matching ``Pushforward`` k=1.
* it is not the boundary conditions: k=1 NBC and k=2 DBC fail alike.

That leaves the RHS.  ``load(k=1, frame='ref')`` pairs its argument DIRECTLY
against the basis with weight ``w * J``, while ``M_1`` carries a ``G^-1``.  To
recover a primal covariant ``omega`` the load must therefore be handed
``G^-1 omega_cov``, NOT ``omega_cov``.  The study passes bare covariant
components.  Its ``frame='phys'`` variant is ``DF @ f1_ref`` where the correct
physical proxy is ``DF^-T f1_ref``, so ``load`` computes
``DF^-1 (DF f1_ref) = f1_ref`` and delivers the SAME wrong object -- which is
why both frames agree with each other and both are wrong.

A metric factor on the RHS does not vanish under refinement, which is exactly a
flat relative error with a converged solver.

This probe solves the SAME problem three ways and reports the same error metric:

    bare   load(f1_ref, frame='ref')            what the study does now
    ginv   load(G^-1 f1_ref, frame='ref')       the proposed correction
    phys   load(DF^-T f1_ref, frame='phys')     the same correction, expressed
                                                through the physical proxy

If the hypothesis holds, ``bare`` reproduces ~37.26 flat while ``ginv`` and
``phys`` agree with each other and converge at order p+1.

    python scripts/debug/poisson_rhs_frame_probe.py --ns 6,8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.nullspace import compute_nullspaces_iterative  # noqa: E402
from mrx.operators import (  # noqa: E402
    assemble_incidence_operators,
    assemble_schur_jacobi_preconditioner,
)
from mrx.quadrature import evaluate_at_xq  # noqa: E402

PI = jnp.pi
TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)
K = 1
DIRICHLET = False


def make_f1_ref(a: float):
    """f1 = d f0 in reference COVARIANT components (verbatim from the study)."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * PI * chi)
        fr = -2.0 * a * jnp.cos(2 * PI * chi) * jnp.cos(2 * PI * z) / R**3
        fchi = 4.0 * PI * a * r * jnp.sin(2 * PI * chi) * jnp.cos(2 * PI * z) / R**3
        fzeta = -2.0 * PI * jnp.sin(2 * PI * z) / R**2
        return jnp.array([fr, fchi, fzeta])
    return f


def v1_exact_ref(x):
    """omega_1 = -2 pi sin(2 pi zeta) dzeta, reference covariant components."""
    _r, _chi, z = x
    return jnp.array([0.0, 0.0, -2.0 * PI * jnp.sin(2 * PI * z)])


def build(n: int, p: int, eps: float, cg_tol: float, cg_maxiter: int, qoff: int):
    ns, ps = (n, 2 * n, n), (p, p, p)
    q = 2 * p + qoff
    seq = DeRhamSequence(ns, ps, q, TYPES, polar=True,
                         tol=cg_tol, maxiter=cg_maxiter, betti_numbers=BETTI)
    seq.set_map(toroid_map(epsilon=eps))
    seq.evaluate_1d()
    ops = assemble_incidence_operators(seq)
    ops = assemble_schur_jacobi_preconditioner(
        seq, ops, ks=(1,), dirichlet_variants=(False,))
    seq.set_operators(ops)
    ops, _ = compute_nullspaces_iterative(seq, seq.get_operators(), BETTI)
    seq.set_operators(ops)
    return seq


def relative_error(seq, u_hat):
    """The study's own metric: physical L2 via the DF G^-1 pushforward."""
    comp_info, comp_shapes = seq._form_comp_info(K)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    v_h_log = evaluate_at_xq(seq.e1_T @ u_hat, comp_info, comp_shapes, quad_shape, 3)
    DF_xq = jax.vmap(jax.jacfwd(seq.map))(seq.quad.x)
    push = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl, v_h_log)
    ex_ref = jax.vmap(v1_exact_ref)(seq.quad.x)
    ex_push = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl, ex_ref)
    d = push - ex_push
    num = jnp.einsum('qi,qi,q,q->', d, d, seq.jacobian_j, seq.quad.w)
    den = jnp.einsum('qi,qi,q,q->', ex_push, ex_push, seq.jacobian_j, seq.quad.w)
    return float(jnp.sqrt(num / den))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default="6,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--eps", type=float, default=1 / 3)
    ap.add_argument("--cg-tol", type=float, default=1e-9)
    ap.add_argument("--cg-maxiter", type=int, default=50000)
    ap.add_argument("--quad-order-offset", type=int, default=0)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    rows = []
    for n in [int(v) for v in cli.ns.split(",")]:
        t0 = time.perf_counter()
        seq = build(n, cli.p, cli.eps, cli.cg_tol, cli.cg_maxiter,
                    cli.quad_order_offset)
        f1_ref = make_f1_ref(cli.eps)
        DF = jax.jacfwd(seq.map)

        def f_ginv(x):
            g_inv = jnp.linalg.inv(DF(x).T @ DF(x))
            return g_inv @ f1_ref(x)

        def f_phys(x):
            # correct physical proxy of a covariant 1-form: DF^-T omega
            return jnp.linalg.solve(DF(x).T, f1_ref(x))

        print(f"\n[n={n}] setup {time.perf_counter() - t0:.1f}s", flush=True)
        row = {"n": n, "p": cli.p}
        for tag, fn, frame in (("bare", f1_ref, 'ref'),
                               ("ginv", f_ginv, 'ref'),
                               ("phys", f_phys, 'phys')):
            t1 = time.perf_counter()
            rhs = seq.load(fn, K, dirichlet=DIRICHLET, frame=frame)
            u_hat, info = seq.apply_inverse_laplacian(
                rhs, K, dirichlet=DIRICHLET, return_info=True)
            jax.block_until_ready(u_hat)
            err = relative_error(seq, u_hat)
            code = int(info)
            row[tag] = {"error": err, "iters": abs(code),
                        "converged": code < 0,
                        "seconds": time.perf_counter() - t1}
            print(f"  {tag:5s} frame={frame:4s}  err={err:.6e}  "
                  f"iters={abs(code):6d}  converged={code < 0}  "
                  f"({time.perf_counter() - t1:.1f}s)", flush=True)
        rows.append(row)

    print(f"\n{'n':>4} {'bare (study)':>16} {'ginv (fix)':>16} {'phys (fix)':>16}")
    for r in rows:
        print(f"{r['n']:>4} {r['bare']['error']:>16.6e} "
              f"{r['ginv']['error']:>16.6e} {r['phys']['error']:>16.6e}")
    if len(rows) > 1:
        a, b = rows[0], rows[-1]
        rat = np.log(b['n'] / a['n'])
        for tag in ("bare", "ginv", "phys"):
            o = np.log(a[tag]['error'] / b[tag]['error']) / rat
            print(f"  observed order, {tag}: {o:+.2f}   (expect ~{cli.p + 1} if correct)")

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {cli.out}")


if __name__ == "__main__":
    main()
