"""Why is the k=2 DBC Poisson study STILL flat after the RHS metric fix?

Fixing the load frame took ``test_torus_poisson_dbc_k2_sparse.py`` from 1.7818
to 1.6796e-01 -- a real 10.6x improvement -- but the error is STILL flat across
n = 6/8/10, so a second, independent defect remains. The k=1 counterpart is now
converging (8.564e-03 / 3.244e-03 / 1.576e-03), so whatever is left is specific
to k=2.

THE DECISIVE TEST NEEDS NO SOLVE. Project ``w2_exact`` into V2 and measure it
with the STUDY'S OWN error metric:

  * if that error is SMALL and shrinks with n, then the exact solution and the
    error metric agree with each other, the discrete space can represent the
    exact field, and the remaining fault is in the RHS or the solve;
  * if it is LARGE and FLAT, then ``w2_exact`` and the error metric disagree --
    the reference is being compared against something it is not -- and no
    amount of solver work will fix it.

A wrong-but-consistent norm cannot produce a flat relative error (any positive
weight still gives 0 as w_h -> w_ex), so a flat projection error can only mean
a CONVENTION mismatch: the k=2 DOF expansion is the contravariant density proxy
(``B_phys = DF omega / J``) while the error metric weights its argument as a
covariant antisymmetric tensor (``g^ii g^jj`` with ``sqrt g``).

Also reported, because they discriminate further:

  * best-fit scalar c minimising ||w_h - c w_ex||, and the residual AFTER
    fitting it. A small post-fit residual with c != 1 means a pure scale or
    convention factor; a large one means the SHAPE is wrong.
  * the same, slot by slot (chi-zeta, r-zeta, r-chi). The exact solution
    occupies only the r-chi slot, so leakage into the other two is diagnostic
    on its own.

    python scripts/debug/poisson_k2_reference_probe.py --ns 6,8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.operators import (  # noqa: E402
    assemble_incidence_operators,
    assemble_projection_operators,
)
from mrx.quadrature import evaluate_at_xq  # noqa: E402

PI = jnp.pi
TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)
K = 2
DIRICHLET = True


def make_w2_exact_ref(a: float):
    """omega_2 = -(2 pi eps^3 r sin(2 pi zeta)/R) dr^dchi -- verbatim from the study."""
    def w(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * PI * chi)
        w_rchi = -2.0 * PI * a**3 * r * jnp.sin(2 * PI * z) / R
        return jnp.array([0.0, 0.0, w_rchi])
    return w


def build(n, p, eps, cg_tol, cg_maxiter, qoff):
    ns, ps = (n, 2 * n, n), (p, p, p)
    seq = DeRhamSequence(ns, ps, 2 * p + qoff, TYPES, polar=True,
                         tol=cg_tol, maxiter=cg_maxiter, betti_numbers=BETTI)
    seq.set_map(toroid_map(epsilon=eps))
    seq.evaluate_1d()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    seq.set_operators(ops)
    return seq


def at_quad(seq, dofs):
    comp_info, comp_shapes = seq._form_comp_info(K)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    return evaluate_at_xq(seq.e2_dbc_T @ dofs, comp_info, comp_shapes, quad_shape, 3)


def study_metric(seq, w_h, w_ex):
    """The study's own error: weights g^ii g^jj, measure sqrt(g) = jacobian."""
    g_inv = seq.metric_inv_jkl
    weights = jnp.stack([
        g_inv[:, 1, 1] * g_inv[:, 2, 2],
        g_inv[:, 0, 0] * g_inv[:, 2, 2],
        g_inv[:, 0, 0] * g_inv[:, 1, 1],
    ], axis=1)
    meas = seq.jacobian_j * seq.quad.w
    d = w_h - w_ex
    num = jnp.einsum('qi,qi,qi,q->', d, d, weights, meas)
    den = jnp.einsum('qi,qi,qi,q->', w_ex, w_ex, weights, meas)
    return float(jnp.sqrt(num / den)), weights, meas


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
        w_ex_fn = make_w2_exact_ref(cli.eps)
        DF = jax.jacfwd(seq.map)

        # Project w_exact into V2. load(frame='phys') wants the Piola proxy
        # DF omega / J; load then forms DF^T DF omega / J = g omega / J, which
        # is what M2 inverts back to omega. (This is the same correction the
        # study's RHS needed.)
        def w_phys(x):
            dF = DF(x)
            return dF @ w_ex_fn(x) / jnp.linalg.det(dF)

        rhs = seq.load(w_phys, K, dirichlet=DIRICHLET)
        w_proj = seq.apply_inverse_mass_matrix(rhs, K, dirichlet=DIRICHLET)
        w_h = at_quad(seq, w_proj)
        w_ex = jax.vmap(w_ex_fn)(seq.quad.x)

        err, weights, meas = study_metric(seq, w_h, w_ex)

        # best-fit scalar and post-fit residual, global then per slot
        def fit(a, b, sl=None):
            if sl is None:
                aa, bb = a, b
                num = jnp.einsum('qi,qi,qi,q->', aa, bb, weights, meas)
                den = jnp.einsum('qi,qi,qi,q->', aa, aa, weights, meas)
                c = float(num / den)
                d = c * aa - bb
                r = float(jnp.sqrt(jnp.einsum('qi,qi,qi,q->', d, d, weights, meas)
                                   / jnp.einsum('qi,qi,qi,q->', bb, bb, weights, meas)))
                return c, r
            aa, bb, ww = a[:, sl], b[:, sl], weights[:, sl]
            den = float(jnp.sum(aa * aa * ww * meas))
            if den == 0.0:
                return float('nan'), float('nan')
            c = float(jnp.sum(aa * bb * ww * meas) / den)
            nb = float(jnp.sum(bb * bb * ww * meas))
            d = c * aa - bb
            r = (float(jnp.sqrt(jnp.sum(d * d * ww * meas) / nb))
                 if nb > 0 else float('nan'))
            return c, r

        c_all, r_all = fit(w_h, w_ex)
        print(f"\n[n={n}] built+projected in {time.perf_counter() - t0:.1f}s")
        print(f"  PROJECTION error, study metric : {err:.6e}")
        print(f"  best-fit scalar c              : {c_all:.6f}")
        print(f"  residual after fitting c       : {r_all:.6e}")
        print("  slot        max|w_h|      max|w_ex|       c        post-fit")
        names = ["chi-zeta", "r-zeta  ", "r-chi   "]
        slots = []
        for s in range(3):
            cs, rs = fit(w_h, w_ex, s)
            mh = float(jnp.max(jnp.abs(w_h[:, s])))
            me = float(jnp.max(jnp.abs(w_ex[:, s])))
            print(f"  {names[s]}  {mh:12.4e}  {me:12.4e}  {cs:9.4f}  {rs:10.3e}")
            slots.append({"slot": names[s].strip(), "max_h": mh, "max_ex": me,
                          "c": cs, "post_fit": rs})
        rows.append({"n": n, "projection_error": err, "c": c_all,
                     "post_fit": r_all, "slots": slots})

    print(f"\n{'n':>4} {'projection err':>16} {'c':>10} {'post-fit':>12}")
    for r in rows:
        print(f"{r['n']:>4} {r['projection_error']:>16.6e} "
              f"{r['c']:>10.4f} {r['post_fit']:>12.4e}")
    print("\nflat & large -> w2_exact and the error metric disagree (convention)")
    print("small & shrinking -> reference is fine; fault is in RHS or solve")

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {cli.out}")


if __name__ == "__main__":
    main()
