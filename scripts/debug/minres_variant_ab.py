"""CGS vs MGS in the MINRES Lanczos step, at production speed.

`mrx/solvers.py:minres` deviates from SOL (Choi/Paige/Saunders) in one place:

    SOL:   y = A*v;  y -= (beta/oldb)*r1;  alfa = v'*y;  y -= (alfa/beta)*r2
    here:  y = A*v;  alfa = v'*y;  y -= (beta/oldb)*r1;  y -= (alfa/beta)*r2

Equal in exact arithmetic (v_k' r_{k-1} = 0 by M^-1-orthogonality). In floating
point they are classical vs MODIFIED Gram-Schmidt, and CGS loses orthogonality
much faster -- which matches the symptom: fine at low iteration counts,
stagnation at high ones.

Everything here stays inside `jax.lax.while_loop`. The audit quantities are
carried as loop STATE -- a running max of |v.r1|/(|v||r1|), a count of negative
M-inner products, a running min of gamma, and a per-iteration `phibar` history
in a preallocated array. One extra reduction per iteration, no host syncs.
(The first version of this script ran the loop in Python with a `float()` per
inner product; it managed zero iterations of useful output in 12 minutes.)

    python scripts/debug/minres_variant_ab.py --geometry w7x --k 2
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


@functools.partial(jax.jit, static_argnums=(0, 1, 4, 5))
def minres_ab(A, M, b, tol, maxiter, mgs):
    """SOL MINRES, alpha-ordering switchable, audit counters in loop state."""
    x = jnp.zeros_like(b)
    r0 = b - A(x)
    y0 = M(r0)
    beta1 = jnp.sqrt(jnp.dot(r0, y0))
    bnorm = jnp.sqrt(jnp.dot(b, M(b)))

    init = dict(
        x=x, y=y0, r1=jnp.zeros_like(b), r2=r0, beta=beta1, oldbeta=0.0,
        cs=-1.0, sn=0.0, dbar=0.0, epsln=0.0, phibar=beta1,
        w_prev=jnp.zeros_like(b), w_pp=jnp.zeros_like(b), k=0,
        converged=False,
        n_neg=0, max_vr1=0.0, min_gamma=jnp.inf,
        hist=jnp.zeros(maxiter),
    )

    def cond(s):
        return jnp.logical_and(s["k"] < maxiter, ~s["converged"])

    def body(s):
        beta, k = s["beta"], s["k"]
        v = s["y"] / beta
        yn = A(v)

        nv = jnp.linalg.norm(v)
        nr = jnp.linalg.norm(s["r1"])
        vr1 = jnp.where((nv > 0) & (nr > 0),
                        jnp.abs(jnp.dot(v, s["r1"])) / (nv * nr), 0.0)

        coef = jnp.where(k >= 1, beta / jnp.where(s["oldbeta"] > 0,
                                                  s["oldbeta"], 1.0), 0.0)
        if mgs:                       # SOL: subtract, THEN form alpha
            yn = yn - coef * s["r1"]
            alpha = jnp.dot(v, yn)
        else:                         # in-tree: form alpha, THEN subtract
            alpha = jnp.dot(v, yn)
            yn = yn - coef * s["r1"]
        yn = yn - (alpha / beta) * s["r2"]

        yp = M(yn)
        ip = jnp.dot(yn, yp)
        beta_new = jnp.sqrt(jnp.abs(ip))    # audit only: count, do not hide
        oldeps = s["epsln"]
        delta = s["cs"] * s["dbar"] + s["sn"] * alpha
        gbar = s["sn"] * s["dbar"] - s["cs"] * alpha
        gamma = jnp.sqrt(gbar ** 2 + beta_new ** 2)
        cs = gbar / gamma
        sn = beta_new / gamma
        phi = cs * s["phibar"]
        phibar = sn * s["phibar"]
        w_new = (v - oldeps * s["w_pp"] - delta * s["w_prev"]) / gamma

        return dict(
            x=s["x"] + phi * w_new, y=yp, r1=s["r2"], r2=yn, beta=beta_new,
            oldbeta=beta, cs=cs, sn=sn, dbar=-cs * beta_new,
            epsln=sn * beta_new, phibar=phibar,
            w_prev=w_new, w_pp=s["w_prev"], k=k + 1,
            converged=phibar < tol * bnorm,
            n_neg=s["n_neg"] + jnp.where(ip < 0, 1, 0),
            max_vr1=jnp.maximum(s["max_vr1"], vr1),
            min_gamma=jnp.minimum(s["min_gamma"], gamma),
            hist=s["hist"].at[k].set(phibar / bnorm),
        )

    out = jax.lax.while_loop(cond, body, init)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=10000)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    for k in [int(v) for v in cli.ks.split(",")]:
        for dbc in (False, True):
            op.assemble_block_jacobi_laplacian_preconditioner(
                seq, ops, ks=(k,), dirichlets=(dbc,))
            suffix = "_dbc" if dbc else ""
            n_u = int(getattr(seq, f"n{k}{suffix}"))
            n_l = int(getattr(seq, f"n{k-1}{suffix}"))

            def A(z, k=k, dbc=dbc, n_u=n_u):
                u, s = z[:n_u], z[n_u:]
                return jnp.concatenate([
                    op.apply_stiffness(seq, ops, u, k, dirichlet=dbc)
                    + op.apply_derivative_matrix(seq, ops, s, k - 1,
                                                 dirichlet_in=dbc,
                                                 dirichlet_out=dbc),
                    op.apply_derivative_matrix(seq, ops, u, k - 1,
                                               dirichlet_in=dbc,
                                               dirichlet_out=dbc,
                                               transpose=True)
                    - op.apply_mass_matrix(seq, ops, s, k - 1, dirichlet=dbc)])

            def M(z, k=k, dbc=dbc, n_u=n_u):
                u, s = z[:n_u], z[n_u:]
                return jnp.concatenate([
                    op.apply_hodge_laplacian_preconditioner(
                        seq, ops, u, k, dirichlet=dbc, kind='block'),
                    op.apply_mass_matrix_preconditioner(
                        seq, ops, s, k - 1, dirichlet=dbc, kind='auto')])

            b = jnp.concatenate([
                jax.random.normal(jax.random.PRNGKey(31 * k + dbc), (n_u,)),
                jnp.zeros(n_l)])

            # WARM THE LAZY CACHES OUTSIDE THE TRACE. Both the mass factors and
            # BlockJacobiLaplacian._build_apply memoise on first use; if that
            # first use happens inside jit/while_loop the cached object closes
            # over tracers and leaks (UnexpectedTracerError from
            # block_jacobi_laplacian.py:_build_apply). Same rule as
            # warm_mass_preconditioner_cache, which the docs state for exactly
            # this reason -- it just also applies to the Laplacian atom.
            op.warm_mass_preconditioner_cache(
                seq, ops, ks=(k - 1, k), dirichlets=(dbc,))
            _ = A(b).block_until_ready()
            _ = M(b).block_until_ready()

            print(f"\n=== k={k} dbc={dbc} n={n_u}+{n_l} ===", flush=True)
            for mgs in (False, True):
                t0 = time.perf_counter()
                s = minres_ab(A, M, b, cli.tol, cli.maxiter, mgs)
                it = int(s["k"])
                hist = np.asarray(s["hist"])[:it]
                marks = [hist[min(j, it - 1)] for j in
                         (249, 999, 2999, it - 1)] if it else []
                print(f"  {'MGS (SOL)' if mgs else 'CGS (in-tree)':<14} "
                      f"{'CONVERGED' if bool(s['converged']) else 'STALLED':<10} "
                      f"it={it:<6} neg_ip={int(s['n_neg']):<5} "
                      f"max|v.r1|={float(s['max_vr1']):.2e} "
                      f"min_gamma={float(s['min_gamma']):.2e} "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
                print("      phibar/|b| @ it 250/1000/3000/end: "
                      + " ".join(f"{m:.2e}" for m in marks), flush=True)


if __name__ == "__main__":
    main()
