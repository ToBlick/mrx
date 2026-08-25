"""Is the vector `compute_nullspaces` returned actually a harmonic form?

`compute_nullspaces` is a chain of Hodge solves with a fixed iteration budget
and NO convergence gate: a solve that runs out of iterations returns a
non-harmonic vector, every deflated solve downstream deflates against it, and
nothing in the pipeline says a word.  Everything built on top -- Poincare
sections, iota, the k=2 against k=1 cross-check -- inherits that silently.

The decisive number is the Rayleigh quotient

    lam = v^T L_k v / v^T M_k v ,

which is zero for a true harmonic form and O(lam_1) for anything else.  It is
reported against two scales that make it readable:

* ``lam_rand``, the same quotient for a random vector, i.e. what a generic
  member of the space scores.  ``lam/lam_rand`` is the dimensionless "how many
  orders of magnitude below generic", and it is the number to quote.
* the exact derivative, ``|div v|`` for the 2-form and ``|curl v|`` for the
  1-form, relative to ``|v|``.  This is the ``d v = 0`` half of "harmonic", and
  it is NOT free of solves: the construction reaches it through a Leray
  projection and a curl subtraction, so a stalled solve shows up here.  It is
  the cheaper and more localised of the two indicators -- it says *which* half
  of the harmonic condition broke, where the Rayleigh quotient only says that
  one did.

L_k is applied EXACTLY (`apply_hodge_laplacian`, nested mass solve).  That is a
Krylov-in-Krylov shape and is banned inside a solve; here it is a diagnostic
evaluated once per vector, which is the one place it is legitimate.

    python scripts/debug/harmonic_audit.py --geometry w7x --ns 12,24,12 --p 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.poincare import logical_field  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402

FIELDS = {"k2": (2, True), "k1": (1, False)}


def rayleigh(seq, v, k, dirichlet):
    lv = seq.apply_hodge_laplacian(v, k, dirichlet=dirichlet)
    mv = seq.apply_mass_matrix(v, k, dirichlet=dirichlet)
    return float(jnp.dot(v, lv) / jnp.dot(v, mv))


def exact_derivative_norm(seq, v, k, dirichlet):
    """``|d v| / |v|`` in L2, via the STRONG operator (no solve involved)."""
    if k == 2:
        dv = seq.apply_strong_div(v, dirichlet, dirichlet)
        out_k = 3
    else:
        dv = seq.apply_strong_curl(v, dirichlet, dirichlet)
        out_k = 2
    return float(seq.l2_norm(dv, out_k, dirichlet=dirichlet)
                 / seq.l2_norm(v, k, dirichlet=dirichlet))


def field_angle(seq, ops, n=512):
    f2 = logical_field(seq, ops.null_2_dbc[0], 2, True)
    f1 = logical_field(seq, ops.null_1[0], 1, False)
    x = jax.random.uniform(jax.random.PRNGKey(7), (n, 3))
    x = x.at[:, 0].multiply(0.95).at[:, 0].add(0.02)
    v2 = jax.vmap(f2)(x)
    v1 = jax.vmap(f1)(x)
    v2 = v2 / jnp.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / jnp.linalg.norm(v1, axis=1, keepdims=True)
    cos = jnp.abs(jnp.sum(v2 * v1, axis=1))
    ang = jnp.arccos(jnp.clip(cos, -1.0, 1.0))
    return float(jnp.max(ang)), float(jnp.median(ang))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    setup = time.perf_counter() - t0

    row = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
           "maxiter": cli.maxiter, "setup_s": setup, "fields": {}}

    for name, (k, dbc) in FIELDS.items():
        v = ops.null_2_dbc[0] if k == 2 else ops.null_1[0]
        n = v.shape[0]
        rand = jax.random.normal(jax.random.PRNGKey(11 * k), (n,))
        lam = rayleigh(seq, v, k, dbc)
        lam_rand = rayleigh(seq, rand, k, dbc)
        dnorm = exact_derivative_norm(seq, v, k, dbc)
        row["fields"][name] = {
            "n_dof": int(n), "rayleigh": lam, "rayleigh_random": lam_rand,
            "ratio": lam / lam_rand, "exact_derivative_rel": dnorm,
        }
        print(f"[{name}] n={n:7d}  lam={lam:12.5e}  lam_rand={lam_rand:12.5e}"
              f"  lam/lam_rand={lam / lam_rand:10.3e}  |dv|/|v|={dnorm:.3e}",
              flush=True)

    amax, amed = field_angle(seq, ops)
    row["angle_max_rad"], row["angle_median_rad"] = amax, amed
    print(f"[angle] k2 vs k1: max {amax:.4e} rad, median {amed:.4e} rad",
          flush=True)

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(row, f, indent=2)
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
