"""Full 8-case verification with the block-Jacobi Laplacian preconditioner.

Same shape as ``verify_default_preconditioners.py`` -- nullspaces by the DIRECT
route, then UNSHIFTED (``eps = 0``) Poisson solves for all four degrees and both
boundary conditions -- but comparing the production Jacobi diagonal against the
block-Jacobi atom (separable bulk + densely-probed core).

The four singular cases (betti ``(1,1,0,0)`` puts harmonic forms at
``(0,free)``, ``(1,free)``, ``(2,dbc)``, ``(3,dbc)``) are handled by DEFLATION:
the right-hand side is projected onto ``range(L) = null(L)^perp`` and the
residual and preconditioned residual are re-projected every iteration, since
round-off otherwise feeds the kernel back in.

The operator is left alone -- the weak term's inner inverse is unchanged --
so these numbers are directly comparable with the earlier run.

Usage:
    python scripts/debug/verify_block_jacobi.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import jax


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.geometries import build_sequence  # noqa: E402, F401  (re-exported: 22 debug scripts import it from here)
from mrx.experimental.metric_lumping_coarse import (  # noqa: E402
    CoarseCorrectedMetricLumping,
)
from mrx.metric_lumping_laplacian import (  # noqa: E402
    MetricLumpingLaplacian)
from mrx.nullspace import compute_nullspaces, get_nullspace  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def make_projector(vecs):
    """Euclidean projector onto ``span(vecs)^perp`` (``L`` symmetric, so that
    is ``range(L)``). Identity when there is no kernel."""
    if vecs is None or vecs.shape[0] == 0:
        return lambda v: v
    v = np.asarray(vecs)
    gram = np.linalg.inv(v @ v.T)

    def proj(x):
        a = np.asarray(x)
        return jnp.asarray(a - (gram @ (v @ a)) @ v)
    return proj


def pcg(a_apply, b, minv, proj, tol=1e-10, maxiter=20000):
    """CG with deflation: the residual is kept in the range every iteration."""
    x = jnp.zeros_like(b)
    r = proj(b)
    z = proj(minv(r))
    p = z
    rz = float(r @ z)
    nb = float(jnp.linalg.norm(b))
    for it in range(1, maxiter + 1):
        ap = proj(a_apply(p))
        pap = float(p @ ap)
        if pap <= 0.0:
            return it, float(jnp.linalg.norm(r)) / nb, False
        alpha = rz / pap
        x = x + alpha * p
        r = r - alpha * ap
        if float(jnp.linalg.norm(r)) <= tol * nb:
            return it, float(jnp.linalg.norm(r)) / nb, True
        z = proj(minv(r))
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter, float(jnp.linalg.norm(r)) / nb, False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid",
                    choices=("toroid", "rot-ellipse", "w7x", "cylinder",
                             "quasr9983", "quasr44970", "w7x-gvec", "hegna"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--arms", default="jacobi,blockjac_r3")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=10000)
    # DECOUPLED from --maxiter on purpose: --maxiter also sets the SEQUENCE's
    # inner solver budget (build_sequence passes it through). This caps the
    # OUTER CG only, so a stalling arm cannot eat a whole sweep job.
    #
    # 2026-08-24: was 20000, with inner_tol 1e-13, and a comment claiming "the
    # W7-X free L_2 solves are known to need all of it". They do not need it --
    # they were STALLING, and the budget was hiding it. The k=1 free harmonic
    # form is `v - weak_curl(L_2^-1 D_1 v)` and inherits that solve's residual
    # 1:1 (measured: relL2 4.9e-08 / 5.7e-10 / 5.6e-12 at inner tol 1e-08 /
    # 1e-10 / 1e-12 on W7-X p=2), which is why the recorded forms degrade with
    # p -- 8.4e-13 at p=2, 3.0e-04 at p=3, 1.7e-01 at p=5. Both numbers now
    # match the DeRhamSequence defaults (tol 1e-12, maxiter 10_000); a solve
    # that cannot finish inside those is a preconditioner problem to fix, not a
    # budget to raise. See docs/research/handoff_2026-08-24_harmonic_k1_free.md.
    ap.add_argument("--cg-maxiter", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--operator", default="approx", choices=("approx", "exact"),
                    help="which L_k the CG iterates on; see a_apply. 'approx' "
                         "is what all recorded sweeps used -- keep it as the "
                         "default so those stay comparable.")
    ap.add_argument("--inner-mass-tol", type=float, default=None,
                    help="--operator exact only: tolerance of the nested mass "
                         "solve (default: the sequence's own tol)")
    # Restrict which (k, dbc) rows are solved. The nullspaces are still built
    # for all k (they are shared), but the CG sweeps are the expensive part.
    ap.add_argument("--ks", default="0,1,2,3",
                    help="comma-separated k values to solve")
    ap.add_argument("--bcs", default="free,dbc",
                    help="comma-separated subset of free,dbc")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    arms = cli.arms.split(",")
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} UNSHIFTED tol={cli.tol} "
          f"operator={cli.operator}"
          + (f" inner_mass_tol={cli.inner_mass_tol}"
             if cli.operator == "exact" else ""), flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "operator": cli.operator,
               "inner_mass_tol": cli.inner_mass_tol,
               "eps": 0.0, "tol": cli.tol, "nullspaces": [], "rows": []}

    t0 = time.perf_counter()
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"\ncompute_nullspaces (direct) {time.perf_counter() - t0:.1f}s",
          flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n_null':>7} {'rayleigh q':>13} {'rel L2 err':>12}", flush=True)
    kernels = {}
    for k in range(4):
        for dbc in (False, True):
            vecs = np.asarray(get_nullspace(ops, k, dbc))
            kernels[(k, dbc)] = vecs if vecs.shape[0] else None
            worst = 0.0
            for v in vecs:
                # RAYLEIGH QUOTIENT, not ||Lv||/||v||. Lv is a DUAL vector, so
                # pairing it with v is the coherent contraction, and dividing by
                # the L2 norm makes the result an eigenvalue -- comparable to the
                # spectrum. lambda_1 is O(1) independent of h, so this is
                # resolution independent, where ||Lv||/||v|| carries ||L|| ~ h^-2
                # and degrades 30-1300x from 8^3 to 12^3 for the same vector.
                # sqrt(lambda) is then the relative L2 error in the form.
                lv = op.apply_hodge_laplacian(seq, ops, jnp.asarray(v), k,
                                              dirichlet=dbc)
                mv = op.apply_mass_matrix(seq, ops, jnp.asarray(v), k,
                                          dirichlet=dbc)
                worst = max(worst, abs(float(v @ lv)) / float(v @ mv))
            print(f"{k:>2} {dbc!s:>5} {vecs.shape[0]:>7} "
                  f"{worst if vecs.shape[0] else float('nan'):>13.3e} "
                  f"{worst ** 0.5 if vecs.shape[0] else float('nan'):>12.3e}",
                  flush=True)
            results["nullspaces"].append(
                {"k": k, "dbc": dbc, "n": int(vecs.shape[0]),
                 "rayleigh": worst, "rel_l2_err": worst ** 0.5})

    print("\nUnshifted solves (deflated where singular)", flush=True)
    print(f"{'k':>2} {'dbc':>5} {'n':>7} {'sing':>5} " +
          " ".join(f"{a:>34}" for a in arms), flush=True)
    want_k = {int(v) for v in cli.ks.split(",")}
    want_bc = set(cli.bcs.split(","))
    for k in range(4):
        for dbc in (False, True):
            if k not in want_k or ("dbc" if dbc else "free") not in want_bc:
                continue
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            vecs = kernels[(k, dbc)]
            proj = make_projector(vecs)
            b = proj(jax.random.normal(jax.random.PRNGKey(31 * k + dbc), (n,)))
            record = {"k": k, "dbc": dbc, "n": n,
                      "singular": vecs is not None}
            cells = []

            # WHICH OPERATOR IS BEING SOLVED. This was a silent choice until
            # 2026-08-24; it is now a flag, because the two are NOT the same
            # operator at k>=1:
            #
            #   approx  S_k + D B D^T,      B = one mass-preconditioner apply
            #   exact   S_k + D M^-1 D^T,   M^-1 by a nested mass CG
            #
            # `approx` is what every bc-alpha sweep number was measured on. It
            # is a legitimate SPD operator and is cheap, but the library's own
            # k>=1 Laplacian solve (`apply_inverse_hodge_laplacian` -> saddle
            # MINRES) solves the EXACT one, so absolute iteration counts here
            # do not describe that path. The natural-BC term being tuned is
            # added to the radial STIFFNESS, i.e. to S_k, which is identical in
            # both -- the reason to expect the a0/a5 ranking to transfer even
            # though the counts do not. `exact` is ~30x more expensive per
            # apply (it is Krylov-in-Krylov) and exists to TEST that transfer,
            # not to sweep with.
            def a_apply(x, k=k, dbc=dbc):
                if cli.operator == "exact":
                    return op.apply_hodge_laplacian(
                        seq, ops, x, k, dirichlet=dbc,
                        tol=cli.inner_mass_tol, maxiter=cli.maxiter)
                return op.apply_hodge_laplacian_approx(seq, ops, x, k,
                                                       dirichlet=dbc)

            for arm in arms:
                t0 = time.perf_counter()
                try:
                    if arm == "jacobi":
                        d = jnp.asarray(op._hodge_diaginv(seq, ops, k, dbc))

                        def minv(v, d=d):
                            return d * v
                    else:
                        # bcsN: multiply the natural-BC penalty by N. N -> inf
                        # is the hard u.n = 0 limit that penalty approximates.
                        # bcsN: integer multiplier. bcpN: N/100, for the
                        # sub-unity range the cross term lives in.
                        sc = re.search(r"bcs(\d+)", arm)
                        pc = re.search(r"bcp(\d+)", arm)
                        os.environ["MRX_BJ_BC_SCALE"] = (
                            str(int(pc.group(1)) / 100.0) if pc else
                            sc.group(1) if sc else "1.0")
                        # bcmN: N/1000, for the arms whose alpha is a
                        # different SIZE (the penalty convention drops the
                        # c(p) amplification, so its optimal s is ~c(p)x
                        # larger and the bcp grid cannot reach it).
                        mc = re.search(r"bcm(\d+)", arm)
                        if mc:
                            os.environ["MRX_BJ_BC_SCALE"] = str(
                                int(mc.group(1)) / 1000.0)
                        m = re.search(r"_r(\d+)", arm)
                        o = re.search(r"_o(\d+)", arm)
                        # fmM / frR: ADDITIVE truncated-Fourier coarse
                        # correction on the outer R rings, modes |m|,|n| <= M.
                        # R L R^T with R a RESTRICTION rather than a row
                        # selection -- one probe apply per coarse vector.
                        fm = re.search(r"fm(\d+)", arm)
                        fr = re.search(r"fr(\d+)", arm)
                        kwargs = dict(
                            ktilde_mode=("roundtrip" if "rt" in arm
                                         else "honest"),
                            lumped="diag",
                            extra_rings=int(m.group(1)) if m else 0,
                            outer_rings=int(o.group(1)) if o else 0,
                            # Ten other spellings of the boundary term were
                            # measured and lost -- the exact 2-D face shape,
                            # both cross-term corrections (one INDEFINITE) and
                            # the "exact" sqrt(g^rr) form, which is worse than
                            # NO term at k=1/2 free. See handoff §9, §12.3,
                            # §14.3. Only the derived term and "off" remain.
                            bc_entry=(False if "nobc" in arm else "ibpd"))
                        if fm or fr:
                            # `fm` is EXPERIMENTAL and opt-in: it lives in
                            # metric_lumping_coarse, NOT on the production
                            # class. ftD holds V and LV only on a slab D
                            # rings deep; ft9 must reproduce the untruncated
                            # arm exactly.
                            ft = re.search(r"ft(\d+)", arm)
                            pre = CoarseCorrectedMetricLumping(
                                seq, ops, k, dbc,
                                coarse_rings=(int(fr.group(1)) if fr else 1),
                                coarse_modes=((int(fm.group(1)),) * 2 if fm
                                              else (3, 3)),
                                coarse_set=("other" if "fso" in arm else
                                            "trace" if "fst" in arm
                                            else "all"),
                                coarse_mode=("additive" if "fadd" in arm
                                             else "hybrid"),
                                coarse_trunc=int(ft.group(1)) if ft else 0,
                                **kwargs)
                        else:
                            pre = MetricLumpingLaplacian(seq, ops, k, dbc,
                                                       **kwargs)
                        minv = pre.apply
                    t_build = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    it, rel, ok = pcg(a_apply, b, minv, proj, tol=cli.tol,
                                      maxiter=cli.cg_maxiter or cli.maxiter)
                    t_solve = time.perf_counter() - t1
                    cells.append(f"{t_build:6.1f}s {it:6d}it {t_solve:6.1f}s "
                                 f"{'y' if ok else 'N'} {rel:8.1e}")
                    record[arm] = {"build_s": t_build, "iters": it,
                                   "solve_s": t_solve,
                                   "total_s": t_build + t_solve,
                                   "rel": rel, "converged": ok}
                except Exception as exc:  # noqa: BLE001
                    cells.append(f"{type(exc).__name__}: {str(exc)[:30]}")
                    record[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{k:>2} {dbc!s:>5} {n:>7} "
                  f"{'yes' if vecs is not None else 'no':>5} " +
                  " ".join(cells), flush=True)
            results["rows"].append(record)
            if cli.out:
                os.makedirs(os.path.dirname(os.path.abspath(cli.out)),
                            exist_ok=True)
                with open(cli.out, "w") as fh:
                    json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
