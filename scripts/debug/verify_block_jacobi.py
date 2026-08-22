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

The operator is left alone -- raw_kron is still the weak term's inner inverse --
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

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    BlockJacobiLaplacian)
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map  # noqa: E402
from mrx.nullspace import compute_nullspaces, get_nullspace  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p, maxiter, inner_tol=1e-13):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=inner_tol, maxiter=maxiter,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "cylinder":
        # Periodic cylinder F(r,chi,z) = (a r cos2pi chi, a r sin2pi chi, h z).
        # a = 0.33 keeps the minor radius comparable to the toroid's; h = 1.0.
        # ZERO angular metric variation -- the least-coupled geometry there is,
        # and so the zero-coupling end of the s_opt trend (see handoff 17.6).
        seq.set_map(cylinder_map(a=0.33, h=1.0))
    elif geometry == "rot-ellipse":
        # Same parameters as every other debug script in this directory
        # (raw_kron_mass_gate, modal_radial_gate, radial_profile_pairs).
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
        jac = np.asarray(seq.geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0:
            raise RuntimeError("W7-X geometry is degenerate")
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


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
                    choices=("toroid", "rot-ellipse", "w7x", "cylinder"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--arms", default="jacobi,blockjac_r3")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--out", default=None)
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
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} UNSHIFTED tol={cli.tol}",
          flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
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
          " ".join(f"{a:>26}" for a in arms), flush=True)
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

            def a_apply(x, k=k, dbc=dbc):
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
                        # tgN: boundary penalty on the NON-trace components,
                        # N/100 of the same alpha. Tests whether their silent
                        # homogeneous-Neumann condition is the k=1/2 free lever.
                        tg = re.search(r"tg(\d+)", arm)
                        os.environ["MRX_BJ_TANG_BC"] = (
                            tg.group(1) if tg else "0")
                        # tmN: MODE-DEPENDENT beta on the non-trace components,
                        # c = N/100. The principled version of tgN.
                        # dbN: NEGATIVE boundary term under Dirichlet, N/100
                        # of the same alpha. Tests the weight-placement
                        # mismatch that §6.3's invariant has been hiding.
                        db = re.search(r"db(\d+)", arm)
                        os.environ["MRX_BJ_DBC_BC"] = (
                            db.group(1) if db else "0")
                        # ntN: Nitsche CONSISTENCY term, N/100. Cross-component
                        # boundary coupling; k=1 free only.
                        nt = re.search(r"nt(\d+)", arm)
                        os.environ["MRX_BJ_NITSCHE"] = (
                            nt.group(1) if nt else "0")
                        tm = re.search(r"tm(\d+)", arm)
                        os.environ["MRX_BJ_TANG_MODE"] = (
                            tm.group(1) if tm else "0")
                        os.environ["MRX_BJ_BC_SCALE"] = (
                            str(int(pc.group(1)) / 100.0) if pc else
                            sc.group(1) if sc else "1.0")
                        # d0sN: multiply the p=1 degree-0 (jump) radial
                        # stiffness by N/100. Diagnostic only.
                        d0 = re.search(r"d0s(\d+)", arm)
                        os.environ["MRX_BJ_D0_SCALE"] = (
                            str(int(d0.group(1)) / 100.0) if d0 else "1.0")
                        # d0old: assemble the p=1 jump form on COEFFICIENTS
                        # (the pre-fix behaviour, under-scaled by h^2).
                        os.environ["MRX_BJ_D0_FORM"] = (
                            "coef" if "d0old" in arm else "value")
                        m = re.search(r"_r(\d+)", arm)
                        o = re.search(r"_o(\d+)", arm)
                        # pin[o|a][d]N: HARD-pin N outer rings of a component
                        # set -- their rows leave the bulk, so the atom sees
                        # the DIRICHLET-eliminated radial factor.
                        #   (none) trace components (they carry the natural
                        #          term, which is switched off when pinned)
                        #   o      the OTHER components -- where the high
                        #          outliers live; the hard limit of tg
                        #   a      all of them
                        #   d      evicted rows get the Jacobi DIAGONAL
                        #          instead of a probe column (no dense block)
                        pin = re.search(r"pin([oa]?)(d?)(\d+)", arm)
                        # fmM / frR: ADDITIVE truncated-Fourier coarse
                        # correction on the outer R rings, modes |m|,|n| <= M.
                        # R L R^T with R a RESTRICTION rather than a row
                        # selection -- one probe apply per coarse vector.
                        fm = re.search(r"fm(\d+)", arm)
                        fr = re.search(r"fr(\d+)", arm)
                        # pinN: HARD-pin the trace components' outer N rings
                        # (their rows go to the exact probe, so the atom sees
                        # the Dirichlet-eliminated radial factor). The other
                        # components' scalar Neumann conditions then ARE the
                        # operator's natural conditions, because pinning the
                        # trace kills the tangential-derivative coupling.
                        # pindN: the same pin, but the evicted rows get the
                        # operator's Jacobi DIAGONAL instead of a probe column.
                        # Separates the pin (free) from the exact boundary
                        # treatment (a dense probe, which does not scale).
                        pre = BlockJacobiLaplacian(
                            seq, ops, k, dbc,
                            ktilde_mode=("roundtrip" if "rt" in arm
                                         else "honest"),
                            lumped="diag",
                            extra_rings=int(m.group(1)) if m else 0,
                            outer_rings=int(o.group(1)) if o else 0,
                            bc_entry=("wibpd" if "wibpd" in arm else
                                      "wibp" if "wibp" in arm else
                                      "woodbury" if "wood" in arm else
                                      "wdiag" if "wdiag" in arm else
                                      "ibpf" if "ibpf" in arm else
                                      "ibpr" if "ibpr" in arm else
                                      "ibps" if "ibps" in arm else
                                      "ibpd" if "ibpd" in arm else
                                      "ibp" if "ibp" in arm else
                                      "exact" if "exact" in arm else
                                      "face" if "face" in arm else
                                      False if "nobc" in arm else "direct"),
                            radial=("modal" if "modal" in arm else "averaged"),
                            core_mode=("atom2d" if "a2d" in arm else "dense"),
                            pin_trace=int(pin.group(3)) if pin else 0,
                            pin_mode=("diag" if pin and pin.group(2)
                                      else "probe"),
                            pin_set=({"o": "other", "a": "all",
                                      "": "trace"}[pin.group(1)]
                                     if pin else "trace"),
                            coarse_rings=(int(fr.group(1)) if fr else
                                          1 if fm else 0),
                            coarse_modes=((int(fm.group(1)),) * 2 if fm
                                          else (3, 3)),
                            coarse_set=("other" if "fso" in arm else
                                        "trace" if "fst" in arm else "all"),
                            coarse_mode=("additive" if "fadd" in arm
                                         else "hybrid"),
                            # ftD: hold V and LV only on a slab D rings deep.
                            # ft9 must reproduce the untruncated arm exactly.
                            coarse_trunc=(int(re.search(r"ft(\d+)", arm).group(1))
                                          if re.search(r"ft(\d+)", arm) else 0))
                        minv = pre.apply
                    t_build = time.perf_counter() - t0
                    it, rel, ok = pcg(a_apply, b, minv, proj, tol=cli.tol,
                                      maxiter=cli.maxiter)
                    cells.append(f"{t_build:6.1f}s {it:6d}it "
                                 f"{'y' if ok else 'N'} {rel:8.1e}")
                    record[arm] = {"build_s": t_build, "iters": it,
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
