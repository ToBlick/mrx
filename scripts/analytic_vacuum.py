"""Convergence of the discrete vacuum field against the analytic 1/R toroidal
field ``B* = e_phi / R`` on a genuinely 3-D (non-axisymmetric) domain.

``B* = grad(phi_geo)`` (phi_geo the geometric azimuth) is curl-free and
div-free everywhere ``R > 0``, hence an *exact* vacuum field on ANY domain --
the QA stellarator included. It is NOT tangent to a non-axisymmetric wall, so it
is a driven (prescribed-normal-flux) vacuum field, not the domain's confined
harmonic form. Because the exact field is closed-form there is no truncation
floor: the error should fall at ``O(h^p)``.

Two constructions of the scalar magnetic potential, at the two ends of the de
Rham complex, are compared. Both reuse the existing preconditioned solvers (no
mixed saddle, no new preconditioner):

  Route A (0-form potential):  H = grad f + a h1,   f in V0,  H in V1 (free).
      G0^T M1 G0 f = G0^T load1(B*)          -- k=0 stiffness, deflated CG.
  Route C (3-form potential):  B = delta f + a h2, f in V3,  B in V2 (free).
      L3 f = -D2 b2*                          -- k=3 Hodge-Laplacian, MINRES;
      delta = weak grad = -M2^{-1} D2^T (a method), B = delta f.

Both fields are curl-free by construction (curl grad = 0 / delta delta = 0);
div-free is the weak Euler-Lagrange condition of the solve. The harmonic
amplitude ``a`` is the free flux parameter, fixed by the M-projection
``a = <B*, h>_M / <h, h>_M`` -- the same c-fit as the VMEC vacuum study, needing
no analytic cross-section.

Error is measured against the ANALYTIC field:
  ||B_h - B*||_M^2 = <B_h, B_h>_M - 2 (B_h . load) + int |B*|^2 dV,
with ``int |B*|^2 dV = sum_q w_q det(DF)_q / R_q^2`` by volume quadrature, and
``B_h . load = <B_h, B*>`` exact since ``load_i = int Lambda_i . B*``.
"""

import argparse
import json
import os
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc",
                    help="equilibrium file or analytic name; the domain only "
                         "(its field is ignored, B* = e_phi/R is analytic).")
    ap.add_argument("--ns", default="6,12,6:8,16,8:10,20,10",
                    help="colon-separated n_r,n_theta,n_zeta rungs (colon is "
                         "shell-safe inside slurm/run.sh's wrapped command).")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--routes", default="A,C")
    ap.add_argument("--out", default="outputs/analytic_vacuum")
    return ap.parse_args(argv)


def _log(msg):
    print(msg, flush=True)


def analytic_norm_sq(seq):
    """``int |B*|^2 dV = sum_q w_q det(DF)_q / R_q^2`` at the quadrature points,
    with ``R^2 = X^2 + Y^2`` from the lab-frame map. This is the M-norm of B*
    in both V1 and V2 (physical L2 of the same vector field)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    Xq = np.asarray(jax.vmap(seq.map)(seq.quad.x))          # (Nq, 3) lab frame
    R2 = Xq[:, 0] ** 2 + Xq[:, 1] ** 2
    w = np.asarray(seq.quad.w)
    J = np.asarray(seq.jacobian_j)
    return float(np.sum(w * J / R2)), jnp


def b_star_phys(seq):
    """``B*(xi) = (-Y, X, 0) / R^2`` evaluated at the lab-frame image of the
    logical point ``xi`` (g = 1). Curl-free, div-free, |B*| = 1/R."""
    import jax.numpy as jnp

    def f(xi):
        X = seq.map(xi)
        R2 = X[0] ** 2 + X[1] ** 2
        return jnp.array([-X[1], X[0], 0.0]) / R2
    return f


def run_rung(seq, ops, routes, tag):
    import numpy as np
    from mrx.nullspace import estimate_spectral_gap, harmonic_rayleigh

    res = dict(tag=tag)
    Bphys = b_star_phys(seq)
    bstar_sq, _ = analytic_norm_sq(seq)
    res["bstar_norm"] = float(np.sqrt(bstar_sq))

    out = {}

    if "A" in routes:
        # --- Route A: 0-form scalar potential, H in V1 (free) ----------------
        t0 = time.perf_counter()
        load1 = seq.load(Bphys, 1, dirichlet=False)                 # int L1 . B*
        rhs = seq.apply_incidence_matrix(load1, 0, dirichlet_in=False,
                                         dirichlet_out=False, transpose=True)  # G0^T load1
        f = seq.apply_inverse_laplacian(rhs, 0, dirichlet=False, operators=ops)
        # solve residual on S0 = G0^T M1 G0
        r = seq.apply_stiffness(f, 0, dirichlet=False) - rhs
        solve_res = float(np.linalg.norm(r) / (np.linalg.norm(rhs) + 1e-300))
        Hgrad = seq.apply_strong_grad(f, dirichlet_in=False, dirichlet_out=False)
        h1 = seq.nullspace(1, False)[0]
        Mh1 = seq.apply_mass_matrix(h1, 1, False)
        a1 = float(load1 @ h1) / float(h1 @ Mh1)                    # <B*,h1>/<h1,h1>
        Hh = Hgrad + a1 * h1
        MHh = seq.apply_mass_matrix(Hh, 1, False)
        err_sq = float(Hh @ MHh) - 2.0 * float(Hh @ load1) + bstar_sq
        relerr = float(np.sqrt(max(err_sq, 0.0)) / np.sqrt(bstar_sq))
        rq = harmonic_rayleigh(seq, h1, 1, False, ops)
        lam1, _ = estimate_spectral_gap(seq, ops, 1, False, maxiter=5)
        out["A"] = dict(relerr=relerr, alpha=a1, solve_res=solve_res,
                        harm_ratio=float(rq / lam1), n=int(seq.n(1, False)),
                        t=time.perf_counter() - t0)
        _log(f"{tag} A: relerr {relerr:.4e}  alpha {a1:+.4e}  solve_res {solve_res:.1e}  "
             f"harm {rq / lam1:.1e}  n1 {seq.n(1, False)}  {out['A']['t']:.1f}s")

    if "C" in routes:
        # --- Route C: 3-form scalar potential, B = delta f in V2 (free) ------
        t0 = time.perf_counter()
        load2 = seq.load(Bphys, 2, dirichlet=False)                 # int L2 . B*
        b2 = seq.apply_inverse_mass_matrix(load2, 2, dirichlet=False)  # L2-proj of B*
        rhs = -seq.apply_derivative_matrix(b2, 2, dirichlet_in=False,
                                           dirichlet_out=True)      # -D2 b2*
        f3 = seq.apply_inverse_laplacian(rhs, 3, dirichlet=True, operators=ops)
        r = seq.apply_laplacian(f3, 3, dirichlet=True, operators=ops) - rhs
        solve_res = float(np.linalg.norm(r) / (np.linalg.norm(rhs) + 1e-300))
        Bgrad = seq.apply_weak_grad(f3, dirichlet_in=True, dirichlet_out=False)  # delta f
        h2 = seq.nullspace(2, True)[0]
        Mh2 = seq.apply_mass_matrix(h2, 2, True)
        # h2 is dbc (n2_dbc); embed the delta-f free-V2 comparison via the loads.
        a2 = float(load2 @ h2) / float(h2 @ Mh2)                    # <B*,h2>/<h2,h2>
        Bh = Bgrad + a2 * h2
        MBh = seq.apply_mass_matrix(Bh, 2, False)
        err_sq = float(Bh @ MBh) - 2.0 * float(Bh @ load2) + bstar_sq
        relerr = float(np.sqrt(max(err_sq, 0.0)) / np.sqrt(bstar_sq))
        rq = harmonic_rayleigh(seq, h2, 2, True, ops)
        lam1, _ = estimate_spectral_gap(seq, ops, 2, True, maxiter=5)
        out["C"] = dict(relerr=relerr, alpha=a2, solve_res=solve_res,
                        harm_ratio=float(rq / lam1), n=int(seq.n(2, False)),
                        t=time.perf_counter() - t0)
        _log(f"{tag} C: relerr {relerr:.4e}  alpha {a2:+.4e}  solve_res {solve_res:.1e}  "
             f"harm {rq / lam1:.1e}  n2 {seq.n(2, False)}  {out['C']['t']:.1f}s")

    res["routes"] = out
    return res


def main(cli):
    import numpy as np

    import mrx
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    os.makedirs(cli.out, exist_ok=True)
    routes = [r.strip() for r in cli.routes.split(",") if r.strip()]
    rungs = [tuple(int(v) for v in chunk.split(",")) for chunk in cli.ns.split(":")]
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)

    records = []
    for ns in rungs:
        tag = f"{ns[0]}x{ns[1]}x{ns[2]}_p{cli.p}"
        t0 = time.perf_counter()
        seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter, tol=cli.tol)
        ops = seq.set_operators(compute_nullspaces(seq, ops))
        rec = run_rung(seq, ops, routes, tag)
        rec.update(ns=list(ns), p=cli.p, h=1.0 / (ns[0] - cli.p),
                   n_elements=ns[0] - cli.p, tol=float(seq.tol),
                   t_total=time.perf_counter() - t0)
        records.append(rec)
        with open(os.path.join(cli.out, "analytic_vacuum.json"), "w") as fh:
            json.dump(dict(geometry=os.path.abspath(cli.geometry), p=cli.p,
                           records=records), fh, indent=2)

    # slopes: log(relerr) vs log(h), consecutive-rung rate
    print("\n=== convergence (relerr vs h = 1/n_el) ===", flush=True)
    for r in routes:
        hs = np.array([rec["h"] for rec in records])
        es = np.array([rec["routes"].get(r, {}).get("relerr", np.nan) for rec in records])
        print(f"route {r}:", flush=True)
        for i, rec in enumerate(records):
            rate = ""
            if i > 0 and np.isfinite(es[i]) and np.isfinite(es[i - 1]):
                rate = f"  rate {np.log(es[i] / es[i - 1]) / np.log(hs[i] / hs[i - 1]):+.2f}"
            print(f"  {rec['tag']:>14}  h {hs[i]:.4f}  relerr {es[i]:.4e}{rate}", flush=True)
    print(f"\nwrote {os.path.join(cli.out, 'analytic_vacuum.json')}", flush=True)


if __name__ == "__main__":
    main(parse_args())
