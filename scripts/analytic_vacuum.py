"""Convergence of the discrete vacuum field against an analytic vacuum field on
a genuinely 3-D (non-axisymmetric) domain.

A vacuum field is ``B = grad Psi`` with ``Psi`` harmonic (``div grad Psi = 0``):
curl-free and div-free. Two closed-form references:

* ``--field toroidal``: ``B* = e_phi / R = grad(phi_geo)``. DEGENERATE test --
  in flux coordinates ``phi_geo = 2 pi zeta``, so ``B*`` is a *constant*
  covariant 1-form ``(0,0,2 pi)`` in logical coordinates, exactly in every
  discrete V1 space at any resolution (it IS the discrete harmonic form). The
  error sits at round-off; useful only as a representation-exactness check.
* ``--field polynomial`` (default): ``B* = grad Psi``, ``Psi = (x^3 - 3 x y^2)
  + z (x^2 - y^2)`` -- two harmonic polynomials, fully 3-D, varying in all
  directions, single-valued (zero cohomology, a pure gradient). This exercises
  the Poisson solve and converges at ``O(h^p)`` with no truncation floor and no
  scale fit.

Two constructions of the potential, at the two ends of the de Rham complex,
both reusing the existing preconditioned solvers (no mixed saddle, no new
preconditioner):

  Route A (0-form potential):  H = grad f + a h1,   f in V0,  H in V1 (free).
      G0^T M1 G0 f = G0^T load1(B*)          -- k=0 stiffness, deflated CG.
  Route C (3-form potential):  B = delta f + a h2, f in V3,  B in V2.
      L3 f = -D2 b2*                          -- k=3 Hodge-Laplacian, MINRES;
      b2* the commuting interpolant of B* (no free-space mass solve),
      delta = weak grad = -M2^{-1} D2^T,  B = delta f.

The harmonic amplitude ``a = <B*, h>_M / <h, h>_M`` is the M-projection (the
VMEC-study c-fit); for the pure-gradient polynomial it comes out ~ 0.

Error vs the ANALYTIC field:
  ||B_h - B*||_M^2 = <B_h, B_h>_M - 2 (B_h . load) + int |B*|^2 dV,
with ``int |B*|^2 dV = sum_q w_q det(DF)_q |B*(F(xi_q))|^2`` and
``B_h . load = <B_h, B*>`` exact (``load_i = int Lambda_i . B*``).
"""

import argparse
import json
import os
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc",
                    help="equilibrium file or analytic name; domain only (its "
                         "field is ignored, B* is analytic).")
    ap.add_argument("--field", default="coil", choices=("coil", "polynomial", "toroidal"))
    ap.add_argument("--lam", type=float, default=1.0,
                    help="ripple amplitude for --field coil: B* = e_phi/R + lam grad(R^2 cos 2phi).")
    ap.add_argument("--ns", default="6,12,6:8,16,8:10,20,10:12,24,12",
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


def b_star_phys(seq, field, lam=1.0):
    """The analytic vacuum field as a lab-frame ``(3,)`` vector at the logical
    point ``xi`` (via the map). Every choice is curl-free and div-free.

    * ``toroidal``:   ``e_phi / R = grad(phi_geo)``     (1-form-exact; degenerate for A)
    * ``polynomial``: ``grad Psi``, ``Psi = (x^3-3xy^2) + z(x^2-y^2)``
    * ``coil``:       ``e_phi/R + lam grad(R^2 cos 2phi)`` -- TF flux + n=2 ripple.
    """
    import jax
    import jax.numpy as jnp

    def tf(X):  # 1/R e_phi = grad(phi_geo), the secular / flux part
        return jnp.array([-X[1], X[0], 0.0]) / (X[0] ** 2 + X[1] ** 2)

    if field == "toroidal":
        def f(xi):
            return tf(seq.map(xi))
        return f

    if field == "coil":
        grad_ripple = jax.grad(lambda X: X[0] ** 2 - X[1] ** 2)  # grad(R^2 cos 2phi)

        def f(xi):
            X = seq.map(xi)
            return tf(X) + lam * grad_ripple(X)
        return f

    def psi(X):  # two harmonic polynomials: fully 3-D, single-valued
        return X[0] ** 3 - 3.0 * X[0] * X[1] ** 2 + X[2] * (X[0] ** 2 - X[1] ** 2)
    grad_psi = jax.grad(psi)

    def f(xi):
        return grad_psi(seq.map(xi))
    return f


def analytic_norm_sq(seq, Bphys):
    """``int |B*|^2 dV = sum_q w_q det(DF)_q |B*(F(xi_q))|^2`` -- the M-norm of
    B* (physical L2 of the vector field, identical in V1 and V2)."""
    import jax
    import numpy as np
    Bq = np.asarray(jax.vmap(Bphys)(seq.quad.x))            # (Nq, 3) lab frame
    w = np.asarray(seq.quad.w)
    J = np.asarray(seq.jacobian_j)
    return float(np.sum(w * J * np.sum(Bq ** 2, axis=1)))


def run_rung(seq, ops, routes, field, lam, tag):
    import jax
    import numpy as np
    from mrx.geometry import map_jacobian_at
    from mrx.nullspace import estimate_spectral_gap, harmonic_rayleigh

    res = dict(tag=tag)
    Bphys = b_star_phys(seq, field, lam)
    bstar_sq = analytic_norm_sq(seq, Bphys)
    res["bstar_norm"] = float(np.sqrt(bstar_sq))
    out = {}

    if "A" in routes:
        # --- Route A: 0-form scalar potential, H in V1 (free) ----------------
        t0 = time.perf_counter()
        load1 = seq.load(Bphys, 1, dirichlet=False)                 # int L1 . B*
        rhs = seq.apply_incidence_matrix(load1, 0, dirichlet_in=False,
                                         dirichlet_out=False, transpose=True)  # G0^T load1
        f = seq.apply_inverse_laplacian(rhs, 0, dirichlet=False, operators=ops)
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
        # --- Route C: 3-form scalar potential, B = delta f in V2 -------------
        t0 = time.perf_counter()
        load2 = seq.load(Bphys, 2, dirichlet=False)                 # int L2 . B*
        b2 = seq.apply_inverse_mass_matrix(load2, 2, dirichlet=False, operators=ops)  # L2 proj (free V2)
        rhs = -seq.apply_derivative_matrix(b2, 2, dirichlet_in=False,
                                           dirichlet_out=True)      # -D2 b2*  (free V2 -> V3 dbc)
        f3 = seq.apply_inverse_laplacian(rhs, 3, dirichlet=True, operators=ops)
        r = seq.apply_laplacian(f3, 3, dirichlet=True, operators=ops) - rhs
        solve_res = float(np.linalg.norm(r) / (np.linalg.norm(rhs) + 1e-300))
        Bgrad = seq.apply_weak_grad(f3, dirichlet=False)           # delta f, free V2 (one-flag API)
        h2 = seq.nullspace(2, True)[0]                             # dbc harmonic 2-form
        # Bgrad (free V2) and h2 (dbc V2) live in different coefficient spaces --
        # the solid torus has no free harmonic 2-form (H^2(Omega)=0), the flux
        # generator is the relative/dbc one -- so pair them, and the analytic B*,
        # as physical vectors at the quadrature points (Piola pushforward, w*J
        # measure = the M inner product; metric factors explicit).
        DF_q = np.asarray(map_jacobian_at(seq.map, seq.quad.x))    # (Nq,3,3)
        Jq = np.asarray(seq.jacobian_j)
        wJ = np.asarray(seq.quad.w) * Jq

        def piola(dof, dirichlet):
            bhat = np.asarray(seq.evaluate_at_quadrature(dof, 2, dirichlet))
            return np.einsum("qik,qk->qi", DF_q, bhat) / Jq[:, None]

        def ip(a, b):
            return float(np.sum(wJ * np.sum(a * b, axis=1)))

        Bgrad_q = piola(Bgrad, False)
        h2_q = piola(h2, True)
        Bstar_q = np.asarray(jax.vmap(Bphys)(seq.quad.x))
        a2 = ip(Bstar_q, h2_q) / ip(h2_q, h2_q)                    # <B*,h2>/<h2,h2>
        Bh_q = Bgrad_q + a2 * h2_q
        relerr = float(np.sqrt(ip(Bh_q - Bstar_q, Bh_q - Bstar_q) / ip(Bstar_q, Bstar_q)))
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
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}  field {cli.field}", flush=True)

    records = []
    for ns in rungs:
        tag = f"{ns[0]}x{ns[1]}x{ns[2]}_p{cli.p}"
        t0 = time.perf_counter()
        seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter, tol=cli.tol)
        ops = seq.set_operators(compute_nullspaces(seq, ops))
        rec = run_rung(seq, ops, routes, cli.field, cli.lam, tag)
        rec.update(ns=list(ns), p=cli.p, h=1.0 / (ns[0] - cli.p),
                   n_elements=ns[0] - cli.p, tol=float(seq.tol),
                   t_total=time.perf_counter() - t0)
        records.append(rec)
        with open(os.path.join(cli.out, "analytic_vacuum.json"), "w") as fh:
            json.dump(dict(geometry=os.path.abspath(cli.geometry), field=cli.field,
                           lam=cli.lam, p=cli.p, records=records), fh, indent=2)

    print("\n=== convergence (relerr vs h = 1/n_el) ===", flush=True)
    for r in routes:
        hs = np.array([rec["h"] for rec in records])
        es = np.array([rec["routes"].get(r, {}).get("relerr", np.nan) for rec in records])
        print(f"route {r}:", flush=True)
        for i, rec in enumerate(records):
            rate = ""
            if i > 0 and np.isfinite(es[i]) and np.isfinite(es[i - 1]) and es[i] > 0:
                rate = f"  rate {np.log(es[i] / es[i - 1]) / np.log(hs[i] / hs[i - 1]):+.2f}"
            print(f"  {rec['tag']:>14}  h {hs[i]:.4f}  relerr {es[i]:.4e}{rate}", flush=True)
    print(f"\nwrote {os.path.join(cli.out, 'analytic_vacuum.json')}", flush=True)


if __name__ == "__main__":
    main(parse_args())
