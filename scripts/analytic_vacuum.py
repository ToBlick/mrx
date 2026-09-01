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

  Route A (scalar potential):  H = grad f + a h1,   f in V0,  H in V1 (free).
      Curl-free H-field: G0^T M1 G0 f = G0^T load1(B*)  -- k=0 stiffness,
      deflated CG. a = <B*, h1>_M / <h1, h1>_M pins the flux (the VMEC-study
      c-fit); for 1/R it is degenerate (grad of a coordinate is exact in V1).
  Route C (vector potential):  B = curl A,          A in V1,  B in V2 (free).
      Div-free B-field: <curl A, curl W> = <B*, curl W>  =>  L1 A = C^T load2,
      the preconditioned k=1 Hodge-Laplacian saddle (well conditioned; NO k=2
      Laplacian inverse). On the solid torus b2 = 0 so B* = curl A fully, flux
      included -- no harmonic term. The weak-grad-of-a-3-form route is WRONG
      here: it yields co-exact (compressible) 2-forms, of which a solenoidal
      field has none.

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
    import numpy as np
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
        # --- Route C: vector potential, B = curl A in V2 (free) --------------
        # A vacuum field is div-free, hence EXACT (curl of a 1-form), NOT
        # co-exact -- so the 2-form route is the vector potential, not the weak
        # grad of a 3-form (that lands in the co-exact/compressible part, which a
        # solenoidal field has none of). On the solid torus b2 = 0, so a closed
        # free 2-form is fully exact: B* = curl A, the toroidal flux carried by
        # A's boundary circulation -- no separate harmonic term.
        #   curl-curl:  <curl A, curl W> = <B*, curl W>  =>  L1 A = C^T load2.
        # C^T load2 is orthogonal to gradients and to the harmonic 1-forms
        # (curl(grad)=0, curl h1=0), so it sits in the co-exact subspace where L1
        # acts as curl-curl; solved by the preconditioned k=1 Hodge-Laplacian
        # saddle (well conditioned -- and no k=2 Laplacian inverse anywhere).
        t0 = time.perf_counter()
        load2 = seq.load(Bphys, 2, dirichlet=False)                # M2-load of B*
        rhs1 = seq.apply_incidence_matrix(load2, 1, dirichlet_in=False,
                                          dirichlet_out=False, transpose=True)  # C^T load2
        A, info = seq.apply_inverse_laplacian(rhs1, 1, dirichlet=False,
                                              operators=ops, return_info=True)
        Bh = seq.apply_strong_curl(A, dirichlet_in=False, dirichlet_out=False)  # curl A, free V2
        MBh = seq.apply_mass_matrix(Bh, 2, False)
        r1 = seq.apply_laplacian(A, 1, dirichlet=False, operators=ops) - rhs1
        solve_res = float(np.linalg.norm(r1) / (np.linalg.norm(rhs1) + 1e-300))
        err_sq = float(Bh @ MBh) - 2.0 * float(Bh @ load2) + bstar_sq
        relerr = float(np.sqrt(max(err_sq, 0.0)) / np.sqrt(bstar_sq))
        out["C"] = dict(relerr=relerr, solve_res=solve_res, n=int(seq.n(2, False)),
                        t=time.perf_counter() - t0)
        _log(f"{tag} C: relerr {relerr:.4e}  solve_res {solve_res:.1e}  "
             f"n2 {seq.n(2, False)}  info {info}  {out['C']['t']:.1f}s")

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
