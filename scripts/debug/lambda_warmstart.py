"""Solve for the warm-start lambda: no equilibrium code, no external data.

``booz_xform``'s own documentation treats lambda as GIVEN ("we assume that we
know the quantity lambda ..."), because in VMEC/GVEC it is an output of the
energy minimisation, not a formula.  But at FIXED geometry -- which is our
situation, the map is already built -- lambda is determined by a small linear
problem, and that is cheap enough to use as a warm start.


THE LAMBDA EQUATION
-------------------
With the map frozen, the field is linear in lambda,

    B_hat = ( 0,  Phi'(iota - lam_zeta),  Phi'(1 + lam_chi) )

so the magnetic energy is QUADRATIC in lambda:

    W = 1/2 int (B_hat^i g_ij B_hat^j) / J

Setting dW/dlambda = 0 and using ``g_ij B_hat^j / J = B_i`` (the covariant
components) gives

    d_chi(Phi' B_zeta) - d_zeta(Phi' B_chi) = 0   <=>   (curl B).grad rho = 0

i.e. the energy-minimising lambda is exactly the one that makes the CURRENT
tangent to the flux surfaces -- the nested-surface condition.  Written out,

    div(A grad lam) = -div(b),   A = adj(G)/J = (1/J)[[g_zz, -g_cz],
                                                     [-g_cz,  g_cc]]
                                 b = (1/J)(g_zz + iota g_cz,
                                           -g_cz - iota g_cc)

with G the angular block of the metric.  Three things make this cheap:

  * A is the adjugate of an SPD 2x2, hence SPD: the operator is ELLIPTIC.
  * lambda enters only through ANGULAR derivatives, so the flux surfaces
    DECOUPLE COMPLETELY -- one independent 2-D problem per rho.
  * the domain is a 2-torus, so a truncated Fourier basis turns each surface
    into a small dense SPD solve.  The nullspace is the constants, which is
    exactly lambda's gauge freedom.

Nothing here needs an equilibrium solve, an iteration, or a data file.

Note this is the FIXED-GEOMETRY lambda.  Full VMEC varies R, Z and lambda
together, so the surfaces themselves relax too; here they cannot.  How much
that costs is the point of the hegna comparison below.


THE AXISYMMETRIC CASE CLOSES IN FORM, AND VALIDATES THE SOLVE
-------------------------------------------------------------
If nothing depends on zeta then g_cz = 0 and the equation collapses to
``d_chi[g_zz(1 + lam_chi)/J] = 0``.  For ``toroid_map`` at kappa = 1,
``g_zz/J = R/(eps^2 rho)``, so

    1 + lam_chi  proportional to  1/R      =>   1 + lam_chi = <1/R>^-1 / R    (*)

using ``<1/R>_chi = 1/sqrt(R0^2 - eps^2 rho^2)`` (verified numerically to
2e-16).  So the closed form used in ``analytic_ic_verify.py`` is not ad hoc: it
is the axisymmetric solution of the lambda equation.  Expanding for small
``e = eps rho / R0``,

    lam ~ -e sin(theta) + (e^2/4) sin(2 theta) + O(e^3)

which is the familiar large-aspect-ratio result.  ``--case toroid`` checks the
general solve against (*) and should reproduce it to truncation error.


WHAT TO EXPECT ON A STELLARATOR
-------------------------------
Measured on the hegna export (see the commit log): lambda is NOT small.
|LA| reaches 0.549 rad = 31 deg, lam_chi reaches 0.83 (so B^zeta varies by 83%
over a surface), and lam_zeta reaches 0.35 against |iota| ~ 0.15-0.24 -- i.e.
lam_zeta is about TWICE iota through most of the volume, so setting lambda = 0
changes B^chi by more than 100% pointwise.

76% of lam_chi's variance is axisymmetric, which is why the 1/R form (*)
correlates at +0.83 with GVEC's lam_chi.  But lam_zeta is 0.000 axisymmetric --
necessarily, since an axisymmetric torus has lam_zeta = 0 identically -- so (*)
supplies NONE of the poloidal correction, which is the larger relative effect.
That is the gap this general solve is meant to close.

    python scripts/debug/lambda_warmstart.py --case toroid
    python scripts/debug/lambda_warmstart.py --case hegna --mpol 8 --ntor 6
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

from mrx.mappings import toroid_map  # noqa: E402


def fourier_modes(mpol, ntor, nfp):
    """(m, n, parity) list; parity 0 = cos, 1 = sin.  (0,0) excluded (gauge).

    n is in FIELD-PERIOD units, matching MRX's zeta in [0,1] spanning one
    period.  m > 0 only for n < 0 duplicates, so the usual half-plane is used.
    """
    modes = []
    for m in range(0, mpol + 1):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n <= 0:
                continue                      # (0,0) is gauge; n<0 duplicates
            for parity in (0, 1):
                modes.append((m, n, parity))
    return modes


def solve_lambda_surface(map_fn, rho, iota, modes, n_ang):
    """Minimise the surface energy over the Fourier coefficients of lambda.

    Returns the coefficient vector for ``modes`` at this rho.  Everything is a
    quadrature sum over a uniform (chi, zeta) grid, which is spectrally exact
    on a periodic domain.
    """
    ang = (jnp.arange(n_ang) + 0.5) / n_ang
    cc, zz = jnp.meshgrid(ang, ang, indexing='ij')
    chi, zet = cc.ravel(), zz.ravel()
    pts = jnp.stack([jnp.full(chi.shape, rho), chi, zet], axis=-1)

    DF = jax.jacfwd(map_fn)

    def metric_at(x):
        dF = DF(x)
        g = dF.T @ dF
        return jnp.array([g[1, 1], g[1, 2], g[2, 2], jnp.linalg.det(dF)])

    mg = jax.vmap(metric_at)(pts)
    g_cc, g_cz, g_zz, J = mg[:, 0], mg[:, 1], mg[:, 2], mg[:, 3]

    # (p, q) = (1 + lam_chi, iota - lam_zeta); the energy density is
    #   [g_zz p^2 + 2 g_cz p q + g_cc q^2] / J
    # p pairs with g_zz because p is the coefficient of B^zeta.
    two_pi = 2.0 * jnp.pi
    D = []
    for (m, n, parity) in modes:
        phase = two_pi * (m * chi - n * zet)
        if parity == 0:                        # cos(phase)
            d_chi = -two_pi * m * jnp.sin(phase)
            d_zet = two_pi * n * jnp.sin(phase)
        else:                                  # sin(phase)
            d_chi = two_pi * m * jnp.cos(phase)
            d_zet = -two_pi * n * jnp.cos(phase)
        D.append(jnp.stack([d_chi, -d_zet]))   # contribution to (p, q)
    D = jnp.stack(D, axis=-1)                  # (2, n_q, n_modes)

    Mq = jnp.stack([jnp.stack([g_zz, g_cz]),
                    jnp.stack([g_cz, g_cc])])  # (2, 2, n_q)
    c = jnp.stack([jnp.ones_like(chi), jnp.full(chi.shape, iota)])   # (2, n_q)

    MD = jnp.einsum('ijq,jqk->iqk', Mq, D)     # (2, n_q, n_modes)
    A = jnp.einsum('iqk,iql,q->kl', D, MD, 1.0 / J)
    r = jnp.einsum('iqk,iq,q->k', D, jnp.einsum('ijq,jq->iq', Mq, c), 1.0 / J)
    return jnp.linalg.solve(A, -r)


def eval_lambda(coeffs, modes, chi, zet):
    """lambda and its two angular derivatives, in MRX normalised coordinates."""
    two_pi = 2.0 * np.pi
    lam = np.zeros_like(chi)
    d_chi = np.zeros_like(chi)
    d_zet = np.zeros_like(chi)
    for a, (m, n, parity) in zip(coeffs, modes):
        phase = two_pi * (m * chi - n * zet)
        if parity == 0:
            lam += a * np.cos(phase)
            d_chi += -a * two_pi * m * np.sin(phase)
            d_zet += a * two_pi * n * np.sin(phase)
        else:
            lam += a * np.sin(phase)
            d_chi += a * two_pi * m * np.cos(phase)
            d_zet += -a * two_pi * n * np.cos(phase)
    return lam, d_chi, d_zet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=("toroid", "hegna"), default="toroid")
    ap.add_argument("--iota", type=float, default=0.4,
                    help="constant iota for the solve (per MRX toroidal turn)")
    ap.add_argument("--mpol", type=int, default=8)
    ap.add_argument("--ntor", type=int, default=0,
                    help="toroidal modes in field-period units; 0 for the "
                         "axisymmetric toroid")
    ap.add_argument("--n-ang", type=int, default=32)
    ap.add_argument("--n-rho", type=int, default=17)
    ap.add_argument("--eps", type=float, default=1 / 3)
    ap.add_argument("--R0", type=float, default=1.0)
    ap.add_argument("--map-ns", default="12,24,12")
    ap.add_argument("--map-p", type=int, default=3)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    if cli.case == "toroid":
        map_fn = toroid_map(epsilon=cli.eps, kappa=1.0, R0=cli.R0)
        nfp = 1
    else:
        from mrx.gvec import build_gvec_map, gvec_path  # noqa: PLC0415
        ns = tuple(int(v) for v in cli.map_ns.split(","))
        map_fn, info = build_gvec_map(gvec_path("hegna"),
                                      map_ns=ns, p=cli.map_p)
        nfp = info["nfp"]
        print(f"[geom] hegna nfp={nfp} sign={info['sign']:+.0f}")

    modes = fourier_modes(cli.mpol, cli.ntor, nfp)
    print(f"[setup] case={cli.case}  iota={cli.iota}  "
          f"mpol={cli.mpol} ntor={cli.ntor} -> {len(modes)} coefficients  "
          f"n_ang={cli.n_ang}^2")

    rhos = np.linspace(0.1, 0.95, cli.n_rho)
    t0 = time.perf_counter()
    coeffs = np.array([np.asarray(solve_lambda_surface(
        map_fn, float(r), cli.iota, modes, cli.n_ang)) for r in rhos])
    print(f"[solve] {cli.n_rho} surfaces in {time.perf_counter() - t0:.1f}s "
          f"({(time.perf_counter() - t0) / cli.n_rho * 1e3:.0f} ms/surface)")

    results = {"case": cli.case, "iota": cli.iota, "mpol": cli.mpol,
               "ntor": cli.ntor, "nfp": nfp, "rho": rhos.tolist(),
               "modes": [list(m) for m in modes],
               "coeffs": coeffs.tolist()}

    # sample grid for all comparisons
    na = 64
    ang = (np.arange(na) + 0.5) / na
    CH, ZE = np.meshgrid(ang, ang, indexing='ij')

    if cli.case == "toroid":
        print("\n[check] general solve vs the closed form  "
              "1 + lam_chi = <1/R>^-1 / R")
        print("    rho    max|lam_chi| solved   closed form    rel.resid")
        rel_all = []
        for i, r in enumerate(rhos):
            _, dchi, _ = eval_lambda(coeffs[i], modes, CH, ZE)
            R = cli.R0 + cli.eps * r * np.cos(2 * np.pi * CH)
            exact = np.sqrt(cli.R0 ** 2 - (cli.eps * r) ** 2) / R - 1.0
            rel = np.linalg.norm(dchi - exact) / np.linalg.norm(exact)
            rel_all.append(rel)
            if i % max(1, cli.n_rho // 6) == 0:
                print(f"    {r:5.3f}   {np.abs(dchi).max():16.6f}   "
                      f"{np.abs(exact).max():11.6f}   {rel:10.3e}")
        print(f"[check] worst relative residual over all surfaces: "
              f"{max(rel_all):.3e}")
        results["toroid_rel_resid"] = rel_all

    else:
        import h5py  # noqa: PLC0415
        from mrx.gvec import gvec_path  # noqa: PLC0415
        with h5py.File(gvec_path("hegna"), "r") as h:
            S = (int(h.attrs["n_rho"]), int(h.attrs["n_theta"]),
                 int(h.attrs["n_zeta"]))
            Lt = np.asarray(h["clebsch"]["dLA_dt"]).reshape(S)
            Lz = np.asarray(h["clebsch"]["dLA_dz"]).reshape(S) / nfp
            rho_d = np.unique(np.asarray(h["eval_points"])[:, 0])
        # GVEC grid -> MRX normalised lambda derivatives (see mrx.gvec.load_clebsch)
        print("\n[check] general solve vs GVEC's own lambda")
        print("    rho   corr(lam_chi)  rel.resid  |  corr(lam_zeta)  rel.resid")
        out = []
        for i, r in enumerate(rhos):
            j = int(np.argmin(np.abs(rho_d - r)))
            # compared on GVEC's own angular grid, below
            # GVEC data is on its own (theta, zeta) grid; compare on that grid
            ga = (np.arange(S[1]) / (S[1] - 1))
            GC, GZ = np.meshgrid(ga, ga, indexing='ij')
            _, dchi_g, dzet_g = eval_lambda(coeffs[i], modes, GC, GZ)
            def cs(a, b):
                a, b = a.ravel(), b.ravel()
                return (float(np.corrcoef(a, b)[0, 1]),
                        float(np.linalg.norm(a - b) / np.linalg.norm(b)))
            c1, r1 = cs(dchi_g, Lt[j])
            c2, r2 = cs(dzet_g, Lz[j])
            out.append((float(r), c1, r1, c2, r2))
            if i % max(1, cli.n_rho // 8) == 0:
                print(f"    {r:5.3f}   {c1:+11.4f}  {r1:9.4f}  |  "
                      f"{c2:+12.4f}  {r2:9.4f}")
        results["hegna_compare"] = out
        arr = np.array(out)
        print(f"[check] median corr  lam_chi {np.median(arr[:,1]):+.4f}   "
              f"lam_zeta {np.median(arr[:,3]):+.4f}")
        print(f"[check] median rel.resid  lam_chi {np.median(arr[:,2]):.4f}   "
              f"lam_zeta {np.median(arr[:,4]):.4f}")
        print("[check] reference: the 1/R closed form gives corr +0.83 on "
              "lam_chi and captures NONE of lam_zeta.")

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
