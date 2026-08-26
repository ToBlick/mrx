"""Verify the logical-profile IC against cases where the answer is known.

Two cases, both built from the same ansatz as ``mrx.initial_conditions`` --
primal reference 2-form components ``omega = (0, Phi'(iota - lam_zeta),
Phi'(1 + lam_chi))`` -- and both chosen because the truth is available in
closed form rather than by comparison with another code.


CASE screwpinch -- the cylinder, where the ansatz IS an equilibrium
------------------------------------------------------------------
On ``cylinder_map(a, h)`` with lambda = 0 and rho-only profiles, everything is
axisymmetric AND z-independent, so the configuration is a straight screw pinch
and the exact radial force balance is

    dp/dr + d/dr[(B_theta^2 + B_z^2)/2] + B_theta^2 / r = 0

(no assumption that B_z is constant).  Pushing the ansatz forward on that map,

    B_theta = omega^chi / (a h)          B_z = omega^zeta / (2 pi a^2 rho)

and with ``r = a rho`` the factor a cancels out of the balance entirely:

    dp/drho = -d/drho[(B_theta^2 + B_z^2)/2] - B_theta^2 / rho              (1)

For ``Phi' = rho^q`` and POLYNOMIAL iota, every term in (1) is a polynomial in
rho, so p follows in closed form by exact polynomial antidifferentiation -- no
quadrature, no fit.  That is the reference this case compares against.

Two independent checks fall out, and neither has a fitted parameter:

  * ``||F||/||B||`` must sit at DISCRETISATION error, because the field is a
    true equilibrium.  This is the sharper of the two: it tests the force
    operator without going near the pressure solve.
  * ``dp/drho`` from ``compute_force`` must match (1).  Comparing the GRADIENT
    sidesteps the gauge entirely -- the k=2 Leray branch fixes p only up to a
    constant.  B is normalised to ``||B||_M = 1`` and p is quadratic in B, so
    the measured p is scaled back by ``B_norm^2`` before comparison; there is
    no free multiplicative constant left.

This generalises the single z-pinch of ``test/test_relaxation.py``, which is the
``B_z = 0`` corner of the same family.


CASE toroid -- where lambda = 0 is NOT even the vacuum field
------------------------------------------------------------
For ``toroid_map(epsilon, R0)`` at kappa = 1 the Jacobian is
``sqrt(g) = 4 pi^2 eps^2 rho R`` with ``R = R0 + eps rho cos(2 pi chi)``, so

    B_phi = Phi'(rho) (1 + lam_chi) / (2 pi eps^2 rho)

and the R cancels: with lambda = 0 the toroidal field has NO 1/R dependence at
all.  Requiring ``R B_phi`` to be a flux function forces

    1 + lam_chi = <1/R>^-1 / R,      <1/R>_chi = 1 / sqrt(R0^2 - eps^2 rho^2)   (2)

using the standard average ``(1/2pi) int dtheta/(A + B cos theta)
= 1/sqrt(A^2 - B^2)``.  The toroid is axisymmetric, so ``lam_zeta = 0``
identically and (2) is the whole of lambda; div B stays exactly zero because
B^chi is rho-only and nothing depends on zeta.

LAMBDA IS NOT ENOUGH ON ITS OWN -- measured 2026-08-25, and it corrected me.
The first version of this script asserted that ``iota = 0`` plus the closed form
(2) would BE the vacuum field, so its force would collapse while lambda = 0 left
an O(1) force.  The run said the opposite: lambda = 0 gave 3.93e-05 and the
closed form gave 1.73e-02.  Both numbers are right and the ASSERTION was wrong.

A purely toroidal vacuum field needs ``R B_phi`` to be a GLOBAL CONSTANT, not a
flux function.  Eq. (2) only forces ``1 + lam_chi = c(rho)/R``, so

    R B_phi = c(rho) Phi'(rho) / (2 pi eps^2 rho)

is still free to vary with rho -- with ``Phi' = rho`` it goes as
``sqrt(R0^2 - eps^2 rho^2)``, a 5.4% spread, so the field carries poloidal
current and a nonzero force is CORRECT.  Fixing it is a job for the FLUX
profile, not for lambda:

    Phi'(rho) = rho <1/R> = rho / sqrt(R0^2 - eps^2 rho^2)      (``--flux vacuum``)

gives ``B_phi = 1/(2 pi eps^2 R)`` and ``R B_phi`` constant to 1e-12.  THAT pair
is the vacuum field, and it is the arm whose force must collapse.

The lambda = 0 arm is small for its own reason, also not anticipated: with
``Phi' = rho`` and ``iota = 0`` the R cancels out of B_phi entirely (see above),
leaving ``B_phi = const``, so ``J x B = grad(-B_phi^2 ln R)`` is a PURE GRADIENT
and P_Leray removes all of it.  A tiny force there means "the residual was a
gradient", NOT "the field was an equilibrium".

At iota != 0 a residual force is expected regardless: (2) fixes the
toroidal-field condition only, and Grad-Shafranov also constrains the surface
SHAPES (the Shafranov shift), which is a property of the map and not something
lambda can supply.  The run reports the ratio so the residual is visible rather
than asserted.

    python scripts/debug/analytic_ic_verify.py --case screwpinch --iota 0.4,0,0.5
    python scripts/debug/analytic_ic_verify.py --case toroid --iota 0.0
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
from numpy.polynomial import Polynomial as P


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.differential_forms import DiscreteFunction, Pushforward  # noqa: E402
from mrx.mappings import cylinder_map, toroid_map  # noqa: E402
from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.relaxation import compute_force  # noqa: E402


def build_seq(which, ns, degree, maxiter, a, h, eps, R0):
    seq = DeRhamSequence(ns, (degree,) * 3, 2 * degree,
                         ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=maxiter,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    seq.set_map(cylinder_map(a=a, h=h) if which == "screwpinch"
                else toroid_map(epsilon=eps, kappa=1.0, R0=R0))
    ops = op.assemble_incidence_operators(seq)
    seq.set_operators(ops)
    ops = seq.assemble_all_sparse(include_preconditioners=False)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    return seq, ops


def analytic_screwpinch(iota_c, q, a, h):
    """Exact ``p(rho)`` and ``dp/drho`` for eq. (1), by polynomial algebra.

    ``iota_c`` are coefficients lowest-order first.  Returns
    ``(p_poly, dp_poly)`` with the constant fixed by ``p(1) = 0``.
    """
    iota = P(list(iota_c))
    rho = P([0.0, 1.0])
    # B_theta = iota rho^q / (a h);  B_z = rho^(q-1) / (2 pi a^2)
    Bt2 = (iota * iota) * rho ** (2 * q) / (a * h) ** 2
    Bz2 = rho ** (2 * q - 2) / (2 * np.pi * a ** 2) ** 2
    u = (Bt2 + Bz2) / 2.0
    # B_theta^2 / rho is a polynomial for q >= 1 (lowest power is 2q-1 >= 1)
    Bt2_over_rho = (iota * iota) * rho ** (2 * q - 1) / (a * h) ** 2
    dp = -(u.deriv() + Bt2_over_rho)
    p = dp.integ()
    return p - p(1.0), dp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=("screwpinch", "toroid"),
                    default="screwpinch")
    ap.add_argument("--iota", default="0.4,0,0.5",
                    help="polynomial coefficients in rho, LOWEST order first")
    ap.add_argument("--flux-exp", type=int, default=1,
                    help="Phi' = rho^q; q >= 1 keeps eq. (1) polynomial")
    ap.add_argument("--flux", choices=("power", "vacuum"), default="power",
                    help="'vacuum' overrides --flux-exp with "
                         "Phi' = rho <1/R> = rho / sqrt(R0^2 - eps^2 rho^2), "
                         "which is what the closed-form lambda needs to give "
                         "the TRUE vacuum field (see below). toroid only.")
    ap.add_argument("--ns", default="12,24,4")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--a", type=float, default=0.33)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1 / 3)
    ap.add_argument("--R0", type=float, default=1.0)
    ap.add_argument("--n-rho", type=int, default=41)
    ap.add_argument("--n-ang", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    iota_c = [float(v) for v in cli.iota.split(",")]
    q = cli.flux_exp
    if q < 1:
        raise ValueError("flux-exp must be >= 1 for eq. (1) to stay polynomial")
    if cli.flux == "vacuum" and cli.case != "toroid":
        raise ValueError("--flux vacuum is defined only for --case toroid; the "
                         "screwpinch analytic pressure assumes Phi' = rho^q")
    iota_j = jnp.asarray(iota_c)

    print(f"[setup] case={cli.case} ns={ns} p={cli.p}  iota(rho) coeffs "
          f"{iota_c} (lowest first)  Phi' = rho^{q}")
    t0 = time.perf_counter()
    seq, ops = build_seq(cli.case, ns, cli.p, cli.maxiter,
                         cli.a, cli.h, cli.eps, cli.R0)
    print(f"[setup] {time.perf_counter() - t0:.1f}s", flush=True)

    def iota_fn(r):
        return jnp.sum(iota_j * r ** jnp.arange(len(iota_c)))

    def lam_chi_toroid(x):
        """Eq. (2): the closed-form lambda_chi for the circular toroid."""
        r, chi = x[0], x[1]
        R = cli.R0 + cli.eps * r * jnp.cos(2 * jnp.pi * chi)
        return jnp.sqrt(cli.R0 ** 2 - (cli.eps * r) ** 2) / R - 1.0

    DF_map = jax.jacfwd(seq.map)

    def dPhi_fn(r):
        """Phi'(rho).  'vacuum' is rho <1/R>; see the note in main()."""
        if cli.flux == "vacuum":
            return r / jnp.sqrt(cli.R0 ** 2 - (cli.eps * r) ** 2)
        return r ** q

    def make_field(with_lambda):
        def omega_ref(x):
            r = x[0]
            f = dPhi_fn(r)
            lam_c = lam_chi_toroid(x) if (with_lambda and cli.case == "toroid") \
                else 0.0
            # lam_zeta is identically zero in both cases: the cylinder has
            # lambda = 0 and the toroid is axisymmetric.
            return jnp.array([0.0, f * iota_fn(r), f * (1.0 + lam_c)])

        def B_phys(x):
            dF = DF_map(x)
            return dF @ omega_ref(x) / jnp.linalg.det(dF)

        B_raw = seq.apply_inverse_mass_matrix(
            seq.load(B_phys, 2, dirichlet=True), 2, dirichlet=True)
        nrm = float(seq.l2_norm(B_raw, 2, dirichlet=True))
        return B_raw / nrm, nrm

    rhos = np.linspace(0.05, 0.95, cli.n_rho)
    ang = (np.arange(cli.n_ang) + 0.5) / cli.n_ang

    def spts(r):
        return jnp.asarray([[r, c, z] for c in ang for z in ang])

    results = {"case": cli.case, "ns": list(ns), "p": cli.p,
               "iota_coeffs": iota_c, "flux_exp": q, "a": cli.a, "h": cli.h,
               "eps": cli.eps, "R0": cli.R0, "rho": rhos.tolist(), "arms": {}}

    arms = [("lambda0", False)] + ([("lambda_closed", True)]
                                   if cli.case == "toroid" else [])
    for name, with_lam in arms:
        t1 = time.perf_counter()
        B, B_norm = make_field(with_lam)
        B_h = DiscreteFunction(B, seq.basis_2, seq.e2_dbc)
        comp = np.array([np.abs(np.asarray(jax.vmap(B_h)(spts(r)))).max(axis=0)
                         for r in rhos])
        brho = float(comp[:, 0].max() / comp[:, 2].max())
        div = float(seq.l2_norm(seq.apply_strong_div(B), 3, dirichlet=True))
        F, p_dof, _, _, _ = compute_force(B, seq)
        F_rel = float(seq.l2_norm(F, 2, dirichlet=True))
        print(f"\n[{name}] built+forced in {time.perf_counter() - t1:.1f}s   "
              f"||B||_M raw = {B_norm:.6e}")
        print(f"[{name}] max|B^rho|/max|B^zeta| = {brho:.3e}   "
              f"||div B||_L2 = {div:.3e}")
        print(f"[{name}] ||F||_L2 / ||B||_L2 = {F_rel:.6e}")
        arm = {"B_norm": B_norm, "Brho_rel": brho, "div": div,
               "force_rel": F_rel}

        p_h = Pushforward(DiscreteFunction(p_dof, seq.basis_3, seq.e3_dbc),
                          seq.map, 3)
        p_meas = np.array([
            float(jnp.mean(jax.vmap(lambda x: p_h(x)[0])(spts(r))))
            for r in rhos]) * B_norm ** 2          # undo the normalisation
        arm["p_measured"] = p_meas.tolist()

        if cli.case == "screwpinch":
            p_ex, dp_ex = analytic_screwpinch(iota_c, q, cli.a, cli.h)
            dp_meas = np.gradient(p_meas, rhos)
            dp_exact = dp_ex(rhos)
            # Gradient comparison: gauge-free AND scale-free, no fit at all.
            rel = float(np.linalg.norm(dp_meas - dp_exact)
                        / np.linalg.norm(dp_exact))
            # p itself, with only the (solver-defined) constant removed.
            pm = p_meas - p_meas.mean()
            pe = p_ex(rhos) - p_ex(rhos).mean()
            rel_p = float(np.linalg.norm(pm - pe) / np.linalg.norm(pe))
            print(f"[{name}] dp/drho vs analytic: relative error {rel:.4e}"
                  f"   (no fitted parameter)")
            print(f"[{name}] p vs analytic (constant removed): {rel_p:.4e}")
            print(f"\n[{name}]   rho    dp/drho meas     dp/drho exact"
                  f"        p meas         p exact")
            for i in range(0, cli.n_rho, max(1, cli.n_rho // 10)):
                print(f"        {rhos[i]:5.3f}  {dp_meas[i]:14.6e}  "
                      f"{dp_exact[i]:14.6e}  {pm[i]:14.6e}  {pe[i]:14.6e}")
            arm.update(dp_measured=dp_meas.tolist(),
                       dp_exact=dp_exact.tolist(),
                       p_exact=p_ex(rhos).tolist(),
                       dp_rel_error=rel, p_rel_error=rel_p)

        results["arms"][name] = arm

    if cli.case == "toroid":
        f0 = results["arms"]["lambda0"]["force_rel"]
        f1 = results["arms"]["lambda_closed"]["force_rel"]
        print(f"\n[toroid] ||F||/||B||   lambda=0: {f0:.6e}   "
              f"closed-form lambda: {f1:.6e}   ratio {f0 / f1:.3f}x")
        if all(c == 0.0 for c in iota_c):
            print("[toroid] iota = 0, so the closed form should BE the vacuum "
                  "field (J = 0) and its force should sit at discretisation "
                  "error.")
        else:
            print("[toroid] iota != 0: eq. (2) fixes the toroidal-field "
                  "condition only.  A residual force is EXPECTED -- "
                  "Grad-Shafranov also constrains the surface shapes.")
        results["force_ratio"] = f0 / f1

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
