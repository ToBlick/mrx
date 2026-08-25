"""Relaxation initial conditions from prescribed logical profiles -- no GVEC.

A k=2 field in the REFERENCE 2-form frame (component order
``dchi^dzeta, dr^dzeta, dr^dchi`` = the rho-, chi- and zeta-directed logical
fluxes).  This IS the GVEC / VMEC representation -- the reference 2-form
components are exactly ``sqrt(g) B^i``, since ``B_phys = DF B_hat / J``:

    B_hat^rho  = 0
    B_hat^chi  = dPhi(rho) * ( iota(rho) - dlambda/dzeta )
    B_hat^zeta = dPhi(rho) * (     1     + dlambda/dchi  )

with ``lambda(rho, chi, zeta)`` the VMEC stream function that turns the map's
poloidal angle into a straight-field-line angle.  ``--lam`` sets it; the default
``lambda = 0`` is the straight-field-line case, in which the map's own angles
are the field's straight-field-line angles.

This is meant as a RELAXATION STARTING POINT, not an equilibrium: it is fine,
and expected, that the initial force is large.  What matters is that the IC is
topologically right, reproducible without external data, and that its ideal
invariants are known analytically.  Since the w7x / quasr / hegna maps are built
on GVEC's own angle grid, the map's angle IS the VMEC angle, so the equilibrium
field on those maps has lambda != 0 -- and lambda = 0 is exactly how far this IC
sits from it.  Generating lambda is the relaxation's job.


WHAT LAMBDA DOES AND DOES NOT CHANGE
------------------------------------
  * div B = 0 for ANY lambda:  ``d_chi(B^chi) + d_zeta(B^zeta)
    = dPhi (-lambda_zetachi + lambda_chizeta) = 0`` -- the mixed partials
    cancel identically.  ``B^rho = 0`` is untouched, so nested surfaces survive.
    lambda is genuinely free.
  * The FLUXES are unchanged: lambda_chi and lambda_zeta integrate to zero over
    the periodic angles, so Phi(rho), X(rho) and hence iota(rho) are exactly
    preserved.  lambda redistributes the field WITHIN a surface, never across.
  * HELICITY is unchanged: lambda is a pure gauge transformation of the
    potential.  With ``d_rho mu = dPhi * lambda`` the potential becomes
    ``A = A_0 + d mu``, and mu is single-valued, so eq. (1) below is blind to
    lambda.  Within the whole GVEC family H depends only on (Phi, iota).

That last point is a sharp, falsifiable check, and the ``--lam`` arms of
``slurm/job_logical_ic.sh`` exist to run it: turning lambda on must move the
force and the Pfirsch-Schlueter spread while leaving H where it was.

lambda also breaks a symmetry that would otherwise be permanent.  With
lambda = 0 the IC depends on rho alone, so on an axisymmetric map the whole
relaxation stays axisymmetric forever -- the evolution cannot break a symmetry
the IC and the map share.  A lambda with n != 0 breaks it.

Three properties come from the ansatz itself, before any solve, on ANY
geometry:

  * ``B^rho = 0``  -- B is tangent to every rho = const surface, so the IC has
    perfectly nested flux surfaces.  A pushforward maps tangent fields to
    tangent fields, so this holds in physical space too.
  * ``div B = 0``  -- the discrete divergence is  d_rho(c0) + d_chi(c1) +
    d_zeta(c2); with c0 = 0 and c1, c2 functions of rho alone every term
    vanishes identically.
  * ``B.n = 0``    -- which is exactly what the k=2 Dirichlet space enforces.

and one by construction: field lines obey ``dchi/dzeta = B^chi / B^zeta``, so
``iota(rho)`` IS the rotational transform, with no metric in the way.


HELICITY -- exact, analytic, and metric-free
--------------------------------------------
Take the potential  ``A = Phi(rho) dchi - X(rho) dzeta``  with

    Phi(rho) = int_0^rho f_zeta      (toroidal flux)
    X(rho)   = int_0^rho f_chi       (poloidal flux),   f_chi = iota f_zeta

Then ``dA`` reproduces the ansatz exactly, and

    A ^ dA = (Phi X' - X Phi') drho ^ dchi ^ dzeta

so, since chi and zeta both run over [0, 1),

    H = int_0^1 (Phi X' - X Phi') drho = int_0^1 Phi(rho)^2 W'(rho) drho       (1)

with ``W = X / Phi`` the flux-averaged transform.  No metric appears -- helicity
is topological, so (1) is as exact on W7-X as on a cylinder.  Consequences:

  * H = 0 identically when iota is CONSTANT.  Helicity is generated purely by
    SHEAR in W, never by the transform itself.
  * The weight is Phi^2, so shear near the edge counts far more than shear near
    the axis.
  * For the power-law profiles below it closes:  with dPhi = rho^q,
    iota = iota0 + Diota rho^e,

        H = Diota * e / [ (q+1)(q+e+1)(2q+e+2) ]                              (2)

    linear in the shear Diota, so the knob is direct.

CAVEAT: on a torus (b1 = 1) helicity is gauge-ambiguous by the harmonic 1-form.
Eq. (1) is the value in the natural gauge above; ``mrx.relaxation``'s
``compute_helicity`` picks a different one (it solves for the co-exact A via the
Hodge Laplacian and adds the harmonic remainder back).  The two therefore do NOT
have to agree, and this script reports both plus their difference, which is the
harmonic contribution.


RELATION TO THE CLEBSCH REPRESENTATION
--------------------------------------
``B = dPhi ^ dchi - dX ^ dzeta``.  When iota is constant, X = iota Phi and this
factors:

    B = dPhi ^ d(chi - iota zeta)

which is exactly Clebsch, ``B = grad alpha x grad beta`` with alpha = Phi and
beta = chi - iota zeta the straight-field-line angle.  In the Clebsch gauge
``A = alpha grad beta``, so ``A.B = alpha grad beta . (grad alpha x grad beta)``
vanishes POINTWISE -- which is why constant iota gives H = 0 above.  The two
statements are the same fact.  With iota varying there is no global two-function
Clebsch, and the obstruction is precisely the helicity.

So the two-function Clebsch is the doubly-degenerate corner of the family here:
lambda = 0 AND zero shear.  Restoring the shear generalises it (H != 0);
restoring lambda generalises it the other way (H unchanged) and recovers the
full GVEC representation.  ``gvec_geometry.py`` lists
``gvec_nfp3_hegna_80cubed_clebsch.h5``, i.e. the hegna export is already in a
Clebsch representation -- the correspondence ``beta <-> chi - iota zeta`` and
GVEC's exact lambda sign convention are worth checking against that file
directly; neither is asserted here.


IOTA AND PRESSURE, WITHOUT ASSUMING A CYLINDER
----------------------------------------------
The exact geometry-general statement is the surface-averaged radial force
balance: for nested surfaces ``grad p = p'(rho) grad rho``, so

    p'(rho) = < (J x B).grad rho > / < |grad rho|^2 >                          (3)

exact, but it needs J x B -- a diagnostic, not a predictor.  The predictive
content sits one level up, in the energy.  The physical field is
``B = (f_chi d_chi F + f_zeta d_zeta F) / J``, so its surface-averaged energy
density depends on geometry ONLY through three metric surface averages plus the
volume element:

    a = <g_chichi/J>   b = <g_chizeta/J>   c = <g_zetazeta/J>   V' = <J>

    u(rho) = 1/2 f_zeta^2 [ a iota^2 + 2 b iota + c ] / V'                     (4)

All four are pure geometry -- no solve, no assembly, just the map.  On the
cylinder map (4) reduces to ``(B_theta^2 + B_z^2)/2`` identically, so the
cylindrical screw pinch is the special case, not the general rule.  Then:

  * (4) is a QUADRATIC in iota with a > 0, so it inverts: given a target
    du/drho you can solve for the iota that delivers it, per surface, from the
    metric alone.
  * b(rho) is purely 3-D -- zero on the cylinder, nonzero wherever the
    theta-zeta metric coupling is.  It makes the energy-minimising transform
    ``iota*(rho) = -b/a``, not zero: the geometry has a preferred iota.  On
    W7-X theta-zeta is the largest off-diagonal, so iota* should be substantial
    there and exactly zero on the cylinder.
  * In 3-D ``(J x B).grad rho / |grad rho|^2`` is generically NOT a flux
    function, so no (iota, dPhi) is an equilibrium.  Its spread over a surface
    is the Pfirsch-Schlueter drive -- reported, not removed.


THE CONSTRUCTION ROUTE, AND WHY IT IS NOT HISTOPOLATION
-------------------------------------------------------
The structure-preserving way to build the DoFs would be histopolation: its DoFs
are logical face fluxes, it is local and tensor-product, and it commutes with d,
so ``c0 = 0`` and rho-only c1, c2 survive exactly.  The L2 route used below goes
through ``M_2``, which carries the metric and couples the three components in
3-D, so it can reintroduce a rho-component.

Histopolation is NOT available here: ``_require_full_tensor_space`` rejects any
nontrivial extraction, which rules out both ``dirichlet=True``
(``n2_dbc < basis_2.n``) and ``polar=True`` -- and this IC is both.  So this
script MEASURES the damage the L2 projection does (gate 1 below) instead of
assuming it away.  If the bulk rho-component comes back large, lifting that
guard for selection-type extractions is the follow-up; ``frame='ref'`` on
``interpolate`` is already in place for it.

    python scripts/debug/logical_profile_ic.py --geometry cylinder --iota 0.4,0.9
    python scripts/debug/logical_profile_ic.py --geometry w7x --iota 0.85,1.0
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

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.differential_forms import DiscreteFunction, Pushforward  # noqa: E402
from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.relaxation import compute_force, compute_helicity  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


# ---------------------------------------------------------------------------
# The profiles, and the invariants that follow from them in closed form
# ---------------------------------------------------------------------------

def make_profiles(iota0, iota1, iota_exp, flux_exp):
    """``iota(rho) = iota0 + (iota1-iota0) rho^iota_exp``, ``dPhi = rho^flux_exp``.

    ``flux_exp = 1`` is the uniform-toroidal-field choice: B^zeta is a flux
    DENSITY in logical coordinates, so a spatially uniform physical B_z has
    dPhi/drho proportional to rho, not constant.  Both profiles vanish at the
    axis as they must -- the area element does.
    """
    def iota(r):
        return iota0 + (iota1 - iota0) * r ** iota_exp

    def dPhi(r):
        return r ** flux_exp

    return iota, dPhi


def parse_lambda(spec):
    """``"m,n,amp;m,n,amp;..."`` -> list of (m, n, amp).  Empty string -> []."""
    if not spec:
        return []
    modes = []
    for term in spec.split(";"):
        m, n, amp = term.split(",")
        modes.append((int(m), int(n), float(amp)))
    return modes


def make_lambda(modes):
    """``lambda = sum_k amp_k rho^|m_k| sin(2 pi (m_k chi - n_k zeta))``.

    The ``rho^|m|`` factor is the regularity condition at the polar axis; no
    edge condition is needed because ``B.n = 0`` holds for any lambda.  Returns
    the two derivatives that the field needs -- lambda itself never appears.
    """
    def dlam(x):
        rho, chi, zeta = x[0], x[1], x[2]
        d_chi = 0.0
        d_zeta = 0.0
        for m, n, amp in modes:
            phase = 2 * jnp.pi * (m * chi - n * zeta)
            radial = amp * rho ** abs(m) if m != 0 else amp
            d_chi = d_chi + radial * 2 * jnp.pi * m * jnp.cos(phase)
            d_zeta = d_zeta - radial * 2 * jnp.pi * n * jnp.cos(phase)
        return d_chi, d_zeta

    return dlam


def analytic_helicity(iota0, iota1, iota_exp, flux_exp):
    """Eq. (2): helicity of the raw (un-normalised) field, natural gauge.

    Linear in the shear ``iota1 - iota0`` and exactly zero without it.
    """
    q, e = flux_exp, iota_exp
    return (iota1 - iota0) * e / ((q + 1) * (q + e + 1) * (2 * q + e + 2))


# ---------------------------------------------------------------------------
# Pure-geometry surface averages: a, b, c, V'.  No solve, no assembly.
# ---------------------------------------------------------------------------

def metric_coefficients(seq, rhos, n_ang):
    """Return ``a, b, c, Vp`` on ``rhos`` -- eq. (4) of the module docstring.

    chi and zeta are periodic on [0, 1), so a uniform grid mean IS the exact
    surface integral for smooth integrands (trapezoid on a periodic domain).
    """
    DF = jax.jacfwd(seq.map)

    def coeffs_at(x):
        dF = DF(x)
        g = dF.T @ dF
        jac = jnp.linalg.det(dF)
        return jnp.array([g[1, 1] / jac, g[1, 2] / jac, g[2, 2] / jac, jac])

    ang = (jnp.arange(n_ang) + 0.5) / n_ang
    cc, zz = jnp.meshgrid(ang, ang, indexing='ij')

    def surface(rho):
        pts = jnp.stack([jnp.full(cc.size, rho), cc.ravel(), zz.ravel()],
                        axis=-1)
        return jnp.mean(jax.vmap(coeffs_at)(pts), axis=0)

    out = np.asarray(jax.vmap(surface)(jnp.asarray(rhos)))
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def surface_points(rho, ang):
    return jnp.asarray([[rho, c, z] for c in ang for z in ang])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="cylinder")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--iota", default="0.4,0.9",
                    help="iota on axis, iota at edge; equal values => zero "
                         "shear => zero helicity by eq. (2)")
    ap.add_argument("--iota-exp", type=float, default=2.0)
    ap.add_argument("--flux-exp", type=float, default=1.0)
    ap.add_argument("--lam", default="",
                    help="VMEC stream function as 'm,n,amp;m,n,amp;...'. "
                         "Empty (default) is the straight-field-line case. "
                         "Must leave the helicity unchanged.")
    ap.add_argument("--n-rho", type=int, default=33)
    ap.add_argument("--n-ang", type=int, default=8)
    ap.add_argument("--axis-band", type=float, default=0.15,
                    help="rho below this counts as the polar-gluing band")
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    iota0, iota1 = (float(v) for v in cli.iota.split(","))
    iota, dPhi = make_profiles(iota0, iota1, cli.iota_exp, cli.flux_exp)
    lam_modes = parse_lambda(cli.lam)
    dlam = make_lambda(lam_modes)

    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}  "
          f"iota {iota0} -> {iota1} (exp {cli.iota_exp})  "
          f"dPhi = rho^{cli.flux_exp}", flush=True)
    print(f"[setup] lambda modes (m, n, amp): "
          f"{lam_modes if lam_modes else 'none -- straight-field-line'}",
          flush=True)

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    ops = seq.assemble_all_sparse(include_preconditioners=False)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] operators + nullspaces {time.perf_counter() - t0:.1f}s",
          flush=True)

    rhos = np.linspace(0.02, 0.98, cli.n_rho)
    ang = (np.arange(cli.n_ang) + 0.5) / cli.n_ang
    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "iota": [iota0, iota1], "iota_exp": cli.iota_exp,
               "flux_exp": cli.flux_exp, "rho": rhos.tolist()}

    # --- pure geometry: a, b, c, V' ----------------------------------------
    t_g = time.perf_counter()
    a_c, b_c, c_c, Vp = metric_coefficients(seq, rhos, cli.n_ang)
    iota_star = -b_c / a_c
    print(f"\n[metric] surface averages in {time.perf_counter() - t_g:.1f}s")
    print("[metric]   rho        a=<g_cc/J>    b=<g_cz/J>    c=<g_zz/J>"
          "        V'=<J>    iota*=-b/a")
    for i in range(0, cli.n_rho, max(1, cli.n_rho // 8)):
        print(f"         {rhos[i]:5.3f}  {a_c[i]:13.5e} {b_c[i]:13.5e} "
              f"{c_c[i]:13.5e} {Vp[i]:13.5e} {iota_star[i]:+11.4f}")
    print(f"[metric] |b|/a  max = {np.abs(b_c / a_c).max():.4e}   "
          f"-- exactly 0 on the cylinder; the 3-D term")
    results.update(a=a_c.tolist(), b=b_c.tolist(), c=c_c.tolist(),
                   Vprime=Vp.tolist(), iota_star=iota_star.tolist())

    # --- the IC ------------------------------------------------------------
    # L2 route: histopolation is unavailable here (see the module docstring),
    # so the metric enters through M_2 and gate 1 below measures what that costs.
    #
    # CONVENTION TRAP.  The DoFs that come out of M_2^{-1} are the PRIMAL
    # reference components omega, i.e. ``B_phys = DF omega / J`` -- that is what
    # DiscreteFunction evaluates and what Pushforward(., 2) consumes.  But
    # ``load(frame='ref')`` pairs its argument straight against the basis, and
    # M_2 itself carries a g/J weight (M2_ij = int Lambda_i^T g Lambda_j / J),
    # so ``load(frame='ref')`` wants ``g omega / J``, NOT omega.  Handing it
    # omega silently builds a different field, off by a metric factor that is
    # component-dependent and rho-dependent even on a cylinder.
    #
    # So push omega forward explicitly and use the physical frame, which does
    # ``DF^T (DF omega / J) = g omega / J`` internally and is unambiguous.
    DF_map = jax.jacfwd(seq.map)

    def omega_ref(x):
        """Primal reference 2-form components -- the GVEC B_hat."""
        r = x[0]
        f = dPhi(r)
        d_chi, d_zeta = dlam(x)
        return jnp.array([0.0, f * (iota(r) - d_zeta), f * (1.0 + d_chi)])

    def B_phys(x):
        dF = DF_map(x)
        return dF @ omega_ref(x) / jnp.linalg.det(dF)

    t1 = time.perf_counter()
    B_raw = seq.apply_inverse_mass_matrix(
        seq.load(B_phys, 2, dirichlet=True), 2, dirichlet=True)
    B_norm = float(seq.l2_norm(B_raw, 2, dirichlet=True))
    B = B_raw / B_norm
    print(f"\n[ic] L2-projected in {time.perf_counter() - t1:.1f}s   "
          f"||B||_M before normalisation = {B_norm:.6e}", flush=True)
    results["B_norm_raw"] = B_norm

    # --- gate 1: is B^rho actually zero? -----------------------------------
    B_h = DiscreteFunction(B, seq.basis_2, seq.e2_dbc)
    comp_by_rho, comp_mean = [], []
    for r in rhos:
        vals = np.asarray(jax.vmap(B_h)(surface_points(r, ang)))
        comp_by_rho.append(np.abs(vals).max(axis=0))
        comp_mean.append(vals.mean(axis=0))
    comp_by_rho = np.array(comp_by_rho)                 # (n_rho, 3) max|.|
    comp_mean = np.array(comp_mean)                     # (n_rho, 3) surface mean
    scale = comp_by_rho[:, 2].max()
    axis_mask = rhos < cli.axis_band
    rho_axis = comp_by_rho[axis_mask, 0].max() / scale
    rho_bulk = comp_by_rho[~axis_mask, 0].max() / scale
    print(f"[gate] max|B^rho|/max|B^zeta|   axis band (rho<{cli.axis_band}): "
          f"{rho_axis:.3e}    bulk: {rho_bulk:.3e}")
    print("[gate]   -- this is the cost of the L2 route; if the bulk number is "
          "large, histopolation is worth unlocking")
    results["Brho_rel_axis"] = float(rho_axis)
    results["Brho_rel_bulk"] = float(rho_bulk)

    # --- gate 2: divergence, both routes -----------------------------------
    div_inc = seq.apply_incidence_matrix(B, 2, dirichlet_in=True,
                                         dirichlet_out=True)
    div_str = seq.apply_strong_div(B)
    n_inc = float(seq.l2_norm(div_inc, 3, dirichlet=True))
    n_str = float(seq.l2_norm(div_str, 3, dirichlet=True))
    print(f"[gate] ||div B||_L2   incidence G_2: {n_inc:.3e}   "
          f"strong_div (M-projected): {n_str:.3e}   (||B||_L2 = 1)")
    results["div_incidence"] = n_inc
    results["div_strong"] = n_str

    # --- gate 3: is the Leray projection a no-op? --------------------------
    t2 = time.perf_counter()
    B_leray, _ = seq.apply_leray_projection(B, k=2)
    leray_rel = float(seq.l2_norm(B_leray - B, 2, dirichlet=True))
    print(f"[gate] ||P_Leray B - B||_L2 / ||B||_L2 = {leray_rel:.3e}   "
          f"({time.perf_counter() - t2:.1f}s)")
    results["leray_change"] = leray_rel

    # --- gate 4: does the discrete field carry the iota we asked for? ------
    # The transform is iota = X'/Phi' = <B^chi>/<B^zeta>, using SURFACE MEANS,
    # not pointwise values: lambda contributes only exact angular derivatives,
    # which average to zero.  So this ratio is the true iota with or without
    # lambda, whereas a pointwise ratio would not be.
    #
    # dr^dzeta is -dzeta^dr, so this may come back as -iota.  That is a basis
    # convention, and this table is what settles it.
    ratio = comp_mean[:, 1] / comp_mean[:, 2]
    target = iota0 + (iota1 - iota0) * rhos ** cli.iota_exp
    print("\n[iota]   rho    target   <B^chi>/<B^zeta>   ratio/target")
    for i in range(0, cli.n_rho, max(1, cli.n_rho // 8)):
        print(f"       {rhos[i]:5.3f}   {target[i]:6.4f}   {ratio[i]:14.6f}"
              f"   {ratio[i] / target[i]:+8.4f}")
    print("[iota]   -- lambda must NOT move this column; that is the claim "
          "that lambda preserves the fluxes")
    results["iota_target"] = target.tolist()
    results["iota_measured_ratio"] = ratio.tolist()

    # --- helicity: eq. (2) against the code's own gauge --------------------
    # Helicity is quadratic in B and B was normalised, so scale eq. (2) the
    # same way before comparing.
    H_analytic = analytic_helicity(iota0, iota1, cli.iota_exp,
                                   cli.flux_exp) / B_norm ** 2
    t_h = time.perf_counter()
    H_code, _ = compute_helicity(B, seq, jnp.zeros(seq.n1_dbc))
    H_code = float(H_code)
    print(f"\n[helicity] eq.(2) natural gauge : {H_analytic:+.6e}")
    print(f"[helicity] compute_helicity      : {H_code:+.6e}   "
          f"({time.perf_counter() - t_h:.1f}s)")
    print(f"[helicity] difference (harmonic gauge contribution): "
          f"{H_code - H_analytic:+.6e}")
    print(f"[helicity]   -- eq.(2) is linear in the shear {iota1 - iota0:+.3f} "
          f"and is exactly 0 without it")
    print("[helicity]   -- and it must not move when lambda is turned on; "
          "lambda is a pure gauge transformation")
    results["H_analytic"] = H_analytic
    results["H_code"] = H_code

    # --- gate 5: does eq. (4) reproduce the energy of the built field? ------
    # Eq. (4) is the lambda = 0 energy.  With lambda on it is EXPECTED to
    # disagree, and the disagreement is exactly what lambda adds -- unlike the
    # fluxes and the helicity, the energy is not blind to lambda.
    f_z = rhos ** cli.flux_exp / B_norm
    u_pred = 0.5 * f_z ** 2 * (a_c * target ** 2 + 2 * b_c * target + c_c) / Vp
    B_phys = Pushforward(B_h, seq.map, 2)

    def bsq_at(x):
        v = B_phys(x)
        return v @ v

    u_meas = np.array([
        0.5 * float(jnp.mean(jax.vmap(bsq_at)(surface_points(r, ang))))
        for r in rhos])
    u_rel = float(np.linalg.norm(u_meas - u_pred) / np.linalg.norm(u_meas))
    print(f"\n[energy] eq.(4) vs measured <B^2>/2 : relative error {u_rel:.4e}")
    print("[energy]   rho     predicted      measured")
    for i in range(0, cli.n_rho, max(1, cli.n_rho // 8)):
        print(f"         {rhos[i]:5.3f}  {u_pred[i]:13.6e} {u_meas[i]:13.6e}")
    results["u_pred"] = u_pred.tolist()
    results["u_meas"] = u_meas.tolist()
    results["u_rel_err"] = u_rel

    # --- the force and the pressure ----------------------------------------
    # A large force here is EXPECTED and fine: this is a relaxation starting
    # point, not an equilibrium.  The number is reported so runs are comparable.
    t3 = time.perf_counter()
    F, p_dof, _, _, JxH = compute_force(B, seq)
    F_rel = float(seq.l2_norm(F, 2, dirichlet=True))
    print(f"\n[force] ||F||_L2 / ||B||_L2 = {F_rel:.4e}   "
          f"({time.perf_counter() - t3:.1f}s)")
    results["force_rel"] = F_rel

    # Pfirsch-Schlueter measure: (JxB).grad rho / |grad rho|^2 must be a flux
    # function for a nested-surface equilibrium to exist.  grad rho is row 0 of
    # DF^{-1}.  The spread over each surface is the drive with nowhere to go.
    JxH_phys = Pushforward(DiscreteFunction(JxH, seq.basis_2, seq.e2_dbc),
                           seq.map, 2)
    DF = jax.jacfwd(seq.map)

    def radial_force_at(x):
        grad_rho = jnp.linalg.inv(DF(x))[0, :]
        return (JxH_phys(x) @ grad_rho) / (grad_rho @ grad_rho)

    ps = np.array([np.asarray(jax.vmap(radial_force_at)(
        surface_points(r, ang))) for r in rhos])
    ps_mean, ps_spread = ps.mean(axis=1), ps.std(axis=1)
    ps_ratio = ps_spread / np.abs(ps_mean)
    print(f"[ps] Pfirsch-Schlueter spread std/|mean| :  "
          f"median {np.median(ps_ratio):.3e}   max {ps_ratio.max():.3e}")
    print("[ps]     rho    surface mean      std        std/|mean|")
    for i in range(0, cli.n_rho, max(1, cli.n_rho // 8)):
        print(f"       {rhos[i]:5.3f}  {ps_mean[i]:13.5e} {ps_spread[i]:12.4e}"
              f"  {ps_ratio[i]:11.4e}")
    results["ps_mean"] = ps_mean.tolist()
    results["ps_spread"] = ps_spread.tolist()

    # --- measured p'(rho) against the metric prediction -du/drho -----------
    p_h = Pushforward(DiscreteFunction(p_dof, seq.basis_3, seq.e3_dbc),
                      seq.map, 3)
    p_prof = np.array([
        float(jnp.mean(jax.vmap(lambda x: p_h(x)[0])(surface_points(r, ang))))
        for r in rhos])
    dp_meas = np.gradient(p_prof, rhos)
    du_pred = -np.gradient(u_pred, rhos)
    # One scalar only: how much of the measured gradient the energy term
    # explains.  The remainder is the tension/curvature contribution, which the
    # geometry decides and which a cylinder formula would get wrong.
    k_fit = float(du_pred @ dp_meas / (du_pred @ du_pred))
    resid = float(np.linalg.norm(dp_meas - k_fit * du_pred)
                  / np.linalg.norm(dp_meas))
    print(f"\n[press] measured dp/drho vs -du/drho:  slope {k_fit:.4f}   "
          f"unexplained {resid:.4e}")
    print("[press]   rho      <p>        dp/drho meas   -du/drho pred")
    for i in range(0, cli.n_rho, max(1, cli.n_rho // 10)):
        print(f"        {rhos[i]:5.3f}  {p_prof[i]:12.5e}  {dp_meas[i]:13.5e}"
              f"  {du_pred[i]:14.5e}")
    results["p_measured"] = p_prof.tolist()
    results["dp_measured"] = dp_meas.tolist()
    results["du_pred"] = du_pred.tolist()
    results["energy_slope"] = k_fit
    results["energy_unexplained"] = resid

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
