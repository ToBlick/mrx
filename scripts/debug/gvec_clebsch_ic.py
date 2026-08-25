"""Build the IC from GVEC's Clebsch ingredients instead of interpolating B.

``data/gvec_nfp3_hegna_80cubed_clebsch.h5`` carries every ingredient by name:

    clebsch/dPhi_dr   clebsch/dchi_dr      the two radial profiles
    clebsch/dLA_dt    clebsch/dLA_dz       lambda's angular derivatives
    pressure                               and a finite-beta pressure profile

and the identities below were VERIFIED against that file's own B, in its own
normalised [0,1] coordinates, with no free factor -- no 2 pi, no sign flip:

    sqrt(g) B^rho   = 0                          (measured 3.8e-16)
    sqrt(g) B^theta = dchi_dr - dPhi_dr * dLA_dz (ratio 1.00000000, std 2.9e-13)
    sqrt(g) B^zeta  = dPhi_dr * (1 + dLA_dt)     (ratio 1.00000000, std 1.7e-16)

Those three ARE the primal reference 2-form components of ``logical_profile_ic``:
``B_hat = (0, dPhi(iota - lam_zeta), dPhi(1 + lam_chi))`` with
``iota = dchi_dr / dPhi_dr``.  GVEC's representation and ours are the same
object, so the equilibrium field can be rebuilt from four scalars rather than
resampled as a vector.


WHY THIS BEATS INTERPOLATING B
------------------------------
``interpolate_B`` fits the Cartesian vector field and then repairs it: the
current path measures ``div B`` after interpolation and applies P_Leray to clean
it (``relax_from_nfs.py``).  Every interpolation error there lands on div B,
on B.n, and on the fluxes.  Going through the Clebsch scalars instead:

  * div B = 0 EXACTLY, for any lambda -- the mixed partials cancel
    (d_chi(B^chi) + d_zeta(B^zeta) = dPhi(-lam_zetachi + lam_chizeta) = 0).
  * B.n = 0 and nested surfaces EXACTLY, because B^rho is set to zero, not
    fitted.
  * The FLUXES, iota and the HELICITY are exact even if lambda is interpolated
    BADLY.  lambda enters only through exact angular derivatives, which average
    to zero over a surface -- so error in lambda can only redistribute field
    WITHIN a surface, never across one, and never touches an ideal invariant.
  * lambda is a SCALAR in logical coordinates, so it is immune to the zeta
    quasi-periodicity seam that bites Cartesian Bx, By (those rotate by
    R_z(-2pi/nfp) per field period and need de-rotating first; see
    docs/w7x_vacuum_bfield_handoff.md).  No de-rotation, no Gibbs seam.

That last pair is the real argument: the structural guarantees do not depend on
the quality of the fit.


WHAT p HAS TO DO WITH IT
------------------------
The causality runs the other way from how it is usually asked.  In VMEC/GVEC,
``p(s)`` and ``iota(s)`` (or the current profile) plus the boundary shape are
INPUTS; ``lambda`` and the surface shapes R, Z are OUTPUTS of the energy
minimisation.  There is no formula ``p = f(dPhi, iota, lambda)``.

But once the MAP is frozen -- and ours is, it is GVEC's own R, Z -- B is
determined completely by (dPhi_dr, iota, lambda); there is nothing else.  So p
IS a function of the three, via force balance:

    p'(rho) = < (J x B).grad rho > / < |grad rho|^2 >

not in closed form, but as a curl and a cross product, which is what
``compute_force`` does.  Physically lambda is where the pressure lives: the
solenoidality of J with ``J_perp = (B x grad p)/B^2`` forces a parallel current
obeying ``B.grad(J_par/B) = -div J_perp``, linear in p'.  That is the
Pfirsch-Schlueter current, and lambda is how the field carries it.  At low beta
that splits lambda into a geometry-and-iota part plus a piece linear in p'.

Which makes this script an end-to-end VALIDATION, not just an IC builder:
rebuild B from the four scalars, run it through our own force operator, and
compare the recovered p(rho) against GVEC's stored ``pressure``.  Agreement
validates map, representation and force operator together -- a much sharper
test than a projection-error check.  Disagreement localises which of the three
is wrong.

CAVEAT, measured not assumed: ``build_gvec_map`` may MIRROR the raw GVEC data to
keep det DF > 0 (see gvec_geometry.py).  A mirror flips orientation, so the sign
of the reconstructed iota relative to the file's ``dchi_dr/dPhi_dr`` is not
known in advance.  Both are printed side by side; one run settles it.

    python scripts/debug/gvec_clebsch_ic.py --geometry hegna --ns 12,24,12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from jax.scipy.interpolate import RegularGridInterpolator  # noqa: E402
from mrx.differential_forms import DiscreteFunction, Pushforward  # noqa: E402
from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.relaxation import compute_force, compute_helicity  # noqa: E402
from gvec_geometry import GVEC_GEOMETRIES  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def load_clebsch(path):
    """Return radial profiles, lambda-derivative interpolants and p(rho).

    The hegna export samples all three logical axes CLOSED on [0, 1] (80 points,
    step 1/79), the angular endpoints duplicating the start -- so the grid is
    directly usable by RegularGridInterpolator with no wrap padding.  That is
    the opposite of the quasr files and is why the axes are read from the file
    rather than assumed (see gvec_geometry.py).
    """
    with h5py.File(path, "r") as h:
        nr = int(h.attrs["n_rho"])
        nt = int(h.attrs["n_theta"])
        nz = int(h.attrs["n_zeta"])
        shape = (nr, nt, nz)
        c = h["clebsch"]
        dPhi = np.asarray(c["dPhi_dr"]).reshape(shape)
        dchi = np.asarray(c["dchi_dr"]).reshape(shape)
        dLt = np.asarray(c["dLA_dt"]).reshape(shape)
        dLz = np.asarray(c["dLA_dz"]).reshape(shape)
        pres = np.asarray(h["pressure"]).reshape(shape)
        ep = np.asarray(h["eval_points"])
        nfp = int(h.attrs["nfp"])

    axes = [np.unique(ep[:, i]) for i in range(3)]
    if not all(len(a) == n for a, n in zip(axes, shape)):
        raise RuntimeError(f"eval_points axes {[len(a) for a in axes]} do not "
                           f"match declared shape {shape}")

    # dPhi_dr and dchi_dr are flux functions; the surface mean is the profile
    # and the surface spread says how well that holds in the data itself.
    prof_dPhi = dPhi.mean(axis=(1, 2))
    prof_dchi = dchi.mean(axis=(1, 2))
    prof_p = pres.mean(axis=(1, 2))
    spread = float(np.nanmax(np.abs(dchi / dPhi - (prof_dchi / prof_dPhi)[:, None, None])
                             [nr // 4:3 * nr // 4]))

    grid = tuple(jnp.asarray(a) for a in axes)
    rgi_Lt = RegularGridInterpolator(grid, jnp.asarray(dLt), method='linear')
    rgi_Lz = RegularGridInterpolator(grid, jnp.asarray(dLz), method='linear')
    return dict(nfp=nfp, rho=axes[0], dPhi=prof_dPhi, dchi=prof_dchi,
                p=prof_p, iota_spread=spread, rgi_Lt=rgi_Lt, rgi_Lz=rgi_Lz)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="hegna")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--no-lambda", action="store_true",
                    help="zero lambda out -- the straight-field-line IC. The "
                         "fluxes, iota and helicity must NOT move; the force "
                         "and the pressure must.")
    ap.add_argument("--n-rho", type=int, default=33)
    ap.add_argument("--n-ang", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    h5 = GVEC_GEOMETRIES[cli.geometry]
    cb = load_clebsch(h5)
    print(f"[data] {h5}  nfp={cb['nfp']}  n_rho={len(cb['rho'])}")
    print(f"[data] iota = dchi_dr/dPhi_dr : "
          f"{cb['dchi'][1] / cb['dPhi'][1]:+.5f} (axis) -> "
          f"{cb['dchi'][-1] / cb['dPhi'][-1]:+.5f} (edge);  max angular "
          f"departure from a flux function in the mid-radius = "
          f"{cb['iota_spread']:.3e}")
    print(f"[data] pressure {cb['p'][0]:.5e} -> {cb['p'][-1]:.5e} Pa"
          f"   (lambda {'ZEROED' if cli.no_lambda else 'on'})")

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    ops = seq.assemble_all_sparse(include_preconditioners=False)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] operators + nullspaces {time.perf_counter() - t0:.1f}s",
          flush=True)

    rho_g = jnp.asarray(cb["rho"])
    dPhi_g = jnp.asarray(cb["dPhi"])
    dchi_g = jnp.asarray(cb["dchi"])
    rgi_Lt, rgi_Lz = cb["rgi_Lt"], cb["rgi_Lz"]
    use_lam = not cli.no_lambda

    # Primal reference 2-form components, straight off the verified identities.
    # See logical_profile_ic.py for why this is pushed forward rather than
    # handed to load(frame='ref'): that entry point wants g omega / J, not omega.
    DF_map = jax.jacfwd(seq.map)

    def omega_ref(x):
        r = jnp.clip(x[0], rho_g[0], rho_g[-1])
        f_phi = jnp.interp(r, rho_g, dPhi_g)
        f_chi = jnp.interp(r, rho_g, dchi_g)
        pt = jnp.array([[r, x[1] % 1.0, x[2] % 1.0]])
        lam_t = rgi_Lt(pt)[0] * use_lam
        lam_z = rgi_Lz(pt)[0] * use_lam
        return jnp.array([0.0, f_chi - f_phi * lam_z, f_phi * (1.0 + lam_t)])

    def B_phys(x):
        dF = DF_map(x)
        return dF @ omega_ref(x) / jnp.linalg.det(dF)

    t1 = time.perf_counter()
    B_raw = seq.apply_inverse_mass_matrix(
        seq.load(B_phys, 2, dirichlet=True), 2, dirichlet=True)
    B_norm = float(seq.l2_norm(B_raw, 2, dirichlet=True))
    B = B_raw / B_norm
    print(f"[ic] built in {time.perf_counter() - t1:.1f}s   "
          f"||B||_M = {B_norm:.6e}", flush=True)

    rhos = np.linspace(0.05, 0.95, cli.n_rho)
    ang = (np.arange(cli.n_ang) + 0.5) / cli.n_ang

    def spts(r):
        return jnp.asarray([[r, c, z] for c in ang for z in ang])

    results = {"geometry": cli.geometry, "h5": h5, "ns": list(ns), "p": cli.p,
               "lambda_on": bool(use_lam), "B_norm": B_norm,
               "rho": rhos.tolist()}

    # --- structure gates ---------------------------------------------------
    B_h = DiscreteFunction(B, seq.basis_2, seq.e2_dbc)
    comp_max, comp_mean = [], []
    for r in rhos:
        vals = np.asarray(jax.vmap(B_h)(spts(r)))
        comp_max.append(np.abs(vals).max(axis=0))
        comp_mean.append(vals.mean(axis=0))
    comp_max, comp_mean = np.array(comp_max), np.array(comp_mean)
    brho = comp_max[:, 0].max() / comp_max[:, 2].max()

    div_str = float(seq.l2_norm(seq.apply_strong_div(B), 3, dirichlet=True))
    B_leray, _ = seq.apply_leray_projection(B, k=2)
    leray = float(seq.l2_norm(B_leray - B, 2, dirichlet=True))
    print(f"\n[gate] max|B^rho|/max|B^zeta| = {brho:.3e}")
    print(f"[gate] ||div B||_L2 = {div_str:.3e}   "
          f"||P_Leray B - B||_L2 = {leray:.3e}   (||B||_L2 = 1)")
    results.update(Brho_rel=float(brho), div_strong=div_str,
                   leray_change=leray)

    # --- iota: ours vs the file's ------------------------------------------
    # Sign is NOT asserted: build_gvec_map may mirror the raw data to keep
    # det DF > 0, which flips orientation.  This table settles it.
    iota_ours = comp_mean[:, 1] / comp_mean[:, 2]
    iota_file = np.interp(rhos, cb["rho"], cb["dchi"] / cb["dPhi"])
    print("\n[iota]   rho    file dchi/dPhi   reconstructed   ratio")
    for i in range(0, cli.n_rho, max(1, cli.n_rho // 8)):
        print(f"       {rhos[i]:5.3f}   {iota_file[i]:+13.6f}   "
              f"{iota_ours[i]:+13.6f}   {iota_ours[i] / iota_file[i]:+7.4f}")
    results["iota_file"] = iota_file.tolist()
    results["iota_ours"] = iota_ours.tolist()

    # --- helicity ----------------------------------------------------------
    H, _ = compute_helicity(B, seq, jnp.zeros(seq.n1_dbc))
    print(f"\n[helicity] compute_helicity = {float(H):+.6e}   "
          f"-- must be unchanged by --no-lambda")
    results["H"] = float(H)

    # --- the end-to-end test: our p against GVEC's -------------------------
    t3 = time.perf_counter()
    F, p_dof, _, _, _ = compute_force(B, seq)
    F_rel = float(seq.l2_norm(F, 2, dirichlet=True))
    print(f"\n[force] ||F||_L2 / ||B||_L2 = {F_rel:.4e}   "
          f"({time.perf_counter() - t3:.1f}s)")
    results["force_rel"] = F_rel

    p_h = Pushforward(DiscreteFunction(p_dof, seq.basis_3, seq.e3_dbc),
                      seq.map, 3)
    p_ours = np.array([float(jnp.mean(jax.vmap(lambda x: p_h(x)[0])(spts(r))))
                       for r in rhos])
    p_file = np.interp(rhos, cb["rho"], cb["p"])
    # B was normalised, so p comes out in the same arbitrary scale: compare
    # SHAPES, with one overall constant fitted and the edge offset removed.
    a_ours = p_ours - p_ours[-1]
    a_file = p_file - p_file[-1]
    k = float(a_ours @ a_file / (a_ours @ a_ours)) if a_ours @ a_ours > 0 else 0.0
    resid = (float(np.linalg.norm(a_file - k * a_ours) / np.linalg.norm(a_file))
             if np.linalg.norm(a_file) > 0 else float('nan'))
    print(f"[press] recovered p vs GVEC pressure: scale {k:.6e}   "
          f"shape residual {resid:.4e}")
    print("[press]   rho    GVEC p (Pa)    ours (scaled)")
    for i in range(0, cli.n_rho, max(1, cli.n_rho // 10)):
        print(f"        {rhos[i]:5.3f}  {a_file[i]:13.5e}  {k * a_ours[i]:13.5e}")
    results.update(p_file=p_file.tolist(), p_ours=p_ours.tolist(),
                   p_scale=k, p_residual=resid)

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
