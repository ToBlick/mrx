r"""Sec. 3.4, App. D and Figs. 6-8: quasi-axisymmetric shape optimization of the Landreman-Paul (LP) vacuum field.

    MRX_DTYPE=float64 python scripts/paper_scripts/ad_recovery.py --stage constrained|trace|baseline [--records DIR] [--tag T] ...

Records in <records>/shape_optimization [--records, MRX_RECORDS or outputs]: constrained -> qa_constrained<tag>.json/.npz,
qa_recover<tag>.json/.npz, qa_guard<tag>.json; trace (reads qa_recover<tag>.npz) -> qa_trace<tag>[_remesh].npz/.json;
baseline -> qa_baseline<tag>.json. GPU, float64; the launchers are scripts/paper_scripts/runs/ad_{constrained,trace,baseline}.sh.

The map is LP's VMEC map interpolated on the (n_r, n_theta, n_zeta) splines, p = 3; every coefficient is a variable
(:class:`mrx.shape_ad.BoundaryShape` ``free="all"``: the boundary ring's change extended harmonically, plus a change
of every inner ring), beta = --beta-scale x, and the map is rescaled after every change to LP's volume and to the
aspect ratio --aspect (:meth:`mrx.shape_ad.BoundaryShape.map_coefficients`).

Stages (--stage, comma-separated, after one setup):

- constrained: min <Q_QA^2>_{r >= h_r} (:func:`mrx.shape_ad.quasisymmetry_residual`, h_r the end of the first radial
  knot span) subject to <iotabar>_s = LP's own value (:func:`mrx.shape_ad.mean_iota`) and P <= --p-max
  (:func:`mrx.shape_ad.normal_field_fraction`), by the augmented Lagrangian of App. D over L-BFGS-B on
  ``jax.value_and_grad``. The start is LP, or LP with a random smooth normal boundary displacement of --perturb-mm
  RMS (:func:`perturbation`), rejected by the fold guard (:func:`start_guard`); a trial point whose map folds is
  infeasible. --stop-qa stops at the baseline, --reference-end measures the end against another run's end.
- trace: Poincare sections of LP and of the end field of a constrained run (with --remesh the end boundary on LP's
  interior), in the archive layout of scripts/poincare_trace.py, which scripts/poincare_plot.py renders (Fig. 8).
- baseline: the criterion on LP's interpolated map at --ns, the runs' unit F_LP (the grey lines of Fig. 6).
"""
import argparse
import copy
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

import mrx
from mrx.geometry import build_sequence, grad_1d
from mrx.gvec import build_gvec_map
from mrx.nullspace import compute_nullspaces
from mrx.poincare import poincare
from mrx.shape_ad import (BoundaryShape, _cylindrical_derivatives, _tables, aspect_ratio, cylindrical_geometry,
                          edge_quasisymmetry_residual, flux_ratio_iota, flux_seed, mean_iota, normal_field_fraction,
                          quasisymmetry_residual, section_moments, vacuum_two_form, with_geometry)
from mrx.spline_bases import basis_derivative_table, basis_table

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/paper_scripts/<this file>
GEOMETRY = os.path.join(REPO, "data", "wout_LandremanPaul2021_QA_lowres.nc")
#: the edge terms (F_edge, the edge iota, the fold rule's edge points) are taken at r = 1 - EPS
EPS = 1e-6
#: the scale of the multiplier term of the mean iota: c = (<iotabar>_s - target) / C_UNIT
C_UNIT = 1e-3


class Problem(eqx.Module):
    shape: BoundaryShape           # every ring free, the boundary change extended harmonically, V and A held
    seed: jnp.ndarray              # the flux 2-form the harmonic form is solved from
    orient: jnp.ndarray            # +-1: the sign of the logical iota
    iota_target: jnp.ndarray       # LP's <iotabar>_s on this mesh
    r_min: float = eqx.field(static=True)
    beta_scale: float = eqx.field(static=True)


def change_of(pb, x):
    """``beta``, the change of every ring ``(2, n_r, n_theta, n_zeta)``: ``beta_scale x``."""
    return pb.beta_scale * jnp.reshape(x, (2,) + tuple(pb.shape.raw_R.shape))


def first_span(seq):
    """The end of the first radial knot span, ``1 / (n_r - p)`` for the clamped uniform radial splines: the polar
    patch, h_r."""
    T = np.asarray(seq.basis_0.Λ[0].T)
    return float(np.min(T[T > 0.0]))


def edge_jacobian(seq, raw_R, raw_Z, nfp, sign):
    """``det DF`` of the map of the raw coefficients on the edge rule, ``r = 1 - EPS`` at the angular quadrature
    points, where :func:`mrx.shape_ad.edge_quasisymmetry_residual` weights with it."""
    (R, _), d1, _ = _cylindrical_derivatives(raw_R, raw_Z, _tables(seq, [1.0 - EPS]))
    dR, dZ = d1[:, 0], d1[:, 1]
    return sign * (2.0 * np.pi / nfp) * R * (dR[:, 1] * dZ[:, 0] - dR[:, 0] * dZ[:, 1])


def terms(x, seq, pb):
    """Every term of the problem at the variables ``x``: the criterion F_qs, the mean iota, P, and the diagnostics
    (the edge terms, the scalings, A, V, min det DF)."""
    R, Z, mu, scale = pb.shape.map_coefficients(seq, change_of(pb, x))
    sq = with_geometry(seq, cylindrical_geometry(seq, R, Z, pb.shape.nfp, pb.shape.sign))
    h, info = vacuum_two_form(sq, pb.seed)
    F_qs, F_par = quasisymmetry_residual(sq, h, R, Z, pb.shape.nfp, pb.shape.sign, pb.r_min)
    V, _, S = section_moments(seq, R, Z, pb.shape.nfp, pb.shape.sign)
    J = sq.jacobian_j
    return dict(F_qs=F_qs, F_par=F_par, F_edge=edge_quasisymmetry_residual(sq, h, R, Z, pb.shape.nfp, pb.shape.sign, EPS),
                mean_iota=pb.orient * mean_iota(sq, h),
                iota_edge=pb.orient * flux_ratio_iota(sq, h, jnp.asarray([1.0 - EPS]))[0][0],
                P=normal_field_fraction(sq, h), info=info, mu=mu, scale=scale,
                aspect=aspect_ratio(V, S, pb.shape.nfp), volume=pb.shape.nfp * V, jmin=jnp.min(J),
                jrel=jnp.min(J) / jnp.mean(J), jmin_edge=jnp.min(edge_jacobian(seq, R, Z, pb.shape.nfp, pb.shape.sign)))


forward = eqx.filter_jit(terms)


@eqx.filter_jit
def min_det_df(x, seq, pb):
    """``min det DF`` of the map at ``x`` on the volume quadrature points and on the edge rule, without a solve: it
    folds where not positive."""
    R, Z, _, _ = pb.shape.map_coefficients(seq, change_of(pb, x))
    return jnp.minimum(jnp.min(cylindrical_geometry(seq, R, Z, pb.shape.nfp, pb.shape.sign).jacobian_j),
                       jnp.min(edge_jacobian(seq, R, Z, pb.shape.nfp, pb.shape.sign)))


def lagrangian(x, seq, pb, al):
    """The augmented Lagrangian of App. D, ``f + lambda_i c + rho c^2 / 2 + (max(0, lambda_P + rho g)^2 -
    lambda_P^2) / (2 rho)`` (PHR for the inequality) in the scaled ``f = F_QS / f_unit``, ``c = (<iotabar>_s -
    target) / C_UNIT``, ``g = (P - eps) / eps`` (``al``: the multipliers, rho and the scales, arrays)."""
    aux = terms(x, seq, pb)
    f = aux["F_qs"] / al["f_unit"]
    c = (aux["mean_iota"] - pb.iota_target) / C_UNIT
    g = (aux["P"] - al["eps"]) / al["eps"]
    rho = al["rho"]
    return (f + al["lam_i"] * c + 0.5 * rho * c ** 2
            + (jnp.maximum(0.0, al["lam_P"] + rho * g) ** 2 - al["lam_P"] ** 2) / (2.0 * rho)), aux


value_and_grad = eqx.filter_jit(jax.value_and_grad(lagrangian, has_aux=True))


# ---------------------------------------------------------------------------------------------------- the start
def collocate(seq, values):
    """Coefficients on the angular splines of ``seq.basis_0`` interpolating ``values`` ``(n_theta, n_zeta)`` at the
    Greville points."""
    ct, cz = np.asarray(seq.greville[1].coll), np.asarray(seq.greville[2].coll)
    return np.linalg.solve(ct, np.linalg.solve(cz, np.asarray(values).T).T)


def boundary_points(seq, raw_R, raw_Z, nfp, sign, theta, zeta):
    """``(F, F_theta, F_zeta, R, Z)`` of the boundary surface (``r = 1``, the last ring of the coefficients) at
    the scattered logical angles ``theta``, ``zeta``: any reals, ``R`` and ``Z`` periodic over a field period and
    the toroidal angle ``2 pi zeta / nfp``."""
    lt, lz = seq.basis_0.Λ[1], seq.basis_0.Λ[2]
    t, z = jnp.asarray(np.mod(theta, 1.0)), jnp.asarray(np.mod(zeta, 1.0))
    Bt, Bz = basis_table(lt, t), basis_table(lz, z)
    Dt = grad_1d(basis_table(seq.basis_0.dΛ[1], t), lt.type)
    Dz = grad_1d(basis_table(seq.basis_0.dΛ[2], z), lz.type)

    def ev(c, A, B):
        return np.asarray(jnp.einsum("jp,jk,kp->p", A, jnp.asarray(c)[-1], B))
    R, Z = ev(raw_R, Bt, Bz), ev(raw_Z, Bt, Bz)
    Rt, Zt, Rz, Zz = ev(raw_R, Dt, Bz), ev(raw_Z, Dt, Bz), ev(raw_R, Bt, Dz), ev(raw_Z, Bt, Dz)
    a = 2.0 * np.pi / nfp
    c, s = np.cos(a * zeta), np.sin(a * zeta)
    F = np.stack([R * c, sign * R * s, Z], -1)
    Ft = np.stack([Rt * c, sign * Rt * s, Zt], -1)
    Fz = np.stack([Rz * c - a * R * s, sign * (Rz * s + a * R * c), Zz], -1)
    return F, Ft, Fz, R, Z


def perturbation(seq, lp, amplitude, seed, m_max=4, n_max=4):
    """A smooth random normal displacement of LP's boundary as a change of the boundary ring ``(2, n_theta,
    n_zeta)``: ``delta = sum a_mn cos 2 pi (m theta - n zeta)`` over ``m <= m_max``, ``|n| <= n_max`` (``n`` per
    field period, so stellarator-symmetric and nfp-periodic), ``a_mn`` normal random over ``1 + m^2 + n^2``, scaled
    to the area-weighted RMS ``amplitude`` (m) at the Greville points. The displacement is ``delta`` along the
    normal, carried by ``(dR, dZ)`` along the normal's ``(R, Z)`` part (R even, Z odd)."""
    gt, gz = (np.asarray(seq.greville[a].point_rule[0][:, 0]) for a in (1, 2))
    T, Zg = (g.reshape(-1) for g in np.meshgrid(gt, gz, indexing="ij"))
    rng = np.random.default_rng(seed)
    delta = np.zeros_like(T)
    for m in range(m_max + 1):
        for n in range(-n_max, n_max + 1):
            delta += rng.standard_normal() / (1.0 + m ** 2 + n ** 2) * np.cos(2.0 * np.pi * (m * T - n * Zg))
    _, Ft, Fz, _, _ = boundary_points(seq, lp.raw_R, lp.raw_Z, lp.nfp, lp.sign, T, Zg)
    normal = np.cross(Ft, Fz)
    area = np.linalg.norm(normal, axis=-1)
    a = 2.0 * np.pi * Zg / lp.nfp
    n_R = normal[:, 0] * np.cos(a) + lp.sign * normal[:, 1] * np.sin(a)
    n_Z = normal[:, 2]
    delta *= amplitude / np.sqrt(np.sum(area * delta ** 2) / np.sum(area))
    t = delta * area / (n_R ** 2 + n_Z ** 2)
    shape2 = (gt.size, gz.size)
    return np.stack([collocate(seq, (t * n_R).reshape(shape2)), collocate(seq, (t * n_Z).reshape(shape2))])


def fine_points(basis, per_span):
    """``per_span`` uniform midpoints in every knot span of ``basis`` in [0, 1]."""
    T = np.asarray(basis.T)
    k = np.unique(T[(T >= 0.0) & (T <= 1.0)])
    return (k[:-1, None] + np.diff(k)[:, None] * ((np.arange(per_span) + 0.5) / per_span)[None, :]).ravel()


def start_guard(S, raw_R, raw_Z, floor=0.2, chunk=32):
    """The fold guard of a perturbed start, before any solve, on a grid ``4 (p + 1)`` points per knot span in
    every direction (4x the quadrature's) plus ``r = 1 - 1e-6``:

    - ``det DF / det DF_LP`` at every point (``det DF = (2 pi / nfp) R (R_theta Z_r - R_r Z_theta)``: the
      constant and the orientation cancel), its min and where, and whether it is ``>= floor``;
    - the boundary curve ``r = 1`` simple in every zeta plane of the grid (no two non-adjacent segments of the
      polygon cross);
    - the in-plane offset ``d`` of the boundary from LP's (at equal logical angles, on the principal normal ``N``
      of LP's cross-section), ``max |d|``, ``max |d kappa|`` and ``max d kappa`` (``kappa`` LP's in-plane
      curvature: ``d kappa = 1`` toward the centre of curvature is a cusp)."""
    seq, lp = S["seq"], S["lp_shape"]
    L, dL = seq.basis_0.Λ, seq.basis_0.dΛ
    n_span = 4 * (L[0].p + 1)
    pts = [fine_points(L[a], n_span) for a in range(3)]
    pts[0] = np.append(pts[0], 1.0 - 1e-6)
    tab = [basis_table(L[a], jnp.asarray(pts[a])) for a in range(3)]
    der = [grad_1d(basis_table(dL[a], jnp.asarray(pts[a])), L[a].type) for a in range(3)]
    dder_t = grad_1d(basis_derivative_table(dL[1], jnp.asarray(pts[1])), L[1].type)

    @jax.jit
    def planar(R, Z, zt):
        cz = [jnp.einsum("ijk,kz->ijz", C, zt) for C in (R, Z)]

        def val(C, A, B):
            return jnp.einsum("ia,ijz,jb->azb", A, C, B)
        return val(cz[0], tab[0], tab[1]) * (val(cz[0], tab[0], der[1]) * val(cz[1], der[0], tab[1])
                                              - val(cz[0], der[0], tab[1]) * val(cz[1], tab[0], der[1]))

    worst, where = np.inf, None
    for k0 in range(0, pts[2].size, chunk):
        zt = tab[2][:, k0:k0 + chunk]
        ratio = np.asarray(planar(jnp.asarray(raw_R), jnp.asarray(raw_Z), zt)
                           / planar(lp.raw_R, lp.raw_Z, zt))                                # (n_r, n_z, n_t)
        i = np.unravel_index(np.argmin(ratio), ratio.shape)
        if ratio[i] < worst:
            worst, where = float(ratio[i]), (float(pts[0][i[0]]), float(pts[1][i[2]]), float(pts[2][k0 + i[1]]))

    def ring(C, A, B):
        return np.asarray(A.T @ jnp.asarray(C)[-1] @ B)                                      # (n_theta, n_zeta)
    Rb, Zb = ring(raw_R, tab[1], tab[2]), ring(raw_Z, tab[1], tab[2])
    crossings = []
    n = Rb.shape[0]
    i, j = np.triu_indices(n, 2)
    keep = (j - i) < n - 1                                                                # not adjacent cyclically
    i, j = i[keep], j[keep]
    for kz in range(Rb.shape[1]):
        P = np.stack([Rb[:, kz], Zb[:, kz]], -1)
        Q = np.roll(P, -1, axis=0)

        def cross(o, a, b):
            return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])
        d1, d2 = cross(P[j], Q[j], P[i]), cross(P[j], Q[j], Q[i])
        d3, d4 = cross(P[i], Q[i], P[j]), cross(P[i], Q[i], Q[j])
        hits = int(np.sum((d1 * d2 < 0.0) & (d3 * d4 < 0.0)))
        if hits:
            crossings.append((float(pts[2][kz]), hits))
    R0, Z0 = ring(lp.raw_R, tab[1], tab[2]), ring(lp.raw_Z, tab[1], tab[2])
    Rt, Zt = ring(lp.raw_R, der[1], tab[2]), ring(lp.raw_Z, der[1], tab[2])
    Rtt, Ztt = ring(lp.raw_R, dder_t, tab[2]), ring(lp.raw_Z, dder_t, tab[2])
    speed = np.sqrt(Rt ** 2 + Zt ** 2)
    kappa = (Rt * Ztt - Zt * Rtt) / speed ** 3
    d = ((Rb - R0) * (-Zt) + (Zb - Z0) * Rt) / speed
    out = dict(det_ratio_min=worst, det_ratio_at_r_theta_zeta=where, det_floor=floor, grid=[int(v.size) for v in pts],
               crossings=crossings, max_offset_mm=1e3 * float(np.abs(d).max()),
               max_abs_offset_kappa=float(np.abs(d * kappa).max()), max_offset_kappa=float((d * kappa).max()),
               ok=bool(worst >= floor and not crossings))
    print(f"[guard] grid {out['grid']}: min det DF / det DF_LP {worst:.4f} at (r, theta, zeta) = "
          f"({where[0]:.4f}, {where[1]:.4f}, {where[2]:.4f}) (floor {floor}); boundary self-crossings in "
          f"{len(crossings)} of {Rb.shape[1]} planes; offset max {out['max_offset_mm']:.3f} mm, max |d kappa| "
          f"{out['max_abs_offset_kappa']:.3f}, max d kappa {out['max_offset_kappa']:+.3f} (1 = cusp); "
          f"{'pass' if out['ok'] else 'FAIL'}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- measures
def surface_distance(S, raw_R, raw_Z, grid=(128, 64), iterations=8, ref=None):
    """The distance of the boundary of the raw coefficients to LP's (or to ``ref``, the raw coefficients of another
    map): for every point ``x`` of it on a uniform ``grid`` of logical angles over a field period, ``(x -
    F_LP(u*)) . n_LP(u*)`` at the closest point ``u*`` (Gauss-Newton from the same logical angles); its RMS
    weighted by the area of this surface (``l2``, d_RMS of Sec. 3.4) and its max, in m."""
    seq, lp = S["seq"], S["lp_shape"]
    nfp, sign = seq.nfp, lp.sign
    ref_R, ref_Z = (lp.raw_R, lp.raw_Z) if ref is None else ref
    theta, zeta = (g.reshape(-1) for g in np.meshgrid(np.arange(grid[0]) / grid[0], np.arange(grid[1]) / grid[1],
                                                      indexing="ij"))
    X, Xt, Xz, _, _ = boundary_points(seq, raw_R, raw_Z, nfp, sign, theta, zeta)
    area = np.linalg.norm(np.cross(Xt, Xz), axis=-1)
    u = np.stack([theta, zeta], -1)
    for _ in range(iterations):
        F, Ft, Fz, _, _ = boundary_points(seq, ref_R, ref_Z, nfp, sign, u[:, 0], u[:, 1])
        Jac = np.stack([Ft, Fz], -1)                                                  # (n, 3, 2)
        u = u + np.linalg.solve(np.einsum("pki,pkj->pij", Jac, Jac),
                                np.einsum("pki,pk->pi", Jac, X - F)[..., None])[..., 0]
    F, Ft, Fz, _, _ = boundary_points(seq, ref_R, ref_Z, nfp, sign, u[:, 0], u[:, 1])
    nrm = np.cross(Ft, Fz)
    d = np.sum((X - F) * nrm / np.linalg.norm(nrm, axis=-1, keepdims=True), -1)
    return dict(l2=float(np.sqrt(np.sum(area * d ** 2) / np.sum(area))), max=float(np.abs(d).max()))


def distance(S, x):
    """:func:`surface_distance` of the boundary of the variables ``x`` to LP's."""
    R, Z, _, _ = S["pb"].shape.map_coefficients(S["seq"], change_of(S["pb"], jnp.asarray(x)))
    return surface_distance(S, R, Z)


def describe(S, d):
    """The distance in one line."""
    return f"{1e3 * d['l2']:.3f}/{1e3 * d['max']:.3f} mm RMS/max (d_RMS / a {d['l2'] / S['a_lp']:.3e})"


def spline_map(seq, raw_R, raw_Z, nfp, sign):
    """The map ``(R cos phi, sign R sin phi, Z)`` of raw coefficients, as :func:`mrx.gvec.build_gvec_map`."""
    basis = seq.basis_0.bases[0]
    a = 2.0 * np.pi / nfp
    raw_R, raw_Z = jnp.asarray(raw_R), jnp.asarray(raw_Z)

    def F(x):
        r = basis.contract(raw_R, x)
        return jnp.array([r * jnp.cos(a * x[2]), sign * r * jnp.sin(a * x[2]), basis.contract(raw_Z, x)])
    return F


def own_sequence(S, raw_R, raw_Z):
    """A copy of the setup's sequence on the map of the raw coefficients with its own preconditioners and harmonic
    forms (the production solve, not the AD one); set_map drops the copy's parity views, which carried LP's
    geometry, and they are rebuilt from the new map."""
    seq1 = copy.copy(S["seq"])
    seq1.set_map(spline_map(seq1, raw_R, raw_Z, seq1.nfp, S["lp_shape"].sign))
    seq1.build_preconditioners()
    compute_nullspaces(seq1, verbose=False)
    return seq1


def remeshed(S, raw_R, raw_Z):
    """``(R, Z, scale)``: the map with the boundary of the raw coefficients and LP's interior, carried along by the
    harmonic extension of the boundary change (:class:`mrx.shape_ad.BoundaryShape`, the boundary ring alone). The
    vacuum field is the boundary's, the mesh inside another one; ``scale`` is the volume rescaling it needed (the
    two maps enclose the same volume up to the quadrature)."""
    seq, lp = S["seq"], S["lp_shape"]
    shape = BoundaryShape.from_coefficients(seq, lp.raw_R, lp.raw_Z, lp.nfp, lp.sign, extension="harmonic")
    beta = jnp.stack([jnp.asarray(raw_R)[-1] - lp.raw_R[-1], jnp.asarray(raw_Z)[-1] - lp.raw_Z[-1]])
    R, Z, _, scale = shape.map_coefficients(seq, beta)
    return R, Z, float(scale)


def map_terms(S, R, Z):
    """The terms of :func:`terms` on the map of the raw coefficients, without the variables."""
    seq, pb = S["seq"], S["pb"]
    nfp, sign = pb.shape.nfp, pb.shape.sign
    sq = with_geometry(seq, cylindrical_geometry(seq, R, Z, nfp, sign))
    h, info = vacuum_two_form(sq, pb.seed)
    V, _, S_area = section_moments(seq, R, Z, nfp, sign)
    return dict(F_qs=float(quasisymmetry_residual(sq, h, R, Z, nfp, sign, pb.r_min)[0]),
                F_edge=float(edge_quasisymmetry_residual(sq, h, R, Z, nfp, sign, EPS)),
                mean_iota=float(pb.orient * mean_iota(sq, h)),
                iota_edge=float(pb.orient * flux_ratio_iota(sq, h, jnp.asarray([1.0 - EPS]))[0][0]),
                P=float(normal_field_fraction(sq, h)), jmin=float(jnp.min(sq.jacobian_j)),
                aspect=float(aspect_ratio(V, S_area, nfp)), volume=float(nfp * V), info=int(info))


def traced_profile(S, sq, cli, name):
    """The production harmonic form of ``sq`` traced (--lines x --periods, :func:`mrx.poincare.poincare`, which
    takes the odd parity view's form) and its traced iota against the flux ratio's (on the full half-period form):
    ``(res, iota_flux_ratio, s_line, order, traced <iota>_s)``, ``s_line`` the flux ratio's s at each line's mean
    logical radius at zeta = 0, ``order`` the regular lines inside out, the traced mean over s constant to the ends."""
    h = sq.nullspace(2, True)[0]
    res = poincare(sq, sq.odd.nullspace(2, True)[0], lines=cli.lines, periods=cli.periods, name=name)
    rbar = np.nanmean(res["sections"][0.0]["logr"], axis=1)
    iota_fr, s_line = (np.asarray(v) for v in flux_ratio_iota(sq, h, jnp.asarray(rbar)))
    shown, iota_tr = np.asarray(res["shown"]), np.asarray(res["iota"])
    order = np.flatnonzero(shown)[np.argsort(s_line[shown])]
    s_all = np.concatenate([[0.0], s_line[order], [1.0]])
    i_all = np.concatenate([iota_tr[order[:1]], iota_tr[order], iota_tr[order[-1:]]])
    return (res, float(S["pb"].orient) * iota_fr, s_line, order,
            float(np.sum(0.5 * (i_all[1:] + i_all[:-1]) * np.diff(s_all))))


# ---------------------------------------------------------------------------------------------------- setup
def setup(cli):
    if mrx.DTYPE != jnp.float64:
        sys.exit("run with MRX_DTYPE=float64")
    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, _ = build_sequence(GEOMETRY, ns, cli.p)
    compute_nullspaces(seq)
    _, info = build_gvec_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)
    lp = BoundaryShape.from_coefficients(seq, info["raw_R"], info["raw_Z"], seq.nfp, info["sign"])
    V_lp, _, S_lp = section_moments(seq, lp.raw_R, lp.raw_Z, lp.nfp, lp.sign)
    pb = Problem(shape=BoundaryShape.from_coefficients(seq, lp.raw_R, lp.raw_Z, lp.nfp, lp.sign, aspect=cli.aspect,
                                                       extension="harmonic", free="all"),
                 seed=flux_seed(seq), orient=jnp.asarray(1.0), iota_target=jnp.asarray(0.0), r_min=first_span(seq),
                 beta_scale=cli.beta_scale)
    zero = jnp.zeros(2 * lp.raw_R.size)
    aux_lp = forward(zero, seq, pb)
    orient = jnp.sign(aux_lp["mean_iota"])
    pb = eqx.tree_at(lambda p: (p.orient, p.iota_target), pb, (orient, orient * aux_lp["mean_iota"]))
    S = dict(seq=seq, ns=ns, lp_shape=lp, pb=pb, A_lp=float(aspect_ratio(V_lp, S_lp, lp.nfp)),
             V_lp=float(lp.nfp * V_lp), a_lp=float(np.sqrt(S_lp / np.pi)), F_lp=float(aux_lp["F_qs"]),
             iota_lp=float(pb.iota_target), P_lp=float(aux_lp["P"]), F_edge_lp=float(aux_lp["F_edge"]),
             iota_edge_lp=float(orient * aux_lp["iota_edge"]))
    full = np.zeros((2,) + tuple(lp.raw_R.shape))
    if cli.perturb_mm:
        full[:, -1] = perturbation(seq, lp, 1e-3 * cli.perturb_mm, cli.perturb_seed)
    S["x0"] = jnp.asarray(full.reshape(-1) / cli.beta_scale)
    if cli.perturb_mm:
        R_start, Z_start, _, _ = pb.shape.map_coefficients(seq, change_of(pb, S["x0"]))
        S["guard"] = start_guard(S, R_start, Z_start)
        with open(os.path.join(cli.out, f"qa_guard{cli.tag}.json"), "w") as fh:
            json.dump(S["guard"], fh, indent=1)
        if not S["guard"]["ok"]:
            sys.exit("the start fails the fold guard: not run")
    aux0 = forward(S["x0"], seq, pb)
    if float(aux0["jmin"]) <= 0.0:
        sys.exit(f"the start map folds: min det DF {float(aux0['jmin']):.3e}")
    print(f"[setup] QA {ns} p={cli.p}: {time.perf_counter() - t0:.0f} s; LP: A {S['A_lp']:.6f}, V {S['V_lp']:.6f} "
          f"m^3, a {S['a_lp']:.5f} m, F_QS {S['F_lp']:.10e}, F_edge {S['F_edge_lp']:.4e}, <iotabar>_s "
          f"{S['iota_lp']:.8f}, edge iota {S['iota_edge_lp']:.5f}, P {S['P_lp']:.2e}; A* {cli.aspect:g}, r >= "
          f"{pb.r_min:.4f}, {zero.size} variables; start: "
          f"{'%g mm RMS normal perturbation (seed %d), ' % (cli.perturb_mm, cli.perturb_seed) if cli.perturb_mm else 'LP, '}"
          f"distance {describe(S, distance(S, S['x0']))}, F_QS {float(aux0['F_qs']):.4e}, <iotabar>_s "
          f"{float(aux0['mean_iota']):.5f}, P {float(aux0['P']):.2e}, min det DF {float(aux0['jmin']):.3e} "
          f"({float(aux0['jrel']):+.3f} of the mean), A {float(aux0['aspect']):.12f}, V {float(aux0['volume']):.12f}",
          flush=True)
    return S


# ---------------------------------------------------------------------------------------------------- stages
def baseline(S, cli):
    """LP's <Q_QA^2>_{r >= h_r} on its interpolated map at --ns, scaled to A* about the axis as in every run (the
    runs' unit F_LP), with its <iotabar>_s and P -> qa_baseline<tag>.json."""
    rec = dict(ns=list(S["ns"]), p=cli.p, r_min=S["pb"].r_min, aspect=cli.aspect, F_lp=S["F_lp"],
               mean_iota=S["iota_lp"], P=S["P_lp"], A_lp=S["A_lp"])
    print(f"[baseline] LP at {S['ns']} p {cli.p}: <Q_QA^2>_(r >= {rec['r_min']:.4f}) {rec['F_lp']:.16e}", flush=True)
    with open(os.path.join(cli.out, f"qa_baseline{cli.tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)


def constrained(S, cli):
    """min <Q_QA^2>_{r >= h_r} subject to <iotabar>_s = LP's and P <= --p-max, V and A exact, by the augmented
    Lagrangian :func:`lagrangian`. Per outer step --al-inner L-BFGS-B iterations on it (a folding trial point is
    infeasible: it returns ten times the last value with the last gradient, so the line search backtracks), then
    the multipliers, lambda_i += rho c and lambda_P = max(0, lambda_P + rho g), and rho x 10 if the violation
    max(|c|, |max(g, -lambda_P / rho)|) is above its tolerance (the tighter of --iota-tol and --p-tol in the scaled
    units) and fell less than 4x. The objective's unit is F_QS of LP (a unit only), c's C_UNIT, g's --p-max.
    Stops when both constraints are within --iota-tol and --p-tol and F_QS changed less than --q-rtol relative over
    the last 100 iterations, or with --stop-qa at the first iteration where F_QS is at most --stop-qa x LP's with
    both constraints within tolerance, or after --maxiter iterations. The end: F_QS on the own and on the remeshed
    map (:func:`remeshed`), P, <iotabar>_s and the traced <iota>_s, the distance to LP and to --reference-end ->
    qa_constrained<tag>.json/.npz (per outer step and at the end, the variables at every outer step, the start and
    end coefficients), every iterate in qa_recover<tag>.npz/.json (:func:`save`)."""
    seq, pb = S["seq"], S["pb"]
    target, eps_P = float(pb.iota_target), cli.p_max
    al = dict(f_unit=S["F_lp"], eps=eps_P, lam_i=0.0, lam_P=0.0, rho=cli.al_rho)
    x = np.asarray(S["x0"], dtype=np.float64)
    R0, Z0, _, _ = pb.shape.map_coefficients(seq, change_of(pb, jnp.asarray(x)))
    kept, outer, it_total, V_prev, F_path = {}, [], 0, None, []
    V_tol = min(cli.iota_tol / C_UNIT, (cli.p_tol - eps_P) / eps_P)
    print(f"[constrained] target <iotabar>_s = {target:.8f}; units: f = F_QS / {al['f_unit']:.4e} (LP), c = "
          f"(<iotabar>_s - {target:.8f}) / {C_UNIT:g}, g = (P - {eps_P:g}) / {eps_P:g}; rho {al['rho']:g}", flush=True)
    cache, history, folded, last, reached = {}, [], [0], {}, [False]
    t_start = time.perf_counter()

    def jal():
        return {k: jnp.asarray(v) for k, v in al.items()}

    def fun(x):
        key = x.tobytes()
        if key not in cache:
            if float(min_det_df(jnp.asarray(x), seq, pb)) <= 0.0:
                folded[0] += 1
                return last["big"], last["g"]
            t0 = time.perf_counter()
            (L, aux), g = value_and_grad(jnp.asarray(x), seq, pb, jal())
            r = {k: float(v) for k, v in aux.items()}
            r.update(Q=float(L), g=np.asarray(g, dtype=np.float64), x=np.array(x), wall=time.perf_counter() - t0)
            cache[key] = r
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
        return cache[key]["Q"], cache[key]["g"]

    def accept(x):
        fun(x)
        r = {k: v for k, v in cache[x.tobytes()].items() if k != "g"}
        r["distance"] = distance(S, x)
        history.append(r)
        return r

    def callback(intermediate_result):
        nonlocal it_total
        it_total += 1
        r = accept(np.asarray(intermediate_result.x))
        F_path.append((it_total, r["F_qs"] / al["f_unit"]))
        if (cli.stop_qa and r["F_qs"] <= cli.stop_qa * S["F_lp"] and r["P"] <= cli.p_tol
                and abs(r["mean_iota"] - target) <= cli.iota_tol):
            reached[0] = True
            raise StopIteration
        if it_total % 10 == 0:
            print(f"[iter {it_total:5d}] L {r['Q']:.5e}  F_QS {r['F_qs']:.4e} ({r['F_qs'] / S['F_lp']:.3f} LP)  iota "
                  f"{r['mean_iota']:.6f}  P {r['P']:.3e}  min J {r['jmin']:.2e}  shape {describe(S, r['distance'])}  "
                  f"({r['wall']:.1f} s)", flush=True)

    def violation(r):
        c = (r["mean_iota"] - target) / C_UNIT
        g = (r["P"] - eps_P) / eps_P
        return c, g, max(abs(c), abs(max(g, -al["lam_P"] / al["rho"])))

    def write(x, reason):
        kept[f"x_{it_total}"] = np.array(x)
        R1, Z1, _, _ = pb.shape.map_coefficients(seq, change_of(pb, jnp.asarray(x)))
        np.savez(os.path.join(cli.out, f"qa_constrained{cli.tag}.npz"),
                 **{**kept, "raw_R0": np.asarray(R0), "raw_Z0": np.asarray(Z0), "raw_R1": np.asarray(R1),
                    "raw_Z1": np.asarray(Z1), "raw_R_lp": np.asarray(S["lp_shape"].raw_R),
                    "raw_Z_lp": np.asarray(S["lp_shape"].raw_Z)})
        with open(os.path.join(cli.out, f"qa_constrained{cli.tag}.json"), "w") as fh:
            json.dump(dict(outer=outer, F_path=F_path, reason=reason, units=dict(f=al["f_unit"], c=C_UNIT, g=eps_P),
                           p_max=eps_P, iota_target=target, perturb_mm=cli.perturb_mm, perturb_seed=cli.perturb_seed),
                      fh, indent=1, default=float)
        return R1, Z1

    def outer_record(r, c, g, V, **more):
        return dict(iterations=it_total, F_qs=r["F_qs"], mean_iota=r["mean_iota"], P=r["P"], jmin=r["jmin"], c=c,
                    g=g, violation=V, lam_i=al["lam_i"], lam_P=al["lam_P"], rho=al["rho"],
                    rms_mm=1e3 * r["distance"]["l2"], max_mm=1e3 * r["distance"]["max"], **more)

    r = accept(x)
    last.update(big=10.0 * abs(r["Q"]) + 1.0, g=cache[x.tobytes()]["g"])
    c, g, _ = violation(r)
    print(f"[start] F_QS {r['F_qs']:.4e} ({r['F_qs'] / S['F_lp']:.1f} LP), <iotabar>_s {r['mean_iota']:.6f} "
          f"(c {c:+.3e}), P {r['P']:.3e} (g {g:+.3e}), min det DF {r['jmin']:.3e}, shape {describe(S, r['distance'])}",
          flush=True)
    outer.append(outer_record(r, c, g, None, wall=0.0))
    write(x, "start")
    reason = f"{cli.maxiter} iterations"
    while it_total < cli.maxiter:
        last.update(big=10.0 * abs(history[-1]["Q"]) + 1.0, g=cache[x.tobytes()]["g"])
        res = scipy.optimize.minimize(fun, x, jac=True, method="L-BFGS-B", callback=callback,
                                      options=dict(maxiter=min(cli.al_inner, cli.maxiter - it_total), ftol=1e-15,
                                                   gtol=1e-15, maxcor=cli.maxcor))
        x = np.asarray(history[-1]["x"])
        r = history[-1]
        c, g, V = violation(r)
        al["lam_i"] += al["rho"] * c
        al["lam_P"] = max(0.0, al["lam_P"] + al["rho"] * g)
        if V_prev is not None and V > V_tol and V > 0.25 * V_prev:
            al["rho"] *= 10.0
        V_prev = V
        cache.clear()
        fun(x)
        history[-1]["Q"] = cache[x.tobytes()]["Q"]
        outer.append(outer_record(r, c, g, V, wall=time.perf_counter() - t_start, inner=res.message,
                                  folded=folded[0]))
        print(f"[outer {len(outer) - 1}] iteration {it_total}: F_QS {r['F_qs']:.4e} ({r['F_qs'] / S['F_lp']:.3f} LP), "
              f"<iotabar>_s {r['mean_iota']:.6f} (c {c:+.3e}), g {g:+.3e}, violation {V:.3e}; lambda_i "
              f"{al['lam_i']:+.4e}, lambda_P {al['lam_P']:.4e}, rho {al['rho']:g}; shape {describe(S, r['distance'])}; "
              f"{res.message} ({res.nit} its, {folded[0]} folded)", flush=True)
        write(x, "running")
        past = [F for i, F in F_path if i <= it_total - 100]
        feasible = r["P"] <= cli.p_tol and abs(r["mean_iota"] - target) <= cli.iota_tol
        merit = r["F_qs"] / al["f_unit"]
        if feasible and past and abs(merit - past[-1]) < cli.q_rtol * merit:
            reason = f"converged at iteration {it_total}"
            break
        if reached[0]:
            reason = f"QA <= {cli.stop_qa:g} LP with both constraints at iteration {it_total}"
            break
        if res.nit == 0:
            reason = f"no progress at iteration {it_total}"
            break
    wall = time.perf_counter() - t_start
    R1, Z1 = write(x, reason)
    own = map_terms(S, R1, Z1)
    Rm, Zm, _ = remeshed(S, R1, Z1)
    rem = map_terms(S, Rm, Zm)
    res, iota_fr, s_line, order, traced = traced_profile(S, own_sequence(S, R1, Z1), cli, f"iteration {it_total}")
    iota_tr = np.asarray(res["iota"])
    first = outer[0]
    end = dict(reason=reason, iterations=it_total, wall=wall, start_rms_mm=first["rms_mm"],
               start_max_mm=first["max_mm"], rms_mm=1e3 * r["distance"]["l2"], max_mm=1e3 * r["distance"]["max"],
               qa_own=own["F_qs"] / S["F_lp"], qa_remeshed=rem["F_qs"] / S["F_lp"], P=own["P"], P_remeshed=rem["P"],
               iota_flux_ratio=own["mean_iota"], iota_traced=traced, regular=f"{order.size}/{iota_tr.size}",
               jmin=own["jmin"], lam_i=al["lam_i"], lam_P=al["lam_P"], rho=al["rho"],
               traced_profile=dict(s=s_line[order].tolist(), iota_traced=iota_tr[order].tolist(),
                                   iota_flux_ratio=iota_fr[order].tolist()),
               traced_minus_flux_ratio_rms=float(np.sqrt(np.mean((iota_tr[order] - iota_fr[order]) ** 2))))
    if cli.reference_end:
        ref = np.load(os.path.join(cli.out, cli.reference_end))
        d_ref = surface_distance(S, R1, Z1, ref=(jnp.asarray(ref["raw_R1"]), jnp.asarray(ref["raw_Z1"])))
        end.update(reference_end=cli.reference_end, ref_rms_mm=1e3 * d_ref["l2"], ref_max_mm=1e3 * d_ref["max"])
    with open(os.path.join(cli.out, f"qa_constrained{cli.tag}.json")) as fh:
        rec = json.load(fh)
    rec["end"] = end
    with open(os.path.join(cli.out, f"qa_constrained{cli.tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(f"[end] {reason}, {wall:.0f} s: {end['start_rms_mm']:.3f}/{end['start_max_mm']:.3f} -> {end['rms_mm']:.3f}/"
          f"{end['max_mm']:.3f} mm, QA/LP own {end['qa_own']:.3f} remeshed {end['qa_remeshed']:.3f}, P {end['P']:.3e} "
          f"(remeshed {end['P_remeshed']:.2e}), <iotabar>_s {end['iota_flux_ratio']:.6f} traced {traced:.6f} "
          f"({end['regular']}), min det DF {end['jmin']:.3e}; lambda_i {al['lam_i']:+.4e} lambda_P {al['lam_P']:.4e} "
          f"rho {al['rho']:g}; traced - flux ratio RMS {end['traced_minus_flux_ratio_rms']:.2e}"
          + (f"; to {cli.reference_end}: {end['ref_rms_mm']:.3f}/{end['ref_max_mm']:.3f} mm" if cli.reference_end
             else ""), flush=True)
    save(S, cli, history, reason, wall)


def save(S, cli, history, message, wall):
    """qa_recover<tag>.npz/.json: every accepted iterate of the constrained run (its terms and distance to LP), the
    raw coefficients of LP's map, of the start and of the end, and the run's summary."""
    seq, pb = S["seq"], S["pb"]
    first, last = history[0], history[-1]
    R0, Z0, _, _ = pb.shape.map_coefficients(seq, change_of(pb, jnp.asarray(first["x"])))
    R1, Z1, _, _ = pb.shape.map_coefficients(seq, change_of(pb, jnp.asarray(last["x"])))
    summary = dict(message=message, wall=wall, iterations=len(history) - 1,
                   **{k: [first[k], last[k]] for k in ("F_qs", "F_edge", "mean_iota", "iota_edge", "P", "aspect",
                                                       "volume", "jmin", "jrel", "jmin_edge", "info")},
                   jmin_min=min(r["jmin"] for r in history), jrel_min=min(r["jrel"] for r in history),
                   jmin_edge_min=min(r["jmin_edge"] for r in history),
                   distance=[first["distance"], last["distance"]], F_lp=S["F_lp"], iota_lp=S["iota_lp"],
                   P_lp=S["P_lp"], F_edge_lp=S["F_edge_lp"], iota_edge_lp=S["iota_edge_lp"], A_lp=S["A_lp"],
                   a_lp=S["a_lp"], iota_target=float(pb.iota_target), eps=EPS, aspect_target=cli.aspect,
                   volume_target=S["V_lp"], perturb_mm=cli.perturb_mm, perturb_seed=cli.perturb_seed, r_min=pb.r_min,
                   ns=S["ns"], p=cli.p, beta_scale=cli.beta_scale, maxcor=cli.maxcor)
    print(f"[{message}] {len(history) - 1} iterations, {wall:.0f} s; F_QS {first['F_qs']:.4e} -> {last['F_qs']:.4e} "
          f"(LP {S['F_lp']:.4e}); <iotabar>_s {last['mean_iota']:.5f}; shape {describe(S, first['distance'])} -> "
          f"{describe(S, last['distance'])}; A - A* {last['aspect'] - cli.aspect:+.1e}, V / V_LP - 1 "
          f"{last['volume'] / S['V_lp'] - 1.0:+.1e}; P {last['P']:.2e}; min det DF {summary['jmin_min']:.3e} "
          f"({summary['jrel_min']:+.3f} of the mean), on the edge {summary['jmin_edge_min']:.3e}", flush=True)
    keys = ("Q", "F_qs", "F_par", "F_edge", "mean_iota", "iota_edge", "P", "aspect", "volume", "mu", "scale", "jmin",
            "jrel", "jmin_edge", "info", "wall")
    lp = S["lp_shape"]
    np.savez(os.path.join(cli.out, f"qa_recover{cli.tag}.npz"),
             **{f"hist_{k}": np.array([r[k] for r in history]) for k in keys},
             **{f"hist_dist_{k}": np.array([r["distance"][k] for r in history]) for k in first["distance"]},
             x_final=last["x"], F_lp=S["F_lp"], iota_lp=S["iota_lp"], F_edge_lp=S["F_edge_lp"],
             iota_edge_lp=S["iota_edge_lp"], raw_R_lp=np.asarray(lp.raw_R), raw_Z_lp=np.asarray(lp.raw_Z),
             raw_R0=np.asarray(R0), raw_Z0=np.asarray(Z0), raw_R1=np.asarray(R1), raw_Z1=np.asarray(Z1))
    with open(os.path.join(cli.out, f"qa_recover{cli.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)


def trace(S, cli):
    """Poincare sections of LP (``start``) and of the end shape of qa_recover<tag>.npz (``final``), each with its
    own production harmonic form, in the archive layout of scripts/poincare_trace.py so scripts/poincare_plot.py
    renders it. Per field: the traced iota of every line against its flux label (the flux ratio's s at the line's
    mean logical radius at zeta = 0), the mean of the traced profile over s against the flux ratio's mean iota,
    the outermost regular lines against the flux ratio's edge iota, and the magnetic axis against LP's coordinate
    axis (its VMEC magnetic axis) and against the field's own coordinate axis -> qa_trace<tag>[_remesh].npz/.json.
    With --remesh the end boundary on LP's interior (:func:`remeshed`): the same vacuum field on a smooth mesh."""
    seq, pb, lp = S["seq"], S["pb"], S["lp_shape"]
    z = np.load(os.path.join(cli.out, f"qa_recover{cli.tag}.npz"))
    R1, Z1 = jnp.asarray(z["raw_R1"]), jnp.asarray(z["raw_Z1"])
    rec, suffix = {}, ""
    if cli.remesh:
        R1, Z1, scale = remeshed(S, R1, Z1)
        suffix = "_remesh"
        rec = dict(remesh_scale_minus_1=scale - 1.0, end_terms=map_terms(S, jnp.asarray(z["raw_R1"]),
                                                                       jnp.asarray(z["raw_Z1"])),
                   remesh_terms=map_terms(S, R1, Z1), remesh_distance=surface_distance(S, R1, Z1))
        print(f"[trace] remesh: the end boundary on LP's interior (volume scale - 1 {scale - 1.0:+.1e}, shape "
              f"{describe(S, rec['remesh_distance'])}); end map {rec['end_terms']}; remeshed {rec['remesh_terms']}",
              flush=True)
    t0 = time.perf_counter()
    seq1 = own_sequence(S, R1, Z1)
    # the AD field of the end shape (LP's frozen preconditioners) against the production harmonic form
    h_ref = seq1.nullspace(2, True)[0]
    h_ref = h_ref / seq1.l2_norm(h_ref, 2)
    h_ad = vacuum_two_form(with_geometry(seq, cylindrical_geometry(seq, R1, Z1, lp.nfp, lp.sign)), pb.seed)[0]
    h_ad = h_ad / seq1.l2_norm(h_ad, 2)
    h_ad = h_ad * jnp.sign(h_ad @ seq1.apply_mass_matrix(h_ref, 2))
    rec["final_h_rel_diff"] = float(seq1.l2_norm(h_ad - h_ref, 2))
    print(f"[trace] end shape: preconditioners + harmonic forms {time.perf_counter() - t0:.0f} s; AD field vs "
          f"production harmonic form, rel. M2 diff {rec['final_h_rel_diff']:.2e}", flush=True)
    plane_table = {}

    def axis_at(raw, plane):
        if plane not in plane_table:
            plane_table[plane] = np.asarray(basis_table(seq.basis_0.Λ[2], jnp.asarray([plane])))[:, 0]
        return float(np.asarray(raw)[0, 0] @ plane_table[plane])
    out = dict(fields=np.array(["start", "final"]), ns=np.array(S["ns"]), p=cli.p, nfp=seq.nfp,
               symmetry=seq.symmetry, source=f"QA design {S['ns']} p={cli.p}", pressure_kind="none",
               trace_precision="float64", movie=False, start_label="LP", final_label="QA design")
    for name, (sq, R, Z) in {"start": (seq, lp.raw_R, lp.raw_Z), "final": (seq1, R1, Z1)}.items():
        t0 = time.perf_counter()
        res, iota_fr, s_line, order, traced_mean = traced_profile(S, sq, cli, name)
        t1 = time.perf_counter()
        h = sq.nullspace(2, True)[0]
        rho_line = np.sqrt(s_line)
        shown, iota_tr = np.asarray(res["shown"]), np.asarray(res["iota"])
        axis = {}
        for plane, sec in res["sections"].items():
            mag = (float(np.mean(sec["axisR"])), float(np.mean(sec["axisZ"])))
            own, ref = (axis_at(R, plane), axis_at(Z, plane)), (axis_at(lp.raw_R, plane), axis_at(lp.raw_Z, plane))
            axis[f"{plane:g}"] = dict(magnetic=mag, coordinate=own, lp_coordinate=ref,
                                      magnetic_minus_lp_mm=1e3 * float(np.hypot(mag[0] - ref[0], mag[1] - ref[1])),
                                      magnetic_minus_own_mm=1e3 * float(np.hypot(mag[0] - own[0], mag[1] - own[1])),
                                      own_minus_lp_mm=1e3 * float(np.hypot(own[0] - ref[0], own[1] - ref[1])))
        rec[name] = dict(
            poincare_s=t1 - t0, drift=float(res["drift"]), n_shown=int(shown.sum()),
            n_lines=int(shown.size), traced_mean_iota=traced_mean,
            flux_ratio_mean_iota=float(pb.orient * mean_iota(sq, h)),
            flux_ratio_edge_iota=float(pb.orient * flux_ratio_iota(sq, h, jnp.asarray([1.0 - EPS]))[0][0]),
            outermost=[dict(rho=float(rho_line[k]), iota=float(iota_tr[k]), iota_flux_ratio=float(iota_fr[k]))
                       for k in order[-5:]],
            traced_minus_flux_ratio_max_abs=float(np.max(np.abs(iota_tr[shown] - iota_fr[shown]))),
            traced_minus_flux_ratio_mean=float(np.mean(iota_tr[shown] - iota_fr[shown])), axis=axis)
        print(f"[trace] {name}: {rec[name]['n_shown']}/{rec[name]['n_lines']} regular lines, drift "
              f"{rec[name]['drift']:.1e}; mean iota traced {traced_mean:.5f}, flux ratio "
              f"{rec[name]['flux_ratio_mean_iota']:.5f}; edge iota flux ratio {rec[name]['flux_ratio_edge_iota']:.5f}; "
              f"traced - flux ratio max |.| {rec[name]['traced_minus_flux_ratio_max_abs']:.1e}, mean "
              f"{rec[name]['traced_minus_flux_ratio_mean']:+.1e}; magnetic axis - LP axis "
              f"{[round(a['magnetic_minus_lp_mm'], 2) for a in axis.values()]} mm, - own coordinate axis "
              f"{[round(a['magnetic_minus_own_mm'], 2) for a in axis.values()]} mm (poincare {t1 - t0:.0f} s)",
              flush=True)
        for key in ("iota", "iota_err", "iota_scatter", "seed_r", "keep", "chaotic", "shown"):
            out[f"{name}_{key}"] = np.asarray(res[key])
        out[f"{name}_rho"], out[f"{name}_iota_proxy"], out[f"{name}_drift"] = rho_line, iota_fr, res["drift"]
        for plane, sec in res["sections"].items():
            for key in ("R", "Z", "axisR", "axisZ", "logr", "logth"):
                out[f"{name}_zeta{plane:g}_{key}"] = np.asarray(sec[key])
        out["planes"], out["steps"] = np.array(list(res["sections"].keys())), res["steps"]
    np.savez(os.path.join(cli.out, f"qa_trace{cli.tag}{suffix}.npz"), **out)
    with open(os.path.join(cli.out, f"qa_trace{cli.tag}{suffix}.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v))


STAGES = dict(constrained=constrained, trace=trace, baseline=baseline)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, help=f"comma-separated, of {', '.join(STAGES)}")
    ap.add_argument("--records", default=os.environ.get("MRX_RECORDS", os.path.join(REPO, "outputs")),
                    help="the records root; the records go to <records>/shape_optimization [MRX_RECORDS or outputs]")
    ap.add_argument("--tag", default="", help="the records' suffix")
    ap.add_argument("--ns", default="24,48,24", help="n_r,n_theta,n_zeta [24,48,24]")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--aspect", type=float, default=6.0, help="the aspect ratio held exactly")
    ap.add_argument("--beta-scale", type=float, default=0.01, help="metres of change per unit of x")
    ap.add_argument("--perturb-mm", type=float, default=0.0,
                    help="the start: LP plus a smooth random normal boundary displacement of this RMS (mm) [0: LP]")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=4000, help="constrained: L-BFGS-B iterations in all")
    ap.add_argument("--al-inner", type=int, default=100, help="constrained: L-BFGS-B iterations per multiplier update")
    ap.add_argument("--maxcor", type=int, default=100, help="constrained: L-BFGS-B correction pairs")
    ap.add_argument("--al-rho", type=float, default=1.0, help="constrained: the initial penalty rho")
    ap.add_argument("--p-max", type=float, default=1e-8, help="constrained: the constraint P <= this")
    ap.add_argument("--p-tol", type=float, default=1.01e-8, help="constrained: feasible when P <= this")
    ap.add_argument("--iota-tol", type=float, default=1e-5,
                    help="constrained: feasible when |<iotabar>_s - target| <= this")
    ap.add_argument("--q-rtol", type=float, default=1e-3,
                    help="constrained: converged when F_QS changed less than this relative over 100 iterations")
    ap.add_argument("--stop-qa", type=float, default=0.0,
                    help="constrained: stop at the first iteration with F_QS <= this x LP's and both constraints "
                         "within tolerance [0: off]")
    ap.add_argument("--reference-end", default="",
                    help="constrained: also the distance of the end boundary to this qa_constrained<tag>.npz's end")
    ap.add_argument("--lines", type=int, default=160, help="trace, and the end of constrained: field lines")
    ap.add_argument("--periods", type=int, default=400, help="trace, and the end of constrained: field periods")
    ap.add_argument("--remesh", action="store_true",
                    help="trace: the end boundary on LP's interior (harmonic), not the end map")
    return ap


def main():
    cli = parser().parse_args()
    cli.out = os.path.join(cli.records, "shape_optimization")
    os.makedirs(cli.out, exist_ok=True)
    print("argv:", " ".join(sys.argv), flush=True)
    S = setup(cli)
    for stage in cli.stage.split(","):
        STAGES[stage](S, cli)


if __name__ == "__main__":
    main()
