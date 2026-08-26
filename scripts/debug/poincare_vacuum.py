"""Poincaré sections of the harmonic (vacuum) field on the production geometries.

The field comes from the nullspace computers, in both of the forms that carry
it.  For a solid torus the harmonic space is one-dimensional at either end of
the sequence:

* ``k = 2`` with essential BCs -- the Neumann harmonic 2-form, ``n . B = 0``;
* ``k = 1`` with natural BCs   -- the absolute harmonic 1-form, ``n _| A = 0``.

Both are ``d``- and ``delta``-closed and boundary-tangent, so up to sign and
normalisation they are the *same physical field*, computed by two different
solve chains.  The script reports the angle between them before plotting
anything: that number is a free correctness check on both nullspace routes, and
the two Poincaré plots should be indistinguishable.

Tracing is :mod:`mrx.poincare` -- toroidal-angle parameterisation, prescribed
step schedule, Cartesian cross-section chart.  Colour is the rotational
transform measured about the magnetic axis.

    python scripts/debug/poincare_vacuum.py --geometry w7x --periods 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import jax


import matplotlib  # noqa: E402

matplotlib.use("Agg")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.nullspace import (  # noqa: E402
    compute_nullspaces, exact_derivative_residual, generic_rayleigh,
    harmonic_rayleigh,
)
from mrx.poincare import (  # noqa: E402
    SURFACE_LABELS, logical_field, render_section,
    require_zeta_parameterisation, section_RZ, seed_from_axis, seed_line,
    surface_label, trace, trace_and_classify,
)
from verify_block_jacobi import build_sequence  # noqa: E402

#: Field periods spanned by logical zeta in [0, 1].  The stellarator maps model
#: ONE field period (``phi = 2 pi zeta / nfp``); the axisymmetric maps model the
#: whole torus, so their nfp is 1 by this definition.  Iota is reported per
#: toroidal turn, which is where the factor enters.
NFP = {
    "toroid": 1, "cylinder": 1, "rot-ellipse": 3, "w7x": 5,
    "quasr9983": 2, "quasr44970": 3, "w7x-gvec": 5, "hegna": 3,
    "quasr65530": 4, "quasr65575": 4, "w7x-ini": 5,
    # The 8x16x8 quasr44970 baseline and its two interior perturbations.  The
    # perturbed files declare nfp=2 and their R/Z data says otherwise; see
    # GVEC_NFP_OVERRIDE in mrx.gvec.
    "quasr44970-c": 3, "pert-axis": 3, "pert-interior": 3,
}

#: (k, dirichlet, label) for the two harmonic fields.
FIELDS = {"k2": (2, True, "k=2, essential BC"),
          "k1": (1, False, "k=1, natural BC")}


def report_preconditioners(seq, ops):
    """Print what every solve in this run will ACTUALLY be preconditioned with.

    The default resolution is a chain of private materialisers, and the last
    time one of them quietly degraded (the `_tensor_available` gate on the
    saddle lower block, and `schur.outer` pinned to 'jacobi') the k>=1 solves
    ran effectively unpreconditioned for months. So resolve them here and say
    so, rather than trusting that 'auto' means what it should.

    RuntimeWarning is promoted to an error for the whole run: the one fallback
    that is still reachable -- schur.outer dropping to the per-DoF diagonal
    when the block atom was not assembled -- announces itself that way, and a
    warning in a 25-job overnight log is a warning nobody reads.
    """
    for k in (1, 2, 3):
        for dbc in (False, True):
            spec = op._materialize_default_saddle_preconditioner(
                seq, ops, k=k, dirichlet=dbc)
            avail = op._metric_lumping_available(seq, k, dbc)
            outer = spec.schur.outer.kind
            # schur.inner is deliberately NOT printed. With
            # outer='metric_lumping' the atom approximates L_k directly and
            # "needs neither the Schur probe nor schur.inner -- it IS the
            # upper-block inverse" (operators.py). Printing it invites reading
            # a dead field as the thing doing the work.
            print(f"[precond] k={k} {'dbc ' if dbc else 'free'}: "
                  f"mass={spec.mass.kind:12s} schur.outer={outer:7s} "
                  f"(atom assembled: {avail})", flush=True)
            if outer != 'metric_lumping':
                raise RuntimeError(
                    f"k={k} dirichlet={dbc} resolved schur.outer={outer!r}, "
                    "not the block atom")
    # Only k=0 FREE is assembled: it is the one k=0 Laplacian solve in mrx/,
    # reached through apply_leray_projection(seed1, k=1). k=0 dbc is expected to
    # read 'jacobi' here and that is correct, not a fallback -- nothing solves
    # it, so nothing should pay to precondition it.
    for dbc in (False, True):
        k0 = op._materialize_default_scalar_hodge_preconditioner(
            seq, ops, k=0, dirichlet=dbc)
        print(f"[precond] k=0 {'dbc ' if dbc else 'free'}: {k0.kind:12s} "
              f"(atom assembled: {op._metric_lumping_available(seq, 0, dbc)})"
              + ("" if dbc else "   <- the Leray solve in the k=1 chain"),
              flush=True)
    k0_free = op._materialize_default_scalar_hodge_preconditioner(
        seq, ops, k=0, dirichlet=False)
    if k0_free.kind != 'metric_lumping':
        raise RuntimeError(
            f"k=0 free resolved {k0_free.kind!r}, not the block atom")


def report_harmonic(seq, ops):
    """Rayleigh quotient of each stored harmonic form. Zero if it is harmonic.

    `compute_nullspaces` is a chain of Hodge solves with a fixed budget and no
    convergence gate of its own: a solve that runs out of iterations returns a
    non-harmonic vector and every deflated solve downstream inherits it in
    silence. This is the number that catches that.
    """
    out = {}
    for name, (k, dbc) in ((n, FIELDS[n][:2]) for n in ("k2", "k1")):
        v = ops.null_2_dbc[0] if k == 2 else ops.null_1[0]
        lam = harmonic_rayleigh(seq, v, k, dirichlet=dbc, operators=ops)
        lam_rand = generic_rayleigh(seq, k, dirichlet=dbc, operators=ops,
                                    seed=11 * k)
        dres = exact_derivative_residual(seq, v, k, dirichlet=dbc)
        out[name] = {"rayleigh": lam, "rayleigh_random": lam_rand,
                     "ratio": lam / lam_rand, "exact_derivative_rel": dres}
        print(f"[harmonic] {name}: rayleigh={lam:12.5e}  "
              f"random={lam_rand:12.5e}  ratio={lam / lam_rand:10.3e}  "
              f"|dv|/|v|={dres:10.3e}", flush=True)
    return out


def field_agreement(seq, ops, n=512):
    """Max angle between the two harmonic fields, over random logical points.

    Both are boundary-tangent harmonic fields on a one-dimensional harmonic
    space, so the angle is zero up to discretisation and solve tolerance.
    """
    f2 = logical_field(seq, ops.null_2_dbc[0], 2, True)
    f1 = logical_field(seq, ops.null_1[0], 1, False)
    key = jax.random.PRNGKey(7)
    x = jax.random.uniform(key, (n, 3)).at[:, 0].multiply(0.95).at[:, 0].add(0.02)
    v2 = jax.vmap(f2)(x)
    v1 = jax.vmap(f1)(x)
    v2 = v2 / jnp.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / jnp.linalg.norm(v1, axis=1, keepdims=True)
    cos = jnp.abs(jnp.sum(v2 * v1, axis=1))
    ang = jnp.arccos(jnp.clip(cos, -1.0, 1.0))
    # Max AND median. The max is one point out of 512 and is easily set by a
    # single sample near r -> 1, where the spline map is nearly singular; it is
    # not a statistic to read a convergence rate off. The median is.
    return (float(jnp.max(ang)), float(jnp.median(ang)),
            float(jnp.percentile(ang, 90)))


def zeta_component_report(field, name, n=4096):
    """Does ``B^zeta`` keep one sign? The reparameterisation dies if it does not.

    The tracer integrates ``dr/dzeta = B^r/B^zeta``, which is exact only where
    ``B^zeta != 0``. For a converged toroidal field that is everywhere. For a
    badly resolved DISCRETE one it need not be, and when it is not the ODE is
    singular, lines leave the domain, and the section still plots -- as
    something that looks like a chaotic sea and is really a broken change of
    variables.

    Refining the step does not help, which is the signature to look for: a step
    drift that stays large as the step shrinks.

    Since 2026-08-25 this RAISES rather than printing and carrying on. It used
    to warn and trace anyway, which put the burden on whoever read the log to
    notice one line among hundreds -- and the failure it guards against renders
    as a plausible chaotic sea, so nobody would.
    """
    info = require_zeta_parameterisation(field, n=n, name=name)
    print(f"[zeta] {name}: B^zeta/|B| in "
          f"[{info['bz_over_b_min']:+.3e}, {info['bz_over_b_max']:+.3e}]  "
          f"(closest approach to zero {info['bz_over_b_absmin']:.3e}, "
          f"tol {info['tol']:g})", flush=True)
    return info


def bench(field, seeds, cli):
    """Time the prescribed schedule against the adaptive one it replaces.

    The third arm is the point: chunking an adaptive batch does not isolate a
    pathological seed, it only bounds how many healthy seeds each one holds up.
    A prescribed schedule has no such coupling, so its cost per seed is flat.
    """
    arms = [("prescribed, vmap", dict(adaptive=False, batch_size=None)),
            ("adaptive, vmap", dict(adaptive=True, batch_size=None)),
            ("adaptive, chunk 8", dict(adaptive=True, batch_size=8))]
    out = {}
    for name, kw in arms:
        t0 = time.perf_counter()
        ys, _ = trace(field, seeds, cli.bench_periods, cli.steps, cli.saves,
                      **kw)
        jnp.asarray(ys).block_until_ready()
        out[name] = time.perf_counter() - t0
        print(f"[bench] {name:20s} {out[name]:8.2f}s", flush=True)
    return out


def run_field(seq, dof, k, dirichlet, nfp, cli):
    field = logical_field(seq, dof, k, dirichlet)
    if cli.seed_from == "axis":
        seeds = seed_from_axis(field, cli.seeds, cli.saves, r_edge=cli.r_max,
                               steps_per_period=cli.steps)
    else:
        seeds = seed_line(cli.seeds, r_min=cli.r_min, r_max=cli.r_max)
    return trace_and_classify(
        field, seeds, nfp, n_periods=cli.periods, steps_per_period=cli.steps,
        saves_per_period=cli.saves, batch_size=cli.batch_size,
        drift_periods=cli.drift_periods)


def plot(res, geometry, label, plane, nfp, RZ, a_eff, xlabel, path):
    R, Z, aR, aZ, cR, cZ, lr, lth = RZ
    keep = ~(res["escaped"] | ~res["ok"])
    offset = float(np.hypot(aR.mean() - cR, aZ.mean() - cZ))
    render_section(
        R, Z, res["iota"], res["resid"], res["seeds"][:, 0], keep,
        title=f"{geometry}  |  {label}  |  $\\zeta = {plane:g}$\n"
              f"{R.shape[1]} crossings/line",
        subtitle=f"nfp = {nfp}   |   h/2 drift {res['drift']:.1e} over "
                 f"{res['drift_lines']} regular lines   |   "
                 f"axis offset {offset:.2e}",
        axis_RZ=(aR, aZ), path=path, profile_x=a_eff,
        profile_xlabel=xlabel, nfp=nfp,
        logical=(lr, lth), chaotic=res["chaotic"])
    return offset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x", choices=sorted(NFP))
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--fields", default="k2,k1")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--r-min", type=float, default=0.03)
    ap.add_argument("--r-max", type=float, default=0.97)
    ap.add_argument("--periods", type=int, default=150)
    ap.add_argument("--steps", type=int, default=24,
                    help="prescribed steps per field period")
    ap.add_argument("--saves", type=int, default=8,
                    help="samples kept per period; must divide --steps")
    ap.add_argument("--planes", default=None,
                    help="logical zeta of each section, comma separated. "
                         "Mutually exclusive with --n-planes")
    ap.add_argument("--n-planes", type=int, default=None,
                    help="number of EVENLY SPACED sections over one field "
                         "period: zeta = k/N for k in 0..N-1. Any N; use this "
                         "instead of --planes when the positions do not matter")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--bench", action="store_true",
                    help="time the prescribed schedule against the adaptive one")
    ap.add_argument("--bench-periods", type=int, default=20)
    ap.add_argument("--profile-x", default="midplane", choices=SURFACE_LABELS,
                    help="surface label on the iota profile: 'midplane'"
                         "distance from the axis on the outboard midplane "
                         "(physical AND monotone by nesting -- the default), "
                         "'mean' distance over all crossings, 'area' "
                         "sqrt(A/pi), or 'seed' radius (monotone by "
                         "construction but map-dependent)")
    ap.add_argument("--seed-from", default="coord", choices=("coord", "axis"),
                    help="'coord' walks out from the logical r=0; 'axis' from "
                         "the MAGNETIC axis, which matters whenever the two "
                         "differ (finite-beta maps: 4.9 cm on w7x-ini) and "
                         "leaves a hole in the section otherwise")
    ap.add_argument("--drift-periods", type=int, default=64,
                    help="periods over which the h vs h/2 step check runs")
    ap.add_argument("--maxiter", type=int, default=200000,
                    help="inner-solve iteration budget; the nullspace chain "
                         "has no convergence gate, so a budget that is too "
                         "small returns a non-harmonic form in silence")
    ap.add_argument("--out", default="outputs/poincare")
    cli = ap.parse_args()

    if cli.planes is not None and cli.n_planes is not None:
        ap.error("--planes and --n-planes are mutually exclusive: pass the "
                 "positions or the count, not both")
    if cli.n_planes is not None:
        if cli.n_planes < 1:
            ap.error(f"--n-planes must be >= 1 (got {cli.n_planes})")
        planes = [k / cli.n_planes for k in range(cli.n_planes)]
    else:
        planes = [float(v) for v in (cli.planes or "0").split(",")]

    # BEFORE compute_nullspaces, which is where a schur.outer fallback would
    # fire: promoted here so a silent downgrade cannot survive the run.
    warnings.simplefilter("error", RuntimeWarning)

    ns = tuple(int(v) for v in cli.ns.split(","))
    nfp = NFP[cli.geometry]
    os.makedirs(cli.out, exist_ok=True)

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    op.warm_mass_preconditioner_cache(seq, ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} maxiter={cli.maxiter} "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    report_preconditioners(seq, ops)
    harmonic = report_harmonic(seq, ops)

    angle, angle_med, angle_p90 = field_agreement(seq, ops)
    print(f"[check] angle between the k=2 and k=1 harmonic fields: "
          f"max {angle:.3e}  p90 {angle_p90:.3e}  median {angle_med:.3e} rad",
          flush=True)

    summary = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "nfp": nfp, "periods": cli.periods, "steps": cli.steps,
               "saves": cli.saves, "seeds": cli.seeds,
               "field_angle_rad": angle,
               "field_angle_median_rad": angle_med,
               "field_angle_p90_rad": angle_p90, "maxiter": cli.maxiter,
               "harmonic": harmonic, "fields": {}}

    for name in cli.fields.split(","):
        k, dirichlet, label = FIELDS[name]
        dof = ops.null_2_dbc[0] if k == 2 else ops.null_1[0]
        if cli.bench:
            summary["bench_" + name] = bench(
                logical_field(seq, dof, k, dirichlet),
                seed_line(cli.seeds, cli.r_min, cli.r_max), cli)

        zeta = zeta_component_report(
            logical_field(seq, dof, k, dirichlet), name)

        res = run_field(seq, dof, k, dirichlet, nfp, cli)

        keep = ~(res["escaped"] | ~res["ok"])
        # A short trace has no converged winding for ANY line -- the half-vs-
        # half test needs length -- and an empty reduction here would report
        # that as a crash instead of as the trace being too short.
        iota = res["iota"][keep & ~res["chaotic"]]
        span = (f"iota {float(iota.min()):.4f}..{float(iota.max()):.4f}"
                if iota.size else "iota: no line converged")
        print(f"[{name}] {label}: {res['walltime']:.1f}s trace, "
              f"{int((~keep).sum())}/{cli.seeds} lost, "
              f"{int((keep & res['chaotic']).sum())} chaotic, "
              f"step drift {res['drift']:.2e}, {span}", flush=True)

        # (R, Z) goes into the archive alongside (u, v): re-deriving it needs
        # the map, and rebuilding the map is the expensive half of this script.
        sections, offsets, a_eff0 = {}, {}, None
        for plane in planes:
            path = os.path.join(
                cli.out, f"poincare_{cli.geometry}_{name}_zeta{plane:g}.png")
            R, Z, aR, aZ, cR, cZ, lr, lth = section_RZ(
                seq, res["ys"], res["axis"], cli.saves, plane)
            # a_eff is the map-INDEPENDENT surface label: the seed radius names
            # a different surface as soon as the map changes, which is exactly
            # what a resolution sweep and an interior perturbation both do.
            # Two labels are archived beside the chosen one so that a re-render
            # can change its mind without paying for the map again.
            seed_r = res["seeds"][:, 0]
            a_area = surface_label("area", R, Z, aR, aZ, seed_r)[0]
            a_mid = surface_label("midplane", R, Z, aR, aZ, seed_r)[0]
            a_eff, xlabel = surface_label(cli.profile_x, R, Z, aR, aZ, seed_r)
            off = plot(res, cli.geometry, label, plane, nfp,
                       (R, Z, aR, aZ, cR, cZ, lr, lth), a_eff, xlabel, path)
            offsets[f"zeta{plane:g}"] = off
            print(f"        -> {path}  (axis offset {off:.3e} m)", flush=True)
            for key, arr in zip(("R", "Z", "axisR", "axisZ", "logr", "logth"),
                                (R, Z, aR, aZ, lr, lth)):
                sections[f"{key}_zeta{plane:g}"] = arr
            sections[f"a_eff_zeta{plane:g}"] = a_eff
            sections[f"a_area_zeta{plane:g}"] = a_area
            sections[f"a_mid_zeta{plane:g}"] = a_mid
            sections[f"coordaxis_zeta{plane:g}"] = np.array([cR, cZ])
            if a_eff0 is None:
                a_eff0 = a_eff

        np.savez_compressed(
            os.path.join(cli.out, f"trace_{cli.geometry}_{name}.npz"),
            ys=res["ys"], iota=res["iota"], resid=res["resid"],
            seeds=res["seeds"], escaped=res["escaped"], ok=res["ok"],
            axis=res["axis"], chaotic=res["chaotic"], nfp=nfp,
            saves_per_period=cli.saves, label=label, **sections)
        summary["fields"][name] = {
            "walltime_s": res["walltime"], "step_drift": res["drift"],
            "lost": int((~keep).sum()),
            "chaotic": int((keep & res["chaotic"]).sum()),
            "iota_min": float(iota.min()), "iota_max": float(iota.max()),
            "resid_max": float(res["resid"][keep].max()),
            "axis_offset_m": offsets, "zeta_component": zeta,
            "a_eff_max": float(a_eff0[keep].max()),
        }

    with open(os.path.join(cli.out, f"summary_{cli.geometry}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
