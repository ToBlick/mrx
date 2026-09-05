"""Measure the direct series-to-spline map constructions against the series.

For one equilibrium file (GVEC state ``.dat`` or VMEC wout ``.nc``) and one
mesh ``(ns, p)`` this builds the polar spline map of ``R`` and ``Z`` two
ways -- sampling the series at the Greville points and solving (the
2026-08-27 route, ``seq.interpolate``; its per-mode closed form, measured
identical to round-off, was deleted with the study) and the per-mode
closed-form L2 projection production uses (:func:`mrx.gvec.series_spline_dofs`)
-- and reports, against the series itself: the coefficient differences,
the map and Jacobian errors on random points, the axis behaviour of
``det DF / rho``, and for the routes named in ``--routes`` the full
pipeline (geometry, preconditioners, harmonic forms, Clebsch initial
condition, force residual, and for a vacuum file the distance to the
harmonic form). ``docs/research/analytic_map_2026-08-28.md`` records the
results, including the interpolant's.
"""
import argparse
import os
import time

# A study of the map projection error: float64 unless asked otherwise (the
# package default is float32 since 2026-09-04).
os.environ.setdefault("MRX_DTYPE", "float64")

import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.gvec import StateField, _det_DF, _map_with_sign, load_clebsch, read_state, series_spline_dofs
from mrx.initial_conditions import clebsch_potential_form, potential_two_form
from mrx.nullspace import compute_nullspaces
from mrx.relaxation import compute_divergence_norm, compute_force
from mrx.vmec import read_wout

TYPES = ("clamped", "periodic", "periodic")


def _sign_of(R_h, Z_h, nfp):
    for s in (1.0, -1.0):
        d = _det_DF(_map_with_sign(R_h, Z_h, nfp, s))
        if np.isfinite(d).all() and d.min() > 0:
            return s
    raise RuntimeError("no handedness gives det DF > 0")


def _map_errors(F, F_ref, pts):
    DF = jax.vmap(jax.jacfwd(F))(pts)
    DF_ref = jax.vmap(jax.jacfwd(F_ref))(pts)
    X, X_ref = jax.vmap(F)(pts), jax.vmap(F_ref)(pts)
    dX = np.linalg.norm(np.asarray(X - X_ref), axis=1)
    dDF = np.linalg.norm(np.asarray(DF - DF_ref).reshape(len(pts), -1), axis=1)
    nDF = np.sqrt(np.mean(np.linalg.norm(np.asarray(DF_ref).reshape(len(pts), -1), axis=1) ** 2))
    det, det_ref = np.asarray(jnp.linalg.det(DF)), np.asarray(jnp.linalg.det(DF_ref))
    return dict(x_max=dX.max(), x_rms=np.sqrt(np.mean(dX ** 2)),
                df_max=dDF.max() / nDF, df_rms=np.sqrt(np.mean(dDF ** 2)) / nDF,
                det_ratio=(float((det / det_ref).min()), float((det / det_ref).max())))


def _axis(F, rho, n_t=64, zeta=0.3):
    pts = jnp.array([[rho, t, zeta] for t in np.linspace(0, 1, n_t, endpoint=False)])
    det = np.asarray(jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(F)(x)))(pts)) / rho
    return det.min(), det.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--ns", type=int, nargs=3, default=(12, 24, 12))
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--routes", default="sampled,l2",
                    help="routes that run the full pipeline (comma list of sampled, l2; '' for none)")
    ap.add_argument("--n-pts", type=int, default=4000)
    args = ap.parse_args()
    ns, p, path = tuple(args.ns), args.p, args.geometry
    name = os.path.basename(path)
    print(f"[study] {name} ns={ns} p={p} dtype={mrx.DTYPE}", flush=True)

    t0 = time.time()
    seq = DeRhamSequence(ns, (p,) * 3, p + 1, TYPES, polar=True, betti_numbers=(1, 1, 0, 0))
    print(f"[study] DeRhamSequence {time.time() - t0:.1f} s", flush=True)

    st = read_wout(path) if path.endswith(".nc") else read_state(path)
    nfp, sp = st["nfp"], st.get("sp")
    m, n_per = st["X1"]["m"], st["X1"]["n"] / nfp
    amp = np.abs(st["X1"]["coef"]).max(axis=1)
    beyond = (np.abs(m) > ns[1] // 2) | (np.abs(n_per) > ns[2] // 2)
    print(f"[study] modes {len(m)}: m in [{m.min()}, {m.max()}], n/nfp in [{n_per.min():.0f}, {n_per.max():.0f}]; "
          f"beyond Nyquist of {ns[1:]}: {beyond.sum()} modes, max |R_mn| {amp[beyond].max() if beyond.any() else 0:.2e}, "
          f"sum {amp[beyond].sum():.2e} (m=1,n=0: {amp[(m == 1) & (n_per == 0)].max():.2e})", flush=True)
    R_fn = StateField(st["X1"], sp, nfp, vector=True)
    Z_fn = StateField(st["X2"], sp, nfp, vector=True)

    dofs, times = {}, {}
    t0 = time.time()
    dofs["sampled"] = (seq.interpolate(R_fn, 0), seq.interpolate(Z_fn, 0))
    times["sampled"] = time.time() - t0
    t0 = time.time()
    dofs["l2"] = (series_spline_dofs(st["X1"], sp, nfp, seq),
                  series_spline_dofs(st["X2"], sp, nfp, seq))
    times["l2"] = time.time() - t0
    scale = float(jnp.abs(dofs["sampled"][0]).max())
    dR = float(jnp.abs(dofs["sampled"][0] - dofs["l2"][0]).max()) / scale
    dZ = float(jnp.abs(dofs["sampled"][1] - dofs["l2"][1]).max()) / scale
    print(f"[study] max |dof sampled - dof l2| / max|R dof|: R {dR:.2e}, Z {dZ:.2e}", flush=True)
    for route in dofs:
        print(f"[study] {route}: coefficients in {times[route]:.2f} s", flush=True)

    maps = {}
    for route, (R_dof, Z_dof) in dofs.items():
        R_h = DiscreteFunction(R_dof, seq.basis_0, seq.E(0))
        Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.E(0))
        sign = _sign_of(R_h, Z_h, nfp)
        maps[route] = (_map_with_sign(R_h, Z_h, nfp, sign), sign)
    sign = maps["l2"][1]
    F_ref = _map_with_sign(R_fn, Z_fn, nfp, sign)
    rng = np.random.default_rng(0)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.05, 0.95, args.n_pts),
                                       rng.uniform(0, 1, args.n_pts), rng.uniform(0, 1, args.n_pts)]))
    print(f"[study] sign {sign:+.0f}; errors against the series on {args.n_pts} random points, rho in [0.05, 0.95]:")
    for route, (F, s) in maps.items():
        e = _map_errors(F, F_ref, pts)
        print(f"[study]   {route:8s} sign {s:+.0f}: |dX| max {e['x_max']:.3e} rms {e['x_rms']:.3e} m; "
              f"|dDF|/|DF| max {e['df_max']:.3e} rms {e['df_rms']:.3e}; det/det_ref in "
              f"[{e['det_ratio'][0]:.4f}, {e['det_ratio'][1]:.4f}]", flush=True)
    for rho in (1e-2, 1e-3, 1e-5):
        line = f"[study] det DF / rho at rho = {rho:.0e} over theta (min, max): series ({_axis(F_ref, rho)[0]:.4f}, {_axis(F_ref, rho)[1]:.4f})"
        for route, (F, _) in maps.items():
            lo, hi = _axis(F, rho)
            line += f"; {route} ({lo:.4f}, {hi:.4f})"
        print(line, flush=True)
    for rho in (1.0, 1.0 - 1e-9):
        x = jnp.array([rho, 0.3, 0.3])
        line = f"[study] det DF at rho = {rho!r}: series {float(jnp.linalg.det(jax.jacfwd(F_ref)(x))):.4e}"
        for route, (F, _) in maps.items():
            line += f"; {route} {float(jnp.linalg.det(jax.jacfwd(F)(x))):.4e}"
        print(line, flush=True)

    routes = [r for r in args.routes.split(",") if r]
    for route in routes:
        F, _ = maps[route]
        print(f"\n[study] ===== pipeline: {route} =====", flush=True)
        t0 = time.time()
        seq.set_map(F)
        jac = np.asarray(seq.jacobian_j)
        t1 = time.time()
        print(f"[study] set_map {t1 - t0:.1f} s; det DF at the quadrature points in [{jac.min():.4e}, {jac.max():.4e}]", flush=True)
        ops = seq.build_preconditioners()
        t2 = time.time()
        print(f"[study] build_preconditioners {t2 - t1:.1f} s", flush=True)
        seq.set_operators(compute_nullspaces(seq, ops))
        t3 = time.time()
        print(f"[study] compute_nullspaces {t3 - t2:.1f} s", flush=True)
        cb = load_clebsch(path)
        B, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
        div = float(compute_divergence_norm(B, seq))
        Fv, _, _, _, _ = compute_force(B, seq)
        F_norm = float(seq.l2_norm(Fv, 2))
        t4 = time.time()
        print(f"[study] IC {t4 - t3:.1f} s: ||B||_M raw {norm:.4e}, ||div B|| {div:.2e}, wall-normal discarded {wall:.2e}, "
              f"||F||_M {F_norm:.4e} at ||B||_M = 1", flush=True)
        if np.abs(cb["p"]).max() == 0.0:
            h = seq.nullspace(2, True)[0]
            h = h / seq.l2_norm(h, 2)
            Bn = B / seq.l2_norm(B, 2)
            err = float(min(seq.l2_norm(Bn - h, 2), seq.l2_norm(Bn + h, 2)))
            print(f"[study] vacuum: ||B_hat -+ h_hat||_M = {err:.4e}", flush=True)
        print(f"[study] RESULT {name} ns={ns} p={p} route={route} F={F_norm:.4e} "
              f"det=[{jac.min():.4e},{jac.max():.4e}] t_geom={t1 - t0:.0f} t_pc={t2 - t1:.0f} t_null={t3 - t2:.0f}", flush=True)


if __name__ == "__main__":
    main()
