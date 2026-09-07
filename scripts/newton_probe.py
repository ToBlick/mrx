"""Probe the second variation on one field: its identities, its spectrum, and the Newton direction's quality.

At the initial field of a geometry file or at a checkpoint's field
(``--restart``), one GPU job (see slurm/README.md)::

    python -u scripts/newton_probe.py --geometry data/wout_li383_1.4m.nc --ns 16,32,32 --p 2 --precision float64

Parts, each printed and written to ``<out>/probe.json``:

1. Identities of :func:`mrx.hessian.second_variation` on random
   divergence-free velocities: the first derivative of the energy along the
   flow against the force's pairing, the quadratic form against the second
   derivative along the second-order flow, symmetry, and the field's own
   direction ``u = B`` (``(B, H B) = 0`` always; ``H B`` itself vanishes
   only at an equilibrium).
2. The spectrum of the Hessian on divergence-free velocities in the L2
   metric, ``P M_2^-1 H``, by ``--lanczos`` steps of Lanczos with full
   reorthogonalisation: the extreme Ritz values, how many are negative, the
   condition number Newton's system would have with the exact projection.
3. The Newton direction (:func:`mrx.hessian.newton_direction`) for every
   ``--shifts`` (fractions of the largest Ritz value) and ``--preconds``,
   solved to 1e-1, 1e-2, 1e-3 in turn (warm-started, so the counts are
   cumulative), against the force and the smoothed force: the descent
   cosine, the line-search step ``dt*`` and its CFL cap, the energy the
   step removes along the first-order path (exact) and along the
   second-order path (evaluated), and the Newton model's prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--ns", default="16,32,32")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--solve-tol", type=float, default=None)
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    ap.add_argument("--restart", default=None, help="a checkpoints/state_<step>.h5: probe its field")
    ap.add_argument("--lanczos", type=int, default=150)
    ap.add_argument("--minres-maxiter", type=int, default=300)
    ap.add_argument("--shifts", default="0,1e-3,1e-2",
                    help="Levenberg-Marquardt shifts as fractions of the largest Ritz value")
    ap.add_argument("--preconds", default="laplacian,laplacian2")
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(cli):
    import h5py
    import jax
    import jax.numpy as jnp
    import numpy as np

    import mrx
    from mrx.geometry import build_sequence
    from mrx.hessian import newton_direction, second_variation
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import compute_force, force_scale, logical_cfl_weights, smoothing_scale

    out = cli.out or os.path.join("outputs", "newton_probe", time.strftime("%Y-%m-%d/%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    results = {"params": dict(vars(cli), out=out)}
    ns = tuple(int(v) for v in cli.ns.split(","))
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)

    t0 = time.perf_counter()
    seq, _ = build_sequence(cli.geometry, ns, cli.p, tol=cli.solve_tol)
    compute_nullspaces(seq)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} tol={seq.tol:.1e} n2={seq.n(2, True)} "
          f"n1={seq.n(1, True)}  {time.perf_counter() - t0:.0f}s", flush=True)
    if cli.restart:
        with h5py.File(cli.restart, "r") as fh:
            B = jnp.asarray(fh["B_n"][...], dtype=mrx.DTYPE)
            step = int(fh.attrs["step"])
        print(f"[field] {cli.restart} at step {step}", flush=True)
    else:
        B, ic = initial_field(seq)
        step = 0
        print(f"[field] {ic['kind']} initial field", flush=True)

    # --- the force, the Hessian, the helpers ------------------------------
    F, p, J, X, JxX = compute_force(B, seq)
    MF = seq.apply_mass_matrix(F, 2)
    F_norm = float(jnp.sqrt(F @ MF))
    scale = float(force_scale(seq)(B))
    E0 = 0.5 * float(seq.l2_norm_sq(B, 2))
    H = jax.jit(second_variation(seq, B, J))
    M2 = jax.jit(lambda v: seq.apply_mass_matrix(v, 2))
    cfl_w = logical_cfl_weights(seq)

    @jax.jit
    def curl_cross(u, Xf):
        """``curl(u x X)`` of 2-forms: the ideal increment."""
        E = seq.apply_inverse_mass_matrix(seq.cross_product_load(u, Xf, 1, 2, 2), 1)
        return seq.apply_incidence_matrix(E, 1)

    @jax.jit
    def leray_unit(w):
        w, _ = seq.apply_leray_projection(w, k=2)
        return w / seq.l2_norm(w, 2)

    @jax.jit
    def energy(Bf):
        return 0.5 * seq.l2_norm_sq(Bf, 2)

    results["field"] = dict(step=step, E=E0, F=F_norm, scale=scale, resid=F_norm / scale)
    print(f"[force] E={E0:.8e}  |F|={F_norm:.4e}  resid={F_norm / scale:.4e}", flush=True)

    # --- 1. identities ------------------------------------------------------
    t1 = time.perf_counter()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    u = leray_unit(jax.random.normal(k1, (seq.n(2, True),), dtype=mrx.DTYPE))
    v = leray_unit(jax.random.normal(k2, (seq.n(2, True),), dtype=mrx.DTYPE))
    Hu, Hv = H(u), H(v)
    Q = curl_cross(u, B)
    R = curl_cross(u, Q)
    dE = float(B @ M2(Q))
    force = -float(u @ M2(JxX))
    quad = float(u @ Hu)
    d2E = float(seq.l2_norm_sq(Q, 2) + B @ M2(R))
    cross, cross_t = float(u @ Hv), float(v @ Hu)
    Bn = B / seq.l2_norm(B, 2)
    HB = H(Bn)
    ident = dict(dE=dE, force=force, dE_rel=abs(dE - force) / abs(force),
                 quad=quad, d2E=d2E, quad_rel=abs(quad - d2E) / abs(d2E),
                 cross=cross, cross_t=cross_t, sym_rel=abs(cross - cross_t) / abs(cross),
                 BHB=float(Bn @ HB), HB_over_Hu=float(jnp.linalg.norm(HB) / jnp.linalg.norm(Hu)),
                 seconds=time.perf_counter() - t1)
    results["identities"] = ident
    print(f"[identities] dE {dE:+.6e} vs -(u,JxB) {force:+.6e} (rel {ident['dE_rel']:.1e});  "
          f"(u,Hu) {quad:+.6e} vs |Q|^2+(B,R) {d2E:+.6e} (rel {ident['quad_rel']:.1e});  "
          f"(u,Hv) {cross:+.6e} vs (v,Hu) {cross_t:+.6e} (rel {ident['sym_rel']:.1e});  "
          f"(B,HB)/|B|^2 {ident['BHB']:+.2e}  |HB|/|Hu| {ident['HB_over_Hu']:.2e}  "
          f"[{ident['seconds']:.0f}s]", flush=True)

    # --- 2. Lanczos on P M_2^-1 H (M_2 inner product, div-free vectors) -----
    t2 = time.perf_counter()

    @jax.jit
    def T(w):
        y = seq.apply_inverse_mass_matrix(H(w), 2)
        y, _ = seq.apply_leray_projection(y, k=2)
        return y

    N = cli.lanczos
    V = np.zeros((N + 1, seq.n(2, True)))
    MV = np.zeros_like(V)
    V[0] = np.asarray(u, dtype=np.float64)
    MV[0] = np.asarray(M2(u), dtype=np.float64)
    alpha, beta = np.zeros(N), np.zeros(N)
    for j in range(N):
        w = np.array(T(jnp.asarray(V[j], dtype=mrx.DTYPE)), dtype=np.float64)   # a copy: the view is read-only
        alpha[j] = w @ MV[j]
        w -= alpha[j] * V[j]
        if j:
            w -= beta[j - 1] * V[j - 1]
        for _ in range(2):                       # full reorthogonalisation, twice
            w -= V[:j + 1].T @ (MV[:j + 1] @ w)
        Mw = np.asarray(M2(jnp.asarray(w, dtype=mrx.DTYPE)), dtype=np.float64)
        beta[j] = np.sqrt(w @ Mw)
        V[j + 1], MV[j + 1] = w / beta[j], Mw / beta[j]
        if (j + 1) % 25 == 0:
            th = np.linalg.eigvalsh(np.diag(alpha[:j + 1]) + np.diag(beta[:j], 1) + np.diag(beta[:j], -1))
            print(f"  lanczos {j + 1:4d}: theta min {th[0]:+.4e}  max {th[-1]:+.4e}  "
                  f"negative {int((th < 0).sum())}  beta {beta[j]:.3e}  [{time.perf_counter() - t2:.0f}s]",
                  flush=True)
    theta = np.linalg.eigvalsh(np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1))
    pos = theta[theta > 0]
    results["lanczos"] = dict(n=N, theta=theta.tolist(), theta_max=float(theta[-1]),
                              theta_min=float(theta[0]), negative=int((theta < 0).sum()),
                              smallest_positive=float(pos.min()) if pos.size else None,
                              seconds=time.perf_counter() - t2)
    print(f"[lanczos] {N} steps: theta in [{theta[0]:+.4e}, {theta[-1]:+.4e}], {int((theta < 0).sum())} "
          f"negative, smallest positive {pos.min() if pos.size else float('nan'):.4e}, "
          f"max/smallest-positive {theta[-1] / pos.min() if pos.size else float('nan'):.2e}  "
          f"[{time.perf_counter() - t2:.0f}s]\n  smallest: {np.array2string(theta[:8], precision=3)}\n"
          f"  largest:  {np.array2string(theta[-8:], precision=3)}", flush=True)
    theta_max = float(theta[-1])

    # --- 3. directions ------------------------------------------------------
    def quality(uu, label, cost):
        """The line search along ``uu`` and the energy it removes."""
        Fu = float(uu @ MF)
        un = float(seq.l2_norm(uu, 2))
        Qu = curl_cross(uu, B)
        Q2 = float(seq.l2_norm_sq(Qu, 2))
        dt_star = Fu / Q2
        cfl_max = float(jnp.max(jnp.abs(seq.evaluate_at_quadrature(uu, 2, True)) * cfl_w))
        dt = min(dt_star, cli.cfl / cfl_max) if dt_star > 0 else 0.0
        dE_first = -dt * Fu + 0.5 * dt ** 2 * Q2
        Ru = curl_cross(uu, Qu)
        dE_second = float(energy(B + dt * Qu + 0.5 * dt ** 2 * Ru)) - E0
        Huu = float(uu @ H(uu))
        dE_model = -dt * Fu + 0.5 * dt ** 2 * Huu
        q = dict(label=label, cost=cost, cos=Fu / (un * F_norm), dt_star=dt_star, dt=dt,
                 cfl_bound=dt < dt_star, dE_first=dE_first, dE_second=dE_second, dE_model=dE_model,
                 dE_star=-0.5 * Fu ** 2 / Q2, uHu_over_Q2=Huu / Q2)
        print(f"  {label:<34s} cost {cost:4d}  cos {q['cos']:+.4f}  dt* {dt_star:.3e}  dt {dt:.3e}"
              f"{' (CFL)' if q['cfl_bound'] else '      '}  dE first {dE_first:+.3e}  second {dE_second:+.3e}"
              f"  model {dE_model:+.3e}  dE* {q['dE_star']:+.3e}  uHu/|Q|^2 {q['uHu_over_Q2']:+.3f}",
              flush=True)
        return q

    print("[directions] cost = MINRES iterations (3 k=1 mass solves each); dE* is the line-search "
          "minimum along the first-order path", flush=True)
    dirs = [quality(F, "force F", 0)]
    mu = smoothing_scale(seq)
    Fs = seq.apply_inverse_mass_plus_eps_laplace_matrix(MF, 2, mu, dirichlet=True)
    dirs.append(quality(Fs, f"smoothed force (mu {mu:.1e})", 0))
    shifts = [float(s) for s in cli.shifts.split(",")]
    for pc in cli.preconds.split(","):
        for frac in shifts:
            shift = frac * theta_max
            a = jnp.zeros(seq.n(1, True), dtype=mrx.DTYPE)
            total = 0
            for tol in (1e-1, 1e-2, 1e-3):
                t3 = time.perf_counter()
                un_, a, info = newton_direction(seq, B, J, MF, a, shift=shift, tol=tol,
                                                maxiter=cli.minres_maxiter, precond=pc)
                total += abs(int(info))
                q = quality(un_, f"newton {pc} shift {frac:g} tol {tol:g}", total)
                q.update(shift=shift, precond=pc, tol=tol, info=int(info),
                         seconds=time.perf_counter() - t3)
                dirs.append(q)
                if int(info) > 0:
                    print(f"    (not converged in {cli.minres_maxiter}: stopping this arm)", flush=True)
                    break
    results["directions"] = dirs
    with open(os.path.join(out, "probe.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {out}/probe.json", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"] = cli.precision
    main(cli)
