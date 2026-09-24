"""Part A of the Landreman verification: the exact finite-beta equilibrium put into MRX, no relaxation.

For each rung ``(n_r, n_theta, n_zeta, p)``: the sequence of the Landreman wout (scripts/landreman_wout.py; the map
is the production L2 projection of its series, the field the production IC ``B = dA'`` of its Clebsch data,
:func:`mrx.initial_conditions.initial_field`) and at the quadrature points

* ``B_err``: ``||B_h - B_exact|| / ||B_exact||`` in L2 over the discrete domain, ``B_h = DF B_hat / J`` (Piola,
  the metric factors explicit) at the physical point ``F_h(x)`` and ``B_exact`` the closed form THERE;
* ``map_err``: ``||F_h - F_exact||`` over the logical points, relative to the minor radius ``sqrt(V / (2 pi^2 R))``
  (L2 and max);
* ``resid``: the squared normalised force residual ``||F||_M^2 / ||grad(B^2/2)||^2`` of relax()'s start sample
  (:func:`mrx.relaxation.compute_force`, :func:`mrx.relaxation.force_scale`), and ``||J|| / ||B||``;
* ``beta_vol`` (:func:`mrx.relaxation.pressure_diagnostics`) against the exact ``beta_V``, and ``p_err``: the
  weak pressure ``p_w`` (units restored with the IC's norm) against the closed-form ``p`` in L2;
* the flux-ratio iota ``nfp <B_hat^theta> / <B_hat^zeta>`` on every radial quadrature layer against the file's.

One GPU job runs all rungs of one family; ``--make-wout`` writes the wout first (numpy, a few minutes). Usage::

    python -u scripts/landreman_verify.py --case iota2 --wout OUT/wout_landreman_iota2.nc --make-wout \
        --rungs 8,2 12,2 16,2 24,2 8,3 12,3 16,3 24,3 --out OUT

A rung ``n,p`` means ``ns = (n, 2n, 2n)``. ``--plot OUT`` (login node, matplotlib only) merges ``OUT/rung_*/result.json``
into ``OUT/convergence.json`` and ``convergence.png`` with the fitted rates against ``h = 1 / (n_r - p)``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compare(seq, B, B_norm_raw, case, exact, p_w=None, p_strong=None):
    """The field ``B_norm_raw * B`` (2-form DoFs) against the closed form at the quadrature points: ``B_err``,
    ``B_err_max_rel``, ``map_err_l2``, ``map_err_max``, the flux-ratio iota per radial layer against the file's
    and, with the weak pressure ``p_w`` of ``B``, ``p_err`` and ``p_err_centered`` (volume means removed), with the
    Leray multiplier ``p_strong`` its ``p_strong_err_centered``."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    import landreman_equilibria as le
    from mrx.geometry import map_jacobian_at
    from mrx.vmec import profile_spline

    odd, even = seq.odd, seq.even
    res = {}
    xq = np.asarray(even.quad.x)
    wq = np.asarray(even.quad.w)
    Jq = np.asarray(even.jacobian_j)
    DF = np.asarray(map_jacobian_at(seq.map, jnp.asarray(xq)))
    Xh = np.asarray(jax.vmap(seq.map)(jnp.asarray(xq)))
    Bref = np.asarray(odd.evaluate_at_quadrature(B, 2, True)) * B_norm_raw
    Bh = np.einsum("qij,qj->qi", DF, Bref) / Jq[:, None]
    Bex = le.field(case, Xh)
    wJ = wq * Jq
    nrm = lambda f: float(np.sqrt(np.sum(wJ * np.sum(f * f, axis=-1))))     # noqa: E731
    res["B_err"] = nrm(Bh - Bex) / nrm(Bex)
    res["B_err_max_rel"] = float(np.max(np.linalg.norm(Bh - Bex, axis=1)) / np.max(np.linalg.norm(Bex, axis=1)))
    Xex = le.point(case, xq[:, 0], xq[:, 1], le.TWO_PI * xq[:, 2] / le.NFP)
    a_minor = np.sqrt(exact["volume"] / (2 * np.pi ** 2 * np.mean(np.hypot(Xex[:, 0], Xex[:, 1]))))
    dX = np.linalg.norm(Xh - Xex, axis=1)
    res["map_err_l2"] = float(np.sqrt(np.sum(wq * dX ** 2) / np.sum(wq))) / a_minor
    res["map_err_max"] = float(dX.max()) / a_minor
    if p_w is not None:
        pw_q = np.asarray(even.evaluate_at_quadrature(p_w, 0, True))[:, 0] * B_norm_raw ** 2
        pex = le.pressure(case, Xh)
        res["p_err"] = float(np.sqrt(np.sum(wJ * (pw_q - pex) ** 2) / np.sum(wJ * pex ** 2)))
        # the same with the volume means removed: a wall-layer error of p_w shifts it by a constant
        mean = lambda f: np.sum(wJ * f) / np.sum(wJ)        # noqa: E731
        dpc, pc = (pw_q - mean(pw_q)) - (pex - mean(pex)), pex - mean(pex)
        res["p_err_centered"] = float(np.sqrt(np.sum(wJ * dpc ** 2) / np.sum(wJ * pc ** 2)))
    if p_strong is not None:
        # the Leray multiplier, a 3-form: p / J at the points, defined up to a constant
        ps_q = np.asarray(even.evaluate_at_quadrature(p_strong, 3, True))[:, 0] / Jq * B_norm_raw ** 2
        pex = le.pressure(case, Xh)
        mean = lambda f: np.sum(wJ * f) / np.sum(wJ)        # noqa: E731
        dpc, pc = (ps_q - mean(ps_q)) - (pex - mean(pex)), pex - mean(pex)
        res["p_strong_err_centered"] = float(np.sqrt(np.sum(wJ * dpc ** 2) / np.sum(wJ * pc ** 2)))
    res["beta_exact"] = exact["beta_V"]

    # --- flux-ratio iota per radial quadrature layer ----------------------
    rho_l, inv = np.unique(xq[:, 0], return_inverse=True)
    num = np.bincount(inv, weights=wq * Bref[:, 1])
    den = np.bincount(inv, weights=wq * Bref[:, 2])
    iota_h = le.NFP * num / den
    st = seq.equilibrium
    iota_file = profile_spline(st, "iota")(rho_l)
    res["iota_rho"] = rho_l.tolist()
    res["iota_h"] = iota_h.tolist()
    res["iota_file"] = np.asarray(iota_file).tolist()
    res["iota_err_max"] = float(np.max(np.abs(iota_h - iota_file)))
    return res


def run_rung(case_name, wout, n, p, out, exact):
    import jax.numpy as jnp
    import numpy as np

    import landreman_equilibria as le
    import mrx
    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import (compute_divergence_norm, compute_force, force_scale_jit, pressure_diagnostics,
                                weak_pressure)

    case = le.CASES[case_name]
    ns = (n, 2 * n, 2 * n)
    tag = f"rung_{n}x{2 * n}x{2 * n}_p{p}"
    d = os.path.join(out, tag)
    os.makedirs(d, exist_ok=True)
    res = dict(case=case_name, ns=list(ns), p=p, h=1.0 / (n - p), precision=str(mrx.DTYPE), wout=os.path.abspath(wout))
    t0 = time.perf_counter()
    seq, _ = build_sequence(wout, ns, p, symmetry="stellarator")
    compute_nullspaces(seq)
    res["t_setup"] = time.perf_counter() - t0
    B, ic = initial_field(seq)
    res["ic"] = ic
    odd, even = seq.odd, seq.even

    # --- force residual, weak pressure, beta: relax()'s start sample -------
    F, pr, J, X, JxX = compute_force(B, seq, False)
    F_norm = float(jnp.sqrt(F @ even.apply_mass_matrix(F, 2)))
    res["resid"] = float((F_norm / force_scale_jit(seq, B)) ** 2)
    res["JoverB"] = float(odd.l2_norm(J, 1) / odd.l2_norm(B, 2))
    res["div"] = float(compute_divergence_norm(B, seq))
    p_w, F_w, v = weak_pressure(J, X, seq, False)
    diag = pressure_diagnostics(B, pr, p_w, F_w, v, seq)
    res.update({k: float(val) for k, val in diag.items()})

    res.update(compare(seq, B, ic["B_norm_raw"], case, exact, p_w=p_w, p_strong=pr))
    res["beta_vol_rel_err"] = res["beta_vol"] / exact["beta_V"] - 1.0
    iota_h = np.asarray(res["iota_h"])
    res["t_total"] = time.perf_counter() - t0
    _log(f"{tag}: B_err {res['B_err']:.3e} (max {res['B_err_max_rel']:.2e})  map_err {res['map_err_l2']:.2e} "
         f"(max {res['map_err_max']:.2e})  resid {res['resid']:.3e}  J/B {res['JoverB']:.4f}  beta_vol "
         f"{res['beta_vol']:.5e} vs {exact['beta_V']:.5e} ({res['beta_vol_rel_err']:+.2e})  p_err {res['p_err']:.2e}  "
         f"iota {iota_h.min():.6f}..{iota_h.max():.6f} (|err| <= {res['iota_err_max']:.1e})  div {res['div']:.1e}  "
         f"{res['t_total']:.0f}s")
    with open(os.path.join(d, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "rung_*", "result.json")))]
    table = {}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for p in sorted({r["p"] for r in rows}):
        rs = sorted((r for r in rows if r["p"] == p), key=lambda r: r["h"])
        h = np.array([r["h"] for r in rs])
        cols = dict(B_err=[r["B_err"] for r in rs], map_err=[r["map_err_l2"] for r in rs],
                    sqrt_resid=[r["resid"] ** 0.5 for r in rs],
                    beta_err=[abs(r["beta_vol_rel_err"]) for r in rs], p_err=[r["p_err"] for r in rs],
                    iota_err=[r["iota_err_max"] for r in rs])
        rates = {k: (np.diff(np.log(v)) / np.diff(np.log(h))).tolist() for k, v in cols.items()}
        table[f"p{p}"] = dict(ns=[r["ns"] for r in rs], h=h.tolist(), **cols,
                              resid=[r["resid"] for r in rs], beta_vol=[r["beta_vol"] for r in rs],
                              JoverB=[r["JoverB"] for r in rs], rates=rates)
        for ax, k in zip(axs, ("B_err", "sqrt_resid", "beta_err")):
            ax.loglog(h, cols[k], "o-", label=f"p={p}")
            for i in range(len(h) - 1):
                ax.annotate(f"{rates[k][i]:.1f}", (np.sqrt(h[i] * h[i + 1]), np.sqrt(cols[k][i] * cols[k][i + 1])),
                            fontsize=8)
    for ax, t in zip(axs, (r"$\|B_h-B\|/\|B\|$", r"$\|F\|_{\rm norm}$", r"$|\beta_h/\beta-1|$")):
        ax.set_xlabel("h = 1/(n_r - p)")
        ax.set_title(t)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergence.png"), dpi=150)
    with open(os.path.join(out, "convergence.json"), "w") as fh:
        json.dump(table, fh, indent=1)
    print(json.dumps(table, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=("iota2", "sheared", "shearedA"))
    ap.add_argument("--wout")
    ap.add_argument("--make-wout", action="store_true")
    ap.add_argument("--wout-ns", type=int, default=201)
    ap.add_argument("--rungs", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--map-batch", type=int, default=8192,
                    help="cells per batch of the quadrature loops (mrx.MAP_BATCH_SIZE_INNER): the lambda series has "
                         "~2000 modes, evaluated at every histopolation point")
    cli = ap.parse_args(argv)
    if cli.plot:
        return plot(cli.plot)
    import mrx
    mrx.MAP_BATCH_SIZE_INNER = cli.map_batch
    import landreman_check
    import landreman_equilibria as le
    os.makedirs(cli.out, exist_ok=True)
    if cli.make_wout:
        import landreman_wout
        landreman_wout.main(["--case", cli.case, "--ns", str(cli.wout_ns), "--out", cli.wout])
    case = le.CASES[cli.case]
    chk = landreman_check.check(case)
    exact = dict(beta_V=chk["beta_V"], volume=chk["volume"])
    _log(f"exact {cli.case}: beta_V {exact['beta_V']:.6e}, volume {exact['volume']:.6e}")
    for spec in cli.rungs:
        n, p = (int(v) for v in spec.split(","))
        run_rung(cli.case, cli.wout, n, p, cli.out, exact)


if __name__ == "__main__":
    main()
