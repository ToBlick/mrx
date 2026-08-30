"""Convergence of a VMEC vacuum equilibrium against MRX's discrete harmonic field.

A vacuum equilibrium (``presf == 0``) is the harmonic k=2 Dirichlet form of
its own boundary: curl-free, divergence-free, ``B . n = 0``, carrying the
toroidal flux. This script compares, at one resolution ``(ns, p)``,

* ``B_w``: the wout field pushed into ``V_2^h`` by the production IC route
  (``load_clebsch`` -> ``clebsch_potential_form`` -> ``potential_two_form``,
  i.e. the commuting projection ``Pi_2 B_VMEC``: exactly divergence-free,
  toroidal flux ``phi_edge`` exactly, Tesla per field period), and
* ``h``: the discrete harmonic form ``seq.nullspace(2, True)[0]`` of the
  direct Hodge construction (``mrx.nullspace.compute_nullspaces``),
  M-normalised, sign arbitrary.

Same-space comparison (no transfer): the L2(M)-optimal scale
``c = <B_w, h>_M / <h, h>_M`` and ``D = ||B_w - c h||_M / ||B_w||_M``, the sine
of the M-angle between the two fields. ``D`` falls like ``h^p`` while the
discretisation dominates and floors at VMEC's own distance from the vacuum
field of its boundary (Fourier truncation, radial mesh, the lambda refit).
The flux-matched scale ``c_flux = Phi(B_w) / Phi(h)`` is the cross-check:
``c_flux / c - 1`` vanishes iff VMEC's field is harmonic.

Cross-resolution comparison: the spaces are not nested (clamped radial knots
have ``n_r - p`` uniform elements; the map is refit at every rung), so both
fields are pushed forward to Cartesian components on a fixed logical grid
(``--grid``: Gauss nodes in rho on (0, 1), midpoints in the angles) and
stored; ``--plot`` compares every rung with the finest one,
``E = ||B_h - B_ref|| / ||B_ref||`` in the grid's physical L2 norm, with the
map difference ``|F_h - F_ref|`` reported alongside (one order higher). The
element size is ``h = 1 / (n_r - p)``.

Gates per rung (float64): ``||div B||`` of the IC, the wall-normal part it
discards, and the harmonic form's Rayleigh quotient against ``lambda_1``
(``ratio <= 1e-10``; the construction has no gate of its own and an
unconverged Hodge solve returns a non-harmonic vector silently).

Usage (one process = one rung; always a GPU job through ``slurm/run.sh``)::

    python -u scripts/vacuum_convergence.py --geometry data/wout_X.nc \
        --ns 16,32,16 --p 3 --out outputs/qa_vacuum [--trace] [--h5 run/B.h5]
    python -u scripts/vacuum_convergence.py --plot outputs/qa_vacuum

``--precision`` is exported as ``MRX_DTYPE`` before ``mrx`` is imported.
Output per rung: ``<out>/rung_<nr>x<nt>x<nz>_p<p>/result.json`` and
``fields.npz`` (the DoFs of both fields and their Cartesian values, the
positions and ``det DF`` on the common grid). ``--plot`` writes
``<out>/convergence.json``, ``<out>/convergence.png`` (log-log error vs h,
slopes annotated; ``||F||_M(h)`` is tabled only, it sits at the solver floor) and ``<out>/residual_zeta0.png`` (``|B_w - c h|`` on the
zeta = 0 section of the finest rung).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", default="data/wout_LandremanPaul2021_QA_lowres.nc")
    ap.add_argument("--ns", default=None, help="n_r,n_theta,n_zeta of the rung")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=10_000)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--precision", default="float64", choices=("float32", "float64"))
    ap.add_argument("--grid", default="48,96,48",
                    help="common evaluation grid: Gauss nodes in rho, midpoints in theta, zeta")
    ap.add_argument("--h5", default=None,
                    help="a relax.py B.h5 at this rung: check B_ic against B_w, report B_final")
    ap.add_argument("--trace", action="store_true",
                    help="field-line iota of B_w and h against the file's iotaf")
    ap.add_argument("--trace-seeds", type=int, default=40)
    ap.add_argument("--trace-periods", type=int, default=200)
    ap.add_argument("--out", default="outputs/qa_vacuum")
    ap.add_argument("--plot", default=None, help="merge the rungs under this directory and plot")
    ap.add_argument("--regrid", default=None,
                    help="rung dir: re-evaluate its stored DoFs on --grid and rewrite "
                         "fields.npz (rebuilds only the geometry, no nullspace solve). Use "
                         "when --grid must be raised to out-resolve the finest rung.")
    return ap.parse_args(argv)


def _log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# One rung
# ---------------------------------------------------------------------------

def common_grid(shape):
    """Logical points ``(N, 3)`` and weights ``(N,)`` of the fixed evaluation
    grid: Gauss-Legendre nodes in rho on (0, 1) (never rho = 1, where the
    spline map's ``det DF`` is an autodiff zero), midpoint rules in the
    periodic angles. ``sum w J f`` is the volume integral over one field
    period, the same domain as the M-norm."""
    import numpy as np
    nr, nt, nz = shape
    xg, wg = np.polynomial.legendre.leggauss(nr)
    rho, w_r = 0.5 * (xg + 1.0), 0.5 * wg
    th = (np.arange(nt) + 0.5) / nt
    ze = (np.arange(nz) + 0.5) / nz
    R, T, Z = np.meshgrid(rho, th, ze, indexing="ij")
    W = np.broadcast_to(w_r[:, None, None] / (nt * nz), R.shape)
    pts = np.stack([R.ravel(), T.ravel(), Z.ravel()], axis=1)
    return pts, W.ravel().copy(), dict(rho=rho, theta=th, zeta=ze)


def pushforward_on_grid(seq, dof, pts, batch=4096, k=2, dirichlet=True):
    """Cartesian components of the ``k``-form ``dof`` at the logical points
    ``pts``, with ``F(pts)`` and ``J = det DF``.

    Piola for the 2-form (``B = DF B_hat / J``) and the covariant rule for the
    1-form (``a = (DF^T)^{-1} a_hat``); the metric factor is explicit in both.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mrx.differential_forms import DiscreteFunction, det33, inv33

    basis = {1: seq.basis_1, 2: seq.basis_2}[k]
    f = DiscreteFunction(jnp.asarray(dof), basis, seq.E(k, dirichlet))
    F = seq.map

    @jax.jit
    def one(x):
        DF = jax.jacfwd(F)(x)
        J = det33(DF)
        v = DF @ f(x) / J if k == 2 else inv33(DF).T @ f(x)
        return v, F(x), J

    many = jax.jit(jax.vmap(one))
    B, X, J = [], [], []
    for i in range(0, len(pts), batch):
        b, x, j = many(jnp.asarray(pts[i:i + batch]))
        B.append(np.asarray(b))
        X.append(np.asarray(x))
        J.append(np.asarray(j))
    return np.concatenate(B), np.concatenate(X), np.concatenate(J)


def toroidal_flux(seq, dof):
    """``Phi = sum_q w_q B_hat^zeta(q)``: the flux through a zeta = const
    section (zeta-independent for a div-free field with ``B . n = 0``, so
    the volume sum over one field period of logical zeta in [0, 1) is it)."""
    import jax.numpy as jnp
    vals = seq.evaluate_at_quadrature(dof, 2, True)
    return float(jnp.sum(seq.quad.w * vals[:, 2]))


def trace_iota(seq, dof, nfp, n_seeds, n_periods, tag):
    """Field-line iota per seed radius, via :func:`mrx.poincare.section_figure`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mrx.poincare import section_figure

    t0 = time.perf_counter()
    fig, res = section_figure(seq, dof, nfp, plane=0.0, n_seeds=n_seeds,
                              n_periods=n_periods, title=tag)
    plt.close(fig)
    keep = res["ok"] & ~res["escaped"] & ~res["chaotic"]
    _log(f"trace {tag}: {time.perf_counter() - t0:.1f}s, {int(keep.sum())}/{len(keep)} regular, "
         f"drift {res['drift']:.1e}, iota {res['iota'][keep].min():.5f}..{res['iota'][keep].max():.5f}")
    return dict(seed_r=np.asarray(res["seeds"][:, 0]).tolist(),
                iota=np.asarray(res["iota"]).tolist(),
                iota_err=np.asarray(res["iota_err"]).tolist(),
                keep=np.asarray(keep).tolist(), drift=float(res["drift"]))


def run_rung(cli):
    import h5py
    import jax.numpy as jnp
    import numpy as np

    import mrx
    # Batch the per-quadrature-point map/projection evaluation to cap peak GPU
    # memory: the coefficient-window gather is materialised over the whole quad
    # grid at once by default (MAP_BATCH_SIZE_INNER = 0), which OOMs the biggest
    # p=4 rungs (a ~10 GiB window at (33,64,32)). A positive batch chunks it.
    _mbs = os.environ.get("MRX_MAP_BATCH_SIZE_INNER")
    if _mbs:
        mrx.MAP_BATCH_SIZE_INNER = int(_mbs)
    from mrx.geometry import build_sequence, geometry_nfp
    from mrx.gvec import load_clebsch
    from mrx.initial_conditions import (clebsch_potential_form, divergence_norm,
                                        potential_two_form)
    from mrx.nullspace import compute_nullspaces, estimate_spectral_gap, harmonic_rayleigh
    from mrx.relaxation import compute_force
    from mrx.vmec import read_wout

    ns = tuple(int(v) for v in cli.ns.split(","))
    p = cli.p
    tag = f"rung_{ns[0]}x{ns[1]}x{ns[2]}_p{p}"
    out = os.path.join(cli.out, tag)
    os.makedirs(out, exist_ok=True)
    print(f"[env] mrx from {mrx.__file__}  precision {mrx.DTYPE}", flush=True)
    res = dict(geometry=os.path.abspath(cli.geometry), ns=list(ns), p=p,
               h=1.0 / (ns[0] - p), n_elements=[ns[0] - p, ns[1], ns[2]],
               precision=str(mrx.DTYPE), grid=[int(v) for v in cli.grid.split(",")])

    # --- geometry, operators, harmonic form -------------------------------
    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, p, cli.maxiter, tol=cli.tol)
    t1 = time.perf_counter()
    ops = seq.set_operators(compute_nullspaces(seq, ops))
    t2 = time.perf_counter()
    h = seq.nullspace(2, True)[0]
    h = h / seq.l2_norm(h, 2)
    rq = harmonic_rayleigh(seq, h, 2, True, ops)
    lam1, sweeps = estimate_spectral_gap(seq, ops, 2, True, maxiter=5)
    ratio = rq / lam1
    # |div h| / |h| in L2 -- the d h = 0 half of harmonic (the deleted
    # nullspace.exact_derivative_residual, inlined).
    div_h = float(seq.l2_norm(seq.apply_strong_div(h, True, True), 3, dirichlet=True)
                  / seq.l2_norm(h, 2, dirichlet=True))
    res.update(n2=int(seq.n(2, True)), tol=float(seq.tol), t_build=t1 - t0, t_nullspace=t2 - t1,
               harmonic=dict(rayleigh=rq, lambda_1=float(lam1), gap_sweeps=int(sweeps),
                             ratio=ratio, div_over_norm=div_h,
                             gate_1e_10=bool(ratio <= 1e-10)))
    _log(f"{tag}: n2={seq.n(2, True)} tol={seq.tol:.1e} build {t1 - t0:.1f}s nullspace {t2 - t1:.1f}s; "
         f"harmonic rq {rq:.3e} lambda_1 {lam1:.3e} ratio {ratio:.1e} "
         f"({'PASS' if ratio <= 1e-10 else 'FAIL'} <= 1e-10), |div h|/|h| {div_h:.1e}")

    # --- the dual harmonic field: the k=1 free (no-BC) harmonic 1-form -------
    # On the solid torus b1 = 1, so nullspace(1, False) is 1-dimensional and
    # Poincare dual to nullspace(2, True): the SAME vacuum toroidal field in a
    # different form degree. Its own gate mirrors h2's, with the d h1 = 0 half
    # being |curl h1| (k=1 -> k=2, free) instead of |div h2|.
    h1 = seq.nullspace(1, False)[0]
    h1 = h1 / seq.l2_norm(h1, 1, dirichlet=False)
    rq1 = harmonic_rayleigh(seq, h1, 1, False, ops)
    lam1_1, sweeps1 = estimate_spectral_gap(seq, ops, 1, False, maxiter=5)
    ratio1 = rq1 / lam1_1
    curl_h1 = float(seq.l2_norm(seq.apply_strong_curl(h1, False, False), 2, dirichlet=False)
                    / seq.l2_norm(h1, 1, dirichlet=False))
    res["harmonic1"] = dict(rayleigh=rq1, lambda_1=float(lam1_1), gap_sweeps=int(sweeps1),
                            ratio=ratio1, curl_over_norm=curl_h1, n1=int(seq.n(1, False)),
                            gate_1e_10=bool(ratio1 <= 1e-10))
    _log(f"{tag} h1: n1={seq.n(1, False)} harmonic rq {rq1:.3e} lambda_1 {lam1_1:.3e} "
         f"ratio {ratio1:.1e} ({'PASS' if ratio1 <= 1e-10 else 'FAIL'} <= 1e-10), "
         f"|curl h1|/|h1| {curl_h1:.1e}")

    # --- the wout field in V_2^h --------------------------------------------
    t3 = time.perf_counter()
    cb = load_clebsch(cli.geometry)
    Bw_hat, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
    div_w = divergence_norm(seq, Bw_hat)
    Bw = Bw_hat * norm                                           # Tesla, one field period
    t4 = time.perf_counter()
    res.update(t_ic=t4 - t3, B_norm=norm, div_B_w=div_w, wall_discarded=wall)
    _log(f"B_w: ||B||_M {norm:.6e} T, ||div B|| {div_w:.2e}, wall-normal discarded {wall:.2e}, {t4 - t3:.1f}s")

    # --- representation independence: h1 (1-form) vs h2 (2-form) as the SAME
    # physical vector field --------------------------------------------------
    # Push both to lab-frame (3,)-vectors at the quadrature points (covariant
    # rule for h1, Piola for h2 and B_w; metric factors DF, J explicit), fit
    # one scale by physical L2 (measure w*J), and compare. If they agree to
    # O(tol) in the bulk the discrete vacuum solution is representation-free.
    from mrx.geometry import map_jacobian_at
    t_rep = time.perf_counter()
    DF_q = np.asarray(map_jacobian_at(seq.map, seq.quad.x))          # (Nq,3,3)
    DFinv_q = np.linalg.inv(DF_q)
    Jq = np.asarray(seq.jacobian_j)                                  # (Nq,)
    wq = np.asarray(seq.quad.w)
    wJ_q = wq * Jq
    rho_q = np.asarray(seq.quad.x)[:, 0]
    h1_ref = np.asarray(seq.evaluate_at_quadrature(h1, 1, False))     # covariant comps
    h2_ref = np.asarray(seq.evaluate_at_quadrature(h, 2, True))       # 2-form comps
    Bw_ref = np.asarray(seq.evaluate_at_quadrature(Bw_hat, 2, True))
    v1 = np.einsum("qki,qk->qi", DFinv_q, h1_ref)                     # (DF^-T) h1_hat
    v2 = np.einsum("qik,qk->qi", DF_q, h2_ref) / Jq[:, None]          # DF h2_hat / J
    vBw = np.einsum("qik,qk->qi", DF_q, Bw_ref) / Jq[:, None]

    def _ip(a, b, mask=None):
        w = wJ_q if mask is None else wJ_q * mask
        return float(np.sum(w * np.sum(a * b, axis=1)))

    def _rep(a, b, mask=None):
        s = _ip(a, b, mask) / _ip(b, b, mask)                        # fit a ~ s b
        resid = float(np.sqrt(_ip(a - s * b, a - s * b, mask) / _ip(a, a, mask)))
        cos = _ip(a, b, mask) / float(np.sqrt(_ip(a, a, mask) * _ip(b, b, mask)))
        return s, resid, cos

    bulk_q = (rho_q >= RHO_BULK).astype(float)
    s12, resid12, cos12 = _rep(v1, v2)
    s12_b, resid12_b, cos12_b = _rep(v1, v2, bulk_q)
    _, _, cosBw1 = _rep(vBw, v1)
    _, _, cosBw1_b = _rep(vBw, v1, bulk_q)
    res["rep_independence"] = dict(
        scale_h1_over_h2=s12, resid_h1_vs_h2=resid12, cos_h1_h2=cos12,
        resid_h1_vs_h2_bulk=resid12_b, cos_h1_h2_bulk=cos12_b,
        cos_Bw_h1=cosBw1, cos_Bw_h1_bulk=cosBw1_b, rho_bulk=RHO_BULK,
        t_rep=time.perf_counter() - t_rep)
    _log(f"rep-indep h1 vs h2 (phys vectors, quad pts): resid {resid12:.3e} "
         f"(bulk {resid12_b:.3e}), M-cos {cos12:+.9f} (bulk {cos12_b:+.9f}); "
         f"cos(B_w, h1) {cosBw1:+.9f} (bulk {cosBw1_b:+.9f})")

    # --- same-space comparison ----------------------------------------------
    Mh = seq.apply_mass_matrix(h, 2, True)
    c = float(Bw @ Mh) / float(h @ Mh)
    D = float(seq.l2_norm(Bw - c * h, 2)) / norm
    cos_theta = c / norm                                         # <B_w, h> / (|B_w| |h|)
    D_from_angle = float(np.sqrt(max(0.0, 1.0 - cos_theta ** 2)))
    Phi_w, Phi_h = toroidal_flux(seq, Bw), toroidal_flux(seq, h)
    c_flux = Phi_w / Phi_h
    D_flux = float(seq.l2_norm(Bw - c_flux * h, 2)) / norm
    st = read_wout(cli.geometry)
    phi_edge = float(st["profiles"]["phi"][-1]) * 2.0 * np.pi   # Wb (the file's phi)
    res.update(c=c, D=D, D_from_angle=D_from_angle, c_flux=c_flux, D_flux=D_flux,
               c_flux_over_c_minus_1=c_flux / c - 1.0, Phi_w=Phi_w, Phi_h=Phi_h,
               phi_edge_file=phi_edge, Phi_w_over_phi_edge_minus_1=Phi_w / phi_edge - 1.0)
    _log(f"scale c {c:.6e} (c_flux {c_flux:.6e}, c_flux/c - 1 = {c_flux / c - 1:+.3e}); "
         f"D = ||B_w - c h||/||B_w|| = {D:.4e} (sin theta {D_from_angle:.4e}); D_flux {D_flux:.4e}; "
         f"Phi_w {Phi_w:.6e} vs file {phi_edge:.6e} ({Phi_w / phi_edge - 1:+.1e})")

    # --- forces at unit M-norm -------------------------------------------------
    t5 = time.perf_counter()
    F_w, _, J_w, _, _ = compute_force(Bw_hat, seq)
    F_h, _, J_h, _, _ = compute_force(h, seq)
    nF_w, nF_h = float(seq.l2_norm(F_w, 2)), float(seq.l2_norm(F_h, 2))
    nJ_w, nJ_h = float(seq.l2_norm(J_w, 1)), float(seq.l2_norm(J_h, 1))
    res.update(F_w=nF_w, F_h=nF_h, J_w=nJ_w, J_h=nJ_h, t_force=time.perf_counter() - t5)
    _log(f"||F||_M at ||B||_M = 1: B_w {nF_w:.4e}, h {nF_h:.4e}; ||J||_M: B_w {nJ_w:.4e}, h {nJ_h:.4e}")

    # --- common grid -----------------------------------------------------------
    t6 = time.perf_counter()
    pts, w, axes = common_grid(res["grid"])
    B_w_xyz, X, J = pushforward_on_grid(seq, Bw, pts)
    B_h_xyz, _, _ = pushforward_on_grid(seq, c * h, pts)
    # h1 as a lab-frame vector on the same grid, calibrated to B_w by physical
    # L2 (its scale and sign are free), so E_h1 across rungs mirrors E_h.
    B_h1_raw, _, _ = pushforward_on_grid(seq, h1, pts, k=1, dirichlet=False)
    wJg = w * J
    c1 = (float(np.sum(wJg * np.sum(B_w_xyz * B_h1_raw, 1)))
          / float(np.sum(wJg * np.sum(B_h1_raw ** 2, 1))))
    B_h1_xyz = c1 * B_h1_raw
    vol = float(np.sum(w * J))
    nBw = np.sqrt(np.sum(w * J * np.sum(B_w_xyz ** 2, 1)))
    D_grid = float(np.sqrt(np.sum(w * J * np.sum((B_w_xyz - B_h_xyz) ** 2, 1))) / nBw)
    D_grid_h1 = float(np.sqrt(np.sum(w * J * np.sum((B_w_xyz - B_h1_xyz) ** 2, 1))) / nBw)
    res["c1_grid"] = c1
    res["D_grid_h1"] = D_grid_h1
    ax_pts = np.stack([np.full(16, 0.01), (np.arange(16) + 0.5) / 16, np.zeros(16)], 1)
    B_ax, _, _ = pushforward_on_grid(seq, Bw, ax_pts)
    B_axis = float(np.mean(np.linalg.norm(B_ax, axis=1)))
    res.update(volume_grid=vol, B_norm_grid=float(nBw), D_grid=D_grid, B_axis_w=B_axis,
               B_norm_grid_over_M=float(nBw) / norm, t_grid=time.perf_counter() - t6)
    _log(f"grid {res['grid']}: volume {vol:.6e}, ||B_w|| {nBw:.6e} (M-norm {norm:.6e}), "
         f"D_grid {D_grid:.4e}, |B| near axis {B_axis:.4f} T, {time.perf_counter() - t6:.1f}s")
    _log(f"grid: D_grid(h1) {D_grid_h1:.4e} (c1 {c1:.6e})")
    np.savez_compressed(os.path.join(out, "fields.npz"), B_w_dof=np.asarray(Bw), h_dof=np.asarray(h),
                        h1_dof=np.asarray(h1), c=c, c1=c1, B_w=B_w_xyz, B_h=B_h_xyz, B_h1=B_h1_xyz,
                        x=X, J=J, w=w, pts=pts, **axes)

    # --- the relaxation run's fields at this rung ---------------------------
    if cli.h5:
        with h5py.File(cli.h5, "r") as f:
            B_ic, B_fin = jnp.asarray(f["B_ic"][()]), jnp.asarray(f["B_final"][()])
        if B_ic.shape[0] != seq.n(2, True):
            raise ValueError(f"{cli.h5}: {B_ic.shape[0]} DoFs, this rung has {seq.n(2, True)}")
        d_ic = float(seq.l2_norm(B_ic - Bw_hat, 2))
        F_fin, _, _, _, _ = compute_force(B_fin, seq)
        c_fin = float(B_fin @ Mh) / float(h @ Mh)
        D_fin = float(seq.l2_norm(B_fin - c_fin * h, 2)) / float(seq.l2_norm(B_fin, 2))
        res["h5"] = dict(path=os.path.abspath(cli.h5), ic_minus_B_w=d_ic,
                         final_D=D_fin, final_F=float(seq.l2_norm(F_fin, 2)),
                         final_norm=float(seq.l2_norm(B_fin, 2)))
        _log(f"h5: ||B_ic - B_w_hat||_M {d_ic:.2e}; B_final: D {D_fin:.4e}, ||F||_M {res['h5']['final_F']:.4e}")

    # --- iota by tracing --------------------------------------------------------
    if cli.trace:
        from scipy.interpolate import interp1d
        nfp = geometry_nfp(cli.geometry)
        iotaf = interp1d(st["profiles"]["rho"], st["profiles"]["iota"], kind="cubic")
        tr = {}
        for name, dof in (("B_w", Bw_hat), ("h", h)):
            t = trace_iota(seq, dof, nfp, cli.trace_seeds, cli.trace_periods, f"{tag} {name}")
            r = np.asarray(t["seed_r"])
            keep = np.asarray(t["keep"])
            d = np.abs(np.asarray(t["iota"]) - iotaf(np.clip(r, 0, 1)))[keep]
            t["max_abs_diota_vs_iotaf"] = float(d.max()) if d.size else float("nan")
            t["rms_diota_vs_iotaf"] = float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")
            tr[name] = t
            _log(f"iota {name} vs iotaf: max {t['max_abs_diota_vs_iotaf']:.2e} rms {t['rms_diota_vs_iotaf']:.2e}")
        res["trace"] = tr

    res["t_total"] = time.perf_counter() - t0
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}/result.json and fields.npz  ({res['t_total']:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Re-evaluate one rung's stored DoFs on a new common grid
# ---------------------------------------------------------------------------

def regrid_rung(cli):
    """Re-push a rung's stored 2-form/1-form DoFs onto ``--grid`` and rewrite
    ``fields.npz`` in place, without recomputing the harmonic forms.

    The cross-resolution ``E`` comparisons sample every rung on the fixed
    common grid, so that grid must OUT-resolve the finest rung or the fine end
    is under-sampled. Raising ``--grid`` then means re-evaluating every rung on
    it. Only the geometry (map + bases) is rebuilt here -- the expensive Hodge
    solve is skipped; the DoFs (``B_w_dof``, ``h_dof``, ``h1_dof`` and the
    scales ``c``, ``c1``) are read back from the rung's own ``fields.npz``.
    """
    import time as _t
    import numpy as np

    import mrx
    _mbs = os.environ.get("MRX_MAP_BATCH_SIZE_INNER")
    if _mbs:
        mrx.MAP_BATCH_SIZE_INNER = int(_mbs)
    from mrx.geometry import build_sequence

    d = cli.regrid
    with open(os.path.join(d, "result.json")) as f:
        res = json.load(f)
    ns, p = tuple(res["ns"]), res["p"]
    npz = dict(np.load(os.path.join(d, "fields.npz")))
    c, c1 = float(npz["c"]), float(npz["c1"])
    t0 = _t.perf_counter()
    seq, _ = build_sequence(res["geometry"], ns, p, cli.maxiter, tol=cli.tol)
    grid = [int(v) for v in cli.grid.split(",")]
    pts, w, axes = common_grid(grid)
    B_w_xyz, X, J = pushforward_on_grid(seq, npz["B_w_dof"], pts, k=2, dirichlet=True)
    B_h_xyz, _, _ = pushforward_on_grid(seq, c * npz["h_dof"], pts, k=2, dirichlet=True)
    B_h1_xyz, _, _ = pushforward_on_grid(seq, c1 * npz["h1_dof"], pts, k=1, dirichlet=False)
    wJ = w * J
    nBw = float(np.sqrt(np.sum(wJ * np.sum(B_w_xyz ** 2, 1))))
    D_grid = float(np.sqrt(np.sum(wJ * np.sum((B_w_xyz - B_h_xyz) ** 2, 1))) / nBw)
    D_grid_h1 = float(np.sqrt(np.sum(wJ * np.sum((B_w_xyz - B_h1_xyz) ** 2, 1))) / nBw)
    ax_pts = np.stack([np.full(16, 0.01), (np.arange(16) + 0.5) / 16, np.zeros(16)], 1)
    B_ax, _, _ = pushforward_on_grid(seq, npz["B_w_dof"], ax_pts, k=2, dirichlet=True)
    res.update(grid=grid, D_grid=D_grid, D_grid_h1=D_grid_h1,
               B_axis_w=float(np.mean(np.linalg.norm(B_ax, axis=1))),
               volume_grid=float(np.sum(wJ)), B_norm_grid=nBw,
               regridded_from=res.get("grid"))
    np.savez_compressed(os.path.join(d, "fields.npz"),
                        B_w_dof=npz["B_w_dof"], h_dof=npz["h_dof"], h1_dof=npz["h1_dof"],
                        c=c, c1=c1, B_w=B_w_xyz, B_h=B_h_xyz, B_h1=B_h1_xyz,
                        x=X, J=J, w=w, pts=pts, **axes)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"regridded {os.path.basename(d)} to {grid}: D_grid {D_grid:.4e} "
          f"(D {res['D']:.4e}), {len(pts)} pts, {_t.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Merge and plot
# ---------------------------------------------------------------------------

#: Logical radius separating the axis region from the bulk in the merge.
RHO_BULK = 0.1


def _rate(e0, h0, e1, h1):
    import numpy as np
    return float(np.log(e0 / e1) / np.log(h0 / h1))


def _slope(hs, es):
    """Least-squares slope of log e against log h."""
    import numpy as np
    if len(hs) < 2:
        return float("nan")
    return float(np.polyfit(np.log(hs), np.log(es), 1)[0])


def plot(cli):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = cli.plot
    rungs = []
    for path in sorted(glob.glob(os.path.join(root, "rung_*", "result.json"))):
        with open(path) as f:
            r = json.load(f)
        r["dir"] = os.path.dirname(path)
        rungs.append(r)
    if not rungs:
        sys.exit(f"no rung_*/result.json under {root}")
    grids = {tuple(r["grid"]) for r in rungs}
    if len(grids) != 1:
        sys.exit(f"rungs were evaluated on different grids: {grids}")

    # Reference: the finest p=3 rung (largest radial element count); at p=3
    # every rung's map is the production one, so E measures MRX's own rate.
    p_main = 3
    main = sorted((r for r in rungs if r["p"] == p_main), key=lambda r: r["n_elements"][0])
    ref = main[-1] if main else sorted(rungs, key=lambda r: r["n_elements"][0])[-1]
    fr = np.load(os.path.join(ref["dir"], "fields.npz"))
    wJ = fr["w"] * fr["J"]
    nB = np.sqrt(np.sum(wJ * np.sum(fr["B_w"] ** 2, 1)))
    nH = np.sqrt(np.sum(wJ * np.sum(fr["B_h"] ** 2, 1)))
    nH1 = np.sqrt(np.sum(wJ * np.sum(fr["B_h1"] ** 2, 1)))
    Lref = np.max(np.linalg.norm(fr["x"], axis=1))

    rows = []
    for r in rungs:
        fz = np.load(os.path.join(r["dir"], "fields.npz"))
        is_ref = r["dir"] == ref["dir"]
        E_h = float(np.sqrt(np.sum(wJ * np.sum((fz["B_h"] - fr["B_h"]) ** 2, 1))) / nH)
        E_w = float(np.sqrt(np.sum(wJ * np.sum((fz["B_w"] - fr["B_w"]) ** 2, 1))) / nB)
        E_h1 = float(np.sqrt(np.sum(wJ * np.sum((fz["B_h1"] - fr["B_h1"]) ** 2, 1))) / nH1)
        # bulk / axis split: the residual of a wout field concentrates at the
        # axis (the half-mesh lambda has no data inside its first node), so
        # the bulk numbers are reported separately.
        wJz = fz["w"] * fz["J"]
        bulk = fr["pts"][:, 0] >= RHO_BULK
        dBw = np.sum((fz["B_w"] - fz["B_h"]) ** 2, 1)
        nBw2 = np.sum(fz["B_w"] ** 2, 1)
        D_bulk = float(np.sqrt(np.sum(wJz[bulk] * dBw[bulk]) / np.sum(wJz[bulk] * nBw2[bulk])))
        D_axis = float(np.sqrt(np.sum(wJz[~bulk] * dBw[~bulk]) / np.sum(wJz[~bulk] * nBw2[~bulk])))
        dH = np.sum((fz["B_h"] - fr["B_h"]) ** 2, 1)
        nH2 = np.sum(fr["B_h"] ** 2, 1)
        E_h_bulk = float(np.sqrt(np.sum(wJ[bulk] * dH[bulk]) / np.sum(wJ[bulk] * nH2[bulk])))
        dx = np.linalg.norm(fz["x"] - fr["x"], axis=1)
        E_map = float(np.sqrt(np.sum(wJ * dx ** 2) / np.sum(wJ)) / Lref)
        E_map_max = float(dx.max() / Lref)
        rows.append(dict(tag=os.path.basename(r["dir"]), dir=r["dir"], ns=r["ns"], p=r["p"], h=r["h"],
                         n2=r["n2"], harmonic_ratio=r["harmonic"]["ratio"],
                         gate=r["harmonic"]["gate_1e_10"], D=r["D"], D_grid=r["D_grid"],
                         c=r["c"], c_flux=r["c_flux"], c_flux_over_c_minus_1=r["c_flux_over_c_minus_1"],
                         F_w=r["F_w"], F_h=r["F_h"], J_w=r["J_w"], div_B_w=r["div_B_w"],
                         E_h=E_h, E_w=E_w, E_h1=E_h1, E_map=E_map, E_map_max=E_map_max, is_reference=is_ref,
                         harmonic1_ratio=r.get("harmonic1", {}).get("ratio"),
                         curl_h1=r.get("harmonic1", {}).get("curl_over_norm"),
                         D_grid_h1=r.get("D_grid_h1"),
                         resid_h1_vs_h2=r.get("rep_independence", {}).get("resid_h1_vs_h2"),
                         resid_h1_vs_h2_bulk=r.get("rep_independence", {}).get("resid_h1_vs_h2_bulk"),
                         cos_h1_h2=r.get("rep_independence", {}).get("cos_h1_h2"),
                         cos_Bw_h1=r.get("rep_independence", {}).get("cos_Bw_h1"),
                         D_bulk=D_bulk, D_axis=D_axis, E_h_bulk=E_h_bulk, rho_bulk=RHO_BULK,
                         B_axis_w=r["B_axis_w"], t_total=r["t_total"],
                         iota_w_vs_iotaf=r.get("trace", {}).get("B_w", {}).get("max_abs_diota_vs_iotaf"),
                         iota_h_vs_iotaf=r.get("trace", {}).get("h", {}).get("max_abs_diota_vs_iotaf"),
                         h5=r.get("h5")))

    # rates along the p=3 ladder
    ladder = sorted((x for x in rows if x["p"] == p_main), key=lambda x: -x["h"])
    for i, x in enumerate(ladder):
        x["rate_D"] = _rate(ladder[i - 1]["D"], ladder[i - 1]["h"], x["D"], x["h"]) if i else None
        x["rate_D_bulk"] = _rate(ladder[i - 1]["D_bulk"], ladder[i - 1]["h"], x["D_bulk"], x["h"]) if i else None
        x["rate_F_w"] = _rate(ladder[i - 1]["F_w"], ladder[i - 1]["h"], x["F_w"], x["h"]) if i else None
        x["rate_E_h"] = (_rate(ladder[i - 1]["E_h"], ladder[i - 1]["h"], x["E_h"], x["h"])
                         if i and not x["is_reference"] else None)
        x["rate_E_w"] = (_rate(ladder[i - 1]["E_w"], ladder[i - 1]["h"], x["E_w"], x["h"])
                         if i and not x["is_reference"] else None)
        x["rate_E_h1"] = (_rate(ladder[i - 1]["E_h1"], ladder[i - 1]["h"], x["E_h1"], x["h"])
                          if i and not x["is_reference"] else None)
    below = [x for x in ladder if not x["is_reference"]]
    slopes = dict(
        D_all=_slope([x["h"] for x in ladder], [x["D"] for x in ladder]),
        D_bulk_all=_slope([x["h"] for x in ladder], [x["D_bulk"] for x in ladder]),
        F_w_all=_slope([x["h"] for x in ladder], [x["F_w"] for x in ladder]),
        E_h_below_ref=_slope([x["h"] for x in below], [x["E_h"] for x in below]),
        E_w_below_ref=_slope([x["h"] for x in below], [x["E_w"] for x in below]),
        E_h1_below_ref=_slope([x["h"] for x in below], [x["E_h1"] for x in below]),
        resid_h1_vs_h2_below=_slope([x["h"] for x in below],
                                    [x["resid_h1_vs_h2"] for x in below])
        if below and all(x["resid_h1_vs_h2"] for x in below) else float("nan"),
        E_map_below_ref=_slope([x["h"] for x in below], [x["E_map"] for x in below]),
        E_h_below_ref_excluding_last=_slope([x["h"] for x in below[:-1]], [x["E_h"] for x in below[:-1]]),
    )
    # The full p x resolution grid: one D ladder per p, its own LS slope. The
    # angular cells are fixed to the p=3 ladder's value per n_elements column,
    # so degree is the only thing that changes at a given h.
    ps_all = sorted({x["p"] for x in rows})
    grid_by_p = {}
    for pv in ps_all:
        lad = sorted((x for x in rows if x["p"] == pv), key=lambda x: -x["h"])
        for i, x in enumerate(lad):
            x["rate_D_p"] = _rate(lad[i - 1]["D"], lad[i - 1]["h"], x["D"], x["h"]) if i else None
        grid_by_p[pv] = lad
    slopes["D_by_p"] = {pv: _slope([x["h"] for x in grid_by_p[pv]], [x["D"] for x in grid_by_p[pv]])
                        for pv in ps_all}
    # The same-space D of every p >= 2 bottoms at the reconstructed VMEC field's
    # own distance from the harmonic field of its boundary (~8e-5); the full LS
    # slope is dragged toward 2 by that plateau, so it hides the O(h^p) rate and
    # the higher p (which reaches the floor at a coarser mesh) even scores lower.
    # The pre-floor slope, over the rungs a decade above the finest D, is the
    # rate that steepens with p.
    # --- elbow detection: the same-space D of every p bottoms at the shared
    # ~8e-5 physics floor. Fit each p's convergence slope over ONLY the
    # pre-elbow rungs (D above ELBOW_FACTOR x the floor) and record the fitted
    # n_elements window and where the elbow (first on-floor rung) sits.
    D_floor = min(x["D"] for x in rows)
    ELBOW_FACTOR = 2.0
    slopes["D_floor"] = float(D_floor)
    slopes["elbow_factor"] = ELBOW_FACTOR
    slopes["D_by_p_preelbow"] = {}
    slopes["preelbow_window"] = {}
    slopes["elbow_nel"] = {}
    for pv in ps_all:
        lad = grid_by_p[pv]
        pre = [x for x in lad if x["D"] > ELBOW_FACTOR * D_floor] or lad[:2]
        slopes["D_by_p_preelbow"][pv] = _slope([x["h"] for x in pre], [x["D"] for x in pre])
        nel = sorted(x["ns"][0] - pv for x in pre)
        slopes["preelbow_window"][pv] = [nel[0], nel[-1]]
        onfloor = [x["ns"][0] - pv for x in lad if x["D"] <= ELBOW_FACTOR * D_floor]
        slopes["elbow_nel"][pv] = min(onfloor) if onfloor else None
    slopes["D_by_p_prefloor"] = slopes["D_by_p_preelbow"]     # alias for the note

    # --- per-p self-convergence: each p's field against its OWN finest rung
    # (now n_el up to 45), so the fine end is no longer pinned to one global
    # reference. E on the common 48x96x48 grid, that p's finest as reference.
    slopes["selfconv_Ew_by_p"] = {}
    slopes["selfconv_Eh_by_p"] = {}
    slopes["selfconv_ref_nel"] = {}
    for pv in ps_all:
        lad = grid_by_p[pv]
        rp = lad[-1]                                          # finest rung of this p
        fp = np.load(os.path.join(rp["dir"], "fields.npz"))
        wJp = fp["w"] * fp["J"]
        nBp = np.sqrt(np.sum(wJp * np.sum(fp["B_w"] ** 2, 1)))
        nHp = np.sqrt(np.sum(wJp * np.sum(fp["B_h"] ** 2, 1)))
        below_p = []
        for x in lad:
            if x["dir"] == rp["dir"]:
                x["E_w_ownp"], x["E_h_ownp"], x["is_ref_ownp"] = 0.0, 0.0, True
                continue
            fx = np.load(os.path.join(x["dir"], "fields.npz"))
            x["E_w_ownp"] = float(np.sqrt(np.sum(wJp * np.sum((fx["B_w"] - fp["B_w"]) ** 2, 1))) / nBp)
            x["E_h_ownp"] = float(np.sqrt(np.sum(wJp * np.sum((fx["B_h"] - fp["B_h"]) ** 2, 1))) / nHp)
            x["is_ref_ownp"] = False
            below_p.append(x)
        for i, x in enumerate(sorted(below_p, key=lambda z: -z["h"])):
            prev = sorted(below_p, key=lambda z: -z["h"])[i - 1] if i else None
            x["rate_E_w_ownp"] = _rate(prev["E_w_ownp"], prev["h"], x["E_w_ownp"], x["h"]) if prev else None
        hh_p = [x["h"] for x in below_p]
        slopes["selfconv_Ew_by_p"][pv] = _slope(hh_p, [x["E_w_ownp"] for x in below_p]) if len(below_p) > 1 else float("nan")
        slopes["selfconv_Eh_by_p"][pv] = _slope(hh_p, [x["E_h_ownp"] for x in below_p]) if len(below_p) > 1 else float("nan")
        slopes["selfconv_ref_nel"][pv] = rp["ns"][0] - pv
    summary = dict(reference=os.path.basename(ref["dir"]), grid=ref["grid"], rows=rows, slopes=slopes)
    with open(os.path.join(root, "convergence.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # --- table to stdout -------------------------------------------------------
    print(f"reference rung: {summary['reference']}; grid {ref['grid']}")
    hdr = (f"{'rung':>18} {'h':>7} {'ratio':>8} {'D':>10} {'rate':>6} {'D_bulk':>10} {'rate':>6} {'D_axis':>10} "
           f"{'F_w':>10} {'rate':>6} {'E_h':>10} {'rate':>6} {'E_w':>10} {'E_map':>10} {'c_fl/c-1':>9}")
    print(hdr)
    for x in sorted(rows, key=lambda x: (x["p"], -x["h"])):
        def fmt(v, w=6):
            return f"{v:>{w}.2f}" if isinstance(v, float) else f"{'--':>{w}}"
        print(f"{x['tag']:>18} {x['h']:7.4f} {x['harmonic_ratio']:8.1e} {x['D']:10.3e} {fmt(x.get('rate_D'))} "
              f"{x['D_bulk']:10.3e} {fmt(x.get('rate_D_bulk'))} {x['D_axis']:10.3e} "
              f"{x['F_w']:10.3e} {fmt(x.get('rate_F_w'))} {x['E_h']:10.3e} {fmt(x.get('rate_E_h'))} "
              f"{x['E_w']:10.3e} {x['E_map']:10.3e} {x['c_flux_over_c_minus_1']:+9.1e}")
    print("slopes: " + "  ".join(f"{k} {v:.2f}" for k, v in slopes.items()
                                  if isinstance(v, float) and k != "D_floor"))
    print("D LS slope per p (full grid): "
          + "  ".join(f"p{pv} {slopes['D_by_p'][pv]:.2f}" for pv in ps_all)
          + f"  [floor {slopes['D_floor']:.1e}]")
    print(f"D pre-elbow slope per p (D > {slopes['elbow_factor']:.0f}x floor {slopes['D_floor']:.1e}):")
    for pv in ps_all:
        w = slopes["preelbow_window"][pv]
        print(f"  p{pv}: slope {slopes['D_by_p_preelbow'][pv]:.2f}  fit n_el {w[0]}..{w[1]}  "
              f"elbow at n_el {slopes['elbow_nel'][pv]}")
    print("per-p self-convergence (each p vs its own finest):")
    for pv in ps_all:
        print(f"  p{pv}: ref n_el {slopes['selfconv_ref_nel'][pv]}  "
              f"E_w slope {slopes['selfconv_Ew_by_p'][pv]:.2f}  "
              f"E_h slope {slopes['selfconv_Eh_by_p'][pv]:.2f}")

    # --- dual harmonic field (h1) table ---------------------------------------
    print("\ndual harmonic field: k=1 free 1-form h1 vs k=2 dbc 2-form h2 (same vacuum B)")
    h1hdr = (f"{'rung':>18} {'h':>7} {'ratio1':>9} {'|curl h1|':>10} {'E_h1':>10} {'rate':>6} "
             f"{'resid h1|h2':>11} {'bulk':>10} {'M-cos':>13} {'cos(Bw,h1)':>13}")
    print(h1hdr)
    for x in sorted(rows, key=lambda x: (x["p"], -x["h"])):
        def fmt(v, w=6):
            return f"{v:>{w}.2f}" if isinstance(v, float) else f"{'--':>{w}}"
        print(f"{x['tag']:>18} {x['h']:7.4f} {x['harmonic1_ratio']:9.1e} {x['curl_h1']:10.1e} "
              f"{x['E_h1']:10.3e} {fmt(x.get('rate_E_h1'))} {x['resid_h1_vs_h2']:11.3e} "
              f"{x['resid_h1_vs_h2_bulk']:10.3e} {x['cos_h1_h2']:+13.10f} {x['cos_Bw_h1']:+13.10f}")
    print(f"slopes(h1): E_h1 {slopes['E_h1_below_ref']:.2f}  "
          f"resid_h1_vs_h2 {slopes['resid_h1_vs_h2_below']:.2f}")

    # --- figure ----------------------------------------------------------------
    C = dict(D="#0072B2", F="#E69F00", Eh="#009E73", Ew="#CC79A7", map="#56B4E9", psweep="#D55E00")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    hs = np.array([x["h"] for x in ladder])
    ax = axes[0]
    ax.loglog(hs, [x["D"] for x in ladder], "o-", color=C["D"], lw=1.5, ms=6,
              label=r"$D=\|B_w - c\,h\|_M/\|B_w\|_M$")
    ax.loglog(hs, [x["D_bulk"] for x in ladder], "o--", color=C["D"], lw=1.2, ms=5, mfc="none",
              label=rf"$D$ on $\rho \geq {RHO_BULK}$ (bulk)")
    ax.loglog(hs, [x["F_w"] for x in ladder], "s-", color=C["F"], lw=1.5, ms=6,
              label=r"$\|F\|_M(B_w)$ at $\|B\|_M=1$")
    ax.set_title("same space: VMEC field vs harmonic form (p = 3)")
    ax = axes[1]
    hb = np.array([x["h"] for x in below])
    ax.loglog(hb, [x["E_h"] for x in below], "o-", color=C["Eh"], lw=1.5, ms=6,
              label=r"$E_h$: $h$ vs finest")
    ax.loglog(hb, [x["E_w"] for x in below], "^-", color=C["Ew"], lw=1.5, ms=6,
              label=r"$E_w$: $B_w$ vs finest")
    ax.loglog(hb, [x["E_map"] for x in below], "d-", color=C["map"], lw=1.2, ms=5,
              label=r"map: rms $|F_h-F_{ref}|/L$")
    ax.set_title(f"vs the finest rung ({summary['reference'].replace('rung_', '')})")
    # the other p ladders of D, faint, for context (the dedicated grid is
    # convergence_grid.png)
    PCOL = {1: "#D55E00", 2: "#E69F00", 3: "#0072B2", 4: "#009E73"}
    for pv in sorted({x["p"] for x in rows} - {p_main}):
        lad = grid_by_p[pv]
        axes[0].loglog([x["h"] for x in lad], [x["D"] for x in lad], "o:", color=PCOL.get(pv, "0.5"),
                       lw=1.0, ms=4, mfc="none", alpha=0.9, label=f"$D$, p = {pv}")
    for ax, (s_name, s_val), anchor in ((axes[0], ("D bulk", slopes["D_bulk_all"]), ladder[0]["D"]),
                                        (axes[1], ("E_h", slopes["E_h_below_ref"]), below[0]["E_h"] if below else 1.0)):
        hh = np.array([hs.min(), hs.max()])
        for q, ls in ((p_main, "--"), (p_main + 1, ":")):
            ax.loglog(hh, anchor * 1.6 * (hh / hs.max()) ** q, ls, color="0.6", lw=1, label=rf"$h^{q}$")
        ax.text(0.03, 0.97, f"LS slope {s_name}: {s_val:.2f}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel(r"$h = 1/(n_r - p)$")
        ax.set_ylabel("relative error")
        ax.grid(True, which="both", color="0.9", lw=0.6)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"QA vacuum: {os.path.basename(rungs[0]['geometry'])}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "convergence.png"), dpi=150)
    plt.close(fig)

    # --- full p x resolution grid: D vs h, one ladder per p -------------------
    figg, axg = plt.subplots(figsize=(6.8, 5.6))
    hmin = min(x["h"] for x in rows)
    for pv in ps_all:
        lad = grid_by_p[pv]
        axg.loglog([x["h"] for x in lad], [x["D"] for x in lad], "o-", color=PCOL.get(pv, "0.3"),
                   lw=1.7, ms=5,
                   label=f"p = {pv}: slope {slopes['D_by_p_preelbow'][pv]:.2f} "
                         f"(n_el {slopes['preelbow_window'][pv][0]}–{slopes['preelbow_window'][pv][1]})")
        # the fitted pre-elbow slope line, drawn over its own window only
        pre = [x for x in lad if x["D"] > slopes["elbow_factor"] * D_floor] or lad[:2]
        hpre = np.array([min(x["h"] for x in pre), max(x["h"] for x in pre)])
        aD = max(pre, key=lambda z: z["h"])["D"]
        aH = max(x["h"] for x in pre)
        s = slopes["D_by_p_preelbow"][pv]
        axg.loglog(hpre, aD * (hpre / aH) ** s, "-", color=PCOL.get(pv, "0.6"), lw=3.2, alpha=0.25)
    axg.axhline(slopes["D_floor"], ls="-", color="0.5", lw=1.0, alpha=0.7)
    axg.text(0.02, slopes["D_floor"] * 1.15, rf"physics floor $\approx${slopes['D_floor']:.1e}",
             transform=axg.get_yaxis_transform(), fontsize=8, color="0.4", va="bottom")
    # h^p guide slopes for reference (p=1 is degree-limited, no guide)
    for pv in ps_all:
        if pv == 1:
            continue
        lad = grid_by_p[pv]
        ah = max(x["h"] for x in lad)
        aD = next(x["D"] for x in lad if x["h"] == ah)
        gh = np.array([hmin, ah])
        axg.loglog(gh, aD * (gh / ah) ** pv, ":", color=PCOL.get(pv, "0.6"), lw=1.0)
    axg.set_xlabel(r"$h = 1/(n_r - p) = 1/n_{\mathrm{elements}}$")
    axg.set_ylabel(r"$D = \|B_w - c\,h\|_M / \|B_w\|_M$")
    axg.set_title("QA vacuum: $D$ vs $h$, full $p \\times$ resolution grid\n"
                  "(thick = pre-elbow slope fit; dotted = $h^p$ guide)", fontsize=10)
    axg.grid(True, which="both", color="0.9", lw=0.6)
    axg.legend(fontsize=8.5, loc="lower right")
    figg.tight_layout()
    figg.savefig(os.path.join(root, "convergence_grid.png"), dpi=150)
    plt.close(figg)

    # --- per-p self-convergence: each p vs its own finest rung ----------------
    figs, axs = plt.subplots(figsize=(6.8, 5.6))
    hmax_all = max(x["h"] for x in rows)
    for pv in ps_all:
        below_p = sorted((x for x in grid_by_p[pv] if not x.get("is_ref_ownp", False)),
                         key=lambda z: -z["h"])
        if not below_p:
            continue
        axs.loglog([x["h"] for x in below_p], [x["E_w_ownp"] for x in below_p], "o-",
                   color=PCOL.get(pv, "0.3"), lw=1.7, ms=5,
                   label=f"p = {pv}: slope {slopes['selfconv_Ew_by_p'][pv]:.2f} "
                         f"(ref n_el {slopes['selfconv_ref_nel'][pv]})")
    for pv in ps_all:
        below_p = [x for x in grid_by_p[pv] if not x.get("is_ref_ownp", False)]
        if len(below_p) < 1:
            continue
        aD = max(below_p, key=lambda z: z["h"])["E_w_ownp"]
        gh = np.array([hmin, hmax_all])
        axs.loglog(gh, aD * (gh / hmax_all) ** pv, ":", color=PCOL.get(pv, "0.6"), lw=1.0)
    axs.set_xlabel(r"$h = 1/(n_r - p) = 1/n_{\mathrm{elements}}$")
    axs.set_ylabel(r"$E_w = \|B_w(h) - B_w(h_{\min})\| / \|B_w(h_{\min})\|$")
    axs.set_title("QA vacuum: per-$p$ self-convergence of $B_w$\n"
                  "(each $p$ vs its own finest rung; dotted = $h^p$ guide)", fontsize=10)
    axs.grid(True, which="both", color="0.9", lw=0.6)
    axs.legend(fontsize=8.5, loc="lower right")
    figs.tight_layout()
    figs.savefig(os.path.join(root, "convergence_selfconv.png"), dpi=150)
    plt.close(figs)

    # --- dual harmonic field figure -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.loglog(hb, [x["E_h1"] for x in below], "o-", color=C["Eh"], lw=1.5, ms=6,
              label=r"$E_{h_1}$: 1-form $h_1$ vs finest")
    ax.loglog(hb, [x["E_h"] for x in below], "^--", color=C["Ew"], lw=1.2, ms=5, mfc="none",
              label=r"$E_h$: 2-form $h_2$ vs finest")
    for pv in sorted({x["p"] for x in rows} - {p_main}):
        lad = grid_by_p[pv]
        ax.loglog([x["h"] for x in lad], [x["E_h1"] for x in lad], "o:", color=PCOL.get(pv, "0.5"),
                  lw=1.0, ms=4, mfc="none", alpha=0.9, label=f"$E_{{h_1}}$, p = {pv}")
    anchor = below[0]["E_h1"] if below else 1.0
    hh = np.array([hs.min(), hs.max()])
    for q, ls in ((p_main, "--"), (p_main + 1, ":")):
        ax.loglog(hh, anchor * 1.6 * (hh / hs.max()) ** q, ls, color="0.6", lw=1, label=rf"$h^{q}$")
    ax.text(0.03, 0.97, f"LS slope $E_{{h_1}}$: {slopes['E_h1_below_ref']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("dual harmonic field: self-convergence of $h_1$ (p = 3)")
    ax = axes[1]
    hL = np.array([x["h"] for x in ladder])
    ax.loglog(hL, [x["resid_h1_vs_h2"] for x in ladder], "o-", color=C["D"], lw=1.5, ms=6,
              label=r"$\|v_1 - s\,v_2\|/\|v_1\|$ (all $\rho$)")
    ax.loglog(hL, [x["resid_h1_vs_h2_bulk"] for x in ladder], "o--", color=C["D"], lw=1.2, ms=5,
              mfc="none", label=rf"same, bulk $\rho \geq {RHO_BULK}$")
    for pv in sorted({x["p"] for x in rows} - {p_main}):
        lad = grid_by_p[pv]
        ax.loglog([x["h"] for x in lad], [x["resid_h1_vs_h2"] for x in lad], "o:",
                  color=PCOL.get(pv, "0.5"), lw=1.0, ms=4, mfc="none", alpha=0.9,
                  label=f"resid, p = {pv}")
    anchor = ladder[0]["resid_h1_vs_h2_bulk"]
    for q, ls in ((p_main, "--"), (p_main + 1, ":")):
        ax.loglog(hh, anchor * 1.6 * (hh / hs.max()) ** q, ls, color="0.6", lw=1, label=rf"$h^{q}$")
    ax.text(0.03, 0.97, f"LS slope (bulk): {slopes['resid_h1_vs_h2_below']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("representation independence: $h_1$ (1-form) vs $h_2$ (2-form)")
    for ax in axes:
        ax.set_xlabel(r"$h = 1/(n_r - p)$")
        ax.set_ylabel("relative error")
        ax.grid(True, which="both", color="0.9", lw=0.6)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"QA vacuum, dual harmonic field: {os.path.basename(rungs[0]['geometry'])}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "convergence_h1.png"), dpi=150)
    plt.close(fig)

    # --- residual on the zeta = 0 section of the reference -----------------
    nr, nt, nz = ref["grid"]
    d = np.linalg.norm(fr["B_w"] - fr["B_h"], axis=1).reshape(nr, nt, nz)[:, :, 0]
    bmax = np.linalg.norm(fr["B_w"], axis=1).max()
    x = fr["x"].reshape(nr, nt, nz, 3)[:, :, 0]
    R, Z = np.hypot(x[..., 0], x[..., 1]), x[..., 2]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    sc = ax.scatter(R.ravel(), Z.ravel(), c=(d / bmax).ravel(), s=12, cmap="Blues", lw=0)
    fig.colorbar(sc, ax=ax, label=r"$|B_w - c\,h| / \max|B_w|$")
    ax.set_aspect("equal")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(f"{summary['reference']}: residual at zeta = 0, max {d.max() / bmax:.2e}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "residual_zeta0.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {root}/convergence.json, convergence.png, convergence_grid.png, convergence_selfconv.png, convergence_h1.png, residual_zeta0.png", flush=True)


if __name__ == "__main__":
    cli = parse_args()
    os.environ["MRX_DTYPE"] = cli.precision
    if cli.plot:
        plot(cli)
    elif cli.regrid:
        regrid_rung(cli)
    elif cli.ns:
        run_rung(cli)
    else:
        sys.exit("give --ns (one rung), --regrid DIR, or --plot DIR")
